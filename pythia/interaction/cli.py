"""Caller-owned interaction loop with a scrollback POSIX terminal shell.

Run with ``python3 -m pythia.interaction.cli``. ``--headless`` runs one explicit
task without a terminal. Line-shell input and active-effect interruption are
deliberately deferred.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import contextvars
import functools
from collections import deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextlib import nullcontext
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import field
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import uuid
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Union

from ._account_http import default_account_opener
from ._cli_editor import Editor
from ._cli_editor import safe_text
from ._cli_terminal import PosixTerminal
from ._debug_trace import DebugTrace
from ._prompt import load_prompt
from .codex_auth import CodexAuthUnavailable
from .compaction import CompactionError
from .compaction import CompactionResult
from .compaction import NothingToCompact
from .compaction import create_default_compactor
from .context import InteractionContext
from .default_environment import DefaultEnvironment
from .display import DisplayItem
from .display import render_interaction_items
from .environment import Environment
from .items import CompactionMetadata
from .items import ContextPrefix
from .items import Init
from .items import Instructions
from .items import Tools
from .items import InteractionItem
from .items import Message
from .items import ModelSampleBoundary
from .items import OpaqueCompaction
from .items import Reasoning
from .items import ToolCall
from .items import ToolResult
from .items import SampleMetadata
from .items import TurnSummary
from .items import UserInteractionBoundary
from .items import UserToolCall
from .items import UserToolResult
from .media import AttachmentError
from .media import parse_user_prompt
from .media import split_leading_references
from .loop import Interrupt
from .loop import Preemption
from .loop import Steer
from .loop import TurnHost
from .loop import Urgency
from .loop import command_urgency
from .loop import escalated_stop
from .loop import run_turn
from .loop.tail import describe_tail
from .loop.tail import turn_tail
from .model import Model
from .model import ModelAuthenticationError
from .model import ModelError
from .model import ModelTimeoutError
from .model import SampleParams
from .model import close_model, retire_model
from .model_config import DEFAULT_SAVE_PATH
from .model_config import _boolean_argument
from .model_config import build_model
from .model_config import build_parser
from .model_config import initial_model_name
from .model_config import resolve_save_path
from .model_config import supports_account_services
from .model_config import frontend_catalog, prepare_namespace, render_model_catalog
from ._model_binding_debug import debug_model_binding_path
from ._model_binding_debug import save_debug_model_bindings
from .runtime_config import InteractionConfig
from .save import InteractionSaveWriter
from .save import resume_interaction_save
from .user import UserInteraction
from .user_tools import UserToolIntent
from .user_tools import create_user_environment
from .user_tools import parse_user_tool


FRAME_INTERVAL = 1 / 128
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_MAX_PENDING_QUERIES = 8
# The status line while closing, by the stop's level.
_CLOSING_STATUS = {
    Urgency.QUEUED: "closing — stopping before the next sample; Ctrl-C or /exit! stops sooner",
    Urgency.IMMEDIATE: "closing — waiting for current operation; Ctrl-C or /exit!! cancels it",
    Urgency.PREEMPT: "closing — cancelling current operation",
}


@dataclass(frozen=True, eq=False)
class _RetryIntent:
    """Identity ticket for one live sampling failure, never model input."""


@dataclass(frozen=True, eq=False)
class _ContinueIntent:
    """/continue: sample the saved context without a user message; never model input."""


@dataclass(frozen=True, eq=False)
class _Submission:
    """Submitted text and its user message, read when Enter accepted the text."""

    text: str
    message: Message


def _plain_message(text: str) -> Message:
    return Message("user", text)


def _reads_no_files(text: str) -> bool:
    return False


@dataclass
class _UIState:
    editor: Editor = field(default_factory=Editor)
    pending: deque[Union[_Submission, UserToolIntent, _RetryIntent, _ContinueIntent]] = field(
        default_factory=deque)
    retry: Optional[_RetryIntent] = None
    headless: bool = False
    displays: deque[DisplayItem] = field(default_factory=deque)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    closing: bool = False
    ready: bool = False
    persistence_failed: bool = False
    phase: str = "starting"
    phase_started: float = field(default_factory=time.monotonic)
    exit_code: int = 0
    auth_required: bool = False
    auth_notice: str = "Model authentication needed; use /login."
    bound_account_id: Optional[str] = None
    login_cancel: threading.Event = field(default_factory=threading.Event)
    transient: queue.Queue[tuple[str, str]] = field(default_factory=lambda: queue.Queue(maxsize=8))
    active_user_call: Optional[str] = None
    trace: Optional[DebugTrace] = None
    active_model: object = field(default=None, repr=False)
    # True while a model turn runs that started with nothing queued: plain text
    # submitted then steers it. Otherwise it queues behind the older input.
    turn_active: bool = False
    # Submitted text -> its user message, raising AttachmentError; the outer
    # loop sets it from the launch options. reads_files tells whether that
    # reads files (leading @ references with experimental media).
    read_query: Callable[[str], Message] = field(default=_plain_message, repr=False)
    reads_files: Callable[[str], bool] = field(default=_reads_no_files, repr=False)
    # A draft whose files are being read, on a helper thread: it stays in the
    # editor, and Enter queues nothing else until it is queued or rejected.
    reading: Optional[str] = None
    # Flushed steers (/steer!, /steer!!) and stops reach the running turn
    # here, from the event loop; the context thread checks it.
    preemption: Preemption = field(default_factory=Preemption, repr=False)
    # The stop's level once one is requested (closing is then true): QUEUED
    # for /exit, IMMEDIATE for /exit! or a first Ctrl-C, PREEMPT for /exit!!
    # or a later one. It is the context's level, which steers flushed before
    # the stop raise to IMMEDIATE (see Preemption.stop). It only rises.
    stop_level: Optional[Urgency] = None
    # The context thread: this context's model, tools, and saves run here, so
    # the event loop thread (terminal, redraw, exit keys) never blocks.
    context_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="interaction-context"),
        repr=False,
    )

    def set_phase(self, phase: str) -> None:
        self.phase, self.phase_started = phase, time.monotonic()

    def notice(self, text: str) -> None:
        text = text.rstrip("\r\n")
        if self.headless:
            print(safe_text(f"[cli] {text}"), file=sys.stderr, flush=True)
        else:
            self.displays.append(DisplayItem(f"[cli] {text}"))

    def request_stop(self, level: Urgency = Urgency.IMMEDIATE) -> None:
        """Exit at ``level``; a lower level than the current one does nothing.

        Every level drops queued input, cancels a login wait, and starts no
        new query. QUEUED (/exit): the turn finishes its tool batch and ends
        before its next sample. IMMEDIATE (/exit!, the first Ctrl-C): the
        sample or tool call in flight finishes and is saved, for every model,
        and nothing new starts. PREEMPT (/exit!!, a later Ctrl-C): that
        operation is also cancelled where possible (a Claude relay sample, a
        command's wait), and the model is retired, which also cancels model
        work outside a turn, such as /compact. Outcomes are still saved.
        """
        level = Urgency(level)
        if self.stop_level is not None and level <= self.stop_level:
            return
        self.closing = True
        self.preemption.stop(level)
        # The context's level: steers flushed before the stop raise it.
        self.stop_level = self.preemption.stop_level
        self.retry = None
        self.pending.clear()
        self.login_cancel.set()
        self.changed.set()
        if self.stop_level >= Urgency.PREEMPT and callable(
                getattr(self.active_model, "retire", None)):
            model = self.active_model
            def retire():
                try:
                    retire_model(model)
                except Exception:
                    self.notice("Model retirement failed; final cleanup will run.")
            threading.Thread(target=retire, name="interaction-model-retire", daemon=True).start()

    def escalate_stop(self) -> None:
        """Ctrl-C, Ctrl-D, or SIGINT: /exit! the first time, then /exit!!."""
        self.request_stop(escalated_stop(self.stop_level))

    def _stop_hint(self) -> str:
        """What a stronger exit would do now, as a sentence (or nothing)."""
        if self.stop_level is None or self.stop_level < Urgency.IMMEDIATE:
            return " Ctrl-C or /exit! stops sooner; /exit!! also cancels the current operation."
        if self.stop_level < Urgency.PREEMPT:
            return " Ctrl-C or /exit!! cancels the current operation."
        return ""

    def handle_key(self, key: str, data: str) -> None:
        if key in {"c-c", "c-d"}:
            # The first press is /exit!; a later one also cancels (/exit!!).
            self.escalate_stop()
        elif self.closing or self.ready:
            if key != "c-m":
                # While closing too, so that a stronger /exit can be typed.
                self.editor = self.editor.edit(key, data)
                return
            text = self.editor.text
            head = text.split(maxsplit=1)[0] if text.strip() else ""
            level = command_urgency(head, "/exit", "/quit")
            if level is not None:
                if self.stop_level is not None and level <= self.stop_level:
                    self.notice(f"Already stopping.{self._stop_hint()}")
                else:
                    self.editor = Editor()
                    self.request_stop(level)
                return
            if self.closing:
                if text.strip():
                    self.notice(f"Stopping; input is not accepted.{self._stop_hint()}")
                return
            if not text.strip():
                self.editor = Editor()
                return
            if self.persistence_failed:
                self.notice("Checkpoint failed; no further work will run. Use /quit.")
                return
            if self.reading is not None:
                self.notice("A submission is still pending. Draft preserved.")
                return
            intent = text
            steer = command_urgency(head, "/steer")
            if head == "/retry":
                error = None
                if text.strip() != "/retry" or "\n" in text or "\r" in text:
                    error = "Usage: /retry (no arguments; single line)."
                elif self.phase not in {"idle", "failed", "auth needed"}:
                    error = "Cannot retry while work is in progress."
                elif self.retry is None:
                    error = "No retryable sampling failure in this session."
                elif any(isinstance(item, _RetryIntent) for item in self.pending):
                    error = "A retry is already queued."
                elif any(isinstance(item, _ContinueIntent) for item in self.pending):
                    error = "A continuation is already queued."
                if error is not None:
                    self.notice(error)
                    self.editor = Editor()
                    return
                intent = self.retry
            elif head == "/continue":
                # Like /retry: no arguments, idle, and not auth-gated (the
                # owner reloads credentials). The owner checks the saved log.
                error = None
                if text.strip() != "/continue" or "\n" in text or "\r" in text:
                    error = "Usage: /continue (no arguments; single line)."
                elif self.phase not in {"idle", "failed", "auth needed"}:
                    error = "Cannot continue while work is in progress."
                elif any(isinstance(item, _ContinueIntent) for item in self.pending):
                    error = "A continuation is already queued."
                elif any(isinstance(item, _RetryIntent) for item in self.pending):
                    error = "A retry is already queued."
                if error is not None:
                    self.notice(error)
                    self.editor = Editor()
                    return
                intent = _ContinueIntent()
            elif steer is not None:
                # Queue the text (if any) like plain text, then flush every
                # queued steer at the command's level: /steer waits for the
                # interrupt point, as plain text does (the notice only reports
                # the queued steers); /steer! starts nothing new before them;
                # /steer!! also cancels the sample or command wait in flight,
                # where possible.
                parts = text.split(maxsplit=1)
                if len(parts) == 1:
                    self.editor = Editor()
                    self._flush_steers(steer)
                elif self.auth_required:
                    self.notice(f"{self.auth_notice} Draft was not submitted.")
                else:
                    self._submit_text(parts[1], draft=text, flush=steer)
                return
            elif head.startswith("/"):
                try:
                    intent = parse_user_tool(text)
                except ValueError as exc:
                    # Never echo an arbitrary slash argument (possibly a secret).
                    self.notice(str(exc))
                    self.editor = Editor()
                    return
            elif self.auth_required:
                self.notice(f"{self.auth_notice} Draft was not submitted.")
                return
            else:
                self._submit_text(text)
                return
            if len(self.pending) >= _MAX_PENDING_QUERIES:
                self.notice("Query queue is full; the draft has not been submitted.")
            else:
                self.pending.append(intent)
                self.editor = Editor()
                self.changed.set()

    def _submit_text(self, text: str, *, draft: Optional[str] = None,
                     flush: Optional[Urgency] = None) -> None:
        """Read a draft's attachments now, then queue its message.

        A bad reference rejects only this draft: it stays in the editor with
        the error, and nothing is queued. A draft with files to read is read
        on a helper thread, as auto submits tasks; meanwhile it stays in the
        editor, and Enter queues nothing else, so the queue keeps its order.
        ``draft`` is the editor text (``/steer <text>``), and ``flush`` the
        urgency to flush the queued steers with once the text is queued.
        """
        draft = text if draft is None else draft
        if len(self.pending) >= _MAX_PENDING_QUERIES:
            self.notice("Query queue is full; the draft has not been submitted.")
            return
        if self.reads_files(text):
            self.reading = draft
            future = asyncio.get_running_loop().run_in_executor(None, self.read_query, text)
            future.add_done_callback(functools.partial(self._read_done, text, draft, flush))
            return
        try:
            message = self.read_query(text)
        except AttachmentError as exc:
            self.notice(str(exc))
            return
        self._queue_submission(_Submission(text, message), draft, flush)

    def _read_done(self, text: str, draft: str, flush: Optional[Urgency], future) -> None:
        self.reading = None
        if future.cancelled() or self.closing:
            return
        error = future.exception()
        if isinstance(error, AttachmentError):
            self.notice(str(error))
        elif error is not None:
            # Never reflect an unexpected error's text (it may quote content).
            self.notice(f"Reading attachments failed ({type(error).__name__}); "
                        "the draft has not been submitted.")
        else:
            self._queue_submission(_Submission(text, future.result()), draft, flush)

    def _queue_submission(self, submission: _Submission, draft: str,
                          flush: Optional[Urgency] = None) -> None:
        if len(self.pending) >= _MAX_PENDING_QUERIES:
            self.notice("Query queue is full; the draft has not been submitted.")
            return
        self.pending.append(submission)
        if self.editor.text == draft:
            self.editor = Editor()
        self.changed.set()
        if flush is not None:
            steered = _steer_count(self) == len(self.pending)
            if not steered:
                # Older input is queued ahead of it, or no turn is running.
                self.notice("No running turn can take a steer now; queued as the next query.")
            # A flush speeds up the steers queued before it; at QUEUED there is
            # nothing to report if the text itself could not steer.
            if _steer_count(self) and (steered or flush > Urgency.QUEUED):
                self._flush_steers(flush)

    def _flush_steers(self, level: Urgency) -> None:
        """Have the running turn take the queued steers at ``level`` (see
        Preemption), and say when they arrive: the urgency only rises, so that
        may be sooner than ``level`` (and at QUEUED nothing changes)."""
        count = _steer_count(self)
        if not count:
            self.notice("No queued steers to flush.")
            return
        self.preemption.flush(level)
        level = max(Urgency(level), self.preemption.level)
        steers, them = (f"{count} steers", "them") if count > 1 else ("1 steer", "it")
        if level < Urgency.IMMEDIATE:
            self.notice(f"{steers} queued for the next sample, after the current tool batch; "
                        f"/steer! or /steer!! delivers {them} sooner.")
        elif level < Urgency.PREEMPT:
            self.notice(f"Flushing {steers}: delivered once the current sample or tool call "
                        "finishes.")
        else:
            self.notice(f"Flushing {steers}: delivered now; the current sample or command wait "
                        "is cancelled where possible.")


def _build_model(args: argparse.Namespace, trace: Optional[DebugTrace]) -> Model:
    """Build with backend-aware tracing; account user tools retain their opener."""
    if trace is None:
        return build_model(args)
    return build_model(args, trace=trace)


@contextmanager
def _traced_operation(state: _UIState, op: str, **tags):
    """Tag the enclosed worker call's HTTP exchanges; report trace failures."""
    if state.trace is None:
        yield
        return
    try:
        with state.trace.operation(op, **tags):
            yield
    finally:
        warning = state.trace.take_warning()
        if warning is not None:
            with suppress(OSError, ValueError):
                state.notice(warning)


async def _checkpoint(
    context: InteractionContext, state: _UIState, writer: InteractionSaveWriter,
) -> None:
    state.set_phase("saving")
    try:
        await _on_context_thread(state, writer.save, context.copy())
    except Exception:
        state.persistence_failed = True
        raise


async def _append(
    context: InteractionContext,
    items: Iterable[InteractionItem],
    state: _UIState,
    writer: InteractionSaveWriter,
) -> None:
    context.extend(items)
    await _checkpoint(context, state, writer)


async def _fail_pending_tools(
    context: InteractionContext, state: _UIState, writer: InteractionSaveWriter,
    *, reason: str,
) -> None:
    """Close missing outcomes without claiming that their effects did not happen."""
    for call in context.pending_tool_calls():
        if state.closing:
            return
        result = ToolResult(
            call_id=call.call_id,
            success=False,
            output=(
                f"Result unavailable after {reason}. "
                "This call cannot be resumed and was not rerun. "
                "It may already have produced side effects."
            ),
        )
        await _append(context, (result,), state, writer)
        state.displays.extend(
            render_interaction_items((result,), source_calls=(call,))
        )


async def _fail_pending_user_tools(
    context: InteractionContext, state: _UIState, writer: InteractionSaveWriter,
) -> None:
    for call in context.pending_user_tool_calls():
        if state.closing:
            return
        if call.call.name == "compact":
            output = (
                "Compaction outcome unavailable after interruption. The command "
                "was not rerun; no durable compaction checkpoint was installed."
            )
        elif call.call.name == "config":
            output = (
                "Config outcome unavailable after interruption. The command "
                "was not rerun; in-memory configuration was initialized from "
                "the current launch arguments."
            )
        else:
            output = (
                "User-tool outcome unavailable after interruption. The command "
                "was not rerun; credential side effects may already have occurred."
            )
        result = UserToolResult(ToolResult(
            call.call.call_id,
            output,
            success=False,
        ))
        await _append(context, (result,), state, writer)
        state.displays.extend(render_interaction_items((result,), source_user_calls=(call,)))


def _has_provider_history(context: InteractionContext) -> bool:
    return any(
        (
            isinstance(i, (SampleMetadata, CompactionMetadata))
            and (
                i.provider_turn_id
                or i.provider_turn_state
                or i.provider_session_id
            )
        )
        or (isinstance(i, Reasoning) and i.encrypted_content)
        or (isinstance(i, OpaqueCompaction) and i.protocol == "responses")
        for i in (*context.items, *context.model_items())
    )


def _mark_auth_required(
    state: _UIState,
    exc: ModelAuthenticationError,
) -> None:
    state.auth_required = True
    if exc.failure is not None and exc.failure.auth_source == "runtime":
        state.auth_notice = "Authenticate as the broker account through the sandbox wrapper, then /retry; /login is not Claude login."
    elif exc.failure is not None and exc.failure.auth_source == "environment":
        state.auth_notice = (
            "Environment credential rejected; update it and restart the process."
        )
    elif exc.failure is not None and exc.failure.auth_source == "static":
        state.auth_notice = (
            "Configured static credential rejected; restart with updated credentials."
        )
    elif exc.failure is not None and exc.failure.auth_source == "none":
        # /login cannot help an anonymous non-Codex endpoint.
        state.auth_notice = (
            "Endpoint rejected anonymous access; restart with "
            "--endpoint-auth env:NAME or supplied."
        )
    else:
        state.auth_notice = "Model authentication needed; use /login."


def _compaction_failure_output(exc: BaseException) -> str:
    if isinstance(exc, ModelAuthenticationError):
        if exc.failure is not None:
            return f"Compaction failed: {exc.failure.message}"
        return "Model authentication needed; use /login."
    if isinstance(exc, CompactionError):
        detail = str(exc).replace("\r", " ").replace("\n", " ").strip()
        if detail:
            return f"Compaction failed: {detail[:512]}"
    if isinstance(exc, ModelError) and exc.failure is not None:
        return f"Compaction failed: {exc.failure.message}"
    return "Compaction failed; provider and response details were withheld."


def _compaction_success_output(result: CompactionResult) -> str:
    checkpoint = result.items[0]
    assert isinstance(checkpoint, ContextPrefix)
    # A pi prefix can carry an earlier Responses checkpoint verbatim.
    remote = result.protocol != "pi" and any(
        isinstance(item, OpaqueCompaction)
        for item in checkpoint.prefix_items
    )
    mode = "a remote opaque checkpoint" if remote else "a pi summary checkpoint"
    return f"Context compacted using {mode}."


def _compact_focus(intent: UserToolIntent) -> Optional[str]:
    """The optional focus text of ``/compact [focus]``."""
    try:
        arguments = json.loads(intent.arguments_json)
    except ValueError:
        return None
    focus = arguments.get("instructions") if isinstance(arguments, dict) else None
    return focus if isinstance(focus, str) else None


async def _compact_user_tool(
    intent: UserToolIntent,
    model: Optional[Model],
    context: InteractionContext,
    model_environment: Environment,
    state: _UIState,
    writer: InteractionSaveWriter,
    config: InteractionConfig,
) -> Optional[Model]:
    # A pending UserToolCall deliberately makes a context non-sampleable. Take
    # the immutable compaction source first, then durably record authorization
    # for the effect before starting it.
    source_context = context.copy()
    snapshot = config.snapshot()
    call = UserToolCall(
        ToolCall(
            intent.name,
            "user_" + uuid.uuid4().hex,
            intent.arguments_json,
        )
    )
    await _append(context, (call,), state, writer)
    state.displays.extend(render_interaction_items((call,)))
    state.active_user_call = call.call.call_id

    if state.closing:
        result_item = UserToolResult(
            ToolResult(
                call.call.call_id,
                "Compaction cancelled before execution.",
                success=False,
            )
        )
        await _append(context, (result_item,), state, writer)
        state.displays.extend(
            render_interaction_items(
                (result_item,),
                source_user_calls=(call,),
            )
        )
        state.active_user_call = None
        return model

    state.set_phase("compacting")
    try:
        if model is None:
            result_item = UserToolResult(
                ToolResult(
                    call.call.call_id,
                    state.auth_notice,
                    success=False,
                )
            )
            contribution: tuple[InteractionItem, ...] = (result_item,)
        else:
            try:
                compactor = create_default_compactor(
                    model, snapshot.compaction_settings(),
                )
                with _traced_operation(state, "compact", context_revision=len(source_context)):
                    compaction = await _on_context_thread(state, 
                        compactor.compact,
                        source_context,
                        tools=model_environment.tool_specs,
                        sample_params=snapshot.sample_params(),
                        instructions=_compact_focus(intent),
                    )
                if not isinstance(compaction, CompactionResult):
                    raise TypeError(
                        "compactor must return CompactionResult, got "
                        f"{type(compaction).__name__}"
                    )
            except NothingToCompact as exc:
                result_item = UserToolResult(
                    ToolResult(
                        call.call.call_id,
                        f"Nothing to compact: {exc}.",
                        success=False,
                    )
                )
                contribution = (result_item,)
            except Exception as exc:
                if isinstance(exc, ModelAuthenticationError):
                    _mark_auth_required(state, exc)
                    model = None
                result_item = UserToolResult(
                    ToolResult(
                        call.call.call_id,
                        _compaction_failure_output(exc),
                        success=False,
                    )
                )
                contribution = (result_item,)
            else:
                result_item = UserToolResult(
                    ToolResult(
                        call.call.call_id,
                        _compaction_success_output(compaction),
                    )
                )
                # Validate and save these together: a durable success result
                # must never exist without the checkpoint it describes.
                contribution = (
                    result_item,
                    *compaction.context_items(),
                )

        await _append(context, contribution, state, writer)
        state.displays.extend(
            render_interaction_items(
                contribution,
                source_user_calls=(call,),
            )
        )
    finally:
        state.active_user_call = None
    return model


async def _user_tool(
    intent: UserToolIntent, model: Optional[Model], context: InteractionContext,
    state: _UIState, writer: InteractionSaveWriter, args: argparse.Namespace,
    model_environment: Environment, config: InteractionConfig,
) -> Optional[Model]:
    if intent.name == "compact":
        return await _compact_user_tool(
            intent,
            model,
            context,
            model_environment,
            state,
            writer,
            config,
        )
    expected_account = state.bound_account_id
    call = UserToolCall(ToolCall(intent.name, "user_" + uuid.uuid4().hex, intent.arguments_json))
    await _append(context, (call,), state, writer)
    state.displays.extend(render_interaction_items((call,)))
    if state.closing:
        return model
    state.login_cancel.clear()
    state.active_user_call = call.call.call_id

    def notify(text):
        try:
            state.transient.put_nowait((call.call.call_id, text))
        except queue.Full:
            pass

    try:
        environment = create_user_environment(
            args, notify=notify, cancel=state.login_cancel,
            config=config,
            expected_account=expected_account,
            provider_history=_has_provider_history(context),
            **({} if state.trace is None else {
                "opener": state.trace.opener(default_account_opener()),
            }),
        )
        state.set_phase(f"user tool: {intent.name}")
        with _traced_operation(state, intent.name, context_revision=len(context)):
            outcome = await _on_context_thread(state, 
                environment.execute_tool_calls, (call.call,),
            )
        result = UserToolResult(outcome.items[0])
        await _append(context, (result,), state, writer)
        state.displays.extend(render_interaction_items((result,), source_user_calls=(call,)))
    finally:
        state.active_user_call = None
    if intent.name == "login" and result.result.success and not state.closing:
        state.set_phase("loading model")
        candidate = None
        try:
            candidate = await _on_context_thread(state, _build_model, args, state.trace)
            model = candidate
            if (expected_account is not None and
                    getattr(getattr(model, "endpoint", None), "account_id", None) != expected_account):
                raise ValueError("credential account changed during activation")
        except Exception:
            await _on_context_thread(state, close_model, candidate)
            model = None
            state.exit_code = 1
            state.pending.clear()
            state.notice("Credentials were saved, but model activation failed. No model request was started.")
        else:
            state.bound_account_id = getattr(getattr(model, "endpoint", None), "account_id", None)
            state.notice(
                "Model ready with reloaded credentials. Backend authorization "
                "will be verified by the next model request; blocked drafts were not "
                "automatically submitted."
            )
        state.auth_required = model is None
        if model is not None:
            state.auth_notice = "Model authentication needed; use /login."
    return model


async def _on_context_thread(state: _UIState, function, /, *args, **kwargs):
    """Run blocking work on the context thread, like ``asyncio.to_thread``."""
    call = functools.partial(contextvars.copy_context().run, function, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(state.context_executor, call)


def _mark_persistence_failed(state: _UIState) -> None:
    state.persistence_failed = True


def _arm_retry(state: _UIState) -> None:
    state.retry = _RetryIntent()


def _bind_account(state: _UIState, account_id: str) -> None:
    if state.bound_account_id is None:
        state.bound_account_id = account_id


def _query_message(query: str, args, cwd: Path) -> Message:
    """The user message for submitted text; raises AttachmentError."""
    message = parse_user_prompt(
        query,
        cwd=cwd,
        enabled=args.enable_experimental_media,
        enable_workspace=args.enable_workspace,
    )
    if message.has_media and args.model_api not in {"codex", "responses", "chat-completions"}:
        raise AttachmentError(
            f"--endpoint-api {args.model_api} does not support "
            "media prompts; use codex, responses, or chat-completions"
        )
    return message


def _reads_files(text: str, args) -> bool:
    """Whether _query_message reads files for this text (leading @ references)."""
    return bool(args.enable_experimental_media and split_leading_references(text)[0])


def _steer_count(state: _UIState) -> int:
    """Leading text submissions in the queue: a running turn takes them as steers."""
    if not state.turn_active:
        return 0
    count = 0
    for entry in state.pending:
        if not isinstance(entry, _Submission):
            break
        count += 1
    return count


def _take_steers(state: _UIState) -> tuple:
    """The steers a running turn takes at its interrupt point; this also resets
    their urgency (flushed steers are delivered now)."""
    submissions = []
    while state.turn_active and state.pending and isinstance(state.pending[0], _Submission):
        submissions.append(state.pending.popleft())
    state.preemption.take()
    return tuple(submissions)


def _safe_notice(state: _UIState, text: str) -> None:
    with suppress(OSError, ValueError):
        state.notice(text)


class _CliHost(TurnHost):
    """The CLI's side of the shared turn loop.

    Its methods run on the context thread. Every change to terminal state is
    posted to the event loop thread, in order, so the redraw (which copies and
    clears ``state.displays``) never races an update. Posted changes run
    before the turn's outcome reaches ``_drive_interaction``.
    """

    def __init__(self, state: _UIState, writer: InteractionSaveWriter, loop,
                 steering=False) -> None:
        self._state, self._writer, self._loop = state, writer, loop
        self._steering = steering

    def _post(self, function, *args) -> None:
        self._loop.call_soon_threadsafe(function, *args)

    def _on_loop(self, function, *args):
        """Run function on the event loop thread and wait for its result."""
        future = concurrent.futures.Future()

        def run():
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(function(*args))
                except BaseException as exc:
                    future.set_exception(exc)
        self._loop.call_soon_threadsafe(run)
        return future.result()

    def interrupt(self):
        """Right before a sample: text submitted since becomes steers.

        Slash commands and /retry wait for the turn to end, and a steer never
        overtakes one queued before it. Each steer's attachments were read
        when Enter accepted it.
        """
        if not self._steering or self._state.closing:
            return super().interrupt()
        submissions = self._on_loop(_take_steers, self._state)
        if self._state.closing:
            return Interrupt.STOP
        if not submissions:
            return Interrupt.CONTINUE
        return Steer(tuple(submission.message for submission in submissions))

    def append(self, context, items) -> None:
        context.extend(items)
        self.phase("saving")
        try:
            self._writer.save(context.copy())
        except Exception:
            self._post(_mark_persistence_failed, self._state)
            raise

    def show(self, items) -> None:
        self._post(self._state.displays.extend, tuple(items))

    def phase(self, phase: str) -> None:
        self._post(self._state.set_phase, phase)

    def tool_phase(self, call) -> None:
        self.phase(f"tool: {call.name}")

    def should_stop(self) -> bool:
        # A stop at IMMEDIATE or above: start no new sample or tool call.
        level = self._state.preemption.stop_level
        return level is not None and level >= Urgency.IMMEDIATE

    def stop_requested(self) -> bool:
        # A stop at any level: sample no more (/exit lets the batch finish).
        return self._state.closing or self._state.preemption.stop_level is not None

    def should_steer(self) -> bool:
        return self._steering and self._state.preemption.due()

    def preemptible(self, cancel):
        return self._state.preemption.operation(cancel)

    @contextmanager
    def trace(self, op: str, **tags):
        trace = self._state.trace
        if trace is None:
            yield
            return
        try:
            with trace.operation(op, **tags):
                yield
        finally:
            warning = trace.take_warning()
            if warning is not None:
                self.notice(warning)

    def notice(self, text: str) -> None:
        self._post(_safe_notice, self._state, text)

    def after_sample(self, model, sample) -> None:
        account_id = getattr(getattr(model, "endpoint", None), "account_id", None)
        if account_id is not None:
            self._post(_bind_account, self._state, account_id)

    def retryable_failure(self) -> None:
        self._post(_arm_retry, self._state)


async def _turn(context, model, environment, state, writer, config, *, steering=False):
    """One turn of the shared loop, on the context thread.

    With ``steering``, text submitted during the turn reaches it right before
    the next sample.
    """
    # Each explicit attempt consumes the preceding failure's ticket; only a
    # failure the loop reports as retryable arms a new one.
    state.retry = None
    host = _CliHost(state, writer, asyncio.get_running_loop(), steering)
    state.turn_active = not state.pending  # a steer never overtakes older input
    try:
        return await _on_context_thread(
            state, run_turn, context, model, environment, config.snapshot(), host)
    finally:
        state.turn_active = False
        state.preemption.take()  # steers left over become the next query


async def _reload_retry_model(
    context: InteractionContext, state: _UIState, args: argparse.Namespace,
) -> Optional[Model]:
    """Reload credentials without initiating login or switching accounts."""
    state.set_phase("loading model")
    state.auth_required = True
    expected_account = state.bound_account_id
    if (
        expected_account is None
        and supports_account_services(args)
        and _has_provider_history(context)
    ):
        state.notice(
            "Cannot verify the account for saved provider state; start a fresh session."
        )
        return None
    model = None
    try:
        model = await _on_context_thread(state, _build_model, args, state.trace)
        account = getattr(getattr(model, "endpoint", None), "account_id", None)
        if expected_account is not None and account != expected_account:
            state.notice(
                "Credential account changed; no model request was started. "
                "Restore the original account or start a fresh session."
            )
            await _on_context_thread(state, close_model, model)
            return None
    except Exception:
        await _on_context_thread(state, close_model, model)
        # As with /login activation, do not reflect credential/provider details.
        state.notice("Model reload failed; no model request was started. Details withheld.")
        state.notice(state.auth_notice)
        return None
    state.bound_account_id = account
    state.auth_required = False
    state.auth_notice = "Model authentication needed; use /login."
    return model


def _ends_with_completed_manual_compaction(context: InteractionContext) -> bool:
    items = tuple(item for item in context.items if not isinstance(item, Tools))
    end = len(items)
    if end and isinstance(items[end - 1], CompactionMetadata):
        end -= 1
    if end < 3 or not isinstance(items[end - 1], ContextPrefix):
        return False
    result = items[end - 2]
    call = items[end - 3]
    return (
        isinstance(result, UserToolResult)
        and result.result.success
        and isinstance(call, UserToolCall)
        and call.call.name == "compact"
        and result.result.call_id == call.call.call_id
    )


def _resume_notice(context: InteractionContext) -> Optional[str]:
    # Inspect the raw tail, not the model context established by a ContextPrefix.
    # A sample boundary does not record stop_reason or turn completion.
    if _ends_with_completed_manual_compaction(context):
        return None
    for item in reversed(context.items):
        if isinstance(
            item,
            (
                ModelSampleBoundary,
                SampleMetadata,
                CompactionMetadata,
                UserInteractionBoundary,
                UserToolCall,
                UserToolResult,
                Tools,
            ),
        ):
            continue
        if isinstance(item, (TurnSummary, Init)):
            return None
        next_step = ("enter /continue to sample from here, or a query to continue "
                     "with a new message" if turn_tail(context.items).unfinished
                     else "enter a query to continue")
        return (
            f"Note: resumed save ends with {describe_tail(item)}, without recorded turn completion. "
            "The model stop reason is not saved. No model request was started; "
            f"{next_step}."
        )
    return None


async def _drive_interaction(model, environment, state, args, path, config):
    state.active_model = model
    try:
        return await _drive_interaction_body(model, environment, state, args, path, config)
    finally:
        if callable(getattr(state.active_model, "close", None)):
            await _on_context_thread(state, close_model, state.active_model)


async def _drive_interaction_body(
    model: Optional[Model],
    environment: Environment,
    state: _UIState,
    args: argparse.Namespace,
    path: Path,
    config: InteractionConfig,
) -> None:
    attachment_cwd = Path(args.cwd).expanduser().resolve()
    # Enter reads each submission's attachments (see _UIState._submit_text).
    state.read_query = functools.partial(_query_message, args=args, cwd=attachment_cwd)
    state.reads_files = functools.partial(_reads_files, args=args)
    tools_snapshot = Tools(environment.tool_specs)
    existing = args.resume and await _on_context_thread(state, path.exists)
    incomplete_line = None
    if existing:
        context, writer, incomplete_line = await _on_context_thread(
            state, resume_interaction_save, path)
        state.displays.extend(render_interaction_items(context.items))
        if incomplete_line is not None:
            state.notice(incomplete_line.warning(path.name))
        state.notice(
            "Note: command sessions and plan state were not restored. "
            "Old command session IDs are not resumable; use only IDs from this run."
        )
        if any(
            isinstance(item, UserToolCall)
            and item.call.name == "config"
            for item in context.items
        ):
            state.notice(
                "Note: in-memory configuration was reset from the current launch "
                "arguments; saved config commands were not replayed."
            )
    else:
        if args.resume:
            state.notice(
                f"Warning: no existing {path.name} was found; a fresh one was created."
            )
        writer = InteractionSaveWriter(path)
        initial = [Init(model=args.model or initial_model_name(model))]
        if args.instructions is not None:
            initial.append(Instructions(args.instructions))
        initial.append(tools_snapshot)
        context = InteractionContext(initial)
    if state.closing:
        return
    initial_query = args.prompt
    startup = True
    while not state.closing:
        sampling_attempt = False
        try:
            if startup:
                if existing:
                    if incomplete_line is not None:
                        # Repair the tail before any other recovery: this
                        # save truncates the line that failed to parse.
                        await _checkpoint(context, state, writer)
                    await _fail_pending_user_tools(context, state, writer)
                    await _fail_pending_tools(
                        context, state, writer, reason="session restart"
                    )
                else:
                    # Initial-save failures retain the fresh context too.
                    await _checkpoint(context, state, writer)
                if state.closing:
                    return
                if tools_snapshot != context.latest_tools():
                    # Compare the raw log, not its compacted model projection.
                    # Runtime tools are never restored from these snapshots.
                    await _append(context, (tools_snapshot,), state, writer)
                    state.displays.extend(render_interaction_items((tools_snapshot,)))
                elif not existing:
                    state.displays.extend(render_interaction_items((tools_snapshot,)))
                if existing and args.instructions is not None:
                    instructions = Instructions(args.instructions)
                    await _append(context, (instructions,), state, writer)
                    state.displays.extend(render_interaction_items((instructions,)))
                if (getattr(args, "debug_save_model_binding", False)
                        and getattr(args, "model_binding", None) is not None):
                    warning = await _on_context_thread(state, 
                        save_debug_model_bindings,
                        debug_model_binding_path(path),
                        {"main": args.model_binding},
                    )
                    if warning is not None:
                        state.notice(warning)
                query = initial_query
                should_sample = model is not None and (query is not None or (
                    existing and args.instructions is not None
                ))
                if existing and not should_sample:
                    notice = _resume_notice(context)
                    if notice is not None:
                        state.notice(notice)
                if model is None:
                    query = None
                    state.notice(
                        f"{state.auth_notice} No model query was submitted."
                    )
                else:
                    state.editor = Editor()
                state.ready = True
                startup = False
            else:
                await state.changed.wait()
                state.changed.clear()
                if state.closing:
                    return
                if not state.pending:
                    continue
                query = state.pending.popleft()
                if state.pending:
                    state.changed.set()
                if isinstance(query, _RetryIntent):
                    # Recheck on the owner: an earlier queued query may have
                    # superseded the failure since Enter accepted this ticket.
                    if query is not state.retry:
                        state.notice("Retry no longer applies to the current task.")
                        continue
                    if context.pending_tool_calls() or context.pending_user_tool_calls():
                        state.retry = None
                        state.notice("Cannot retry with unresolved tool outcomes.")
                        continue
                    if model is None:
                        model = await _reload_retry_model(context, state, args)
                        state.active_model = model
                        if model is None:
                            state.set_phase("auth needed")
                            continue
                    query = None  # Continue context; do not append a user turn.
                elif isinstance(query, _ContinueIntent):
                    # Continue the saved turn where it stopped, without a user
                    # message (and so without a boundary), if its next step
                    # would be a sample. Unlike /retry, this reads the log, so
                    # it also works after a restart.
                    if context.pending_tool_calls() or context.pending_user_tool_calls():
                        state.notice("Cannot continue with unresolved tool outcomes.")
                        continue
                    refusal = turn_tail(context.items).refusal
                    if refusal is not None:
                        state.notice(f"Nothing to continue: {refusal}. Enter a query instead.")
                        continue
                    if model is None:
                        model = await _reload_retry_model(context, state, args)
                        state.active_model = model
                        if model is None:
                            state.set_phase("auth needed")
                            continue
                    query = None
                else:
                    await _fail_pending_user_tools(context, state, writer)
                    # A new query is not permission to retry old calls whose
                    # side effects may already have happened.
                    await _fail_pending_tools(
                        context, state, writer, reason="an interrupted operation"
                    )
                    if state.closing:
                        return
                    if isinstance(query, UserToolIntent):
                        previous_model = model
                        model = await _user_tool(
                            query,
                            model,
                            context,
                            state,
                            writer,
                            args,
                            environment,
                            config,
                        )
                        if previous_model is not model:
                            await _on_context_thread(state, close_model, previous_model)
                        state.active_model = model
                        state.set_phase("auth needed" if state.auth_required else "idle")
                        continue
                    if model is None:
                        state.editor = Editor(query.text, len(query.text))
                        state.notice(f"{state.auth_notice} Draft was not submitted.")
                        state.set_phase("auth needed")
                        continue
                    state.retry = None
                should_sample = True
            if state.closing:
                return
            if query is not None:
                if isinstance(query, _Submission):
                    message = query.message  # read when Enter accepted it
                else:  # the startup prompt
                    try:
                        message = _query_message(query, args, attachment_cwd)
                    except AttachmentError as exc:
                        state.notice(str(exc))
                        if state.headless:
                            state.exit_code = 1
                            state.set_phase("failed")
                            return
                        state.editor = Editor(query, len(query))
                        state.set_phase("idle")
                        continue
                user = UserInteraction((message,))
                await _append(context, user.context_items(), state, writer)
                state.displays.extend(user.display_items())
            if should_sample:
                sampling_attempt = True
                await _turn(
                    context,
                    model,
                    environment,
                    state,
                    writer,
                    config,
                    steering=True,
                )
            state.set_phase("auth needed" if state.auth_required else "idle")
            if state.headless:
                return
        except Exception as exc:
            state.exit_code = 1
            state.ready = True
            startup = False
            state.set_phase("failed")
            if isinstance(exc, ModelAuthenticationError):
                if state.bound_account_id is None:
                    state.bound_account_id = getattr(
                        getattr(model, "endpoint", None), "account_id", None
                    )
                await _on_context_thread(state, close_model, model)
                model = None
                state.active_model = None
                _mark_auth_required(state, exc)
            state.notice(f"{type(exc).__name__}: {exc}")
            if state.pending:
                state.notice(
                    "Queued queries discarded after failure; submit again explicitly."
                )
            state.pending.clear()
            state.changed.clear()
            if state.persistence_failed:
                state.retry = None
                # Retain the unsaved context here until exit; never replay the effect.
                state.notice(
                    "Checkpoint failed; unsaved state remains in memory. "
                    + ("No further work will run." if state.headless else "Use /quit.")
                )
                if state.headless:
                    return
                while not state.closing:
                    await state.changed.wait()
                    state.changed.clear()
                return
            if state.headless:
                return
            if sampling_attempt and state.retry is not None and not state.closing:
                # Keep all existing diagnostics above, then append guidance.
                # Authentication guidance suggests /login; it is not a gate
                # on /retry, which can reload externally refreshed credentials.
                state.notice(
                    state.auth_notice if isinstance(exc, ModelAuthenticationError)
                    else "Sampling failed. Use /retry to try again."
                )
                timeout = config.get("request_timeout_seconds")
                if isinstance(exc, ModelTimeoutError) and timeout is not None:
                    # Claude Relay has no HTTP request timeout (None), and its
                    # deadlines are launch-only, so it gets no /config advice.
                    state.notice(
                        f"The request timeout is {timeout:g} seconds; to wait "
                        "longer, use /config request_timeout_seconds N before /retry."
                    )


def _runtime_config(environment, args):
    workspace_update = getattr(environment, "set_enable_workspace", None)
    config = InteractionConfig.from_namespace(
        args,
        on_enable_workspace=(
            workspace_update if callable(workspace_update) else None
        ),
    )
    if callable(workspace_update):
        workspace_update(config.get("enable_workspace"))
    return config


def _startup_notices(state, args, path):
    state.notice(f"Save log: {path}")
    if state.trace is not None:
        state.notice(
            f"Debug trace: {state.trace.request_path} and "
            f"{state.trace.response_path}; events: {state.trace.event_path} "
            "(append-only; sensitive payloads, headers, native/MCP data and possibly credentials)."
        )
    if args.enable_default_tools:
        state.notice(
            "Warning: exec_command runs without a sandbox; use a trusted model and workspace."
        )
        if args.model_api == "claude-relay":
            state.notice("Warning: Claude Relay host tools run as the Pythia user, "
                         "outside the Claude sandbox.")
        if not args.enable_workspace:
            state.notice(
                "Warning: workspace path restrictions are disabled; "
                "exec_command workdir and apply_patch paths may resolve "
                "outside --cwd."
            )
    else:
        state.notice("Note: default model tools are disabled; user commands remain available."
                     if not state.headless else "Note: default model tools are disabled.")


async def _run_headless(model, environment, args, path, *, trace=None):
    """One explicit task, quiet context display, and orderly effect draining."""
    if model is None:
        raise ValueError("Headless execution requires an available model.")
    path = Path(path).absolute()
    config = _runtime_config(environment, args)
    state = _UIState(
        headless=True,
        bound_account_id=getattr(getattr(model, "endpoint", None), "account_id", None),
        trace=trace,
    )
    _startup_notices(state, args, path)
    loop = asyncio.get_running_loop()
    interrupted = False
    previous_sigint = None

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        # Like Ctrl-C: the first signal is /exit!, a later one /exit!!.
        loop.call_soon_threadsafe(state.escalate_stop)

    if threading.current_thread() is threading.main_thread():
        # asyncio.run on older Python versions cancels *all* tasks on SIGINT.
        # Request a cooperative stop instead, so the effect owner can save.
        previous_sigint = signal.signal(signal.SIGINT, interrupt)
    worker = asyncio.create_task(_drive_interaction(model, environment, state, args, path, config))
    try:
        while not worker.done():
            # The controller's display queue is transient, not a second log.
            state.displays.clear()
            await asyncio.wait((worker,), timeout=0.05)
        worker.result()
    finally:
        try:
            state.request_stop(Urgency.IMMEDIATE)  # never lowers a stop
            # Cancellation/SIGINT must not close command resources before an
            # in-flight request/tool has finished and checkpointed its outcome.
            await asyncio.shield(worker)
            state.displays.clear()
        finally:
            if previous_sigint is not None:
                signal.signal(signal.SIGINT, previous_sigint)
    return 130 if interrupted else state.exit_code


async def _run(
    model: Optional[Model],
    environment: Environment,
    terminal: PosixTerminal,
    args: argparse.Namespace,
    path: Path = DEFAULT_SAVE_PATH,
    *,
    trace: Optional[DebugTrace] = None,
) -> int:
    path = Path(path).absolute()
    config = _runtime_config(environment, args)
    prompt = args.prompt or ""
    state = _UIState(
        editor=Editor(prompt, len(prompt)), auth_required=model is None,
        bound_account_id=getattr(getattr(model, "endpoint", None), "account_id", None),
        trace=trace,
    )
    state.notice(
        "pythia.interaction — /retry, /continue, /steer [text], /compact [focus], /config, "
        "/config.json, /login, /quota, /exit or /quit. /steer and /exit wait for the current "
        "tool batch; with ! they wait only for the current operation, and with !! they "
        "cancel it. Ctrl-C/Ctrl-D is /exit!, and a second press /exit!!."
    )
    _startup_notices(state, args, path)
    worker = None
    frame = 0
    with terminal:
        try:
            # Draw the pre-filled editor before starting startup I/O/auto-submit.
            terminal.render(state.editor, "starting", tuple(state.displays))
            state.displays.clear()
            worker = asyncio.create_task(
                _drive_interaction(model, environment, state, args, path, config)
            )
            while True:
                for key in terminal.read_keys():
                    state.handle_key(key.key, key.data or "")
                if terminal.closed and state.stop_level != Urgency.PREEMPT:
                    # No one is left to press again: stop and cancel at once.
                    state.request_stop(Urgency.PREEMPT)
                while True:
                    try:
                        call_id, text = state.transient.get_nowait()
                    except queue.Empty:
                        break
                    if call_id == state.active_user_call:
                        state.notice(text)
                busy = state.phase not in {"idle", "failed", "auth needed"}
                status = (state.phase if state.stop_level is None
                          else _CLOSING_STATUS[state.stop_level])
                if busy:
                    status += f" {int(time.monotonic() - state.phase_started)}s"
                steers = _steer_count(state)
                if steers:
                    status += f" | steers={steers}"
                    level = state.preemption.level
                    if level >= Urgency.IMMEDIATE:
                        status += (" (preempting)" if level >= Urgency.PREEMPT
                                   else " (immediate)")
                if len(state.pending) > steers:
                    status += f" | queued={len(state.pending) - steers}"
                prompt = (
                    f"{_SPINNER[(frame // 16) % len(_SPINNER)]}> "
                    if busy else ":> "
                )
                terminal.render(state.editor, status, tuple(state.displays), prompt)
                state.displays.clear()
                if worker.done():
                    worker.result()
                    break
                frame += 1
                await asyncio.sleep(FRAME_INTERVAL)
        finally:
            state.request_stop(Urgency.IMMEDIATE)  # never lowers a stop
            if worker is not None:
                # Do not cancel a thread-backed effect or close its environment early.
                await asyncio.shield(worker)
    return state.exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "Interactive POSIX shell or headless task using Pythia's caller-owned interaction API.",
        allow_prompt_file=True,
    )
    # Unlike the one-shot demo, the CLI continues its save by default;
    # --resume=False starts a new save, replacing the file.
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--headless", nargs="?", const=True, default=False,
        type=_boolean_argument, metavar="{False,True}",
        help=("run one explicit prompt without a TUI, stdin reads, or context display; "
              "save the interaction and exit. Requires --prompt/--prompt-file or "
              "--resume with --instructions and an existing save. A bare flag "
              "means True (default: %(default)s)"),
    )
    parser.add_argument(
        "--enable-default-tools",
        nargs="?",
        const=True,
        default=True,
        type=_boolean_argument,
        metavar="{False,True}",
        help=(
            "enable default model tools (exec_command, write_stdin, apply_patch, "
            "update_plan); user commands remain available when False. "
            "Launch-only: repeat when resuming. A bare flag means True "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--debug-trace",
        action="store_true",
        help=(
            "append HTTP requests/responses and native/MCP events, including "
            "sensitive payloads and credentials, to the --save path with its extension replaced by "
            ".trace.req.jsonl, .trace.res.jsonl and .trace.events.jsonl; "
            "never truncated, even when --resume=False replaces the save; launch-only"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    model = None
    trace = None
    try:
        catalog = frontend_catalog(args)
        if args.list_models:
            print(render_model_catalog(catalog))
            return 0
        args = prepare_namespace(args, catalog)
        args.prompt = load_prompt(args)
        if not args.headless and (os.name != "posix" or not sys.stdin.isatty() or not sys.stdout.isatty()):
            raise ValueError(
                "interaction CLI requires POSIX terminal stdin/stdout; "
                "use --headless with --prompt or --prompt-file for one task"
            )
        if args.prompt is not None and not args.prompt.strip():
            raise ValueError("prompt must be a non-empty string or None")
        if args.max_samples is not None and args.max_samples <= 0:
            raise ValueError("max_samples must be a positive integer or None")
        if args.max_output_tokens is not None:
            SampleParams(max_output_tokens=args.max_output_tokens)
        save_path = resolve_save_path(args.save_path)
        if (args.headless and args.prompt is None
                and not (args.resume and args.instructions is not None and save_path.is_file())):
            raise ValueError(
                "--headless requires --prompt or --prompt-file, or "
                "--resume with --instructions and an existing save."
            )
        trace = DebugTrace.open(save_path, events=True, context_id=1, role="main") if args.debug_trace else None
        try:
            model = _build_model(args, trace)
        except CodexAuthUnavailable:
            if args.headless or not supports_account_services(args):
                raise
            model = None
        cwd = Path(args.cwd).expanduser().resolve()
        environment_manager = (
            DefaultEnvironment(cwd=cwd, enable_workspace=args.enable_workspace)
            if args.enable_default_tools else nullcontext(Environment())
        )
        with environment_manager as environment:
            if args.headless:
                return asyncio.run(_run_headless(
                    model, environment, args, path=save_path, trace=trace,
                ))
            terminal = PosixTerminal(sys.stdin, sys.stdout)
            return asyncio.run(_run(
                model, environment, terminal, args, path=save_path, trace=trace,
            ))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"interaction CLI failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            close_model(model)
        finally:
            if trace is not None:
                trace.close()
                warning = trace.take_warning()
                if warning:
                    with suppress(OSError, ValueError):
                        print(warning, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
