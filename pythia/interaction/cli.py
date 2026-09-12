"""Caller-owned interaction loop with a scrollback POSIX terminal shell.

Run with ``python3 -m pythia.interaction.cli``. Line-shell compatibility and
active-effect interruption are deliberately deferred.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
import os
from pathlib import Path
import queue
import sys
import threading
import time
import uuid
from typing import Optional
from typing import Sequence
from typing import Union

from ._cli_editor import Editor
from ._cli_terminal import PosixTerminal
from .codex_auth import CodexAuthUnavailable
from .compaction import CompactionError
from .compaction import CompactionResult
from .compaction import create_default_compactor
from .compaction import should_auto_compact
from .context import ModelContext
from .default_environment import DefaultEnvironment
from .display import DisplayItem
from .display import render_interaction_items
from .environment import Environment
from .items import CompactionMetadata
from .items import ContextPrefix
from .items import Init
from .items import Instructions
from .items import InteractionItem
from .items import Message
from .items import ModelFailure
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
from .items import summarize_turn_usage
from .model import Model
from .model import ModelAuthenticationError
from .model import ModelError
from .model import SamplingOptions
from .model_config import DEFAULT_SAVE_PATH
from .model_config import build_model
from .model_config import build_parser
from .model_config import initial_model_name
from .model_config import resolve_save_path
from .model_config import supports_account_services
from .save import load_interaction_save
from .save import save_interaction_save
from .user import UserInteraction
from .user_tools import UserToolIntent
from .user_tools import create_user_environment
from .user_tools import parse_user_tool


FRAME_INTERVAL = 1 / 128
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_MAX_PENDING_QUERIES = 8


@dataclass
class _UIState:
    editor: Editor = field(default_factory=Editor)
    pending: deque[Union[str, UserToolIntent]] = field(default_factory=deque)
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

    def set_phase(self, phase: str) -> None:
        self.phase, self.phase_started = phase, time.monotonic()

    def notice(self, text: str) -> None:
        text = text.rstrip("\r\n")
        self.displays.append(DisplayItem(f"[cli] {text}"))

    def request_exit(self) -> None:
        self.closing = True
        self.pending.clear()
        self.login_cancel.set()
        self.changed.set()

    def handle_key(self, key: str, data: str) -> None:
        if key in {"c-c", "c-d"}:
            self.request_exit()
        elif not self.closing and self.ready:
            if key != "c-m":
                self.editor = self.editor.edit(key, data)
                return
            text = self.editor.text
            head = text.split(maxsplit=1)[0] if text.strip() else ""
            if head in {"/quit", "/exit"}:
                self.request_exit()
                return
            if not text.strip():
                self.editor = Editor()
                return
            if self.persistence_failed:
                self.notice("Checkpoint failed; no further work will run. Use /quit.")
                return
            intent = text
            if head.startswith("/"):
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
            if len(self.pending) >= _MAX_PENDING_QUERIES:
                self.notice("Query queue is full; the draft has not been submitted.")
            else:
                self.pending.append(intent)
                self.editor = Editor()
                self.changed.set()


async def _checkpoint(context: ModelContext, state: _UIState, path: Path) -> None:
    state.set_phase("saving")
    try:
        await asyncio.to_thread(save_interaction_save, path, context.copy())
    except Exception:
        state.persistence_failed = True
        raise


async def _append(
    context: ModelContext,
    items: Iterable[InteractionItem],
    state: _UIState,
    path: Path,
) -> None:
    context.extend(items)
    await _checkpoint(context, state, path)


async def _sweep_tools(
    context: ModelContext, environment: Environment, state: _UIState, path: Path
) -> None:
    for call in context.pending_tool_calls():
        if state.closing:
            return
        state.set_phase(f"tool: {call.name}")
        result = await asyncio.to_thread(environment.execute_tool_calls, (call,))
        await _append(context, result.context_items(), state, path)
        state.displays.extend(result.display_items(source_calls=(call,)))


async def _fail_pending_tools(
    context: ModelContext, state: _UIState, path: Path, *, reason: str
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
        await _append(context, (result,), state, path)
        state.displays.extend(
            render_interaction_items((result,), source_calls=(call,))
        )


async def _fail_pending_user_tools(
    context: ModelContext, state: _UIState, path: Path,
) -> None:
    for call in context.pending_user_tool_calls():
        if state.closing:
            return
        if call.call.name == "compact":
            output = (
                "Compaction outcome unavailable after interruption. The command "
                "was not rerun; no durable compaction checkpoint was installed."
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
        await _append(context, (result,), state, path)
        state.displays.extend(render_interaction_items((result,), source_user_calls=(call,)))


def _has_provider_history(context: ModelContext) -> bool:
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
    if exc.failure is not None and exc.failure.auth_source == "environment":
        state.auth_notice = (
            "Environment credential rejected; update it and restart the process."
        )
    elif exc.failure is not None and exc.failure.auth_source == "static":
        state.auth_notice = (
            "Configured static credential rejected; restart with updated credentials."
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
    opaque = any(
        isinstance(item, OpaqueCompaction)
        for item in checkpoint.prefix_items
    )
    mode = "a remote opaque checkpoint" if opaque else "a prompt summary checkpoint"
    return f"Context compacted using {mode}."


async def _compact_user_tool(
    intent: UserToolIntent,
    model: Optional[Model],
    context: ModelContext,
    model_environment: Environment,
    state: _UIState,
    path: Path,
) -> Optional[Model]:
    # A pending UserToolCall deliberately makes a context non-sampleable. Take
    # the immutable compaction source first, then durably record authorization
    # for the effect before starting it.
    source_context = context.copy()
    call = UserToolCall(
        ToolCall(
            intent.name,
            "user_" + uuid.uuid4().hex,
            intent.arguments_json,
        )
    )
    await _append(context, (call,), state, path)
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
        await _append(context, (result_item,), state, path)
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
                compactor = create_default_compactor(model)
                compaction = await asyncio.to_thread(
                    compactor.compact,
                    source_context,
                    tools=model_environment.tool_specs,
                )
                if not isinstance(compaction, CompactionResult):
                    raise TypeError(
                        "compactor must return CompactionResult, got "
                        f"{type(compaction).__name__}"
                    )
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

        await _append(context, contribution, state, path)
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
    intent: UserToolIntent, model: Optional[Model], context: ModelContext,
    state: _UIState, path: Path, args: argparse.Namespace,
    model_environment: Environment,
) -> Optional[Model]:
    if intent.name == "compact":
        return await _compact_user_tool(
            intent,
            model,
            context,
            model_environment,
            state,
            path,
        )
    expected_account = state.bound_account_id
    call = UserToolCall(ToolCall(intent.name, "user_" + uuid.uuid4().hex, intent.arguments_json))
    await _append(context, (call,), state, path)
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
            expected_account=expected_account,
            provider_history=_has_provider_history(context),
        )
        state.set_phase(f"user tool: {intent.name}")
        outcome = await asyncio.to_thread(environment.execute_tool_calls, (call.call,))
        result = UserToolResult(outcome.items[0])
        await _append(context, (result,), state, path)
        state.displays.extend(render_interaction_items((result,), source_user_calls=(call,)))
    finally:
        state.active_user_call = None
    if intent.name == "login" and result.result.success and not state.closing:
        state.set_phase("loading model")
        try:
            model = await asyncio.to_thread(build_model, args)
            if (expected_account is not None and
                    getattr(getattr(model, "endpoint", None), "account_id", None) != expected_account):
                raise ValueError("credential account changed during activation")
        except Exception:
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


async def _turn(
    context: ModelContext,
    model: Model,
    environment: Environment,
    state: _UIState,
    path: Path,
    args: argparse.Namespace,
    options: Optional[SamplingOptions],
) -> None:
    turn_started = time.perf_counter()
    samples = 0
    while not state.closing:
        if args.max_samples is not None and samples >= args.max_samples:
            raise RuntimeError(
                "model did not produce a final answer within "
                f"{args.max_samples} samples"
            )
        threshold = getattr(model, "auto_compact_context_tokens", None)
        if (
            isinstance(threshold, int)
            and not isinstance(threshold, bool)
            and threshold > 0
            and should_auto_compact(context, threshold)
        ):
            state.set_phase("compacting")
            compactor = create_default_compactor(model)
            compaction = await asyncio.to_thread(
                compactor.compact,
                context.copy(),
                tools=environment.tool_specs,
            )
            if not isinstance(compaction, CompactionResult):
                raise TypeError(
                    "compactor must return CompactionResult, got "
                    f"{type(compaction).__name__}"
                )
            await _append(
                context,
                compaction.context_items(),
                state,
                path,
            )
            state.displays.extend(compaction.display_items())
            if state.closing:
                return
        state.set_phase("sampling")
        try:
            sample = await asyncio.to_thread(
                model.sample,
                context.copy(),
                tools=environment.tool_specs,
                options=options,
            )
        except ModelError as exc:
            contribution = (
                *exc.completed_items,
                *((exc.failure,) if exc.failure is not None else ()),
            )
            if contribution:
                recovered = (*contribution, ModelSampleBoundary())
                await _append(context, recovered, state, path)
                state.displays.extend(render_interaction_items(contribution))
                recovered_calls = tuple(
                    item for item in exc.completed_items
                    if isinstance(item, ToolCall)
                )
                if recovered_calls:
                    results = tuple(
                        ToolResult(
                            call_id=call.call_id,
                            output=(
                                "Not executed because the model response did "
                                "not complete."
                            ),
                            success=False,
                        )
                        for call in recovered_calls
                    )
                    await _append(context, results, state, path)
                    state.displays.extend(
                        render_interaction_items(
                            results,
                            source_calls=recovered_calls,
                        )
                    )
            raise
        samples += 1
        model_account_id = getattr(
            getattr(model, "endpoint", None),
            "account_id",
            None,
        )
        if state.bound_account_id is None and model_account_id is not None:
            state.bound_account_id = model_account_id
        await _append(context, sample.context_items(), state, path)
        state.displays.extend(sample.display_items())
        if sample.stop_reason == "compaction":
            continue
        if not sample.tool_calls:
            final_text = sample.last_assistant_text
            if not final_text or not final_text.strip():
                raise RuntimeError("model returned no final assistant text")
            summary = summarize_turn_usage(
                context.items,
                elapsed_seconds=time.perf_counter() - turn_started,
            )
            await _append(context, (summary,), state, path)
            state.displays.extend(render_interaction_items((summary,)))
            return
        await _sweep_tools(context, environment, state, path)


def _ends_with_completed_manual_compaction(context: ModelContext) -> bool:
    items = context.items
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


def _resume_notice(context: ModelContext) -> Optional[str]:
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
            ),
        ):
            continue
        if isinstance(item, (TurnSummary, Init)):
            return None
        if isinstance(item, OpaqueCompaction):
            tail = "a compaction checkpoint"
        elif isinstance(item, ContextPrefix):
            tail = "a context-prefix checkpoint"
        elif isinstance(item, ToolResult):
            tail = "tool results"
        elif isinstance(item, Message) and item.role == "user":
            tail = "a user submission"
        elif isinstance(item, Message) and item.role == "assistant":
            tail = "assistant output"
        elif isinstance(item, Instructions):
            tail = "an instructions update"
        elif isinstance(item, ModelFailure):
            tail = "a failed model attempt"
        else:
            tail = "incomplete model output"
        return (
            f"Resumed save ends with {tail}, without recorded turn completion. "
            "The model stop reason is not saved. No model request was started; "
            "enter a query to continue."
        )
    return None


async def _drive_interaction(
    model: Optional[Model],
    environment: Environment,
    state: _UIState,
    args: argparse.Namespace,
    path: Path,
    options: Optional[SamplingOptions],
) -> None:
    existing = args.resume and await asyncio.to_thread(path.exists)
    if existing:
        context = await asyncio.to_thread(load_interaction_save, path)
        state.displays.extend(render_interaction_items(context.items))
        state.notice(
            "Command sessions and plan state were not restored. "
            "Old command session IDs are not resumable; use only IDs from this run."
        )
    else:
        if args.resume:
            state.notice(
                f"Warning: no existing {path.name} was found; a fresh one was created."
            )
        initial = [Init(model=args.model or initial_model_name(model))]
        if args.instructions is not None:
            initial.append(Instructions(args.instructions))
        context = ModelContext(initial)
    if state.closing:
        return
    initial_query = args.prompt
    startup = True
    while not state.closing:
        try:
            if startup:
                if existing:
                    await _fail_pending_user_tools(context, state, path)
                    await _fail_pending_tools(
                        context, state, path, reason="session restart"
                    )
                else:
                    # Initial-save failures retain the fresh context too.
                    await _checkpoint(context, state, path)
                if state.closing:
                    return
                if existing and args.instructions is not None:
                    instructions = Instructions(args.instructions)
                    await _append(context, (instructions,), state, path)
                    state.displays.extend(render_interaction_items((instructions,)))
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
                await _fail_pending_user_tools(context, state, path)
                if state.pending:
                    state.changed.set()
                # An explicit new query after a failed effect is not permission
                # to retry old calls whose side effects may already have happened.
                await _fail_pending_tools(
                    context, state, path, reason="an interrupted operation"
                )
                if state.closing:
                    return
                if isinstance(query, UserToolIntent):
                    model = await _user_tool(
                        query,
                        model,
                        context,
                        state,
                        path,
                        args,
                        environment,
                    )
                    state.set_phase("auth needed" if state.auth_required else "idle")
                    continue
                if model is None:
                    state.editor = Editor(query, len(query))
                    state.notice(f"{state.auth_notice} Draft was not submitted.")
                    state.set_phase("auth needed")
                    continue
                should_sample = True
            if state.closing:
                return
            if query is not None:
                user = UserInteraction((Message(role="user", content=query),))
                await _append(context, user.context_items(), state, path)
                state.displays.extend(user.display_items())
            if should_sample:
                await _turn(context, model, environment, state, path, args, options)
            state.set_phase("auth needed" if state.auth_required else "idle")
        except Exception as exc:
            state.exit_code = 1
            state.ready = True
            startup = False
            state.set_phase("failed")
            if isinstance(exc, ModelAuthenticationError):
                model = None
                _mark_auth_required(state, exc)
            state.notice(f"{type(exc).__name__}: {exc}")
            if state.pending:
                state.notice(
                    "Queued queries discarded after failure; submit again explicitly."
                )
            state.pending.clear()
            state.changed.clear()
            if state.persistence_failed:
                # Retain the unsaved context here until exit; never replay the effect.
                state.notice(
                    "Checkpoint failed; unsaved state remains in memory. Use /quit."
                )
                while not state.closing:
                    await state.changed.wait()
                    state.changed.clear()
                return


async def _run(
    model: Optional[Model],
    environment: Environment,
    terminal: PosixTerminal,
    args: argparse.Namespace,
    path: Path = DEFAULT_SAVE_PATH,
) -> int:
    path = Path(path).absolute()
    options = (
        SamplingOptions(max_tokens=args.max_tokens)
        if args.max_tokens is not None else None
    )
    prompt = args.prompt or ""
    state = _UIState(
        editor=Editor(prompt, len(prompt)), auth_required=model is None,
        bound_account_id=getattr(getattr(model, "endpoint", None), "account_id", None),
    )
    state.notice(
        "pythia.interaction — /compact, /login, /quota; "
        "/quit or /exit; Ctrl-C/Ctrl-D exit."
    )
    state.notice(f"Save log: {path}")
    state.notice(
        "Warning: exec_command runs without a sandbox; use a trusted model and workspace."
    )
    if not args.enable_workspace:
        state.notice(
            "Warning: workspace path restrictions are disabled; "
            "exec_command workdir and apply_patch paths may resolve "
            "outside --cwd."
        )
    worker = None
    frame = 0
    with terminal:
        try:
            # Draw the pre-filled editor before starting startup I/O/auto-submit.
            terminal.render(state.editor, "starting", tuple(state.displays))
            state.displays.clear()
            worker = asyncio.create_task(
                _drive_interaction(model, environment, state, args, path, options)
            )
            while True:
                for key in terminal.read_keys():
                    state.handle_key(key.key, key.data or "")
                if terminal.closed:
                    state.request_exit()
                while True:
                    try:
                        call_id, text = state.transient.get_nowait()
                    except queue.Empty:
                        break
                    if call_id == state.active_user_call:
                        state.notice(text)
                busy = state.phase not in {"idle", "failed", "auth needed"}
                status = (
                    "closing — waiting for current operation"
                    if state.closing else state.phase
                )
                if busy:
                    status += f" {int(time.monotonic() - state.phase_started)}s"
                if state.pending:
                    status += f" | queued={len(state.pending)}"
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
            state.request_exit()
            if worker is not None:
                # Do not cancel a thread-backed effect or close its environment early.
                await asyncio.shield(worker)
    return state.exit_code


def _build_parser() -> argparse.ArgumentParser:
    return build_parser(
        "Interactive POSIX shell using Pythia's caller-owned interaction API."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if os.name != "posix" or not sys.stdin.isatty() or not sys.stdout.isatty():
            raise ValueError(
                "interaction CLI requires POSIX terminal stdin/stdout; "
                "line-shell mode is deferred"
            )
        if args.prompt is not None and not args.prompt.strip():
            raise ValueError("prompt must be a non-empty string or None")
        if args.max_samples is not None and args.max_samples <= 0:
            raise ValueError("max_samples must be a positive integer or None")
        if args.max_tokens is not None:
            SamplingOptions(max_tokens=args.max_tokens)
        save_path = resolve_save_path(args.save_path)
        try:
            model = build_model(args)
        except CodexAuthUnavailable:
            if not supports_account_services(args):
                raise
            model = None
        cwd = Path(args.cwd).expanduser().resolve()
        with DefaultEnvironment(
            cwd=cwd,
            enable_workspace=args.enable_workspace,
        ) as environment:
            terminal = PosixTerminal(sys.stdin, sys.stdout)
            return asyncio.run(_run(model, environment, terminal, args, path=save_path))
    except Exception as exc:
        print(f"interaction CLI failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
