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
from .context import ModelContext
from .default_environment import DefaultEnvironment
from .display import DisplayItem
from .display import render_interaction_items
from .environment import Environment
from .items import ContextCompaction
from .items import Init
from .items import Instructions
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
from .items import summarize_turn_usage
from .model import Model
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
                self.notice("Model authentication needed; use /login. Draft was not submitted.")
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
        result = UserToolResult(ToolResult(
            call.call.call_id,
            "User-tool outcome unavailable after interruption. The command was not rerun; "
            "credential side effects may already have occurred.", success=False,
        ))
        await _append(context, (result,), state, path)
        state.displays.extend(render_interaction_items((result,), source_user_calls=(call,)))


def _has_provider_history(context: ModelContext) -> bool:
    return any(
        (isinstance(i, SampleMetadata) and (i.provider_turn_id or i.provider_turn_state or i.provider_session_id))
        or (isinstance(i, Reasoning) and i.encrypted_content)
        or (isinstance(i, OpaqueCompaction) and i.protocol == "responses")
        for i in (*context.items, *context.model_items())
    )


async def _user_tool(
    intent: UserToolIntent, model: Optional[Model], context: ModelContext,
    state: _UIState, path: Path, args: argparse.Namespace,
) -> Optional[Model]:
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
            state.notice("Model ready. Submit a query; blocked drafts were not automatically submitted.")
        state.auth_required = model is None
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
    samples = 0
    while not state.closing:
        if args.max_samples is not None and samples >= args.max_samples:
            raise RuntimeError(
                "model did not produce a final answer within "
                f"{args.max_samples} samples"
            )
        state.set_phase("sampling")
        sample = await asyncio.to_thread(
            model.sample,
            context.copy(),
            tools=environment.tool_specs,
            options=options,
        )
        samples += 1
        await _append(context, sample.context_items(), state, path)
        state.displays.extend(sample.display_items())
        if sample.stop_reason == "compaction":
            continue
        if not sample.tool_calls:
            final_text = sample.last_assistant_text
            if not final_text or not final_text.strip():
                raise RuntimeError("model returned no final assistant text")
            summary = summarize_turn_usage(context.items)
            await _append(context, (summary,), state, path)
            state.displays.extend(render_interaction_items((summary,)))
            return
        await _sweep_tools(context, environment, state, path)


def _resume_notice(context: ModelContext) -> Optional[str]:
    # Inspect the raw tail, not a compaction's replacement model context.
    # A sample boundary does not record stop_reason or turn completion.
    for item in reversed(context.items):
        if isinstance(
            item,
            (ModelSampleBoundary, SampleMetadata, UserInteractionBoundary, UserToolCall, UserToolResult),
        ):
            continue
        if isinstance(item, (TurnSummary, Init)):
            return None
        if isinstance(item, (OpaqueCompaction, ContextCompaction)):
            tail = "a compaction checkpoint"
        elif isinstance(item, ToolResult):
            tail = "tool results"
        elif isinstance(item, Message) and item.role == "user":
            tail = "a user submission"
        elif isinstance(item, Message) and item.role == "assistant":
            tail = "assistant output"
        elif isinstance(item, Instructions):
            tail = "an instructions update"
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
                    state.notice("Model authentication needed; use /login. No model query was submitted.")
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
                    model = await _user_tool(query, model, context, state, path, args)
                    state.set_phase("auth needed" if state.auth_required else "idle")
                    continue
                if model is None:
                    state.editor = Editor(query, len(query))
                    state.notice("Model authentication needed; use /login. Draft was not submitted.")
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
    state.notice("pythia.interaction — /login, /quota; /quit or /exit; Ctrl-C/Ctrl-D exit.")
    state.notice(f"Save log: {path}")
    state.notice(
        "Warning: exec_command runs without a sandbox; use a trusted model and workspace."
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
        with DefaultEnvironment(cwd=cwd) as environment:
            terminal = PosixTerminal(sys.stdin, sys.stdout)
            return asyncio.run(_run(model, environment, terminal, args, path=save_path))
    except Exception as exc:
        print(f"interaction CLI failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
