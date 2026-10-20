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
from datetime import datetime
from datetime import time as day_time
from datetime import timedelta
import importlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
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
from ._cli_save import CliSave
from ._cli_save import CliSaveWriter
from ._cli_save import NO_MIGRATION_OPTION
from ._cli_save import check_cli_save
from ._cli_save import cli_save_exists
from ._cli_save import prepare_cli_save
from ._cli_save import resume_cli_save
from ._cli_terminal import PosixTerminal
from ._debug_trace import DebugTrace
from ._prompt import load_prompt
from .codex_auth import CodexAuthUnavailable
from .compaction import CompactionError
from .compaction import CompactionResult
from .compaction import NothingToCompact
from .compaction import coldstart_unavailable
from .compaction import compaction_log
from .compaction import create_default_compactor
from .compaction import run_compaction
from .context import InteractionContext
from .default_environment import DefaultEnvironment
from .display import DisplayItem
from .display import message_summary
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
from .media import require_media_api
from .media import split_leading_references
from .loop import Interrupt
from .loop import Icron
from .loop import IcronError
from .loop import Preemption
from .loop import Steer
from .loop import TurnHost
from .loop import Urgency
from .loop import command_urgency
from .loop import escalated_stop
from .loop import run_turn
from .loop import turn_yield
from .loop.icron import Job as IcronJob
from .loop.icron import log_state as icron_log_state
from .loop.tail import describe_tail
from .loop.tail import turn_tail
from .model import Model
from .model import ModelAuthenticationError
from .model import ModelError
from .model import ModelTimeoutError
from .model import SampleParams
from .model import close_model
from .model_config import DEFAULT_SAVE_PREFIX
from .model_config import _boolean_argument
from .model_config import build_model
from .model_config import build_parser
from .model_config import initial_model_name
from .model_config import resolve_save_path
from .model_config import supports_account_services
from .model_config import frontend_catalog, prepare_namespace, render_model_catalog
from ._model_binding_debug import save_debug_model_bindings
from .runtime_config import ConfigError
from .runtime_config import InteractionConfig
from .save import interaction_item_from_dict
from .save import interaction_item_to_dict
from .save import InteractionSaveWriter
from .save import PUBLIC_LOG_REASONING
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
    Urgency.IMMEDIATE: ("closing — waiting for current operation; Ctrl-C or /exit!! cancels it "
                        "where possible"),
    Urgency.PREEMPT: "closing — cancelling current operation where possible",
}


@dataclass(frozen=True, eq=False)
class _RetryIntent:
    """Identity ticket for one live sampling failure, never model input."""


@dataclass(frozen=True, eq=False)
class _ContinueIntent:
    """/continue: sample the saved context without a user message; never model input.

    It also samples when the log ends with /compact's coldstart pointer, even
    after a finished turn (_ends_with_coldstart_compact): the model then
    follows the pointer. /compact queues one for that itself (_follow_pointer),
    as does a cancelled reboot that was to follow it (_reboot_ready).
    """


@dataclass(frozen=True, eq=False)
class _WakeIntent:
    """icron's due wake-ups, queued as text the user typed then would be.

    One marker stands for every due wake-up: a running turn takes it as a
    steer, and otherwise it starts a turn, both with icron's message (see
    _take_interrupt and the outer loop). It isn't the user's input: it never
    counts toward the queue's limit or its counts (_queued_input).
    """


@dataclass(frozen=True, eq=False)
class _Submission:
    """Submitted text and its user message, read when Enter accepted the text.

    ``steers`` is false for ``/task <text>``: a running turn never takes it as
    a steer, so it becomes the next query, and text typed after it queues
    behind it (a steer never overtakes input queued before it).
    """

    text: str
    message: Message
    steers: bool = True


@dataclass
class _Reboot:
    """A requested reboot: the CLI restarts on the current code (see _reboot_ready).

    ``cause`` is "/reboot", or "coldstart" for a coldstart compaction with
    compaction_coldstart_reboot. ``continues`` is set once the reboot ends a
    running turn, which the restarted CLI then continues. ``pointer`` is set
    for /compact's coldstart compaction: the restarted CLI then has the model
    follow the pointer at once (_rebooted_goes_on). ``ready`` is set once the
    session shuts down for it.
    """

    cause: str
    continues: bool = False
    pointer: bool = False
    ready: bool = False


@dataclass(frozen=True)
class _Rebooted:
    """What a restarted CLI (--rebooted) carries over from the process it replaced.

    ``cause`` is that process's _Reboot.cause, and None when the state was
    lost (``warning`` says why): then nothing else carries over. The restarted
    CLI works out _Reboot.pointer from the cause and the save.
    """

    cause: Optional[str] = None
    continues: bool = False
    config: dict = field(default_factory=dict)  # the /config changes: key -> value
    draft: Editor = field(default_factory=Editor)
    queue: tuple = ()  # queued input, as _Submission, UserToolIntent, and _ContinueIntent
    exit_code: int = 0
    warning: Optional[str] = None


def _plain_message(text: str) -> Message:
    return Message("user", text)


def _reads_no_files(text: str) -> bool:
    return False


@dataclass
class _UIState:
    editor: Editor = field(default_factory=Editor)
    pending: deque[Union[_Submission, UserToolIntent, _RetryIntent, _ContinueIntent,
                         _WakeIntent]] = field(default_factory=deque)
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
    # icron (--enable-experimental-icron), or None. cron_entries maps the
    # jobs that /cron made to their entries as typed, for /crontab;
    # icron_resumes counts the turns that icron started since the user's
    # last input; icron_auth_noticed is true once the notice that due
    # wake-ups wait for a model has been shown.
    icron: Optional[Icron] = field(default=None, repr=False)
    cron_entries: dict = field(default_factory=dict)
    icron_resumes: int = 0
    icron_auth_noticed: bool = False
    # A requested reboot (/reboot, compaction_coldstart_reboot), or None; set
    # on the event loop thread, and read by the context thread at a turn's
    # interrupt point. rebooted is what this process carries over from the
    # one it replaced, when it is a restarted CLI.
    reboot: Optional[_Reboot] = None
    rebooted: Optional[_Rebooted] = field(default=None, repr=False)

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
        command's wait), and the model is closed. Closing cancels the model's
        work in flight, also outside the turn's operations, such as a
        compaction (automatic or /compact), and the model then refuses every
        request, so a compaction between its summary requests stops too.
        Retiring would not do: it cancels only a request in flight. Nothing
        uses the model after a stop. Outcomes are still saved.
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
        if self.icron is not None:
            self.icron.stop()  # every stop cancels running code
        self.changed.set()
        if self.stop_level >= Urgency.PREEMPT and callable(
                getattr(self.active_model, "close", None)):
            model = self.active_model

            def close():
                try:
                    close_model(model)
                except Exception:
                    self.notice("Closing the model failed; final cleanup will run.")
            threading.Thread(target=close, name="interaction-model-close", daemon=True).start()

    def escalate_stop(self) -> None:
        """Ctrl-C, Ctrl-D, or SIGINT: /exit! the first time, then /exit!!."""
        self.request_stop(escalated_stop(self.stop_level))

    def _stop_hint(self) -> str:
        """What a stronger exit would do now, as a sentence (or nothing)."""
        if self.stop_level is None or self.stop_level < Urgency.IMMEDIATE:
            return (" Ctrl-C or /exit! stops sooner; /exit!! also cancels the current operation "
                    "where possible.")
        if self.stop_level < Urgency.PREEMPT:
            return " Ctrl-C or /exit!! cancels the current operation where possible."
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
            elif head == "/task":
                # The next query, never a steer: it waits for the running turn
                # to end, as plain text does when no turn can take a steer.
                parts = text.split(maxsplit=1)
                if len(parts) == 1:
                    self.notice("Usage: /task <text> queues the text as the next query "
                                "instead of steering the running turn.")
                elif self.auth_required:
                    self.notice(f"{self.auth_notice} Draft was not submitted.")
                else:
                    self._submit_text(parts[1], draft=text, steers=False)
                return
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
            elif head == "/reboot":
                self._request_reboot(text)
                return
            elif self.icron is not None and head in ("/cron", "/crontab", "/py"):
                self._icron_command(head, text)
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
            if _queued_input(self) >= _MAX_PENDING_QUERIES:
                self.notice("Query queue is full; the draft has not been submitted.")
            else:
                self.pending.append(intent)
                self.editor = Editor()
                self.changed.set()
                if isinstance(intent, UserToolIntent) and intent.notice is not None:
                    self.notice(intent.notice)

    def _submit_text(self, text: str, *, draft: Optional[str] = None,
                     flush: Optional[Urgency] = None, steers: bool = True) -> None:
        """Read a draft's attachments now, then queue its message.

        A bad reference rejects only this draft: it stays in the editor with
        the error, and nothing is queued. A draft with files to read is read
        on a helper thread, as auto submits tasks; meanwhile it stays in the
        editor, and Enter queues nothing else, so the queue keeps its order.
        ``draft`` is the editor text (``/steer <text>``, ``/task <text>``),
        ``flush`` the urgency to flush the queued steers with once the text is
        queued, and ``steers`` false for ``/task <text>`` (see _Submission).
        """
        draft = text if draft is None else draft
        if _queued_input(self) >= _MAX_PENDING_QUERIES:
            self.notice("Query queue is full; the draft has not been submitted.")
            return
        if self.reads_files(text):
            self.reading = draft
            future = asyncio.get_running_loop().run_in_executor(None, self.read_query, text)
            future.add_done_callback(
                functools.partial(self._read_done, text, draft, flush, steers))
            return
        try:
            message = self.read_query(text)
        except AttachmentError as exc:
            self.notice(str(exc))
            return
        self._queue_submission(_Submission(text, message, steers), draft, flush)

    def _read_done(self, text: str, draft: str, flush: Optional[Urgency], steers: bool,
                   future) -> None:
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
            self._queue_submission(_Submission(text, future.result(), steers), draft, flush)

    def _queue_submission(self, submission: _Submission, draft: str,
                          flush: Optional[Urgency] = None) -> None:
        if _queued_input(self) >= _MAX_PENDING_QUERIES:
            self.notice("Query queue is full; the draft has not been submitted.")
            return
        self.pending.append(submission)
        if self.editor.text == draft:
            self.editor = Editor()
        self.changed.set()
        if not submission.steers and self.phase not in {"idle", "failed", "auth needed"}:
            self.notice("Queued as the next query; it waits for the current turn to end.")
        if flush is not None:
            steered = _steer_count(self) == _queued_input(self)
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
        may be sooner than ``level`` (and at QUEUED nothing changes).

        During a compaction they arrive right after it, whatever the level: a
        steer never skips or cancels compaction, since the sample it goes to
        needs it. The flush still applies, in case the compaction just ended.
        """
        count = _steer_count(self)
        if not count:
            self.notice("No queued steers to flush.")
            return
        self.preemption.flush(level)
        level = max(Urgency(level), self.preemption.level)
        steers, them = (f"{count} steers", "them") if count > 1 else ("1 steer", "it")
        if self.phase == "compacting":
            self.notice(f"{steers} queued for the next sample, right after the compaction in "
                        "progress; steers never cancel compaction.")
        elif level < Urgency.IMMEDIATE:
            self.notice(f"{steers} queued for the next sample, after the current tool batch; "
                        f"/steer! or /steer!! delivers {them} sooner.")
        elif level < Urgency.PREEMPT:
            self.notice(f"Flushing {steers}: delivered once the current sample or tool call "
                        "finishes.")
        else:
            self.notice(f"Flushing {steers}: delivered as soon as the current sample or tool "
                        "call ends; a sample or command wait is cancelled where possible.")

    def _request_reboot(self, text: str) -> None:
        """/reboot: restart the CLI on the current code, at once while idle.

        Like /exit, it waits for a running turn's tool batch: the turn ends
        before its next sample, once it takes the steers queued so far, and
        the restarted CLI continues it. It waits for a running command too.
        Handled at once, like /exit: it is never queued.
        """
        self.editor = Editor()
        if text.strip() != "/reboot" or "\n" in text or "\r" in text:
            self.notice("Usage: /reboot (no arguments; single line).")
            return
        if self.reboot is not None:
            self.notice("A reboot is already pending.")
            return
        refusal = _reboot_refusal(self)
        if refusal is not None:
            self.notice(f"Cannot reboot: {refusal}.")
            return
        self.reboot = _Reboot("/reboot")
        if self.phase not in {"idle", "failed", "auth needed"}:
            self.notice("Rebooting once the current operation allows: a running turn finishes "
                        "its tool batch and continues after the restart. /exit exits instead.")
        self.changed.set()

    def _icron_command(self, head: str, text: str) -> None:
        """/cron, /crontab, and /py (--enable-experimental-icron): handled at
        once, on the event loop, even while main is busy or needs
        authentication; never queued or saved. A command that fails leaves
        its draft in the editor."""
        command = {"/cron": self._cron, "/crontab": self._crontab, "/py": self._py}[head]
        try:
            used = command(text)
        except IcronError as exc:
            self.notice(f"{exc} Draft preserved.")
            return
        if used:
            self.editor = Editor()

    def _cron(self, text: str) -> bool:
        """/cron: schedule the user's message, cancel jobs, or list them."""
        icron, words = self.icron, text.split(maxsplit=3)
        if len(words) == 1:
            self.notice(_cron_list(icron))
            return True
        form = words[1]
        if form == "cancel" and len(words) == 3:
            return self._cron_cancel(words[2])
        seconds = at = None
        if form in ("in", "every") and len(words) == 4:
            seconds = _parse_duration(words[2])
        elif form == "at" and len(words) == 4:
            at = _next_clock_time(words[2], icron.clock())
        if (seconds is None and at is None) or not words[3].strip():
            self.notice(f"{_CRON_USAGE} Draft preserved.")
            return False
        when, message, now = words[2], words[3].rstrip(), icron.clock()
        if form == "in":
            job = icron.schedule(message, after=seconds, by="user")
            text = f"at {_hms(job.due)} (in {_span(job.due - now)})"
        elif form == "every":
            job = icron.schedule(message, every=seconds, by="user")
            text = f"every {when}, first at {_hms(job.due)}"
        else:
            job = icron.schedule(message, at=at, by="user")
            tomorrow = " tomorrow" if at.date() != datetime.fromtimestamp(now).date() else ""
            text = f"at {_hms(job.due)}{tomorrow} (in {_span(job.due - now)})"
        self.cron_entries[job.job_id] = f"{form} {when} {message}"
        self.notice(f"cron: job {job.job_id} sends your message {text}.")
        return True

    def _cron_cancel(self, target: str) -> bool:
        icron = self.icron
        suspended = icron.main.state == "suspended"
        if target == "all":
            jobs = [job for job in icron.jobs(by="user") if job.pending]
            for job in jobs:
                with suppress(IcronError):  # it ended meanwhile
                    icron.cancel(job.job_id, by="user")
            text = (f"cron: cancelled {len(jobs)} job{'s' if len(jobs) != 1 else ''}" if jobs
                    else "cron: no pending jobs to cancel")
        else:
            job = icron.cancel(target, by="user")
            text = f"cron: cancelled job {job.job_id} ({_OWNER[job.origin]})"
        if suspended and icron.main.state != "suspended":
            text += "; main now waits for your message"
        self.notice(text + ".")
        return True

    def _crontab(self, text: str) -> bool:
        """/crontab: the user's /cron jobs as one table; /crontab clear cancels them."""
        icron, words = self.icron, text.split()
        pending = {job.job_id for job in icron.jobs(by="user") if job.pending}
        for job_id in [job_id for job_id in self.cron_entries if job_id not in pending]:
            del self.cron_entries[job_id]
        if words == ["/crontab"]:
            self.notice(_crontab_table(icron, self.cron_entries))
            return True
        if words != ["/crontab", "clear"]:
            self.notice(f"{_CRONTAB_USAGE} Draft preserved.")
            return False
        suspended, cancelled = icron.main.state == "suspended", 0
        for job_id in list(self.cron_entries):
            with suppress(IcronError):  # it ended meanwhile
                icron.cancel(job_id, by="user")
                cancelled += 1
            del self.cron_entries[job_id]
        text = (f"crontab: cancelled {cancelled} job{'s' if cancelled != 1 else ''}" if cancelled
                else "crontab: your table is empty; nothing to cancel")
        if suspended and icron.main.state != "suspended":
            text += "; main now waits for your message"
        self.notice(text + ".")
        return True

    def _py(self, text: str) -> bool:
        """/py <code>: run the user's code on icron's watcher thread, without
        waiting; its output, value, and traceback come as notices."""
        code = _command_body(text, "/py")
        if not code.strip():
            self.notice(f"{_PY_USAGE} Draft preserved.")
            return False
        result = self.icron.exec(code, wait=0, by="user")
        if result.job is None:  # it didn't compile
            self.notice(f"py: failed:\n{textwrap.indent(result.error or '', '  ')}\n"
                        "Draft preserved.")
            return False
        return True


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


def _compaction_log(writer, environment):
    """The log that coldstart points to: a CLI save's, or else the file that ``writer`` saves."""
    own = getattr(writer, "compaction_log", None)
    return own(environment) if callable(own) else compaction_log(writer.path, environment)


def _drift_note(writer) -> Optional[str]:
    """A note when a coldstart pointer names a public log known to differ from the full log."""
    note = getattr(writer, "drift_note", None)
    return note() if callable(note) else None


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


# Host methods, whose checkpoints may carry an earlier remote checkpoint.
_HOST_COMPACTION_PROTOCOLS = frozenset(("pi", "warmpi", "coldstart"))


def _compaction_success_output(result: CompactionResult, *,
                               log_lines: Optional[int] = None) -> str:
    """The /compact result text, by the checkpoint's protocol.

    ``log_lines`` is how many lines of the save a coldstart pointer names.
    """
    checkpoint = result.items[0]
    assert isinstance(checkpoint, ContextPrefix)
    if result.protocol == "coldstart":
        lines = "" if log_lines is None else f" (lines 1\u2013{log_lines})"
        return (f"Context compacted using a coldstart pointer to the log{lines}; "
                "no model request was made.")
    # A host checkpoint can carry an earlier Responses checkpoint verbatim.
    remote = result.protocol == "responses_compaction_v2" or (
        result.protocol not in _HOST_COMPACTION_PROTOCOLS
        and any(isinstance(item, OpaqueCompaction) for item in checkpoint.prefix_items)
    )
    mode = (
        "a remote opaque checkpoint" if remote
        else "a warmpi summary checkpoint" if result.protocol == "warmpi"
        else "a pi summary checkpoint"
    )
    return f"Context compacted using {mode}."


def _with_failed_attempts(failures, output: str, error=None) -> str:
    """Prefix /compact's result with its failed attempts, one line each.

    They go in the result rather than in ModelFailure items: nothing may come
    between the user-tool call and its result, and a failure item after the
    result would make a finished turn look unfinished to /continue.
    """
    lines = [report.notice() for report in failures if report.error is not error]
    return "\n".join((*lines, output))


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
    rebooting = False  # a coldstart compaction with compaction_coldstart_reboot
    coldstart = False  # a coldstart compaction that succeeded
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
            failures = []
            try:
                with _traced_operation(state, "compact", context_revision=len(source_context)):
                    compaction = await _on_context_thread(
                        state, run_compaction, source_context, model,
                        snapshot.compaction_settings(),
                        factory=create_default_compactor,
                        tools=model_environment.tool_specs,
                        sample_params=snapshot.sample_params(),
                        instructions=_compact_focus(intent),
                        reason="manual",
                        log=_compaction_log(writer, model_environment),
                        stop_requested=lambda: (state.closing
                                                or state.stop_level is not None),
                        on_failure=failures.append,
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
                        _with_failed_attempts(failures, f"Nothing to compact: {exc}.", exc),
                        success=False,
                    )
                )
                contribution = (result_item,)
            except Exception as exc:
                if isinstance(exc, ModelAuthenticationError):
                    _mark_auth_required(state, exc)
                    model = None
                # A preempting stop closes the model, which fails the
                # compaction: that is a cancel, not a failure. A stop at a
                # lower level cancels nothing, so its failures are real.
                cancelled = (state.stop_level is not None
                             and state.stop_level >= Urgency.PREEMPT)
                result_item = UserToolResult(
                    ToolResult(
                        call.call.call_id,
                        _with_failed_attempts(failures, (
                            "Compaction cancelled during execution by the stop; no "
                            "compaction checkpoint was installed." if cancelled
                            else _compaction_failure_output(exc)), exc),
                        success=False,
                    )
                )
                contribution = (result_item,)
            else:
                result_item = UserToolResult(
                    ToolResult(
                        call.call.call_id,
                        _with_failed_attempts(failures, _compaction_success_output(
                            compaction, log_lines=len(source_context))),
                    )
                )
                # Validate and save these together: a durable success result
                # must never exist without the checkpoint it describes.
                contribution = (
                    result_item,
                    *compaction.context_items(),
                )
                coldstart = compaction.protocol == "coldstart"
                rebooting = (compaction.protocol == "coldstart"
                             and snapshot.compaction_coldstart_reboot)

        await _append(context, contribution, state, writer)
        state.displays.extend(
            render_interaction_items(
                contribution,
                source_user_calls=(call,),
            )
        )
        note = _drift_note(writer) if coldstart else None
        if note is not None:
            state.notice(note)
        if rebooting:
            _coldstart_reboot(state, pointer=True)
        if coldstart:
            _follow_pointer(state, context)
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


def _query_message(query: str, args, cwd: Path,
                   config: Optional[InteractionConfig] = None) -> Message:
    """The user message for submitted text; raises AttachmentError.

    Local paths follow ``config``'s current enable_workspace, which /config
    changes, as the tools do; without ``config``, the launch option.
    """
    message = parse_user_prompt(
        query,
        cwd=cwd,
        enabled=args.enable_experimental_media,
        enable_workspace=(args.enable_workspace if config is None
                          else config.get("enable_workspace")),
    )
    require_media_api(message, args.model_api, subject=f"--endpoint-api {args.model_api}")
    return message


def _reads_files(text: str, args) -> bool:
    """Whether _query_message reads files for this text (leading @ references)."""
    return bool(args.enable_experimental_media and split_leading_references(text)[0])


def _steer_count(state: _UIState) -> int:
    """Leading steering submissions in the queue: a running turn takes them as
    steers. A command, or a /task submission, ends the run (see _Submission).
    A wake-up marker in the run is taken with them, but isn't counted."""
    if not state.turn_active:
        return 0
    count = 0
    for entry in state.pending:
        if not _steers(entry):
            break
        count += isinstance(entry, _Submission)
    return count


def _steers(entry) -> bool:
    return isinstance(entry, _WakeIntent) or (isinstance(entry, _Submission) and entry.steers)


def _queued_input(state: _UIState) -> int:
    """The user's queued entries: a wake-up marker isn't one."""
    return sum(not isinstance(entry, _WakeIntent) for entry in state.pending)


def _take_steers(state: _UIState) -> tuple:
    """The steers a running turn takes at its interrupt point; this also resets
    their urgency (flushed steers are delivered now)."""
    submissions = []
    while state.turn_active and state.pending and _steers(state.pending[0]):
        submissions.append(state.pending.popleft())
    state.preemption.take()
    return tuple(submissions)


def _take_interrupt(state: _UIState) -> tuple:
    """The user messages a running turn takes at its interrupt point: its
    steers, with icron's message in a wake-up marker's place among them."""
    messages = []
    for entry in _take_steers(state):
        if not isinstance(entry, _WakeIntent):
            messages.append(entry.message)
            continue
        wakeup = None if state.icron is None else state.icron.take_wakes()
        if wakeup is not None:  # None: withdrawn since it was queued
            messages.append(Message("user", wakeup.text()))
    return tuple(messages)


# icron (--enable-experimental-icron): its wake-ups, its commands, and its notices.

_CRON_USAGE = ("Usage: /cron in 10m <text>, /cron at 14:30 <text>, /cron every 1h <text>, "
               "/cron cancel <job-id>|all, or /cron alone to list jobs (units: s, m, h, d).")
_CRONTAB_USAGE = ("Usage: /crontab shows your /cron jobs as a table, and /crontab clear "
                  "cancels them.")
_PY_USAGE = ("Usage: /py <code> runs Python in this process, on icron's watcher thread, with "
             "icron, main, asyncio, and print.")
_DURATION = re.compile(r"([1-9][0-9]*)([smhd])")
_CLOCK_TIME = re.compile(r"([01]?[0-9]|2[0-3]):([0-5][0-9])")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_OWNER = {"main": "main's", "user": "yours", "host": "the host's"}
_ICRON_YIELD_NOTICE = ("Note: the save ends with main yielded to icron. Jobs are not saved, so no "
                       "wake-up will resume it; enter a message to continue.")
_ICRON_RESTART_NOTICE = ("Note: icron jobs, running code, and Python state from before the "
                         "restart were not restored.")
_ICRON_CONTINUE_REFUSAL = ("Nothing to continue: main's last turn yielded to icron, which still "
                           "needs a wake-up. Enter a message instead.")


def _hms(seconds: float) -> str:
    return datetime.fromtimestamp(seconds).strftime("%H:%M:%S")


def _span(seconds: float) -> str:
    """A duration such as 10m, 1h 30m, or 23h 59m 10s."""
    seconds = max(0, int(round(seconds)))
    parts = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        if seconds >= size:
            parts.append(f"{seconds // size}{unit}")
            seconds %= size
    return " ".join(parts[:3]) or "0s"


def _one_line(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _parse_duration(word: str) -> Optional[int]:
    """/cron's <duration>: a positive whole number and one unit (s, m, h, d)."""
    match = _DURATION.fullmatch(word)
    return None if match is None else int(match.group(1)) * _UNIT_SECONDS[match.group(2)]


def _next_clock_time(word: str, now: float) -> Optional[datetime]:
    """/cron's <HH:MM>: the next such local time, tomorrow once it has passed today."""
    match = _CLOCK_TIME.fullmatch(word)
    if match is None:
        return None
    clock = day_time(int(match.group(1)), int(match.group(2)))
    today = datetime.fromtimestamp(now).date()
    when = datetime.combine(today, clock).astimezone()
    if when.timestamp() <= now:
        when = datetime.combine(today + timedelta(days=1), clock).astimezone()
    return when


def _command_body(text: str, head: str) -> str:
    """What follows a command's word, dedented: /py's code."""
    body = text.lstrip()[len(head):].lstrip(" \t")
    if body.startswith("\n"):
        body = body[1:]
    return textwrap.dedent(body)


def _job_timing(job: IcronJob, now: float) -> str:
    repeat = "once" if job.every is None else f"every {_span(job.every)}"
    if job.state != "running":
        return f"{_hms(job.due)}, in {_span(job.due - now)}, {repeat}"
    if job.kind == "message":
        text = "sent" if job.started is None else f"sent at {_hms(job.started)}"
    else:
        text = ("code waiting to start" if job.started is None
                else f"running code since {_hms(job.started)}")
    return text if job.due is None else f"{text}, next {_hms(job.due)}"


def _job_line(job: IcronJob, now: float) -> str:
    """One pending job, as /cron lists it and /crontab's comments show it."""
    what = job.message if job.message is not None else job.code or ""
    return f"{job.job_id} ({_OWNER[job.origin]}) {_job_timing(job, now)}: {_one_line(what)}"


def _cron_list(icron: Icron) -> str:
    now = icron.clock()
    jobs = [job for job in icron.jobs(by="user") if job.pending]
    if not jobs:
        return "cron: no pending jobs."
    lines = [f"cron: {len(jobs)} pending job{'s' if len(jobs) != 1 else ''} (now {_hms(now)}):"]
    lines += [f"  {_job_line(job, now)}" for job in jobs]
    return "\n".join(lines)


def _crontab_table(icron: Icron, entries: dict) -> str:
    """The user's /cron jobs as entries, each under a comment with its ID and
    next firing, and the other pending jobs as comments."""
    now = icron.clock()
    pending = [job for job in icron.jobs(by="user") if job.pending]
    mine = [job for job in pending if job.job_id in entries]
    others = [job for job in pending if job.job_id not in entries]
    head = f"crontab (now {_hms(now)}): "
    if mine:
        head += (f"{len(mine)} job{'s' if len(mine) != 1 else ''} of yours; /crontab clear "
                 "cancels them.")
    else:
        head += "no jobs of yours; /cron schedules one."
    lines = [head]
    for job in mine:
        if job.state == "scheduled":
            repeat = "once" if job.every is None else f"every {_span(job.every)}"
            when = f"next {_hms(job.due)} (in {_span(job.due - now)}), {repeat}"
        else:
            when = _job_timing(job, now)
        lines += [f"# {job.job_id}: {when}", _one_line(entries[job.job_id])]
    if others:
        lines.append("# Not in your table (/cron cancel ends them):")
        lines += [f"#   {_job_line(job, now)}" for job in others]
    return "\n".join(lines)


def _icron_status(icron: Icron) -> str:
    """icron's part of the status line, or nothing."""
    status = icron.status()
    if status.stalled is not None:
        job_id, seconds = status.stalled
        return f"icron: {job_id} has held the watcher for {int(seconds)}s"
    if status.main_state == "suspended" and not status.undelivered:
        job = status.suspended_on
        if job is None:
            return "icron: main waits for a wake-up"
        if job.due is None or job.state == "running":
            return f"icron: main waits for {job.job_id} (running)"
        return f"icron: main waits for {job.job_id} at {_hms(job.due)}"
    if status.undelivered:
        return f"icron: {status.undelivered} due"
    running = sum(job.state == "running" and job.kind != "message" for job in status.jobs)
    dues = [job.due for job in status.jobs if job.due is not None]
    parts = [f"{running} running"] if running else []
    if dues:
        parts.append(f"{len(dues)} job{'s' if len(dues) != 1 else ''}, next {_hms(min(dues))}")
    return f"icron: {', '.join(parts)}" if parts else ""


def _yield_notice(job: Optional[IcronJob], now: float) -> str:
    if job is None:
        return ("main yielded to icron, but nothing that could resume it is pending; it waits "
                "for your message.")
    if job.due is None or job.state == "running":
        return (f"main yielded to icron; it waits for job {job.job_id} (running code), unless "
                "you write first.")
    return (f"main yielded to icron; job {job.job_id} resumes it at {_hms(job.due)} "
            f"(in {_span(job.due - now)}), unless you write first.")


def _queue_wake(state: _UIState) -> None:
    """On the event loop: queue the delivery of icron's due wake-ups, as text
    the user typed now would be. One marker stands for all of them. While
    the CLI needs authentication they wait in icron, with one notice."""
    icron = state.icron
    if icron is None or state.closing or state.persistence_failed:
        return
    count = icron.undelivered()
    if not count or any(isinstance(entry, _WakeIntent) for entry in state.pending):
        return
    if state.auth_required:
        if not state.icron_auth_noticed:
            state.icron_auth_noticed = True
            due = "1 wake-up is" if count == 1 else f"{count} wake-ups are"
            state.notice(f"icron: {due} due; delivered once the model is ready (/login).")
        return
    state.icron_auth_noticed = False
    state.pending.append(_WakeIntent())
    state.changed.set()


def _waits_for_icron(state: _UIState) -> bool:
    """True while main is suspended: its last turn yielded to icron."""
    return state.icron is not None and state.icron.main.state == "suspended"


def _post(loop, function, *args) -> None:
    """Run function on the event loop thread, from any thread."""
    try:
        loop.call_soon_threadsafe(function, *args)
    except RuntimeError:
        pass  # the event loop has closed


def _last_user_text(context: InteractionContext) -> str:
    """The raw log's last user message, as the report of a turn without one."""
    for item in reversed(context.items):
        if isinstance(item, Message) and item.role == "user":
            return message_summary(item)
    return ""


def _safe_notice(state: _UIState, text: str) -> None:
    with suppress(OSError, ValueError):
        state.notice(text)


# Reboots (/reboot, compaction_coldstart_reboot): the CLI shuts down as on
# exit, then runs its own command line again, with --rebooted. The restarted
# CLI resumes the save on the code as it is now, and goes on where this one
# stopped (see _Rebooted).

# Ends a restarted CLI's command line. With "=check", the CLI only loads its
# code and parses its options, then exits: the start check (_check_reboot).
_REBOOTED_OPTION = "--rebooted"
# Names the descriptor of an unlinked file that holds what a restarted CLI
# carries over (_reboot_state): it survives exec and leaves nothing behind.
_REBOOT_STATE_FD = "PYTHIA_INTERACTION_REBOOT_FD"
_REBOOT_STATE_VERSION = 1
_REBOOT_CHECK_SECONDS = 60
_REBOOT_CAUSES = ("/reboot", "coldstart")
# This process's command line (_command_line), set by main() when it parses
# the process's own arguments; otherwise None, and the CLI can't reboot.
_reboot_argv: Optional[list] = None


def _command_line() -> Optional[list]:
    """This process's command line, interpreter options included, without
    the --rebooted that a reboot appended; None if it can't be known.

    Python 3.9 has no ``sys.orig_argv``: there, only a module run with -m or
    a script runs again, without interpreter options.
    """
    if not sys.executable:
        return None
    original = getattr(sys, "orig_argv", None)
    if original:
        argv = [sys.executable, *original[1:]]
    else:
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        if spec is not None and spec.name:
            argv = [sys.executable, "-m", spec.name, *sys.argv[1:]]
        elif sys.argv and sys.argv[0] not in ("", "-c"):
            argv = [sys.executable, *sys.argv]
        else:
            return None
    if len(argv) > 1 and argv[-1] == _REBOOTED_OPTION:
        argv.pop()
    return argv


def _reboot_command() -> Optional[list]:
    """The restarted CLI's command line, or None (see _reboot_argv)."""
    return None if _reboot_argv is None else [*_reboot_argv, _REBOOTED_OPTION]


def _reboot_refusal(state: _UIState) -> Optional[str]:
    """Why the CLI can't reboot now, as a clause, or None.

    It never reboots while icron has jobs or wake-ups: they live only in this
    process. (icron's Python state goes, as on any restart.)
    """
    if _reboot_command() is None:
        return "this process's command line isn't known, so it can't be run again"
    if state.persistence_failed:
        return "a checkpoint failed, and unsaved state remains in memory"
    if state.icron is not None:
        status = state.icron.status()
        if status.jobs:
            count = len(status.jobs)
            return (f"icron has {count} pending job{'s' if count != 1 else ''}, and jobs "
                    "live only in this process (/cron lists them)")
        if status.undelivered or status.main_state == "suspended":
            return "main waits for an icron wake-up, which lives only in this process"
    return None


def _coldstart_reboot(state: _UIState, *, pointer: bool = False) -> None:
    """compaction_coldstart_reboot, after a coldstart compaction: request a
    reboot, unless an exit is pending or _reboot_refusal objects.

    ``pointer`` is for /compact's compaction, which runs between turns: the
    restarted CLI then has the model follow its pointer at once, as the turn
    of an automatic compaction continues. A /reboot typed during /compact
    becomes this reboot; otherwise a pending reboot stays as it is.
    """
    if state.closing:
        return
    if state.reboot is not None:
        if pointer:
            state.reboot.cause, state.reboot.pointer = "coldstart", True
        return
    refusal = _reboot_refusal(state)
    if refusal is not None:
        state.notice(f"compaction_coldstart_reboot: not rebooting, since {refusal}.")
        return
    state.reboot = _Reboot("coldstart", pointer=pointer)


_POINTER_RESUME_NOTICE = (
    "Note: resumed save ends with /compact's coldstart pointer, which the model hasn't "
    "followed yet. No model request was started; enter /continue to have the model "
    "follow it, or a query to continue with a new message.")
_ICRON_POINTER_NOTICE = ("Note: main yielded to icron, so the model follows the coldstart "
                         "pointer when a wake-up or your message resumes it.")


def _follow_pointer(state: _UIState, context: InteractionContext) -> None:
    """After /compact's coldstart compaction: the model follows the pointer at
    once, as within the turn of an automatic compaction. The pointer is a user
    message that asks the model to update the summary and pick up where the
    log left off, so a continuation (/continue) goes first in the queue. Input
    typed meanwhile waits for that turn to end, as input queued before any turn
    does (a steer never overtakes older input).

    An exit wins, and a reboot that is to follow the pointer does so itself
    (_coldstart_reboot). After an icron yield, a wake-up or the user's message
    resumes main instead, and its turn starts with the pointer, as after a
    coldstart reboot (_rebooted_goes_on).
    """
    if state.closing or (state.reboot is not None and state.reboot.pointer):
        return
    if Icron.yield_recorded_in(context.items):
        state.notice(_ICRON_POINTER_NOTICE)
        return
    state.pending.appendleft(_ContinueIntent())
    state.changed.set()


def _check_reboot() -> Optional[str]:
    """The start check: the restarted CLI's command line with --rebooted=check,
    which loads the current code and parses the options. None if it passes,
    else why not, as a clause. It blocks: run it off the event loop."""
    command = _reboot_command()
    if command is None:
        return "this process's command line isn't known"
    try:
        check = subprocess.run([*command[:-1], f"{_REBOOTED_OPTION}=check"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, timeout=_REBOOT_CHECK_SECONDS)
    except subprocess.TimeoutExpired:
        return f"the restarted CLI's start check didn't finish within {_REBOOT_CHECK_SECONDS} s"
    except OSError as exc:
        return f"the restarted CLI's start check couldn't run ({exc.strerror or type(exc).__name__})"
    if check.returncode == 0:
        return None
    lines = [line for line in check.stderr.decode("utf-8", "replace").splitlines()
             if line.strip()]
    detail = f": {_one_line(lines[-1], 240)}" if lines else ""
    return (f"the restarted CLI would fail to start (its start check exited with status "
            f"{check.returncode}{detail})")


async def _reboot_ready(state: _UIState) -> bool:
    """Whether the session shuts down now for its requested reboot.

    The reboot is dropped, with a notice, if _reboot_refusal now objects (as
    when main scheduled an icron job meanwhile), or if the start check fails.
    A turn that it ended then continues here, and a pointer it was to follow
    is followed here.
    """
    reboot, phase = state.reboot, state.phase
    refusal = _reboot_refusal(state)
    if refusal is None:
        state.set_phase("rebooting")
        try:
            refusal = await asyncio.get_running_loop().run_in_executor(None, _check_reboot)
        except Exception as exc:  # dropped below, so that the loop never retries it
            refusal = f"the restarted CLI's start check failed ({type(exc).__name__})"
    if state.closing:
        return False  # an exit came meanwhile: it wins
    if refusal is None:
        reboot.ready = True
        state.notice("Rebooting: the CLI restarts on the current code"
                     + (", and the model follows the coldstart pointer." if reboot.pointer
                        else " and continues the turn." if reboot.continues else "."))
        return True
    state.reboot = None
    state.set_phase(phase)
    state.notice(f"Reboot cancelled: {refusal}."
                 + (" The model follows the coldstart pointer here." if reboot.pointer
                    else " The turn continues here." if reboot.continues else ""))
    if reboot.continues or reboot.pointer:
        state.pending.appendleft(_ContinueIntent())
    if state.pending:
        state.changed.set()  # run the queue: the continuation first, if any
    return False


def _reboot_state(state: _UIState, config: InteractionConfig, queued=None) -> dict:
    """What the restarted CLI carries over (_Rebooted), as JSON. ``queued`` is
    the queue as it was before the session's last stop, by default the queue."""
    initial = config.initial_values()
    changes = {key: value for key, value in config.values().items() if value != initial[key]}
    queue = []
    for entry in (state.pending if queued is None else queued):
        if isinstance(entry, _Submission):
            queue.append({"kind": "text", "text": entry.text, "steers": entry.steers,
                          "message": interaction_item_to_dict(entry.message)})
        elif isinstance(entry, UserToolIntent):
            queue.append({"kind": "command", "name": entry.name,
                          "arguments_json": entry.arguments_json})
        elif isinstance(entry, (_ContinueIntent, _RetryIntent)):
            # A retry's ticket belongs to this process; /continue reads the
            # log, which has the failure too.
            queue.append({"kind": "continue"})
        # icron has no wake-ups to carry (_reboot_refusal).
    return {
        "version": _REBOOT_STATE_VERSION, "cause": state.reboot.cause,
        "continues": state.reboot.continues, "config": changes,
        "draft": {"text": state.editor.text, "cursor": state.editor.cursor},
        "queue": queue, "exit_code": state.exit_code,
    }


def _parse_reboot_state(data) -> _Rebooted:
    """The _Rebooted of _reboot_state's JSON. Raises KeyError, TypeError, or
    ValueError for anything else."""
    if not isinstance(data, dict) or data.get("version") != _REBOOT_STATE_VERSION:
        raise ValueError("unknown reboot state")
    cause, config, draft, queued = data["cause"], data["config"], data["draft"], []
    text, cursor, exit_code = draft["text"], draft["cursor"], data["exit_code"]
    if (cause not in _REBOOT_CAUSES or not isinstance(config, dict)
            or not isinstance(text, str) or not _is_int(cursor) or not _is_int(exit_code)):
        raise ValueError("invalid reboot state")
    for entry in data["queue"]:
        kind = entry["kind"]
        if kind == "text":
            message = interaction_item_from_dict(entry["message"])
            if not (isinstance(entry["text"], str) and isinstance(message, Message)
                    and message.role == "user"):
                raise ValueError("invalid queued text")
            queued.append(_Submission(entry["text"], message, entry["steers"] is True))
        elif kind == "command":
            if not (isinstance(entry["name"], str) and isinstance(entry["arguments_json"], str)):
                raise ValueError("invalid queued command")
            queued.append(UserToolIntent(entry["name"], entry["arguments_json"]))
        elif kind == "continue":
            queued.append(_ContinueIntent())
        else:
            raise ValueError("invalid queued input")
    return _Rebooted(cause=cause, continues=data["continues"] is True, config=config,
                     draft=Editor(text, min(max(cursor, 0), len(text))),
                     queue=tuple(queued), exit_code=exit_code)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _take_reboot_state() -> _Rebooted:
    """What the replaced process left this restarted CLI (_reboot_exec). It
    closes the descriptor that the state came in, and never raises."""
    name = os.environ.pop(_REBOOT_STATE_FD, None)  # command sessions don't inherit it
    if name is None:
        return _Rebooted(warning="it got no state from the process it replaced")
    try:
        descriptor = int(name)
        if descriptor < 3:
            raise ValueError("not a state file's descriptor")
        with os.fdopen(descriptor, "rb") as file:
            file.seek(0)
            return _parse_reboot_state(json.loads(file.read().decode("utf-8")))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return _Rebooted(warning=f"its state couldn't be read ({type(exc).__name__})")


def _rebooted_goes_on(rebooted: Optional[_Rebooted],
                      context: InteractionContext) -> Optional[str]:
    """How a restarted CLI goes on at once, without a new message, or None.

    "turn": it continues the turn that the reboot ended, as /continue would.
    "pointer": the save ends with the coldstart pointer of the /compact that
    caused the reboot, and the model follows it: it updates the summary and
    picks up where the log left off, as after an automatic compaction. The
    cause and the save decide this, not _Reboot.pointer, so that a process on
    older code, which doesn't mark it, gets the same. Neither applies after
    an icron yield, which only a wake-up resumes.
    """
    if rebooted is None or Icron.yield_recorded_in(context.items):
        return None
    if rebooted.cause == "coldstart" and _ends_with_coldstart_compact(context):
        return "pointer"
    if rebooted.continues and turn_tail(context.items).unfinished:
        return "turn"
    return None


def _rebooted_notice(rebooted: _Rebooted, goes_on: Optional[str]) -> str:
    """The restarted CLI's startup line; ``goes_on`` as _rebooted_goes_on,
    if the CLI samples."""
    if rebooted.cause is None:
        return (f"Note: this CLI restarted for a reboot, but {rebooted.warning}: the draft, "
                "queued input, and /config changes were not carried over.")
    cause = ("/reboot" if rebooted.cause == "/reboot"
             else "a coldstart compaction (compaction_coldstart_reboot)")
    return (f"Rebooted after {cause}: this is the current code"
            + {"turn": "; the turn continues.",
               "pointer": "; the model follows the coldstart pointer."}.get(goes_on, "."))


class _Rebooting(Exception):
    """Raised by _run and _run_headless once the session has shut down for a
    reboot, with what the restarted CLI carries over; main() then runs it."""

    def __init__(self, state: dict) -> None:
        super().__init__("rebooting")
        self.state = state


def _reboot_exec(state: dict) -> int:
    """Replace this process with the restarted CLI; returns only on failure."""
    command = _reboot_command()
    if command is None:  # _reboot_refusal checked it
        print("interaction CLI failed: the reboot can't run this process's command line "
              "again; rerun it with --resume to continue from the save.", file=sys.stderr)
        return 1
    with tempfile.TemporaryFile() as file:
        file.write(json.dumps(state, ensure_ascii=False).encode("utf-8"))
        file.flush()
        os.set_inheritable(file.fileno(), True)
        environment = {**os.environ, _REBOOT_STATE_FD: str(file.fileno())}
        with suppress(OSError, ValueError):
            sys.stdout.flush()
            sys.stderr.flush()
        try:
            os.execve(command[0], command, environment)
        except OSError as exc:
            print(f"interaction CLI failed: the reboot couldn't run {command[0]} "
                  f"({exc.strerror or type(exc).__name__}); rerun the command with --resume "
                  "to continue from the save.", file=sys.stderr)
            return 1


class _CliHost(TurnHost):
    """The CLI's side of the shared turn loop.

    Its methods run on the context thread. Every change to terminal state is
    posted to the event loop thread, in order, so the redraw (which copies and
    clears ``state.displays``) never races an update. Posted changes run
    before the turn's outcome reaches ``_drive_interaction``.
    """

    def __init__(self, state: _UIState, writer: InteractionSaveWriter, loop,
                 steering=False, config=None) -> None:
        self._state, self._writer, self._loop = state, writer, loop
        self._steering = steering
        self._config = config  # the turn's InteractionConfigSnapshot

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
        when Enter accepted it. icron's due wake-ups among the steers come as
        its message (_take_interrupt).
        """
        if not self._steering or self._state.closing:
            return super().interrupt()
        messages = self._on_loop(_take_interrupt, self._state)
        if self._state.closing:
            return Interrupt.STOP
        if not messages:
            return Interrupt.CONTINUE
        return Steer(messages)

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

    def reboot_requested(self) -> bool:
        return self._state.reboot is not None

    def compacted(self, result) -> None:
        note = _drift_note(self._writer) if result.protocol == "coldstart" else None
        if note is not None:
            self.notice(note)
        if (result.protocol == "coldstart" and self._config is not None
                and self._config.compaction_coldstart_reboot):
            # Synchronously: the interrupt point that follows must see it.
            self._on_loop(_coldstart_reboot, self._state)

    def preemptible(self, cancel):
        return self._state.preemption.operation(cancel)

    def log_path(self):
        save = getattr(self._writer, "cli_save", None)
        return self._writer.path if save is None else save.shown

    def compaction_log(self, environment):
        return _compaction_log(self._writer, environment)

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


async def _turn(context, model, environment, state, writer, config, *, steering=False,
                job_text="", resumes=0):
    """One turn of the shared loop, on the context thread.

    With ``steering``, text submitted during the turn reaches it right before
    the next sample. With icron, main's tool can end the turn with a yield,
    and the turn's end is main's handoff to icron: a Yield of ``job_text``,
    the turn's input as typed, after ``resumes`` turns that icron started
    since the user's last input.
    """
    # Each explicit attempt consumes the preceding failure's ticket; only a
    # failure the loop reports as retryable arms a new one.
    state.retry = None
    snapshot = config.snapshot()
    host = _CliHost(state, writer, asyncio.get_running_loop(), steering, snapshot)
    state.turn_active = not state.pending  # a steer never overtakes older input
    icron = state.icron
    if icron is not None:
        icron.begin_turn()
    try:
        result = await _on_context_thread(
            state, run_turn, context, model, environment, snapshot, host,
            control=icron)
    except Exception as exc:
        if icron is not None:
            icron.handoff(turn_yield(exc, context, job_text=job_text, resumes=resumes),
                          context.items)
        raise
    finally:
        state.turn_active = False
        state.preemption.take()  # steers left over become the next query
    if icron is not None:
        waits_for = icron.handoff(
            turn_yield(result, context, job_text=job_text, resumes=resumes), context.items)
        if result.kind == "yielded":
            state.notice(_yield_notice(waits_for, icron.clock()))
    return result


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


def _ends_with_coldstart_compact(context: InteractionContext) -> bool:
    """Whether the raw log ends with a coldstart checkpoint from /compact
    (its CompactionMetadata says so, as for compaction_coldstart_reboot).

    Its pointer is a user message that asks the model to update the summary
    and pick up where the log left off, and nothing has answered it yet.
    /compact has the model follow it at once (_follow_pointer), and /continue
    follows it too, even after a finished turn: turn_tail skips checkpoints,
    so for it the records before this one decide. A coldstart reboot's
    restarted CLI follows it as well (_rebooted_goes_on).
    """
    if not _ends_with_completed_manual_compaction(context):
        return False
    last = next(item for item in reversed(context.items) if not isinstance(item, Tools))
    return isinstance(last, CompactionMetadata) and last.protocol == "coldstart"


def _resume_notice(context: InteractionContext) -> Optional[str]:
    # Inspect the raw tail, not the model context established by a ContextPrefix.
    # A sample boundary does not record stop_reason or turn completion.
    if _ends_with_coldstart_compact(context):
        return _POINTER_RESUME_NOTICE
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


async def _drive_interaction(model, environment, state, args, save, config):
    state.active_model = model
    icron = state.icron
    if icron is not None:
        # icron posts its deliveries and notices to this event loop.
        loop = asyncio.get_running_loop()
        icron.start(functools.partial(_post, loop, _queue_wake, state),
                    functools.partial(_post, loop, _safe_notice, state))
    try:
        return await _drive_interaction_body(model, environment, state, args, save, config)
    finally:
        if icron is not None:
            # Before the model and the environment: no code runs after this.
            dropped = await _on_context_thread(state, icron.close)
            if dropped:
                _safe_notice(state, f"icron: dropped {len(dropped)} pending "
                                    f"job{'s' if len(dropped) != 1 else ''}; jobs and Python "
                                    "state live only in this process.")
        if callable(getattr(state.active_model, "close", None)):
            await _on_context_thread(state, close_model, state.active_model)


async def _drive_interaction_body(
    model: Optional[Model],
    environment: Environment,
    state: _UIState,
    args: argparse.Namespace,
    save: CliSave,
    config: InteractionConfig,
) -> None:
    attachment_cwd = Path(args.cwd).expanduser().resolve()
    # Enter reads each submission's attachments (see _UIState._submit_text).
    state.read_query = functools.partial(_query_message, args=args, cwd=attachment_cwd,
                                         config=config)
    state.reads_files = functools.partial(_reads_files, args=args)
    tools_snapshot = Tools(environment.tool_specs)
    # The save log shows reasoning as save_log_reasoning says while each line
    # is written, so a /config change applies from the next line on.
    reasoning = functools.partial(config.get, "save_log_reasoning")
    thinking = _relay_thinking(args)
    # Resuming reads the full log alone; the public log is brought up to date.
    existing = args.resume and await _on_context_thread(state, save.log.exists)
    incomplete_line = None
    if existing:
        context, writer, incomplete_line = await _on_context_thread(
            state, resume_cli_save, save, reasoning=reasoning, thinking=thinking)
        if state.rebooted is None:
            # A restarted CLI's terminal shows them already, from the process
            # it replaced.
            state.displays.extend(render_interaction_items(context.items))
        if incomplete_line is not None:
            state.notice(incomplete_line.warning(save.log_name))
        state.notice(
            "Note: command sessions and plan state were not restored. "
            "Old command session IDs are not resumable; use only IDs from this run."
        )
        if state.rebooted is None and any(
            isinstance(item, UserToolCall)
            and item.call.name == "config"
            for item in context.items
        ):
            # (A reboot carries the /config changes over.)
            state.notice(
                "Note: in-memory configuration was reset from the current launch "
                "arguments; saved config commands were not replayed."
            )
    else:
        if args.resume:
            state.notice(
                f"Warning: no existing {save.shown.name} was found; a fresh one was created."
            )
        writer = CliSaveWriter(save, reasoning=reasoning, thinking=thinking)
        initial = [Init(model=args.model or initial_model_name(model))]
        if args.instructions is not None:
            initial.append(Instructions(args.instructions))
        initial.append(tools_snapshot)
        context = InteractionContext(initial)
    # The public log's warnings, such as a drift that resuming found, become
    # notices; the log never fails a save.
    writer.on_warning = functools.partial(_post, asyncio.get_running_loop(), _safe_notice, state)
    writer.report_warnings()
    if state.closing:
        return
    initial_query = args.prompt
    startup = True
    while not state.closing:
        sampling_attempt = False
        wake_turn = False  # this iteration's turn takes icron's wake-up as its input
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
                        save.binding_path,
                        {"main": args.model_binding},
                    )
                    if warning is not None:
                        state.notice(warning)
                query = initial_query
                rebooted = state.rebooted
                # A restarted CLI goes on at once where it can: it continues a
                # turn that the reboot ended, or follows /compact's pointer.
                goes_on = _rebooted_goes_on(rebooted, context) if existing else None
                should_sample = model is not None and (query is not None or (
                    existing and args.instructions is not None
                ) or goes_on is not None)
                if rebooted is not None:
                    state.notice(_rebooted_notice(rebooted, goes_on if should_sample else None))
                # A save that used icron says so, with the flag or without it.
                icron_note = icron_log_state(context.items) if existing else None
                if existing and not should_sample:
                    # A continuation that a reboot carried over goes on by itself.
                    continues = any(isinstance(entry, _ContinueIntent) for entry in state.pending)
                    notice = (_ICRON_YIELD_NOTICE if icron_note == "yielded"
                              else None if continues else _resume_notice(context))
                    if notice is not None:
                        state.notice(notice)
                if icron_note == "jobs":
                    state.notice(_ICRON_RESTART_NOTICE)
                if model is None:
                    query = None
                    state.notice(
                        f"{state.auth_notice} No model query was submitted."
                    )
                elif state.rebooted is None:
                    # The prompt is submitted (a restarted CLI's editor holds
                    # the draft it carried over instead).
                    state.editor = Editor()
                state.ready = True
                startup = False
            else:
                if state.reboot is not None and await _reboot_ready(state):
                    return
                _queue_wake(state)  # due wake-ups that a failure or a missing model held
                if state.headless:
                    # A headless run goes on only while main waits for icron.
                    while not state.changed.is_set():
                        if not _waits_for_icron(state):
                            return
                        with suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(state.changed.wait(), 0.5)
                else:
                    await state.changed.wait()
                state.changed.clear()
                if state.closing:
                    return
                if state.reboot is not None:
                    continue  # it goes before queued input: the next iteration reboots
                if not state.pending:
                    continue
                query = state.pending.popleft()
                if state.pending:
                    state.changed.set()
                if isinstance(query, _WakeIntent):
                    # Between turns, icron's due wake-ups start one, as text
                    # the user typed would.
                    wakeup = (state.icron.take_wakes()
                              if state.icron is not None and model is not None else None)
                    if wakeup is None:
                        continue  # withdrawn since it was queued, or no model: they stay due
                    text = wakeup.text()
                    query, wake_turn = _Submission(text, Message("user", text)), True
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
                    if Icron.yield_recorded_in(context.items):
                        state.notice(_ICRON_CONTINUE_REFUSAL)
                        continue
                    refusal = turn_tail(context.items).refusal
                    if refusal is not None and not _ends_with_coldstart_compact(context):
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
                    turn_text = query.text
                else:  # the startup prompt
                    try:
                        message = _query_message(query, args, attachment_cwd, config)
                    except AttachmentError as exc:
                        state.notice(str(exc))
                        if state.headless:
                            state.exit_code = 1
                            state.set_phase("failed")
                            return
                        state.editor = Editor(query, len(query))
                        state.set_phase("idle")
                        continue
                    turn_text = query
                user = UserInteraction((message,))
                await _append(context, user.context_items(), state, writer)
                state.displays.extend(user.display_items())
            else:  # /retry, /continue, or a resumed save's new instructions
                turn_text = _last_user_text(context)
            if should_sample:
                sampling_attempt = True
                state.icron_resumes = state.icron_resumes + 1 if wake_turn else 0
                result = await _turn(
                    context,
                    model,
                    environment,
                    state,
                    writer,
                    config,
                    steering=True,
                    job_text=turn_text,
                    resumes=state.icron_resumes,
                )
            else:
                result = None
            state.set_phase("auth needed" if state.auth_required else "idle")
            if state.reboot is not None:
                # Only a reboot ends a turn as stopped without an exit; the
                # restarted CLI continues it. The next iteration reboots.
                state.reboot.continues = (result is not None and result.kind == "stopped"
                                          and not state.closing)
                continue
            if state.headless and not (result is not None and result.kind == "yielded"
                                       and _waits_for_icron(state)):
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
            if _queued_input(state):
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


def _coldstart_unavailable(environment, args, path) -> Optional[str]:
    """Why this session can't run coldstart compaction, if it can't."""
    reason = coldstart_unavailable(environment.tool_specs, path)
    if reason is not None and path is not None and not getattr(args, "enable_default_tools", True):
        reason += ", and default model tools are disabled"
    return reason


def _runtime_config(environment, args, path=None):
    workspace_update = getattr(environment, "set_enable_workspace", None)
    config = InteractionConfig.from_namespace(
        args,
        on_enable_workspace=(
            workspace_update if callable(workspace_update) else None
        ),
        coldstart_unavailable=_coldstart_unavailable(environment, args, path),
    )
    if callable(workspace_update):
        workspace_update(config.get("enable_workspace"))
    return config


def _prepared_save(path, args) -> CliSave:
    """The save to run on: main prepares it, and a --save path is prepared here."""
    if isinstance(path, CliSave):
        return path
    return prepare_cli_save(resolve_save_path(Path(path)), resume=args.resume,
                            auto_migrate=getattr(args, "auto_migrate_save", True),
                            reasoning=getattr(args, "save_log_reasoning", "nil"))


def _relay_thinking(args) -> bool:
    """Whether the save log may show this session's thinking whole (save.public_log_line).

    Claude Relay's thinking has no summary, and it is text alone, as Chat
    Completions' full reasoning is, so only the session can tell them apart.
    """
    return getattr(args, "model_api", None) == "claude-relay"


def _startup_notices(state, args, save, config=None):
    for notice in save.notices:  # what preparing the save did, such as a migration
        state.notice(notice)
    reasoning = (getattr(args, "save_log_reasoning", "nil") if config is None
                 else config.get("save_log_reasoning"))
    state.notice(f"Save log: {save.shown}")
    if save.private is None:
        state.notice(f"Note: this save keeps its original format ({NO_MIGRATION_OPTION}): "
                     "it has the reasoning, and debug files go next to it.")
        if reasoning != "nil":
            state.notice(f"Note: save_log_reasoning = {reasoning!r} has no effect: a save "
                         "in its original format has no save log.")
    elif reasoning == "nil":
        state.notice(f"Private: {save.private}/, with the full log, which has the reasoning "
                     "that the save log leaves out, and debug files.")
    else:
        shown = ("reasoning summaries, and this session's Claude Relay thinking,"
                 if _relay_thinking(args) else "reasoning summaries")
        state.notice(f"Private: {save.private}/, with the full log, which resuming reads, "
                     f"and debug files. The save log shows {shown} as assistant messages "
                     "(save_log_reasoning).")
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
    if state.icron is not None:
        state.notice("icron (experimental): main can schedule work and run Python in this "
                     "process, unsandboxed, with the icron tool"
                     + ("." if state.headless else
                        "; /cron schedules your messages, /crontab shows them, and /py runs "
                        "your code.")
                     + " Jobs and Python state live only in this process.")
    dropped = None if config is None else config.coldstart_dropped
    if dropped is not None:
        state.notice(
            f"Note: this session compacts with {dropped[1]}, not compaction_mode "
            f"{getattr(args, 'compaction_mode', None) or dropped[0]} (--compaction-mode): "
            f"{config.coldstart_unavailable}."
        )


def _carry_over(state: _UIState, config: InteractionConfig, rebooted: _Rebooted) -> None:
    """Restore what a restarted CLI carries over, but the draft (see _run):
    the /config changes, the queued input, and the exit status."""
    for key, value in rebooted.config.items():
        try:
            config.set(key, value)
        except (ConfigError, TypeError, ValueError) as exc:
            state.notice(f"Note: the reboot didn't carry over {key} = {value!r}: {exc}")
    state.pending.extend(rebooted.queue)
    if state.pending:
        state.changed.set()
    state.exit_code = rebooted.exit_code


async def _run_headless(model, environment, args, path, *, trace=None, icron=None,
                        rebooted=None):
    """One explicit task, quiet context display, and orderly effect draining."""
    if model is None:
        raise ValueError("Headless execution requires an available model.")
    save = _prepared_save(path, args)
    config = _runtime_config(environment, args, save.shown)
    state = _UIState(
        headless=True,
        bound_account_id=getattr(getattr(model, "endpoint", None), "account_id", None),
        trace=trace,
        icron=icron,
        rebooted=rebooted,
    )
    _startup_notices(state, args, save, config)
    if rebooted is not None:
        _carry_over(state, config, rebooted)
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
    worker = asyncio.create_task(_drive_interaction(model, environment, state, args, save, config))
    try:
        while not worker.done():
            # The controller's display queue is transient, not a second log.
            state.displays.clear()
            await asyncio.wait((worker,), timeout=0.05)
        worker.result()
    finally:
        # A reboot that the session shut down for, unless a stop came first.
        rebooting = state.reboot is not None and state.reboot.ready and state.stop_level is None
        queued = tuple(state.pending)  # before the stop below drops it
        try:
            state.request_stop(Urgency.IMMEDIATE)  # never lowers a stop
            # Cancellation/SIGINT must not close command resources before an
            # in-flight request/tool has finished and checkpointed its outcome.
            await asyncio.shield(worker)
            state.displays.clear()
        finally:
            if previous_sigint is not None:
                signal.signal(signal.SIGINT, previous_sigint)
    if rebooting:
        raise _Rebooting(_reboot_state(state, config, queued))
    return 130 if interrupted else state.exit_code


async def _run(
    model: Optional[Model],
    environment: Environment,
    terminal: PosixTerminal,
    args: argparse.Namespace,
    path: Union[Path, CliSave] = DEFAULT_SAVE_PREFIX,
    *,
    trace: Optional[DebugTrace] = None,
    icron: Optional[Icron] = None,
    rebooted: Optional[_Rebooted] = None,
) -> int:
    save = _prepared_save(path, args)
    config = _runtime_config(environment, args, save.shown)
    prompt = args.prompt or ""
    state = _UIState(
        editor=Editor(prompt, len(prompt)) if rebooted is None else rebooted.draft,
        auth_required=model is None,
        bound_account_id=getattr(getattr(model, "endpoint", None), "account_id", None),
        trace=trace,
        icron=icron,
        rebooted=rebooted,
    )
    state.notice(
        "pythia.interaction — /retry, /continue, /steer [text], /task <text>, /compact [focus], "
        "/config, /config.json, /login, /quota, /reboot, /exit or /quit. /steer and /exit wait "
        "for the current tool batch; with ! they wait only for the current operation, and with "
        "!! they also cancel it where possible. Ctrl-C/Ctrl-D is /exit!, and a second press "
        "/exit!!."
    )
    _startup_notices(state, args, save, config)
    if rebooted is not None:
        _carry_over(state, config, rebooted)
    worker = None
    frame = 0
    with terminal:
        try:
            # Draw the pre-filled editor before starting startup I/O/auto-submit.
            terminal.render(state.editor, "starting", tuple(state.displays))
            state.displays.clear()
            worker = asyncio.create_task(
                _drive_interaction(model, environment, state, args, save, config)
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
                queued = _queued_input(state) - steers
                if queued > 0:
                    status += f" | queued={queued}"
                if state.icron is not None and state.stop_level is None:
                    part = _icron_status(state.icron)
                    if part:
                        status += f" | {part}"
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
            # A reboot that the session shut down for, unless an exit came first.
            rebooting = (state.reboot is not None and state.reboot.ready
                         and state.stop_level is None)
            queued = tuple(state.pending)  # before the stop below drops it
            state.request_stop(Urgency.IMMEDIATE)  # never lowers a stop
            if worker is not None:
                # Do not cancel a thread-backed effect or close its environment early.
                await asyncio.shield(worker)
    if rebooting:
        raise _Rebooting(_reboot_state(state, config, queued))
    return state.exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        "Interactive POSIX shell or headless task using Pythia's caller-owned interaction API.",
        allow_prompt_file=True,
        save_prefix=True,
    )
    # Unlike the one-shot demo, the CLI continues its save by default;
    # --resume=False starts a new save, replacing the file.
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--auto-migrate-save",
        nargs="?",
        const=True,
        default=True,
        type=_boolean_argument,
        metavar="{False,True}",
        help=(
            "when resuming a save in the original format, one file with reasoning at the "
            "--save path, migrate it: the file becomes PREFIX._private/context.jsonl, its "
            "debug files move along, and PREFIX.jsonl becomes its save log. A migration "
            "that fails is undone, and the CLI exits. False keeps using such a save as it "
            "is. Launch-only. A bare flag means True (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--save-log-reasoning",
        choices=PUBLIC_LOG_REASONING,
        default="nil",
        help=(
            "how the save log, PREFIX.jsonl, shows each reasoning item, on its line: nil, as "
            "a {\"type\": \"nil\"} line; assistant, as an assistant message with the item's "
            "summary, or a nil line when it has none (with Claude Relay, the session's own "
            "thinking shows whole). The full log keeps the reasoning either way. It applies "
            "to the lines written while it is in effect; tunable with /config "
            "save_log_reasoning (default: %(default)s)"
        ),
    )
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
            "sensitive payloads and credentials, to PREFIX._private/trace.req.jsonl, "
            "trace.res.jsonl and trace.events.jsonl (for a save kept in its original format, "
            "next to it, its extension replaced by .trace.req.jsonl and so on); never "
            "truncated, even when --resume=False replaces the save; launch-only"
        ),
    )
    parser.add_argument(
        "--enable-experimental-icron",
        nargs="?",
        const=True,
        default=False,
        type=_boolean_argument,
        metavar="{False,True}",
        help=(
            "experimental: run icron, an in-process scheduler. The model gets the icron "
            "tool, which schedules wake-ups and runs Python in this process, unsandboxed; "
            "/cron and /crontab schedule and show your messages, and /py runs your code. "
            "Jobs and Python state live only in memory. Launch-only: repeat when resuming. "
            "A bare flag means True (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--compaction-coldstart-reboot",
        nargs="?",
        const=True,
        default=False,
        type=_boolean_argument,
        metavar="{False,True}",
        help=(
            "restart the CLI on the current code after each coldstart compaction, as /reboot "
            "does, to load code changed during the session; tunable with /config "
            "compaction_coldstart_reboot. A bare flag means True (default: %(default)s)"
        ),
    )
    # Appended by a reboot (_reboot_command): this run replaces a process that
    # rebooted. --rebooted=check only loads the code and parses the options.
    parser.add_argument("--rebooted", nargs="?", const="restart", default=None,
                        choices=("restart", "check"), help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _reboot_argv
    args = _build_parser().parse_args(argv)
    # Only a CLI run on the process's own arguments can run them again.
    _reboot_argv = _command_line() if argv is None else None
    model = None
    trace = None
    icron = None
    rebooted = None
    reboot_state = None
    try:
        catalog = frontend_catalog(args)
        if args.list_models:
            print(render_model_catalog(catalog))
            return 0
        if args.rebooted is not None:
            # A restarted CLI continues its save. The first launch used the
            # prompt and the instructions already.
            args.resume, args.prompt, args.prompt_file, args.instructions = True, None, None, None
        args = prepare_namespace(args, catalog)
        if args.rebooted == "check":
            if not args.headless:
                importlib.import_module("pythia.term_input")  # the terminal imports it late
            # A save that the restarted CLI couldn't migrate fails the check,
            # which keeps the running session.
            check_cli_save(resolve_save_path(args.save_path),
                           auto_migrate=args.auto_migrate_save)
            return 0  # the start check (_check_reboot): the code loads, and the options parse
        if args.rebooted is not None:
            rebooted = _take_reboot_state()
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
        if (args.headless and args.prompt is None and rebooted is None
                and not (args.resume and args.instructions is not None
                         and cli_save_exists(save_path))):
            raise ValueError(
                "--headless requires --prompt or --prompt-file, or "
                "--resume with --instructions and an existing save."
            )
        # Before the trace and the model: a migration that fails ends the CLI here.
        save = prepare_cli_save(save_path, resume=args.resume,
                                auto_migrate=args.auto_migrate_save,
                                reasoning=args.save_log_reasoning)
        trace = (DebugTrace.open_paths(*save.trace_paths, context_id=1, role="main")
                 if args.debug_trace else None)
        try:
            model = _build_model(args, trace)
        except CodexAuthUnavailable:
            if args.headless or not supports_account_services(args):
                raise
            model = None
        cwd = Path(args.cwd).expanduser().resolve()
        if args.enable_experimental_icron:
            icron = Icron()
        tools = () if icron is None else (icron.tool(),)
        environment_manager = (
            DefaultEnvironment(cwd=cwd, enable_workspace=args.enable_workspace,
                               **({"extra_tools": tools} if tools else {}))
            if args.enable_default_tools else nullcontext(Environment(tools))
        )
        with environment_manager as environment:
            if args.headless:
                return asyncio.run(_run_headless(
                    model, environment, args, path=save, trace=trace, icron=icron,
                    rebooted=rebooted,
                ))
            terminal = PosixTerminal(sys.stdin, sys.stdout)
            return asyncio.run(_run(
                model, environment, terminal, args, path=save, trace=trace, icron=icron,
                rebooted=rebooted,
            ))
    except _Rebooting as rebooting:
        reboot_state = rebooting.state  # the session has shut down: restart once it's closed
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"interaction CLI failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            try:
                if icron is not None:
                    icron.close()  # idempotent: the run closes it first
            finally:
                close_model(model)
        finally:
            if trace is not None:
                trace.close()
                warning = trace.take_warning()
                if warning:
                    with suppress(OSError, ValueError):
                        print(warning, file=sys.stderr)
    return _reboot_exec(reboot_state)


if __name__ == "__main__":
    raise SystemExit(main())
