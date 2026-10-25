"""icron: an in-process scheduler that supervises main.

Main hands off to icron whenever its turn loop stops (:meth:`Icron.handoff`).
A *job* runs code on the watcher thread's asyncio event loop, once or
repeatedly, first after a delay counted from the call that scheduled it
(``after``) or at an absolute time (``at``). A job's ``message`` is its
default code: it sends main the message and waits for the end of the turn
that receives it. :meth:`Icron.exec` runs code at once. Code has a handle to
main, ``main``: ``await main.resume(text)`` sends main a *wake-up* and returns
main's first handoff after receiving it.

The host delivers wake-ups (:meth:`Icron.take_wakes`) as it delivers text the
user typed: at an interrupt point of main's running turn, or as the input of a
new turn. Main can end its turn with ``yield`` (main's tool, :meth:`Icron.tool`)
and wait for a wake-up; meanwhile it is *suspended*. Nothing is saved: a
restart drops every job, running code, and Python state.

Threads. The watcher thread, ``interaction-icron``, runs the event loop: the
timetable and every run of code. The API's methods take one lock and never wait
for that loop, except :meth:`Icron.exec`, which waits for its own run, and
:meth:`Icron.close`. Code must not touch a context: only the context's own
thread may. It reads main through ``main.view()`` and ``main.last()``, and acts
on main only by sending it messages.

Trust. Code runs in this process, with its authority and unsandboxed, as the
commands of ``exec_command`` run as the user.
"""

from __future__ import annotations

import ast
import asyncio
import builtins
from collections import deque
from collections.abc import Mapping
import concurrent.futures
import contextvars
from dataclasses import dataclass
from datetime import datetime, timezone
import difflib
import inspect
import json
import linecache
import math
import re
import secrets
import sys
import threading
import time
import traceback
from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..environment import Tool
from ..environment import ToolOutcome
from ..environment import ToolSpec
from ..environment import current_cancel
from ..environment import current_results_due
from ..items import InteractionItem
from ..items import ToolCall
from ..items import ToolResult
from ..items import UserInteractionBoundary
from ..timeouts import DEFAULT_TOOL_TIMEOUT_SECONDS
from .supervision import Yield


JobId = str  # "<run>-<n>", such as "k3f9-2"; <run> is random per process
ContextId = int  # as in auto; v1 supervises only main
MAIN: ContextId = 1
TOOL_NAME = "icron"
ACTIONS = ("schedule", "exec", "yield", "cancel", "jobs")  # main's tool
ORIGINS = ("main", "user", "host")  # who scheduled a job, or ran code
WAKE_HEADER = "Wake-up from icron"
# Code is Python source, or, from host code only, a function or coroutine
# function, which each run calls with its Run.
Code = Union[str, Callable[["Run"], Optional[Awaitable[object]]]]

_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
_TICK_SECONDS = 30.0  # the timetable rereads the clock at least this often
_BEAT_SECONDS = 0.5  # the watcher loop's heartbeat, for the stall check
_ENDED_KEPT = 8  # ended jobs that jobs() still lists
_VALUE_CHARS = 2000  # a final expression's repr
_ERROR_CHARS = 8000  # a traceback
_LISTED_CODE_CHARS = 2000  # a job's code, as jobs() shows it: its first characters
_LISTED_OUTPUT_CHARS = 4000  # a job's latest output, as jobs() shows it: its last characters
_STREAM_LINES = 40  # output lines that one of the user's runs shows as notices
_WHO = {"main": "main's", "user": "the user's", "host": "the host's"}
# Why an exec call returned before its run ended, or why the run didn't run to
# its end: the reason in parentheses on the result's Wall time line (_exec_text).
_REASON_WAIT = "the wait ran out"
_REASON_MESSAGE = "it sent you a message, which you get after this tool batch"
_REASON_HANDOFF = "it awaits the end of your turn"
_REASON_INTERRUPTED = "the user interrupted the wait"
_REASON_DEADLINE = "the call reached its deadline"
_REASON_STOPPED = "a stop cancelled it"
_REASON_UNCOMPILED = "it didn't run"

# The run whose code is executing: tasks and asyncio.to_thread copy it.
_CURRENT_RUN: contextvars.ContextVar = contextvars.ContextVar("pythia_icron_run", default=None)


class IcronError(ValueError):
    """An invalid call. The text names the call and is safe to show a model."""


@dataclass(frozen=True)
class IcronLimits:
    max_jobs: int = 16                  # per origin: pending jobs, and undelivered wake-ups
    min_delay: float = 1.0              # earliest first firing for after=, in seconds
    max_delay: float = 7 * 24 * 3600.0  # latest first firing, and longest repeat
    min_every: float = 30.0             # shortest repeat
    max_chars: int = 8000               # a message or a note, like YIELD_MAX_CHARS
    max_wait: float = DEFAULT_TOOL_TIMEOUT_SECONDS  # exec's wait, within main's tool call
    max_output: int = 16000             # characters of a run's output that are kept
    stall_seconds: float = 5.0          # how long code may hold the watcher loop before status() says so


@dataclass(frozen=True)
class Job:
    """A snapshot of one job. The API returns snapshots, never live state."""

    job_id: JobId
    origin: str                      # who scheduled it: one of ORIGINS
    kind: str                        # "message", "code", or "exec"
    state: str                       # "scheduled", "running", "done", "failed", or "cancelled"
    due: Optional[float] = None      # the next firing (epoch seconds), if any
    message: Optional[str] = None    # a message job: the text it sends
    code: Optional[str] = None       # a code job: its source, or its function's name
    every: Optional[float] = None    # seconds between firings; None: once
    remaining: Optional[int] = None  # firings still to come; None: no limit
    created: float = 0.0             # epoch seconds
    started: Optional[float] = None  # the latest run's start
    ended: Optional[float] = None
    output: str = ""                 # the latest run's output, clipped
    error: Optional[str] = None      # a failed run's traceback

    @property
    def pending(self) -> bool:
        return self.state in ("scheduled", "running")


@dataclass(frozen=True)
class ExecResult:
    state: str                    # "done", "failed", "cancelled", "running", or "pending"
    output: str = ""              # what the run printed, clipped
    value: Optional[str] = None   # a final expression's value, as repr
    error: Optional[str] = None   # the traceback, when it failed
    job: Optional[Job] = None     # the job it goes on as, or ran as; None if it didn't compile
    # Why the call returned before the run ended, or why the run didn't run to
    # its end, as the result's Wall time line puts it: "the wait ran out", "a
    # stop cancelled it", "it didn't run", and so on. None for any other run
    # that ended within the call.
    reason: Optional[str] = None


@dataclass(frozen=True)
class Delivery:
    """One wake-up: what a run sent main, or the end of one of main's jobs."""

    job: Optional[Job]       # None: host code that ran outside any run
    origin: str
    text: str
    kind: str = "message"    # or "end"
    due: Optional[float] = None  # the slot that fired the run that sent it
    merged: int = 0          # firings merged into that run by the delivery


@dataclass(frozen=True)
class Wakeup:
    """The wake-ups that main takes together, as one user message."""

    deliveries: Tuple[Delivery, ...]
    resumes_yield: bool  # it starts main's first turn after a yield
    at: float            # when it was delivered (epoch seconds)

    def text(self) -> str:
        """Main's user message: WAKE_HEADER and the time, then each delivery,
        with its job's ID and origin and, for a scheduled job, its due time,
        repeat, and merged firings."""
        head = f"{WAKE_HEADER} at {_iso(self.at)}."
        if self.resumes_yield:
            head += " It resumes your turn that ended with yield."
        parts = [head]
        for delivery in self.deliveries:
            label = _delivery_label(delivery)
            parts.append("")
            if delivery.kind == "end":
                parts.append(f"{label} {delivery.text}")
            else:
                parts += [f"{label}:", delivery.text]
        return "\n".join(parts)


@dataclass(frozen=True)
class IcronStatus:
    jobs: Tuple[Job, ...]        # every pending job: running ones, then the soonest first
    main_state: str              # "running", "quiescent", or "suspended"
    suspended_on: Optional[Job]  # while main is suspended: the job it waits for
    undelivered: int             # wake-ups not yet delivered
    stalled: Optional[Tuple[str, float]]  # what holds the watcher loop (a job ID, or "?"), and for how long


class Run:
    """One run of a job's code, or of exec's: ``icron.current``, in code.

    ``due`` is the slot that fired it (None for exec), and ``merged`` counts
    the firings that merged into it: slots missed before it started, and
    firings while it ran.
    """

    def __init__(self, icron: "Icron", job: "_Job", *, due: Optional[float], merged: int) -> None:
        self._icron, self._job = icron, job
        self.job_id, self.origin, self.due, self.merged = job.job_id, job.origin, due, merged
        self._task = None
        self._started: Optional[float] = None
        self._done = False
        self._cancelled = False
        self._outcome: Optional[str] = None
        self._value: Optional[str] = None
        self._error: Optional[str] = None
        self._chunks: List[str] = []
        self._size = 0
        self._clipped = 0
        self._stream = job.origin == "user" and job.kind == "exec"
        self._partial = ""
        self._streamed = 0
        self._sent_wake = False
        self._awaits_main = False
        self._caller_waits = False  # an exec call waits for this run's result

    def __repr__(self) -> str:
        return f"<icron run of {self.job_id} ({self.origin})>"

    def output(self) -> str:
        """What the run printed so far, clipped to its last characters."""
        with self._icron._lock:
            text = "".join(self._chunks)
            if self._clipped:
                text = f"[{self._clipped} earlier characters omitted]\n" + text
            return text

    def _write(self, text: str) -> None:
        if not text:
            return
        icron, notices = self._icron, []
        limit = icron._limits.max_output
        with icron._lock:
            self._chunks.append(text)
            self._size += len(text)
            if self._size > 2 * limit:
                joined = "".join(self._chunks)
                drop = len(joined) - limit
                self._clipped += drop
                self._chunks, self._size = [joined[drop:]], limit
            if self._stream:
                *lines, self._partial = (self._partial + text).split("\n")
                notices = self._stream_lines_locked(lines)
        for notice in notices:
            icron._say(notice)

    def _stream_lines_locked(self, lines) -> List[str]:
        notices = []
        for line in lines:
            self._streamed += 1
            if self._streamed <= _STREAM_LINES:
                notices.append(f"py {self.job_id}: {line}")
            elif self._streamed == _STREAM_LINES + 1:
                notices.append(f"py {self.job_id}: more output omitted; /cron shows its latest.")
        return notices


class ContextHandle:
    """Code's handle to a context that icron supervises: in v1, main."""

    def __init__(self, icron: "Icron", context_id: ContextId) -> None:
        self._icron, self.id = icron, context_id

    @property
    def state(self) -> str:
        """"running", "quiescent", or "suspended" (its last turn yielded)."""
        with self._icron._lock:
            return self._icron._state

    def last(self) -> Optional[Yield]:
        """Its latest handoff."""
        with self._icron._lock:
            return self._icron._last_yield

    def view(self) -> Tuple[InteractionItem, ...]:
        """Its log as of its latest handoff."""
        with self._icron._lock:
            return self._icron._last_view

    def resume(self, text: str):
        """Send text now, as a wake-up, and return a future for the first
        handoff after its delivery. Awaiting it is optional: unawaited, resume
        only sends. Off the watcher loop it returns a concurrent future."""
        return self._icron._send_wake(text)

    def handoff(self):
        """A future for its next handoff."""
        return self._icron._next_handoff()

    def __repr__(self) -> str:
        return f"<icron context #{self.id} ({self.state})>"


class _Job:
    """A job's live state, guarded by its icron's lock."""

    def __init__(self, job_id, origin, kind, created, *, message=None, code=None,
                 compiled=None, value_name=None, due=None, every=None, remaining=None):
        self.job_id, self.origin, self.kind, self.created = job_id, origin, kind, created
        self.message, self.code = message, code
        self.compiled, self.value_name = compiled, value_name
        self.due, self.every, self.remaining = due, every, remaining
        self.state = "running" if kind == "exec" else "scheduled"
        # An exec run is listed as a job once its call returns before it ends,
        # and the user's are listed from the start (/cron shows them).
        self.listed = kind != "exec" or origin == "user"
        self.run: Optional[Run] = None
        self.started: Optional[float] = None
        self.ended: Optional[float] = None
        self.output = ""
        self.error: Optional[str] = None
        self.cancelled = False
        self.woke_in_suspension = False

    def code_text(self) -> Optional[str]:
        if self.code is None or isinstance(self.code, str):
            return self.code
        return getattr(self.code, "__qualname__", None) or repr(self.code)

    def snapshot(self) -> Job:
        output = self.output if self.run is None else self.run.output()
        state = self.state
        if self.cancelled and state in ("scheduled", "running"):
            state = "cancelled"  # its run, if any, is unwinding
        return Job(self.job_id, self.origin, self.kind, state, self.due, self.message,
                   self.code_text(), self.every, self.remaining, self.created, self.started,
                   self.ended, output, self.error)


class _Wake:
    """A wake-up for main, from a run or host code (kind "message"), or a
    job's end (kind "end"), until its delivery. Its future, if any, resolves
    at main's first handoff after the delivery."""

    def __init__(self, job, run, text, kind, future, origin):
        self.job, self.run, self.text, self.kind = job, run, text, kind
        self.future, self.origin = future, origin


class _CodeIcron:
    """The icron that code sees: bound to its origin, without exec or yield_."""

    def __init__(self, icron: "Icron", origin: str) -> None:
        self._icron, self._origin = icron, origin

    def schedule(self, message: Optional[str] = None, *, code: Optional[str] = None,
                 after: Optional[float] = None, at=None, every: Optional[float] = None,
                 count: Optional[int] = None) -> Job:
        """Schedule a job of yours, and return it (a Job): help('schedule')."""
        return self._icron.schedule(message, code=code, after=after, at=at, every=every,
                                    count=count, by=self._origin)

    def cancel(self, job_id: JobId) -> Job:
        """End one of your jobs, and return it (a Job): help('cancel')."""
        return self._icron.cancel(job_id, by=self._origin)

    def jobs(self) -> Tuple[Job, ...]:
        """Your jobs and the user's, as Jobs: help('jobs')."""
        return self._icron.jobs(by=self._origin)

    def now(self) -> datetime:
        """The current time, as an aware local datetime."""
        return datetime.fromtimestamp(self._icron.clock()).astimezone()

    @property
    def current(self) -> Optional[Run]:
        """The run of the code that asks."""
        return _CURRENT_RUN.get()

    def __repr__(self) -> str:
        return f"<icron for {_WHO[self._origin]} code>"


class Icron:
    """In-process scheduler that supervises main (later, more contexts).

    Thread-safe: its methods take one lock and, except exec() and close(),
    never wait for the watcher thread, whose event loop fires jobs and runs
    code. The host reports main's turns (begin_turn, handoff) and delivers
    wake-ups (take_wakes) where it delivers typed text. close() drops
    everything: nothing is saved.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time,
                 limits: IcronLimits = IcronLimits()) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable.")
        if not isinstance(limits, IcronLimits):
            raise TypeError("limits must be IcronLimits.")
        self._clock, self._limits = clock, limits
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        # Held around each call to the host's callbacks: once close() has
        # taken it, none runs again.
        self._callout = threading.Lock()
        self._prefix = "".join(secrets.choice(_ID_ALPHABET) for _ in range(4))
        self._counter = 0
        self._jobs: Dict[JobId, _Job] = {}  # every job that hasn't ended
        self._ended = deque(maxlen=_ENDED_KEPT)
        self._wakes: List[_Wake] = []  # sent, not yet delivered
        self._delivered: List[_Wake] = []  # delivered since the last handoff
        self._handoff_waiters: list = []
        self._namespaces: dict = {}
        self._main = ContextHandle(self, MAIN)
        self._state = "quiescent"
        self._turn_open = False
        self._note: Optional[str] = None
        self._suspended_on: Optional[JobId] = None
        self._last_yield: Optional[Yield] = None
        self._last_view: Tuple[InteractionItem, ...] = ()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._poke_event: Optional[asyncio.Event] = None
        self._tasks: set = set()  # run tasks, touched on the watcher loop only
        self._deliver: Optional[Callable[[], None]] = None
        self._notice: Optional[Callable[[str], None]] = None
        self._started = self._stopping = self._closed = False
        self._heartbeat = time.monotonic()
        self._close_timeout = 5.0
        self._tool: Optional[Tool] = None

    # Lifecycle (host).

    @property
    def main(self) -> ContextHandle:
        """The handle to main that code gets as ``main``."""
        return self._main

    def clock(self) -> float:
        """The current time on icron's clock (epoch seconds)."""
        return self._clock()

    def start(self, deliver: Callable[[], None],
              notice: Optional[Callable[[str], None]] = None) -> None:
        """Start the watcher thread and its event loop.

        The loop calls deliver(), without icron's lock, whenever code sends
        main a wake-up; the host then queues its delivery. notice(text)
        reports what the host should show: the output of the user's runs, a
        job's end or failure, and code abandoned at close().
        """
        if not callable(deliver) or (notice is not None and not callable(notice)):
            raise TypeError("deliver and notice must be callable.")
        with self._lock:
            if self._started or self._closed:
                raise RuntimeError("icron can be started only once.")
            self._started = True
            self._deliver, self._notice = deliver, notice
            self._loop = asyncio.new_event_loop()
        self._loop.set_exception_handler(self._loop_exception)
        ready = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, args=(ready,),
                                        name="interaction-icron", daemon=True)
        self._thread.start()
        ready.wait()

    def stop(self) -> None:
        """A stop was requested: cancel running code at its next await, and
        start no new run. Jobs stay listed until close()."""
        with self._lock:
            if self._stopping or self._closed:
                return
            self._stopping = True
            tasks = self._cancel_runs_locked()
            self._changed.notify_all()
        self._cancel_tasks(tasks)
        self._poke()

    def close(self, timeout: float = 5.0) -> Tuple[Job, ...]:
        """Stop firing, cancel running code at its next await, drop every job
        and undelivered wake-up, and join the watcher thread, so deliver()
        never runs after this returns. Code blocked without an await is
        abandoned after timeout seconds (the thread is a daemon), with a
        notice. Returns the dropped jobs. Idempotent."""
        with self._lock:
            if self._closed:
                return ()
            self._closed = self._stopping = True
            self._close_timeout = timeout
            dropped = tuple(job.snapshot() for job in self._pending_locked())
            futures = [wake.future for wake in self._wakes + self._delivered
                       if wake.future is not None] + self._handoff_waiters
            self._wakes, self._delivered, self._handoff_waiters = [], [], []
            tasks = self._cancel_runs_locked()
            self._jobs.clear()
            self._ended.clear()
            self._changed.notify_all()
        with self._callout:
            pass  # a callback in flight has returned; none starts again
        self._settle(futures, cancel=True)
        thread = self._thread
        if thread is not None:
            self._cancel_tasks(tasks)
            self._poke()
            thread.join(timeout + 1.0)
            if thread.is_alive() and self._notice is not None:
                try:
                    self._notice("icron: code that never awaited still held the watcher "
                                 "thread at exit; it was abandoned.")
                except Exception:
                    pass
        return dropped

    # The API. Main's tool calls it with by="main", the user's commands with
    # by="user", and code with its own origin. by is for host code only.

    def schedule(self, message: Optional[str] = None, *, code: Optional[Code] = None,
                 after: Optional[float] = None, at: Union[str, datetime, None] = None,
                 every: Optional[float] = None, count: Optional[int] = None,
                 by: str = "host") -> Job:
        """Schedule a job, and return it.

        The first firing is `after` seconds from this call, or at `at`: an
        aware datetime, or ISO 8601 text with a date and a time (with an
        offset or Z; without one, in the host's local zone). Give at most
        one; with neither, it is `every` seconds from this call. With every,
        the job fires again every `every` seconds: `count` firings in all
        (default: 1 without every, no limit with it). Give exactly one of
        code (run it at each firing) and message (the default code,
        await main.resume(message)).
        """
        call = "icron.schedule()"
        _check_origin(call, by)
        if (message is None) == (code is None):
            raise IcronError(f"{call}: give a message or code"
                             f"{', not both' if message is not None else ''}.")
        limits = self._limits
        if message is not None:
            _check_text(call, "message", message, limits.max_chars)
        else:
            _check_code(call, code, by)
        if after is not None and at is not None:
            raise IcronError(f"{call}: give after or at, not both.")
        if every is not None:
            every = _check_seconds(call, "every", every, limits.min_every, limits.max_delay)
        if count is not None:
            if every is None:
                raise IcronError(f"{call}: count needs every.")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise IcronError(f"{call}: count must be a positive integer.")
        now = self._clock()
        if after is not None:
            due = now + _check_seconds(call, "after", after, limits.min_delay, limits.max_delay)
        elif at is not None:
            due = _parse_at(call, at, now, limits.max_delay)
        elif every is not None:
            due = now + every
        else:
            raise IcronError(f"{call}: give after, at, or every.")
        remaining = count if count is not None else (None if every is not None else 1)
        with self._lock:
            if self._closed or self._stopping:
                raise IcronError(f"{call}: icron is stopping; nothing was scheduled.")
            self._check_room_locked(call, by)
            job_id = self._new_id_locked()
            compiled = value_name = None
            if isinstance(code, str):
                try:
                    compiled, value_name = _compile(code, job_id)
                except SyntaxError as exc:
                    raise IcronError(f"{call}: the code doesn't compile: "
                                     f"{_syntax_summary(exc)}.") from None
            job = _Job(job_id, by, "message" if message is not None else "code", now,
                       message=message, code=code, compiled=compiled, value_name=value_name,
                       due=due, every=every, remaining=remaining)
            self._jobs[job_id] = job
            snapshot = job.snapshot()
        self._poke()
        return snapshot

    def exec(self, code: Code, *, wait: float = 10.0, by: str = "host") -> ExecResult:
        """Run code now, and wait for it, at most `wait` seconds.

        Returns when the run ends, when it has sent main a wake-up or awaits
        main's handoff (main gets neither until its call returns), after
        `wait`, on the running tool call's cancel, or by its results-due time.
        A run still going goes on as a job. Fails on the watcher thread, where
        waiting would block the run.
        """
        call = "icron.exec()"
        _check_origin(call, by)
        _check_code(call, code, by)
        if isinstance(wait, bool) or not isinstance(wait, (int, float)) \
                or not math.isfinite(wait) or wait < 0:
            raise IcronError(f"{call}: wait must be a nonnegative number of seconds.")
        if by == "main" and wait > self._limits.max_wait:
            raise IcronError(f"{call}: wait must be at most {self._limits.max_wait:g} seconds.")
        if self._on_watcher_thread():
            raise IcronError(f"{call}: code can't wait for code on the watcher thread; "
                             "schedule it instead.")
        with self._lock:
            if not self._started or self._closed or self._stopping:
                raise IcronError(f"{call}: icron isn't running.")
            self._check_room_locked(call, by)
            job_id = self._new_id_locked()
            compiled = value_name = None
            if isinstance(code, str):
                try:
                    compiled, value_name = _compile(code, job_id)
                except SyntaxError as exc:
                    return ExecResult("failed", error=_format_syntax_error(exc),
                                      reason=_REASON_UNCOMPILED)
            job = _Job(job_id, by, "exec", self._clock(), code=code, compiled=compiled,
                       value_name=value_name)
            run = Run(self, job, due=None, merged=0)
            run._caller_waits = True
            job.run = run
            self._jobs[job_id] = job
        try:
            self._loop.call_soon_threadsafe(self._spawn, run)
        except RuntimeError:  # the loop closed meanwhile
            self._finish(run, "cancelled")
        return self._await_exec(run, wait)

    def cancel(self, job_id: JobId, *, by: str = "host") -> Job:
        """End a job: no more firings, its undelivered wake-up is withdrawn,
        and its running code is cancelled at its next await, unless the
        caller is that code, which goes on. Main may cancel only its own
        jobs."""
        call = "icron.cancel()"
        _check_origin(call, by)
        if not isinstance(job_id, str):
            raise IcronError(f"{call}: job_id must be a string.")
        current = _CURRENT_RUN.get()
        task = None
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not job.listed:
                if any(ended.job_id == job_id for ended in self._ended):
                    raise IcronError(f"{call}: job {job_id} has already ended.")
                raise IcronError(f"{call}: there is no job {job_id}.")
            if by == "main" and job.origin != "main":
                raise IcronError(f"{call}: job {job_id} is {_WHO[job.origin]}; you can cancel "
                                 "only your own jobs.")
            if job.cancelled:
                raise IcronError(f"{call}: job {job_id} is already cancelled.")
            own = current is not None and current._job is job
            job.due, job.remaining = None, 0
            job.cancelled = job.cancelled or not own
            futures = [wake.future for wake in self._wakes
                       if wake.job is job and wake.future is not None]
            self._wakes = [wake for wake in self._wakes if wake.job is not job]
            run = job.run
            if run is None:
                self._end_locked(job, "cancelled")
            elif not own:
                run._cancelled = True
                task = run._task
            self._release_if_idle_locked()
            snapshot = job.snapshot()
            self._changed.notify_all()
        self._settle(futures, cancel=True)
        if task is not None:
            self._cancel_tasks((task,))
        self._poke()
        return snapshot

    def jobs(self, *, by: str = "host") -> Tuple[Job, ...]:
        """Pending jobs (running ones, then the soonest first), then the few
        that ended last, latest first, after cancelled ones whose code still
        unwinds. Main sees its own and the user's."""
        _check_origin("icron.jobs()", by)
        with self._lock:
            pending = [job for job in self._pending_locked() if _visible(job, by)]
            unwinding = [job for job in self._jobs.values()
                         if job.listed and job.cancelled and _visible(job, by)]
            ended = [job for job in reversed(self._ended) if _visible(job, by)]
            return tuple(job.snapshot() for job in (*pending, *unwinding, *ended))

    def yield_(self, note: str) -> Optional[Job]:
        """End main's running turn without a final answer, once its tool batch
        is saved.

        Needs something that can resume main: a pending job, main's or the
        user's, or main's running code. Returns the job main then waits for.
        Main stays suspended until a wake-up resumes it or the user writes
        first. Only during a turn, and once per turn. The tool's action is
        "yield"; the method is yield_ because yield is a Python keyword.
        """
        call = "icron.yield_()"
        _check_text(call, "note", note, self._limits.max_chars)
        with self._lock:
            if not self._turn_open:
                raise IcronError(f"{call}: no turn is open.")
            if self._note is not None:
                raise IcronError(f"{call}: a yield is already recorded; your turn ends once "
                                 "this tool batch is saved.")
            resumer = self._resumer_locked()
            if resumer is None and not self._wakes:
                raise IcronError(f"{call}: nothing could resume you. Schedule a job first, or "
                                 "give a final answer.")
            self._note = note
            if resumer is None:
                resumer = next((wake.job for wake in self._wakes if wake.job is not None), None)
            return None if resumer is None else resumer.snapshot()

    def help(self, name: Optional[str] = None) -> str:
        """The reference that main's tool and code get, as text generated from
        this API and these limits: an overview, or one name in full (a method,
        icron, main, main.resume, Job, or Run)."""
        if name is None:
            return _help_overview(self._limits)
        if not isinstance(name, str):
            raise IcronError("icron.help(): name must be text, such as 'schedule'.")
        key = _help_key(name)
        if key is None:
            close = difflib.get_close_matches(name.strip(), _HELP_NAMES, n=1)
            raise IcronError(f"icron.help(): there is no name {name.strip()!r}"
                             + (f"; did you mean {close[0]!r}?" if close else ".")
                             + " help() lists them.")
        return _help_detail(key, self._limits)

    # Supervision (host): main as icron's coroutine.

    def begin_turn(self) -> None:
        """Main starts a turn, whatever its input: it runs, and a suspension ends."""
        with self._lock:
            self._turn_open, self._note = True, None
            self._state, self._suspended_on = "running", None

    def request(self) -> Optional[str]:
        """run_turn's control: the note that yield_ recorded in this turn."""
        with self._lock:
            return self._note

    def handoff(self, yield_: Yield, view: Sequence[InteractionItem] = ()) -> Optional[Job]:
        """Main's turn loop stopped, as yield_ says: loop.supervision's Yield,
        the record that auto's watcher receives. view is main's log as of then.

        Resolves the futures of code that waits for this handoff. After a
        yielded handoff, main is suspended on the job it waits for, which this
        returns, or released (None) if nothing is left. After any other kind,
        main is quiescent, and this returns None.
        """
        if not isinstance(yield_, Yield):
            raise TypeError("handoff() requires a Yield.")
        view = tuple(view)
        with self._lock:
            self._turn_open, self._note = False, None
            self._last_yield, self._last_view = yield_, view
            futures = [wake.future for wake in self._delivered] + self._handoff_waiters
            self._delivered, self._handoff_waiters = [], []
            waits_for = None
            self._state, self._suspended_on = "quiescent", None
            if yield_.kind == "yielded":
                resumer = self._resumer_locked()
                if resumer is None:
                    resumer = next((wake.job for wake in self._wakes if wake.job is not None),
                                   None)
                if resumer is not None or self._wakes:
                    self._state = "suspended"
                    for job in self._jobs.values():
                        job.woke_in_suspension = False
                    if resumer is not None:
                        self._suspended_on = resumer.job_id
                        waits_for = resumer.snapshot()
            self._changed.notify_all()
        self._settle(futures, yield_)
        return waits_for

    def take_wakes(self) -> Optional[Wakeup]:
        """Take every undelivered wake-up, oldest first, as one Wakeup; None if
        there is none. The host calls it where it delivers them: at an
        interrupt point of main's turn, or as the input of a new turn."""
        with self._lock:
            if not self._wakes:
                return None
            wakes, self._wakes = self._wakes, []
            resumes = self._state == "suspended"
            if resumes:  # the suspension ends with this delivery
                self._state, self._suspended_on = "running", None
            deliveries = tuple(Delivery(
                None if wake.job is None else wake.job.snapshot(), wake.origin, wake.text,
                wake.kind, None if wake.run is None else wake.run.due,
                0 if wake.run is None else wake.run.merged) for wake in wakes)
            self._delivered.extend(wake for wake in wakes if wake.future is not None)
            return Wakeup(deliveries, resumes, self._clock())

    def undelivered(self) -> int:
        """How many wake-ups wait for delivery."""
        with self._lock:
            return len(self._wakes)

    def status(self) -> IcronStatus:
        with self._lock:
            jobs = tuple(job.snapshot() for job in self._pending_locked())
            suspended_on = None
            if self._suspended_on is not None and self._suspended_on in self._jobs:
                suspended_on = self._jobs[self._suspended_on].snapshot()
            state, undelivered = self._state, len(self._wakes)
        return IcronStatus(jobs, state, suspended_on, undelivered, self._stall())

    # Main's binding.

    def tool(self) -> Tool:
        """Main's icron tool: each call is one call of this API."""
        if self._tool is None:
            spec = ToolSpec(TOOL_NAME, _description(self._limits), _PARAMETERS)
            self._tool = Tool(spec, self._handle, timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS)
        return self._tool

    @staticmethod
    def yield_recorded_in(items: Sequence[InteractionItem]) -> bool:
        """True when a raw log's last turn has a successful icron yield.

        Like YieldTool.recorded_in, but it reads the action from the call's
        arguments. The last turn is the log after its last
        UserInteractionBoundary (steers add none).
        """
        items = tuple(items)
        start = max((index + 1 for index, item in enumerate(items)
                     if isinstance(item, UserInteractionBoundary)), default=0)
        turn = items[start:]
        calls = {item.call_id for item in turn if isinstance(item, ToolCall)
                 and item.name == TOOL_NAME and _call_action(item) == "yield"}
        return any(isinstance(item, ToolResult) and item.success and item.call_id in calls
                   for item in turn)

    # The tool's handler, on main's context thread.

    def _handle(self, arguments, *, timeout_seconds=None) -> ToolOutcome:
        del timeout_seconds
        started = time.monotonic()  # exec's result reports the call's wall time
        call = _read_call(arguments)
        method = getattr(self, call.name)
        fields = dict(call.arguments)
        if "by" in inspect.signature(method).parameters:
            fields["by"] = "main"
        result = method(**fields)
        if call.name == "help":
            return ToolOutcome(result)
        now = self._clock()
        # Every result is text (doc/icron-tool-text-plan.md, part 1); exec's
        # reads as Codex's and pi's code mode report a script.
        if call.name == "exec":
            return ToolOutcome(_exec_text(result, time.monotonic() - started, now),
                               success=result.state not in ("failed", "cancelled"))
        text = {"schedule": _schedule_text, "yield_": _yield_text, "cancel": _cancel_text,
                "jobs": _jobs_text}[call.name](result, now)
        return ToolOutcome(text)

    # Internals. Methods ending in _locked run with the lock held.

    def _on_watcher_thread(self) -> bool:
        return self._thread is not None and threading.current_thread() is self._thread

    def _new_id_locked(self) -> JobId:
        self._counter += 1
        return f"{self._prefix}-{self._counter}"

    def _check_room_locked(self, call: str, by: str) -> None:
        if by == "host":
            return
        if sum(job.origin == by for job in self._jobs.values()) >= self._limits.max_jobs:
            raise IcronError(f"{call}: {'main' if by == 'main' else 'the user'} has "
                             f"{self._limits.max_jobs} pending jobs already; cancel one first.")

    def _pending_locked(self) -> List[_Job]:
        pending = [job for job in self._jobs.values() if job.listed and not job.cancelled]
        pending.sort(key=lambda job: (job.run is None,
                                      job.started if job.run is not None and job.started
                                      else math.inf,
                                      math.inf if job.due is None else job.due, job.created))
        return pending

    def _resumer_locked(self) -> Optional[_Job]:
        """The job that would resume a suspended main: main's running code,
        then the soonest scheduled firing, then any other running code. A
        cancelled job's run, still unwinding, resumes nothing."""
        live = [job for job in self._jobs.values() if job.listed and not job.cancelled]
        running = [job for job in live if job.run is not None and job.kind != "message"]
        own = [job for job in running if job.origin == "main"]
        if own:
            return own[0]
        scheduled = [job for job in live if job.due is not None]
        if scheduled:
            return min(scheduled, key=lambda job: job.due)
        return running[0] if running else None

    def _release_if_idle_locked(self) -> bool:
        """A suspended main with nothing left to resume it is released."""
        if self._state != "suspended" or self._wakes:
            return False
        resumer = self._resumer_locked()
        if resumer is None:
            self._state, self._suspended_on = "quiescent", None
            return True
        self._suspended_on = resumer.job_id
        return False

    def _cancel_runs_locked(self) -> list:
        tasks = []
        for job in self._jobs.values():
            if job.run is not None:
                job.run._cancelled = True
                if job.run._task is not None:
                    tasks.append(job.run._task)
        return tasks

    def _cancel_tasks(self, tasks) -> None:
        loop = self._loop
        for task in tasks:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # the loop has closed

    def _poke(self) -> None:
        """Have the timetable reread the clock and the jobs (thread-safe)."""
        loop, event = self._loop, self._poke_event
        if loop is None or event is None:
            return
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:
            pass  # the loop has closed

    def _call_deliver(self) -> None:
        with self._callout:
            if self._closed or self._deliver is None:
                return
            try:
                self._deliver()
            except Exception:
                pass

    def _say(self, text: str) -> None:
        with self._callout:
            if self._closed or self._notice is None:
                return
            try:
                self._notice(text)
            except Exception:
                pass

    def _settle(self, futures, value=None, *, cancel: bool = False) -> None:
        """Resolve (or cancel) futures from any thread."""
        for future in futures:
            if isinstance(future, concurrent.futures.Future):
                try:
                    if not future.done():
                        future.cancel() if cancel else future.set_result(value)
                except concurrent.futures.InvalidStateError:
                    pass
                continue
            try:
                self._loop.call_soon_threadsafe(_settle_async, future, value, cancel)
            except RuntimeError:
                pass  # the loop has closed

    def _new_future(self):
        if self._loop is not None and self._on_watcher_thread():
            return self._loop.create_future()
        return concurrent.futures.Future()

    def _send_wake(self, text: str):
        call = "main.resume()"
        _check_text(call, "text", text, self._limits.max_chars)
        run = _CURRENT_RUN.get()
        future = self._new_future()
        with self._lock:
            if self._closed or self._stopping:
                raise IcronError(f"{call}: icron is stopping; nothing was sent.")
            origin = "host" if run is None else run.origin
            if origin != "host" and sum(wake.origin == origin for wake in self._wakes) \
                    >= self._limits.max_jobs:
                raise IcronError(f"{call}: {self._limits.max_jobs} wake-ups from "
                                 f"{_WHO[origin]} code already wait for delivery.")
            job = None if run is None else run._job
            self._wakes.append(_Wake(job, run, text, "message", future, origin))
            if run is not None:
                run._sent_wake = True
                if self._state == "suspended":
                    job.woke_in_suspension = True
            self._changed.notify_all()
        self._call_deliver()
        return future

    def _next_handoff(self):
        run = _CURRENT_RUN.get()
        future = self._new_future()
        with self._lock:
            if self._closed or self._stopping:
                raise IcronError("main.handoff(): icron is stopping.")
            self._handoff_waiters.append(future)
            if run is not None:
                run._awaits_main = True
            self._changed.notify_all()
        return future

    def _namespace(self, origin: str) -> dict:
        with self._lock:
            namespace = self._namespaces.get(origin)
            if namespace is None:
                printer = self._printer(origin)
                code_icron = _CodeIcron(self, origin)
                namespace = self._namespaces[origin] = {
                    "__name__": f"__icron_{origin}__", "__builtins__": builtins,
                    "icron": code_icron, "main": self._main,
                    "asyncio": asyncio, "print": printer,
                    "help": self._helper(code_icron, printer),
                }
            return namespace

    def _printer(self, origin: str):
        def print(*args, sep=" ", end="\n", file=None, flush=False):  # noqa: A001
            """Print to this run's output (see icron)."""
            if file is not None and file is not sys.stdout:
                builtins.print(*args, sep=sep, end=end, file=file, flush=flush)
                return
            text = (" " if sep is None else sep).join(map(str, args))
            text += "\n" if end is None else end
            run = _CURRENT_RUN.get()
            if run is not None:
                run._write(text)
            elif text.strip():
                self._say(f"icron ({_WHO[origin]} code): {text.rstrip()}")
        return print

    def _helper(self, code_icron: "_CodeIcron", printer):
        def help(thing=None):  # noqa: A001
            """Print icron's reference, as the tool's help() gives it: help() for
            the overview, help('schedule') or help(main.resume) for one name.
            For any other object, Python's help, as text."""
            name = _help_name_of(thing, code_icron, self._main)
            if thing is None or name is not None:
                printer(self.help(name))
                return
            import pydoc  # only when code asks, since it's slow to import
            try:
                printer(pydoc.render_doc(thing, renderer=pydoc.plaintext))
            except ImportError:  # a name that nothing defines, as Python's help says
                printer(f"No Python documentation found for {thing!r}; help() shows icron's "
                        "reference.")
        return help

    # The watcher thread.

    def _thread_main(self, ready: threading.Event) -> None:
        loop = self._loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._watch(ready))
        except BaseException as exc:
            ready.set()
            self._say(f"icron: the watcher stopped ({type(exc).__name__}); no job runs "
                      "again.")
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except BaseException:
                pass
            loop.close()

    async def _watch(self, ready: threading.Event) -> None:
        self._poke_event = asyncio.Event()
        self._heartbeat = time.monotonic()
        beat = asyncio.ensure_future(self._beat())
        ready.set()
        try:
            while True:
                self._poke_event.clear()
                with self._lock:
                    if self._closed:
                        break
                    timeout = self._fire_due_locked()
                try:
                    await asyncio.wait_for(self._poke_event.wait(), timeout)
                except asyncio.TimeoutError:
                    pass
        finally:
            beat.cancel()
            tasks = [task for task in self._tasks if not task.done()]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.wait(tasks, timeout=self._close_timeout)

    async def _beat(self) -> None:
        while True:
            self._heartbeat = time.monotonic()
            await asyncio.sleep(_BEAT_SECONDS)

    def _stall(self) -> Optional[Tuple[str, float]]:
        """What holds the watcher loop, and for how long, past stall_seconds."""
        thread = self._thread
        if thread is None or self._closed or not thread.is_alive():
            return None
        held = time.monotonic() - self._heartbeat - _BEAT_SECONDS
        if held <= self._limits.stall_seconds:
            return None
        frame = sys._current_frames().get(thread.ident)
        job_id = "?"
        while frame is not None:
            filename = frame.f_code.co_filename
            if filename.startswith("<icron ") and filename.endswith(">"):
                job_id = filename[len("<icron "):-1]
            frame = frame.f_back
        return job_id, held

    def _fire_due_locked(self) -> float:
        """Fire the due jobs; the seconds until the next one is due."""
        if self._stopping:
            return _TICK_SECONDS
        now = self._clock()
        soonest = None
        for job in list(self._jobs.values()):
            if job.due is not None and job.due <= now:
                self._fire_locked(job, now)
            if job.due is not None:
                soonest = job.due if soonest is None else min(soonest, job.due)
        if soonest is None:
            return _TICK_SECONDS
        return min(_TICK_SECONDS, max(0.0, soonest - now))

    def _fire_locked(self, job: _Job, now: float) -> None:
        """Fire a due job: the slots due by now merge into one run, at a fixed rate."""
        slots = 1 if job.every is None else 1 + int((now - job.due) // job.every)
        if job.remaining is not None:
            slots = min(slots, job.remaining)
            job.remaining -= slots
        due = job.due if job.every is None else job.due + (slots - 1) * job.every
        if job.every is not None and (job.remaining is None or job.remaining > 0):
            job.due += slots * job.every
        else:
            job.due = None
        if job.run is not None:
            job.run.merged += slots
            return
        run = Run(self, job, due=due, merged=slots - 1)
        job.run, job.state = run, "running"
        self._begin_task_locked(run)

    def _spawn(self, run: Run) -> None:
        """On the watcher loop: start an exec call's run."""
        with self._lock:
            if run._done:
                return
            if run._cancelled or self._stopping:
                cancelled = True
            else:
                cancelled = False
                self._begin_task_locked(run)
        if cancelled:
            self._finish(run, "cancelled")

    def _begin_task_locked(self, run: Run) -> None:
        task = self._loop.create_task(self._execute(run))
        run._task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _execute(self, run: Run) -> None:
        _CURRENT_RUN.set(run)
        with self._lock:
            run._started = run._job.started = self._clock()
            self._changed.notify_all()
        try:
            value = await self._call(run)
        except asyncio.CancelledError:
            self._finish(run, "cancelled")
        except BaseException as exc:
            self._finish(run, "failed", error=_format_error(exc))
        else:
            self._finish(run, "done", value=value)

    async def _call(self, run: Run) -> Optional[str]:
        job = run._job
        if job.kind == "message":
            await self._main.resume(job.message)
            return None
        if job.compiled is None:  # a host function
            result = job.code(run)
            if inspect.isawaitable(result):
                result = await result
            return None if result is None else _clip_head(repr(result), _VALUE_CHARS)
        namespace = self._namespace(job.origin)
        result = eval(job.compiled, namespace)
        if inspect.iscoroutine(result):
            await result
        if job.value_name is None:
            return None
        value = namespace.pop(job.value_name, None)
        return None if value is None else _clip_head(repr(value), _VALUE_CHARS)

    def _finish(self, run: Run, outcome: str, *, value: Optional[str] = None,
                error: Optional[str] = None) -> None:
        """A run ended: record it, end its job if no firing is left, and
        report the end where nobody waits for it."""
        notices, deliver = [], False
        with self._lock:
            if run._done:
                return
            run._done = True
            run._outcome, run._value, run._error = outcome, value, error
            job = run._job
            if job.run is run:
                job.run = None
            job.output = run.output()
            if error is not None:
                job.error = error
            if run._stream:
                notices = self._stream_end_locked(run)
            if outcome != "done" or job.due is None or job.cancelled:
                self._end_locked(job, outcome)
                # The end of a code job goes where nobody has it yet: to a
                # suspended main that waits for it, or to a notice. A cancel's
                # caller knows of it, and a message job's message was its point.
                if not self._closed and not self._stopping and not run._stream \
                        and not run._caller_waits and job.kind != "message" \
                        and outcome != "cancelled":
                    if self._end_reaches_main_locked(job):
                        self._wakes.append(_Wake(job, run, _end_text(job, run), "end", None,
                                                 job.origin))
                        deliver = True
                    elif job.origin != "host" or outcome == "failed":
                        notices.append(_end_notice(job, run))
                self._release_if_idle_locked()
            else:
                job.state = "scheduled"
            self._changed.notify_all()
        for notice in notices:
            self._say(notice)
        if deliver:
            self._call_deliver()

    def _end_locked(self, job: _Job, outcome: str) -> None:
        job.state = "cancelled" if job.cancelled and outcome != "failed" else outcome
        job.ended = self._clock()
        job.due = None
        self._jobs.pop(job.job_id, None)
        if job.listed and not self._closed:
            self._ended.append(job)

    def _end_reaches_main_locked(self, job: _Job) -> bool:
        """While main is suspended, the end of a job of main's that sent it no
        wake-up during the suspension goes to main."""
        return (job.origin == "main" and self._state == "suspended"
                and not job.woke_in_suspension
                and not any(wake.job is job for wake in self._wakes))

    def _stream_end_locked(self, run: Run) -> List[str]:
        notices = []
        if run._partial:
            notices += run._stream_lines_locked([run._partial])
            run._partial = ""
        if run._outcome == "done":
            notices.append(f"py {run.job_id}: done" + ("." if run._value is None
                                                     else f": {run._value}"))
        elif run._outcome == "failed":
            notices.append(f"py {run.job_id}: failed:\n{_indent(run._error or '')}")
        else:
            notices.append(f"py {run.job_id}: cancelled.")
        return notices

    def _await_exec(self, run: Run, wait: float) -> ExecResult:
        deadline, limited_by = time.monotonic() + wait, "wait"
        due = current_results_due()
        if due is not None and due < deadline:
            deadline, limited_by = due, "results_due"
        cancel = current_cancel()
        job = run._job
        with self._changed:
            while True:
                if run._done:
                    # A stop (or close) marks the runs it cancels, as cancel()
                    # does; code that raised CancelledError itself ended by itself.
                    stopped = run._outcome == "cancelled" and run._cancelled and self._stopping
                    return ExecResult(run._outcome, run.output(), run._value, run._error,
                                      job.snapshot(), _REASON_STOPPED if stopped else None)
                left = deadline - time.monotonic()
                reason = None
                if run._sent_wake:
                    reason = _REASON_MESSAGE
                elif run._awaits_main:
                    reason = _REASON_HANDOFF
                elif cancel is not None and cancel.cancelled:
                    reason = _REASON_INTERRUPTED
                elif left <= 0:
                    reason = _REASON_DEADLINE if limited_by == "results_due" else _REASON_WAIT
                if reason is not None:
                    run._caller_waits = False
                    job.listed = True
                    # Not started: the watcher is busy with other code. The
                    # reason stays the one the call returned for.
                    state = "running" if run._started is not None else "pending"
                    return ExecResult(state, run.output(), job=job.snapshot(), reason=reason)
                self._changed.wait(left if cancel is None else min(left, 0.05))

    def _loop_exception(self, loop, context) -> None:
        if self._closed:
            return
        exception = context.get("exception")
        detail = "" if exception is None else f" ({type(exception).__name__})"
        self._say(f"icron: {context.get('message', 'error in the watcher loop')}{detail}")


def log_state(items: Sequence[InteractionItem], *, now: Optional[float] = None) -> Optional[str]:
    """What a resumed raw log says about icron, which a restart loses.

    "yielded": its last turn ended with an icron yield, which no wake-up will
    resume. "jobs": it ran code with exec, or scheduled a job that could still
    be pending (a repeat, or a later due time). None otherwise.
    """
    items = tuple(items)
    if Icron.yield_recorded_in(items):
        return "yielded"
    now = time.time() if now is None else now
    calls = {item.call_id: item for item in items
             if isinstance(item, ToolCall) and item.name == TOOL_NAME}
    for item in items:
        if not (isinstance(item, ToolResult) and item.success and item.call_id in calls):
            continue
        action = _call_action(calls[item.call_id])
        if action == "exec":
            return "jobs"
        if action != "schedule":
            continue
        scheduled = _SCHEDULED.match(item.output)
        if scheduled is not None:  # text, since October 25
            try:
                if scheduled["repeat"] or _timestamp(scheduled["due"]) > now:
                    return "jobs"
            except ValueError:
                pass
            continue
        try:
            data = json.loads(item.output)
            if data.get("every") is not None:
                return "jobs"
            if datetime.fromisoformat(data["due"]).timestamp() > now:
                return "jobs"
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
    return None


# A schedule result (_schedule_text): its due time, and whether it repeats.
_SCHEDULED = re.compile(r"Scheduled job \S+ \([^;]*; due (?P<due>[^,]+), in \d+ s"
                        r"(?P<repeat>, every )?")


def _timestamp(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


# Helpers.

def _settle_async(future, value, cancel: bool) -> None:
    if not future.done():
        if cancel:
            future.cancel()
        else:
            future.set_result(value)


def _visible(job: _Job, by: str) -> bool:
    return by != "main" or job.origin in ("main", "user")


def _check_origin(call: str, by: str) -> None:
    if by not in ORIGINS:
        raise IcronError(f"{call}: by must be one of {', '.join(ORIGINS)}.")


def _check_text(call: str, name: str, value, limit: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise IcronError(f"{call}: {name} must be nonblank text.")
    if len(value) > limit:
        raise IcronError(f"{call}: {name} must be at most {limit} characters.")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise IcronError(f"{call}: {name} must be valid UTF-8 text.") from None


def _check_code(call: str, code, by: str) -> None:
    if callable(code) and not isinstance(code, str):
        if by != "host":
            raise IcronError(f"{call}: code must be Python source.")
        return
    if not isinstance(code, str) or not code.strip():
        raise IcronError(f"{call}: code must be nonblank Python source.")
    try:
        code.encode("utf-8")
    except UnicodeError:
        raise IcronError(f"{call}: code must be valid UTF-8 text.") from None


def _check_seconds(call: str, name: str, value, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise IcronError(f"{call}: {name} must be a number of seconds.")
    if not low <= value <= high:
        raise IcronError(f"{call}: {name} must be from {low:g} to {high:g} seconds.")
    return float(value)


_DATE_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


def _parse_at(call: str, value, now: float, horizon: float) -> float:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise IcronError(f"{call}: at must be an aware datetime.")
        when = value
    elif isinstance(value, str):
        text = value.strip()
        if text[-1:] in ("Z", "z"):
            text = text[:-1] + "+00:00"  # fromisoformat takes Z only from Python 3.11
        try:
            if not _DATE_TIME.match(text):
                raise ValueError(text)
            when = datetime.fromisoformat(text)
        except ValueError:
            raise IcronError(f"{call}: at must be an ISO 8601 date and time, such as "
                             "2026-10-18T17:30:00+02:00.") from None
        if when.tzinfo is None:
            when = when.astimezone()  # the host's local zone
    else:
        raise IcronError(f"{call}: at must be ISO 8601 text or an aware datetime.")
    due = when.timestamp()
    if due < now - 1:
        raise IcronError(f"{call}: at is in the past (now {_iso(now)}).")
    if due - now > horizon:
        raise IcronError(f"{call}: at must be at most {horizon:g} seconds ahead "
                         f"(now {_iso(now)}).")
    return due


def _compile(source: str, job_id: JobId):
    """Compile code for a job, with top-level await; raises SyntaxError.

    A final expression is assigned to a hidden name, whose value is the run's
    value. The source is registered with linecache, so tracebacks show it.
    """
    filename = f"<icron {job_id}>"
    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    tree = compile(source, filename, "exec", flags=ast.PyCF_ONLY_AST | flags, dont_inherit=True)
    value_name = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        value_name = "_icron_value_" + job_id.replace("-", "_")
        last = tree.body[-1]
        tree.body[-1] = ast.copy_location(
            ast.Assign(targets=[ast.Name(id=value_name, ctx=ast.Store())], value=last.value),
            last)
        ast.fix_missing_locations(tree)
    code = compile(tree, filename, "exec", flags=flags, dont_inherit=True)
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    return code, value_name


def _syntax_summary(exc: SyntaxError) -> str:
    line = "" if exc.lineno is None else f" (line {exc.lineno})"
    return f"{exc.msg}{line}"


def _format_syntax_error(exc: SyntaxError) -> str:
    return _clip_tail("".join(traceback.format_exception_only(type(exc), exc)).rstrip(),
                      _ERROR_CHARS)


def _format_error(exc: BaseException) -> str:
    """The traceback from the code's first frame on, without icron's frames."""
    tb = exc.__traceback__
    while tb is not None and not tb.tb_frame.f_code.co_filename.startswith("<icron "):
        tb = tb.tb_next
    if tb is None:
        tb = exc.__traceback__
    text = "".join(traceback.format_exception(type(exc), exc, tb)).rstrip()
    return _clip_tail(text, _ERROR_CHARS)


def _clip_head(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [{len(text) - limit} more characters]"


def _clip_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"[{len(text) - limit} earlier characters omitted]\n" + text[-limit:]


def _indent(text: str) -> str:
    return "\n".join("  " + line for line in text.splitlines())


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds).astimezone().isoformat(timespec="seconds")


def _hms(seconds: float) -> str:
    return datetime.fromtimestamp(seconds).strftime("%H:%M:%S")


def _call_action(call: ToolCall) -> Optional[str]:
    """A saved call's action, read as the tool reads it ("yield" for yield_),
    in any form; None if the tool would refuse it."""
    try:
        return _read_call(json.loads(call.arguments_json)).action
    except ValueError:  # not JSON, or an IcronError
        return None


def call_text(arguments) -> Optional[Tuple[str, Optional[str]]]:
    """A tool input as icron reads it, for transcripts: the call in canonical
    form, such as icron.schedule(message='Hi.', after=600), and its code. Each
    form of a call reads the same. None if the tool would refuse it."""
    try:
        call = _read_call(arguments)
    except IcronError:
        return None
    return call.text(), call.arguments.get("code")


# Main's tool input (doc/icron-tool-text-plan.md, part 2): one method call in
# Python, {"call": "schedule('Hi.', after=600)"}, with code in its own field,
# {"call": "exec(wait=30)", "code": "…"}. Also read, since each means only one
# call: code alone (exec), an "icron." prefix, yield(…) for yield_(…), and the
# first version's {"action": "schedule", "message": "Hi.", "after": 600}.
# Forms mixed together are refused. Nothing in a call is evaluated: its values
# are literals, or numbers combined with + - * /.

_METHODS = ("exec", "schedule", "yield_", "cancel", "jobs", "help")  # what main's tool calls
_TAKES_CODE = ("exec", "schedule")
_NO_CALL = "icron: give call, one method call such as jobs(), or code to run now."


@dataclass(frozen=True)
class _Call:
    """A tool input as icron reads it: the method, and its arguments by name
    in the signature's order, code included."""

    name: str
    arguments: Dict[str, object]

    @property
    def action(self) -> str:
        """The first version's action: the method, but "yield" for yield_."""
        return "yield" if self.name == "yield_" else self.name

    def text(self) -> str:
        """The call without its code: icron.schedule(message='Hi.', after=600)."""
        keywords = ", ".join(f"{key}={value!r}" for key, value in self.arguments.items()
                             if key != "code")
        return f"icron.{self.name}({keywords})"


def _read_call(arguments) -> _Call:
    """The call that a tool input stands for. An input that could mean two
    things, or none, fails with an IcronError that says what to give."""
    if not isinstance(arguments, Mapping):
        raise IcronError(_NO_CALL)
    fields = dict(arguments)
    if "action" in fields:
        if "call" in fields:
            raise IcronError("icron: give call or action, not both.")
        return _read_action(fields)
    code, text = fields.pop("code", None), fields.pop("call", None)
    if fields:
        raise IcronError(f"icron: unexpected field {sorted(fields)[0]!r}; give call, and code "
                         "for exec() or schedule().")
    if text is None:
        if code is None:
            raise IcronError(_NO_CALL)
        text = "exec()"  # code alone runs now: schedule needs a time
    if not isinstance(text, str):
        raise IcronError("icron: call must be text, one method call such as jobs().")
    source = re.sub(r"^icron\s*\.\s*", "", text.strip())  # as transcripts show calls
    source = re.sub(r"^yield\s*\(", "yield_(", source)  # yield is a Python keyword
    try:
        node = ast.parse(source, mode="eval").body
    except SyntaxError as exc:
        raise IcronError(f"icron: call must be one method call, such as jobs(); "
                         f"{exc.msg}.") from None
    except (ValueError, RecursionError, MemoryError):  # a null character; deep nesting
        node = None
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
        raise IcronError("icron: call must be one method call, such as jobs().")
    name = node.func.id
    if name not in _METHODS:
        close = difflib.get_close_matches(name, _METHODS, n=1)
        raise IcronError(f"icron: there is no method {name}()"
                         + (f"; did you mean {close[0]}()?" if close else "."))
    call = f"icron.{name}()"
    positional = [parameter.name for parameter
                  in inspect.signature(getattr(Icron, name)).parameters.values()
                  if parameter.kind is parameter.POSITIONAL_OR_KEYWORD][1:]  # after self
    args = []
    for index, value in enumerate(node.args):
        if isinstance(value, ast.Starred):
            raise IcronError(f"{call}: give each argument by itself, not *sequence.")
        what = positional[index] if index < len(positional) else f"argument {index + 1}"
        args.append(_literal(value, call, what))
    keywords = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise IcronError(f"{call}: give each argument by itself, not **mapping.")
        keywords[keyword.arg] = _literal(keyword.value, call, keyword.arg)
    if code is not None:
        if name not in _TAKES_CODE:
            raise IcronError(f"{call}: it takes no code; leave the code field out.")
        if "code" in keywords or (name == "exec" and args):
            raise IcronError(f"{call}: give the code in the code field only.")
        keywords["code"] = code
    elif name == "exec" and not args and "code" not in keywords:
        raise IcronError(f"{call}: give the code to run in the code field.")
    return _bind(name, args, keywords)


def _read_action(fields: Dict[str, object]) -> _Call:
    """The first version's form: {"action": "schedule", "message": "Hi.", …}."""
    action = fields.pop("action")
    if action not in ACTIONS:
        raise IcronError(f"icron: action must be one of {', '.join(ACTIONS)}.")
    return _bind("yield_" if action == "yield" else action, [], fields)


def _bind(name: str, args: List[object], keywords: Dict[str, object]) -> _Call:
    """Bind a call's values to its method's signature, so that Python's own
    messages name a missing or unexpected argument."""
    call = f"icron.{name}()"
    if "by" in keywords:
        raise IcronError(f"{call}: by is for host code only.")
    try:
        bound = inspect.signature(getattr(Icron, name)).bind(None, *args, **keywords)
    except TypeError as exc:
        raise IcronError(f"{call}: {exc}") from None
    arguments = dict(bound.arguments)
    del arguments["self"]
    if "code" in arguments and not isinstance(arguments["code"], str):
        raise IcronError(f"{call}: code must be a string.")
    return _Call(name, arguments)


def _literal(node: ast.expr, call: str, what: str):
    """A value in a call: a literal, or numbers combined with + - * /."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    try:
        value = _number(node)
    except RecursionError:  # 1+1+…+1, thousands of terms long
        value = None
    if value is None:
        raise IcronError(f"{call}: {what} must be a literal, such as 600 or 'text'.")
    return value


def _number(node: ast.expr) -> Union[int, float, None]:
    """Numbers combined with + - * /, such as 10*60; None for anything else."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _number(node.operand)
        if value is None:
            return None
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = _number(node.left), _number(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return None if right == 0 else left / right
    return None


def _delivery_label(delivery: Delivery) -> str:
    job = delivery.job
    if job is None:
        return "From the host"
    if job.kind == "exec":
        text = f"{_whose(job)} at {_hms(job.started or job.created)}"
    else:
        text = _whose(job)
        if delivery.due is not None:
            text += f"; due {_iso(delivery.due)}"
            if job.every is not None:
                text += f", every {job.every:g} s"
        if delivery.merged:
            text += (f"; {delivery.merged} earlier firing"
                     f"{'s' if delivery.merged > 1 else ''} merged")
    return f"Job {job.job_id} ({text})"


def _whose(job) -> str:
    """Whose a job is, as wake-ups and results say it to main: "yours", "your
    code", "the user's, from /cron", "your code, from exec", and so on."""
    code = {"main": "your code", "user": "the user's code", "host": "the host's code"}[job.origin]
    if job.kind == "exec":
        return f"{code}, from {'/py' if job.origin == 'user' else 'exec'}"
    if job.kind == "message":
        whose = {"main": "yours", "user": "the user's", "host": "the host's"}[job.origin]
        return whose + (", from /cron" if job.origin == "user" else "")
    return code


def _end_text(job: _Job, run: Run) -> str:
    """A wake-up's text for the end of one of main's jobs."""
    if run._outcome == "failed":
        return f"failed:\n{run._error}"
    text = "ended without a message for you."
    output = run.output()
    text += f" Its last output:\n{output.rstrip()}" if output.strip() else " It printed nothing."
    if run._value is not None:
        text += f"\nIts value: {run._value}"
    return text


def _end_notice(job: _Job, run: Run) -> str:
    head = f"icron: job {job.job_id} ({_WHO[job.origin]} code)"
    if run._outcome == "failed":
        last = (run._error or "").strip().splitlines()[-1:] or ["?"]
        return f"{head} failed: {_clip_head(last[0], 300)}"
    return f"{head} finished."


# Results as main reads them (doc/icron-tool-text-plan.md, part 1): a status
# line, single-line facts as "Name: value", and a line that ends in a colon
# opens a block of raw text, indented two spaces deeper, so nothing inside a
# block reads as a label. Times are ISO 8601, and spans are in seconds.

_EXEC_ENDED = {"done": "Script completed", "failed": "Script failed",
               "cancelled": "Script cancelled"}


def _block(title: str, text: str, depth: int = 0) -> List[str]:
    """A titled block of raw text, each line two spaces deeper than its
    title; nothing when the text is empty."""
    text = text.rstrip("\n")
    if not text:
        return []
    pad = "  " * depth
    return [f"{pad}{title}:"] + [f"{pad}  {line}" if line else "" for line in text.split("\n")]


def _job_ref(job: Job, now: float) -> str:
    """A job's ID and label, in the style of wake-ups': ID (whose; when)."""
    if job.state == "scheduled" and job.due is not None:
        when = f"due {_iso(job.due)}, in {max(0, round(job.due - now))} s"
        if job.every is not None:
            when += f", every {job.every:g} s"
            if job.remaining is not None:
                when += f", {job.remaining} firing{'s' if job.remaining != 1 else ''} left"
    elif job.state == "running":
        if job.started is None:
            when = "waiting to start"
        else:
            when = f"{'sent' if job.kind == 'message' else 'running since'} {_iso(job.started)}"
        if job.due is not None:
            when += f", next due {_iso(job.due)}"
    else:
        when = job.state if job.ended is None else f"{job.state} at {_iso(job.ended)}"
    return f"{job.job_id} ({_whose(job)}; {when})"


def _exec_text(result: ExecResult, seconds: float, now: float) -> str:
    """exec's result, in the wording of Codex's and pi's code mode.

    The first line is the run's state: "Script completed", "Script failed", or
    "Script cancelled"; for a run that goes on as a job, "Script running with
    job ID …", or "Script pending with job ID … until icron's event loop is
    free" when the watcher hasn't started it. Then the call's wall time, with
    the reason in parentheses when the code didn't run to its end within the
    call; the time; a final expression's value; what the run printed, as an
    "Output:" block ("Output so far:" for a job); and its error, as an
    "Error:" block.
    """
    job_id = None if result.job is None else result.job.job_id
    if result.state == "running":
        status = f"Script running with job ID {job_id}"
    elif result.state == "pending":
        status = f"Script pending with job ID {job_id} until icron's event loop is free"
    else:
        status = _EXEC_ENDED[result.state]
    wall = f"Wall time {seconds:.1f} seconds"
    if result.reason is not None:
        wall += f" ({result.reason})"
    lines = [status, wall, f"Now: {_iso(now)}"]
    if result.value is not None:  # a repr: a block when it takes several lines
        lines += (_block("Value", result.value) if "\n" in result.value
                  else [f"Value: {result.value}"])
    lines += _block("Output so far" if result.state in ("running", "pending") else "Output",
                    result.output)
    if result.error is not None:
        lines += _block("Error", result.error)
    return "\n".join(lines)


def _schedule_text(job: Job, now: float) -> str:
    """schedule's result: the job, without its message or code, which main
    just wrote."""
    return f"Scheduled job {_job_ref(job, now)}\nNow: {_iso(now)}"


def _yield_text(job: Optional[Job], now: float) -> str:
    """yield_'s result: when the turn ends, and what resumes main.

    The job is yield_'s: main's running code, the soonest firing, or other
    running code; failing those, the job of a wake-up that already waits (a
    message job's message, or the end of a job), or None when that wake-up
    is host code's."""
    if job is None:
        how = "A wake-up already waits for you."
    elif job.due is not None and (job.state == "scheduled" or job.kind == "message"):
        how = f"Job {_job_ref(job, now)} resumes you when it fires, unless the user writes first."
    elif job.kind == "message" or not job.pending:
        how = f"A wake-up from job {_job_ref(job, now)} already waits for you."
    else:
        how = (f"Job {_job_ref(job, now)} resumes you when it sends you a message or ends, "
               "unless the user writes first.")
    return f"Yielded: your turn ends once this tool batch is saved\n{how}\nNow: {_iso(now)}"


def _cancel_text(job: Job, now: float) -> str:
    """cancel's result: the job, and, for code that was running, when it stops."""
    text = f"Cancelled job {job.job_id} ({_whose(job)})"
    if job.kind != "message" and job.started is not None and job.ended is None:
        text += ": its code stops at its next await"
    return f"{text}\nNow: {_iso(now)}"


def _jobs_text(jobs: Sequence[Job], now: float) -> str:
    """jobs()'s result: the counts, then each job with its message or code,
    latest output, and error as blocks."""
    pending = sum(job.pending for job in jobs)
    if not jobs:
        head = "No jobs"
    else:
        head = (f"{pending} pending job{'s' if pending != 1 else ''}" if pending
                else "No pending jobs")
        if len(jobs) > pending:
            head += f", and {len(jobs) - pending} that ended"
    lines = [head, f"Now: {_iso(now)}"]
    for job in jobs:
        lines.append(f"Job {_job_ref(job, now)}:")
        if job.message is not None:
            lines += _block("Message", job.message, 1)
        elif job.code is not None:
            lines += _block("Code", _clip_head(job.code, _LISTED_CODE_CHARS), 1)
        lines += _block("Output so far" if job.state == "running" else "Output",
                        _clip_tail(job.output, _LISTED_OUTPUT_CHARS), 1)
        if job.error is not None:
            lines += _block("Error", job.error, 1)
    return "\n".join(lines)


# The tool's description and help() (doc/icron-tool-text-plan.md, part 3),
# generated from the API: each method's line has its signature from inspect,
# without self, the internal by, and code, which goes in its own field, and its
# summary below; help() adds each name's details and a result rendered by the
# formatters above. Bounds come from the instance's IcronLimits. A keyword
# added to a method appears in both on its own; its sentence under Arguments
# in _DETAILS is the one edit, and a test checks that each keyword has one.

_SUMMARIES = {
    "exec": ("run the code field now, in this process, on icron's event loop; top-level await "
             "works. Returns what it printed, its last expression's value, or the traceback; if "
             "it still runs after wait seconds (at most {max_wait}), it goes on as a job."),
    "schedule": ("a job that sends you message, or runs the code field. It first fires after "
                 "`after` seconds, or at `at` (ISO 8601 with a UTC offset), or else in `every` "
                 "seconds; with every it repeats, count times in all (default: no limit). A job "
                 "runs once at a time: a message's run lasts until the turn that receives it "
                 "ends, and firings meanwhile merge into it. Write a message that makes sense on "
                 "its own; a compaction may have removed this call by then."),
    "yield_": ("end your turn without a final answer, once this tool batch is saved; note says "
               "what you're waiting for. It needs a pending job, yours or the user's, or your "
               "running code: the next message from icron resumes you, unless the user writes "
               "first."),
    "cancel": "end one of your jobs, and stop its running code.",
    "jobs": ("your jobs and the user's (the user schedules messages with /cron), with the "
             "current time and each job's latest output."),
    "help": ("the reference: every keyword, its limits, the results, and what code sees; "
             "help('schedule') shows one name in full."),
}

_DESCRIPTION_HEAD = (
    "Run Python, schedule work, and pause your turn, through icron: this session's "
    "in-process scheduler. call is one call of a method below, in Python with literal "
    "arguments; code holds the Python source for exec or for a code job, and code alone runs "
    "it now, as exec(). For example:\n"
    '  {"call": "schedule(\'Check CI run 123.\', after=600)"}\n'
    '  {"code": "import os\\nprint(os.getloadavg())"}\n'
    '  {"call": "schedule(every=600)", "code": "await check_credit()"}'
)

_DESCRIPTION_TAIL = (
    "Code sees icron (schedule, cancel, jobs, now, current), main (main.resume(text) sends "
    "you text as a message, and awaiting it waits for the end of the turn that receives it), "
    "asyncio, print and help. Variables persist between calls, so a code job can use a "
    "function that exec defined. Use asyncio.to_thread for blocking work.\n"
    "Messages from icron reach you as text typed by the user would: during a turn, right "
    "before your next sample, once the current tool batch is done; between turns, they start "
    "one. Jobs and Python state exist only while this process runs. When the task is done, "
    "give a final answer instead of yielding; if you need something only the user can "
    "provide, ask in your final answer."
)

_HELP_INTRO = (
    "icron: this session's in-process scheduler. A tool call is one call of a method below, "
    "in call: Python with literal arguments, where numbers may be combined with + - * /. "
    "code holds the Python source for exec or for a code job, and code alone runs it now, as "
    "exec(). help('schedule') shows one name in full: a method, or icron, main, main.resume, "
    "Job, or Run."
)

_HELP_REST = (
    "In code:\n"
    "  icron: schedule, cancel, and jobs, which return Job snapshots; now(), the time; "
    "current, the Run of the code that asks.\n"
    "  main: you. main.resume(text) sends you text as a message, and main.handoff() waits "
    "for the end of your next turn.\n"
    "  asyncio; print, which writes to the run's output; and help, which prints this "
    "reference.\n"
    "  Top-level await works, and a final expression is exec's value. Variables persist "
    "between calls in your namespace, so a code job can use a function that exec defined. "
    "All code runs on icron's one event loop, so use asyncio.to_thread for blocking work: "
    "while code blocks the loop, no job fires.\n"
    "Messages:\n"
    "  Messages from icron reach you as text typed by the user would: during a turn, right "
    "before your next sample, once the current tool batch is done; between turns, they start "
    "one, and after a yield they resume you. Each names the job it comes from.\n"
    "Limits:\n"
    "  {max_jobs} pending jobs and {max_jobs} undelivered messages each, for you and for the "
    "user; after from {min_delay} to {max_delay} seconds ({max_days}); `at` up to {max_days} "
    "ahead; every from {min_every} to {max_delay} seconds; message, note, and main.resume's "
    "text up to {max_chars} characters; exec's wait up to {max_wait} seconds; a run's "
    "output, its last {max_output} characters.\n"
    "Results:\n"
    "  Every result is text: a first line that says what happened, facts as Name: value, and "
    "blocks of raw text (code, output, errors, messages) under a line that ends in a colon, "
    "indented two spaces deeper. Times are ISO 8601, and spans are in seconds."
)

# Names that help() shows besides the methods: (title, summary).
_HELP_TITLES = {
    "icron": ("icron, in code", "The icron that code sees: what it schedules is yours."),
    "main": ("main, in code", "You, as code sees you: the context that icron supervises."),
    "main.resume": ("main.resume(text), in code",
                    "Send you text as a message from icron, now. It returns a future for the "
                    "end of the turn that receives it."),
    "Job": ("Job", "A snapshot of one job, as code's icron.schedule(), icron.cancel() and "
                   "icron.jobs() return it: never live state."),
    "Run": ("Run", "One run of a job's code, or of exec's: icron.current, in code."),
}

# Each name's details for help(name): text with the bounds' placeholders, and
# examples of tool inputs, or of code.
_DETAILS = {
    "exec": (
        "Arguments:\n"
        "  code: the code field: Python source. Its final expression, if any, is the run's "
        "value.\n"
        "  wait: seconds the call waits for the code, from 0 to {max_wait} (default 10).\n"
        "The call returns when the code ends, sends you a message (you get it after this tool "
        "batch), or awaits the end of your turn; when wait runs out; when the user interrupts "
        "the wait; or at the call's deadline. Code still running then goes on as a job, which "
        "jobs() lists and cancel() ends; if you yield meanwhile, its next message or its end "
        "resumes you.\n"
        "The result's first line is Script completed, Script failed, Script cancelled, Script "
        "running with job ID …, or Script pending with job ID … until icron's event loop is "
        "free. Then the call's wall time, with the reason in parentheses when the code didn't "
        "run to its end; Now:, the time; Value:, the final expression's repr, unless it is "
        "None; and blocks: Output: (Output so far: for a job), the last {max_output} "
        "characters that it printed, and Error:, the traceback. A syntax error fails the call "
        "at once (it didn't run).",
        ('{"code": "print(sum(range(10)))"}',
         '{"call": "exec(wait=30)", "code": "for step in (1, 2):\\n    print(\'step\', step)\\n'
         '    await asyncio.sleep(20)"}')),
    "schedule": (
        "Arguments:\n"
        "  message: the text it sends you, nonblank, up to {max_chars} characters. Its run is "
        "await main.resume(message). Give a message or the code field, not both.\n"
        "  code: the code field: Python source, run at each firing in your namespace, as exec "
        "runs it. It compiles now, so a syntax error refuses the call.\n"
        "  after: seconds from now to the first firing, from {min_delay} to {max_delay} "
        "({max_days}).\n"
        "  at: the first firing's time, in ISO 8601 with a date, a time, and an offset or Z "
        "(without one, the host's local time), at most {max_days} ahead. Give after or at, not "
        "both; with neither, the first firing is every seconds from now.\n"
        "  every: seconds between firings, from {min_every} to {max_delay}; it repeats until "
        "cancelled, or count times.\n"
        "  count: how many firings in all, with every. Default: no limit.\n"
        "A failed run ends its job. You can have {max_jobs} pending jobs.",
        ('{"call": "schedule(\'Check CI run 123.\', after=600)"}',
         '{"call": "schedule(\'Post the summary.\', at=\'2026-10-25T17:00:00+00:00\')"}',
         '{"call": "schedule(every=60, count=3)", "code": "await check_credit()"}')),
    "yield_": (
        "Arguments:\n"
        "  note: what you're waiting for, nonblank, up to {max_chars} characters.\n"
        "Once per turn. It needs something that can resume you: a pending job, yours or the "
        "user's, your running code, or a message that already waits. The next message from "
        "icron resumes you: a job's message, a message from code, or the end of a job of yours "
        "that sent you none while you waited. If the user writes first, their message starts "
        "your next turn instead.\n"
        "Write yield_, since yield is a Python keyword; yield('…') is read as yield_('…').",
        ('{"call": "yield_(\'Waiting for CI run 123.\')"}',)),
    "cancel": (
        "Arguments:\n"
        "  job_id: the job's ID, from schedule's, exec's or jobs()'s result.\n"
        "The job fires no more, a message of its that waits for delivery is withdrawn, and its "
        "running code is cancelled at its next await. You can cancel only your own jobs; the "
        "user cancels theirs with /cron cancel.",
        ('{"call": "cancel(\'k3f9-3\')"}',)),
    "jobs": (
        "Running jobs come first, then the soonest due, then up to {ended_kept} that ended, "
        "latest first. Each job's line says whose it is (yours, your code, the user's, from "
        "/cron) and when it fires or ran; under it are blocks: its message or code (the first "
        "{listed_code} characters), its latest run's output (the last {listed_output}), and a "
        "failed run's error.",
        ('{"call": "jobs()"}',)),
    "help": (
        "Arguments:\n"
        "  name: a method (exec, schedule, yield_, cancel, jobs, help), or a name that code "
        "sees (icron, main, main.resume, Job, Run). Without it, the overview.\n"
        "In code, help() prints the same text to the run's output, and help(obj) of any other "
        "object prints Python's help for it.",
        ('{"call": "help(\'schedule\')"}', '{"code": "help(main.resume)"}')),
    "icron": (
        "  {code_schedule}: as the tool's schedule, with code as Python source; it returns a "
        "Job.\n"
        "  {code_cancel}: it returns the Job. Code that cancels its own job goes on running.\n"
        "  icron.jobs(): your jobs and the user's, as a tuple of Jobs.\n"
        "  icron.now(): the current time, as an aware local datetime.\n"
        "  icron.current: the Run of the code that asks; None outside a run.\n"
        "exec and yield_ are for tool calls only.",
        ("job = icron.schedule('Check the deploy.', after=900)\nprint(job.job_id, job.due)",)),
    "main": (
        "  main.resume(text): send you text as a message now; help('main.resume').\n"
        "  main.handoff(): a future for the end of your next turn, however it ends.\n"
        "  main.state: \"running\", \"quiescent\", or \"suspended\" (your last turn yielded).\n"
        "  main.last(): your latest handoff: its kind (\"ended\", \"yielded\", \"stopped\", or "
        "\"failed\"), final_text, and yield_text.\n"
        "  main.view(): your log as of that handoff, as interaction items.",
        ("await main.handoff()\nprint(main.last().kind)",)),
    "main.resume": (
        "Awaiting it is optional: unawaited, resume only sends. During your turn, the message "
        "arrives right before your next sample, once the current tool batch is done; between "
        "turns, it starts one, and after a yield it resumes you.\n"
        "Arguments:\n"
        "  text: nonblank, up to {max_chars} characters. Up to {max_jobs} messages from your "
        "code wait for delivery at a time.",
        ("await main.resume(f'CI run 123 finished: {status}')",)),
    "Job": (
        "  job_id; origin: \"main\" (yours), \"user\", or \"host\"; kind: \"message\", "
        "\"code\", or \"exec\".\n"
        "  state: \"scheduled\", \"running\", \"done\", \"failed\", or \"cancelled\"; pending: "
        "scheduled or running.\n"
        "  due (the next firing), created, started (the latest run's start), and ended: epoch "
        "seconds, or None.\n"
        "  message or code; every: seconds between firings, or None (once); remaining: "
        "firings still to come, or None (no limit).\n"
        "  output: the latest run's output; error: a failed run's traceback.",
        ()),
    "Run": (
        "  job_id and origin, as its Job's.\n"
        "  due: the firing that started it (None for exec); merged: firings that merged into "
        "it, missed before it started or due while it ran.\n"
        "  output(): what it has printed so far, its last {max_output} characters.",
        ()),
}

_HELP_NAMES = (*_METHODS, *_HELP_TITLES)
# Other ways to write a name, as help() reads them.
_HELP_ALIASES = {"yield": "yield_", "now": "icron", "current": "icron",
                 "main.handoff": "main", "main.state": "main", "main.last": "main",
                 "main.view": "main"}
# The time of help's sample results: fixed, so that the text is the same on
# every call.
_SAMPLE_NOW = datetime(2026, 10, 25, 7, 43, tzinfo=timezone.utc).timestamp()


def _description(limits: IcronLimits) -> str:
    """The tool's description: the call format, a line per method, what code
    sees, and how messages arrive."""
    bounds = _bounds(limits)
    methods = "\n".join(f"- {_signature_text(Icron, name)}: {_SUMMARIES[name].format(**bounds)}"
                        for name in _METHODS)
    return f"{_DESCRIPTION_HEAD}\n{methods}\n{_DESCRIPTION_TAIL}"


def _help_overview(limits: IcronLimits) -> str:
    bounds = _bounds(limits)
    lines = [_HELP_INTRO, "Methods:"]
    for name in _METHODS:
        lines += [f"  {_signature_text(Icron, name)}",
                  f"    {_sentence(_SUMMARIES[name].format(**bounds))}"]
    lines.append(_HELP_REST.format(**bounds))
    return "\n".join(lines)


def _help_detail(key: str, limits: IcronLimits) -> str:
    bounds = _bounds(limits)
    if key in _METHODS:
        title = _signature_text(Icron, key)
        summary = _sentence(_SUMMARIES[key].format(**bounds))
    else:
        title, summary = _HELP_TITLES[key]
    body, examples = _DETAILS[key]
    lines = [title, f"  {summary}", body.format(**bounds)]
    sample = _help_samples().get(key)
    if sample is not None:
        lines += _block("Result, for example", sample)
    if examples:
        lines += _block("Example" if len(examples) == 1 else "Examples", "\n".join(examples))
    return "\n".join(lines)


def _help_key(name: str) -> Optional[str]:
    """A name as help() reads it: also icron.schedule, schedule(), or yield."""
    key = name.strip()
    key = key[:-2] if key.endswith("()") else key
    if key.startswith("icron.") and key != "icron.":
        key = key[len("icron."):]
    key = _HELP_ALIASES.get(key, key)
    return key if key in _HELP_NAMES else None


def _help_name_of(thing, code_icron: "_CodeIcron", main: "ContextHandle") -> Optional[str]:
    """The name that code's help(thing) stands for, or None for Python's help."""
    if isinstance(thing, str):
        return _help_key(thing)
    if thing is code_icron:
        return "icron"
    if thing is main:
        return "main"
    owner, method = getattr(thing, "__self__", None), getattr(thing, "__name__", None)
    if owner is code_icron:
        return method if method in ("schedule", "cancel", "jobs") else "icron"
    if owner is main:
        return "main.resume" if method == "resume" else "main"
    for kind in (Job, Run):
        if thing is kind or isinstance(thing, kind):
            return kind.__name__
    return None


def _bounds(limits: IcronLimits) -> Dict[str, str]:
    """The values that the description and help() quote, by placeholder."""
    days = limits.max_delay / 86400
    return {
        "max_jobs": str(limits.max_jobs), "min_delay": f"{limits.min_delay:g}",
        "max_delay": f"{limits.max_delay:g}",
        "max_days": f"{days:g} day{'' if days == 1 else 's'}",
        "min_every": f"{limits.min_every:g}", "max_chars": str(limits.max_chars),
        "max_wait": f"{limits.max_wait:g}", "max_output": str(limits.max_output),
        "ended_kept": str(_ENDED_KEPT), "listed_code": str(_LISTED_CODE_CHARS),
        "listed_output": str(_LISTED_OUTPUT_CHARS),
        "code_schedule": _signature_text(_CodeIcron, "schedule", "icron."),
        "code_cancel": _signature_text(_CodeIcron, "cancel", "icron."),
    }


def _signature_text(owner, name: str, prefix: str = "") -> str:
    """A method's signature as calls write it: without self, the internal by,
    and, for the tool's methods, code, which goes in its own field."""
    skip = ("self", "by", "code") if owner is Icron else ("self",)
    parts, keywords = [], False
    for parameter in inspect.signature(getattr(owner, name)).parameters.values():
        if parameter.name in skip:
            continue
        if parameter.kind is parameter.KEYWORD_ONLY and not keywords:
            keywords = True
            if parts:
                parts.append("*")
        if parameter.default is parameter.empty:
            parts.append(parameter.name)
            continue
        default = parameter.default
        if isinstance(default, float) and default.is_integer():
            default = int(default)
        parts.append(f"{parameter.name}={default!r}")
    return f"{prefix}{name}({', '.join(parts)})"


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def _help_samples() -> Dict[str, str]:
    """Results as the formatters write them, at a fixed time, for help()."""
    now = _SAMPLE_NOW
    once = Job("k3f9-1", "main", "message", "scheduled", due=now + 600,
               message="Check CI run 123.", remaining=1, created=now)
    repeat = Job("k3f9-2", "main", "code", "scheduled", due=now + 60,
                 code="await check_credit()", every=60, remaining=3, created=now)
    code = "for step in (1, 2):\n    print('step', step)\n    await asyncio.sleep(20)"
    running = Job("k3f9-3", "main", "exec", "running", code=code, created=now - 30,
                  started=now - 30, output="step 1\nstep 2\n")
    stopping = Job("k3f9-3", "main", "exec", "cancelled", code=code, created=now - 30,
                   started=now - 30, output="step 1\nstep 2\n")
    return {
        "exec": _exec_text(ExecResult("running", running.output, job=running,
                                      reason=_REASON_WAIT), 30.0, now),
        "schedule": _schedule_text(repeat, now),
        "yield_": _yield_text(once, now),
        "cancel": _cancel_text(stopping, now),
        "jobs": _jobs_text((running, once), now),
    }


_PARAMETERS = {
    "type": "object",
    "properties": {
        "call": {"type": "string",
                 "description": "One icron method call, in Python with literal arguments, such "
                                "as jobs() or schedule('Check CI run 123.', after=600)."},
        "code": {"type": "string",
                 "description": "Python source for exec() or schedule(); alone, it runs now, as "
                                "exec(). Leave it out for other calls."},
    },
    # Nothing is required, since code alone is a call, and other fields aren't
    # refused here: the first version's {"action": …} calls, which saves and a
    # resumed model's history hold, must reach the handler (_read_call).
}


__all__ = [
    "ACTIONS", "Code", "ContextHandle", "Delivery", "ExecResult", "Icron", "IcronError",
    "IcronLimits", "IcronStatus", "Job", "MAIN", "ORIGINS", "Run", "TOOL_NAME", "WAKE_HEADER",
    "Wakeup", "log_state",
]
