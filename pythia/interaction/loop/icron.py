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
import concurrent.futures
import contextvars
from dataclasses import dataclass
from datetime import datetime
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
_STREAM_LINES = 40  # output lines that one of the user's runs shows as notices
_WHO = {"main": "main's", "user": "the user's", "host": "the host's"}

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
    detail: Optional[str] = None  # why the call returned before the run ended


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
        return self._icron.schedule(message, code=code, after=after, at=at, every=every,
                                    count=count, by=self._origin)

    def cancel(self, job_id: JobId) -> Job:
        return self._icron.cancel(job_id, by=self._origin)

    def jobs(self) -> Tuple[Job, ...]:
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
                    return ExecResult("failed", error=_format_syntax_error(exc))
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
            self._tool = Tool(ToolSpec(TOOL_NAME, _DESCRIPTION, _PARAMETERS), self._handle,
                              timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS)
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
        fields = dict(arguments)
        action = fields.pop("action", None)
        if action not in ACTIONS:
            raise IcronError(f"icron: action must be one of {', '.join(ACTIONS)}.")
        name = "yield_" if action == "yield" else action
        if "by" in fields:
            raise IcronError(f"icron.{name}(): by is for host code only.")
        if "code" in fields and not isinstance(fields["code"], str):
            raise IcronError(f"icron.{name}(): code must be a string.")
        method = getattr(self, name)
        try:
            inspect.signature(method).bind(**fields)
        except TypeError as exc:
            raise IcronError(f"icron.{name}(): {exc}") from None
        if action != "yield":
            fields["by"] = "main"
        result = method(**fields)
        now = self._clock()
        if action == "schedule":
            data = {**_job_json(result, now), "now": _iso(now)}
        elif action == "exec":
            data = {"state": result.state}
            if result.job is not None and result.state in ("running", "pending"):
                data["job_id"] = result.job.job_id
            data["output"] = result.output
            if result.value is not None:
                data["value"] = result.value
            if result.error is not None:
                data["error"] = result.error
            if result.detail is not None:
                data["detail"] = result.detail
            data["now"] = _iso(now)
            return ToolOutcome(json.dumps(data, ensure_ascii=False),
                               success=result.state not in ("failed", "cancelled"))
        elif action == "yield":
            data = {"yielded": True, "now": _iso(now),
                    "waits_for": None if result is None else _job_json(result, now),
                    "effect": _yield_effect(result)}
        elif action == "cancel":
            data = {"cancelled": result.job_id, "now": _iso(now)}
        else:
            data = {"now": _iso(now), "jobs": [_job_json(job, now, full=True) for job in result]}
        return ToolOutcome(json.dumps(data, ensure_ascii=False))

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
                namespace = self._namespaces[origin] = {
                    "__name__": f"__icron_{origin}__", "__builtins__": builtins,
                    "icron": _CodeIcron(self, origin), "main": self._main,
                    "asyncio": asyncio, "print": self._printer(origin),
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
                    return ExecResult(run._outcome, run.output(), run._value, run._error,
                                      job.snapshot())
                left = deadline - time.monotonic()
                detail = None
                if run._sent_wake:
                    detail = "It sent you a message, which you get after this call."
                elif run._awaits_main:
                    detail = "It waits for the end of your turn, so it goes on as a job."
                elif cancel is not None and cancel.cancelled:
                    detail = "The wait was interrupted; the code goes on as a job."
                elif left <= 0:
                    detail = ("The model needs this result now; the code goes on as a job."
                              if limited_by == "results_due" else
                              f"It still ran after {wait:g} seconds, so it goes on as a job.")
                if detail is not None:
                    run._caller_waits = False
                    job.listed = True
                    state = "running"
                    if run._started is None:
                        state = "pending"
                        detail = ("The watcher is busy, so the code hasn't started; it runs "
                                  "as a job once the watcher is free.")
                    return ExecResult(state, run.output(), job=job.snapshot(), detail=detail)
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
        try:
            data = json.loads(item.output)
            if data.get("every") is not None:
                return "jobs"
            if datetime.fromisoformat(data["due"]).timestamp() > now:
                return "jobs"
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
    return None


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
    try:
        arguments = json.loads(call.arguments_json)
    except ValueError:
        return None
    action = arguments.get("action") if isinstance(arguments, dict) else None
    return action if isinstance(action, str) else None


def _delivery_label(delivery: Delivery) -> str:
    job = delivery.job
    if job is None:
        return "From the host"
    yours = {"main": "yours", "user": "the user's", "host": "the host's"}[job.origin]
    code = {"main": "your code", "user": "the user's code", "host": "the host's code"}[job.origin]
    if job.kind == "exec":
        source = "/py" if job.origin == "user" else "exec"
        text = f"{code}, from {source} at {_hms(job.started or job.created)}"
    else:
        text = yours if job.kind == "message" else code
        if job.kind == "message" and job.origin == "user":
            text += ", from /cron"
        if delivery.due is not None:
            text += f"; due {_iso(delivery.due)}"
            if job.every is not None:
                text += f", every {job.every:g} s"
        if delivery.merged:
            text += (f"; {delivery.merged} earlier firing"
                     f"{'s' if delivery.merged > 1 else ''} merged")
    return f"Job {job.job_id} ({text})"


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


def _job_json(job: Job, now: float, *, full: bool = False) -> dict:
    data = {"job_id": job.job_id}
    if full:
        data["origin"] = job.origin
    data["state"] = job.state
    if job.due is not None:
        data["due"] = _iso(job.due)
        data["in_seconds"] = max(0, round(job.due - now))
    if job.state == "running" and job.started is not None:
        data["started"] = _iso(job.started)
    if job.kind != "exec":
        data["every"] = job.every
        data["remaining"] = job.remaining
    if full:
        if job.message is not None:
            data["message"] = job.message
        if job.code is not None:
            data["code"] = _clip_head(job.code, 2000)
        if job.output:
            data["output"] = _clip_tail(job.output, 4000)
    if job.error is not None:
        data["error"] = job.error if full else job.error.strip().splitlines()[-1]
    return data


def _yield_effect(job: Optional[Job]) -> str:
    text = "Your turn ends once this tool batch is saved."
    if job is None:
        return text + " A wake-up already waits for you."
    if job.due is None or job.state == "running":
        return (text + f" Job {job.job_id}, running code, resumes you when it sends you a "
                "message or ends, unless the user writes first.")
    return (text + f" Job {job.job_id} resumes you at {_iso(job.due)}, unless the user writes "
            "first.")


_DESCRIPTION = (
    "Run Python, schedule work, and pause your turn, through icron: this session's "
    "in-process scheduler. Each call is one icron API call: action names the function, and "
    "the other fields are its keyword arguments.\n"
    "- exec(code, wait=10): run Python now, in this process, on icron's event loop; "
    "top-level await works. Returns what it printed, a final expression's value, or the "
    f"traceback. After `wait` seconds (at most {IcronLimits.max_wait:g}), it keeps running "
    "as a job. Variables "
    "persist between calls. Code sees icron (schedule, cancel, jobs, now, current), main "
    "(you: main.resume(text) sends you text as a message, and awaiting it waits for the end "
    "of the turn that receives it), asyncio, and print. Use asyncio.to_thread for blocking "
    "work.\n"
    "- schedule(message=None, code=None, after=None, at=None, every=None, count=None): run "
    "code later: `after` seconds from now, or at `at` (ISO 8601 with a UTC offset); with "
    "neither, `every` seconds from now. With every (30 or more), it repeats; count is how "
    "many times in all (default 1, or no limit with every). A message is the default code, "
    "await main.resume(message). A job runs once at a time, and firings during a run merge "
    "into it.\n"
    "- yield(note): end your turn once this tool batch is saved, without a final answer; "
    "note says what you are waiting for. It needs a pending job, yours or the user's, or "
    "your running code: the next message from icron resumes you, unless the user writes "
    "first. If a job of yours ends while you wait without messaging you, you get its end.\n"
    "- cancel(job_id) ends one of your jobs and stops its running code. jobs() lists yours "
    "and the user's (the user schedules messages with /cron), with the current time and "
    "each job's latest output.\n"
    "Messages from icron reach you as text typed by the user would: during a turn, right "
    "before your next sample, once the current tool batch is done; between turns, they "
    "start one. Jobs and Python state exist only while this process runs. When the task is "
    "done, give a final answer instead of yielding; if you need something only the user can "
    "provide, ask in your final answer."
)

_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "message": {"type": "string", "maxLength": IcronLimits.max_chars},
        "code": {"type": "string"},
        "after": {"type": "number", "minimum": 1, "maximum": 604800},
        "at": {"type": "string"},
        "every": {"type": "number", "minimum": 30, "maximum": 604800},
        "count": {"type": "integer", "minimum": 1},
        "wait": {"type": "number", "minimum": 0, "maximum": IcronLimits.max_wait},
        "note": {"type": "string", "maxLength": IcronLimits.max_chars},
        "job_id": {"type": "string"},
    },
    "required": ["action"],
    "additionalProperties": False,
}


__all__ = [
    "ACTIONS", "Code", "ContextHandle", "Delivery", "ExecResult", "Icron", "IcronError",
    "IcronLimits", "IcronStatus", "Job", "MAIN", "ORIGINS", "Run", "TOOL_NAME", "WAKE_HEADER",
    "Wakeup", "log_state",
]
