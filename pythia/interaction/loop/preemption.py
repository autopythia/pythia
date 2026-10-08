"""Where a context's turn loop meets its input side for steering and stopping.

The turn loop brackets each sample and tool call with an *operation*. The
input side, on another thread, *flushes* the queued steers: at
``Urgency.IMMEDIATE`` (``/steer!``) the loop starts no new operation before it
delivers them, and at ``Urgency.PREEMPT`` (``/steer!!``) it also cancels the
operation in flight, if that can be cancelled. The steers themselves still
enter the turn only at the interrupt point, where the host hands them over and
resets the urgency (``take``). Plain text (or ``/steer``) is ``QUEUED``: it
waits for the interrupt point.

A *stop* has the same three levels: at ``QUEUED`` (``/exit``) the turn ends
at its next interrupt point, after its tool batch; at ``IMMEDIATE``
(``/exit!``, the first Ctrl-C) every later operation is skipped; and at
``PREEMPT`` (``/exit!!``, a later Ctrl-C) the operation in flight is cancelled
too. A stop drops the steers, but steers flushed before it still keep the turn
from starting anything new.
"""

from __future__ import annotations

import enum
import threading
from typing import Callable, Optional


class Urgency(enum.IntEnum):
    """How soon steers reach the turn, or a stop ends it: the number of ``!``.

    ``QUEUED`` is falsy; test an optional level with ``is None``.
    """

    QUEUED = 0  # at the next interrupt point, after the batch (plain text, /steer, /exit)
    IMMEDIATE = 1  # once the operation in flight ends (/steer!, /exit!)
    PREEMPT = 2  # also cancel the operation in flight (/steer!!, /exit!!)


def command_urgency(word: str, *names: str) -> Optional[Urgency]:
    """The urgency of a command word, by its trailing ``!``; None if it isn't one.

    ``command_urgency("/exit!", "/exit", "/quit")`` is ``IMMEDIATE``. Three or
    more ``!`` read as two, so a command never fails for an extra one.
    """
    stem = word.rstrip("!")
    if stem not in names:
        return None
    return Urgency(min(len(word) - len(stem), Urgency.PREEMPT))


def escalated_stop(level: Optional[Urgency]) -> Urgency:
    """The stop that Ctrl-C requests, given the current one (or None).

    The first press is ``/exit!``; once stopping at that level or above, a
    press is ``/exit!!``. So after ``/exit``, Ctrl-C goes to IMMEDIATE, then
    to PREEMPT.
    """
    if level is not None and level >= Urgency.IMMEDIATE:
        return Urgency.PREEMPT
    return Urgency.IMMEDIATE


class _NotPreemptible:
    """The default host's operation: never skipped, never cancelled."""

    skipped = False
    cancelled = False

    def __enter__(self) -> "_NotPreemptible":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


NOT_PREEMPTIBLE = _NotPreemptible()


class Preemption:
    """One context's urgency, stop level, and operation in flight; thread-safe.

    The context thread brackets each sample and tool call with
    ``operation(cancel)``, answers ``TurnHost.should_steer`` with ``due()``,
    and calls ``take()`` where the host hands over the steers. The input side
    calls ``flush`` once steers are queued, and ``stop`` when a stop is
    requested; the host's ``should_stop`` and ``stop_requested`` read
    ``stop_level``.

    Cancels run on a helper thread, never the caller's, since one can take
    seconds (retiring a Claude relay model). Each operation is cancelled at
    most once, and only while it runs: leaving an operation waits for its
    cancel, so a late cancel never reaches the next one (a relay retire is
    model-wide).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._level = Urgency.QUEUED
        self._stop: Optional[Urgency] = None
        self._current: Optional[_Operation] = None

    @property
    def level(self) -> Urgency:
        """The steers' urgency; QUEUED once stopping (a stop drops them)."""
        with self._lock:
            return self._level

    @property
    def stop_level(self) -> Optional[Urgency]:
        """The stop's level, or None: what ``stop`` requested, but at least
        IMMEDIATE if steers were flushed before it."""
        with self._lock:
            return self._stop

    def flush(self, level: Urgency) -> None:
        """Have the queued steers delivered at ``level`` (it only rises).

        Does nothing once stopping: the stop dropped the steers.
        """
        level = Urgency(level)
        with self._lock:
            if self._stop is not None:
                return
            self._level = max(self._level, level)
            current = self._current if level >= Urgency.PREEMPT else None
        if current is not None:
            current._cancel_async()

    def stop(self, level: Urgency) -> None:
        """Stop at ``level`` (it only rises).

        QUEUED: the turn ends at its next interrupt point. IMMEDIATE: every
        later operation is skipped. PREEMPT: the operation in flight is also
        cancelled. The steers' urgency resets, since a stop drops the steers,
        but steers flushed before the first stop still keep the turn from
        starting anything new: that stop is at least IMMEDIATE. It never
        becomes PREEMPT that way; a preempting steer already cancelled the
        operation it found.
        """
        level = Urgency(level)
        with self._lock:
            if self._stop is None:
                level = max(level, min(self._level, Urgency.IMMEDIATE))
            self._stop = level if self._stop is None else max(self._stop, level)
            self._level = Urgency.QUEUED
            current = self._current if self._stop >= Urgency.PREEMPT else None
        if current is not None:
            current._cancel_async()

    def due(self) -> bool:
        """True while flushed steers wait for the interrupt point (never once
        stopping)."""
        with self._lock:
            return self._stop is None and self._level >= Urgency.IMMEDIATE

    def take(self) -> Urgency:
        """With the steers, at the interrupt point: reset and return the urgency."""
        with self._lock:
            level, self._level = self._level, Urgency.QUEUED
            return level

    def operation(self, cancel: Optional[Callable[[], None]] = None) -> "_Operation":
        """Bracket one sample or tool call (see ``TurnHost.preemptible``)."""
        return _Operation(self, cancel)


class _Operation:
    """One sample or tool call: a context manager, entered once.

    ``skipped`` is true if it must not start (a flush, or a stop at
    IMMEDIATE or above, came first); ``cancelled`` once its cancel was
    started.
    """

    def __init__(self, owner: Preemption, cancel: Optional[Callable[[], None]]) -> None:
        self._owner, self._cancel = owner, cancel
        self.skipped = False
        self.cancelled = False
        self._cancel_done = threading.Event()
        self._cancel_done.set()

    def __enter__(self) -> "_Operation":
        owner = self._owner
        with owner._lock:
            if owner._current is not None:
                raise RuntimeError("Preemptible operations do not nest.")
            # One lock decides: input after the interrupt point either skips
            # this operation or reaches it while it runs.
            stop = owner._stop
            self.skipped = ((stop is not None and stop >= Urgency.IMMEDIATE)
                            or owner._level >= Urgency.IMMEDIATE)
            if not self.skipped:
                owner._current = self
        return self

    def __exit__(self, *exc_info) -> bool:
        with self._owner._lock:
            if self._owner._current is self:
                self._owner._current = None
        self._cancel_done.wait()  # a cancel never outlives its operation
        return False

    def _cancel_async(self) -> None:
        with self._owner._lock:
            if self.cancelled or self._cancel is None or self._owner._current is not self:
                return
            self.cancelled = True
            self._cancel_done.clear()
        threading.Thread(target=self._run_cancel, name="interaction-cancel",
                         daemon=True).start()

    def _run_cancel(self) -> None:
        try:
            self._cancel()
        except Exception:
            # A cancel that fails leaves the operation to finish or fail on
            # its own; the turn loop handles either.
            pass
        finally:
            self._cancel_done.set()


__all__ = ["NOT_PREEMPTIBLE", "Preemption", "Urgency", "command_urgency", "escalated_stop"]
