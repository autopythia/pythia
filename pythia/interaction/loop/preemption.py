"""Where a context's turn loop meets its input side for immediate steering.

The turn loop brackets each sample and tool call with an *operation*. The
input side, on another thread, *flushes* the queued steers: at
``Urgency.IMMEDIATE`` (``/steer``) the loop starts no new operation before it
delivers them, and at ``Urgency.PREEMPT`` (``/steer!``) it also cancels the
operation in flight, if that can be cancelled. A *preempting stop* (a second
Ctrl-C, or ``/exit!``) cancels it too, and every later operation is skipped.
The steers themselves still enter the turn only at the interrupt point, where
the host hands them over and resets the urgency (``take``).
"""

from __future__ import annotations

import enum
import threading
from typing import Callable, Optional


class Urgency(enum.IntEnum):
    """How soon queued steers reach the turn."""

    QUEUED = 0  # at the next interrupt point, after the batch (plain text)
    IMMEDIATE = 1  # once the operation in flight ends (/steer)
    PREEMPT = 2  # also cancel the operation in flight (/steer!)


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
    """One context's urgency and its operation in flight; thread-safe.

    The context thread brackets each sample and tool call with
    ``operation(cancel)``, answers ``TurnHost.should_steer`` with ``due()``,
    and calls ``take()`` where the host hands over the steers. The input side
    calls ``flush`` once steers are queued, and ``preempt_stop`` once a stop
    is requested (the host's ``should_stop`` must already be true).

    Cancels run on a helper thread, never the caller's, since one can take
    seconds (retiring a Claude relay model). Each operation is cancelled at
    most once, and only while it runs: leaving an operation waits for its
    cancel, so a late cancel never reaches the next one (a relay retire is
    model-wide).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._level = Urgency.QUEUED
        self._stopped = False
        self._current: Optional[_Operation] = None

    @property
    def level(self) -> Urgency:
        with self._lock:
            return self._level

    @property
    def stopped(self) -> bool:
        """True after ``preempt_stop``."""
        with self._lock:
            return self._stopped

    def flush(self, level: Urgency) -> None:
        """Have the queued steers delivered at ``level`` (it only rises)."""
        level = Urgency(level)
        with self._lock:
            self._level = max(self._level, level)
            current = self._current if level >= Urgency.PREEMPT else None
        if current is not None:
            current._cancel_async()

    def preempt_stop(self) -> None:
        """Cancel the operation in flight, and skip every later one."""
        with self._lock:
            self._stopped, current = True, self._current
        if current is not None:
            current._cancel_async()

    def due(self) -> bool:
        """True while flushed steers wait for the interrupt point."""
        with self._lock:
            return self._level >= Urgency.IMMEDIATE

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

    ``skipped`` is true if it must not start (a flush or a preempting stop
    came first); ``cancelled`` once its cancel was started.
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
            self.skipped = owner._stopped or owner._level >= Urgency.IMMEDIATE
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


__all__ = ["NOT_PREEMPTIBLE", "Preemption", "Urgency"]
