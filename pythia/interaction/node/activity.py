"""What a root context is doing, for the dot that the glui view shows before it.

A dot has a hue, and words that say why:

- Gray, inactive: no CLI runs on the context.
- Green, running: a CLI works on a turn, as its phase says.
- Yellow, waiting: a CLI waits for input, for a login, or for icron to wake
  it, as after a yield.
- Orange, retrying: a model request failed, and the CLI samples again.
- Red, failed: the last turn ended in a failure, or the CLI stopped in the
  middle of a turn.

A CLI that runs on a save holds its lock, and says there what it's doing:
its phase, and whether it retries (``pythia.interaction._save_lock``). A
save whose lock no CLI holds is at rest, and its log's tail says how its
last turn ended. A save without a lock file was last run by a CLI from
before the lock, which may run still: its dot is a guess from its log's
tail and age, and says so.
"""

from __future__ import annotations

import json
import time
from typing import Optional
from typing import Tuple

from .._save_lock import describe
from .._save_lock import holder
from .._save_lock import lock_path
from .contexts import Context

GRAY, GREEN, YELLOW, ORANGE, RED = "Gray", "Green", "Yellow", "Orange", "Red"
# Lines that say nothing of how a turn went, which the tail skips.
_QUIET = frozenset(("model_sample_boundary", "sample_metadata", "turn_metadata",
                    "user_interaction_boundary", "nil"))
# The kinds of line that end a turn, or that a log starts with.
_ENDED = frozenset(("turn_summary", "init", "session_init"))
# How far back the tail looks for the last call of a turn that ended.
_TAIL_LINES = 64
# For a guess: how long a turn may go without a line before its CLI seems to
# have stopped, a failure before it seems final, and a CLI between turns
# before it seems gone, in seconds.
_WORKING_SECONDS = 600
_RETRYING_SECONDS = 300
_WAITING_SECONDS = 1800


def activity(context: Context, now: Optional[float] = None) -> Tuple[str, str]:
    """``context``'s hue, and why, as a sentence. Reads its log's index, which it
    refreshes, its lock, and the last lines of its log."""
    context.refresh()
    ended, last = _tail(context)
    try:
        owner = holder(context.private)
        locked = owner is not None or lock_path(context.private).exists()
    except OSError:
        owner, locked = None, False
    if owner is not None:
        return _running(owner, _yielded(context, last) if ended == "ended" else False)
    if locked:
        if ended == "failed":
            return RED, "Failed: its last turn ended in a failure, and no CLI runs on it."
        if ended == "unfinished":
            return RED, "Failed: its CLI stopped in the middle of a turn."
        return GRAY, "Inactive: no CLI runs on it."
    return _guess(ended, (time.time() if now is None else now) - context.index.changed)


def _running(owner: dict, yielded: bool) -> Tuple[str, str]:
    """The hue of a context that a CLI runs on, as the lock's line says, and why."""
    by = f" CLI {describe(owner)}."
    phase = owner.get("phase")
    if owner.get("retrying"):
        return ORANGE, "Retrying: a model request failed, and the CLI samples again." + by
    if phase == "failed":
        return RED, "Failed: the last turn ended in a failure; the CLI waits." + by
    if phase == "auth needed":
        return YELLOW, "Waiting: the CLI needs a login." + by
    if phase == "idle":
        return YELLOW, ("Waiting: it yielded to icron, until a job wakes it." if yielded
                        else "Waiting: the CLI waits for input.") + by
    return GREEN, f"Running: {phase or 'a CLI runs on it'}." + by


def _guess(ended: str, age: float) -> Tuple[str, str]:
    """The hue of a context without a lock file, from how its last turn went and how
    long ago its log last changed, in seconds, and why."""
    guess = f" Its log last changed {_ago(age)}, and it has no lock, so this is a guess."
    if ended == "failed":
        if age < _RETRYING_SECONDS:
            return ORANGE, "Retrying, it seems: a model request failed." + guess
        return RED, "Failed, it seems: its last turn ended in a failure." + guess
    if ended == "unfinished":
        if age < _WORKING_SECONDS:
            return GREEN, "Running, it seems: a turn is under way." + guess
        return GRAY, "Inactive, it seems: a turn stopped." + guess
    if age < _WAITING_SECONDS:
        return YELLOW, "Waiting, it seems: between turns." + guess
    return GRAY, "Inactive, it seems." + guess


def _ago(age: float) -> str:
    """``age`` seconds, in words, as long ago: ``5 min ago``, ``3 h ago``, ``2 days ago``."""
    minutes = max(0, int(age // 60))
    if minutes < 1:
        return "less than a minute ago"
    if minutes < 120:
        return f"{minutes} min ago"
    if minutes < 48 * 60:
        return f"{minutes // 60} h ago"
    return f"{minutes // (24 * 60)} days ago"


def _tail(context: Context) -> Tuple[str, int]:
    """How the log's last turn went, by its last line that says: "ended", by a turn
    summary, or as a log starts; "failed", by a model's failure; or "unfinished";
    with that line, from 0, or -1."""
    index = context.index
    with index.lock:
        line = len(index) - 1
        while line >= 0 and index.kind(line)[0] in _QUIET:
            line -= 1
        kind = index.kind(line)[0] if line >= 0 else "init"
    if kind in _ENDED:
        return "ended", line
    return ("failed" if kind == "model_failure" else "unfinished"), line


def _yielded(context: Context, end: int) -> bool:
    """Whether the turn that ended at line ``end`` (from 0) ended with a yield to icron:
    its last call, among the lines before, is icron's ``yield_``."""
    if end < 0:
        return False
    index = context.index
    with index.lock:
        start = max(0, end - _TAIL_LINES)
        for _, record in reversed(context.records(start, end)):
            if record.get("type") == "tool_call":
                if record.get("name") != "icron":
                    return False
                try:
                    call = json.loads(record.get("arguments_json") or "{}").get("call", "")
                except (ValueError, AttributeError):
                    return False
                return isinstance(call, str) and call.lstrip().startswith("yield_(")
            if record.get("type") == "turn_summary":  # an earlier turn's
                return False
    return False
