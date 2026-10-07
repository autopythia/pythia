"""Role-agnostic supervision: a supervised context yields to its supervisor.

A supervised context publishes one :class:`Yield` each time its turn loop
stops, then waits (unless stopping or unsupervised) for the supervisor's
answer: a resume message, or None to release it. The supervisor drives it as a
coroutine through :class:`SupervisedHandle`::

    yield_ = await handle(resume)   # answer the outstanding yield, await the next

A non-resumable yield is raised to the supervisor as :class:`Fault`. The
transport is a plain ``queue.Queue`` whose ``None`` sentinel is sent only after
the supervised owner has exited, so no yield is lost at shutdown.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import queue
import threading
from typing import Awaitable, Callable, Optional, Sequence, Tuple

from ..display import render_interaction_items
from ..environment import Tool, ToolOutcome, ToolSpec
from ..items import InteractionItem, Message, ModelFailure, ToolCall, ToolResult
from ..items import UserInteractionBoundary
from ..model import ModelError
from ..save import SaveError
from ..user import UserInteraction
from .kernel import run_turn


YIELD_KINDS = ("ended", "failed", "stopped", "yielded")


@dataclass(frozen=True)
class Yield:
    """Why a supervised turn loop stopped. Pure data; safe to show a model."""

    context: int
    job_id: Optional[str]
    job_text: str
    kind: str
    resumable: bool
    # The supervised log length at the yield: items 0..revision-1 are
    # addressable through the handle's view; the log only grows.
    revision: int
    reason: Optional[str] = None
    failure: Optional[ModelFailure] = None
    final_text: Optional[str] = None
    resumes: int = 0
    # Main's handoff note when it called the yield tool (kind "yielded").
    yield_text: Optional[str] = None
    # User steers during the task, in order: (text, delivered to main yet).
    steers: Tuple[Tuple[str, bool], ...] = ()
    # Context for the report, shown only when distinct from job_text: the
    # supervised log's chronologically first user message, and, for a job
    # queued from a steer the supervised context never received, the job text
    # that steer was sent to.
    first_user_text: Optional[str] = None
    steer_target_text: Optional[str] = None
    # The task continued the supervised log's unfinished turn (/continue): its
    # first turn had no new user message, and job_text is the request that
    # turn belongs to.
    continued: bool = False

    def __post_init__(self) -> None:
        if self.kind not in YIELD_KINDS:
            raise ValueError(f"Yield kind must be one of {YIELD_KINDS}.")
        for name in ("resumable", "continued"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")
        for name in ("revision", "resumes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        for name in ("first_user_text", "steer_target_text"):
            if not isinstance(getattr(self, name), (str, type(None))):
                raise TypeError(f"{name} must be a string or None.")
        if self.failure is not None and not isinstance(self.failure, ModelFailure):
            raise TypeError("failure must be ModelFailure or None.")
        if self.kind == "yielded":
            if not isinstance(self.yield_text, str) or not self.yield_text.strip():
                raise ValueError("A yielded report requires a nonblank yield_text.")
            if self.final_text is not None or self.failure is not None or self.reason is not None:
                raise ValueError("A yielded report has no final text and no failure.")
        elif self.yield_text is not None:
            raise ValueError("Only a yielded report has yield_text.")
        steers = tuple(tuple(steer) for steer in self.steers)
        if not all(len(steer) == 2 and isinstance(steer[0], str) and type(steer[1]) is bool
                   for steer in steers):
            raise TypeError("steers must be (text, delivered) pairs.")
        object.__setattr__(self, "steers", steers)


class Fault(Exception):
    """A non-resumable yield, raised to the supervisor."""

    def __init__(self, yield_: Yield) -> None:
        reason = f" ({yield_.reason})" if yield_.reason else ""
        super().__init__(f"#{yield_.context} {yield_.kind}{reason}")
        self.yield_ = yield_


class _Reply:
    """One-shot answer slot guarded by its channel's condition; first write wins."""

    def __init__(self, condition: threading.Condition) -> None:
        self._condition = condition
        self.done = False
        self.value: Optional[str] = None

    def set(self, value: Optional[str]) -> None:
        with self._condition:
            if not self.done:
                self.done, self.value = True, value
                self._condition.notify_all()


class YieldChannel:
    """Supervised-side end of the yield queue."""

    def __init__(self, stop: threading.Event, transport: Optional[queue.Queue] = None) -> None:
        self._stop = stop
        self._queue = queue.Queue() if transport is None else transport
        self._changed = threading.Condition()
        self._detached = False

    def signal(self, yield_: Yield, view: Sequence[InteractionItem] = ()) -> Optional[str]:
        """Publish a yield and its read-only log view; return the answer.

        Waits until the supervisor answers unless stopping or unsupervised, in
        which case the yield is only a notification. None releases.
        """
        if not isinstance(yield_, Yield):
            raise TypeError("signal() requires a Yield.")
        view = tuple(view)
        with self._changed:
            if self._stop.is_set() or self._detached:
                self._queue.put((yield_, view, None))
                return None
            reply = _Reply(self._changed)
            self._queue.put((yield_, view, reply))
            while not (reply.done or self._stop.is_set() or self._detached):
                self._changed.wait()
            return reply.value if reply.done else None

    def wake(self) -> None:
        """Re-check waits; call after setting the stop event."""
        with self._changed:
            self._changed.notify_all()

    def detach(self) -> None:
        """No supervisor remains: release waits; later yields only notify."""
        with self._changed:
            self._detached = True
            self._changed.notify_all()

    def close(self) -> None:
        """Send the shutdown sentinel, after the supervised owner has exited."""
        self._queue.put(None)


class SupervisedHandle:
    """Supervisor-side coroutine handle: ``yield_ = await handle(resume)``.

    Use from one task on one event loop. ``bridge`` is a dedicated executor for
    blocking queue reads (a single thread suffices).
    """

    def __init__(self, channel: YieldChannel, bridge) -> None:
        self._channel, self._bridge = channel, bridge
        self._pending = None  # (yield_, view, reply) for the outstanding yield
        self._read = None  # in-flight queue read; survives caller cancellation
        self._busy = False

    @property
    def view(self) -> Tuple[InteractionItem, ...]:
        """The supervised log as of the outstanding yield (empty if none)."""
        return () if self._pending is None else self._pending[1]

    async def __call__(self, resume: Optional[str] = None) -> Optional[Yield]:
        """Answer the outstanding yield, then await the next one.

        Returns a resumable Yield, or None once the supervised side has exited.
        Raises Fault for a non-resumable yield; the handle stays usable. A
        cancelled call loses nothing: its read stays in flight for the next
        call, and any resume it carried was already delivered.
        """
        if self._busy:
            raise RuntimeError("The supervised handle already has a caller.")
        if resume is not None and not isinstance(resume, str):
            raise TypeError("resume must be a string or None.")
        if self._pending is not None:
            yield_, _view, reply = self._pending
            if resume is not None and not yield_.resumable:
                raise ValueError("A non-resumable yield cannot be resumed.")
            self._pending = None
            if reply is not None:
                reply.set(resume)
        elif resume is not None:
            raise ValueError("There is no yield to resume.")
        self._busy = True
        try:
            if self._read is None:
                loop = asyncio.get_running_loop()
                self._read = loop.run_in_executor(self._bridge, self._channel._queue.get)
            item = await asyncio.shield(self._read)
            self._read = None
        finally:
            self._busy = False
        if item is None:
            return None
        self._pending = item
        if not item[0].resumable:
            raise Fault(item[0])
        return item[0]

    def detach(self) -> None:
        """Stop supervising: release a waiting supervised side for good."""
        pending, self._pending = self._pending, None
        self._channel.detach()
        if pending is not None and pending[2] is not None:
            pending[2].set(None)


async def supervise(
    handle: SupervisedHandle,
    decide: Callable[[Yield], Awaitable[Optional[str]]],
    *,
    on_fault: Optional[Callable[[Fault], None]] = None,
) -> None:
    """Drive a supervised context until it exits; never leave it waiting.

    ``decide`` maps each resumable yield to a resume message or None. A Fault
    is observed through ``on_fault`` and then propagated.
    """
    resume: Optional[str] = None
    try:
        while True:
            try:
                yield_ = await handle(resume)
            except Fault as fault:
                if on_fault is not None:
                    on_fault(fault)
                # TODO(supervision): handle non-resumable yields here instead
                # of propagating, e.g. rewrite the supervised context or
                # hot-reload code, then resume it (needs a richer reply type).
                raise
            if yield_ is None:
                return
            resume = await decide(yield_)
    finally:
        handle.detach()


# The supervisor's side: its tools, its report, and one model-driven decision.

READ_LIMIT, READ_MAX_LIMIT, READ_ITEM_CHARS = 20, 50, 2000


def _clip(text: str) -> str:
    """Cut text like one ``read_items`` entry, saying how much was left out."""
    if len(text) <= READ_ITEM_CHARS:
        return text
    return text[:READ_ITEM_CHARS] + f"\n[{len(text) - READ_ITEM_CHARS} more characters omitted]"


def read_items(view: Sequence[InteractionItem], start: int, limit: int) -> dict:
    """Render items start..start+limit-1 of a read-only log view as plain text."""
    calls = [item for item in view[:start] if isinstance(item, ToolCall)]
    entries = []
    for index in range(start, min(len(view), start + limit)):
        item = view[index]
        text = "\n".join(display.text for display in render_interaction_items(
            (item,), source_calls=calls, color=False))
        if isinstance(item, ToolCall):
            calls.append(item)
        entries.append({"index": index, "type": type(item).__name__, "text": _clip(text)})
    end = start + len(entries)
    return {"revision": len(view), "start": start, "next": end,
            "has_more": end < len(view), "items": entries}


class SupervisorTools:
    """The supervisor's tools, bound to the one context it supervises.

    Touched only on the supervisor's thread. ``resume`` records a message that
    takes effect once the supervisor's turn ends; ``read_context`` pages
    through the supervised log as of the current report (``handle.view``).
    """

    def __init__(self, supervised: str = "main (#1)") -> None:
        self.handle: Optional[SupervisedHandle] = None
        self._supervised = supervised
        self._open = False
        self._resume: Optional[str] = None

    def begin(self) -> None:
        self._open, self._resume = True, None

    def end(self) -> Optional[str]:
        resume, self._open, self._resume = self._resume, False, None
        return resume

    def tools(self) -> Tuple[Tool, ...]:
        def resume(arguments, *, timeout_seconds=None):
            del timeout_seconds
            content = arguments.get("content")
            if set(arguments) != {"content"} or not isinstance(content, str) or not content.strip():
                raise ValueError("resume requires only nonempty content.")
            if not self._open:
                raise ValueError("No report is awaiting a decision.")
            if self._resume is not None:
                raise ValueError("A resume is already recorded for this report; end your turn.")
            self._resume = content
            return ToolOutcome(f"Recorded: {self._supervised} resumes with this message "
                               "after your turn ends.")

        def read_context(arguments, *, timeout_seconds=None):
            del timeout_seconds
            view = () if self.handle is None else self.handle.view
            if set(arguments) - {"start", "limit"}:
                raise ValueError("read_context accepts only start and limit.")
            limit = arguments.get("limit", READ_LIMIT)
            if type(limit) is not int or not 1 <= limit <= READ_MAX_LIMIT:
                raise ValueError(f"limit must be an integer from 1 to {READ_MAX_LIMIT}.")
            start = arguments.get("start", max(0, len(view) - limit))
            if type(start) is not int or not 0 <= start <= len(view):
                raise ValueError(f"start must be an integer from 0 to {len(view)}.")
            return ToolOutcome(json.dumps(read_items(view, start, limit), ensure_ascii=False))

        return (
            Tool(ToolSpec(
                "resume",
                f"Resume {self._supervised} with a follow-up message once your turn ends. "
                "Call at most once per report; end your turn without calling it to "
                "release it.",
                {"type": "object", "properties": {"content": {"type": "string"}},
                 "required": ["content"], "additionalProperties": False}),
                resume, timeout_seconds=10),
            Tool(ToolSpec(
                "read_context",
                f"Read the log of {self._supervised} as of the current report. Items are "
                "addressed by index 0..revision-1; omit start for the last items. Follow "
                "next while has_more.",
                {"type": "object", "properties": {
                    "start": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": READ_MAX_LIMIT}},
                 "additionalProperties": False}),
                read_context, timeout_seconds=10),
        )


def report_text(yield_: Yield, *, supervised: str = "Main") -> str:
    """The supervisor's user message for one handoff (no log contents).

    The first user message and the steer target appear only when they differ
    from the job text; a steer target equal to the quoted first user message
    is noted rather than quoted twice. Both are context, so each is cut to
    ``READ_ITEM_CHARS`` like a ``read_context`` entry; the request is whole.
    """
    outcome = yield_.kind if yield_.reason is None else f"{yield_.kind} ({yield_.reason})"
    first, target = (text if text is not None and text.strip() and text != yield_.job_text
                     else None for text in (yield_.first_user_text, yield_.steer_target_text))
    lines = [f"{supervised} (#{yield_.context}) handed off task {yield_.job_id} "
             f"(watcher resumes so far: {yield_.resumes}).", ""]
    if yield_.continued:
        lines += [f"{supervised} continued an unfinished turn from its saved log (/continue); "
                  "the user sent no new message. The request below is the one that turn "
                  "belongs to.", ""]
    if first is not None:
        lines += ["First user message of the session (for context):", _clip(first), ""]
    if target is not None:
        why = f"; {supervised} never received that steer, so it was queued as a separate request"
        if target == first:
            lines += [f"The user sent the request below to steer the message above{why}.", ""]
        else:
            lines += [f"Earlier request (the user sent the request below to steer it{why}):",
                      _clip(target), ""]
    lines += ["User request:", yield_.job_text]
    if yield_.steers:
        lines += ["", "User steers during this task:"]
        lines += [f"{number}. {'' if delivered else '(not yet delivered) '}{text}"
                  for number, (text, delivered) in enumerate(yield_.steers, 1)]
    lines += ["", f"Outcome: {outcome}"]
    if yield_.failure is not None:
        lines.append(f"Failure: {yield_.failure.category}: {yield_.failure.message}")
    if yield_.final_text is not None:
        lines += ["", f"{supervised}'s final answer:", yield_.final_text]
    if yield_.yield_text is not None:
        lines += ["", f"{supervised}'s handoff note:", yield_.yield_text]
    lines += ["", f"{supervised}'s log has {yield_.revision} items (read_context indices "
                  f"0..{yield_.revision - 1})."]
    return "\n".join(lines)


YIELD_MAX_CHARS = 8000


class YieldTool:
    """The supervised context's ``yield`` tool, and ``run_turn``'s control.

    Touched only on the supervised context's thread. The task loop opens it
    (``begin``) for each turn and clears it (``end``) in ``finally``, so a
    request never leaks into a resumed turn or the next task. The handler only
    records the note; the turn ends once the batch is saved.
    """

    NAME = "yield"

    def __init__(self, supervisor: str = "the watcher") -> None:
        self._supervisor = supervisor
        self._open = False
        self._note: Optional[str] = None

    def begin(self) -> None:
        self._open, self._note = True, None

    def end(self) -> Optional[str]:
        note, self._open, self._note = self._note, False, None
        return note

    def request(self) -> Optional[str]:
        """The recorded handoff note, if any (``run_turn``'s control)."""
        return self._note

    @classmethod
    def recorded_in(cls, items: Sequence[InteractionItem]) -> bool:
        """True when a raw log's last turn has a successful ``yield`` result.

        The last turn is the log after its last ``UserInteractionBoundary``
        (steers and ``/continue`` add none). Such a turn handed off, or would
        have once its batch was saved (a stop can cut the batch short), so its
        next step is a reply to the note, not a sample.
        """
        items = tuple(items)
        start = max((index + 1 for index, item in enumerate(items)
                     if isinstance(item, UserInteractionBoundary)), default=0)
        turn = items[start:]
        calls = {item.call_id for item in turn
                 if isinstance(item, ToolCall) and item.name == cls.NAME}
        return any(isinstance(item, ToolResult) and item.success and item.call_id in calls
                   for item in turn)

    def tools(self) -> Tuple[Tool, ...]:
        def yield_(arguments, *, timeout_seconds=None):
            del timeout_seconds
            content = arguments.get("content")
            if set(arguments) != {"content"} or not isinstance(content, str) or not content.strip():
                raise ValueError("yield requires only nonempty content.")
            if len(content) > YIELD_MAX_CHARS:
                raise ValueError(f"content must be at most {YIELD_MAX_CHARS} characters.")
            try:
                content.encode("utf-8")
            except UnicodeError:
                raise ValueError("content must be valid UTF-8 text.") from None
            if not self._open:
                raise ValueError("No turn is open for a handoff.")
            if self._note is not None:
                raise ValueError("A handoff is already recorded; it takes effect once "
                                 "this tool batch is saved.")
            self._note = content
            return ToolOutcome(f"Recorded. Control passes to {self._supervisor} after this "
                               "tool batch is saved.")

        return (Tool(ToolSpec(
            self.NAME,
            f"Pause this task and hand off to {self._supervisor} for review or a decision. "
            "content says what you have done, why you are handing off, and what you "
            "need. Your turn ends once this tool batch is saved; you get a follow-up "
            f"message in a new turn if {self._supervisor} resumes you. Call it on its own. "
            "It is not a final answer: when the task is done, answer instead; if you "
            "need something only the user can provide, ask in your final answer.",
            {"type": "object",
             "properties": {"content": {"type": "string", "maxLength": YIELD_MAX_CHARS}},
             "required": ["content"], "additionalProperties": False}),
            yield_, timeout_seconds=10),)


def decide(yield_: Yield, context, model, environment, config, host,
           tools: SupervisorTools, *, supervised: str = "Main"):
    """One model-driven decision on a handoff, on the supervisor's thread.

    Saves the report as a user message, runs one turn with the supervisor's
    tools, and returns ``(resume, error)``: the recorded resume message (None
    releases the supervised context) and the turn's failure, if any. A
    recorded resume stands even if the turn then failed.
    """
    report = UserInteraction((Message("user", report_text(yield_, supervised=supervised)),))
    tools.begin()
    error = None
    try:
        host.append(context, report.context_items())
        host.show(report.display_items())
        run_turn(context, model, environment, config, host)
    except Exception as exc:
        error = exc
    finally:
        resume = tools.end()
    return resume, error


# The supervised side: one task's turns, each ending in a handoff.

def supervised_turn(text: str, resumes: int, context, model, environment, config, host, *,
                    job_text: str, job_id: Optional[str] = None, context_id: int = 1,
                    control: Optional[YieldTool] = None,
                    follow_up: Optional[Callable[[str], str]] = None,
                    steers: Optional[Callable[[], Sequence[Tuple[str, bool]]]] = None,
                    first_user_text: Optional[str] = None,
                    steer_target_text: Optional[str] = None,
                    continued: bool = False) -> Yield:
    """Run one supervised turn on ``text`` and describe how its loop stopped.

    The first turn of a task (``resumes == 0``) gets ``text`` verbatim; a
    resumed turn gets ``follow_up(text)``. The first turn of a ``continued``
    task gets no user message: it continues the saved turn where it stopped
    (``/continue``), and ``text`` only names its request. ``control`` (the context's
    :class:`YieldTool`) is opened for this turn only. ``steers`` returns the
    task's steers as (text, delivered) pairs once the turn has stopped.
    ``first_user_text`` and ``steer_target_text`` are copied into the Yield for
    the report. Any failure becomes a ``failed`` Yield, resumable unless the log
    is unsafe to continue (unsaved state or unanswered tool calls).
    """
    fields = {"context": context_id, "job_id": job_id, "job_text": job_text,
              "resumes": resumes, "first_user_text": first_user_text,
              "steer_target_text": steer_target_text, "continued": continued}
    try:
        if not (continued and resumes == 0):
            content = text if resumes == 0 or follow_up is None else follow_up(text)
            user = UserInteraction((Message("user", content),))
            host.append(context, user.context_items())
            host.show(user.display_items())
        if control is not None:
            control.begin()
        try:
            result = run_turn(context, model, environment, config, host, control=control)
        finally:
            if control is not None:
                control.end()  # A request never leaks into the next turn.
    except Exception as exc:
        fields["steers"] = () if steers is None else tuple(steers())
        resumable = not (isinstance(exc, SaveError) or context.pending_tool_calls())
        return Yield(kind="failed", resumable=resumable, reason=type(exc).__name__,
                     failure=exc.failure if isinstance(exc, ModelError) else None,
                     revision=len(context), **fields)
    fields["steers"] = () if steers is None else tuple(steers())  # the turn has stopped
    if result.kind == "stopped":
        return Yield(kind="stopped", resumable=False, revision=len(context), **fields)
    if result.kind == "yielded":
        return Yield(kind="yielded", resumable=True, yield_text=result.note,
                     revision=len(context), **fields)
    return Yield(kind="ended", resumable=True, final_text=result.final_text,
                 revision=len(context), **fields)


def run_supervised_task(job_text: str, context, model, environment, config, host,
                        channel: YieldChannel, *, job_id: Optional[str] = None,
                        context_id: int = 1, control: Optional[YieldTool] = None,
                        follow_up: Optional[Callable[[str], str]] = None,
                        steers: Optional[Callable[[], Sequence[Tuple[str, bool]]]] = None,
                        stopping: Optional[Callable[[], bool]] = None,
                        on_fault: Optional[Callable[[Yield], None]] = None,
                        settle: Optional[Callable[[Optional[Yield]], None]] = None,
                        waiting_phase: str = "awaiting supervisor",
                        first_user_text: Optional[str] = None,
                        steer_target_text: Optional[str] = None,
                        continued: bool = False) -> Optional[Yield]:
    """Run one task's turns until the supervisor releases it; return the last Yield.

    Runs on the supervised context's thread. Each turn ends in a handoff: its
    Yield is published on ``channel`` and this thread blocks until the
    supervisor answers. A resume message starts another turn on the same task.
    A release (None), a non-resumable Yield, or ``stopping()`` ends the task.
    ``on_fault`` runs before a non-resumable Yield is published. ``settle``
    runs exactly once at the end with the last Yield (None if none was built),
    even if an error escapes. The other keywords are passed to
    :func:`supervised_turn`.
    """
    text, resumes, yield_ = job_text, 0, None
    try:
        while True:
            yield_ = supervised_turn(
                text, resumes, context, model, environment, config, host,
                job_text=job_text, job_id=job_id, context_id=context_id,
                control=control, follow_up=follow_up, steers=steers,
                first_user_text=first_user_text, steer_target_text=steer_target_text,
                continued=continued)
            if not yield_.resumable and on_fault is not None:
                on_fault(yield_)
            host.phase(waiting_phase)
            resume = channel.signal(yield_, context.items)
            if not yield_.resumable:
                # TODO(supervision): apply a supervisor repair verdict here (e.g.
                # rewrite the context, hot-reload code) instead of always
                # propagating the fault.
                break
            if resume is None or (stopping is not None and stopping()):
                break
            text, resumes = resume, resumes + 1
        return yield_
    finally:
        if settle is not None:
            settle(yield_)


__all__ = ["Fault", "SupervisedHandle", "SupervisorTools", "Yield", "YieldChannel", "YieldTool",
           "YIELD_KINDS", "YIELD_MAX_CHARS", "decide", "read_items", "report_text",
           "run_supervised_task", "supervise", "supervised_turn"]
