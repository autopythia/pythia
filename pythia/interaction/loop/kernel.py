"""The shared inner turn loop: sample the model, run its tool calls, repeat.

One copy serves every caller: the CLI, auto's roles, and the demo. What
differs between them lives in a :class:`TurnHost`: how items are saved and
shown, the phase, stopping, tracing, and a few hooks. Every host method is
called on the context thread, the one thread that runs this context's model,
tools, and saves.

Right before each sample the loop resumes the host's *interrupt*, the logical
coroutine that stands for the user's top-level input. Its blocking case is
the caller's quiescent wait between tasks; here it answers at once.
"""

from __future__ import annotations

import contextlib
import enum
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Optional, Sequence, Tuple

from ..compaction import CompactionResult
from ..compaction import NothingToCompact
from ..compaction import auto_compaction_due
from ..compaction import create_default_compactor
from ..compaction import uses_host_auto_compaction
from ..display import render_interaction_items
from ..items import InteractionItem
from ..items import Message
from ..items import ModelSampleBoundary
from ..items import ToolCall
from ..items import ToolResult
from ..items import summarize_turn_usage
from ..model import ModelContextWindowError
from ..model import ModelError
from ..model import ModelSample
from ..model import retire_model


NOT_EXECUTED_OUTPUT = "Not executed: the model response did not complete."
OVERFLOW_NOTICE = "Model context window exceeded; compacting before one retry."


class SampleLimitExceeded(RuntimeError):
    """A turn reached its explicit per-turn sample limit."""


class MissingFinalText(RuntimeError):
    """A turn ended without nonblank final assistant text."""


class Interrupt(enum.Enum):
    """The interrupt's answer right before a sample."""

    CONTINUE = "continue"
    STOP = "stop"


@dataclass(frozen=True)
class Steer:
    """The interrupt's answer when the user entered text during the turn.

    The messages are saved as user messages, in order, right before the
    sample. Like messages that tools inject, a steer continues the current
    turn: no ``UserInteractionBoundary`` follows it, so provider turn
    continuity (such as the Responses turn ID and Codex turn state) is kept.
    """

    messages: Tuple[Message, ...]

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        if not messages or not all(isinstance(m, Message) and m.role == "user"
                                   for m in messages):
            raise ValueError("A steer requires one or more user messages.")
        object.__setattr__(self, "messages", messages)


TURN_KINDS = ("ended", "yielded", "stopped")


@dataclass(frozen=True)
class TurnResult:
    """How a turn ended without failing. Failures are raised instead."""

    kind: str
    final_text: Optional[str] = None
    note: Optional[str] = None  # a yielded turn's handoff note

    def __post_init__(self) -> None:
        if self.kind not in TURN_KINDS:
            raise ValueError(f"TurnResult kind must be one of {TURN_KINDS}.")
        if self.kind == "ended":
            if not isinstance(self.final_text, str) or not self.final_text.strip():
                raise ValueError("An ended turn requires nonblank final text.")
        elif self.final_text is not None:
            raise ValueError("Only an ended turn has final text.")
        if self.kind == "yielded":
            if not isinstance(self.note, str) or not self.note.strip():
                raise ValueError("A yielded turn requires a nonblank note.")
        elif self.note is not None:
            raise ValueError("Only a yielded turn has a note.")


STOPPED = TurnResult("stopped")


class TurnHost:
    """The caller-specific half of :func:`run_turn`.

    Subclasses must implement :meth:`append`; every other method has a
    default. All methods are called on the context thread.
    """

    def append(self, context, items: Sequence[InteractionItem]) -> None:
        """Extend the context with items and save it; raise if saving fails."""
        raise NotImplementedError

    def show(self, items) -> None:
        """Show display items."""

    def phase(self, phase: str) -> None:
        """Report the context's phase, such as "sampling"."""

    def tool_phase(self, call: ToolCall) -> None:
        """Report that one tool call is about to run."""
        self.phase("executing tools")

    def should_stop(self) -> bool:
        """True once the turn must start no new effects. May raise."""
        return False

    def interrupt(self):
        """Resume the interrupt right before a sample.

        Returns ``Interrupt.CONTINUE``, ``Interrupt.STOP``, or a :class:`Steer`
        with what the user entered since. The default only checks for a stop,
        as for a host with no top-level input (the watcher, the worker, the
        demo).
        """
        return Interrupt.STOP if self.should_stop() else Interrupt.CONTINUE

    def trace(self, op: str, **tags):
        """A context manager around each sample and compaction."""
        return contextlib.nullcontext()

    def notice(self, text: str) -> None:
        """Show a one-line host notice."""

    def after_sample(self, model, sample: ModelSample) -> None:
        """Called after each successful sample, before it is saved."""

    def retryable_failure(self) -> None:
        """Called before raising a failure that a retry could fix."""


def run_turn(context, model, environment, config, host: TurnHost, *,
             sample_params=None, control=None) -> TurnResult:
    """Run one turn on the context thread until it ends, stops, or fails.

    ``config`` is an ``InteractionConfig`` snapshot. Returns ``ended`` with
    the final text, ``yielded`` with a handoff note, or ``stopped``; raises on
    failure. The model is retired when the turn ends, however it ends.
    ``sample_params`` overrides ``config.sample_params()``.

    ``control`` (only supervised main passes one) lets a tool end the turn:
    after each batch is saved, ``control.request()`` returns the recorded
    handoff note, if any. The loop never inspects tool names or output.
    """
    try:
        return _run_turn(context, model, environment, config, host, sample_params, control)
    finally:
        retire_model(model)


def _run_turn(context, model, environment, config, host, sample_params, control):
    started = perf_counter()
    if sample_params is None:
        sample_params = config.sample_params()
    samples = 0
    # Pi's overflow recovery: one compact-and-retry per turn.
    overflow_recovered = False
    while config.max_samples is None or samples < config.max_samples:
        if host.should_stop():
            return STOPPED
        if auto_compaction_due(model, context, config):
            compact(context, model, environment, config, host, sample_params)
        answer = host.interrupt()
        if answer is Interrupt.STOP:
            return STOPPED
        if isinstance(answer, Steer):
            host.append(context, answer.messages)
            host.show(render_interaction_items(answer.messages))
        host.phase("sampling")
        try:
            with host.trace("sample", context_revision=len(context)):
                sample = model.sample(
                    context.copy(), tools=environment.tool_specs,
                    sample_params=sample_params,
                )
        except ModelError as exc:
            record_sample_failure(context, exc, host)
            if host.should_stop():
                # The stop caused the failure (e.g. it retired the model).
                return STOPPED
            if (isinstance(exc, ModelContextWindowError) and not overflow_recovered
                    and config.enable_auto_compaction and uses_host_auto_compaction(model)):
                overflow_recovered = True
                host.notice(OVERFLOW_NOTICE)
                try:
                    compacted = compact(context, model, environment, config, host,
                                        sample_params)
                except Exception:
                    host.retryable_failure()
                    raise
                if compacted:
                    continue
                # Nothing to compact: the sampling error stands.
            host.retryable_failure()
            raise
        except Exception:
            # Adapters normally raise ModelError, but a failure at this
            # sampling boundary is still distinct from a failed local effect.
            host.retryable_failure()
            raise
        # The failed attempt before an overflow retry does not count.
        samples += 1
        if not isinstance(sample, ModelSample):
            raise TypeError("Expected ModelSample.")
        host.after_sample(model, sample)
        host.append(context, sample.context_items())
        host.show(sample.display_items())
        if sample.stop_reason == "compaction":
            # A paused provider compaction has no final text; sample again.
            continue
        if not sample.tool_calls:
            text = sample.last_assistant_text
            if not text or not text.strip():
                host.retryable_failure()
                raise MissingFinalText("Model returned no final assistant text.")
            _save_summary(context, host, started)
            return TurnResult("ended", text)
        if not run_tool_calls(context, environment, sample.tool_calls, host):
            return STOPPED
        note = None if control is None else control.request()
        if note is not None:
            # A handoff takes effect only now, with the whole batch saved.
            _save_summary(context, host, started)
            return TurnResult("yielded", note=note)
    raise SampleLimitExceeded(
        f"Model did not produce a final answer within {config.max_samples} samples.")


def _save_summary(context, host, started) -> None:
    summary = summarize_turn_usage(context.items, elapsed_seconds=perf_counter() - started)
    host.append(context, (summary,))
    host.show(render_interaction_items((summary,)))


def compact(context, model, environment, config, host, sample_params) -> bool:
    """Install one automatic compaction; False when there is nothing to compact."""
    host.phase("compacting")
    compactor = create_default_compactor(model, config.compaction_settings())
    try:
        with host.trace("compact", context_revision=len(context)):
            result = compactor.compact(context.copy(), tools=environment.tool_specs,
                                       sample_params=sample_params)
    except NothingToCompact:
        return False
    if not isinstance(result, CompactionResult):
        raise TypeError("Expected CompactionResult.")
    host.append(context, result.context_items())
    host.show(result.display_items())
    return True


def record_sample_failure(context, exc: ModelError, host: TurnHost) -> None:
    """Durably close a failed sample: its completed output and failure.

    Completed tool calls are closed as not executed; they never run.
    """
    contribution = (*exc.completed_items,
                    *((exc.failure,) if exc.failure is not None else ()))
    if not contribution:
        return
    host.append(context, (*contribution, ModelSampleBoundary()))
    host.show(render_interaction_items(contribution))
    calls = tuple(item for item in exc.completed_items if isinstance(item, ToolCall))
    if calls:
        results = tuple(ToolResult(call.call_id, NOT_EXECUTED_OUTPUT, success=False)
                        for call in calls)
        host.append(context, results)
        host.show(render_interaction_items(results, source_calls=calls))


def run_tool_calls(context, environment, calls: Iterable[ToolCall],
                   host: TurnHost) -> bool:
    """Run one batch, one call at a time; False if a stop came first.

    Each result is saved and shown as it arrives. User messages that tools
    inject wait until every call has a result. On a stop, calls that have not
    started stay unanswered; the next use of the context closes them.
    """
    held: list[Message] = []
    for call in calls:
        if host.should_stop():
            return False
        host.tool_phase(call)
        result = environment.execute_tool_calls((call,))
        host.append(context, result.items)
        host.show(render_interaction_items(result.items, source_calls=(call,)))
        held.extend(result.user_messages)
    if held:
        host.append(context, tuple(held))
        host.show(render_interaction_items(tuple(held)))
    return True
