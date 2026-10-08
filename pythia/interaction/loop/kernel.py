"""The shared inner turn loop: sample the model, run its tool calls, repeat.

One copy serves every caller: the CLI, auto's roles, and the demo. What
differs between them lives in a :class:`TurnHost`: how items are saved and
shown, the phase, stopping, tracing, and a few hooks. Every host method is
called on the context thread, the one thread that runs this context's model,
tools, and saves.

Right before each sample the loop resumes the host's *interrupt*, the logical
coroutine that stands for the user's top-level input. Its blocking case is
the caller's quiescent wait between tasks; here it answers at once.

Steers enter a turn only there. Flushed steers (``/steer!``) get the turn there
sooner: the loop starts no new sample or tool call before it delivers them,
and closes the calls they kept from starting. Each sample and tool call is
*preemptible*: a preempting steer or stop can cancel it from another thread
(see ``preemption``), where the model or tool supports that.

A stop ends the turn at one of three levels (``Urgency``): at the next
interrupt point, after the tool batch (``stop_requested``); once the operation
in flight ends (``should_stop``); or by also cancelling it.
"""

from __future__ import annotations

import contextlib
import enum
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Iterable, Optional, Sequence, Tuple

from ..compaction import CompactionResult
from ..compaction import NothingToCompact
from ..compaction import auto_compaction_due
from ..compaction import create_default_compactor
from ..compaction import uses_host_auto_compaction
from ..display import render_interaction_items
from ..environment import CancelToken
from ..environment import cancel_scope
from ..items import InteractionItem
from ..items import Message
from ..items import ModelSampleBoundary
from ..items import ToolCall
from ..items import ToolResult
from ..items import summarize_turn_usage
from ..model import ModelContextWindowError
from ..model import ModelContinuationExpired
from ..model import ModelError
from ..model import ModelSample
from ..model import retire_model
from .preemption import NOT_PREEMPTIBLE


NOT_EXECUTED_OUTPUT = "Not executed: the model response did not complete."
SKIPPED_OUTPUT = "Not executed: the user sent a message before this call started."
OVERFLOW_NOTICE = "Model context window exceeded; compacting before one retry."
CONTINUATION_NOTICE = "Model continuation expired; restarting from saved tool results (1/1)."
CONTINUATION_RECOVERY = "continuation_expired_cold_restart"


class SampleLimitExceeded(RuntimeError):
    """A turn reached its explicit per-turn sample limit."""


class MissingFinalText(RuntimeError):
    """A turn ended without nonblank final assistant text."""


class _CompactionStopped(Exception):
    """A compaction failed while the host was stopping (the stop caused it)."""


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
        """True once the turn must start no new effects. May raise.

        That is a stop that waits only for the operation in flight, or also
        cancels it (levels 2 and 3). The loop checks it before each tool call.
        """
        return False

    def stop_requested(self) -> bool:
        """True once a stop at any level is requested. May raise.

        The turn then samples no more: it ends at its next interrupt point,
        though it may finish the tool batch in flight first (level 1, unless
        :meth:`should_stop` is true too). The default is :meth:`should_stop`,
        for a host whose every stop is at level 2 or above.
        """
        return self.should_stop()

    def interrupt(self):
        """Resume the interrupt right before a sample.

        Returns ``Interrupt.CONTINUE``, ``Interrupt.STOP``, or a :class:`Steer`
        with what the user entered since. The default only checks for a stop,
        as for a host with no top-level input (the watcher, the worker, the
        demo). A host with flushed steers resets their urgency here
        (``Preemption.take``), since :meth:`should_steer` must turn false.
        """
        return Interrupt.STOP if self.stop_requested() else Interrupt.CONTINUE

    def should_steer(self) -> bool:
        """True while flushed steers wait for the interrupt point.

        The turn then starts no new sample or tool call, closes the calls
        that have not started, and goes to the interrupt point. The default
        host has no steers. It must be false once a stop is requested, which
        drops the steers.
        """
        return False

    def preemptible(self, cancel):
        """Bracket one sample or tool call; ``cancel`` (or None) ends it early.

        Returns a context manager whose ``skipped`` is true when the operation
        must not start (input came after the interrupt point), and whose
        ``cancelled`` is true once ``cancel`` was started from another thread.
        The default never skips or cancels.
        """
        return NOT_PREEMPTIBLE

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
    continuation_recovered = False
    recovery_pending = False
    while config.max_samples is None or samples < config.max_samples:
        if host.stop_requested():
            return STOPPED
        if auto_compaction_due(model, context, config):
            try:
                compact(context, model, environment, config, host, sample_params)
            except _CompactionStopped:
                return STOPPED
        answer = host.interrupt()
        if answer is Interrupt.STOP:
            return STOPPED
        if isinstance(answer, Steer):
            host.append(context, answer.messages)
            host.show(render_interaction_items(answer.messages))
        operation = host.preemptible(_retirer(model))
        try:
            with operation:
                if operation.skipped:
                    continue  # input came after the interrupt point: go back to it
                host.phase("sampling")
                with host.trace("sample", context_revision=len(context)):
                    sample = model.sample(
                        context.copy(), tools=environment.tool_specs,
                        sample_params=sample_params,
                    )
        except ModelError as exc:
            record_sample_failure(context, exc, host)
            if host.stop_requested():
                # The stop caused the failure (e.g. it retired the model), or
                # the turn may sample no more: no retry or recovery.
                return STOPPED
            if operation.cancelled:
                # A preempting steer cancelled it: not a failure, and not
                # counted. The steer is delivered at the interrupt point.
                continue
            if isinstance(exc, ModelContinuationExpired):
                ready = exc.failure is not None and not exc.completed_items
                try:
                    context.assert_model_ready()
                except ValueError:
                    ready = False
                allowed = ready and not continuation_recovered
                with host.trace("continuation_recovery", context_revision=len(context),
                                allowed=allowed, error_code=exc.failure.error_code if exc.failure else None):
                    if allowed:
                        continuation_recovered = True
                        recovery_pending = True
                        host.notice(CONTINUATION_NOTICE)
                if allowed:
                    # The provider already revoked the old continuation. Repeat
                    # only sampling, through the normal stop/steering boundary;
                    # never re-enter the previous host tool batch.
                    continue
            if (isinstance(exc, ModelContextWindowError) and not overflow_recovered
                    and config.enable_auto_compaction and uses_host_auto_compaction(model)):
                overflow_recovered = True
                host.notice(OVERFLOW_NOTICE)
                try:
                    compacted = compact(context, model, environment, config, host,
                                        sample_params)
                except _CompactionStopped:
                    return STOPPED
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
        if recovery_pending:
            sample = replace(sample, recovery=(*sample.recovery, CONTINUATION_RECOVERY))
            recovery_pending = False
        host.after_sample(model, sample)
        host.append(context, sample.context_items())
        host.show(sample.display_items())
        if sample.stop_reason == "compaction":
            # A paused provider compaction has no final text; sample again.
            continue
        if host.should_steer() and (config.max_samples is None
                                    or samples < config.max_samples):
            # Flushed steers wait: close this sample's calls unstarted and go
            # to the interrupt point. After a final answer too: the user
            # steered this turn, so it goes on, if another sample is allowed.
            skip_calls(context, sample.tool_calls, host)
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
    if host.stop_requested():
        # The last allowed sample's batch finished under a stop: that is the
        # stop the user asked for, not a failure.
        return STOPPED
    raise SampleLimitExceeded(
        f"Model did not produce a final answer within {config.max_samples} samples.")


def _save_summary(context, host, started) -> None:
    summary = summarize_turn_usage(context.items, elapsed_seconds=perf_counter() - started)
    host.append(context, (summary,))
    host.show(render_interaction_items((summary,)))


def compact(context, model, environment, config, host, sample_params) -> bool:
    """Install one automatic compaction; False when there is nothing to compact.

    A compaction that fails while the host is stopping (as when a stop retires
    the model) raises ``_CompactionStopped``, which the turn loop treats as a
    stop, not a failure.
    """
    host.phase("compacting")
    compactor = create_default_compactor(model, config.compaction_settings())
    try:
        with host.trace("compact", context_revision=len(context)):
            result = compactor.compact(context.copy(), tools=environment.tool_specs,
                                       sample_params=sample_params)
    except NothingToCompact:
        return False
    except Exception:
        if host.stop_requested():
            raise _CompactionStopped() from None
        raise
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

    Each result is saved and shown as it arrives, and each call runs with a
    cancel token (``cancel_scope``) that a preempting steer or stop can set.
    Flushed steers close the calls that have not started (``skip_calls``).
    User messages that tools inject wait until every call has a result. On a
    stop at level 2 or 3 (``should_stop``), calls that have not started stay
    unanswered; the next use of the context closes them. A stop at level 1
    lets the batch finish.
    """
    calls = tuple(calls)
    held: list[Message] = []
    for index, call in enumerate(calls):
        if host.should_stop():
            return False
        with CancelToken() as token:
            operation = host.preemptible(token.cancel)
            with operation:
                if not operation.skipped:
                    host.tool_phase(call)
                    with cancel_scope(token):
                        result = environment.execute_tool_calls((call,))
        if operation.skipped:
            if host.should_stop():
                return False  # a stop, not a steer
            skip_calls(context, calls[index:], host)
            break
        host.append(context, result.items)
        host.show(render_interaction_items(result.items, source_calls=(call,)))
        held.extend(result.user_messages)
    if held:
        host.append(context, tuple(held))
        host.show(render_interaction_items(tuple(held)))
    return True


def skip_calls(context, calls: Iterable[ToolCall], host: TurnHost) -> None:
    """Close tool calls that a steer kept from starting; they never ran."""
    calls = tuple(calls)
    if not calls:
        return
    results = tuple(ToolResult(call.call_id, SKIPPED_OUTPUT, success=False) for call in calls)
    host.append(context, results)
    host.show(render_interaction_items(results, source_calls=calls))


def _retirer(model):
    """A sample's cancel: retire the model's live continuation, if it has one."""
    if callable(getattr(model, "retire", None)):
        return lambda: retire_model(model)
    return None
