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

Compaction is not preemptible. A steer waits for it, since the interrupt point
comes right after it and the sample the steer goes to needs it; a steer never
skips or cancels it. A preempting stop reaches it through the model instead:
the apps close their models, which cancels the request in flight and refuses
the next one, and a compaction that fails during a stop ends the turn as
stopped. A compaction runs the config's sequence of methods
(``compaction.run_compaction``); each failed attempt is saved as a
``ModelFailure`` before the next one.

A stop ends the turn at one of three levels (``Urgency``): at the next
interrupt point, after the tool batch (``stop_requested``); once the operation
in flight ends (``should_stop``); or by also cancelling it.

A reboot (``reboot_requested``) also ends the turn at the interrupt point, but
only once the steers taken there are saved, so that the restarted program,
which continues the turn, has them. The CLI reboots for ``/reboot``, and after
a coldstart compaction with ``compaction_coldstart_reboot`` (``compacted``).

A failed sample that the model vouches for (``RetryState.ELIGIBLE``: a
transient HTTP failure, a rate limit, a relay's rejected tool name or expired
continuation) is sampled again under an automatic-retry policy, after a
notice and a wait each time (``_auto_retry``). So the loop's own samples ask
HTTP adapters for no retries of their own (``SampleParams.transient_retries``);
compaction keeps them.
"""

from __future__ import annotations

import contextlib
import enum
import threading
from dataclasses import dataclass, replace
from time import monotonic, perf_counter
from typing import Iterable, Optional, Sequence, Tuple

from ..compaction import CompactionResult
from ..compaction import NothingToCompact
from ..compaction import auto_compaction_due
from ..compaction import compaction_log
from ..compaction import create_default_compactor
from ..compaction import run_compaction
from ..compaction import uses_host_auto_compaction
from ..display import InteractionItemRenderer
from ..display import render_interaction_items
from ..environment import CancelToken
from ..environment import cancel_scope
from ..environment import EnvironmentResult
from ..environment import results_due_scope
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
from ..model import ModelRateLimitError
from ..model import RetryState
from ..model import SampleParams
from ..model import park_model
from ..model import retire_model
from ..model_catalog import DEFAULT_RATE_LIMIT_RETRIES
from ..model_catalog import RATE_LIMIT_RETRIES_UNBOUNDED
from .preemption import NOT_PREEMPTIBLE


NOT_EXECUTED_OUTPUT = "Not executed: the model response did not complete."
SKIPPED_OUTPUT = "Not executed: the user sent a message before this call started."
OVERFLOW_NOTICE = "Model context window exceeded; compacting before one retry."
CONTINUATION_RECOVERY = "continuation_expired_cold_restart"
RATE_LIMIT_RECOVERY = "rate_limit_retry"
# Before each automatic retry: its wait, then how many of the sample's
# attempts failed so far (also the number of the retry to come), of its
# attempts in all.
AUTO_RETRY_NOTICE = "Sampling again in {delay} s ({failed}/{attempts})."
CONTINUATION_NOTICE = ("Model continuation expired; restarting from saved tool results "
                       "in {delay} s ({failed}/{attempts}).")
# A rate limit's waits, in whole seconds, before its 1st, 2nd... retry; the
# last repeats. A provider's Retry-After isn't followed yet: it is traced.
RATE_LIMIT_DELAYS_SECONDS = (2, 2, 5, 10, 20, 30)
# How often a retry's wait checks whether something should end it early.
_WAIT_POLL_SECONDS = 0.25


@dataclass(frozen=True)
class AutoRetryPolicy:
    """How the turn loop samples again, by itself, after an eligible failure
    (``RetryState.ELIGIBLE``): an automatic retry, unlike the user's /retry.

    ``attempts`` counts a sample's attempts under this policy in all, the
    first included; None is unbounded. ``delays`` are the waits, in whole
    seconds, before the 1st, 2nd... retry; the last repeats. ``label`` is the
    recovery label of each retry; None derives it from the failure
    (``_auto_retry``).
    """

    name: str
    attempts: Optional[int]
    delays: Tuple[int, ...]
    label: Optional[str] = None


# Every eligible failure but a rate limit: a transient HTTP failure (a
# retryable status, a timeout, a connection error, a broken stream), and the
# Claude relay's (a rejected tool name, an expired continuation).
DEFAULT_AUTO_RETRY = AutoRetryPolicy("default", 3, (2,))
# A frontend shows a failure's error when the turn ends, which a long retry,
# such as a rate limit's, may put off for minutes, or forever when unbounded.
# So from the attempt where the default policy gives up, each failed attempt
# that the loop retries shows its error too (TurnHost.show_error): a bounded
# retry's display is then the unbounded one's, cut short where its attempts
# end. The default policy itself gives up there, so it shows the error only
# when the turn ends, as before.
SHOW_ERRORS_FROM_ATTEMPT = DEFAULT_AUTO_RETRY.attempts


def auto_retry_policy(exc: ModelError, config) -> AutoRetryPolicy:
    """The policy of an eligible failure under ``config`` (a snapshot).

    A rate limit (``ModelRateLimitError``) gets the config's
    ``endpoint_rate_limit_retries`` (``DEFAULT_RATE_LIMIT_RETRIES`` when it
    has none), which counts every retry, and RATE_LIMIT_DELAYS_SECONDS; any
    other failure, the default.
    """
    if not isinstance(exc, ModelRateLimitError):
        return DEFAULT_AUTO_RETRY
    retries = getattr(config, "endpoint_rate_limit_retries", None)
    if retries is None:
        retries = DEFAULT_RATE_LIMIT_RETRIES
    unbounded = retries == RATE_LIMIT_RETRIES_UNBOUNDED
    return AutoRetryPolicy(
        "rate_limit", None if unbounded else retries + 1, RATE_LIMIT_DELAYS_SECONDS,
        label=RATE_LIMIT_RECOVERY,
    )


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
    The turn loop shows them as they are saved.
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

    # How show()'s items render the model's samples: a host may show more of
    # their tool calls (InteractionItemRenderer's cwd and awaits_approval).
    renderer: InteractionItemRenderer = InteractionItemRenderer()

    def phase(self, phase: str) -> None:
        """Report the context's phase, such as "sampling"."""

    def tool_phase(self, call: ToolCall) -> None:
        """Report that one tool call is about to run."""
        self.phase("executing tools")

    def tool_approval(self, call: ToolCall) -> Optional[ToolResult]:
        """Approve one tool call before its environment runs.

        Return ``None`` to execute the call normally. Return a ``ToolResult``
        to use that result instead: the environment is not called and the
        result is not traced as a tool operation. This is normally a failed
        result naming ``call.call_id``.
        """
        return None

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

    def reboot_requested(self) -> bool:
        """True once the host will restart its program on the current code.

        The turn then ends at its next interrupt point, as ``stopped``, right
        after it saves the steers taken there: the restarted program continues
        the turn from the save. Unlike a stop, a reboot doesn't drop the
        steers. It lets the tool batch in flight finish, unless the host also
        flushed the turn's operations (``preemptible``) to end it sooner, as
        the CLI's /reboot! and /reboot!! do. The default host never reboots.
        """
        return False

    def skipped_output(self) -> str:
        """The result of a tool call that a flush kept from starting
        (``skip_calls``): by default, that the user sent a message."""
        return SKIPPED_OUTPUT

    def compacted(self, result: CompactionResult) -> None:
        """Called after an automatic compaction is saved and shown.

        It may request a reboot (:meth:`reboot_requested`), as the CLI does
        after a coldstart compaction with ``compaction_coldstart_reboot``.
        """

    def preemptible(self, cancel):
        """Bracket one sample or tool call; ``cancel`` (or None) ends it early.

        Returns a context manager whose ``skipped`` is true when the operation
        must not start (input came after the interrupt point), and whose
        ``cancelled`` is true once ``cancel`` was started from another thread.
        The default never skips or cancels.
        """
        return NOT_PREEMPTIBLE

    def trace(self, op: str, **tags):
        """A context manager around each sample, tool call, and compaction."""
        return contextlib.nullcontext()

    def log_path(self):
        """The save file of this context, which coldstart compaction points to.

        None (the default) when the context has no save.
        """
        return None

    def compaction_log(self, environment):
        """The log that coldstart compaction points to (``compaction.CompactionLog``), or None.

        The default is ``log_path``'s, with ``environment``'s working
        directory. A host whose save keeps its full log apart from the log
        that the pointer names (the CLI's public log) returns its own.
        """
        return compaction_log(self.log_path(), environment)

    def notice(self, text: str) -> None:
        """Show a one-line host notice."""

    def after_sample(self, model, sample: ModelSample) -> None:
        """Called after each successful sample, before it is saved."""

    def retryable_failure(self) -> None:
        """Called before raising a failure that a retry could fix."""

    def retry_cancelled(self) -> bool:
        """True once the user stopped the automatic retries (the CLI's /cancel).

        The loop asks before each automatic retry and during its wait; the
        failure then ends the turn, ``RetryState.CANCELLED``. The host resets
        it, as the CLI does when a sample succeeds or a turn starts. The
        default host never cancels.
        """
        return False

    def show_error(self, exc: ModelError) -> None:
        """Show the error of a failed attempt that the loop retries, as the
        frontend shows a failed turn's: from the 3rd attempt on, which only a
        rate limit's retries reach (SHOW_ERRORS_FROM_ATTEMPT). The default
        shows nothing, as auto withholds error details."""

    def retry_progress(self, failed: int, attempts: Optional[int]) -> None:
        """Before each automatic retry, with its notice: ``failed`` of the
        sample's ``attempts`` (None: unbounded) failed so far. A frontend may
        show it with its status, while the loop waits and while the retried
        sample runs, until a sample succeeds (``after_sample``) or the turn
        ends. The default shows nothing."""


def run_turn(context, model, environment, config, host: TurnHost, *,
             sample_params=None, control=None) -> TurnResult:
    """Run one turn on the context thread until it ends, stops, or fails.

    ``config`` is an ``InteractionConfig`` snapshot. Returns ``ended`` with
    the final text, ``yielded`` with a handoff note, or ``stopped``; raises on
    failure. The model is retired when the turn stops or fails. After a final
    answer it may keep its work for the next turn, as the Claude relay keeps
    its native process (doc/claude-relay-warm-delivery-plan.md); after a yield
    it is parked (``park_model``): the relay's process waits for the batch's
    results, which the turn that resumes delivers
    (doc/claude-relay-warm-yield-plan.md).
    ``sample_params`` overrides ``config.sample_params()``.

    ``control`` (only supervised main passes one) lets a tool end the turn:
    after each batch is saved, ``control.request()`` returns the recorded
    handoff note, if any. The loop never inspects tool names or output.
    """
    try:
        result = _run_turn(context, model, environment, config, host, sample_params, control)
    except BaseException:
        retire_model(model)
        raise
    if result.kind == "yielded":
        park_model(model)
    elif result.kind != "ended":
        retire_model(model)
    return result


def _run_turn(context, model, environment, config, host, sample_params, control):
    started = perf_counter()
    if sample_params is None:
        sample_params = config.sample_params()
    # The loop's own samples: an HTTP adapter retries no transient failure
    # itself, since the loop retries it (_auto_retry), visibly. Compaction
    # gets the turn's params, and keeps the adapters' retries.
    own_params = sample_params
    if isinstance(sample_params, SampleParams) and sample_params.transient_retries is None:
        own_params = replace(sample_params, transient_retries=0)
    samples = 0
    # Pi's overflow recovery: one compact-and-retry per turn.
    overflow_recovered = False
    # The current sample's attempt under each automatic-retry policy, by name
    # (_auto_retry), and the recovery labels of its retries, in order, for
    # the metadata of the attempt that succeeds.
    attempts = {}
    recovery = []
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
        if host.reboot_requested():
            return STOPPED  # the restarted program continues the turn
        operation = host.preemptible(_retirer(model))
        try:
            with operation:
                if operation.skipped:
                    continue  # input came after the interrupt point: go back to it
                host.phase("sampling")
                with host.trace("sample", context_revision=len(context)):
                    sample = model.sample(
                        context.copy(), tools=environment.tool_specs,
                        sample_params=own_params,
                    )
        except ModelError as exc:
            # The central handler of failed samples, for every frontend and
            # model: each adapter raises a ModelError subclass (model.py), and
            # the Claude relay its own too (claude_relay/), unwrapped on the
            # way here. Each failure is saved first; then a stop ends the turn,
            # and the cases that are safe to recover sample again: a preempting
            # steer; an error the model vouches for (RetryState.ELIGIBLE, such
            # as a transient HTTP failure, a rate limit, an expired
            # continuation or a rejected tool name), within its policy's
            # attempts (_auto_retry); an overflow. The rest end the turn with
            # /retry offered. Compaction retries its own requests
            # (run_compaction).
            record_sample_failure(context, exc, host)
            if host.stop_requested():
                # The stop caused the failure (e.g. it retired the model), or
                # the turn may sample no more: no retry or recovery.
                return STOPPED
            if operation.cancelled:
                # A preempting steer cancelled it: not a failure, and not
                # counted. The steer is delivered at the interrupt point.
                continue
            if exc.retry_state is RetryState.ELIGIBLE:
                policy = auto_retry_policy(exc, config)
                attempt = attempts.get(policy.name, 1)
                label = _auto_retry(context, exc, host, policy, attempt)
                if label is not None:
                    # The model already dropped the failed attempt (as the
                    # relay retires its process). Repeat only sampling, through
                    # the normal stop/steering boundary; never re-enter the
                    # previous host tool batch.
                    attempts[policy.name] = attempt + 1
                    recovery.append(label)
                    continue
            if (isinstance(exc, ModelContextWindowError) and not overflow_recovered
                    and config.enable_auto_compaction and uses_host_auto_compaction(model)):
                overflow_recovered = True
                host.notice(OVERFLOW_NOTICE)
                try:
                    compacted = compact(context, model, environment, config, host,
                                        sample_params, reason="overflow")
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
        # The failed attempts before a retry do not count.
        samples += 1
        attempts = {}
        if not isinstance(sample, ModelSample):
            raise TypeError("Expected ModelSample.")
        if recovery:
            # In order: the loop's retries came before this attempt's own
            # recovery (such as a Codex credential reload).
            sample = replace(sample, recovery=(*recovery, *sample.recovery))
            recovery = []
        host.after_sample(model, sample)
        host.append(context, sample.context_items())
        host.show(sample.display_items(host.renderer))
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
        if not run_tool_calls(context, environment, sample.tool_calls, host,
                              results_due=sample.results_due):
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


def compact(context, model, environment, config, host, sample_params, *,
            reason="threshold") -> bool:
    """Install one automatic compaction; False when there is nothing to compact.

    It runs the config's compaction sequence (``run_compaction``), for the
    ``threshold`` or for ``overflow`` recovery. Each failed attempt that the
    sequence moves past is saved as a ``ModelFailure``, shown, and announced
    with a notice. It is not a preemptible operation (see the module
    docstring). A compaction that fails while the host is stopping (as when a
    preempting stop closes the model) raises ``_CompactionStopped``, which the
    turn loop treats as a stop, not a failure.
    A compaction that succeeds is reported to ``host.compacted``.
    """
    host.phase("compacting")

    def failed(report):
        host.append(context, (report.failure,))
        host.show(render_interaction_items((report.failure,)))
        host.notice(report.notice())

    settings = config.compaction_settings()
    # Only coldstart points to the save.
    log = (host.compaction_log(environment)
           if any(method == "coldstart" for method, _ in settings.methods) else None)
    try:
        with host.trace("compact", context_revision=len(context)):
            result = run_compaction(
                context, model, settings,
                factory=create_default_compactor, tools=environment.tool_specs,
                sample_params=sample_params, reason=reason, log=log,
                stop_requested=host.stop_requested, on_failure=failed,
            )
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
    host.compacted(result)
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
    host.show(host.renderer.render_items(contribution))
    calls = tuple(item for item in exc.completed_items if isinstance(item, ToolCall))
    if calls:
        results = tuple(ToolResult(call.call_id, NOT_EXECUTED_OUTPUT, success=False)
                        for call in calls)
        host.append(context, results)
        host.show(render_interaction_items(results, source_calls=calls))


def _auto_retry(context, exc: ModelError, host: TurnHost, policy: AutoRetryPolicy,
                attempt: int) -> Optional[str]:
    """Settle an eligible failure (``RetryState.ELIGIBLE``) of the sample's
    ``attempt`` under ``policy`` (1 for its first), after it is saved.

    The failure is RETRIED: after its notice and the wait
    (``_wait_to_auto_retry``), this returns the recovery label of the retry.
    The notice names the wait, and ``attempt``, the attempts failed so far, of
    the policy's attempts, which the host gets too (``TurnHost.retry_progress``).
    From the 3rd attempt on, its error shows before the notice
    (``TurnHost.show_error``), since a long retry puts off the turn's end,
    where it would show. Otherwise this returns None, and the turn ends with
    /retry offered: the failure is BLOCKED when the loop's own checks fail (no
    failure was saved, it has completed output, or the context isn't ready
    for the model); CANCELLED when the user stopped the automatic retries
    (``TurnHost.retry_cancelled``), before the wait or during it; EXHAUSTED
    when the policy's attempts are used up. Each decision is traced, with a
    rate limit's Retry-After, which the wait doesn't follow yet.
    """
    ready = exc.failure is not None and not exc.completed_items
    if ready:
        try:
            context.assert_model_ready()
        except ValueError:
            ready = False
    retry_after = getattr(exc, "retry_after_seconds", None)
    delay = policy.delays[min(attempt, len(policy.delays)) - 1]
    if not ready:
        settled = RetryState.BLOCKED
    elif host.retry_cancelled():
        settled = RetryState.CANCELLED
    elif policy.attempts is not None and attempt >= policy.attempts:
        settled = RetryState.EXHAUSTED
    else:
        settled = None  # retried, once the wait ends
    expired = isinstance(exc, ModelContinuationExpired)
    error_code = exc.failure.error_code if exc.failure else None
    with host.trace("sample_retry", context_revision=len(context), policy=policy.name,
                    attempt=attempt, error_code=error_code, retry_after_seconds=retry_after,
                    retry_state=(RetryState.RETRIED if settled is None else settled).value,
                    delay_seconds=delay if settled is None else None):
        if settled is not None:
            exc.retry_state = settled
            return None
        if attempt >= SHOW_ERRORS_FROM_ATTEMPT:
            host.show_error(exc)
        host.notice((CONTINUATION_NOTICE if expired else AUTO_RETRY_NOTICE).format(
            delay=delay, failed=attempt,
            attempts="∞" if policy.attempts is None else policy.attempts))
        host.retry_progress(attempt, policy.attempts)
        _wait_to_auto_retry(host, delay)
    if host.retry_cancelled():
        # During the wait: no retry after all.
        with host.trace("sample_retry", context_revision=len(context), policy=policy.name,
                        attempt=attempt, error_code=error_code,
                        retry_after_seconds=retry_after,
                        retry_state=RetryState.CANCELLED.value, delay_seconds=None):
            exc.retry_state = RetryState.CANCELLED
        return None
    exc.retry_state = RetryState.RETRIED
    if expired:
        return CONTINUATION_RECOVERY
    if policy.label is not None:
        return policy.label
    if exc.failure.http_status is not None:
        # A transient HTTP status, named as an adapter's own retry names it.
        return f"http_{exc.failure.http_status}_retry"
    # The failure's own code (or category), bounded like every recovery label.
    return f"{(exc.failure.error_code or exc.failure.category)[:240]}_retry"


def run_tool_calls(context, environment, calls: Iterable[ToolCall],
                   host: TurnHost, *, results_due: Optional[float] = None) -> bool:
    """Run one batch, one call at a time; False if a stop came first.

    Each result is saved and shown as it arrives, and each call runs with a
    cancel token (``cancel_scope``) that a preempting steer or stop can set.
    Before execution, a host may return its result from ``tool_approval``;
    that replacement is not traced and does not call the environment.
    Each call also runs with the batch's ``results_due`` (``results_due_scope``),
    when the sample's model needs the results (``ModelSample.results_due``),
    and is traced as a ``tool_call`` operation.
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
                    approved = host.tool_approval(call)
                    if approved is not None:
                        if not isinstance(approved, ToolResult):
                            raise TypeError(
                                "tool_approval must return ToolResult or None, got "
                                f"{type(approved).__name__}"
                            )
                        if approved.call_id != call.call_id:
                            raise ValueError(
                                "tool_approval's result does not name the pending call: "
                                f"{approved.call_id!r} != {call.call_id!r}"
                            )
                        result = EnvironmentResult((approved,), ())
                    else:
                        with host.trace("tool_call", context_revision=len(context),
                                        call_id=call.call_id, tool=call.name,
                                        results_due_in_seconds=_seconds_until(results_due)), \
                                cancel_scope(token), results_due_scope(results_due):
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
    """Close tool calls that a flush (a steer, or the CLI's /reboot!) kept from
    starting; they never ran. Their result is ``host.skipped_output()``."""
    calls = tuple(calls)
    if not calls:
        return
    output = host.skipped_output()
    results = tuple(ToolResult(call.call_id, output, success=False) for call in calls)
    host.append(context, results)
    host.show(render_interaction_items(results, source_calls=calls))


def _seconds_until(due: Optional[float]) -> Optional[float]:
    """A trace field: the seconds left before ``due`` (negative once past it)."""
    return None if due is None else round(due - monotonic(), 3)


def _retirer(model):
    """A sample's cancel: retire the model's work in flight, if it can be retired.

    That is a Claude relay continuation, or an HTTP model's request.
    """
    if callable(getattr(model, "retire", None)):
        return lambda: retire_model(model)
    return None


def _wait_to_auto_retry(host: TurnHost, seconds: float) -> None:
    """Wait ``seconds`` before sampling again, in the phase "retrying".

    The wait is a preemptible operation, as a sample is: a flush or stop at
    IMMEDIATE or above that came first skips it, and one at PREEMPT ends it
    through its cancel. Unlike a sample, it also ends early, within
    ``_WAIT_POLL_SECONDS``, once nothing would come of waiting on: a stop or a
    reboot at any level, flushed steers (``should_steer``), or the user's
    cancel (``retry_cancelled``). A plain steer waits, then goes in before the
    next attempt. A host that doesn't preempt (the default) waits it out.
    """
    woken = threading.Event()
    operation = host.preemptible(woken.set)
    with operation:
        if operation.skipped:
            return
        host.phase("retrying")
        deadline = monotonic() + seconds
        while not woken.wait(max(0.0, min(deadline - monotonic(), _WAIT_POLL_SECONDS))):
            if (monotonic() >= deadline or host.stop_requested() or host.reboot_requested()
                    or host.should_steer() or host.retry_cancelled()):
                return
