"""run_turn with a scripted model and a host that records every call."""

from __future__ import annotations

import contextlib
import select
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from pythia.interaction import Environment, Init, InteractionContext, Message
from pythia.interaction import ModelSample, Tool, ToolCall, ToolOutcome, ToolResult, ToolSpec
from pythia.interaction.environment import current_cancel
from pythia.interaction.items import ModelFailure, ModelSampleBoundary, SampleMetadata, TurnSummary
from pythia.interaction.loop import Interrupt, MissingFinalText, Preemption, SampleLimitExceeded, Steer
from pythia.interaction.loop import TurnHost, TurnResult, run_turn
from pythia.interaction.loop import Urgency, command_urgency, escalated_stop
from pythia.interaction.loop import kernel
from pythia.interaction.model import ModelContinuationExpired, ModelContextWindowError, ModelTransportError
from pythia.interaction.runtime_config import InteractionConfigSnapshot


def _call(call_id, name="probe"):
    return ToolCall(name=name, call_id=call_id, arguments_json="{}")


def _answer(text):
    return ModelSample((Message("assistant", text),))


def _calls(*calls):
    return ModelSample(tuple(calls))


def _expired():
    return ModelContinuationExpired('expired', failure=ModelFailure(
        'ModelContinuationExpired', 'Native MCP operation expired', provider='claude-relay',
        error_code='native_mcp_timeout'))


class _Model:
    def __init__(self, host, *script):
        self.host, self.script, self.retired, self.threads = host, list(script), 0, set()

    def sample(self, context, *, tools=(), sample_params=None):
        self.threads.add(threading.get_ident())
        self.host.events.append(("sample", len(context)))
        step = self.script.pop(0)
        if callable(step):
            step = step()
        if isinstance(step, BaseException):
            raise step
        return step

    def retire(self):
        self.retired += 1


class _Host(TurnHost):
    def __init__(self, stop_after_calls=None):
        self.events, self.stop, self.retryable, self.notices = [], False, 0, []
        self.stop_after_calls, self.calls_run, self.threads = stop_after_calls, 0, set()

    def append(self, context, items):
        self.threads.add(threading.get_ident())
        items = tuple(items)
        context.extend(items)
        self.events.append(("append", tuple(type(i).__name__ for i in items)))

    def show(self, items):
        self.events.append(("show", len(tuple(items))))

    def phase(self, phase):
        self.events.append(("phase", phase))

    def should_stop(self):
        if self.stop_after_calls is not None and self.calls_run >= self.stop_after_calls:
            self.stop = True
        return self.stop

    def interrupt(self):
        self.events.append(("interrupt",))
        return super().interrupt()

    def trace(self, op, **tags):
        events = self.events

        class _Trace:
            def __enter__(self):
                events.append(("trace", op))

            def __exit__(self, *exc):
                events.append(("trace-end", op))
        return _Trace()

    def notice(self, text):
        self.notices.append(text)

    def retryable_failure(self):
        self.retryable += 1


def _environment(host, *, inject=None):
    def handler(arguments, *, timeout_seconds=None):
        host.calls_run += 1
        host.events.append(("tool", host.calls_run))
        messages = ()
        if inject is not None and host.calls_run == inject:
            messages = (Message("user", "injected"),)
        return ToolOutcome(output=f"ran {host.calls_run}", user_messages=messages)
    return Environment((Tool(ToolSpec("probe", "Probe.", {"type": "object"}), handler),))


def _context():
    return InteractionContext((Init(model="scripted"), Message("user", "task")))


def _config(**values):
    return InteractionConfigSnapshot(**values)


class RunTurnTests(unittest.TestCase):
    def test_continuation_recovery_retries_sampling_not_tools(self):
        host, context = _Host(), _context()
        model = _Model(host, _calls(_call('a')), _expired(), _answer('recovered'))
        result = run_turn(context, model, _environment(host), _config(max_samples=2), host)
        self.assertEqual(result.final_text, 'recovered')
        self.assertEqual(host.calls_run, 1)
        self.assertEqual(host.notices, [kernel.CONTINUATION_NOTICE])
        self.assertEqual(host.retryable, 0)
        metadata = [i for i in context if isinstance(i, SampleMetadata)]
        self.assertEqual(metadata[-1].recovery, (kernel.CONTINUATION_RECOVERY,))
        self.assertEqual(metadata[-1].request_attempts, 1)
        order = [e for e in host.events if e[0] in ('append', 'sample', 'tool')]
        failure = next(n for n,e in enumerate(order) if e[0] == 'append' and 'ModelFailure' in e[1])
        self.assertEqual(order[failure + 1][0], 'sample')

    def test_recovery_budget_spans_later_tool_batches_and_compaction(self):
        host, context = _Host(), _context()
        model = _Model(host, _expired(), _calls(_call('a')), _expired(), _answer('never'))
        with _fake_compaction(host, due=False), mock.patch.object(
                kernel, 'auto_compaction_due', side_effect=[False, True, False]):
            with self.assertRaises(ModelContinuationExpired):
                run_turn(context, model, _environment(host), _config(), host)
        self.assertEqual(host.calls_run, 1)
        self.assertEqual(host.notices, [kernel.CONTINUATION_NOTICE])
        self.assertEqual(host.retryable, 1)
        self.assertEqual(len(model.script), 1)
        self.assertIn(('compacted',), host.events)

    def test_failed_failure_checkpoint_prevents_relaunch(self):
        class CannotSave(_Host):
            def append(self, context, items):
                if any(isinstance(i, ModelFailure) for i in items):
                    raise OSError('save failed')
                super().append(context, items)
        host = CannotSave()
        model = _Model(host, _expired(), _answer('never'))
        with self.assertRaisesRegex(OSError, 'save failed'):
            run_turn(_context(), model, _environment(host), _config(), host)
        self.assertEqual(len(model.script), 1)
        self.assertEqual(host.notices, [])

    def test_stop_or_steering_rechecked_before_cold_retry(self):
        class StopAfterNotice(_Host):
            def notice(self, text):
                super().notice(text)
                self.stop = True
        host = StopAfterNotice()
        model = _Model(host, _expired(), _answer('never'))
        self.assertEqual(run_turn(_context(), model, _environment(host), _config(), host).kind, 'stopped')
        self.assertEqual(len(model.script), 1)
        class SteerAfterNotice(_Host):
            def interrupt(self):
                return Steer((Message('user', 'steer recovery'),)) if self.notices else Interrupt.CONTINUE
        host, context = SteerAfterNotice(), _context()
        model = _Model(host, _expired(), _answer('done'))
        run_turn(context, model, _environment(host), _config(), host)
        self.assertIn(Message('user', 'steer recovery'), context.items)

    def test_generic_transport_error_and_unresolved_context_do_not_auto_retry(self):
        for unresolved in (False, True):
            host, context = _Host(), _context()
            if unresolved:
                context.append(_call('unresolved'))
            error = _expired() if unresolved else ModelTransportError('other transport error')
            model = _Model(host, error, _answer('never'))
            with self.subTest(unresolved=unresolved), self.assertRaises(ModelTransportError):
                run_turn(context, model, _environment(host), _config(), host)
            self.assertEqual(host.calls_run, 0)
            self.assertEqual(host.notices, [])
            self.assertEqual(len(model.script), 1)

    def test_marker_contract_and_loop_defenses_require_metadata_and_no_partial_output(self):
        with self.assertRaises(TypeError):
            ModelContinuationExpired('no metadata')
        with self.assertRaises(ValueError):
            ModelContinuationExpired('partial', failure=_expired().failure, completed_items=(_call('a'),))
        for field, value in (('failure', None), ('completed_items', (_call('a'),))):
            error = _expired()
            setattr(error, field, value)  # a provider violating the marker contract
            host, context = _Host(), _context()
            model = _Model(host, error, _answer('never'))
            with self.subTest(field=field), self.assertRaises(ModelContinuationExpired):
                run_turn(context, model, _environment(host), _config(), host)
            self.assertEqual(host.notices, [])
            self.assertEqual(host.calls_run, 0)

    def test_saves_each_result_before_the_next_call_and_ends(self):
        host, context = _Host(), _context()
        model = _Model(host, _calls(_call("a"), _call("b")), _answer("done"))
        result = run_turn(context, model, _environment(host), _config(), host)
        self.assertEqual(result, TurnResult("ended", "done"))
        appends = [e for e in host.events if e[0] in ("append", "tool")]
        self.assertEqual(appends[1:5], [("tool", 1), ("append", ("ToolResult",)),
                                        ("tool", 2), ("append", ("ToolResult",))])
        self.assertEqual(appends[0][0], "append")  # the response, before any tool
        self.assertIn("ToolCall", appends[0][1])
        self.assertIsInstance(context.items[-1], TurnSummary)
        self.assertEqual(context.pending_tool_calls(), ())
        self.assertEqual(model.retired, 1)
        self.assertEqual(model.threads | host.threads, {threading.get_ident()})

    def test_injected_messages_wait_for_the_whole_batch(self):
        host, context = _Host(), _context()
        model = _Model(host, _calls(_call("a"), _call("b")), _answer("done"))
        run_turn(context, model, _environment(host, inject=1), _config(), host)
        kinds = [type(i).__name__ for i in context.items]
        start = kinds.index("ToolResult")
        self.assertEqual(kinds[start:start + 3], ["ToolResult", "ToolResult", "Message"])
        self.assertEqual(context.items[start + 2].content, "injected")

    def test_stop_before_a_tool_call_leaves_unstarted_calls_unanswered(self):
        host, context = _Host(stop_after_calls=1), _context()
        model = _Model(host, _calls(_call("a"), _call("b")))
        result = run_turn(context, model, _environment(host), _config(), host)
        self.assertEqual(result.kind, "stopped")
        self.assertEqual(host.calls_run, 1)
        self.assertEqual([c.call_id for c in context.pending_tool_calls()], ["b"])
        self.assertEqual(model.retired, 1)

    def test_interrupt_runs_once_right_before_each_sample(self):
        host, context = _Host(), _context()
        model = _Model(host, _calls(_call("a"), _call("b")), _answer("done"))
        run_turn(context, model, _environment(host), _config(), host)
        order = [e[0] for e in host.events if e[0] in ("interrupt", "sample", "tool")]
        self.assertEqual(order, ["interrupt", "sample", "tool", "tool", "interrupt", "sample"])

    def test_interrupt_stop_returns_stopped_without_sampling(self):
        class Stopping(_Host):
            def interrupt(self):
                return Interrupt.STOP
        host = Stopping()
        model = _Model(host)
        result = run_turn(_context(), model, _environment(host), _config(), host)
        self.assertEqual(result.kind, "stopped")
        self.assertEqual(model.retired, 1)

    def test_a_steer_is_saved_right_before_the_sample_and_continues_the_turn(self):
        class Steering(_Host):
            steered = False

            def interrupt(self):
                super().interrupt()
                if self.calls_run and not self.steered:
                    self.steered = True
                    return Steer((Message("user", "also this"), Message("user", "and that")))
                return Interrupt.CONTINUE
        host, context = Steering(), _context()
        model = _Model(host, _calls(_call("a")), _answer("done"))
        run_turn(context, model, _environment(host), _config(), host)
        kinds = [type(item).__name__ for item in context.items]
        start = kinds.index("ToolResult") + 1
        self.assertEqual([item.content for item in context.items[start:start + 2]],
                         ["also this", "and that"])
        self.assertIn(("sample", start + 2), host.events)  # the next sample saw both
        # Like tool-injected messages: no UserInteractionBoundary, so the
        # provider turn (e.g. Codex turn state) continues.
        self.assertNotIn("UserInteractionBoundary", kinds)
        with self.assertRaises(ValueError):
            Steer(())

    def test_failed_sample_saves_partial_output_and_closes_its_calls(self):
        host, context = _Host(), _context()
        error = ModelTransportError("lost", completed_items=(_call("a"),))
        model = _Model(host, error)
        with self.assertRaises(ModelTransportError):
            run_turn(context, model, _environment(host), _config(), host)
        self.assertEqual(host.calls_run, 0)
        self.assertEqual(host.retryable, 1)
        self.assertIsInstance(context.items[-2], ModelSampleBoundary)
        closing = context.items[-1]
        self.assertIsInstance(closing, ToolResult)
        self.assertFalse(closing.success)
        self.assertIn("Not executed", closing.output)
        self.assertEqual(model.retired, 1)

    def test_failure_after_a_stop_request_is_a_stop(self):
        host, context = _Host(), _context()

        def retired_by_stop():
            host.stop = True
            return ModelTransportError("retired", completed_items=(Message("assistant", "par"),))
        model = _Model(host, retired_by_stop)
        result = run_turn(context, model, _environment(host), _config(), host)
        self.assertEqual(result.kind, "stopped")
        self.assertEqual(host.retryable, 0)
        self.assertEqual(context.items[-2].content, "par")  # the partial output is saved

    def test_overflow_compacts_and_retries_at_most_once(self):
        host, context = _Host(), _context()
        model = _Model(host, ModelContextWindowError("full"), ModelContextWindowError("full"))
        with _fake_compaction(host, due=False):
            with self.assertRaises(ModelContextWindowError):
                run_turn(context, model, _environment(host), _config(), host)
        self.assertEqual([e for e in host.events if e[0] == "compacted"], [("compacted",)])
        self.assertEqual(host.notices, [kernel.OVERFLOW_NOTICE])
        self.assertEqual(host.retryable, 1)

    def test_a_compaction_that_a_stop_fails_is_a_stop(self):
        for overflow, stopping in ((False, True), (True, True), (False, False), (True, False)):
            host, context = _Host(), _context()
            script = (ModelContextWindowError("full"),) if overflow else ()
            model = _Model(host, *script, _answer("never"))

            class Compactor:
                def compact(self, context, *, tools=(), sample_params=None):
                    host.stop = stopping  # e.g. a stop that retired the model
                    raise ModelTransportError("retired")
            with self.subTest(overflow=overflow, stopping=stopping), \
                    mock.patch.object(kernel, "auto_compaction_due", return_value=not overflow), \
                    mock.patch.object(kernel, "uses_host_auto_compaction", return_value=True), \
                    mock.patch.object(kernel, "create_default_compactor", return_value=Compactor()):
                if stopping:
                    result = run_turn(context, model, _environment(host), _config(), host)
                    self.assertEqual(result.kind, "stopped")
                    self.assertEqual(host.retryable, 0)
                else:
                    with self.assertRaises(ModelTransportError):
                        run_turn(context, model, _environment(host), _config(), host)
                    self.assertEqual(host.retryable, 1 if overflow else 0)
                self.assertEqual(len(model.script), 1)  # never sampled again
                self.assertEqual(model.retired, 1)

    def test_due_compaction_runs_before_the_interrupt_and_sample(self):
        host, context = _Host(), _context()
        model = _Model(host, _answer("done"))
        with _fake_compaction(host, due=True):
            run_turn(context, model, _environment(host), _config(), host)
        order = [e[0] if e[0] != "trace" else e for e in host.events
                 if e[0] in ("compacted", "interrupt", "sample", "trace")]
        self.assertEqual(order, [("trace", "compact"), "compacted", "interrupt",
                                 ("trace", "sample"), "sample"])

    def test_compaction_pause_samples_again(self):
        host, context = _Host(), _context()
        paused = ModelSample((Message("assistant", "..."),), stop_reason="compaction")
        model = _Model(host, paused, _answer("done"))
        result = run_turn(context, model, _environment(host), _config(), host)
        self.assertEqual(result.final_text, "done")
        self.assertEqual(len([e for e in host.events if e[0] == "sample"]), 2)

    def test_limits_and_blank_answers_raise_the_shared_types(self):
        host = _Host()
        model = _Model(host, _calls(_call("a")))
        with self.assertRaises(SampleLimitExceeded):
            run_turn(_context(), model, _environment(host), _config(max_samples=1), host)
        host = _Host()
        model = _Model(host, _answer("   "))
        with self.assertRaises(MissingFinalText):
            run_turn(_context(), model, _environment(host), _config(), host)
        self.assertEqual(host.retryable, 1)
        self.assertEqual(model.retired, 1)

    def test_turn_result_validation(self):
        with self.assertRaises(ValueError):
            TurnResult("ended", " ")
        with self.assertRaises(ValueError):
            TurnResult("stopped", "text")
        with self.assertRaises(ValueError):
            TurnResult("yielded")
        with self.assertRaises(ValueError):
            TurnResult("ended", "text", note="note")
        with self.assertRaises(ValueError):
            TurnResult("paused")

    def test_a_recorded_yield_ends_the_turn_after_the_saved_batch(self):
        host, context = _Host(), _context()
        model = _Model(host, _calls(_call("a"), _call("b")), _answer("never"))
        result = run_turn(context, model, _environment(host), _config(), host,
                          control=_Control(host, after_calls=1))
        self.assertEqual(result, TurnResult("yielded", note="please review"))
        self.assertEqual(host.calls_run, 2)  # the rest of the batch still runs
        self.assertEqual(context.pending_tool_calls(), ())
        self.assertIsInstance(context.items[-1], TurnSummary)
        self.assertEqual(len([e for e in host.events if e[0] == "sample"]), 1)
        self.assertEqual(model.retired, 1)

    def test_a_yield_on_the_last_allowed_sample_is_a_yield(self):
        host, context = _Host(), _context()
        model = _Model(host, _calls(_call("a")))
        result = run_turn(context, model, _environment(host), _config(max_samples=1), host,
                          control=_Control(host, after_calls=1))
        self.assertEqual(result.kind, "yielded")

    def test_a_stop_during_the_batch_wins_over_a_yield(self):
        host, context = _Host(stop_after_calls=1), _context()
        model = _Model(host, _calls(_call("a"), _call("b")))
        result = run_turn(context, model, _environment(host), _config(), host,
                          control=_Control(host, after_calls=1))
        self.assertEqual(result.kind, "stopped")


class _Control:
    """Reports a handoff note once a given number of calls has run."""

    def __init__(self, host, after_calls):
        self.host, self.after_calls = host, after_calls

    def request(self):
        return "please review" if self.host.calls_run >= self.after_calls else None


class _FakeCompaction:
    def __init__(self, host):
        self.host = host

    def context_items(self):
        return ()

    def display_items(self):
        return ()


def _fake_compaction(host, *, due):
    class Compactor:
        def compact(self, context, *, tools=(), sample_params=None):
            host.events.append(("compacted",))
            return _FakeCompaction(host)
    patches = (
        mock.patch.object(kernel, "auto_compaction_due",
                          side_effect=[due] + [False] * 10),
        mock.patch.object(kernel, "uses_host_auto_compaction", return_value=True),
        mock.patch.object(kernel, "create_default_compactor",
                          return_value=Compactor()),
        mock.patch.object(kernel, "CompactionResult", _FakeCompaction),
    )
    combined = contextlib.ExitStack()
    for patch in patches:
        combined.enter_context(patch)
    return combined


class _SteeringHost(_Host):
    """A host with steers: a queue, and a Preemption that tests flush."""

    def __init__(self, **options):
        super().__init__(**options)
        self.preemption, self.queued = Preemption(), []

    def steer(self, text, level=Urgency.IMMEDIATE):
        self.queued.append(Message("user", text))
        self.preemption.flush(level)

    def interrupt(self):
        answer = super().interrupt()
        if answer is Interrupt.STOP:
            return answer
        self.preemption.take()
        messages, self.queued = tuple(self.queued), []
        return Steer(messages) if messages else answer

    def should_steer(self):
        return self.preemption.due()

    def preemptible(self, cancel):
        return self.preemption.operation(cancel)


def _results(context):
    return [(item.call_id, item.output, item.success) for item in context
            if isinstance(item, ToolResult)]


def _tools(*handlers):
    """An environment of tools named t0, t1, ... with these handlers."""
    return Environment(tuple(Tool(ToolSpec(f"t{index}", "Tool.", {"type": "object"}), handler)
                             for index, handler in enumerate(handlers)))


def _ok(arguments, *, timeout_seconds=None):
    return ToolOutcome("ok")


class SteeringTests(unittest.TestCase):
    """Immediate and preempting steers, and preempting stops, in run_turn."""

    def test_an_immediate_steer_during_a_sample_skips_its_calls_and_comes_next(self):
        host, context = _SteeringHost(), _context()

        def sample():
            host.steer("change of plan")
            return _calls(_call("a", "t0"), _call("b", "t0"))
        model = _Model(host, sample, _answer("done"))
        result = run_turn(context, model, _tools(_ok), _config(), host)
        self.assertEqual(result.final_text, "done")
        skipped = kernel.SKIPPED_OUTPUT
        self.assertEqual(_results(context), [("a", skipped, False), ("b", skipped, False)])
        kinds = [type(item).__name__ for item in context.items]
        steer = kinds.index("ToolResult") + 2
        self.assertEqual(context.items[steer], Message("user", "change of plan"))
        self.assertIn(("sample", steer + 1), host.events)  # the next sample saw it
        self.assertNotIn("UserInteractionBoundary", kinds)
        self.assertFalse(host.preemption.due())

    def test_a_steer_during_a_final_answer_continues_the_turn(self):
        host, context = _SteeringHost(), _context()

        def sample():
            host.steer("one more thing")
            return _answer("first")
        model = _Model(host, sample, _answer("second"))
        result = run_turn(context, model, _environment(host), _config(), host)
        self.assertEqual(result.final_text, "second")
        texts = [item.content for item in context.items if isinstance(item, Message)]
        self.assertEqual(texts, ["task", "first", "one more thing", "second"])
        self.assertEqual(sum(isinstance(item, TurnSummary) for item in context.items), 1)

    def test_a_steer_on_the_last_allowed_sample_waits_for_the_next_turn(self):
        host, context = _SteeringHost(), _context()

        def sample():
            host.steer("too late")
            return _answer("final")
        model = _Model(host, sample)
        result = run_turn(context, model, _environment(host), _config(max_samples=1), host)
        self.assertEqual(result.final_text, "final")  # no sample is left for the steer
        self.assertEqual([m.content for m in host.queued], ["too late"])

    def test_a_steer_during_a_call_skips_the_rest_of_the_batch(self):
        host, context = _SteeringHost(), _context()

        def steering(arguments, *, timeout_seconds=None):
            host.steer("stop that")
            return ToolOutcome("ran", user_messages=(Message("user", "injected"),))
        model = _Model(host, _calls(_call("a", "t0"), _call("b", "t1"), _call("c", "t1")),
                       _answer("done"))
        run_turn(context, model, _tools(steering, _ok), _config(), host)
        skipped = kernel.SKIPPED_OUTPUT
        self.assertEqual(_results(context), [("a", "ran", True), ("b", skipped, False),
                                             ("c", skipped, False)])
        tail = [item.content for item in context.items if isinstance(item, Message)]
        self.assertEqual(tail[-3:], ["injected", "stop that", "done"])

    def test_a_preempting_steer_cancels_a_sample_without_a_failure(self):
        host, context = _SteeringHost(), _context()

        def sample():
            host.steer("now", Urgency.PREEMPT)
            deadline = time.monotonic() + 5
            while not model.retired and time.monotonic() < deadline:
                time.sleep(0.001)  # the cancel retires the model on a helper thread
            return ModelTransportError("retired", completed_items=(Message("assistant", "par"),))
        model = _Model(host, sample, _answer("done"))
        result = run_turn(context, model, _environment(host), _config(max_samples=1), host)
        self.assertEqual(result.final_text, "done")  # the cancelled sample did not count
        self.assertEqual(model.retired, 2)  # the cancel, then the turn's end
        self.assertEqual(host.retryable, 0)
        texts = [item.content for item in context.items if isinstance(item, Message)]
        self.assertEqual(texts, ["task", "par", "now", "done"])

    def test_a_preempting_steer_cancels_a_tool_call_s_wait(self):
        host, context = _SteeringHost(), _context()

        def waiting(arguments, *, timeout_seconds=None):
            host.steer("enough", Urgency.PREEMPT)
            token = current_cancel()
            woke = select.select([token.fileno()], [], [], 5)[0]
            return ToolOutcome("cut short" if woke else "waited")
        model = _Model(host, _calls(_call("a", "t0"), _call("b", "t0")), _answer("done"))
        run_turn(context, model, _tools(waiting), _config(), host)
        self.assertEqual(_results(context), [("a", "cut short", True),
                                             ("b", kernel.SKIPPED_OUTPUT, False)])
        self.assertEqual(model.retired, 1)  # a tool's cancel never retires the model

    def test_input_after_the_interrupt_point_skips_the_sample(self):
        class Late(_SteeringHost):
            def interrupt(self):
                answer = super().interrupt()
                if not self.steered:
                    self.steered = True
                    self.steer("just in time")  # arrives right after the handover
                return answer
        host, context = Late(), _context()
        host.steered = False
        model = _Model(host, _answer("done"))
        run_turn(context, model, _environment(host), _config(), host)
        samples = [event for event in host.events if event[0] == "sample"]
        self.assertEqual(len(samples), 1)
        self.assertEqual([e[0] for e in host.events if e[0] in ("interrupt", "sample")],
                         ["interrupt", "interrupt", "sample"])
        texts = [item.content for item in context.items if isinstance(item, Message)]
        self.assertEqual(texts, ["task", "just in time", "done"])

    def test_a_stop_beats_a_steer_and_a_recorded_yield_beats_a_steer(self):
        host, context = _SteeringHost(), _context()

        def stopping(arguments, *, timeout_seconds=None):
            host.steer("too late")
            host.stop = True
            return ToolOutcome("ran")
        model = _Model(host, _calls(_call("a", "t0"), _call("b", "t0")))
        self.assertEqual(run_turn(context, model, _tools(stopping), _config(), host).kind,
                         "stopped")
        self.assertEqual([c.call_id for c in context.pending_tool_calls()], ["b"])
        host, context = _SteeringHost(), _context()
        control = SimpleNamespace(note=None, request=lambda: control.note)

        def handing_off(arguments, *, timeout_seconds=None):
            control.note = "Please review."
            host.steer("meanwhile")
            return ToolOutcome("Recorded.")
        model = _Model(host, _calls(_call("y", "t0"), _call("b", "t1")))
        result = run_turn(context, model, _tools(handing_off, _ok), _config(), host,
                          control=control)
        self.assertEqual((result.kind, result.note), ("yielded", "Please review."))
        self.assertEqual(_results(context)[1], ("b", kernel.SKIPPED_OUTPUT, False))
        self.assertEqual([m.content for m in host.queued], ["meanwhile"])  # still pending

    def test_a_preempting_stop_cancels_and_skips_every_later_operation(self):
        host, context = _SteeringHost(), _context()

        def waiting(arguments, *, timeout_seconds=None):
            host.stop = True
            host.preemption.stop(Urgency.PREEMPT)
            woke = select.select([current_cancel().fileno()], [], [], 5)[0]
            return ToolOutcome("cut short" if woke else "waited")
        model = _Model(host, _calls(_call("a", "t0"), _call("b", "t0")))
        self.assertEqual(run_turn(context, model, _tools(waiting), _config(), host).kind,
                         "stopped")
        self.assertEqual(_results(context), [("a", "cut short", True)])
        self.assertEqual([c.call_id for c in context.pending_tool_calls()], ["b"])


class _LevelHost(_SteeringHost):
    """A host whose stops have levels, as the apps' hosts do: both stop
    checks read the Preemption, and a stop drops the queued steers."""

    def request_stop(self, level):
        self.queued = []
        self.preemption.stop(level)

    def should_stop(self):
        level = self.preemption.stop_level
        return level is not None and level >= Urgency.IMMEDIATE

    def stop_requested(self):
        return self.preemption.stop_level is not None


def _compaction_due_after_the_first_sample(host):
    """Compaction that is due at every interrupt point but the first."""
    stack = _fake_compaction(host, due=False)
    stack.enter_context(mock.patch.object(kernel, "auto_compaction_due",
                                          side_effect=[False] + [True] * 10))
    return stack


class StopLevelTests(unittest.TestCase):
    """Stops at the three levels in run_turn, and with flushed steers."""

    def test_a_gentle_stop_lets_the_whole_batch_run_then_stops_before_compacting(self):
        host, context = _LevelHost(), _context()

        def sample():
            host.request_stop(Urgency.QUEUED)  # /exit during the sample
            return _calls(_call("a", "t0"), _call("b", "t1"))

        def stopping(arguments, *, timeout_seconds=None):
            host.request_stop(Urgency.QUEUED)  # and again during a call
            return ToolOutcome("ran")
        model = _Model(host, sample, _answer("never"))
        with _compaction_due_after_the_first_sample(host):
            result = run_turn(context, model, _tools(stopping, _ok), _config(), host)
        self.assertEqual(result.kind, "stopped")
        self.assertEqual(_results(context), [("a", "ran", True), ("b", "ok", True)])
        self.assertEqual(context.pending_tool_calls(), ())
        self.assertEqual([e[0] for e in host.events if e[0] in ("sample", "compacted")],
                         ["sample"])  # no compaction and no sample after the stop
        self.assertFalse(any(isinstance(item, TurnSummary) for item in context.items))

    def test_a_gentle_stop_lets_a_final_answer_end_the_turn(self):
        host, context = _LevelHost(), _context()

        def sample():
            host.request_stop(Urgency.QUEUED)
            return _answer("done")
        result = run_turn(context, _Model(host, sample), _environment(host), _config(), host)
        self.assertEqual((result.kind, result.final_text), ("ended", "done"))
        self.assertIsInstance(context.items[-1], TurnSummary)

    def test_a_gentle_stop_after_a_failed_sample_neither_retries_nor_recovers(self):
        for failure in (ModelTransportError("lost"), _expired()):
            with self.subTest(failure=type(failure).__name__):
                host, context = _LevelHost(), _context()

                def sample():
                    host.request_stop(Urgency.QUEUED)
                    return failure
                model = _Model(host, sample, _answer("never"))
                result = run_turn(context, model, _environment(host), _config(), host)
                self.assertEqual(result.kind, "stopped")
                self.assertEqual(len([e for e in host.events if e[0] == "sample"]), 1)
                self.assertEqual((host.retryable, host.notices), (0, []))

    def test_a_stop_after_the_last_allowed_batch_is_not_a_sample_limit_failure(self):
        for level in Urgency:
            with self.subTest(level=level.name):
                host, context = _LevelHost(), _context()

                def stopping(arguments, *, timeout_seconds=None):
                    host.request_stop(level)
                    return ToolOutcome("ran")
                model = _Model(host, _calls(_call("a", "t0")))
                result = run_turn(context, model, _tools(stopping), _config(max_samples=1),
                                  host)
                self.assertEqual(result.kind, "stopped")
                self.assertEqual(_results(context), [("a", "ran", True)])
        host = _LevelHost()  # without a stop, the limit is still a failure
        with self.assertRaises(SampleLimitExceeded):
            run_turn(_context(), _Model(host, _calls(_call("a", "t0"))), _tools(_ok),
                     _config(max_samples=1), host)

    def test_a_stop_drops_a_flushed_steer_but_keeps_its_calls_from_starting(self):
        host, context = _LevelHost(), _context()

        def sample():
            host.steer("change of plan")  # /steer!, then /exit
            host.request_stop(Urgency.QUEUED)
            return _calls(_call("a", "t0"), _call("b", "t0"))
        tool = mock.Mock(return_value=ToolOutcome("ran"))
        model = _Model(host, sample, _answer("never"))
        result = run_turn(context, model, _tools(tool), _config(), host)
        self.assertEqual(result.kind, "stopped")
        tool.assert_not_called()
        # Not closed as skipped "because the user sent a message": the stop
        # dropped the message. They stay unanswered, as at any level-2 stop.
        self.assertEqual(_results(context), [])
        self.assertEqual([c.call_id for c in context.pending_tool_calls()], ["a", "b"])
        self.assertEqual(host.preemption.stop_level, Urgency.IMMEDIATE)
        texts = [item.content for item in context.items if isinstance(item, Message)]
        self.assertEqual(texts, ["task"])

    def test_a_stop_after_a_flushed_steer_lets_a_final_answer_stand(self):
        host, context = _LevelHost(), _context()

        def sample():
            host.steer("one more thing")
            host.request_stop(Urgency.IMMEDIATE)  # a first Ctrl-C
            return _answer("first")
        model = _Model(host, sample, _answer("never"))
        result = run_turn(context, model, _environment(host), _config(), host)
        self.assertEqual((result.kind, result.final_text), ("ended", "first"))
        self.assertEqual(len([e for e in host.events if e[0] == "sample"]), 1)


class PreemptionTests(unittest.TestCase):
    def test_a_cancel_runs_once_and_only_while_its_operation_runs(self):
        preemption, cancels = Preemption(), []
        with preemption.operation(lambda: cancels.append(1)) as operation:
            self.assertFalse(operation.skipped)
            preemption.flush(Urgency.IMMEDIATE)  # waits; cancels nothing
            self.assertFalse(operation.cancelled)
            preemption.flush(Urgency.PREEMPT)
            preemption.flush(Urgency.PREEMPT)
            self.assertEqual(preemption.level, Urgency.PREEMPT)
            preemption.stop(Urgency.PREEMPT)
        self.assertTrue(operation.cancelled)
        self.assertEqual(cancels, [1])  # done before the operation ended
        self.assertEqual(preemption.level, Urgency.QUEUED)  # the stop dropped the steers
        self.assertEqual(preemption.stop_level, Urgency.PREEMPT)
        with preemption.operation(lambda: cancels.append(2)) as later:
            self.assertTrue(later.skipped)  # after a preempting stop, nothing starts
        self.assertEqual(cancels, [1])

    def test_a_stop_only_rises_and_skips_operations_from_level_2(self):
        preemption, cancels = Preemption(), []
        preemption.stop(Urgency.QUEUED)  # /exit
        with preemption.operation(lambda: cancels.append(1)) as operation:
            self.assertFalse(operation.skipped)  # the batch may finish
            preemption.flush(Urgency.PREEMPT)  # a stop dropped the steers
            self.assertFalse(preemption.due())
            self.assertEqual(preemption.level, Urgency.QUEUED)
            preemption.stop(Urgency.IMMEDIATE)  # Ctrl-C
            preemption.stop(Urgency.QUEUED)
        self.assertEqual((cancels, preemption.stop_level), ([], Urgency.IMMEDIATE))
        with preemption.operation() as later:
            self.assertTrue(later.skipped)

    def test_steers_flushed_before_a_stop_raise_it_to_level_2_but_never_3(self):
        for flushed in (Urgency.QUEUED, Urgency.IMMEDIATE, Urgency.PREEMPT):
            with self.subTest(flushed=flushed.name):
                preemption, cancels = Preemption(), []
                with preemption.operation(lambda: cancels.append(1)):
                    preemption.flush(flushed)
                    preemption.stop(Urgency.QUEUED)
                    self.assertFalse(preemption.due())
                expected = Urgency.QUEUED if flushed == Urgency.QUEUED else Urgency.IMMEDIATE
                self.assertEqual(preemption.stop_level, expected)
                # Only the preempting steer cancelled; the stop cancels nothing.
                self.assertEqual(cancels, [1] if flushed == Urgency.PREEMPT else [])

    def test_the_notation_and_ctrl_c(self):
        names = ("/exit", "/quit")
        for word, expected in (("/exit", Urgency.QUEUED), ("/quit!", Urgency.IMMEDIATE),
                               ("/exit!!", Urgency.PREEMPT), ("/quit!!!!", Urgency.PREEMPT)):
            with self.subTest(word=word):
                self.assertIs(command_urgency(word, *names), expected)
        for word in ("/exits", "/exit!x", "exit", "!", "", "/steer"):
            with self.subTest(word=word):
                self.assertIsNone(command_urgency(word, *names))
        self.assertIs(command_urgency("/steer!", "/steer"), Urgency.IMMEDIATE)
        self.assertEqual([escalated_stop(level) for level in (None, *Urgency)],
                         [Urgency.IMMEDIATE, Urgency.IMMEDIATE, Urgency.PREEMPT,
                          Urgency.PREEMPT])

    def test_leaving_waits_for_a_slow_cancel_so_it_never_reaches_the_next_operation(self):
        preemption, events = Preemption(), []
        started, release = threading.Event(), threading.Event()

        def slow_cancel():
            started.set()
            release.wait(5)
            events.append("cancelled")
        with preemption.operation(slow_cancel):
            preemption.flush(Urgency.PREEMPT)
            self.assertTrue(started.wait(5))
            threading.Timer(0.05, release.set).start()
        events.append("left")
        self.assertEqual(events, ["cancelled", "left"])
        preemption.take()
        with preemption.operation(lambda: events.append("wrong")) as later:
            self.assertFalse(later.skipped)
        self.assertEqual(events, ["cancelled", "left"])

    def test_operations_do_not_nest_and_a_failed_cancel_is_contained(self):
        preemption = Preemption()

        def broken():
            raise RuntimeError("cannot cancel")
        with preemption.operation(broken) as operation:
            with self.assertRaises(RuntimeError):
                with preemption.operation():
                    pass
            preemption.flush(Urgency.PREEMPT)
        self.assertTrue(operation.cancelled)
        self.assertTrue(preemption.due())


if __name__ == "__main__":
    unittest.main()
