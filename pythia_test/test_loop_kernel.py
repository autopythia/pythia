"""run_turn with a scripted model and a host that records every call."""

from __future__ import annotations

import contextlib
import threading
import unittest
from unittest import mock

from pythia.interaction import Environment, Init, InteractionContext, Message
from pythia.interaction import ModelSample, Tool, ToolCall, ToolOutcome, ToolResult, ToolSpec
from pythia.interaction.items import ModelSampleBoundary, TurnSummary
from pythia.interaction.loop import Interrupt, MissingFinalText, SampleLimitExceeded, Steer
from pythia.interaction.loop import TurnHost, TurnResult, run_turn
from pythia.interaction.loop import kernel
from pythia.interaction.model import ModelContextWindowError, ModelTransportError
from pythia.interaction.runtime_config import InteractionConfigSnapshot


def _call(call_id, name="probe"):
    return ToolCall(name=name, call_id=call_id, arguments_json="{}")


def _answer(text):
    return ModelSample((Message("assistant", text),))


def _calls(*calls):
    return ModelSample(tuple(calls))


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


if __name__ == "__main__":
    unittest.main()
