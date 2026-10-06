"""Native-expiry fixtures; never discover, download, or execute Claude."""
from copy import deepcopy
import json
import os
import threading
import time
import unittest
from unittest import mock

from pythia.interaction import ModelContinuationExpired
from pythia.interaction._debug_trace import DebugTrace
from pythia.interaction.claude_relay import ClaudeRelayModel
from pythia.interaction.claude_relay._context import Snapshot
from pythia.interaction.claude_relay._mcp import Mailbox, MCPError
from pythia.interaction.claude_relay._recovery import NativeMCPTimeout, NATIVE_TIMEOUT_TEXT
from pythia.interaction.claude_relay._runtime import Runtime
from pythia.interaction.context import InteractionContext
from pythia.interaction.environment import Environment, Tool, ToolOutcome
from pythia.interaction.items import Message, ModelFailure, SampleMetadata, ToolCall, ToolResult
from pythia.interaction.loop import TurnHost, run_turn
from pythia.interaction.model import ModelError, ModelSample
from pythia.interaction.runtime_config import InteractionConfigSnapshot
from pythia.interaction.save import load_interaction_save, save_interaction_save
from pythia_test import test_interaction_claude_relay as fixture


def timed_out(ident='a', **values):
    return {'type': 'tool_result', 'tool_use_id': ident, 'is_error': True,
            'content': NATIVE_TIMEOUT_TEXT, **values}


class NativeResultTests(unittest.TestCase):
    def box(self, *, claimed=True):
        box = Mailbox(fixture.TOOLS)
        box.register([('a', 'echo', {'value': 1})])
        # Unit state injection; integration tests below exercise the real callback.
        box.slots['a']['claimed'] = claimed
        return box

    def test_pending_claimed_timeout_invalidates_without_fabricating_a_result(self):
        box = self.box()
        failure = mock.Mock(); box.on_failure = failure
        with self.assertRaises(NativeMCPTimeout) as caught:
            box.validate_native_results([timed_out()])
        self.assertEqual(caught.exception.call.native_id, 'a')
        self.assertFalse(caught.exception.call.released)
        self.assertFalse(caught.exception.call.returned)
        self.assertIs(box.failure, caught.exception)
        self.assertTrue(box.closed)
        self.assertIsNone(box.slots['a']['result'])
        failure.assert_called_once_with(caught.exception)
        # Late callbacks / secondary errors cannot replace the expiry cause.
        self.assertIs(box.fail('secondary'), caught.exception)

    def test_release_and_return_races_do_not_authorize_synthetic_timeouts(self):
        for returned in (False, True):
            box = self.box()
            box.release({'a': ('real success or running-session handle', True)})
            box.slots['a']['returned'] = returned
            with self.subTest(returned=returned), self.assertRaises(NativeMCPTimeout) as caught:
                box.validate_native_results([timed_out()])
            self.assertTrue(caught.exception.call.released)
            self.assertEqual(caught.exception.call.returned, returned)
            self.assertEqual(box.slots['a']['result']['content'][0]['text'], 'real success or running-session handle')

    def test_real_host_timeout_echo_is_an_ordinary_tool_error(self):
        for content in (NATIVE_TIMEOUT_TEXT, [{'type': 'text', 'text': NATIVE_TIMEOUT_TEXT}]):
            box = self.box(claimed=False)
            box.release({'a': (NATIVE_TIMEOUT_TEXT, False)})
            box.call('a', 'echo', {'value': 1})
            before = deepcopy(box.slots)
            box.validate_native_results([timed_out(content=content)])
            self.assertFalse(box.closed)
            self.assertEqual(box.slots, before)

    def test_unknown_malformed_unclaimed_and_other_premature_results_stay_fatal(self):
        for blocks in ([timed_out('unknown')], [timed_out(is_error=1)], [timed_out(is_error=False)],
                       [timed_out(content='prefix ' + NATIVE_TIMEOUT_TEXT)],
                       [timed_out(content=[{'type': 'text', 'text': NATIVE_TIMEOUT_TEXT}, {'type': 'text', 'text': ''}])],
                       [timed_out(), timed_out()], [timed_out(), {'type': 'text', 'text': 'other'}]):
            with self.subTest(blocks=blocks), self.assertRaises(MCPError):
                self.box().validate_native_results(blocks)
        with self.assertRaises(MCPError):
            self.box(claimed=False).validate_native_results([timed_out()])

    def test_mixed_envelope_is_validated_before_classifying_expiry(self):
        box = self.box()
        box.register([('b', 'echo', {'value': 2})])
        box.slots['b']['claimed'] = box.slots['b']['returned'] = True
        with self.assertRaises(MCPError):
            box.validate_native_results([timed_out(), {'type': 'tool_result', 'tool_use_id': 'b', 'content': 'ok'}])
        self.assertNotIsInstance(box.failure, NativeMCPTimeout)


class SavingHost(TurnHost):
    def __init__(self, path, trace=None):
        self.path, self.tracer = path, trace
        self.notices = []

    def append(self, context, items):
        context.extend(items)
        save_interaction_save(self.path, context)

    def notice(self, text):
        self.notices.append(text)

    def trace(self, op, **tags):
        return super().trace(op, **tags) if self.tracer is None else self.tracer.operation(op, **tags)


@unittest.skipIf(os.getuid() == 0, 'non-root fixture broker required')
class RecoveryIntegrationTests(unittest.TestCase):
    setUp = fixture.ModelTests.setUp
    stop_broker = fixture.ModelTests.stop_broker

    def parked(self):
        context = InteractionContext((Message('user', 'native-timeout'),))
        sample = self.model.sample(context, tools=fixture.TOOLS)
        context.extend(sample.context_items())
        runtime = self.model._runtime
        self.wait_claim(runtime)
        return context, sample, runtime

    def wait_claim(self, runtime):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with runtime.mailbox.condition:
                if runtime.mailbox.slots.get('toolu_a', {}).get('claimed'):
                    return
            time.sleep(.005)
        self.fail('Fixture did not claim its MCP request')

    def expire(self, runtime):
        async def send():
            runtime.proc.stdin.write(b'fixture-expire\n')
            await runtime.proc.stdin.drain()
        runtime._submit(send(), 2)
        deadline = time.monotonic() + 3
        while runtime.error is None and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertIsInstance(runtime.error, NativeMCPTimeout)

    def complete(self, context, sample):
        context.extend(ToolResult(call.call_id, f'host-{i}') for i, call in enumerate(sample.tool_calls, 1))

    def test_direct_api_marks_only_complete_published_outcomes_and_cold_imports(self):
        context, sample, old = self.parked()
        self.expire(old)
        self.complete(context, sample)
        with self.assertRaises(ModelContinuationExpired) as caught:
            self.model.sample(context, tools=fixture.TOOLS)
        self.assertTrue(old.retirement_complete)
        self.assertIsNone(self.model._runtime)
        self.assertEqual(caught.exception.failure.error_code, 'native_mcp_timeout')
        self.assertFalse(caught.exception.completed_items)
        self.assertTrue(all(s['result'] is None for s in old.mailbox.slots.values()))
        final = self.model.sample(context, tools=fixture.TOOLS)
        self.assertEqual(final.last_assistant_text, 'recovered:host-1|host-2')
        self.assertFalse(final.tool_calls)

    def test_shared_loop_finishes_batch_once_and_recovers_with_durable_outcomes(self):
        for tracing in (False, True):
            self.model.close()
            trace = DebugTrace.open(self.root / 'recovery.jsonl', events=True) if tracing else None
            if trace:
                self.addCleanup(trace.close)
            self.model = ClaudeRelayModel(self.endpoint, trace=trace)
            self.addCleanup(self.model.close)
            context = InteractionContext((Message('user', 'native-timeout'),))
            host = SavingHost(self.root / 'history.jsonl', trace)
            effects, old = [], []
            def execute(arguments, **kwargs):
                self.assertEqual(arguments, {'value': 1})
                effects.append(len(effects) + 1)
                if len(effects) == 1:
                    old.append(self.model._runtime)
                    self.wait_claim(old[0]); self.expire(old[0])
                return ToolOutcome(f'host-{len(effects)}')
            environment = Environment((Tool(fixture.TOOLS[0], execute),))
            with self.subTest(tracing=tracing):
                result = run_turn(context, self.model, environment, InteractionConfigSnapshot(), host)
                self.assertEqual(result.final_text, 'recovered:host-1|host-2')
                self.assertEqual(effects, [1, 2])
                self.assertEqual(len(host.notices), 1)
                self.assertEqual(load_interaction_save(host.path).items, context.items)
                self.assertEqual(len([i for i in context if isinstance(i, ToolCall)]), 2)
                self.assertEqual(len([i for i in context if isinstance(i, ToolResult)]), 2)
                failure = next(i for i in context if isinstance(i, ModelFailure))
                self.assertEqual(failure.category, 'ModelContinuationExpired')
                recovered = [i for i in context if isinstance(i, SampleMetadata) and i.recovery]
                self.assertEqual(len(recovered), 1)
                self.assertEqual(recovered[0].request_attempts, 1)
                self.assertTrue(old[0].retirement_complete)
                self.assertNotIn('The operation timed out.', '|'.join(i.output for i in context if isinstance(i, ToolResult)))
                if trace:
                    trace.close()
                    rows = [json.loads(line) for line in trace.event_path.read_text().splitlines()]
                    self.assertTrue(any(r['type'] == 'continuation_recovery_eligibility' and r['eligible'] for r in rows))
                    self.assertEqual(len({r['runtime_id'] for r in rows if r['type'] == 'claude_invocation'}), 2)

    def test_missing_result_is_not_automatic_recovery(self):
        context, sample, old = self.parked()
        self.expire(old)
        context.append(ToolResult(sample.tool_calls[0].call_id, 'only first outcome'))
        with self.assertRaises(ModelError) as caught:
            self.model.sample(context, tools=fixture.TOOLS)
        self.assertNotIsInstance(caught.exception, ModelContinuationExpired)
        self.assertIsNone(self.model._runtime)

    def test_unpublished_proposal_does_not_grant_recovery(self):
        next_message = Runtime.next_message
        def expire_before_delivery(runtime):
            self.wait_claim(runtime)
            self.expire(runtime)
            return next_message(runtime)
        with mock.patch.object(Runtime, 'next_message', expire_before_delivery), self.assertRaises(NativeMCPTimeout):
            self.model.sample(InteractionContext((Message('user', 'native-timeout'),)), tools=fixture.TOOLS)
        self.assertIsNone(self.model._runtime)

    def test_cleanup_failure_denies_recovery_permission(self):
        context, sample, old = self.parked()
        self.expire(old); self.complete(context, sample)
        close = Runtime.close
        def broken(runtime):
            close(runtime)
            raise RuntimeError('private cleanup details')
        with mock.patch.object(Runtime, 'close', broken), self.assertRaises(NativeMCPTimeout) as caught:
            self.model.sample(context, tools=fixture.TOOLS)
        self.assertIn('cleanup failed', caught.exception.failure.message)
        self.assertNotIn('private cleanup details', caught.exception.failure.message)

    def test_unconfirmed_cleanup_without_exception_also_denies_recovery(self):
        context, sample, old = self.parked()
        self.expire(old); self.complete(context, sample)
        close = Runtime.close
        def incomplete(runtime):
            close(runtime)
            runtime.retirement_complete = False
        with mock.patch.object(Runtime, 'close', incomplete), self.assertRaises(NativeMCPTimeout):
            self.model.sample(context, tools=fixture.TOOLS)

    def test_external_retirement_during_recovery_validation_wins(self):
        context, sample, old = self.parked()
        self.expire(old); self.complete(context, sample)
        prompt = Snapshot.prompt
        def cancel(snapshot):
            self.model.retire()
            return prompt(snapshot)
        with mock.patch.object(Snapshot, 'prompt', cancel), self.assertRaises(NativeMCPTimeout):
            self.model.sample(context, tools=fixture.TOOLS)
        self.assertIsNone(self.model._runtime)

    def test_context_change_already_requires_cold_start(self):
        context, sample, old = self.parked()
        self.expire(old); self.complete(context, sample)
        context.append(Message('user', 'new instructions'))
        final = self.model.sample(context, tools=fixture.TOOLS)
        self.assertEqual(final.last_assistant_text, 'cold:new instructions')
        self.assertFalse(final.tool_calls)
        self.assertTrue(old.retirement_complete)

    def test_auto_recovers_without_a_failed_task_or_extra_watcher_turn(self):
        from pythia.interaction import auto
        from pythia.interaction._auto_config import resolve_config
        settings = resolve_config(overrides={'cwd': str(self.root), 'model_api': 'claude-relay',
                                             'model': 'fixture-model'}, roles=(1, -1))
        effects, reports = [], []
        class Watcher:
            def sample(self, context, **kwargs):
                reports.append([i.content_text for i in context if isinstance(i, Message) and i.role == 'user'][-1])
                return ModelSample((Message('assistant', 'Complete.'),))
        def effect(arguments, **kwargs):
            self.assertEqual(arguments, {'value': 1})
            effects.append(1)
            if len(effects) == 1:
                old = self.model._runtime
                self.wait_claim(old); self.expire(old)
            return ToolOutcome(f'host-{len(effects)}')
        def environment(index, args, tools):
            return (Environment((Tool(fixture.TOOLS[0], effect),)) if index == 1
                    else auto._environment_factory(index, args, tools))
        session = auto._Session(self.root / 'auto-recovery', settings,
                                model_factory=lambda index, args: self.model if index == 1 else Watcher(),
                                environment_factory=environment)
        try:
            session.start()
            task = session.submit('native-timeout')
            deadline = time.monotonic() + 10
            while session.task_result(task) is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(session.task_result(task))
            self.assertFalse(session.has_errors)
            self.assertEqual(len(effects), 2)
            self.assertEqual(len(reports), 1)
            self.assertIn('Outcome: ended', reports[0])
            self.assertNotIn('Outcome: failed', reports[0])
        finally:
            session.close()
