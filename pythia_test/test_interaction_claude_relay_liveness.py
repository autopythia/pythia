"""Keepalives and slow batch completion; only Python fixtures, never Claude."""
import concurrent.futures
from copy import deepcopy
from dataclasses import replace
import json
import os
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction.claude_relay import ClaudeRelayModel
from pythia.interaction.claude_relay._cli_protocol import Assembler
from pythia.interaction.claude_relay import _mcp
from pythia.interaction.claude_relay._mcp import Mailbox
from pythia.interaction.context import InteractionContext
from pythia.interaction.items import Message, ToolResult
from pythia.interaction.model import ModelError, ModelTimeoutError
from pythia_test import test_interaction_claude_relay as fixture
from pythia_test import test_interaction_claude_relay_protocol as protocol
from pythia_test import test_interaction_claude_relay_heartbeat as dispatch


PING = {'type': 'stream_event', 'event': {'type': 'ping'}}


class PingTests(unittest.TestCase):
    def unchanged(self, assembler):
        before = deepcopy(vars(assembler))
        self.assertIsNone(assembler.feed(PING))
        self.assertEqual(vars(assembler), before)

    def test_pings_preserve_all_state_and_output(self):
        plain, pinged = Assembler(), Assembler()
        self.unchanged(pinged)
        for record in protocol.observed_records():
            self.assertEqual(plain.feed(record), pinged.feed(record))
            self.unchanged(pinged)
            self.unchanged(pinged)

    def test_ping_during_tool_json_does_not_complete_or_modify_it(self):
        a = Assembler()
        a.feed(protocol.stream_event('message_start', message={'id': 'm'}))
        a.feed(protocol.stream_event('content_block_start', index=0, content_block={
            'type': 'tool_use', 'id': 'a', 'name': 'mcp__pythia__echo', 'input': {}}))
        a.feed(protocol.stream_event('content_block_delta', index=0,
                                    delta={'type': 'input_json_delta', 'partial_json': '{"value":'}))
        self.unchanged(a)
        a.feed(protocol.stream_event('content_block_delta', index=0,
                                    delta={'type': 'input_json_delta', 'partial_json': '1}'}))
        a.feed(protocol.stream_event('content_block_stop', index=0))
        self.unchanged(a)
        a.feed(protocol.stream_event('message_delta', delta={'stop_reason': 'tool_use'}, usage={'output_tokens': 3}))
        message = a.feed(protocol.stream_event('message_stop'))
        self.assertEqual(message.blocks[0]['input'], {'value': 1})
        self.assertEqual(message.usage.output_tokens, 3)
        self.unchanged(a)

    def test_errors_unknown_events_and_incomplete_batches_still_fail(self):
        for event in ({'type': 'error'}, {'type': 'fixture_unknown'}, {}):
            with self.subTest(event=event), self.assertRaises(ModelError):
                Assembler().feed({'type': 'stream_event', 'event': event})
        a = Assembler()
        self.unchanged(a)
        with self.assertRaises(ModelError):
            a.feed(protocol.stream_event('message_stop'))


class PingRuntimeTests(unittest.IsolatedAsyncioTestCase):
    run_records = dispatch.HeartbeatDispatchTests.run_records

    async def test_existing_runtime_guards_still_apply_to_ping(self):
        rt = await self.run_records([PING, *dispatch.FINAL])
        rt._failure.assert_not_called()
        for records, initialized, message in (
                ([PING, *dispatch.FINAL], False, 'before verified initialization'),
                ([*dispatch.FINAL, PING], True, 'after terminal result'),
                ([{**PING, 'parent_tool_use_id': 'subagent'}, *dispatch.FINAL], True, 'subagent'),
                ([{**PING, 'error': {'type': 'overloaded_error'}}, *dispatch.FINAL], True, 'model error')):
            with self.subTest(message=message):
                rt = await self.run_records(records, initialize=initialized)
                self.assertIn(message, str(rt._failure.call_args.args[0]))


class RegistrationWaitTests(unittest.TestCase):
    def test_registration_budget_is_independent_and_does_not_reset_on_wakeup(self):
        box = Mailbox(fixture.TOOLS, wait_seconds=.5, registration_seconds=25)
        clock = SimpleNamespace(now=0.0)
        def wake(timeout):
            clock.now += 11
        with mock.patch.object(_mcp, 'time', SimpleNamespace(monotonic=lambda: clock.now)), \
                mock.patch.object(box.condition, 'wait', side_effect=wake) as wait:
            with self.assertRaisesRegex(ModelError, 'never registered'):
                box.call('a', 'echo', {'value': 1})
        self.assertEqual([call.args[0] for call in wait.call_args_list], [25, 14, 3])

    def test_registration_after_old_grace_still_checks_identity_and_arguments(self):
        for wrong in (False, True):
            box = Mailbox(fixture.TOOLS, wait_seconds=.5, registration_seconds=25)
            clock = SimpleNamespace(now=0.0)
            def register(timeout):
                clock.now = 11
                box.register([('a', 'echo', {'value': 2 if wrong else 1})])
                box.release({'a': ('real result', True)})
            with self.subTest(wrong=wrong), mock.patch.object(_mcp, 'time', SimpleNamespace(monotonic=lambda: clock.now)), \
                    mock.patch.object(box.condition, 'wait', side_effect=register):
                if wrong:
                    with self.assertRaisesRegex(ModelError, 'does not match'):
                        box.call('a', 'echo', {'value': 1})
                else:
                    self.assertEqual(box.call('a', 'echo', {'value': 1})['content'][0]['text'], 'real result')


@unittest.skipIf(os.getuid() == 0, 'non-root relay fixture required')
class StreamLivenessTests(unittest.TestCase):
    setUp = fixture.ModelTests.setUp
    stop_broker = fixture.ModelTests.stop_broker

    def wait_early(self):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            rt = self.model._runtime
            if rt is not None:
                with rt.mailbox.condition:
                    if 'toolu_a' in rt.mailbox.early:
                        return rt
            time.sleep(.005)
        self.fail('Fixture callback did not arrive')

    def finish_stream(self, rt):
        async def send():
            rt.proc.stdin.write(b'fixture-finish\n')
            await rt.proc.stdin.drain()
        rt._submit(send(), 2)

    def test_early_callback_waits_for_whole_batch_past_old_grace(self):
        self.model.close()
        self.model = ClaudeRelayModel(replace(self.endpoint, generation_timeout_seconds=60, parked_timeout_seconds=2))
        self.addCleanup(self.model.close)
        context = InteractionContext((Message('user', 'early-callback'),))
        clock = SimpleNamespace(now=100.0)
        with mock.patch.object(_mcp, 'time', SimpleNamespace(monotonic=lambda: clock.now)), \
                concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(self.model.sample, context, tools=fixture.TOOLS)
            rt = self.wait_early()
            self.assertEqual(rt.mailbox.registration_seconds, 60)
            self.assertEqual(rt.mailbox.wait_seconds, 2)
            waited = threading.Event()
            wait = rt.mailbox.condition.wait
            def observed(timeout):
                waited.set()
                return wait(timeout)
            try:
                with mock.patch.object(rt.mailbox.condition, 'wait', side_effect=observed):
                    with rt.mailbox.condition:
                        clock.now += 11
                        rt.mailbox.condition.notify_all()
                    self.assertTrue(waited.wait(2))  # old min(10, parked) would have failed
                    self.assertFalse(future.done())
                    with rt.mailbox.condition:
                        self.assertEqual(rt.mailbox.slots, {})
                    self.assertTrue(rt.events.empty())
                    self.assertIsNone(rt.error)
                    self.finish_stream(rt)
                    sample = future.result(3)
                self.assertEqual(len(sample.tool_calls), 2)
                context.extend(sample.context_items())
                context.extend(ToolResult(c.call_id, f'result-{i}') for i,c in enumerate(sample.tool_calls))
                final = self.model.sample(context, tools=fixture.TOOLS)
                self.assertEqual(final.last_assistant_text, 'result-0|result-1')
                self.assertTrue(rt.mailbox.all_returned())
            finally:
                self.model.retire()  # never leave a blocked fixture on assertion failure

    def test_incomplete_batch_and_retirement_wake_early_callback(self):
        for finish in (True, False):
            context = InteractionContext((Message('user', 'early-callback-invalid'),))
            with self.subTest(finish=finish), concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(self.model.sample, context, tools=fixture.TOOLS)
                try:
                    rt = self.wait_early()
                    self.assertEqual(rt.mailbox.slots, {})
                    if finish:
                        self.finish_stream(rt)  # complete JSON, but max_tokens: no calls may execute
                    else:
                        self.model.retire()
                    with self.assertRaises(ModelError):
                        future.result(3)
                    self.assertTrue(rt.mailbox.closed)
                    self.assertEqual(rt.mailbox.slots, {})
                finally:
                    self.model.retire()

    def test_stream_pings_are_accepted_but_do_not_extend_generation_wait(self):
        final = self.model.sample(InteractionContext((Message('user', 'stream-ping'),)))
        self.assertEqual(final.last_assistant_text, 'cold:stream-ping')
        self.model.close()
        self.model = ClaudeRelayModel(replace(self.endpoint, generation_timeout_seconds=1))
        self.addCleanup(self.model.close)
        with self.assertRaises(ModelTimeoutError) as caught:
            self.model.sample(InteractionContext((Message('user', 'ping-only'),)))
        self.assertEqual(caught.exception.failure.last_event_type, 'stream_event/ping')
