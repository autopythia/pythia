"""Progress telemetry is not a tool call, subagent, completion or keepalive budget."""
import asyncio
from copy import deepcopy
import json
import os
import queue
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction._debug_trace import DebugTrace
from pythia.interaction.claude_relay import ClaudeRelayModel
from pythia.interaction.claude_relay._diagnostics import Diagnostics
from pythia.interaction.claude_relay._mcp import Mailbox
from pythia.interaction.claude_relay._runtime import Runtime
from pythia.interaction.claude_relay._trace import RelayTrace
from pythia.interaction.context import InteractionContext
from pythia.interaction.items import Message, ToolResult
from pythia.interaction.model import ModelTimeoutError
from pythia_test import test_interaction_claude_relay as fixture


def heartbeat(**fields):
    return {'type': 'tool_progress', 'heartbeat': True, 'tool_use_id': 'call-heartbeat-0',
            'parent_tool_use_id': 'call', 'tool_name': 'mcp__pythia__echo',
            'elapsed_time_seconds': 30, **fields}


FINAL = [
    {'type': 'assistant', 'message': {'id': 'm', 'content': [{'type': 'text', 'text': 'done'}],
                                     'stop_reason': 'end_turn', 'usage': {'input_tokens': 1, 'output_tokens': 2}}},
    {'type': 'result', 'subtype': 'success', 'is_error': False},
]


class HeartbeatDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def run_records(self, records, *, initialize=True, known_call=False):
        rt = Runtime.__new__(Runtime)  # no subprocess or runtime thread
        rt.diagnostics = Diagnostics()
        rt.trace = RelayTrace(None, 'fixture')
        rt.trace.emit = mock.Mock()
        rt.mailbox = Mailbox(fixture.TOOLS if known_call else ())
        rt.names = {'mcp__pythia__echo': 'echo'} if known_call else {}
        if known_call:
            rt.mailbox.register([('call', 'echo', {})])
            rt.mailbox.release({'call': ('real result', True)})
            rt.mailbox.call('call', 'echo', {})
        before = deepcopy(rt.mailbox.slots)
        rt.initialized = rt.result_seen = False
        rt.final_message = None
        rt.generation = 'fixture'
        rt.endpoint = SimpleNamespace(stop_timeout_seconds=1, generation_timeout_seconds=.1)
        rt.closed = threading.Event()
        rt.error = None
        rt.proc = SimpleNamespace(stdout=asyncio.StreamReader(), stdin=mock.Mock(), wait=mock.AsyncMock(return_value=0))
        rt._put = mock.Mock()
        rt._failure = mock.Mock()
        rt.tasks = [None, asyncio.create_task(asyncio.sleep(0))]
        initial = [{'type': 'system', 'subtype': 'init', 'tools': list(rt.names),
                    'mcp_servers': [{'name': 'pythia', 'status': 'connected'}]}] if initialize else []
        for record in initial + records:
            rt.proc.stdout.feed_data((json.dumps(record) + '\n').encode())
        rt.proc.stdout.feed_eof()
        await rt._stdout()
        await rt.tasks[1]
        self.assertEqual(rt.mailbox.slots, before)
        return rt

    async def test_heartbeat_only_changes_diagnostics_not_mailbox_or_completion(self):
        rt = await self.run_records([heartbeat(), heartbeat(tool_use_id='call-heartbeat-1'), heartbeat()])
        # EOF without a model message still fails; heartbeats cannot complete it.
        self.assertIn('without a successful run result', str(rt._failure.call_args.args[0]))
        rt._put.assert_not_called()
        self.assertIsNone(rt.final_message)
        snapshot = rt.diagnostics.snapshot()
        self.assertEqual(snapshot.last_event_type, 'tool_progress/heartbeat')
        self.assertEqual(snapshot.phase, 'awaiting_model_message')
        self.assertIsNone(snapshot.last_completion_age)
        rt.events = mock.Mock()
        rt.events.get.side_effect = queue.Empty
        with self.assertRaises(ModelTimeoutError):
            rt.next_message()
        rt.events.get.assert_called_once_with(timeout=.1)

    async def test_payload_validation_is_intentionally_deferred(self):
        for record in (heartbeat(), {'type': 'tool_progress', 'heartbeat': True},
                       heartbeat(parent_tool_use_id={'unknown': 'parent'}, tool_name='unknown',
                                 tool_use_id=None, elapsed_time_seconds=-1)):
            with self.subTest(record=record):
                rt = await self.run_records([record, *FINAL])
                rt._failure.assert_not_called()
                self.assertEqual(rt._put.call_count, 1)
                message, _, calls = rt._put.call_args.args[0]
                self.assertFalse(calls)
                self.assertEqual(message.usage.total_tokens, 3)
                self.assertTrue(any(call.args[0] == 'tool_heartbeat'
                                    and call.kwargs['validation'] == 'deferred' for call in rt.trace.emit.call_args_list))

    async def test_ordinary_progress_and_late_heartbeat_keep_existing_checks(self):
        rt = await self.run_records([{'type': 'tool_progress', 'tool_use_id': 'call'}, heartbeat(), *FINAL], known_call=True)
        rt._failure.assert_not_called()  # already returned call; no state changes
        rt = await self.run_records([{'type': 'tool_progress', 'tool_use_id': 'unknown'}, *FINAL])
        self.assertIn('Progress for an unknown native tool', str(rt._failure.call_args.args[0]))

    async def test_only_exact_heartbeat_discriminator_exempts_the_parent_guard(self):
        records = [heartbeat(type=kind) for kind in ('assistant', 'user', 'stream_event', 'control_request')]
        records += [heartbeat(heartbeat=value) for value in (False, None, 1, 'true')]
        records += [{'type': 'tool_progress', 'parent_tool_use_id': 'call', 'tool_use_id': 'call'}]
        for record in records:
            with self.subTest(record=record):
                rt = await self.run_records([record, *FINAL])
                self.assertIn('Native subagent output is unsupported', str(rt._failure.call_args.args[0]))

    async def test_heartbeat_does_not_bypass_init_terminal_or_native_errors(self):
        rt = await self.run_records([heartbeat(), *FINAL], initialize=False)
        self.assertIn('before verified initialization', str(rt._failure.call_args.args[0]))
        rt = await self.run_records([*FINAL, heartbeat()])
        self.assertIn('after terminal result', str(rt._failure.call_args.args[0]))
        rt = await self.run_records([heartbeat(error={'type': 'overloaded_error'}), *FINAL])
        self.assertIn('Native CLI reported a model error', str(rt._failure.call_args.args[0]))


@unittest.skipIf(os.getuid() == 0, 'non-root Python relay fixture required')
class HeartbeatHandoffTests(unittest.TestCase):
    setUp = fixture.ModelTests.setUp
    stop_broker = fixture.ModelTests.stop_broker

    def test_withheld_batch_survives_heartbeats_and_reuses_the_native_process(self):
        for tracing in (False, True):
            with self.subTest(tracing=tracing):
                self.model.close()
                trace = DebugTrace.open(self.root / 'heartbeat.jsonl', events=True) if tracing else None
                if trace:
                    self.addCleanup(trace.close)
                self.model = ClaudeRelayModel(self.endpoint, trace=trace)
                self.addCleanup(self.model.close)
                context = InteractionContext((Message('user', 'heartbeat-batch'),))
                sample = self.model.sample(context, tools=fixture.TOOLS)
                old = self.model._runtime
                self.assertEqual(len(sample.tool_calls), 2)
                self.assertEqual(sample.usage.total_tokens, 15)
                context.extend(sample.context_items())
                # First host outcome is ready, but do not release an incomplete batch.
                context.append(ToolResult(sample.tool_calls[0].call_id, 'fast'))
                deadline = time.monotonic() + 3
                while ('tool_progress/heartbeat' not in old.event_types and old.error is None
                       and time.monotonic() < deadline):
                    time.sleep(.01)
                self.assertIsNone(old.error)
                self.assertIn('tool_progress/heartbeat', old.event_types)
                with old.mailbox.condition:
                    self.assertEqual(set(old.mailbox.slots), {'toolu_a', 'toolu_b'})
                    self.assertTrue(all(slot['result'] is None and not slot['returned']
                                        for slot in old.mailbox.slots.values()))
                self.assertTrue(old.events.empty())
                self.assertIsNone(old.final_message)
                self.assertIsNone(old.proc.returncode)
                context.append(ToolResult(sample.tool_calls[1].call_id, 'slow'))
                with mock.patch('pythia.interaction.claude_relay._model.Runtime', side_effect=AssertionError('unexpected restart')):
                    final = self.model.sample(context, tools=fixture.TOOLS)
                self.assertEqual(final.last_assistant_text, 'fast|slow')
                self.assertEqual(final.usage.total_tokens, 9)
                self.assertTrue(old.mailbox.all_returned())
                self.assertIsNone(self.model._runtime)
                if trace:
                    trace.close()
                    events = [json.loads(line) for line in trace.event_path.read_text().splitlines()]
                    pulses = [row for row in events if row['type'] == 'tool_heartbeat']
                    self.assertTrue(pulses)
                    self.assertTrue(all(row['validation'] == 'deferred' for row in pulses))
                    first = next(row for row in events if row['type'] == 'sample_begin')
                    self.assertTrue(all(row['sample_id'] == first['sample_id'] for row in pulses))
