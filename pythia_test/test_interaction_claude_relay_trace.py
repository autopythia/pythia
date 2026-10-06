"""Real Python relay + synthetic native protocol, never Claude."""
import base64
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from pythia.interaction._debug_trace import DebugTrace
from pythia.interaction.claude_relay import ClaudeRelayModel
from pythia.interaction.context import InteractionContext
from pythia.interaction.items import Message, ToolResult
from pythia.interaction.items import ModelFailure, ModelSampleBoundary
from pythia.interaction.compaction import PiCompactor
from pythia.interaction.save import load_interaction_save
from pythia.interaction.claude_relay._runtime import Runtime
from pythia.interaction.model import ModelError, ModelTimeoutError
from pythia_test import test_interaction_claude_relay as fixtures


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def payload(row):
    return (base64.b64decode(row['payload_base64']) if 'payload_base64' in row
            else row.get('payload', '').encode())


@unittest.skipIf(os.getuid() == 0, 'non-root relay fixture required')
class RelayTraceTests(unittest.TestCase):
    setUp = fixtures.ModelTests.setUp
    stop_broker = fixtures.ModelTests.stop_broker
    launch_options = fixtures.ModelTests.launch_options

    def traced(self, *, timeout=None):
        self.model.close()
        trace = DebugTrace.open(self.root / 'traced.jsonl', events=True, context_id=1, role='main')
        self.addCleanup(trace.close)
        endpoint = self.endpoint if timeout is None else replace(self.endpoint, generation_timeout_seconds=timeout)
        self.model = ClaudeRelayModel(endpoint, trace=trace)
        self.addCleanup(self.model.close)
        return trace

    def test_unknown_native_event_captured_before_failure_with_safe_metadata(self):
        trace = self.traced()
        with trace.operation('sample'), self.assertRaises(ModelError) as raised:
            self.model.sample(InteractionContext((Message('user', 'trace-error'),)))
        failure = raised.exception.failure
        self.assertEqual(failure.last_event_type, 'stream_event/error')
        self.assertEqual(failure.error_code, 'overloaded_error')
        self.assertGreater(failure.event_count, 64)
        self.assertEqual(len(failure.event_types), 64)
        self.assertGreater(failure.elapsed_seconds, 0)
        self.assertNotIn('PRIVATE_UPSTREAM_BODY', json.dumps(asdict(failure)))
        self.model.close(); trace.close()
        events = rows(trace.event_path)
        raw = b''.join(payload(r) for r in events if r['type'] == 'claude_stdout')
        self.assertIn(b'PRIVATE_UPSTREAM_BODY', raw)
        self.assertIn(b'"type": "error"', raw)
        stderr = b''.join(payload(r) for r in events if r['type'] == 'claude_stderr')
        self.assertIn(b'private-native-stderr\xff', stderr)
        self.assertTrue(any(r['type'] == 'native_failure' for r in events))
        self.assertTrue(any(r['type'] == 'process_exit' for r in events))
        self.assertTrue(all(r['context_id'] == 1 for r in events))

    def test_without_trace_failure_metadata_is_still_useful(self):
        for text, expected in (('trace-unknown', 'stream_event/fixture_unknown'), ('trace-missing', 'stream_event/<missing>'),
                               ('trace-malformed', 'invalid_record')):
            with self.subTest(text=text), self.assertRaises(ModelError) as raised:
                self.model.sample(InteractionContext((Message('user', text),)))
            self.assertEqual(raised.exception.failure.last_event_type, expected)
        self.assertFalse(list(self.root.glob('*.trace.*')))

    def test_timeout_reports_partial_byte_activity_without_extending_deadline(self):
        # Leave enough time for the disposable Python relay/fixture to start;
        # the fixture streams partial bytes for two seconds without completing.
        trace = self.traced(timeout=1)
        with trace.operation('sample'), self.assertRaises(ModelTimeoutError) as raised:
            self.model.sample(InteractionContext((Message('user', 'trace-progress'),)))
        failure = raised.exception.failure
        self.assertIn('budget=1s', failure.message)
        self.assertNotIn('last_stdout_byte_age=none', failure.message)
        self.model.close(); trace.close()
        events = rows(trace.event_path)
        timeout = next(r for r in events if r['type'] == 'generation_timeout')
        self.assertLess(timeout['last_byte_age'], 1)
        self.assertTrue(any(r['type'] == 'claude_stdout' and payload(r).startswith(b'{') for r in events))

    def test_warm_mcp_attribution_and_no_deliberate_bearer_dump(self):
        trace = self.traced()
        context = InteractionContext((Message('user', 'two'),))
        with trace.operation('sample', context_revision=len(context)):
            sample = self.model.sample(context, tools=fixtures.TOOLS)
        old = self.model._runtime
        token = old.mcp.token
        context.extend(sample.context_items())
        context.extend(ToolResult(call.call_id, 'real-host-result') for call in sample.tool_calls)
        with trace.operation('sample', context_revision=len(context)):
            final = self.model.sample(context, tools=fixtures.TOOLS)
        self.assertEqual(final.last_assistant_text, 'real-host-result|real-host-result')
        trace.close()
        events = rows(trace.event_path)
        begins = [r for r in events if r['type'] == 'sample_begin']
        self.assertEqual(len(begins), 2)
        self.assertNotEqual(begins[0]['operation_id'], begins[1]['operation_id'])
        release = next(r for r in events if r['type'] == 'mailbox_release')
        self.assertEqual(release['sample_id'], begins[1]['sample_id'])
        requests = [r for r in events if r['type'] == 'mcp_request_meta' and r['method'] == 'tools/call']
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(r['sample_id'] == begins[0]['sample_id'] for r in requests))
        self.assertTrue(all(r['runtime_id'] == old.generation for r in requests))
        self.assertNotIn(token, trace.event_path.read_text())
        self.assertEqual(trace.request_path.read_text(), '')

    def test_frontend_failures_save_metadata_and_keep_payload_only_in_trace(self):
        for frontend in ('cli', 'auto'):
            save = self.root / (frontend + ('.jsonl' if frontend == 'cli' else ''))
            opts = ['--resume', 'False'] if frontend == 'cli' else ['--watcher-observe-only']
            run = subprocess.run([
                '/usr/bin/python3', '-I', '-S', '-c',
                'import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); '
                'runpy.run_module(sys.argv.pop(1),run_name="__main__",alter_sys=True)',
                str(fixtures.ROOT), 'pythia.interaction.' + frontend, *self.launch_options(),
                '--headless', '--prompt', 'trace-error', '--debug-trace',
                '--cwd', str(self.root), '--save', str(save), *opts,
            ], cwd=fixtures.ROOT, env={**os.environ, 'HOME': str(self.root)},
                capture_output=True, text=True, timeout=15)
            with self.subTest(frontend=frontend):
                self.assertEqual(run.returncode, 1)
                context_path = save if frontend == 'cli' else save / 'contexts/1.jsonl'
                context = load_interaction_save(context_path)
                failure = next(i for i in context if isinstance(i, ModelFailure))
                self.assertEqual(failure.last_event_type, 'stream_event/error')
                self.assertNotIn('PRIVATE_UPSTREAM_BODY', context_path.read_text() + run.stdout + run.stderr)
                events = rows(context_path.with_suffix('.trace.events.jsonl'))
                self.assertTrue(any(r['type'] == 'sample_begin' and r['op'] == 'sample' for r in events))
                self.assertIn(b'PRIVATE_UPSTREAM_BODY', b''.join(payload(r) for r in events if r['type'] == 'claude_stdout'))

    def test_pi_scopes_use_new_runtimes_and_trace_is_not_context_input(self):
        trace = self.traced()
        context = InteractionContext((Message('user', 'old context ' * 100), Message('assistant', 'old answer'),
                                      ModelSampleBoundary(), Message('user', 'two')))
        with trace.operation('sample'):
            sample = self.model.sample(context, tools=fixtures.TOOLS)
        old = self.model._runtime
        context.extend(sample.context_items())
        context.extend(ToolResult(c.call_id, 'checkpointed') for c in sample.tool_calls)
        with trace.operation('compact'):
            result = PiCompactor(self.model, keep_recent_tokens=1).compact(context)
        self.assertTrue(old.closed.is_set())
        self.assertTrue(all(s['result'] is None for s in old.mailbox.slots.values()))
        context.extend(result.context_items())
        with trace.operation('sample'):
            final = self.model.sample(context, tools=fixtures.TOOLS)
        self.assertFalse(final.tool_calls)
        self.model.close(); trace.close()
        events = rows(trace.event_path)
        summaries = [r for r in events if r['type'] == 'sample_begin' and r['op'] == 'compact']
        self.assertTrue(summaries)
        self.assertGreaterEqual(len({r['runtime_id'] for r in events if r['type'] == 'claude_invocation'}), 3)
        native_input = b''.join(payload(r) for r in events if r['type'] == 'claude_stdin')
        self.assertNotIn(b'pythia.debug-event', native_input)
        self.assertNotIn(b'trace.events.jsonl', native_input)

    def test_cleanup_error_preserves_primary_failure(self):
        trace = self.traced()
        real_close = Runtime.close
        def close_then_fail(runtime):
            real_close(runtime)
            raise RuntimeError('PRIVATE_CLEANUP_BODY')
        with mock.patch.object(Runtime, 'close', close_then_fail), self.assertRaises(ModelError) as raised:
            self.model.sample(InteractionContext((Message('user', 'trace-unknown'),)))
        self.assertEqual(raised.exception.failure.last_event_type, 'stream_event/fixture_unknown')
        self.assertIn('cleanup failed (RuntimeError)', raised.exception.failure.message)
        self.assertNotIn('PRIVATE_CLEANUP_BODY', raised.exception.failure.message)
        trace.close()
