"""Opt-in, billable native checks through an ALREADY RUNNING relay; stdlib only.

Normal test discovery skips these. Set PYTHIA_TEST_CLAUDE_RELAY_LIVE=1 plus
CLAUDE_RELAY_{LAUNCHER,SOCKET,SERVER_UID,CLI_VERSION,TEST_MODEL} explicitly.
No download, login, broker restart, native execution under A, shell tool or
large-context stress test. Only an in-memory receipt tool and private temp saves.
"""
from dataclasses import replace
import os
from pathlib import Path
import secrets
import tempfile
import time
import unittest
from unittest import mock

from pythia.interaction.claude_relay import ClaudeRelayEndpoint, ClaudeRelayModel
from pythia.interaction.claude_relay._runtime import Runtime
from pythia.interaction.compaction import PiCompactor, auto_compaction_due
from pythia.interaction.context import InteractionContext
from pythia.interaction.environment import Environment, Tool, ToolOutcome, ToolSpec
from pythia.interaction.items import ContextPrefix, Instructions, Message, ModelSampleBoundary, ToolResult
from pythia.interaction.model import SampleParams
from pythia.interaction.runtime_config import InteractionConfigSnapshot
from pythia.interaction.save import load_interaction_save, save_interaction_save


@unittest.skipUnless(os.environ.get('PYTHIA_TEST_CLAUDE_RELAY_LIVE') == '1', 'explicit live relay opt-in required')
class LiveRelayTests(unittest.TestCase):
    def setUp(self):
        def setting(name):
            value = os.environ.get('CLAUDE_RELAY_' + name)
            if not value:
                self.fail('Live check requires CLAUDE_RELAY_' + name)
            return value
        endpoint = ClaudeRelayEndpoint(
            setting('TEST_MODEL'), setting('LAUNCHER'), setting('SOCKET'),
            int(setting('SERVER_UID')), setting('CLI_VERSION'),
            generation_timeout_seconds=90, parked_timeout_seconds=120)
        binding = endpoint.binding.with_extra_sample_params({'output_config': {'effort': 'max'}})
        self.model = ClaudeRelayModel(replace(endpoint, binding=binding))
        self.addCleanup(self.model.close)
        self.tmp = tempfile.TemporaryDirectory(prefix='pythia-relay-live-')
        self.addCleanup(self.tmp.cleanup)
        self.save = Path(self.tmp.name) / 'conversation.jsonl'
        self.executions = 0
        self.receipt = 'PYTHIA_RECEIPT_' + secrets.token_hex(8)
        self.environment = Environment((Tool(
            ToolSpec('probe_echo', 'Return a unique test receipt for a value.',
                     {'type': 'object', 'properties': {'value': {'type': 'integer'}},
                      'required': ['value'], 'additionalProperties': False}),
            self.echo),))

    def echo(self, arguments, **kwargs):
        self.assertEqual(arguments, {'value': 17})
        self.executions += 1
        return ToolOutcome(self.receipt)

    def checkpoint(self, context):
        save_interaction_save(self.save, context)
        self.assertEqual(load_interaction_save(self.save).items, context.items)

    def parked(self, *, older_history=False):
        older = (Message('user', 'Earlier context: this is a harmless relay check; do not access files.'),
                 Message('assistant', 'Understood.'), ModelSampleBoundary()) if older_history else ()
        context = InteractionContext((
            Instructions('Use only supplied tools. Never fabricate a tool result or repeat a completed call.'), *older,
            Message('user', 'Call probe_echo exactly once with value 17. After receiving its result, '
                    'reply with the returned receipt exactly, with no other text.')))
        sample = self.model.sample(context, tools=self.environment.tool_specs)
        self.assertEqual(sample.stop_reason, 'tool_use')
        self.assertEqual(len(sample.tool_calls), 1)
        self.assertGreater(sample.usage.output_tokens, 0)
        context.extend(sample.context_items())
        self.checkpoint(context)
        runtime = self.model._runtime
        self.assertIsNotNone(runtime)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with runtime.mailbox.condition:
                claimed = all(slot['claimed'] for slot in runtime.mailbox.slots.values())
            if claimed or runtime.error:
                break
            time.sleep(.02)
        self.assertIsNone(runtime.error)
        self.assertTrue(claimed, 'Native callback did not claim the streamed tool ID')
        time.sleep(.2)
        self.assertFalse(runtime.mailbox.all_returned())
        self.assertIsNone(runtime.final_message)
        self.assertTrue(runtime.events.empty())
        self.assertEqual(self.executions, 0)  # MCP is a mailbox, not an executor
        self.assertTrue(runtime.mcp.initialized)
        outcomes = self.environment.execute_tool_calls(sample.tool_calls)
        self.assertTrue(all(item.success for item in outcomes.items))
        context.extend(outcomes.context_items())
        self.checkpoint(context)  # effects are durable BEFORE any native release
        self.assertEqual(self.executions, 1)
        return context, runtime

    def final(self, context, **kwargs):
        sample = self.model.sample(context, tools=self.environment.tool_specs, **kwargs)
        self.assertEqual(sample.stop_reason, 'end_turn')
        self.assertFalse(sample.tool_calls)
        self.assertEqual(sample.last_assistant_text.strip(), self.receipt)
        self.assertGreater(sample.usage.output_tokens, 0)
        self.assertEqual(self.executions, 1)
        self.assertIsNone(self.model._runtime)
        context.extend(sample.context_items())
        self.checkpoint(context)

    def test_text_default_thinking_and_explicit_max_effort(self):
        for extra in ({}, None):  # clear effort, then inherit catalog max
            with self.subTest(extra=extra):
                sample = self.model.sample(InteractionContext((
                    Message('user', 'What is 19 times 23? Return only the integer; use no tools.'),)),
                    sample_params=SampleParams(extra=extra))
                self.assertEqual(sample.last_assistant_text.strip(), '437')
                self.assertEqual(sample.stop_reason, 'end_turn')
                self.assertGreater(sample.usage.output_tokens, 0)
                self.assertIsNone(self.model._runtime)

    def test_mcp_auth_native_id_park_and_warm_handoff(self):
        context, old = self.parked()
        # Same effective setting, supplied differently: no new native process.
        with mock.patch('pythia.interaction.claude_relay._model.Runtime', side_effect=Runtime) as starts:
            self.final(context, sample_params=SampleParams(extra={'output_config': {'effort': 'max'}}))
            starts.assert_not_called()
        self.assertTrue(old.mailbox.all_returned())
        self.assertTrue(old.closed.is_set())

    def test_effort_change_retires_before_release_without_effect_replay(self):
        context, old = self.parked()
        with mock.patch('pythia.interaction.claude_relay._model.Runtime', side_effect=Runtime) as starts:
            self.final(context, sample_params=SampleParams(extra={'output_config': {'effort': 'high'}}))
            self.assertEqual(starts.call_count, 1)
        self.assertTrue(old.closed.is_set())
        self.assertTrue(all(slot['result'] is None for slot in old.mailbox.slots.values()))

    def test_pi_summary_and_fresh_post_compaction_continuation(self):
        context, old = self.parked(older_history=True)
        self.assertTrue(auto_compaction_due(self.model, context, InteractionConfigSnapshot(auto_compact_tokens=1)))
        before = context.items
        summaries = []
        def start(*args, **kwargs):
            runtime = Runtime(*args, **kwargs)
            summaries.append(runtime)
            return runtime
        with mock.patch('pythia.interaction.claude_relay._model.Runtime', side_effect=start):
            compacted = PiCompactor(self.model, keep_recent_tokens=1).compact(
                context, tools=self.environment.tool_specs)
        self.assertEqual(context.items, before)
        self.assertTrue(old.closed.is_set())
        self.assertTrue(all(slot['result'] is None for slot in old.mailbox.slots.values()))
        self.assertTrue(summaries)
        for runtime in summaries:
            self.assertIsNone(runtime.mcp)
            self.assertEqual(runtime.sampling.effort, 'max')
            self.assertEqual(runtime.environment()['DISABLE_AUTO_COMPACT'], '1')
            self.assertTrue(runtime.closed.is_set())
        self.assertGreater(compacted.usage.output_tokens, 0)
        context.extend(compacted.context_items())
        self.assertTrue(any(isinstance(item, ContextPrefix) for item in context.items))
        self.assertTrue(any(isinstance(item, ToolResult) and item.output == self.receipt for item in context.model_items()))
        self.checkpoint(context)
        with mock.patch('pythia.interaction.claude_relay._model.Runtime', side_effect=Runtime) as starts:
            self.final(context)
            self.assertEqual(starts.call_count, 1)


if __name__ == '__main__':
    unittest.main()
