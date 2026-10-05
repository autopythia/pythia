"""Pure policy/codec tests: never contact Claude or change the running broker."""
from argparse import Namespace
import os
import unittest
from unittest import mock

from pythia.interaction.claude_relay._sampling import EFFORT_LEVELS, resolve_extra, resolve_sampling
from pythia.interaction.claude_relay._cli_protocol import Assembler
from pythia.interaction.model import ModelConfigurationError, ModelResponseError, SampleParams, ModelSample
from pythia.interaction.model_catalog import BUILTIN_MODEL_CATALOG, EndpointSpec, ModelBinding
from pythia.interaction.model_catalog_config import parse_model_catalog
from pythia.interaction.model_config import build_model, build_parser, relay_endpoint
from pythia.interaction.compaction import PiCompactor, CompactionError
from pythia.interaction.context import InteractionContext
from pythia.interaction.items import Message, ModelSampleBoundary


def binding(extra=None):
    return ModelBinding('test', EndpointSpec('claude-relay', None, 'claude-opus-5-5', 'runtime'),
                        extra_sample_params=extra or {})


class RelayLaunchConfigurationTests(unittest.TestCase):
    def environment(self):
        return {'CLAUDE_RELAY_LAUNCHER': '/trusted/claude_relay.py',
                'CLAUDE_RELAY_SOCKET': '/tmp/relay/control.sock',
                'CLAUDE_RELAY_SERVER_UID': '1003', 'CLAUDE_RELAY_CLI_VERSION': '2.1.289'}

    def test_missing_or_empty_launcher_explains_export_and_no_dotenv_loading(self):
        for value in (None, ''):
            env = self.environment()
            if value is None:
                del env['CLAUDE_RELAY_LAUNCHER']
            else:
                env['CLAUDE_RELAY_LAUNCHER'] = value
            with self.subTest(value=value), mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ModelConfigurationError) as raised:
                    relay_endpoint(Namespace(), binding())
                message = str(raised.exception)
                for expected in ('launcher is not configured', '--claude-relay-launcher',
                                 'CLAUDE_RELAY_LAUNCHER', 'A-side claude_relay.py',
                                 '.env files are not loaded automatically'):
                    self.assertIn(expected, message)

    def test_environment_fallback_and_explicit_flag_precedence(self):
        with mock.patch.dict(os.environ, self.environment(), clear=True):
            endpoint = relay_endpoint(Namespace(), binding())
            self.assertEqual(endpoint.launcher, '/trusted/claude_relay.py')
            self.assertEqual(endpoint.socket_path, '/tmp/relay/control.sock')
            self.assertEqual(endpoint.server_uid, 1003)
            explicit = relay_endpoint(Namespace(claude_relay_launcher='/other/client.py'), binding())
            self.assertEqual(explicit.launcher, '/other/client.py')

    def test_relative_and_unexpanded_launcher_paths_still_fail(self):
        for value in ('claude-relay/claude_relay.py', '$PYTHIA_REPO/claude-relay/claude_relay.py',
                      '~/claude-relay/claude_relay.py'):
            env = {**self.environment(), 'CLAUDE_RELAY_LAUNCHER': value, 'PYTHIA_REPO': '/trusted'}
            with self.subTest(value=value), mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ModelConfigurationError, 'launcher must be an absolute path'):
                    relay_endpoint(Namespace(), binding())

    def test_missing_socket_has_its_own_configuration_hint(self):
        env = self.environment()
        del env['CLAUDE_RELAY_SOCKET']
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ModelConfigurationError,
                                        'socket_path is not configured.*--claude-relay-socket.*CLAUDE_RELAY_SOCKET'):
                relay_endpoint(Namespace(), binding())


class SamplingTests(unittest.TestCase):
    def test_messages_convention_and_exact_levels(self):
        for effort in EFFORT_LEVELS:
            policy = resolve_extra({'output_config': {'effort': effort}})
            self.assertEqual(policy.cli_args(), ['--effort', effort])
        self.assertEqual(resolve_extra({}).cli_args(), [])
        self.assertEqual(resolve_extra({'output_config': {}}), resolve_extra({}))

    def test_binding_overlay_and_call_replacement_have_existing_precedence(self):
        base = binding({'output_config': {'effort': 'max'}})
        launch = base.with_extra_sample_params({'output_config': {'effort': 'high'}})
        self.assertEqual(resolve_sampling(base, None).effort, 'max')
        self.assertEqual(resolve_sampling(launch, SampleParams()).effort, 'high')
        self.assertEqual(resolve_sampling(launch, SampleParams(extra={'output_config': {'effort': 'low'}})).effort, 'low')
        self.assertEqual(resolve_sampling(base, None), resolve_sampling(base, SampleParams(extra={'output_config': {'effort': 'max'}})))
        self.assertIsNone(resolve_sampling(launch, SampleParams(extra={})).effort)
        self.assertIsNone(resolve_sampling(launch, SampleParams(extra={'output_config': {}})).effort)
        self.assertIsNone(resolve_sampling(base.with_extra_sample_params({'output_config': {}}), None).effort)
        self.assertEqual(base.extra_sample_params['output_config']['effort'], 'max')

    def test_invalid_or_unverified_settings_are_rejected(self):
        for value in ({'effort': 'max'}, {'output_config.effort': 'max'}, {'thinking': {'type': 'adaptive'}},
                      {'output_config': None}, {'output_config': []}, {'output_config': {'format': {}}},
                      *({'output_config': {'effort': x}} for x in (None, True, 7, [], 'MAX', 'auto', ' max', ''))):
            with self.subTest(value=value), self.assertRaises(ModelConfigurationError):
                resolve_extra(value)
        for params in (SampleParams(max_output_tokens=1), SampleParams(max_output_tokens=128000),
                       SampleParams(temperature=0), SampleParams(top_p=1), SampleParams(seed=1), SampleParams(stop=('STOP',))):
            with self.subTest(params=params), self.assertRaises(ModelConfigurationError):
                resolve_sampling(binding(), params)

    def test_catalog_native_capacities_are_preserved_not_requested_budgets(self):
        native = BUILTIN_MODEL_CATALOG.get_model_spec('messages', 'claude-opus-5.5')
        limits = native.limits
        catalog = parse_model_catalog(f'''[catalog]
version=4
[model.opus55-relay-max]
endpoint.api=claude-relay
endpoint.model={native.endpoint.model}
limits.max_context_tokens={limits.max_context_tokens}
limits.max_output_tokens={limits.max_output_tokens}
limits.auto_compact_context_tokens={limits.auto_compact_context_tokens}
extra_sample_params.output_config={{"effort":"max"}}
''')
        resolved = catalog.bind(None, 'opus55-relay-max')
        self.assertEqual(resolved.limits, limits)
        self.assertEqual(resolve_sampling(resolved, None).cli_args(), ['--effort', 'max'])
        args = build_parser('test').parse_args([
            '--model', 'opus55-relay-max', '--claude-relay-launcher', '/fixture/client.py',
            '--claude-relay-socket', '/fixture/control.sock', '--claude-relay-server-uid', '1003',
            '--claude-relay-cli-version', '2.1.289'])
        model = build_model(args, catalog=catalog)
        try:
            self.assertEqual(model.endpoint.model, native.endpoint.model)
            self.assertEqual(model.binding.limits, limits)
            self.assertIsNone(model._runtime)
        finally:
            model.close()


class StopReasonTests(unittest.TestCase):
    def message(self, reason='end_turn', *, ident='m', usage=None, blocks=None, assembler=None):
        return (assembler or Assembler()).feed({'type': 'assistant', 'message': {
            'id': ident, 'content': blocks or [{'type': 'text', 'text': 'partial or full text'}],
            'usage': usage or {}, 'stop_reason': reason}})

    def test_reasons_are_preserved_and_length_normalized(self):
        for reason in ('end_turn', 'stop_sequence', 'tool_use', 'max_tokens', 'length', 'refusal',
                       'pause_turn', 'model_context_window_exceeded'):
            message = self.message(reason)
            self.assertEqual(message.stop_reason, 'max_tokens' if reason == 'length' else reason)
        for reason in (None, '', False, {}, 'unknown', 'compaction'):
            with self.subTest(reason=reason), self.assertRaises(ModelResponseError):
                self.message(reason)

    def test_limited_or_contradictory_tool_batches_never_produce_calls(self):
        block = {'type': 'tool_use', 'id': 'toolu_a', 'name': 'mcp__pythia__echo', 'input': {}}
        for reason in ('max_tokens', 'length', 'refusal', 'pause_turn', 'end_turn', 'model_context_window_exceeded'):
            with self.subTest(reason=reason), self.assertRaises(ModelResponseError):
                self.message(reason, blocks=[block]).sample_items('generation', {'mcp__pythia__echo': 'echo'})
        with self.assertRaises(ModelResponseError):
            self.message('tool_use').sample_items('generation', {})

    def test_duplicate_stop_or_usage_conflict_is_not_hidden_by_dedup(self):
        for changed in ('reason', 'usage'):
            assembler = Assembler()
            self.message(usage={'input_tokens': 12, 'output_tokens': 3}, assembler=assembler)
            self.assertIsNone(self.message(None, assembler=assembler))  # absent duplicate metadata is not a new claim
            with self.subTest(changed=changed), self.assertRaises(ModelResponseError):
                self.message('length' if changed == 'reason' else 'end_turn',
                             usage={'input_tokens': 14} if changed == 'usage' else {}, assembler=assembler)

    def test_completed_hint_can_supply_missing_stop_without_hiding_a_conflict(self):
        assembler = Assembler()
        assembler.feed({'type': 'stream_event', 'event': {'type': 'message_start', 'message': {'id': 'm', 'usage': {'input_tokens': 2}}}})
        block = {'type': 'text', 'text': 'text'}
        assembler.feed({'type': 'stream_event', 'event': {'type': 'content_block_start', 'index': 0, 'content_block': block}})
        assembler.feed({'type': 'stream_event', 'event': {'type': 'content_block_stop', 'index': 0}})
        self.message('max_tokens', blocks=[block], usage={'input_tokens': 2, 'output_tokens': 1}, assembler=assembler)
        message = assembler.feed({'type': 'stream_event', 'event': {'type': 'message_stop'}})
        self.assertEqual(message.stop_reason, 'max_tokens')
        self.assertEqual(message.usage.total_tokens, 3)

    def test_pi_rejects_all_incomplete_summary_aliases(self):
        context = InteractionContext((Message('user', 'older context ' * 100), Message('assistant', 'old answer'),
                                      ModelSampleBoundary(), Message('user', 'recent request')))
        before = context.items
        for reason in ('max_tokens', 'length', 'refusal', 'pause_turn', 'model_context_window_exceeded'):
            class SummaryModel:
                def sample(self, *args, **kwargs):
                    return ModelSample((Message('assistant', 'incomplete summary'),), stop_reason=reason)
            with self.subTest(reason=reason), self.assertRaises(CompactionError):
                PiCompactor(SummaryModel(), keep_recent_tokens=0).compact(context)
            self.assertEqual(context.items, before)


if __name__ == '__main__':
    unittest.main()
