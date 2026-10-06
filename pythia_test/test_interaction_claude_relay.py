"""All-Python fixtures: never download, discover, or execute Claude."""
import concurrent.futures
from dataclasses import replace
import json
import http.client
import socket
import os
import signal
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction.claude_relay import ClaudeRelayEndpoint, ClaudeRelayModel
from pythia.interaction.context import InteractionContext
from pythia.interaction.environment import ToolSpec
from pythia.interaction.items import Instructions, Message, ToolCall, ToolResult, Tools, SampleMetadata, ModelSampleBoundary, ContextPrefix
from pythia.interaction.model import ModelError, ModelConfigurationError, SampleParams, TokenUsage
from pythia.interaction.claude_relay._cli_protocol import Assembler, decode
from pythia.interaction.claude_relay._context import Snapshot
from pythia.interaction.claude_relay._mcp import Mailbox, MCPServer
from pythia.interaction.claude_relay._runtime import Runtime
from pythia.interaction.compaction import PiCompactor, CompactionError, auto_compaction_due
from pythia.interaction.runtime_config import InteractionConfigSnapshot
from pythia.interaction.model_catalog import EndpointSpec, BUILTIN_MODEL_CATALOG
from pythia.interaction.model_config import build_parser, build_model, prepare_namespace, relay_endpoint
from pythia.interaction.save import load_interaction_save
from pythia.interaction.model_catalog_config import parse_model_catalog
from pythia.interaction.timeouts import (
    DEFAULT_CLAUDE_RELAY_GENERATION_TIMEOUT_SECONDS, DEFAULT_CLAUDE_RELAY_PARKED_TIMEOUT_SECONDS,
    DEFAULT_CLAUDE_RELAY_STARTUP_TIMEOUT_SECONDS, DEFAULT_CLAUDE_RELAY_STOP_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
)

ROOT = Path(__file__).resolve().parents[1]
RELAY = ROOT / 'claude-relay/claude_relay.py'
FIXTURE = ROOT / 'pythia_test/fixtures/claude_relay_cli.py'
TOOLS = (ToolSpec('echo', 'Return a value', {'type': 'object', 'properties': {'value': {'type': 'integer'}}}),)
DEFAULT_TOOLS = {'exec_command', 'write_stdin', 'apply_patch', 'update_plan'}


class PackageTests(unittest.TestCase):
    def test_public_exports_and_private_modules_stay_in_package(self):
        import importlib
        import pythia.interaction as public
        import pythia.interaction.claude_relay as package
        self.assertEqual(package.__all__, ['ClaudeRelayEndpoint', 'ClaudeRelayModel'])
        self.assertIs(public.ClaudeRelayEndpoint, package.ClaudeRelayEndpoint)
        self.assertIs(public.ClaudeRelayModel, package.ClaudeRelayModel)
        directory = Path(package.__file__).parent
        for name in ('_model', '_runtime', '_cli_protocol', '_mcp', '_context', '_sampling', '_diagnostics', '_trace', '_recovery'):
            module = importlib.import_module(f'pythia.interaction.claude_relay.{name}')
            self.assertEqual(Path(module.__file__).parent, directory)
        self.assertEqual(ClaudeRelayModel.__module__, 'pythia.interaction.claude_relay._model')


class HostToolPolicyTests(unittest.TestCase):
    def test_auto_main_and_worker_tools_are_not_filtered_by_backend(self):
        from pythia.interaction.auto import _environment_factory
        with tempfile.TemporaryDirectory() as root:
            for api in ('claude-relay', 'chat-completions'):
                args = SimpleNamespace(model_api=api, cwd=root, enable_workspace=True)
                for role in (1, 2):
                    with self.subTest(api=api, role=role), _environment_factory(role, args, ()) as environment:
                        self.assertEqual({tool.name for tool in environment.tool_specs}, DEFAULT_TOOLS)
                watcher = _environment_factory(-1, args, ())
                self.assertEqual(watcher.tool_specs, ())  # watcher keeps its existing role boundary

    def test_mixed_auto_roles_do_not_remove_host_tools_from_main(self):
        from pythia.interaction import auto
        from pythia.interaction._auto_config import resolve_config
        catalog = parse_model_catalog('''[catalog]
version=4
[model.fixture-http]
endpoint.api=chat-completions
endpoint.model=fixture-http
endpoint.url=http://127.0.0.1:1/v1/chat/completions
endpoint.auth=none
[model.fixture-relay]
endpoint.api=claude-relay
endpoint.model=fixture-relay
''')
        class IdleModel:
            def sample(self, *args, **kwargs):
                raise AssertionError('No model request expected')
        with tempfile.TemporaryDirectory() as root:
            for main, watcher in (('fixture-http', 'fixture-relay'), ('fixture-relay', 'fixture-http')):
                settings = resolve_config(overrides={'cwd': root}, catalog=catalog, roles=(1, -1),
                                          role_models={1: main, -1: watcher})
                path = Path(root) / main
                session = auto._Session(path, settings, model_factory=lambda *_: IdleModel())
                try:
                    session.start()
                    self.assertEqual(session.bindings[1].api, 'claude-relay' if main == 'fixture-relay' else 'chat-completions')
                    for role, expected in ((1, {*DEFAULT_TOOLS, 'yield'}), (-1, {'resume', 'read_context'})):
                        context = load_interaction_save(path / 'contexts' / f'{role}.jsonl')
                        tools = next(item for item in context if isinstance(item, Tools))
                        self.assertEqual({tool.name for tool in tools.specs}, expected)
                finally:
                    session.close()


class MailboxTests(unittest.TestCase):
    def test_result_before_callback_and_identical_calls(self):
        box = Mailbox(TOOLS, wait_seconds=2)
        box.register([('a', 'echo', {'value': 1}), ('b', 'echo', {'value': 1})])
        box.release({'a': ('one', True), 'b': ('two', False)})
        self.assertEqual(box.call('b', 'echo', {'value': 1})['content'][0]['text'], 'two')
        self.assertFalse(box.all_returned())
        self.assertFalse(box.call('a', 'echo', {'value': 1})['isError'])
        self.assertTrue(box.all_returned())
        with self.assertRaises(ModelError):
            box.call('a', 'echo', {'value': 1})

    def test_callback_before_registration_and_retirement(self):
        box = Mailbox(TOOLS, wait_seconds=2)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(box.call, 'a', 'echo', {'value': 1})
            box.register([('a', 'echo', {'value': 1})])
            self.assertFalse(future.done())
            box.release({'a': ('done', True)})
            self.assertEqual(future.result(2)['content'][0]['text'], 'done')
        box.close()
        with self.assertRaises(ModelError):
            box.call('x', 'echo', {})

    def test_missing_native_id_is_never_correlated_by_arguments(self):
        box = Mailbox(TOOLS)
        box.register([('a', 'echo', {'value': 1})])
        with self.assertRaisesRegex(ModelError, 'native tool-use ID'):
            box.call(None, 'echo', {'value': 1})


class ProtocolTests(unittest.TestCase):
    def test_rejects_invalid_json_and_partial_message(self):
        for line in (b'not JSON\n', b'{"type":"assistant","type":"result"}\n', b'{"type":"assistant","x":NaN}\n',
                     b'{"type":"assistant","x":1e999}\n'):
            with self.assertRaises(ModelError):
                decode(line)
        assembler = Assembler()
        with self.assertRaises(ModelError):
            assembler.feed({'type': 'stream_event', 'event': {'type': 'message_stop'}})

    def test_non_http_binding_rejects_api_credentials_and_urls(self):
        endpoint = BUILTIN_MODEL_CATALOG.bind('claude-relay', 'native-model').endpoint
        self.assertIsNone(endpoint.url)
        self.assertEqual(endpoint.auth, 'runtime')
        for url, auth in (("https://example.com", 'runtime'), (None, 'env:API_KEY'), (None, 'none')):
            with self.assertRaises(ValueError):
                EndpointSpec('claude-relay', url, 'native-model', auth)
        with self.assertRaises(ValueError):
            EndpointSpec('messages', 'https://example.com', 'native-model', 'runtime')

    def test_catalog_non_http_model_needs_no_fake_url(self):
        catalog = parse_model_catalog('[catalog]\nversion=4\n[model.relay-test]\nendpoint.api=claude-relay\nendpoint.model=fixture-model\n')
        binding = catalog.bind(None, 'relay-test')
        self.assertEqual(binding.api, 'claude-relay')
        self.assertIsNone(binding.endpoint.url)
        self.assertEqual(binding.endpoint.auth, 'runtime')


class RelayTimeoutConfigTests(unittest.TestCase):
    """Relay deadlines: launch option, then catalog, then default. No HTTP timeout."""

    CATALOG = ('[catalog]\nversion=4\n[model.relay-test]\nendpoint.api=claude-relay\n'
               'endpoint.model=fixture-model\ntimeouts.generation_seconds=2400\n'
               'timeouts.parked_seconds=3600\ntimeouts.startup_seconds=45\ntimeouts.stop_seconds=2.5\n')

    def args(self, model, *flags, catalog=None):
        return prepare_namespace(build_parser('test').parse_args([
            '--model', model, '--claude-relay-launcher', '/nonexistent/claude_relay.py',
            '--claude-relay-socket', '/nonexistent/broker.sock', '--claude-relay-server-uid', '1000',
            '--claude-relay-cli-version', '2.1.0-fixture', *flags]), catalog)

    def deadlines(self, endpoint):
        return (endpoint.generation_timeout_seconds, endpoint.parked_timeout_seconds,
                endpoint.startup_timeout_seconds, endpoint.stop_timeout_seconds)

    def test_catalog_deadlines_reach_the_endpoint_and_launch_options_win(self):
        catalog = parse_model_catalog(self.CATALOG)
        args = self.args('relay-test', catalog=catalog)
        self.assertEqual(self.deadlines(relay_endpoint(args, args.model_binding)), (2400, 3600, 45, 2.5))
        args = self.args('relay-test', '--claude-relay-generation-timeout', '30',
                         '--claude-relay-stop-timeout', '1', catalog=catalog)
        self.assertEqual(self.deadlines(relay_endpoint(args, args.model_binding)), (30, 3600, 45, 1))
        # Without catalog values the defaults are unchanged.
        args = self.args('fixture-model', '--endpoint-api', 'claude-relay')
        self.assertEqual(self.deadlines(relay_endpoint(args, args.model_binding)), (
            DEFAULT_CLAUDE_RELAY_GENERATION_TIMEOUT_SECONDS, DEFAULT_CLAUDE_RELAY_PARKED_TIMEOUT_SECONDS,
            DEFAULT_CLAUDE_RELAY_STARTUP_TIMEOUT_SECONDS, DEFAULT_CLAUDE_RELAY_STOP_TIMEOUT_SECONDS))
        self.assertEqual(self.deadlines(ClaudeRelayEndpoint('fixture-model', '/a', '/b', 1000, '2.1.0')),
                         self.deadlines(relay_endpoint(args, args.model_binding)))
        for value in ('0', str(MAX_TIMEOUT_SECONDS * 2)):
            with self.subTest(value=value), self.assertRaises(ModelConfigurationError):
                args = self.args('fixture-model', '--endpoint-api', 'claude-relay',
                                 '--claude-relay-parked-timeout', value)
                relay_endpoint(args, args.model_binding)

    def test_http_request_timeouts_do_not_apply(self):
        args = self.args('relay-test', '--request-timeout-seconds', '60',
                         catalog=parse_model_catalog(self.CATALOG))
        with self.assertRaisesRegex(ValueError, 'timeouts.generation_seconds'):
            build_model(args)
        model = ClaudeRelayModel(ClaudeRelayEndpoint('fixture-model', '/a', '/b', 1000, '2.1.0'))
        self.addCleanup(model.close)
        with self.assertRaisesRegex(ModelConfigurationError, 'request_timeout_seconds'):
            model.sample(InteractionContext(), sample_params=SampleParams(request_timeout_seconds=5))
        self.assertIsNone(model._runtime)


class MCPTransportTests(unittest.TestCase):
    def setUp(self):
        self.box = Mailbox(TOOLS, wait_seconds=2)
        self.server = MCPServer(self.box)
        self.addCleanup(self.server.close)

    def post(self, value, *, token=None, extra=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server.server_port, timeout=3)
        headers = {'Authorization': 'Bearer ' + (self.server.token if token is None else token),
                   'Content-Type': 'application/json', **(extra or {})}
        try:
            connection.request('POST', '/mcp', json.dumps(value).encode(), headers)
            response = connection.getresponse()
            data = response.read()
            return response.status, json.loads(data) if data else None
        finally:
            connection.close()

    def test_auth_origin_initialization_and_json_response_subset(self):
        init = {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                'params': {'protocolVersion': '2025-03-26', 'capabilities': {},
                           'clientInfo': {'name': 'test', 'version': '1'}}}
        self.assertEqual(self.post(init, token='wrong')[0], 401)
        self.assertFalse(self.box.closed)
        self.assertEqual(self.post(init, extra={'Origin': 'http://attacker.invalid'})[0], 403)
        code, result = self.post(init)
        self.assertEqual(code, 200)
        self.assertEqual(result['result']['protocolVersion'], '2025-03-26')
        headers = {'MCP-Protocol-Version': '2025-03-26'}
        self.assertEqual(self.post({'jsonrpc': '2.0', 'method': 'notifications/initialized'}, extra=headers)[0], 202)
        _, result = self.post({'jsonrpc': '2.0', 'id': 'catalog', 'method': 'tools/list'}, extra=headers)
        self.assertEqual(result['result']['tools'][0]['name'], 'echo')

    def test_native_discovery_refusal_allows_standard_initialize_fallback(self):
        code, result = self.post({'jsonrpc': '2.0', 'id': 0, 'method': 'server/discover',
                                  'params': {'_meta': {'io.modelcontextprotocol/protocolVersion': '2026-07-28'}}},
                                 extra={'MCP-Protocol-Version': '2026-07-28'})
        self.assertEqual(code, 200)
        self.assertEqual(result['error']['code'], -32601)
        self.assertFalse(self.box.closed)
        self.assertIsNone(self.server.protocol)
        self.test_auth_origin_initialization_and_json_response_subset()

    def test_discovery_does_not_authorize_uninitialized_calls_or_new_protocol(self):
        _, result = self.post({'jsonrpc': '2.0', 'id': 0, 'method': 'server/discover'})
        self.assertEqual(result['error']['code'], -32601)
        _, result = self.post({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'},
                              extra={'MCP-Protocol-Version': '2026-07-28'})
        self.assertIn('error', result)
        self.assertTrue(self.box.closed)

    def test_close_unblocks_incomplete_http_request(self):
        with socket.create_connection(('127.0.0.1', self.server.server.server_port), timeout=2) as peer:
            peer.sendall(b'POST /mcp HTTP/1.1\r\n')
            deadline = time.monotonic() + 2
            while not self.server.connections and time.monotonic() < deadline:
                time.sleep(.01)
            self.server.close()
            self.assertFalse(self.server.workers)


@unittest.skipIf(os.getuid() == 0, 'non-root fixture process required')
class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.socket = self.root / 'control/broker.sock'
        self.broker = subprocess.Popen([str(RELAY), '--relay-serve', '--relay-socket', str(self.socket),
                                       '--relay-allow-uid', str(os.getuid()), '--relay-wrapper', str(FIXTURE),
                                       '--relay-stop-grace', '0.2'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(self.stop_broker)
        deadline = time.monotonic() + 5
        while not self.socket.exists() and self.broker.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(self.socket.exists())
        self.endpoint = ClaudeRelayEndpoint('fixture-model', str(RELAY), str(self.socket), os.getuid(), '2.1.0-fixture',
                                            generation_timeout_seconds=5, stop_timeout_seconds=2)
        self.model = ClaudeRelayModel(self.endpoint)
        self.addCleanup(self.model.close)

    def stop_broker(self):
        if self.broker.poll() is None:
            self.broker.terminate()
        self.broker.communicate(timeout=5)

    def test_complete_two_call_handoff_and_per_message_usage(self):
        context = InteractionContext((Instructions(''), Message('user', 'two')))
        sample = self.model.sample(context, tools=TOOLS)
        self.assertEqual(len(sample.tool_calls), 2)
        self.assertEqual(sample.usage.total_tokens, 15)
        self.assertNotEqual(sample.tool_calls[0].call_id, sample.tool_calls[1].call_id)
        context.extend(sample.context_items())
        context.append(SampleMetadata(TokenUsage()))  # audit-only edits do not invalidate a handoff
        time.sleep(.03)
        context.extend([ToolResult(sample.tool_calls[0].call_id, 'one'), ToolResult(sample.tool_calls[1].call_id, 'two', False)])
        final = self.model.sample(context, tools=TOOLS)
        self.assertEqual(final.last_assistant_text, 'one|two')
        self.assertEqual(final.usage.total_tokens, 9)
        self.assertIsNone(self.model._runtime)

    def test_missing_result_is_an_error_and_context_rewrite_retires(self):
        context = InteractionContext((Message('user', 'two'),))
        sample = self.model.sample(context, tools=TOOLS)
        context.extend(sample.context_items())
        with self.assertRaisesRegex(ModelError, 'complete outstanding'):
            self.model.sample(context, tools=TOOLS)
        self.assertIsNone(self.model._runtime)
        self.model.sample(InteractionContext((Message('user', 'two'),)), tools=TOOLS)
        result = self.model.sample(InteractionContext((Message('user', 'summary'),)), tools=())
        self.assertEqual(result.last_assistant_text, 'cold:summary')

    def test_late_failure_inventory_and_no_automatic_native_background_results(self):
        for text, tools in (('late-error', ()), ('inventory-bad', ()), ('autonomous', TOOLS), ('missing-meta', TOOLS)):
            with self.subTest(text=text):
                context = InteractionContext((Message('user', text),))
                try:
                    sample = self.model.sample(context, tools=tools)
                    context.extend(sample.context_items())
                    if sample.tool_calls:
                        time.sleep(.1)
                        context.extend([ToolResult(c.call_id, 'real result') for c in sample.tool_calls])
                        with self.assertRaises(ModelError):
                            self.model.sample(context, tools=tools)
                    else:
                        self.fail('invalid run returned success')
                except ModelError:
                    pass

    def test_retire_wakes_parked_sample_without_waiting_for_sample_lock(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.model.sample, InteractionContext((Message('user', 'park'),)))
            deadline = time.monotonic() + 5
            while self.model._runtime is None and time.monotonic() < deadline:
                time.sleep(.01)
            time.sleep(.15)
            self.model.retire()
            with self.assertRaises(ModelError):
                future.result(4)

    def test_native_compaction_and_hook_records_still_retire_the_run(self):
        for text in ('system-compacting', 'system-compact-boundary', 'system-hook'):
            with self.subTest(text=text), self.assertRaisesRegex(ModelError, 'Unexpected system event'):
                self.model.sample(InteractionContext((Message('user', text),)))
            self.assertIsNone(self.model._runtime)

    def test_retire_during_validation_prevents_a_later_launch(self):
        entered, release = threading.Event(), threading.Event()
        original = Snapshot.build
        def blocked(*args, **kwargs):
            entered.set()
            release.wait(3)
            return original(*args, **kwargs)
        with mock.patch.object(Snapshot, 'build', side_effect=blocked), \
                mock.patch('pythia.interaction.claude_relay._model.Runtime') as runtime, \
                concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.model.sample, InteractionContext((Message('user', 'text-only'),)))
            self.assertTrue(entered.wait(2))
            self.model.retire()
            release.set()
            with self.assertRaises(ModelError):
                future.result(3)
            runtime.assert_not_called()

    def test_nondefault_sampling_rejected_before_launch(self):
        with self.assertRaises(ModelConfigurationError):
            self.model.sample(InteractionContext(), sample_params=SampleParams(temperature=.5))
        self.assertIsNone(self.model._runtime)

    def bind_effort(self, effort):
        binding = self.endpoint.binding.with_extra_sample_params({'output_config': {'effort': effort}})
        self.model.close()
        self.model = ClaudeRelayModel(replace(self.endpoint, binding=binding))
        self.addCleanup(self.model.close)

    def fixture_state(self, runtime=None):
        data = self.model.stderr_tail if runtime is None else bytes(runtime.stderr_tail)
        return [json.loads(line.removeprefix('fixture-state:'))
                for line in data.decode().splitlines() if line.startswith('fixture-state:')][-1]

    def test_effort_precedence_and_no_implicit_flag(self):
        self.bind_effort('max')
        for extra, expected in ((None, 'max'), ({'output_config': {'effort': 'high'}}, 'high'),
                                ({}, None), ({'output_config': {}}, None)):
            with self.subTest(extra=extra):
                sample = self.model.sample(InteractionContext((Message('user', 'text-only'),)),
                                           sample_params=SampleParams(extra=extra))
                self.assertEqual(sample.stop_reason, 'end_turn')
                self.assertEqual(self.fixture_state()['effort'], expected)

    def test_equivalent_effort_reuses_live_handoff(self):
        self.bind_effort('max')
        context = InteractionContext((Message('user', 'two'),))
        sample = self.model.sample(context, tools=TOOLS)
        old = self.model._runtime
        context.extend(sample.context_items())
        context.extend([ToolResult(call.call_id, f'result-{i}') for i, call in enumerate(sample.tool_calls)])
        final = self.model.sample(context, tools=TOOLS,
                                  sample_params=SampleParams(extra={'output_config': {'effort': 'max'}}))
        self.assertEqual(final.last_assistant_text, 'result-0|result-1')
        self.assertTrue(old.mailbox.all_returned())
        self.assertEqual(self.fixture_state()['pid'], self.fixture_state(old)['pid'])

    def test_effort_change_retires_before_releasing_results_then_cold_imports(self):
        self.bind_effort('max')
        context = InteractionContext((Message('user', 'two'),))
        sample = self.model.sample(context, tools=TOOLS)
        old = self.model._runtime
        context.extend(sample.context_items())
        context.extend([ToolResult(call.call_id, 'already executed') for call in sample.tool_calls])
        final = self.model.sample(context, tools=TOOLS,
                                  sample_params=SampleParams(extra={'output_config': {'effort': 'low'}}))
        self.assertTrue(old.closed.is_set())
        self.assertTrue(all(slot['result'] is None for slot in old.mailbox.slots.values()))
        self.assertFalse(final.tool_calls)
        state = self.fixture_state()
        self.assertNotEqual(state['pid'], self.fixture_state(old)['pid'])
        self.assertEqual(state['effort'], 'low')
        self.assertEqual(state['history_result_ids'], [call.call_id for call in sample.tool_calls])

    def test_native_incomplete_text_is_not_reported_as_end_turn(self):
        for reason in ('max_tokens', 'length', 'refusal', 'pause_turn', 'model_context_window_exceeded'):
            with self.subTest(reason=reason):
                sample = self.model.sample(InteractionContext((Message('user', 'limited:' + reason),)))
                self.assertEqual(sample.stop_reason, 'max_tokens' if reason == 'length' else reason)
                self.assertIsNone(self.model._runtime)
        with self.assertRaises(ModelError):
            self.model.sample(InteractionContext((Message('user', 'limited-tools'),)), tools=TOOLS)
        with self.assertRaises(ModelError):
            self.model.sample(InteractionContext((Message('user', 'conflicting-final-stop'),)))

    def test_pi_summary_retires_parked_cli_and_next_sample_uses_compacted_context(self):
        self.bind_effort('max')
        context = InteractionContext((Instructions('Be brief.'), Message('user', 'older context ' * 100),
                                      Message('assistant', 'old answer'), ModelSampleBoundary(), Message('user', 'two')))
        sample = self.model.sample(context, tools=TOOLS)
        old = self.model._runtime
        context.extend(sample.context_items())
        context.extend([ToolResult(call.call_id, 'already executed') for call in sample.tool_calls])
        self.assertTrue(auto_compaction_due(self.model, context, InteractionConfigSnapshot(auto_compact_tokens=1)))
        before = context.items
        summaries = []
        def construct(*args, **kwargs):
            value = Runtime(*args, **kwargs)
            summaries.append(value)
            return value
        with mock.patch('pythia.interaction.claude_relay._model.Runtime', side_effect=construct):
            compacted = PiCompactor(self.model, keep_recent_tokens=1).compact(context, tools=TOOLS)
        self.assertEqual(context.items, before)  # caller, not adapter, installs the checkpoint
        self.assertTrue(old.closed.is_set())
        self.assertTrue(all(slot['result'] is None for slot in old.mailbox.slots.values()))
        self.assertTrue(summaries)
        for runtime in summaries:
            state = self.fixture_state(runtime)
            self.assertTrue(state['summary'])
            self.assertEqual(state['effort'], 'max')
            self.assertEqual(state['tool_server_count'], 0)
            self.assertEqual(state['disable_auto_compact'], '1')
            self.assertTrue(runtime.closed.is_set())
        context.extend(compacted.context_items())
        self.assertTrue(any(isinstance(item, ContextPrefix) for item in context.items))
        self.assertEqual([item.call_id for item in context.model_items() if isinstance(item, ToolResult)],
                         [call.call_id for call in sample.tool_calls])
        final = self.model.sample(context, tools=TOOLS)
        self.assertFalse(final.tool_calls)
        self.assertFalse(self.fixture_state()['summary'])
        self.assertEqual(self.fixture_state()['effort'], 'max')
        self.assertNotIn(self.fixture_state()['pid'], [self.fixture_state(value)['pid'] for value in summaries])

    def test_pi_never_installs_a_limited_native_summary(self):
        context = InteractionContext((Message('user', 'limit-summary ' * 100), Message('assistant', 'old answer'),
                                      ModelSampleBoundary(), Message('user', 'recent request')))
        before = context.items
        with self.assertRaisesRegex(CompactionError, 'stop_reason max_tokens'):
            PiCompactor(self.model, keep_recent_tokens=0).compact(context)
        self.assertEqual(context.items, before)
        self.assertIsNone(self.model._runtime)

    def test_synthetic_user_message_after_results_forces_cold_import(self):
        context = InteractionContext((Message('user', 'two'),))
        sample = self.model.sample(context, tools=TOOLS)
        context.extend(sample.context_items())
        context.extend([*(ToolResult(call.call_id, 'done') for call in sample.tool_calls), Message('user', 'new context')])
        result = self.model.sample(context, tools=TOOLS)
        self.assertEqual(result.last_assistant_text, 'cold:new context')

    def launch_options(self):
        return ['--endpoint-api', 'claude-relay', '--model', 'fixture-model',
                '--claude-relay-launcher', str(RELAY), '--claude-relay-socket', str(self.socket),
                '--claude-relay-server-uid', str(os.getuid()), '--claude-relay-cli-version', '2.1.0-fixture',
                '--no-user-model-catalog', '--claude-relay-generation-timeout', '5']

    def test_factory_and_catalog_configuration(self):
        args = build_parser('test').parse_args(self.launch_options())
        model = build_model(args)
        self.addCleanup(model.close)
        self.assertIsInstance(model, ClaudeRelayModel)
        self.assertIsNone(prepare_namespace(args).model_binding.endpoint.url)
        for flag, value in (('--endpoint-url', 'http://localhost:9999'), ('--endpoint-api-key', 'not-secret'),
                            ('--compaction-mode', 'provider'), ('--max-output-tokens', '10')):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                build_model(build_parser('test').parse_args(self.launch_options() + [flag, value]))

    def test_headless_cli_and_auto_through_real_relay_with_synthetic_native(self):
        path = self.root / 'conversation.jsonl'
        env = {**os.environ, 'HOME': str(self.root)}
        result = subprocess.run(['/usr/bin/python3', '-m', 'pythia.interaction.cli', *self.launch_options(),
                                 '--headless', '--prompt', 'text-only', '--enable-default-tools', 'False',
                                 '--save', str(path), '--resume', 'False', '--cwd', str(self.root)],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        context = load_interaction_save(path)
        self.assertIn('cold:text-only', [item.content for item in context.items if isinstance(item, Message)])
        self.assertFalse(any(item.specs for item in context if isinstance(item, Tools)))
        self.assertNotIn('exec_command runs without a sandbox', result.stderr)
        path = self.root / 'auto'
        result = subprocess.run(['/usr/bin/python3', '-m', 'pythia.interaction.auto', *self.launch_options(),
                                 '--headless', '--prompt', 'text-only', '--watcher-observe-only',
                                 '--save', str(path), '--cwd', str(self.root)],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('claude_relay_launcher', (path / 'config.json').read_text())
        context = load_interaction_save(path / 'contexts/1.jsonl')
        self.assertIn('cold:text-only', [item.content for item in context.items if isinstance(item, Message)])
        catalog = next(item for item in context.items if isinstance(item, Tools))
        self.assertEqual({tool.name for tool in catalog.specs}, DEFAULT_TOOLS)

    def test_cli_and_auto_execute_host_command_sessions_through_mcp(self):
        for frontend in ('cli', 'auto'):
            path = self.root / ('commands.jsonl' if frontend == 'cli' else 'auto-commands')
            options = ['--resume', 'False'] if frontend == 'cli' else ['--watcher-observe-only']
            result = subprocess.run([
                '/usr/bin/python3', '-I', '-S', '-c',
                'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); '
                'runpy.run_module(sys.argv.pop(1), run_name="__main__", alter_sys=True)',
                str(ROOT), f'pythia.interaction.{frontend}', *self.launch_options(),
                '--headless', '--prompt', 'host-command-tools', '--cwd', str(self.root), '--save', str(path), *options,
            ], cwd=ROOT, env={**os.environ, 'HOME': str(self.root)}, capture_output=True, text=True, timeout=20)
            with self.subTest(frontend=frontend):
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                context = load_interaction_save(path if frontend == 'cli' else path / 'contexts/1.jsonl')
                tools = next(item for item in context if isinstance(item, Tools))
                self.assertEqual({tool.name for tool in tools.specs}, DEFAULT_TOOLS)
                self.assertEqual([item.name for item in context if isinstance(item, ToolCall)], ['exec_command', 'write_stdin'])
                results = [item for item in context if isinstance(item, ToolResult)]
                self.assertEqual(len(results), 2)
                self.assertTrue(all(item.success for item in results))
                output = '\n'.join(item.output for item in results)
                self.assertIn(json.dumps({'host_uid': os.getuid(), 'host_cwd': str(self.root)}), output)
                self.assertIn('relay-host-session-ok', output)
                if frontend == 'cli':
                    self.assertIn('exec_command runs without a sandbox', result.stderr)
                    self.assertIn('host tools run as the Pythia user, outside the Claude sandbox', result.stderr)

    def test_headless_interrupt_retires_a_live_model_before_generation_deadline(self):
        save = self.root / 'interrupted.jsonl'
        with subprocess.Popen(['/usr/bin/python3', '-m', 'pythia.interaction.cli', *self.launch_options(),
                               '--claude-relay-generation-timeout', '60', '--headless', '--prompt', 'park',
                               '--enable-default-tools', 'False', '--save', str(save), '--resume', 'False'],
                              cwd=ROOT, env={**os.environ, 'HOME': str(self.root)},
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            try:
                deadline = time.monotonic() + 5
                while not save.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(save.exists())
                time.sleep(.4)
                process.send_signal(signal.SIGINT)
                _, errors = process.communicate(timeout=7)
                self.assertEqual(process.returncode, 130, errors)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)


if __name__ == '__main__':
    unittest.main()
