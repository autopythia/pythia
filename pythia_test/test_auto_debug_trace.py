"""Auto trace lifecycle and HTTP/native routing, with local fixtures only."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from pythia.interaction import auto
from pythia.interaction.loop import kernel
from pythia.interaction._auto_config import build_parser, resolve_config
from pythia.interaction._debug_trace import DebugTrace
from pythia.interaction._debug_trace import capture_trace_scope
from pythia.interaction.items import Message, ModelSampleBoundary
from pythia.interaction.context import InteractionContext
from pythia.interaction.compaction import PiCompactor
from pythia.interaction.environment import Environment
from pythia.interaction.runtime_config import InteractionConfigSnapshot
from pythia.interaction.model import ModelSample
from pythia_test import test_interaction_claude_relay as relay_fixture
from pythia_test import test_debug_trace as http_fixture


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class AutoTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = resolve_config(overrides={'cwd': str(self.root)})
        self.stderr = io.StringIO()

    def session(self, *, trace=False, resume=False):
        class Model:
            def sample(self, *args, **kwargs):
                return ModelSample((Message('assistant', 'done'),))
        session = auto._Session(self.root / 'run', self.settings, watcher_observe_only=True,
                                debug_trace=trace, resume=resume, model_factory=lambda *_: Model())
        self.addCleanup(session.close)
        with redirect_stderr(self.stderr):
            session.start()
        return session

    def finish(self, session):
        task = session.submit('hello')
        deadline = time.monotonic() + 5
        while session.task_result(task) is None and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(session.task_result(task))
        session.close()

    def test_off_by_default_and_not_saved_or_restored(self):
        self.assertFalse(build_parser().parse_args([]).debug_trace)
        self.finish(self.session())
        self.assertFalse(list((self.root / 'run').rglob('*.trace.*')))
        self.finish(self.session(trace=True, resume=True))
        prefix = self.root / 'run/contexts'
        self.assertTrue((prefix / '1.trace.events.jsonl').exists())
        self.assertFalse(list(prefix.glob('-1.trace.*')))
        self.assertNotIn('debug_trace', (self.root / 'run/config.json').read_text())
        before = (prefix / '1.trace.events.jsonl').read_bytes()
        self.finish(self.session(resume=True))
        self.assertEqual((prefix / '1.trace.events.jsonl').read_bytes(), before)
        self.finish(self.session(trace=True, resume=True))
        events = read(prefix / '1.trace.events.jsonl')
        self.assertEqual(len({r['run_id'] for r in events}), 2)
        self.assertTrue(all(r['context_id'] == 1 and r['op'] == 'sample' for r in events))
        self.assertIn('sensitive', self.stderr.getvalue())

    def test_preflight_failure_and_lock_precede_model_or_trace_construction(self):
        self.finish(self.session())
        bad = self.root / 'run/contexts/1.trace.events.jsonl'
        bad.mkdir()
        factory = mock.Mock()
        session = auto._Session(self.root / 'run', self.settings, watcher_observe_only=True,
                                resume=True, debug_trace=True, model_factory=factory)
        with self.assertRaises(ValueError):
            session.start()
        factory.assert_not_called()
        bad.rmdir()
        active = self.session(resume=True)
        other = auto._Session(self.root / 'run', self.settings, resume=True, debug_trace=True)
        with mock.patch.object(DebugTrace, 'open') as opening, self.assertRaises(Exception):
            other.start()
        opening.assert_not_called()
        active.close()

    def test_print_config_creates_no_trace_and_warning_is_not_task_failure(self):
        with mock.patch.object(DebugTrace, 'open') as opening, mock.patch('sys.stdout', io.StringIO()):
            self.assertEqual(auto.main(['--no-user-model-catalog', '--print-config', '--debug-trace',
                                        '--save', str(self.root / 'not-created')]), 0)
        opening.assert_not_called()
        session = self.session(trace=True)
        session._traces[1]._fail('Warning: debug trace disabled; injected failure.')
        with redirect_stderr(self.stderr):
            self.finish(session)
        self.assertFalse(session.has_errors)
        self.assertEqual(self.stderr.getvalue().count('injected failure'), 1)

    def test_auto_pi_scope_and_interactive_warning_delivery(self):
        session = self.session(trace=True)
        scopes = []
        class Summary:
            def sample(self, context, **kwargs):
                scopes.append(capture_trace_scope())
                return ModelSample((Message('assistant', 'summary'),))
        context = InteractionContext((Message('user', 'older material ' * 100), Message('assistant', 'old'),
                                      ModelSampleBoundary(), Message('user', 'recent')))
        with mock.patch.object(kernel, 'create_default_compactor', return_value=PiCompactor(Summary(), keep_recent_tokens=0)):
            self.assertTrue(kernel.compact(context, Summary(), Environment(), InteractionConfigSnapshot(),
                                                auto._AutoHost(session, 1), None))
        self.assertTrue(scopes)
        self.assertTrue(all(scope['op'] == 'compact' for scope in scopes))
        session._trace_to_events = True
        session._traces[1]._fail('Warning: debug trace disabled; interactive notice.')
        session._trace_warning(1)
        self.assertTrue(any('interactive notice' in item.text for event in session.drain_events() for item in event.items))
        self.assertNotIn('interactive notice', self.stderr.getvalue())
        self.assertFalse(session.has_errors)
        session.close()


@unittest.skipIf(os.getuid() == 0, 'non-root fixture broker required')
class MixedTraceTests(unittest.TestCase):
    setUp = relay_fixture.ModelTests.setUp
    stop_broker = relay_fixture.ModelTests.stop_broker

    def test_relay_main_and_http_watcher_have_separate_trace_files(self):
        server = http_fixture._Server(self, (200, [('Content-Type', 'application/json')],
                                              json.dumps(http_fixture.CHAT_OK).encode()))
        catalog = self.root / 'catalog.ini'
        catalog.write_text(f'''[catalog]
version=4
[model.main-relay]
endpoint.api=claude-relay
endpoint.model=fixture-model
[model.watch-http]
endpoint.api=chat-completions
endpoint.model=watch-http
endpoint.url={server.url}/v1/chat/completions
endpoint.auth=none
''')
        save = self.root / 'mixed'
        with redirect_stderr(io.StringIO()) as stderr:
            result = auto.main([
                '--headless', '--prompt', 'text-only', '--save', str(save), '--cwd', str(self.root),
                '--model-catalog', str(catalog), '--main-model', 'main-relay', '--watcher-model', 'watch-http',
                '--claude-relay-launcher', str(relay_fixture.RELAY), '--claude-relay-socket', str(self.socket),
                '--claude-relay-server-uid', str(os.getuid()), '--claude-relay-cli-version', '2.1.0-fixture',
                '--debug-trace'])
        self.assertEqual(result, 0, stderr.getvalue())
        files = save / 'contexts'
        native = read(files / '1.trace.events.jsonl')
        requests = read(files / '-1.trace.req.jsonl')
        responses = read(files / '-1.trace.res.jsonl')
        self.assertTrue(any(r['type'] == 'claude_stdout' for r in native))
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]['id'], responses[0]['id'])
        self.assertEqual(requests[0]['context_id'], -1)
        self.assertEqual(requests[0]['role'], 'watcher')
        self.assertEqual(requests[0]['op'], 'sample')
        self.assertEqual(requests[0]['run_id'], native[0]['run_id'])
        self.assertEqual((files / '1.trace.req.jsonl').read_text(), '')
        self.assertNotIn('claude_stdout', requests[0]['payload'])
        self.assertNotIn('debug_trace', (save / 'config.json').read_text())
