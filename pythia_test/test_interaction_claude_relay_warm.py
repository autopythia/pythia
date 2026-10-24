"""Warm delivery on the Claude relay (doc/claude-relay-warm-delivery-plan.md),
and warm resumes after a yield (doc/claude-relay-warm-yield-plan.md).

A real Python relay with the synthetic native CLI, never Claude. The fixture
serves turn after turn while its stdin stays open, as CLI 2.1.289 does: a
user record with a uuid is reported read (queued) and taken up (started), its
turn's first reply echoes the uuid, and a record read while tool calls wait
goes to the model with their results.
"""
from dataclasses import replace
import json
import os
import re
import subprocess
import time
import unittest

from pythia.interaction._debug_trace import DebugTrace
from pythia.interaction._cli_save import full_log_path
from pythia.interaction.claude_relay import ClaudeRelayEndpoint, ClaudeRelayModel
from pythia.interaction.claude_relay._cli_protocol import PROFILE
from pythia.interaction.claude_relay._context import Delivery, Snapshot, handoff
from pythia.interaction.compaction import CompactionSettings, run_compaction
from pythia.interaction.context import InteractionContext
from pythia.interaction.default_environment import DefaultEnvironment
from pythia.interaction.items import ContextPrefix, Instructions, Message, ModelFailure, ToolCall
from pythia.interaction.items import ToolResult, TurnSummary, UserInteractionBoundary
from pythia.interaction.loop import TurnHost, run_turn
from pythia.interaction.model import ModelConfigurationError, ModelResponseError
from pythia.interaction.model_catalog_config import parse_model_catalog
from pythia.interaction.model_config import relay_endpoint
from pythia.interaction.runtime_config import InteractionConfigSnapshot
from pythia.interaction.save import load_interaction_save
from pythia.interaction.timeouts import DEFAULT_CLAUDE_RELAY_IDLE_TIMEOUT_SECONDS
from pythia.interaction.timeouts import DEFAULT_CLAUDE_RELAY_SHORT_CACHE_IDLE_TIMEOUT_SECONDS
from pythia.interaction.timeouts import MAX_TIMEOUT_SECONDS
from pythia_test import test_interaction_claude_relay as fixtures

TOOLS = fixtures.TOOLS
UUID = r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'


class MemoryTrace(DebugTrace):
    """The relay's trace events, kept in memory as they happen; no payloads."""

    def __init__(self, root):
        super().__init__(root / 'trace.req.jsonl', root / 'trace.res.jsonl',
                         event_path=root / 'trace.events.jsonl')
        self.rows = []

    def event(self, kind, *, scope=None, data=None, **fields):
        self.rows.append({'type': kind, **fields})

    @property
    def enabled(self):
        return False  # so RelayTrace records no stdin/stdout payloads

    def of(self, kind):
        return [row for row in list(self.rows) if row['type'] == kind]


class Host(TurnHost):
    def append(self, context, items):
        context.extend(items)


class HandoffTests(unittest.TestCase):
    """What a sample sends the live process instead of a cold import."""

    CALL = ToolCall('echo', 'g:toolu_a', '{"value": 1}')
    PENDING = {'g:toolu_a': 'toolu_a'}

    def handoff(self, *items, pending=PENDING, base=(Message('user', 'go'), CALL)):
        acknowledged = Snapshot.build(InteractionContext(base), ()).records
        return handoff(Snapshot.build(InteractionContext((*base, *items)), ()), acknowledged, pending)

    def test_results_then_user_messages_go_to_the_live_process(self):
        result = ToolResult('g:toolu_a', 'out')
        self.assertEqual(self.handoff(result), Delivery({'toolu_a': ('out', True)}, ()))
        self.assertEqual(self.handoff(result, Message('user', 'steer'), Message('user', 'more')),
                         Delivery({'toolu_a': ('out', True)}, ('steer', 'more')))
        # After a final answer, nothing is pending: the user's messages start a turn.
        answered = (Message('user', 'go'), Message('assistant', 'done'))
        self.assertEqual(self.handoff(Message('user', 'next'), pending={}, base=answered),
                         Delivery({}, ('next',)))

    def test_anything_else_starts_a_new_process(self):
        # (A message before the results, or with results missing, is no valid
        # context: InteractionContext refuses it.)
        result = ToolResult('g:toolu_a', 'out')
        answered = (Message('user', 'go'), Message('assistant', 'done'))
        for items, pending, base in (
                ((), {}, answered),  # nothing new, as /continue after a final answer
                ((result, Message('assistant', 'injected')), self.PENDING, None),
                ((result, Message('user', ' ')), self.PENDING, None),  # blank text
                ((Message('assistant', 'more'), Message('user', 'next')), {}, answered)):
            with self.subTest(items=items):
                self.assertIsNone(self.handoff(*items, pending=pending, base=base or (Message('user', 'go'), self.CALL)))
        # A changed history, such as a checkpoint in place of the old turns.
        self.assertIsNone(handoff(Snapshot.build(InteractionContext((Message('user', 'summary'),)), ()),
                                  Snapshot.build(InteractionContext(answered), ()).records, {}))

    def test_missing_results_alone_are_an_error_as_before(self):
        with self.assertRaisesRegex(ModelResponseError, 'complete outstanding'):
            self.handoff()


class IdleSettingTests(unittest.TestCase):
    """The idle deadline: launch option, then catalog, then default; 0 keeps no process."""

    CATALOG = fixtures.RelayTimeoutConfigTests.CATALOG
    args = fixtures.RelayTimeoutConfigTests.args

    def endpoint(self, *flags, model='fixture-model', catalog=None):
        args = self.args(model, *(('--endpoint-api', 'claude-relay') if catalog is None else ()), *flags,
                         catalog=catalog)
        return relay_endpoint(args, args.model_binding)

    def test_the_deadline_comes_from_the_launch_option_the_catalog_or_the_default(self):
        self.assertEqual(DEFAULT_CLAUDE_RELAY_IDLE_TIMEOUT_SECONDS, 3595)  # an hour, less 5 s
        self.assertEqual(DEFAULT_CLAUDE_RELAY_SHORT_CACHE_IDLE_TIMEOUT_SECONDS, 305)  # 5 minutes, plus 5 s
        endpoint = self.endpoint()
        self.assertEqual((endpoint.idle_timeout_seconds, endpoint.short_cache_idle_timeout_seconds), (3595, 305))
        catalog = parse_model_catalog(self.CATALOG + 'timeouts.idle_seconds=600\n')
        self.assertEqual(self.endpoint(model='relay-test', catalog=catalog).idle_timeout_seconds, 600)
        self.assertEqual(self.endpoint('--claude-relay-idle-timeout', '120', model='relay-test',
                                       catalog=catalog).idle_timeout_seconds, 120)
        # 0 ends the process at each final answer, as before.
        self.assertEqual(self.endpoint('--claude-relay-idle-timeout', '0').idle_timeout_seconds, 0)
        catalog = parse_model_catalog(self.CATALOG + 'timeouts.idle_seconds=0\n')
        self.assertEqual(self.endpoint(model='relay-test', catalog=catalog).idle_timeout_seconds, 0)

    def test_negative_or_unbounded_deadlines_fail(self):
        for value in ('-1', str(MAX_TIMEOUT_SECONDS * 2), 'nan'):
            with self.subTest(value=value), self.assertRaises((ModelConfigurationError, ValueError)):
                self.endpoint('--claude-relay-idle-timeout', value)
        with self.assertRaises(ValueError):  # catalog entry errors name no field
            parse_model_catalog(self.CATALOG + 'timeouts.idle_seconds=-5\n')
        endpoint = ClaudeRelayEndpoint('fixture-model', '/a', '/b', 1000, '2.1.0')
        for value in (0, -1, True, MAX_TIMEOUT_SECONDS * 2):
            with self.subTest(value=value), self.assertRaises(ModelConfigurationError):
                replace(endpoint, short_cache_idle_timeout_seconds=value)
        with self.assertRaises(ModelConfigurationError):
            replace(endpoint, idle_timeout_seconds=-1)


@unittest.skipIf(os.getuid() == 0, 'non-root relay fixture required')
class WarmTests(unittest.TestCase):
    setUp = fixtures.ModelTests.setUp
    stop_broker = fixtures.ModelTests.stop_broker
    fixture_state = fixtures.ModelTests.fixture_state

    def relaunch(self, **settings):
        """A traced model on this broker, with some endpoint settings changed."""
        self.model.close()
        self.trace = MemoryTrace(self.root)
        self.model = ClaudeRelayModel(replace(self.endpoint, **settings), trace=self.trace)
        self.addCleanup(self.model.close)

    def turns(self, runtime):
        """The fixture's account of each turn delivered to ``runtime``'s process."""
        return [json.loads(line.removeprefix('fixture-turn:'))
                for line in bytes(runtime.stderr_tail).decode().splitlines() if line.startswith('fixture-turn:')]

    def ask(self, context, text, **options):
        """The user's next message, and the model's answer, both added to the context."""
        context.append(Message('user', text))
        sample = self.model.sample(context, **options)
        context.extend(sample.context_items())
        return sample

    def answered(self, text='text-only', **options):
        """A context whose first turn has ended with a final answer, kept warm."""
        context = InteractionContext((Instructions(''),))
        self.ask(context, text, **options)
        return context

    def until(self, predicate, seconds=5):
        deadline = time.monotonic() + seconds
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.02)
        return predicate()

    def abandoned(self):
        return [row['reason'] for row in self.trace.of('continuation_abandoned')]


class WarmTurnTests(WarmTests):
    """Part 1: the process stays after a final answer, for the next turn."""

    def test_a_later_turn_runs_in_the_same_process(self):
        self.relaunch()
        context = InteractionContext((Instructions(''), Message('user', 'two')))
        sample = self.model.sample(context, tools=TOOLS)
        context.extend(sample.context_items())
        context.extend(ToolResult(call.call_id, f'r{i}') for i, call in enumerate(sample.tool_calls))
        final = self.model.sample(context, tools=TOOLS)
        self.assertEqual(final.last_assistant_text, 'r0|r1')
        context.extend(final.context_items())
        runtime = self.model._runtime
        self.assertIsNotNone(runtime)
        self.assertTrue(runtime.idle)
        state = self.fixture_state(runtime)
        self.assertRegex(state['import_uuid'], UUID)
        self.assertEqual(state['import_keys'], ['message', 'type', 'uuid'])
        self.assertEqual(state['cross_session_inbound'], 'refuse')
        self.assertEqual(self.ask(context, 'next question', tools=TOOLS).last_assistant_text, 'warm:next question')
        # Two messages since the last answer (a steer flushed after it, say) go as one record.
        context.append(Message('user', 'first'))
        self.assertEqual(self.ask(context, 'second', tools=TOOLS).last_assistant_text, 'warm:first|second')
        self.assertIs(self.model._runtime, runtime)
        turns = self.turns(runtime)
        self.assertEqual([turn['texts'] for turn in turns], [['next question'], ['first', 'second']])
        for turn in turns:
            self.assertEqual(turn['pid'], state['pid'])
            self.assertRegex(turn['uuid'], UUID)
            self.assertIs(turn['client_composed'], True)
            self.assertIsNone(turn['parent_tool_use_id'])
        self.assertEqual(len(self.trace.of('claude_invocation')), 2)  # the version query, then the CLI
        deliveries = self.trace.of('continuation_delivery')
        self.assertEqual([(row['message_count'], row['mid_turn']) for row in deliveries], [(1, False), (2, False)])
        self.assertEqual([row['uuid'] for row in deliveries], [turn['uuid'] for turn in turns])
        self.assertEqual(len(self.trace.of('continuation_kept')), 3)
        self.assertEqual(self.trace.of('claude_invocation')[-1]['profile'], PROFILE)
        self.assertEqual(PROFILE, 'claude-stream-json-v10')

    def test_a_changed_history_or_nothing_new_starts_a_new_process(self):
        context = self.answered()
        old = self.model._runtime
        # Nothing new since the answer, as /continue after a final answer.
        self.assertEqual(self.model.sample(context).last_assistant_text, 'cold:text-only')
        self.assertTrue(old.closed.is_set())
        # The answer replaced, as /retry does, or a checkpoint in place of the history.
        for items in ((Message('user', 'text-only'), Message('assistant', 'edited'), Message('user', 'after edit')),
                      (ContextPrefix((Message('user', 'summary'),)), Message('user', 'after edit'))):
            with self.subTest(items=items):
                old = self.model._runtime
                self.assertEqual(self.model.sample(InteractionContext(items)).last_assistant_text, 'cold:after edit')
                self.assertTrue(old.closed.is_set())

    def test_the_turn_loop_keeps_the_process_between_turns(self):
        config = InteractionConfigSnapshot(enable_auto_compaction=False)
        context = InteractionContext((Instructions(''), Message('user', 'text-only')))
        with DefaultEnvironment(cwd=self.root) as environment:
            first = run_turn(context, self.model, environment, config, Host())
            runtime = self.model._runtime
            context.append(Message('user', 'next'))
            second = run_turn(context, self.model, environment, config, Host())
        self.assertEqual((first.kind, first.final_text), ('ended', 'cold:text-only'))
        self.assertEqual((second.kind, second.final_text), ('ended', 'warm:next'))
        self.assertIs(self.model._runtime, runtime)
        self.assertEqual(self.turns(runtime)[0]['pid'], self.fixture_state(runtime)['pid'])

    def test_the_idle_deadline_retires_the_process(self):
        self.relaunch(idle_timeout_seconds=1)
        context = self.answered()
        runtime = self.model._runtime
        self.assertIsNotNone(runtime)
        self.assertTrue(self.until(lambda: self.model._runtime is None, 4))
        self.assertTrue(runtime.closed.is_set())
        self.assertEqual(self.abandoned(), ['idle_deadline'])
        sample = self.ask(context, 'later')
        self.assertEqual(sample.last_assistant_text, 'cold:later')
        self.assertEqual(sample.recovery, ())  # no failure, nothing to report
        self.assertNotEqual(self.fixture_state()['pid'], self.fixture_state(runtime)['pid'])

    def test_the_deadline_counts_from_the_final_answers_request_start(self):
        self.relaunch(idle_timeout_seconds=2)
        self.answered('slow-final')  # 0.4 s from its request's start to its first event
        kept = self.trace.of('continuation_kept')[-1]
        self.assertLess(kept['idle_deadline_seconds'], 1.65)
        self.assertGreater(kept['idle_deadline_seconds'], 0)
        self.assertIsNone(kept['cache_lifetime'])  # its usage reported no cache writes

    def test_five_minute_cache_writes_shorten_the_deadline(self):
        self.relaunch(idle_timeout_seconds=30, short_cache_idle_timeout_seconds=1)
        context = self.answered('cache-5m')
        kept = self.trace.of('continuation_kept')[-1]
        self.assertEqual(kept['cache_lifetime'], '5m')
        self.assertLessEqual(kept['idle_deadline_seconds'], 1)
        self.assertTrue(self.until(lambda: self.model._runtime is None, 4))
        self.assertEqual(self.abandoned(), ['idle_deadline'])
        # The configured deadline, when shorter, still applies.
        self.relaunch(idle_timeout_seconds=1, short_cache_idle_timeout_seconds=30)
        self.ask(context, 'cache-5m again')
        self.assertLessEqual(self.trace.of('continuation_kept')[-1]['idle_deadline_seconds'], 1)
        self.assertTrue(self.until(lambda: self.model._runtime is None, 4))

    def test_no_idle_deadline_ends_the_process_at_each_final_answer(self):
        self.relaunch(idle_timeout_seconds=0)
        context = self.answered()
        self.assertIsNone(self.model._runtime)
        self.assertEqual(self.trace.of('continuation_kept'), [])
        self.assertEqual(self.abandoned(), [])  # as before: no deadline that passed
        self.assertEqual(self.ask(context, 'next').last_assistant_text, 'cold:next')

    def test_an_exit_or_a_stray_record_while_idle_ends_the_process_quietly(self):
        for marker, reason in (('idle-exit', 'idle_exit'), ('idle-record', 'idle_record'),
                               ('idle-telemetry', 'idle_record')):  # harmless mid-turn, not while idle
            with self.subTest(marker=marker):
                self.relaunch()
                context = self.answered(marker)
                runtime = self.model._runtime
                self.assertTrue(self.until(lambda: runtime.idle_end is not None, 3))
                self.assertEqual(runtime.idle_end, reason)
                sample = self.ask(context, 'next')
                self.assertEqual(sample.last_assistant_text, 'cold:next')
                self.assertEqual(sample.recovery, ())
                self.assertEqual(self.abandoned(), [reason])
                self.assertTrue(runtime.closed.is_set())

    def test_a_delivery_the_cli_does_not_take_up_is_answered_cold_without_a_failure(self):
        for marker in ('no-echo', 'no-queued', 'exit-on-delivery'):
            with self.subTest(marker=marker):
                self.relaunch(startup_timeout_seconds=3)
                context = self.answered()
                old = self.model._runtime
                sample = self.ask(context, marker)
                self.assertEqual(sample.last_assistant_text, 'cold:' + marker)
                self.assertEqual(sample.recovery, ('delivery_unacknowledged',))
                self.assertTrue(old.closed.is_set())
                self.assertNotEqual(self.fixture_state()['pid'], self.fixture_state(old)['pid'])
                self.assertEqual(self.abandoned(), ['delivery_unacknowledged'])

    def test_a_delivered_turn_that_breaks_the_protocol_fails(self):
        for marker in ('second-init', 'unknown-uuid', 'bad-result-index', 'bad-result-uuids'):
            with self.subTest(marker=marker):
                context = self.answered()
                context.append(Message('user', marker))
                with self.assertRaises(ModelResponseError):
                    self.model.sample(context)
                self.assertIsNone(self.model._runtime)

    def test_a_cold_imports_first_reply_must_answer_it(self):
        with self.assertRaisesRegex(ModelResponseError, 'does not answer the cold import'):
            self.model.sample(InteractionContext((Message('user', 'import-no-echo'),)))
        self.assertIsNone(self.model._runtime)

    def test_the_cli_closes_its_kept_process_when_it_exits(self):
        path = self.root / 'conversation.jsonl'
        result = subprocess.run(['/usr/bin/python3', '-m', 'pythia.interaction.cli', *fixtures.ModelTests.launch_options(self),
                                 '--headless', '--prompt', 'text-only', '--enable-default-tools', 'False',
                                 '--save', str(path), '--resume', 'False', '--cwd', str(self.root)],
                                cwd=fixtures.ROOT, env={**os.environ, 'HOME': str(self.root)},
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.until(lambda: not native_processes(self.broker.pid), 5))


def native_processes(root):
    """The fixture CLIs running under the broker ``root``."""
    parents, found = {}, set()
    for entry in os.listdir('/proc'):
        if entry.isdigit():
            try:
                with open(f'/proc/{entry}/stat') as stat:
                    parents[int(entry)] = int(stat.read().rsplit(')', 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
    frontier = {root}
    while frontier:
        frontier = {pid for pid, parent in parents.items() if parent in frontier} - found
        found |= frontier
    def fixture(pid):
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as cmdline:
                return b'claude_relay_cli.py' in cmdline.read()
        except OSError:
            return False
    return {pid for pid in found if fixture(pid)}


class MidTurnTests(WarmTests):
    """Part 2: user messages after a batch's results go to the running process."""

    def batch(self, context):
        sample = self.model.sample(context, tools=TOOLS)
        context.extend(sample.context_items())
        context.extend(ToolResult(call.call_id, f'r{i}') for i, call in enumerate(sample.tool_calls))
        return sample

    def test_a_steer_after_the_results_goes_with_them(self):
        self.relaunch()
        context = InteractionContext((Instructions(''), Message('user', 'two')))
        self.batch(context)
        runtime = self.model._runtime
        final = self.ask(context, 'also this', tools=TOOLS)
        self.assertEqual(final.last_assistant_text, 'r0|r1 +also this')
        self.assertEqual(final.recovery, ())
        self.assertIs(self.model._runtime, runtime)
        delivery = self.trace.of('continuation_delivery')[-1]
        self.assertEqual((delivery['message_count'], delivery['mid_turn']), (1, True))
        # The turn's result named both records, and the process goes on.
        self.assertEqual(self.ask(context, 'next', tools=TOOLS).last_assistant_text, 'warm:next')
        self.assertIs(self.model._runtime, runtime)

    def test_a_steer_in_a_later_turn_goes_with_its_results(self):
        self.relaunch()
        context = self.answered(tools=TOOLS)
        runtime = self.model._runtime
        context.append(Message('user', 'warm-tools'))
        self.batch(context)
        self.assertEqual(self.ask(context, 'steer', tools=TOOLS).last_assistant_text, 'r0|r1 +steer')
        self.assertIs(self.model._runtime, runtime)
        self.assertEqual(len(self.trace.of('claude_invocation')), 2)

    def test_a_missed_pickup_is_answered_cold_without_a_failure(self):
        self.relaunch()
        context = InteractionContext((Instructions(''), Message('user', 'two')))
        sample = self.batch(context)
        runtime = self.model._runtime
        final = self.ask(context, 'missed-steer now', tools=TOOLS)
        self.assertEqual(final.last_assistant_text, 'cold:missed-steer now')
        self.assertEqual(final.recovery, ('delivery_unacknowledged',))
        self.assertTrue(runtime.closed.is_set())
        self.assertEqual(self.fixture_state()['history_result_ids'], [call.call_id for call in sample.tool_calls])
        self.assertEqual(self.abandoned(), ['delivery_unacknowledged'])


class WarmPiTests(WarmTests):
    """warmpi's request reaches the live process; the compaction then retires it."""

    SETTINGS = CompactionSettings(mode='warmpi', keep_recent_tokens=0)

    def test_at_a_turns_start_the_instruction_goes_with_the_users_message(self):
        self.relaunch()
        context = self.answered()
        runtime = self.model._runtime
        context.append(Message('user', 'next task'))
        result = run_compaction(context, self.model, self.SETTINGS)
        self.assertEqual(result.protocol, 'warmpi')
        texts = self.turns(runtime)[-1]['texts']
        self.assertEqual(texts[0], 'next task')
        self.assertIn('Your next reply is a context checkpoint', texts[1])
        self.assertTrue(runtime.closed.is_set())  # retired when the compaction ended
        self.assertIsNone(self.model._runtime)

    def test_mid_turn_the_instruction_goes_with_the_results(self):
        self.relaunch()
        context = InteractionContext((Instructions(''), Message('user', 'two')))
        sample = self.model.sample(context, tools=TOOLS)
        context.extend(sample.context_items())
        context.extend(ToolResult(call.call_id, f'r{i}') for i, call in enumerate(sample.tool_calls))
        runtime = self.model._runtime
        result = run_compaction(context, self.model, self.SETTINGS, tools=TOOLS)
        self.assertEqual(result.protocol, 'warmpi')
        self.assertTrue(self.trace.of('continuation_delivery')[-1]['mid_turn'])
        self.assertEqual(len(self.trace.of('claude_invocation')), 2)  # no new process for the summary
        self.assertTrue(runtime.closed.is_set())
        # The next sample starts from the checkpoint, in a new process.
        context.extend(result.context_items())
        self.assertEqual(self.ask(context, 'text-only', tools=TOOLS).last_assistant_text, 'cold:text-only')


class ParkedYieldTests(WarmTests):
    """After a yield, the process waits for its batch's results, as long as an idle
    one waits after a final answer; the turn that resumes delivers them, with its
    message (doc/claude-relay-warm-yield-plan.md)."""

    def batch(self, text='two'):
        """A context whose turn yielded with a tool batch, which the model parked."""
        context = InteractionContext((Instructions(''), Message('user', text)))
        sample = self.model.sample(context, tools=TOOLS)
        context.extend(sample.context_items())
        self.model.park()
        return context, sample

    def resume(self, context, sample, text):
        """The batch's results and the turn's end, then the resuming turn's message."""
        context.extend(ToolResult(call.call_id, f'r{i}') for i, call in enumerate(sample.tool_calls))
        context.append(TurnSummary(sample_count=1))
        final = self.ask(context, text, tools=TOOLS)
        context.append(UserInteractionBoundary())
        return final

    def test_the_turn_that_resumes_gets_the_results_with_its_message_warm(self):
        self.relaunch()
        context, sample = self.batch()
        runtime = self.model._runtime
        [parked] = self.trace.of('continuation_parked')
        # An hour, less the time since the batch's request started.
        self.assertTrue(3590 < parked['parked_deadline_seconds'] <= 3595, parked)
        self.assertIsNone(parked['cache_lifetime'])  # the fixture's batch wrote no cache entries
        final = self.resume(context, sample, 'Wake-up from icron')
        self.assertEqual(final.last_assistant_text, 'r0|r1 +Wake-up from icron')
        self.assertEqual(final.recovery, ())
        self.assertIs(self.model._runtime, runtime)
        delivery = self.trace.of('continuation_delivery')[-1]
        self.assertEqual((delivery['message_count'], delivery['mid_turn']), (1, True))
        self.assertEqual(self.abandoned(), [])
        self.assertEqual(len(self.trace.of('claude_invocation')), 2)  # the version query, then the CLI

    def test_at_the_deadline_the_process_is_retired_quietly(self):
        self.relaunch(idle_timeout_seconds=1)
        context, sample = self.batch()
        runtime = self.model._runtime
        self.assertLessEqual(self.trace.of('continuation_parked')[-1]['parked_deadline_seconds'], 1)
        self.assertTrue(self.until(lambda: self.model._runtime is None, 4))
        self.assertTrue(runtime.closed.is_set())
        self.assertEqual(self.abandoned(), ['parked_deadline'])
        final = self.resume(context, sample, 'later')
        self.assertEqual(final.last_assistant_text, 'cold:later')
        self.assertEqual(final.recovery, ())  # no failure, nothing to report
        self.assertNotEqual(self.fixture_state()['pid'], self.fixture_state(runtime)['pid'])

    def test_five_minute_cache_writes_shorten_the_deadline(self):
        self.relaunch(idle_timeout_seconds=30, short_cache_idle_timeout_seconds=1)
        self.batch('cache-5m')
        parked = self.trace.of('continuation_parked')[-1]
        self.assertEqual(parked['cache_lifetime'], '5m')
        self.assertLessEqual(parked['parked_deadline_seconds'], 1)
        self.assertTrue(self.until(lambda: self.model._runtime is None, 4))
        self.assertEqual(self.abandoned(), ['parked_deadline'])

    def test_the_deadline_never_passes_the_results_due_time(self):
        # The native client gives up 6 s after the call: results due 1 s after it.
        self.relaunch(native_tool_timeout_seconds=6)
        context, sample = self.batch()
        parked = self.trace.of('continuation_parked')[-1]
        self.assertLessEqual(parked['parked_deadline_seconds'], 1)
        self.assertLessEqual(time.monotonic() + parked['parked_deadline_seconds'], sample.results_due + .1)
        self.assertTrue(self.until(lambda: self.model._runtime is None, 4))
        self.assertEqual(self.abandoned(), ['parked_deadline'])

    def test_otherwise_park_keeps_nothing(self):
        # No kept processes at all.
        self.relaunch(idle_timeout_seconds=0)
        context = InteractionContext((Instructions(''), Message('user', 'two')))
        self.model.sample(context, tools=TOOLS)
        runtime = self.model._runtime
        self.model.park()
        self.assertIsNone(self.model._runtime)
        self.assertTrue(runtime.closed.is_set())
        # A batch whose call the CLI rewrote: its process is retired already.
        self.relaunch()
        sample = self.model.sample(InteractionContext((Message('user', 'rewritten-tool-input'),)), tools=TOOLS)
        self.assertEqual(len(sample.tool_calls), 1)
        self.assertIsNone(self.model._runtime)
        self.model.park()
        self.assertIsNone(self.model._runtime)
        # After a final answer, the idle process keeps its own deadline.
        self.relaunch()
        self.answered()
        runtime = self.model._runtime
        self.model.park()
        self.assertIs(self.model._runtime, runtime)
        self.assertEqual(self.trace.of('continuation_parked'), [])
        self.assertEqual(len(self.trace.of('continuation_kept')), 1)

    def test_the_cli_resumes_an_icron_yield_in_the_same_process(self):
        # The fixture's model schedules a message a second away and yields; the
        # headless CLI waits for icron, and the wake-up's turn goes to the
        # process that made the call.
        path = self.root / 'conversation.jsonl'
        result = subprocess.run(['/usr/bin/python3', '-m', 'pythia.interaction.cli',
                                 *fixtures.ModelTests.launch_options(self), '--headless',
                                 '--enable-experimental-icron', '--prompt', 'icron-yield',
                                 '--enable-default-tools', 'False', '--debug-trace',
                                 '--save', str(path), '--resume', 'False', '--cwd', str(self.root)],
                                cwd=fixtures.ROOT, env={**os.environ, 'HOME': str(self.root)},
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        # The CLI's full log and traces are in its save's private directory.
        full = full_log_path(path)
        saved = load_interaction_save(full)
        [*_, answer] = [item for item in saved if isinstance(item, Message) and item.role == 'assistant']
        self.assertRegex(answer.content_text, r'^pid=\d+ yielded=True \+Wake-up from icron at ')
        self.assertFalse(any(isinstance(item, ModelFailure) for item in saved))
        events = [json.loads(line) for line in full.with_name('trace.events.jsonl').read_text().splitlines()]
        kinds = [event['type'] for event in events]
        self.assertEqual(kinds.count('claude_invocation'), 2)  # one process: the version query, then the CLI
        self.assertEqual(kinds.count('continuation_parked'), 1)
        [delivery] = [event for event in events if event['type'] == 'continuation_delivery']
        self.assertTrue(delivery['mid_turn'])
        self.assertNotIn('continuation_abandoned', kinds)
        self.assertTrue(self.until(lambda: not native_processes(self.broker.pid), 5))


if __name__ == '__main__':
    unittest.main()
