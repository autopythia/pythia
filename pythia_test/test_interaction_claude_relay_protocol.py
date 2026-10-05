"""Sanitized 2.1.289 record shapes, no native binary or network access."""
import unittest
from unittest import mock

from pythia.interaction.claude_relay._cli_protocol import Assembler
from pythia.interaction.claude_relay._runtime import Runtime
from pythia.interaction.model import ModelResponseError


def stream_event(kind, **data):
    return {'type': 'stream_event', 'event': {'type': kind, **data}}


def snapshot(blocks, *, ident='m', reason=None, usage=None):
    return {'type': 'assistant', 'message': {'id': ident, 'content': blocks,
            'stop_reason': reason, 'usage': {'input_tokens': 468, 'output_tokens': 4} if usage is None else usage}}


THINKING = {'type': 'thinking', 'thinking': '', 'signature': 'synthetic-not-a-real-signature'}
TEXT = {'type': 'text', 'text': '346'}


def observed_records():
    """Actual ordering, but synthetic IDs/signature and harmless public text."""
    return [
        stream_event('message_start', message={'id': 'm', 'usage': {'input_tokens': 468, 'output_tokens': 4}}),
        stream_event('content_block_start', index=0, content_block={'type': 'thinking', 'thinking': ''}),
        stream_event('content_block_delta', index=0, delta={'type': 'thinking_delta', 'thinking': ''}),
        stream_event('content_block_delta', index=0, delta={'type': 'signature_delta', 'signature': THINKING['signature']}),
        snapshot([THINKING]),  # BEFORE block_stop, NOT a complete assistant turn
        stream_event('content_block_stop', index=0),
        stream_event('content_block_start', index=1, content_block={'type': 'text', 'text': ''}),
        stream_event('content_block_delta', index=1, delta={'type': 'text_delta', 'text': TEXT['text']}),
        snapshot([TEXT]),
        stream_event('content_block_stop', index=1),
        stream_event('message_delta', delta={'stop_reason': 'end_turn'}, usage={'input_tokens': 468, 'output_tokens': 546}),
        stream_event('message_stop'),
    ]


class NativeStreamTests(unittest.TestCase):
    def complete(self, assembler=None, records=None):
        assembler = assembler or Assembler()
        records = observed_records() if records is None else records
        for value in records[:-1]:
            self.assertIsNone(assembler.feed(value))
        return assembler.feed(records[-1])

    def test_per_block_records_do_not_emit_or_replace_final_usage(self):
        message = self.complete()
        self.assertEqual(message.stop_reason, 'end_turn')
        self.assertEqual(message.usage.output_tokens, 546)
        self.assertEqual(message.usage.input_tokens, 468)
        self.assertEqual(message.blocks, (THINKING, TEXT))
        items, calls = message.sample_items('generation', {})
        self.assertEqual([item.content_text for item in items], ['346'])
        self.assertEqual(calls, [])

    def test_late_null_stop_shadows_must_match_but_usage_is_provisional(self):
        assembler = Assembler()
        self.complete(assembler)
        for blocks in ([THINKING], [TEXT], [THINKING, TEXT]):
            self.assertIsNone(assembler.feed(snapshot(blocks)))
        with self.assertRaises(ModelResponseError):
            assembler.feed(snapshot([{'type': 'text', 'text': 'different'}]))
        with self.assertRaises(ModelResponseError):
            assembler.feed(snapshot([TEXT], usage={'output_tokens': -1}))

    def test_shadows_conflicting_with_stream_fail_before_emit(self):
        for index, value in ((8, snapshot([{'type': 'text', 'text': 'different'}])),
                             (8, snapshot([TEXT], reason='end_turn'))):  # explicit complete record cannot be partial
            records = observed_records()
            records[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ModelResponseError):
                self.complete(records=records)

    def test_completed_records_still_reject_reason_and_usage_conflicts(self):
        for reason, usage in (('max_tokens', {'output_tokens': 546}), ('end_turn', {'output_tokens': 4})):
            assembler = Assembler()
            self.complete(assembler)
            with self.subTest(reason=reason), self.assertRaises(ModelResponseError):
                assembler.feed(snapshot([THINKING, TEXT], reason=reason, usage=usage))

    def test_provisional_records_cannot_supply_missing_stop_or_start_new_message(self):
        records = observed_records()
        records[-2]['event']['delta'] = {}
        with self.assertRaises(ModelResponseError):
            self.complete(records=records)
        with self.assertRaises(ModelResponseError):
            Assembler().feed(snapshot([TEXT]))
        assembler = Assembler()
        assembler.feed(observed_records()[0])
        with self.assertRaises(ModelResponseError):
            assembler.feed(snapshot([TEXT], ident='other', reason='end_turn'))

    def test_start_and_shadow_output_usage_are_not_a_final_usage_fallback(self):
        records = observed_records()
        records[-2]['event']['usage'] = {}
        with self.assertRaisesRegex(ModelResponseError, 'final output usage'):
            self.complete(records=records)
        records.insert(-1, snapshot([THINKING, TEXT], reason='end_turn',
                                    usage={'input_tokens': 468, 'output_tokens': 546}))
        self.assertEqual(self.complete(records=records).usage.output_tokens, 546)

    def test_incomplete_block_malformed_delta_and_hint_flood_rejected(self):
        records = observed_records()
        del records[-3]  # no text content_block_stop
        with self.assertRaises(ModelResponseError):
            self.complete(records=records)
        for delta in ({'type': 'input_json_delta', 'partial_json': '{}'},
                      {'type': 'signature_delta', 'signature': 1}):
            records = observed_records()
            records[3]['event']['delta'] = delta
            with self.subTest(delta=delta), self.assertRaises(ModelResponseError):
                self.complete(records=records)
        assembler = Assembler()
        assembler.feed(observed_records()[0])
        with mock.patch('pythia.interaction.claude_relay._cli_protocol.MAX_RECORD', 32), self.assertRaises(ModelResponseError):
            assembler.feed(snapshot([TEXT]))

    def test_tool_snapshots_wait_for_complete_batch_and_valid_stop(self):
        for reason in ('tool_use', 'max_tokens'):
            assembler = Assembler()
            assembler.feed(stream_event('message_start', message={'id': 'tools'}))
            for index in range(2):
                block = {'type': 'tool_use', 'id': f'toolu_{index}', 'name': 'mcp__pythia__echo', 'input': {'v': index}}
                self.assertIsNone(assembler.feed(stream_event('content_block_start', index=index, content_block=block)))
                self.assertIsNone(assembler.feed(snapshot([block], ident='tools')))
                self.assertIsNone(assembler.feed(stream_event('content_block_stop', index=index)))
            assembler.feed(stream_event('message_delta', delta={'stop_reason': reason}, usage={'output_tokens': 20}))
            message = assembler.feed(stream_event('message_stop'))
            if reason == 'tool_use':
                self.assertEqual(len(message.sample_items('g', {'mcp__pythia__echo': 'echo'})[1]), 2)
            else:
                with self.assertRaises(ModelResponseError):
                    message.sample_items('g', {'mcp__pythia__echo': 'echo'})


class SystemEventTests(unittest.TestCase):
    def runtime(self):
        value = Runtime.__new__(Runtime)  # no subprocesses, threads, or sockets
        value.initialized, value.names = False, {}
        return value

    def test_narrow_telemetry_after_init_only(self):
        runtime = self.runtime()
        status = {'subtype': 'status', 'status': 'requesting'}
        with self.assertRaises(ModelResponseError):
            runtime._system(status)
        runtime._system({'subtype': 'init', 'tools': []})
        runtime._system(status)
        runtime._system({'subtype': 'thinking_tokens', 'estimated_tokens': 50, 'estimated_tokens_delta': 50})
        self.assertTrue(runtime.initialized)

    def test_compaction_hooks_unknown_status_and_invalid_telemetry_still_fail(self):
        runtime = self.runtime()
        runtime._system({'subtype': 'init', 'tools': []})
        invalid = [{'subtype': subtype} for subtype in ('compact_boundary', 'hook_started', 'unknown', 'init')]
        invalid += [{'subtype': 'status', 'status': status} for status in ('compacting', None, 'unknown')]
        invalid += [{'subtype': 'thinking_tokens', 'estimated_tokens': value, 'estimated_tokens_delta': 1}
                    for value in (-1, True, 1.5, None)]
        invalid.append({'subtype': 'thinking_tokens', 'estimated_tokens': 50})
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ModelResponseError):
                runtime._system(value)


if __name__ == '__main__':
    unittest.main()
