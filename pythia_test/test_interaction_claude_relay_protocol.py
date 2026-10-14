"""Sanitized 2.1.289 record shapes, no native binary or network access."""
import unittest
from unittest import mock

from pythia.interaction.claude_relay._cli_protocol import Assembler, NativeRefusal
from pythia.interaction.claude_relay._runtime import Runtime
from pythia.interaction.model import ModelRefusal, ModelResponseError


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


# Observed 2.1.289 refusal (Oct 14): its reason only in message_delta.
REFUSAL_DETAILS = {'type': 'refusal', 'category': 'cyber', 'explanation': (
    "This request triggered restrictions on violative cyber content and was blocked under Anthropic's Usage "
    "Policy. To learn more, see https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback.")}


def refused_records(details=REFUSAL_DETAILS):
    """The observed refused stream: thinking without text, then a refusal stop."""
    return [
        stream_event('message_start', message={'id': 'm', 'stop_details': None,
                                               'usage': {'input_tokens': 2, 'output_tokens': 5}}),
        stream_event('content_block_start', index=0, content_block={'type': 'thinking', 'thinking': '', 'signature': ''}),
        stream_event('content_block_delta', index=0, delta={'type': 'thinking_delta', 'thinking': ''}),
        stream_event('content_block_delta', index=0, delta={'type': 'signature_delta', 'signature': THINKING['signature']}),
        snapshot([THINKING]),
        stream_event('content_block_stop', index=0),
        stream_event('message_delta', delta={'stop_reason': 'refusal', 'stop_sequence': None, 'stop_details': details},
                     usage={'input_tokens': 2, 'output_tokens': 11152}),
        stream_event('message_stop'),
    ]


def outcome(records):
    """Feed every record; the completed message and its sample items."""
    assembler, message = Assembler(), None
    for value in records:
        message = assembler.feed(value)
    return message, message.sample_items('generation', {'mcp__pythia__echo': 'echo'})


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

    def test_a_call_without_arguments_streams_its_input_as_empty_json(self):
        # Observed 2.1.289 shape: start input {}, then one empty partial_json.
        start = {'type': 'tool_use', 'id': 'toolu_x', 'name': 'mcp__pythia__echo', 'input': {}, 'caller': {'type': 'direct'}}
        def records(partials, final_input):
            return [stream_event('message_start', message={'id': 'tools', 'usage': {'input_tokens': 2}}),
                    stream_event('content_block_start', index=0, content_block=start),
                    *(stream_event('content_block_delta', index=0, delta={'type': 'input_json_delta', 'partial_json': text})
                      for text in partials),
                    snapshot([{**start, 'input': final_input}], ident='tools'),
                    stream_event('content_block_stop', index=0),
                    stream_event('message_delta', delta={'stop_reason': 'tool_use'}, usage={'output_tokens': 20}),
                    stream_event('message_stop')]
        for partials, final_input, encoded in ((('',), {}, '{}'), ((), {}, '{}'),
                                               (('', '{"value":', '1}'), {'value': 1}, '{"value":1}')):
            with self.subTest(partials=partials):
                _, (items, calls) = outcome(records(partials, final_input))
                self.assertEqual([(item.name, item.arguments_json) for item in items], [('echo', encoded)])
                self.assertEqual(calls, [('toolu_x', 'echo', final_input)])
        for partials in ((' ',), ('{',)):  # anything else still has to be complete JSON
            with self.subTest(partials=partials), self.assertRaisesRegex(ModelResponseError, 'tool JSON'):
                outcome(records(partials, {}))


class RefusalTests(unittest.TestCase):
    def test_the_observed_refusal_reports_its_category_and_explanation(self):
        with self.assertRaises(NativeRefusal) as caught:
            outcome(refused_records())
        refusal = caught.exception
        self.assertIsInstance(refusal, ModelRefusal)  # shared with the Messages adapter
        self.assertIsInstance(refusal, ModelResponseError)
        self.assertEqual((refusal.error_code, refusal.kind, refusal.category), ('refusal', 'refusal', 'cyber'))
        self.assertEqual(refusal.explanation, REFUSAL_DETAILS['explanation'])
        self.assertEqual(str(refusal), 'Request refused (category=cyber): ' + REFUSAL_DETAILS['explanation'])

    def test_any_reason_is_reported_as_given_on_one_bounded_line(self):
        cases = (({'type': 'refusal', 'category': 'bio'}, 'Request refused (category=bio)'),
                 ({'type': 'other_block', 'explanation': 'Two\nlines,\x00 one\tline.'},
                  'Request refused (type=other_block): Two lines, one line.'),
                 ({'category': 'x' * 100}, 'Request refused (category=' + 'x' * 61 + '...)'),
                 (None, 'Request refused (no reason given)'),
                 ('not an object', 'Request refused (no reason given)'))
        for details, expected in cases:
            with self.subTest(details=details), self.assertRaises(NativeRefusal) as caught:
                outcome(refused_records(details))
            self.assertEqual(str(caught.exception), expected)
        with self.assertRaises(NativeRefusal) as caught:
            outcome(refused_records({'explanation': 'y' * 1000}))
        self.assertEqual(caught.exception.explanation, 'y' * 397 + '...')

    def test_a_refusal_before_complete_content_or_with_tools_still_reports_it(self):
        refusal = stream_event('message_delta', delta={'stop_reason': 'refusal', 'stop_details': REFUSAL_DETAILS},
                               usage={'output_tokens': 1})
        start = stream_event('message_start', message={'id': 'm', 'usage': {'input_tokens': 2}})
        tool = {'type': 'tool_use', 'id': 'toolu_x', 'name': 'mcp__pythia__echo', 'input': {'value': 1}}
        shapes = {
            'no content': [start, refusal, stream_event('message_stop')],
            'open block': [start, stream_event('content_block_start', index=0, content_block={'type': 'text', 'text': ''}),
                           stream_event('content_block_delta', index=0, delta={'type': 'text_delta', 'text': 'partial'}),
                           refusal, stream_event('message_stop')],
            'tool batch': [start, stream_event('content_block_start', index=0, content_block=tool),
                           stream_event('content_block_stop', index=0), refusal, stream_event('message_stop')],
            'complete record': [{'type': 'assistant', 'message': {
                'id': 'r', 'content': [], 'stop_reason': 'refusal', 'stop_details': REFUSAL_DETAILS,
                'usage': {'input_tokens': 2, 'output_tokens': 1}}}],
        }
        for shape, records in shapes.items():
            with self.subTest(shape=shape), self.assertRaises(NativeRefusal) as caught:
                outcome(records)
            self.assertEqual(caught.exception.category, 'cyber')

    def test_a_refusal_after_usable_text_stays_a_stop_reason(self):
        records = observed_records()
        records[-2] = stream_event('message_delta', delta={'stop_reason': 'refusal', 'stop_details': REFUSAL_DETAILS},
                                   usage={'input_tokens': 468, 'output_tokens': 546})
        message, (items, calls) = outcome(records)
        self.assertEqual((message.stop_reason, message.stop_details), ('refusal', REFUSAL_DETAILS))
        self.assertEqual([item.content_text for item in items], ['346'])

    def test_thinking_or_blank_text_without_an_answer_is_still_a_refusal(self):
        # Visible thinking alone is not an answer: the turn would otherwise end
        # as "no final assistant text", without the refusal's reason.
        visible = {**THINKING, 'thinking': 'Visible reasoning.'}
        records = refused_records()
        records[2] = stream_event('content_block_delta', index=0,
                                  delta={'type': 'thinking_delta', 'thinking': visible['thinking']})
        records[4] = snapshot([visible])
        blank = {'type': 'text', 'text': ' \n'}
        blank_text = refused_records()
        blank_text[-2:-2] = [stream_event('content_block_start', index=1, content_block={'type': 'text', 'text': ''}),
                             stream_event('content_block_delta', index=1, delta={'type': 'text_delta', 'text': blank['text']}),
                             snapshot([blank]), stream_event('content_block_stop', index=1)]
        for case, shape in (('visible thinking', records), ('blank text', blank_text)):
            with self.subTest(case=case), self.assertRaises(NativeRefusal) as caught:
                outcome(shape)
            self.assertEqual(caught.exception.category, 'cyber')


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
