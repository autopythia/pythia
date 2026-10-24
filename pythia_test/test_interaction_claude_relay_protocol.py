"""Sanitized 2.1.289 record shapes, no native binary or network access."""
import json
import unittest
from unittest import mock

from pythia.interaction.claude_relay._cli_protocol import Assembler, MAX_CONSECUTIVE_INTERRUPTIONS
from pythia.interaction.claude_relay._cli_protocol import NativeRefusal, NativeStreamInterrupted, native_input_rewrite
from pythia.interaction.claude_relay._cli_protocol import NativeToolInputRewritten, block_difference, input_difference
from pythia.interaction.claude_relay._mcp import Mailbox
from pythia.interaction.claude_relay._runtime import Runtime
from pythia.interaction.environment import ToolSpec
from pythia.interaction.items import Message, Reasoning, ToolCall
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


def cut_records(ident='cut', *, closed=True):
    """The observed cut message (2.1.289, untraced): thinking deltas without a
    signature or block snapshot, then content_block_stop and message_stop, and
    no message_delta."""
    return [
        stream_event('message_start', message={'id': ident, 'usage': {'input_tokens': 2, 'output_tokens': 5}}),
        stream_event('content_block_start', index=0, content_block={'type': 'thinking', 'thinking': '', 'signature': ''}),
        *(stream_event('content_block_delta', index=0, delta={'type': 'thinking_delta', 'thinking': 'cut '})
          for _ in range(3)),
        *((stream_event('content_block_stop', index=0),) if closed else ()),
        stream_event('message_stop'),
    ]


def feed_all(assembler, records):
    """Feed records; return what the last one completed."""
    for value in records[:-1]:
        assert assembler.feed(value) is None, value
    return assembler.feed(records[-1])


# Observed 2.1.289 (Oct 16), synthetic contents: the model streams literal
# \u2013 text in a tool input; the CLI's snapshot shows each one-backslash
# escape decoded into an en dash, and the escaped backslash (two) left alone.
STREAMED_PATCH = ('+    self.assertIn("lines 1\\u201311;", pointer)\n'
                  '+    self.assertNotIn("1\\u20132", pointer)  # nor "\\\\u2013"\n')
NATIVE_PATCH = ('+    self.assertIn("lines 1\u201311;", pointer)\n'
                '+    self.assertNotIn("1\u20132", pointer)  # nor "\\\\u2013"\n')
TOOL_START = {'type': 'tool_use', 'id': 'toolu_x', 'name': 'mcp__pythia__echo', 'input': {}, 'caller': {'type': 'direct'}}


def rewrite_records(snapshot_input, *, fragment=7, streamed=None):
    """The failing message's observed shape: thinking, text, then a tool block
    streamed in many fragments, each block followed by the CLI's snapshot (none
    of the tool block for ``snapshot_input=None``)."""
    raw = json.dumps({'patch': STREAMED_PATCH} if streamed is None else streamed)
    return [*observed_records()[:-2],
            stream_event('content_block_start', index=2, content_block=TOOL_START),
            *(stream_event('content_block_delta', index=2,
                           delta={'type': 'input_json_delta', 'partial_json': raw[start:start + fragment]})
              for start in range(0, len(raw), fragment)),
            *(() if snapshot_input is None else (snapshot([{**TOOL_START, 'input': snapshot_input}]),)),
            stream_event('content_block_stop', index=2),
            stream_event('message_delta', delta={'stop_reason': 'tool_use'},
                         usage={'input_tokens': 468, 'output_tokens': 546}),
            stream_event('message_stop')]


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


class NativeInputRewriteTests(unittest.TestCase):
    def test_the_rule_decodes_unicode_escapes_after_an_odd_backslash_run(self):
        # All observed on 2.1.289, in traces or the live probe (October 24).
        cases = {
            'lines 1\\u201311': 'lines 1\u201311',  # one backslash
            '\\\\\\u2013': '\\\\\u2013',  # three: an escaped backslash, then an escape
            '\\\\\\\\\\u2013': '\\\\\\\\\u2013',  # five
            '\\ufeffone\\r\\ntwo': '\ufeffone\\r\\ntwo',  # \r and \n stay text
            '\\uFEFF \\u00E9 \\u00eA': '\ufeff \u00e9 \u00ea',  # either hex case
            '\\u0022\\u0027\\u005c \\u0041 \\u00a0 \\ufffe': '"\'\\ A \u00a0 \ufffe',  # ASCII, U+00A0, noncharacters
            '\\u005cu2013': '\\u2013',  # one pass: a decoded backslash starts no escape
            '\\ud83d\\ude00 \\uD83D\\uDE00': '\U0001F600 \U0001F600',  # a valid pair is one character
            '\\ud83d\\ud83d\\ude00': '\\ud83d\U0001F600',  # a lone high, then a pair
            '\\u2013\\udc00 \\u001b\\udc00': '\u2013\\udc00 \\u001b\\udc00',  # a low after any other escape is lone
        }
        for streamed, native in cases.items():
            with self.subTest(streamed=streamed):
                self.assertEqual(native_input_rewrite(streamed), native)
        for kept in (
            '\\\\u2013', '\\\\\\\\u2013', '\\\\\\\\\\\\u2013',  # even runs: escaped backslashes
            # Controls and lone surrogates stay as written, in either case.
            '\\u0000 \\u0009 \\u000a \\u000d \\u001B \\u001f \\u007f \\u0080 \\u0085 \\u009B \\u009f',
            '\\ud83d x \\ude00 \\ude00\\ud83d \\ude00\\ude00 \\uDC80',
            '\\u201 \\u20G3 \\U0001F600 \\U2013 \\x41 u2013',  # not escapes
        ):
            with self.subTest(kept=kept):
                self.assertIs(native_input_rewrite(kept), kept)
        value = {'cmd': 'a\\u2013', 'plan': [{'step': 'b\\u2013', 'status': 'pending'}], 'k\\u2013': 1, 'yes': True}
        self.assertEqual(native_input_rewrite(value), {'cmd': 'a\u2013', 'plan': [{'step': 'b\u2013', 'status': 'pending'}],
                                                       'k\\u2013': 1, 'yes': True})  # keys are left alone
        unchanged = {'cmd': 'no \\n escapes \u2013', 'items': [1, 'two', {'three': None}], 'k\\u2013': 2}
        self.assertIs(native_input_rewrite(unchanged), unchanged)

    def test_the_observed_tool_snapshot_shows_the_rewrite_and_the_stream_is_adopted(self):
        for fragment in (1, 7, 10_000):  # the escapes are split across fragments, or not
            with self.subTest(fragment=fragment):
                message, (items, calls) = outcome(rewrite_records({'patch': NATIVE_PATCH}, fragment=fragment))
                self.assertEqual(message.rewritten_tool_ids, ('toolu_x',))
                self.assertEqual(calls, [('toolu_x', 'echo', {'patch': STREAMED_PATCH})])
                self.assertEqual(json.loads(items[-1].arguments_json), {'patch': STREAMED_PATCH})
        self.assertEqual(outcome(observed_records())[0].rewritten_tool_ids, ())

    def test_controls_and_lone_surrogates_stay_as_written_in_the_copies(self):
        # The shape of the October 23 and 24 failures, synthetic contents: the
        # CLI decodes the bidi escape and the pair, and keeps the others.
        streamed = {'cmd': 'print({"csi": "\\u009b31m", "nel": "a\\u0085b", "rlo": "f\\u202eg", '
                           '"lone": ["\\ud800", "\\udc80"], "pair": "\\ud83d\\ude00", "esc": "\\u001b[0m"})'}
        copy = {'cmd': 'print({"csi": "\\u009b31m", "nel": "a\\u0085b", "rlo": "f\u202eg", '
                       '"lone": ["\\ud800", "\\udc80"], "pair": "\U0001F600", "esc": "\\u001b[0m"})'}
        message, (items, calls) = outcome(rewrite_records(copy, streamed=streamed))
        self.assertEqual(message.rewritten_tool_ids, ('toolu_x',))
        self.assertEqual(calls, [('toolu_x', 'echo', streamed)])
        self.assertEqual(json.loads(items[-1].arguments_json), streamed)

    def test_a_call_is_rewritten_only_when_the_cli_s_copies_show_it(self):
        # A copy as streamed (2.1.289 kept a whole string at times) is the CLI's
        # history: no rewrite, so the warm process goes on.
        for snapshot_input, rewritten in (({'patch': STREAMED_PATCH}, ()), ({'patch': NATIVE_PATCH}, ('toolu_x',)),
                                          (None, ('toolu_x',))):  # no copy: the rule decides
            with self.subTest(rewritten=rewritten, snapshot=snapshot_input is not None):
                message, (_, calls) = outcome(rewrite_records(snapshot_input))
                self.assertEqual((message.rewritten_tool_ids, calls[0][2]), (rewritten, {'patch': STREAMED_PATCH}))
        # A complete record decides as a block snapshot does.
        usage = {'input_tokens': 468, 'output_tokens': 546}
        for copy, rewritten in ((STREAMED_PATCH, ()), (NATIVE_PATCH, ('toolu_x',))):
            with self.subTest(complete=rewritten):
                records = rewrite_records(None)
                records.insert(-1, snapshot([THINKING, TEXT, {**TOOL_START, 'input': {'patch': copy}}],
                                            reason='tool_use', usage=usage))
                self.assertEqual(feed_all(Assembler(), records).rewritten_tool_ids, rewritten)
        # Once judged as streamed, a later record can't show the rewrite.
        assembler = Assembler()
        self.assertEqual(feed_all(assembler, rewrite_records({'patch': STREAMED_PATCH})).rewritten_tool_ids, ())
        with self.assertRaisesRegex(ModelResponseError, 'snapshot disagrees'):
            assembler.feed(snapshot([{**TOOL_START, 'input': {'patch': NATIVE_PATCH}}]))

    def test_any_other_difference_fails_naming_the_call_and_where_it_differs(self):
        prefix = ('Native CLI rewrote tool call toolu_x (mcp__pythia__echo) beyond its known \\uXXXX decoding, '
                  'in its snapshot: ')
        cases = {  # the copy keeps an escape the rule decodes, decodes an escaped backslash, etc.
            NATIVE_PATCH.replace('1\u20132', '1\\u20132'): 'argument patch has a backslash at line 2, column 25, '
                                                           'where a non-ASCII character was expected (103 characters, 98 expected)',
            NATIVE_PATCH.replace('\\\\u2013', '\\\u2013'): 'argument patch has a non-ASCII character at line 2, column 48, '
                                                           'where a backslash was expected (93 characters, 98 expected)',
            NATIVE_PATCH + ' ': 'argument patch has whitespace at line 3, column 1, past its expected end '
                                '(99 characters, 98 expected)',
            NATIVE_PATCH[:40]: 'argument patch ends at line 1, column 41, where a letter or digit was expected '
                               '(40 characters, 98 expected)',
        }
        for patch, detail in cases.items():
            with self.subTest(detail=detail), self.assertRaises(NativeToolInputRewritten) as caught:
                outcome(rewrite_records({'patch': patch}))
            self.assertEqual(str(caught.exception), prefix + detail + '; no tool calls may execute')
            self.assertEqual(caught.exception.error_code, 'native_tool_input_rewritten')
            self.assertNotIn('pointer', str(caught.exception))  # no model output is copied
        with self.assertRaisesRegex(NativeToolInputRewritten, 'in its snapshot: its input adds value;'):
            outcome(rewrite_records({'patch': NATIVE_PATCH, 'value': 1}))
        # Text and thinking blocks are not rewritten (2.1.289 left both alone).
        records = observed_records()
        records[7] = stream_event('content_block_delta', index=1, delta={'type': 'text_delta', 'text': 'en \\u2013'})
        records[8] = snapshot([{'type': 'text', 'text': 'en \u2013'}])
        with self.assertRaises(ModelResponseError) as caught:
            outcome(records)
        self.assertEqual(str(caught.exception), 'Native assistant snapshot disagrees with its stream (its text block)')
        self.assertNotIsInstance(caught.exception, NativeToolInputRewritten)

    def test_differences_are_described_by_position_and_kind_only(self):
        cases = (
            ({'cmd': 'a b'}, {'cmd': 'a\tb'},
             'argument cmd has whitespace at line 1, column 2, where other whitespace was expected (3 characters, 3 expected)'),
            ({'cmd': 'x\ny\\u2013'}, {'cmd': 'x\nyz'},  # compared with the input as the rule rewrites it
             'argument cmd has a letter or digit at line 2, column 2, where a non-ASCII character was expected '
             '(4 characters, 4 expected)'),
            ({'plan': [{'step': 'a\\u2013'}]}, {'plan': [{'step': 'a-'}]},
             'argument plan[0].step has an ASCII symbol at line 1, column 2, where a non-ASCII character was expected '
             '(2 characters, 2 expected)'),
            ({'plan': [{'step': 'a'}]}, {'plan': [{'step': 'a'}, {'step': 'b'}]}, 'argument plan has 2 items where 1 was expected'),
            ({'value': 1}, {'value': '1'}, 'argument value is a string where a number was expected'),
            ({'value': 1}, {'value': 1.0}, 'argument value is a different number'),
            ({'cmd': 'x'}, {'cmd': 'x', 'secret key!': 0, 'b': 1, 'c': 2, 'd': 3}, 'its input adds <invalid>, b, c and 1 more'),
            ({'cmd': 'x'}, {}, 'its input lacks cmd'),
            ({'cmd': 'x'}, ['x'], 'its input is a list where an object was expected'),
        )
        for streamed, copy, detail in cases:
            with self.subTest(detail=detail):
                self.assertEqual(input_difference(streamed, copy), detail)
        block = {**TOOL_START, 'input': {'patch': STREAMED_PATCH}}
        self.assertEqual(block_difference(block, {**block, 'caller': {'type': 'other'}}), 'its caller field differs')
        error = NativeToolInputRewritten('bad id!', 'a name', 'MCP call', 'its caller field differs', adopted=True)
        self.assertEqual(str(error), 'Native CLI rewrote tool call <invalid> (<invalid>) beyond its known \\uXXXX '
                                     'decoding, in its MCP call: its caller field differs')

    def test_late_and_complete_records_compare_the_same_way(self):
        streamed = {**TOOL_START, 'input': {'patch': STREAMED_PATCH}}
        native = {**TOOL_START, 'input': {'patch': NATIVE_PATCH}}
        wrong = {**TOOL_START, 'input': {'patch': NATIVE_PATCH + ' '}}
        usage = {'input_tokens': 468, 'output_tokens': 546}
        records = rewrite_records({'patch': NATIVE_PATCH})
        records.insert(-1, snapshot([THINKING, TEXT, native], reason='tool_use', usage=usage))
        assembler = Assembler()
        self.assertEqual(feed_all(assembler, records).rewritten_tool_ids, ('toolu_x',))
        for blocks in ([native], [streamed], [THINKING, TEXT, native]):
            self.assertIsNone(assembler.feed(snapshot(blocks)))  # late block snapshots
        self.assertIsNone(assembler.feed(snapshot([THINKING, TEXT, native], reason='tool_use', usage=usage)))
        with self.assertRaisesRegex(ModelResponseError, 'snapshot disagrees'):
            assembler.feed(snapshot([wrong]))
        with self.assertRaisesRegex(ModelResponseError, 'Inconsistent repeated'):
            assembler.feed(snapshot([THINKING, TEXT, wrong], reason='tool_use', usage=usage))
        records[-2] = snapshot([THINKING, TEXT, wrong], reason='tool_use', usage=usage)
        with self.assertRaisesRegex(NativeToolInputRewritten, 'in its complete record: argument patch has whitespace'):
            feed_all(Assembler(), records)


def surrogate_records(tool_id='toolu_x'):
    """A message whose thinking, text and tool input hold lone surrogates, as
    JSON escapes decode them (\\udc80): the stream's and the CLI's copies agree."""
    thinking = {**THINKING, 'thinking': 'think \ud800'}
    tool = {**TOOL_START, 'id': tool_id}
    return [
        stream_event('message_start', message={'id': 'm', 'usage': {'input_tokens': 468, 'output_tokens': 4}}),
        stream_event('content_block_start', index=0, content_block={'type': 'thinking', 'thinking': ''}),
        stream_event('content_block_delta', index=0, delta={'type': 'thinking_delta', 'thinking': 'think \ud800'}),
        stream_event('content_block_delta', index=0, delta={'type': 'signature_delta', 'signature': THINKING['signature']}),
        snapshot([thinking]),
        stream_event('content_block_stop', index=0),
        stream_event('content_block_start', index=1, content_block={'type': 'text', 'text': ''}),
        stream_event('content_block_delta', index=1, delta={'type': 'text_delta', 'text': 'ok \udc80'}),
        snapshot([{'type': 'text', 'text': 'ok \udc80'}]),
        stream_event('content_block_stop', index=1),
        stream_event('content_block_start', index=2, content_block=tool),
        stream_event('content_block_delta', index=2,
                     delta={'type': 'input_json_delta', 'partial_json': '{"cmd": "echo \\udc80"}'}),
        snapshot([{**tool, 'input': {'cmd': 'echo \udc80'}}]),
        stream_event('content_block_stop', index=2),
        stream_event('message_delta', delta={'stop_reason': 'tool_use'}, usage={'input_tokens': 468, 'output_tokens': 546}),
        stream_event('message_stop'),
    ]


class LoneSurrogateTests(unittest.TestCase):
    """A lone surrogate can't be saved (UTF-8 can't encode one): the saved items
    hold U+FFFD instead, but the CLI's MCP call is matched with the stream."""

    def test_the_items_are_repaired_and_the_calls_keep_the_streamed_input(self):
        message, (items, calls) = outcome(surrogate_records())
        self.assertEqual(items, (Reasoning('think \ufffd'), Message('assistant', 'ok \ufffd'),
                                 ToolCall('echo', 'generation:toolu_x', '{"cmd":"echo \ufffd"}')))
        self.assertEqual(calls, [('toolu_x', 'echo', {'cmd': 'echo \udc80'})])
        # The CLI's MCP call carries the input as streamed, which matches the
        # call; the repaired input would not (the host runs that one).
        box = Mailbox((ToolSpec('echo', 'Return a value', {'type': 'object'}),), wait_seconds=2)
        box.register(calls)
        box.release({'toolu_x': ('done', True)})
        self.assertEqual(box.call('toolu_x', 'echo', {'cmd': 'echo \udc80'})['content'][0]['text'], 'done')
        box = Mailbox((ToolSpec('echo', 'Return a value', {'type': 'object'}),), wait_seconds=2)
        box.register(calls)
        with self.assertRaises(NativeToolInputRewritten):
            box.call('toolu_x', 'echo', {'cmd': 'echo \ufffd'})

    def test_a_native_tool_id_with_one_is_malformed(self):
        # The ID names the call to the CLI, so it can't be repaired.
        with self.assertRaisesRegex(ModelResponseError, 'malformed native tool use'):
            outcome(surrogate_records(tool_id='toolu_\udc80'))


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


class InterruptedMessageTests(unittest.TestCase):
    def test_the_observed_cut_message_is_discarded_and_a_retry_adopted(self):
        assembler = Assembler()
        self.assertIsNone(feed_all(assembler, cut_records()))
        self.assertIsNone(assembler.current)
        self.assertEqual((assembler.interruptions, assembler.consecutive_interruptions), (1, 1))
        self.assertEqual(assembler.last_interruption, {
            'native_message_id': 'cut', 'block_types': ('thinking',), 'open_blocks': 0,
            'unsigned_thinking_blocks': 1, 'consecutive': 1})
        message = feed_all(assembler, observed_records())  # the CLI's retry, a new stream
        self.assertEqual((message.id, message.blocks, message.usage.output_tokens), ('m', (THINKING, TEXT), 546))
        self.assertEqual(assembler.consecutive_interruptions, 0)
        self.assertEqual(assembler.interruptions, 1)

    def test_a_complete_record_in_place_of_a_new_stream_is_adopted(self):
        assembler = Assembler()
        feed_all(assembler, cut_records())
        message = assembler.feed(snapshot([TEXT], ident='fallback', reason='end_turn',
                                          usage={'input_tokens': 5, 'output_tokens': 4}))
        self.assertEqual((message.id, message.stop_reason, message.usage.total_tokens), ('fallback', 'end_turn', 9))

    def test_a_cut_with_an_open_block_is_discarded_too(self):
        assembler = Assembler()
        self.assertIsNone(feed_all(assembler, cut_records(closed=False)))
        self.assertEqual(assembler.last_interruption['open_blocks'], 1)

    def test_no_later_record_for_a_cut_message_is_adopted(self):
        assembler = Assembler()
        feed_all(assembler, cut_records())
        self.assertIsNone(assembler.feed(snapshot([THINKING], ident='cut')))  # a late block snapshot
        with self.assertRaisesRegex(ModelResponseError, 'interrupted message'):
            assembler.feed(snapshot([THINKING], ident='cut', reason='end_turn',
                                    usage={'input_tokens': 2, 'output_tokens': 9}))
        with self.assertRaisesRegex(ModelResponseError, 'message ID'):
            assembler.feed(cut_records()[0])  # a new stream reusing its ID

    def test_a_cut_message_with_tool_calls_fails_closed(self):
        tool = {'type': 'tool_use', 'id': 'toolu_x', 'name': 'mcp__pythia__echo', 'input': {'value': 1}}
        records = [stream_event('message_start', message={'id': 'cut', 'usage': {'input_tokens': 2}}),
                   stream_event('content_block_start', index=0, content_block={**tool, 'input': {}}),
                   stream_event('content_block_delta', index=0, delta={'type': 'input_json_delta', 'partial_json': '{"value":1}'}),
                   snapshot([tool], ident='cut'),
                   stream_event('content_block_stop', index=0),
                   stream_event('message_stop')]
        with self.assertRaises(NativeStreamInterrupted) as caught:
            feed_all(Assembler(), records)
        self.assertEqual(caught.exception.error_code, 'native_stream_interrupted')
        self.assertIn('no tool calls may execute', str(caught.exception))

    def test_only_repeated_cuts_in_a_row_fail(self):
        self.assertEqual(MAX_CONSECUTIVE_INTERRUPTIONS, 3)
        assembler = Assembler()
        for n in range(3):
            feed_all(assembler, cut_records(f'cut-{n}'))
        feed_all(assembler, observed_records())  # a completed message resets the count
        for n in range(3, 6):
            feed_all(assembler, cut_records(f'cut-{n}'))
        with self.assertRaisesRegex(NativeStreamInterrupted, '4 messages in a row'):
            feed_all(assembler, cut_records('cut-6'))
        self.assertEqual(assembler.interruptions, 7)

    def test_a_delta_without_stop_or_usage_is_still_a_protocol_error(self):
        # Only a missing message_delta means a cut; an incomplete one is drift.
        for delta, usage in (({}, {'output_tokens': 546}), ({'stop_reason': 'end_turn'}, {})):
            records = observed_records()
            records[-2] = stream_event('message_delta', delta=delta, usage=usage)
            with self.subTest(delta=delta, usage=usage), self.assertRaises(ModelResponseError) as caught:
                feed_all(Assembler(), records)
            self.assertNotIsInstance(caught.exception, NativeStreamInterrupted)


class SystemEventTests(unittest.TestCase):
    def runtime(self):
        value = Runtime.__new__(Runtime)  # no subprocesses, threads, or sockets
        value.initialized, value.names = False, {}
        value._init_turns()
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
