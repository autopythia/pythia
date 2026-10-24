"""Pinned native stream-json codec (including observed CLI 2.1.289 block snapshots).

Only message_stop completes a stream. Null-stop assistant records are shadow
block snapshots, not new messages or authoritative usage. They must match their
stream, except that a tool block may show the CLI's rewrite of its input
(native_input_rewrite; any other difference there is NativeToolInputRewritten),
and must never split a tool batch or overwrite final token accounting. Only
streamed blocks are adopted, never the CLI's copies.
A message_stop without the message's final message_delta marks an interrupted
message, which is discarded (MAX_CONSECUTIVE_INTERRUPTIONS).
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import re

from ..items import Message, Reasoning, ToolCall
from ..model import ModelRefusal, ModelResponseError, ModelTransportError, RetryState
from ..model_catalog import parse_json_value
from .._provider_text import repaired_json, repaired_text
from ..usage import TokenUsage
from ._diagnostics import identifier

# v7 adds, since v6: empty streamed tool input, refusal reasons, interrupted
# messages, callable tool names in the cold import, native tool-name rejections.
# v8 adds, since v7: the CLI's rewrite of \uXXXX text in tool input, accepted in
# its snapshots and MCP calls but never executed (native_input_rewrite).
# v9 adds, since v8: kept processes (doc/claude-relay-warm-delivery-plan.md).
# Each user record written has a uuid; the CLI reports it read, taken up and
# completed (command_lifecycle), writes a system/init at each turn's start,
# echoes the uuid on the turn's first reply, and numbers its results
# (result_index). A record read while tool calls wait goes with their results.
PROFILE = 'claude-stream-json-v9'
# CLI 2.1.289 gives up on an MCP tools/call MCP_TOOL_TIMEOUT milliseconds after
# sending it, and answers the call itself (_recovery.native_timeout_texts).
# Unset, it aborts the call's HTTP request after 60 s instead. The relay sets it
# with --settings env, from ClaudeRelayEndpoint.native_tool_timeout_seconds,
# this by default. The CLI sends one call at a time and gets no answer before
# the host has run the whole batch, so the first call's clock covers the batch
# (doc/claude-relay-tool-deadline-plan.md). It was 75 s, near the CLI's own
# 60 s. An hour lets a process wait parked after a yield as long as its prompt
# cache lives, and still come after the relay's own deadline for that wait,
# 3,595 s from the start of the request that made the call, which the CLI
# sends later (doc/claude-relay-warm-yield-plan.md, 4.2). Measured through the
# relay from 20 to 240 s, and held for 91 s, 6 minutes and 58 minutes at this
# value, with the CLI's 300 s idle abort off (probe P5, October 20); re-measure
# after changing the pinned version.
DEFAULT_NATIVE_TOOL_TIMEOUT_SECONDS = 3600
MAX_RECORD = 8 * 1024 * 1024
# A message the CLI closes without its final message_delta (no stop reason, no
# final usage) is interrupted: seen on 2.1.289, untraced, after a long thinking
# block was cut short. Its block had no signature and no per-block assistant
# record, and content_block_stop and message_stop followed at once, so the CLI
# closed it itself. Nothing from it is adopted. The relay keeps reading, so the
# CLI can still complete the response, as it owns its API retries; more than
# this many interruptions in a row fail the run.
MAX_CONSECUTIVE_INTERRUPTIONS = 3
STOP_REASONS = frozenset(('end_turn', 'stop_sequence', 'tool_use', 'max_tokens',
                          'refusal', 'pause_turn', 'model_context_window_exceeded'))
USAGE_FIELDS = ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')
# CLI 2.1.289 rewrites a streamed tool input before it records or calls it
# (doc/claude-relay-tool-input-rewrite-plan.md): in each string, it decodes
# literal \uXXXX text (once the JSON is decoded: a backslash, u, four hex
# digits) into that character. A backslash that is itself escaped, an even run
# before the u, is left alone, as is all other text, such as \n or \x41.
NATIVE_INPUT_REWRITE = 'native_input_rewrite'  # its trace label
_UNICODE_ESCAPE = re.compile(r'(?<!\\)((?:\\\\)*)\\u([0-9A-Fa-f]{4})')


def normalized_stop_reason(value, *, allow_missing=False):
    if value is None and allow_missing:
        return None
    if value == 'length':
        return 'max_tokens'  # canonical Pythia incomplete-completion spelling
    if not isinstance(value, str) or value not in STOP_REASONS:
        raise ModelResponseError('Native completion lacks a supported stop reason')
    return value


class NativeRefusal(ModelRefusal):
    """A refusal in the native stream (see ModelRefusal); its reason is the
    stream's stop_details. The class name is the relay's saved failure category.
    """


class NativeStreamInterrupted(ModelTransportError):
    """The CLI ended a message's stream before its final message_delta, and did
    not then complete the response (see MAX_CONSECUTIVE_INTERRUPTIONS).

    Nothing from the interrupted message was adopted, and none of its tool
    calls ran. Sampling the same context again is safe. The class name is the
    saved failure category.
    """

    error_code = 'native_stream_interrupted'


class NativeToolInputRewritten(ModelResponseError):
    """The CLI's copy of a streamed tool call differs from the stream other than
    by native_input_rewrite: in its block snapshot or complete record (then no
    call from the message runs), or in its MCP call.

    The message names the call and where its copy first differs (input_difference):
    argument, line, column and kind of character, never the text itself. The
    class name is the saved failure category.
    """

    error_code = 'native_tool_input_rewritten'

    def __init__(self, ident, name, source, detail, *, adopted, retry_state=RetryState.NONE):
        message = (f'Native CLI rewrote tool call {identifier(ident)} ({identifier(name)}) beyond its '
                   f'known \\uXXXX decoding, in its {source}: {detail}')
        super().__init__(message if adopted else message + '; no tool calls may execute',
                         retry_state=retry_state)


@dataclass(frozen=True)
class NativeMessage:
    id: str
    blocks: tuple
    usage: TokenUsage
    stop_reason: str | None
    stop_details: dict | None = None  # a refusal's reason (NativeRefusal), as streamed
    # Native IDs of the streamed tool blocks whose input native_input_rewrite
    # changes: the CLI's copies of them, and so its history, differ from these.
    rewritten_tool_ids: tuple = ()

    def sample_items(self, generation, names):
        reason = normalized_stop_reason(self.stop_reason)
        has_tools = any(block.get('type') == 'tool_use' for block in self.blocks)
        if reason == 'refusal' and (has_tools or not any(
                block.get('type') == 'text' and isinstance(block.get('text'), str) and block['text'].strip()
                for block in self.blocks)):
            # No answer text (thinking alone is not one), or a tool batch that
            # must never run: adopt nothing, and report the refusal's reason.
            raise NativeRefusal(self.stop_details)
        if has_tools and reason != 'tool_use':
            raise ModelResponseError('Incomplete or contradictory native tool batch; no tool calls may execute')
        if not has_tools and reason == 'tool_use':
            raise ModelResponseError('Native tool stop has no tool calls')
        items, calls = [], []
        # The items are saved, so a lone surrogate in what the model streamed
        # becomes U+FFFD there (_provider_text), as in the HTTP adapters, and
        # the host runs a call with its repaired input. The calls keep the
        # model's own input, which the CLI's MCP call must match. A native ID
        # can't be repaired, since it names the call to the CLI: one with a
        # lone surrogate is malformed.
        for block in self.blocks:
            kind = block.get('type')
            if kind == 'text':
                text = block.get('text')
                if not isinstance(text, str):
                    raise ModelResponseError('Invalid native text block')
                if text:
                    items.append(Message('assistant', repaired_text(text)))
            elif kind == 'thinking':
                text = block.get('thinking')
                if not isinstance(text, str):
                    raise ModelResponseError('Invalid native thinking block')
                if text:
                    items.append(Reasoning(repaired_text(text)))
            elif kind == 'tool_use':
                native_id, name, args = block.get('id'), block.get('name'), block.get('input')
                if (not isinstance(native_id, str) or not 0 < len(native_id) <= 256
                        or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in native_id)
                        or name not in names or not isinstance(args, dict)):
                    raise ModelResponseError('Unknown/built-in or malformed native tool use')
                encoded = json.dumps(repaired_json(args), ensure_ascii=False, sort_keys=True,
                                     allow_nan=False, separators=(',', ':'))
                items.append(ToolCall(names[name], generation + ':' + native_id, encoded))
                calls.append((native_id, names[name], args))
            else:
                raise ModelResponseError('Unsupported native content block')
        if not items:
            raise ModelResponseError('Native assistant message has no supported content')
        # A refusal after answer text stays a stop reason, as for the other
        # adapters: the caller sees the text is incomplete (Pi rejects it).
        return tuple(items), calls


def token_usage(value):
    if not isinstance(value, dict):
        raise ModelResponseError('Invalid native usage')
    parsed = {}
    for key in USAGE_FIELDS:
        number = value.get(key, 0)
        if type(number) is not int or number < 0:
            raise ModelResponseError('Invalid native token count')
        parsed[key] = number
    incoming = parsed['input_tokens'] + parsed['cache_read_input_tokens'] + parsed['cache_creation_input_tokens']
    return TokenUsage(incoming, parsed['output_tokens'], incoming + parsed['output_tokens'], parsed['cache_read_input_tokens'])


def decode(line):
    if len(line) > MAX_RECORD:
        raise ModelResponseError('Native protocol record exceeds limit')
    try:
        value = parse_json_value(line.decode('utf-8'))
    except (ValueError, UnicodeError):
        raise ModelResponseError('Native stdout is not a valid finite JSON record') from None
    if not isinstance(value, dict) or not isinstance(value.get('type'), str) or not re.fullmatch('[a-z_]{1,64}', value['type']):
        raise ModelResponseError('Malformed native record')
    return value


def block_fingerprints(blocks):
    if (not isinstance(blocks, list) or not 0 < len(blocks) <= 1024
            or not all(isinstance(b, dict) for b in blocks)):
        raise ModelResponseError('Invalid native assistant content')
    return tuple(hashlib.sha256(json.dumps(b, sort_keys=True, allow_nan=False).encode()).digest()
                 for b in blocks)


def native_input_rewrite(value):
    """A streamed tool input as CLI 2.1.289 records it, in its block snapshot,
    its MCP call and presumably its own history; the model streamed ``value``.

    Traced on 6 of 2,864 calls: each copy matched this rule exactly, and no
    call showed any other difference. Seen: one backslash (decoded), two and
    four (left alone), lowercase hex, top-level strings. Uppercase hex, longer
    odd runs, nested strings and surrogate pairs (joined, as UTF-16 strings
    are) follow the same reading, unobserved; where the CLI differs, the
    comparison still fails closed. Keys and non-strings are left alone.
    Returns ``value`` itself when nothing changes.
    """
    if isinstance(value, str):
        if '\\u' not in value:
            return value
        text = _UNICODE_ESCAPE.sub(lambda match: match[1] + chr(int(match[2], 16)), value)
        if text == value:
            return value
        return text.encode('utf-16-le', 'surrogatepass').decode('utf-16-le', 'surrogatepass')
    if isinstance(value, list):
        items = [native_input_rewrite(item) for item in value]
        return value if all(new is old for new, old in zip(items, value)) else items
    if isinstance(value, dict):
        items = {key: native_input_rewrite(item) for key, item in value.items()}
        return value if all(items[key] is item for key, item in value.items()) else items
    return value


def native_view(block):
    """``block`` as the CLI records it: a tool block's input rewritten
    (native_input_rewrite). Returns ``block`` itself when nothing changes."""
    if block.get('type') == 'tool_use' and isinstance(block.get('input'), dict):
        value = native_input_rewrite(block['input'])
        if value is not block['input']:
            return {**block, 'input': value}
    return block


def shadow_matches(parts, complete, native=None):
    # 2.1.289 emits one block per assistant record, before content_block_stop.
    # Also accept an exact full snapshot. Neither form supplies batch boundaries.
    # ``native`` fingerprints the same blocks as the CLI records them (native_view).
    return any(parts == expected or (len(parts) == 1 and parts[0] in expected)
               for expected in (complete, native or complete))


def _kind(value):
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'a boolean'
    if isinstance(value, (int, float)):
        return 'a number'
    return {str: 'a string', list: 'a list', dict: 'an object'}.get(type(value), 'a value')


def _character(text, index):
    """The kind of character at ``index``, never the character itself."""
    if index >= len(text):
        return None
    char = text[index]
    if char == '\\':
        return 'a backslash'
    if char in '\r\n':
        return 'a line break'
    if ord(char) > 127:
        return 'a non-ASCII character'
    if char.isspace():
        return 'whitespace'
    return 'a letter or digit' if char.isalnum() else 'an ASCII symbol'


def _first_difference(expected, actual):
    end, start = min(len(expected), len(actual)), 0
    while start < end and expected[start:start + 4096] == actual[start:start + 4096]:
        start += 4096
    return next((i for i in range(start, min(start + 4096, end)) if expected[i] != actual[i]), end)


def _count(number, noun):
    return f'{number:,} {noun}' + ('' if number == 1 else 's')


def _difference(path, expected, actual):
    """Where JSON value ``actual`` first differs from ``expected``, or None."""
    where = f'argument {path}' if path else 'its input'
    if _kind(expected) != _kind(actual):
        return f'{where} is {_kind(actual)} where {_kind(expected)} was expected'
    if isinstance(expected, dict):
        changes = [f'{verb} ' + ', '.join(identifier(key) for key in keys[:3]) + (
                   f' and {len(keys) - 3} more' if len(keys) > 3 else '')
                   for verb, keys in (('adds', [k for k in actual if k not in expected]),
                                      ('lacks', [k for k in expected if k not in actual])) if keys]
        if changes:
            return f'{where} ' + ' and '.join(changes)
        items = ((f'{path}.{identifier(key)}' if path else identifier(key), expected[key], actual[key])
                 for key in sorted(expected))
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return (f'{where} has {_count(len(actual), "item")} where {len(expected)} '
                    f'{"was" if len(expected) == 1 else "were"} expected')
        items = ((f'{path}[{index}]', want, got) for index, (want, got) in enumerate(zip(expected, actual)))
    elif isinstance(expected, str):
        if expected == actual:
            return None
        index = _first_difference(expected, actual)
        at = f'line {actual.count(chr(10), 0, index) + 1}, column {index - actual.rfind(chr(10), 0, index)}'
        got, want = _character(actual, index), _character(expected, index)
        sizes = f'({_count(len(actual), "character")}, {len(expected):,} expected)'
        if got is None:
            return f'{where} ends at {at}, where {want} was expected {sizes}'
        if want is None:
            return f'{where} has {got} at {at}, past its expected end {sizes}'
        if got == want:
            want = 'other whitespace' if want == 'whitespace' else 'another ' + want.split(' ', 1)[1]
        return f'{where} has {got} at {at}, where {want} was expected {sizes}'
    else:
        same = json.dumps(expected) == json.dumps(actual)
        return None if same else f'{where} is a different {_kind(expected).removeprefix("a ")}'
    return next((found for found in (_difference(*item) for item in items) if found), None)


def input_difference(streamed, copy):
    """Where a CLI copy of a tool input first differs from what the relay
    accepts, the streamed input with native_input_rewrite applied: argument,
    line, column and kind of character, for a failure message. Plain-identifier
    names only; no model output is copied."""
    return _difference('', native_input_rewrite(streamed), copy) or 'its input differs'


def block_difference(streamed, copy):
    """Where a CLI copy of a streamed tool block first differs (input_difference)."""
    for key in sorted(set(streamed) | set(copy)):
        if key != 'input' and streamed.get(key) != copy.get(key):
            return f'its {identifier(key)} field differs'
    return input_difference(streamed.get('input'), copy.get('input'))


def copy_mismatch(parts, copies, blocks, accepted, source, generic):
    """The error for a CLI record of the current message that matches neither
    its streamed blocks nor their native views (fingerprints ``accepted``): a
    differing tool block is named, and where it differs; others stay generic."""
    tools = {block['id']: block for block in blocks
             if block.get('type') == 'tool_use' and isinstance(block.get('id'), str)}
    for part, copy in zip(parts, copies):
        if part in accepted:
            continue
        kind = copy.get('type')
        if kind == 'tool_use' and isinstance(copy.get('id'), str) and copy['id'] in tools:
            return NativeToolInputRewritten(copy['id'], copy.get('name'), source,
                                            block_difference(tools[copy['id']], copy), adopted=False)
        return ModelResponseError(f'{generic} (a tool block it did not stream)' if kind == 'tool_use'
                                  else f'{generic} (its {identifier(kind)} block)')
    return ModelResponseError(generic)


class Assembler:
    def __init__(self):
        self.current = None
        self.blocks = {}
        self.closed = set()
        self.usage = {}
        self.stop_reason = None
        self.stop_details = None
        self.seen = {}
        self.hints = []
        self.hint_size = 0
        self.size = 0
        self.delta_seen = False  # the current message's final message_delta arrived
        self.interrupted_ids = set()  # never adopt a later record for these
        self.interruptions = 0  # in this process, for tracing
        self.consecutive_interruptions = 0  # since the last completed message
        self.last_interruption = None

    def _interrupted(self):
        """Discard the current message: its stream ended without a final delta."""
        blocks = [self.blocks[index] for index in sorted(self.blocks)]
        kinds = tuple(block.get('type') for block in blocks)
        if 'tool_use' in kinds:
            # Its calls may already have reached the MCP endpoint, so this
            # process cannot safely go on. None is registered, so none runs.
            raise NativeStreamInterrupted('Native message with tool calls ended without its final delta; '
                                          'no tool calls may execute')
        self.interruptions += 1
        self.consecutive_interruptions += 1
        self.interrupted_ids.add(self.current)
        self.last_interruption = {
            'native_message_id': self.current, 'block_types': kinds,
            'open_blocks': len(set(self.blocks) - self.closed),
            'unsigned_thinking_blocks': sum(1 for block in blocks
                                            if block.get('type') == 'thinking' and not block.get('signature')),
            'consecutive': self.consecutive_interruptions,
        }
        if self.consecutive_interruptions > MAX_CONSECUTIVE_INTERRUPTIONS:
            raise NativeStreamInterrupted(
                f'Native stream ended {self.consecutive_interruptions} messages in a row without a final delta')
        self.current = None
        self.blocks, self.closed, self.hints = {}, set(), []
        self.usage, self.stop_reason, self.stop_details = {}, None, None
        self.size = self.hint_size = 0
        self.delta_seen = False
        return None

    def open_tool_uses(self):
        """The tool_use blocks of the message being streamed: native ID -> name."""
        if self.current is None:
            return {}
        return {block['id']: block['name'] for block in self.blocks.values()
                if block.get('type') == 'tool_use' and isinstance(block.get('id'), str)
                and isinstance(block.get('name'), str)}

    def emit(self, ident, blocks, usage, stop, *, stop_details=None, streamed=False, native=None, rewritten=()):
        """``native`` fingerprints a streamed message's blocks as the CLI records
        them (native_view), and ``rewritten`` names its rewritten tool blocks."""
        if not isinstance(ident, str) or not 0 < len(ident) <= 256 or any(ord(c) < 32 for c in ident):
            raise ModelResponseError('Native assistant ID is missing')
        if stop == 'refusal' and not blocks:
            raise NativeRefusal(stop_details)  # refused before any content: nothing to assemble
        parts = block_fingerprints(blocks)
        parsed_usage = token_usage(usage)
        reason = normalized_stop_reason(stop, allow_missing=ident in self.seen)
        if ident in self.seen:
            previous, previous_native, previous_reason, raw_usage, was_streamed = self.seen[ident]
            if was_streamed and reason is None:
                if not shadow_matches(parts, previous, previous_native):
                    # Its streamed blocks are gone: no copy_mismatch detail.
                    raise ModelResponseError('Native assistant snapshot disagrees with its stream')
                # Null-stop snapshots retain provisional usage, even when late.
                return None
            if parts not in (previous, previous_native):
                raise ModelResponseError('Inconsistent repeated native message; cannot infer batch boundaries')
            if reason is not None and reason != previous_reason:
                raise ModelResponseError('Conflicting native stop reasons for the same message')
            if any(usage[k] != raw_usage.get(k, 0) for k in USAGE_FIELDS if k in usage):
                raise ModelResponseError('Conflicting native usage for the same message')
            return None
        # A message the CLI did not stream has only its own copy to compare with.
        self.seen[ident] = (parts, native or parts, reason, {k: usage[k] for k in USAGE_FIELDS if k in usage}, streamed)
        if len(self.seen) > 4096:
            raise ModelResponseError('Native message limit exceeded')
        self.consecutive_interruptions = 0
        return NativeMessage(ident, tuple(blocks), parsed_usage, reason, stop_details, rewritten)

    def feed(self, value):
        if value['type'] == 'assistant':
            msg = value.get('message', {})
            if not isinstance(msg, dict):
                raise ModelResponseError('Invalid assistant envelope')
            if msg.get('id') in self.interrupted_ids:
                # A late block snapshot of an interrupted message is ignored, as
                # its message was; a completion for it contradicts the stream.
                if msg.get('stop_reason') is not None:
                    raise ModelResponseError('Native completion for an interrupted message')
                return None
            if self.current is not None and self.current == msg.get('id'):
                parts = block_fingerprints(msg.get('content'))
                reason = normalized_stop_reason(msg.get('stop_reason'), allow_missing=True)
                usage = msg.get('usage', {})
                token_usage(usage)
                self.hint_size += len(json.dumps(msg).encode())
                if self.hint_size > MAX_RECORD or len(self.hints) >= 2048:
                    raise ModelResponseError('Native assistant snapshots exceed aggregate limit')
                # Its blocks too, until message_stop, to name what differs (copy_mismatch).
                self.hints.append((parts, reason, {k: usage[k] for k in USAGE_FIELDS if k in usage}, msg['content']))
                return None
            if self.current is not None and msg.get('id') not in self.seen:
                raise ModelResponseError('Assistant record overlaps another native stream')
            return self.emit(msg.get('id'), msg.get('content'), msg.get('usage', {}), msg.get('stop_reason'),
                             stop_details=msg.get('stop_details'))
        if value['type'] != 'stream_event':
            return None
        event = value.get('event', {})
        if not isinstance(event, dict):
            raise ModelResponseError('Invalid stream event')
        kind = event.get('type')
        if kind == 'ping':
            # Native stream keepalive, not a message boundary or model progress.
            # Leave every assembler field untouched; runtime guards/deadlines
            # still apply, and other unknown/error event types still fail.
            return None
        if kind == 'message_start':
            if self.current is not None:
                raise ModelResponseError('Overlapping native messages')
            msg = event.get('message', {})
            if not isinstance(msg, dict):
                raise ModelResponseError('Invalid streamed message')
            self.current = msg.get('id')
            if (not isinstance(self.current, str) or not 0 < len(self.current) <= 256
                    or any(ord(c) < 32 for c in self.current) or self.current in self.seen
                    or self.current in self.interrupted_ids):
                raise ModelResponseError('Invalid streamed message ID')
            self.blocks, self.closed, self.hints = {}, set(), []
            self.size = self.hint_size = 0
            self.delta_seen = False
            token_usage(msg.get('usage', {}))
            # message_start's output count is provisional too. Require a final
            # delta (or an explicit full completion) rather than using it as a
            # silent fallback for Pi's accounting anchor.
            self.usage = {k: v for k, v in msg.get('usage', {}).items()
                          if k in USAGE_FIELDS and k != 'output_tokens'}
            self.stop_reason = None
            self.stop_details = None
        elif kind in ('content_block_start', 'content_block_delta', 'content_block_stop'):
            index = event.get('index')
            if self.current is None or type(index) is not int or not 0 <= index < 1024:
                raise ModelResponseError('Invalid streamed block index')
            if kind == 'content_block_start':
                if (index in self.blocks or not isinstance(event.get('content_block'), dict)
                        or '_json' in event['content_block']):
                    raise ModelResponseError('Duplicate/invalid block start')
                self.blocks[index] = dict(event['content_block'])
                self.size += len(json.dumps(event['content_block']).encode())
            elif index not in self.blocks or index in self.closed:
                raise ModelResponseError('Delta/stop outside an open block')
            elif kind == 'content_block_stop':
                self.closed.add(index)
                block = self.blocks[index]
                if '_json' in block:
                    raw = block.pop('_json')
                    # 2.1.289 streams a call without arguments as one empty
                    # partial_json: no JSON at all, so the block keeps its start
                    # input ({}), as when no delta came.
                    if raw:
                        try:
                            block['input'] = parse_json_value(raw)
                        except ValueError:
                            raise ModelResponseError('Incomplete or invalid native tool JSON') from None
            else:
                block, delta = self.blocks[index], event.get('delta', {})
                if not isinstance(delta, dict):
                    raise ModelResponseError('Invalid streamed delta')
                field = {'text_delta': 'text', 'thinking_delta': 'thinking', 'input_json_delta': '_json'}.get(delta.get('type'))
                if field:
                    wire_key = 'partial_json' if field == '_json' else field
                    expected_type = {'text': 'text', 'thinking': 'thinking', '_json': 'tool_use'}[field]
                    if (block.get('type') != expected_type or not isinstance(delta.get(wire_key), str)
                            or not isinstance(block.get(field, ''), str)):
                        raise ModelResponseError('Invalid streamed delta')
                    block[field] = block.get(field, '') + delta[wire_key]
                    # surrogatepass: a lone surrogate counts, and sample_items
                    # repairs it (_provider_text), instead of failing the record.
                    self.size += len(delta[wire_key].encode('utf-8', 'surrogatepass'))
                    if len(block[field]) > MAX_RECORD:
                        raise ModelResponseError('Native block exceeds limit')
                elif delta.get('type') == 'signature_delta':
                    if (block.get('type') != 'thinking' or not isinstance(delta.get('signature'), str)
                            or not isinstance(block.get('signature', ''), str)):
                        raise ModelResponseError('Invalid native thinking signature')
                    block['signature'] = block.get('signature', '') + delta.get('signature', '')
                    self.size += len(delta.get('signature', ''))
                else:
                    raise ModelResponseError('Unsupported native block delta')
            if self.size > MAX_RECORD:
                raise ModelResponseError('Native message exceeds aggregate limit')
        elif kind == 'message_delta':
            if self.current is None:
                raise ModelResponseError('Message delta without start')
            self.delta_seen = True
            token_usage(event.get('usage', {}))
            self.usage.update({k: v for k, v in event.get('usage', {}).items() if k in USAGE_FIELDS})
            delta = event.get('delta', {})
            if not isinstance(delta, dict):
                raise ModelResponseError('Invalid native message delta')
            next_reason = delta.get('stop_reason')
            if next_reason is not None:
                next_reason = normalized_stop_reason(next_reason)
                if self.stop_reason is not None and self.stop_reason != next_reason:
                    raise ModelResponseError('Conflicting streamed stop reasons')
                self.stop_reason = next_reason
            if isinstance(delta.get('stop_details'), dict):
                self.stop_details = delta['stop_details']  # informational: a refusal's reason
        elif kind == 'message_stop':
            if self.current is None:
                raise ModelResponseError('Incomplete native message at stop')
            if self.stop_reason == 'refusal' and (not self.blocks or set(self.blocks) != self.closed):
                # Refused before any content, or mid-block: report the refusal,
                # not the incomplete stream it explains.
                raise NativeRefusal(self.stop_details)
            if not self.delta_seen and all(hint[1] is None for hint in self.hints):
                # No final delta, and no complete assistant record in its
                # place: the stream was cut, whatever the blocks look like.
                return self._interrupted()
            if set(self.blocks) != self.closed or set(self.blocks) != set(range(len(self.blocks))):
                raise ModelResponseError('Incomplete native message at stop')
            blocks = [self.blocks[i] for i in range(len(self.blocks))]
            complete = block_fingerprints(blocks)
            # The CLI's records may show its rewrite of tool input. Only that
            # difference is accepted, and only the streamed blocks are adopted.
            views = [native_view(block) for block in blocks]
            rewritten = tuple(block.get('id') for block, view in zip(blocks, views) if view is not block)
            native = block_fingerprints(views) if rewritten else complete
            accepted = set(complete) | set(native)
            for parts, hint_reason, hint_usage, copies in self.hints:
                if hint_reason is None:
                    if not shadow_matches(parts, complete, native):
                        raise copy_mismatch(parts, copies, blocks, accepted, 'snapshot',
                                            'Native assistant snapshot disagrees with its stream')
                    continue
                if parts not in (complete, native):
                    raise copy_mismatch(parts, copies, blocks, accepted, 'complete record',
                                        'Streamed and completed native content disagree')
                if self.stop_reason is not None and hint_reason != self.stop_reason:
                    raise ModelResponseError('Streamed and completed stop reasons disagree')
                self.stop_reason = hint_reason
                for key, value in hint_usage.items():
                    if key in self.usage and value != self.usage[key]:
                        raise ModelResponseError('Streamed and completed usage disagree')
                    self.usage[key] = value
            if 'output_tokens' not in self.usage:
                raise ModelResponseError('Native stream lacks final output usage')
            result = self.emit(self.current, blocks, self.usage, self.stop_reason,
                               stop_details=self.stop_details, streamed=True, native=native, rewritten=rewritten)
            self.current = None
            self.blocks, self.hints = {}, []  # do not retain text/signatures after completion
            return result
        else:
            raise ModelResponseError('Unsupported native stream event')
        return None
