"""Pinned native stream-json codec (including observed CLI 2.1.289 block snapshots).

Only message_stop completes a stream. Null-stop assistant records are shadow
block snapshots, not new messages or authoritative usage. They must match their
stream, but must never split a tool batch or overwrite final token accounting.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import re

from ..items import Message, Reasoning, ToolCall
from ..model import ModelResponseError
from ..model_catalog import parse_json_value
from ..usage import TokenUsage

PROFILE = 'claude-stream-json-v6'  # stream authority, keepalives, nonterminal API retry telemetry
# CLI 2.1.289 gives up on an MCP tools/call MCP_TOOL_TIMEOUT milliseconds after
# sending it, and answers the call itself (_recovery.native_timeout_texts).
# Unset, it aborts the call's HTTP request after 60 s instead. The relay sets it
# with --settings env, from ClaudeRelayEndpoint.native_tool_timeout_seconds,
# this by default: near the CLI's own 60 s, with room for a full 59 s command
# wait. The CLI sends one call at a time and gets no answer before the host has
# run the whole batch, so the first call's clock covers the batch. Measured
# through the relay from 20 to 240 s; re-measure after changing the pinned
# version (doc/claude-relay-tool-deadline-plan.md).
DEFAULT_NATIVE_TOOL_TIMEOUT_SECONDS = 75
MAX_RECORD = 8 * 1024 * 1024
STOP_REASONS = frozenset(('end_turn', 'stop_sequence', 'tool_use', 'max_tokens',
                          'refusal', 'pause_turn', 'model_context_window_exceeded'))
USAGE_FIELDS = ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')


def normalized_stop_reason(value, *, allow_missing=False):
    if value is None and allow_missing:
        return None
    if value == 'length':
        return 'max_tokens'  # canonical Pythia incomplete-completion spelling
    if not isinstance(value, str) or value not in STOP_REASONS:
        raise ModelResponseError('Native completion lacks a supported stop reason')
    return value


def _one_line(value, limit):
    """An untrusted native string as one bounded line, or None."""
    if not isinstance(value, str):
        return None
    text = ' '.join(''.join(c if c.isprintable() else ' ' for c in value).split())
    if len(text) > limit:
        text = text[:limit - 3] + '...'
    return text or None


class NativeRefusal(ModelResponseError):
    """A response stopped with a refusal, with nothing usable to adopt.

    Its reason is the stream's stop_details, as the API sends it (on 2.1.289:
    type "refusal", a category such as "cyber", and an explanation). Any type
    or category is reported as given, never matched against a list.
    """

    error_code = 'refusal'

    def __init__(self, details=None):
        details = details if isinstance(details, dict) else {}
        self.kind = _one_line(details.get('type'), 64)
        self.category = _one_line(details.get('category'), 64)
        self.explanation = _one_line(details.get('explanation'), 400)
        tags = [f'{name}={value}' for name, value in (
            ('type', None if self.kind == 'refusal' else self.kind), ('category', self.category)) if value]
        message = 'Request refused' + (f' ({", ".join(tags)})' if tags else '')
        if self.explanation:
            message += ': ' + self.explanation
        elif not tags:
            message += ' (no reason given)'
        super().__init__(message)


@dataclass(frozen=True)
class NativeMessage:
    id: str
    blocks: tuple
    usage: TokenUsage
    stop_reason: str | None
    stop_details: dict | None = None  # a refusal's reason (NativeRefusal), as streamed

    def sample_items(self, generation, names):
        reason = normalized_stop_reason(self.stop_reason)
        has_tools = any(block.get('type') == 'tool_use' for block in self.blocks)
        if has_tools and reason == 'refusal':
            raise NativeRefusal(self.stop_details)  # a refused tool batch never executes
        if has_tools and reason != 'tool_use':
            raise ModelResponseError('Incomplete or contradictory native tool batch; no tool calls may execute')
        if not has_tools and reason == 'tool_use':
            raise ModelResponseError('Native tool stop has no tool calls')
        items, calls = [], []
        for block in self.blocks:
            kind = block.get('type')
            if kind == 'text':
                text = block.get('text')
                if not isinstance(text, str):
                    raise ModelResponseError('Invalid native text block')
                if text:
                    items.append(Message('assistant', text))
            elif kind == 'thinking':
                text = block.get('thinking')
                if not isinstance(text, str):
                    raise ModelResponseError('Invalid native thinking block')
                if text:
                    items.append(Reasoning(text))
            elif kind == 'tool_use':
                native_id, name, args = block.get('id'), block.get('name'), block.get('input')
                if (not isinstance(native_id, str) or not 0 < len(native_id) <= 256
                        or any(ord(c) < 32 for c in native_id) or name not in names or not isinstance(args, dict)):
                    raise ModelResponseError('Unknown/built-in or malformed native tool use')
                encoded = json.dumps(args, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))
                items.append(ToolCall(names[name], generation + ':' + native_id, encoded))
                calls.append((native_id, names[name], args))
            else:
                raise ModelResponseError('Unsupported native content block')
        if not items:
            if reason == 'refusal':
                raise NativeRefusal(self.stop_details)
            raise ModelResponseError('Native assistant message has no supported content')
        # A refusal after usable text stays a stop reason, as for the other
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


def check_shadow(parts, complete):
    # 2.1.289 emits one block per assistant record, before content_block_stop.
    # Also accept an exact full snapshot. Neither form supplies batch boundaries.
    if parts != complete and not (len(parts) == 1 and parts[0] in complete):
        raise ModelResponseError('Native assistant snapshot disagrees with its stream')


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

    def emit(self, ident, blocks, usage, stop, *, stop_details=None, streamed=False):
        if not isinstance(ident, str) or not 0 < len(ident) <= 256 or any(ord(c) < 32 for c in ident):
            raise ModelResponseError('Native assistant ID is missing')
        if stop == 'refusal' and not blocks:
            raise NativeRefusal(stop_details)  # refused before any content: nothing to assemble
        parts = block_fingerprints(blocks)
        parsed_usage = token_usage(usage)
        reason = normalized_stop_reason(stop, allow_missing=ident in self.seen)
        if ident in self.seen:
            previous, previous_reason, raw_usage, was_streamed = self.seen[ident]
            if was_streamed and reason is None:
                check_shadow(parts, previous)
                # Null-stop snapshots retain provisional usage, even when late.
                return None
            if previous != parts:
                raise ModelResponseError('Inconsistent repeated native message; cannot infer batch boundaries')
            if reason is not None and reason != previous_reason:
                raise ModelResponseError('Conflicting native stop reasons for the same message')
            if any(usage[k] != raw_usage.get(k, 0) for k in USAGE_FIELDS if k in usage):
                raise ModelResponseError('Conflicting native usage for the same message')
            return None
        self.seen[ident] = (parts, reason, {k: usage[k] for k in USAGE_FIELDS if k in usage}, streamed)
        if len(self.seen) > 4096:
            raise ModelResponseError('Native message limit exceeded')
        return NativeMessage(ident, tuple(blocks), parsed_usage, reason, stop_details)

    def feed(self, value):
        if value['type'] == 'assistant':
            msg = value.get('message', {})
            if not isinstance(msg, dict):
                raise ModelResponseError('Invalid assistant envelope')
            if self.current is not None and self.current == msg.get('id'):
                parts = block_fingerprints(msg.get('content'))
                reason = normalized_stop_reason(msg.get('stop_reason'), allow_missing=True)
                usage = msg.get('usage', {})
                token_usage(usage)
                self.hint_size += len(json.dumps(msg).encode())
                if self.hint_size > MAX_RECORD or len(self.hints) >= 2048:
                    raise ModelResponseError('Native assistant snapshots exceed aggregate limit')
                self.hints.append((parts, reason, {k: usage[k] for k in USAGE_FIELDS if k in usage}))
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
                    or any(ord(c) < 32 for c in self.current) or self.current in self.seen):
                raise ModelResponseError('Invalid streamed message ID')
            self.blocks, self.closed, self.hints = {}, set(), []
            self.size = self.hint_size = 0
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
                    self.size += len(delta[wire_key].encode('utf-8'))
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
            if set(self.blocks) != self.closed or set(self.blocks) != set(range(len(self.blocks))):
                raise ModelResponseError('Incomplete native message at stop')
            blocks = [self.blocks[i] for i in range(len(self.blocks))]
            complete = block_fingerprints(blocks)
            for parts, hint_reason, hint_usage in self.hints:
                if hint_reason is None:
                    check_shadow(parts, complete)
                    continue
                if parts != complete:
                    raise ModelResponseError('Streamed and completed native content disagree')
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
                               stop_details=self.stop_details, streamed=True)
            self.current = None
            self.blocks, self.hints = {}, []  # do not retain text/signatures after completion
            return result
        else:
            raise ModelResponseError('Unsupported native stream event')
        return None
