"""Text-only cold import and exact warm-handoff comparison. No provider state in saves."""
from __future__ import annotations

from dataclasses import dataclass
import json

from ..items import Instructions, Message, Reasoning, ToolCall, ToolResult, OpaqueCompaction
from ..model import ModelConfigurationError, ModelResponseError
from ..model_catalog import parse_json_value, thaw_json

CODEC = 'pythia.claude-relay.text.v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def record(item):
    if isinstance(item, Message):
        if item.has_media:
            raise ModelConfigurationError('Claude Relay initially supports text-only context')
        return canonical({'kind': 'message', 'role': item.role, 'text': item.content})
    if isinstance(item, Reasoning):
        if item.encrypted_content is not None:
            raise ModelConfigurationError('Claude Relay cannot import encrypted reasoning')
        # Explicit codec policy: visible reasoning text only, never provider signatures.
        return canonical({'kind': 'reasoning', 'text': item.content, 'summary': item.summary})
    if isinstance(item, ToolCall):
        try:
            args = parse_json_value(item.arguments_json)
        except ValueError:
            raise ModelConfigurationError('Historical tool arguments must be finite JSON') from None
        if not isinstance(args, dict):
            raise ModelConfigurationError('Historical tool arguments must be an object')
        return canonical({'kind': 'call', 'id': item.call_id, 'name': item.name, 'arguments': args})
    if isinstance(item, ToolResult):
        return canonical({'kind': 'result', 'id': item.call_id, 'output': item.output, 'success': item.success})
    if isinstance(item, OpaqueCompaction):
        raise ModelConfigurationError('Claude Relay cannot import opaque provider compaction')
    return None  # audit metadata/boundaries/tool snapshots are not model input


@dataclass(frozen=True)
class Snapshot:
    instructions: str | None
    records: tuple[str, ...]
    catalog: str

    @classmethod
    def build(cls, context, tools):
        instructions = None
        records = []
        for item in context.model_items():
            if isinstance(item, Instructions):
                instructions = item.text
            else:
                value = record(item)
                if value is not None:
                    records.append(value)
        catalog = canonical([{'name': t.name, 'description': t.description, 'schema': thaw_json(t.parameters)}
                             for t in tools])
        return cls(instructions, tuple(records), catalog)

    def prompt(self):
        pending = set()
        for value in self.records:
            row = json.loads(value)
            if row['kind'] == 'call':
                pending.add(row['id'])
            elif row['kind'] == 'result':
                pending.discard(row['id'])
        if pending:
            raise ModelConfigurationError('Cannot cold-import unresolved historical tool calls')
        return canonical({'format': CODEC, 'history': [json.loads(v) for v in self.records]})


def handoff(snapshot, acknowledged, pending):
    """None means cold restart. An unchanged but incomplete handoff is an error."""
    if snapshot.records[:len(acknowledged)] != acknowledged:
        return None
    suffix = [json.loads(row) for row in snapshot.records[len(acknowledged):]]
    if any(row['kind'] != 'result' for row in suffix):
        return None  # e.g. synthetic user context after the real results
    ids = [row['id'] for row in suffix]
    if len(ids) != len(set(ids)) or set(ids) != set(pending):
        raise ModelResponseError('Claude Relay handoff requires exactly the complete outstanding result set')
    return {pending[row['id']]: (row['output'], row['success']) for row in suffix}
