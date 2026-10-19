"""Text-only cold import and exact warm-handoff comparison. No provider state in saves.

A live native process acknowledged some records (``acknowledged``): its
history. A sample whose records extend them by exactly the pending results,
then user messages, goes to that process (``handoff``); anything else starts a
new process with a cold import of every record.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from ..items import Instructions, Message, Reasoning, ToolCall, ToolResult, OpaqueCompaction
from ..model import ModelConfigurationError, ModelResponseError
from ..model_catalog import parse_json_value, thaw_json

# v2: a historical call to a tool in the current catalog is shown under the
# native name the model calls it by (Snapshot.prompt).
CODEC = 'pythia.claude-relay.text.v2'


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

    def prompt(self, names=None):
        """The cold-import input: the history as one JSON document.

        ``names`` maps a Pythia tool name to the native name the model calls it
        by. A historical call to such a tool is shown under that name, so the
        history's calls match the tool list: right after a cold import, the
        history holds the model's only examples of calls, and CLI 2.1.289
        rejects a call by the bare Pythia name. Other calls keep their names.
        ``records``, and so the warm-handoff comparison, never change.
        """
        pending = set()
        history = []
        for value in self.records:
            row = json.loads(value)
            if row['kind'] == 'call':
                pending.add(row['id'])
                row['name'] = (names or {}).get(row['name'], row['name'])
            elif row['kind'] == 'result':
                pending.discard(row['id'])
            history.append(row)
        if pending:
            raise ModelConfigurationError('Cannot cold-import unresolved historical tool calls')
        return canonical({'format': CODEC, 'history': history})


@dataclass(frozen=True)
class Delivery:
    """What a sample sends the live native process instead of a cold import.

    ``results`` maps each pending native call ID to its (output, success), and
    ``messages`` are the texts of the user messages that follow them.
    """

    results: dict
    messages: tuple


def handoff(snapshot, acknowledged, pending):
    """The ``Delivery`` for the live process, or None for a cold restart.

    The new records must be the complete results of the ``pending`` calls,
    then text-only user messages: after a tool batch, the results, and any
    steers or tools' messages that followed them (doc/claude-relay-warm-delivery-plan.md,
    Part 2); after a final answer, when nothing is pending, the next turn's
    user messages (Part 1). Anything else is a cold restart: an assistant
    message, a result after a message, a blank message, or nothing new at all,
    as ``/continue`` after a final answer. Results alone that are incomplete
    are an error, as before: an unchanged but incomplete handoff.
    """
    if snapshot.records[:len(acknowledged)] != acknowledged:
        return None
    suffix = [json.loads(row) for row in snapshot.records[len(acknowledged):]]
    count = next((index for index, row in enumerate(suffix) if row['kind'] != 'result'), len(suffix))
    results, messages = suffix[:count], suffix[count:]
    if any(row['kind'] != 'message' or row['role'] != 'user' or not row['text'].strip() for row in messages):
        return None  # CLI 2.1.289's requests carry no blank text blocks
    ids = [row['id'] for row in results]
    complete = len(ids) == len(set(ids)) and set(ids) == set(pending)
    if not messages:
        if not pending:
            return None  # nothing to deliver
        if not complete:
            raise ModelResponseError('Claude Relay handoff requires exactly the complete outstanding result set')
    elif not complete:
        return None
    return Delivery({pending[row['id']]: (row['output'], row['success']) for row in results},
                    tuple(row['text'] for row in messages))
