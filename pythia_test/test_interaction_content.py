from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pythia.interaction import ContextCompaction
from pythia.interaction import Instructions
from pythia.interaction import Message
from pythia.interaction import ModelContext
from pythia.interaction import Reasoning
from pythia.interaction import SaveError
from pythia.interaction import interaction_item_from_dict
from pythia.interaction import interaction_item_to_dict
from pythia.interaction import load_interaction_save
from pythia.interaction import save_interaction_save


_CONTENT_RECORDS = (
    {"type": "message", "role": "user"},
    {
        "type": "reasoning",
        "summary": ["brief"],
        "encrypted_content": "encrypted-reasoning",
        "content_signature": "thinking-signature",
    },
)


class InteractionContentTests(unittest.TestCase):
    def test_message_and_reasoning_content_remains_plain_text(self):
        for item_type, fields in ((Message, {"role": "user"}), (Reasoning, {})):
            for content in ("", " \t\n", "hello", "café\n世界"):
                with self.subTest(item_type=item_type, content=content):
                    item = item_type(content=content, **fields)
                    self.assertEqual(item.content, content)
                    self.assertFalse(hasattr(item, "text"))
            for content in (None, True, 123, [], {}, [{"type": "text", "text": "hi"}]):
                with self.subTest(item_type=item_type, content=content):
                    with self.assertRaisesRegex(TypeError, "content must be a string"):
                        item_type(content=content, **fields)

    def test_legacy_text_is_only_supported_at_the_save_boundary(self):
        for item_type, fields in ((Message, {"role": "user"}), (Reasoning, {})):
            with self.subTest(item_type=item_type):
                with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'text'"):
                    item_type(text="legacy", **fields)

    def test_instructions_keep_text(self):
        for text in ("", " \t\n", "Be concise."):
            with self.subTest(text=text):
                item = Instructions(text=text)
                record = {"type": "instructions", "text": text}
                self.assertEqual(item.text, text)
                self.assertFalse(hasattr(item, "content"))
                self.assertEqual(interaction_item_to_dict(item), record)
                self.assertEqual(interaction_item_from_dict(record), item)
        with self.assertRaisesRegex(SaveError, "instructions.text must be a string"):
            interaction_item_from_dict({"type": "instructions", "content": "wrong field"})

    def test_codec_reads_both_spellings_and_writes_only_content(self):
        for base in _CONTENT_RECORDS:
            for content in ("", " \t\n", "hello", "café\n世界"):
                canonical = {**base, "content": content}
                for fields in (
                    {"content": content},
                    {"text": content},
                    {"content": content, "text": content},
                ):
                    with self.subTest(item_type=base["type"], fields=fields):
                        item = interaction_item_from_dict({**base, **fields})
                        self.assertEqual(item.content, content)
                        self.assertEqual(interaction_item_to_dict(item), canonical)

    def test_codec_rejects_conflicting_content_and_text(self):
        for base in _CONTENT_RECORDS:
            for content, text in (("new", "old"), ("", "old"), ("new", ""), (" ", "")):
                with self.subTest(item_type=base["type"], content=content, text=text):
                    with self.assertRaisesRegex(SaveError, "content and .*text must match"):
                        interaction_item_from_dict({**base, "content": content, "text": text})

    def test_codec_rejects_missing_or_invalid_content_without_fallback(self):
        fields_to_reject = [{}]
        for value in (None, True, 123, [], {}, [{"type": "text", "text": "hi"}]):
            fields_to_reject.extend((
                {"content": value},
                {"text": value},
                {"content": value, "text": "valid legacy text"},
                {"content": "valid content", "text": value},
            ))
        for base in _CONTENT_RECORDS:
            for fields in fields_to_reject:
                with self.subTest(item_type=base["type"], fields=fields):
                    with self.assertRaisesRegex(SaveError, "(content|text) must be a string"):
                        interaction_item_from_dict({**base, **fields})

    def test_jsonl_loads_legacy_new_and_mixed_logs_and_normalizes_on_save(self):
        expected = ModelContext((
            Instructions(text="Be concise."),
            Message(role="user", content=" \t\n"),
            Reasoning(content="café\n世界", summary=("brief",)),
            ContextCompaction((
                Instructions(text="Keep this instruction."),
                Message(role="user", content="summary"),
                Reasoning(content="", encrypted_content="encrypted-reasoning"),
                Message(role="assistant", content="answer"),
            )),
        ))
        canonical = [interaction_item_to_dict(item) for item in expected.items]
        for message_field, reasoning_field in (
            ("text", "text"),
            ("content", "content"),
            ("text", "content"),
            ("content", "text"),
        ):
            with self.subTest(message_field=message_field, reasoning_field=reasoning_field):
                records = [
                    {"type": "instructions", "text": "Be concise."},
                    {"type": "message", "role": "user", message_field: " \t\n"},
                    {"type": "reasoning", reasoning_field: "café\n世界", "summary": ["brief"]},
                    {
                        "type": "context_compaction",
                        "replacement_items": [
                            {"type": "instructions", "text": "Keep this instruction."},
                            {"type": "message", "role": "user", message_field: "summary"},
                            {
                                "type": "reasoning",
                                reasoning_field: "",
                                "summary": [],
                                "encrypted_content": "encrypted-reasoning",
                            },
                            {"type": "message", "role": "assistant", message_field: "answer"},
                        ],
                    },
                ]
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / "interaction.jsonl"
                    original = "".join(json.dumps(record) + "\n" for record in records)
                    path.write_text(original, encoding="utf-8")

                    restored = load_interaction_save(path)

                    self.assertEqual(path.read_text(encoding="utf-8"), original)
                    self.assertEqual(restored.items, expected.items)
                    self.assertEqual(restored.model_items(), expected.model_items())
                    save_interaction_save(path, restored)
                    self.assertEqual(
                        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()],
                        canonical,
                    )
                    self.assertEqual(load_interaction_save(path).items, expected.items)

    def test_jsonl_reports_nested_field_conflicts_without_modifying_file(self):
        for base in _CONTENT_RECORDS:
            with self.subTest(item_type=base["type"]), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "interaction.jsonl"
                records = [
                    {"type": "instructions", "text": "Keep this instruction."},
                    {
                        "type": "context_compaction",
                        "replacement_items": [{**base, "content": "new", "text": "old"}],
                    },
                ]
                original = "".join(json.dumps(record) + "\n" for record in records)
                path.write_text(original, encoding="utf-8")
                with self.assertRaisesRegex(SaveError, "at line 2: .*content and .*text must match"):
                    load_interaction_save(path)
                self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
