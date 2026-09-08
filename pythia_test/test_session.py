from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pythia.interaction import ContextCompaction
from pythia.interaction import DefaultEnvironment
from pythia.interaction import Init
from pythia.interaction import Message
from pythia.interaction import ModelContext
from pythia.interaction import ModelSample
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import OpaqueCompaction
from pythia.interaction import Reasoning
from pythia.interaction import SaveError
from pythia.interaction import TokenUsage
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import TurnMetadata
from pythia.interaction import TurnSummary
from pythia.interaction import UserInteractionBoundary
from pythia.interaction import interaction_item_from_dict
from pythia.interaction import interaction_item_to_dict
from pythia.interaction import load_interaction_save
from pythia.interaction import save_interaction_save
from pythia.interaction.demo import run_repository_summary


class SessionTests(unittest.TestCase):
    def test_atomic_save_failures_preserve_previous_log_and_remove_temporary_file(self):
        for failure in ("tempfile.NamedTemporaryFile", "os.fsync", "os.replace"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "interaction.jsonl"
                original = ModelContext((Init("old"),))
                save_interaction_save(path, original)
                old_bytes = path.read_bytes()
                with mock.patch("pythia.interaction.save." + failure,
                                side_effect=OSError("injected disk failure")):
                    with self.assertRaisesRegex(SaveError, "injected disk failure"):
                        save_interaction_save(path, ModelContext((Init("new"),)))
                self.assertEqual(path.read_bytes(), old_bytes)
                self.assertEqual(load_interaction_save(path).items, original.items)
                self.assertEqual(tuple(path.parent.glob(".interaction.jsonl.*.tmp")), ())

    def test_session_init_is_first_and_round_trips(self):
        context = ModelContext(
            (
                Init("session-test", model="initial-model"),
                Message(role="user", content="hello"),
                UserInteractionBoundary(),
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interaction.jsonl"
            save_interaction_save(path, context)
            first_record = json.loads(path.read_text().splitlines()[0])
            restored = load_interaction_save(path)

        self.assertEqual(
            first_record,
            {"type": "init", "session_id": "session-test", "model": "initial-model"},
        )
        self.assertEqual(restored.items, context.items)
        self.assertNotIn(restored.items[0], restored.model_items())

    def test_legacy_session_init_loads_and_saves_as_init(self):
        records = [
            {"type": "session_init", "session_id": "legacy-session"},
            {"type": "message", "role": "user", "text": "hello"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interaction.jsonl"
            original = "".join(json.dumps(record) + "\n" for record in records)
            path.write_text(original, encoding="utf-8")

            restored = load_interaction_save(path)

            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(
                restored.items,
                (Init("legacy-session"), Message(role="user", content="hello")),
            )
            save_interaction_save(path, restored)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8").splitlines()[0]),
                {"type": "init", "session_id": "legacy-session"},
            )
            self.assertEqual(load_interaction_save(path).items, restored.items)

    def test_legacy_session_init_validates_fields(self):
        for session_id in (None, 123, True):
            with self.subTest(session_id=session_id):
                with self.assertRaisesRegex(SaveError, "session_id must be a string"):
                    interaction_item_from_dict(
                        {"type": "session_init", "session_id": session_id}
                    )

    def test_session_init_must_be_first_and_not_compacted(self):
        with self.assertRaisesRegex(ValueError, "must be the first"):
            ModelContext(
                (
                    Message(role="user", content="hello"),
                    Init("session-test"),
                )
            )
        with self.assertRaisesRegex(ValueError, "must not contain"):
            ModelContext(
                (
                    ContextCompaction(
                        (Init("session-test"),)
                    ),
                )
            )

    def test_legacy_session_is_loaded_without_migration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interaction.jsonl"
            path.write_text(
                json.dumps(
                    {"type": "message", "role": "user", "text": "hello"}
                )
                + "\n",
                encoding="utf-8",
            )

            restored = load_interaction_save(path)

        self.assertEqual(
            restored.items,
            (Message(role="user", content="hello"),),
        )
        self.assertEqual(
            interaction_item_from_dict(
                {
                    "type": "opaque_compaction",
                    "encrypted_content": "legacy-encrypted",
                }
            ),
            OpaqueCompaction.from_responses("legacy-encrypted"),
        )

    def test_interaction_items_round_trip_through_jsonl(self):
        items = (
            Message(role="user", content="hello"),
            UserInteractionBoundary(),
            Reasoning(
                content="thinking",
                summary=("short",),
                encrypted_content="encrypted-reasoning",
                content_signature="thinking-signature",
            ),
            Message(role="assistant", content="calling tool"),
            ToolCall(
                name="lookup",
                call_id="call-1",
                arguments_json='{"q":"pythia"}',
            ),
            ModelSampleBoundary(),
            ToolResult(call_id="call-1", output="done", success=False),
            TurnMetadata(
                usage=TokenUsage(
                    input_tokens=20,
                    output_tokens=5,
                    total_tokens=25,
                    cached_input_tokens=4,
                ),
                provider_session_id="session-1",
                provider_turn_id="turn-1",
                provider_turn_state="turn-state-1",
            ),
            OpaqueCompaction.from_responses("opaque"),
            OpaqueCompaction.from_messages("summary"),
            ContextCompaction(
                (
                    Message(role="user", content="summary"),
                    UserInteractionBoundary(),
                    Message(role="assistant", content="answer"),
                    ModelSampleBoundary(),
                )
            ),
            TurnSummary(
                input_tokens_sum=20,
                output_tokens_sum=5,
                cached_input_tokens_sum=4,
                cached_input_tokens_max=4,
                non_cached_input_tokens_sum=16,
                context_tokens=25,
                sample_count=1,
                compaction_count=3,
            ),
        )
        context = ModelContext(items)
        encoded_reasoning = interaction_item_to_dict(items[2])
        self.assertEqual(
            encoded_reasoning["content_signature"],
            "thinking-signature",
        )
        self.assertNotIn("signature", encoded_reasoning)
        self.assertEqual(
            interaction_item_to_dict(items[9]),
            {
                "type": "opaque_compaction",
                "protocol": "messages",
                "payload": "summary",
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interaction.jsonl"
            save_interaction_save(path, context)
            restored = load_interaction_save(path)

            self.assertEqual(restored.items, context.items)
            self.assertEqual(len(path.read_text().splitlines()), len(items))
            self.assertEqual(
                tuple(path.glob(".interaction.jsonl.*.tmp")),
                (),
            )

    def test_item_codec_rejects_unknown_type(self):
        with self.assertRaisesRegex(ValueError, "unknown interaction item"):
            interaction_item_from_dict({"type": "not_an_item"})
        with self.assertRaisesRegex(ValueError, "cannot encode interaction"):
            interaction_item_to_dict(object())


class SessionResumeTests(unittest.TestCase):
    def test_resume_without_session_warns_and_starts_fresh(self):
        class Model:
            def __init__(self):
                self.contexts = []

            def sample(self, context, *, tools=(), options=None):
                del tools, options
                self.contexts.append(context.copy())
                return ModelSample(
                    items=(Message(role="assistant", content="fresh answer"),),
                    stop_reason="end_turn",
                )

        model = Model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interaction.jsonl"
            with DefaultEnvironment(cwd=Path(tmpdir)) as environment:
                with mock.patch("builtins.print") as print_mock:
                    summary = run_repository_summary(
                        model,
                        environment,
                        prompt=None,
                        max_samples=1,
                        save_path=path,
                        resume=True,
                    )

            restored = load_interaction_save(path)

        self.assertEqual(summary, "fresh answer")
        self.assertEqual(len(model.contexts), 1)
        self.assertIsInstance(restored.items[0], Init)
        self.assertEqual(
            tuple(
                item.content
                for item in restored.items
                if isinstance(item, Message)
            ),
            (
                "Summarize the repository in the current working directory.",
                "fresh answer",
            ),
        )
        self.assertTrue(
            any(
                call.args
                == (
                    "Warning: no existing interaction.jsonl was found; "
                    "a fresh one was created.",
                )
                and call.kwargs.get("file") is sys.stderr
                for call in print_mock.call_args_list
            )
        )

    def test_resume_replays_existing_items(self):
        context = ModelContext(
            (
                Message(role="user", content="original request"),
                UserInteractionBoundary(),
                Message(role="assistant", content="resumed answer"),
            )
        )

        class Model:
            def sample(self, context, *, tools=(), options=None):
                del context, tools, options
                raise AssertionError("completed sessions should not sample")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interaction.jsonl"
            save_interaction_save(path, context)

            with DefaultEnvironment(cwd=Path(tmpdir)) as environment:
                with mock.patch("builtins.print") as print_mock:
                    summary = run_repository_summary(
                        Model(),
                        environment,
                        prompt=None,
                        max_samples=1,
                        save_path=path,
                        resume=True,
                    )

        self.assertEqual(summary, "resumed answer")
        emitted = tuple(call.args[0] for call in print_mock.call_args_list)
        self.assertEqual(
            tuple(item.text for item in emitted),
            (
                "[user] original request",
                "[assistant] resumed answer",
            ),
        )

    def test_resume_appends_follow_up_prompt_once(self):
        context = ModelContext(
            (
                Message(role="user", content="original request"),
                UserInteractionBoundary(),
                Message(role="assistant", content="previous answer"),
            )
        )

        class Model:
            def __init__(self):
                self.contexts = []

            def sample(self, context, *, tools=(), options=None):
                del tools, options
                self.contexts.append(context.copy())
                return ModelSample(
                    items=(Message(role="assistant", content="new answer"),),
                    stop_reason="end_turn",
                )

        model = Model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interaction.jsonl"
            save_interaction_save(path, context)

            with DefaultEnvironment(cwd=Path(tmpdir)) as environment:
                with mock.patch("builtins.print"):
                    summary = run_repository_summary(
                        model,
                        environment,
                        prompt="follow-up",
                        max_samples=1,
                        save_path=path,
                        resume=True,
                    )

                resumed = load_interaction_save(path)

        self.assertEqual(summary, "new answer")
        self.assertEqual(len(model.contexts), 1)
        self.assertEqual(
            tuple(
                item.content
                for item in resumed.items
                if isinstance(item, Message)
            ),
            (
                "original request",
                "previous answer",
                "follow-up",
                "new answer",
            ),
        )

    def test_resume_executes_pending_tool_call_before_sampling(self):
        call = ToolCall(
            name="exec_command",
            call_id="pending-1",
            arguments_json=json.dumps(
                {
                    "cmd": "printf swept",
                    "yield_time_ms": 1_000,
                }
            ),
        )
        interrupted = ModelContext(
            (
                Message(role="user", content="original request"),
                UserInteractionBoundary(),
                call,
            )
        )

        class Model:
            def __init__(self):
                self.contexts = []

            def sample(self, context, *, tools=(), options=None):
                del tools, options
                self.contexts.append(context.copy())
                return ModelSample(
                    items=(Message(role="assistant", content="resumed answer"),),
                    stop_reason="end_turn",
                )

        model = Model()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "interaction.jsonl"
            save_interaction_save(path, interrupted)
            with DefaultEnvironment(cwd=root) as environment:
                with mock.patch("builtins.print"):
                    summary = run_repository_summary(
                        model,
                        environment,
                        prompt=None,
                        max_samples=1,
                        save_path=path,
                        resume=True,
                    )

                resumed = load_interaction_save(path)

        self.assertEqual(summary, "resumed answer")
        self.assertEqual(len(model.contexts), 1)
        tool_result = model.contexts[0].items[-1]
        self.assertIsInstance(tool_result, ToolResult)
        self.assertEqual(tool_result.call_id, call.call_id)
        self.assertTrue(tool_result.success)
        self.assertIn("swept", tool_result.output)
        self.assertEqual(
            resumed.items,
            (
                *interrupted.items,
                tool_result,
                Message(role="assistant", content="resumed answer"),
                TurnMetadata(usage=TokenUsage()),
                ModelSampleBoundary(),
                TurnSummary(sample_count=1),
            ),
        )


if __name__ == "__main__":
    unittest.main()
