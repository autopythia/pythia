from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pythia.interaction import ContextCompaction
from pythia.interaction import DefaultEnvironment
from pythia.interaction import Message
from pythia.interaction import ModelContext
from pythia.interaction import ModelSample
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import OpaqueCompaction
from pythia.interaction import Reasoning
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import TokenUsage
from pythia.interaction import TurnMetadata
from pythia.interaction import UserInteractionBoundary
from pythia.interaction import interaction_item_from_dict
from pythia.interaction import interaction_item_to_dict
from pythia.interaction import load_interaction_session
from pythia.interaction import save_interaction_session
from pythia.interaction.demo import run_repository_summary


class SessionTests(unittest.TestCase):
    def test_interaction_items_round_trip_through_jsonl(self):
        items = (
            Message(role="user", text="hello"),
            UserInteractionBoundary(),
            Reasoning(text="thinking", summary=("short",)),
            Message(role="assistant", text="calling tool"),
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
                )
            ),
            OpaqueCompaction("opaque"),
            ContextCompaction(
                (
                    Message(role="user", text="summary"),
                    UserInteractionBoundary(),
                    Message(role="assistant", text="answer"),
                    ModelSampleBoundary(),
                )
            ),
        )
        context = ModelContext(items)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interaction.jsonl"
            save_interaction_session(path, context)
            restored = load_interaction_session(path)

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
    def test_resume_replays_existing_items(self):
        context = ModelContext(
            (
                Message(role="user", text="original request"),
                UserInteractionBoundary(),
                Message(role="assistant", text="resumed answer"),
            )
        )

        class Model:
            def sample(self, context, *, tools=(), options=None):
                del context, tools, options
                raise AssertionError("completed sessions should not sample")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interaction.jsonl"
            save_interaction_session(path, context)

            with DefaultEnvironment(cwd=Path(tmpdir)) as environment:
                with mock.patch("builtins.print") as print_mock:
                    summary = run_repository_summary(
                        Model(),
                        environment,
                        prompt=None,
                        max_samples=1,
                        session_path=path,
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
                Message(role="user", text="original request"),
                UserInteractionBoundary(),
                Message(role="assistant", text="previous answer"),
            )
        )

        class Model:
            def __init__(self):
                self.contexts = []

            def sample(self, context, *, tools=(), options=None):
                del tools, options
                self.contexts.append(context.copy())
                return ModelSample(
                    items=(Message(role="assistant", text="new answer"),),
                    stop_reason="end_turn",
                )

        model = Model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "interaction.jsonl"
            save_interaction_session(path, context)

            with DefaultEnvironment(cwd=Path(tmpdir)) as environment:
                with mock.patch("builtins.print"):
                    summary = run_repository_summary(
                        model,
                        environment,
                        prompt="follow-up",
                        max_samples=1,
                        session_path=path,
                        resume=True,
                    )

                resumed = load_interaction_session(path)

        self.assertEqual(summary, "new answer")
        self.assertEqual(len(model.contexts), 1)
        self.assertEqual(
            tuple(
                item.text
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
                Message(role="user", text="original request"),
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
                    items=(Message(role="assistant", text="resumed answer"),),
                    stop_reason="end_turn",
                )

        model = Model()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "interaction.jsonl"
            save_interaction_session(path, interrupted)
            with DefaultEnvironment(cwd=root) as environment:
                with mock.patch("builtins.print"):
                    summary = run_repository_summary(
                        model,
                        environment,
                        prompt=None,
                        max_samples=1,
                        session_path=path,
                        resume=True,
                    )

                resumed = load_interaction_session(path)

        self.assertEqual(summary, "resumed answer")
        self.assertEqual(len(model.contexts), 1)
        self.assertEqual(
            tuple(type(item) for item in resumed.items[-4:]),
            (
                ToolResult,
                Message,
                TurnMetadata,
                ModelSampleBoundary,
            ),
        )
        self.assertIn("swept", resumed.items[-4].output)


if __name__ == "__main__":
    unittest.main()
