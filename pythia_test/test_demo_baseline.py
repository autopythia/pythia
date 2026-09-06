"""Executable demo baselines for the interaction CLI port's stage 0.

These exercise the existing one-shot demo, not an unimplemented REPL. The CLI's
empty-editor default, timer, exit keys, and visual pre-fill are later-stage tests.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pythia.interaction import DisplayItem
from pythia.interaction import Instructions
from pythia.interaction import Message
from pythia.interaction import ModelContext
from pythia.interaction import ModelSample
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import SamplingOptions
from pythia.interaction import SessionInit
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import TokenUsage
from pythia.interaction import TurnMetadata
from pythia.interaction import TurnSummary
from pythia.interaction import UserInteractionBoundary
from pythia.interaction import load_interaction_session
from pythia.interaction import save_interaction_session
from pythia.interaction import demo


DEMO_ARGUMENT_DEFAULTS = {
    "model_api": "chat-completions",
    "api_url": None,
    "model": None,
    "api_key": None,
    "codex_home": None,
    "codex_auth_file": None,
    "messages_server_compaction": False,
    "messages_compaction_trigger_tokens": None,
    "messages_pause_after_compaction": False,
    "messages_compaction_instructions": None,
    "cwd": ".",
    "max_samples": None,
    "max_tokens": None,
    "request_timeout_seconds": 60.0,
    "prompt": None,
    "instructions": None,
    "resume": False,
}

COMPLETED_SESSION_ITEMS = (
    SessionInit("baseline-session"),
    Instructions("Original instructions."),
    Message(role="user", text="Original request."),
    UserInteractionBoundary(),
    Message(role="assistant", text="Previous answer."),
    TurnMetadata(
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            cached_input_tokens=4,
        ),
        provider_turn_id="previous-turn",
        provider_turn_state="previous-state",
    ),
    ModelSampleBoundary(),
    TurnSummary(
        input_tokens_sum=10,
        output_tokens_sum=2,
        cached_input_tokens_sum=4,
        cached_input_tokens_max=4,
        non_cached_input_tokens_sum=6,
        context_tokens=12,
        sample_count=1,
    ),
)

PLAN_CALL = ToolCall(
    name="update_plan",
    call_id="plan-1",
    arguments_json='{"plan":[{"step":"Inspect","status":"in_progress"}]}',
)
ANSWER = ModelSample(
    items=(Message(role="assistant", text="Done."),),
    stop_reason="end_turn",
)


class _CheckpointRecordingModel:
    def __init__(self, samples):
        self.samples = iter(samples)
        self.calls = []
        self.checkpoints = []

    def sample(self, context, *, tools=(), options=None):
        self.calls.append((context.copy(), tuple(tools), options))
        self.checkpoints.append(load_interaction_session("interaction.jsonl"))
        try:
            return next(self.samples)
        except StopIteration:
            raise AssertionError("unexpected model sample") from None


class DemoArgumentBaselineTests(unittest.TestCase):
    def test_argument_defaults(self):
        self.assertEqual(
            vars(demo._build_parser().parse_args([])),
            DEMO_ARGUMENT_DEFAULTS,
        )

    def test_explicit_arguments_preserve_empty_instructions_and_query_text(self):
        query = " /quit\nTreat this as one user query.\n"
        args = demo._build_parser().parse_args(
            [
                "--model-api", "codex",
                "--model", "gpt-6-astra",
                "--api-url", "https://proxy.example.test/codex",
                "--codex-auth-file", "auth.json",
                "--cwd", "workspace",
                "--max-samples", "2",
                "--max-tokens", "77",
                "--request-timeout-seconds", "9",
                "--instructions", "",
                "--prompt", query,
                "--resume",
            ]
        )
        self.assertEqual(
            vars(args),
            {
                **DEMO_ARGUMENT_DEFAULTS,
                "model_api": "codex",
                "model": "gpt-6-astra",
                "api_url": "https://proxy.example.test/codex",
                "codex_auth_file": "auth.json",
                "cwd": "workspace",
                "max_samples": 2,
                "max_tokens": 77,
                "request_timeout_seconds": 9.0,
                "instructions": "",
                "prompt": query,
                "resume": True,
            },
        )


class DemoStartupBaselineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.launch = self.root / "launch"
        self.workspace = self.root / "workspace"
        self.launch.mkdir()
        self.workspace.mkdir()
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.launch)
        self.path = self.launch / "interaction.jsonl"

    def _run_demo(self, argv=(), samples=(ANSWER,)):
        model = _CheckpointRecordingModel(samples)
        with mock.patch.object(demo, "_build_model", return_value=model):
            with mock.patch("builtins.print") as print_mock:
                status = demo.main(argv)
        printed = tuple(call.args[0] for call in print_mock.call_args_list)
        self.assertEqual(len(model.calls), len(model.checkpoints))
        for (context, _tools, _options), checkpoint in zip(
            model.calls, model.checkpoints
        ):
            self.assertEqual(context.items, checkpoint.items)
        return status, model, printed

    def test_one_shot_default_query_is_still_injected(self):
        status, model, _printed = self._run_demo()

        self.assertEqual(status, 0)
        self.assertEqual(len(model.calls), 1)
        context, _tools, options = model.calls[0]
        self.assertIsInstance(context.items[0], SessionInit)
        self.assertEqual(
            context.items[1:],
            (
                Message(role="user", text=demo.DEFAULT_PROMPT),
                UserInteractionBoundary(),
            ),
        )
        self.assertIsNone(options)

    def test_initial_query_is_one_item_once_across_tool_follow_up(self):
        query = "/quit\nInspect café without splitting this query.\n"
        samples = (
            ModelSample(
                items=(PLAN_CALL,),
                usage=TokenUsage(
                    input_tokens=20,
                    output_tokens=4,
                    total_tokens=24,
                    cached_input_tokens=5,
                ),
            ),
            ModelSample(
                items=ANSWER.items,
                usage=TokenUsage(
                    input_tokens=30,
                    output_tokens=6,
                    total_tokens=36,
                    cached_input_tokens=10,
                ),
            ),
        )
        status, model, printed = self._run_demo(
            ["--prompt", query, "--max-samples", "2", "--max-tokens", "77"],
            samples,
        )

        self.assertEqual(status, 0)
        self.assertEqual(len(model.calls), 2)
        for context, _tools, options in model.calls:
            self.assertEqual(
                tuple(item for item in context if isinstance(item, Message)),
                (Message(role="user", text=query),),
            )
            self.assertEqual(options, SamplingOptions(max_tokens=77))
        self.assertEqual(
            tuple(
                item.text for item in printed if isinstance(item, DisplayItem)
            ),
            (
                f"[user] {query.rstrip()}",
                "[tool-call] update_plan (plan-1)",
                "[turn] usage input=20 output=4 total=24 cached=5",
                "[tool-ret]  update_plan (plan-1) [ok]\n"
                "[plan] Updated plan\n[plan] [>] Inspect",
                "[assistant] Done.",
                "[turn] usage input=30 output=6 total=36 cached=10",
                "[turn summary] input_sum=50 output_sum=10 cached_sum=15 "
                "cached_max=10 cold_sum=35 context=36 samples=2 compactions=0",
            ),
        )
        restored = load_interaction_session(self.path)
        self.assertEqual(
            restored.items.count(Message(role="user", text=query)), 1
        )
        self.assertEqual(restored.items.count(UserInteractionBoundary()), 1)
        self.assertEqual(
            restored.items[-1],
            TurnSummary(
                input_tokens_sum=50,
                output_tokens_sum=10,
                cached_input_tokens_sum=15,
                cached_input_tokens_max=10,
                non_cached_input_tokens_sum=35,
                context_tokens=36,
                sample_count=2,
            ),
        )

    def test_fresh_start_replaces_launch_session_not_workspace_session(self):
        save_interaction_session(
            self.path, ModelContext(COMPLETED_SESSION_ITEMS)
        )
        workspace_path = self.workspace / "interaction.jsonl"
        workspace_path.write_text("workspace sentinel\n", encoding="utf-8")

        status, model, _printed = self._run_demo(
            ["--cwd", str(self.workspace), "--prompt", "Fresh query."],
            (
                ModelSample(
                    items=(
                        ToolCall(
                            name="exec_command",
                            call_id="cwd-1",
                            arguments_json='{"cmd":"pwd","yield_time_ms":1000}',
                        ),
                    ),
                ),
                ANSWER,
            ),
        )

        self.assertEqual(status, 0)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(Path.cwd(), self.launch)
        restored = load_interaction_session(self.path)
        self.assertIsInstance(restored.items[0], SessionInit)
        self.assertNotEqual(restored.items[0], COMPLETED_SESSION_ITEMS[0])
        self.assertEqual(
            model.calls[0][0].items[1:],
            (
                Message(role="user", text="Fresh query."),
                UserInteractionBoundary(),
            ),
        )
        tool_result = model.calls[1][0].items[-1]
        self.assertIsInstance(tool_result, ToolResult)
        self.assertTrue(tool_result.success)
        self.assertIn(f"\n{self.workspace}\n", tool_result.output)
        self.assertEqual(workspace_path.read_text(), "workspace sentinel\n")

    def test_completed_resume_replays_summary_without_sampling_or_appending(self):
        save_interaction_session(
            self.path, ModelContext(COMPLETED_SESSION_ITEMS)
        )
        before = self.path.read_bytes()

        status, model, printed = self._run_demo(["--resume"], samples=())

        self.assertEqual(status, 0)
        self.assertEqual(model.calls, [])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(
            tuple(
                item.text for item in printed if isinstance(item, DisplayItem)
            ),
            (
                "[instructions] Original instructions.",
                "[user] Original request.",
                "[assistant] Previous answer.",
                "[turn] usage input=10 output=2 total=12 cached=4",
                "[turn summary] input_sum=10 output_sum=2 cached_sum=4 "
                "cached_max=4 cold_sum=6 context=12 samples=1 compactions=0",
            ),
        )

    def test_resume_sweeps_tools_before_override_and_initial_query(self):
        interrupted = (
            *COMPLETED_SESSION_ITEMS[:4],
            PLAN_CALL,
            ModelSampleBoundary(),
        )
        save_interaction_session(self.path, ModelContext(interrupted))

        status, model, _printed = self._run_demo(
            ["--resume", "--instructions", "", "--prompt", "Follow-up."]
        )

        self.assertEqual(status, 0)
        self.assertEqual(len(model.calls), 1)
        expected = (
            *interrupted,
            ToolResult(call_id=PLAN_CALL.call_id, output="Plan updated"),
            Instructions(""),
            Message(role="user", text="Follow-up."),
            UserInteractionBoundary(),
        )
        self.assertEqual(model.calls[0][0].items, expected)
        self.assertEqual(model.calls[0][0].model_items()[0], Instructions(""))
        self.assertEqual(
            load_interaction_session(self.path).items[:len(expected)], expected
        )

    def test_instructions_only_resume_appends_empty_or_nonempty_override(self):
        for instructions in ("", "New instructions."):
            with self.subTest(instructions=instructions):
                save_interaction_session(
                    self.path, ModelContext(COMPLETED_SESSION_ITEMS)
                )
                status, model, _printed = self._run_demo(
                    ["--resume", "--instructions", instructions]
                )

                self.assertEqual(status, 0)
                self.assertEqual(len(model.calls), 1)
                self.assertEqual(
                    model.calls[0][0].items,
                    (*COMPLETED_SESSION_ITEMS, Instructions(instructions)),
                )

    def test_missing_resume_file_warns_and_uses_supplied_or_demo_default_query(self):
        for prompt in (None, "Explicit query."):
            with self.subTest(prompt=prompt):
                self.path.unlink(missing_ok=True)
                argv = ["--resume"]
                if prompt is not None:
                    argv.extend(("--prompt", prompt))
                status, model, printed = self._run_demo(argv)

                self.assertEqual(status, 0)
                self.assertIn(
                    "Warning: no existing interaction.jsonl was found; "
                    "a fresh one was created.",
                    printed,
                )
                self.assertEqual(
                    model.calls[0][0].items[1:],
                    (
                        Message(role="user", text=prompt or demo.DEFAULT_PROMPT),
                        UserInteractionBoundary(),
                    ),
                )

    def test_malformed_resume_file_fails_without_replacement_or_sampling(self):
        malformed = b'{"type":\n'
        self.path.write_bytes(malformed)

        status, model, printed = self._run_demo(
            ["--resume", "--prompt", "Follow-up."], samples=()
        )

        self.assertEqual(status, 1)
        self.assertEqual(model.calls, [])
        self.assertEqual(self.path.read_bytes(), malformed)
        self.assertTrue(any("invalid JSON" in str(item) for item in printed))

    def test_empty_initial_query_fails_before_replacing_session_or_sampling(self):
        save_interaction_session(
            self.path, ModelContext(COMPLETED_SESSION_ITEMS)
        )
        before = self.path.read_bytes()
        for prompt in ("", " \n\t"):
            with self.subTest(prompt=prompt):
                status, model, _printed = self._run_demo(
                    ["--prompt", prompt], samples=()
                )
                self.assertEqual(status, 1)
                self.assertEqual(model.calls, [])
                self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
