from __future__ import annotations

import json
import unittest

from pythia.interaction import CompactionResult
from pythia.interaction import ContextCompaction
from pythia.interaction import DisplayItem
from pythia.interaction import EnvironmentResult
from pythia.interaction import InteractionItemRenderer
from pythia.interaction import Message
from pythia.interaction import ModelSample
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import OpaqueCompaction
from pythia.interaction import Reasoning
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import UserInteraction
from pythia.interaction import UserInteractionBoundary
from pythia.interaction import render_interaction_items


ANSI_GREEN = "\x1b[32m"
ANSI_RED = "\x1b[31m"
ANSI_RESET = "\x1b[0m"


class DisplayItemTests(unittest.TestCase):
    def test_display_item_is_nominal_printable_text(self):
        item = DisplayItem("line one\nline two")

        self.assertEqual(item.text, "line one\nline two")
        self.assertEqual(str(item), "line one\nline two")
        self.assertIn("DisplayItem", repr(item))

        for invalid in (None, 1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    DisplayItem(invalid)
        for invalid in ("", "text\n", "text\r"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    DisplayItem(invalid)


class InteractionItemRendererTests(unittest.TestCase):
    def test_messages_reasoning_boundaries_and_compaction(self):
        checkpoint = ContextCompaction(
            replacement_items=(
                Message(role="user", text="summary"),
                UserInteractionBoundary(),
            )
        )
        rendered = render_interaction_items(
            (
                Message(role="system", text="system"),
                Message(role="developer", text="developer"),
                Message(role="user", text="user"),
                Message(role="assistant", text="assistant"),
                Message(role="assistant", text="   "),
                Reasoning(
                    text="hidden fallback",
                    summary=("first", " ", "second"),
                ),
                Reasoning(text="fallback"),
                Reasoning(text=" "),
                ModelSampleBoundary(),
                UserInteractionBoundary(),
                OpaqueCompaction(encrypted_content="secret"),
                checkpoint,
            )
        )

        self.assertEqual(
            tuple(str(item) for item in rendered),
            (
                "[system] system",
                "[developer] developer",
                "[user] user",
                "[assistant] assistant",
                "[reasoning] first",
                "[reasoning] second",
                "[reasoning] fallback",
                "[compaction] opaque checkpoint",
                "[compaction] context checkpoint (2 replacement items)",
            ),
        )
        self.assertTrue(all(isinstance(item, DisplayItem) for item in rendered))

    def test_producer_display_items_and_compaction_context_items(self):
        user = UserInteraction(
            items=(Message(role="user", text="hello"),),
        )
        sample = ModelSample(
            items=(
                Reasoning(text="inspect"),
                Message(role="assistant", text="answer"),
            )
        )
        checkpoint = ContextCompaction(
            replacement_items=(Message(role="user", text="summary"),)
        )
        compaction = CompactionResult(items=(checkpoint,))

        self.assertEqual(
            user.display_items(),
            (DisplayItem("[user] hello"),),
        )
        self.assertEqual(
            sample.display_items(),
            (
                DisplayItem("[reasoning] inspect"),
                DisplayItem("[assistant] answer"),
            ),
        )
        self.assertEqual(compaction.context_items(), (checkpoint,))
        self.assertEqual(
            compaction.display_items(),
            (
                DisplayItem(
                    "[compaction] context checkpoint "
                    "(1 replacement items)"
                ),
            ),
        )

    def test_shell_generic_and_malformed_tool_calls(self):
        exec_call = ToolCall(
            name="exec_command",
            call_id="exec-1",
            arguments_json=json.dumps(
                {
                    "cmd": "git status --short",
                    "yield_time_ms": 1_000,
                }
            ),
        )
        stdin_call = ToolCall(
            name="write_stdin",
            call_id="stdin-1",
            arguments_json=json.dumps(
                {
                    "session_id": 7,
                    "chars": "x\n",
                }
            ),
        )
        generic_call = ToolCall(
            name="lookup",
            call_id="lookup-1",
            arguments_json='{"b":2,"a":1}',
        )
        malformed_call = ToolCall(
            name="lookup",
            call_id="lookup-2",
            arguments_json='{"broken"',
        )

        self.assertEqual(
            render_interaction_items(
                (exec_call, stdin_call, generic_call)
            ),
            (
                DisplayItem(
                    "[tool-call] exec_command (exec-1)\n"
                    "git status --short"
                ),
                DisplayItem(
                    "[tool-call] write_stdin (stdin-1)\n"
                    "session_id=7 chars=2 bytes"
                ),
                DisplayItem("[tool-call] lookup (lookup-1)"),
            ),
        )

        renderer = InteractionItemRenderer(
            show_generic_arguments=True,
        )
        self.assertEqual(
            renderer.render_items((generic_call, malformed_call)),
            (
                DisplayItem("[tool-call] lookup (lookup-1)"),
                DisplayItem('{"a":1,"b":2}'),
                DisplayItem("[tool-call] lookup (lookup-2)"),
                DisplayItem('{"broken"'),
            ),
        )

    def test_environment_results_resolve_source_calls_without_storing_them(self):
        exec_call = ToolCall(
            name="exec_command",
            call_id="exec-1",
            arguments_json='{"cmd":"printf hello"}',
        )
        result = EnvironmentResult(
            items=(
                ToolResult(
                    call_id="exec-1",
                    output="Process exited with code 0\nOutput:\nhello\n",
                ),
            )
        )

        self.assertEqual(
            result.display_items(source_calls=(exec_call,)),
            (
                DisplayItem(
                    "[tool-ret]  exec_command (exec-1) [ok]\n"
                    "Process exited with code 0\n"
                    "Output:\n"
                    "hello"
                ),
            ),
        )
        self.assertEqual(
            result.display_items(),
            (
                DisplayItem(
                    "[tool-ret]  tool (exec-1) [ok]\n"
                    "Process exited with code 0\n"
                    "Output:\n"
                    "hello"
                ),
            ),
        )
        self.assertEqual(
            result.context_items(),
            result.items,
        )
        self.assertFalse(hasattr(result, "source_calls"))

        with self.assertRaisesRegex(ValueError, "duplicate source"):
            result.display_items(source_calls=(exec_call, exec_call))
        with self.assertRaisesRegex(TypeError, "source_calls"):
            result.display_items(source_calls=(object(),))

    def test_item_sequence_correlates_calls_and_results(self):
        call = ToolCall(
            name="lookup",
            call_id="call-1",
            arguments_json="{}",
        )
        rendered = render_interaction_items(
            (
                call,
                ModelSampleBoundary(),
                ToolResult(
                    call_id="call-1",
                    output="not found",
                    success=False,
                ),
            )
        )

        self.assertEqual(
            rendered,
            (
                DisplayItem("[tool-call] lookup (call-1)"),
                DisplayItem(
                    "[tool-ret]  lookup (call-1) [error]\nnot found"
                ),
            ),
        )

    def test_update_plan_result_uses_source_arguments(self):
        call = ToolCall(
            name="update_plan",
            call_id="plan-1",
            arguments_json=json.dumps(
                {
                    "explanation": "Use a clear sequence.",
                    "plan": [
                        {"step": "Inspect", "status": "completed"},
                        {"step": "Implement", "status": "in_progress"},
                        {"step": "Verify", "status": "pending"},
                    ],
                }
            ),
        )
        success = EnvironmentResult(
            items=(
                ToolResult(
                    call_id="plan-1",
                    output="Plan updated",
                ),
            )
        )
        failure = EnvironmentResult(
            items=(
                ToolResult(
                    call_id="plan-1",
                    output="observer failed",
                    success=False,
                ),
            )
        )

        self.assertEqual(
            success.display_items(source_calls=(call,)),
            (
                DisplayItem(
                    "[tool-ret]  update_plan (plan-1) [ok]\n"
                    "[plan] Updated plan\n"
                    "[plan] note: Use a clear sequence.\n"
                    "[plan] [x] Inspect\n"
                    "[plan] [>] Implement\n"
                    "[plan] [ ] Verify"
                ),
            ),
        )
        self.assertEqual(
            failure.display_items(source_calls=(call,)),
            (
                DisplayItem(
                    "[tool-ret]  update_plan (plan-1) [error]\n"
                    "observer failed"
                ),
            ),
        )

    def test_apply_patch_payload_uses_separate_blocks_and_optional_color(self):
        patch = "\n".join(
            (
                "*** Begin Patch",
                "*** Update File: example.txt",
                "@@",
                "-old",
                "+new",
                "*** End Patch",
            )
        )
        call = ToolCall(
            name="apply_patch",
            call_id="patch-1",
            arguments_json=json.dumps({"patch": f"{patch}\n"}),
        )

        self.assertEqual(
            render_interaction_items((call,)),
            (
                DisplayItem("[tool-call] apply_patch (patch-1)"),
                DisplayItem(patch),
            ),
        )
        self.assertEqual(
            InteractionItemRenderer(color=True).render_items((call,)),
            (
                DisplayItem("[tool-call] apply_patch (patch-1)"),
                DisplayItem(
                    "\n".join(
                        (
                            "*** Begin Patch",
                            "*** Update File: example.txt",
                            "@@",
                            f"{ANSI_RED}-old{ANSI_RESET}",
                            f"{ANSI_GREEN}+new{ANSI_RESET}",
                            "*** End Patch",
                        )
                    )
                ),
            ),
        )

    def test_git_diff_output_color_requires_git_diff_command(self):
        diff = "\n".join(
            (
                "diff --git a/example.txt b/example.txt",
                "--- a/example.txt",
                "+++ b/example.txt",
                "@@ -1 +1 @@",
                "-old",
                "+new",
            )
        )
        git_call = ToolCall(
            name="exec_command",
            call_id="git-1",
            arguments_json='{"cmd":"git diff -- example.txt"}',
        )
        cat_call = ToolCall(
            name="exec_command",
            call_id="cat-1",
            arguments_json='{"cmd":"cat example.diff"}',
        )
        renderer = InteractionItemRenderer(color=True)

        git_result = renderer.render_items(
            (ToolResult(call_id="git-1", output=diff),),
            source_calls=(git_call,),
        )
        cat_result = renderer.render_items(
            (ToolResult(call_id="cat-1", output=diff),),
            source_calls=(cat_call,),
        )

        self.assertEqual(
            git_result,
            (
                DisplayItem(
                    "[tool-ret]  exec_command (git-1) [ok]\n"
                    "diff --git a/example.txt b/example.txt\n"
                    "--- a/example.txt\n"
                    "+++ b/example.txt\n"
                    "@@ -1 +1 @@\n"
                    f"{ANSI_RED}-old{ANSI_RESET}\n"
                    f"{ANSI_GREEN}+new{ANSI_RESET}"
                ),
            ),
        )
        self.assertEqual(
            cat_result,
            (
                DisplayItem(
                    "[tool-ret]  exec_command (cat-1) [ok]\n"
                    f"{diff}"
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
