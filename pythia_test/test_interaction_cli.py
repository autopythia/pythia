from __future__ import annotations

import asyncio
from collections import deque
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import DefaultEnvironment
from pythia.interaction import DisplayItem
from pythia.interaction import Environment
from pythia.interaction import Init
from pythia.interaction import Instructions
from pythia.interaction import Message
from pythia.interaction import ModelContext
from pythia.interaction import ModelSample
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import OpaqueCompaction
from pythia.interaction import SamplingOptions
from pythia.interaction import SaveError
from pythia.interaction import Tool
from pythia.interaction import ToolCall
from pythia.interaction import ToolOutcome
from pythia.interaction import ToolResult
from pythia.interaction import ToolSpec
from pythia.interaction import TurnSummary
from pythia.interaction import UserInteractionBoundary
from pythia.interaction import cli
from pythia.interaction import demo
from pythia.interaction import load_interaction_save
from pythia.interaction import save_interaction_save
from pythia.interaction._cli_editor import Editor
from pythia.interaction._cli_editor import Layout
from pythia.interaction._cli_editor import cell_width
from pythia.interaction._cli_editor import layout_editor
from pythia.interaction._cli_terminal import PosixTerminal


def _answer(text="done"):
    return ModelSample(items=(Message(role="assistant", text=text),))


class _Terminal:
    def __init__(self, on_frame):
        self.on_frame = on_frame
        self.keys = deque()
        self.frames = []
        self.items = []
        self.closed = False
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exited = True

    def key(self, key, data=""):
        self.keys.append(SimpleNamespace(key=key, data=data))

    def submit(self, text):
        self.key("c-u")
        self.key("c-k")
        self.key("<bracketed-paste>", text)
        self.key("c-m", "\r")

    def read_keys(self):
        keys = tuple(self.keys)
        self.keys.clear()
        return keys

    def render(self, editor, status, items, prompt=":> "):
        self.frames.append((editor, status, prompt))
        self.items.extend(items)
        self.on_frame(self, editor, status)


class _Model:
    def __init__(self, path, *outcomes):
        self.path = path
        self.outcomes = deque(outcomes)
        self.calls = []
        self.checkpoints = []
        self.threads = []

    def sample(self, context, *, tools=(), options=None):
        self.calls.append((context.copy(), tuple(tools), options))
        self.checkpoints.append(load_interaction_save(self.path).items)
        self.threads.append(threading.get_ident())
        if not self.outcomes:
            raise AssertionError("unexpected sample")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome(context) if callable(outcome) else outcome


class EditorTests(unittest.TestCase):
    def test_editing_is_value_based_and_keeps_cursor_and_suffix(self):
        initial = Editor("first second", 12)
        left = initial.edit("word-left")
        self.assertEqual(left, Editor("first second", 6))
        self.assertEqual(initial, Editor("first second", 12))
        self.assertEqual(left.edit("c-w"), Editor("second", 0))
        self.assertEqual(left.edit("c-u"), Editor("second", 0))
        self.assertEqual(left.edit("c-k"), Editor("first ", 6))
        self.assertEqual(left.edit("x", "x").edit("c-h"), left)
        self.assertEqual(left.edit("right"), Editor(initial.text, 7))
        self.assertEqual(left.edit("word-right"), initial)
        self.assertEqual(initial.edit("c-a").edit("c-h"), Editor(initial.text, 0))
        self.assertEqual(initial.edit("up"), initial)

    def test_paste_and_multiline_editing_do_not_become_commands(self):
        pasted = Editor().edit("<bracketed-paste>", "/quit\r\n\x03界")
        self.assertEqual(pasted, Editor("/quit\n\x03界", 8))
        self.assertEqual(pasted.edit("c-j"), Editor("/quit\n\x03界\n", 9))

    def test_layout_wraps_and_counts_wide_and_combining_characters(self):
        self.assertEqual(
            layout_editor(Editor("abcd", 4), 7),
            Layout((":> abc", " > d"), 1, 4),
        )
        self.assertEqual(
            layout_editor(Editor("a\n界e\u0301", 5), 8),
            Layout((":> a", " > 界e\u0301"), 1, 6),
        )
        for columns in (1, 2, 3, 8, 80):
            with self.subTest(columns=columns):
                text = "a\t界\n\x1b[2J"
                layout = layout_editor(Editor(text, len(text)), columns)
                self.assertGreaterEqual(layout.cursor_column, 0)
                self.assertLess(layout.cursor_column, columns)
                self.assertNotIn("\x1b", "".join(layout.lines))
                self.assertTrue(all(
                    sum(cell_width(char) for char in line) <= max(1, columns - 1)
                    for line in layout.lines
                ))


class CLIConfigurationTests(unittest.TestCase):
    def test_working_tree_imports_help_and_legacy_target_without_posix_imports(self):
        script = '''
import builtins
import sys
from pathlib import Path

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "termios":
        raise AssertionError("help/import must not load POSIX terminal support")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from pythia.interaction import cli, demo
import pythia.auto as legacy
assert "pythia.auto.cli" not in sys.modules
project = Path(cli.__file__).resolve().parents[2] / "pyproject.toml"
assert 'autopythia = "pythia.auto:_main"' in project.read_text()
assert callable(legacy._main)
for module in (cli, demo):
    try:
        module.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("help did not exit")
'''
        result = subprocess.run([sys.executable, "-c", script],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_parser_matches_shared_demo_defaults_and_overrides(self):
        for argv in (
            [],
            ["--model-api", "codex", "--model", "gpt-6-astra", "--resume",
             "--prompt", "/quit\nA literal query", "--instructions", "",
             "--max-samples", "3", "--max-tokens", "77", "--cwd", "work",
             "--save", "chosen.jsonl"],
        ):
            demo_args = vars(demo._build_parser().parse_args(argv))
            self.assertFalse(demo_args.pop("experimental_user_message_injection"))
            self.assertEqual(vars(cli._build_parser().parse_args(argv)), demo_args)

    def test_user_message_experiment_is_demo_only(self):
        with mock.patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                cli._build_parser().parse_args(["--experimental-user-message-injection"])
        self.assertEqual(raised.exception.code, 2)

    def test_non_tty_fails_before_model_environment_or_session_effects(self):
        with mock.patch.object(cli.sys, "stdin", io.StringIO()):
            with mock.patch.object(cli, "build_model") as model:
                with mock.patch.object(cli, "DefaultEnvironment") as environment:
                    with mock.patch.object(cli, "save_interaction_save") as save:
                        with mock.patch("builtins.print"):
                            self.assertEqual(cli.main(["--prompt", "hello"]), 1)
        model.assert_not_called()
        environment.assert_not_called()
        save.assert_not_called()

    def test_module_help_works_without_a_tty(self):
        result = subprocess.run(
            [sys.executable, "-m", "pythia.interaction.cli", "--help"],
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resume", result.stdout)
        self.assertIn("--prompt", result.stdout)
        self.assertIn("--save PATH", result.stdout)

    def test_invalid_initial_options_fail_before_effects_even_with_a_tty(self):
        for argv in (["--prompt", " "], ["--max-samples", "0"], ["--max-tokens", "0"]):
            with self.subTest(argv=argv):
                terminal_stream = SimpleNamespace(isatty=lambda: True)
                with mock.patch.object(cli.sys, "stdin", terminal_stream):
                    with mock.patch.object(cli.sys, "stdout", terminal_stream):
                        with mock.patch.object(cli, "build_model") as model:
                            with mock.patch("builtins.print"):
                                self.assertEqual(cli.main(argv), 1)
                model.assert_not_called()


class TerminalRenderTests(unittest.TestCase):
    def test_content_controls_are_visible_and_decoration_is_applied_once(self):
        class Output(io.StringIO):
            def fileno(self):
                return 1

        output = Output()
        terminal = PosixTerminal(io.StringIO(), output)
        item = DisplayItem("-old\n+new\x1b[2J", is_diff=True)
        with mock.patch("os.get_terminal_size", return_value=os.terminal_size((40, 8))):
            terminal.render(Editor("draft", 5), "tool: name\nextra", (item,))
            rendered = output.getvalue()
            self.assertIn("tool: name extra", rendered)
            self.assertNotIn("\x1b[2J", rendered)
            self.assertIn("\\u001b[2J", rendered)
            self.assertEqual(rendered.count("\x1b[90m⌜"), 1)
            self.assertEqual(rendered.count("\x1b[90m⌞"), 1)
            self.assertIn("\x1b[31m-old", rendered)
            self.assertIn("\x1b[32m+new", rendered)
            terminal.render(Editor("draft", 5), "tool: name\nextra", ())
            self.assertEqual(output.getvalue(), rendered)
        self.assertEqual(item.text, "-old\n+new\x1b[2J")


class _ControllerTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "interaction.jsonl"

    async def _run(self, model, terminal, argv=(), environment=None):
        args = cli._build_parser().parse_args(argv)
        result = await asyncio.wait_for(
            cli._run(model, environment or Environment(), terminal, args, self.path),
            timeout=4,
        )
        self.assertTrue(terminal.exited)
        self.assertEqual(
            [context.items for context, _tools, _options in model.calls],
            model.checkpoints,
        )
        self.assertTrue(all(t != threading.get_ident() for t in model.threads))
        return result


class CLIControllerTests(_ControllerTestCase):
    async def test_empty_start_keeps_timer_and_never_submits_demo_default(self):
        ticks = 0

        def frame(terminal, editor, status):
            nonlocal ticks
            if status == "idle":
                ticks += 1
                if ticks == 4:
                    terminal.key("c-d")

        model = _Model(self.path)
        terminal = _Terminal(frame)
        with mock.patch.object(cli.asyncio, "sleep", wraps=asyncio.sleep) as sleep:
            self.assertEqual(await self._run(model, terminal), 0)
        self.assertGreaterEqual(sleep.await_count, 4)
        self.assertTrue(all(call.args == (1 / 128,) for call in sleep.await_args_list))
        self.assertEqual(model.calls, [])
        self.assertGreaterEqual(ticks, 4)
        self.assertTrue(all(not editor.text for editor, _, _ in terminal.frames))
        saved = load_interaction_save(self.path)
        self.assertEqual(len(saved.items), 1)
        self.assertIsInstance(saved.items[0], Init)

    async def test_full_input_queue_preserves_unaccepted_draft(self):
        state = cli._UIState(ready=True)
        for index in range(9):
            text = f"query-{index}"
            state.editor = Editor(text, len(text))
            state.handle_key("c-m", "\r")
        self.assertEqual(tuple(state.pending), tuple(f"query-{i}" for i in range(8)))
        self.assertEqual(state.editor, Editor("query-8", 7))
        state.handle_key("c-d", "")
        self.assertTrue(state.closing)
        self.assertEqual(tuple(state.pending), ())

    async def test_initial_literal_query_and_second_turn_each_submit_once(self):
        query = "/quit\nA single multiline query.\n"
        step = 0

        def frame(terminal, editor, status):
            nonlocal step
            if status == "idle":
                step += 1
                terminal.submit("second" if step == 1 else "/quit")

        terminal = _Terminal(frame)
        model = _Model(self.path, _answer("first"), _answer("second"))
        code = await self._run(
            model, terminal, ["--prompt", query, "--max-samples", "1", "--max-tokens", "77"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(terminal.frames[0][0], Editor(query, len(query)))
        self.assertEqual(len(model.calls), 2)
        saved = load_interaction_save(self.path)
        users = tuple(i for i in saved if isinstance(i, Message) and i.role == "user")
        self.assertEqual(users, (Message("user", query), Message("user", "second")))
        self.assertEqual(saved.items.count(UserInteractionBoundary()), 2)
        self.assertEqual(
            tuple(i.sample_count for i in saved if isinstance(i, TurnSummary)), (1, 2)
        )
        self.assertEqual([call[2] for call in model.calls], [SamplingOptions(max_tokens=77)] * 2)
        texts = [item.text for item in terminal.items]
        self.assertEqual(texts.count("[assistant] first"), 1)
        self.assertEqual(texts.count("[assistant] second"), 1)

    async def test_unknown_slash_command_is_not_a_query(self):
        step = 0

        def frame(terminal, editor, status):
            nonlocal step
            if status == "idle":
                step += 1
                terminal.submit("/model" if step == 1 else "/exit")

        model = _Model(self.path)
        terminal = _Terminal(frame)
        self.assertEqual(await self._run(model, terminal), 0)
        self.assertEqual(model.calls, [])
        self.assertTrue(any("Unsupported command." in i.text for i in terminal.items))

    async def test_tool_results_are_checkpointed_individually_before_follow_up(self):
        calls = (ToolCall("record", "one", "{}"), ToolCall("record", "two", "{}"))
        checkpoints = []

        def record(arguments, *, timeout_seconds=None):
            checkpoints.append(load_interaction_save(self.path).items)
            return ToolOutcome("recorded")

        environment = Environment((Tool(ToolSpec("record", "", {}), record),))
        terminal = _Terminal(lambda t, e, s: t.submit("/quit") if s == "idle" else None)
        model = _Model(self.path, ModelSample(items=calls), _answer())
        self.assertEqual(await self._run(model, terminal, ["--prompt", "tools"], environment), 0)
        self.assertEqual(len(checkpoints), 2)
        self.assertFalse(any(isinstance(i, ToolResult) for i in checkpoints[0]))
        self.assertEqual(checkpoints[1][-1], ToolResult("one", "recorded"))
        self.assertEqual(model.calls[1][0].items[-2:], (
            ToolResult("one", "recorded"), ToolResult("two", "recorded"),
        ))

    async def test_pending_resume_precedes_empty_override_and_follow_up(self):
        call = ToolCall("missing", "pending", "{}")
        original = (Init("resumed"), Instructions("old"), Message("user", "old"),
                    UserInteractionBoundary(), call, ModelSampleBoundary())
        save_interaction_save(self.path, ModelContext(original))
        terminal = _Terminal(lambda t, e, s: t.submit("/quit") if s == "idle" else None)
        model = _Model(self.path, _answer())
        self.assertEqual(await self._run(
            model, terminal, ["--resume", "--instructions", "", "--prompt", "follow-up"]
        ), 0)
        received = model.calls[0][0]
        self.assertEqual(received.items[:len(original)], original)
        result = received.items[-4]
        self.assertIsInstance(result, ToolResult)
        self.assertFalse(result.success)
        self.assertEqual(result.call_id, "pending")
        self.assertIn("was not rerun", result.output)
        self.assertEqual(received.items[-3:], (
            Instructions(""), Message("user", "follow-up"), UserInteractionBoundary(),
        ))
        self.assertEqual(received.model_items()[0], Instructions(""))

    async def test_resume_without_query_marks_pending_calls_unrecoverable_and_waits(self):
        original = (Init("saved"), Message("user", "original"),
                    UserInteractionBoundary(), ToolCall("missing", "pending", "{}"))
        save_interaction_save(self.path, ModelContext(original))
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        model = _Model(self.path)
        self.assertEqual(await self._run(model, terminal, ["--resume"]), 0)
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items[:-1], original)
        self.assertEqual(saved.items[-1].call_id, "pending")
        self.assertFalse(saved.items[-1].success)
        self.assertIn("session restart", saved.items[-1].output)
        self.assertEqual(saved.pending_tool_calls(), ())
        self.assertEqual(model.calls, [])

    async def test_instructions_only_resume_samples_without_a_new_user_message(self):
        original = (Init("saved"), Instructions("old"),
                    Message("assistant", "previous"), TurnSummary(sample_count=1))
        save_interaction_save(self.path, ModelContext(original))
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        model = _Model(self.path, _answer())
        self.assertEqual(await self._run(
            model, terminal, ["--resume", "--instructions", ""]
        ), 0)
        self.assertEqual(model.calls[0][0].items, (*original, Instructions("")))

    async def test_new_session_replaces_existing_log_and_missing_resume_keeps_query(self):
        for resume in (False, True):
            with self.subTest(resume=resume):
                if resume:
                    self.path.unlink()
                else:
                    save_interaction_save(self.path, ModelContext((Init("old"),)))
                terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
                model = _Model(self.path, _answer())
                argv = ["--prompt", "fresh"] + (["--resume"] if resume else [])
                self.assertEqual(await self._run(model, terminal, argv), 0)
                self.assertIn(
                    "[cli] Warning: exec_command runs without a sandbox; use a trusted model and workspace.",
                    [item.text for item in terminal.items],
                )
                context = model.calls[0][0]
                self.assertNotEqual(context.items[0], Init("old"))
                self.assertEqual(context.items[1:], (
                    Message("user", "fresh"), UserInteractionBoundary(),
                ))

    async def test_completed_resume_does_not_repeat_answer_or_summary(self):
        original = (Init("saved"), Message("assistant", "previous"),
                    ModelSampleBoundary(), TurnSummary(sample_count=1))
        save_interaction_save(self.path, ModelContext(original))
        terminal = _Terminal(lambda t, e, s: t.submit("/exit") if s == "idle" else None)
        model = _Model(self.path)
        self.assertEqual(await self._run(model, terminal, ["--resume"]), 0)
        self.assertEqual(load_interaction_save(self.path).items, original)
        self.assertEqual([i.text for i in terminal.items].count("[assistant] previous"), 1)
        self.assertEqual(model.calls, [])

    async def test_missing_resume_is_fresh_but_does_not_inject_default_query(self):
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        model = _Model(self.path)
        self.assertEqual(await self._run(model, terminal, ["--resume"]), 0)
        self.assertEqual(model.calls, [])
        self.assertTrue(any("no existing interaction.jsonl" in i.text for i in terminal.items))

    async def test_paused_messages_compaction_continues(self):
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        checkpoint = OpaqueCompaction.from_messages("summary")
        model = _Model(self.path, ModelSample(items=(checkpoint,), stop_reason="compaction"), _answer())
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 0)
        self.assertIn(checkpoint, model.calls[1][0].items)
        self.assertEqual(load_interaction_save(self.path).items[-1], TurnSummary(sample_count=2, compaction_count=1))

    async def test_failure_after_tool_keeps_checkpoint_and_tui_alive(self):
        called = []

        def record(arguments, *, timeout_seconds=None):
            called.append(True)
            return ToolOutcome("effect done")

        environment = Environment((Tool(ToolSpec("record", "", {}), record),))
        terminal = _Terminal(lambda t, e, s: t.key("c-c") if s == "failed" else None)
        model = _Model(self.path, ModelSample(items=(ToolCall("record", "one", "{}"),)), RuntimeError("sample failed\n"))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"], environment), 1)
        self.assertEqual(called, [True])
        self.assertEqual(load_interaction_save(self.path).items[-1], ToolResult("one", "effect done"))
        self.assertTrue(any("RuntimeError: sample failed" in i.text for i in terminal.items))

    async def test_queue_does_not_mutate_active_request(self):
        entered, release = threading.Event(), threading.Event()

        def first(context):
            entered.set()
            if not release.wait(2):
                raise AssertionError("test did not release model")
            return _answer("first")

        queued = False

        def frame(t, editor, status):
            nonlocal queued
            if entered.is_set() and not queued:
                queued = True
                t.submit("queued")
            elif "queued=1" in status:
                release.set()
            if status == "idle" and any(i.text == "[assistant] second" for i in t.items):
                t.key("c-d")

        terminal = _Terminal(frame)
        model = _Model(self.path, first, _answer("second"))
        try:
            self.assertEqual(await self._run(model, terminal, ["--prompt", "first"]), 0)
        finally:
            release.set()
        self.assertEqual(len(model.calls), 2)
        self.assertNotIn(Message("user", "queued"), model.calls[0][0].items)
        self.assertIn(Message("user", "queued"), model.calls[1][0].items)

    async def test_quit_drains_sample_without_executing_returned_tools(self):
        entered, release = threading.Event(), threading.Event()
        call = ToolCall("must_not_run", "one", "{}")

        def sample(context):
            entered.set()
            if not release.wait(2):
                raise AssertionError("test did not release model")
            return ModelSample(items=(call,))

        def frame(t, editor, status):
            if entered.is_set():
                t.key("c-c")
            if status.startswith("closing"):
                release.set()

        terminal = _Terminal(frame)
        model = _Model(self.path, sample)
        try:
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 0)
        finally:
            release.set()
        self.assertEqual(load_interaction_save(self.path).pending_tool_calls(), (call,))

    async def test_quit_during_tool_checkpoints_its_result_and_skips_remaining_calls(self):
        entered, release = threading.Event(), threading.Event()
        executions = []

        def record(arguments, *, timeout_seconds=None):
            executions.append(True)
            entered.set()
            if not release.wait(2):
                raise AssertionError("test did not release tool")
            return ToolOutcome("completed before exit")

        def frame(t, editor, status):
            if entered.is_set():
                t.key("c-d")
            if status.startswith("closing"):
                release.set()

        calls = (ToolCall("record", "one", "{}"), ToolCall("record", "two", "{}"))
        environment = Environment((Tool(ToolSpec("record", "", {}), record),))
        model = _Model(self.path, ModelSample(items=calls))
        try:
            self.assertEqual(await self._run(
                model, _Terminal(frame), ["--prompt", "hello"], environment
            ), 0)
        finally:
            release.set()
        saved = load_interaction_save(self.path)
        self.assertEqual(executions, [True])
        self.assertEqual(saved.items[-1], ToolResult("one", "completed before exit"))
        self.assertEqual(saved.pending_tool_calls(), (calls[1],))

    async def test_command_session_survives_across_user_turns(self):
        step = 0

        def frame(t, editor, status):
            nonlocal step
            if status == "idle":
                step += 1
                t.submit("continue" if step == 1 else "/quit")

        start = ToolCall(
            "exec_command", "start",
            json.dumps({
                "cmd": "read line; printf '%s' \"$line\"",
                "yield_time_ms": 1,
            }),
        )
        write = ToolCall(
            "write_stdin", "write",
            '{"session_id":1,"chars":"persistent\\n","yield_time_ms":1000}',
        )
        model = _Model(self.path, ModelSample(items=(start,)), _answer("first"),
                       ModelSample(items=(write,)), _answer("second"))
        with DefaultEnvironment(cwd=self.path.parent) as environment:
            self.assertEqual(await self._run(
                model, _Terminal(frame), ["--prompt", "start"], environment
            ), 0)
            result = model.calls[-1][0].items[-1]
            self.assertIsInstance(result, ToolResult)
            self.assertTrue(result.success)
            self.assertIn("persistent", result.output)

    async def test_failed_checkpoint_blocks_follow_up_and_new_queries(self):
        real_save = save_interaction_save

        def fail_sample_save(path, context):
            if any(isinstance(i, ModelSampleBoundary) for i in context):
                raise SaveError("disk unavailable")
            real_save(path, context)

        step = 0

        def frame(t, editor, status):
            nonlocal step
            if status == "failed":
                step += 1
                t.submit("must not run" if step == 1 else "/quit")

        terminal = _Terminal(frame)
        model = _Model(self.path, _answer())
        with mock.patch.object(cli, "save_interaction_save", side_effect=fail_sample_save):
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(load_interaction_save(self.path).items[-1], UserInteractionBoundary())


if __name__ == "__main__":
    unittest.main()
