"""Tool approval's timers in the CLI (cli._CliHost.tool_approval): a call that
gets no answer is approved once tool_approval_auto_seconds have passed, unless
/hold stops the timer, and a ui call then counts down
(tool_approval_ui_countdown_seconds, whose own tests are in test_ui_tool.py).
Each approval prompt reads both from the live config. The timers read
cli._approval_clock, which these tests replace with a clock that moves only
when they move it."""

from __future__ import annotations

import asyncio
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from pythia.interaction import Environment, Message, ModelSample, Tool, ToolCall
from pythia.interaction import ToolOutcome, ToolResult, ToolSpec, cli, demo
from pythia.interaction import load_interaction_save
from pythia.interaction._cli_editor import Editor
from pythia.interaction.loop import Urgency
from pythia.interaction.media_tool import Outbox
from pythia.interaction.runtime_config import ConfigError, InteractionConfig
from pythia.interaction.runtime_config import InteractionConfigSnapshot
from pythia.interaction.runtime_config import parse_config_literal, validate_config_value
from pythia.interaction.ui_tool import UiTool, load_ui_plugin
from pythia_test.test_interaction_cli import _Model, _Terminal, _answer
from pythia_test.test_ui_tool import _OPTION as _UI_OPTION
from pythia_test.test_ui_tool import _SCREENSHOT, _FakeUi


_AUTO = "tool_approval_auto_seconds"
_AUTO_OPTION = "--tool-approval-auto-seconds"
_APPROVAL = "--require-tool-approval"
_VALUES = ("tool_approval_auto_seconds requires a whole number of seconds from 30 to 59, "
           "or null.")
_APPLIES = "tool_approval_auto_seconds applies only with --require-tool-approval."
_RECORD = ToolCall("record", "r1", "{}")


class _Clock:
    """A monotonic clock that moves only when a test moves it."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _config(*argv):
    return InteractionConfig.from_namespace(cli._build_parser().parse_args(list(argv)))


def _texts(items):
    return [item.text for item in items]


def _use_clock(case):
    """Make the approval timers read a _Clock for the test; return it."""
    clock = _Clock()
    patcher = mock.patch.object(cli, "_approval_clock", clock)
    patcher.start()
    case.addCleanup(patcher.stop)
    return clock


class SettingTests(unittest.TestCase):
    def test_the_key_takes_whole_seconds_from_30_to_59_or_null(self):
        for value in (30, 45, 59, None):
            self.assertEqual(validate_config_value(_AUTO, value), value)
        for value in (29, 60, 0, True, 45.5, "45"):
            with self.subTest(value=value), self.assertRaises(ConfigError) as caught:
                validate_config_value(_AUTO, value)
            self.assertEqual(str(caught.exception), _VALUES)
        self.assertEqual(parse_config_literal(_AUTO, " 45 "), 45)
        for text in ("null", "None"):
            self.assertIsNone(parse_config_literal(_AUTO, text))
        for text in ("29", "60", "45.5", "true", "45s", ""):
            with self.subTest(text=text), self.assertRaises(ConfigError) as caught:
                parse_config_literal(_AUTO, text)
            self.assertEqual(str(caught.exception), _VALUES)

    def test_it_is_off_by_default_and_applies_only_with_tool_approval(self):
        for argv in ((_APPROVAL,), (_APPROVAL, "exec")):
            with self.subTest(argv=argv):
                config = _config(*argv)
                self.assertIsNone(config.get(_AUTO))
                self.assertEqual(config.set(_AUTO, 45), 45)
                self.assertIsNone(config.set(_AUTO, None))  # null turns it off
        for argv in ((), (_APPROVAL, "False"), (_UI_OPTION,)):
            with self.subTest(argv=argv):
                config = _config(*argv)
                self.assertIsNone(config.get(_AUTO))
                with self.assertRaises(ConfigError) as caught:
                    config.set(_AUTO, 45)
                self.assertEqual(str(caught.exception), _APPLIES)
                self.assertIsNone(config.set(_AUTO, None))
        # Other frontends never approve on their own.
        self.assertIsNone(InteractionConfig.from_namespace(demo._build_parser().parse_args([]))
                          .get(_AUTO))
        with self.assertRaises(ConfigError):
            InteractionConfig(InteractionConfigSnapshot(tool_approval_auto_seconds=45))

    def test_the_launch_value_and_changes(self):
        config = _config(_APPROVAL, _AUTO_OPTION, "45")
        self.assertEqual(config.get(_AUTO), 45)
        self.assertEqual(config.set(_AUTO, 30), 30)
        self.assertEqual(config.reset(_AUTO).tool_approval_auto_seconds, 45)
        self.assertEqual(config.render(_AUTO, json_output=False),
                         "tool_approval_auto_seconds = 45")
        self.assertIsNone(config.set(_AUTO, None))

    def test_the_option_takes_whole_seconds_and_follows_tool_approval_in_help(self):
        parse = cli._build_parser().parse_args
        self.assertIsNone(parse([]).tool_approval_auto_seconds)
        self.assertEqual(parse([_AUTO_OPTION, "59"]).tool_approval_auto_seconds, 59)
        for value in ("29", "60", "45.5", "null", "x"):
            stderr = io.StringIO()
            with self.subTest(value=value), redirect_stderr(stderr), \
                    self.assertRaises(SystemExit):
                parse([_AUTO_OPTION, value])
            self.assertIn("argument --tool-approval-auto-seconds: expected a whole number of "
                          "seconds from 30 to 59", stderr.getvalue())
        help_text = cli._build_parser().format_help()
        positions = [help_text.index(f"\n  {option}") for option in (
            _APPROVAL, _AUTO_OPTION, "--tool-approval-ui-countdown-seconds", "--debug-trace")]
        self.assertEqual(positions, sorted(positions))

    def test_the_launch_stops_without_tool_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = _Model(root / "s.jsonl", _answer())
            stderr = io.StringIO()
            with mock.patch.object(cli, "build_model", return_value=model), \
                    redirect_stderr(stderr):
                code = cli.main(["--headless", "--cwd", str(root), "--save", str(root / "s"),
                                 "--endpoint-api", "messages", "--model", "claude-test",
                                 "--max-output-tokens", "64", _AUTO_OPTION, "45",
                                 "--prompt", "Go."])
            self.assertEqual(code, 1)
            self.assertIn(f"interaction CLI failed: {_AUTO_OPTION} applies only with "
                          "--require-tool-approval.", stderr.getvalue())
            self.assertEqual(model.calls, [])
            self.assertFalse((root / "s._private").exists())


class TimerTests(unittest.TestCase):
    def setUp(self):
        self.clock = _use_clock(self)

    def test_a_call_without_an_answer_is_approved_when_the_timer_ends(self):
        request = cli._ToolApproval(_RECORD, auto=45)
        self.assertEqual((request.stage(), request.auto_seconds_left()), ("awaiting", 45))
        self.clock.advance(44.5)
        self.assertIsNone(request.auto_approve())
        self.assertEqual((request.stage(), request.auto_seconds_left()), ("awaiting", 1))
        self.clock.advance(0.5)
        self.assertEqual(request.auto_seconds_left(), 0)
        self.assertEqual(request.auto_approve(), "approved")
        self.assertEqual((request.stage(), request.auto_seconds_left()), ("approved", None))
        self.assertTrue(request.automatic)
        self.assertIsNone(request.auto_approve())
        self.assertIsNone(request.answer(False, "too late"))
        self.assertIsNone(request.hold())

    def test_the_timer_starts_a_ui_calls_countdown(self):
        request = cli._ToolApproval(_SCREENSHOT, auto=45, countdown=5)
        self.clock.advance(45)
        self.assertEqual(request.auto_approve(), "counting")
        self.assertEqual((request.stage(), request.seconds_left()), ("counting", 5))
        self.assertTrue(request.automatic)
        self.assertEqual(request.hold(), "counting")
        self.assertEqual(request.answer(False, "not now"), "withdrawn")
        self.assertEqual((request.stage(), request.denial_reason), ("denied", "not now"))
        self.assertFalse(request.automatic)

    def test_an_answer_or_hold_stops_the_timer(self):
        answered = cli._ToolApproval(_RECORD, auto=45)
        self.assertEqual(answered.answer(True), "approved")
        self.assertFalse(answered.automatic)
        denied = cli._ToolApproval(_RECORD, auto=45)
        self.assertEqual(denied.answer(False), "denied")
        held = cli._ToolApproval(_RECORD, auto=45)
        self.assertEqual(held.hold(), "held")
        self.assertIsNone(held.auto_seconds_left())
        self.assertEqual(held.hold(), "untimed")
        self.clock.advance(100)
        for request in (answered, denied, held):
            self.assertIsNone(request.auto_approve())
        self.assertEqual([request.stage() for request in (answered, denied, held)],
                         ["approved", "denied", "awaiting"])
        self.assertEqual(held.answer(True), "approved")
        self.assertEqual(cli._ToolApproval(_RECORD).hold(), "untimed")

    def test_a_wait_ends_with_the_timer(self):
        request = cli._ToolApproval(_RECORD, auto=45)
        self.clock.advance(45)
        started = time.monotonic()
        self.assertFalse(request.wait(5))
        self.assertLess(time.monotonic() - started, 1)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.clock = _use_clock(self)

    @staticmethod
    def state(request):
        state = cli._UIState(ready=True, turn_active=True, require_tool_approval=True)
        state.pending_tool_approval = request
        return state

    @staticmethod
    def submit(state, text):
        state.editor = Editor(text, len(text))
        state.handle_key("c-m", "\r")
        return state.displays[-1].text

    def test_hold_and_the_refusals(self):
        state = self.state(cli._ToolApproval(_RECORD, auto=45))
        state.set_phase("auto-approval: record")
        self.assertEqual(self.submit(state, "later work"),
                         "[cli] Only /yes, /no or /hold is accepted while a tool call awaits "
                         "approval. Draft preserved.")
        self.assertEqual(state.editor.text, "later work")
        self.assertEqual(self.submit(state, "/hold now"), "[cli] Usage: /hold (no arguments).")
        self.assertEqual(self.submit(state, "/hold"),
                         "[cli] Tool call 'record' held: it waits for /yes or /no.")
        self.assertEqual(state.phase, "approval: record")
        self.assertEqual(self.submit(state, "/hold"),
                         "[cli] This tool call has no auto-approval to hold.")
        self.assertEqual(self.submit(state, "/config.json"),
                         "[cli] Only /yes or /no is accepted while a tool call awaits "
                         "approval. Draft preserved.")
        self.assertEqual(self.submit(state, "/no"), "[cli] Tool call 'record' not approved.")
        self.assertEqual(self.submit(state, "/hold"),
                         "[cli] This tool call already has an approval decision.")
        self.assertEqual(len(state.pending), 0)

    def test_the_countdown_cant_be_held(self):
        request = cli._ToolApproval(_SCREENSHOT, auto=45, countdown=5)
        state = self.state(request)
        self.assertEqual(self.submit(state, "/yes"),
                         "[cli] Tool call 'ui' approved; it runs in 5s. /yes runs it now, /no "
                         "[reason] withdraws it.")
        self.assertEqual(self.submit(state, "/hold"),
                         "[cli] The countdown can't be held; /yes runs the call now, /no "
                         "[reason] withdraws it.")
        self.assertEqual(request.stage(), "counting")

    def test_an_answer_after_an_automatic_approval_is_too_late(self):
        request = cli._ToolApproval(_RECORD, auto=45)
        state = self.state(request)
        self.clock.advance(45)
        self.assertEqual(request.auto_approve(), "approved")
        for text in ("/no not that", "/yes", "/hold"):
            with self.subTest(text=text):
                self.assertEqual(self.submit(state, text),
                                 "[cli] Tool call 'record' was already approved automatically.")
        self.assertIsNone(request.denial_reason)


class HostTests(unittest.IsolatedAsyncioTestCase):
    """The host's side, as a turn runs it: tool_approval on a thread of its
    own, with the session's live config."""

    def setUp(self):
        self.clock = _use_clock(self)

    def session(self, *argv, ui=None):
        """A state and a host, with tool approval and these options; the
        turn's snapshot is the session's config as it is now."""
        state = cli._UIState(ready=True, turn_active=True, require_tool_approval=True, ui=ui)
        state.config = _config(_APPROVAL, *argv)
        host = cli._CliHost(state, mock.Mock(), asyncio.get_running_loop(), steering=True,
                            config=state.config.snapshot())
        return state, host

    async def approval(self, host, state, call=_RECORD):
        """Start host.tool_approval(call) on a thread; its task, once the
        approval is published. A cleanup ends a wait that the test left, as a
        stop does: a thread that waits for an answer keeps the process from
        exiting."""
        task = asyncio.create_task(asyncio.to_thread(host.tool_approval, call))
        self.addAsyncCleanup(self._end, state, task)
        for _ in range(200):
            if state.pending_tool_approval is not None:
                return task
            await asyncio.sleep(0.01)
        raise AssertionError("the approval was not published")

    @staticmethod
    async def _end(state, task):
        state.closing = True
        await asyncio.wait((task,), timeout=2)

    @staticmethod
    async def settle():
        """Give the host's thread time to look again, and its posts to run."""
        await asyncio.sleep(0.3)

    @staticmethod
    def submit(state, text):
        state.editor = Editor(text, len(text))
        state.handle_key("c-m", "\r")

    async def test_without_an_answer_the_call_runs_when_the_timer_ends(self):
        state, host = self.session(_AUTO_OPTION, "45")
        task = await self.approval(host, state)
        self.assertEqual(state.phase, "auto-approval: record")
        self.assertEqual(_texts(state.displays)[-1],
                         "[cli] Approve tool call 'record' (r1)? /yes or /no [reason]. Without "
                         "an answer, it's approved in 45s; /hold keeps it waiting.")
        self.clock.advance(44.9)
        await self.settle()
        self.assertFalse(task.done())
        self.clock.advance(0.1)
        self.assertIsNone(await asyncio.wait_for(task, 2))
        await self.settle()
        self.assertIn("[cli] Tool call 'record' approved automatically: no answer in 45s.",
                      _texts(state.displays))
        self.assertIsNone(state.pending_tool_approval)
        self.assertEqual(state.phase, "tool: record")

    async def test_answers_before_the_timer_ends(self):
        state, host = self.session(_AUTO_OPTION, "45")
        task = await self.approval(host, state)
        self.submit(state, "/yes")
        self.assertIsNone(await asyncio.wait_for(task, 2))
        self.assertIn("[cli] Tool call 'record' approved.", _texts(state.displays))
        state, host = self.session(_AUTO_OPTION, "45")
        task = await self.approval(host, state)
        self.submit(state, "/no not safe")
        self.assertEqual(await asyncio.wait_for(task, 2),
                         ToolResult("r1", "Tool call not approved: not safe", success=False))

    async def test_hold_keeps_the_call_waiting_for_an_answer(self):
        state, host = self.session(_AUTO_OPTION, "45")
        task = await self.approval(host, state)
        self.submit(state, "/hold")
        self.assertEqual(state.phase, "approval: record")
        self.clock.advance(100)
        await self.settle()
        self.assertFalse(task.done())
        self.submit(state, "/yes")
        self.assertIsNone(await asyncio.wait_for(task, 2))
        self.assertIn("[cli] Tool call 'record' held: it waits for /yes or /no.",
                      _texts(state.displays))

    async def test_a_stop_or_a_steer_by_the_timers_end_keeps_the_call_from_running(self):
        state, host = self.session(_AUTO_OPTION, "45")
        task = await self.approval(host, state)
        state.closing = True  # as a stop sets it (_UIState.request_stop)
        self.clock.advance(45)
        self.assertEqual(await asyncio.wait_for(task, 2),
                         ToolResult("r1", cli._TOOL_APPROVAL_STOPPED_OUTPUT, success=False))
        state, host = self.session(_AUTO_OPTION, "45")
        task = await self.approval(host, state)
        state.pending.append(cli._Submission("Hurry.", Message("user", "Hurry.")))
        state.preemption.flush(Urgency.IMMEDIATE)
        self.clock.advance(45)
        self.assertEqual(await asyncio.wait_for(task, 2),
                         ToolResult("r1", cli._TOOL_APPROVAL_STEER_OUTPUT, success=False))

    async def test_a_ui_call_counts_down_after_an_automatic_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            dylib = Path(temporary) / "libpythia_ui.dylib"
            dylib.write_bytes(b"a library")
            fake = _FakeUi()
            with fake.patches(dylib):
                ui = UiTool(load_ui_plugin(api="messages", default_tools=True),
                            outbox=Outbox(api="messages"))
        state, host = self.session(_UI_OPTION, _AUTO_OPTION, "45", ui=ui)
        task = await self.approval(host, state, _SCREENSHOT)
        self.assertEqual(_texts(state.displays)[-1],
                         "[cli] Approve tool call 'ui' (u1)? /yes or /no [reason]. Without an "
                         "answer, it's approved in 45s; /hold keeps it waiting. After approval, "
                         "it waits 5s before it runs, so that you can bring another window to "
                         "the front.")
        self.clock.advance(45)
        await self.settle()
        self.assertEqual(state.phase, "countdown: ui")
        self.assertIn("[cli] Tool call 'ui' approved automatically: no answer in 45s. It runs "
                      "in 5s; /yes runs it now, /no [reason] withdraws it.",
                      _texts(state.displays))
        self.assertFalse(task.done())
        self.clock.advance(5)
        self.assertIsNone(await asyncio.wait_for(task, 2))

    async def test_each_prompt_reads_the_live_config(self):
        state, host = self.session()  # the turn started without a timer
        self.assertEqual(host._approval_timers(_RECORD), (None, None))
        state.config.set(_AUTO, 45)  # as /config sets it, during the turn
        self.assertEqual(host._approval_timers(_RECORD), (45, None))
        task = await self.approval(host, state)
        self.assertEqual(state.pending_tool_approval.auto, 45)
        self.assertEqual(state.phase, "auto-approval: record")
        self.submit(state, "/yes")
        self.assertIsNone(await asyncio.wait_for(task, 2))
        state.config.set(_AUTO, None)
        self.assertEqual(host._approval_timers(_RECORD), (None, None))

    async def test_no_timer_without_the_setting_or_for_an_exempt_call(self):
        state, host = self.session()
        task = await self.approval(host, state)
        self.assertEqual(state.phase, "approval: record")
        self.assertEqual(_texts(state.displays)[-1],
                         "[cli] Approve tool call 'record' (r1)? /yes or /no [reason].")
        self.clock.advance(1000)
        await self.settle()
        self.assertFalse(task.done())
        self.submit(state, "/yes")
        self.assertIsNone(await asyncio.wait_for(task, 2))
        state, host = self.session(_AUTO_OPTION, "45")
        state.require_tool_approval = cli._APPROVAL_EXEC
        plan = ToolCall("update_plan", "p1", "{}")
        self.assertIsNone(await asyncio.to_thread(host.tool_approval, plan))
        self.assertIsNone(state.pending_tool_approval)


class CliTests(unittest.IsolatedAsyncioTestCase):
    """The CLI with the fake terminal: the model calls record, then answers."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "interaction.jsonl"
        self.clock = _use_clock(self)
        self.executed = []  # the clock's time at each run of record

    def environment(self):
        def record(arguments, *, timeout_seconds=None):
            self.executed.append(self.clock.now)
            return ToolOutcome("ran")

        return Environment((Tool(ToolSpec("record", "", {}), record),))

    async def run_cli(self, frame, *argv):
        model = _Model(self.path, ModelSample(items=(_RECORD,)), _answer("done"))
        terminal = _Terminal(frame)
        args = cli._build_parser().parse_args(["--prompt", "Go.", _APPROVAL, *argv])
        code = await asyncio.wait_for(
            cli._run(model, self.environment(), terminal, args, self.path), timeout=8)
        return code, terminal

    def results(self):
        return [item for item in load_interaction_save(self.path).items
                if isinstance(item, ToolResult)]

    async def test_the_status_line_counts_down_and_the_call_runs_without_an_answer(self):
        seen = []

        def frame(terminal, editor, status):
            if status.startswith("auto-approval: record") and status not in seen:
                seen.append(status)
                self.clock.advance(15)
            elif status == "idle" and "[assistant] done" in _texts(terminal.items):
                terminal.key("c-d")

        code, terminal = await self.run_cli(frame, _AUTO_OPTION, "45")
        self.assertEqual(code, 0)
        self.assertEqual(seen[:3], [
            f"auto-approval: record {left}s | /yes or /no; /hold keeps it waiting"
            for left in (45, 30, 15)])
        self.assertEqual(self.results(), [ToolResult("r1", "ran")])
        self.assertEqual(len(self.executed), 1)
        self.assertGreaterEqual(self.executed[0], 1045)
        texts = _texts(terminal.items)
        self.assertIn("[cli] Tool-call approval is enabled; answer /yes or /no [reason] for each "
                      "tool call. Without an answer, a call is approved after 45s "
                      "(tool_approval_auto_seconds); /hold keeps one waiting.", texts)
        self.assertIn("[cli] Tool call 'record' approved automatically: no answer in 45s.",
                      texts)

    async def test_hold_then_yes(self):
        steps = []

        def frame(terminal, editor, status):
            if status.startswith("auto-approval: record") and not steps:
                terminal.submit("/hold")
                steps.append("hold")
            elif status.startswith("approval: record") and steps == ["hold"]:
                self.clock.advance(100)  # long past the timer's end
                steps.append(len(terminal.frames))
            elif (len(steps) == 2 and status.startswith("approval: record")
                    and len(terminal.frames) > steps[1] + 30):
                terminal.submit("/yes")
                steps.append("yes")
            elif status == "idle" and "[assistant] done" in _texts(terminal.items):
                terminal.key("c-d")

        code, terminal = await self.run_cli(frame, _AUTO_OPTION, "45")
        self.assertEqual(code, 0)
        self.assertEqual(steps[0::2], ["hold", "yes"])
        self.assertEqual(self.results(), [ToolResult("r1", "ran")])
        texts = _texts(terminal.items)
        self.assertIn("[cli] Tool call 'record' held: it waits for /yes or /no.", texts)
        self.assertNotIn("[cli] Tool call 'record' approved automatically: no answer in 45s.",
                         texts)


if __name__ == "__main__":
    unittest.main()
