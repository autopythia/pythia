"""The CLI with icron (--enable-icron, on by default): wake-ups, commands, and notices."""

from __future__ import annotations

import asyncio
from contextlib import redirect_stderr
from datetime import datetime
import io
import json
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest
from unittest import mock
import warnings

from pythia.interaction import Environment, Icron, IcronLimits, Init, InteractionContext, Message
from pythia.interaction import ModelSample, ModelSampleBoundary, ModelTransportError, Tool
from pythia.interaction import ToolCall, ToolOutcome, ToolResult, ToolSpec, Tools
from pythia.interaction import UserInteractionBoundary, cli, save_interaction_save
from pythia.interaction._cli_editor import Editor
from pythia.interaction.loop.icron import WAKE_HEADER, Job
from pythia.interaction.loop.supervision import Yield
from pythia_test.test_interaction_cli import _Model, _Terminal, _answer
from pythia_test.interaction_helpers import seed_cli_save


NOW = datetime(2027, 1, 15, 12, 0).timestamp()  # noon, local time


class _Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


def _until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _icron(call_id, **arguments):
    return ToolCall("icron", call_id, json.dumps(arguments))


def _calls(*calls):
    return ModelSample(items=tuple(calls))


def _has(terminal, text):
    return any(text in item.text for item in terminal.items)


def _texts(terminal):
    return [item.text for item in terminal.items]


def _users(items):
    return [item.content_text for item in items
            if isinstance(item, Message) and item.role == "user"]


def _blocking_probe():
    entered, release = threading.Event(), threading.Event()

    def probe(arguments, *, timeout_seconds=None):
        entered.set()
        if not release.wait(4):
            raise AssertionError("the test did not release the tool")
        return ToolOutcome("probed")

    return Tool(ToolSpec("probe", "Probe.", {"type": "object"}), probe), entered, release


def _stepping_probe():
    """A probe whose every call waits until the test releases it: its calls,
    and the semaphore that releases one."""
    calls, releases = [], threading.Semaphore(0)

    def probe(arguments, *, timeout_seconds=None):
        calls.append(arguments)
        if not releases.acquire(timeout=4):
            raise AssertionError("the test did not release the tool")
        return ToolOutcome("probed")

    return Tool(ToolSpec("probe", "Probe.", {"type": "object"}), probe), calls, releases


class _IcronControllerCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "interaction.jsonl"
        self.clock = _Clock()
        self.icron = Icron(clock=self.clock)
        self.addCleanup(self.icron.close, 1.0)

    def advance(self, seconds):
        self.clock.now += seconds
        self.icron._poke()

    async def _run(self, model, terminal, argv=(), tools=(), *, icron=True, path=None,
                   icron_tool=True):
        args = cli._build_parser().parse_args(list(argv))
        icron = self.icron if icron else None
        extra = () if icron is None or not icron_tool else (icron.tool(),)
        environment = Environment((*tools, *extra))
        result = await asyncio.wait_for(
            cli._run(model, environment, terminal, args, path or self.path, icron=icron),
            timeout=8)
        self.assertTrue(terminal.exited)
        return result


def _once(steps):
    """A frame callback that runs each (condition, action) step once, in order."""
    state = {"step": 0}

    def frame(terminal, editor, status):
        if state["step"] < len(steps):
            condition, action = steps[state["step"]]
            if condition(terminal, status):
                state["step"] += 1
                action(terminal)

    frame.state = state
    return frame


def _idle_with(text):
    return lambda terminal, status: _phase(status) == "idle" and _has(terminal, text)


def _phase(status):
    """The status line's phase, without its counts and icron's part."""
    return status.split(" | ")[0]


def _exit(terminal):
    terminal.key("c-d")


class LaunchTests(unittest.TestCase):
    def test_icron_is_on_by_default_unless_the_default_tools_are_off(self):
        cases = [  # (options, whether the model gets icron's tool)
            ([], True),
            (["--enable-icron"], True),
            (["--enable-icron", "False"], False),
            (["--enable-icron=False"], False),
            # Its name while it was experimental is an alias, with no warning.
            (["--enable-experimental-icron"], True),
            (["--enable-experimental-icron", "False"], False),
            (["--enable-experimental-icron=False", "--enable-icron"], True),
            # Like exec_command, icron's exec runs the model's code: without the
            # default tools, a session gets icron only when it asks.
            (["--enable-default-tools", "False"], False),
            (["--enable-default-tools", "False", "--enable-icron"], True),
            (["--enable-default-tools", "False", "--enable-experimental-icron"], True),
        ]
        for options, expected in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                seen = {}

                async def headless(model, environment, args, **kwargs):
                    seen["icron"] = kwargs["icron"]
                    seen["tools"] = [spec.name for spec in environment.tool_specs]
                    return 0

                with mock.patch.object(cli, "build_model", return_value=object()), \
                        mock.patch.object(cli, "_run_headless", headless), \
                        warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    code = cli.main(["--headless", "--prompt", "hello", "--cwd", directory,
                                     "--save", str(Path(directory) / "s.jsonl"), *options])
                self.assertEqual(code, 0)
                self.assertEqual((seen["icron"] is not None, "icron" in seen["tools"]),
                                 (expected, expected))
                self.assertEqual([str(item.message) for item in caught
                                  if "icron" in str(item.message)], [])

    def test_icron_timing_still_parses_and_help_hides_it(self):
        parser = cli._build_parser()
        for value in ("turn", "clock"):
            self.assertEqual(parser.parse_args(["--icron-timing", value]).icron_timing, value)
        self.assertFalse(hasattr(parser.parse_args([]), "icron_timing"))
        self.assertNotIn("--icron-timing", parser.format_help())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--icron-timing", "sometimes"])

    def test_help_lists_the_option_and_not_its_alias(self):
        text = cli._build_parser().format_help()
        self.assertIn("--enable-icron [{False,True}]", text)
        self.assertNotIn("--enable-experimental-icron [", text)
        flat = " ".join(re.sub(r"-\n\s+", "-", text).split())  # argparse wraps at hyphens
        self.assertIn("--enable-experimental-icron, its name while experimental, is an alias",
                      flat)
        # /py is off (_PY_ENABLED), and /icron is /cron's alias.
        self.assertIn("unsandboxed; /cron (or /icron) schedules your messages and lists jobs. "
                      "Jobs and Python state", flat)
        self.assertNotIn("/py", flat)


class RepeatTests(_IcronControllerCase):
    async def test_a_running_turn_takes_a_repeat_s_message_at_each_firing(self):
        probe, calls, releases = _stepping_probe()
        model = _Model(self.path,
                       _calls(_icron("c1", call="schedule('Check the log.', every=60)"),
                              ToolCall("probe", "p1", "{}")),
                       _calls(ToolCall("probe", "p2", "{}")),
                       _answer("done"))
        start = self.clock.now

        def due():
            return next(job.due for job in self.icron.jobs() if job.message == "Check the log.")

        terminal = _Terminal(_once([
            (lambda t, s: len(calls) == 1, lambda t: self.advance(60)),
            (lambda t, s: self.icron.undelivered() == 1, lambda t: releases.release()),
            (lambda t, s: len(calls) == 2, lambda t: self.advance(60)),
            # The job fired again: its new message waits for the next interrupt point.
            (lambda t, s: self.icron.undelivered() == 1 and due() == start + 180,
             lambda t: releases.release()),
            (_idle_with("[assistant] done"), _exit)]))
        try:
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"], (probe,)), 0)
        finally:
            releases.release()
            releases.release()
        self.assertEqual(len(model.calls), 3)
        items = model.calls[2][0].items
        self.assertEqual(sum(isinstance(item, UserInteractionBoundary) for item in items), 1)
        for index in (1, 2):  # one turn, a message at each firing
            message = _users(model.calls[index][0].items)[-1]
            self.assertTrue(message.startswith(WAKE_HEADER)
                            and message.endswith("):\nCheck the log."), message)
        (notice,) = [text for text in _texts(terminal) if text.startswith("[cli] icron: main")]
        self.assertTrue(notice.endswith(" and /auto hands main a task and goes away. Jobs and "
                                        "Python state live only in this process."), notice)


class TimingOptionTests(_IcronControllerCase):
    async def test_icron_timing_is_read_and_ignored_with_a_note(self):
        # A restart reruns its command line, so an option that went is still read.
        model = _Model(self.path, _answer("done"))
        terminal = _Terminal(_once([(_idle_with("[assistant] done"), _exit)]))
        self.assertEqual(await self._run(model, terminal,
                                         ["--prompt", "hello", "--icron-timing", "turn"]), 0)
        self.assertIn("[cli] Note: --icron-timing is gone, and ignored: repeating messages reach "
                      "a running turn at each firing.", _texts(terminal))


class DisplayContractTests(_IcronControllerCase):
    async def test_the_display_is_today_s_until_icron_is_used(self):
        probe = Tool(ToolSpec("probe", "Probe.", {"type": "object"}),
                     lambda arguments, **kwargs: ToolOutcome("probed"))

        async def session(icron, path):
            model = _Model(path, _calls(ToolCall("probe", "p1", "{}")), _answer("done"))
            terminal = _Terminal(_once([(_idle_with("[assistant] done"), _exit)]))
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"], (probe,),
                                             icron=icron, path=path), 0)
            return terminal

        def normal(text, path):
            # The save's private directory is named after its prefix.
            text = text.replace(str(path), "<save>").replace(str(path.with_suffix("")), "<prefix>")
            text = text.replace(path.name, "<name>")
            return re.sub(r"\d+(\.\d+)?", "N", text)

        plain_path, icron_path = self.directory / "plain.jsonl", self.directory / "icron.jsonl"
        plain = await session(False, plain_path)
        with_icron = await session(True, icron_path)
        expected = [normal(text, plain_path) for text in _texts(plain)]
        shown = [normal(text, icron_path).replace("[tools] probe, icron", "[tools] probe")
                 for text in _texts(with_icron)
                 if not text.startswith("[cli] icron: ")]
        self.assertEqual(shown, expected)
        self.assertEqual(len(_texts(with_icron)), len(_texts(plain)) + 1)
        self.assertIn("[cli] icron: main can schedule work and run Python in "
                      "this process, unsandboxed, with the icron tool; /cron (or /icron) "
                      "schedules your messages and lists jobs, /afk tells main you're away, "
                      "/autoafk does it by itself, and /auto hands main a task and goes away. "
                      "Jobs and Python state live only in this process.",
                      _texts(with_icron))
        self.assertFalse(any("icron" in status for _editor, status, _prompt in with_icron.frames))
        self.assertIn("idle", [status for _editor, status, _prompt in with_icron.frames])


class YieldNoticeTests(unittest.TestCase):
    def test_the_notice_says_what_resumes_main(self):
        def job(kind, state="scheduled", **fields):
            return Job("k3f9-1", "main", kind, state, created=NOW, **fields)

        later = NOW + 600
        self.assertEqual(cli._yield_notice(job("message", due=later, message="Check CI."), NOW),
                         "main yielded to icron; job k3f9-1 resumes it at 12:10:00 (in 10m), "
                         "unless you write first.")
        # Code resumes main with its message or its end, not by firing.
        self.assertEqual(cli._yield_notice(job("code", due=later, code="print(1)", every=600.0),
                                           NOW),
                         "main yielded to icron; it waits for job k3f9-1's code, next run at "
                         "12:10:00 (in 10m), to message it or end, unless you write first.")
        self.assertEqual(cli._yield_notice(job("exec", "running", code="print(1)"), NOW),
                         "main yielded to icron; it waits for job k3f9-1 (running code), unless "
                         "you write first.")
        self.assertEqual(cli._yield_notice(None, NOW),
                         "main yielded to icron, but nothing that could resume it is pending; "
                         "it waits for your message.")


class WakeUpTests(_IcronControllerCase):
    async def test_a_yield_waits_for_its_wake_up_which_starts_the_next_turn(self):
        model = _Model(self.path,
                       _calls(_icron("c1", call="schedule('Check the probe.', after=10*60)")),
                       _calls(_icron("c2", call="yield_('Waiting for the probe.')")),
                       _answer("The probe is done."))
        statuses = []

        def waiting(terminal, status):
            if "icron: main waits for" in status:
                statuses.append(status)
                return True
            return False

        terminal = _Terminal(_once([(waiting, lambda t: self.advance(600)),
                                    (_idle_with("[assistant] The probe is done."), _exit)]))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "Wait for the probe."]), 0)
        self.assertEqual(len(model.calls), 3)
        self.assertRegex(statuses[0], r"^idle \| icron: main waits for [a-z0-9]{4}-1 at 12:10:00$")
        texts = _texts(terminal)
        self.assertIn("[tool-call] icron (c1)\nicron.schedule(message='Check the probe.', "
                      "after=600)", texts)
        notice = next(text for text in texts if text.startswith("[cli] main yielded to icron"))
        self.assertRegex(notice, r"^\[cli\] main yielded to icron; job [a-z0-9]{4}-1 resumes it "
                                 r"at 12:10:00 \(in 10m\), unless you write first\.$")
        items = model.calls[2][0].items
        self.assertIsInstance(items[-1], UserInteractionBoundary)
        wake = items[-2].content_text
        self.assertTrue(wake.startswith(f"{WAKE_HEADER} at "), wake)
        self.assertIn("It resumes your turn that ended with yield.", wake)
        self.assertRegex(wake, r"Job [a-z0-9]{4}-1 \(yours; due [^)]+\):\nCheck the probe\.$")

    async def test_a_wake_up_while_idle_starts_a_turn(self):
        model = _Model(self.path,
                       _calls(_icron("c1", action="schedule", message="Look again.", after=60)),
                       _answer("Scheduled."), _answer("Looked."))
        terminal = _Terminal(_once([(_idle_with("[assistant] Scheduled."),
                                     lambda t: self.advance(60)),
                                    (_idle_with("[assistant] Looked."), _exit)]))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 0)
        items = model.calls[2][0].items
        self.assertIsInstance(items[-1], UserInteractionBoundary)
        self.assertNotIn("It resumes your turn", items[-2].content_text)
        self.assertTrue(items[-2].content_text.endswith("):\nLook again."))

    async def test_a_wake_up_during_a_tool_call_is_a_steer_in_its_place(self):
        probe, entered, release = _blocking_probe()
        model = _Model(self.path,
                       _calls(_icron("c1", action="schedule", message="From icron.", after=60),
                              ToolCall("probe", "p1", "{}")),
                       _answer("done"))
        seen = {"due": 0}

        def due_for_two_frames(terminal, status):
            seen["due"] += "icron: 1 due" in status
            return seen["due"] >= 2

        terminal = _Terminal(_once([
            (lambda t, s: entered.is_set(), lambda t: t.submit("First steer.")),
            (lambda t, s: "steers=1" in s, lambda t: self.advance(60)),
            (due_for_two_frames, lambda t: t.submit("Second steer.")),
            (lambda t, s: "steers=2 | icron: 1 due" in s, lambda t: release.set()),
            (_idle_with("[assistant] done"), _exit)]))
        try:
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"], (probe,)), 0)
        finally:
            release.set()
        self.assertEqual(len(model.calls), 2)  # the steers and the wake-up joined the turn
        items = model.calls[1][0].items
        self.assertEqual(items[-4], ToolResult("p1", "probed"))
        first, wake, second = _users(items[-3:])
        self.assertEqual((first, second), ("First steer.", "Second steer."))
        self.assertTrue(wake.startswith(WAKE_HEADER) and wake.endswith("):\nFrom icron."), wake)
        self.assertFalse(any(isinstance(item, UserInteractionBoundary) for item in items[-4:]))
        statuses = [status for _editor, status, _prompt in terminal.frames]
        self.assertFalse(any("steers=3" in status or "queued=" in status for status in statuses))

    async def test_behind_a_task_a_wake_up_waits_its_turn(self):
        probe, entered, release = _blocking_probe()
        model = _Model(self.path,
                       _calls(_icron("c1", action="schedule", message="From icron.", after=60),
                              ToolCall("probe", "p1", "{}")),
                       _answer("first"), _answer("task"), _answer("woke"))
        terminal = _Terminal(_once([
            (lambda t, s: entered.is_set(), lambda t: t.submit("/task Next query.")),
            (lambda t, s: "queued=1" in s, lambda t: self.advance(60)),
            (lambda t, s: "queued=1 | icron: 1 due" in s, lambda t: release.set()),
            (_idle_with("[assistant] woke"), _exit)]))
        try:
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"], (probe,)), 0)
        finally:
            release.set()
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(_users(model.calls[1][0].items), ["hello"])
        self.assertEqual(_users(model.calls[2][0].items)[-1], "Next query.")
        self.assertTrue(_users(model.calls[3][0].items)[-1].endswith("):\nFrom icron."))

    async def test_main_s_code_drives_main_until_the_tests_pass(self):
        code = ("for attempt in range(3):\n"
                "    if attempt == 2:\n"
                "        print('The tests pass.')\n"
                "        break\n"
                "    y = await main.resume(f'The suite fails ({attempt}). Fix it.')\n"
                "    print(y.kind, y.final_text)\n")
        model = _Model(self.path, _calls(_icron("c1", call="exec(wait=5)", code=code)),
                       _answer("Fixed 0."), _answer("Fixed 1."))
        terminal = _Terminal(_once([(lambda t, s: s == "idle" and _has(t, "finished."),
                                     _exit)]))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "Make the tests pass."]),
                         0)
        self.assertEqual(len(model.calls), 3)
        first = model.calls[1][0].items  # the same turn: the message was a steer
        result = first[-2].output.splitlines()
        self.assertRegex(result[0], r"^Script running with job ID [a-z0-9]{4}-1$")
        self.assertRegex(result[1], r"^Wall time \d+\.\d seconds \(it sent you a message, "
                                    r"which you get after this tool batch\)$")
        self.assertTrue(_users(first)[-1].endswith("The suite fails (0). Fix it."))
        self.assertIn("(your code, from exec at ", _users(first)[-1])
        second = model.calls[2][0].items  # a new turn, started by the next message
        self.assertIsInstance(second[-1], UserInteractionBoundary)
        self.assertTrue(_users(second)[-1].endswith("The suite fails (1). Fix it."))
        texts = _texts(terminal)
        self.assertTrue(any(re.fullmatch(r"\[cli\] icron: job [a-z0-9]{4}-1 \(main's code\) "
                                         r"finished\.", text) for text in texts), texts)
        self.assertIn("[tool-call] icron (c1)\nicron.exec(wait=5)", texts)

    @mock.patch.object(cli, "_PY_ENABLED", True)  # /py is off for now; its code still works
    async def test_the_user_s_code_can_resume_main(self):
        model = _Model(self.path, _answer("Ready."), _answer("Got it."))
        terminal = _Terminal(_once([
            (_idle_with("[assistant] Ready."),
             lambda t: t.submit("/py await main.resume('From the user.')")),
            (_idle_with("[assistant] Got it."), _exit)]))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 0)
        wake = _users(model.calls[1][0].items)[-1]
        self.assertRegex(wake, r"Job [a-z0-9]{4}-1 \(the user's code, from /py at "
                               r"\d\d:\d\d:\d\d\):\nFrom the user\.$")

    async def test_after_a_failed_turn_a_due_wake_up_still_runs(self):
        model = _Model(self.path,
                       _calls(_icron("c1", action="schedule", message="Try again.", after=60)),
                       ModelTransportError("lost"), _answer("Recovered."))
        terminal = _Terminal(_once([
            (lambda t, s: _phase(s) == "failed" and _has(t, "Sampling failed."),
             lambda t: self.advance(60)),
            (_idle_with("[assistant] Recovered."), lambda t: t.submit("/retry")),
            (lambda t, s: _has(t, "No retryable sampling failure"), _exit)]))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        self.assertEqual(len(model.calls), 3)
        self.assertTrue(_users(model.calls[2][0].items)[-1].endswith("):\nTry again."))

    async def test_exit_drops_jobs_and_no_wake_up_runs_after_it(self):
        model = _Model(self.path,
                       _calls(_icron("c1", action="schedule", message="Later.", after=600),
                              _icron("c2", action="exec", code="await asyncio.sleep(30)",
                                     wait=0)),
                       _answer("ok"))
        terminal = _Terminal(_once([(_idle_with("[assistant] ok"), lambda t: t.submit("/exit"))]))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 0)
        self.assertTrue(any(re.fullmatch(r"\[cli\] icron: dropped [12] pending jobs?; jobs and "
                                         r"Python state live only in this process\.", text)
                            for text in _texts(terminal)), _texts(terminal))
        self.assertEqual(self.icron.jobs(), ())
        self.advance(600)
        time.sleep(0.05)
        self.assertEqual(len(model.calls), 2)

    @mock.patch.object(cli, "_PY_ENABLED", True)  # /py holds the watcher here
    async def test_the_status_line_shows_running_code_and_a_stall(self):
        self.icron = Icron(clock=self.clock, limits=IcronLimits(stall_seconds=0.2))
        self.addCleanup(self.icron.close, 3.0)
        model = _Model(self.path,
                       _calls(_icron("c1", action="exec", code="await asyncio.sleep(2)",
                                     wait=0.05)),
                       _answer("started"))
        terminal = _Terminal(_once([
            (lambda t, s: s == "idle | icron: 1 running",
             lambda t: t.submit("/py import time\ntime.sleep(1.5)")),
            (lambda t, s: re.search(r"icron: [a-z0-9]{4}-2 has held the watcher for 1s$", s),
             _exit)]))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 0)

    async def test_main_s_tool_runs_on_the_context_thread_and_code_on_the_watcher(self):
        threads, tool = set(), self.icron.tool()

        def handler(arguments, **kwargs):
            threads.add(threading.current_thread().name)
            return tool.handler(arguments, **kwargs)

        wrapped = Tool(tool.spec, handler, timeout_seconds=tool.timeout_seconds)
        model = _Model(self.path,
                       _calls(_icron("c1", action="exec",
                                     code="import threading\nthreading.current_thread().name")),
                       _answer("done"))
        terminal = _Terminal(_once([(_idle_with("[assistant] done"), _exit)]))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"], (wrapped,),
                                         icron_tool=False), 0)
        self.assertEqual(threads, {"interaction-context_0"})
        self.assertIn("Value: 'interaction-icron'",
                      model.calls[1][0].items[-1].output.splitlines())


class ResumeTests(_IcronControllerCase):
    async def test_a_save_that_ended_with_an_icron_yield_needs_a_message(self):
        # A stop cut the yield's batch short: no turn summary follows it.
        items = (Init("old"), Tools(), Message("user", "q"), UserInteractionBoundary(),
                 _icron("y", action="yield", note="Waiting."), ModelSampleBoundary(),
                 ToolResult("y", json.dumps({"yielded": True})))
        for icron in (True, False):
            with self.subTest(icron=icron):
                seed_cli_save(self.path, InteractionContext(items))
                model = _Model(self.path)
                terminal = _Terminal(_once([
                    (lambda t, s: _phase(s) == "idle", lambda t: t.submit("/continue")),
                    (lambda t, s: _has(t, cli._ICRON_CONTINUE_REFUSAL), _exit)]))
                self.assertEqual(await self._run(model, terminal, ["--resume"], icron=icron), 0)
                self.assertEqual(model.calls, [])
                self.assertIn(f"[cli] {cli._ICRON_YIELD_NOTICE}", _texts(terminal))

    async def test_a_save_with_jobs_that_could_be_pending_says_they_were_not_restored(self):
        due = datetime.fromtimestamp(time.time() + 3600).astimezone().isoformat()
        items = (Init("old"), Tools(), Message("user", "q"), UserInteractionBoundary(),
                 _icron("s", action="schedule", message="x", after=3600), ModelSampleBoundary(),
                 ToolResult("s", json.dumps({"job_id": "k-1", "due": due, "every": None})),
                 Message("assistant", "Scheduled."))
        seed_cli_save(self.path, InteractionContext(items))
        terminal = _Terminal(_once([(lambda t, s: _phase(s) == "idle", _exit)]))
        self.assertEqual(await self._run(_Model(self.path), terminal, ["--resume"]), 0)
        self.assertIn(f"[cli] {cli._ICRON_RESTART_NOTICE}", _texts(terminal))


class HeadlessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "interaction.jsonl"
        self.clock = _Clock()
        self.icron = Icron(clock=self.clock)
        self.addCleanup(self.icron.close, 1.0)

    async def _run_headless(self, model):
        args = cli._build_parser().parse_args(["--headless", "--prompt", "Wait for the probe."])
        with redirect_stderr(io.StringIO()) as stderr:
            code = await asyncio.wait_for(cli._run_headless(
                model, Environment((self.icron.tool(),)), args, self.path, icron=self.icron), 8)
        return code, stderr.getvalue()

    async def test_a_yield_keeps_a_headless_run_going_until_its_wake_up_turn_ends(self):
        model = _Model(self.path,
                       _calls(_icron("c1", action="schedule", message="Check.", after=600)),
                       _calls(_icron("c2", action="yield", note="Waiting.")),
                       _answer("done"))

        def wake():
            if _until(lambda: self.icron.main.state == "suspended", 6):
                self.clock.now += 600
                self.icron._poke()

        threading.Thread(target=wake, daemon=True).start()
        code, stderr = await self._run_headless(model)
        self.assertEqual((code, len(model.calls)), (0, 3))
        self.assertIn("[cli] main yielded to icron; job ", stderr)
        self.assertIn("[cli] icron: main can schedule work and run Python in "
                      "this process, unsandboxed, with the icron tool. Jobs and Python state "
                      "live only in this process.", stderr)

    async def test_a_final_answer_ends_a_headless_run_and_drops_its_jobs(self):
        model = _Model(self.path,
                       _calls(_icron("c1", action="schedule", message="Check.", after=600)),
                       _answer("done"))
        code, stderr = await self._run_headless(model)
        self.assertEqual((code, len(model.calls)), (0, 2))
        self.assertIn("[cli] icron: dropped 1 pending job; jobs and Python state live only in "
                      "this process.", stderr)


def _enter(state, text):
    state.editor = Editor(text, len(text))
    state.handle_key("c-m", "\r")


def _notices(state):
    texts = [item.text.removeprefix("[cli] ") for item in state.displays]
    state.displays.clear()
    return texts


class QueueTests(unittest.TestCase):
    def test_a_wake_up_marker_steers_with_the_steers_but_is_not_the_user_s_input(self):
        state = cli._UIState(ready=True, turn_active=True)
        state.set_phase("tool: probe")
        state.pending.append(cli._WakeIntent())
        _enter(state, "/steer!")
        self.assertEqual(_notices(state), ["No queued steers to flush."])
        _enter(state, "Also check the logs.")
        self.assertEqual((cli._steer_count(state), cli._queued_input(state)), (1, 1))
        for _ in range(cli._MAX_PENDING_QUERIES - 1):
            _enter(state, "more")
        self.assertEqual(len(state.pending), cli._MAX_PENDING_QUERIES + 1)
        _enter(state, "one too many")
        self.assertEqual(_notices(state)[-1],
                         "Query queue is full; the draft has not been submitted.")

    def test_the_interrupt_takes_icron_s_message_in_the_marker_s_place(self):
        icron = Icron()
        state = cli._UIState(ready=True, turn_active=True, icron=icron)
        state.pending.extend((cli._WakeIntent(), cli._Submission("after", Message("user",
                                                                                  "after"))))
        self.assertEqual(cli._take_interrupt(state).messages,
                         (Message("user", "after"),))  # withdrawn
        icron.main.resume("From the host.")
        state.pending.extend((cli._Submission("before", Message("user", "before")),
                              cli._WakeIntent()))
        taken = cli._take_interrupt(state)
        before, wake = taken.messages
        self.assertEqual(before, Message("user", "before"))
        self.assertTrue(wake.content_text.endswith("From the host:\nFrom the host."), wake)
        self.assertEqual((list(state.pending), icron.undelivered()), ([], 0))

    def test_due_wake_ups_wait_for_a_model_with_one_notice(self):
        icron = Icron()
        icron.main.resume("x")
        state = cli._UIState(ready=True, auth_required=True, icron=icron)
        cli._queue_wake(state)
        cli._queue_wake(state)
        self.assertEqual(list(state.pending), [])
        self.assertEqual(_notices(state), [
            "icron: 1 wake-up is due; delivered once the model is ready (/login)."])
        state.auth_required = False
        cli._queue_wake(state)
        cli._queue_wake(state)  # one marker stands for every due wake-up
        self.assertEqual([type(entry) for entry in state.pending], [cli._WakeIntent])
        self.assertTrue(state.changed.is_set())
        closing = cli._UIState(ready=True, closing=True, icron=icron)
        cli._queue_wake(closing)
        self.assertEqual(list(closing.pending), [])


class _CommandCase(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.icron = Icron(clock=self.clock)
        self.icron_notices = []
        self.icron.start(lambda: None, self.icron_notices.append)
        self.addCleanup(self.icron.close, 1.0)
        self.state = cli._UIState(ready=True, icron=self.icron)
        self.state.set_phase("sampling")  # the commands work while main is busy

    def enter(self, text):
        _enter(self.state, text)
        return _notices(self.state)

    def assert_kept(self, text, notice):
        self.assertEqual(self.enter(text), [notice])
        self.assertEqual(self.state.editor.text, text)

    def suspend_main_on(self, message, after):
        job = self.icron.schedule(message, after=after, by="main")
        self.icron.begin_turn()
        self.icron.yield_("Waiting.")
        self.icron.handoff(Yield(context=1, job_id=None, job_text="task", kind="yielded",
                                 resumable=True, revision=0, yield_text="Waiting."))
        return job


class CronCommandTests(_CommandCase):
    def test_in_every_and_at_schedule_the_user_s_message(self):
        (notice,) = self.enter("/cron in 10m Check the CI run.")
        job_id = re.fullmatch(r"cron: job (\S+) sends your message at 12:10:00 \(in 10m\)\.",
                              notice).group(1)
        self.assertEqual(self.state.editor.text, "")
        job = self.icron.jobs()[0]
        self.assertEqual((job.job_id, job.origin, job.message, job.due),
                         (job_id, "user", "Check the CI run.", NOW + 600))
        self.assertRegex(self.enter("/cron every 1h Poll the training log.")[0],
                         r"^cron: job \S+ sends your message every 1h, first at 13:00:00\.$")
        self.assertRegex(self.enter("/cron at 14:30 Post the summary.")[0],
                         r"^cron: job \S+ sends your message at 14:30:00 \(in 2h 30m\)\.$")
        self.assertRegex(self.enter("/cron at 9:00 Stand up.")[0],
                         r"^cron: job \S+ sends your message at 09:00:00 tomorrow \(in 21h\)\.$")
        self.enter("/cron in 1m line one\nline two")
        self.assertIn("line one\nline two", [job.message for job in self.icron.jobs()])

    def test_malformed_commands_keep_the_draft(self):
        usage = f"{cli._CRON_USAGE} Draft preserved."
        for text in ("/cron in ten x", "/cron in 10m", "/cron at 25:00 x", "/cron soon x",
                     "/cron cancel", "/cron in 0s x", "/cron in 1h30m x", "/cron every x y",
                     "/cron clear x", "/cron cancel! a b", "/cron cancel? all"):
            with self.subTest(text=text):
                self.assert_kept(text, usage)
        self.assert_kept("/cron in 8d x", "icron.schedule(): after must be from 1 to 604800 "
                                          "seconds. Draft preserved.")
        self.assert_kept("/cron every 29s x", "icron.schedule(): every must be from 30 to "
                                              "604800 seconds. Draft preserved.")
        self.assertEqual(self.icron.jobs(), ())

    def test_cancel_and_the_list(self):
        self.assertEqual(self.enter("/cron"), ["cron: no pending jobs."])
        main_job = self.suspend_main_on("Check CI run 123.", 600)
        running = self.icron.exec("await asyncio.sleep(5)\nprint('done')", wait=0.05,
                                  by="main")
        self.assertEqual(self.enter("/cron"), [
            "cron: 2 pending jobs (now 12:00:00):\n"
            f"  {running.job.job_id} (main's) running code since 12:00:00: "
            "await asyncio.sleep(5) print('done')\n"
            f"  {main_job.job_id} (main's) 12:10:00, in 10m, once: Check CI run 123."])
        # Main's jobs take cancel! or cancel!!; a bare cancel keeps the draft.
        job_id = running.job.job_id
        self.assert_kept(f"/cron cancel {job_id}",
                         f"cron: job {job_id} is main's: /cron cancel! {job_id} cancels it and "
                         f"lets its running code finish its run, and /cron cancel!! {job_id} "
                         "stops that code too. Draft preserved.")
        self.assertEqual(self.enter(f"/cron cancel!! {job_id}"),
                         [f"cron: cancelled job {job_id} (main's); its running code stops at "
                          "its next await."])
        self.assertEqual(self.enter(f"/cron cancel! {main_job.job_id}"),
                         [f"cron: cancelled job {main_job.job_id} (main's); main now waits for "
                          "your message."])
        # So does every job; three or more ! read as two.
        self.enter("/cron in 1m a")
        self.icron.schedule("b", after=120, by="main")
        self.assert_kept("/cron cancel all", f"{cli._CRON_CANCEL_ALL_HINT} Draft preserved.")
        self.assertEqual(self.enter("/cron cancel!!! all"), ["cron: cancelled 2 jobs."])
        self.assertEqual(self.enter("/cron cancel! all"), ["cron: no pending jobs to cancel."])
        self.assert_kept("/cron cancel nope-1",
                         "icron.cancel(): there is no job nope-1. Draft preserved.")

    def test_cancel_with_one_bang_lets_running_code_finish_its_run(self):
        mine = self.enter("/cron in 10m Check the CI run.")[0].split()[2]
        self.assertEqual(self.enter(f"/cron cancel {mine}"),  # the user's own: no bang
                         [f"cron: cancelled job {mine} (yours)."])
        job_id = self.icron.exec("await asyncio.sleep(0.2)\nprint('done')", wait=0.05,
                                 by="main").job.job_id
        self.assertEqual(self.enter(f"/cron cancel! {job_id}"),
                         [f"cron: cancelled job {job_id} (main's); its running code finishes "
                          "its run."])
        self.assertTrue(_until(lambda: self.icron_notices
                               == [f"icron: job {job_id} (main's code) finished."]))
        ended = next(job for job in self.icron.jobs() if job.job_id == job_id)
        self.assertEqual((ended.state, ended.output), ("done", "done\n"))
        quick, slow = (self.icron.exec(f"await asyncio.sleep({seconds})", wait=0.05,
                                       by="main").job.job_id for seconds in (0.2, 30))

        def state(job_id):
            return next(job.state for job in self.icron.jobs() if job.job_id == job_id)

        self.assertEqual(self.enter("/cron cancel! all"),
                         ["cron: cancelled 2 jobs; code running in 2 of them finishes its run."])
        self.assertTrue(_until(lambda: state(quick) == "done"))
        self.assertEqual(state(slow), "running")  # until cancel!!
        self.assertEqual(self.enter("/cron cancel!! all"),
                         ["cron: cancelled 1 job; code running in 1 of them stops at its next "
                          "await."])
        self.assertTrue(_until(lambda: state(slow) == "cancelled"))

    def test_the_commands_work_while_the_model_needs_authentication(self):
        self.state.auth_required = True
        self.assertRegex(self.enter("/cron in 10m x")[0], r"^cron: job \S+ sends your message")

    def test_icron_is_an_alias_of_cron(self):
        (notice,) = self.enter("/icron in 10m Check the CI run.")
        job_id = re.fullmatch(r"cron: job (\S+) sends your message at 12:10:00 \(in 10m\)\.",
                              notice).group(1)
        (job,) = self.icron.jobs()
        # Its job says /cron, as the command's other name's would.
        self.assertEqual((job.job_id, job.origin, job.via), (job_id, "user", "/cron"))
        self.assertRegex(self.enter("/icron")[0], r"^cron: 1 pending job \(now 12:00:00\):")
        self.assert_kept("/icron soon x", f"{cli._CRON_USAGE} Draft preserved.")
        self.assertEqual(self.enter("/icron clear"), ["cron: cancelled 1 job of yours."])

    def test_without_icron_the_commands_are_unsupported(self):
        state = cli._UIState(ready=True)
        for text in ("/cron in 10m x", "/icron in 10m x", "/crontab", "/py 1"):
            _enter(state, text)
            self.assertTrue(_notices(state)[0].startswith("Unsupported command. Use /retry"))


class ClearCommandTests(_CommandCase):
    def test_clear_cancels_only_the_user_s_cron_jobs(self):
        self.assertEqual(self.enter("/cron clear"), ["cron: no /cron jobs of yours to clear."])
        self.enter("/cron in 10m Check the CI run.")
        self.enter("/icron every 1h Poll the training log.")
        mains = self.icron.schedule("Check CI run 123.", after=300, by="main")
        afk = self.icron.schedule("Away.", every=1800, by="user", via="/afk")
        self.assertEqual(self.enter("/cron clear"), ["cron: cancelled 2 jobs of yours."])
        self.assertEqual(sorted(job.job_id for job in self.icron.jobs() if job.pending),
                         sorted((mains.job_id, afk.job_id)))

    def test_clear_says_when_main_is_left_waiting_for_the_user(self):
        self.enter("/cron in 10m Check the CI run.")
        self.icron.begin_turn()
        self.icron.yield_("Waiting.")  # on the user's job: nothing else could resume it
        self.icron.handoff(Yield(context=1, job_id=None, job_text="task", kind="yielded",
                                 resumable=True, revision=0, yield_text="Waiting."))
        self.assertEqual(self.enter("/cron clear"),
                         ["cron: cancelled 1 job of yours; main now waits for your message."])

    def test_crontab_is_gone_with_a_hint(self):
        for text in ("/crontab", "/crontab clear"):
            self.assert_kept(text, f"{cli._CRONTAB_GONE} Draft preserved.")


class PyCommandTests(_CommandCase):
    def test_py_is_disabled(self):
        self.assertFalse(cli._PY_ENABLED)
        self.assert_kept("/py print(1)", "py: /py is disabled. Draft preserved.")
        self.assertEqual((self.icron.jobs(), self.icron_notices), ((), []))

    @mock.patch.object(cli, "_PY_ENABLED", True)  # its code, for when it's back on
    def test_py_runs_the_user_s_code_and_shows_what_it_prints(self):
        self.assertEqual(self.enter("/py print(1)\n6 * 7"), [])
        self.assertEqual(self.state.editor.text, "")
        self.assertTrue(_until(lambda: len(self.icron_notices) == 2))
        job_id = self.icron_notices[0].split()[1].rstrip(":")
        self.assertEqual(self.icron_notices, [f"py {job_id}: 1", f"py {job_id}: done: 42"])
        self.enter("/py\n    y = 5\n    print(y)")  # an indented paste is dedented
        self.assertTrue(_until(lambda: len(self.icron_notices) == 4))
        self.assertTrue(self.icron_notices[2].endswith(": 5"))
        self.enter("/py y * 2")  # the user's namespace lasts for the session
        self.assertTrue(_until(lambda: len(self.icron_notices) == 5))
        self.assertTrue(self.icron_notices[4].endswith(": done: 10"))
        (notice,) = self.enter("/py def f(:")
        self.assertTrue(notice.startswith("py: failed:\n  "))
        self.assertIn("SyntaxError", notice)
        self.assertTrue(notice.endswith("\nDraft preserved."))
        self.assertEqual(self.state.editor.text, "/py def f(:")
        self.assert_kept("/py", f"{cli._PY_USAGE} Draft preserved.")


if __name__ == "__main__":
    unittest.main()
