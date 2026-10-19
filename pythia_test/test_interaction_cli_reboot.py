"""Reboots: /reboot and compaction_coldstart_reboot restart the CLI on the current code."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, redirect_stderr
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import CompactionMetadata, CompactionResult, ContextPrefix, Environment
from pythia.interaction import Icron, Init, InteractionConfig, InteractionConfigSnapshot
from pythia.interaction import InteractionContext, Message, ModelSample, ModelSampleBoundary
from pythia.interaction import SampleMetadata, TokenUsage, Tool, ToolCall, ToolOutcome, ToolResult
from pythia.interaction import ToolSpec, Tools, TurnSummary, UserInteractionBoundary, UserToolCall
from pythia.interaction import UserToolResult, cli, demo, load_interaction_save
from pythia.interaction import save_interaction_save
from pythia.interaction._cli_editor import Editor
from pythia.interaction.loop import kernel
from pythia.interaction.loop.tail import turn_tail
from pythia.interaction.runtime_config import ConfigError, parse_config_literal
from pythia.interaction.user_tools import UserToolIntent
from pythia_test.interaction_helpers import patch_compaction
from pythia_test.test_interaction_cli import _Model, _Terminal, _answer


# A command line that the start check can run: it exits at once.
_COMMAND = (sys.executable, "-c", "pass", "--rebooted")


def _rebootable(check=None):
    """Let the CLI reboot: a known command line, and a start check with ``check``'s
    answer (None passes)."""
    stack = ExitStack()
    stack.enter_context(mock.patch.object(cli, "_reboot_command", side_effect=lambda: [*_COMMAND]))
    stack.enter_context(mock.patch.object(cli, "_check_reboot", return_value=check))
    return stack


def _submit(state, text):
    state.editor = Editor(text, len(text))
    state.handle_key("c-m", "\r")


def _notices(items):
    return [item.text[len("[cli] "):] for item in items if item.text.startswith("[cli] ")]


def _texts(terminal):
    return [item.text for item in terminal.items]


def _has(terminal, text):
    return any(text in item.text for item in terminal.items)


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


def _coldstart_result():
    return CompactionResult((ContextPrefix((Message("user", "a pointer to the log"),)),),
                            protocol="coldstart", request_attempts=0)


class _Compactor:
    """Returns a coldstart checkpoint, once ``gate`` (if any) opens."""

    def __init__(self, gate=None):
        self.gate, self.entered, self.calls = gate, threading.Event(), 0

    def compact(self, source, *, tools=(), sample_params=None, instructions=None):
        self.calls += 1
        self.entered.set()
        if self.gate is not None and not self.gate.wait(4):
            raise AssertionError("the test did not open the gate")
        return _coldstart_result()


def _steps(*steps):
    """A frame callback that runs each (condition, action) step once, in order."""
    progress = {"step": 0}

    def frame(terminal, editor, status):
        if progress["step"] < len(steps):
            condition, action = steps[progress["step"]]
            if condition(terminal, status):
                progress["step"] += 1
                action(terminal)

    frame.progress = progress
    return frame


def _idle_with(text):
    # The phase, without the status line's counts and icron's part.
    return lambda terminal, status: status.split(" | ")[0] == "idle" and _has(terminal, text)


def _exit(terminal):
    terminal.key("c-d")


def _comparable(rebooted):
    """A _Rebooted's fields; queued entries compare by identity, so these by value."""
    queue = tuple((type(entry).__name__, getattr(entry, "text", None),
                   getattr(entry, "message", None), getattr(entry, "steers", None),
                   getattr(entry, "name", None), getattr(entry, "arguments_json", None))
                  for entry in rebooted.queue)
    return (rebooted.cause, rebooted.continues, rebooted.config, rebooted.draft, queue,
            rebooted.exit_code, rebooted.warning)


class RebootConfigTests(unittest.TestCase):
    def test_the_key_is_a_live_boolean_and_a_cli_launch_option(self):
        parser = cli._build_parser()
        for argv, expected in (([], False), (["--compaction-coldstart-reboot"], True),
                               (["--compaction-coldstart-reboot=False"], False)):
            args = parser.parse_args(argv)
            self.assertIs(args.compaction_coldstart_reboot, expected)
            self.assertIs(InteractionConfig.from_namespace(args).get(
                "compaction_coldstart_reboot"), expected)
        self.assertIs(parse_config_literal("compaction_coldstart_reboot", "true"), True)
        config = InteractionConfig()
        self.assertIs(config.set("compaction_coldstart_reboot", True), True)
        with self.assertRaisesRegex(ConfigError, "requires True or False"):
            config.set("compaction_coldstart_reboot", 1)
        help_text = " ".join(parser.format_help().split())
        self.assertIn("--compaction-coldstart-reboot [{False,True}]", help_text)
        self.assertNotIn("--rebooted", help_text)  # the reboot's own option
        # Only the CLI reboots: the demo has no option, and leaves the key False.
        demo_args = demo._build_parser().parse_args([])
        self.assertFalse(hasattr(demo_args, "compaction_coldstart_reboot"))
        self.assertIs(InteractionConfig.from_namespace(demo_args).get(
            "compaction_coldstart_reboot"), False)


class _Host(kernel.TurnHost):
    def __init__(self, steer=None, reboot=True):
        self.compacted_results = []
        self.steer, self.reboot, self.rebooting = steer, reboot, False

    def append(self, context, items):
        context.extend(items)

    def interrupt(self):
        if self.steer is not None:
            steer, self.steer = self.steer, None
            return kernel.Steer((Message("user", steer),))
        return kernel.Interrupt.CONTINUE

    def compacted(self, result):
        self.compacted_results.append(result)
        self.rebooting = self.reboot

    def reboot_requested(self):
        return self.rebooting


class _Sampler:
    def __init__(self):
        self.calls = []

    def sample(self, context, *, tools=(), sample_params=None):
        self.calls.append(context.copy())
        return _answer("done")


class KernelRebootTests(unittest.TestCase):
    def _turn(self, host):
        context = InteractionContext((Init("m"), Tools(), Message("user", "go"),
                                      UserInteractionBoundary()))
        model = _Sampler()
        config = InteractionConfigSnapshot(auto_compaction_context_tokens=1, compaction_mode="pi")
        with mock.patch.object(kernel, "create_default_compactor", return_value=_Compactor()):
            result = kernel.run_turn(context, model, Environment(), config, host)
        return result, model, context

    def test_a_reboot_ends_the_turn_once_the_steers_of_its_interrupt_point_are_saved(self):
        host = _Host(steer="Also this.")
        result, model, context = self._turn(host)
        self.assertEqual(result, kernel.STOPPED)
        self.assertEqual(model.calls, [])
        self.assertEqual([result.protocol for result in host.compacted_results], ["coldstart"])
        prefix, metadata, steer = context.items[-3:]
        self.assertIsInstance(prefix, ContextPrefix)
        self.assertIsInstance(metadata, CompactionMetadata)
        self.assertEqual(steer, Message("user", "Also this."))
        self.assertTrue(turn_tail(context.items).unfinished)  # so it can be continued

    def test_without_a_reboot_the_turn_goes_on(self):
        result, model, _context = self._turn(_Host(steer="Also this.", reboot=False))
        self.assertEqual(result, kernel.TurnResult("ended", "done"))
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0].items[-1], Message("user", "Also this."))


class RebootCommandTests(unittest.TestCase):
    def test_usage_refusals_and_a_pending_reboot(self):
        state = cli._UIState(ready=True, phase="idle")
        _submit(state, "/reboot")  # main() didn't run: no command line to run again
        with _rebootable():
            _submit(state, "/reboot now")
            _submit(state, "/reboot\nFAKE_SECRET")
            self.assertIsNone(state.reboot)
            _submit(state, "/reboot")
            self.assertEqual(state.reboot, cli._Reboot("/reboot"))
            self.assertTrue(state.changed.is_set())
            _submit(state, "/reboot")
            busy = cli._UIState(ready=True, phase="sampling")
            _submit(busy, "/reboot")
            _submit(busy, "/reboot!")
        self.assertEqual(_notices(state.displays), [
            "Cannot reboot: this process's command line isn't known, so it can't be run again.",
            "Usage: /reboot (no arguments; single line).",
            "Usage: /reboot (no arguments; single line).",
            "A reboot is already pending.",
        ])
        self.assertEqual((state.editor, list(state.pending)), (Editor(), []))
        self.assertEqual(busy.reboot, cli._Reboot("/reboot"))
        notices = _notices(busy.displays)
        self.assertEqual(notices[0], "Rebooting once the current operation allows: a running "
                                     "turn finishes its tool batch and continues after the "
                                     "restart. /exit exits instead.")
        self.assertTrue(notices[1].startswith("Unsupported command. Use /retry"))
        self.assertIn("/reboot", notices[1])

    def test_icron_jobs_live_only_in_this_process(self):
        icron = Icron()
        self.addCleanup(icron.close, 1.0)
        state = cli._UIState(ready=True, phase="idle", icron=icron)
        icron.schedule("ping", after=600, by="user")
        with _rebootable():
            _submit(state, "/reboot")
            self.assertIsNone(cli._reboot_refusal(cli._UIState()))
        self.assertIsNone(state.reboot)
        self.assertEqual(_notices(state.displays), [
            "Cannot reboot: icron has 1 pending job, and jobs live only in this process "
            "(/cron lists them)."])


class RebootStateTests(unittest.TestCase):
    def _state(self):
        state = cli._UIState(editor=Editor("half typed", 4), exit_code=1,
                             reboot=cli._Reboot("/reboot", continues=True))
        state.pending.extend((
            cli._Submission("Then this.", Message("user", "Then this."), steers=False),
            UserToolIntent("config", '{"key":"max_samples","value":7}'),
            cli._RetryIntent(),  # its ticket stays: /continue reads the log instead
        ))
        return state

    def test_the_state_carries_config_changes_the_draft_and_the_queue(self):
        config = InteractionConfig(InteractionConfigSnapshot(max_samples=3))
        config.set("max_samples", 4)
        config.set("enable_workspace", True)  # unchanged: not carried
        carried = cli._reboot_state(self._state(), config)
        self.assertEqual(json.loads(json.dumps(carried)), carried)
        self.assertEqual(carried["config"], {"max_samples": 4})
        self.assertEqual(carried["draft"], {"text": "half typed", "cursor": 4})
        self.assertEqual([entry["kind"] for entry in carried["queue"]],
                         ["text", "command", "continue"])
        rebooted = cli._parse_reboot_state(json.loads(json.dumps(carried)))
        self.assertEqual(rebooted.cause, "/reboot")
        self.assertTrue(rebooted.continues)
        self.assertEqual(rebooted.config, {"max_samples": 4})
        self.assertEqual(rebooted.draft, Editor("half typed", 4))
        self.assertEqual(rebooted.exit_code, 1)
        text, command, continuation = rebooted.queue
        self.assertEqual((text.text, text.message, text.steers),
                         ("Then this.", Message("user", "Then this."), False))
        self.assertEqual(command, UserToolIntent("config", '{"key":"max_samples","value":7}'))
        self.assertIsInstance(continuation, cli._ContinueIntent)
        self.assertIsNone(rebooted.warning)

    def test_it_reaches_the_restarted_cli_through_an_inherited_descriptor(self):
        carried = cli._reboot_state(self._state(), InteractionConfig())
        seen = {}

        def execve(path, argv, environment):
            descriptor = int(environment[cli._REBOOT_STATE_FD])
            seen.update(path=path, argv=argv, inheritable=os.get_inheritable(descriptor),
                        data=os.pread(descriptor, 1 << 20, 0))

        with _rebootable(), mock.patch.object(cli.os, "execve", side_effect=execve):
            cli._reboot_exec(carried)
        self.assertEqual((seen["path"], seen["argv"]), (sys.executable, [*_COMMAND]))
        self.assertTrue(seen["inheritable"])
        self.assertEqual(json.loads(seen["data"]), carried)
        # The restarted CLI's side: it reads the state, and closes the descriptor.
        with tempfile.TemporaryFile() as file:
            file.write(seen["data"])
            file.flush()
            descriptor = os.dup(file.fileno())
        with mock.patch.dict(os.environ, {cli._REBOOT_STATE_FD: str(descriptor)}):
            rebooted = cli._take_reboot_state()
            self.assertNotIn(cli._REBOOT_STATE_FD, os.environ)
        self.assertEqual(_comparable(rebooted), _comparable(cli._parse_reboot_state(carried)))
        with self.assertRaises(OSError):
            os.fstat(descriptor)

    def test_a_state_that_cannot_be_read_is_a_warning(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(cli._REBOOT_STATE_FD, None)
            self.assertEqual(cli._take_reboot_state(), cli._Rebooted(
                warning="it got no state from the process it replaced"))
        for data in (b"not json", b'{"version": 2}', b'{"version": 1, "cause": "other"}'):
            with self.subTest(data=data), tempfile.TemporaryFile() as file:
                file.write(data)
                file.flush()
                descriptor = os.dup(file.fileno())
                with mock.patch.dict(os.environ, {cli._REBOOT_STATE_FD: str(descriptor)}):
                    rebooted = cli._take_reboot_state()
                self.assertIsNone(rebooted.cause)
                self.assertRegex(rebooted.warning, r"^its state couldn't be read \(\w+\)$")
        for name in ("1", "x"):  # never stdout, nor a name that isn't a descriptor
            with mock.patch.dict(os.environ, {cli._REBOOT_STATE_FD: name}):
                self.assertEqual(cli._take_reboot_state().warning,
                                 "its state couldn't be read (ValueError)")
        os.fstat(1)

    def test_the_command_line_keeps_interpreter_options_and_drops_the_reboot_s_option(self):
        argv = ["python3", "-P", "-m", "pythia.interaction.cli", "--resume", "--rebooted"]
        with mock.patch.object(sys, "orig_argv", argv, create=True):
            self.assertEqual(cli._command_line(),
                             [sys.executable, "-P", "-m", "pythia.interaction.cli", "--resume"])
        main = SimpleNamespace(__spec__=SimpleNamespace(name="pythia.interaction.cli"))
        script = SimpleNamespace(__spec__=None)
        for module, args, expected in (  # Python 3.9: no orig_argv
                (main, ["/x/cli.py", "--resume", "--rebooted"],
                 [sys.executable, "-m", "pythia.interaction.cli", "--resume"]),
                (script, ["tool.py", "--resume"], [sys.executable, "tool.py", "--resume"]),
                (script, ["-c", "--resume"], None)):
            with self.subTest(args=args), mock.patch.object(sys, "orig_argv", None, create=True), \
                    mock.patch.object(sys, "argv", args), \
                    mock.patch.dict(sys.modules, {"__main__": module}):
                self.assertEqual(cli._command_line(), expected)

    def test_main_knows_the_command_line_only_for_the_process_s_own_arguments(self):
        with mock.patch.object(cli, "frontend_catalog", side_effect=RuntimeError("stop")), \
                redirect_stderr(io.StringIO()):
            cli.main(["--save", "x.jsonl"])
            self.assertIsNone(cli._reboot_command())
            with mock.patch.object(sys, "argv", ["cli", "--save", "x.jsonl"]):
                cli.main()
                self.assertEqual(cli._reboot_command()[-1], "--rebooted")
            cli.main(["--save", "x.jsonl"])
        self.assertIsNone(cli._reboot_command())

    def test_the_start_check_runs_the_command_line_with_rebooted_check(self):
        echo = [sys.executable, "-c",
                "import sys; sys.exit(int(sys.argv[1:] != ['--resume', '--rebooted=check']))",
                "--resume", "--rebooted"]
        with mock.patch.object(cli, "_reboot_command", return_value=echo):
            self.assertIsNone(cli._check_reboot())
        failing = [sys.executable, "-c",
                   "raise SystemExit('SyntaxError: invalid syntax (cli.py, line 1)')", "--rebooted"]
        with mock.patch.object(cli, "_reboot_command", return_value=failing):
            self.assertEqual(cli._check_reboot(), (
                "the restarted CLI would fail to start (its start check exited with status 1: "
                "SyntaxError: invalid syntax (cli.py, line 1))"))


class _RebootCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "interaction.jsonl"

    async def _session(self, model, terminal, argv=(), environment=None, **options):
        args = cli._build_parser().parse_args(list(argv))
        return await asyncio.wait_for(
            cli._run(model, environment or Environment(), terminal, args, self.path, **options),
            timeout=6)

    async def _rebooted(self, model, terminal, argv=(), environment=None, **options):
        """Run a session that reboots; returns what the restarted CLI would carry over."""
        with self.assertRaises(cli._Rebooting) as caught:
            await self._session(model, terminal, argv, environment, **options)
        self.assertTrue(terminal.exited)
        self.assertEqual([context.items for context, _, _ in model.calls], model.checkpoints)
        return caught.exception.state

    def _resumable(self):
        """A finished turn whose usage is over a threshold of 100 tokens."""
        save_interaction_save(self.path, InteractionContext((
            Init("saved"), Tools(), Message("user", "old request"), UserInteractionBoundary(),
            Message("assistant", "old answer"), SampleMetadata(TokenUsage(90, 10, 100, 20)),
            ModelSampleBoundary(), TurnSummary(sample_count=1, context_tokens=100))))


class RebootFlowTests(_RebootCase):
    async def test_reboot_while_idle_restarts_at_once(self):
        model = _Model(self.path, _answer("done"))
        terminal = _Terminal(_steps((_idle_with("[assistant] done"),
                                     lambda t: t.submit("/reboot"))))
        with _rebootable():
            carried = await self._rebooted(model, terminal, ["--prompt", "hello"])
        self.assertEqual(carried, {
            "version": 1, "cause": "/reboot", "continues": False, "config": {},
            "draft": {"text": "", "cursor": 0}, "queue": [], "exit_code": 0})
        self.assertIn("Rebooting: the CLI restarts on the current code.", _notices(terminal.items))
        self.assertEqual(_users(load_interaction_save(self.path)), ["hello"])

    async def test_reboot_during_a_turn_takes_its_steers_and_carries_the_rest(self):
        probe, entered, release = _blocking_probe()
        self._resumable()

        def type_during_the_tool(terminal):
            terminal.submit("Also check the docs.")  # a steer
            terminal.submit("/task Then this.")  # waits for the turn to end
            terminal.submit("/config max_samples 7")  # so does a command
            terminal.submit("/reboot")
            terminal.key("<bracketed-paste>", "half typed")  # a draft

        terminal = _Terminal(_steps(
            (lambda t, s: s == "idle",
             lambda t: t.submit("/config auto_compaction_context_tokens 5000")),
            (_idle_with("auto_compaction_context_tokens = 5000"), lambda t: t.submit("go")),
            (lambda t, s: entered.is_set(), type_during_the_tool),
            (lambda t, s: "queued=2" in s and "steers=1" in s, lambda t: release.set())))
        model = _Model(self.path, ModelSample((ToolCall("probe", "p1", "{}"),)))
        try:
            with _rebootable():
                carried = await self._rebooted(model, terminal, ["--resume"],
                                               Environment((probe,)))
        finally:
            release.set()
        self.assertEqual(len(model.calls), 1)
        saved = load_interaction_save(self.path)
        # The batch finished, and the steer reached the turn before it ended.
        self.assertEqual(saved.items[-2:], (ToolResult("p1", "probed"),
                                            Message("user", "Also check the docs.")))
        self.assertTrue(turn_tail(saved.items).unfinished)
        self.assertTrue(carried["continues"])
        self.assertEqual(carried["config"], {"auto_compaction_context_tokens": 5000})
        self.assertEqual(carried["draft"], {"text": "half typed", "cursor": 10})
        self.assertEqual(carried["queue"], [
            {"kind": "text", "text": "Then this.", "steers": False,
             "message": {"type": "message", "role": "user", "content": "Then this."}},
            {"kind": "command", "name": "config",
             "arguments_json": '{"key":"max_samples","value":7}'}])
        self.assertIn("Rebooting: the CLI restarts on the current code and continues the turn.",
                      _notices(terminal.items))

    async def test_an_exit_after_a_reboot_request_wins(self):
        probe, entered, release = _blocking_probe()

        def type_during_the_tool(terminal):
            terminal.submit("/reboot")
            terminal.submit("/exit")

        terminal = _Terminal(_steps((lambda t, s: entered.is_set(), type_during_the_tool),
                                    (lambda t, s: s.startswith("closing"), lambda t: release.set())))
        model = _Model(self.path, ModelSample((ToolCall("probe", "p1", "{}"),)))
        try:
            with _rebootable():
                self.assertEqual(await self._session(
                    model, terminal, ["--prompt", "go"], Environment((probe,))), 0)
        finally:
            release.set()
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(any(text.startswith("Rebooting:") for text in _notices(terminal.items)))

    async def test_a_coldstart_compaction_reboots_with_the_option_once_steers_are_saved(self):
        self._resumable()
        gate = threading.Event()
        compactor = _Compactor(gate)
        terminal = _Terminal(_steps(
            (lambda t, s: compactor.entered.is_set(), lambda t: t.submit("Note this.")),
            (lambda t, s: "steers=1" in s, lambda t: gate.set())))
        model = _Model(self.path)
        try:
            with _rebootable(), patch_compaction(cli, "create_default_compactor",
                                                 return_value=compactor):
                carried = await self._rebooted(model, terminal, [
                    "--resume", "--prompt", "follow up", "--auto-compaction-context-tokens",
                    "100", "--compaction-coldstart-reboot"])
        finally:
            gate.set()
        self.assertEqual(model.calls, [])  # the restarted CLI samples
        self.assertEqual(compactor.calls, 1)
        self.assertEqual((carried["cause"], carried["continues"]), ("coldstart", True))
        items = load_interaction_save(self.path).items
        self.assertIsInstance(items[-3], ContextPrefix)
        self.assertIsInstance(items[-2], CompactionMetadata)
        self.assertEqual(items[-1], Message("user", "Note this."))

    async def test_without_the_option_a_coldstart_compaction_goes_on(self):
        self._resumable()
        model = _Model(self.path, _answer("after"))
        terminal = _Terminal(_steps((_idle_with("[assistant] after"), _exit)))
        with _rebootable(), patch_compaction(cli, "create_default_compactor",
                                             return_value=_Compactor()):
            self.assertEqual(await self._session(model, terminal, [
                "--resume", "--prompt", "follow up", "--auto-compaction-context-tokens",
                "100"]), 0)
        self.assertEqual(len(model.calls), 1)

    async def test_a_failed_start_check_cancels_the_reboot_and_the_turn_continues_here(self):
        self._resumable()
        model = _Model(self.path, _answer("continued here"))
        terminal = _Terminal(_steps((_idle_with("[assistant] continued here"), _exit)))
        failure = "the restarted CLI would fail to start (its start check exited with status 1)"
        with _rebootable(check=failure), patch_compaction(cli, "create_default_compactor",
                                                          return_value=_Compactor()):
            self.assertEqual(await self._session(model, terminal, [
                "--resume", "--prompt", "follow up", "--auto-compaction-context-tokens", "100",
                "--compaction-coldstart-reboot"]), 0)
        self.assertIn(f"Reboot cancelled: {failure}. The turn continues here.",
                      _notices(terminal.items))
        self.assertEqual(len(model.calls), 1)
        saved = load_interaction_save(self.path)
        self.assertEqual(_users(saved), ["old request", "follow up"])  # no message continued it
        self.assertIsInstance(saved.items[-1], TurnSummary)

    async def test_a_start_check_that_raises_cancels_the_reboot_once(self):
        model = _Model(self.path, _answer("done"))
        terminal = _Terminal(_steps(
            (_idle_with("[assistant] done"), lambda t: t.submit("/reboot")),
            (_idle_with("Reboot cancelled"), _exit)))
        with _rebootable(), mock.patch.object(cli, "_check_reboot",
                                              side_effect=RuntimeError("bug")) as check:
            self.assertEqual(await self._session(model, terminal, ["--prompt", "hello"]), 0)
        check.assert_called_once()
        self.assertIn("Reboot cancelled: the restarted CLI's start check failed (RuntimeError).",
                      _notices(terminal.items))

    async def test_icron_jobs_keep_a_coldstart_compaction_from_rebooting(self):
        self._resumable()
        icron = Icron()
        self.addCleanup(icron.close, 1.0)
        icron.schedule("ping", after=600, by="user")
        model = _Model(self.path, _answer("after"))
        terminal = _Terminal(_steps((_idle_with("[assistant] after"), _exit)))
        with _rebootable(), patch_compaction(cli, "create_default_compactor",
                                             return_value=_Compactor()):
            self.assertEqual(await self._session(model, terminal, [
                "--resume", "--prompt", "follow up", "--auto-compaction-context-tokens", "100",
                "--compaction-coldstart-reboot"], icron=icron), 0)
        self.assertIn("compaction_coldstart_reboot: not rebooting, since icron has 1 pending "
                      "job, and jobs live only in this process (/cron lists them).",
                      _notices(terminal.items))
        self.assertEqual(len(model.calls), 1)

    async def test_compact_reboots_with_the_option_once_it_is_saved(self):
        self._resumable()
        model = _Model(self.path)
        terminal = _Terminal(_steps((lambda t, s: s == "idle", lambda t: t.submit("/compact"))))
        with _rebootable(), patch_compaction(cli, "create_default_compactor",
                                             return_value=_Compactor()):
            carried = await self._rebooted(model, terminal, [
                "--resume", "--compaction-coldstart-reboot"])
        self.assertEqual((carried["cause"], carried["continues"]), ("coldstart", False))
        items = load_interaction_save(self.path).items
        self.assertEqual([type(item) for item in items[-4:-1]],
                         [UserToolCall, UserToolResult, ContextPrefix])


class RestartedCLITests(_RebootCase):
    def _unfinished(self):
        """A turn that a reboot ended after its tool batch."""
        original = (Init("saved"), Tools(), Message("user", "old query"),
                    UserInteractionBoundary(), ToolCall("probe", "p1", "{}"),
                    ModelSampleBoundary(), ToolResult("p1", "probed"))
        save_interaction_save(self.path, InteractionContext(original))
        return original

    async def test_it_continues_the_turn_with_what_it_carried_over(self):
        original = self._unfinished()
        rebooted = cli._Rebooted(
            cause="coldstart", continues=True, config={"auto_compaction_context_tokens": 777},
            draft=Editor("half typed", 4),
            queue=(cli._Submission("Then this.", Message("user", "Then this.")),),
            exit_code=1)
        model = _Model(self.path, _answer("continued"), _answer("then"))
        drafts = []

        def frame(terminal, editor, status):
            drafts.append(editor.text)
            if status == "idle" and _has(terminal, "[assistant] then"):
                terminal.key("c-d")

        terminal = _Terminal(frame)
        self.assertEqual(await self._session(model, terminal, rebooted=rebooted), 1)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0][0].items, original)  # continued: no new message
        self.assertEqual(model.calls[0][2].auto_compaction_context_tokens, 777)
        self.assertEqual(_users(model.calls[1][0].items), ["old query", "Then this."])
        self.assertIn("half typed", drafts)
        self.assertNotIn("[user] old query", _texts(terminal))  # its terminal shows the log
        notices = _notices(terminal.items)
        self.assertIn("Rebooted after a coldstart compaction (compaction_coldstart_reboot): "
                      "this is the current code; the turn continues.", notices)
        self.assertFalse(any(text.startswith("Note: resumed save") for text in notices))

    async def test_a_lost_state_still_resumes_without_continuing(self):
        self._unfinished()
        model = _Model(self.path)
        terminal = _Terminal(_steps((lambda t, s: s == "idle", _exit)))
        rebooted = cli._Rebooted(warning="its state couldn't be read (ValueError)")
        self.assertEqual(await self._session(model, terminal, rebooted=rebooted), 0)
        self.assertEqual(model.calls, [])
        notices = _notices(terminal.items)
        self.assertIn("Note: this CLI restarted for a reboot, but its state couldn't be read "
                      "(ValueError): the draft, queued input, and /config changes were not "
                      "carried over.", notices)
        self.assertTrue(any("enter /continue to sample from here" in text for text in notices))

    async def test_a_headless_run_reboots_and_its_restart_finishes_the_task(self):
        self._resumable()
        args = cli._build_parser().parse_args([
            "--headless", "--resume", "--prompt", "follow up",
            "--auto-compaction-context-tokens", "100", "--compaction-coldstart-reboot"])
        model = _Model(self.path)
        with _rebootable(), patch_compaction(cli, "create_default_compactor",
                                             return_value=_Compactor()), \
                redirect_stderr(io.StringIO()) as stderr, \
                self.assertRaises(cli._Rebooting) as caught:
            await asyncio.wait_for(cli._run_headless(model, Environment(), args, self.path), 6)
        self.assertEqual(model.calls, [])
        self.assertIn("[cli] Rebooting: the CLI restarts on the current code and continues the "
                      "turn.", stderr.getvalue())
        rebooted = cli._parse_reboot_state(caught.exception.state)
        restarted_args = cli._build_parser().parse_args([
            "--headless", "--resume", "--auto-compaction-context-tokens", "100"])
        model = _Model(self.path, _answer("finished"))
        with redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(await asyncio.wait_for(cli._run_headless(
                model, Environment(), restarted_args, self.path, rebooted=rebooted), 6), 0)
        self.assertEqual(len(model.calls), 1)
        self.assertIn("this is the current code; the turn continues.", stderr.getvalue())
        self.assertEqual(_users(load_interaction_save(self.path)), ["old request", "follow up"])


class RebootEntryPointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "interaction.jsonl"

    def test_rebooted_check_loads_and_parses_and_starts_nothing(self):
        with mock.patch.object(cli, "build_model", side_effect=AssertionError("model built")), \
                mock.patch.object(cli, "PosixTerminal", side_effect=AssertionError("TUI")):
            self.assertEqual(cli.main(["--save", str(self.path), "--prompt", "x",
                                       "--resume=False", "--rebooted=check"]), 0)
        self.assertFalse(self.path.exists())

    def test_main_runs_the_restarted_cli_once_the_session_has_closed(self):
        events = []

        async def run_headless(model, environment, args, path, **options):
            raise cli._Rebooting({"version": 1})

        def execve(path, argv, environment):
            events.append(("exec", argv, os.pread(int(environment[cli._REBOOT_STATE_FD]),
                                                  4096, 0)))

        model = SimpleNamespace(close=lambda: events.append(("close",)))
        with _rebootable(), mock.patch.object(cli, "build_model", return_value=model), \
                mock.patch.object(cli, "_run_headless", side_effect=run_headless), \
                mock.patch.object(cli.os, "execve", side_effect=execve):
            cli.main(["--headless", "--prompt", "x", "--save", str(self.path)])
        self.assertEqual(events, [("close",), ("exec", [*_COMMAND], b'{"version": 1}')])

    def test_a_failed_exec_says_how_to_continue(self):
        async def run_headless(model, environment, args, path, **options):
            raise cli._Rebooting({"version": 1})

        stderr = io.StringIO()
        with _rebootable(), mock.patch.object(cli, "build_model", return_value=object()), \
                mock.patch.object(cli, "_run_headless", side_effect=run_headless), \
                mock.patch.object(cli.os, "execve",
                                  side_effect=FileNotFoundError(2, "No such file")), \
                redirect_stderr(stderr):
            self.assertEqual(cli.main(["--headless", "--prompt", "x", "--save",
                                       str(self.path)]), 1)
        self.assertTrue(stderr.getvalue().endswith(
            f"interaction CLI failed: the reboot couldn't run {sys.executable} (No such file); "
            "rerun the command with --resume to continue from the save.\n"), stderr.getvalue())

    def test_the_restarted_cli_resumes_without_the_prompt_or_the_instructions(self):
        seen = {}

        async def run_headless(model, environment, args, path, **options):
            seen.update(args=args, **options)
            return 0

        state = cli._reboot_state(cli._UIState(reboot=cli._Reboot("/reboot", continues=True)),
                                  InteractionConfig())
        with tempfile.TemporaryFile() as file:
            file.write(json.dumps(state).encode())
            file.flush()
            descriptor = os.dup(file.fileno())
        with mock.patch.object(cli, "build_model", return_value=object()), \
                mock.patch.object(cli, "_run_headless", side_effect=run_headless), \
                mock.patch.dict(os.environ, {cli._REBOOT_STATE_FD: str(descriptor)}):
            self.assertEqual(cli.main([
                "--headless", "--prompt", "x", "--instructions", "y", "--resume=False",
                "--save", str(self.path), "--rebooted"]), 0)
            self.assertNotIn(cli._REBOOT_STATE_FD, os.environ)
        args = seen["args"]
        self.assertEqual((args.resume, args.prompt, args.instructions), (True, None, None))
        self.assertEqual(seen["rebooted"], cli._parse_reboot_state(state))


if __name__ == "__main__":
    unittest.main()
