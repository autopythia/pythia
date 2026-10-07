"""Role-agnostic supervision primitives, independent of any auto session."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import json
from pathlib import Path
import unittest

from pythia.interaction import Message, ModelFailure
from pythia.interaction.loop.supervision import Fault, SupervisedHandle, Yield, YieldChannel, supervise
from pythia.interaction.loop.supervision import YIELD_MAX_CHARS, YieldTool, report_text
from pythia.interaction.loop.supervision import run_supervised_task
from pythia.interaction.loop import TurnHost
from pythia.interaction import Environment, Init, InteractionContext, ModelSample, ToolCall
from pythia.interaction import ToolResult, TurnSummary, UserInteractionBoundary
from pythia.interaction.runtime_config import InteractionConfigSnapshot
from pythia.interaction.save import SaveError


def make(kind="ended", resumable=True, **fields):
    fields.setdefault("revision", 0)
    return Yield(context=1, job_id=fields.pop("job_id", "1"), job_text="task",
                 kind=kind, resumable=resumable, **fields)


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for test condition")
        time.sleep(0.005)


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.stop = threading.Event()
        self.channel = YieldChannel(self.stop)
        self.bridge = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.bridge.shutdown, wait=False)
        self.answers = []

    def supervised(self, *yields):
        """Publish yields in order on a thread, then close the channel."""
        def run():
            for yield_, view in yields:
                self.answers.append(self.channel.signal(yield_, view))
            self.channel.close()
        thread = threading.Thread(target=run)
        thread.start()
        self.addCleanup(thread.join, 5)
        return thread

    def run_supervisor(self, coroutine):
        return asyncio.run(asyncio.wait_for(coroutine, 5))

    def test_resume_then_release_round_trip_with_log_view(self):
        items = (Message("user", "task"), Message("assistant", "partial"))
        thread = self.supervised((make(revision=2), items),
                                 (make("failed", reason="ModelTransportError", resumes=1), items))
        handle = SupervisedHandle(self.channel, self.bridge)

        async def supervisor():
            first = await handle()
            self.assertEqual((first.kind, first.revision), ("ended", 2))
            self.assertEqual(handle.view, items)
            second = await handle("continue")
            self.assertEqual((second.kind, second.reason, second.resumes),
                             ("failed", "ModelTransportError", 1))
            return await handle(None)

        self.assertIsNone(self.run_supervisor(supervisor()))
        thread.join(5)
        self.assertEqual(self.answers, ["continue", None])
        self.assertEqual(handle.view, ())

    def test_stop_wakes_a_waiting_supervised_side_and_later_yields_only_notify(self):
        result = []
        thread = threading.Thread(target=lambda: result.append(self.channel.signal(make())))
        thread.start()
        wait_for(lambda: self.channel._queue.qsize() == 1)
        self.assertTrue(thread.is_alive())
        self.stop.set()
        self.channel.wake()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [None])
        self.assertIsNone(self.channel.signal(make(job_id="2")))
        self.assertEqual(self.channel._queue.qsize(), 2)

    def test_fault_is_raised_and_the_handle_stays_usable(self):
        failure = ModelFailure("save", "safe summary")
        self.supervised((make("failed", False, reason="SaveError", failure=failure), ()))
        handle = SupervisedHandle(self.channel, self.bridge)

        async def supervisor():
            with self.assertRaises(Fault) as raised:
                await handle()
            self.assertEqual(raised.exception.yield_.failure, failure)
            self.assertEqual(str(raised.exception), "#1 failed (SaveError)")
            with self.assertRaisesRegex(ValueError, "non-resumable"):
                await handle("resume anyway")
            return await handle(None)

        self.assertIsNone(self.run_supervisor(supervisor()))
        self.assertEqual(self.answers, [None])

    def test_cancelled_calls_lose_no_yield(self):
        def late():
            time.sleep(0.1)
            self.answers.append(self.channel.signal(make(job_id="late")))
            self.channel.close()
        thread = threading.Thread(target=late)
        thread.start()
        handle = SupervisedHandle(self.channel, self.bridge)

        async def supervisor():
            timeouts = 0
            while True:
                try:
                    yield_ = await asyncio.wait_for(handle(), 0.01)
                    break
                except asyncio.TimeoutError:
                    timeouts += 1
            return timeouts, yield_, await handle(None)

        timeouts, yield_, end = self.run_supervisor(supervisor())
        thread.join(5)
        self.assertGreater(timeouts, 0)
        self.assertEqual(yield_.job_id, "late")
        self.assertIsNone(end)
        self.assertEqual(self.answers, [None])

    def test_one_caller_at_a_time_and_resume_requires_a_yield(self):
        handle = SupervisedHandle(self.channel, self.bridge)

        async def supervisor():
            with self.assertRaisesRegex(ValueError, "no yield"):
                await handle("too early")
            first = asyncio.ensure_future(handle())
            await asyncio.sleep(0)
            with self.assertRaisesRegex(RuntimeError, "already has a caller"):
                await handle()
            self.channel.close()
            return await first

        self.assertIsNone(self.run_supervisor(supervisor()))

    def test_supervise_decides_observes_faults_and_always_detaches(self):
        thread = self.supervised(
            (make("failed", reason="ModelTransportError"), ()),
            (make(resumes=1), ()),
            (make("failed", False, reason="SaveError", resumes=1), ()),
            (make(job_id="2"), ()),  # after the fault propagated: notification only
        )
        handle = SupervisedHandle(self.channel, self.bridge)
        decided, faults = [], []

        async def decide(yield_):
            decided.append(yield_.kind)
            return "retry" if yield_.kind == "failed" else None

        with self.assertRaises(Fault):
            self.run_supervisor(supervise(handle, decide, on_fault=faults.append))
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(decided, ["failed", "ended"])
        self.assertEqual(self.answers, ["retry", None, None, None])
        self.assertEqual([fault.yield_.reason for fault in faults], ["SaveError"])

    def test_supervise_returns_when_the_supervised_side_exits(self):
        self.supervised((make(), ()))
        handle = SupervisedHandle(self.channel, self.bridge)

        async def decide(yield_):
            return None

        self.assertIsNone(self.run_supervisor(supervise(handle, decide)))
        self.assertEqual(self.answers, [None])
        self.assertIsNone(self.channel.signal(make(job_id="after")))  # detached: no wait

    def test_yield_validation(self):
        for fields, error in (({"kind": "halted"}, ValueError),
                              ({"resumable": 1}, TypeError),
                              ({"revision": -1}, ValueError),
                              ({"resumes": True}, ValueError),
                              ({"failure": "not safe"}, TypeError)):
            with self.subTest(fields=fields), self.assertRaises(error):
                make(**{"kind": "ended", "resumable": True, **fields})
        with self.assertRaises(TypeError):
            self.channel.signal("not a yield")



class HandoffTests(unittest.TestCase):
    def _yield(self, **fields):
        base = dict(context=1, job_id="3", job_text="Trace it.", kind="yielded",
                    resumable=True, revision=74, yield_text="Please check the ordering.")
        base.update(fields)
        return Yield(**base)

    def test_yielded_reports_carry_only_a_note(self):
        self.assertEqual(self._yield().yield_text, "Please check the ordering.")
        for fields in ({"yield_text": None}, {"yield_text": "  "},
                       {"final_text": "done"}, {"reason": "X"},
                       {"failure": ModelFailure("transport", "down")}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self._yield(**fields)
        with self.assertRaises(ValueError):
            self._yield(kind="ended", final_text="done")  # a note only on yielded

    def test_report_text_names_the_handoff_and_the_note(self):
        text = report_text(self._yield())
        self.assertEqual(text.splitlines()[0],
                         "Main (#1) handed off task 3 (watcher resumes so far: 0).")
        self.assertIn("Outcome: yielded", text)
        self.assertIn("Main's handoff note:\nPlease check the ordering.", text)
        self.assertNotIn("final answer", text)
        self.assertTrue(text.endswith("Main's log has 74 items (read_context indices 0..73)."))

    def test_report_text_quotes_the_first_message_and_steer_target_only_when_distinct(self):
        plain = report_text(self._yield())
        self.assertNotIn("First user message", plain)
        for text in ("Trace it.", " "):  # equal to the request, or blank: nothing added
            self.assertEqual(report_text(self._yield(first_user_text=text,
                                                     steer_target_text=text)), plain)
        why = "Main never received that steer, so it was queued as a separate request"
        both = report_text(self._yield(first_user_text="Build it.", steer_target_text="Test it."))
        self.assertIn("(watcher resumes so far: 0).\n\n"
                      "First user message of the session (for context):\nBuild it.\n\n"
                      f"Earlier request (the user sent the request below to steer it; {why}):\n"
                      "Test it.\n\nUser request:\nTrace it.\n\nOutcome: yielded", both)
        # A steer target equal to the quoted first message is noted, not repeated.
        noted = report_text(self._yield(first_user_text="Build it.", steer_target_text="Build it."))
        self.assertEqual(noted.count("Build it."), 1)
        self.assertIn("Build it.\n\nThe user sent the request below to steer the message "
                      f"above; {why}.\n\nUser request:\nTrace it.\n", noted)
        target_only = report_text(self._yield(steer_target_text="Test it."))
        self.assertNotIn("First user message", target_only)
        self.assertIn(f"steer it; {why}):\nTest it.\n\nUser request:\nTrace it.", target_only)
        for fields in ({"first_user_text": 1}, {"steer_target_text": b"Test it."}):
            with self.subTest(fields=fields), self.assertRaises(TypeError):
                self._yield(**fields)

    def test_report_text_cuts_long_context_but_not_the_request(self):
        from pythia.interaction.loop.supervision import READ_ITEM_CHARS
        long_first, long_target = "F" * (READ_ITEM_CHARS + 5), "T" * (READ_ITEM_CHARS + 7)
        text = report_text(self._yield(job_text="R" * (READ_ITEM_CHARS + 9),
                                       first_user_text=long_first,
                                       steer_target_text=long_target))
        self.assertIn("F" * READ_ITEM_CHARS + "\n[5 more characters omitted]\n\n", text)
        self.assertIn("T" * READ_ITEM_CHARS + "\n[7 more characters omitted]\n\n", text)
        self.assertIn("User request:\n" + "R" * (READ_ITEM_CHARS + 9) + "\n", text)
        self.assertNotIn("F" * (READ_ITEM_CHARS + 1), text)
        # Distinctness is judged on the whole texts, before cutting.
        same = report_text(self._yield(job_text=long_first, first_user_text=long_first))
        self.assertNotIn("First user message", same)

    def test_yield_tool_records_one_valid_note_per_turn(self):
        binding = YieldTool()
        (tool,) = binding.tools()
        self.assertEqual(tool.spec.name, "yield")
        with self.assertRaisesRegex(ValueError, "No turn is open"):
            tool.handler({"content": "early"})
        binding.begin()
        for arguments in ({}, {"content": " "}, {"content": 1}, {"content": "x", "extra": 1},
                          {"content": "x" * (YIELD_MAX_CHARS + 1)}, {"content": "\ud800"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                tool.handler(arguments)
        self.assertIsNone(binding.request())
        self.assertTrue(tool.handler({"content": "first"}).success)
        with self.assertRaisesRegex(ValueError, "already recorded"):
            tool.handler({"content": "second"})
        self.assertEqual(binding.request(), "first")
        self.assertEqual(binding.end(), "first")
        self.assertIsNone(binding.request())  # cleared for the next turn
        with self.assertRaisesRegex(ValueError, "No turn is open"):
            tool.handler({"content": "late"})

    def test_recorded_in_finds_a_successful_yield_in_the_last_turn(self):
        note = json.dumps({"content": "Decide?"})
        start = (Init(model="m"), Message("user", "task"), UserInteractionBoundary())
        yielded = (ToolCall("yield", "y1", note), ToolResult("y1", "Recorded."))
        self.assertTrue(YieldTool.recorded_in((*start, *yielded)))
        self.assertTrue(YieldTool.recorded_in((*start, *yielded, TurnSummary())))
        self.assertTrue(YieldTool.recorded_in((*start, *yielded, Message("user", "a steer"))))
        self.assertFalse(YieldTool.recorded_in(start))
        self.assertFalse(YieldTool.recorded_in(
            (*start, ToolCall("yield", "y1", note), ToolResult("y1", "Bad.", success=False))))
        self.assertFalse(YieldTool.recorded_in(
            (*start, ToolCall("probe", "p1", "{}"), ToolResult("p1", "ok"))))
        # A follow-up or a new task starts a new turn, after a boundary.
        self.assertFalse(YieldTool.recorded_in(
            (*start, *yielded, TurnSummary(), Message("user", "next"), UserInteractionBoundary())))



class SupervisedTaskLoopTests(unittest.TestCase):
    """run_supervised_task with a scripted model, a recording host, and a fake
    channel: the supervised side, exercised without auto."""

    class Host(TurnHost):
        def __init__(self, fail_on_append=None):
            self.phases, self.appends, self.fail_on_append = [], 0, fail_on_append

        def append(self, context, items):
            self.appends += 1
            if self.appends == self.fail_on_append:
                raise SaveError("injected")
            context.extend(items)

        def phase(self, phase):
            self.phases.append(phase)

    class Model:
        def __init__(self, *script):
            self.script, self.contexts = list(script), []

        def sample(self, context, *, tools=(), sample_params=None):
            self.contexts.append(context)
            return self.script.pop(0)

    class Channel:
        def __init__(self, *answers):
            self.answers, self.yields = list(answers), []

        def signal(self, yield_, view=()):
            self.yields.append(yield_)
            return self.answers.pop(0)

    def _run(self, model, channel, host=None, **options):
        binding = YieldTool()
        settled = []
        context = InteractionContext((Init(model="scripted"),))
        result = run_supervised_task(
            "Trace it.", context, model, Environment(binding.tools()),
            InteractionConfigSnapshot(), host or self.Host(), channel, job_id="7",
            control=binding, follow_up=lambda text: "Follow-up: " + text,
            settle=settled.append, waiting_phase="awaiting test", **options)
        return result, settled, context

    def test_yield_resume_then_release_after_a_final_answer(self):
        note = json.dumps({"content": "Please check."})
        model = self.Model(ModelSample((ToolCall("yield", "y1", note),)),
                           ModelSample((Message("assistant", "done"),)))
        channel = self.Channel("Checked; go on.", None)
        host = self.Host()
        result, settled, context = self._run(model, channel, host,
                                             steers=lambda: (("also this", True),))
        self.assertEqual([y.kind for y in channel.yields], ["yielded", "ended"])
        first, last = channel.yields
        self.assertEqual((first.job_id, first.job_text, first.yield_text, first.resumes),
                         ("7", "Trace it.", "Please check.", 0))
        self.assertEqual((last.final_text, last.resumes), ("done", 1))
        self.assertEqual(last.steers, (("also this", True),))
        self.assertIs(result, last)
        self.assertEqual(settled, [last])
        self.assertEqual(host.phases.count("awaiting test"), 2)
        users = [i.content for i in model.contexts[1].items
                 if isinstance(i, Message) and i.role == "user"]
        self.assertEqual(users, ["Trace it.", "Follow-up: Checked; go on."])
        self.assertEqual(context.pending_tool_calls(), ())

    def test_a_non_resumable_failure_is_published_once_then_settled(self):
        faults = []
        model = self.Model(ModelSample((Message("assistant", "unsaved"),)))
        channel = self.Channel("ignored")
        result, settled, _ = self._run(model, channel, self.Host(fail_on_append=2),
                                       on_fault=faults.append)
        self.assertEqual((result.kind, result.reason, result.resumable),
                         ("failed", "SaveError", False))
        self.assertEqual(faults, [result])
        self.assertEqual(channel.yields, [result])
        self.assertEqual(settled, [result])

    def test_stopping_ends_the_task_even_after_a_resume(self):
        model = self.Model(ModelSample((Message("assistant", "draft"),)))
        channel = self.Channel("More.")
        result, settled, _ = self._run(model, channel, stopping=lambda: True)
        self.assertEqual(result.kind, "ended")
        self.assertEqual(len(channel.yields), 1)
        self.assertEqual(settled, [result])

    def test_a_continued_task_starts_without_a_user_message(self):
        model = self.Model(ModelSample((Message("assistant", "draft"),)),
                           ModelSample((Message("assistant", "done"),)))
        channel = self.Channel("More.", None)
        self._run(model, channel, continued=True)
        users = [[item.content for item in context.items
                  if isinstance(item, Message) and item.role == "user"]
                 for context in model.contexts]
        self.assertEqual(users, [[], ["Follow-up: More."]])
        self.assertEqual([(y.continued, y.resumes) for y in channel.yields],
                         [(True, 0), (True, 1)])
        self.assertIn("Main continued an unfinished turn", report_text(channel.yields[0]))
        with self.assertRaises(TypeError):
            Yield(context=1, job_id="1", job_text="x", kind="ended", resumable=True,
                  revision=0, final_text="ok", continued=1)

    def test_every_yield_of_the_task_carries_the_context_texts(self):
        texts = {"first_user_text": "Build it.", "steer_target_text": "Plan it."}
        model = self.Model(ModelSample((Message("assistant", "draft"),)),
                           ModelSample((Message("assistant", "done"),)))
        channel = self.Channel("More.", None)
        self._run(model, channel, **texts)
        self.assertEqual([(y.first_user_text, y.steer_target_text) for y in channel.yields],
                         [("Build it.", "Plan it.")] * 2)
        failed, _, _ = self._run(self.Model(ModelSample((Message("assistant", "x"),))),
                                 self.Channel("ignored"), self.Host(fail_on_append=2), **texts)
        self.assertEqual((failed.kind, failed.first_user_text, failed.steer_target_text),
                         ("failed", "Build it.", "Plan it."))

    def test_settle_runs_even_if_the_channel_raises(self):
        class Broken:
            def signal(self, yield_, view=()):
                raise RuntimeError("transport")
        model = self.Model(ModelSample((Message("assistant", "done"),)))
        settled = []
        with self.assertRaises(RuntimeError):
            run_supervised_task("Task.", InteractionContext((Init(model="m"),)), model,
                                Environment(), InteractionConfigSnapshot(), self.Host(),
                                Broken(), settle=settled.append)
        self.assertEqual([y.kind for y in settled], ["ended"])


class LoopPackageTests(unittest.TestCase):
    def test_public_names_and_one_way_dependencies(self):
        import ast
        import pythia.interaction as interaction
        import pythia.interaction.loop as loop
        for name in loop.__all__:
            self.assertTrue(hasattr(loop, name), name)
        for name in loop.__all__:  # re-exported, like local_tools
            self.assertIs(getattr(interaction, name), getattr(loop, name))
        for path in Path(loop.__file__).parent.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    self.assertFalse(module.split(".")[0] in {"auto", "cli", "demo"}
                                     or module.startswith("pythia.interaction.auto"),
                                     f"{path.name} imports {module}")
                    self.assertFalse(node.level >= 2 and module in {"auto", "cli", "demo"},
                                     f"{path.name} imports {module}")


if __name__ == "__main__":
    unittest.main()
