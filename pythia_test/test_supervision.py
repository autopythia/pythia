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
