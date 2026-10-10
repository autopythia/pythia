"""The CLI runs each turn on its context thread; terminal state stays on the loop."""

from __future__ import annotations

import asyncio
import collections
import functools
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pythia.interaction import Environment, Init, InteractionContext, Message, ModelSample
from pythia.interaction import Tool, ToolCall, ToolOutcome, ToolSpec
from pythia.interaction import InteractionSaveWriter
from pythia.interaction import cli
from pythia_test.interaction_helpers import patch_saves, real_save
from pythia.interaction._cli_editor import Editor
from pythia.interaction.loop import Urgency, kernel
from pythia.interaction.model import ModelTransportError
from pythia.interaction.runtime_config import InteractionConfig, InteractionConfigSnapshot


def _context():
    return InteractionContext((Init(model="scripted"), Message("user", "task")))


def _submission(text):
    return cli._Submission(text, Message("user", text))


class CliContextThreadTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_work_shares_one_context_thread_and_ui_stays_on_the_loop(self):
        loop_thread = threading.get_ident()
        seen = collections.defaultdict(set)

        class Displays(collections.deque):
            def extend(self, items):
                seen["display"].add(threading.get_ident())
                super().extend(items)

        class Model:
            calls = 0

            def sample(self, context, *, tools=(), sample_params=None):
                seen["sample"].add(threading.get_ident())
                Model.calls += 1
                if Model.calls == 1:
                    return ModelSample((ToolCall(name="probe", call_id="a", arguments_json="{}"),))
                return ModelSample((Message("assistant", "done"),))

        def handler(arguments, *, timeout_seconds=None):
            seen["tool"].add(threading.get_ident())
            return ToolOutcome(output="ok")
        environment = Environment((Tool(ToolSpec("probe", "Probe.", {"type": "object"}), handler),))
        def save(writer, context):
            seen["save"].add(threading.get_ident())
            return real_save(writer, context)
        state = cli._UIState(headless=True)
        state.displays = Displays()
        with tempfile.TemporaryDirectory() as directory, patch_saves(save):
            result = await cli._turn(_context(), Model(), environment, state,
                                     InteractionSaveWriter(Path(directory) / "log.jsonl"),
                                     InteractionConfig(InteractionConfigSnapshot()))
        self.assertEqual(result.final_text, "done")
        context_threads = seen["sample"] | seen["tool"] | seen["save"]
        self.assertEqual(len(context_threads), 1)
        self.assertNotIn(loop_thread, context_threads)
        # Every display change ran on the event loop, before the turn returned.
        self.assertEqual(seen["display"], {loop_thread})
        self.assertTrue(any("done" in item.text for item in state.displays))
        self.assertEqual(state.retry, None)

    async def test_a_sample_failure_caused_by_exit_is_a_stop_without_retry(self):
        state = cli._UIState(headless=True)

        class Model:
            def sample(self, context, *, tools=(), sample_params=None):
                state.closing = True  # exit retired the model while it waited
                raise ModelTransportError("Claude Relay continuation was retired during sampling")
        with tempfile.TemporaryDirectory() as directory:
            result = await cli._turn(_context(), Model(), Environment(), state,
                                     InteractionSaveWriter(Path(directory) / "log.jsonl"),
                                     InteractionConfig(InteractionConfigSnapshot()))
        self.assertEqual(result.kind, "stopped")
        self.assertIsNone(state.retry)

    async def test_a_sampling_failure_arms_retry_on_the_loop(self):
        state = cli._UIState(headless=True)

        class Model:
            def sample(self, context, *, tools=(), sample_params=None):
                raise ModelTransportError("lost")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ModelTransportError):
                await cli._turn(_context(), Model(), Environment(), state,
                                InteractionSaveWriter(Path(directory) / "log.jsonl"),
                                InteractionConfig(InteractionConfigSnapshot()))
        self.assertIsInstance(state.retry, cli._RetryIntent)



class CliSteeringTests(unittest.IsolatedAsyncioTestCase):
    async def _turn(self, state, model, environment=None):
        with tempfile.TemporaryDirectory() as directory:
            return await cli._turn(_context(), model, environment or Environment(), state,
                                   InteractionSaveWriter(Path(directory) / "log.jsonl"),
                                   InteractionConfig(InteractionConfigSnapshot()),
                                   steering=True)

    def _model(self, seen):
        class Model:
            calls = 0

            def sample(self, context, *, tools=(), sample_params=None):
                Model.calls += 1
                seen.append([item.content for item in context.items
                             if isinstance(item, Message) and item.role == "user"])
                if Model.calls == 1:
                    return ModelSample((ToolCall(name="probe", call_id="a", arguments_json="{}"),))
                return ModelSample((Message("assistant", "done"),))
        return Model()

    async def test_text_submitted_during_a_turn_steers_its_next_sample(self):
        loop, state, seen = asyncio.get_running_loop(), cli._UIState(headless=True), []

        def handler(arguments, *, timeout_seconds=None):
            loop.call_soon_threadsafe(state.pending.append, _submission("also check the tests"))
            return ToolOutcome(output="ok")
        environment = Environment((Tool(ToolSpec("probe", "Probe.", {"type": "object"}), handler),))
        result = await self._turn(state, self._model(seen), environment)
        self.assertEqual(result.final_text, "done")
        self.assertEqual(seen, [["task"], ["task", "also check the tests"]])
        self.assertEqual(list(state.pending), [])
        self.assertTrue(any("also check the tests" in item.text for item in state.displays))
        self.assertFalse(state.turn_active)

    async def test_older_queued_input_is_never_overtaken(self):
        loop, state, seen = asyncio.get_running_loop(), cli._UIState(headless=True), []
        state.pending.append(_submission("older query"))

        def handler(arguments, *, timeout_seconds=None):
            loop.call_soon_threadsafe(state.pending.append, _submission("typed later"))
            return ToolOutcome(output="ok")
        environment = Environment((Tool(ToolSpec("probe", "Probe.", {"type": "object"}), handler),))
        await self._turn(state, self._model(seen), environment)
        self.assertEqual(seen, [["task"], ["task"]])
        self.assertEqual([entry.text for entry in state.pending], ["older query", "typed later"])

    async def test_a_bad_attachment_rejects_only_its_own_draft(self):
        # Enter reads a draft's @ files; a bad one keeps that draft, and the
        # submissions after it steer as usual.
        loop, state, seen = asyncio.get_running_loop(), cli._UIState(ready=True), []
        drafts = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text("from the file\n", encoding="utf-8")
            args = SimpleNamespace(enable_experimental_media=True, enable_workspace=True,
                                   model_api="responses")
            state.read_query = functools.partial(cli._query_message, args=args, cwd=root)
            state.reads_files = functools.partial(cli._reads_files, args=args)

            async def enter(text):
                state.editor = Editor(text, len(text))
                state.handle_key("c-m", "\r")
                while state.reading is not None:
                    await asyncio.sleep(0.001)
                drafts.append(state.editor.text)

            def handler(arguments, *, timeout_seconds=None):
                for text in ("@missing.png bad", "@notes.txt good steer", "plain steer"):
                    asyncio.run_coroutine_threadsafe(enter(text), loop).result()
                (root / "notes.txt").write_text("changed later\n", encoding="utf-8")
                return ToolOutcome(output="ok")
            environment = Environment((Tool(ToolSpec("probe", "Probe.", {"type": "object"}),
                                            handler),))
            result = await self._turn(state, self._model(seen), environment)
        self.assertEqual(result.final_text, "done")
        self.assertEqual(drafts, ["@missing.png bad", "", ""])  # only the bad draft stays
        self.assertEqual(seen, [["task"], ["task", "from the file\n\ngood steer", "plain steer"]])
        self.assertEqual(list(state.pending), [])
        notices = [item.text for item in state.displays if item.text.startswith("[cli]")]
        self.assertEqual(len(notices), 1)
        self.assertIn("missing.png", notices[0])

    async def test_enter_during_a_read_keeps_the_draft_and_the_order(self):
        state, gate = cli._UIState(ready=True), threading.Event()

        def read(text):
            gate.wait(5)
            return Message("user", text.upper())
        state.read_query, state.reads_files = read, lambda text: text.startswith("@")
        for text in ("@slow", "second"):
            state.editor = Editor(text, len(text))
            state.handle_key("c-m", "\r")
        self.assertEqual(state.reading, "@slow")
        self.assertEqual(state.editor.text, "second")  # kept, not queued
        self.assertEqual(list(state.pending), [])
        self.assertIn("A submission is still pending. Draft preserved.",
                      [item.text.removeprefix("[cli] ") for item in state.displays])
        gate.set()
        while state.reading is not None:
            await asyncio.sleep(0.001)
        self.assertEqual([entry.message.content for entry in state.pending], ["@SLOW"])
        self.assertEqual(state.editor.text, "second")  # edited meanwhile: not cleared

    def test_steers_stop_at_the_first_command(self):
        state = cli._UIState(headless=True)
        retry = cli._RetryIntent()
        first, second, after = map(_submission, ("first", "second", "after"))
        state.pending.extend([first, second, retry, after])
        self.assertEqual(cli._steer_count(state), 0)  # no turn is running
        state.turn_active = True
        self.assertEqual(cli._steer_count(state), 2)
        self.assertEqual(cli._take_steers(state), (first, second))
        self.assertEqual(list(state.pending), [retry, after])
        self.assertEqual(cli._take_steers(state), ())

    async def test_steer_now_cancels_a_sample_and_delivers_it_next(self):
        loop, state, seen = asyncio.get_running_loop(), cli._UIState(ready=True), []
        retired = threading.Event()

        async def enter(text):
            state.editor = Editor(text, len(text))
            state.handle_key("c-m", "\r")

        class Model:
            calls = retires = 0

            def retire(self):
                Model.retires += 1
                retired.set()

            def sample(self, context, *, tools=(), sample_params=None):
                Model.calls += 1
                seen.append([item.content for item in context.items
                             if isinstance(item, Message) and item.role == "user"])
                if Model.calls == 1:
                    asyncio.run_coroutine_threadsafe(enter("/steer!! stop that"), loop).result()
                    if not retired.wait(5):
                        raise AssertionError("the steer did not retire the model")
                    raise ModelTransportError("Claude Relay continuation retired")
                return ModelSample((Message("assistant", "done"),))
        result = await self._turn(state, Model())
        self.assertEqual(result.final_text, "done")
        self.assertEqual(seen, [["task"], ["task", "stop that"]])
        self.assertEqual(Model.retires, 2)  # the cancel, then the turn's end
        self.assertIsNone(state.retry)  # a cancelled sample is not a failure
        self.assertEqual(state.preemption.level, Urgency.QUEUED)
        self.assertIn("[cli] Flushing 1 steer: delivered as soon as the current sample or tool "
                      "call ends; a sample or command wait is cancelled where possible.",
                      [item.text for item in state.displays])


class CliSteerInputTests(unittest.TestCase):
    def _enter(self, state, text):
        state.editor = Editor(text, len(text))
        state.handle_key("c-m", "\r")

    def _notices(self, state):
        return [item.text.removeprefix("[cli] ") for item in state.displays]

    def test_steer_without_text_flushes_the_queued_steers(self):
        state = cli._UIState(ready=True, turn_active=True)
        self._enter(state, "/steer")
        self.assertEqual(self._notices(state), ["No queued steers to flush."])
        self.assertEqual(state.preemption.level, Urgency.QUEUED)
        self._enter(state, "first")
        self._enter(state, "second")
        self._enter(state, "/steer!")
        self.assertEqual(state.preemption.level, Urgency.IMMEDIATE)
        self._enter(state, "/steer!!!")  # more than two ! read as two
        self.assertEqual(state.preemption.level, Urgency.PREEMPT)
        self.assertEqual([entry.text for entry in state.pending], ["first", "second"])
        self.assertEqual(state.editor.text, "")
        self.assertEqual(cli._take_steers(state)[-1].text, "second")
        self.assertEqual(state.preemption.level, Urgency.QUEUED)  # delivered: reset

    def test_bare_steer_flushes_nothing_and_reports_when_the_steers_arrive(self):
        state = cli._UIState(ready=True, turn_active=True)
        self._enter(state, "first")
        self._enter(state, "/steer")
        self.assertEqual(state.preemption.level, Urgency.QUEUED)
        self._enter(state, "second")
        self._enter(state, "/steer")
        self._enter(state, "/steer!")
        self._enter(state, "/steer")  # the urgency only rises: it stays immediate
        self.assertEqual(state.preemption.level, Urgency.IMMEDIATE)
        self.assertEqual(self._notices(state), [
            "1 steer queued for the next sample, after the current tool batch; "
            "/steer! or /steer!! delivers it sooner.",
            "2 steers queued for the next sample, after the current tool batch; "
            "/steer! or /steer!! delivers them sooner.",
            "Flushing 2 steers: delivered once the current sample or tool call finishes.",
            "Flushing 2 steers: delivered once the current sample or tool call finishes."])

    def test_steer_with_text_queues_it_and_flushes_with_the_earlier_steers(self):
        state = cli._UIState(ready=True, turn_active=True)
        self._enter(state, "earlier")
        self._enter(state, "/steer!! and now")
        self.assertEqual([entry.text for entry in state.pending], ["earlier", "and now"])
        self.assertEqual(state.pending[1].message, Message("user", "and now"))
        self.assertEqual(state.preemption.level, Urgency.PREEMPT)
        self.assertEqual(state.editor.text, "")

    def test_steer_with_text_and_no_bang_is_plain_text(self):
        state = cli._UIState(ready=True, turn_active=True)
        self._enter(state, "/steer /tmp/notes.md has the details")  # text starting with /
        self.assertEqual([entry.text for entry in state.pending],
                         ["/tmp/notes.md has the details"])
        self.assertEqual(state.preemption.level, Urgency.QUEUED)
        self.assertEqual(self._notices(state), [
            "1 steer queued for the next sample, after the current tool batch; "
            "/steer! or /steer!! delivers it sooner."])

    def test_steer_text_that_cannot_steer_queues_as_the_next_query(self):
        for turn_active, ahead in ((False, None), (True, cli._RetryIntent())):
            with self.subTest(turn_active=turn_active):
                state = cli._UIState(ready=True, turn_active=turn_active)
                if ahead is not None:
                    state.pending.append(ahead)
                self._enter(state, "/steer later")
                self.assertEqual(state.pending[-1].text, "later")
                self.assertEqual(state.preemption.level, Urgency.QUEUED)
                self.assertIn("No running turn can take a steer now; queued as the next query.",
                              self._notices(state))

    def test_steers_during_compaction_say_they_come_right_after_it(self):
        # Compaction is neither skipped nor cancelled, whatever the level; the
        # flush still applies, in case the compaction has just ended.
        state = cli._UIState(ready=True, turn_active=True)
        state.set_phase("compacting")
        self._enter(state, "first")
        self._enter(state, "/steer")
        self._enter(state, "/steer!! second")
        self.assertEqual(state.preemption.level, Urgency.PREEMPT)
        self.assertEqual([entry.text for entry in state.pending], ["first", "second"])
        self.assertEqual(self._notices(state), [
            "1 steer queued for the next sample, right after the compaction in progress; "
            "steers never cancel compaction.",
            "2 steers queued for the next sample, right after the compaction in progress; "
            "steers never cancel compaction."])

    def test_a_steer_with_a_bad_attachment_flushes_nothing(self):
        state = cli._UIState(ready=True, turn_active=True)
        self._enter(state, "queued steer")

        def read(text):
            raise cli.AttachmentError("cannot read attachment: missing.png")
        state.read_query = read
        self._enter(state, "/steer!! @missing.png look")
        self.assertEqual(state.editor.text, "/steer!! @missing.png look")
        self.assertEqual([entry.text for entry in state.pending], ["queued steer"])
        self.assertEqual(state.preemption.level, Urgency.QUEUED)
        self.assertEqual(self._notices(state), ["cannot read attachment: missing.png"])


if __name__ == "__main__":
    unittest.main()
