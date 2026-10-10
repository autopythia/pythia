"""/exit!! cancels a Claude relay compaction wherever it is; steers wait for it.

These run the real turn loop, Preemption, PiCompactor, ClaudeRelayModel, and the
CLI's own input handling; only the relay's native process is faked
(``fake_relay_runtime``). A compaction makes one or two summary requests, each
its own native run, and spends time before and between them with no request in
flight, so a preempting stop must reach the model itself.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from pythia.interaction import CompactionMetadata, ContextPrefix, Environment, Init
from pythia.interaction import Instructions, InteractionContext, InteractionSaveWriter, Message
from pythia.interaction import ModelConfigurationError, ModelSampleBoundary, SampleMetadata
from pythia.interaction import TokenUsage, ToolCall, ToolResult, TurnSummary
from pythia.interaction import UserInteractionBoundary, UserToolCall, UserToolResult
from pythia.interaction import cli, compaction
from pythia.interaction import load_interaction_save, save_interaction_save
from pythia.interaction._cli_editor import Editor
from pythia.interaction.runtime_config import InteractionConfig, InteractionConfigSnapshot
from pythia_test.interaction_helpers import fake_relay_runtime, relay_model
from pythia_test.test_interaction_cli import _Terminal


# Compaction is due before every sample, and keeps only the last unit verbatim.
_CONFIG = InteractionConfig(InteractionConfigSnapshot(auto_compact_tokens=1,
                                                      compaction_keep_recent_tokens=1))
_CANCELLED = "Compaction cancelled during execution by the stop; no compaction checkpoint was installed."


def _one_request_context():
    """Due before a new request: the history is one summary request."""
    return InteractionContext((
        Init(model="fixture-model"), Instructions("Be brief."),
        Message("user", "older context " * 400), Message("assistant", "old answer"),
        ModelSampleBoundary(), Message("user", "two"),
    ))


def _two_request_context():
    """Due mid-turn: the cut splits the turn, so the history and the turn's
    prefix are two summary requests, one after the other."""
    return InteractionContext((
        Init(model="fixture-model"), Instructions("Be brief."),
        Message("user", "older context " * 400), Message("assistant", "old answer"),
        ModelSampleBoundary(), Message("user", "task " * 50),
        ToolCall(name="probe", call_id="c1", arguments_json="{}"), ModelSampleBoundary(),
        ToolResult("c1", "x" * 2000),
    ))


def _wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for test condition")
        time.sleep(0.001)


def _reached(model, epoch):
    """Wait until the stop's helper thread has acted on the model: closing it
    (or retiring it) starts a new epoch."""
    _wait_for(lambda: model._epoch != epoch)


def _assert_closed(test, model):
    """The stop closed the model, not only retired it: it refuses every request.
    Call it under fake_relay_runtime, so an open model never starts a native run."""
    with test.assertRaisesRegex(ModelConfigurationError, "closed"):
        model.sample(InteractionContext((Message("user", "after the stop"),)))


class TurnCompactionTests(unittest.IsolatedAsyncioTestCase):
    """Automatic compaction inside a CLI turn, on the relay."""

    async def _turn(self, state, model, context):
        state.active_model = model  # as _drive_interaction sets it
        with tempfile.TemporaryDirectory() as directory:
            return await cli._turn(context, model, Environment(), state,
                                   InteractionSaveWriter(Path(directory) / "log.jsonl"),
                                   _CONFIG, steering=True)

    @staticmethod
    def _enter(state, text):
        state.editor = Editor(text, len(text))
        state.handle_key("c-m", "\r")

    async def test_exit_cancels_the_summary_request_in_flight(self):
        model, state, context = relay_model(), cli._UIState(ready=True), _one_request_context()
        self.addCleanup(model.close)
        with fake_relay_runtime() as launches:
            turn = asyncio.ensure_future(self._turn(state, model, context))
            summary = await asyncio.to_thread(launches.summaries.get, timeout=5)
            self._enter(state, "/exit!!")
            result = await asyncio.wait_for(turn, 5)
            self.assertEqual(launches.all, [summary])  # nothing started after it
            _assert_closed(self, model)
        self.assertEqual(result.kind, "stopped")
        self.assertTrue(summary.closed.is_set())
        self.assertFalse(any(isinstance(item, ContextPrefix) for item in context.items))
        self.assertIsNone(state.retry)  # a stop, not a failure

    async def _exit_while_building(self, builder, context, *, answer_first=False):
        """/exit!! while the compactor builds a summary request's prompt: no
        request is in flight then, so only the model can refuse the next one."""
        loop = asyncio.get_running_loop()
        model, state = relay_model(), cli._UIState(ready=True)
        self.addCleanup(model.close)
        original = getattr(compaction, builder)

        async def exit_now():
            self._enter(state, "/exit!!")

        def stopping(*args, **kwargs):
            epoch = model._epoch
            asyncio.run_coroutine_threadsafe(exit_now(), loop).result(5)
            _reached(model, epoch)
            return original(*args, **kwargs)
        with fake_relay_runtime() as launches, \
                mock.patch.object(compaction, builder, side_effect=stopping):
            turn = asyncio.ensure_future(self._turn(state, model, context))
            if answer_first:
                first = await asyncio.to_thread(launches.summaries.get, timeout=5)
                first.answer("HISTORY SUMMARY")
            result = await asyncio.wait_for(turn, 10)
            # No summary request started after the stop.
            self.assertEqual(len(launches.all), 1 if answer_first else 0)
            _assert_closed(self, model)
        self.assertEqual(result.kind, "stopped")
        self.assertFalse(any(isinstance(item, ContextPrefix) for item in context.items))
        self.assertIsNone(state.retry)

    async def test_exit_before_the_first_summary_request_starts_none(self):
        await self._exit_while_building("_history_prompt", _one_request_context())

    async def test_exit_between_the_summary_requests_starts_no_second(self):
        await self._exit_while_building("_turn_prefix_prompt", _two_request_context(),
                                        answer_first=True)

    async def test_a_preempting_steer_waits_for_compaction_and_comes_right_after(self):
        model, state, context = relay_model(), cli._UIState(ready=True), _one_request_context()
        self.addCleanup(model.close)
        with fake_relay_runtime() as launches:
            turn = asyncio.ensure_future(self._turn(state, model, context))
            summary = await asyncio.to_thread(launches.summaries.get, timeout=5)
            self._enter(state, "/steer!! go left")
            await asyncio.sleep(0.1)
            self.assertFalse(summary.closed.is_set())  # the steer cancelled nothing
            summary.answer("SUMMARY")
            result = await asyncio.wait_for(turn, 5)
        self.assertEqual(result.final_text, "final answer")
        items = list(context.items)
        prefix = next(i for i, item in enumerate(items) if isinstance(item, ContextPrefix))
        self.assertGreater(items.index(Message("user", "go left")), prefix)
        sample = launches.all[-1]
        self.assertFalse(sample.summary)
        self.assertTrue(any("go left" in record for record in sample.snapshot.records))
        self.assertIn("[cli] 1 steer queued for the next sample, right after the compaction in "
                      "progress; steers never cancel compaction.",
                      [item.text for item in state.displays])


class CompactCommandTests(unittest.IsolatedAsyncioTestCase):
    """/compact runs outside a turn: /exit!! cancels it too, and its record says so."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "interaction.jsonl"
        save_interaction_save(self.path, InteractionContext((
            Init("fixture-model"), Message("user", "old request"), UserInteractionBoundary(),
            Message("assistant", "previous answer"), SampleMetadata(TokenUsage(total_tokens=10)),
            ModelSampleBoundary(), TurnSummary(sample_count=1),
        )))
        self.args = cli._build_parser().parse_args([
            "--endpoint-api", "claude-relay", "--model", "fixture-model",
            "--claude-relay-launcher", "/nonexistent/claude_relay.py",
            "--claude-relay-socket", "/nonexistent/broker.sock",
            "--claude-relay-server-uid", "1000", "--claude-relay-cli-version", "2.1.0-fixture",
            "--no-user-model-catalog", "--compaction-keep-recent-tokens", "0",
        ])
        self.args.resume = True

    async def _run(self, model, frame):
        terminal = _Terminal(frame)
        code = await asyncio.wait_for(
            cli._run(model, Environment(), terminal, self.args, self.path), 10)
        self.assertTrue(terminal.exited)
        return code

    def _compact_outcome(self):
        saved = load_interaction_save(self.path)
        index = max(i for i, item in enumerate(saved.items) if isinstance(item, UserToolCall))
        self.assertEqual(saved.items[index].call.name, "compact")
        return saved.items[index + 1:]

    async def _exit_during_compact(self, *, in_flight):
        """/compact, then /exit!! while its summary request is in flight, or
        before it starts (while the compactor builds its prompt)."""
        model, steps, building = relay_model(), [], threading.Event()
        self.addCleanup(model.close)
        original = compaction._history_prompt

        def prompt(*args, **kwargs):
            if not in_flight:
                epoch = model._epoch
                building.set()  # the frame types /exit!! now
                _reached(model, epoch)
            return original(*args, **kwargs)

        def frame(t, editor, status):
            if status == "idle" and not steps:
                steps.append("compact")
                t.submit("/compact")
            elif steps == ["compact"] and (launches.all if in_flight else building.is_set()):
                steps.append("exit")
                t.submit("/exit!!")
        with fake_relay_runtime() as launches, \
                mock.patch.object(compaction, "_history_prompt", side_effect=prompt):
            self.assertEqual(await self._run(model, frame), 0)
        [result] = self._compact_outcome()  # and no checkpoint
        self.assertIsInstance(result, UserToolResult)
        self.assertFalse(result.result.success)
        self.assertEqual(result.result.output, _CANCELLED)
        return launches

    async def test_exit_cancels_compact_in_flight_and_records_it(self):
        launches = await self._exit_during_compact(in_flight=True)
        self.assertEqual(len(launches.all), 1)
        self.assertTrue(launches.all[0].closed.is_set())

    async def test_exit_before_compact_s_request_starts_none_and_records_it(self):
        launches = await self._exit_during_compact(in_flight=False)
        self.assertEqual(launches.all, [])  # no request started after the stop

    async def test_exit_without_cancelling_lets_compact_finish_and_saves_it(self):
        model, steps = relay_model(), []
        self.addCleanup(model.close)

        def frame(t, editor, status):
            if status == "idle" and not steps:
                steps.append("compact")
                t.submit("/compact")
            elif steps == ["compact"] and launches.all:
                steps.append("exit")
                t.submit("/exit!")
            elif steps == ["compact", "exit"] and status.startswith(
                    "closing — waiting for current operation"):
                steps.append("answer")
                launches.all[0].answer("## Goal\nShip it.")
        with fake_relay_runtime() as launches:
            self.assertEqual(await self._run(model, frame), 0)
        result, checkpoint, metadata = self._compact_outcome()
        self.assertTrue(result.result.success)
        self.assertEqual(result.result.output, "Context compacted using a pi summary checkpoint.")
        self.assertIsInstance(checkpoint, ContextPrefix)
        self.assertIsInstance(metadata, CompactionMetadata)
        self.assertEqual(len(launches.all), 1)


if __name__ == "__main__":
    unittest.main()
