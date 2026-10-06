"""The CLI runs each turn on its context thread; terminal state stays on the loop."""

from __future__ import annotations

import asyncio
import collections
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from pythia.interaction import Environment, Init, InteractionContext, Message, ModelSample
from pythia.interaction import Tool, ToolCall, ToolOutcome, ToolSpec
from pythia.interaction import cli
from pythia.interaction.model import ModelTransportError
from pythia.interaction.runtime_config import InteractionConfig, InteractionConfigSnapshot


def _context():
    return InteractionContext((Init(model="scripted"), Message("user", "task")))


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
        original_save = cli.save_interaction_save

        def save(path, context):
            seen["save"].add(threading.get_ident())
            return original_save(path, context)
        state = cli._UIState(headless=True)
        state.displays = Displays()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(cli, "save_interaction_save", save):
            result = await cli._turn(_context(), Model(), environment, state,
                                     Path(directory) / "log.jsonl",
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
                                     Path(directory) / "log.jsonl",
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
                                Path(directory) / "log.jsonl",
                                InteractionConfig(InteractionConfigSnapshot()))
        self.assertIsInstance(state.retry, cli._RetryIntent)



class CliSteeringTests(unittest.IsolatedAsyncioTestCase):
    async def _turn(self, state, model, environment=None):
        with tempfile.TemporaryDirectory() as directory:
            return await cli._turn(_context(), model, environment or Environment(), state,
                                   Path(directory) / "log.jsonl",
                                   InteractionConfig(InteractionConfigSnapshot()),
                                   steer=lambda text: Message("user", text))

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
            loop.call_soon_threadsafe(state.pending.append, "also check the tests")
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
        state.pending.append("older query")

        def handler(arguments, *, timeout_seconds=None):
            loop.call_soon_threadsafe(state.pending.append, "typed later")
            return ToolOutcome(output="ok")
        environment = Environment((Tool(ToolSpec("probe", "Probe.", {"type": "object"}), handler),))
        await self._turn(state, self._model(seen), environment)
        self.assertEqual(seen, [["task"], ["task"]])
        self.assertEqual(list(state.pending), ["older query", "typed later"])

    def test_steers_stop_at_the_first_command(self):
        state = cli._UIState(headless=True)
        retry = cli._RetryIntent()
        state.pending.extend(["first", "second", retry, "after"])
        self.assertEqual(cli._steer_count(state), 0)  # no turn is running
        state.turn_active = True
        self.assertEqual(cli._steer_count(state), 2)
        self.assertEqual(cli._take_steers(state), ("first", "second"))
        self.assertEqual(list(state.pending), [retry, "after"])
        self.assertEqual(cli._take_steers(state), ())


if __name__ == "__main__":
    unittest.main()
