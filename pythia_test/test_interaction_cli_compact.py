"""/compact in the CLI: the model follows a coldstart checkpoint's pointer at once.

A coldstart checkpoint ends with its pointer, a user message that asks the
model to update the summary from the saved log and pick up where the log left
off. After /compact the CLI samples it right away, with no new message, as
the turn of an automatic compaction does. A typed /continue follows a pointer
that is still unanswered, as after an exit during /compact.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import threading
import unittest

from pythia.interaction import CompactionMetadata, CompactionResult, ContextPrefix, Environment
from pythia.interaction import Init, InteractionContext, Message, ModelFailure, ModelSampleBoundary
from pythia.interaction import SampleMetadata, TokenUsage, ToolCall, ToolResult, Tools, TurnSummary
from pythia.interaction import UserInteractionBoundary, UserToolCall, UserToolResult, cli
from pythia.interaction import load_interaction_save
from pythia_test.interaction_helpers import patch_compaction, seed_cli_save
from pythia_test.test_interaction_cli import _Model, _Terminal, _answer
from pythia_test.test_interaction_cli_reboot import (
    _Compactor, _exit, _has, _idle_with, _notices, _rebootable, _steps, _users)

_POINTER = "a pointer to the log"  # what _Compactor's checkpoint holds
_ANSWERED = "[assistant] summary updated"
_RESUMED = ("Note: resumed save ends with /compact's coldstart pointer, which the model hasn't "
            "followed yet. No model request was started; enter /continue to have the model "
            "follow it, or a query to continue with a new message.")


def _finished():
    """A turn that ended with a final answer."""
    return (Init("saved"), Tools(), Message("user", "old request"), UserInteractionBoundary(),
            Message("assistant", "old answer"), SampleMetadata(TokenUsage(90, 10, 100, 20)),
            ModelSampleBoundary(), TurnSummary(sample_count=1, context_tokens=100))


def _failed():
    """A turn whose sample failed: its next step would be a sample."""
    return (Init("saved"), Tools(), Message("user", "old request"), UserInteractionBoundary(),
            ModelFailure("ModelTransportError", "timed out"))


def _yielded():
    """A turn that main ended with icron's yield: a wake-up resumes it."""
    return (Init("saved"), Tools(), Message("user", "old request"), UserInteractionBoundary(),
            ToolCall("icron", "call_1", '{"action": "yield", "note": "Wait for CI."}'),
            ModelSampleBoundary(), ToolResult("call_1", "Yielded."), TurnSummary(sample_count=1))


def _compacted():
    """A finished turn, then /compact's coldstart checkpoint, unanswered: as an
    exit during /compact, or an older CLI, leaves the save."""
    call = ToolCall("compact", "user_1", "{}")
    return (*_finished(), UserToolCall(call), UserToolResult(ToolResult("user_1", "compacted")),
            ContextPrefix((Message("user", _POINTER),)), CompactionMetadata(TokenUsage(), "coldstart"))


class _PiCompactor:
    """Returns pi's checkpoint, which keeps the conversation's tail: no pointer."""

    def compact(self, source, *, tools=(), sample_params=None, instructions=None):
        return CompactionResult((ContextPrefix((Message("user", "a summary"),)),), protocol="pi")


def _compact_then(*steps):
    """/compact at the first idle frame, then ``steps``."""
    return _Terminal(_steps((lambda t, s: s == "idle", lambda t: t.submit("/compact")), *steps))


class _Case(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "interaction.jsonl"

    async def _session(self, model, terminal, argv=("--resume",), compactor=None, **options):
        seeded = load_interaction_save(self.path).items
        args = cli._build_parser().parse_args(list(argv))
        with patch_compaction(cli, "create_default_compactor",
                              return_value=compactor or _Compactor()):
            code = await asyncio.wait_for(
                cli._run(model, Environment(), terminal, args, self.path, **options), timeout=6)
        return code, seeded

    def assertFollowedThePointer(self, model, seeded):
        """One request, the checkpoint with its pointer last: no new message."""
        [(context, _, _)] = model.calls
        self.assertIsInstance(context.items[-1], CompactionMetadata)
        self.assertEqual(context.items[:len(seeded)], seeded)
        self.assertEqual(context.model_items()[-1], Message("user", _POINTER))
        saved = load_interaction_save(self.path)
        self.assertEqual(_users(saved), ["old request"])
        self.assertIsInstance(saved.items[-1], TurnSummary)


class AfterCompactTests(_Case):
    async def test_after_a_finished_turn_the_model_follows_the_pointer_at_once(self):
        seed_cli_save(self.path, InteractionContext(_finished()))
        model = _Model(self.path, _answer("summary updated"))
        terminal = _compact_then((_idle_with(_ANSWERED), _exit))
        code, seeded = await self._session(model, terminal)
        self.assertEqual(code, 0)
        self.assertFollowedThePointer(model, seeded)
        self.assertFalse(any(text.startswith("Nothing to continue") for text in _notices(terminal.items)))

    async def test_after_a_failed_sample_the_turn_goes_on_from_the_pointer(self):
        seed_cli_save(self.path, InteractionContext(_failed()))
        model = _Model(self.path, _answer("summary updated"))
        code, seeded = await self._session(model, _compact_then((_idle_with(_ANSWERED), _exit)))
        self.assertEqual(code, 0)
        self.assertFollowedThePointer(model, seeded)

    async def test_text_typed_during_compact_waits_for_the_pointer_s_turn(self):
        # As any input queued before a turn starts: the next query, never a steer.
        seed_cli_save(self.path, InteractionContext(_finished()))
        gate = threading.Event()
        compactor = _Compactor(gate)
        model = _Model(self.path, _answer("summary updated"), _answer("done next"))
        terminal = _compact_then(
            (lambda t, s: compactor.entered.is_set(), lambda t: t.submit("next request")),
            (lambda t, s: "queued=1" in s, lambda t: gate.set()),
            (_idle_with("[assistant] done next"), _exit))
        try:
            code, _ = await self._session(model, terminal, compactor=compactor)
        finally:
            gate.set()
        self.assertEqual(code, 0)
        first, second = (context for context, _, _ in model.calls)
        self.assertEqual(first.model_items()[-1], Message("user", _POINTER))
        self.assertEqual(_users(second.items)[-1:], ["next request"])

    async def test_a_checkpoint_without_a_pointer_starts_nothing(self):
        seed_cli_save(self.path, InteractionContext(_finished()))
        model = _Model(self.path)
        terminal = _compact_then((_idle_with("Context compacted using a pi summary checkpoint"), _exit))
        code, _ = await self._session(model, terminal, compactor=_PiCompactor())
        self.assertEqual((code, model.calls), (0, []))
        self.assertFalse(any(text.startswith("Nothing to continue") for text in _notices(terminal.items)))

    async def test_an_exit_typed_during_compact_wins(self):
        seed_cli_save(self.path, InteractionContext(_finished()))
        gate = threading.Event()
        compactor = _Compactor(gate)
        model = _Model(self.path)
        terminal = _compact_then(
            (lambda t, s: compactor.entered.is_set(), lambda t: t.submit("/exit")),
            (lambda t, s: s.startswith("closing"), lambda t: gate.set()))
        try:
            code, _ = await self._session(model, terminal, compactor=compactor)
        finally:
            gate.set()
        self.assertEqual((code, model.calls), (0, []))
        # The checkpoint is saved, unanswered; a typed /continue follows it later.
        self.assertIsInstance(load_interaction_save(self.path).items[-1], CompactionMetadata)

    async def test_after_an_icron_yield_the_pointer_waits_for_main_to_be_resumed(self):
        seed_cli_save(self.path, InteractionContext(_yielded()))
        model = _Model(self.path)
        note = ("Note: main yielded to icron, so the model follows the coldstart pointer when "
                "a wake-up or your message resumes it.")
        terminal = _compact_then((_idle_with(note), _exit))
        code, _ = await self._session(model, terminal)
        self.assertEqual((code, model.calls), (0, []))
        self.assertFalse(any(text.startswith("Nothing to continue") for text in _notices(terminal.items)))

    async def test_with_the_reboot_option_but_no_reboot_the_pointer_is_followed_here(self):
        # This process's command line isn't known in a test, so it can't reboot.
        seed_cli_save(self.path, InteractionContext(_finished()))
        model = _Model(self.path, _answer("summary updated"))
        terminal = _compact_then((_idle_with(_ANSWERED), _exit))
        code, seeded = await self._session(model, terminal,
                                           argv=("--resume", "--compaction-coldstart-reboot"))
        self.assertEqual(code, 0)
        self.assertIn("compaction_coldstart_reboot: not rebooting, since this process's command "
                      "line isn't known, so it can't be run again.", _notices(terminal.items))
        self.assertFollowedThePointer(model, seeded)

    async def test_a_reboot_that_follows_the_pointer_queues_no_continuation(self):
        seed_cli_save(self.path, InteractionContext(_finished()))
        model = _Model(self.path)
        with _rebootable(), self.assertRaises(cli._Rebooting) as caught:
            await self._session(model, _compact_then(),
                                argv=("--resume", "--compaction-coldstart-reboot"))
        carried = caught.exception.state
        self.assertEqual((carried["cause"], carried["queue"]), ("coldstart", []))
        self.assertEqual(model.calls, [])

    async def test_a_reboot_typed_during_compact_carries_the_continuation(self):
        # Without the option, /reboot reboots as typed; its restart follows the pointer.
        seed_cli_save(self.path, InteractionContext(_finished()))
        gate = threading.Event()
        compactor = _Compactor(gate)
        terminal = _compact_then(
            (lambda t, s: compactor.entered.is_set(), lambda t: t.submit("/reboot")),
            (lambda t, s: _has(t, "Rebooting once"), lambda t: gate.set()))
        try:
            with _rebootable(), self.assertRaises(cli._Rebooting) as caught:
                await self._session(_Model(self.path), terminal, compactor=compactor)
        finally:
            gate.set()
        carried = caught.exception.state
        self.assertEqual((carried["cause"], carried["queue"]), ("/reboot", [{"kind": "continue"}]))
        rebooted = cli._parse_reboot_state(carried)
        model = _Model(self.path, _answer("summary updated"))
        terminal = _Terminal(_steps((_idle_with(_ANSWERED), _exit)))
        code, seeded = await self._session(model, terminal, rebooted=rebooted)
        self.assertEqual(code, 0)
        self.assertFollowedThePointer(model, seeded)
        self.assertNotIn(_RESUMED, _notices(terminal.items))  # the continuation was queued


class TypedContinueTests(_Case):
    async def test_a_typed_continue_follows_an_unanswered_pointer_once(self):
        seed_cli_save(self.path, InteractionContext(_compacted()))
        model = _Model(self.path, _answer("summary updated"))
        refusal = "Nothing to continue: the last turn ended. Enter a query instead."
        terminal = _Terminal(_steps(
            (lambda t, s: s == "idle", lambda t: t.submit("/continue")),
            (_idle_with(_ANSWERED), lambda t: t.submit("/continue")),
            (_idle_with(refusal), _exit)))
        code, seeded = await self._session(model, terminal)
        self.assertEqual(code, 0)
        self.assertIn(_RESUMED, _notices(terminal.items))
        # The pointer, once; then the turn it started has ended.
        self.assertFollowedThePointer(model, seeded)


if __name__ == "__main__":
    unittest.main()
