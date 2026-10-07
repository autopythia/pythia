"""Append-only interaction saves, and resuming a save whose last line was cut off."""

from __future__ import annotations

import contextlib
from contextlib import redirect_stderr, redirect_stdout
import errno
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pythia.interaction import (
    Environment, Init, InteractionContext, InteractionSaveWriter, Message, ModelSample,
    ModelSampleBoundary, SaveError, Tool, ToolCall, ToolOutcome, ToolSpec, Tools,
    TurnSummary, UserInteractionBoundary, demo, interaction_item_to_dict,
    load_interaction_save, render_interaction_items, resume_interaction_save,
    save_interaction_save,
)
from pythia.interaction import cli
from pythia_test.interaction_helpers import patch_saves, real_save
from pythia_test.test_interaction_cli import _ControllerTestCase, _Model, _Terminal, _answer


def _line(item):
    return (json.dumps(interaction_item_to_dict(item), ensure_ascii=False) + "\n").encode("utf-8")


def _lines(items):
    return b"".join(_line(item) for item in items)


def _quit_when_idle(terminal, editor, status):
    if status == "idle":
        terminal.key("c-d")


_COMPLETE = (Init("prefix"), Tools(), Message("user", "question"), UserInteractionBoundary())
# A line whose cuts include one inside a multi-byte UTF-8 character.
_TAIL = _line(Message("assistant", "answer — with ünïcode ✓ " * 3))


class _SaveTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "log.jsonl"


class WriterTests(_SaveTestCase):
    def test_first_save_replaces_atomically_and_later_saves_only_append(self):
        self.path.write_bytes(b"previous file\n")
        previous = self.path.stat().st_ino
        context = InteractionContext(_COMPLETE[:2])
        writer = InteractionSaveWriter(self.path)
        writer.save(context)
        created = self.path.stat()
        self.assertNotEqual(created.st_ino, previous)  # replaced, not appended to
        self.assertEqual(created.st_mode & 0o777, 0o600)
        self.assertEqual(self.path.read_bytes(), _lines(_COMPLETE[:2]))

        with self.path.open("rb") as follower:  # as `tail -f` holds the file
            follower.read()
            context.extend(_COMPLETE[2:])
            with mock.patch("pythia.interaction.save.os.replace") as replace:
                writer.save(context.copy())
            replace.assert_not_called()
            self.assertEqual(follower.read(), _lines(_COMPLETE[2:]))
        self.assertEqual(self.path.stat().st_ino, created.st_ino)
        self.assertEqual(self.path.read_bytes(), _lines(_COMPLETE))
        self.assertEqual(load_interaction_save(self.path).items, _COMPLETE)

        # Nothing new: the file is not even opened.
        with mock.patch("pythia.interaction.save.os.open", wraps=os.open) as opened:
            writer.save(context)
        opened.assert_not_called()

    def test_save_requires_a_context_extending_the_saved_items(self):
        writer = InteractionSaveWriter(self.path)
        writer.save(InteractionContext(_COMPLETE[:3]))
        before = self.path.read_bytes()
        for other in (_COMPLETE[:2], (*_COMPLETE[:2], Message("user", "different"))):
            with self.subTest(other=other), self.assertRaisesRegex(SaveError, "does not extend"):
                writer.save(InteractionContext(other))
        self.assertEqual(self.path.read_bytes(), before)

    def test_failed_append_restores_the_file_and_the_next_save_writes_it_again(self):
        real_write = os.write

        def partial_write(descriptor, data):
            real_write(descriptor, bytes(data[:len(data) // 2]))
            raise OSError(errno.ENOSPC, "No space left on device")

        write = mock.patch("pythia.interaction.save.os.write", side_effect=partial_write)
        fsync = mock.patch("pythia.interaction.save.os.fsync",
                           side_effect=OSError(errno.EIO, "fsync failed"))
        # An append truncates only to clean up after its own failure.
        cleanup = mock.patch("pythia.interaction.save.os.ftruncate",
                             side_effect=OSError(errno.EIO, "cleanup failed"))
        for name, patches, cleaned in (("partial write", (write,), True),
                                       ("fsync", (fsync,), True),
                                       ("partial write, failed cleanup", (write, cleanup), False)):
            with self.subTest(failure=name):
                writer = InteractionSaveWriter(self.path)
                context = InteractionContext(_COMPLETE[:2])
                writer.save(context)
                durable = self.path.read_bytes()
                context.extend((Message("user", "x" * 10_000), UserInteractionBoundary()))
                with contextlib.ExitStack() as stack:
                    for patch in patches:
                        stack.enter_context(patch)
                    with self.assertRaisesRegex(SaveError, "could not append .*(No space|fsync)"):
                        writer.save(context)
                if cleaned:
                    self.assertEqual(self.path.read_bytes(), durable)
                else:  # the partial line stays until the next save truncates it
                    partial = self.path.read_bytes()
                    self.assertTrue(partial.startswith(durable))
                    self.assertGreater(len(partial), len(durable))
                    self.assertEqual(load_interaction_save(self.path).items, _COMPLETE[:2])
                writer.save(context)
                self.assertEqual(self.path.read_bytes(), durable + _lines(context.items[2:]))
                self.assertEqual(load_interaction_save(self.path).items, context.items)

    def test_refuses_to_append_to_a_file_replaced_or_changed_since_saving(self):
        changes = {
            "replaced": lambda: save_interaction_save(self.path, InteractionContext(_COMPLETE[:2])),
            "appended": lambda: self.path.write_bytes(self.path.read_bytes() + _line(Tools())),
            "truncated": lambda: os.truncate(self.path, 3),
        }
        for change, apply in changes.items():
            with self.subTest(change=change):
                writer = InteractionSaveWriter(self.path)
                context = InteractionContext(_COMPLETE[:2])
                writer.save(context)
                apply()
                changed = self.path.read_bytes()
                context.extend(_COMPLETE[2:])
                with self.assertRaisesRegex(SaveError, "replaced or changed"):
                    writer.save(context)
                self.assertEqual(self.path.read_bytes(), changed)

    def test_resumed_writer_appends_after_saved_lines_and_keeps_legacy_records(self):
        legacy = (b'{"type": "session_init", "session_id": "legacy"}\n'
                  b'{"type": "message", "role": "user", "text": "hi"}\n')
        self.path.write_bytes(legacy)
        inode = self.path.stat().st_ino
        context, writer, incomplete = resume_interaction_save(self.path)
        self.assertIsNone(incomplete)
        self.assertEqual(context.items, (Init("legacy"), Message("user", "hi")))
        self.assertEqual(self.path.read_bytes(), legacy)  # resuming is read-only
        context.append(Message("assistant", "hello"))
        writer.save(context)
        self.assertEqual(self.path.read_bytes(), legacy + _line(Message("assistant", "hello")))
        self.assertEqual(self.path.stat().st_ino, inode)
        self.assertEqual(load_interaction_save(self.path).items, context.items)


class IncompleteLineTests(_SaveTestCase):
    def test_cut_off_final_line_is_ignored_and_truncated_by_the_resumed_writer(self):
        durable = _lines(_COMPLETE)
        multibyte = _TAIL.index("✓".encode()) + 1
        tails = {f"cut at {cut}": _TAIL[:cut]
                 for cut in (1, 10, multibyte, len(_TAIL) // 2, len(_TAIL) - 2)}
        tails["zero-filled"] = b"\0" * 4096  # unwritten blocks after a power loss
        tails["with newline"] = b'{"type": "message", "role\n'
        for name, tail in tails.items():
            for repair_only in (False, True):
                with self.subTest(tail=name, repair_only=repair_only):
                    self.path.write_bytes(durable + tail)
                    self.assertEqual(load_interaction_save(self.path).items, _COMPLETE)
                    context, writer, incomplete = resume_interaction_save(self.path)
                    self.assertEqual(self.path.read_bytes(), durable + tail)  # nothing modified yet
                    self.assertEqual(context.items, _COMPLETE)
                    self.assertEqual((incomplete.line_number, incomplete.offset, incomplete.size),
                                     (5, len(durable), len(tail)))
                    self.assertEqual(
                        incomplete.warning("log.jsonl"),
                        f"Warning: log.jsonl ends with a line that fails to parse (line 5, "
                        f"{len(tail)} bytes), probably cut off by an interrupted save. It was "
                        "not loaded and is truncated from the file.")
                    new = () if repair_only else (Message("assistant", "again"), ModelSampleBoundary())
                    context.extend(new)
                    writer.save(context)
                    self.assertEqual(self.path.read_bytes(), durable + _lines(new))
                    again = resume_interaction_save(self.path)
                    self.assertIsNone(again.incomplete_line)
                    self.assertEqual(again.context.items, context.items)

    def test_final_line_missing_only_its_newline_is_complete(self):
        self.path.write_bytes(_lines(_COMPLETE) + _TAIL[:-1])
        context, writer, incomplete = resume_interaction_save(self.path)
        self.assertIsNone(incomplete)
        self.assertIsInstance(context[-1], Message)
        self.assertEqual(context.items[:-1], _COMPLETE)
        context.append(TurnSummary())
        writer.save(context)
        self.assertEqual(self.path.read_bytes(), _lines(_COMPLETE) + _TAIL + _line(TurnSummary()))

    def test_other_corruption_is_an_error_and_never_modifies_the_save(self):
        first, second = _line(_COMPLETE[0]), _line(_COMPLETE[1])
        cases = (
            (first + b"{broken\n" + second, r"invalid JSON in save .* at line 2"),
            (first + b"\xff\xfe\n" + second, r"invalid UTF-8 in save .* at line 2"),
            (first + b"{broken\n\n", r"invalid JSON in save .* at line 2"),
            (b"{broken", r"invalid JSON in save .* at line 1"),
            (b"\n\n{broken", r"invalid JSON in save .* at line 3"),
            (first + b'{"type": "bogus"}', r"invalid save item in .* at line 2"),
            (first + b"123\n", r"invalid save item in .* at line 2"),
            (second + first, r"invalid save .*Init must be the first"),
        )
        for data, message in cases:
            with self.subTest(data=data):
                self.path.write_bytes(data)
                for load in (load_interaction_save, resume_interaction_save):
                    with self.assertRaisesRegex(SaveError, message):
                        load(self.path)
                self.assertEqual(self.path.read_bytes(), data)

    def test_truncation_is_refused_if_the_save_changed_after_resuming(self):
        cut_off = _lines(_COMPLETE) + _TAIL[:10]
        self.path.write_bytes(cut_off)
        context, writer, incomplete = resume_interaction_save(self.path)
        self.assertIsNotNone(incomplete)
        with self.path.open("ab") as file:  # e.g. another process still writing
            file.write(_TAIL[10:])
        with self.assertRaisesRegex(SaveError, "replaced or changed"):
            writer.save(context)
        self.assertEqual(self.path.read_bytes(), _lines(_COMPLETE) + _TAIL)


class CLIAppendTests(_ControllerTestCase):
    async def test_session_saves_append_without_rewriting_saved_lines(self):
        handler = mock.Mock(return_value=ToolOutcome("looked"))
        environment = Environment((Tool(ToolSpec("lookup", "Look up.", {}), handler),))
        model = _Model(self.path, ModelSample((ToolCall("lookup", "one", "{}"),)), _answer())
        states = []

        def save(writer, context):
            real_save(writer, context)
            states.append((self.path.stat().st_ino, self.path.read_bytes()))

        with patch_saves(save):
            self.assertEqual(await self._run(model, _Terminal(_quit_when_idle),
                                             ["--prompt", "hello"], environment), 0)
        handler.assert_called_once()
        self.assertGreaterEqual(len(states), 5)
        self.assertEqual(len({inode for inode, _ in states}), 1)  # created once
        for (_, before), (_, after) in zip(states, states[1:]):
            self.assertTrue(after.startswith(before))
        self.assertIsInstance(load_interaction_save(self.path)[-1], TurnSummary)

    async def test_resume_warns_and_truncates_a_cut_off_line_before_other_work(self):
        save_interaction_save(self.path, InteractionContext(_COMPLETE))
        durable = self.path.read_bytes()
        self.path.write_bytes(durable + _TAIL[:20])
        model, terminal = _Model(self.path), _Terminal(_quit_when_idle)
        self.assertEqual(await self._run(model, terminal, ["--resume"]), 0)
        self.assertEqual(model.calls, [])
        self.assertEqual(self.path.read_bytes(), durable)
        notices = [item.text for item in terminal.items if item.text.startswith("[cli]")]
        self.assertIn(
            "[cli] Warning: interaction.jsonl ends with a line that fails to parse (line 5, "
            "20 bytes), probably cut off by an interrupted save. It was not loaded and is "
            "truncated from the file.", notices)
        self.assertEqual(tuple(item for item in terminal.items if not item.text.startswith("[cli]")),
                         render_interaction_items(_COMPLETE))
        # Shown once: the next resume finds a complete save.
        terminal = _Terminal(_quit_when_idle)
        self.assertEqual(await self._run(_Model(self.path), terminal, ["--resume"]), 0)
        self.assertFalse(any("fails to parse" in item.text for item in terminal.items))

    async def test_resumed_query_continues_from_the_complete_items(self):
        complete = (Init("old"), Tools(), Message("assistant", "previous"), TurnSummary())
        save_interaction_save(self.path, InteractionContext(complete))
        durable = self.path.read_bytes()
        self.path.write_bytes(durable + b'{"type": "message", "role": "user", "con')
        model = _Model(self.path, _answer("next answer"))
        self.assertEqual(await self._run(model, _Terminal(_quit_when_idle),
                                         ["--resume", "--prompt", "next"]), 0)
        [(context, _tools, _params)] = model.calls
        self.assertEqual(context.items, (*complete, Message("user", "next"),
                                         UserInteractionBoundary()))
        self.assertTrue(self.path.read_bytes().startswith(durable))
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items[:len(context)], context.items)
        self.assertIsInstance(saved[-1], TurnSummary)

    async def test_resume_fails_on_a_bad_line_before_the_end_without_modifying_it(self):
        data = _line(_COMPLETE[0]) + b"{broken\n" + _line(_COMPLETE[1])
        self.path.write_bytes(data)
        model = _Model(self.path)
        with self.assertRaisesRegex(SaveError, "invalid JSON in save .* at line 2"):
            await self._run(model, _Terminal(_quit_when_idle), ["--resume"])
        self.assertEqual(model.calls, [])
        self.assertEqual(self.path.read_bytes(), data)


class DemoAppendTests(_SaveTestCase):
    def test_resume_warns_truncates_and_appends(self):
        complete = (Init("old"), Tools(), Message("assistant", "previous"), TurnSummary())
        save_interaction_save(self.path, InteractionContext(complete))
        durable = self.path.read_bytes()
        self.path.write_bytes(durable + _TAIL[:30])
        model = mock.Mock(spec=["sample"])
        model.sample.return_value = _answer("demo answer")
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(demo.run(model, Environment(), prompt="next", save_path=self.path,
                                      resume=True), "demo answer")
        self.assertIn("Warning: log.jsonl ends with a line that fails to parse (line 5, 30 bytes)",
                      stderr.getvalue())
        [call] = model.sample.call_args_list
        self.assertEqual(call.args[0].items[:len(complete)], complete)
        self.assertTrue(self.path.read_bytes().startswith(durable))
        self.assertIsInstance(load_interaction_save(self.path)[-1], TurnSummary)


if __name__ == "__main__":
    unittest.main()
