"""The CLI's save: --save PREFIX, its public log and private directory, and migration."""

from __future__ import annotations

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import errno
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import (
    CompactionMetadata, ContextPrefix, Environment, Init, InteractionContext, Message,
    ModelSample, ModelSampleBoundary, Reasoning, SaveError, Tool, ToolCall, ToolOutcome,
    ToolResult, ToolSpec, Tools, TokenUsage, TurnSummary, UserInteractionBoundary,
    UserToolCall, UserToolResult, cli, interaction_item_to_dict, load_interaction_save,
    save_interaction_save,
)
from pythia.interaction import _cli_save
from pythia.interaction._cli_save import (
    CliSave, CliSaveError, CliSaveWriter, check_cli_save, cli_save_exists, full_log_path,
    prepare_cli_save, private_dir, public_log_path, resume_cli_save, save_prefix,
)
from pythia.interaction.compaction import COLDSTART_POINTER_PREFIX
from pythia.interaction.compaction import ColdstartCompactor
from pythia.interaction.compaction import CompactionLogError
from pythia.interaction.compaction import is_coldstart_pointer
from pythia.interaction.save import PublicLogError
from pythia.interaction.save import PublicLogWriter
from pythia.interaction.save import public_log_line
from pythia.interaction.save import public_log_lines
from pythia_test.interaction_helpers import seed_cli_save
from pythia_test.test_interaction_cli import _ControllerTestCase, _Model, _Terminal, _answer


_SECRET = "SECRET-REASONING"
_REASONING = Reasoning(f"{_SECRET} text", (f"{_SECRET} summary",),
                       encrypted_content=f"{_SECRET}-cipher")
_SIGNED = Reasoning(f"{_SECRET} thinking", content_signature=f"{_SECRET}-signature")
# A finished turn with reasoning on its fifth line.
_HISTORY = (Init("session"), Tools(), Message("user", "question"), UserInteractionBoundary(),
            _REASONING, Message("assistant", "answer"), ModelSampleBoundary(),
            TurnSummary(sample_count=1))
_NIL = b'{"type": "nil"}\n'
_EXEC = ToolSpec("exec_command", "Run a command.", {"type": "object"})
_DEBUG_FILES = ("context.jsonl.model-binding.json", "trace.events.jsonl", "trace.req.jsonl",
                "trace.res.jsonl")


def _encoded(items):
    return b"".join((json.dumps(interaction_item_to_dict(item), ensure_ascii=False) + "\n")
                    .encode("utf-8") for item in items)


def _public(items):
    return b"".join(public_log_lines(_encoded(items)))


def _tree(root):
    """Every path under ``root``: a file's bytes and inode, a link's target, or "dir"."""
    tree = {}
    for directory, directories, files in os.walk(root):
        for name in (*directories, *files):
            path = Path(directory) / name
            key = str(path.relative_to(root))
            if path.is_symlink():
                tree[key] = ("link", os.readlink(path))
            elif path.is_dir():
                tree[key] = "dir"
            else:
                tree[key] = (path.read_bytes(), path.stat().st_ino)
    return tree


def _quit_when_idle(terminal, editor, status):
    if status == "idle":
        terminal.key("c-d")


def _submit_then_quit(text, until):
    """Submit ``text`` at the first idle frame, then quit once ``until`` is shown."""
    sent = []

    def frame(terminal, editor, status):
        if status != "idle":
            return
        if not sent:
            sent.append(text)
            terminal.submit(text)
        elif any(until in item.text for item in terminal.items):
            terminal.key("c-d")
    return frame


class _Crash(BaseException):
    """The process dying mid-migration: nothing is undone until the next launch."""


class PrefixTests(unittest.TestCase):
    def test_a_trailing_jsonl_is_ignored(self):
        for argument, prefix in (("review.jsonl", "review"), ("review", "review"),
                                 ("auto-0.1", "auto-0.1"), ("notes.log", "notes.log"),
                                 ("a.jsonl.jsonl", "a.jsonl"), (".jsonl", ".jsonl")):
            with self.subTest(argument=argument):
                self.assertEqual(save_prefix(Path("/x") / argument), Path("/x") / prefix)
        prefix = Path("/x/auto-0.1")
        self.assertEqual(public_log_path(prefix), Path("/x/auto-0.1.jsonl"))
        self.assertEqual(private_dir(prefix), Path("/x/auto-0.1._private"))
        self.assertEqual(full_log_path("/x/review.jsonl"), Path("/x/review._private/context.jsonl"))
        save = CliSave.at(prefix)
        self.assertEqual((save.log, save.public, save.private, save.legacy, save.shown), (
            Path("/x/auto-0.1._private/context.jsonl"), Path("/x/auto-0.1.jsonl"),
            Path("/x/auto-0.1._private"), False, Path("/x/auto-0.1.jsonl")))
        self.assertEqual(save.log_name, "auto-0.1._private/context.jsonl")
        self.assertEqual(save.binding_path,
                         Path("/x/auto-0.1._private/context.jsonl.model-binding.json"))
        self.assertEqual(save.trace_paths, (Path("/x/auto-0.1._private/trace.req.jsonl"),
                                            Path("/x/auto-0.1._private/trace.res.jsonl"),
                                            Path("/x/auto-0.1._private/trace.events.jsonl")))
        legacy = CliSave(Path("/x/old.jsonl"))
        self.assertEqual((legacy.legacy, legacy.shown, legacy.log_name), (True, legacy.log, "old.jsonl"))
        self.assertEqual(legacy.trace_paths, (Path("/x/old.trace.req.jsonl"),
                                              Path("/x/old.trace.res.jsonl"),
                                              Path("/x/old.trace.events.jsonl")))


class _FilesTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "s.jsonl"
        self.prefix = self.root / "s"
        self.private = self.root / "s._private"
        self.full = self.private / "context.jsonl"


class PublicLogTests(_FilesTestCase):
    def test_reasoning_is_a_nil_line_and_every_other_line_is_kept_byte_for_byte(self):
        checkpoint = ContextPrefix((Message("user", "summary"), _REASONING,
                                    Message("assistant", "kept")))
        items = (*_HISTORY, _SIGNED, Message("assistant", "after"), checkpoint)
        full = _encoded(items)
        public = public_log_lines(full)
        self.assertEqual(public, [public_log_line(item) for item in items])
        originals = full.splitlines(keepends=True)
        self.assertEqual((len(public), len(originals)), (len(items), len(items)))
        for item, line, original in zip(items, public, originals):
            if isinstance(item, Reasoning):
                self.assertEqual(line, _NIL)  # no other field
            elif isinstance(item, ContextPrefix):  # its reasoning is left out
                self.assertEqual(json.loads(line), {"type": "context_prefix", "prefix_items": [
                    interaction_item_to_dict(Message("user", "summary")),
                    interaction_item_to_dict(Message("assistant", "kept"))]})
            else:
                self.assertEqual(line, original)
        self.assertNotIn(_SECRET.encode(), b"".join(public))

    def test_a_public_log_does_not_load_as_a_save(self):
        self.path.write_bytes(_public(_HISTORY))
        with self.assertRaises(SaveError) as caught:
            load_interaction_save(self.path)
        self.assertIsInstance(caught.exception.__cause__, PublicLogError)
        self.assertIn("at line 5", str(caught.exception))
        self.assertIn("PREFIX._private/context.jsonl", str(caught.exception))

    def test_the_writer_writes_the_log_anew_then_appends(self):
        self.path.write_bytes(b"an older save\n")
        context = InteractionContext(_HISTORY[:4])
        writer = PublicLogWriter(self.path)
        writer.save(context)
        inode = self.path.stat().st_ino
        context.extend(_HISTORY[4:])
        writer.save(context)
        self.assertEqual(self.path.stat().st_ino, inode)
        self.assertEqual(self.path.read_bytes(), _public(_HISTORY))
        self.assertEqual(writer.take_warnings(), ())
        self.assertFalse(writer.drifted)

    def test_edits_are_kept_and_new_lines_follow_them_with_one_note(self):
        context = InteractionContext(_HISTORY[:4])
        writer = PublicLogWriter(self.path)
        writer.save(context)
        with self.path.open("ab") as file:
            file.write(b'{"note": "added by hand"}')  # without its newline
        context.extend(_HISTORY[4:])
        writer.save(context)
        self.assertEqual(self.path.read_bytes(), _public(_HISTORY[:4])
                         + b'{"note": "added by hand"}\n' + _public(_HISTORY[4:]))
        [note] = writer.take_warnings()
        self.assertIn("s.jsonl was changed outside this session. It is informational, so it "
                      "was kept as it is", note)
        self.assertTrue(writer.drifted)
        context.append(Message("user", "next"))
        writer.save(context)
        self.assertEqual(writer.take_warnings(), ())  # once

    def test_a_missing_public_log_is_written_anew(self):
        context = InteractionContext(_HISTORY[:4])
        writer = PublicLogWriter(self.path)
        writer.save(context)
        self.path.unlink()
        context.extend(_HISTORY[4:])
        writer.save(context)
        self.assertEqual(self.path.read_bytes(), _public(_HISTORY))
        self.assertEqual(writer.take_warnings(),
                         ("Note: s.jsonl was missing; it was written anew from the full log.",))
        self.assertFalse(writer.drifted)

    def test_a_failed_write_warns_once_and_the_next_save_writes_it(self):
        error = SaveError("could not write")
        error.__cause__ = OSError(errno.ENOSPC, "No space left on device")
        context = InteractionContext(_HISTORY[:4])
        writer = PublicLogWriter(self.path)
        with mock.patch("pythia.interaction.save._replace_atomically", side_effect=error):
            writer.save(context)
            context.append(_HISTORY[4])
            writer.save(context)
        self.assertEqual(writer.take_warnings(), (
            "Warning: could not write s.jsonl (No space left on device). It is informational: "
            "the save is unaffected, and the next save tries again.",))
        self.assertFalse(self.path.exists())
        context.extend(_HISTORY[5:])
        writer.save(context)
        self.assertEqual(self.path.read_bytes(), _public(_HISTORY))
        self.assertEqual(writer.take_warnings(), ())

    def test_a_failed_append_is_dropped_and_written_again(self):
        context = InteractionContext(_HISTORY[:4])
        writer = PublicLogWriter(self.path)
        writer.save(context)
        before = self.path.read_bytes()

        def partial(descriptor, data):
            os.write(descriptor, data[:7])
            raise OSError(errno.EIO, "Input/output error")

        context.append(_HISTORY[4])
        with mock.patch("pythia.interaction.save._write_all", side_effect=partial):
            writer.save(context)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(len(writer.take_warnings()), 1)
        context.extend(_HISTORY[5:])
        writer.save(context)
        self.assertEqual(self.path.read_bytes(), _public(_HISTORY))
        self.assertEqual(writer.take_warnings(), ())
        self.assertFalse(writer.drifted)

    def test_resuming_completes_a_public_log_that_lags_without_a_note(self):
        full, expected = _encoded(_HISTORY), _public(_HISTORY)
        for kept in (0, 7, len(expected) - 3, len(expected)):
            with self.subTest(kept=kept):
                self.path.write_bytes(expected[:kept])
                writer = PublicLogWriter(self.path)
                writer.sync(full, len(_HISTORY))
                self.assertEqual(self.path.read_bytes(), expected)
                self.assertEqual(writer.take_warnings(), ())
                self.assertFalse(writer.drifted)

    def test_resuming_keeps_a_public_log_that_differs(self):
        full, expected = _encoded(_HISTORY), public_log_lines(_encoded(_HISTORY))
        edited = b"".join(expected[:2]) + b'{"edited": true}\n' + b"".join(expected[3:])
        self.path.write_bytes(edited)
        writer = PublicLogWriter(self.path)
        writer.sync(full, len(_HISTORY))
        self.assertEqual(self.path.read_bytes(), edited)
        [note] = writer.take_warnings()
        self.assertIn("s.jsonl differs from the full log from line 3 on", note)
        self.assertTrue(writer.drifted)
        writer.save(InteractionContext((*_HISTORY, Message("user", "next"))))
        self.assertEqual(self.path.read_bytes(), edited + public_log_line(Message("user", "next")))
        self.assertEqual(writer.take_warnings(), ())

    def test_lines_encoded_otherwise_as_the_same_json_are_no_difference(self):
        full = _encoded(_HISTORY)
        compact = b"".join(json.dumps(json.loads(line), separators=(",", ":")).encode() + b"\n"
                           for line in public_log_lines(full))
        self.path.write_bytes(compact)
        writer = PublicLogWriter(self.path)
        writer.sync(full, len(_HISTORY))
        self.assertEqual((writer.take_warnings(), writer.drifted), ((), False))
        self.assertEqual(self.path.read_bytes(), compact)

    def test_a_missing_public_log_is_written_on_resume(self):
        writer = PublicLogWriter(self.path)
        writer.sync(_encoded(_HISTORY), len(_HISTORY))
        self.assertEqual(self.path.read_bytes(), _public(_HISTORY))
        self.assertEqual(writer.take_warnings(),
                         ("Note: s.jsonl was missing; it was written anew from the full log.",))


class MigrationTests(_FilesTestCase):
    def _original(self, path=None, items=_HISTORY):
        """A save in the original format, with its traces and snapshot; returns its bytes."""
        path = self.path if path is None else path
        save_interaction_save(path, InteractionContext(items))
        for suffix in (".trace.req.jsonl", ".trace.res.jsonl", ".trace.events.jsonl"):
            path.with_suffix(suffix).write_bytes(f"{suffix} trace\n".encode())
        path.with_name(path.name + ".model-binding.json").write_bytes(b"{}\n")
        return path.read_bytes()

    def test_a_resumed_save_moves_into_its_private_directory(self):
        data = self._original()
        inode = self.path.stat().st_ino
        summary = self.root / "s.summary.md"
        summary.write_text("Covers lines 1\u20138 of session.\n")
        save = prepare_cli_save(self.path)
        self.assertEqual((save.log, save.public, save.private), (self.full, self.path, self.private))
        [notice] = save.notices
        self.assertEqual(notice, (
            f"Migrated the save {self.path} to the current format: its full log, with "
            f"reasoning, is now {self.full}, and {self.path} is its public log, without "
            f"reasoning; its debug files moved into {self.private}."))
        self.assertEqual(self.full.read_bytes(), data)
        self.assertEqual(self.full.stat().st_ino, inode)  # moved, not copied
        self.assertEqual(self.path.read_bytes(), b"".join(public_log_lines(data)))
        self.assertNotIn(_SECRET.encode(), self.path.read_bytes())
        self.assertEqual(stat.S_IMODE(self.private.stat().st_mode), 0o700)
        self.assertEqual(sorted(os.listdir(self.private)), ["context.jsonl", *_DEBUG_FILES])
        for name in ("trace.req.jsonl", "trace.res.jsonl", "trace.events.jsonl"):
            self.assertEqual((self.private / name).read_bytes(), f".{name} trace\n".encode())
        self.assertEqual(sorted(os.listdir(self.root)), ["s._private", "s.jsonl", "s.summary.md"])
        self.assertEqual(summary.read_text(), "Covers lines 1\u20138 of session.\n")
        context, writer, incomplete = resume_cli_save(save)  # from the full log alone
        self.assertEqual((context.items, incomplete, writer.take_warnings()), (_HISTORY, None, ()))
        self.assertEqual(prepare_cli_save(self.path), CliSave.at(self.prefix))
        self.assertEqual(prepare_cli_save(self.prefix), CliSave.at(self.prefix))

    def test_a_save_named_without_jsonl_moves_from_its_path(self):
        literal = self.root / "notes.log"
        data = self._original(literal)
        save = prepare_cli_save(literal)
        self.assertEqual(save, CliSave.at(literal, save.notices))
        self.assertEqual(save.public, self.root / "notes.log.jsonl")
        self.assertFalse(literal.exists())
        self.assertEqual(save.log.read_bytes(), data)
        self.assertEqual(save.public.read_bytes(), b"".join(public_log_lines(data)))
        self.assertEqual(sorted(os.listdir(save.private)), ["context.jsonl", *_DEBUG_FILES])
        self.assertEqual(sorted(os.listdir(self.root)), ["notes.log._private", "notes.log.jsonl"])

    def test_two_saves_for_one_prefix_are_refused(self):
        self._original(self.prefix)  # as --save s wrote it
        save_interaction_save(self.path, InteractionContext(_HISTORY[:4]))  # --save s.jsonl
        before = _tree(self.root)
        with self.assertRaises(CliSaveError) as caught:
            prepare_cli_save(self.prefix)
        self.assertIn(f"both {self.prefix} and {self.path} exist", str(caught.exception))
        self.assertIn("rename one of them, or choose another --save prefix", str(caught.exception))
        self.assertEqual(_tree(self.root), before)

    def test_a_save_or_debug_file_that_is_a_link_is_left_as_it_was(self):
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        for linked in ("save", "debug file"):
            with self.subTest(linked=linked):
                for path in self.root.glob("s.*"):
                    path.unlink()
                if linked == "save":
                    save_interaction_save(elsewhere / "real.jsonl", InteractionContext(_HISTORY))
                    self.path.symlink_to(elsewhere / "real.jsonl")
                    what = f"the save {self.path}"
                else:
                    self._original()
                    trace = self.root / "s.trace.res.jsonl"
                    trace.unlink()
                    trace.symlink_to(elsewhere)
                    what = f"its debug file {trace}"
                before = _tree(self.root)
                with self.assertRaises(CliSaveError) as caught:
                    prepare_cli_save(self.path)
                self.assertEqual(str(caught.exception), (
                    f"could not migrate the save {self.path} to the current format: {what} is "
                    "a symbolic link. Nothing was changed. To use the save in its original "
                    "format, rerun with --auto-migrate-save=False."))
                self.assertEqual(_tree(self.root), before)
                self.assertEqual(prepare_cli_save(self.path, auto_migrate=False), CliSave(self.path))

    def test_a_public_log_without_its_full_log_is_refused(self):
        self.private.mkdir()
        self.path.write_bytes(_public(_HISTORY))
        before = _tree(self.root)
        with self.assertRaises(CliSaveError) as caught:
            prepare_cli_save(self.path)
        self.assertEqual(str(caught.exception), (
            f"the save's full log {self.full} is missing, and {self.path} is a public log, "
            "without reasoning, not a full log: restore the full log, or start a new save "
            "with --resume=False"))
        self.assertEqual(_tree(self.root), before)

    def test_a_failure_at_any_step_restores_the_original_save_byte_for_byte(self):
        rename, unlink, write_all = os.rename, os.unlink, _cli_save._write_all

        def failing_rename(when):
            def fail(source, destination):
                if when(Path(source), Path(destination)):
                    raise OSError(errno.EIO, "Input/output error", str(destination))
                return rename(source, destination)
            return mock.patch.object(_cli_save.os, "rename", side_effect=fail)

        def failing_journal_removal():
            removed = []

            def fail(path, *args, **kwargs):
                if Path(path).name == "migration.json" and not removed:
                    removed.append(path)
                    raise OSError(errno.EIO, "Input/output error", str(path))
                return unlink(path, *args, **kwargs)
            return mock.patch.object(_cli_save.os, "unlink", side_effect=fail)

        def full_disk():
            calls = []

            def fail(descriptor, data):  # the journal, then the public log
                calls.append(data)
                if len(calls) == 2:
                    raise OSError(errno.ENOSPC, "No space left on device")
                return write_all(descriptor, data)
            return mock.patch.object(_cli_save, "_write_all", side_effect=fail)

        steps = {
            "moving a debug file": (failing_rename(
                lambda source, destination: destination.name == "trace.res.jsonl"),
                f"Input/output error: {self.private / 'trace.res.jsonl'}"),
            "moving the log": (failing_rename(
                lambda source, destination: destination.name == "context.jsonl"),
                f"Input/output error: {self.full}"),
            "writing the public log": (full_disk(), "No space left on device"),
            "moving the public log into place": (failing_rename(
                lambda source, destination: destination == self.path),
                f"Input/output error: {self.path}"),
            "removing the journal": (failing_journal_removal(),
                                     f"Input/output error: {self.private / 'migration.json'}"),
        }
        for step, (patch, reason) in steps.items():
            with self.subTest(step=step):
                for path in self.root.glob("s.*"):
                    path.unlink()
                self._original()
                before = _tree(self.root)
                with patch, self.assertRaises(CliSaveError) as caught:
                    prepare_cli_save(self.path)
                self.assertEqual(str(caught.exception), (
                    f"could not migrate the save {self.path} to the current format: {reason}. "
                    "The original save was restored. To use the save in its original format, "
                    "rerun with --auto-migrate-save=False."))
                self.assertEqual(_tree(self.root), before)  # bytes and inodes

    def test_a_save_that_changes_while_it_is_migrated_goes_back(self):
        data = self._original()
        rename = os.rename

        def written_meanwhile(source, destination):
            if Path(destination).name == "context.jsonl":
                with open(source, "ab") as file:
                    file.write(_encoded((Message("user", "meanwhile"),)))
            return rename(source, destination)

        with mock.patch.object(_cli_save.os, "rename", side_effect=written_meanwhile), \
                self.assertRaisesRegex(CliSaveError, "the save changed while it was being "
                                       "migrated; is another process using it[?]. The original "
                                       "save was restored."):
            prepare_cli_save(self.path)
        self.assertEqual(self.path.read_bytes(), data + _encoded((Message("user", "meanwhile"),)))
        self.assertFalse(self.private.exists())

    def _crash(self, at):
        """Migrate, and die at a step: the journal stays, and nothing is undone."""
        steps = {
            "after a debug file moved": mock.patch.object(
                _cli_save, "_rename_new", side_effect=_after(_cli_save._rename_new, 1)),
            "after the log moved": mock.patch.object(
                _cli_save, "_check_unchanged", side_effect=_Crash),
            "after the public log was written": mock.patch.object(
                _cli_save, "_write_public", side_effect=_after(_cli_save._write_public, 1)),
        }
        with steps[at], mock.patch.object(_cli_save, "_undo", return_value=[]), \
                self.assertRaises(_Crash):
            prepare_cli_save(self.path)
        self.assertTrue((self.private / "migration.json").exists())

    def test_an_interrupted_migration_is_undone_at_the_next_launch(self):
        for at in ("after a debug file moved", "after the log moved",
                   "after the public log was written"):
            for auto_migrate in (False, True):
                with self.subTest(at=at, auto_migrate=auto_migrate):
                    for path in self.root.glob("s.*"):
                        path.unlink() if not path.is_dir() else _remove_tree(path)
                    data = self._original()
                    before = _tree(self.root)
                    self._crash(at)
                    self.assertNotEqual(_tree(self.root), before)
                    save = prepare_cli_save(self.path, auto_migrate=auto_migrate)
                    self.assertEqual(save.notices[0], (
                        f"Note: a migration of the save {self.path} to the current format was "
                        "interrupted; it was undone."))
                    if auto_migrate:  # then migrated again
                        self.assertTrue(save.notices[1].startswith("Migrated the save"))
                        self.assertEqual(self.full.read_bytes(), data)
                        self.assertFalse((self.private / "migration.json").exists())
                    else:
                        self.assertEqual(save, CliSave(self.path, notices=save.notices))
                        self.assertEqual(_tree(self.root), before)

    def test_undoing_leaves_what_the_migration_did_not_write(self):
        data = self._original()
        self._crash("after the public log was written")
        self.path.write_bytes(b"edited by hand\n")
        for _ in range(2):  # tried again at each launch, with the same outcome
            with self.assertRaises(CliSaveError) as caught:
                prepare_cli_save(self.path, auto_migrate=False)
            message = str(caught.exception)
            self.assertIn(f"undoing it failed: {self.path} was changed after the migration "
                          f"wrote it, so the original save was left at {self.full}", message)
            self.assertIn("rerun with --auto-migrate-save=False", message)
            self.assertEqual(self.full.read_bytes(), data)
            self.assertEqual(self.path.read_bytes(), b"edited by hand\n")
            self.assertTrue((self.private / "migration.json").exists())

    def test_a_journal_that_names_other_files_is_refused(self):
        self._original()
        self._crash("after the log moved")
        journal = self.private / "migration.json"
        record = json.loads(journal.read_text())
        record["moves"].append(["../../elsewhere", "context.jsonl"])
        journal.write_text(json.dumps(record))
        before = _tree(self.root)
        with self.assertRaises(CliSaveError) as caught:
            prepare_cli_save(self.path)
        self.assertIn(f"its journal {journal} can't be used (it names a file that a migration "
                      "doesn't move)", str(caught.exception))
        self.assertEqual(_tree(self.root), before)

    def test_a_crash_while_the_journal_was_written_leaves_nothing_to_undo(self):
        # The journal goes into place whole: a crash before that leaves only
        # its temporary file, and nothing moved.
        data = self._original()
        self.private.mkdir(mode=0o700)
        (self.private / "migration.json.tmp").write_bytes(b'{"version": 1, "sou')
        save = prepare_cli_save(self.path)
        self.assertEqual(len(save.notices), 1)
        self.assertTrue(save.notices[0].startswith("Migrated the save"))
        self.assertEqual(self.full.read_bytes(), data)
        self.assertEqual(sorted(os.listdir(self.private)), ["context.jsonl", *_DEBUG_FILES])

    def test_a_private_directory_that_is_a_link_is_refused(self):
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        self.private.symlink_to(elsewhere)
        with self.assertRaises(CliSaveError) as caught:
            prepare_cli_save(self.path)
        self.assertEqual(str(caught.exception),
                         f"the save's private directory {self.private} is a symbolic link")
        self.assertEqual(os.listdir(elsewhere), [])

    def test_a_new_save_replaces_one_in_the_original_format_without_migrating_it(self):
        self._original()
        save = prepare_cli_save(self.path, resume=False)
        self.assertEqual(save, CliSave.at(self.prefix))
        self.assertEqual(os.listdir(self.private), [])
        self.assertTrue((self.root / "s.trace.req.jsonl").exists())  # left where it was
        CliSaveWriter(save).save(InteractionContext((Init("new"),)))
        self.assertEqual(load_interaction_save(self.full).items, (Init("new"),))
        self.assertEqual(load_interaction_save(self.path).items, (Init("new"),))

    def test_without_auto_migration_a_save_keeps_its_original_format(self):
        data = self._original()
        save = prepare_cli_save(self.path, auto_migrate=False)
        self.assertEqual(save, CliSave(self.path))
        self.assertFalse(self.private.exists())
        context, writer, _ = resume_cli_save(save)
        context.append(Message("user", "more"))
        writer.save(context)
        self.assertEqual(self.path.read_bytes(), data + _encoded((Message("user", "more"),)))
        log = save.compaction_log()
        self.assertEqual((log.path, log.source, log.private), (self.path, None, None))

    def test_the_start_check_changes_nothing(self):
        self._original()
        before = _tree(self.root)
        check_cli_save(self.path)
        self.assertEqual(_tree(self.root), before)
        self.path.write_bytes(b"not a save\n")
        before = _tree(self.root)
        with self.assertRaisesRegex(CliSaveError, "it doesn't load as a save"):
            check_cli_save(self.path)
        check_cli_save(self.path, auto_migrate=False)
        check_cli_save(self.path, resume=False)
        self.assertEqual(_tree(self.root), before)

    def test_a_reboot_onto_a_save_that_cant_be_migrated_fails_its_start_check(self):
        save_interaction_save(self.root / "real.jsonl", InteractionContext(_HISTORY))
        self.path.symlink_to(self.root / "real.jsonl")
        for argv, status in (([], 1), (["--auto-migrate-save=False"], 0)):
            with self.subTest(argv=argv), redirect_stdout(io.StringIO()), \
                    redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(cli.main(["--headless", "--save", str(self.path),
                                           "--rebooted=check", *argv]), status)
            self.assertEqual("is a symbolic link" in stderr.getvalue(), bool(status))
            self.assertTrue(self.path.is_symlink())
            self.assertFalse(self.private.exists())

    def test_an_existing_save_in_either_format(self):
        self.assertFalse(cli_save_exists(self.path) or cli_save_exists(self.prefix))
        self._original()
        self.assertTrue(cli_save_exists(self.path) and cli_save_exists(self.prefix))
        prepare_cli_save(self.path)
        self.path.unlink()  # the public log is informational
        self.assertTrue(cli_save_exists(self.path) and cli_save_exists(self.prefix))


def _after(function, count):
    """``function``, which dies after it has run ``count`` times."""
    calls = []

    def run(*args, **kwargs):
        result = function(*args, **kwargs)
        calls.append(args)
        if len(calls) == count:
            raise _Crash()
        return result
    return run


def _remove_tree(path):
    for child in path.iterdir():
        _remove_tree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
    path.rmdir()


class ColdstartTests(_FilesTestCase):
    def test_the_pointer_names_the_public_log_and_the_check_reads_the_full_log(self):
        save = CliSave.at(self.prefix)
        self.private.mkdir()
        context = InteractionContext(_HISTORY)
        CliSaveWriter(save).save(context)
        where = SimpleNamespace(cwd=self.root)
        for damage in (lambda: None, lambda: self.path.write_bytes(b"garbage\n"), self.path.unlink):
            damage()
            result = ColdstartCompactor(save.compaction_log(where)).compact(context, tools=(_EXEC,))
            pointer = result.items[0].prefix_items[-1]
            self.assertTrue(is_coldstart_pointer(pointer))
            self.assertIn("saved in `s.jsonl` (relative to your tools' working directory), "
                          "lines 1\u20138; later lines continue from here.", pointer.content)
            self.assertIn('"context_prefix" (an earlier compaction) or "nil" (an item left out '
                          'of this log).', pointer.content)
            self.assertIn("A summary template is in `s.summary.md`; it covers no lines yet. "
                          "Ignore the contents of `s._private/`; they are private.",
                          pointer.content)
        self.full.write_bytes(_encoded(_HISTORY[:3]))
        with self.assertRaisesRegex(CompactionLogError, "the save s._private/context.jsonl "
                                    "has 3 lines, fewer than the context's 8 items"):
            ColdstartCompactor(save.compaction_log(where)).compact(context, tools=(_EXEC,))

    def test_a_legacy_save_s_pointer_is_unchanged(self):
        save_interaction_save(self.path, InteractionContext(_HISTORY))
        log = CliSave(self.path).compaction_log(SimpleNamespace(cwd=self.root))
        result = ColdstartCompactor(log).compact(InteractionContext(_HISTORY), tools=(_EXEC,))
        text = result.items[0].prefix_items[-1].content
        self.assertIn('"tool_result" ("output") or "context_prefix" (an earlier compaction).', text)
        self.assertNotIn('"nil"', text)
        self.assertNotIn("_private", text)


class CLISaveLayoutTests(_ControllerTestCase):
    def setUp(self):
        super().setUp()
        self.full = full_log_path(self.path)
        self.private = self.full.parent

    def notices(self, terminal):
        return [item.text for item in terminal.items if item.text.startswith("[cli] ")]

    async def test_reasoning_stays_in_the_private_directory(self):
        model = _Model(self.path, ModelSample(items=(_REASONING, Message("assistant", "done"))))
        terminal = _Terminal(_quit_when_idle)
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 0)
        full, public = self.full.read_bytes(), self.path.read_bytes()
        self.assertIn(_REASONING, load_interaction_save(self.full).items)
        self.assertEqual(public, b"".join(public_log_lines(full)))
        self.assertIn(_NIL, public)
        self.assertNotIn(_SECRET.encode(), public)
        self.assertEqual(stat.S_IMODE(self.private.stat().st_mode), 0o700)
        self.assertEqual(sorted(os.listdir(self.path.parent)),
                         ["interaction._private", "interaction.jsonl"])
        notices = self.notices(terminal)
        self.assertIn(f"[cli] Save log: {self.path}", notices)
        self.assertIn(f"[cli] Private: {self.private}/, with the full log, which has the "
                      "reasoning that the save log leaves out, and debug files.", notices)

    async def test_an_edited_public_log_is_kept_and_resuming_reads_the_full_log(self):
        self.assertEqual(await self._run(_Model(self.path, _answer("first")),
                                         _Terminal(_quit_when_idle), ["--prompt", "one"]), 0)
        lines = self.path.read_bytes().split(b"\n")
        edited_line = next(index for index, line in enumerate(lines) if b'"first"' in line)
        lines[edited_line] = lines[edited_line].replace(b'"first"', b'"FIRST, edited"')
        edited = b"\n".join(lines)
        self.path.write_bytes(edited)
        model = _Model(self.path, _answer("second"))
        terminal = _Terminal(_quit_when_idle)
        self.assertEqual(await self._run(model, terminal, ["--resume", "--prompt", "two"]), 0)
        self.assertIn(Message("assistant", "first"), model.calls[0][0].items)
        public = self.path.read_bytes()
        self.assertTrue(public.startswith(edited))
        self.assertIn(b'"second"', public[len(edited):])
        self.assertEqual([text for text in self.notices(terminal) if "full log" in text and
                          "interaction.jsonl" in text], [
            f"[cli] Note: interaction.jsonl differs from the full log from line {edited_line + 1} "
            "on, as when it is edited. It is informational, so it was kept as it is: new lines "
            "are appended after its end, and its line numbers may not match the full log's."])

    async def test_a_missing_or_lagging_public_log_is_brought_up_to_date(self):
        self.assertEqual(await self._run(_Model(self.path, _answer("first")),
                                         _Terminal(_quit_when_idle), ["--prompt", "one"]), 0)
        complete = self.path.read_bytes()
        for damage, note in ((self.path.unlink, "[cli] Note: interaction.jsonl was missing; it "
                                                "was written anew from the full log."),
                             (lambda: self.path.write_bytes(complete[:-10]), None)):
            with self.subTest(note=note):
                damage()
                terminal = _Terminal(_quit_when_idle)
                self.assertEqual(await self._run(_Model(self.path), terminal, ["--resume"]), 0)
                self.assertEqual(self.path.read_bytes(), complete)
                self.assertEqual([text for text in self.notices(terminal)
                                  if "interaction.jsonl" in text and "Save log" not in text],
                                 [note] if note else [])

    async def test_a_public_log_that_cant_be_written_never_fails_the_session(self):
        from pythia.interaction import save as save_module
        replace = save_module._replace_atomically

        def read_only(destination, data):
            if Path(destination) == self.path:
                error = SaveError(f"could not write save to {destination}")
                error.__cause__ = OSError(errno.EROFS, "Read-only file system")
                raise error
            return replace(destination, data)

        terminal = _Terminal(_quit_when_idle)
        with mock.patch.object(save_module, "_replace_atomically", side_effect=read_only):
            self.assertEqual(await self._run(_Model(self.path, _answer()), terminal,
                                             ["--prompt", "hello"]), 0)
        self.assertFalse(self.path.exists())
        self.assertIsInstance(load_interaction_save(self.full).items[-1], TurnSummary)
        self.assertEqual([text for text in self.notices(terminal) if "could not write" in text], [
            "[cli] Warning: could not write interaction.jsonl (Read-only file system). It is "
            "informational: the save is unaffected, and the next save tries again."])
        self.assertFalse(any("Checkpoint failed" in text for text in self.notices(terminal)))
        self.assertEqual(await self._run(_Model(self.path), _Terminal(_quit_when_idle),
                                         ["--resume"]), 0)
        self.assertEqual(self.path.read_bytes(), b"".join(public_log_lines(self.full.read_bytes())))

    async def test_a_save_kept_in_its_original_format(self):
        seed_cli_save(self.path, InteractionContext(_HISTORY))
        model = _Model(self.path, ModelSample(items=(_REASONING, Message("assistant", "again"))))
        terminal = _Terminal(_quit_when_idle)
        self.assertEqual(await self._run(model, terminal, [
            "--resume", "--auto-migrate-save=False", "--prompt", "more"]), 0)
        self.assertFalse(self.private.exists())
        saved = load_interaction_save(self.path).items
        self.assertEqual(saved[:len(_HISTORY)], _HISTORY)
        self.assertEqual(saved.count(_REASONING), 2)
        self.assertIn("[cli] Note: this save keeps its original format (--auto-migrate-save="
                      "False): it has the reasoning, and debug files go next to it.",
                      self.notices(terminal))

    async def test_compact_points_to_the_public_log_even_when_it_was_edited(self):
        environment = Environment((Tool(_EXEC, lambda arguments, **kwargs: ToolOutcome("ran")),))
        argv = ["--compaction-mode", "coldstart"]
        model = _Model(self.path, ModelSample(items=(_REASONING, Message("assistant", "first"))))
        self.assertEqual(await self._run(model, _Terminal(_quit_when_idle),
                                         [*argv, "--prompt", "one"], environment), 0)
        with self.path.open("ab") as file:
            file.write(b'{"note": "added by hand"}\n')
        terminal = _Terminal(_submit_then_quit(
            "/compact", "Context compacted using a coldstart pointer"))
        self.assertEqual(await self._run(_Model(self.path), terminal, [*argv, "--resume"],
                                         environment), 0)
        saved = load_interaction_save(self.full).items
        lines = next(index for index, item in enumerate(saved) if isinstance(item, UserToolCall))
        checkpoint = next(item for item in reversed(saved) if isinstance(item, ContextPrefix))
        pointer = checkpoint.prefix_items[-1]
        self.assertTrue(is_coldstart_pointer(pointer))
        self.assertIn(f"saved in `{self.path}`, lines 1\u2013{lines};", pointer.content)
        self.assertIn('or "nil" (an item left out of this log).', pointer.content)
        self.assertIn(f"Ignore the contents of `{self.private}/`; they are private.",
                      pointer.content)
        self.assertTrue(self.path.with_suffix(".summary.md").is_file())
        self.assertIn("[cli] Note: the coldstart pointer names interaction.jsonl, which differs "
                      "from the full log, so its line numbers may not match the pointer's.",
                      self.notices(terminal))

    async def test_a_coldstart_reboot_onto_a_save_in_the_original_format(self):
        # As a process on older code leaves it: /compact's pointer names lines
        # of the save, reasoning included, which line up in the public log.
        pointer = Message("user", f"{COLDSTART_POINTER_PREFIX} `{self.path}`, lines 1\u20138; "
                                  "later lines continue from here.")
        original = (*_HISTORY, UserToolCall(ToolCall("compact", "user_1", "{}")),
                    UserToolResult(ToolResult("user_1", "compacted")), ContextPrefix((pointer,)),
                    CompactionMetadata(TokenUsage(), "coldstart"))
        seed_cli_save(self.path, InteractionContext(original))
        data = self.path.read_bytes()
        model = _Model(self.path, _answer("summary updated"))
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" and any(
            "summary updated" in item.text for item in t.items) else None)
        args = cli._build_parser().parse_args(["--resume"])
        self.assertEqual(await asyncio.wait_for(cli._run(
            model, Environment(), terminal, args, self.path,
            rebooted=cli._Rebooted(cause="coldstart")), 4), 0)
        [(context, _, _)] = model.calls
        self.assertEqual(context.items, original)  # it follows the pointer
        full = self.full.read_bytes()
        self.assertTrue(full.startswith(data))
        public = self.path.read_bytes()
        self.assertEqual(public, b"".join(public_log_lines(full)))
        pairs = list(zip(public.split(b"\n")[:8], full.split(b"\n")[:8]))
        self.assertEqual([index for index, (mine, theirs) in enumerate(pairs) if mine != theirs], [4])
        self.assertEqual(pairs[4][0] + b"\n", _NIL)
        self.assertTrue(any(text.startswith(f"[cli] Migrated the save {self.path} ")
                            for text in self.notices(terminal)))


if __name__ == "__main__":
    unittest.main()
