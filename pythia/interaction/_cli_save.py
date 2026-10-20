"""The CLI's save, which ``--save PREFIX`` names: a public log and a private directory.

A trailing ``.jsonl`` is ignored, so ``--save review.jsonl`` names the same
save as ``--save review``, which has these files:

- ``PREFIX.jsonl``, the public log (the save log): the full log line for
  line, with each reasoning item shown as ``save_log_reasoning`` says while
  its line is written: a ``{"type": "nil"}`` line, or an assistant message
  with its summary (``save.PublicLogWriter``). It is informational: editing
  it out-of-band at most causes a warning. Resuming and compaction never
  read it; coldstart's pointer names it.
- ``PREFIX.summary.md``, coldstart compaction's summary, as before.
- ``PREFIX._private/``, the private directory (mode 0700): ``context.jsonl``,
  the full log, which is all that resuming reads; ``--debug-trace``'s
  ``trace.req.jsonl``, ``trace.res.jsonl`` and ``trace.events.jsonl``;
  and ``--debug-save-model-binding``'s ``context.jsonl.model-binding.json``.

A save in the original format is one file, its full log, at the ``--save``
path or at ``PREFIX.jsonl``, with its traces and snapshot next to it. When
the CLI resumes one, it migrates it, unless ``--auto-migrate-save=False``:
the file moves to ``PREFIX._private/context.jsonl`` with its traces and
snapshot, and its public log is written. Before it changes anything, the
migration writes a journal of what it will do, ``PREFIX._migration.json``,
which it removes last. One that fails is undone before the CLI exits with an
error that names ``--auto-migrate-save=False``, and one that was interrupted
is undone at the next launch. With ``--auto-migrate-save=False`` the CLI
keeps using such a save as it is, as a legacy save.

A save whose private directory exists is in the current format, so the CLI
never resumes a file at ``PREFIX.jsonl`` as a full log while the private
directory lacks ``context.jsonl``: that file is the save log, which may load
as a save once reasoning shows as assistant messages.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from ._debug_trace import TRACE_EVENT_SUFFIX
from ._debug_trace import TRACE_REQUEST_SUFFIX
from ._debug_trace import TRACE_RESPONSE_SUFFIX
from ._model_binding_debug import debug_model_binding_path
from .compaction import CompactionLog
from .compaction import compaction_log
from .context import InteractionContext
from .save import IncompleteSaveLine
from .save import InteractionSaveWriter
from .save import PublicLogError
from .save import PublicLogWriter
from .save import SaveError
from .save import _fsync_directory
from .save import _write_all
from .save import public_log_lines
from .save import read_interaction_save
from .save import resume_interaction_save_lines


PUBLIC_SUFFIX = ".jsonl"
PRIVATE_SUFFIX = "._private"
CONTEXT_NAME = "context.jsonl"
# A migration's journal is PREFIX._migration.json, next to the save, while
# the migration runs. It is written before the private directory is made.
JOURNAL_SUFFIX = "._migration.json"
# The option that keeps a save in the original format, as messages name it.
NO_MIGRATION_OPTION = "--auto-migrate-save=False"
_JOURNAL_VERSION = 1
# A migration writes the public log here, then moves it into place.
_PUBLIC_TEMPORARY = "public.jsonl.tmp"
_TRACE_SUFFIXES = (TRACE_REQUEST_SUFFIX, TRACE_RESPONSE_SUFFIX, TRACE_EVENT_SUFFIX)
# --debug-trace's logs in the private directory: the trace suffixes without
# their leading dot, trace.req.jsonl, trace.res.jsonl and trace.events.jsonl.
_TRACE_NAMES = tuple(suffix[1:] for suffix in _TRACE_SUFFIXES)
_NEW_FILE_FLAGS = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                   | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0))


class CliSaveError(Exception):
    """A save that the CLI can't use as it is; the message says why, and what to do."""


class _Unframed(CliSaveError):
    """A reason not to migrate that is not a migration failure: shown as it is."""


def save_prefix(path) -> Path:
    """The save prefix that a ``--save`` path names: the path without a trailing ``.jsonl``."""
    path = Path(path)
    name = path.name
    if name.endswith(PUBLIC_SUFFIX) and len(name) > len(PUBLIC_SUFFIX):
        return path.with_name(name[:-len(PUBLIC_SUFFIX)])
    return path


def public_log_path(prefix) -> Path:
    """``PREFIX.jsonl``, the public log."""
    prefix = Path(prefix)
    return prefix.with_name(prefix.name + PUBLIC_SUFFIX)


def private_dir(prefix) -> Path:
    """``PREFIX._private``, the private directory."""
    prefix = Path(prefix)
    return prefix.with_name(prefix.name + PRIVATE_SUFFIX)


def full_log_path(path) -> Path:
    """The full log of the save that a ``--save`` path names, in the current format."""
    return private_dir(save_prefix(path)) / CONTEXT_NAME


def migration_journal_path(prefix) -> Path:
    """``PREFIX._migration.json``, the journal of a migration of the save at ``prefix``."""
    prefix = Path(prefix)
    return prefix.with_name(prefix.name + JOURNAL_SUFFIX)


@dataclass(frozen=True)
class CliSave:
    """Where a CLI save's files are.

    ``log`` is the full log, which resuming reads. A save in the current
    format has its ``public`` log and its ``private`` directory. A legacy
    save, a save in the original format kept as it is, has neither: ``log``
    is its one file. ``notices`` say what preparing the save did, such as a
    migration, for the CLI to show.
    """

    log: Path
    public: Optional[Path] = None
    private: Optional[Path] = None
    notices: Tuple[str, ...] = ()

    @classmethod
    def at(cls, prefix, notices=()) -> "CliSave":
        """The save at ``prefix``, in the current format."""
        private = private_dir(prefix)
        return cls(private / CONTEXT_NAME, public_log_path(prefix), private, tuple(notices))

    @property
    def legacy(self) -> bool:
        return self.public is None

    @property
    def shown(self) -> Path:
        """The save's path to show: its public log, or a legacy save's file."""
        return self.log if self.public is None else self.public

    @property
    def log_name(self) -> str:
        """The full log's name in notices: ``PREFIX._private/context.jsonl``, or a legacy file's."""
        if self.private is None:
            return self.log.name
        return f"{self.private.name}/{self.log.name}"

    @property
    def binding_path(self) -> Path:
        """Where ``--debug-save-model-binding`` writes its snapshot."""
        return debug_model_binding_path(self.log)

    @property
    def trace_paths(self) -> Tuple[Path, Path, Path]:
        """Where ``--debug-trace`` writes its request, response and event logs.

        ``trace.req.jsonl``, ``trace.res.jsonl`` and ``trace.events.jsonl`` in
        the private directory; next to a legacy save's file, its path with
        ``.trace.req.jsonl`` and so on in place of its extension.
        """
        if self.private is None:
            return tuple(self.log.with_suffix(suffix) for suffix in _TRACE_SUFFIXES)
        return tuple(self.private / name for name in _TRACE_NAMES)

    def compaction_log(self, environment=None) -> Optional[CompactionLog]:
        """The log that coldstart points to: the public log, checked against the full log."""
        if self.public is None:
            return compaction_log(self.log, environment)
        return compaction_log(self.public, environment, source=self.log, private=self.private)


# Finding and preparing a save.


@dataclass(frozen=True)
class _Found:
    """What a ``--save`` path names (``_inspect``)."""

    prefix: Path
    journal: bool  # a migration's journal: one was interrupted
    current: bool  # a full log in the current format
    original: Optional[Path]  # otherwise, a save in the original format, or None
    private: bool  # the private directory exists


def _require_private_dir(private: Path) -> None:
    """Raise unless the private directory, where it exists, is a directory and not a link."""
    if os.path.islink(private):
        raise CliSaveError(f"the save's private directory {private} is a symbolic link")
    if os.path.lexists(private) and not private.is_dir():
        raise CliSaveError(f"the save's private directory {private} is not a directory")


def _inspect(argument: Path) -> _Found:
    """Find the save that a resolved ``--save`` path names, changing nothing."""
    prefix = save_prefix(argument)
    public, private = public_log_path(prefix), private_dir(prefix)
    log = private / CONTEXT_NAME
    _require_private_dir(private)
    if public.exists() and not public.is_file():
        raise CliSaveError(f"the save's public log {public} is not a regular file")
    if os.path.lexists(log) and not log.is_file():
        raise CliSaveError(f"the save's full log {log} is not a regular file")
    journal = os.path.lexists(migration_journal_path(prefix))
    current = os.path.lexists(log)
    made = os.path.isdir(private)
    if journal or current:
        return _Found(prefix, journal, current, None, made)
    # In the original format, the save was the file at the --save path, which
    # is the public log's path unless the path lacks .jsonl.
    candidates = [path for path in dict.fromkeys((argument, public)) if path.is_file()]
    if len(candidates) > 1:
        raise CliSaveError(
            f"both {argument} and {public} exist. --save names the save prefix {prefix}, "
            f"whose log is {public}, but {argument} would be its save in the original "
            "format: rename one of them, or choose another --save prefix"
        )
    return _Found(prefix, False, False, candidates[0] if candidates else None, made)


def _missing_full_log(found: _Found) -> CliSaveError:
    """Why a resumed save whose private directory lacks its full log can't be used."""
    private = private_dir(found.prefix)
    log = private / CONTEXT_NAME
    return CliSaveError(
        f"the save's full log {log} is missing, but its private directory is there, so "
        f"{found.original} is taken for its save log, not a full log: restore the full log, "
        f"or copy the save log to {log} to resume from it as it is, or start a new save with "
        f"--resume=False. If {found.original} is a save in the original format, as when a "
        f"--resume=False launch stopped before its first save, remove {private} and rerun"
    )


def prepare_cli_save(argument, *, resume: bool = True, auto_migrate: bool = True,
                     reasoning: str = "nil") -> CliSave:
    """Make the save that a resolved ``--save`` path names ready to use.

    An interrupted migration is undone first. A save in the original format
    is then migrated when it is resumed (``resume``) and ``auto_migrate`` is
    true; kept as it is, as a legacy save, when ``auto_migrate`` is false;
    and otherwise replaced by a new save, as ``--resume=False`` replaces any
    save. A migration writes the public log with ``reasoning``
    (``save.PUBLIC_LOG_REASONING``). A resumed save whose private directory
    lacks its full log is refused (``_missing_full_log``). A save in the
    current format gets its private directory when it lacks one. Raises
    ``CliSaveError``.
    """
    argument = Path(argument)
    found = _inspect(argument)
    notices: List[str] = []
    if found.journal:
        notices.append(_undo_interrupted(found.prefix))
        found = _inspect(argument)
    if found.original is not None:
        if resume and found.private:
            raise _missing_full_log(found)
        if not auto_migrate:
            return CliSave(found.original, notices=tuple(notices))
        if resume:
            notices.append(_migrate(found.prefix, found.original, reasoning))
    save = CliSave.at(found.prefix, notices)
    _make_private(save.private)
    return save


def check_cli_save(argument, *, resume: bool = True, auto_migrate: bool = True) -> None:
    """Check, changing nothing, that ``prepare_cli_save`` would find the save usable.

    A migration is checked as far as it can be without being made: the save
    and its debug files are regular files, and the save loads. Undoing an
    interrupted migration is left to ``prepare_cli_save``. Raises
    ``CliSaveError``.
    """
    found = _inspect(Path(argument))
    if found.original is not None and resume:
        if found.private:
            raise _missing_full_log(found)
        if auto_migrate:
            _checked_plan(found.prefix, found.original)


def cli_save_exists(argument) -> bool:
    """Whether a resolved ``--save`` path names a save, in either format; changes nothing."""
    argument = Path(argument)
    prefix = save_prefix(argument)
    private = private_dir(prefix)
    return (os.path.lexists(private / CONTEXT_NAME)
            or os.path.lexists(migration_journal_path(prefix))
            or argument.is_file() or public_log_path(prefix).is_file())


def _make_private(private: Path) -> None:
    try:
        os.mkdir(private, 0o700)
    except FileExistsError:
        _require_private_dir(private)
    except OSError as exc:
        raise CliSaveError(
            f"could not create the save's private directory {private}: {_reason(exc)}") from None


def _reason(error: BaseException) -> str:
    if isinstance(error, OSError):
        reason = error.strerror or str(error) or type(error).__name__
        return f"{reason}: {error.filename}" if error.filename else reason
    return str(error) or type(error).__name__


# Migration.


@dataclass(frozen=True)
class _Plan:
    """A migration, as ``_plan`` checked it before anything moves."""

    source: Path  # the save in the original format
    log: Path  # its new path, as the full log
    public: Path  # its public log, to write
    moves: Tuple[Tuple[Path, Path], ...]  # its debug files, to move along
    left: Tuple[Path, ...]  # its debug files left in place: their new paths are taken
    public_data: bytes
    status: os.stat_result  # the source's, when it was read
    size: int  # the bytes read

    @property
    def private(self) -> Path:
        return self.log.parent


@dataclass(frozen=True)
class _Journal:
    """A migration's journal: what ``_undo`` needs to undo it."""

    path: Path  # the journal's own
    source: Path
    log: Path
    public: Path
    moves: Tuple[Tuple[Path, Path], ...]
    device: int  # the source's st_dev and st_ino: the file that moves to log
    inode: int
    public_sha256: str  # of the public log that the migration writes

    @property
    def private(self) -> Path:
        return self.log.parent


def _companions(source: Path, private: Path) -> Iterator[Tuple[Path, Path]]:
    """A save's debug files: at their paths next to ``source``, and in ``private``."""
    for suffix, name in zip(_TRACE_SUFFIXES, _TRACE_NAMES):
        yield source.with_suffix(suffix), private / name
    yield debug_model_binding_path(source), debug_model_binding_path(private / CONTEXT_NAME)


def _require_regular(path: Path, what: str) -> None:
    status = os.lstat(path)
    if stat.S_ISLNK(status.st_mode):
        raise CliSaveError(f"{what} {path} is a symbolic link")
    if not stat.S_ISREG(status.st_mode):
        raise CliSaveError(f"{what} {path} is not a regular file")


def _plan(prefix: Path, source: Path, reasoning: str = "nil") -> _Plan:
    """Check a migration of ``source`` without making it; raise ``CliSaveError`` with the reason."""
    public, private = public_log_path(prefix), private_dir(prefix)
    log = private / CONTEXT_NAME
    moves, left = [], []
    try:
        _require_regular(source, "the save")
        for old, new in _companions(source, private):
            if not os.path.lexists(old):
                continue
            _require_regular(old, "its debug file")
            if os.path.lexists(new):
                left.append(old)
            else:
                moves.append((old, new))
        read = read_interaction_save(source)
    except SaveError as exc:
        if isinstance(exc.__cause__, PublicLogError):
            raise _Unframed(
                f"the save's full log {log} is missing, and {source} is its save log, not "
                "a full log: restore the full log, or start a new save with --resume=False"
            ) from None
        raise CliSaveError(f"it doesn't load as a save ({exc})") from None
    except OSError as exc:
        raise CliSaveError(_reason(exc)) from None
    complete = read.data[:read.complete_size]
    # The save's items are from earlier sessions: they show only summaries
    # (see save.PublicLogWriter).
    return _Plan(source, log, public, tuple(moves), tuple(left),
                 b"".join(public_log_lines(complete, reasoning)), read.status, len(read.data))


def _checked_plan(prefix: Path, source: Path, reasoning: str = "nil") -> _Plan:
    try:
        return _plan(prefix, source, reasoning)
    except _Unframed:
        raise
    except CliSaveError as exc:
        raise CliSaveError(_failure(source, str(exc), "Nothing was changed.")) from None


def _failure(source: Path, reason: str, outcome: str) -> str:
    return (f"could not migrate the save {source} to the current format: {reason}. {outcome} "
            f"To use the save in its original format, rerun with {NO_MIGRATION_OPTION}.")


def _outcome(problems: List[str], journal: Path) -> str:
    if not problems:
        return "The original save was restored."
    return (f"Restoring the original save failed: {'; '.join(problems)}. The migration's "
            f"journal {journal} lists the files it moved: put them back by hand and remove "
            "the journal first.")


def _migrate(prefix: Path, source: Path, reasoning: str = "nil") -> str:
    """Migrate the save in the original format at ``source``; return a notice that says so.

    The journal comes first, so that a crash at any later point leaves it for
    the next launch to undo the migration. Raises ``CliSaveError`` when it
    fails, once what it did is undone.
    """
    plan = _checked_plan(prefix, source, reasoning)
    private, journal_path = plan.private, migration_journal_path(prefix)
    try:
        _write_journal(journal_path, plan)
    except OSError as exc:
        raise CliSaveError(_failure(source, _reason(exc), "Nothing was changed.")) from None
    try:
        os.mkdir(private, 0o700)
        for old, new in plan.moves:
            _rename_new(old, new)
        _rename_new(plan.source, plan.log)
        _check_unchanged(plan)
        _write_public(plan)
        for directory in {private, plan.public.parent, plan.source.parent}:
            _fsync_directory(directory)
        os.unlink(journal_path)  # the migration is done
        _fsync_directory(journal_path.parent)
    except BaseException as exc:
        problems = _undo(_journal_of(plan, journal_path))
        if not isinstance(exc, Exception):
            raise
        reason = str(exc) if isinstance(exc, CliSaveError) else _reason(exc)
        raise CliSaveError(_failure(source, reason, _outcome(problems, journal_path))) from None
    return _migrated_notice(plan)


def _migrated_notice(plan: _Plan) -> str:
    text = (f"Migrated the save {plan.source} to the current format: its full log, with "
            f"reasoning, is now {plan.log}, and {plan.public} is its save log")
    if plan.moves:
        text += f"; its debug files moved into {plan.private}"
    if plan.left:
        text += ("; " + ", ".join(str(path) for path in plan.left)
                 + f" stayed in place, since their new paths in {plan.private} are taken")
    return text + "."


def _journal_of(plan: _Plan, path: Path) -> _Journal:
    return _Journal(path, plan.source, plan.log, plan.public, plan.moves, plan.status.st_dev,
                    plan.status.st_ino, hashlib.sha256(plan.public_data).hexdigest())


def _write_journal(path: Path, plan: _Plan) -> None:
    """Write a migration's journal, whole and durably, before anything moves.

    It is written beside its path and then renamed into place, so a crash
    leaves either no journal, and nothing moved, or a complete one. It names
    files without their directories (the save prefix's, and the private
    directory), so that ``_read_journal`` can check them.
    """
    journal = _journal_of(plan, path)
    record = {
        "version": _JOURNAL_VERSION,
        "source": journal.source.name,
        "moves": [[old.name, new.name] for old, new in journal.moves],
        "device": journal.device,
        "inode": journal.inode,
        "public_sha256": journal.public_sha256,
    }
    data = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with suppress(FileNotFoundError):
        os.unlink(temporary)  # left by a crash while a journal was written
    descriptor = os.open(temporary, _NEW_FILE_FLAGS, 0o600)
    try:
        try:
            _write_all(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if os.path.lexists(path):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(path))
        os.rename(temporary, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary)
        raise
    _fsync_directory(path.parent)


def _read_journal(prefix: Path) -> _Journal:
    """The journal of an interrupted migration of the save at ``prefix``.

    Raises ``OSError`` or ``ValueError``, also when the journal names a file
    that no migration of this save touches.
    """
    public, private = public_log_path(prefix), private_dir(prefix)
    log = private / CONTEXT_NAME
    path = migration_journal_path(prefix)
    with open(path, "rb") as file:
        record = json.loads(file.read().decode("utf-8"))
    if not isinstance(record, dict) or record.get("version") != _JOURNAL_VERSION:
        raise ValueError("it is not a migration journal of a known version")
    sources = {prefix.name: prefix, public.name: public}
    source = sources.get(record.get("source")) if isinstance(record.get("source"), str) else None
    if source is None:
        raise ValueError("it names another save")
    allowed = {(old.name, new.name): (old, new) for old, new in _companions(source, private)}
    moves = []
    raw_moves = record.get("moves")
    for move in raw_moves if isinstance(raw_moves, list) else (None,):
        key = tuple(move) if isinstance(move, list) else None
        if key not in allowed:
            raise ValueError("it names a file that a migration doesn't move")
        moves.append(allowed[key])
    device, inode = record.get("device"), record.get("inode")
    digest = record.get("public_sha256")
    if not (isinstance(device, int) and isinstance(inode, int) and isinstance(digest, str)):
        raise ValueError("it is incomplete")
    return _Journal(path, source, log, public, tuple(moves), device, inode, digest)


def _rename_new(source: Path, destination: Path) -> None:
    """Rename a file to a path that must be free."""
    if os.path.lexists(destination):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(destination))
    os.rename(source, destination)


def _check_unchanged(plan: _Plan) -> None:
    """Raise if the save changed after it was read, as when another process writes it."""
    now, before = os.lstat(plan.log), plan.status
    if ((now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns)
            != (before.st_dev, before.st_ino, plan.size, before.st_mtime_ns)):
        raise CliSaveError("the save changed while it was being migrated; is another "
                           "process using it?")


def _write_public(plan: _Plan) -> None:
    """Write the public log in the private directory, then move it into place whole."""
    temporary = plan.private / _PUBLIC_TEMPORARY
    with suppress(FileNotFoundError):
        os.unlink(temporary)
    descriptor = os.open(temporary, _NEW_FILE_FLAGS, 0o600)
    try:
        _write_all(descriptor, plan.public_data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _rename_new(temporary, plan.public)


def _same_file(path: Path, device: int, inode: int) -> bool:
    try:
        status = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(status.st_mode) and (status.st_dev, status.st_ino) == (device, inode)


def _written(path: Path, digest: str) -> bool:
    """Whether ``path`` is the public log that a migration wrote: a regular file with its digest."""
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return False
        with open(path, "rb") as file:
            return hashlib.sha256(file.read()).hexdigest() == digest
    except OSError:
        return False


def _undo(journal: _Journal) -> List[str]:
    """Undo a migration as its journal describes it; return what couldn't be undone.

    Nothing that the migration didn't write is replaced or removed: the full
    log moves back only while it is the file that moved there (its device
    and inode), and a public log goes only when it is the one the migration
    wrote. Steps that were never taken are skipped. The private directory,
    which the migration made, goes once it is empty, and the journal goes
    last, only when everything was undone, so that an undo that fails or is
    interrupted is tried again at the next launch, and none leaves a private
    directory without a journal.
    """
    problems: List[str] = []

    def attempt(action: Callable[[], None], failure: str) -> None:
        try:
            action()
        except OSError as exc:
            problems.append(f"{failure} ({_reason(exc)})")

    source, log, public, private = journal.source, journal.log, journal.public, journal.private
    temporary = private / _PUBLIC_TEMPORARY
    if os.path.lexists(temporary):
        attempt(lambda: os.unlink(temporary), f"could not remove {temporary}")
    public_written = _written(public, journal.public_sha256)
    if os.path.lexists(log):
        if not _same_file(log, journal.device, journal.inode):
            problems.append(f"{log} is not the save that the migration moved there, so it "
                            "was left in place")
        elif source == public:
            if os.path.lexists(public) and not public_written:
                problems.append(f"{public} was changed after the migration wrote it, so the "
                                f"original save was left at {log}")
            else:  # this replaces the public log that the migration wrote
                attempt(lambda: os.replace(log, source),
                        f"could not move {log} back to {source}")
        elif os.path.lexists(source):
            problems.append(f"{source} exists again, so the original save was left at {log}")
        else:
            attempt(lambda: os.rename(log, source), f"could not move {log} back to {source}")
    elif not os.path.lexists(source):
        problems.append(f"the original save is at neither {source} nor {log}")
    if source != public and public_written:
        attempt(lambda: os.unlink(public), f"could not remove {public}")
    for old, new in reversed(journal.moves):
        if not os.path.lexists(new):
            continue  # never moved
        if os.path.lexists(old):
            problems.append(f"both {old} and {new} exist, so {new} was left in place")
        else:
            attempt(lambda: os.rename(new, old), f"could not move {new} back to {old}")
    for directory in {private, source.parent}:
        _fsync_directory(directory)
    if problems:
        return problems
    with suppress(OSError):
        os.rmdir(private)  # the migration makes it; it goes once nothing else is in it
    attempt(lambda: os.unlink(journal.path), f"could not remove the journal {journal.path}")
    _fsync_directory(journal.path.parent)
    return problems


def _undo_interrupted(prefix: Path) -> str:
    """Undo a migration that was interrupted; return a notice that says so."""
    path = migration_journal_path(prefix)
    try:
        journal = _read_journal(prefix)
    except (OSError, ValueError) as exc:
        raise CliSaveError(
            f"a migration of the save at {prefix} to the current format was interrupted, "
            f"and its journal {path} can't be used ({_reason(exc)}). Put the save's files "
            "back by hand and remove the journal; then rerun, with "
            f"{NO_MIGRATION_OPTION} to use the save in its original format") from None
    problems = _undo(journal)
    if problems:
        raise CliSaveError(
            f"a migration of the save {journal.source} to the current format was "
            f"interrupted, and undoing it failed: {'; '.join(problems)}. Its journal {path} "
            "lists the files it moved: put them back by hand and remove the journal; then "
            f"rerun with {NO_MIGRATION_OPTION} to use the save in its original format")
    return (f"Note: a migration of the save {journal.source} to the current format was "
            "interrupted; it was undone.")


# Saving and resuming.


class CliSaveWriter:
    """Saves a CLI save: its full log strictly, then its public log at best.

    ``save`` saves the full log with an ``InteractionSaveWriter``, raising
    when that fails, and only then the public log, with a
    ``PublicLogWriter``, which never raises. Its problems become warnings,
    which go to ``on_warning`` once it is set, and wait for
    ``take_warnings`` until then. A legacy save has only its full log.
    ``reasoning`` and ``thinking`` are the public log's (``PublicLogWriter``).
    """

    def __init__(self, save: CliSave, full: Optional[InteractionSaveWriter] = None, *,
                 reasoning: Union[str, Callable[[], str]] = "nil",
                 thinking: bool = False) -> None:
        self.cli_save = save
        self.full = InteractionSaveWriter(save.log) if full is None else full
        self.public = (None if save.public is None else PublicLogWriter(
            save.public, name=save.public.name, reasoning=reasoning, thinking=thinking))
        self.on_warning: Optional[Callable[[str], None]] = None

    @property
    def path(self) -> Path:
        """The full log's path, which resuming reads."""
        return self.full.path

    def save(self, context: InteractionContext) -> None:
        self.full.save(context)
        if self.public is not None:
            self.public.save(context)
            self.report_warnings()

    def sync_public(self, lines: bytes, count: int) -> None:
        """Bring the public log up to date with a resumed full log (``PublicLogWriter.sync``)."""
        if self.public is not None:
            self.public.sync(lines, count)

    def take_warnings(self) -> Tuple[str, ...]:
        return () if self.public is None else self.public.take_warnings()

    def report_warnings(self) -> None:
        """Pass the waiting warnings to ``on_warning``, once it is set."""
        if self.on_warning is not None:
            for warning in self.take_warnings():
                self.on_warning(warning)

    def compaction_log(self, environment=None) -> Optional[CompactionLog]:
        return self.cli_save.compaction_log(environment)

    def drift_note(self) -> Optional[str]:
        """A note for a coldstart pointer to a public log that is known to have drifted."""
        if self.public is None or not self.public.drifted:
            return None
        return (f"Note: the coldstart pointer names {self.public.path.name}, which differs "
                "from the full log, so its line numbers may not match the pointer's.")


def resume_cli_save(
    save: CliSave, *, reasoning: Union[str, Callable[[], str]] = "nil", thinking: bool = False,
) -> Tuple[InteractionContext, CliSaveWriter, Optional[IncompleteSaveLine]]:
    """Load a save from its full log alone, with a writer that continues it.

    The public log is brought up to date at once; its warnings wait in the
    writer. ``reasoning`` and ``thinking`` are as for ``CliSaveWriter``.
    Raises ``SaveError`` as ``resume_interaction_save`` does.
    """
    resumed = resume_interaction_save_lines(save.log)
    writer = CliSaveWriter(save, resumed.writer, reasoning=reasoning, thinking=thinking)
    writer.sync_public(resumed.lines, len(resumed.context))
    return resumed.context, writer, resumed.incomplete_line


__all__ = [
    "CONTEXT_NAME",
    "CliSave",
    "CliSaveError",
    "CliSaveWriter",
    "JOURNAL_SUFFIX",
    "NO_MIGRATION_OPTION",
    "check_cli_save",
    "cli_save_exists",
    "full_log_path",
    "migration_journal_path",
    "prepare_cli_save",
    "private_dir",
    "public_log_path",
    "resume_cli_save",
    "save_prefix",
]
