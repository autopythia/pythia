"""A CLI save's lifetime lock, ``PREFIX._private/1.lock``: whether a CLI runs on the save.

A CLI holds a write lock on the whole file for as long as it runs, interactive
or headless, and the kernel drops the lock with the process: so a lock that's
held means that a CLI runs on the save, and one that's free, that none does,
though one may have crashed there. A reboot keeps it, as the restarted CLI
inherits its descriptor (``adopt``). The file holds a line of JSON that names
the holder: the program, its PID, the node, the kernel's boot ID, and when it
took the lock (``describe``); and what the CLI is doing, as it changes, its
``phase`` and whether an automatic retry is under way (``retrying``), which
the CLI rewrites in place (``SaveLock.note``). ``1`` is the root context's
number, as the contexts plan names the file. A save in the original format
has no private directory, and no lock.

On Linux the lock is an open file description lock (``F_OFD_SETLK``), which a
reader tests without taking it (``F_OFD_GETLK``); elsewhere it's ``flock``,
which a reader tests by taking it, shared, for a moment, so that a CLI that
starts in that moment tries again before it finds the save in use. The two
kinds don't see each other, so a reader tests the kind that a CLI takes on
the same system.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import socket
import struct
import time
from contextlib import suppress
from pathlib import Path
from typing import Optional
from typing import Tuple

LOCK_NAME = "1.lock"
# The program that a holder's line names.
PROGRAM = "pythia"
# Open file description locks, where the system has them (Linux).
_OFD = hasattr(fcntl, "F_OFD_SETLK")
# struct flock: l_type, l_whence, l_start, l_len and l_pid, padded at its end.
_FLOCK = "hhqqi0q"
# With flock, how often a CLI tries to take the lock, 10 ms apart, while a
# reader may hold it, shared, for a moment.
_TRIES = 10
_BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
# The most of a holder's line that a reader reads; and its size, newline
# included, padded with spaces, so that a CLI rewrites it in place, the same
# size, as what it's doing changes.
_MAX_LINE = 4096
_LINE_BYTES = 512
# A holder's fields, each with its kind.
_FIELDS = {"program": str, "pid": int, "node": str, "boot": str, "since": float,
           "phase": str, "retrying": bool}
_FLAGS = os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
_BUSY = (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK)


def lock_path(private) -> Path:
    """The lock file of the save whose private directory is ``private``."""
    return Path(private) / LOCK_NAME


class SaveLock:
    """A save's lock, held for as long as its descriptor stays open; ``owner`` is its
    line, as the holder last wrote it."""

    def __init__(self, path: Path, descriptor: int, owner: Optional[dict] = None) -> None:
        self.path = path
        self.descriptor = descriptor
        self.owner = {} if owner is None else owner

    def fileno(self) -> int:
        return self.descriptor

    def note(self, **fields) -> None:
        """Say more in the holder's line, such as what the CLI is doing, in place of what
        it said; at best: a line that can't be written changes nothing."""
        owner = {**self.owner, **fields}
        if owner != self.owner and self.descriptor >= 0:
            with suppress(OSError):
                _write(self.descriptor, owner)
                self.owner = owner

    def close(self) -> None:
        """Let the lock go; the file stays."""
        if self.descriptor >= 0:
            descriptor, self.descriptor = self.descriptor, -1
            os.close(descriptor)


def take(private) -> Tuple[Optional[SaveLock], Optional[dict]]:
    """Take the lock of the save whose private directory is ``private``, and name this
    process in its line: the lock, and None; or None, and the line of the holder, if a
    CLI holds it already, as ``holder`` gives it. Raises ``OSError`` when the file
    can't be opened, locked or written."""
    path = lock_path(private)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | _FLAGS, 0o600)
    taken = False
    try:
        if not _lock(descriptor):
            return None, _line(descriptor)
        owner = _own()
        _write(descriptor, owner)
        taken = True
        return SaveLock(path, descriptor, owner), None
    finally:
        if not taken:
            os.close(descriptor)


def holder(private) -> Optional[dict]:
    """Whether a CLI holds the lock of the save whose private directory is ``private``:
    None if none does, or there's no lock file; else the line that names the holder,
    with only the fields that make sense, or {} if it can't be read. On Linux it takes
    no lock. Raises ``OSError`` when the file can't be opened or tested."""
    try:
        descriptor = os.open(lock_path(private), os.O_RDONLY | _FLAGS)
    except (FileNotFoundError, NotADirectoryError):
        return None
    try:
        return _line(descriptor) if _held(descriptor) else None
    finally:
        os.close(descriptor)


def adopt(descriptor: int, private) -> Optional[SaveLock]:
    """The lock that a restarted CLI inherited from the process it replaced, at
    ``descriptor``, if that's the lock file of the save whose private directory is
    ``private`` and it still holds the lock; no longer inherited by what the CLI
    starts. Else None, with the descriptor closed, unless it's standard input, output
    or error."""
    if descriptor < 3:
        return None
    path = lock_path(private)
    try:
        mine, there = os.fstat(descriptor), os.stat(path, follow_symlinks=False)
        if (mine.st_dev, mine.st_ino) == (there.st_dev, there.st_ino) and _lock(descriptor):
            os.set_inheritable(descriptor, False)
            return SaveLock(path, descriptor, _line(descriptor))
    except OSError:
        pass
    with suppress(OSError):
        os.close(descriptor)
    return None


def describe(owner: dict) -> str:
    """A holder's line, in words: ``PID 1234 on autopythia, since 2026-11-09 13:20``."""
    words = f"PID {owner['pid']}" if "pid" in owner else "PID unknown"
    if "node" in owner:
        words += f" on {owner['node']}"
    if "since" in owner:
        words += time.strftime(", since %Y-%m-%d %H:%M", time.localtime(owner["since"]))
    return words


def _request(kind: int) -> bytes:
    """A struct flock for a lock of ``kind`` on the whole file."""
    return struct.pack(_FLOCK, kind, os.SEEK_SET, 0, 0, 0)


def _lock(descriptor: int) -> bool:
    """Lock ``descriptor``'s file for writing, without waiting: whether it could. A
    descriptor that holds the lock already holds it still."""
    if _OFD:
        try:
            fcntl.fcntl(descriptor, fcntl.F_OFD_SETLK, _request(fcntl.F_WRLCK))
            return True
        except OSError as exc:
            if exc.errno in _BUSY:
                return False
            raise
    for attempt in range(_TRIES):
        if attempt:
            time.sleep(0.01)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if exc.errno not in _BUSY:
                raise
    return False


def _held(descriptor: int) -> bool:
    """Whether something else holds ``descriptor``'s file locked for writing."""
    if _OFD:
        found = fcntl.fcntl(descriptor, fcntl.F_OFD_GETLK, _request(fcntl.F_WRLCK))
        return struct.unpack(_FLOCK, found)[0] != fcntl.F_UNLCK
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in _BUSY:
            return True
        raise
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    return False


def _own() -> dict:
    """The fields that name this process as the holder."""
    try:
        boot = _BOOT_ID.read_text(encoding="ascii").strip()
    except (OSError, ValueError):
        boot = ""
    return {"program": PROGRAM, "pid": os.getpid(), "node": socket.gethostname(),
            "boot": boot, "since": round(time.time(), 3)}


def _write(descriptor: int, owner: dict) -> None:
    """Write the holder's line, ``owner``, padded, in place of the file's contents."""
    data = json.dumps(owner, ensure_ascii=True).encode("ascii")
    data += b" " * max(0, _LINE_BYTES - 1 - len(data)) + b"\n"
    os.pwrite(descriptor, data, 0)
    os.ftruncate(descriptor, len(data))


def _line(descriptor: int) -> dict:
    """The holder's line in ``descriptor``'s file: its fields that make sense, or {}."""
    try:
        value = json.loads(os.pread(descriptor, _MAX_LINE, 0).decode("utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {name: value[name] for name in _FIELDS if _sensible(name, value.get(name))}


def _sensible(name: str, value) -> bool:
    """Whether ``value`` makes sense as the holder's field ``name``."""
    kind = _FIELDS[name]
    if kind is bool or isinstance(value, bool):
        return kind is bool and isinstance(value, bool)
    if kind is float:
        return isinstance(value, (int, float)) and 0 <= value < 1e11
    if kind is str:
        return isinstance(value, str) and len(value) <= 256
    return isinstance(value, kind)


__all__ = ["LOCK_NAME", "SaveLock", "adopt", "describe", "holder", "lock_path", "take"]
