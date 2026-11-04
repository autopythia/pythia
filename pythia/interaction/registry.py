"""The registry of root contexts: a node's own index of its saves, by symlinks.

The registry is a directory, ``~/.pythia/root/`` by default, that holds only
symbolic links. A root context's names there mirror its save's three parts
(``_cli_save``), so that its key and its short number each name it as a save
prefix does:

- ``KEY.jsonl -> /real/path/PREFIX.jsonl``, the save log;
  ``KEY.summary.md -> /real/path/PREFIX.summary.md``, coldstart compaction's
  summary, which may not exist yet; and ``KEY._private ->
  /real/path/PREFIX._private``, the private directory, with the full log.
- ``N.jsonl -> KEY.jsonl``, ``N.summary.md -> KEY.summary.md`` and
  ``N._private -> KEY._private``, for the short number, so that
  ``~/.pythia/root/14.jsonl`` reads the save log and
  ``~/.pythia/root/14._private/context.jsonl`` the full log.

The save-log links are the registry: their exclusive creates claim keys and
numbers, and only they are read to decide anything. The other links follow
them: each is made after its save-log link, and made again where it's found
missing or stale. A key is read from the full log, never from the save log,
which is only informational.

A context's key is ``context_`` and the 32 hex digits of a random UUID: its
``Init.prefix_id``. The prefix ID of a save made before keys had that stem,
``session_`` and the same kind of digits, is read as the key ``context_``
with its digits; the digits are the identity, and no save is rewritten.

Registering a save leaves the save as it is: a save is registered where it
is, and a save opened away from its registered path has moved when that path
no longer holds its key, and the key's links follow it; when the path still
holds the key, the save is a copy, which isn't registered.

Claims are exclusive creates, so that a crash leaves a gap, never two roots
with one number, and the registry never removes a number. The process that
creates a key's link claims its number; another process that finds the link
without a number waits for it, then claims one itself, and a claim whose
number already names the key adopts it. A root's number is the lowest that
names it, and a number that any link names, even without its save-log link,
isn't claimed again. Readers take no lock. Other names are ignored, such as
the temporary links that re-point a link.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

from ._cli_save import CONTEXT_NAME
from ._cli_save import PRIVATE_SUFFIX
from ._cli_save import PUBLIC_SUFFIX
from .compaction import _SUMMARY_SUFFIX
from .save import interaction_item_from_dict


REGISTRY_ENV = "PYTHIA_REGISTRY"
REGISTRY_OFF = "none"
KEY_STEM = "context_"
# A prefix ID that is a key: context_, or session_ before keys had that stem.
_PREFIX_ID = re.compile(r"(?:context|session)_([0-9a-f]{32})\Z")
_KEY = re.compile(r"context_[0-9a-f]{32}\Z")
_NUMBER = re.compile(r"[1-9][0-9]*\Z")
# The fewest hex digits that name a key, as git abbreviates hashes.
ABBREVIATED_DIGITS = 4
# --context's handles: a short number or a key, with #1, the root itself.
_HANDLE = re.compile(r"(?:(?P<number>[1-9][0-9]*)|(?:context|session)_"
                     rf"(?P<digits>[0-9a-f]{{{ABBREVIATED_DIGITS},32}}))"
                     r"(?:#(?P<context>[1-9][0-9]*))?\Z")
# How long an opener waits for the number that a key's creator claims.
WAIT_SECONDS = 1.0
_POLL_SECONDS = 0.02
# A full log's first line, its Init, is short; a longer one isn't a key's.
_FIRST_LINE_BYTES = 65_536
# The links that follow a root's save-log link, named as its save names its
# other parts: coldstart's summary, and the private directory.
_FOLLOWERS = (_SUMMARY_SUFFIX, PRIVATE_SUFFIX)


def _stem(name: str, suffix: str) -> Optional[str]:
    """``name`` without ``suffix``, or None if it doesn't end with it."""
    if name.endswith(suffix) and len(name) > len(suffix):
        return name[:-len(suffix)]
    return None


class RegistryError(Exception):
    """The registry can't do what was asked; the message says why."""


@dataclass(frozen=True)
class Registration:
    """What registering a save did (``Registry.register``).

    ``status`` is ``new``; ``same``, registered there already; ``moved``, the
    key's link now points here, and pointed to ``previous``; or ``copy``, not
    registered, since the key's registered path, ``home``, still holds it.
    ``home`` is otherwise the save's private directory.
    """

    status: str
    key: str
    number: Optional[int]
    home: Path
    previous: Optional[Path] = None


@dataclass(frozen=True)
class Entry:
    """A short number in the registry, as listed (``Registry.entries``).

    ``state`` is ``ok``; ``missing``, nothing is at ``home``; ``replaced``, the
    full log there names another key; or ``unreadable``, there's no readable
    full log there. ``first`` is a lower number that names the same root.
    """

    number: int
    key: str
    home: Optional[Path]
    state: str
    first: Optional[int] = None


@dataclass(frozen=True)
class Root:
    """A registered root context that ``--context`` names (``Registry.find``)."""

    number: Optional[int]
    key: str
    home: Path

    @property
    def prefix(self) -> Path:
        """The save prefix, which ``--save`` would name."""
        return save_prefix_of(self.home)


def default_registry_path() -> Path:
    return Path.home() / ".pythia" / "root"


def registry_path(option: Optional[str] = None) -> Optional[Path]:
    """The registry's directory, or None when it's off.

    ``option`` (``--registry``), else ``$PYTHIA_REGISTRY``, else
    ``~/.pythia/root``; ``none`` turns the registry off. A relative path is
    taken from the launch directory.
    """
    value = option if option is not None else os.environ.get(REGISTRY_ENV)
    if value is None or not value.strip():
        return default_registry_path()
    if value.strip() == REGISTRY_OFF:
        return None
    return Path(value).expanduser().absolute()


def key_of(prefix_id: object) -> str:
    """The key that a prefix ID names: ``context_`` and its 32 hex digits."""
    match = _PREFIX_ID.match(prefix_id) if isinstance(prefix_id, str) else None
    if match is None:
        raise RegistryError(f"the prefix ID {prefix_id!r} isn't a key, which is {KEY_STEM} "
                            "and 32 hex digits")
    return KEY_STEM + match.group(1)


def key_at(private) -> Optional[str]:
    """The key that the full log in a private directory names, or None."""
    try:
        with open(Path(private) / CONTEXT_NAME, "rb") as log:
            line = log.readline(_FIRST_LINE_BYTES)
        item = interaction_item_from_dict(json.loads(line))
    except (OSError, ValueError, TypeError):
        return None
    with suppress(RegistryError):
        return key_of(getattr(item, "prefix_id", None))
    return None


def save_prefix_of(private: Path) -> Path:
    """The save prefix of a private directory: its path without ``._private``."""
    name = private.name
    if name.endswith(PRIVATE_SUFFIX) and len(name) > len(PRIVATE_SUFFIX):
        return private.with_name(name[:-len(PRIVATE_SUFFIX)])
    return private


def _link(path: Path) -> Optional[str]:
    try:
        return os.readlink(path)
    except OSError:
        return None


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _state(home: Path, key: str) -> str:
    if not os.path.lexists(home):
        return "missing"
    held = key_at(home)
    if held == key:
        return "ok"
    return "unreadable" if held is None else "replaced"


def abbreviate(key: str, keys: Iterable[str]) -> str:
    """The shortest abbreviation of ``key`` that no other of ``keys`` shares."""
    digits = key[len(KEY_STEM):]
    others = [other[len(KEY_STEM):] for other in keys if other != key]
    for size in range(ABBREVIATED_DIGITS, len(digits)):
        if not any(other.startswith(digits[:size]) for other in others):
            return KEY_STEM + digits[:size]
    return key


class Registry:
    """The registry at ``path``, a directory that ``register`` makes when it's missing."""

    def __init__(self, path) -> None:
        self.path = Path(path)

    def _names(self) -> List[str]:
        try:
            return os.listdir(self.path)
        except FileNotFoundError:
            return []

    def keys(self) -> List[str]:
        """The keys that the registry has save-log links for."""
        return [stem for stem in (_stem(name, PUBLIC_SUFFIX) for name in self._names())
                if stem is not None and _KEY.match(stem)]

    def abbreviation(self, key: str) -> str:
        return abbreviate(key, self.keys())

    def _prefix(self, key: str) -> Optional[Path]:
        """The save prefix whose save log the key's link names, or None."""
        target = _link(self.path / (key + PUBLIC_SUFFIX))
        stem = None if target is None else _stem(target, PUBLIC_SUFFIX)
        if stem is None:
            return None
        return Path(stem) if os.path.isabs(stem) else self.path / stem

    def _home(self, key: str) -> Optional[Path]:
        """The private directory of the save whose save log the key's link names."""
        prefix = self._prefix(key)
        return None if prefix is None else prefix.with_name(prefix.name + PRIVATE_SUFFIX)

    def _scan(self, key: Optional[str]) -> Tuple[List[int], int]:
        """The numbers that name ``key``, and the highest number, from one listing.

        A number's save-log link names its root. A number that only another of
        its links names still counts toward the highest.
        """
        mine, highest = [], 0
        for name in self._names():
            for suffix in (PUBLIC_SUFFIX, *_FOLLOWERS):
                stem = _stem(name, suffix)
                if stem is None or not _NUMBER.match(stem):
                    continue
                highest = max(highest, int(stem))
                if (suffix == PUBLIC_SUFFIX and key is not None
                        and _link(self.path / name) == key + PUBLIC_SUFFIX):
                    mine.append(int(stem))
        return sorted(mine), highest

    def number_of(self, key: str) -> Optional[int]:
        """The root's number: the lowest that names its key, or None."""
        mine, _ = self._scan(key)
        return mine[0] if mine else None

    def _claim(self, key: str, *, alone: bool = False) -> int:
        """Claim a number for ``key``; ``alone``: this process created the key's link."""
        mine, highest = self._scan(key)
        if mine:
            return mine[0]
        number = highest + 1
        while True:
            link = self.path / f"{number}{PUBLIC_SUFFIX}"
            try:
                os.symlink(key + PUBLIC_SUFFIX, link)
                break
            except FileExistsError:
                if _link(link) == key + PUBLIC_SUFFIX:
                    break  # another opener claimed it for this key
                number += 1
        # Others claim only for a key whose creator stopped: the lowest wins.
        return number if alone else (self.number_of(key) or number)

    def _wait_for_number(self, key: str) -> Optional[int]:
        deadline = time.monotonic() + WAIT_SECONDS
        while True:
            number = self.number_of(key)
            if number is not None or time.monotonic() >= deadline:
                return number
            time.sleep(_POLL_SECONDS)

    def _make(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(FileExistsError):
            os.mkdir(self.path, 0o700)
        if not self.path.is_dir():
            raise RegistryError(f"the registry {self.path} is not a directory")

    def _point(self, link: Path, target) -> None:
        """Re-point ``link`` at ``target`` atomically: a new link renamed over it."""
        temporary = link.with_name(f".{link.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        os.symlink(target, temporary)
        try:
            os.replace(temporary, link)
        except BaseException:
            with suppress(OSError):
                os.unlink(temporary)
            raise

    def _ensure(self, link: Path, target: str) -> None:
        """Make ``link`` point to ``target``: create it, or re-point it."""
        if _link(link) == target:
            return
        try:
            os.symlink(target, link)
        except FileExistsError:
            if _link(link) != target:
                self._point(link, target)

    def _follow(self, key: str, number: Optional[int]) -> None:
        """Make the links that follow the root's save-log links agree with them.

        A failure leaves the root registered by its save-log links, and its
        next registration tries again.
        """
        with suppress(OSError):
            for _ in range(3):  # until the key's link stays as it was read
                prefix = self._prefix(key)
                if prefix is None:
                    return
                for suffix in _FOLLOWERS:
                    self._ensure(self.path / (key + suffix), f"{prefix}{suffix}")
                    if number is not None:
                        self._ensure(self.path / f"{number}{suffix}", key + suffix)
                if self._prefix(key) == prefix:
                    return

    def register(self, prefix_id: str, private) -> Registration:
        """Register the save whose private directory is ``private``, under its key.

        The save-log links come first: the key's, whose exclusive create
        claims the root, then the number's. The links that follow them come
        last (``_follow``). Raises ``RegistryError``, or ``OSError`` when the
        registry can't be read or written.
        """
        key = key_of(prefix_id)
        prefix = save_prefix_of(Path(private))
        # The directory's real path, not its files': the links' targets are
        # then the save's own parts, side by side, as the CLI finds them.
        real = Path(os.path.realpath(prefix.parent)) / prefix.name
        log = real.with_name(real.name + PUBLIC_SUFFIX)
        real_private = real.with_name(real.name + PRIVATE_SUFFIX)
        self._make()
        link = self.path / (key + PUBLIC_SUFFIX)
        try:
            os.symlink(log, link)
        except FileExistsError:
            pass
        else:
            number = self._claim(key, alone=True)
            self._follow(key, number)
            return Registration("new", key, number, real_private)
        home = self._home(key)
        if home is None:
            raise RegistryError(f"{link} is not a symbolic link to a save log")
        previous = None
        if home == real_private or _same_file(home, real_private):
            status = "same"
        elif key_at(home) == key:
            return Registration("copy", key, self.number_of(key), home)
        else:
            self._point(link, log)
            status, previous = "moved", home
        number = self._wait_for_number(key)
        if number is None:
            number = self._claim(key)  # its creator stopped before it claimed one
        self._follow(key, number)
        return Registration(status, key, number, real_private, previous)

    def entries(self) -> List[Entry]:
        """Each short number, lowest first, with what it names."""
        stems = (_stem(name, PUBLIC_SUFFIX) for name in self._names())
        numbers = sorted(int(stem) for stem in stems if stem is not None and _NUMBER.match(stem))
        rows: List[Entry] = []
        first_of: Dict[str, int] = {}
        for number in numbers:
            target = _link(self.path / f"{number}{PUBLIC_SUFFIX}")
            if target is None:
                continue  # not a link, so not a number of the registry's
            key = _stem(target, PUBLIC_SUFFIX) or target
            home = self._home(key) if _KEY.match(key) else None
            state = "missing" if home is None else _state(home, key)
            first = first_of.setdefault(key, number)
            rows.append(Entry(number, key, home, state, None if first == number else first))
        return rows

    def find(self, handle: str) -> Root:
        """The root context that ``handle`` names, which must be where it's registered.

        A handle is a short number, ``14``, or a key, ``context_`` and at least
        4 of its hex digits, either followed by ``#1``, the root itself.
        Raises ``RegistryError``.
        """
        match = _HANDLE.match(handle.strip())
        if match is None:
            raise RegistryError(
                f"{handle!r} names no root context: give a short number, such as 14, or a "
                f"key, {KEY_STEM} and at least {ABBREVIATED_DIGITS} of its hex digits, "
                "either one optionally followed by #1")
        inner = match.group("context")
        if inner is not None and inner != "1":
            raise RegistryError(f"{handle.strip()} names context #{inner}, but so far a root "
                                "context has only #1, itself")
        if match.group("number") is not None:
            number: Optional[int] = int(match.group("number"))
            target = _link(self.path / f"{number}{PUBLIC_SUFFIX}")
            if target is None:
                raise RegistryError(f"there's no root context {number} in {self.path}/")
            key = _stem(target, PUBLIC_SUFFIX) or target
            if not _KEY.match(key):
                raise RegistryError(f"{self.path}/{number}{PUBLIC_SUFFIX} doesn't name a key")
        else:
            stem = KEY_STEM + match.group("digits")
            found = [key for key in self.keys() if key.startswith(stem)]
            if not found:
                raise RegistryError(f"no key in {self.path}/ starts with {stem}")
            if len(found) > 1:
                raise RegistryError(f"{len(found)} keys in {self.path}/ start with {stem}: "
                                    "give more of its digits")
            key = found[0]
            number = self.number_of(key)
        named = (f"root context {number} ({self.abbreviation(key)})" if number is not None
                 else f"root context {self.abbreviation(key)}")
        home = self._home(key)
        if home is None:
            raise RegistryError(f"{named} has no link to a save log in {self.path}/")
        prefix = save_prefix_of(home)
        state = _state(home, key)
        if state == "missing":
            raise RegistryError(f"{named} isn't at {prefix} any more: open its save with --save "
                                "where it is now, which registers it there")
        if state == "replaced":
            other = key_at(home)
            raise RegistryError(f"{named}'s path, {prefix}, now holds another save, "
                                f"{abbreviate(other, self.keys())}")
        if state == "unreadable":
            raise RegistryError(f"{named} has no readable full log at {prefix}")
        return Root(number, key, home)


def render_roots(path: Path, entries: List[Entry], *, home: Optional[str] = None) -> str:
    """``--list-roots``: one line per short number, after a heading.

    Each line has the number, the key abbreviated, its state, the full log's
    last change in local time (when it's ``ok``), and the save prefix, with
    ``home`` shown as ``~``.
    """
    def shown(value) -> str:
        text = str(value)
        if home and (text == home or text.startswith(home.rstrip("/") + "/")):
            return "~" + text[len(home.rstrip("/")):]
        return text

    if not entries:
        return f"Root contexts in {shown(path)}/: none yet."
    numbers = [entry.number for entry in entries]
    lines = [f"Root contexts in {shown(path)}/: {len(entries)}, from {min(numbers)} to "
             f"{max(numbers)}."]
    keys = [entry.key for entry in entries]
    for entry in entries:
        key = abbreviate(entry.key, keys) if _KEY.match(entry.key) else entry.key
        state = entry.state if entry.first is None else f"={entry.first}"
        when = ""
        if entry.state == "ok" and entry.first is None:
            with suppress(OSError):
                changed = os.stat(Path(entry.home) / CONTEXT_NAME).st_mtime
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(changed))
        where = "-" if entry.home is None else shown(save_prefix_of(Path(entry.home)))
        lines.append(f"{entry.number:>4}  {key:<12}  {state:<10}{when:<16}  {where}")
    return "\n".join(lines)
