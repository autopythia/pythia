"""The node server's store: what the glui view keeps across the server's runs, as a
browser keeps its profile, in a directory of its own, ``~/.pythia/node/`` unless
``--store`` says otherwise, that only you can read (0700, each file 0600).

Each of its files is a JSON object that a page of the view reads, and changes by POST,
from the server's own pages alone (``server.App``): ``session.json``, the view's settings,
such as its look, its theme and whether a log's runs fold, each a name and a short word,
which a page replaces whole; ``stars.json``, the lines starred in each root's log, by the
root's key, so that they follow a save that moves, which a page sets or clears one at a
time; and ``roots.json``, each root's label and tags, by its key, which a page sets for
one root at a time. A write goes to a temporary file, which then takes the old one's
place, so that a crash leaves the old file or the new; two pages that write at once each
keep what they wrote last. A change reads what it changes and writes it under the store's
lock, ``.lock`` in its directory, which every process that keeps the store takes, as two
node servers may, so that neither loses what the other changed; where a file system
can't lock, a change goes on without it.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple

# The most bytes that a page may send to the store, and that a file of it holds.
MAX_BYTES = 16 * 1024
MAX_FILE_BYTES = 1024 * 1024
MAX_SETTINGS = 32
# The most roots with stars, and stars in a root's log.
MAX_ROOTS = 1000
MAX_STARS = 10_000
# A root's label, in characters, and its tags, at most; and what a tag is.
MAX_LABEL = 120
MAX_TAGS = 12
_TAG = re.compile(r"[a-z0-9][a-z0-9._-]{0,31}\Z")
_SETTING = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")
_WORD = re.compile(r"[A-Za-z0-9_-]{0,64}\Z")
_ROOT = re.compile(r"context_[0-9a-f]{32}\Z")


class Invalid(ValueError):
    """A value that the store doesn't keep; the message says why."""


def settings(value: object) -> Dict[str, str]:
    """``value`` as the view's settings: a JSON object of at most ``MAX_SETTINGS`` names,
    each with a short word. Raises Invalid."""
    if not isinstance(value, dict):
        raise Invalid("the settings aren't an object")
    if len(value) > MAX_SETTINGS:
        raise Invalid(f"there are more than {MAX_SETTINGS} settings")
    for name, word in value.items():
        if not _SETTING.match(name) or not isinstance(word, str) or not _WORD.match(word):
            raise Invalid(f"the setting {name[:40]!r} isn't a name with a short word")
    return dict(value)


def stars(value: object) -> Dict[str, List[int]]:
    """``value`` as stars: a JSON object of at most ``MAX_ROOTS`` roots, each by its key,
    with the lines starred in its log, at most ``MAX_STARS``, each from 1; in order, once
    each. Raises Invalid."""
    if not isinstance(value, dict):
        raise Invalid("the stars aren't an object")
    if len(value) > MAX_ROOTS:
        raise Invalid(f"there are stars in more than {MAX_ROOTS} roots")
    kept = {}
    for key, lines in value.items():
        if not _ROOT.match(key):
            raise Invalid(f"{key[:40]!r} isn't a root's key")
        if (not isinstance(lines, list) or len(lines) > MAX_STARS
                or not all(type(line) is int and 1 <= line < 10 ** 12 for line in lines)):
            raise Invalid(f"{key}'s stars aren't a list of at most {MAX_STARS:,} lines")
        if lines:
            kept[key] = sorted(set(lines))
    return kept


def root(value: object) -> Tuple[str, List[str]]:
    """``value`` as a root's label and tags, as a page sends them: ``{"label": TEXT,
    "tags": [TAG, ...]}``, a label of ``MAX_LABEL`` characters at most, none of them a
    control character, and at most ``MAX_TAGS`` tags, each a word of lowercase letters,
    digits, '.', '_' and '-', 32 characters at most; the label without spaces around it,
    and the tags in order, once each. Raises Invalid."""
    if not isinstance(value, dict) or set(value) != {"label", "tags"}:
        raise Invalid('a root is {"label": TEXT, "tags": [TAG, ...]}')
    label, tags = value["label"], value["tags"]
    if (not isinstance(label, str) or len(label) > MAX_LABEL
            or any(ord(c) < 0x20 or 0x7f <= ord(c) < 0xa0 for c in label)):
        raise Invalid(f"a label is text of {MAX_LABEL} characters at most")
    if (not isinstance(tags, list) or len(tags) > MAX_TAGS
            or not all(isinstance(tag, str) and _TAG.match(tag) for tag in tags)):
        raise Invalid(f"a root has {MAX_TAGS} tags at most, each a word of lowercase letters, "
                      "digits, '.', '_' and '-', 32 characters at most")
    return label.strip(), sorted(set(tags))


def roots(value: object) -> Dict[str, dict]:
    """``value`` as what the view says of roots: a JSON object of at most ``MAX_ROOTS`` roots,
    each by its key, with its label and tags (``root``); a root with neither is left out.
    Raises Invalid."""
    if not isinstance(value, dict):
        raise Invalid("the roots aren't an object")
    if len(value) > MAX_ROOTS:
        raise Invalid(f"there are labels or tags for more than {MAX_ROOTS} roots")
    kept = {}
    for key, said in value.items():
        if not _ROOT.match(key):
            raise Invalid(f"{key[:40]!r} isn't a root's key")
        label, tags = root(said)
        if label or tags:
            kept[key] = {"label": label, "tags": tags}
    return kept


# The store's files, each by its name, with what checks a value for it; and those that a
# page replaces whole.
FILES: Dict[str, Callable[[object], dict]] = {"session": settings, "stars": stars,
                                              "roots": roots}
REPLACED = ("session",)


class Store:
    """The files under ``path``, which it makes when it first writes one."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def read(self, name: str) -> dict:
        """File ``name``'s object, one of ``FILES``; empty if there's none yet, or it isn't one
        that the store would write."""
        try:
            with open(self.path / f"{name}.json", "rb") as file:
                data = file.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                return {}
            return FILES[name](json.loads(data.decode("utf-8")))
        except (OSError, ValueError):  # Invalid and UnicodeDecodeError among them
            return {}

    def write(self, name: str, value: object) -> None:
        """Replaces file ``name``, one of ``FILES``, with ``value``, once it's checked. Raises
        Invalid, or OSError if it can't be written."""
        with self._locked():
            self._replace(name, FILES[name](value))

    def star(self, key: str, line: int, on: bool) -> None:
        """Stars line ``line`` of the log of the root whose key is ``key``, or clears its star:
        the rest of the stars stay as they are. Raises Invalid, or OSError."""
        with self._locked():
            kept = self.read("stars")
            lines = set(kept.get(key, ()))
            (lines.add if on else lines.discard)(line)
            kept[key] = sorted(lines)
            self._replace("stars", stars(kept))

    def root(self, key: str, value: object) -> None:
        """Gives the root whose key is ``key`` the label and tags that ``value`` says, as
        ``root`` checks them, or neither, if they're empty: the rest stay as they are.
        Raises Invalid, or OSError."""
        label, tags = root(value)
        with self._locked():
            kept = self.read("roots")
            if label or tags:
                kept[key] = {"label": label, "tags": tags}
            else:
                kept.pop(key, None)
            self._replace("roots", roots(kept))

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """The store's lock: this one's, among its threads, then ``.lock``'s, among the
        processes that keep the store, if its file system can lock it."""
        with self._lock:
            self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
            handle = os.open(self.path / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                with contextlib.suppress(OSError):  # as on a file system that can't
                    fcntl.flock(handle, fcntl.LOCK_EX)
                yield
            finally:
                os.close(handle)  # which lets the lock go

    def _replace(self, name: str, value: dict) -> None:
        """Replaces file ``name`` with ``value``, checked, through a temporary file; with the
        store's lock held."""
        data = json.dumps(value, sort_keys=True).encode("utf-8")
        if len(data) > MAX_FILE_BYTES:
            raise Invalid(f"the store holds {MAX_FILE_BYTES:,} bytes at most for a file")
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=".json",
                                             dir=self.path)  # 0600
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path / f"{name}.json")
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
