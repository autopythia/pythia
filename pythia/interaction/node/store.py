"""The node server's store: what the glui view keeps across the server's runs, as a
browser keeps its profile, in a directory of its own, ``~/.pythia/node/`` unless
``--store`` says otherwise, that only you can read (0700, each file 0600).

Each of its files is a JSON object that a page of the view reads, and replaces whole by
POST, from the server's own pages alone (``server.App``): ``session.json``, the view's
settings, such as its look, its theme and whether a log's runs fold, each a name and a
short word. A write goes to a temporary file, which then takes the old one's place, so
that a crash leaves the old file or the new; two pages that write at once each keep what
they wrote last.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Callable, Dict

# The most bytes that a file of the store holds, and that a page may send for one.
MAX_BYTES = 16 * 1024
MAX_SETTINGS = 32
_SETTING = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")
_WORD = re.compile(r"[A-Za-z0-9_-]{0,64}\Z")


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


# The store's files, each by its name, with what checks a value for it.
FILES: Dict[str, Callable[[object], dict]] = {"session": settings}


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
                data = file.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                return {}
            return FILES[name](json.loads(data.decode("utf-8")))
        except (OSError, ValueError):  # Invalid and UnicodeDecodeError among them
            return {}

    def write(self, name: str, value: object) -> None:
        """Replaces file ``name``, one of ``FILES``, with ``value``, once it's checked. Raises
        Invalid, or OSError if it can't be written."""
        data = json.dumps(FILES[name](value), sort_keys=True).encode("utf-8")
        with self._lock:
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
