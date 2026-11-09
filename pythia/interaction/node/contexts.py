"""The node's root contexts, from its registry, and their logs' line indexes.

The node server shows root contexts only: those that the node's registry
lists (``pythia/interaction/registry.py``), ``~/.pythia/root/`` by default,
each by its short number. A root is a CLI save, registered where it is: its
full log, ``PREFIX._private/context.jsonl``, is the context, and its save log
and summary go with it. A second number that names a root is left out, and
so is a root whose save isn't where it's registered; the registry's state
says why (``Roots.entries``).

A log is read through its line index (``Index``) only: each complete line's
byte offset and kind, from one pass, after which only what's appended is
read. Nothing here writes, in a save or in the registry.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import stat
import threading
from array import array
from bisect import bisect_right
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from .._cli_save import CONTEXT_NAME, PUBLIC_SUFFIX
from .._save_lock import LOCK_NAME
from ..compaction import _SUMMARY_SUFFIX as SUMMARY_SUFFIX
from ..registry import Entry, Registry, RegistryError, save_prefix_of
from ..registry import key_of as _registry_key_of

# A log's first line is its init, which is short: no more of it is read.
FIRST_LINE_BYTES = 65_536
# The type of a line that isn't a JSON object with a type.
BAD = "<bad>"
# A turn's input, as a context's page lists it: at most this many of the
# user's lines that start it.
INPUT_LINES = 8

Kind = Tuple[str, Optional[str]]  # a line's type, and a message's role
Call = Tuple[str, str, bool]  # a call's ID, its tool's name, and whether the user made it

_INIT_TYPES = frozenset(("init", "session_init"))

# pythia's encoder writes a record's type first, then a message's role, or a
# tool call's name and call ID, so these give most lines' kinds without
# parsing them. Any other line is parsed whole.
_HEAD = re.compile(rb'\{"type": "([a-z_]+)"(?:, "role": "([a-z_]+)")?')
_STRING = rb'"((?:[^"\\]|\\.)*)"'
_TOOL_CALL = re.compile(rb'\{"type": "tool_call", "name": ' + _STRING
                        + rb', "call_id": ' + _STRING)
_CHUNK_BYTES = 1 << 24
# The last bytes indexed, kept to see a log that was rewritten in place.
_TAIL_BYTES = 64
# A log with more kinds than this, which pythia never writes, shares one.
_MAX_KINDS = 60_000
_OTHER: Kind = ("<other>", None)

# For turns: the user's lines, and the lines that aren't the model's work.
_USER = frozenset((("message", "user"), ("user_tool_call", None), ("user_tool_result", None)))
# Older logs' names count as the current ones: session_init for init,
# context_compaction for context_prefix, and turn_metadata for sample_metadata.
_QUIET = frozenset(("init", "session_init", "tools", "instructions", "context_prefix",
                    "context_compaction", "compaction_metadata", "opaque_compaction",
                    "model_sample_boundary", "user_interaction_boundary"))
_SAMPLES = frozenset(("sample_metadata", "turn_metadata"))
_BOUNDARY = "user_interaction_boundary"
_TURN_SUMMARY = "turn_summary"


def _names(directory: Optional[Path]) -> List[str]:
    if directory is None:
        return []
    try:
        return sorted(os.listdir(directory))
    except OSError:
        return []


def _is_file(path: Path) -> bool:
    """Whether ``path`` is a regular file, through links; False when that can't be told."""
    try:
        return stat.S_ISREG(os.stat(path).st_mode)
    except (OSError, ValueError):
        return False


def shown_path(path: object) -> str:
    """A path, with the home directory as ``~``, as ``--list-roots`` shows one."""
    text, home = str(path), os.path.expanduser("~").rstrip("/")
    if home and (text == home or text.startswith(home + "/")):
        return "~" + text[len(home):]
    return text


_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)


def _open(path: Path):
    """``path``, opened to read bytes: without blocking, should it be a FIFO."""
    return open(os.open(path, _OPEN_FLAGS), "rb")


def key_of(prefix_id: object) -> Optional[str]:
    """The key that a prefix ID names, as the registry reads it, or None."""
    try:
        return _registry_key_of(prefix_id)
    except RegistryError:
        return None


def parse(raw: bytes) -> dict:
    """A line's record, or a stand-in of type ``<bad>`` that holds it raw."""
    try:
        record = json.loads(raw)
    except (ValueError, RecursionError):
        record = None
    if isinstance(record, dict):
        return record
    return {"type": BAD, "raw": raw.decode("utf-8", "replace")}


def message_content(record: dict) -> object:
    """A message's content: ``content``, or an older log's ``text``."""
    return record["content"] if "content" in record else record.get("text")


def content_text(content: object) -> str:
    """A message's text: its content, or the text of its text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part["text"] for part in content
                         if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


def _first_record(path: Path) -> Optional[dict]:
    try:
        with _open(path) as log:
            record = json.loads(log.readline(FIRST_LINE_BYTES))
    except (OSError, ValueError, RecursionError):
        return None
    return record if isinstance(record, dict) else None


def _line_end(data: bytes, start: int) -> int:
    """Where the line at ``start`` ends: its newline, or -1 without one.

    Only ``\\n`` ends a line. ``str.splitlines`` also ends one at U+2028 and
    others, which older logs hold raw inside their strings.
    """
    return data.find(b"\n", start)


def _decoded(text: bytes) -> str:
    return json.loads(b'"' + text + b'"')


def record_kind(record: object) -> Tuple[Kind, Optional[Call]]:
    """A parsed record's kind, and the call it makes, if any."""
    if not isinstance(record, dict) or not isinstance(record.get("type"), str):
        return (BAD, None), None
    item_type = record["type"]
    role = record.get("role")
    kind = (item_type, role if item_type == "message" and isinstance(role, str) else None)
    call = (record.get("call") if item_type == "user_tool_call"
            else record if item_type == "tool_call" else None)
    if (isinstance(call, dict) and isinstance(call.get("name"), str)
            and isinstance(call.get("call_id"), str)):
        return kind, (call["call_id"], call["name"], item_type == "user_tool_call")
    return kind, None


def line_kind(data: bytes, start: int = 0, end: Optional[int] = None
              ) -> Tuple[Kind, Optional[Call]]:
    """The kind of the line ``data[start:end]``, and the call it makes, if any."""
    end = len(data) if end is None else end
    head = _HEAD.match(data, start, end)
    if head is not None:
        item_type, role = head.group(1), head.group(2)
        if item_type == b"tool_call":
            call = _TOOL_CALL.match(data, start, end)
            if call is not None:
                try:
                    return ("tool_call", None), (_decoded(call.group(2)),
                                                 _decoded(call.group(1)), False)
                except ValueError:
                    pass
        elif item_type == b"message":
            if role is not None:
                return ("message", role.decode("ascii")), None
        elif item_type != b"user_tool_call":
            return (item_type.decode("ascii"), None), None
    try:
        record = json.loads(data[start:end])
    except (ValueError, RecursionError):
        return (BAD, None), None
    return record_kind(record)


# Each index's generations, from one count, so that no two are alike (Index).
_GENERATIONS = itertools.count(1)


class Index:
    """A log's line index: each complete line's byte offset and kind.

    ``refresh`` reads only what was appended since the last one. A log that
    shrank, was replaced (it has another inode), or whose last indexed bytes
    changed is indexed anew, and ``generation`` counts up, across every index
    of the process, so that it names what one indexed. A last line without
    its newline is still being written, so it waits. Each method holds the
    index's lock.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock = threading.RLock()
        self.generation = 0
        self.size = 0
        self.changed = 0.0
        self._reset(None)

    def _reset(self, stamp: Optional[Tuple[int, int]]) -> None:
        self.generation = next(_GENERATIONS)
        self.stamp = stamp
        self.end = 0  # the bytes indexed: complete lines only
        self.mtime_ns: Optional[int] = None
        self.tail = b""
        self.offsets = array("q")
        self.codes = array("H")
        self.kinds: List[Kind] = []  # by code
        self.code_of: Dict[Kind, int] = {}
        self.calls: Dict[str, str] = {}  # a model tool call's ID: its tool's name
        self.user_calls: Dict[str, str] = {}  # a user tool call's

    def __len__(self) -> int:
        return len(self.offsets)

    def refresh(self) -> "Index":
        """Index what was appended; raises OSError when the log can't be read."""
        with self.lock:
            status = os.stat(self.path)
            if ((status.st_dev, status.st_ino) == self.stamp and status.st_size == self.end
                    and status.st_mtime_ns == self.mtime_ns):
                return self  # unchanged: one stat
            # What's opened is what's indexed, should the log be replaced meanwhile.
            with _open(self.path) as log:
                status = os.fstat(log.fileno())
                if not stat.S_ISREG(status.st_mode):
                    raise OSError(f"{self.path} isn't a regular file")
                stamp = (status.st_dev, status.st_ino)
                if stamp != self.stamp or status.st_size < self.end:
                    self._reset(stamp)
                elif self.tail:
                    log.seek(self.end - len(self.tail))
                    if log.read(len(self.tail)) != self.tail:
                        self._reset(stamp)
                log.seek(self.end)
                self._index(log)
            self.size, self.changed, self.mtime_ns = (status.st_size, status.st_mtime,
                                                      status.st_mtime_ns)
            return self

    def _index(self, log) -> None:
        offset = self.end  # where data starts in the log
        pending = b""  # a line without its newline yet
        while True:
            chunk = log.read(_CHUNK_BYTES)
            if not chunk:
                break
            data = pending + chunk if pending else chunk
            start = 0
            while True:
                end = _line_end(data, start)
                if end < 0:
                    break
                self._add(data, start, end, offset + start)
                start = end + 1
            offset += start
            pending = data[start:]
        # A last line without its newline is still being written: it waits.
        if offset != self.end:
            self.end = offset
            log.seek(max(0, offset - _TAIL_BYTES))
            self.tail = log.read(offset - max(0, offset - _TAIL_BYTES))

    def _add(self, data: bytes, start: int, end: int, offset: int) -> None:
        kind, call = line_kind(data, start, end)
        code = self.code_of.get(kind)
        if code is None:
            if len(self.kinds) >= _MAX_KINDS:
                kind = _OTHER
                code = self.code_of.get(kind)
            if code is None:
                code = len(self.kinds)
                self.kinds.append(kind)
                self.code_of[kind] = code
        self.offsets.append(offset)
        self.codes.append(code)
        if call is not None:
            call_id, name, by_user = call
            (self.user_calls if by_user else self.calls)[call_id] = name

    def kind(self, line: int) -> Kind:
        return self.kinds[self.codes[line]]

    def read(self, start: int, stop: int) -> List[bytes]:
        """Lines ``start`` to ``stop - 1``, from 0, without their newlines."""
        with self.lock:
            stop = min(stop, len(self.offsets))
            if start < 0 or start >= stop:
                return []
            first = self.offsets[start]
            last = self.offsets[stop] if stop < len(self.offsets) else self.end
            with _open(self.path) as log:
                status = os.fstat(log.fileno())
                if (status.st_dev, status.st_ino) != self.stamp:
                    raise OSError(f"{self.path} was replaced while it was read")
                log.seek(first)
                data = log.read(last - first)
        return data.split(b"\n")[:stop - start]


def turn_starts(codes: Sequence[int], kinds: Sequence[Kind]) -> List[int]:
    """Each turn's first line, from 0, from the lines' kinds alone.

    A turn starts at the user's lines just before a user_interaction_boundary,
    or at the user's first line after a turn_summary, past lines that aren't
    the model's work. A start with no model work since the last one joins
    that turn. So /config and /compact between turns open the next, /retry
    and /steer stay in theirs, and a /compact that the model works on after
    is a turn of its own.
    """
    user = [kind in _USER for kind in kinds]
    quiet = [kind in _USER or kind[0] in _QUIET for kind in kinds]
    boundary = [kind[0] == _BOUNDARY for kind in kinds]
    summary = [kind[0] == _TURN_SUMMARY for kind in kinds]
    count = len(codes)
    starts = set()
    for line in range(count):
        code = codes[line]
        if boundary[code]:
            start = line
            while start > 0 and user[codes[start - 1]]:
                start -= 1
            starts.add(start)
        elif summary[code]:
            start = line + 1
            while (start < count and quiet[codes[start]] and not user[codes[start]]
                   and not boundary[codes[start]]):
                start += 1
            if start < count and user[codes[start]]:
                starts.add(start)
    kept: List[int] = []
    checked = 0  # since the last kept start, the lines before this are quiet
    for start in sorted(starts):
        if kept:
            line = max(checked, kept[-1])
            while line < start and quiet[codes[line]]:
                line += 1
            if line >= start:
                checked = start
                continue
        kept.append(start)
        checked = start
    return kept


class TurnRow(NamedTuple):
    """A turn, as a context's page lists it."""

    number: int
    start: int  # its first line, from 0
    stop: int  # past its last line
    label: str  # the user's input: a message's first line, or /NAME for a user tool
    samples: int
    calls: int
    failures: int
    summary: Optional[dict]  # its turn_summary, when it has one


def input_label(records: Iterable[dict]) -> str:
    """The user's lines that start a turn, as one line: ``/config · Fix it.``."""
    parts = []
    for record in records:
        if record.get("type") == "message":
            content = message_content(record)
            text = content_text(content).strip()
            if text:
                parts.append(text.split("\n", 1)[0])
            else:
                parts.append("[image]" if isinstance(content, list) and content else "(empty)")
        elif record.get("type") == "user_tool_call":
            call = record.get("call")
            name = call.get("name") if isinstance(call, dict) else None
            parts.append(f"/{name if isinstance(name, str) else '?'}")
    return " · ".join(parts)


class Context:
    """A root context: the save that root ``number`` names, with its full log's index.

    Its ID is the number, as text; its name, its save prefix's last part.
    """

    def __init__(self, number: int, key: str, private: Path) -> None:
        self.number = number
        self.id = str(number)
        self.key = key  # the registry's, which the full log's init names
        self.private = Path(private)  # the save's private directory, as registered
        prefix = save_prefix_of(self.private)
        self.prefix = prefix
        self.name = prefix.name
        self.log = self.private / CONTEXT_NAME
        self.public = prefix.with_name(prefix.name + PUBLIC_SUFFIX)  # the save log
        self.summary = prefix.with_name(prefix.name + SUMMARY_SUFFIX)  # which may not exist
        self.prefix_id: Optional[str] = None
        self.model: Optional[str] = None
        self.index = Index(self.log)
        self._turns: Tuple = (None, None, [])
        self._title: Tuple = (None, 0, None)
        self._rows: Dict[Tuple[int, int, int], TurnRow] = {}

    def __repr__(self) -> str:
        return f"Context({self.number}, {self.name!r})"

    @property
    def label(self) -> str:
        """How pages name it: its save's name and its number, ``ui-0 (14)``."""
        return f"{self.name} ({self.number})"

    def read_head(self, record: Optional[dict] = None) -> None:
        """The prefix ID and the model, from the log's first line, its init."""
        record = _first_record(self.log) if record is None else record
        if record is None or record.get("type") not in _INIT_TYPES:
            record = {}
        prefix_id = record.get("prefix_id", record.get("session_id"))
        self.prefix_id = prefix_id if isinstance(prefix_id, str) else None
        model = record.get("model")
        self.model = model if isinstance(model, str) else None

    def names(self, entry: Entry) -> bool:
        """Whether a registry's entry names this very context: its key, at its path."""
        return (entry.key, entry.home) == (self.key, self.private)

    def refresh(self) -> "Context":
        self.index.refresh()
        return self

    def __len__(self) -> int:
        return len(self.index)

    def records(self, start: int, stop: int) -> List[Tuple[int, dict]]:
        """Lines ``start`` to ``stop - 1``, from 0, as (line number, from 1; record)."""
        return [(start + offset + 1, parse(raw))
                for offset, raw in enumerate(self.index.read(start, stop))]

    def turns(self) -> List[int]:
        """Each turn's first line, from 0."""
        index = self.index
        with index.lock:
            if self._turns[:2] != (index.generation, len(index)):
                self._turns = (index.generation, len(index),
                               turn_starts(index.codes, index.kinds))
            return self._turns[2]

    def turn_of(self, line: int) -> int:
        """The turn that line ``line`` (from 0) is in, from 1; 0 for the start."""
        return bisect_right(self.turns(), line)

    def turn_range(self, number: int) -> Tuple[int, int]:
        """Turn ``number``'s lines, from 0, as a range; turn 0 is the start."""
        starts = self.turns()
        count = len(self.index)
        if number == 0:
            return 0, starts[0] if starts else count
        return starts[number - 1], starts[number] if number < len(starts) else count

    def title(self) -> str:
        """The first line of the first user message, or ""."""
        index = self.index
        with index.lock:
            generation, searched, found = self._title
            if generation != index.generation:
                searched, found = 0, None
            code = index.code_of.get(("message", "user"))
            if found is None and code is not None:
                for line in range(searched, len(index)):
                    if index.codes[line] == code:
                        record = self.records(line, line + 1)[0][1]
                        found = content_text(message_content(record)).strip().split("\n", 1)[0]
                        break
            self._title = (index.generation, len(index), found)
            return found or ""

    def turn_rows(self) -> List[TurnRow]:
        """Each turn, with its input and counts, for a context's page."""
        index = self.index
        with index.lock:
            starts = self.turns()
            flags = [(kind == ("message", "user") or kind[0] == "user_tool_call",
                      kind in _USER or kind[0] in _QUIET, kind[0] in _SAMPLES,
                      kind[0] == "tool_call", kind[0] == "model_failure",
                      kind[0] == _TURN_SUMMARY)
                     for kind in index.kinds]
            rows, cached = [], {}
            for number, (start, stop) in enumerate(zip(starts, starts[1:] + [len(index)]), 1):
                key = (index.generation, start, stop)
                row = self._rows.get(key) or self._turn_row(start, stop, flags)
                cached[key] = row
                rows.append(row._replace(number=number))
            self._rows = cached
            return rows

    def _turn_row(self, start: int, stop: int, flags: List[Tuple[bool, ...]]) -> TurnRow:
        codes = self.index.codes
        samples = calls = failures = 0
        summary_line = None
        for line in range(start, stop):
            _shown, _quiet, sample, call, failure, summary = flags[codes[line]]
            samples += sample
            calls += call
            failures += failure
            if summary:
                summary_line = line
        # The user's input: their messages and tools before the model's first work, so
        # that /config or /compact before a message shows the message too.
        shown, line = [], start
        while (line < stop and len(shown) < INPUT_LINES and line - start < 8 * INPUT_LINES
               and flags[codes[line]][1]):
            if flags[codes[line]][0]:
                shown.append(line)
            line += 1
        label = input_label(self.records(each, each + 1)[0][1] for each in shown)
        summary_record = (None if summary_line is None
                          else self.records(summary_line, summary_line + 1)[0][1])
        return TurnRow(0, start, stop, label, samples, calls, failures, summary_record)

    def files(self) -> List[Tuple[str, Path]]:
        """The files that go with the context, that exist: (what each is, where)."""
        found = [("full log", self.log), ("save log", self.public), ("summary", self.summary)]
        found += [(_what(name), self.private / name) for name in _names(self.private)
                  if name != CONTEXT_NAME]
        return [(what, path) for what, path in found if _is_file(path)]


def _what(name: str) -> str:
    if name == LOCK_NAME:
        return "lock"
    return "trace" if "trace." in name else "file"


class Roots:
    """The node's root contexts: those that its registry lists, by short number.

    Each listing, and each root looked up, reads the registry again, as
    ``--list-roots`` does, so that a root shows only while it's where it's
    registered; the indexes of the roots already known are kept.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.registry = Registry(self.path)
        self._lock = threading.Lock()
        self._contexts: Dict[str, Context] = {}

    def problem(self) -> Optional[str]:
        """Why the registry can't be listed, or None."""
        try:
            os.listdir(self.path)
        except FileNotFoundError:
            return f"There's no registry at {shown_path(self.path)}/ yet."
        except OSError as exc:
            return f"The registry {shown_path(self.path)}/ can't be read: {exc.strerror or exc}."
        return None

    def entries(self) -> List[Entry]:
        """Each root, by its lowest number, with its state: ``ok``, ``missing``,
        ``replaced`` or ``unreadable`` (``registry.Entry``)."""
        try:
            entries = self.registry.entries()
        except OSError:
            return []
        return [entry for entry in entries if entry.first is None]

    def contexts(self) -> List[Context]:
        """The roots whose saves are where they're registered, lowest number first."""
        entries = self.entries()
        with self._lock:
            current: Dict[str, Context] = {}
            for entry in entries:
                if entry.state != "ok" or entry.home is None:
                    continue
                context = self._contexts.get(str(entry.number))
                if context is None or not context.names(entry):
                    context = Context(entry.number, entry.key, entry.home)
                    context.read_head()
                current[context.id] = context
            self._contexts = current
        return list(current.values())

    def get(self, cid: str) -> Optional[Context]:
        """Root ``cid``, a number as text, if it's where it's registered; never joined to a
        path."""
        return next((context for context in self.contexts() if context.id == cid), None)

    def entry(self, cid: str) -> Optional[Entry]:
        """The registry's entry for number ``cid``, as text: a root's, or another number
        that names one (its ``first``)."""
        try:
            entries = self.registry.entries()
        except OSError:
            return None
        return next((entry for entry in entries if str(entry.number) == cid), None)
