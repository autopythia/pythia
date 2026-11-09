"""The node server's pages: its root contexts, a context and its turns, runs
of lines, and items.

Every string from a log or the directory goes through ``h``: first, each
character that pythia's terminal shows only as an escape (``safe_text``)
becomes one, such as ``\\u202e``, styled apart; then ``html.escape`` makes the
rest inert. A page has one inline stylesheet, its style's (``styles.py``),
``<details>`` for folds, and no script. Items show as the terminal shows
them, with its labels (``pythia/interaction/display.py``). Each item's
element has the classes ``it``, its kind (``user``, ``assistant``, ``tool``,
``err`` or ``meta``) and its type (``t-tool_call``), for stylesheets; the
item types that a style folds show as ``<details>``, one line each.

A page's style is the URL's ``?style=NAME``, else the server's (``--style``).
Every link on it keeps its style, and the menu at its top right links to the
same page in each style.
"""

from __future__ import annotations

import copy
import html
import json
import os
import re
import time
from pathlib import Path, PurePosixPath
from typing import (Callable, Dict, FrozenSet, Iterable, List, NamedTuple, Optional, Sequence,
                    Tuple)
from urllib.parse import quote, urlencode

from .._cli_editor import safe_text
from .._provider_text import escaped_character, escaped_characters
from ..compaction import COLDSTART_POINTER_PREFIX as POINTER_PREFIX
from ..compaction import COMPACTION_SUMMARY_PREFIX as SUMMARY_PREFIX
from ..compaction import COMPACTION_SUMMARY_SUFFIX as SUMMARY_SUFFIX
from ..compaction import _LEGACY_COLDSTART_POINTER_PREFIX as LEGACY_POINTER_PREFIX
from ..compaction import _LEGACY_SUMMARY_PREFIX as LEGACY_SUMMARY_PREFIX
from ..display import _describe_non_text_source, _runs_text
from ..display import _format_byte_count as byte_count
from ..registry import KEY_STEM, Entry, abbreviate, save_prefix_of
from .contexts import (BAD, Context, Roots, TurnRow, key_of, message_content, parse,
                       shown_path)
from .styles import DEFAULT_STYLE, FIELD, Style, Styles

# A block longer than this many characters is cut, with a link to its line.
CUT = 4000
# A page shows at most this many lines: of a turn, or of a run of lines.
PAGE_LINES = 2000
# A context's page shows at most this many lines of its start.
START_LINES = 100
# A title, or a fold's first line, is cut to this many characters.
TITLE_CHARS = 120
# A folded item's line shows this much of what it holds, at most.
GIST_CHARS = 160
# At most this much of a summary file is shown.
SUMMARY_BYTES = 4 << 20

# A key, as the registry writes them.
_KEY = re.compile(r"context_[0-9a-f]{32}\Z")
# Why a root isn't shown, by its state in the registry (registry.Entry).
ABSENT = {
    "missing": ("Nothing is at its path any more: its save was moved or removed. The CLI "
                "registers a save where it is when it opens it."),
    "replaced": "Its path now holds another save.",
    "unreadable": "Its full log can't be read.",
}


def _code(char: str) -> str:
    code = ord(char)
    return f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}"


def h(value: object) -> str:
    """``value`` as HTML: its characters to escape as code points, styled apart; the rest inert."""
    text = str(value)
    parts = []
    last = 0
    for match in escaped_characters().finditer(text):
        char = match.group()
        if not escaped_character(char):
            continue
        parts.append(html.escape(text[last:match.start()]))
        parts.append(f"<span class=esc>{_code(char)}</span>")
        last = match.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts)


def linked(text: str, spans: Sequence[Tuple[int, int, str]] = ()) -> str:
    """``text`` as HTML, with each span (start, end, URL) a link."""
    parts = []
    last = 0
    for start, end, url in sorted(spans):
        if start < last:
            continue
        parts.append(h(text[last:start]))
        parts.append(f'<a href="{h(url)}">{h(text[start:end])}</a>')
        last = end
    parts.append(h(text[last:]))
    return "".join(parts)


def cut(text: str, size: int) -> str:
    return text if len(text) <= size else text[:size - 1] + "…"


def absent_note(absent: int) -> str:
    """Where a list of roots puts those that aren't shown: `` The 1 that isn't where it's
    registered comes last.``, or nothing."""
    if not absent:
        return ""
    if absent == 1:
        return " The 1 that isn't where it's registered comes last."
    return f" The {absent} that aren't where they're registered come last."


def when(seconds: float) -> str:
    """A time as ``--list-roots`` shows one, in local time: ``2026-11-04 09:51``."""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(seconds))


def _number(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def short_count(value: object) -> str:
    """A count in three digits: ``1.97M``, ``34.2K``."""
    number = _number(value)
    if number is None:
        return ""
    for limit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= limit:
            return f"{number / limit:.3g}{suffix}"
    return f"{number:g}"


def duration(value: object) -> str:
    seconds = _number(value)
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def describe_source(uri: object) -> str:
    """A media part's source as the terminal shows it, never its payload: ``image/png 1.2 KiB``."""
    return _describe_non_text_source(uri) if isinstance(uri, str) else "?"


def _json(value: object, indent: Optional[int] = None) -> str:
    if indent is None:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return json.dumps(value, ensure_ascii=False, indent=indent)


def _loads(text: str) -> object:
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


def _arguments(parsed: object, raw: str) -> str:
    """A tool call's arguments: compact JSON, as the terminal shows them, or indented if long."""
    if parsed is None:
        return raw
    compact = _json(parsed)
    return compact if len(compact) <= 200 else json.dumps(parsed, ensure_ascii=False, indent=2,
                                                          sort_keys=True)




def runs_text(labels: Sequence[object]) -> str:
    """A list of labels as the terminal shows it, bounded (``display._runs_text``)."""
    return _runs_text(tuple(str(label) for label in labels))


def _usage(value: object) -> Tuple[object, object, object, object]:
    usage = value if isinstance(value, dict) else {}
    return tuple(usage.get(name, 0) for name in (  # type: ignore[return-value]
        "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"))


def _packed_count(value: object) -> Optional[int]:
    """The values that a token record's base64 array holds, 4 bytes each, without decoding it."""
    if not isinstance(value, str):
        return None
    return (len(value) * 3 // 4 - value[-2:].count("=")) // 4


def _token_trace(tokens: object, cached: object) -> str:
    """A sample's token trace as its ``[sample]`` line shows it: ``tokens=prefix/sample``."""
    if not isinstance(tokens, dict):
        return ""
    prefix = _packed_count(tokens.get("prefix_token_ids"))
    sample = _packed_count(tokens.get("sample_token_ids"))
    text = f" tokens={'-' if prefix is None else prefix}/{'-' if sample is None else sample}"
    offset = tokens.get("prefix_offset")
    if offset is not None and offset != cached:
        text += f" offset={offset}"
    gap = _packed_count(tokens.get("prefix_gap_token_ids"))
    if gap is not None:
        text += f" gap={gap}"
    return text


def _elapsed(record: dict) -> str:
    seconds = _number(record.get("elapsed_seconds"))
    return "" if seconds is None else f" elapsed={seconds:.2f}s"


def _attempts(record: dict) -> str:
    attempts = record.get("request_attempts", 1)
    return "" if attempts == 1 else f" attempts={attempts}"


def _recovery(record: dict) -> str:
    recovery = record.get("recovery")
    return f" recovery={runs_text(recovery)}" if isinstance(recovery, list) and recovery else ""


def _summary_text(item: object) -> Optional[str]:
    """A compaction summary's text, in either style, from a context prefix's item."""
    if not (isinstance(item, dict) and item.get("type") == "message"
            and item.get("role") == "user"):
        return None
    text = item.get("content")
    if not isinstance(text, str):
        return None
    if (len(text) >= len(SUMMARY_PREFIX) + len(SUMMARY_SUFFIX)
            and text.startswith(SUMMARY_PREFIX) and text.endswith(SUMMARY_SUFFIX)):
        return text[len(SUMMARY_PREFIX):len(text) - len(SUMMARY_SUFFIX)]
    if text.startswith(LEGACY_SUMMARY_PREFIX + "\n"):
        return text[len(LEGACY_SUMMARY_PREFIX) + 1:]
    return None


def _is_pointer(item: object) -> bool:
    return (isinstance(item, dict) and item.get("type") == "message"
            and item.get("role") == "user" and isinstance(item.get("content"), str)
            and item["content"].startswith((POINTER_PREFIX, LEGACY_POINTER_PREFIX)))


_SHELL_TOOLS = frozenset(("shell", "container.exec", "local_shell", "shell_command",
                          "exec_command"))
_CODE_TOOLS = frozenset(("apply_patch", "create_file", "edit_file", "insert_into_file",
                         "replace_in_file", "write_file"))
_CODE_KEYS = ("patch", "input", "diff", "raw", "value", "content", "text", "code")
_PLAN_MARKS = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}
_GIT_DIFF = re.compile(r"\bgit\b[^\n;&|]*?\b(?:diff|show)\b")
_OPAQUE = (("encrypted_content", "encrypted content"), ("content_signature", "signature"))


def _command(payload: dict) -> Optional[str]:
    value = payload.get("cmd", payload.get("command"))
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        return " ".join(str(part).strip() for part in value if str(part).strip()) or None
    return None


def _extras(payload: dict, shown: Iterable[str]) -> str:
    """The arguments that a call's main block doesn't show, as ``name=value``."""
    skipped = set(shown)
    return " ".join(f"{name}={_json(value)}" for name, value in payload.items()
                    if name not in skipped)


def _code_payload(parsed: object, raw: str) -> Optional[str]:
    if isinstance(parsed, dict):
        for name in _CODE_KEYS:
            value = parsed.get(name)
            if isinstance(value, str) and value.strip():
                return value
        return None
    if isinstance(parsed, str):
        return parsed if parsed.strip() else None
    if parsed is None:
        return raw if raw.strip() else None
    return None


def _icron_call(arguments: dict) -> Tuple[str, Optional[str]]:
    """An icron call, as the model wrote it, and its code."""
    code = arguments.get("code")
    code = code if isinstance(code, str) and code.strip() else None
    written = arguments.get("call")
    if isinstance(written, str):
        written = written.strip()
        return (written if written.startswith("icron.") else f"icron.{written}"), code
    action = arguments.get("action")
    if isinstance(action, str):
        keywords = ", ".join(f"{name}={value!r}" for name, value in arguments.items()
                             if name not in ("action", "code"))
        return f"icron.{'yield_' if action == 'yield' else action}({keywords})", code
    if code is not None:
        return "icron.exec()", code
    keywords = ", ".join(f"{name}={value!r}" for name, value in arguments.items())
    return f"icron({keywords})", None


def _method_call(tool: str, arguments: dict) -> str:
    """A media or ui tool call, as the model wrote it: ``media.attach('shot.png')``."""
    written = arguments.get("call")
    if isinstance(written, str):
        written = written.strip()
        return written if written.startswith(f"{tool}.") else f"{tool}.{written}"
    keywords = ", ".join(f"{name}={value!r}" for name, value in arguments.items())
    return f"{tool}({keywords})"


def _plan_lines(payload: dict) -> List[str]:
    plan = payload.get("plan")
    if not isinstance(plan, list):
        return []
    lines = []
    explanation = payload.get("explanation")
    if isinstance(explanation, str) and explanation.strip():
        lines.append(f"note: {explanation.strip()}")
    steps = 0
    for item in plan:
        step = str(item.get("step") or "").strip() if isinstance(item, dict) else ""
        if step:
            status = str(item.get("status") or "").strip()
            lines.append(f"{_PLAN_MARKS.get(status, '[' + (status or '?') + ']')} {step}")
            steps += 1
    if not steps:
        lines.append("(no steps provided)")
    return lines


def _diff_line(line: str) -> str:
    if line.startswith(("+++", "---", "*** ", "diff --git")):
        css = "d-file"
    elif line.startswith("@@"):
        css = "d-hunk"
    elif line.startswith("+"):
        css = "d-add"
    elif line.startswith("-"):
        css = "d-del"
    else:
        return h(line)
    return f"<span class={css}>{h(line)}</span>"


# A coldstart pointer: where its log is, which lines it covers, and the
# lines it quotes from it (compaction.py's _pointer).
_POINTER = re.compile(
    r"saved in `(?P<path>[^`\n]+)`(?: \(relative to your tools' working directory\))?, "
    r"(?P<lines>lines (?P<first>[0-9]+)\s*[-\u2013]\s*(?P<last>[0-9]+))")
_BACKTICKED = re.compile(r"`([^`\n]+)`")
_ON_LINE = re.compile(r"is on (?P<ref>line (?P<line>[0-9]+)):")
# A summary's first line (compaction.py's _summary_line).
_COVERS = re.compile(r"Covers (?P<lines>lines (?P<first>[0-9]+)\s*[-\u2013\u2014]\s*"
                     r"(?P<last>[0-9]+)) of (?P<key>(?:context|session)_[0-9a-f]{32})")
_LINES = re.compile(r"\s*([0-9]{1,12})\s*(?:[-\u2013]\s*([0-9]{1,12})\s*)?\Z")
_KEY_HANDLE = re.compile(r"(?:context|session)_([0-9a-f]{4,32})\Z")
_DIGITS = re.compile(r"[0-9]{1,12}\Z")

# Gists: what a folded item's one line shows of it, as plain text.

_PATCH_FILE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.M)
# exec_command's and write_stdin's results (pythia/interaction/local_tools/command.py).
_EXITED = re.compile(r"^Process exited with code (-?[0-9]+)$", re.M)
_RUNNING = re.compile(r"^Process running with session ID ([0-9]+)$", re.M)
_OUTPUT = "\nOutput:\n"


def _first_line(text: str) -> str:
    """A text's first line that isn't blank, with … when more follows."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""
    return lines[0] + (" …" if len(lines) > 1 else "")


def _lines(count: int) -> str:
    return f"{count:,} line{'' if count == 1 else 's'}"


def _reasoning_text(record: dict) -> str:
    """A reasoning item's summary, else its text."""
    summary = record.get("summary")
    parts = [part.strip() for part in summary if isinstance(part, str) and part.strip()] \
        if isinstance(summary, list) else []
    text = "\n\n".join(parts)
    if not text:
        content = message_content(record)
        text = content.strip() if isinstance(content, str) else ""
    return text


def _prefix_items(record: dict) -> list:
    """A context prefix's items: an older log's context_compaction may hold them as
    replacement_items."""
    name = ("prefix_items" if "prefix_items" in record or "replacement_items" not in record
            else "replacement_items")
    items = record.get(name)
    return items if isinstance(items, list) else []


def call_gist(record: dict) -> str:
    """A tool call's line: its command, the files a patch changes, or its arguments."""
    name = record.get("name") if isinstance(record.get("name"), str) else "?"
    arguments = record.get("arguments_json")
    raw = arguments if isinstance(arguments, str) else _json(arguments)
    parsed = _loads(raw)
    payload = parsed if isinstance(parsed, dict) else {}
    if name in _SHELL_TOOLS and _command(payload) is not None:
        return _first_line(_command(payload))
    if name == "write_stdin" and payload:
        parts = []
        if "session_id" in payload:
            parts.append(f"session_id={payload['session_id']}")
        if "chars" in payload:
            parts.append(f"chars={str(payload.get('chars') or '')!r}")
        return " ".join(parts) or _json(payload)
    if name in _CODE_TOOLS:
        code = _code_payload(parsed, raw)
        if code is not None:
            files = _PATCH_FILE.findall(code)
            if files:
                more = f" +{len(files) - 3} more" if len(files) > 3 else ""
                return ", ".join(path.strip() for path in files[:3]) + more
            return _first_line(code)
    if name == "icron" and isinstance(parsed, dict):
        return _first_line(_icron_call(parsed)[0])
    if name in ("media", "ui") and isinstance(parsed, dict):
        return _first_line(_method_call(name, parsed))
    if name == "update_plan" and isinstance(parsed, dict):
        steps = [line for line in _plan_lines(parsed) if not line.startswith("note: ")]
        explanation = parsed.get("explanation")
        if steps:
            return "  ".join(steps)
        if isinstance(explanation, str) and explanation.strip():
            return _first_line(explanation)
    return _first_line(raw) if parsed is None else _json(parsed)


def result_gist(record: dict) -> str:
    """A tool result's line: a command's exit and its output's length, else the output's
    first line."""
    output = record.get("output")
    text = output if isinstance(output, str) else _json(output)
    head = text[:2000]
    exited, running = _EXITED.search(head), _RUNNING.search(head)
    parts = []
    if exited is not None or running is not None:
        parts.append(f"exit {exited.group(1)}" if exited is not None
                     else f"running, session {running.group(1)}")
        _, found, text = text.partition(_OUTPUT)
        if not found:
            text = ""
    lines = [line for line in text.strip("\n").split("\n")] if text.strip() else []
    if not lines:
        parts.append("no output")
    elif parts:
        parts.append(_lines(len(lines)))
    else:
        first = _first_line(text).removesuffix(" …")
        parts.append(first if len(lines) == 1 else f"{first} … {_lines(len(lines))}")
    return " · ".join(parts)


def reasoning_gist(record: dict) -> str:
    text = _reasoning_text(record)
    if text:
        return _first_line(text)
    return ", ".join(what for name, what in _OPAQUE if isinstance(record.get(name), str))


def prefix_gist(record: dict) -> str:
    """A checkpoint's line: its summary's, its coldstart pointer's, or what a provider's
    opaque checkpoint is."""
    items = _prefix_items(record)
    for item in reversed(items):
        text = _summary_text(item)
        if text is not None:
            return _first_line(text)
    if items and _is_pointer(items[-1]):
        return _first_line(items[-1]["content"])
    last = items[-1] if items else None
    if isinstance(last, dict) and last.get("type") == "opaque_compaction":
        protocol = last.get("protocol")
        return "an opaque checkpoint" + (f", protocol={protocol}" if protocol else "")
    return ""


def failure_gist(record: dict) -> str:
    message = record.get("message")
    return _first_line(message) if isinstance(message, str) else ""


def _nested(gist: Callable[[dict], str], name: str) -> Callable[[dict], str]:
    def nested(record: dict) -> str:
        inner = record.get(name)
        return gist(inner) if isinstance(inner, dict) else ""
    return nested


_GISTS: Dict[str, Callable[[dict], str]] = {
    "tool_call": call_gist,
    "tool_result": result_gist,
    "user_tool_call": _nested(call_gist, "call"),
    "user_tool_result": _nested(result_gist, "result"),
    "reasoning": reasoning_gist,
    "context_prefix": prefix_gist,
    "context_compaction": prefix_gist,
    "model_failure": failure_gist,
}


def gist(record: dict) -> str:
    """What a folded item's one line shows of it: plain text, or ""."""
    kind = record.get("type")
    function = _GISTS.get(kind) if isinstance(kind, str) else None
    if function is None:
        return ""
    try:
        return cut(function(record), GIST_CHARS)
    except Exception:  # a known type with fields it doesn't expect: its label says enough
        return ""


def _type_class(kind: object) -> str:
    """An item's type as a class: t-tool_call; t-bad for a bad line; t-other else."""
    if kind == BAD:
        return "t-bad"
    if isinstance(kind, str) and kind in _RENDERERS:
        return f"t-{kind}"
    return "t-other"


class Response:
    """A page: its status and body, and a redirect's location or other headers; and its
    content security policy, or None for the server's, which allows no script."""

    def __init__(self, status: int, body: str = "", *, content_type: str = "text/html",
                 location: Optional[str] = None,
                 headers: Sequence[Tuple[str, str]] = (), csp: Optional[str] = None) -> None:
        self.status = status
        self.body = body
        self.content_type = content_type
        self.location = location
        self.headers = tuple(headers)
        self.csp = csp


class BadRequest(Exception):
    """A query that the node server can't take; the message says why."""


def _whole(text: Optional[str], name: str) -> int:
    if text is None or not _DIGITS.match(text.strip()):
        raise BadRequest(f"{name}={text!r} isn't a number.")
    return int(text)


class Linker:
    """URLs for one context's pages, and for the files that its pointers name."""

    def __init__(self, pages: "Pages", context: Context) -> None:
        self.pages = pages
        self.context = context
        self._targets: Optional[List[Tuple[Tuple[str, ...], Context, str]]] = None

    def line(self, line: int) -> str:
        return self.pages.url(self.context.id, f"line={line}")

    def _resolve(self, written: str) -> Optional[Tuple[Context, str]]:
        """The context whose log or summary a path names, by the path's last parts."""
        parts = tuple(part for part in PurePosixPath(written).parts
                      if part not in ("/", ".", ".."))
        if not parts:
            return None
        if self._targets is None:
            self._targets = []
            for context in self.pages.roots.contexts():
                for what, path in (("log", context.public), ("summary", context.summary)):
                    if path is None:
                        continue
                    for each in {path.parts, Path(os.path.realpath(path)).parts}:
                        self._targets.append((each, context, what))
        found = {(context.id, what): (context, what) for each, context, what in self._targets
                 if len(each) >= len(parts) and each[-len(parts):] == parts}
        return next(iter(found.values())) if len(found) == 1 else None

    def pointer_spans(self, text: str) -> List[Tuple[int, int, str]]:
        """Links in a coldstart pointer: to the log it names, its lines, and its summary."""
        spans = []
        main = _POINTER.search(text)
        target = None
        if main is not None:
            found = self._resolve(main.group("path"))
            if found is not None and found[1] == "log":
                target = found[0]
                lines = f"lines={main.group('first')}-{main.group('last')}"
                spans.append((main.start("path"), main.end("path"), self.pages.url(target.id)))
                spans.append((main.start("lines"), main.end("lines"),
                              self.pages.url(target.id, lines)))
        for match in _BACKTICKED.finditer(text):
            if main is not None and match.start(1) == main.start("path"):
                continue
            found = self._resolve(match.group(1))
            if found is not None and found[1] == "summary":
                spans.append((match.start(1), match.end(1),
                              self.pages.url(found[0].id, "summary")))
        if target is not None:
            for match in _ON_LINE.finditer(text):
                spans.append((match.start("ref"), match.end("ref"),
                              self.pages.url(target.id, f"line={match.group('line')}")))
        return spans


# What an item shows after its label, as parts: a page shows them as HTML
# (Renderer.html), and the glui view's facts as text (view.py).

class Block(NamedTuple):
    """Preformatted text, cut past the renderer's ``cut``: with links, each (start, end,
    URL), or a diff, its lines colored."""
    text: str
    diff: bool = False
    links: Tuple[Tuple[int, int, str], ...] = ()


class Inline(NamedTuple):
    """Text on the label's own line."""
    text: str


class Note(NamedTuple):
    """A dim line of its own, about the rest."""
    text: str


class Line(NamedTuple):
    """A line of its own."""
    text: str


class Fold(NamedTuple):
    """Parts that open in place, under a summary."""
    summary: str
    parts: tuple


class Nest(NamedTuple):
    """Items within an item, as a compaction's prefix holds them."""
    records: Tuple[dict, ...]


class Shown(NamedTuple):
    """An item as it shows: its kind's class, its label, its label's tooltip, its parts."""
    css: str
    label: str
    tip: Optional[str]
    parts: tuple


class Renderer:
    """Items as pythia's terminal shows them, with its labels: what each shows
    (``shown``), and that as HTML (``item``).

    ``calls`` and ``user_calls`` name each call's tool, from the whole log,
    so a result names its tool when its call is on an earlier page. A block
    longer than ``cut`` is cut, with a link to its line's page. An item whose
    type is in ``fold`` shows as one line, its label and its gist, which
    opens in place (``<details>``).
    """

    def __init__(self, linker: Linker, calls: Dict[str, str], user_calls: Dict[str, str],
                 *, cut: Optional[int] = CUT, fold: FrozenSet[str] = frozenset()) -> None:
        self.linker = linker
        self.calls = calls
        self.user_calls = user_calls
        self.cut = cut
        self.fold = frozenset(fold)
        self.line: Optional[int] = None  # the line being shown
        self._diffs: set = set()  # calls on this page whose output is a diff

    def items(self, records: Iterable[Tuple[int, dict]]) -> str:
        return "".join(self.item(line, record) for line, record in records)

    def shown(self, record: dict) -> Optional[Shown]:
        """What an item shows; None for a boundary. A known type with fields that it doesn't
        expect shows as an unknown type does, with a note why."""
        kind = record.get("type")
        render = (_RENDERERS.get(kind) if isinstance(kind, str) else None) or Renderer._other
        try:
            shown = render(self, record)
        except Exception as exc:  # a known type with fields it doesn't expect
            shown = Renderer._other(self, record,
                                    note=f"not shown as {kind}: {type(exc).__name__}: {exc}")
        return None if shown is None else Shown(*shown)

    def item(self, line: Optional[int], record: dict) -> str:
        """One line's item, or a nested one's (``line`` None), as HTML; "" for a boundary."""
        if line is not None:
            self.line = line
        kind = record.get("type")
        shown = self.shown(record)
        if shown is None:
            return ""
        css, label, tip, parts = shown
        body = self.html(parts)
        classes = " ".join(part for part in ("it", css, _type_class(kind)) if part)
        anchor = "" if line is None else f' id="L{line}"'
        number = ("" if line is None
                  else f'<a class=ln href="{h(self.linker.line(line))}">{line}</a>')
        title = f' title="{h(tip)}"' if tip else ""
        head = f"{number}<span class=lb{title}>{h(label)}</span>"
        if self._folds(kind) and body:
            line_gist = gist(record)
            shown_gist = f" <span class=gist>{h(line_gist)}</span>" if line_gist else ""
            return (f'<details class="{classes}"{anchor}><summary>{head}{shown_gist}</summary>'
                    f"{body}</details>\n")
        return f'<div class="{classes}"{anchor}>{head}{body}</div>\n'

    def _folds(self, kind: object) -> bool:
        return isinstance(kind, str) and kind in self.fold

    def html(self, parts: Iterable) -> str:
        """An item's parts as HTML."""
        return "".join(self._part(part) for part in parts)

    def _part(self, part) -> str:
        if isinstance(part, Block):
            return self.block(part.text, diff=part.diff, links=part.links)
        if isinstance(part, Inline):
            return f" {h(part.text)}"
        if isinstance(part, Note):
            return f"<div class=dim>{h(part.text)}</div>"
        if isinstance(part, Line):
            return f"<div>{h(part.text)}</div>"
        if isinstance(part, Fold):
            return f"<details><summary>{h(part.summary)}</summary>{self.html(part.parts)}</details>"
        if isinstance(part, Nest):
            nested = "".join(self.item(None, record) for record in part.records
                             if isinstance(record, dict))
            return f"<div class=nest>{nested}</div>"
        raise TypeError(f"not a part of an item: {part!r}")

    def block(self, text: str, *, diff: bool = False,
              links: Sequence[Tuple[int, int, str]] = ()) -> str:
        """Text, preformatted: cut past ``cut`` characters, with a link to its line."""
        more = 0
        if self.cut is not None and len(text) > self.cut:
            more = len(text) - self.cut
            text = text[:self.cut]
            links = [span for span in links if span[1] <= self.cut]
        if diff:
            inner = "\n".join(_diff_line(part) for part in text.split("\n"))
        else:
            inner = linked(text, links)
        shown = f"<pre>{inner}</pre>"
        if more:
            whole = ("" if self.line is None else
                     f': <a href="{h(self.linker.line(self.line))}">the whole item</a>')
            shown += f"<div class=cut>… {more:,} more characters{whole}</div>"
        return shown

    def _text(self, text: str, *, pointer: bool = False) -> Block:
        if pointer and text.startswith((POINTER_PREFIX, LEGACY_POINTER_PREFIX)):
            return Block(text, links=tuple(self.linker.pointer_spans(text)))
        return Block(text)

    def _message(self, record: dict):
        role = record.get("role")
        role = role.strip() if isinstance(role, str) and role.strip() else "message"
        content = message_content(record)
        parts: list = []
        for part in content if isinstance(content, list) else [content]:
            if isinstance(part, dict) and part.get("type") == "text":
                part = part.get("text")
            if isinstance(part, str):
                if part.strip():
                    parts.append(self._text(part, pointer=role == "user"))
            elif isinstance(part, dict) and part.get("type") == "media":
                parts.append(Line("[image] " + describe_source(part.get("source_uri"))))
            elif part is not None:
                parts.append(Block(_json(part, indent=2)))
        css = role if role in ("user", "assistant") else ""
        return css, f"[{role}]", None, tuple(parts)

    def _reasoning(self, record: dict):
        text = _reasoning_text(record)
        parts: list = []
        if text and self._folds(record.get("type")):
            parts.append(Block(text))  # the item itself folds: no fold within it
        elif text:
            parts.append(Fold(cut(text.split("\n", 1)[0], TITLE_CHARS), (Block(text),)))
        notes = [f"{what}, {len(record[name]):,} characters" for name, what in _OPAQUE
                 if isinstance(record.get(name), str)]
        if notes:
            parts.append(Note("; ".join(notes)))
        return "meta", "[reasoning]", None, tuple(parts)

    def _tool_call(self, record: dict, *, user: bool = False):
        name = record.get("name") if isinstance(record.get("name"), str) else "?"
        call_id = record.get("call_id") if isinstance(record.get("call_id"), str) else ""
        arguments = record.get("arguments_json")
        raw = arguments if isinstance(arguments, str) else _json(arguments)
        label = f"[{'user-tool-call' if user else 'tool-call'}] {name}"
        return "tool", label, call_id, self._call_body(name, call_id, _loads(raw), raw, user)

    def _call_body(self, name: str, call_id: str, parsed: object, raw: str, user: bool) -> tuple:
        payload = parsed if isinstance(parsed, dict) else {}
        if user:
            if isinstance(parsed, dict) and not parsed:
                return ()
            return (Block(_arguments(parsed, raw)),)
        if name == "icron" and isinstance(parsed, dict):
            call, code = _icron_call(parsed)
            return (Block(call),) + (() if code is None else (Block(code),))
        if name in ("media", "ui") and isinstance(parsed, dict):
            return (Block(_method_call(name, parsed)),)
        if name == "write_stdin" and payload:
            parts = []
            if "session_id" in payload:
                parts.append(f"session_id={payload['session_id']}")
            if "chars" in payload:
                parts.append(f"chars={str(payload.get('chars') or '')!r}")
            extras = _extras(payload, ("session_id", "chars"))
            return (Block(" ".join(parts + ([extras] if extras else []))),)
        if name in _SHELL_TOOLS:
            command = _command(payload)
            if command is not None:
                if _GIT_DIFF.search(command):
                    self._diffs.add(call_id)
                extras = _extras(payload, ("cmd", "command"))
                return ((Note(extras),) if extras else ()) + (Block(command),)
        if name in _CODE_TOOLS:
            code = _code_payload(parsed, raw)
            if code is not None:
                extras = _extras(payload, _CODE_KEYS)
                return (((Note(extras),) if extras else ())
                        + (Block(code, diff=name == "apply_patch"),))
        if name == "update_plan" and isinstance(parsed, dict):
            return (Block("\n".join(_plan_lines(parsed))),)
        return (Block(_arguments(parsed, raw)),) if raw.strip() else ()

    def _tool_result(self, record: dict, *, user: bool = False):
        call_id = record.get("call_id") if isinstance(record.get("call_id"), str) else ""
        name = (self.user_calls if user else self.calls).get(call_id, "tool")
        status = "ok" if record.get("success", True) else "error"
        output = record.get("output")
        text = output if isinstance(output, str) else _json(output, indent=2)
        label = f"[{'user-tool-ret' if user else 'tool-ret'}] {name} [{status}]"
        body = (Block(text, diff=call_id in self._diffs),) if text.strip() else ()
        return ("tool" if status == "ok" else "tool err"), label, call_id, body

    def _user_tool_call(self, record: dict):
        call = record.get("call")
        return self._tool_call(call if isinstance(call, dict) else {}, user=True)

    def _user_tool_result(self, record: dict):
        result = record.get("result")
        return self._tool_result(result if isinstance(result, dict) else {}, user=True)

    def _sample_metadata(self, record: dict):
        usage = _usage(record.get("usage"))
        text = (f"input={usage[0]} output={usage[1]} total={usage[2]} cached={usage[3]}"
                f"{_token_trace(record.get('tokens'), usage[3])}{_elapsed(record)}"
                f"{_attempts(record)}{_recovery(record)}")
        return "meta", "[sample]", None, (Inline(text),)

    def _compaction_metadata(self, record: dict):
        usage = _usage(record.get("usage"))
        text = (f"protocol={record.get('protocol')} input={usage[0]} output={usage[1]} "
                f"total={usage[2]} cached={usage[3]}{_elapsed(record)}{_attempts(record)}"
                f"{_recovery(record)}")
        return "meta", "[compaction]", None, (Inline(text),)

    def _turn_summary(self, record: dict):
        get = record.get
        text = (f"input_sum={get('input_tokens_sum')} output_sum={get('output_tokens_sum')} "
                f"cold_sum={get('non_cached_input_tokens_sum')} "
                f"cached_sum={get('cached_input_tokens_sum')} "
                f"cached_max={get('cached_input_tokens_max')} context={get('context_tokens')} "
                f"samples={get('sample_count')} compactions={get('compaction_count')}"
                f"{_elapsed(record)}")
        return "meta", "[turn]", None, (Inline(text),)

    def _model_failure(self, record: dict):
        fields = []
        for name, field in (("provider", "provider"), ("model", "model"),
                            ("auth_source", "auth_source"), ("status", "http_status"),
                            ("attempts", "attempt_count"), ("request_id", "request_id"),
                            ("response_id", "response_id"), ("cf_ray", "cf_ray"),
                            ("auth_error", "authorization_error"),
                            ("auth_code", "auth_error_code"), ("error_code", "error_code"),
                            ("events", "event_count"),
                            ("completed_items", "completed_item_count"),
                            ("last_event", "last_event_type"),
                            ("last_sequence", "last_sequence_number")):
            value = record.get(field)
            if value is None or (field == "attempt_count" and value == 1) or (
                    field in ("event_count", "completed_item_count") and not value):
                continue
            fields.append(f"{name}={value}")
        fields.append(_elapsed(record).strip())
        fields.append(_recovery(record).strip())
        types = record.get("event_types")
        if isinstance(types, list) and types:
            fields.append(f"event_types={runs_text(types)}")
        fields = [field for field in fields if field]
        message = record.get("message")
        parts = [Block(f"{message if isinstance(message, str) else ''} "
                       f"kind={record.get('category')}")]
        if fields:
            parts.append(Fold(f"{len(fields)} more fields", (Block(" ".join(fields)),)))
        return "err", "[model]", None, tuple(parts)

    def _context_prefix(self, record: dict):
        items = _prefix_items(record)
        noun = "item" if len(items) == 1 else "items"
        parts: list = []
        for item in reversed(items):
            text = _summary_text(item)
            if text is not None:
                parts.append(Block(text))
                break
        else:
            if items and _is_pointer(items[-1]):
                parts.append(self._text(items[-1]["content"], pointer=True))
        nested = Nest(tuple(item for item in items if isinstance(item, dict)))
        parts.append(Fold(f"{len(items)} {noun}", (nested,)))
        return "", f"[context prefix] {len(items)} {noun}", None, tuple(parts)

    def _opaque_compaction(self, record: dict):
        text = f"opaque checkpoint, protocol={record.get('protocol')}"
        payload = record.get("payload")
        if isinstance(payload, str):
            text += f", {len(payload):,} characters"
        return "meta", "[compaction]", None, (Inline(text),)

    def _init(self, record: dict):
        prefix_id = record.get("prefix_id", record.get("session_id"))
        model = record.get("model")
        text = " ".join(part for part in (str(prefix_id) if prefix_id is not None else "",
                                          f"model={model}" if model is not None else "") if part)
        return "meta", f"[{record.get('type')}]", None, (Inline(text),)

    def _tools(self, record: dict):
        specs = record.get("specs")
        specs = specs if isinstance(specs, list) else []
        names = ", ".join(str(spec.get("name", "?")) if isinstance(spec, dict) else "?"
                          for spec in specs) or "(none)"
        return ("meta", "[tools]", None,
                (Inline(names), Fold(f"{len(specs)} specs", (Block(_json(specs, indent=2)),))))

    def _instructions(self, record: dict):
        text = record.get("text")
        text = text if isinstance(text, str) else _json(text)
        first = text.strip().split("\n", 1)[0]
        return ("meta", "[instructions]", None,
                (Fold(cut(first, TITLE_CHARS) or "(empty)", (Block(text),)),))

    def _hidden(self, record: dict):
        return None  # a boundary: it marks samples and turns

    def _bad(self, record: dict):
        return "err", "[bad line]", None, (Block(str(record.get("raw", ""))),)

    def _other(self, record: dict, note: Optional[str] = None):
        kind = record.get("type")
        label = f"[{kind}]" if isinstance(kind, str) and kind else "[?]"
        fields = {name: value for name, value in record.items() if name != "type"}
        parts = (((Note(note),) if note else ())
                 + ((Block(_json(fields, indent=2)),) if fields else ()))
        return "", label, None, parts


_RENDERERS: Dict[str, Callable] = {
    "message": Renderer._message,
    "reasoning": Renderer._reasoning,
    "tool_call": Renderer._tool_call,
    "tool_result": Renderer._tool_result,
    "user_tool_call": Renderer._user_tool_call,
    "user_tool_result": Renderer._user_tool_result,
    "sample_metadata": Renderer._sample_metadata,
    "compaction_metadata": Renderer._compaction_metadata,
    "turn_summary": Renderer._turn_summary,
    "model_failure": Renderer._model_failure,
    "context_prefix": Renderer._context_prefix,
    "context_compaction": Renderer._context_prefix,  # its older name
    "opaque_compaction": Renderer._opaque_compaction,
    "init": Renderer._init,
    "session_init": Renderer._init,
    "turn_metadata": Renderer._sample_metadata,  # sample_metadata's older name
    "tools": Renderer._tools,
    "instructions": Renderer._instructions,
    "model_sample_boundary": Renderer._hidden,
    "user_interaction_boundary": Renderer._hidden,
    BAD: Renderer._bad,
}


def hidden(kind: object) -> bool:
    """Whether a line of type ``kind`` shows nothing: a boundary, which marks samples and turns."""
    return isinstance(kind, str) and _RENDERERS.get(kind) is Renderer._hidden


class Pages:
    """The node server's pages for its registry's roots. Every URL starts with the token,
    or with ``base``, a path, if it's given, as a node on its standard streams has it
    (``stdio.py``).

    The pages come in ``style``, by default ``default``, the server's. A
    request gets its own copy (``view``), with its style, and ``here``: its
    path and its query's fields but its style, from which the menu links to
    the same page in each style.
    """

    def __init__(self, roots: Roots, token: str, *, style: str = DEFAULT_STYLE,
                 styles: Optional[Styles] = None, base: Optional[str] = None) -> None:
        self.roots = roots
        self.base = f"/{token}" if base is None else base
        self.styles = Styles() if styles is None else styles
        self.default = style
        self.style = style
        self.here: Tuple[str, List[Tuple[str, str]]] = (f"{self.base}/", [])

    def view(self, *, style: Optional[str] = None,
             here: Optional[Tuple[str, List[Tuple[str, str]]]] = None) -> "Pages":
        """A copy of these pages for one request: in ``style``, at ``here``."""
        view = copy.copy(self)
        if style is not None:
            view.style = style
        if here is not None:
            view.here = (here[0], list(here[1]))
        return view

    # URLs. A context's path is its ID, its number, percent-encoded as any ID would be.
    # Each keeps the page's style, when it isn't the server's.

    def _styled(self, query: str) -> str:
        if self.style == self.default:
            return query
        return f"{query}&{FIELD}={self.style}" if query else f"{FIELD}={self.style}"

    def url(self, cid: Optional[str] = None, query: str = "", fragment: str = "") -> str:
        path = (f"{self.base}/" if cid is None else
                f"{self.base}/c/{quote(cid.encode('utf-8', 'surrogateescape'), safe='/')}")
        query = self._styled(query)
        return path + (f"?{query}" if query else "") + (f"#{fragment}" if fragment else "")

    def key_url(self, key: str) -> str:
        query = self._styled("")
        return f"{self.base}/k/{quote(key)}" + (f"?{query}" if query else "")

    def view_url(self, cid: Optional[str] = None) -> str:
        """The glui view's page of the roots, or of root ``cid``'s log (view.py): without
        the page's style, since the view has looks of its own."""
        return f"{self.base}/lem/" + ("" if cid is None else f"c/{quote(cid)}")

    def here_in(self, style: str) -> str:
        """This page's URL in ``style``."""
        path, fields = self.here
        fields = [*fields, *(() if style == self.default else ((FIELD, style),))]
        return path + (f"?{urlencode(fields)}" if fields else "")

    def _style_field(self) -> str:
        """A GET form's field that keeps the page's style: its action's query doesn't."""
        if self.style == self.default:
            return ""
        return f"<input type=hidden name={FIELD} value={h(self.style)}>"

    def current(self) -> Style:
        """The page's style; a name that css/ lost since the request was checked shows
        as itself, with base.css alone."""
        return self.styles.get(self.style) or Style(self.style, self.style)

    def _a(self, url: str, text: str) -> str:
        return f'<a href="{h(url)}">{h(text)}</a>'

    def _menu(self) -> str:
        """The style menu, at the page's top right: this page in each style."""
        items = []
        for style in self.styles.all():
            current = " aria-current=page" if style.name == self.style else ""
            note = f"<span class=note>{h(style.note)}</span>" if style.note else ""
            items.append(f'<li><a href="{h(self.here_in(style.name))}"{current}>'
                         f"{h(style.label)}{note}</a></li>")
        ignored = self.styles.ignored()
        if ignored:
            items.append(f"<li class=note>Not styles, by their names: {h(', '.join(ignored))}. "
                         "A style's name has a–z, 0–9, - and _, and starts with a letter or "
                         "a digit.</li>")
        return (f'<details class=node-menu><summary title="How the pages look">style: '
                f"{h(self.current().label)}</summary><ul>{''.join(items)}</ul></details>\n")

    def _page(self, title: str, crumbs: Sequence[Tuple[str, Optional[str]]], body: str,
              status: int = 200) -> Response:
        nav = " › ".join(self._a(url, text) if url else h(text) for text, url in crumbs)
        css, problems = self.styles.css(self.style)
        notes = "".join(f"<p class=node-note>{h(problem)}</p>\n" for problem in problems)
        return Response(status, (
            "<!doctype html>\n<html lang=en>\n<meta charset=utf-8>\n"
            '<meta name=viewport content="width=device-width, initial-scale=1">\n'
            "<meta name=referrer content=no-referrer>\n"
            f"<title>{h(title)} · node</title>\n<style>\n{css}\n</style>\n"
            f"<nav>{nav}</nav>\n{self._menu()}{notes}{body}\n"))

    def not_found(self, message: str) -> Response:
        return self._page("not found", [("contexts", self.url()), ("not found", None)],
                          f"<h1>Not found</h1>\n<p>{h(message)}</p>", status=404)

    def bad_request(self, message: str) -> Response:
        return self._page("bad request", [("contexts", self.url()), ("bad request", None)],
                          f"<h1>Bad request</h1>\n<p>{h(message)}</p>", status=400)

    def _renderer(self, context: Context, *, cut: Optional[int] = CUT,
                  fold: Optional[FrozenSet[str]] = None) -> Renderer:
        """A renderer for ``context``'s items, which folds what the page's style folds."""
        index = context.index
        fold = self.current().fold if fold is None else fold
        return Renderer(Linker(self, context), index.calls, index.user_calls, cut=cut,
                        fold=fold)

    def _crumbs(self, context: Context, *rest: Tuple[str, Optional[str]]):
        return [("contexts", self.url()), (context.label, self.url(context.id)), *rest]

    # The contexts.

    def contexts(self) -> Response:
        entries = self.roots.entries()
        found = {context.id: context for context in self.roots.contexts()}
        problems: Dict[str, str] = {}
        for context in found.values():
            try:
                context.refresh()
            except OSError as exc:
                problems[context.id] = str(exc)
        keys = [entry.key for entry in entries]
        absent = [entry for entry in entries if str(entry.number) not in found]
        rows = [self._context_row(context, keys, problems)
                for context in sorted(found.values(),
                                      key=lambda each: (-each.index.changed, each.number))]
        rows += [self._absent_row(entry, keys) for entry in absent]
        count = len(entries)
        note = self.roots.problem() or (
            f"{count} root context{'' if count == 1 else 's'}, the last changed first."
            f"{absent_note(len(absent))}" if entries else "None yet.")
        table = ("<table>\n<tr><th class=r>#</th><th>context</th><th>save in</th><th>key</th>"
                 "<th>model</th><th class=r>turns</th><th class=r>lines</th><th class=r>size</th>"
                 "<th>changed</th><th>title</th></tr>\n" + "".join(rows) + "</table>"
                 if rows else "")
        view = self._a(self.view_url(), "In glui's view")
        body = (f"<h1>Root contexts in {h(shown_path(self.roots.path))}/</h1>\n"
                f"<p class=dim>{h(note)}</p>\n"
                f"<p>{view}</p>\n{table}")
        return self._page("contexts", [("contexts", None)], body)

    def _context_row(self, context: Context, keys: List[str], problems: Dict[str, str]) -> str:
        key = (f'<span class=mono title="{h(context.key)}">'
               f"{h(abbreviate(context.key, keys))}</span>")
        if context.id in problems:
            counts = "<td class=r>?</td><td class=r>?</td><td class=r>?</td><td>?</td>"
            title = f"<span class=dim>can't read it: {h(problems[context.id])}</span>"
        else:
            counts = (f"<td class=r>{len(context.turns()):,}</td><td class=r>{len(context):,}</td>"
                      f"<td class=r>{byte_count(context.index.size)}</td>"
                      f"<td class=nw>{when(context.index.changed)}</td>")
            title = h(cut(context.title(), TITLE_CHARS))
        return (f"<tr><td class=r>{context.number}</td>"
                f"<td class=nw>{self._a(self.url(context.id), context.name)}</td>"
                f"<td class=nw><span class=dim>{h(shown_path(context.prefix.parent))}</span></td>"
                f"<td class=nw>{key}</td><td>{h(context.model or '')}</td>{counts}"
                f"<td>{title}</td></tr>\n")

    def _absent_row(self, entry: Entry, keys: List[str]) -> str:
        """A root whose save isn't where it's registered: its number, key, path and why."""
        where = "-" if entry.home is None else shown_path(save_prefix_of(Path(entry.home)))
        key = abbreviate(entry.key, keys) if _KEY.match(entry.key) else entry.key
        return (f"<tr class=dim><td class=r>{entry.number}</td><td colspan=2>{h(where)}</td>"
                f"<td class=nw><span class=mono>{h(key)}</span></td>"
                f"<td colspan=6>{h(ABSENT.get(entry.state, entry.state))}</td></tr>\n")

    # A context, and its views.

    def context(self, cid: str, params: Dict[str, str]) -> Response:
        context = self.roots.get(cid)
        if context is None:
            return self._not_shown(cid, params)
        try:
            context.refresh()
            if "line" in params:
                return self._line(context, params["line"])
            if "lines" in params:
                return self._lines(context, params["lines"])
            if "turn" in params:
                return self._turn(context, params["turn"], params.get("page"))
            if "summary" in params:
                return self._summary(context)
            return self._overview(context)
        except BadRequest as exc:
            return self.bad_request(str(exc))
        except OSError as exc:
            return self.not_found(f"{context.label}'s log can't be read: {exc}")

    def _not_shown(self, cid: str, params: Dict[str, str]) -> Response:
        """A number that no shown root has: another number of a root, which leads to it;
        a root that isn't where it's registered, and why; or nothing."""
        entry = self.roots.entry(cid)
        if entry is not None and entry.first is not None:
            return Response(303, location=self.url(str(entry.first),
                                                   urlencode(list(params.items()))))
        if entry is not None:
            where = ("" if entry.home is None else
                     f", {shown_path(save_prefix_of(Path(entry.home)))},")
            return self.not_found(f"Root context {cid}{where} isn't shown. "
                                  f"{ABSENT.get(entry.state, entry.state)}")
        return self.not_found(f"There's no root context {cid} in "
                              f"{shown_path(self.roots.path)}/.")

    def _overview(self, context: Context) -> Response:
        count = len(context)
        starts = context.turns()
        facts = [("root context", f"{context.number} in {h(shown_path(self.roots.path))}/"),
                 ("key", h(context.key))]
        if context.prefix_id and context.prefix_id != context.key:
            facts.append(("prefix ID", h(context.prefix_id)))
        if context.model:
            facts.append(("model", h(context.model)))
        facts.append(("save", h(context.prefix)))
        facts.append(("glui view", self._a(self.view_url(context.id),
                                           "its log, in glui's widgets")))
        for what, path in context.files():
            try:
                size = byte_count(path.stat().st_size)
            except OSError:
                size = "?"
            if what == "summary":
                shown = self._a(self.url(context.id, "summary"), str(path))
            else:
                shown = h(path)
            extra = f", {count:,} lines" if what == "full log" else ""
            facts.append((what, f"{shown} <span class=dim>({size}{extra})</span>"))
        dl = "".join(f"<dt>{h(term)}</dt><dd>{value}</dd>" for term, value in facts)
        first = starts[0] if starts else count
        start = ""
        if first:
            shown = min(first, START_LINES)
            all_lines = self._a(self.url(context.id, f"lines=1-{first}"), f"All {first:,} lines")
            more = "" if shown == first else f"<p>{all_lines}</p>"
            start = (f"<details><summary>The start: lines 1–{first}</summary>"
                     f"{self._renderer(context).items(context.records(0, shown))}{more}</details>")
        every = self._a(self.url(context.id, f"lines=1-{count}"), "all lines")
        goto = (f'<form action="{h(self.url(context.id))}" method=get>'
                f"<label>Line <input name=line type=number min=1 max={max(count, 1)} required>"
                f"</label>{self._style_field()} <button>Go</button> "
                f"<span class=dim>· {every}</span></form>")
        rows = "".join(self._turn_row(context, row) for row in context.turn_rows())
        turns = (("<table>\n<tr><th class=r>turn</th><th class=r>lines</th><th>input</th>"
                  "<th class=r>samples</th><th class=r>tool calls</th><th class=r>failures</th>"
                  "<th class=r>tokens</th><th class=r>time</th></tr>\n" + rows + "</table>")
                 if rows else "<p class=dim>No turns yet.</p>")
        body = (f"<h1>{h(context.label)}</h1>\n<dl>{dl}</dl>\n{start}\n{goto}\n"
                f"<h2>{len(starts)} turn{'' if len(starts) == 1 else 's'}</h2>\n{turns}")
        return self._page(context.label, self._crumbs(context)[:-1] + [(context.label, None)],
                          body)

    def _turn_row(self, context: Context, row: TurnRow) -> str:
        turn = self.url(context.id, f"turn={row.number}")
        tokens = tip = ""
        seconds = ""
        if row.summary is not None:
            get = row.summary.get
            tokens = (f"{short_count(get('input_tokens_sum'))} in · "
                      f"{short_count(get('output_tokens_sum'))} out")
            tip = (f"input {get('input_tokens_sum')}, cached {get('cached_input_tokens_sum')}, "
                   f"output {get('output_tokens_sum')}, context {get('context_tokens')}")
            seconds = duration(get("elapsed_seconds"))
        lines = self._a(self.url(context.id, f"lines={row.start + 1}-{row.stop}"),
                        f"{row.start + 1}–{row.stop}")
        label = self._a(turn, cut(row.label, TITLE_CHARS) or "(no input)")
        return (f"<tr><td class=r>{self._a(turn, str(row.number))}</td>"
                f"<td class=r>{lines}</td><td>{label}</td>"
                f"<td class=r>{row.samples}</td><td class=r>{row.calls}</td>"
                f"<td class=r>{row.failures or ''}</td>"
                f'<td class=r title="{h(tip)}">{h(tokens)}</td>'
                f"<td class=r>{h(seconds)}</td></tr>\n")

    def _turn(self, context: Context, text: str, page_text: Optional[str]) -> Response:
        number = _whole(text, "turn")
        starts = context.turns()
        start, stop = context.turn_range(number) if number <= len(starts) else (0, 0)
        if number > len(starts) or start == stop:
            what = "lines before its first turn" if number == 0 else f"turn {number}"
            return self.not_found(f"{context.label} has no {what}; it has "
                                  f"{len(starts)} turn{'' if len(starts) == 1 else 's'}.")
        pages = -(-(stop - start) // PAGE_LINES)
        page = 1 if page_text is None else _whole(page_text, "page")
        if not 1 <= page <= pages:
            return self.not_found(f"Turn {number} of {context.label} has {pages} pages.")
        first = start + (page - 1) * PAGE_LINES
        last = min(stop, first + PAGE_LINES)
        name = "the start" if number == 0 else f"turn {number} of {len(starts)}"
        links = []
        has_start = bool(starts) and starts[0] > 0
        if number > 1 or (number == 1 and has_start):
            links.append(self._a(self.url(context.id, f"turn={number - 1}"),
                                 "← the start" if number == 1 else f"← turn {number - 1}"))
        if page > 1:
            links.append(self._a(self.url(context.id, f"turn={number}&page={page - 1}"),
                                 f"← page {page - 1}"))
        if page < pages:
            links.append(self._a(self.url(context.id, f"turn={number}&page={page + 1}"),
                                 f"page {page + 1} →"))
        if number < len(starts):
            links.append(self._a(self.url(context.id, f"turn={number + 1}"),
                                 f"turn {number + 1} →"))
        pager = f"<div class=pager>{' · '.join(links)}</div>" if links else ""
        where = f"lines {start + 1}–{stop}" + (f", page {page} of {pages}: lines "
                                               f"{first + 1}–{last}" if pages > 1 else "")
        items = self._renderer(context).items(context.records(first, last))
        body = (f"<h1>{h(context.label)}: {h(name)}</h1>\n<p class=dim>{h(where)}</p>\n"
                f"{pager}\n{items}{pager}")
        return self._page(f"{context.label}: {name}", self._crumbs(context, (name, None)), body)

    def _lines(self, context: Context, text: str) -> Response:
        match = _LINES.match(text)
        if match is None:
            raise BadRequest(f"lines={text!r}: give a line or a range of lines, such as 1-200.")
        first, last = int(match.group(1)), int(match.group(2) or match.group(1))
        count = len(context)
        if first < 1 or last < first:
            raise BadRequest(f"lines={text!r}: give a range from line 1 up.")
        if first > count:
            return self.not_found(f"{context.label} has {count:,} lines.")
        last = min(last, count)
        stop = min(last, first + PAGE_LINES - 1)
        links = []
        if first > 1:
            earlier = f"lines={max(1, first - PAGE_LINES)}-{first - 1}"
            links.append(self._a(self.url(context.id, earlier), "← earlier lines"))
        if stop < last:
            links.append(self._a(self.url(context.id, f"lines={stop + 1}-{last}"),
                                 f"lines {stop + 1}–{last} →"))
        pager = f"<div class=pager>{' · '.join(links)}</div>" if links else ""
        turns = sorted({context.turn_of(first - 1), context.turn_of(stop - 1)})
        named = " to ".join("the start" if turn == 0 else f"turn {turn}" for turn in turns)
        name = f"lines {first}–{stop}"
        items = self._renderer(context).items(context.records(first - 1, stop))
        body = (f"<h1>{h(context.label)}: {h(name)}</h1>\n<p class=dim>of {count:,}; "
                f"{h(named)}</p>\n{pager}\n{items}{pager}")
        return self._page(f"{context.label}: {name}", self._crumbs(context, (name, None)), body)

    def _line(self, context: Context, text: str) -> Response:
        number = _whole(text, "line")
        count = len(context)
        if not 1 <= number <= count:
            return self.not_found(f"{context.label} has {count:,} lines.")
        raw = context.index.read(number - 1, number)[0]
        record = parse(raw)
        # One item, whole: never folded, in any style.
        item = self._renderer(context, cut=None, fold=frozenset()).item(number, record)
        turn = context.turn_of(number - 1)
        page = (number - 1 - context.turn_range(turn)[0]) // PAGE_LINES + 1
        links = []
        if number > 1:
            links.append(self._a(self.url(context.id, f"line={number - 1}"), "← line before"))
        query = f"turn={turn}" + (f"&page={page}" if page > 1 else "")
        links.append(self._a(self.url(context.id, query, f"L{number}"),
                             "the start" if turn == 0 else f"turn {turn}"))
        if number < count:
            links.append(self._a(self.url(context.id, f"line={number + 1}"), "line after →"))
        if record.get("type") == BAD:
            source = raw.decode("utf-8", "replace")
        else:
            source = json.dumps(record, ensure_ascii=False, indent=2)
        name = f"line {number}"
        body = (f"<h1>{h(context.label)}: {h(name)} of {count:,}</h1>\n"
                f"<div class=pager>{' · '.join(links)}</div>\n{item}"
                f"<details open><summary>The line, {byte_count(len(raw))}</summary>"
                f"<pre>{h(source)}</pre></details>")
        return self._page(f"{context.label}: {name}", self._crumbs(context, (name, None)), body)

    def _summary(self, context: Context) -> Response:
        path = context.summary
        if path is None or not path.is_file():
            return self.not_found(f"{context.label} has no summary.")
        with open(path, "rb") as summary:
            data = summary.read(SUMMARY_BYTES + 1)
        text = data[:SUMMARY_BYTES].decode("utf-8", "replace")
        spans = []
        covers = _COVERS.match(text)
        if covers is not None:
            key = key_of(covers.group("key"))
            holders = ([context] if context.key == key else
                       [other for other in self.roots.contexts() if other.key == key])
            lines = f"lines={covers.group('first')}-{covers.group('last')}"
            url = (self.url(holders[0].id, lines) if len(holders) == 1
                   else self.key_url(covers.group("key")))
            spans.append((covers.start("lines"), covers.end("lines"), url))
        more = ("" if len(data) <= SUMMARY_BYTES else
                f"<p class=dim>Only its first {byte_count(SUMMARY_BYTES)} are shown.</p>")
        body = (f"<h1>{h(context.label)}: its summary</h1>\n<p class=dim>{h(path)}</p>\n"
                f"<pre>{linked(text, spans)}</pre>{more}")
        return self._page(f"{context.label}: summary", self._crumbs(context, ("summary", None)),
                          body)

    # Keys.

    def key(self, text: str) -> Response:
        match = _KEY_HANDLE.match(text)
        if match is None:
            return self.bad_request(f"{text!r} isn't a key: give context_ and at least 4 of "
                                    "its hex digits.")
        stem = KEY_STEM + match.group(1)
        found = sorted((context for context in self.roots.contexts()
                        if context.key.startswith(stem)), key=lambda context: context.number)
        if len(found) == 1:
            return Response(303, location=self.url(found[0].id))
        if not found:
            absent = [entry for entry in self.roots.entries() if entry.key.startswith(stem)]
            if len(absent) == 1:  # the key's root, which isn't shown: why
                return self._not_shown(str(absent[0].number), {})
            return self.not_found(f"No root context in {shown_path(self.roots.path)}/ has a key "
                                  f"that starts with {stem}.")
        items = "".join(f"<li>{self._a(self.url(context.id), context.label)} "
                        f"<span class=dim>{h(context.key)}</span></li>" for context in found)
        body = (f"<h1>{len(found)} root contexts have a key that starts with {h(stem)}</h1>\n"
                f"<ul>{items}</ul>")
        return self._page(stem, [("contexts", self.url()), (stem, None)], body)
