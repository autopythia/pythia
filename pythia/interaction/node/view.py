"""The glui view: the node's root contexts, in glui's widgets, in a browser.

Its two pages, ``/TOKEN/lem/`` for the roots and ``/TOKEN/lem/c/N`` for root
N's log, each load glui's scripts and stylesheets, and the program that
shows them, ``lem/contexts.lem`` compiled, from ``_pythia_ui/`` beside the
package (``Files``), at ``/TOKEN/lem/files/NAME``. A page's facts come as
JSON steps (``_pythia_ui/src/lem/host.rs``), from ``/TOKEN/lem/facts/`` and
``/TOKEN/lem/facts/c/N``: for a log, its turns, folded. A turn's items come
when it's opened, from ``/TOKEN/lem/turn/N/TURN``, so that a page holds
only what's been opened; and an item's body when its entry opens, from
``/TOKEN/lem/body/N/LINE``. An item's label and gist are those of the
server's plain pages, and its body holds what they show for it
(``pages.Renderer``), as text, cut as they cut it. Every text is as pythia's
terminal shows it, each character to escape as its code point.

The view's pages have a content security policy of their own, which runs
the server's scripts, styles and fetches, and nothing else; its facts,
bodies and files allow nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote

from .._cli_editor import safe_text
from ..display import _format_byte_count as byte_count
from .contexts import Context, parse, shown_path
from .pages import (GIST_CHARS, TITLE_CHARS, Block, Fold, Inline, Line, Nest, Note, Pages,
                    Renderer, Response, cut, gist, h, hidden, when)

# _pythia_ui/ in the checkout that holds the package: pythia's installs are
# editable, so it's always there.
UI_DIRECTORY = Path(__file__).resolve().parents[3] / "_pythia_ui"
# The view's files: each name in its URL, and its path in _pythia_ui/.
FILES = {
    "lem.js": "src/lem/lem.js",
    "glem.js": "src/glem/glem.js",
    "glui.js": "src/glui/glui.js",
    "contexts.js": "web/contexts.js",
    "glui.css": "web/glui.css",
    "native.css": "web/native.css",
    "log.css": "web/log.css",
    "contexts.css": "web/contexts.css",
    # contexts.js fetches these beside itself.
    "compiled/contexts.json": "web/compiled/contexts.json",
    "compiled/theme-dark.json": "web/compiled/theme-dark.json",
    "compiled/theme-light.json": "web/compiled/theme-light.json",
}
TYPES = {".js": "text/javascript", ".css": "text/css", ".json": "application/json"}
SCRIPTS = ("lem.js", "glem.js", "glui.js", "contexts.js")
STYLESHEETS = ("glui.css", "native.css", "log.css", "contexts.css")
PAGE_CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
DATA_CSP = "default-src 'none'; frame-ancestors 'none'"
# An item's tone, by its kind's class on the plain pages; any other is Dim.
TONES = {"user": "User", "assistant": "Assistant", "tool": "Tool", "tool err": "Error",
         "err": "Error"}
# How far back a result's body looks for its call, which may make it a diff.
CALL_LINES = 64
# A line's number in a body's URL, and a turn's in a turn's.
_LINE = re.compile(r"[1-9][0-9]{0,11}\Z")
_TURN = re.compile(r"(?:0|[1-9][0-9]{0,11})\Z")


class Files:
    """The view's files, by name (``FILES``), read from ``directory`` for each request, so
    that an edit shows at the next reload; no other file, and no path from a URL."""

    def __init__(self, directory=UI_DIRECTORY) -> None:
        self.directory = Path(directory)

    def missing(self) -> List[str]:
        """The files that the view needs and that aren't there, by their paths."""
        return [path for path in FILES.values() if not (self.directory / path).is_file()]

    def get(self, name: str) -> Optional[Response]:
        """File ``name``, with its type; None if it isn't one of the view's, or can't be read."""
        path = FILES.get(name)
        if path is None:
            return None
        try:
            text = (self.directory / path).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        return Response(200, text, content_type=TYPES[Path(path).suffix], csp=DATA_CSP)


def _steps(steps: list) -> Response:
    return Response(200, json.dumps(steps, ensure_ascii=False, separators=(",", ":")),
                    content_type="application/json", csp=DATA_CSP)


def _plain(status: int, text: str) -> Response:
    return Response(status, text + "\n", content_type="text/plain", csp=DATA_CSP)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def turn_label(name: str, count: int) -> str:
    """A turn's label: its name and how many items it has, ``Turn 12 · 51 items``."""
    return f"{name} \u00b7 {count:,} item{'' if count == 1 else 's'}"


def texts(renderer: Renderer, parts) -> List[Tuple[str, bool]]:
    """An item's parts as its body's texts, each with whether it's a diff: a block, cut as
    the plain pages cut it, then how much more it has; a fold's parts, without its
    summary; and an item within it, its label, then its parts."""
    found: List[Tuple[str, bool]] = []
    for part in parts:
        if isinstance(part, Block):
            text = part.text
            if renderer.cut is not None and len(text) > renderer.cut:
                found.append((text[:renderer.cut], part.diff))
                found.append((f"\u2026 {len(text) - renderer.cut:,} more characters: the whole "
                              "item is on its line's page.", False))
            else:
                found.append((text, part.diff))
        elif isinstance(part, (Inline, Note, Line)):
            if part.text.strip():
                found.append((part.text, False))
        elif isinstance(part, Fold):
            found.extend(texts(renderer, part.parts))
        elif isinstance(part, Nest):
            for record in part.records:
                shown = renderer.shown(record)
                if shown is not None:
                    found.append((shown.label, False))
                    found.extend(texts(renderer, shown.parts))
    return found


def item(renderer: Renderer, line: int, turn: int, record: dict) -> Optional[dict]:
    """Line ``line``'s Item: its tone, label and gist, and whether it has a body, which it
    has unless its gist is all of it, on one line. None for a line that shows nothing.

    A message has no gist of its own: its gist is its text, on one line, cut.
    """
    shown = renderer.shown(record)
    if shown is None:
        return None
    body = texts(renderer, shown.parts)
    line_gist = gist(record) or (cut(_one_line(body[0][0]), GIST_CHARS) if body else "")
    whole = len(body) == 1 and not body[0][1] and body[0][0].strip() == line_gist
    return {"line": line, "turn": turn, "tone": TONES.get(shown.css, "Dim"),
            "label": safe_text(shown.label), "gist": safe_text(line_gist),
            "body": bool(body) and not whole}


class View:
    """The view's routes, past ``/TOKEN/lem/``, on the server's pages (``pages.Pages``)."""

    def __init__(self, files: Optional[Files] = None) -> None:
        self.files = Files() if files is None else files

    def respond(self, pages: Pages, rest: str) -> Response:
        """``rest``, the path past ``/TOKEN/lem/``, percent-decoded."""
        what, _, name = rest.partition("/")
        if not rest:
            return self._page(pages, "root contexts", f"{pages.base}/lem/facts/")
        if what == "c" and name:
            return self._log_page(pages, name)
        if what == "facts" and not name:
            return _steps(self.index_facts(pages))
        if what == "facts" and name.startswith("c/"):
            facts = self.log_facts(pages, name[2:])
            return _plain(404, "There's no such root context here.") if facts is None \
                else _steps(facts)
        if what == "turn":
            cid, _, turn = name.partition("/")
            return self.turn(pages, cid, turn)
        if what == "body":
            cid, _, line = name.partition("/")
            return self.body(pages, cid, line)
        if what == "files":
            return self.files.get(name) or _plain(404, "The glui view has no such file.")
        return pages.not_found("The node server has no such page.")

    # The pages.

    def _page(self, pages: Pages, title: str, facts: str, body: Optional[str] = None, *,
              turn: Optional[str] = None, plain: Optional[str] = None) -> Response:
        """A page of the view: the URLs of its facts, its turns' items and its bodies, what
        it loads, and, for a browser without script, a link to the same in the plain pages,
        ``plain``."""
        missing = self.files.missing()
        if missing:
            return pages.not_found(f"The glui view needs files that aren't in "
                                   f"{shown_path(self.files.directory)}/: "
                                   f"{', '.join(missing)}.")
        files = f"{pages.base}/lem/files/"
        data = (f' data-facts="{h(facts)}"' + ("" if turn is None else f' data-turn="{h(turn)}"')
                + ("" if body is None else f' data-body="{h(body)}"'))
        plain = f"{pages.base}/" if plain is None else plain
        return Response(200, (
            f"<!doctype html>\n<html lang=en{data}>\n<head>\n<meta charset=utf-8>\n"
            '<meta name=viewport content="width=device-width, initial-scale=1">\n'
            "<meta name=referrer content=no-referrer>\n"
            f"<title>{h(title)} · node</title>\n"
            + "".join(f'<link rel=stylesheet href="{h(files + name)}">\n' for name in STYLESHEETS)
            + "</head>\n<body>\n"
            + f'<noscript><p>The glui view runs in JavaScript; <a href="{h(plain)}">the plain '
            "pages</a> need none.</p></noscript>\n"
            + "".join(f'<script src="{h(files + name)}"></script>\n' for name in SCRIPTS)
            + "</body>\n</html>\n"), csp=PAGE_CSP)

    def _log_page(self, pages: Pages, cid: str) -> Response:
        context = pages.roots.get(cid)
        if context is None:
            entry = pages.roots.entry(cid)
            if entry is not None and entry.first is not None:  # another number of a root
                return Response(303, location=f"{pages.base}/lem/c/{entry.first}")
            return pages.context(cid, {})  # why there's no such root here
        number = quote(context.id)
        return self._page(pages, context.label, f"{pages.base}/lem/facts/c/{number}",
                          f"{pages.base}/lem/body/{number}/{{line}}",
                          turn=f"{pages.base}/lem/turn/{number}/{{turn}}",
                          plain=f"{pages.base}/c/{number}")

    # The facts.

    def index_facts(self, pages: Pages) -> list:
        """The roots, the last changed first, each with what the plain index shows, and a
        link to its log; a root that can't be read says so in its title."""
        roots = pages.roots
        found = roots.contexts()
        problems = {}
        for context in found:
            try:
                context.refresh()
            except OSError as exc:
                problems[context.id] = str(exc)
        title = roots.problem() or f"Root contexts in {shown_path(roots.path)}/"
        steps: list = [["+", "Page", {"title": safe_text(title)}],
                       ["+", "Nav", {"n": 0, "text": "Plain pages", "to": f"{pages.base}/"}]]
        absent = len(roots.entries()) - len(found)
        if absent > 0:  # listed on the plain pages, with why
            steps.append(["+", "Nav", {"n": 1, "to": f"{pages.base}/", "text": (
                "1 isn't where it's registered" if absent == 1
                else f"{absent} aren't where they're registered")}])
        ordered = sorted(found, key=lambda each: (-each.index.changed, each.number))
        for rank, context in enumerate(ordered):
            problem = problems.get(context.id)
            steps.append(["+", "Context", {
                "id": safe_text(context.label),
                "title": safe_text(f"can't read it: {problem}" if problem
                                   else cut(context.title(), TITLE_CHARS)),
                "model": safe_text(context.model or ""),
                "turns": 0 if problem else len(context.turns()),
                "lines": 0 if problem else len(context),
                "size": "?" if problem else byte_count(context.index.size),
                "changed": "?" if problem else when(context.index.changed),
                "rank": rank, "to": f"{pages.base}/lem/c/{quote(context.id)}"}])
        return steps

    def log_facts(self, pages: Pages, cid: str) -> Optional[list]:
        """Root ``cid``'s log: its turns, each with how many items it shows, whose items come
        when it's opened (``turn``); "The start" for the lines before the first. None if
        there's no such root here, or its log can't be read."""
        context = pages.roots.get(cid)
        if context is None:
            return None
        try:
            context.refresh()
        except OSError:
            return None
        number = quote(context.id)
        steps: list = [
            ["+", "Page", {"title": safe_text(context.label),
                           "items": f"{pages.base}/c/{number}?line=",
                           "turns": f"{pages.base}/c/{number}?turn="}],
            ["+", "Nav", {"n": 0, "text": "Root contexts", "to": f"{pages.base}/lem/"}],
            ["+", "Nav", {"n": 1, "text": "Plain pages", "to": f"{pages.base}/c/{number}"}]]
        index = context.index
        with index.lock:
            def count(start: int, stop: int) -> int:
                return sum(1 for line in range(start, stop) if not hidden(index.kind(line)[0]))

            starts = context.turns()
            first = starts[0] if starts else len(index)
            if first > 0:
                steps.append(["+", "Turn", {"n": 0, "label": turn_label("The start", count(0, first)),
                                            "title": ""}])
            for row in context.turn_rows():
                steps.append(["+", "Turn", {
                    "n": row.number, "label": turn_label(f"Turn {row.number}",
                                                         count(row.start, row.stop)),
                    "title": safe_text(cut(row.label, TITLE_CHARS) or "(no input)")}])
        return steps

    def turn(self, pages: Pages, cid: str, text: str) -> Response:
        """Turn ``text`` of root ``cid``: that its items came, and an Item for each of its
        lines that shows; turn 0 is the lines before the first."""
        context = pages.roots.get(cid)
        if context is None:
            return _plain(404, "There's no such root context here.")
        try:
            context.refresh()
            number = int(text) if _TURN.match(text) else -1
            if not 0 <= number <= len(context.turns()):
                return _plain(404, f"{context.label} has no turn {text[:24]}.")
            start, stop = context.turn_range(number)
            if start == stop:
                return _plain(404, f"{context.label} has no lines before its first turn.")
            records = context.records(start, stop)
        except OSError as exc:
            return _plain(404, f"{context.label}'s log can't be read: {exc}")
        renderer = pages._renderer(context, fold=frozenset())
        steps: list = [["+", "Loaded", {"turn": number}]]
        for line, record in records:
            fact = item(renderer, line, number, record)
            if fact is not None:
                steps.append(["+", "Item", fact])
        return _steps(steps)

    def body(self, pages: Pages, cid: str, text: str) -> Response:
        """Line ``text`` of root ``cid``: its item's body, as Body facts."""
        context = pages.roots.get(cid)
        if context is None:
            return _plain(404, "There's no such root context here.")
        try:
            context.refresh()
            line = int(text) if _LINE.match(text) else 0
            if not 1 <= line <= len(context):
                return _plain(404, f"{context.label} has no line {text[:24]}.")
            record = parse(context.index.read(line - 1, line)[0])
            renderer = pages._renderer(context, fold=frozenset())
            self._call_of(renderer, context, line, record)
            shown = renderer.shown(record)
        except OSError as exc:
            return _plain(404, f"{context.label}'s log can't be read: {exc}")
        parts = [] if shown is None else texts(renderer, shown.parts)
        if not parts:
            return _plain(404, f"Line {line} of {context.label} has no body.")
        return _steps([["+", "Body", {"line": line, "part": part, "text": safe_text(body),
                                      "diff": diff}]
                       for part, (body, diff) in enumerate(parts)])

    def _call_of(self, renderer: Renderer, context: Context, line: int, record: dict) -> None:
        """A tool's result: its call, shown first to ``renderer``, which shows the result as a
        diff when the call was a git diff. Looks back ``CALL_LINES`` lines at most."""
        call_id = record.get("call_id")
        if record.get("type") != "tool_result" or not isinstance(call_id, str):
            return
        index = context.index
        for before in range(line - 2, max(-1, line - 2 - CALL_LINES), -1):
            if index.kind(before) == ("tool_call", None):
                call = parse(index.read(before, before + 1)[0])
                if call.get("call_id") == call_id:
                    renderer.shown(call)
                    return
