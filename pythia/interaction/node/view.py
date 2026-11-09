"""The glui view: the node's root contexts, in glui's widgets, in a browser.

Its two pages, ``/TOKEN/lem/`` for the roots and ``/TOKEN/lem/c/N`` for root
N's log, each load glui's scripts and stylesheets, and the program that shows
them, ``lem/contexts.lem`` compiled, from ``_pythia_ui/`` beside the package
(``Files``), at ``/TOKEN/lem/files/NAME``. Each also names what a page with
tabs needs: the index's facts, where a root's tab finds its log's URLs,
``/TOKEN/lem/tab/N``, and, for a log, the tab that it opens. A page's facts
come as JSON steps (``_pythia_ui/src/lem/host.rs``), from
``/TOKEN/lem/facts/`` and ``/TOKEN/lem/facts/c/N``: for a log, its turns, each
with the line where it starts, and its last rows. A log's rows are its
messages, each with its text, and its runs, the other items between two
messages within a turn, each with what's in it. More rows come as either end
of those on the page comes near the screen, from
``/TOKEN/lem/rows/N/before/LINE`` and ``/TOKEN/lem/rows/N/after/LINE``, so
that a page holds what's been scrolled through, and those around a line that
the page jumps to, from ``/TOKEN/lem/rows/N/around/LINE``; a run's items when
it's opened, from ``/TOKEN/lem/run/N/LINE``; and an item's body when its entry
opens, from ``/TOKEN/lem/body/N/LINE``. A log that grows says so, to a page
that asks every few seconds what it gained since it had LINES lines, from
``/TOKEN/lem/grow/N/GENERATION/LINES``: its length, and the turns that
changed; its later rows then come as others do, the run that it ended with
whole, if it went on. A log's facts hold the lines starred in it, which a page
sets or clears at ``/TOKEN/lem/stars/N``, and the server keeps by the root's
key (``store.py``); and the pins of its page's panel: each line starred, and
the user's messages that a starred answer answers, which come anew from
``/TOKEN/lem/pins/N`` once a star changes. The index's facts and a log's say
what each context is doing, for the dot before it (``activity.py``), and so
does each answer to what a log gained, since a context may change without a
line. After this node's roots, the index's facts hold each remote node's
(``remote.py``): a row for the node, which says how many roots it has, or why
it can't be reached, then its roots, as its own index gives them, by ids of
their own, as ``NAME/ID``, linking through this server. The index's facts give
each of this node's roots its label and tags, as the store keeps them by its
key, which a page sets at ``/TOKEN/lem/roots/N``, as each root's facts say,
and each tag in use, with how many roots have it, by which the index groups
them; a log's title, and its tab's, end with its label. An item's label and
gist are those of the server's plain pages, and its body holds what they show
for it (``pages.Renderer``), as text, cut as they cut it. Every text is as
pythia's terminal shows it, each character to escape as its code point.

The view's pages have a content security policy of their own, which runs
the server's scripts, styles and fetches, and nothing else; its facts,
bodies and files allow nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

from .._cli_editor import safe_text
from ..display import _format_byte_count as byte_count
from .activity import GRAY, activity
from .contexts import Context, parse, shown_path
from .store import Store
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
    "compiled/tabs.json": "web/compiled/tabs.json",
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
_WHOLE = re.compile(r"(?:0|[1-9][0-9]{0,11})\Z")
# About how many rows a stretch of a log holds that the view sends at once:
# it ends where a row ends.
CHUNK_ROWS = 200
# A message shows whole, but for one of more than this many characters, which
# is cut there, and which its line's page shows whole; an item's body is cut
# as the plain pages cut it.
MESSAGE_CUT = 100_000
# A run's summary names this many tools at most.
SUMMARY_TOOLS = 3
# The lines that are rows alone: a user's message, or the assistant's.
_TALK = (("message", "user"), ("message", "assistant"))


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


def item(renderer: Renderer, line: int, run: int, record: dict) -> Optional[dict]:
    """Line ``line``'s Item, in the run that starts at line ``run``: its tone, label and
    gist, and whether it has a body, which it has unless its gist is all of it, on one
    line. None for a line that shows nothing.

    A message has no gist of its own: its gist is its text, on one line, cut.
    """
    shown = renderer.shown(record)
    if shown is None:
        return None
    body = texts(renderer, shown.parts)
    line_gist = gist(record) or (cut(_one_line(body[0][0]), GIST_CHARS) if body else "")
    whole = len(body) == 1 and not body[0][1] and body[0][0].strip() == line_gist
    return {"line": line, "run": run, "tone": TONES.get(shown.css, "Dim"),
            "label": safe_text(shown.label), "gist": safe_text(line_gist),
            "body": bool(body) and not whole}


def spans(context: Context, start: int, forward: bool, most: Optional[int] = None
          ) -> Tuple[int, int, List[Tuple[int, int]]]:
    """Up to about ``most`` of ``context``'s rows, ``CHUNK_ROWS`` by default, from line ``start`` (from 0), forward or
    back, in order, stopping where a row ends: a user's or the assistant's message is a row
    alone, and the other lines that show, from one message to the next within a turn, are
    a run. Each row is its first and last lines, from 0; and the lines that the rows cover,
    from 0, as a range, hold those that show nothing too, and, at an end of the log, the
    lines before its first row or after its last."""
    index = context.index
    most = CHUNK_ROWS if most is None else most
    found: List[Tuple[int, int]] = []
    run: Optional[List[int]] = None  # the run being gathered: its ends, and its turn
    line = start
    while 0 <= line < len(index):
        kind = index.kind(line)
        if not hidden(kind[0]):
            talk = kind in _TALK
            turn = context.turn_of(line)
            if run is not None and (talk or run[2] != turn):
                found.append((min(run[0], run[1]), max(run[0], run[1])))
                run = None
            if run is None:
                if len(found) >= most:
                    break
                if talk:
                    found.append((line, line))
                else:
                    run = [line, line, turn]
            else:
                run[1] = line
        line += 1 if forward else -1
    if run is not None:
        found.append((min(run[0], run[1]), max(run[0], run[1])))
    if forward:
        return start, line, found
    found.reverse()
    return line + 1, start + 1, found


def around(context: Context, at: int) -> Tuple[int, int, List[Tuple[int, int]]]:
    """About ``CHUNK_ROWS`` of ``context``'s rows around line ``at`` (from 0), as ``spans``
    gives them: half of them before the row that holds it, whole, and the rest from that
    row on."""
    back = spans(context, at, False, 1)[2]
    start = back[0][0] if back else at
    low, _, before = spans(context, start - 1, False, CHUNK_ROWS // 2) if start > 0 \
        else (start, start, [])
    _, high, after = spans(context, start, True, CHUNK_ROWS - len(before))
    return low, high, before + after


def summary(context: Context, first: int, last: int) -> str:
    """What the run of lines ``first`` to ``last`` (from 0) of ``context``'s log holds, such
    as ``9 tool calls: exec_command 7, apply_patch 2; 2 reasoning``."""
    index = context.index
    calls: Dict[str, int] = {}
    results = reasoning = other = 0
    for offset, raw in enumerate(index.read(first, last + 1)):
        kind = index.kind(first + offset)[0]
        if hidden(kind):
            continue
        if kind in ("tool_call", "user_tool_call"):
            record = parse(raw)
            call = record.get("call") if kind == "user_tool_call" else record
            name = call.get("name") if isinstance(call, dict) else None
            named = isinstance(name, str) and name.strip()
            name = safe_text(_one_line(name)) if named else "a tool"
            calls[name] = calls.get(name, 0) + 1
        elif kind in ("tool_result", "user_tool_result"):
            results += 1
        elif kind == "reasoning":
            reasoning += 1
        else:
            other += 1
    parts = []
    total = sum(calls.values())
    if total:
        named = sorted(calls.items(), key=lambda pair: (-pair[1], pair[0]))
        tools = ", ".join(f"{name} {count}" for name, count in named[:SUMMARY_TOOLS])
        more = ", \u2026" if len(named) > SUMMARY_TOOLS else ""
        parts.append(f"{total} tool call{'' if total == 1 else 's'}: {tools}{more}")
    elif results:
        parts.append(f"{results} tool result{'' if results == 1 else 's'}")
    if reasoning:
        parts.append(f"{reasoning} reasoning")
    if other:
        parts.append(f"{other} other")
    return "; ".join(parts) or "nothing that shows"


class View:
    """The view's routes, past ``/TOKEN/lem/``, on the server's pages (``pages.Pages``)."""

    def __init__(self, files: Optional[Files] = None, store: Optional[Store] = None) -> None:
        self.files = Files() if files is None else files
        self.store = store

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
        if what == "rows":
            cid, _, rest = name.partition("/")
            way, _, line = rest.partition("/")
            return self.rows(pages, cid, way, line)
        if what == "run":
            cid, _, line = name.partition("/")
            return self.run(pages, cid, line)
        if what == "pins":
            return self.pins(pages, name)
        if what == "tab":
            return self.tab(pages, name)
        if what == "grow":
            cid, _, rest = name.partition("/")
            generation, _, lines = rest.partition("/")
            return self.grow(pages, cid, generation, lines)
        if what == "body":
            cid, _, line = name.partition("/")
            return self.body(pages, cid, line)
        if what == "files":
            return self.files.get(name) or _plain(404, "The glui view has no such file.")
        return pages.not_found("The node server has no such page.")

    # The pages.

    def _page(self, pages: Pages, title: str, facts: str, body: Optional[str] = None, *,
              rows: Optional[str] = None, run: Optional[str] = None,
              grow: Optional[str] = None, stars: Optional[str] = None,
              pins: Optional[str] = None, plain: Optional[str] = None,
              opens: Optional[str] = None) -> Response:
        """A page of the view: the URLs of its facts, its rows before a line, after one and
        around one (``rows``, with ``{way}`` for ``before``, ``after`` or ``around``), its
        runs' items, its bodies, what its log gained since it had ``{lines}`` lines
        (``grow``), where it keeps a star (``stars``) and its panel's pins (``pins``); the
        settings that the server keeps for the view (``store.py``); for its tabs, the URLs
        of the index's facts, of a context's tab (``tab``) and its page, by its number, and
        of the view's first page; the tab that it opens (``opens``), a log's; what it
        loads; and, for a browser without script, a link to the same in the plain pages,
        ``plain``."""
        missing = self.files.missing()
        if missing:
            return pages.not_found(f"The glui view needs files that aren't in "
                                   f"{shown_path(self.files.directory)}/: "
                                   f"{', '.join(missing)}.")
        files = f"{pages.base}/lem/files/"
        lem = f"{h(pages.base)}/lem/"
        data = (f' data-facts="{h(facts)}" data-store="{lem}store/session"'
                f' data-index="{lem}facts/" data-tab="{lem}tab/{{id}}" data-log="{lem}c/{{id}}"'
                f' data-home="{lem}"' + ("" if opens is None else f' data-open="{h(opens)}"'))
        if rows is not None:
            data += "".join(f' data-{way}="{h(rows.replace("{way}", way))}"'
                            for way in ("before", "after", "around"))
        data += ("" if run is None else f' data-run="{h(run)}"') + (
            "" if body is None else f' data-body="{h(body)}"') + (
            "" if grow is None else f' data-grow="{h(grow)}"') + (
            "" if stars is None else f' data-stars="{h(stars)}"') + (
            "" if pins is None else f' data-pins="{h(pins)}"')
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
        urls = self._log_urls(pages, context)
        rows = urls["before"].replace("/before/", "/{way}/")
        return self._page(pages, self._title(context), urls["facts"], urls["body"], rows=rows,
                          run=urls["run"], grow=urls["grow"], stars=urls["stars"],
                          pins=urls["pins"], plain=urls["plain"], opens=context.id)

    @staticmethod
    def _log_urls(pages: Pages, context: Context) -> Dict[str, str]:
        """The URLs of root ``context``'s log, as its page names them: its facts, its rows
        before a line, after one and around one, its runs' items, its bodies, what it gained
        since it had ``{lines}`` lines, its stars, its pins, and its plain page."""
        number = quote(context.id)
        try:
            generation = context.refresh().index.generation
        except OSError:
            generation = 0  # its facts say why; growing, it asks again
        lem = f"{pages.base}/lem"
        return {"facts": f"{lem}/facts/c/{number}",
                **{way: f"{lem}/rows/{number}/{way}/{{line}}"
                   for way in ("before", "after", "around")},
                "run": f"{lem}/run/{number}/{{run}}", "body": f"{lem}/body/{number}/{{line}}",
                "grow": f"{lem}/grow/{number}/{generation}/{{lines}}",
                "stars": f"{lem}/stars/{number}", "pins": f"{lem}/pins/{number}",
                "plain": f"{pages.base}/c/{number}"}

    def tab(self, pages: Pages, cid: str) -> Response:
        """Root ``cid``'s tab, for a page of the view with tabs: its number, its title, and
        the URLs of its log, as its own page names them (``_log_urls``)."""
        context = pages.roots.get(cid)
        if context is None:
            return _plain(404, "There's no such root context here.")
        value = {"id": context.id, "title": safe_text(self._title(context)),
                 **self._log_urls(pages, context)}
        return Response(200, json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                        content_type="application/json", csp=DATA_CSP)

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
        said = self._roots()
        # The tags in use, by name, each with how many of the roots have it.
        tags: Dict[str, int] = {}
        for context in ordered:
            for tag in said.get(context.key, {}).get("tags", []):
                tags[tag] = tags.get(tag, 0) + 1
        steps.extend(["+", "Tag", {"name": tag, "rank": rank, "count": tags[tag]}]
                     for rank, tag in enumerate(sorted(tags)))
        for rank, context in enumerate(ordered):
            problem = problems.get(context.id)
            given = said.get(context.key, {"label": "", "tags": []})
            steps.append(["+", "Context", {
                "id": safe_text(context.label),
                "title": safe_text(f"can't read it: {problem}" if problem
                                   else cut(context.title(), TITLE_CHARS)),
                "model": safe_text(context.model or ""),
                "turns": 0 if problem else len(context.turns()),
                "lines": 0 if problem else len(context),
                "size": "?" if problem else byte_count(context.index.size),
                "changed": "?" if problem else when(context.index.changed),
                "rank": rank, "to": f"{pages.base}/lem/c/{quote(context.id)}",
                "label": safe_text(given["label"]), "tags": " ".join(given["tags"]),
                "keep": "" if self.store is None else
                f"{pages.base}/lem/roots/{quote(context.id)}"}])
            steps.extend(["+", "Tagged", {"id": safe_text(context.label), "tag": tag}]
                         for tag in given["tags"])
            steps.append(self.activity_fact(context, problem))
        rank = len(ordered)
        for listing in pages.listings():
            remote = self.remote_facts(listing, rank)
            rank = 1 + max(step[2]["rank"] for step in remote if "rank" in step[2])
            steps += remote
        return steps

    @staticmethod
    def remote_facts(listing, rank: int) -> list:
        """A remote node's row in the index, at ``rank``, then its roots, after it, as its own
        index gives them (``remote.listings``), by ids of their own, as ``NAME/ID``, and what
        each is doing; what isn't a root of its, linking through it, isn't shown. A root has
        the label that its node gives it, but neither its tags nor a way to keep them, here,
        until where they're kept is settled (decision 11 of the contexts view's plan)."""
        node = listing.remote
        roots, doing = listing.roots(), listing.doing()
        host = (node.hello or {}).get("host")
        title = listing.why or (f"{host if isinstance(host, str) else node.name}: "
                                f"{len(roots)} root context{'' if len(roots) == 1 else 's'}")
        steps = [["+", "Node", {"name": node.name, "rank": rank, "title": safe_text(title),
                                "reached": listing.why is None, "to": f"{node.base}/lem/"}]]
        for fields in roots:
            rank += 1
            mine = {name: value for name, value in fields.items() if name not in ("tags", "keep")}
            steps.append(["+", "Context", {**mine, "id": f"{node.name}/{fields['id']}",
                                           "rank": rank, "node": node.name}])
            if fields["id"] in doing:
                steps.append(["+", "Activity", {**doing[fields["id"]],
                                                "id": f"{node.name}/{fields['id']}"}])
        return steps

    @staticmethod
    def activity_fact(context: Context, problem: Optional[str] = None) -> list:
        """What ``context`` is doing, for the dot before it (``activity``): an Activity, by
        its label; gray, with why, for a context that can't be read."""
        hue, why = (GRAY, f"Unknown: it can't be read: {problem}") if problem else \
            activity(context)
        return ["+", "Activity", {"id": safe_text(context.label), "hue": hue,
                                  "why": safe_text(why)}]

    def log_facts(self, pages: Pages, cid: str) -> Optional[list]:
        """Root ``cid``'s log: its length; its turns (``turn_facts``); and its last rows
        (``chunk``). None if there's no such root here, or its log can't be read."""
        context = pages.roots.get(cid)
        if context is None:
            return None
        try:
            context.refresh()
            number = quote(context.id)
            index = context.index
            with index.lock:
                steps: list = [
                    ["+", "Page", {"title": safe_text(self._title(context)),
                                   "items": f"{pages.base}/c/{number}?line=",
                                   "turns": f"{pages.base}/c/{number}?turn=",
                                   "first": 1, "id": safe_text(context.label)}],
                    self.activity_fact(context),
                    ["+", "Length", {"lines": len(index)}],
                    *(["+", "Stars", {"line": line}] for line in self._stars(context)),
                    *self.pin_facts(pages, context),
                    ["+", "Nav", {"n": 0, "text": "Root contexts", "to": f"{pages.base}/lem/"}],
                    ["+", "Nav", {"n": 1, "text": "Plain pages", "to": f"{pages.base}/c/{number}"}]]
                steps.extend(self.turn_facts(context, 0))
                steps.extend(self.chunk(pages, context, *spans(context, len(index) - 1, False)))
        except OSError:
            return None
        return steps

    def pin_facts(self, pages: Pages, context: Context) -> list:
        """A Pin for each line starred in ``context``'s log, for the page's panel, with its
        turn, tone, label and gist, as an item has them; and for a starred answer, the
        assistant's, one for each of the user's messages before it in its turn, which it
        answers; once for each line, in order. Holds the index's lock."""
        index = context.index
        pinned: Dict[int, int] = {}  # line, from 1: its turn
        for line in self._stars(context):
            if line > len(index):
                continue
            turn = context.turn_of(line - 1)
            pinned[line] = turn
            if index.kind(line - 1) == ("message", "assistant"):
                start = context.turn_range(turn)[0]
                pinned.update((each + 1, turn) for each in range(start, line - 1)
                              if index.kind(each) == ("message", "user"))
        renderer = pages._renderer(context, fold=frozenset())
        steps: list = []
        for line in sorted(pinned):
            fact = item(renderer, line, 0, context.records(line - 1, line)[0][1])
            if fact is not None:
                steps.append(["+", "Pin", {"line": line, "turn": pinned[line],
                                           "tone": fact["tone"], "label": fact["label"],
                                           "gist": fact["gist"]}])
        return steps

    def _stars(self, context: Context) -> List[int]:
        """The lines starred in ``context``'s log, as the store keeps them by its key."""
        return [] if self.store is None else self.store.read("stars").get(context.key, [])

    def _roots(self) -> Dict[str, dict]:
        """Each root's label and tags, as the store keeps them by its key."""
        return {} if self.store is None else self.store.read("roots")

    def _title(self, context: Context) -> str:
        """``context``'s name, as its log's page and tab say it: its label, such as
        ``ui-0 (10)``, and then the label that it's given, if it's given one."""
        given = self._roots().get(context.key, {}).get("label", "")
        return f"{context.label} · {given}" if given else context.label

    def turn_facts(self, context: Context, since: int) -> list:
        """A Turn for each of ``context``'s turns, with how many items it shows and the line
        where it first shows, "The start" for the lines before the first; from the turn
        before the one that line ``since - 1`` (from 0) is in, as the turns that lines
        added from line ``since`` on may change: the last, and those that they start.
        Holds the index's lock."""
        index = context.index

        def shown(start: int, stop: int) -> List[int]:
            return [line for line in range(start, stop) if not hidden(index.kind(line)[0])]

        steps: list = []
        starts = context.turns()
        least = max(0, context.turn_of(since - 1) - 1) if since > 0 else 0
        first = starts[0] if starts else len(index)
        if first > 0 and least == 0:
            lines = shown(0, first)
            steps.append(["+", "Turn", {
                "n": 0, "label": turn_label("The start", len(lines)), "title": "",
                "line": (lines[0] if lines else 0) + 1}])
        for row in context.turn_rows()[max(0, least - 1):]:
            lines = shown(row.start, row.stop)
            steps.append(["+", "Turn", {
                "n": row.number, "label": turn_label(f"Turn {row.number}", len(lines)),
                "title": safe_text(cut(row.label, TITLE_CHARS) or "(no input)"),
                "line": (lines[0] if lines else row.start) + 1}])
        return steps

    def chunk(self, pages: Pages, context: Context, low: int, high: int,
              found: List[Tuple[int, int]]) -> list:
        """The rows ``found`` (``spans``) of ``context``'s log, which cover lines ``low`` to
        ``high - 1``, from 0: a Chunk, those lines from 1; and a Message, with its text,
        whole, up to ``MESSAGE_CUT`` characters, or a Run, with what's in it, for each
        row."""
        renderer = pages._renderer(context, cut=MESSAGE_CUT, fold=frozenset())
        index = context.index
        steps: list = [["+", "Chunk", {"first": low + 1, "last": high}]]
        for first, last in found:
            turn = context.turn_of(first)
            if first == last and index.kind(first) in _TALK:
                record = parse(index.read(first, first + 1)[0])
                shown = renderer.shown(record)
                said = "" if shown is None else "\n\n".join(
                    text for text, _diff in texts(renderer, shown.parts))
                steps.append(["+", "Message", {
                    "line": first + 1, "turn": turn,
                    "tone": "Dim" if shown is None else TONES.get(shown.css, "Dim"),
                    "label": "" if shown is None else safe_text(shown.label),
                    "text": safe_text(said)}])
            else:
                steps.append(["+", "Run", {"line": first + 1, "turn": turn, "last": last + 1,
                                           "text": summary(context, first, last)}])
        return steps

    def rows(self, pages: Pages, cid: str, way: str, text: str) -> Response:
        """Root ``cid``'s rows before line ``text`` (``way`` "before"), after it ("after"),
        or around it ("around"), about ``CHUNK_ROWS`` of them, as ``chunk`` gives them. The
        run that a log ended with, at line ``text``, may go on as the log grows: then the
        rows after it start with that run, whole, and the items of its lines after line
        ``text``, for a page that has it open."""
        context = pages.roots.get(cid)
        if context is None:
            return _plain(404, "There's no such root context here.")
        try:
            context.refresh()
            index = context.index
            with index.lock:
                line = int(text) if _WHOLE.match(text) else -1
                grown = None
                if way == "before" and 1 <= line <= len(index) + 1:
                    found = spans(context, line - 2, False)
                elif way == "around" and 1 <= line <= len(index):
                    found = around(context, line - 1)
                elif way == "after" and 0 <= line <= len(index):
                    back = spans(context, line - 1, False, 1)[2] if line > 0 else []
                    start = back[0][0] if back else line
                    run = back and not (back[0] == (start, start) and index.kind(start) in _TALK)
                    ahead = spans(context, start, True, 1)[2] if run else []
                    grown = (start, ahead[0][1]) if ahead and ahead[0][1] >= line else None
                    found = spans(context, start if grown else line, True)
                else:
                    return _plain(404, f"{context.label} has no rows {way[:12]} line {text[:24]}.")
                steps = self.chunk(pages, context, *found)
                if grown is not None:
                    records = context.records(line, grown[1] + 1)
        except OSError as exc:
            return _plain(404, f"{context.label}'s log can't be read: {exc}")
        if grown is not None:
            renderer = pages._renderer(context, fold=frozenset())
            for number, record in records:
                fact = item(renderer, number, grown[0] + 1, record)
                if fact is not None:
                    steps.append(["+", "Item", fact])
        return _steps(steps)

    def pins(self, pages: Pages, cid: str) -> Response:
        """Root ``cid``'s pins, for its page's panel, as ``pin_facts`` gives them, which a
        page asks for anew once a star changes."""
        context = pages.roots.get(cid)
        if context is None:
            return _plain(404, "There's no such root context here.")
        try:
            context.refresh()
            with context.index.lock:
                return _steps(self.pin_facts(pages, context))
        except OSError as exc:
            return _plain(404, f"{context.label}'s log can't be read: {exc}")

    def grow(self, pages: Pages, cid: str, generation: str, text: str) -> Response:
        """What root ``cid``'s log gained since it had ``text`` lines, for a page from when
        its index was of generation ``generation``: what its context is doing, as it may
        change without a line, in place of what it was doing; and, if it gained lines, its
        length, in place of the last, and the turns that changed (``turn_facts``). A log
        that was indexed anew since, as one replaced or cut short is, can't answer: the
        page must load anew."""
        context = pages.roots.get(cid)
        if context is None:
            return _plain(404, "There's no such root context here.")
        try:
            context.refresh()
            index = context.index
            with index.lock:
                if not (_WHOLE.match(generation) and _WHOLE.match(text)):
                    return _plain(404, f"{context.label} has no length {text[:24]}.")
                lines = int(text)
                if int(generation) != index.generation or lines > len(index):
                    return _plain(409, f"{context.label}'s log was replaced: load its page anew.")
                steps: list = [self.activity_fact(context)]
                if lines < len(index):
                    steps += [["-", "Length", {"lines": lines}],
                              ["+", "Length", {"lines": len(index)}]]
                    steps.extend(self.turn_facts(context, lines))
        except OSError as exc:
            return _plain(404, f"{context.label}'s log can't be read: {exc}")
        return _steps(steps)

    def run(self, pages: Pages, cid: str, text: str) -> Response:
        """The run of root ``cid``'s log that starts at line ``text``: that its items came,
        and an Item for each of its lines that shows."""
        context = pages.roots.get(cid)
        if context is None:
            return _plain(404, "There's no such root context here.")
        try:
            context.refresh()
            index = context.index
            with index.lock:
                line = int(text) if _LINE.match(text) else 0
                start = line - 1
                back = spans(context, start, False, 1)[2] if 1 <= line <= len(index) else []
                kind = index.kind(start) if back else None
                if not back or back[0][0] != start or kind in _TALK or hidden(kind[0]):
                    return _plain(404, f"{context.label} has no run at line {text[:24]}.")
                first, last = spans(context, start, True, 1)[2][0]
                records = context.records(first, last + 1)
        except OSError as exc:
            return _plain(404, f"{context.label}'s log can't be read: {exc}")
        renderer = pages._renderer(context, fold=frozenset())
        steps: list = [["+", "RunLoaded", {"run": line}]]
        for number, record in records:
            fact = item(renderer, number, line, record)
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
