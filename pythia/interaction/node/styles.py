"""The node server's styles: how pages look, picked from the menu at each
page's top right.

A style is a stylesheet and a layout. Its stylesheet is ``css/NAME.css``,
under ``css/base.css``, which styles the menu itself. The server reads both
for each page, so an edit shows at the next reload, with no restart. Its layout is
what ``Style`` says: for now, which item types fold into one line each.

Three styles are built in: ``log-dark`` and ``log-light``, every item as the
log has it, in fixed dark or light colors; and ``chat``, the conversation
first. Log light inherits Log dark's stylesheet and overrides its colors.
Any other ``css/NAME.css`` is a style too, which the menu lists by its name:
one from scratch, with base.css alone under it; or, named ``log-dark-NAME``,
``log-light-NAME`` or ``chat-NAME``, a variant of a built-in style, with its
layout, and all its stylesheets under the variant's own.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, List, Optional, Sequence, Tuple

CSS_DIRECTORY = Path(__file__).resolve().parent / "css"
BASE = "base"  # css/base.css, under every style's own stylesheet
DEFAULT_STYLE = "log-dark"
FIELD = "style"  # a URL's query field that names its style: ?style=chat
# A style's name, in URLs and in file names (css/NAME.css).
NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}\Z")
MAX_CSS_BYTES = 256 * 1024
# What would end the page's <style> element, and so let a stylesheet write HTML.
_END_STYLE = re.compile(r"</style", re.IGNORECASE)


@dataclass(frozen=True)
class Style:
    """How pages look: a stylesheet, ``css/NAME.css``, and a layout.

    ``fold``: the item types that show as one line each, which opens in
    place, such as ``tool_call``. ``parent``: the built-in style that a
    variant builds on, whose layout it has and whose stylesheet comes under
    its own.
    """

    name: str
    label: str
    note: str = ""
    fold: FrozenSet[str] = frozenset()
    parent: Optional[str] = None


BUILT_IN: Tuple[Style, ...] = (
    Style("log-dark", "Log dark", "every item, as the log has it; dark colors"),
    Style("log-light", "Log light", "every item, as the log has it; light colors",
          parent="log-dark"),
    Style("chat", "Chat", "the conversation first; tool calls, results, reasoning and "
          "checkpoints folded",
          fold=frozenset(("tool_call", "tool_result", "reasoning", "context_prefix",
                          "context_compaction"))),
)


def _regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(os.stat(path).st_mode)
    except (OSError, ValueError):
        return False


class Styles:
    """The styles that a directory of stylesheets holds now: listed again each time."""

    def __init__(self, directory=CSS_DIRECTORY, built_in: Sequence[Style] = BUILT_IN) -> None:
        self.directory = Path(directory)
        self.built_in = tuple(built_in)

    def _files(self) -> Tuple[List[str], List[str]]:
        """The styles' names that css/ has files for, and its .css files that aren't
        styles by their names. Hidden files, such as an editor's, are neither."""
        try:
            names = sorted(os.listdir(self.directory))
        except OSError:
            return [], []
        styles, others = [], []
        for name in names:
            if not name.endswith(".css") or name.startswith("."):
                continue
            stem = name[:-len(".css")]
            if stem == BASE or not _regular(self.directory / name):
                continue
            if NAME.match(stem):
                styles.append(stem)
            else:
                others.append(name)
        return styles, others

    def all(self) -> List[Style]:
        """Each style, as the menu lists them: each built-in one, then its variants;
        then the others."""
        names, _ = self._files()
        styles: List[Style] = []
        for built in self.built_in:
            styles.append(built)
            prefix = built.name + "-"
            styles.extend(Style(name, name, f"{built.label}, with css/{name}.css",
                                fold=built.fold, parent=built.name)
                          for name in names
                          if name.startswith(prefix) and len(name) > len(prefix))
        listed = {style.name for style in styles}
        styles.extend(Style(name, name, f"css/{name}.css") for name in names
                      if name not in listed)
        return styles

    def ignored(self) -> List[str]:
        """The .css files in css/ that aren't styles, by their names."""
        return self._files()[1]

    def get(self, name: str) -> Optional[Style]:
        return next((style for style in self.all() if style.name == name), None)

    def css(self, name: str) -> Tuple[str, List[str]]:
        """Style ``name``'s stylesheet: base.css, its ancestors', then its own; and
        what kept a file out of it, one sentence each."""
        available = {style.name: style for style in self.all()}
        style = available.get(name)
        chain: List[str] = []
        texts, problems = [], []
        while style is not None:
            if style.name in chain:
                problems.append(f"Style {name!r} has a parent cycle at {style.name!r}.")
                break
            chain.append(style.name)
            style = available.get(style.parent) if style.parent else None
        for each in [BASE, *reversed(chain)]:
            text, problem = self._read(each)
            if problem is not None:
                problems.append(problem)
            if text:
                texts.append(text)
        return "\n".join(texts), problems

    def _read(self, name: str) -> Tuple[str, Optional[str]]:
        path = self.directory / f"{name}.css"
        if not os.path.lexists(path):
            return "", f"{path} is missing."
        if not _regular(path):
            return "", f"{path} isn't used: it isn't a file."
        try:
            with open(path, "rb") as file:
                data = file.read(MAX_CSS_BYTES + 1)
        except OSError as exc:
            return "", f"{path} can't be read: {exc.strerror or exc}."
        if len(data) > MAX_CSS_BYTES:
            return "", f"{path} isn't used: it's over {MAX_CSS_BYTES // 1024} KiB."
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return "", f"{path} isn't used: it isn't UTF-8."
        if _END_STYLE.search(text):
            return "", (f"{path} isn't used: it holds </style, which would end the page's "
                        "stylesheet.")
        return text, None
