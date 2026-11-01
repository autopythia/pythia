"""Markdown in assistant messages, styled for a terminal (``display.assistant``).

Three things are styled with ANSI escapes, and everything else shows as
written (doc/interaction-markdown-display-plan.md):

- ATX headings: the ``#`` markers stay, in gray; the text is bold, and also
  underlined for ``#`` and ``##``. A heading on the first line goes below the
  prefix (the label), after a blank line.
- ``**bold**``: the markers are hidden, and the text is bold.
- Pipe tables: their columns are aligned and, given a width, fitted to it by
  wrapping the cells; the header row is bold and the borders are gray.

Code is never styled: neither fenced blocks, fences included, nor inline code
spans, backticks included. The text must already be safe for a terminal
(``_cli_editor.safe_text``): the only escape sequences in the result are those
added here, and every line closes the styles that it opens.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

from ._cli_editor import cell_width


_BOLD, _NOT_BOLD = "\x1b[1m", "\x1b[22m"
_UNDERLINE, _NOT_UNDERLINE = "\x1b[4m", "\x1b[24m"
# The gutter's bright black, then the default color rather than a full reset.
_GRAY, _DEFAULT_COLOR = "\x1b[90m", "\x1b[39m"

# A fence: up to 3 spaces of indentation, then 3 or more backticks or tildes;
# a backtick fence's info string has no backtick. It is closed by a fence of
# the same character, at least as long, with nothing after it but spaces.
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_CLOSING_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")
# An ATX heading: up to 3 spaces, 1 to 6 #, then a space or tab or the end of
# the line, and an optional closing run of # after a space or tab.
_HEADING = re.compile(
    r"^(?P<indent> {0,3})(?P<marks>#{1,6})"
    r"(?:(?P<gap>[ \t]+)(?P<text>.*?))?"
    r"(?:(?P<before>[ \t]+)(?P<closing>#+))?(?P<trail>[ \t]*)$"
)
_DELIMITER_CELL = re.compile(r"^:?-+:?$")
# The fewest columns of text that a fitted table column gets (fewer when its
# widest cell is narrower); a table that can't give every column that many
# shows as written.
_MIN_COLUMN = 3

# A cell's text, one (character, bold) pair per character.
_Chars = List[Tuple[str, bool]]


def style_markdown(text: str, *, prefix: str = "", width: Optional[int] = None) -> str:
    """``text``'s markdown, styled; ``prefix``, such as ``[assistant] ``, goes
    first, unstyled.

    ``width`` is the columns that each line may take, which tables are
    fitted to; None leaves them at their natural widths.

    When the first line is a heading, ``prefix`` (without its trailing space)
    is a line of its own, then a blank line, which sets the heading off as
    the start of a section. When the first line starts a table that is laid
    out, ``prefix`` is a line of its own, so that the table's rows line up.
    """
    lines = text.split("\n")
    out: List[str] = []
    # What follows prefix, without its trailing space, when the first line
    # asks for it: a blank line before a heading, a line break before a
    # table. None keeps prefix as given.
    separator = None
    fence = None  # the open fence's run of backticks or tildes
    index = 0
    while index < len(lines):
        line = lines[index]
        if fence is not None:
            out.append(line)
            closing = _CLOSING_FENCE.match(line)
            if (closing is not None and closing.group(1)[0] == fence[0]
                    and len(closing.group(1)) >= len(fence)):
                fence = None
            index += 1
            continue
        opening = _FENCE.match(line)
        if opening is not None and not (opening.group(1)[0] == "`" and "`" in opening.group(2)):
            fence = opening.group(1)
            out.append(line)
            index += 1
            continue
        table = _table_at(lines, index)
        if table is not None:
            end, header, aligns, body = table
            laid_out = _layout_table(lines[index], header, aligns, body, width)
            if laid_out is None:  # too narrow: as written
                out.extend(_inline(row) for row in lines[index:end])
            else:
                if index == 0:
                    separator = "\n"
                out.extend(laid_out)
            index = end
            continue
        heading = _HEADING.match(line)
        if heading is not None and index == 0:
            separator = "\n\n"
        out.append(_heading(heading) if heading is not None else _inline(line))
        index += 1
    if not prefix:
        return "\n".join(out)
    if separator is None:
        return prefix + "\n".join(out)
    return prefix.rstrip() + separator + "\n".join(out)


# Headings and inline styles.

def _heading(match) -> str:
    text = match.group("text") or ""
    level = len(match.group("marks"))
    on, off = ((_BOLD + _UNDERLINE, _NOT_UNDERLINE + _NOT_BOLD) if level <= 2
               else (_BOLD, _NOT_BOLD))
    parts = [match.group("indent"), _gray(match.group("marks"))]
    if match.group("gap") is not None:
        parts.append(match.group("gap"))
        if text:
            parts.append(on + _render(_inline_runs(text), bold=True) + off)
    if match.group("closing") is not None:
        parts += [match.group("before"), _gray(match.group("closing"))]
    parts.append(match.group("trail"))
    return "".join(parts)


def _gray(text: str) -> str:
    return f"{_GRAY}{text}{_DEFAULT_COLOR}"


def _inline(line: str) -> str:
    """A line with its ``**bold**`` styled, and its code spans as written."""
    return _render(_inline_runs(line))


def _render(runs: Sequence[Tuple[str, bool]], *, bold: bool = False) -> str:
    """Runs of (text, bold) as a line shows them; in a bold context, such as a
    heading, the runs' own bold is already on."""
    if bold:
        return "".join(text for text, _ in runs)
    return "".join(f"{_BOLD}{text}{_NOT_BOLD}" if strong else text for text, strong in runs)


def _inline_runs(line: str) -> List[Tuple[str, bool]]:
    """``line`` as runs of (text, bold), with the paired ``**`` taken out.

    A delimiter is a run of exactly two asterisks, unescaped and outside code
    spans, that can open (followed by a non-space, CommonMark's left-flanking)
    or close (preceded by one, right-flanking). A closer pairs with the
    nearest opener before it, pairs don't nest, and an unpaired delimiter
    shows as written.
    """
    spans = _code_spans(line)
    pairs = []
    opened = None
    for position in _delimiters(line, spans):
        before = line[position - 1] if position else None
        after = line[position + 2] if position + 2 < len(line) else None
        if opened is not None and _left_flanking(after, before):  # right-flanking
            pairs.append((opened, position))
            opened = None
        elif _left_flanking(before, after):
            opened = position  # a later opener: an earlier one shows as written
    runs: List[Tuple[str, bool]] = []
    start = 0
    for opening, closing in pairs:
        if opening > start:
            runs.append((line[start:opening], False))
        runs.append((line[opening + 2:closing], True))
        start = closing + 2
    if start < len(line) or not runs:
        runs.append((line[start:], False))
    return [run for run in runs if run[0]]


def _code_spans(line: str) -> List[Tuple[int, int]]:
    """The code spans of ``line``, as (start, end): a run of backticks, not
    escaped, up to the next run of the same length. A run without one is
    literal."""
    spans = []
    position = 0
    while position < len(line):
        if line[position] != "`" or _escaped(line, position):
            position += 1
            continue
        end = position
        while end < len(line) and line[end] == "`":
            end += 1
        run = end - position
        search = end
        closing = None
        while search < len(line):
            found = line.find("`" * run, search)
            if found < 0:
                break
            after = found + run
            if ((found == 0 or line[found - 1] != "`")
                    and (after >= len(line) or line[after] != "`")):
                closing = found
                break
            while found < len(line) and line[found] == "`":
                found += 1
            search = found
        if closing is None:
            position = end
            continue
        spans.append((position, closing + run))
        position = closing + run
    return spans


def _escaped(line: str, position: int) -> bool:
    """Whether the character at ``position`` follows an odd run of backslashes."""
    count = 0
    while position - count - 1 >= 0 and line[position - count - 1] == "\\":
        count += 1
    return count % 2 == 1


def _delimiters(line: str, spans: Sequence[Tuple[int, int]]):
    """The positions of the ``**`` runs (exactly two asterisks) outside code
    spans, unescaped."""
    position = 0
    while True:
        position = line.find("**", position)
        if position < 0:
            return
        end = position
        while end < len(line) and line[end] == "*":
            end += 1
        if (end - position == 2 and (position == 0 or line[position - 1] != "*")
                and not _escaped(line, position)
                and not any(start <= position < stop for start, stop in spans)):
            yield position
        position = end


def _left_flanking(before: Optional[str], after: Optional[str]) -> bool:
    """CommonMark's left-flanking delimiter run, between ``before`` and ``after``
    (None at the line's ends); swapped, its right-flanking."""
    if after is None or after.isspace():
        return False
    if not _punctuation(after):
        return True
    return before is None or before.isspace() or _punctuation(before)


def _punctuation(char: str) -> bool:
    return unicodedata.category(char)[0] in "PS"


# Tables.

def _cells(line: str) -> Optional[List[str]]:
    """A table row's cells, trimmed, or None when ``line`` has no unescaped
    ``|`` outside code spans. A leading and a trailing ``|`` are borders."""
    text = line.strip()
    spans = _code_spans(text)
    pipes = [position for position, char in enumerate(text)
             if char == "|" and not _escaped(text, position)
             and not any(start <= position < stop for start, stop in spans)]
    if not pipes:
        return None
    if pipes[0] == 0:
        pipes = pipes[1:]
        text_start = 1
    else:
        text_start = 0
    text_end = len(text)
    if pipes and pipes[-1] == len(text) - 1:
        pipes = pipes[:-1]
        text_end -= 1
    bounds = [text_start, *(pipe for pipe in pipes), text_end]
    cells = []
    for index in range(len(bounds) - 1):
        start = bounds[index] + (1 if index else 0)
        cells.append(text[start:bounds[index + 1]].strip())
    return cells


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _table_at(lines: Sequence[str], index: int):
    """The table that starts at ``lines[index]``, as (end, header, aligns,
    body), or None.

    It is a row, then a delimiter row with as many cells, each ``-``, ``:-``,
    ``-:`` or ``:-:`` with any number of dashes, both indented at most 3
    spaces; then the rows up to a blank line, a line without an unescaped
    ``|``, a heading or a fence.
    """
    if index + 1 >= len(lines) or _indent(lines[index]) > 3 or _indent(lines[index + 1]) > 3:
        return None
    header = _cells(lines[index])
    delimiter = _cells(lines[index + 1])
    if (header is None or delimiter is None or len(header) != len(delimiter)
            or not all(_DELIMITER_CELL.match(cell) for cell in delimiter)):
        return None
    aligns = [_alignment(cell) for cell in delimiter]
    body = []
    end = index + 2
    while end < len(lines) and lines[end].strip():
        line = lines[end]
        if _HEADING.match(line) or _FENCE.match(line):
            break
        cells = _cells(line)
        if cells is None:
            break
        body.append(cells)
        end += 1
    return end, header, aligns, body


def _alignment(cell: str) -> Optional[str]:
    left, right = cell.startswith(":"), cell.endswith(":")
    if left and right:
        return "center"
    return "right" if right else "left" if left else None


def _chars(text: str, *, bold: bool = False) -> _Chars:
    """A cell's visible characters, with their bold; a tab counts as a space."""
    return [(char, bold or strong) for part, strong in _inline_runs(text.replace("\t", " "))
            for char in part]


def _width(chars: Sequence[Tuple[str, bool]]) -> int:
    return sum(cell_width(char) for char, _ in chars)


def _words(chars: _Chars) -> List[_Chars]:
    words, word = [], []
    for item in chars:
        if item[0] == " ":
            if word:
                words.append(word)
                word = []
        else:
            word.append(item)
    if word:
        words.append(word)
    return words


def _layout_table(first: str, header: List[str], aligns: List[Optional[str]],
                  body: List[List[str]], width: Optional[int]) -> Optional[List[str]]:
    """The table's lines, aligned and fitted to ``width``; None when it is too
    narrow to fit. A row with more cells than the header adds columns; one
    with fewer gets empty cells."""
    indent = " " * _indent(first)
    columns = max([len(header), *(len(row) for row in body)])
    aligns = aligns + [None] * (columns - len(aligns))
    rows = [[_chars(text, bold=number == 0) for text in row] + [[]] * (columns - len(row))
            for number, row in enumerate([header, *body])]
    natural = [max(1, max(_width(row[column]) for row in rows)) for column in range(columns)]
    if width is None:
        widths = natural
    else:
        longest = [max([_width(word) for row in rows for word in _words(row[column])] or [1])
                   for column in range(columns)]
        widths = _fit(natural, longest, width - len(indent) - (3 * columns + 1))
        if widths is None:
            return None
    lines = []
    for number, row in enumerate(rows):
        wrapped = [_wrap(cell, widths[column]) for column, cell in enumerate(row)]
        for line in range(max(len(cell) for cell in wrapped)):
            parts = [indent, _gray("|")]
            for column, cell in enumerate(wrapped):
                text = cell[line] if line < len(cell) else []
                parts += [" ", _pad(text, widths[column], aligns[column]), " ", _gray("|")]
            lines.append("".join(parts))
        if number == 0:
            lines.append(indent + _gray("|" + "|".join(
                _delimiter(widths[column], aligns[column]) for column in range(columns)) + "|"))
    return lines


def _fit(natural: List[int], longest: List[int], available: int) -> Optional[List[int]]:
    """Column widths that fit ``available``: natural ones if they do. Else
    each column gets its longest word, if all of them fit, or else a floor
    of ``_MIN_COLUMN`` (and long words break); and the columns share the
    rest fairly up to their natural widths, so the widest give up the most.
    None if not even the floor fits."""
    if sum(natural) <= available:
        return list(natural)
    floor = [min(width, _MIN_COLUMN) for width in natural]
    if available < sum(floor):
        return None
    base = [max(low, min(width, word)) for low, width, word in zip(floor, natural, longest)]
    if sum(base) > available:
        base = floor
    shares = _share([width - low for low, width in zip(base, natural)], available - sum(base))
    return [low + share for low, share in zip(base, shares)]


def _share(needs: List[int], amount: int) -> List[int]:
    """``amount`` shared fairly: equal shares, each capped at its need, with
    what a capped one leaves shared again; a remainder goes left to right."""
    given = [0] * len(needs)
    open_ = [index for index, need in enumerate(needs) if need > 0]
    while open_ and amount > 0:
        share = amount // len(open_)
        if share == 0:
            for index in open_[:amount]:
                given[index] += 1
            break
        for index in open_:
            take = min(share, needs[index] - given[index])
            given[index] += take
            amount -= take
        open_ = [index for index in open_ if given[index] < needs[index]]
    return given


def _wrap(chars: _Chars, width: int) -> List[_Chars]:
    """A cell's lines at ``width`` columns: words fill each line, a space
    apart; a word wider than a line is broken."""
    if _width(chars) <= width:
        return [chars]
    lines: List[_Chars] = []
    line: _Chars = []
    used = 0
    for word in _words(chars):
        size = _width(word)
        if line and used + 1 + size <= width:
            line += [(" ", line[-1][1] and word[0][1])] + word
            used += 1 + size
            continue
        if line:
            lines.append(line)
        while size > width:
            head, word = _take(word, width)
            lines.append(head)
            size = _width(word)
        line, used = word, size
    if line or not lines:
        lines.append(line)
    return lines


def _take(word: _Chars, width: int) -> Tuple[_Chars, _Chars]:
    """``word``'s first characters that fit ``width`` (at least one), and the rest."""
    used = 0
    for index, (char, _) in enumerate(word):
        used += cell_width(char)
        if used > width and index:
            return word[:index], word[index:]
    return word, []


def _pad(chars: _Chars, width: int, align: Optional[str]) -> str:
    """A cell line's text, styled, padded to ``width`` as its column aligns."""
    text = _render(_runs(chars))
    space = max(0, width - _width(chars))
    if align == "right":
        return " " * space + text
    if align == "center":
        return " " * (space // 2) + text + " " * (space - space // 2)
    return text + " " * space


def _runs(chars: _Chars) -> List[Tuple[str, bool]]:
    runs: List[Tuple[str, bool]] = []
    for char, strong in chars:
        if runs and runs[-1][1] == strong:
            runs[-1] = (runs[-1][0] + char, strong)
        else:
            runs.append((char, strong))
    return runs


def _delimiter(width: int, align: Optional[str]) -> str:
    """A column's part of the delimiter row: its width and the spaces around
    it in dashes, with the alignment's colons."""
    dashes = width + 2
    if align == "center":
        return ":" + "-" * (dashes - 2) + ":"
    if align == "left":
        return ":" + "-" * (dashes - 1)
    if align == "right":
        return "-" * (dashes - 1) + ":"
    return "-" * dashes


__all__ = ["style_markdown"]
