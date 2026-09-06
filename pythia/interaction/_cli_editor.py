"""Value-based editor transitions and terminal-cell layout (no terminal I/O)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import unicodedata


def safe_text(text: str) -> str:
    """Make content controls visible before adding application ANSI decoration."""
    return "".join(
        char if char in "\n\t" or unicodedata.category(char) not in {"Cc", "Cs"}
        else f"\\u{ord(char):04x}"
        for char in text
    )


def cell_width(char: str) -> int:
    # Codepoint layout, not a full grapheme/emoji-cluster implementation.
    if unicodedata.category(char) in {"Mn", "Me", "Cf"}:
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


@dataclass(frozen=True)
class Editor:
    text: str = ""
    cursor: int = 0

    def edit(self, key: str, data: str = "") -> "Editor":
        text, pos = self.text, self.cursor
        if key in {"left", "right", "c-a", "c-e"}:
            pos = {
                "left": max(0, pos - 1),
                "right": min(len(text), pos + 1),
                "c-a": 0,
                "c-e": len(text),
            }[key]
        elif key in {"word-left", "c-w"}:
            start = pos
            while pos > 0 and text[pos - 1].isspace():
                pos -= 1
            while pos > 0 and not text[pos - 1].isspace():
                pos -= 1
            if key == "c-w":
                text = text[:pos] + text[start:]
        elif key == "word-right":
            while pos < len(text) and text[pos].isspace():
                pos += 1
            while pos < len(text) and not text[pos].isspace():
                pos += 1
        elif key == "c-h":
            text = text[:max(0, pos - 1)] + text[pos:]
            pos = max(0, pos - 1)
        elif key == "c-u":
            text, pos = text[pos:], 0
        elif key == "c-k":
            text = text[:pos]
        elif key == "c-j" or key == "<bracketed-paste>" or len(key) == 1:
            inserted = "\n" if key == "c-j" else data
            inserted = inserted.replace("\r\n", "\n").replace("\r", "\n")
            text = text[:pos] + inserted + text[pos:]
            pos += len(inserted)
        return Editor(text, pos)


@dataclass(frozen=True)
class Layout:
    lines: Tuple[str, ...]
    cursor_row: int
    cursor_column: int


def layout_editor(editor: Editor, columns: int, prompt: str = ":> ") -> Layout:
    """Reserve the last terminal column to avoid delayed automatic wrapping."""
    limit = max(1, columns - 1)
    prefix = prompt[:max(0, limit - 1)]
    continuation = " > "[:len(prefix)]
    lines = [prefix]
    row, column = 0, len(prefix)
    cursor_row, cursor_column = row, column
    for index, char in enumerate(editor.text):
        if char == "\n":
            if index == editor.cursor:
                cursor_row, cursor_column = row, column
            lines.append(continuation)
            row, column = row + 1, len(continuation)
            continue
        visible = " " * (4 - column % 4) if char == "\t" else safe_text(char)
        for offset, glyph in enumerate(visible):
            width = cell_width(glyph)
            if width > limit - len(prefix):
                glyph, width = "?", 1
            if column + width > limit:
                lines.append(continuation)
                row, column = row + 1, len(continuation)
            if index == editor.cursor and offset == 0:
                cursor_row, cursor_column = row, column
            lines[row] += glyph
            column += width
    if editor.cursor == len(editor.text):
        cursor_row, cursor_column = row, column
    return Layout(tuple(lines), cursor_row, min(cursor_column, max(0, columns - 1)))
