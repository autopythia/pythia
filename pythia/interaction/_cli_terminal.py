"""The POSIX CLI's sole terminal writer; model/display code never owns the TTY."""

from __future__ import annotations

from collections import deque
from contextlib import ExitStack
import os
from typing import Iterable
from typing import TextIO

from ._cli_editor import Editor
from ._cli_editor import cell_width
from ._cli_editor import layout_editor
from ._cli_editor import safe_text
from .display import DisplayItem


class PosixTerminal:
    def __init__(self, stdin: TextIO, stdout: TextIO) -> None:
        self.stdin = stdin
        self.stdout = stdout
        self._keys = deque()
        self._stack = ExitStack()
        self._input = None
        self._read_error = None
        self._cursor_row = 0
        self._drawn = False
        self._last_frame = None

    def __enter__(self) -> "PosixTerminal":
        # Keep CLI imports and --help usable without termios/non-POSIX support.
        from pythia.term_input import create_input

        try:
            self._input = create_input(self.stdin)
            self._stack.enter_context(self._input.raw_mode())
            self._stack.enter_context(self._input.attach(self._read))
            self.stdout.write("\x1b[?2004h")
            self.stdout.flush()
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    def _read(self) -> None:
        if self._read_error is None:
            try:
                self._keys.extend(self._input.read_keys())
            except Exception as exc:
                # Reader callbacks cannot unwind the controller. Hand the error
                # to the next frame instead of logging repeatedly in raw mode.
                self._read_error = exc

    def read_keys(self):
        if self._read_error is not None:
            raise self._read_error
        keys = tuple(self._keys)
        self._keys.clear()
        return keys

    @property
    def closed(self) -> bool:
        return self._input is not None and self._input.closed

    def _clear(self) -> str:
        if not self._drawn:
            return ""
        up = f"\x1b[{self._cursor_row}A" if self._cursor_row else ""
        return f"\r{up}\x1b[J"

    def render(
        self,
        editor: Editor,
        status: str,
        items: Iterable[DisplayItem],
        prompt: str = ":> ",
    ) -> None:
        items = tuple(items)
        size = os.get_terminal_size(self.stdout.fileno())
        columns, rows = max(1, size.columns), max(1, size.lines)
        frame_key = (editor, status, prompt, columns, rows)
        if not items and self._last_frame == frame_key:
            return
        layout = layout_editor(editor, columns, prompt)
        height = max(1, rows - 1)
        start = max(0, layout.cursor_row - height + 1)
        lines = list(layout.lines[start:start + height])
        cursor_row = layout.cursor_row - start
        if rows > 1:
            clipped, width = "", 0
            for char in safe_text(status).replace("\n", " ").replace("\t", " "):
                width += cell_width(char)
                if width > columns - 1:
                    break
                clipped += char
            lines.insert(0, clipped)
            cursor_row += 1
        parts = [self._clear()]
        for item in items:
            # Sanitize raw content BEFORE __str__ adds trusted gutter/diff ANSI.
            rendered = str(DisplayItem(safe_text(item.text), is_diff=item.is_diff))
            parts.extend((rendered.replace("\n", "\r\n"), "\r\n"))
        parts.append("\r\n".join(lines))
        parts.append("\r")
        up = len(lines) - 1 - cursor_row
        if up:
            parts.append(f"\x1b[{up}A")
        if layout.cursor_column:
            parts.append(f"\x1b[{layout.cursor_column}C")
        self.stdout.write("".join(parts))
        self.stdout.flush()
        self._cursor_row = cursor_row
        self._drawn = True
        self._last_frame = frame_key

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            try:
                self.stdout.write(self._clear() + "\x1b[0m\x1b[?2004l\r\n")
                self.stdout.flush()
            finally:
                # Detach and restore cooked mode even if stdout is broken.
                self._stack.close()
        except Exception:
            # A cleanup failure must not hide the original startup/I/O error.
            if exc is None:
                raise
