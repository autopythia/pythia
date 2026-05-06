from __future__ import annotations

import contextlib
import io
import os
import select
import sys
import termios
import tty
from asyncio import AbstractEventLoop, get_running_loop
from codecs import getincrementaldecoder
from dataclasses import dataclass
from enum import Enum
from typing import Callable, ContextManager, Generator, TextIO

__all__ = [
    "KeyPress",
    "Keys",
    "TerminalInput",
    "create_input",
]


class Keys(str, Enum):
    Escape = "escape"

    ControlA = "c-a"
    ControlC = "c-c"
    ControlD = "c-d"
    ControlE = "c-e"
    ControlH = "c-h"
    ControlJ = "c-j"
    ControlK = "c-k"
    ControlM = "c-m"
    ControlU = "c-u"
    ControlW = "c-w"

    Left = "left"
    Right = "right"
    Up = "up"
    Down = "down"
    WordLeft = "word-left"
    WordRight = "word-right"

    BracketedPaste = "<bracketed-paste>"

    Enter = ControlM
    Backspace = ControlH


@dataclass
class KeyPress:
    key: str | Keys
    data: str | None = None

    def __post_init__(self) -> None:
        if self.data is None:
            self.data = self.key.value if isinstance(self.key, Keys) else self.key


_ESCAPE_SEQUENCES: dict[str, str | Keys | tuple[Keys, ...]] = {
    "\x01": Keys.ControlA,
    "\x03": Keys.ControlC,
    "\x04": Keys.ControlD,
    "\x05": Keys.ControlE,
    "\x08": Keys.ControlH,
    "\x0a": Keys.ControlJ,
    "\x0b": Keys.ControlK,
    "\x0d": Keys.ControlM,
    "\x15": Keys.ControlU,
    "\x17": Keys.ControlW,
    "\x1b": Keys.Escape,
    "\x7f": Keys.ControlH,
    "\x1b[A": Keys.Up,
    "\x1b[B": Keys.Down,
    "\x1b[C": Keys.Right,
    "\x1b[D": Keys.Left,
    "\x1bOA": Keys.Up,
    "\x1bOB": Keys.Down,
    "\x1bOC": Keys.Right,
    "\x1bOD": Keys.Left,
    "\x1bb": Keys.WordLeft,
    "\x1bf": Keys.WordRight,
    "\x1b[1;3C": Keys.WordRight,
    "\x1b[1;3D": Keys.WordLeft,
    "\x1b[1;5C": Keys.WordRight,
    "\x1b[1;5D": Keys.WordLeft,
    "\x1b[1;9C": Keys.WordRight,
    "\x1b[1;9D": Keys.WordLeft,
    "\x1b[5C": Keys.WordRight,
    "\x1b[5D": Keys.WordLeft,
    "\x1bOc": Keys.WordRight,
    "\x1bOd": Keys.WordLeft,
    "\x1b[200~": Keys.BracketedPaste,
}
_BRACKETED_PASTE_END = "\x1b[201~"
_ESCAPE_PREFIXES = {
    seq[:idx]
    for seq in tuple(_ESCAPE_SEQUENCES) + (_BRACKETED_PASTE_END,)
    for idx in range(1, len(seq))
}


class _InputParser:
    def __init__(self, callback: Callable[[KeyPress], None]) -> None:
        self._callback = callback
        self._pending = ""
        self._in_bracketed_paste = False
        self._paste_buffer = ""

    def feed(self, data: str) -> None:
        if not data:
            return
        if self._in_bracketed_paste:
            self._feed_bracketed_paste(data)
            return

        self._pending += data
        self._drain_pending(flush=False)

    def flush(self) -> None:
        if self._in_bracketed_paste:
            self._emit(KeyPress(Keys.BracketedPaste, self._paste_buffer))
            self._in_bracketed_paste = False
            self._paste_buffer = ""
        self._drain_pending(flush=True)

    def _feed_bracketed_paste(self, data: str) -> None:
        self._paste_buffer += data
        end_idx = self._paste_buffer.find(_BRACKETED_PASTE_END)
        if end_idx < 0:
            return

        self._emit(KeyPress(Keys.BracketedPaste, self._paste_buffer[:end_idx]))
        remaining = self._paste_buffer[end_idx + len(_BRACKETED_PASTE_END) :]
        self._in_bracketed_paste = False
        self._paste_buffer = ""
        if remaining:
            self.feed(remaining)

    def _drain_pending(self, flush: bool) -> None:
        while self._pending:
            match = _ESCAPE_SEQUENCES.get(self._pending)
            if match is not None and (flush or self._pending not in _ESCAPE_PREFIXES):
                self._dispatch(match, self._pending)
                self._pending = ""
                continue

            if not flush and self._pending in _ESCAPE_PREFIXES:
                return

            matched = False
            for idx in range(len(self._pending), 0, -1):
                match = _ESCAPE_SEQUENCES.get(self._pending[:idx])
                if match is None:
                    continue
                self._dispatch(match, self._pending[:idx])
                self._pending = self._pending[idx:]
                if self._in_bracketed_paste:
                    remaining = self._pending
                    self._pending = ""
                    self._feed_bracketed_paste(remaining)
                matched = True
                break
            if matched:
                continue

            self._emit(KeyPress(self._pending[0], self._pending[0]))
            self._pending = self._pending[1:]

    def _dispatch(self, key: str | Keys | tuple[Keys, ...], data: str) -> None:
        if isinstance(key, tuple):
            for idx, subkey in enumerate(key):
                payload = data if idx == 0 else ""
                self._emit(KeyPress(subkey, payload))
            return

        if key == Keys.BracketedPaste:
            self._in_bracketed_paste = True
            self._paste_buffer = ""
            return

        self._emit(KeyPress(key, data))

    def _emit(self, key_press: KeyPress) -> None:
        self._callback(key_press)


class _StdinReader:
    def __init__(self, fd: int, encoding: str = "utf-8", errors: str = "surrogateescape") -> None:
        decoder_cls = getincrementaldecoder(encoding)
        self._decoder = decoder_cls(errors=errors)
        self._fd = fd
        self.closed = False

    def read(self, count: int = 1024) -> str:
        if self.closed:
            return ""

        try:
            if not select.select([self._fd], [], [], 0)[0]:
                return ""
        except OSError:
            self.closed = True
            return ""

        try:
            data = os.read(self._fd, count)
        except OSError:
            return ""

        if data == b"":
            self.closed = True
            return ""

        return self._decoder.decode(data)


class TerminalInput:
    _fds_not_a_terminal: set[int] = set()

    def __init__(self, stdin: TextIO) -> None:
        try:
            fd = stdin.fileno()
        except io.UnsupportedOperation as exc:
            raise io.UnsupportedOperation("Stdin is not a terminal.") from exc

        if not stdin.isatty() and fd not in self._fds_not_a_terminal:
            sys.stderr.write(f"Warning: Input is not a terminal (fd={fd!r}).\n")
            sys.stderr.flush()
            self._fds_not_a_terminal.add(fd)

        self.stdin = stdin
        self._fileno = fd
        self._buffer: list[KeyPress] = []
        self._reader = _StdinReader(fd, encoding=stdin.encoding or "utf-8")
        # Keep appending to the current buffer even after `read_keys` swaps in a
        # fresh list for the next batch.
        self._parser = _InputParser(lambda key_press: self._buffer.append(key_press))

    def read_keys(self) -> list[KeyPress]:
        self._parser.feed(self._reader.read())
        keys = self._buffer
        self._buffer = []
        return keys

    def flush_keys(self) -> list[KeyPress]:
        self._parser.flush()
        keys = self._buffer
        self._buffer = []
        return keys

    @property
    def closed(self) -> bool:
        return self._reader.closed

    def raw_mode(self) -> ContextManager[None]:
        return raw_mode(self._fileno)

    def cooked_mode(self) -> ContextManager[None]:
        return cooked_mode(self._fileno)

    def fileno(self) -> int:
        return self._fileno

    def attach(self, callback: Callable[[], None]) -> ContextManager[None]:
        return _attached_input(self, callback)

    def detach(self) -> ContextManager[None]:
        return _detached_input(self)


def create_input(stdin: TextIO | None = None, always_prefer_tty: bool = False) -> TerminalInput:
    if sys.platform == "win32":
        raise NotImplementedError("Terminal input is only implemented for POSIX terminals.")

    if stdin is None:
        stdin = sys.stdin
        if always_prefer_tty:
            for obj in (sys.stdin, sys.stdout, sys.stderr):
                if obj is not None and obj.isatty():
                    stdin = obj
                    break

    if stdin is None:
        raise io.UnsupportedOperation("stdin is unavailable")

    return TerminalInput(stdin)


_current_callbacks: dict[tuple[AbstractEventLoop, int], Callable[[], None] | None] = {}


@contextlib.contextmanager
def _attached_input(
    input: TerminalInput, callback: Callable[[], None]
) -> Generator[None, None, None]:
    loop = get_running_loop()
    fd = input.fileno()
    previous = _current_callbacks.get((loop, fd))

    def callback_wrapper() -> None:
        if input.closed:
            loop.remove_reader(fd)
        callback()

    try:
        loop.add_reader(fd, callback_wrapper)
    except PermissionError as exc:
        raise EOFError from exc

    _current_callbacks[loop, fd] = callback
    try:
        yield
    finally:
        loop.remove_reader(fd)
        if previous is not None:
            loop.add_reader(fd, previous)
            _current_callbacks[loop, fd] = previous
        else:
            _current_callbacks.pop((loop, fd), None)


@contextlib.contextmanager
def _detached_input(input: TerminalInput) -> Generator[None, None, None]:
    loop = get_running_loop()
    fd = input.fileno()
    previous = _current_callbacks.get((loop, fd))

    if previous is not None:
        loop.remove_reader(fd)
        _current_callbacks[loop, fd] = None

    try:
        yield
    finally:
        if previous is not None:
            loop.add_reader(fd, previous)
            _current_callbacks[loop, fd] = previous


class raw_mode:
    def __init__(self, fileno: int) -> None:
        self.fileno = fileno
        try:
            self._attrs_before = termios.tcgetattr(fileno)
        except termios.error:
            self._attrs_before = None

    def __enter__(self) -> None:
        try:
            newattr = termios.tcgetattr(self.fileno)
        except termios.error:
            return

        newattr[tty.LFLAG] = self._patch_lflag(newattr[tty.LFLAG])
        newattr[tty.IFLAG] = self._patch_iflag(newattr[tty.IFLAG])

        # Some platforms default VMIN to values >1 in non-canonical mode.
        newattr[tty.CC][termios.VMIN] = 1

        termios.tcsetattr(self.fileno, termios.TCSANOW, newattr)

    @classmethod
    def _patch_lflag(cls, attrs: int) -> int:
        return attrs & ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)

    @classmethod
    def _patch_iflag(cls, attrs: int) -> int:
        return attrs & ~(
            termios.IXON
            | termios.IXOFF
            | termios.ICRNL
            | termios.INLCR
            | termios.IGNCR
        )

    def __exit__(self, *args: object) -> None:
        if self._attrs_before is None:
            return

        try:
            termios.tcsetattr(self.fileno, termios.TCSANOW, self._attrs_before)
        except termios.error:
            pass


class cooked_mode(raw_mode):
    @classmethod
    def _patch_lflag(cls, attrs: int) -> int:
        return attrs | (termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)

    @classmethod
    def _patch_iflag(cls, attrs: int) -> int:
        return attrs | termios.ICRNL
