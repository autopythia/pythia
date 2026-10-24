"""Text from model providers: repaired before it is kept, bounded when quoted,
escaped where Pythia shows it or writes it into its files.

JSON's ``\\u`` escapes can name one half of a UTF-16 surrogate pair alone, and
``json.loads`` keeps such a lone surrogate, but UTF-8 can't encode one, so a
save that holds it fails. A server sends one when it decodes bad bytes with
``surrogateescape``, as Python servers may. The HTTP adapters decode responses
with :func:`loads_json`, and the Claude relay repairs the values of the items
it builds (:func:`repaired_text`, :func:`repaired_json`), so a context never
holds one from a provider. Tool-call arguments that come as JSON text get the
same repair in that text (:func:`repaired_json_text`), so a tool never runs
with one either.

Some characters that model, provider and tool text may hold show as nothing
or act on the terminal (:data:`ESCAPED_CATEGORIES`). The CLI shows them as
escapes (``_cli_editor.safe_text``), and the files Pythia writes hold them as
JSON escapes (:func:`json_text_escaped`), from the same definition.
"""

from __future__ import annotations

import functools
import itertools
import json
import re
from typing import Any
import unicodedata

# A provider's unexpected value, quoted in an error message (quoted): at most
# this many characters of its repr, which shows control and format characters
# as escapes.
QUOTED_VALUE_CHARS = 80

# Shown, and written into files, only as escapes: control characters (on
# screen, all but newline and tab); lone surrogates; format characters, which
# are invisible (zero-width spaces, word joiners, byte order marks, soft
# hyphens, tag characters) or can reorder the rest of a line in a terminal
# that implements right-to-left text (bidi controls); and the line and
# paragraph separators, which show as nothing. The zero-width joiner and
# non-joiner stay: emoji sequences and some scripts need them.
ESCAPED_CATEGORIES = frozenset({"Cc", "Cs", "Cf", "Zl", "Zp"})
KEPT_FORMAT_CHARACTERS = frozenset("\u200c\u200d")

# A \u escape of a surrogate in JSON text, whose backslash isn't itself
# escaped: a high one, with the low one right after it if any; or a low one.
_SURROGATE_ESCAPE = re.compile(
    r"(?<!\\)((?:\\\\)*)\\u[dD][89abAB][0-9a-fA-F]{2}(\\u[dD][c-fC-F][0-9a-fA-F]{2})?"
    r"|(?<!\\)((?:\\\\)*)\\u[dD][c-fC-F][0-9a-fA-F]{2}"
)


def repaired_text(text: str) -> str:
    """``text`` with each lone surrogate replaced by U+FFFD.

    Halves that stand next to each other, as joining the pieces of a stream
    can leave them, are joined into their character first.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    return text


def repaired_json(value: Any) -> Any:
    """A decoded JSON value with every string in it, keys too, repaired
    (:func:`repaired_text`)."""
    if isinstance(value, str):
        return repaired_text(value)
    if isinstance(value, list):
        return [repaired_json(item) for item in value]
    if isinstance(value, dict):
        return {repaired_text(key): repaired_json(item) for key, item in value.items()}
    return value


def loads_json(text: Any) -> Any:
    """``json.loads`` of a provider's text (or UTF-8 bytes), with lone
    surrogates repaired (:func:`repaired_json`).

    Only a ``\\u`` escape can make one, so text without ``\\ud`` decodes as it
    is, with no extra pass.
    """
    value = json.loads(text)
    marks = (b"\\ud", b"\\uD") if isinstance(text, (bytes, bytearray)) else ("\\ud", "\\uD")
    if marks[0] in text or marks[1] in text:
        value = repaired_json(value)
    return value


def repaired_json_text(text: str) -> str:
    """JSON text, such as a tool call's arguments as a provider sends them,
    with each escaped lone surrogate in it, such as ``\\udc80``, written as
    ``\\ufffd``: decoded, it holds U+FFFD there, as :func:`repaired_text`
    makes it. An escaped pair stays, and so does all other text, byte for
    byte. Text that isn't JSON is left as it is.
    """
    if "\\ud" not in text and "\\uD" not in text:
        return text
    try:
        json.loads(text)
    except (ValueError, RecursionError):
        return text
    return _SURROGATE_ESCAPE.sub(_repaired_escape, text)


def _repaired_escape(match: "re.Match[str]") -> str:
    if match[3] is not None:  # a low one alone
        return match[3] + "\\ufffd"
    return match[0] if match[2] is not None else match[1] + "\\ufffd"  # a pair, or a high one alone


def quoted(value: Any, limit: int = QUOTED_VALUE_CHARS) -> str:
    """A provider's unexpected value as an error message quotes it: its repr,
    cut to ``limit`` characters, ending in ``...`` when cut."""
    text = repr(value[: limit + 1] if isinstance(value, str) else value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def escaped_character(char: str) -> bool:
    """Whether ``char`` is shown, and written into files, only as an escape
    (:data:`ESCAPED_CATEGORIES`)."""
    return (char not in KEPT_FORMAT_CHARACTERS
            and unicodedata.category(char) in ESCAPED_CATEGORIES)


@functools.lru_cache(maxsize=None)
def escaped_characters() -> "re.Pattern[str]":
    """A pattern for finding the characters to escape, but newline and tab.

    Up to U+FFFF it matches exactly those (:func:`escaped_character`); beyond,
    it matches every character, for the caller to check with
    :func:`escaped_character`: they are rare in text, and finding the few
    escaped ones among them would take a tenth of a second at the first use.
    Built at the first use, in a few milliseconds.
    """
    codes = (code for code in itertools.compress(range(0x10000), map(
                 ESCAPED_CATEGORIES.__contains__, map(unicodedata.category, map(chr, range(0x10000)))))
             if chr(code) not in KEPT_FORMAT_CHARACTERS and chr(code) not in "\n\t")
    ranges: list[list[int]] = []
    for code in codes:
        if ranges and ranges[-1][1] == code - 1:
            ranges[-1][1] = code
        else:
            ranges.append([code, code])
    members = "".join(re.escape(chr(first)) + ("" if first == last else "-" + re.escape(chr(last)))
                      for first, last in ranges)
    return re.compile(f"[{members}\U00010000-\U0010ffff]")


def json_text_escaped(text: str) -> str:
    """JSON text, as ``json.dumps(..., ensure_ascii=False)`` writes it, with
    each character to escape that it leaves raw written as a ``\\u`` escape
    instead: DEL and C1 controls, format characters, and the line and
    paragraph separators (:func:`escaped_character`). One beyond U+FFFF is
    written as its surrogate pair, as JSON spells it.

    Every JSON reader gets the same value, and ``cat`` or ``tail -f`` of the
    file can't drive or garble a terminal. A lone surrogate stays as it is:
    UTF-8 can't encode one, so a save refuses it, as before (a debug trace
    writes it escaped).
    """
    if text.isascii() and "\x7f" not in text:
        return text
    return escaped_characters().sub(_json_escape, text)


def _json_escape(match: "re.Match[str]") -> str:
    char = match[0]
    code = ord(char)
    if 0xD800 <= code <= 0xDFFF or not escaped_character(char):
        return char
    if code > 0xFFFF:
        code -= 0x10000
        return f"\\u{0xD800 + (code >> 10):04x}\\u{0xDC00 + (code & 0x3FF):04x}"
    return f"\\u{code:04x}"


__all__ = [
    "ESCAPED_CATEGORIES", "KEPT_FORMAT_CHARACTERS", "QUOTED_VALUE_CHARS", "escaped_character",
    "escaped_characters", "json_text_escaped", "loads_json", "quoted", "repaired_json",
    "repaired_json_text", "repaired_text",
]
