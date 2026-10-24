"""Text from model providers: repaired before it is kept, bounded when quoted.

JSON's ``\\u`` escapes can name one half of a UTF-16 surrogate pair alone, and
``json.loads`` keeps such a lone surrogate, but UTF-8 can't encode one, so a
save that holds it fails. A server sends one when it decodes bad bytes with
``surrogateescape``, as Python servers may. The HTTP adapters decode responses
with :func:`loads_json`, and the Claude relay repairs the values of the items
it builds (:func:`repaired_text`, :func:`repaired_json`), so a context never
holds one from a provider.
"""

from __future__ import annotations

import json
from typing import Any

# A provider's unexpected value, quoted in an error message (quoted): at most
# this many characters of its repr, which shows control and format characters
# as escapes.
QUOTED_VALUE_CHARS = 80


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


def quoted(value: Any, limit: int = QUOTED_VALUE_CHARS) -> str:
    """A provider's unexpected value as an error message quotes it: its repr,
    cut to ``limit`` characters, ending in ``...`` when cut."""
    text = repr(value[: limit + 1] if isinstance(value, str) else value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


__all__ = ["QUOTED_VALUE_CHARS", "loads_json", "quoted", "repaired_json", "repaired_text"]
