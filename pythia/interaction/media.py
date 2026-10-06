"""Experimental expansion of leading ``@path-or-uri`` user-prompt components.

This module is pure and provider-neutral. It turns the leading whitespace
separated ``@`` references of a user prompt into ordered
``pythia.interaction.items`` content parts and returns a normal
``Message(role="user", ...)``. It performs host file reads at submission time.
Nothing here talks to a model.

- A local file with a text suffix (``DEFAULT_TEXT_SUFFIXES``: ``.md`` and
  ``.txt``, any letter case) is *pasted*: its strict UTF-8 text, with
  ``rstrip()`` applied, becomes text. The suffix is taken from the file
  actually read, after ``~`` expansion and symlinks.
- Any other local file must be an image and is inlined as a ``data:`` URL;
  remote ``http(s)`` images are forwarded by URL.

Neighbouring text (pasted files and the trailing typed text) is joined with
``TEXT_PASTE_SEPARATOR``, a blank line, and order is kept around images. A
prompt without images therefore yields a plain-string message, exactly as if
the text had been typed. Pasted text is never expanded again.

Images are the only non-text item because :class:`MediaPart` is currently
serialized as the Responses ``input_image`` content item (see the ``TODO`` on
that type); any other reference is rejected rather than mislabeled.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import mimetypes
from pathlib import Path
from pathlib import PurePath
from pathlib import PurePosixPath
import re
import stat
from typing import FrozenSet
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Tuple
from urllib.parse import urlparse

from .items import ContentPart
from .items import MediaPart
from .items import Message
from .items import TextPart


class AttachmentError(ValueError):
    """A leading ``@`` reference could not be turned into a content part."""


DEFAULT_MAX_CONTENT_ITEMS = 8
DEFAULT_MAX_ITEM_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 40 * 1024 * 1024
DEFAULT_MAX_TEXT_BYTES = 1024 * 1024

# Local files with these suffixes are pasted as text. They are compared
# case-insensitively with the suffix of the file actually read.
DEFAULT_TEXT_SUFFIXES = frozenset({".md", ".txt"})
# Joins neighbouring text: pasted files and the trailing typed text.
TEXT_PASTE_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class ContentLimits:
    max_items: int = DEFAULT_MAX_CONTENT_ITEMS
    max_item_bytes: int = DEFAULT_MAX_ITEM_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    # Per pasted text file; max_item_bytes and max_total_bytes apply too.
    max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES


DEFAULT_CONTENT_LIMITS = ContentLimits()


_SCHEME_RE = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*)://")
_BARE_SCHEME_RE = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*):")
_REJECTED_SCHEMES = frozenset({"data", "file"})


def _sniff_image_media_type(data: bytes) -> Optional[str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _image_media_type(name: str, data: bytes) -> Optional[str]:
    guessed, _ = mimetypes.guess_type(name)
    if guessed is not None and guessed.startswith("image/"):
        return guessed
    return _sniff_image_media_type(data)


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _text_suffix_set(text_suffixes: Iterable[str]) -> FrozenSet[str]:
    """Validate suffixes such as ``.txt`` and lowercase them."""
    if isinstance(text_suffixes, str):
        raise TypeError("text_suffixes must be a collection of suffixes, not a string")
    suffixes = set()
    for suffix in text_suffixes:
        if not isinstance(suffix, str):
            raise TypeError("text_suffixes entries must be strings")
        # Exactly what Path.suffix can return, so "txt" and ".tar.gz" fail.
        if not suffix or PurePath("x" + suffix).suffix != suffix:
            raise ValueError(f"text suffix must look like '.txt': {suffix!r}")
        suffixes.add(suffix.lower())
    return frozenset(suffixes)


def split_leading_references(prompt: str) -> Tuple[Tuple[str, ...], str]:
    """Split leading ``@`` references from a prompt.

    Returns ``(references, text)`` where ``references`` are the token values
    without their ``@`` and ``text`` is the remaining prompt text (the leading
    separator whitespace before the text is dropped, the rest is preserved).
    ``@@`` escapes a literal leading ``@`` and ends the reference run; a bare
    ``@`` ends the run too. Never performs I/O and never raises.
    """
    length = len(prompt)
    index = 0
    while index < length and prompt[index].isspace():
        index += 1
    if index >= length or prompt[index] != "@":
        return (), prompt

    references = []
    while index < length and prompt[index] == "@":
        end = index + 1
        while end < length and not prompt[end].isspace():
            end += 1
        token = prompt[index:end]
        if token == "@":
            return tuple(references), prompt[index:]
        if token.startswith("@@"):
            return tuple(references), "@" + token[2:] + prompt[end:]
        references.append(token[1:])
        index = end
        while index < length and prompt[index].isspace():
            index += 1
    return tuple(references), prompt[index:]


def _resolve_remote(reference: str, text_suffixes: FrozenSet[str]) -> ContentPart:
    parsed = urlparse(reference)
    if not parsed.netloc:
        raise AttachmentError(f"attachment URL has no host: {reference}")
    if PurePosixPath(parsed.path).suffix.lower() in text_suffixes:
        # The host fetches nothing, so only local files can be pasted.
        raise AttachmentError(
            f"text files are pasted only from local paths: {reference}"
        )
    guessed, _ = mimetypes.guess_type(parsed.path)
    if guessed is None or not guessed.startswith("image/"):
        raise AttachmentError(
            f"attachment URL is not a supported image: {reference}"
        )
    return MediaPart(source_uri=reference)


def _local_path(reference: str, *, cwd: Path, enable_workspace: bool) -> Path:
    """The resolved path: symlinks are followed before any other check."""
    try:
        path = Path(reference).expanduser()
        if not path.is_absolute():
            path = cwd / path
        path = path.resolve()
    except (OSError, RuntimeError, ValueError):
        # An unknown home directory, a symlink loop (RuntimeError before
        # Python 3.13), or an embedded NUL (ValueError).
        raise AttachmentError(
            f"invalid attachment path: {reference}"
        ) from None
    if enable_workspace and not _within(path, cwd):
        raise AttachmentError(
            f"attachment path is outside --cwd: {reference}"
        )
    return path


def _too_large(size: int, limit: int, reference: str) -> AttachmentError:
    return AttachmentError(
        f"attachment is too large ({size} bytes; limit {limit}): {reference}"
    )


def _read_regular_file(path: Path, reference: str, max_bytes: int) -> bytes:
    try:
        info = path.stat()
    except OSError as exc:
        reason = exc.strerror or "not found"
        raise AttachmentError(
            f"cannot read attachment {reference!r}: {reason}"
        ) from None
    if not stat.S_ISREG(info.st_mode):
        raise AttachmentError(f"attachment is not a regular file: {reference}")
    if info.st_size > max_bytes:
        raise _too_large(info.st_size, max_bytes, reference)
    try:
        data = path.read_bytes()
    except OSError as exc:
        reason = exc.strerror or "read failed"
        raise AttachmentError(
            f"cannot read attachment {reference!r}: {reason}"
        ) from None
    if len(data) > max_bytes:
        raise _too_large(len(data), max_bytes, reference)
    return data


def _paste_text(reference: str, data: bytes) -> TextPart:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # As for --prompt-file, never echo decoder details (they quote bytes).
        raise AttachmentError(
            f"text attachment is not valid UTF-8: {reference}"
        ) from None
    text = text.rstrip()
    if not text:
        raise AttachmentError(f"text attachment is empty: {reference}")
    return TextPart(text)


def _unsupported_local(
    reference: str, path: Path, text_suffixes: FrozenSet[str]
) -> AttachmentError:
    if text_suffixes:
        kinds = f"image or text file ({', '.join(sorted(text_suffixes))})"
    else:
        kinds = "image"
    message = f"attachment is not a supported {kinds}: {reference}"
    if PurePath(reference).suffix.lower() != path.suffix.lower():
        # The file actually read decides, e.g. for a notes.txt -> .env link.
        message += f" (resolves to {path.name})"
    return AttachmentError(message)


def _resolve_local(
    reference: str,
    *,
    cwd: Path,
    enable_workspace: bool,
    limits: ContentLimits,
    text_suffixes: FrozenSet[str],
) -> Tuple[ContentPart, int]:
    path = _local_path(reference, cwd=cwd, enable_workspace=enable_workspace)
    if path.suffix.lower() in text_suffixes:
        max_bytes = min(limits.max_item_bytes, limits.max_text_bytes)
        data = _read_regular_file(path, reference, max_bytes)
        return _paste_text(reference, data), len(data)
    data = _read_regular_file(path, reference, limits.max_item_bytes)
    media_type = _image_media_type(path.name, data)
    if media_type is None:
        raise _unsupported_local(reference, path, text_suffixes)
    encoded = base64.b64encode(data).decode("ascii")
    return MediaPart(source_uri=f"data:{media_type};base64,{encoded}"), len(data)


def _resolve_one(
    reference: str,
    *,
    cwd: Path,
    enable_workspace: bool,
    limits: ContentLimits,
    text_suffixes: FrozenSet[str],
) -> Tuple[ContentPart, int]:
    if not isinstance(reference, str) or not reference or reference != reference.strip():
        raise AttachmentError(f"invalid attachment reference: {reference!r}")
    match = _SCHEME_RE.match(reference)
    if match is not None:
        scheme = match.group("scheme").lower()
        if scheme in {"http", "https"}:
            return _resolve_remote(reference, text_suffixes), 0
        raise AttachmentError(
            f"unsupported URL scheme in attachment: {scheme}://"
        )
    bare = _BARE_SCHEME_RE.match(reference)
    if bare is not None and bare.group("scheme").lower() in _REJECTED_SCHEMES:
        scheme = bare.group("scheme").lower()
        raise AttachmentError(
            f"unsupported URL scheme in attachment: {scheme}:"
        )
    return _resolve_local(
        reference,
        cwd=cwd,
        enable_workspace=enable_workspace,
        limits=limits,
        text_suffixes=text_suffixes,
    )


def resolve_content(
    references: Sequence[str],
    *,
    cwd: Path,
    enable_workspace: bool,
    limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
    text_suffixes: Iterable[str] = DEFAULT_TEXT_SUFFIXES,
) -> Tuple[ContentPart, ...]:
    """Resolve reference strings into ordered content parts.

    A local file whose suffix is in ``text_suffixes`` becomes one ``TextPart``
    (not merged with its neighbours); any other reference must be an image.
    Invalid ``text_suffixes`` raise a plain ``TypeError``/``ValueError``,
    never :class:`AttachmentError`.
    """
    suffixes = _text_suffix_set(text_suffixes)
    if len(references) > limits.max_items:
        raise AttachmentError(
            f"too many attachments: {len(references)} > {limits.max_items}"
        )
    parts = []
    total = 0
    for reference in references:
        part, size = _resolve_one(
            reference,
            cwd=cwd,
            enable_workspace=enable_workspace,
            limits=limits,
            text_suffixes=suffixes,
        )
        total += size
        if total > limits.max_total_bytes:
            raise AttachmentError(
                f"attachments exceed {limits.max_total_bytes} bytes in total"
            )
        parts.append(part)
    return tuple(parts)


def _merge_text_runs(parts: Sequence[ContentPart]) -> Tuple[ContentPart, ...]:
    """Join neighbouring text parts with ``TEXT_PASTE_SEPARATOR``."""
    merged = []
    for part in parts:
        if isinstance(part, TextPart) and merged and isinstance(merged[-1], TextPart):
            merged[-1] = TextPart(merged[-1].text + TEXT_PASTE_SEPARATOR + part.text)
        else:
            merged.append(part)
    return tuple(merged)


def parse_user_prompt(
    prompt: str,
    *,
    cwd: Path,
    enabled: bool,
    enable_workspace: bool,
    limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
    text_suffixes: Iterable[str] = DEFAULT_TEXT_SUFFIXES,
) -> Message:
    """Build the user message for a submitted prompt.

    With ``enabled`` false this is the identity (a plain string message) and
    performs no I/O. With ``enabled`` true, leading ``@`` references are
    resolved in order: text files are pasted and images become ``MediaPart``
    items. Neighbouring text, including the trailing typed text, is joined
    with ``TEXT_PASTE_SEPARATOR``. Without images the result is a plain
    string message, exactly as if the text had been typed. Raises
    :class:`AttachmentError` on any bad reference.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    suffixes = _text_suffix_set(text_suffixes)
    if not enabled:
        return Message(role="user", content=prompt)
    references, text = split_leading_references(prompt)
    if not references:
        return Message(role="user", content=text)
    parts = list(
        resolve_content(
            references,
            cwd=cwd,
            enable_workspace=enable_workspace,
            limits=limits,
            text_suffixes=suffixes,
        )
    )
    if text:
        parts.append(TextPart(text))
    merged = _merge_text_runs(parts)
    if not any(isinstance(part, MediaPart) for part in merged):
        # Every part was text, so merging left exactly one.
        (only,) = merged
        return Message(role="user", content=only.text)
    return Message(role="user", content=merged)


def content_item_to_responses(part: ContentPart, role: str) -> dict:
    """Map one in-memory content part to a Responses content-array object."""
    if isinstance(part, TextPart):
        return {
            "type": "output_text" if role == "assistant" else "input_text",
            "text": part.text,
        }
    if isinstance(part, MediaPart):
        # TODO: MediaPart is always emitted as input_image for now; revisit with
        # input_file/`detail` support.
        return {"type": "input_image", "image_url": part.source_uri}
    raise TypeError(f"unsupported content part: {type(part).__name__}")


__all__ = [
    "AttachmentError",
    "ContentLimits",
    "DEFAULT_CONTENT_LIMITS",
    "DEFAULT_MAX_CONTENT_ITEMS",
    "DEFAULT_MAX_ITEM_BYTES",
    "DEFAULT_MAX_TEXT_BYTES",
    "DEFAULT_MAX_TOTAL_BYTES",
    "DEFAULT_TEXT_SUFFIXES",
    "TEXT_PASTE_SEPARATOR",
    "content_item_to_responses",
    "parse_user_prompt",
    "resolve_content",
    "split_leading_references",
]
