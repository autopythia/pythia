"""Experimental parsing of leading ``@path-or-uri`` user-prompt components.

This module is pure and provider-neutral. It turns the leading whitespace
separated ``@`` references of a user prompt into ordered
``pythia.interaction.items`` content parts and returns a normal
``Message(role="user", ...)``. It performs host file reads at submission time
and inlines local images as ``data:`` URLs; remote ``http(s)`` images are
forwarded by URL. Nothing here talks to a model.

Only images are accepted in this slice because :class:`MediaPart` is currently
serialized as the Responses ``input_image`` content item (see the ``TODO`` on
that type); any non-image reference is rejected rather than mislabeled.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import mimetypes
from pathlib import Path
import re
import stat
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


@dataclass(frozen=True)
class ContentLimits:
    max_items: int = DEFAULT_MAX_CONTENT_ITEMS
    max_item_bytes: int = DEFAULT_MAX_ITEM_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES


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


def _resolve_remote(reference: str) -> ContentPart:
    parsed = urlparse(reference)
    if not parsed.netloc:
        raise AttachmentError(f"attachment URL has no host: {reference}")
    guessed, _ = mimetypes.guess_type(parsed.path)
    if guessed is None or not guessed.startswith("image/"):
        raise AttachmentError(
            f"attachment URL is not a supported image: {reference}"
        )
    return MediaPart(source_uri=reference)


def _resolve_local(
    reference: str,
    *,
    cwd: Path,
    enable_workspace: bool,
    limits: ContentLimits,
) -> Tuple[ContentPart, int]:
    try:
        path = Path(reference).expanduser()
    except (RuntimeError, ValueError):
        raise AttachmentError(
            f"invalid attachment path: {reference}"
        ) from None
    if not path.is_absolute():
        path = cwd / path
    path = path.resolve()
    if enable_workspace and not _within(path, cwd):
        raise AttachmentError(
            f"attachment path is outside --cwd: {reference}"
        )
    try:
        info = path.stat()
    except OSError as exc:
        reason = exc.strerror or "not found"
        raise AttachmentError(
            f"cannot read attachment {reference!r}: {reason}"
        ) from None
    if not stat.S_ISREG(info.st_mode):
        raise AttachmentError(f"attachment is not a regular file: {reference}")
    if info.st_size > limits.max_item_bytes:
        raise AttachmentError(
            f"attachment is too large ({info.st_size} bytes): {reference}"
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        reason = exc.strerror or "read failed"
        raise AttachmentError(
            f"cannot read attachment {reference!r}: {reason}"
        ) from None
    if len(data) > limits.max_item_bytes:
        raise AttachmentError(
            f"attachment is too large ({len(data)} bytes): {reference}"
        )
    media_type = _image_media_type(path.name, data)
    if media_type is None:
        raise AttachmentError(
            f"attachment is not a supported image: {reference}"
        )
    encoded = base64.b64encode(data).decode("ascii")
    return MediaPart(source_uri=f"data:{media_type};base64,{encoded}"), len(data)


def _resolve_one(
    reference: str,
    *,
    cwd: Path,
    enable_workspace: bool,
    limits: ContentLimits,
) -> Tuple[ContentPart, int]:
    if not isinstance(reference, str) or not reference or reference != reference.strip():
        raise AttachmentError(f"invalid attachment reference: {reference!r}")
    match = _SCHEME_RE.match(reference)
    if match is not None:
        scheme = match.group("scheme").lower()
        if scheme in {"http", "https"}:
            return _resolve_remote(reference), 0
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
    )


def resolve_content(
    references: Sequence[str],
    *,
    cwd: Path,
    enable_workspace: bool,
    limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
) -> Tuple[ContentPart, ...]:
    """Resolve reference strings into ordered content parts."""
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
        )
        total += size
        if total > limits.max_total_bytes:
            raise AttachmentError(
                f"attachments exceed {limits.max_total_bytes} bytes in total"
            )
        parts.append(part)
    return tuple(parts)


def parse_user_prompt(
    prompt: str,
    *,
    cwd: Path,
    enabled: bool,
    enable_workspace: bool,
    limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
) -> Message:
    """Build the user message for a submitted prompt.

    With ``enabled`` false this is the identity (a plain string message) and
    performs no I/O. With ``enabled`` true, leading ``@`` references become
    ``MediaPart`` items followed by the trailing ``TextPart`` (omitted for an
    attachment-only prompt). Raises :class:`AttachmentError` on any bad
    reference.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
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
        )
    )
    if text:
        parts.append(TextPart(text))
    return Message(role="user", content=tuple(parts))


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
    "DEFAULT_MAX_TOTAL_BYTES",
    "content_item_to_responses",
    "parse_user_prompt",
    "resolve_content",
    "split_leading_references",
]
