"""Experimental media tool: images that the model attaches to its own context.

The tool, ``media``, takes one method call, as icron's tool does:
``{"call": "attach('build/shot.png', text='Is the header green now?')"}``.
:meth:`MediaTool.attach` resolves its refs as the user's leading ``@``
references are resolved (:func:`media.resolve_content`), with the tool's own
limits, and queues one user message: a header that names the refs, the parts
in order, then the text. The host takes the queue (:meth:`MediaTool.take`) at
the turn's next interrupt point, where it takes the user's steers, so the
model sees the images in its next sample, once its tool batch is done. Chat
Completions takes images only in user messages; a user message works the same
way with every API that takes images (``MEDIA_APIS``).

Only ``attach`` is implemented (doc/interaction-media-tool-plan.md also plans
``images()``, ``help()`` and calls sent from icron). Nothing is saved until the
host appends the message: one that no turn took when the process ends is lost.
"""

from __future__ import annotations

import ast
import base64
from collections.abc import Mapping
from dataclasses import dataclass
import inspect
import mimetypes
from pathlib import Path
import re
import threading
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

from .display import _describe_non_text_source
from .display import _format_byte_count
from .environment import Tool
from .environment import ToolOutcome
from .environment import ToolSpec
from .items import InteractionItem
from .items import MediaPart
from .items import Message
from .items import TextPart
from .media import AttachmentError
from .media import ContentLimits
from .media import _sniff_image_media_type
from .media import join_user_messages
from .media import resolve_content
from .timeouts import DEFAULT_TOOL_TIMEOUT_SECONDS


TOOL_NAME = "media"
# The delivered message starts with this, so that the model knows the message
# isn't the user's, as icron's WAKE_HEADER does for its wake-ups.
MEDIA_HEADER = "Media from your media tool"

_MIB = 1024 * 1024
# The user's limits (ContentLimits), smaller: 7 MiB is 9.3 MiB in base64,
# under the Claude API's 10 MB per image; a pasted text file stays some
# thousands of tokens.
MEDIA_TOOL_LIMITS = ContentLimits(max_items=4, max_item_bytes=7 * _MIB,
                                  max_total_bytes=14 * _MIB, max_text_bytes=64 * 1024)
# What Anthropic and OpenAI take. The user's @ references accept any image/*
# name (the TODO above media._image_media_type); the tool's local images must
# have one of these signatures, and are labeled by it.
_IMAGE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp")
_IMAGE_KINDS = "PNG, JPEG, GIF or WebP"


@dataclass(frozen=True)
class ImageCap:
    """The most images that one request may carry, the user's included, and
    the most inline image data (``data:`` URLs) among them, in bytes."""

    max_images: int
    max_inline_bytes: int


# Per API; an API without one has no cap yet. Messages: past 20 images in a
# request, a stricter size limit applies to every image in it, and pythia
# can't check image dimensions; a request takes at most 32 MB (Anthropic's
# vision guide), so its images get 24 MiB, with room for the rest.
IMAGE_CAPS: Dict[str, ImageCap] = {"messages": ImageCap(20, 24 * _MIB)}


class MediaToolError(ValueError):
    """A media call that can't be read, or an attach that fails; its text is
    the tool's result."""


@dataclass(frozen=True)
class Attached:
    """What attach queued: the message, and its parts' labels."""

    message: Message
    images: Tuple[str, ...]  # each as the transcript labels it: image/png 14.2 KiB, or the URL
    texts: Tuple[str, ...]  # the pasted text files' refs


_DEFAULT_CAP = object()


class MediaTool:
    """The media tool and its outbox.

    ``attach`` resolves its refs at once, so that a bad ref fails its call,
    and queues one user message; the host takes the queue (``take``) at the
    turn's next interrupt point. The host binds what the handler reads live
    (``bind``): the tools' workspace policy, and the items that the next
    request would carry, for the cap. The handler and ``take`` run on the
    context thread; a lock guards the queue all the same.
    """

    def __init__(self, *, cwd: Union[str, Path], api: Optional[str],
                 limits: ContentLimits = MEDIA_TOOL_LIMITS, cap=_DEFAULT_CAP) -> None:
        if not isinstance(limits, ContentLimits):
            raise TypeError("limits must be ContentLimits.")
        cap = IMAGE_CAPS.get(api) if cap is _DEFAULT_CAP else cap
        if cap is not None and not isinstance(cap, ImageCap):
            raise TypeError("cap must be an ImageCap or None.")
        self._cwd = Path(cwd).expanduser().resolve()
        self._api, self._limits, self._cap = api, limits, cap
        self._lock = threading.Lock()
        self._outbox: List[Message] = []
        self._enable_workspace: Callable[[], bool] = lambda: True
        self._context_items: Callable[[], Iterable[InteractionItem]] = tuple
        self._tool: Optional[Tool] = None

    def bind(self, *, enable_workspace: Optional[Callable[[], bool]] = None,
             context_items: Optional[Callable[[], Iterable[InteractionItem]]] = None) -> None:
        """Set what the handler reads when it runs: the workspace policy now
        (true keeps local refs inside cwd), and the context's items as the
        next request would carry them, before the queued messages
        (``InteractionContext.model_items``)."""
        if enable_workspace is not None:
            if not callable(enable_workspace):
                raise TypeError("enable_workspace must be callable.")
            self._enable_workspace = enable_workspace
        if context_items is not None:
            if not callable(context_items):
                raise TypeError("context_items must be callable.")
            self._context_items = context_items

    def tool(self) -> Tool:
        """The model's media tool: each call is one call of this API."""
        if self._tool is None:
            spec = ToolSpec(TOOL_NAME, _description(self._limits, self._api, self._cap),
                            _PARAMETERS)
            self._tool = Tool(spec, self._handle, timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS)
        return self._tool

    def take(self) -> Tuple[Message, ...]:
        """Take every queued message, oldest first. The host calls it at the
        turn's interrupt points and appends what it returns, before the
        user's steers."""
        with self._lock:
            taken, self._outbox = tuple(self._outbox), []
        return taken

    def queued(self) -> int:
        """How many messages wait for delivery."""
        with self._lock:
            return len(self._outbox)

    def attach(self, *refs: str, text: Optional[str] = None) -> Attached:
        """Resolve ``refs`` now and queue one user message with them and
        ``text``. Raises MediaToolError, and then queues nothing."""
        call = "media.attach()"
        if not refs:
            raise MediaToolError(f"{call}: give at least one ref, such as "
                                 "attach('build/shot.png').")
        for index, ref in enumerate(refs, 1):
            if not isinstance(ref, str):
                raise MediaToolError(f"{call}: ref {index} must be text, such as "
                                     f"'build/shot.png', not {type(ref).__name__}.")
        if text is not None and not isinstance(text, str):
            raise MediaToolError(f"{call}: text must be text, not {type(text).__name__}.")
        try:
            parts = resolve_content(refs, cwd=self._cwd,
                                    enable_workspace=bool(self._enable_workspace()),
                                    limits=self._limits)
        except AttachmentError as exc:
            raise MediaToolError(f"{call}: {exc}") from None
        parts = tuple(_checked_part(part, ref, call) for part, ref in zip(parts, refs))
        header = TextPart(f"{MEDIA_HEADER}: {', '.join(repr(ref) for ref in refs)}.")
        tail = (TextPart(text.strip()),) if text is not None and text.strip() else ()
        # Neighbouring text joins as in a user's prompt (TEXT_PASTE_SEPARATOR).
        message = join_user_messages(Message("user", (part,))
                                     for part in (header, *parts, *tail))
        images = tuple(part for part in parts if isinstance(part, MediaPart))
        with self._lock:
            if images and self._cap is not None:
                self._check_cap_locked(call, images)
            self._outbox.append(message)
        return Attached(message, tuple(_describe_non_text_source(part.source_uri)
                                       for part in images),
                        tuple(ref for part, ref in zip(parts, refs)
                              if isinstance(part, TextPart)))

    # The tool's handler, on the context thread.

    def _handle(self, arguments, *, timeout_seconds=None) -> ToolOutcome:
        del timeout_seconds
        try:
            call = _read_call(arguments)
            attached = self.attach(*call.refs, text=call.text)
        except MediaToolError as exc:
            return ToolOutcome(str(exc), success=False)
        return ToolOutcome(_attached_text(attached))

    def _check_cap_locked(self, call: str, images: Sequence[MediaPart]) -> None:
        cap, api = self._cap, self._api
        present = [part for item in self._context_items() if isinstance(item, Message)
                   for part in item.parts if isinstance(part, MediaPart)]
        present += [part for message in self._outbox for part in message.parts
                    if isinstance(part, MediaPart)]
        count = len(present) + len(images)
        if count > cap.max_images:
            raise MediaToolError(
                f"{call}: the next request would carry {count} images, over this session's "
                f"cap of {cap.max_images} for {api}; the context and the queue hold "
                f"{len(present)} already, the user's included. Nothing was queued.")
        inline = sum(len(part.source_uri) for part in (*present, *images)
                     if part.source_uri.startswith("data:"))
        if inline > cap.max_inline_bytes:
            held, most = _format_byte_count(inline), _format_byte_count(cap.max_inline_bytes)
            raise MediaToolError(
                f"{call}: the next request's images would hold {held} of inline data, over "
                f"this session's cap of {most} for {api}. Nothing was queued.")


def _checked_part(part, ref: str, call: str):
    """A resolved part, with the tool's own type rule: a local image must have
    a PNG, JPEG, GIF or WebP signature, which also names its type; a URL must
    name one of those types."""
    if isinstance(part, TextPart):
        return part
    uri = part.source_uri
    if not uri.startswith("data:"):
        guessed, _ = mimetypes.guess_type(urlparse(uri).path)
        if guessed not in _IMAGE_TYPES:
            raise MediaToolError(f"{call}: {ref!r} doesn't name a {_IMAGE_KINDS} image.")
        return part
    header, _, encoded = uri.partition(",")
    # 16 base64 characters are 12 bytes, enough for each signature.
    found = _sniff_image_media_type(base64.b64decode(encoded[:16]))
    if found is None:
        raise MediaToolError(f"{call}: {ref!r} is not a {_IMAGE_KINDS} image, the types "
                             "that the model APIs take.")
    if header != f"data:{found};base64":
        part = MediaPart(f"data:{found};base64,{encoded}")
    return part


def _attached_text(attached: Attached) -> str:
    pieces = []
    if attached.images:
        count = len(attached.images)
        pieces.append(f"{count} image{'s' if count != 1 else ''} "
                      f"({', '.join(attached.images)})")
    if attached.texts:
        count = len(attached.texts)
        pieces.append(f"{count} text file{'s' if count != 1 else ''} "
                      f"({', '.join(repr(ref) for ref in attached.texts)})")
    return (f"Queued {' and '.join(pieces)}, in a user message that arrives right before "
            "your next sample, once this tool batch is done.")


# The tool's input: one method call in Python, as icron's tool takes,
# {"call": "attach('build/shot.png', text='Is the header green now?')"}. Also
# read: a "media." prefix, as transcripts show calls. Nothing in a call is
# evaluated: its values are literals.

_METHODS = ("attach",)
_NO_CALL = ("media: give call, one method call such as "
            "attach('build/shot.png', text='Is the header green now?').")


@dataclass(frozen=True)
class _Call:
    """A tool input as the tool reads it."""

    refs: Tuple[object, ...]
    text: object = None

    def text_form(self) -> str:
        """The call in canonical form: media.attach('a.png', text='Hi.')."""
        arguments = [repr(ref) for ref in self.refs]
        if self.text is not None:
            arguments.append(f"text={self.text!r}")
        return f"media.attach({', '.join(arguments)})"


def _read_call(arguments) -> _Call:
    """The call that a tool input stands for; a MediaToolError says what to
    give instead."""
    if not isinstance(arguments, Mapping):
        raise MediaToolError(_NO_CALL)
    fields = dict(arguments)
    source = fields.pop("call", None)
    if fields:
        raise MediaToolError(f"media: unexpected field {sorted(fields)[0]!r}; give call only.")
    if source is None:
        raise MediaToolError(_NO_CALL)
    if not isinstance(source, str):
        raise MediaToolError("media: call must be text, one method call such as "
                             "attach('build/shot.png').")
    source = re.sub(r"^media\s*\.\s*", "", source.strip())
    try:
        node = ast.parse(source, mode="eval").body
    except SyntaxError as exc:
        raise MediaToolError("media: call must be one method call, such as "
                             f"attach('build/shot.png'); {exc.msg}.") from None
    except (ValueError, RecursionError, MemoryError):  # a null character; deep nesting
        node = None
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
        raise MediaToolError("media: call must be one method call, such as "
                             "attach('build/shot.png').")
    name = node.func.id
    if name not in _METHODS:
        raise MediaToolError(f"media: there is no method {name}(); the tool has one, "
                             "attach(*refs, text=None).")
    call = f"media.{name}()"
    args = []
    for index, value in enumerate(node.args, 1):
        if isinstance(value, ast.Starred):
            raise MediaToolError(f"{call}: give each ref by itself, not *sequence.")
        args.append(_literal(value, call, f"ref {index}"))
    keywords = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise MediaToolError(f"{call}: give each argument by itself, not **mapping.")
        keywords[keyword.arg] = _literal(keyword.value, call, keyword.arg)
    try:
        bound = inspect.signature(MediaTool.attach).bind(None, *args, **keywords)
    except TypeError as exc:
        raise MediaToolError(f"{call}: {exc}") from None
    return _Call(tuple(bound.arguments.get("refs", ())), bound.arguments.get("text"))


def _literal(node: ast.expr, call: str, what: str):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        raise MediaToolError(f"{call}: {what} must be a literal, such as "
                             "'build/shot.png'.") from None


def call_text(arguments) -> Optional[str]:
    """A tool input as the tool reads it, for transcripts: the call in
    canonical form, such as media.attach('build/shot.png', text='Hi.'). None
    if the tool would refuse to read it."""
    try:
        return _read_call(arguments).text_form()
    except MediaToolError:
        return None


def _description(limits: ContentLimits, api: Optional[str], cap: Optional[ImageCap]) -> str:
    capped = ("" if cap is None else
              f" With {api}, a request carries at most {cap.max_images} images, the user's "
              f"included, and {_format_byte_count(cap.max_inline_bytes)} of image data; past "
              "that, attach fails.")
    return (
        "Attach images to your own context, to look at them: they arrive in a user message "
        "right before your next sample, once this tool batch is done. call is one method "
        "call, in Python with literal arguments; for example:\n"
        '  {"call": "attach(\'build/shot.png\', text=\'Is the header green now?\')"}\n'
        "- attach(*refs, text=None): each ref is what the user writes after @ in a prompt: "
        "a file path, relative to the working directory (~ expands), or an http(s) URL of an "
        "image, which the model provider fetches. A local image must be a "
        f"{_IMAGE_KINDS} file; a .md or .txt file is pasted as text. text follows them in "
        f"the message, as your note. At most {limits.max_items} refs per call, each file at "
        f"most {_format_byte_count(limits.max_item_bytes)}.\n"
        "Images stay in the context and are sent again with every request until a "
        f"compaction removes them, so attach only what you need.{capped}"
    )


_PARAMETERS = {
    "type": "object",
    "properties": {
        "call": {"type": "string",
                 "description": "One media method call, in Python with literal arguments, such "
                                "as attach('build/shot.png', text='Is the header green now?')."},
    },
    "required": ["call"],
    "additionalProperties": False,
}


__all__ = [
    "Attached", "IMAGE_CAPS", "ImageCap", "MEDIA_HEADER", "MEDIA_TOOL_LIMITS", "MediaTool",
    "MediaToolError", "TOOL_NAME", "call_text",
]
