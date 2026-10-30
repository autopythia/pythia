"""Experimental media tool: images that the model attaches to its own context.

The tool, ``media``, takes one method call, as icron's tool does:
``{"call": "attach('build/shot.png', text='Is the header green now?')"}``.
:meth:`MediaTool.attach` resolves its refs as the user's leading ``@``
references are resolved (:func:`media.resolve_content`), with the tool's own
limits, and queues one user message: a header that names the refs, the parts
in order, then the text. It queues in an outbox (:class:`Outbox`), which the
ui tool's screenshots share (ui_tool.py). The host takes the queue
(:meth:`Outbox.take`) at the turn's next interrupt point, where it takes the
user's steers, so the model sees the images in its next sample, once its tool
batch is done. Chat Completions takes images only in user messages; a user
message works the same way with every API that takes images (``MEDIA_APIS``).

Only ``attach`` is implemented (doc/interaction-media-tool-plan.md also plans
``images()``, ``help()`` and calls sent from icron). Nothing is saved until the
host appends the message: one that no turn took when the process ends is lost.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import mimetypes
from pathlib import Path
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
from .method_call import MethodCall
from .method_call import MethodCallError
from .method_call import MethodCallReader
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


class OutboxError(ValueError):
    """A message that the outbox refuses, past the image cap; the text is the
    calling tool's result."""


class Outbox:
    """What the model's own tools queue for its context: the media tool's
    attachments, and the ui tool's screenshots.

    The host takes the queue (``take``) at the turn's next interrupt point,
    where it takes the user's steers, and appends it before them. ``put``
    keeps the API's image cap (``IMAGE_CAPS``) for the next request, which
    carries the context's images, as the host binds them (``bind``), the
    queued ones, and the new message's. The writers and ``take`` run on the
    context thread; a lock guards the queue all the same.
    """

    def __init__(self, *, api: Optional[str], cap=_DEFAULT_CAP) -> None:
        cap = IMAGE_CAPS.get(api) if cap is _DEFAULT_CAP else cap
        if cap is not None and not isinstance(cap, ImageCap):
            raise TypeError("cap must be an ImageCap or None.")
        self.api, self.cap = api, cap
        # The tools that write here, by name, for the host's notices.
        self.writers: Tuple[str, ...] = ()
        self._lock = threading.Lock()
        self._queue: List[Message] = []
        self._context_items: Callable[[], Iterable[InteractionItem]] = tuple

    def add_writer(self, name: str) -> None:
        """Note a tool that writes here."""
        if name not in self.writers:
            self.writers += (name,)

    def bind(self, *, context_items: Callable[[], Iterable[InteractionItem]]) -> None:
        """Set what ``put`` counts toward the cap: the context's items as the
        next request would carry them, before the queued messages
        (``InteractionContext.model_items``)."""
        if not callable(context_items):
            raise TypeError("context_items must be callable.")
        self._context_items = context_items

    def put(self, message: Message, *, call: str) -> None:
        """Queue ``message``, unless its images would take the next request
        past the cap: then raise OutboxError, whose text starts with ``call``,
        and queue nothing."""
        images = [part for part in message.parts if isinstance(part, MediaPart)]
        with self._lock:
            if images and self.cap is not None:
                self._check_cap_locked(call, images)
            self._queue.append(message)

    def take(self) -> Tuple[Message, ...]:
        """Take every queued message, oldest first. The host calls it at the
        turn's interrupt points and appends what it returns, before the
        user's steers."""
        with self._lock:
            taken, self._queue = tuple(self._queue), []
        return taken

    def queued(self) -> int:
        """How many messages wait for delivery."""
        with self._lock:
            return len(self._queue)

    def _check_cap_locked(self, call: str, images: Sequence[MediaPart]) -> None:
        cap, api = self.cap, self.api
        present = [part for item in self._context_items() if isinstance(item, Message)
                   for part in item.parts if isinstance(part, MediaPart)]
        present += [part for message in self._queue for part in message.parts
                    if isinstance(part, MediaPart)]
        count = len(present) + len(images)
        if count > cap.max_images:
            raise OutboxError(
                f"{call}: the next request would carry {count} images, over this session's "
                f"cap of {cap.max_images} for {api}; the context and the queue hold "
                f"{len(present)} already, the user's included. Nothing was queued.")
        inline = sum(len(part.source_uri) for part in (*present, *images)
                     if part.source_uri.startswith("data:"))
        if inline > cap.max_inline_bytes:
            held, most = _format_byte_count(inline), _format_byte_count(cap.max_inline_bytes)
            raise OutboxError(
                f"{call}: the next request's images would hold {held} of inline data, over "
                f"this session's cap of {most} for {api}. Nothing was queued.")


class MediaTool:
    """The media tool, which queues in an outbox (``Outbox``).

    ``attach`` resolves its refs at once, so that a bad ref fails its call,
    and queues one user message; the host takes the outbox's queue at the
    turn's next interrupt point. The host binds what the handler reads live
    (``bind``): the tools' workspace policy, and the items that the next
    request would carry, for the outbox's cap. The handler runs on the
    context thread.
    """

    def __init__(self, *, cwd: Union[str, Path], api: Optional[str],
                 limits: ContentLimits = MEDIA_TOOL_LIMITS, cap=_DEFAULT_CAP,
                 outbox: Optional[Outbox] = None) -> None:
        """``outbox``, shared with the ui tool, brings its own cap; without
        one, the tool has an outbox of its own, for ``api``, with ``cap`` (by
        default the API's, ``IMAGE_CAPS``)."""
        if not isinstance(limits, ContentLimits):
            raise TypeError("limits must be ContentLimits.")
        if outbox is None:
            outbox = Outbox(api=api, cap=cap)
        elif not isinstance(outbox, Outbox):
            raise TypeError("outbox must be an Outbox or None.")
        elif cap is not _DEFAULT_CAP:
            raise TypeError("give cap or outbox, not both: an outbox has its own cap.")
        elif outbox.api != api:
            raise ValueError(f"the outbox is for {outbox.api}, not {api}.")
        self._cwd = Path(cwd).expanduser().resolve()
        self._limits = limits
        self.outbox = outbox
        outbox.add_writer(TOOL_NAME)
        self._enable_workspace: Callable[[], bool] = lambda: True
        self._tool: Optional[Tool] = None

    def bind(self, *, enable_workspace: Optional[Callable[[], bool]] = None,
             context_items: Optional[Callable[[], Iterable[InteractionItem]]] = None) -> None:
        """Set what the handler reads when it runs: the workspace policy now
        (true keeps local refs inside cwd), and the context's items as the
        next request would carry them, for the outbox's cap
        (``Outbox.bind``)."""
        if enable_workspace is not None:
            if not callable(enable_workspace):
                raise TypeError("enable_workspace must be callable.")
            self._enable_workspace = enable_workspace
        if context_items is not None:
            self.outbox.bind(context_items=context_items)

    def tool(self) -> Tool:
        """The model's media tool: each call is one call of this API."""
        if self._tool is None:
            spec = ToolSpec(TOOL_NAME, _description(self._limits, self.outbox.api,
                                                    self.outbox.cap), _PARAMETERS)
            self._tool = Tool(spec, self._handle, timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS)
        return self._tool

    def take(self) -> Tuple[Message, ...]:
        """Take every message in the outbox, oldest first (``Outbox.take``)."""
        return self.outbox.take()

    def queued(self) -> int:
        """How many messages wait in the outbox."""
        return self.outbox.queued()

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
        try:
            self.outbox.put(message, call=call)
        except OutboxError as exc:
            raise MediaToolError(str(exc)) from None
        return Attached(message, tuple(_describe_non_text_source(part.source_uri)
                                       for part in images),
                        tuple(ref for part, ref in zip(parts, refs)
                              if isinstance(part, TextPart)))

    # The tool's handler, on the context thread.

    def _handle(self, arguments, *, timeout_seconds=None) -> ToolOutcome:
        del timeout_seconds
        try:
            call = _read_call(arguments)
            attached = self.attach(*call.arguments.get("refs", ()),
                                   text=call.arguments.get("text"))
        except (MethodCallError, MediaToolError) as exc:
            return ToolOutcome(str(exc), success=False)
        return ToolOutcome(_attached_text(attached))


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
# {"call": "attach('build/shot.png', text='Is the header green now?')"}
# (method_call.py).
_READER = MethodCallReader(
    TOOL_NAME, {"attach": MediaTool.attach},
    example="attach('build/shot.png', text='Is the header green now?')",
    short_example="attach('build/shot.png')",
    listing="the tool has one, attach(*refs, text=None)",
    positional="ref", literal="'build/shot.png'")


def _read_call(arguments) -> MethodCall:
    """The call that a tool input stands for; a MethodCallError says what to
    give instead."""
    return _READER.read(arguments)


def call_text(arguments) -> Optional[str]:
    """A tool input as the tool reads it, for transcripts: the call in
    canonical form, such as media.attach('build/shot.png', text='Hi.'). None
    if the tool would refuse to read it."""
    try:
        return _READER.read(arguments).text
    except MethodCallError:
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
    "MediaToolError", "Outbox", "OutboxError", "TOOL_NAME", "call_text",
]
