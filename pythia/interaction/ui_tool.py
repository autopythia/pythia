"""Experimental ui tool: screenshots of this Mac's displays, for the model.

The tool, ``ui``, takes one method call, as the media tool does:
``{"call": "screenshot()"}``. The UI plugin does the work: the crate in
``_pythia_ui/``, which ``python3 install.py -e --enable-experimental-ui-plugin``
builds and installs as ``~/.pythia/plugin/libpythia_ui.dylib``, and which the
CLI loads at launch with --enable-experimental-ui-plugin (:func:`load_ui_plugin`,
:mod:`.plugin`). A screenshot goes into the media tool's outbox
(:class:`.media_tool.Outbox`), so it reaches the model the same way: in a user
message right before its next sample, once its tool batch is done, within the
same image cap.

Only ``screenshot`` is implemented; moving the cursor, clicking and typing
come later (doc/interaction-ui-tool-plan.md).
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import struct
import sys
from typing import Optional, Tuple, Union

from .display import _format_byte_count
from .environment import Tool
from .environment import ToolOutcome
from .environment import ToolSpec
from .items import MediaPart
from .items import Message
from .items import TextPart
from .media import MEDIA_APIS
from .media_tool import MEDIA_TOOL_LIMITS
from .media_tool import Outbox
from .media_tool import OutboxError
from .method_call import MethodCallError
from .method_call import MethodCallReader
from .plugin import Plugin
from .plugin import PluginError
from .plugin import PluginVersionError
from .plugin import load_plugin
from .timeouts import DEFAULT_TOOL_TIMEOUT_SECONDS


TOOL_NAME = "ui"
OPTION = "--enable-experimental-ui-plugin"
PLUGIN_FILE = "libpythia_ui.dylib"
INSTALL_COMMAND = "python3 install.py -e --enable-experimental-ui-plugin"
# The delivered message starts with this, as the media tool's starts with its
# MEDIA_HEADER, so that the model knows the message isn't the user's.
UI_HEADER = "Screenshot from your ui tool"
# The methods that the tool offers, each when the plugin lists its op (info).
METHODS = ("screenshot",)
# The most that a screenshot's PNG may have: the media tool's limit per image,
# under the Claude API's 10 MB once in base64.
MAX_SCREENSHOT_BYTES = MEDIA_TOOL_LIMITS.max_item_bytes
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def plugin_path() -> Path:
    """Where install.py puts the UI plugin, and the CLI loads it from."""
    return Path.home() / ".pythia" / "plugin" / PLUGIN_FILE


def _platform() -> str:
    return sys.platform


class UiPluginError(Exception):
    """The UI plugin can't be used; the launch stops with this text."""


class UiToolError(ValueError):
    """A ui call that fails; its text is the tool's result."""


@dataclass(frozen=True)
class UiPlugin:
    """The loaded UI plugin, its info reply, and the methods it serves."""

    plugin: Plugin
    info: Mapping[str, object]
    methods: Tuple[str, ...]


def load_ui_plugin(*, api: Optional[str], default_tools: bool,
                   path: Union[str, Path, None] = None) -> UiPlugin:
    """Load the UI plugin for --enable-experimental-ui-plugin, from
    ``plugin_path()`` by default, and read its info. The plugin was asked
    for, so a UiPluginError, which says why it can't be used, stops the
    launch rather than the session going on without the tool."""
    if not default_tools:
        raise UiPluginError(f"{OPTION} needs the default tools, which "
                            "--enable-default-tools False turns off.")
    if api not in MEDIA_APIS:
        raise UiPluginError(f"{OPTION}: --endpoint-api {api} takes no images; use codex, "
                            "responses, chat-completions, or messages.")
    if _platform() != "darwin":
        raise UiPluginError(f"{OPTION}: the UI plugin is for macOS, and this is "
                            f"{_platform()}.")
    path = plugin_path() if path is None else Path(path)
    if not path.is_file():
        raise UiPluginError(f"{OPTION}: there is no UI plugin at {path}; build and install it "
                            f"from a pythia checkout on this Mac with `{INSTALL_COMMAND}`.")
    try:
        plugin = load_plugin(path)
        info = plugin.call({"op": "info"})
    except PluginVersionError as exc:
        raise UiPluginError(f"{OPTION}: {exc} Rebuild it from this pythia's checkout with "
                            f"`{INSTALL_COMMAND}`.") from None
    except PluginError as exc:
        raise UiPluginError(f"{OPTION}: {exc}") from None
    if info["ok"] is not True:
        raise UiPluginError(f"{OPTION}: the UI plugin's info failed: {info.get('error')}")
    ops = info.get("ops") if isinstance(info.get("ops"), list) else []
    methods = tuple(method for method in METHODS if method in ops)
    if not methods:
        raise UiPluginError(f"{OPTION}: the UI plugin at {path} offers none of the ui tool's "
                            f"methods ({', '.join(METHODS)}); its ops are "
                            f"{', '.join(map(str, ops)) or 'none'}.")
    return UiPlugin(plugin, info, methods)


@dataclass(frozen=True)
class Shot:
    """What screenshot queued: the message, and the image's facts."""

    message: Message
    display: int
    width: int
    height: int
    size: int  # the PNG's bytes
    cursor: bool  # whether the pointer is drawn


class UiTool:
    """The ui tool, whose calls run in the UI plugin, and which queues in an
    outbox (``Outbox``).

    ``screenshot`` asks the plugin for one, checks the PNG, and queues one
    user message: a header that names the display and the size, then the
    image. The host takes the outbox's queue at the turn's next interrupt
    point. The handler runs on the context thread; the plugin's call
    releases the GIL while it runs, and keeps its own deadline.
    """

    def __init__(self, plugin: UiPlugin, *, outbox: Outbox) -> None:
        if not isinstance(plugin, UiPlugin):
            raise TypeError("plugin must be a UiPlugin.")
        if not isinstance(outbox, Outbox):
            raise TypeError("outbox must be an Outbox.")
        self._plugin = plugin
        self.outbox = outbox
        outbox.add_writer(TOOL_NAME)
        self._tool: Optional[Tool] = None

    def tool(self) -> Tool:
        """The model's ui tool: each call is one call of this API."""
        if self._tool is None:
            spec = ToolSpec(TOOL_NAME, _description(self._plugin, self.outbox), _PARAMETERS)
            self._tool = Tool(spec, self._handle, timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS)
        return self._tool

    def notice(self) -> str:
        """The CLI's notice at launch: what the tool can see, and whether
        macOS allows screen recording yet."""
        info = self._plugin.info
        text = (f"ui: the model can take screenshots of this Mac's displays with the ui tool "
                f"({info.get('name')} {info.get('version')}, {self._plugin.plugin.path}); "
                "each one goes to the model provider, with whatever the display shows.")
        capture = info.get("capture")
        if isinstance(capture, Mapping) and capture.get("allowed") is False:
            text += (" Screen recording isn't allowed yet: the first screenshot asks macOS "
                     "for it, and fails.")
        return text

    def screenshot(self, display: Optional[int] = None, cursor: bool = True) -> Shot:
        """Take a screenshot of ``display`` (by default the main one) and
        queue it. Raises UiToolError, and then queues nothing."""
        call = "ui.screenshot()"
        if display is not None and (type(display) is not int or not 0 <= display < 2 ** 32):
            raise UiToolError(f"{call}: display must be a display ID, such as "
                              f"{_first_display(self._plugin.info)}, or None for the main "
                              f"display, not {display!r}.")
        if type(cursor) is not bool:
            raise UiToolError(f"{call}: cursor must be True or False, not {cursor!r}.")
        request = {"op": "screenshot"}
        if display is not None:
            request["display"] = display
        if not cursor:
            request["cursor"] = False
        try:
            reply = self._plugin.plugin.call(request)
        except PluginError as exc:
            raise UiToolError(f"{call}: {exc}") from None
        if reply["ok"] is not True:
            error = str(reply.get("error") or "the UI plugin gave no reason.")
            # The plugin's messages name their op, as the call does.
            raise UiToolError(f"{call}: {error.removeprefix('screenshot: ')}")
        shot = _read_shot(reply, call)
        if shot.size > MAX_SCREENSHOT_BYTES:
            raise UiToolError(
                f"{call}: the screenshot of display {shot.display} is "
                f"{_format_byte_count(shot.size)} as PNG ({shot.width}×{shot.height} px), over "
                f"the {_format_byte_count(MAX_SCREENSHOT_BYTES)} that one image may have. "
                "Nothing was queued.")
        try:
            self.outbox.put(shot.message, call=call)
        except OutboxError as exc:
            raise UiToolError(str(exc)) from None
        return shot

    # The tool's handler, on the context thread.

    def _handle(self, arguments, *, timeout_seconds=None) -> ToolOutcome:
        del timeout_seconds  # the plugin keeps its own deadline, of 10 seconds
        try:
            call = _READER.read(arguments)
            if call.method not in self._plugin.methods:
                raise UiToolError(f"ui: this UI plugin has no {call.method}().")
            shot = self.screenshot(**call.arguments)
        except (MethodCallError, UiToolError) as exc:
            return ToolOutcome(str(exc), success=False)
        return ToolOutcome(_shot_text(shot, cursor=call.arguments.get("cursor", True)))


def _read_shot(reply: Mapping[str, object], call: str) -> Shot:
    """A screenshot reply's image and facts, checked: a reply that would put
    a broken image in the context fails the call instead."""
    def malformed(what: str) -> UiToolError:
        return UiToolError(f"{call}: the UI plugin's reply is malformed: {what}.")

    facts = {name: reply.get(name) for name in ("display", "width", "height", "bytes")}
    for name, value in facts.items():
        if type(value) is not int or value < 0:
            raise malformed(f"{name} is {value!r}")
    if reply.get("media_type") != "image/png":
        raise malformed(f"media_type is {reply.get('media_type')!r}, not 'image/png'")
    encoded = reply.get("png")
    if not isinstance(encoded, str):
        raise malformed("it has no png")
    try:
        png = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise malformed("png isn't base64") from None
    if not png.startswith(_PNG_SIGNATURE) or png[12:16] != b"IHDR":
        raise malformed("png isn't a PNG image")
    if len(png) != facts["bytes"]:
        raise malformed(f"png has {len(png)} bytes, not {facts['bytes']}")
    size = struct.unpack(">II", png[16:24])
    if size != (facts["width"], facts["height"]):
        raise malformed(f"png is {size[0]}×{size[1]} px, not {facts['width']}×{facts['height']}")
    header = TextPart(f"{UI_HEADER}: display {facts['display']}, "
                      f"{facts['width']}×{facts['height']} px.")
    message = Message("user", (header, MediaPart(f"data:image/png;base64,{encoded}")))
    return Shot(message, facts["display"], facts["width"], facts["height"], len(png),
                reply.get("cursor") is True)


def _shot_text(shot: Shot, *, cursor: bool) -> str:
    missing = (", without the pointer, which this capture path leaves out"
               if cursor and not shot.cursor else "")
    return (f"Queued a screenshot of display {shot.display} (image/png "
            f"{shot.width}×{shot.height} px, {_format_byte_count(shot.size)}{missing}), in a "
            "user message that arrives right before your next sample, once this tool batch "
            "is done.")


def _displays(info: Mapping[str, object]) -> str:
    """info's displays, for the description: 1 (main, built-in, 3024×1964 px)."""
    shown = []
    displays = info.get("displays")
    for display in displays if isinstance(displays, list) else ():
        if not isinstance(display, Mapping):
            continue
        traits = [trait for trait, key in (("main", "main"), ("built-in", "builtin"))
                  if display.get(key) is True]
        pixels = display.get("pixels")
        if isinstance(pixels, Mapping) and all(type(pixels.get(key)) is int
                                               for key in ("width", "height")):
            traits.append(f"{pixels['width']}×{pixels['height']} px")
        shown.append(f"{display.get('id')} ({', '.join(traits)})" if traits
                     else str(display.get("id")))
    return ", ".join(shown)


def _first_display(info: Mapping[str, object]) -> object:
    displays = info.get("displays")
    if isinstance(displays, list) and displays and isinstance(displays[0], Mapping):
        return displays[0].get("id", 1)
    return 1


def _description(plugin: UiPlugin, outbox: Outbox) -> str:
    displays = _displays(plugin.info)
    cap = outbox.cap
    capped = ("" if cap is None else
              f" With {outbox.api}, a request carries at most {cap.max_images} images, the "
              f"user's included, and {_format_byte_count(cap.max_inline_bytes)} of image data; "
              "past that, screenshot fails.")
    return (
        "See this Mac's screen: a screenshot arrives in a user message right before your next "
        "sample, once this tool batch is done. call is one method call, in Python with "
        "literal arguments; for example:\n"
        '  {"call": "screenshot()"}\n'
        "- screenshot(display=None, cursor=True): the display's own pixels, as a PNG of at "
        f"most {_format_byte_count(MAX_SCREENSHOT_BYTES)}; display is a display ID, by default "
        "the main display, and cursor draws the pointer where the capture path can."
        + (f" Displays: {displays}." if displays else "") + "\n"
        "A screenshot shows everything on the display, other apps and notifications "
        "included, and goes to the model provider. Screenshots stay in the context and are "
        "sent again with every request until a compaction removes them, so take only what "
        f"you need.{capped}"
    )


# The tool's input: one method call in Python, as the media tool takes,
# {"call": "screenshot()"} (method_call.py).
_READER = MethodCallReader(
    TOOL_NAME, {"screenshot": UiTool.screenshot},
    example="screenshot()", short_example="screenshot()",
    listing="the tool has one, screenshot(display=None, cursor=True)")


def call_text(arguments) -> Optional[str]:
    """A tool input as the tool reads it, for transcripts: the call in
    canonical form, such as ui.screenshot(display=2). None if the tool would
    refuse to read it."""
    try:
        return _READER.read(arguments).text
    except MethodCallError:
        return None


_PARAMETERS = {
    "type": "object",
    "properties": {
        "call": {"type": "string",
                 "description": "One ui method call, in Python with literal arguments, such "
                                "as screenshot()."},
    },
    "required": ["call"],
    "additionalProperties": False,
}


__all__ = [
    "INSTALL_COMMAND", "MAX_SCREENSHOT_BYTES", "METHODS", "OPTION", "PLUGIN_FILE", "Shot",
    "TOOL_NAME", "UI_HEADER", "UiPlugin", "UiPluginError", "UiTool", "UiToolError",
    "call_text", "load_ui_plugin", "plugin_path",
]
