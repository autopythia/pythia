"""Native plugins: dynamic libraries that pythia loads into its own process
with ctypes, through a C interface of three functions, with no Python headers
and no PyO3 (doc/interaction-ui-tool-plan.md, "The plugin interface"):

    uint32_t pythia_plugin_abi(void);  /* PLUGIN_ABI */
    int32_t pythia_plugin_call(const uint8_t *request, size_t request_len,
                               uint8_t **reply, size_t *reply_len);
    void pythia_plugin_free(uint8_t *reply, size_t reply_len);

A request and its reply are one JSON object each, in UTF-8. The reply echoes
the request's ``id``, and says ``"ok": true`` with its fields, or
``"ok": false`` with a ``kind`` and an ``error``. ``pythia_plugin_call``
returns 0 with a reply, and nonzero only when it couldn't make one. A reply
belongs to the plugin until ``pythia_plugin_free``, which takes its length.

Nothing here is specific to one plugin; the UI plugin (``_pythia_ui/``,
:mod:`.ui_tool`) is the first. A plugin is never unloaded: one that registers
Objective-C classes or system callbacks can't be, safely, so a new build of
it takes a restart.
"""

from __future__ import annotations

from collections.abc import Mapping
import ctypes
import itertools
import json
from pathlib import Path
import threading
from typing import Dict, Union

PLUGIN_ABI = 1

_REPLY = ctypes.POINTER(ctypes.c_uint8)
# Plain C functions (CFUNCTYPE): a call releases the GIL while it runs, so the
# terminal keeps drawing during a slow one, such as a screenshot.
_ABI = ctypes.CFUNCTYPE(ctypes.c_uint32)
_CALL = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t,
                         ctypes.POINTER(_REPLY), ctypes.POINTER(ctypes.c_size_t))
_FREE = ctypes.CFUNCTYPE(None, _REPLY, ctypes.c_size_t)
_FUNCTIONS = (("pythia_plugin_abi", _ABI), ("pythia_plugin_call", _CALL),
              ("pythia_plugin_free", _FREE))


class PluginError(Exception):
    """A plugin that can't be loaded, or a call that got no usable reply; the
    text says why."""


class PluginVersionError(PluginError):
    """A plugin whose interface version (``abi``) this pythia doesn't know."""

    def __init__(self, message: str, abi: int) -> None:
        super().__init__(message)
        self.abi = abi


class Plugin:
    """A loaded plugin: one request in, its reply out, one call at a time."""

    def __init__(self, library, *, path: Union[str, Path]) -> None:
        """Bind the three functions of ``library``, a ``ctypes.CDLL`` or an
        object with the same attributes, and check its interface version."""
        self.path = Path(path)
        functions = []
        for name, prototype in _FUNCTIONS:
            symbol = getattr(library, name, None)
            if symbol is None:
                raise PluginError(f"{self.path} isn't a pythia plugin: it has no {name}().")
            functions.append(ctypes.cast(symbol, prototype))
        self._library = library  # kept, as the plugin is never unloaded
        abi_function, self._call, self._free = functions
        self.abi = int(abi_function())
        if self.abi != PLUGIN_ABI:
            raise PluginVersionError(f"{self.path} has plugin interface version {self.abi}, "
                                     f"and this pythia knows version {PLUGIN_ABI}.", self.abi)
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    def call_bytes(self, request: bytes) -> bytes:
        """One request's bytes in, its reply's bytes out; a PluginError if the
        plugin made no reply."""
        if not isinstance(request, bytes):
            raise TypeError("request must be bytes.")
        reply, length = _REPLY(), ctypes.c_size_t()
        with self._lock:
            status = self._call(request, len(request), ctypes.byref(reply), ctypes.byref(length))
            if status != 0:
                raise PluginError(f"the plugin made no reply (pythia_plugin_call returned "
                                  f"{status}).")
            if not reply:
                raise PluginError("the plugin made no reply (a null pointer).")
            try:
                return ctypes.string_at(reply, length.value)
            finally:
                self._free(reply, length.value)

    def call(self, request: Mapping[str, object]) -> Dict[str, object]:
        """Send ``request``, an op and its fields, under an id of its own, and
        return the reply, which may report a failure (``"ok": false``). A
        PluginError means that there was no usable reply."""
        if not isinstance(request, Mapping) or "id" in request:
            raise ValueError("request must be a mapping without an id.")
        number = next(self._ids)
        raw = self.call_bytes(json.dumps({"id": number, **request}).encode("utf-8"))
        try:
            reply = json.loads(raw)
        except ValueError as exc:  # also a UnicodeDecodeError
            raise PluginError(f"the plugin's reply isn't JSON: {exc}.") from None
        if not isinstance(reply, dict):
            raise PluginError("the plugin's reply isn't a JSON object.")
        if type(reply.get("id")) is not int or reply["id"] != number:
            raise PluginError(f"the plugin's reply has id {reply.get('id')!r}, not {number}.")
        if not isinstance(reply.get("ok"), bool):
            raise PluginError("the plugin's reply has no ok.")
        return reply


def _open_library(path: Path):
    """dlopen ``path``. Tests replace this to hand over a fake library."""
    return ctypes.CDLL(str(path))


def load_plugin(path: Union[str, Path]) -> Plugin:
    """Load the plugin at ``path`` and check its interface version; a
    PluginError says why it can't be used."""
    path = Path(path)
    try:
        library = _open_library(path)
    except OSError as exc:
        raise PluginError(f"can't load {path}: {exc}") from None
    return Plugin(library, path=path)


__all__ = ["PLUGIN_ABI", "Plugin", "PluginError", "PluginVersionError", "load_plugin"]
