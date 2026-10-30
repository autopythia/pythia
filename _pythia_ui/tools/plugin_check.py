#!/usr/bin/env python3
"""Load the plugin with ctypes, as pythia will, and check its interface.

usage: plugin_check.py [--screenshot OUT.png [--capture PATH] [--display ID]
                        [--no-cursor]] PLUGIN

It checks the ABI version, `info`, some malformed requests, and that 20,000
calls leave memory flat (each reply is freed). Where the plugin can't take
screenshots, it checks that a screenshot is unsupported; on a Mac,
--screenshot OUT.png takes one and writes it out. Run it with the Python that
runs pythia, which is what loads the plugin. Exits 1 if a check fails.
"""

import argparse
import base64
import ctypes
import json
import resource
import struct
import sys

ABI_VERSION = 1
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class Plugin:
    """The plugin's three C functions, through ctypes."""

    def __init__(self, path):
        lib = ctypes.CDLL(path)
        reply = ctypes.POINTER(ctypes.c_uint8)
        lib.pythia_plugin_abi.argtypes = []
        lib.pythia_plugin_abi.restype = ctypes.c_uint32
        lib.pythia_plugin_call.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(reply), ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.pythia_plugin_call.restype = ctypes.c_int32
        lib.pythia_plugin_free.argtypes = [reply, ctypes.c_size_t]
        lib.pythia_plugin_free.restype = None
        self.lib = lib

    def abi(self):
        return self.lib.pythia_plugin_abi()

    def raw(self, request):
        """One request's bytes in, the reply's bytes out."""
        reply, length = ctypes.POINTER(ctypes.c_uint8)(), ctypes.c_size_t()
        status = self.lib.pythia_plugin_call(request, len(request), ctypes.byref(reply),
                                             ctypes.byref(length))
        if status != 0:
            raise RuntimeError(f"pythia_plugin_call returned {status}")
        try:
            return ctypes.string_at(reply, length.value)
        finally:
            self.lib.pythia_plugin_free(reply, length.value)

    def call(self, request):
        return json.loads(self.raw(json.dumps(request).encode()))


failed = []


def check(name, ok, detail=None):
    print(f"{'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        failed.append(name)
        if detail is not None:
            print(f"     {detail}")


def fields(reply):
    return json.dumps({key: value for key, value in reply.items() if key not in ("id", "ok")})


def max_rss():
    """The process's peak memory, in bytes (macOS counts bytes, Linux KiB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("plugin", help="the plugin: libpythia_ui.dylib, or a .so on Linux")
    parser.add_argument("--screenshot", metavar="OUT.png", help="take a screenshot and write it")
    parser.add_argument("--capture", default="auto", help="auto, screencapturekit or coregraphics")
    parser.add_argument("--display", type=int, help="a display ID from info (default: the main one)")
    parser.add_argument("--no-cursor", action="store_true", help="leave the cursor out")
    args = parser.parse_args()

    plugin = Plugin(args.plugin)
    print(f"loaded {args.plugin} into Python {sys.version.split()[0]} ({sys.executable})")
    check("ABI version", plugin.abi() == ABI_VERSION, f"{plugin.abi()}, not {ABI_VERSION}")
    info = plugin.call({"id": 1, "op": "info"})
    check("info", info.get("ok") is True and info.get("id") == 1 and "info" in info.get("ops", []),
          info)
    print(f"     {fields(info)}")
    for request, kind in [
        (b"", "request"),
        (b'{"op":', "request"),
        (b'{"op": "click"}', "unsupported"),
        (b'{"op": "screenshot", "cursor": "yes"}', "request"),
    ]:
        reply = json.loads(plugin.raw(request))
        check(f"refuses {request.decode()!r} ({kind})",
              reply.get("ok") is False and reply.get("kind") == kind, reply)

    big = json.dumps({"id": "x" * 4096, "op": "click"}).encode()
    for _ in range(1000):
        plugin.raw(big)
    before = max_rss()
    for _ in range(20000):
        plugin.raw(big)
    grown = max_rss() - before
    check("20,000 calls with 4 KiB replies leave memory flat", grown < 8 << 20,
          f"peak memory grew {grown >> 10} KiB")

    if "screenshot" not in info.get("ops", []):
        reply = plugin.call({"id": 2, "op": "screenshot"})
        check("a screenshot is unsupported here", reply.get("kind") == "unsupported", reply)
    elif args.screenshot:
        request = {"id": 2, "op": "screenshot", "capture": args.capture, "cursor": not args.no_cursor}
        if args.display is not None:
            request["display"] = args.display
        reply = plugin.call(request)
        if not reply.get("ok"):
            check("screenshot", False, f"{reply.get('kind')}: {reply.get('error')}")
        else:
            png = base64.b64decode(reply.pop("png"))
            with open(args.screenshot, "wb") as out:
                out.write(png)
            size = struct.unpack(">II", png[16:24]) if png[12:16] == b"IHDR" else None
            check("screenshot: a PNG of the reply's size",
                  png.startswith(PNG_SIGNATURE) and len(png) == reply["bytes"]
                  and size == (reply["width"], reply["height"]),
                  f"{len(png)} bytes, IHDR size {size}")
            print(f"     {fields(reply)} -> {args.screenshot}")
    else:
        print("     (screenshot: pass --screenshot OUT.png to take one)")

    if failed:
        print(f"{len(failed)} check{'s' if len(failed) != 1 else ''} failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
