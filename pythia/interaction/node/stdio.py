"""The node server on its standard streams (``--stdio``): its pages as lines of JSON, for a
node server elsewhere that reaches this node by ssh (``--remote``), with no port and no
token, so that ssh's own login is the only way in.

The first line in says the protocol, ``PROTOCOL``, and the base path that the pages' links
start with, such as ``/TOKEN/n/NAME``, so that they lead back through the server that
asks: ``{"node": 1, "base": BASE}``. It's answered with ``{"node": 1, "host": HOSTNAME,
"roots": N}``, or with ``{"node": 1, "error": WHY}``, and the end. Each line after is a
request, ``{"id": ID, "method": METHOD, "target": TARGET, "host": HOST}``: the request
line's path, as http.server gives it, and the Host header, as the server that asks got
them; with a POST's ``origins``, ``marked``, ``type`` and ``body``, in base64, or null if
it was too long to read (``App.respond``). Each is answered with a line, in order:
``{"id": ID, "status": STATUS, "type": TYPE, "body": BODY, "location": LOCATION,
"headers": [[NAME, VALUE], ...], "csp": CSP}``, with an ``id`` of null for a line that
isn't a request. The end of the input is the end. What goes wrong goes to standard
error.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import socket
import sys
import traceback
from typing import BinaryIO, Optional, Tuple

from .contexts import Roots
from .pages import Response
from .server import MAX_TARGET, App, plain
from .store import MAX_BYTES, Store
from .styles import DEFAULT_STYLE, Styles

PROTOCOL = 1
# A request's line at most: its target, and a POST's body in base64.
MAX_LINE = 2 * MAX_TARGET + 2 * MAX_BYTES
# A base: a path of a few segments, each of characters that need no quoting in a URL, but
# not dots alone, which a browser would take away.
BASE = re.compile(r"(?:/(?!\.+(?:/|\Z))[A-Za-z0-9_.~-]+){1,8}\Z")
MAX_BASE = 256


def serve(roots: Roots, *, store: Optional[Store] = None, style: str = DEFAULT_STYLE,
          styles: Optional[Styles] = None, stdin: Optional[BinaryIO] = None,
          stdout: Optional[BinaryIO] = None) -> int:
    """Answers requests from ``stdin`` on ``stdout`` until ``stdin`` ends: 0 then, or 1 if
    its first line isn't one that this node takes."""
    stdin = sys.stdin.buffer if stdin is None else stdin
    stdout = sys.stdout.buffer if stdout is None else stdout

    def send(value: dict) -> None:
        stdout.write(json.dumps(value).encode("ascii") + b"\n")
        stdout.flush()

    base, why = hello(_line(stdin))
    if base is None:
        send({"node": PROTOCOL, "error": why})
        print(f"node: {why}", file=sys.stderr)
        return 1
    app = App(roots, style=style, styles=styles, store=store, base=base)
    send({"node": PROTOCOL, "host": socket.gethostname(), "roots": len(roots.entries())})
    while True:
        line = _line(stdin)
        if line is None:
            return 0
        send(answer(app, line))


def _line(stdin: BinaryIO) -> Optional[bytes]:
    """The next line, without its end; b"" in place of one longer than MAX_LINE, which is
    read to its end; or None at the end of the input."""
    line = stdin.readline(MAX_LINE + 1)
    if not line:
        return None
    if line.endswith(b"\n"):
        return line[:-1]
    if len(line) <= MAX_LINE:
        return line  # the last line, without an end
    while line and not line.endswith(b"\n"):
        line = stdin.readline(MAX_LINE + 1)
    return b""


def hello(line: Optional[bytes]) -> Tuple[Optional[str], Optional[str]]:
    """The base that a first line gives, and None; or None, and why it isn't one that this
    node takes."""
    try:
        first = json.loads(line or b"")
    except ValueError:
        first = None
    if not isinstance(first, dict) or "node" not in first:
        return None, "the first line isn't a node's: it says {\"node\": 1, \"base\": BASE}"
    if first["node"] != PROTOCOL:
        return None, (f"this node speaks protocol {PROTOCOL}, not "
                      f"{json.dumps(first['node'])[:20]}")
    base = first.get("base")
    if not isinstance(base, str) or len(base) > MAX_BASE or not BASE.match(base):
        return None, "the base isn't a path, such as /TOKEN/n/NAME"
    return base, None


def answer(app: App, line: bytes) -> dict:
    """The answer to a request's line."""
    try:
        request = json.loads(line)
    except ValueError:
        request = None
    rid = request.get("id") if isinstance(request, dict) else None
    if type(rid) is not int:
        return _answered(None, plain(400, "The line isn't a request: it's JSON, with an id."))
    method, target, host = (request.get(name) for name in ("method", "target", "host"))
    origins = request.get("origins", [])
    content_type = request.get("type")
    if (not all(isinstance(value, str) for value in (method, target, host))
            or not isinstance(origins, list)
            or not all(isinstance(origin, str) for origin in origins)
            or not isinstance(request.get("marked", False), bool)
            or not isinstance(content_type, (str, type(None)))):
        return _answered(rid, plain(400, "The request's fields aren't what a request has."))
    body = request.get("body")
    if body is not None:
        try:
            body = base64.b64decode(body, validate=True) if isinstance(body, str) else None
        except (ValueError, binascii.Error):
            body = None
        if body is None:
            return _answered(rid, plain(400, "The request's body isn't in base64."))
    try:
        response = app.respond(method, target, [host], origins=origins,
                               marked=request.get("marked", False),
                               content_type=content_type, body=body)
    except Exception as exc:
        print("node: a page failed:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        response = plain(500, "The node server failed on this page: "
                         f"{type(exc).__name__}: {exc}")
    return _answered(rid, response)


def _answered(rid: Optional[int], response: Response) -> dict:
    return {"id": rid, "status": response.status, "type": response.content_type,
            "body": response.body, "location": response.location,
            "headers": [list(header) for header in response.headers], "csp": response.csp}
