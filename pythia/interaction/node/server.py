"""The node server's HTTP server: on 127.0.0.1 only, behind a token, and read-only.

Every URL starts with the token, 32 random bytes made new for each run and
compared in constant time; any other path is a plain 404. The Host header
must name loopback (127.0.0.1, localhost or [::1]) at any port, so that a
DNS-rebinding page can't name it while an SSH forward from another port
still works. Only GET and HEAD are answered. Every response has
``nosniff``, ``no-store`` and ``no-referrer``, and a CSP: the plain pages'
allows no script; the glui view's pages, under ``/TOKEN/lem/``, run the
server's own scripts and styles, and its facts, bodies and files allow
nothing (``view.py``). At most 8 requests run at once, each socket with a
5 s timeout, as auto's board has them (``pythia/interaction/_auto_board.py``),
and no request is logged.

Every plain page takes ``?style=NAME``, one of the styles that ``css/``
holds (``styles.py``); a name that isn't one is a bad request.
"""

from __future__ import annotations

import gzip
import hmac
import re
import secrets
import socketserver
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, unquote_to_bytes, urlsplit

from .contexts import Roots
from .pages import BadRequest, Pages, Response
from .styles import DEFAULT_STYLE, FIELD, NAME, Styles
from .view import View

ADDRESS = "127.0.0.1"
TOKEN_BYTES = 32
MAX_REQUESTS = 8
SOCKET_TIMEOUT = 5.0
# A body at least this long is compressed, when the browser takes gzip.
GZIP_BYTES = 64 * 1024
MAX_TARGET = 4096
MAX_FIELDS = 8
# A context's views, as query fields (pages.Pages.context); other pages take none.
CONTEXT_FIELDS = frozenset(("line", "lines", "turn", "page", "summary"))
LOOPBACK_HOST = re.compile(r"(?:127\.0\.0\.1|localhost|\[::1\])(?::[0-9]{1,5})?\Z",
                           re.IGNORECASE)
# The plain pages' CSP, and the server's own answers': no script. The glui
# view's responses each give theirs (view.py).
CSP = ("default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
       "form-action 'self'; frame-ancestors 'none'")
HEADERS: Tuple[Tuple[str, str], ...] = (
    ("Cache-Control", "no-store"),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Frame-Options", "DENY"),
)


def new_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def plain(status: int, text: str, headers: Iterable[Tuple[str, str]] = ()) -> Response:
    return Response(status, text + "\n", content_type="text/plain", headers=tuple(headers))


def accepts_gzip(header: Optional[str]) -> bool:
    """Whether an Accept-Encoding header takes gzip, with a q above 0."""
    for coding in (header or "").split(","):
        name, _, parameters = coding.partition(";")
        if name.strip().lower() not in ("gzip", "x-gzip"):
            continue
        quality = 1.0
        for parameter in parameters.split(";"):
            key, _, value = parameter.partition("=")
            if key.strip().lower() == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        return quality > 0
    return False


def _unquoted(text: str) -> str:
    """A path segment as a name: its bytes, percent-decoded, as os.listdir decodes names."""
    return unquote_to_bytes(text.encode("latin-1", "replace")).decode("utf-8", "surrogateescape")


class App:
    """What the node server answers: the routes behind the token, with no sockets involved."""

    def __init__(self, roots: Roots, token: Optional[str] = None, *,
                 style: str = DEFAULT_STYLE, styles: Optional[Styles] = None,
                 view: Optional[View] = None) -> None:
        self.token = token or new_token()
        self._token = self.token.encode("ascii")
        self.pages = Pages(roots, self.token, style=style, styles=styles)
        self.view = View() if view is None else view

    @property
    def path(self) -> str:
        """The contexts page's path: the token, between slashes."""
        return f"/{self.token}/"

    def respond(self, method: str, target: str, hosts: List[str]) -> Response:
        """The response to ``method`` on ``target``, the request line's path as
        http.server gives it (decoded as Latin-1), with ``hosts``, its Host headers."""
        if len(hosts) != 1 or not LOOPBACK_HOST.match(hosts[0].strip()):
            return plain(403, "The node server answers requests to 127.0.0.1, localhost or "
                         "[::1] only.")
        if len(target) > MAX_TARGET:
            return plain(414, "The URL is too long.")
        parts = urlsplit(target)
        segments = parts.path.split("/", 2)
        given = (segments[1] if len(segments) > 1 and not segments[0] and not parts.scheme
                 and not parts.netloc else "")
        if not hmac.compare_digest(given.encode("latin-1", "replace"), self._token):
            return plain(404, "Not found.")
        if method not in ("GET", "HEAD"):
            return plain(405, "The node server only reads: GET and HEAD.",
                         (("Allow", "GET, HEAD"),))
        if len(segments) < 3:
            return Response(308, location=self.path)
        rest = segments[2]
        route, slash, name = rest.partition("/")
        if route == "lem" and not slash:
            return Response(308, location=f"{self.path}lem/")
        pages = self.pages.view(here=(parts.path, []))
        try:
            pairs = parse_qsl(parts.query, keep_blank_values=True, max_num_fields=MAX_FIELDS)
        except ValueError:
            return pages.bad_request("The URL has too many query fields.")
        fields = [(key, value) for key, value in pairs if key != FIELD]
        pages = pages.view(here=(parts.path, fields))
        try:
            pages = pages.view(style=self._style([value for key, value in pairs
                                                  if key == FIELD]))
            params = self._params(fields, CONTEXT_FIELDS if route == "c" else ())
        except BadRequest as exc:
            return pages.bad_request(str(exc))
        if not rest:
            return pages.contexts()
        name = _unquoted(name)
        if route == "c" and name:
            return pages.context(name, params)
        if route == "k" and name:
            return pages.key(name)
        if route == "lem":
            return self.view.respond(pages, name)
        return pages.not_found("The node server has no such page.")

    def _style(self, given: List[str]) -> str:
        """The style that a query's style fields name, else the server's."""
        if not given:
            return self.pages.default
        if len(given) > 1:
            raise BadRequest(f"The query field {FIELD!r} is given twice.")
        if NAME.match(given[0]) is None or self.pages.styles.get(given[0]) is None:
            names = ", ".join(style.name for style in self.pages.styles.all())
            raise BadRequest(f"There's no style {given[0]!r}: give one of {names}, or pick "
                             "one from the menu at the top right.")
        return given[0]

    def _params(self, pairs: List[Tuple[str, str]], allowed: Iterable[str]) -> Dict[str, str]:
        params: Dict[str, str] = {}
        for name, value in pairs:
            if name not in allowed:
                raise BadRequest(f"This page takes no query field {name!r}.")
            if name in params:
                raise BadRequest(f"The query field {name!r} is given twice.")
            params[name] = value
        return params


class Handler(BaseHTTPRequestHandler):
    """Hands each request to the server's app; HTTP/1.0, so one request per connection."""

    server_version = "node"
    sys_version = ""

    def version_string(self) -> str:
        return self.server_version

    def log_message(self, format, *args) -> None:
        pass  # every path holds the token, so no request is logged

    def _answer(self) -> None:
        try:
            response = self.server.app.respond(self.command, self.path,
                                               self.headers.get_all("Host") or [])
        except Exception as exc:
            print("node: a page failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            response = plain(500, "The node server failed on this page: "
                             f"{type(exc).__name__}: {exc}")
        self._send(response)

    do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _answer

    def _send(self, response: Response) -> None:
        # Pages escape every lone surrogate (pages.h); the rest, such as an
        # error's text, shows one as an escape.
        body = response.body.encode("utf-8", "backslashreplace")
        headers: List[Tuple[str, str]] = [
            ("Content-Type", f"{response.content_type}; charset=utf-8"), *HEADERS,
            ("Content-Security-Policy", response.csp or CSP)]
        if response.location is not None:
            headers.append(("Location", response.location))
        headers.extend(response.headers)
        if len(body) >= GZIP_BYTES:
            headers.append(("Vary", "Accept-Encoding"))
            if accepts_gzip(self.headers.get("Accept-Encoding")):
                body = gzip.compress(body, compresslevel=6)
                headers.append(("Content-Encoding", "gzip"))
        headers.append(("Content-Length", str(len(body))))
        self.send_response(response.status)
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


class Server(ThreadingHTTPServer):
    """An app's server on 127.0.0.1: at most 8 requests at once, 5 s socket timeouts.

    Raises OSError when the port can't be had.
    """

    daemon_threads = True  # Ctrl-C doesn't wait for a slow client

    def __init__(self, app: App, port: int) -> None:
        self.app = app
        self.slots = threading.BoundedSemaphore(MAX_REQUESTS)
        super().__init__((ADDRESS, port), Handler)

    def server_bind(self) -> None:
        # HTTPServer's would look the address up in DNS, for a name it never uses.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = ADDRESS, self.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{ADDRESS}:{self.server_port}{self.app.path}"

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(SOCKET_TIMEOUT)
        return connection, address

    def process_request(self, request, address) -> None:
        if not self.slots.acquire(blocking=False):
            try:
                request.settimeout(0.1)
                try:
                    request.recv(MAX_TARGET + 4096)  # its request, so closing sends no reset
                except OSError:
                    pass
                request.sendall(b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n"
                                b"Retry-After: 1\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, address) -> None:
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()

    def handle_error(self, request, address) -> None:
        pass  # a client that went away: no traceback on the terminal
