"""The node server's HTTP server: on 127.0.0.1 only, behind a token, and reading only,
but for its view's store.

Every URL starts with the token, 32 random bytes made new for each run and
compared in constant time; any other path is a plain 404. The Host header
must name loopback (127.0.0.1, localhost or [::1]) at any port, so that a
DNS-rebinding page can't name it while an SSH forward from another port
still works. GET and HEAD are answered; and POST, which changes the glui
view's store (``store.py``), alone: at ``/TOKEN/lem/store/NAME``, which
replaces its settings, ``/TOKEN/lem/stars/N``, which sets or clears a star in
root N's log, and ``/TOKEN/lem/roots/N``, which sets root N's label and tags;
from a page of the server's own origin (its Origin
header), with the header that only the view's scripts send
(``STORE_HEADER``), as JSON of a few KB at most. Every response has
``nosniff``, ``no-store`` and ``no-referrer``, and a CSP: the plain pages'
allows no script; the glui view's pages, under ``/TOKEN/lem/``, run the
server's own scripts and styles, and its facts, bodies and files allow
nothing (``view.py``). At most 8 requests run at once, each socket with a
5 s timeout, as auto's board has them (``pythia/interaction/_auto_board.py``),
and no request is logged.

Every plain page takes ``?style=NAME``, one of the styles that ``css/``
holds (``styles.py``); a name that isn't one is a bad request.

Behind ``/TOKEN/n/NAME/`` are the pages of remote node NAME (node.ini's
``[remote NAME]``, or ``--remote``; ``remote.py``), as it answers them, a site
within this server's, with its own store: GET, HEAD and its view's POSTs go to
it, which checks them as this server does. What it answers passes on only as
this server's own pages would: as HTML, plain text or JSON, with none of its
headers but Allow, under this server's content security policies, and with
redirects within its own pages; and its view's scripts, styles and programs
are this server's own, so that its pages run nothing that it sent.
"""

from __future__ import annotations

import gzip
import hmac
import json
import re
import secrets
import socketserver
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, unquote_to_bytes, urlsplit

from .contexts import Roots
from .pages import BadRequest, Pages, Response
from .remote import Remote, Way, listings
from .store import FILES, MAX_BYTES, REPLACED, Store
from .styles import DEFAULT_STYLE, FIELD, NAME, Styles
from .view import DATA_CSP, PAGE_CSP, View

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
# What a remote's answer passes on as: its types, and its headers.
REMOTE_TYPES = frozenset(("text/html", "text/plain", "application/json"))
REMOTE_HEADERS = frozenset(("Allow",))
# The header that the glui view's scripts send with a write to its store, which a page
# elsewhere can't send to this server without its leave (a CORS preflight, which it never
# gives).
STORE_HEADER = "X-Pythia-View"
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
    """What the node server answers: the routes behind the token, with no sockets involved.

    With ``base``, a path such as ``/TOKEN/n/NAME``, the routes are behind it instead,
    as a node on its standard streams has them, which another server reaches by ssh
    (``stdio.py``): every URL starts with it, so that its pages link back through that
    server.
    """

    def __init__(self, roots: Roots, token: Optional[str] = None, *,
                 style: str = DEFAULT_STYLE, styles: Optional[Styles] = None,
                 view: Optional[View] = None, store: Optional[Store] = None,
                 base: Optional[str] = None,
                 remotes: Sequence[Sequence] = ()) -> None:
        self.token = token or new_token()
        self.base = f"/{self.token}" if base is None else base
        self._base = self.base.encode("latin-1", "replace")
        self._depth = self.base.count("/")  # the base's segments
        self.pages = Pages(roots, self.token, style=style, styles=styles, base=self.base)
        self.view = View(store=store) if view is None else view
        self.store = store
        # The remote nodes, each by its name, as it's reached (a Way, or its fields), whose
        # roots the index shows after this node's.
        ways = [Way(*each) for each in remotes]
        self.remotes: Dict[str, Remote] = {
            way.name: Remote(way.name, way.argv, f"{self.base}/n/{way.name}",
                             timeout=way.timeout) for way in ways}
        if self.remotes:
            self.pages.listings = lambda: listings(list(self.remotes.values()))

    def close(self) -> None:
        """Ends the remotes' processes."""
        for remote in self.remotes.values():
            remote.close()

    @property
    def path(self) -> str:
        """The contexts page's path: the token, or the base, and a slash."""
        return f"{self.base}/"

    def respond(self, method: str, target: str, hosts: List[str], *,
                origins: Sequence[str] = (), marked: bool = False,
                content_type: Optional[str] = None, body: Optional[bytes] = None) -> Response:
        """The response to ``method`` on ``target``, the request line's path as
        http.server gives it (decoded as Latin-1), with ``hosts``, its Host headers; and,
        for a POST, ``origins``, its Origin headers, whether it has ``STORE_HEADER``, its
        content type, and its body, or None if it was too long to read."""
        if len(hosts) != 1 or not LOOPBACK_HOST.match(hosts[0].strip()):
            return plain(403, "The node server answers requests to 127.0.0.1, localhost or "
                         "[::1] only.")
        if len(target) > MAX_TARGET:
            return plain(414, "The URL is too long.")
        parts = urlsplit(target)
        depth = self._depth
        segments = parts.path.split("/", depth + 1)
        given = ("/".join(segments[:depth + 1]) if len(segments) > depth and not segments[0]
                 and not parts.scheme and not parts.netloc else "")
        if not hmac.compare_digest(given.encode("latin-1", "replace"), self._base):
            return plain(404, "Not found.")
        rest = segments[depth + 1] if len(segments) > depth + 1 else ""
        if rest.startswith("n/"):
            return self._remote(method, target, rest[2:], hosts[0].strip(), origins, marked,
                                content_type, body)
        stored = rest[len("lem/store/"):] if rest.startswith("lem/store/") else None
        starred = rest[len("lem/stars/"):] if rest.startswith("lem/stars/") else None
        rooted = rest[len("lem/roots/"):] if rest.startswith("lem/roots/") else None
        own = (hosts[0].strip(), origins, marked, content_type, body)
        if method == "POST" and stored in REPLACED and not parts.query:
            return self._refused(*own) or self._keep(stored, body or b"")
        if method == "POST" and starred and not parts.query:
            return self._refused(*own) or self._star(_unquoted(starred), body or b"")
        if method == "POST" and rooted and not parts.query:
            return self._refused(*own) or self._root(_unquoted(rooted), body or b"")
        if method not in ("GET", "HEAD"):
            return plain(405, "The node server reads, with GET and HEAD; it takes POST for its "
                         "view's store alone.", (("Allow", "GET, HEAD"),))
        if len(segments) < depth + 2:
            return Response(308, location=self.path)
        if stored is not None and not parts.query:
            if stored not in FILES:
                return plain(404, "The view's store has no such file.")
            kept = {} if self.store is None else self.store.read(stored)
            return Response(200, json.dumps(kept, sort_keys=True),
                            content_type="application/json", csp=DATA_CSP)
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

    def _remote(self, method: str, target: str, rest: str, host: str,
                origins: Sequence[str], marked: bool, content_type: Optional[str],
                body: Optional[bytes]) -> Response:
        """The answer of remote node NAME, ``rest`` being ``NAME/...``; the view's files,
        this server's own."""
        name, _, path = rest.partition("/")
        remote = self.remotes.get(name)
        if remote is None:
            return plain(404, "There's no such remote node here.")
        if path.startswith("lem/files/") and method in ("GET", "HEAD"):
            return self.view.respond(self.pages, _unquoted(path[len("lem/"):]))
        answer = remote.request(method, target, host, origins=origins, marked=marked,
                                content_type=content_type, body=body)
        within = answer.location is None or (answer.location + "?").startswith(
            (f"{remote.base}/", f"{remote.base}?"))
        if answer.content_type not in REMOTE_TYPES or not within:
            return plain(502, f"The remote node {name} answered with what this server "
                         "doesn't pass on.")
        return Response(answer.status, answer.body, content_type=answer.content_type,
                        location=answer.location,
                        headers=[header for header in answer.headers
                                 if header[0] in REMOTE_HEADERS],
                        csp=answer.csp if answer.csp in (PAGE_CSP, DATA_CSP) else None)

    def _refused(self, host: str, origins: Sequence[str], marked: bool,
                 content_type: Optional[str], body: Optional[bytes]) -> Optional[Response]:
        """Why a POST to the view's store is refused, or None: it must come from a page of
        the server's own origin, the one that ``host``, the Host header, names, with
        ``STORE_HEADER``, as JSON of ``MAX_BYTES`` at most; and the server must keep a
        store."""
        if len(origins) != 1 or origins[0].strip().lower() != f"http://{host}".lower():
            return plain(403, "The view's store takes POST from the node server's own pages "
                         "only.")
        if not marked:
            return plain(403, "The view's store takes POST from the view's own scripts only.")
        if (content_type or "").split(";")[0].strip().lower() != "application/json":
            return plain(415, "The view's store takes JSON.")
        if body is None or len(body) > MAX_BYTES:
            return plain(413, f"The view's store takes {MAX_BYTES:,} bytes at most.")
        if self.store is None:
            return plain(404, "This node server keeps no store.")
        return None

    def _keep(self, name: str, body: bytes) -> Response:
        """A POST that replaces file ``name`` of the view's store with its JSON, which the
        store checks."""
        return self._stored(lambda store: store.write(name, json.loads(body.decode("utf-8"))))

    def _star(self, number: str, body: bytes) -> Response:
        """A POST that sets or clears a star in root ``number``'s log: ``{"line": LINE,
        "on": BOOL}``, kept by the root's key."""
        context = self.pages.roots.get(number)
        if context is None:
            return plain(404, "There's no such root context here.")

        def star(store: Store) -> None:
            value = json.loads(body.decode("utf-8"))
            line = value.get("line") if isinstance(value, dict) else None
            on = value.get("on") if isinstance(value, dict) else None
            if type(line) is not int or line < 1 or not isinstance(on, bool):
                raise ValueError('a star is {"line": LINE, "on": BOOL}')
            store.star(context.key, line, on)

        return self._stored(star)

    def _root(self, number: str, body: bytes) -> Response:
        """A POST that sets root ``number``'s label and tags: ``{"label": TEXT, "tags":
        [TAG, ...]}``, kept by the root's key."""
        context = self.pages.roots.get(number)
        if context is None:
            return plain(404, "There's no such root context here.")
        return self._stored(lambda store: store.root(context.key,
                                                     json.loads(body.decode("utf-8"))))

    def _stored(self, change) -> Response:
        """The response to a change of the view's store: 204 once it's made."""
        try:
            change(self.store)
        except ValueError as exc:  # not JSON, not UTF-8, or not what the store keeps
            return plain(400, f"The view's store doesn't keep that: {exc}")
        except OSError as exc:
            print(f"node: the view's store couldn't be written: {exc}", file=sys.stderr)
            return plain(500, "The view's store couldn't be written.")
        return Response(204, content_type="text/plain", csp=DATA_CSP)

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
            post = self._post() if self.command == "POST" else {}
            response = self.server.app.respond(self.command, self.path,
                                               self.headers.get_all("Host") or [], **post)
        except Exception as exc:
            print("node: a page failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            response = plain(500, "The node server failed on this page: "
                             f"{type(exc).__name__}: {exc}")
        self._send(response)

    do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _answer

    def _post(self) -> dict:
        """What App.respond takes of a POST: its Origin headers, whether it has
        STORE_HEADER, its content type, and its body, read only if it says how long it is,
        and that's at most MAX_BYTES; else None."""
        length = (self.headers.get("Content-Length") or "").strip()
        body = None
        if length.isdigit() and int(length) <= MAX_BYTES:
            body = self.rfile.read(int(length))
            body = body if len(body) == int(length) else None
        return {"origins": self.headers.get_all("Origin") or [],
                "marked": self.headers.get(STORE_HEADER) is not None,
                "content_type": self.headers.get("Content-Type"), "body": body}

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
