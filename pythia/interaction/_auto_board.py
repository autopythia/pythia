"""Private, stdlib-only board for the fixed auto app (no session replay)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
from typing import Optional
from urllib.parse import parse_qs, urlsplit
import uuid


MAX_BODY = 1_048_576
MAX_CONTENT = 262_144
MAX_BATCH = 100
_THREAD_PATH = re.compile(r"^/threads/(thread_[0-9a-f]{32})(/messages)?$")


class BoardError(RuntimeError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class BoardPersistenceError(BoardError):
    def __init__(self):
        super().__init__("Board persistence failed; writes are blocked.", 503)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def parse_json(data):
    def invalid_constant(_value):
        raise ValueError("non-finite JSON value")
    return json.loads(data, object_pairs_hook=_unique_object,
                      parse_constant=invalid_constant)


def atomic_text(path: Path, text: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, delete=False) as file:
            temporary = file.name
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


@dataclass(frozen=True)
class Record:
    sequence: int
    record_id: str
    thread_id: str
    author: str
    kind: str
    content: str
    request_id: str
    reply_to: Optional[str] = None
    success: Optional[bool] = None


class Board:
    def __init__(self, directory: Path, *, max_records=4096,
                 max_bytes=16_777_216, max_pending=16):
        self.directory = directory
        self.changed = threading.Condition()
        self._records = []
        self._by_id = {}
        self._threads = {}
        self._requests = {}
        self._terminal = set()
        self._started = set()
        self._bytes = 0
        self._max_records = max_records
        self._max_bytes = max_bytes
        self._max_pending = max_pending
        self.accepting = False
        self.failed = False
        self.view_stale = False
        self._closed = False
        self._journal = (directory / "index.jsonl").open("x", encoding="utf-8")
        try:
            os.chmod(directory / "index.jsonl", 0o600)
            self._journal.flush()
            os.fsync(self._journal.fileno())
            atomic_text(directory / "index.md", self._markdown())
        except BaseException:
            self._journal.close()
            raise

    def records(self, thread_id=None):
        with self.changed:
            return tuple(r for r in self._records
                         if thread_id is None or r.thread_id == thread_id)

    def wake(self):
        with self.changed:
            self.changed.notify_all()

    def wait_input(self, kind, after, stop):
        """Owner-side post-commit wait; not HTTP polling or another write path."""
        with self.changed:
            while not stop.is_set() and not self._closed and not self.failed:
                for record in self._records[after:]:
                    if record.kind == kind:
                        return record
                self.changed.wait()
            return None

    def post(self, author: str, thread_id: Optional[str], body: dict) -> Record:
        if not isinstance(body, dict):
            raise BoardError("Expected a JSON object.")
        fields = {"request_id", "content"} if thread_id is None else {
            "request_id", "content", "kind", "reply_to", "success"}
        if set(body) - fields:
            raise BoardError("Unknown board field.")
        request_id, content = body.get("request_id"), body.get("content")
        if (not isinstance(request_id, str) or
                re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id) is None):
            raise BoardError("A nonempty request_id (letters/digits/_/-) is required.")
        if not isinstance(content, str) or not content.strip():
            raise BoardError("Nonempty content is required.")
        try:
            size = len(content.encode("utf-8"))
        except UnicodeError:
            raise BoardError("Content must be valid UTF-8.") from None
        if size > MAX_CONTENT:
            raise BoardError("Board content is too large.", 413)
        kind = "user" if thread_id is None else body.get("kind")
        allowed = {"user": ("user",), "1": ("plan", "answer"),
                   "2": ("started", "result"), "-1": ()}
        if kind not in allowed.get(author, ()):
            raise BoardError("This author cannot publish that kind of record.", 403)
        reply_to, success = body.get("reply_to"), body.get("success")
        if reply_to is not None and not isinstance(reply_to, str):
            raise BoardError("reply_to must be a record ID.")
        if kind in {"answer", "result"}:
            if not isinstance(success, bool):
                raise BoardError("An outcome requires a boolean success field.")
        elif success is not None:
            raise BoardError("Only outcomes have a success field.")
        fingerprint = (thread_id, kind, content, reply_to, success)
        key = (author, request_id)

        with self.changed:
            prior = self._requests.get(key)
            if prior is not None:
                if prior[0] != fingerprint:
                    raise BoardError("request_id was already used for another request.", 409)
                return prior[1]
            if self.failed:
                raise BoardPersistenceError()
            if self._closed or (kind == "user" and not self.accepting):
                raise BoardError("The session is not accepting new user threads.", 503)
            if thread_id is not None and thread_id not in self._threads:
                raise BoardError("Unknown board thread.", 404)
            if kind != "user":
                parent = self._by_id.get(reply_to)
                expected = "plan" if kind in {"started", "result"} else "user"
                if parent is None or parent.thread_id != thread_id or parent.kind != expected:
                    raise BoardError("reply_to must reference the source record in this thread.")
                if reply_to in self._terminal:
                    raise BoardError("The source record already has an outcome.", 409)
                if kind == "started" and reply_to in self._started:
                    raise BoardError("This plan was already started.", 409)
                if kind == "result" and reply_to not in self._started:
                    raise BoardError("A plan must be started before its result.", 409)
            if kind in {"user", "plan"}:
                pending = sum(r.kind == kind and r.record_id not in self._terminal
                              for r in self._records)
                if pending >= self._max_pending:
                    raise BoardError("Pending work limit reached; request was not accepted.", 429)
            sequence = len(self._records) + 1
            record = Record(sequence, str(sequence),
                            thread_id or "thread_" + uuid.uuid4().hex,
                            author, kind, content, request_id, reply_to, success)
            line = json.dumps(asdict(record), ensure_ascii=False) + "\n"
            encoded_size = len(line.encode("utf-8"))
            if len(self._records) >= self._max_records or self._bytes + encoded_size > self._max_bytes:
                raise BoardError("Board storage limit reached; request was not accepted.", 507)
            try:
                self._journal.write(line)
                self._journal.flush()
                os.fsync(self._journal.fileno())
            except (OSError, ValueError):
                self.failed = True
                self.changed.notify_all()
                raise BoardPersistenceError() from None
            self._bytes += encoded_size
            self._records.append(record)
            self._by_id[record.record_id] = record
            self._requests[key] = (fingerprint, record)
            if kind == "user":
                self._threads[record.thread_id] = record.record_id
            if kind == "started":
                self._started.add(reply_to)
            if kind in {"answer", "result"}:
                self._terminal.add(reply_to)
            # This small MVP serializes projection writes too. A stale view is
            # never grounds to undo a committed post or retry it with a new ID.
            try:
                atomic_text(self.directory / "index.md", self._markdown())
                self.view_stale = False
            except OSError:
                self.view_stale = True
            self.changed.notify_all()
            return record

    def read(self, thread_id, after=0, limit=MAX_BATCH):
        if (isinstance(after, bool) or not isinstance(after, int) or after < 0 or
                isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_BATCH):
            raise BoardError("Invalid after/limit cursor.")
        with self.changed:
            if thread_id not in self._threads:
                raise BoardError("Unknown board thread.", 404)
            records = [r for r in self._records if r.thread_id == thread_id and r.sequence > after]
            batch, byte_count = [], 0
            for record in records[:limit]:
                size = len(json.dumps(asdict(record)).encode("utf-8"))
                if batch and byte_count + size > MAX_BODY:
                    break
                batch.append(record)
                byte_count += size
            return {"thread_id": thread_id, "records": [asdict(r) for r in batch],
                    "next_after": batch[-1].sequence if batch else after,
                    "has_more": len(records) > limit,
                    "high_watermark": len(self._records)}

    def _markdown(self):
        lines = ["# Shared message board", "", f"Rendered through sequence: {len(self._records)}", ""]
        threads = {}
        for record in self._records:
            threads.setdefault(record.thread_id, []).append(record)
        for thread_id, records in threads.items():
            lines.extend((f"## {thread_id}", ""))
            for r in records:
                lines.extend((f"### {r.record_id}: {r.author} / {r.kind}", ""))
                # Fence untrusted data, including embedded fences/HTML/controls.
                value = json.dumps(asdict(r), ensure_ascii=True, indent=2)
                fence = "`" * max(3, 1 + max((len(m[0]) for m in re.finditer(r"`+", value)), default=0))
                lines.extend((fence + "json", value, fence, ""))
        return "\n".join(lines)

    def close(self):
        with self.changed:
            if not self._closed:
                self.accepting = False
                self._closed = True
                self.changed.notify_all()
                self._journal.close()


class _Server(ThreadingHTTPServer):
    # Bounded, non-daemon request threads drain before the journal is closed.
    def __init__(self, board, port):
        self.board = board
        self.tokens = {secrets.token_urlsafe(32): actor for actor in ("user", "1", "2", "-1")}
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(("127.0.0.1", port), _Handler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(5.0)
        return connection, address

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            try:
                request.settimeout(0.1)
                request.sendall(b"HTTP/1.0 503 Busy\r\nContent-Length: 0\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()

    def handle_error(self, request, address):
        pass  # Never dump request bodies/headers or tracebacks to the TTY.


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _reply(self, status, value, content_type="application/json"):
        body = (value if isinstance(value, str) else json.dumps(value)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self, post):
        try:
            base = f"http://127.0.0.1:{self.server.server_port}"
            origins = self.headers.get_all("Origin") or []
            if (self.headers.get_all("Host") != [base[7:]] or
                    (origins and origins != [base])):
                raise BoardError("Invalid Host/Origin.", 403)
            if len(self.path) > 4096:
                raise BoardError("Request path too large.", 414)
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc or parsed.fragment:
                raise BoardError("Invalid request path.")
            if not post and parsed.path == "/README.md" and not parsed.query:
                self._reply(200, _readme(base), "text/markdown")
                return
            auth = self.headers.get_all("Authorization") or []
            actor = self.server.tokens.get(auth[0][7:]) if len(auth) == 1 and auth[0].startswith("Bearer ") else None
            if actor is None:
                raise BoardError("Board authorization required.", 401)
            match = _THREAD_PATH.fullmatch(parsed.path)
            if post:
                if parsed.query or self.headers.get("Transfer-Encoding") is not None:
                    raise BoardError("Unsupported request framing.")
                if self.headers.get_content_type() != "application/json":
                    raise BoardError("Content-Type must be application/json.", 415)
                lengths = self.headers.get_all("Content-Length") or []
                if len(lengths) != 1 or len(lengths[0]) > 10 or not lengths[0].isascii() or not lengths[0].isdigit():
                    raise BoardError("A single Content-Length is required.", 411)
                length = int(lengths[0])
                if not 0 < length <= MAX_BODY:
                    raise BoardError("Request body too large or empty.", 413)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise BoardError("Incomplete request body.")
                try:
                    value = parse_json(body)
                except (ValueError, UnicodeError, RecursionError):
                    raise BoardError("Invalid JSON body.") from None
                if parsed.path == "/threads":
                    record = self.server.board.post(actor, None, value)
                elif match is not None and match[2]:
                    record = self.server.board.post(actor, match[1], value)
                else:
                    raise BoardError("Unknown board route.", 404)
                self._reply(200, asdict(record))
            elif match is not None and not match[2]:
                try:
                    query = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=2)
                    if set(query) - {"after", "limit"} or any(len(v) != 1 for v in query.values()):
                        raise ValueError()
                    after = int(query.get("after", ["0"])[0])
                    limit = int(query.get("limit", [str(MAX_BATCH)])[0])
                except ValueError:
                    raise BoardError("Invalid read cursor.") from None
                self._reply(200, self.server.board.read(match[1], after, limit))
            else:
                raise BoardError("Unknown board route.", 404)
        except BoardError as exc:
            self._reply(exc.status, {"error": str(exc)})
        except (OSError, TimeoutError):
            self.close_connection = True
        except Exception:
            self._reply(500, {"error": "Board request failed; details withheld."})

    def do_GET(self):
        self._dispatch(False)

    def do_POST(self):
        self._dispatch(True)


def _readme(base):
    return f"""# Auto message board (version 1)

Address: {base}

User input creates a new task thread, not a new model/OS thread. Main is #1,
worker #2, watcher #-1. Prefer the bound board_read_thread/board_post_plan tools;
they supply the current thread, author, credentials, and request IDs privately.

* GET /README.md: this public discovery document.
* POST /threads: user only; JSON request_id and content. Creates the thread and
  initial user record atomically and returns its thread_id/record_id.
* GET /threads/<id>?after=0&limit=100: authorized, non-destructive read. Returns
  records, next_after, has_more, and high_watermark. Follow next_after for pages.
* POST /threads/<id>/messages: request_id, kind, content, reply_to; outcomes also
  require boolean success. Main posts plan/answer replying to the initial user
  record. Worker posts started/result replying to a plan in the same thread.
  success describes runtime completion, not independent verification of the work.

Data routes require Authorization: Bearer <private capability>. Capabilities
are host-provided, never provider API keys, URLs, or model arguments. Example
discovery: curl {base}/README.md
Host-side example: curl -H 'Content-Type: application/json' \\
  -H \"Authorization: Bearer $BOARD_CAPABILITY\" \\
  -d '{{"request_id":"example-1","content":"Inspect the repository"}}' {base}/threads

Reuse a request_id with the same body after an uncertain HTTP outcome; conflicting
reuse returns 409. A new ID is new work. Records commit to index.jsonl before
acknowledgement; index.md is a generated view, not editable state. This MVP only
creates fresh saves and does not automatically resume interrupted work.

Limits: {MAX_CONTENT} UTF-8 content bytes, {MAX_BODY} request bytes, {MAX_BATCH}
records per read, 16 pending user tasks/plans each, 4096 records/16 MiB per board,
8 concurrent HTTP handlers. Invalid input is 400, unauthorized 401/403, missing
thread 404, conflicting state 409, excessive size 413, full pending queue 429,
unavailable/persistence failure 503, and full storage 507. No cross-origin access,
streaming waits, arbitrary file routes, or automatic task replay is provided.
"""


class BoardService:
    def __init__(self, directory: Path, *, port=0):
        self.board = Board(directory)
        try:
            self.server = _Server(self.board, port)
        except BaseException:
            self.board.close()
            raise
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.05}, name="auto-board")
        self._closed = False
        try:
            self.thread.start()
        except BaseException:
            self.server.server_close()
            self.board.close()
            raise

    def client(self, actor, timeout=10.0):
        token = next(k for k, v in self.server.tokens.items() if v == actor)
        return BoardClient(self.server.server_port, token, timeout)

    def close(self):
        if not self._closed:
            self._closed = True
            self.server.shutdown()
            self.thread.join()
            self.server.server_close()
            self.board.close()


class BoardClient:
    def __init__(self, port, token, timeout=10.0):
        self.port, self._token, self.timeout = port, token, timeout

    def request(self, method, path, body=None):
        encoded = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
        # A direct numeric loopback connection: no proxy or redirect handling.
        for attempt in range(2):
            connection = HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
            try:
                connection.request(method, path, body=encoded, headers={
                    "Authorization": "Bearer " + self._token,
                    "Content-Type": "application/json",
                })
                response = connection.getresponse()
                data = response.read(2 * MAX_BODY + 1)
                if len(data) > 2 * MAX_BODY:
                    raise BoardError("Board response too large.", 502)
                if response.status != 200:
                    # Don't propagate arbitrary response text into model/tool logs.
                    raise BoardError(f"Board request rejected (HTTP {response.status}).", response.status)
                return parse_json(data)
            except (OSError, HTTPException):
                if attempt:
                    raise BoardError("Board connection failed; outcome may be committed. Do not repost with a new ID.", 503) from None
            finally:
                connection.close()

    def create_thread(self, content, *, request_id=None):
        return self.request("POST", "/threads", {
            "request_id": request_id or uuid.uuid4().hex, "content": content})

    def post(self, thread_id, kind, content, reply_to, *, success=None, request_id=None):
        body = {"request_id": request_id or uuid.uuid4().hex,
                "kind": kind, "content": content, "reply_to": reply_to}
        if success is not None:
            body["success"] = success
        return self.request("POST", f"/threads/{thread_id}/messages", body)

    def read(self, thread_id, *, after=0, limit=20):
        return self.request("GET", f"/threads/{thread_id}?after={after}&limit={limit}")
