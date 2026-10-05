"""Opt-in HTTP and incremental event traces; never resume authority.

Each traced exchange appends one ``http_request`` line to ``STEM.trace.req.jsonl``
before the request is sent, and one ``http_response`` line to
``STEM.trace.res.jsonl`` once the response is closed, fails, or is rejected
with an HTTP error status (the dual-log convention of contradex tracing).
``STEM`` is the save path without its extension, so ``review.jsonl`` is traced
to ``review.trace.req.jsonl`` and ``review.trace.res.jsonl``.
Payloads and headers are recorded exactly as sent or read, including
credentials. The logs are append-only and are never read back.

The tracer wraps an injectable opener, so the caller observes the same
statuses, headers, body bytes, and exceptions as without tracing. It never
raises its own errors into the caller: adapters treat an ``OSError`` from the
opener as a retryable connection failure. After its first internal failure,
tracing stops and a one-time warning is kept for the frontend to report.

Frontends also opt into ``STEM.trace.events.jsonl`` for lifecycle/native/MCP
events. A byte-bounded queue keeps its writer off the native IO paths. HTTP
payloads remain verbatim; selected MCP metadata deliberately omits auth headers.
All payload-bearing traces are sensitive, including raw native stdout/stderr.
"""

from __future__ import annotations

import base64
from collections import deque
from contextlib import contextmanager
import contextvars
from datetime import datetime
from datetime import timezone
import io
import json
import os
from pathlib import Path
import stat
import threading
import time
import traceback
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Optional
import urllib.error
import uuid


TRACE_REQUEST_SUFFIX = ".trace.req.jsonl"
TRACE_RESPONSE_SUFFIX = ".trace.res.jsonl"
TRACE_EVENT_SUFFIX = ".trace.events.jsonl"
MAX_EVENT_QUEUE_BYTES = 8 * 1024 * 1024
MAX_EVENT_QUEUE_ITEMS = 512


def debug_trace_paths(save_path) -> tuple[Path, Path]:
    """Return the request and response log paths derived from a save path.

    The trace suffixes replace the save's extension (``review.jsonl`` gives
    ``review.trace.req.jsonl``); a save without an extension gets them appended.
    """
    path = Path(save_path)
    return (
        path.with_suffix(TRACE_REQUEST_SUFFIX),
        path.with_suffix(TRACE_RESPONSE_SUFFIX),
    )


def _timestamp() -> str:
    # The contradex trace Timestamp format (UTC ISO 8601 with a "Z" suffix),
    # with microseconds always present.
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class _Operation:
    """Retry counters for the exchanges of one traced frontend operation."""

    def __init__(self, op: str, tags=None) -> None:
        self.op = op
        self.id = uuid.uuid4().hex
        self.tags = dict(tags or {})
        self._lock = threading.Lock()
        self._counts: dict[Optional[str], int] = {}

    def next_retry(self, op: Optional[str]) -> int:
        with self._lock:
            retry = self._counts.get(op, 0)
            self._counts[op] = retry + 1
            return retry


_OPERATION: contextvars.ContextVar[Optional[_Operation]] = contextvars.ContextVar(
    "pythia_debug_trace_operation",
    default=None,
)


@contextmanager
def trace_operation(op: str, **tags) -> Iterator[None]:
    """Tag exchanges started in this context (including ``asyncio.to_thread``).

    Within one operation, ``retry`` counts the preceding exchanges with the
    same ``op``, starting at 0. Outside any operation, ``op`` and ``retry``
    are null unless the opener has a fixed ``op``.
    """
    if not isinstance(op, str) or not op:
        raise ValueError("trace operation must be a non-empty string")
    token = _OPERATION.set(_Operation(op, tags))
    try:
        yield
    finally:
        _OPERATION.reset(token)


def capture_trace_scope():
    """Copy operation metadata for explicit transfer to persistent IO threads."""
    scope = _OPERATION.get()
    return ({"op": None, "operation_id": None} if scope is None else
            {**scope.tags, "op": scope.op, "operation_id": scope.id})


def _private_opener(path, flags):
    if Path(path).is_symlink():
        raise ValueError(f"debug trace destination must not be a symlink: {path}")
    fd = os.open(path, flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_NONBLOCK", 0), 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"debug trace destination must be a regular file: {path}")
        if os.name == 'posix' and (info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise ValueError(f"debug trace destination must be owned by this user and private (0600): {path}")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _prepare_log(path: Path) -> None:
    """Create one log (0600) and terminate a line left partial by a crash."""
    if path.exists() and not path.is_file():
        raise ValueError(f"debug trace destination must be a regular file: {path}")
    try:
        descriptor = _private_opener(path, os.O_RDWR | os.O_CREAT | os.O_APPEND)
    except OSError as exc:
        raise ValueError(
            f"cannot open debug trace {path}: {exc.strerror or exc}"
        ) from None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(
                f"debug trace destination must be a regular file: {path}"
            )
        if info.st_size:
            os.lseek(descriptor, -1, os.SEEK_END)
            if os.read(descriptor, 1) != b"\n":
                # O_APPEND writes at the end regardless of the read offset.
                os.write(descriptor, b"\n")
    except OSError as exc:
        raise ValueError(
            f"cannot prepare debug trace {path}: {exc.strerror or exc}"
        ) from None
    finally:
        os.close(descriptor)


def _header_pairs(headers: Any) -> Optional[list[list[str]]]:
    """Verbatim ordered header pairs, including duplicates; None if absent."""
    if headers is None:
        return None
    try:
        items = headers.items() if hasattr(headers, "items") else headers
        return [[str(name), str(value)] for name, value in items]
    except Exception:
        return None


def _payload_fields(data: Any) -> dict[str, Any]:
    if data is None:
        return {"payload": None}
    if isinstance(data, str):
        return {"payload": data}
    if not isinstance(data, (bytes, bytearray, memoryview)):
        # Iterable or file request bodies are not captured.
        return {"payload": None}
    raw = bytes(data)
    try:
        return {"payload": raw.decode("utf-8")}
    except UnicodeDecodeError:
        return {
            "payload": None,
            "payload_base64": base64.b64encode(raw).decode("ascii"),
        }


def _exception_fields(exc: BaseException) -> dict[str, str]:
    try:
        value = str(exc)
    except Exception:
        value = f"<unprintable {type(exc).__name__}>"
    try:
        formatted = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    except Exception:
        formatted = ""
    return {
        "exc_type": type(exc).__name__,
        "exc_val": value,
        "exc_tb": formatted,
    }


class _FailingReader(io.RawIOBase):
    """Re-serve an error body whose original read failed."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self._error = error

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        raise self._error


class DebugTrace:
    """Append-only HTTP logs and optional event sidecar for one context save."""

    def __init__(self, request_path, response_path, *, event_path=None, run_id=None,
                 context_id=None, role=None) -> None:
        self.request_path = Path(request_path)
        self.response_path = Path(response_path)
        self._lock = threading.RLock()
        self._write_lock = threading.RLock()
        self._failed = False
        self._warning: Optional[str] = None
        self.event_path = None if event_path is None else Path(event_path)
        self.run_id = run_id or uuid.uuid4().hex
        self.context_id, self.role = context_id, role
        self._event_condition = threading.Condition()
        self._event_queue = deque()
        self._event_bytes = 0
        self._event_sequence = 0
        self._closed = False
        self._writer = None

    @classmethod
    def open(cls, save_path, *, events=False, run_id=None, context_id=None, role=None) -> "DebugTrace":
        """Prepare private logs before model construction; optionally start an event writer."""
        trace = cls(*debug_trace_paths(save_path),
                    event_path=Path(save_path).with_suffix(TRACE_EVENT_SUFFIX) if events else None,
                    run_id=run_id, context_id=context_id, role=role)
        for path in (trace.request_path, trace.response_path, trace.event_path):
            if path is None:
                continue
            _prepare_log(path)
        if events:
            trace._writer = threading.Thread(target=trace._write_events, name='debug-trace-writer', daemon=True)
            trace._writer.start()
        return trace

    def identity(self):
        return {"run_id": self.run_id, "context_id": self.context_id, "role": self.role}

    @contextmanager
    def operation(self, op, **tags):
        started = time.monotonic()
        error = None
        with trace_operation(op, **tags):
            self.event('operation_begin')
            try:
                yield
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                self.event('operation_end', exception_type=error, elapsed_seconds=time.monotonic() - started)

    def event(self, kind, *, scope=None, data=None, **fields):
        """Best-effort incremental capture, never blocking on filesystem IO.

        ``data`` is private opt-in payload, not safe ModelFailure metadata.
        Scope must be passed explicitly by persistent/background runtimes.
        """
        if self.event_path is None or self._failed or self._closed:
            return
        try:
            row = {**self.identity(), **(capture_trace_scope() if scope is None else scope), **fields,
                   "schema": "pythia.debug-event.v1", "type": kind,
                   "timestamp": _timestamp(), "monotonic_seconds": time.monotonic()}
            if data is not None:
                row.update(_payload_fields(data))
            with self._event_condition:
                if self._closed or self._failed:
                    return
                self._event_sequence += 1
                row['sequence'] = self._event_sequence
                line = (json.dumps(row, ensure_ascii=True, allow_nan=False) + '\n').encode()
                if (len(line) + self._event_bytes > MAX_EVENT_QUEUE_BYTES
                        or len(self._event_queue) >= MAX_EVENT_QUEUE_ITEMS):
                    self._fail('Warning: debug trace disabled; event queue full; trace is incomplete.')
                    return
                self._event_queue.append(line)
                self._event_bytes += len(line)
                self._event_condition.notify()
        except Exception:
            self._fail('Warning: debug trace disabled after an event-recording failure; trace is incomplete.')

    def _write_events(self):
        while True:
            with self._event_condition:
                while not self._event_queue and not self._closed:
                    self._event_condition.wait()
                if not self._event_queue:
                    return
                line = self._event_queue.popleft()
                # Count the in-flight row against the bound until the write ends.
            if not self._failed:
                try:
                    with open(self.event_path, 'ab', opener=_private_opener) as stream:
                        stream.write(line)
                except Exception as exc:
                    self._fail(f'Warning: debug trace disabled; event write failed ({type(exc).__name__}); trace is incomplete.')
            with self._event_condition:
                self._event_bytes -= len(line)

    def close(self, timeout=2):
        """Call after model/IO cleanup. Never indefinitely join a blocked writer."""
        with self._event_condition:
            self._closed = True
            self._event_condition.notify_all()
        if self._writer is not None:
            self._writer.join(timeout)
            if self._writer.is_alive():
                self._fail('Warning: debug trace shutdown timed out; trace may be incomplete.')

    @property
    def enabled(self) -> bool:
        return not self._failed and not self._closed

    def opener(
        self,
        inner: Callable[..., Any],
        *,
        op: Optional[str] = None,
    ) -> "TracingOpener":
        """Wrap ``inner``; a fixed ``op`` overrides the ambient operation tag."""
        if isinstance(inner, TracingOpener) and inner.trace is self:
            if op is None or op == inner.op:
                return inner
            inner = inner.inner
        return TracingOpener(self, inner, op=op)

    def take_warning(self) -> Optional[str]:
        """Return the one-time failure warning, if any, and clear it."""
        with self._lock:
            warning, self._warning = self._warning, None
            return warning

    def _fail(self, message: str) -> None:
        with self._lock:
            if not self._failed:
                self._failed = True
                self._warning = message

    def _write(self, path: Path, row: dict[str, Any]) -> None:
        with self._write_lock:
            if self._failed:
                return
            try:
                line = json.dumps(row, ensure_ascii=False) + "\n"
                # A lone surrogate (from undecodable argv or exception text)
                # is written as a backslash escape, a valid JSON string escape.
                with open(
                    path,
                    "a",
                    encoding="utf-8",
                    errors="backslashreplace",
                    newline="\n",
                    opener=_private_opener,
                ) as stream:
                    stream.write(line)
            except Exception as exc:
                self._fail(
                    f"Warning: debug trace disabled; could not write {path}: {exc}"
                )

    def _begin(self, request: Any, fixed_op: Optional[str]) -> Optional["_Exchange"]:
        if self._failed or self._closed:
            return None
        try:
            scope = _OPERATION.get()
            op = fixed_op if fixed_op is not None else (
                None if scope is None else scope.op
            )
            retry = None if scope is None else scope.next_retry(op)
            get_method = getattr(request, "get_method", None)
            header_items = getattr(request, "header_items", None)
            exchange = _Exchange(
                self,
                op=op,
                retry=retry,
                method=get_method() if callable(get_method) else None,
                url=(
                    request if isinstance(request, str)
                    else getattr(request, "full_url", None)
                ),
            )
            exchange.write_request(
                _header_pairs(header_items()) if callable(header_items) else None,
                None if isinstance(request, str) else getattr(request, "data", None),
            )
        except Exception as exc:
            self._fail(
                "Warning: debug trace disabled after an internal failure "
                f"({type(exc).__name__})."
            )
            return None
        return exchange


class _Exchange:
    def __init__(
        self,
        trace: DebugTrace,
        *,
        op: Optional[str],
        retry: Optional[int],
        method: Optional[str],
        url: Optional[str],
    ) -> None:
        self._trace = trace
        self._scope = capture_trace_scope()
        self._op = op
        self._id = uuid.uuid4().hex
        self._retry = retry
        self._t0 = _timestamp()
        self._method = method
        self._url = url
        self._finished = False
        self._lock = threading.Lock()

    def _common(self) -> dict[str, Any]:
        return {**self._trace.identity(), **self._scope, "op": self._op, "id": self._id, "retry": self._retry}

    def write_request(self, headers: Any, data: Any) -> None:
        row = {
            "type": "http_request",
            **self._common(),
            "t0": self._t0,
            "method": self._method,
            "url": self._url,
            "headers": headers,
        }
        row.update(_payload_fields(data))
        self._trace._write(self._trace.request_path, row)

    def finish(
        self,
        *,
        status: Any = None,
        headers: Any = None,
        body: Any = None,
        exception: Optional[BaseException] = None,
    ) -> None:
        """Write the single response line for this exchange, never raising."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
        try:
            row = {
                "type": "http_response",
                **self._common(),
                "t0": self._t0,
                "t1": _timestamp(),
                "method": self._method,
                "url": self._url,
                "status": (
                    status
                    if isinstance(status, int) and not isinstance(status, bool)
                    else None
                ),
                "headers": headers,
            }
            row.update(_payload_fields(body))
            row["exception"] = (
                None if exception is None else _exception_fields(exception)
            )
            self._trace._write(self._trace.response_path, row)
        except Exception as exc:
            self._trace._fail(
                "Warning: debug trace disabled after an internal failure "
                f"({type(exc).__name__})."
            )

    def http_error(self, exc: urllib.error.HTTPError) -> BaseException:
        """Log a complete HTTP error response; return an equivalent to raise.

        The entire error body is read here (the adapters read at most a
        bounded prefix) and re-served unchanged, so the caller's handling of
        the status, headers, and body is the same as without tracing.
        """
        body = b""
        read_error: Optional[BaseException] = None
        try:
            body = exc.read()
        except Exception as error:
            read_error = error
            # http.client.IncompleteRead retains the bytes that did arrive.
            partial = getattr(error, "partial", None)
            body = partial if isinstance(partial, (bytes, bytearray)) else b""
        self.finish(
            status=getattr(exc, "code", None),
            headers=_header_pairs(getattr(exc, "hdrs", None)),
            body=body,
            exception=read_error,
        )
        try:
            stream = (
                _FailingReader(read_error) if read_error is not None
                else io.BytesIO(bytes(body) if isinstance(body, (bytes, bytearray)) else b"")
            )
            replacement = urllib.error.HTTPError(
                getattr(exc, "url", None),
                exc.code,
                exc.msg,
                exc.hdrs,
                stream,
            )
        except Exception:
            return exc
        try:
            exc.close()
        except Exception:
            pass
        return replacement


class TracingOpener:
    """An ``urlopen``-compatible opener that logs each exchange verbatim."""

    def __init__(
        self,
        trace: DebugTrace,
        inner: Callable[..., Any],
        *,
        op: Optional[str] = None,
    ) -> None:
        if not isinstance(trace, DebugTrace):
            raise TypeError("trace must be DebugTrace")
        if not callable(inner):
            raise TypeError("inner opener must be callable")
        if op is not None and (not isinstance(op, str) or not op):
            raise ValueError("fixed trace op must be a non-empty string or None")
        self.trace = trace
        self.inner = inner
        self.op = op

    def __call__(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        exchange = self.trace._begin(request, self.op)
        try:
            response = self.inner(request, *args, **kwargs)
        except urllib.error.HTTPError as exc:
            if exchange is None:
                raise
            replacement = exchange.http_error(exc)
            if replacement is exc:
                raise
            raise replacement
        except BaseException as exc:
            if exchange is not None:
                exchange.finish(exception=exc)
            raise
        if exchange is None:
            return response
        return _TeeResponse(response, exchange)


class _TeeResponse:
    """Transparent response proxy recording the body bytes as they are read.

    The response line is written on the first ``close()``. Its ``exception``
    is the error raised by the last read attempt, if that attempt failed.
    """

    _OWN = frozenset({"_response", "_exchange", "_chunks", "_error"})

    def __init__(self, response: Any, exchange: _Exchange) -> None:
        self._response = response
        self._exchange: Optional[_Exchange] = exchange
        self._chunks: list[bytes] = []
        self._error: Optional[BaseException] = None

    def __getattr__(self, name: str) -> Any:
        if name in _TeeResponse._OWN:
            raise AttributeError(name)
        return getattr(self._response, name)

    def _capture(self, data: Any) -> None:
        if isinstance(data, (bytes, bytearray)):
            self._chunks.append(bytes(data))
        elif isinstance(data, str):
            self._chunks.append(data.encode("utf-8", "surrogateescape"))

    def read(self, *args: Any, **kwargs: Any) -> Any:
        try:
            data = self._response.read(*args, **kwargs)
        except BaseException as exc:
            self._error = exc
            raise
        self._error = None
        self._capture(data)
        return data

    def readline(self, *args: Any, **kwargs: Any) -> Any:
        try:
            data = self._response.readline(*args, **kwargs)
        except BaseException as exc:
            self._error = exc
            raise
        self._error = None
        self._capture(data)
        return data

    def __iter__(self) -> Iterator[Any]:
        # Resolve the iterator eagerly: a non-iterable response must still
        # raise TypeError from iter(), as the adapters expect.
        return self._lines(iter(self._response))

    def _lines(self, iterator: Iterator[Any]) -> Iterator[Any]:
        while True:
            try:
                line = next(iterator)
            except StopIteration:
                return
            except BaseException as exc:
                self._error = exc
                raise
            self._error = None
            self._capture(line)
            yield line

    def close(self) -> None:
        try:
            close = getattr(self._response, "close", None)
            if callable(close):
                close()
        finally:
            self._finish()

    def _finish(self) -> None:
        exchange, self._exchange = self._exchange, None
        if exchange is None:
            return
        try:
            status = getattr(self._response, "status", None)
            if status is None:
                status = getattr(self._response, "code", None)
        except Exception:
            status = None
        try:
            headers = _header_pairs(getattr(self._response, "headers", None))
        except Exception:
            headers = None
        exchange.finish(
            status=status,
            headers=headers,
            body=b"".join(self._chunks),
            exception=self._error,
        )

    def __enter__(self) -> "_TeeResponse":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.close()
        return False


__all__ = [
    "DebugTrace",
    "TRACE_REQUEST_SUFFIX",
    "TRACE_RESPONSE_SUFFIX",
    "TRACE_EVENT_SUFFIX",
    "TracingOpener",
    "debug_trace_paths",
    "trace_operation",
    "capture_trace_scope",
]
