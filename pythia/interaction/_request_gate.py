"""Cancel an HTTP model's request from any thread: ``retire`` and ``close``.

The HTTP adapters (Responses and Codex, Chat Completions, Messages) give each
model a :class:`RequestGate`, and run each model call (a sample, or Codex's
remote compaction) in a *call scope*, ``with gate.call() as call:``. The scope
records the gate's epoch when the call starts, and the sockets that the call's
requests open.

- ``retire()`` cancels the calls in progress and leaves the model usable: it
  moves the epoch and shuts down those sockets, which wakes a thread blocked
  in a connect, a TLS handshake, or a read at once. A call that starts later
  records the new epoch: it is a new request.
- ``close()`` does the same, and refuses every later call.

A cancelled call ends with :class:`ModelCancelled`, never with a retry. A
shut-down socket reads as a transient failure (a reset connection, a short
body, a stream that ends early), which the adapters would retry. So they check
the call before each attempt (``check``), wait out a retry's backoff with
``wait``, which a cancel ends at once, and the scope turns any failure that
leaves a cancelled call into ``ModelCancelled``.

Sockets register themselves through :func:`cancellable_urlopen`, the adapters'
default opener. Its connections find the current call scope in a context
variable, and register each socket before it connects and each TLS socket
before its handshake, so that a cancel reaches every phase of a request. An
injected opener (tests) registers nothing; then only the checks apply. A
request outside a call scope, or inside ``uncancellable()`` (an OAuth token
refresh), is never cancelled.
"""

from __future__ import annotations

import contextlib
import contextvars
import http.client
import socket
import threading
import time
import urllib.request
from dataclasses import replace
from typing import Any, Callable, Iterator, List, Optional

from .items import ModelFailure
from .model import ModelCancelled
from .model import ModelError


CANCELLED_CATEGORY = "request_cancelled"

_CALL: "contextvars.ContextVar[Optional[RequestCall]]" = contextvars.ContextVar(
    "pythia_interaction_request_call", default=None)


class RequestGate:
    """One HTTP model's cancellation state; ``retire`` and ``close`` are thread-safe.

    ``provider``, ``label``, and ``model`` describe the model in a
    cancellation's message and ``ModelFailure``: for example
    ``"chat-completions"``, ``"Chat Completions"``, and the model's name. The
    gate has a lock of its own, never an adapter's, so that a cancel never
    waits for the request it cancels.
    """

    def __init__(self, *, provider: str, label: str, model: Optional[str] = None) -> None:
        self.provider, self.label, self.model = provider, label, model
        self._condition = threading.Condition()
        self._epoch = 0
        self._closed = False
        self._calls: List[RequestCall] = []

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def call(self) -> "RequestCall":
        """The scope of one model call; enter it with ``with``."""
        return RequestCall(self)

    def retire(self) -> None:
        """Cancel the calls in progress; a later call is a new request."""
        self._cancel(close=False)

    def close(self) -> None:
        """Cancel the calls in progress, and refuse every later call."""
        self._cancel(close=True)

    def _cancel(self, *, close: bool) -> None:
        with self._condition:
            if close:
                self._closed = True
            self._epoch += 1
            # A socket registers under this lock, after checking the epoch: it
            # is either shut down here or refused (RequestCall._register).
            sockets = [sock for call in self._calls for sock in call._sockets]
            self._condition.notify_all()  # ends the backoffs in progress
        for sock in sockets:
            _shutdown(sock)


class RequestCall:
    """The scope of one model call: its attempts, backoffs, and sockets.

    Entering it refuses a closed model with ``ModelCancelled`` (with no
    failure: nothing was sent), and records the gate's epoch: the call is
    cancelled once the epoch moves or the gate closes. A failure that leaves
    the scope while the call is cancelled becomes ``ModelCancelled``, chained
    from it, and keeps the output that completed before the cancel.
    """

    def __init__(self, gate: RequestGate) -> None:
        self._gate = gate
        self._epoch = -1
        self._sockets: List[socket.socket] = []
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> "RequestCall":
        gate = self._gate
        with gate._condition:
            if gate._closed:
                raise ModelCancelled(f"{gate.label} model is closed; no request was sent.")
            self._epoch = gate._epoch
            gate._calls.append(self)
        self._token = _CALL.set(self)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        _CALL.reset(self._token)
        gate = self._gate
        with gate._condition:
            gate._calls.remove(self)
            self._sockets = []
            cancelled = self._cancelled_locked()
        if cancelled and isinstance(exc, Exception) and not isinstance(exc, ModelCancelled):
            raise self._cancellation(exc) from exc
        # A call that completed is returned, even if a cancel came at its end.
        return False

    @property
    def cancelled(self) -> bool:
        with self._gate._condition:
            return self._cancelled_locked()

    def check(self, cause: Optional[BaseException] = None) -> None:
        """Raise ``ModelCancelled`` if the call is cancelled.

        Before each attempt, and before work that a cancel makes pointless.
        ``cause`` is the failure in hand, if any: the cancellation is chained
        from it and keeps its completed output.
        """
        if self.cancelled:
            error = self._cancellation(cause)
            if cause is None:
                raise error
            raise error from cause

    def wait(self, seconds: float, sleep: Optional[Callable[[float], Any]] = None, *,
             cause: Optional[BaseException] = None) -> None:
        """A retry's backoff: wait ``seconds``, or raise as soon as the call is cancelled.

        ``cause`` is the failure being retried (see ``check``). An injected
        ``sleep`` (a test's ``retry_sleep``) replaces the wait, with a check on
        either side.
        """
        self.check(cause)
        if sleep is not None:
            sleep(seconds)
        else:
            gate = self._gate
            deadline = time.monotonic() + seconds
            with gate._condition:
                while not self._cancelled_locked():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    gate._condition.wait(remaining)
        self.check(cause)

    def _register(self, sock: socket.socket) -> None:
        """Track ``sock`` until the call ends; refuse it if the call is cancelled."""
        with self._gate._condition:
            if not self._cancelled_locked():
                self._sockets.append(sock)
                return
        raise ConnectionAbortedError("Request cancelled")

    def _cancelled_locked(self) -> bool:
        return self._gate._closed or self._gate._epoch != self._epoch

    def _cancellation(self, cause: Optional[BaseException]) -> ModelCancelled:
        gate = self._gate
        with gate._condition:
            closed = gate._closed
        message = f"{gate.label} request cancelled" + (": the model was closed." if closed
                                                       else ".")
        completed, failure = (), None
        if isinstance(cause, ModelError):
            completed, failure = cause.completed_items, cause.failure
        if failure is None:
            failure = ModelFailure(CANCELLED_CATEGORY, message, provider=gate.provider,
                                   model=gate.model)
        else:
            # Keep the request's diagnostics (attempts, stream events, IDs).
            failure = replace(failure, category=CANCELLED_CATEGORY, message=message)
        return ModelCancelled(message, failure=failure, completed_items=completed)


@contextlib.contextmanager
def uncancellable() -> Iterator[None]:
    """Run requests inside a call scope that no cancel may end.

    Their sockets register with no call. An OAuth token refresh is one: a
    cancel could cut it off after the provider rotated the refresh token, but
    before the new one was saved. Check the call before and after instead.
    """
    token = _CALL.set(None)
    try:
        yield
    finally:
        _CALL.reset(token)


def _shutdown(sock: socket.socket) -> None:
    """Shut ``sock`` down for both directions; any thread may call it.

    This wakes a thread blocked on it in a connect, a TLS handshake, or a
    read. It calls the plain socket's shutdown even on a TLS socket:
    ``SSLSocket.shutdown`` also drops the socket's SSL object, which a read on
    another thread may be about to use. A closed socket object has no
    descriptor any more, so a late shutdown fails here instead of reaching a
    reused descriptor.
    """
    try:
        socket.socket.shutdown(sock, socket.SHUT_RDWR)
    except OSError:
        pass  # closed, or not connected yet (_create_connection checks after it)


def _create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
    """``socket.create_connection``, registering each socket with the current call.

    Each socket registers before it connects, so a cancel ends a connect in
    progress. A cancel that lands between the two found nothing connected to
    shut down; the check after the connect catches it.
    """
    call = _CALL.get()
    if call is None:
        return socket.create_connection(address, timeout, source_address)
    host, port = address
    error: Optional[OSError] = None
    for family, kind, proto, _name, sockaddr in socket.getaddrinfo(host, port, 0,
                                                                   socket.SOCK_STREAM):
        sock = None
        try:
            sock = socket.socket(family, kind, proto)
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            call._register(sock)
            sock.connect(sockaddr)
            if call.cancelled:
                raise ConnectionAbortedError("Request cancelled")
            return sock
        except OSError as exc:
            error = exc
            if sock is not None:
                sock.close()
            if call.cancelled:
                raise  # tries no other address
    if error is not None:
        raise error
    raise OSError("getaddrinfo returns an empty list")


class _HTTPConnection(http.client.HTTPConnection):
    """An HTTP connection whose socket registers with the current call."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _create_connection


class _HTTPHandler(urllib.request.HTTPHandler):
    def do_open(self, http_class, req, **http_conn_args):
        return super().do_open(_HTTPConnection, req, **http_conn_args)


if hasattr(http.client, "HTTPSConnection") and hasattr(urllib.request, "HTTPSHandler"):

    class _HTTPSConnection(http.client.HTTPSConnection):
        """An HTTPS connection whose TCP and TLS sockets register with the current call."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._create_connection = _create_connection

        def connect(self) -> None:
            # HTTPSConnection.connect, with the handshake split from
            # wrap_socket, so that the TLS socket registers before it.
            http.client.HTTPConnection.connect(self)  # TCP, and a proxy's tunnel
            server_hostname = self._tunnel_host or self.host
            self.sock = self._context.wrap_socket(
                self.sock, server_hostname=server_hostname, do_handshake_on_connect=False)
            call = _CALL.get()
            if call is not None:
                call._register(self.sock)
            self.sock.do_handshake()

    class _HTTPSHandler(urllib.request.HTTPSHandler):
        def do_open(self, http_class, req, **http_conn_args):
            return super().do_open(_HTTPSConnection, req, **http_conn_args)

else:  # Python without ssl: plain HTTP only, as urllib's own opener
    _HTTPSHandler = None


def cancellable_opener(context=None) -> Callable[..., Any]:
    """An opener like ``urllib.request.urlopen``'s, whose requests a cancel can end.

    It has urlopen's default handlers (proxies from the environment, redirects,
    HTTP errors) and TLS context, with connections that register their sockets
    with the current call. ``context`` replaces the TLS context (tests).
    """
    handlers = [_HTTPHandler()]
    if _HTTPSHandler is not None:
        handlers.append(_HTTPSHandler(context=context))
    return urllib.request.build_opener(*handlers).open


_DEFAULT_OPENER: Optional[Callable[..., Any]] = None
_DEFAULT_LOCK = threading.Lock()


def cancellable_urlopen(url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
    """``urllib.request.urlopen`` with cancellable connections: the HTTP adapters' default.

    Like urlopen, it builds its opener once, on first use.
    """
    global _DEFAULT_OPENER
    opener = _DEFAULT_OPENER
    if opener is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_OPENER is None:
                _DEFAULT_OPENER = cancellable_opener()
            opener = _DEFAULT_OPENER
    return opener(url, data, timeout)


__all__ = [
    "CANCELLED_CATEGORY",
    "RequestCall",
    "RequestGate",
    "cancellable_opener",
    "cancellable_urlopen",
    "uncancellable",
]
