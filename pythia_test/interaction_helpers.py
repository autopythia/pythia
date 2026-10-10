"""Canonical endpoint builders used by interaction adapter tests."""

import contextlib
from dataclasses import replace
import http.server
import json
import queue
import threading
from types import SimpleNamespace
from unittest import mock

from pythia.interaction import BUILTIN_MODEL_CATALOG
from pythia.interaction import ChatCompletionsEndpoint
from pythia.interaction import ClaudeRelayEndpoint
from pythia.interaction import ClaudeRelayModel
from pythia.interaction import CodexResponsesModel
from pythia.interaction import InteractionSaveWriter
from pythia.interaction import Message
from pythia.interaction import MessagesEndpoint
from pythia.interaction import StreamingResponsesEndpoint
from pythia.interaction import TokenUsage
from pythia.interaction.codex_auth import _resolve_auth_file
from pythia.interaction.compaction import SUMMARIZATION_SYSTEM_PROMPT
from pythia.interaction.model import ModelTimeoutError
from pythia.interaction.model import ModelTransportError


def _full_url(prefix, suffix):
    return prefix.strip().rstrip("/") + suffix


def chat_endpoint(
    api_url="http://127.0.0.1:8000",
    model=None,
    *,
    api_key=None,
    binding=None,
    extra_sample_params=None,
    **kwargs,
):
    if binding is None:
        binding = BUILTIN_MODEL_CATALOG.bind(
            "chat-completions",
            model,
            endpoint_url=_full_url(api_url, "/v1/chat/completions"),
            endpoint_auth="supplied" if api_key is not None else "none",
            extra_sample_params=extra_sample_params,
        )
    return ChatCompletionsEndpoint(binding=binding, api_key=api_key, **kwargs)


def messages_endpoint(
    api_url,
    model,
    *,
    api_key=None,
    binding=None,
    **kwargs,
):
    if binding is None:
        binding = BUILTIN_MODEL_CATALOG.bind(
            "messages",
            model,
            endpoint_url=_full_url(api_url, "/v1/messages"),
            endpoint_auth="supplied" if api_key is not None else "none",
        )
    return MessagesEndpoint(binding=binding, api_key=api_key, **kwargs)


def responses_endpoint(
    api_url,
    model,
    bearer_token=None,
    *,
    account_id=None,
    api_provider="api",
    binding=None,
    **kwargs,
):
    if binding is None:
        if not isinstance(api_provider, str):
            raise TypeError("api_provider must be a string")
        api_provider = api_provider.strip().lower()
        if api_provider not in {"api", "codex"}:
            raise ValueError("api_provider must be api or codex")
        api = "codex" if api_provider == "codex" else "responses"
        binding = BUILTIN_MODEL_CATALOG.bind(
            api,
            model,
            endpoint_url=_full_url(api_url, "/responses"),
            endpoint_auth="supplied" if bearer_token is not None else "none",
        )
    return StreamingResponsesEndpoint(
        binding=binding,
        bearer_token=bearer_token,
        account_id=account_id,
        **kwargs,
    )


def codex_model(
    endpoint=None,
    *,
    model=None,
    auth=None,
    api_url=None,
    request_timeout_seconds=None,
    codex_home=None,
    auth_file=None,
    binding=None,
    **kwargs,
):
    if endpoint is not None:
        return CodexResponsesModel(endpoint=endpoint, **kwargs)
    if binding is None:
        binding = BUILTIN_MODEL_CATALOG.bind(
            "codex",
            model,
            endpoint_url=(
                None if api_url is None else _full_url(api_url, "/responses")
            ),
            endpoint_auth=(
                "supplied" if auth is not None
                else "codex-login" if codex_home is not None or auth_file is not None
                else None
            ),
        )
    if codex_home is not None or auth_file is not None:
        binding = replace(
            binding,
            endpoint=replace(
                binding.endpoint,
                auth="codex-login",
                auth_file=str(_resolve_auth_file(
                    codex_home=codex_home,
                    auth_file=auth_file,
                ).resolve()),
            ),
        )
    return CodexResponsesModel(
        binding=binding,
        auth=auth,
        request_timeout_seconds=request_timeout_seconds,
        **kwargs,
    )


@contextlib.contextmanager
def patch_compaction(app, name, *args, **kwargs):
    """Patch a compaction hook with one mock wherever an app uses it.

    Automatic compaction runs in the shared turn loop, while the CLI's manual
    /compact keeps its own reference. Modules lacking the name are skipped.
    """
    from pythia.interaction.loop import kernel

    targets = [module for module in (app, kernel) if hasattr(module, name)]
    with mock.patch.object(targets[0], name, *args, **kwargs) as patched:
        with contextlib.ExitStack() as stack:
            for target in targets[1:]:
                stack.enter_context(mock.patch.object(target, name, patched))
            yield patched


# The unpatched save, for patch_saves side effects that do save.
real_save = InteractionSaveWriter.save


def patch_saves(side_effect=None):
    """Patch ``InteractionSaveWriter.save``, every frontend's checkpoint path.

    The mock records each call as ``(writer, context)`` and passes the same to
    ``side_effect``, which may raise to inject a failure or call ``real_save``.
    """
    return mock.patch.object(InteractionSaveWriter, "save", autospec=True,
                             side_effect=side_effect)


class FakeRelayRuntime:
    """A stand-in for one native Claude relay run (``claude_relay._runtime.Runtime``).

    Only the native process is faked: ClaudeRelayModel's own epoch, retire, and
    close logic runs unchanged. A summary request (a Pi compaction's, known by
    the summarizer's instructions) waits until the test calls :meth:`answer`,
    the run is closed (as a retire or close does), or ``WAIT_SECONDS`` pass;
    any other request answers "final answer" at once.
    """

    WAIT_SECONDS = 2.0
    launches = None  # set per patch (fake_relay_runtime)

    def __init__(self, endpoint, tools, token, *, sampling=None, trace=None):
        self.endpoint, self.generation = endpoint, token
        self.closed, self.events = threading.Event(), queue.Queue()
        self.error, self.retirement_complete, self.stderr_tail = None, False, b""
        self.snapshot = None
        self.trace = SimpleNamespace(emit=lambda *args, **kwargs: None,
                                     parked=lambda: None, scope=lambda: {})
        self.diagnostics = SimpleNamespace(set_phase=lambda *args: None,
                                           snapshot=lambda freeze=False: None)
        self.mailbox = SimpleNamespace(release=lambda results: None)

    @property
    def summary(self):
        return (self.snapshot is not None
                and self.snapshot.instructions == SUMMARIZATION_SYSTEM_PROMPT)

    def begin_sample(self, started, scope):
        pass

    def start(self, snapshot):
        self.snapshot = snapshot
        self.launches.all.append(self)
        if self.summary:
            self.launches.summaries.put(self)
        else:
            self.answer("final answer")

    def answer(self, text):
        usage = TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        self.events.put((SimpleNamespace(stop_reason="end_turn", usage=usage, id="msg_fake"),
                         (Message("assistant", text),), ()))

    def next_message(self):
        try:
            event = self.events.get(timeout=self.WAIT_SECONDS)
        except queue.Empty:
            raise ModelTimeoutError("Fake relay run was not answered") from None
        if isinstance(event, Exception):
            raise event
        if self.closed.is_set():
            raise ModelTransportError("Claude Relay continuation retired")
        return event

    def close(self):
        # Like the real run: wake a waiting sample before stopping.
        if not self.closed.is_set():
            self.closed.set()
            self.events.put_nowait(ModelTransportError("Claude Relay continuation retired"))
        self.retirement_complete = True


@contextlib.contextmanager
def fake_relay_runtime():
    """Run ClaudeRelayModel on FakeRelayRuntime instead of a native process.

    Yields the launches: ``summaries``, a queue of the summary runs in launch
    order, and ``all``, a list of every run.
    """
    launches = SimpleNamespace(summaries=queue.Queue(), all=[])

    class Runtime(FakeRelayRuntime):
        pass
    Runtime.launches = launches
    with mock.patch("pythia.interaction.claude_relay._model.Runtime", Runtime):
        yield launches


def relay_model():
    """A ClaudeRelayModel for fake_relay_runtime; its launcher and socket are never used."""
    return ClaudeRelayModel(ClaudeRelayEndpoint(
        "fixture-model", "/nonexistent/claude_relay.py", "/nonexistent/broker.sock",
        1000, "2.1.0-fixture"))


class ScriptedGateway:
    """A local model endpoint that runs one scripted step per POST, on its own thread.

    Each step is called as ``step(handler, gateway)``; with no step left, a
    request stalls. ``requests`` holds each request's JSON body, ``arrivals``
    gets each request's index as it arrives, and ``stalled`` gets it once its
    step has sent what it sends and waits. ``close()`` releases the waiting
    steps and stops the server. With ``tls`` (a server ``SSLContext``) it
    serves HTTPS on ``localhost``.
    """

    def __init__(self, *steps, tls=None) -> None:
        self.steps = list(steps)
        self.requests = []
        self.arrivals, self.stalled = queue.Queue(), queue.Queue()
        self.release = threading.Event()
        self._lock = threading.Lock()
        gateway = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                with gateway._lock:
                    gateway.requests.append(json.loads(body) if body else None)
                    index = len(gateway.requests) - 1
                    step = gateway.steps.pop(0) if gateway.steps else stall
                gateway.arrivals.put(index)
                self.index = index
                try:
                    step(self, gateway)
                except OSError:
                    pass  # the client went away: a cancel shut its socket down

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass  # a cancelled client resets its connection mid-request

        self.server = Server(("127.0.0.1", 0), Handler)
        self.scheme = "http"
        if tls is not None:
            self.server.socket = tls.wrap_socket(self.server.socket, server_side=True)
            self.scheme = "https"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01},
                         daemon=True).start()

    @property
    def url(self) -> str:
        host = "localhost" if self.scheme == "https" else "127.0.0.1"
        return f"{self.scheme}://{host}:{self.server.server_address[1]}"

    def arrived(self, timeout=5):
        """Wait for the next request to arrive; return its JSON body."""
        return self.requests[self.arrivals.get(timeout=timeout)]

    def wait_stalled(self, timeout=5) -> int:
        """Wait until the next stalling step waits; return its request's index."""
        return self.stalled.get(timeout=timeout)

    def close(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


def stall(handler, gateway):
    """A step that never answers: the client waits for the headers."""
    gateway.stalled.put(handler.index)
    gateway.release.wait(30)


def answer(payload, *, status=200, headers=(), content_type="application/json"):
    """A step that sends a complete response: JSON, or ``payload`` bytes as given."""
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def step(handler, gateway):
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        for name, value in headers:
            handler.send_header(name, value)
        handler.end_headers()
        handler.wfile.write(data)
    return step


def stall_body(prefix=b'{"choices": '):
    """A step that sends the headers and the start of a longer body, then waits."""
    def step(handler, gateway):
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(prefix) + 1000))
        handler.end_headers()
        handler.wfile.write(prefix)
        gateway.stalled.put(handler.index)
        gateway.release.wait(30)
    return step


def stall_stream(*events):
    """A step that sends an SSE stream's first events, then waits."""
    def step(handler, gateway):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        handler.wfile.write(sse(*events))
        gateway.stalled.put(handler.index)
        gateway.release.wait(30)
    return step


def sse(*events) -> bytes:
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)


def chat_answer(text):
    """A Chat Completions response body with final text."""
    return {"choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}


def messages_answer(text):
    """A Messages response body with final text."""
    return {"type": "message", "role": "assistant", "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 2, "output_tokens": 1}}


def responses_message(text):
    """A Responses stream event for one completed assistant message."""
    return {"type": "response.output_item.done", "output_index": 0, "item": {
        "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": text}]}}


def responses_answer(text):
    """A complete Responses stream (SSE bytes) with final text."""
    return sse(responses_message(text), {"type": "response.completed", "response": {
        "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}}})
