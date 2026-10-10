"""HTTP models cancel their requests: RequestGate, the cancellable opener, the adapters.

A cancel (``retire`` or ``close``) shuts down the sockets of the call in
progress, which wakes it in any phase of a request, and the call then ends with
ModelCancelled instead of a retry. The servers are local, and stall where each
test needs them to (``ScriptedGateway``).
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import http.client
import json
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.error
import urllib.request

from pythia.interaction import ChatCompletionsModel, ContextPrefix, Environment, Init
from pythia.interaction import InteractionContext
from pythia.interaction import InteractionSaveWriter, Message, MessagesModel, ModelCancelled
from pythia.interaction import ModelFailure, ModelTransportError, ResponsesModel, cli, compaction
from pythia.interaction import _request_gate
from pythia.interaction._cli_editor import Editor
from pythia.interaction._request_gate import RequestGate, cancellable_opener
from pythia.interaction._request_gate import cancellable_urlopen, uncancellable
from pythia.interaction.compaction import SUMMARIZATION_SYSTEM_PROMPT
from pythia.interaction.responses import ResponsesOpaqueCompactor
from pythia.interaction.runtime_config import InteractionConfig, InteractionConfigSnapshot
from pythia_test.interaction_helpers import ScriptedGateway, answer, chat_answer
from pythia_test.interaction_helpers import chat_endpoint, codex_model, messages_answer
from pythia_test.interaction_helpers import messages_endpoint, responses_answer
from pythia_test.interaction_helpers import responses_endpoint, responses_message, stall
from pythia_test.interaction_helpers import stall_body, stall_stream
from pythia_test.test_compaction_preemption import _CONFIG as _COMPACTING
from pythia_test.test_compaction_preemption import _two_request_context


_CONTEXT = InteractionContext((Message("user", "hi"),))


def _tcp_pair():
    listener = socket.create_server(("127.0.0.1", 0))
    client = socket.create_connection(listener.getsockname())
    server, _ = listener.accept()
    listener.close()
    return client, server


def _in_thread(function):
    """Run ``function`` on a thread; the box gets its result or error, and its end time."""
    box = {}

    def run():
        try:
            box["result"] = function()
        except BaseException as exc:
            box["error"] = exc
        finally:
            box["ended"] = time.monotonic()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def _wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for test condition")
        time.sleep(0.001)


class RequestGateTests(unittest.TestCase):
    """The gate and its call scope, without sockets."""

    @staticmethod
    def gate():
        return RequestGate(provider="test-provider", label="Test", model="test-model")

    def test_close_refuses_every_later_call_before_sending_anything(self):
        gate = self.gate()
        gate.close()
        for _ in range(2):
            with self.assertRaises(ModelCancelled) as caught:
                with gate.call():
                    self.fail("a closed gate entered a call")
            self.assertEqual(str(caught.exception), "Test model is closed; no request was sent.")
            self.assertIsNone(caught.exception.failure)  # nothing was sent: nothing to record
        self.assertTrue(gate.closed)

    def test_retire_cancels_the_call_in_progress_and_not_a_later_one(self):
        gate = self.gate()
        with gate.call() as call:
            call.check()
            gate.retire()
            self.assertTrue(call.cancelled)
            with self.assertRaises(ModelCancelled) as caught:
                call.check()
        failure = caught.exception.failure
        self.assertEqual((failure.category, failure.message, failure.provider, failure.model),
                         ("request_cancelled", "Test request cancelled.", "test-provider",
                          "test-model"))
        self.assertFalse(gate.closed)
        with gate.call() as later:  # a new request
            later.check()
            self.assertFalse(later.cancelled)

    def test_a_failure_that_leaves_a_cancelled_call_becomes_a_cancellation(self):
        gate = self.gate()
        partial = Message("assistant", "partial")
        failure = ModelFailure("stream_closed", "Responses stream closed before response.completed",
                               provider="api", model="m", event_count=3, attempt_count=2)
        cause = ModelTransportError("closed", failure=failure, completed_items=(partial,))
        with self.assertRaises(ModelCancelled) as caught:
            with gate.call():
                gate.retire()
                raise cause
        error = caught.exception
        self.assertIs(error.__cause__, cause)
        self.assertEqual(error.completed_items, (partial,))  # the turn loop saves it
        self.assertEqual(error.failure, replace(failure, category="request_cancelled",
                                                message="Test request cancelled."))

        # Any failure counts once the call is cancelled: a cancel never fails a turn.
        with self.assertRaises(ModelCancelled) as caught:
            with gate.call():
                gate.close()
                raise http.client.IncompleteRead(b"{", 10)
        self.assertEqual(str(caught.exception), "Test request cancelled: the model was closed.")
        self.assertIsInstance(caught.exception.__cause__, http.client.IncompleteRead)

        # A failure of a call that was not cancelled stands, and so does a cancellation.
        for error, cancel in ((ModelTransportError("lost"), False),
                              (ModelCancelled("already"), True)):
            gate = self.gate()
            with self.subTest(error=type(error).__name__), \
                    self.assertRaises(type(error)) as caught:
                with gate.call():
                    if cancel:
                        gate.retire()
                    raise error
            self.assertIs(caught.exception, error)

        # A call that completed is returned, even if a cancel came at its end.
        gate = self.gate()
        with gate.call():
            gate.retire()

    def test_wait_ends_at_once_when_the_call_is_cancelled(self):
        gate, cause = self.gate(), urllib.error.URLError("reset")
        started = {}

        def backoff():
            with gate.call() as call:
                started["at"] = time.monotonic()
                call.wait(30, cause=cause)
        thread, box = _in_thread(backoff)
        time.sleep(0.05)
        gate.retire()
        thread.join(5)
        self.assertLess(box["ended"] - started["at"], 1)
        self.assertIsInstance(box["error"], ModelCancelled)
        self.assertIs(box["error"].__cause__, cause)
        with gate.call() as call:  # uncancelled, it waits it out
            begun = time.monotonic()
            call.wait(0.05)
            self.assertGreaterEqual(time.monotonic() - begun, 0.05)

    def test_an_injected_sleep_is_checked_on_either_side(self):
        gate, slept = self.gate(), []

        def sleep(seconds):
            slept.append(seconds)
            gate.retire()  # a cancel during the sleep
        with gate.call() as call:
            with self.assertRaises(ModelCancelled):
                call.wait(7, sleep)
            with self.assertRaises(ModelCancelled):
                call.wait(7, sleep)  # cancelled already: no sleep
        self.assertEqual(slept, [7])

    def test_a_cancel_shuts_down_the_call_s_sockets_and_refuses_new_ones(self):
        gate = self.gate()
        client, server = _tcp_pair()
        self.addCleanup(client.close)
        self.addCleanup(server.close)
        with gate.call() as call:
            call._register(client)
            thread, box = _in_thread(lambda: client.recv(10))
            time.sleep(0.05)
            cancelled = time.monotonic()
            gate.retire()
            thread.join(2)
            self.assertEqual(box["result"], b"")  # woke at once, at an end of file
            self.assertLess(box["ended"] - cancelled, 1)
            fresh = socket.socket()
            self.addCleanup(fresh.close)
            with self.assertRaises(ConnectionAbortedError):
                call._register(fresh)  # registered after the cancel: refused
        self.assertEqual(gate._calls, [])  # an ended call's sockets are not the gate's

    def test_uncancellable_requests_register_with_no_call(self):
        gate = self.gate()
        self.assertIsNone(_request_gate._CALL.get())
        with gate.call() as call:
            self.assertIs(_request_gate._CALL.get(), call)
            with uncancellable():
                self.assertIsNone(_request_gate._CALL.get())
            self.assertIs(_request_gate._CALL.get(), call)
        self.assertIsNone(_request_gate._CALL.get())


class _PhaseTests(unittest.TestCase):
    """One request through a cancellable opener, in a call scope, cancelled mid-way."""

    def gateway(self, *steps, tls=None):
        gateway = ScriptedGateway(*steps, tls=tls)
        self.addCleanup(gateway.close)
        return gateway

    def cancel_during(self, url, ready, *, read=None, opener=cancellable_urlopen):
        """Open ``url`` on a thread; once ``ready(reading)`` returns, retire the gate.

        ``reading`` is set once the response's headers are in. Returns the
        thread's box and the seconds from the cancel to the end of the call.
        """
        gate, reading = RequestGate(provider="test", label="Test"), threading.Event()

        def request():
            with gate.call():
                with opener(urllib.request.Request(url, data=b"{}", method="POST"),
                            timeout=10) as response:
                    reading.set()
                    return (read or (lambda r: r.read()))(response)
        thread, box = _in_thread(request)
        ready(reading)
        cancelled = time.monotonic()
        gate.retire()
        thread.join(5)
        self.assertFalse(thread.is_alive(), "the cancel did not end the request")
        return box, box["ended"] - cancelled

    def assert_cancelled(self, box, seconds, cause):
        self.assertLess(seconds, 1)
        self.assertIsInstance(box.get("error"), ModelCancelled, box)
        self.assertIsInstance(box["error"].__cause__, cause)

    @staticmethod
    def in_body(gateway):
        """Ready once the client reads a body that stalls."""
        def ready(reading):
            reading.wait(5)
            gateway.wait_stalled()
            time.sleep(0.1)  # inside the read now
        return ready

    @staticmethod
    def read_lines(response):
        return [line for line in response]  # as the SSE reader does


class CancellableOpenerTests(_PhaseTests):
    """Each phase of a plain HTTP request ends at once on a cancel."""

    def test_waiting_for_the_headers(self):
        gateway = self.gateway(stall)
        box, seconds = self.cancel_during(gateway.url, lambda reading: gateway.wait_stalled())
        self.assert_cancelled(box, seconds, http.client.RemoteDisconnected)

    def test_reading_a_body(self):
        gateway = self.gateway(stall_body())
        box, seconds = self.cancel_during(gateway.url, self.in_body(gateway))
        self.assert_cancelled(box, seconds, http.client.IncompleteRead)

    def test_reading_a_stream(self):
        # The stream just ends; the Responses adapter reads that as stream_closed.
        gateway = self.gateway(stall_stream(responses_message("partial")))
        box, seconds = self.cancel_during(gateway.url, self.in_body(gateway),
                                          read=self.read_lines)
        self.assertLess(seconds, 1)
        self.assertEqual(len(box["result"]), 2)  # the event and its blank line, then the end

    def test_connecting(self):
        # A listener whose accept queue is full: on Linux, the next connect hangs.
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(0)
        self.addCleanup(listener.close)
        filler = socket.create_connection(listener.getsockname())
        self.addCleanup(filler.close)
        probe = socket.socket()
        probe.settimeout(0.2)
        try:
            probe.connect(listener.getsockname())
            self.skipTest("a connect to a full accept queue does not hang here")
        except socket.timeout:
            pass
        except OSError:
            self.skipTest("a connect to a full accept queue fails at once here")
        finally:
            probe.close()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}/"
        box, seconds = self.cancel_during(url, lambda reading: time.sleep(0.3))
        self.assert_cancelled(box, seconds, urllib.error.URLError)

    def test_a_call_cancelled_before_its_request_connects_nothing(self):
        gateway, gate = self.gateway(answer(chat_answer("never"))), RequestGate(
            provider="test", label="Test")
        with self.assertRaises(ModelCancelled) as caught:
            with gate.call():
                gate.retire()
                cancellable_urlopen(urllib.request.Request(gateway.url, data=b"{}"), timeout=5)
        reason = caught.exception.__cause__.reason
        self.assertIsInstance(reason, ConnectionAbortedError)  # refused to register
        self.assertEqual(gateway.requests, [])

    def test_outside_a_call_it_is_urllib_s_opener(self):
        gateway = self.gateway(answer(chat_answer("ok")), answer({"error": "gone"}, status=404))
        with cancellable_urlopen(urllib.request.Request(gateway.url, data=b"{}"),
                                 timeout=5) as response:
            self.assertEqual(json.loads(response.read()), chat_answer("ok"))
        with self.assertRaises(urllib.error.HTTPError) as caught:
            cancellable_urlopen(urllib.request.Request(gateway.url, data=b"{}"), timeout=5)
        self.assertEqual(caught.exception.code, 404)
        caught.exception.close()


def _tls_contexts(directory: Path):
    """A server context with a new certificate for ``localhost`` only, and a
    client context that trusts it; None without ``openssl``."""
    openssl = shutil.which("openssl")
    if openssl is None:
        return None
    key, cert = directory / "key.pem", directory / "cert.pem"
    completed = subprocess.run(
        [openssl, "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
         "-nodes", "-keyout", str(key), "-out", str(cert), "-days", "2",
         "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost"],
        capture_output=True, timeout=60)
    if completed.returncode != 0:
        return None
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(str(cert), str(key))
    return server, ssl.create_default_context(cafile=str(cert))


class TlsTests(_PhaseTests):
    """The same phases over TLS, and the handshake; certificates are still verified."""

    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        contexts = _tls_contexts(Path(cls._directory.name))
        if contexts is None:
            cls._directory.cleanup()
            raise unittest.SkipTest("openssl cannot make a test certificate here")
        cls.server_context, cls.client_context = contexts
        cls.opener = staticmethod(cancellable_opener(context=cls.client_context))

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_waiting_for_the_headers(self):
        gateway = self.gateway(stall, tls=self.server_context)
        box, seconds = self.cancel_during(gateway.url, lambda reading: gateway.wait_stalled(),
                                          opener=self.opener)
        self.assert_cancelled(box, seconds, http.client.RemoteDisconnected)

    def test_reading_a_body(self):
        gateway = self.gateway(stall_body(), tls=self.server_context)
        box, seconds = self.cancel_during(gateway.url, self.in_body(gateway), opener=self.opener)
        self.assert_cancelled(box, seconds, http.client.IncompleteRead)

    def test_reading_a_stream(self):
        gateway = self.gateway(stall_stream(responses_message("partial")),
                               tls=self.server_context)
        box, seconds = self.cancel_during(gateway.url, self.in_body(gateway),
                                          read=self.read_lines, opener=self.opener)
        self.assertLess(seconds, 1)
        self.assertEqual(len(box["result"]), 2)

    def test_the_handshake(self):
        # A server that accepts the connection and never answers the handshake.
        listener = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(listener.close)
        accepted, held = threading.Event(), []

        def accept():
            connection, _ = listener.accept()
            held.append(connection)
            accepted.set()
        threading.Thread(target=accept, daemon=True).start()
        self.addCleanup(lambda: [connection.close() for connection in held])

        def ready(reading):
            accepted.wait(5)
            time.sleep(0.1)  # inside the handshake now
        url = f"https://localhost:{listener.getsockname()[1]}/"
        box, seconds = self.cancel_during(url, ready, opener=self.opener)
        self.assert_cancelled(box, seconds, urllib.error.URLError)

    def test_certificates_are_still_verified(self):
        gateway = self.gateway(answer(chat_answer("ok")), tls=self.server_context)
        with self.opener(urllib.request.Request(gateway.url, data=b"{}"), timeout=5) as response:
            self.assertEqual(json.loads(response.read()), chat_answer("ok"))
        # The certificate names localhost, not 127.0.0.1.
        wrong = gateway.url.replace("localhost", "127.0.0.1")
        with self.assertRaises(urllib.error.URLError) as caught:
            self.opener(urllib.request.Request(wrong, data=b"{}"), timeout=5)
        self.assertIsInstance(caught.exception.reason, ssl.SSLCertVerificationError)

    def test_an_adapter_over_tls_is_cancelled_and_still_works(self):
        gateway = self.gateway(stall, answer(chat_answer("after")), tls=self.server_context)
        model = ChatCompletionsModel(chat_endpoint(api_url=gateway.url), opener=self.opener)
        thread, box = _in_thread(lambda: model.sample(_CONTEXT))
        gateway.wait_stalled()
        cancelled = time.monotonic()
        model.retire()
        thread.join(5)
        self.assertLess(box["ended"] - cancelled, 1)
        self.assertIsInstance(box["error"], ModelCancelled)
        self.assertEqual(model.sample(_CONTEXT).last_assistant_text, "after")
        self.assertEqual(len(gateway.requests), 2)


def _sse_answer(text):
    return answer(responses_answer(text), content_type="text/event-stream")


# (name, label, provider, build(url), a step that answers "after", streams)
_ADAPTERS = (
    ("chat-completions", "Chat Completions", "chat-completions",
     lambda url: ChatCompletionsModel(chat_endpoint(api_url=url)),
     lambda: answer(chat_answer("after")), False),
    ("messages", "Messages", "messages",
     lambda url: MessagesModel(messages_endpoint(url, "model", max_output_tokens=100)),
     lambda: answer(messages_answer("after")), False),
    ("responses", "Responses", "api",
     lambda url: ResponsesModel(responses_endpoint(url, "model")),
     lambda: _sse_answer("after"), True),
    ("codex", "Codex Responses", "codex",
     lambda url: codex_model(responses_endpoint(url, "model", bearer_token="FAKE",
                                                api_provider="codex")),
     lambda: _sse_answer("after"), True),
)


class AdapterTests(unittest.TestCase):
    """Each HTTP adapter, with its default opener, against a local server that stalls."""

    def start(self, build, *steps):
        gateway = ScriptedGateway(*steps)
        self.addCleanup(gateway.close)
        model = build(gateway.url)
        self.addCleanup(model.close)
        return gateway, model

    def cancel(self, model, how, ready, call=None):
        """Sample on a thread; once ``ready()`` returns, retire or close the model."""
        thread, box = _in_thread(call or (lambda: model.sample(_CONTEXT)))
        ready()
        cancelled = time.monotonic()
        getattr(model, how)()
        thread.join(5)
        self.assertFalse(thread.is_alive(), "the cancel did not end the request")
        self.assertLess(box["ended"] - cancelled, 1)
        self.assertIsInstance(box.get("error"), ModelCancelled, box)
        return box["error"]

    def test_retire_cancels_a_request_waiting_for_its_headers_and_the_model_still_works(self):
        for name, label, provider, build, answered, _ in _ADAPTERS:
            with self.subTest(adapter=name):
                gateway, model = self.start(build, stall, answered())
                error = self.cancel(model, "retire", gateway.wait_stalled)
                self.assertEqual(str(error), f"{label} request cancelled.")
                self.assertEqual((error.failure.category, error.failure.provider),
                                 ("request_cancelled", provider))
                self.assertIsInstance(error.__cause__, (OSError, http.client.HTTPException))
                self.assertEqual(len(gateway.requests), 1)  # no retry
                self.assertEqual(model.sample(_CONTEXT).last_assistant_text, "after")
                self.assertEqual(len(gateway.requests), 2)

    def test_retire_cancels_a_body_or_stream_and_keeps_the_completed_output(self):
        for name, _, _, build, _, streams in _ADAPTERS:
            with self.subTest(adapter=name):
                step = stall_stream(responses_message("partial")) if streams else stall_body()
                gateway, model = self.start(build, step)

                def ready():
                    gateway.wait_stalled()
                    time.sleep(0.1)  # reading now
                error = self.cancel(model, "retire", ready)
                self.assertEqual([item.content_text for item in error.completed_items],
                                 ["partial"] if streams else [])
                self.assertEqual(error.failure.category, "request_cancelled")
                self.assertEqual(len(gateway.requests), 1)

    def test_close_cancels_the_request_and_refuses_the_next_without_sending_it(self):
        for name, label, _, build, _, _ in _ADAPTERS:
            with self.subTest(adapter=name):
                gateway, model = self.start(build, stall)
                error = self.cancel(model, "close", gateway.wait_stalled)
                self.assertEqual(str(error), f"{label} request cancelled: the model was closed.")
                with self.assertRaises(ModelCancelled) as caught:
                    model.sample(_CONTEXT)
                self.assertIsNone(caught.exception.failure)
                self.assertEqual(len(gateway.requests), 1)

    def test_retire_ends_a_retry_backoff_at_once(self):
        for name, _, _, build, answered, _ in _ADAPTERS:
            with self.subTest(adapter=name):
                busy = answer({"error": {"message": "busy"}}, status=503,
                              headers=(("Retry-After", "30"),))
                gateway, model = self.start(build, busy, answered())

                def ready():
                    gateway.arrived()
                    time.sleep(0.2)  # in the 30-second backoff now
                self.cancel(model, "retire", ready)
                self.assertEqual(len(gateway.requests), 1)  # the retry never started

    def test_codex_remote_compaction_is_cancelled_too(self):
        _, _, _, build, _, _ = _ADAPTERS[-1]
        gateway, model = self.start(build, stall)
        context = InteractionContext((Message("user", "a long task"),
                                      Message("assistant", "done so far")))
        self.cancel(model, "retire", gateway.wait_stalled,
                    call=lambda: ResponsesOpaqueCompactor(model).compact(context))
        self.assertEqual(gateway.requests[0]["input"][-1], {"type": "compaction_trigger"})


class CliTests(unittest.IsolatedAsyncioTestCase):
    """/steer!! and /exit!! during an HTTP sample or compaction, through the CLI's own
    input handling."""

    async def turn(self, state, model, context, config=None):
        state.active_model = model  # as _drive_interaction sets it
        with tempfile.TemporaryDirectory() as directory:
            return await cli._turn(context, model, Environment(), state,
                                   InteractionSaveWriter(Path(directory) / "log.jsonl"),
                                   config or InteractionConfig(InteractionConfigSnapshot()),
                                   steering=True)

    @staticmethod
    def enter(state, text):
        state.editor = Editor(text, len(text))
        state.handle_key("c-m", "\r")

    def start(self, *steps):
        gateway = ScriptedGateway(*steps)
        self.addCleanup(gateway.close)
        model = ChatCompletionsModel(chat_endpoint(api_url=gateway.url))
        self.addCleanup(model.close)
        context = InteractionContext((Init(model="fixture"), Message("user", "task")))
        return gateway, model, context, cli._UIState(ready=True)

    async def test_steer_cancels_the_sample_and_comes_next(self):
        gateway, model, context, state = self.start(stall, answer(chat_answer("after")))
        turn = asyncio.ensure_future(self.turn(state, model, context))
        await asyncio.to_thread(gateway.wait_stalled)
        steered = time.monotonic()
        self.enter(state, "/steer!! go left")
        result = await asyncio.wait_for(turn, 5)
        self.assertLess(time.monotonic() - steered, 2)  # the stall would last 30 s
        self.assertEqual(result.final_text, "after")
        self.assertEqual(len(gateway.requests), 2)
        self.assertIn("go left", json.dumps(gateway.requests[1]["messages"][-1]))
        failures = [item for item in context.items if isinstance(item, ModelFailure)]
        self.assertEqual([failure.category for failure in failures], ["request_cancelled"])
        self.assertIsNone(state.retry)  # a cancelled sample is not a failure

    async def test_exit_cancels_the_sample_and_closes_the_model(self):
        gateway, model, context, state = self.start(stall)
        turn = asyncio.ensure_future(self.turn(state, model, context))
        await asyncio.to_thread(gateway.wait_stalled)
        stopped = time.monotonic()
        self.enter(state, "/exit!!")
        result = await asyncio.wait_for(turn, 5)
        self.assertLess(time.monotonic() - stopped, 2)
        self.assertEqual(result.kind, "stopped")
        self.assertEqual(len(gateway.requests), 1)
        _wait_for(lambda: model._gate.closed)  # the stop closes it on a helper thread
        with self.assertRaises(ModelCancelled):
            model.sample(_CONTEXT)  # closed: nothing is sent
        self.assertEqual(len(gateway.requests), 1)

    async def test_exit_cancels_an_http_compaction_in_flight_or_between_its_requests(self):
        # A Pi compaction that the cut splits makes two summary requests; the
        # stop comes while the first is in flight, or after it, while the
        # compactor builds the second's prompt, with no request in flight.
        loop = asyncio.get_running_loop()
        for between in (False, True):
            with self.subTest(between=between):
                steps = (answer(chat_answer("HISTORY SUMMARY")),) if between else (stall,)
                gateway, model, _, state = self.start(*steps)
                context = _two_request_context()
                original = compaction._turn_prefix_prompt

                async def exit_now():
                    self.enter(state, "/exit!!")

                def prompt(*args, **kwargs):
                    if between:
                        epoch = model._gate._epoch
                        asyncio.run_coroutine_threadsafe(exit_now(), loop).result(5)
                        _wait_for(lambda: model._gate._epoch != epoch)  # the stop reached it
                    return original(*args, **kwargs)
                with mock.patch.object(compaction, "_turn_prefix_prompt", side_effect=prompt):
                    turn = asyncio.ensure_future(self.turn(state, model, context, _COMPACTING))
                    if not between:
                        await asyncio.to_thread(gateway.wait_stalled)
                        self.enter(state, "/exit!!")
                    result = await asyncio.wait_for(turn, 5)
                self.assertEqual(result.kind, "stopped")
                self.assertEqual(len(gateway.requests), 1)  # no second summary request
                self.assertEqual(gateway.requests[0]["messages"][0]["content"],
                                 SUMMARIZATION_SYSTEM_PROMPT)
                self.assertFalse(any(isinstance(item, ContextPrefix) for item in context.items))
                self.assertIsNone(state.retry)  # a stop, not a failure
                _wait_for(lambda: model._gate.closed)


if __name__ == "__main__":
    unittest.main()
