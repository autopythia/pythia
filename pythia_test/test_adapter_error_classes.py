"""Each HTTP adapter's failures by the class of the turn loop's automatic
retries (ModelError.retry_class), and the retries of the failures that the
adapters newly vouch for: undecodable and malformed responses, refusals, and
the Responses stream's failed, incomplete and empty responses
(doc/model-error-recovery-plan.md, sections 4 to 6). Each status's class is
in its adapter's own test module; the settings are test_error_retries', and
the loop's policies test_loop_kernel's.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pythia.interaction import ChatCompletionsModel, Environment, Init, InteractionContext
from pythia.interaction import Message, MessagesModel, ModelAuthenticationError, ModelRefusal
from pythia.interaction import ModelResponseError
from pythia.interaction.items import ModelFailure, SampleMetadata
from pythia.interaction.loop import TurnHost, kernel, run_turn
from pythia.interaction.model import RetryState, SampleParams
from pythia.interaction.runtime_config import InteractionConfigSnapshot
from pythia_test import test_messages as messages_tests
from pythia_test import test_responses as responses_tests
from pythia_test.interaction_helpers import chat_endpoint, codex_model, responses_endpoint
from pythia_test.test_interaction import _FakeHTTPResponse


class _Host(TurnHost):
    """Keeps what the turn loop appends to the context, and its notices."""

    def __init__(self):
        self.notices, self.retryable = [], 0

    def append(self, context, items):
        context.extend(items)

    def show(self, items):
        pass

    def notice(self, text):
        self.notices.append(text)

    def retryable_failure(self):
        self.retryable += 1


def _turn(model, **config):
    """One turn of ``model`` on a fresh context, with waits of 0 s."""
    context = InteractionContext((Init(model="model"), Message("user", "hello")))
    host = _Host()
    with mock.patch.object(kernel, "AUTO_RETRY_DELAYS_SECONDS", (0,)):
        result = run_turn(context, model, Environment(), InteractionConfigSnapshot(
            enable_auto_compaction=False, **config), host)
    return result, context, host


def _failures(context):
    return [(item.category, item.error_code) for item in context if isinstance(item, ModelFailure)]


def _recovery(context):
    return [item.recovery for item in context if isinstance(item, SampleMetadata)]


class _RawResponse:
    """A response whose body is these bytes, as they are."""

    def __init__(self, body, status=200):
        self.status, self.headers, self.body, self.closed = status, {}, body, False

    def read(self, *args):
        return self.body

    def close(self):
        self.closed = True


def _messages(*outcomes):
    return MessagesModel(messages_tests._endpoint(api_url="http://localhost:8000", model="model"),
                         opener=messages_tests._ScriptedOpener(*outcomes))


def _messages_answer(text="Done."):
    return messages_tests._FakeResponse({
        "type": "message", "role": "assistant", "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })


def _chat(*outcomes):
    return ChatCompletionsModel(chat_endpoint(api_url="https://api.example.test"),
                                opener=messages_tests._ScriptedOpener(*outcomes))


def _chat_answer(text="Done."):
    return _FakeHTTPResponse({"choices": [{
        "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})


def _responses(*outcomes):
    return codex_model(responses_endpoint(api_url="https://api.example.test/v1",
                                          model="wire-model", bearer_token="api-key"),
                       opener=responses_tests._ScriptedOpener(*outcomes),
                       retry_sleep=lambda _delay: None)


def _stream(*events):
    return responses_tests._FakeSSEResponse(*events)


def _responses_answer(text="Done."):
    return _stream(responses_tests._message_event(0, text), responses_tests._completed_event())


class InvalidResponseTests(unittest.TestCase):
    """An undecodable or malformed body gets its failure, and is eligible:
    nothing of the response is kept."""

    def test_messages_and_chat_completions_bodies(self):
        for label, make, malformed in (
                ("Messages", _messages, {"type": "message"}),
                ("Chat Completions", _chat, {"choices": []})):
            for name, body in (("not UTF-8", b"\xff\xfe"), ("not JSON", b"{"),
                               ("malformed", json.dumps(malformed).encode())):
                with self.subTest(label=label, body=name):
                    model = make(_RawResponse(body))
                    with self.assertRaises(ModelResponseError) as raised:
                        model.sample(InteractionContext((Message("user", "hello"),)),
                                     sample_params=SampleParams(transient_retries=0))
                    error = raised.exception
                    self.assertIs(error.retry_state, RetryState.ELIGIBLE)
                    self.assertEqual(error.retry_class, "response")
                    self.assertEqual((error.failure.category, error.failure.message,
                                      error.failure.http_status, error.failure.attempt_count),
                                     ("invalid_response", f"{label} response was invalid",
                                      200, 1))

    def test_the_turn_loop_samples_an_invalid_body_again(self):
        for name, model in (("Messages", _messages(_RawResponse(b"{"), _messages_answer())),
                            ("Chat Completions", _chat(_RawResponse(b"{"), _chat_answer()))):
            with self.subTest(name):
                result, context, host = _turn(model)
                self.assertEqual(result.final_text, "Done.")
                self.assertEqual(_failures(context), [("invalid_response", None)])
                self.assertEqual(_recovery(context), [("invalid_response_retry",)])
                self.assertEqual(host.notices, ["Sampling again in 0 s (1/8)."])


class RefusalRetryTests(unittest.TestCase):
    """A refusal is eligible, of class refusal: the turn loop samples the
    same context again, nothing of the refused response adopted."""

    def test_a_messages_refusal_is_sampled_again(self):
        model = _messages(messages_tests._refused([]), _messages_answer())
        result, context, host = _turn(model)
        self.assertEqual(result.final_text, "Done.")
        self.assertEqual(_failures(context), [("refusal", "refusal")])
        self.assertEqual(_recovery(context), [("refusal_retry",)])

    def test_its_own_count_ends_it(self):
        refusals = [messages_tests._refused([]) for _ in range(2)]
        model = _messages(*refusals, _messages_answer())
        with self.assertRaises(ModelRefusal) as raised:
            _turn(model, endpoint_error_retries=5, endpoint_refusal_error_retries=1)
        self.assertIs(raised.exception.retry_state, RetryState.EXHAUSTED)
        self.assertEqual(raised.exception.retry_class, "refusal")


class ResponsesStreamTests(unittest.TestCase):
    """A Responses stream that failed, was cut short or held nothing: eligible,
    by its code's class."""

    def test_each_failed_stream_s_class(self):
        def failed(code):
            return {"type": "response.failed", "response": {"error": {"code": code,
                                                                      "message": "x"}}}

        def incomplete(reason):
            return {"type": "response.incomplete",
                    "response": {"incomplete_details": {"reason": reason}}}
        overloaded = {"type": "error", "code": "server_is_overloaded", "message": "x",
                      "sequence_number": 1}
        for name, events, category, retry_class in (
                ("overloaded", (overloaded,), "response_error_event", "transient"),
                ("server error", (failed("server_error"),), "response_failed", "transient"),
                ("other code", (failed("invalid_prompt"),), "response_failed", "response"),
                ("content filter", (incomplete("content_filter"),), "response_incomplete",
                 "refusal"),
                ("output limit", (incomplete("max_output_tokens"),), "response_incomplete",
                 "response"),
                ("empty", (responses_tests._completed_event(),), "empty_response",
                 "empty_response")):
            with self.subTest(name):
                with self.assertRaises(ModelResponseError) as raised:
                    _responses(_stream(*events)).sample(
                        InteractionContext((Message("user", "hello"),)),
                        sample_params=SampleParams(transient_retries=0))
                error = raised.exception
                self.assertEqual(error.failure.category, category)
                self.assertIs(error.retry_state, RetryState.ELIGIBLE)
                self.assertEqual(error.retry_class, retry_class)

    def test_a_failed_stream_keeps_its_output_for_the_turn_loop_to_drop(self):
        events = (responses_tests._message_event(0, "half"),
                  {"type": "response.failed", "response": {"error": {"code": "invalid_prompt"}}})
        with self.assertRaises(ModelResponseError) as raised:
            _responses(_stream(*events)).sample(InteractionContext((Message("user", "hello"),)),
                                                sample_params=SampleParams(transient_retries=0))
        self.assertIs(raised.exception.retry_state, RetryState.ELIGIBLE)
        self.assertEqual(raised.exception.completed_items, (Message("assistant", "half"),))
        # The loop retries it without the output, and the answer follows.
        result, context, host = _turn(_responses(_stream(*events), _responses_answer()))
        self.assertEqual(result.final_text, "Done.")
        self.assertNotIn(Message("assistant", "half"), context.items)
        [failure] = [item for item in context if isinstance(item, ModelFailure)]
        self.assertEqual(failure.completed_item_count, 1)

    def test_the_turn_loop_samples_an_overloaded_server_again(self):
        overloaded = _stream({"type": "error", "code": "server_is_overloaded", "message": "x",
                              "sequence_number": 1})
        result, context, host = _turn(_responses(overloaded, _responses_answer()))
        self.assertEqual(result.final_text, "Done.")
        self.assertEqual(_failures(context), [("response_error_event", "server_is_overloaded")])
        self.assertEqual(_recovery(context), [("server_is_overloaded_retry",)])
        self.assertEqual(host.notices, ["Sampling again in 0 s (1/8)."])


class CodexRefreshTests(unittest.TestCase):
    def test_credentials_from_its_own_refresh_are_not_refreshed_again(self):
        # As when the turn loop retries an auth failure: each attempt reloads
        # and retries once, but the refresh happens once.
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "auth.json"
            auth_file.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {
                "access_token": "old-token", "refresh_token": "old-refresh",
                "id_token": responses_tests._account_id_token("account-1"),
                "account_id": "account-1",
            }}), encoding="utf-8")
            opener = responses_tests._ScriptedOpener(
                *(responses_tests._http_error(401) for _ in range(5)))
            auth_opener = responses_tests._ScriptedOpener(responses_tests._FakeSSEResponse(
                status=200, body=json.dumps({
                    "access_token": "new-token", "refresh_token": "new-refresh",
                    "id_token": responses_tests._account_id_token("account-1"),
                }).encode("utf-8")))
            model = codex_model(model="codex-test", auth_file=auth_file, opener=opener,
                                auth_opener=auth_opener, retry_sleep=lambda _delay: None)
            errors = []
            for _ in range(2):
                with self.assertRaises(ModelAuthenticationError) as raised:
                    model.sample(InteractionContext([Message("user", "hello")]),
                                 sample_params=SampleParams(transient_retries=0))
                errors.append(raised.exception)
        self.assertEqual(len(auth_opener.calls), 1)
        self.assertEqual(len(opener.calls), 5)
        self.assertEqual([error.failure.recovery for error in errors], [
            ("credential_reload_unchanged", "http_401_retry", "oauth_refresh"),
            ("credential_reload_unchanged", "http_401_retry", "oauth_refresh_skipped"),
        ])
        self.assertEqual({(error.retry_state, error.retry_class) for error in errors},
                         {(RetryState.ELIGIBLE, "auth")})


if __name__ == "__main__":
    unittest.main()
