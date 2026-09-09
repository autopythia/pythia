from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from pythia.interaction import CODEX_RESPONSES_API_URL
from pythia.interaction import ChatCompletionsModel
from pythia.interaction import CodexAuth
from pythia.interaction import CodexResponsesModel
from pythia.interaction import DEFAULT_REQUEST_TIMEOUT_SECONDS
from pythia.interaction import Environment
from pythia.interaction import Init
from pythia.interaction import META_RESPONSES_API_URL
from pythia.interaction import Message
from pythia.interaction import ModelConfigurationError
from pythia.interaction import ModelContext
from pythia.interaction import ModelResponseError
from pythia.interaction import ModelTransportError
from pythia.interaction import OpaqueCompaction
from pythia.interaction import Reasoning
from pythia.interaction import SamplingOptions
from pythia.interaction import StreamingResponsesEndpoint
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import ToolSpec
from pythia.interaction import SampleMetadata
from pythia.interaction import UserInteraction
from pythia.interaction import UserInteractionBoundary
from pythia.interaction import load_interaction_save
from pythia.interaction import save_interaction_save
from pythia.interaction.demo import DEFAULT_PROMPT
from pythia.interaction.demo import EXPERIMENTAL_USER_MESSAGE_PROMPT
from pythia.interaction.demo import _build_model
from pythia.interaction.demo import _build_parser
from pythia.interaction.demo import run
from pythia.interaction.experimental_tools import create_inject_user_message_tool


def _event_lines(payload, *, event_name=None, crlf=False):
    ending = "\r\n" if crlf else "\n"
    lines = []
    if event_name is not None:
        lines.append(f"event: {event_name}{ending}".encode("utf-8"))
    lines.append(
        f"data: {json.dumps(payload, separators=(',', ':'))}{ending}".encode(
            "utf-8"
        )
    )
    lines.append(ending.encode("utf-8"))
    return lines


def _completed_event(
    *,
    input_tokens=0,
    output_tokens=0,
    total_tokens=0,
    cached_tokens=0,
):
    return {
        "type": "response.completed",
        "response": {
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "input_tokens_details": {
                    "cached_tokens": cached_tokens,
                },
            }
        },
    }


def _message_event(index, text):
    event = {
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": text,
                }
            ],
        },
    }
    if index is not None:
        event["output_index"] = index
    return event


def _reasoning_event(index, *, text, summary, encrypted_content):
    return {
        "type": "response.output_item.done",
        "output_index": index,
        "item": {
            "type": "reasoning",
            "summary": [
                {
                    "type": "summary_text",
                    "text": summary,
                }
            ],
            "content": [
                {
                    "type": "reasoning_text",
                    "text": text,
                }
            ],
            "encrypted_content": encrypted_content,
        },
    }


def _tool_call_event(index, *, call_id="call-1"):
    return {
        "type": "response.output_item.done",
        "output_index": index,
        "item": {
            "type": "function_call",
            "name": "lookup",
            "call_id": call_id,
            "arguments": '{"query":"pythia"}',
        },
    }


class _FakeSSEResponse:
    def __init__(
        self,
        *payloads,
        headers=None,
        status=200,
        body=b"",
        crlf=False,
    ):
        self.status = status
        self.headers = dict(headers or {})
        self.closed = False
        self._body = body
        self._lines = []
        for payload in payloads:
            self._lines.extend(_event_lines(payload, crlf=crlf))

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return self._body

    def close(self):
        self.closed = True


class _ScriptedOpener:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, request, *, timeout):
        self.calls.append((request, timeout))
        if not self.outcomes:
            raise AssertionError("unexpected HTTP request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _request_payload(opener, index=0):
    request, _ = opener.calls[index]
    return json.loads(request.data.decode("utf-8"))


def _request_headers(opener, index=0):
    request, _ = opener.calls[index]
    return {
        name.lower(): value
        for name, value in request.header_items()
    }


class StreamingResponsesEndpointTests(unittest.TestCase):
    def test_endpoint_normalizes_url_and_redacts_token(self):
        endpoint = StreamingResponsesEndpoint(
            api_url=" HTTPS://api.example.test:8443/proxy/root/ ",
            model=" codex-test ",
            bearer_token=" secret-token ",
            account_id=" account-1 ",
            api_provider=" CODEX ",
        )

        self.assertEqual(
            endpoint.api_url,
            "https://api.example.test:8443/proxy/root",
        )
        self.assertEqual(
            endpoint.url,
            "https://api.example.test:8443/proxy/root/responses",
        )
        self.assertEqual(endpoint.model, "codex-test")
        self.assertEqual(endpoint.account_id, "account-1")
        self.assertEqual(endpoint.api_provider, "codex")
        self.assertNotIn("secret-token", repr(endpoint))

    def test_endpoint_rejects_invalid_configuration(self):
        base = {
            "api_url": "https://api.example.test/v1",
            "model": "codex-test",
            "bearer_token": "token",
        }
        cases = (
            {"api_url": ""},
            {"api_url": "localhost:8000"},
            {"api_url": "ftp://localhost"},
            {"api_url": "http:///missing-host"},
            {"api_url": "http://user:password@localhost"},
            {"api_url": "http://localhost/prefix?query=value"},
            {"api_url": "http://localhost/prefix#fragment"},
            {"api_url": "http://localhost:not-a-port"},
            {"api_url": "http://localhost:0"},
            {"api_url": "http://localhost/v1/responses"},
            {"model": " "},
            {"bearer_token": " "},
            {"bearer_token": "two words"},
            {"api_provider": "unknown"},
            {"account_id": "account-1"},
            {"request_timeout_seconds": 0},
        )
        for override in cases:
            kwargs = dict(base)
            kwargs.update(override)
            with self.subTest(override=override):
                with self.assertRaises(ModelConfigurationError):
                    StreamingResponsesEndpoint(**kwargs)


class CodexResponsesConstructionTests(unittest.TestCase):
    def test_model_loads_auth_and_builds_default_codex_endpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "auth.json"
            auth_file.write_text(
                json.dumps(
                    {
                        "tokens": {
                            "access_token": "codex-token",
                            "account_id": "account-1",
                        }
                    }
                ),
                encoding="utf-8",
            )

            model = CodexResponsesModel(
                model="codex-test",
                auth_file=auth_file,
            )

        self.assertEqual(model.endpoint.api_url, CODEX_RESPONSES_API_URL)
        self.assertEqual(model.endpoint.model, "codex-test")
        self.assertEqual(model.endpoint.account_id, "account-1")
        self.assertEqual(model.endpoint.api_provider, "codex")
        self.assertEqual(
            model.endpoint.request_timeout_seconds,
            DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        self.assertNotIn("codex-token", repr(model.endpoint))

    def test_model_accepts_explicit_auth_and_endpoint_overrides(self):
        model = CodexResponsesModel(
            model="codex-test",
            auth=CodexAuth(
                access_token="override-token",
                account_id="account-2",
            ),
            api_url="https://proxy.example.test/codex",
            request_timeout_seconds=12,
        )

        self.assertEqual(
            model.endpoint.api_url,
            "https://proxy.example.test/codex",
        )
        self.assertEqual(model.endpoint.account_id, "account-2")
        self.assertEqual(model.endpoint.request_timeout_seconds, 12.0)

        endpoint = StreamingResponsesEndpoint(
            api_url="https://api.example.test/v1",
            model="generic-model",
            bearer_token="api-key",
        )
        overridden = CodexResponsesModel(endpoint)
        self.assertIs(overridden.endpoint, endpoint)

    def test_context_token_metadata_respects_model_and_provider_routes(self):
        cases = (
            ("gpt-5.6-sol", (272_000, 872_000)),
            ("gpt-5.6-sol-medium", (272_000, 872_000)),
            ("gpt-5.6-sol-max", (272_000, 872_000)),
            ("gpt-6-astra", (272_000, 872_000)),
            ("gpt-6-astra-medium", (272_000, 872_000)),
            ("gpt-6-astra-max", (272_000, 872_000)),
            ("muse-spark-1.3", (None, None)),
            ("gpt-5.6-sol-high", (None, None)),
            ("unknown-model", (None, None)),
        )
        for api_provider in ("codex", "api"):
            for requested_model, codex_limits in cases:
                with self.subTest(
                    model=requested_model,
                    api_provider=api_provider,
                ):
                    model = CodexResponsesModel(
                        StreamingResponsesEndpoint(
                            api_url="https://api.example.test/v1",
                            model=requested_model,
                            bearer_token="token",
                            api_provider=api_provider,
                        ),
                    )

                    self.assertEqual(
                        (
                            model.default_context_tokens,
                            model.max_context_tokens,
                        ),
                        codex_limits if api_provider == "codex" else (None, None),
                    )

    def test_muse_model_uses_meta_responses_endpoint_by_default(self):
        with mock.patch.dict(
            "os.environ",
            {"META_API_KEY": " meta-api-key "},
            clear=True,
        ):
            model = CodexResponsesModel(model=" muse-spark-1.3 ")

        self.assertEqual(model.endpoint.api_url, META_RESPONSES_API_URL)
        self.assertEqual(
            model.endpoint.url,
            "https://api.meta.ai/v1/responses",
        )
        self.assertEqual(model.endpoint.model, "muse-spark-1.3")
        self.assertEqual(model.endpoint.api_provider, "codex")
        self.assertEqual(model.endpoint.bearer_token, "meta-api-key")
        self.assertIsNone(model.endpoint.account_id)
        self.assertNotIn("meta-api-key", repr(model.endpoint))

    def test_muse_model_requires_meta_api_key_without_explicit_auth(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(
                ModelConfigurationError,
                "META_API_KEY",
            ):
                CodexResponsesModel(model="muse-spark-1.3-xhigh")

    def test_explicit_api_url_overrides_muse_model_default(self):
        model = CodexResponsesModel(
            model="muse-spark-1.3",
            auth=CodexAuth(access_token="codex-token"),
            api_url="https://proxy.example.test/meta",
        )

        self.assertEqual(
            model.endpoint.api_url,
            "https://proxy.example.test/meta",
        )

    def test_model_rejects_missing_or_conflicting_construction_options(self):
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "model is required",
        ):
            CodexResponsesModel()
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "model must not be empty",
        ):
            CodexResponsesModel(
                model=" ",
                auth=CodexAuth(access_token="token"),
            )

        endpoint = StreamingResponsesEndpoint(
            api_url="https://api.example.test/v1",
            model="generic-model",
            bearer_token="api-key",
        )
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "cannot be combined",
        ):
            CodexResponsesModel(
                endpoint,
                model="other-model",
            )
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "auth cannot be combined",
        ):
            CodexResponsesModel(
                model="codex-test",
                auth=CodexAuth(access_token="token"),
                auth_file="/tmp/auth.json",
            )


class CodexResponsesModelTests(unittest.TestCase):
    def test_responses_opaque_compaction_subtype_is_enforced(self):
        response = _FakeSSEResponse(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "compaction",
                    "encrypted_content": "new-encrypted-summary",
                },
            },
            _message_event(1, "continued"),
            _completed_event(),
        )
        opener = _ScriptedOpener(response)
        model = CodexResponsesModel(
            StreamingResponsesEndpoint(
                api_url="http://localhost:8000/v1",
                model="model",
                bearer_token="token",
                api_provider="api",
            ),
            opener=opener,
        )

        sample = model.sample(
            ModelContext(
                (
                    OpaqueCompaction.from_responses("encrypted-summary"),
                    Message(role="user", content="continue"),
                )
            )
        )

        self.assertEqual(
            _request_payload(opener)["input"][0],
            {
                "type": "compaction",
                "encrypted_content": "encrypted-summary",
            },
        )
        self.assertEqual(
            sample.items,
            (
                OpaqueCompaction.from_responses("new-encrypted-summary"),
                Message(role="assistant", content="continued"),
            ),
        )
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "Messages opaque compaction",
        ):
            model.sample(
                ModelContext(
                    (
                        OpaqueCompaction.from_messages("summary"),
                        Message(role="user", content="continue"),
                    )
                )
            )

    def test_session_init_owns_codex_session_and_prompt_cache_key(self):
        opener = _ScriptedOpener(
            _FakeSSEResponse(
                _message_event(0, "done"),
                _completed_event(),
            )
        )

        turn_identifiers = iter(("turn-1",))
        model = CodexResponsesModel(
            StreamingResponsesEndpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="token",
                api_provider="codex",
            ),
            opener=opener,
            identifier_factory=lambda: next(turn_identifiers),
        )
        context = ModelContext(
            (
                Init("session-from-context"),
                Message(role="user", content="hello"),
            )
        )

        sample = model.sample(context)

        payload = _request_payload(opener)
        headers = _request_headers(opener)
        self.assertEqual(
            payload["prompt_cache_key"],
            "session-from-context",
        )
        self.assertEqual(headers["session_id"], "session-from-context")
        self.assertIsNone(sample.provider_session_id)
        metadata = sample.context_items()[-2]
        self.assertIsNone(metadata.provider_session_id)

    def test_codex_reasoning_model_aliases_set_base_model_and_effort(self):
        cases = (
            ("gpt-5.6-sol-medium", "gpt-5.6-sol", "medium"),
            ("gpt-5.6-sol-max", "gpt-5.6-sol", "max"),
            ("gpt-5.6-sol", "gpt-5.6-sol", None),
            ("gpt-5.6-sol-high", "gpt-5.6-sol-high", None),
        )
        for requested_model, expected_model, expected_effort in cases:
            with self.subTest(model=requested_model):
                opener = _ScriptedOpener(
                    _FakeSSEResponse(
                        _message_event(0, "done"),
                        _completed_event(),
                    )
                )
                model = CodexResponsesModel(
                    StreamingResponsesEndpoint(
                        api_url=CODEX_RESPONSES_API_URL,
                        model=requested_model,
                        bearer_token="token",
                        api_provider="codex",
                    ),
                    opener=opener,
                )

                model.sample(
                    ModelContext([Message(role="user", content="hello")])
                )

                payload = _request_payload(opener)
                self.assertEqual(payload["model"], expected_model)
                self.assertNotIn("default_context_tokens", payload)
                self.assertNotIn("max_context_tokens", payload)
                if expected_effort is None:
                    self.assertNotIn("reasoning", payload)
                else:
                    self.assertEqual(
                        payload["reasoning"],
                        {"effort": expected_effort},
                    )

    def test_astra_models_set_reasoning_and_low_verbosity(self):
        cases = (
            ("gpt-6-astra", {"summary": "auto"}),
            ("gpt-6-astra-medium", {"effort": "medium", "summary": "auto"}),
            ("gpt-6-astra-max", {"effort": "max", "summary": "auto"}),
        )
        for requested_model, expected_reasoning in cases:
            with self.subTest(model=requested_model):
                opener = _ScriptedOpener(
                    _FakeSSEResponse(
                        _message_event(0, "done"),
                        _completed_event(),
                    )
                )
                model = CodexResponsesModel(
                    model=requested_model,
                    auth=CodexAuth(access_token="token"),
                    opener=opener,
                )

                model.sample(
                    ModelContext([Message(role="user", content="hello")])
                )

                request, _ = opener.calls[0]
                payload = _request_payload(opener)
                self.assertEqual(
                    request.full_url,
                    f"{CODEX_RESPONSES_API_URL}/responses",
                )
                self.assertEqual(payload["model"], "gpt-6-astra")
                self.assertEqual(payload["reasoning"], expected_reasoning)
                self.assertEqual(payload["text"], {"verbosity": "low"})
                self.assertNotIn("default_context_tokens", payload)
                self.assertNotIn("max_context_tokens", payload)

    def test_reasoning_model_aliases_are_not_applied_to_generic_responses(self):
        for requested_model in (
            "gpt-5.6-sol-max",
            "gpt-6-astra",
            "gpt-6-astra-medium",
            "gpt-6-astra-max",
        ):
            with self.subTest(model=requested_model):
                opener = _ScriptedOpener(
                    _FakeSSEResponse(
                        _message_event(0, "done"),
                        _completed_event(),
                    )
                )
                model = CodexResponsesModel(
                    StreamingResponsesEndpoint(
                        api_url="https://api.example.test/v1",
                        model=requested_model,
                        bearer_token="api-key",
                        api_provider="api",
                    ),
                    opener=opener,
                )

                model.sample(
                    ModelContext([Message(role="user", content="hello")])
                )

                payload = _request_payload(opener)
                self.assertEqual(payload["model"], requested_model)
                self.assertNotIn("reasoning", payload)
                self.assertNotIn("text", payload)

    def test_muse_models_route_to_contributor_model_and_reasoning(self):
        cases = (
            ("muse-spark-1.3", None),
            ("muse-spark-1.3-xhigh", "xhigh"),
        )
        for requested_model, expected_effort in cases:
            with self.subTest(model=requested_model):
                opener = _ScriptedOpener(
                    _FakeSSEResponse(
                        _message_event(0, "done"),
                        _completed_event(),
                    )
                )
                model = CodexResponsesModel(
                    model=requested_model,
                    auth=CodexAuth(access_token="meta-api-key"),
                    opener=opener,
                    identifier_factory=(
                        iter(("session-1", "turn-1")).__next__
                    ),
                )

                model.sample(
                    ModelContext([Message(role="user", content="hello")])
                )

                request, _ = opener.calls[0]
                payload = _request_payload(opener)
                self.assertEqual(
                    request.full_url,
                    "https://api.meta.ai/v1/responses",
                )
                self.assertEqual(
                    payload["model"],
                    "muse-spark-1.3-contributor",
                )
                if expected_effort is None:
                    self.assertNotIn("reasoning", payload)
                else:
                    self.assertEqual(
                        payload["reasoning"],
                        {"effort": expected_effort},
                    )

    def test_sample_encodes_request_and_collects_ordered_output_items(self):
        response = _FakeSSEResponse(
            _tool_call_event(2),
            _reasoning_event(
                0,
                text="Need repository facts.",
                summary="Inspect the repository.",
                encrypted_content="encrypted-reasoning",
            ),
            _message_event(1, "I will inspect it."),
            _completed_event(
                input_tokens=30,
                output_tokens=8,
                total_tokens=38,
                cached_tokens=6,
            ),
            headers={"x-codex-turn-state": "turn-state-1"},
            crlf=True,
        )
        opener = _ScriptedOpener(response)
        identifiers = iter(("session-1", "turn-1"))
        model = CodexResponsesModel(
            StreamingResponsesEndpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="secret-token",
                account_id="account-1",
                api_provider="codex",
            ),
            opener=opener,
            identifier_factory=lambda: next(identifiers),
        )
        context = ModelContext(
            UserInteraction(
                items=(Message(role="user", content="Summarize this repo."),)
            ).context_items()
        )
        before = context.items
        tool = ToolSpec(
            name="lookup",
            description="Look up repository facts.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        )

        sample = model.sample(
            context,
            tools=(tool,),
            options=SamplingOptions(max_tokens=200),
        )

        self.assertEqual(context.items, before)
        self.assertEqual(
            sample.items,
            (
                Reasoning(
                    content="Need repository facts.",
                    summary=("Inspect the repository.",),
                    encrypted_content="encrypted-reasoning",
                ),
                Message(role="assistant", content="I will inspect it."),
                ToolCall(
                    name="lookup",
                    call_id="call-1",
                    arguments_json='{"query":"pythia"}',
                ),
            ),
        )
        self.assertEqual(sample.stop_reason, "tool_use")
        self.assertEqual(sample.usage.input_tokens, 30)
        self.assertEqual(sample.usage.output_tokens, 8)
        self.assertEqual(sample.usage.total_tokens, 38)
        self.assertEqual(sample.usage.cached_input_tokens, 6)
        self.assertEqual(sample.provider_session_id, "session-1")
        self.assertEqual(sample.provider_turn_id, "turn-1")
        self.assertEqual(sample.provider_turn_state, "turn-state-1")
        metadata = sample.context_items()[-2]
        self.assertIsInstance(metadata, SampleMetadata)
        self.assertEqual(metadata.provider_session_id, "session-1")
        self.assertEqual(metadata.provider_turn_id, "turn-1")
        self.assertEqual(metadata.provider_turn_state, "turn-state-1")
        self.assertNotIn("turn-state-1", repr(sample))
        self.assertNotIn("turn-state-1", repr(metadata))
        self.assertTrue(response.closed)

        request, timeout = opener.calls[0]
        payload = _request_payload(opener)
        headers = _request_headers(opener)
        self.assertEqual(timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(
            request.full_url,
            f"{CODEX_RESPONSES_API_URL}/responses",
        )
        self.assertEqual(headers["authorization"], "Bearer secret-token")
        self.assertEqual(headers["chatgpt-account-id"], "account-1")
        self.assertEqual(headers["session_id"], "session-1")
        self.assertNotIn("x-codex-turn-state", headers)
        self.assertEqual(
            json.loads(headers["x-codex-turn-metadata"]),
            {
                "turn_id": "turn-1",
                "sandbox": "none",
            },
        )
        self.assertEqual(payload["model"], "codex-test")
        self.assertTrue(payload["stream"])
        self.assertFalse(payload["store"])
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(
            payload["include"],
            ["reasoning.encrypted_content"],
        )
        self.assertEqual(payload["prompt_cache_key"], "session-1")
        self.assertEqual(payload["max_output_tokens"], 200)
        self.assertEqual(
            payload["input"],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Summarize this repo.",
                        }
                    ],
                }
            ],
        )
        self.assertEqual(
            payload["tools"],
            [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up repository facts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                    },
                    "strict": False,
                }
            ],
        )

    def test_sol_medium_experiment_encodes_user_message_and_preserves_turn(self):
        tool = create_inject_user_message_tool()
        opener = _ScriptedOpener(
            _FakeSSEResponse(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "name": tool.spec.name,
                        "call_id": "inject-1",
                        "arguments": "{}",
                    },
                },
                _completed_event(),
                headers={"x-codex-turn-state": "sticky-state"},
            ),
            _FakeSSEResponse(
                _message_event(0, "received: hello world"), _completed_event(),
            ),
        )
        model = CodexResponsesModel(
            model="gpt-5.6-sol-medium",
            auth=CodexAuth(access_token="test-token"),
            opener=opener,
            # A second generated turn ID would exhaust the iterator.
            identifier_factory=iter(("turn-1",)).__next__,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.jsonl"
            with mock.patch("builtins.print"):
                answer = run(
                    model, Environment((tool,)),
                    prompt=EXPERIMENTAL_USER_MESSAGE_PROMPT,
                    max_samples=2, save_path=path,
                )
            restored = load_interaction_save(path)
        self.assertEqual(answer, "received: hello world")
        self.assertEqual(len(opener.calls), 2)
        for index in (0, 1):
            payload = _request_payload(opener, index)
            self.assertEqual(payload["model"], "gpt-5.6-sol")
            self.assertEqual(payload["reasoning"], {"effort": "medium"})
            self.assertEqual(payload["tool_choice"], "auto")
            self.assertEqual(tuple(spec["name"] for spec in payload["tools"]), (tool.spec.name,))
        self.assertNotIn("hello world", json.dumps(_request_payload(opener, 0)))
        self.assertEqual(_request_payload(opener, 1)["input"][-2:], [
            {
                "type": "function_call_output",
                "call_id": "inject-1",
                "output": "Synthetic user message queued.",
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello world"}],
            },
        ])
        first_headers = _request_headers(opener, 0)
        second_headers = _request_headers(opener, 1)
        self.assertEqual(second_headers["session_id"], first_headers["session_id"])
        self.assertEqual(second_headers["x-codex-turn-state"], "sticky-state")
        self.assertEqual(
            json.loads(second_headers["x-codex-turn-metadata"])["turn_id"], "turn-1",
        )
        self.assertEqual(restored.items.count(Message("user", "hello world")), 1)
        self.assertEqual(restored.items.count(UserInteractionBoundary()), 1)

    def test_context_owns_sticky_turn_state_and_resets_it_for_new_user_turn(self):
        first_response = _FakeSSEResponse(
            _reasoning_event(
                0,
                text="Use the tool.",
                summary="Tool needed.",
                encrypted_content="encrypted-1",
            ),
            _tool_call_event(1),
            _completed_event(),
            headers={"x-codex-turn-state": "sticky-state"},
        )
        second_response = _FakeSSEResponse(
            _message_event(0, "First turn complete."),
            _completed_event(),
            headers={"x-codex-turn-state": "replacement-state"},
        )
        third_response = _FakeSSEResponse(
            _message_event(0, "Second turn complete."),
            _completed_event(),
            headers={"x-codex-turn-state": "second-turn-state"},
        )
        opener = _ScriptedOpener(
            first_response,
            second_response,
            third_response,
        )
        identifiers = iter(("session-1", "turn-1", "turn-2"))
        model = CodexResponsesModel(
            StreamingResponsesEndpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="token",
                api_provider="codex",
            ),
            opener=opener,
            identifier_factory=lambda: next(identifiers),
        )
        context = ModelContext(
            UserInteraction(
                items=(Message(role="user", content="First turn"),)
            ).context_items()
        )

        first = model.sample(context)
        context.extend(first.context_items())
        context.append(
            ToolResult(
                call_id="call-1",
                output="repository facts",
            )
        )
        second = model.sample(context)
        context.extend(second.context_items())
        context.extend(
            UserInteraction(
                items=(Message(role="user", content="Second turn"),)
            ).context_items()
        )
        third = model.sample(context)

        second_headers = _request_headers(opener, 1)
        self.assertEqual(second_headers["session_id"], "session-1")
        self.assertEqual(
            second_headers["x-codex-turn-state"],
            "sticky-state",
        )
        self.assertEqual(
            json.loads(second_headers["x-codex-turn-metadata"])["turn_id"],
            "turn-1",
        )
        second_input = _request_payload(opener, 1)["input"]
        self.assertIn(
            {
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "Tool needed.",
                    }
                ],
                "content": [
                    {
                        "type": "reasoning_text",
                        "text": "Use the tool.",
                    }
                ],
                "encrypted_content": "encrypted-1",
            },
            second_input,
        )
        self.assertEqual(
            second_input[-1],
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "repository facts",
            },
        )
        self.assertEqual(second.provider_turn_state, "sticky-state")

        third_headers = _request_headers(opener, 2)
        self.assertEqual(third_headers["session_id"], "session-1")
        self.assertNotIn("x-codex-turn-state", third_headers)
        self.assertEqual(
            json.loads(third_headers["x-codex-turn-metadata"])["turn_id"],
            "turn-2",
        )
        self.assertEqual(third.provider_session_id, "session-1")
        self.assertEqual(third.provider_turn_id, "turn-2")
        self.assertEqual(
            third.provider_turn_state,
            "second-turn-state",
        )

    def test_generic_responses_endpoint_omits_codex_state(self):
        response = _FakeSSEResponse(
            _message_event(0, "generic response"),
            _completed_event(),
        )
        opener = _ScriptedOpener(response)

        def unexpected_identifier():
            raise AssertionError("generic Responses must not create Codex IDs")

        model = CodexResponsesModel(
            StreamingResponsesEndpoint(
                api_url="https://api.example.test/v1",
                model="generic-model",
                bearer_token="api-key",
            ),
            opener=opener,
            identifier_factory=unexpected_identifier,
        )

        sample = model.sample(
            ModelContext([Message(role="user", content="hello")])
        )

        headers = _request_headers(opener)
        payload = _request_payload(opener)
        self.assertNotIn("chatgpt-account-id", headers)
        self.assertNotIn("session_id", headers)
        self.assertNotIn("x-codex-turn-metadata", headers)
        self.assertNotIn("x-codex-turn-state", headers)
        self.assertNotIn("prompt_cache_key", payload)
        self.assertIsNone(sample.provider_session_id)
        self.assertIsNone(sample.provider_turn_id)
        self.assertIsNone(sample.provider_turn_state)
        self.assertEqual(sample.last_assistant_text, "generic response")

    def test_session_resume_replays_state_after_pending_tool_call(self):
        first_response = _FakeSSEResponse(
            _tool_call_event(0),
            _completed_event(),
            headers={"x-codex-turn-state": "resume-state"},
        )
        endpoint = StreamingResponsesEndpoint(
            api_url=CODEX_RESPONSES_API_URL,
            model="codex-test",
            bearer_token="token",
            api_provider="codex",
        )
        identifiers = iter(("session-1", "turn-1"))
        first_model = CodexResponsesModel(
            endpoint,
            opener=_ScriptedOpener(first_response),
            identifier_factory=lambda: next(identifiers),
        )
        context = ModelContext(
            UserInteraction(
                items=(Message(role="user", content="Use the tool"),)
            ).context_items()
        )
        first_sample = first_model.sample(context)
        context.extend(first_sample.context_items())

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "interaction.jsonl"
            save_interaction_save(save_path, context)
            resumed = load_interaction_save(save_path)

        resumed.append(
            ToolResult(
                call_id="call-1",
                output="resumed result",
            )
        )
        second_response = _FakeSSEResponse(
            _message_event(0, "resumed answer"),
            _completed_event(),
        )
        opener = _ScriptedOpener(second_response)

        def unexpected_identifier():
            raise AssertionError("resumed context must supply provider IDs")

        second_model = CodexResponsesModel(
            endpoint,
            opener=opener,
            identifier_factory=unexpected_identifier,
        )
        second_sample = second_model.sample(resumed)

        headers = _request_headers(opener)
        self.assertEqual(headers["session_id"], "session-1")
        self.assertEqual(
            headers["x-codex-turn-state"],
            "resume-state",
        )
        self.assertEqual(
            json.loads(headers["x-codex-turn-metadata"])["turn_id"],
            "turn-1",
        )
        self.assertEqual(
            second_sample.last_assistant_text,
            "resumed answer",
        )

    def test_named_multiline_sse_event_is_supported(self):
        item = _message_event(0, "multiline")
        completed = _completed_event()
        item_json = json.dumps(item, separators=(",", ":"))
        split_at = item_json.index('"item"')
        lines = [
            b"event: response.output_item.done\r\n",
            f"data: {item_json[:split_at]}\r\n".encode("utf-8"),
            f"data: {item_json[split_at:]}\r\n".encode("utf-8"),
            b"\r\n",
            b": keepalive\r\n",
            *_event_lines(completed, crlf=True),
        ]
        response = _FakeSSEResponse()
        response._lines = lines
        opener = _ScriptedOpener(response)
        model = CodexResponsesModel(
            StreamingResponsesEndpoint(
                api_url="https://api.example.test/v1",
                model="generic-model",
                bearer_token="api-key",
            ),
            opener=opener,
        )

        sample = model.sample(
            ModelContext([Message(role="user", content="hello")])
        )

        self.assertEqual(sample.last_assistant_text, "multiline")

    def test_output_items_without_indices_preserve_event_order(self):
        response = _FakeSSEResponse(
            _message_event(None, "first"),
            _message_event(None, "second"),
            _completed_event(),
        )
        model = CodexResponsesModel(
            StreamingResponsesEndpoint(
                api_url="https://api.example.test/v1",
                model="generic-model",
                bearer_token="api-key",
            ),
            opener=_ScriptedOpener(response),
        )

        sample = model.sample(
            ModelContext([Message(role="user", content="hello")])
        )

        self.assertEqual(
            sample.items,
            (
                Message(role="assistant", content="first"),
                Message(role="assistant", content="second"),
            ),
        )

    def test_errors_are_typed_and_unsupported_options_are_rejected(self):
        model = CodexResponsesModel(
            StreamingResponsesEndpoint(
                api_url="https://api.example.test/v1",
                model="generic-model",
                bearer_token="api-key",
            ),
            opener=_ScriptedOpener(),
        )
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "temperature",
        ):
            model.sample(
                ModelContext([Message(role="user", content="hello")]),
                options=SamplingOptions(temperature=0.5),
            )

        incomplete_response = _FakeSSEResponse(
            _message_event(0, "partial"),
        )
        with self.assertRaisesRegex(ModelResponseError, "before"):
            CodexResponsesModel(
                model.endpoint,
                opener=_ScriptedOpener(incomplete_response),
            ).sample(
                ModelContext([Message(role="user", content="hello")])
            )
        self.assertTrue(incomplete_response.closed)

        unsupported_response = _FakeSSEResponse(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "custom_tool_call",
                    "name": "custom",
                },
            },
            _completed_event(),
        )
        with self.assertRaisesRegex(ModelResponseError, "unsupported"):
            CodexResponsesModel(
                model.endpoint,
                opener=_ScriptedOpener(unsupported_response),
            ).sample(
                ModelContext([Message(role="user", content="hello")])
            )

        malformed_response = _FakeSSEResponse()
        malformed_response._lines = [b"data: {\n", b"\n"]
        with self.assertRaisesRegex(ModelResponseError, "invalid JSON"):
            CodexResponsesModel(
                model.endpoint,
                opener=_ScriptedOpener(malformed_response),
            ).sample(
                ModelContext([Message(role="user", content="hello")])
            )

    def test_codex_401_is_actionable_and_does_not_expose_token(self):
        error = urllib.error.HTTPError(
            CODEX_RESPONSES_API_URL,
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":{"message":"expired"}}'),
        )
        model = CodexResponsesModel(
            StreamingResponsesEndpoint(
                api_url=CODEX_RESPONSES_API_URL,
                model="codex-test",
                bearer_token="secret-token",
                api_provider="codex",
            ),
            opener=_ScriptedOpener(error),
            identifier_factory=iter(("session-1", "turn-1")).__next__,
        )

        with self.assertRaisesRegex(
            ModelTransportError,
            "codex login",
        ) as raised:
            model.sample(
                ModelContext([Message(role="user", content="hello")])
            )

        self.assertNotIn("secret-token", str(raised.exception))


class DemoConfigurationTests(unittest.TestCase):
    def test_demo_max_samples_is_unbounded_by_default(self):
        args = _build_parser().parse_args([])

        self.assertIsNone(args.max_samples)
        self.assertEqual(
            _build_parser().parse_args(["--max-samples=25"]).max_samples,
            25,
        )

    def test_demo_defaults_to_chat_completions_and_repository_prompt(self):
        args = _build_parser().parse_args([])
        model = _build_model(args)

        self.assertIsInstance(model, ChatCompletionsModel)
        self.assertEqual(
            model.endpoint.api_url,
            "http://127.0.0.1:8000",
        )
        self.assertEqual(
            DEFAULT_PROMPT,
            "Summarize the repository in the current working directory.",
        )

    def test_demo_builds_optional_codex_responses_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "auth.json"
            auth_file.write_text(
                json.dumps(
                    {
                        "tokens": {
                            "access_token": "codex-token",
                            "account_id": "account-1",
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = _build_parser().parse_args(
                [
                    "--model-api",
                    "codex-responses",
                    "--model",
                    "codex-test",
                    "--codex-auth-file",
                    str(auth_file),
                ]
            )

            model = _build_model(args)

        self.assertIsInstance(model, CodexResponsesModel)
        self.assertEqual(model.endpoint.api_url, CODEX_RESPONSES_API_URL)
        self.assertEqual(model.endpoint.model, "codex-test")
        self.assertEqual(model.endpoint.account_id, "account-1")
        self.assertNotIn("codex-token", repr(model.endpoint))

    def test_demo_accepts_codex_model_api_shorthand(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = Path(tmpdir) / "auth.json"
            auth_file.write_text(
                json.dumps(
                    {
                        "tokens": {
                            "access_token": "codex-token",
                            "account_id": "account-1",
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = _build_parser().parse_args(
                [
                    "--model-api=codex",
                    "--model",
                    "codex-test",
                    "--codex-auth-file",
                    str(auth_file),
                ]
            )

            model = _build_model(args)

        self.assertIsInstance(model, CodexResponsesModel)
        self.assertEqual(model.endpoint.api_url, CODEX_RESPONSES_API_URL)
        self.assertEqual(model.endpoint.model, "codex-test")
        self.assertEqual(model.endpoint.account_id, "account-1")

    def test_demo_builds_muse_model_from_meta_api_key(self):
        args = _build_parser().parse_args(
            [
                "--model-api=codex",
                "--model=muse-spark-1.3-xhigh",
            ]
        )

        with mock.patch.dict(
            "os.environ",
            {"META_API_KEY": "meta-api-key"},
            clear=True,
        ):
            model = _build_model(args)

        self.assertIsInstance(model, CodexResponsesModel)
        self.assertEqual(model.endpoint.api_url, META_RESPONSES_API_URL)
        self.assertEqual(model.endpoint.model, "muse-spark-1.3-xhigh")
        self.assertEqual(model.endpoint.bearer_token, "meta-api-key")
        self.assertIsNone(model.endpoint.account_id)
        self.assertNotIn("meta-api-key", repr(model.endpoint))

    def test_demo_rejects_ambiguous_auth_options(self):
        with self.assertRaisesRegex(ValueError, "required"):
            _build_model(
                _build_parser().parse_args(
                    ["--model-api", "codex-responses"]
                )
            )
        with self.assertRaisesRegex(ValueError, "not used"):
            _build_model(
                _build_parser().parse_args(
                    [
                        "--model-api",
                        "codex-responses",
                        "--model",
                        "codex-test",
                        "--api-key",
                        "token",
                    ]
                )
            )
        with self.assertRaisesRegex(ValueError, "require"):
            _build_model(
                _build_parser().parse_args(
                    ["--codex-home", "/tmp/codex"]
                )
            )

    def test_demo_rejects_unsupported_model_api_namespace(self):
        args = _build_parser().parse_args([])
        args.model_api = "unsupported-api"

        with self.assertRaisesRegex(
            ValueError,
            "unsupported model API: 'unsupported-api'",
        ):
            _build_model(args)


if __name__ == "__main__":
    unittest.main()
