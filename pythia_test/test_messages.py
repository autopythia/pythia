from __future__ import annotations

import io
import json
import urllib.error
import unittest
from typing import Any
from unittest import mock

from pythia.interaction import ANTHROPIC_MESSAGES_API_URL
from pythia.interaction import Message
from pythia.interaction import MessagesEndpoint
from pythia.interaction import MessagesModel
from pythia.interaction import ModelConfigurationError
from pythia.interaction import ModelContext
from pythia.interaction import ModelContextWindowError
from pythia.interaction import ModelResponseError
from pythia.interaction import Reasoning
from pythia.interaction import SamplingOptions
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import ToolSpec
from pythia.interaction import UserInteractionBoundary
from pythia.interaction.demo import _build_model
from pythia.interaction.demo import _build_parser


class _FakeResponse:
    def __init__(self, payload: Any, *, status: int = 200):
        self.status = status
        self.payload = payload
        self.closed = False

    def read(self) -> bytes:
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")

    def close(self) -> None:
        self.closed = True


class _Opener:
    def __init__(self, response: Any):
        self.response = response
        self.calls = []

    def __call__(self, request, *, timeout):
        self.calls.append((request, timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _payload(opener: _Opener) -> dict[str, Any]:
    request, _ = opener.calls[-1]
    return json.loads(request.data.decode("utf-8"))


class MessagesEndpointTests(unittest.TestCase):
    def test_normalizes_url_and_redacts_key(self):
        endpoint = MessagesEndpoint(
            api_url=" HTTPS://api.example.test:8443/proxy/ ",
            model=" model-name ",
            api_key=" secret-key ",
        )

        self.assertEqual(endpoint.api_url, "https://api.example.test:8443/proxy")
        self.assertEqual(
            endpoint.url,
            "https://api.example.test:8443/proxy/v1/messages",
        )
        self.assertEqual(endpoint.model, "model-name")
        self.assertEqual(endpoint.api_key, "secret-key")
        self.assertNotIn("secret-key", repr(endpoint))
        self.assertEqual(
            ANTHROPIC_MESSAGES_API_URL,
            "https://api.anthropic.com",
        )

    def test_rejects_invalid_configuration(self):
        cases = [
            {"api_url": "", "model": "model"},
            {"api_url": "localhost", "model": "model"},
            {"api_url": "ftp://localhost", "model": "model"},
            {"api_url": "http://user:pass@localhost", "model": "model"},
            {"api_url": "http://localhost/v1/messages", "model": "model"},
            {"api_url": "http://localhost", "model": ""},
            {"api_url": "http://localhost", "model": "model", "api_key": ""},
            {
                "api_url": "http://localhost",
                "model": "model",
                "anthropic_version": "bad\nheader",
            },
            {
                "api_url": "http://localhost",
                "model": "model",
                "default_max_tokens": 0,
            },
            {
                "api_url": "http://localhost",
                "model": "model",
                "request_timeout_seconds": 0,
            },
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ModelConfigurationError):
                    MessagesEndpoint(**kwargs)


class MessagesModelTests(unittest.TestCase):
    def test_encodes_context_tools_options_and_decodes_response(self):
        response = _FakeResponse(
            {
                "type": "message",
                "role": "assistant",
                "model": "resolved-model",
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Need the tool.",
                        "signature": "new-signature",
                    },
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "call-new",
                        "name": "lookup",
                        "input": {"q": "new"},
                    },
                ],
                "usage": {
                    "input_tokens": 20,
                    "cache_creation_input_tokens": 7,
                    "cache_read_input_tokens": 4,
                    "output_tokens": 5,
                },
            }
        )
        opener = _Opener(response)
        model = MessagesModel(
            MessagesEndpoint(
                api_url="https://api.example.test/anthropic",
                model="claude-sonnet-5",
                api_key="secret-key",
                default_max_tokens=2048,
            ),
            opener=opener,
        )
        context = ModelContext(
            (
                Message(role="system", text="system text"),
                Message(role="developer", text="developer text"),
                Message(role="user", text="question"),
                UserInteractionBoundary(),
                Reasoning(
                    text="prior thought",
                    encrypted_content="responses-only-data",
                    content_signature="prior-signature",
                ),
                Message(role="assistant", text="calling"),
                ToolCall(
                    name="lookup",
                    call_id="call-old",
                    arguments_json='{"q":"old"}',
                ),
                ToolResult(call_id="call-old", output="result", success=False),
            )
        )
        tool = ToolSpec(
            name="lookup",
            description="Look something up.",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )

        sample = model.sample(
            context,
            tools=(tool,),
            options=SamplingOptions(
                max_tokens=100,
                temperature=0.25,
                top_p=0.9,
                stop=("END",),
            ),
        )

        request, timeout = opener.calls[0]
        payload = _payload(opener)
        self.assertEqual(timeout, 60.0)
        self.assertEqual(
            request.full_url,
            "https://api.example.test/anthropic/v1/messages",
        )
        self.assertEqual(request.get_header("X-api-key"), "secret-key")
        self.assertEqual(request.get_header("Anthropic-version"), "2023-06-01")
        self.assertEqual(payload["model"], "claude-sonnet-5")
        self.assertEqual(payload["max_tokens"], 100)
        self.assertEqual(payload["temperature"], 0.25)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["stop_sequences"], ["END"])
        self.assertFalse(payload["stream"])
        self.assertEqual(
            payload["system"],
            [
                {"type": "text", "text": "system text"},
                {"type": "text", "text": "developer text"},
            ],
        )
        self.assertEqual(
            payload["messages"],
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "question"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "prior thought",
                            "signature": "prior-signature",
                        },
                        {"type": "text", "text": "calling"},
                        {
                            "type": "tool_use",
                            "id": "call-old",
                            "name": "lookup",
                            "input": {"q": "old"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-old",
                            "content": "result",
                            "is_error": True,
                        }
                    ],
                },
            ],
        )
        self.assertNotIn("responses-only-data", request.data.decode("utf-8"))
        self.assertEqual(
            payload["tools"],
            [
                {
                    "name": "lookup",
                    "description": "Look something up.",
                    "input_schema": tool.parameters,
                }
            ],
        )
        self.assertEqual(
            sample.items,
            (
                Reasoning(
                    text="Need the tool.",
                    content_signature="new-signature",
                ),
                Message(role="assistant", text="Checking."),
                ToolCall(
                    name="lookup",
                    call_id="call-new",
                    arguments_json='{"q":"new"}',
                ),
            ),
        )
        self.assertEqual(sample.stop_reason, "tool_use")
        self.assertEqual(sample.usage.input_tokens, 20)
        self.assertEqual(sample.usage.output_tokens, 5)
        self.assertEqual(sample.usage.total_tokens, 25)
        self.assertEqual(sample.usage.cached_input_tokens, 4)
        self.assertTrue(response.closed)

    def test_uses_default_tokens_and_omits_optional_fields(self):
        opener = _Opener(
            _FakeResponse(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                }
            )
        )
        model = MessagesModel(
            MessagesEndpoint(
                api_url="http://localhost:8000",
                model="model",
                default_max_tokens=77,
            ),
            opener=opener,
        )

        model.sample(ModelContext((Message(role="user", text="hello"),)))

        payload = _payload(opener)
        self.assertEqual(payload["max_tokens"], 77)
        self.assertNotIn("system", payload)
        self.assertNotIn("tools", payload)
        request, _ = opener.calls[0]
        self.assertIsNone(request.get_header("X-api-key"))

    def test_rejects_unsupported_or_unsafe_context(self):
        model = MessagesModel(
            MessagesEndpoint(api_url="http://localhost:8000", model="model"),
            opener=_Opener(_FakeResponse({})),
        )
        cases = [
            (
                ModelContext(
                    (
                        Message(role="user", text="hello"),
                        Message(role="system", text="late"),
                    )
                ),
                None,
                "appears after",
            ),
            (
                ModelContext(
                    (
                        Message(role="user", text="hello"),
                        ToolCall(
                            name="tool",
                            call_id="call",
                            arguments_json="[]",
                        ),
                        ToolResult(call_id="call", output="done"),
                    )
                ),
                None,
                "decode to an object",
            ),
            (
                ModelContext((Message(role="user", text="hello"),)),
                SamplingOptions(seed=1),
                "seed",
            ),
        ]
        for context, options, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ModelConfigurationError, message):
                    model.sample(context, options=options)

    def test_rejects_unknown_response_blocks(self):
        model = MessagesModel(
            MessagesEndpoint(api_url="http://localhost:8000", model="model"),
            opener=_Opener(
                _FakeResponse(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "redacted_thinking", "data": "x"}],
                    }
                )
            ),
        )
        with self.assertRaisesRegex(ModelResponseError, "redacted_thinking"):
            model.sample(ModelContext((Message(role="user", text="hello"),)))

    def test_context_window_http_error_is_typed(self):
        error = urllib.error.HTTPError(
            url="http://localhost:8000/v1/messages",
            code=400,
            msg="bad request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"context window exceeded"}}'),
        )
        model = MessagesModel(
            MessagesEndpoint(api_url="http://localhost:8000", model="model"),
            opener=_Opener(error),
        )

        with self.assertRaises(ModelContextWindowError):
            model.sample(ModelContext((Message(role="user", text="hello"),)))


class ReasoningSignatureTests(unittest.TestCase):
    def test_signature_is_validated_and_redacted(self):
        reasoning = Reasoning(
            text="thought",
            content_signature="signature-value",
        )
        self.assertNotIn("signature-value", repr(reasoning))
        with self.assertRaisesRegex(ValueError, "content_signature"):
            Reasoning(text="thought", content_signature="")


class MessagesDemoTests(unittest.TestCase):
    def test_builds_anthropic_messages_model(self):
        args = _build_parser().parse_args(
            [
                "--model-api",
                "messages",
                "--model",
                "claude-sonnet-5",
            ]
        )

        with mock.patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "secret-key"},
        ):
            model = _build_model(args)

        self.assertIsInstance(model, MessagesModel)
        self.assertEqual(model.endpoint.api_url, ANTHROPIC_MESSAGES_API_URL)
        self.assertEqual(
            model.endpoint.model,
            "claude-sonnet-5",
        )
        self.assertNotIn("secret-key", repr(model.endpoint))


if __name__ == "__main__":
    unittest.main()
