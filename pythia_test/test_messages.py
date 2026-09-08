from __future__ import annotations

import io
import json
import urllib.error
import unittest
from typing import Any
from unittest import mock

from pythia.interaction import ANTHROPIC_MESSAGES_API_URL
from pythia.interaction import Environment
from pythia.interaction import MESSAGES_COMPACTION_BETA
from pythia.interaction import Message
from pythia.interaction import MessagesEndpoint
from pythia.interaction import MessagesModel
from pythia.interaction import MessagesServerCompaction
from pythia.interaction import ModelConfigurationError
from pythia.interaction import ModelContext
from pythia.interaction import ModelContextWindowError
from pythia.interaction import ModelResponseError
from pythia.interaction import ModelSample
from pythia.interaction import OpaqueCompaction
from pythia.interaction import Reasoning
from pythia.interaction import SamplingOptions
from pythia.interaction import ToolCall
from pythia.interaction import ToolResult
from pythia.interaction import ToolSpec
from pythia.interaction import UserInteractionBoundary
from pythia.interaction.demo import _build_model
from pythia.interaction.demo import _build_parser
from pythia.interaction.demo import run
from pythia.interaction.experimental_tools import create_inject_user_message_tool


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


class _ScriptedOpener:
    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, request, *, timeout):
        self.calls.append((request, timeout))
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        return self.responses.pop(0)


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

    def test_server_compaction_configuration_is_validated(self):
        default = MessagesServerCompaction()
        self.assertEqual(
            default.request_edit(),
            {"type": "compact_20260112"},
        )
        configured = MessagesServerCompaction(
            trigger_input_tokens=150_000,
            pause_after_compaction=True,
            instructions=" summarize without tools ",
        )
        self.assertEqual(
            configured.request_edit(),
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 150_000},
                "pause_after_compaction": True,
                "instructions": "summarize without tools",
            },
        )
        with self.assertRaisesRegex(ModelConfigurationError, "50000"):
            MessagesServerCompaction(trigger_input_tokens=49_999)
        with self.assertRaisesRegex(ModelConfigurationError, "instructions"):
            MessagesServerCompaction(instructions=" ")
        with self.assertRaisesRegex(TypeError, "server_compaction"):
            MessagesEndpoint(
                api_url="http://localhost:8000",
                model="model",
                server_compaction=object(),
            )


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
                Message(role="system", content="system text"),
                Message(role="developer", content="developer text"),
                Message(role="user", content="question"),
                UserInteractionBoundary(),
                Reasoning(
                    content="prior thought",
                    encrypted_content="responses-only-data",
                    content_signature="prior-signature",
                ),
                Message(role="assistant", content="calling"),
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
                    content="Need the tool.",
                    content_signature="new-signature",
                ),
                Message(role="assistant", content="Checking."),
                ToolCall(
                    name="lookup",
                    call_id="call-new",
                    arguments_json='{"q":"new"}',
                ),
            ),
        )
        self.assertEqual(sample.stop_reason, "tool_use")
        self.assertEqual(sample.usage.input_tokens, 31)
        self.assertEqual(sample.usage.output_tokens, 5)
        self.assertEqual(sample.usage.total_tokens, 36)
        self.assertEqual(sample.usage.cached_input_tokens, 4)
        self.assertTrue(response.closed)

    def test_injected_message_is_user_text_after_tool_result_block(self):
        tool = create_inject_user_message_tool()
        call = ToolCall(tool.spec.name, "inject-1", "{}")
        context = ModelContext((Message("user", "Run the experiment."),))
        context.extend(ModelSample(items=(call,)).context_items())
        environment = Environment((tool,))
        context.extend(environment.execute_tool_calls((call,)).context_items())
        opener = _Opener(_FakeResponse({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "received: hello world"}],
            "stop_reason": "end_turn",
        }))
        model = MessagesModel(
            MessagesEndpoint(api_url="http://localhost:8000", model="model"),
            opener=opener,
        )
        model.sample(context, tools=environment.tool_specs)
        self.assertEqual(_payload(opener)["messages"][-1], {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call.call_id,
                    "content": "Synthetic user message queued.",
                    "is_error": False,
                },
                {"type": "text", "text": "hello world"},
            ],
        })

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

        model.sample(ModelContext((Message(role="user", content="hello"),)))

        payload = _payload(opener)
        self.assertEqual(payload["max_tokens"], 77)
        self.assertNotIn("system", payload)
        self.assertNotIn("tools", payload)
        self.assertNotIn("context_management", payload)
        request, _ = opener.calls[0]
        self.assertIsNone(request.get_header("X-api-key"))
        self.assertIsNone(request.get_header("Anthropic-beta"))

    def test_rejects_unsupported_or_unsafe_context(self):
        model = MessagesModel(
            MessagesEndpoint(api_url="http://localhost:8000", model="model"),
            opener=_Opener(_FakeResponse({})),
        )
        cases = [
            (
                ModelContext(
                    (
                        Message(role="user", content="hello"),
                        Message(role="system", content="late"),
                    )
                ),
                None,
                "appears after",
            ),
            (
                ModelContext(
                    (
                        Message(role="user", content="hello"),
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
                ModelContext((Message(role="user", content="hello"),)),
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
            model.sample(ModelContext((Message(role="user", content="hello"),)))

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
            model.sample(ModelContext((Message(role="user", content="hello"),)))

    def test_server_compaction_round_trips_and_projects_latest_block(self):
        first_response = _FakeResponse(
            {
                "type": "message",
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "compaction",
                        "content": "Summary of old work.",
                    },
                    {"type": "text", "text": "First answer."},
                ],
                "usage": {
                    "input_tokens": 23_000,
                    "output_tokens": 1_000,
                    "cache_read_input_tokens": 500,
                    "iterations": [
                        {
                            "type": "compaction",
                            "input_tokens": 180_000,
                            "output_tokens": 3_500,
                        },
                        {
                            "type": "message",
                            "input_tokens": 23_000,
                            "output_tokens": 1_000,
                            "cache_read_input_tokens": 500,
                        },
                    ],
                },
            }
        )
        second_response = _FakeResponse(
            {
                "type": "message",
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Second answer."}],
                "usage": {"input_tokens": 20, "output_tokens": 5},
            }
        )
        opener = _ScriptedOpener(first_response, second_response)
        model = MessagesModel(
            MessagesEndpoint(
                api_url="http://localhost:8000",
                model="claude-sonnet-5",
                server_compaction=MessagesServerCompaction(),
            ),
            opener=opener,
        )
        context = ModelContext(
            (
                Message(role="system", content="instructions"),
                Message(role="user", content="old question"),
            )
        )

        first = model.sample(context)

        first_request, _ = opener.calls[0]
        first_payload = json.loads(first_request.data.decode("utf-8"))
        self.assertEqual(
            first_request.get_header("Anthropic-beta"),
            MESSAGES_COMPACTION_BETA,
        )
        self.assertEqual(
            first_payload["context_management"],
            {"edits": [{"type": "compact_20260112"}]},
        )
        self.assertEqual(
            first.items,
            (
                OpaqueCompaction.from_messages("Summary of old work."),
                Message(role="assistant", content="First answer."),
            ),
        )
        self.assertEqual(first.usage.input_tokens, 203_500)
        self.assertEqual(first.usage.output_tokens, 4_500)
        self.assertEqual(first.usage.total_tokens, 208_000)
        self.assertEqual(first.usage.cached_input_tokens, 500)

        context.extend(first.context_items())
        context.extend(
            (
                Message(role="user", content="new question"),
                UserInteractionBoundary(),
            )
        )
        second = model.sample(context)

        self.assertEqual(second.last_assistant_text, "Second answer.")
        second_request, _ = opener.calls[1]
        second_payload = json.loads(second_request.data.decode("utf-8"))
        self.assertEqual(
            second_payload["messages"],
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "compaction",
                            "content": "Summary of old work.",
                        },
                        {"type": "text", "text": "First answer."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "new question"}
                    ],
                },
            ],
        )
        self.assertNotIn("old question", second_request.data.decode("utf-8"))
        self.assertEqual(
            second_payload["system"],
            [{"type": "text", "text": "instructions"}],
        )

    def test_rejects_null_compaction_and_responses_subtype(self):
        null_model = MessagesModel(
            MessagesEndpoint(api_url="http://localhost:8000", model="model"),
            opener=_Opener(
                _FakeResponse(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "compaction", "content": None}
                        ],
                    }
                )
            ),
        )
        with self.assertRaisesRegex(ModelResponseError, "content"):
            null_model.sample(
                ModelContext((Message(role="user", content="hello"),))
            )

        wrong_protocol_model = MessagesModel(
            MessagesEndpoint(api_url="http://localhost:8000", model="model"),
            opener=_Opener(_FakeResponse({})),
        )
        with self.assertRaisesRegex(
            ModelConfigurationError,
            "Responses opaque compaction",
        ):
            wrong_protocol_model.sample(
                ModelContext(
                    (
                        OpaqueCompaction.from_responses("encrypted"),
                        Message(role="user", content="hello"),
                    )
                )
            )

    def test_http_413_is_a_context_window_error(self):
        error = urllib.error.HTTPError(
            url="http://localhost:8000/v1/messages",
            code=413,
            msg="request too large",
            hdrs=None,
            fp=io.BytesIO(b'{"type":"error","error":{"type":"request_too_large"}}'),
        )
        model = MessagesModel(
            MessagesEndpoint(api_url="http://localhost:8000", model="model"),
            opener=_Opener(error),
        )
        with self.assertRaises(ModelContextWindowError):
            model.sample(ModelContext((Message(role="user", content="hello"),)))


class ReasoningSignatureTests(unittest.TestCase):
    def test_signature_is_validated_and_redacted(self):
        reasoning = Reasoning(
            content="thought",
            content_signature="signature-value",
        )
        self.assertNotIn("signature-value", repr(reasoning))
        with self.assertRaisesRegex(ValueError, "content_signature"):
            Reasoning(content="thought", content_signature="")


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

    def test_demo_builds_server_compaction_options(self):
        args = _build_parser().parse_args(
            [
                "--model-api",
                "messages",
                "--model",
                "claude-sonnet-5",
                "--messages-server-compaction",
                "--messages-compaction-trigger-tokens",
                "200000",
                "--messages-pause-after-compaction",
                "--messages-compaction-instructions",
                "Keep implementation state.",
            ]
        )

        model = _build_model(args)

        self.assertEqual(
            model.endpoint.server_compaction,
            MessagesServerCompaction(
                trigger_input_tokens=200_000,
                pause_after_compaction=True,
                instructions="Keep implementation state.",
            ),
        )

        missing_enable = _build_parser().parse_args(
            [
                "--model-api",
                "messages",
                "--model",
                "claude-sonnet-5",
                "--messages-compaction-trigger-tokens",
                "200000",
            ]
        )
        with self.assertRaisesRegex(
            ValueError,
            "--messages-server-compaction",
        ):
            _build_model(missing_enable)

    def test_demo_continues_after_paused_compaction(self):
        class Model:
            def __init__(self):
                self.calls = []

            def sample(self, context, *, tools=(), options=None):
                del tools, options
                self.calls.append(context.copy())
                if len(self.calls) == 1:
                    return ModelSample(
                        items=(
                            OpaqueCompaction.from_messages("summary"),
                        ),
                        stop_reason="compaction",
                    )
                return ModelSample(
                    items=(Message(role="assistant", content="done"),),
                    stop_reason="end_turn",
                )

        model = Model()
        with mock.patch("builtins.print"):
            result = run(
                model,
                Environment(),
                prompt="hello",
                max_samples=2,
            )

        self.assertEqual(result, "done")
        self.assertEqual(len(model.calls), 2)
        self.assertIn(
            OpaqueCompaction.from_messages("summary"),
            model.calls[1].items,
        )


if __name__ == "__main__":
    unittest.main()
