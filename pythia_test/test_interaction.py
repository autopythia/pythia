from __future__ import annotations

import io
import json
import urllib.error
import unittest

from pythia.interaction import ChatCompletionsEndpoint
from pythia.interaction import ChatCompletionsModel
from pythia.interaction import CompactionError
from pythia.interaction import ContextCompaction
from pythia.interaction import ContextValidationError
from pythia.interaction import DEFAULT_SUMMARY_PREFIX
from pythia.interaction import Environment
from pythia.interaction import EnvironmentError
from pythia.interaction import EnvironmentResult
from pythia.interaction import Message
from pythia.interaction import ModelConfigurationError
from pythia.interaction import ModelContext
from pythia.interaction import ModelContextWindowError
from pythia.interaction import ModelSample
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import PromptSummarizingCompactor
from pythia.interaction import Reasoning
from pythia.interaction import SamplingOptions
from pythia.interaction import TokenUsage
from pythia.interaction import Tool
from pythia.interaction import ToolCall
from pythia.interaction import ToolOutcome
from pythia.interaction import ToolResult
from pythia.interaction import ToolSpec
from pythia.interaction import TurnMetadata
from pythia.interaction import UserInteraction
from pythia.interaction import UserInteractionBoundary


class _FakeHTTPResponse:
    def __init__(self, payload, *, status=200):
        self.status = status
        self._payload = json.dumps(payload).encode("utf-8")
        self.closed = False

    def read(self):
        return self._payload

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


class ModelContextTests(unittest.TestCase):
    def test_context_is_append_only_and_projects_compaction(self):
        original = [
            Message(role="user", text="old request"),
            Message(role="assistant", text="old answer"),
        ]
        context = ModelContext(original)
        checkpoint = ContextCompaction(
            replacement_items=(
                Message(role="user", text="summary"),
            )
        )

        context.append(checkpoint)
        context.append(Message(role="user", text="new request"))

        self.assertEqual(
            context.items,
            (
                *original,
                checkpoint,
                Message(role="user", text="new request"),
            ),
        )
        self.assertEqual(
            context.model_items(),
            (
                Message(role="user", text="summary"),
                Message(role="user", text="new request"),
            ),
        )

    def test_nested_compaction_is_rejected_without_mutation(self):
        context = ModelContext([Message(role="user", text="hello")])
        nested = ContextCompaction(
            replacement_items=(
                ContextCompaction(
                    replacement_items=(Message(role="user", text="summary"),)
                ),
            )
        )

        with self.assertRaisesRegex(
            ContextValidationError,
            "must not contain another ContextCompaction",
        ):
            context.append(nested)

        self.assertEqual(
            context.items,
            (Message(role="user", text="hello"),),
        )

    def test_new_interaction_is_rejected_before_tool_results(self):
        call = ToolCall(
            name="lookup",
            call_id="call-1",
            arguments_json="{}",
        )
        context = ModelContext(
            [
                call,
                ModelSampleBoundary(),
            ]
        )

        with self.assertRaisesRegex(
            ContextValidationError,
            "before unresolved tool results",
        ):
            context.append(Message(role="user", text="continue"))

        self.assertEqual(
            context.items,
            (
                call,
                ModelSampleBoundary(),
            ),
        )

    def test_compaction_projection_preserves_interaction_boundaries(self):
        user_boundary = UserInteractionBoundary()
        sample_boundary = ModelSampleBoundary()
        checkpoint = ContextCompaction(
            replacement_items=(
                Message(role="user", text="retained request"),
                user_boundary,
                Message(role="assistant", text="retained answer"),
                sample_boundary,
            )
        )
        context = ModelContext(
            [
                Message(role="user", text="old request"),
                checkpoint,
                Message(role="user", text="new request"),
            ]
        )

        self.assertEqual(
            context.model_items(),
            (
                Message(role="user", text="retained request"),
                user_boundary,
                Message(role="assistant", text="retained answer"),
                sample_boundary,
                Message(role="user", text="new request"),
            ),
        )

    def test_pending_tool_calls_are_derived_from_effective_context(self):
        context = ModelContext(
            [
                ToolCall(
                    name="lookup",
                    call_id="call-1",
                    arguments_json="{}",
                )
            ]
        )

        self.assertEqual(
            tuple(call.call_id for call in context.pending_tool_calls()),
            ("call-1",),
        )
        with self.assertRaisesRegex(ContextValidationError, "unresolved"):
            context.assert_model_ready()

        context.append(
            ToolResult(call_id="call-1", output="done")
        )
        context.assert_model_ready()

    def test_turn_metadata_is_a_transparent_control_item(self):
        call = ToolCall(
            name="lookup",
            call_id="call-1",
            arguments_json="{}",
        )
        metadata = TurnMetadata(
            usage=TokenUsage(
                input_tokens=20,
                output_tokens=5,
                total_tokens=25,
                cached_input_tokens=4,
            )
        )
        context = ModelContext(
            [
                Message(role="user", text="lookup"),
                UserInteractionBoundary(),
                call,
                metadata,
                ModelSampleBoundary(),
            ]
        )

        self.assertIn(call, context.pending_tool_calls())
        self.assertEqual(
            context.model_items(),
            (
                Message(role="user", text="lookup"),
                UserInteractionBoundary(),
                call,
                metadata,
                ModelSampleBoundary(),
            ),
        )

        context.append(ToolResult(call_id="call-1", output="done"))
        context.assert_model_ready()

    def test_unknown_tool_result_is_rejected_atomically(self):
        context = ModelContext([Message(role="user", text="hello")])

        with self.assertRaisesRegex(ContextValidationError, "does not match"):
            context.extend(
                [
                    Message(role="assistant", text="answer"),
                    ToolResult(call_id="missing", output="bad"),
                ]
            )

        self.assertEqual(
            context.items,
            (Message(role="user", text="hello"),),
        )

    def test_copy_branches_the_context(self):
        context = ModelContext([Message(role="user", text="root")])
        branch = context.copy()
        branch.append(Message(role="assistant", text="branch"))

        self.assertEqual(len(context), 1)
        self.assertEqual(len(branch), 2)


class UserInteractionTests(unittest.TestCase):
    def test_context_items_adds_non_emitting_boundary(self):
        interaction = UserInteraction(
            items=[Message(role="user", text="hello")],
        )

        self.assertEqual(
            interaction.items,
            (Message(role="user", text="hello"),),
        )
        self.assertEqual(
            interaction.context_items(),
            (
                Message(role="user", text="hello"),
                UserInteractionBoundary(),
            ),
        )

    def test_rejects_empty_or_non_user_items(self):
        cases = (
            (),
            (Message(role="assistant", text="answer"),),
            (Reasoning(text="thought"),),
        )

        for items in cases:
            with self.subTest(items=items):
                with self.assertRaises(ValueError):
                    UserInteraction(items=items)


class EndpointTests(unittest.TestCase):
    def test_endpoint_builds_url_after_path_prefix(self):
        endpoint = ChatCompletionsEndpoint(
            api_url=" HTTPS://api.example.test:8443/proxy/root/ ",
            model=" example-model ",
        )

        self.assertEqual(
            endpoint.api_url,
            "https://api.example.test:8443/proxy/root",
        )
        self.assertEqual(
            endpoint.url,
            "https://api.example.test:8443/proxy/root/v1/chat/completions",
        )
        self.assertEqual(endpoint.model, "example-model")

    def test_endpoint_normalizes_absent_model(self):
        endpoint = ChatCompletionsEndpoint(
            api_url="http://localhost:9000/",
            model=" ",
        )

        self.assertIsNone(endpoint.model)
        self.assertEqual(endpoint.api_url, "http://localhost:9000")
        self.assertEqual(
            endpoint.url,
            "http://localhost:9000/v1/chat/completions",
        )

    def test_endpoint_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(TypeError, "api_url"):
            ChatCompletionsEndpoint(api_url=object())

        cases = [
            {"api_url": ""},
            {"api_url": "localhost:8000"},
            {"api_url": "ftp://localhost"},
            {"api_url": "http:///missing-host"},
            {"api_url": "http://user:password@localhost"},
            {"api_url": "http://localhost/prefix?query=value"},
            {"api_url": "http://localhost/prefix#fragment"},
            {"api_url": "http://localhost:not-a-port"},
            {"api_url": "http://localhost:0"},
            {"api_url": "http://localhost/v1/chat/completions"},
            {
                "api_url": "http://localhost",
                "request_timeout_seconds": 0,
            },
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ModelConfigurationError):
                    ChatCompletionsEndpoint(**kwargs)

    def test_endpoint_validates_and_redacts_api_key(self):
        endpoint = ChatCompletionsEndpoint(
            api_url="http://localhost:8000",
            api_key=" example-secret ",
        )

        self.assertEqual(endpoint.api_key, "example-secret")
        self.assertNotIn("example-secret", repr(endpoint))

        with self.assertRaisesRegex(TypeError, "api_key"):
            ChatCompletionsEndpoint(
                api_url="http://localhost:8000",
                api_key=object(),
            )
        for api_key in ("", " ", "two words"):
            with self.subTest(api_key=api_key):
                with self.assertRaisesRegex(
                    ModelConfigurationError,
                    "api_key",
                ):
                    ChatCompletionsEndpoint(
                        api_url="http://localhost:8000",
                        api_key=api_key,
                    )


class ChatCompletionsModelTests(unittest.TestCase):
    def test_sample_context_items_adds_non_emitting_boundary(self):
        sample = ModelSample(
            items=(Message(role="assistant", text="answer"),),
        )
        usage = TokenUsage(
            input_tokens=20,
            output_tokens=5,
            total_tokens=25,
            cached_input_tokens=4,
        )
        sample_with_usage = ModelSample(
            items=(Message(role="assistant", text="answer"),),
            usage=usage,
        )

        self.assertEqual(
            sample.context_items(),
            (
                Message(role="assistant", text="answer"),
                TurnMetadata(usage=TokenUsage()),
                ModelSampleBoundary(),
            ),
        )
        self.assertEqual(
            sample_with_usage.context_items(),
            (
                Message(role="assistant", text="answer"),
                TurnMetadata(usage=usage),
                ModelSampleBoundary(),
            ),
        )
        self.assertEqual(
            sample.items,
            (Message(role="assistant", text="answer"),),
        )

    def test_sample_encodes_request_and_decodes_tool_call(self):
        response = _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I will check.",
                            "reasoning_content": "Need the tool.",
                            "tool_calls": [
                                {
                                    "id": "call-weather",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            }
        )
        opener = _ScriptedOpener(response)
        model = ChatCompletionsModel(
            ChatCompletionsEndpoint(
                api_url="http://localhost:8000",
                model="demo",
            ),
            opener=opener,
        )
        context = ModelContext(
            [
                Message(role="system", text="Be concise."),
                Message(role="user", text="Weather in Paris?"),
            ]
        )
        before = context.items
        spec = ToolSpec(
            name="lookup_weather",
            description="Look up weather.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        )

        sample = model.sample(
            context,
            tools=(spec,),
            options=SamplingOptions(
                max_tokens=100,
                temperature=0.25,
                stop=("END",),
                seed=7,
            ),
        )

        self.assertEqual(context.items, before)
        self.assertEqual(
            sample.items,
            (
                Reasoning(text="Need the tool."),
                Message(role="assistant", text="I will check."),
                ToolCall(
                    name="lookup_weather",
                    call_id="call-weather",
                    arguments_json='{"city":"Paris"}',
                ),
            ),
        )
        self.assertEqual(sample.stop_reason, "tool_use")
        self.assertEqual(sample.usage.total_tokens, 25)
        self.assertEqual(sample.usage.cached_input_tokens, 4)
        self.assertEqual(
            sample.context_items()[-2:],
            (
                TurnMetadata(usage=sample.usage),
                ModelSampleBoundary(),
            ),
        )
        self.assertEqual(
            sample.display_items()[-1].text,
            "[turn] usage input=20 output=5 total=25 cached=4",
        )
        self.assertTrue(response.closed)

        request, timeout = opener.calls[0]
        payload = _request_payload(opener)
        self.assertEqual(timeout, 60.0)
        self.assertEqual(
            request.full_url,
            "http://localhost:8000/v1/chat/completions",
        )
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(payload["model"], "demo")
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertEqual(payload["max_tokens"], 100)
        self.assertEqual(payload["temperature"], 0.25)
        self.assertEqual(payload["stop"], ["END"])
        self.assertEqual(payload["seed"], 7)

    def test_api_key_uses_bearer_authorization(self):
        response = _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "authenticated",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        opener = _ScriptedOpener(response)
        endpoint = ChatCompletionsEndpoint(
            api_url="https://api.example.test/proxy",
            api_key=" example-secret ",
        )
        model = ChatCompletionsModel(
            endpoint,
            opener=opener,
        )

        sample = model.sample(
            ModelContext([Message(role="user", text="hello")])
        )

        request, _ = opener.calls[0]
        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer example-secret",
        )
        self.assertEqual(
            request.full_url,
            "https://api.example.test/proxy/v1/chat/completions",
        )
        self.assertEqual(sample.last_assistant_text, "authenticated")

    def test_caller_drives_tool_follow_up(self):
        first_response = _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "echo",
                                        "arguments": '{"value":"hello"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        final_response = _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The tool said hello.",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        opener = _ScriptedOpener(first_response, final_response)
        model = ChatCompletionsModel(
            ChatCompletionsEndpoint(api_url="http://localhost:8000"),
            opener=opener,
        )

        def echo(arguments, *, timeout_seconds=None):
            self.assertEqual(timeout_seconds, 3.0)
            return ToolOutcome(output=str(arguments["value"]))

        environment = Environment(
            tools=(
                Tool(
                    spec=ToolSpec(
                        name="echo",
                        description="Echo a value.",
                        parameters={"type": "object"},
                    ),
                    handler=echo,
                    timeout_seconds=3.0,
                ),
            )
        )
        context = ModelContext()
        user_interaction = UserInteraction(
            items=(Message(role="user", text="Use echo."),),
        )
        context.extend(user_interaction.context_items())

        first_sample = model.sample(
            context,
            tools=environment.tool_specs,
        )
        context.extend(first_sample.context_items())
        environment_result = environment.execute_tool_calls(
            first_sample.tool_calls
        )
        context.extend(environment_result.context_items())
        final_sample = model.sample(
            context,
            tools=environment.tool_specs,
        )
        context.extend(final_sample.context_items())

        self.assertEqual(
            final_sample.last_assistant_text,
            "The tool said hello.",
        )
        second_payload = _request_payload(opener, 1)
        self.assertEqual(
            second_payload["messages"],
            [
                {"role": "user", "content": "Use echo."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"hello"}',
                            },
                        }
                    ],
                    "content": None,
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "hello",
                },
            ],
        )

    def test_sample_boundaries_separate_adjacent_assistant_messages(self):
        opener = _ScriptedOpener(
            _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "new answer",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
        )
        model = ChatCompletionsModel(
            ChatCompletionsEndpoint(api_url="http://localhost:8000"),
            opener=opener,
        )
        context = ModelContext(
            [
                Message(role="user", text="question"),
                UserInteractionBoundary(),
                Reasoning(text="first reasoning"),
                Message(role="assistant", text="first answer"),
                ModelSampleBoundary(),
                Reasoning(text="second reasoning"),
                Message(role="assistant", text="second answer"),
                ModelSampleBoundary(),
                Message(role="user", text="continue"),
            ]
        )

        model.sample(context)

        self.assertEqual(
            _request_payload(opener)["messages"],
            [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "reasoning_content": "first reasoning",
                    "content": "first answer",
                },
                {
                    "role": "assistant",
                    "reasoning_content": "second reasoning",
                    "content": "second answer",
                },
                {"role": "user", "content": "continue"},
            ],
        )

    def test_turn_metadata_is_not_sent_to_chat_completions(self):
        opener = _ScriptedOpener(
            _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "answer",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
        )
        model = ChatCompletionsModel(
            ChatCompletionsEndpoint(api_url="http://localhost:8000"),
            opener=opener,
        )
        context = ModelContext(
            [
                Message(role="user", text="question"),
                UserInteractionBoundary(),
                Message(role="assistant", text="prior answer"),
                TurnMetadata(
                    usage=TokenUsage(
                        input_tokens=20,
                        output_tokens=5,
                        total_tokens=25,
                        cached_input_tokens=4,
                    )
                ),
                ModelSampleBoundary(),
                Message(role="user", text="continue"),
            ]
        )

        model.sample(context)

        self.assertEqual(
            _request_payload(opener)["messages"],
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "prior answer"},
                {"role": "user", "content": "continue"},
            ],
        )

    def test_context_without_boundaries_keeps_adjacency_collation(self):
        opener = _ScriptedOpener(
            _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "new answer",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
        )
        model = ChatCompletionsModel(
            ChatCompletionsEndpoint(api_url="http://localhost:8000"),
            opener=opener,
        )
        context = ModelContext(
            [
                Message(role="user", text="question"),
                Message(role="assistant", text="first"),
                Message(role="assistant", text="second"),
                Message(role="user", text="continue"),
            ]
        )

        model.sample(context)

        self.assertEqual(
            _request_payload(opener)["messages"],
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "firstsecond"},
                {"role": "user", "content": "continue"},
            ],
        )

    def test_optional_model_is_omitted(self):
        opener = _ScriptedOpener(
            _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "ok",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
        )
        model = ChatCompletionsModel(
            ChatCompletionsEndpoint(api_url="http://localhost:8000"),
            opener=opener,
        )

        model.sample(ModelContext([Message(role="user", text="hello")]))

        self.assertNotIn("model", _request_payload(opener))

    def test_context_window_http_error_is_typed(self):
        error = urllib.error.HTTPError(
            url="http://localhost:8000/v1/chat/completions",
            code=400,
            msg="bad request",
            hdrs=None,
            fp=io.BytesIO(
                b'{"error":{"message":"maximum context length exceeded"}}'
            ),
        )
        model = ChatCompletionsModel(
            ChatCompletionsEndpoint(api_url="http://localhost:8000"),
            opener=_ScriptedOpener(error),
        )

        with self.assertRaises(ModelContextWindowError):
            model.sample(ModelContext([Message(role="user", text="hello")]))


class EnvironmentTests(unittest.TestCase):
    def test_environment_result_context_items_are_directly_appendable(self):
        item = ToolResult(call_id="call-1", output="done")
        result = EnvironmentResult(items=(item,))

        self.assertEqual(result.context_items(), (item,))

    def test_environment_executes_sequentially_and_preserves_order(self):
        seen = []

        def handler(arguments, *, timeout_seconds=None):
            seen.append((arguments["value"], timeout_seconds))
            return ToolOutcome(output=f"out:{arguments['value']}")

        environment = Environment(
            tools=(
                Tool(
                    spec=ToolSpec(
                        name="ordered",
                        description="Record order.",
                        parameters={"type": "object"},
                    ),
                    handler=handler,
                    timeout_seconds=2.0,
                ),
            )
        )
        calls = (
            ToolCall("ordered", "1", '{"value":"a"}'),
            ToolCall("ordered", "2", '{"value":"b"}'),
        )

        result = environment.execute_tool_calls(calls)

        self.assertEqual(seen, [("a", 2.0), ("b", 2.0)])
        self.assertEqual(
            result.items,
            (
                ToolResult("1", "out:a"),
                ToolResult("2", "out:b"),
            ),
        )

    def test_environment_converts_recoverable_failures_to_results(self):
        def failing(arguments, *, timeout_seconds=None):
            del arguments, timeout_seconds
            raise RuntimeError("boom")

        environment = Environment(
            tools=(
                Tool(
                    spec=ToolSpec(
                        name="failing",
                        description="Fail.",
                        parameters={"type": "object"},
                    ),
                    handler=failing,
                ),
            )
        )

        result = environment.execute_tool_calls(
            (
                ToolCall("missing", "1", "{}"),
                ToolCall("failing", "2", "{}"),
                ToolCall("failing", "3", "[]"),
                ToolCall("failing", "4", "{"),
            )
        )

        self.assertEqual(
            tuple(item.call_id for item in result.items),
            ("1", "2", "3", "4"),
        )
        self.assertTrue(all(not item.success for item in result.items))

    def test_environment_rejects_duplicate_call_ids_before_execution(self):
        invoked = []

        def handler(arguments, *, timeout_seconds=None):
            invoked.append(arguments)
            return ToolOutcome(output="ok")

        environment = Environment(
            tools=(
                Tool(
                    spec=ToolSpec(
                        name="x",
                        description="x",
                        parameters={"type": "object"},
                    ),
                    handler=handler,
                ),
            )
        )

        with self.assertRaisesRegex(EnvironmentError, "duplicate"):
            environment.execute_tool_calls(
                (
                    ToolCall("x", "same", "{}"),
                    ToolCall("x", "same", "{}"),
                )
            )

        self.assertEqual(invoked, [])


class _ScriptedModel:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def sample(self, context, *, tools=(), options=None):
        self.calls.append((context.copy(), tuple(tools), options))
        if not self.outcomes:
            raise AssertionError("unexpected model sample")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class CompactionTests(unittest.TestCase):
    def test_prompt_compactor_returns_append_only_checkpoint(self):
        usage = TokenUsage(input_tokens=80, output_tokens=20, total_tokens=100)
        model = _ScriptedModel(
            ModelSample(
                items=(Message(role="assistant", text="Condensed work."),),
                stop_reason="end_turn",
                usage=usage,
            )
        )
        context = ModelContext(
            [
                Message(role="system", text="Base instructions."),
                Message(role="user", text="First request."),
                Message(role="assistant", text="First answer."),
                Message(
                    role="user",
                    text=f"{DEFAULT_SUMMARY_PREFIX}\nOld summary.",
                ),
                Message(role="user", text="Latest request."),
                Message(role="assistant", text="Latest answer."),
            ]
        )
        before = context.items
        compactor = PromptSummarizingCompactor(model)

        result = compactor.compact(context)

        self.assertEqual(context.items, before)
        self.assertEqual(result.usage, usage)
        self.assertEqual(len(result.items), 1)
        checkpoint = result.items[0]
        self.assertIsInstance(checkpoint, ContextCompaction)
        assert isinstance(checkpoint, ContextCompaction)
        self.assertEqual(
            checkpoint.replacement_items,
            (
                Message(role="system", text="Base instructions."),
                Message(role="user", text="First request."),
                Message(role="user", text="Latest request."),
                Message(
                    role="user",
                    text=f"{DEFAULT_SUMMARY_PREFIX}\nCondensed work.",
                ),
            ),
        )
        temporary_context, tools, options = model.calls[0]
        self.assertEqual(tools, ())
        self.assertIsInstance(options, SamplingOptions)
        self.assertNotEqual(temporary_context.items, context.items)
        self.assertIn("CONTEXT CHECKPOINT COMPACTION", temporary_context[-1].text)

        context.extend(result.items)
        self.assertEqual(context.model_items(), checkpoint.replacement_items)
        self.assertEqual(context.items[: len(before)], before)

    def test_prompt_compactor_fits_only_temporary_request(self):
        model = _ScriptedModel(
            ModelContextWindowError("too large"),
            ModelSample(
                items=(Message(role="assistant", text="summary"),),
            ),
        )
        context = ModelContext(
            [
                Message(role="system", text="instructions"),
                Message(role="user", text="old"),
                Message(role="assistant", text="old answer"),
                Message(role="user", text="new"),
                Message(role="assistant", text="new answer"),
            ]
        )
        before = context.items
        compactor = PromptSummarizingCompactor(model)

        result = compactor.compact(context)

        self.assertEqual(context.items, before)
        self.assertEqual(len(model.calls), 2)
        self.assertLess(len(model.calls[1][0]), len(model.calls[0][0]))
        self.assertIsInstance(result.items[0], ContextCompaction)

    def test_prompt_compactor_rejects_tool_calls(self):
        model = _ScriptedModel(
            ModelSample(
                items=(
                    ToolCall(
                        name="unexpected",
                        call_id="call-1",
                        arguments_json="{}",
                    ),
                ),
            )
        )
        compactor = PromptSummarizingCompactor(model)

        with self.assertRaisesRegex(CompactionError, "must not contain tool calls"):
            compactor.compact(
                ModelContext([Message(role="user", text="hello")])
            )


if __name__ == "__main__":
    unittest.main()
