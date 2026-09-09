from __future__ import annotations

import inspect
import io
import tempfile
import threading
import unittest
import urllib.error
from unittest import mock

from pythia.interaction import ChatCompletionsEndpoint
from pythia.interaction import ChatCompletionsModel
from pythia.interaction import CodexAuth
from pythia.interaction import CodexResponsesModel
from pythia.interaction import CommandRuntime
from pythia.interaction import DEFAULT_LOGIN_TIMEOUT_SECONDS
from pythia.interaction import DEFAULT_REQUEST_TIMEOUT_SECONDS
from pythia.interaction import Message
from pythia.interaction import MessagesEndpoint
from pythia.interaction import MessagesModel
from pythia.interaction import ModelConfigurationError
from pythia.interaction import ModelContext
from pythia.interaction import ModelTimeoutError
from pythia.interaction import StreamingResponsesEndpoint
from pythia.interaction import Tool
from pythia.interaction import ToolCall
from pythia.interaction import ToolSpec
from pythia.interaction import cli
from pythia.interaction import codex_login
from pythia.interaction import codex_quota
from pythia.interaction import demo
from pythia.interaction import user_tools
from pythia.interaction._account_http import AccountServiceError
from pythia.interaction.model_config import build_model


def _models(opener, **options):
    return (
        ("chat", ChatCompletionsModel(
            ChatCompletionsEndpoint(api_url="http://localhost", **options),
            opener=opener,
        )),
        ("messages", MessagesModel(
            MessagesEndpoint(api_url="http://localhost", model="model", **options),
            opener=opener,
        )),
        ("responses endpoint", CodexResponsesModel(
            StreamingResponsesEndpoint(
                api_url="http://localhost", model="model", bearer_token="FAKE", **options,
            ),
            opener=opener,
        )),
        ("responses convenience", CodexResponsesModel(
            model="model", auth=CodexAuth("FAKE"), opener=opener, **options,
        )),
    )


class _ReadTimeoutResponse:
    status = 200
    headers = {}

    def __init__(self):
        self.closed = False

    def read(self):
        raise TimeoutError("offline read timeout")

    def __iter__(self):
        raise TimeoutError("offline stream timeout")

    def close(self):
        self.closed = True


class InteractionTimeoutTests(unittest.TestCase):
    def test_model_defaults_and_overrides_reach_http_transport(self):
        context = ModelContext((Message("user", "Hello."),))
        for options, expected in (
            ({}, DEFAULT_REQUEST_TIMEOUT_SECONDS),
            ({"request_timeout_seconds": 17.5}, 17.5),
        ):
            for error in (TimeoutError("offline"), urllib.error.URLError(TimeoutError("offline"))):
                opener = mock.Mock(side_effect=error)
                for name, model in _models(opener, **options):
                    with self.subTest(model=name, options=options, error=type(error)):
                        opener.reset_mock()
                        self.assertEqual(model.endpoint.request_timeout_seconds, expected)
                        with self.assertRaises(ModelTimeoutError):
                            model.sample(context)
                        opener.assert_called_once()
                        self.assertEqual(opener.call_args.kwargs["timeout"], expected)

    def test_body_and_stream_timeouts_remain_typed_and_close_responses(self):
        opener = mock.Mock()
        for name, model in _models(opener):
            with self.subTest(model=name):
                response = _ReadTimeoutResponse()
                opener.return_value = response
                with self.assertRaises(ModelTimeoutError):
                    model.sample(ModelContext((Message("user", "Hello."),)))
                self.assertTrue(response.closed)

    def test_responses_preserves_none_sentinel_and_explicit_endpoint(self):
        self.assertIsNone(
            inspect.signature(CodexResponsesModel).parameters["request_timeout_seconds"].default
        )
        endpoint = StreamingResponsesEndpoint(
            api_url="http://localhost", model="model", bearer_token="FAKE",
            request_timeout_seconds=17.5,
        )
        self.assertIs(CodexResponsesModel(endpoint).endpoint, endpoint)
        self.assertIs(
            CodexResponsesModel(endpoint, request_timeout_seconds=None).endpoint, endpoint,
        )
        with self.assertRaisesRegex(ModelConfigurationError, "endpoint cannot be combined"):
            CodexResponsesModel(endpoint, request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS)

    def test_endpoint_timeout_validation_is_unchanged(self):
        for endpoint_type, fields in (
            (ChatCompletionsEndpoint, {}),
            (MessagesEndpoint, {"model": "model"}),
            (StreamingResponsesEndpoint, {"model": "model", "bearer_token": "FAKE"}),
        ):
            for value in (None, True, 0, -1, float("inf"), float("nan"), "300"):
                with self.subTest(endpoint=endpoint_type, value=value):
                    with self.assertRaises(ModelConfigurationError):
                        endpoint_type(api_url="http://localhost", request_timeout_seconds=value, **fields)

    def test_cli_and_demo_share_defaults_and_preserve_explicit_overrides(self):
        for frontend in (cli, demo):
            for api in ("chat-completions", "messages", "codex", "codex-responses"):
                for flags, expected in (
                    ([], DEFAULT_REQUEST_TIMEOUT_SECONDS),
                    (["--request-timeout-seconds", "9.5"], 9.5),
                ):
                    with self.subTest(frontend=frontend.__name__, api=api, flags=flags):
                        args = frontend._build_parser().parse_args([
                            "--model-api", api, "--model", "model", *flags,
                        ])
                        with mock.patch(
                            "pythia.interaction.responses._load_default_model_auth",
                            return_value=CodexAuth("FAKE"),
                        ):
                            model = build_model(args)
                        self.assertEqual(args.request_timeout_seconds, expected)
                        self.assertEqual(model.endpoint.request_timeout_seconds, expected)

    def test_help_displays_the_shared_default_and_io_semantics(self):
        for frontend in (cli, demo):
            with self.subTest(frontend=frontend.__name__):
                help_text = " ".join(frontend._build_parser().format_help().split())
                self.assertIn(f"default: {DEFAULT_REQUEST_TIMEOUT_SECONDS} seconds", help_text)
                self.assertIn("not an overall deadline", help_text)

    def test_direct_account_service_defaults_use_shared_constants(self):
        login_options = inspect.signature(codex_login.login).parameters
        self.assertEqual(login_options["timeout_seconds"].default, DEFAULT_LOGIN_TIMEOUT_SECONDS)
        self.assertEqual(
            login_options["request_timeout_seconds"].default, DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        for options, expected in (
            ({}, DEFAULT_REQUEST_TIMEOUT_SECONDS),
            ({"timeout_seconds": 17.5}, 17.5),
        ):
            with self.subTest(options=options):
                response = io.BytesIO(b"{}")
                opener = mock.Mock(return_value=response)
                codex_quota.query_quota(CodexAuth("FAKE"), opener=opener, **options)
                self.assertEqual(opener.call_args.kwargs["timeout"], expected)
                self.assertTrue(response.closed)

    def test_user_tools_share_http_timeout_but_keep_the_separate_login_budget(self):
        for flags, expected in (
            ([], DEFAULT_REQUEST_TIMEOUT_SECONDS),
            (["--request-timeout-seconds", "9.5"], 9.5),
        ):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                args = cli._build_parser().parse_args([
                    "--model-api", "codex", "--model", "model",
                    "--codex-home", directory, *flags,
                ])
                environment = user_tools.create_user_environment(
                    args, notify=mock.Mock(), cancel=threading.Event(),
                )
                with mock.patch.object(user_tools, "login") as login:
                    result = environment.execute_tool_calls((ToolCall("login", "login-1", "{}"),))
                self.assertTrue(result.items[0].success)
                login.assert_called_once()
                self.assertEqual(login.call_args.kwargs["timeout_seconds"], DEFAULT_LOGIN_TIMEOUT_SECONDS)
                self.assertEqual(login.call_args.kwargs["request_timeout_seconds"], expected)
                with mock.patch.object(user_tools, "load_codex_auth", return_value=CodexAuth("FAKE")):
                    with mock.patch.object(user_tools, "query_quota", return_value="snapshot") as quota:
                        result = environment.execute_tool_calls((ToolCall("quota", "quota-1", "{}"),))
                self.assertTrue(result.items[0].success)
                quota.assert_called_once_with(CodexAuth("FAKE"), timeout_seconds=expected)

    def test_account_timeout_errors_still_withhold_provider_details(self):
        opener = mock.Mock(side_effect=TimeoutError("FAKE_SECRET"))
        with self.assertRaises(AccountServiceError) as raised:
            codex_quota.query_quota(CodexAuth("FAKE_SECRET"), opener=opener)
        self.assertNotIn("FAKE_SECRET", str(raised.exception))

    def test_command_and_ui_timings_remain_independent(self):
        defaults = inspect.signature(CommandRuntime).parameters
        self.assertEqual(defaults["default_exec_yield_time_ms"].default, 10_000)
        self.assertEqual(defaults["default_write_yield_time_ms"].default, 250)
        self.assertEqual(defaults["max_yield_time_ms"].default, 60_000)
        self.assertEqual(cli.FRAME_INTERVAL, 1 / 128)
        self.assertIsNone(Tool(ToolSpec("tool", "A tool.", {}), mock.Mock()).timeout_seconds)


if __name__ == "__main__":
    unittest.main()
