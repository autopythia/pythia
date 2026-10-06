from __future__ import annotations

from pythia_test.interaction_helpers import chat_endpoint
from pythia_test.interaction_helpers import messages_endpoint
from pythia_test.interaction_helpers import responses_endpoint
from pythia_test.interaction_helpers import codex_model

import inspect
import io
import json
from pathlib import Path
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
from pythia.interaction import InteractionConfig
from pythia.interaction import MAX_TIMEOUT_SECONDS
from pythia.interaction import Message
from pythia.interaction import MessagesEndpoint
from pythia.interaction import MessagesModel
from pythia.interaction import ModelAuthenticationError
from pythia.interaction import ModelConfigurationError
from pythia.interaction import InteractionContext
from pythia.interaction import ModelTimeoutError
from pythia.interaction import ResponsesOpaqueCompactor
from pythia.interaction import SampleParams
from pythia.interaction import StreamingResponsesEndpoint
from pythia.interaction import Tool
from pythia.interaction import ToolCall
from pythia.interaction import ToolSpec
from pythia.interaction import cli
from pythia.interaction import codex_login
from pythia.interaction import codex_quota
from pythia.interaction import demo
from pythia.interaction import parse_model_catalog
from pythia.interaction import responses
from pythia.interaction import user_tools
from pythia.interaction._account_http import AccountServiceError
from pythia.interaction.model_config import build_model
from pythia.interaction.model_config import prepare_namespace


# One catalog timeout per HTTP API; anonymous local endpoints need no credentials.
_TIMEOUT_CATALOG = "\n".join((
    "[catalog]",
    "version = 4",
    *(line for api, port in (("chat-completions", 8000), ("messages", 8001),
                             ("codex", 8002), ("responses", 8003)) for line in (
        f"[model.slow-{api}]",
        f"endpoint.api = {api}",
        f"endpoint.url = http://127.0.0.1:{port}/v1/model",
        "endpoint.model = served",
        "endpoint.auth = none",
        "limits.max_output_tokens = 100",
        "timeouts.request_seconds = 1800",
    )),
))


def _no_retry_sleep(_delay):
    return None


def _models(opener, **options):
    return (
        ("chat", ChatCompletionsModel(
            chat_endpoint(api_url="http://localhost", **options),
            opener=opener,
            retry_sleep=_no_retry_sleep,
        )),
        ("messages", MessagesModel(
            messages_endpoint(
                api_url="http://localhost",
                model="model",
                max_output_tokens=100,
                **options,
            ),
            opener=opener,
            retry_sleep=_no_retry_sleep,
        )),
        ("responses endpoint", codex_model(
            responses_endpoint(
                api_url="http://localhost", model="model", bearer_token="FAKE", **options,
            ),
            opener=opener,
            retry_sleep=_no_retry_sleep,
        )),
        ("responses convenience", codex_model(
            model="model", auth=CodexAuth("FAKE"), opener=opener,
            retry_sleep=_no_retry_sleep, **options,
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


def _unauthorized(request, *, timeout):
    del timeout
    raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"{}"))


class InteractionTimeoutTests(unittest.TestCase):
    def test_model_defaults_and_overrides_reach_http_transport(self):
        context = InteractionContext((Message("user", "Hello."),))
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
                        self.assertEqual(opener.call_count, 3)
                        self.assertTrue(all(
                            call.kwargs["timeout"] == expected
                            for call in opener.call_args_list
                        ))

    def test_body_and_stream_timeouts_remain_typed_and_close_responses(self):
        opener = mock.Mock()
        for name, model in _models(opener):
            with self.subTest(model=name):
                response = _ReadTimeoutResponse()
                opener.return_value = response
                with self.assertRaises(ModelTimeoutError):
                    model.sample(InteractionContext((Message("user", "Hello."),)))
                self.assertTrue(response.closed)

    def test_per_call_timeout_reaches_every_attempt_and_names_its_budget(self):
        context = InteractionContext((Message("user", "Hello."),))
        params = SampleParams(request_timeout_seconds=17.5)
        for failure, side_effect in (
            ("open", TimeoutError("offline")),
            ("read", lambda request, **kwargs: _ReadTimeoutResponse()),
        ):
            opener = mock.Mock(side_effect=side_effect)
            for name, model in _models(opener):
                with self.subTest(model=name, failure=failure):
                    opener.reset_mock()
                    with self.assertRaises(ModelTimeoutError) as raised:
                        model.sample(context, sample_params=params)
                    self.assertEqual(opener.call_count, 3)
                    self.assertTrue(all(
                        call.kwargs["timeout"] == 17.5 for call in opener.call_args_list
                    ))
                    self.assertIn("(request_timeout_seconds=17.5)", str(raised.exception))
                    self.assertIn("(request_timeout_seconds=17.5)", raised.exception.failure.message)
                    # Calls without a per-call value keep the endpoint's own timeout.
                    self.assertEqual(
                        model.endpoint.request_timeout_seconds, DEFAULT_REQUEST_TIMEOUT_SECONDS,
                    )

    def test_remote_compaction_uses_the_turn_request_timeout(self):
        opener = mock.Mock(side_effect=TimeoutError("offline"))
        model = codex_model(
            model="codex-gpt-6-astra", auth=CodexAuth("FAKE"), opener=opener,
            retry_sleep=_no_retry_sleep,
        )
        with self.assertRaises(ModelTimeoutError) as raised:
            ResponsesOpaqueCompactor(model).compact(
                InteractionContext((Message("user", "Hello."),)),
                sample_params=SampleParams(max_output_tokens=9, request_timeout_seconds=17.5),
            )
        self.assertEqual(opener.call_count, 3)
        self.assertTrue(all(call.kwargs["timeout"] == 17.5 for call in opener.call_args_list))
        self.assertIn("(request_timeout_seconds=17.5)", str(raised.exception))

    def test_in_sample_oauth_refresh_uses_the_call_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            auth_file = Path(directory) / "auth.json"
            auth_file.write_text(json.dumps({"tokens": {
                "access_token": "old-token", "refresh_token": "old-refresh",
            }}), encoding="utf-8")
            opener = mock.Mock(side_effect=_unauthorized)
            model = codex_model(
                model="codex-test", auth_file=auth_file, opener=opener,
                retry_sleep=_no_retry_sleep,
            )
            with mock.patch.object(
                responses, "refresh_codex_credentials", side_effect=AccountServiceError("offline"),
            ) as refresh:
                with self.assertRaises(ModelAuthenticationError):
                    model.sample(
                        InteractionContext((Message("user", "Hello."),)),
                        sample_params=SampleParams(request_timeout_seconds=17.5),
                    )
        refresh.assert_called_once()
        self.assertEqual(refresh.call_args.kwargs["timeout_seconds"], 17.5)
        self.assertEqual(opener.call_count, 2)
        self.assertTrue(all(call.kwargs["timeout"] == 17.5 for call in opener.call_args_list))

    def test_launch_option_then_catalog_then_default_for_each_http_api(self):
        catalog = parse_model_catalog(_TIMEOUT_CATALOG)
        for api in ("chat-completions", "messages", "codex", "responses"):
            for flags, expected in (([], 1800.0), (["--request-timeout-seconds", "9.5"], 9.5)):
                with self.subTest(api=api, flags=flags):
                    args = prepare_namespace(cli._build_parser().parse_args(
                        ["--model", f"slow-{api}", *flags]), catalog)
                    self.assertEqual(build_model(args).endpoint.request_timeout_seconds, expected)
                    config = InteractionConfig.from_namespace(args)
                    self.assertEqual(config.get("request_timeout_seconds"), expected)
                    self.assertEqual(config.snapshot().sample_params().request_timeout_seconds, expected)
                    # null restores the catalog value, not the launch option.
                    self.assertEqual(config.set("request_timeout_seconds", None), 1800.0)
            with self.subTest(api=api, model="uncatalogued"):
                args = prepare_namespace(cli._build_parser().parse_args([
                    "--endpoint-api", api, "--model", "other", "--endpoint-auth", "none",
                    "--endpoint-url", "http://127.0.0.1:9000/v1/model", "--max-output-tokens", "100",
                ]), catalog)
                self.assertEqual(
                    build_model(args).endpoint.request_timeout_seconds, DEFAULT_REQUEST_TIMEOUT_SECONDS,
                )

    def test_responses_preserves_none_sentinel_and_explicit_endpoint(self):
        self.assertIsNone(
            inspect.signature(CodexResponsesModel).parameters["request_timeout_seconds"].default
        )
        endpoint = responses_endpoint(
            api_url="http://localhost", model="model", bearer_token="FAKE",
            request_timeout_seconds=17.5,
        )
        self.assertIs(codex_model(endpoint).endpoint, endpoint)
        self.assertIs(
            codex_model(endpoint, request_timeout_seconds=None).endpoint, endpoint,
        )
        with self.assertRaisesRegex(ModelConfigurationError, "endpoint cannot be combined"):
            CodexResponsesModel(
                endpoint,
                request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
            )

    def test_endpoint_timeouts_are_bounded_to_usable_socket_values(self):
        for endpoint_factory in (
            lambda value: chat_endpoint(
                "http://localhost", request_timeout_seconds=value,
            ),
            lambda value: messages_endpoint(
                "http://localhost", "model", max_output_tokens=100,
                request_timeout_seconds=value,
            ),
            lambda value: responses_endpoint(
                "http://localhost", "model", "FAKE",
                request_timeout_seconds=value,
            ),
        ):
            # 1e12 used to pass validation, then overflow socket.settimeout.
            for value in (None, True, 0, -1, float("inf"), float("nan"), "300",
                          MAX_TIMEOUT_SECONDS + 1, 1e12):
                with self.subTest(endpoint=endpoint_factory, value=value):
                    with self.assertRaises(ModelConfigurationError):
                        endpoint_factory(value)
            self.assertEqual(
                endpoint_factory(MAX_TIMEOUT_SECONDS).request_timeout_seconds, MAX_TIMEOUT_SECONDS,
            )
        with self.assertRaises(ModelConfigurationError):
            codex_model(model="model", auth=CodexAuth("FAKE"), request_timeout_seconds=1e12)
        for value in (True, 0, -1, float("nan"), 1e12, "300"):
            with self.subTest(sample_param=value), self.assertRaises(ValueError):
                SampleParams(request_timeout_seconds=value)
        self.assertEqual(SampleParams(request_timeout_seconds=600).request_timeout_seconds, 600.0)

    def test_cli_and_demo_share_defaults_and_preserve_explicit_overrides(self):
        for frontend in (cli, demo):
            for api in ("chat-completions", "messages", "codex"):
                for flags, launch, expected in (
                    ([], None, DEFAULT_REQUEST_TIMEOUT_SECONDS),
                    (["--request-timeout-seconds", "9.5"], 9.5, 9.5),
                ):
                    with self.subTest(frontend=frontend.__name__, api=api, flags=flags):
                        args = frontend._build_parser().parse_args([
                            "--endpoint-api", api, "--model", "model",
                            *(
                                ("--max-output-tokens", "100", "--endpoint-auth", "none")
                                if api == "messages"
                                else ()
                            ),
                            *flags,
                        ])
                        with mock.patch(
                            "pythia.interaction.responses._load_default_model_auth",
                            return_value=CodexAuth("FAKE"),
                        ):
                            model = build_model(args)
                        # An omitted option stays None, so a catalog value can apply.
                        self.assertEqual(args.request_timeout_seconds, launch)
                        self.assertEqual(model.endpoint.request_timeout_seconds, expected)

    def test_help_displays_the_shared_default_and_io_semantics(self):
        for frontend in (cli, demo):
            with self.subTest(frontend=frontend.__name__):
                help_text = " ".join(frontend._build_parser().format_help().split())
                self.assertIn(f"default: {DEFAULT_REQUEST_TIMEOUT_SECONDS} seconds", help_text)
                self.assertIn("not an overall deadline", help_text)
                self.assertIn("timeouts.request_seconds", help_text)
                self.assertIn("/config request_timeout_seconds", help_text)

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

    def test_user_tools_keep_the_fixed_account_timeout(self):
        # Neither the launch option nor a live /config value reaches /login or /quota.
        for flags in ([], ["--request-timeout-seconds", "9.5"]):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                args = cli._build_parser().parse_args([
                    "--endpoint-api", "codex", "--model", "model",
                    "--endpoint-auth-home", directory, *flags,
                ])
                config = InteractionConfig.from_namespace(args)
                config.set("request_timeout_seconds", 600)
                environment = user_tools.create_user_environment(
                    args, notify=mock.Mock(), cancel=threading.Event(), config=config,
                )
                with mock.patch.object(user_tools, "login") as login:
                    result = environment.execute_tool_calls((ToolCall("login", "login-1", "{}"),))
                self.assertTrue(result.items[0].success)
                login.assert_called_once()
                self.assertEqual(login.call_args.kwargs["timeout_seconds"], DEFAULT_LOGIN_TIMEOUT_SECONDS)
                self.assertEqual(
                    login.call_args.kwargs["request_timeout_seconds"], DEFAULT_LOGIN_TIMEOUT_SECONDS,
                )
                with mock.patch.object(user_tools, "load_codex_auth", return_value=CodexAuth("FAKE")):
                    with mock.patch.object(user_tools, "query_quota", return_value="snapshot") as quota:
                        result = environment.execute_tool_calls((ToolCall("quota", "quota-1", "{}"),))
                self.assertTrue(result.items[0].success)
                quota.assert_called_once_with(
                    CodexAuth("FAKE"), timeout_seconds=DEFAULT_LOGIN_TIMEOUT_SECONDS,
                )

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
