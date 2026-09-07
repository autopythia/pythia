from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import (
    ChatCompletionsEndpoint, ChatCompletionsModel, CodexAuth, CodexAuthUnavailable,
    CodexResponsesModel, ContextCompaction, ContextValidationError, Environment,
    Instructions, Message, MessagesEndpoint, MessagesModel, ModelContext, ModelSample,
    ModelSampleBoundary, OpaqueCompaction, PromptSummarizingCompactor, Init,
    TokenUsage, ToolCall, ToolResult, TurnMetadata, TurnSummary, UserInteraction,
    UserInteractionBoundary, UserToolCall, UserToolResult, load_interaction_save,
    render_interaction_items, save_interaction_save,
)
from pythia.interaction import cli, demo, user_tools
from pythia.interaction._cli_editor import Editor
from pythia.interaction.model_config import build_model, supports_account_services
from pythia_test.test_interaction_cli import _Model, _Terminal, _answer


def _records(name="quota", call_id="user_one"):
    return (UserToolCall(ToolCall(name, call_id, "{}")),
            UserToolResult(ToolResult(call_id, "private account snapshot")))


class UserToolValueTests(unittest.TestCase):
    def test_empty_user_tool_arguments_are_elided_only_from_display(self):
        for name in ("quota", "login"):
            for raw_arguments in ("{}", "{ }", " \n{\n}\t "):
                for show_arguments in (False, True):
                    with self.subTest(name=name, raw=raw_arguments, show=show_arguments):
                        call = UserToolCall(ToolCall(name, "user_one", raw_arguments))
                        display = render_interaction_items(
                            (call,), show_generic_arguments=show_arguments,
                        )
                        self.assertEqual(tuple(i.text for i in display), (
                            f"[user-tool-call] {name} (user_one)",
                        ))
                        self.assertEqual(call.call.arguments_json, raw_arguments)

    def test_nonempty_falsy_and_malformed_user_arguments_remain_visible(self):
        cases = (
            (' { "workspace_id": "workspace" } ', '{"workspace_id":"workspace"}'),
            ("[]", "[]"), ("null", "null"), ("false", "false"), ("0", "0"),
            ('""', '""'), ('{"broken"', '{"broken"'),
        )
        for raw_arguments, expected in cases:
            for show_arguments in (False, True):
                with self.subTest(raw=raw_arguments, show=show_arguments):
                    call = UserToolCall(ToolCall("login", "user_one", raw_arguments))
                    display = render_interaction_items(
                        (call,), show_generic_arguments=show_arguments,
                    )
                    self.assertEqual(tuple(i.text for i in display), (
                        "[user-tool-call] login (user_one)", expected,
                    ))
                    self.assertEqual(call.call.arguments_json, raw_arguments)

    def test_model_empty_arguments_and_all_empty_object_results_are_unchanged(self):
        call = ToolCall("lookup", "model_one", "{}")
        self.assertEqual(tuple(i.text for i in render_interaction_items((call,))), (
            "[tool-call] lookup (model_one)",
        ))
        self.assertEqual(tuple(i.text for i in render_interaction_items(
            (call,), show_generic_arguments=True,
        )), ("[tool-call] lookup (model_one)", "{}"))
        result = ToolResult("model_one", "{}")
        for show_arguments in (False, True):
            with self.subTest(show=show_arguments):
                self.assertEqual(tuple(i.text for i in render_interaction_items(
                    (result,), source_calls=(call,), show_generic_arguments=show_arguments,
                )), ("[tool-ret]  lookup (model_one) [ok]\n{}",))
                self.assertEqual(tuple(i.text for i in render_interaction_items(
                    (UserToolResult(result),), source_user_calls=(UserToolCall(call),),
                    show_generic_arguments=show_arguments,
                )), ("[user-tool-ret]  lookup (model_one) [ok]\n{}",))

    def test_empty_arguments_round_trip_and_old_quota_result_replay_without_rewriting(self):
        raw_arguments = " {\n } "
        old_output = "Quota snapshot at earlier\nplan: unavailable"
        original = (Init("old"),
                    UserToolCall(ToolCall("quota", "user_one", raw_arguments)),
                    UserToolResult(ToolResult("user_one", old_output)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interaction.jsonl"
            save_interaction_save(path, ModelContext(original))
            before = path.read_bytes()
            restored = load_interaction_save(path)
            for _ in range(2):
                self.assertEqual(tuple(i.text for i in render_interaction_items(restored.items)), (
                    "[user-tool-call] quota (user_one)",
                    f"[user-tool-ret]  quota (user_one) [ok]\n{old_output}",
                ))
                self.assertEqual(restored.items, original)
                self.assertEqual(path.read_bytes(), before)

    def test_round_trip_projection_provider_payloads_and_turn_state(self):
        base = (Init("session"), Message("user", "hello"), UserInteractionBoundary(),
                Message("assistant", "answer"),
                TurnMetadata(TokenUsage(), provider_turn_id="turn", provider_turn_state="opaque"),
                ModelSampleBoundary(), TurnSummary(sample_count=1))
        context = ModelContext((*base, *_records()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            save_interaction_save(path, context)
            self.assertIn('"type": "user_tool_call"', path.read_text())
            self.assertEqual(load_interaction_save(path).items, context.items)
        self.assertEqual(context.model_items(), ModelContext(base).model_items())
        models = (
            ChatCompletionsModel(ChatCompletionsEndpoint("http://localhost:8000")),
            MessagesModel(MessagesEndpoint(api_url="https://api.anthropic.com", model="test", api_key="fake")),
            CodexResponsesModel(model="test", auth=CodexAuth("fake")),
        )
        for model in models:
            with self.subTest(model=type(model).__name__):
                self.assertEqual(model._build_request_payload(context, (), None),
                                 model._build_request_payload(ModelContext(base), (), None))
                self.assertNotIn("private account", repr(model._build_request_payload(context, (), None)))
        self.assertIsNone(cli._resume_notice(context))
        self.assertEqual(demo._final_assistant_text(context), "answer")

    def test_origin_aware_display_and_validation(self):
        call, result = _records(call_id="same")
        model_call, model_result = ToolCall("model_tool", "same", "{}"), ToolResult("same", "model result")
        context = ModelContext((model_call, model_result, call, result))
        display = "\n".join(i.text for i in render_interaction_items(context.items))
        self.assertIn("[tool-ret]  model_tool (same)", display)
        self.assertIn("[user-tool-ret]  quota (same)", display)
        for items in (
            (result,), (call, model_result), (model_call, result),
            (call, call), (call, Message("user", "early")),
            (call, result, call), (model_call, call),
            (ContextCompaction((call, result)),),
        ):
            with self.subTest(items=items), self.assertRaises(ContextValidationError):
                ModelContext(items)
        pending = ModelContext((call,))
        self.assertEqual(pending.pending_tool_calls(), ())
        self.assertEqual(pending.pending_user_tool_calls(), (call,))
        with self.assertRaises(ContextValidationError):
            pending.assert_model_ready()
        with self.assertRaises(ValueError):
            UserInteraction((call,))
        with self.assertRaises(Exception):
            ModelSample((call,))
        with self.assertRaises(TypeError):
            UserToolResult(call)

    def test_compaction_excludes_user_tools_but_retains_raw_replay(self):
        model = mock.Mock()
        model.sample.return_value = _answer("summary")
        context = ModelContext((Message("user", "hello"), *_records()))
        result = PromptSummarizingCompactor(model).compact(context)
        submitted = model.sample.call_args.args[0]
        self.assertFalse(any(isinstance(i, (UserToolCall, UserToolResult)) for i in submitted))
        context.extend(result.context_items())
        self.assertNotIn("private account", repr(context.model_items()))
        self.assertIn("private account", "\n".join(i.text for i in render_interaction_items(context.items)))

    def test_syntax_no_secret_echo_and_model_dispatch_cannot_call_user_tools(self):
        self.assertEqual(user_tools.parse_user_tool("/login workspace").arguments_json,
                         '{"workspace_id": "workspace"}')
        for text in ("/login secret.code", "/quota secret-token", "/login\ncode", "/unknown-secret"):
            with self.subTest(text=text), self.assertRaises(ValueError) as error:
                user_tools.parse_user_tool(text)
            self.assertNotIn("secret", str(error.exception))
        for name in ("login", "quota"):
            result = Environment().execute_tool_calls((ToolCall(name, "model", "{}"),))
            self.assertFalse(result.items[0].success)
            self.assertIn("Unknown tool", result.items[0].output)

    def test_auth_needed_editor_blocks_only_model_submissions_and_quit_bypasses_queue(self):
        state = cli._UIState(ready=True, auth_required=True, editor=Editor("hello", 5))
        state.handle_key("c-m", "")
        self.assertEqual(state.editor.text, "hello")
        self.assertEqual(tuple(state.pending), ())
        for _ in range(9):
            state.editor = Editor("/quota", 6)
            state.handle_key("c-m", "")
        self.assertEqual(len(state.pending), 8)
        self.assertEqual(state.editor.text, "/quota")
        self.assertTrue(all(isinstance(i, user_tools.UserToolIntent) for i in state.pending))
        state.editor = Editor("/quit", 5)
        state.handle_key("c-m", "")
        self.assertTrue(state.closing)
        self.assertTrue(state.login_cancel.is_set())
        self.assertFalse(state.pending)


class AuthConfigurationTests(unittest.TestCase):
    def test_main_opens_only_missing_auth_case_and_rejects_other_config_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            base = ["--model-api", "codex", "--model", "test", "--codex-home", directory]
            for extra, code, opens in (([], 0, True), (["--api-url", "bad-url"], 1, False),
                                       (["--request-timeout-seconds", "0"], 1, False)):
                with self.subTest(extra=extra):
                    stream = SimpleNamespace(isatty=lambda: True)
                    with mock.patch.object(cli.sys, "stdin", stream), mock.patch.object(cli.sys, "stdout", stream):
                        with mock.patch.object(cli, "DefaultEnvironment") as environment:
                            with mock.patch.object(cli, "PosixTerminal"):
                                with mock.patch.object(cli, "_run", new_callable=mock.AsyncMock, return_value=0) as run:
                                    with mock.patch("builtins.print"):
                                        self.assertEqual(cli.main([*base, *extra]), code)
                    if opens:
                        self.assertIsNone(run.call_args.args[0])
                        self.assertEqual(environment.call_count, 1)
                    else:
                        environment.assert_not_called()
                        run.assert_not_called()

    def test_non_auth_validation_precedes_missing_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ["--model-api", "codex", "--model", "test", "--codex-home", directory]
            with self.assertRaises(CodexAuthUnavailable):
                build_model(cli._build_parser().parse_args(argv))
            for extra in (["--api-url", "not-a-url"], ["--request-timeout-seconds", "nan"],
                          ["--messages-server-compaction"], ["--api-key", "secret"],
                          ["--codex-home", ""]):
                with self.subTest(extra=extra):
                    with self.assertRaises(ValueError) as error:
                        build_model(cli._build_parser().parse_args([*argv, *extra]))
                    self.assertNotIsInstance(error.exception, CodexAuthUnavailable)

    def test_capabilities_do_not_follow_wire_api_name_and_demo_stays_fail_fast(self):
        for argv, supported in (
            (["--model-api", "codex", "--model", "test"], True),
            (["--model-api", "codex", "--model", "muse-spark-1.3"], False),
            (["--model-api", "codex", "--model", "test", "--api-url", "https://example.org"], False),
            ([], False), (["--model-api", "messages", "--model", "test"], False),
        ):
            self.assertEqual(supports_account_services(cli._build_parser().parse_args(argv)), supported)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(demo, "DefaultEnvironment") as environment:
                with mock.patch("sys.stderr", io.StringIO()):
                    self.assertEqual(demo.main(["--model-api", "codex", "--model", "test",
                                                "--codex-home", directory]), 1)
                environment.assert_not_called()


class UserToolControllerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "interaction.jsonl"
        self.auth_path = self.root / "auth.json"
        self.args = cli._build_parser().parse_args([
            "--model-api", "codex", "--model", "test", "--codex-auth-file", str(self.auth_path),
        ])

    async def run_cli(self, model, terminal):
        result = await asyncio.wait_for(cli._run(model, Environment(), terminal, self.args, self.path), 4)
        self.assertTrue(terminal.exited)
        return result

    async def test_quota_plan_propagates_http_to_log_and_replay_without_empty_argument_block(self):
        token = "FAKE_BEARER"
        self.auth_path.write_text(json.dumps({
            "tokens": {"access_token": token, "account_id": "account"},
        }))
        for plan_type, expected_plan in (("prolite", "prolite"), (token, "[redacted]")):
            with self.subTest(plan_type=plan_type):
                self.args.resume = False
                submitted = False
                model = _Model(self.path)
                response = io.BytesIO(json.dumps({
                    "plan_type": plan_type,
                    "rate_limit": {"primary_window": {"used_percent": 25}},
                }).encode())

                def open_request(request, *, timeout):
                    pending = load_interaction_save(self.path).pending_user_tool_calls()
                    self.assertEqual(len(pending), 1)
                    self.assertEqual(pending[0].call.name, "quota")
                    self.assertEqual(pending[0].call.arguments_json, "{}")
                    self.assertEqual(request.get_method(), "GET")
                    self.assertEqual(request.full_url, "https://chatgpt.com/backend-api/wham/usage")
                    self.assertEqual(request.get_header("Authorization"), f"Bearer {token}")
                    self.assertEqual(timeout, self.args.request_timeout_seconds)
                    return response

                def frame(t, editor, status):
                    nonlocal submitted
                    if status == "idle":
                        if not submitted:
                            submitted = True
                            t.submit("/quota")
                        elif any(i.text.startswith("[user-tool-ret]  quota") for i in t.items):
                            t.key("c-d")

                terminal = _Terminal(frame)
                with mock.patch("pythia.interaction._account_http.urllib.request.build_opener") as factory:
                    opener = factory.return_value.open
                    opener.side_effect = open_request
                    self.assertEqual(await self.run_cli(model, terminal), 0)
                    opener.assert_called_once()
                    self.assertTrue(response.closed)
                    saved = load_interaction_save(self.path)
                    self.assertEqual(tuple(type(i) for i in saved), (
                        Init, UserToolCall, UserToolResult,
                    ))
                    self.assertEqual(saved.items[1].call.arguments_json, "{}")
                    self.assertTrue(saved.items[2].result.success)
                    self.assertIn(f"\nplan: {expected_plan}\n", saved.items[2].result.output)
                    self.assertNotIn(token, self.path.read_text())
                    self.assertNotIn("{}", tuple(i.text for i in terminal.items))
                    transcript = tuple(i for i in terminal.items if not i.text.startswith("[cli]"))
                    self.assertEqual(len(transcript), 2)
                    self.assertEqual(transcript, render_interaction_items(saved.items))

                    before = self.path.read_bytes()
                    self.args.resume = True
                    replay = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
                    with mock.patch.object(cli, "save_interaction_save") as save:
                        self.assertEqual(await self.run_cli(model, replay), 0)
                    save.assert_not_called()
                    opener.assert_called_once()
                    self.assertEqual(self.path.read_bytes(), before)
                    self.assertEqual(tuple(i for i in replay.items if not i.text.startswith("[cli]")), transcript)
                self.assertEqual(model.calls, [])

    async def test_missing_auth_login_then_explicit_query_never_replays_initial_prompt(self):
        self.args.prompt = "initial draft"
        model = _Model(self.path, _answer())
        step = 0

        def frame(t, editor, status):
            nonlocal step
            if status == "auth needed" and step == 0:
                self.assertEqual(editor.text, "initial draft")
                t.key("c-m")  # blocked, no user boundary
                step = 1
            elif status == "auth needed" and step == 1:
                self.assertEqual(editor.text, "initial draft")
                t.submit("/quota")
                step = 2
            elif status == "auth needed" and step == 2 and any(isinstance(i, UserToolResult) for i in load_interaction_save(self.path)):
                t.submit("/login")
                step = 3
            elif status == "idle" and step == 3:
                self.assertEqual(model.calls, [])
                t.submit("explicit query")
                step = 4
            elif status == "idle" and step == 4:
                t.submit("/quit")

        def login(path, **kwargs):
            pending = load_interaction_save(self.path).pending_user_tool_calls()
            self.assertEqual(pending[0].call.name, "login")
            self.assertFalse(any(isinstance(i, UserInteractionBoundary) for i in load_interaction_save(self.path)))
            path.write_text('{"tokens":{"access_token":"FAKE_SECRET","account_id":"account"}}')

        def activate(args):
            saved = load_interaction_save(self.path)
            self.assertIsInstance(saved.items[-1], UserToolResult)
            self.assertTrue(saved.items[-1].result.success)
            return model

        terminal = _Terminal(frame)
        with mock.patch.object(user_tools, "login", side_effect=login):
            with mock.patch.object(cli, "build_model", side_effect=activate):
                self.assertEqual(await self.run_cli(None, terminal), 0)
        saved = load_interaction_save(self.path)
        self.assertEqual([i for i in saved if isinstance(i, Message) and i.role == "user"], [Message("user", "explicit query")])
        self.assertEqual([i.call.name for i in saved if isinstance(i, UserToolCall)], ["quota", "login"])
        self.assertEqual(len(model.calls), 1)
        self.assertNotIn("FAKE_SECRET", self.path.read_text())
        self.assertNotIn("FAKE_SECRET", "\n".join(i.text for i in terminal.items))
        self.assertEqual(model.calls[0][0].items, model.checkpoints[0])

    async def test_resume_unfinished_user_tool_never_executes_and_preserves_model_tail(self):
        original = (Init("old"), Message("assistant", "answer"), TurnSummary(), _records("login")[0])
        save_interaction_save(self.path, ModelContext(original))
        self.args.resume = True
        model = _Model(self.path)
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        with mock.patch.object(user_tools, "login") as login:
            self.assertEqual(await self.run_cli(model, terminal), 0)
        login.assert_not_called()
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items[:-1], original)
        self.assertFalse(saved.items[-1].result.success)
        self.assertIsNone(cli._resume_notice(saved))
        self.assertEqual(model.calls, [])
        before = self.path.read_bytes()
        await self.run_cli(model, _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None))
        self.assertEqual(self.path.read_bytes(), before)

    async def test_user_call_and_result_save_failures_block_login_or_activation(self):
        for fail_result in (False, True):
            with self.subTest(fail_result=fail_result):
                attempted = []

                def save(path, context):
                    target = UserToolResult if fail_result else UserToolCall
                    if any(isinstance(i, target) for i in context):
                        raise OSError("disk failed")
                    save_interaction_save(path, context)

                def login(path, **kwargs):
                    attempted.append(True)

                def frame(t, editor, status):
                    if status == "auth needed":
                        t.submit("/login")
                    if status == "failed":
                        t.key("c-d")

                with mock.patch.object(cli, "save_interaction_save", side_effect=save):
                    with mock.patch.object(user_tools, "login", side_effect=login):
                        with mock.patch.object(cli, "build_model") as activate:
                            self.assertEqual(await self.run_cli(None, _Terminal(frame)), 1)
                activate.assert_not_called()
                self.assertEqual(len(attempted), int(fail_result))
                saved = load_interaction_save(self.path)
                self.assertEqual(bool(saved.pending_user_tool_calls()), fail_result)

    async def test_quit_during_login_cancels_wait_and_checkpoints_safe_result(self):
        entered = threading.Event()

        def login(path, *, notify, cancel, **kwargs):
            notify("fake transient challenge")
            entered.set()
            if not cancel.wait(2):
                raise AssertionError("login was not cancelled")
            raise user_tools.AccountServiceError("Login cancelled.")

        def frame(t, editor, status):
            if status == "auth needed" and not entered.is_set():
                t.submit("/login")
            if any("fake transient challenge" in i.text for i in t.items):
                t.key("c-c")

        with mock.patch.object(user_tools, "login", side_effect=login):
            self.assertEqual(await self.run_cli(None, _Terminal(frame)), 0)
        saved = load_interaction_save(self.path)
        self.assertFalse(saved.pending_user_tool_calls())
        self.assertFalse(saved.items[-1].result.success)
        self.assertNotIn("fake transient challenge", self.path.read_text())

    async def test_login_failure_and_activation_failure_are_safe_and_do_not_sample(self):
        for activation in (False, True):
            with self.subTest(activation=activation):
                calls = []

                def login(path, **kwargs):
                    calls.append(True)
                    if not activation:
                        raise RuntimeError("FAKE_SECRET")

                def frame(t, editor, status):
                    if status == "auth needed":
                        t.submit("/quit" if calls else "/login")

                terminal = _Terminal(frame)
                with mock.patch.object(user_tools, "login", side_effect=login):
                    with mock.patch.object(cli, "build_model", side_effect=RuntimeError("FAKE_SECRET")):
                        self.assertEqual(await self.run_cli(None, terminal), int(activation))
                self.assertNotIn("FAKE_SECRET", self.path.read_text())
                self.assertNotIn("FAKE_SECRET", "\n".join(i.text for i in terminal.items))
                self.assertFalse(any(isinstance(i, UserInteractionBoundary) for i in load_interaction_save(self.path)))

    async def test_authenticated_prompt_starting_with_login_is_literal_text(self):
        self.args.prompt = "/login"
        model = _Model(self.path, _answer())
        with mock.patch.object(user_tools, "login") as login:
            await self.run_cli(model, _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None))
        login.assert_not_called()
        self.assertIn(Message("user", "/login"), model.calls[0][0].items)

    async def test_quota_queued_during_sample_runs_after_turn_and_before_follow_up(self):
        self.auth_path.write_text('{"tokens":{"access_token":"fake","account_id":"account"}}')
        entered, release = threading.Event(), threading.Event()
        step = 0

        def first(context):
            entered.set()
            if not release.wait(2):
                raise AssertionError("sample not released")
            return _answer("first")

        def quota(auth, **kwargs):
            items = load_interaction_save(self.path).items
            self.assertIsInstance(items[-1], UserToolCall)
            self.assertIsInstance(items[-2], TurnSummary)
            self.assertNotIn(Message("user", "follow-up"), items)
            return "quota snapshot"

        def frame(t, editor, status):
            nonlocal step
            if entered.is_set() and step == 0:
                t.submit("/quota")
                t.submit("follow-up")
                step = 1
            if "queued=2" in status:
                release.set()
            if status == "idle" and any(i.text == "[assistant] second" for i in t.items):
                t.key("c-d")

        self.args.prompt = "first"
        model = _Model(self.path, first, _answer("second"))
        try:
            with mock.patch.object(user_tools, "query_quota", side_effect=quota) as query:
                self.assertEqual(await self.run_cli(model, _Terminal(frame)), 0)
        finally:
            release.set()
        query.assert_called_once()
        self.assertEqual(len(model.calls), 2)
        saved = load_interaction_save(self.path)
        self.assertEqual(len([i for i in saved if isinstance(i, TurnSummary)]), 2)

    async def test_same_account_login_rebinds_without_sampling_or_changing_provider_state(self):
        metadata = TurnMetadata(TokenUsage(), provider_turn_id="turn", provider_turn_state="state")
        original = (Init("old"), Message("assistant", "answer"), metadata, TurnSummary())
        save_interaction_save(self.path, ModelContext(original))
        self.args.resume = True
        model = _Model(self.path)
        model.endpoint = SimpleNamespace(account_id="account")
        rebuilt = []
        submitted = False

        def login(path, **kwargs):
            self.assertEqual(kwargs["expected_account"], "account")
            path.write_text('{"tokens":{"access_token":"new-token","account_id":"account"}}')

        def activate(args):
            candidate = build_model(args)
            rebuilt.append(candidate)
            return candidate

        def frame(t, editor, status):
            nonlocal submitted
            if status == "idle":
                if rebuilt:
                    t.submit("/quit")
                elif not submitted:
                    submitted = True
                    t.submit("/login")

        with mock.patch.object(user_tools, "login", side_effect=login):
            with mock.patch.object(cli, "build_model", side_effect=activate):
                self.assertEqual(await self.run_cli(model, _Terminal(frame)), 0)
        self.assertEqual(len(rebuilt), 1)
        self.assertEqual(rebuilt[0].endpoint.bearer_token, "new-token")
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items[:len(original)], original)
        self.assertEqual(saved.model_items(), ModelContext(original).model_items())
        self.assertEqual(len(saved.items), len(original) + 2)
        self.assertEqual(model.calls, [])

    async def test_auth_needed_instructions_resume_does_not_sample_or_add_a_user_boundary(self):
        original = (Init("old"), Message("assistant", "answer"), TurnSummary())
        save_interaction_save(self.path, ModelContext(original))
        self.args.resume, self.args.instructions = True, "new instructions"
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "auth needed" else None)
        self.assertEqual(await self.run_cli(None, terminal), 0)
        saved = load_interaction_save(self.path)
        self.assertEqual(saved.items, (*original, Instructions("new instructions")))


if __name__ == "__main__":
    unittest.main()
