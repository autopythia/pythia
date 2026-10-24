"""Explicit, CLI-local retries: no new user turn and no old tool replay."""

from collections import deque
from dataclasses import replace
import json
import threading
from types import SimpleNamespace
import unittest
from pythia_test.interaction_helpers import chat_endpoint, full_log, patch_compaction, patch_saves
from pythia_test.interaction_helpers import real_save, seed_cli_save
from unittest import mock

from pythia.interaction import (
    ChatCompletionsModel, CompactionContextWindowError, CompactionMetadata, CompactionResult,
    CompactionSettings, ContextPrefix, DEFAULT_REQUEST_TIMEOUT_SECONDS, Environment, Init,
    InteractionContext,
    Message, ModelAuthenticationError, ModelContextWindowError, ModelFailure, ModelRateLimitError,
    ModelResponseError, NothingToCompact,
    ModelSample, ModelSampleBoundary, ModelTimeoutError, OpaqueCompaction,
    Reasoning, SampleParams, SaveError, Tool, ToolCall, ToolOutcome,
    ToolResult, ToolSpec, Tools, TurnSummary, UserInteractionBoundary, UserToolCall, UserToolResult,
    cli, load_interaction_save, save_interaction_save,
)
from pythia.interaction import user_tools
from pythia.interaction._cli_editor import Editor
from pythia.interaction.claude_relay._mcp import NativeToolRejected
from pythia.interaction.items import SampleMetadata
from pythia.interaction.loop import kernel
from pythia_test.test_interaction import _FakeHTTPResponse, _ScriptedOpener
from pythia_test.test_interaction_cli import _ControllerTestCase, _Model, _Terminal, _answer


HINT = "[cli] Sampling failed. Use /retry to try again."
AUTH_HINT = "[cli] Model authentication needed; use /login."
TIMEOUT_HINT = ("[cli] The request timeout is {:g} seconds; to wait longer, "
                "use /config timeouts_request_seconds N before /retry.")


def submit(state, text="/retry"):
    state.editor = Editor(text, len(text))
    state.handle_key("c-m", "\r")


def retry_then_quit(count=1):
    retries = 0

    def frame(terminal, editor, status):
        nonlocal retries
        if status == "failed" and retries < count:
            retries += 1
            terminal.submit("/retry")
        elif status == "idle":
            terminal.key("c-d")
    return frame


class RetryInputTests(unittest.TestCase):
    def test_retry_is_typed_deduplicated_and_not_auth_gated(self):
        ticket = cli._RetryIntent()
        state = cli._UIState(ready=True, phase="failed", retry=ticket, auth_required=True)
        submit(state)
        submit(state)
        self.assertEqual(tuple(state.pending), (ticket,))
        self.assertTrue(any("already queued" in item.text for item in state.displays))
        state.request_stop()
        self.assertIsNone(state.retry)
        self.assertFalse(state.pending)

    def test_retry_rejects_busy_missing_failure_and_invalid_arguments(self):
        cases = [("idle", None, "/retry", "No retryable sampling failure")]
        for phase in ("sampling", "saving", "compacting", "loading model", "tool: exec_command"):
            cases.append((phase, cli._RetryIntent(), "/retry", "work is in progress"))
        for text in ("/retry FAKE_SECRET", "/retry\nFAKE_SECRET", "/retry\r"):
            cases.append(("failed", cli._RetryIntent(), text, "Usage: /retry"))
        for phase, ticket, text, notice in cases:
            with self.subTest(phase=phase, text=text):
                state = cli._UIState(ready=True, phase=phase, retry=ticket)
                submit(state, text)
                self.assertFalse(state.pending)
                displayed = "\n".join(item.text for item in state.displays)
                self.assertIn(notice, displayed)
                self.assertNotIn("FAKE_SECRET", displayed)

    def test_full_queue_and_checkpoint_failure_preserve_unaccepted_draft(self):
        for persistence_failed in (False, True):
            with self.subTest(persistence_failed=persistence_failed):
                state = cli._UIState(ready=True, phase="failed", retry=cli._RetryIntent(),
                                     persistence_failed=persistence_failed)
                state.pending.extend(str(i) for i in range(8))
                submit(state)
                self.assertEqual(len(state.pending), 8)
                self.assertEqual(state.editor.text, "/retry")


def _rejected():
    """A rejected tool name, as the Claude relay raises it: eligible for a retry."""
    error = NativeToolRejected("probe", "mcp__pythia__probe")
    error.failure = ModelFailure("NativeToolRejected", str(error), provider="claude-relay",
                                 error_code="native_tool_name_rejected")
    return error


def _rate_limited(retry_after=None):
    """A rate limit, as an HTTP adapter raises it once its own retries are used up."""
    return ModelRateLimitError("Chat Completions HTTP 429: slow down", failure=ModelFailure(
        "rate_limit", "Chat Completions HTTP 429: rate limited", provider="chat-completions",
        http_status=429, attempt_count=3), retry_after_seconds=retry_after)


def _notices(state):
    texts = [item.text.removeprefix("[cli] ") for item in state.displays]
    state.displays.clear()
    return texts


class _Adapter(_Model):
    """A real model adapter, recorded as the scripted model is."""

    def __init__(self, path, adapter):
        super().__init__(path)
        self.adapter = adapter

    def sample(self, context, *, tools=(), sample_params=None):
        self.calls.append((context.copy(), tuple(tools), sample_params))
        self.checkpoints.append(load_interaction_save(full_log(self.path)).items)
        self.threads.append(threading.get_ident())
        return self.adapter.sample(context, tools=tools, sample_params=sample_params)


class CancelInputTests(unittest.TestCase):
    """/cancel stops the running turn's automatic retries, and only those."""

    def test_cancel_needs_an_automatic_retry_in_progress_and_no_arguments(self):
        for phase in ("idle", "sampling", "executing tools", "failed"):
            with self.subTest(phase=phase):
                state = cli._UIState(ready=True)
                state.set_phase(phase)
                submit(state, "/cancel")
                self.assertEqual(_notices(state), ["No automatic retry to cancel."])
                self.assertFalse(state.retry_cancel.is_set())
                self.assertEqual(tuple(state.pending), ())
        state = cli._UIState(ready=True)
        state.set_phase("retrying")
        submit(state, "/cancel now")
        self.assertEqual(_notices(state), ["Usage: /cancel (no arguments; single line)."])
        submit(state, "/cancel!")  # it takes no bangs
        [notice] = _notices(state)
        self.assertTrue(notice.startswith("Unsupported command. Use /retry, /continue, /cancel, "),
                        notice)
        self.assertFalse(state.retry_cancel.is_set())

    def test_cancel_takes_effect_at_once_until_the_sample_succeeds_or_the_turn_ends(self):
        state = cli._UIState(ready=True)
        state.set_phase("retrying")
        submit(state, "/cancel")
        self.assertTrue(state.retry_cancel.is_set())
        # Its record shows at once, and waits for the next safe point to be logged.
        shown = _notices(state)
        self.assertEqual(len(shown), 2)
        self.assertTrue(shown[0].startswith("[user-tool-call] cancel (user_"), shown)
        self.assertTrue(shown[1].endswith("[ok]\nStopped the automatic retries."), shown)
        [entry] = state.pending
        self.assertIsInstance(entry, cli._RecordEntry)
        self.assertEqual((entry.call.call.name, entry.call.call.arguments_json), ("cancel", "{}"))
        submit(state, "/cancel")
        self.assertEqual(_notices(state), ["The automatic retries are already cancelled."])
        # A retried attempt still counts as the retry in progress.
        state.set_phase("sampling")
        self.assertTrue(state.auto_retrying)
        for end in ("idle", "failed", "auth needed"):
            state.set_phase("retrying")
            state.set_phase(end)
            self.assertFalse(state.auto_retrying, end)
        state.set_phase("retrying")
        cli._end_auto_retry(state)  # a sample succeeded
        self.assertFalse(state.auto_retrying)


class AutomaticRetryTests(_ControllerTestCase):
    """Failures the model vouches for, which the turn loop samples again by itself."""

    async def test_an_eligible_failure_is_sampled_again_while_the_status_says_retrying(self):
        model = _Model(self.path, _rejected(), _answer("recovered"))
        statuses = []

        def frame(terminal, editor, status):
            statuses.append(status)
            if status == "idle":
                terminal.key("c-d")
        terminal = _Terminal(frame)
        self.assertEqual(await self._run(model, terminal, [
            "--resume=False", "--prompt", "original", "--enable-default-tools=False",
        ]), 0)
        self.assertEqual(len(model.calls), 2)
        # The status shows how many of the sample's attempts failed, of all.
        self.assertTrue(any(status.startswith("retrying") and status.endswith(" (1/3)")
                            for status in statuses), statuses)
        self.assertFalse(any("(1/3)" in status for status in statuses[statuses.index("idle"):]))
        shown = [item.text for item in terminal.items]
        notice = "[cli] Sampling again in 2 s (1/3)."
        self.assertIn(notice, shown)
        self.assertNotIn(HINT, shown)
        saved = load_interaction_save(self.path)
        failures = [item for item in saved if isinstance(item, ModelFailure)]
        self.assertEqual([failure.error_code for failure in failures], ["native_tool_name_rejected"])
        self.assertEqual([item.recovery for item in saved if isinstance(item, SampleMetadata)],
                         [("native_tool_name_rejected_retry",)])
        self.assertIsInstance(saved[-1], TurnSummary)

    async def test_the_last_attempt_s_failure_says_how_often_sampling_failed(self):
        model = _Model(self.path, _rejected(), _rejected(), _rejected(), _answer("recovered"))
        terminal = _Terminal(retry_then_quit())
        with mock.patch.object(kernel, "DEFAULT_AUTO_RETRY", kernel.AutoRetryPolicy("default", 3, (0, 0))):
            self.assertEqual(await self._run(model, terminal, [
                "--resume=False", "--prompt", "original", "--enable-default-tools=False",
            ]), 1)  # the sticky failure exit status
        self.assertEqual(len(model.calls), 4)  # three attempts, then the /retry
        shown = [item.text for item in terminal.items]
        exhausted = "[cli] Sampling failed 3 times. Use /retry to try again."
        self.assertEqual(shown.count(exhausted), 1)
        self.assertNotIn(HINT, shown)
        self.assertLess(shown.index("[cli] NativeToolRejected: Native CLI rejected a call to probe: "
                                    "no such tool; its callable name is mcp__pythia__probe"),
                        shown.index(exhausted))
        saved = load_interaction_save(self.path)
        self.assertEqual(sum(isinstance(item, ModelFailure) for item in saved), 3)
        self.assertEqual([i.call.name for i in saved if isinstance(i, UserToolCall)], ["retry"])

    async def test_a_rate_limit_retries_as_often_as_the_turn_s_config_says(self):
        # Launched with 2 retries; a /config change between turns sets the next one's.
        model = _Model(self.path, *(_rate_limited() for _ in range(4)))
        failures = 0
        last = None

        def frame(terminal, editor, status):
            nonlocal failures, last
            if status == "failed" and last != "failed":
                failures += 1
                if failures == 1:
                    terminal.submit("/config endpoint_rate_limit_retries 0")
                    terminal.submit("/retry")
                else:
                    terminal.key("c-d")
            last = status
        terminal = _Terminal(frame)
        with mock.patch.object(kernel, "RATE_LIMIT_DELAYS_SECONDS", (0,)):
            self.assertEqual(await self._run(model, terminal, [
                "--resume=False", "--prompt", "original", "--enable-default-tools=False",
                "--endpoint-rate-limit-retries", "2",
            ]), 1)
        self.assertEqual(len(model.calls), 4)  # three attempts, then one after /retry
        shown = [item.text for item in terminal.items]
        self.assertEqual([text for text in shown if text.startswith("[cli] Sampling")], [
            "[cli] Sampling again in 0 s (1/3).", "[cli] Sampling again in 0 s (2/3).",
            "[cli] Sampling failed 3 times. Use /retry to try again.",
            HINT,  # no retries: the one attempt failed
        ])
        # With 3 attempts, the 3rd is the last: the error shows when the turn ends.
        error = "[cli] ModelRateLimitError: Chat Completions HTTP 429: slow down"
        self.assertEqual(shown.count(error), 2)
        self.assertEqual(shown.index(error) + 1,
                         shown.index("[cli] Sampling failed 3 times. Use /retry to try again."))
        saved = load_interaction_save(self.path)
        self.assertEqual([item.category for item in saved if isinstance(item, ModelFailure)],
                         ["rate_limit"] * 4)

    async def test_an_unbounded_retry_shows_each_error_from_its_3rd_attempt(self):
        # It may never end the turn, where the error would show.
        model = _Model(self.path, *(_rate_limited() for _ in range(4)), _answer("recovered"))
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        with mock.patch.object(kernel, "RATE_LIMIT_DELAYS_SECONDS", (0,)):
            self.assertEqual(await self._run(model, terminal, [
                "--resume=False", "--prompt", "original", "--enable-default-tools=False",
                "--endpoint-rate-limit-retries", "unbounded",
            ]), 0)
        error = "[cli] ModelRateLimitError: Chat Completions HTTP 429: slow down"
        shown = [("failure" if text.startswith("[model failure]") else text)
                 for text in (item.text for item in terminal.items)
                 if text.startswith(("[model failure]", "[cli] Sampling")) or text == error]
        self.assertEqual(shown, [
            "failure", "[cli] Sampling again in 0 s (1/∞).",
            "failure", "[cli] Sampling again in 0 s (2/∞).",
            "failure", error, "[cli] Sampling again in 0 s (3/∞).",
            "failure", error, "[cli] Sampling again in 0 s (4/∞).",
        ])

    async def test_a_bounded_retry_shows_its_errors_as_an_unbounded_one_does(self):
        # Its display is the unbounded one's, cut short where its attempts end,
        # where the failure shows the last error.
        model = _Model(self.path, *(_rate_limited() for _ in range(5)))
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "failed" else None)
        with mock.patch.object(kernel, "RATE_LIMIT_DELAYS_SECONDS", (0,)):
            self.assertEqual(await self._run(model, terminal, [
                "--resume=False", "--prompt", "original", "--enable-default-tools=False",
                "--endpoint-rate-limit-retries", "4",
            ]), 1)
        error = "[cli] ModelRateLimitError: Chat Completions HTTP 429: slow down"
        shown = [("failure" if text.startswith("[model failure]") else text)
                 for text in (item.text for item in terminal.items)
                 if text.startswith(("[model failure]", "[cli] Sampling")) or text == error]
        self.assertEqual(shown, [
            "failure", "[cli] Sampling again in 0 s (1/5).",
            "failure", "[cli] Sampling again in 0 s (2/5).",
            "failure", error, "[cli] Sampling again in 0 s (3/5).",
            "failure", error, "[cli] Sampling again in 0 s (4/5).",
            "failure", error, "[cli] Sampling failed 5 times. Use /retry to try again.",
        ])

    async def test_cancel_during_the_wait_ends_the_turn_with_retry_offered(self):
        model = _Model(self.path, _rate_limited(), _answer("never"))
        cancelled = False

        def frame(terminal, editor, status):
            nonlocal cancelled
            if status.startswith("retrying") and not cancelled:
                cancelled = True
                terminal.submit("/cancel")  # during a 30 s wait
            elif status == "failed":
                terminal.key("c-d")
        terminal = _Terminal(frame)
        with mock.patch.object(kernel, "RATE_LIMIT_DELAYS_SECONDS", (30,)):
            self.assertEqual(await self._run(model, terminal, [
                "--resume=False", "--prompt", "original", "--enable-default-tools=False",
            ]), 1)
        self.assertEqual(len(model.calls), 1)
        shown = [item.text for item in terminal.items]
        self.assertIn("[cli] Sampling again in 30 s (1/8).", shown)
        cancel_hint = "[cli] Automatic retry cancelled. Use /retry to try again."
        self.assertIn(cancel_hint, shown)
        self.assertNotIn(HINT, shown)
        # Its record showed when it was typed, and is logged where the turn ended.
        record = next(n for n, text in enumerate(shown) if text.startswith("[user-tool-call] cancel"))
        self.assertLess(record, shown.index(cancel_hint))
        saved = load_interaction_save(self.path)
        names = [type(item).__name__ for item in saved]
        self.assertLess(names.index("ModelFailure"), names.index("UserToolCall"))
        [result] = [item for item in saved if isinstance(item, UserToolResult)]
        self.assertEqual(result.result.output, "Stopped the automatic retries.")

    async def test_cancel_during_a_retried_attempt_lets_it_finish_but_nothing_follows(self):
        go = threading.Event()

        def second_attempt(context):
            self.assertTrue(go.wait(3))
            raise _rate_limited()
        model = _Model(self.path, _rate_limited(), second_attempt, _answer("never"))
        cancelled = False
        cancelled_during = None

        def frame(terminal, editor, status):
            nonlocal cancelled, cancelled_during
            if len(model.calls) == 2 and status.startswith("sampling") and not cancelled:
                cancelled = True
                cancelled_during = status
                terminal.submit("/cancel")
            elif cancelled and not go.is_set() and any(
                    item.text.startswith("[user-tool-call] cancel") for item in terminal.items):
                go.set()  # the attempt in flight goes on once /cancel has taken effect
            elif cancelled and status in ("failed", "idle"):
                terminal.key("c-d")
        terminal = _Terminal(frame)
        with mock.patch.object(kernel, "RATE_LIMIT_DELAYS_SECONDS", (0,)):
            self.assertEqual(await self._run(model, terminal, [
                "--resume=False", "--prompt", "original", "--enable-default-tools=False",
            ]), 1)
        self.assertEqual(len(model.calls), 2)
        # The retried attempt's status keeps the retry's count.
        self.assertTrue(cancelled_during.endswith(" (1/8)"), cancelled_during)
        shown = [item.text for item in terminal.items]
        self.assertEqual(shown.count("[cli] Automatic retry cancelled. Use /retry to try again."), 1)
        saved = load_interaction_save(self.path)
        self.assertEqual(sum(isinstance(item, ModelFailure) for item in saved), 2)

    async def test_a_provider_s_long_retry_after_is_not_followed_yet(self):
        # The turn loop's own schedule sets the wait, however long it asks.
        model = _Model(self.path, _rate_limited(retry_after=900.5), _answer("recovered"))
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        with mock.patch.object(kernel, "RATE_LIMIT_DELAYS_SECONDS", (0,)):
            self.assertEqual(await self._run(model, terminal, [
                "--resume=False", "--prompt", "original", "--enable-default-tools=False",
            ]), 0)
        self.assertEqual(len(model.calls), 2)
        shown = [item.text for item in terminal.items]
        self.assertIn("[cli] Sampling again in 0 s (1/8).", shown)
        self.assertFalse(any("asks to wait" in text for text in shown))

    async def test_a_transient_http_failure_is_sampled_again_by_the_turn_loop(self):
        # With a real adapter: the turn's own requests get no HTTP retries of
        # their own, so each failed request shows, and the loop retries it.
        opener = _ScriptedOpener(
            _FakeHTTPResponse({"error": {"message": "busy"}}, status=503),
            _FakeHTTPResponse({"choices": [{
                "message": {"role": "assistant", "content": "recovered"},
                "finish_reason": "stop",
            }]}),
        )
        model = _Adapter(self.path, ChatCompletionsModel(
            chat_endpoint(api_url="https://api.example.test"), opener=opener))
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        with mock.patch.object(kernel, "DEFAULT_AUTO_RETRY",
                               replace(kernel.DEFAULT_AUTO_RETRY, delays=(0,))):
            self.assertEqual(await self._run(model, terminal, [
                "--resume=False", "--prompt", "original", "--enable-default-tools=False",
            ]), 0)
        self.assertEqual(len(opener.calls), 2)
        shown = [item.text for item in terminal.items]
        [failure] = [text for text in shown if text.startswith("[model failure]")]
        self.assertTrue(failure.startswith(
            "[model failure] Chat Completions HTTP 503: request failed kind=http_error"), failure)
        self.assertNotIn("attempts=", failure)  # one request: shown past 1 only
        self.assertIn("[cli] Sampling again in 0 s (1/3).", shown)
        [sample] = [text for text in shown if text.startswith("[sample]")]
        self.assertTrue(sample.endswith(" recovery=http_503_retry"), sample)
        saved = load_interaction_save(self.path)
        self.assertEqual([item.recovery for item in saved if isinstance(item, SampleMetadata)],
                         [("http_503_retry",)])


class RetryControllerTests(_ControllerTestCase):
    async def test_errors_are_preserved_then_hint_and_retry_does_not_add_user_input(self):
        for outcome, diagnostic in (
            (ModelTimeoutError("timed out"), "ModelTimeoutError: timed out"),
            (RuntimeError("adapter failed"), "RuntimeError: adapter failed"),
            (_answer(" "), "MissingFinalText: Model returned no final assistant text."),
        ):
            with self.subTest(diagnostic=diagnostic):
                model = _Model(self.path, outcome, _answer("recovered"))
                terminal = _Terminal(retry_then_quit())
                with mock.patch.object(cli, "build_model") as build:
                    self.assertEqual(await self._run(model, terminal, [
                        "--resume=False", "--prompt", "original", "--enable-default-tools=False",
                    ]), 1)  # Existing sticky failure exit status is unchanged.
                build.assert_not_called()
                self.assertEqual(len(model.calls), 2)
                self.assertTrue(all(tools == () for _, tools, _ in model.calls))
                saved = load_interaction_save(self.path)
                self.assertEqual([i for i in saved if isinstance(i, Message) and i.role == "user"],
                                 [Message("user", "original")])
                self.assertEqual(sum(isinstance(i, UserInteractionBoundary) for i in saved), 1)
                # /retry's record, but no user input.
                self.assertEqual([i.call.name for i in saved if isinstance(i, UserToolCall)],
                                 ["retry"])
                shown = [item.text for item in terminal.items]
                self.assertEqual(shown.count(HINT), 1)
                self.assertLess(shown.index("[cli] " + diagnostic), shown.index(HINT))
                self.assertIsInstance(saved[-1], TurnSummary)

    async def test_repeated_failures_require_fresh_explicit_retries(self):
        model = _Model(self.path, ModelTimeoutError("first"), ModelTimeoutError("second"), _answer())
        terminal = _Terminal(retry_then_quit(2))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(sum(item.text == HINT for item in terminal.items), 2)

    async def test_timeout_hint_names_the_live_timeout_and_retry_uses_a_new_one(self):
        model = _Model(self.path, ModelTimeoutError("timed out"), _answer("recovered"))
        commands = deque(("/config timeouts_request_seconds 2400", "/retry", "/quit"))
        terminal = _Terminal(lambda t, e, s: t.submit(commands.popleft())
                             if s in {"failed", "idle"} and commands else None)
        self.assertEqual(await self._run(model, terminal, [
            "--prompt", "hello", "--enable-default-tools=False",
        ]), 1)
        shown = [item.text for item in terminal.items]
        hint = TIMEOUT_HINT.format(DEFAULT_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(shown.count(hint), 1)
        self.assertLess(shown.index("[cli] ModelTimeoutError: timed out"), shown.index(HINT))
        self.assertLess(shown.index(HINT), shown.index(hint))
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0][2].request_timeout_seconds, DEFAULT_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(model.calls[1][2].request_timeout_seconds, 2400.0)
        self.assertIn("[assistant] recovered", shown)

    async def test_claude_relay_timeouts_get_no_request_timeout_hint(self):
        model = _Model(self.path, ModelTimeoutError("generation wait exceeded"))
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "failed" else None)
        self.assertEqual(await self._run(model, terminal, [
            "--prompt", "hello", "--enable-default-tools=False",
            "--endpoint-api", "claude-relay", "--model", "native-model",
        ]), 1)
        shown = [item.text for item in terminal.items]
        self.assertEqual(shown.count(HINT), 1)
        self.assertFalse(any("timeouts_request_seconds" in text for text in shown))
        self.assertIsNone(model.calls[0][2].request_timeout_seconds)

    async def test_retry_uses_a_fresh_budget_but_does_not_bypass_its_limit(self):
        model = _Model(self.path, ModelTimeoutError("first"),
                       ModelSample((ToolCall("noop", "new", "{}"),)))
        effects = []
        environment = Environment((Tool(ToolSpec("noop", "", {}),
            lambda args, **kwargs: (effects.append(True) or ToolOutcome("ok"))),))
        commands = deque(("/retry", "/retry", "/quit"))
        terminal = _Terminal(lambda t, e, s: t.submit(commands.popleft())
                             if s == "failed" and commands else None)
        self.assertEqual(await self._run(model, terminal, [
            "--prompt", "hello", "--max-samples", "1",
        ], environment), 1)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(effects, [True])
        shown = [i.text for i in terminal.items]
        self.assertEqual(shown.count(HINT), 1)
        self.assertTrue(any("within 1 samples" in text for text in shown))
        self.assertTrue(any("No retryable" in text for text in shown))

    async def test_partial_output_is_retained_old_calls_stay_closed_and_new_calls_execute(self):
        effects = []
        environment = Environment((Tool(ToolSpec("record", "", {}),
            lambda args, **kwargs: (effects.append(args["label"]) or ToolOutcome("done"))),))
        old = ToolCall("record", "old", '{"label":"old"}')
        partial = ToolCall("record", "partial", '{"label":"must not run"}')
        new = ToolCall("record", "new", '{"label":"new"}')
        failure = ModelFailure(category="stream_closed", message="stream interrupted")
        error = ModelResponseError("stream interrupted", failure=failure,
                                   completed_items=(Message("assistant", "partial output"), partial))
        model = _Model(self.path, ModelSample((old,)), error, ModelSample((new,)), _answer())
        self.assertEqual(await self._run(model, _Terminal(retry_then_quit()),
                                        ["--prompt", "hello"], environment), 1)
        self.assertEqual(effects, ["old", "new"])
        retry_context = model.calls[2][0]
        self.assertIn(failure, retry_context)
        self.assertIn(Message("assistant", "partial output"), retry_context)
        self.assertEqual([i.call_id for i in retry_context if isinstance(i, ToolResult)],
                         ["old", "partial"])
        self.assertFalse(next(i for i in retry_context if isinstance(i, ToolResult)
                              and i.call_id == "partial").success)
        self.assertFalse(retry_context.pending_tool_calls())

    async def test_ticket_survives_config_quota_login_and_manual_compaction(self):
        auth = self.path.parent / "auth.json"
        auth.write_text(json.dumps({"tokens": {"access_token": "FAKE_SECRET", "account_id": "same"}}))
        model = _Model(self.path, ModelTimeoutError("first"), _answer())
        model.endpoint = SimpleNamespace(account_id="same")
        commands = deque(("/config max_output_tokens 17", "/quota", "/login", "/compact", "/retry", "/quit"))
        terminal = _Terminal(lambda t, e, s: t.submit(commands.popleft())
                             if s in {"failed", "idle"} and commands else None)
        compactor = mock.Mock()
        compactor.compact.return_value = CompactionResult((ContextPrefix((Message("user", "summary"),)),))
        with mock.patch.object(user_tools, "query_quota", return_value="quota"), \
                mock.patch.object(user_tools, "login") as login, \
                mock.patch.object(cli, "build_model", return_value=model) as build, \
                patch_compaction(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self._run(model, terminal, [
                "--prompt", "hello", "--endpoint-api", "codex", "--model", "test",
                "--endpoint-auth-file", str(auth), "--enable-default-tools=False",
            ]), 1)
        login.assert_called_once()
        build.assert_called_once()  # /login activation only; retry reuses it.
        self.assertEqual(len(model.calls), 2)
        # The turn loop's own sample asks the model for no transient retries:
        # the loop retries those itself.
        self.assertEqual(model.calls[1][2], SampleParams(
            max_output_tokens=17, enable_auto_compaction=True,
            request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS, transient_retries=0,
        ))
        self.assertEqual(model.calls[1][1], ())
        self.assertIn(Message("user", "summary"), model.calls[1][0].model_items())
        # /quota and /login answer with notices; /retry is recorded before its sample.
        self.assertEqual([i.call.name for i in load_interaction_save(self.path) if isinstance(i, UserToolCall)],
                         ["config", "compact", "retry"])

    async def test_new_query_invalidates_queued_retry_and_success_disables_retry(self):
        step = 0

        def frame(t, e, s):
            nonlocal step
            if s == "failed" and step == 0:
                t.submit("new task")
                t.submit("/retry")
                step = 1
            elif s == "idle" and step == 1 and any("no longer applies" in i.text for i in t.items):
                t.submit("/retry")
                step = 2
            elif s == "idle" and step == 2 and any("No retryable" in i.text for i in t.items):
                t.key("c-d")

        model = _Model(self.path, ModelTimeoutError("first"), _answer())
        self.assertEqual(await self._run(model, _Terminal(frame), ["--prompt", "old task"]), 1)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual([i.content for i in model.calls[1][0] if isinstance(i, Message) and i.role == "user"],
                         ["old task", "new task"])

    async def test_resumed_failure_is_not_retryable_and_prompt_retry_is_literal(self):
        original = InteractionContext((Init("old"), Tools(), Message("user", "hello"),
            ModelFailure(category="timeout", message="old failure"), ModelSampleBoundary()))
        seed_cli_save(self.path, original)
        before = self.path.read_bytes()
        commands = deque(("/retry", "/quit"))
        terminal = _Terminal(lambda t, e, s: t.submit(commands.popleft()) if s == "idle" and commands else None)
        model = _Model(self.path)
        self.assertEqual(await self._run(model, terminal, ["--resume"]), 0)
        self.assertEqual(model.calls, [])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue(any("No retryable" in i.text for i in terminal.items))
        model = _Model(self.path, _answer())
        self.assertEqual(await self._run(model, _Terminal(retry_then_quit()), ["--prompt", "/retry"]), 0)
        self.assertIn(Message("user", "/retry"), model.calls[0][0])

    async def test_non_sampling_failures_do_not_offer_retry(self):
        for kind in ("tool", "compaction", "limit"):
            with self.subTest(kind=kind):
                sample = (ModelSample((OpaqueCompaction.from_messages("summary"),), stop_reason="compaction")
                          if kind == "limit" else ModelSample((ToolCall("record", "call", "{}"),)))
                model = _Model(self.path, sample)
                environment = Environment()
                submitted = False

                def frame(t, e, s):
                    nonlocal submitted
                    if s == "failed":
                        if not submitted:
                            submitted = True
                            t.submit("/retry")
                        else:
                            t.key("c-d")

                compactor = mock.Mock()
                compactor.compact.side_effect = ModelTimeoutError("compaction failed")
                terminal = _Terminal(frame)
                with mock.patch.object(environment, "execute_tool_calls", side_effect=RuntimeError("lost tool result")) as execute, \
                        patch_compaction(cli, "auto_compaction_due", return_value=kind == "compaction"), \
                        patch_compaction(cli, "create_default_compactor", return_value=compactor):
                    self.assertEqual(await self._run(model, terminal, [
                        "--prompt", "hello", "--max-samples", "1", "--auto-compaction-context-tokens", "100",
                    ], environment), 1)
                self.assertEqual(execute.call_count, int(kind == "tool"))
                self.assertEqual(len(model.calls), int(kind != "compaction"))
                self.assertNotIn(HINT, [i.text for i in terminal.items])
                self.assertTrue(any("No retryable" in i.text for i in terminal.items))

    async def test_checkpoint_failure_blocks_retry_even_during_failure_cleanup(self):
        failure = ModelFailure(category="timeout", message="timed out")
        model = _Model(self.path, ModelTimeoutError("timed out", failure=failure))
        submitted = False

        def frame(t, e, s):
            nonlocal submitted
            if s == "failed":
                if not submitted:
                    submitted = True
                    t.submit("/retry")
                else:
                    t.key("c-d")

        def save(writer, context):
            if failure in context:
                raise SaveError("disk failed")
            real_save(writer, context)

        terminal = _Terminal(frame)
        with patch_saves(save):
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        self.assertEqual(len(model.calls), 1)
        self.assertNotIn(HINT, [i.text for i in terminal.items])
        self.assertTrue(any("no further work" in i.text for i in terminal.items))
        self.assertNotIn(failure, load_interaction_save(self.path))

    async def test_queued_query_diagnostic_precedes_retry_hint(self):
        entered, release = threading.Event(), threading.Event()
        step = 0

        def fail(context):
            entered.set()
            if not release.wait(3):
                raise AssertionError("test did not release sample")
            raise ModelTimeoutError("timed out")

        def frame(t, e, s):
            nonlocal step
            if entered.is_set() and step == 0:
                t.submit("queued task")
                step = 1
            if "steers=1" in s:  # text typed mid-turn is a pending steer
                release.set()
            if s == "failed":
                t.key("c-d")

        terminal = _Terminal(frame)
        try:
            self.assertEqual(await self._run(_Model(self.path, fail), terminal, ["--prompt", "hello"]), 1)
        finally:
            release.set()
        shown = [i.text for i in terminal.items]
        self.assertLess(shown.index("[cli] ModelTimeoutError: timed out"), shown.index(HINT))
        self.assertLess(next(n for n, text in enumerate(shown) if "Queued queries discarded" in text), shown.index(HINT))


class RetryAuthenticationTests(_ControllerTestCase):
    def failed_model(self, *, source="codex_file", account="same", partial=()):
        failure = ModelFailure(category="authentication", message="credentials rejected", auth_source=source)
        model = _Model(self.path, ModelAuthenticationError(failure.message, failure=failure, completed_items=partial))
        model.endpoint = SimpleNamespace(account_id=account)
        return model

    async def test_auth_retry_reloads_without_login_and_preserves_account(self):
        for source in ("codex_file", "environment", "static"):
            with self.subTest(source=source):
                model = self.failed_model(source=source)
                replacement = _Model(self.path, _answer())
                replacement.endpoint = SimpleNamespace(account_id="same")
                terminal = _Terminal(retry_then_quit())
                with mock.patch.object(cli, "build_model", return_value=replacement) as build, \
                        mock.patch.object(user_tools, "login") as login:
                    self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
                build.assert_called_once()
                login.assert_not_called()
                self.assertEqual(len(replacement.calls), 1)
                self.assertEqual(replacement.calls[0][0].items, replacement.checkpoints[0])
                shown = [i.text for i in terminal.items]
                self.assertNotIn(HINT, shown)
                if source == "codex_file":
                    self.assertLess(shown.index("[cli] ModelAuthenticationError: credentials rejected"), shown.index(AUTH_HINT))
                else:
                    self.assertTrue(any("restart" in text for text in shown))

    async def test_reload_failure_keeps_retry_available_and_withholds_details(self):
        model = self.failed_model()
        replacement = _Model(self.path, _answer())
        replacement.endpoint = SimpleNamespace(account_id="same")
        commands = deque(("/retry", "/retry"))
        terminal = _Terminal(lambda t, e, s: t.submit(commands.popleft())
                             if s in {"failed", "auth needed"} and commands
                             else t.key("c-d") if s == "idle" else None)
        with mock.patch.object(cli, "build_model", side_effect=[ValueError("FAKE_SECRET"), replacement]) as build:
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        self.assertEqual(build.call_count, 2)
        self.assertEqual(len(replacement.calls), 1)
        self.assertNotIn("FAKE_SECRET", "\n".join(i.text for i in terminal.items))
        self.assertNotIn("FAKE_SECRET", self.path.read_text())

    async def test_login_can_recover_auth_but_does_not_automatically_retry(self):
        model = self.failed_model()
        replacement = _Model(self.path, _answer())
        replacement.endpoint = SimpleNamespace(account_id="same")
        commands = deque(("/login", "/retry", "/quit"))

        def frame(t, e, s):
            if s in {"failed", "idle"} and commands:
                command = commands.popleft()
                if command == "/retry":
                    self.assertEqual(replacement.calls, [])
                t.submit(command)

        with mock.patch.object(user_tools, "login") as login, \
                mock.patch.object(cli, "build_model", return_value=replacement) as build:
            self.assertEqual(await self._run(model, _Terminal(frame), [
                "--prompt", "hello", "--endpoint-api", "codex", "--model", "test",
                "--endpoint-auth-file", str(self.path.parent / "auth.json"),
            ]), 1)
        login.assert_called_once()
        build.assert_called_once()  # Retry reuses the model activated by /login.
        self.assertEqual(len(replacement.calls), 1)
        self.assertEqual([i for i in replacement.calls[0][0] if isinstance(i, Message) and i.role == "user"],
                         [Message("user", "hello")])

    async def test_reloading_a_different_or_unknown_account_does_not_sample(self):
        for account in ("same", None):
            with self.subTest(account=account):
                model = self.failed_model(account=account, partial=(Reasoning("", encrypted_content="opaque"),))
                replacement = _Model(self.path, _answer())
                replacement.endpoint = SimpleNamespace(account_id="different")
                submitted = False

                def frame(t, e, s):
                    nonlocal submitted
                    if s == "failed" and not submitted:
                        submitted = True
                        t.submit("/retry")
                    elif s == "auth needed":
                        t.key("c-d")

                with mock.patch.object(cli, "build_model", return_value=replacement) as build:
                    self.assertEqual(await self._run(model, _Terminal(frame), [
                        "--prompt", "hello", "--endpoint-api", "codex", "--model", "test",
                    ]), 1)
                self.assertEqual(build.call_count, int(account is not None))
                self.assertEqual(replacement.calls, [])

    async def test_exit_during_auth_reload_does_not_sample_the_replacement(self):
        model = self.failed_model()
        replacement = _Model(self.path, _answer())
        replacement.endpoint = SimpleNamespace(account_id="same")
        entered, release = threading.Event(), threading.Event()
        submitted = False

        def build(args):
            entered.set()
            if not release.wait(3):
                raise AssertionError("test did not release model reload")
            return replacement

        def frame(t, e, s):
            nonlocal submitted
            if s == "failed" and not submitted:
                submitted = True
                t.submit("/retry")
            if entered.is_set():
                t.key("c-c")
            if s.startswith("closing"):
                release.set()

        try:
            with mock.patch.object(cli, "build_model", side_effect=build):
                self.assertEqual(await self._run(model, _Terminal(frame), ["--prompt", "hello"]), 1)
        finally:
            release.set()
        self.assertTrue(entered.is_set())
        self.assertEqual(replacement.calls, [])


OVERFLOW = ModelFailure(category="context_window", message="Messages HTTP 400: context window exceeded")
OVERFLOW_NOTICE = "[cli] Model context window exceeded; compacting before one retry."


def overflow():
    return ModelContextWindowError("prompt is too long", failure=OVERFLOW)


def quit_when_settled(terminal, editor, status):
    if status in {"idle", "failed"}:
        terminal.key("c-d")


class OverflowRecoveryTests(_ControllerTestCase):
    def compactor(self, **kwargs):
        compactor = mock.Mock()
        compactor.compact.return_value = CompactionResult(
            (ContextPrefix((Message("user", "summary"),)),), protocol="pi",
        )
        for name, value in kwargs.items():
            setattr(compactor.compact, name, value)
        return compactor

    async def test_context_window_error_compacts_once_and_retries_the_sample(self):
        model = _Model(self.path, overflow(), _answer("recovered"))
        compactor = self.compactor()
        terminal = _Terminal(quit_when_settled)
        with patch_compaction(cli, "create_default_compactor", return_value=compactor) as create:
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 0)
        create.assert_called_once_with(model, CompactionSettings(mode="warmpi"),
                                       reason="overflow")
        compactor.compact.assert_called_once()
        # Compaction gets the turn's params, so the adapters' transient retries
        # stay on for it; only the turn loop's own samples turn them off.
        self.assertEqual(model.calls[0][2].transient_retries, 0)
        self.assertEqual(compactor.compact.call_args.kwargs["sample_params"],
                         replace(model.calls[0][2], transient_retries=None))
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[1][0].model_items(), (Message("user", "summary"),))
        saved = load_interaction_save(self.path)
        # The failure is recorded before the compaction that recovers it.
        failure_index = saved.items.index(OVERFLOW)
        self.assertIsInstance(saved.items[failure_index + 1], ModelSampleBoundary)
        self.assertIsInstance(saved.items[failure_index + 2], ContextPrefix)
        self.assertIsInstance(saved.items[failure_index + 3], CompactionMetadata)
        self.assertIsInstance(saved.items[-1], TurnSummary)
        texts = [item.text for item in terminal.items]
        self.assertIn(OVERFLOW_NOTICE, texts)
        self.assertNotIn(HINT, texts)

    async def test_second_overflow_in_a_turn_fails_and_arms_retry(self):
        model = _Model(self.path, overflow(), overflow(), _answer("after retry"))
        compactor = self.compactor()
        with patch_compaction(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self._run(model, _Terminal(retry_then_quit()), ["--prompt", "hello"]), 1)
        # One recovery per turn; the explicit /retry starts a new attempt.
        compactor.compact.assert_called_once()
        self.assertEqual(len(model.calls), 3)

    async def test_nothing_to_compact_keeps_the_sampling_error(self):
        model = _Model(self.path, overflow())
        compactor = self.compactor(side_effect=NothingToCompact(
            "the context fits in compaction_pi_keep_recent_tokens",
        ))
        terminal = _Terminal(quit_when_settled)
        with patch_compaction(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        self.assertEqual(len(model.calls), 1)
        texts = [item.text for item in terminal.items]
        self.assertIn("[cli] ModelContextWindowError: prompt is too long", texts)
        self.assertIn(HINT, texts)
        self.assertFalse(any(isinstance(item, ContextPrefix) for item in load_interaction_save(self.path)))

    async def test_failed_recovery_compaction_fails_the_turn_with_retry(self):
        message = (
            "summary request for the history (412 items, ~905,000 estimated tokens) "
            "exceeded the model's context window"
        )
        model = _Model(self.path, overflow(), _answer("after retry"))
        compactor = self.compactor(side_effect=CompactionContextWindowError(message))
        terminal = _Terminal(retry_then_quit())
        with patch_compaction(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        texts = [item.text for item in terminal.items]
        self.assertIn(f"[cli] CompactionContextWindowError: {message}", texts)
        self.assertIn(HINT, texts)
        self.assertEqual(len(model.calls), 2)

    async def test_no_recovery_without_host_owned_automatic_compaction(self):
        for argv, owner in (
            (["--enable-auto-compaction=False"], "host"),
            ([], "server"),
        ):
            with self.subTest(argv=argv, owner=owner):
                model = _Model(self.path, overflow())
                model.auto_compaction_owner = owner
                compactor = self.compactor()
                terminal = _Terminal(quit_when_settled)
                with patch_compaction(cli, "create_default_compactor", return_value=compactor) as create:
                    self.assertEqual(await self._run(model, terminal, ["--prompt", "hello", *argv]), 1)
                create.assert_not_called()
                self.assertIn(HINT, [item.text for item in terminal.items])

    async def test_threshold_nothing_to_compact_samples_normally(self):
        model = _Model(self.path, _answer("done"))
        compactor = self.compactor(side_effect=NothingToCompact("nothing precedes the recent tail"))
        with patch_compaction(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self._run(model, _Terminal(quit_when_settled), [
                "--prompt", "hello", "--auto-compaction-context-tokens", "1",
            ]), 0)
        compactor.compact.assert_called_once()
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(any(isinstance(item, CompactionMetadata) for item in load_interaction_save(self.path)))

    async def test_failed_threshold_compaction_fails_the_turn_without_retry(self):
        message = "summary request for the history (3 items, ~9 estimated tokens) exceeded the model's context window"
        model = _Model(self.path)
        compactor = self.compactor(side_effect=CompactionContextWindowError(message))
        terminal = _Terminal(quit_when_settled)
        with patch_compaction(cli, "create_default_compactor", return_value=compactor):
            self.assertEqual(await self._run(model, terminal, [
                "--prompt", "hello", "--auto-compaction-context-tokens", "1",
            ]), 1)
        self.assertEqual(model.calls, [])
        texts = [item.text for item in terminal.items]
        self.assertIn(f"[cli] CompactionContextWindowError: {message}", texts)
        self.assertNotIn(HINT, texts)


if __name__ == "__main__":
    unittest.main()
