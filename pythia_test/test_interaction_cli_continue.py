"""/continue in the CLI: sample a saved turn where it stopped, without a user message."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import Init, InteractionContext, Message, ModelAuthenticationError
from pythia.interaction import ModelFailure, ModelSample, ModelSampleBoundary, ModelTransportError
from pythia.interaction import ToolCall, ToolResult, Tools, TurnSummary, UserInteractionBoundary
from pythia.interaction import cli, load_interaction_save, save_interaction_save, user_tools
from pythia.interaction._cli_editor import Editor
from pythia_test.test_interaction_cli import _ControllerTestCase, _Model, _Terminal, _answer


def _submit(state, text):
    state.editor = Editor(text, len(text))
    state.handle_key("c-m", "\r")


def _shown(state):
    return "\n".join(item.text for item in state.displays)


def _steps(*steps, until):
    """Submit each text at an idle frame, once the previous step's text is shown."""
    remaining = list(steps)
    waiting = [None]

    def frame(terminal, editor, status):
        if status not in {"idle", "failed", "auth needed"}:
            return
        shown = [item.text for item in terminal.items]
        if waiting[0] is not None and not any(waiting[0] in text for text in shown):
            return
        if remaining:
            text, waiting[0] = remaining.pop(0)
            terminal.submit(text)
        elif any(until in text for text in shown):
            terminal.key("c-d")
    return frame


def _users(context):
    return [item.content for item in context if isinstance(item, Message) and item.role == "user"]


class ContinueInputTests(unittest.TestCase):
    def test_continue_is_typed_like_retry_and_not_auth_gated(self):
        state = cli._UIState(ready=True, phase="idle", auth_required=True)
        _submit(state, "/continue")
        _submit(state, "/continue")
        self.assertEqual(len(state.pending), 1)
        self.assertIsInstance(state.pending[0], cli._ContinueIntent)
        self.assertIn("A continuation is already queued.", _shown(state))
        retry = cli._UIState(ready=True, phase="failed", retry=cli._RetryIntent())
        _submit(retry, "/continue")
        _submit(retry, "/retry")
        self.assertEqual(len(retry.pending), 1)
        self.assertIn("A continuation is already queued.", _shown(retry))

    def test_continue_rejects_arguments_busy_and_a_queued_retry(self):
        cases = [("idle", "/continue FAKE_SECRET", "Usage: /continue"),
                 ("idle", "/continue\nFAKE_SECRET", "Usage: /continue"),
                 ("sampling", "/continue", "Cannot continue while work is in progress."),
                 ("tool: exec_command", "/continue", "work is in progress")]
        for phase, text, notice in cases:
            with self.subTest(phase=phase, text=text):
                state = cli._UIState(ready=True, phase=phase)
                _submit(state, text)
                self.assertFalse(state.pending)
                self.assertIn(notice, _shown(state))
                self.assertNotIn("FAKE_SECRET", _shown(state))
        state = cli._UIState(ready=True, phase="failed", retry=cli._RetryIntent())
        _submit(state, "/retry")
        _submit(state, "/continue")
        self.assertEqual(len(state.pending), 1)
        self.assertIn("A retry is already queued.", _shown(state))


class ContinueControllerTests(_ControllerTestCase):
    async def test_continue_after_a_restart_samples_the_saved_turn_once(self):
        original = (Init("old"), Tools(), Message("user", "old query"), UserInteractionBoundary(),
                    ToolCall("record", "one", "{}"), ModelSampleBoundary())
        save_interaction_save(self.path, InteractionContext(original))
        model = _Model(self.path, _answer("finished"))
        terminal = _Terminal(_steps(("/continue", None), until="[assistant] finished"))
        self.assertEqual(await self._run(model, terminal, ["--resume"]), 0)
        self.assertEqual(len(model.calls), 1)
        items = model.calls[0][0].items
        self.assertEqual(items, model.checkpoints[0])  # exactly the saved context
        self.assertEqual(items[:len(original)], original)
        (closing,) = items[len(original):]  # startup closed the call; nothing else
        self.assertIsInstance(closing, ToolResult)
        self.assertFalse(closing.success)
        saved = load_interaction_save(self.path)
        self.assertEqual(_users(saved), ["old query"])
        self.assertEqual(sum(isinstance(item, UserInteractionBoundary) for item in saved), 1)
        self.assertIsInstance(saved[-1], TurnSummary)
        (notice,) = [i.text for i in terminal.items
                     if i.text.startswith("[cli] Note: resumed save")]
        self.assertIn("ends with tool results", notice)
        self.assertIn("enter /continue to sample from here, or a query to continue with a "
                      "new message.", notice)

    async def test_continue_is_refused_where_the_next_step_is_not_a_sample(self):
        start = (Init("old"), Tools(), Message("user", "q"), UserInteractionBoundary())
        answered = (*start, Message("assistant", "answer"), ModelSampleBoundary())
        for saved, reason in (
                ((*answered, TurnSummary(sample_count=1)), "the last turn ended"),
                (answered, "the log ends with assistant text, which may be a final answer"),
                ((Init("old"), Tools()), "the last turn ended")):
            with self.subTest(reason=reason):
                save_interaction_save(self.path, InteractionContext(saved))
                expected = f"[cli] Nothing to continue: {reason}. Enter a query instead."
                model = _Model(self.path)
                terminal = _Terminal(_steps(("/continue", None), until=expected))
                self.assertEqual(await self._run(model, terminal, ["--resume"]), 0)
                self.assertEqual(model.calls, [])
                self.assertEqual(load_interaction_save(self.path).items, saved)
                self.assertFalse(any("/continue to sample" in i.text for i in terminal.items))

    async def test_continue_after_the_sample_limit_in_one_process(self):
        model = _Model(self.path, ModelSample((ToolCall("record", "one", "{}"),)),
                       _answer("finished"))
        terminal = _Terminal(_steps(("/continue", None), until="[assistant] finished"))
        self.assertEqual(await self._run(model, terminal, [
            "--prompt", "hello", "--max-samples", "1"]), 1)  # the earlier failure stays
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[1][0].items, model.checkpoints[1])
        self.assertIsInstance(model.calls[1][0].items[-1], ToolResult)
        saved = load_interaction_save(self.path)
        self.assertEqual(_users(saved), ["hello"])
        self.assertEqual(sum(isinstance(item, UserInteractionBoundary) for item in saved), 1)

    async def test_continue_consumes_a_retry_ticket(self):
        model = _Model(self.path, ModelTransportError("lost"), _answer("recovered"))
        terminal = _Terminal(_steps(("/continue", None), ("/retry", "[assistant] recovered"),
                                    until="No retryable sampling failure"))
        self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(_users(load_interaction_save(self.path)), ["hello"])

    async def test_continue_without_a_model_reloads_credentials(self):
        failure = ModelFailure(category="authentication", message="credentials rejected",
                               auth_source="codex_file")
        model = _Model(self.path, ModelAuthenticationError(failure.message, failure=failure))
        model.endpoint = SimpleNamespace(account_id="same")
        replacement = _Model(self.path, _answer("recovered"))
        replacement.endpoint = SimpleNamespace(account_id="same")
        terminal = _Terminal(_steps(("/continue", None), until="[assistant] recovered"))
        with mock.patch.object(cli, "build_model", return_value=replacement) as build, \
                mock.patch.object(user_tools, "login") as login:
            self.assertEqual(await self._run(model, terminal, ["--prompt", "hello"]), 1)
        build.assert_called_once()
        login.assert_not_called()
        self.assertEqual(len(replacement.calls), 1)
        self.assertEqual(replacement.calls[0][0].items, replacement.checkpoints[0])
        self.assertEqual(_users(load_interaction_save(self.path)), ["hello"])

    async def test_a_literal_prompt_continue_is_user_text(self):
        model = _Model(self.path, _answer("ok"))
        terminal = _Terminal(lambda t, e, s: t.key("c-d") if s == "idle" else None)
        self.assertEqual(await self._run(model, terminal, ["--prompt", "/continue"]), 0)
        self.assertEqual(_users(model.calls[0][0]), ["/continue"])


if __name__ == "__main__":
    unittest.main()
