"""Part 4 of the warm-cold compaction plan: warmpi, a cache-friendly pi.

warmpi asks for pi's summary with the conversation's own request: the request
the next sample would send, unchanged, plus one user message. That message
doesn't mention the kept tail, so the summary covers the whole conversation.
The checkpoint is pi's, with protocol warmpi. warmpi sends pi's request
instead in overflow recovery, and when its own request is too large or its
signed thinking is rejected.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import queue
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import (
    COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX, ChatCompletionsModel,
    CompactionContextWindowError, CompactionError, CompactionMetadata, CompactionResult,
    CompactionSettings, ContextPrefix, Environment, Init, Instructions, InteractionConfig,
    InteractionConfigSnapshot, InteractionContext, Message, MessagesModel,
    ModelContextWindowError, ModelRefusal, ModelSample, ModelSampleBoundary,
    ModelThinkingBindingError, ModelTransportError, OpaqueCompaction, PiCompactor, Reasoning,
    ResponsesModel, SampleMetadata, SampleParams, TokenUsage, ToolCall, ToolResult, ToolSpec,
    UserInteractionBoundary, WarmPiCompactor, cli, create_default_compactor,
    load_interaction_save, resume_interaction_save, run_compaction, save_interaction_save,
)
from pythia.interaction.compaction import (
    SUMMARIZATION_PROMPT, SUMMARIZATION_SYSTEM_PROMPT, UPDATE_SUMMARIZATION_PROMPT,
    WARMPI_INSTRUCTION,
)
from pythia.interaction.loop import kernel
from pythia.interaction.messages import MessagesPromptCaching, _messages_http_failure
from pythia_test.interaction_helpers import (
    FakeRelayRuntime, ScriptedGateway, answer, chat_answer, chat_endpoint, codex_model,
    messages_answer, messages_endpoint, relay_model, responses_answer, responses_endpoint,
)
from pythia_test.test_compaction_sequences import _Host, _Script, _exec_tool, _history, _result


EXEC = ToolSpec("exec_command", "Run a command.", {"type": "object"})
TOOLS = (EXEC,)
SUMMARY = "## Goal\nFix the bug."
# The instruction: pi's format, or pi's update rules after an earlier summary.
NEW = (f"{WARMPI_INSTRUCTION}\n\n"
       + SUMMARIZATION_PROMPT[SUMMARIZATION_PROMPT.index("Use this EXACT format:"):])
UPDATE = (f"{WARMPI_INSTRUCTION}\n\nAn earlier part of this conversation was already "
          "summarized. Update that summary with the messages after it. "
          + UPDATE_SUMMARIZATION_PROMPT[UPDATE_SUMMARIZATION_PROMPT.index("RULES:"):])
TURN_PARAMS = SampleParams(max_output_tokens=128_000, enable_auto_compaction=True,
                           auto_compaction_context_tokens=900, request_timeout_seconds=600.0)


def _turn(*, thinking=True):
    """A turn in progress that ends with tool results, as automatic compactions do."""
    def think(text, signature):
        return (Reasoning(text, content_signature=signature),) if thinking else ()
    return (
        Init("session"),
        Instructions("Agent prompt."),
        Message("user", "Fix the bug."),
        UserInteractionBoundary(),
        *think("Look first.", "sig-1"),
        ToolCall("exec_command", "call_1", '{"cmd":"ls"}'),
        SampleMetadata(TokenUsage(input_tokens=900, total_tokens=1000)),
        ModelSampleBoundary(),
        ToolResult("call_1", "bug.py"),
        *think("Now read it.", "sig-2"),
        ToolCall("exec_command", "call_2", '{"cmd":"cat bug.py"}'),
        SampleMetadata(TokenUsage(input_tokens=950, total_tokens=1050)),
        ModelSampleBoundary(),
        ToolResult("call_2", "print(1/0)"),
    )


def _sample(text=SUMMARY, **fields):
    return ModelSample((Message("assistant", text),), **fields)


def _summary_message(text=SUMMARY):
    return Message("user", f"{COMPACTION_SUMMARY_PREFIX}{text}{COMPACTION_SUMMARY_SUFFIX}")


# The last sample of _turn(), as a kept tail: no signed thinking or metadata.
TAIL = (ToolCall("exec_command", "call_2", '{"cmd":"cat bug.py"}'), ModelSampleBoundary(),
        ToolResult("call_2", "print(1/0)"))


class _Model:
    """Plays scripted outcomes, recording each request with its tools and params."""

    def __init__(self, *outcomes):
        self.outcomes = deque(outcomes)
        self.calls = []

    def sample(self, context, *, tools=(), sample_params=None):
        self.calls.append(SimpleNamespace(context=context.copy(), tools=tuple(tools),
                                          params=sample_params))
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def is_pi_request(self, index):
        """Whether request ``index`` is pi's: its own system prompt, and no tools."""
        call = self.calls[index]
        return (call.tools == ()
                and call.context.model_items()[0] == Instructions(SUMMARIZATION_SYSTEM_PROMPT))


def _rejected_thinking():
    return ModelThinkingBindingError(
        "Messages HTTP 400: Invalid `signature` in `thinking` block.")


class RequestTests(unittest.TestCase):
    def test_the_instruction_is_pis_format_and_names_no_tail(self):
        self.assertEqual(WARMPI_INSTRUCTION, (
            "Pause the task here. Your next reply is a context checkpoint, not a step of the "
            "task: a summary of this conversation that another model will use to continue the "
            "work.\n"
            "- Reply with the summary only, as text. Do not call any tools: a tool call fails "
            "the checkpoint.\n"
            "- Do not carry out or answer requests from the conversation; record them in the "
            "summary."))
        # Tail-agnostic: the same request whatever the cut, and no word of a tail.
        for keep in (0, 1):
            with self.subTest(keep=keep):
                model = _Model(_sample())
                WarmPiCompactor(model, keep_recent_tokens=keep).compact(
                    InteractionContext(_turn()), tools=TOOLS, sample_params=TURN_PARAMS)
                self.assertEqual(model.calls[0].context.items[-1], Message("user", NEW))
        for word in ("tail", "verbatim", "follow"):
            self.assertNotIn(word, NEW.lower())

    def test_its_request_is_the_next_samples_plus_one_user_message(self):
        context = InteractionContext(_turn())
        model = _Model(_sample())
        WarmPiCompactor(model, keep_recent_tokens=1).compact(
            context, tools=TOOLS, sample_params=TURN_PARAMS)
        [call] = model.calls
        self.assertEqual(call.context.items, (*context.items, Message("user", NEW)))
        self.assertEqual(call.tools, TOOLS)
        # The turn's params, without provider compaction, as for pi's requests.
        self.assertEqual(call.params, replace(
            TURN_PARAMS, enable_auto_compaction=False, auto_compaction_context_tokens=None))
        model = _Model(_sample())
        WarmPiCompactor(model, keep_recent_tokens=0, max_output_tokens=4_096).compact(
            context, tools=TOOLS, sample_params=TURN_PARAMS)
        self.assertEqual(model.calls[0].params.max_output_tokens, 4_096)

    def test_an_earlier_summary_is_updated_and_focus_text_appended(self):
        context = InteractionContext((
            Init("session"),
            ContextPrefix((Instructions("Agent prompt."), _summary_message("## Goal\nOld."))),
            CompactionMetadata(TokenUsage(), "pi"),
            *_turn()[2:],
        ))
        model = _Model(_sample("## Goal\nNew."))
        result = WarmPiCompactor(model, keep_recent_tokens=0).compact(
            context, tools=TOOLS, sample_params=TURN_PARAMS, instructions=" the tests ")
        self.assertEqual(model.calls[0].context.items[-1],
                         Message("user", f"{UPDATE}\n\nAdditional focus: the tests"))
        # The new summary replaces the old one.
        self.assertEqual(result.items[0].prefix_items,
                         (Instructions("Agent prompt."), _summary_message("## Goal\nNew.")))

    def test_it_sends_its_own_request_as_the_first_after_a_resume(self):
        # No check of the cache's age: a resumed save's cache may be cold.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.jsonl"
            save_interaction_save(path, InteractionContext(_turn()))
            context, _writer, _incomplete = resume_interaction_save(path)
        model = _Model(_sample())
        result = WarmPiCompactor(model, keep_recent_tokens=0).compact(context, tools=TOOLS)
        self.assertEqual((result.protocol, model.calls[0].context.items[-1]),
                         ("warmpi", Message("user", NEW)))


class CheckpointTests(unittest.TestCase):
    def test_the_checkpoint_is_pis_with_protocol_warmpi(self):
        context = InteractionContext(_turn())
        usage = TokenUsage(input_tokens=1000, output_tokens=50, total_tokens=1050,
                           cached_input_tokens=990)
        model = _Model(_sample(usage=usage, request_attempts=2, recovery=("credential_reload",),
                               provider_turn_id="msg_1"))
        result = WarmPiCompactor(model, keep_recent_tokens=1).compact(
            context, tools=TOOLS, sample_params=TURN_PARAMS)
        self.assertEqual(result.protocol, "warmpi")
        self.assertEqual((result.usage, result.request_attempts, result.recovery,
                          result.provider_turn_id), (usage, 2, ("credential_reload",), "msg_1"))
        self.assertEqual(result.items[0].prefix_items,
                         (Instructions("Agent prompt."), _summary_message(), *TAIL))
        self.assertEqual(result.context_items()[-1].protocol, "warmpi")
        # pi cuts in the same place.
        pi = PiCompactor(_Model(_sample()), keep_recent_tokens=1).compact(
            context, sample_params=TURN_PARAMS)
        self.assertEqual(pi.items[0].prefix_items[2:], TAIL)

    def test_pi_and_warmpi_update_each_others_summaries(self):
        for first, then in ((PiCompactor, WarmPiCompactor), (WarmPiCompactor, PiCompactor)):
            with self.subTest(first=first.__name__):
                context = InteractionContext(_turn())
                earlier = first(_Model(_sample("## Goal\nOld.")), keep_recent_tokens=0).compact(
                    context, tools=TOOLS)
                context.extend((*earlier.context_items(), Message("user", "Now the tests."),
                                UserInteractionBoundary()))
                model = _Model(_sample("## Goal\nNew."))
                result = then(model, keep_recent_tokens=0).compact(context, tools=TOOLS)
                prompt = model.calls[0].context.model_items()[-1].content
                if then is WarmPiCompactor:
                    self.assertEqual(prompt, UPDATE)
                else:
                    self.assertIn("<previous-summary>\n## Goal\nOld.\n</previous-summary>", prompt)
                self.assertEqual(result.items[0].prefix_items[-1], _summary_message("## Goal\nNew."))

    def test_replies_that_arent_a_summary_fail_the_attempt(self):
        call = ToolCall("exec_command", "call_3", "{}")
        for label, sample, message in (
            ("tool calls", ModelSample((Message("assistant", SUMMARY), call)),
             "returned tool calls"),
            ("no text", ModelSample((Message("assistant", " "),)), "returned no text"),
            ("stopped early", _sample(stop_reason="max_tokens"), "stopped early"),
            ("refused after text", _sample(stop_reason="refusal"), "stopped early"),
            ("not pi's format", _sample("I'll run the tests next."),
             re.escape("doesn't start with \"## Goal\"")),
        ):
            with self.subTest(label), self.assertRaisesRegex(CompactionError, message):
                WarmPiCompactor(_Model(sample), keep_recent_tokens=0).compact(
                    InteractionContext(_turn()), tools=TOOLS)


class PiRequestInsteadTests(unittest.TestCase):
    def test_overflow_recovery_sends_exactly_pis_request(self):
        context = InteractionContext(_turn())
        warm = _Model(_sample(), _sample())
        result = WarmPiCompactor(warm, keep_recent_tokens=1, reason="overflow").compact(
            context, tools=TOOLS, sample_params=TURN_PARAMS)
        pi = _Model(_sample(), _sample())
        expected = PiCompactor(pi, keep_recent_tokens=1).compact(
            context, tools=TOOLS, sample_params=TURN_PARAMS)
        self.assertEqual([(c.context.items, c.tools, c.params) for c in warm.calls],
                         [(c.context.items, c.tools, c.params) for c in pi.calls])
        self.assertEqual((result.protocol, result.recovery, result.items),
                         ("pi", (), expected.items))

    def test_too_large_or_rejected_thinking_sends_pis_request_in_the_same_attempt(self):
        spent = TokenUsage(input_tokens=10, output_tokens=2, total_tokens=12)
        for label, outcome, recovery in (
            ("context window", ModelContextWindowError("too long"), "warmpi_too_large"),
            ("stopped at the window",
             _sample(stop_reason="model_context_window_exceeded", usage=spent),
             "warmpi_too_large"),
            ("thinking rejected", _rejected_thinking(), "warmpi_thinking_rejected"),
        ):
            with self.subTest(label):
                pis = TokenUsage(input_tokens=5, output_tokens=1, total_tokens=6)
                model = _Model(outcome, _sample("## Goal\nFrom pi.", usage=pis))
                result = WarmPiCompactor(model, keep_recent_tokens=1).compact(
                    InteractionContext(_turn()), tools=TOOLS, sample_params=TURN_PARAMS)
                self.assertFalse(model.is_pi_request(0))
                self.assertTrue(model.is_pi_request(1))
                self.assertEqual((result.protocol, result.recovery), ("pi", (recovery,)))
                self.assertIn("From pi.", result.items[0].prefix_items[1].content)
                self.assertEqual(result.request_attempts, 2)
                self.assertEqual(result.usage.input_tokens,
                                 15 if isinstance(outcome, ModelSample) else 5)
        # pi's request too large as well: the attempt fails as too large.
        model = _Model(ModelContextWindowError("too long"), ModelContextWindowError("too long"))
        with self.assertRaises(CompactionContextWindowError):
            WarmPiCompactor(model, keep_recent_tokens=1).compact(
                InteractionContext(_turn()), tools=TOOLS)

    def test_other_failures_are_the_attempts_own(self):
        # Not about the request's size or thinking: the sequence's attempts decide.
        for error in (ModelRefusal({"type": "refusal", "category": "reasoning_extraction"}),
                      ModelTransportError("timed out")):
            with self.subTest(type(error).__name__):
                model = _Model(error, _sample())
                with self.assertRaises(type(error)):
                    WarmPiCompactor(model, keep_recent_tokens=1).compact(
                        InteractionContext(_turn()), tools=TOOLS)
                self.assertEqual(len(model.calls), 1)


class SequenceTests(unittest.TestCase):
    def test_the_factory_builds_warmpi_with_the_compactions_reason(self):
        settings = CompactionSettings(mode="warmpi", keep_recent_tokens=7, max_output_tokens=9)
        compactor = create_default_compactor(_Model(), settings, reason="manual")
        self.assertIsInstance(compactor, WarmPiCompactor)
        self.assertEqual((compactor.keep_recent_tokens, compactor.max_output_tokens,
                          compactor.reason), (7, 9, "manual"))
        self.assertEqual(create_default_compactor(_Model(), settings).reason, "threshold")
        with self.assertRaises(ValueError):
            create_default_compactor(_Model(), settings, reason="later")
        # The runner passes its reason to warmpi only.
        script = _Script(CompactionError("no"), _result())
        run_compaction(InteractionContext(_history()), _Model(),
                       CompactionSettings(mode="warmpi,pi"), factory=script, tools=TOOLS,
                       reason="overflow")
        self.assertEqual([options for _mode, options, _kwargs in script.calls],
                         [{"reason": "overflow"}, {}])

    def test_a_refused_request_is_retried_while_attempts_last(self):
        refusal = ModelRefusal({"type": "refusal", "category": "reasoning_extraction"})
        model = _Model(refusal, _sample())
        reports = []
        result = run_compaction(
            InteractionContext(_turn()), model, CompactionSettings(mode="warmpi:2",
                                                                   keep_recent_tokens=1),
            tools=TOOLS, sample_params=TURN_PARAMS, on_failure=reports.append)
        self.assertEqual((result.protocol, result.recovery, len(model.calls)),
                         ("warmpi", ("warmpi_failed",), 2))
        self.assertTrue(reports[0].retry)

    def test_focus_text_reaches_warmpi_on_codex(self):
        # Remote compaction takes no focus text; warmpi, in the default, does.
        codex = _Model(_sample())
        codex.supports_remote_compaction = True
        result = run_compaction(
            InteractionContext(_turn()), codex,
            CompactionSettings(mode="coldstart,warmpi:2", keep_recent_tokens=1),
            tools=TOOLS, sample_params=TURN_PARAMS, instructions="the bug")
        # coldstart is skipped, since there is no save.
        self.assertEqual(result.protocol, "warmpi")
        self.assertTrue(codex.calls[0].context.items[-1].content.endswith(
            "\n\nAdditional focus: the bug"))


class TurnLoopTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "run.jsonl"

    def test_the_default_falls_back_to_warmpi_when_coldstart_cant_use_the_save(self):
        context = InteractionContext(_history())
        host = _Host(self.path, context)
        missing = self.path.with_name("moved.jsonl")
        host.log_path = lambda: missing
        model = _Model(_sample(), _sample("done"))
        environment = Environment((_exec_tool(),))
        result = kernel.run_turn(context, model, environment, InteractionConfigSnapshot(
            auto_compaction_context_tokens=100, compaction_mode="coldstart,warmpi:2",
            compaction_pi_keep_recent_tokens=0), host)
        self.assertEqual(result.final_text, "done")
        warm, after = model.calls
        self.assertEqual((warm.tools, warm.context.items[-1]),
                         (environment.tool_specs, Message("user", NEW)))
        [metadata] = [item for item in load_interaction_save(self.path)
                      if isinstance(item, CompactionMetadata)]
        self.assertEqual((metadata.protocol, metadata.recovery), ("warmpi", ("coldstart_failed",)))
        # The next sample starts from warmpi's checkpoint.
        self.assertEqual(after.context.model_items()[1], _summary_message())

    def test_overflow_recovery_sends_pis_request(self):
        context = InteractionContext(_history())
        host = _Host(self.path, context)
        model = _Model(ModelContextWindowError("too long"), _sample(), _sample("fits"))
        result = kernel.run_turn(context, model, Environment((_exec_tool(),)),
                                 InteractionConfigSnapshot(compaction_mode="warmpi",
                                                           compaction_pi_keep_recent_tokens=0),
                                 host)
        self.assertEqual(result.final_text, "fits")
        self.assertTrue(model.is_pi_request(1))
        [metadata] = [item for item in load_interaction_save(self.path)
                      if isinstance(item, CompactionMetadata)]
        self.assertEqual(metadata.protocol, "pi")


class FrontendTests(unittest.TestCase):
    def test_compact_reports_a_warmpi_checkpoint(self):
        # A carried Responses checkpoint doesn't make it read as remote.
        remote = OpaqueCompaction("encrypted", protocol="responses")
        result = CompactionResult((ContextPrefix((remote, _summary_message())),),
                                  protocol="warmpi")
        self.assertEqual(cli._compaction_success_output(result),
                         "Context compacted using a warmpi summary checkpoint.")


def _sse(text):
    return answer(responses_answer(text), content_type="text/event-stream")


# (name, build(url), a step that answers text)
_WIRES = (
    ("messages", lambda url: MessagesModel(messages_endpoint(
        url, "claude-opus-5.5-max", api_key="test-key", prompt_caching=MessagesPromptCaching())),
     lambda text: answer(messages_answer(text))),
    ("responses", lambda url: ResponsesModel(responses_endpoint(url, "model")), _sse),
    ("codex", lambda url: codex_model(responses_endpoint(url, "model", bearer_token="FAKE",
                                                         api_provider="codex")), _sse),
    ("chat-completions", lambda url: ChatCompletionsModel(chat_endpoint(api_url=url)),
     lambda text: answer(chat_answer(text))),
)


def _without(payload, key):
    return {name: value for name, value in payload.items() if name != key}


class WireTests(unittest.TestCase):
    """Each HTTP adapter: the next sample's request, then warmpi's, through one server."""

    def requests(self, name, context):
        [(build, step)] = [(build, step) for wire, build, step in _WIRES if wire == name]
        gateway = ScriptedGateway(step("Next."), step(SUMMARY))
        self.addCleanup(gateway.close)
        model = build(gateway.url)
        self.addCleanup(model.close)
        params = InteractionConfig.from_model(model).snapshot().sample_params()
        model.sample(context, tools=TOOLS, sample_params=params)
        result = WarmPiCompactor(model, keep_recent_tokens=0).compact(
            context, tools=TOOLS, sample_params=params)
        self.assertEqual(result.protocol, "warmpi")
        return gateway.requests

    def test_messages_sends_the_next_samples_request_plus_one_text_block(self):
        sample, warm = self.requests("messages", InteractionContext(_turn()))

        def blocks(payload):
            return [block for message in payload["messages"] for block in message["content"]]

        def uncached(payload):
            payload = json.loads(json.dumps(payload))
            for block in blocks(payload):
                block.pop("cache_control", None)
            return payload

        # The same system prompt, tools, thinking, effort, and messages.
        self.assertEqual(_without(warm, "messages"), _without(sample, "messages"))
        for key in ("system", "tools", "thinking", "output_config"):
            self.assertIn(key, warm)
        self.assertNotIn("tool_choice", warm)
        self.assertIn('"signature": "sig-2"', json.dumps(warm["messages"]))
        before, after = uncached(sample)["messages"], uncached(warm)["messages"]
        self.assertEqual(after[:-1], before[:-1])
        # After tool results, the instruction shares their user message.
        self.assertEqual(after[-1], {**before[-1], "content": [
            *before[-1]["content"], {"type": "text", "text": NEW}]})
        # The cache breakpoint moves to it.
        marked = [block for block in blocks(warm) if "cache_control" in block]
        self.assertEqual(marked, [{"type": "text", "text": NEW,
                                   "cache_control": {"type": "ephemeral", "ttl": "5m"}}])

    def test_responses_and_codex_send_the_same_input_plus_one_message(self):
        for name in ("responses", "codex"):
            with self.subTest(name):
                sample, warm = self.requests(name, InteractionContext(_turn(thinking=False)))
                self.assertEqual(_without(warm, "input"), _without(sample, "input"))
                self.assertEqual(warm["input"][:-1], sample["input"])
                self.assertEqual(warm["input"][-1], {
                    "type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": NEW}]})
                self.assertIn("tools", warm)
        self.assertIn("prompt_cache_key", warm)  # Codex's

    def test_chat_completions_sends_the_same_messages_plus_one(self):
        sample, warm = self.requests("chat-completions", InteractionContext(_turn(thinking=False)))
        self.assertEqual(_without(warm, "messages"), _without(sample, "messages"))
        self.assertEqual(warm["messages"], [*sample["messages"], {"role": "user", "content": NEW}])


class _SummaryRuntime(FakeRelayRuntime):
    """A fake native run that answers every request with a summary."""

    def start(self, snapshot):
        self.snapshot = snapshot
        self.launches.all.append(self)
        self.answer(SUMMARY)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


class RelayTests(unittest.TestCase):
    def test_the_relay_gets_warmpis_own_request(self):
        launches = SimpleNamespace(summaries=queue.Queue(), all=[])
        runtime = type("Runtime", (_SummaryRuntime,), {"launches": launches})
        with mock.patch("pythia.interaction.claude_relay._model.Runtime", runtime):
            model = relay_model()
            self.addCleanup(model.close)
            result = WarmPiCompactor(model, keep_recent_tokens=0).compact(
                InteractionContext(_turn(thinking=False)), tools=TOOLS)
        self.assertEqual(result.protocol, "warmpi")
        [run] = launches.all
        # The conversation's own instructions and tools, then the instruction.
        self.assertEqual(run.snapshot.instructions, "Agent prompt.")
        self.assertIn("exec_command", run.snapshot.catalog)
        self.assertIn(NEW, list(_strings(json.loads(run.snapshot.records[-1]))))


class MessagesRejectionTests(unittest.TestCase):
    def test_a_rejected_thinking_signature_is_a_transport_error_of_its_own(self):
        detail = json.dumps({"type": "error", "error": {
            "type": "invalid_request_error",
            "message": ("messages.1.content.0: Invalid `signature` in `thinking` block. The "
                        "block is bound to a different conversation. The `tools` list differs "
                        "from the one this block was created with.")}})
        error = _messages_http_failure(400, detail, model="claude-opus-5-5", attempt_count=1,
                                       recovery=())
        self.assertIsInstance(error, ModelThinkingBindingError)
        # Everywhere but warmpi, it is handled as before.
        self.assertIsInstance(error, ModelTransportError)
        self.assertEqual((error.failure.category, error.failure.message), (
            "thinking_binding",
            "Messages HTTP 400: a signed thinking block is bound to a different conversation"))
        other = _messages_http_failure(400, '{"error": {"message": "bad"}}',
                                       model="claude-opus-5-5", attempt_count=1, recovery=())
        self.assertIs(type(other), ModelTransportError)


if __name__ == "__main__":
    unittest.main()
