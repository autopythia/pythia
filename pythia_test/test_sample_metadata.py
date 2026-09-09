from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pythia.interaction import ChatCompletionsEndpoint
from pythia.interaction import ChatCompletionsModel
from pythia.interaction import CodexResponsesModel
from pythia.interaction import ContextCompaction
from pythia.interaction import Message
from pythia.interaction import MessagesEndpoint
from pythia.interaction import MessagesModel
from pythia.interaction import ModelContext
from pythia.interaction import ModelSample
from pythia.interaction import ModelSampleBoundary
from pythia.interaction import ModelTimeoutError
from pythia.interaction import SampleMetadata
from pythia.interaction import SaveError
from pythia.interaction import StreamingResponsesEndpoint
from pythia.interaction import TokenUsage
from pythia.interaction import TurnSummary
from pythia.interaction import chat_completions
from pythia.interaction import interaction_item_from_dict
from pythia.interaction import interaction_item_to_dict
from pythia.interaction import load_interaction_save
from pythia.interaction import messages
from pythia.interaction import render_interaction_items
from pythia.interaction import responses
from pythia.interaction import save_interaction_save
from pythia.interaction import summarize_turn_usage


_USAGE = TokenUsage(20, 5, 25, 4)
_BASE_RECORD = {
    "usage": {
        "input_tokens": 20, "output_tokens": 5,
        "total_tokens": 25, "cached_input_tokens": 4,
    },
    "provider_session_id": "session-private",
    "provider_turn_id": "turn-private",
    "provider_turn_state": "state-private",
}
_BAD_TYPES = (True, False, "12.5", [], {})
_BAD_NUMBERS = (-1, -0.1, float("nan"), float("inf"), -float("inf"), 10**1000)


class SampleMetadataTests(unittest.TestCase):
    def test_elapsed_validation_and_unknown_default(self):
        for item_type, fields in (
            (SampleMetadata, {"usage": _USAGE}),
            (ModelSample, {"items": (Message("assistant", "Done."),)}),
        ):
            with self.subTest(item_type=item_type):
                self.assertIsNone(item_type(**fields).elapsed_seconds)
                for value in (None, 0, 0.0, 12, 12.3456789):
                    item = item_type(**fields, elapsed_seconds=value)
                    self.assertEqual(item.elapsed_seconds, value)
                    if value is not None:
                        self.assertIsInstance(item.elapsed_seconds, float)
                for value in _BAD_TYPES:
                    with self.assertRaisesRegex(TypeError, "elapsed_seconds"):
                        item_type(**fields, elapsed_seconds=value)
                for value in _BAD_NUMBERS:
                    with self.assertRaisesRegex(ValueError, "elapsed_seconds"):
                        item_type(**fields, elapsed_seconds=value)

    def test_elapsed_is_appended_after_existing_positional_fields(self):
        metadata = SampleMetadata(_USAGE, "session", "turn", "state")
        sample = ModelSample((Message("assistant", "Done."),), "end_turn",
                             _USAGE, "session", "turn", "state")
        self.assertIsNone(metadata.elapsed_seconds)
        self.assertEqual(sample.context_items()[-2], metadata)

    def test_codec_reads_legacy_and_canonical_types_and_writes_canonical(self):
        for kind in ("turn_metadata", "sample_metadata"):
            for timing in ({}, {"elapsed_seconds": None}, {"elapsed_seconds": 0},
                           {"elapsed_seconds": 12.3456789}):
                with self.subTest(kind=kind, timing=timing):
                    item = interaction_item_from_dict({"type": kind, **_BASE_RECORD, **timing})
                    self.assertIsInstance(item, SampleMetadata)
                    self.assertEqual(item.usage, _USAGE)
                    self.assertEqual(item.elapsed_seconds, timing.get("elapsed_seconds"))
                    canonical = {"type": "sample_metadata", **_BASE_RECORD}
                    if item.elapsed_seconds is not None:
                        canonical["elapsed_seconds"] = item.elapsed_seconds
                    self.assertEqual(interaction_item_to_dict(item), canonical)

    def test_codec_rejects_invalid_elapsed_values(self):
        for kind in ("turn_metadata", "sample_metadata"):
            for value in (*_BAD_TYPES, *_BAD_NUMBERS):
                with self.subTest(kind=kind, value=value):
                    with self.assertRaisesRegex(SaveError, kind + r"\.elapsed_seconds"):
                        interaction_item_from_dict({
                            "type": kind, **_BASE_RECORD, "elapsed_seconds": value,
                        })

    def test_mixed_legacy_and_nested_jsonl_normalizes_only_on_save(self):
        records = [
            {"type": "message", "role": "user", "text": "Old request."},
            {"type": "turn_metadata", **_BASE_RECORD},
            {"type": "sample_metadata", **_BASE_RECORD, "elapsed_seconds": 12.3456789},
            {
                "type": "context_compaction",
                "replacement_items": [
                    {"type": "message", "role": "user", "content": "Summary."},
                    {"type": "turn_metadata", **_BASE_RECORD},
                    {"type": "sample_metadata", **_BASE_RECORD, "elapsed_seconds": 0},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interaction.jsonl"
            original = "".join(json.dumps(record) + "\n" for record in records)
            path.write_text(original, encoding="utf-8")
            restored = load_interaction_save(path)
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertIsInstance(restored[1], SampleMetadata)
            self.assertIsNone(restored[1].elapsed_seconds)
            self.assertEqual(restored[2].elapsed_seconds, 12.3456789)
            checkpoint = restored[3]
            self.assertIsInstance(checkpoint, ContextCompaction)
            self.assertIsInstance(checkpoint.replacement_items[1], SampleMetadata)
            self.assertIsNone(checkpoint.replacement_items[1].elapsed_seconds)
            self.assertEqual(checkpoint.replacement_items[2].elapsed_seconds, 0.0)
            save_interaction_save(path, restored)
            encoded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for record in (encoded[1], encoded[2], *encoded[3]["replacement_items"][1:]):
                self.assertEqual(record["type"], "sample_metadata")
            self.assertNotIn("elapsed_seconds", encoded[1])
            self.assertEqual(load_interaction_save(path).items, restored.items)

    def test_live_and_replay_display_share_timing_and_hide_provider_fields(self):
        for elapsed, suffix in (
            (None, ""), (0, " elapsed=0.00s"), (12.3456789, " elapsed=12.35s"),
        ):
            with self.subTest(elapsed=elapsed), tempfile.TemporaryDirectory() as directory:
                sample = ModelSample(
                    items=(Message("assistant", "Done."),), usage=_USAGE,
                    provider_session_id="session-private", provider_turn_id="turn-private",
                    provider_turn_state="state-private", elapsed_seconds=elapsed,
                )
                with mock.patch("pythia.interaction.model.perf_counter",
                                side_effect=AssertionError("must not remeasure")):
                    metadata = sample.context_items()[-2]
                    self.assertEqual(metadata.elapsed_seconds, elapsed)
                    self.assertEqual(sample.context_items()[-1], ModelSampleBoundary())
                    context = ModelContext(sample.context_items())
                    path = Path(directory) / "interaction.jsonl"
                    save_interaction_save(path, context)
                    replay = render_interaction_items(load_interaction_save(path).items)
                    self.assertEqual(replay, sample.display_items())
                self.assertEqual(
                    replay[-1].text, "[sample] input=20 output=5 total=25 cached=4" + suffix,
                )
                for private in ("session-private", "turn-private", "state-private"):
                    self.assertNotIn(private, repr(sample))
                    self.assertNotIn(private, repr(metadata))
                    self.assertNotIn(private, "\n".join(item.text for item in replay))

    def test_turn_summary_keeps_usage_semantics_and_uses_short_label(self):
        items = tuple(SampleMetadata(_USAGE, elapsed_seconds=value) for value in (None, 1, 2))
        summary = summarize_turn_usage((*items, TurnSummary(sample_count=99)))
        self.assertEqual(summary, TurnSummary(
            input_tokens_sum=60, output_tokens_sum=15, cached_input_tokens_sum=12,
            cached_input_tokens_max=4, non_cached_input_tokens_sum=48,
            context_tokens=25, sample_count=3,
        ))
        self.assertEqual(render_interaction_items((summary,))[0].text,
                         "[turn] input_sum=60 output_sum=15 cached_sum=12 cached_max=4 "
                         "cold_sum=48 context=25 samples=3 compactions=0")
        self.assertEqual(interaction_item_to_dict(summary)["type"], "turn_summary")
        self.assertFalse(hasattr(summary, "elapsed_seconds"))

    def test_timed_metadata_remains_invisible_to_provider_content(self):
        items = (Message("user", "Question."), Message("assistant", "Answer."))
        metadata = SampleMetadata(_USAGE, elapsed_seconds=12.3456789)
        for encode in (chat_completions._encode_context_messages,
                       messages._encode_context, responses._encode_context_items):
            with self.subTest(encoder=encode.__module__):
                self.assertEqual(encode((*items, metadata)), encode(items))


class _Clock:
    def __init__(self):
        self.now = 100.0
        self.reads = []

    def __call__(self):
        self.reads.append(self.now)
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _TimedResponse:
    status = 200
    headers = {"x-codex-turn-state": "state-private"}

    def __init__(self, name, clock, *, fail=False):
        self.clock = clock
        self.fail = fail
        self.closed = False
        self.body = (
            {"choices": [{"message": {"role": "assistant", "content": "Done."},
                          "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25,
                       "prompt_tokens_details": {"cached_tokens": 4}}}
            if name == "chat" else
            {"type": "message", "role": "assistant", "stop_reason": "end_turn",
             "content": [{"type": "text", "text": "Done."}],
             "usage": {"input_tokens": 16, "cache_read_input_tokens": 4, "output_tokens": 5}}
        )

    def read(self):
        self.clock.advance(3)
        if self.fail:
            raise TimeoutError("offline body timeout")
        return json.dumps(self.body).encode()

    def __iter__(self):
        self.clock.advance(1)
        yield "data: " + json.dumps({
            "type": "response.output_item.done", "output_index": 0,
            "item": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "Done."}]},
        }) + "\n"
        yield "\n"
        self.clock.advance(2)
        if self.fail:
            raise TimeoutError("offline stream timeout")
        yield "data: " + json.dumps({
            "type": "response.completed", "response": {"usage": {
                "input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
                "input_tokens_details": {"cached_tokens": 4},
            }},
        }) + "\n"
        yield "\n"

    def close(self):
        self.clock.advance(5)
        self.closed = True


def _timing_case(name, clock, *, fail=False):
    response = _TimedResponse(name, clock, fail=fail)

    def opener(request, *, timeout):
        clock.advance(2)
        return response

    if name == "chat":
        model = ChatCompletionsModel(ChatCompletionsEndpoint("http://localhost"), opener=opener)
        return model, response, chat_completions, "_decode_response"
    if name == "messages":
        model = MessagesModel(MessagesEndpoint("http://localhost", "model"), opener=opener)
        return model, response, messages, "_decode_response"
    identifiers = iter(("session-private", "turn-private"))
    model = CodexResponsesModel(StreamingResponsesEndpoint(
        api_url="http://localhost", model="model", bearer_token="FAKE",
        api_provider="codex" if name == "codex" else "api",
    ), opener=opener, identifier_factory=lambda: next(identifiers))
    return model, response, responses, "_collect_sample"


class SampleTimingTests(unittest.TestCase):
    def test_full_adapter_call_is_timed_through_decode_and_cleanup(self):
        for name in ("chat", "messages", "responses", "codex"):
            with self.subTest(name=name):
                clock = _Clock()
                model, response, provider, decoder_name = _timing_case(name, clock)
                build = model._build_request_payload
                decode = getattr(provider, decoder_name)

                def timed_build(*args, **kwargs):
                    clock.advance(1)
                    return build(*args, **kwargs)

                def timed_decode(*args, **kwargs):
                    clock.advance(4)
                    return decode(*args, **kwargs)

                context = ModelContext((Message("user", "Hello."),))
                before = context.items
                with ExitStack() as stack:
                    stack.enter_context(mock.patch("pythia.interaction.model.perf_counter", clock))
                    stack.enter_context(mock.patch.object(model, "_build_request_payload", timed_build))
                    stack.enter_context(mock.patch.object(provider, decoder_name, timed_decode))
                    sample = model.sample(context)
                    self.assertEqual(sample.elapsed_seconds, 15.0)
                    self.assertTrue(response.closed)
                    self.assertEqual(sample.usage, _USAGE)
                    self.assertEqual(sample.stop_reason, "end_turn")
                    self.assertEqual(sample.items, (Message("assistant", "Done."),))
                    self.assertEqual(context.items, before)
                    clock.advance(1000)  # Later tools, saving, and display must not change it.
                    self.assertEqual(sample.context_items()[-2].elapsed_seconds, 15.0)
                    self.assertTrue(sample.display_items()[-1].text.endswith("elapsed=15.00s"))
                    self.assertEqual(clock.reads, [100.0, 115.0])
                if name == "codex":
                    metadata = sample.context_items()[-2]
                    self.assertEqual(metadata.provider_session_id, "session-private")
                    self.assertEqual(metadata.provider_turn_id, "turn-private")
                    self.assertEqual(metadata.provider_turn_state, "state-private")

    def test_timeouts_close_responses_without_producing_sample_metadata(self):
        for name in ("chat", "messages", "responses", "codex"):
            with self.subTest(name=name):
                clock = _Clock()
                model, response, _, _ = _timing_case(name, clock, fail=True)
                context = ModelContext((Message("user", "Hello."),))
                before = context.items
                with mock.patch("pythia.interaction.model.perf_counter", clock):
                    with self.assertRaises(ModelTimeoutError):
                        model.sample(context)
                self.assertTrue(response.closed)
                self.assertEqual(context.items, before)
                self.assertEqual(clock.reads, [100.0])

    def test_messages_compaction_result_gets_one_sample_measurement(self):
        clock = _Clock()
        model, response, _, _ = _timing_case("messages", clock)
        response.body.update(stop_reason="compaction", content=[{
            "type": "compaction", "content": "Summary.",
        }])
        with mock.patch("pythia.interaction.model.perf_counter", clock):
            sample = model.sample(ModelContext((Message("user", "Hello."),)))
        self.assertEqual(sample.stop_reason, "compaction")
        self.assertEqual(sample.elapsed_seconds, 10.0)
        self.assertEqual(
            [item for item in sample.context_items() if isinstance(item, SampleMetadata)],
            [SampleMetadata(_USAGE, elapsed_seconds=10.0)],
        )
        self.assertEqual(summarize_turn_usage(sample.context_items()).sample_count, 1)
        self.assertEqual(clock.reads, [100.0, 110.0])


if __name__ == "__main__":
    unittest.main()
