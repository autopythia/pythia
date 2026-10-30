"""Token traces (pythia.interaction.tokens): the record, its save codec, the
public log without it, its display, and rebuilding prompts from a save."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import tempfile
import unittest

from pythia.interaction import (
    ContextPrefix, InteractionContext, LogProbs, Message, ModelSample, SampleMetadata,
    SaveError, TokenIds, TokenRecord, TokenUsage, chat_completions, interaction_item_from_dict,
    interaction_item_to_dict, load_interaction_save, messages, render_interaction_items,
    responses, save_interaction_save,
)
from pythia.interaction.loop.supervision import read_items
from pythia.interaction.save import PublicLogWriter
from pythia.interaction.save import public_log_line
from pythia.interaction.save import public_log_lines
from pythia.interaction.tokens import MAX_TOKEN_ID
from pythia.interaction.tokens import TOKEN_RECORD_FIELDS
from pythia.interaction.tokens import TokenChain
from pythia.interaction.tokens import prompt_sha256
from pythia.interaction.tokens import token_record_from_dict
from pythia.interaction.tokens import token_record_to_dict

_NAN = float("nan")
_MODEL = "glm-5.3-flash-nvfp4"
_OBS = 151338  # the ID of <|observation|>, a stop token


def _ids(start, count):
    return list(range(start, start + count))


# The plan's examples (doc/chat-completions-token-trace-plan.md, sections 4.5
# and 5): request 1 has 6,100 prompt tokens and samples 900, the last its
# stop; request 2's prompt starts with those 7,000 IDs less the stop, then
# 2,012 rendered tokens (the template's own <|observation|> first), 9,011 in
# all.
_PROMPT_1 = _ids(1000, 6100)
_SAMPLE_1 = [*_ids(20000, 899), _OBS]
_PROMPT_2 = [*_PROMPT_1, *_SAMPLE_1[:-1], _OBS, *_ids(30000, 2011)]
_SAMPLE_2 = [*_ids(40000, 399), _OBS]


def _record_1(**changes):
    fields = dict(
        model=_MODEL, prefix_offset=0, prefix_sha256=prompt_sha256(TokenIds(_PROMPT_1)),
        prefix_token_ids=_PROMPT_1, prefix_log_probs=[_NAN, *[-0.5] * 6099],
        prefix_generation_prompt=2, sample_token_ids=_SAMPLE_1,
        sample_log_probs=[-0.25] * 900, sample_stop=1,
    )
    fields.update(changes)
    return TokenRecord(**fields)


def _record_2(**changes):  # example 2: the hit at 5,632, the previous tokens sent
    fields = dict(
        model=_MODEL, prefix_offset=5632, prefix_sha256=prompt_sha256(TokenIds(_PROMPT_2)),
        prefix_token_ids=_PROMPT_2[5632:], prefix_log_probs=[_NAN, *[-0.75] * 3378],
        prefix_generation_prompt=2, sample_token_ids=_SAMPLE_2, sample_stop=1,
    )
    fields.update(changes)
    return TokenRecord(**fields)


def _record_2b():  # example 5: a retry that hits 8,704, past the 6,999 IDs kept
    return _record_2(prefix_offset=8704, prefix_gap_token_ids=_PROMPT_2[6999:8704],
                     prefix_token_ids=_PROMPT_2[8704:], prefix_log_probs=[_NAN] * 307)


def _metadata(record, prompt_tokens, cached, output_tokens=400):
    return SampleMetadata(
        TokenUsage(prompt_tokens, output_tokens, prompt_tokens + output_tokens, cached),
        elapsed_seconds=6.2, tokens=record,
    )


class PackedArrayTests(unittest.TestCase):
    def test_token_ids_are_little_endian_int32_and_checked(self):
        ids = TokenIds([1, 256, MAX_TOKEN_ID])
        self.assertEqual(ids.to_bytes(), bytes.fromhex("01000000" "00010000" "ffffff7f"))
        self.assertEqual(ids.to_base64(), "AQAAAAABAAD///9/")
        self.assertEqual((len(ids), ids[0], ids[-1], list(ids)), (3, 1, MAX_TOKEN_ID, [1, 256, MAX_TOKEN_ID]))
        self.assertEqual(ids[1:], TokenIds([256, MAX_TOKEN_ID]))
        self.assertEqual(ids[::2], TokenIds([1, MAX_TOKEN_ID]))
        self.assertEqual(ids[:1] + ids[2:], TokenIds([1, MAX_TOKEN_ID]))
        self.assertEqual(TokenIds.from_base64(ids.to_base64()), ids)
        self.assertEqual(hash(TokenIds.from_bytes(ids.to_bytes())), hash(ids))
        self.assertNotEqual(ids, ids.tolist())
        self.assertEqual(repr(TokenIds([5])), "TokenIds([5])")
        self.assertEqual(repr(TokenIds(range(9))), "TokenIds(<9 values>)")
        with self.assertRaises(IndexError):
            ids[3]
        for bad in (-1, MAX_TOKEN_ID + 1, 1.0, True, "1", None):
            with self.subTest(bad=bad), self.assertRaisesRegex(
                    ValueError, r"token ID \[1\] must be an integer from 0 to 2147483647"):
                TokenIds([0, bad])
        for packed in (b"abc", "", bytearray(b"1234")):
            with self.subTest(packed=packed), self.assertRaisesRegex(TypeError, "from_bytes"):
                TokenIds(packed)

    def test_packed_data_needs_whole_values_and_strict_base64(self):
        with self.assertRaisesRegex(ValueError, "packed token IDs take 4 bytes each, not 6"):
            TokenIds.from_bytes(b"\x00" * 6)
        for text in ("AAAA AAA=", "AAA", "AAAAAA", "AAAA\nAAA=", "AAAAAAA=é"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                TokenIds.from_base64(text)
        with self.assertRaisesRegex(TypeError, "base64 string"):
            TokenIds.from_base64(b"AAAAAA==")

    def test_log_probs_are_float32_kept_exactly_with_nan_and_vllms_minus_9999(self):
        values = [_NAN, -0.5, -9999.0, -1e-7, float("-inf")]
        probs = LogProbs(values)
        self.assertEqual(len(probs.to_bytes()), 20)
        decoded = probs.tolist()
        self.assertTrue(math.isnan(decoded[0]))
        self.assertEqual(decoded[1:3], [-0.5, -9999.0])
        self.assertEqual(decoded[3], LogProbs.from_bytes(probs.to_bytes())[3])  # float32's value
        self.assertNotEqual(decoded[3], -1e-7)
        self.assertEqual(decoded[4], float("-inf"))
        self.assertEqual(LogProbs.from_base64(probs.to_base64()), probs)  # NaN included
        self.assertEqual(LogProbs([decoded[3]]).tolist(), [decoded[3]])  # float32 values stay
        for bad in (True, "x", None):
            with self.subTest(bad=bad), self.assertRaisesRegex(TypeError, r"log prob \[0\]"):
                LogProbs([bad])
        for bad in (1e39, 10**400):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "beyond float32"):
                LogProbs([bad])


class TokenRecordTests(unittest.TestCase):
    def test_values_become_packed_arrays_and_an_empty_gap_none(self):
        record = TokenRecord(prefix_offset=3, prefix_gap_token_ids=[], prefix_token_ids=[7, 8],
                             prefix_log_probs=(_NAN, -1), sample_token_ids=[9])
        self.assertEqual(record.prefix_token_ids, TokenIds([7, 8]))
        self.assertIsInstance(record.prefix_log_probs, LogProbs)
        self.assertIsNone(record.prefix_gap_token_ids)
        self.assertEqual(record.prompt_length, 5)
        self.assertIsNone(TokenRecord(sample_token_ids=[1]).prompt_length)
        self.assertEqual(TokenRecord(), TokenRecord())

    def test_parts_need_their_arrays_and_counts_must_fit(self):
        cases = (
            (dict(prefix_offset=0), "prefix_offset needs prefix_token_ids"),
            (dict(prefix_sha256="0" * 16), "prefix_sha256 needs prefix_token_ids"),
            (dict(prefix_log_probs=[_NAN]), "prefix_log_probs needs prefix_token_ids"),
            (dict(prefix_token_ids=[1]), "prefix_token_ids needs prefix_offset"),
            (dict(prefix_offset=0, prefix_token_ids=[1], prefix_log_probs=[_NAN, -1]),
             "prefix_log_probs has 2 values for 1 prefix_token_ids"),
            (dict(prefix_offset=2, prefix_gap_token_ids=[1, 2, 3], prefix_token_ids=[4]),
             "prefix_gap_token_ids has 3 tokens, more than the 2 below prefix_offset"),
            (dict(prefix_offset=1, prefix_token_ids=[4], prefix_generation_prompt=3),
             "prefix_generation_prompt is 3 tokens, more than the prompt's 2"),
            (dict(sample_log_probs=[-1]), "sample_log_probs needs sample_token_ids"),
            (dict(sample_stop=1), "sample_stop needs sample_token_ids"),
            (dict(sample_token_ids=[1], sample_log_probs=[]),
             "sample_log_probs has 0 values for 1 sample_token_ids"),
            (dict(sample_token_ids=[1], sample_stop=2),
             "sample_stop is 2 tokens, more than the sample's 1"),
            (dict(prefix_offset=-1, prefix_token_ids=[1]),
             "prefix_offset must be a nonnegative integer or None"),
            (dict(preemptions=True), "preemptions must be a nonnegative integer or None"),
            (dict(prefix_offset=0, prefix_token_ids=[1], prefix_sha256="ABCDEF0123456789"),
             "prefix_sha256 must be 16 lowercase hex digits"),
            (dict(model=" "), "model must be a nonempty line"),
            (dict(log_probs_mode="raw\nlogprobs"), "log_probs_mode must be a nonempty line"),
            (dict(sample_token_ids=[-5]), r"sample_token_ids: token ID \[0\]"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, message):
                TokenRecord(**arguments)
        with self.assertRaisesRegex(TypeError, "model must be a string or None"):
            TokenRecord(model=5)

    def test_metadata_and_samples_carry_a_record_out_of_their_repr(self):
        record = _record_1()
        metadata = SampleMetadata(TokenUsage(6100, 900, 7000, 0), tokens=record)
        self.assertNotIn("TokenRecord", repr(metadata))
        self.assertNotEqual(metadata, SampleMetadata(TokenUsage(6100, 900, 7000, 0)))
        with self.assertRaisesRegex(TypeError, "tokens must be TokenRecord or None"):
            SampleMetadata(TokenUsage(), tokens={})
        sample = ModelSample((Message("assistant", "Done."),), usage=TokenUsage(6100, 900, 7000, 0),
                             tokens=record)
        [saved] = [item for item in sample.context_items() if isinstance(item, SampleMetadata)]
        self.assertIs(saved.tokens, record)
        self.assertNotIn("TokenRecord", repr(sample))
        with self.assertRaisesRegex(TypeError, "tokens must be TokenRecord or None"):
            ModelSample((Message("assistant", "Done."),), tokens=[1])


class CodecTests(unittest.TestCase):
    def test_the_record_goes_last_in_field_order_with_arrays_in_base64(self):
        record = TokenRecord(model="m", log_probs_mode="raw_logprobs", prefix_offset=2,
                             prefix_sha256="0123456789abcdef", prefix_gap_token_ids=[1],
                             prefix_token_ids=[5, 6], prefix_log_probs=[_NAN, -1.0],
                             prefix_generation_prompt=1, sample_token_ids=[7],
                             sample_log_probs=[-0.25], sample_stop=1, preemptions=1)
        encoded = interaction_item_to_dict(SampleMetadata(TokenUsage(4, 1, 5, 2), tokens=record,
                                                          recovery=("http_503_retry",)))
        self.assertEqual(list(encoded), ["type", "usage", "recovery", "tokens"])
        self.assertEqual(encoded["tokens"], {
            "model": "m", "log_probs_mode": "raw_logprobs", "preemptions": 1,
            "prefix_offset": 2, "prefix_sha256": "0123456789abcdef",
            "prefix_gap_token_ids": "AQAAAA==", "prefix_token_ids": "BQAAAAYAAAA=",
            "prefix_log_probs": "AADAfwAAgL8=", "prefix_generation_prompt": 1,
            "sample_token_ids": "BwAAAA==", "sample_log_probs": "AACAvg==", "sample_stop": 1,
        })
        self.assertEqual(list(encoded["tokens"]), list(TOKEN_RECORD_FIELDS))
        self.assertEqual(token_record_from_dict(token_record_to_dict(record)), record)
        self.assertNotIn("tokens", interaction_item_to_dict(SampleMetadata(TokenUsage())))

    def test_a_save_round_trips_its_records(self):
        items = (Message("user", "Fix the test."), _metadata(_record_1(), 6100, 0, 900),
                 Message("assistant", "Done."), _metadata(_record_2b(), 9011, 8704))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            save_interaction_save(path, InteractionContext(items))
            loaded = load_interaction_save(path)
            lines = path.read_bytes().splitlines()
        self.assertEqual(tuple(loaded.items), items)
        self.assertTrue(math.isnan(loaded.items[1].tokens.prefix_log_probs[0]))
        self.assertTrue(lines[1].endswith(b'"sample_stop": 1}}'))

    def test_null_fields_are_absent_and_unknown_keys_are_ignored(self):
        # A later version may add fields, in the record or beside it: older
        # versions read its lines without them, as today's did this field.
        line = {"type": "sample_metadata",
                "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4,
                          "cached_input_tokens": 0},
                "routed_experts": "AAAA",
                "tokens": {"model": None, "prefix_offset": 0, "prefix_token_ids": "AQAAAA==",
                           "prefix_log_probs": None, "future_field": [1, 2]}}
        item = interaction_item_from_dict(line)
        self.assertEqual(item.tokens, TokenRecord(prefix_offset=0, prefix_token_ids=[1]))
        self.assertEqual(interaction_item_to_dict(item)["tokens"],
                         {"prefix_offset": 0, "prefix_token_ids": "AQAAAA=="})

    def test_a_malformed_record_is_a_save_error_naming_its_field(self):
        usage = {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4,
                 "cached_input_tokens": 0}
        cases = (
            ([1], "sample_metadata.tokens must be an object"),
            ({"prefix_offset": 0, "prefix_token_ids": "AAA"},
             "sample_metadata.tokens.prefix_token_ids must be base64 of packed 4-byte values"),
            ({"sample_token_ids": "AAAAAAA="},
             "sample_metadata.tokens.sample_token_ids must be base64 of packed 4-byte values"),
            ({"sample_log_probs": 5},
             "sample_metadata.tokens.sample_log_probs must be base64 of packed 4-byte values"),
            ({"prefix_offset": "0", "prefix_token_ids": "AQAAAA=="},
             "sample_metadata.tokens.prefix_offset must be a nonnegative integer"),
            ({"sample_token_ids": "AQAAAA==", "sample_stop": True},
             "sample_metadata.tokens.sample_stop must be a nonnegative integer"),
            ({"model": 5}, "sample_metadata.tokens.model must be a string"),
            ({"prefix_offset": 0, "prefix_token_ids": "AQAAAA==",
              "prefix_log_probs": "AADAfwAAgL8="},
             "sample_metadata.tokens.prefix_log_probs has 2 values for 1 prefix_token_ids"),
            ({"prefix_sha256": "0123456789abcdef"},
             "sample_metadata.tokens.prefix_sha256 needs prefix_token_ids"),
        )
        for tokens, message in cases:
            with self.subTest(tokens=tokens), self.assertRaisesRegex(SaveError, f"^{message}$"):
                interaction_item_from_dict({"type": "sample_metadata", "usage": usage,
                                            "tokens": tokens})

    def test_no_request_includes_a_record(self):
        items = (Message("user", "Question."), Message("assistant", "Answer."))
        traced = (*items, _metadata(_record_1(), 6100, 0, 900))
        for encode in (chat_completions._encode_context_messages, messages._encode_context,
                       responses._encode_context_items):
            with self.subTest(encoder=encode.__module__):
                self.assertEqual(encode(traced), encode(items))


class PublicLogTests(unittest.TestCase):
    def test_a_public_log_leaves_records_out_line_for_line(self):
        traced = _metadata(_record_1(), 6100, 0, 900)
        plain = SampleMetadata(traced.usage, elapsed_seconds=6.2)
        checkpoint = ContextPrefix((Message("user", "summary"), traced))
        items = (Message("user", "Fix the test."), traced, plain, checkpoint)
        full = b"".join(public_log_line(item) if item is plain else
                        (json.dumps(interaction_item_to_dict(item), ensure_ascii=False) + "\n").encode()
                        for item in items)
        public = public_log_lines(full)
        self.assertEqual(public, [public_log_line(item) for item in items])
        self.assertEqual(public[1], public_log_line(plain))
        self.assertEqual(public[2], full.splitlines(keepends=True)[2])  # byte for byte
        self.assertEqual(json.loads(public[3]), {"type": "context_prefix", "prefix_items": [
            interaction_item_to_dict(Message("user", "summary")), interaction_item_to_dict(plain)]})
        self.assertNotIn(b'"tokens"', b"".join(public))
        self.assertNotIn(_record_1().sample_token_ids.to_base64()[:40].encode(), b"".join(public))

    def test_the_writer_and_a_resume_see_no_difference(self):
        items = (Message("user", "Fix the test."), _metadata(_record_1(), 6100, 0, 900),
                 Message("assistant", "Done."), _metadata(_record_2(), 9011, 5632))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.jsonl"
            writer = PublicLogWriter(path)
            writer.save(InteractionContext(items[:2]))
            writer.save(InteractionContext(items))
            expected = b"".join(public_log_line(item) for item in items)
            self.assertEqual(path.read_bytes(), expected)
            full = b"".join((json.dumps(interaction_item_to_dict(item), ensure_ascii=False)
                             + "\n").encode() for item in items)
            resumed = PublicLogWriter(path)
            resumed.sync(full, len(items))
            self.assertEqual((resumed.take_warnings(), resumed.drifted), ((), False))
            self.assertEqual(path.read_bytes(), expected)


class DisplayTests(unittest.TestCase):
    def _line(self, item):
        [display] = render_interaction_items((item,), color=False)
        return display.text

    def test_the_sample_line_shows_the_saved_tokens_then_offset_and_gap(self):
        cases = (
            (_metadata(_record_2(), 9011, 5632),
             "[sample] input=9011 output=400 total=9411 cached=5632 tokens=3379/400 elapsed=6.20s"),
            (_metadata(_record_2b(), 9011, 8704),
             "[sample] input=9011 output=400 total=9411 cached=8704 tokens=307/400 gap=1705 "
             "elapsed=6.20s"),
            (_metadata(_record_2(prefix_offset=5120, prefix_token_ids=_PROMPT_2[5120:],
                                 prefix_log_probs=None, preemptions=1), 9011, 5632),
             "[sample] input=9011 output=400 total=9411 cached=5632 tokens=3891/400 offset=5120 "
             "elapsed=6.20s"),
            (_metadata(TokenRecord(model=_MODEL, sample_token_ids=_SAMPLE_2), 9011, 5632),
             "[sample] input=9011 output=400 total=9411 cached=5632 tokens=-/400 elapsed=6.20s"),
            (_metadata(TokenRecord(model=_MODEL), 9011, 5632),
             "[sample] input=9011 output=400 total=9411 cached=5632 tokens=-/- elapsed=6.20s"),
            (SampleMetadata(TokenUsage(9011, 400, 9411, 5632), elapsed_seconds=6.2),
             "[sample] input=9011 output=400 total=9411 cached=5632 elapsed=6.20s"),
        )
        for item, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(self._line(item), expected)

    def test_samples_and_the_supervisors_reads_show_the_same(self):
        expected = self._line(_metadata(_record_2(), 9011, 5632))
        sample = ModelSample((Message("assistant", "Done."),), usage=TokenUsage(9011, 400, 9411, 5632),
                             elapsed_seconds=6.2, tokens=_record_2())
        self.assertEqual(sample.display_items()[-1].text, expected)
        [entry] = read_items((_metadata(_record_2(), 9011, 5632),), 0, 1)["items"]
        self.assertEqual(entry["text"], expected)


class TokenChainTests(unittest.TestCase):
    def test_records_rebuild_their_prompts_from_the_tokens_before_them(self):
        chain = TokenChain()
        first = chain.add(_record_1(), 6100)
        self.assertEqual(first.prompt, TokenIds(_PROMPT_1))
        self.assertIsNone(first.problem)
        self.assertEqual(chain.tokens, TokenIds(_PROMPT_1 + _SAMPLE_1))
        second = chain.add(_record_2(), 9011)
        self.assertEqual((second.prompt, second.problem), (TokenIds(_PROMPT_2), None))
        self.assertEqual(chain.tokens, TokenIds(_PROMPT_2 + _SAMPLE_2))

    def test_a_gap_fills_what_no_record_holds(self):
        chain = TokenChain()
        chain.add(_record_1(), 6100)
        rebuilt = chain.add(_record_2b(), 9011)
        self.assertEqual((rebuilt.prompt, rebuilt.problem), (TokenIds(_PROMPT_2), None))
        # A save's first record: the gap holds everything below the hit.
        first = TokenChain().add(_record_1(prefix_offset=5632, prefix_gap_token_ids=_PROMPT_1[:5632],
                                               prefix_token_ids=_PROMPT_1[5632:],
                                               prefix_log_probs=None), 6100)
        self.assertEqual((first.prompt, first.problem), (TokenIds(_PROMPT_1), None))

    def test_a_record_that_cant_be_rebuilt_ends_the_chain(self):
        chain = TokenChain()
        chain.add(_record_1(), 6100)
        wrong = chain.add(_record_2(prefix_sha256="0" * 16), 9011)
        self.assertEqual(wrong, (None, "the rebuilt prompt doesn't match prefix_sha256"))
        self.assertIsNone(chain.tokens)
        self.assertEqual(chain.add(_record_2(), 9011),
                         (None, "its first 5632 tokens are in no earlier record"))
        self.assertEqual(chain.add(_record_1(), 6100).problem, None)  # a new chain
        self.assertEqual(chain.add(_record_2(), 9000),
                         (None, "the rebuilt prompt has 9011 tokens, where the sample's input "
                                "has 9000"))
        chain.add(_record_1(), 6100)
        self.assertEqual(chain.add(_record_2(prefix_offset=7500, prefix_token_ids=_PROMPT_2[7500:],
                                             prefix_log_probs=None), 9011),
                         (None, "its first 7500 tokens pass the 7000 that the earlier records "
                                "hold"))
        chain.add(_record_1(), 6100)
        self.assertEqual(chain.add(None, 9011), (None, "the sample has no token record"))
        self.assertEqual(chain.add(_record_2(), 9011).problem,
                         "its first 5632 tokens are in no earlier record")
        chain.add(_record_1(), 6100)
        self.assertEqual(chain.add(TokenRecord(model=_MODEL, sample_token_ids=_SAMPLE_2), 9011),
                         (None, "the record has no prefix_token_ids"))

    def test_another_models_record_and_one_without_a_sample_start_new_chains(self):
        chain = TokenChain()
        chain.add(_record_1(), 6100)
        self.assertEqual(chain.add(_record_2(model="another"), 9011).problem,
                         "its first 5632 tokens are in no earlier record")
        chain = TokenChain()
        self.assertEqual(chain.add(_record_1(sample_token_ids=None, sample_log_probs=None,
                                             sample_stop=None), 6100).problem, None)
        self.assertIsNone(chain.tokens)
        chain = TokenChain()
        chain.add(_record_1(), None)  # no length check without the input's length
        self.assertEqual(chain.add(_record_2(prefix_sha256="0" * 16), 9011, check_hash=False),
                         (TokenIds(_PROMPT_2), None))

    def test_an_edited_save_fails_its_hash_where_the_change_shows(self):
        # Record 2 is example 5's: a retry's hit at 8,704 leaves a gap, and
        # its prompt's first 6,999 tokens come from record 1's prefix and sample.
        items = (Message("user", "Fix the test."), _metadata(_record_1(), 6100, 0, 900),
                 Message("assistant", "Done."), _metadata(_record_2b(), 9011, 8704))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            save_interaction_save(path, InteractionContext(items))
            saved = path.read_bytes()
            self.assertEqual(self._problems(load_interaction_save(path)), [None, None])
            # One of sample 1's IDs: record 1's own prompt is unchanged, and
            # record 2's, which holds that ID, fails its hash.
            self._edit(path, saved, "sample_token_ids")
            self.assertEqual(self._problems(load_interaction_save(path)),
                             [None, "the rebuilt prompt doesn't match prefix_sha256"])
            # One of record 1's prefix IDs: its own hash fails, which ends the
            # chain, so record 2 has nothing to start from.
            self._edit(path, saved, "prefix_token_ids")
            self.assertEqual(self._problems(load_interaction_save(path)), [
                "the rebuilt prompt doesn't match prefix_sha256",
                "its first 6999 tokens are in no earlier record"])

    @staticmethod
    def _edit(path, saved, field):
        """Change the 11th ID of ``field`` in record 1 of the save ``saved``."""
        lines = saved.splitlines(keepends=True)
        record = json.loads(lines[1])
        packed = bytearray(base64.b64decode(record["tokens"][field]))
        packed[40] ^= 1
        record["tokens"][field] = base64.b64encode(bytes(packed)).decode()
        lines[1] = (json.dumps(record) + "\n").encode()
        path.write_bytes(b"".join(lines))

    @staticmethod
    def _problems(context):
        chain = TokenChain()
        return [chain.add(item.tokens, item.usage.input_tokens).problem
                for item in context.items if isinstance(item, SampleMetadata)]


if __name__ == "__main__":
    unittest.main()
