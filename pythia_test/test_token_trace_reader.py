"""Reading token traces back: each sample's prompt rebuilt from a save, the
segments that join samples into sequences, and the JSONL that
``python3 -m pythia.interaction.tokens SAVE`` writes."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pythia.interaction import (
    ContextPrefix, InteractionContext, Message, SampleMetadata, SampleParams, TokenIds,
    TokenRecord, TokenUsage, ToolResult, load_interaction_save, save_interaction_save,
)
from pythia.interaction._cli_save import full_log_path
from pythia.interaction.tokens import prompt_sha256
from pythia.interaction.tokens import segments
from pythia.interaction.tokens import traced_samples
from pythia.interaction.tokens.__main__ import main
from pythia_test.test_token_trace_chat import ChainFixture, _Server, _model

_NAN = float("nan")


def _record(prompt, offset, sample, *, probs=None, stop=None, gap=None):
    return TokenRecord(model="m", prefix_offset=offset, prefix_sha256=prompt_sha256(TokenIds(prompt)),
                       prefix_gap_token_ids=gap, prefix_token_ids=prompt[offset:],
                       sample_token_ids=sample, sample_log_probs=probs, sample_stop=stop)


def _metadata(record, prompt_length):
    return SampleMetadata(TokenUsage(prompt_length, 2, prompt_length + 2, 0), tokens=record)


# Three samples: 2 extends 1 with its stop, 3 extends 2 with another token in
# place of its stop (the template's), and 4 starts over after a compaction.
_P1, _S1 = list(range(100, 110)), [7, 8, 99]
_P2 = [*_P1, *_S1, 20, 21]
_S2 = [9, 99]
_P3 = [*_P2, 9, 98, 22]
_S3 = [10, 99]
_P4 = [100, 101, 30, 31]
_ITEMS = (
    Message("user", "q"), _metadata(_record(_P1, 0, _S1, probs=[-0.5, -0.25, -0.125], stop=1), 10),
    Message("user", "r"), _metadata(_record(_P2, 8, _S2, probs=[-1.0, -2.0], stop=1), 15),
    Message("user", "s"), _metadata(_record(_P3, 12, _S3, stop=1), 18),
    ContextPrefix((Message("user", "summary"),)),
    _metadata(_record(_P4, 0, [11], gap=None), 4),
)


class ReaderTests(unittest.TestCase):
    def test_each_samples_prompt_is_rebuilt_from_the_records_before_it(self):
        samples = list(traced_samples(_ITEMS))
        self.assertEqual([sample.line for sample in samples], [2, 4, 6, 8])
        self.assertEqual([sample.prompt for sample in samples],
                         [TokenIds(prompt) for prompt in (_P1, _P2, _P3, _P4)])
        self.assertEqual([sample.problem for sample in samples], [None] * 4)

    def test_segments_join_the_samples_that_extend_each_other(self):
        first, second = segments(traced_samples(_ITEMS))
        self.assertEqual([sample.line for sample in first.samples], [2, 4, 6])
        self.assertEqual(first.token_ids, TokenIds([*_P3, *_S3]))
        # Sample 2's stop is in the final sequence; the template wrote 98 in
        # place of sample 2's stop (99), which is left out.
        self.assertEqual(first.sampled, ((10, 13), (15, 16), (18, 20)))
        values = first.log_probs.tolist()
        self.assertEqual(values[10:13], [-0.5, -0.25, -0.125])
        self.assertEqual(values[15], -1.0)
        self.assertTrue(all(math.isnan(value) for value in values[:10] + values[16:]))
        self.assertEqual(second.token_ids, TokenIds([*_P4, 11]))
        self.assertEqual(second.sampled, ((4, 5),))

    def test_a_sample_without_a_trace_ends_a_segment_and_a_gap_doesnt(self):
        gap = _record(_P2, 13, _S2, gap=_P2[10:13])  # the hit passed the save's 10 tokens
        items = (_ITEMS[1], _metadata(gap, 15),
                 SampleMetadata(TokenUsage(17, 2, 19, 0)),  # no trace
                 _metadata(_record(_P3, 0, _S3), 18))
        samples = list(traced_samples(items))
        self.assertEqual(samples[1].problem, None)  # its gap holds what record 1 lacks
        self.assertEqual(samples[2].problem, "the sample has no token record")
        runs = [[sample.line for sample in segment.samples] for segment in segments(samples)]
        self.assertEqual(runs, [[1, 2], [4]])

    def test_the_prompts_rebuilt_are_the_tokens_that_were_sent(self):
        # An adapter session against a patched server: each request that
        # sends the previous tokens back sends what the save rebuilds.
        chain = ChainFixture()
        server = _Server(chain._first(), chain._second(5632))
        model = _model(server)
        context = chain._turn(model)
        context.extend(model.sample(context, sample_params=SampleParams(
            token_trace="token_ids")).context_items())
        samples = list(traced_samples(context.items))
        sent = server.requests[1]
        self.assertEqual(sent["prompt_token_ids"], samples[0].prompt.tolist())
        self.assertEqual(sent["token_ids"], samples[0].record.sample_token_ids.tolist())
        self.assertEqual(samples[1].prompt, TokenIds(chain._P2))
        [segment] = segments(samples)
        self.assertEqual(len(segment.samples), 2)


class CommandTests(unittest.TestCase):
    def _run(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, [json.loads(line) for line in out.getvalue().splitlines()], err.getvalue()

    def test_a_cli_save_is_read_from_its_full_log(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "session"
            full = full_log_path(prefix)
            full.parent.mkdir()
            save_interaction_save(full, InteractionContext(_ITEMS))
            for argument in (prefix, f"{prefix}.jsonl"):
                with self.subTest(argument=argument):
                    code, lines, _ = self._run(str(argument))
                    self.assertEqual(code, 0)
                    self.assertEqual([line["type"] for line in lines],
                                     ["record", "record", "record", "segment", "record", "segment"])
            code, lines, _ = self._run(str(prefix), "--segments")
            self.assertEqual([line["lines"] for line in lines], [[2, 6], [8, 8]])
            self.assertEqual(lines[0]["sampled"], [[10, 13], [15, 16], [18, 20]])
            self.assertEqual(lines[0]["log_probs"][9:11], [None, -0.5])
            code, lines, _ = self._run(str(prefix), "--records")
            self.assertEqual(lines[0], {
                "type": "record", "line": 2, "input_tokens": 10, "rebuilt": True, "model": "m",
                "prefix_offset": 0, "prefix_sha256": prompt_sha256(TokenIds(_P1)),
                "prefix_token_ids": _P1, "sample_token_ids": _S1,
                "sample_log_probs": [-0.5, -0.25, -0.125], "sample_stop": 1})

    def test_a_failed_rebuild_is_said_and_an_unreadable_save_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            broken = (_ITEMS[1], _metadata(_record(_P2, 8, _S2), 16))  # the wrong length
            save_interaction_save(path, InteractionContext(broken))
            code, lines, _ = self._run(str(path), "--records")
            self.assertEqual(code, 0)
            self.assertEqual((lines[1]["rebuilt"], lines[1]["problem"]),
                             (False, "the rebuilt prompt has 15 tokens, where the sample's input "
                                     "has 16"))
            code, lines, err = self._run(str(Path(directory) / "missing.jsonl"))
            self.assertEqual((code, lines), (1, []))
            self.assertIn("tokens: can't read", err)

    def test_it_runs_as_a_module_without_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            save_interaction_save(path, InteractionContext(_ITEMS[:2]))
            result = subprocess.run([sys.executable, "-W", "error", "-m",
                                     "pythia.interaction.tokens", str(path)],
                                    capture_output=True, text=True, timeout=60)
        self.assertEqual((result.returncode, result.stderr), (0, ""))
        self.assertEqual([json.loads(line)["type"] for line in result.stdout.splitlines()],
                         ["record", "segment"])


if __name__ == "__main__":
    unittest.main()
