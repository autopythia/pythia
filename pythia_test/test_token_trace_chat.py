"""Token traces from a vLLM Chat Completions server: each level's request
fields, and the response's token fields made into the sample's record, with
its checks and notices, its offset, and its gap.

The responses are scripted, shaped like vLLM's and, where the fields are the
fork's (patches 0056 and 0057), like the patched server's
(doc/chat-completions-token-trace-plan.md, section 3.4).
"""

from __future__ import annotations

import json
import math
import unittest

from pythia.interaction import (
    ChatCompletionsModel, ContextPrefix, Init, InteractionContext, Message, ModelTransportError,
    SampleMetadata, SampleParams, TokenIds, TokenRecord, TokenUsage, ToolResult, ToolSpec,
)
from pythia.interaction._token_trace import NOTICES
from pythia.interaction.loop.kernel import run_turn
from pythia.interaction.tokens import TokenChain
from pythia.interaction.tokens import prompt_sha256
from pythia_test.interaction_helpers import chat_endpoint
from pythia_test.test_loop_kernel import _Host, _config, _environment

_MODEL = "glm-5.3-flash-nvfp4"
_OBS = 151338


def _ids(start, count):
    return list(range(start, start + count))


class _Response:
    status, headers = 200, {}

    def __init__(self, payload, status=200):
        self._body = json.dumps(payload).encode()
        self.status = status

    def read(self):
        return self._body

    def close(self):
        pass


def _response(prompt=None, sample=(), *, cached=0, details=True, log_probs=None,
              prompt_log_probs=None, prompt_tokens=None, message=None, **fields):
    """A vLLM response: ``prompt`` and ``sample`` IDs (None leaves a field out),
    ``log_probs`` for the sampled tokens (a value per ID, or entries), and
    ``prompt_log_probs`` from ``prompt_logprobs_offset`` (or 0) on. A patched
    server's ``prompt`` holds only the IDs past ``prompt_token_ids_offset``,
    so ``prompt_tokens`` gives the whole prompt's length."""
    choice = {"message": message or {"role": "assistant", "content": "Done."},
              "finish_reason": "stop",
              **{key: fields.pop(key) for key in ("num_stop_tokens",) if key in fields}}
    if sample is not None:
        choice["token_ids"] = list(sample)
    if log_probs is not None:
        entries = [entry if isinstance(entry, dict) else
                   {"token": f"token_id:{token}", "logprob": entry, "bytes": [], "top_logprobs": []}
                   for token, entry in zip(sample or (), log_probs)] if sample is not None else log_probs
        choice["logprobs"] = {"content": entries}
    usage = {"prompt_tokens": (prompt_tokens if prompt_tokens is not None
                               else 0 if prompt is None else len(prompt)),
             "completion_tokens": len(sample or ()), "total_tokens": 0}
    if details:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    payload = {"model": _MODEL, "choices": [choice], "usage": usage, **fields}
    if prompt is not None:
        payload["prompt_token_ids"] = list(prompt)
    if prompt_log_probs is not None:
        start = fields.get("prompt_logprobs_offset", 0)
        payload["prompt_logprobs"] = [None] + [
            {str(token): {"logprob": value, "rank": 1, "decoded_token": "x"}}
            for token, value in zip(prompt[start + 1:], prompt_log_probs)]
    return payload


class _Server:
    """A scripted server: one response per request, and the requests it got."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def __call__(self, request, timeout):
        self.requests.append(json.loads(request.data))
        response = self.responses.pop(0)
        if isinstance(response, tuple):  # (status, payload)
            return _Response(response[1], status=response[0])
        return _Response(response)


def _model(server):
    return ChatCompletionsModel(chat_endpoint(model=_MODEL), opener=server)


def _context(*items):
    return InteractionContext((Init(model="scripted"), Message("user", "Fix the test."), *items))


def _sample(model, level, context=None):
    return model.sample(context or _context(), sample_params=SampleParams(token_trace=level))


class RequestTests(unittest.TestCase):
    def test_each_level_adds_its_fields_to_those_of_the_one_before(self):
        fields = {
            None: {},
            "none": {},
            "token_ids": {"return_token_ids": True},
            "sample_log_probs": {"return_token_ids": True, "logprobs": True, "top_logprobs": 0,
                                 "return_tokens_as_token_ids": True},
            "prefix_log_probs": {"return_token_ids": True, "logprobs": True, "top_logprobs": 0,
                                 "return_tokens_as_token_ids": True, "prompt_logprobs": 0,
                                 "skip_reading_prefix_cache": False},
        }
        for level, expected in fields.items():
            with self.subTest(level=level):
                server = _Server(_response(_ids(100, 10), [7, _OBS],
                                           log_probs=[-0.5, -0.25],
                                           prompt_log_probs=[-1.0] * 9))
                _sample(_model(server), level)
                [request] = server.requests
                self.assertEqual({key: request[key] for key in request
                                  if key not in ("messages", "model", "stream")}, expected)


class RecordTests(unittest.TestCase):
    def test_a_response_becomes_the_samples_record_at_each_level(self):
        prompt, sample = _ids(100, 10), [7, 8, _OBS]
        response = _response(prompt, sample, log_probs=[-0.5, -9999.0, -0.25],
                             prompt_log_probs=[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0])
        whole = dict(model=_MODEL, prefix_offset=0, prefix_sha256=prompt_sha256(TokenIds(prompt)),
                     prefix_token_ids=prompt, sample_token_ids=sample)
        for level, extra in (
            ("token_ids", {}),
            ("sample_log_probs", dict(sample_log_probs=[-0.5, -9999.0, -0.25])),
            ("prefix_log_probs", dict(sample_log_probs=[-0.5, -9999.0, -0.25],
                                      prefix_log_probs=[math.nan, -1.0, -2.0, -3.0, -4.0, -5.0,
                                                        -6.0, -7.0, -8.0, -9.0])),
        ):
            with self.subTest(level=level):
                traced = _sample(_model(_Server(response)), level)
                self.assertEqual(traced.tokens, TokenRecord(**whole, **extra))
                self.assertEqual(traced.notices, ())
        # The prefix's values score its tokens from position 1, the sample's
        # follow on: each value at the index of the token it scores.
        record = traced.tokens
        self.assertEqual(len(record.prefix_log_probs) + len(record.sample_log_probs), 13)
        self.assertTrue(math.isnan(record.prefix_log_probs[0]))
        # The saved metadata keeps it.
        [metadata] = [item for item in traced.context_items() if isinstance(item, SampleMetadata)]
        self.assertIs(metadata.tokens, record)

    def test_the_offset_is_where_the_servers_processing_began(self):
        prompt = _ids(1000, 9011)
        server = _Server(
            _response(prompt, [7], cached=5632),
            # Patch 0057: prompt log probs from the cache hit, and where they start.
            _response(prompt, [7], cached=5632, prompt_logprobs_offset=5632,
                      prompt_log_probs=[-1.0] * 3378, logprobs_mode="raw_logprobs",
                      num_preemptions=0),
            # A preemption during the prefill: the last prefill began elsewhere.
            _response(prompt, [7], cached=5632, prompt_logprobs_offset=8704,
                      prompt_log_probs=[-1.0] * 306, num_preemptions=1),
        )
        model = _model(server)
        first = _sample(model, "token_ids").tokens
        self.assertEqual((first.prefix_offset, first.prefix_token_ids), (5632, TokenIds(prompt[5632:])))
        self.assertEqual(first.prefix_gap_token_ids, TokenIds(prompt[:5632]))  # no earlier record
        self.assertEqual(first.prefix_sha256, prompt_sha256(TokenIds(prompt)))
        second = _sample(model, "prefix_log_probs").tokens
        self.assertEqual((second.prefix_offset, len(second.prefix_log_probs), second.log_probs_mode,
                          second.preemptions), (5632, 3379, "raw_logprobs", None))
        third = _sample(model, "prefix_log_probs")
        self.assertEqual((third.tokens.prefix_offset, len(third.tokens.prefix_token_ids),
                          third.tokens.preemptions), (8704, 307, 1))
        self.assertIn(" tokens=307/1 offset=8704 gap=8704 ", third.display_items()[-1].text)

    def test_prompt_log_probs_after_a_hit_need_their_offset(self):
        prompt = _ids(1000, 100)
        # A server without patch 0057 that read the cache anyway labels the
        # rows from position 1: their positions are unknown.
        model = _model(_Server(_response(prompt, [7], cached=64,
                                         prompt_log_probs=[-1.0] * 35),
                               _response(prompt, [7], cached=64, prompt_log_probs=[-1.0] * 35)))
        traced = _sample(model, "prefix_log_probs")
        self.assertEqual((traced.tokens.prefix_offset, traced.tokens.prefix_log_probs), (64, None))
        self.assertEqual(traced.notices, (NOTICES["unplaced_prompt_log_probs"],
                                          NOTICES["no_log_probs"]))
        # Each notice shows once.
        self.assertEqual(_sample(model, "prefix_log_probs").notices, ())
        # Stock vLLM reads no cache for them: a whole list starts at 0.
        whole = _sample(_model(_Server(_response(prompt, [7], cached=0,
                                                 prompt_log_probs=[-1.0] * 99))),
                        "prefix_log_probs").tokens
        self.assertEqual((whole.prefix_offset, len(whole.prefix_log_probs)), (0, 100))

    def test_vllm_before_0_10_2_names_the_sampled_ids_in_its_log_probs(self):
        entries = [{"token": "token_id:7", "logprob": -0.5}, {"token": "token_id:8", "logprob": -1.0}]
        traced = _sample(_model(_Server(_response(None, None, log_probs=entries))),
                         "sample_log_probs")
        self.assertEqual(traced.tokens, TokenRecord(model=_MODEL, sample_token_ids=[7, 8],
                                                    sample_log_probs=[-0.5, -1.0]))
        self.assertEqual(traced.notices, (NOTICES["no_prompt_ids"],))
        # At token_ids, nothing names them.
        traced = _sample(_model(_Server(_response(None, None))), "token_ids")
        self.assertEqual(traced.tokens, TokenRecord(model=_MODEL))
        self.assertEqual(traced.notices, (NOTICES["no_prompt_ids"], NOTICES["no_sample_ids"]))
        self.assertIn(" tokens=-/- ", traced.display_items()[-1].text)

    def test_a_part_that_fails_a_check_is_left_out_with_a_notice(self):
        prompt, sample = _ids(100, 10), [7, 8]
        cases = (
            (_response([*prompt[:-1], -1], sample), "token_ids", "bad_prompt_ids",
             dict(sample_token_ids=sample)),
            (dict(_response(prompt, sample), usage={"prompt_tokens": 11, "completion_tokens": 2,
                                                    "prompt_tokens_details": {"cached_tokens": 0}}),
             "token_ids", "bad_prompt_ids", dict(sample_token_ids=sample)),
            (_response(prompt, [7, True]), "token_ids", "bad_sample_ids", None),
            (_response(prompt, sample, log_probs=[-0.5]), "sample_log_probs", "bad_log_probs", None),
            (_response(prompt, sample, log_probs=[-0.5, None]), "sample_log_probs",
             "bad_log_probs", None),
            (_response(prompt, sample, log_probs=[{"token": "token_id:7", "logprob": -0.5},
                                                  {"token": "token_id:9", "logprob": -0.5}]),
             "sample_log_probs", "bad_log_probs", None),
            (_response(prompt, sample, log_probs=[-0.5, -0.5], prompt_log_probs=[-1.0] * 8),
             "prefix_log_probs", "bad_prompt_log_probs", None),
            (_response(prompt, sample, num_generation_prompt_tokens=11), "token_ids",
             "bad_counts", None),
            (_response(prompt, sample, num_stop_tokens=3), "token_ids", "bad_counts", None),
            (_response(prompt, sample, cached=11), "token_ids", "bad_offset",
             dict(sample_token_ids=sample)),
        )
        for response, level, problem, kept in cases:
            with self.subTest(problem=problem, level=level):
                traced = _sample(_model(_Server(response)), level)
                self.assertIn(NOTICES[problem], traced.notices)
                if kept is not None:
                    self.assertEqual(traced.tokens, TokenRecord(model=_MODEL, **kept))
                self.assertNotEqual(traced.tokens, None)

    def test_the_patched_servers_counts_are_kept(self):
        prompt = [*_ids(100, 8), 151331, 151350]  # ... <|assistant|><think>
        traced = _sample(_model(_Server(_response(prompt, [7, _OBS], num_stop_tokens=1,
                                                  num_generation_prompt_tokens=2))), "token_ids")
        self.assertEqual((traced.tokens.prefix_generation_prompt, traced.tokens.sample_stop), (2, 1))

    def test_without_cached_tokens_a_record_starts_where_the_saves_tokens_end(self):
        prompt = _ids(100, 10)
        before = _context(SampleMetadata(TokenUsage(6, 2, 8, 0), tokens=TokenRecord(
            model=_MODEL, prefix_offset=0, prefix_token_ids=prompt[:6], sample_token_ids=prompt[6:8])))
        traced = _sample(_model(_Server(_response(prompt, [7], details=False))), "token_ids", before)
        self.assertEqual((traced.tokens.prefix_offset, traced.tokens.prefix_gap_token_ids), (8, None))
        self.assertEqual(traced.notices, (NOTICES["no_cached_tokens"],))


class GapTests(unittest.TestCase):
    # The plan's example 5, without IDs sent: request 2 re-renders sample 1,
    # so its prompt first differs from sample 1 at 6,710; a retry hits 8,704.
    _P1, _S1 = _ids(1000, 6100), [*_ids(20000, 899), _OBS]
    _P2 = [*_P1, *_S1[:610], *_ids(30000, 2301)]  # 9,011 tokens, re-rendered from 6,710

    def _first(self):
        return SampleMetadata(TokenUsage(6100, 900, 7000, 0), tokens=TokenRecord(
            model=_MODEL, prefix_offset=0, prefix_sha256=prompt_sha256(TokenIds(self._P1)),
            prefix_token_ids=self._P1, sample_token_ids=self._S1))

    def test_the_gap_is_what_the_save_lacks_below_the_hit(self):
        context = _context(self._first(), Message("assistant", "Running it."))
        for cached, gap in ((5632, None), (8704, self._P2[6710:8704])):
            with self.subTest(cached=cached):
                record = _sample(_model(_Server(_response(self._P2, [7], cached=cached))),
                                 "token_ids", context).tokens
                self.assertEqual((record.prefix_offset, record.prefix_gap_token_ids),
                                 (cached, None if gap is None else TokenIds(gap)))
                self.assertEqual(record.prefix_token_ids, TokenIds(self._P2[cached:]))

    def test_another_models_record_holds_nothing_for_this_one(self):
        other = SampleMetadata(TokenUsage(6100, 900, 7000, 0), tokens=TokenRecord(
            model="another", prefix_offset=0, prefix_token_ids=self._P1, sample_token_ids=self._S1))
        record = _sample(_model(_Server(_response(self._P2, [7], cached=5632))), "token_ids",
                         _context(other)).tokens
        self.assertEqual(record.prefix_gap_token_ids, TokenIds(self._P2[:5632]))

    def test_a_resumed_model_rebuilds_the_tokens_from_the_saves_records(self):
        # Record 2, as saved: its prompt from 5,632, and its sample.
        second = SampleMetadata(TokenUsage(9011, 400, 9411, 5632), tokens=TokenRecord(
            model=_MODEL, prefix_offset=5632, prefix_sha256=prompt_sha256(TokenIds(self._P2)),
            prefix_token_ids=self._P2[5632:], sample_token_ids=_ids(40000, 400)))
        prompt3 = [*self._P2, *_ids(40000, 400), *_ids(50000, 300)]
        context = _context(self._first(), Message("assistant", "Running it."), second)
        record = _sample(_model(_Server(_response(prompt3, [7], cached=9216))), "token_ids",
                         context).tokens
        self.assertEqual((record.prefix_offset, record.prefix_gap_token_ids), (9216, None))
        # A record whose rebuild fails its hash holds nothing: the gap is the hit.
        broken = SampleMetadata(second.usage, tokens=TokenRecord(
            model=_MODEL, prefix_offset=5632, prefix_sha256="0" * 16,
            prefix_token_ids=self._P2[5632:], sample_token_ids=_ids(40000, 400)))
        record = _sample(_model(_Server(_response(prompt3, [7], cached=9216))), "token_ids",
                         _context(self._first(), broken)).tokens
        self.assertEqual(record.prefix_gap_token_ids, TokenIds(prompt3[:9216]))

    def test_the_models_own_last_sample_is_kept_for_the_next(self):
        prompt2 = [*self._P1, *self._S1[:-1], _OBS, *_ids(30000, 2011)]
        server = _Server(_response(self._P1, self._S1), _response(prompt2, [7], cached=8704))
        model = _model(server)
        context = _context()
        first = _sample(model, "token_ids", context)
        context.extend(first.context_items())
        record = _sample(model, "token_ids", context).tokens
        # The hit passes the 7,000 tokens that record 1 shares with prompt 2:
        # the template's <|observation|> is the sampled stop's ID.
        self.assertEqual(record.prefix_gap_token_ids, TokenIds(prompt2[7000:8704]))


class ChainFixture:
    """The plan's examples 2 and 5, for sending the latest sample's tokens back
    (patch 0056): request 1 has 6,100 prompt tokens and samples 900, a tool
    call that ends with its stop; request 2 sends both back, of which the
    server keeps 6,999, dropping the stop, and returns the 2,012 IDs after
    them, the template's <|observation|> first."""

    _P1, _S1 = _ids(1000, 6100), [*_ids(20000, 899), _OBS]
    _P2 = [*_P1, *_S1[:-1], _OBS, *_ids(30000, 2011)]
    @staticmethod
    def _call(number):
        """An assistant message with tool call ``call_N``."""
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{number}", "type": "function",
             "function": {"name": "exec_command", "arguments": "{}"}}]}

    _CALL = _call.__func__(1)

    def _first(self, **fields):
        return _response(self._P1, self._S1, message=self._CALL, **fields)

    def _second(self, cached, kept=6999, **fields):  # a patched server's response 2
        return _response(self._P2[kept:], [7, _OBS], cached=cached, prompt_tokens=9011,
                         prompt_token_ids_offset=kept, message=self._call(2), **fields)

    def _turn(self, model, context=None):
        """Sample 1, its call's result: the context that sample 2 extends."""
        context = context or _context()
        context.extend(_sample(model, "token_ids", context).context_items())
        context.append(ToolResult("call_1", "ok"))
        return context


class ChainTests(ChainFixture, unittest.TestCase):
    def test_a_request_that_extends_the_latest_sample_sends_its_tokens_back(self):
        server = _Server(self._first(), self._second(5632))
        model = _model(server)
        context = self._turn(model)
        traced = _sample(model, "token_ids", context)
        first, second = server.requests
        self.assertNotIn("prompt_token_ids", first)
        self.assertNotIn("token_ids", first)
        # Exactly as returned: the prompt whole, the sample with its stop.
        self.assertEqual((second["prompt_token_ids"], second["token_ids"]), (self._P1, self._S1))
        record = traced.tokens
        self.assertEqual((record.prefix_offset, record.prefix_gap_token_ids, record.prefix_token_ids),
                         (5632, None, TokenIds(self._P2[5632:])))
        self.assertEqual(record.prefix_sha256, prompt_sha256(TokenIds(self._P2)))
        context.extend(traced.context_items())
        chain = TokenChain()
        rebuilt = [chain.add(item.tokens, item.usage.input_tokens) for item in context
                   if isinstance(item, SampleMetadata)]
        self.assertEqual(rebuilt[1], (TokenIds(self._P2), None))

    def test_a_retry_that_hits_past_the_kept_ids_keeps_a_gap(self):
        # 2a fails after the server's prefill, which leaves a state at 8,704.
        server = _Server(self._first(), (503, {"error": "busy"}), self._second(8704))
        model = _model(server)
        context = self._turn(model)
        params = SampleParams(token_trace="token_ids", transient_retries=0)
        with self.assertRaises(ModelTransportError):
            model.sample(context, sample_params=params)
        record = model.sample(context, sample_params=params).tokens
        # The retry sends the same IDs.
        self.assertEqual(server.requests[1]["prompt_token_ids"], server.requests[2]["prompt_token_ids"])
        self.assertEqual((record.prefix_offset, record.prefix_gap_token_ids, len(record.prefix_token_ids)),
                         (8704, TokenIds(self._P2[6999:8704]), 307))

    def test_nothing_is_sent_without_a_chain_or_after_a_change(self):
        tool = ToolSpec("exec_command", "Run a command.", {"type": "object"})
        cases = {
            # The latest record has no prefix IDs: nothing to send.
            "no prefix": (self._first(prompt_tokens=6100, **{}), lambda context: context),
            "a compaction": (self._first(), lambda context: InteractionContext(
                (*context, ContextPrefix((Message("user", "summary"),)), Message("user", "next")))),
            "other instructions": (self._first(), None),
            "another model": (dict(self._first(), model="another"), lambda context: context),
        }
        for name, (first, change) in cases.items():
            with self.subTest(name=name):
                if name == "no prefix":
                    first = dict(first)
                    del first["prompt_token_ids"]
                server = _Server(first, _response(self._P2, [7], cached=5632))
                model = _model(server)
                if name == "other instructions":
                    context = self._turn(model)
                    from pythia.interaction import Instructions
                    context.append(Instructions("Be brief."))
                else:
                    context = change(self._turn(model))
                _sample(model, "token_ids", context)
                self.assertNotIn("prompt_token_ids", server.requests[1])
        # Other tools than the last traced request's.
        server = _Server(self._first(), _response(self._P2, [7], cached=5632))
        model = _model(server)
        context = self._turn(model)
        model.sample(context, tools=(tool,), sample_params=SampleParams(token_trace="token_ids"))
        self.assertNotIn("prompt_token_ids", server.requests[1])

    def test_a_server_that_ignores_them_gets_them_no_more(self):
        # Stock vLLM: no prompt_token_ids_offset, and the whole prompt.
        server = _Server(self._first(), _response(self._P2, [7], cached=5632, message=self._call(2)),
                         _response([*self._P2, 7, *_ids(60000, 50)], [8], cached=8960))
        model = _model(server)
        context = self._turn(model)
        traced = _sample(model, "token_ids", context)
        self.assertIn("prompt_token_ids", server.requests[1])
        self.assertEqual(traced.notices, (NOTICES["ignored_ids"],))
        # Decoded as without them: the whole prompt, the gap by comparison.
        self.assertEqual((traced.tokens.prefix_offset, traced.tokens.prefix_gap_token_ids),
                         (5632, None))
        context.extend(traced.context_items())
        context.append(ToolResult("call_2", "ok"))
        self.assertEqual(_sample(model, "token_ids", context).notices, ())
        self.assertNotIn("prompt_token_ids", server.requests[2])

    def test_a_server_that_refuses_them_gets_the_request_again_without_them(self):
        refusal = {"error": {"message": "Unrecognized field: prompt_token_ids", "type": "BadRequest"}}
        for raised in (False, True):
            with self.subTest(raised=raised):
                server = _Server(self._first(), (400, refusal),
                                 _response(self._P2, [7], cached=5632, message=self._call(2)),
                                 _response([*self._P2, 7, *_ids(60000, 50)], [8], cached=8960))
                opener = server
                if raised:
                    def opener(request, timeout, server=server):
                        response = server(request, timeout)
                        if response.status == 400:
                            import io
                            import urllib.error
                            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {},
                                                         io.BytesIO(response.read()))
                        return response
                model = ChatCompletionsModel(chat_endpoint(model=_MODEL), opener=opener)
                context = self._turn(model)
                traced = _sample(model, "token_ids", context)
                self.assertIn("prompt_token_ids", server.requests[1])
                self.assertNotIn("prompt_token_ids", server.requests[2])
                self.assertEqual(traced.recovery, ("token_ids_refused",))
                self.assertEqual(traced.request_attempts, 2)
                self.assertEqual(traced.notices, (NOTICES["refused_ids"],))
                context.extend(traced.context_items())
                context.append(ToolResult("call_2", "ok"))
                _sample(model, "token_ids", context)
                self.assertNotIn("prompt_token_ids", server.requests[3])

    def test_a_resumed_session_sends_the_tokens_its_save_holds(self):
        import tempfile
        from pathlib import Path
        from pythia.interaction import load_interaction_save, save_interaction_save
        context = self._turn(_model(_Server(self._first())))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            save_interaction_save(path, context)
            resumed = load_interaction_save(path)
        server = _Server(self._second(5632))
        traced = _sample(_model(server), "token_ids", resumed)
        self.assertEqual((server.requests[0]["prompt_token_ids"], server.requests[0]["token_ids"]),
                         (self._P1, self._S1))
        self.assertEqual(traced.tokens.prefix_sha256, prompt_sha256(TokenIds(self._P2)))

    def test_the_kept_ids_must_be_all_or_all_but_the_stop(self):
        server = _Server(self._first(), self._second(5632, kept=6998))
        model = _model(server)
        traced = _sample(model, "token_ids", self._turn(model))
        self.assertIsNone(traced.tokens.prefix_token_ids)
        self.assertEqual(traced.notices, (NOTICES["bad_prompt_ids"],))


class TurnLoopTests(unittest.TestCase):
    def test_the_turn_loop_shows_a_samples_notices_after_it(self):
        server = _Server(_response(None, [7]), _response(None, [8]))
        model, host = _model(server), _Host()
        context = _context()
        run_turn(context, model, _environment(host), _config(endpoint_token_trace="token_ids"), host)
        self.assertEqual(host.notices, [NOTICES["no_prompt_ids"]])
        [metadata] = [item for item in context if isinstance(item, SampleMetadata)]
        self.assertEqual(metadata.tokens, TokenRecord(model=_MODEL, sample_token_ids=[7]))


if __name__ == "__main__":
    unittest.main()
