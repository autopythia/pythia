"""How a Chat Completions model asks a vLLM server for a token trace, and
makes the response's token fields into a record (``tokens.TokenRecord``).

A level (``endpoint.token_trace``, ``SampleParams.token_trace``) adds request
fields, each level those of the one before:

- ``token_ids``: ``return_token_ids``, for the prompt's IDs
  (``prompt_token_ids``) and the sample's (``choices[0].token_ids``);
- ``sample_log_probs``: ``logprobs`` with ``top_logprobs`` 0, and
  ``return_tokens_as_token_ids``, so that each entry names its token's ID;
- ``prefix_log_probs``: ``prompt_logprobs`` 0, and ``skip_reading_prefix_cache``
  false, which a server with the fork's patch 0057 reads: its prompt log probs
  then keep the prefix cache, starting at ``prompt_logprobs_offset``.

A part of the response that's missing or fails a check is left out of the
record, with a notice once per model; the sample still succeeds. The record's
prompt starts where the server's processing began (``prefix_offset``: the
response's ``prompt_logprobs_offset``, else its cached tokens), and the
prompt's tokens below that which no earlier record holds are its gap.

While a request extends the latest sample (``extends_latest_sample``), it
also sends that sample's prompt and the sample back, exactly as the server
returned them, as ``prompt_token_ids`` and ``token_ids``: a server with the
fork's patch 0056 then uses them as they are, and renders and tokenizes only
the messages after them. Its response's ``prompt_token_ids`` start at
``prompt_token_ids_offset``, the IDs that it kept of those sent.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

from .items import SampleMetadata
from .model_catalog import TOKEN_TRACE_FIELDS
from .tokens import MAX_TOKEN_ID
from .tokens import LogProbs
from .tokens import TokenChain
from .tokens import TokenIds
from .tokens import TokenRecord
from .tokens import common_prefix_length
from .tokens import prompt_sha256
from .usage import TokenUsage

# The levels in order; each asks for what the one before does, and more.
_ORDER = {"token_ids": 1, "sample_log_probs": 2, "prefix_log_probs": 3}
_TOKEN_ID_PREFIX = "token_id:"  # return_tokens_as_token_ids
# The request fields that don't change how a server renders the prompt: the
# rest of a request (its model, tools and extensions) must stay the same
# for the previous tokens to stand for the messages that they replace.
_UNRENDERED_FIELDS = frozenset(("messages", "stream", "max_tokens", "temperature", "top_p",
                                "stop", "seed")) | TOKEN_TRACE_FIELDS


def level_of(level: Optional[str]) -> int:
    """A level's rank: 0 for None and none."""
    return _ORDER.get(level, 0)


def request_fields(level: Optional[str]) -> Dict[str, Any]:
    """The request fields that ``level`` adds (none for None and ``none``)."""
    rank = level_of(level)
    fields: Dict[str, Any] = {}
    if rank >= 1:
        fields["return_token_ids"] = True
    if rank >= 2:
        fields.update(logprobs=True, top_logprobs=0, return_tokens_as_token_ids=True)
    if rank >= 3:
        fields.update(prompt_logprobs=0, skip_reading_prefix_cache=False)
    return fields


def frame_of(payload: Mapping) -> str:
    """What of a request, besides its messages, the server renders its prompt with."""
    return json.dumps({key: value for key, value in payload.items()
                       if key not in _UNRENDERED_FIELDS}, sort_keys=True, ensure_ascii=False)


# Notices, once per model each: what a response lacked, or what failed a check.
NOTICES = {
    "no_prompt_ids": "Token trace: the server sent no prompt_token_ids, so samples keep no "
                     "prompt tokens (it needs vLLM 0.10.2 or later).",
    "bad_prompt_ids": "Token trace: the server's prompt_token_ids failed a check (token IDs "
                      "that add up to its prompt_tokens); a sample's prompt tokens were left out.",
    "bad_offset": "Token trace: the server's cache hit or prompt_logprobs_offset passed the "
                  "prompt's end; a sample's prompt tokens were left out.",
    "no_cached_tokens": "Token trace: the server reports no cached tokens (vLLM's "
                        "--enable-prompt-tokens-details), so a record's prompt starts where the "
                        "save's tokens end, not where the server's processing began.",
    "no_sample_ids": "Token trace: the server sent no token_ids for the sample (it needs vLLM "
                     "0.10.2 or later, or log probs, whose entries name their tokens).",
    "bad_sample_ids": "Token trace: the server's token_ids for the sample aren't token IDs; "
                      "they were left out.",
    "no_log_probs": "Token trace: the server sent no log probs for the sampled tokens.",
    "bad_log_probs": "Token trace: the server's log probs for the sampled tokens failed a check "
                     "(a finite value for each sampled ID, which its entry names); they were "
                     "left out.",
    "no_prompt_log_probs": "Token trace: the server sent no prompt_logprobs.",
    "bad_prompt_log_probs": "Token trace: the server's prompt_logprobs failed a check (an entry "
                            "for each processed prompt token, keyed by its ID, the first null); "
                            "they were left out.",
    "unplaced_prompt_log_probs": "Token trace: prompt log probs after a cache hit came without "
                                 "prompt_logprobs_offset, so their positions are unknown; they "
                                 "were left out (the server needs the fork's patch 0057).",
    "bad_counts": "Token trace: the server's num_generation_prompt_tokens or num_stop_tokens "
                  "didn't fit its token IDs; they were left out.",
    "ignored_ids": "Token trace: the server ignored the previous tokens sent back "
                   "(prompt_token_ids and token_ids; it needs the fork's patch 0056), so this "
                   "session sends them no more.",
    "refused_ids": "Token trace: the server refused the previous tokens sent back "
                   "(prompt_token_ids and token_ids, HTTP 400); the request went again without "
                   "them, and this session sends them no more.",
}


class Decoded(NamedTuple):
    """A response's token trace: its record, the whole prompt (for the chain),
    and the notices' keys (``NOTICES``) of the parts left out."""

    record: TokenRecord
    prompt: Optional[TokenIds]
    problems: Tuple[str, ...]


def _token_ids(value: Any) -> Optional[List[int]]:
    """A list of token IDs, or None when ``value`` isn't one."""
    if not isinstance(value, list):
        return None
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= MAX_TOKEN_ID:
            return None
    return value


def _log_prob(value: Any) -> Optional[float]:
    """A finite log prob (vLLM's −9999.0 included), or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _count(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _text(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip() or "\r" in value or "\n" in value:
        return None
    return value


def _entry_token_id(entry: Mapping) -> Optional[int]:
    """The ID that a ``logprobs.content`` entry names (``token_id:N``)."""
    token = entry.get("token")
    if not isinstance(token, str) or not token.startswith(_TOKEN_ID_PREFIX):
        return None
    digits = token[len(_TOKEN_ID_PREFIX):]
    if not digits.isdigit() or not digits.isascii():
        return None
    value = int(digits)
    return value if value <= MAX_TOKEN_ID else None


def _sample_parts(choice: Mapping, rank: int, problems: List[str]):
    """The sample's IDs and log probs, as lists, each None when left out."""
    ids = None
    raw = choice.get("token_ids")
    if raw is not None:
        ids = _token_ids(raw)
        if ids is None:
            problems.append("bad_sample_ids")
    entries = None
    if rank >= 2:
        logprobs = choice.get("logprobs")
        content = logprobs.get("content") if isinstance(logprobs, Mapping) else None
        if content is None:
            problems.append("no_log_probs")
        elif not isinstance(content, list) or not all(isinstance(e, Mapping) for e in content):
            problems.append("bad_log_probs")
        else:
            entries = content
    if ids is None and raw is None:
        # vLLM before 0.10.2: the entries name the sampled IDs.
        named = None if entries is None else [_entry_token_id(entry) for entry in entries]
        if named is not None and None not in named:
            ids = named
        else:
            problems.append("no_sample_ids")
    probs = None
    if entries is not None and ids is not None:
        values = [_log_prob(entry.get("logprob")) for entry in entries]
        if (len(entries) == len(ids) and None not in values
                and all(_entry_token_id(entry) == token for entry, token in zip(entries, ids))):
            probs = values
        else:
            problems.append("bad_log_probs")
    return ids, probs


def _prompt_log_probs(raw: Any, prompt: TokenIds, start: int,
                      problems: List[str]) -> Optional[List[float]]:
    """The prompt's log probs from ``start`` on: one per entry, NaN for the
    first; None when left out."""
    if not isinstance(raw, list) or len(raw) != len(prompt) - start or not raw or raw[0] is not None:
        problems.append("bad_prompt_log_probs")
        return None
    scored_ids = prompt[start:].tolist()
    values = [math.nan]
    for token, entry in zip(scored_ids[1:], raw[1:]):
        scored = entry.get(str(token)) if isinstance(entry, Mapping) else None
        value = _log_prob(scored.get("logprob")) if isinstance(scored, Mapping) else None
        if value is None:
            problems.append("bad_prompt_log_probs")
            return None
        values.append(value)
    return values


def _cached_tokens(payload: Mapping) -> Optional[int]:
    """The cached tokens that the response's usage reports, or None when it
    reports none (a vLLM server without --enable-prompt-tokens-details)."""
    usage = payload.get("usage")
    details = usage.get("prompt_tokens_details") if isinstance(usage, Mapping) else None
    return _count(details.get("cached_tokens")) if isinstance(details, Mapping) else None


def decode(payload: Mapping, level: Optional[str], usage: TokenUsage,
           before: Optional[TokenIds] = None, before_model: Optional[str] = None,
           sent: Optional[Tuple[TokenIds, TokenIds]] = None) -> Decoded:
    """A response's token trace, at ``level``.

    ``before`` is the tokens before this sample, as the save holds them (the
    previous record's whole prompt and its sample; ``TokenChain``), of the
    model ``before_model``; None when there are none, which a save's first
    record and a record after the chain ended have. The prompt's tokens below
    the cache hit that ``before`` doesn't hold are the record's gap.

    ``sent`` is the previous prompt and sample that the request sent back,
    if it did. The whole prompt is then the IDs that the server kept of them
    (``prompt_token_ids_offset``: all, or all but a stop that it dropped),
    then the IDs that it returned, and the save holds the kept ones. A
    response without that offset came from a server that ignored them.

    The record's ``prefix_offset`` is where the server's processing began: its
    ``prompt_logprobs_offset``; else 0 when its prompt log probs cover the
    whole prompt, as stock vLLM's do, which read no cache; else its cached
    tokens. A server that reports none gets the tokens that ``before`` holds
    instead, with a notice.
    """
    rank = level_of(level)
    problems: List[str] = []
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, Mapping):
        choice = {}
    model = _text(payload.get("model"))
    mode = _text(payload.get("logprobs_mode"))
    preemptions = _count(payload.get("num_preemptions")) or None

    prompt = None
    raw = payload.get("prompt_token_ids")
    kept = None if sent is None else _count(payload.get("prompt_token_ids_offset"))
    if sent is not None and kept is None:
        problems.append("ignored_ids")
    if raw is None:
        problems.append("no_prompt_ids")
    else:
        returned = _token_ids(raw)
        if returned is not None and kept is not None:
            ids = sent[0] + sent[1]
            prompt = (ids[:kept] + TokenIds(returned)
                      if kept in (len(ids), len(ids) - 1) else None)
        elif returned is not None:
            prompt = TokenIds(returned)
        if prompt is None or (usage.input_tokens and len(prompt) != usage.input_tokens):
            problems.append("bad_prompt_ids")
            prompt = None

    prefix: Dict[str, Any] = {}
    whole = None
    if prompt is not None:
        whole = prompt
        held = 0
        if kept is not None:
            held = kept  # the server took them as they are
        elif before is not None and (before_model is None or model is None
                                     or before_model == model):
            held = common_prefix_length(whole, before)
        placed = _count(payload.get("prompt_logprobs_offset"))
        listed = payload.get("prompt_logprobs") if rank >= 3 else None
        cached = _cached_tokens(payload)
        if placed is not None:
            offset = placed
        elif isinstance(listed, list) and len(listed) == len(prompt):
            offset = 0  # log probs for the whole prompt: it was all processed
        elif cached is not None:
            offset = cached
        else:
            offset = held
            problems.append("no_cached_tokens")
        if offset > len(prompt):
            problems.append("bad_offset")
            whole = None
    if whole is not None:
        prefix = dict(prefix_offset=offset, prefix_sha256=prompt_sha256(whole),
                      prefix_token_ids=whole[offset:])
        if offset > held:
            prefix["prefix_gap_token_ids"] = whole[held:offset]
        if rank >= 3:
            if listed is None:
                problems.append("no_prompt_log_probs")
            elif placed is None and offset > 0:
                problems.append("unplaced_prompt_log_probs")
            else:
                probs = _prompt_log_probs(listed, whole, offset, problems)
                if probs is not None:
                    prefix["prefix_log_probs"] = probs
        generation = payload.get("num_generation_prompt_tokens")
        if generation is not None:
            if _count(generation) is not None and generation <= len(prompt):
                prefix["prefix_generation_prompt"] = generation
            else:
                problems.append("bad_counts")

    ids, probs = _sample_parts(choice, rank, problems)
    sample: Dict[str, Any] = {}
    if ids is not None:
        sample["sample_token_ids"] = ids
        if probs is not None:
            sample["sample_log_probs"] = probs
        stop = choice.get("num_stop_tokens")
        if stop is not None:
            if _count(stop) is not None and stop <= len(ids):
                sample["sample_stop"] = stop
            else:
                problems.append("bad_counts")
    record = TokenRecord(model=model, log_probs_mode=mode, preemptions=preemptions,
                         **prefix, **sample)
    return Decoded(record, whole, tuple(dict.fromkeys(problems)))


class TraceChain:
    """The tokens before a model's next sample, for its gap and to send back:
    the context's latest record's whole prompt and its sample.

    It is rebuilt from the context's records (``TokenChain``) when the latest
    isn't the one it last saw, as after a resume, and otherwise kept from the
    model's own last sample.
    """

    def __init__(self) -> None:
        self._record: Optional[TokenRecord] = None  # what the kept tokens end with
        self._prompt: Optional[TokenIds] = None  # its whole prompt, if rebuilt

    def _latest(self, context) -> Optional[TokenRecord]:
        """The context's latest record, with its prompt rebuilt; None when the
        latest sample has none."""
        latest = next((item for item in reversed(context) if isinstance(item, SampleMetadata)),
                      None)
        record = None if latest is None else latest.tokens
        if record is not None and record is not self._record:
            records = [item for item in context if isinstance(item, SampleMetadata)]
            chain = TokenChain()
            rebuilt = None
            for index, item in enumerate(records):
                # The last one's hash checks the rebuild of every earlier one.
                rebuilt = chain.add(item.tokens, item.usage.input_tokens or None,
                                    check_hash=index == len(records) - 1).prompt
            self._record, self._prompt = record, rebuilt
        return record

    def before(self, context) -> Tuple[Optional[TokenIds], Optional[str]]:
        """The tokens before the next sample of ``context``, and their model."""
        previous = self.previous(context)
        if previous is None:
            return None, None
        prompt, sample, model = previous
        return prompt + sample, model

    def previous(self, context) -> Optional[Tuple[TokenIds, TokenIds, Optional[str]]]:
        """The latest record's whole prompt, its sample, and its model, or None
        when the save doesn't hold both."""
        record = self._latest(context)
        if record is None or self._prompt is None or record.sample_token_ids is None:
            return None
        return self._prompt, record.sample_token_ids, record.model

    def after(self, decoded: Decoded) -> None:
        """Keep a sample's tokens, for the one after it."""
        self._record, self._prompt = decoded.record, decoded.prompt


def extends_latest_sample(context, messages: Sequence[Mapping],
                          encode: Any) -> bool:
    """Whether a request's ``messages`` extend the context's latest sample:
    they start with the messages through its assistant message, as
    ``encode`` makes them from the context's items up to it, and go on.
    A compaction since, or other instructions, make them start otherwise."""
    items = context.items
    index = next((position for position in range(len(items) - 1, -1, -1)
                  if isinstance(items[position], SampleMetadata)), None)
    if index is None:
        return False
    from .context import InteractionContext
    try:
        head = encode(InteractionContext(items[:index]).model_items())
    except Exception:
        return False
    return (bool(head) and head[-1].get("role") == "assistant"
            and len(messages) > len(head) and list(messages[:len(head)]) == head)


__all__ = ["Decoded", "NOTICES", "TraceChain", "decode", "extends_latest_sample", "frame_of",
           "level_of", "request_fields"]
