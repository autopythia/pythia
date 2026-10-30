"""Token traces: a sample's token IDs and log probs, as a save keeps them.

A Chat Completions model whose catalog entry sets ``endpoint.token_trace``
asks a vLLM server for the IDs of the tokens that it processed and sampled,
and for their log probs. A sample's ``SampleMetadata`` keeps them as a
:class:`TokenRecord`, its ``tokens`` field, which no model request includes
and the public log leaves out. (A token trace is not the debug trace, whose
files hold requests and responses.)

A record has two parts, the prompt (``prefix_*``) and the sample
(``sample_*``):

- ``prefix_offset`` is where the server's computation of the prompt began,
  its prefix-cache hit, and ``prefix_token_ids`` are the prompt's IDs from
  there to its end: the tokens processed in the request. So consecutive
  records overlap by what the server computed again, and the part of a
  prompt below its offset is in the records before it.
- ``prefix_gap_token_ids`` are the prompt's tokens below the offset that the
  save doesn't hold yet. That happens when the server's cache reached past
  what the earlier records share with the prompt, as a request that left no
  record can leave it (a failed or cancelled one, or another session's).
- ``prefix_log_probs`` and ``sample_log_probs`` each score the token at the
  same index, given every token before it in the whole sequence. The
  prefix's first value is NaN: the logits before it came from the cache,
  or, at position 0, don't exist.
- ``sample_token_ids`` are the sampled IDs, its stop included.
  ``sample_stop`` counts the sample's last tokens that are its stop, and
  ``prefix_generation_prompt`` the prompt's last tokens that are the
  template's generation prompt.

Arrays are packed, little-endian int32 IDs and float32 log probs, 4 bytes a
value, in memory as in saves, which write them in base64; only readers
decode them. :class:`TokenChain` rebuilds each record's whole prompt from
the records before it, and checks it against the sample's length and the
record's hash; :func:`traced_samples` does that for a save's samples, and
:func:`segments` joins them into training sequences.

``python3 -m pythia.interaction.tokens SAVE`` writes a save's token traces as
JSONL, a line per record and a line per segment (``__main__``).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import re
import struct
import sys
from array import array
from dataclasses import dataclass
from dataclasses import fields
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union


# The largest token ID: IDs are packed as int32.
MAX_TOKEN_ID = 2**31 - 1
# prefix_sha256 holds this many hex digits of the SHA-256.
SHA256_DIGITS = 16

_SHA256_RE = re.compile(rf"^[0-9a-f]{{{SHA256_DIGITS}}}$")
_INT32 = next(code for code in "il" if array(code).itemsize == 4)
_FLOAT32 = "f"
_FLOAT32_MAX = struct.unpack("<f", b"\xff\xff\x7f\x7f")[0]
_SWAP = sys.byteorder == "big"  # arrays are packed little-endian
_SHOWN_VALUES = 8  # a repr shows the values of an array up to this long


class _Packed:
    """An immutable array of 4-byte values, packed little-endian."""

    __slots__ = ("_data",)
    _code = ""  # the array type code
    _struct = ""  # the struct format of one value
    _what = ""  # what a value is, in errors

    def __init__(self, values: Iterable = ()) -> None:
        if isinstance(values, (bytes, bytearray, memoryview, str)):
            raise TypeError(f"{type(self).__name__} takes values; "
                            f"{type(self).__name__}.from_bytes takes packed data")
        packed = array(self._code, self._checked(values))
        if _SWAP:
            packed.byteswap()
        self._data = packed.tobytes()

    @staticmethod
    def _checked(values: Iterable) -> List:
        raise NotImplementedError

    @classmethod
    def from_bytes(cls, data: Union[bytes, bytearray, memoryview]):
        """The array that ``data`` packs. Its values aren't checked one by one."""
        data = bytes(data)
        if len(data) % 4:
            raise ValueError(f"packed {cls._what}s take 4 bytes each, "
                             f"not {len(data)} bytes in all")
        packed = cls.__new__(cls)
        packed._data = data
        return packed

    @classmethod
    def from_base64(cls, text: str):
        """The array that ``text``, standard base64 with padding, packs."""
        if not isinstance(text, str):
            raise TypeError("packed values must be a base64 string")
        try:
            data = base64.b64decode(text.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error):
            raise ValueError("packed values must be base64") from None
        return cls.from_bytes(data)

    def to_bytes(self) -> bytes:
        return self._data

    def to_base64(self) -> str:
        return base64.b64encode(self._data).decode("ascii")

    def _values(self) -> array:
        values = array(self._code)
        values.frombytes(self._data)
        if _SWAP:
            values.byteswap()
        return values

    def tolist(self) -> List:
        return self._values().tolist()

    def __len__(self) -> int:
        return len(self._data) // 4

    def __iter__(self) -> Iterator:
        return iter(self._values())

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return type(self).from_bytes(self._data[4 * start:4 * max(start, stop)])
            return type(self)(self._values()[index])
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"{type(self).__name__} indices must be integers or slices")
        count = len(self)
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise IndexError(f"{type(self).__name__} index out of range")
        return struct.unpack_from(self._struct, self._data, 4 * index)[0]

    def __add__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return type(self).from_bytes(self._data + other._data)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._data == other._data

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._data))

    def __repr__(self) -> str:
        if len(self) <= _SHOWN_VALUES:
            return f"{type(self).__name__}({self.tolist()!r})"
        return f"{type(self).__name__}(<{len(self)} values>)"


class TokenIds(_Packed):
    """Token IDs, from 0 to ``MAX_TOKEN_ID``, packed as little-endian int32."""

    __slots__ = ()
    _code = _INT32
    _struct = "<i"
    _what = "token ID"

    @staticmethod
    def _checked(values: Iterable) -> List[int]:
        ids = list(values)
        for index, value in enumerate(ids):
            if (isinstance(value, bool) or not isinstance(value, int)
                    or not 0 <= value <= MAX_TOKEN_ID):
                raise ValueError(
                    f"token ID [{index}] must be an integer from 0 to {MAX_TOKEN_ID}"
                )
        return ids


class LogProbs(_Packed):
    """Log probs, packed as little-endian float32.

    vLLM computes them in float32, so this keeps its values exactly. NaN
    stands for a value that isn't there, and vLLM's −9999.0 is kept as sent.
    """

    __slots__ = ()
    _code = _FLOAT32
    _struct = "<f"
    _what = "log prob"

    @staticmethod
    def _checked(values: Iterable) -> List[float]:
        probs = list(values)
        for index, value in enumerate(probs):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"log prob [{index}] must be a number")
            try:
                number = float(value)
            except OverflowError:  # an integer beyond every float
                raise ValueError(f"log prob [{index}] is beyond float32's range") from None
            if math.isfinite(number) and abs(number) > _FLOAT32_MAX:
                raise ValueError(f"log prob [{index}] is beyond float32's range")
        return probs


def prompt_sha256(token_ids: TokenIds) -> str:
    """A whole prompt's ``prefix_sha256``: the first ``SHA256_DIGITS`` hex
    digits of the SHA-256 of its packed IDs."""
    if not isinstance(token_ids, TokenIds):
        raise TypeError("token_ids must be TokenIds")
    return hashlib.sha256(token_ids.to_bytes()).hexdigest()[:SHA256_DIGITS]


def common_prefix_length(first: TokenIds, second: TokenIds) -> int:
    """How many tokens two arrays share from their start."""
    if not isinstance(first, TokenIds) or not isinstance(second, TokenIds):
        raise TypeError("common_prefix_length takes TokenIds")
    a, b = memoryview(first.to_bytes()), memoryview(second.to_bytes())
    low, high = 0, min(len(first), len(second))  # a and b share low tokens; high is a bound
    while low < high:
        middle = (low + high + 1) // 2
        if a[4 * low:4 * middle] == b[4 * low:4 * middle]:
            low = middle
        else:
            high = middle - 1
    return low


_IDS_FIELDS = ("prefix_gap_token_ids", "prefix_token_ids", "sample_token_ids")
_LOG_PROBS_FIELDS = ("prefix_log_probs", "sample_log_probs")
_COUNT_FIELDS = ("preemptions", "prefix_offset", "prefix_generation_prompt", "sample_stop")
_TEXT_FIELDS = ("model", "log_probs_mode")
_PREFIX_PARTS = ("prefix_offset", "prefix_sha256", "prefix_gap_token_ids", "prefix_log_probs")
_SAMPLE_PARTS = ("sample_log_probs", "sample_stop")


@dataclass(frozen=True)
class TokenRecord:
    """One sample's token trace: the ``tokens`` of its ``SampleMetadata``.

    Every field is optional: a part that the server didn't send is None.
    The prompt's parts need ``prefix_token_ids``, and the sample's
    ``sample_token_ids``. Arrays may be given as values, which are checked,
    or as :class:`TokenIds` and :class:`LogProbs`, which aren't.

    - ``model``: the response's ``model``, the served model's name.
    - ``log_probs_mode``: the server's ``--logprobs-mode`` as its response
      says it (``raw_logprobs`` and the like); None when it doesn't.
    - ``preemptions``: how often the server preempted the request, when it
      did (then ``prefix_offset`` may differ from the sample's cached
      tokens).
    - ``prefix_offset``: where the server's computation of the prompt began.
    - ``prefix_sha256``: the whole prompt's hash (:func:`prompt_sha256`).
    - ``prefix_gap_token_ids``: the prompt's tokens below ``prefix_offset``
      that no earlier record holds; never empty.
    - ``prefix_token_ids``: the prompt's IDs from ``prefix_offset`` on, and
      ``prefix_log_probs``, one for each of them, the first NaN.
    - ``prefix_generation_prompt``: the prompt's last tokens that are the
      template's generation prompt.
    - ``sample_token_ids``: the sampled IDs, its stop included, and
      ``sample_log_probs``, one for each of them.
    - ``sample_stop``: the sample's last tokens that are its stop.
    """

    model: Optional[str] = None
    log_probs_mode: Optional[str] = None
    preemptions: Optional[int] = None
    prefix_offset: Optional[int] = None
    prefix_sha256: Optional[str] = None
    prefix_gap_token_ids: Optional[TokenIds] = None
    prefix_token_ids: Optional[TokenIds] = None
    prefix_log_probs: Optional[LogProbs] = None
    prefix_generation_prompt: Optional[int] = None
    sample_token_ids: Optional[TokenIds] = None
    sample_log_probs: Optional[LogProbs] = None
    sample_stop: Optional[int] = None

    def __post_init__(self) -> None:
        for name in _TEXT_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
            if not value.strip() or "\r" in value or "\n" in value:
                raise ValueError(f"{name} must be a nonempty line")
        for name in _COUNT_FIELDS:
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                      or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        for names, kind in ((_IDS_FIELDS, TokenIds), (_LOG_PROBS_FIELDS, LogProbs)):
            for name in names:
                value = getattr(self, name)
                if value is None or isinstance(value, kind):
                    continue
                try:
                    object.__setattr__(self, name, kind(value))
                except (TypeError, ValueError) as exc:
                    raise type(exc)(f"{name}: {exc}") from None
        if self.prefix_gap_token_ids is not None and not self.prefix_gap_token_ids:
            object.__setattr__(self, "prefix_gap_token_ids", None)
        digest = self.prefix_sha256
        if digest is not None and (not isinstance(digest, str) or not _SHA256_RE.match(digest)):
            raise ValueError(f"prefix_sha256 must be {SHA256_DIGITS} lowercase hex digits")

        prefix = self.prefix_token_ids
        if prefix is None:
            for name in _PREFIX_PARTS:
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} needs prefix_token_ids")
        else:
            if self.prefix_offset is None:
                raise ValueError("prefix_token_ids needs prefix_offset")
            if self.prefix_log_probs is not None and len(self.prefix_log_probs) != len(prefix):
                raise ValueError(
                    f"prefix_log_probs has {len(self.prefix_log_probs)} values "
                    f"for {len(prefix)} prefix_token_ids"
                )
            gap = self.prefix_gap_token_ids
            if gap is not None and len(gap) > self.prefix_offset:
                raise ValueError(
                    f"prefix_gap_token_ids has {len(gap)} tokens, "
                    f"more than the {self.prefix_offset} below prefix_offset"
                )
            generation = self.prefix_generation_prompt
            if generation is not None and generation > self.prompt_length:
                raise ValueError(
                    f"prefix_generation_prompt is {generation} tokens, "
                    f"more than the prompt's {self.prompt_length}"
                )

        sample = self.sample_token_ids
        if sample is None:
            for name in _SAMPLE_PARTS:
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} needs sample_token_ids")
        else:
            if self.sample_log_probs is not None and len(self.sample_log_probs) != len(sample):
                raise ValueError(
                    f"sample_log_probs has {len(self.sample_log_probs)} values "
                    f"for {len(sample)} sample_token_ids"
                )
            if self.sample_stop is not None and self.sample_stop > len(sample):
                raise ValueError(
                    f"sample_stop is {self.sample_stop} tokens, "
                    f"more than the sample's {len(sample)}"
                )

    @property
    def prompt_length(self) -> Optional[int]:
        """The whole prompt's length, when the record has its prefix IDs."""
        if self.prefix_token_ids is None:
            return None
        return self.prefix_offset + len(self.prefix_token_ids)


# The record's fields, in the order its JSON object has them.
TOKEN_RECORD_FIELDS = tuple(item.name for item in fields(TokenRecord))


def token_record_to_dict(record: TokenRecord) -> Dict[str, Any]:
    """A record as a save writes it: its fields that aren't None, arrays in base64."""
    if not isinstance(record, TokenRecord):
        raise TypeError("record must be TokenRecord")
    encoded: Dict[str, Any] = {}
    for name in TOKEN_RECORD_FIELDS:
        value = getattr(record, name)
        if value is not None:
            encoded[name] = value.to_base64() if isinstance(value, _Packed) else value
    return encoded


def token_record_from_dict(value: Mapping[str, Any]) -> TokenRecord:
    """The record that :func:`token_record_to_dict` wrote.

    A null field counts as absent, and keys it doesn't know are ignored, as
    a later version may add some. Errors start with the field's name.
    """
    if not isinstance(value, Mapping):
        raise TypeError("a token record must be an object")
    arguments: Dict[str, Any] = {}
    for name in TOKEN_RECORD_FIELDS:
        raw = value.get(name)
        if raw is None:
            continue
        if name in _IDS_FIELDS or name in _LOG_PROBS_FIELDS:
            kind = TokenIds if name in _IDS_FIELDS else LogProbs
            try:
                arguments[name] = kind.from_base64(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{name} must be base64 of packed 4-byte values"
                ) from None
        elif name in _COUNT_FIELDS:
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            arguments[name] = raw
        else:
            if not isinstance(raw, str):
                raise ValueError(f"{name} must be a string")
            arguments[name] = raw
    return TokenRecord(**arguments)


class Rebuilt(NamedTuple):
    """A record's whole prompt as :class:`TokenChain` rebuilt it, or why it
    couldn't (``problem``)."""

    prompt: Optional[TokenIds]
    problem: Optional[str] = None


class TokenChain:
    """Rebuilds each record's whole prompt from the records before it.

    Give it every sample's record in the save's order, or None for a sample
    without one (:meth:`add`). The tokens before a record are the previous
    record's whole prompt, as rebuilt, followed by its sample. A record's
    prompt is their first m tokens, then its gap, then its prefix IDs, m
    being its ``prefix_offset`` less the gap's length; its length must be
    the sample's input tokens, and its hash ``prefix_sha256``.

    A record that can't be rebuilt ends the chain, and so does one without
    sample IDs, after its own prompt. The next record then starts a new one,
    which it can only if it holds its whole prompt: a ``prefix_offset`` of
    0, or a gap from 0. So does a record of another ``model``.
    """

    def __init__(self) -> None:
        self._tokens: Optional[bytes] = None  # packed: the tokens before the next record
        self._model: Optional[str] = None

    @property
    def tokens(self) -> Optional[TokenIds]:
        """The tokens before the next record, or None when the chain has ended."""
        return None if self._tokens is None else TokenIds.from_bytes(self._tokens)

    def add(self, record: Optional[TokenRecord], input_tokens: Optional[int] = None,
            *, check_hash: bool = True) -> Rebuilt:
        """Rebuild the next sample's prompt from ``record`` and the records before it.

        ``input_tokens`` is the sample's ``usage.input_tokens``, which the
        prompt's length must equal; None skips that check. ``check_hash``
        False skips the hash check: the length check stays, and a prompt
        that follows a wrong one is then most likely wrong too, which the
        next check of a hash finds.
        """
        tokens, model = self._tokens, self._model
        self._tokens = self._model = None  # until this record is rebuilt
        if record is None:
            return Rebuilt(None, "the sample has no token record")
        if not isinstance(record, TokenRecord):
            raise TypeError("record must be TokenRecord or None")
        if record.prefix_token_ids is None:
            return Rebuilt(None, "the record has no prefix_token_ids")
        if tokens is not None and model != record.model:
            tokens = None  # another model's tokens: a new chain
        gap = b"" if record.prefix_gap_token_ids is None else record.prefix_gap_token_ids.to_bytes()
        held = record.prefix_offset - len(gap) // 4
        available = 0 if tokens is None else len(tokens) // 4
        if held > available:
            if tokens is None:
                return Rebuilt(None, f"its first {held} tokens are in no earlier record")
            return Rebuilt(None, f"its first {held} tokens pass the {available} "
                                 "that the earlier records hold")
        prompt = (tokens[:4 * held] if held else b"") + gap + record.prefix_token_ids.to_bytes()
        length = len(prompt) // 4
        if input_tokens is not None and length != input_tokens:
            return Rebuilt(None, f"the rebuilt prompt has {length} tokens, "
                                 f"where the sample's input has {input_tokens}")
        rebuilt = TokenIds.from_bytes(prompt)
        if (check_hash and record.prefix_sha256 is not None
                and prompt_sha256(rebuilt) != record.prefix_sha256):
            return Rebuilt(None, "the rebuilt prompt doesn't match prefix_sha256")
        if record.sample_token_ids is not None:
            self._tokens = prompt + record.sample_token_ids.to_bytes()
            self._model = record.model
        return Rebuilt(rebuilt)


class TracedSample(NamedTuple):
    """A sample of a save, as :func:`traced_samples` reads it: its metadata's
    line in the save (from 1), its record (None without one), its input
    tokens, and its whole prompt, rebuilt, or why it couldn't be."""

    line: int
    record: Optional[TokenRecord]
    input_tokens: int
    prompt: Optional[TokenIds]
    problem: Optional[str]


def traced_samples(items: Sequence[Any]) -> Iterator[TracedSample]:
    """Each sample of a save's items (its ``SampleMetadata``), in order, with
    its prompt rebuilt from the records before it (:class:`TokenChain`)."""
    from ..items import SampleMetadata

    chain = TokenChain()
    for index, item in enumerate(items):
        if isinstance(item, SampleMetadata):
            rebuilt = chain.add(item.tokens, item.usage.input_tokens or None)
            yield TracedSample(index + 1, item.tokens, item.usage.input_tokens,
                               rebuilt.prompt, rebuilt.problem)


class Segment(NamedTuple):
    """A run of samples whose prompts each start with the sample before it,
    its prompt and its sampled tokens: one sequence, its last prompt and
    sample. ``sampled`` are the spans of it that were sampled, as
    ``(start, end)``, stops included and the template's generation prompts
    not; a sampled stop that the next prompt holds another token in place of
    is left out. ``log_probs`` holds a sampled token's log prob at its
    position, NaN elsewhere."""

    samples: Tuple[TracedSample, ...]
    token_ids: TokenIds
    sampled: Tuple[Tuple[int, int], ...]
    log_probs: LogProbs


def _whole(sample: TracedSample) -> Optional[TokenIds]:
    """A sample's prompt followed by its sampled tokens, when both are known."""
    if sample.prompt is None or sample.record is None or sample.record.sample_token_ids is None:
        return None
    return sample.prompt + sample.record.sample_token_ids


def _continues(previous: TracedSample, sample: TracedSample) -> bool:
    """Whether a prompt starts with the sample before it, less a stop that
    the template wrote otherwise (``sample_stop``, else one token)."""
    tokens, prompt = _whole(previous).to_bytes(), sample.prompt.to_bytes()
    if prompt.startswith(tokens):
        return True
    stop = previous.record.sample_stop
    stop = 1 if stop is None else stop
    return 0 < stop <= len(previous.record.sample_token_ids) and prompt.startswith(
        tokens[:len(tokens) - 4 * stop])


def _segment(samples: Sequence[TracedSample]) -> Segment:
    last = samples[-1]
    final = _whole(last)
    tokens = final.tolist()
    values = [math.nan] * len(tokens)
    marked = [False] * len(tokens)
    for sample in samples:
        start = len(sample.prompt)
        ids = sample.record.sample_token_ids.tolist()
        probs = (None if sample.record.sample_log_probs is None
                 else sample.record.sample_log_probs.tolist())
        for offset, token in enumerate(ids):
            position = start + offset
            if position < len(tokens) and tokens[position] == token:
                marked[position] = True
                if probs is not None:
                    values[position] = probs[offset]
    spans: List[Tuple[int, int]] = []
    for position, is_sampled in enumerate(marked):
        if is_sampled and spans and spans[-1][1] == position:
            spans[-1] = (spans[-1][0], position + 1)
        elif is_sampled:
            spans.append((position, position + 1))
    return Segment(tuple(samples), final, tuple(spans), LogProbs(values))


def segments(samples: Iterable[TracedSample]) -> Iterator[Segment]:
    """The runs of ``samples`` (from :func:`traced_samples`) that make one
    sequence each. A sample without a rebuilt prompt or sampled IDs ends a
    run, and so does a prompt that doesn't start with the sample before it,
    as after a compaction or a change of instructions, tools or options, or
    without token IDs sent, a prompt rendered otherwise. A gap ends none."""
    run: List[TracedSample] = []
    for sample in samples:
        usable = _whole(sample) is not None
        if run and usable and _continues(run[-1], sample):
            run.append(sample)
            continue
        if run:
            yield _segment(run)
        run = [sample] if usable else []
    if run:
        yield _segment(run)


__all__ = [
    "LogProbs",
    "MAX_TOKEN_ID",
    "Rebuilt",
    "SHA256_DIGITS",
    "Segment",
    "TOKEN_RECORD_FIELDS",
    "TokenChain",
    "TokenIds",
    "TokenRecord",
    "TracedSample",
    "common_prefix_length",
    "prompt_sha256",
    "segments",
    "token_record_from_dict",
    "token_record_to_dict",
    "traced_samples",
]
