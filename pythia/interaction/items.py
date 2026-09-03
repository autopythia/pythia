from __future__ import annotations

import uuid
from dataclasses import dataclass
from dataclasses import field
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

from .usage import TokenUsage


def _require_string(value: object, field_name: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _fresh_session_id() -> str:
    return f"session_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class SessionInit:
    """Durable identity established before an interaction begins."""

    session_id: str = field(default_factory=_fresh_session_id)

    def __post_init__(self) -> None:
        _require_string(self.session_id, "session_id", allow_empty=False)
        if "\r" in self.session_id or "\n" in self.session_id:
            raise ValueError("session_id must not contain newlines")


@dataclass(frozen=True)
class Message:
    role: str
    text: str

    def __post_init__(self) -> None:
        _require_string(self.role, "role", allow_empty=False)
        _require_string(self.text, "text")


@dataclass(frozen=True)
class Instructions:
    """Optional persistent system instructions.

    Corresponds to a Chat Completions ``system`` message. Empty and
    whitespace-only text is supported; ``no instructions`` is represented
    only by the absence of an ``Instructions`` item. When several
    ``Instructions`` items appear in a log, the last one overrides all
    earlier ones (see ``ModelContext.model_items``).
    """

    text: str

    def __post_init__(self) -> None:
        _require_string(self.text, "text")


@dataclass(frozen=True)
class Reasoning:
    text: str
    summary: Tuple[str, ...] = ()
    encrypted_content: Optional[str] = field(default=None, repr=False)
    content_signature: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _require_string(self.text, "text")
        summary = tuple(self.summary)
        for index, value in enumerate(summary):
            _require_string(value, f"summary[{index}]")
        object.__setattr__(self, "summary", summary)
        if self.encrypted_content is not None:
            _require_string(
                self.encrypted_content,
                "encrypted_content",
                allow_empty=False,
            )
        if self.content_signature is not None:
            _require_string(
                self.content_signature,
                "content_signature",
                allow_empty=False,
            )


@dataclass(frozen=True)
class ToolCall:
    name: str
    call_id: str
    arguments_json: str

    def __post_init__(self) -> None:
        _require_string(self.name, "name", allow_empty=False)
        _require_string(self.call_id, "call_id", allow_empty=False)
        _require_string(self.arguments_json, "arguments_json")


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    output: str
    success: bool = True

    def __post_init__(self) -> None:
        _require_string(self.call_id, "call_id", allow_empty=False)
        _require_string(self.output, "output")
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")


@dataclass(frozen=True)
class ModelSampleBoundary:
    """Marks the end of one model sample without emitting provider content."""


@dataclass(frozen=True)
class UserInteractionBoundary:
    """Marks the end of one user interaction without emitting provider content."""


@dataclass(frozen=True)
class TurnMetadata:
    """Durable per-sample provider usage result.

    One ``TurnMetadata`` is recorded per completed model sample
    (see ``ModelSample.context_items``). It preserves the provider's
    ``TokenUsage`` for that sample plus the provider continuity tokens.
    It is *not* a cumulative end-of-turn aggregate; see ``TurnSummary``.
    """

    usage: TokenUsage
    provider_session_id: Optional[str] = field(default=None, repr=False)
    provider_turn_id: Optional[str] = field(default=None, repr=False)
    provider_turn_state: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")
        for field_name in (
            "provider_session_id",
            "provider_turn_id",
            "provider_turn_state",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            _require_string(value, field_name, allow_empty=False)
            if "\r" in value or "\n" in value:
                raise ValueError(f"{field_name} must not contain newlines")


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class TurnSummary:
    """Cumulative end-of-turn usage derived from per-sample ``TurnMetadata``.

    Ports the earlier autopythia/contradex ``AgentState`` accounting
    (``output_tokens_sum``, ``cache_hit_*`` warm stats, ``non_cache_hit`` cold
    stats, ``total_usage_tokens`` context, ``compaction_count``) to
    ``pythia.interaction`` without changing ``TurnMetadata`` semantics.

    ``TurnSummary`` is encoder-transparent (never sent to the provider) and
    durable. It is derived via :func:`summarize_turn_usage`, which folds over
    the raw log's ``TurnMetadata`` items and counts compaction markers.
    Existing ``TurnSummary`` items are skipped by the fold so re-summarizing
    a context that already contains summaries does not double-count.
    """

    input_tokens_sum: int = 0
    output_tokens_sum: int = 0
    cached_input_tokens_sum: int = 0
    cached_input_tokens_max: int = 0
    non_cached_input_tokens_sum: int = 0
    context_tokens: int = 0
    sample_count: int = 0
    compaction_count: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens_sum",
            "output_tokens_sum",
            "cached_input_tokens_sum",
            "cached_input_tokens_max",
            "non_cached_input_tokens_sum",
            "context_tokens",
            "sample_count",
            "compaction_count",
        ):
            _require_nonnegative_int(getattr(self, field_name), field_name)

    @property
    def goal_accounting_tokens(self) -> int:
        """Billable tokens: cold (non-cached) input + output."""
        return self.non_cached_input_tokens_sum + self.output_tokens_sum

    @property
    def has_detailed_usage(self) -> bool:
        """Whether any folded sample carried nonzero usage."""
        return bool(
            self.input_tokens_sum
            or self.output_tokens_sum
            or self.cached_input_tokens_sum
            or self.cached_input_tokens_max
            or self.non_cached_input_tokens_sum
            or self.sample_count
        )


def summarize_turn_usage(items) -> "TurnSummary":
    """Fold per-sample ``TurnMetadata`` into a cumulative ``TurnSummary``.

    Mirrors ``contradex.kernel.AgentState.add_response_usage``:
    warm per sample is ``min(cached, input)``, cold is ``max(0, input-warm)``.
    ``context_tokens`` tracks the last sample's ``total_tokens`` (contradex
    ``total_usage_tokens`` semantics: current window, not a sum).
    ``compaction_count`` counts ``OpaqueCompaction``/``ContextCompaction``
    markers. ``TurnSummary`` items in the input are skipped.

    Pass the raw log (``context.items``) for session-cumulative stats, or a
    slice after the last ``UserInteractionBoundary`` for per-turn stats.
    """
    input_tokens_sum = 0
    output_tokens_sum = 0
    cached_input_tokens_sum = 0
    cached_input_tokens_max = 0
    non_cached_input_tokens_sum = 0
    context_tokens = 0
    sample_count = 0
    compaction_count = 0
    for item in items:
        if isinstance(item, TurnMetadata):
            usage = item.usage
            warm = min(usage.cached_input_tokens, usage.input_tokens)
            cold = max(0, usage.input_tokens - warm)
            input_tokens_sum += usage.input_tokens
            output_tokens_sum += usage.output_tokens
            cached_input_tokens_sum += warm
            cached_input_tokens_max = max(cached_input_tokens_max, warm)
            non_cached_input_tokens_sum += cold
            context_tokens = usage.total_tokens
            sample_count += 1
        elif isinstance(item, (OpaqueCompaction, ContextCompaction)):
            compaction_count += 1
        elif isinstance(item, TurnSummary):
            continue
    return TurnSummary(
        input_tokens_sum=input_tokens_sum,
        output_tokens_sum=output_tokens_sum,
        cached_input_tokens_sum=cached_input_tokens_sum,
        cached_input_tokens_max=cached_input_tokens_max,
        non_cached_input_tokens_sum=non_cached_input_tokens_sum,
        context_tokens=context_tokens,
        sample_count=sample_count,
        compaction_count=compaction_count,
    )


@dataclass(frozen=True)
class OpaqueCompaction:
    payload: str = field(repr=False)
    protocol: Literal["responses", "messages"] = "responses"

    def __post_init__(self) -> None:
        _require_string(
            self.payload,
            "payload",
            allow_empty=False,
        )
        if not isinstance(self.protocol, str):
            raise TypeError("protocol must be a string")
        if self.protocol not in {"responses", "messages"}:
            raise ValueError(
                "protocol must be 'responses' or 'messages'"
            )

    @classmethod
    def from_responses(cls, encrypted_content: str) -> "OpaqueCompaction":
        return cls(payload=encrypted_content, protocol="responses")

    @classmethod
    def from_messages(cls, content: str) -> "OpaqueCompaction":
        return cls(payload=content, protocol="messages")


@dataclass(frozen=True)
class ContextCompaction:
    replacement_items: Tuple["InteractionItem", ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "replacement_items", tuple(self.replacement_items))


InteractionItem = Union[
    SessionInit,
    Instructions,
    Message,
    Reasoning,
    ToolCall,
    ToolResult,
    ModelSampleBoundary,
    TurnMetadata,
    TurnSummary,
    UserInteractionBoundary,
    OpaqueCompaction,
    ContextCompaction,
]

INTERACTION_ITEM_TYPES = (
    SessionInit,
    Instructions,
    Message,
    Reasoning,
    ToolCall,
    ToolResult,
    ModelSampleBoundary,
    TurnMetadata,
    TurnSummary,
    UserInteractionBoundary,
    OpaqueCompaction,
    ContextCompaction,
)


def is_interaction_item(value: object) -> bool:
    return isinstance(value, INTERACTION_ITEM_TYPES)


__all__ = [
    "ContextCompaction",
    "Instructions",
    "InteractionItem",
    "Message",
    "ModelSampleBoundary",
    "OpaqueCompaction",
    "Reasoning",
    "SessionInit",
    "ToolCall",
    "ToolResult",
    "TurnMetadata",
    "TurnSummary",
    "UserInteractionBoundary",
    "is_interaction_item",
    "summarize_turn_usage",
]
