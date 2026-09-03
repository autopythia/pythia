from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Tuple

from .context import ModelContext
from .items import InteractionItem
from .items import Message
from .items import ModelSampleBoundary
from .items import OpaqueCompaction
from .items import Reasoning
from .items import ToolCall
from .items import TurnMetadata
from .usage import TokenUsage

if TYPE_CHECKING:
    from .display import DisplayItem
    from .environment import ToolSpec


class ModelError(RuntimeError):
    pass


class ModelConfigurationError(ModelError, ValueError):
    pass


class ModelTransportError(ModelError):
    pass


class ModelTimeoutError(ModelTransportError):
    pass


class ModelResponseError(ModelError):
    pass


class ModelContextWindowError(ModelResponseError):
    pass


def _validate_optional_finite_number(
    value: object,
    field_name: str,
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric or None")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


@dataclass(frozen=True)
class SamplingOptions:
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Tuple[str, ...] = ()
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_tokens is not None:
            if (
                isinstance(self.max_tokens, bool)
                or not isinstance(self.max_tokens, int)
                or self.max_tokens <= 0
            ):
                raise ValueError("max_tokens must be a positive integer or None")

        temperature = _validate_optional_finite_number(
            self.temperature,
            "temperature",
        )
        if temperature is not None and temperature < 0:
            raise ValueError("temperature must be nonnegative")
        object.__setattr__(self, "temperature", temperature)

        top_p = _validate_optional_finite_number(self.top_p, "top_p")
        if top_p is not None and not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        object.__setattr__(self, "top_p", top_p)

        stop = tuple(self.stop)
        for index, value in enumerate(stop):
            if not isinstance(value, str):
                raise TypeError(f"stop[{index}] must be a string")
        object.__setattr__(self, "stop", stop)

        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")


@dataclass(frozen=True)
class ModelSample:
    items: Tuple[InteractionItem, ...]
    stop_reason: Optional[str] = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    provider_session_id: Optional[str] = field(default=None, repr=False)
    provider_turn_id: Optional[str] = field(default=None, repr=False)
    provider_turn_state: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if not items:
            raise ModelResponseError("model sample must contain at least one item")
        for index, item in enumerate(items):
            if not isinstance(
                item,
                (Message, Reasoning, ToolCall, OpaqueCompaction),
            ):
                raise ModelResponseError(
                    "model sample items must be assistant messages, reasoning, "
                    "tool calls, or opaque compactions; "
                    f"item {index} is {type(item).__name__}"
                )
            if isinstance(item, Message) and item.role != "assistant":
                raise ModelResponseError(
                    f"model message at item {index} must have role 'assistant'"
                )
        object.__setattr__(self, "items", items)

        if self.stop_reason is not None and not isinstance(self.stop_reason, str):
            raise TypeError("stop_reason must be a string or None")
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
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None")
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
            if "\r" in value or "\n" in value:
                raise ValueError(f"{field_name} must not contain newlines")

    @property
    def tool_calls(self) -> Tuple[ToolCall, ...]:
        return tuple(item for item in self.items if isinstance(item, ToolCall))

    def context_items(self) -> Tuple[InteractionItem, ...]:
        """Return this sample's output, turn metadata, and durable boundary."""
        return (
            *self.items,
            TurnMetadata(
                usage=self.usage,
                provider_session_id=self.provider_session_id,
                provider_turn_id=self.provider_turn_id,
                provider_turn_state=self.provider_turn_state,
            ),
            ModelSampleBoundary(),
        )

    def display_items(self) -> Tuple["DisplayItem", ...]:
        from .display import render_interaction_items

        return render_interaction_items(
            (*self.items, TurnMetadata(usage=self.usage)),
        )

    @property
    def assistant_messages(self) -> Tuple[Message, ...]:
        return tuple(item for item in self.items if isinstance(item, Message))

    @property
    def last_assistant_text(self) -> Optional[str]:
        for item in reversed(self.items):
            if isinstance(item, Message) and item.text:
                return item.text
        return None


class Model(Protocol):
    def sample(
        self,
        context: ModelContext,
        *,
        tools: Sequence["ToolSpec"] = (),
        options: Optional[SamplingOptions] = None,
    ) -> ModelSample:
        ...


__all__ = [
    "Model",
    "ModelConfigurationError",
    "ModelContextWindowError",
    "ModelError",
    "ModelResponseError",
    "ModelSample",
    "ModelTimeoutError",
    "ModelTransportError",
    "SamplingOptions",
    "TokenUsage",
]
