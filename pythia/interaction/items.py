from __future__ import annotations

import uuid
from dataclasses import dataclass
from dataclasses import field
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
class Reasoning:
    text: str
    summary: Tuple[str, ...] = ()
    encrypted_content: Optional[str] = field(default=None, repr=False)

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
    """Durable, non-provider control metadata for one completed turn."""

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


@dataclass(frozen=True)
class OpaqueCompaction:
    encrypted_content: str

    def __post_init__(self) -> None:
        _require_string(
            self.encrypted_content,
            "encrypted_content",
            allow_empty=False,
        )


@dataclass(frozen=True)
class ContextCompaction:
    replacement_items: Tuple["InteractionItem", ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "replacement_items", tuple(self.replacement_items))


InteractionItem = Union[
    SessionInit,
    Message,
    Reasoning,
    ToolCall,
    ToolResult,
    ModelSampleBoundary,
    TurnMetadata,
    UserInteractionBoundary,
    OpaqueCompaction,
    ContextCompaction,
]

INTERACTION_ITEM_TYPES = (
    SessionInit,
    Message,
    Reasoning,
    ToolCall,
    ToolResult,
    ModelSampleBoundary,
    TurnMetadata,
    UserInteractionBoundary,
    OpaqueCompaction,
    ContextCompaction,
)


def is_interaction_item(value: object) -> bool:
    return isinstance(value, INTERACTION_ITEM_TYPES)


__all__ = [
    "ContextCompaction",
    "InteractionItem",
    "Message",
    "ModelSampleBoundary",
    "OpaqueCompaction",
    "Reasoning",
    "SessionInit",
    "ToolCall",
    "ToolResult",
    "TurnMetadata",
    "UserInteractionBoundary",
    "is_interaction_item",
]
