from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Union

from .context import ModelContext
from .items import ContextCompaction
from .items import InteractionItem
from .items import Instructions
from .items import Message
from .items import ModelSampleBoundary
from .items import OpaqueCompaction
from .items import Reasoning
from .items import SessionInit
from .items import ToolCall
from .items import ToolResult
from .items import TurnMetadata
from .items import TurnSummary
from .items import UserInteractionBoundary
from .items import UserToolCall
from .items import UserToolResult
from .usage import TokenUsage


SessionPath = Union[str, os.PathLike]


class SessionError(ValueError):
    pass


_ITEM_TYPES = {
    SessionInit: "session_init",
    Instructions: "instructions",
    Message: "message",
    Reasoning: "reasoning",
    ToolCall: "tool_call",
    ToolResult: "tool_result",
    UserToolCall: "user_tool_call",
    UserToolResult: "user_tool_result",
    ModelSampleBoundary: "model_sample_boundary",
    TurnMetadata: "turn_metadata",
    TurnSummary: "turn_summary",
    UserInteractionBoundary: "user_interaction_boundary",
    OpaqueCompaction: "opaque_compaction",
    ContextCompaction: "context_compaction",
}
_ITEM_TYPE_NAMES = frozenset(_ITEM_TYPES.values())


def _item_type_name(item: InteractionItem) -> str:
    item_type = type(item)
    if item_type not in _ITEM_TYPES:
        raise SessionError(
            f"cannot encode interaction item type {item_type.__name__}"
        )
    return _ITEM_TYPES[item_type]


def interaction_item_to_dict(item: InteractionItem) -> Dict[str, Any]:
    """Encode an interaction item as a JSON-compatible dictionary."""
    encoded: Dict[str, Any] = {"type": _item_type_name(item)}

    if isinstance(item, UserToolCall):
        encoded["call"] = interaction_item_to_dict(item.call)
    elif isinstance(item, UserToolResult):
        encoded["result"] = interaction_item_to_dict(item.result)
    elif isinstance(item, Instructions):
        encoded.update(text=item.text)
    elif isinstance(item, Message):
        encoded.update(role=item.role, text=item.text)
    elif isinstance(item, SessionInit):
        encoded["session_id"] = item.session_id
    elif isinstance(item, Reasoning):
        encoded.update(text=item.text, summary=list(item.summary))
        if item.encrypted_content is not None:
            encoded["encrypted_content"] = item.encrypted_content
        if item.content_signature is not None:
            encoded["content_signature"] = item.content_signature
    elif isinstance(item, ToolCall):
        encoded.update(
            name=item.name,
            call_id=item.call_id,
            arguments_json=item.arguments_json,
        )
    elif isinstance(item, ToolResult):
        encoded.update(
            call_id=item.call_id,
            output=item.output,
            success=item.success,
        )
    elif isinstance(item, OpaqueCompaction):
        encoded.update(protocol=item.protocol, payload=item.payload)
    elif isinstance(item, ContextCompaction):
        encoded["replacement_items"] = [
            interaction_item_to_dict(nested)
            for nested in item.replacement_items
        ]
    elif isinstance(item, TurnMetadata):
        encoded["usage"] = {
            "input_tokens": item.usage.input_tokens,
            "output_tokens": item.usage.output_tokens,
            "total_tokens": item.usage.total_tokens,
            "cached_input_tokens": item.usage.cached_input_tokens,
        }
        for field_name in (
            "provider_session_id",
            "provider_turn_id",
            "provider_turn_state",
        ):
            value = getattr(item, field_name)
            if value is not None:
                encoded[field_name] = value
    elif isinstance(item, TurnSummary):
        encoded.update(
            input_tokens_sum=item.input_tokens_sum,
            output_tokens_sum=item.output_tokens_sum,
            cached_input_tokens_sum=item.cached_input_tokens_sum,
            cached_input_tokens_max=item.cached_input_tokens_max,
            non_cached_input_tokens_sum=item.non_cached_input_tokens_sum,
            context_tokens=item.context_tokens,
            sample_count=item.sample_count,
            compaction_count=item.compaction_count,
        )
    elif isinstance(
        item,
        (ModelSampleBoundary, UserInteractionBoundary),
    ):
        pass
    else:
        raise SessionError(
            f"cannot encode interaction item type {type(item).__name__}"
        )

    return encoded


def _require_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SessionError("interaction item must be an object")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise SessionError(f"{field_name} must be a string")
    return value


def _optional_string(
    mapping: Mapping[str, Any],
    key: str,
    field_name: str,
) -> Optional[str]:
    value = mapping.get(key)
    if value is None:
        return None
    return _require_string(value, field_name)


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SessionError(f"{field_name} must be a nonnegative integer")
    return value


def interaction_item_from_dict(value: Any) -> InteractionItem:
    """Decode a dictionary produced by ``interaction_item_to_dict``."""
    mapping = _require_mapping(value)
    item_type = mapping.get("type")
    if not isinstance(item_type, str) or item_type not in _ITEM_TYPE_NAMES:
        raise SessionError(f"unknown interaction item type: {item_type!r}")

    if item_type == "user_tool_call":
        return UserToolCall(interaction_item_from_dict(mapping.get("call")))
    if item_type == "user_tool_result":
        return UserToolResult(interaction_item_from_dict(mapping.get("result")))
    if item_type == "instructions":
        return Instructions(
            text=_require_string(mapping.get("text"), "instructions.text"),
        )
    if item_type == "message":
        return Message(
            role=_require_string(mapping.get("role"), "message.role"),
            text=_require_string(mapping.get("text"), "message.text"),
        )
    if item_type == "session_init":
        return SessionInit(
            session_id=_require_string(
                mapping.get("session_id"),
                "session_init.session_id",
            )
        )
    if item_type == "reasoning":
        summary_value = mapping.get("summary", ())
        if not isinstance(summary_value, list):
            raise SessionError("reasoning.summary must be a list")
        return Reasoning(
            text=_require_string(mapping.get("text"), "reasoning.text"),
            summary=[
                _require_string(value, f"reasoning.summary[{index}]")
                for index, value in enumerate(summary_value)
            ],
            encrypted_content=_optional_string(
                mapping,
                "encrypted_content",
                "reasoning.encrypted_content",
            ),
            content_signature=_optional_string(
                mapping,
                "content_signature",
                "reasoning.content_signature",
            ),
        )
    if item_type == "tool_call":
        return ToolCall(
            name=_require_string(mapping.get("name"), "tool_call.name"),
            call_id=_require_string(
                mapping.get("call_id"),
                "tool_call.call_id",
            ),
            arguments_json=_require_string(
                mapping.get("arguments_json"),
                "tool_call.arguments_json",
            ),
        )
    if item_type == "tool_result":
        success = mapping.get("success", True)
        if not isinstance(success, bool):
            raise SessionError("tool_result.success must be a boolean")
        return ToolResult(
            call_id=_require_string(
                mapping.get("call_id"),
                "tool_result.call_id",
            ),
            output=_require_string(
                mapping.get("output"),
                "tool_result.output",
            ),
            success=success,
        )
    if item_type == "opaque_compaction":
        # Existing session files stored only Responses encrypted content.
        if "protocol" not in mapping and "payload" not in mapping:
            return OpaqueCompaction.from_responses(
                _require_string(
                    mapping.get("encrypted_content"),
                    "opaque_compaction.encrypted_content",
                )
            )
        protocol = _require_string(
            mapping.get("protocol"),
            "opaque_compaction.protocol",
        )
        return OpaqueCompaction(
            payload=_require_string(
                mapping.get("payload"),
                "opaque_compaction.payload",
            ),
            protocol=protocol,
        )
    if item_type == "context_compaction":
        replacement = mapping.get("replacement_items")
        if not isinstance(replacement, list):
            raise SessionError(
                "context_compaction.replacement_items must be a list"
            )
        return ContextCompaction(
            tuple(interaction_item_from_dict(item) for item in replacement)
        )
    if item_type == "turn_metadata":
        usage = mapping.get("usage")
        if not isinstance(usage, Mapping):
            raise SessionError("turn_metadata.usage must be an object")
        return TurnMetadata(
            usage=TokenUsage(
                input_tokens=_require_nonnegative_int(
                    usage.get("input_tokens"),
                    "turn_metadata.usage.input_tokens",
                ),
                output_tokens=_require_nonnegative_int(
                    usage.get("output_tokens"),
                    "turn_metadata.usage.output_tokens",
                ),
                total_tokens=_require_nonnegative_int(
                    usage.get("total_tokens"),
                    "turn_metadata.usage.total_tokens",
                ),
                cached_input_tokens=_require_nonnegative_int(
                    usage.get("cached_input_tokens"),
                    "turn_metadata.usage.cached_input_tokens",
                ),
            ),
            provider_session_id=_optional_string(
                mapping,
                "provider_session_id",
                "turn_metadata.provider_session_id",
            ),
            provider_turn_id=_optional_string(
                mapping,
                "provider_turn_id",
                "turn_metadata.provider_turn_id",
            ),
            provider_turn_state=_optional_string(
                mapping,
                "provider_turn_state",
                "turn_metadata.provider_turn_state",
            ),
        )
    if item_type == "turn_summary":
        return TurnSummary(
            input_tokens_sum=_require_nonnegative_int(
                mapping.get("input_tokens_sum"),
                "turn_summary.input_tokens_sum",
            ),
            output_tokens_sum=_require_nonnegative_int(
                mapping.get("output_tokens_sum"),
                "turn_summary.output_tokens_sum",
            ),
            cached_input_tokens_sum=_require_nonnegative_int(
                mapping.get("cached_input_tokens_sum"),
                "turn_summary.cached_input_tokens_sum",
            ),
            cached_input_tokens_max=_require_nonnegative_int(
                mapping.get("cached_input_tokens_max"),
                "turn_summary.cached_input_tokens_max",
            ),
            non_cached_input_tokens_sum=_require_nonnegative_int(
                mapping.get("non_cached_input_tokens_sum"),
                "turn_summary.non_cached_input_tokens_sum",
            ),
            context_tokens=_require_nonnegative_int(
                mapping.get("context_tokens"),
                "turn_summary.context_tokens",
            ),
            sample_count=_require_nonnegative_int(
                mapping.get("sample_count"),
                "turn_summary.sample_count",
            ),
            compaction_count=_require_nonnegative_int(
                mapping.get("compaction_count"),
                "turn_summary.compaction_count",
            ),
        )
    if item_type == "model_sample_boundary":
        return ModelSampleBoundary()
    if item_type == "user_interaction_boundary":
        return UserInteractionBoundary()

    raise SessionError(f"unknown interaction item type: {item_type!r}")


def iter_interaction_items(
    items: Iterable[InteractionItem],
) -> Iterator[Dict[str, Any]]:
    for item in items:
        yield interaction_item_to_dict(item)


def save_interaction_session(path: SessionPath, context: ModelContext) -> None:
    """Atomically write a context as one JSON interaction item per line."""
    destination = Path(path)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            for encoded in iter_interaction_items(context.items):
                temporary.write(
                    json.dumps(
                        encoded,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    except OSError as exc:
        raise SessionError(
            f"could not save session to {destination}: {exc}"
        ) from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def load_interaction_session(path: SessionPath) -> ModelContext:
    """Load and validate a JSONL interaction-session file."""
    source = Path(path)
    items: list[InteractionItem] = []
    try:
        with source.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SessionError(
                        f"invalid JSON in session {source} at line "
                        f"{line_number}: {exc}"
                    ) from exc

                try:
                    items.append(interaction_item_from_dict(value))
                except (SessionError, TypeError, ValueError) as exc:
                    raise SessionError(
                        f"invalid session item in {source} at line "
                        f"{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise SessionError(f"could not load session {source}: {exc}") from exc

    try:
        return ModelContext(items)
    except (TypeError, ValueError) as exc:
        raise SessionError(f"invalid session {source}: {exc}") from exc


__all__ = [
    "SessionError",
    "interaction_item_from_dict",
    "interaction_item_to_dict",
    "iter_interaction_items",
    "load_interaction_session",
    "save_interaction_session",
]
