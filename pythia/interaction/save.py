from __future__ import annotations

import errno
import io
import json
import os
import tempfile
import threading
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import BinaryIO
from typing import Callable
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

from .context import InteractionContext
from .items import CompactionMetadata
from .items import ContentPart
from .items import ContextPrefix
from .items import Init
from .items import Instructions
from .items import InteractionItem
from .items import MediaPart
from .items import Message
from .items import ModelFailure
from .items import ModelSampleBoundary
from .items import OpaqueCompaction
from .items import Reasoning
from .items import ToolCall
from .items import ToolResult
from .items import Tools
from .items import SampleMetadata
from .items import TextPart
from .items import TurnSummary
from .items import UserInteractionBoundary
from .items import UserToolCall
from .items import UserToolResult
from .items import _validate_elapsed_seconds
from .tokens import token_record_from_dict
from .tokens import token_record_to_dict
from .usage import TokenUsage
from ._provider_text import json_text_escaped
from ._tool_spec import ToolSpec
from ._tool_spec import _copy_schema


SavePath = Union[str, os.PathLike]


class SaveError(ValueError):
    pass


class PublicLogError(SaveError):
    """A ``{"type": "nil"}`` line: the file is a public log, not a full log."""


_ITEM_TYPES = {
    Init: "init",
    Instructions: "instructions",
    Tools: "tools",
    Message: "message",
    Reasoning: "reasoning",
    ToolCall: "tool_call",
    ToolResult: "tool_result",
    UserToolCall: "user_tool_call",
    UserToolResult: "user_tool_result",
    ModelSampleBoundary: "model_sample_boundary",
    SampleMetadata: "sample_metadata",
    CompactionMetadata: "compaction_metadata",
    ModelFailure: "model_failure",
    TurnSummary: "turn_summary",
    UserInteractionBoundary: "user_interaction_boundary",
    OpaqueCompaction: "opaque_compaction",
    ContextPrefix: "context_prefix",
}
# Accept legacy records on read; new saves use only the canonical types above.
_ITEM_TYPE_NAMES = frozenset(_ITEM_TYPES.values()) | {
    "context_compaction",
    "session_init",
    "turn_metadata",
}


def _item_type_name(item: InteractionItem) -> str:
    item_type = type(item)
    if item_type not in _ITEM_TYPES:
        raise SaveError(
            f"cannot encode interaction item type {item_type.__name__}"
        )
    return _ITEM_TYPES[item_type]


def _content_part_to_dict(part: ContentPart) -> Dict[str, Any]:
    """Encode one in-memory content part using its own (non-wire) shape.

    The durable record mirrors the in-memory ``TextPart``/``MediaPart`` types;
    the Responses ``input_text``/``input_image`` mapping happens only when a
    request is assembled (see ``responses._encode_context_items``).
    """
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, MediaPart):
        return {"type": "media", "source_uri": part.source_uri}
    raise SaveError(f"cannot encode content part type {type(part).__name__}")


def interaction_item_to_dict(item: InteractionItem) -> Dict[str, Any]:
    """Encode an interaction item as a JSON-compatible dictionary."""
    encoded: Dict[str, Any] = {"type": _item_type_name(item)}

    if isinstance(item, UserToolCall):
        encoded["call"] = interaction_item_to_dict(item.call)
    elif isinstance(item, UserToolResult):
        encoded["result"] = interaction_item_to_dict(item.result)
    elif isinstance(item, Instructions):
        encoded.update(text=item.text)
    elif isinstance(item, Tools):
        encoded["specs"] = [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": _copy_schema(spec.parameters),
            }
            for spec in item.specs
        ]
    elif isinstance(item, Message):
        if isinstance(item.content, str):
            encoded.update(role=item.role, content=item.content)
        else:
            encoded.update(
                role=item.role,
                content=[_content_part_to_dict(part) for part in item.content],
            )
    elif isinstance(item, Init):
        encoded["prefix_id"] = item.prefix_id
        if item.model is not None:
            encoded["model"] = item.model
    elif isinstance(item, Reasoning):
        encoded.update(content=item.content, summary=list(item.summary))
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
        encoded.update(payload=item.payload, protocol=item.protocol)
    elif isinstance(item, ContextPrefix):
        encoded["prefix_items"] = [
            interaction_item_to_dict(nested)
            for nested in item.prefix_items
        ]
    elif isinstance(item, (SampleMetadata, CompactionMetadata)):
        encoded["usage"] = {
            "input_tokens": item.usage.input_tokens,
            "output_tokens": item.usage.output_tokens,
            "total_tokens": item.usage.total_tokens,
            "cached_input_tokens": item.usage.cached_input_tokens,
        }
        if isinstance(item, CompactionMetadata):
            encoded["protocol"] = item.protocol
        provider_fields = (
            "provider_session_id",
            "provider_turn_id",
            "provider_turn_state",
        )
        if isinstance(item, CompactionMetadata):
            provider_fields = (*provider_fields, "provider_response_id")
        for field_name in provider_fields:
            value = getattr(item, field_name)
            if value is not None:
                encoded[field_name] = value
        if item.elapsed_seconds is not None:
            encoded["elapsed_seconds"] = item.elapsed_seconds
        if item.request_attempts != 1:
            encoded["request_attempts"] = item.request_attempts
        if item.recovery:
            encoded["recovery"] = list(item.recovery)
        if isinstance(item, SampleMetadata) and item.tokens is not None:
            # Last: the arrays are long, and the public log drops them.
            encoded["tokens"] = token_record_to_dict(item.tokens)
    elif isinstance(item, ModelFailure):
        for field_name in (
            "category",
            "message",
            "provider",
            "model",
            "auth_source",
            "http_status",
            "request_id",
            "response_id",
            "cf_ray",
            "authorization_error",
            "auth_error_code",
            "error_code",
            "attempt_count",
            "event_count",
            "event_types",
            "completed_item_count",
            "last_event_type",
            "last_sequence_number",
            "recovery",
            "elapsed_seconds",
        ):
            value = getattr(item, field_name)
            if field_name in {"event_types", "recovery"}:
                if value:
                    encoded[field_name] = list(value)
            elif value is not None:
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
        if item.elapsed_seconds is not None:
            encoded["elapsed_seconds"] = item.elapsed_seconds
    elif isinstance(
        item,
        (ModelSampleBoundary, UserInteractionBoundary),
    ):
        pass
    else:
        raise SaveError(
            f"cannot encode interaction item type {type(item).__name__}"
        )

    return encoded


def _require_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SaveError("interaction item must be an object")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise SaveError(f"{field_name} must be a string")
    return value


def _require_content(mapping: Mapping[str, Any], item_type: str) -> str:
    """Read content or legacy text, rejecting invalid or conflicting fields."""
    if "content" not in mapping and "text" in mapping:
        return _require_string(mapping["text"], f"{item_type}.text")

    content = _require_string(mapping.get("content"), f"{item_type}.content")
    if "text" in mapping:
        legacy_text = _require_string(mapping["text"], f"{item_type}.text")
        if content != legacy_text:
            raise SaveError(
                f"{item_type}.content and {item_type}.text must match"
            )
    return content


def _require_init_prefix_id(mapping: Mapping[str, Any]) -> str:
    """Read the canonical prefix ID or its legacy session ID spelling."""
    if "prefix_id" not in mapping and "session_id" in mapping:
        return _require_string(mapping["session_id"], "init.session_id")

    prefix_id = _require_string(mapping.get("prefix_id"), "init.prefix_id")
    if "session_id" in mapping:
        session_id = _require_string(mapping["session_id"], "init.session_id")
        if prefix_id != session_id:
            raise SaveError("init.prefix_id and init.session_id must match")
    return prefix_id


def _message_content_parts(value: Any) -> Tuple[ContentPart, ...]:
    """Decode a durable content array into in-memory content parts."""
    parts = []
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise SaveError(f"message.content[{index}] must be an object")
        part_type = entry.get("type")
        if part_type == "text":
            parts.append(
                TextPart(
                    text=_require_string(
                        entry.get("text"),
                        f"message.content[{index}].text",
                    )
                )
            )
        elif part_type == "media":
            source_uri = entry.get("source_uri")
            if not isinstance(source_uri, str) or not source_uri:
                raise SaveError(
                    f"message.content[{index}].source_uri must be a "
                    "non-empty string"
                )
            parts.append(MediaPart(source_uri=source_uri))
        else:
            raise SaveError(
                f"unsupported message.content[{index}] type: {part_type!r}"
            )
    if not parts:
        raise SaveError("message.content list must not be empty")
    return tuple(parts)


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
        raise SaveError(f"{field_name} must be a nonnegative integer")
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SaveError(f"{field_name} must be a positive integer")
    return value


def _optional_http_status(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 100 <= value <= 599:
        raise SaveError("model_failure.http_status must be from 100 through 599")
    return value


def _optional_nonnegative_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    return _require_nonnegative_int(value, field_name)


def interaction_item_from_dict(value: Any) -> InteractionItem:
    """Decode a dictionary produced by ``interaction_item_to_dict``."""
    mapping = _require_mapping(value)
    item_type = mapping.get("type")
    if item_type == "nil":
        raise PublicLogError(
            '"nil" stands for an item that a public log leaves out, so this is not '
            "a full log (a CLI save keeps its full log in PREFIX._private/context.jsonl)"
        )
    if not isinstance(item_type, str) or item_type not in _ITEM_TYPE_NAMES:
        raise SaveError(f"unknown interaction item type: {item_type!r}")

    if item_type == "user_tool_call":
        return UserToolCall(interaction_item_from_dict(mapping.get("call")))
    if item_type == "user_tool_result":
        return UserToolResult(interaction_item_from_dict(mapping.get("result")))
    if item_type == "instructions":
        return Instructions(
            text=_require_string(mapping.get("text"), "instructions.text"),
        )
    if item_type == "tools":
        specs = mapping.get("specs")
        if not isinstance(specs, list):
            raise SaveError("tools.specs must be a list")
        decoded = []
        for index, spec in enumerate(specs):
            if not isinstance(spec, Mapping):
                raise SaveError(f"tools.specs[{index}] must be an object")
            try:
                decoded.append(ToolSpec(
                    name=spec.get("name"),
                    description=spec.get("description"),
                    parameters=spec.get("parameters"),
                ))
            except (TypeError, ValueError) as exc:
                raise SaveError(f"tools.specs[{index}]: {exc}") from exc
        try:
            return Tools(tuple(decoded))
        except (TypeError, ValueError) as exc:
            raise SaveError(f"tools.specs: {exc}") from exc
    if item_type == "message":
        role = _require_string(mapping.get("role"), "message.role")
        raw_content = mapping.get("content")
        if isinstance(raw_content, list):
            return Message(
                role=role,
                content=_message_content_parts(raw_content),
            )
        return Message(
            role=role,
            content=_require_content(mapping, "message"),
        )
    if item_type in {"init", "session_init"}:
        return Init(
            prefix_id=_require_init_prefix_id(mapping),
            model=_optional_string(mapping, "model", "init.model"),
        )
    if item_type == "reasoning":
        summary_value = mapping.get("summary", ())
        if not isinstance(summary_value, list):
            raise SaveError("reasoning.summary must be a list")
        return Reasoning(
            content=_require_content(mapping, "reasoning"),
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
            raise SaveError("tool_result.success must be a boolean")
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
        # Existing save files stored only Responses encrypted content.
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
    if item_type in {"context_prefix", "context_compaction"}:
        key = (
            "prefix_items"
            if "prefix_items" in mapping or "replacement_items" not in mapping
            else "replacement_items"
        )
        prefix = mapping.get(key)
        if not isinstance(prefix, list):
            raise SaveError(
                f"{item_type}.{key} must be a list"
            )
        prefix_items = tuple(
            interaction_item_from_dict(item) for item in prefix
        )
        if key == "prefix_items" and "replacement_items" in mapping:
            replacement = mapping["replacement_items"]
            if not isinstance(replacement, list):
                raise SaveError(
                    f"{item_type}.replacement_items must be a list"
                )
            replacement_items = tuple(
                interaction_item_from_dict(item) for item in replacement
            )
            if prefix_items != replacement_items:
                raise SaveError(
                    f"{item_type}.prefix_items and "
                    f"{item_type}.replacement_items must match"
                )
        return ContextPrefix(prefix_items)
    if item_type == "compaction_metadata":
        usage = mapping.get("usage")
        if not isinstance(usage, Mapping):
            raise SaveError("compaction_metadata.usage must be an object")
        try:
            elapsed = _validate_elapsed_seconds(mapping.get("elapsed_seconds"))
        except (TypeError, ValueError) as exc:
            raise SaveError(f"compaction_metadata.{exc}") from exc
        recovery = mapping.get("recovery", [])
        if not isinstance(recovery, list):
            raise SaveError("compaction_metadata.recovery must be a list")
        try:
            return CompactionMetadata(
                usage=TokenUsage(
                    input_tokens=_require_nonnegative_int(
                        usage.get("input_tokens"),
                        "compaction_metadata.usage.input_tokens",
                    ),
                    output_tokens=_require_nonnegative_int(
                        usage.get("output_tokens"),
                        "compaction_metadata.usage.output_tokens",
                    ),
                    total_tokens=_require_nonnegative_int(
                        usage.get("total_tokens"),
                        "compaction_metadata.usage.total_tokens",
                    ),
                    cached_input_tokens=_require_nonnegative_int(
                        usage.get("cached_input_tokens"),
                        "compaction_metadata.usage.cached_input_tokens",
                    ),
                ),
                protocol=_require_string(
                    mapping.get("protocol"),
                    "compaction_metadata.protocol",
                ),
                provider_session_id=_optional_string(
                    mapping,
                    "provider_session_id",
                    "compaction_metadata.provider_session_id",
                ),
                provider_turn_id=_optional_string(
                    mapping,
                    "provider_turn_id",
                    "compaction_metadata.provider_turn_id",
                ),
                provider_turn_state=_optional_string(
                    mapping,
                    "provider_turn_state",
                    "compaction_metadata.provider_turn_state",
                ),
                provider_response_id=_optional_string(
                    mapping,
                    "provider_response_id",
                    "compaction_metadata.provider_response_id",
                ),
                elapsed_seconds=elapsed,
                request_attempts=_require_nonnegative_int(
                    mapping.get("request_attempts", 1),
                    "compaction_metadata.request_attempts",
                ),
                recovery=tuple(
                    _require_string(
                        value,
                        f"compaction_metadata.recovery[{index}]",
                    )
                    for index, value in enumerate(recovery)
                ),
            )
        except (TypeError, ValueError) as exc:
            raise SaveError(f"compaction_metadata.{exc}") from exc
    if item_type in {"sample_metadata", "turn_metadata"}:
        usage = mapping.get("usage")
        if not isinstance(usage, Mapping):
            raise SaveError(f"{item_type}.usage must be an object")
        try:
            elapsed = _validate_elapsed_seconds(mapping.get("elapsed_seconds"))
        except (TypeError, ValueError) as exc:
            raise SaveError(f"{item_type}.{exc}") from exc
        recovery = mapping.get("recovery", [])
        if not isinstance(recovery, list):
            raise SaveError(f"{item_type}.recovery must be a list")
        tokens = mapping.get("tokens")
        if tokens is not None:
            if not isinstance(tokens, Mapping):
                raise SaveError(f"{item_type}.tokens must be an object")
            try:
                tokens = token_record_from_dict(tokens)
            except (TypeError, ValueError) as exc:
                raise SaveError(f"{item_type}.tokens.{exc}") from exc
        return SampleMetadata(
            usage=TokenUsage(
                input_tokens=_require_nonnegative_int(
                    usage.get("input_tokens"),
                    f"{item_type}.usage.input_tokens",
                ),
                output_tokens=_require_nonnegative_int(
                    usage.get("output_tokens"),
                    f"{item_type}.usage.output_tokens",
                ),
                total_tokens=_require_nonnegative_int(
                    usage.get("total_tokens"),
                    f"{item_type}.usage.total_tokens",
                ),
                cached_input_tokens=_require_nonnegative_int(
                    usage.get("cached_input_tokens"),
                    f"{item_type}.usage.cached_input_tokens",
                ),
            ),
            provider_session_id=_optional_string(
                mapping,
                "provider_session_id",
                f"{item_type}.provider_session_id",
            ),
            provider_turn_id=_optional_string(
                mapping,
                "provider_turn_id",
                f"{item_type}.provider_turn_id",
            ),
            provider_turn_state=_optional_string(
                mapping,
                "provider_turn_state",
                f"{item_type}.provider_turn_state",
            ),
            elapsed_seconds=elapsed,
            request_attempts=_require_positive_int(
                mapping.get("request_attempts", 1),
                f"{item_type}.request_attempts",
            ),
            recovery=tuple(
                _require_string(value, f"{item_type}.recovery[{index}]")
                for index, value in enumerate(recovery)
            ),
            tokens=tokens,
        )
    if item_type == "model_failure":
        recovery = mapping.get("recovery", [])
        if not isinstance(recovery, list):
            raise SaveError("model_failure.recovery must be a list")
        event_types = mapping.get("event_types", [])
        if not isinstance(event_types, list):
            raise SaveError("model_failure.event_types must be a list")
        try:
            elapsed = _validate_elapsed_seconds(mapping.get("elapsed_seconds"))
        except (TypeError, ValueError) as exc:
            raise SaveError(f"model_failure.{exc}") from exc
        try:
            return ModelFailure(
                category=_require_string(
                    mapping.get("category"), "model_failure.category",
                ),
                message=_require_string(
                    mapping.get("message"), "model_failure.message",
                ),
                provider=_optional_string(
                    mapping, "provider", "model_failure.provider",
                ),
                model=_optional_string(
                    mapping, "model", "model_failure.model",
                ),
                auth_source=_optional_string(
                    mapping,
                    "auth_source",
                    "model_failure.auth_source",
                ),
                http_status=_optional_http_status(mapping.get("http_status")),
                request_id=_optional_string(
                    mapping, "request_id", "model_failure.request_id",
                ),
                response_id=_optional_string(
                    mapping, "response_id", "model_failure.response_id",
                ),
                cf_ray=_optional_string(
                    mapping, "cf_ray", "model_failure.cf_ray",
                ),
                authorization_error=_optional_string(
                    mapping,
                    "authorization_error",
                    "model_failure.authorization_error",
                ),
                auth_error_code=_optional_string(
                    mapping,
                    "auth_error_code",
                    "model_failure.auth_error_code",
                ),
                error_code=_optional_string(
                    mapping,
                    "error_code",
                    "model_failure.error_code",
                ),
                attempt_count=_require_positive_int(
                    mapping.get("attempt_count", 1),
                    "model_failure.attempt_count",
                ),
                event_count=_require_nonnegative_int(
                    mapping.get("event_count", 0),
                    "model_failure.event_count",
                ),
                event_types=tuple(
                    _require_string(
                        value,
                        f"model_failure.event_types[{index}]",
                    )
                    for index, value in enumerate(event_types)
                ),
                completed_item_count=_require_nonnegative_int(
                    mapping.get("completed_item_count", 0),
                    "model_failure.completed_item_count",
                ),
                last_event_type=_optional_string(
                    mapping,
                    "last_event_type",
                    "model_failure.last_event_type",
                ),
                last_sequence_number=_optional_nonnegative_int(
                    mapping.get("last_sequence_number"),
                    "model_failure.last_sequence_number",
                ),
                recovery=tuple(
                    _require_string(
                        value,
                        f"model_failure.recovery[{index}]",
                    )
                    for index, value in enumerate(recovery)
                ),
                elapsed_seconds=elapsed,
            )
        except (TypeError, ValueError) as exc:
            raise SaveError(f"model_failure.{exc}") from exc
    if item_type == "turn_summary":
        try:
            elapsed = _validate_elapsed_seconds(mapping.get("elapsed_seconds"))
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
                elapsed_seconds=elapsed,
            )
        except (TypeError, ValueError) as exc:
            raise SaveError(f"turn_summary.{exc}") from exc
    if item_type == "model_sample_boundary":
        return ModelSampleBoundary()
    if item_type == "user_interaction_boundary":
        return UserInteractionBoundary()

    raise SaveError(f"unknown interaction item type: {item_type!r}")


def iter_interaction_items(
    items: Iterable[InteractionItem],
) -> Iterator[Dict[str, Any]]:
    for item in items:
        yield interaction_item_to_dict(item)


# Binary, so Windows does not translate newlines in appended lines.
_OPEN_FLAGS = getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)


def _encode_items(items: Iterable[InteractionItem]) -> bytes:
    """One JSON line per item, encoded completely before any file is touched.

    Characters that show as nothing or act on a terminal are written as JSON
    escapes (json_text_escaped), so viewing the file can't drive the terminal.
    """
    return "".join(
        json_text_escaped(json.dumps(encoded, ensure_ascii=False)) + "\n"
        for encoded in iter_interaction_items(items)
    ).encode("utf-8")


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError(errno.EIO, "short write")
        view = view[written:]


def _fsync_directory(directory: Path) -> None:
    """Make a rename in ``directory`` durable, where the platform allows it."""
    try:
        descriptor = os.open(directory, os.O_RDONLY | _OPEN_FLAGS)
    except OSError:
        return  # Windows cannot open a directory; the file itself was synced.
    try:
        os.fsync(descriptor)
    except OSError:
        pass  # Some filesystems cannot sync directories.
    finally:
        os.close(descriptor)


def _replace_atomically(destination: Path, data: bytes) -> os.stat_result:
    """Replace ``destination`` with ``data`` atomically; return the file's status."""
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
            status = os.fstat(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        _fsync_directory(destination.parent)
    except OSError as exc:
        raise SaveError(
            f"could not write save to {destination}: {exc}"
        ) from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return status


def save_interaction_save(path: SavePath, context: InteractionContext) -> None:
    """Atomically write a context as one JSON interaction item per line.

    This writes a complete snapshot, replacing any file at ``path``. Use an
    :class:`InteractionSaveWriter` to save a growing context by appending.
    """
    _replace_atomically(Path(path), _encode_items(context.items))


@dataclass(frozen=True)
class IncompleteSaveLine:
    """A save's final line that fails to parse, as an interrupted append leaves it.

    ``offset`` is where the line starts (the length of the save's complete
    part), ``size`` its length in bytes, and ``error`` why it failed to parse.
    Loading ignores the line; the writer of a resumed save truncates it at
    its first save.
    """

    line_number: int
    offset: int
    size: int
    error: str

    def warning(self, name: str) -> str:
        """The notice for resuming the save called ``name`` and truncating the line."""
        return (
            f"Warning: {name} ends with a line that fails to parse "
            f"(line {self.line_number}, {self.size} bytes), probably cut off by an "
            "interrupted save. It was not loaded and is truncated from the file."
        )


class _Unparsed(NamedTuple):
    line: IncompleteSaveLine
    message: str
    error: ValueError


class _ReadSave(NamedTuple):
    context: InteractionContext
    status: os.stat_result
    size: int  # bytes read
    complete_size: int  # bytes before an incomplete final line
    terminated: bool  # the complete part is empty or ends with a newline
    incomplete: Optional[IncompleteSaveLine]
    data: Optional[bytes] = None  # the bytes read, when kept (keep_data)


def _parse_lines(
    file: BinaryIO, source: Path,
) -> Tuple[List[InteractionItem], int, bool, Optional[_Unparsed]]:
    items: List[InteractionItem] = []
    offset = 0
    terminated = True
    unparsed: Optional[_Unparsed] = None
    # Split on b"\n" only: a line written before such characters were escaped
    # (json_text_escaped), or by hand, may hold another line separator, such
    # as U+2028, inside a string.
    for line_number, line in enumerate(file, 1):
        if unparsed is not None:
            # Data follows the line, so no interrupted append left it.
            raise SaveError(unparsed.message) from unparsed.error
        start, offset = offset, offset + len(line)
        terminated = line.endswith(b"\n")
        if not line.strip():
            continue
        try:
            value = json.loads(line.decode("utf-8"))
        except ValueError as exc:  # UnicodeDecodeError or JSONDecodeError
            kind = "UTF-8" if isinstance(exc, UnicodeDecodeError) else "JSON"
            unparsed = _Unparsed(
                IncompleteSaveLine(line_number, start, len(line), str(exc)),
                f"invalid {kind} in save {source} at line {line_number}: {exc}",
                exc,
            )
            continue

        try:
            items.append(interaction_item_from_dict(value))
        except (SaveError, TypeError, ValueError) as exc:
            raise SaveError(
                f"invalid save item in {source} at line "
                f"{line_number}: {exc}"
            ) from exc
    if unparsed is not None and not items:
        # A save's first items are written atomically, so an interrupted
        # append cannot leave only this line: it is not a save.
        raise SaveError(unparsed.message) from unparsed.error
    return items, offset, terminated, unparsed


def _read_save(source: Path, *, keep_data: bool = False) -> _ReadSave:
    try:
        with source.open("rb") as file:
            status = os.fstat(file.fileno())
            data = file.read() if keep_data else None
            items, size, terminated, unparsed = _parse_lines(
                file if data is None else io.BytesIO(data), source)
    except OSError as exc:
        raise SaveError(f"could not load save {source}: {exc}") from exc

    try:
        context = InteractionContext(items)
    except (TypeError, ValueError) as exc:
        raise SaveError(f"invalid save {source}: {exc}") from exc
    if unparsed is None:
        return _ReadSave(context, status, size, size, terminated, None, data)
    # The incomplete line starts right after a newline.
    return _ReadSave(context, status, size, unparsed.line.offset, True, unparsed.line, data)


def load_interaction_save(path: SavePath) -> InteractionContext:
    """Load and validate a JSONL interaction-save file. The file is never modified.

    A final line that fails to parse (invalid JSON or UTF-8) after complete
    items is what an interrupted append leaves, and is ignored: see
    :func:`resume_interaction_save`. Any other invalid line is an error.
    """
    return _read_save(Path(path)).context


class SaveData(NamedTuple):
    """A loaded save with the bytes it was loaded from (``read_interaction_save``)."""

    context: InteractionContext
    data: bytes  # the file as read
    complete_size: int  # the bytes before an incomplete final line
    incomplete_line: Optional[IncompleteSaveLine]
    status: os.stat_result  # the file's, when it was opened


def read_interaction_save(path: SavePath) -> SaveData:
    """Load a save as :func:`load_interaction_save` does, with the bytes it was loaded from."""
    read = _read_save(Path(path), keep_data=True)
    assert read.data is not None
    return SaveData(read.context, read.data, read.complete_size, read.incomplete, read.status)


class InteractionSaveWriter:
    """Append-only writer of one interaction save.

    ``InteractionSaveWriter(path)`` starts a new save: its first :meth:`save`
    atomically replaces any file at ``path`` with the whole context, as
    :func:`save_interaction_save` does. Every later save appends only the
    items added since, in one write followed by ``fsync``; saved lines are
    never rewritten. :func:`resume_interaction_save` returns a writer that
    continues an existing save.

    A crash while appending can leave an incomplete final line, which
    loading ignores and a resumed writer truncates. A failed save restores
    the file's previous length where it can, and the next save writes the
    unsaved items again. The writer refuses to append to a file that was
    replaced or changed since it was loaded or last saved. Saves are
    serialized by a lock.
    """

    def __init__(self, path: SavePath) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        # (st_dev, st_ino) of the save; None until the first save creates it.
        self._identity: Optional[Tuple[int, int]] = None
        self._end = 0  # where the saved lines end and the next append starts
        # The file's expected length: more than _end with an incomplete final
        # line; None when a failed append may have left part of its data.
        self._size: Optional[int] = 0
        self._newline = False  # the last saved line lacks its newline
        self._count = 0  # items saved
        self._last: Optional[InteractionItem] = None

    def save(self, context: InteractionContext) -> None:
        """Save the items added to ``context`` since the last save.

        ``context`` must extend the saved items: the same context, grown, or
        a ``copy()`` of it. Afterwards the file holds exactly its items.
        """
        with self._lock:
            count = self._count
            if len(context) < count or (
                count and not _same_item(context[count - 1], self._last)
            ):
                raise SaveError(
                    f"cannot save to {self.path}: the context does not extend "
                    "the saved items"
                )
            new = context[count:]
            if self._identity is None:
                self._create(new)
            elif new or self._size != self._end or self._newline:
                self._append(_encode_items(new))
            else:
                return
            self._count = count + len(new)
            if new:
                self._last = new[-1]

    def _continue(self, read: _ReadSave) -> None:
        self._identity = (read.status.st_dev, read.status.st_ino)
        self._end, self._size = read.complete_size, read.size
        self._newline = not read.terminated
        self._count = len(read.context)
        self._last = read.context[-1] if len(read.context) else None

    def _create(self, items: List[InteractionItem]) -> None:
        data = _encode_items(items)
        status = _replace_atomically(self.path, data)
        self._identity = (status.st_dev, status.st_ino)
        self._end = self._size = len(data)
        self._newline = False

    def _append(self, data: bytes) -> None:
        if self._newline:
            data = b"\n" + data
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | _OPEN_FLAGS)
        except OSError as exc:
            raise SaveError(f"could not append to save {self.path}: {exc}") from exc
        try:
            status = os.fstat(descriptor)
            if (
                (status.st_dev, status.st_ino) != self._identity
                or status.st_size < self._end
                or (self._size is not None and status.st_size != self._size)
            ):
                raise SaveError(
                    f"cannot append to save {self.path}: the file was replaced "
                    "or changed since it was loaded or last saved"
                )
            if status.st_size != self._end:
                # Drop an incomplete final line, or a failed append's data.
                os.ftruncate(descriptor, self._end)
                self._size = self._end
            try:
                _write_all(descriptor, data)
                os.fsync(descriptor)
            except BaseException:
                # Keep the file ending with a complete line where possible.
                self._size = None
                with suppress(OSError):
                    os.ftruncate(descriptor, self._end)
                    self._size = self._end
                raise
        except OSError as exc:
            raise SaveError(f"could not append to save {self.path}: {exc}") from exc
        finally:
            os.close(descriptor)
        self._end = self._size = self._end + len(data)
        self._newline = False


def _same_item(item: InteractionItem, saved: Optional[InteractionItem]) -> bool:
    return item is saved or item == saved


class ResumedInteractionSave(NamedTuple):
    """A loaded save, a writer that continues it, and any incomplete line."""

    context: InteractionContext
    writer: InteractionSaveWriter
    incomplete_line: Optional[IncompleteSaveLine]


def resume_interaction_save(path: SavePath) -> ResumedInteractionSave:
    """Load a save to continue it with an append-only writer.

    Loading is as by :func:`load_interaction_save` and modifies nothing. An
    incomplete final line is returned for the caller to report (see
    :meth:`IncompleteSaveLine.warning`); the writer's first save truncates
    it, or terminates a final line that lacks only its newline, before
    appending.
    """
    source = Path(path)
    read = _read_save(source)
    writer = InteractionSaveWriter(source)
    writer._continue(read)
    return ResumedInteractionSave(read.context, writer, read.incomplete)


class ResumedSaveLines(NamedTuple):
    """A resumed save (see :func:`resume_interaction_save`) with its complete lines."""

    context: InteractionContext
    writer: InteractionSaveWriter
    incomplete_line: Optional[IncompleteSaveLine]
    lines: bytes  # the save as read, without an incomplete final line


def resume_interaction_save_lines(path: SavePath) -> ResumedSaveLines:
    """Load a save to continue it, as :func:`resume_interaction_save` does.

    It also returns the bytes of the save's complete lines, read with them,
    from which :func:`public_log_lines` makes its public log.
    """
    source = Path(path)
    read = _read_save(source, keep_data=True)
    writer = InteractionSaveWriter(source)
    writer._continue(read)
    assert read.data is not None
    return ResumedSaveLines(read.context, writer, read.incomplete, read.data[:read.complete_size])


# Public logs: a full log line for line, with its reasoning left out, or shown
# by its summaries, and without the samples' token traces.

# A public log has this line in place of each reasoning item of its full log
# that it doesn't show, so that line N of the public log stands for line N of
# the full log.
NIL_LINE = b'{"type": "nil"}\n'
# How a public log shows reasoning (the CLI's save_log_reasoning): "nil",
# never, with a NIL_LINE for each reasoning item; "assistant", as an assistant
# message with the item's summary, and a NIL_LINE for an item without one.
PUBLIC_LOG_REASONING = ("nil", "assistant")
# Every form of a public log's reasoning lines, as (reasoning, thinking) (see
# public_log_line). The form applies to the lines written while it is in
# effect, so comparisons accept a line in any of them.
_PUBLIC_LOG_FORMS = (("nil", False), ("assistant", False), ("assistant", True))


def _require_public_log_reasoning(reasoning: object) -> str:
    if not isinstance(reasoning, str) or reasoning not in PUBLIC_LOG_REASONING:
        raise ValueError(f"reasoning must be one of: {', '.join(PUBLIC_LOG_REASONING)}")
    return reasoning


def _reasoning_message(item: Reasoning, reasoning: str, thinking: bool) -> Optional[Message]:
    """The assistant message that shows a reasoning item, or None for a nil line.

    With ``reasoning`` ``"assistant"``, its text is the item's summary: its
    parts, trimmed, without empty ones, a blank line apart. With
    ``thinking``, an item that is text alone, without a summary, encrypted
    content or a signature, as Claude Relay's thinking is, shows its whole
    text, ``content``, instead. Encrypted content and signatures never show.
    """
    if reasoning != "assistant":
        return None
    text = "\n\n".join(part.strip() for part in item.summary if part.strip())
    if (not text and thinking and item.encrypted_content is None
            and item.content_signature is None):
        text = item.content.strip()
    return Message("assistant", text) if text else None


def _shown_record(record: Dict[str, Any], reasoning: str,
                  thinking: bool) -> Optional[Dict[str, Any]]:
    """A full log's reasoning record, as a public log shows it; None for a nil line.

    The record is decoded first, so that a legacy one shows as its item does.
    """
    if reasoning == "nil":
        return None
    try:
        item = interaction_item_from_dict(record)
    except (SaveError, TypeError, ValueError):
        return None
    message = (_reasoning_message(item, reasoning, thinking)
               if isinstance(item, Reasoning) else None)
    return None if message is None else interaction_item_to_dict(message)


def _record_line(record: Mapping[str, Any]) -> bytes:
    return (json_text_escaped(json.dumps(record, ensure_ascii=False)) + "\n").encode("utf-8")


def _traced(record: object) -> bool:
    """Whether a record is a sample's metadata with a token trace, which a
    public log leaves out: its arrays are long, and its sampled IDs decode to
    the reasoning that the public log may leave out too."""
    return (isinstance(record, dict) and record.get("type") == "sample_metadata"
            and "tokens" in record)


def _untraced(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items() if key != "tokens"}


def _public_line(line: bytes, reasoning: str = "nil", thinking: bool = False) -> bytes:
    """A full log's line, without its newline, as its public log has it."""
    # Only a line that names one of the two types, or a token trace, can
    # change: skip parsing the others.
    if b'"reasoning"' in line or b'"context_prefix"' in line or b'"tokens"' in line:
        try:
            value = json.loads(line)
        except ValueError:
            value = None
        if isinstance(value, dict):
            if value.get("type") == "reasoning":
                shown = _shown_record(value, reasoning, thinking)
                return NIL_LINE if shown is None else _record_line(shown)
            if _traced(value):
                return _record_line(_untraced(value))
            items = value.get("prefix_items")
            if value.get("type") == "context_prefix" and isinstance(items, list):
                kept, changed = [], False
                for nested in items:
                    if isinstance(nested, dict) and nested.get("type") == "reasoning":
                        changed = True
                        shown = _shown_record(nested, reasoning, thinking)
                        if shown is not None:
                            kept.append(shown)
                    elif _traced(nested):
                        changed = True
                        kept.append(_untraced(nested))
                    else:
                        kept.append(nested)
                if changed:
                    value["prefix_items"] = kept
                    return _record_line(value)
    return line + b"\n"


def _full_log_lines(data: bytes) -> List[bytes]:
    """A full log's complete lines, ``data``, each without its newline."""
    if not data:
        return []
    pieces = data.split(b"\n")
    if not pieces[-1]:
        pieces.pop()
    return pieces


def public_log_lines(data: bytes, reasoning: str = "nil", *,
                     thinking: bool = False) -> List[bytes]:
    """The public log of a full log's complete lines, ``data``: a line for each.

    A reasoning item's line shows it as ``reasoning`` and ``thinking`` say
    (see :func:`public_log_line`), and a ``context_prefix`` shows the
    reasoning among its items so, leaving out those it wouldn't show. A
    sample's metadata leaves out its token trace, ``tokens``, also among a
    ``context_prefix``'s items. Every other line is kept byte for byte, blank
    or not, and each ends with a newline.
    """
    _require_public_log_reasoning(reasoning)
    return [_public_line(piece, reasoning, thinking) for piece in _full_log_lines(data)]


def public_log_line(item: InteractionItem, reasoning: str = "nil", *,
                    thinking: bool = False) -> bytes:
    """An item's line in a public log, as :func:`public_log_lines` makes it.

    ``reasoning`` says how a reasoning item shows (``PUBLIC_LOG_REASONING``):
    ``"nil"``, as a ``{"type": "nil"}`` line; ``"assistant"``, as an
    assistant message with the item's summary, or a nil line when it has
    none. ``thinking`` also shows the whole text, ``content``, of an item
    that is text alone; the CLI sets it for Claude Relay's own thinking,
    which is text alone, as Chat Completions' full reasoning is once saved.
    Encrypted content and signatures never show, nor does a sample's token
    trace (``SampleMetadata.tokens``).
    """
    _require_public_log_reasoning(reasoning)
    if isinstance(item, Reasoning):
        message = _reasoning_message(item, reasoning, thinking)
        return NIL_LINE if message is None else _encode_items((message,))
    if isinstance(item, SampleMetadata) and item.tokens is not None:
        return _encode_items((replace(item, tokens=None),))
    line = _encode_items((item,))
    if isinstance(item, ContextPrefix) and any(
            isinstance(nested, Reasoning)
            or isinstance(nested, SampleMetadata) and nested.tokens is not None
            for nested in item.prefix_items):
        return _public_line(line[:-1], reasoning, thinking)
    return line


def _same_line(line: bytes, expected: bytes) -> bool:
    """Whether a public log's line is the expected one, byte for byte or as JSON.

    Comparing JSON values too keeps a line that an older version encoded
    otherwise from counting as a difference.
    """
    if line == expected:
        return True
    try:
        return json.loads(line) == json.loads(expected)
    except ValueError:
        return False


def _shown_forms(piece: bytes, reasoning: str, thinking: bool) -> Iterator[bytes]:
    """A full log's line, ``piece``, in the given form, then in the others."""
    given = (reasoning, thinking and reasoning != "nil")
    yield _public_line(piece, *given)
    for form in _PUBLIC_LOG_FORMS:
        if form != given:
            yield _public_line(piece, *form)


def _shows(line: bytes, piece: bytes, reasoning: str, thinking: bool) -> bool:
    """Whether a public log's line stands for the full log's line ``piece``, in any form."""
    return any(_same_line(line, shown) for shown in _shown_forms(piece, reasoning, thinking))


def _public_log_state(current: bytes, pieces: Sequence[bytes], reasoning: str,
                      thinking: bool = False) -> Tuple[Optional[int], bytes]:
    """Where a public log first differs from its full log's lines, ``pieces``, and what it lacks.

    Its lines may show reasoning in any form (``_PUBLIC_LOG_FORMS``), whichever
    was in effect when they were written. Returns the first differing line's
    number, or None, and the bytes that complete it: the rest of the full
    log's lines, in the given form, after the lines it has; and the rest of a
    line it has only the start of, as an interrupted append leaves it, in the
    form that the line started in.
    """
    lines = current.split(b"\n")
    partial = lines.pop()  # b"" when the log is empty or ends with a newline
    for index, line in enumerate(lines):
        if index >= len(pieces) or not _shows(line + b"\n", pieces[index], reasoning, thinking):
            return index + 1, b""
    rest = [_public_line(piece, reasoning, thinking) for piece in pieces[len(lines):]]
    if partial:
        started = None if not rest else next(
            (shown for shown in _shown_forms(pieces[len(lines)], reasoning, thinking)
             if shown.startswith(partial)), None)
        if started is None:
            return len(lines) + 1, b""
        rest[0] = started[len(partial):]
    return None, b"".join(rest)


def _failure_reason(error: BaseException) -> str:
    """Why a write failed, without the path it names."""
    cause = error.__cause__ if isinstance(error, SaveError) else error
    if isinstance(cause, OSError):
        return cause.strerror or str(cause) or type(cause).__name__
    return str(error) or type(error).__name__


def _read_from_start(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1 << 20)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _ends_with_newline(descriptor: int, size: int) -> bool:
    if not size:
        return True
    os.lseek(descriptor, size - 1, os.SEEK_SET)
    return os.read(descriptor, 1) == b"\n"


class PublicLogWriter:
    """Best-effort writer of a save's public log: its full log without its reasoning,
    or with it shown by its summaries.

    Line N of the public log stands for item N of the full log
    (:func:`public_log_line`). The full log is the save, which is all that
    resuming needs; the public log is
    informational, and may be edited out-of-band. So this writer never
    raises. It keeps the file as it finds it, appends after its end, and
    reports a change it finds, or a write that failed, as a warning
    (:meth:`take_warnings`); a failed write is tried again at the next save.

    ``reasoning`` says how a reasoning item shows, ``"nil"`` or ``"assistant"``
    (``PUBLIC_LOG_REASONING``), or is a function that returns one, read at
    each save. A change applies to the lines written from then on: lines
    already written keep their form, and comparisons accept a line in any
    form. ``thinking`` also shows the whole text of a reasoning item without
    a summary (Claude Relay's thinking), but only in the lines of the items
    that this writer's session adds: all of a new save's, or those after a
    resumed save's (:meth:`sync`). An earlier item may come from another
    model, whose items of that kind hold its full reasoning, so its line
    shows only a summary, also when a missing public log is written anew.

    A new writer's first :meth:`save` writes the whole public log, replacing
    any file at ``path``, as a new save does; :meth:`sync` continues one
    instead. A missing public log is written anew from the full log. Each
    write ends with ``fsync``. ``drifted`` is true once the public log is
    known to differ from the full log, so that its line numbers may not
    match the full log's.
    """

    def __init__(self, path: SavePath, *, name: Optional[str] = None,
                 reasoning: Union[str, Callable[[], str]] = "nil",
                 thinking: bool = False) -> None:
        self.path = Path(path)
        self._name = self.path.name if name is None else name  # in warnings
        if callable(reasoning):
            self._reasoning = reasoning
        else:
            fixed = _require_public_log_reasoning(reasoning)
            self._reasoning = lambda: fixed
        self._thinking = bool(thinking)
        self._lock = threading.Lock()
        self._warnings_lock = threading.Lock()
        self._warnings: List[str] = []
        self._count = 0  # items of the full log that the public log has
        self._own = 0  # where the items that this writer's session adds start
        self._fresh = True  # the next save writes the whole public log anew
        # (st_dev, st_ino, st_size) as the last write left the file, or None
        # when that isn't known (then the next append reports no change).
        self._state: Optional[Tuple[int, int, int]] = None
        self._failed = False  # the last write failed, which was reported
        self.drifted = False

    def take_warnings(self) -> Tuple[str, ...]:
        """The warnings since the last call, for the frontend to show."""
        with self._warnings_lock:
            warnings, self._warnings = tuple(self._warnings), []
        return warnings

    def _warn(self, text: str) -> None:
        with self._warnings_lock:
            self._warnings.append(text)

    def _write_failed(self, error: BaseException) -> None:
        if not self._failed:
            self._warn(
                f"Warning: could not write {self._name} ({_failure_reason(error)}). It is "
                "informational: the save is unaffected, and the next save tries again."
            )
        self._failed = True

    def _missing_note(self) -> str:
        return f"Note: {self._name} was missing; it was written anew from the full log."

    def _lines(self, context: InteractionContext, start: int, reasoning: str) -> bytes:
        """The public log's lines of ``context``'s items from ``start`` on."""
        return b"".join(
            public_log_line(item, reasoning, thinking=self._thinking and index >= self._own)
            for index, item in enumerate(context[start:], start)
        )

    def sync(self, lines: bytes, count: int) -> None:
        """Continue the public log of a full log of ``count`` items, whose complete lines are ``lines``.

        A public log that has the first of its lines, as a crash between the
        two saves or an interrupted append leaves it, gets the rest. One that
        differs is kept as it is, with a warning: later saves append after
        its end. A missing one is written anew. Lines that show reasoning in
        another form, as written while another mode was in effect, are no
        difference. The items after these ``count`` are this session's.
        """
        with self._lock:
            self._fresh, self._count, self._own = False, count, count
            try:
                reasoning = _require_public_log_reasoning(self._reasoning())
                pieces = _full_log_lines(lines)
                try:
                    descriptor = os.open(self.path, os.O_RDWR | os.O_APPEND | _OPEN_FLAGS)
                except FileNotFoundError:
                    self._replace(b"".join(_public_line(piece, reasoning) for piece in pieces),
                                  note=self._missing_note())
                    return
                try:
                    status = os.fstat(descriptor)
                    differs, rest = _public_log_state(
                        _read_from_start(descriptor), pieces, reasoning)
                    if differs is not None:
                        self._state = None  # appends start on a line of their own
                        self.drifted = True
                        self._warn(
                            f"Note: {self._name} differs from the full log from line {differs} "
                            "on, as when it is edited. It is informational, so it was kept as "
                            "it is: new lines are appended after its end, and its line numbers "
                            "may not match the full log's."
                        )
                        return
                    if rest:
                        try:
                            _write_all(descriptor, rest)
                            os.fsync(descriptor)
                        except OSError:
                            # It had the full log's first lines: writing it
                            # anew at the next save loses nothing.
                            self._fresh = True
                            raise
                    self._state = (status.st_dev, status.st_ino, status.st_size + len(rest))
                    self._failed = False
                finally:
                    os.close(descriptor)
            except Exception as exc:
                self._write_failed(exc)

    def save(self, context: InteractionContext) -> None:
        """Add the items of ``context`` that the public log lacks. Never raises."""
        with self._lock:
            try:
                reasoning = _require_public_log_reasoning(self._reasoning())
                if self._fresh:
                    if self._replace(self._lines(context, 0, reasoning)):
                        self._count = len(context)
                    return
                if (len(context) > self._count
                        and self._append(context, self._lines(context, self._count, reasoning),
                                         reasoning)):
                    self._count = len(context)
            except Exception as exc:
                self._write_failed(exc)

    def _replace(self, data: bytes, *, note: Optional[str] = None) -> bool:
        try:
            status = _replace_atomically(self.path, data)
        except SaveError as exc:
            self._fresh = True
            self._write_failed(exc)
            return False
        self._state = (status.st_dev, status.st_ino, len(data))
        self._fresh = self._failed = self.drifted = False
        if note is not None:
            self._warn(note)
        return True

    def _append(self, context: InteractionContext, data: bytes, reasoning: str) -> bool:
        try:
            descriptor = os.open(self.path, os.O_RDWR | os.O_APPEND | _OPEN_FLAGS)
        except FileNotFoundError:
            return self._replace(self._lines(context, 0, reasoning), note=self._missing_note())
        except OSError as exc:
            self._write_failed(exc)
            return False
        try:
            status = os.fstat(descriptor)
            if (status.st_dev, status.st_ino, status.st_size) != self._state:
                if self._state is not None:
                    self.drifted = True
                    self._warn(
                        f"Note: {self._name} was changed outside this session. It is "
                        "informational, so it was kept as it is: new lines are appended "
                        "after its end, and its line numbers may not match the full log's."
                    )
                if not _ends_with_newline(descriptor, status.st_size):
                    data = b"\n" + data
            try:
                _write_all(descriptor, data)
                os.fsync(descriptor)
            except OSError:
                # Drop this write's part, so the next save can write it again
                # (on a line of its own: the state is no longer known).
                self._state = None
                try:
                    os.ftruncate(descriptor, status.st_size)
                except OSError:
                    self.drifted = True  # part of this write may remain
                raise
            self._state = (status.st_dev, status.st_ino, status.st_size + len(data))
            self._failed = False
            return True
        except OSError as exc:
            self._write_failed(exc)
            return False
        finally:
            os.close(descriptor)


__all__ = [
    "IncompleteSaveLine",
    "InteractionSaveWriter",
    "NIL_LINE",
    "PUBLIC_LOG_REASONING",
    "PublicLogError",
    "PublicLogWriter",
    "ResumedInteractionSave",
    "ResumedSaveLines",
    "SaveData",
    "SaveError",
    "interaction_item_from_dict",
    "interaction_item_to_dict",
    "iter_interaction_items",
    "load_interaction_save",
    "public_log_line",
    "public_log_lines",
    "read_interaction_save",
    "resume_interaction_save",
    "resume_interaction_save_lines",
    "save_interaction_save",
]
