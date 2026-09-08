from __future__ import annotations

import json
import math
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Tuple

from .codex_auth import CodexAuth
from .codex_auth import CodexAuthPath
from .codex_auth import load_codex_auth
from .context import ModelContext
from .items import ContextCompaction
from .items import Init
from .items import Instructions
from .items import InteractionItem
from .items import Message
from .items import ModelSampleBoundary
from .items import OpaqueCompaction
from .items import Reasoning
from .items import ToolCall
from .items import ToolResult
from .items import TurnMetadata
from .items import TurnSummary
from .items import UserInteractionBoundary
from .model import ModelConfigurationError
from .model import ModelContextWindowError
from .model import ModelResponseError
from .model import ModelSample
from .model import ModelTimeoutError
from .model import ModelTransportError
from .model import SamplingOptions
from .usage import TokenUsage


OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1"
CODEX_RESPONSES_API_URL = "https://chatgpt.com/backend-api/codex"
META_RESPONSES_API_URL = "https://api.meta.ai/v1"
X_CODEX_TURN_STATE_HEADER = "x-codex-turn-state"


@dataclass(frozen=True)
class _CodexModelRoute:
    api_model: str
    api_url: str = CODEX_RESPONSES_API_URL
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    text_verbosity: Optional[str] = None
    api_key_environment_variable: Optional[str] = None
    # Catalog context capacities in tokens, not Responses request parameters.
    default_context_tokens: Optional[int] = None
    max_context_tokens: Optional[int] = None


# Context capacities follow codex-latest-20260904's models-manager/models.json.
_CODEX_MODEL_ROUTES = {
    "gpt-5.6-sol": _CodexModelRoute(
        api_model="gpt-5.6-sol",
        default_context_tokens=272_000,
        max_context_tokens=872_000,
    ),
    "gpt-5.6-sol-medium": _CodexModelRoute(
        api_model="gpt-5.6-sol",
        reasoning_effort="medium",
        default_context_tokens=272_000,
        max_context_tokens=872_000,
    ),
    "gpt-5.6-sol-max": _CodexModelRoute(
        api_model="gpt-5.6-sol",
        reasoning_effort="max",
        default_context_tokens=272_000,
        max_context_tokens=872_000,
    ),
    "gpt-6-astra": _CodexModelRoute(
        api_model="gpt-6-astra",
        reasoning_summary="auto",
        text_verbosity="low",
        default_context_tokens=272_000,
        max_context_tokens=872_000,
    ),
    "gpt-6-astra-medium": _CodexModelRoute(
        api_model="gpt-6-astra",
        reasoning_effort="medium",
        reasoning_summary="auto",
        text_verbosity="low",
        default_context_tokens=272_000,
        max_context_tokens=872_000,
    ),
    "gpt-6-astra-max": _CodexModelRoute(
        api_model="gpt-6-astra",
        reasoning_effort="max",
        # The catalog default is no summary; this alias deliberately opts in.
        reasoning_summary="auto",
        text_verbosity="low",
        default_context_tokens=272_000,
        max_context_tokens=872_000,
    ),
    "muse-spark-1.3": _CodexModelRoute(
        api_model="muse-spark-1.3-contributor",
        api_url=META_RESPONSES_API_URL,
        api_key_environment_variable="META_API_KEY",
    ),
    "muse-spark-1.3-xhigh": _CodexModelRoute(
        api_model="muse-spark-1.3-contributor",
        api_url=META_RESPONSES_API_URL,
        reasoning_effort="xhigh",
        api_key_environment_variable="META_API_KEY",
    ),
}


def _normalize_configuration(
    api_url: str, model: str, request_timeout_seconds: float,
) -> Tuple[str, str, float]:
    """Validate non-secret endpoint options before attempting credential loading."""
    if not isinstance(api_url, str):
        raise TypeError("api_url must be a string")
    api_url = api_url.strip()
    if not api_url:
        raise ModelConfigurationError("api_url must not be empty")
    if any(character.isspace() for character in api_url):
        raise ModelConfigurationError(
            "api_url must not contain whitespace"
        )

    try:
        parsed = urllib.parse.urlsplit(api_url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ModelConfigurationError("api_url is invalid") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ModelConfigurationError(
            "api_url scheme must be 'http' or 'https'"
        )
    if not parsed.netloc or hostname is None:
        raise ModelConfigurationError(
            "api_url must be an absolute URL with a host"
        )
    if port == 0:
        raise ModelConfigurationError(
            "api_url port must be from 1 through 65535"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ModelConfigurationError(
            "api_url must not contain user information"
        )
    if parsed.query:
        raise ModelConfigurationError("api_url must not contain a query")
    if parsed.fragment:
        raise ModelConfigurationError(
            "api_url must not contain a fragment"
        )
    path_prefix = parsed.path.rstrip("/")
    if path_prefix.endswith("/responses"):
        raise ModelConfigurationError(
            "api_url must not include the fixed /responses path"
        )
    api_url = urllib.parse.urlunsplit(
        (scheme, parsed.netloc, path_prefix, "", "")
    )

    if not isinstance(model, str):
        raise TypeError("model must be a string")
    model = model.strip()
    if not model:
        raise ModelConfigurationError("model must not be empty")

    timeout = request_timeout_seconds
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0
    ):
        raise ModelConfigurationError(
            "request_timeout_seconds must be positive and finite"
        )
    return api_url, model, float(timeout)


@dataclass(frozen=True)
class StreamingResponsesEndpoint:
    api_url: str
    model: str
    bearer_token: str = field(repr=False)
    account_id: Optional[str] = None
    api_provider: str = "api"
    request_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        api_url, model, timeout = _normalize_configuration(
            self.api_url, self.model, self.request_timeout_seconds
        )
        object.__setattr__(self, "api_url", api_url)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "request_timeout_seconds", timeout)

        if not isinstance(self.bearer_token, str):
            raise TypeError("bearer_token must be a string")
        bearer_token = self.bearer_token.strip()
        if not bearer_token:
            raise ModelConfigurationError("bearer_token must not be empty")
        if any(character.isspace() for character in bearer_token):
            raise ModelConfigurationError(
                "bearer_token must not contain whitespace"
            )
        object.__setattr__(self, "bearer_token", bearer_token)

        if not isinstance(self.api_provider, str):
            raise TypeError("api_provider must be a string")
        api_provider = self.api_provider.strip().lower()
        if api_provider not in {"api", "codex"}:
            raise ModelConfigurationError(
                "api_provider must be 'api' or 'codex'"
            )
        object.__setattr__(self, "api_provider", api_provider)

        if self.account_id is not None:
            if not isinstance(self.account_id, str):
                raise TypeError("account_id must be a string or None")
            account_id = self.account_id.strip()
            if not account_id:
                raise ModelConfigurationError("account_id must not be empty")
            if "\r" in account_id or "\n" in account_id:
                raise ModelConfigurationError(
                    "account_id must not contain newlines"
                )
            if api_provider != "codex":
                raise ModelConfigurationError(
                    "account_id requires api_provider='codex'"
                )
            object.__setattr__(self, "account_id", account_id)

    @property
    def url(self) -> str:
        return f"{self.api_url}/responses"


@dataclass(frozen=True)
class _ProviderState:
    session_id: Optional[str] = None
    turn_id: Optional[str] = None
    turn_state: Optional[str] = None
    persist_session_id: bool = False


def _resolve_request_route(
    endpoint: StreamingResponsesEndpoint,
) -> _CodexModelRoute:
    if endpoint.api_provider != "codex":
        return _CodexModelRoute(api_model=endpoint.model)
    route = _CODEX_MODEL_ROUTES.get(endpoint.model)
    if route is None:
        return _CodexModelRoute(api_model=endpoint.model)
    return route


def _resolve_default_codex_api_url(model: str) -> str:
    route = _CODEX_MODEL_ROUTES.get(model)
    if route is None:
        return CODEX_RESPONSES_API_URL
    return route.api_url


def _load_default_model_auth(
    model: str,
    *,
    codex_home: Optional[CodexAuthPath],
    auth_file: Optional[CodexAuthPath],
) -> CodexAuth:
    route = _CODEX_MODEL_ROUTES.get(model)
    api_key_environment_variable = (
        route.api_key_environment_variable
        if route is not None
        else None
    )
    if (
        api_key_environment_variable is not None
        and codex_home is None
        and auth_file is None
    ):
        api_key = os.environ.get(api_key_environment_variable)
        if api_key is None or not api_key.strip():
            raise ModelConfigurationError(
                f"{api_key_environment_variable} is required for model "
                f"{model!r}"
            )
        return CodexAuth(access_token=api_key)
    return load_codex_auth(
        codex_home=codex_home,
        auth_file=auth_file,
    )


def _encode_context_items(
    items: Sequence[InteractionItem],
) -> List[Dict[str, Any]]:
    encoded: List[Dict[str, Any]] = []
    # Last-wins Instructions -> leading system message (parity with
    # Chat Completions system). Empty preserved; absence means none.
    effective: Optional[Instructions] = None
    for item in items:
        if isinstance(item, Instructions):
            effective = item
    if effective is not None:
        encoded.append(
            {
                "type": "message",
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": effective.text,
                    }
                ],
            }
        )
    for index, item in enumerate(items):
        if isinstance(
            item,
            (
                ModelSampleBoundary,
                Init,
                TurnMetadata,
                TurnSummary,
                UserInteractionBoundary,
            ),
        ):
            continue

        if isinstance(item, Instructions):
            continue

        if isinstance(item, Message):
            if item.role not in {"system", "developer", "user", "assistant"}:
                raise ModelConfigurationError(
                    f"unsupported message role at item {index}: {item.role!r}"
                )
            content_type = (
                "output_text" if item.role == "assistant" else "input_text"
            )
            encoded.append(
                {
                    "type": "message",
                    "role": item.role,
                    "content": [
                        {
                            "type": content_type,
                            "text": item.content,
                        }
                    ],
                }
            )
            continue

        if isinstance(item, Reasoning):
            reasoning: Dict[str, Any] = {
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": text,
                    }
                    for text in item.summary
                ],
            }
            if item.content:
                reasoning["content"] = [
                    {
                        "type": "reasoning_text",
                        "text": item.content,
                    }
                ]
            if item.encrypted_content is not None:
                reasoning["encrypted_content"] = item.encrypted_content
            encoded.append(reasoning)
            continue

        if isinstance(item, ToolCall):
            encoded.append(
                {
                    "type": "function_call",
                    "name": item.name,
                    "arguments": item.arguments_json,
                    "call_id": item.call_id,
                }
            )
            continue

        if isinstance(item, ToolResult):
            encoded.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": item.output,
                }
            )
            continue

        if isinstance(item, OpaqueCompaction):
            if item.protocol != "responses":
                raise ModelConfigurationError(
                    "Responses cannot encode a Messages opaque compaction"
                )
            encoded.append(
                {
                    "type": "compaction",
                    "encrypted_content": item.payload,
                }
            )
            continue

        if isinstance(item, ContextCompaction):
            raise ModelConfigurationError(
                "ContextCompaction must be projected before request encoding"
            )

        raise ModelConfigurationError(
            f"unsupported interaction item at index {index}: {item!r}"
        )

    if not encoded:
        raise ModelConfigurationError("cannot sample an empty model context")
    return encoded


def _encode_tools(tools: Sequence[Any]) -> List[Dict[str, Any]]:
    encoded: List[Dict[str, Any]] = []
    seen = set()
    for index, tool in enumerate(tools):
        name = getattr(tool, "name", None)
        description = getattr(tool, "description", None)
        parameters = getattr(tool, "parameters", None)
        if not isinstance(name, str) or not name.strip():
            raise ModelConfigurationError(
                f"tool {index} must have a non-empty name"
            )
        if name in seen:
            raise ModelConfigurationError(f"duplicate tool name: {name!r}")
        seen.add(name)
        if not isinstance(description, str):
            raise ModelConfigurationError(
                f"tool {name!r} must have a string description"
            )
        if not isinstance(parameters, Mapping):
            raise ModelConfigurationError(
                f"tool {name!r} parameters must be a mapping"
            )
        encoded.append(
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": dict(parameters),
                "strict": False,
            }
        )
    return encoded


def _apply_sampling_options(
    payload: Dict[str, Any],
    options: Optional[SamplingOptions],
) -> None:
    if options is None:
        return
    unsupported = []
    if options.temperature is not None:
        unsupported.append("temperature")
    if options.top_p is not None:
        unsupported.append("top_p")
    if options.stop:
        unsupported.append("stop")
    if options.seed is not None:
        unsupported.append("seed")
    if unsupported:
        raise ModelConfigurationError(
            "Codex Responses does not support these sampling options yet: "
            + ", ".join(unsupported)
        )
    if options.max_tokens is not None:
        payload["max_output_tokens"] = options.max_tokens


def _new_identifier(factory: Callable[[], Any], field_name: str) -> str:
    value = str(factory()).strip()
    if not value or "\r" in value or "\n" in value:
        raise ModelConfigurationError(
            f"{field_name} factory returned an invalid value"
        )
    return value


def _latest_metadata_value(
    items: Sequence[InteractionItem],
    field_name: str,
    *,
    start: int = 0,
) -> Optional[str]:
    for item in reversed(items[start:]):
        if isinstance(item, TurnMetadata):
            value = getattr(item, field_name)
            if value is not None:
                return value
    return None


def _resolve_provider_state(
    context: ModelContext,
    identifier_factory: Callable[[], Any],
) -> _ProviderState:
    items = context.items
    session_init = items[0] if items else None
    if isinstance(session_init, Init):
        session_id = session_init.session_id
        persist_session_id = False
    else:
        session_id = _latest_metadata_value(
            items,
            "provider_session_id",
        )
        if session_id is None:
            session_id = _new_identifier(
                identifier_factory,
                "provider_session_id",
            )
        persist_session_id = True

    current_turn_start = 0
    for index, item in enumerate(items):
        if isinstance(item, UserInteractionBoundary):
            current_turn_start = index + 1

    turn_id = _latest_metadata_value(
        items,
        "provider_turn_id",
        start=current_turn_start,
    )
    if turn_id is None:
        turn_id = _new_identifier(
            identifier_factory,
            "provider_turn_id",
        )

    turn_state = _latest_metadata_value(
        items,
        "provider_turn_state",
        start=current_turn_start,
    )
    return _ProviderState(
        session_id=session_id,
        turn_id=turn_id,
        turn_state=turn_state,
        persist_session_id=persist_session_id,
    )


def _parse_sse_payload(
    event_name: Optional[str],
    data_lines: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if not data_lines:
        return None
    data = "\n".join(data_lines)
    if data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ModelResponseError(
            f"Responses stream contains invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelResponseError(
            "Responses stream event data must be a JSON object"
        )
    if event_name and "type" not in payload:
        payload["type"] = event_name
    return payload


def _iter_sse_payloads(response: Iterable[Any]) -> Iterator[Dict[str, Any]]:
    event_name: Optional[str] = None
    data_lines: List[str] = []
    try:
        iterator = iter(response)
    except TypeError as exc:
        raise ModelResponseError(
            "Responses HTTP body must be an iterable SSE stream"
        ) from exc

    for raw_line in iterator:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        elif isinstance(raw_line, str):
            line = raw_line
        else:
            raise ModelResponseError(
                "Responses SSE lines must be bytes or strings"
            )
        line = line.rstrip("\n")
        if line.endswith("\r"):
            line = line[:-1]

        if line == "":
            payload = _parse_sse_payload(event_name, data_lines)
            if payload is not None:
                yield payload
            event_name = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue

        field_name, separator, value = line.partition(":")
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field_name == "event":
            event_name = value
        elif field_name == "data":
            data_lines.append(value)

    payload = _parse_sse_payload(event_name, data_lines)
    if payload is not None:
        yield payload


def _require_output_string(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> str:
    if not isinstance(value, str):
        raise ModelResponseError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise ModelResponseError(f"{field_name} must not be empty")
    return value


def _require_output_list(value: Any, field_name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise ModelResponseError(f"{field_name} must be a list")
    return value


def _decode_text_entries(
    value: Any,
    *,
    field_name: str,
    expected_type: str,
) -> Tuple[str, ...]:
    if value is None:
        return ()
    entries = _require_output_list(value, field_name)
    decoded: List[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ModelResponseError(
                f"{field_name}[{index}] must be an object"
            )
        entry_type = entry.get("type")
        if entry_type != expected_type:
            raise ModelResponseError(
                f"unsupported {field_name}[{index}] type: {entry_type!r}"
            )
        decoded.append(
            _require_output_string(
                entry.get("text"),
                f"{field_name}[{index}].text",
            )
        )
    return tuple(decoded)


def _decode_output_item(value: Any) -> InteractionItem:
    if not isinstance(value, Mapping):
        raise ModelResponseError("response output item must be an object")
    item_type = value.get("type")

    if item_type == "message":
        role = value.get("role")
        if role != "assistant":
            raise ModelResponseError(
                f"Responses output message has unsupported role: {role!r}"
            )
        text = "".join(
            _decode_text_entries(
                value.get("content"),
                field_name="message.content",
                expected_type="output_text",
            )
        )
        return Message(role="assistant", content=text)

    if item_type == "reasoning":
        summary = _decode_text_entries(
            value.get("summary", ()),
            field_name="reasoning.summary",
            expected_type="summary_text",
        )
        content = _decode_text_entries(
            value.get("content"),
            field_name="reasoning.content",
            expected_type="reasoning_text",
        )
        encrypted_content_value = value.get("encrypted_content")
        encrypted_content: Optional[str]
        if encrypted_content_value is None:
            encrypted_content = None
        else:
            encrypted_content = _require_output_string(
                encrypted_content_value,
                "reasoning.encrypted_content",
                allow_empty=False,
            )
        return Reasoning(
            content="\n".join(content),
            summary=summary,
            encrypted_content=encrypted_content,
        )

    if item_type == "function_call":
        name = _require_output_string(
            value.get("name"),
            "function_call.name",
            allow_empty=False,
        )
        call_id = _require_output_string(
            value.get("call_id"),
            "function_call.call_id",
            allow_empty=False,
        )
        arguments = value.get("arguments", "")
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            except (TypeError, ValueError) as exc:
                raise ModelResponseError(
                    "function_call.arguments are not JSON-compatible"
                ) from exc
        return ToolCall(
            name=name,
            call_id=call_id,
            arguments_json=arguments,
        )

    if item_type == "compaction":
        return OpaqueCompaction.from_responses(
            _require_output_string(
                value.get("encrypted_content"),
                "compaction.encrypted_content",
                allow_empty=False,
            )
        )

    raise ModelResponseError(
        f"unsupported Responses output item type: {item_type!r}"
    )


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _decode_usage(response_object: Any) -> TokenUsage:
    if not isinstance(response_object, Mapping):
        return TokenUsage()
    usage = response_object.get("usage")
    if not isinstance(usage, Mapping):
        return TokenUsage()
    input_tokens = _nonnegative_int(usage.get("input_tokens"))
    output_tokens = _nonnegative_int(usage.get("output_tokens"))
    total_tokens = _nonnegative_int(usage.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    cached_input_tokens = 0
    details = usage.get("input_tokens_details")
    if isinstance(details, Mapping):
        cached_input_tokens = min(
            input_tokens,
            _nonnegative_int(details.get("cached_tokens")),
        )
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def _optional_output_index(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelResponseError(
            "response.output_item.done output_index must be a "
            "nonnegative integer"
        )
    return value


def _failed_response_message(payload: Mapping[str, Any]) -> str:
    response_object = payload.get("response")
    if isinstance(response_object, Mapping):
        error = response_object.get("error")
        if isinstance(error, Mapping):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message
    return "response.failed event received"


def _incomplete_response_message(payload: Mapping[str, Any]) -> str:
    reason: Optional[str] = None
    response_object = payload.get("response")
    if isinstance(response_object, Mapping):
        details = response_object.get("incomplete_details")
        if isinstance(details, Mapping):
            candidate = details.get("reason")
            if isinstance(candidate, str) and candidate.strip():
                reason = candidate
    return f"incomplete Responses result: {reason or 'unknown reason'}"


def _collect_sample(
    response: Iterable[Any],
    *,
    provider_state: _ProviderState,
    captured_turn_state: Optional[str],
) -> ModelSample:
    output_items: List[Tuple[Optional[int], InteractionItem]] = []
    indexed_output_items: Dict[int, InteractionItem] = {}
    usage = TokenUsage()
    completed = False

    for payload in _iter_sse_payloads(response):
        event_type = payload.get("type")
        if event_type == "response.output_item.done":
            output_index = _optional_output_index(
                payload.get("output_index")
            )
            decoded = _decode_output_item(payload.get("item"))
            if output_index is not None:
                existing = indexed_output_items.get(output_index)
                if existing is not None:
                    if existing != decoded:
                        raise ModelResponseError(
                            "Responses stream contains conflicting completed "
                            f"output items at index {output_index}"
                        )
                    continue
                indexed_output_items[output_index] = decoded
            output_items.append((output_index, decoded))
            continue

        if event_type in {"response.completed", "response.done"}:
            usage = _decode_usage(payload.get("response"))
            completed = True
            break

        if event_type == "response.failed":
            raise ModelResponseError(_failed_response_message(payload))
        if event_type == "response.incomplete":
            raise ModelResponseError(_incomplete_response_message(payload))
        if event_type == "error":
            message = payload.get("message")
            raise ModelResponseError(
                str(message or "Responses error event received")
            )

    if not completed:
        raise ModelResponseError(
            "Responses stream closed before response.completed"
        )
    if not output_items:
        raise ModelResponseError("Responses result contains no output items")

    if all(index is not None for index, _ in output_items):
        items = tuple(
            item
            for _, item in sorted(
                output_items,
                key=lambda indexed_item: (
                    indexed_item[0]
                    if indexed_item[0] is not None
                    else -1
                ),
            )
        )
    else:
        # Codex streams commonly omit output_index. In that form, completed
        # item events are already emitted in provider order.
        items = tuple(item for _, item in output_items)
    stop_reason = (
        "tool_use"
        if any(isinstance(item, ToolCall) for item in items)
        else "end_turn"
    )
    return ModelSample(
        items=items,
        stop_reason=stop_reason,
        usage=usage,
        provider_session_id=(
            provider_state.session_id
            if provider_state.persist_session_id
            else None
        ),
        provider_turn_id=provider_state.turn_id,
        provider_turn_state=captured_turn_state,
    )


def _bounded_text(value: str, limit: int = 4096) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        payload = exc.read()
    except Exception:
        return ""
    if not isinstance(payload, (bytes, bytearray)):
        return ""
    return _bounded_text(bytes(payload).decode("utf-8", errors="replace"))


def _is_context_window_error(text: str) -> bool:
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in (
            "context window",
            "maximum context length",
            "context length exceeded",
            "too many tokens",
        )
    )


def _raise_http_error(
    status: int,
    detail: str,
    *,
    api_provider: str,
) -> NoReturn:
    if _is_context_window_error(detail):
        raise ModelContextWindowError(detail)
    if status == 401 and api_provider == "codex":
        raise ModelTransportError(
            "Codex Responses HTTP 401: authentication failed; "
            "run `codex login` to create or refresh the Codex credentials"
        )
    label = "Codex Responses" if api_provider == "codex" else "Responses"
    raise ModelTransportError(
        f"{label} HTTP {status}: {detail or 'request failed'}"
    )


def _response_header(response: Any, name: str) -> Optional[str]:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except Exception:
        return None
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        return None
    if "\r" in value or "\n" in value:
        raise ModelResponseError(
            f"Responses header {name!r} contains a newline"
        )
    return value


class CodexResponsesModel:
    """Codex-compatible Responses model with model and endpoint routing."""

    def __init__(
        self,
        endpoint: Optional[StreamingResponsesEndpoint] = None,
        *,
        model: Optional[str] = None,
        auth: Optional[CodexAuth] = None,
        api_url: Optional[str] = None,
        request_timeout_seconds: Optional[float] = None,
        codex_home: Optional[CodexAuthPath] = None,
        auth_file: Optional[CodexAuthPath] = None,
        opener: Optional[Callable[..., Any]] = None,
        identifier_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        if endpoint is not None:
            if not isinstance(endpoint, StreamingResponsesEndpoint):
                raise TypeError(
                    "endpoint must be StreamingResponsesEndpoint or None"
                )
            conflicting_options = [
                name
                for name, value in (
                    ("model", model),
                    ("auth", auth),
                    ("api_url", api_url),
                    (
                        "request_timeout_seconds",
                        request_timeout_seconds,
                    ),
                    ("codex_home", codex_home),
                    ("auth_file", auth_file),
                )
                if value is not None
            ]
            if conflicting_options:
                raise ModelConfigurationError(
                    "endpoint cannot be combined with endpoint-construction "
                    "options: "
                    + ", ".join(conflicting_options)
                )
            resolved_endpoint = endpoint
        else:
            if model is None:
                raise ModelConfigurationError(
                    "model is required when endpoint is not supplied"
                )
            if not isinstance(model, str):
                raise TypeError("model must be a string")
            model = model.strip()
            if not model:
                raise ModelConfigurationError("model must not be empty")
            resolved_url, model, resolved_timeout = _normalize_configuration(
                _resolve_default_codex_api_url(model) if api_url is None else api_url,
                model, 300.0 if request_timeout_seconds is None else request_timeout_seconds,
            )
            if auth is not None:
                if not isinstance(auth, CodexAuth):
                    raise TypeError("auth must be CodexAuth or None")
                if codex_home is not None or auth_file is not None:
                    raise ModelConfigurationError(
                        "auth cannot be combined with codex_home or auth_file"
                    )
                resolved_auth = auth
            else:
                resolved_auth = _load_default_model_auth(
                    model,
                    codex_home=codex_home,
                    auth_file=auth_file,
                )
            resolved_endpoint = StreamingResponsesEndpoint(
                api_url=resolved_url,
                model=model,
                bearer_token=resolved_auth.access_token,
                account_id=resolved_auth.account_id,
                api_provider="codex",
                request_timeout_seconds=resolved_timeout,
            )

        if identifier_factory is not None and not callable(identifier_factory):
            raise TypeError("identifier_factory must be callable or None")
        self.endpoint = resolved_endpoint
        self._opener = opener or urllib.request.urlopen
        self._identifier_factory = identifier_factory or uuid.uuid4

    @property
    def default_context_tokens(self) -> Optional[int]:
        """Catalog default window in tokens, or None if unknown; not enforced."""
        return _resolve_request_route(self.endpoint).default_context_tokens

    @property
    def max_context_tokens(self) -> Optional[int]:
        """Catalog window override ceiling in tokens, or None; not enforced."""
        return _resolve_request_route(self.endpoint).max_context_tokens

    def _build_request_payload(
        self,
        context: ModelContext,
        tools: Sequence[Any],
        options: Optional[SamplingOptions],
    ) -> Tuple[Dict[str, Any], _ProviderState]:
        if not isinstance(context, ModelContext):
            raise TypeError("context must be ModelContext")
        context.assert_model_ready()

        if self.endpoint.api_provider == "codex":
            provider_state = _resolve_provider_state(
                context,
                self._identifier_factory,
            )
        else:
            provider_state = _ProviderState()

        request_route = _resolve_request_route(self.endpoint)
        payload: Dict[str, Any] = {
            "model": request_route.api_model,
            "input": _encode_context_items(context.model_items()),
            "tools": _encode_tools(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        reasoning: Dict[str, str] = {}
        if request_route.reasoning_effort is not None:
            reasoning["effort"] = request_route.reasoning_effort
        if request_route.reasoning_summary is not None:
            reasoning["summary"] = request_route.reasoning_summary
        if reasoning:
            payload["reasoning"] = reasoning
        if request_route.text_verbosity is not None:
            payload["text"] = {"verbosity": request_route.text_verbosity}
        if provider_state.session_id is not None:
            payload["prompt_cache_key"] = provider_state.session_id
        _apply_sampling_options(payload, options)
        return payload, provider_state

    def _build_headers(
        self,
        provider_state: _ProviderState,
    ) -> Dict[str, str]:
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self.endpoint.bearer_token}",
            "Content-Type": "application/json",
            "User-Agent": "pythia-interaction/0.1",
        }
        if self.endpoint.api_provider != "codex":
            return headers

        if self.endpoint.account_id is not None:
            headers["ChatGPT-Account-ID"] = self.endpoint.account_id
        if provider_state.session_id is not None:
            headers["session_id"] = provider_state.session_id
        if provider_state.turn_id is not None:
            headers["x-codex-turn-metadata"] = json.dumps(
                {
                    "turn_id": provider_state.turn_id,
                    "sandbox": "none",
                },
                separators=(",", ":"),
            )
        if provider_state.turn_state is not None:
            headers[X_CODEX_TURN_STATE_HEADER] = provider_state.turn_state
        return headers

    def sample(
        self,
        context: ModelContext,
        *,
        tools: Sequence[Any] = (),
        options: Optional[SamplingOptions] = None,
    ) -> ModelSample:
        if options is not None and not isinstance(options, SamplingOptions):
            raise TypeError("options must be SamplingOptions or None")
        payload, provider_state = self._build_request_payload(
            context,
            tools,
            options,
        )
        try:
            request_data = json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ModelConfigurationError(
                "Responses request is not JSON-serializable"
            ) from exc

        request = urllib.request.Request(
            self.endpoint.url,
            data=request_data,
            headers=self._build_headers(provider_state),
            method="POST",
        )
        try:
            response = self._opener(
                request,
                timeout=self.endpoint.request_timeout_seconds,
            )
        except urllib.error.HTTPError as exc:
            detail = _read_http_error_body(exc) or _bounded_text(str(exc))
            _raise_http_error(
                exc.code,
                detail,
                api_provider=self.endpoint.api_provider,
            )
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise ModelTimeoutError(str(exc)) from exc
            raise ModelTransportError(str(exc)) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ModelTimeoutError(str(exc)) from exc
        except OSError as exc:
            raise ModelTransportError(str(exc)) from exc

        try:
            status = getattr(response, "status", None)
            if isinstance(status, int) and not 200 <= status < 300:
                try:
                    raw = response.read()
                except (TimeoutError, socket.timeout) as exc:
                    raise ModelTimeoutError(str(exc)) from exc
                except OSError as exc:
                    raise ModelTransportError(str(exc)) from exc
                if not isinstance(raw, (bytes, bytearray)):
                    raise ModelResponseError(
                        "HTTP response body must be bytes"
                    )
                detail = _bounded_text(
                    bytes(raw).decode("utf-8", errors="replace")
                )
                _raise_http_error(
                    status,
                    detail,
                    api_provider=self.endpoint.api_provider,
                )

            captured_turn_state = provider_state.turn_state
            if (
                self.endpoint.api_provider == "codex"
                and captured_turn_state is None
            ):
                captured_turn_state = _response_header(
                    response,
                    X_CODEX_TURN_STATE_HEADER,
                )
            try:
                return _collect_sample(
                    response,
                    provider_state=provider_state,
                    captured_turn_state=captured_turn_state,
                )
            except (TimeoutError, socket.timeout) as exc:
                raise ModelTimeoutError(str(exc)) from exc
            except OSError as exc:
                raise ModelTransportError(str(exc)) from exc
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()


__all__ = [
    "CODEX_RESPONSES_API_URL",
    "CodexResponsesModel",
    "META_RESPONSES_API_URL",
    "OPENAI_RESPONSES_API_URL",
    "StreamingResponsesEndpoint",
    "X_CODEX_TURN_STATE_HEADER",
]
