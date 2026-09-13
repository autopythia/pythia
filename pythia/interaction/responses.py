from __future__ import annotations

import base64
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
from dataclasses import replace
from threading import Lock
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

from .codex_auth import CodexAuth
from .codex_auth import CodexAuthPath
from .codex_auth import CodexCredentials
from .codex_auth import _resolve_auth_file
from .codex_auth import load_codex_auth
from .codex_auth import load_codex_credentials
from .codex_login import refresh_codex_credentials
from .compaction import CompactionError
from .compaction import CompactionResult
from .compaction import DEFAULT_SUMMARY_PREFIX
from .compaction import _leading_instruction_prefix
from .compaction import _select_retained_user_messages
from .compaction import _timed_compact
from .context import ContextValidationError
from .context import ModelContext
from .items import CompactionMetadata
from .items import ContextPrefix
from .items import Init
from .items import Instructions
from .items import InteractionItem
from .items import Message
from .items import ModelFailure
from .items import ModelSampleBoundary
from .items import OpaqueCompaction
from .items import Reasoning
from .items import ToolCall
from .items import ToolResult
from .items import SampleMetadata
from .items import TurnSummary
from .items import UserInteractionBoundary
from .model import ModelConfigurationError
from .model import ModelAuthenticationError
from .model import ModelContextWindowError
from .model import ModelError
from .model import ModelResponseError
from .model import ModelSample
from .model import ModelTimeoutError
from .model import ModelTransportError
from .model import SamplingOptions
from .model import _timed_sample
from .model_catalog import CODEX_RESPONSES_API_URL
from .model_catalog import META_RESPONSES_API_URL
from .model_catalog import OPENAI_RESPONSES_API_URL
from .model_catalog import ModelSpec
from .model_catalog import get_model_route
from .model_catalog import get_model_spec
from .timeouts import DEFAULT_REQUEST_TIMEOUT_SECONDS
from .usage import TokenUsage


X_CODEX_TURN_STATE_HEADER = "x-codex-turn-state"
REMOTE_COMPACTION_V2_RETAINED_USER_MESSAGE_TOKENS = 64_000
_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_MAX_TRANSIENT_HTTP_RETRIES = 1
_MAX_DIAGNOSTIC_VALUE_CHARS = 256
_MAX_DIAGNOSTIC_EVENT_TYPES = 32


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
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS

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


@dataclass(frozen=True)
class _RemoteCompactionResponse:
    item: OpaqueCompaction
    usage: TokenUsage = field(default_factory=TokenUsage)
    provider_session_id: Optional[str] = field(default=None, repr=False)
    provider_turn_id: Optional[str] = field(default=None, repr=False)
    provider_turn_state: Optional[str] = field(default=None, repr=False)
    provider_response_id: Optional[str] = field(default=None, repr=False)
    request_attempts: int = 1
    recovery: Tuple[str, ...] = ()


def _resolve_model_spec(endpoint: StreamingResponsesEndpoint) -> Optional[ModelSpec]:
    profile = "codex" if endpoint.api_provider == "codex" else "responses"
    return get_model_spec(profile, endpoint.model)


@dataclass(frozen=True)
class _CredentialSnapshot:
    auth: CodexAuth
    credentials: Optional[CodexCredentials] = field(default=None, repr=False)


class _StaticCredentialSource:
    kind = "static"

    def __init__(self, auth: CodexAuth) -> None:
        self._snapshot = _CredentialSnapshot(auth)

    def load(self) -> _CredentialSnapshot:
        return self._snapshot

    def refresh(
        self,
        snapshot: _CredentialSnapshot,
        *,
        timeout_seconds: float,
        opener: Optional[Callable[..., Any]],
    ) -> Optional[_CredentialSnapshot]:
        del snapshot, timeout_seconds, opener
        return None


class _EnvironmentCredentialSource:
    kind = "environment"

    def __init__(self, variable: str) -> None:
        self.variable = variable

    def load(self) -> _CredentialSnapshot:
        value = os.environ.get(self.variable)
        if value is None or not value.strip():
            raise ModelConfigurationError(
                f"{self.variable} is required for this Responses model"
            )
        return _CredentialSnapshot(CodexAuth(access_token=value))

    def refresh(
        self,
        snapshot: _CredentialSnapshot,
        *,
        timeout_seconds: float,
        opener: Optional[Callable[..., Any]],
    ) -> Optional[_CredentialSnapshot]:
        del timeout_seconds, opener
        current = self.load()
        return current if current.auth != snapshot.auth else None


class _FileCredentialSource:
    kind = "codex_file"

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> _CredentialSnapshot:
        credentials = load_codex_credentials(auth_file=self.path)
        return _CredentialSnapshot(credentials.auth, credentials)

    def refresh(
        self,
        snapshot: _CredentialSnapshot,
        *,
        timeout_seconds: float,
        opener: Optional[Callable[..., Any]],
    ) -> Optional[_CredentialSnapshot]:
        if snapshot.credentials is None:
            return None
        if snapshot.credentials.auth_mode not in {None, "chatgpt"}:
            return None
        refreshed = refresh_codex_credentials(
            snapshot.credentials,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )
        return _CredentialSnapshot(refreshed.auth, refreshed)


def _load_default_model_auth(
    model: str,
    *,
    codex_home: Optional[CodexAuthPath],
    auth_file: Optional[CodexAuthPath],
) -> CodexAuth:
    route = get_model_route("codex", model)
    api_key_environment_variable = route.api_key_environment_variable
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


def _default_credential_source(
    model: str,
    *,
    codex_home: Optional[CodexAuthPath],
    auth_file: Optional[CodexAuthPath],
):
    route = get_model_route("codex", model)
    variable = route.api_key_environment_variable
    if variable is not None and codex_home is None and auth_file is None:
        return _EnvironmentCredentialSource(variable)
    path = _resolve_auth_file(
        codex_home=codex_home,
        auth_file=auth_file,
    ).resolve()
    return _FileCredentialSource(path)


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
                ModelFailure,
                Init,
                SampleMetadata,
                CompactionMetadata,
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

        if isinstance(item, ContextPrefix):
            raise ModelConfigurationError(
                "ContextPrefix must be projected before request encoding"
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
    if options.max_output_tokens is not None:
        payload["max_output_tokens"] = options.max_output_tokens


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
        if isinstance(item, (SampleMetadata, CompactionMetadata)):
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
        session_id = session_init.prefix_id
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


@dataclass
class _StreamTrace:
    event_count: int = 0
    last_event_type: Optional[str] = None
    last_sequence_number: Optional[int] = None
    response_id: Optional[str] = None
    error_code: Optional[str] = None
    event_type_counts: Dict[str, int] = field(default_factory=dict)
    forbidden_values: Tuple[str, ...] = ()

    def observe(self, payload: Mapping[str, Any]) -> None:
        self.event_count += 1
        event_type = payload.get("type")
        self.last_event_type = _safe_diagnostic_value(
            event_type,
            self.forbidden_values,
        )
        if self.last_event_type is not None and (
            self.last_event_type in self.event_type_counts
            or len(self.event_type_counts) < _MAX_DIAGNOSTIC_EVENT_TYPES
        ):
            self.event_type_counts[self.last_event_type] = (
                self.event_type_counts.get(self.last_event_type, 0) + 1
            )
        sequence = payload.get("sequence_number")
        if isinstance(sequence, int) and not isinstance(sequence, bool) and sequence >= 0:
            self.last_sequence_number = sequence
        if event_type == "response.created":
            response = payload.get("response")
            if isinstance(response, Mapping):
                self.response_id = _safe_diagnostic_value(
                    response.get("id"),
                    self.forbidden_values,
                )
        if event_type == "response.incomplete":
            response = payload.get("response")
            if isinstance(response, Mapping):
                details = response.get("incomplete_details")
                if isinstance(details, Mapping):
                    self.error_code = _safe_diagnostic_value(
                        details.get("reason"),
                        self.forbidden_values,
                    )
        elif event_type in {"response.failed", "error"}:
            response = payload.get("response")
            error = (
                response.get("error")
                if isinstance(response, Mapping)
                else payload.get("error")
            )
            if isinstance(error, Mapping):
                self.error_code = _safe_diagnostic_value(
                    error.get("code"),
                    self.forbidden_values,
                )


def _safe_diagnostic_value(
    value: Any,
    forbidden_values: Tuple[str, ...] = (),
) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        return None
    if any(secret and secret in normalized for secret in forbidden_values):
        return None
    return normalized[:_MAX_DIAGNOSTIC_VALUE_CHARS]


def _safe_header(
    headers: Any,
    name: str,
    forbidden_values: Tuple[str, ...] = (),
) -> Optional[str]:
    if headers is None:
        return None
    try:
        return _safe_diagnostic_value(headers.get(name), forbidden_values)
    except Exception:
        return None


def _error_code_from_json(value: Any) -> Optional[str]:
    if not isinstance(value, Mapping):
        return None
    error = value.get("error")
    if not isinstance(error, Mapping):
        return None
    return _safe_diagnostic_value(error.get("code"))


def _http_auth_error_code(
    headers: Any,
    forbidden_values: Tuple[str, ...],
) -> Optional[str]:
    encoded = _safe_header(headers, "x-error-json", forbidden_values)
    if encoded is not None and len(encoded) <= 4096:
        try:
            decoded = base64.b64decode(encoded, validate=True)
            code = _error_code_from_json(json.loads(decoded))
            if code is not None:
                return code
        except Exception:
            pass
    return None


def _http_body_error_code(
    detail: str,
    forbidden_values: Tuple[str, ...],
) -> Optional[str]:
    try:
        code = _error_code_from_json(json.loads(detail))
        if code is not None and any(
            secret and secret in code for secret in forbidden_values
        ):
            return None
        return code
    except Exception:
        return None


def _diagnostic_request_id(
    headers: Any,
    forbidden_values: Tuple[str, ...] = (),
) -> Optional[str]:
    return _safe_header(headers, "x-request-id", forbidden_values) or _safe_header(
        headers,
        "x-oai-request-id",
        forbidden_values,
    )


def _ordered_output_items(
    output_items: List[Tuple[Optional[int], InteractionItem]],
) -> Tuple[InteractionItem, ...]:
    if all(index is not None for index, _ in output_items):
        return tuple(
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
    # Codex streams commonly omit output_index. In that form, completed item
    # events are already emitted in provider order.
    return tuple(item for _, item in output_items)


def _stream_failure(
    message: str,
    *,
    exception_message: Optional[str] = None,
    category: str,
    trace: _StreamTrace,
    output_items: List[Tuple[Optional[int], InteractionItem]],
    provider: str,
    model: str,
    auth_source: str,
    attempt_count: int,
    recovery: Tuple[str, ...],
    headers: Any,
) -> ModelResponseError:
    completed = _ordered_output_items(output_items)
    failure = ModelFailure(
        category=category,
        message=message,
        provider=provider,
        model=model,
        auth_source=auth_source,
        request_id=_diagnostic_request_id(headers, trace.forbidden_values),
        response_id=trace.response_id,
        cf_ray=_safe_header(headers, "cf-ray", trace.forbidden_values),
        authorization_error=_safe_header(
            headers,
            "x-openai-authorization-error",
            trace.forbidden_values,
        ),
        auth_error_code=_http_auth_error_code(
            headers,
            trace.forbidden_values,
        ),
        error_code=trace.error_code,
        attempt_count=attempt_count,
        event_count=trace.event_count,
        event_types=tuple(
            f"{name}:{count}"
            for name, count in sorted(trace.event_type_counts.items())
        ),
        completed_item_count=len(completed),
        last_event_type=trace.last_event_type,
        last_sequence_number=trace.last_sequence_number,
        recovery=recovery,
    )
    return ModelResponseError(
        exception_message or message,
        failure=failure,
        completed_items=completed,
    )


def _collect_sample(
    response: Iterable[Any],
    *,
    provider_state: _ProviderState,
    captured_turn_state: Optional[str],
    provider: str = "responses",
    model: str = "unknown",
    auth_source: str = "static",
    attempt_count: int = 1,
    recovery: Tuple[str, ...] = (),
    response_headers: Any = None,
    forbidden_values: Tuple[str, ...] = (),
) -> ModelSample:
    output_items: List[Tuple[Optional[int], InteractionItem]] = []
    indexed_output_items: Dict[int, InteractionItem] = {}
    usage = TokenUsage()
    completed = False
    trace = _StreamTrace(forbidden_values=forbidden_values)
    iterator = _iter_sse_payloads(response)

    while True:
        try:
            payload = next(iterator)
        except StopIteration:
            break
        except (TimeoutError, socket.timeout) as exc:
            partial = _stream_failure(
                "Responses stream timed out before response.completed",
                category="stream_timeout",
                trace=trace,
                output_items=output_items,
                provider=provider,
                model=model,
                auth_source=auth_source,
                attempt_count=attempt_count,
                recovery=recovery,
                headers=response_headers,
            )
            raise ModelTimeoutError(
                str(partial),
                failure=partial.failure,
                completed_items=partial.completed_items,
            ) from exc
        except OSError as exc:
            partial = _stream_failure(
                "Responses stream failed before response.completed",
                category="stream_transport",
                trace=trace,
                output_items=output_items,
                provider=provider,
                model=model,
                auth_source=auth_source,
                attempt_count=attempt_count,
                recovery=recovery,
                headers=response_headers,
            )
            raise ModelTransportError(
                str(partial),
                failure=partial.failure,
                completed_items=partial.completed_items,
            ) from exc
        except ModelResponseError as exc:
            raise _stream_failure(
                "Responses stream contained an invalid event",
                exception_message=str(exc),
                category="invalid_sse",
                trace=trace,
                output_items=output_items,
                provider=provider,
                model=model,
                auth_source=auth_source,
                attempt_count=attempt_count,
                recovery=recovery,
                headers=response_headers,
            ) from exc
        trace.observe(payload)
        event_type = payload.get("type")
        if event_type == "response.output_item.done":
            try:
                output_index = _optional_output_index(
                    payload.get("output_index")
                )
                decoded = _decode_output_item(payload.get("item"))
            except ModelResponseError as exc:
                raise _stream_failure(
                    "Responses stream contained an invalid completed item",
                    exception_message=str(exc),
                    category="invalid_output_item",
                    trace=trace,
                    output_items=output_items,
                    provider=provider,
                    model=model,
                    auth_source=auth_source,
                    attempt_count=attempt_count,
                    recovery=recovery,
                    headers=response_headers,
                ) from exc
            if output_index is not None:
                existing = indexed_output_items.get(output_index)
                if existing is not None:
                    if existing != decoded:
                        raise _stream_failure(
                            "Responses stream contained conflicting completed items",
                            category="conflicting_output_items",
                            trace=trace,
                            output_items=output_items,
                            provider=provider,
                            model=model,
                            auth_source=auth_source,
                            attempt_count=attempt_count,
                            recovery=recovery,
                            headers=response_headers,
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
            raise _stream_failure(
                "Responses reported a failed response",
                category="response_failed",
                trace=trace,
                output_items=output_items,
                provider=provider,
                model=model,
                auth_source=auth_source,
                attempt_count=attempt_count,
                recovery=recovery,
                headers=response_headers,
            )
        if event_type == "response.incomplete":
            raise _stream_failure(
                "Responses reported an incomplete response",
                category="response_incomplete",
                trace=trace,
                output_items=output_items,
                provider=provider,
                model=model,
                auth_source=auth_source,
                attempt_count=attempt_count,
                recovery=recovery,
                headers=response_headers,
            )
        if event_type == "error":
            raise _stream_failure(
                "Responses error event received",
                category="response_error_event",
                trace=trace,
                output_items=output_items,
                provider=provider,
                model=model,
                auth_source=auth_source,
                attempt_count=attempt_count,
                recovery=recovery,
                headers=response_headers,
            )

    if not completed:
        raise _stream_failure(
            "Responses stream closed before response.completed",
            category="stream_closed",
            trace=trace,
            output_items=output_items,
            provider=provider,
            model=model,
            auth_source=auth_source,
            attempt_count=attempt_count,
            recovery=recovery,
            headers=response_headers,
        )
    if not output_items:
        raise _stream_failure(
            "Responses result contains no output items",
            category="empty_response",
            trace=trace,
            output_items=output_items,
            provider=provider,
            model=model,
            auth_source=auth_source,
            attempt_count=attempt_count,
            recovery=recovery,
            headers=response_headers,
        )

    items = _ordered_output_items(output_items)
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
        request_attempts=attempt_count,
        recovery=recovery,
    )


def _collect_remote_compaction_v2(
    response: Iterable[Any],
    *,
    provider_state: _ProviderState,
    captured_turn_state: Optional[str],
    provider: str = "responses",
    model: str = "unknown",
    auth_source: str = "static",
    attempt_count: int = 1,
    recovery: Tuple[str, ...] = (),
    response_headers: Any = None,
    forbidden_values: Tuple[str, ...] = (),
) -> _RemoteCompactionResponse:
    """Collect one opaque V2 checkpoint without treating output as a sample.

    Remote compaction can emit unrelated output items. They are deliberately
    ignored: in particular, a function call produced during compaction is never
    returned to the interaction controller and can therefore never execute.
    """
    compaction_items: List[Tuple[Optional[int], InteractionItem]] = []
    indexed_compactions: Dict[int, InteractionItem] = {}
    usage = TokenUsage()
    completed = False
    provider_response_id: Optional[str] = None
    trace = _StreamTrace(forbidden_values=forbidden_values)
    iterator = _iter_sse_payloads(response)

    def failure(
        message: str,
        *,
        category: str,
        exception_message: Optional[str] = None,
    ) -> ModelResponseError:
        return _stream_failure(
            message,
            exception_message=exception_message,
            category=category,
            trace=trace,
            output_items=compaction_items,
            provider=provider,
            model=model,
            auth_source=auth_source,
            attempt_count=attempt_count,
            recovery=recovery,
            headers=response_headers,
        )

    while True:
        try:
            payload = next(iterator)
        except StopIteration:
            break
        except (TimeoutError, socket.timeout) as exc:
            partial = failure(
                "Remote Responses compaction timed out before response.completed",
                category="stream_timeout",
            )
            raise ModelTimeoutError(
                str(partial),
                failure=partial.failure,
                completed_items=partial.completed_items,
            ) from exc
        except OSError as exc:
            partial = failure(
                "Remote Responses compaction failed before response.completed",
                category="stream_transport",
            )
            raise ModelTransportError(
                str(partial),
                failure=partial.failure,
                completed_items=partial.completed_items,
            ) from exc
        except ModelResponseError as exc:
            raise failure(
                "Remote Responses compaction stream contained an invalid event",
                exception_message=str(exc),
                category="invalid_sse",
            ) from exc

        trace.observe(payload)
        event_type = payload.get("type")
        if event_type == "response.output_item.done":
            raw_item = payload.get("item")
            # Unknown and unrelated output is not part of the compaction
            # contract. Avoid decoding it through normal sample handling.
            if not isinstance(raw_item, Mapping) or raw_item.get("type") != "compaction":
                continue
            try:
                output_index = _optional_output_index(payload.get("output_index"))
                decoded = _decode_output_item(raw_item)
            except ModelResponseError as exc:
                raise failure(
                    "Remote Responses compaction contained an invalid checkpoint",
                    exception_message=str(exc),
                    category="invalid_output_item",
                ) from exc
            if not isinstance(decoded, OpaqueCompaction):
                raise failure(
                    "Remote Responses compaction returned a non-opaque checkpoint",
                    category="invalid_output_item",
                )
            if output_index is not None:
                existing = indexed_compactions.get(output_index)
                if existing is not None:
                    if existing != decoded:
                        raise failure(
                            "Remote Responses compaction contained conflicting checkpoints",
                            category="conflicting_output_items",
                        )
                    continue
                indexed_compactions[output_index] = decoded
            compaction_items.append((output_index, decoded))
            continue

        if event_type == "response.completed":
            response_object = payload.get("response")
            usage = _decode_usage(response_object)
            if isinstance(response_object, Mapping):
                provider_response_id = _safe_diagnostic_value(
                    response_object.get("id"),
                    forbidden_values,
                )
            if provider_response_id is None:
                provider_response_id = trace.response_id
            completed = True
            break
        if event_type == "response.failed":
            raise failure(
                "Remote Responses compaction reported a failed response",
                category="response_failed",
            )
        if event_type == "response.incomplete":
            raise failure(
                "Remote Responses compaction reported an incomplete response",
                category="response_incomplete",
            )
        if event_type == "error":
            raise failure(
                "Remote Responses compaction error event received",
                category="response_error_event",
            )

    if not completed:
        raise failure(
            "Remote Responses compaction stream closed before response.completed",
            category="stream_closed",
        )
    ordered = _ordered_output_items(compaction_items)
    if len(ordered) != 1:
        raise failure(
            "Remote Responses compaction expected exactly one opaque checkpoint, "
            f"got {len(ordered)}",
            category="invalid_compaction_count",
        )
    item = ordered[0]
    assert isinstance(item, OpaqueCompaction)
    return _RemoteCompactionResponse(
        item=item,
        usage=usage,
        provider_session_id=(
            provider_state.session_id
            if provider_state.persist_session_id
            else None
        ),
        provider_turn_id=provider_state.turn_id,
        provider_turn_state=captured_turn_state,
        provider_response_id=provider_response_id,
        request_attempts=attempt_count,
        recovery=recovery,
    )


def _bounded_text(value: str, limit: int = 4096) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _read_http_body(stream: Any) -> str:
    try:
        try:
            payload = stream.read(1_048_577)
        except TypeError:
            # Keep compatibility with small injectable response fakes.
            payload = stream.read()
    except Exception:
        return ""
    if not isinstance(payload, (bytes, bytearray)):
        return ""
    if len(payload) > 1_048_576:
        return ""
    return _bounded_text(bytes(payload).decode("utf-8", errors="replace"))


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    return _read_http_body(exc)


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


def _http_failure(
    status: int,
    detail: str,
    *,
    api_provider: str,
    model: str,
    auth_source: str,
    headers: Any,
    attempt_count: int,
    recovery: Tuple[str, ...],
    forbidden_values: Tuple[str, ...] = (),
) -> ModelError:
    label = "Codex Responses" if api_provider == "codex" else "Responses"
    request_id = _diagnostic_request_id(headers, forbidden_values)
    cf_ray = _safe_header(headers, "cf-ray", forbidden_values)
    authorization_error = _safe_header(
        headers,
        "x-openai-authorization-error",
        forbidden_values,
    )
    auth_error_code = _http_auth_error_code(headers, forbidden_values)
    error_code = _http_body_error_code(detail, forbidden_values)
    message = f"{label} HTTP {status}: request failed"
    category = "http_error"
    error_type = ModelTransportError
    if _is_context_window_error(detail):
        message = f"{label} HTTP {status}: context window exceeded"
        category = "context_window"
        error_type = ModelContextWindowError
    if status == 401:
        if api_provider == "codex" and auth_source != "environment":
            message = (
                "Codex Responses HTTP 401: authentication failed after "
                "credential recovery; run `codex login` to refresh the "
                "Codex credentials"
            )
        elif auth_source == "environment":
            message = (
                f"{label} HTTP 401: the environment credential was rejected; "
                "update it and restart the process"
            )
        else:
            message = f"{label} HTTP 401: authentication failed"
        category = "authentication"
        error_type = ModelAuthenticationError
    failure = ModelFailure(
        category=category,
        message=message,
        provider=api_provider,
        model=model,
        auth_source=auth_source,
        http_status=status,
        request_id=request_id,
        cf_ray=cf_ray,
        authorization_error=authorization_error,
        auth_error_code=auth_error_code,
        error_code=error_code,
        attempt_count=attempt_count,
        recovery=recovery,
    )
    return error_type(message, failure=failure)


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
        auth_opener: Optional[Callable[..., Any]] = None,
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
            credential_source = _StaticCredentialSource(
                CodexAuth(endpoint.bearer_token, endpoint.account_id)
            )
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
                get_model_route("codex", model).api_url if api_url is None else api_url,
                model,
                (
                    DEFAULT_REQUEST_TIMEOUT_SECONDS
                    if request_timeout_seconds is None
                    else request_timeout_seconds
                ),
            )
            if auth is not None:
                if not isinstance(auth, CodexAuth):
                    raise TypeError("auth must be CodexAuth or None")
                if codex_home is not None or auth_file is not None:
                    raise ModelConfigurationError(
                        "auth cannot be combined with codex_home or auth_file"
                    )
                resolved_auth = auth
                credential_source = _StaticCredentialSource(auth)
            else:
                credential_source = _default_credential_source(
                    model,
                    codex_home=codex_home,
                    auth_file=auth_file,
                )
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
        if auth_opener is not None and not callable(auth_opener):
            raise TypeError("auth_opener must be callable or None")
        self.endpoint = resolved_endpoint
        self._opener = opener or urllib.request.urlopen
        self._auth_opener = auth_opener
        self._identifier_factory = identifier_factory or uuid.uuid4
        self._credential_source = credential_source
        self._expected_account_id = resolved_endpoint.account_id
        self._credential_lock = Lock()

    @property
    def auto_compact_context_tokens(self) -> Optional[int]:
        """Catalog threshold for caller-owned automatic compaction, or None."""
        spec = _resolve_model_spec(self.endpoint)
        return (
            None
            if spec is None
            else spec.limits.auto_compact_context_tokens
        )

    @property
    def max_context_tokens(self) -> Optional[int]:
        """Catalog window override ceiling in tokens, or None; not enforced."""
        spec = _resolve_model_spec(self.endpoint)
        return None if spec is None else spec.limits.max_context_tokens

    @property
    def max_output_tokens(self) -> Optional[int]:
        """Catalog output ceiling in tokens, or None; not enforced."""
        spec = _resolve_model_spec(self.endpoint)
        return None if spec is None else spec.limits.max_output_tokens

    @property
    def supports_remote_compaction(self) -> bool:
        """Whether this resolved route has the known Codex V2 capability.

        Responses wire compatibility alone is not capability advertisement.
        In particular, the Meta preset and arbitrary endpoint overrides must
        not receive a private compaction trigger by default.
        """
        if self.endpoint.api_provider != "codex":
            return False
        route = get_model_route("codex", self.endpoint.model)
        return (
            route.provider == "chatgpt"
            and self.endpoint.api_url == CODEX_RESPONSES_API_URL
        )

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

        spec = _resolve_model_spec(self.endpoint)
        payload: Dict[str, Any] = {
            "model": self.endpoint.model if spec is None else spec.api_model,
            "input": _encode_context_items(context.model_items()),
            "tools": _encode_tools(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        reasoning: Dict[str, str] = {}
        defaults = None if spec is None else spec.responses
        if defaults is not None:
            if defaults.reasoning_effort is not None:
                reasoning["effort"] = defaults.reasoning_effort
            if defaults.reasoning_summary is not None:
                reasoning["summary"] = defaults.reasoning_summary
        if reasoning:
            payload["reasoning"] = reasoning
        if defaults is not None and defaults.text_verbosity is not None:
            payload["text"] = {"verbosity": defaults.text_verbosity}
        if provider_state.session_id is not None:
            payload["prompt_cache_key"] = provider_state.session_id
        _apply_sampling_options(payload, options)
        return payload, provider_state

    def _build_headers(
        self,
        provider_state: _ProviderState,
        auth: Optional[CodexAuth] = None,
        *,
        beta_features: Sequence[str] = (),
    ) -> Dict[str, str]:
        auth = auth or CodexAuth(
            self.endpoint.bearer_token,
            self.endpoint.account_id,
        )
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {auth.access_token}",
            "Content-Type": "application/json",
            "User-Agent": "pythia-interaction/0.1",
        }
        normalized_beta_features = tuple(
            feature.strip()
            for feature in beta_features
            if isinstance(feature, str) and feature.strip()
        )
        if normalized_beta_features:
            headers["x-codex-beta-features"] = ",".join(
                normalized_beta_features
            )
        if self.endpoint.api_provider != "codex":
            return headers

        if auth.account_id is not None:
            headers["ChatGPT-Account-ID"] = auth.account_id
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

    def _checked_credential(self) -> _CredentialSnapshot:
        try:
            snapshot = self._credential_source.load()
        except Exception as exc:
            message = "Responses credentials could not be loaded"
            failure = ModelFailure(
                category="authentication",
                message=message,
                provider=self.endpoint.api_provider,
                model=self.endpoint.model,
                auth_source=self._credential_source.kind,
            )
            raise ModelAuthenticationError(message, failure=failure) from exc
        account_id = snapshot.auth.account_id
        if self._expected_account_id is None and account_id is not None:
            self._expected_account_id = account_id
        if (
            self._expected_account_id is not None
            and account_id != self._expected_account_id
        ):
            message = "Responses credential account changed; start a fresh session"
            failure = ModelFailure(
                category="authentication",
                message=message,
                provider=self.endpoint.api_provider,
                model=self.endpoint.model,
                auth_source=self._credential_source.kind,
            )
            raise ModelAuthenticationError(message, failure=failure)
        self.endpoint = replace(
            self.endpoint,
            bearer_token=snapshot.auth.access_token,
            account_id=account_id,
        )
        return snapshot

    def _reload_after_unauthorized(
        self,
        snapshot: _CredentialSnapshot,
    ) -> Optional[_CredentialSnapshot]:
        current = self._checked_credential()
        return current if current != snapshot else None

    def _refresh_after_unauthorized(
        self,
        snapshot: _CredentialSnapshot,
    ) -> Optional[_CredentialSnapshot]:
        try:
            refreshed = self._credential_source.refresh(
                snapshot,
                timeout_seconds=self.endpoint.request_timeout_seconds,
                opener=self._auth_opener,
            )
        except Exception:
            return None
        if refreshed is None:
            return None
        if (
            self._expected_account_id is not None
            and refreshed.auth.account_id != self._expected_account_id
        ):
            return None
        self.endpoint = replace(
            self.endpoint,
            bearer_token=refreshed.auth.access_token,
            account_id=refreshed.auth.account_id,
        )
        return refreshed

    @_timed_sample
    def sample(
        self,
        context: ModelContext,
        *,
        tools: Sequence[Any] = (),
        options: Optional[SamplingOptions] = None,
    ) -> ModelSample:
        with self._credential_lock:
            return self._sample_locked(context, tools, options)

    def _sample_locked(
        self,
        context: ModelContext,
        tools: Sequence[Any],
        options: Optional[SamplingOptions],
    ) -> ModelSample:
        if options is not None and not isinstance(options, SamplingOptions):
            raise TypeError("options must be SamplingOptions or None")
        payload, provider_state = self._build_request_payload(context, tools, options)
        return self._execute_request_locked(
            payload,
            provider_state,
            collector=_collect_sample,
        )

    def _compact_responses_v2(
        self,
        context: ModelContext,
        tools: Sequence[Any],
    ) -> _RemoteCompactionResponse:
        with self._credential_lock:
            payload, provider_state = self._build_request_payload(
                context,
                tools,
                None,
            )
            payload["input"].append({"type": "compaction_trigger"})
            return self._execute_request_locked(
                payload,
                provider_state,
                collector=_collect_remote_compaction_v2,
                beta_features=("remote_compaction_v2",),
            )

    def _execute_request_locked(
        self,
        payload: Mapping[str, Any],
        provider_state: _ProviderState,
        *,
        collector: Callable[..., Any],
        beta_features: Sequence[str] = (),
    ) -> Any:
        try:
            request_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ModelConfigurationError(
                "Responses request is not JSON-serializable"
            ) from exc

        snapshot = self._checked_credential()
        recovery: List[str] = []
        attempts = 0
        transient_retries = 0
        reloaded = False
        refreshed = False

        while True:
            attempts += 1
            request = urllib.request.Request(
                self.endpoint.url,
                data=request_data,
                headers=self._build_headers(
                    provider_state,
                    snapshot.auth,
                    beta_features=beta_features,
                ),
                method="POST",
            )
            response = None
            try:
                try:
                    response = self._opener(
                        request,
                        timeout=self.endpoint.request_timeout_seconds,
                    )
                except urllib.error.HTTPError as exc:
                    detail = _read_http_error_body(exc) or _bounded_text(str(exc))
                    status = exc.code
                    headers = exc.headers
                    exc.close()
                else:
                    status = getattr(response, "status", None)
                    headers = getattr(response, "headers", None)
                    detail = ""
                    if isinstance(status, int) and not 200 <= status < 300:
                        try:
                            raw = response.read(1_048_577)
                        except TypeError:
                            raw = response.read()
                        if not isinstance(raw, (bytes, bytearray)):
                            raise ModelResponseError("HTTP response body must be bytes")
                        detail = (
                            ""
                            if len(raw) > 1_048_576
                            else _bounded_text(
                                bytes(raw).decode("utf-8", errors="replace")
                            )
                        )

                if isinstance(status, int) and not 200 <= status < 300:
                    if status == 401 and self.endpoint.api_provider == "codex":
                        if not reloaded and self._credential_source.kind != "static":
                            reloaded = True
                            try:
                                loaded = self._reload_after_unauthorized(snapshot)
                            except ModelAuthenticationError:
                                recovery.append("credential_reload_rejected")
                                raise _http_failure(
                                    status,
                                    detail,
                                    api_provider=self.endpoint.api_provider,
                                    model=self.endpoint.model,
                                    auth_source=self._credential_source.kind,
                                    headers=headers,
                                    attempt_count=attempts,
                                    recovery=tuple(recovery),
                                    forbidden_values=(snapshot.auth.access_token,),
                                )
                            if loaded is not None:
                                snapshot = loaded
                                recovery.append("credential_reload")
                                continue
                            recovery.append("credential_reload_unchanged")
                        if not refreshed and self._credential_source.kind == "codex_file":
                            refreshed = True
                            loaded = self._refresh_after_unauthorized(snapshot)
                            if loaded is not None:
                                snapshot = loaded
                                recovery.append("oauth_refresh")
                                continue
                            recovery.append("oauth_refresh_failed")
                    if (
                        status in _RETRYABLE_HTTP_STATUSES
                        and not _is_context_window_error(detail)
                        and transient_retries < _MAX_TRANSIENT_HTTP_RETRIES
                    ):
                        transient_retries += 1
                        recovery.append(f"http_{status}_retry")
                        continue
                    raise _http_failure(
                        status,
                        detail,
                        api_provider=self.endpoint.api_provider,
                        model=self.endpoint.model,
                        auth_source=self._credential_source.kind,
                        headers=headers,
                        attempt_count=attempts,
                        recovery=tuple(recovery),
                        forbidden_values=(snapshot.auth.access_token,),
                    )

                assert response is not None
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
                    return collector(
                        response,
                        provider_state=provider_state,
                        captured_turn_state=captured_turn_state,
                        provider=self.endpoint.api_provider,
                        model=self.endpoint.model,
                        auth_source=self._credential_source.kind,
                        attempt_count=attempts,
                        recovery=tuple(recovery),
                        response_headers=headers,
                        forbidden_values=(snapshot.auth.access_token,),
                    )
                except (TimeoutError, socket.timeout) as exc:
                    raise ModelTimeoutError(str(exc)) from exc
                except OSError as exc:
                    raise ModelTransportError(str(exc)) from exc
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                    raise ModelTimeoutError(str(exc)) from exc
                raise ModelTransportError(str(exc)) from exc
            except (TimeoutError, socket.timeout) as exc:
                raise ModelTimeoutError(str(exc)) from exc
            except OSError as exc:
                raise ModelTransportError(str(exc)) from exc
            finally:
                if response is not None:
                    close = getattr(response, "close", None)
                    if callable(close):
                        close()


class ResponsesOpaqueCompactor:
    """Remote Responses V2 compactor with client-built prefix history."""

    def __init__(
        self,
        model: CodexResponsesModel,
        *,
        retained_user_message_tokens: int = (
            REMOTE_COMPACTION_V2_RETAINED_USER_MESSAGE_TOKENS
        ),
        retain_user_message: Optional[Callable[[Message], bool]] = None,
    ) -> None:
        if not isinstance(model, CodexResponsesModel):
            raise TypeError("model must be CodexResponsesModel")
        if (
            isinstance(retained_user_message_tokens, bool)
            or not isinstance(retained_user_message_tokens, int)
            or retained_user_message_tokens < 0
        ):
            raise ValueError(
                "retained_user_message_tokens must be a nonnegative integer"
            )
        if retain_user_message is not None and not callable(retain_user_message):
            raise TypeError("retain_user_message must be callable or None")
        self._model = model
        self._retained_user_message_tokens = retained_user_message_tokens
        self._retain_user_message = retain_user_message

    def _is_retained_user_message(self, message: Message) -> bool:
        if message.role != "user":
            return False
        if message.content.startswith(f"{DEFAULT_SUMMARY_PREFIX}\n"):
            return False
        if self._retain_user_message is not None:
            return bool(self._retain_user_message(message))
        return True

    @_timed_compact
    def compact(
        self,
        context: ModelContext,
        *,
        tools: Sequence[Any] = (),
    ) -> CompactionResult:
        if not isinstance(context, ModelContext):
            raise TypeError("context must be ModelContext")
        try:
            context.assert_model_ready()
        except ContextValidationError as exc:
            raise CompactionError(str(exc)) from exc

        active_items = context.model_items()
        instruction_prefix = _leading_instruction_prefix(active_items)
        user_messages = tuple(
            item
            for item in active_items
            if isinstance(item, Message)
            and self._is_retained_user_message(item)
        )
        remote = self._model._compact_responses_v2(context, tools)
        retained_users = _select_retained_user_messages(
            user_messages,
            self._retained_user_message_tokens,
        )
        checkpoint = ContextPrefix(
            prefix_items=(
                *instruction_prefix,
                *retained_users,
                remote.item,
            )
        )
        return CompactionResult(
            items=(checkpoint,),
            usage=remote.usage,
            protocol="responses_compaction_v2",
            provider_session_id=remote.provider_session_id,
            provider_turn_id=remote.provider_turn_id,
            provider_turn_state=remote.provider_turn_state,
            provider_response_id=remote.provider_response_id,
            request_attempts=remote.request_attempts,
            recovery=remote.recovery,
        )


__all__ = [
    "CODEX_RESPONSES_API_URL",
    "CodexResponsesModel",
    "META_RESPONSES_API_URL",
    "OPENAI_RESPONSES_API_URL",
    "REMOTE_COMPACTION_V2_RETAINED_USER_MESSAGE_TOKENS",
    "ResponsesOpaqueCompactor",
    "StreamingResponsesEndpoint",
    "X_CODEX_TURN_STATE_HEADER",
]
