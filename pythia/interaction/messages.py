from __future__ import annotations

import json
import math
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

from .context import ContextValidationError
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
from .timeouts import DEFAULT_REQUEST_TIMEOUT_SECONDS
from .usage import TokenUsage


ANTHROPIC_MESSAGES_API_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
MESSAGES_COMPACTION_BETA = "compact-2026-01-12"


@dataclass(frozen=True)
class MessagesPromptCaching:
    """Automatic prompt caching at the last cacheable block of each request."""

    ttl: Literal["5m", "1h"] = "5m"

    def __post_init__(self) -> None:
        if not isinstance(self.ttl, str):
            raise TypeError("prompt caching ttl must be a string")
        if self.ttl not in {"5m", "1h"}:
            raise ModelConfigurationError(
                "prompt caching ttl must be '5m' or '1h'"
            )

    def request_cache_control(self) -> Dict[str, str]:
        return {"type": "ephemeral", "ttl": self.ttl}


@dataclass(frozen=True)
class MessagesServerCompaction:
    trigger_input_tokens: Optional[int] = None
    pause_after_compaction: bool = False
    instructions: Optional[str] = None

    def __post_init__(self) -> None:
        trigger = self.trigger_input_tokens
        if trigger is not None and (
            isinstance(trigger, bool)
            or not isinstance(trigger, int)
            or trigger < 50_000
        ):
            raise ModelConfigurationError(
                "trigger_input_tokens must be an integer of at least 50000 "
                "or None"
            )
        if not isinstance(self.pause_after_compaction, bool):
            raise TypeError("pause_after_compaction must be a bool")
        if self.instructions is not None:
            if not isinstance(self.instructions, str):
                raise TypeError("instructions must be a string or None")
            instructions = self.instructions.strip()
            if not instructions:
                raise ModelConfigurationError(
                    "compaction instructions must not be empty"
                )
            object.__setattr__(self, "instructions", instructions)

    def request_edit(self) -> Dict[str, Any]:
        edit: Dict[str, Any] = {"type": "compact_20260112"}
        if self.trigger_input_tokens is not None:
            edit["trigger"] = {
                "type": "input_tokens",
                "value": self.trigger_input_tokens,
            }
        if self.pause_after_compaction:
            edit["pause_after_compaction"] = True
        if self.instructions is not None:
            edit["instructions"] = self.instructions
        return edit


@dataclass(frozen=True)
class MessagesEndpoint:
    api_url: str
    model: str
    api_key: Optional[str] = field(default=None, repr=False)
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION
    default_max_tokens: int = 4096
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    server_compaction: Optional[MessagesServerCompaction] = None
    prompt_caching: Optional[MessagesPromptCaching] = None

    def __post_init__(self) -> None:
        if not isinstance(self.api_url, str):
            raise TypeError("api_url must be a string")
        api_url = self.api_url.strip()
        if not api_url:
            raise ModelConfigurationError("api_url must not be empty")
        if any(character.isspace() for character in api_url):
            raise ModelConfigurationError("api_url must not contain whitespace")
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
        if path_prefix.endswith("/v1/messages"):
            raise ModelConfigurationError(
                "api_url must not include the fixed /v1/messages path"
            )
        object.__setattr__(
            self,
            "api_url",
            urllib.parse.urlunsplit(
                (scheme, parsed.netloc, path_prefix, "", "")
            ),
        )

        if not isinstance(self.model, str):
            raise TypeError("model must be a string")
        model = self.model.strip()
        if not model:
            raise ModelConfigurationError("model must not be empty")
        object.__setattr__(self, "model", model)

        if self.api_key is not None:
            if not isinstance(self.api_key, str):
                raise TypeError("api_key must be a string or None")
            api_key = self.api_key.strip()
            if not api_key:
                raise ModelConfigurationError("api_key must not be empty")
            if any(character.isspace() for character in api_key):
                raise ModelConfigurationError(
                    "api_key must not contain whitespace"
                )
            object.__setattr__(self, "api_key", api_key)

        if not isinstance(self.anthropic_version, str):
            raise TypeError("anthropic_version must be a string")
        anthropic_version = self.anthropic_version.strip()
        if not anthropic_version:
            raise ModelConfigurationError(
                "anthropic_version must not be empty"
            )
        if "\r" in anthropic_version or "\n" in anthropic_version:
            raise ModelConfigurationError(
                "anthropic_version must not contain newlines"
            )
        object.__setattr__(self, "anthropic_version", anthropic_version)

        if (
            isinstance(self.default_max_tokens, bool)
            or not isinstance(self.default_max_tokens, int)
            or self.default_max_tokens <= 0
        ):
            raise ModelConfigurationError(
                "default_max_tokens must be a positive integer"
            )

        timeout = self.request_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ModelConfigurationError(
                "request_timeout_seconds must be positive and finite"
            )
        object.__setattr__(self, "request_timeout_seconds", float(timeout))

        if self.server_compaction is not None and not isinstance(
            self.server_compaction,
            MessagesServerCompaction,
        ):
            raise TypeError(
                "server_compaction must be MessagesServerCompaction or None"
            )
        if self.prompt_caching is not None and not isinstance(
            self.prompt_caching,
            MessagesPromptCaching,
        ):
            raise TypeError(
                "prompt_caching must be MessagesPromptCaching or None"
            )

    @property
    def url(self) -> str:
        return f"{self.api_url}/v1/messages"


def _append_block(
    messages: List[Dict[str, Any]],
    pending: Optional[Dict[str, Any]],
    role: str,
    block: Dict[str, Any],
) -> Dict[str, Any]:
    if pending is None or pending["role"] != role:
        if pending is not None:
            messages.append(pending)
        pending = {"role": role, "content": []}
    pending["content"].append(block)
    return pending


def _tool_input(arguments_json: str, index: int) -> Dict[str, Any]:
    try:
        value = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise ModelConfigurationError(
            f"tool call at item {index} has invalid JSON arguments"
        ) from exc
    if not isinstance(value, dict):
        raise ModelConfigurationError(
            f"tool call at item {index} arguments must decode to an object"
        )
    return value


def _encode_context(
    items: Sequence[InteractionItem],
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    # Anthropic ignores content before the latest Messages compaction block.
    # Keep the append-only ModelContext intact while avoiding an ever-growing
    # outbound HTTP body.
    latest_compaction = -1
    for index, item in enumerate(items):
        if isinstance(item, OpaqueCompaction) and item.protocol == "messages":
            latest_compaction = index
    if latest_compaction >= 0:
        instruction_prefix: List[InteractionItem] = []
        for item in items[:latest_compaction]:
            if isinstance(item, Instructions):
                instruction_prefix.append(item)
                continue
            if isinstance(item, Message) and item.role in {
                "system",
                "developer",
            }:
                instruction_prefix.append(item)
                continue
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
            break
        items = (*instruction_prefix, *items[latest_compaction:])

    system: List[Dict[str, str]] = []
    messages: List[Dict[str, Any]] = []
    pending: Optional[Dict[str, Any]] = None
    conversation_started = False

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            messages.append(pending)
            pending = None

    # Last-wins Instructions: effective maps to system prompt. Empty text
    # preserved; absence means no entry. model_items() already collapses,
    # this is defensive for direct encoder calls.
    effective: Optional[Instructions] = None
    for item in items:
        if isinstance(item, Instructions):
            effective = item
    if effective is not None:
        system.append({"type": "text", "text": effective.text})

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
            flush()
            continue

        if isinstance(item, Instructions):
            continue

        if isinstance(item, Message):
            if item.role in {"system", "developer"}:
                if conversation_started:
                    raise ModelConfigurationError(
                        f"{item.role} message at item {index} appears after "
                        "Messages conversation content"
                    )
                system.append({"type": "text", "text": item.content})
                continue
            if item.role not in {"user", "assistant"}:
                raise ModelConfigurationError(
                    f"unsupported message role at item {index}: {item.role!r}"
                )
            conversation_started = True
            pending = _append_block(
                messages,
                pending,
                item.role,
                {"type": "text", "text": item.content},
            )
            continue

        if isinstance(item, Reasoning):
            conversation_started = True
            block: Dict[str, Any] = {
                "type": "thinking",
                "thinking": item.content or "\n".join(item.summary),
            }
            if item.content_signature is not None:
                block["signature"] = item.content_signature
            pending = _append_block(messages, pending, "assistant", block)
            continue

        if isinstance(item, ToolCall):
            conversation_started = True
            pending = _append_block(
                messages,
                pending,
                "assistant",
                {
                    "type": "tool_use",
                    "id": item.call_id,
                    "name": item.name,
                    "input": _tool_input(item.arguments_json, index),
                },
            )
            continue

        if isinstance(item, ToolResult):
            conversation_started = True
            pending = _append_block(
                messages,
                pending,
                "user",
                {
                    "type": "tool_result",
                    "tool_use_id": item.call_id,
                    "content": item.output,
                    "is_error": not item.success,
                },
            )
            continue

        if isinstance(item, OpaqueCompaction):
            if item.protocol != "messages":
                raise ModelConfigurationError(
                    "Messages cannot encode a Responses opaque compaction"
                )
            conversation_started = True
            pending = _append_block(
                messages,
                pending,
                "assistant",
                {"type": "compaction", "content": item.payload},
            )
            continue
        if isinstance(item, ContextCompaction):
            raise ModelConfigurationError(
                "ContextCompaction must be projected before request encoding"
            )
        raise ModelConfigurationError(
            f"unsupported interaction item at index {index}: {item!r}"
        )

    flush()
    if not messages:
        raise ModelConfigurationError("cannot sample an empty model context")
    return system, messages


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
                "name": name,
                "description": description,
                "input_schema": dict(parameters),
            }
        )
    return encoded


def _apply_sampling_options(
    payload: Dict[str, Any],
    options: Optional[SamplingOptions],
) -> None:
    if options is None:
        return
    if options.seed is not None:
        raise ModelConfigurationError(
            "Messages does not support the seed sampling option"
        )
    if options.max_tokens is not None:
        payload["max_tokens"] = options.max_tokens
    if options.temperature is not None:
        payload["temperature"] = options.temperature
    if options.top_p is not None:
        payload["top_p"] = options.top_p
    if options.stop:
        payload["stop_sequences"] = list(options.stop)


def _require_string(
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


def _decode_content(value: Any) -> Tuple[InteractionItem, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ModelResponseError("message.content must be a list")
    items: List[InteractionItem] = []
    for index, block in enumerate(value):
        if not isinstance(block, Mapping):
            raise ModelResponseError(
                f"message.content[{index}] must be an object"
            )
        block_type = block.get("type")
        if block_type == "text":
            items.append(
                Message(
                    role="assistant",
                    content=_require_string(
                        block.get("text"),
                        f"message.content[{index}].text",
                    ),
                )
            )
            continue
        if block_type == "thinking":
            signature_value = block.get("signature")
            content_signature = (
                None
                if signature_value is None
                else _require_string(
                    signature_value,
                    f"message.content[{index}].signature",
                    allow_empty=False,
                )
            )
            items.append(
                Reasoning(
                    content=_require_string(
                        block.get("thinking"),
                        f"message.content[{index}].thinking",
                    ),
                    content_signature=content_signature,
                )
            )
            continue
        if block_type == "compaction":
            items.append(
                OpaqueCompaction.from_messages(
                    _require_string(
                        block.get("content"),
                        f"message.content[{index}].content",
                        allow_empty=False,
                    )
                )
            )
            continue
        if block_type == "tool_use":
            tool_input = block.get("input", {})
            if not isinstance(tool_input, Mapping):
                raise ModelResponseError(
                    f"message.content[{index}].input must be an object"
                )
            try:
                arguments_json = json.dumps(
                    dict(tool_input),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError) as exc:
                raise ModelResponseError(
                    f"message.content[{index}].input is not JSON-compatible"
                ) from exc
            items.append(
                ToolCall(
                    name=_require_string(
                        block.get("name"),
                        f"message.content[{index}].name",
                        allow_empty=False,
                    ),
                    call_id=_require_string(
                        block.get("id"),
                        f"message.content[{index}].id",
                        allow_empty=False,
                    ),
                    arguments_json=arguments_json,
                )
            )
            continue
        raise ModelResponseError(
            f"unsupported message.content[{index}] type: {block_type!r}"
        )
    return tuple(items)


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _usage_counts(value: Mapping[str, Any]) -> Tuple[int, int, int]:
    # Anthropic input_tokens excludes cache writes and reads. The optional
    # cache_creation TTL breakdown is already included in its aggregate below.
    cache_creation_tokens = _nonnegative_int(
        value.get("cache_creation_input_tokens")
    )
    cache_read_tokens = _nonnegative_int(
        value.get("cache_read_input_tokens")
    )
    return (
        _nonnegative_int(value.get("input_tokens"))
        + cache_creation_tokens
        + cache_read_tokens,
        _nonnegative_int(value.get("output_tokens")),
        cache_read_tokens,
    )


def _decode_usage(value: Any) -> TokenUsage:
    if not isinstance(value, Mapping):
        return TokenUsage()
    input_tokens, output_tokens, cached_input_tokens = _usage_counts(value)
    iterations = value.get("iterations")
    if isinstance(iterations, Sequence) and not isinstance(
        iterations,
        (str, bytes, bytearray),
    ):
        iteration_input_tokens = 0
        iteration_output_tokens = 0
        iteration_cached_input_tokens = 0
        iteration_cache_creation_tokens = 0
        iteration_count = 0
        for iteration in iterations:
            if not isinstance(iteration, Mapping):
                continue
            iteration_count += 1
            iteration_input, iteration_output, iteration_cached = (
                _usage_counts(iteration)
            )
            iteration_input_tokens += iteration_input
            iteration_output_tokens += iteration_output
            iteration_cached_input_tokens += iteration_cached
            iteration_cache_creation_tokens += _nonnegative_int(
                iteration.get("cache_creation_input_tokens")
            )
        if iteration_count:
            # Some beta response versions report cache fields only at the
            # top level, outside the per-iteration breakdown.
            if (
                iteration_cached_input_tokens == 0
                and iteration_cache_creation_tokens == 0
            ):
                top_level_cache_creation = _nonnegative_int(
                    value.get("cache_creation_input_tokens")
                )
                iteration_input_tokens += (
                    cached_input_tokens + top_level_cache_creation
                )
            input_tokens = iteration_input_tokens
            output_tokens = iteration_output_tokens
            if iteration_cached_input_tokens:
                cached_input_tokens = iteration_cached_input_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_input_tokens=min(
            input_tokens,
            cached_input_tokens,
        ),
    )


def _decode_response(payload: Any) -> ModelSample:
    if not isinstance(payload, Mapping):
        raise ModelResponseError("Messages response must be an object")
    if payload.get("type") != "message":
        raise ModelResponseError(
            f"Messages response has unsupported type: {payload.get('type')!r}"
        )
    if payload.get("role") != "assistant":
        raise ModelResponseError(
            f"Messages response has unsupported role: {payload.get('role')!r}"
        )
    return ModelSample(
        items=_decode_content(payload.get("content")),
        stop_reason=(
            None
            if payload.get("stop_reason") is None
            else _require_string(payload.get("stop_reason"), "stop_reason")
        ),
        usage=_decode_usage(payload.get("usage")),
    )


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        payload = exc.read()
    except Exception:
        return ""
    if not isinstance(payload, (bytes, bytearray)):
        return ""
    return bytes(payload).decode("utf-8", errors="replace")[:4096]


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


class MessagesModel:
    def __init__(
        self,
        endpoint: MessagesEndpoint,
        *,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not isinstance(endpoint, MessagesEndpoint):
            raise TypeError("endpoint must be MessagesEndpoint")
        self.endpoint = endpoint
        self._opener = opener or urllib.request.urlopen

    def _build_request_payload(
        self,
        context: ModelContext,
        tools: Sequence[Any],
        options: Optional[SamplingOptions],
    ) -> Dict[str, Any]:
        if not isinstance(context, ModelContext):
            raise TypeError("context must be ModelContext")
        context.assert_model_ready()
        system, messages = _encode_context(context.model_items())
        payload: Dict[str, Any] = {
            "model": self.endpoint.model,
            "max_tokens": self.endpoint.default_max_tokens,
            "messages": messages,
            "stream": False,
        }
        if system:
            payload["system"] = system
        encoded_tools = _encode_tools(tools)
        if encoded_tools:
            payload["tools"] = encoded_tools
        if self.endpoint.prompt_caching is not None:
            # Let the API place and advance the breakpoint. In particular,
            # do not attach explicit cache controls to thinking/empty blocks.
            payload["cache_control"] = (
                self.endpoint.prompt_caching.request_cache_control()
            )
        if self.endpoint.server_compaction is not None:
            payload["context_management"] = {
                "edits": [
                    self.endpoint.server_compaction.request_edit()
                ]
            }
        _apply_sampling_options(payload, options)
        return payload

    def sample(
        self,
        context: ModelContext,
        *,
        tools: Sequence[Any] = (),
        options: Optional[SamplingOptions] = None,
    ) -> ModelSample:
        if options is not None and not isinstance(options, SamplingOptions):
            raise TypeError("options must be SamplingOptions or None")
        payload = self._build_request_payload(context, tools, options)
        try:
            request_data = json.dumps(payload, ensure_ascii=False).encode(
                "utf-8"
            )
        except (TypeError, ValueError) as exc:
            raise ModelConfigurationError(
                "Messages request is not JSON-serializable"
            ) from exc
        headers = {
            "Accept": "application/json",
            "Anthropic-Version": self.endpoint.anthropic_version,
            "Content-Type": "application/json",
            "User-Agent": "pythia-interaction/0.1",
        }
        if self.endpoint.server_compaction is not None:
            headers["Anthropic-Beta"] = MESSAGES_COMPACTION_BETA
        if self.endpoint.api_key is not None:
            headers["X-API-Key"] = self.endpoint.api_key
        request = urllib.request.Request(
            self.endpoint.url,
            data=request_data,
            headers=headers,
            method="POST",
        )
        try:
            response = self._opener(
                request,
                timeout=self.endpoint.request_timeout_seconds,
            )
        except urllib.error.HTTPError as exc:
            detail = _read_http_error_body(exc) or str(exc)
            if exc.code == 413 or _is_context_window_error(detail):
                raise ModelContextWindowError(detail) from exc
            raise ModelTransportError(
                f"Messages HTTP {exc.code}: {detail}"
            ) from exc
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
            raw = response.read()
        except (TimeoutError, socket.timeout) as exc:
            raise ModelTimeoutError(str(exc)) from exc
        except OSError as exc:
            raise ModelTransportError(str(exc)) from exc
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

        if not isinstance(raw, (bytes, bytearray)):
            raise ModelResponseError("HTTP response body must be bytes")
        if isinstance(status, int) and not 200 <= status < 300:
            detail = bytes(raw).decode("utf-8", errors="replace")[:4096]
            if status == 413 or _is_context_window_error(detail):
                raise ModelContextWindowError(detail)
            raise ModelTransportError(f"Messages HTTP {status}: {detail}")
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelResponseError("response is not UTF-8 JSON") from exc
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelResponseError("response is not valid JSON") from exc
        try:
            return _decode_response(decoded)
        except ContextValidationError as exc:
            raise ModelResponseError(str(exc)) from exc
        except ModelResponseError:
            raise
        except (TypeError, ValueError) as exc:
            raise ModelResponseError(str(exc)) from exc


__all__ = [
    "ANTHROPIC_MESSAGES_API_URL",
    "DEFAULT_ANTHROPIC_VERSION",
    "MESSAGES_COMPACTION_BETA",
    "MessagesEndpoint",
    "MessagesModel",
    "MessagesPromptCaching",
    "MessagesServerCompaction",
]
