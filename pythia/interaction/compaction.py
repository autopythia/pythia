"""Context compaction: the compactor protocol, its methods, and sequences.

``PiCompactor`` ports the coding agent's compaction from pi
(``packages/coding-agent/src/core/compaction/``). It keeps the most recent
context verbatim except signed thinking, summarizes the older span from a
plain-text transcript under a dedicated summarizer system prompt, updates the
previous summary iteratively, and summarizes the start of a split turn
separately. The remote Responses compactor lives with its adapter
(``responses.ResponsesOpaqueCompactor``). ``ColdstartCompactor`` makes no model
request: its checkpoint points to the saved log, which the model then reads
with its tools, and to a summary file next to it, which the model updates. It
first checks that the save holds the context.
``WarmPiCompactor`` (warmpi) makes pi's checkpoint from the conversation's own
request instead of pi's, so it reads the conversation's prompt cache.

A compaction sequence (``compaction_mode``, such as ``coldstart,warmpi:2``, the
default of every route without server-side compaction) lists methods with
their attempts; ``run_compaction`` tries them in order.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from functools import wraps
from pathlib import Path
from pathlib import PurePosixPath
from time import perf_counter
from typing import List
from typing import Optional
from typing import Protocol
from typing import TYPE_CHECKING
from typing import Tuple

from .context import ContextValidationError
from .context import InteractionContext
from .items import CompactionMetadata
from .items import ContextPrefix
from .items import Init
from .items import InteractionItem
from .items import Instructions
from .items import Message
from .items import ModelFailure
from .items import ModelSampleBoundary
from .items import OpaqueCompaction
from .items import Reasoning
from .items import SampleMetadata
from .items import TextPart
from .items import ToolCall
from .items import ToolResult
from .items import TurnSummary
from .items import UserInteractionBoundary
from .model import Model
from .model import ModelAuthenticationError
from .model import ModelCancelled
from .model import ModelConfigurationError
from .model import ModelContextWindowError
from .model import ModelError
from .model import ModelSample
from .model import ModelThinkingBindingError
from .model import retire_model
from .model import SampleParams
from .model import TokenUsage

if TYPE_CHECKING:
    from .display import DisplayItem
    from .environment import ToolSpec


# Compaction methods. ``pi`` summarizes on the host. ``warmpi`` makes pi's
# checkpoint from the conversation's own request, which reads its prompt
# cache. ``provider`` selects the provider's own compaction: Codex remote
# compaction, or Anthropic server-side compaction. ``coldstart`` makes no
# request: its checkpoint points to the log.
COMPACTION_METHODS = ("pi", "warmpi", "provider", "coldstart")
# What a compaction_mode entry may name: a method, or ``legacydefault``, the
# route's own method (``provider`` on the official Codex route and on a
# Messages endpoint with server-side compaction, ``pi`` elsewhere). It was the
# default before coldstart came first.
COMPACTION_MODES = ("legacydefault", *COMPACTION_METHODS)
# What an unset compaction_mode means, on every route but a Messages endpoint
# with server-side compaction: coldstart, then warmpi up to twice, for
# transient API failures. coldstart needs no retry: it fails only when its
# save can't be used, which another attempt wouldn't change. See
# runtime_config.CompactionRoute.default_sequence.
DEFAULT_COMPACTION_MODE = "coldstart,warmpi:2"
MAX_COMPACTION_ATTEMPTS = 9
_ATTEMPTS_RE = re.compile(r"[1-9]")
# Why a compaction runs: the threshold, overflow recovery, or /compact.
COMPACTION_REASONS = ("threshold", "overflow", "manual")
# Pi's ``keepRecentTokens``, ``TOOL_RESULT_MAX_CHARS``, and
# ``ESTIMATED_IMAGE_CHARS``.
DEFAULT_KEEP_RECENT_TOKENS = 20_000
TOOL_RESULT_MAX_CHARS = 2_000
ESTIMATED_IMAGE_CHARS = 4_800

# Pi's prompt texts, verbatim.
SUMMARIZATION_SYSTEM_PROMPT = """\
You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."""

SUMMARIZATION_PROMPT = """\
The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """\
The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

TURN_PREFIX_SUMMARIZATION_PROMPT = """\
The messages above are earlier context from an ongoing conversation. Later messages are stored separately and do not need to be reconstructed.

Create a concise checkpoint of the user's request and the progress shown above. This checkpoint will be placed before the later messages so the conversation can continue with the necessary context.

## Original Request
[What did the user ask for?]

## Progress So Far
- [Key decisions and work completed in these messages]

## Context Needed to Continue
- [Information from these messages needed to understand the later work]

Only summarize information explicitly present above. Do not infer or recreate later messages."""

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the "
    "following summary:\n\n<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

# The removed prompt summarizer's prefix. Its summaries remain in older saves,
# where they become the previous summary at the next compaction.
_LEGACY_SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a "
    "summary of its thinking process. You also have access to the state of "
    "the tools that were used by that language model. Use this to build on "
    "the work that has already been done and avoid duplicating work. Here is "
    "the summary produced by the other language model, use the information "
    "in this summary to assist with your own analysis:"
)

_SPLIT_TURN_SEPARATOR = "\n\n---\n\n**Turn Context (split turn):**\n\n"
_NO_PRIOR_HISTORY = "No prior history."
_HISTORY_PART = "the history"
_TURN_PREFIX_PART = "the earlier part of the current turn"
# A truncated, refused, or paused summary must never become a checkpoint.
_INCOMPLETE_STOP_REASONS = frozenset({
    "max_tokens",
    "length",
    "refusal",
    "pause_turn",
    "compaction",
    "model_context_window_exceeded",
})
_MODEL_VISIBLE_TYPES = (
    Instructions,
    Message,
    Reasoning,
    ToolCall,
    ToolResult,
    OpaqueCompaction,
)


class CompactionError(RuntimeError):
    pass


class NothingToCompact(CompactionError):
    """The context has nothing to summarize outside the kept recent tail.

    This is not a failure: automatic compaction skips it and samples.
    """


class CompactionContextWindowError(CompactionError):
    """A summary request exceeded the model's context window."""


class CompactionLogError(CompactionError):
    """coldstart's save is missing, can't be loaded, or doesn't hold the context.

    A sequence moves on to its next method at once: the same save would fail
    again.
    """


def parse_compaction_mode(value: object) -> Tuple[Tuple[str, int], ...]:
    """Parse a compaction sequence into ``((method, attempts), ...)``.

    The grammar is ``METHOD[:ATTEMPTS][,METHOD[:ATTEMPTS]]...``: a name from
    ``COMPACTION_MODES``, with 1 to 9 attempts, 1 when omitted. Spaces around
    ``,`` and ``:`` are ignored. Any method may follow ``coldstart``, which
    fails when its save can't be used (``CompactionLogError``).
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "compaction_mode requires a sequence of methods, such as 'pi' or "
            "'coldstart,pi:2'."
        )
    entries: List[Tuple[str, int]] = []
    for raw in value.split(","):
        method, colon, count = (part.strip() for part in raw.partition(":"))
        if method not in COMPACTION_MODES:
            raise ValueError(
                f"compaction_mode: unknown method {method!r}; use "
                f"{', '.join(COMPACTION_MODES)}."
            )
        if colon and _ATTEMPTS_RE.fullmatch(count) is None:
            raise ValueError(
                f"compaction_mode: the attempts for {method} must be 1 to "
                f"{MAX_COMPACTION_ATTEMPTS}."
            )
        entries.append((method, int(count) if colon else 1))
    return tuple(entries)


def format_compaction_mode(entries: Iterable[Tuple[str, int]]) -> str:
    """A sequence's normal form: no spaces, and no ``:1``."""
    return ",".join(
        method if attempts == 1 else f"{method}:{attempts}"
        for method, attempts in entries
    )


def normalize_compaction_mode(value: object) -> str:
    """Check a compaction sequence and return its normal form."""
    return format_compaction_mode(parse_compaction_mode(value))


def compaction_methods(value: object) -> Tuple[str, ...]:
    """The methods that a compaction sequence names, in order."""
    return tuple(method for method, _attempts in parse_compaction_mode(value))


def _require_keep_recent_tokens(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("keep_recent_tokens must be a nonnegative integer")
    return value


def _require_summary_budget(value: object) -> Optional[int]:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
    ):
        raise ValueError("max_output_tokens must be a positive integer or None")
    return value


@dataclass(frozen=True)
class CompactionSettings:
    """Compactor selection and pi options, from one config snapshot.

    ``mode`` is a compaction sequence (``parse_compaction_mode``), kept in its
    normal form. ``max_output_tokens=None`` gives each summary request the
    turn's budget.
    """

    mode: str = "pi"
    keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS
    max_output_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", normalize_compaction_mode(self.mode))
        _require_keep_recent_tokens(self.keep_recent_tokens)
        _require_summary_budget(self.max_output_tokens)

    @property
    def methods(self) -> Tuple[Tuple[str, int], ...]:
        """The sequence's entries: ``((method, attempts), ...)``."""
        return parse_compaction_mode(self.mode)


@dataclass(frozen=True)
class CompactionResult:
    items: Tuple[InteractionItem, ...]
    usage: TokenUsage = field(default_factory=TokenUsage)
    protocol: str = "unspecified"
    provider_session_id: Optional[str] = field(default=None, repr=False)
    provider_turn_id: Optional[str] = field(default=None, repr=False)
    provider_turn_state: Optional[str] = field(default=None, repr=False)
    provider_response_id: Optional[str] = field(default=None, repr=False)
    elapsed_seconds: Optional[float] = None
    request_attempts: int = 1
    recovery: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if len(items) != 1 or not isinstance(items[0], ContextPrefix):
            raise CompactionError(
                "compaction result must contain exactly one ContextPrefix"
            )
        try:
            InteractionContext(items)
        except ContextValidationError as exc:
            raise CompactionError(str(exc)) from exc
        metadata = self._metadata()
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "elapsed_seconds", metadata.elapsed_seconds)
        object.__setattr__(self, "recovery", metadata.recovery)

    def context_items(self) -> Tuple[InteractionItem, ...]:
        """Return the installed prefix followed by durable operation metadata."""
        return (*self.items, self._metadata())

    def _metadata(self) -> CompactionMetadata:
        return CompactionMetadata(
            usage=self.usage,
            protocol=self.protocol,
            provider_session_id=self.provider_session_id,
            provider_turn_id=self.provider_turn_id,
            provider_turn_state=self.provider_turn_state,
            provider_response_id=self.provider_response_id,
            elapsed_seconds=self.elapsed_seconds,
            request_attempts=self.request_attempts,
            recovery=self.recovery,
        )

    def display_items(self) -> Tuple["DisplayItem", ...]:
        from .display import render_interaction_items

        return render_interaction_items(self.context_items())


class Compactor(Protocol):
    """Build a replacement prefix for a model-ready context.

    ``sample_params`` are the turn's params; compactors derive their own
    request params from them. ``instructions`` is optional focus text from
    ``/compact [focus]``.
    """

    def compact(
        self,
        context: InteractionContext,
        *,
        tools: Sequence["ToolSpec"] = (),
        sample_params: Optional[SampleParams] = None,
        instructions: Optional[str] = None,
    ) -> CompactionResult:
        ...


def uses_host_auto_compaction(model: Model) -> bool:
    """Ownership is independent of numeric limits; custom models default to host."""
    return getattr(model, "auto_compaction_owner", "host") == "host"


def _estimate_text_tokens(chars: int) -> int:
    return (chars + 3) // 4


def _reasoning_text(item: Reasoning) -> str:
    return item.content or "\n".join(item.summary)


def _item_chars(item: object) -> int:
    if isinstance(item, Message):
        if isinstance(item.content, str):
            return len(item.content)
        return sum(
            len(part.text) if isinstance(part, TextPart)
            else ESTIMATED_IMAGE_CHARS
            for part in item.content
        )
    if isinstance(item, Reasoning):
        return len(_reasoning_text(item))
    if isinstance(item, ToolCall):
        return len(item.name) + len(item.arguments_json)
    if isinstance(item, ToolResult):
        return len(item.output)
    if isinstance(item, Instructions):
        return len(item.text)
    if isinstance(item, OpaqueCompaction):
        return len(item.payload)
    return 0


def estimate_item_tokens(item: InteractionItem) -> int:
    """Estimate one item as characters / 4, rounded up (pi's ``estimateTokens``).

    Media parts count as ``ESTIMATED_IMAGE_CHARS`` characters each. Boundaries,
    metadata, and user-tool records count as 0.
    """
    return _estimate_text_tokens(_item_chars(item))


def estimate_context_tokens(context: InteractionContext) -> int:
    """Estimate the effective context, anchored on provider-reported usage.

    This follows pi's ``estimateContextTokens``. The anchor is the latest
    top-level ``SampleMetadata`` after the latest ``ContextPrefix`` that
    reports usage; the estimate is its ``total_tokens`` plus the estimates of
    the raw items after it. Without an anchor, for example just after a
    compaction or on a server that reports no usage, it estimates all of
    ``context.model_items()``. Character estimates undercount contexts whose
    reasoning is stored only as signatures, hence the anchor.
    """
    if not isinstance(context, InteractionContext):
        raise TypeError("context must be InteractionContext")
    items = context.items
    start = 0
    for index in range(len(items) - 1, -1, -1):
        if isinstance(items[index], ContextPrefix):
            start = index + 1
            break
    for index in range(len(items) - 1, start - 1, -1):
        item = items[index]
        if isinstance(item, SampleMetadata) and item.usage.total_tokens > 0:
            return item.usage.total_tokens + sum(
                estimate_item_tokens(later) for later in items[index + 1:]
            )
    return sum(estimate_item_tokens(item) for item in context.model_items())


def should_auto_compact(
    context: InteractionContext,
    threshold_tokens: int,
) -> bool:
    """Return whether the estimated context reached a threshold.

    A context with no model-visible item after its latest ``ContextPrefix``
    was just compacted, and is not compacted again.
    """
    if not isinstance(context, InteractionContext):
        raise TypeError("context must be InteractionContext")
    if (
        isinstance(threshold_tokens, bool)
        or not isinstance(threshold_tokens, int)
        or threshold_tokens <= 0
    ):
        raise ValueError("threshold_tokens must be a positive integer")
    for item in reversed(context.items):
        if isinstance(item, ContextPrefix):
            return False
        if isinstance(item, _MODEL_VISIBLE_TYPES):
            break
    else:
        return False
    return estimate_context_tokens(context) >= threshold_tokens


def auto_compaction_due(model: Model, context: InteractionContext, snapshot) -> bool:
    """Return whether a frontend should compact before its next sample.

    ``snapshot`` is the turn's ``InteractionConfigSnapshot``. A failed
    automatic compaction fails the turn; ``NothingToCompact`` is not a
    failure.
    """
    # TODO: Pi reports a failed threshold compaction and keeps sampling
    # (``_runAutoCompaction`` in pi's ``agent-session.ts``); here the failure
    # fails the turn.
    threshold = snapshot.auto_compaction_context_tokens
    return bool(
        snapshot.enable_auto_compaction
        and uses_host_auto_compaction(model)
        and threshold is not None
        and should_auto_compact(context, threshold)
    )


def _timed_compact(
    method: Callable[..., CompactionResult],
) -> Callable[..., CompactionResult]:
    """Measure a complete successful compactor call inside its worker."""
    @wraps(method)
    def measured(*args, **kwargs) -> CompactionResult:
        started = perf_counter()
        result = method(*args, **kwargs)
        if not isinstance(result, CompactionResult):
            raise TypeError(
                "compactor must return CompactionResult, got "
                f"{type(result).__name__}"
            )
        return replace(
            result,
            elapsed_seconds=perf_counter() - started,
        )

    return measured


def _model_method(model: Model, method: str) -> str:
    """Resolve ``legacydefault`` for a model: remote compaction where it has it."""
    if method != "legacydefault":
        return method
    remote = getattr(model, "supports_remote_compaction", False) is True
    return "provider" if remote else "pi"


def create_default_compactor(
    model: Model,
    settings: CompactionSettings = CompactionSettings(),
    *,
    log: Optional["CompactionLog"] = None,
    reason: str = "threshold",
) -> Compactor:
    """Build the compactor for the first method of ``settings.mode``.

    ``run_compaction`` calls it once for each method, with that method alone
    as ``settings.mode``. ``provider`` uses remote opaque compaction when the
    model advertises it: the built-in ChatGPT/Codex Responses route.
    Otherwise ``provider`` uses ``PiCompactor``, as manual compaction does for
    Messages in ``provider`` mode, since Anthropic has no on-demand server
    compaction. ``coldstart`` needs the save's ``log``, and ``warmpi`` takes
    the compaction's ``reason`` (``COMPACTION_REASONS``): in overflow recovery
    it sends pi's request. No provider is probed.
    """
    if not isinstance(settings, CompactionSettings):
        raise TypeError("settings must be CompactionSettings")
    if reason not in COMPACTION_REASONS:
        raise ValueError(f"reason must be one of {COMPACTION_REASONS}")
    method = _model_method(model, settings.methods[0][0])
    if method == "coldstart":
        if log is None:
            raise CompactionError("coldstart compaction needs a save to point to")
        return ColdstartCompactor(log)
    if method == "warmpi":
        return WarmPiCompactor(
            model,
            keep_recent_tokens=settings.keep_recent_tokens,
            max_output_tokens=settings.max_output_tokens,
            reason=reason,
        )
    if (
        method == "provider"
        and getattr(model, "supports_remote_compaction", False) is True
    ):
        # ``responses`` imports this module for its concrete compactor.
        from .responses import ResponsesOpaqueCompactor

        return ResponsesOpaqueCompactor(model)
    return PiCompactor(
        model,
        keep_recent_tokens=settings.keep_recent_tokens,
        max_output_tokens=settings.max_output_tokens,
    )


def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return "(tokens truncated)"
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()} ...(tokens truncated)"


def _leading_instruction_prefix(
    items: Sequence[InteractionItem],
) -> Tuple[InteractionItem, ...]:
    return _split_instruction_prefix(items)[0]


def _split_instruction_prefix(
    items: Sequence[InteractionItem],
) -> Tuple[Tuple[InteractionItem, ...], Tuple[InteractionItem, ...]]:
    """Split the effective instructions and leading system/developer messages."""
    effective = None
    for item in items:
        if isinstance(item, Instructions):
            effective = item
    prefix: List[InteractionItem] = []
    if effective is not None:
        prefix.append(effective)
    rest: List[InteractionItem] = []
    leading = True
    for item in items:
        if isinstance(item, Instructions):
            continue
        if (
            leading
            and isinstance(item, Message)
            and item.role in {"system", "developer"}
        ):
            prefix.append(item)
            continue
        leading = False
        rest.append(item)
    return tuple(prefix), tuple(rest)


def _select_retained_user_messages(
    messages: Sequence[Message],
    max_tokens: int,
) -> Tuple[Message, ...]:
    selected_reversed = []
    remaining = max(0, max_tokens)
    for message in reversed(messages):
        if remaining == 0:
            break
        tokens = _approx_token_count(message.content_text)
        if tokens <= remaining:
            selected_reversed.append(message)
            remaining -= tokens
            continue
        # Truncation drops any non-text parts: a partial image/file is not
        # meaningful and would still consume provider budget.
        selected_reversed.append(
            Message(
                role="user",
                content=_truncate_text_to_tokens(message.content_text, remaining),
            )
        )
        break
    selected_reversed.reverse()
    return tuple(selected_reversed)


def _compaction_summary_text(item: object) -> Optional[str]:
    """Return a summary message's text, for new-style and old-style summaries."""
    if not isinstance(item, Message) or item.role != "user":
        return None
    text = item.content
    if not isinstance(text, str):
        return None
    if (
        len(text) >= len(COMPACTION_SUMMARY_PREFIX) + len(COMPACTION_SUMMARY_SUFFIX)
        and text.startswith(COMPACTION_SUMMARY_PREFIX)
        and text.endswith(COMPACTION_SUMMARY_SUFFIX)
    ):
        return text[len(COMPACTION_SUMMARY_PREFIX):len(text) - len(COMPACTION_SUMMARY_SUFFIX)]
    legacy_prefix = f"{_LEGACY_SUMMARY_PREFIX}\n"
    if text.startswith(legacy_prefix):
        return text[len(legacy_prefix):]
    return None


def is_compaction_summary(message: object) -> bool:
    """Whether a message is a compaction summary, in either style.

    New-style summaries are user messages wrapped in
    ``COMPACTION_SUMMARY_PREFIX`` and ``COMPACTION_SUMMARY_SUFFIX``. Old-style
    summaries, from the removed prompt summarizer, remain in older saves.
    """
    return _compaction_summary_text(message) is not None


def _is_user_message(item: object) -> bool:
    return isinstance(item, Message) and item.role == "user"


def _is_output_item(item: object) -> bool:
    """A sample's output: reasoning, assistant text, a tool call, or a checkpoint."""
    if isinstance(item, (Reasoning, ToolCall, OpaqueCompaction)):
        return True
    return isinstance(item, Message) and item.role == "assistant"


def _is_signed_thinking(item: object) -> bool:
    """Messages thinking, which is valid only after its original history."""
    return isinstance(item, Reasoning) and item.content_signature is not None


def _has_content(items: Iterable[InteractionItem]) -> bool:
    return any(isinstance(item, _MODEL_VISIBLE_TYPES) for item in items)


def _message_text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        part.text if isinstance(part, TextPart) else "[image]"
        for part in message.content
    )


def _format_tool_call(call: ToolCall) -> str:
    try:
        arguments = json.loads(call.arguments_json)
    except (ValueError, RecursionError):
        arguments = None
    if isinstance(arguments, dict):
        rendered = ", ".join(
            f"{key}="
            + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            for key, value in arguments.items()
        )
    else:
        rendered = call.arguments_json
    return f"{call.name}({rendered})"


def _truncate_for_summary(text: str) -> str:
    if len(text) <= TOOL_RESULT_MAX_CHARS:
        return text
    omitted = len(text) - TOOL_RESULT_MAX_CHARS
    return (
        f"{text[:TOOL_RESULT_MAX_CHARS]}\n\n"
        f"[... {omitted} more characters truncated]"
    )


_ROLE_LABELS = {"user": "User", "system": "System", "developer": "Developer"}


def _serialize_transcript(items: Sequence[InteractionItem]) -> str:
    """Render items as text, following pi's ``serializeConversation``.

    Each sample's output is grouped as thinking, then text, then tool calls,
    so the summarizer reads a transcript rather than a conversation to
    continue. Tool results are truncated; boundaries and metadata are skipped.
    """
    entries: List[str] = []
    thinking: List[str] = []
    texts: List[str] = []
    calls: List[str] = []

    def flush_sample() -> None:
        if thinking:
            entries.append("[Assistant thinking]: " + "\n".join(thinking))
        if texts:
            entries.append("[Assistant]: " + "\n".join(texts))
        if calls:
            entries.append("[Assistant tool calls]: " + "; ".join(calls))
        thinking.clear()
        texts.clear()
        calls.clear()

    for item in items:
        if isinstance(item, Reasoning):
            text = _reasoning_text(item)
            if text.strip():
                thinking.append(text)
        elif isinstance(item, Message) and item.role == "assistant":
            if item.content_text.strip():
                texts.append(item.content_text)
        elif isinstance(item, ToolCall):
            calls.append(_format_tool_call(item))
        elif isinstance(item, OpaqueCompaction):
            # A Messages compaction block is readable summary text. Responses
            # checkpoints are encrypted and never reach the transcript.
            if item.protocol == "messages" and item.payload.strip():
                texts.append(item.payload)
        elif isinstance(item, Message):
            flush_sample()
            text = _message_text(item)
            if text.strip():
                label = _ROLE_LABELS.get(item.role, item.role.strip().title())
                entries.append(f"[{label}]: {text}")
        elif isinstance(item, ToolResult):
            flush_sample()
            if item.output:
                entries.append(
                    f"[Tool result]: {_truncate_for_summary(item.output)}"
                )
        elif isinstance(item, (ModelSampleBoundary, UserInteractionBoundary)):
            flush_sample()
    flush_sample()
    return "\n\n".join(entries)


def _focus_suffix(focus: Optional[str]) -> str:
    return "" if focus is None else f"\n\nAdditional focus: {focus}"


def _history_prompt(
    history: Sequence[InteractionItem],
    previous_summary: Optional[str],
    focus: Optional[str],
) -> str:
    prompt = f"<conversation>\n{_serialize_transcript(history)}\n</conversation>\n\n"
    if previous_summary is None:
        instructions = SUMMARIZATION_PROMPT
    else:
        prompt += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        instructions = UPDATE_SUMMARIZATION_PROMPT
    return prompt + instructions + _focus_suffix(focus)


def _turn_prefix_prompt(
    turn_prefix: Sequence[InteractionItem],
    focus: Optional[str],
) -> str:
    return (
        f"# Conversation\n{_serialize_transcript(turn_prefix)}\n\n"
        f"# Instructions\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
        + _focus_suffix(focus)
    )


@dataclass(frozen=True)
class _CompactionPlan:
    instruction_prefix: Tuple[InteractionItem, ...]
    checkpoints: Tuple[InteractionItem, ...]
    previous_summary: Optional[str]
    history: Tuple[InteractionItem, ...]
    # Non-empty only when the cut splits a turn.
    turn_prefix: Tuple[InteractionItem, ...]
    tail: Tuple[InteractionItem, ...]


def _extract_summary_sources(
    rest: Tuple[InteractionItem, ...],
) -> Tuple[Optional[str], Tuple[InteractionItem, ...], Tuple[InteractionItem, ...]]:
    """Return the previous summary, carried checkpoints, and the span."""
    for index in range(len(rest) - 1, -1, -1):
        item = rest[index]
        if isinstance(item, OpaqueCompaction) and item.protocol == "messages":
            # Anthropic ignores everything before its latest compaction block
            # except the instructions, and so does the summary.
            rest = rest[index:]
            break
    source = None
    previous_summary = None
    for index, item in enumerate(rest):
        if isinstance(item, OpaqueCompaction) and item.protocol == "messages":
            source, previous_summary = index, item.payload
            continue
        text = _compaction_summary_text(item)
        if text is not None:
            source, previous_summary = index, text
    checkpoints: List[InteractionItem] = []
    span: List[InteractionItem] = []
    for index, item in enumerate(rest):
        if index == source:
            continue
        if isinstance(item, OpaqueCompaction) and item.protocol == "responses":
            # Encrypted: carried into the new prefix verbatim.
            checkpoints.append(item)
            continue
        span.append(item)
    if previous_summary is not None and not previous_summary.strip():
        previous_summary = None
    return previous_summary, tuple(checkpoints), tuple(span)


def _cut_points(span: Sequence[InteractionItem]) -> List[int]:
    """Indices where the kept tail may start (pi's ``findValidCutPoints``).

    These are the first of each run of consecutive user messages and the
    first output item of each sample. Tool results are never cut points.
    """
    points: List[int] = []
    after_user = False
    in_sample = False
    for index, item in enumerate(span):
        if isinstance(item, ModelSampleBoundary):
            in_sample = False
            continue
        if not isinstance(item, _MODEL_VISIBLE_TYPES):
            continue
        if _is_user_message(item):
            if not after_user:
                points.append(index)
            after_user = True
            in_sample = False
            continue
        after_user = False
        if _is_output_item(item):
            if not in_sample:
                points.append(index)
            in_sample = True
        else:
            in_sample = False
    return points


def _find_cut(
    span: Sequence[InteractionItem],
    points: Sequence[int],
    keep_recent_tokens: int,
) -> int:
    """Follow pi's ``findProjectedCutPoint``, except that 0 keeps nothing."""
    if keep_recent_tokens == 0:
        return len(span)
    total = 0
    for index in range(len(span) - 1, -1, -1):
        tokens = estimate_item_tokens(span[index])
        if not tokens:
            continue
        total += tokens
        if total >= keep_recent_tokens:
            for point in points:
                if point >= index:
                    return point
            # Trailing tool results alone exceed the budget: keep their
            # sample's first output item rather than cutting between them.
            if points:
                return points[-1]
            raise NothingToCompact("nothing precedes the recent tail")
    raise NothingToCompact("the context fits in compaction_pi_keep_recent_tokens")


def _plan_compaction(
    items: Sequence[InteractionItem],
    keep_recent_tokens: int,
) -> _CompactionPlan:
    instruction_prefix, rest = _split_instruction_prefix(items)
    previous_summary, checkpoints, span = _extract_summary_sources(rest)
    if not _has_content(span):
        raise NothingToCompact("there is nothing new to summarize")
    points = _cut_points(span)
    cut = _find_cut(span, points, keep_recent_tokens)
    turn_start = None
    if cut < len(span) and _is_output_item(span[cut]):
        # The cut splits a turn when a user message starts it earlier in the
        # span: that run of user messages starts the turn prefix.
        for point in reversed(points):
            if point < cut and _is_user_message(span[point]):
                turn_start = point
                break
    if turn_start is None:
        history, turn_prefix = span[:cut], ()
    else:
        history, turn_prefix = span[:turn_start], span[turn_start:cut]
    if not _has_content((*history, *turn_prefix)):
        raise NothingToCompact("nothing precedes the recent tail")
    return _CompactionPlan(
        instruction_prefix=instruction_prefix,
        checkpoints=checkpoints,
        previous_summary=previous_summary,
        history=tuple(history),
        turn_prefix=tuple(turn_prefix),
        tail=tuple(span[cut:]),
    )


def _context_window_message(
    part: str,
    items: Sequence[InteractionItem],
    prompt: str,
) -> str:
    count = sum(1 for item in items if isinstance(item, _MODEL_VISIBLE_TYPES))
    noun = "item" if count == 1 else "items"
    tokens = _estimate_text_tokens(len(SUMMARIZATION_SYSTEM_PROMPT) + len(prompt))
    return (
        f"summary request for {part} ({count} {noun}, ~{tokens:,} estimated "
        "tokens) exceeded the model's context window; pi compaction sends "
        "each part in one request and does not split it. Lower "
        "compaction_pi_max_output_tokens, or raise compaction_pi_keep_recent_tokens "
        "to summarize less."
    )


def _summary_text(sample: ModelSample, part: str) -> str:
    """Check a summary response, as pi's ``getSummarizationFailure`` does."""
    if sample.tool_calls:
        raise CompactionError(f"summary request for {part} returned tool calls")
    if sample.stop_reason in _INCOMPLETE_STOP_REASONS:
        raise CompactionError(
            f"summary request for {part} stopped early "
            f"(stop_reason {sample.stop_reason}); an incomplete summary is "
            "never installed"
        )
    # All text blocks, as pi's ``contentText``, not only the last one.
    text = "\n".join(
        message.content_text for message in sample.assistant_messages
    ).strip()
    if not text:
        raise CompactionError(f"summary request for {part} returned no text")
    return text


def _sum_usage(usages: Iterable[TokenUsage]) -> TokenUsage:
    input_tokens = output_tokens = total_tokens = cached_input_tokens = 0
    for usage in usages:
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        total_tokens += usage.total_tokens
        cached_input_tokens += usage.cached_input_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def _pi_checkpoint(plan: _CompactionPlan, summary: str) -> ContextPrefix:
    """pi's checkpoint: the instructions, carried checkpoints, the summary, the tail.

    pi and warmpi build the same one; only their summaries come from
    different requests.
    """
    # Boundaries stay: the Messages and Chat Completions encoders flush
    # assistant blocks at them. The other items are log-only, and
    # ModelFailure is not allowed in prefixes.
    # Signed thinking leaves too. Anthropic binds each thinking block to
    # the conversation that preceded it, which the summary replaces, so
    # the next request would fail with HTTP 400 ("Invalid `signature` in
    # `thinking` block. The block is bound to a different conversation.").
    # Responses reasoning (``encrypted_content``) stays.
    # TODO: Preserve thinking across compaction if Anthropic adds a way;
    # the binding has no keep option today. Its
    # ``thinking-binding-controls-2026-08-01`` beta header, together with
    # ``thinking.block_binding.prefix_mismatch_behavior: "drop_block"``
    # (default ``"error"``), makes the server drop unbound blocks instead.
    # That would also unstick saves whose prefix already kept them, but it
    # was verified only on claude-opus-5-5 with adaptive thinking, and it
    # would silently drop thinking after any other history bug too.
    tail = tuple(
        item
        for item in plan.tail
        if not isinstance(item, (SampleMetadata, TurnSummary, ModelFailure))
        and not _is_signed_thinking(item)
    )
    return ContextPrefix(
        prefix_items=(
            *plan.instruction_prefix,
            *plan.checkpoints,
            Message(
                role="user",
                content=f"{COMPACTION_SUMMARY_PREFIX}{summary}{COMPACTION_SUMMARY_SUFFIX}",
            ),
            *tail,
        ),
    )


class PiCompactor:
    """Pi's compaction: summarize older context and keep a recent tail.

    The new prefix is the instructions, any Responses checkpoints, the summary
    as a user message, and the kept tail verbatim, minus signed thinking.
    ``keep_recent_tokens`` is the estimated size of that tail; 0 keeps nothing
    (pi still keeps the last unit). ``max_output_tokens`` overrides the turn's
    budget for summary requests. ``tools`` are accepted for the protocol but
    never sent: the requests carry the transcript as text.
    """

    def __init__(
        self,
        model: Model,
        *,
        keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS,
        max_output_tokens: Optional[int] = None,
    ) -> None:
        if not hasattr(model, "sample") or not callable(model.sample):
            raise TypeError("model must provide sample(...)")
        self._model = model
        self.keep_recent_tokens = _require_keep_recent_tokens(keep_recent_tokens)
        self.max_output_tokens = _require_summary_budget(max_output_tokens)

    @_timed_compact
    def compact(
        self,
        context: InteractionContext,
        *,
        tools: Sequence["ToolSpec"] = (),
        sample_params: Optional[SampleParams] = None,
        instructions: Optional[str] = None,
    ) -> CompactionResult:
        del tools
        if not isinstance(context, InteractionContext):
            raise TypeError("context must be InteractionContext")
        if sample_params is not None and not isinstance(sample_params, SampleParams):
            raise TypeError("sample_params must be SampleParams or None")
        if instructions is not None and not isinstance(instructions, str):
            raise TypeError("instructions must be a string or None")
        focus = None if instructions is None else instructions.strip() or None
        try:
            context.assert_model_ready()
        except ContextValidationError as exc:
            raise CompactionError(str(exc)) from exc

        plan = _plan_compaction(context.model_items(), self.keep_recent_tokens)
        params = self._summary_params(sample_params)
        samples: List[ModelSample] = []
        has_history = _has_content(plan.history)
        # TODO: Run the history and turn-prefix requests in parallel. They are
        # independent and merge in a fixed order, so wall time would be the
        # longer request rather than the sum (plan section 9.2). Build both
        # request contexts first (InteractionContext is not thread-safe), run
        # each on a two-worker ThreadPoolExecutor in its own
        # contextvars.copy_context() (--debug-trace tags need it), wait for
        # both, raise the first failure in fixed order, and combine accounting
        # in that order.
        history_text = plan.previous_summary or _NO_PRIOR_HISTORY
        if has_history:
            history_text, sample = self._summarize(
                context,
                _history_prompt(plan.history, plan.previous_summary, focus),
                params,
                part=_HISTORY_PART,
                items=plan.history,
            )
            samples.append(sample)
        summary = history_text
        if plan.turn_prefix:
            # Without a history request, focus text goes here rather than
            # being dropped (pi drops it).
            turn_text, sample = self._summarize(
                context,
                _turn_prefix_prompt(
                    plan.turn_prefix,
                    None if has_history else focus,
                ),
                params,
                part=_TURN_PREFIX_PART,
                items=plan.turn_prefix,
            )
            samples.append(sample)
            summary = f"{history_text}{_SPLIT_TURN_SEPARATOR}{turn_text}"

        checkpoint = _pi_checkpoint(plan, summary)
        last = samples[-1]
        return CompactionResult(
            items=(checkpoint,),
            usage=_sum_usage(sample.usage for sample in samples),
            protocol="pi",
            provider_session_id=last.provider_session_id,
            provider_turn_id=last.provider_turn_id,
            provider_turn_state=last.provider_turn_state,
            request_attempts=sum(sample.request_attempts for sample in samples),
            recovery=tuple(
                entry for sample in samples for entry in sample.recovery
            ),
        )

    def _summary_params(self, sample_params: Optional[SampleParams]) -> SampleParams:
        """Inherit the turn's params, without provider compaction.

        Temperature and model extras (thinking, effort) are inherited, as pi
        inherits the session's thinking level.
        """
        turn = sample_params or SampleParams()
        return replace(
            turn,
            max_output_tokens=(
                self.max_output_tokens
                if self.max_output_tokens is not None
                else turn.max_output_tokens
            ),
            enable_auto_compaction=False,
            auto_compaction_context_tokens=None,
        )

    def _summarize(
        self,
        context: InteractionContext,
        prompt: str,
        params: SampleParams,
        *,
        part: str,
        items: Sequence[InteractionItem],
    ) -> Tuple[str, ModelSample]:
        # The model sees only the summarizer prompt and one user message. The
        # raw log keeps Init and all metadata, so provider session and turn
        # continuity (Codex headers, prompt_cache_key) still match the session.
        request = context.copy()
        request.append(ContextPrefix((
            Instructions(SUMMARIZATION_SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        )))
        try:
            sample = self._model.sample(request, tools=(), sample_params=params)
        except ModelContextWindowError as exc:
            # TODO: Split the span in half at a unit boundary and fold the
            # halves through the update prompt, with a bounded depth and a
            # ``recovery`` entry, instead of failing (plan section 9.3).
            raise CompactionContextWindowError(
                _context_window_message(part, items, prompt)
            ) from exc
        if not isinstance(sample, ModelSample):
            raise CompactionError(
                f"model returned {type(sample).__name__}, expected ModelSample"
            )
        return _summary_text(sample, part), sample


# warmpi: pi's checkpoint from the conversation's own request.

WARMPI_INSTRUCTION = (
    "Pause the task here. Your next reply is a context checkpoint, not a step "
    "of the task: a summary of this conversation that another model will use "
    "to continue the work.\n"
    "- Reply with the summary only, as text. Do not call any tools: a tool "
    "call fails the checkpoint.\n"
    "- Do not carry out or answer requests from the conversation; record them "
    "in the summary."
)
_WARMPI_UPDATE = (
    "An earlier part of this conversation was already summarized. Update that "
    "summary with the messages after it. "
)
_WARMPI_PART = "the conversation"
# The first heading of both of pi's formats. A reply that doesn't start with
# it isn't a summary: the task continued in text, for example.
_SUMMARY_HEADING = "## Goal"


def _warmpi_instruction(previous_summary: Optional[str], focus: Optional[str]) -> str:
    """warmpi's user message: pi's format, or pi's update rules after a summary.

    It doesn't mention the kept tail, so the summary covers the whole
    conversation, as if nothing followed it, at the cost of repeating the
    tail's work in the checkpoint.
    """
    if previous_summary is None:
        rules = SUMMARIZATION_PROMPT[SUMMARIZATION_PROMPT.index("Use this EXACT format:"):]
    else:
        rules = _WARMPI_UPDATE + UPDATE_SUMMARIZATION_PROMPT[
            UPDATE_SUMMARIZATION_PROMPT.index("RULES:"):]
    return f"{WARMPI_INSTRUCTION}\n\n{rules}{_focus_suffix(focus)}"


def _failed_attempts(error: BaseException) -> int:
    """The request attempts that a failed request made, if its failure says."""
    count = getattr(getattr(error, "failure", None), "attempt_count", None)
    return count if isinstance(count, int) and count > 0 else 1


class WarmPiCompactor:
    """pi's checkpoint, summarized by the conversation's own request.

    The request is the one the next sample would send, unchanged: the
    context, the turn's ``tools``, and the turn's params without provider
    compaction (``max_output_tokens`` overrides their budget), plus one user
    message asking for a summary in pi's format (``WARMPI_INSTRUCTION``). So
    it reads the conversation's prompt cache; it never checks the cache's age.
    The message doesn't mention the kept tail: the summary covers the whole
    conversation. The checkpoint is pi's (``_pi_checkpoint``), with this
    summary and protocol ``warmpi``. A reply with tool calls, without text,
    stopped early, or not starting with the format's first heading fails the
    attempt.

    It sends pi's request instead (``PiCompactor``; protocol ``pi``):
    - in overflow recovery (``reason="overflow"``): the conversation didn't
      fit, so the conversation and an instruction won't either;
    - within the same attempt, when its own request is too large for the
      window (``recovery`` ``warmpi_too_large``), or when the provider rejects
      its signed thinking (``ModelThinkingBindingError``;
      ``warmpi_thinking_rejected``). pi's transcript cuts tool results, and
      has no signed blocks.
    """

    def __init__(
        self,
        model: Model,
        *,
        keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS,
        max_output_tokens: Optional[int] = None,
        reason: str = "threshold",
    ) -> None:
        if reason not in COMPACTION_REASONS:
            raise ValueError(f"reason must be one of {COMPACTION_REASONS}")
        self._pi = PiCompactor(
            model,
            keep_recent_tokens=keep_recent_tokens,
            max_output_tokens=max_output_tokens,
        )
        self._model = model
        self.keep_recent_tokens = self._pi.keep_recent_tokens
        self.max_output_tokens = self._pi.max_output_tokens
        self.reason = reason

    @_timed_compact
    def compact(
        self,
        context: InteractionContext,
        *,
        tools: Sequence["ToolSpec"] = (),
        sample_params: Optional[SampleParams] = None,
        instructions: Optional[str] = None,
    ) -> CompactionResult:
        if not isinstance(context, InteractionContext):
            raise TypeError("context must be InteractionContext")
        if sample_params is not None and not isinstance(sample_params, SampleParams):
            raise TypeError("sample_params must be SampleParams or None")
        if instructions is not None and not isinstance(instructions, str):
            raise TypeError("instructions must be a string or None")
        pi = {"sample_params": sample_params, "instructions": instructions}
        if self.reason == "overflow":
            return self._pi.compact(context, **pi)
        focus = None if instructions is None else instructions.strip() or None
        try:
            context.assert_model_ready()
        except ContextValidationError as exc:
            raise CompactionError(str(exc)) from exc
        plan = _plan_compaction(context.model_items(), self.keep_recent_tokens)
        request = context.copy()
        request.append(Message(
            role="user",
            content=_warmpi_instruction(plan.previous_summary, focus),
        ))
        try:
            sample = self._model.sample(
                request,
                tools=tuple(tools),
                sample_params=self._pi._summary_params(sample_params),
            )
        except ModelContextWindowError as exc:
            return self._pi_instead(context, pi, "warmpi_too_large", _failed_attempts(exc))
        except ModelThinkingBindingError as exc:
            return self._pi_instead(
                context, pi, "warmpi_thinking_rejected", _failed_attempts(exc))
        if not isinstance(sample, ModelSample):
            raise CompactionError(
                f"model returned {type(sample).__name__}, expected ModelSample"
            )
        if sample.stop_reason == "model_context_window_exceeded":
            return self._pi_instead(
                context, pi, "warmpi_too_large", sample.request_attempts, sample.usage)
        summary = _summary_text(sample, _WARMPI_PART)
        if not summary.startswith(_SUMMARY_HEADING):
            raise CompactionError(
                f"summary request for {_WARMPI_PART} returned text that doesn't start "
                f'with "{_SUMMARY_HEADING}", the first heading of its format; an '
                "answer in another form is never installed"
            )
        return CompactionResult(
            items=(_pi_checkpoint(plan, summary),),
            usage=sample.usage,
            protocol="warmpi",
            provider_session_id=sample.provider_session_id,
            provider_turn_id=sample.provider_turn_id,
            provider_turn_state=sample.provider_turn_state,
            request_attempts=sample.request_attempts,
            recovery=sample.recovery,
        )

    def _pi_instead(
        self,
        context: InteractionContext,
        pi: dict,
        label: str,
        attempts: int,
        usage: Optional[TokenUsage] = None,
    ) -> CompactionResult:
        """pi's request, within this attempt, after warmpi's own request.

        ``attempts`` and ``usage`` are those of warmpi's request; ``label`` is
        why it sent pi's, the first ``recovery`` entry.
        """
        result = self._pi.compact(context, **pi)
        return replace(
            result,
            usage=result.usage if usage is None else _sum_usage((usage, result.usage)),
            request_attempts=attempts + result.request_attempts,
            recovery=(label, *result.recovery),
        )


# Coldstart compaction: a pointer to the saved log, and a summary file that the
# model keeps next to it.

COLDSTART_POINTER_PREFIX = (
    "This conversation was compacted. Its log up to this point is saved in"
)
# Pointers began so until October 17, when they had no summary file.
_LEGACY_COLDSTART_POINTER_PREFIX = "This conversation was compacted without a summary."
COLDSTART_USER_MESSAGE_CHARS = 2_000
# The tool that reads the log. On Claude Relay the model calls it by its native
# name, mcp__pythia__exec_command, so the pointer names no tool.
_COLDSTART_READER = "exec_command"
_COLDSTART_LOG_FORMAT = (
    'Each line is a JSON item with a "type", such as "message" ("role", '
    '"content"), "tool_call" ("name", "arguments_json"), "tool_result" '
    '("output") or "context_prefix" (an earlier compaction).'
)
# The summary file: the save's path with this suffix in place of ``.jsonl``,
# as for the debug trace files. Its first line says what it covers, such as
# "Covers lines 1–3756 of session_4a8f….", and the rest is pi's format.
_SUMMARY_SUFFIX = ".summary.md"
# Only this much of the summary is read: its first line.
_SUMMARY_LINE_BYTES = 4_096
# The first line, read leniently: spaces, any dash, a trailing period, and the
# session in backticks or not.
_SUMMARY_FIRST_LINE = re.compile(
    r"Covers (?:lines 1\s*[-\u2013\u2014]\s*(?P<last>[0-9]+)|(?P<empty>no lines yet))"
    r"(?: of (?P<session>.+?))?\.?"
)
_SESSION_TOKEN = re.compile(r"[^\s`]+")
# The recovery entry of a compaction that moved another session's summary aside.
_SUMMARY_MOVED_ASIDE = "summary_moved_aside"


@dataclass(frozen=True)
class CompactionLog:
    """The save that a coldstart pointer names.

    ``path`` is the save file. ``cwd`` is the tools' working directory: the
    pointer gives the path relative to it when the save is inside it.
    """

    path: Path
    cwd: Optional[Path] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.cwd is not None:
            object.__setattr__(self, "cwd", Path(self.cwd))

    def reference(self) -> Tuple[str, bool]:
        """The path to show, and whether it is relative to ``cwd``."""
        if self.cwd is not None:
            for path, base in (
                (self.path.absolute(), self.cwd.absolute()),
                (self.path.resolve(), self.cwd.resolve()),
            ):
                try:
                    return path.relative_to(base).as_posix(), True
                except ValueError:
                    continue
        return str(self.path.absolute()), False

    @property
    def summary_path(self) -> Path:
        """The summary file next to the save: ``.jsonl`` becomes ``.summary.md``."""
        return self.path.with_suffix(_SUMMARY_SUFFIX)

    def summary_reference(self) -> str:
        """The summary's path to show: relative to ``cwd`` when the save's is."""
        reference, relative = self.reference()
        if relative:
            return PurePosixPath(reference).with_suffix(_SUMMARY_SUFFIX).as_posix()
        return str(self.summary_path.absolute())

    def prepare_summary(self, session: Optional[str], length: int) -> "_SummaryFile":
        """Read the summary's first line, or write a template; say what it covers.

        ``session`` is the save's (its ``Init``'s prefix ID), or None, and
        ``length`` the number of its lines that the compaction covers. A
        missing summary gets the template (``_summary_template``), a new
        private file, never written through a link. A summary whose first line
        names another session was written for an earlier save at this path,
        since replaced (``--resume=False``): it is renamed to the first free
        ``.summary.md.<n>``, from 0, and gets the template. Nothing else is
        written; the model edits the summary. Raises ``CompactionLogError``
        when the summary can't be read, moved, or created.
        """
        path, name = self.summary_path, self.summary_reference()
        data = _read_summary_line(path, name)
        if data is None:
            state, last = "template", None
        else:
            state, last = _summary_state(data, session, length)
        moved = state == "foreign"
        if moved:
            _move_summary_aside(path, name)
        if data is None or moved:
            _create_summary(path, name, _summary_template(session))
            state, last = "template", None
        return _SummaryFile(name, state, last, moved_aside=moved)

    def check(self, context: InteractionContext) -> None:
        """Check that the save holds ``context``, or raise ``CompactionLogError``.

        The save's first N lines must be the context's N items, in order.
        Later lines may follow: ``/compact`` saves its own call before it
        compacts. An incomplete last line, which an interrupted append leaves,
        is ignored, as on resume. Items are compared as objects, then by their
        save encoding when the objects differ.
        """
        from .save import SaveError
        from .save import load_interaction_save

        name = self.reference()[0]
        try:
            saved = load_interaction_save(self.path)
        except SaveError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                raise CompactionLogError(f"the save {name} was not found") from None
            raise CompactionLogError(f"the save {name} can't be loaded: {exc}") from None
        if len(saved) < len(context):
            raise CompactionLogError(
                f"the save {name} has {len(saved)} lines, fewer than the context's "
                f"{len(context)} items"
            )
        for line, (item, saved_item) in enumerate(zip(context.items, saved.items), 1):
            if item is saved_item or item == saved_item:
                continue
            if _save_encoding(item) != _save_encoding(saved_item):
                raise CompactionLogError(
                    f"line {line} of the save {name} differs from the context"
                )


def _save_encoding(item: InteractionItem) -> str:
    """An item as its save line encodes it, with sorted keys."""
    from .save import interaction_item_to_dict

    return json.dumps(interaction_item_to_dict(item), sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class _SummaryFile:
    """A coldstart summary as its pointer describes it (``prepare_summary``).

    ``state`` is ``"template"`` (it covers no lines yet), ``"covers"`` (lines
    1 to ``last``), or ``"unknown"`` (its first line doesn't say which lines
    of this log it covers). ``reference`` is its path to show.
    """

    reference: str
    state: str
    last: Optional[int] = None
    moved_aside: bool = False


def _summary_line(last: Optional[int], session: Optional[str]) -> str:
    """A summary's first line: it covers no lines yet, or lines 1 to ``last``."""
    covered = "no lines yet" if last is None else f"lines 1\u2013{last}"
    return f"Covers {covered}{'' if session is None else f' of {session}'}."


def _summary_template(session: Optional[str]) -> str:
    """A new summary: its first line, then pi's sections with their hints."""
    start = SUMMARIZATION_PROMPT.index("## Goal")
    end = SUMMARIZATION_PROMPT.index("\n\nKeep each section concise")
    return f"{_summary_line(None, session)}\n\n{SUMMARIZATION_PROMPT[start:end]}\n"


def _summary_state(data: bytes, session: Optional[str], length: int) -> Tuple[str, Optional[int]]:
    """What a summary's first line says, given the save's ``session`` and ``length``.

    Returns ``("covers", K)`` for lines 1 to K, where K is at most
    ``length``; ``("template", None)``; ``("foreign", None)`` when it names
    another session, as one word (an ID); or ``("unknown", None)``. A line
    that names no session is ``unknown`` in a save that has one.
    """
    try:
        line = data.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        return "unknown", None
    match = _SUMMARY_FIRST_LINE.fullmatch(line)
    if match is None:
        return "unknown", None
    named = match["session"]
    if named is not None:
        named = named.strip().strip("`").strip()
    if named != session:
        if named is not None and _SESSION_TOKEN.fullmatch(named):
            return "foreign", None
        return "unknown", None
    if match["empty"] is not None:
        return "template", None
    last = int(match["last"])
    if not 1 <= last <= length:
        return "unknown", None
    return "covers", last


def _os_reason(error: OSError) -> str:
    """An OS error's reason, without the path it names."""
    return error.strerror or str(error) or type(error).__name__


def _read_summary_line(path: Path, name: str) -> Optional[bytes]:
    """The first line of the summary at ``path``, as bytes; None if there is none.

    It is opened without blocking and must be a regular file; only its first
    ``_SUMMARY_LINE_BYTES`` are read.
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        if os.path.islink(path):
            raise CompactionLogError(
                f"the summary {name} is a symbolic link to a missing file") from None
        return None
    except OSError as exc:
        raise CompactionLogError(f"the summary {name} can't be read: {_os_reason(exc)}") from None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CompactionLogError(f"the summary {name} is not a regular file")
        data = b""
        while len(data) < _SUMMARY_LINE_BYTES and b"\n" not in data:
            chunk = os.read(descriptor, _SUMMARY_LINE_BYTES - len(data))
            if not chunk:
                break
            data += chunk
    except OSError as exc:
        raise CompactionLogError(f"the summary {name} can't be read: {_os_reason(exc)}") from None
    finally:
        os.close(descriptor)
    return data.split(b"\n", 1)[0]


def _move_summary_aside(path: Path, name: str) -> None:
    """Rename a summary to the first free ``<name>.<n>``, from 0."""
    index = 0
    while os.path.lexists(path.with_name(f"{path.name}.{index}")):
        index += 1
    try:
        os.rename(path, path.with_name(f"{path.name}.{index}"))
    except OSError as exc:
        raise CompactionLogError(
            f"the summary {name} can't be moved aside: {_os_reason(exc)}") from None


def _create_summary(path: Path, name: str, text: str) -> None:
    """Create the summary as a new private file, never through a link."""
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_CLOEXEC", 0))
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise CompactionLogError(
            f"the summary {name} can't be created: {_os_reason(exc)}") from None
    try:
        try:
            data = memoryview(text.encode("utf-8"))
            while data:
                data = data[os.write(descriptor, data):]
        finally:
            os.close(descriptor)
    except OSError as exc:
        try:
            os.unlink(path)  # no half-written template
        except OSError:
            pass
        raise CompactionLogError(
            f"the summary {name} can't be written: {_os_reason(exc)}") from None


def _session_id(context: InteractionContext) -> Optional[str]:
    """The save's session: its ``Init``'s prefix ID, or None without one."""
    first = context[0] if len(context) else None
    return first.prefix_id if isinstance(first, Init) else None


def compaction_log(path, environment=None) -> Optional[CompactionLog]:
    """The log of the save at ``path``, with ``environment``'s working directory.

    ``None`` without a save. An environment without a ``cwd`` (one without
    the default tools) gives the pointer an absolute path.
    """
    if path is None:
        return None
    cwd = getattr(environment, "cwd", None)
    return CompactionLog(path, cwd if isinstance(cwd, (str, os.PathLike)) else None)


def coldstart_unavailable(tool_specs: Iterable[object], log_path) -> Optional[str]:
    """Why coldstart compaction can't run with these tools and save, or None.

    It needs a save for its pointer to name, and ``exec_command`` among the
    tools to read it. Frontends check this once per role at launch: a role's
    tools and save don't change while it runs.
    """
    if log_path is None:
        return "coldstart compaction needs a save to point to"
    if not any(getattr(spec, "name", None) == _COLDSTART_READER for spec in tool_specs):
        return "coldstart compaction needs exec_command to read the saved log"
    return None


def is_coldstart_pointer(item: object) -> bool:
    """Whether an item is the user message of a coldstart checkpoint, old or new."""
    return (
        isinstance(item, Message)
        and item.role == "user"
        and isinstance(item.content, str)
        and item.content.startswith(
            (COLDSTART_POINTER_PREFIX, _LEGACY_COLDSTART_POINTER_PREFIX))
    )


def _is_quotable_request(item: object) -> bool:
    """A top-level user message that is neither a summary nor a pointer.

    Summaries and pointers live in checkpoints, not as top-level user
    messages, but they are skipped anyway.
    """
    return (
        isinstance(item, Message)
        and item.role == "user"
        and not is_compaction_summary(item)
        and not is_coldstart_pointer(item)
    )


def _quoted_text(message: Message) -> str:
    """A message's text, cut to ``COLDSTART_USER_MESSAGE_CHARS``."""
    text = _message_text(message)
    if len(text) <= COLDSTART_USER_MESSAGE_CHARS:
        return text
    omitted = len(text) - COLDSTART_USER_MESSAGE_CHARS
    return f"{text[:COLDSTART_USER_MESSAGE_CHARS]}\n[{omitted} more characters omitted]"


def _quoted_user_messages(
    items: Sequence[InteractionItem],
) -> Tuple[Tuple[str, int, str], ...]:
    """The raw log's initial and latest user messages, as (label, line, text).

    The labels are "The initial user message" and "The latest user message",
    or just "The user message" when the log has only one; there are none when
    it has none (``_is_quotable_request``). Each text is cut to
    ``COLDSTART_USER_MESSAGE_CHARS``.
    """
    initial = next(
        (index for index, item in enumerate(items) if _is_quotable_request(item)), None)
    if initial is None:
        return ()
    latest = next(index for index in range(len(items) - 1, initial - 1, -1)
                  if _is_quotable_request(items[index]))
    if initial == latest:
        chosen = (("The user message", latest),)
    else:
        chosen = (("The initial user message", initial), ("The latest user message", latest))
    return tuple((label, index + 1, _quoted_text(items[index])) for label, index in chosen)


class ColdstartCompactor:
    """Compaction without a model request: a pointer to the saved log and its summary.

    The new prefix holds the instructions, any Responses checkpoints, and one
    user message. The message names the save and its lines before the
    compaction (1 to N, where N is the context's length; saves are
    append-only, so those lines never change), the lines' format, and the
    summary file next to the save, with the lines it covers. It quotes the
    session's initial and latest user messages with their lines, or its only
    one (``_quoted_user_messages``). It asks the model to read the summary and
    what it needs from the log, then, before it continues, to update the
    summary to cover lines 1 to N, and to pick up where the log leaves off.
    The summary's first line says what it covers, so a skipped update only
    means more to read next time. It needs ``exec_command`` among
    ``tools``. Before it installs the pointer, it checks that the save holds
    the context (``CompactionLog.check``), and raises ``CompactionLogError``
    if not, so a sequence moves on to its next method; likewise when the
    summary can't be read or created (``CompactionLog.prepare_summary``). A
    summary moved aside is recorded as ``recovery`` ``summary_moved_aside``.
    """

    def __init__(self, log: CompactionLog) -> None:
        if not isinstance(log, CompactionLog):
            raise TypeError("log must be CompactionLog")
        self.log = log

    @_timed_compact
    def compact(
        self,
        context: InteractionContext,
        *,
        tools: Sequence["ToolSpec"] = (),
        sample_params: Optional[SampleParams] = None,
        instructions: Optional[str] = None,
    ) -> CompactionResult:
        del sample_params
        if not isinstance(context, InteractionContext):
            raise TypeError("context must be InteractionContext")
        if instructions is not None and not isinstance(instructions, str):
            raise TypeError("instructions must be a string or None")
        reason = coldstart_unavailable(tools, self.log.path)
        if reason is not None:
            raise CompactionError(reason)
        try:
            context.assert_model_ready()
        except ContextValidationError as exc:
            raise CompactionError(str(exc)) from exc
        instruction_prefix, rest = _split_instruction_prefix(context.model_items())
        checkpoints = tuple(
            item for item in rest
            if isinstance(item, OpaqueCompaction) and item.protocol == "responses"
        )
        if not _has_content(
            item for item in rest
            if not (isinstance(item, OpaqueCompaction) and item.protocol == "responses")
            and not is_coldstart_pointer(item)
        ):
            raise NothingToCompact(
                "there is nothing to compact beyond a coldstart pointer"
            )
        self.log.check(context)
        session = _session_id(context)
        summary = self.log.prepare_summary(session, len(context))
        focus = None if instructions is None else instructions.strip() or None
        pointer = Message(role="user", content=self._pointer(context, focus, summary, session))
        return CompactionResult(
            items=(ContextPrefix((*instruction_prefix, *checkpoints, pointer)),),
            protocol="coldstart",
            request_attempts=0,
            recovery=(_SUMMARY_MOVED_ASIDE,) if summary.moved_aside else (),
        )

    def _pointer(
        self,
        context: InteractionContext,
        focus: Optional[str],
        summary: _SummaryFile,
        session: Optional[str],
    ) -> str:
        reference, relative = self.log.reference()
        where = (
            f"`{reference}` (relative to your tools' working directory)"
            if relative else f"`{reference}`"
        )
        length = len(context)
        paragraphs = [
            f"{COLDSTART_POINTER_PREFIX} {where}, lines 1\u2013{length}; later "
            f"lines continue from here. {_COLDSTART_LOG_FORMAT} "
            f"{_summary_sentence(summary)}"
        ]
        for label, line, text in _quoted_user_messages(context.items):
            paragraphs.append(
                f"{label} is on line {line}:\n"
                f"<user-message>\n{text}\n</user-message>"
            )
        paragraphs.append(_summary_instruction(summary, length, session))
        return "\n\n".join(paragraphs) + _focus_suffix(focus)


def _summary_sentence(summary: _SummaryFile) -> str:
    """The pointer's sentence about the summary: where it is, what it covers."""
    where = f"`{summary.reference}`"
    if summary.state == "template":
        return f"A summary template is in {where}; it covers no lines yet."
    if summary.state == "covers":
        return f"A summary of lines 1\u2013{summary.last} is in {where}."
    return (f"A summary is in {where}, but its first line doesn't say which "
            "lines of this log it covers.")


def _summary_instruction(summary: _SummaryFile, length: int, session: Optional[str]) -> str:
    """The pointer's last paragraph: read, update the summary to cover lines 1–``length``, go on.

    The model reads the log's messages after the lines the summary covers,
    or all of them, since the update needs what they say.
    """
    if summary.state == "covers":
        read = (f"Read the summary, then the messages after line {summary.last} and "
                "anything else you need from the log.")
    else:
        read = ("Read the summary, then the log's messages and anything else you "
                "need from it.")
    if summary.state == "template":
        update = (f"fill in the summary to cover lines 1\u2013{length}: keep exact "
                  "paths, names and error messages")
    else:
        update = (f"edit the summary in place to cover lines 1\u2013{length}: keep what "
                  "still applies, add what is new, keep exact paths, names and error "
                  "messages")
    return (f"{read} Before you continue, {update}, and change its first line to "
            f"\"{_summary_line(length, session)}\" Then pick up where the log left off.")


# Compaction sequences.

_NOTICE_DETAIL_CHARS = 300


@dataclass(frozen=True)
class CompactionFailure:
    """One failed attempt of a compaction sequence (see ``run_compaction``).

    ``failure`` is the item to save for it: the error's own ``ModelFailure``,
    or one made from the error. ``next_method`` is what the sequence tries
    next, or None when nothing is left; ``retry`` says whether that is another
    attempt of the same entry.
    """

    method: str
    error: BaseException
    failure: ModelFailure
    attempt: int
    attempts: int
    next_method: Optional[str] = None
    retry: bool = False

    @property
    def label(self) -> str:
        """Its ``CompactionMetadata.recovery`` entry."""
        return f"{self.method}_failed"

    def notice(self) -> str:
        """A one-line notice, such as "pi compaction failed (...); trying coldstart."."""
        detail = self.failure.message
        if len(detail) > _NOTICE_DETAIL_CHARS:
            detail = detail[:_NOTICE_DETAIL_CHARS].rstrip() + "..."
        head = f"{self.method} compaction failed ({type(self.error).__name__}: {detail})"
        if self.next_method is None:
            return f"{head}."
        if self.retry:
            return (
                f"{head}; trying {self.method} again (attempt {self.attempt + 1} "
                f"of {self.attempts})."
            )
        return f"{head}; trying {self.next_method}."


def _failure_item(error: BaseException) -> ModelFailure:
    """The item that records a failed attempt in the save."""
    failure = getattr(error, "failure", None)
    if isinstance(error, ModelError):
        if isinstance(failure, ModelFailure):
            return failure
        # Adapter messages can carry response bodies; failure items don't.
        message = "details were withheld"
    else:
        message = " ".join(str(error).split()) or type(error).__name__
    return ModelFailure(category=type(error).__name__[:128], message=message[:1024])


def _ends_compaction(error: BaseException, *, reason: str, later: bool) -> bool:
    """Whether a failure ends the whole sequence rather than one attempt.

    A cancel or an authentication error ends it, as does any exception that
    is neither a compaction nor a model failure (a bug). ``NothingToCompact``
    ends it too, except in overflow recovery with a later method: pi can't
    make the context fit there, but coldstart can.
    """
    if isinstance(error, NothingToCompact):
        return reason != "overflow" or not later
    if isinstance(error, (ModelCancelled, ModelAuthenticationError)):
        return True
    return not isinstance(error, (CompactionError, ModelError))


def _retryable(error: BaseException) -> bool:
    """Whether another attempt of the same method could succeed.

    Not after a request too large for the window, a configuration error (such
    as text the route can't encode), or a coldstart save that can't be used:
    the same request, or the same save, would fail again.
    """
    return not isinstance(error, (
        NothingToCompact, CompactionContextWindowError, ModelContextWindowError,
        ModelConfigurationError, CompactionLogError,
    ))


def _rejects_focus(model: Model, method: str) -> bool:
    """Codex remote compaction takes no focus text."""
    return method == "provider" and getattr(model, "supports_remote_compaction", False) is True


def run_compaction(
    context: InteractionContext,
    model: Model,
    settings: CompactionSettings,
    *,
    factory: Optional[Callable[..., Compactor]] = None,
    tools: Sequence["ToolSpec"] = (),
    sample_params: Optional[SampleParams] = None,
    instructions: Optional[str] = None,
    reason: str = "threshold",
    log: Optional[CompactionLog] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    on_failure: Optional[Callable[[CompactionFailure], None]] = None,
) -> CompactionResult:
    """Run the sequence of ``settings.mode``: each method in turn, up to its attempts.

    ``factory`` builds one method's compactor, as ``create_default_compactor``
    does (the default); callers pass their own reference so tests can patch
    it. Each attempt compacts a copy of ``context``, so items that
    ``on_failure`` appends to it are part of the next attempt's log.

    - A transport, response, or compaction failure is retried while the
      entry's attempts last, then the next method runs. A summary request too
      large for the window (``CompactionContextWindowError``), a
      ``ModelConfigurationError``, or a coldstart save that can't be used
      (``CompactionLogError``) moves on at once.
    - A stop (``stop_requested``), a cancel, an authentication error, or any
      other exception ends the sequence, raised as it is.
    - ``NothingToCompact`` is raised as it is: nothing is compacted. In
      overflow recovery (``reason="overflow"``) it moves on instead, while a
      later method remains.
    - With focus text (``instructions``), a method that rejects it is
      skipped while a later method remains.
    - coldstart is skipped where it can't run, without a ``log`` or without
      ``exec_command`` among ``tools``, while a later method remains. As the
      last method, it runs and fails.
    - warmpi is built with the compaction's ``reason``.

    ``on_failure`` is called after each failed attempt that doesn't end the
    sequence, including the last one. When every attempt fails, the last
    failure is raised. The result's ``recovery`` lists each failed attempt.

    However the sequence ends, the model is then retired (``retire_model``),
    for the turn loop and the CLI's ``/compact`` alike.
    """
    if reason not in COMPACTION_REASONS:
        raise ValueError(f"reason must be one of {COMPACTION_REASONS}")
    if not isinstance(settings, CompactionSettings):
        raise TypeError("settings must be CompactionSettings")
    try:
        factory = create_default_compactor if factory is None else factory
        stopping = stop_requested if stop_requested is not None else (lambda: False)
        focus = instructions is not None and bool(instructions.strip())
        entries = [(_model_method(model, method), attempts) for method, attempts in settings.methods]
        log_path = None if log is None else log.path

        def skipped(method: str) -> bool:
            if focus and _rejects_focus(model, method):
                return True
            return method == "coldstart" and coldstart_unavailable(tools, log_path) is not None

        runnable = [
            (method, attempts) for index, (method, attempts) in enumerate(entries)
            if not (index + 1 < len(entries) and skipped(method))
        ]
        calls = {"tools": tools, "sample_params": sample_params}
        if instructions is not None:
            calls["instructions"] = instructions
        failed: List[str] = []
        last: Optional[BaseException] = None
        for position, (method, attempts) in enumerate(runnable):
            later = position + 1 < len(runnable)
            # Only coldstart points to the save, and only warmpi takes the reason.
            options = ({"log": log} if method == "coldstart"
                       else {"reason": reason} if method == "warmpi" else {})
            for attempt in range(1, attempts + 1):
                try:
                    compactor = factory(model, replace(settings, mode=method), **options)
                    result = compactor.compact(context.copy(), **calls)
                except Exception as error:
                    if stopping() or _ends_compaction(error, reason=reason, later=later):
                        raise
                    last = error
                else:
                    # Callers check the result's type; it matters here only when
                    # the failed attempts are added to it.
                    if failed:
                        if not isinstance(result, CompactionResult):
                            raise TypeError(
                                "compactor must return CompactionResult, got "
                                f"{type(result).__name__}"
                            )
                        result = replace(result, recovery=(*failed, *result.recovery))
                    return result
                retry = attempt < attempts and _retryable(last)
                report = CompactionFailure(
                    method=method,
                    error=last,
                    failure=_failure_item(last),
                    attempt=attempt,
                    attempts=attempts,
                    next_method=(
                        method if retry else runnable[position + 1][0] if later else None
                    ),
                    retry=retry,
                )
                failed.append(report.label)
                if on_failure is not None:
                    on_failure(report)
                if not retry:
                    break
        assert last is not None
        raise last
    finally:
        # A model may have kept work for the next turn that this compaction
        # leaves behind: the Claude relay's process that answered pi's or
        # warmpi's request, or one whose history a checkpoint replaces.
        retire_model(model)


__all__ = [
    "COLDSTART_POINTER_PREFIX",
    "COMPACTION_METHODS",
    "COMPACTION_MODES",
    "COMPACTION_REASONS",
    "COMPACTION_SUMMARY_PREFIX",
    "COMPACTION_SUMMARY_SUFFIX",
    "DEFAULT_COMPACTION_MODE",
    "ColdstartCompactor",
    "CompactionContextWindowError",
    "CompactionError",
    "CompactionFailure",
    "CompactionLog",
    "CompactionLogError",
    "CompactionResult",
    "CompactionSettings",
    "Compactor",
    "DEFAULT_KEEP_RECENT_TOKENS",
    "ESTIMATED_IMAGE_CHARS",
    "MAX_COMPACTION_ATTEMPTS",
    "NothingToCompact",
    "PiCompactor",
    "SUMMARIZATION_PROMPT",
    "SUMMARIZATION_SYSTEM_PROMPT",
    "TOOL_RESULT_MAX_CHARS",
    "TURN_PREFIX_SUMMARIZATION_PROMPT",
    "UPDATE_SUMMARIZATION_PROMPT",
    "WARMPI_INSTRUCTION",
    "WarmPiCompactor",
    "auto_compaction_due",
    "coldstart_unavailable",
    "compaction_log",
    "compaction_methods",
    "create_default_compactor",
    "estimate_context_tokens",
    "estimate_item_tokens",
    "format_compaction_mode",
    "is_coldstart_pointer",
    "is_compaction_summary",
    "normalize_compaction_mode",
    "parse_compaction_mode",
    "run_compaction",
    "should_auto_compact",
    "uses_host_auto_compaction",
]
