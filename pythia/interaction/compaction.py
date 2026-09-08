from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional
from typing import Protocol
from typing import TYPE_CHECKING
from typing import Tuple

from .context import ContextValidationError
from .context import ModelContext
from .items import ContextCompaction
from .items import InteractionItem
from .items import Instructions
from .items import Message
from .model import Model
from .model import ModelContextWindowError
from .model import ModelSample
from .model import SamplingOptions
from .model import TokenUsage

if TYPE_CHECKING:
    from .environment import ToolSpec
    from .display import DisplayItem


DEFAULT_COMPACTION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff "
    "summary for another LLM that will resume the task.\n\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n\n"
    "Be concise, structured, and focused on helping the next LLM seamlessly "
    "continue the work."
)

DEFAULT_SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a "
    "summary of its thinking process. You also have access to the state of "
    "the tools that were used by that language model. Use this to build on "
    "the work that has already been done and avoid duplicating work. Here is "
    "the summary produced by the other language model, use the information "
    "in this summary to assist with your own analysis:"
)


class CompactionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompactionResult:
    items: Tuple[InteractionItem, ...]
    usage: TokenUsage = field(default_factory=TokenUsage)

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if len(items) != 1 or not isinstance(items[0], ContextCompaction):
            raise CompactionError(
                "compaction result must contain exactly one ContextCompaction"
            )
        try:
            ModelContext(items)
        except ContextValidationError as exc:
            raise CompactionError(str(exc)) from exc
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")
        object.__setattr__(self, "items", items)

    def context_items(self) -> Tuple[InteractionItem, ...]:
        return self.items

    def display_items(self) -> Tuple["DisplayItem", ...]:
        from .display import render_interaction_items

        return render_interaction_items(self.items)


class Compactor(Protocol):
    def compact(
        self,
        context: ModelContext,
        *,
        tools: Sequence["ToolSpec"] = (),
    ) -> CompactionResult:
        ...


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
    effective = None
    for _item in items:
        if isinstance(_item, Instructions):
            effective = _item
    prefix = []
    if effective is not None:
        prefix.append(effective)
    for item in items:
        if isinstance(item, Instructions):
            continue
        if isinstance(item, Message) and item.role in {"system", "developer"}:
            prefix.append(item)
            continue
        break
    return tuple(prefix)


def _select_retained_user_messages(
    messages: Sequence[Message],
    max_tokens: int,
) -> Tuple[Message, ...]:
    selected_reversed = []
    remaining = max(0, max_tokens)
    for message in reversed(messages):
        if remaining == 0:
            break
        tokens = _approx_token_count(message.content)
        if tokens <= remaining:
            selected_reversed.append(message)
            remaining -= tokens
            continue
        selected_reversed.append(
            Message(
                role="user",
                content=_truncate_text_to_tokens(message.content, remaining),
            )
        )
        break
    selected_reversed.reverse()
    return tuple(selected_reversed)


def _drop_oldest_non_instruction_item(
    items: List[InteractionItem],
    compaction_prompt: Message,
) -> bool:
    index = 0
    while index < len(items) and (
        isinstance(items[index], Instructions)
        or (
            isinstance(items[index], Message)
            and items[index].role in {"system", "developer"}
        )
    ):
        index += 1
    if index >= len(items):
        return False

    del items[index]
    while True:
        try:
            ModelContext((*items, compaction_prompt))
            return True
        except ContextValidationError:
            if index >= len(items):
                return False
            del items[index]


class PromptSummarizingCompactor:
    def __init__(
        self,
        model: Model,
        *,
        prompt: str = DEFAULT_COMPACTION_PROMPT,
        summary_prefix: str = DEFAULT_SUMMARY_PREFIX,
        retained_user_message_tokens: int = 20_000,
        options: Optional[SamplingOptions] = None,
        retain_user_message: Optional[Callable[[Message], bool]] = None,
    ) -> None:
        if not hasattr(model, "sample") or not callable(model.sample):
            raise TypeError("model must provide sample(...)")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not isinstance(summary_prefix, str) or not summary_prefix.strip():
            raise ValueError("summary_prefix must not be empty")
        if (
            isinstance(retained_user_message_tokens, bool)
            or not isinstance(retained_user_message_tokens, int)
            or retained_user_message_tokens < 0
        ):
            raise ValueError(
                "retained_user_message_tokens must be a nonnegative integer"
            )
        if options is not None and not isinstance(options, SamplingOptions):
            raise TypeError("options must be SamplingOptions or None")
        if retain_user_message is not None and not callable(retain_user_message):
            raise TypeError("retain_user_message must be callable or None")

        self._model = model
        self._prompt = prompt
        self._summary_prefix = summary_prefix
        self._retained_user_message_tokens = retained_user_message_tokens
        self._options = options or SamplingOptions(
            temperature=0.0,
            max_tokens=2_000,
        )
        self._retain_user_message = retain_user_message

    def _is_retained_user_message(self, message: Message) -> bool:
        if message.role != "user":
            return False
        if message.content.startswith(f"{self._summary_prefix}\n"):
            return False
        if self._retain_user_message is not None:
            return bool(self._retain_user_message(message))
        return True

    def compact(
        self,
        context: ModelContext,
        *,
        tools: Sequence["ToolSpec"] = (),
    ) -> CompactionResult:
        del tools
        if not isinstance(context, ModelContext):
            raise TypeError("context must be ModelContext")
        try:
            context.assert_model_ready()
        except ContextValidationError as exc:
            raise CompactionError(str(exc)) from exc

        active_items = list(context.model_items())
        instruction_prefix = _leading_instruction_prefix(active_items)
        user_messages = tuple(
            item
            for item in active_items
            if isinstance(item, Message)
            and self._is_retained_user_message(item)
        )
        compaction_prompt = Message(role="user", content=self._prompt)
        request_items = list(active_items)

        while True:
            try:
                temporary_context = ModelContext(
                    (*request_items, compaction_prompt)
                )
                sample = self._model.sample(
                    temporary_context,
                    tools=(),
                    options=self._options,
                )
                break
            except ModelContextWindowError as exc:
                if not _drop_oldest_non_instruction_item(
                    request_items,
                    compaction_prompt,
                ):
                    raise CompactionError(
                        "compaction request exceeds the model context window "
                        "after all removable items were discarded"
                    ) from exc

        if not isinstance(sample, ModelSample):
            raise CompactionError(
                f"model returned {type(sample).__name__}, expected ModelSample"
            )
        if sample.tool_calls:
            raise CompactionError(
                "compaction model response must not contain tool calls"
            )
        summary_text = sample.last_assistant_text
        if summary_text is None or not summary_text.strip():
            raise CompactionError(
                "compaction model response has no non-empty assistant summary"
            )

        retained_users = _select_retained_user_messages(
            user_messages,
            self._retained_user_message_tokens,
        )
        summary_message = Message(
            role="user",
            content=f"{self._summary_prefix}\n{summary_text.strip()}",
        )
        replacement_items = (
            *instruction_prefix,
            *retained_users,
            summary_message,
        )
        checkpoint = ContextCompaction(
            replacement_items=tuple(replacement_items),
        )
        return CompactionResult(
            items=(checkpoint,),
            usage=sample.usage,
        )


__all__ = [
    "CompactionError",
    "CompactionResult",
    "Compactor",
    "DEFAULT_COMPACTION_PROMPT",
    "DEFAULT_SUMMARY_PREFIX",
    "PromptSummarizingCompactor",
]
