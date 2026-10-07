"""Where a saved log's last turn stopped, for continuing it without a message.

A log can stop inside a turn: the app was quit or crashed during a sample or
a tool batch, or a sample failed and was not retried. When the turn loop's
next step would be a sample, ``/continue`` can take that sample on the saved
context, with no new user message. This module decides that from the raw log
alone; it never looks at tool names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ..items import CompactionMetadata
from ..items import ContextPrefix
from ..items import InteractionItem
from ..items import Instructions
from ..items import Message
from ..items import ModelFailure
from ..items import ModelSampleBoundary
from ..items import OpaqueCompaction
from ..items import Reasoning
from ..items import SampleMetadata
from ..items import ToolCall
from ..items import ToolResult
from ..items import Tools
from ..items import UserInteractionBoundary
from ..items import UserToolCall
from ..items import UserToolResult


# Records that say nothing about where the last turn stopped: metadata,
# boundaries, tool snapshots, user tools, and records that can follow a
# finished turn as well as an unfinished one (an instructions update, such as
# auto's restart notice, and a compaction checkpoint). The records before them
# decide.
_SKIPPED = (ModelSampleBoundary, SampleMetadata, CompactionMetadata, UserInteractionBoundary,
            UserToolCall, UserToolResult, Tools, Instructions, ContextPrefix)


def describe_tail(item: Optional[InteractionItem]) -> str:
    """A short description of a log's last record, as in resume notices."""
    if isinstance(item, OpaqueCompaction):
        return "a compaction checkpoint"
    if isinstance(item, ContextPrefix):
        return "a context-prefix checkpoint"
    if isinstance(item, ToolResult):
        return "tool results"
    if isinstance(item, Message) and item.role == "user":
        return "a user submission"
    if isinstance(item, Message) and item.role == "assistant":
        return "assistant output"
    if isinstance(item, Instructions):
        return "an instructions update"
    if isinstance(item, ModelFailure):
        return "a failed model attempt"
    return "incomplete model output"


@dataclass(frozen=True)
class TurnTail:
    """The record that decides where a saved log's last turn stopped.

    ``item`` is None for a log with no such record.
    """

    item: Optional[InteractionItem]

    @property
    def unfinished(self) -> bool:
        """True when the turn loop's next step would be a sample.

        That is after tool results (including results closed after a restart,
        a failed sample, or a skip), model input (a user message: a query, a
        steer, a follow-up, or a tool-injected message), a failed sample, a
        paused provider compaction, or model output without an answer
        (reasoning, or blank assistant text). Assistant text may be a final
        answer whose turn summary was not saved, since the stop reason is not
        saved, so it never counts.
        """
        item = self.item
        if isinstance(item, (ToolResult, ModelFailure, OpaqueCompaction, Reasoning)):
            return True
        if isinstance(item, Message):
            return item.role != "assistant" or not item.content_text.strip()
        return False  # a turn summary, the log's start, or unresolved tool calls

    @property
    def refusal(self) -> Optional[str]:
        """Why a sample cannot continue the last turn; None when it can."""
        if self.unfinished:
            return None
        if isinstance(self.item, Message):
            return "the log ends with assistant text, which may be a final answer"
        if isinstance(self.item, ToolCall):
            return "the log ends with unresolved tool calls"
        return "the last turn ended"

    @property
    def description(self) -> str:
        return describe_tail(self.item)


def turn_tail(items: Iterable[InteractionItem]) -> TurnTail:
    """Find where the last turn of a raw log stopped (see :class:`TurnTail`).

    Pass the raw log (``context.items``), not the model projection: a
    compaction checkpoint is skipped, so the records before it decide.
    """
    for item in reversed(tuple(items)):
        if not isinstance(item, _SKIPPED):
            return TurnTail(item)  # a TurnSummary or Init means the last turn ended
    return TurnTail(None)


def unfinished_turn(items: Iterable[InteractionItem]) -> bool:
    """True when a sample would continue the raw log's last turn."""
    return turn_tail(items).unfinished


__all__ = ["TurnTail", "describe_tail", "turn_tail", "unfinished_turn"]
