"""Where a saved log's last turn stopped: the rule behind /continue."""

from __future__ import annotations

import unittest

from pythia.interaction import CompactionMetadata, ContextPrefix, Init, Instructions, Message
from pythia.interaction import ModelFailure, ModelSampleBoundary, OpaqueCompaction, Reasoning
from pythia.interaction import SampleMetadata, TokenUsage, ToolCall, ToolResult, Tools
from pythia.interaction import TurnSummary, UserInteractionBoundary
from pythia.interaction.items import UserToolCall, UserToolResult
from pythia.interaction.loop import turn_tail, unfinished_turn


_START = (Init("m"), Tools(), Message("user", "task"), UserInteractionBoundary())
_ENDED = "the last turn ended"
_ANSWER = "the log ends with assistant text, which may be a final answer"


def _compact(prefix_item):
    """A manual /compact: user-tool records around a checkpoint."""
    call = ToolCall("compact", "user_1", "{}")
    return (UserToolCall(call), UserToolResult(ToolResult("user_1", "compacted")),
            ContextPrefix((prefix_item,)), CompactionMetadata(TokenUsage(), "pi"))


class TurnTailTests(unittest.TestCase):
    def test_one_case_per_kind_of_last_record(self):
        sample = (SampleMetadata(TokenUsage()), ModelSampleBoundary())
        cases = (
            ("nothing", (), False, _ENDED),
            ("the log's start", (Init("m"), Instructions("x"), Tools()), False, _ENDED),
            ("a turn summary", (*_START, Message("assistant", "done"), *sample,
                                TurnSummary(sample_count=1)), False, _ENDED),
            ("tool results", (*_START, ToolCall("t", "a", "{}"), *sample,
                              ToolResult("a", "ok")), True, None),
            ("a closed or skipped result", (*_START, ToolCall("t", "a", "{}"), *sample,
                                            ToolResult("a", "unavailable", success=False)),
             True, None),
            ("a user message", _START, True, None),
            ("a steer", (*_START, ToolCall("t", "a", "{}"), *sample, ToolResult("a", "ok"),
                         Message("user", "steer")), True, None),
            ("a failed sample", (*_START, ModelFailure("transport", "lost"),
                                 ModelSampleBoundary()), True, None),
            ("a paused compaction", (*_START, OpaqueCompaction.from_messages("x"), *sample),
             True, None),
            ("reasoning", (*_START, Reasoning("thinking"), *sample), True, None),
            ("a blank answer", (*_START, Message("assistant", " "), *sample), True, None),
            ("assistant text", (*_START, Message("assistant", "maybe final"), *sample),
             False, _ANSWER),
            ("unresolved calls", (*_START, ToolCall("t", "a", "{}"), *sample), False,
             "the log ends with unresolved tool calls"),
        )
        for name, items, unfinished, refusal in cases:
            with self.subTest(name):
                tail = turn_tail(items)
                self.assertEqual(tail.unfinished, unfinished)
                self.assertEqual(unfinished_turn(items), unfinished)
                self.assertEqual(tail.refusal, refusal)

    def test_records_after_a_turn_do_not_decide(self):
        finished = (*_START, Message("assistant", "done"), ModelSampleBoundary(),
                    TurnSummary(sample_count=1))
        stopped = (*_START, ToolCall("t", "a", "{}"), ModelSampleBoundary(),
                   ToolResult("a", "unavailable", success=False))
        for name, tail in (("auto's restart notice", (Instructions("Restart notice"),)),
                           ("an instructions-only resume", (Instructions("new"), Tools())),
                           ("a manual /compact", _compact(Message("user", "summary")))):
            with self.subTest(name):
                self.assertFalse(unfinished_turn((*finished, *tail)))
                self.assertTrue(unfinished_turn((*stopped, *tail)))
        # An automatic compaction runs inside a turn, after its input.
        automatic = (ContextPrefix((Message("user", "summary"),)),
                     CompactionMetadata(TokenUsage(), "pi"))
        self.assertTrue(unfinished_turn((*_START, *automatic)))
        self.assertEqual(turn_tail((*stopped, *automatic)).description, "tool results")


if __name__ == "__main__":
    unittest.main()
