from __future__ import annotations

import json
import unittest
from typing import Any

from pythia.interaction.chat_completions import ChatCompletionsEndpoint
from pythia.interaction.chat_completions import ChatCompletionsModel
from pythia.interaction.context import ModelContext
from pythia.interaction.items import Message
from pythia.interaction.items import Reasoning


class _FakeResponse:
    def __init__(self, payload: Any):
        self.status = 200
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        return None


class _FakeOpener:
    def __init__(self, payload: Any):
        self.payload = payload

    def __call__(self, request, *, timeout):
        return _FakeResponse(self.payload)


def _sample_message(message: dict[str, Any]):
    opener = _FakeOpener(
        {
            "choices": [
                {
                    "finish_reason": (
                        "tool_calls" if message.get("tool_calls") else "stop"
                    ),
                    "message": message,
                }
            ]
        }
    )
    endpoint = ChatCompletionsEndpoint(api_url="http://127.0.0.1:1")
    model = ChatCompletionsModel(endpoint, opener=opener)
    context = ModelContext((Message(role="user", text="hello"),))
    return model.sample(context)


class ReasoningFieldTests(unittest.TestCase):
    def test_prefers_nonempty_reasoning_over_empty_legacy_field(self):
        sample = _sample_message(
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "",
                "reasoning": "new field reasoning",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "tool", "arguments": "{}"},
                    }
                ],
            }
        )

        self.assertEqual(sample.stop_reason, "tool_use")
        self.assertEqual(
            tuple(item.text for item in sample.items if isinstance(item, Reasoning)),
            ("new field reasoning",),
        )
        self.assertEqual(len(sample.tool_calls), 1)

    def test_supports_legacy_reasoning_content(self):
        sample = _sample_message(
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "legacy reasoning",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "tool", "arguments": "{}"},
                    }
                ],
            }
        )

        self.assertEqual(sample.stop_reason, "tool_use")
        self.assertEqual(
            tuple(item.text for item in sample.items if isinstance(item, Reasoning)),
            ("legacy reasoning",),
        )


if __name__ == "__main__":
    unittest.main()
