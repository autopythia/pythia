"""Opt-in model tools for experiments; never part of the default tool set."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from .environment import Tool
from .environment import ToolOutcome
from .environment import ToolSpec
from .items import Message


def create_inject_user_message_tool() -> Tool:
    """Create a stateless test hook requesting one fixed user message per call.

    The caller must append the whole environment result after completing the
    pending tool-call batch. The handler never mutates the interaction context.
    """

    def execute(
        arguments: Mapping[str, object],
        *,
        timeout_seconds: Optional[float] = None,
    ) -> ToolOutcome:
        del timeout_seconds
        if not isinstance(arguments, Mapping):
            raise TypeError("tool arguments must be a mapping")
        if arguments:
            raise ValueError("experimental_inject_user_message takes no arguments")
        return ToolOutcome(
            output="Synthetic user message queued.",
            user_messages=(Message(role="user", text="hello world"),),
        )

    return Tool(
        spec=ToolSpec(
            name="experimental_inject_user_message",
            description=(
                "Experimental integration-test hook. Requests that the host "
                "append one fixed synthetic user message after the current "
                "tool-call batch completes. Takes no arguments; does not "
                "contact or wait for a human."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        ),
        handler=execute,
    )


__all__ = ["create_inject_user_message_tool"]
