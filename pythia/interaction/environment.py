from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Tuple

from .items import InteractionItem
from .items import ToolCall
from .items import ToolResult

if TYPE_CHECKING:
    from .display import DisplayItem


_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class EnvironmentError(ValueError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool name must not be empty")
        name = self.name.strip()
        if _TOOL_NAME_RE.fullmatch(name) is None:
            raise ValueError(
                "tool name may contain only letters, digits, underscores, and hyphens"
            )
        object.__setattr__(self, "name", name)

        if not isinstance(self.description, str):
            raise TypeError("tool description must be a string")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("tool parameters must be a mapping")
        object.__setattr__(self, "parameters", dict(self.parameters))


@dataclass(frozen=True)
class ToolOutcome:
    output: str
    success: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.output, str):
            raise TypeError("tool output must be a string")
        if not isinstance(self.success, bool):
            raise TypeError("tool success must be a bool")


class ToolHandler(Protocol):
    def __call__(
        self,
        arguments: Mapping[str, object],
        *,
        timeout_seconds: Optional[float] = None,
    ) -> ToolOutcome:
        ...


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    handler: ToolHandler
    timeout_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ToolSpec):
            raise TypeError("spec must be ToolSpec")
        if not callable(self.handler):
            raise TypeError("handler must be callable")
        if self.timeout_seconds is not None:
            value = self.timeout_seconds
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(
                    "timeout_seconds must be positive and finite or None"
                )
            object.__setattr__(self, "timeout_seconds", float(value))


@dataclass(frozen=True)
class EnvironmentResult:
    items: Tuple[InteractionItem, ...]

    def __post_init__(self) -> None:
        items = tuple(self.items)
        for index, item in enumerate(items):
            if not isinstance(item, ToolResult):
                raise EnvironmentError(
                    "environment results must contain only ToolResult items; "
                    f"item {index} is {type(item).__name__}"
                )
        object.__setattr__(self, "items", items)

    def context_items(self) -> Tuple[InteractionItem, ...]:
        return self.items

    def display_items(
        self,
        *,
        source_calls: Iterable[ToolCall] = (),
    ) -> Tuple["DisplayItem", ...]:
        from .display import render_interaction_items

        return render_interaction_items(
            self.items,
            source_calls=source_calls,
        )


def _failed_result(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        output=message,
        success=False,
    )


class Environment:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        registrations: Dict[str, Tool] = {}
        for index, tool in enumerate(tools):
            if not isinstance(tool, Tool):
                raise TypeError(f"tools[{index}] must be Tool")
            if tool.spec.name in registrations:
                raise EnvironmentError(
                    f"duplicate tool registration: {tool.spec.name!r}"
                )
            registrations[tool.spec.name] = tool
        self._tools = registrations

    @property
    def tool_specs(self) -> Tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    def execute_tool_calls(
        self,
        calls: Sequence[ToolCall],
    ) -> EnvironmentResult:
        ordered_calls = tuple(calls)
        seen_call_ids = set()
        for index, call in enumerate(ordered_calls):
            if not isinstance(call, ToolCall):
                raise EnvironmentError(f"calls[{index}] must be ToolCall")
            if call.call_id in seen_call_ids:
                raise EnvironmentError(
                    f"duplicate tool call id: {call.call_id!r}"
                )
            seen_call_ids.add(call.call_id)

        results = []
        for call in ordered_calls:
            try:
                parsed_arguments = json.loads(call.arguments_json)
            except json.JSONDecodeError as exc:
                results.append(
                    _failed_result(
                        call,
                        f"Malformed JSON arguments for tool {call.name!r}: {exc}",
                    )
                )
                continue

            if not isinstance(parsed_arguments, Mapping):
                results.append(
                    _failed_result(
                        call,
                        f"Tool {call.name!r} arguments must decode to an object",
                    )
                )
                continue

            registration = self._tools.get(call.name)
            if registration is None:
                results.append(
                    _failed_result(call, f"Unknown tool: {call.name}")
                )
                continue

            try:
                outcome = registration.handler(
                    dict(parsed_arguments),
                    timeout_seconds=registration.timeout_seconds,
                )
                if not isinstance(outcome, ToolOutcome):
                    raise TypeError(
                        "tool handler must return ToolOutcome, got "
                        f"{type(outcome).__name__}"
                    )
            except TimeoutError as exc:
                detail = str(exc).strip()
                suffix = f": {detail}" if detail else ""
                results.append(
                    _failed_result(
                        call,
                        f"Tool {call.name!r} timed out{suffix}",
                    )
                )
                continue
            except Exception as exc:
                results.append(
                    _failed_result(
                        call,
                        f"Tool {call.name!r} failed: "
                        f"{exc.__class__.__name__}: {exc}",
                    )
                )
                continue

            results.append(
                ToolResult(
                    call_id=call.call_id,
                    output=outcome.output,
                    success=outcome.success,
                )
            )

        return EnvironmentResult(items=tuple(results))


__all__ = [
    "Environment",
    "EnvironmentError",
    "EnvironmentResult",
    "Tool",
    "ToolHandler",
    "ToolOutcome",
    "ToolSpec",
]
