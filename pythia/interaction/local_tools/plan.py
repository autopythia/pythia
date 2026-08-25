from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Optional
from typing import Tuple

from ..environment import Tool
from ..environment import ToolOutcome
from ..environment import ToolSpec


_PLAN_STATUSES = frozenset({"pending", "in_progress", "completed"})


@dataclass(frozen=True)
class PlanStep:
    step: str
    status: str

    def __post_init__(self) -> None:
        if not isinstance(self.step, str) or not self.step.strip():
            raise ValueError("plan step must not be empty")
        if self.status not in _PLAN_STATUSES:
            raise ValueError(
                "plan status must be pending, in_progress, or completed"
            )


@dataclass(frozen=True)
class PlanState:
    explanation: Optional[str]
    plan: Tuple[PlanStep, ...]

    def __post_init__(self) -> None:
        if self.explanation is not None and not isinstance(self.explanation, str):
            raise TypeError("plan explanation must be a string or None")
        plan = tuple(self.plan)
        for index, item in enumerate(plan):
            if not isinstance(item, PlanStep):
                raise TypeError(f"plan[{index}] must be PlanStep")
        if sum(item.status == "in_progress" for item in plan) > 1:
            raise ValueError("at most one plan step can be in_progress")
        object.__setattr__(self, "plan", plan)


class PlanStore:
    def __init__(
        self,
        on_update: Optional[Callable[[PlanState], None]] = None,
    ) -> None:
        if on_update is not None and not callable(on_update):
            raise TypeError("on_update must be callable or None")
        self._on_update = on_update
        self._latest: Optional[PlanState] = None
        self._lock = Lock()
        self._update_lock = Lock()

    @property
    def latest(self) -> Optional[PlanState]:
        with self._lock:
            return self._latest

    def update(
        self,
        arguments: Mapping[str, object],
        *,
        timeout_seconds: Optional[float] = None,
    ) -> ToolOutcome:
        del timeout_seconds
        if not isinstance(arguments, Mapping):
            raise TypeError("plan arguments must be a mapping")

        raw_explanation = arguments.get("explanation")
        if raw_explanation is not None and not isinstance(raw_explanation, str):
            raise ValueError("plan explanation must be a string")

        raw_plan = arguments.get("plan")
        if not isinstance(raw_plan, list):
            raise ValueError("update_plan requires a plan array")

        steps = []
        for index, raw_step in enumerate(raw_plan):
            if not isinstance(raw_step, Mapping):
                raise ValueError(f"plan[{index}] must be an object")
            step = raw_step.get("step")
            status = raw_step.get("status")
            if not isinstance(step, str) or not step.strip():
                raise ValueError(f"plan[{index}].step must not be empty")
            if not isinstance(status, str) or status not in _PLAN_STATUSES:
                raise ValueError(
                    f"plan[{index}].status must be pending, in_progress, "
                    "or completed"
                )
            steps.append(PlanStep(step=step, status=status))

        state = PlanState(
            explanation=raw_explanation,
            plan=tuple(steps),
        )
        with self._update_lock:
            with self._lock:
                self._latest = state
            if self._on_update is not None:
                self._on_update(state)
        return ToolOutcome(output="Plan updated")


def create_update_plan_tool(
    store: PlanStore,
    *,
    timeout_seconds: Optional[float] = None,
) -> Tool:
    if not isinstance(store, PlanStore):
        raise TypeError("store must be PlanStore")
    return Tool(
        spec=ToolSpec(
            name="update_plan",
            description=(
                "Update the task plan. At most one step may be in progress."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "in_progress",
                                        "completed",
                                    ],
                                },
                            },
                            "required": ["step", "status"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["plan"],
                "additionalProperties": False,
            },
        ),
        handler=store.update,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "PlanState",
    "PlanStep",
    "PlanStore",
    "create_update_plan_tool",
]
