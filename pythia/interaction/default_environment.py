from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Optional
from typing import Union

from .environment import Environment
from .local_tools import CommandRuntime
from .local_tools import PlanState
from .local_tools import PlanStore
from .local_tools import create_apply_patch_tool
from .local_tools import create_exec_command_tool
from .local_tools import create_update_plan_tool
from .local_tools import create_write_stdin_tool


class DefaultEnvironment(Environment):
    """Composable, unsandboxed local tools for a caller-owned interaction."""

    def __init__(
        self,
        cwd: Union[str, Path] = ".",
        *,
        shell: Optional[str] = None,
        enable_exec_command: bool = True,
        enable_write_stdin: bool = True,
        enable_apply_patch: bool = True,
        enable_update_plan: bool = True,
        on_plan_update: Optional[Callable[[PlanState], None]] = None,
    ) -> None:
        for field_name, value in (
            ("enable_exec_command", enable_exec_command),
            ("enable_write_stdin", enable_write_stdin),
            ("enable_apply_patch", enable_apply_patch),
            ("enable_update_plan", enable_update_plan),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{field_name} must be a bool")

        root = Path(cwd).expanduser().resolve()
        self._command_runtime = CommandRuntime(root, shell=shell)
        self._plan_store = PlanStore(on_update=on_plan_update)
        self._closed = False

        tools = []
        if enable_exec_command:
            tools.append(create_exec_command_tool(self._command_runtime))
        if enable_write_stdin:
            tools.append(create_write_stdin_tool(self._command_runtime))
        if enable_update_plan:
            tools.append(create_update_plan_tool(self._plan_store))
        if enable_apply_patch:
            tools.append(create_apply_patch_tool(root))
        super().__init__(tools=tools)

    @property
    def latest_plan(self) -> Optional[PlanState]:
        return self._plan_store.latest

    @property
    def command_runtime(self) -> CommandRuntime:
        return self._command_runtime

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._command_runtime.close()

    def __enter__(self) -> "DefaultEnvironment":
        if self._closed:
            raise RuntimeError("default environment is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = [
    "DefaultEnvironment",
]
