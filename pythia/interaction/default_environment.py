from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from pathlib import Path
from typing import Optional
from typing import Union

from .environment import Environment
from .environment import Tool
from .local_tools import CommandRuntime
from .local_tools import PlanState
from .local_tools import PlanStore
from .local_tools import create_apply_patch_tool
from .local_tools import create_exec_command_tool
from .local_tools import create_update_plan_tool
from .local_tools import create_write_stdin_tool
from .local_tools._workspace import WorkspacePolicy


# Command tools stop waiting for output after this, whatever wait the call asked
# for (up to their 60 s maximum). Claude Code 2.1.289 gives up on an MCP tool
# call 60 s after sending it unless told otherwise. The Claude relay raises that
# deadline, so this is a backstop; see doc/claude-relay-tool-deadline-plan.md.
COMMAND_WAIT_CAP_SECONDS = 59.0


class DefaultEnvironment(Environment):
    """Unsandboxed local tools, optionally extended with caller-supplied tools."""

    def __init__(
        self,
        cwd: Union[str, Path] = ".",
        *,
        shell: Optional[str] = None,
        enable_workspace: bool = True,
        enable_exec_command: bool = True,
        enable_write_stdin: bool = True,
        enable_apply_patch: bool = True,
        enable_update_plan: bool = True,
        on_plan_update: Optional[Callable[[PlanState], None]] = None,
        extra_tools: Iterable[Tool] = (),
    ) -> None:
        for field_name, value in (
            ("enable_workspace", enable_workspace),
            ("enable_exec_command", enable_exec_command),
            ("enable_write_stdin", enable_write_stdin),
            ("enable_apply_patch", enable_apply_patch),
            ("enable_update_plan", enable_update_plan),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{field_name} must be a bool")

        extra_tools = tuple(extra_tools)
        root = Path(cwd).expanduser().resolve()
        self._workspace_policy = WorkspacePolicy(enable_workspace)
        self._command_runtime = CommandRuntime(
            root,
            shell=shell,
            enable_workspace=enable_workspace,
            _workspace_policy=self._workspace_policy,
        )
        self._plan_store = PlanStore(on_update=on_plan_update)
        self._closed = False

        tools = []
        if enable_exec_command:
            tools.append(create_exec_command_tool(
                self._command_runtime, timeout_seconds=COMMAND_WAIT_CAP_SECONDS))
        if enable_write_stdin:
            tools.append(create_write_stdin_tool(
                self._command_runtime, timeout_seconds=COMMAND_WAIT_CAP_SECONDS))
        if enable_update_plan:
            tools.append(create_update_plan_tool(self._plan_store))
        if enable_apply_patch:
            tools.append(
                create_apply_patch_tool(
                    root,
                    enable_workspace=enable_workspace,
                    _workspace_policy=self._workspace_policy,
                )
            )
        try:
            super().__init__(tools=(*tools, *extra_tools))
        except Exception:
            self.close()
            raise

    @property
    def latest_plan(self) -> Optional[PlanState]:
        return self._plan_store.latest

    @property
    def command_runtime(self) -> CommandRuntime:
        return self._command_runtime

    @property
    def cwd(self) -> Path:
        """The tools' working directory (relative paths in commands start here)."""
        return self._command_runtime.cwd

    @property
    def enable_workspace(self) -> bool:
        return self._workspace_policy.enabled

    def set_enable_workspace(self, enabled: bool) -> None:
        self._workspace_policy.set_enabled(enabled)

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
