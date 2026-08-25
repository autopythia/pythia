from __future__ import annotations

import os
import select
import signal
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from threading import Lock
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

from ..environment import Tool
from ..environment import ToolOutcome
from ..environment import ToolSpec


@dataclass
class _CommandSession:
    session_id: int
    process: subprocess.Popen
    started_at: float
    cwd: Path
    cmd: str
    lock: Lock = field(default_factory=Lock)


def _require_nonnegative_integer(
    value: object,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


def _optional_positive_integer(
    value: object,
    field_name: str,
) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer or null")
    return value


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _truncate_output(text: str, max_tokens: Optional[int]) -> str:
    if max_tokens is None:
        return text
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    marker = "\n... output truncated ...\n"
    available = max(0, max_chars - len(marker))
    prefix_chars = available // 2
    suffix_chars = available - prefix_chars
    prefix = text[:prefix_chars]
    suffix = text[-suffix_chars:] if suffix_chars else ""
    return f"{prefix}{marker}{suffix}"


def _nonempty_environment_value(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def _detect_default_shell() -> str:
    bash_detected = any(
        _nonempty_environment_value(name) is not None
        for name in ("BASH_VERSION", "BASH_VERSINFO")
    )
    zsh_detected = _nonempty_environment_value("ZSH_VERSION") is not None

    if bash_detected and not zsh_detected:
        bash_path = _nonempty_environment_value("BASH")
        if bash_path is not None:
            return bash_path
        discovered_bash = shutil.which("bash")
        if discovered_bash is not None:
            return discovered_bash

    if zsh_detected and not bash_detected:
        discovered_zsh = shutil.which("zsh")
        if discovered_zsh is not None:
            return discovered_zsh

    configured_shell = _nonempty_environment_value("SHELL")
    if configured_shell is not None:
        return configured_shell

    return shutil.which("bash") or shutil.which("sh") or "/bin/sh"


class CommandRuntime:
    """Owns unsandboxed local subprocesses shared by command tools."""

    def __init__(
        self,
        cwd: Union[str, Path] = ".",
        *,
        shell: Optional[str] = None,
        default_exec_yield_time_ms: int = 10_000,
        default_write_yield_time_ms: int = 250,
        default_max_output_tokens: Optional[int] = 4_000,
        max_yield_time_ms: int = 60_000,
    ) -> None:
        root = Path(cwd).expanduser().resolve()
        if not root.exists():
            raise ValueError(f"cwd does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"cwd is not a directory: {root}")

        if shell is not None and (
            not isinstance(shell, str) or not shell.strip()
        ):
            raise ValueError("shell must not be empty")
        selected_shell = shell if shell is not None else _detect_default_shell()
        resolved_shell = shutil.which(selected_shell)
        if resolved_shell is None:
            raise ValueError(f"shell is not executable: {selected_shell}")

        self.cwd = root
        self.shell = resolved_shell
        self.default_exec_yield_time_ms = _require_nonnegative_integer(
            default_exec_yield_time_ms,
            "default_exec_yield_time_ms",
        )
        self.default_write_yield_time_ms = _require_nonnegative_integer(
            default_write_yield_time_ms,
            "default_write_yield_time_ms",
        )
        self.default_max_output_tokens = _optional_positive_integer(
            default_max_output_tokens,
            "default_max_output_tokens",
        )
        self.max_yield_time_ms = _require_nonnegative_integer(
            max_yield_time_ms,
            "max_yield_time_ms",
        )
        if self.default_exec_yield_time_ms > self.max_yield_time_ms:
            raise ValueError(
                "default_exec_yield_time_ms must not exceed max_yield_time_ms"
            )
        if self.default_write_yield_time_ms > self.max_yield_time_ms:
            raise ValueError(
                "default_write_yield_time_ms must not exceed "
                "max_yield_time_ms"
            )

        self._sessions: Dict[int, _CommandSession] = {}
        self._next_session_id = 1
        self._next_chunk_id = 1
        self._lock = Lock()
        self._closed = False

    @property
    def active_session_ids(self) -> Tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._sessions))

    def _ensure_open(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("command runtime is closed")

    def _resolve_workdir(self, value: object) -> Path:
        if value is None or value == "":
            return self.cwd
        if not isinstance(value, str):
            raise ValueError("workdir must be a string")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.cwd / candidate
        resolved = candidate.resolve()
        if resolved != self.cwd and self.cwd not in resolved.parents:
            raise ValueError(f"workdir escapes configured cwd: {value}")
        if not resolved.is_dir():
            raise ValueError(f"workdir is not a directory: {value}")
        return resolved

    def _next_chunk(self) -> str:
        with self._lock:
            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1
        return str(chunk_id)

    def _next_session(self, process: subprocess.Popen, cwd: Path, cmd: str) -> int:
        with self._lock:
            if self._closed:
                self._terminate_process(process)
                raise RuntimeError("command runtime is closed")
            session_id = self._next_session_id
            self._next_session_id += 1
            self._sessions[session_id] = _CommandSession(
                session_id=session_id,
                process=process,
                started_at=time.monotonic(),
                cwd=cwd,
                cmd=cmd,
            )
            return session_id

    def _wait_seconds(
        self,
        arguments: Mapping[str, object],
        timeout_seconds: Optional[float],
        *,
        default_yield_time_ms: int,
    ) -> float:
        raw_yield = arguments.get("yield_time_ms", default_yield_time_ms)
        yield_time_ms = _require_nonnegative_integer(
            raw_yield,
            "yield_time_ms",
        )
        if yield_time_ms > self.max_yield_time_ms:
            raise ValueError(
                f"yield_time_ms must not exceed {self.max_yield_time_ms}"
            )
        wait_seconds = yield_time_ms / 1000.0
        if timeout_seconds is not None:
            wait_seconds = min(wait_seconds, timeout_seconds)
        return wait_seconds

    def _max_output_tokens(
        self,
        arguments: Mapping[str, object],
    ) -> Optional[int]:
        return _optional_positive_integer(
            arguments.get(
                "max_output_tokens",
                self.default_max_output_tokens,
            ),
            "max_output_tokens",
        )

    def exec_command(
        self,
        arguments: Mapping[str, object],
        *,
        timeout_seconds: Optional[float] = None,
    ) -> ToolOutcome:
        self._ensure_open()
        if not isinstance(arguments, Mapping):
            raise TypeError("exec_command arguments must be a mapping")

        cmd = arguments.get("cmd")
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("exec_command requires a non-empty cmd")
        raw_login = arguments.get("login", True)
        if not isinstance(raw_login, bool):
            raise ValueError("login must be a bool")

        workdir = self._resolve_workdir(arguments.get("workdir"))
        wait_seconds = self._wait_seconds(
            arguments,
            timeout_seconds,
            default_yield_time_ms=self.default_exec_yield_time_ms,
        )
        max_output_tokens = self._max_output_tokens(arguments)
        argv = [self.shell, "-lc" if raw_login else "-c", cmd]

        started_at = time.monotonic()
        process = subprocess.Popen(
            argv,
            cwd=str(workdir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            start_new_session=os.name == "posix",
        )
        if process.stdout is not None:
            os.set_blocking(process.stdout.fileno(), False)

        output = self._read_process_output(process, wait_seconds)
        exit_code = process.poll()
        session_id = None
        if exit_code is None:
            session_id = self._next_session(process, workdir, cmd)
        else:
            self._close_process_pipes(process)

        return ToolOutcome(
            output=self._format_response(
                chunk_id=self._next_chunk(),
                wall_time_seconds=time.monotonic() - started_at,
                exit_code=exit_code,
                session_id=session_id,
                output=output,
                max_output_tokens=max_output_tokens,
            ),
            success=True,
        )

    def write_stdin(
        self,
        arguments: Mapping[str, object],
        *,
        timeout_seconds: Optional[float] = None,
    ) -> ToolOutcome:
        self._ensure_open()
        if not isinstance(arguments, Mapping):
            raise TypeError("write_stdin arguments must be a mapping")

        raw_session_id = arguments.get("session_id")
        if (
            isinstance(raw_session_id, bool)
            or not isinstance(raw_session_id, int)
            or raw_session_id <= 0
        ):
            raise ValueError("write_stdin requires a positive session_id")
        chars = arguments.get("chars", "")
        if not isinstance(chars, str):
            raise ValueError("chars must be a string")

        wait_seconds = self._wait_seconds(
            arguments,
            timeout_seconds,
            default_yield_time_ms=self.default_write_yield_time_ms,
        )
        max_output_tokens = self._max_output_tokens(arguments)
        with self._lock:
            session = self._sessions.get(raw_session_id)
        if session is None:
            raise ValueError(f"unknown session_id: {raw_session_id}")

        started_at = time.monotonic()
        process = session.process
        with session.lock:
            if chars and process.poll() is None:
                if process.stdin is None:
                    raise ValueError("process stdin is unavailable")
                try:
                    process.stdin.write(chars.encode("utf-8"))
                    process.stdin.flush()
                except BrokenPipeError as exc:
                    raise ValueError("process stdin is closed") from exc

            output = self._read_process_output(process, wait_seconds)
            exit_code = process.poll()
            session_id = raw_session_id
            if exit_code is not None:
                with self._lock:
                    self._sessions.pop(raw_session_id, None)
                self._close_process_pipes(process)
                session_id = None

        return ToolOutcome(
            output=self._format_response(
                chunk_id=self._next_chunk(),
                wall_time_seconds=time.monotonic() - started_at,
                exit_code=exit_code,
                session_id=session_id,
                output=output,
                max_output_tokens=max_output_tokens,
            ),
            success=True,
        )

    def _read_process_output(
        self,
        process: subprocess.Popen,
        timeout_seconds: float,
    ) -> str:
        if process.stdout is None:
            return ""
        fd = process.stdout.fileno()
        chunks = []
        deadline = time.monotonic() + max(0.0, timeout_seconds)

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                break
            try:
                data = os.read(fd, 65_536)
            except BlockingIOError:
                continue
            if not data:
                break
            chunks.append(data)

        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            try:
                data = os.read(fd, 65_536)
            except BlockingIOError:
                break
            if not data:
                break
            chunks.append(data)

        return b"".join(chunks).decode("utf-8", errors="replace")

    def _format_response(
        self,
        *,
        chunk_id: str,
        wall_time_seconds: float,
        exit_code: Optional[int],
        session_id: Optional[int],
        output: str,
        max_output_tokens: Optional[int],
    ) -> str:
        rendered_output = _truncate_output(output, max_output_tokens)
        sections = [
            f"Chunk ID: {chunk_id}",
            f"Wall time: {wall_time_seconds:.4f} seconds",
        ]
        if exit_code is not None:
            sections.append(f"Process exited with code {exit_code}")
        if session_id is not None:
            sections.append(f"Process running with session ID {session_id}")
        sections.extend(
            (
                f"Original token count: {_approx_token_count(output)}",
                "Output:",
                rendered_output,
            )
        )
        return "\n".join(sections)

    def _close_process_pipes(self, process: subprocess.Popen) -> None:
        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, stream_name, None)
            if stream is None:
                continue
            try:
                stream.close()
            except OSError:
                pass

    def _terminate_process(self, process: subprocess.Popen) -> None:
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=1.0)
                except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            except (OSError, ProcessLookupError):
                pass
        self._close_process_pipes(process)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            with session.lock:
                self._terminate_process(session.process)

    def __enter__(self) -> "CommandRuntime":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()


def create_exec_command_tool(
    runtime: CommandRuntime,
    *,
    timeout_seconds: Optional[float] = None,
) -> Tool:
    if not isinstance(runtime, CommandRuntime):
        raise TypeError("runtime must be CommandRuntime")
    return Tool(
        spec=ToolSpec(
            name="exec_command",
            description=(
                "Run a shell command with pipes, returning output or a "
                "session ID for an ongoing process."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "workdir": {
                        "type": "string",
                        "description": (
                            "Working directory under the configured root."
                        ),
                    },
                    "yield_time_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "How long to wait for output. Defaults to 10000."
                        ),
                    },
                    "max_output_tokens": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Approximate returned-output limit. Defaults to "
                            "4000."
                        ),
                    },
                    "login": {
                        "type": "boolean",
                        "description": (
                            "Run the selected shell as a login shell. "
                            "Defaults to true."
                        ),
                    },
                },
                "required": ["cmd"],
                "additionalProperties": False,
            },
        ),
        handler=runtime.exec_command,
        timeout_seconds=timeout_seconds,
    )


def create_write_stdin_tool(
    runtime: CommandRuntime,
    *,
    timeout_seconds: Optional[float] = None,
) -> Tool:
    if not isinstance(runtime, CommandRuntime):
        raise TypeError("runtime must be CommandRuntime")
    return Tool(
        spec=ToolSpec(
            name="write_stdin",
            description=(
                "Write text to an existing command session, or poll it for "
                "recent output."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "Running command-session identifier.",
                    },
                    "chars": {
                        "type": "string",
                        "description": (
                            "Text to write, or an empty string to poll."
                        ),
                    },
                    "yield_time_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "How long to wait for output. Defaults to 250."
                        ),
                    },
                    "max_output_tokens": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Approximate returned-output limit. Defaults to "
                            "4000."
                        ),
                    },
                },
                "required": ["session_id"],
                "additionalProperties": False,
            },
        ),
        handler=runtime.write_stdin,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "CommandRuntime",
    "create_exec_command_tool",
    "create_write_stdin_tool",
]
