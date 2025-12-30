from typing import Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
from time import sleep
from uuid import uuid4
import json
import os
import shlex

HOME = os.environ["HOME"]
TMP_DIR = os.path.join(HOME, ".pythia", "tmp")

@dataclass
class IOCommandResult:
    ret: Optional[int] = None
    out: Optional[str] = None
    err: Optional[str] = None

def exec_command(
    cmd: Union[str, list[str]],
    cwd: Optional[str] = None,
    timeout: Optional[Union[float, int]] = None,
    capture: bool = False,
) -> IOCommandResult:
    p = Popen(
        cmd,
        stdout=PIPE,
        stderr=PIPE,
        shell=False,
        cwd=cwd,
        encoding="utf-8",
        text=True,
    )
    if capture:
        try:
            out, err = p.communicate(timeout=timeout)
        except (Exception, BaseException) as e:
            p.kill()
            try:
                out, err = p.communicate()
            except (Exception, BaseException) as e2:
                out = None
                err = None
        ret = p.returncode
        return IOCommandResult(ret=ret, out=out, err=err)
    else:
        try:
            p.wait(timeout=timeout)
        except (Exception, BaseException) as e:
            p.kill()
        ret = p.returncode
        return IOCommandResult(ret=ret)

@dataclass
class ShellIOCommandController:
    def exec_command(
        self,
        cmd: Union[str, list[str]],
        cwd: Optional[str] = None,
        timeout: Optional[Union[float, int]] = None,
        capture: bool = False,
    ) -> IOCommandResult:
        if isinstance(cmd, str):
            pass
        elif isinstance(cmd, list):
            pass
        else:
            raise ValueError
        return exec_command(cmd, cwd=cwd, timeout=timeout, capture=capture)

@dataclass
class SleepTimer:
    t1: Optional[datetime] = None

    def set(self):
        self.t1 = datetime.utcnow()

    def sleep(self, min_s, max_s):
        while True:
            t = datetime.utcnow()
            ds = (t - self.t1).total_seconds()
            if ds >= min_s:
                break
            sleep(max_s - ds)
        self.t1 = t

@dataclass
class ScreenBuffer:
    prefix: str
    canary: Optional[str] = None
    output: Optional[str] = None
    suffix: str = None
    trail:  str = None

    def __post_init__(self):
        trail_end = len(self.prefix)
        end = 0
        for pos in reversed(range(trail_end)):
            if self.prefix[pos] != "\n":
                end = pos + 1
                break
        suffix_end = end
        end = 0
        for pos in reversed(range(suffix_end)):
            if self.prefix[pos] == "\n":
                end = pos + 1
                break
        prefix_end = end
        prefix = self.prefix[:prefix_end]
        if self.canary is not None:
            canary_pat = f"{self.canary}\n"
            if prefix.endswith(canary_pat):
                prefix_end -= len(canary_pat)
                prefix = self.prefix[:prefix_end]
            else:
                self.canary = None
        output_end = None
        if self.canary is not None:
            pre_canary_pat = f" ; echo {self.canary}\n"
            pre_canary_pos = prefix.rfind(pre_canary_pat)
            if pre_canary_pos >= 0:
                output_end = prefix_end
                prefix_end = pre_canary_pos + len(pre_canary_pat)
                prefix = self.prefix[:prefix_end]
        if output_end is not None:
            output = self.prefix[prefix_end:output_end]
            suffix = self.prefix[output_end:suffix_end]
        else:
            output = None
            suffix = self.prefix[prefix_end:suffix_end]
        trail = self.prefix[suffix_end:]
        self.prefix = prefix
        self.output = output
        self.suffix = suffix
        self.trail = trail

    def removeprefix(self, other: "ScreenBuffer") -> Optional[str]:
        if (
            self.prefix.startswith(other.prefix) and
            self.prefix.startswith(other.suffix, len(other.prefix))
        ):
            return self.prefix[(len(other.prefix) + len(other.suffix)):]
        else:
            return None

@dataclass
class ScreenIOCommandController:
    session: str
    session_cmd: Union[None, str, list[str]] = None
    executable: Optional[str] = None
    cmd_canary: bool = False

    _started: bool = False
    _closed: bool = False
    _canary: Optional[str] = None
    _timer: SleepTimer = None

    def __post_init__(self):
        if self._timer is None:
            self._timer = SleepTimer()

    def _fresh_canary(self):
        self._canary = f"{uuid4()}"

    def _start(self, executable: str):
        start_cmd = [
            executable,
            "-S", self.session,
            "-d", "-m",
        ]
        if isinstance(self.session_cmd, str):
            start_cmd.append(self.session_cmd)
        elif isinstance(self.session_cmd, list):
            start_cmd.extend(self.session_cmd)
        exec_command(start_cmd)
        self._timer.set()

    def _close(self, executable: str):
        close_cmd = [
            executable,
            "-S", self.session,
            "-X", "quit",
        ]
        exec_command(close_cmd)

    def close(self):
        if self._closed:
            return
        executable = self.executable
        if executable is None:
            executable = "screen"
        self._close(executable)
        self._closed = True

    def exec_command(
        self,
        cmd: Union[str, list[str]],
        cwd: Optional[str] = None,
        timeout: Optional[Union[float, int]] = None,
        capture: bool = False,
    ) -> IOCommandResult:
        if isinstance(cmd, str):
            pass
        elif isinstance(cmd, list):
            cmd = shlex.join(cmd)
        else:
            raise ValueError
        escaped_cmd_parts = []
        for c in cmd:
            if c == "\\":
                escaped_cmd_parts.append("\\\\")
            elif c == "\n":
                escaped_cmd_parts.append("\\n")
            elif c == "\r":
                escaped_cmd_parts.append("\\r")
            elif c == "$":
                escaped_cmd_parts.append("\\$")
            else:
                escaped_cmd_parts.append(c)
        escaped_cmd = "".join(escaped_cmd_parts)
        if self.cmd_canary:
            self._fresh_canary()
            escaped_cmd = escaped_cmd + f" ; echo {self._canary}\\r"
        else:
            escaped_cmd = escaped_cmd + "\\r"
        executable = self.executable
        if executable is None:
            executable = "screen"
        if not self._started:
            self._start(executable)
            self._started = True
        min_sleep = 0.0625
        max_sleep = 0.125
        capture = capture or self.cmd_canary
        # if capture:
        #     n_reps = 0
        while capture:
            dst_file = NamedTemporaryFile("w", encoding="utf-8", dir=TMP_DIR, delete=False)
            dst_path = dst_file.name
            dst_file.close()
            # print(f"DEBUG: ScreenIOCommandController.exec_command: pre dst path  = {dst_path}", flush=True)
            hardcopy_cmd = [
                executable,
                "-S", self.session,
                "-X", "hardcopy",
                "-h",
                dst_path,
            ]
            # print(f"DEBUG: ScreenIOCommandController.exec_command: pre cmd       = {hardcopy_cmd}", flush=True)
            self._timer.sleep(min_sleep, max_sleep)
            exec_command(hardcopy_cmd)
            self._timer.set()
            # print(f"DEBUG: ScreenIOCommandController.exec_command: pre result    = {pre_result}", flush=True)
            with open(dst_path, "r", encoding="utf-8") as f:
                pre_buf = f.read()
            os.remove(dst_path)
            # print(f"DEBUG: ScreenIOCommandController.exec_command: pre buf       = {pre_buf}", flush=True)
            pre_buf = ScreenBuffer(pre_buf)
            if pre_buf.suffix:
                break
            # n_reps += 1
        if capture:
            # print(f"DEBUG: ScreenIOCommandController.exec_command: pre reps      = {n_reps}", flush=True)
            # print(f"DEBUG: ScreenIOCommandController.exec_command: pre buf       = {pre_buf}", flush=True)
            pass
        stuff_cmd = [
            executable,
            "-S", self.session,
            "-X", "stuff",
            escaped_cmd,
        ]
        # TODO: sleep only on capture?
        self._timer.sleep(min_sleep, max_sleep)
        result = exec_command(stuff_cmd, cwd=cwd, timeout=timeout)
        self._timer.set()
        while capture:
            dst_file = NamedTemporaryFile("w", encoding="utf-8", dir=TMP_DIR, delete=False)
            dst_path = dst_file.name
            dst_file.close()
            # print(f"DEBUG: ScreenIOCommandController.exec_command: post dst path = {dst_path}", flush=True)
            hardcopy_cmd = [
                executable,
                "-S", self.session,
                "-X", "hardcopy",
                "-h",
                dst_path,
            ]
            # print(f"DEBUG: ScreenIOCommandController.exec_command: post cmd      = {hardcopy_cmd}", flush=True)
            self._timer.sleep(min_sleep, max_sleep)
            exec_command(hardcopy_cmd)
            self._timer.set()
            # print(f"DEBUG: ScreenIOCommandController.exec_command: post result   = {post_result}", flush=True)
            with open(dst_path, "r", encoding="utf-8") as f:
                post_buf = f.read()
            os.remove(dst_path)
            # print(f"DEBUG: ScreenIOCommandController.exec_command: post buf      = {post_buf}", flush=True)
            # TODO: quiescence condition.
            if self._canary is not None:
                post_buf = ScreenBuffer(post_buf, self._canary)
                if post_buf.canary is not None:
                    break
            else:
                post_buf = ScreenBuffer(post_buf)
                break
        if capture:
            print(f"DEBUG: ScreenIOCommandController.exec_command: post buf      = {post_buf}", flush=True)
            if post_buf.canary is not None:
                out = post_buf.output
            elif (out := post_buf.removeprefix(pre_buf)) is None:
                print(f"DEBUG: ScreenIOCommandController.exec_command: pre is NOT prefix of post", flush=True)
                out = post_buf.prefix
            return IOCommandResult(ret=result.ret, out=out)
        else:
            return IOCommandResult(ret=result.ret)

@dataclass
class SSHScreenIOCommandController:
    ssh_hostname: Optional[str] = None
    ssh_username: Optional[str] = None
    ssh_executable: Optional[str] = None
    screen_session: Optional[str] = None
    screen_executable: Optional[str] = None

    _screen_ctl: ScreenIOCommandController = None

    def __post_init__(self):
        if self._screen_ctl is None:
            hostname = self.ssh_hostname
            if hostname is None:
                # TODO: configure ssh hostname.
                hostname = "autopythia.internal"
            username = self.ssh_username
            if username is not None:
                addr = f"{username}@{hostname}"
            else:
                addr = hostname
            executable = self.ssh_executable
            if executable is None:
                executable = "ssh"
            session = self.screen_session
            if session is None:
                session = f"ssh.{uuid4()}"
            self._screen_ctl = ScreenIOCommandController(
                session=session,
                session_cmd=[executable, addr],
                executable=self.screen_executable,
                cmd_canary=True,
            )

    def close(self):
        self._screen_ctl.close()

    def exec_command(
        self,
        cmd: Union[str, list[str]],
        cwd: Optional[str] = None,
        timeout: Optional[Union[float, int]] = None,
        capture: bool = False,
    ) -> IOCommandResult:
        return self._screen_ctl.exec_command(
            cmd=cmd,
            cwd=cwd,
            timeout=timeout,
            capture=capture,
        )

@dataclass
class LogWrapIOCommandController:
    log_path: str
    wrapped: Any

    _log_file: Any = None

    def __post_init__(self):
        if self._log_file is None:
            self._log_file = open(self.log_path, "a")

    def unwrap(self) -> Any:
        return self.wrapped

    def exec_command(
        self,
        cmd: Union[str, list[str]],
        cwd: Optional[str] = None,
        timeout: Optional[Union[float, int]] = None,
        capture: bool = False,
    ) -> IOCommandResult:
        if isinstance(cmd, str):
            pass
        elif isinstance(cmd, list):
            cmd = shlex.join(cmd)
        else:
            raise ValueError
        t0 = datetime.utcnow().isoformat()
        log_item = {
            "t0": t0,
            "cmd": cmd,
            "cwd": cwd,
        }
        print(json.dumps(log_item), file=self._log_file, flush=True)
        result = self.wrapped.exec_command(
            cmd=cmd,
            cwd=cwd,
            timeout=timeout,
            capture=capture,
        )
        return result

if __name__ == "__main__":
    # ctl = ScreenIOCommandController("test-20251214")
    # ctl = SSHScreenIOCommandController(screen_session="sshtest")
    ctl = SSHScreenIOCommandController()
    # result = ctl.exec_command("echo hello")
    # result = ctl.exec_command("echo $HOME")
    # result = ctl.exec_command("echo $HOME", capture=True)
    # result = ctl.exec_command("echo \\ $HOME", capture=True)
    # result = ctl.exec_command("echo \\\r$HOME", capture=True)
    result = ctl.exec_command("echo \\\r $HOME  \\\r $HOME", capture=True)
    print(result)
    ctl.close()
