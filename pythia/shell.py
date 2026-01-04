from typing import Optional, Union
from dataclasses import dataclass, field
import os
import shlex

@dataclass
class _ShellPipelineStage:
    cmd_args: list[str]
    in_arg: Optional[str] = None
    in_hdoc: Optional[str] = None
    in_hstr: Optional[str] = None
    out_arg: Optional[str] = None
    err_arg: Optional[str] = None
    err2out: Optional[bool] = None
    pipe: Optional[bool] = None

@dataclass
class ShellPipeline:
    cmd: Union[str, list[str]]
    cmd_args: list[str] = None
    parsing_error: bool = False
    not_supported: bool = False
    stages: list[_ShellPipelineStage] = field(default_factory=list)

    def __post_init__(self):
        if self.cmd_args is None:
            if isinstance(self.cmd, str):
                # FIXME: this can fail!
                try:
                    self.cmd_args = shlex.split(self.cmd)
                except ValueError:
                    self.parsing_error = True
                    return
            elif isinstance(self.cmd, list):
                self.cmd_args = self.cmd
            else:
                raise ValueError
        stage = _ShellPipelineStage([])
        cmd_args_iter = iter(self.cmd_args)
        for arg in cmd_args_iter:
            if arg == "|":
                stage.pipe = True
                self.stages.append(stage)
                stage = _ShellPipelineStage([])
            elif arg == "<<<":
                stage.in_hstr = next(cmd_args_iter)
            elif arg.startswith("<<<"):
                stage.in_hstr = arg[3:]
            elif arg == "<<":
                stage.in_hdoc = next(cmd_args_iter)
            elif arg.startswith("<<"):
                stage.in_hdoc = arg[2:]
            elif arg.startswith("<("):
                self.not_supported = True
                return
            elif arg == "<":
                stage.in_arg = next(cmd_args_iter)
            elif arg.startswith("<"):
                stage.in_arg = arg[3:]
            elif arg == "2>&1":
                stage.err2out = True
            elif arg == "2>":
                stage.err_arg = next(cmd_args_iter)
            elif arg.startswith("2>"):
                stage.err_arg = arg[2:]
            elif arg in ("1>", ">"):
                stage.out_arg = next(cmd_args_iter)
            elif (
                arg.startswith("1>(") or
                arg.startswith(">(")
            ):
                self.not_supported = True
                return
            elif arg.startswith("1>"):
                stage.out_arg = arg[2:]
            elif arg.startswith(">"):
                stage.out_arg = arg[1:]
            else:
                stage.cmd_args.append(arg)
        if stage.cmd_args:
            self.stages.append(stage)

def detect_shell() -> Optional[str]:
    bash_version = os.environ.get("BASH_VERSINFO", None)
    zsh_version = os.environ.get("ZSH_VERSION", None)
    if bash_version is not None and zsh_version is None:
        return "bash"
    elif bash_version is None and zsh_version is not None:
        return "zsh"
    return None
