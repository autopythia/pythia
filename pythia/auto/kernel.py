from typing import Any, Optional, TypedDict, Union
from dataclasses import dataclass, field
from tempfile import NamedTemporaryFile
import asyncio
import functools
import json
import os
import shlex
import textwrap

from pythia.api import APIServices
from pythia.auto.prompts import *
from pythia.clock import Timestamp
from pythia.contrib.atomicswap import swap as swap_paths
from pythia.experimental.extract import extract_struct
from pythia.extract import (
    Message,
    MarkdownCodeBlock,
)
from pythia.io_control.command import (
    ShellIOCommandController,
)
from pythia.term_utils import *

HOME = os.environ["HOME"]
GLOBAL_DIR = os.path.join(HOME, ".pythia", "auto")
GLOBAL_SESSION_DIR = os.path.join(GLOBAL_DIR, "session")

def fresh_ctr(prefix: str, key: str = "ctr") -> int:
    ctr_path = os.path.join(prefix, "ctr.json")
    while True:
        try:
            ctr_file = open(ctr_path, "r", encoding="utf-8")
        except OSError:
            os.makedirs(prefix, exist_ok=True)
            init_ctr_item = {
                key: 0,
            }
            with open(ctr_path, "w", encoding="utf-8") as file:
                print(json.dumps(init_ctr_item), file=file, flush=True)
            ctr_file = open(ctr_path, "r", encoding="utf-8")
        ctr_item = json.loads(ctr_file.read().rstrip())
        ctr_file.close()
        ctr_item[key] += 1
        tmp_file = NamedTemporaryFile("w", encoding="utf-8", dir=GLOBAL_DIR, delete=False)
        tmp_path = tmp_file.name
        print(json.dumps(ctr_item), file=tmp_file, flush=True)
        tmp_file.close()
        swapped = False
        try:
            swap_paths(ctr_path, tmp_path)
            swapped = True
        except OSError:
            pass
        os.remove(tmp_path)
        if swapped:
            return ctr_item[key]

def get_ctr(prefix: str, key: str = "ctr") -> int:
    ctr_path = os.path.join(prefix, "ctr.json")
    while True:
        try:
            ctr_file = open(ctr_path, "r", encoding="utf-8")
        except OSError:
            os.makedirs(prefix, exist_ok=True)
            init_ctr_item = {
                key: 0,
            }
            with open(ctr_path, "w", encoding="utf-8") as file:
                print(json.dumps(init_ctr_item), file=file, flush=True)
            ctr_file = open(ctr_path, "r", encoding="utf-8")
        ctr_item = json.loads(ctr_file.read().rstrip())
        ctr_file.close()
        return ctr_item[key]

class _ReturnTailAwait(Exception):
    def __init__(self, fun, args, kwargs):
        self.fun = fun
        self.args = args
        self.kwargs = kwargs

def tail_await(fun, *args, **kwargs):
    raise _ReturnTailAwait(fun, args, kwargs)

def async_tail(fun):
    @functools.wraps(fun)
    async def wrapped_fun(*args, **kwargs):
        fun_ = fun
        while True:
            try:
                return await fun_(*args, **kwargs)
            except _ReturnTailAwait as ret:
                fun_ = ret.fun
                if fun_ is wrapped_fun:
                    fun_ = fun
                args = ret.args
                kwargs = ret.kwargs
                continue
    return wrapped_fun

@dataclass
class StartControlEvent:
    step_ctr: int

@dataclass
class EndControlEvent:
    step_ctr: int

@dataclass
class OutputEvent:
    pass

    def __str__(self) -> str:
        raise NotImplementedError

    def leaf_type(self) -> str:
        raise NotImplementedError

    def leaf_events(self):
        raise NotImplementedError

@dataclass
class AtomicOutputEvent(OutputEvent):
    events: list[OutputEvent] = field(default_factory=list)

    def __str__(self) -> str:
        return "\n\n".join([f"{event}" for event in self.events])

    def leaf_events(self):
        for event in self.events:
            yield from event.leaf_events()

    def append(self, event: OutputEvent):
        self.events.append(event)

@dataclass
class BasicOutputEvent(OutputEvent):
    text: str

    def __str__(self) -> str:
        return self.text

    def leaf_type(self) -> str:
        return "basic"

    def leaf_events(self):
        yield self

@dataclass
class ThinkingOutputEvent(OutputEvent):
    text: str

    def __str__(self) -> str:
        return f"""{dim("<think>", bold=True)}\n{dim(self.text)}\n{dim("</think>", bold=True)}"""

    def leaf_type(self) -> str:
        return "thinking"

    def leaf_events(self):
        yield self

@dataclass
class AnswerOutputEvent(OutputEvent):
    text: str

    def __str__(self) -> str:
        return self.text

    def leaf_type(self) -> str:
        return "answer"

    def leaf_events(self):
        yield self

async def echo_event(event):
    return event

class SafeStruct(TypedDict):
    safe: Optional[bool]
    # safe: bool

@dataclass
class _ShellPipelineStage:
    cmd_args: list[str]
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

@dataclass
class ShellExecResult:
    cmd: str
    pipeline: ShellPipeline
    allowed: bool
    partial: bool
    final_output: Optional[str]

@dataclass
class Autopythia:
    working_model:  str = "deepseek-ai/deepseek-v3.2-thinking-off"
    thinking_model: str = "deepseek-ai/deepseek-v3.2-thinking"
    services: APIServices = None

    _session: Optional[str] = None
    _workqueue: Any = None

    def __post_init__(self):
        if self.services is None:
            self.services = APIServices(enable_journal=False)
        if self._workqueue is None:
            self._workqueue = set()

    def _fresh_session_ctr(self) -> str:
        ctr = fresh_ctr(GLOBAL_SESSION_DIR, "session_ctr")
        return f"{ctr}"

    def _get_session_ctr(self) -> str:
        ctr = get_ctr(GLOBAL_SESSION_DIR, "session_ctr")
        return f"{ctr}"

    def _fresh_step_ctr(self, session_ctr: str) -> str:
        ctr = fresh_ctr(os.path.join(GLOBAL_SESSION_DIR, session_ctr), "step_ctr")
        return f"{ctr}"

    def _set_session(self, session_ctr: str):
        self._session = session_ctr

    def append_history(self, session_ctr: str, step_ctr: str, query: str, t0=None):
        if t0 is None:
            t0 = Timestamp()
        prefix = os.path.join(GLOBAL_SESSION_DIR, session_ctr)
        history_path = os.path.join(prefix, "history.jsonl")
        try:
            history_file = open(history_path, "a", encoding="utf-8")
        except OSError:
            os.makedirs(prefix, exist_ok=True)
            history_file = open(history_path, "a", encoding="utf-8")
        history_item = {
            "t0": f"{t0}",
            "session_ctr": session_ctr,
            "session_uid": None,
            "step_ctr": step_ctr,
            "step_uid": None,
            "query": query,
        }
        print(json.dumps(history_item), file=history_file, flush=True)
        history_file.close()
        return t0

    async def init(self, step_ctr: str, query: str):
        model_path = self.working_model
        model = self.services.registry.find_model(model_path)
        sampling_params = {
            "max_tokens": 8192,
            # "max_tokens": 16384,
            # "max_tokens": 65536,
            # "temperature": 0.6,
            "temperature": 1.0,
        }

        think_model_path = self.thinking_model
        think_model = self.services.registry.find_model(think_model_path)
        think_sampling_params = {
            # "max_tokens": 8192,
            # "max_tokens": 16384,
            # "max_tokens": 32768,
            "max_tokens": 65536,
            # "temperature": 0.6,
            "temperature": 1.0,
        }

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)

        event = StartControlEvent(step_ctr)
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        formatted_query = textwrap.indent(query, "    ")
        plan_query = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": INIT_PROMPT.format(query=formatted_query),
            },
        ]
        # print(plan_query)
        event = BasicOutputEvent(green("Thinking...", bold=True))
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        t0 = Timestamp()
        plan_result = await self.services.client.message(
            think_model,
            plan_query,
            think_sampling_params,
            fresh=True,
        )
        t1 = Timestamp()
        # print(plan_result)

        res_event = AtomicOutputEvent()
        event = BasicOutputEvent(green(f"Thought for {(t1 - t0).pretty_format()}", bold=True))
        res_event.append(event)

        message = plan_result.message()
        thinking_part = Message.get_thinking_part(message)
        answer = Message.get_text(message)

        if thinking_part:
            # print(f"""<think>\n{thinking_part["thinking"]}\n</think>\n""")
            event = ThinkingOutputEvent(thinking_part["thinking"])
            res_event.append(event)
        # print(answer)
        event = AnswerOutputEvent(answer)
        res_event.append(event)
        self._workqueue.add(asyncio.create_task(echo_event(res_event)))

        event = EndControlEvent(step_ctr)
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        # TODO

        block_start = None
        plan_block = None
        while True:
            plan_block = MarkdownCodeBlock.extract_next(answer, block_start)
            if plan_block is None:
                # print(f"DEBUG: init: no initial code block", flush=True)
                break
            elif plan_block["lang"] == "markdown":
                break
            block_start = plan_block["end"]

        plan = None
        if plan_block is not None:
            # print(f"DEBUG: init: initial code block: {plan_block}", flush=True)
            plan = plan_block["text"].rstrip()

        if not plan:
            return await self.init(None, query)
        else:
            new_results = self._parse_shell_commands(answer)
            results = new_results

            return await self.eval(None, query, plan, None, results)

    async def eval(self, step_ctr: Optional[str], query: str, plan: str, scratch: Optional[str], results: list = []):
        model_path = self.working_model
        model = self.services.registry.find_model(model_path)
        sampling_params = {
            "max_tokens": 8192,
            # "max_tokens": 16384,
            # "max_tokens": 65536,
            # "temperature": 0.6,
            "temperature": 1.0,
        }

        think_model_path = self.thinking_model
        think_model = self.services.registry.find_model(think_model_path)
        think_sampling_params = {
            # "max_tokens": 8192,
            # "max_tokens": 16384,
            # "max_tokens": 32768,
            "max_tokens": 65536,
            # "temperature": 0.6,
            "temperature": 1.0,
        }

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)

        event = StartControlEvent(step_ctr)
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        formatted_query = textwrap.indent(query, "    ")
        formatted_plan = plan

        if results:
            formatted_results_parts = []
            for result in results:
                block = (
    f"""Command: `{result.cmd}`
    Not fully executed? {result.partial}
    Output: {result.final_output}"""
                )
                # print(block)
                formatted_results_parts.append(block)
            # print(formatted_results_parts)
            formatted_results = "\n\n".join(formatted_results_parts)

            eval_prompt = EVAL_PROMPT.format(
                query=formatted_query,
                plan=formatted_plan,
                results=formatted_results,
            )

        else:
            eval_prompt = EVAL_PROMPT_0.format(
                query=formatted_query,
                plan=formatted_plan,
            )

        eval_query = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": eval_prompt,
            },
        ]
        # print(eval_query)
        event = BasicOutputEvent(green("Thinking...", bold=True))
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        t0 = Timestamp()
        eval_result = await self.services.client.message(
            think_model,
            eval_query,
            think_sampling_params,
            fresh=True,
        )
        t1 = Timestamp()
        # print(eval_result)

        res_event = AtomicOutputEvent()
        event = BasicOutputEvent(green(f"Thought for {(t1 - t0).pretty_format()}", bold=True))
        res_event.append(event)

        message = eval_result.message()
        thinking_part = Message.get_thinking_part(message)
        answer = Message.get_text(message)

        if thinking_part:
            # print(f"""<think>\n{thinking_part["thinking"]}\n</think>\n""")
            event = ThinkingOutputEvent(thinking_part["thinking"])
            res_event.append(event)
        # print(answer)
        event = AnswerOutputEvent(answer)
        res_event.append(event)
        self._workqueue.add(asyncio.create_task(echo_event(res_event)))

        event = EndControlEvent(step_ctr)
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        # TODO

        new_results = self._parse_shell_commands(answer)
        results.extend(new_results)

        if not new_results:
            return await self.eval(None, query, plan, None, results)
        else:
            return await self.backup(None, query, plan, None, results)

    async def backup(self, step_ctr, query: str, plan: str, scratch: Optional[str], results = []):
        model_path = self.working_model
        model = self.services.registry.find_model(model_path)
        sampling_params = {
            "max_tokens": 8192,
            # "max_tokens": 16384,
            # "max_tokens": 65536,
            # "temperature": 0.6,
            "temperature": 1.0,
        }

        think_model_path = self.thinking_model
        think_model = self.services.registry.find_model(think_model_path)
        think_sampling_params = {
            # "max_tokens": 8192,
            # "max_tokens": 16384,
            # "max_tokens": 32768,
            "max_tokens": 65536,
            # "temperature": 0.6,
            "temperature": 1.0,
        }

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)

        event = StartControlEvent(step_ctr)
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        formatted_query = textwrap.indent(query, "    ")
        formatted_plan = plan

        formatted_results_parts = []
        for result in results:
            block = (
f"""Command: `{result.cmd}`
Not fully executed? {result.partial}
Output: {result.final_output}"""
            )
            # print(block)
            formatted_results_parts.append(block)
        # print(formatted_results_parts)
        formatted_results = "\n\n".join(formatted_results_parts)

        backup_query = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": BACKUP_PROMPT.format(
                    query=formatted_query,
                    plan=formatted_plan,
                    results=formatted_results,
                ),
            },
        ]
        # print(backup_query)
        event = BasicOutputEvent(green("Thinking...", bold=True))
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        t0 = Timestamp()
        backup_result = await self.services.client.message(
            think_model,
            backup_query,
            think_sampling_params,
            fresh=True,
        )
        t1 = Timestamp()
        # print(backup_result)

        res_event = AtomicOutputEvent()
        event = BasicOutputEvent(green(f"Thought for {(t1 - t0).pretty_format()}", bold=True))
        res_event.append(event)

        message = backup_result.message()
        thinking_part = Message.get_thinking_part(message)
        answer = Message.get_text(message)

        if thinking_part:
            # print(f"""<think>\n{thinking_part["thinking"]}\n</think>\n""")
            event = ThinkingOutputEvent(thinking_part["thinking"])
            res_event.append(event)
        # print(answer)
        event = AnswerOutputEvent(answer)
        res_event.append(event)
        self._workqueue.add(asyncio.create_task(echo_event(res_event)))

        event = EndControlEvent(step_ctr)
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        # TODO

        new_results = self._parse_shell_commands(answer)
        results.extend(new_results)

        # TODO

        write_plan = answer.find("/plan") >= 0
        block_start = None
        plan_block = None
        while write_plan:
            plan_block = MarkdownCodeBlock.extract_next(answer, block_start)
            if plan_block is None:
                # print(f"DEBUG: init: no initial code block", flush=True)
                break
            elif plan_block["lang"] == "markdown":
                break
            block_start = plan_block["end"]

        # new_plan = None
        if plan_block is not None:
            # print(f"DEBUG: init: initial code block: {plan_block}", flush=True)
            plan = plan_block["text"].rstrip()
            # new_plan = plan

        if not new_results:
            return await self.eval(None, query, plan, None, results)
        else:
            return await self.backup(None, query, plan, None, results)

    def _parse_shell_commands(self, answer):
        block_start = None
        code_blocks = []
        while True:
            code_block = MarkdownCodeBlock.extract_next(answer, block_start)
            if code_block is None:
                break
            elif code_block["lang"] in ("sh", "bash", "zsh"):
                code_blocks.append(code_block)
            block_start = code_block["end"]
        print(f"DEBUG: code blocks = {code_blocks} ...", flush=True)

        allow_cmds = [
            ("ls",),
            ("cat",),
            ("head",),
            ("find",),
            ("grep",),
            # ("rg",),
            ("git", "log"),
            ("git", "show"),
            ("git", "status"),
        ]

        new_results = []
        cmd_control = ShellIOCommandController()
        for code in code_blocks:
            for cmd_line in code["text"].splitlines():
                cmd_line = cmd_line.strip()
                if not cmd_line:
                    continue
                elif cmd_line.startswith("#"):
                    continue
                # cmd_args = shlex.split(cmd_line)
                cmd_pipeline = ShellPipeline(cmd_line)
                if cmd_pipeline.parsing_error or cmd_pipeline.not_supported:
                    # TODO: report error here.
                    continue
                cmd_results = []
                allowed = False
                capture = True
                for cmd_stage in cmd_pipeline.stages:
                    cmd_args = cmd_stage.cmd_args
                    print(f"DEBUG: cmd args = {cmd_args} ...", flush=True)
                    allowed = False
                    for allow_cmd_args in allow_cmds:
                        if (
                            len(cmd_args) >= len(allow_cmd_args) and
                            tuple(cmd_args[:len(allow_cmd_args)]) == allow_cmd_args
                        ):
                            allowed = True
                            break
                    if not allowed:
                        break
                    result = cmd_control.exec_command(cmd_args, capture=capture)
                    cmd_results.append(result)
                    # if cmd_stage.pipe:
                    capture = result.out
                    print(f"DEBUG: cmd args = {cmd_args} output = {repr(result.out)}", flush=True)
                if isinstance(capture, str):
                    result = ShellExecResult(
                        cmd_line,
                        cmd_pipeline,
                        allowed,
                        not allowed,
                        capture,
                    )
                else:
                    result = ShellExecResult(
                        cmd_line,
                        cmd_pipeline,
                        allowed,
                        not allowed,
                        None,
                    )
                new_results.append(result)
        # print(new_results)
        return new_results

if __name__ == "__main__":
    pass
