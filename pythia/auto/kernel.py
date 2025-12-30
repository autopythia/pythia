from typing import Any, Optional, TypedDict
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

class SafeStruct(TypedDict):
    safe: Optional[bool]
    # safe: bool

async def echo_event(event):
    return event

@dataclass
class EndControlEvent:
    pass

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

@dataclass
class Autopythia:
    working_model:  str = "deepseek-ai/deepseek-v3.2-thinking-off"
    thinking_model: str = "deepseek-ai/deepseek-v3.2-thinking"
    services: APIServices = None

    _workqueue: Any = None

    def __post_init__(self):
        if self.services is None:
            self.services = APIServices(enable_journal=False)
        if self._workqueue is None:
            self._workqueue = set()

    def fresh_session_ctr(self) -> str:
        ctr = fresh_ctr(GLOBAL_SESSION_DIR, "session_ctr")
        return f"{ctr}"

    def fresh_query_ctr(self, session_ctr: str) -> str:
        ctr = fresh_ctr(os.path.join(GLOBAL_SESSION_DIR, session_ctr), "query_ctr")
        return f"{ctr}"

    def append_history(self, session_ctr: str, query_ctr: str, query: str, t0=None):
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
            "query_ctr": query_ctr,
            "query_uid": None,
            "query": query,
        }
        print(json.dumps(history_item), file=history_file, flush=True)
        history_file.close()
        return t0

    async def init(self, query: str):
        model_path = self.working_model
        model = self.services.registry.find_model(model_path)
        sampling_params = {
            "max_tokens": 8192,
            # "max_tokens": 16384,
            # "max_tokens": 65536,
            "temperature": 0.6,
            # "temperature": 1.0,
        }

        think_model_path = self.thinking_model
        think_model = self.services.registry.find_model(think_model_path)
        think_sampling_params = {
            # "max_tokens": 8192,
            # "max_tokens": 16384,
            # "max_tokens": 32768,
            "max_tokens": 65536,
            "temperature": 0.6,
            # "temperature": 1.0,
        }

        formatted_query = textwrap.indent(query, "    ")
        plan_query = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": PLAN_INIT_PROMPT.format(query=formatted_query),
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

        event = EndControlEvent()
        self._workqueue.add(asyncio.create_task(echo_event(event)))

    async def run(self, query):
        model_path = self.working_model
        model = self.services.registry.find_model(model_path)
        sampling_params = {
            "max_tokens": 8192,
            # "max_tokens": 16384,
            # "max_tokens": 65536,
            "temperature": 0.6,
            # "temperature": 1.0,
        }

        think_model_path = self.thinking_model
        think_model = self.services.registry.find_model(think_model_path)
        think_sampling_params = {
            # "max_tokens": 8192,
            # "max_tokens": 16384,
            "max_tokens": 32768,
            # "max_tokens": 65536,
            "temperature": 0.6,
            # "temperature": 1.0,
        }

        user_query = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
        ] + query
        # print(query[0]["content"])
        # if len(query) > 1:
        print(user_query[-1]["content"])
        result = await self.services.client.message(
            model,
            user_query,
            sampling_params,
            fresh=True,
        )
        message = result.message()
        thinking = Message.get_thinking_part(message)
        response = Message.get_text(message)
        print(response)

        block = MarkdownCodeBlock.extract_next(response)
        if block is None:
            return

        commands = block["text"].rstrip()
        return await self.run_commands(user_query, commands)

    @async_tail
    async def run_commands(self, user_query, commands, depth=0):
        model_path = self.working_model
        model = self.services.registry.find_model(model_path)
        sampling_params = {
            "max_tokens": 8192,
            # "max_tokens": 16384,
            # "max_tokens": 65536,
            "temperature": 0.6,
            # "temperature": 1.0,
        }

        think_model_path = self.thinking_model
        think_model = self.services.registry.find_model(think_model_path)
        think_sampling_params = {
            # "max_tokens": 8192,
            # "max_tokens": 16384,
            "max_tokens": 32768,
            # "max_tokens": 65536,
            "temperature": 0.6,
            # "temperature": 1.0,
        }

        # TODO: (Command(s) are unsafe if, for example, they could lead to irreversible data loss, privilege escalation.)

        safe_query = [
            {
                "role": "user",
                "content": SAFE_PROMPT.format(commands=commands),
            },
        ]
        print(safe_query[-1]["content"])
        safe_max_quorum = 5
        safe_min_quorum = 5
        safe_quorum = 0
        results_work = []
        for _ in range(safe_max_quorum):
            w = self.services.client.message(
                model,
                safe_query,
                sampling_params,
                fresh=True,
            )
            results_work.append(w)
        results = await asyncio.gather(*results_work)
        for result in results:
            print(result)
            message = result.message()
            # thinking = Message.get_thinking_part(message)
            response = Message.get_text(message)
            print(response)
            safe_struct = extract_struct(response, SafeStruct)
            print(safe_struct)
            if safe_struct["safe"]:
                safe_quorum += 1
        if safe_quorum < safe_min_quorum:
            return

        allow_cmds = [
            ("ls",),
            ("cat",),
            ("head",),
            ("find",),
            ("grep",),
            ("rg",),
            ("git", "log"),
            ("git", "show"),
            ("git", "status"),
        ]

        cmd_control = ShellIOCommandController()
        cmd_results = []
        for command_line in commands.splitlines():
            cmd_line = command_line.strip()
            if not cmd_line:
                continue
            if cmd_line.startswith("#"):
                continue
            cmd_args = shlex.split(cmd_line)
            print(cmd_args)
            allow = False
            for allow_cmd_args in allow_cmds:
                if (
                    len(cmd_args) >= len(allow_cmd_args) and
                    tuple(cmd_args[:len(allow_cmd_args)]) == allow_cmd_args
                ):
                    allow = True
                    break
            override_allow = False
            # if cmd_args[0] not in ("ls", "cat", "head", "find"):
            if not allow:
                for _ in range(3):
                    print(f"auto: {repr(cmd_args[0])} command is not whitelisted, allow? [y/n]", flush=True)
                    i = input()
                    if i.lower() in ("n", "no"):
                        break
                    elif i.lower() in ("y", "yes"):
                        override_allow = True
                        break
            if not (allow or override_allow):
                print("auto: {repr(cmd_args[0])} command is not allowed, skip...", flush=True)
                continue
            cmd_result = cmd_control.exec_command(cmd_args, capture=True)
            print(cmd_result.ret)
            print(cmd_result.out)
            cmd_results.append(cmd_result)
        outputs = "\n".join([cmd_result.out for cmd_result in cmd_results])

        output_query = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
f"""You are given the following user query:

{textwrap.indent(user_query[-1]["content"], "    ")}

and you are given the following command(s):

```sh
{commands}
```

the output of which in the current directory is the following:


```console
{outputs}
```

Formulate a response to the user query.

If appropriate, suggest any next steps, including commands to be run."""
                ),
            },
        ]
        result = await self.services.client.message(
            think_model,
            output_query,
            think_sampling_params,
            fresh=True,
        )
        message = result.message()
        thinking_part = Message.get_thinking_part(message)
        response = Message.get_text(message)
        if thinking_part is not None:
            print("<think>")
            print(thinking_part["thinking"])
            print("</think>\n")
        print(response)

        blocks = []
        start = 0
        while True:
            block = MarkdownCodeBlock.extract_next(response, start)
            if block is None:
                break
            blocks.append(block)
            start = block["end"]
        if not blocks:
            return

        if depth >= 2:
            return

        commands = "\n".join([block["text"].rstrip() for block in blocks])
        return tail_await(self.run_commands, user_query, commands, depth=depth+1)

if __name__ == "__main__":
    pass
