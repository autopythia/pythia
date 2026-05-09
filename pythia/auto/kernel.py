from typing import Any, Callable, Optional, TypedDict
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
import asyncio
import functools
import json
import os
# import shlex
import textwrap

from pythia.api import APIServices
from pythia.auto.prompts import *
from pythia.clock import Timestamp
from pythia.contrib.atomicswap import swap as swap_paths
from pythia.experimental.extract import extract_struct
from pythia.extract import (
    Message,
    MarkdownCodeBlock,
    MarkdownIndex,
)
from pythia.io_control.command import (
    ShellIOCommandController,
)
from pythia.shell import ShellPipeline, detect_shell
from pythia.term_utils import green, bright_key

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

def tail_task(a):
    return a
    # return asyncio.create_task(a)

# FIXME: should not be global but part of state.
_EVENT_CTR = 0

def _fresh_event_ctr() -> int:
    global _EVENT_CTR
    ctr = _EVENT_CTR + 1
    _EVENT_CTR = ctr
    return ctr

@dataclass
class StartControlEvent:
    step_ctr: int

    def __post_init__(self):
        self._ctr = _fresh_event_ctr()

@dataclass
class EndControlEvent:
    step_ctr: int

    def __post_init__(self):
        self._ctr = _fresh_event_ctr()

@dataclass
class OutputEvent:
    @classmethod
    async def afresh(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __post_init__(self):
        self._ctr = _fresh_event_ctr()

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
        return (
f"""{bright_key("<think>", bold=True)}
{bright_key(self.text)}
{bright_key("</think>", bold=True)}"""
        )

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
class ShellExecResult:
    cmd: str
    pipeline: ShellPipeline
    allowed: bool
    partial: bool
    final_output: Optional[str]

    def format(self) -> str:
        section = (
f"""Command: `{self.cmd}`
Success: {not result.partial}
Output: {self.final_output or ""}"""
        )
        return section

@dataclass
class DiffEditResult:
    path: str
    diff_text: str
    success: bool

    def format(self) -> str:
        section = (
f"""Path: `{self.path}`
Success? {result.success}
Diff:
{self.diff_text or ""}"""
        )
        return section

@dataclass
class CatEditResult:
    path: str
    text: str
    success: bool

    def format(self) -> str:
        section = (
f"""Path: `{self.path}`
Success? {result.success}
Content:
{self.text or ""}"""
        )
        return section

@dataclass
class Autopythia:
    shell: str = None

    # work_model:  str = "deepseek-ai/deepseek-v3.2-thinking-off"
    # think_model: str = "deepseek-ai/deepseek-v3.2-thinking"
    think_off_model: str = "moonshotai/kimi-k2.5-thinking-off"
    think_model: str = "moonshotai/kimi-k2.5-thinking"
    services: APIServices = None

    _session: Optional[int] = None
    _workqueue: Any = None

    def __post_init__(self):
        if self.shell is None:
            self.shell = detect_shell()
        if self.services is None:
            self.services = APIServices(enable_journal=False)
        if self._workqueue is None:
            self._workqueue = set()

    def shutdown(self):
        pass

    @classmethod
    def _plugin_extension_names(cls) -> tuple[str, ...]:
        return getattr(cls, "_autopythia_plugin_extensions", ())

    def _resolve_plugin_extension(self, name: str):
        if name.startswith("/"):
            name = name[1:]
        if name not in type(self)._plugin_extension_names():
            return None
        return getattr(self, name)

    def _resolve_default_plugin_extension(self):
        extension_names = type(self)._plugin_extension_names()
        if len(extension_names) == 1:
            return getattr(self, extension_names[0])
        if "contradex" in extension_names:
            return getattr(self, "contradex")
        return None

    def _fresh_session_ctr(self) -> int:
        ctr = fresh_ctr(GLOBAL_SESSION_DIR, "session_ctr")
        return ctr

    def _get_session_ctr(self) -> int:
        ctr = get_ctr(GLOBAL_SESSION_DIR, "session_ctr")
        return ctr

    def _fresh_step_ctr(self, session_ctr: int) -> int:
        ctr = fresh_ctr(os.path.join(GLOBAL_SESSION_DIR, f"{session_ctr}"), "step_ctr")
        return ctr

    def _set_session(self, session_ctr: int):
        self._session = session_ctr

    def _enqueue_event(self, event) -> None:
        self._workqueue.add(asyncio.create_task(echo_event(event)))

    def _enqueue_event_threadsafe(
        self,
        loop: asyncio.AbstractEventLoop,
        event_factory: Callable[[], OutputEvent | StartControlEvent | EndControlEvent],
    ) -> None:
        def enqueue() -> None:
            self._enqueue_event(event_factory())

        loop.call_soon_threadsafe(enqueue)

    def _emit_output_threadsafe(self, loop: asyncio.AbstractEventLoop, text: str) -> None:
        if not text:
            return
        self._enqueue_event_threadsafe(loop, lambda: BasicOutputEvent(text=text))

    def _resolve_default_contradex_api_provider(self) -> str:
        return "auto"

    def _resolve_default_contradex_model_path(self) -> str:
        return "gpt-5.4"

    def _load_contradex_config_from_env(self) -> dict[str, Any]:
        raw_display_level = os.environ.get("AUTO_PYTHIA_CONTRADEX_DISPLAY_LEVEL")
        display_level = 1
        if raw_display_level is not None:
            try:
                display_level = int(raw_display_level)
            except ValueError:
                display_level = 1

        raw_provider = os.environ.get("AUTO_PYTHIA_CONTRADEX_API_PROVIDER")
        if raw_provider is not None and raw_provider.strip():
            api_provider = raw_provider.strip()
        else:
            api_provider = self._resolve_default_contradex_api_provider()

        raw_model_path = os.environ.get("AUTO_PYTHIA_CONTRADEX_MODEL")
        if self.contradex_model_path is not None and self.contradex_model_path.strip():
            model_path = self.contradex_model_path.strip()
        elif raw_model_path is not None and raw_model_path.strip():
            model_path = raw_model_path.strip()
        else:
            model_path = self._resolve_default_contradex_model_path()

        return {
            "cwd": Path.cwd(),
            "model_client_kind": os.environ.get("AUTO_PYTHIA_CONTRADEX_MODEL_CLIENT", "urllib"),
            "api_key": os.environ.get("AUTO_PYTHIA_CONTRADEX_API_KEY"),
            "api_base_url": (
                os.environ.get("AUTO_PYTHIA_CONTRADEX_API_BASE_URL")
                or self.contradex_api_base_url
            ),
            "api_provider": api_provider,
            "model_path": model_path,
            "reasoning_effort": os.environ.get("AUTO_PYTHIA_CONTRADEX_REASONING_EFFORT", "xhigh"),
            "reasoning_summary": os.environ.get("AUTO_PYTHIA_CONTRADEX_REASONING_SUMMARY", "auto"),
            "codex_home": os.environ.get("AUTO_PYTHIA_CONTRADEX_CODEX_HOME"),
            "auth_file": os.environ.get("AUTO_PYTHIA_CONTRADEX_AUTH_FILE"),
            "display_level": display_level,
        }

    def _load_or_create_contradex_session(self):
        from contradex.integrations.autopythia import AutopythiaContradexConfig
        from contradex.integrations.autopythia import AutopythiaContradexSession

        config = AutopythiaContradexConfig(**self._load_contradex_config_from_env())
        if self._contradex_session is None or self._contradex_config != config:
            self._contradex_session = AutopythiaContradexSession.create(config)
            self._contradex_config = config
        return config, self._contradex_session

    def append_transcript(self, session_ctr, event):
        prefix = os.path.join(GLOBAL_SESSION_DIR, f"{session_ctr}")
        transcript_path = os.path.join(prefix, "transcript.txt")
        try:
            transcript_file = open(transcript_path, "a", encoding="utf-8")
        except OSError:
            os.makedirs(prefix, exist_ok=True)
            transcript_file = open(transcript_path, "a", encoding="utf-8")
        if isinstance(event, str):
            print(event, file=transcript_file, flush=True)
        elif isinstance(event, ThinkingOutputEvent):
            print(f"\n<think>{event.text}</think>", file=transcript_file, flush=True)
        elif isinstance(event, AnswerOutputEvent):
            print(f"\n{event.text}", file=transcript_file, flush=True)
        else:
            raise NotImplementedError

    def append_history(
        self,
        session_ctr: int,
        step_ctr: int,
        query: Optional[str] = None,
        messages: Optional[list] = None,
        t0=None,
        t1=None,
    ):
        if t0 is None:
            t0 = Timestamp()
        prefix = os.path.join(GLOBAL_SESSION_DIR, f"{session_ctr}")
        history_path = os.path.join(prefix, "history.jsonl")
        try:
            history_file = open(history_path, "a", encoding="utf-8")
        except OSError:
            os.makedirs(prefix, exist_ok=True)
            history_file = open(history_path, "a", encoding="utf-8")
        history_item = {
            "t0": f"{t0}",
            "t1": f"{t1}" if t1 is not None else None,
            "session_ctr": session_ctr,
            "session_uid": None,
            "step_ctr": step_ctr,
            "step_uid": None,
        }
        if query is not None:
            history_item["query"] = query
        elif messages is not None:
            history_item["messages"] = messages
        print(json.dumps(history_item), file=history_file, flush=True)
        history_file.close()
        return t0

    async def qq(self, step_ctr: int, query: str):
        think_model_path = self.think_model
        think_model = self.services.registry.find_model(think_model_path)
        think_sampling_params = {
            # "max_tokens": 65536,
            "max_tokens": 262144,
            "temperature": 1.0,
        }

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)
        # print(f"DEBUG: init: step ctr = {step_ctr}")

        event = StartControlEvent(step_ctr)
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        qq_query = [
            {
                "role": "user",
                "content": query,
            },
        ]
        self.append_transcript(self._session, query)

        event = BasicOutputEvent(green("Thinking...", bold=True))
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        t0 = Timestamp()
        qq_result = self.services.client.message(
            think_model,
            qq_query,
            think_sampling_params,
            fresh=True,
        )
        self.append_history(self._session, step_ctr, messages=qq_query, t0=t0)
        qq_result = await qq_result
        t1 = Timestamp()
        # print(qq_result)

        res_event = AtomicOutputEvent()
        event = BasicOutputEvent(green(f"Thought for {(t1 - t0).pretty_format()}", bold=True))
        res_event.append(event)

        message = qq_result.message()
        thinking_part = Message.get_thinking_part(message)
        answer = Message.get_text(message)

        if thinking_part:
            # print(f"""<think>\n{thinking_part["thinking"]}\n</think>\n""")
            event = ThinkingOutputEvent(thinking_part["thinking"])
            self.append_transcript(self._session, event)
            res_event.append(event)
        # print(answer)
        event = AnswerOutputEvent(answer)
        self.append_transcript(self._session, event)
        res_event.append(event)
        self._workqueue.add(asyncio.create_task(echo_event(res_event)))

        event = EndControlEvent(step_ctr)
        self._workqueue.add(asyncio.create_task(echo_event(event)))

    async def cleanhtml(self, step_ctr: int, query: str):
        think_model_path = self.think_off_model
        # think_model_path = self.think_model
        think_model = self.services.registry.find_model(think_model_path)
        think_sampling_params = {
            "max_tokens": 65536,
            # "max_tokens": 131072,
            # "max_tokens": 262144,
            "temperature": 1.0,
        }

        input_path = query.strip()
        # with open(input_path, "r") as input_file:
        #     input_text = input_file.read()
        input_file = open(input_path, "r")

        target_chunk_size = 5000
        # target_chunk_size = 10000
        # target_chunk_size = 25000
        # target_chunk_size = 50000
        # target_chunk_size = 100000
        # chunk_lens = []
        chunks = []
        chunk_len = 0
        chunk = []
        for line in input_file:
            chunk_len += len(line)
            chunk.append(line)
            if chunk_len >= target_chunk_size:
                chunk = "".join(chunk)
                chunks.append(chunk)
                chunk_len = 0
                chunk = []
        if chunk:
            chunk = "".join(chunk)
            chunks.append(chunk)

        input_file.close()

        input_chunks = chunks
        output_chunks = []

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)
        # print(f"DEBUG: init: step ctr = {step_ctr}")

        event = StartControlEvent(step_ctr)
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        event = BasicOutputEvent(green(f"Computed {len(input_chunks)} chunks", bold=True))
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        event = BasicOutputEvent(green("Thinking...", bold=True))
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        chunk_tasks = []
        chunk_results = {}
        chunk_outputs = {}

        async def _pair(key, fut):
            return key, await fut

        def create_task_pair(key, fut):
            return asyncio.create_task(_pair(key, fut))

        total_t0 = Timestamp()

# - Remove class, id, style, and non-rendered "metadata" fields inside tags.
# - Remove link, style (CSS), and script (Javascript) blocks.
# - Remove leading spaces or tabs that are not rendered (e.g. not inside a pre block).
# - Do not add any delimiters (e.g. fenced code block) around your HTML output.

        for chunk_idx, chunk in enumerate(input_chunks):
            chunk_query = [
                {
                    "role": "system",
                    "content": (
"""You are a meticulous data analyst. Below, you will be given a section of HTML to clean for algorithmic processing. Broadly, the intention of this HTML cleaning is to preserve the structure and content of the HTML document, while removing unneeded style and format.

You should follow these specific HTML cleaning criteria:

- Remove `class`, `style`, and non-rendered metadata fields inside tags.
- Remove `link`, `script`, and `style` blocks.
- Remove leading spaces or tabs that are not rendered (e.g. not inside a pre block).
- Do not add any delimiters (e.g. fenced code block) around your HTML output.

Your output should consist of and only of the cleaned HTML corresponding to the user input.
"""
                    ),
                },
                {
                    "role": "user",
                    "content": chunk,
                },
            ]
            # self.append_transcript(self._session, query)

            # event = BasicOutputEvent(green("Thinking...", bold=True))
            # self._workqueue.add(asyncio.create_task(echo_event(event)))

            t0 = Timestamp()
            chunk_result = self.services.client.message(
                think_model,
                chunk_query,
                think_sampling_params,
                fresh=True,
            )
            # chunk_results.append(chunk_result)
            # chunk_results.append(asyncio.create_task(chunk_result))
            chunk_tasks.append(create_task_pair(chunk_idx, chunk_result))
            self.append_history(self._session, step_ctr, messages=chunk_query, t0=t0)

        # chunk_results = await asyncio.gather(*chunk_results)

        async def _wait_tasks(tasks):
            rem_tasks = set([t for t in tasks])
            while rem_tasks:
                done, _pending = await asyncio.wait(rem_tasks, return_when=asyncio.FIRST_COMPLETED)
                rem_tasks -= done
                # rem_tasks = pending
                for e in done:
                    yield e.result()

        async for chunk_idx, chunk_result in _wait_tasks(chunk_tasks):
            # chunk_result = await chunk_result
            t1 = Timestamp()
            # print(chunk_result)

            res_event = AtomicOutputEvent()
            event = BasicOutputEvent(green(f"Thought for {(t1 - total_t0).pretty_format()}", bold=True))
            res_event.append(event)

            message = chunk_result.message()
            thinking_part = Message.get_thinking_part(message)
            answer = Message.get_text(message)

            output_text = answer
            if not output_text.endswith("\n"):
                output_text = f"{output_text}\n"

            with open(f"_tmp_clean.{chunk_idx}.html", "w") as output_file:
                print(output_text, end="", file=output_file, flush=True)

            chunk_outputs[chunk_idx] = output_text

            if thinking_part:
                # print(f"""<think>\n{thinking_part["thinking"]}\n</think>\n""")
                event = ThinkingOutputEvent(thinking_part["thinking"])
                self.append_transcript(self._session, event)
                res_event.append(event)
            # print(answer)
            event = AnswerOutputEvent(answer)
            # self.append_transcript(self._session, event)
            res_event.append(event)
            self._workqueue.add(asyncio.create_task(echo_event(res_event)))

            # step_ctr = self._fresh_step_ctr(self._session)

        event = EndControlEvent(step_ctr)
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        for chunk_idx in range(len(input_chunks)):
            output_text = chunk_outputs[chunk_idx]
            output_chunks.append(output_text)

        output_text = "".join(output_chunks)
        with open(f"_tmp_clean.html", "w") as output_file:
            print(output_text, end="", file=output_file, flush=True)

    async def init(self, step_ctr: int, query: str):
        think_model_path = self.think_model
        think_model = self.services.registry.find_model(think_model_path)
        think_sampling_params = {
            # "max_tokens": 8192,
            # "max_tokens": 16384,
            # "max_tokens": 32768,
            # "max_tokens": 65536,
            "max_tokens": 262144,
            # "temperature": 0.6,
            "temperature": 1.0,
        }

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)
        # print(f"DEBUG: init: step ctr = {step_ctr}")

        event = StartControlEvent(step_ctr)
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        formatted_query = textwrap.indent(query, "    ")
        plan_query = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(shell=self.shell),
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
        plan_result = self.services.client.message(
            think_model,
            plan_query,
            think_sampling_params,
            fresh=True,
        )
        self.append_history(self._session, step_ctr, messages=plan_query, t0=t0)
        plan_result = await plan_result
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

        index = MarkdownIndex.new(answer)
        # print(f"DEBUG: markdown index = {index}")
        plan_block = None
        for _, block in index._code_blocks.items():
            if block["lang"] == "markdown":
                block["text"] = index.extract_text(answer, block)["text"]
                plan_block = block
                break

        plan = None
        if plan_block is not None:
            # print(f"DEBUG: init: initial code block: {plan_block}", flush=True)
            plan = plan_block["text"].rstrip()

        if not plan:
            return await tail_task(self.init(None, query))

        new_results = self._parse_shell_commands(answer)
        results = new_results

        # return await tail_task(self.eval(None, query, plan, None, results))
        return await tail_task(self.backup(None, query, plan, None, results))

    async def eval(self, step_ctr: Optional[int], query: str, plan: str, scratch: Optional[str], results: list = []):
        think_model_path = self.think_model
        think_model = self.services.registry.find_model(think_model_path)
        think_sampling_params = {
            # "max_tokens": 8192,
            # "max_tokens": 16384,
            # "max_tokens": 32768,
            # "max_tokens": 65536,
            "max_tokens": 262144,
            # "temperature": 0.6,
            "temperature": 1.0,
        }

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)

        event = StartControlEvent(step_ctr)
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        formatted_query = textwrap.indent(query, "    ")
        formatted_scratch = textwrap.indent(scratch, "    ") if scratch is not None else ""
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
                results=formatted_results,
                scratch=formatted_scratch,
                plan=formatted_plan,
            )

        else:
            eval_prompt = EVAL_PROMPT_0.format(
                query=formatted_query,
                scratch=formatted_scratch,
                plan=formatted_plan,
            )

        eval_query = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(shell=self.shell),
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
        eval_result = self.services.client.message(
            think_model,
            eval_query,
            think_sampling_params,
            fresh=True,
        )
        self.append_history(self._session, step_ctr, messages=eval_query, t0=t0)
        eval_result = await eval_result
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

        index = MarkdownIndex.new(answer)
        # print(f"DEBUG: markdown index = {index}")
        write_plan = False
        write_scratch = False
        plan_block = None
        scratch_block = None
        for _, block in index._code_blocks.items():
            if block["lang"] == "markdown":
                prev_line = index.extract_prev_line_text(answer, block)
                if prev_line is not None and prev_line["text"].strip() == "/plan":
                    write_plan = True
                    block["text"] = index.extract_text(answer, block)["text"]
                    plan_block = block
                    # break
                elif prev_line is not None and prev_line["text"].strip() == "/scratch":
                    write_scratch = True
                    block["text"] = index.extract_text(answer, block)["text"]
                    scratch_block = block

        if scratch_block is not None:
            scratch = scratch_block["text"].rstrip()

        if not new_results:
            return await tail_task(self.eval(None, query, plan, scratch, results))
        else:
            return await tail_task(self.backup(None, query, plan, scratch, results))

    async def backup(self, step_ctr: Optional[int], query: str, plan: str, scratch: Optional[str], results = []):
        think_model_path = self.think_model
        think_model = self.services.registry.find_model(think_model_path)
        think_sampling_params = {
            # "max_tokens": 8192,
            # "max_tokens": 16384,
            # "max_tokens": 32768,
            # "max_tokens": 65536,
            "max_tokens": 262144,
            # "temperature": 0.6,
            "temperature": 1.0,
        }

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)

        event = StartControlEvent(step_ctr)
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        formatted_query = textwrap.indent(query, "    ")
        formatted_scratch = textwrap.indent(scratch, "    ") if scratch is not None else ""
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
                "content": SYSTEM_PROMPT.format(shell=self.shell),
            },
            {
                "role": "user",
                "content": BACKUP_PROMPT.format(
                    query=formatted_query,
                    results=formatted_results,
                    scratch=formatted_scratch,
                    plan=formatted_plan,
                ),
            },
        ]
        # print(backup_query)
        event = BasicOutputEvent(green("Thinking...", bold=True))
        self._workqueue.add(asyncio.create_task(echo_event(event)))

        t0 = Timestamp()
        backup_result = self.services.client.message(
            think_model,
            backup_query,
            think_sampling_params,
            fresh=True,
        )
        self.append_history(self._session, step_ctr, messages=backup_query, t0=t0)
        backup_result = await backup_result
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

        new_results = self._parse_edits(answer)
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

        # TODO: index-based impl.
        final_ = answer.find("/final") >= 0

        index = MarkdownIndex.new(answer)
        # print(f"DEBUG: markdown index = {index}")
        write_plan = False
        write_scratch = False
        plan_block = None
        scratch_block = None
        for _, block in index._code_blocks.items():
            if block["lang"] == "markdown":
                prev_line = index.extract_prev_line_text(answer, block)
                if prev_line is not None and prev_line["text"].strip() == "/plan":
                    write_plan = True
                    block["text"] = index.extract_text(answer, block)["text"]
                    plan_block = block
                    # break
                elif prev_line is not None and prev_line["text"].strip() == "/scratch":
                    write_scratch = True
                    block["text"] = index.extract_text(answer, block)["text"]
                    scratch_block = block

        if final_:
            event = BasicOutputEvent(green("Done.", bold=True))
            self._workqueue.add(asyncio.create_task(echo_event(event)))
            return

        # new_plan = None
        if plan_block is not None:
            plan = plan_block["text"].rstrip()
            # new_plan = plan

        if scratch_block is not None:
            scratch = scratch_block["text"].rstrip()

        # if not new_results:
        #     return await tail_task(self.eval(None, query, plan, scratch, results))
        # else:

        return await tail_task(self.backup(None, query, plan, scratch, results))

    def _old_extract_markdown_code_blocks(self, answer):
        block_start = None
        code_blocks = []
        while True:
            code_block = MarkdownCodeBlock.extract_next(answer, block_start)
            if code_block is None:
                break
            elif code_block["lang"] in ("sh", "bash", "zsh"):
                code_blocks.append(code_block)
            block_start = code_block["end"]
        # print(f"DEBUG: code blocks = {code_blocks} ...", flush=True)
        return code_blocks

    def _extract_markdown_code_blocks(self, haystack, index=None):
        if index is None:
            index = MarkdownIndex.new(haystack)
        # print(f"DEBUG: markdown index = {index}")
        code_blocks = []
        for _, block in index._code_blocks.items():
            block["text"] = index.extract_text(haystack, block)["text"]
            code_blocks.append(block)
        # print(f"DEBUG: code blocks = {code_blocks} ...", flush=True)
        return code_blocks

    def _parse_shell_commands(self, answer):
        index = MarkdownIndex.new(answer)
        code_blocks = self._extract_markdown_code_blocks(answer, index)
        print(f"DEBUG: parse shell: code blocks = {code_blocks}", flush=True)

        shell_code_blocks = []
        for block in code_blocks:
            if block["lang"] not in ("sh", "bash", "zsh"):
                # print(f"DEBUG: parse shell: not sh", flush=True)
                continue
            print(f"DEBUG: parse shell: block = {repr(answer[block['block_start']:block['block_end']])}", flush=True)
            prev_line_block = index.extract_prev_line_text(answer, block)
            if not prev_line_block:
                # print(f"DEBUG: parse shell: no prev line", flush=True)
                continue
            prev_line_parts = prev_line_block["text"].strip().split()
            if not prev_line_parts:
                # print(f"DEBUG: parse shell: no prev line parts", flush=True)
                continue
            if prev_line_parts[0] != "/exec":
                # print(f"DEBUG: parse shell: not /exec", flush=True)
                continue
            shell_code_blocks.append(block)
        print(f"DEBUG: parse shell: shell code blocks = {shell_code_blocks} ...", flush=True)

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
        for block in shell_code_blocks:
            cmd_line = None
            cmd_pipeline = None
            block_lines = block["text"].splitlines()
            for cmd_line in block_lines[:1]:
                cmd_line = cmd_line.strip()
                if not cmd_line:
                    continue
                elif cmd_line.startswith("#"):
                    continue
                # cmd_args = shlex.split(cmd_line)
                cmd_pipeline = ShellPipeline(cmd_line)
            if not cmd_pipeline:
                continue
            if cmd_pipeline.parsing_error or cmd_pipeline.not_supported:
                # TODO: report error here.
                continue
            # TODO: heredoc quoting?
            if (
                cmd_pipeline.stages and
                cmd_pipeline.stages[-1].in_hdoc
            ):
                cmd_pipeline.stages[-1].in_hdoclines = []
                for line in block_lines[1:]:
                    if line.rstrip() == cmd_pipeline.stages[-1].in_hdoc:
                        cmd_pipeline.stages[-1].in_hdoclines.append("")
                        break
                    cmd_pipeline.stages[-1].in_hdoclines.append(line)
            if (
                len(cmd_pipeline.stages) == 1 and
                cmd_pipeline.stages[0].cmd_args[0] == "cat" and
                cmd_pipeline.stages[0].in_hdoc and
                cmd_pipeline.stages[0].out_arg
            ):
                output = "\n".join(cmd_pipeline.stages[0].in_hdoclines)
                out_path = os.path.abspath(cmd_pipeline.stages[0].out_arg)
                cwd_path = os.path.abspath(os.getcwd())
                if (
                    os.path.commonpath([cwd_path]) ==
                    os.path.commonpath([cwd_path, out_path])
                ):
                    with open(out_path, "w") as out_file:
                        out_file.write(output)
                    allowed = True
                else:
                    allowed = False
                capture = ""
                result = ShellExecResult(
                    cmd_line,
                    cmd_pipeline,
                    allowed,
                    not allowed,
                    capture,
                )
                new_results.append(result)
            elif (
                len(cmd_pipeline.stages) == 1 and
                cmd_pipeline.stages[0].cmd_args[0] == "cat" and
                cmd_pipeline.stages[0].in_hdoc and
                not cmd_pipeline.stages[0].out_arg
            ):
                capture = "\n".join(cmd_pipeline.stages[0].in_hdoclines)
                allowed = True
                result = ShellExecResult(
                    cmd_line,
                    cmd_pipeline,
                    allowed,
                    not allowed,
                    capture,
                )
                new_results.append(result)
            elif True:
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

    def _parse_edits(self, answer):
        index = MarkdownIndex.new(answer)
        code_blocks = self._extract_markdown_code_blocks(answer, index)
        print(f"DEBUG: parse edits: code blocks = {code_blocks} ...", flush=True)

        edit_args = []
        edit_code_blocks = []
        for block in code_blocks:
            prev_line_block = index.extract_prev_line_text(answer, block)
            if prev_line_block:
                prev_line_parts = prev_line_block["text"].strip().split()
                if prev_line_parts and prev_line_parts[0] == "/edit":
                    edit_args.append(prev_line_parts[1:])
                    edit_code_blocks.append(block)
        print(f"DEBUG: parse edits: edit code blocks = {edit_code_blocks} ...", flush=True)

        new_results = []

        for args, block in zip(edit_args, edit_code_blocks):
            success = False
            if not args:
                continue
            lang = block["lang"]
            is_diff = lang in ("diff", "patch")
            dst_text = block["text"]
            if is_diff:
                diff_text = dst_text
            src_path = args[1]
            src_abspath = os.path.abspath(src_path)
            cwd_abspath = os.path.abspath(os.getcwd())
            if not (
                os.path.commonpath([cwd_abspath]) ==
                os.path.commonpath([cwd_abspath, src_abspath])
            ):
                if is_diff:
                    result = DiffEditResult(
                        src_path,
                        diff_text,
                        success,
                    )
                else:
                    result = CatEditResult(
                        src_path,
                        dst_text,
                        success,
                    )
                new_results.append(result)
                continue
            with open(src_path, "r") as src_file:
                src_text = src_file.read()
            if is_diff:
                diff = parse_diff(diff_text, ".")
                if diff is None:
                    result = DiffEditResult(
                        src_path,
                        diff_text,
                        success,
                    )
                    new_results.append(result)
                    continue
                dst_text = apply_diff(diff, src_text)
            if src_text != dst_text:
                with open(src_path, "w") as src_file:
                    src_file.write(dst_text)
                if is_diff:
                    result = DiffEditResult(
                        src_path,
                        diff_text,
                        success,
                    )
                else:
                    result = CatEditResult(
                        src_path,
                        dst_text,
                        success,
                    )
                new_results.append(result)

        return new_results

from pythia.auto.plugin import resolve_autopythia_class

Autopythia = resolve_autopythia_class(Autopythia)

if __name__ == "__main__":
    pass
