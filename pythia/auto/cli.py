from typing import Any, Optional
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
import asyncio
import functools
import itertools
import os
import platform
import signal
import sys
import textwrap

from prompt_toolkit.input import create_input
from prompt_toolkit.keys import Keys

from pythia.auto.kernel import (
    Autopythia, EndControlEvent, OutputEvent,
)
from pythia.clock import Timestamp
from pythia.io_control.command import exec_command
from pythia.python_utils import _py_version
from pythia.term_utils import (
    rclear, bold, plain, cyan, dim, underline,
)

HOME = os.environ["HOME"]
GLOBAL_DIR = os.path.join(HOME, ".pythia", "auto")

@dataclass
class _InputState:
    halt: asyncio.Event
    ret: asyncio.Event
    buf: list
    rbuf: list
    pos: int
    _input: Any

    def handle_key_presses(self):
        for key_press in self._input.read_keys():
            if key_press.key == Keys.ControlC:
                self.halt.set()
            elif key_press.key == Keys.ControlD:
                self.halt.set()
            elif key_press.key == Keys.Enter:
                self.ret.set()
            elif key_press.key == Keys.Backspace:
                if self.buf:
                    self.buf.pop()
                    if self.pos > 0:
                        self.pos -= 1
            elif key_press.key == Keys.ControlW:
                pass
            elif key_press.key == Keys.ControlU:
                self.buf.clear()
                self.rbuf.clear()
                self.pos = -1
            elif key_press.key == Keys.ControlK:
                self.rbuf.clear()
                self.pos = -1
            elif key_press.key == Keys.ControlA:
                self.pos = 0
                self.rbuf = self.buf + self.rbuf
                self.buf.clear()
            elif key_press.key == Keys.ControlE:
                self.pos = -1
                self.buf = self.buf + self.rbuf
                self.rbuf.clear()
            elif key_press.key == Keys.Left:
                if self.pos < 0:
                    self.pos = max(0, len(self.buf) + len(self.rbuf) - 1)
                else:
                    self.pos = max(0, self.pos - 1)
                assert self.pos <= len(self.buf)
                self.rbuf = self.buf[self.pos:] + self.rbuf
                self.buf = self.buf[:self.pos]
            elif key_press.key == Keys.Right:
                if self.pos < 0:
                    pass
                else:
                    self.pos = min(len(self.buf) + len(self.rbuf), self.pos + 1)
                if self.pos < 0:
                    pass
                elif self.pos >= len(self.buf) + len(self.rbuf):
                    self.pos = -1
                    self.buf = self.buf + self.rbuf
                    self.rbuf.clear()
                else:
                    assert self.pos >= len(self.buf)
                    buf_len = len(self.buf)
                    self.buf = self.buf + self.rbuf[:(self.pos - buf_len)]
                    self.rbuf = self.rbuf[(self.pos - buf_len):]
            elif key_press.data is not None and len(key_press.data) == 1:
                self.buf.append(key_press.data)
                if self.pos >= 0:
                    self.pos += 1

    def build_input_line(self, prompt: str, end: str = "") -> str:
        input_len = len(self.buf) + len(self.rbuf)
        input_pos = self.pos
        trailing = f"{rclear()}"
        if input_pos < 0:
            pass
        elif input_pos < input_len:
            if input_pos < 0:
                input_pos = input_len
            back = input_len - self.pos
            if back:
                back = "\b" * back
                trailing = f"""{trailing}{back}"""
        return f"""\r{prompt} {"".join(self.buf)}{"".join(self.rbuf)}{trailing}{end}"""

    def flush(self) -> str:
        text = f"""{"".join(self.buf)}{"".join(self.rbuf)}"""
        self.buf.clear()
        self.rbuf.clear()
        self.pos = -1
        return text

def quotewrap(haystack: str) -> str:
    parts = haystack.split("\n", maxsplit=1)
    first = parts[0]
    quote_both = f"""   {dim("[")}"""
    if not first:
        return quote_both
    if len(parts) > 1:
        haystack = parts[1]
    else:
        return textwrap.indent(first, quote_both)
    parts = haystack.rsplit("\n", maxsplit=1)
    haystack = parts[0]
    quote_first = f"""   {dim("⌜")}"""
    quote_last = f"""   {dim("⌞")}"""
    if len(parts) > 1:
        last = parts[1]
    else:
    # if not last:
        return f"""{textwrap.indent(first, quote_first)}\n{textwrap.indent(haystack, quote_last)}"""
    return f"""{textwrap.indent(first, quote_first)}\n{textwrap.indent(haystack, "    ")}\n{textwrap.indent(last, quote_last)}"""

async def _setup_main(args):
    input_state = _InputState(
        halt = asyncio.Event(),
        ret = asyncio.Event(),
        buf = [],
        rbuf = [],
        pos = -1,
        _input = create_input(),
    )
    def input_key_presses():
        input_state.handle_key_presses()
    with input_state._input.raw_mode():
        with input_state._input.attach(input_key_presses):
            await _run_main(args, input_state)

def test_relpath(prefix, suffix):
    p = os.path.join(prefix, suffix)
    if not os.path.exists(p):
        p = None
    return p

async def sleeping_beauty():
    while True:
        await asyncio.sleep(3)

async def _run_main(args, input_state: _InputState):
    arr = ">"
    spin = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    spin_len = len(spin)
    spin_step = 16
    fps = 128
    delay = 1 / fps
    arch = platform.machine().lower()
    if arch == "amd64":
        arch = "x86_64"
    elif arch == "arm64":
        arch = "aarch64"
    osys = platform.system().lower()
    py_ver = ".".join([f"{v}" for v in _py_version])
    # version = getattr(pythia, "__version__", None)
    version = None
    githash = None
    # if version is None:
    if False:
        version = "editable"
        try:
            githash_result = exec_command(["git", "log", "-n", "1", "--format=%H"], capture=True)
        except Exception:
            githash_result = None
        if (
            githash_result is not None and
            githash_result.out is not None and
            len(githash_result.out) >= 8 and
            not githash_result.out.startswith("fatal:")
        ):
            githash = githash_result.out.rstrip()
            version = f"editable+git:{githash[:8]}"
    cwd = os.getcwd()
    local_work_dir = test_relpath(cwd, ".autopythia")
    agents_md_path = test_relpath(cwd, "AGENTS.md")
    claude_md_path = test_relpath(cwd, "CLAUDE.md")
    style_md_path = test_relpath(cwd, "STYLE.md")
    screen_var = os.environ.get("STY", None)
    screen_pid = None
    screen_name = None
    if screen_var is not None:
        screen_var_parts = screen_var.split(".", maxsplit=1)
        try:
            screen_pid = int(screen_var_parts[0])
        except ValueError:
            screen_pid = None
        if len(screen_var_parts) > 1:
            screen_name = screen_var_parts[1]
    if githash is not None:
        workcopy_githash = githash
    else:
        workcopy_githash_result = exec_command(["git", "log", "-n", "1", "--format=%H"], capture=True)
        if (
            workcopy_githash_result is not None and
            workcopy_githash_result.out is not None and
            len(workcopy_githash_result.out) >= 8 and
            not workcopy_githash_result.out.startswith("fatal:")
        ):
            workcopy_githash = workcopy_githash_result.out.rstrip()
        else:
            workcopy_githash = None
    auto = Autopythia()
    workqueue = auto._workqueue
    workqueue.add(asyncio.create_task(sleeping_beauty()))
    session_ctr = auto.fresh_session_ctr()
    print(f"""{bold("autopythia")} {arch}-{osys} python{py_ver}""")
    if True:
        print(f"""{plain("Isolation mode")}      = {"Current working dir"}""")
    else:
        print(f"""{plain("Isolation mode")}      = {dim("<None>")}""")
    print(f"""{plain("Session ctr")}         = {session_ctr}""")
    if screen_var is not None:
        print(f"""{plain("Screen session")}      = {underline(f"{screen_pid}")}.{underline(f"{screen_name}")}""")
    elif args.verbose:
        print(f"""{plain("Screen session")}      = {dim("<None>")}""")
    print(f"""{plain("Global state dir")}    = {underline(GLOBAL_DIR)}""")
    print(f"""{plain("Current working dir")} = {underline(cwd)}""")
    if local_work_dir is not None:
        print(f"""{plain("Working state dir")}   = {underline(local_work_dir)}""")
    elif args.verbose:
        print(f"""{plain("Working state dir")}   = {dim("<None>")}""")
    if agents_md_path is not None:
        print(f"""{plain("AGENTS.md path")}      = {underline(agents_md_path)}""")
    elif args.verbose:
        print(f"""{plain("AGENTS.md path")}      = {dim("<None>")}""")
    if claude_md_path is not None:
        print(f"""{plain("CLAUDE.md path")}      = {underline(claude_md_path)}""")
    elif args.verbose:
        print(f"""{plain("CLAUDE.md path")}      = {dim("<None>")}""")
    if style_md_path is not None:
        print(f"""{plain("STYLE.md path")}       = {underline(style_md_path)}""")
    elif args.verbose:
        print(f"""{plain("STYLE.md path")}       = {dim("<None>")}""")
    print(f"""\r{spin[-1]} {rclear()}""", end="", flush=True)
    query_ctr = 0
    query = None
    start = set()
    halt = False
    for t in itertools.count():
        done, _pending = await asyncio.wait(workqueue, return_when=asyncio.FIRST_COMPLETED, timeout=delay)
        workqueue -= done
        # print(f"DEBUG: done={len(done)} work={len(work)}", flush=True)
        if not workqueue or input_state.halt.is_set():
            halt = True
            break
        print1 = False
        ret = False
        if input_state.ret.is_set():
            ret = True
            input_state.ret.clear()
        for task in done:
            event = task.result()
            output = None
            if isinstance(event, EndControlEvent):
                # FIXME
                start.clear()
                # start.remove(_)
            elif isinstance(event, OutputEvent):
                output = event
            if output is not None:
                if not print1:
                    prefix = f"\r{rclear()}\n"
                    print1 = True
                else:
                    prefix = "\n"
                # output = quotewrap(f"{output}")
                outputs = []
                block = []
                for leaf in output.leaf_events():
                    if leaf.leaf_type() == "basic":
                        if block:
                            outputs.append(quotewrap("\n\n".join(block)))
                            block.clear()
                        outputs.append(quotewrap(f"{leaf}"))
                    elif leaf.leaf_type() in ("thinking", "answer"):
                        block.append(f"{leaf}")
                    else:
                        raise NotImplementedError
                if block:
                    outputs.append(quotewrap("\n\n".join(block)))
                    block.clear()
                output = "\n\n".join(outputs)
                print(f"""{prefix}{output}""", flush=True)
                # auto.append_history(session_ctr, query_ctr, output=output)
        if ret:
            prompt = f"{arr}{arr}"
        elif not start:
            prompt = f":{arr}"
        else:
            prompt = f"{spin[(t // spin_step) % spin_len]}{arr}"
        if ret:
            end = "\n"
        else:
            end = ""
        print(input_state.build_input_line(prompt, end), end="", flush=True)
        if ret:
            input_query = input_state.flush()
            query = input_query.strip()
            if query.startswith("/"):
                if query in ("/exit", "/q", "/quit"):
                    break
                elif query in ("/h", "/help"):
                    pass
                elif query in ("/a", "/accept"):
                    pass
                elif query in ("/revise",):
                    pass
                elif query in ("/review",):
                    pass
                # auto.append_history(session_ctr, query_ctr, query=query)
            elif query:
                query_ctr = auto.fresh_query_ctr(session_ctr)
                workqueue.add(asyncio.create_task(auto.init(query)))
                auto.append_history(session_ctr, query_ctr, query=query)
                start.add(query_ctr)
            # query = None
    if args.verbose:
        print("\nGoodbye.", flush=True)
    elif halt:
        print("", flush=True)
    cur = asyncio.current_task()
    for t in asyncio.all_tasks():
        if t is cur:
            continue
        t.cancel()

def main(args):
    asyncio.run(_setup_main(args))

def parse_args(argv: Optional[list[str]] = None):
    args = ArgumentParser()
    args.add_argument("--verbose", "-v", action=BooleanOptionalAction, default=False)
    if argv is not None:
        args = args.parse_args(argv)
    else:
        args = args.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
