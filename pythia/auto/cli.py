from typing import Any, Optional
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass, field
import asyncio
import inspect
import itertools
import os
import platform
import shutil
import subprocess
import sys

from pythia.auto.kernel import (
    Autopythia,
    StartControlEvent,
    EndControlEvent,
    OutputEvent,
    BasicOutputEvent,
)
from pythia.clock import Timestamp
from pythia.term_input import create_input, Keys
from pythia.io_control.command import exec_command
from pythia.python_utils import _py_version
from pythia.term_utils import (
    rclear, bold, plain, cyan, bright_key, dim, underline,
)

HOME = os.environ["HOME"]
GLOBAL_DIR = os.path.join(HOME, ".pythia", "auto")


def _resolve_plugin_query_parameter(plugin_extension):
    try:
        signature = inspect.signature(plugin_extension)
    except (TypeError, ValueError):
        return None
    parameters = list(signature.parameters.values())
    if len(parameters) < 2:
        return None
    return parameters[1]


def _plugin_extension_supports_empty_query(plugin_extension) -> bool:
    query_parameter = _resolve_plugin_query_parameter(plugin_extension)
    if query_parameter is None:
        return True
    if query_parameter.kind in (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    ):
        return True
    return query_parameter.default is not inspect.Parameter.empty


def _call_plugin_extension(plugin_extension, step_ctr: int, query_text: str):
    query_parameter = _resolve_plugin_query_parameter(plugin_extension)
    if query_parameter is None:
        return plugin_extension(step_ctr)
    return plugin_extension(step_ctr, query_text)

@dataclass
class _InputEvent:
    def __post_init__(self):
        self._ctr = -1

    @classmethod
    async def afresh(cls):
        return cls()

@dataclass
class _InputLineBuffer:
    buf: list[str] = field(default_factory=list)
    rbuf: list[str] = field(default_factory=list)
    pos: int = -1

    def buffer_len(self) -> int:
        return len(self.buf) + len(self.rbuf)

    def buffer_pos(self) -> int:
        if self.pos < 0:
            return self.buffer_len()
        else:
            return self.pos

    def flush(self) -> str:
        text = f"""{"".join(self.buf)}{"".join(self.rbuf)}"""
        self.buf.clear()
        self.rbuf.clear()
        self.pos = -1
        return text

    def backspace(self):
        if self.buf:
            self.buf.pop()
            if self.pos > 0:
                self.pos -= 1
        if self.pos == 0 and len(self.rbuf) <= 0:
            self.pos = -1

    def clear(self):
        self.buf.clear()
        self.rbuf.clear()
        self.pos = -1

    def clear_left(self):
        self.buf.clear()
        self.pos = 0
        if len(self.rbuf) <= 0:
            self.pos = -1

    def clear_right(self):
        self.rbuf.clear()
        self.pos = -1

    def _deprecated_resplit_left(self):
        # TODO: deprecate.
        # Re-split the bipartite line buffer at the current cursor position,
        # assuming it has been moved to the left.
        assert self.pos >= 0
        assert self.pos <= len(self.buf)
        if self.pos == len(self.buf) and len(self.rbuf) <= 0:
            self.pos = -1
            return
        self.rbuf = self.buf[self.pos:] + self.rbuf
        self.buf = self.buf[:self.pos]
        if len(self.rbuf) <= 0:
            self.pos = -1

    def _resplit(self):
        # Re-split the bipartite line buffer at the current cursor position.
        if self.pos < 0 and len(self.rbuf) <= 0:
            pass
        elif self.pos < 0 or self.pos >= len(self.buf) + len(self.rbuf):
            self.pos = -1
            self.buf = self.buf + self.rbuf
            self.rbuf.clear()
        elif self.pos >= len(self.buf):
            buf_len = len(self.buf)
            self.buf = self.buf + self.rbuf[:(self.pos - buf_len)]
            self.rbuf = self.rbuf[(self.pos - buf_len):]
        else:
            self.rbuf = self.buf[self.pos:] + self.rbuf
            self.buf = self.buf[:self.pos]
        # if len(self.rbuf) <= 0:
        #     self.pos = -1

    def pop_left(self):
        pos = self.pos
        if pos < 0:
            pos = len(self.buf)
        init = True
        for p in range(pos - 1, -1, -1):
            if self.buf[p] in (" ", "\t"):
                if not init:
                    break
            else:
                init = False
            pos = p
        self.pos = pos
        self.buf = self.buf[:self.pos]
        self._resplit()

    def key_left(self):
        if self.pos < 0:
            self.pos = max(0, len(self.buf) + len(self.rbuf) - 1)
        else:
            self.pos = max(0, self.pos - 1)
        self._resplit()

    def key_right(self):
        if self.pos < 0:
            pass
        else:
            self.pos = min(len(self.buf) + len(self.rbuf), self.pos + 1)
        self._resplit()

    def word_left(self):
        full_buf = self.buf + self.rbuf
        pos = self.buffer_pos()
        if pos <= 0:
            self.snap_left()
            return

        pos -= 1
        while pos > 0 and full_buf[pos] in (" ", "\t"):
            pos -= 1
        while pos > 0 and full_buf[pos - 1] not in (" ", "\t"):
            pos -= 1

        self.pos = pos
        self._resplit()

    def word_right(self):
        full_buf = self.buf + self.rbuf
        pos = self.buffer_pos()
        limit = len(full_buf)
        if pos >= limit:
            self.snap_right()
            return

        while pos < limit and full_buf[pos] in (" ", "\t"):
            pos += 1
        while pos < limit and full_buf[pos] not in (" ", "\t"):
            pos += 1

        self.pos = pos
        self._resplit()

    def snap_left(self):
        self.rbuf = self.buf + self.rbuf
        self.buf.clear()
        self.pos = 0
        if len(self.rbuf) <= 0:
            self.pos = -1

    def snap_right(self):
        self.buf = self.buf + self.rbuf
        self.rbuf.clear()
        self.pos = -1

    def append(self, data: str):
        # TODO: multi-char keypresses?
        for elem in data:
            self.buf.append(elem)
            if self.pos >= 0:
                self.pos += 1

@dataclass
class _InputState:
    halt: asyncio.Event
    ret: asyncio.Event
    lbuf: _InputLineBuffer
    hpos: int
    hmax: int
    width: int
    prompt_width: int
    _input: Any
    _workqueue: Optional[set] = None

    def handle_key_presses(self):
        pressed = False
        for key_press in self._input.read_keys():
            pressed = True
            self._handle_key_press(key_press)
        if pressed and self._workqueue is not None:
            self._workqueue.add(asyncio.create_task(_InputEvent.afresh()))

    def _handle_key_press(self, key_press):
        while True:
            if key_press.key == Keys.ControlC:
                if False:
                    for _ in range(40):
                        print()
                    print(f"DEBUG: _InputState: ctrl-c: buf      = {self.lbuf.buf}")
                    print(f"DEBUG: _InputState: ctrl-c: buf.len  = {len(self.lbuf.buf)}")
                    print(f"DEBUG: _InputState: ctrl-c: rbuf.len = {len(self.lbuf.rbuf)}")
                    print(f"DEBUG: _InputState: ctrl-c: pos      = {self.lbuf.pos}")
                    print(f"DEBUG: _InputState: ctrl-c: hpos     = {self.hpos}")
                    print(f"DEBUG: _InputState: ctrl-c: hmax     = {self.hmax}")
                    print(f"DEBUG: _InputState: ctrl-c: width    = {self.width}")
                self.halt.set()
            elif key_press.key == Keys.ControlD:
                self.halt.set()
            elif key_press.key == Keys.Enter:
                self.ret.set()
            elif key_press.key == Keys.Backspace:
                self.lbuf.backspace()
            elif key_press.key == Keys.ControlW:
                self.lbuf.pop_left()
            elif key_press.key == Keys.ControlU:
                self.lbuf.clear_left()
            elif key_press.key == Keys.ControlK:
                self.lbuf.clear_right()
            elif key_press.key == Keys.ControlA:
                self.lbuf.snap_left()
            elif key_press.key == Keys.ControlE:
                self.lbuf.snap_right()
            elif key_press.key == Keys.ControlJ:
                pass
            elif key_press.key == Keys.Escape:
                pass
            elif key_press.key == Keys.Up:
                pass
            elif key_press.key == Keys.Down:
                pass
            elif key_press.key == Keys.Left:
                self.lbuf.key_left()
            elif key_press.key == Keys.Right:
                self.lbuf.key_right()
            elif key_press.key == Keys.WordLeft:
                self.lbuf.word_left()
            elif key_press.key == Keys.WordRight:
                self.lbuf.word_right()
            elif key_press.data is not None:
                self.lbuf.append(key_press.data)
            break

    def buffer_len(self) -> int:
        input_len = self.lbuf.buffer_len()
        return input_len

    def reset_input_line(self) -> str:
        # assert self.prompt_width == len(prompt)
        input_width = self.width
        input_len = self.lbuf.buffer_len()
        input_pos = self.lbuf.buffer_pos()
        input_height = (input_len + input_width) // input_width
        save_hpos = self.hpos
        save_hmax = self.hmax
        if self.hmax < input_height:
            self.hmax = input_height
        self.hpos = input_pos // input_width
        hoff = self.hmax - self.hpos - 1
        roff = self.prompt_width + input_pos % input_width
        return f"""\x1b[{save_hpos + 1}A"""

    def build_input_line(self, prompt: str, prompt1: str = "\n > ", prompt2: str = "\n   ", end: str = "", reset: bool = True) -> str:
        # assert self.prompt_width == len(prompt)
        input_width = self.width
        input_len = self.lbuf.buffer_len()
        input_pos = self.lbuf.buffer_pos()
        if False:
            trailing = f"{rclear()}"
            if input_pos < 0:
                pass
            elif input_pos < input_len:
                if input_pos < 0:
                    input_pos = input_len
                back = input_len - self.lbuf.pos
                if back:
                    back = "\b" * back
                    trailing = f"""{trailing}{back}"""
            return f"""\r{prompt}{"".join(self.lbuf.buf)}{"".join(self.lbuf.rbuf)}{trailing}{end}"""
        input_height = (input_len + input_width) // input_width
        save_hpos = self.hpos
        save_hmax = self.hmax
        if self.hmax < input_height:
            self.hmax = input_height
        self.hpos = input_pos // input_width
        hoff = self.hmax - self.hpos - 1
        roff = self.prompt_width + input_pos % input_width
        input_parts = []
        if reset:
            input_parts.append(f"""\x1b[{save_hpos + 1}A""")
        full_buf = self.lbuf.buf + self.lbuf.rbuf
        line = (
            f"""\r{prompt}{"".join(full_buf[:input_width])}{rclear()}"""
        )
        input_parts.append(line)
        for h in range(1, input_height):
            line = (
                f"""\r{prompt1}{"".join(full_buf[(input_width*h):(input_width*(h+1))])}{rclear()}"""
            )
            input_parts.append(line)
        for _ in range(input_height, self.hmax):
            line = f"""\r{prompt2}{rclear()}"""
            input_parts.append(line)
        if roff > 0:
            part = f"""\r\x1b[{roff}C"""
            input_parts.append(part)
        if hoff > 0:
            part = f"""\x1b[{hoff}A"""
            input_parts.append(part)
        input_parts.append(end)
        return "".join(input_parts)

    def flush(self) -> str:
        if False:
            text = f"""{"".join(self.lbuf.buf)}{"".join(self.lbuf.rbuf)}"""
            self.lbuf.buf.clear()
            self.lbuf.rbuf.clear()
            self.lbuf.pos = -1
            return text
        text = self.lbuf.flush()
        self.hpos = 0
        self.hmax = 1
        return text

def quotewrap(haystack: str) -> str:
    quote_both = f"""   {bright_key("[")}"""
    lines = haystack.split("\n")
    if haystack.endswith("\n"):
        # Treat a single terminal newline as a line terminator, not as an
        # extra blank content line.
        lines.pop()
    if not lines or (len(lines) == 1 and not lines[0]):
        return quote_both
    if len(lines) == 1:
        return f"{quote_both}{lines[0]}"
    quote_first = f"""   {bright_key("⌜")}"""
    quote_last = f"""   {bright_key("⌞")}"""
    parts = [f"{quote_first}{lines[0]}"]
    parts.extend(
        f"    {line}" if line else ""
        for line in lines[1:-1]
    )
    parts.append(f"{quote_last}{lines[-1]}")
    return "\n".join(parts)

async def _setup_main(args):
    width, _ = shutil.get_terminal_size()
    input_state = _InputState(
        halt = asyncio.Event(),
        ret = asyncio.Event(),
        lbuf = _InputLineBuffer(),
        hpos = 0,
        hmax = 1,
        width = width - 1 - 3,
        prompt_width = 3,
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
    # arr = ">"
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
    if args.target is not None:
        file_target = args.f
        dir_target = args.d
        assert not (file_target and dir_target)
        if not dir_target:
            file_target = file_target or (
                args.target.endswith(".tar") or
                args.target.endswith(".tar.gz") or
                args.target.endswith(".tar.bz2") or
                args.target.endswith(".tar.xz")
            )
        if file_target:
            raise NotImplementedError
        else:
            os.chdir(args.target)
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
    auto_kwargs = {
        "contradex_api_base_url": args.api_base_url,
    }
    if args.model is not None:
        auto_kwargs["contradex_model_path"] = args.model
    auto = Autopythia(**auto_kwargs)
    workqueue = auto._workqueue
    workqueue.add(asyncio.create_task(sleeping_beauty()))
    input_state._workqueue = workqueue
    if args.resume:
        session_ctr = auto._get_session_ctr()
    else:
        session_ctr = auto._fresh_session_ctr()
    auto._set_session(session_ctr)
    print(f"""{bold("autopythia")} {arch}-{osys} python{py_ver}""")
    if args.verbose:
        print(f"""{plain("Py executable")}       = {sys.executable}""")
        print(f"""{plain("Py argv")}             = {sys.argv}""")
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
    print(f"""\r\n{spin[-1]} {rclear()}""", end="", flush=True)
    done_events = []
    # flush = False
    query = None
    # ret = False
    step_ctr = 0
    start = set()
    halt = False
    reboot = False
    for frame_ctr in itertools.count():
        done, _pending = await asyncio.wait(workqueue, return_when=asyncio.FIRST_COMPLETED, timeout=delay)
        workqueue -= done
        # print(f"DEBUG: done={len(done)} work={len(work)}", flush=True)
        if not workqueue or input_state.halt.is_set():
            halt = True
            break
        for task in done:
            event = task.result()
            if event is None:
                continue
            done_events.append(event)
        print1 = False
        ret = False
        if input_state.ret.is_set():
            ret = True
            input_state.ret.clear()
        # if ret:
        if False:
            prompt = f"\n>> "
            if flush:
                prompt = f"\n{prompt}"
                flush = False
            end = "\n"
            print(input_state.build_input_line(prompt, end=end), end="", flush=True)
            continue
        if not ret:
            done_events.sort(key=lambda event: event._ctr)
            reset = True
            output = None
            for event in done_events:
                output = None
                if isinstance(event, StartControlEvent):
                    start.add(event.step_ctr)
                elif isinstance(event, EndControlEvent):
                    start.remove(event.step_ctr)
                elif isinstance(event, OutputEvent):
                    output = event
                if output is not None:
                    if not print1:
                        # prefix = f"\r{rclear()}\n"
                        # prefix = "\n"
                        print1 = True
                    else:
                        # prefix = "\n"
                        pass
                    # prefix = ""
                    # suffix = "\n"
                    # if input_state.buffer_len() <= 0:
                    # if False:
                    if True:
                        if reset:
                            prefix = f"{input_state.reset_input_line()}\n"
                            # prefix = f"{input_state.reset_input_line()}\r{rclear()}\n"
                            reset = False
                        else:
                            prefix = "\n"
                        suffix = ""
                    else:
                        prefix = "\n"
                        suffix = ""
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
                    print(f"""{prefix}{output}{suffix}""", flush=True)
                    # auto.append_history(session_ctr, step_ctr, output=output)
            done_events.clear()
        # if output is not None:
        #     print("", flush=True)
        if ret:
            prompt = f"\n>> "
        elif not start:
            prompt = f"\n:> "
        else:
            prompt = f"\n{spin[(frame_ctr // spin_step) % spin_len]}> "
        # if flush:
        #     prompt = f"\n{prompt}"
        #     flush = False
        if ret:
            end = "\n"
        else:
            end = ""
        # print(input_state.build_input_line(prompt, end=end, reset=reset), end="", flush=True)
        input_line = input_state.build_input_line(prompt, end="", reset=reset)
        halt = False
        reboot = False
        # flush = False
        flush = True
        if ret:
            query = input_state.flush().strip()
            query_args = query.split()
            query_head = query_args[0] if query_args else None
            query_args = query_args[1:] if query_args else None
            if query_head and query_head.startswith("/"):
                if query_head in ("/exit", "/quit"):
                    halt = True
                    flush = False
                # elif query_head in ("/reboot",):
                #     reboot = True
                #     flush = False
                elif query_head.startswith("//"):
                    flush = False
                    pass
                elif query_head in ("/date", "/now"):
                    t0 = Timestamp()
                    workqueue.add(asyncio.create_task(
                        BasicOutputEvent.afresh(text=f"{t0}")
                    ))
                elif query_head in ("/echo",):
                    qargs_text = query[5:].lstrip()
                    workqueue.add(asyncio.create_task(
                        BasicOutputEvent.afresh(text=qargs_text)
                    ))
                elif query_head in ("/auto", "/pythia"):
                    step_ctr = auto._fresh_step_ctr(session_ctr)
                    qargs_text = query[len(query_head):].lstrip()
                    workqueue.add(asyncio.create_task(auto.init(step_ctr, qargs_text)))
                    auto.append_history(session_ctr, step_ctr, query=query)
                    start.add(step_ctr)
                elif query_head in ("/qq",):
                    step_ctr = auto._fresh_step_ctr(session_ctr)
                    qargs_text = query[3:].lstrip()
                    workqueue.add(asyncio.create_task(auto.qq(step_ctr, qargs_text)))
                    auto.append_history(session_ctr, step_ctr, query=qargs_text)
                    start.add(step_ctr)
                elif query_head in ("/cleanhtml",):
                    step_ctr = auto._fresh_step_ctr(session_ctr)
                    qargs_text = query[len(query_head):].lstrip()
                    workqueue.add(asyncio.create_task(auto.cleanhtml(step_ctr, qargs_text)))
                    auto.append_history(session_ctr, step_ctr, query=qargs_text)
                    start.add(step_ctr)
                elif query_head in ("/cd",):
                    pass
                elif query_head in ("/vim",):
                    p = subprocess.Popen(
                        ["vim"] + query_args,
                        stdin=sys.stdin,
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                        shell=False,
                    )
                    p.communicate()
                elif query_head in ("/nano",):
                    p = subprocess.Popen(
                        ["nano"] + query_args,
                        stdin=sys.stdin,
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                        shell=False,
                    )
                    p.communicate()
                elif query_head in ("/h", "/help"):
                    pass
                elif query_head in ("/a", "/accept"):
                    pass
                elif query_head in ("/status"):
                    pass
                elif query_head in ("/revise",):
                    pass
                elif query_head in ("/review",):
                    pass
                elif (plugin_extension := auto._resolve_plugin_extension(query_head)) is not None:
                    qargs_text = query[len(query_head):].lstrip()
                    if not qargs_text and not _plugin_extension_supports_empty_query(
                        plugin_extension
                    ):
                        workqueue.add(
                            asyncio.create_task(
                                BasicOutputEvent.afresh(text=f"Usage: {query_head} <query>")
                            )
                        )
                    else:
                        step_ctr = auto._fresh_step_ctr(session_ctr)
                        workqueue.add(
                            asyncio.create_task(
                                _call_plugin_extension(plugin_extension, step_ctr, qargs_text)
                            )
                        )
                        auto.append_history(session_ctr, step_ctr, query=qargs_text or query)
                        start.add(step_ctr)
                # auto.append_history(session_ctr, step_ctr, query=query)
            elif query:
                default_plugin_extension = auto._resolve_default_plugin_extension()
                if default_plugin_extension is not None:
                    step_ctr = auto._fresh_step_ctr(session_ctr)
                    workqueue.add(asyncio.create_task(default_plugin_extension(step_ctr, query)))
                    auto.append_history(session_ctr, step_ctr, query=query)
                    start.add(step_ctr)
            else:
                # flush = False
                pass
            # query = None
        if ret:
            if flush:
                end = "\n\n"
            else:
                end = "\n"
        else:
            end = ""
        print(input_line, end=end, flush=True)
        if halt or reboot:
            break
    if args.verbose:
        print("\nGoodbye.", flush=True)
    else:
        print("", flush=True)
    if reboot:
        pass
    auto.shutdown()
    cur = asyncio.current_task()
    for t in asyncio.all_tasks():
        if t is cur:
            continue
        t.cancel()

def main(args):
    asyncio.run(_setup_main(args))

def parse_args(argv: Optional[list[str]] = None):
    args = ArgumentParser()
    args.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help="Optional base URL override passed to /contradex backend requests",
    )
    args.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional contradex model override. Defaults to gpt-5.4 when unset.",
    )
    args.add_argument("--resume", action=BooleanOptionalAction, default=False)
    args.add_argument("-v", "--verbose", action=BooleanOptionalAction, default=False)
    args.add_argument("-d", action=BooleanOptionalAction, default=False)
    args.add_argument("-f", action=BooleanOptionalAction, default=False)
    args.add_argument("target", nargs="?", type=str, default=None)
    if argv is not None:
        args = args.parse_args(argv)
    else:
        args = args.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
