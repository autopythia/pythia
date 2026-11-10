"""The node server's tests (``pythia/interaction/node/``): the root contexts
that a registry lists, their logs' indexes and turns, the pages, the styles,
the HTTP server and its options.

With ``PYTHIA_LIVE_NODE_REGISTRY`` naming a registry, such as
``~/.pythia/root``, a live test serves every page of its root contexts but
single lines and runs of lines, then a few single lines of each. With
``PYTHIA_NODE_JS`` naming a JavaScript shell, such as ``qjs``, the glui
view's facts and bodies run through glui's own scripts and the program that
shows them, with a stand-in for a browser's DOM.
"""

from __future__ import annotations

import contextlib
import gzip
import html
import http.client
import io
import json
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from pythia.interaction._save_lock import take as take_lock
from pythia.interaction.node import activity, contexts, pages, server, styles, view
from pythia.interaction.node.store import MAX_BYTES as MAX_STORE
from pythia.interaction.node.store import MAX_FILE_BYTES as MAX_FILE
from pythia.interaction.node.store import Invalid, Store
from pythia.interaction.node.__main__ import main
from pythia.interaction.registry import Registry

NODE = Path(contexts.__file__).resolve().parent  # the package under test
CHECKOUT = NODE.parents[2]  # the pythia checkout that holds it
TOKEN = "t0ken_" + "x" * 37
KEY_A = "context_" + "a1" * 16  # local-0's
KEY_B = "context_" + "b2" * 16  # old's: its log has session_ and these digits
KEY_C = "context_" + "c3" * 16  # what b's path holds now
KEY_D = "context_" + "d4" * 16  # a save that was never registered
KEY_E = "context_" + "e5" * 16  # proj's, in another directory
KEY_F = "context_" + "f6" * 16  # gone's: registered, but its save was removed
KEY_G = "context_" + "97" * 16  # b's: registered at a path that now holds another save
KEY_H = "context_" + "08" * 16  # u's: registered, but its full log can't be read
KEY_I = "context_" + "1b" * 16  # a save whose name needs quoting
KEY_J = "context_" + "2c" * 16  # a save whose name isn't UTF-8
# make_workspace's roots, by number, as the pages name them.
LOCAL, OLD, PROJ, GONE, REPLACED, UNREADABLE, SECOND = "1", "2", "3", "4", "5", "6", "7"
LONG = "x" * 40_000
IMAGE = "data:image/png;base64," + "A" * 16_000
PATCH = "*** Begin Patch\n*** Update File: a.py\n-old\n+new\n*** End Patch"
USAGE = {"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280,
         "cached_input_tokens": 1000}
BOUNDARY = {"type": "user_interaction_boundary"}
SAMPLE_END = {"type": "model_sample_boundary"}
TOOLS = {"type": "tools", "specs": [{"name": "exec_command", "description": "Run a command.",
                                     "parameters": {"type": "object"}}]}
HREF = re.compile(r'(?:href|action)="([^"]*)"')
MENU = re.compile(r"<details class=node-menu>.*?</details>\n", re.S)
STYLESHEET = re.compile(r"<style>.*?</style>", re.S)
# python3 -m pythia.interaction.node, with Ctrl-C's handler. Python installs
# none when its parent ignores SIGINT, as a non-interactive shell's background
# job does.
# The program that the glui view runs, compiled (view.py).
PROGRAM = view.UI_DIRECTORY / "web" / "compiled" / "contexts.json"
RUN_NODE = ("import runpy, signal; signal.signal(signal.SIGINT, signal.default_int_handler); "
            "runpy.run_module('pythia.interaction.node', run_name='__main__', alter_sys=True)")


# Records, as pythia writes them: the type first.

def init(prefix_id, model="model-a"):
    return {"type": "init", "prefix_id": prefix_id, "model": model}


def message(role, content):
    return {"type": "message", "role": role, "content": content}


def call(name, call_id, arguments):
    return {"type": "tool_call", "name": name, "call_id": call_id,
            "arguments_json": json.dumps(arguments)}


def result(call_id, output, success=True):
    return {"type": "tool_result", "call_id": call_id, "output": output, "success": success}


def user_tool(name, call_id, output="ok"):
    return [{"type": "user_tool_call", "call": call(name, call_id, {})},
            {"type": "user_tool_result", "result": result(call_id, output)}]


def sample():
    return {"type": "sample_metadata", "usage": dict(USAGE), "elapsed_seconds": 1.5}


def summary():
    return {"type": "turn_summary", "input_tokens_sum": 2400, "output_tokens_sum": 160,
            "cached_input_tokens_sum": 2000, "cached_input_tokens_max": 1000,
            "non_cached_input_tokens_sum": 400, "context_tokens": 1280, "sample_count": 2,
            "compaction_count": 0, "elapsed_seconds": 75.0}


def failure():
    """A model request that failed, as the CLI records it."""
    return {"type": "model_failure", "category": "http_error", "message": "HTTP 500"}


def compaction(kind="coldstart"):
    return [{"type": "context_prefix", "prefix_items": [message("user", "Compacted.")]},
            {"type": "compaction_metadata", "usage": {name: 0 for name in USAGE},
             "protocol": kind, "elapsed_seconds": 0.02, "request_attempts": 0}]


def work(call_id="w1"):
    """A sample that calls a tool, and the tool's result: the model's work."""
    return [call("exec_command", call_id, {"cmd": "true"}), sample(), SAMPLE_END,
            result(call_id, "done")]


def answer(text="Done."):
    return [message("assistant", text), sample(), SAMPLE_END]


def turn(text, *, tools=1, ends=True, prefix="c"):
    """A user's turn: its message and boundary, the model's work, an answer, and a summary."""
    items = [message("user", text), BOUNDARY]
    for number in range(tools):
        items += work(f"{prefix}{number}")
    return items + answer() + ([summary()] if ends else [])


def write_log(path, items, tail=""):
    """A log of ``items``, records or raw lines, and ``tail``, a last line without its newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
             for item in items)
    path.write_bytes(("".join(line + "\n" for line in lines) + tail).encode("utf-8"))


def pointer(last):
    return (f"{pages.POINTER_PREFIX} `saves/local-0.jsonl` (relative to your tools' working "
            f"directory), lines 1\u2013{last}; later lines continue from here. A summary of "
            f"lines 1\u2013{last} is in `saves/local-0.summary.md`.\n\n"
            "The initial user message is on line 3:\n<user-message>\nFix the bug\n"
            "</user-message>")


def local_items():
    """local-0's full log: 63 lines in 4 turns, with each kind of item."""
    items = [
        init(KEY_A),  # 1
        TOOLS,
        message("user", 'Fix the bug <script>alert(1)</script> & "quoted"'),  # 3
        BOUNDARY,
        {"type": "reasoning", "content": "", "summary": ["Looking at the files."],
         "encrypted_content": "abc"},  # 5
        call("exec_command", "t1", {"cmd": "ls", "yield_time_ms": 1000}),
        sample(), SAMPLE_END,
        result("t1", "a.py\nb.py"),  # 9
        call("apply_patch", "t2", {"patch": PATCH}),
        sample(), SAMPLE_END,
        result("t2", "Done!"),  # 13
        call("update_plan", "t3", {"plan": [{"step": "Read", "status": "completed"},
                                            {"step": "Fix", "status": "in_progress"}]}),
        sample(), SAMPLE_END,
        result("t3", "Plan updated"),  # 17
        message("assistant", "Fixed\u202e it.\u2028Next\U000e0041 \U0001f469\u200d\U0001f4bb"),
        sample(), SAMPLE_END,
        summary(),  # 21: turn 1 ends
        *user_tool("compact", "u1", "Context compacted using a coldstart pointer."),  # 22
        {"type": "context_prefix", "prefix_items": [message("user", pointer(21))]},  # 24
        compaction()[1],
        call("exec_command", "t4", {"cmd": "git diff"}),  # 26: the model works on
        sample(), SAMPLE_END,
        result("t4", "diff --git a/a.py b/a.py\n-old\n+new"),  # 29
        message("assistant", "Read the log."),
        sample(), SAMPLE_END,
        summary(),  # 33: turn 2, the /compact, ends
        message("user", "Next step."),  # 34
        BOUNDARY,
        {"type": "model_failure", "category": "ModelRateLimitError",
         "message": "Too many requests", "provider": "p", "http_status": 429,
         "attempt_count": 2, "event_types": ["a", "a", "b"]},  # 36
        SAMPLE_END,
        *user_tool("retry", "u2", "Retrying."),  # 38
        call("icron", "t5", {"call": "schedule('Check CI.', after=600)"}),  # 40
        sample(), SAMPLE_END,
        result("t5", "Job c1 scheduled."),
        call("exec_command", "t6", {"cmd": "cat big.txt"}),  # 44
        sample(), SAMPLE_END,
        result("t6", LONG),  # 47
        message("assistant", "All done."),
        sample(), SAMPLE_END,
        summary(),  # 51: turn 3 ends
        *user_tool("config", "u3", "save_log_reasoning = assistant"),  # 52
        message("user", [{"type": "text", "text": "Look at this."},
                         {"type": "media", "source_uri": IMAGE}]),  # 54
        BOUNDARY,
        {"type": "opaque_compaction", "payload": "xyz", "protocol": "provider"},  # 56
        {"type": "future_item", "x": 1},
        {"type": "nil"},
        "not json at all",  # 59
        json.dumps(message("assistant", "Lone \ud800 surrogate.")),  # 60: as an escape
        sample(), SAMPLE_END,
        summary(),  # 63
    ]
    assert len(items) == 63
    return items


def make_workspace(tmp):
    """A registry of root contexts, registered as the CLI registers them, and saves.

    Its roots, by number: 1, local-0, with every kind of item, and a summary;
    2, old, an older log; 3, proj, in another directory; 4, gone, whose save
    was removed; 5, b, whose path now holds another save; 6, u, whose full log
    can't be read; and 7, a second number of 1's, made by hand. Not roots: a
    copy of local-0, with its key, and a save that was never registered.
    """
    tmp = Path(tmp)
    saves, elsewhere, copy = tmp / "saves", tmp / "elsewhere", tmp / "copy"
    local = local_items()
    write_log(saves / "local-0._private" / "context.jsonl", local)
    write_log(saves / "local-0.jsonl", local)  # its save log
    write_log(saves / "local-0._private" / "trace.events.jsonl", [{"event": 1}])
    (saves / "local-0.summary.md").write_text(
        f"Covers lines 1\u201321 of session_{KEY_A[8:]}.\n\n## Goal\nFix the bug.\n")
    old = [
        json.dumps({"session_id": "session_" + KEY_B[8:], "type": "session_init"}),
        json.dumps({"role": "user", "text": "Old question.", "type": "message"}),
        json.dumps(BOUNDARY),
        json.dumps({"role": "assistant", "text": "Old answer.", "type": "message"}),
        json.dumps({"provider_turn_id": "r1", "type": "turn_metadata", "usage": USAGE},
                   sort_keys=True),
        json.dumps({"replacement_items": [{"role": "user", "text": "Earlier: it was fixed.",
                                           "type": "message"}],
                    "type": "context_compaction"}),
    ]
    write_log(saves / "old._private" / "context.jsonl", old)
    write_log(saves / "old.jsonl", old[:1])
    write_log(elsewhere / "proj._private" / "context.jsonl",
              [init(KEY_E, "model-e"), *turn("Elsewhere.")])
    write_log(elsewhere / "proj.jsonl", [init(KEY_E, "model-e")])
    for name, key in (("gone", KEY_F), ("b", KEY_G), ("u", KEY_H)):
        write_log(saves / f"{name}._private" / "context.jsonl", [init(key), *turn("Hi.")])
        write_log(saves / f"{name}.jsonl", [init(key)])
    registry = tmp / "registry"
    roots = Registry(registry)
    for prefix_id, private in ((KEY_A, saves / "local-0._private"),
                               ("session_" + KEY_B[8:], saves / "old._private"),
                               (KEY_E, elsewhere / "proj._private"),
                               (KEY_F, saves / "gone._private"),
                               (KEY_G, saves / "b._private"),
                               (KEY_H, saves / "u._private")):
        registration = roots.register(prefix_id, private)
        assert registration.status == "new", registration
    os.symlink(f"{KEY_A}.jsonl", registry / f"{SECOND}.jsonl")  # a second number, by hand
    shutil.rmtree(saves / "gone._private")
    (saves / "gone.jsonl").unlink()
    write_log(saves / "b._private" / "context.jsonl",
              [init(KEY_C), message("user", "TOP SECRET")])
    write_log(saves / "u._private" / "context.jsonl",
              ["not json", message("user", "TOP SECRET")])
    write_log(copy / "local-0._private" / "context.jsonl",
              [init(KEY_A), message("user", "TOP SECRET")])
    write_log(saves / "secret._private" / "context.jsonl",
              [init(KEY_D), message("user", "TOP SECRET")])
    write_log(saves / "secret.jsonl", [init(KEY_D)])
    return types.SimpleNamespace(tmp=tmp, saves=saves, elsewhere=elsewhere, copy=copy,
                                 registry=registry)


def log_context(private, key=KEY_A):
    """Root 1, unregistered: the save whose private directory is ``private``, refreshed."""
    return contexts.Context(1, key, private).refresh()


def step_problems(program, steps):
    """Why the compiled ``program`` would refuse ``steps`` from its host, which only add
    facts, as its runtimes check them (_pythia_ui/src/lem/host.rs): a line each, none
    when it takes them all."""
    rels = {rel["name"]: rel for rel in program["rels"]}
    nullary = {}  # a type's constructors that take no fields: Tone's User, ...
    for ctor in program["ctors"]:
        if not ctor["fields"]:
            nullary.setdefault(ctor["type"], set()).add(ctor["name"])
    problems, keys = [], set()
    for step in steps:
        if not (isinstance(step, list) and len(step) == 3 and step[0] == "+"
                and isinstance(step[2], dict)):
            problems.append(f"{step!r} doesn't add a fact")
            continue
        _, name, fields = step
        rel = rels.get(name)
        if rel is None or rel["kind"] != "input":
            problems.append(f"{name} isn't an input")
            continue
        problems += [f"{name} has no field {extra}"
                     for extra in sorted(set(fields) - {field[0] for field in rel["fields"]})]
        values = []
        for field in rel["fields"]:
            if field[0] not in fields:
                if len(field) < 3:
                    problems.append(f"{name} needs its {field[0]}")
                values.append(field[2] if len(field) > 2 else None)
                continue
            value = fields[field[0]]
            values.append(value)
            kind = field[1]
            while program["types"].get(kind) is not None:  # an alias: CtxId is a Text
                kind = program["types"][kind]
            if kind == "Int":
                good = (isinstance(value, int) and not isinstance(value, bool)
                        and abs(value) <= 2 ** 53 - 1)
            elif kind in ("Text", "Bool"):
                good = isinstance(value, str if kind == "Text" else bool)
            else:
                good = isinstance(value, str) and value in nullary.get(kind, ())
            if not good:
                problems.append(f"{name}'s {field[0]} is a {field[1]}, not {value!r}")
        if rel["key"]:
            key = (name, json.dumps([values[index] for index in rel["key"]]))
            if key in keys:
                problems.append(f"two {name} facts have the key {key[1]}")
            keys.add(key)
    return problems


# Runs the view's pages in a JavaScript shell, after lem.js, glem.js, glui.js,
# the stand-in DOM, contexts.js and NODE, the program and the facts: each
# page, as contexts.start shows it, with every turn opened and its items
# given, then every entry that has a body opened and its body given. Prints
# NODE and what each page showed, and what it said went wrong, as JSON.
JS_RUN = """
(function () {
  "use strict";
  function out(text) {
    if (typeof print === "function" && typeof window === "undefined") {
      print(text);
    } else {
      console.log(text);
    }
  }
  function run(facts, runs, bodies) {
    var doc = new standInDom.Document();
    var waiting = new Map();
    var wrong = [];
    var shown = contexts.start({
      root: doc.createElement("div"), program: NODE.program, facts: "facts",
      before: "before/{line}", after: "after/{line}", run: "run/{run}", body: "body/{line}",
      get: function (url, done) { waiting.set(url, done); },
      status: function (text, bad) { if (bad) { wrong.push(text); } }
    });
    waiting.get("facts")(null, facts);
    var ran = 0;
    Object.keys(runs).forEach(function (line) {
      shown.send(["Toggled", ["RunBox", Number(line)], true]);
      var done = waiting.get("run/" + line);
      if (done === undefined) {
        wrong.push("run " + line + " asked for no items");
        return;
      }
      done(null, runs[line]);
      ran += 1;
    });
    var opened = 0;
    Object.keys(bodies).forEach(function (line) {
      shown.send(["Toggled", ["Ent", Number(line)], true]);
      var done = waiting.get("body/" + line);
      if (done === undefined) {
        wrong.push("line " + line + " asked for no body");
        return;
      }
      done(null, bodies[line]);
      opened += 1;
    });
    function count(name) {
      var prog = shown.runtime.prog.lem;
      return shown.runtime.lem.relFacts(prog.rel(name)).length;
    }
    return { wrong: wrong, entries: count("Entry"), codes: count("Code"),
             rows: count("TableRow"), ran: ran, opened: opened };
  }
  out("NODE " + JSON.stringify({ index: run(NODE.index, {}, {}),
                                 log: run(NODE.log, NODE.runs, NODE.bodies) }));
})();
"""


def request(port, path, *, method="GET", hosts=None, headers=(), body=None):
    """(status, headers by lowercase name, body); ``hosts``: the Host headers to send;
    ``body``: bytes to send, after the headers, which say how long it is unless they say
    so themselves."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    try:
        connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for host in ([f"127.0.0.1:{port}"] if hosts is None else hosts):
            connection.putheader("Host", host)
        for name, value in headers:
            connection.putheader(name, value)
        if body is not None and not any(name.lower() == "content-length" for name, _ in headers):
            connection.putheader("Content-Length", str(len(body)))
        connection.endheaders(body)
        response = connection.getresponse()
        body = response.read()
        return (response.status, {name.lower(): value for name, value in response.getheaders()},
                body)
    finally:
        connection.close()


def crawl(port, start, *, keep=lambda link: True, look=lambda path, body: None):
    """Every page that ``start`` leads to, through its links and redirects: {path: status}.

    ``look`` sees each page's path and body.
    """
    statuses, todo = {}, [start]
    while todo:
        path = todo.pop()
        if path in statuses:
            continue
        status, headers, body = request(port, path)
        statuses[path] = status
        look(path, body)
        if status in (303, 308):
            todo.append(headers["location"])
        elif status == 200 and headers["content-type"].startswith("text/html"):
            for link in HREF.findall(body.decode("utf-8")):
                link = html.unescape(link).split("#", 1)[0]
                if link.startswith(f"/{TOKEN}/") and keep(link):
                    todo.append(link)
    return statuses


def snapshot(*roots):
    """Each entry under ``roots``: a link's target, a file's bytes and times, a directory's."""
    state = {}
    for root in roots:
        for directory, dirs, files in os.walk(root):
            for name in dirs + files:
                path = os.path.join(directory, name)
                info = os.lstat(path)
                if stat.S_ISLNK(info.st_mode):
                    state[path] = ("link", os.readlink(path))
                elif stat.S_ISREG(info.st_mode):
                    with open(path, "rb") as file:
                        state[path] = ("file", file.read(), info.st_mtime_ns, info.st_mode)
                else:
                    state[path] = ("other", info.st_mtime_ns, info.st_mode)
    return state


def run_main(*argv):
    """main's status, and what it printed: (status, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            status = main(list(argv))
        except SystemExit as exc:
            status = exc.code
    return status, out.getvalue(), err.getvalue()


def outside(body):
    """A page without its stylesheet and its style menu."""
    return MENU.sub("", STYLESHEET.sub("", body))


@contextlib.contextmanager
def chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def environ(**changes) -> dict:
    """This process's environment, with the checkout importable."""
    return {**os.environ, "PYTHONPATH": str(CHECKOUT), **changes}


def unprivileged_ports() -> int:
    """The lowest port that any user may bind on Linux, else 1024."""
    try:
        with open("/proc/sys/net/ipv4/ip_unprivileged_port_start") as setting:
            return int(setting.read())
    except (OSError, ValueError):
        return 1024


class WorkspaceTest(unittest.TestCase):
    """A test with its own workspace (make_workspace), made once for its class."""

    @classmethod
    def setUpClass(cls):
        cls.ws = make_workspace(tempfile.mkdtemp(prefix="node-test-"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.ws.tmp)

    def pages(self):
        return pages.Pages(contexts.Roots(self.ws.registry), TOKEN)


class KeyTests(unittest.TestCase):
    def test_keys(self):
        self.assertEqual(contexts.key_of(KEY_A), KEY_A)
        self.assertEqual(contexts.key_of("session_" + KEY_B[8:]), KEY_B)
        for other in ("context_" + "a" * 31, "session_" + "A" * 32, "other", 7, None,
                      KEY_A + "\n"):
            self.assertIsNone(contexts.key_of(other), other)


class RootTests(WorkspaceTest):
    """Which of a registry's roots are shown, and what each holds."""

    def test_only_roots_where_they_are_registered_are_shown(self):
        roots = contexts.Roots(self.ws.registry)
        self.assertIsNone(roots.problem())
        # 7, a second number of 1's, isn't a root of its own.
        self.assertEqual([(entry.number, entry.state) for entry in roots.entries()],
                         [(1, "ok"), (2, "ok"), (3, "ok"), (4, "missing"), (5, "replaced"),
                          (6, "unreadable")])
        self.assertEqual([context.id for context in roots.contexts()], [LOCAL, OLD, PROJ])
        for cid in (GONE, REPLACED, UNREADABLE, SECOND, "8", "01", "x", "../1", ""):
            self.assertIsNone(roots.get(cid), cid)
        self.assertEqual(roots.entry(SECOND).first, 1)
        self.assertIsNone(roots.entry("8"))

    def test_keys_names_models_and_titles(self):
        found = {context.id: context.refresh()
                 for context in contexts.Roots(self.ws.registry).contexts()}
        self.assertEqual({cid: (context.key, context.name, context.label)
                          for cid, context in found.items()},
                         {LOCAL: (KEY_A, "local-0", "local-0 (1)"),
                          OLD: (KEY_B, "old", "old (2)"), PROJ: (KEY_E, "proj", "proj (3)")})
        self.assertEqual(found[OLD].prefix_id, "session_" + KEY_B[8:])
        self.assertIsNone(found[OLD].model)
        self.assertEqual(found[PROJ].model, "model-e")
        self.assertEqual(found[OLD].title(), "Old question.")  # text, in place of content
        self.assertEqual(found[LOCAL].title(),
                         'Fix the bug <script>alert(1)</script> & "quoted"')
        real = Path(os.path.realpath(self.ws.tmp))
        self.assertEqual(found[PROJ].private, real / "elsewhere" / "proj._private")

    def test_the_files_that_go_with_a_root(self):
        roots = contexts.Roots(self.ws.registry)
        saves = Path(os.path.realpath(self.ws.saves))
        self.assertEqual(roots.get(LOCAL).files(), [
            ("full log", saves / "local-0._private" / "context.jsonl"),
            ("save log", saves / "local-0.jsonl"),
            ("summary", saves / "local-0.summary.md"),
            ("trace", saves / "local-0._private" / "trace.events.jsonl")])
        self.assertEqual([what for what, _ in roots.get(OLD).files()], ["full log", "save log"])

    def test_a_root_that_moves_is_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            write_log(tmp / "a" / "s._private" / "context.jsonl", [init(KEY_A), *turn("Here.")])
            registry = Registry(tmp / "registry")
            registry.register(KEY_A, tmp / "a" / "s._private")
            roots = contexts.Roots(tmp / "registry")
            first = roots.get("1")
            self.assertIs(roots.get("1"), first)  # with its index
            self.assertEqual(first.refresh().title(), "Here.")
            os.rename(tmp / "a", tmp / "b")
            self.assertEqual(registry.register(KEY_A, tmp / "b" / "s._private").status, "moved")
            moved = roots.get("1")
            self.assertIsNot(moved, first)
            self.assertEqual(moved.private, tmp / "b" / "s._private")
            self.assertEqual(moved.refresh().title(), "Here.")
            os.rename(tmp / "b", tmp / "c")  # until the CLI opens it again, it's missing
            self.assertIsNone(roots.get("1"))
            self.assertEqual([entry.state for entry in roots.entries()], ["missing"])

    def test_a_registry_that_cant_be_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            nowhere = Path(tmp) / "root"
            roots = contexts.Roots(nowhere)
            self.assertEqual(roots.problem(),
                             f"There's no registry at {contexts.shown_path(nowhere)}/ yet.")
            self.assertEqual((roots.entries(), roots.contexts(), roots.get("1")), ([], [], None))
            self.assertFalse(nowhere.exists())  # nothing is made
            if os.geteuid() != 0:
                locked = Path(tmp) / "locked"
                locked.mkdir(mode=0)
                try:
                    roots = contexts.Roots(locked)
                    self.assertEqual(roots.problem(), f"The registry {contexts.shown_path(locked)}"
                                     "/ can't be read: Permission denied.")
                    self.assertEqual((roots.entries(), roots.contexts()), ([], []))
                finally:
                    locked.chmod(0o700)


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="node-test-"))
        self.private = self.tmp / "s._private"
        self.path = self.private / "context.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def context(self):
        return log_context(self.private)

    def test_no_two_indexes_have_a_generation_alike(self):
        # So that a page of the glui view, which names its log's generation as it asks what
        # the log gained, learns that a root's log is another (view.py).
        write_log(self.path, [init(KEY_A)])
        write_log(self.tmp / "t._private" / "context.jsonl", [init(KEY_B)])
        one = contexts.Index(self.path).refresh()
        other = contexts.Index(self.tmp / "t._private" / "context.jsonl").refresh()
        self.assertNotEqual(one.generation, other.generation)

    def test_kinds_and_calls_match_json(self):
        write_log(self.path, [
            init(KEY_A), message("user", "Hi"), call('a"b', "id\u00e9\\x", {}), result("i", "x"),
            json.dumps({"role": "assistant", "text": "Old.", "type": "message"}),
            json.dumps({"call": call("compact", "u1", {}), "type": "user_tool_call"}),
            {"type": "user_tool_call", "call": call("config", "u2", {})},
            {"type": "message", "content": "no role"}, message("Custom Role", "x"),
            {"type": "future_item"}, "[1, 2]", "not json", '{"type": 5}', "{}",
            '{"type": "tool_call", "call_id": "late", "name": "sorted"}'])
        index = self.context().index
        lines = self.path.read_bytes().split(b"\n")[:-1]
        self.assertEqual(len(index), len(lines))
        for number, line in enumerate(lines):
            try:
                expected = contexts.record_kind(json.loads(line))[0]
            except ValueError:
                expected = (contexts.BAD, None)
            self.assertEqual(index.kind(number), expected, line)
        self.assertEqual(index.calls, {"id\u00e9\\x": 'a"b', "late": "sorted"})
        self.assertEqual(index.user_calls, {"u1": "compact", "u2": "config"})

    def test_a_line_separator_inside_a_string_ends_no_line(self):
        text = "one\u2028two\u2029three\x85four\x0bfive\x1cend"
        write_log(self.path, [init(KEY_A), message("user", text), message("assistant", "ok")])
        self.assertIn("\u2028".encode(), self.path.read_bytes())  # raw, as older logs hold it
        context = self.context()
        self.assertEqual(len(context), 3)
        self.assertEqual(context.records(1, 2), [(2, message("user", text))])

    def test_a_last_line_waits_for_its_newline(self):
        write_log(self.path, [init(KEY_A), message("user", "Hi")],
                  tail='{"type": "message", "role": "assis')
        context = self.context()
        index = context.index
        self.assertEqual(len(index), 2)
        generation = index.generation
        with open(self.path, "ab") as log:
            log.write(b'tant", "content": "ok"}\n')
        index.refresh()
        self.assertEqual(len(index), 3)
        self.assertEqual(index.kind(2), ("message", "assistant"))
        self.assertEqual(context.records(2, 3), [(3, message("assistant", "ok"))])
        self.assertEqual(index.generation, generation)  # appended to, not indexed anew

    def test_appended_lines_alone_are_indexed(self):
        write_log(self.path, [init(KEY_A), *turn("One.")])
        index = self.context().index
        offsets, generation = list(index.offsets), index.generation
        with open(self.path, "a") as log:
            log.write("".join(json.dumps(item) + "\n" for item in turn("Two.")))
        with mock.patch.object(contexts, "line_kind", wraps=contexts.line_kind) as kinds:
            index.refresh()
        self.assertEqual(kinds.call_count, len(turn("Two.")))
        self.assertEqual(list(index.offsets[:len(offsets)]), offsets)
        self.assertEqual(index.generation, generation)
        with mock.patch.object(contexts, "line_kind", wraps=contexts.line_kind) as kinds:
            index.refresh()  # unchanged
        self.assertEqual(kinds.call_count, 0)

    def test_a_truncated_log_is_indexed_anew(self):
        write_log(self.path, [init(KEY_A), message("user", "Hi"), message("assistant", "Yo")])
        index = self.context().index
        generation = index.generation
        with open(self.path, "r+b") as log:
            log.truncate(index.offsets[1])
        index.refresh()
        self.assertEqual(len(index), 1)
        self.assertGreater(index.generation, generation)

    def test_a_replaced_log_is_indexed_anew(self):
        write_log(self.path, [init(KEY_A), message("user", "Hi")])
        index = self.context().index
        generation = index.generation
        write_log(self.tmp / "new.jsonl", [init(KEY_C), message("assistant", "New."),
                                           message("user", "And more.")])
        os.replace(self.tmp / "new.jsonl", self.path)
        with self.assertRaises(OSError):
            index.read(0, 1)  # not the log that was indexed
        index.refresh()
        self.assertGreater(index.generation, generation)
        self.assertEqual([index.kind(line) for line in range(len(index))],
                         [("init", None), ("message", "assistant"), ("message", "user")])

    def test_a_log_rewritten_in_place_is_indexed_anew(self):
        write_log(self.path, [init(KEY_A), message("user", "Hi")])
        index = self.context().index
        generation = index.generation
        rewritten = b"".join(json.dumps(item).encode() + b"\n" for item in (
            init(KEY_C), message("assistant", "Rewritten, and longer."), sample()))
        with open(self.path, "r+b") as log:  # the same inode, longer
            log.write(rewritten)
        index.refresh()
        self.assertGreater(index.generation, generation)
        self.assertEqual(len(index), 3)

    def test_a_log_replaced_by_a_longer_copy_is_read(self):
        # Its first bytes are the same, so only its new inode tells it apart.
        items = [init(KEY_A), message("user", "Hi")]
        write_log(self.path, items)
        index = self.context().index
        write_log(self.tmp / "new.jsonl", [*items, message("assistant", "More.")])
        os.replace(self.tmp / "new.jsonl", self.path)
        index.refresh()
        self.assertEqual(len(index), 3)
        self.assertEqual(index.read(2, 3), [json.dumps(message("assistant", "More.")).encode()])

    def test_a_new_inode_is_indexed_anew_even_where_its_end_matches(self):
        # A rewrite that kept each line's length, renamed into place: the old end's
        # last bytes match, so only the new inode says that the titles may differ.
        last = message("assistant", "Yo" * 40)  # longer than the tail that's compared
        write_log(self.path, [init(KEY_A), message("user", "Hi"), last])
        context = self.context()
        self.assertEqual(context.title(), "Hi")
        generation = context.index.generation
        write_log(self.tmp / "new.jsonl",
                  [init(KEY_A), message("user", "Ho"), last, message("user", "More.")])
        os.replace(self.tmp / "new.jsonl", self.path)
        context.refresh()
        self.assertGreater(context.index.generation, generation)
        self.assertEqual((len(context), context.title()), (4, "Ho"))

    def test_a_bad_line_shows_raw(self):
        write_log(self.path, [init(KEY_A), "not json", message("user", "Hi")])
        context = self.context()
        self.assertEqual(context.index.kind(1), (contexts.BAD, None))
        self.assertEqual(context.records(1, 2), [(2, {"type": contexts.BAD, "raw": "not json"})])


class TurnTests(unittest.TestCase):
    def turns(self, items):
        """Each turn's first line, from 1, and the turns' rows."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s._private" / "context.jsonl"
            write_log(path, items)
            context = log_context(path.parent)
            return [start + 1 for start in context.turns()], context.turn_rows()

    def test_a_message_and_its_boundary_start_a_turn(self):
        starts, rows = self.turns([init(KEY_A), TOOLS, *turn("One."), *turn("Two.")])
        self.assertEqual(starts, [3, 13])
        self.assertEqual([(row.label, row.samples, row.calls) for row in rows],
                         [("One.", 2, 1), ("Two.", 2, 1)])
        self.assertEqual(rows[0].summary["input_tokens_sum"], 2400)

    def test_retry_and_steer_stay_in_their_turn(self):
        starts, rows = self.turns([
            init(KEY_A), message("user", "One."), BOUNDARY,
            {"type": "model_failure", "category": "E", "message": "failed"}, SAMPLE_END,
            *user_tool("retry", "u1"), *work("w1"), *user_tool("steer", "u2"),
            message("user", "Also this."), *answer(), summary(), *turn("Two.")])
        self.assertEqual(starts, [2, 19])
        self.assertEqual((rows[0].label, rows[0].failures), ("One.", 1))

    def test_config_between_turns_starts_the_next(self):
        starts, rows = self.turns([init(KEY_A), *turn("One."), *user_tool("config", "u1"),
                                   *turn("Two.")])
        self.assertEqual(starts, [2, 12])
        self.assertEqual(rows[1].label, "/config · Two.")

    def test_compact_with_work_after_it_is_a_turn_of_its_own(self):
        starts, rows = self.turns([
            init(KEY_A), *turn("One."), *user_tool("compact", "u1"), *compaction(),
            *work("w9"), *answer(), summary(), *turn("Two.")])
        self.assertEqual(starts, [2, 12, 24])
        self.assertEqual([row.label for row in rows], ["One.", "/compact", "Two."])

    def test_compact_before_a_message_joins_its_turn(self):
        starts, rows = self.turns([init(KEY_A), *turn("One."), *user_tool("compact", "u1"),
                                   *compaction(), *turn("Two.")])
        self.assertEqual(starts, [2, 12])
        self.assertEqual(rows[1].label, "/compact · Two.")

    def test_turns_without_summaries(self):
        starts, _ = self.turns([init(KEY_A), *turn("One.", ends=False),
                                *turn("Two.", ends=False)])
        self.assertEqual(starts, [2, 11])

    def test_the_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s._private" / "context.jsonl"
            write_log(path, [init(KEY_A), {"type": "instructions", "text": "Be brief."}, TOOLS,
                             *turn("One.")])
            context = log_context(path.parent)
            self.assertEqual(context.turn_range(0), (0, 3))
            self.assertEqual(context.turn_range(1), (3, 13))
            self.assertEqual([context.turn_of(line) for line in (0, 2, 3, 12)], [0, 0, 1, 1])
            write_log(path, [init(KEY_A), TOOLS])  # no turn yet: all start
            context = log_context(path.parent)
            self.assertEqual((context.turns(), context.turn_range(0)), ([], (0, 2)))


class RenderTests(WorkspaceTest):
    def page(self, cid, **params):
        response = self.pages().context(cid, {name: str(value) for name, value in params.items()})
        self.assertEqual(response.status, 200, response.body)
        response.body.encode("utf-8")  # strictly: no lone surrogate is left
        return response.body

    def test_the_first_turn(self):
        body = self.page(LOCAL, turn=1)
        for shown in ("[user]", "[reasoning]", "Looking at the files.",
                      "encrypted content, 3 characters", "[tool-call] exec_command",
                      "yield_time_ms=1000", "<pre>ls</pre>", "[tool-ret] exec_command [ok]",
                      "[tool-call] apply_patch", "<span class=d-add>+new</span>",
                      "<span class=d-del>-old</span>", "[tool-call] update_plan",
                      "[x] Read\n[&gt;] Fix", "[tool-ret] update_plan [ok]", "[assistant]",
                      "[sample]", "input=1200 output=80 total=1280 cached=1000 elapsed=1.50s",
                      "[turn]", "input_sum=2400 output_sum=160 cold_sum=400 cached_sum=2000 "
                      "cached_max=1000 context=1280 samples=2 compactions=0 elapsed=75.00s"):
            self.assertIn(shown, body)
        self.assertIn('title="t1">[tool-call] exec_command', body)  # the call ID, as a tooltip
        self.assertNotIn("model_sample_boundary", body)

    def test_text_is_inert_and_invisible_characters_show(self):
        body = self.page(LOCAL, turn=1)
        self.assertNotIn("<script>", body)
        self.assertIn("Fix the bug &lt;script&gt;alert(1)&lt;/script&gt; &amp; "
                      "&quot;quoted&quot;", body)
        for code in ("\\u202e", "\\u2028", "\\U000e0041"):
            self.assertIn(f"<span class=esc>{code}</span>", body)
        for char in ("\u202e", "\u2028", "\U000e0041"):
            self.assertNotIn(char, body)
        self.assertIn("\U0001f469\u200d\U0001f4bb", body)  # the joiner stays
        self.assertIn("<span class=esc>\\ud800</span>", self.page(LOCAL, turn=4))
        self.assertEqual(pages.h("a\x1b[31m<b>\tc\n"),
                         "a<span class=esc>\\u001b</span>[31m&lt;b&gt;\tc\n")

    def test_a_coldstart_pointer_links_what_it_names(self):
        body = self.page(LOCAL, turn=2)
        for shown in ("[user-tool-call] compact", "[user-tool-ret] compact [ok]",
                      "Context compacted using a coldstart pointer.", "[context prefix] 1 item",
                      "[compaction]", "protocol=coldstart input=0 output=0 total=0 cached=0 "
                      "elapsed=0.02s attempts=0",
                      f'<a href="/{TOKEN}/c/1">saves/local-0.jsonl</a>',
                      f'<a href="/{TOKEN}/c/1?lines=1-21">lines 1\u201321</a>',
                      f'<a href="/{TOKEN}/c/1?summary">saves/local-0.summary.md</a>',
                      f'<a href="/{TOKEN}/c/1?line=3">line 3</a>',
                      "<span class=d-file>diff --git a/a.py b/a.py</span>"):
            self.assertIn(shown, body)

    def test_failures_user_tools_icron_and_long_blocks(self):
        body = self.page(LOCAL, turn=3)
        for shown in ("[model]", "Too many requests kind=ModelRateLimitError", "4 more fields",
                      "status=429", "attempts=2", "event_types=a(2),b",
                      "[user-tool-call] retry", "[user-tool-ret] retry [ok]",
                      "[tool-call] icron", html.escape("icron.schedule('Check CI.', after=600)"),
                      "… 36,000 more characters",
                      f'<a href="/{TOKEN}/c/1?line=47">the whole item</a>'):
            self.assertIn(shown, body)
        self.assertNotIn(LONG, body)
        line = self.page(LOCAL, line=47)
        self.assertIn(LONG, line)
        self.assertNotIn("more characters", line)

    def test_media_unknown_and_bad_lines(self):
        body = self.page(LOCAL, turn=4)
        for shown in ("[user-tool-call] config", "Look at this.", "[image] image/png 11.7 KiB",
                      "opaque checkpoint, protocol=provider, 3 characters", "[future_item]",
                      "&quot;x&quot;: 1", "[nil]", "[bad line]", "not json at all"):
            self.assertIn(shown, body)
        self.assertNotIn("AAAAAAAA", body)  # an image's payload is never shown

    def test_a_result_names_its_tool_from_a_call_on_another_page(self):
        self.assertIn("[tool-ret] exec_command [ok]", self.page(LOCAL, lines="47-47"))
        self.assertIn(f'<a class=ln href="/{TOKEN}/c/1?line=47">47</a>',
                      self.page(LOCAL, lines="47"))

    def test_older_logs(self):
        start = self.page(OLD, turn=0)
        self.assertIn("[session_init]", start)
        self.assertIn("session_" + KEY_B[8:], start)
        body = self.page(OLD, turn=1)
        for shown in ("[user]", "Old question.", "[assistant]", "Old answer.",
                      "[sample]", "input=1200 output=80 total=1280 cached=1000",
                      "[context prefix] 1 item", "Earlier: it was fixed."):
            self.assertIn(shown, body)
        rows = self.pages().roots.get(OLD).refresh().turn_rows()
        self.assertEqual([(row.label, row.samples) for row in rows], [("Old question.", 1)])

    def test_a_context_page(self):
        body = self.page(LOCAL)
        saves = Path(os.path.realpath(self.ws.saves))
        for shown in ("<h1>local-0 (1)</h1>",
                      f"<dt>root context</dt><dd>1 in {contexts.shown_path(self.ws.registry)}/"
                      "</dd>", f"<dt>key</dt><dd>{KEY_A}</dd>",
                      f"<dt>save</dt><dd>{saves / 'local-0'}</dd>",
                      "The start: lines 1–2", "[init]", "[tools]</span> exec_command",
                      "<h2>4 turns</h2>", "Fix the bug &lt;script&gt;", ">/compact<",
                      ">Next step.<", ">/config · Look at this.<",
                      f'href="/{TOKEN}/c/1?lines=22-33"', "2.4K in · 160 out", "1m 15s",
                      str(saves / "local-0.summary.md"), "trace.events.jsonl",
                      f'action="/{TOKEN}/c/1"',
                      f'<dt>glui view</dt><dd><a href="/{TOKEN}/lem/c/1">its log, in glui&#x27;s '
                      "widgets</a></dd>"):
            self.assertIn(shown, body)
        old = self.page(OLD)
        for shown in ("<h1>old (2)</h1>", f"<dt>key</dt><dd>{KEY_B}</dd>",
                      f"<dt>prefix ID</dt><dd>session_{KEY_B[8:]}</dd>", "<h2>1 turn</h2>"):
            self.assertIn(shown, old)
        self.assertNotIn("<dt>model</dt>", old)

    def test_pages_of_a_long_turn(self):
        with mock.patch.object(pages, "PAGE_LINES", 5):
            first = self.page(LOCAL, turn=3)
            second = self.page(LOCAL, turn=3, page=2)
            line = self.page(LOCAL, line=44)
        self.assertIn("page 1 of 4: lines 34–38", first)
        self.assertIn(f'href="/{TOKEN}/c/1?turn=3&amp;page=2"', first)
        self.assertIn("page 2 of 4: lines 39–43", second)
        self.assertIn(f'href="/{TOKEN}/c/1?turn=3&amp;page=3#L44"', line)

    def test_the_summary_links_the_lines_it_covers(self):
        body = self.page(LOCAL, summary="")
        self.assertIn(f'<a href="/{TOKEN}/c/1?lines=1-21">lines 1\u201321</a>', body)

    def test_the_contexts_page(self):
        body = self.pages().contexts().body
        for shown in (f"<h1>Root contexts in {contexts.shown_path(self.ws.registry)}/</h1>",
                      "<p class=dim>6 root contexts, the last changed first. The 3 that "
                      "aren&#x27;t where they&#x27;re registered come last.</p>",
                      f'<tr><td class=r>1</td><td class=nw><a href="/{TOKEN}/c/1">local-0</a>',
                      f'<a href="/{TOKEN}/c/3">proj</a>', f'title="{KEY_A}">context_a1a1</span>',
                      "model-e", "Old question.", "<span class=mono>context_f6f6</span>",
                      "Nothing is at its path any more", "Its path now holds another save.",
                      "Its full log can&#x27;t be read.",
                      f'<p><a href="/{TOKEN}/lem/">In glui&#x27;s view</a></p>'):
            self.assertIn(shown, body)
        self.assertNotIn("TOP SECRET", body)
        numbers = re.findall(r"<tr(?: class=dim)?><td class=r>([0-9]+)</td>", body)
        self.assertEqual(sorted(numbers[:3]), [LOCAL, OLD, PROJ])  # the last changed first
        self.assertEqual(numbers[3:], [GONE, REPLACED, UNREADABLE])
        with tempfile.TemporaryDirectory() as tmp:
            nowhere = Path(tmp) / "root"
            body = pages.Pages(contexts.Roots(nowhere), TOKEN).contexts().body
            self.assertIn(f"There&#x27;s no registry at {contexts.shown_path(nowhere)}/ yet.",
                          body)
            self.assertNotIn("<table>", body)
            nowhere.mkdir()
            body = pages.Pages(contexts.Roots(nowhere), TOKEN).contexts().body
            self.assertIn("<p class=dim>None yet.</p>", body)
            self.assertNotIn("<table>", body)

    def test_keys(self):
        found = self.pages()
        self.assertEqual(found.key(KEY_E[:14]).location, f"/{TOKEN}/c/3")
        self.assertEqual(found.key("session_" + KEY_B[8:20]).location, f"/{TOKEN}/c/2")
        self.assertEqual(found.key(KEY_A).location, f"/{TOKEN}/c/1")
        for key, why in ((KEY_F[:12], "Nothing is at its path any more"),
                         (KEY_G[:12], "Its path now holds another save."),
                         (KEY_H[:12], "Its full log can&#x27;t be read.")):
            response = found.key(key)
            self.assertEqual(response.status, 404, key)
            self.assertIn(why, response.body)
        for key in (KEY_C[:12], KEY_D[:12], "context_ffff"):  # no root's key
            response = found.key(key)
            self.assertEqual(response.status, 404, key)
            self.assertIn("has a key that starts with", response.body)
        for key in ("context_zz", "context_a1a", "a1a1a1a1"):
            self.assertEqual(found.key(key).status, 400, key)

    def test_a_key_that_two_roots_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            registry = Registry(tmp / "registry")
            for name, key in (("x", KEY_A), ("y", KEY_A[:12] + "0" * 28)):
                write_log(tmp / f"{name}._private" / "context.jsonl", [init(key), *turn(name)])
                registry.register(key, tmp / f"{name}._private")
            found = pages.Pages(contexts.Roots(tmp / "registry"), TOKEN)
            shared = found.key("context_a1a1")
            self.assertEqual(shared.status, 200)
            self.assertIn("2 root contexts have a key that starts with context_a1a1", shared.body)
            self.assertEqual(found.key("context_a1a1a").location, f"/{TOKEN}/c/1")
            self.assertIn(f'title="{KEY_A}">context_a1a1a</span>', found.contexts().body)

    def test_roots_that_arent_shown_say_why(self):
        found = self.pages()
        saves = contexts.shown_path(os.path.realpath(self.ws.saves))
        for cid, name, why in ((GONE, "gone", "Nothing is at its path any more"),
                               (REPLACED, "b", "Its path now holds another save."),
                               (UNREADABLE, "u", "Its full log can&#x27;t be read.")):
            response = found.context(cid, {"turn": "1"})
            self.assertEqual(response.status, 404, cid)
            self.assertIn(f"Root context {cid}, {saves}/{name}, isn&#x27;t shown. {why}",
                          response.body)
            self.assertNotIn("TOP SECRET", response.body)
        self.assertIn("There&#x27;s no root context 8 in", found.context("8", {}).body)

    def test_a_second_number_leads_to_its_root(self):
        found = self.pages()
        self.assertEqual(found.context(SECOND, {}).location, f"/{TOKEN}/c/1")
        self.assertEqual(found.context(SECOND, {"turn": "2", "page": "1"}).location,
                         f"/{TOKEN}/c/1?turn=2&page=1")
        self.assertEqual(found.view(style="chat").context(SECOND, {"line": "3"}).location,
                         f"/{TOKEN}/c/1?line=3&style=chat")

    def test_queries_that_name_nothing(self):
        found = self.pages()
        for params, status in (({"turn": "9"}, 404), ({"turn": "x"}, 400), ({"line": "0"}, 404),
                               ({"line": "64"}, 404), ({"lines": "5-2"}, 400),
                               ({"lines": "70-80"}, 404), ({"turn": "1", "page": "2"}, 404)):
            self.assertEqual(found.context(LOCAL, params).status, status, params)
        self.assertEqual(found.context(OLD, {"summary": ""}).status, 404)
        for cid in ("99", "nothing", "01", "1/", "-1"):
            self.assertEqual(found.context(cid, {}).status, 404, cid)


class StyleTests(WorkspaceTest):
    """The menu, links that keep their page's style, and chat's folds."""

    def page(self, cid, style="chat", **params):
        view = self.pages().view(style=style)
        response = view.context(cid, {name: str(value) for name, value in params.items()})
        self.assertEqual(response.status, 200, response.body)
        return response.body

    def test_the_menu_links_this_page_in_each_style(self):
        view = self.pages().view(style="chat", here=(f"/{TOKEN}/c/1", [("turn", "3")]))
        menu = MENU.search(view.context(LOCAL, {"turn": "3"}).body).group(0)
        self.assertIn('<summary title="How the pages look">style: Chat</summary>', menu)
        self.assertIn(f'<li><a href="/{TOKEN}/c/1?turn=3">Log dark<span class=note>every '
                      "item, as the log has it; dark colors</span></a></li>", menu)
        self.assertIn(f'<li><a href="/{TOKEN}/c/1?turn=3&amp;style=log-light">'
                      "Log light<span class=note>", menu)
        self.assertIn(f'<li><a href="/{TOKEN}/c/1?turn=3&amp;style=chat" '
                      "aria-current=page>Chat<span class=note>", menu)
        self.assertEqual(menu.count("aria-current"), 1)
        for style in styles.BUILT_IN:
            body = view.view(style=style.name).context(LOCAL, {"turn": "3"}).body
            menu = MENU.search(body).group(0)
            self.assertIn(f"style: {style.label}</summary>", menu)
            self.assertIn(f"aria-current=page>{style.label}<span class=note>", menu)
            self.assertEqual(menu.count("aria-current"), 1)

    def test_every_page_has_the_menu(self):
        found = self.pages()
        for response in (found.contexts(), found.context(LOCAL, {}),
                         found.context(LOCAL, {"turn": "1"}),
                         found.context(LOCAL, {"line": "3"}),
                         found.context(LOCAL, {"lines": "1-5"}),
                         found.context(LOCAL, {"summary": ""}), found.key(KEY_F[:12]),
                         found.key("context_zz"), found.context(GONE, {}),
                         found.not_found("x"), found.bad_request("y")):
            self.assertEqual(len(MENU.findall(response.body)), 1, response.body[-300:])

    def test_links_keep_the_style(self):
        for style, kept in (("log-dark", False), ("log-light", True), ("chat", True)):
            bodies = (self.page(LOCAL, style), self.page(LOCAL, style, turn=2),
                      self.page(LOCAL, style, line=24),
                      self.page(LOCAL, style, lines="20-30"),
                      self.page(LOCAL, style, summary=""),
                      self.pages().view(style=style).contexts().body)
            for body in bodies:
                links = [html.unescape(link) for link in HREF.findall(outside(body))]
                self.assertGreaterEqual(len(links), 3)
                for link in links:
                    self.assertTrue(link.startswith(f"/{TOKEN}/"), link)
                    if link.startswith(f"/{TOKEN}/lem/"):
                        continue  # the glui view, whose looks are its own
                    self.assertEqual("style=" in link, kept, link)
                    if kept:
                        self.assertIn(f"style={style}", link)
            if kept:
                self.assertIn(f"<input type=hidden name=style value={style}>", bodies[0])
            else:
                self.assertNotIn("type=hidden", bodies[0])
            query = f"?style={style}" if kept else ""
            view = self.pages().view(style=style)
            self.assertEqual(view.key(KEY_E[:14]).location, f"/{TOKEN}/c/3{query}")
            self.assertEqual(view.context(SECOND, {}).location, f"/{TOKEN}/c/1{query}")

    def test_log_styles_render_the_same_pages(self):
        roots = self.pages().roots
        # Each as the default: links and forms need no style field, so the
        # markup outside the stylesheet and menu must be identical.
        dark, light = (pages.Pages(roots, TOKEN, style=name)
                       for name in ("log-dark", "log-light"))
        self.assertEqual(outside(dark.contexts().body), outside(light.contexts().body))
        for params in ({}, {"turn": "0"}, {"turn": "1"}, {"turn": "2"}, {"turn": "3"},
                       {"turn": "4"}, {"line": "5"}, {"line": "6"}, {"line": "24"},
                       {"line": "47"}, {"line": "61"}, {"lines": "1-63"}, {"summary": ""}):
            with self.subTest(params=params):
                a, b = dark.context(LOCAL, params), light.context(LOCAL, params)
                self.assertEqual((a.status, b.status), (200, 200))
                self.assertEqual(outside(a.body), outside(b.body))

    def test_item_types_are_classes(self):
        body = self.page(LOCAL, "log-dark", turn=4)
        for shown in ('<div class="it user t-message" id="L54">',
                      '<div class="it meta t-sample_metadata" id="L61">',
                      '<div class="it meta t-opaque_compaction" id="L56">',
                      '<div class="it t-other" id="L57">', '<div class="it t-other" id="L58">',
                      '<div class="it err t-bad" id="L59">'):
            self.assertIn(shown, body)
        self.assertNotIn("<details class=\"it", body)  # the log style folds nothing

    def test_chat_folds_the_models_work(self):
        body = self.page(LOCAL, turn=1)
        self.assertIn('<details class="it tool t-tool_call" id="L6"><summary>'
                      f'<a class=ln href="/{TOKEN}/c/1?line=6&amp;style=chat">6</a>'
                      '<span class=lb title="t1">[tool-call] exec_command</span> '
                      "<span class=gist>ls</span></summary>", body)
        for line, label, line_gist in (
                (9, "[tool-ret] exec_command [ok]", "a.py … 2 lines"),
                (10, "[tool-call] apply_patch", "a.py"),
                (13, "[tool-ret] apply_patch [ok]", "Done!"),
                (14, "[tool-call] update_plan", "[x] Read  [&gt;] Fix"),
                (5, "[reasoning]", "Looking at the files.")):
            self.assertRegex(body, re.escape(f'id="L{line}"><summary>') + ".*?"
                             + re.escape(f"{label}</span> <span class=gist>{line_gist}</span>"
                                         "</summary>"))
        # Reasoning that folds holds no fold of its own.
        self.assertNotIn("<details><summary>Looking at the files.</summary>", body)
        self.assertIn('<div class="it user t-message" id="L3">', body)
        self.assertIn('<div class="it assistant t-message" id="L18">', body)
        compact = self.page(LOCAL, turn=2)
        self.assertIn('<details class="it t-context_prefix" id="L24"><summary>', compact)
        self.assertIn("<span class=gist>This conversation was compacted.", compact)
        self.assertIn(f'<a href="/{TOKEN}/c/1?lines=1-21&amp;style=chat">', compact)
        self.assertIn("<span class=gist>diff --git a/a.py b/a.py … 3 lines</span>", compact)
        failures = self.page(LOCAL, turn=3)
        self.assertIn('<div class="it err t-model_failure" id="L36">', failures)
        self.assertIn("<span class=gist>icron.schedule(&#x27;Check CI.&#x27;, after=600)</span>",
                      failures)
        self.assertIn("… 36,000 more characters", failures)

    def test_a_line_shows_whole_in_every_style(self):
        for style in styles.BUILT_IN:
            for line in (5, 6, 24, 47):
                body = outside(self.page(LOCAL, style.name, line=line))
                self.assertIn(f'<div class="it', body)
                self.assertNotIn(f'<details class="it', body)
            self.assertIn(LONG, self.page(LOCAL, style.name, line=47))

    def test_gists(self):
        exec_output = ("Chunk ID: 1\nWall time: 0.0100 seconds\nProcess exited with code 2\n"
                       "Original token count: 3\nOutput:\nx\ny\n")
        running = ("Chunk ID: 2\nWall time: 0.0100 seconds\nProcess running with session ID 7\n"
                   "Original token count: 0\nOutput:\n")
        patch = ("*** Begin Patch\n*** Add File: a\n*** Update File: b\n*** Delete File: c\n"
                 "*** Update File: d\n*** End Patch")
        for record, expected in (
                (call("exec_command", "c", {"cmd": "cat <<EOF\nhi\nEOF"}), "cat <<EOF …"),
                (call("apply_patch", "c", {"patch": patch}), "a, b, c +1 more"),
                (call("write_stdin", "c", {"session_id": 3, "chars": "q\n"}),
                 "session_id=3 chars='q\\n'"),
                (call("update_plan", "c", {"explanation": "Why.",
                                           "plan": [{"step": "A", "status": "completed"}]}),
                 "[x] A"),
                (call("icron", "c", {"call": "jobs()"}), "icron.jobs()"),
                (call("update_plan", "c", {"explanation": "Only why.\nMore."}), "Only why. …"),
                (call("other", "c", {"b": 1, "a": [2]}), '{"a":[2],"b":1}'),
                (result("c", exec_output), "exit 2 · 2 lines"),
                (result("c", running), "running, session 7 · no output"),
                (result("c", ""), "no output"), (result("c", "Done!"), "Done!"),
                (result("c", "\nfirst\nsecond\n"), "first … 2 lines"),
                (result("c", "x" * 500), "x" * (pages.GIST_CHARS - 1) + "…"),
                ({"type": "reasoning", "content": "", "summary": [], "encrypted_content": "e"},
                 "encrypted content"),
                ({"type": "model_failure", "message": "Too many\nrequests"}, "Too many …"),
                ({"type": "user_tool_call", "call": call("compact", "u", {})}, "{}"),
                ({"type": "context_prefix",
                  "prefix_items": [message("user", "Kept."),
                                   {"type": "opaque_compaction", "payload": "p",
                                    "protocol": "provider"}]},
                 "an opaque checkpoint, protocol=provider"),
                ({"type": "message", "role": "user", "content": "x"}, ""),
                ({"type": "tool_call", "name": 5, "arguments_json": ["odd"]}, '["odd"]'),
                ({"type": "tool_result", "output": None}, "null")):
            self.assertEqual(pages.gist(record), expected, record)


class StyleFileTests(unittest.TestCase):
    """css/'s files: each a style, read again for each page."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="node-test-"))
        self.css = self.tmp / "css"
        shutil.copytree(NODE / "css", self.css)
        self.styles = styles.Styles(self.css)
        self.saves = self.tmp / "saves"
        write_log(self.saves / "s._private" / "context.jsonl", [init(KEY_A), *turn("Hi.")])
        self.registry = self.tmp / "registry"
        Registry(self.registry).register(KEY_A, self.saves / "s._private")  # root 1

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def pages(self, style):
        return pages.Pages(contexts.Roots(self.registry), TOKEN,
                           styles=self.styles).view(style=style)

    def test_a_stylesheet_is_a_style(self):
        (self.css / "mine.css").write_text("body{color:red}\n")
        (self.css / "chat-wide.css").write_text("body{max-width:none}\n")
        (self.css / "My Style.css").write_text("x{}\n")
        (self.css / ".#chat.css").write_text("")  # an editor's, hidden: neither
        (self.css / "notes.txt").write_text("")
        self.assertEqual([style.name for style in self.styles.all()],
                         ["log-dark", "log-light", "chat", "chat-wide", "mine"])
        wide, chat = self.styles.get("chat-wide"), self.styles.get("chat")
        self.assertEqual((wide.parent, wide.fold), ("chat", chat.fold))
        self.assertEqual((self.styles.get("mine").parent, self.styles.get("mine").fold),
                         (None, frozenset()))
        base, chat_css = ((self.css / name).read_text() for name in ("base.css", "chat.css"))
        self.assertEqual(self.styles.css("chat-wide"),
                         ("\n".join((base, chat_css, "body{max-width:none}\n")), []))
        self.assertEqual(self.styles.css("mine"), ("\n".join((base, "body{color:red}\n")), []))
        self.assertEqual(self.styles.ignored(), ["My Style.css"])
        menu = MENU.search(self.pages("chat-wide").contexts().body).group(0)
        self.assertIn("style: chat-wide</summary>", menu)
        self.assertIn(">chat-wide<span class=note>Chat, with css/chat-wide.css</span>", menu)
        self.assertIn("Not styles, by their names: My Style.css.", menu)
        body = self.pages("chat-wide").context("1", {"turn": "1"}).body
        self.assertIn('<details class="it tool t-tool_call"', body)  # chat's layout

    def test_log_variants_inherit_the_whole_stylesheet_chain(self):
        wide_css = "body{max-width:none}\n"
        for name in ("log-dark", "log-light"):
            (self.css / f"{name}-wide.css").write_text(wide_css)
        self.assertEqual([style.name for style in self.styles.all()],
                         ["log-dark", "log-dark-wide", "log-light", "log-light-wide", "chat"])
        base, dark, light = ((self.css / name).read_text()
                            for name in ("base.css", "log-dark.css", "log-light.css"))
        self.assertEqual(self.styles.get("log-light").parent, "log-dark")
        for name, layers in (("log-dark", (base, dark)), ("log-light", (base, dark, light))):
            with self.subTest(style=name):
                parent, variant = self.styles.get(name), self.styles.get(f"{name}-wide")
                self.assertEqual(parent.fold, frozenset())
                self.assertEqual((variant.parent, variant.fold), (name, parent.fold))
                self.assertEqual(self.styles.css(name), ("\n".join(layers), []))
                self.assertEqual(self.styles.css(variant.name),
                                 ("\n".join((*layers, wide_css)), []))
                body = self.pages(variant.name).context("1", {"turn": "1"}).body
                self.assertIn('<div class="it tool t-tool_call"', body)
                self.assertNotIn('<details class="it', body)

    def test_log_palettes_are_explicit(self):
        palettes = {}
        for name, scheme, background, foreground in (
                ("log-dark", "only dark", "#0d1117", "#e6edf3"),
                ("log-light", "only light", "#fff", "#1f2328")):
            css, problems = self.styles.css(name)
            self.assertEqual(problems, [])
            self.assertNotIn("prefers-color-scheme", css)
            roots = re.findall(r":root\s*\{([^{}]*)\}", css)
            # Both palettes are unconditional :root rules with the same
            # specificity. The last declaration wins, in either OS theme.
            properties = dict(re.findall(r"([\w-]+)\s*:\s*([^;]+)", ";".join(roots)))
            self.assertEqual(properties["color-scheme"], scheme)
            self.assertEqual(properties["--bg"], background)
            self.assertEqual(properties["--fg"], foreground)
            own = (self.css / f"{name}.css").read_text()
            palettes[name] = set(re.findall(r"(--[\w-]+)\s*:", own))
        self.assertEqual(palettes["log-dark"], palettes["log-light"])
        self.assertEqual(palettes["log-light"], {
            "--fg", "--dim", "--bg", "--rule", "--box", "--a", "--user", "--asst",
            "--tool", "--err", "--esc", "--escfg", "--add", "--del", "--hunk"})

    def test_inherited_stylesheets_are_reloaded_and_report_problems(self):
        (self.css / "log-light-wide.css").write_text("body{max-width:none}\n")
        view = self.pages("log-light-wide")
        dark = self.css / "log-dark.css"
        dark.write_text(dark.read_text() + "\npre{tab-size:4}\n")
        self.assertIn("pre{tab-size:4}", view.contexts().body)
        light = self.css / "log-light.css"
        light.write_text(light.read_text() + "\n:root{--bg:#fafafa}\n")
        body = view.contexts().body
        self.assertIn(":root{--bg:#fafafa}", body)
        self.assertLess(body.index("pre{tab-size:4}"), body.index(":root{--bg:#fafafa}"))
        dark.unlink()
        body = view.contexts().body
        self.assertIn(f"<p class=node-note>{pages.h(str(dark) + ' is missing.')}</p>", body)
        self.assertIn(":root{--bg:#fafafa}", body)
        self.assertIn("body{max-width:none}", body)
        self.assertIn("<details class=node-menu>", body)

    def test_a_parent_cycle_is_reported(self):
        cyclic = styles.Styles(self.css, built_in=(
            styles.Style("a", "A", parent="b"), styles.Style("b", "B", parent="a")))
        for name in ("a", "b"):
            (self.css / f"{name}.css").write_text(f"/* {name} */")
        css, problems = cyclic.css("a")
        self.assertEqual(problems, ["Style 'a' has a parent cycle at 'a'."])
        self.assertEqual(css.count("/* a */"), 1)
        self.assertEqual(css.count("/* b */"), 1)

    def test_an_edit_shows_at_the_next_page(self):
        path = self.css / "mine.css"
        path.write_text("body{color:red}\n")
        self.assertIn("body{color:red}", self.pages("mine").contexts().body)
        path.write_text("body{color:blue}\n")
        self.assertIn("body{color:blue}", self.pages("mine").contexts().body)
        app = server.App(contexts.Roots(self.registry), TOKEN, styles=self.styles)
        self.assertEqual(app.respond("GET", f"/{TOKEN}/?style=mine", ["127.0.0.1"]).status, 200)
        path.unlink()
        self.assertIsNone(self.styles.get("mine"))
        self.assertEqual(app.respond("GET", f"/{TOKEN}/?style=mine", ["127.0.0.1"]).status, 400)

    def test_a_stylesheet_that_would_end_its_element_isnt_used(self):
        (self.css / "evil.css").write_text("x{}</STYLE><script>alert(1)</script>\n")
        problem = (f"{self.css / 'evil.css'} isn't used: it holds </style, which would end the "
                   "page's stylesheet.")
        self.assertEqual(self.styles.css("evil"), ((self.css / "base.css").read_text(), [problem]))
        body = self.pages("evil").contexts().body
        self.assertNotIn("<script>", body)
        self.assertNotIn("alert(1)", body)
        self.assertIn(f"<p class=node-note>{pages.h(problem)}</p>", body)

    def test_problems_show_on_the_page(self):
        (self.css / "latin.css").write_bytes(b"x{content:'\xff'}")
        (self.css / "big.css").write_text("x{}" * (styles.MAX_CSS_BYTES // 3 + 1))
        for name, problem in (("latin", "isn't UTF-8."), ("big", "it's over 256 KiB."),
                              ("log-dark", "base.css is missing.")):
            if name == "log-dark":
                (self.css / "base.css").unlink()
            _, problems = self.styles.css(name)
            self.assertEqual(len(problems), 1, problems)
            self.assertTrue(problems[0].endswith(problem), problems)
            body = self.pages(name).contexts().body
            self.assertIn(f"<p class=node-note>{pages.h(problems[0])}</p>", body)
            self.assertIn("<details class=node-menu>", body)  # the menu still switches

    def test_the_shipped_stylesheets(self):
        shipped = styles.Styles()
        self.assertEqual([style.name for style in shipped.built_in],
                         ["log-dark", "log-light", "chat"])
        self.assertEqual(styles.DEFAULT_STYLE, "log-dark")
        self.assertIsNone(shipped.get("log"))
        for style in shipped.built_in:
            name = style.name
            css, problems = shipped.css(name)
            self.assertEqual(problems, [], name)
            plain = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
            self.assertNotIn("/*", plain)
            self.assertNotIn("*/", plain)
            for opening, closing in (("{", "}"), ("(", ")")):
                self.assertEqual(plain.count(opening), plain.count(closing), (name, opening))
            self.assertEqual(plain.count('"') % 2, 0, name)
            self.assertNotRegex(plain, r"[;{]\s*;|:\s*[;}]", name)


class ViewTests(WorkspaceTest):
    """The glui view: its pages, facts, bodies and files (view.py)."""

    def get(self, path, files=None):
        """GET ``/TOKEN/path``; ``files``: a directory in place of _pythia_ui/."""
        shown = None if files is None else view.View(view.Files(files))
        app = server.App(contexts.Roots(self.ws.registry), TOKEN, view=shown)
        return app.respond("GET", f"/{TOKEN}/{path}", ["127.0.0.1"])

    def steps(self, path):
        response = self.get(path)
        self.assertEqual((response.status, response.content_type, response.csp),
                         (200, "application/json", view.DATA_CSP), response.body[:300])
        steps = json.loads(response.body)
        self.assertEqual({step[0] for step in steps}, {"+"})
        return steps

    def facts(self, steps, name):
        return [fields for _, rel, fields in steps if rel == name]

    def test_the_pages_load_glui_and_name_their_facts(self):
        index = self.get("lem/")
        self.assertEqual((index.status, index.content_type, index.csp),
                         (200, "text/html", view.PAGE_CSP))
        files = f"/{TOKEN}/lem/files/"
        self.assertEqual(re.findall(r'<script src="([^"]*)"></script>', index.body),
                         [files + name for name in view.SCRIPTS])
        self.assertEqual(re.findall(r'<link rel=stylesheet href="([^"]*)">', index.body),
                         [files + name for name in view.STYLESHEETS])
        # Nothing that its policy refuses: no script but those, no style in the page.
        self.assertEqual(index.body.count("<script"), len(view.SCRIPTS))
        self.assertNotRegex(index.body, r"<style|style=|javascript:")
        tabs = (f'data-index="/{TOKEN}/lem/facts/" data-tab="/{TOKEN}/lem/tab/{{id}}" '
                f'data-log="/{TOKEN}/lem/c/{{id}}" data-home="/{TOKEN}/lem/"')
        for shown in (f'<html lang=en data-facts="/{TOKEN}/lem/facts/" '
                      f'data-store="/{TOKEN}/lem/store/session" {tabs}>',
                      "<title>root contexts · node</title>",
                      f'<noscript><p>The glui view runs in JavaScript; <a href="/{TOKEN}/">the '
                      "plain pages</a> need none.</p></noscript>"):
            self.assertIn(shown, index.body)
        log = self.get("lem/c/1")
        self.assertEqual((log.status, log.csp), (200, view.PAGE_CSP))
        for shown in (f'<html lang=en data-facts="/{TOKEN}/lem/facts/c/1" '
                      f'data-store="/{TOKEN}/lem/store/session" {tabs} data-open="1" '
                      f'data-before="/{TOKEN}/lem/rows/1/before/{{line}}" '
                      f'data-after="/{TOKEN}/lem/rows/1/after/{{line}}" '
                      f'data-around="/{TOKEN}/lem/rows/1/around/{{line}}" '
                      f'data-run="/{TOKEN}/lem/run/1/{{run}}" '
                      f'data-body="/{TOKEN}/lem/body/1/{{line}}" '
                      f'data-grow="/{TOKEN}/lem/grow/1/',
                      "<title>local-0 (1) · node</title>", f'<a href="/{TOKEN}/c/1">the plain'):
            self.assertIn(shown, log.body)
        self.assertRegex(log.body, f'data-grow="/{TOKEN}/lem/grow/1/[1-9][0-9]*/{{lines}}" '
                                   f'data-stars="/{TOKEN}/lem/stars/1" '
                                   f'data-pins="/{TOKEN}/lem/pins/1">')
        second = self.get(f"lem/c/{SECOND}")  # another number of 1's
        self.assertEqual((second.status, second.location), (303, f"/{TOKEN}/lem/c/1"))
        self.assertEqual((self.get("lem").status, self.get("lem").location),
                         (308, f"/{TOKEN}/lem/"))
        for cid, why in ((REPLACED, "Its path now holds another save."),
                         (GONE, "Nothing is at its path any more"),
                         ("99", "There&#x27;s no root context 99 in")):
            response = self.get(f"lem/c/{cid}")
            self.assertEqual(response.status, 404, cid)
            self.assertIn(why, response.body)
            self.assertNotIn("TOP SECRET", response.body)

    def test_the_index(self):
        steps = self.steps("lem/facts/")
        self.assertEqual(self.facts(steps, "Page"), [
            {"title": f"Root contexts in {contexts.shown_path(self.ws.registry)}/"}])
        self.assertEqual(self.facts(steps, "Nav"), [
            {"n": 0, "text": "Plain pages", "to": f"/{TOKEN}/"},
            {"n": 1, "text": "3 aren't where they're registered", "to": f"/{TOKEN}/"}])
        found = {fact["id"]: fact for fact in self.facts(steps, "Context")}
        self.assertEqual(sorted(found), ["local-0 (1)", "old (2)", "proj (3)"])
        self.assertEqual(sorted(fact["rank"] for fact in found.values()), [0, 1, 2])
        log = Path(os.path.realpath(self.ws.saves)) / "local-0._private" / "context.jsonl"
        local = found.pop("local-0 (1)")
        self.assertRegex(local.pop("changed"), r"\A[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\Z")
        local.pop("rank")
        self.assertEqual(local, {"id": "local-0 (1)",
                                 "title": 'Fix the bug <script>alert(1)</script> & "quoted"',
                                 "model": "model-a", "turns": 4, "lines": 63,
                                 "size": pages.byte_count(log.stat().st_size),
                                 "to": f"/{TOKEN}/lem/c/1", "label": "", "tags": "",
                                 "keep": ""})  # no store, so no label nor tags to keep
        # The rest, by id: their order is their logs' last change, which may be the same.
        self.assertEqual(sorted((fact["id"], fact["title"], fact["model"], fact["to"])
                                for fact in found.values()),
                         [("old (2)", "Old question.", "", f"/{TOKEN}/lem/c/2"),
                          ("proj (3)", "Elsewhere.", "model-e", f"/{TOKEN}/lem/c/3")])

    def test_the_tags_in_use_by_name_with_how_many_roots_have_each(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "node")
            store.root(KEY_A, {"label": "", "tags": ["x", "b"]})
            store.root(KEY_E, {"label": "", "tags": ["x"]})
            app = server.App(contexts.Roots(self.ws.registry), TOKEN, store=store)
            steps = json.loads(app.respond("GET", f"/{TOKEN}/lem/facts/", ["127.0.0.1"]).body)
        self.assertEqual([fields for _, rel, fields in steps if rel == "Tag"],
                         [{"name": "b", "rank": 0, "count": 1},
                          {"name": "x", "rank": 1, "count": 2}])
        self.assertEqual(sorted((fields["id"], fields["tag"]) for _, rel, fields in steps
                                if rel == "Tagged"),
                         [("local-0 (1)", "b"), ("local-0 (1)", "x"), ("proj (3)", "x")])
        self.assertEqual(step_problems(json.loads(PROGRAM.read_text()), steps), [])

    def test_a_log(self):
        steps = self.steps("lem/facts/c/1")
        self.assertEqual(self.facts(steps, "Page"), [
            {"title": "local-0 (1)", "items": f"/{TOKEN}/c/1?line=",
             "turns": f"/{TOKEN}/c/1?turn=", "first": 1, "id": "local-0 (1)"}])
        self.assertEqual([fields["id"] for fields in self.facts(steps, "Activity")],
                         ["local-0 (1)"])  # what it's doing, for its dot (ActivityTests)
        self.assertEqual(self.facts(steps, "Length"), [{"lines": 63}])
        self.assertEqual(self.facts(steps, "Nav"), [
            {"n": 0, "text": "Root contexts", "to": f"/{TOKEN}/lem/"},
            {"n": 1, "text": "Plain pages", "to": f"/{TOKEN}/c/1"}])
        turns = self.facts(steps, "Turn")
        self.assertEqual([(fact["n"], fact["label"], fact["title"], fact["line"])
                          for fact in turns], [
            (0, "The start · 2 items", "", 1),
            (1, "Turn 1 · 14 items", 'Fix the bug <script>alert(1)</script> & "quoted"', 3),
            (2, "Turn 2 · 10 items", "/compact", 22), (3, "Turn 3 · 13 items", "Next step.", 34),
            (4, "Turn 4 · 10 items", "/config · Look at this.", 52)])
        # The whole log, which holds fewer rows than a chunk: its messages, each with its
        # text, its compactions, each a row alone with its item, and its runs, each with
        # what's in it.
        self.assertEqual(self.facts(steps, "Chunk"), [{"first": 1, "last": 63}])
        messages = {fact["line"]: fact for fact in self.facts(steps, "Message")}
        runs = {fact["line"]: fact for fact in self.facts(steps, "Run")}
        self.assertEqual(sorted(messages), [3, 18, 30, 34, 48, 54, 60])
        self.assertEqual([(line, run["last"], run["text"]) for line, run in sorted(runs.items())], [
            (1, 2, "2 other"),
            (5, 17, "3 tool calls: apply_patch 1, exec_command 1, update_plan 1; 1 reasoning; "
             "3 other"),
            (19, 21, "2 other"), (22, 23, "1 tool call: compact 1"),
            (25, 29, "1 tool call: exec_command 1; 2 other"), (31, 33, "2 other"),
            (36, 47, "3 tool calls: exec_command 1, icron 1, retry 1; 3 other"),
            (49, 51, "2 other"), (52, 53, "1 tool call: config 1"), (57, 59, "3 other"),
            (61, 63, "2 other")])
        # A compaction, between the runs before and after it: a row alone, whose item, of
        # its own run, comes with it, in the tone of what's read.
        lones = {fact["line"]: fact for fact in self.facts(steps, "Lone")}
        alone = {fact["line"]: fact for fact in self.facts(steps, "Item")}
        self.assertEqual(sorted(lones), [24, 56])
        self.assertEqual([(line, fact["run"], fact["tone"], fact["label"], fact["body"])
                          for line, fact in sorted(alone.items())],
                         [(24, 24, "Normal", "[context prefix] 1 item", True),
                          (56, 56, "Normal", "[compaction]", False)])
        self.assertEqual(self.get("lem/run/1/24").status, 404)  # no run's
        context = contexts.Roots(self.ws.registry).get(LOCAL).refresh()
        rows = {**messages, **runs, **lones}
        self.assertEqual({line: fact["turn"] for line, fact in rows.items()},
                         {line: context.turn_of(line - 1) for line in rows})
        for line, tone, label, text in (
                (3, "User", "[user]", 'Fix the bug <script>alert(1)</script> & "quoted"'),
                (18, "Assistant", "[assistant]",
                 "Fixed\\u202e it.\\u2028Next\\U000e0041 \U0001f469\u200d\U0001f4bb"),
                (54, "User", "[user]", "Look at this.\n\n[image] image/png 11.7 KiB"),
                (60, "Assistant", "[assistant]", "Lone \\ud800 surrogate.")):
            self.assertEqual(messages[line], {"line": line, "turn": context.turn_of(line - 1),
                                              "tone": tone, "label": label, "text": text}, line)
        # A run's items come when it's opened: with the messages, every line but the
        # boundaries, each in its run.
        items, run_of = dict(alone), {line: line for line in alone}
        for line, run in runs.items():
            loaded = self.steps(f"lem/run/1/{line}")
            self.assertEqual(loaded[0], ["+", "RunLoaded", {"run": line}])
            mine = self.facts(loaded, "Item")
            self.assertEqual({fact["run"] for fact in mine}, {line})
            run_of.update((fact["line"], line) for fact in mine)
            self.assertTrue(all(line <= fact["line"] <= run["last"] for fact in mine), line)
            items.update((fact["line"], fact) for fact in mine)
        boundaries = [line for line, record in enumerate(local_items(), 1)
                      if record in (BOUNDARY, SAMPLE_END)]
        self.assertEqual(sorted([*items, *messages]),
                         [line for line in range(1, 64) if line not in boundaries])
        for line, tone, label, line_gist, body in (
                (1, "Dim", "[init]", f"{KEY_A} model=model-a", False),
                (5, "Normal", "[reasoning]", "Looking at the files.", True),
                (6, "Tool", "[tool-call] exec_command", "ls", True),
                (9, "Tool", "[tool-ret] exec_command [ok]", "a.py … 2 lines", True),
                (13, "Tool", "[tool-ret] apply_patch [ok]", "Done!", False),
                (21, "Dim", "[turn]", "input_sum=2400 output_sum=160 cold_sum=400 cached_sum=2000 "
                 "cached_max=1000 context=1280 samples=2 compactions=0 elapsed=75.00s", False),
                (36, "Error", "[model]", "Too many requests", True),
                (47, "Tool", "[tool-ret] exec_command [ok]", "x" * (pages.GIST_CHARS - 1) + "…",
                 True),
                (59, "Error", "[bad line]", "not json at all", False)):
            self.assertEqual(items[line], {"line": line, "run": run_of[line], "tone": tone,
                                           "label": label, "gist": line_gist, "body": body}, line)

    def test_a_turn_marked_where_it_first_shows(self):
        # A boundary with no message just before it starts a turn on a line that shows
        # nothing. The turn's line is where it first shows, where its first row starts, so
        # that its mark has a row to go before.
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / "s._private"
            write_log(private / "context.jsonl",
                      [init(KEY_A), *turn("One."), BOUNDARY, *work("w2"), *answer("Two."),
                       summary()])
            write_log(Path(tmp) / "s.jsonl", [init(KEY_A)])
            registry = Path(tmp) / "registry"
            self.assertEqual(Registry(registry).register(KEY_A, private).status, "new")
            app = server.App(contexts.Roots(registry), TOKEN)
            steps = json.loads(app.respond("GET", f"/{TOKEN}/lem/facts/c/1",
                                           ["127.0.0.1"]).body)
        rows = [fields["line"] for _, rel, fields in steps if rel in ("Message", "Run")]
        self.assertEqual([(fact["n"], fact["line"]) for fact in self.facts(steps, "Turn")],
                         [(0, 1), (1, 2), (2, 13)])
        self.assertIn(13, rows)

    def test_rows_come_in_chunks(self):
        # Three rows a chunk: the page's facts give the last; the rows before a chunk's
        # first line, those before it, to the log's start; and the rows after a line, those
        # after it, to its end. Each chunk ends where a row ends, and they tile the log,
        # each row in one.
        whole = [(rel, fields["line"]) for _, rel, fields in self.steps("lem/facts/c/1")
                 if rel in ("Message", "Lone", "Run")]
        with mock.patch.object(view, "CHUNK_ROWS", 3):
            chunks = []
            steps = self.steps("lem/facts/c/1")
            while True:
                chunk = self.facts(steps, "Chunk")[0]
                rows = [(rel, fields["line"]) for _, rel, fields in steps
                        if rel in ("Message", "Lone", "Run")]
                self.assertLessEqual(len(rows), 3, chunk)
                chunks.insert(0, (chunk["first"], chunk["last"], rows))
                if chunk["first"] == 1:
                    break
                steps = self.steps(f"lem/rows/1/before/{chunk['first']}")
            self.assertEqual([row for _, _, rows in chunks for row in rows], whole)
            self.assertEqual([chunks[0][0]] + [last + 1 for _, last, _ in chunks],
                             [first for first, _, _ in chunks] + [64])
            forward, after = [], 0
            while after < 63:
                steps = self.steps(f"lem/rows/1/after/{after}")
                chunk = self.facts(steps, "Chunk")[0]
                self.assertEqual(chunk["first"], after + 1)
                forward += [(rel, fields["line"]) for _, rel, fields in steps
                            if rel in ("Message", "Lone", "Run")]
                after = chunk["last"]
            self.assertEqual(forward, whole)
            self.assertEqual(self.facts(self.steps("lem/rows/1/after/63"), "Chunk"),
                             [{"first": 64, "last": 63}])

    def test_a_tab_names_its_logs_urls_as_its_page_does(self):
        response = self.get("lem/tab/1")
        self.assertEqual((response.status, response.content_type, response.csp),
                         (200, "application/json", view.DATA_CSP))
        tab = json.loads(response.body)
        # Each request here reads the registry anew, so a log's index, whose generation
        # names its growth, is new each time.
        page = re.sub(r"/grow/1/[0-9]+/", "/grow/1/G/", self.get("lem/c/1").body)
        tab["grow"] = re.sub(r"/grow/1/[0-9]+/", "/grow/1/G/", tab["grow"])
        for name in ("facts", "before", "after", "around", "run", "body", "grow", "stars",
                     "pins"):
            self.assertIn(f'data-{name}="{html.escape(tab[name])}"', page, name)
        self.assertEqual((tab["id"], tab["title"], tab["plain"]),
                         ("1", "local-0 (1)", f"/{TOKEN}/c/1"))
        self.assertEqual(sorted(tab), ["after", "around", "before", "body", "facts", "grow",
                                       "id", "pins", "plain", "run", "stars", "title"])
        for cid in ("99", "", "x", REPLACED):
            self.assertEqual(self.get(f"lem/tab/{cid}").status, 404, cid)

    def test_rows_around_a_line_keep_its_row_whole(self):
        # Line 6 is inside turn 1's run, lines 5 to 17: the rows around it start with the
        # row before that run, and the run comes whole.
        with mock.patch.object(view, "CHUNK_ROWS", 3):
            steps = self.steps("lem/rows/1/around/6")
        self.assertEqual(self.facts(steps, "Chunk"), [{"first": 3, "last": 18}])
        self.assertEqual([(rel, fields["line"]) for _, rel, fields in steps if rel != "Chunk"],
                         [("Message", 3), ("Run", 5), ("Message", 18)])
        with mock.patch.object(view, "CHUNK_ROWS", 3):
            first = self.steps("lem/rows/1/around/1")
        self.assertEqual(self.facts(first, "Chunk"), [{"first": 1, "last": 17}])  # 3 after
        for line in ("0", "64", "x", ""):
            self.assertEqual(self.get(f"lem/rows/1/around/{line}").status, 404, line)

    def test_bodies(self):
        def body(line):
            return [(fact["part"], fact["text"], fact["diff"])
                    for fact in self.facts(self.steps(f"lem/body/1/{line}"), "Body")]
        self.assertEqual(body(10), [(0, PATCH, True)])
        self.assertEqual(body(9), [(0, "a.py\nb.py", False)])
        self.assertEqual(body(29), [(0, "diff --git a/a.py b/a.py\n-old\n+new", True)])  # git diff
        self.assertEqual(body(6), [(0, "yield_time_ms=1000", False), (1, "ls", False)])
        self.assertEqual(body(36), [(0, "Too many requests kind=ModelRateLimitError", False),
                                    (1, "provider=p status=429 attempts=2 event_types=a(2),b",
                                     False)])
        self.assertEqual(body(47), [(0, "x" * pages.CUT, False),
                                    (1, "… 36,000 more characters: the whole item is on its "
                                     "line's page.", False)])
        self.assertEqual(body(18), [(0, "Fixed\\u202e it.\\u2028Next\\U000e0041 "
                                     "\U0001f469\u200d\U0001f4bb", False)])
        self.assertEqual(body(54), [(0, "Look at this.", False),
                                    (1, "[image] image/png 11.7 KiB", False)])
        self.assertEqual(body(24), [(0, pointer(21), False), (1, "[user]", False),
                                    (2, pointer(21), False)])  # the prefix, and its item
        for path, why in (("lem/body/1/0", "local-0 (1) has no line 0."),
                          ("lem/body/1/64", "local-0 (1) has no line 64."),
                          ("lem/body/1/x", "local-0 (1) has no line x."),
                          ("lem/body/1/4", "Line 4 of local-0 (1) has no body."),  # a boundary
                          (f"lem/body/{REPLACED}/2", "There's no such root context here."),
                          (f"lem/body/{SECOND}/3", "There's no such root context here."),
                          ("lem/facts/c/99", "There's no such root context here."),
                          ("lem/rows/1/before/0", "local-0 (1) has no rows before line 0."),
                          ("lem/rows/1/before/65", "local-0 (1) has no rows before line 65."),
                          ("lem/rows/1/after/64", "local-0 (1) has no rows after line 64."),
                          ("lem/rows/1/after/01", "local-0 (1) has no rows after line 01."),
                          ("lem/rows/1/after/x", "local-0 (1) has no rows after line x."),
                          ("lem/rows/1/aside/3", "local-0 (1) has no rows aside line 3."),
                          ("lem/run/1/3", "local-0 (1) has no run at line 3."),  # a message
                          ("lem/run/1/6", "local-0 (1) has no run at line 6."),  # within one
                          ("lem/run/1/4", "local-0 (1) has no run at line 4."),  # a boundary
                          ("lem/run/1/0", "local-0 (1) has no run at line 0."),
                          ("lem/run/1/x", "local-0 (1) has no run at line x."),
                          (f"lem/rows/{REPLACED}/before/2", "There's no such root context here."),
                          (f"lem/run/{SECOND}/1", "There's no such root context here."),
                          (f"lem/facts/c/{UNREADABLE}", "There's no such root context here.")):
            response = self.get(path)
            self.assertEqual((response.status, response.content_type, response.body),
                             (404, "text/plain", why + "\n"), path)
        self.assertEqual(self.get("lem/facts/x").status, 404)

    def test_files(self):
        # What the pages load, and what contexts.js fetches beside itself, as it says:
        # the programs, and each of its themes.
        script = (view.UI_DIRECTORY / "web" / "contexts.js").read_text()
        listed = re.search(r"var THEMES = \[(.*?)\];", script, re.S)[1]
        themes = re.findall(r'\["([a-z]+)", ', listed)
        self.assertEqual(themes, ["dark", "light"])
        fetched = re.findall(r'new URL\("(compiled/[a-z-]+\.json)", here\)', script)
        self.assertIn("compiled/contexts.json", fetched)
        self.assertEqual(sorted(view.FILES), sorted(
            [*view.SCRIPTS, *view.STYLESHEETS, *fetched,
             *(f"compiled/theme-{theme}.json" for theme in themes)]))
        for name, path in view.FILES.items():
            response = self.get(f"lem/files/{name}")
            self.assertEqual((response.status, response.csp), (200, view.DATA_CSP), name)
            self.assertEqual(response.content_type, view.TYPES[Path(path).suffix], name)
            self.assertEqual(response.body, (view.UI_DIRECTORY / path).read_text(), name)
        for name in ("x.js", "../web/demo.js", "..%2Fweb%2Fdemo.js", "%2E%2E/web/demo.js",
                     "compiled/form.json", "", "src/lem/lem.js", "web/contexts.js", "LEM.JS",
                     "lem.js/", "/lem.js"):
            response = self.get(f"lem/files/{name}")
            self.assertEqual((response.status, response.body),
                             (404, "The glui view has no such file.\n"), name)
        with tempfile.TemporaryDirectory() as tmp:  # a checkout without most of them
            (Path(tmp) / "web").mkdir()
            shutil.copy(view.UI_DIRECTORY / "web" / "contexts.js", Path(tmp) / "web")
            response = self.get("lem/", files=tmp)
            self.assertEqual(response.status, 404)
            self.assertIn("The glui view needs files that aren&#x27;t in", response.body)
            self.assertIn("src/lem/lem.js, src/glem/glem.js", response.body)
            self.assertNotIn("web/contexts.js", response.body)
            self.assertEqual(self.get("lem/files/contexts.js", files=tmp).status, 200)
            self.assertEqual(self.get("lem/files/lem.js", files=tmp).status, 404)
            self.assertEqual(self.get("", files=tmp).status, 200)  # the plain pages don't mind

    def test_glui_s_pages_preview_the_policy(self):
        # Those that `make web-serve` serves in _pythia_ui/, but frame-ancestors, which
        # only a header can give, so that a browser that runs them checks it too.
        meta = view.PAGE_CSP.replace("; frame-ancestors 'none'", "")
        self.assertNotEqual(meta, view.PAGE_CSP)
        for page in ("demo.html", "test.html", "contexts-index.html", "contexts-log.html"):
            text = (view.UI_DIRECTORY / "web" / page).read_text()
            self.assertIn(f'<meta http-equiv="Content-Security-Policy" content="{meta}">', text,
                          page)

    def test_the_facts_suit_the_program(self):
        program = json.loads(PROGRAM.read_text())
        self.assertEqual(step_problems(program, self.steps("lem/facts/")), [])
        for cid in (LOCAL, OLD, PROJ):
            steps = self.steps(f"lem/facts/c/{cid}")
            self.assertEqual(step_problems(program, steps), [], cid)
            every = list(steps)  # as a page has them once each run is open
            for run in self.facts(steps, "Run"):
                loaded = self.steps(f"lem/run/{cid}/{run['line']}")
                self.assertEqual(step_problems(program, loaded), [], (cid, run["line"]))
                every += loaded
                for item in self.facts(loaded, "Item"):
                    if item["body"]:
                        body = self.steps(f"lem/body/{cid}/{item['line']}")
                        self.assertEqual(step_problems(program, body), [], (cid, item["line"]))
            self.assertEqual(step_problems(program, every), [], cid)
        # The check itself.
        self.assertEqual(step_problems(program, [
            ["+", "Item", {"line": 1, "run": 1, "tone": "Blue", "label": "x", "gist": "y",
                           "body": 1}],
            ["+", "Turn", {"n": 1}], ["+", "Nope", {}], ["-", "Turn", {"n": 2, "title": "t"}],
            ["+", "Turn", {"n": 2.5, "label": "l", "title": "t", "x": 0}],
            ["+", "Page", {"title": "p"}],
            ["+", "Nav", {"n": 1, "text": "a", "to": "/"}],
            ["+", "Nav", {"n": 1, "text": "b", "to": "/"}]]), [
            "Item's tone is a Tone, not 'Blue'", "Item's body is a Bool, not 1",
            "Turn needs its label", "Turn needs its title", "Turn needs its line",
            "Nope isn't an input", "['-', 'Turn', {'n': 2, 'title': 't'}] doesn't add a fact",
            "Turn has no field x", "Turn's n is a Int, not 2.5", "Turn needs its line",
            "two Nav facts have the key [1]"])


class StoreTests(unittest.TestCase):
    """The glui view's store, which keeps its settings across the server's runs (store.py),
    and the POST that writes it, from the view's own pages alone (server.py)."""

    OWN = {"origins": ["http://127.0.0.1:8765"], "marked": True,
           "content_type": "application/json"}

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="node-store-"))
        self.store = Store(self.tmp / "node")
        self.app = server.App(contexts.Roots(self.tmp / "registry"), TOKEN, store=self.store)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def post(self, body, name="session", host="127.0.0.1:8765", **options):
        sent = dict(self.OWN, body=body)
        sent.update(options)
        return self.app.respond("POST", f"/{TOKEN}/lem/store/{name}", [host], **sent)

    def kept(self, name="session"):
        response = self.app.respond("GET", f"/{TOKEN}/lem/store/{name}", ["127.0.0.1:8765"])
        self.assertEqual((response.content_type, response.csp),
                         ("application/json", view.DATA_CSP))
        return response.status, json.loads(response.body) if response.status == 200 else None

    def test_settings_kept_whole_and_read_back(self):
        self.assertEqual(self.store.read("session"), {})  # nothing yet, nor a directory
        self.assertFalse((self.tmp / "node").exists())
        self.store.write("session", {"look": "log", "theme": "dark"})
        self.store.write("session", {"look": "native"})  # replaced whole
        self.assertEqual(self.store.read("session"), {"look": "native"})
        # No temporary file; the lock that a change takes.
        self.assertEqual(sorted(os.listdir(self.tmp / "node")), [".lock", "session.json"])
        for value, why in (([], "the settings aren't an object"),
                           ({f"s{n}": "x" for n in range(33)}, "more than 32 settings"),
                           ({"Look": "log"}, "'Look'"), ({"look": 1}, "'look'"),
                           ({"look": "a b"}, "'look'"), ({"look": "x" * 65}, "'look'")):
            with self.assertRaisesRegex(Invalid, re.escape(why)):
                self.store.write("session", value)
        self.assertEqual(self.store.read("session"), {"look": "native"})
        for data in (b"{not json", b'{"look": 1}', b"[]", b"{" + b" " * MAX_FILE + b"}",
                     b'{"look": "log"}' + b" " * MAX_FILE):  # longer than it writes
            (self.tmp / "node" / "session.json").write_bytes(data)
            self.assertEqual(self.store.read("session"), {}, data[:20])

    def test_a_write_that_fails_leaves_the_old_file(self):
        self.store.write("session", {"look": "log"})
        with mock.patch("os.fsync", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.store.write("session", {"look": "native"})
        self.assertEqual(self.store.read("session"), {"look": "log"})
        self.assertEqual(sorted(os.listdir(self.tmp / "node")), [".lock", "session.json"])
        with mock.patch("os.fsync", side_effect=OSError("disk full")), \
                contextlib.redirect_stderr(io.StringIO()) as said:
            response = self.post(b'{"look": "native"}')
        self.assertEqual((response.status, response.body),
                         (500, "The view's store couldn't be written.\n"))
        self.assertEqual(said.getvalue(), "node: the view's store couldn't be written: disk full\n")

    def test_stars_kept_by_root_one_at_a_time(self):
        self.assertEqual(self.store.read("stars"), {})
        for key, line, on in ((KEY_A, 5, True), (KEY_A, 9, True), (KEY_B, 2, True),
                              (KEY_A, 5, False), (KEY_A, 9, True), (KEY_B, 2, False)):
            self.store.star(key, line, on)
        self.assertEqual(self.store.read("stars"), {KEY_A: [9]})  # a root's last star goes
        for value, why in (({"x": [1]}, "'x' isn't a root's key"),
                           ({KEY_A: [0]}, "aren't a list"), ({KEY_A: ["1"]}, "aren't a list"),
                           ({KEY_A: list(range(1, 10_002))}, "aren't a list")):
            with self.assertRaisesRegex(Invalid, re.escape(why)):
                self.store.write("stars", value)
        self.assertEqual(self.store.read("stars"), {KEY_A: [9]})

    def test_two_stores_of_one_directory_lose_no_change(self):
        # As two node servers keep one store, a remote node's own, and the one
        # that another server runs there on its standard streams: each change
        # reads the file and writes it anew, under the store's lock.
        other = Store(self.tmp / "node")

        def star(store, lines):
            for line in lines:
                store.star(KEY_A, line, True)

        threads = [threading.Thread(target=star, args=(store, range(first, 401, 2)))
                   for store, first in ((self.store, 1), (other, 2))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        self.assertEqual(self.store.read("stars"), {KEY_A: list(range(1, 401))})

    def test_labels_and_tags_kept_by_root_one_at_a_time(self):
        self.assertEqual(self.store.read("roots"), {})
        self.store.root(KEY_A, {"label": "  glui  ", "tags": ["ui", "glui", "ui"]})
        self.store.root(KEY_B, {"label": "", "tags": ["old"]})
        self.store.root(KEY_B, {"label": "", "tags": []})  # with neither, a root goes
        kept = {KEY_A: {"label": "glui", "tags": ["glui", "ui"]}}
        self.assertEqual(self.store.read("roots"), kept)
        label = "a label is text of 120 characters at most"
        tags = ("a root has 12 tags at most, each a word of lowercase letters, digits, '.', "
                "'_' and '-', 32 characters at most")
        for value, why in (([], "a root is"), ({"label": "x"}, "a root is"),
                           ({"label": "x", "tags": [], "more": 1}, "a root is"),
                           ({"label": 1, "tags": []}, label),
                           ({"label": "x" * 121, "tags": []}, label),
                           ({"label": "a\nb", "tags": []}, label),
                           ({"label": "a\x85b", "tags": []}, label),
                           ({"label": "", "tags": "ui"}, tags),
                           ({"label": "", "tags": ["UI"]}, tags),
                           ({"label": "", "tags": ["a b"]}, tags),
                           ({"label": "", "tags": ["-x"]}, tags),
                           ({"label": "", "tags": ["x" * 33]}, tags),
                           ({"label": "", "tags": ["x"] * 13}, tags)):
            with self.assertRaisesRegex(Invalid, re.escape(why)):
                self.store.root(KEY_A, value)
        for value, why in (({"x": {"label": "y", "tags": []}}, "'x' isn't a root's key"),
                           ({KEY_A: {"label": "y"}}, "a root is"), ([], "aren't an object")):
            with self.assertRaisesRegex(Invalid, re.escape(why)):
                self.store.write("roots", value)
        self.assertEqual(self.store.read("roots"), kept)
        self.store.write("roots", {**kept, KEY_B: {"label": " ", "tags": []}})  # neither
        self.assertEqual(self.store.read("roots"), kept)

    def test_a_post_from_the_views_own_page(self):
        response = self.post(b'{"look": "native", "runs": "open"}')
        self.assertEqual((response.status, response.body), (204, ""))
        self.assertEqual(self.kept(), (200, {"look": "native", "runs": "open"}))
        # Its own origin, as the Host header names it, any loopback name and port.
        response = self.post(b'{"look": "log"}', host="LOCALHOST:9000",
                             origins=["http://localhost:9000"])
        self.assertEqual(response.status, 204)
        self.assertEqual(self.kept(), (200, {"look": "log"}))

    def test_what_the_store_refuses(self):
        self.post(b'{"look": "native"}')
        for options, status, why in (
                ({"origins": []}, 403, "own pages"),
                ({"origins": ["http://evil.example"]}, 403, "own pages"),
                ({"origins": ["http://127.0.0.1:9999"]}, 403, "own pages"),
                ({"origins": ["http://127.0.0.1:8765"] * 2}, 403, "own pages"),
                ({"origins": ["null"]}, 403, "own pages"),
                ({"marked": False}, 403, "own scripts"),
                ({"content_type": "text/plain"}, 415, "JSON"),
                ({"content_type": None}, 415, "JSON"),
                ({"body": None}, 413, "bytes at most"),
                ({"body": b"{not json"}, 400, "doesn't keep that"),
                ({"body": b"\xff"}, 400, "doesn't keep that"),
                ({"body": b'{"look": ["log"]}'}, 400, "isn't a name with a short word")):
            sent = {"body": b'{"look": "log"}'}
            sent.update(options)
            response = self.post(**sent)
            self.assertEqual(response.status, status, options)
            self.assertIn(why, response.body, options)
        self.assertEqual(self.kept(), (200, {"look": "native"}))  # as it was
        for path, method, status in (("lem/store/other", "POST", 405),
                                     ("lem/store/other", "GET", 404),
                                     ("lem/store/session?x=1", "POST", 405),
                                     ("lem/facts/", "POST", 405), ("", "POST", 405)):
            response = self.app.respond(method, f"/{TOKEN}/{path}", ["127.0.0.1:8765"],
                                        body=b"{}", **{k: v for k, v in self.OWN.items()})
            self.assertEqual(response.status, status, (path, method))
        self.assertEqual(self.app.respond("POST", "/lem/store/session", ["127.0.0.1:8765"],
                                          body=b"{}", **self.OWN).status, 404)  # no token

    def test_the_views_script_sends_what_the_store_asks_for(self):
        script = (view.UI_DIRECTORY / "web" / "contexts.js").read_text()
        self.assertIn(f'"{server.STORE_HEADER}": "1"', script)
        self.assertIn('getAttribute("data-store")', script)
        self.assertIn('"Content-Type": "application/json"', script)

    def test_a_server_that_keeps_no_store(self):
        app = server.App(contexts.Roots(self.tmp / "registry"), TOKEN)
        response = app.respond("POST", f"/{TOKEN}/lem/store/session", ["127.0.0.1:8765"],
                               body=b"{}", **self.OWN)
        self.assertEqual((response.status, response.body), (404, "This node server keeps no "
                                                            "store.\n"))
        response = app.respond("GET", f"/{TOKEN}/lem/store/session", ["127.0.0.1:8765"])
        self.assertEqual((response.status, response.body), (200, "{}"))


class GrowTests(unittest.TestCase):
    """A log that grows, as a page of the glui view that asks what it gained sees it, from
    one server throughout (view.py)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="node-grow-"))
        self.log = self.tmp / "s._private" / "context.jsonl"
        write_log(self.log, [init(KEY_A), message("user", "One."), BOUNDARY, *work("w1")])
        write_log(self.tmp / "s.jsonl", [init(KEY_A)])
        registry = self.tmp / "registry"
        self.assertEqual(Registry(registry).register(KEY_A, self.log.parent).status, "new")
        self.app = server.App(contexts.Roots(registry), TOKEN)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def get(self, path):
        return self.app.respond("GET", f"/{TOKEN}/{path}", ["127.0.0.1"])

    def steps(self, path):
        response = self.get(path)
        self.assertEqual((response.status, response.content_type), (200, "application/json"),
                         response.body[:300])
        return json.loads(response.body)

    def append(self, items):
        with open(self.log, "ab") as log:
            log.write("".join(json.dumps(item) + "\n" for item in items).encode("utf-8"))

    def grown(self):
        """The page's URL for what its log gained since it had a number of lines."""
        found = re.search(r'data-grow="/t0ken_x+/(lem/grow/1/[0-9]+/)\{lines\}"',
                          self.get("lem/c/1").body)
        self.assertIsNotNone(found)
        return found.group(1)

    @staticmethod
    def facts(steps, name):
        return [(step[0], fields) for step in steps for rel, fields in [step[1:]] if rel == name]

    def test_a_message_shows_whole_up_to_its_cut(self):
        text = " ".join(["word"] * 1200)  # 5,999 characters, past the plain pages' cut
        self.append([message("assistant", text), sample(), SAMPLE_END])

        def said():
            return [fields["text"] for _, rel, fields in self.steps("lem/facts/c/1")
                    if rel == "Message" and fields["text"].startswith("word")]

        self.assertEqual(said(), [text])
        with mock.patch.object(view, "MESSAGE_CUT", 1000):
            self.assertEqual(said(), [text[:1000] + "\n\n\u2026 4,999 more characters: the "
                                      "whole item is on its line's page."])

    def test_its_length_and_the_turns_that_changed(self):
        grow = self.grown()
        self.assertEqual(self.facts(self.steps("lem/facts/c/1"), "Length"), [("+", {"lines": 7})])
        self.assertEqual([rel for _, rel, _ in self.steps(grow + "7")], ["Activity"])  # no lines
        self.append([*answer(), summary(), *turn("Two.")])  # lines 8 to 21
        steps = self.steps(grow + "7")
        self.assertEqual(self.facts(steps, "Length"), [("-", {"lines": 7}), ("+", {"lines": 21})])
        self.assertEqual([(fields["n"], fields["label"], fields["title"], fields["line"])
                          for _, fields in self.facts(steps, "Turn")],
                         [(0, "The start · 1 item", "", 1), (1, "Turn 1 · 7 items", "One.", 2),
                          (2, "Turn 2 · 7 items", "Two.", 12)])
        self.assertEqual([rel for _, rel, _ in self.steps(grow + "21")], ["Activity"])
        self.append(turn("Three."))  # lines 22 to 31: turn 2's labels stay, and 3's come
        steps = self.steps(grow + "21")
        self.assertEqual([fields["n"] for _, fields in self.facts(steps, "Turn")], [1, 2, 3])
        program = json.loads(PROGRAM.read_text())
        self.assertEqual(step_problems(program, [step for step in steps if step[0] == "+"]), [])

    def test_a_page_of_a_log_that_was_replaced_loads_anew(self):
        grow = self.grown()
        for path, status in ((grow + "8", 409), (grow + "x", 404), (grow + "", 404),
                             ("lem/grow/1/x/7", 404), ("lem/grow/9/1/7", 404)):
            self.assertEqual(self.get(path).status, status, path)
        write_log(self.log, [init(KEY_A)])  # shorter: indexed anew
        response = self.get(grow + "1")
        self.assertEqual((response.status, response.body),
                         (409, "s (1)'s log was replaced: load its page anew.\n"))
        self.assertEqual([rel for _, rel, _ in self.steps(self.grown() + "1")], ["Activity"])

    def test_stars_set_and_cleared_from_the_page_by_the_roots_key(self):
        own = {"origins": ["http://127.0.0.1:8765"], "marked": True,
               "content_type": "application/json"}
        store = Store(self.tmp / "node")
        app = server.App(self.app.pages.roots, TOKEN, store=store)

        def post(path, body):
            return app.respond("POST", f"/{TOKEN}/{path}", ["127.0.0.1:8765"], body=body, **own)

        self.assertEqual(post("lem/stars/1", b'{"line": 2, "on": true}').status, 204)
        self.assertEqual(post("lem/stars/1", b'{"line": 5, "on": true}').status, 204)
        self.assertEqual(post("lem/stars/1", b'{"line": 2, "on": false}').status, 204)
        self.assertEqual(store.read("stars"), {KEY_A: [5]})
        steps = json.loads(app.respond("GET", f"/{TOKEN}/lem/facts/c/1", ["127.0.0.1"]).body)
        self.assertEqual([fields for _, rel, fields in steps if rel == "Stars"], [{"line": 5}])
        for path, body, status in (("lem/stars/9", b'{"line": 2, "on": true}', 404),
                                   ("lem/stars/1", b'{"line": 0, "on": true}', 400),
                                   ("lem/stars/1", b'{"line": 2, "on": 1}', 400),
                                   ("lem/stars/1", b'[2]', 400),
                                   ("lem/store/stars", b'{}', 405)):
            self.assertEqual(post(path, body).status, status, (path, body))
        elsewhere = dict(own, origins=["http://evil.example"])
        self.assertEqual(app.respond("POST", f"/{TOKEN}/lem/stars/1", ["127.0.0.1:8765"],
                                     body=b'{"line": 7, "on": true}', **elsewhere).status, 403)
        self.assertEqual(store.read("stars"), {KEY_A: [5]})

    def test_a_roots_label_and_tags_set_from_the_page_by_its_key(self):
        own = {"origins": ["http://127.0.0.1:8765"], "marked": True,
               "content_type": "application/json"}
        store = Store(self.tmp / "node")
        app = server.App(self.app.pages.roots, TOKEN, store=store)

        def post(path, body, **more):
            return app.respond("POST", f"/{TOKEN}/{path}", ["127.0.0.1:8765"], body=body,
                               **dict(own, **more))

        def get(path):
            return app.respond("GET", f"/{TOKEN}/{path}", ["127.0.0.1:8765"])

        self.assertEqual(post("lem/roots/1", b'{"label": "the bug", "tags": ["ui", "fix"]}')
                         .status, 204)
        self.assertEqual(store.read("roots"), {KEY_A: {"label": "the bug", "tags": ["fix", "ui"]}})
        index = json.loads(get("lem/facts/").body)
        self.assertEqual(step_problems(json.loads(PROGRAM.read_text()), index), [])
        self.assertEqual([(fields["label"], fields["tags"], fields["keep"])
                          for _, rel, fields in index if rel == "Context"],
                         [("the bug", "fix ui", f"/{TOKEN}/lem/roots/1")])
        self.assertEqual([fields for _, rel, fields in index if rel == "Tagged"],
                         [{"id": "s (1)", "tag": "fix"}, {"id": "s (1)", "tag": "ui"}])
        self.assertEqual([fields for _, rel, fields in index if rel == "Tag"],
                         [{"name": "fix", "rank": 0, "count": 1},
                          {"name": "ui", "rank": 1, "count": 1}])
        # Its log's page, its facts' and its tab's, say its label after its name.
        facts = json.loads(get("lem/facts/c/1").body)
        self.assertEqual([fields["title"] for _, rel, fields in facts if rel == "Page"],
                         ["s (1) · the bug"])
        self.assertEqual(json.loads(get("lem/tab/1").body)["title"], "s (1) · the bug")
        self.assertIn("<title>s (1) · the bug · node</title>", get("lem/c/1").body)
        for path, body, status in (("lem/roots/9", b'{"label": "x", "tags": []}', 404),
                                   ("lem/roots/1", b'{"label": 1, "tags": []}', 400),
                                   ("lem/roots/1", b'{"label": "x", "tags": ["X"]}', 400),
                                   ("lem/roots/1", b'[]', 400), ("lem/store/roots", b'{}', 405)):
            self.assertEqual(post(path, body).status, status, (path, body))
        self.assertEqual(post("lem/roots/1", b'{"label": "x", "tags": []}',
                              origins=["http://evil.example"]).status, 403)
        self.assertEqual(store.read("roots"), {KEY_A: {"label": "the bug", "tags": ["fix", "ui"]}})
        self.assertEqual(post("lem/roots/1", b'{"label": "", "tags": []}').status, 204)
        self.assertEqual(store.read("roots"), {})
        index = json.loads(get("lem/facts/").body)
        self.assertEqual([(rel, fields.get("label"), fields.get("tags")) for _, rel, fields in index
                          if rel in ("Context", "Tagged", "Tag")], [("Context", "", "")])
        self.assertEqual(json.loads(get("lem/tab/1").body)["title"], "s (1)")

    def test_pins_of_the_lines_starred_with_the_questions_that_an_answer_answers(self):
        store = Store(self.tmp / "node")
        app = server.App(self.app.pages.roots, TOKEN, store=store)
        self.append([*answer(), summary(), *turn("Two.")])  # lines 8 to 21: turn 2 from 12

        def pins(path):
            response = app.respond("GET", f"/{TOKEN}/{path}", ["127.0.0.1"])
            self.assertEqual(response.status, 200, response.body[:300])
            return [(fields["line"], fields["turn"], fields["tone"], fields["label"])
                    for _, rel, fields in json.loads(response.body) if rel == "Pin"]

        self.assertEqual(pins("lem/pins/1"), [])
        # Turn 1's answer, turn 2's tool call, a boundary, which shows nothing, and a line
        # past the log's end.
        for line in (8, 14, 3, 99):
            store.star(KEY_A, line, True)
        want = [(2, 1, "User", "[user]"), (8, 1, "Assistant", "[assistant]"),
                (14, 2, "Tool", "[tool-call] exec_command")]
        self.assertEqual(pins("lem/pins/1"), want)
        self.assertEqual(pins("lem/facts/c/1"), want)
        store.star(KEY_A, 18, True)  # turn 2's answer, with its question, line 12
        self.assertEqual([pin[0] for pin in pins("lem/pins/1")], [2, 8, 12, 14, 18])
        store.star(KEY_A, 12, True)  # the question starred too: still one pin
        self.assertEqual([pin[0] for pin in pins("lem/pins/1")], [2, 8, 12, 14, 18])
        steps = json.loads(app.respond("GET", f"/{TOKEN}/lem/pins/1", ["127.0.0.1"]).body)
        self.assertEqual(step_problems(json.loads(PROGRAM.read_text()), steps), [])
        for path in ("lem/pins/9", "lem/pins/", "lem/pins/x"):
            self.assertEqual(app.respond("GET", f"/{TOKEN}/{path}", ["127.0.0.1"]).status, 404,
                             path)

    def test_the_rows_after_a_run_that_went_on_start_with_it(self):
        # Lines 4 to 7 are a run, which the log ends with; four more lines go on with it.
        rows = self.steps("lem/rows/1/after/7")
        self.assertEqual(self.facts(rows, "Chunk"), [("+", {"first": 8, "last": 7})])
        self.append(work("w2"))  # lines 8 to 11
        rows = self.steps("lem/rows/1/after/7")
        self.assertEqual(self.facts(rows, "Chunk"), [("+", {"first": 4, "last": 11})])
        self.assertEqual([(fields["line"], fields["last"], fields["text"])
                          for _, fields in self.facts(rows, "Run")],
                         [(4, 11, "2 tool calls: exec_command 2; 2 other")])
        items = self.facts(rows, "Item")
        self.assertEqual([(fields["line"], fields["run"]) for _, fields in items],
                         [(8, 4), (9, 4), (11, 4)])
        self.assertEqual(self.facts(self.steps("lem/run/1/4"), "Item")[3:],
                         self.facts(rows, "Item"))
        # A message ends the run: the rows after it are new ones.
        self.append(answer("Done."))  # lines 12 to 14
        rows = self.steps("lem/rows/1/after/11")
        self.assertEqual(self.facts(rows, "Chunk"), [("+", {"first": 12, "last": 14})])
        self.assertEqual([(rel, fields["line"]) for _, rel, fields in rows if rel != "Chunk"],
                         [("Message", 12), ("Run", 13)])


class ActivityTests(unittest.TestCase):
    """What a root context is doing, for its dot (activity.py): by the lock that a CLI
    holds on its save, and what it says there; else by its log's tail and age."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="node-activity-"))
        self.addCleanup(shutil.rmtree, self.tmp)
        self.log = self.tmp / "s._private" / "context.jsonl"
        write_log(self.tmp / "s.jsonl", [init(KEY_A)])
        self.write([init(KEY_A), *turn("One.")])
        registry = self.tmp / "registry"
        self.assertEqual(Registry(registry).register(KEY_A, self.log.parent).status, "new")
        self.roots = contexts.Roots(registry)

    def write(self, items):
        write_log(self.log, items)

    def hue(self, ago=0.0):
        """The hue and why, with the log last changed ``ago`` seconds before."""
        context = self.roots.get("1")
        context.refresh()
        return activity.activity(context, now=context.index.changed + ago)

    def held(self, **noted):
        lock, _ = take_lock(self.log.parent)
        self.addCleanup(lock.close)
        lock.note(**noted)
        return lock

    def test_without_a_lock_file_it_guesses_from_the_logs_tail_and_age(self):
        guesses = []
        for items in ([*turn("One.")],  # between turns
                      [message("user", "Two."), BOUNDARY, *work("w2")],  # in a turn
                      [message("user", "Two."), BOUNDARY, failure(), SAMPLE_END]):  # it failed
            self.write([init(KEY_A), *items])
            for ago in (60, 7200, 3 * 86400):
                hue, why = self.hue(ago)
                self.assertTrue(why.endswith(" ago, and it has no lock, so this is a guess."), why)
                guesses.append((hue, why.split(".")[0], why.split("changed ")[1].split(",")[0]))
        self.assertEqual(guesses, [
            ("Yellow", "Waiting, it seems: between turns", "1 min ago"),
            ("Gray", "Inactive, it seems", "2 h ago"),
            ("Gray", "Inactive, it seems", "3 days ago"),
            ("Green", "Running, it seems: a turn is under way", "1 min ago"),
            ("Gray", "Inactive, it seems: a turn stopped", "2 h ago"),
            ("Gray", "Inactive, it seems: a turn stopped", "3 days ago"),
            ("Orange", "Retrying, it seems: a model request failed", "1 min ago"),
            ("Red", "Failed, it seems: its last turn ended in a failure", "2 h ago"),
            ("Red", "Failed, it seems: its last turn ended in a failure", "3 days ago")])

    def test_a_lock_that_no_cli_holds_says_none_runs(self):
        self.held().close()  # a CLI with the lock ran, and ended
        self.assertEqual(self.hue(), ("Gray", "Inactive: no CLI runs on it."))
        self.assertIn(("lock", self.log.parent / "1.lock"), self.roots.get("1").files())
        self.write([init(KEY_A), message("user", "Two."), BOUNDARY, *work("w2")])
        self.assertEqual(self.hue(), ("Red", "Failed: its CLI stopped in the middle of a turn."))
        self.write([init(KEY_A), message("user", "Two."), BOUNDARY, failure(), SAMPLE_END])
        self.assertEqual(self.hue(), (
            "Red", "Failed: its last turn ended in a failure, and no CLI runs on it."))

    def test_a_cli_that_holds_the_lock_says_what_it_is_doing(self):
        lock = self.held()
        seen = []
        for noted in ({}, {"phase": "sampling", "retrying": False},
                      {"phase": "tool: exec_command"}, {"phase": "sampling", "retrying": True},
                      {"phase": "failed", "retrying": False}, {"phase": "auth needed"},
                      {"phase": "idle"}):
            lock.note(**noted)
            hue, why = self.hue(7200)  # its log's age doesn't matter
            self.assertTrue(why.endswith(f" CLI PID {os.getpid()} on {socket.gethostname()}"
                                         + time.strftime(", since %Y-%m-%d %H:%M.", time.localtime(
                                             lock.owner["since"]))), why)
            seen.append((hue, why.split(" CLI PID")[0]))
        # A turn that ended with a yield to icron waits for a job to wake it.
        self.write([init(KEY_A), message("user", "Two."), BOUNDARY,
                    call("icron", "y1", {"call": "yield_('the build')"}), result("y1", "Yielded."),
                    summary()])
        seen.append(self.hue())
        self.assertEqual([each[:2] if len(each) == 2 else each for each in seen], [
            ("Green", "Running: a CLI runs on it."),
            ("Green", "Running: sampling."),
            ("Green", "Running: tool: exec_command."),
            ("Orange", "Retrying: a model request failed, and the CLI samples again."),
            ("Red", "Failed: the last turn ended in a failure; the CLI waits."),
            ("Yellow", "Waiting: the CLI needs a login."),
            ("Yellow", "Waiting: the CLI waits for input."),
            ("Yellow", "Waiting: it yielded to icron, until a job wakes it."
             + seen[-1][1].split("wakes it.")[1])])

    def test_the_index_and_a_log_say_it_and_a_growing_log_says_it_again(self):
        app = server.App(self.roots, TOKEN)

        def steps(path):
            return json.loads(app.respond("GET", f"/{TOKEN}/{path}", ["127.0.0.1"]).body)

        def said(found):
            return [(fields["id"], fields["hue"]) for _, rel, fields in found if rel == "Activity"]

        self.held().close()
        self.assertEqual(said(steps("lem/facts/")), [("s (1)", "Gray")])
        self.assertEqual(said(steps("lem/facts/c/1")), [("s (1)", "Gray")])
        page = app.respond("GET", f"/{TOKEN}/lem/c/1", ["127.0.0.1"]).body
        grow = re.search(r'data-grow="/t0ken_x+/(lem/grow/1/[0-9]+/)\{lines\}"', page).group(1)
        self.held(phase="sampling")  # a CLI runs on it now, though its log hasn't changed
        self.assertEqual(said(steps(grow + str(len(self.roots.get("1"))))), [("s (1)", "Green")])
        program = json.loads(PROGRAM.read_text())
        self.assertEqual(step_problems(program, steps("lem/facts/")), [])


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = make_workspace(tempfile.mkdtemp(prefix="node-test-"))
        registry = Registry(cls.ws.registry)
        odd = cls.ws.saves / "odd name?#%._private"
        write_log(odd / "context.jsonl", [init(KEY_I), *turn("Odd.")])
        assert registry.register(KEY_I, odd).number == 8
        cls.bad_name = sys.platform.startswith("linux")
        if cls.bad_name:  # a name that isn't UTF-8
            bad = Path(os.fsdecode(bytes(cls.ws.saves) + b"/bad\xff._private"))
            write_log(bad / "context.jsonl", [init(KEY_J), *turn("Bad.")])
            assert registry.register(KEY_J, bad).number == 9
        cls.store = cls.ws.tmp / "store"
        cls.server = server.Server(server.App(contexts.Roots(cls.ws.registry), TOKEN,
                                              store=Store(cls.store)), 0)
        cls.port = cls.server.server_port
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        shutil.rmtree(cls.ws.tmp)

    def get(self, path, **options):
        return request(self.port, path, **options)

    def test_the_token_is_needed(self):
        self.assertEqual(self.get(f"/{TOKEN}/")[0], 200)
        # (http.server itself reduces a leading // to /, so //TOKEN/ is the token's page.)
        for path in ("/", "/c/1", f"/{TOKEN[:-1]}/", f"/{TOKEN}x/", f"/%74{TOKEN[1:]}/",
                     f"/x/{TOKEN}/", "*"):
            status, headers, body = self.get(path)
            self.assertEqual((status, body), (404, b"Not found.\n"), path)
            self.assertEqual(headers["content-type"], "text/plain; charset=utf-8")
        status, headers, _ = self.get(f"/{TOKEN}")
        self.assertEqual((status, headers["location"]), (308, f"/{TOKEN}/"))

    def test_the_host_must_name_loopback(self):
        for hosts in ([f"localhost:{self.port}"], ["localhost:9000"], ["127.0.0.1:1"],
                      ["[::1]:9000"], ["LOCALHOST"], ["127.0.0.1"]):
            self.assertEqual(self.get(f"/{TOKEN}/", hosts=hosts)[0], 200, hosts)
        for hosts in ([f"evil.example:{self.port}"], ["localhost.evil.example:9000"],
                      [f"127.0.0.1.evil.example:{self.port}"], [], ["localhost", "localhost"],
                      ["10.0.0.1:8765"], ["localhost:9000x"]):
            status, _, body = self.get(f"/{TOKEN}/", hosts=hosts)
            self.assertEqual(status, 403, hosts)
            self.assertNotIn(TOKEN.encode(), body)

    def test_the_store_takes_post_from_the_views_own_pages(self):
        path = f"/{TOKEN}/lem/store/session"
        own = [("Origin", f"http://127.0.0.1:{self.port}"), (server.STORE_HEADER, "1"),
               ("Content-Type", "application/json")]
        status, headers, body = self.get(path, method="POST", headers=own,
                                         body=b'{"look": "native"}')
        self.assertEqual((status, body), (204, b""))
        self.assertEqual(headers["content-security-policy"], view.DATA_CSP)
        self.assertEqual(json.loads(self.get(path)[2]), {"look": "native"})
        self.assertEqual(stat.S_IMODE(os.stat(self.store / "session.json").st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(self.store).st_mode), 0o700)
        # Too long, or not saying how long: refused unread.
        for extra, sent in (([("Content-Length", str(MAX_STORE + 1))], b"{}"),
                            ([("Content-Length", "x")], b"{}")):
            status, _, _ = self.get(path, method="POST", headers=own + extra, body=sent)
            self.assertEqual(status, 413, extra)
        # Another page's: refused, whatever it sends.
        for headers in ([("Origin", "http://evil.example"), *own[1:]], own[1:], own[::2]):
            status, _, _ = self.get(path, method="POST", headers=headers, body=b"{}")
            self.assertEqual(status, 403, headers)
        self.assertEqual(json.loads(self.get(path)[2]), {"look": "native"})

    def test_only_get_and_head(self):
        for method in ("POST", "PUT", "DELETE", "PATCH", "OPTIONS"):
            status, headers, _ = self.get(f"/{TOKEN}/", method=method)
            self.assertEqual((status, headers.get("allow")), (405, "GET, HEAD"), method)
            self.assertEqual(self.get("/", method=method)[0], 404, method)
        get = self.get(f"/{TOKEN}/c/1")
        head = self.get(f"/{TOKEN}/c/1", method="HEAD")
        self.assertEqual((head[0], head[2]), (200, b""))
        self.assertEqual(head[1]["content-length"], str(len(get[2])))

    def test_headers(self):
        for path, status in ((f"/{TOKEN}/", 200), (f"/{TOKEN}/c/nothing", 404), ("/", 404)):
            got, headers, _ = self.get(path)
            self.assertEqual(got, status)
            self.assertEqual(headers["content-security-policy"], server.CSP)
            for name, value in (("x-content-type-options", "nosniff"),
                                ("cache-control", "no-store"),
                                ("referrer-policy", "no-referrer"), ("x-frame-options", "DENY")):
                self.assertEqual(headers[name], value, path)
        self.assertTrue(server.CSP.startswith("default-src 'none';"))
        self.assertNotIn("script-src", server.CSP)  # so no script runs
        self.assertEqual(self.get(f"/{TOKEN}/")[1]["content-type"], "text/html; charset=utf-8")

    def test_gzip_only_when_accepted(self):
        path = f"/{TOKEN}/c/1?line=47"  # over 64 KiB
        _, plain_headers, plain_body = self.get(path)
        self.assertGreater(len(plain_body), server.GZIP_BYTES)
        self.assertNotIn("content-encoding", plain_headers)
        _, headers, body = self.get(path, headers=[("Accept-Encoding", "br, gzip;q=0.5")])
        self.assertEqual(headers["content-encoding"], "gzip")
        self.assertEqual(gzip.decompress(body), plain_body)
        _, headers, _ = self.get(path, headers=[("Accept-Encoding", "gzip;q=0")])
        self.assertNotIn("content-encoding", headers)
        _, headers, _ = self.get(f"/{TOKEN}/c/2", headers=[("Accept-Encoding", "gzip")])
        self.assertNotIn("content-encoding", headers)  # too short to bother

    def test_nothing_but_roots_is_read(self):
        for path in (f"/{TOKEN}/c/..%2Fsecret", f"/{TOKEN}/c/../secret",
                     f"/{TOKEN}/c/%2E%2E/secret", f"/{TOKEN}/c/..%2F..%2F..%2Fetc%2Fpasswd",
                     f"/{TOKEN}/c/secret", f"/{TOKEN}/c/local-0", f"/{TOKEN}/c/saves%2Fsecret",
                     f"/{TOKEN}/c/{REPLACED}", f"/{TOKEN}/c/{REPLACED}?turn=1",
                     f"/{TOKEN}/c/{UNREADABLE}?line=2", f"/{TOKEN}/c/{GONE}",
                     f"/{TOKEN}/k/{KEY_C[:12]}", f"/{TOKEN}/k/{KEY_D[:12]}"):
            status, _, body = self.get(path)
            self.assertEqual(status, 404, path)
            self.assertNotIn(b"TOP SECRET", body)
        for path in (f"/{TOKEN}/x", f"/{TOKEN}/c/", f"/{TOKEN}/k/", f"/{TOKEN}/r/1"):
            self.assertEqual(self.get(path)[0], 404, path)

    def test_queries(self):
        for path in (f"/{TOKEN}/?x=1", f"/{TOKEN}/c/1?turn=1&turn=2",
                     f"/{TOKEN}/c/1?bogus=1", f"/{TOKEN}/k/{KEY_A[:12]}?line=2",
                     f"/{TOKEN}/c/1?" + "&".join(f"a{n}=1" for n in range(20))):
            self.assertEqual(self.get(path)[0], 400, path)
        self.assertEqual(self.get(f"/{TOKEN}/c/1?turn=2&page=1")[0], 200)

    def test_names_show_as_text(self):
        _, _, body = self.get(f"/{TOKEN}/")
        self.assertIn(f'<a href="/{TOKEN}/c/8">odd name?#%</a>', body.decode())
        status, _, page = self.get(f"/{TOKEN}/c/8?turn=1")
        self.assertEqual(status, 200)
        self.assertIn(b"<h1>odd name?#% (8): turn 1 of 1</h1>", page)
        if self.bad_name:
            self.assertIn(f'<a href="/{TOKEN}/c/9">bad<span class=esc>\\udcff</span></a>',
                          body.decode())
            status, _, page = self.get(f"/{TOKEN}/c/9?turn=1")
            self.assertEqual(status, 200)
            self.assertIn(b"Bad.", page)

    def test_every_page_answers_and_nothing_is_written(self):
        before = snapshot(self.ws.tmp)
        shown = []  # pages that show a save that isn't a root

        def look(path, body):
            if b"TOP SECRET" in body:
                shown.append(path)

        statuses = crawl(self.port, f"/{TOKEN}/", look=look)
        self.assertEqual({path: status for path, status in statuses.items()
                          if status not in (200, 303)}, {})
        self.assertGreater(len(statuses), 100)
        self.assertEqual(shown, [])
        for style in ("log-light", "chat"):  # through the menu
            self.assertTrue(any(f"style={style}" in path for path in statuses))
        for turn_page in (f"/{TOKEN}/c/1?turn={number}" for number in range(5)):
            self.assertIn(turn_page, statuses)
        for cid in (OLD, PROJ, "8"):
            self.assertIn(f"/{TOKEN}/c/{cid}", statuses)
        self.assertEqual(snapshot(self.ws.tmp), before)

    def test_a_page_in_a_style(self):
        default = self.get(f"/{TOKEN}/c/1?turn=1")[2].decode()
        self.assertIn("style: Log dark</summary>", default)
        self.assertIn("color-scheme:only dark", default)
        for style in styles.BUILT_IN:
            first_line = (NODE / "css" / f"{style.name}.css").read_text().split("\n", 1)[0]
            status, _, body = self.get(f"/{TOKEN}/c/1?turn=1&style={style.name}")
            self.assertEqual(status, 200)
            self.assertIn(first_line, body.decode())
            self.assertIn(f"style: {style.label}</summary>", body.decode())
            if style.name != "log-dark":
                self.assertNotIn(first_line, default)
        for path in (f"/{TOKEN}/?style=nope", f"/{TOKEN}/?style=", f"/{TOKEN}/?style=base",
                     f"/{TOKEN}/?style=log", f"/{TOKEN}/?style=chat&style=log-dark",
                     f"/{TOKEN}/?style=%3Cscript%3E",
                     f"/{TOKEN}/?style=CHAT", f"/{TOKEN}/?style=..%2Fbase",
                     f"/{TOKEN}/c/1?turn=1&style=chat&bogus=1"):
            status, _, body = self.get(path)
            self.assertEqual(status, 400, path)
            self.assertNotIn(b"<script>", body)
        for style in ("log-light", "chat"):
            for path, location in ((f"/{TOKEN}/k/{KEY_B[:14]}?style={style}", "c/2"),
                                   (f"/{TOKEN}/c/{SECOND}?style={style}", "c/1")):
                status, headers, _ = self.get(path)
                self.assertEqual((status, headers["location"]),
                                 (303, f"/{TOKEN}/{location}?style={style}"))

    def test_the_servers_own_style(self):
        for style in ("log-light", "chat"):
            app = server.App(contexts.Roots(self.ws.registry), TOKEN, style=style)
            body = app.respond("GET", f"/{TOKEN}/c/1?turn=1", ["127.0.0.1"]).body
            tag = "details" if style == "chat" else "div"
            self.assertIn(f'<{tag} class="it tool t-tool_call" id="L6">', body)
            self.assertIn(f'<a href="/{TOKEN}/c/1?turn=1&amp;style=log-dark">Log dark', body)
            # Its links keep its style without naming it.
            self.assertNotIn("style=", outside(body))
            overview = app.respond("GET", f"/{TOKEN}/c/1", ["127.0.0.1"]).body
            self.assertNotIn("type=hidden", overview)

    def test_a_ninth_request_at_once_is_refused(self):
        held = [socket.create_connection(("127.0.0.1", self.port))
                for _ in range(server.MAX_REQUESTS)]
        try:
            deadline = time.monotonic() + 10
            while self.server.slots._value and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(self.server.slots._value, 0)
            self.assertEqual(self.get(f"/{TOKEN}/")[0], 503)
        finally:
            for connection in held:
                connection.close()
        deadline = time.monotonic() + 10
        while self.server.slots._value < server.MAX_REQUESTS and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.get(f"/{TOKEN}/")[0], 200)

    def test_an_idle_connection_is_closed(self):
        with mock.patch.object(server, "SOCKET_TIMEOUT", 0.5):
            with socket.create_connection(("127.0.0.1", self.port)) as idle:
                idle.settimeout(10)
                started = time.monotonic()
                self.assertEqual(idle.recv(1), b"")
                self.assertLess(time.monotonic() - started, 5)

    def test_each_route_has_its_policy_and_type(self):
        for path, csp, kind in (
                (f"/{TOKEN}/", server.CSP, "text/html"), (f"/{TOKEN}/c/1", server.CSP, "text/html"),
                (f"/{TOKEN}/lem/", view.PAGE_CSP, "text/html"),
                (f"/{TOKEN}/lem/c/1", view.PAGE_CSP, "text/html"),
                (f"/{TOKEN}/lem/facts/", view.DATA_CSP, "application/json"),
                (f"/{TOKEN}/lem/facts/c/1", view.DATA_CSP, "application/json"),
                (f"/{TOKEN}/lem/body/1/10", view.DATA_CSP, "application/json"),
                (f"/{TOKEN}/lem/files/lem.js", view.DATA_CSP, "text/javascript"),
                (f"/{TOKEN}/lem/files/glui.css", view.DATA_CSP, "text/css"),
                (f"/{TOKEN}/lem/files/compiled/contexts.json", view.DATA_CSP, "application/json"),
                (f"/{TOKEN}/lem/files/nope.js", view.DATA_CSP, "text/plain")):
            _, headers, _ = self.get(path)
            self.assertEqual(headers["content-security-policy"], csp, path)
            self.assertEqual(headers["content-type"], f"{kind}; charset=utf-8", path)
            self.assertEqual(headers["x-content-type-options"], "nosniff", path)
        # The view's pages run the server's scripts, styles and fetches, and nothing else;
        # its facts, bodies and files, nothing at all.
        self.assertEqual(view.PAGE_CSP, "default-src 'none'; script-src 'self'; style-src 'self'; "
                         "connect-src 'self'; base-uri 'none'; form-action 'none'; "
                         "frame-ancestors 'none'")
        self.assertEqual(view.DATA_CSP, "default-src 'none'; frame-ancestors 'none'")
        self.assertEqual(self.get(f"/{TOKEN}/lem")[1]["location"], f"/{TOKEN}/lem/")
        for path in (f"/{TOKEN}/lem/?x=1", f"/{TOKEN}/lem/facts/c/1?line=2"):
            self.assertEqual(self.get(path)[0], 400, path)
        for path in ("/lem/", f"/{TOKEN[:-1]}/lem/", "/lem/files/lem.js", f"/x/{TOKEN}/lem/"):
            self.assertEqual(self.get(path)[0::2], (404, b"Not found.\n"), path)
        head = self.get(f"/{TOKEN}/lem/files/lem.js", method="HEAD")
        self.assertEqual((head[0], head[2]), (200, b""))

    def test_the_server_names_only_itself(self):
        self.assertEqual(self.get(f"/{TOKEN}/")[1]["server"], "node")


class OptionTests(WorkspaceTest):
    def node(self, *argv, **options):
        """``python3 -m pythia.interaction.node``, in a process of its own, from this checkout."""
        return subprocess.run([sys.executable, "-m", "pythia.interaction.node", *argv],
                              env=environ(), capture_output=True, text=True, timeout=60,
                              **options)

    def test_the_style_option(self):
        for name in ("nope", "log"):
            status, _, err = run_main("--style", name, "--list", "--registry",
                                      str(self.ws.registry))
            self.assertEqual(status, 2)
            self.assertIn(f"argument --style: {name!r} isn't a style: give one of "
                          "log-dark, log-light, chat", err)
        for name in ("log-dark", "log-light", "chat"):
            status, out, _ = run_main("--style", name, "--list", "--registry",
                                      str(self.ws.registry))
            self.assertEqual(status, 0)
            self.assertIn("local-0", out)
        status, out, _ = run_main("--help")
        self.assertEqual(status, 0)
        self.assertIn("log-dark, log-light, chat", " ".join(out.split()))
        self.assertIn("default:log-dark", "".join(out.split()))

    def test_list(self):
        with mock.patch.dict(os.environ, {"COLUMNS": "80"}):
            status, out, err = run_main("--list", "--registry", str(self.ws.registry))
        self.assertEqual((status, err), (0, ""))
        lines = out.splitlines()
        self.assertEqual(lines[0], f"Root contexts in {contexts.shown_path(self.ws.registry)}/: "
                         "6, the last changed first. The 3 that aren't where they're "
                         "registered come last.")
        self.assertEqual(lines[1].split(), ["#", "context", "key", "changed", "turns", "title"])
        row = re.compile(r" +(?P<number>[0-9]+)  (?P<name>\S+) +(?P<key>context_\S+)  "
                         r"(?P<changed>[0-9-]{10} [0-9:]{5})  +(?P<turns>[0-9]+)  (?P<title>.*)")
        rows = {match["name"]: match for match in map(row.fullmatch, lines[2:5])}
        self.assertEqual(sorted(rows), ["local-0", "old", "proj"])
        self.assertEqual((rows["local-0"]["number"], rows["local-0"]["turns"]), ("1", "4"))
        self.assertEqual((rows["old"]["key"], rows["old"]["title"]),
                         ("context_b2b2", "Old question."))
        self.assertTrue(rows["local-0"]["title"].endswith("…"))  # cut to fit
        self.assertEqual([line.split()[:4] for line in lines[5:]],
                         [["4", "-", "context_f6f6", "missing"],
                          ["5", "-", "context_9797", "replaced"],
                          ["6", "-", "context_0808", "unreadable"]])
        self.assertTrue(lines[5].endswith("/saves/gone"), lines[5])  # cut at its start
        self.assertEqual(len(lines), 8)  # 7, a second number of 1's, isn't a root
        self.assertLessEqual(max(map(len, lines[1:])), 80)

    def test_list_binds_no_port(self):
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            port = str(taken.getsockname()[1])
            status, out, _ = run_main("--list", "--port", port, "--registry",
                                      str(self.ws.registry))
        self.assertEqual(status, 0)
        self.assertIn("local-0", out)

    def test_the_registry_is_found_as_the_cli_finds_it(self):
        off = ("node: the registry is off (--registry none, or PYTHIA_REGISTRY=none), and the "
               "node server shows its root contexts: give one with --registry.\n")
        with mock.patch.dict(os.environ, {"HOME": str(self.ws.tmp), "COLUMNS": "200"}):
            os.environ.pop("PYTHIA_REGISTRY", None)
            self.assertEqual(run_main("--list"),
                             (0, "There's no registry at ~/.pythia/root/ yet.\n", ""))
            os.environ["PYTHIA_REGISTRY"] = "~/registry"
            status, out, _ = run_main("--list")
            self.assertTrue(out.startswith("Root contexts in ~/registry/: 6,"), out)
            self.assertEqual(run_main("--list", "--registry", "none"), (1, "", off))
            with chdir(self.ws.tmp):  # a relative path is taken from the launch directory
                status, out, _ = run_main("--list", "--registry", "registry")
                shown = contexts.shown_path(Path.cwd() / "registry")
                self.assertTrue(out.startswith(f"Root contexts in {shown}/: 6,"), out)
            os.environ["PYTHIA_REGISTRY"] = "none"
            self.assertEqual(run_main("--list"), (1, "", off))
            status, out, _ = run_main("--list", "--registry", str(self.ws.registry))
            self.assertEqual(status, 0)
            self.assertIn("local-0", out)

    def test_registry_problems_never_stop_it(self):
        missing = self.ws.tmp / "no-registry-yet"
        self.assertEqual(run_main("--list", "--registry", str(missing)),
                         (0, f"There's no registry at {contexts.shown_path(missing)}/ yet.\n",
                          ""))
        self.assertFalse(missing.exists())
        if os.geteuid() != 0:
            locked = self.ws.tmp / "locked-registry"
            locked.mkdir(mode=0)
            try:
                status, out, err = run_main("--list", "--registry", str(locked))
            finally:
                locked.chmod(0o700)
                locked.rmdir()
            self.assertEqual((status, out, err),
                             (0, f"The registry {contexts.shown_path(locked)}/ can't be read: "
                              "Permission denied.\n", ""))

    def test_ports(self):
        for port in ("70000", "-1", "eighty", ""):
            status, _, err = run_main("--port", port, "--registry", str(self.ws.registry))
            self.assertEqual(status, 2, port)
            self.assertIn(f"argument --port: {port!r} isn't a port: give 0 to 65535", err)

    def test_a_taken_port_is_an_error_not_a_fallback(self):
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            port = taken.getsockname()[1]
            done = self.node("--port", str(port), "--registry", str(self.ws.registry))
        self.assertEqual((done.returncode, done.stdout), (1, ""))
        self.assertIn(f"node: can't serve on 127.0.0.1:{port}: ", done.stderr)
        self.assertIn("; give another port with --port, or --port 0 for a free one.",
                      done.stderr)

    @unittest.skipUnless(sys.platform.startswith("linux") and os.geteuid() != 0
                         and unprivileged_ports() > 80, "port 80 may be had here")
    def test_a_privileged_port_is_an_error(self):
        # In a process, so that a server that starts can't hang this one.
        done = self.node("--port", "80", "--registry", str(self.ws.registry))
        self.assertEqual((done.returncode, done.stdout), (1, ""))
        self.assertIn("node: can't serve on 127.0.0.1:80: ", done.stderr)

    def test_serving_says_where_and_stops_on_ctrl_c(self):
        process = subprocess.Popen(
            [sys.executable, "-c", RUN_NODE, "--port", "0", "--registry", str(self.ws.registry)],
            cwd=self.ws.tmp, env=environ(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True)
        lines = queue.Queue()
        threading.Thread(target=lambda: [lines.put(line) for line in process.stdout],
                         daemon=True).start()
        try:
            printed = [lines.get(timeout=60) for _ in range(4)]
            self.assertEqual(printed[0], f"node: 6 root contexts in "
                             f"{contexts.shown_path(self.ws.registry)}/; 3 aren't where they're "
                             "registered.\n")
            match = re.fullmatch(r"node: (http://127\.0\.0\.1:([0-9]+)/[A-Za-z0-9_-]{43}/)\n",
                                 printed[1])
            self.assertIsNotNone(match, printed[1])
            port = int(match[2])
            self.assertNotEqual(port, 0)
            self.assertEqual(printed[2], f"node: in glui's view: {match[1]}lem/\n")
            self.assertEqual(printed[3], f"node: from another machine: ssh -L {port}:127.0.0.1:"
                             f"{port} HOST\n")
            with urllib.request.urlopen(match[1], timeout=20) as response:
                self.assertEqual(response.status, 200)
                self.assertIn(b"local-0", response.read())
            with urllib.request.urlopen(match[1] + "lem/", timeout=20) as response:
                self.assertEqual(response.headers["Content-Security-Policy"], view.PAGE_CSP)
            process.send_signal(signal.SIGINT)
            self.assertEqual(process.wait(timeout=60), 0)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()

    def test_runs_from_anywhere(self):
        done = self.node("--list", "--registry", "registry", cwd=self.ws.tmp)
        self.assertEqual(done.returncode, 0, done.stderr)
        shown = contexts.shown_path(Path(os.path.realpath(self.ws.tmp)) / "registry")
        self.assertTrue(done.stdout.startswith(f"Root contexts in {shown}/: 6,"), done.stdout)


@unittest.skipUnless(os.environ.get("PYTHIA_NODE_JS"), "PYTHIA_NODE_JS names no JavaScript shell")
class ViewInJavaScriptTests(WorkspaceTest):
    """The view's facts and bodies, through glui's scripts and the program that the view
    runs, in a JavaScript shell (PYTHIA_NODE_JS, such as qjs), with the stand-in DOM of
    _pythia_ui/tools/stand_in_dom.js in place of a browser's."""

    def test_the_program_takes_the_facts_and_bodies(self):
        app = server.App(contexts.Roots(self.ws.registry), TOKEN)

        def steps(path):
            return json.loads(app.respond("GET", f"/{TOKEN}/{path}", ["127.0.0.1"]).body)

        log = steps("lem/facts/c/1")
        runs = {fields["line"]: steps(f"lem/run/1/{fields['line']}")
                for _, rel, fields in log if rel == "Run"}
        items = [fields for loaded in runs.values() for _, rel, fields in loaded
                 if rel == "Item"]
        lones = [fields for _, rel, fields in log if rel == "Item"]  # compactions' rows
        marks = [fields for _, rel, fields in log if rel == "Turn"]
        messages = [fields for _, rel, fields in log if rel == "Message"]
        bodies = {item["line"]: steps(f"lem/body/1/{item['line']}")
                  for item in items + lones if item["body"]}
        node = {"program": json.loads(PROGRAM.read_text()), "index": steps("lem/facts/"),
                "log": log, "runs": runs, "bodies": bodies}
        scripts = ("src/lem/lem.js", "src/glem/glem.js", "src/glui/glui.js",
                   "tools/stand_in_dom.js", "web/contexts.js")
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run.js"
            run.write_text("".join((view.UI_DIRECTORY / name).read_text() for name in scripts)
                           + f"var NODE = {json.dumps(node)};\n" + JS_RUN)
            done = subprocess.run([*shlex.split(os.environ["PYTHIA_NODE_JS"]), str(run)],
                                  capture_output=True, text=True, timeout=600)
        said = [line for line in (done.stdout + done.stderr).splitlines()
                if line.startswith("NODE ")]
        self.assertEqual(len(said), 1, done.stdout + done.stderr)
        got = json.loads(said[0][len("NODE "):])
        self.assertEqual(got["index"], {"wrong": [], "entries": 0, "codes": 0, "rows": 4,
                                        "ran": 0, "opened": 0})
        self.assertEqual(got["log"], {
            "wrong": [], "entries": len(marks) + len(messages) + len(lones) + len(runs)
            + len(items),
            "codes": len(messages) + sum(map(len, bodies.values())), "rows": 0,
            "ran": len(runs), "opened": len(bodies)})


@unittest.skipUnless(os.environ.get("PYTHIA_LIVE_NODE_REGISTRY"),
                     "PYTHIA_LIVE_NODE_REGISTRY names no registry")
class LiveTests(unittest.TestCase):
    def test_every_page_answers(self):
        path = Path(os.environ["PYTHIA_LIVE_NODE_REGISTRY"]).expanduser().absolute()
        roots = contexts.Roots(path)
        live = server.Server(server.App(roots, TOKEN), 0)
        threading.Thread(target=live.serve_forever, daemon=True).start()
        try:
            started = time.monotonic()
            statuses = crawl(live.server_port, f"/{TOKEN}/",
                             keep=lambda link: "line=" not in link and "lines=" not in link)
            program, problems = json.loads(PROGRAM.read_text()), {}

            def facts(path):  # the view's: they must suit its program
                status, _, body = request(live.server_port, path)
                statuses[path] = status
                steps = json.loads(body) if status == 200 else []
                problems.update({path: step_problems(program, steps)} if steps else {})
                return steps

            facts(f"/{TOKEN}/lem/facts/")
            for context in roots.contexts():
                count = len(context.refresh())
                for line in sorted({1, (count + 1) // 2, count} - {0}):
                    page = live.app.pages.url(context.id, f"line={line}")
                    statuses[page] = request(live.server_port, page)[0]
                steps = facts(f"/{TOKEN}/lem/facts/c/{context.id}")
                first = [fields["first"] for _, rel, fields in steps if rel == "Chunk"][0]
                if first > 1:
                    steps += facts(f"/{TOKEN}/lem/rows/{context.id}/before/{first}")
                facts(f"/{TOKEN}/lem/rows/{context.id}/after/0")
                runs = [fields["line"] for _, rel, fields in steps if rel == "Run"]
                lines = []
                for run in sorted(set(runs[:3] + runs[-3:] + runs[::max(1, len(runs) // 8)])):
                    lines += [fields["line"] for _, rel, fields
                              in facts(f"/{TOKEN}/lem/run/{context.id}/{run}")
                              if rel == "Item" and fields["body"]]
                for line in sorted(set(lines[:5] + lines[-5:] + lines[::max(1, len(lines) // 10)])):
                    facts(f"/{TOKEN}/lem/body/{context.id}/{line}")
            elapsed = time.monotonic() - started
            self.assertEqual({page: status for page, status in statuses.items()
                              if status not in (200, 303)}, {})
            self.assertEqual({path: found for path, found in problems.items() if found}, {})
            print(f"\nlive: {len(statuses)} pages of the root contexts in {path}/, in "
                  f"{elapsed:.1f} s", file=sys.stderr)
        finally:
            live.shutdown()
            live.server_close()


if __name__ == "__main__":
    unittest.main()
