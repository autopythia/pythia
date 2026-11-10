"""The node server's settings, ``~/.pythia/node.ini``
(``pythia/interaction/node/config.py``): what each key says, what isn't a setting, where
the file is found, the options over it, and the command that reaches a remote, as the
remote's shell runs it."""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from pythia.interaction.node import config, remote, server, view
from pythia.interaction.node.config import Invalid, Settings, parse, read
from pythia.interaction.node.contexts import Roots, shown_path
from pythia.interaction.node.remote import RUN, Way, node_words
from pythia.interaction.node.store import Store
from pythia_test.test_interaction_node import (CHECKOUT, RUN_NODE, chdir, environ,
                                               make_workspace, run_main)
from pythia_test.test_interaction_node_remote import node

HOME = Path("/home/u")
# The node server that a remote runs, as ssh hands its words to the remote's shell.
THERE = ["python3", "-c", f"'{RUN}'", "--stdio"]


def ssh(*words):
    return ["ssh", "-o", "BatchMode=yes", *words]


class ParseTests(unittest.TestCase):
    def test_a_remote_with_no_keys_is_reached_by_its_name_with_pythia_as_installed(self):
        installed = ["env", "PYTHONPATH=.pythia/lib/pythia-dev", *THERE]
        self.assertEqual(parse("[remote cloud-0]\n"),
                         Settings(remotes=(Way("cloud-0", ssh("cloud-0", *installed)),)))
        self.assertEqual(remote.parse("cloud-0"),
                         Way("cloud-0", ssh("cloud-0", *installed)))
        self.assertEqual(parse(""), Settings())

    def test_every_key(self):
        text = """# This node's server.
[node]
registry = reg
store = ~/kept
port = 8766
style = log-light

; A remote, every key but command.
[remote cloud-0]
host = cloud-0.example.org
user = me
port = 2222
identity = ~/.ssh/id_cloud
options = -o ProxyJump=gateway
  -o "ServerAliveInterval 15"
pythia = ~/src/pythia
python = "/opt/python 3/bin/python3" -X utf8
registry = /data/root
timeout = 12.5

[remote sim]
command = python3 -m pythia.interaction.node --stdio --registry 'a b'
timeout = 5
"""
        self.assertEqual(parse(text, home=HOME), Settings(
            None, "/home/u/reg", "/home/u/kept", 8766, "log-light", (
                Way("cloud-0", ssh("-l", "me", "-p", "2222", "-i", "/home/u/.ssh/id_cloud",
                                   "-o", "ProxyJump=gateway", "-o", "ServerAliveInterval 15",
                                   "cloud-0.example.org", "env", "PYTHONPATH=src/pythia",
                                   "'/opt/python 3/bin/python3'", "-X", "utf8", "-c",
                                   f"'{RUN}'", "--stdio", "--registry", "/data/root"), 12.5),
                Way("sim", ["python3", "-m", "pythia.interaction.node", "--stdio",
                            "--registry", "a b"], 5.0))))

    def test_paths_here_and_there(self):
        def here(key, value):
            return getattr(parse(f"[node]\n{key} = {value}\n", home=HOME), key)

        def there(key, value):
            return parse(f"[remote r]\n{key} = {value}\n", home=HOME).remotes[0].argv[4:]

        for value, path in (("/srv/root", "/srv/root"), ("reg", "/home/u/reg"),
                            ("~/reg", "/home/u/reg"), ("~", "/home/u"), ("none", "none")):
            with self.subTest(value=value):
                self.assertEqual(here("registry", value), path)
                self.assertEqual(here("store", value), path)
        identity = parse("[remote r]\nidentity = k\n", home=HOME).remotes[0].argv
        self.assertEqual(identity[3:5], ["-i", "/home/u/k"])
        for value, words in (("", THERE), ("~", ["env", "PYTHONPATH=.", *THERE]),
                             ("~/", ["env", "PYTHONPATH=.", *THERE]),
                             ("/opt/p", ["env", "PYTHONPATH=/opt/p", *THERE]),
                             ("my pythia", ["env", "PYTHONPATH='my pythia'", *THERE])):
            with self.subTest(pythia=value):
                self.assertEqual(there("pythia", value), words)
        self.assertEqual(there("registry", "~/r")[-2:], ["--registry", "r"])
        self.assertEqual(there("registry", "a b")[-2:], ["--registry", "'a b'"])

    def test_what_isnt_a_setting(self):
        name = "isn't a node's name: letters, digits, '.', '_' and '-', from a letter or a digit"
        keys = ("its keys are host, user, port, identity, options, pythia, python, registry, "
                "timeout and command")
        for text, why in (
                ("port = 1\n", "line 1: 'port = 1' comes before any [section]"),
                ("[node]\nhost\n", "line 2: 'host' isn't a [section], or a key = value"),
                ("[node]\n[node]\n", "line 2: [node] comes twice"),
                ("[node]\n[ node ]\n", "[ node ]: [node] comes twice"),
                ("[node]\nport = 1\nport = 2\n", "line 3: [node] has port twice"),
                ("[nodes]\n", "[nodes] isn't a section of node.ini: give [node], or "
                              "[remote NAME]"),
                ("[DEFAULT]\nport = 1\n", "[DEFAULT] isn't a section of node.ini: give "
                                          "[node], or [remote NAME]"),
                ("[remote a b]\n", "[remote a b] isn't a section of node.ini: give [node], "
                                   "or [remote NAME]"),
                ("[remote -a]\n", f"[remote -a]: '-a' {name}"),
                ("[remote a@b]\n", f"[remote a@b]: 'a@b' {name}"),
                ("[remote a]\n[remote  a]\n", "[remote  a]: remote a comes twice"),
                ("[node]\nPort = 1\n", "[node] has no Port: its keys are registry, store, "
                                       "port and style"),
                ("[remote a]\nhosts = b\n", f"[remote a] has no hosts: {keys}"),
                ("[node]\nport = x\n", "[node] port: 'x' isn't a port: give 0 to 65535"),
                ("[node]\nport = 65536\n", "[node] port: '65536' isn't a port: give 0 to "
                                           "65535"),
                ("[remote a]\nport = 0\n", "[remote a] port: '0' isn't a port: give 1 to "
                                           "65535"),
                ("[node]\nstore =\n", "[node] store: give a path, on one line, or none"),
                ("[node]\nregistry = a\n b\n", "[node] registry: give a path, on one line, "
                                               "or none"),
                ("[node]\nstyle = a b\n", "[node] style: give a style's name"),
                ("[remote a]\ntimeout = 0\n", "[remote a] timeout: '0' isn't a timeout: give "
                                              "seconds, more than 0 and 3600 at most"),
                ("[remote a]\ntimeout = 1e3\n", "[remote a] timeout: '1e3' isn't a timeout: "
                                                "give seconds, more than 0 and 3600 at most"),
                ("[remote a]\ntimeout = 3600.5\n", "[remote a] timeout: '3600.5' isn't a "
                                                   "timeout: give seconds, more than 0 and "
                                                   "3600 at most"),
                ("[remote a]\nhost = -oProxyCommand=x\n",
                 "[remote a] host: '-oProxyCommand=x' isn't a host that ssh takes: a word "
                 "that doesn't start with '-'"),
                ("[remote a]\nuser = a b\n", "[remote a] user: 'a b' isn't a user that ssh "
                                             "takes: a word that doesn't start with '-'"),
                ("[remote a]\noptions = 'x\n", "[remote a] options: No closing quotation"),
                ("[remote a]\npython =\n", "[remote a] python: give a command"),
                ("[remote a]\npythia = ~bob/p\n", "[remote a] pythia: '~bob/p': give a path "
                                                  "from the home directory there, or an "
                                                  "absolute one"),
                ("[remote a]\nregistry =\n", "[remote a] registry: give a path, on one line"),
                ("[remote a]\ncommand =\n", "[remote a] command: give a command"),
                ("[remote a]\ncommand = x\nhost = y\nuser = z\n",
                 "[remote a]: its command reaches it in the place of host and user: give "
                 "one or the other")):
            with self.subTest(text=text):
                with self.assertRaises(Invalid) as caught:
                    parse(text, "f.ini")
                self.assertEqual(str(caught.exception), f"f.ini: {why}")


class ReadTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="node-config-"))
        self.addCleanup(shutil.rmtree, self.home)
        patched = mock.patch.dict(os.environ, {"HOME": str(self.home)})
        patched.start()
        self.addCleanup(patched.stop)
        os.environ.pop(config.ENV, None)

    def test_the_default_file_if_its_there_else_another_or_none(self):
        self.assertEqual(read(), Settings())
        default = self.home / ".pythia" / "node.ini"
        default.parent.mkdir()
        default.write_text("[node]\nport = 2\n")
        other = self.home / "other.ini"
        other.write_text("[node]\nport = 1\n")
        self.assertEqual(read(), Settings(path=default, port=2))
        self.assertEqual(read(str(other)), Settings(path=other, port=1))
        self.assertEqual(read("none"), Settings())
        with chdir(self.home):  # a relative path is from the launch directory
            self.assertEqual(read("other.ini"), Settings(path=other, port=1))
        os.environ[config.ENV] = "~/other.ini"
        self.assertEqual(read(), Settings(path=other, port=1))
        self.assertEqual(read(str(default)), Settings(path=default, port=2))  # the option first
        os.environ[config.ENV] = " none "
        self.assertEqual(read(), Settings())
        os.environ[config.ENV] = ""
        self.assertEqual(read(), Settings(path=default, port=2))

    def test_what_cant_be_read_says_where(self):
        (self.home / "dir").mkdir()
        (self.home / "long.ini").write_text("#" * config.MAX_BYTES + "\n")
        (self.home / "latin.ini").write_bytes(b"[node]\nstyle = caf\xe9\n")
        (self.home / "bad.ini").write_text("[node]\nport = x\n")
        for name, why in (
                ("missing.ini", "there's no such file"),
                ("dir", "it can't be read: Configuration must be a regular file"),
                ("long.ini", "it can't be read: Configuration exceeds the size limit"),
                ("latin.ini", "it isn't UTF-8 text"),
                ("bad.ini", "[node] port: 'x' isn't a port: give 0 to 65535")):
            with self.subTest(name=name):
                with self.assertRaises(Invalid) as caught:
                    read(str(self.home / name))
                self.assertEqual(str(caught.exception), f"~/{name}: {why}")
        if os.geteuid() != 0:
            locked = self.home / "locked.ini"
            locked.write_text("[node]\n")
            locked.chmod(0)
            with self.assertRaises(Invalid) as caught:
                read(str(locked))
            self.assertEqual(str(caught.exception), "~/locked.ini: it can't be read: "
                                                    "Permission denied")


class ServedTests(unittest.TestCase):
    """The settings as the server takes them, and the command that reaches a remote."""

    @classmethod
    def setUpClass(cls):
        # A space in its path, for the commands' quoting.
        cls.ws = make_workspace(tempfile.mkdtemp(prefix="node config "))
        cls.roots = len(Roots(cls.ws.registry).entries())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.ws.tmp)

    def settings(self, text):
        """A settings file that says ``text``: its path."""
        handle, path = tempfile.mkstemp(suffix=".ini", dir=self.ws.tmp)
        with os.fdopen(handle, "w") as file:
            file.write(text)
        self.addCleanup(os.unlink, path)
        return path

    def test_the_command_runs_a_node_as_the_remotes_shell_takes_it(self):
        start = self.ws.tmp / "start"  # where ssh would start: a file there shadows nothing
        start.mkdir(exist_ok=True)
        (start / "json.py").write_text("raise SystemExit('shadowed by the launch directory')\n")
        hello = json.dumps({"node": 1, "base": "/t0ken/n/x"}) + "\n"
        for pythia, registry in ((str(CHECKOUT), str(self.ws.registry)),
                                 (os.path.relpath(CHECKOUT, start),
                                  os.path.relpath(self.ws.registry, start))):
            with self.subTest(pythia=pythia, registry=registry):
                words = node_words(pythia=pythia, python=[sys.executable], registry=registry)
                done = subprocess.run(["sh", "-c", " ".join(words)], input=hello,
                                      capture_output=True, text=True, cwd=start, timeout=60,
                                      env={**environ(), "PYTHONPATH": ""})
                self.assertEqual((done.returncode, done.stderr), (0, ""))
                self.assertEqual(json.loads(done.stdout),
                                 {"node": 1, "host": mock.ANY, "roots": self.roots})

    def test_the_registry_and_its_style_from_the_settings(self):
        registry = self.settings(f"[node]\nregistry = {self.ws.registry}\n")
        status, out, _ = run_main("--list", "--config", registry)
        self.assertEqual(status, 0)
        self.assertIn("local-0", out)
        missing = self.ws.tmp / "no-registry-here"
        self.assertEqual(run_main("--list", "--config", registry, "--registry", str(missing)),
                         (0, f"There's no registry at {shown_path(missing)}/ yet.\n", ""))
        off = self.settings("[node]\nregistry = none\n")
        self.assertEqual(run_main("--list", "--config", off), (
            1, "", f"node: the registry is off (registry = none in {shown_path(off)}), and "
                   "the node server shows its root contexts: give one with --registry.\n"))
        self.assertEqual(run_main("--list", "--config", off, "--registry", "none"), (
            1, "", "node: the registry is off (--registry none), and the node server shows "
                   "its root contexts: give one with --registry.\n"))
        plaid = self.settings(f"[node]\nregistry = {self.ws.registry}\nstyle = plaid\n")
        status, out, err = run_main("--list", "--config", plaid)
        self.assertEqual((status, out), (2, ""))
        self.assertRegex(err, rf"\Anode: {re.escape(shown_path(plaid))}: \[node\] style: "
                              r"'plaid' isn't a style: give one of .*log-dark.*\.\n\Z")
        self.assertEqual(run_main("--list", "--config", plaid, "--style", "chat")[0], 0)

    def test_settings_that_arent_stop_it(self):
        bad = self.settings("[remote a]\nport = x\n")
        missing = self.ws.tmp / "missing.ini"
        for argv, why in (
                (["--config", bad], f"{shown_path(bad)}: [remote a] port: 'x' isn't a port: "
                                    "give 1 to 65535"),
                (["--config", str(missing)], f"{shown_path(missing)}: there's no such file"),
                (["--stdio", "--config", bad], f"{shown_path(bad)}: [remote a] port: 'x' "
                                               "isn't a port: give 1 to 65535")):
            with self.subTest(argv=argv):
                self.assertEqual(run_main("--registry", str(self.ws.registry), *argv),
                                 (2, "", f"node: {why}.\n"))

    def test_the_server_is_made_with_the_settings_and_the_options_over_them(self):
        made = self.settings("[node]\nport = 8799\nstore = none\nstyle = log-light\n\n"
                             "[remote a]\ncommand = a-node\n\n"
                             "[remote b]\ncommand = b-node\ntimeout = 3\n")
        kept = self.ws.tmp / "kept"
        for options, port, style, remotes in (
                ([], 8799, "log-light", [Way("a", ["a-node"]), Way("b", ["b-node"], 3.0)]),
                (["--port", "8800", "--style", "chat", "--store", str(kept), "--remote",
                  "b=other", "--remote", "c=c-node"], 8800, "chat",
                 [Way("a", ["a-node"]), Way("b", ["other"]), Way("c", ["c-node"])])):
            with self.subTest(options=options):
                with mock.patch("pythia.interaction.node.__main__.App") as app, \
                        mock.patch("pythia.interaction.node.__main__.Server",
                                   side_effect=OSError("no port in this test")):
                    status, out, err = run_main("--registry", str(self.ws.registry),
                                                "--config", made, *options)
                self.assertEqual((status, out), (1, ""))
                self.assertIn(f"can't serve on 127.0.0.1:{port}: no port in this test", err)
                given = app.call_args.kwargs
                self.assertEqual((given["style"], given["remotes"]), (style, remotes))
                store = given["store"]
                self.assertEqual(None if store is None else store.path,
                                 kept if options else None)
                self.assertTrue(store is None or isinstance(store, Store))

    def test_each_remote_waits_as_long_as_its_settings_say(self):
        app = server.App(Roots(self.ws.registry), remotes=[
            Way("a", ["a-node"], 3.0), Way("b", ["b-node"]), ("c", ["c-node"])])
        self.addCleanup(app.close)
        self.assertEqual({name: (one.argv, one.timeout) for name, one in app.remotes.items()},
                         {"a": (["a-node"], 3.0), "b": (["b-node"], 30.0),
                          "c": (["c-node"], 30.0)})

    def test_served_with_the_remote_that_the_settings_name(self):
        made = self.settings(f"[node]\nport = 0\nstore = none\n"
                             f"registry = {self.ws.registry}\n\n"
                             f"[remote sim]\ncommand = {shlex.join(node(self.ws.registry))}\n")
        process = subprocess.Popen([sys.executable, "-c", RUN_NODE, "--config", made],
                                   cwd=self.ws.tmp, env=environ(), stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True)
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: [lines.put(line) for line in process.stdout],
                         daemon=True).start()
        try:
            printed = [lines.get(timeout=60) for _ in range(5)]
            self.assertEqual(printed[0], f"node: settings: {shown_path(made)}\n")
            url = re.fullmatch(r"node: (http://127\.0\.0\.1:[0-9]+/[A-Za-z0-9_-]{43}/)\n",
                               printed[2])
            self.assertIsNotNone(url, printed[2])
            self.assertEqual(printed[4], f"node: remote node sim: {url[1]}n/sim/, and in "
                                         f"glui's view: {url[1]}n/sim/lem/\n")
            with urllib.request.urlopen(url[1] + "n/sim/lem/", timeout=20) as response:
                self.assertEqual(response.headers["Content-Security-Policy"], view.PAGE_CSP)
            process.send_signal(signal.SIGINT)
            self.assertEqual(process.wait(timeout=60), 0)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()

    def test_a_node_reached_by_its_standard_streams_shows_no_remotes(self):
        made = self.settings(f"[node]\nregistry = {self.ws.registry}\n\n"
                             "[remote far]\nhost = nowhere.invalid\n")
        hello = json.dumps({"node": 1, "base": "/t0ken/n/x"}) + "\n"
        done = subprocess.run([sys.executable, "-m", "pythia.interaction.node", "--stdio",
                               "--config", made], input=hello + json.dumps(
                                   {"id": 1, "method": "GET", "target": "/t0ken/n/x/lem/facts/",
                                    "host": "127.0.0.1:8765"}) + "\n",
                              capture_output=True, text=True, cwd=self.ws.tmp, env=environ(),
                              timeout=60)
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        first, answer = map(json.loads, done.stdout.splitlines())
        self.assertEqual(first, {"node": 1, "host": mock.ANY, "roots": self.roots})
        self.assertEqual((answer["status"], answer["type"]), (200, "application/json"))
        self.assertIn('"Context"', answer["body"])
        self.assertNotIn('"Node"', answer["body"])


if __name__ == "__main__":
    unittest.main()
