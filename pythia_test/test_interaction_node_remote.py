"""The node server's remote nodes: a node on its standard streams (``--stdio``,
``pythia/interaction/node/stdio.py``), whose pages start with a base path in place
of a token; and the remotes of a node server (``--remote``, ``remote.py``), each
such a node, reached here by a command that runs one on this node, as ssh would
run one on another, whose pages the server answers ``/TOKEN/n/NAME/...`` with."""

from __future__ import annotations

import base64
import gzip
import io
import json
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from pythia.interaction.node import contexts, remote, server, stdio, view
from pythia.interaction.node.store import Store
from pythia_test.test_interaction_node import (CHECKOUT, HREF, LOCAL, OLD, PROGRAM, PROJ,
                                               RUN_NODE, TOKEN, environ, make_workspace,
                                               request, run_main, step_problems)

BASE = "/t0ken/n/sim"
HOST = "127.0.0.1:8765"
HELLO = {"node": 1, "base": BASE}
# A remote that answers each request with the answer that its argument gives for the
# request's path past its base, and its id.
FAKE = """import json, sys
answers = json.loads(sys.argv[1])
base = json.loads(sys.stdin.readline())["base"]
print(json.dumps({"node": 1, "host": "fake", "roots": 0}), flush=True)
for line in sys.stdin:
    asked = json.loads(line)
    print(json.dumps({"id": asked["id"], **answers[asked["target"][len(base):]]}), flush=True)
"""

# A remote that answers its first request, then nothing, as one whose connection stalled.
ONCE = """import json, sys, time
sys.stdin.readline()
print(json.dumps({"node": 1, "host": "once", "roots": 0}), flush=True)
asked = json.loads(sys.stdin.readline())
print(json.dumps({"id": asked["id"], "status": 200, "type": "application/json", "body": "[]"}),
      flush=True)
time.sleep(60)
"""


def node(registry, store="none"):
    """The command that runs a node server on its standard streams, from this checkout,
    for ``registry``."""
    return ["env", f"PYTHONPATH={CHECKOUT}", sys.executable, "-m", "pythia.interaction.node",
            "--stdio", "--registry", str(registry), "--store", str(store)]


def python(code, *argv):
    """The command that runs ``code`` in Python."""
    return [sys.executable, "-c", code, *argv]


def get(rid, path):
    """A request's line, for GET ``path``, below the base."""
    return {"id": rid, "method": "GET", "target": BASE + path, "host": HOST}


def talk(roots, *lines, store=None):
    """What stdio.serve says to ``lines``, each JSON, or bytes as they are: its status,
    and its lines, as JSON."""
    given = b"".join((line if isinstance(line, bytes) else json.dumps(line).encode())
                     + b"\n" for line in lines)
    out = io.BytesIO()
    status = stdio.serve(roots, store=store, stdin=io.BytesIO(given), stdout=out)
    return status, [json.loads(line) for line in out.getvalue().splitlines()]


class RemoteTest(unittest.TestCase):
    """A test with make_workspace's registry, made once for its class."""

    @classmethod
    def setUpClass(cls):
        cls.ws = make_workspace(tempfile.mkdtemp(prefix="node-remote-test-"))
        cls.roots = contexts.Roots(cls.ws.registry)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.ws.tmp)


class BaseTests(RemoteTest):
    def test_the_pages_are_behind_the_base_and_link_through_it(self):
        app = server.App(self.roots, base=BASE)
        self.assertEqual(app.path, BASE + "/")
        for path in ("/", f"/c/{LOCAL}", "/lem/", f"/lem/c/{LOCAL}"):
            with self.subTest(path=path):
                response = app.respond("GET", BASE + path, [HOST])
                self.assertEqual(response.status, 200, response.body[:200])
                links = [link for link in HREF.findall(response.body) if link.startswith("/")]
                self.assertTrue(links)
                self.assertEqual([link for link in links if not link.startswith(BASE + "/")],
                                 [])
        self.assertEqual(app.respond("GET", BASE, [HOST]).location, BASE + "/")
        for path in ("/t0ken/", "/t0ken/n/", "/t0ken/n/simx/", "/t0ken/n/sim2", "/sim/", "/"):
            with self.subTest(path=path):
                self.assertEqual(app.respond("GET", path, [HOST]).status, 404)

    def test_the_views_facts_link_through_the_base(self):
        app = server.App(self.roots, base=BASE)
        steps = json.loads(app.respond("GET", BASE + "/lem/facts/", [HOST]).body)
        links = [step[2]["to"] for step in steps if "to" in step[2]]
        self.assertIn(f"{BASE}/lem/c/{LOCAL}", links)
        self.assertEqual([link for link in links if not link.startswith(BASE + "/")], [])

    def test_a_token_is_a_base_of_one_segment(self):
        app = server.App(self.roots, "t0ken")
        self.assertEqual((app.base, app.path), ("/t0ken", "/t0ken/"))
        self.assertEqual(app.respond("GET", "/t0ken/", [HOST]).status, 200)
        self.assertEqual(app.respond("GET", "/t0ken", [HOST]).location, "/t0ken/")
        self.assertEqual(app.respond("GET", "/t0ken2/", [HOST]).status, 404)


class StdioTests(RemoteTest):
    def test_it_says_hello_then_answers_each_request_in_order(self):
        status, lines = talk(self.roots, HELLO, get(1, "/"), get(2, "/lem/facts/"),
                             get(3, ""), get(4, "/nope"))
        self.assertEqual(status, 0)
        self.assertEqual(lines[0], {"node": 1, "host": mock.ANY,
                                    "roots": len(self.roots.entries())})
        self.assertEqual([(line["id"], line["status"]) for line in lines[1:]],
                         [(1, 200), (2, 200), (3, 308), (4, 404)])
        self.assertIn(f'href="{BASE}/c/{LOCAL}"', lines[1]["body"])
        self.assertEqual(lines[2]["type"], "application/json")
        self.assertEqual(lines[2]["csp"], "default-src 'none'; frame-ancestors 'none'")
        self.assertEqual(lines[3]["location"], BASE + "/")

    def test_a_first_line_that_isnt_a_nodes_ends_it(self):
        node = "the first line isn't a node's: it says {\"node\": 1, \"base\": BASE}"
        path = "the base isn't a path, such as /TOKEN/n/NAME"
        for first, why in ((b"nope", node), ({"base": BASE}, node), ([1], node),
                           ({"node": 2, "base": BASE}, "this node speaks protocol 1, not 2"),
                           ({"node": 1}, path), ({"node": 1, "base": "t0ken"}, path),
                           ({"node": 1, "base": "/t0ken/../x"}, path),
                           ({"node": 1, "base": "/t0ken/"}, path),
                           ({"node": 1, "base": "/t0ken/n/s m"}, path)):
            with self.subTest(first=first):
                with mock.patch("sys.stderr", io.StringIO()) as err:
                    status, lines = talk(self.roots, first, get(1, "/"))
                self.assertEqual((status, lines), (1, [{"node": 1, "error": why}]))
                self.assertEqual(err.getvalue(), f"node: {why}\n")
        with mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(talk(self.roots), (1, [{"node": 1, "error": node}]))

    def test_lines_that_arent_requests_are_refused_and_it_goes_on(self):
        post = {"id": 5, "method": "POST", "target": BASE + "/lem/store/session",
                "host": HOST, "body": "e30=!"}  # {} but for the "!"
        status, lines = talk(self.roots, HELLO, b"nope", {"id": "1"}, {"id": True},
                             {"id": 3, "method": 3, "target": BASE + "/", "host": HOST},
                             {**get(4, "/"), "origins": "x"}, post,
                             b"x" * (stdio.MAX_LINE + 10), get(6, "/"))
        self.assertEqual(status, 0)
        why = "The line isn't a request: it's JSON, with an id.\n"
        fields = "The request's fields aren't what a request has.\n"
        self.assertEqual([(line["id"], line["status"], line["body"]) for line in lines[1:-1]],
                         [(None, 400, why), (None, 400, why), (None, 400, why),
                          (3, 400, fields), (4, 400, fields),
                          (5, 400, "The request's body isn't in base64.\n"),
                          (None, 400, why)])
        self.assertEqual((lines[-1]["id"], lines[-1]["status"]), (6, 200))

    def test_the_views_posts_keep_its_settings_in_its_nodes_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "node")

            def post(rid, value, **fields):
                body = base64.b64encode(json.dumps(value).encode()).decode()
                return {"id": rid, "method": "POST", "target": BASE + "/lem/store/session",
                        "host": HOST, "origins": [f"http://{HOST}"], "marked": True,
                        "type": "application/json", "body": body, **fields}

            status, lines = talk(self.roots, HELLO, post(1, {"look": "log"}),
                                 post(2, {"look": "native"}, marked=False),
                                 post(3, {"look": "native"}, body=None),
                                 get(4, "/lem/store/session"), store=store)
            self.assertEqual(status, 0)
            self.assertEqual([(line["id"], line["status"]) for line in lines[1:]],
                             [(1, 204), (2, 403), (3, 413), (4, 200)])
            self.assertEqual(json.loads(lines[4]["body"]), {"look": "log"})
            self.assertEqual(store.read("session"), {"look": "log"})

    def test_a_page_that_fails_is_a_500_and_it_goes_on(self):
        respond = server.App.respond

        def failing(app, method, target, *args, **kwargs):
            if target.endswith("/fails"):
                raise RuntimeError("boom")
            return respond(app, method, target, *args, **kwargs)

        with mock.patch.object(server.App, "respond", failing), \
                mock.patch("sys.stderr", io.StringIO()) as err:
            status, lines = talk(self.roots, HELLO, get(1, "/fails"), get(2, "/"))
        self.assertEqual(status, 0)
        self.assertEqual([(line["id"], line["status"]) for line in lines[1:]],
                         [(1, 500), (2, 200)])
        self.assertEqual(lines[1]["body"],
                         "The node server failed on this page: RuntimeError: boom\n")
        self.assertIn("RuntimeError: boom", err.getvalue())


class ProcessTests(RemoteTest):
    def test_python3_m_node_stdio_answers_on_its_streams(self):
        given = "".join(json.dumps(line) + "\n" for line in (HELLO, get(1, f"/c/{LOCAL}")))
        done = subprocess.run([sys.executable, "-m", "pythia.interaction.node", "--stdio",
                               "--registry", str(self.ws.registry), "--store", "none"],
                              input=given, env=environ(), capture_output=True, text=True,
                              timeout=60)
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        lines = [json.loads(line) for line in done.stdout.splitlines()]
        self.assertEqual(lines[0]["roots"], len(self.roots.entries()))
        self.assertEqual((lines[1]["id"], lines[1]["status"]), (1, 200))
        self.assertIn("local-0", lines[1]["body"])

    def test_stdio_and_list_dont_go_together(self):
        status, out, err = run_main("--stdio", "--list")
        self.assertEqual((status, out), (2, ""))
        self.assertIn("argument --list: not allowed with argument --stdio", err)



class ParseTests(unittest.TestCase):
    def test_a_host_is_reached_by_ssh_and_a_name_by_its_command(self):
        self.assertEqual(remote.parse("cloud-0"), (
            "cloud-0", ["ssh", "-o", "BatchMode=yes", "cloud-0", "python3", "-m",
                        "pythia.interaction.node", "--stdio"]))
        self.assertEqual(remote.parse("sim=python3 -m x --registry 'a b'"),
                         ("sim", ["python3", "-m", "x", "--registry", "a b"]))
        self.assertEqual(remote.parse("a.b_c=x=y"), ("a.b_c", ["x=y"]))

    def test_what_isnt_a_remote(self):
        name = ("isn't a node's name: letters, digits, '.', '_' and '-', from a letter or a "
                "digit; for a host that isn't, give NAME=COMMAND")
        for text, why in (("-oProxyCommand=x", f"'-oProxyCommand' {name}"),
                          ("user@host", f"'user@host' {name}"), ("", f"'' {name}"),
                          ("a/b", f"'a/b' {name}"), ("sim=", "sim's command is empty"),
                          ("sim= ", "sim's command is empty"),
                          ("sim='x", "No closing quotation")):
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as caught:
                    remote.parse(text)
                self.assertEqual(str(caught.exception), why)


class RemoteTests(RemoteTest):
    def app(self, *remotes, **options):
        """An app with ``remotes``, each with ``options``, and a timeout of 10 s unless they
        say otherwise."""
        app = server.App(self.roots, TOKEN, remotes=remotes)
        for one in app.remotes.values():
            for name, value in {"timeout": 10.0, **options}.items():
                setattr(one, name, value)
        self.addCleanup(app.close)
        return app

    def get(self, app, path, method="GET"):
        return app.respond(method, f"/{TOKEN}{path}", [HOST])

    def test_a_remotes_pages_come_through_the_local_server(self):
        app = self.app(("sim", node(self.ws.registry)))
        base = f"/{TOKEN}/n/sim/"
        index = self.get(app, "/n/sim/")
        self.assertEqual((index.status, index.content_type, index.csp), (200, "text/html", None))
        links = [link for link in HREF.findall(index.body) if link.startswith("/")]
        self.assertIn(f"{base}c/{LOCAL}", links)
        self.assertEqual([link for link in links if not link.startswith(base)], [])
        page = self.get(app, "/n/sim/lem/")
        self.assertEqual((page.status, page.csp), (200, view.PAGE_CSP))
        self.assertIn(f'src="{base}lem/files/contexts.js"', page.body)
        facts = self.get(app, f"/n/sim/lem/facts/c/{LOCAL}")
        self.assertEqual((facts.status, facts.content_type, facts.csp),
                         (200, "application/json", view.DATA_CSP))
        self.assertEqual(self.get(app, "/n/sim/c/1?line=1").status, 200)
        self.assertEqual(self.get(app, "/n/sim", "HEAD").location, base)
        self.assertEqual(self.get(app, "/n/nope/").status, 404)
        self.assertEqual(self.get(app, "/n/").status, 404)
        self.assertEqual(app.respond("PUT", f"{base}", [HOST]).status, 405)  # as it says
        self.assertEqual(app.remotes["sim"].hello,
                         {"node": 1, "host": mock.ANY, "roots": len(self.roots.entries())})
        process = app.remotes["sim"]._process
        app.close()
        self.assertIsNotNone(process.poll())
        self.assertEqual(self.get(app, "/n/sim/").body,
                         "The remote node sim can't be reached: the server is stopping.\n")

    def test_the_views_files_are_the_local_servers_own(self):
        app = self.app(("down", python("import sys; sys.exit(1)")))
        mine = self.get(server.App(self.roots, TOKEN), "/lem/files/contexts.js")
        for path in ("/n/down/lem/files/contexts.js", "/n/down/lem/files/compiled/tabs.json"):
            with self.subTest(path=path):
                got = self.get(app, path)
                self.assertEqual(got.status, 200)
        self.assertEqual(self.get(app, "/n/down/lem/files/contexts.js").body, mine.body)
        self.assertEqual(self.get(app, "/n/down/lem/files/nope.js").status, 404)
        self.assertEqual(self.get(app, "/n/down/lem/").status, 502)

    def test_a_remote_that_cant_be_reached_says_why_and_is_tried_again(self):
        app = self.app(("gone", ["/nonexistent/node"]),
                       ("fails", python("import sys; sys.stderr.write('no registry here.\\n'); "
                                        "sys.exit(3)")),
                       ("refuses", python("import json, sys; sys.stdin.readline(); "
                                          "print(json.dumps({'node': 1, 'error': 'nope'}))")),
                       ("other", python("print('hello')")),
                       ("old", python("import sys; sys.stdin.readline(); print('{\"node\": 0}')")))
        said = {name: self.get(app, f"/n/{name}/").body for name in app.remotes}
        self.assertEqual(said, {
            "gone": "The remote node gone can't be reached: its command couldn't start: "
                    "[Errno 2] No such file or directory: '/nonexistent/node'.\n",
            "fails": "The remote node fails can't be reached: it ended, with status 3: "
                     "no registry here.\n",
            "refuses": "The remote node refuses can't be reached: it refused: nope.\n",
            "other": "The remote node other can't be reached: it didn't answer as a node "
                     "does.\n",
            "old": "The remote node old can't be reached: it didn't answer as a node "
                   "does.\n"})
        again = self.get(app, "/n/fails/")
        self.assertEqual(again.status, 502)
        self.assertRegex(again.body, r"no registry here; it's tried again in [1-5] s\.\n\Z")
        app.remotes["fails"].retry = 0
        self.assertEqual(self.get(app, "/n/fails/").body, said["fails"])

    def test_a_remote_that_doesnt_answer_in_time_gets_504_and_starts_again(self):
        app = self.app(("slow", python("import sys, time; sys.stdin.readline(); "
                                       "time.sleep(60)")), timeout=0.3)
        started = time.monotonic()
        got = self.get(app, "/n/slow/")
        self.assertEqual((got.status, got.body), (504, (
            "The remote node slow didn't answer in 0.3 s; it's started again for the next "
            "request.\n")))
        self.assertLess(time.monotonic() - started, 10)
        self.assertIsNone(app.remotes["slow"]._process)
        self.assertEqual(app.remotes["slow"].why, "it didn't answer in 0.3 s")

    def test_a_remote_ends_once_a_request_has_waited_its_timeout_and_not_before(self):
        app = self.app(("once", python(ONCE)), timeout=1.0)
        once = app.remotes["once"]
        self.assertEqual(self.get(app, "/n/once/lem/facts/").status, 200)
        time.sleep(1.2)  # longer than its timeout, since a request that it answered
        late, = remote.listings([once], wait=0.3)
        self.assertEqual(late.why, "The remote node once didn't answer in 0.3 s.")
        process = once._process
        self.assertIsNone(process.poll())
        time.sleep(0.8)  # and now, since one that it didn't
        later, = remote.listings([once], wait=0.3)
        self.assertEqual(later.why, "The remote node once didn't answer in 1 s; it's started "
                         "again for the next request.")
        self.assertIsNotNone(process.poll())
        self.assertEqual(once.why, "it didn't answer in 1 s")

    def test_what_a_remote_answers_passes_on_only_as_the_local_servers_own_pages_would(self):
        base = f"/{TOKEN}/n/fake"
        answers = {
            "/script": {"status": 200, "type": "text/javascript", "body": "alert(1)"},
            "/away": {"status": 303, "type": "text/plain", "body": "",
                      "location": "http://elsewhere/"},
            "/sideways": {"status": 303, "type": "text/plain", "body": "",
                          "location": f"/{TOKEN}/n/fakes/"},
            "/near": {"status": 303, "type": "text/plain", "body": "",
                      "location": f"{base}/lem/c/1"},
            "/loose": {"status": 200, "type": "text/html", "body": "<p>hi",
                       "csp": "default-src *", "headers": [["Set-Cookie", "a=b"],
                                                           ["Allow", "GET"]]},
            "/view": {"status": 200, "type": "text/html", "body": "", "csp": view.PAGE_CSP},
            "/broken": {"status": "200", "type": "text/html", "body": ""},
        }
        app = self.app(("fake", python(FAKE, json.dumps(answers))))
        got = {path: self.get(app, f"/n/fake{path}") for path in answers}
        refused = "The remote node fake answered with what this server doesn't pass on.\n"
        self.assertEqual({path: (one.status, one.body) for path, one in got.items()
                          if one.status == 502}, {
            "/script": (502, refused), "/away": (502, refused), "/sideways": (502, refused),
            "/broken": (502, "The remote node fake answered with a line that isn't an "
                             "answer.\n")})
        self.assertEqual(got["/near"].location, f"{base}/lem/c/1")
        self.assertEqual((got["/loose"].body, got["/loose"].csp, got["/loose"].headers),
                         ("<p>hi", None, (("Allow", "GET"),)))
        self.assertEqual(got["/view"].csp, view.PAGE_CSP)

    def test_the_views_posts_go_to_the_remotes_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = self.app(("sim", node(self.ws.registry, Path(tmp) / "node")))

            def post(path, value, **fields):
                return app.respond("POST", f"/{TOKEN}/n/sim{path}", [HOST], **{
                    "origins": [f"http://{HOST}"], "marked": True,
                    "content_type": "application/json", "body": json.dumps(value).encode(),
                    **fields})

            self.assertEqual(post("/lem/store/session", {"look": "log"}).status, 204)
            self.assertEqual(post("/lem/store/session", {"look": "x"}, marked=False).status,
                             403)
            self.assertEqual(post("/lem/store/session", {}, body=None).status, 413)
            self.assertEqual(post(f"/lem/stars/{LOCAL}", {"line": 2, "on": True}).status, 204)
            self.assertEqual(post("/lem/files/contexts.js", {}).status, 405)
            self.assertEqual(json.loads(self.get(app, "/n/sim/lem/store/session").body),
                             {"look": "log"})
            kept = Store(Path(tmp) / "node")
            self.assertEqual(kept.read("session"), {"look": "log"})
            self.assertEqual(list(kept.read("stars").values()), [[2]])
            self.assertEqual(app.respond("POST", f"/{TOKEN}/lem/store/session", [HOST],
                                         origins=[f"http://{HOST}"], marked=True,
                                         content_type="application/json", body=b"{}").status,
                             404)  # the local server keeps no store

    def test_requests_at_once_each_get_their_own_answer(self):
        app = self.app(("sim", node(self.ws.registry)))
        numbers = [LOCAL, OLD, PROJ] * 3
        answers = queue.Queue()

        def ask(number):
            answers.put((number, self.get(app, f"/n/sim/lem/tab/{number}")))

        threads = [threading.Thread(target=ask, args=(number,)) for number in numbers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        got = [answers.get_nowait() for _ in numbers]
        self.assertEqual(sorted((number, json.loads(one.body)["id"]) for number, one in got),
                         sorted((number, number) for number in numbers))

    def test_served_with_a_remote_and_stopped_on_ctrl_c(self):
        process = subprocess.Popen(
            [sys.executable, "-c", RUN_NODE, "--port", "0", "--registry", str(self.ws.registry),
             "--store", "none", "--remote", "sim=" + " ".join(node(self.ws.registry))],
            cwd=self.ws.tmp, env=environ(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True)
        lines = queue.Queue()
        threading.Thread(target=lambda: [lines.put(line) for line in process.stdout],
                         daemon=True).start()
        try:
            printed = [lines.get(timeout=60) for _ in range(4)]
            url = re.fullmatch(r"node: (http://127\.0\.0\.1:([0-9]+)/[A-Za-z0-9_-]{43}/)\n",
                               printed[1])
            self.assertIsNotNone(url, printed[1])
            self.assertEqual(printed[3], f"node: remote node sim: {url[1]}n/sim/, and in glui's "
                             f"view: {url[1]}n/sim/lem/\n")
            with urllib.request.urlopen(url[1] + "n/sim/lem/", timeout=20) as response:
                self.assertEqual(response.headers["Content-Security-Policy"], view.PAGE_CSP)
                self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
            port = int(url[2])
            status, headers, body = request(port, url[1][len(f"http://127.0.0.1:{port}"):]
                                            + f"n/sim/c/{LOCAL}",
                                            headers=[("Accept-Encoding", "gzip")])
            self.assertEqual(status, 200)
            if headers.get("content-encoding") == "gzip":
                body = gzip.decompress(body)
            self.assertIn("local-0", body.decode())
            process.send_signal(signal.SIGINT)
            self.assertEqual(process.wait(timeout=60), 0)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()

    def test_remote_options_that_dont_go(self):
        registry = str(self.ws.registry)
        # Should one go, the server can't start, rather than serve in this test.
        unserved = mock.patch("pythia.interaction.node.__main__.Server",
                              side_effect=OSError("no port in this test"))
        unserved.start()
        self.addCleanup(unserved.stop)
        for argv, why in (
                (["--remote", "a@b"], "argument --remote: 'a@b' isn't a node's name"),
                (["--stdio", "--remote", "x"], "argument --remote: not allowed with argument "
                                               "--stdio"),
                (["--remote", "x", "--remote", "x=y"], "argument --remote: x is given twice")):
            with self.subTest(argv=argv):
                status, out, err = run_main("--registry", registry, *argv)
                self.assertEqual((status, out), (2, ""))
                self.assertIn(why, err)



class IndexTests(RemoteTest):
    def app(self, *remotes):
        app = server.App(self.roots, TOKEN, remotes=remotes)
        self.addCleanup(app.close)
        return app

    def test_the_index_lists_each_remotes_roots_after_this_nodes(self):
        app = self.app(("sim", node(self.ws.registry)), ("gone", python("import sys; sys.exit(1)")))
        steps = json.loads(app.respond("GET", f"/{TOKEN}/lem/facts/", [HOST]).body)
        self.assertEqual(step_problems(json.loads(PROGRAM.read_text()), steps), [])
        local = [step[2] for step in steps if step[1] == "Context" and "/" not in step[2]["id"]]
        mine = [step[2] for step in steps
                if step[1] == "Context" and step[2]["id"].startswith("sim/")]
        nodes = [step[2] for step in steps if step[1] == "Node"]
        self.assertEqual([(node["name"], node["rank"], node["reached"], node["to"])
                          for node in nodes],
                         [("sim", len(local), True, f"/{TOKEN}/n/sim/lem/"),
                          ("gone", len(local) + 1 + len(mine), False, f"/{TOKEN}/n/gone/lem/")])
        host = app.remotes["sim"].hello["host"]
        self.assertEqual(nodes[0]["title"], f"{host}: {len(mine)} root contexts")
        self.assertEqual(nodes[1]["title"], "The remote node gone can't be reached: it ended, "
                         "with status 1: it said nothing.")
        # Its roots, as this node's own are, by ids of their own, ranked after its row.
        self.assertEqual([(root["id"][4:], root["title"]) for root in mine],
                         [(root["id"], root["title"]) for root in local])
        self.assertEqual([root["rank"] for root in mine],
                         list(range(len(local) + 1, len(local) + 1 + len(mine))))
        self.assertEqual([root["to"] for root in mine],
                         [f"/{TOKEN}/n/sim{root['to'][len(f'/{TOKEN}'):]}" for root in local])
        doing = {step[2]["id"] for step in steps if step[1] == "Activity"}
        self.assertTrue({root["id"] for root in mine} <= doing)

    def test_what_a_remote_lists_that_isnt_a_root_of_its_isnt_shown(self):
        base = f"/{TOKEN}/n/fake"

        def root(cid, rank, **fields):
            return ["+", "Context", {"id": cid, "title": "t", "model": "m", "turns": 1,
                                     "lines": 2, "size": "3 B", "changed": "now", "rank": rank,
                                     "to": f"{base}/lem/c/{rank}", **fields}]

        listed = [["+", "Page", {"title": "Root contexts"}], ["+", "Nav", {"n": 0, "text": "x",
                                                                            "to": "/"}],
                  root("b (2)", 2), root("a (1)", 1), root("a (1)", 3),
                  root("turns", 4, turns="1"), root("away", 5, to="/elsewhere/lem/c/5"),
                  root("more", 6, extra=1),
                  root("named", 8, label="given", tags="a b", keep=f"{base}/lem/roots/8"),
                  root("misnamed", 9, label=5), ["+", "Context", "x"], ["-", "Context", {}], 7,
                  ["+", "Activity", {"id": "a (1)", "hue": "Green", "why": "w"}],
                  ["+", "Activity", {"id": "b (2)", "hue": "Pink", "why": "w"}],
                  ["+", "Activity", {"id": "other", "hue": "Red", "why": "w"}]]
        answer = {"status": 200, "type": "application/json", "body": json.dumps(listed)}
        app = self.app(("fake", python(FAKE, json.dumps({"/lem/facts/": answer}))))
        steps = json.loads(app.respond("GET", f"/{TOKEN}/lem/facts/", [HOST]).body)
        shown = [[step[1], step[2].get("id", step[2].get("name")), step[2].get("rank"),
                  step[2].get("to", "")[len(base):]]
                 for step in steps if step[1] == "Node" or (
                     step[1] in ("Context", "Activity") and step[2]["id"].startswith("fake/"))]
        local = len([step for step in steps if step[1] == "Context" and "/" not in step[2]["id"]])
        self.assertEqual(shown, [["Node", "fake", local, "/lem/"],
                                 ["Context", "fake/a (1)", local + 1, "/lem/c/1"],
                                 ["Activity", "fake/a (1)", None, ""],
                                 ["Context", "fake/b (2)", local + 2, "/lem/c/2"],
                                 ["Context", "fake/named", local + 3, "/lem/c/8"]])
        self.assertEqual([step[2].get("label") for step in steps if step[1] == "Context"
                          and step[2]["id"].startswith("fake/")], [None, None, "given"])
        # Its label, as its node gives it, but neither its tags nor a way to keep them.
        self.assertEqual([sorted(set(step[2]) & {"tags", "keep"}) for step in steps
                          if step[1] == "Context" and step[2]["id"] == "fake/named"], [[]])

    def test_a_slow_remote_is_waited_for_briefly_and_left_running(self):
        slow = python("import json, sys, time; sys.stdin.readline(); "
                      "print(json.dumps({'node': 1, 'host': 'h', 'roots': 0}), flush=True); "
                      "time.sleep(60)")
        app = self.app(("slow", slow))
        listing, = remote.listings(list(app.remotes.values()), wait=0.3)
        self.assertEqual((listing.steps, listing.why),
                         ([], "The remote node slow didn't answer in 0.3 s."))
        self.assertIsNone(app.remotes["slow"]._process.poll())
        with mock.patch.object(remote.Remote, "request", side_effect=RuntimeError("boom")):
            listing, = remote.listings(list(app.remotes.values()))
        self.assertEqual(listing.why, "The remote node slow couldn't be asked: RuntimeError: boom")

    def test_the_plain_index_lists_each_remotes_roots(self):
        app = self.app(("sim", node(self.ws.registry)), ("gone", python("import sys; sys.exit(1)")))
        body = app.respond("GET", f"/{TOKEN}/", [HOST]).body
        sim = body[body.index("<h2>Remote node sim</h2>"):body.index("<h2>Remote node gone")]
        self.assertIn(f'<a href="/{TOKEN}/n/sim/">Its pages</a> · '
                      f'<a href="/{TOKEN}/n/sim/lem/">in glui&#x27;s view</a>', sim)
        self.assertIn(f'<a href="/{TOKEN}/n/sim/c/{LOCAL}">local-0 ({LOCAL})</a>', sim)
        self.assertEqual(len(re.findall("<tr><td class=nw><a ", sim)),
                         len(self.roots.contexts()))
        self.assertIn("<h2>Remote node gone</h2>", body)
        self.assertIn("<p class=dim>The remote node gone can&#x27;t be reached: it ended, with "
                      "status 1: it said nothing.</p>", body)
        self.assertEqual(server.App(self.roots, TOKEN).respond("GET", f"/{TOKEN}/", [HOST])
                         .body.count("<h2>"), 0)


if __name__ == "__main__":
    unittest.main()
