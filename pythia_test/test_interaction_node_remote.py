"""The node server's remote nodes: a node on its standard streams (``--stdio``,
``pythia/interaction/node/stdio.py``), whose pages start with a base path in place
of a token, which another node server reaches by ssh."""

from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pythia.interaction.node import contexts, server, stdio
from pythia.interaction.node.store import Store
from pythia_test.test_interaction_node import HREF, LOCAL, environ, make_workspace, run_main

BASE = "/t0ken/n/sim"
HOST = "127.0.0.1:8765"
HELLO = {"node": 1, "base": BASE}


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


if __name__ == "__main__":
    unittest.main()
