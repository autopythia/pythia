from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from pythia.interaction import _auto_board as board_module
from pythia.interaction._auto_board import Board, BoardError, BoardService, MAX_CONTENT


class BoardTests(unittest.TestCase):
    def test_invalid_programmatic_board_auth_policy_creates_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for invalid in (None, 0, 1, "False"):
                path = Path(tmp) / str(type(invalid).__name__ + str(invalid))
                path.mkdir()
                with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                    BoardService(path, enable_board_auth=invalid)
                self.assertEqual(list(path.iterdir()), [])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.service = BoardService(self.path)
        self.addCleanup(self.service.close)
        self.service.board.accepting = True
        self.user = self.service.client("user")
        self.main = self.service.client("1")
        self.worker = self.service.client("2")

    def raw(self, method, path, *, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.service.server.server_port, timeout=2)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read(), response.getheader("Content-Type")
        finally:
            connection.close()

    def test_readme_discovery_auth_origin_and_no_file_server(self):
        status, body, content_type = self.raw("GET", "/README.md")
        self.assertEqual(status, 200)
        self.assertIn("text/markdown", content_type)
        self.assertIn(self.service.base_url.encode(), body)
        self.assertIn(b"POST /threads", body)
        for secret in self.service.server.tokens:
            self.assertNotIn(secret.encode(), body)
        self.assertEqual(self.raw("GET", "/threads/missing")[0], 401)
        for headers in ({"Host": "evil.test"}, {"Origin": "https://evil.test"}):
            self.assertEqual(self.raw("GET", "/README.md", headers=headers)[0], 403)
        with self.assertRaises(BoardError) as raised:
            self.main.request("GET", "/config.json")
        self.assertEqual(raised.exception.status, 404)

    def test_empty_and_multithread_html_is_static_ordered_and_complete(self):
        empty = (self.path / "index.html").read_text()
        self.assertTrue(empty.startswith("<!doctype html>\n<html"))
        self.assertIn("<meta charset=\"utf-8\">", empty)
        self.assertIn("Rendered through sequence: 0", empty)
        self.assertTrue(empty.endswith("</body></html>\n"))

        first = self.user.create_thread("first content", request_id="first-user")
        plan = self.main.post(first["thread_id"], "plan", "plan content",
                              first["record_id"], request_id="plan")
        self.worker.post(first["thread_id"], "started", "started content",
                         plan["record_id"], request_id="started")
        second = self.user.create_thread("second content", request_id="second-user")
        self.main.post(first["thread_id"], "answer", "answer content",
                       first["record_id"], success=True, request_id="answer")
        self.worker.post(first["thread_id"], "result", "result content",
                         plan["record_id"], success=False, request_id="result")
        text = (self.path / "index.html").read_text()
        records = self.service.board.records()
        self.assertIn("Rendered through sequence: 6", text)
        self.assertEqual(text.count('<article id="record-'), len(records))
        self.assertLess(text.index(first["thread_id"]), text.index(second["thread_id"]))
        for record in records:
            self.assertEqual(text.count(f'<article id="record-{record.sequence}">'), 1)
            self.assertIn(f"Record {record.record_id}: {record.author} / {record.kind}", text)
            self.assertIn(record.request_id, text)
        self.assertIn("Success: true</p>", text)
        self.assertIn("Success: false</p>", text)
        self.assertNotIn("<dl>", text)
        self.assertNotIn("display:grid", text)

    def test_authorized_live_html_routes_headers_and_readme(self):
        token = next(key for key, actor in self.service.server.tokens.items() if actor == "1")
        headers = {"Authorization": "Bearer " + token}

        def get(path):
            connection = HTTPConnection("127.0.0.1", self.service.server.server_port, timeout=2)
            try:
                connection.request("GET", path, headers=headers)
                response = connection.getresponse()
                body = response.read()
                return response, body
            finally:
                connection.close()

        root_response, root = get("/")
        index_response, index = get("/index.html")
        self.assertEqual(root_response.status, 200)
        self.assertEqual(index_response.status, 200)
        self.assertEqual(root, index)
        self.assertEqual(root.decode(), self.service.board._html_snapshot())
        for response, body in ((root_response, root), (index_response, index)):
            self.assertEqual(response.getheader("Content-Type"), "text/html; charset=utf-8")
            self.assertEqual(int(response.getheader("Content-Length")), len(body))
            self.assertEqual(response.getheader("Cache-Control"), "no-store")
            self.assertEqual(response.getheader("X-Content-Type-Options"), "nosniff")
            self.assertEqual(
                response.getheader("Content-Security-Policy"),
                "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
                "form-action 'none'; frame-ancestors 'none'",
            )

        record = self.user.create_thread("Unicode \u2028 </pre><script>x</script>",
                                         request_id="live-html")
        _response, populated = get("/index.html")
        self.assertIn(f'id="record-{record["record_id"]}"'.encode(), populated)
        self.assertIn("Unicode \u2028 &lt;/pre&gt;&lt;script&gt;x&lt;/script&gt;".encode(), populated)

        status, readme, _content_type = self.raw("GET", "/README.md")
        self.assertEqual(status, 200)
        self.assertIn(b"GET / or GET /index.html", readme)
        self.assertIn((self.service.base_url + "/index.html").encode(), readme)
        self.assertIn(b"Authorization: Bearer $BOARD_CAPABILITY", readme)
        self.assertIn(b"render the in-memory", readme)

    def test_html_routes_require_auth_and_reject_queries_methods_and_files(self):
        token = next(key for key, actor in self.service.server.tokens.items() if actor == "user")
        auth = {"Authorization": "Bearer " + token}
        for path, headers, expected in (("/", {}, 401), ("/index.html", {}, 401),
                                        ("/", {"Authorization": "Bearer bad"}, 401),
                                        ("/?token=" + token, {}, 401),
                                        ("/?x=1", auth, 400),
                                        ("/index.html?x=1", auth, 400),
                                        ("/index.html/", auth, 404),
                                        ("/config.json", auth, 404)):
            with self.subTest(path=path, headers=headers):
                status, body, _content_type = self.raw("GET", path, headers=headers)
                self.assertEqual(status, expected)
                self.assertNotIn(b"Shared message board</h1>", body)
        for headers in ({"Authorization": "Bearer " + token, "Host": "evil.test"},
                        {"Authorization": "Bearer " + token, "Origin": "https://evil.test"}):
            self.assertEqual(self.raw("GET", "/", headers=headers)[0], 403)

        post_headers = {**auth, "Content-Type": "application/json", "Content-Length": "2"}
        self.assertEqual(self.raw("POST", "/", body=b"{}", headers=post_headers)[0], 404)
        self.assertEqual(self.service.board.records(), ())

        connection = HTTPConnection("127.0.0.1", self.service.server.server_port, timeout=2)
        try:
            connection.putrequest("GET", "/", skip_host=True)
            connection.putheader("Host", f"127.0.0.1:{self.service.server.server_port}")
            connection.putheader("Authorization", "Bearer " + token)
            connection.putheader("Authorization", "Bearer " + token)
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 401)
            self.assertNotIn(b"Shared message board</h1>", response.read())
        finally:
            connection.close()

    def test_disabled_auth_anonymous_user_and_valid_role_tokens(self):
        self.service.close()
        with tempfile.TemporaryDirectory() as tmp:
            service = BoardService(Path(tmp), enable_board_auth=False)
            try:
                service.board.accepting = True
                def raw(method, path, body=None, headers=None):
                    connection = HTTPConnection("127.0.0.1", service.server.server_port,
                                                timeout=2)
                    try:
                        connection.request(method, path, body=body, headers=headers or {})
                        response = connection.getresponse()
                        return response.status, response.read(), response.getheader("Content-Type")
                    finally:
                        connection.close()

                for path in ("/", "/index.html"):
                    status, body, content_type = raw("GET", path)
                    self.assertEqual(status, 200)
                    self.assertEqual(content_type, "text/html; charset=utf-8")
                    self.assertIn(b"Shared message board", body)
                payload = json.dumps({"request_id": "anonymous", "content": "debug task"}).encode()
                headers = {"Content-Type": "application/json",
                           "Content-Length": str(len(payload))}
                status, body, _ = raw("POST", "/threads", payload, headers)
                self.assertEqual(status, 200)
                created = json.loads(body)
                authenticated_retry = service.client("user").create_thread(
                    "debug task", request_id="anonymous"
                )
                self.assertEqual(authenticated_retry, created)
                status, body, _ = raw("GET", f"/threads/{created['thread_id']}")
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["records"][0]["author"], "user")

                main, worker = service.client("1"), service.client("2")
                plan = main.post(created["thread_id"], "plan", "delegated",
                                 created["record_id"], request_id="main-plan")
                worker.post(created["thread_id"], "started", "working",
                            plan["record_id"], request_id="worker-start")
                main.post(created["thread_id"], "answer", "main done",
                          created["record_id"], success=True, request_id="main-answer")
                worker.post(created["thread_id"], "result", "worker done",
                            plan["record_id"], success=True, request_id="worker-result")
                self.assertEqual([record.author for record in service.board.records()],
                                 ["user", "1", "2", "1", "2"])

                token = next(key for key, actor in service.server.tokens.items()
                             if actor == "user")
                for authorization in ("Bearer bad", "Basic bad", "Bearer"):
                    self.assertEqual(raw("GET", "/", headers={
                        "Authorization": authorization})[0], 401)
                connection = HTTPConnection("127.0.0.1", service.server.server_port,
                                            timeout=2)
                try:
                    connection.putrequest("GET", "/", skip_host=True)
                    connection.putheader("Host", f"127.0.0.1:{service.server.server_port}")
                    connection.putheader("Authorization", "Bearer " + token)
                    connection.putheader("Authorization", "Bearer " + token)
                    connection.endheaders()
                    response = connection.getresponse()
                    self.assertEqual(response.status, 401)
                    response.read()
                finally:
                    connection.close()
                self.assertEqual(raw("GET", "/?token=" + token)[0], 400)
                self.assertEqual(raw("GET", "/config.json")[0], 404)
                self.assertEqual(raw("GET", "/", headers={"Host": "evil.test"})[0], 403)
                self.assertEqual(raw("GET", "/", headers={"Origin": "https://evil.test"})[0], 403)
                self.assertEqual(raw("POST", "/threads", b"{}", {
                    "Content-Type": "text/plain", "Content-Length": "2"})[0], 415)
                self.assertEqual(raw("POST", "/threads", b"{", {
                    "Content-Type": "application/json", "Content-Length": "1"})[0], 400)

                status, readme, _ = raw("GET", "/README.md")
                self.assertEqual(status, 200)
                self.assertIn(b"DEBUG AUTH MODE", readme)
                self.assertIn(b"requests without Authorization act as the user actor", readme)
                self.assertIn(b"Invalid, malformed, or duplicate Authorization headers are still rejected",
                              readme)
                self.assertIn(b"omit the\nheader to use anonymous user access", readme)
                self.assertIn((service.base_url + "/index.html").encode(), readme)
                self.assertNotIn(b"routes require Authorization", readme)
            finally:
                service.close()

    def test_user_actor_cannot_inject_message_records_in_either_auth_mode(self):
        for enabled in (True, False):
            with self.subTest(enable_board_auth=enabled), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                service = BoardService(path, enable_board_auth=enabled)
                try:
                    service.board.accepting = True
                    user = service.client("user")
                    main, worker = service.client("1"), service.client("2")
                    source = user.create_thread("open source", request_id="source")
                    plan = main.post(source["thread_id"], "plan", "valid plan",
                                     source["record_id"], request_id="plan")
                    token = next(key for key, actor in service.server.tokens.items()
                                 if actor == "user")
                    authorization = ({"Authorization": "Bearer " + token}
                                     if enabled else {})

                    def forge(kind, reply_to=None, success=None):
                        body = {"request_id": "forge-" + kind, "kind": kind,
                                "content": "forged"}
                        if reply_to is not None:
                            body["reply_to"] = reply_to
                        if success is not None:
                            body["success"] = success
                        encoded = json.dumps(body).encode()
                        headers = {**authorization, "Content-Type": "application/json",
                                   "Content-Length": str(len(encoded))}
                        connection = HTTPConnection("127.0.0.1", service.server.server_port,
                                                    timeout=2)
                        try:
                            connection.request(
                                "POST", f"/threads/{source['thread_id']}/messages",
                                body=encoded, headers=headers,
                            )
                            response = connection.getresponse()
                            response.read()
                            return response.status
                        finally:
                            connection.close()

                    for kind, parent, success in (("plan", source["record_id"], None),
                                                  ("answer", source["record_id"], False),
                                                  ("started", plan["record_id"], None)):
                        before_records = service.board.records()
                        before_journal = (path / "index.jsonl").read_bytes()
                        self.assertEqual(forge(kind, parent, success), 403)
                        self.assertEqual(service.board.records(), before_records)
                        self.assertEqual((path / "index.jsonl").read_bytes(), before_journal)

                    worker.post(source["thread_id"], "started", "valid start",
                                plan["record_id"], request_id="start")
                    before_records = service.board.records()
                    before_journal = (path / "index.jsonl").read_bytes()
                    self.assertEqual(forge("result", plan["record_id"], True), 403)
                    # This payload deliberately omits success and reply_to. Before
                    # the route/kind guard it was accepted as a second user source.
                    self.assertEqual(forge("user"), 403)
                    self.assertEqual(service.board.records(), before_records)
                    self.assertEqual((path / "index.jsonl").read_bytes(), before_journal)
                finally:
                    service.close()

    def test_html_pre_formatting_preserves_all_leading_content_newlines(self):
        contents = ("hello", "\nhello", "\n\nhello")
        for index, content in enumerate(contents):
            self.user.create_thread(content, request_id=f"leading-{index}")

        class PreText(HTMLParser):
            def __init__(self):
                super().__init__(convert_charrefs=True)
                self.current = None
                self.values = []

            def handle_starttag(self, tag, attrs):
                if tag == "pre":
                    self.current = []

            def handle_data(self, data):
                if self.current is not None:
                    self.current.append(data)

            def handle_endtag(self, tag):
                if tag == "pre":
                    self.values.append("".join(self.current))
                    self.current = None

        parsed = PreText()
        parsed.feed((self.path / "index.html").read_text())
        self.assertEqual(len(parsed.values), len(contents))
        # HTMLParser retains the renderer-owned first LF. HTML5 browsers remove
        # exactly that LF immediately after <pre>, leaving the payload intact.
        self.assertEqual([value[1:] for value in parsed.values], list(contents))
        self.assertTrue(all(value.startswith("\n") for value in parsed.values))

    def test_html_escapes_adversarial_content_as_readable_text(self):
        payload = ('</pre><script>alert("x")</script><img onerror="bad" src="https://evil/">'
                   '</style><a href="https://evil/">quoted & text</a>\n```\n'
                   'unicode \u2028 \u0085 \u2029 ' + "x" * 300)
        self.user.create_thread(payload, request_id="unsafe")
        text = (self.path / "index.html").read_text()

        class Parsed(HTMLParser):
            def __init__(self):
                super().__init__(convert_charrefs=True)
                self.tags, self.text = [], []

            def handle_starttag(self, tag, attrs):
                self.tags.append((tag, dict(attrs)))

            def handle_data(self, data):
                self.text.append(data)

        parsed = Parsed()
        parsed.feed(text)
        self.assertNotIn("script", [tag for tag, _attrs in parsed.tags])
        self.assertNotIn("img", [tag for tag, _attrs in parsed.tags])
        self.assertNotIn("a", [tag for tag, _attrs in parsed.tags])
        self.assertTrue(all(not ({"src", "href"} & set(attrs))
                            and not any(name.startswith("on") for name in attrs)
                            for _tag, attrs in parsed.tags))
        self.assertIn(payload, "".join(parsed.text))
        self.assertNotIn("<script>alert", text)
        self.assertNotIn("http-equiv=\"refresh\"", text.casefold())
        self.assertNotIn("<script", text.casefold())

    def test_thread_is_one_commit_and_retry_does_not_duplicate(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.user.create_thread("hello", request_id="same"), range(4)))
        self.assertTrue(all(r == results[0] for r in results))
        records = self.service.board.records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].author, "user")
        self.assertEqual(records[0].kind, "user")
        self.assertEqual(records[0].content, "hello")
        self.assertEqual(len((self.path / "index.jsonl").read_text().splitlines()), 1)
        with self.assertRaises(BoardError) as raised:
            self.user.create_thread("different", request_id="same")
        self.assertEqual(raised.exception.status, 409)

    def test_restore_preserves_ids_idempotency_and_resets_pending_quota(self):
        originals = [self.user.create_thread(f"historical pending {index}", request_id=f"old-{index}")
                     for index in range(16)]
        original = originals[0]
        self.service.close()
        journal = (self.path / "index.jsonl").read_bytes()
        (self.path / "index.html").unlink()
        restored = BoardService(self.path, restored=Board.restore(self.path))
        self.service = restored
        self.addCleanup(restored.close)
        restored.board.accepting = True
        self.assertTrue((self.path / "index.html").is_file())
        self.assertEqual((self.path / "index.jsonl").read_bytes(), journal)
        token = next(key for key, actor in restored.server.tokens.items() if actor == "1")
        connection = HTTPConnection("127.0.0.1", restored.server.server_port, timeout=2)
        try:
            connection.request("GET", "/index.html",
                               headers={"Authorization": "Bearer " + token})
            response = connection.getresponse()
            rendered = response.read()
            self.assertEqual(response.status, 200)
            self.assertIn(b"Rendered through sequence: 16", rendered)
        finally:
            connection.close()
        user = restored.client("user")
        self.assertEqual(user.create_thread("historical pending 0", request_id="old-0"), original)
        with self.assertRaises(BoardError) as raised:
            user.create_thread("conflict", request_id="old-0")
        self.assertEqual(raised.exception.status, 409)
        new = user.create_thread("new work", request_id="new")
        self.assertEqual(new["record_id"], "17")
        self.assertEqual(len(restored.board.records()), 17)
        self.assertEqual(restored.board.records()[-1].content, "new work")

    def test_restore_rejects_torn_or_malformed_history_without_repair(self):
        self.user.create_thread("valid", request_id="valid")
        self.service.close()
        journal = self.path / "index.jsonl"
        for suffix in ('{"sequence":2}', '{"sequence":2}\n'):
            with self.subTest(suffix=suffix):
                original = journal.read_bytes()
                journal.write_text(original.decode() + suffix)
                broken = journal.read_bytes()
                with self.assertRaises(ValueError):
                    Board.restore(self.path)
                self.assertEqual(journal.read_bytes(), broken)
                journal.write_bytes(original)

    def test_restore_round_trips_unicode_and_escaped_newlines_without_rewrite(self):
        contents = ("valid Unicode: a\u2028b\u0085c\u2029d", "ordinary\nescaped newline")
        for index, content in enumerate(contents):
            self.user.create_thread(content, request_id=f"unicode-{index}")
        self.service.close()
        journal = self.path / "index.jsonl"
        original = journal.read_bytes()
        restored = Board.restore(self.path)
        self.assertEqual([record.content for record, _size in restored], list(contents))
        self.assertEqual(journal.read_bytes(), original)
        self.assertEqual(sum(size for _record, size in restored), len(original))

    def test_restore_reads_only_through_configured_byte_limit(self):
        self.user.create_thread("larger than tiny limit", request_id="limited")
        self.service.close()
        journal = self.path / "index.jsonl"
        original = journal.read_bytes()

        class BoundedReader:
            def __init__(self, file):
                self.file = file

            def __enter__(self):
                self.file.__enter__()
                return self

            def __exit__(self, *args):
                return self.file.__exit__(*args)

            def read(self, size=-1):
                self.assert_size = size
                return self.file.read(size)

        opened = []
        real_open = Path.open
        def open_path(path, *args, **kwargs):
            reader = BoundedReader(real_open(path, *args, **kwargs))
            opened.append(reader)
            return reader
        with mock.patch.object(Path, "open", open_path), self.assertRaises(ValueError):
            Board.restore(self.path, max_bytes=16)
        self.assertEqual(opened[0].assert_size, 17)
        self.assertEqual(journal.read_bytes(), original)

    def test_restore_rejects_terminal_reply_and_route_suffix_thread_id(self):
        source = self.user.create_thread("task", request_id="source")
        self.main.post(source["thread_id"], "answer", "done", source["record_id"],
                       success=True, request_id="answer")
        self.service.close()
        journal = self.path / "index.jsonl"
        original = journal.read_bytes()
        plan = {
            "sequence": 3, "record_id": "3", "thread_id": source["thread_id"],
            "author": "1", "kind": "plan", "content": "too late",
            "request_id": "late", "reply_to": source["record_id"], "success": None,
        }
        journal.write_bytes(original + (json.dumps(plan) + "\n").encode())
        with self.assertRaises(ValueError):
            Board.restore(self.path)
        invalid_graph = journal.read_bytes()
        self.assertEqual(journal.read_bytes(), invalid_graph)

        first = json.loads(original.split(b"\n", 1)[0])
        first["thread_id"] += "/messages"
        journal.write_text(json.dumps(first) + "\n")
        invalid_id = journal.read_bytes()
        with self.assertRaises(ValueError):
            Board.restore(self.path)
        self.assertEqual(journal.read_bytes(), invalid_id)

    def test_plan_lifecycle_correlations_pagination_and_permissions(self):
        root = self.user.create_thread("task")
        other = self.user.create_thread("another")
        thread, source = root["thread_id"], root["record_id"]
        with self.assertRaises(BoardError):
            self.worker.create_thread("cannot act as user")
        with self.assertRaises(BoardError):
            self.user.post(thread, "plan", "spoof main", source)
        with self.assertRaises(BoardError):
            self.main.post(thread, "plan", "cross-thread", other["record_id"])
        plan = self.main.post(thread, "plan", "execute", source)
        with self.assertRaises(BoardError):
            self.worker.post(thread, "result", "not started", plan["record_id"], success=True)
        self.worker.post(thread, "started", "working", plan["record_id"])
        self.main.post(thread, "answer", "delegated", source, success=True)
        with self.assertRaises(BoardError) as raised:
            self.main.post(thread, "plan", "too late", source)
        self.assertEqual(raised.exception.status, 409)
        self.worker.post(thread, "result", "complete", plan["record_id"], success=True)
        seen, after = [], 0
        while True:
            page = self.main.read(thread, after=after, limit=2)
            seen.extend(page["records"])
            after = page["next_after"]
            if not page["has_more"]:
                break
        self.assertEqual([r["kind"] for r in seen], ["user", "plan", "started", "answer", "result"])
        self.assertTrue(all(r["thread_id"] == thread for r in seen))
        self.assertEqual(self.main.read(thread, limit=2)["records"], seen[:2])

    def test_bad_bodies_limits_and_no_reflected_credentials(self):
        token = next(k for k, v in self.service.server.tokens.items() if v == "user")
        headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
        cases = (
            b'{"request_id":"x","request_id":"y","content":"a"}',
            b'{"request_id":"x","content":NaN}',
            b'[]', b'{"request_id":"x","content":""}',
            json.dumps({"request_id": "x", "content": "hi", "author": token}).encode(),
        )
        for body in cases:
            with self.subTest(body=body):
                status, response, _ = self.raw("POST", "/threads", body=body, headers=headers)
                self.assertEqual(status, 400)
                self.assertNotIn(token.encode(), response)
        with self.assertRaises(BoardError) as raised:
            self.user.create_thread("x" * (MAX_CONTENT + 1))
        self.assertEqual(raised.exception.status, 413)
        self.assertEqual(self.service.board.records(), ())
        self.service.board._max_pending = 1
        self.user.create_thread("first")
        with self.assertRaises(BoardError) as raised:
            self.user.create_thread("full")
        self.assertEqual(raised.exception.status, 429)

    def test_save_failure_never_publishes_or_wakes_a_job(self):
        entered, finished = threading.Event(), threading.Event()
        output = []
        def wait():
            entered.set()
            output.append(self.service.board.wait_input("user", 0, threading.Event()))
            finished.set()
        waiter = threading.Thread(target=wait)
        waiter.start()
        self.addCleanup(waiter.join, 2)
        self.assertTrue(entered.wait(2))
        with mock.patch("pythia.interaction._auto_board.os.fsync", side_effect=OSError("disk")):
            with self.assertRaises(BoardError) as raised:
                self.user.create_thread("not committed")
        self.assertEqual(raised.exception.status, 503)
        self.assertTrue(finished.wait(2))
        self.assertEqual(output, [None])
        self.assertEqual(self.service.board.records(), ())
        self.assertTrue(self.service.board.failed)
        with self.assertRaises(BoardError):
            self.user.create_thread("not retried")

    def test_markdown_failure_does_not_undo_commit_and_payload_is_fenced(self):
        with mock.patch("pythia.interaction._auto_board.atomic_text", side_effect=OSError("view")):
            record = self.user.create_thread("```\n<script>example</script>\n```", request_id="view")
        self.assertTrue(self.service.board.view_stale)
        self.assertEqual(self.user.create_thread("```\n<script>example</script>\n```", request_id="view"), record)
        self.assertEqual(len(self.service.board.records()), 1)
        self.main.post(record["thread_id"], "answer", "done", record["record_id"], success=True)
        text = (self.path / "index.md").read_text()
        self.assertIn("Rendered through sequence: 2", text)
        self.assertIn("````json", text)
        self.assertFalse(self.service.board.view_stale)
        serialized = [json.loads(line) for line in (self.path / "index.jsonl").read_text().splitlines()]
        self.assertEqual(len(serialized), 2)

    def test_each_derived_view_failure_is_independent_and_recovers(self):
        for failed_name, current_name in (("index.html", "index.md"),
                                          ("index.md", "index.html")):
            with self.subTest(failed_name=failed_name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                service = BoardService(path)
                try:
                    service.board.accepting = True
                    user, main = service.client("user"), service.client("1")
                    original_atomic = board_module.atomic_text
                    def selective_failure(target, text):
                        if target.name == failed_name:
                            raise OSError("selected view failure")
                        return original_atomic(target, text)
                    with mock.patch.object(board_module, "atomic_text", selective_failure):
                        record = user.create_thread("committed once", request_id="same")
                    self.assertTrue(service.board.view_stale)
                    self.assertIn("Rendered through sequence: 1",
                                  (path / current_name).read_text())
                    self.assertIn("Rendered through sequence: 0",
                                  (path / failed_name).read_text())
                    journal = (path / "index.jsonl").read_bytes()
                    self.assertEqual(user.create_thread("committed once", request_id="same"), record)
                    self.assertEqual((path / "index.jsonl").read_bytes(), journal)
                    self.assertTrue(service.board.view_stale)
                    main.post(record["thread_id"], "answer", "recovered", record["record_id"],
                              success=True)
                    self.assertFalse(service.board.view_stale)
                    for name in ("index.md", "index.html"):
                        self.assertIn("Rendered through sequence: 2", (path / name).read_text())
                    self.assertEqual(len((path / "index.jsonl").read_text().splitlines()), 2)
                finally:
                    service.close()

    def test_live_html_ignores_sidecar_and_writes_response_outside_board_lock(self):
        record = self.user.create_thread("authoritative live content", request_id="live")
        journal = (self.path / "index.jsonl").read_bytes()
        markdown = (self.path / "index.md").read_bytes()
        sidecar = self.path / "index.html"
        sidecar.write_text("tampered sidecar")
        sidecar.unlink()
        self.service.board.view_stale = True
        before_records = self.service.board.records()
        token = next(key for key, actor in self.service.server.tokens.items() if actor == "1")
        released = []
        original_reply = board_module._Handler._reply

        def checked_reply(handler, *args, **kwargs):
            acquired = threading.Event()
            def contend():
                with handler.server.board.changed:
                    acquired.set()
            contender = threading.Thread(target=contend)
            contender.start()
            released.append(acquired.wait(1))
            contender.join(1)
            return original_reply(handler, *args, **kwargs)

        with mock.patch.object(board_module._Handler, "_reply", checked_reply):
            status, body, content_type = self.raw(
                "GET", "/", headers={"Authorization": "Bearer " + token}
            )
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "text/html; charset=utf-8")
        self.assertIn(f'id="record-{record["record_id"]}"'.encode(), body)
        self.assertIn(b"authoritative live content", body)
        self.assertEqual(released, [True])
        self.assertFalse(sidecar.exists())
        self.assertTrue(self.service.board.view_stale)
        self.assertEqual(self.service.board.records(), before_records)
        self.assertEqual((self.path / "index.jsonl").read_bytes(), journal)
        self.assertEqual((self.path / "index.md").read_bytes(), markdown)

    def test_initial_html_failure_closes_journal_and_can_be_restored(self):
        self.service.close()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            original_atomic = board_module.atomic_text
            def fail_html(target, text):
                if target.name == "index.html":
                    raise OSError("html startup failure")
                return original_atomic(target, text)
            with (mock.patch.object(board_module, "atomic_text", fail_html),
                  self.assertRaises(OSError)):
                BoardService(path)
            journal = (path / "index.jsonl").read_bytes()
            restored = BoardService(path, restored=Board.restore(path))
            try:
                self.assertTrue((path / "index.html").is_file())
                self.assertEqual((path / "index.jsonl").read_bytes(), journal)
            finally:
                restored.close()

    def test_notifications_retain_a_commit_before_consumer_waits(self):
        record = self.user.create_thread("fast")
        restored = self.service.board.wait_input("user", 0, threading.Event())
        self.assertEqual(restored.record_id, record["record_id"])
        stop = threading.Event()
        stop.set()
        self.assertIsNone(self.service.board.wait_input("user", 0, stop))


if __name__ == "__main__":
    unittest.main()
