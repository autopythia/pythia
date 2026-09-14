from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from pythia.interaction._auto_board import BoardError, BoardService, MAX_CONTENT


class BoardTests(unittest.TestCase):
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

    def test_notifications_retain_a_commit_before_consumer_waits(self):
        record = self.user.create_thread("fast")
        restored = self.service.board.wait_input("user", 0, threading.Event())
        self.assertEqual(restored.record_id, record["record_id"])
        stop = threading.Event()
        stop.set()
        self.assertIsNone(self.service.board.wait_input("user", 0, stop))


if __name__ == "__main__":
    unittest.main()
