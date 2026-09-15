from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from pythia.interaction._auto_board import Board, BoardError, BoardService, MAX_CONTENT


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

    def test_restore_preserves_ids_idempotency_and_resets_pending_quota(self):
        originals = [self.user.create_thread(f"historical pending {index}", request_id=f"old-{index}")
                     for index in range(16)]
        original = originals[0]
        self.service.close()
        restored = BoardService(self.path, restored=Board.restore(self.path))
        self.service = restored
        self.addCleanup(restored.close)
        restored.board.accepting = True
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

    def test_notifications_retain_a_commit_before_consumer_waits(self):
        record = self.user.create_thread("fast")
        restored = self.service.board.wait_input("user", 0, threading.Event())
        self.assertEqual(restored.record_id, record["record_id"])
        stop = threading.Event()
        stop.set()
        self.assertIsNone(self.service.board.wait_input("user", 0, stop))


if __name__ == "__main__":
    unittest.main()
