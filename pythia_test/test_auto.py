from __future__ import annotations

import asyncio
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import DisplayItem, Environment, Message, ModelSample, Tool, ToolCall
from pythia.interaction import ToolOutcome, ToolSpec, ToolResult, TurnSummary
from pythia.interaction import ModelFailure, ModelTransportError, Reasoning, OpaqueCompaction
from pythia.interaction import load_interaction_save
from pythia.interaction import auto
from pythia.interaction._auto_board import BoardError
from pythia.interaction._auto_config import build_parser, namespace, resolve_config
from pythia.interaction.messages import resolve_messages_max_output_tokens
from pythia.interaction.runtime_config import InteractionConfig


def answer(text="done"):
    return ModelSample((Message("assistant", text),))


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    raise AssertionError("timed out waiting for test condition")


class DisplayTests(unittest.TestCase):
    def setUp(self):
        self.session = SimpleNamespace(names={1: "main", 2: "custom worker", -1: "watcher"})

    def test_context_is_folded_into_every_item_label(self):
        sample = ModelSample((
            Reasoning("", summary=("first thought", "second thought")),
            Message("assistant", "answer\n[reasoning] quoted body stays unchanged"),
        ))
        original = sample.display_items()
        rendered = auto._display_events(self.session, (
            auto._Event(1, original),
            auto._Event(2, answer("worker answer").display_items()),
            auto._Event(-1, (DisplayItem("[debug] condition fired", label="debug"),), "debug"),
        ))
        self.assertEqual([item.text for item in rendered], [
            "[#1 (main) - reasoning] first thought",
            "[#1 (main) - reasoning] second thought",
            "[#1 (main) - assistant] answer\n[reasoning] quoted body stays unchanged",
            "[#1 (main) - sample] input=0 output=0 total=0 cached=0",
            "[#2 (custom worker) - assistant] worker answer",
            "[#2 (custom worker) - sample] input=0 output=0 total=0 cached=0",
            "[#-1 (watcher) - debug] condition fired",
        ])
        self.assertEqual(original, sample.display_items())
        self.assertEqual(original[0].text, "[reasoning] first thought")

    def test_global_events_and_empty_batches_have_no_context_heading(self):
        global_items = (DisplayItem("Board: http://localhost/README.md"),
                        DisplayItem("[assistant] global notice", label="assistant"),
                        DisplayItem("-old\n+new", is_diff=True))
        rendered = auto._display_events(self.session, (
            auto._Event(1, ()),
            auto._Event(None, global_items),
            auto._Event(-1, ()),
        ))
        self.assertEqual(rendered, list(global_items))
        self.assertTrue(all(a is b for a, b in zip(rendered, global_items)))

    def test_unlabeled_context_notice_stays_in_one_item(self):
        notice = DisplayItem("Task failed; details withheld.")
        rendered = auto._display_events(self.session, (auto._Event(1, (notice,), "error"),))
        self.assertEqual(rendered, [DisplayItem("[#1 (main)]\nTask failed; details withheld.")])

    def test_arbitrary_message_role_is_not_parsed_as_bracket_syntax(self):
        original = auto.render_interaction_items((Message("custom] role", "body"),))
        rendered = auto._display_events(self.session, (auto._Event(1, original),))
        self.assertEqual(rendered[0].text, "[#1 (main) - custom] role] body")
        self.assertEqual(rendered[0].label, "#1 (main) - custom] role")

    def test_tool_payloads_keep_their_body_and_diff_colors(self):
        patch = "--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new"
        for name, payload in (("apply_patch", patch),
                              ("write_file", "[reasoning]\nthis is file content"),
                              ("write_file", '["a", "b"]')):
            with self.subTest(name=name, payload=payload):
                call = ToolCall(name, "edit", json.dumps({"content": payload}))
                original = auto.render_interaction_items((call,))
                rendered = auto._display_events(self.session, (auto._Event(2, original),))
                self.assertEqual(len(rendered), 2)
                self.assertEqual(rendered[0].text, f"[#2 (custom worker) - tool-call] {name} (edit)")
                self.assertEqual(rendered[1].text, f"[#2 (custom worker)]\n{payload}")
                self.assertEqual([item.is_diff for item in rendered],
                                 [item.is_diff for item in original])
                self.assertEqual(original[1].text, payload)
                if name == "apply_patch":
                    printed = str(rendered[1])
                    self.assertIn("\x1b[31m-old\x1b[0m", printed)
                    self.assertIn("\x1b[32m+new\x1b[0m", printed)
                    self.assertNotIn("\x1b[31m---", printed)
                    self.assertNotIn("\x1b[32m+++", printed)

    def test_tool_result_label_is_folded_without_rewriting_body_labels(self):
        call = ToolCall("exec_command", "diff", '{"cmd":"git diff"}')
        original = auto.render_interaction_items((
            ToolResult("diff", "[assistant] literal output\n@@ -1 +1 @@\n-old\n+new"),
        ), source_calls=(call,))
        rendered = auto._display_events(self.session, (auto._Event(1, original),))
        self.assertEqual(len(rendered), 1)
        self.assertEqual(rendered[0].text,
                         "[#1 (main) - tool-ret]  exec_command (diff) [ok]\n"
                         "[assistant] literal output\n@@ -1 +1 @@\n-old\n+new")
        self.assertTrue(rendered[0].is_diff)
        self.assertIn("\x1b[31m-old\x1b[0m", str(rendered[0]))
        self.assertIn("\x1b[32m+new\x1b[0m", str(rendered[0]))


class ConfigTests(unittest.TestCase):
    def test_default_limits_are_unset(self):
        args = build_parser().parse_args([])
        self.assertFalse(hasattr(args, "max_samples"))
        self.assertFalse(hasattr(args, "max_output_tokens"))
        for settings in resolve_config().values():
            self.assertIsNone(settings["max_samples"])
            self.assertIsNone(settings["max_output_tokens"])
            snapshot = InteractionConfig.from_namespace(namespace(settings)).snapshot()
            self.assertIsNone(snapshot.max_samples)
            self.assertIsNone(snapshot.sampling_options())

    def test_explicit_limits_and_per_context_null_overrides(self):
        args = build_parser().parse_args(["--max-samples", "3", "--max-output-tokens", "128"])
        overrides = {"max_samples": args.max_samples, "max_output_tokens": args.max_output_tokens}
        for settings in resolve_config(overrides=overrides).values():
            self.assertEqual(settings["max_samples"], 3)
            self.assertEqual(settings["max_output_tokens"], 128)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({
                "version": 1,
                "defaults": {"max_samples": 5, "max_output_tokens": 512},
                "contexts": {"2": {"max_samples": None, "max_output_tokens": None}},
            }))
            settings = resolve_config(path, overrides)
            self.assertEqual(settings[1]["max_samples"], 3)
            self.assertEqual(settings[1]["max_output_tokens"], 128)
            self.assertIsNone(settings[2]["max_samples"])
            self.assertIsNone(settings[2]["max_output_tokens"])
        for key in ("max_samples", "max_output_tokens"):
            for value in (0, -1, True, False, 1.5, "3"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    resolve_config(overrides={key: value})

    def test_unset_messages_budget_uses_catalog_or_requires_explicit_limit(self):
        model = "claude-fable-5-1"
        settings = resolve_config(overrides={"model_api": "messages", "model": model})[2]
        self.assertIsNone(settings["max_output_tokens"])
        snapshot = InteractionConfig.from_namespace(namespace(settings)).snapshot()
        self.assertEqual(snapshot.max_output_tokens, resolve_messages_max_output_tokens(model, None))
        with self.assertRaisesRegex(ValueError, "provide it explicitly"):
            resolve_config(overrides={"model_api": "messages", "model": "uncatalogued-auto-test-model"})

    def test_file_overrides_launch_defaults_and_resets_cross_api_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({
                "version": 1,
                "defaults": {"model_api": "codex", "model": "file-main", "codex_auth_file": "auth.json"},
                "contexts": {"2": {"model_api": "messages", "model": "worker", "api_key_env": "WORKER_KEY",
                                       "max_output_tokens": 64}},
            }))
            parsed = build_parser().parse_args(["--context-config", str(path), "--prompt", "hi"])
            self.assertFalse(hasattr(parsed, "model_api"))
            settings = resolve_config(path, {"model": "launch-main"})
            self.assertEqual(settings[1]["model"], "launch-main")
            self.assertEqual(settings[-1]["model"], "launch-main")
            self.assertEqual(settings[1]["codex_auth_file"], str(Path(tmp) / "auth.json"))
            self.assertEqual(settings[2]["model_api"], "messages")
            self.assertEqual(settings[2]["model"], "worker")
            self.assertIsNone(settings[2]["codex_auth_file"])
            self.assertEqual(settings[2]["api_key_env"], "WORKER_KEY")
            self.assertEqual([settings[i]["name"] for i in (1, 2, -1)], ["main", "worker", "watcher"])

    def test_config_validation_and_instruction_scope(self):
        settings = resolve_config(overrides={"instructions": ""})
        self.assertEqual(settings[1]["instructions"], "")
        self.assertIsNone(settings[2]["instructions"])
        bad = ({"max_samples": 0}, {"request_timeout_seconds": float("nan")},
               {"api_url": "http://user:secret@localhost"}, {"api_key": "SECRET"},
               {"model_api": "other"}, {"max_output_tokens": True},
               {"cwd": None}, {"api_url": True}, {"api_url": "http://localhost:bad"})
        for overrides in bad:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolve_config(overrides=overrides)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            for content in ('{"version":1,"contexts":{"0":{}}}',
                            '{"version":1,"version":1}',
                            '{"version":1,"contexts":{"-1":{"name":"bad\\nname"}}}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    resolve_config(path)


class InstructionTests(unittest.TestCase):
    def setUp(self):
        self.settings = resolve_config()
        self.base_url = "http://127.0.0.1:54321"

    def text(self, index):
        return auto._instructions(index, self.settings[index], self.base_url).text

    def test_cooperation_paragraph_precedes_each_default_role(self):
        preambles = []
        for index, name in ((1, "main"), (2, "worker"), (-1, "watcher")):
            with self.subTest(index=index):
                text = self.text(index)
                preamble, rest = text.split("\n\n", 1)
                preambles.append(preamble)
                self.assertIn("cooperative effort to complete the user's task", preamble)
                self.assertIn("shared message board", preamble)
                self.assertIn("iterative revisions toward a verified result", preamble)
                self.assertTrue(rest.startswith(f"You are {name}"))
                self.assertEqual(text.count("# Shared message board instructions"), 1)
                self.assertIn(f"Address: {self.base_url}\n", text)
                self.assertIn(f"Read {self.base_url}/README.md", text)
        self.assertEqual(len(set(preambles)), 1)
        self.assertIn("The host displays a debug event", self.text(-1))
        self.assertIn("No model polling is needed", self.text(-1))

    def test_main_delegates_implementation_and_reviews_iteratively(self):
        text = self.text(1)
        for requirement in (
            "You are main, the user-facing planner and reviewer",
            "acceptance criteria, and required checks",
            "Call board_post_plan to assign implementation and tests to worker #2",
            "Worker owns code and test edits",
            "do not instead assign worker a read-only review and implement the change yourself",
            "Use board_read_thread to obtain the result for each plan",
            "Review the diff and reported checks against the user's requirements",
            "post a concrete follow-up execution plan in the same thread",
            "review the revised work",
            "remain active through worker execution and your review",
            "later worker results do not automatically restart it",
            "If blocked, report what remains unresolved",
            "Respect explicitly planning-only or review-only user requests",
            "Do not resubmit a plan after an uncertain tool outcome without checking the board",
            "Local update_plan only maintains your checklist; it does not delegate work",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, text)

    def test_worker_implements_revises_and_reports_checks(self):
        text = self.text(2)
        for requirement in (
            "You are worker, the implementer",
            "including code changes and tests when implementation is requested",
            "Do not substitute another implementation proposal",
            "For follow-up assignments, revise the existing work according to main's review",
            "Respect explicit read-only assignments",
            "Finish relevant commands before handing work back to main for review",
            "changed files, checks and their outcomes, and remaining blockers",
            "do not claim unperformed work",
            "The host will publish your final outcome",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, text)

    def test_custom_instructions_replace_defaults_but_keep_board_discovery(self):
        for index in (1, 2, -1):
            for custom in ("Use my custom role contract.", "", " \n"):
                with self.subTest(index=index, custom=custom):
                    self.settings[index]["instructions"] = custom
                    body, board = self.text(index).split("\n\n# Shared message board instructions\n\n", 1)
                    self.assertEqual(body, custom)
                    self.assertIn(f"Address: {self.base_url}\n", board)
                    self.assertIn(f"Read {self.base_url}/README.md", board)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "session"
        self.calls = {1: [], 2: []}
        self.options = {1: [], 2: []}
        self.effects = []
        self.threads = {i: [] for i in (1, 2, -1)}
        self.closed = []

    def session(self, scripts, extra_tools=(), *, settings_overrides=None, **kwargs):
        test = self
        class Model:
            def __init__(self, index):
                self.index = index
                self.outcomes = deque(scripts.get(index, ()))
                test.threads[index].append(threading.get_ident())

            def sample(self, context, **options):
                test.threads[self.index].append(threading.get_ident())
                test.calls[self.index].append(context.copy())
                test.options[self.index].append(options.get("options"))
                saved = load_interaction_save(test.path / "contexts" / f"{self.index}.jsonl")
                test.assertEqual(saved.items, context.items)
                test.assertFalse(context.pending_tool_calls())
                if not self.outcomes:
                    raise AssertionError("unexpected model request")
                value = self.outcomes.popleft()
                if isinstance(value, BaseException):
                    raise value
                return value(context) if callable(value) else value

        class Tools(Environment):
            def __init__(self, index, tools):
                self.index = index
                test.threads[index].append(threading.get_ident())
                super().__init__((*tools, *extra_tools))
            def execute_tool_calls(self, calls):
                test.threads[self.index].append(threading.get_ident())
                saved = load_interaction_save(test.path / "contexts" / f"{self.index}.jsonl")
                test.assertEqual(saved.pending_tool_calls(), tuple(calls))
                return super().execute_tool_calls(calls)
            def close(self):
                test.threads[self.index].append(threading.get_ident())
                test.closed.append(self.index)

        settings = resolve_config(overrides={"cwd": self.temp.name, **(settings_overrides or {})})
        session = auto._Session(self.path, settings,
                               model_factory=lambda i, args: Model(i),
                               environment_factory=lambda i, args, tools: Tools(i, tools), **kwargs).start()
        self.addCleanup(session.close)
        return session

    def finished(self, session, thread):
        wait_for(lambda: session.thread_result(thread) is not None)
        return session.thread_result(thread)

    def test_main_and_worker_can_run_past_eight_samples_by_default(self):
        tool = Tool(ToolSpec("noop", "continue the test", {}),
                    lambda *args, **kwargs: ToolOutcome("ok"))
        def rounds(prefix, count):
            return [ModelSample((ToolCall("noop", f"{prefix}-{i}", "{}"),)) for i in range(count)]
        session = self.session({
            1: [ModelSample((ToolCall("board_post_plan", "delegate", '{"content":"long task"}'),)),
                *rounds("main", 8), answer("main complete")],
            2: [*rounds("worker", 9), answer("worker complete")],
        }, (tool,))
        source = session.submit("long conversation")
        self.assertTrue(self.finished(session, source["thread_id"]))
        for index in (1, 2):
            self.assertEqual(len(self.calls[index]), 10)
            self.assertTrue(all(options is None for options in self.options[index]))

    def test_explicit_limits_still_apply(self):
        tool = Tool(ToolSpec("noop", "continue the test", {}),
                    lambda *args, **kwargs: ToolOutcome("ok"))
        session = self.session({1: [
            ModelSample((ToolCall("noop", "first", "{}"),)),
            ModelSample((ToolCall("noop", "second", "{}"),)),
            answer("must not be sampled"),
        ]}, (tool,), settings_overrides={"max_samples": 2, "max_output_tokens": 128})
        source = session.submit("bounded conversation")
        self.assertFalse(self.finished(session, source["thread_id"]))
        self.assertEqual(len(self.calls[1]), 2)
        self.assertTrue(all(options.max_output_tokens == 128 for options in self.options[1]))
        self.assertFalse(any(e.kind == "debug" for e in session.drain_events()))

    def test_final_sample_at_explicit_limit_succeeds(self):
        session = self.session({1: [answer()]}, settings_overrides={"max_samples": 1})
        source = session.submit("one sample")
        self.assertTrue(self.finished(session, source["thread_id"]))
        self.assertEqual(len(self.calls[1]), 1)

    def test_quiescent_board_first_mixed_roles_thread_affinity_and_debug_only_watcher(self):
        tool = Tool(ToolSpec("do_work", "Write proof", {"type": "object"}),
                    lambda args, **kwargs: (self.effects.append(threading.get_ident()) or ToolOutcome("proof written")))
        session = self.session({
            1: [ModelSample((Message("assistant", "Delegating, not final"),
                             ToolCall("board_post_plan", "p1", '{"content":"Write proof"}'))), answer("main done")],
            2: [ModelSample((ToolCall("do_work", "p1", "{}"),)), answer("worker done")],
        }, (tool,))
        for i in (1, 2, -1):
            self.assertIn("quiescent", session.status(i))
            context = load_interaction_save(self.path / "contexts" / f"{i}.jsonl")
            self.assertEqual(len(context), 2)
            self.assertIn(session.service.base_url + "/README.md", context[1].text)
        self.assertEqual(session.service.board.records(), ())
        self.assertEqual(self.calls, {1: [], 2: []})
        self.assertEqual(self.effects, [])
        first = session.submit("original task", request_id="user-request")
        self.assertEqual(session.submit("original task", request_id="user-request"), first)
        self.assertTrue(self.finished(session, first["thread_id"]))
        records = session.service.board.records()
        self.assertEqual([r.kind for r in records].count("user"), 1)
        self.assertEqual([r.kind for r in records].count("plan"), 1)
        self.assertTrue(all(r.thread_id == first["thread_id"] for r in records))
        plan = next(r for r in records if r.kind == "plan")
        result = next(r for r in records if r.kind == "result")
        self.assertEqual(result.reply_to, plan.record_id)
        self.assertEqual(result.content, "worker done")
        self.assertEqual(len(self.calls[1]), 2)
        self.assertEqual(len(self.calls[2]), 2)
        self.assertEqual(len(self.effects), 1)
        debug = [e for e in session.drain_events() if e.kind == "debug"]
        self.assertEqual(len(debug), 1)
        self.assertEqual(debug[0].index, -1)
        self.assertIn("condition fired", debug[0].items[0].text)
        # The latest user request only asks for a debug event, not a watcher model turn/log schema.
        self.assertEqual(len(load_interaction_save(self.path / "contexts" / "-1.jsonl")), 2)
        session.close()
        self.assertCountEqual(self.closed, (1, 2, -1))
        for index, thread_ids in self.threads.items():
            self.assertEqual(len(set(thread_ids)), 1, index)
        self.assertEqual(len({v[0] for v in self.threads.values()}), 3)
        self.assertEqual(self.effects[0], self.threads[2][0])
        self.assertFalse(any(t.is_alive() for t in session._threads.values()))

    def test_waits_for_worker_and_fast_completion_watch_notification(self):
        worker_entered, release_worker, release_watcher = (threading.Event() for _ in range(3))
        original_watch = auto._Session._watch
        def delayed_watch(session):
            release_watcher.wait(5)
            original_watch(session)
        def worker(_context):
            worker_entered.set()
            self.assertTrue(release_worker.wait(5))
            return answer("worker late")
        with mock.patch.object(auto._Session, "_watch", delayed_watch):
            session = self.session({
                1: [ModelSample((ToolCall("board_post_plan", "p", '{"content":"slow"}'),)), answer()],
                2: [worker],
            })
        try:
            source = session.submit("task")
            self.assertTrue(worker_entered.wait(3))
            wait_for(lambda: source["record_id"] in session._done)
            self.assertIsNone(session.thread_result(source["thread_id"]))
            release_worker.set()
            wait_for(lambda: any(r.kind == "result" for r in session.service.board.records()))
            self.assertIsNone(session.thread_result(source["thread_id"]))
            release_watcher.set()
            self.assertTrue(self.finished(session, source["thread_id"]))
        finally:
            release_worker.set()
            release_watcher.set()
            session.close()

    def test_failure_and_reasoning_only_do_not_fire_watch_or_execute_recovered_calls(self):
        failure = ModelTransportError("SECRET", completed_items=(
            Message("assistant", "partial"), ToolCall("forbidden", "bad", "{}")),
            failure=ModelFailure("transport", "safe failure"))
        tool = Tool(ToolSpec("forbidden", "must not execute", {}),
                    lambda *args, **kwargs: self.fail("recovered call executed"))
        session = self.session({1: [failure, ModelSample((Reasoning("thought only"),))]}, (tool,))
        for prompt in ("first", "second"):
            source = session.submit(prompt)
            self.assertFalse(self.finished(session, source["thread_id"]))
        context = load_interaction_save(self.path / "contexts" / "1.jsonl")
        self.assertFalse(context.pending_tool_calls())
        self.assertTrue(any(isinstance(i, ToolResult) and not i.success for i in context))
        self.assertFalse(any(e.kind == "debug" for e in session.drain_events()))
        self.assertNotIn("SECRET", (self.path / "index.jsonl").read_text())

    def test_whole_tool_batch_places_injected_messages_after_all_results(self):
        tool = Tool(ToolSpec("inject", "inject a user message", {}),
                    lambda *args, **kwargs: ToolOutcome("ok", user_messages=(Message("user", "injected"),)))
        session = self.session({1: [ModelSample((ToolCall("inject", "a", "{}"),
                                                ToolCall("inject", "b", "{}"))), answer()]}, (tool,))
        source = session.submit("batch")
        self.assertTrue(self.finished(session, source["thread_id"]))
        items = self.calls[1][1].items
        results = [n for n, i in enumerate(items) if isinstance(i, ToolResult)]
        messages = [n for n, i in enumerate(items) if isinstance(i, Message) and i.content == "injected"]
        self.assertEqual(len(results), 2)
        self.assertLess(max(results), min(messages))

    def test_summary_save_failure_blocks_watcher(self):
        session = self.session({1: [answer()]})
        real_save = auto.save_interaction_save
        def save(path, context):
            if isinstance(context.items[-1], TurnSummary):
                from pythia.interaction import SaveError
                raise SaveError("disk")
            return real_save(path, context)
        with mock.patch.object(auto, "save_interaction_save", save):
            source = session.submit("fail summary")
            self.assertFalse(self.finished(session, source["thread_id"]))
            session.close()
        self.assertFalse(any(e.kind == "debug" for e in session.drain_events()))

    def test_encoding_failure_is_a_fatal_checkpoint_failure(self):
        session = self.session({1: [answer("unencodable: \ud800")]})
        source = session.submit("bad model text")
        self.assertFalse(self.finished(session, source["thread_id"]))
        self.assertTrue(session._stop.is_set())
        session.close()
        saved = load_interaction_save(self.path / "contexts" / "1.jsonl")
        self.assertFalse(any(isinstance(i, Message) and i.role == "assistant" for i in saved))
        self.assertFalse(any(e.kind == "debug" for e in session.drain_events()))

    def test_compaction_continuation_is_not_an_end_of_turn(self):
        reached, release = threading.Event(), threading.Event()
        def final(_context):
            reached.set()
            self.assertTrue(release.wait(5))
            return answer("after compaction")
        session = self.session({1: [ModelSample((OpaqueCompaction("opaque", "messages"),),
                                               stop_reason="compaction"), final]})
        try:
            source = session.submit("compact")
            self.assertTrue(reached.wait(3))
            self.assertFalse(any(e.kind == "debug" for e in session.drain_events()))
            release.set()
            self.assertTrue(self.finished(session, source["thread_id"]))
            self.assertEqual(len([e for e in session.drain_events() if e.kind == "debug"]), 1)
        finally:
            release.set()
            session.close()

    def test_stop_during_sample_checkpoints_but_does_not_execute_new_calls(self):
        reached, release = threading.Event(), threading.Event()
        def sample(_context):
            reached.set()
            self.assertTrue(release.wait(5))
            return ModelSample((ToolCall("effect", "pending", "{}"),))
        tool = Tool(ToolSpec("effect", "must not run after stop", {}),
                    lambda *args, **kwargs: (self.effects.append("bad") or ToolOutcome("bad")))
        session = self.session({1: [sample]}, (tool,))
        session.submit("stop in flight")
        self.assertTrue(reached.wait(3))
        closer = threading.Thread(target=session.close)
        closer.start()
        try:
            wait_for(session._stop.is_set)
            release.set()
            closer.join(3)
            self.assertFalse(closer.is_alive())
            self.assertEqual(self.effects, [])
            saved = load_interaction_save(self.path / "contexts" / "1.jsonl")
            self.assertEqual([c.call_id for c in saved.pending_tool_calls()], ["pending"])
            self.assertFalse(any(e.kind == "debug" for e in session.drain_events()))
        finally:
            release.set()
            closer.join(5)

    def test_board_failure_during_sample_blocks_the_following_tool_batch(self):
        reached, release = threading.Event(), threading.Event()
        def sample(_context):
            reached.set()
            self.assertTrue(release.wait(5))
            return ModelSample((ToolCall("effect", "pending", "{}"),))
        tool = Tool(ToolSpec("effect", "must not run with failed persistence", {}),
                    lambda *args, **kwargs: (self.effects.append("bad") or ToolOutcome("bad")))
        session = self.session({1: [sample]}, (tool,))
        try:
            source = session.submit("first")
            self.assertTrue(reached.wait(3))
            with mock.patch("pythia.interaction._auto_board.os.fsync", side_effect=OSError("disk")):
                with self.assertRaises(BoardError):
                    session.submit("failed commit")
            release.set()
            wait_for(session._stop.is_set)
            session.close()
            self.assertEqual(self.effects, [])
            self.assertFalse(session.thread_result(source["thread_id"]))
            self.assertEqual([r.content for r in session.service.board.records()], ["first"])
        finally:
            release.set()
            session.close()

    def test_startup_failure_closes_constructed_environments_without_role_work(self):
        closed = []
        class Env(Environment):
            def __init__(self, index):
                super().__init__()
                self.index = index
                self.owner = threading.get_ident()
            def close(self):
                closed.append((self.index, self.owner, threading.get_ident()))
        def model(index, args):
            if index == 2:
                raise RuntimeError("FAKE_SECRET")
            return mock.Mock()
        session = auto._Session(self.path, resolve_config(), model_factory=model,
                               environment_factory=lambda i, a, t: Env(i))
        with self.assertRaises(RuntimeError):
            session.start()
        self.assertCountEqual([i for i, _, _ in closed], (1, 2, -1))
        self.assertTrue(all(a == b for _, a, b in closed))
        self.assertFalse(any(t.is_alive() for t in session._threads.values()))
        self.assertEqual(session.service.board.records(), ())
        self.assertNotIn("FAKE_SECRET", repr(session.drain_events()))

    def test_interactive_navigation_is_responsive_without_direct_worker_input(self):
        reached, release = threading.Event(), threading.Event()
        def sample(_context):
            reached.set()
            self.assertTrue(release.wait(5))
            return answer("interactive done")
        session = self.session({1: [sample]})
        test = self
        class Terminal:
            closed = False
            def __init__(self):
                self.keys = deque()
                self.frames, self.items = [], []
                self.stage = 0
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def submit(self, text):
                self.keys.extend(SimpleNamespace(key=k, data=v) for k, v in (
                    ("c-u", ""), ("c-k", ""), ("<bracketed-paste>", text), ("c-m", "\r")))
            def read_keys(self):
                keys = tuple(self.keys)
                self.keys.clear()
                return keys
            def render(self, editor, status, items, prompt=":> "):
                self.frames.append(status)
                self.items.extend(items)
                if self.stage == 0:
                    self.submit("/context 2")
                    self.stage = 1
                elif self.stage == 1 and status.startswith("#2"):
                    self.submit("do not send to worker")
                    self.stage = 2
                elif self.stage == 2:
                    test.assertEqual(editor.text, "do not send to worker")
                    test.assertEqual(session.service.board.records(), ())
                    self.submit("/context 1")
                    self.stage = 3
                elif self.stage == 3:
                    self.submit("actual task")
                    self.stage = 4
                elif self.stage == 4 and reached.is_set():
                    self.submit("/context #-1")
                    self.stage = 5
                elif self.stage == 5 and status.startswith("#-1"):
                    self.submit("/contexts")
                    release.set()
                    self.stage = 6
                elif self.stage == 6 and any("condition fired" in i.text for i in self.items):
                    source = session.service.board.records()[0]
                    if session.thread_result(source.thread_id) is not None:
                        self.submit("/quit")
                        self.stage = 7
        terminal = Terminal()
        try:
            result = asyncio.run(asyncio.wait_for(auto._interactive(session, terminal), timeout=6))
            self.assertEqual(result, 0)
            self.assertEqual(len(self.calls[1]), 1)
            self.assertEqual(self.calls[2], [])
            self.assertEqual([r.kind for r in session.service.board.records()], ["user", "answer"])
            self.assertTrue(any(s.startswith("#1 (main) - sampling") for s in terminal.frames))
            self.assertTrue(any(s.startswith("#-1") for s in terminal.frames))
            self.assertTrue(any(i.text == "[#1 (main) - assistant] interactive done" for i in terminal.items))
            self.assertTrue(any(i.text.startswith("[#-1 (watcher) - debug]") for i in terminal.items))
            self.assertFalse(any(i.text == "[#1 (main)]" for i in terminal.items))
            self.assertFalse(any(t.is_alive() for t in session._threads.values()))
        finally:
            release.set()
            session.close()

    def test_stop_before_input_and_local_navigation_do_not_wake_owners(self):
        session = self.session({})
        for text, target in (("/context 2", 2), ("/context #-1", -1), ("/contexts", 1)):
            selected, items, stop = auto._local_command(text, 1, session)
            self.assertEqual(selected, target)
            self.assertFalse(stop)
        for text in ("/context 0", "/context -2", "/context -1\nhello", "/resume"):
            with self.assertRaises(ValueError):
                auto._local_command(text, 1, session)
        session.close()
        session.close()
        self.assertEqual(self.calls, {1: [], 2: []})
        self.assertEqual(session.service.board.records(), ())
        self.assertCountEqual(self.closed, (1, 2, -1))

    def test_existing_directory_is_not_modified(self):
        self.path.mkdir()
        marker = self.path / "keep"
        marker.write_text("untouched")
        with self.assertRaises(FileExistsError):
            self.session({})
        self.assertEqual(list(self.path.iterdir()), [marker])
        self.assertEqual(marker.read_text(), "untouched")


class EntryPointTests(unittest.TestCase):
    def test_codex_auto_flow_rejects_system_wire_messages_like_the_real_endpoint(self):
        # The original auto fixtures covered Chat Completions/Messages, but did
        # not enforce Codex's role restriction on generated default instructions.
        seen = []

        class CodexGateway(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append(request)
                if any(item.get("role") == "system" for item in request["input"]):
                    data = b'{"detail":"System messages are not allowed"}'
                    status, content_type = 400, "application/json"
                else:
                    main = request["model"] == "gpt-6-astra"
                    main_calls = sum(r["model"] == "gpt-6-astra" for r in seen)
                    if main and main_calls == 1:
                        item = {"type": "function_call", "name": "board_post_plan",
                                "call_id": "delegate", "arguments": '{"content":"Review the worker task"}'}
                    else:
                        item = {"type": "message", "role": "assistant", "content": [{
                            "type": "output_text", "text": "codex main done" if main else "codex worker done"}]}
                    events = (
                        {"type": "response.output_item.done", "output_index": 0, "item": item},
                        {"type": "response.completed", "response": {
                            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}}},
                    )
                    data = b"".join(("data: " + json.dumps(event) + "\n\n").encode() for event in events)
                    status, content_type = 200, "text/event-stream"
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        gateway = ThreadingHTTPServer(("127.0.0.1", 0), CodexGateway)
        server_thread = threading.Thread(target=gateway.serve_forever, kwargs={"poll_interval": .01})
        server_thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                auth = root / "auth.json"
                auth.write_text(json.dumps({"tokens": {
                    "access_token": "FAKE_CODEX_SECRET", "account_id": "test-account"}}))
                settings = root / "auto.json"
                settings.write_text(json.dumps({
                    "version": 1,
                    "defaults": {"model_api": "codex", "model": "gpt-6-astra-max",
                                 "api_url": f"http://127.0.0.1:{gateway.server_port}",
                                 "codex_auth_file": str(auth), "request_timeout_seconds": 3,
                                 "cwd": tmp},
                    "contexts": {"2": {"model": "gpt-5.6-sol-max"}},
                }))
                result = subprocess.run([
                    sys.executable, "-m", "pythia.interaction.auto", "--context-config", str(settings),
                    "--save", str(root / "run"), "--prompt", "Review auto startup.",
                ], capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("codex main done", result.stdout)
                self.assertIn("codex worker done", result.stdout)
                self.assertIn("condition fired", result.stdout)
                self.assertEqual(len(seen), 3)
                self.assertEqual({r["model"] for r in seen}, {"gpt-6-astra", "gpt-5.6-sol"})
                for request in seen:
                    self.assertEqual(request["input"][0]["role"], "developer")
                    self.assertIn("/README.md", request["input"][0]["content"][0]["text"])
                    self.assertEqual(request["reasoning"]["effort"], "max")
                    self.assertFalse(any(item.get("role") == "system" for item in request["input"]))
                for path in (root / "run").rglob("*"):
                    if path.is_file():
                        self.assertNotIn("FAKE_CODEX_SECRET", path.read_text())
                self.assertNotIn("FAKE_CODEX_SECRET", result.stdout + result.stderr)
        finally:
            gateway.shutdown()
            server_thread.join()
            gateway.server_close()

    @unittest.skipUnless(os.name == "posix", "requires a POSIX pseudo-terminal")
    def test_pty_quiescence_navigation_and_exit_without_model_work(self):
        import fcntl
        import pty
        import select
        import struct
        import termios

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pty-run"
            master, slave = pty.openpty()
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 120, 0, 0))
            process = subprocess.Popen([
                sys.executable, "-m", "pythia.interaction.auto", "--save", str(path),
                "--api-url", "http://127.0.0.1:1", "--request-timeout-seconds", "1",
            ], stdin=slave, stdout=slave, stderr=subprocess.PIPE,
               env={**os.environ, "TERM": "xterm-256color"})
            os.close(slave)
            output = bytearray()
            def expect(marker, after=0):
                deadline = time.monotonic() + 5
                while marker not in output[after:]:
                    if time.monotonic() >= deadline:
                        self.fail(f"PTY did not show {marker!r}: {bytes(output)!r}")
                    if select.select([master], [], [], .05)[0]:
                        try:
                            output.extend(os.read(master, 65536))
                        except OSError:
                            self.fail(f"PTY closed early: {bytes(output)!r}")
            try:
                expect(b"#1 (main) - quiescent")
                start = len(output)
                os.write(master, b"/context -1\r")
                expect(b"#-1 (watcher) - quiescent", start)
                start = len(output)
                os.write(master, b"not a task\r")
                expect(b"Switch to /context 1", start)
                os.write(master, b"\x04")
                self.assertEqual(process.wait(timeout=5), 0)
                self.assertEqual(process.stderr.read(), b"")
                self.assertEqual((path / "index.jsonl").read_text(), "")
                for index in (1, 2, -1):
                    self.assertEqual(len(load_interaction_save(path / "contexts" / f"{index}.jsonl")), 2)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stderr.close()
                os.close(master)

    def test_one_prompt_real_http_adapters_with_separate_apis_and_no_watcher_sample(self):
        seen = []
        class Gateway(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((self.path, request))
                if self.path.endswith("/messages"):
                    worker_calls = sum(p.endswith("/messages") for p, _ in seen)
                    content = ([{"type": "tool_use", "id": "work", "name": "exec_command",
                                 "input": {"cmd": "printf proof > proof.txt", "yield_time_ms": 1000}}]
                               if worker_calls == 1 else [{"type": "text", "text": "worker proof"}])
                    payload = {"id": "m", "type": "message", "role": "assistant",
                               "model": "worker", "content": content,
                               "stop_reason": "tool_use" if worker_calls == 1 else "end_turn",
                               "usage": {"input_tokens": 5, "output_tokens": 2}}
                else:
                    main_calls = sum(not p.endswith("/messages") for p, _ in seen)
                    message = ({"role": "assistant", "content": None, "tool_calls": [{
                        "id": "delegate", "type": "function", "function": {
                            "name": "board_post_plan", "arguments": '{"content":"Produce a proof"}'}}]}
                        if main_calls == 1 else {"role": "assistant", "content": "main answer"})
                    payload = {"choices": [{"message": message, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        gateway = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
        server_thread = threading.Thread(target=gateway.serve_forever, kwargs={"poll_interval": .01})
        server_thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                settings = root / "contexts.json"
                url = f"http://127.0.0.1:{gateway.server_port}"
                settings.write_text(json.dumps({
                    "version": 1, "defaults": {"model_api": "chat-completions", "api_url": url,
                        "model": "main", "request_timeout_seconds": 3, "cwd": tmp},
                    "contexts": {"2": {"model_api": "messages", "model": "worker", "api_url": url,
                                           "api_key_env": "AUTO_TEST_KEY", "max_output_tokens": 128}},
                }))
                result = subprocess.run([sys.executable, "-m", "pythia.interaction.auto",
                    "--context-config", str(settings), "--save", str(root / "run"), "--prompt", "Do the task"],
                    env={**os.environ, "AUTO_TEST_KEY": "FAKE_PROVIDER_SECRET"},
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("[#1 (main) - assistant] main answer", result.stdout)
                self.assertIn("[#2 (worker) - assistant] worker proof", result.stdout)
                self.assertIn("[#-1 (watcher) - debug]", result.stdout)
                self.assertNotIn("[#1 (main)]", result.stdout)
                self.assertEqual((root / "proof.txt").read_text(), "proof")
                self.assertIn("condition fired", result.stdout)
                self.assertIn("#1 (main)", result.stdout)
                self.assertNotIn("FAKE_PROVIDER_SECRET", result.stdout + result.stderr)
                for file in (root / "run").rglob("*"):
                    if file.is_file():
                        self.assertNotIn("FAKE_PROVIDER_SECRET", file.read_text())
                records = [json.loads(line) for line in (root / "run" / "index.jsonl").read_text().splitlines()]
                self.assertEqual([r["kind"] for r in records].count("user"), 1)
                self.assertEqual([r["kind"] for r in records].count("plan"), 1)
                self.assertEqual(len({r["thread_id"] for r in records}), 1)
                self.assertEqual(len(seen), 4)
                self.assertTrue(any(p.endswith("/messages") for p, _ in seen))
                for path, request in seen:
                    if path.endswith("/messages"):
                        self.assertEqual(request["max_tokens"], 128)
                    else:
                        self.assertNotIn("max_tokens", request)
                        self.assertNotIn("max_completion_tokens", request)
                self.assertEqual(len(load_interaction_save(root / "run" / "contexts" / "-1.jsonl")), 2)
        finally:
            gateway.shutdown()
            server_thread.join()
            gateway.server_close()

    def test_help_and_no_ignored_resume(self):
        result = subprocess.run([sys.executable, "-m", "pythia.interaction.auto", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--context-config", result.stdout)
        self.assertNotIn("--resume", result.stdout)
        result = subprocess.run([sys.executable, "-m", "pythia.interaction.auto", "--resume"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
