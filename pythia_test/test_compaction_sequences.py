"""Compaction sequences (compaction_mode), coldstart compaction, and live /config.

See doc/warm-cold-compaction-plan.md, Parts 1 and 2.
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from pythia.interaction import (
    BUILTIN_MODEL_CATALOG, COLDSTART_POINTER_PREFIX, COMPACTION_MODES,
    COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX, DEFAULT_COMPACTION_MODE,
    ColdstartCompactor,
    CompactionContextWindowError, CompactionError, CompactionLog, CompactionLogError,
    CompactionMetadata,
    CompactionResult, CompactionSettings, ConfigError, ContextPrefix, DefaultEnvironment,
    DisplayItem, Environment, Init, Instructions, InteractionConfig, InteractionConfigSnapshot,
    InteractionContext, InteractionSaveWriter, Message, ModelAuthenticationError,
    ModelCancelled, ModelConfigurationError, ModelContextWindowError, ModelFailure,
    ModelSample, ModelSampleBoundary,
    ModelTransportError, NothingToCompact, OpaqueCompaction, PiCompactor, SampleMetadata,
    TokenUsage, Tool, ToolCall, ToolOutcome, ToolResult, ToolSpec, TurnSummary,
    UserInteractionBoundary, UserToolCall, UserToolResult, create_default_compactor,
    is_coldstart_pointer, load_interaction_save, normalize_compaction_mode,
    parse_compaction_mode, resume_interaction_save, run_compaction, save_interaction_save,
)
from pythia.interaction import auto, cli, demo
from pythia.interaction._auto_config import build_parser as build_auto_parser, resolve_config
from pythia.interaction.loop import kernel
from pythia.interaction.loop.tail import turn_tail
from pythia.interaction.messages import _messages_http_failure
from pythia.interaction.runtime_config import parse_config_literal, resolve_compaction_mode
from pythia.interaction.save import interaction_item_from_dict


EXEC = ToolSpec("exec_command", "Run a command.", {"type": "object"})
TOOLS = (EXEC,)
NO_EXEC = ("coldstart compaction needs exec_command to read the saved log")
# The summary template that coldstart writes, after its first line: pi's sections.
SECTIONS = (
    "## Goal\n"
    "[What is the user trying to accomplish? Can be multiple items if the session covers "
    "different tasks.]\n\n"
    "## Constraints & Preferences\n"
    "- [Any constraints, preferences, or requirements mentioned by user]\n"
    '- [Or "(none)" if none were mentioned]\n\n'
    "## Progress\n"
    "### Done\n"
    "- [x] [Completed tasks/changes]\n\n"
    "### In Progress\n"
    "- [ ] [Current work]\n\n"
    "### Blocked\n"
    "- [Issues preventing progress, if any]\n\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Brief rationale]\n\n"
    "## Next Steps\n"
    "1. [Ordered list of what should happen next]\n\n"
    "## Critical Context\n"
    "- [Any data, examples, or references needed to continue]\n"
    '- [Or "(none)" if not applicable]\n'
)


def _answer(text="done"):
    return ModelSample((Message("assistant", text),))


def _exec_tool():
    return Tool(EXEC, lambda arguments, *, timeout_seconds=None: ToolOutcome("ok"))


def _history():
    """Two turns; the second request has no answer yet. Usage is over 100."""
    return (
        Init("session"),
        Instructions("Agent prompt."),
        Message("user", "Fix the bug."),
        UserInteractionBoundary(),
        ToolCall("exec_command", "call_1", '{"cmd":"ls"}'),
        ModelSampleBoundary(),
        ToolResult("call_1", "bug.py"),
        Message("assistant", "Fixed."),
        SampleMetadata(TokenUsage(input_tokens=900, total_tokens=1000)),
        ModelSampleBoundary(),
        Message("user", "Now the tests."),
        UserInteractionBoundary(),
    )


def _result(protocol="pi", *prefix):
    return CompactionResult((ContextPrefix(prefix or (Message("user", "summary"),)),),
                            protocol=protocol)


def _pointer(result):
    pointer = result.items[0].prefix_items[-1]
    assert is_coldstart_pointer(pointer), pointer
    return pointer.content


class _Script:
    """A compactor factory whose compactors play scripted outcomes in turn."""

    def __init__(self, *outcomes):
        self.outcomes = deque(outcomes)
        self.calls = []

    def __call__(self, model, settings, **options):
        script = self

        class Compactor:
            def compact(self, context, **kwargs):
                script.calls.append((settings.mode, options, kwargs))
                outcome = script.outcomes.popleft()
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome
        return Compactor()

    @property
    def methods(self):
        return [mode for mode, _options, _kwargs in self.calls]


class _Model:
    """Plays scripted outcomes for model.sample, recording each request."""

    def __init__(self, *outcomes):
        self.outcomes = deque(outcomes)
        self.calls = []

    def sample(self, context, *, tools=(), sample_params=None):
        self.calls.append(context.copy())
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _timeout():
    return ModelTransportError("timed out", failure=ModelFailure("transport", "timed out"))


class SequenceGrammarTests(unittest.TestCase):
    def test_sequences_parse_to_their_normal_form(self):
        for text, entries, normal in (
            ("pi", (("pi", 1),), "pi"),
            ("provider", (("provider", 1),), "provider"),
            ("legacydefault", (("legacydefault", 1),), "legacydefault"),
            ("coldstart", (("coldstart", 1),), "coldstart"),
            ("legacydefault:2,coldstart:1", (("legacydefault", 2), ("coldstart", 1)), "legacydefault:2,coldstart"),
            (" legacydefault : 2 , coldstart ", (("legacydefault", 2), ("coldstart", 1)),
             "legacydefault:2,coldstart"),
            ("provider:2,pi,coldstart", (("provider", 2), ("pi", 1), ("coldstart", 1)),
             "provider:2,pi,coldstart"),
            ("pi:1,pi:9", (("pi", 1), ("pi", 9)), "pi,pi:9"),
            # coldstart can fail, so methods may follow it.
            ("coldstart,legacydefault:2", (("coldstart", 1), ("legacydefault", 2)), "coldstart,legacydefault:2"),
            ("coldstart:2,pi", (("coldstart", 2), ("pi", 1)), "coldstart:2,pi"),
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_compaction_mode(text), entries)
                self.assertEqual(normalize_compaction_mode(text), normal)
                self.assertEqual(CompactionSettings(mode=text).mode, normal)

    def test_invalid_sequences_fail_naming_the_setting(self):
        for value in ("remote", "pi:0", "pi:10", "pi:x", "pi:", "pi,,coldstart",
                      "", " ", None, 3):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "compaction_mode"):
                parse_compaction_mode(value)

    def test_the_routes_own_method_is_named_legacydefault_not_default(self):
        # It was the default before coldstart came first, and is no longer.
        self.assertEqual(COMPACTION_MODES,
                         ("legacydefault", "pi", "warmpi", "provider", "coldstart"))
        self.assertEqual(DEFAULT_COMPACTION_MODE, "coldstart,warmpi:2")
        message = (r"^compaction_mode: unknown method 'default'; use legacydefault, pi, "
                   r"warmpi, provider, coldstart\.$")
        for value in ("default", "default:2,coldstart", "coldstart,default:2"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                parse_compaction_mode(value)
        self.assertEqual(parse_config_literal("compaction_mode", "legacydefault"), "legacydefault")
        with self.assertRaisesRegex(ConfigError, "unknown method 'default'"):
            parse_config_literal("compaction_mode", "default")
        # The options name it in their help, and reject the old name.
        for name, parser in (("cli", cli._build_parser()), ("demo", demo._build_parser()),
                             ("auto", build_auto_parser())):
            with self.subTest(parser=name):
                self.assertIn("legacydefault",
                              parser._option_string_actions["--compaction-mode"].help)
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit):
                    parser.parse_args(["--compaction-mode", "default"])
                self.assertIn("unknown method 'default'", stderr.getvalue())
        # So do auto's config files and saves.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "roles.json"
            path.write_text(json.dumps({"version": 4, "main": {"compaction_mode": "default"}}))
            with self.assertRaisesRegex(ValueError, "unknown method 'default'"):
                resolve_config(path)

    def test_legacydefault_resolves_per_route_and_provider_is_checked(self):
        catalog = BUILTIN_MODEL_CATALOG
        official = catalog.bind("codex", "codex-gpt-6-astra")
        claude = catalog.bind("messages", "claude-opus-5.5")
        local = catalog.bind("chat-completions", "local")
        for binding, value, resolved in (
            (official, "legacydefault:2,coldstart", "provider:2,coldstart"),
            (claude, "legacydefault:2,coldstart", "pi:2,coldstart"),
            (local, "legacydefault", "pi"),
            (official, "provider:2,pi,coldstart", "provider:2,pi,coldstart"),
            (claude, "provider,coldstart", "provider,coldstart"),
            # Unset: coldstart, then warmpi up to twice, on every route.
            (official, None, "coldstart,warmpi:2"),
            (claude, None, "coldstart,warmpi:2"),
            (local, None, "coldstart,warmpi:2"),
            (official, "coldstart,legacydefault:2", "coldstart,provider:2"),
        ):
            with self.subTest(binding=binding.selector, value=value):
                self.assertEqual(resolve_compaction_mode(binding, value), resolved)
        with self.assertRaisesRegex(ConfigError, "provider may only come first"):
            resolve_compaction_mode(claude, "pi,provider")
        with self.assertRaisesRegex(ConfigError, "no provider compaction"):
            resolve_compaction_mode(local, "pi,provider")


class ColdstartCompactorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def saved(self, context, path=None, cwd=None):
        """Save ``context`` at ``path`` (``run.jsonl`` by default); return its log."""
        path = self.root / "run.jsonl" if path is None else path
        path.parent.mkdir(parents=True, exist_ok=True)
        save_interaction_save(path, context)
        return CompactionLog(path, cwd)

    def test_the_pointer_names_the_log_its_summary_and_the_initial_and_latest_requests(self):
        context = InteractionContext(_history())
        log = self.saved(context, self.root / ".interaction" / "run.jsonl", self.root)
        result = ColdstartCompactor(log).compact(context, tools=TOOLS, instructions=" the plan ")
        self.assertEqual(result.protocol, "coldstart")
        self.assertEqual(result.usage, TokenUsage())
        self.assertEqual(result.request_attempts, 0)
        self.assertEqual(result.context_items()[-1].request_attempts, 0)
        prefix = result.items[0].prefix_items
        self.assertEqual(len(prefix), 2)
        self.assertEqual(prefix[0], Instructions("Agent prompt."))
        # It names no tool (tool names differ by route), and the session's
        # initial request comes before the latest one. The summary file is
        # new, so it is a template, which the model fills in first.
        self.assertEqual(_pointer(result), (
            "This conversation was compacted. Its log up to this point is saved in "
            "`.interaction/run.jsonl` (relative to your tools' working directory), "
            "lines 1\u201312; later lines continue from here. Each line is a JSON item "
            'with a "type", such as "message" ("role", "content"), "tool_call" '
            '("name", "arguments_json"), "tool_result" ("output") or "context_prefix" '
            "(an earlier compaction). A summary template is in "
            "`.interaction/run.summary.md`; it covers no lines yet.\n\n"
            "The initial user message is on line 3:\n"
            "<user-message>\nFix the bug.\n</user-message>\n\n"
            "The latest user message is on line 11:\n"
            "<user-message>\nNow the tests.\n</user-message>\n\n"
            "Read the summary, then the log's messages and anything else you need from it. "
            "Before you continue, fill in the summary to cover lines 1\u201312: keep exact "
            "paths, names and error messages, and change its first line to \"Covers lines "
            "1\u201312 of session.\" Then pick up where the log left off.\n\n"
            "Additional focus: the plan"))
        self.assertTrue(_pointer(result).startswith(COLDSTART_POINTER_PREFIX + " "))
        # Coldstart wrote the template next to the save, as private as the save.
        summary = self.root / ".interaction" / "run.summary.md"
        self.assertEqual(log.summary_path, summary)
        self.assertEqual(summary.read_text(encoding="utf-8"),
                         f"Covers no lines yet of session.\n\n{SECTIONS}")
        self.assertEqual(stat.S_IMODE(summary.stat().st_mode) & 0o077, 0)

    def test_the_summarys_first_line_says_what_it_covers_and_is_only_read(self):
        context = InteractionContext(_history())
        compactor = ColdstartCompactor(self.saved(context, cwd=self.root))
        context.extend(compactor.compact(context, tools=TOOLS).context_items())
        context.extend((Message("assistant", "Updated the summary."), ModelSampleBoundary()))
        self.saved(context)
        summary = self.root / "run.summary.md"
        line = "change its first line to \"Covers lines 1\u201316 of session.\" Then pick up"
        covers = ("A summary of lines 1\u201312 is in `run.summary.md`.",
                  "Read the summary, then the messages after line 12 and anything else you "
                  "need from the log. Before you continue, edit the summary in place to cover "
                  "lines 1\u201316: keep what still applies, add what is new, keep exact paths, "
                  f"names and error messages, and {line} where the log left off.")
        template = ("A summary template is in `run.summary.md`; it covers no lines yet.",
                    "Read the summary, then the log's messages and anything else you need "
                    "from it. Before you continue, fill in the summary to cover lines "
                    "1\u201316: keep exact paths, names and error messages, and "
                    f"{line} where the log left off.")
        unknown = ("A summary is in `run.summary.md`, but its first line doesn't say which "
                   "lines of this log it covers.",
                   "Read the summary, then the log's messages and anything else you need "
                   "from it. Before you continue, edit the summary in place to cover lines "
                   "1\u201316: keep what still applies, add what is new, keep exact paths, "
                   f"names and error messages, and {line} where the log left off.")
        for label, first, (about, last) in (
            ("as asked", "Covers lines 1\u201312 of session.", covers),
            ("a hyphen, spaces and backticks", "  Covers lines 1 - 12 of `session`", covers),
            ("a byte order mark and CRLF", "\ufeffCovers lines 1\u201412 of session.\r", covers),
            ("still the template", "Covers no lines yet of session.", template),
            ("more lines than the log", "Covers lines 1\u201399 of session.", unknown),
            ("no lines", "Covers lines 1\u20130 of session.", unknown),
            ("another first line", "# Summary", unknown),
            ("an empty file", None, unknown),
        ):
            with self.subTest(label):
                data = b"" if first is None else (first + "\n\n## Goal\nShip it.\n").encode()
                summary.write_bytes(data)
                pointer = _pointer(compactor.compact(context, tools=TOOLS))
                self.assertIn(f"(an earlier compaction). {about}\n\n", pointer)
                self.assertTrue(pointer.endswith(f"\n\n{last}"), pointer)
                self.assertEqual(summary.read_bytes(), data)  # only read, never written
        summary.write_bytes(b"\xff\xfeCovers lines 1\xe2\x80\x9312 of session.\n")  # not UTF-8
        self.assertIn(unknown[0], _pointer(compactor.compact(context, tools=TOOLS)))
        self.assertEqual(sorted(path.name for path in self.root.iterdir()),
                         ["run.jsonl", "run.summary.md"])

    def test_a_summary_from_another_session_is_moved_aside_for_a_new_template(self):
        context = InteractionContext(_history())
        summary = self.root / "run.summary.md"
        other = "Covers lines 1\u201340 of session_other.\n\n## Goal\nOld work.\n"
        summary.write_text(other, encoding="utf-8")
        older = self.root / "run.summary.md.0"
        older.write_text("An older copy.\n", encoding="utf-8")
        result = ColdstartCompactor(self.saved(context, cwd=self.root)).compact(context, tools=TOOLS)
        pointer = _pointer(result)
        self.assertIn("A summary template is in `run.summary.md`; it covers no lines yet.", pointer)
        self.assertEqual(summary.read_text(encoding="utf-8"),
                         f"Covers no lines yet of session.\n\n{SECTIONS}")
        self.assertEqual(older.read_text(encoding="utf-8"), "An older copy.\n")
        self.assertEqual((self.root / "run.summary.md.1").read_text(encoding="utf-8"), other)
        # The compaction records the move; the pointer leaves the old file out.
        self.assertEqual(result.recovery, ("summary_moved_aside",))
        self.assertEqual(result.context_items()[-1].recovery, ("summary_moved_aside",))
        self.assertNotIn("summary.md.1", pointer)

    def test_without_an_init_line_the_summary_names_no_session(self):
        context = InteractionContext(_history()[1:])
        compactor = ColdstartCompactor(self.saved(context, cwd=self.root))
        pointer = _pointer(compactor.compact(context, tools=TOOLS))
        summary = self.root / "run.summary.md"
        self.assertEqual(summary.read_text(encoding="utf-8"), f"Covers no lines yet.\n\n{SECTIONS}")
        self.assertIn("change its first line to \"Covers lines 1\u201311.\" Then pick up", pointer)
        summary.write_text("Covers lines 1\u20135.\n", encoding="utf-8")
        self.assertIn("A summary of lines 1\u20135 is in `run.summary.md`.",
                      _pointer(compactor.compact(context, tools=TOOLS)))
        # A summary that names a session belongs to another save at this path.
        summary.write_text("Covers lines 1\u20135 of session.\n", encoding="utf-8")
        self.assertIn("A summary template is in", _pointer(compactor.compact(context, tools=TOOLS)))
        self.assertEqual((self.root / "run.summary.md.0").read_text(encoding="utf-8"),
                         "Covers lines 1\u20135 of session.\n")

    def test_an_unusable_summary_fails_the_compaction_and_nothing_is_written(self):
        context = InteractionContext(_history())
        compactor = ColdstartCompactor(self.saved(context, cwd=self.root))
        summary = self.root / "run.summary.md"
        summary.mkdir()
        with self.assertRaisesRegex(CompactionLogError,
                                    r"^the summary run\.summary\.md is not a regular file$"):
            compactor.compact(context, tools=TOOLS)
        summary.rmdir()
        # A link is never written through, even one to a missing file.
        target = self.root / "elsewhere.md"
        summary.symlink_to(target)
        with self.assertRaisesRegex(
                CompactionLogError,
                r"^the summary run\.summary\.md is a symbolic link to a missing file$"):
            compactor.compact(context, tools=TOOLS)
        self.assertFalse(target.exists())
        summary.unlink()
        if os.getuid() != 0:  # root writes anyway
            locked = self.root / "locked"
            locked.mkdir()
            log = self.saved(context, locked / "run.jsonl", self.root)
            locked.chmod(0o500)
            self.addCleanup(locked.chmod, 0o700)
            with self.assertRaisesRegex(
                    CompactionLogError,
                    r"^the summary locked/run\.summary\.md can't be created: Permission denied$"):
                ColdstartCompactor(log).compact(context, tools=TOOLS)
        # Nothing to compact: no summary either.
        empty = InteractionContext(_history()[:2])
        with self.assertRaises(NothingToCompact):
            ColdstartCompactor(self.saved(empty, cwd=self.root)).compact(empty, tools=TOOLS)
        self.assertFalse(os.path.lexists(summary))

    def test_pointers_old_and_new_are_recognized_and_other_messages_are_not(self):
        for text, expected in (
            (f"{COLDSTART_POINTER_PREFIX} `run.jsonl`, lines 1\u20137; later lines continue.", True),
            # The opening before October 17.
            ("This conversation was compacted without a summary. Its log up to this point is "
             "saved in `run.jsonl`, lines 1\u20137.", True),
            ("This conversation was compacted. Please carry on.", False),
        ):
            with self.subTest(text):
                self.assertEqual(is_coldstart_pointer(Message("user", text)), expected)
        self.assertFalse(is_coldstart_pointer(Message("assistant", f"{COLDSTART_POINTER_PREFIX} x")))

    def test_a_single_request_is_the_user_message_and_summaries_are_never_quoted(self):
        summary = f"{COMPACTION_SUMMARY_PREFIX}## Goal\nOld work.{COMPACTION_SUMMARY_SUFFIX}"
        context = InteractionContext((
            Init("session"),
            Instructions("Agent prompt."),
            Message("user", summary),
            # A pointer from before October 17.
            Message("user", "This conversation was compacted without a summary. An older pointer."),
            Message("user", "Fix the bug."),
            UserInteractionBoundary(),
            Message("assistant", "Working."),
            ModelSampleBoundary(),
        ))
        pointer = _pointer(ColdstartCompactor(self.saved(context, cwd=self.root)).compact(
            context, tools=TOOLS))
        self.assertIn("lines 1\u20138; later lines continue from here.", pointer)
        self.assertIn("\n\nThe user message is on line 5:\n"
                      "<user-message>\nFix the bug.\n</user-message>\n\n", pointer)
        for text in ("initial user message", "latest user message", "Old work.",
                     "An older pointer."):
            self.assertNotIn(text, pointer)

    def test_a_save_outside_the_working_directory_is_named_by_its_absolute_path(self):
        context = InteractionContext(_history())
        outside = self.root / "saves" / "run.jsonl"
        self.saved(context, outside)
        for log in (CompactionLog(outside, self.root / "work"), CompactionLog(outside)):
            with self.subTest(log=log):
                pointer = _pointer(ColdstartCompactor(log).compact(context, tools=TOOLS))
                self.assertIn(f"saved in `{outside}`, lines 1\u201312;", pointer)
                self.assertIn(f"A summary template is in `{outside.with_suffix('.summary.md')}`;",
                              pointer)

    def test_it_keeps_responses_checkpoints_and_quotes_only_top_level_requests(self):
        checkpoint = OpaqueCompaction("encrypted", protocol="responses")
        context = InteractionContext((
            Init("session"),
            ContextPrefix((Instructions("Agent prompt."), checkpoint,
                           Message("user", "Inside the checkpoint."))),
            CompactionMetadata(TokenUsage(), "responses_compaction_v2"),
            Message("assistant", "Working."),
            ModelSampleBoundary(),
        ))
        result = ColdstartCompactor(self.saved(context, cwd=self.root)).compact(
            context, tools=TOOLS)
        prefix = result.items[0].prefix_items
        self.assertEqual(prefix[:2], (Instructions("Agent prompt."), checkpoint))
        self.assertNotIn("<user-message>", _pointer(result))  # neither quote

    def test_long_requests_are_cut_and_nothing_new_is_nothing_to_compact(self):
        history = _history()
        context = InteractionContext((*history[:2], Message("user", "y" * 2_100), *history[3:-2],
                                      Message("user", "x" * 2_500), UserInteractionBoundary()))
        compactor = ColdstartCompactor(self.saved(context, cwd=self.root))
        result = compactor.compact(context, tools=TOOLS)
        quotes = ("The initial user message is on line 3:\n<user-message>\n" + "y" * 2_000
                  + "\n[100 more characters omitted]\n</user-message>",
                  "The latest user message is on line 11:\n<user-message>\n" + "x" * 2_000
                  + "\n[500 more characters omitted]\n</user-message>")
        for quote in quotes:
            self.assertIn(quote, _pointer(result))
        context.extend(result.context_items())
        self.saved(context)
        with self.assertRaisesRegex(NothingToCompact, "beyond a coldstart pointer"):
            compactor.compact(context, tools=TOOLS)
        context.extend((Message("assistant", "Reading the log."), ModelSampleBoundary()))
        self.saved(context)
        again = compactor.compact(context, tools=TOOLS)
        self.assertEqual(again.protocol, "coldstart")
        # The session's initial request stays the initial one: the earlier pointer,
        # inside its checkpoint, is never quoted.
        for quote in quotes:
            self.assertIn(quote, _pointer(again))

    def test_it_fails_unless_the_save_holds_the_context(self):
        context = InteractionContext(_history())
        path = self.root / "run.jsonl"
        compactor = ColdstartCompactor(CompactionLog(path, self.root))
        with self.assertRaisesRegex(CompactionLogError, r"^the save run\.jsonl was not found$"):
            compactor.compact(context, tools=TOOLS)
        self.saved(context)
        good = path.read_bytes()
        lines = good.splitlines(keepends=True)
        for label, data, message in (
            ("an invalid line", good.replace(b'"Fix the bug."', b'"Fix the bug.'),
             r"^the save run\.jsonl can't be loaded: invalid JSON in save .* at line 3: "),
            ("an invalid item", good.replace(b'"type": "tool_call"', b'"type": "tool_cal"'),
             r"^the save run\.jsonl can't be loaded: invalid save item in .* at line 5: "),
            ("fewer lines", b"".join(lines[:-1]),
             r"^the save run\.jsonl has 11 lines, fewer than the context's 12 items$"),
            ("a different item", good.replace(b"Fix the bug.", b"Fix the bus."),
             r"^line 3 of the save run\.jsonl differs from the context$"),
        ):
            with self.subTest(label):
                self.assertNotEqual(data, good)
                path.write_bytes(data)
                with self.assertRaisesRegex(CompactionLogError, message):
                    compactor.compact(context, tools=TOOLS)
        # A save it can't use leaves no summary behind.
        self.assertFalse((self.root / "run.summary.md").exists())
        # Later lines may follow, as /compact saves its own call first, and an
        # incomplete last line (an interrupted append) is ignored.
        later = InteractionContext((*context.items, Message("assistant", "Later.")))
        save_interaction_save(path, later)
        with path.open("ab") as file:
            file.write(b'{"type": "mess')
        result = compactor.compact(context, tools=TOOLS)
        self.assertIn("lines 1\u201312;", _pointer(result))
        self.assertTrue(issubclass(CompactionLogError, CompactionError))

    def test_it_needs_a_save_and_exec_command(self):
        context = InteractionContext(_history())
        with self.assertRaisesRegex(CompactionError, NO_EXEC):
            ColdstartCompactor(CompactionLog(self.root / "run.jsonl")).compact(context, tools=())
        with self.assertRaisesRegex(CompactionError, "needs a save to point to"):
            create_default_compactor(_Model(), CompactionSettings(mode="coldstart"))
        compactor = create_default_compactor(
            _Model(), CompactionSettings(mode="coldstart"),
            log=CompactionLog(self.root / "run.jsonl"))
        self.assertIsInstance(compactor, ColdstartCompactor)

    def test_the_stated_lines_of_the_save_are_the_compacted_items(self):
        path = self.root / "run.jsonl"
        original = InteractionContext(_history())
        InteractionSaveWriter(path).save(original)
        with path.open("ab") as file:  # an interrupted append
            file.write(b'{"type": "mess')
        context, writer, incomplete = resume_interaction_save(path)
        self.assertIsNotNone(incomplete)
        result = ColdstartCompactor(CompactionLog(path, self.root)).compact(context, tools=TOOLS)
        self.assertIn(f"lines 1\u2013{len(original)};", _pointer(result))
        context.extend(result.context_items())
        writer.save(context)
        lines = path.read_bytes().splitlines()
        self.assertEqual(
            tuple(interaction_item_from_dict(json.loads(line)) for line in lines[:len(original)]),
            original.items)
        self.assertEqual(json.loads(lines[-1])["request_attempts"], 0)
        saved = load_interaction_save(path)
        self.assertEqual(saved.items, context.items)
        self.assertEqual(saved[-1].request_attempts, 0)


class RunCompactionTests(unittest.TestCase):
    def run_sequence(self, mode, script, *, model=None, reports=None, **kwargs):
        context = InteractionContext(_history())
        log = CompactionLog(Path("run.jsonl"))
        return run_compaction(
            context, model or _Model(), CompactionSettings(mode=mode), factory=script,
            tools=TOOLS, log=log,
            on_failure=None if reports is None else reports.append, **kwargs)

    def test_attempts_are_counted_per_method_and_each_failure_is_reported(self):
        script = _Script(_timeout(), _timeout(), _result("coldstart"))
        reports = []
        result = self.run_sequence("pi:2,coldstart", script, reports=reports)
        self.assertEqual(script.methods, ["pi", "pi", "coldstart"])
        self.assertEqual([options for _mode, options, _kwargs in script.calls],
                         [{}, {}, {"log": CompactionLog(Path("run.jsonl"))}])
        self.assertEqual(result.protocol, "coldstart")
        self.assertEqual(result.recovery, ("pi_failed", "pi_failed"))
        self.assertEqual([report.notice() for report in reports], [
            "pi compaction failed (ModelTransportError: timed out); trying pi again "
            "(attempt 2 of 2).",
            "pi compaction failed (ModelTransportError: timed out); trying coldstart.",
        ])
        self.assertEqual(reports[0].failure, ModelFailure("transport", "timed out"))

    def test_a_request_too_large_or_a_configuration_error_moves_on_at_once(self):
        for error in (ModelConfigurationError("not representable by the text codec"),
                      CompactionContextWindowError("summary request too large"),
                      ModelContextWindowError("too long")):
            with self.subTest(error=type(error).__name__):
                script = _Script(error, _result("coldstart"))
                reports = []
                self.run_sequence("pi:3,coldstart", script, reports=reports)
                self.assertEqual(script.methods, ["pi", "coldstart"])
                self.assertFalse(reports[0].retry)
        self.assertEqual(reports[0].failure.category, "ModelContextWindowError")
        # Adapter messages may carry response bodies; the saved item doesn't.
        self.assertEqual(reports[0].failure.message, "details were withheld")

    def test_compaction_failures_retry_like_transport_failures(self):
        script = _Script(CompactionError("summary request for the history returned no text"),
                         _result())
        reports = []
        self.assertEqual(self.run_sequence("pi:2", script, reports=reports).recovery,
                         ("pi_failed",))
        self.assertEqual(reports[0].failure, ModelFailure(
            "CompactionError", "summary request for the history returned no text"))

    def test_a_log_error_moves_on_at_once(self):
        script = _Script(CompactionLogError("the save run.jsonl was not found"), _result())
        reports = []
        result = self.run_sequence("coldstart:3,pi:2", script, reports=reports)
        self.assertEqual(script.methods, ["coldstart", "pi"])
        self.assertEqual(result.recovery, ("coldstart_failed",))
        self.assertEqual([report.notice() for report in reports], [
            "coldstart compaction failed (CompactionLogError: the save run.jsonl was not "
            "found); trying pi."])
        self.assertEqual(reports[0].failure,
                         ModelFailure("CompactionLogError", "the save run.jsonl was not found"))

    def test_coldstart_is_skipped_where_it_cant_run_while_a_later_method_remains(self):
        for log, tools in ((None, TOOLS), (CompactionLog(Path("run.jsonl")), ())):
            with self.subTest(log=log, tools=tools):
                script = _Script(_result())
                reports = []
                result = run_compaction(
                    InteractionContext(_history()), _Model(),
                    CompactionSettings(mode="coldstart,pi:2"), factory=script, tools=tools,
                    log=log, on_failure=reports.append)
                self.assertEqual((script.methods, result.recovery, reports), (["pi"], (), []))
        # As the last method, it runs, and fails as before.
        with self.assertRaisesRegex(CompactionError, "needs a save to point to"):
            run_compaction(InteractionContext(_history()), _Model(),
                           CompactionSettings(mode="coldstart"), tools=TOOLS)

    def test_cancels_authentication_errors_bugs_and_stops_end_the_sequence(self):
        for error, stopping in ((ModelCancelled("cancelled"), False),
                                (ModelAuthenticationError("expired"), False),
                                (RuntimeError("bug"), False),
                                (_timeout(), True)):
            with self.subTest(error=type(error).__name__, stopping=stopping):
                script = _Script(error, _result("coldstart"))
                reports = []
                with self.assertRaises(type(error)):
                    self.run_sequence("pi:2,coldstart", script, reports=reports,
                                      stop_requested=lambda: stopping)
                self.assertEqual(script.methods, ["pi"])
                self.assertEqual(reports, [])

    def test_nothing_to_compact_moves_on_only_in_overflow_recovery(self):
        for reason in ("threshold", "manual"):
            with self.subTest(reason=reason):
                script = _Script(NothingToCompact("the context fits"), _result("coldstart"))
                with self.assertRaises(NothingToCompact):
                    self.run_sequence("pi,coldstart", script, reason=reason)
                self.assertEqual(script.methods, ["pi"])
        script = _Script(NothingToCompact("the context fits"), _result("coldstart"))
        result = self.run_sequence("pi,coldstart", script, reason="overflow")
        self.assertEqual((script.methods, result.recovery), (["pi", "coldstart"], ("pi_failed",)))
        with self.assertRaises(NothingToCompact):  # nothing later to try
            self.run_sequence("pi", _Script(NothingToCompact("fits")), reason="overflow")

    def test_every_attempt_failing_raises_the_last_failure(self):
        first, last = _timeout(), CompactionError("stopped early")
        reports = []
        with self.assertRaises(CompactionError) as raised:
            self.run_sequence("pi:2", _Script(first, last), reports=reports)
        self.assertIs(raised.exception, last)
        self.assertEqual(reports[-1].notice(),
                         "pi compaction failed (CompactionError: stopped early).")

    def test_legacydefault_and_focus_text_follow_the_model(self):
        remote = mock.Mock(supports_remote_compaction=True)
        script = _Script(_result("responses_compaction_v2"))
        self.run_sequence("legacydefault:2,coldstart", script, model=remote)
        self.assertEqual(script.methods, ["provider"])
        # Codex remote compaction takes no focus text: skipped while a later method remains.
        script = _Script(_result("coldstart"))
        self.run_sequence("provider,coldstart", script, model=remote, instructions="the bug")
        self.assertEqual(script.methods, ["coldstart"])
        self.assertEqual(script.calls[0][2]["instructions"], "the bug")
        script = _Script(CompactionError("remote Responses compaction does not accept focus text"))
        with self.assertRaisesRegex(CompactionError, "focus text"):
            self.run_sequence("provider", script, model=remote, instructions="the bug")
        script = _Script(_result())
        self.run_sequence("legacydefault", script, model=_Model())
        self.assertEqual(script.methods, ["pi"])
        self.assertNotIn("instructions", script.calls[0][2])


class _Host(kernel.TurnHost):
    def __init__(self, path, context):
        self.path, self.notices, self.shown = path, [], []
        self.writer = InteractionSaveWriter(path)
        self.writer.save(context)

    def append(self, context, items):
        context.extend(items)
        self.writer.save(context)

    def show(self, items):
        self.shown.extend(items)

    def notice(self, text):
        self.notices.append(text)

    def log_path(self):
        return self.path


class TurnLoopTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "run.jsonl"

    def turn(self, model, **config):
        context = InteractionContext(_history())
        self.host = _Host(self.path, context)
        result = kernel.run_turn(context, model, Environment((_exec_tool(),)),
                                 InteractionConfigSnapshot(**config), self.host)
        return result, load_interaction_save(self.path)

    def test_a_threshold_coldstart_makes_no_model_request(self):
        model = _Model(_answer("done"))
        result, saved = self.turn(model, auto_compaction_context_tokens=100, compaction_mode="coldstart")
        self.assertEqual(result.final_text, "done")
        [request] = model.calls
        visible = request.model_items()
        self.assertEqual(visible[0], Instructions("Agent prompt."))
        self.assertTrue(is_coldstart_pointer(visible[1]))
        self.assertEqual(len(visible), 2)
        self.assertIn("lines 1\u201312;", visible[1].content)
        # Without a working directory, both files are named by absolute path.
        summary = self.path.with_suffix(".summary.md")
        self.assertIn(f"A summary template is in `{summary}`; it covers no lines yet.",
                      visible[1].content)
        self.assertTrue(summary.read_text(encoding="utf-8").startswith(
            "Covers no lines yet of session.\n\n## Goal\n"))
        # The turn shows the pointer, as it shows a pi summary.
        self.assertIn(DisplayItem(visible[1].content), self.host.shown)
        [metadata] = [item for item in saved if isinstance(item, CompactionMetadata)]
        self.assertEqual((metadata.protocol, metadata.request_attempts), ("coldstart", 0))

    def test_overflow_recovery_with_coldstart_fits_the_retry(self):
        overflow = ModelContextWindowError(
            "too long", failure=ModelFailure("context_window", "context window exceeded"))
        model = _Model(overflow, _answer("fits"))
        result, _saved = self.turn(model, compaction_mode="coldstart")
        self.assertEqual(result.final_text, "fits")
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(is_coldstart_pointer(model.calls[1].model_items()[1]))
        self.assertIn(kernel.OVERFLOW_NOTICE, self.host.notices)

    def test_a_failed_pi_compaction_falls_back_and_is_saved(self):
        model = _Model(_timeout(), _answer("done"))  # the summary request, then the sample
        result, saved = self.turn(model, auto_compaction_context_tokens=100, compaction_mode="pi,coldstart",
                                  compaction_pi_keep_recent_tokens=0)
        self.assertEqual(result.final_text, "done")
        self.assertIn("pi compaction failed (ModelTransportError: timed out); trying coldstart.",
                      self.host.notices)
        items = saved.items
        failure = next(i for i, item in enumerate(items) if isinstance(item, ModelFailure))
        prefix = next(i for i, item in enumerate(items) if isinstance(item, ContextPrefix))
        self.assertEqual(items[failure], ModelFailure("transport", "timed out"))
        self.assertEqual(prefix, failure + 1)
        # The pointer's lines include the saved failure.
        self.assertIn(f"lines 1\u2013{prefix};", items[prefix].prefix_items[-1].content)
        metadata = items[prefix + 1]
        self.assertEqual((metadata.protocol, metadata.recovery), ("coldstart", ("pi_failed",)))
        self.assertTrue(is_coldstart_pointer(model.calls[1].model_items()[1]))

    def test_a_coldstart_that_cant_use_its_save_falls_back_and_is_saved(self):
        model = _Model(*(_answer("done") for _ in range(3)))  # pi's requests, then the sample
        context = InteractionContext(_history())
        self.host = _Host(self.path, context)
        missing = self.path.with_name("moved.jsonl")
        self.host.log_path = lambda: missing  # it names a save that isn't there
        result = kernel.run_turn(
            context, model, Environment((_exec_tool(),)), InteractionConfigSnapshot(
                auto_compaction_context_tokens=100, compaction_mode="coldstart,pi:2",
                compaction_pi_keep_recent_tokens=0), self.host)
        self.assertEqual(result.final_text, "done")
        message = f"the save {missing} was not found"
        self.assertIn(f"coldstart compaction failed (CompactionLogError: {message}); trying pi.",
                      self.host.notices)
        items = load_interaction_save(self.path).items
        [failure] = [item for item in items if isinstance(item, ModelFailure)]
        self.assertEqual(failure, ModelFailure("CompactionLogError", message))
        [metadata] = [item for item in items if isinstance(item, CompactionMetadata)]
        self.assertEqual((metadata.protocol, metadata.recovery), ("pi", ("coldstart_failed",)))

    def test_a_summary_it_cant_create_falls_back_with_a_notice(self):
        model = _Model(*(_answer("done") for _ in range(3)))  # pi's requests, then the sample
        context = InteractionContext(_history())
        self.host = _Host(self.path, context)
        summary = self.path.with_suffix(".summary.md")
        summary.mkdir()  # something other than a file at the summary's path
        result = kernel.run_turn(
            context, model, Environment((_exec_tool(),)), InteractionConfigSnapshot(
                auto_compaction_context_tokens=100, compaction_mode="coldstart,pi",
                compaction_pi_keep_recent_tokens=0), self.host)
        self.assertEqual(result.final_text, "done")
        message = f"the summary {summary} is not a regular file"
        self.assertIn(f"coldstart compaction failed (CompactionLogError: {message}); trying pi.",
                      self.host.notices)
        items = load_interaction_save(self.path).items
        self.assertIn(ModelFailure("CompactionLogError", message), items)
        [metadata] = [item for item in items if isinstance(item, CompactionMetadata)]
        self.assertEqual((metadata.protocol, metadata.recovery), ("pi", ("coldstart_failed",)))
        self.assertTrue(summary.is_dir())

    def test_overflow_recovery_moves_on_when_pi_has_nothing_to_compact(self):
        overflow = ModelContextWindowError("too long")
        model = _Model(overflow, _answer("fits"))
        result, saved = self.turn(model, compaction_mode="pi,coldstart")  # all of it fits the tail
        self.assertEqual(result.final_text, "fits")
        self.assertEqual(len(model.calls), 2)  # no summary request
        [metadata] = [item for item in saved if isinstance(item, CompactionMetadata)]
        self.assertEqual(metadata.recovery, ("pi_failed",))


class LiveConfigTests(unittest.TestCase):
    def config(self, *argv, reason=None):
        return InteractionConfig.from_namespace(cli._build_parser().parse_args(argv),
                                                coldstart_unavailable=reason)

    def test_compaction_mode_changes_are_checked_against_the_route(self):
        codex = self.config("--endpoint-api", "codex", "--model", "codex-gpt-6-astra")
        self.assertEqual(codex.get("compaction_mode"), "coldstart,warmpi:2")
        self.assertEqual(codex.set("compaction_mode", "legacydefault:2,coldstart"),
                         "provider:2,coldstart")
        self.assertEqual(codex.set("compaction_mode", None), "coldstart,warmpi:2")
        local = self.config("--endpoint-api", "chat-completions")
        with self.assertRaisesRegex(ConfigError, "no provider compaction"):
            local.set("compaction_mode", "pi,provider")
        messages = ("--endpoint-api", "messages", "--model", "claude-fable-5.1")
        pi = self.config(*messages)
        with self.assertRaisesRegex(ConfigError, "can't add or remove provider"):
            pi.set("compaction_mode", "provider,coldstart")
        server = self.config(*messages, "--compaction-mode", "provider")
        with self.assertRaisesRegex(ConfigError, "can't add or remove provider"):
            server.set("compaction_mode", "pi")
        self.assertEqual(server.set("compaction_mode", "legacydefault:2,coldstart"),
                         "provider:2,coldstart")
        self.assertEqual(server.set("compaction_mode", None), "provider")
        self.assertEqual(parse_config_literal("compaction_mode", "legacydefault:2,coldstart"),
                         "legacydefault:2,coldstart")

    def test_a_session_that_cant_run_coldstart_drops_it_at_launch_and_refuses_it_later(self):
        config = self.config("--endpoint-api", "messages", "--model", "claude-fable-5.1",
                             "--compaction-mode", "legacydefault:2,coldstart", reason="no exec_command")
        self.assertEqual(config.get("compaction_mode"), "pi:2")
        self.assertEqual(config.coldstart_dropped, ("pi:2,coldstart", "pi:2"))
        self.assertEqual(config.initial_values("compaction_mode"), {"compaction_mode": "pi:2"})
        with self.assertRaisesRegex(
                ConfigError, r"^compaction_mode can't include coldstart in this session: "
                             r"no exec_command\. It stays pi:2\.$"):
            config.set("compaction_mode", "coldstart")
        self.assertEqual(config.get("compaction_mode"), "pi:2")
        # null gives the default without coldstart.
        self.assertEqual(config.set("compaction_mode", None), "warmpi:2")
        self.assertEqual(config.reset("compaction_mode").compaction_mode, "pi:2")
        # coldstart alone becomes the default without coldstart.
        for argv in (("--endpoint-api", "messages", "--model", "claude-fable-5.1"),
                     ("--endpoint-api", "codex", "--model", "codex-gpt-6-astra")):
            with self.subTest(argv=argv):
                alone = self.config(*argv, "--compaction-mode", "coldstart", reason="r")
                self.assertEqual(alone.get("compaction_mode"), "warmpi:2")
                self.assertEqual(alone.coldstart_dropped, ("coldstart", "warmpi:2"))
        unchanged = self.config("--compaction-mode", "pi:2", reason="r")
        self.assertIsNone(unchanged.coldstart_dropped)

    def test_unset_it_is_coldstart_then_warmpi_twice_on_every_route(self):
        messages = ("--endpoint-api", "messages", "--model", "claude-fable-5.1")
        # The official Codex route too, which defaulted to remote compaction.
        for argv in (messages, ("--endpoint-api", "codex", "--model", "codex-gpt-6-astra"),
                     ("--endpoint-api", "chat-completions")):
            with self.subTest(argv=argv):
                config = self.config(*argv)
                self.assertEqual(config.get("compaction_mode"), "coldstart,warmpi:2")
                self.assertEqual(config.set("compaction_mode", "pi"), "pi")
                self.assertEqual(config.set("compaction_mode", None), "coldstart,warmpi:2")
                # A session that can't run coldstart loses it from the default
                # silently, and from a value equal to the default too.
                for launch in ((), ("--compaction-mode", "coldstart,warmpi:2")):
                    dropped = self.config(*argv, *launch, reason="r")
                    self.assertEqual(dropped.get("compaction_mode"), "warmpi:2")
                    self.assertIsNone(dropped.coldstart_dropped)
                    self.assertEqual(dropped.set("compaction_mode", None), "warmpi:2")
        # Server-side compaction runs inside samples, and provider may only
        # come first there: its default stays provider.
        server = self.config(*messages, "--compaction-mode", "provider")
        self.assertEqual(server.set("compaction_mode", None), "provider")
        # Without a route (a bare config), the default stays pi.
        self.assertEqual(InteractionConfig().get("compaction_mode"), "pi")
        self.assertEqual(InteractionConfigSnapshot().compaction_settings().mode, "pi")


class _Terminal:
    def __init__(self, commands):
        self.commands = deque(commands)
        self.keys = deque()
        self.items = []
        self.closed = False
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.exited = True

    def read_keys(self):
        keys = tuple(self.keys)
        self.keys.clear()
        return keys

    def render(self, editor, status, items, prompt=":> "):
        self.items.extend(items)
        if status == "idle" and self.commands and not self.keys:
            from types import SimpleNamespace
            for key, data in (("c-u", ""), ("c-k", ""),
                              ("<bracketed-paste>", self.commands.popleft()), ("c-m", "\r")):
                self.keys.append(SimpleNamespace(key=key, data=data))


class CLITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "interaction.jsonl"
        save_interaction_save(self.path, InteractionContext((
            Init("old"), Message("user", "Old request."), UserInteractionBoundary(),
            Message("assistant", "Old answer."),
            SampleMetadata(TokenUsage(input_tokens=900, total_tokens=1000)),
            ModelSampleBoundary(), TurnSummary(sample_count=1, context_tokens=1000),
        )))

    async def run_cli(self, model, commands, argv, environment):
        terminal = _Terminal((*commands, "/quit"))
        args = cli._build_parser().parse_args(["--resume", *argv])
        await asyncio.wait_for(cli._run(model, environment, terminal, args, self.path),
                               timeout=4)
        self.assertTrue(terminal.exited)
        return [item.text for item in terminal.items], load_interaction_save(self.path)

    def user_results(self, saved):
        return [item.result for item in saved if isinstance(item, UserToolResult)]

    async def test_compact_with_coldstart_points_to_the_save(self):
        model = _Model()
        with DefaultEnvironment(cwd=self.path.parent) as environment:
            _texts, saved = await self.run_cli(
                model, ["/compact"], ["--compaction-mode", "coldstart"], environment)
        self.assertEqual(model.calls, [])
        call = next(i for i, item in enumerate(saved) if isinstance(item, UserToolCall))
        [result] = self.user_results(saved)
        self.assertEqual(result.output, (
            f"Context compacted using a coldstart pointer to the log (lines 1\u2013{call}); "
            "no model request was made."))
        prefix = next(item for item in saved if isinstance(item, ContextPrefix))
        self.assertIn("saved in `interaction.jsonl` (relative to your tools' working directory)",
                      prefix.prefix_items[-1].content)
        # The summary file sits next to the save and names its session.
        self.assertIn("A summary template is in `interaction.summary.md`; it covers no lines yet.",
                      prefix.prefix_items[-1].content)
        self.assertTrue(self.path.with_suffix(".summary.md").read_text(encoding="utf-8").startswith(
            "Covers no lines yet of old.\n\n## Goal\n"))

    async def test_compact_with_the_default_compacts_cold(self):
        model = _Model()
        with DefaultEnvironment(cwd=self.path.parent) as environment:
            _texts, saved = await self.run_cli(model, ["/compact"], [], environment)
        self.assertEqual(model.calls, [])
        [result] = self.user_results(saved)
        self.assertTrue(result.output.startswith("Context compacted using a coldstart pointer"),
                        result.output)

    async def test_a_session_without_exec_command_drops_coldstart_and_refuses_it(self):
        texts, saved = await self.run_cli(
            _Model(), ["/config compaction_mode coldstart", "/config.json compaction_mode"],
            ["--enable-default-tools=False", "--compaction-mode", "legacydefault:2,coldstart"],
            Environment())
        self.assertIn(
            "[cli] Note: this session compacts with pi:2, not compaction_mode "
            "legacydefault:2,coldstart (--compaction-mode): coldstart compaction needs exec_command "
            "to read the saved log, and default model tools are disabled.", texts)
        refused, shown = self.user_results(saved)
        self.assertFalse(refused.success)
        self.assertEqual(refused.output, (
            "compaction_mode can't include coldstart in this session: coldstart compaction "
            "needs exec_command to read the saved log, and default model tools are disabled. "
            "It stays pi:2."))
        self.assertEqual(json.loads(shown.output),
                         {"__init__": {"compaction_mode": "pi:2"}, "compaction_mode": "pi:2"})

    async def test_a_config_change_applies_to_the_next_turn(self):
        model = _Model(_answer("next answer"))
        with DefaultEnvironment(cwd=self.path.parent) as environment:
            _texts, saved = await self.run_cli(
                model, ["/config compaction_mode coldstart", "next"],
                ["--auto-compaction-context-tokens", "100"], environment)
        [request] = model.calls
        # The compaction at the turn's start replaced the new request too;
        # the pointer quotes it.
        [pointer] = request.model_items()
        self.assertTrue(is_coldstart_pointer(pointer))
        self.assertIn("<user-message>\nnext\n</user-message>", pointer.content)
        metadata = next(item for item in saved if isinstance(item, CompactionMetadata))
        self.assertEqual(metadata.protocol, "coldstart")

    async def test_compact_lists_failed_attempts_in_its_result_only(self):
        model = _Model(_timeout())
        with DefaultEnvironment(cwd=self.path.parent) as environment:
            _texts, saved = await self.run_cli(
                model, ["/compact"], ["--compaction-mode", "pi,coldstart",
                                      "--compaction-pi-keep-recent-tokens", "0"], environment)
        self.assertEqual(len(model.calls), 1)  # pi's summary request
        call = next(i for i, item in enumerate(saved) if isinstance(item, UserToolCall))
        [result] = self.user_results(saved)
        self.assertTrue(result.success)
        self.assertEqual(result.output, (
            "pi compaction failed (ModelTransportError: timed out); trying coldstart.\n"
            f"Context compacted using a coldstart pointer to the log (lines 1\u2013{call}); "
            "no model request was made."))
        self.assertFalse(any(isinstance(item, ModelFailure) for item in saved))
        # The finished turn still looks finished: /continue stays refused.
        self.assertEqual(turn_tail(saved.items).refusal, "the last turn ended")

    async def test_a_summary_it_cant_create_is_shown_and_the_next_method_runs(self):
        self.path.with_suffix(".summary.md").mkdir()
        model = _Model(_answer("Summary."), _answer("next answer"), _answer("Summary again."))
        with DefaultEnvironment(cwd=self.path.parent) as environment:
            texts, saved = await self.run_cli(
                model, ["next", "/compact"],
                ["--compaction-mode", "coldstart,pi", "--compaction-pi-keep-recent-tokens", "0",
                 "--auto-compaction-context-tokens", "100"], environment)
        self.assertEqual(len(model.calls), 3)
        failed = ("coldstart compaction failed (CompactionLogError: the summary "
                  "interaction.summary.md is not a regular file); trying pi.")
        # At the turn's start, a notice; after /compact, its result.
        self.assertIn(f"[cli] {failed}", texts)
        [result] = self.user_results(saved)
        self.assertEqual(result.output, f"{failed}\nContext compacted using a pi summary checkpoint.")

    def test_the_result_text_follows_the_protocol(self):
        remote = OpaqueCompaction("encrypted", protocol="responses")
        pointer = Message("user", f"{COLDSTART_POINTER_PREFIX} ...")
        for result, text in (
            (_result("coldstart", remote, pointer),
             "Context compacted using a coldstart pointer to the log (lines 1\u20137); "
             "no model request was made."),
            (_result("pi", remote, Message("user", "summary")),
             "Context compacted using a pi summary checkpoint."),
            (_result("responses_compaction_v2", remote),
             "Context compacted using a remote opaque checkpoint."),
        ):
            with self.subTest(protocol=result.protocol):
                self.assertEqual(cli._compaction_success_output(result, log_lines=7), text)


class AutoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)

    def start(self, settings, **options):
        session = auto._Session(self.temp / "run", settings,
                                model_factory=lambda index, args: mock.Mock(),
                                **options).start()
        self.addCleanup(session.close)
        texts = [item.text for event in session.drain_events() for item in event.items]
        session.close()
        return texts

    def notices(self, texts):
        return [text for text in texts if "compaction_mode" in text]

    def test_roles_without_exec_command_drop_coldstart_with_a_notice(self):
        settings = resolve_config(overrides={"cwd": str(self.temp),
                                             "compaction_mode": "legacydefault:2,coldstart"},
                                  roles=(1, -1))
        # The watcher on main's model takes main's option too.
        self.assertEqual(settings.mode_sources, {1: "command line", -1: "command line"})
        texts = self.start(settings, enable_default_tools=False)
        # The watcher, on main's value, drops coldstart silently.
        self.assertEqual(self.notices(texts), [
            "Note: #1 (main) compacts with pi:2, not compaction_mode legacydefault:2,coldstart "
            "(command line): coldstart compaction needs exec_command to read the saved log, "
            "and default model tools are disabled."])
        saved = json.loads((self.temp / "run" / "config.json").read_text())
        self.assertEqual(saved["main"]["compaction_mode"], "legacydefault:2,coldstart")
        self.assertEqual(self.notices(self.start(resolve_config(
            overrides={"cwd": str(self.temp)}, saved={1: saved["main"], -1: saved["watcher"]},
            roles=(1, -1)), resume=True)), [])

    def test_the_watchers_own_entry_gets_the_notice(self):
        path = self.temp / "auto.json"
        path.write_text(json.dumps({"version": 3, "watcher": {"compaction_mode": "coldstart"}}))
        settings = resolve_config(path, overrides={"cwd": str(self.temp)}, roles=(1, -1))
        self.assertEqual(settings.mode_sources, {1: None, -1: "config file"})
        note = ("Note: #-1 (watcher) compacts with warmpi:2, not compaction_mode coldstart ({}): "
                "coldstart compaction needs exec_command to read the saved log, and the "
                "watcher's only tools are read_context and resume.")
        self.assertEqual(self.notices(self.start(settings)), [note.format("config file")])
        # Resumed without the file, the value is the watcher's own saved entry.
        saved = json.loads((self.temp / "run" / "config.json").read_text())
        self.assertEqual(saved["watcher"]["compaction_mode"], "coldstart")
        resumed = resolve_config(overrides={"cwd": str(self.temp)},
                                 saved={1: saved["main"], -1: saved["watcher"]}, roles=(1, -1))
        self.assertEqual(self.notices(self.start(resumed, resume=True)), [note.format("saved")])

    def test_print_config_shows_each_roles_sequence(self):
        for flags, main, watcher in (
            (["--compaction-mode", "coldstart"], "compaction coldstart",
             "compaction warmpi:2 (coldstart dropped)"),
            (["--compaction-mode", "coldstart", "--enable-default-tools=False"],
             "compaction warmpi:2 (coldstart dropped)", "compaction warmpi:2 (coldstart dropped)"),
            # The default loses coldstart silently.
            ([], "compaction coldstart,warmpi:2", "compaction warmpi:2"),
        ):
            with self.subTest(flags=flags):
                stdout = io.StringIO()
                with (mock.patch.object(auto, "_Session") as session, redirect_stdout(stdout),
                      redirect_stderr(io.StringIO())):
                    code = auto.main([
                        "--no-user-model-catalog", "--model", "big",
                        "--endpoint-url", "http://gpu:8000/v1/chat/completions",
                        "--save", str(self.temp / "x"),
                        "--print-config", *flags])
                self.assertEqual(code, 0)
                session.assert_not_called()
                lines = stdout.getvalue().splitlines()
                self.assertTrue(lines[0].endswith("; " + main), lines[0])
                self.assertTrue(lines[1].endswith("; " + watcher), lines[1])


class DemoTests(unittest.TestCase):
    def test_run_drops_coldstart_without_a_save_or_exec_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            for environment, save_path, reason in (
                (Environment((_exec_tool(),)), None,
                 "coldstart compaction needs a save to point to, and save_path is None"),
                (Environment(), Path(temporary) / "run.jsonl", NO_EXEC),
            ):
                with self.subTest(reason=reason):
                    stderr = io.StringIO()
                    with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                        answer = demo.run(_Model(_answer("hi")), environment, prompt="hello",
                                          save_path=save_path, compaction_mode="coldstart")
                    self.assertEqual(answer, "hi")
                    self.assertIn(f"Note: compacting with warmpi:2, not compaction_mode coldstart: "
                                  f"{reason}.", stderr.getvalue())


class MessagesOverflowTests(unittest.TestCase):
    def test_prompt_is_too_long_is_a_context_window_error(self):
        detail = ('{"type":"error","error":{"type":"invalid_request_error",'
                  '"message":"prompt is too long: 1000123 tokens > 1000000 maximum"}}')
        error = _messages_http_failure(400, detail, model="claude-opus-5-5",
                                       attempt_count=1, recovery=())
        self.assertIsInstance(error, ModelContextWindowError)
        self.assertEqual(error.failure.category, "context_window")
        # A summary request that gets it fails as too large, so a sequence moves on.
        model = _Model(error)
        with self.assertRaises(CompactionContextWindowError):
            PiCompactor(model, keep_recent_tokens=0).compact(InteractionContext(_history()))


if __name__ == "__main__":
    unittest.main()
