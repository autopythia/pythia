"""Renamed settings: three in part 5 of the warm-cold compaction plan, then
request_timeout_seconds, which became timeouts_request_seconds.

The new names work everywhere. The old ones keep working with a warning that
names the new one: as options, as /config keys, in auto's config files, and as
demo.run keywords (the three that are). Auto saves are version 5, and
version-3 and version-4 saves load silently.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock
import warnings

from pythia.interaction import Environment, InteractionConfig, UserToolCall, load_interaction_save
from pythia.interaction import cli, demo, user_tools
from pythia.interaction._auto_config import (
    CONFIG_VERSION, build_parser as build_auto_parser, load_config_file, load_saved_config,
    resolve_config, saved_document,
)
from pythia.interaction.runtime_config import CONFIG_KEYS, RENAMED_CONFIG_KEYS
from pythia_test.test_interaction_cli import _Model, _Terminal, _answer


RENAMES = {
    "auto_compact_tokens": "auto_compaction_context_tokens",
    "compaction_keep_recent_tokens": "compaction_pi_keep_recent_tokens",
    "compaction_max_output_tokens": "compaction_pi_max_output_tokens",
}
# Named after the model catalog's timeouts.request_seconds; not a demo.run keyword.
TIMEOUT_RENAME = {"request_timeout_seconds": "timeouts_request_seconds"}
ALL_RENAMES = {**RENAMES, **TIMEOUT_RENAME}
OLD_NAMES = {new: old for old, new in ALL_RENAMES.items()}


def _option(key):
    return "--" + key.replace("_", "-")


@contextmanager
def _caught():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


class OptionAndConfigKeyTests(unittest.TestCase):
    def test_one_mapping_names_every_rename(self):
        self.assertEqual(RENAMED_CONFIG_KEYS, ALL_RENAMES)
        self.assertLessEqual(set(ALL_RENAMES.values()), set(CONFIG_KEYS))
        self.assertFalse(set(ALL_RENAMES) & set(CONFIG_KEYS))

    def test_old_options_are_hidden_aliases_that_warn(self):
        for frontend, parser, unset in (("cli", cli._build_parser(), None),
                                        ("demo", demo._build_parser(), None),
                                        ("auto", build_auto_parser(), "absent")):
            usage = parser.format_help()
            defaults = vars(parser.parse_args([]))
            for old, new in ALL_RENAMES.items():
                with self.subTest(frontend=frontend, old=old):
                    self.assertIn(_option(new), usage)
                    self.assertNotIn(_option(old), usage)
                    # Unused, an alias leaves the new option's default.
                    self.assertEqual(defaults.get(new, "absent"), unset)
                    self.assertNotIn(old, defaults)
                    with _caught() as caught:
                        self.assertEqual(getattr(parser.parse_args([_option(new), "7"]), new), 7)
                    self.assertEqual(caught, [])
                    with self.assertWarnsRegex(FutureWarning, "^" + re.escape(
                            f"{_option(old)} is deprecated; use {_option(new)}.") + "$"):
                        args = parser.parse_args([_option(old), "7"])
                    self.assertEqual(getattr(args, new), 7)
                    self.assertNotIn(old, vars(args))
        # The timeout's old option takes decimal seconds, as its new one does.
        with self.assertWarns(FutureWarning):
            args = cli._build_parser().parse_args(["--request-timeout-seconds", "7.5"])
        self.assertEqual(args.timeouts_request_seconds, 7.5)

    def test_old_config_keys_parse_as_the_new_ones_with_a_notice(self):
        for old, new in ALL_RENAMES.items():
            for command in ("/config {}", "/config {} 5", "/config.json {}", "/config.json {} null"):
                with self.subTest(command=command.format(old)):
                    intent = user_tools.parse_user_tool(command.format(old))
                    current = user_tools.parse_user_tool(command.format(new))
                    self.assertEqual(intent, current)
                    self.assertEqual(intent.arguments_json, current.arguments_json)
                    self.assertEqual(json.loads(intent.arguments_json)["key"], new)
                    self.assertEqual(intent.notice, f"Note: {old} is deprecated; use {new}.")
                    self.assertIsNone(current.notice)
        self.assertIsNone(user_tools.parse_user_tool("/config").notice)
        self.assertIsNone(user_tools.parse_user_tool("/compact").notice)
        # Old names are not valid values elsewhere, and bad values still fail.
        for command in ("/config auto_compact_tokens -1", "/config compaction_keep_recent_tokens x"):
            with self.subTest(command=command), self.assertRaises(ValueError):
                user_tools.parse_user_tool(command)


class ConfigCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "interaction.jsonl"

    async def test_the_cli_notes_an_old_key_and_shows_and_saves_the_new_one(self):
        args = cli._build_parser().parse_args([
            "--endpoint-api", "codex", "--model", "test",
            "--endpoint-auth-file", str(self.root / "auth.json"),
        ])
        submitted = False

        def frame(terminal, editor, status):
            nonlocal submitted
            if status != "idle":
                return
            if not submitted:
                terminal.submit("/config compaction_keep_recent_tokens 0")
                submitted = True
            elif any(item.text.startswith("[user-tool-ret]") for item in terminal.items):
                terminal.key("c-d")

        terminal = _Terminal(frame)
        result = await asyncio.wait_for(
            cli._run(_Model(self.path), Environment(), terminal, args, self.path), 4)
        self.assertEqual(result, 0)
        texts = [item.text for item in terminal.items]
        self.assertIn("[cli] Note: compaction_keep_recent_tokens is deprecated; "
                      "use compaction_pi_keep_recent_tokens.", texts)
        [output] = [text for text in texts if text.startswith("[user-tool-ret]")]
        self.assertIn("compaction_pi_keep_recent_tokens = 0", output)
        self.assertNotIn("compaction_keep_recent_tokens", output)
        [call] = [item for item in load_interaction_save(self.path).items
                  if isinstance(item, UserToolCall)]
        self.assertEqual(json.loads(call.call.arguments_json),
                         {"key": "compaction_pi_keep_recent_tokens", "value": 0})


class AutoConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "roles.json"

    def write(self, document):
        self.path.write_text(json.dumps(document))
        return self.path

    def test_config_files_may_say_version_3_4_or_5(self):
        self.assertEqual(CONFIG_VERSION, 5)
        for version in (3, 4, 5):
            with self.subTest(version=version), _caught() as caught:
                entries = load_config_file(self.write({
                    "version": version, "main": {"compaction_pi_max_output_tokens": 9}}))
                self.assertEqual(entries, {1: {"compaction_pi_max_output_tokens": 9}})
                self.assertEqual(caught, [])
        for version in (2, 6, "5", None):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "version 5"):
                load_config_file(self.write({"version": version, "main": {}}))

    def test_old_keys_load_renamed_with_one_warning_naming_the_file_and_keys(self):
        path = self.write({
            "version": 3,
            "main": {"auto_compact_tokens": 500, "compaction_keep_recent_tokens": 0},
            "watcher": {"compaction_max_output_tokens": 9, "auto_compact_tokens": 700},
        })
        with _caught() as caught:
            entries = load_config_file(path)
        self.assertEqual(entries, {
            1: {"auto_compaction_context_tokens": 500, "compaction_pi_keep_recent_tokens": 0},
            -1: {"compaction_pi_max_output_tokens": 9, "auto_compaction_context_tokens": 700},
        })
        [warning] = caught
        self.assertIs(warning.category, FutureWarning)
        self.assertEqual(str(warning.message), (
            f"Auto configuration {str(path)!r} uses the deprecated auto_compact_tokens, "
            "compaction_keep_recent_tokens and compaction_max_output_tokens; rename them to "
            "auto_compaction_context_tokens, compaction_pi_keep_recent_tokens and "
            "compaction_pi_max_output_tokens."))
        with _caught() as caught:
            settings = resolve_config(self.write({
                "version": 4, "main": {"compaction_keep_recent_tokens": 0}}), {"cwd": str(self.root)})
        self.assertEqual(settings[1]["compaction_pi_keep_recent_tokens"], 0)
        self.assertEqual([str(warning.message) for warning in caught], [
            f"Auto configuration {str(self.path)!r} uses the deprecated "
            "compaction_keep_recent_tokens; rename it to compaction_pi_keep_recent_tokens."])
        with _caught() as caught:
            entries = load_config_file(self.write({
                "version": 5, "main": {"request_timeout_seconds": 90.0}}))
        self.assertEqual(entries, {1: {"timeouts_request_seconds": 90.0}})
        self.assertEqual([str(warning.message) for warning in caught], [
            f"Auto configuration {str(self.path)!r} uses the deprecated "
            "request_timeout_seconds; rename it to timeouts_request_seconds."])

    def test_a_role_that_sets_both_names_fails(self):
        self.write({"version": 4, "watcher": {
            "compaction_keep_recent_tokens": 0, "compaction_pi_keep_recent_tokens": 0}})
        with _caught() as caught, self.assertRaisesRegex(
                ValueError, "^The watcher entry sets both compaction_keep_recent_tokens and its "
                            "new name compaction_pi_keep_recent_tokens; set only the new name.$"):
            load_config_file(self.path)
        self.assertEqual(caught, [])

    def test_saves_are_version_5_and_older_saves_load_renamed_silently(self):
        current = resolve_config(overrides={
            "cwd": str(self.root), "model": "saved-model", "auto_compaction_context_tokens": 500,
            "compaction_pi_keep_recent_tokens": 0, "compaction_pi_max_output_tokens": 9,
            "timeouts_request_seconds": 90.0,
        })
        document = saved_document(current)
        self.assertEqual(document["version"], 5)
        version_3 = {"version": 3, **{
            role: {OLD_NAMES.get(key, key): value for key, value in entry.items()}
            for role, entry in document.items() if role != "version"
        }}
        self.assertLessEqual(set(ALL_RENAMES), set(version_3["main"]))
        # Version 4 kept only the timeout's old name.
        version_4 = {"version": 4, **{
            role: {("request_timeout_seconds" if key == "timeouts_request_seconds" else key): value
                   for key, value in entry.items()}
            for role, entry in document.items() if role != "version"
        }}
        self.assertIn("request_timeout_seconds", version_4["main"])
        path = self.root / "config.json"
        for saved in (document, version_3, version_4):
            with self.subTest(version=saved["version"]), _caught() as caught:
                path.write_text(json.dumps(saved))
                self.assertEqual(resolve_config(saved=load_saved_config(path)), current)
                self.assertEqual(caught, [])
        # Each version has only its own old names, and a role may not set both.
        for label, broken in (
                ("old names in version 4", {**version_3, "version": 4}),
                ("old names in version 5", {**version_4, "version": 5}),
                ("both names", {**version_3, "main": {
                    **version_3["main"], "auto_compaction_context_tokens": 500}})):
            with self.subTest(label), self.assertRaisesRegex(ValueError, "^Invalid saved"):
                path.write_text(json.dumps(broken))
                load_saved_config(path)


class _DemoModel:
    def sample(self, context, *, tools=(), sample_params=None):
        return _answer("hi")


class DemoRunTests(unittest.TestCase):
    VALUES = {"auto_compaction_context_tokens": 500, "compaction_pi_keep_recent_tokens": 0,
              "compaction_pi_max_output_tokens": 9}

    def inputs(self, **keywords):
        """The config inputs that ``demo.run`` builds from ``keywords``."""
        with mock.patch.object(InteractionConfig, "from_model",
                               wraps=InteractionConfig.from_model) as from_model, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(demo.run(_DemoModel(), Environment(), prompt="hello", **keywords), "hi")
        return from_model.call_args.args[1]

    def test_new_keywords_work_silently(self):
        with _caught() as caught:
            inputs = self.inputs(**self.VALUES)
        self.assertEqual([str(warning.message) for warning in caught
                          if issubclass(warning.category, FutureWarning)], [])
        self.assertEqual({key: getattr(inputs, key) for key in self.VALUES}, self.VALUES)

    def test_old_keywords_work_with_a_warning(self):
        for old, new in RENAMES.items():
            with self.subTest(old=old):
                with self.assertWarnsRegex(FutureWarning, "^" + re.escape(
                        f"demo.run's {old} keyword is deprecated; use {new}.") + "$"):
                    inputs = self.inputs(**{old: self.VALUES[new]})
                self.assertEqual(getattr(inputs, new), self.VALUES[new])
                with self.assertRaisesRegex(TypeError, re.escape(
                        f"run() got both {old!r} and its new name {new!r}")):
                    demo.run(_DemoModel(), Environment(), **{old: 1, new: 1})

    def test_the_timeout_is_not_a_run_keyword_under_either_name(self):
        # demo.run's request timeout comes with sample_params.
        for name in (*TIMEOUT_RENAME, *TIMEOUT_RENAME.values()):
            with self.subTest(name=name), _caught() as caught, self.assertRaises(TypeError):
                demo.run(_DemoModel(), Environment(), **{name: 1})
            self.assertEqual([warning for warning in caught
                              if issubclass(warning.category, FutureWarning)], [])


if __name__ == "__main__":
    unittest.main()
