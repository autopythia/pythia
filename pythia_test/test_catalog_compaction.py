"""Part 3 of the warm-cold compaction plan: compaction settings in the model catalog.

The catalog's [compaction] section sets defaults for every model, and a model
section's compaction.* fields set that model's. Each setting resolves on its
own: the launch value, else the model's catalog value, else [compaction], else
the built-in default. /config KEY null restores the catalog's value.
"""

from __future__ import annotations

import asyncio
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pythia.interaction import (
    BUILTIN_MODEL_CATALOG, ChatCompletionsModel, Environment, InteractionConfig, auto, cli,
)
from pythia.interaction._auto_config import resolve_config, saved_document
from pythia.interaction._model_binding_debug import save_debug_model_bindings
from pythia.interaction.model_catalog import CompactionDefaults, ModelCatalog
from pythia.interaction.model_catalog_config import parse_model_catalog
from pythia.interaction.model_config import (
    build_model, frontend_catalog, prepare_namespace, render_model_catalog,
)
from pythia.interaction.runtime_config import ConfigError
from pythia_test.interaction_helpers import chat_endpoint
from pythia_test.test_compaction_sequences import _Model, _Terminal


HEADER = "[catalog]\nversion = 4\n"
LOCAL = """
[model.local]
endpoint.api = chat-completions
endpoint.model = served
endpoint.url = http://127.0.0.1:8000/v1/chat/completions
endpoint.auth = none
"""
SMALL = """
[model.small]
endpoint.api = chat-completions
endpoint.model = served-small
endpoint.url = http://127.0.0.1:8001/v1/chat/completions
endpoint.auth = none
"""
KEYS = ("compaction_mode", "compaction_pi_keep_recent_tokens", "compaction_pi_max_output_tokens")
OPUS_MAX = "[model.claude-opus-5.5-max]\noverride = true\nendpoint.api = messages\n"


def catalog(text, *, base=BUILTIN_MODEL_CATALOG):
    return parse_model_catalog(HEADER + text, base=base, source="test.ini")


def prepared(registry, *flags):
    return prepare_namespace(cli._build_parser().parse_args(flags), registry)


def config(registry, *flags, reason=None):
    return InteractionConfig.from_namespace(prepared(registry, *flags),
                                            coldstart_unavailable=reason)


def values(config):
    return {key: config.get(key) for key in KEYS}


class ParsingTests(unittest.TestCase):
    def test_the_section_and_model_fields_parse_and_the_binding_merges_them(self):
        registry = catalog("""
[compaction]
mode = coldstart , warmpi:3 , pi
pi_keep_recent_tokens = 0
pi_max_output_tokens = 4096
""" + LOCAL + """compaction.mode = "pi:2"
compaction.pi_keep_recent_tokens = 30000
""")
        self.assertEqual(registry.compaction, CompactionDefaults(
            mode="coldstart,warmpi:3,pi", pi_keep_recent_tokens=0, pi_max_output_tokens=4096))
        self.assertEqual(registry.get_model_spec("chat-completions", "local").compaction,
                         CompactionDefaults(mode="pi:2", pi_keep_recent_tokens=30000))
        # Field by field: the entry's values, then [compaction]'s.
        self.assertEqual(registry.bind("chat-completions", "local").compaction,
                         CompactionDefaults("pi:2", 30000, 4096))
        # [compaction] reaches built-in and uncatalogued models too.
        for api, name in (("messages", "claude-opus-5.5"), ("chat-completions", "other")):
            with self.subTest(name=name):
                self.assertEqual(registry.bind(api, name).compaction, registry.compaction)
        self.assertEqual(BUILTIN_MODEL_CATALOG.compaction, CompactionDefaults())
        self.assertEqual(BUILTIN_MODEL_CATALOG.bind("messages", "claude-opus-5.5").compaction,
                         CompactionDefaults())

    def test_the_plans_example_parses(self):
        registry = catalog("""
[compaction]
mode = coldstart,warmpi:3,pi

[model.codex-gpt-6-astra-max]
override = true
endpoint.api = codex
compaction.mode = provider:2

[model.claude-opus-5.5-max]
override = true
endpoint.api = messages
compaction.pi_keep_recent_tokens = 30000
""")
        self.assertEqual(registry.bind("codex", "codex-gpt-6-astra-max").compaction,
                         CompactionDefaults("provider:2"))
        self.assertEqual(registry.bind("messages", "claude-opus-5.5-max").compaction,
                         CompactionDefaults("coldstart,warmpi:3,pi", 30000))

    def test_unknown_names_and_invalid_values_fail_naming_them(self):
        for text, message in (
            ("[compaction]\nkeep_recent_tokens = 1\n",
             r"^Invalid model catalog \[compaction\] key 'keep_recent_tokens' in test\.ini; "
             r"expected mode, pi_keep_recent_tokens, or pi_max_output_tokens$"),
            ("[compaction]\nmode = pii\n",
             r"^Invalid model catalog \[compaction\] mode in test\.ini: compaction_mode: "
             r"unknown method 'pii'"),
            ("[compaction]\nmode = 3\n",
             r"^Invalid model catalog \[compaction\] mode in test\.ini: compaction_mode: "
             r"unknown method '3'"),
            ("[compaction]\npi_keep_recent_tokens = -1\n",
             r"^Invalid model catalog \[compaction\] pi_keep_recent_tokens in test\.ini: "
             r"expected a nonnegative integer or null$"),
            ("[compaction]\npi_max_output_tokens = 0\n",
             r"^Invalid model catalog \[compaction\] pi_max_output_tokens in test\.ini: "
             r"expected a positive integer or null$"),
            ("[compaction]\npi_max_output_tokens = 1.5\n",
             r"^Invalid model catalog \[compaction\] pi_max_output_tokens in test\.ini: "
             r"expected a positive integer or null$"),
            (LOCAL + "compaction.keep_recent_tokens = 1\n",
             r"^Invalid model catalog entry 'model\.local' in test\.ini: "
             r"compaction\.keep_recent_tokens is not a compaction field; use compaction\.mode, "
             r"compaction\.pi_keep_recent_tokens, or compaction\.pi_max_output_tokens$"),
            (LOCAL + "compaction.mode = pi:0\n",
             r"^Invalid model catalog entry 'model\.local' in test\.ini: compaction\.mode: "
             r"compaction_mode: the attempts for pi must be 1 to 9\.$"),
            (LOCAL + "compaction.pi_keep_recent_tokens = lots\n",
             r"^Invalid model catalog entry 'model\.local' in test\.ini: "
             r"compaction\.pi_keep_recent_tokens: expected a nonnegative integer or null$"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                catalog(text)

    def test_provider_only_where_the_route_has_it(self):
        # [compaction] reaches every route; legacydefault names each one's method.
        with self.assertRaisesRegex(ValueError, (
                r"^Invalid model catalog \[compaction\] mode in test\.ini: it may not name "
                r"provider, which only some routes have; legacydefault names each route's own "
                r"method$")):
            catalog("[compaction]\nmode = coldstart,provider\n")
        for text, message in (
            (LOCAL + "compaction.mode = coldstart,provider\n",
             "compaction.mode: this endpoint has no provider compaction$"),
            (OPUS_MAX + "compaction.mode = pi,provider\n",
             "compaction.mode: on Messages, provider may only come first: it turns on "
             "server-side compaction, which runs inside samples$"),
            # An override that moves Codex off the official route loses provider.
            ("[model.codex-gpt-6-astra-max]\noverride = true\nendpoint.api = codex\n"
             "endpoint.url = https://proxy.example.test/v1/responses\nendpoint.auth = none\n"
             "compaction.mode = provider\n",
             "compaction.mode: this endpoint has no provider compaction$"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                catalog(text)
        registry = catalog(OPUS_MAX + "compaction.mode = provider,pi\n"
                           "[model.codex-gpt-6-astra-max]\noverride = true\nendpoint.api = codex\n"
                           "compaction.mode = provider:2,pi\n")
        self.assertEqual(registry.bind("messages", "claude-opus-5.5-max").compaction.mode,
                         "provider,pi")
        self.assertEqual(registry.bind("codex", "codex-gpt-6-astra-max").compaction.mode,
                         "provider:2,pi")
        # Python-built specs and catalogs follow the same rules.
        local = catalog(LOCAL).get_model_spec("chat-completions", "local")
        with self.assertRaisesRegex(ValueError, "no provider compaction"):
            replace(local, compaction=CompactionDefaults(mode="provider"))
        with self.assertRaisesRegex(ValueError, "may not name provider"):
            ModelCatalog((), compaction=CompactionDefaults(mode="provider"))
        with self.assertRaisesRegex(ValueError, "compaction_mode: unknown method"):
            CompactionDefaults(mode="pii")
        with self.assertRaisesRegex(ValueError, "pi_keep_recent_tokens"):
            CompactionDefaults(pi_keep_recent_tokens=-1)
        with self.assertRaisesRegex(ValueError, "pi_max_output_tokens"):
            CompactionDefaults(pi_max_output_tokens=True)

    def test_null_removes_a_value_so_the_next_layer_applies(self):
        first = catalog("[compaction]\nmode = coldstart,pi:3\n" + OPUS_MAX
                        + "compaction.mode = pi\ncompaction.pi_keep_recent_tokens = 30000\n")
        self.assertEqual(first.bind("messages", "claude-opus-5.5-max").compaction.mode, "pi")
        # In a model section, null removes the entry's own value: [compaction] applies.
        second = catalog(OPUS_MAX + "compaction.mode = null\n", base=first)
        self.assertEqual(second.get_model_spec("messages", "claude-opus-5.5-max").compaction,
                         CompactionDefaults(pi_keep_recent_tokens=30000))
        self.assertEqual(second.bind("messages", "claude-opus-5.5-max").compaction,
                         CompactionDefaults("coldstart,pi:3", 30000))
        # In [compaction], null unsets a value: the built-in default applies.
        third = catalog("[compaction]\nmode = null\npi_keep_recent_tokens = 100\n", base=second)
        self.assertEqual(third.compaction, CompactionDefaults(pi_keep_recent_tokens=100))

    def test_an_override_patches_one_entry_and_the_section_reaches_every_variant(self):
        registry = catalog("[compaction]\nmode = coldstart,pi:3\n" + OPUS_MAX
                           + "compaction.pi_keep_recent_tokens = 30000\n")
        self.assertEqual(registry.bind("messages", "claude-opus-5.5-max").compaction,
                         CompactionDefaults("coldstart,pi:3", 30000))
        self.assertEqual(registry.bind("messages", "claude-opus-5.5").compaction,
                         CompactionDefaults("coldstart,pi:3"))


class ResolutionTests(unittest.TestCase):
    def test_each_setting_resolves_on_its_own_and_null_restores_the_catalogs(self):
        registry = catalog("[compaction]\nmode = coldstart,pi:3\npi_max_output_tokens = 4096\n"
                           + OPUS_MAX + "compaction.pi_keep_recent_tokens = 30000\n")
        opus = ("--model", "claude-opus-5.5-max")
        self.assertEqual(values(config(registry, *opus)), {
            "compaction_mode": "coldstart,pi:3", "compaction_pi_keep_recent_tokens": 30000,
            "compaction_pi_max_output_tokens": 4096})
        # The other effort variant has only [compaction]'s values.
        self.assertEqual(config(registry, "--model", "claude-opus-5.5").get(
            "compaction_pi_keep_recent_tokens"), 20000)
        # A launch value wins for its own setting only.
        launched = config(registry, *opus, "--compaction-mode", "warmpi",
                          "--compaction-pi-keep-recent-tokens", "5")
        self.assertEqual(values(launched), {
            "compaction_mode": "warmpi", "compaction_pi_keep_recent_tokens": 5,
            "compaction_pi_max_output_tokens": 4096})
        # null restores the catalog's value, and reset the launch value.
        for key, value in (("compaction_mode", "coldstart,pi:3"),
                           ("compaction_pi_keep_recent_tokens", 30000),
                           ("compaction_pi_max_output_tokens", 4096)):
            with self.subTest(key=key):
                self.assertEqual(launched.set(key, None), value)
        self.assertEqual(launched.reset().compaction_pi_keep_recent_tokens, 5)
        # Without the user catalog, the built-in defaults.
        self.assertEqual(values(config(BUILTIN_MODEL_CATALOG, *opus)), {
            "compaction_mode": "coldstart,warmpi:2", "compaction_pi_keep_recent_tokens": 20000,
            "compaction_pi_max_output_tokens": None})

    def test_legacydefault_in_the_catalog_resolves_per_route(self):
        registry = catalog("[compaction]\nmode = coldstart,legacydefault:2\n")
        self.assertEqual(config(registry, "--model", "codex-gpt-6-astra").get("compaction_mode"),
                         "coldstart,provider:2")
        self.assertEqual(config(registry, "--model", "claude-opus-5.5").get("compaction_mode"),
                         "coldstart,pi:2")

    def test_without_coldstart_the_catalogs_default_loses_it(self):
        registry = catalog("[compaction]\nmode = coldstart,pi:3\n" + LOCAL)
        # The default loses coldstart silently; a set value with the notice.
        unset = config(registry, "--model", "local", reason="no exec_command")
        self.assertEqual((unset.get("compaction_mode"), unset.coldstart_dropped), ("pi:3", None))
        alone = config(registry, "--model", "local", "--compaction-mode", "coldstart",
                       reason="no exec_command")
        self.assertEqual((alone.get("compaction_mode"), alone.coldstart_dropped),
                         ("pi:3", ("coldstart", "pi:3")))
        self.assertEqual(alone.set("compaction_mode", None), "pi:3")
        # A catalog default of coldstart alone gives the built-in one without it.
        only = config(catalog("[compaction]\nmode = coldstart\n" + LOCAL), "--model", "local",
                      reason="no exec_command")
        self.assertEqual(only.get("compaction_mode"), "warmpi:2")

    def test_a_messages_entry_with_provider_turns_on_server_compaction(self):
        registry = catalog("[compaction]\nmode = coldstart,pi\n" + OPUS_MAX
                           + "compaction.mode = provider\n")
        flags = ("--model", "claude-opus-5.5-max", "--endpoint-auth", "none")
        args = prepared(registry, *flags)
        model = build_model(args)
        self.addCleanup(model.close)
        self.assertIsNotNone(model.endpoint.server_compaction)
        server = InteractionConfig.from_namespace(args)
        self.assertEqual(server.get("compaction_mode"), "provider")
        self.assertEqual(server.set("compaction_mode", None), "provider")
        # Launched with pi, the session has no server-side compaction, so null
        # can't restore provider: it restores the built-in default.
        args = prepared(registry, *flags, "--compaction-mode", "pi")
        model = build_model(args)
        self.addCleanup(model.close)
        self.assertIsNone(model.endpoint.server_compaction)
        self.assertEqual(InteractionConfig.from_namespace(args).set("compaction_mode", None),
                         "coldstart,warmpi:2")
        # Likewise [compaction]'s value in a session with server-side compaction.
        launched = config(registry, "--model", "claude-opus-5.5", "--compaction-mode", "provider")
        self.assertEqual(launched.set("compaction_mode", None), "provider")

    def test_a_catalog_mode_the_launched_route_cant_run_fails_naming_the_catalog(self):
        registry = catalog("[model.codex-gpt-6-astra-max]\noverride = true\nendpoint.api = codex\n"
                           "compaction.mode = provider:2\n")
        self.assertEqual(config(registry, "--model", "codex-gpt-6-astra-max").get(
            "compaction_mode"), "provider:2")
        proxy = ("--model", "codex-gpt-6-astra-max", "--endpoint-auth", "none",
                 "--endpoint-url", "https://proxy.example.test/v1/responses")
        with self.assertRaisesRegex(ConfigError, (
                r"^compaction_mode provider:2 \(from the model catalog\): this route has no "
                r"provider compaction; use --compaction-mode pi$")):
            config(registry, *proxy)
        with self.assertRaisesRegex(ConfigError, "from the model catalog"):
            build_model(prepared(registry, *proxy))
        # With a launch value, null restores the built-in default instead.
        launched = config(registry, *proxy, "--compaction-mode", "pi")
        self.assertEqual(launched.set("compaction_mode", None), "coldstart,warmpi:2")

    def test_python_built_models_take_their_bindings_catalog_values(self):
        registry = catalog("[compaction]\nmode = coldstart,pi\npi_keep_recent_tokens = 100\n")
        binding = registry.bind("chat-completions", "served", endpoint_auth="none")
        model = ChatCompletionsModel(chat_endpoint(binding=binding))
        self.addCleanup(model.close)
        launched = InteractionConfig.from_model(model)
        self.assertEqual(launched.get("compaction_mode"), "coldstart,pi")
        # The default snapshot sets a 20,000-token tail; null restores the catalog's.
        self.assertEqual(launched.get("compaction_pi_keep_recent_tokens"), 20000)
        self.assertEqual(launched.set("compaction_pi_keep_recent_tokens", None), 100)

    def test_no_user_model_catalog_drops_the_values(self):
        with tempfile.TemporaryDirectory() as home:
            path = Path(home) / ".pythia" / "model-catalog.ini"
            path.parent.mkdir()
            path.write_text(HEADER + "[compaction]\nmode = coldstart,pi\n")
            with mock.patch.dict(os.environ, {"HOME": home}):
                found = frontend_catalog(cli._build_parser().parse_args([]))
                dropped = frontend_catalog(
                    cli._build_parser().parse_args(["--no-user-model-catalog"]))
        self.assertEqual(found.compaction, CompactionDefaults(mode="coldstart,pi"))
        self.assertEqual(dropped.compaction, CompactionDefaults())


class DisplayTests(unittest.TestCase):
    def registry(self):
        return catalog("[compaction]\nmode = coldstart,warmpi:3,pi\npi_keep_recent_tokens = 30000\n"
                       + LOCAL + "compaction.mode = pi:2\ncompaction.pi_max_output_tokens = 64\n")

    def test_list_models_shows_the_values_and_the_section(self):
        lines = render_model_catalog(self.registry()).splitlines()
        [local] = [line for line in lines if line.startswith("local:")]
        self.assertTrue(local.endswith(
            ", source=test.ini, compaction.mode=pi:2, compaction.pi_max_output_tokens=64"), local)
        self.assertEqual(lines[-1], (
            "[compaction] mode = coldstart,warmpi:3,pi, pi_keep_recent_tokens = 30000"))
        self.assertNotIn("compaction", render_model_catalog(BUILTIN_MODEL_CATALOG))

    def test_the_debug_binding_save_has_the_merged_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bindings.json"
            self.assertIsNone(save_debug_model_bindings(path, {
                "main": self.registry().bind("chat-completions", "local"),
                "other": BUILTIN_MODEL_CATALOG.bind("messages", "claude-opus-5.5"),
            }))
            bindings = json.loads(path.read_text())["bindings"]
        self.assertEqual(bindings["main"]["compaction"], {
            "mode": "pi:2", "pi_keep_recent_tokens": 30000, "pi_max_output_tokens": 64})
        self.assertEqual(bindings["other"]["compaction"], {
            "mode": None, "pi_keep_recent_tokens": None, "pi_max_output_tokens": None})


class AutoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)

    def catalog_file(self, name, text):
        path = self.temp / name
        path.write_text(HEADER + text)
        return path

    def print_config(self, *argv):
        stdout = io.StringIO()
        with (mock.patch.object(auto, "_Session") as session, redirect_stdout(stdout),
              redirect_stderr(io.StringIO())):
            self.assertEqual(auto.main([*argv, "--print-config"]), 0)
        session.assert_not_called()
        return stdout.getvalue().splitlines()

    def test_each_role_resolves_its_own_models_values(self):
        path = self.catalog_file("roles.ini", "[compaction]\nmode = coldstart,pi:3\n"
                                 + LOCAL + "compaction.mode = coldstart,warmpi\n" + SMALL)
        main, watcher = self.print_config("--model-catalog", str(path), "--model", "local",
                                          "--watcher-model", "small",
                                          "--save", str(self.temp / "x"))[:2]
        self.assertTrue(main.endswith("; compaction coldstart,warmpi"), main)
        # The watcher can't run coldstart, and loses it from a catalog value silently.
        self.assertTrue(watcher.endswith("; compaction pi:3"), watcher)

    def test_the_watcher_drops_a_catalog_coldstart_without_a_notice(self):
        registry = catalog("[compaction]\nmode = coldstart,pi:3\n" + LOCAL)
        settings = resolve_config(overrides={"cwd": str(self.temp), "model": "local"},
                                  catalog=registry, roles=(1, -1))
        session = auto._Session(self.temp / "run", settings,
                                model_factory=lambda index, args: mock.Mock()).start()
        self.addCleanup(session.close)
        texts = [item.text for event in session.drain_events() for item in event.items]
        session.close()
        self.assertEqual([text for text in texts if "compaction_mode" in text], [])

    def test_a_catalog_change_applies_on_resume(self):
        first = catalog("[compaction]\nmode = coldstart,pi:3\n" + LOCAL)
        save = self.temp / "run"
        save.mkdir()
        document = saved_document(resolve_config(
            overrides={"cwd": str(self.temp), "model": "local"}, catalog=first, roles=(1, -1)))
        # Unset settings are saved as null, so each launch resolves them again.
        self.assertIsNone(document["main"]["compaction_mode"])
        (save / "config.json").write_text(json.dumps(document))
        second = self.catalog_file("second.ini", "[compaction]\nmode = coldstart,warmpi:4\n" + LOCAL)
        main = self.print_config("--model-catalog", str(second), "--save", str(save),
                                 "--resume")[0]
        self.assertTrue(main.endswith("; compaction coldstart,warmpi:4"), main)


class CLITests(unittest.IsolatedAsyncioTestCase):
    async def test_a_session_without_exec_command_drops_a_catalog_coldstart_silently(self):
        registry = catalog("[compaction]\nmode = coldstart,pi:3\n" + LOCAL)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "interaction.jsonl"
            terminal = _Terminal(("/config.json compaction_mode", "/quit"))
            args = prepared(registry, "--model", "local", "--enable-default-tools=False")
            await asyncio.wait_for(cli._run(_Model(), Environment(), terminal, args, path), 4)
        texts = [item.text for item in terminal.items]
        self.assertFalse([text for text in texts if "this session compacts with" in text], texts)
        [shown] = [text for text in texts if text.startswith("[user-tool-ret]")]
        self.assertEqual(json.loads(shown.split("\n", 1)[1]),
                         {"compaction_mode": "pi:3", "__init__": {"compaction_mode": "pi:3"}})


if __name__ == "__main__":
    unittest.main()
