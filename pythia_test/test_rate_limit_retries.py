"""Rate-limit retries and Messages prompt caching as settings: the model
catalog's endpoint.rate_limit_retries, its launch option, and the /config keys
endpoint_rate_limit_retries and endpoint_prompt_caching.

How the turn loop uses the count is test_loop_kernel's RateLimitRetryTests;
how each adapter raises a rate limit is in its own test module.
"""

from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest import mock

from pythia.interaction import InteractionConfig, auto, cli, user_tools
from pythia.interaction._auto_config import namespace, resolve_config
from pythia.interaction._model_binding_debug import _debug_entry
from pythia.interaction.model_catalog import DEFAULT_RATE_LIMIT_RETRIES
from pythia.interaction.model_catalog import binding_from_namespace
from pythia.interaction.model_catalog_config import parse_model_catalog
from pythia.interaction.model_config import render_model_catalog
from pythia.interaction.runtime_config import ConfigError
from pythia.interaction.runtime_config import parse_config_literal


def _http(path):
    """The route lines of an anonymous local endpoint."""
    return (f"endpoint.url = http://127.0.0.1:9/v1/{path}\nendpoint.model = fixture\n"
            "endpoint.auth = none\n")


# An entry on each API, with and without the two settings.
_CATALOG = (
    "[catalog]\nversion = 4\n"
    "[model.chat]\nendpoint.api = chat-completions\n" + _http("chat/completions")
    + "endpoint.rate_limit_retries = 8\n"
    "[model.patient]\nendpoint.api = messages\n" + _http("messages")
    + "endpoint.rate_limit_retries = unbounded\nendpoint.prompt_caching = 1h\n"
    "limits.max_output_tokens = 100\n"
    "[model.quick]\nendpoint.api = messages\n" + _http("messages")
    + "limits.max_output_tokens = 100\n"
    "[model.plain]\nendpoint.api = responses\n" + _http("responses")
    + "[model.relay]\nendpoint.api = claude-relay\nendpoint.model = claude-opus-5-5\n"
)


def _config(catalog, *argv):
    """A CLI session's config, as it launches with ``argv``."""
    args = cli._build_parser().parse_args(list(argv))
    return InteractionConfig.from_namespace(args, catalog=catalog)


class RateLimitRetriesCatalogTests(unittest.TestCase):
    def test_the_catalog_sets_the_count_and_the_option_wins(self):
        catalog = parse_model_catalog(_CATALOG)
        catalog_values = {"chat": 8, "patient": "unbounded", "plain": DEFAULT_RATE_LIMIT_RETRIES}
        for model, flags, expected in (
            ("chat", [], 8),
            ("chat", ["--endpoint-rate-limit-retries", "2"], 2),
            ("chat", ["--endpoint-rate-limit-retries", "unbounded"], "unbounded"),
            ("patient", [], "unbounded"),
            ("patient", ["--endpoint-rate-limit-retries", "0"], 0),
            ("plain", [], DEFAULT_RATE_LIMIT_RETRIES),
        ):
            with self.subTest(model=model, flags=flags):
                config = _config(catalog, "--model", model, *flags)
                self.assertEqual(config.get("endpoint_rate_limit_retries"), expected)
                # null restores the catalog's value, else the default; not the option's.
                self.assertEqual(config.set("endpoint_rate_limit_retries", None),
                                 catalog_values[model])
        self.assertEqual(DEFAULT_RATE_LIMIT_RETRIES, 5)
        # null in an override removes an entry's value, so the default applies.
        cleared = parse_model_catalog(
            "[catalog]\nversion = 4\n[model.chat]\noverride = true\n"
            "endpoint.api = chat-completions\nendpoint.rate_limit_retries = null\n", base=catalog)
        self.assertIsNone(
            cleared.get_model_spec("chat-completions", "chat").endpoint.rate_limit_retries)
        self.assertEqual(_config(cleared, "--model", "chat").get("endpoint_rate_limit_retries"),
                         DEFAULT_RATE_LIMIT_RETRIES)
        # --list-models and --debug-save-model-binding show it.
        listing = render_model_catalog(catalog).splitlines()
        line = {entry.split(":", 1)[0]: entry for entry in listing}
        self.assertIn("auth=none, rate_limit_retries=8, source=user", line["chat"])
        self.assertIn("auth=none, prompt_caching=1h, rate_limit_retries=unbounded, source=user",
                      line["patient"])
        self.assertNotIn("rate_limit_retries", line["plain"])
        self.assertEqual(
            _debug_entry(catalog.bind("chat-completions", "chat"))["endpoint"]["rate_limit_retries"],
            8)

    def test_catalog_counts_are_checked(self):
        entry = ("[catalog]\nversion = 4\n[model.x]\nendpoint.api = {api}\n{route}"
                 "endpoint.rate_limit_retries = {value}\n")
        wrong_value = "endpoint.rate_limit_retries must be a nonnegative integer, unbounded, or null"
        wrong_api = "endpoint.rate_limit_retries does not apply to endpoint.api claude-relay"
        for api, route, value, message in (
            ("chat-completions", _http("chat/completions"), "-1", wrong_value),
            ("chat-completions", _http("chat/completions"), "1.5", wrong_value),
            ("chat-completions", _http("chat/completions"), "Unbounded", wrong_value),
            ("messages", _http("messages"), "lots", wrong_value),
            ("claude-relay", "endpoint.model = claude-opus-5-5\n", "3", wrong_api),
        ):
            with self.subTest(api=api, value=value):
                with self.assertRaisesRegex(ValueError, "^" + re.escape(
                        f"Invalid model catalog entry 'model.x' in user: {message}") + "$"):
                    parse_model_catalog(entry.format(api=api, route=route, value=value))
        # Python callers get the same checks from EndpointSpec.
        catalog = parse_model_catalog(_CATALOG)
        endpoint = catalog.get_model_spec("chat-completions", "chat").endpoint
        for value in (-1, True, 1.5, "lots"):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError,
                    "^endpoint.rate_limit_retries must be a nonnegative integer or unbounded$"):
                replace(endpoint, rate_limit_retries=value)
        relay = catalog.get_model_spec("claude-relay", "relay").endpoint
        with self.assertRaisesRegex(
                ValueError, "^endpoint.rate_limit_retries does not apply to Claude Relay$"):
            replace(relay, rate_limit_retries=3)

    def test_the_option_takes_a_count_or_unbounded_on_every_api_but_claude_relay(self):
        parser = cli._build_parser()
        for value, expected in (("7", 7), ("0", 0), ("unbounded", "unbounded")):
            self.assertEqual(
                parser.parse_args(["--endpoint-rate-limit-retries", value])
                .endpoint_rate_limit_retries, expected)
        for value in ("-1", "1.5", "lots", "Unbounded"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()) as stderr, \
                    self.assertRaises(SystemExit):
                parser.parse_args(["--endpoint-rate-limit-retries", value])
            self.assertIn("must be a nonnegative integer or unbounded", stderr.getvalue())
        route = ["--endpoint-url", "http://127.0.0.1:9/v1/responses", "--endpoint-model",
                 "fixture", "--endpoint-auth", "none"]
        args = parser.parse_args(["--endpoint-api", "responses", *route,
                                  "--endpoint-rate-limit-retries", "3"])
        self.assertEqual(binding_from_namespace(args).endpoint.rate_limit_retries, 3)
        # Claude Code retries the relay's rate limits itself.
        args = parser.parse_args(["--endpoint-api", "claude-relay", "--endpoint-model",
                                  "claude-opus-5-5", "--endpoint-rate-limit-retries", "3"])
        with self.assertRaisesRegex(ValueError, "^--endpoint-rate-limit-retries does not apply "
                                                "to --endpoint-api claude-relay$"):
            binding_from_namespace(args)

    def test_auto_gives_the_option_to_every_role_but_claude_relay(self):
        catalog = parse_model_catalog(_CATALOG)
        for options, expected in (({}, {1: 8, -1: None}),
                                  ({"endpoint_rate_limit_retries": "unbounded"},
                                   {1: "unbounded", -1: None})):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                settings = resolve_config(overrides={"cwd": tmp}, catalog=catalog,
                                          role_models={1: "chat", -1: "relay"})
                settings.http_options = dict(options)
                # The bindings, which --debug-save-model-binding records, resolve
                # when the session is made; it isn't started.
                session = auto._Session(Path(tmp) / "session", settings,
                                        model_factory=None, environment_factory=None)
                self.assertEqual({index: binding.endpoint.rate_limit_retries
                                  for index, binding in session.bindings.items()}, expected)
                # Main's config takes it from its binding, as its owner builds it.
                args = namespace(session.settings[1], catalog, binding=session.bindings[1])
                self.assertEqual(
                    InteractionConfig.from_namespace(args).get("endpoint_rate_limit_retries"),
                    expected[1])
                self.assertFalse((Path(tmp) / "session").exists())

    def test_auto_takes_the_option_at_launch(self):
        for flags, expected in (([], {}), (["--endpoint-rate-limit-retries", "unbounded"],
                                           {"endpoint_rate_limit_retries": "unbounded"})):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as tmp:
                seen = []

                def stop(settings, *_args):
                    seen.append(settings.http_options)
                    raise ValueError("stopped before the session")

                with mock.patch.object(auto, "_check_credentials", stop), \
                        redirect_stderr(io.StringIO()) as stderr:
                    code = auto.main(["--no-user-model-catalog", "--model", "claude-sonnet-5.5",
                                      "--prompt", "Hello.", "--save", str(Path(tmp) / "save"),
                                      *flags])
                self.assertEqual(code, 1)
                self.assertIn("auto failed: stopped before the session", stderr.getvalue())
                self.assertEqual(seen, [expected])
                self.assertFalse((Path(tmp) / "save").exists())


class RateLimitRetriesConfigTests(unittest.TestCase):
    def test_the_key_takes_a_count_unbounded_or_null(self):
        for literal, value in (("0", 0), ("12", 12), ("unbounded", "unbounded"), ("null", None)):
            with self.subTest(literal=literal):
                self.assertEqual(parse_config_literal("endpoint_rate_limit_retries", literal), value)
        for literal in ("-1", "true", "1.5", "Unbounded", "none", "lots"):
            with self.subTest(literal=literal), self.assertRaisesRegex(ConfigError, "^" + re.escape(
                    "endpoint_rate_limit_retries requires a nonnegative integer, unbounded, "
                    "or null.") + "$"):
                parse_config_literal("endpoint_rate_limit_retries", literal)

    def test_config_changes_it_and_claude_relay_refuses_it(self):
        catalog = parse_model_catalog(_CATALOG)
        config = _config(catalog, "--model", "chat")
        intent = user_tools.parse_user_tool("/config endpoint_rate_limit_retries unbounded")
        outcome = user_tools.run_config(config, json.loads(intent.arguments_json))
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.output, "# init: endpoint_rate_limit_retries = 8\n"
                                         "endpoint_rate_limit_retries = 'unbounded'")
        # The next turn's snapshot has it; the turn loop reads it there.
        self.assertEqual(config.snapshot().endpoint_rate_limit_retries, "unbounded")
        relay = _config(catalog, "--model", "relay")
        self.assertIsNone(relay.get("endpoint_rate_limit_retries"))
        with self.assertRaisesRegex(ConfigError, "^" + re.escape(
                "endpoint_rate_limit_retries does not apply to Claude Relay, whose native CLI "
                "retries its own rate limits.") + "$"):
            relay.set("endpoint_rate_limit_retries", 3)
        self.assertIsNone(relay.set("endpoint_rate_limit_retries", None))

    def test_a_reboot_carries_both_keys(self):
        catalog = parse_model_catalog(_CATALOG)
        config = _config(catalog, "--model", "patient")
        config.set("endpoint_rate_limit_retries", 2)
        config.set("endpoint_prompt_caching", "none")
        carried = cli._reboot_state(cli._UIState(reboot=cli._Reboot("/reboot")), config)
        self.assertEqual(carried["config"], {"endpoint_rate_limit_retries": 2,
                                             "endpoint_prompt_caching": "none"})
        rebooted = cli._parse_reboot_state(json.loads(json.dumps(carried)))
        state, restarted = cli._UIState(), _config(catalog, "--model", "patient")
        cli._carry_over(state, restarted, rebooted)
        self.assertEqual(restarted.get("endpoint_rate_limit_retries"), 2)
        self.assertEqual(restarted.get("endpoint_prompt_caching"), "none")
        self.assertFalse(any("didn't carry over" in item.text for item in state.displays))


class PromptCachingConfigTests(unittest.TestCase):
    def test_the_key_starts_like_its_option_and_null_restores_the_catalog_s(self):
        catalog = parse_model_catalog(_CATALOG)
        for model, flags, expected, restored in (
            ("patient", [], "1h", "1h"),
            ("patient", ["--endpoint-prompt-caching", "none"], "none", "1h"),
            ("quick", [], "5m", "5m"),  # no catalog value: 5m
            ("quick", ["--endpoint-prompt-caching", "1h"], "1h", "5m"),
        ):
            with self.subTest(model=model, flags=flags):
                config = _config(catalog, "--model", model, *flags)
                self.assertEqual(config.get("endpoint_prompt_caching"), expected)
                # Each request of a turn gets it.
                self.assertEqual(config.snapshot().sample_params().prompt_caching, expected)
                self.assertEqual(config.set("endpoint_prompt_caching", "none"), "none")
                self.assertEqual(config.set("endpoint_prompt_caching", None), restored)

    def test_only_messages_has_it(self):
        catalog = parse_model_catalog(_CATALOG)
        for model in ("chat", "plain", "relay"):
            with self.subTest(model=model):
                config = _config(catalog, "--model", model)
                self.assertIsNone(config.get("endpoint_prompt_caching"))
                self.assertIsNone(config.snapshot().sample_params().prompt_caching)
                with self.assertRaisesRegex(
                        ConfigError, r"^endpoint_prompt_caching applies only to the Messages API\.$"):
                    config.set("endpoint_prompt_caching", "1h")
        for literal, value in (("5m", "5m"), ("1h", "1h"), ("none", "none"), ("null", None)):
            with self.subTest(literal=literal):
                self.assertEqual(parse_config_literal("endpoint_prompt_caching", literal), value)
        with self.assertRaisesRegex(
                ConfigError, r"^endpoint_prompt_caching requires 5m, 1h, none, or null\.$"):
            parse_config_literal("endpoint_prompt_caching", "2h")


if __name__ == "__main__":
    unittest.main()
