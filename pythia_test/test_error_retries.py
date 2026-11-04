"""Automatic retries and Messages prompt caching as settings: the model
catalog's endpoint.error_retries and endpoint.<class>_error_retries (for rate
limits, timeouts, refusals and auth errors; endpoint.rate_limit_retries is the
rate limit's old name), their launch options and /config keys, and
endpoint_prompt_caching.

How the turn loop uses the counts is test_loop_kernel's ErrorClassTests and
RateLimitRetryTests; how each adapter classifies its errors is in its own test
module.
"""

from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest import mock

from pythia.interaction import InteractionConfig, auto, cli, user_tools
from pythia.interaction._auto_config import namespace, resolve_config
from pythia.interaction._model_binding_debug import _debug_entry
from pythia.interaction.model_catalog import DEFAULT_ERROR_RETRIES, ERROR_RETRY_FIELDS
from pythia.interaction.model_catalog import binding_from_namespace
from pythia.interaction.model_catalog_config import parse_model_catalog
from pythia.interaction.model_config import render_model_catalog
from pythia.interaction.runtime_config import ConfigError, ERROR_RETRY_KEYS
from pythia.interaction.runtime_config import parse_config_literal


def _http(path):
    """The route lines of an anonymous local endpoint."""
    return (f"endpoint.url = http://127.0.0.1:9/v1/{path}\nendpoint.model = fixture\n"
            "endpoint.auth = none\n")


# An entry on each API, with and without the settings.
_CATALOG = (
    "[catalog]\nversion = 4\n"
    "[model.chat]\nendpoint.api = chat-completions\n" + _http("chat/completions")
    + "endpoint.error_retries = 8\n"
    "[model.patient]\nendpoint.api = messages\n" + _http("messages")
    + "endpoint.rate_limit_error_retries = unbounded\nendpoint.timeout_error_retries = 1\n"
    "endpoint.prompt_caching = 1h\nlimits.max_output_tokens = 100\n"
    "[model.quick]\nendpoint.api = messages\n" + _http("messages")
    + "limits.max_output_tokens = 100\n"
    "[model.plain]\nendpoint.api = responses\n" + _http("responses")
    + "[model.relay]\nendpoint.api = claude-relay\nendpoint.model = claude-opus-5-5\n"
    "endpoint.refusal_error_retries = 0\n"
)


def _config(catalog, *argv):
    """A CLI session's config, as it launches with ``argv``."""
    args = cli._build_parser().parse_args(list(argv))
    return InteractionConfig.from_namespace(args, catalog=catalog)


def _counts(config):
    """The config's automatic-retry keys, by field name."""
    return {key[len("endpoint_"):]: config.get(key) for key in ERROR_RETRY_KEYS}


class ErrorRetriesCatalogTests(unittest.TestCase):
    def test_the_keys_are_the_general_one_and_four_classes(self):
        self.assertEqual(ERROR_RETRY_FIELDS, ("error_retries", "rate_limit_error_retries",
                                              "timeout_error_retries", "refusal_error_retries",
                                              "auth_error_retries"))
        self.assertEqual(ERROR_RETRY_KEYS,
                         tuple("endpoint_" + name for name in ERROR_RETRY_FIELDS))
        # Every class starts with today's rate-limit count.
        self.assertEqual(DEFAULT_ERROR_RETRIES, 7)

    def test_the_catalog_sets_the_counts_and_the_options_win(self):
        catalog = parse_model_catalog(_CATALOG)
        unset = dict.fromkeys(ERROR_RETRY_FIELDS)
        for model, flags, expected in (
            ("chat", [], {**unset, "error_retries": 8}),
            ("chat", ["--endpoint-error-retries", "2"], {**unset, "error_retries": 2}),
            ("chat", ["--endpoint-auth-error-retries", "unbounded"],
             {**unset, "error_retries": 8, "auth_error_retries": "unbounded"}),
            ("patient", [], {**unset, "error_retries": DEFAULT_ERROR_RETRIES,
                             "rate_limit_error_retries": "unbounded", "timeout_error_retries": 1}),
            ("patient", ["--endpoint-rate-limit-error-retries", "0"],
             {**unset, "error_retries": DEFAULT_ERROR_RETRIES, "rate_limit_error_retries": 0,
              "timeout_error_retries": 1}),
            ("plain", [], {**unset, "error_retries": DEFAULT_ERROR_RETRIES}),
            ("relay", [], {**unset, "error_retries": DEFAULT_ERROR_RETRIES,
                           "refusal_error_retries": 0}),
        ):
            with self.subTest(model=model, flags=flags):
                config = _config(catalog, "--model", model, *flags)
                self.assertEqual(_counts(config), expected)
        # null restores the catalog's value, not the option's: for a class
        # without one, that is following the general key again.
        config = _config(catalog, "--model", "patient", "--endpoint-error-retries", "3",
                         "--endpoint-timeout-error-retries", "4",
                         "--endpoint-refusal-error-retries", "5")
        self.assertEqual(config.set("endpoint_error_retries", None), DEFAULT_ERROR_RETRIES)
        self.assertEqual(config.set("endpoint_timeout_error_retries", None), 1)
        self.assertIsNone(config.set("endpoint_refusal_error_retries", None))
        # null in an override removes an entry's value.
        cleared = parse_model_catalog(
            "[catalog]\nversion = 4\n[model.chat]\noverride = true\n"
            "endpoint.api = chat-completions\nendpoint.error_retries = null\n", base=catalog)
        self.assertIsNone(
            cleared.get_model_spec("chat-completions", "chat").endpoint.error_retries)
        self.assertEqual(_config(cleared, "--model", "chat").get("endpoint_error_retries"),
                         DEFAULT_ERROR_RETRIES)
        # --list-models and --debug-save-model-binding show them.
        listing = render_model_catalog(catalog).splitlines()
        line = {entry.split(":", 1)[0]: entry for entry in listing}
        self.assertIn("auth=none, error_retries=8, source=user", line["chat"])
        self.assertIn("auth=none, prompt_caching=1h, rate_limit_error_retries=unbounded, "
                      "timeout_error_retries=1, source=user", line["patient"])
        self.assertNotIn("retries", line["plain"])
        self.assertEqual(
            {name: value for name, value in
             _debug_entry(catalog.bind("messages", "patient"))["endpoint"].items()
             if name in ERROR_RETRY_FIELDS},
            {**unset, "rate_limit_error_retries": "unbounded", "timeout_error_retries": 1})

    def test_the_old_name_of_the_rate_limit_s_key_still_loads_with_a_warning(self):
        entry = ("[catalog]\nversion = 4\n[model.chat]\nendpoint.api = chat-completions\n"
                 + _http("chat/completions") + "endpoint.rate_limit_retries = 4\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            catalog = parse_model_catalog(entry)
        endpoint = catalog.get_model_spec("chat-completions", "chat").endpoint
        self.assertEqual(endpoint.rate_limit_error_retries, 4)
        self.assertEqual([(warning.category, str(warning.message)) for warning in caught], [(
            FutureWarning, "Model catalog 'user' uses the deprecated endpoint.rate_limit_retries "
            "in [model.chat]; rename it to endpoint.rate_limit_error_retries.")])
        with self.assertRaisesRegex(ValueError, re.escape(
                "endpoint.rate_limit_retries is the deprecated name of "
                "endpoint.rate_limit_error_retries; set only the new name")):
            parse_model_catalog(entry + "endpoint.rate_limit_error_retries = 4\n")

    def test_catalog_counts_are_checked(self):
        entry = ("[catalog]\nversion = 4\n[model.x]\nendpoint.api = {api}\n{route}"
                 "endpoint.{field} = {value}\n")
        relay = "endpoint.model = claude-opus-5-5\n"
        wrong_api = ("endpoint.rate_limit_error_retries does not apply to endpoint.api "
                     "claude-relay")
        for field in ERROR_RETRY_FIELDS:
            message = f"endpoint.{field} must be a nonnegative integer, unbounded, or null"
            for api, route, value in (
                ("chat-completions", _http("chat/completions"), "-1"),
                ("chat-completions", _http("chat/completions"), "1.5"),
                ("chat-completions", _http("chat/completions"), "Unbounded"),
                ("messages", _http("messages"), "lots"),
            ):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(
                        ValueError, "^" + re.escape(
                            f"Invalid model catalog entry 'model.x' in user: {message}") + "$"):
                    parse_model_catalog(entry.format(api=api, route=route, field=field,
                                                     value=value))
        # Claude Relay takes all but the rate limit's, under either name.
        for field in ("rate_limit_error_retries", "rate_limit_retries"):
            with self.subTest(field=field), warnings.catch_warnings(), self.assertRaisesRegex(
                    ValueError, "^" + re.escape(
                        f"Invalid model catalog entry 'model.x' in user: {wrong_api}") + "$"):
                warnings.simplefilter("ignore")
                parse_model_catalog(entry.format(api="claude-relay", route=relay, field=field,
                                                 value=3))
        for field in ("error_retries", "timeout_error_retries", "refusal_error_retries",
                      "auth_error_retries"):
            catalog = parse_model_catalog(entry.format(api="claude-relay", route=relay,
                                                       field=field, value=3))
            self.assertEqual(getattr(catalog.get_model_spec("claude-relay", "x").endpoint,
                                     field), 3)
        # Python callers get the same checks from EndpointSpec.
        catalog = parse_model_catalog(_CATALOG)
        endpoint = catalog.get_model_spec("chat-completions", "chat").endpoint
        for field in ERROR_RETRY_FIELDS:
            for value in (-1, True, 1.5, "lots"):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(
                        ValueError, "^" + re.escape(
                            f"endpoint.{field} must be a nonnegative integer or unbounded") + "$"):
                    replace(endpoint, **{field: value})
        relay = catalog.get_model_spec("claude-relay", "relay").endpoint
        with self.assertRaisesRegex(
                ValueError, "^endpoint.rate_limit_error_retries does not apply to Claude Relay$"):
            replace(relay, rate_limit_error_retries=3)

    def test_the_options_take_a_count_or_unbounded(self):
        parser = cli._build_parser()
        options = ["--endpoint-" + field.replace("_", "-") for field in ERROR_RETRY_FIELDS]
        usage = parser.format_help()
        for option, key in zip(options, ERROR_RETRY_KEYS):
            self.assertIn(option, usage)
            for value, expected in (("7", 7), ("0", 0), ("unbounded", "unbounded")):
                self.assertEqual(getattr(parser.parse_args([option, value]), key), expected)
            for value in ("-1", "1.5", "lots", "Unbounded"):
                with self.subTest(option=option, value=value), \
                        redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
                    parser.parse_args([option, value])
                self.assertIn("must be a nonnegative integer or unbounded", stderr.getvalue())
        # The rate limit's old option takes the same, with a warning.
        with self.assertWarnsRegex(FutureWarning, re.escape(
                "--endpoint-rate-limit-retries is deprecated; use "
                "--endpoint-rate-limit-error-retries.")):
            args = parser.parse_args(["--endpoint-rate-limit-retries", "unbounded"])
        self.assertEqual(args.endpoint_rate_limit_error_retries, "unbounded")
        route = ["--endpoint-url", "http://127.0.0.1:9/v1/responses", "--endpoint-model",
                 "fixture", "--endpoint-auth", "none"]
        args = parser.parse_args(["--endpoint-api", "responses", *route,
                                  "--endpoint-rate-limit-error-retries", "3",
                                  "--endpoint-error-retries", "4"])
        endpoint = binding_from_namespace(args).endpoint
        self.assertEqual((endpoint.rate_limit_error_retries, endpoint.error_retries), (3, 4))
        # Claude Code retries the relay's rate limits itself; the rest apply.
        relay = ["--endpoint-api", "claude-relay", "--endpoint-model", "claude-opus-5-5"]
        args = parser.parse_args([*relay, "--endpoint-rate-limit-error-retries", "3"])
        with self.assertRaisesRegex(ValueError, "^--endpoint-rate-limit-error-retries does not "
                                                "apply to --endpoint-api claude-relay$"):
            binding_from_namespace(args)
        args = parser.parse_args([*relay, "--endpoint-timeout-error-retries", "3",
                                  "--endpoint-error-retries", "unbounded"])
        endpoint = binding_from_namespace(args).endpoint
        self.assertEqual((endpoint.timeout_error_retries, endpoint.error_retries),
                         (3, "unbounded"))

    def test_auto_gives_the_options_to_its_roles(self):
        catalog = parse_model_catalog(_CATALOG)
        for http, others, expected in (
                ({}, {}, {1: (8, None, None), -1: (None, None, 0)}),
                ({"endpoint_rate_limit_error_retries": "unbounded"},
                 {"endpoint_error_retries": 2, "endpoint_refusal_error_retries": 1},
                 {1: (2, "unbounded", 1), -1: (2, None, 1)})):
            with self.subTest(http=http, others=others), tempfile.TemporaryDirectory() as tmp:
                settings = resolve_config(overrides={"cwd": tmp}, catalog=catalog,
                                          role_models={1: "chat", -1: "relay"})
                settings.http_options = dict(http)
                settings.error_retry_options = dict(others)
                # The bindings, which --debug-save-model-binding records, resolve
                # when the session is made; it isn't started.
                session = auto._Session(Path(tmp) / "session", settings,
                                        model_factory=None, environment_factory=None)
                self.assertEqual({index: (binding.endpoint.error_retries,
                                          binding.endpoint.rate_limit_error_retries,
                                          binding.endpoint.refusal_error_retries)
                                  for index, binding in session.bindings.items()}, expected)
                # Main's config takes them from its binding, as its owner builds it.
                args = namespace(session.settings[1], catalog, binding=session.bindings[1])
                config = InteractionConfig.from_namespace(args)
                self.assertEqual(config.get("endpoint_rate_limit_error_retries"), expected[1][1])
                self.assertEqual(config.get("endpoint_error_retries"), expected[1][0])
                self.assertFalse((Path(tmp) / "session").exists())

    def test_auto_takes_the_options_at_launch(self):
        for flags, http, others in (
                ([], {}, {}),
                (["--endpoint-rate-limit-error-retries", "unbounded", "--endpoint-error-retries",
                  "0"], {"endpoint_rate_limit_error_retries": "unbounded"},
                 {"endpoint_error_retries": 0})):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as tmp:
                seen = []

                def stop(settings, *_args):
                    seen.append((settings.http_options, settings.error_retry_options))
                    raise ValueError("stopped before the session")

                with mock.patch.object(auto, "_check_credentials", stop), \
                        redirect_stderr(io.StringIO()) as stderr:
                    code = auto.main(["--no-user-model-catalog", "--model", "claude-sonnet-5.5",
                                      "--prompt", "Hello.", "--save", str(Path(tmp) / "save"),
                                      *flags])
                self.assertEqual(code, 1)
                self.assertIn("auto failed: stopped before the session", stderr.getvalue())
                self.assertEqual(seen, [(http, others)])
                self.assertFalse((Path(tmp) / "save").exists())


class ErrorRetriesConfigTests(unittest.TestCase):
    def test_each_key_takes_a_count_unbounded_or_null(self):
        for key in ERROR_RETRY_KEYS:
            for literal, value in (("0", 0), ("12", 12), ("unbounded", "unbounded"),
                                   ("null", None)):
                with self.subTest(key=key, literal=literal):
                    self.assertEqual(parse_config_literal(key, literal), value)
            for literal in ("-1", "true", "1.5", "Unbounded", "none", "lots"):
                with self.subTest(key=key, literal=literal), self.assertRaisesRegex(
                        ConfigError, "^" + re.escape(
                            f"{key} requires a nonnegative integer, unbounded, or null.") + "$"):
                    parse_config_literal(key, literal)

    def test_config_changes_them_and_claude_relay_refuses_only_the_rate_limit_s(self):
        catalog = parse_model_catalog(_CATALOG)
        config = _config(catalog, "--model", "chat")

        def run(command):
            intent = user_tools.parse_user_tool(command)
            outcome = user_tools.run_config(config, json.loads(intent.arguments_json))
            self.assertTrue(outcome.success)
            return outcome.output
        self.assertEqual(run("/config endpoint_error_retries unbounded"),
                         "# init: endpoint_error_retries = 8\n"
                         "endpoint_error_retries = 'unbounded'")
        # A class key without a value follows the general one, and says so.
        self.assertEqual(run("/config endpoint_timeout_error_retries"),
                         "endpoint_timeout_error_retries = None  "
                         "# follows endpoint_error_retries (unbounded)")
        self.assertEqual(run("/config endpoint_timeout_error_retries 2"),
                         "# init: endpoint_timeout_error_retries = None\n"
                         "endpoint_timeout_error_retries = 2")
        # The next turn's snapshot has them; the turn loop reads them there.
        snapshot = config.snapshot()
        self.assertEqual(
            (snapshot.endpoint_error_retries, snapshot.endpoint_timeout_error_retries),
            ("unbounded", 2))
        relay = _config(catalog, "--model", "relay")
        self.assertIsNone(relay.get("endpoint_rate_limit_error_retries"))
        with self.assertRaisesRegex(ConfigError, "^" + re.escape(
                "endpoint_rate_limit_error_retries does not apply to Claude Relay, whose native "
                "CLI retries its own rate limits.") + "$"):
            relay.set("endpoint_rate_limit_error_retries", 3)
        self.assertIsNone(relay.set("endpoint_rate_limit_error_retries", None))
        # It isn't said to follow the general key there: it doesn't apply.
        self.assertEqual(relay.render("endpoint_rate_limit_error_retries", json_output=False),
                         "endpoint_rate_limit_error_retries = None")
        self.assertEqual(relay.set("endpoint_auth_error_retries", "unbounded"), "unbounded")
        self.assertEqual(relay.set("endpoint_error_retries", 1), 1)

    def test_the_old_key_name_still_works_with_a_notice(self):
        intent = user_tools.parse_user_tool("/config endpoint_rate_limit_retries 2")
        self.assertEqual(json.loads(intent.arguments_json),
                         {"key": "endpoint_rate_limit_error_retries", "value": 2})
        self.assertEqual(intent.notice, "Note: endpoint_rate_limit_retries is deprecated; use "
                                        "endpoint_rate_limit_error_retries.")
        intent = user_tools.parse_user_tool("/config endpoint_rate_limit_retries unbounded")
        self.assertEqual(json.loads(intent.arguments_json),
                         {"key": "endpoint_rate_limit_error_retries", "value": "unbounded"})

    def test_a_reboot_carries_the_keys(self):
        catalog = parse_model_catalog(_CATALOG)
        config = _config(catalog, "--model", "patient")
        config.set("endpoint_rate_limit_error_retries", 2)
        config.set("endpoint_error_retries", "unbounded")
        config.set("endpoint_prompt_caching", "none")
        carried = cli._reboot_state(cli._UIState(reboot=cli._Reboot("/reboot")), config)
        self.assertEqual(carried["config"], {"endpoint_error_retries": "unbounded",
                                             "endpoint_rate_limit_error_retries": 2,
                                             "endpoint_prompt_caching": "none"})
        rebooted = cli._parse_reboot_state(json.loads(json.dumps(carried)))
        state, restarted = cli._UIState(), _config(catalog, "--model", "patient")
        cli._carry_over(state, restarted, rebooted)
        self.assertEqual(restarted.get("endpoint_rate_limit_error_retries"), 2)
        self.assertEqual(restarted.get("endpoint_error_retries"), "unbounded")
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
