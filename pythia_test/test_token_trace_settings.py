"""Token traces as a setting: the model catalog's endpoint.token_trace, its
launch option, the /config key endpoint_token_trace, SampleParams.token_trace,
the vLLM request fields it reserves, and who asks: the turn loop's own
samples, never compaction.

The record itself is test_tokens.
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

from pythia.interaction import (
    CompactionSettings, InteractionConfig, InteractionContext, Message, SampleParams, auto, cli,
    run_compaction, user_tools,
)
from pythia.interaction._auto_config import namespace, resolve_config
from pythia.interaction._model_binding_debug import _debug_entry
from pythia.interaction.loop import kernel
from pythia.interaction.loop.kernel import run_turn
from pythia.interaction.model_catalog import TOKEN_TRACE_FIELDS
from pythia.interaction.model_catalog import TOKEN_TRACE_LEVELS
from pythia.interaction.model_catalog import binding_from_namespace
from pythia.interaction.model_catalog_config import parse_model_catalog
from pythia.interaction.model_config import render_model_catalog
from pythia.interaction.runtime_config import ConfigError
from pythia.interaction.runtime_config import parse_config_literal
from pythia_test.test_loop_kernel import _FakeCompaction, _Host, _Model, _answer, _config
from pythia_test.test_loop_kernel import _context, _environment


def _http(path):
    """The route lines of an anonymous local endpoint."""
    return (f"endpoint.url = http://127.0.0.1:9/v1/{path}\nendpoint.model = fixture\n"
            "endpoint.auth = none\n")


# Chat Completions entries with and without a level, and entries on other APIs.
_CATALOG = (
    "[catalog]\nversion = 4\n"
    "[model.traced]\nendpoint.api = chat-completions\n" + _http("chat/completions")
    + "endpoint.token_trace = sample_log_probs\n"
    "[model.chat]\nendpoint.api = chat-completions\n" + _http("chat/completions")
    + "[model.quick]\nendpoint.api = messages\n" + _http("messages")
    + "limits.max_output_tokens = 100\n"
    "[model.plain]\nendpoint.api = responses\n" + _http("responses")
    + "[model.relay]\nendpoint.api = claude-relay\nendpoint.model = claude-opus-5-5\n"
)
_ENTRY = ("[catalog]\nversion = 4\n[model.x]\nendpoint.api = {api}\n{route}{lines}")
_LEVELS = "none, token_ids, sample_log_probs, prefix_log_probs"


def _config_for(catalog, *argv):
    """A CLI session's config, as it launches with ``argv``."""
    args = cli._build_parser().parse_args(list(argv))
    return InteractionConfig.from_namespace(args, catalog=catalog)


class TokenTraceCatalogTests(unittest.TestCase):
    def test_the_catalog_sets_the_level_and_the_option_wins(self):
        catalog = parse_model_catalog(_CATALOG)
        for model, flags, expected, restored in (
            ("traced", [], "sample_log_probs", "sample_log_probs"),
            ("traced", ["--endpoint-token-trace", "none"], "none", "sample_log_probs"),
            ("chat", [], "none", "none"),  # no catalog value: none
            ("chat", ["--endpoint-token-trace", "prefix_log_probs"], "prefix_log_probs", "none"),
        ):
            with self.subTest(model=model, flags=flags):
                config = _config_for(catalog, "--model", model, *flags)
                self.assertEqual(config.get("endpoint_token_trace"), expected)
                # Each sample of a turn gets it; none asks for nothing, as None does.
                self.assertEqual(config.snapshot().sample_params().token_trace,
                                 None if expected == "none" else expected)
                self.assertEqual(config.set("endpoint_token_trace", "token_ids"), "token_ids")
                # null restores the catalog's value, else none; not the option's.
                self.assertEqual(config.set("endpoint_token_trace", None), restored)
        # null in an override removes an entry's value, so none applies.
        cleared = parse_model_catalog(
            "[catalog]\nversion = 4\n[model.traced]\noverride = true\n"
            "endpoint.api = chat-completions\nendpoint.token_trace = null\n", base=catalog)
        self.assertIsNone(cleared.get_model_spec("chat-completions", "traced").endpoint.token_trace)
        self.assertEqual(_config_for(cleared, "--model", "traced").get("endpoint_token_trace"),
                         "none")
        # --list-models and --debug-save-model-binding show it.
        line = {entry.split(":", 1)[0]: entry for entry in render_model_catalog(catalog).splitlines()}
        self.assertIn("auth=none, token_trace=sample_log_probs, source=user", line["traced"])
        self.assertNotIn("token_trace", line["chat"])
        self.assertEqual(
            _debug_entry(catalog.bind("chat-completions", "traced"))["endpoint"]["token_trace"],
            "sample_log_probs")
        self.assertIsNone(
            _debug_entry(catalog.bind("chat-completions", "chat"))["endpoint"]["token_trace"])

    def test_catalog_levels_are_checked(self):
        chat = ("chat-completions", _http("chat/completions"))
        for (api, route), lines, message in (
            (chat, "endpoint.token_trace = all\n",
             f"endpoint.token_trace must be {_LEVELS}, or null"),
            (chat, "endpoint.token_trace = Token_IDs\n",
             f"endpoint.token_trace must be {_LEVELS}, or null"),
            (("messages", _http("messages")),
             "endpoint.token_trace = none\nlimits.max_output_tokens = 100\n",
             "endpoint.token_trace requires endpoint.api chat-completions"),
            (("responses", _http("responses")), "endpoint.token_trace = token_ids\n",
             "endpoint.token_trace requires endpoint.api chat-completions"),
            (("claude-relay", "endpoint.model = claude-opus-5-5\n"),
             "endpoint.token_trace = token_ids\n",
             "endpoint.token_trace requires endpoint.api chat-completions"),
        ):
            with self.subTest(api=api, lines=lines):
                with self.assertRaisesRegex(ValueError, "^" + re.escape(
                        f"Invalid model catalog entry 'model.x' in user: {message}") + "$"):
                    parse_model_catalog(_ENTRY.format(api=api, route=route, lines=lines))
        # Python callers get the same checks from EndpointSpec.
        catalog = parse_model_catalog(_CATALOG)
        endpoint = catalog.get_model_spec("chat-completions", "chat").endpoint
        for value in ("all", True, 1):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "^endpoint.token_trace must be none, token_ids, "
                                "sample_log_probs, or prefix_log_probs$"):
                replace(endpoint, token_trace=value)
        responses = catalog.get_model_spec("responses", "plain").endpoint
        with self.assertRaisesRegex(
                ValueError, "^endpoint.token_trace requires the Chat Completions API$"):
            replace(responses, token_trace="token_ids")
        # Not part of the connection identity.
        self.assertEqual(replace(endpoint, token_trace="token_ids").connection_identity,
                         endpoint.connection_identity)

    def test_the_option_takes_a_level_on_chat_completions_only(self):
        parser = cli._build_parser()
        for level in TOKEN_TRACE_LEVELS:
            self.assertEqual(parser.parse_args(["--endpoint-token-trace", level])
                             .endpoint_token_trace, level)
        with redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
            parser.parse_args(["--endpoint-token-trace", "all"])
        self.assertIn("invalid choice: 'all'", stderr.getvalue())
        chat = ["--endpoint-url", "http://127.0.0.1:9/v1/chat/completions", "--endpoint-model",
                "fixture", "--endpoint-auth", "none"]
        args = parser.parse_args(["--endpoint-api", "chat-completions", *chat,
                                  "--endpoint-token-trace", "token_ids"])
        self.assertEqual(binding_from_namespace(args).endpoint.token_trace, "token_ids")
        responses = ["--endpoint-url", "http://127.0.0.1:9/v1/responses", "--endpoint-model",
                     "fixture", "--endpoint-auth", "none"]
        args = parser.parse_args(["--endpoint-api", "responses", *responses,
                                  "--endpoint-token-trace", "token_ids"])
        with self.assertRaisesRegex(ValueError, "^--endpoint-token-trace requires "
                                                "--endpoint-api chat-completions$"):
            binding_from_namespace(args)

    def test_the_vllm_fields_belong_to_the_setting(self):
        self.assertEqual(TOKEN_TRACE_FIELDS, {
            "return_token_ids", "logprobs", "top_logprobs", "return_tokens_as_token_ids",
            "prompt_logprobs", "skip_reading_prefix_cache", "prompt_token_ids", "token_ids"})
        message = "extra_sample_params.{0}: {0} belongs to endpoint.token_trace; set that instead"
        for lines, name in (("extra_sample_params.logprobs = true\n", "logprobs"),
                            ('extra_sample_params = {"Prompt_Logprobs": 0}\n', "Prompt_Logprobs")):
            with self.subTest(lines=lines), self.assertRaisesRegex(ValueError, "^" + re.escape(
                    "Invalid model catalog entry 'model.x' in user: " + message.format(name)) + "$"):
                parse_model_catalog(_ENTRY.format(api="chat-completions",
                                                  route=_http("chat/completions"), lines=lines))
        # Other APIs don't read them: an extension may still send them there.
        other = parse_model_catalog(_ENTRY.format(
            api="responses", route=_http("responses"), lines="extra_sample_params.logprobs = true\n"))
        self.assertEqual(dict(other.get_model_spec("responses", "x").extra_sample_params),
                         {"logprobs": True})
        # The launch option and per-call extensions are refused too.
        args = cli._build_parser().parse_args([
            "--endpoint-api", "chat-completions", "--endpoint-url",
            "http://127.0.0.1:9/v1/chat/completions", "--endpoint-model", "fixture",
            "--endpoint-auth", "none", "--extra-sample-params", '{"token_ids": [1]}'])
        with self.assertRaisesRegex(ValueError, "^Extra sample param token_ids belongs to "
                                                r"endpoint\.token_trace; set that instead\.$"):
            binding_from_namespace(args)
        binding = parse_model_catalog(_CATALOG).bind("chat-completions", "chat")
        with self.assertRaisesRegex(ValueError, "^Extra sample param return_token_ids belongs"):
            binding.with_extra_sample_params({"return_token_ids": True})

    def test_auto_gives_the_option_to_its_chat_completions_roles_only(self):
        catalog = parse_model_catalog(_CATALOG)
        for options, expected in (({}, {1: "sample_log_probs", -1: None}),
                                  ({"endpoint_token_trace": "token_ids"},
                                   {1: "token_ids", -1: None})):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                settings = resolve_config(overrides={"cwd": tmp}, catalog=catalog,
                                          role_models={1: "traced", -1: "quick"})
                settings.chat_completions_options = dict(options)
                session = auto._Session(Path(tmp) / "session", settings,
                                        model_factory=None, environment_factory=None)
                self.assertEqual({index: binding.endpoint.token_trace
                                  for index, binding in session.bindings.items()}, expected)
                # Main's config takes it from its binding, as its owner builds it.
                args = namespace(session.settings[1], catalog, binding=session.bindings[1])
                self.assertEqual(InteractionConfig.from_namespace(args).get("endpoint_token_trace"),
                                 expected[1])
                args = namespace(session.settings[-1], catalog, binding=session.bindings[-1])
                self.assertIsNone(InteractionConfig.from_namespace(args).get("endpoint_token_trace"))
                self.assertFalse((Path(tmp) / "session").exists())

    def test_auto_takes_the_option_at_launch(self):
        for flags, expected in (([], {}), (["--endpoint-token-trace", "token_ids"],
                                           {"endpoint_token_trace": "token_ids"})):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as tmp:
                seen = []

                def stop(settings, *_args):
                    seen.append(settings.chat_completions_options)
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


class TokenTraceConfigTests(unittest.TestCase):
    def test_the_key_takes_a_level_or_null(self):
        for level in TOKEN_TRACE_LEVELS:
            self.assertEqual(parse_config_literal("endpoint_token_trace", level), level)
        for literal in ("null", "None"):
            self.assertIsNone(parse_config_literal("endpoint_token_trace", literal))
        for literal in ("all", "true", "0", "Token_IDs"):
            with self.subTest(literal=literal), self.assertRaisesRegex(ConfigError, "^" + re.escape(
                    f"endpoint_token_trace requires {_LEVELS}, or null.") + "$"):
                parse_config_literal("endpoint_token_trace", literal)

    def test_config_changes_it_and_other_apis_refuse_it(self):
        catalog = parse_model_catalog(_CATALOG)
        config = _config_for(catalog, "--model", "traced")
        intent = user_tools.parse_user_tool("/config endpoint_token_trace prefix_log_probs")
        outcome = user_tools.run_config(config, json.loads(intent.arguments_json))
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.output, "# init: endpoint_token_trace = 'sample_log_probs'\n"
                                         "endpoint_token_trace = 'prefix_log_probs'")
        # The next turn's snapshot has it.
        self.assertEqual(config.snapshot().sample_params().token_trace, "prefix_log_probs")
        for model in ("quick", "plain", "relay"):
            with self.subTest(model=model):
                other = _config_for(catalog, "--model", model)
                self.assertIsNone(other.get("endpoint_token_trace"))
                self.assertIsNone(other.snapshot().sample_params().token_trace)
                with self.assertRaisesRegex(ConfigError, "^endpoint_token_trace applies only to "
                                                         r"the Chat Completions API\.$"):
                    other.set("endpoint_token_trace", "none")
                self.assertIsNone(other.set("endpoint_token_trace", None))

    def test_a_reboot_carries_it(self):
        catalog = parse_model_catalog(_CATALOG)
        config = _config_for(catalog, "--model", "chat")
        config.set("endpoint_token_trace", "token_ids")
        carried = cli._reboot_state(cli._UIState(reboot=cli._Reboot("/reboot")), config)
        self.assertEqual(carried["config"], {"endpoint_token_trace": "token_ids"})
        rebooted = cli._parse_reboot_state(json.loads(json.dumps(carried)))
        state, restarted = cli._UIState(), _config_for(catalog, "--model", "chat")
        cli._carry_over(state, restarted, rebooted)
        self.assertEqual(restarted.get("endpoint_token_trace"), "token_ids")
        self.assertFalse(any("didn't carry over" in item.text for item in state.displays))

    def test_sample_params_take_a_level_or_none(self):
        self.assertIsNone(SampleParams().token_trace)
        for level in TOKEN_TRACE_LEVELS:
            self.assertEqual(SampleParams(token_trace=level).token_trace, level)
        for value in ("all", True, ""):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, f"^token_trace must be {_LEVELS}, or None$"):
                SampleParams(token_trace=value)


class WhoAsksTests(unittest.TestCase):
    def test_the_loops_own_samples_ask_and_its_compactions_dont(self):
        seen, compacted = [], []

        class Model(_Model):
            def sample(self, context, *, tools=(), sample_params=None):
                seen.append(sample_params.token_trace)
                return super().sample(context, tools=tools, sample_params=sample_params)

        class Compactor:
            def compact(self, context, *, tools=(), sample_params=None):
                compacted.append(sample_params.token_trace)
                host.events.append(("compacted",))
                return _FakeCompaction(host)

        host = _Host()
        with mock.patch.object(kernel, "auto_compaction_due", side_effect=[True, False]), \
                mock.patch.object(kernel, "uses_host_auto_compaction", return_value=True), \
                mock.patch.object(kernel, "create_default_compactor", return_value=Compactor()), \
                mock.patch.object(kernel, "CompactionResult", _FakeCompaction):
            run_turn(_context(), Model(host, _answer("done")), _environment(host),
                     _config(endpoint_token_trace="sample_log_probs"), host)
        self.assertEqual((seen, compacted), (["sample_log_probs"], [None]))

    def test_run_compaction_drops_it_for_every_caller(self):
        # The CLI's /compact calls run_compaction directly.
        received = []

        class Compactor:
            def compact(self, context, *, tools=(), sample_params=None):
                received.append(sample_params)
                raise ValueError("stop here")

        params = SampleParams(token_trace="prefix_log_probs", max_output_tokens=50)
        with self.assertRaisesRegex(ValueError, "stop here"):
            run_compaction(InteractionContext((Message("user", "task"),)), object(),
                           CompactionSettings(mode="pi"), factory=lambda *a, **k: Compactor(),
                           sample_params=params)
        self.assertEqual(received, [replace(params, token_trace=None)])


if __name__ == "__main__":
    unittest.main()
