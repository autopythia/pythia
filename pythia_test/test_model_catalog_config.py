from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pythia.interaction import (
    BUILTIN_MODEL_CATALOG, ConfigError, Environment, InteractionConfig,
    InteractionContext, Message, ModelCatalog, ModelSample, ResolvedSamplingParams,
    ResolvedSamplingOptions, SamplingParams, SamplingOptions,
    load_model_catalog, parse_model_catalog,
    sample_model,
)
from pythia.interaction import auto, cli, demo, model_catalog, responses
from pythia.interaction._auto_config import build_parser, load_saved_config, namespace, resolve_config
from pythia.interaction._catalog_session import (
    catalog_manifest_path, check_catalog_manifest, save_catalog_manifest,
)
from pythia.interaction.chat_completions import ChatCompletionsEndpoint, ChatCompletionsModel
from pythia.interaction.messages import MessagesEndpoint, MessagesModel
from pythia.interaction.model_config import build_model, prepare_namespace, supports_account_services
from pythia.interaction.save import SaveError
from pythia.interaction.model_catalog_config import MAX_CATALOG_BYTES


HEADER = "[catalog]\nversion = 1\n"
LOCAL = """
[model.local-max]
route.api = chat-completions
route.api_model = served-local
aliases = ["local-alias"]
route.provider = local
route.api_url = http://127.0.0.1:8000
route.auth_source = explicit
limits.auto_compact_context_tokens = 100000
limits.max_context_tokens = 150000
limits.max_output_tokens = 32000
request_params.thinking = {"type": "enabled", "budget_tokens": 1000}
request_params.reasoning_effort = "max"
"""
MESSAGE = """
[model.worker]
api = messages
api_model = served-messages
route.provider = remote
route.api_url = https://messages.example.test
route.auth_source = explicit
limits.auto_compact_context_tokens = 60000
limits.max_context_tokens = 80000
limits.max_output_tokens = 8000
messages.output_effort = high
"""
CODEX = """
[model.code-env]
route.api = codex
route.api_model = served-code
route.provider = custom
route.api_url = https://responses.example.test/v1
route.auth_source = environment
route.api_key_environment_variable = CATALOG_TEST_TOKEN
responses.reasoning_effort = high
"""


def catalog(text=LOCAL, **kwargs):
    return parse_model_catalog(HEADER + text, **kwargs)


def args_for(registry, *flags):
    return prepare_namespace(cli._build_parser().parse_args(flags), registry)


def context():
    return InteractionContext((Message("user", "hello"),))


_environment_get = os.environ.get


def no_credentials(key, default=None):
    if key == "CATALOG_TEST_TOKEN" or key == "CODEX_HOME" or key.endswith("_API_KEY"):
        raise AssertionError("unexpected credential lookup")
    return _environment_get(key, default)  # argparse/gettext may read locale settings.


class CatalogParserTests(unittest.TestCase):
    def test_new_entry_routes_aliases_and_preserves_structured_values(self):
        registry = catalog(LOCAL + MESSAGE)
        spec = registry.get_model_spec("chat-completions", "local-alias")
        self.assertEqual(spec.name, "local-max")
        self.assertEqual(spec.api_model, "served-local")
        self.assertEqual(spec.request_params["thinking"]["budget_tokens"], 1000)
        self.assertEqual(spec.limits.max_context_tokens, 150000)
        self.assertEqual(registry.bind(name="worker").api, "messages")
        self.assertIsNone(model_catalog.get_model_spec("chat-completions", "local-max"))
        self.assertIs(registry.get_model_route("chat-completions", "local-alias"), spec.route)
        self.assertIn(spec, registry.list_model_specs("chat-completions"))

    def test_parser_preserves_case_percent_comments_and_multiline_json(self):
        registry = catalog(LOCAL + '''request_params.CaseSensitive = "100% # literal ; text"
request_params.a.b = 7
request_params.nested = {
    "inner": [true, null, "value"]
    }
''')
        params = registry.bind(name="local-max").request_params
        self.assertEqual(params["CaseSensitive"], "100% # literal ; text")
        self.assertNotIn("casesensitive", params)
        self.assertEqual(params["a.b"], 7)
        self.assertEqual(params["nested"]["inner"], (True, None, "value"))

    def test_granular_override_preserves_other_fields_and_builtin_object(self):
        original = model_catalog.get_model_spec("codex", "gpt-6-astra")
        registry = catalog("""
[model.gpt-6-astra]
override = true
limits.auto_compact_context_tokens = 700000
responses.text_verbosity = medium
""")
        changed = registry.get_model_spec("codex", original.name)
        self.assertEqual(changed.limits.auto_compact_context_tokens, 700000)
        self.assertEqual(changed.responses.text_verbosity, "medium")
        self.assertEqual(changed.route, original.route)
        self.assertEqual(changed.api_model, original.api_model)
        self.assertEqual(changed.responses.reasoning_summary, original.responses.reasoning_summary)
        self.assertEqual(changed.limits.max_context_tokens, original.limits.max_context_tokens)
        self.assertEqual(original.limits.auto_compact_context_tokens, 872000)
        self.assertIs(registry.get_model_spec("codex", "gpt-6-astra-max"),
                      model_catalog.get_model_spec("codex", "gpt-6-astra-max"))

    def test_override_request_values_are_atomic_and_map_can_be_cleared(self):
        base = catalog()
        changed = catalog('''[model.local-max]
override = true
request_params.thinking = {"type": "disabled"}
''', base=base)
        params = changed.bind(name="local-max").request_params
        self.assertEqual(dict(params["thinking"]), {"type": "disabled"})
        self.assertEqual(params["reasoning_effort"], "max")
        literal_null = catalog('''[model.local-max]
override = true
request_params.thinking = null
''', base=base)
        self.assertIsNone(literal_null.bind(name="local-max").request_params["thinking"])
        cleared = catalog('''[model.local-max]
override = true
request_params = {}
''', base=base)
        self.assertEqual(dict(cleared.bind(name="local-max").request_params), {})

    def test_alias_replacement_and_nullable_field_clear(self):
        registry = catalog('''[model.claude-fable-5-1]
override = true
route.api = messages
aliases = ["my-fable"]
limits.auto_compact_context_tokens = null
''')
        self.assertIsNone(registry.get_model_spec("messages", "claude-fable-5.1"))
        self.assertEqual(registry.get_model_spec("messages", "my-fable").name, "claude-fable-5-1")
        self.assertIsNone(registry.bind("messages", "my-fable").limits.auto_compact_context_tokens)
        self.assertIsNotNone(model_catalog.get_model_spec("messages", "claude-fable-5.1"))

    def test_shorthands_duplicate_spellings_and_invalid_schema(self):
        valid = LOCAL.replace("route.api =", "api =").replace("route.api_model =", "api_model =")
        self.assertEqual(catalog(valid).bind(name="local-max").api_model, "served-local")
        bad = (
            LOCAL + "api = chat-completions\n",
            LOCAL + "api_model = other\n",
            LOCAL + "unknown = value\n",
            LOCAL.replace("limits.max_context_tokens = 150000", "limits.max_context_tokens = true"),
            LOCAL.replace('aliases = ["local-alias"]', 'aliases = "alias"'),
            LOCAL.replace('request_params.reasoning_effort = "max"', 'request_params.reasoning_effort = max'),
            LOCAL + "request_params = {}\n",
            LOCAL + "request_params.duplicate = {\"x\":1,\"x\":2}\n",
            LOCAL + "request_params.nonfinite = NaN\n",
            LOCAL + "request_params.nonfinite = 1e999\n",
            LOCAL.replace("route.auth_source = explicit", "route.auth_source = codex-login"),
            LOCAL.replace("route.api_url = http://127.0.0.1:8000", "route.api_url = https://user:secret@example.test"),
            LOCAL.replace("route.auth_source = explicit", "route.auth_source = environment\nroute.api_key_environment_variable = BAD-NAME"),
            MESSAGE.replace("60000", "49999"),
            MESSAGE + 'request_params.thinking = {"type":"enabled"}\n',
            '[model.gpt-6-astra]\noverride = true\nroute.api = null\n',
            '[model.missing]\noverride = true\n',
            '[model.claude-fable-5.1]\noverride = true\n',
            LOCAL.replace("local-max", "gpt-6-astra").replace("chat-completions", "codex"),
        )
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                catalog(text)

    def test_duplicate_sections_keys_defaults_and_versions_rejected(self):
        for text in (
            HEADER + LOCAL + LOCAL,
            HEADER + LOCAL + 'request_params.reasoning_effort = "low"\n',
            HEADER + "[DEFAULT]\noverride = true\n" + LOCAL,
            HEADER.replace("1", "2") + LOCAL,
            LOCAL, HEADER + "[other]\nx = 1\n", HEADER + "unknown = 1\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_model_catalog(text)

    def test_alias_collision_is_not_file_order_dependent(self):
        other = LOCAL.replace("[model.local-max]", "[model.other]")
        for text in (LOCAL + other, other + LOCAL):
            with self.assertRaisesRegex(ValueError, "collision"):
                catalog(text)

    def test_protected_fields_and_errors_do_not_disclose_values(self):
        for key in ("model", "messages", "stream", "tools", "max_tokens", "max_output_tokens",
                    "max_completion_tokens", "temperature", "seed", "api_key", "Authorization"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as error:
                    catalog(LOCAL + f'request_params.{key} = "VERY_SECRET_TOKEN"\n')
                self.assertNotIn("VERY_SECRET_TOKEN", str(error.exception))

    def test_catalog_and_nested_request_params_are_immutable(self):
        registry = catalog()
        binding = registry.bind(name="local-max")
        with self.assertRaises(FrozenInstanceError):
            registry.specs = ()
        with self.assertRaises(TypeError):
            registry.origins[("chat-completions", "local-max")] = "other"
        with self.assertRaises(TypeError):
            binding.request_params["thinking"]["type"] = "disabled"
        with self.assertRaises(TypeError):
            binding.spec.request_params["reasoning_effort"] = "low"

    def test_explicit_api_isolation_and_bare_name_ambiguity(self):
        registry = catalog(LOCAL.replace("local-max", "gpt-6-astra"))
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            registry.bind(name="gpt-6-astra")
        self.assertEqual(registry.bind("codex", "gpt-6-astra").api_model, "gpt-6-astra")
        self.assertEqual(registry.bind("chat-completions", "gpt-6-astra").api_model, "served-local")
        isolated = registry.bind("messages", "gpt-6-astra")
        self.assertIsNone(isolated.spec)
        self.assertFalse(isolated.request_params)
        with self.assertRaises(ValueError):
            catalog('[model.gpt-6-astra]\noverride = true\nsource = patch\n', base=registry)
        patched = catalog('[model.gpt-6-astra]\noverride = true\napi = codex-responses\nsource = patch\n', base=registry)
        self.assertEqual(patched.get_model_spec("codex", "gpt-6-astra").source, "patch")

    def test_discovery_is_explicit_bounded_and_missing_default_is_optional(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(Path, "home", return_value=Path(directory)):
            self.assertIs(load_model_catalog(), BUILTIN_MODEL_CATALOG)
            path = Path(directory) / ".pythia" / "model-catalog.ini"
            path.parent.mkdir()
            path.write_text(HEADER + LOCAL)
            loaded = load_model_catalog()
            self.assertEqual(loaded.bind(name="local-max").origin, str(path))
            path.write_text("malformed")
            with self.assertRaises(ValueError):
                load_model_catalog()
            self.assertIs(load_model_catalog(enabled=False), BUILTIN_MODEL_CATALOG)
            with self.assertRaises(ValueError):
                load_model_catalog(Path(directory) / "missing.ini")
            with self.assertRaises(ValueError):
                load_model_catalog(path, enabled=False)
            path.write_bytes(b"x" * (MAX_CATALOG_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "size limit"):
                load_model_catalog(path)

    def test_catalog_reader_rejects_special_files_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_model_catalog(directory)
            if hasattr(os, "mkfifo"):
                path = Path(directory) / "pipe.ini"
                os.mkfifo(path)
                with self.assertRaisesRegex(ValueError, "regular file"):
                    load_model_catalog(path)


class BoundRequestTests(unittest.TestCase):
    def test_catalog_route_model_and_policy_feed_one_config_and_request(self):
        args = args_for(catalog(), "--model", "local-alias")
        model = build_model(args)
        config = InteractionConfig.from_namespace(args)
        self.assertEqual(args.model_api, "chat-completions")
        self.assertEqual(config.get("auto_compact_tokens"), 100000)
        self.assertEqual(config.get("max_context_tokens"), 150000)
        self.assertIsNone(config.get("max_output_tokens"))  # The ceiling is not a budget.
        self.assertEqual(config.initial_values(), config.values())
        for params in (None, SamplingParams(), SamplingParams(max_output_tokens=77),
                       config.snapshot().sampling_params()):
            payload = model._build_request_payload(context(), (), params)
            self.assertEqual(payload["model"], "served-local")
            self.assertEqual(payload["reasoning_effort"], "max")
            self.assertEqual(payload["thinking"]["type"], "enabled")
        payload["thinking"]["type"] = "changed"
        self.assertEqual(model._build_request_payload(context(), (), None)["thinking"]["type"], "enabled")
        self.assertEqual(config.get("request_params")["thinking"]["type"], "enabled")
        with self.assertRaisesRegex(ConfigError, "launch-only"):
            config.set("request_params", {})
        rendered = json.loads(config.render(json_output=True))
        self.assertEqual(next(iter(rendered)), "__init__")
        self.assertEqual(rendered["request_params"], rendered["__init__"]["request_params"])

    def test_explicit_request_overlay_and_resolved_empty_never_fall_back(self):
        args = args_for(catalog(), "--model", "local-max", "--request-params",
                        '{"thinking":{"type":"disabled"},"custom":true}')
        cfg = InteractionConfig.from_namespace(args)
        model = build_model(args)
        payload = model._build_request_payload(context(), (), cfg.snapshot().sampling_params())
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertIs(payload["custom"], True)
        empty = model._build_request_payload(context(), (), ResolvedSamplingParams())
        self.assertNotIn("thinking", empty)
        self.assertNotIn("reasoning_effort", empty)

    def test_user_messages_limits_defaults_and_wire_model_do_not_lookup_builtins(self):
        args = args_for(catalog(MESSAGE), "--model", "worker")
        cfg = InteractionConfig.from_namespace(args)
        model = build_model(args)
        self.assertIsInstance(model, MessagesModel)
        self.assertEqual(model.endpoint.max_output_tokens, 8000)
        self.assertEqual(model.max_context_tokens, 80000)
        with mock.patch("pythia.interaction.messages.get_model_spec", side_effect=AssertionError("late lookup")):
            payload = model._build_request_payload(context(), (), cfg.snapshot().sampling_params())
        self.assertEqual(payload["model"], "served-messages")
        self.assertEqual(payload["output_config"], {"effort": "high"})
        self.assertEqual(payload["context_management"]["edits"][0]["trigger"]["value"], 60000)
        self.assertEqual(payload["max_tokens"], 8000)

    def test_user_responses_route_credentials_and_presets_are_bound(self):
        args = args_for(catalog(CODEX), "--model", "code-env")
        with mock.patch.dict("os.environ", {"CATALOG_TEST_TOKEN": "fake-token"}), \
                mock.patch.object(responses, "load_codex_auth", side_effect=AssertionError("ambient credentials")):
            model = build_model(args)
        self.assertEqual(model.endpoint.bearer_token, "fake-token")
        self.assertEqual(model.endpoint.api_url, "https://responses.example.test/v1")
        with mock.patch.object(responses, "get_model_spec", side_effect=AssertionError("late lookup")):
            payload, _ = model._build_request_payload(context(), (), None)
        self.assertEqual(payload["model"], "served-code")
        self.assertEqual(payload["reasoning"]["effort"], "high")
        self.assertFalse(model.supports_remote_compaction)
        self.assertFalse(supports_account_services(args))
        args.codex_home = "/must-not-read"
        with mock.patch.dict("os.environ", {"CATALOG_TEST_TOKEN": "fake-token"}), \
                mock.patch.object(responses, "load_codex_auth", side_effect=AssertionError("stale raw path")):
            self.assertEqual(build_model(args).endpoint.bearer_token, "fake-token")

    def test_forged_provider_label_cannot_grant_account_or_compaction_services(self):
        text = CODEX.replace("route.provider = custom", "route.provider = chatgpt").replace(
            "https://responses.example.test/v1", model_catalog.CODEX_RESPONSES_API_URL)
        args = args_for(catalog(text), "--model", "code-env")
        with mock.patch.dict("os.environ", {"CATALOG_TEST_TOKEN": "fake-token"}):
            model = build_model(args)
        self.assertFalse(supports_account_services(args))
        self.assertFalse(model.supports_remote_compaction)

    def test_chat_environment_reference_resolved_only_for_selected_model(self):
        text = LOCAL.replace("route.auth_source = explicit", "route.auth_source = environment\nroute.api_key_environment_variable = CATALOG_TEST_TOKEN")
        with mock.patch("os.environ.get", side_effect=no_credentials):
            registry = catalog(text)
            args = args_for(registry, "--model", "local-max")
            InteractionConfig.from_namespace(args)
        with mock.patch.dict("os.environ", {"CATALOG_TEST_TOKEN": "fake-token"}):
            self.assertEqual(build_model(args).endpoint.api_key, "fake-token")
        with mock.patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            build_model(args)

    def test_bindings_are_stable_when_file_or_other_catalog_changes(self):
        first = catalog()
        args = args_for(first, "--model", "local-max")
        cfg, model = InteractionConfig.from_namespace(args), build_model(args)
        second = catalog(LOCAL.replace("served-local", "other-wire").replace("100000", "110000"))
        self.assertEqual(second.bind(name="local-max").api_model, "other-wire")
        self.assertEqual(cfg.set("auto_compact_tokens", None), 100000)
        self.assertEqual(model._build_request_payload(context(), (), None)["model"], "served-local")
        self.assertIsNone(model_catalog.get_model_spec("chat-completions", "local-max"))

    def test_direct_library_endpoints_do_not_load_a_home_catalog(self):
        with mock.patch("os.open", side_effect=AssertionError("implicit catalog read")):
            endpoint = ChatCompletionsEndpoint("http://localhost:8000", "local-max",
                                               request_params={"reasoning_effort": "high"})
            payload = ChatCompletionsModel(endpoint)._build_request_payload(context(), (), None)
        self.assertEqual(payload["model"], "local-max")
        self.assertEqual(payload["reasoning_effort"], "high")

    def test_namespace_helpers_require_explicit_catalog_binding(self):
        args = cli._build_parser().parse_args(["--model", "local-max", "--model-catalog", "not-loaded.ini"])
        with self.assertRaisesRegex(ValueError, "Load and bind"):
            build_model(args)
        registry = catalog()
        self.assertEqual(build_model(args, catalog=registry).binding.api_model, "served-local")
        self.assertEqual(InteractionConfig.from_namespace(args, catalog=registry).get("auto_compact_tokens"), 100000)

    def test_unconfigured_chat_sampling_does_not_enable_reasoning(self):
        model = ChatCompletionsModel(ChatCompletionsEndpoint("http://localhost:8000", "literal"))
        for params in (None, SamplingParams(max_output_tokens=77), ResolvedSamplingParams(max_output_tokens=77)):
            payload = model._build_request_payload(context(), (), params)
            self.assertNotIn("thinking", payload)
            self.assertNotIn("reasoning_effort", payload)

    def test_api_inference_does_not_send_ambient_codex_login_to_a_proxy(self):
        args = cli._build_parser().parse_args(["--model", "gpt-6-astra", "--api-url", "http://localhost:8000"])
        with mock.patch.object(responses, "load_codex_auth", side_effect=AssertionError("credential read")):
            with self.assertRaisesRegex(ValueError, "explicit --api"):
                build_model(args)
        args.model_api = "chat-completions"
        self.assertIsInstance(build_model(args), ChatCompletionsModel)

    def test_params_aliases_and_keyword_conflicts(self):
        self.assertIs(SamplingParams, SamplingOptions)
        self.assertIs(ResolvedSamplingParams, ResolvedSamplingOptions)
        cfg = InteractionConfig()
        self.assertEqual(cfg.snapshot().sampling_params(), cfg.snapshot().sampling_options())
        models = [build_model(args_for(catalog(LOCAL + MESSAGE), "--model", name)) for name in ("local-max", "worker")]
        models.append(responses.CodexResponsesModel(
            responses.StreamingResponsesEndpoint("https://responses.example.test/v1", "wire", "fake")))
        for model in models:
            with self.subTest(model=type(model)), mock.patch.object(model, "_opener", side_effect=AssertionError("network")):
                with self.assertRaisesRegex(TypeError, "not both"):
                    model.sample(context(), options=None, sampling_params=None)
        with self.assertRaisesRegex(TypeError, "not both"):
            demo.run(models[0], Environment(), options=None, sampling_params=None)

    def test_custom_modern_and_legacy_models_work_without_retrying_typeerrors(self):
        calls = []
        class Modern:
            def sample(self, ctx, *, tools=(), sampling_params=None):
                calls.append(("modern", sampling_params))
                return ModelSample((Message("assistant", "done"),))
        class Legacy:
            def sample(self, ctx, *, tools=(), options=None):
                calls.append(("legacy", options))
                return ModelSample((Message("assistant", "done"),))
        params = SamplingParams(max_output_tokens=17)
        for model in (Modern(), Legacy()):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(demo.run(model, Environment(), sampling_params=params), "done")
        self.assertEqual([kind for kind, _ in calls], ["modern", "legacy"])
        self.assertTrue(all(value.max_output_tokens == 17 for _, value in calls))
        class Failing:
            def sample(self, ctx, *, tools=(), sampling_params=None):
                calls.append(("failed", sampling_params))
                raise TypeError("error inside sample")
        with self.assertRaisesRegex(TypeError, "inside sample"):
            sample_model(Failing(), context(), sampling_params=params)
        self.assertEqual([kind for kind, _ in calls].count("failed"), 1)


class AutoCatalogTests(unittest.TestCase):
    def test_context_api_can_disambiguate_a_common_model_without_repeating_it(self):
        registry = catalog(LOCAL.replace("local-max", "gpt-6-astra"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.json"
            path.write_text(json.dumps({"version": 1,
                "defaults": {"model": "gpt-6-astra"},
                "contexts": {"1": {"model_api": "codex"},
                             "2": {"model_api": "chat-completions"},
                             "-1": {"model_api": "codex"}},
            }))
            settings = resolve_config(path, catalog=registry)
            self.assertEqual(settings[1]["model"], "gpt-6-astra")
            self.assertEqual(namespace(settings[1], registry).model_binding.api_model, "gpt-6-astra")
            self.assertEqual(namespace(settings[2], registry).model_binding.api_model, "served-local")

    def test_context_api_inference_clearing_and_params_do_not_leak(self):
        registry = catalog(LOCAL + MESSAGE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.json"
            path.write_text(json.dumps({"version": 1, "contexts": {
                "2": {"model_api": None, "model": "worker"},
            }}))
            settings = resolve_config(path, {"model": "local-max", "model_api": "chat-completions",
                                            "request_params": {"custom": True}}, catalog=registry)
            main = InteractionConfig.from_namespace(namespace(settings[1], registry))
            worker_args = namespace(settings[2], registry)
            worker = InteractionConfig.from_namespace(worker_args)
            self.assertEqual(worker_args.model_api, "messages")
            self.assertEqual(worker.get("auto_compact_tokens"), 60000)
            self.assertEqual(worker.get("request_params"), {})
            self.assertTrue(main.get("request_params")["custom"])
            self.assertIsNone(settings[2]["model_api"])
            self.assertIsNone(settings[2]["request_params"])

    def test_missing_map_inherits_null_resets_and_nested_values_replace(self):
        registry = catalog()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.json"
            path.write_text(json.dumps({"version": 1, "contexts": {
                "2": {"request_params": None},
                "-1": {"request_params": {"thinking": {"type": "disabled"}}},
            }}))
            settings = resolve_config(path, {"model": "local-max", "request_params": {"custom": 1}}, catalog=registry)
            one, two, watch = (InteractionConfig.from_namespace(namespace(settings[i], registry)) for i in (1, 2, -1))
            self.assertEqual(one.get("request_params")["custom"], 1)
            self.assertNotIn("custom", two.get("request_params"))
            self.assertEqual(watch.get("request_params")["thinking"], {"type": "disabled"})
            self.assertEqual(watch.get("request_params")["custom"], 1)

    def test_api_clear_retains_selector_and_old_concrete_api_stays_scoped(self):
        saved = resolve_config(overrides={"model_api": "chat-completions", "model": "worker"})
        registry = catalog(MESSAGE)
        unchanged = resolve_config(saved=saved, catalog=registry)
        self.assertEqual(namespace(unchanged[1], registry).model_api, "chat-completions")
        self.assertIsNone(namespace(unchanged[1], registry).model_binding.spec)
        cleared = resolve_config(saved=saved, overrides={"model_api": None}, catalog=registry)
        self.assertEqual(cleared[1]["model"], "worker")
        self.assertEqual(namespace(cleared[1], registry).model_api, "messages")

    def test_different_provider_route_clears_inherited_credentials(self):
        registry = catalog(CODEX)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.json"
            path.write_text(json.dumps({"version": 1, "contexts": {"2": {"model": "code-env"}}}))
            settings = resolve_config(path, {"model_api": "codex", "model": "gpt-6-astra",
                                            "codex_home": directory}, catalog=registry)
            self.assertIsNone(settings[2]["codex_home"])
            self.assertEqual(settings[1]["codex_home"], directory)

    def test_save_roundtrip_keeps_raw_nulls_and_legacy_schema_upgrades(self):
        registry = catalog()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            settings = resolve_config(overrides={"model": "local-max"}, catalog=registry)
            document = {"version": 1, "contexts": {str(i): dict(s) for i, s in settings.items()}}
            for row in document["contexts"].values():
                row.pop("request_params")
            path.write_text(json.dumps(document))
            saved = load_saved_config(path)
            current = resolve_config(saved=saved, catalog=registry)
            self.assertIsNone(current[1]["request_params"])
            self.assertIsNone(current[1]["model_api"])
            self.assertEqual(namespace(current[1], registry).model_binding.api_model, "served-local")


class CatalogEntrypointTests(unittest.TestCase):
    def test_api_flag_alias_and_help_are_independent_of_home_catalog(self):
        for parser in (cli._build_parser(), demo._build_parser(), build_parser()):
            self.assertEqual(parser.parse_args(["--api", "messages"]).model_api,
                             parser.parse_args(["--model-api", "messages"]).model_api)
        for frontend in (cli, demo, auto):
            with mock.patch("os.open", side_effect=AssertionError("help must not read catalog")), \
                    redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                frontend.main(["--help"])
            self.assertEqual(result.exception.code, 0)

    def test_listing_loads_selected_catalog_without_tty_or_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.conf"
            path.write_text(HEADER + LOCAL + CODEX)
            for frontend in (cli, demo, auto):
                output = io.StringIO()
                with redirect_stdout(output), redirect_stderr(io.StringIO()), \
                        mock.patch("os.environ.get", side_effect=no_credentials):
                    result = frontend.main(["--list-models", "--model-catalog", str(path)])
                self.assertEqual(result, 0)
                self.assertIn("local-max", output.getvalue())
                self.assertIn("served-local", output.getvalue())

    def test_cli_headless_runs_catalog_model_and_persists_provenance(self):
        requests = []
        class Response:
            status = 200
            headers = {}
            def read(self):
                return b'{"choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}'
            def close(self):
                pass
        def opener(request, **kwargs):
            requests.append(json.loads(request.data))
            return Response()
        original = cli.build_model
        def model_factory(args):
            model = original(args)
            model._opener = opener
            return model
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.ini"
            log = Path(directory) / "log.jsonl"
            path.write_text(HEADER + LOCAL)
            with mock.patch.object(cli, "build_model", side_effect=model_factory), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = cli.main(["--headless", "--enable-default-tools", "false",
                                   "--model-catalog", str(path), "--model", "local-max",
                                   "--prompt", "hello", "--save", str(log)])
            self.assertEqual(result, 0)
            self.assertTrue(log.is_file())
            self.assertTrue(catalog_manifest_path(log).is_file())
            self.assertEqual(requests[0]["model"], "served-local")
            self.assertEqual(requests[0]["reasoning_effort"], "max")


class CatalogProvenanceTests(unittest.TestCase):
    def test_missing_and_changed_catalog_on_resume(self):
        old = catalog().bind(name="local-max")
        new = catalog(LOCAL.replace('"max"', '"high"')).bind(name="local-max")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            self.assertEqual(check_catalog_manifest(path, {"main": old}), ())
            save_catalog_manifest(path, {"main": old})
            self.assertEqual(check_catalog_manifest(path, {"main": old}), ())
            self.assertEqual(len(check_catalog_manifest(path, {"main": new})), 1)
            missing = BUILTIN_MODEL_CATALOG.bind(name="local-max")
            for reselected in ((), ("main",)):
                with self.assertRaisesRegex(SaveError, "unavailable"):
                    check_catalog_manifest(path, {"main": missing}, reselected=reselected)
            literal = BUILTIN_MODEL_CATALOG.bind("chat-completions", "served-local")
            self.assertTrue(check_catalog_manifest(path, {"main": literal}, reselected={"main"}))
            contents = path.read_text()
            self.assertNotIn("request_params", contents)
            self.assertNotIn("thinking", contents)

    def test_moving_api_requires_explicit_reselection(self):
        old = catalog().bind(name="local-max")
        moved = catalog(MESSAGE.replace("worker", "local-max")).bind(name="local-max")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            save_catalog_manifest(path, {"main": old})
            with self.assertRaises(SaveError):
                check_catalog_manifest(path, {"main": moved}, reselected={"main"})
            self.assertTrue(check_catalog_manifest(path, {"main": replace(moved, api_explicit=True)},
                                                   reselected={"main"}))
            path.write_text('{"version":1,"bindings":[]}')
            with self.assertRaises(SaveError):
                check_catalog_manifest(path, {"main": old})


if __name__ == "__main__":
    unittest.main()
