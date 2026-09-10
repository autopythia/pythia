from __future__ import annotations

from contextlib import ExitStack
from dataclasses import FrozenInstanceError
from dataclasses import replace
import importlib.util
import sys
import unittest
from unittest import mock

import pythia.interaction as interaction
from pythia.interaction import ChatCompletionsEndpoint
from pythia.interaction import ChatCompletionsModel
from pythia.interaction import CodexAuth
from pythia.interaction import CodexResponsesModel
from pythia.interaction import Init
from pythia.interaction import Message
from pythia.interaction import ModelConfigurationError
from pythia.interaction import ModelContext
from pythia.interaction import ModelLimits
from pythia.interaction import ModelRoute
from pythia.interaction import ModelSpec
from pythia.interaction import ResponsesDefaults
from pythia.interaction import StreamingResponsesEndpoint
from pythia.interaction import cli
from pythia.interaction import demo
from pythia.interaction import get_model_route
from pythia.interaction import get_model_spec
from pythia.interaction import list_model_specs
from pythia.interaction import messages
from pythia.interaction import model_catalog
from pythia.interaction import model_config
from pythia.interaction import responses
from pythia.interaction.model_config import build_model
from pythia.interaction.model_config import supports_account_services


_CODEX_PRESETS = (
    ("gpt-5.6-sol", "gpt-5.6-sol", {}, None, "chatgpt"),
    ("gpt-5.6-sol-medium", "gpt-5.6-sol", {"effort": "medium"}, None, "chatgpt"),
    ("gpt-5.6-sol-max", "gpt-5.6-sol", {"effort": "max"}, None, "chatgpt"),
    ("gpt-6-astra", "gpt-6-astra", {"summary": "auto"}, "low", "chatgpt"),
    ("gpt-6-astra-medium", "gpt-6-astra", {"effort": "medium", "summary": "auto"}, "low", "chatgpt"),
    ("gpt-6-astra-max", "gpt-6-astra", {"effort": "max", "summary": "auto"}, "low", "chatgpt"),
    ("muse-spark-1.3", "muse-spark-1.3-contributor", {}, None, "meta"),
    ("muse-spark-1.3-xhigh", "muse-spark-1.3-contributor", {"effort": "xhigh"}, None, "meta"),
)


class ModelCatalogTests(unittest.TestCase):
    def test_presets_preserve_routes_defaults_limits_and_request_payloads(self):
        self.assertEqual(tuple(spec.name for spec in list_model_specs("codex")),
                         tuple(row[0] for row in _CODEX_PRESETS))
        for name, wire_model, reasoning, verbosity, provider in _CODEX_PRESETS:
            with self.subTest(name=name):
                spec = get_model_spec("codex", name)
                self.assertIsInstance(spec, ModelSpec)
                self.assertEqual(spec.api_model, wire_model)
                self.assertEqual(spec.route.provider, provider)
                self.assertEqual(spec.responses, ResponsesDefaults(
                    reasoning_effort=reasoning.get("effort"),
                    reasoning_summary=reasoning.get("summary"), text_verbosity=verbosity,
                ))
                expected_url = (model_catalog.META_RESPONSES_API_URL if provider == "meta"
                                else model_catalog.CODEX_RESPONSES_API_URL)
                self.assertEqual(spec.route.api_url, expected_url)
                self.assertEqual(spec.route.api_key_environment_variable,
                                 "META_API_KEY" if provider == "meta" else None)
                self.assertEqual(spec.route.auth_source,
                                 "environment" if provider == "meta" else "codex-login")
                limits = ModelLimits() if provider == "meta" else ModelLimits(272_000, 872_000)
                self.assertEqual(spec.limits, limits)
                self.assertTrue(spec.source)

                model = CodexResponsesModel(model=name, auth=CodexAuth("FAKE"),
                                            identifier_factory=lambda: "fixed-turn")
                context = ModelContext((Init("session"), Message("user", "Hello.")))
                payload, _ = model._build_request_payload(context, (), None)
                self.assertEqual(model.endpoint.api_url, expected_url)
                self.assertEqual(model.endpoint.model, name)
                self.assertEqual(payload["model"], wire_model)
                self.assertEqual(payload.get("reasoning", {}), reasoning)
                self.assertEqual(payload.get("text"), None if verbosity is None else {"verbosity": verbosity})
                self.assertEqual(model.default_context_tokens, limits.default_context_tokens)
                self.assertEqual(model.max_context_tokens, limits.max_context_tokens)
                for field in ("default_context_tokens", "max_context_tokens", "max_output_tokens"):
                    self.assertNotIn(field, payload)

    def test_aliases_are_identical_but_effort_presets_are_distinct(self):
        fable = get_model_spec("messages", "claude-fable-5-1")
        self.assertIs(get_model_spec("messages", " claude-fable-5.1 "), fable)
        self.assertEqual(fable.aliases, ("claude-fable-5.1",))
        self.assertEqual(fable.limits, ModelLimits(max_context_tokens=1_000_000, max_output_tokens=128_000))
        self.assertEqual(fable.route.auth_source, "environment")
        self.assertEqual(fable.route.api_key_environment_variable, "ANTHROPIC_API_KEY")
        self.assertIsNone(fable.responses)
        self.assertEqual(len(list_model_specs()), 9)
        for base_name, preset_name in (("gpt-5.6-sol", "gpt-5.6-sol-medium"),
                                       ("gpt-6-astra", "gpt-6-astra-max"),
                                       ("muse-spark-1.3", "muse-spark-1.3-xhigh")):
            base = get_model_spec("codex", base_name)
            preset = get_model_spec("codex", preset_name)
            self.assertIsNot(base, preset)
            self.assertIs(base.limits, preset.limits)
            self.assertIs(base.route, preset.route)
            self.assertNotEqual(base.responses, preset.responses)
        self.assertIs(get_model_spec("codex-responses", "gpt-6-astra"),
                      get_model_spec("codex", "gpt-6-astra"))

    def test_unknown_models_and_other_profiles_do_not_inherit_presets(self):
        for profile in ("responses", "chat-completions"):
            self.assertEqual(list_model_specs(profile), ())
            for spec in list_model_specs():
                for name in (spec.name, *spec.aliases):
                    self.assertIsNone(get_model_spec(profile, name))
        for name in (None, "", "unknown", "GPT-6-ASTRA", "gpt-6-astra-low",
                     "gpt-5.6-sol-high", "muse-spark-1.3-max"):
            with self.subTest(name=name):
                self.assertIsNone(get_model_spec("codex", name))
                self.assertEqual(get_model_route("codex", name).api_url, model_catalog.CODEX_RESPONSES_API_URL)
        for name in ("claude-fable-5.2", "claude-fable-5-1-20260901", "gpt-6-astra"):
            self.assertIsNone(get_model_spec("messages", name))
        self.assertIsNone(get_model_spec("codex", "claude-fable-5.1"))

    def test_generic_responses_and_chat_completions_remain_pass_through(self):
        for spec in list_model_specs():
            for name in (spec.name, *spec.aliases):
                with self.subTest(name=name):
                    context = ModelContext((Message("user", "Hello."),))
                    model = CodexResponsesModel(StreamingResponsesEndpoint(
                        api_url="https://proxy.example.test/v1", model=name, bearer_token="FAKE",
                    ))
                    payload, state = model._build_request_payload(context, (), None)
                    self.assertEqual(payload["model"], name)
                    self.assertNotIn("reasoning", payload)
                    self.assertNotIn("text", payload)
                    self.assertIsNone(model.default_context_tokens)
                    self.assertIsNone(model.max_context_tokens)
                    self.assertNotIn("session_id", model._build_headers(state))
                    chat = ChatCompletionsModel(ChatCompletionsEndpoint("http://localhost", model=name))
                    self.assertEqual(chat._build_request_payload(context, (), None)["model"], name)

    def test_catalog_is_immutable_and_rejects_colliding_selectors(self):
        spec = get_model_spec("messages", "claude-fable-5-1")
        for obj, field, value in ((spec, "name", "other"), (spec.limits, "max_context_tokens", 1),
                                  (spec.route, "api_url", "https://other.test")):
            with self.assertRaises(FrozenInstanceError):
                setattr(obj, field, value)
        with self.assertRaises(TypeError):
            model_catalog._MODEL_INDEX[("messages", "other")] = spec
        for specs in (
            (spec, spec),
            (spec, replace(spec, name="other")),  # Colliding alias.
            (spec, replace(spec, name="claude-fable-5.1", aliases=())),
            (replace(spec, aliases=(spec.name,)),),
        ):
            with self.assertRaisesRegex(ValueError, "duplicate model selector"):
                model_catalog._build_index(specs)
        self.assertEqual(len(model_catalog._build_index((spec, replace(spec, profile="responses")))), 4)

    def test_catalog_validation(self):
        for field in ("default_context_tokens", "max_context_tokens", "max_output_tokens"):
            for value in (True, 0, -1, 1.5, "100"):
                with self.assertRaises(ValueError):
                    ModelLimits(**{field: value})
        with self.assertRaises(ValueError):
            ModelLimits(default_context_tokens=200, max_context_tokens=100)
        spec = get_model_spec("messages", "claude-fable-5-1")
        for fields in ({"name": ""}, {"api_model": "bad\nname"}, {"profile": "bad"},
                       {"aliases": "alias"}, {"aliases": [" "]},
                       {"responses": ResponsesDefaults(reasoning_effort="max")}):
            with self.assertRaises((TypeError, ValueError)):
                replace(spec, **fields)
        with self.assertRaises(TypeError):
            ModelRoute("meta", "https://example.test", "environment")
        with self.assertRaises(ValueError):
            ModelRoute("meta", "https://example.test", "explicit", "META_API_KEY")
        with self.assertRaises(ValueError):
            get_model_spec("unknown-profile", "model")

    def test_import_and_lookup_do_not_access_credentials_or_network(self):
        module_name = "_isolated_interaction_model_catalog"
        module_spec = importlib.util.spec_from_file_location(module_name, model_catalog.__file__)
        module = importlib.util.module_from_spec(module_spec)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {module_name: module}))
            for target in ("os.environ.get", "builtins.open", "pathlib.Path.read_text",
                           "urllib.request.urlopen"):
                stack.enter_context(mock.patch(target, side_effect=AssertionError("unexpected effect")))
            module_spec.loader.exec_module(module)
            self.assertEqual(len(module.list_model_specs()), 9)
            self.assertEqual(module.get_model_route("codex", "muse-spark-1.3").api_key_environment_variable,
                             "META_API_KEY")
            self.assertIsNone(module.get_model_spec("responses", "gpt-6-astra"))

    def test_public_url_constants_keep_existing_import_paths(self):
        for name in ("CODEX_RESPONSES_API_URL", "META_RESPONSES_API_URL", "OPENAI_RESPONSES_API_URL"):
            self.assertEqual(getattr(interaction, name), getattr(model_catalog, name))
            self.assertEqual(getattr(responses, name), getattr(model_catalog, name))
        self.assertEqual(interaction.ANTHROPIC_MESSAGES_API_URL, model_catalog.ANTHROPIC_MESSAGES_API_URL)
        self.assertEqual(messages.ANTHROPIC_MESSAGES_API_URL, model_catalog.ANTHROPIC_MESSAGES_API_URL)
        self.assertEqual(model_config.ANTHROPIC_MESSAGES_API_URL, model_catalog.ANTHROPIC_MESSAGES_API_URL)
        self.assertEqual(model_config.CODEX_RESPONSES_API_URL, model_catalog.CODEX_RESPONSES_API_URL)

    def test_cli_and_demo_help_list_catalog_presets_and_aliases(self):
        for frontend in (cli, demo):
            parser = frontend._build_parser()
            model_help = next(action.help for action in parser._actions if action.dest == "model")
            for spec in list_model_specs():
                for name in (spec.name, *spec.aliases):
                    self.assertIn(name, model_help)
            self.assertIn("META_API_KEY", model_help)
            self.assertNotIn("gpt-6-astra-low", model_help)
            self.assertNotIn("muse-spark-1.3-max", model_help)
            self.assertEqual(parser.parse_args(["--model", "future-model"]).model, "future-model")


class CatalogAuthParityTests(unittest.TestCase):
    def test_meta_environment_defaults_and_explicit_auth_url_and_file_precedence(self):
        with mock.patch.dict("os.environ", {"META_API_KEY": " env-token "}, clear=True):
            model = CodexResponsesModel(model="muse-spark-1.3")
            self.assertEqual(model.endpoint.bearer_token, "env-token")
            with mock.patch.object(responses, "_load_default_model_auth",
                                   side_effect=AssertionError("must use explicit auth")):
                model = CodexResponsesModel(
                    model="muse-spark-1.3", auth=CodexAuth("explicit-token"),
                    api_url="https://proxy.example.test", request_timeout_seconds=7,
                )
            self.assertEqual(model.endpoint.bearer_token, "explicit-token")
            self.assertEqual(model.endpoint.api_url, "https://proxy.example.test")
            self.assertEqual(model.endpoint.request_timeout_seconds, 7)
            for options in ({"codex_home": "/fake-home"}, {"auth_file": "/fake-auth.json"}):
                with self.subTest(options=options):
                    with mock.patch.object(responses, "load_codex_auth", return_value=CodexAuth("file-token")) as load:
                        model = CodexResponsesModel(model="muse-spark-1.3", **options)
                    load.assert_called_once_with(codex_home=options.get("codex_home"),
                                                 auth_file=options.get("auth_file"))
                    self.assertEqual(model.endpoint.bearer_token, "file-token")
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ModelConfigurationError, "META_API_KEY"):
                CodexResponsesModel(model="muse-spark-1.3-xhigh")

    def test_frontend_messages_auth_and_chat_defaults_do_not_cross_profiles(self):
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "anthropic-token",
                                            "META_API_KEY": "meta-token"}, clear=True):
            for flags, expected in (([], "anthropic-token"), (["--api-key", "explicit"], "explicit")):
                args = demo._build_parser().parse_args([
                    "--model-api", "messages", "--model", "claude-fable-5.1", *flags,
                ])
                self.assertEqual(build_model(args).endpoint.api_key, expected)
            args = demo._build_parser().parse_args(["--model", "muse-spark-1.3"])
            model = build_model(args)
            self.assertEqual(model.endpoint.api_url, "http://127.0.0.1:8000")
            self.assertIsNone(model.endpoint.api_key)
            self.assertNotIn("anthropic-token", repr(list_model_specs()))
            self.assertNotIn("meta-token", repr(list_model_specs()))

    def test_account_services_require_actual_trusted_route_not_just_model_identity(self):
        for api in ("codex", "codex-responses", "messages", "chat-completions"):
            for name in ("gpt-6-astra", "unknown-model", "muse-spark-1.3", "muse-spark-1.3-xhigh"):
                for url in (None, model_catalog.CODEX_RESPONSES_API_URL,
                            model_catalog.CODEX_RESPONSES_API_URL + "/",
                            "https://proxy.example.test", model_catalog.META_RESPONSES_API_URL):
                    with self.subTest(api=api, name=name, url=url):
                        args = cli._build_parser().parse_args(["--model-api", api, "--model", name])
                        args.api_url = url
                        expected = (api in {"codex", "codex-responses"}
                                    and not name.startswith("muse-")
                                    and (url is None or url.rstrip("/") == model_catalog.CODEX_RESPONSES_API_URL))
                        self.assertEqual(supports_account_services(args), expected)


if __name__ == "__main__":
    unittest.main()
