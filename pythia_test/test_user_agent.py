"""The User-Agent header: Pythia's default, and a model's own
(endpoint.user_agent in the model catalog, or --endpoint-user-agent)."""

from __future__ import annotations

import io
import re
import tempfile
import unittest
import urllib.request
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from pythia.interaction import (
    BUILTIN_MODEL_CATALOG, Init, InteractionContext, Message, USER_AGENT, auto, cli,
)
from pythia.interaction._auto_config import resolve_config
from pythia.interaction._model_binding_debug import _debug_entry
from pythia.interaction.model_catalog import MAX_USER_AGENT_CHARS, binding_from_namespace
from pythia.interaction.model_catalog_config import parse_model_catalog
from pythia.interaction.model_config import build_model, prepare_namespace
from pythia.interaction.model_config import render_model_catalog

_EXPECTED = (f"printable ASCII text, up to {MAX_USER_AGENT_CHARS} characters, without "
             "leading or trailing spaces")
# Each HTTP API's request path.
_PATHS = {"chat-completions": "chat/completions", "messages": "messages",
          "responses": "responses", "codex": "responses"}
_RELAY = ("[model.relay]\nendpoint.api = claude-relay\nendpoint.model = claude-opus-5-5\n")


def _catalog(agent=None, *, extra="", base=None):
    """An anonymous local entry on each HTTP API, named by its API, which sets
    ``agent`` as endpoint.user_agent unless it is None, then ``extra``."""
    text = "[catalog]\nversion = 4\n"
    for api, path in _PATHS.items():
        text += (f"[model.{api}]\nendpoint.api = {api}\n"
                 f"endpoint.url = http://127.0.0.1:9/v1/{path}\nendpoint.model = fixture\n"
                 + ("endpoint.auth = env:USER_AGENT_TEST_TOKEN\n" if api == "codex"
                    else "endpoint.auth = none\n")
                 + "limits.max_output_tokens = 100\n"
                 + ("" if agent is None else f"endpoint.user_agent = {agent}\n"))
    return parse_model_catalog(text + extra, **({} if base is None else {"base": base}))


def _bind(registry, *flags):
    return prepare_namespace(cli._build_parser().parse_args(list(flags)), registry).model_binding


class _Stop(Exception):
    """Raised by the opener once it has seen a request."""


class DefaultTests(unittest.TestCase):
    def test_the_default_names_pythia_with_pyproject_s_version(self):
        # The [project] table's version, read as text: tomllib needs Python 3.11,
        # and pyproject.toml allows 3.9.
        text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text("utf-8")
        project = re.search(r"(?ms)^\[project\]\n(.*?)(?=^\[|\Z)", text).group(1)
        version = re.search(r'(?m)^version = "([^"]+)"$', project).group(1)
        self.assertEqual(USER_AGENT, f"pythia/{version}")
        # Not urllib's own, which Cloudflare's browser-signature check refuses
        # with error 1010 at some endpoints, such as Tinker's.
        urllib_headers = dict(urllib.request.OpenerDirector().addheaders)
        self.assertNotEqual(USER_AGENT, urllib_headers["User-agent"])


class CatalogTests(unittest.TestCase):
    def test_an_entry_sets_the_whole_value_and_an_override_inherits_or_clears_it(self):
        registry = _catalog("lab-eval/2.0 (pythia; +https://example.test)")
        for api in _PATHS:
            with self.subTest(api=api):
                self.assertEqual(registry.bind(api, api).endpoint.user_agent,
                                 "lab-eval/2.0 (pythia; +https://example.test)")
        # JSON text too; without the key, the default.
        quoted = _catalog('"lab-eval/2.0"')
        self.assertEqual(quoted.bind("messages", "messages").endpoint.user_agent, "lab-eval/2.0")
        self.assertIsNone(_catalog().bind("messages", "messages").endpoint.user_agent)
        # An override keeps the entry's, and null brings back the default.
        header = "[catalog]\nversion = 4\n[model.chat-completions]\noverride = true\n"
        kept = parse_model_catalog(header + "limits.max_context_tokens = 1000\n",
                                   base=registry)
        self.assertEqual(kept.bind(name="chat-completions").endpoint.user_agent,
                         "lab-eval/2.0 (pythia; +https://example.test)")
        cleared = parse_model_catalog(header + "endpoint.user_agent = null\n", base=registry)
        self.assertIsNone(cleared.bind(name="chat-completions").endpoint.user_agent)
        # As does an override of a built-in model.
        builtin = parse_model_catalog("[catalog]\nversion = 4\n[model.codex-gpt-6-astra]\n"
                                      "override = true\nendpoint.api = codex\n"
                                      "endpoint.user_agent = lab-eval/2.0\n",
                                      base=BUILTIN_MODEL_CATALOG)
        self.assertEqual(builtin.bind("codex", "codex-gpt-6-astra").endpoint.user_agent,
                         "lab-eval/2.0")
        self.assertIsNone(BUILTIN_MODEL_CATALOG.bind("codex", "codex-gpt-6-astra")
                          .endpoint.user_agent)

    def test_the_value_is_printable_ascii_text_and_claude_relay_refuses_it(self):
        for value in ('""', '" lab-eval/2.0"', '"lab-eval/2.0 "', '"lab\\r\\nX-Injected: 1"',
                      '"lab\\teval"', '"lab-\\u00e9val"', "x" * (MAX_USER_AGENT_CHARS + 1)):
            with self.subTest(value=value[:20]), self.assertRaisesRegex(
                    ValueError, r"^Invalid model catalog entry 'model\.chat-completions' in "
                                rf"user: endpoint\.user_agent must be {_EXPECTED}, or null$"):
                _catalog(value)
        longest = "x" * MAX_USER_AGENT_CHARS
        self.assertEqual(_catalog(longest).bind(name="messages").endpoint.user_agent, longest)
        with self.assertRaisesRegex(ValueError, r"^Invalid model catalog entry 'model\.relay' in "
                                                r"user: endpoint\.user_agent does not apply to "
                                                r"endpoint\.api claude-relay$"):
            _catalog(extra=_RELAY + "endpoint.user_agent = lab-eval/2.0\n")

    def test_list_models_and_the_debug_binding_show_it(self):
        registry = _catalog("lab-eval/2.0 (pythia, test)", extra=_RELAY)
        lines = {line.split(":")[0]: line for line in render_model_catalog(registry).splitlines()}
        # Quoted, since the value may hold commas and spaces.
        self.assertIn(', user_agent="lab-eval/2.0 (pythia, test)", ', lines["chat-completions"])
        self.assertNotIn("user_agent", lines["relay"])
        self.assertNotIn("user_agent", render_model_catalog(_catalog()))
        self.assertEqual(_debug_entry(registry.bind(name="messages"))["endpoint"]["user_agent"],
                         "lab-eval/2.0 (pythia, test)")
        self.assertIsNone(_debug_entry(registry.bind(name="relay"))["endpoint"]["user_agent"])


class LaunchTests(unittest.TestCase):
    def test_the_option_replaces_the_catalog_s(self):
        registry = _catalog("catalog/1")
        self.assertEqual(_bind(registry, "--model", "messages").endpoint.user_agent, "catalog/1")
        self.assertEqual(_bind(registry, "--model", "messages", "--endpoint-user-agent",
                               "launch/2 (x)").endpoint.user_agent, "launch/2 (x)")
        self.assertEqual(_bind(_catalog(), "--model", "messages", "--endpoint-user-agent",
                               "launch/2").endpoint.user_agent, "launch/2")

    def test_the_option_takes_printable_ascii_text_and_claude_relay_refuses_it(self):
        parser = cli._build_parser()
        self.assertIn("--endpoint-user-agent TEXT", parser.format_help())
        for value in ("", " launch/2", "launch\nX-Injected: 1", "launch/\u00e9",
                      "x" * (MAX_USER_AGENT_CHARS + 1)):
            with self.subTest(value=value[:20]), redirect_stderr(io.StringIO()) as stderr, \
                    self.assertRaises(SystemExit):
                parser.parse_args(["--endpoint-user-agent", value])
            self.assertIn(f"argument --endpoint-user-agent: must be {_EXPECTED}",
                          stderr.getvalue())
        args = parser.parse_args(["--endpoint-api", "claude-relay", "--endpoint-model",
                                  "claude-opus-5-5", "--endpoint-user-agent", "launch/2"])
        with self.assertRaisesRegex(ValueError, "^--endpoint-user-agent does not apply to "
                                                "--endpoint-api claude-relay$"):
            binding_from_namespace(args)

    def test_auto_gives_the_option_to_its_http_roles(self):
        registry = _catalog("catalog/1", extra=_RELAY)
        for http, expected in (({}, {1: "catalog/1", -1: None}),
                               ({"endpoint_user_agent": "launch/2"}, {1: "launch/2", -1: None})):
            with self.subTest(http=http), tempfile.TemporaryDirectory() as tmp:
                settings = resolve_config(overrides={"cwd": tmp}, catalog=registry,
                                          role_models={1: "chat-completions", -1: "relay"})
                settings.http_options = dict(http)
                # The bindings resolve when the session is made; it isn't started.
                session = auto._Session(Path(tmp) / "session", settings,
                                        model_factory=None, environment_factory=None)
                self.assertEqual({index: binding.endpoint.user_agent
                                  for index, binding in session.bindings.items()}, expected)

    def test_auto_takes_the_option_at_launch(self):
        for flags, http in (([], {}),
                            (["--endpoint-user-agent", "launch/2"],
                             {"endpoint_user_agent": "launch/2"})):
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
                self.assertEqual(seen, [http])


class RequestTests(unittest.TestCase):
    def _sent(self, registry, api, *flags, compact=False):
        """The User-Agent of the first request that the model of ``api`` sends."""
        seen = []

        def opener(request, **_kwargs):
            seen.append(request.get_header("User-agent"))
            raise _Stop()

        with mock.patch.dict("os.environ", {"USER_AGENT_TEST_TOKEN": "token"}):
            args = prepare_namespace(cli._build_parser().parse_args(
                ["--model", api, *flags]), registry)
            model = build_model(args, opener=opener)
            context = InteractionContext((Init("session-1"), Message("user", "hello")))
            with self.assertRaises(_Stop):
                if compact:
                    model._compact_responses_v2(context, ())
                else:
                    model.sample(context)
        self.assertEqual(len(seen), 1)
        return seen[0]

    def test_each_http_api_sends_the_model_s_or_the_default(self):
        own, default = _catalog("lab-eval/2.0 (pythia)"), _catalog()
        for api in _PATHS:
            with self.subTest(api=api):
                self.assertEqual(self._sent(own, api), "lab-eval/2.0 (pythia)")
                self.assertEqual(self._sent(default, api), USER_AGENT)
                self.assertEqual(self._sent(default, api, "--endpoint-user-agent", "launch/2"),
                                 "launch/2")
        # Codex's remote compaction too.
        self.assertEqual(self._sent(own, "codex", compact=True), "lab-eval/2.0 (pythia)")
        self.assertEqual(self._sent(default, "codex", compact=True), USER_AGENT)


if __name__ == "__main__":
    unittest.main()
