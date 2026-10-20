"""Shared argument defaults and provider construction for interaction frontends."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import warnings

from ._prompt import add_prompt_arguments
from ._request_gate import cancellable_urlopen
from .chat_completions import ChatCompletionsEndpoint
from .chat_completions import ChatCompletionsModel
from .compaction import compaction_methods
from .compaction import normalize_compaction_mode
from .compaction import DEFAULT_KEEP_RECENT_TOKENS
from .messages import MessagesEndpoint
from .messages import MessagesModel
from .messages import MessagesPromptCaching
from .messages import MessagesServerCompaction
from .messages import MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS
from .model import Model
from .model_catalog import list_model_specs
from .model_catalog import binding_from_namespace
from .model_catalog import COMPACTION_DEFAULT_FIELDS
from .model_catalog import MODEL_TIMEOUT_FIELDS
from .model_catalog import SAMPLE_PARAM_DEFAULT_FIELDS
from .model_catalog import parse_json_value, freeze_extra_sample_params, thaw_json
from .codex_auth import CodexAuth, _resolve_auth_file
from .model_catalog_config import load_model_catalog
from .responses import CodexResponsesModel
from .responses import ResponsesModel
from .runtime_config import RENAMED_CONFIG_KEYS
from .timeouts import DEFAULT_CLAUDE_RELAY_GENERATION_TIMEOUT_SECONDS
from .timeouts import DEFAULT_CLAUDE_RELAY_IDLE_TIMEOUT_SECONDS
from .timeouts import DEFAULT_CLAUDE_RELAY_PARKED_TIMEOUT_SECONDS
from .timeouts import DEFAULT_CLAUDE_RELAY_STARTUP_TIMEOUT_SECONDS
from .timeouts import DEFAULT_CLAUDE_RELAY_STOP_TIMEOUT_SECONDS
from .timeouts import DEFAULT_REQUEST_TIMEOUT_SECONDS


DEFAULT_SAVE_PATH = Path("interaction.jsonl")
# The CLI's --save names a save prefix (see _cli_save): interaction.jsonl and
# interaction._private/.
DEFAULT_SAVE_PREFIX = Path("interaction")
_CLAUDE_RELAY_TIMEOUT_DEFAULTS = {
    "generation": DEFAULT_CLAUDE_RELAY_GENERATION_TIMEOUT_SECONDS,
    "parked": DEFAULT_CLAUDE_RELAY_PARKED_TIMEOUT_SECONDS,
    "startup": DEFAULT_CLAUDE_RELAY_STARTUP_TIMEOUT_SECONDS,
    "stop": DEFAULT_CLAUDE_RELAY_STOP_TIMEOUT_SECONDS,
    "idle": DEFAULT_CLAUDE_RELAY_IDLE_TIMEOUT_SECONDS,
}

# Invocation-only, never restored from conversation/auto config snapshots.
CLAUDE_RELAY_FIELDS = (
    "claude_relay_launcher", "claude_relay_socket", "claude_relay_server_uid",
    "claude_relay_cli_version", "claude_relay_tool_id_pointer",
    "claude_relay_generation_timeout", "claude_relay_parked_timeout",
    "claude_relay_startup_timeout", "claude_relay_stop_timeout",
    "claude_relay_idle_timeout",
)


def relay_endpoint(args, binding):
    from .claude_relay import ClaudeRelayEndpoint
    def value(name, env=None, default=None):
        supplied = getattr(args, "claude_relay_" + name, None)
        return supplied if supplied is not None else os.environ.get(env, default) if env else default
    raw_uid = value("server_uid", "CLAUDE_RELAY_SERVER_UID")
    try:
        uid = int(raw_uid) if raw_uid is not None and not isinstance(raw_uid, bool) else None
    except (ValueError, TypeError):
        raise ValueError("Claude Relay expected server UID must be an integer") from None
    if getattr(args, "max_output_tokens", None) is not None or getattr(args, "compaction_pi_max_output_tokens", None) is not None:
        # TODO(claude-relay output-budgets): remove this guard only alongside the
        # verified _sampling/runtime mapping and relay/wrapper capability checks.
        raise ValueError("Claude Relay does not yet support explicit output-token budgets")
    if getattr(args, "request_timeout_seconds", None) is not None:
        raise ValueError("Use --claude-relay-generation-timeout or the model catalog's "
                         "timeouts.generation_seconds; HTTP request timeouts do not apply")

    def deadline(name):
        # Launch option, then the bound model's catalog value, then the default.
        configured = getattr(binding.timeouts, f"{name}_seconds")
        return value(f"{name}_timeout", default=(
            _CLAUDE_RELAY_TIMEOUT_DEFAULTS[name] if configured is None else configured))

    return ClaudeRelayEndpoint(
        model=binding.endpoint.model,
        launcher=value("launcher", "CLAUDE_RELAY_LAUNCHER"),
        socket_path=value("socket", "CLAUDE_RELAY_SOCKET"), server_uid=uid,
        expected_version=value("cli_version", "CLAUDE_RELAY_CLI_VERSION"),
        tool_id_pointer=value("tool_id_pointer", default="/params/_meta/claudecode~1toolUseId"),
        generation_timeout_seconds=deadline("generation"),
        parked_timeout_seconds=deadline("parked"),
        startup_timeout_seconds=deadline("startup"),
        stop_timeout_seconds=deadline("stop"),
        idle_timeout_seconds=deadline("idle"), binding=binding,
    )


def compaction_mode_argument(value: str) -> str:
    """``--compaction-mode``: a compaction sequence, such as ``coldstart,pi:2``."""
    try:
        return normalize_compaction_mode(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


COMPACTION_MODE_HELP = (
    "compaction methods to try in order, each with its attempts (1 when "
    "omitted), such as coldstart,warmpi:2. pi summarizes older context on "
    "the host and keeps recent items verbatim; warmpi makes pi's checkpoint "
    "from the conversation's own request, which reads its prompt cache; "
    "provider uses Codex remote compaction or Anthropic server-side "
    "compaction (on Messages only first); coldstart makes no model request "
    "and points the model to the saved log, and fails when the save doesn't "
    "hold the context. legacydefault is the route's own method, the default "
    "until October 2026: provider on the official ChatGPT/Codex route, pi "
    "elsewhere"
)


def _boolean_argument(value: str) -> bool:
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError("expected True or False")
    normalized = value.casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected True or False")


def _option(key: str) -> str:
    return "--" + key.replace("_", "-")


class _RenamedOption(argparse.Action):
    """A renamed option's old name: it sets the new option, with a warning."""

    def __init__(self, option_strings, dest, *, new_option, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
        self.new_option = new_option

    def __call__(self, parser, namespace, values, option_string=None):
        # FutureWarning, not DeprecationWarning: Python shows it by default.
        warnings.warn(f"{option_string} is deprecated; use {self.new_option}.",
                      FutureWarning)
        setattr(namespace, self.dest, values)


def add_renamed_options(parser: argparse.ArgumentParser) -> None:
    """Add the old names of renamed options (RENAMED_CONFIG_KEYS), hidden.

    Each sets its new option's destination, with a FutureWarning that names
    the new option. All three take an integer. They set no default, so the
    new option's default stands.
    """
    for old, new in RENAMED_CONFIG_KEYS.items():
        parser.add_argument(
            _option(old), action=_RenamedOption, new_option=_option(new), dest=new,
            type=int, default=argparse.SUPPRESS, metavar="N", help=argparse.SUPPRESS,
        )


def _save_path_argument(value: str) -> Path:
    # Validate before Path("") can turn an empty argument into the current dir.
    if not value.strip() or "\x00" in value or value == "-":
        raise argparse.ArgumentTypeError(
            "expected a non-empty file path (--save does not support stdin/stdout)"
        )
    return Path(value)


def resolve_save_path(path: Path) -> Path:
    """Anchor a frontend's log to its launch directory without resolving links."""
    selected = path.expanduser().absolute()
    if selected.exists() and not selected.is_file():
        raise ValueError(f"save destination must be a regular file: {selected}")
    if not selected.parent.is_dir():
        raise ValueError(f"save parent must be an existing directory: {selected.parent}")
    return selected


def initial_model_name(model: Optional[Model]) -> Optional[str]:
    """Return the configured model name when exposed by an adapter."""
    name = getattr(getattr(model, "endpoint", None), "model", None)
    return name if isinstance(name, str) and name.strip() else None


def supports_account_services(args: argparse.Namespace) -> bool:
    """Only the official ChatGPT endpoint supports initial login/quota tools."""
    if not getattr(args, "model", None):
        return False
    return binding_from_namespace(args).supports_account_services


def prepare_namespace(args, catalog=None):
    """Resolve without changing raw launch/saved input or rereading a catalog."""
    api = getattr(args, "model_api", None)
    if api is not None and api not in {"chat-completions", "messages", "codex", "responses", "claude-relay"}:
        raise ValueError(f"unsupported model API: {api!r}")
    binding = binding_from_namespace(args, catalog)
    endpoint = binding.endpoint
    if (endpoint.auth == "codex-login" and not endpoint.is_official_codex
            and not binding.api_explicit):
        raise ValueError(
            "A custom Codex destination requires explicit --endpoint-api codex; "
            "use --endpoint-api chat-completions for a local chat model."
        )
    if binding.api not in {"chat-completions", "messages", "codex", "responses", "claude-relay"}:
        raise ValueError(f"The frontend does not support the {binding.api} API.")
    if not getattr(args, "_endpoint_prepared", False):
        if getattr(args, "codex_home", None) is not None or getattr(args, "codex_auth_file", None) is not None:
            if endpoint.api != "codex":
                raise ValueError("Endpoint auth paths require --endpoint-api codex")
            if endpoint.auth != "codex-login":
                raise ValueError("Codex credential files require endpoint-auth codex-login.")
    if endpoint.auth == "codex-login" and endpoint.auth_file is None:
        endpoint = replace(endpoint, auth_file=str(_resolve_auth_file(
            codex_home=getattr(args, "codex_home", None), auth_file=getattr(args, "codex_auth_file", None),
        ).resolve()))
        binding = replace(binding, endpoint=endpoint)
    values = vars(args).copy()
    values.update(model_api=binding.api, model_binding=binding, _endpoint_prepared=True,
                  model=binding.selector if binding.selector is not None else endpoint.model)
    return argparse.Namespace(**values)


def frontend_catalog(args):
    return load_model_catalog(
        getattr(args, "model_catalog", None),
        enabled=not getattr(args, "no_user_model_catalog", False),
    )


def render_model_catalog(catalog):
    lines = []
    for spec in catalog.specs:
        aliases = f" (aliases: {', '.join(spec.aliases)})" if spec.aliases else ""
        origin = catalog.origins[(spec.endpoint.api, spec.name)]
        settings = "".join(f", {setting}" for setting in (
            *_request_settings(spec), *_sample_param_settings(spec), *_timeout_settings(spec),
            *(f"compaction.{name}={value}" for name, value in _compaction_settings(spec.compaction))))
        lines.append(f"{spec.name}{aliases}: api={spec.endpoint.api}, model={spec.endpoint.model}, "
                     f"url={spec.endpoint.url}, auth={spec.endpoint.auth}, source={origin}{settings}")
    if catalog.auto_models:
        lines.append("[auto] " + ", ".join(
            f"{role}.model = {name}" for role, name in catalog.auto_models.items()))
    compaction = _compaction_settings(catalog.compaction)
    if compaction:
        lines.append("[compaction] " + ", ".join(f"{name} = {value}" for name, value in compaction))
    return "\n".join(lines)


def _compaction_settings(defaults) -> list:
    """A catalog's set compaction values, as (name, value), in the catalog's names."""
    return [(name, getattr(defaults, name)) for name in COMPACTION_DEFAULT_FIELDS
            if getattr(defaults, name) is not None]


def _extra_sample_params_argument(text):
    try:
        value = parse_json_value(text)
        return None if value is None else thaw_json(freeze_extra_sample_params(value))
    except ValueError:
        raise argparse.ArgumentTypeError("expected a JSON object of extra sample params or null") from None


def add_catalog_arguments(parser, *, suppress_extra_sample_params=False):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model-catalog", type=Path,
                       help="INI user catalog (default: ~/.pythia/model-catalog.ini)")
    group.add_argument("--no-user-model-catalog", action="store_true", help="use only the built-in model catalog")
    parser.add_argument("--list-models", action="store_true", help="list the selected catalog without loading credentials")
    parser.add_argument(
        "--debug-save-model-binding",
        action="store_true",
        help=(
            "write an opt-in resolved model-binding snapshot next to the save; "
            "may include endpoint and request configuration"
        ),
    )
    parser.add_argument("--extra-sample-params", type=_extra_sample_params_argument,
                        default=argparse.SUPPRESS if suppress_extra_sample_params else None,
                        help=("JSON object of model-specific request-body extensions, "
                              "overlaid on the catalog's extra_sample_params; launch-only"))


def add_endpoint_arguments(parser, *, auto=False):
    default = argparse.SUPPRESS if auto else None
    parser.add_argument("--endpoint-api", dest="model_api",
                        choices=("chat-completions", "messages", "codex", "responses", "claude-relay"),
                        default=default, help="endpoint API; omitted infers a unique catalog selection")
    parser.add_argument("--endpoint-url", default=default, help="complete model POST URL")
    parser.add_argument("--endpoint-model", default=default, help="wire model ID (not a catalog selector)")
    parser.add_argument("--endpoint-auth", default=default, help="none, env:NAME, codex-login, supplied, or runtime (claude-relay)")
    parser.add_argument("--claude-relay-launcher", default=argparse.SUPPRESS, help="absolute standalone relay client path; launch-only")
    parser.add_argument("--claude-relay-socket", default=argparse.SUPPRESS, help="broker UNIX socket; launch-only")
    parser.add_argument("--claude-relay-server-uid", type=int, default=argparse.SUPPRESS, help="expected non-root broker UID")
    parser.add_argument("--claude-relay-cli-version", default=argparse.SUPPRESS, help="required pinned native version (not live-verified by Pythia)")
    parser.add_argument("--claude-relay-tool-id-pointer", default=argparse.SUPPRESS,
                        help="JSON pointer into MCP params._meta; default /params/_meta/claudecode~1toolUseId")
    for name, fallback in _CLAUDE_RELAY_TIMEOUT_DEFAULTS.items():
        what = (f"Claude Relay {name} deadline in seconds" if name != "idle" else
                "seconds a Claude Relay process waits for the next turn after a final answer, "
                "counted from the start of the answer's request; 0 keeps none")
        parser.add_argument(f"--claude-relay-{name}-timeout", type=float, default=argparse.SUPPRESS,
                            help=(f"{what} (default: the model catalog's timeouts.{name}_seconds, "
                                  f"else {fallback:g}); launch-only"))
    if not auto:
        parser.add_argument("--endpoint-api-key", dest="api_key", default=None,
                            help="supplied credential; prefer an env:NAME reference")
    parser.add_argument("--endpoint-auth-home", dest="codex_home", default=default)
    parser.add_argument("--endpoint-auth-file", dest="codex_auth_file", default=default)


def _endpoint_api_key(args, binding):
    endpoint = binding.endpoint
    if endpoint.auth == "none":
        return None
    if endpoint.auth == "supplied":
        value = args.api_key
    elif endpoint.environment_variable is not None:
        value = os.environ.get(endpoint.environment_variable)
    else:
        raise ValueError("This endpoint requires its Codex credential manager.")
    if value is None or not isinstance(value, str) or not value.strip():
        raise ValueError("Required endpoint credential is unavailable.")
    return value


def build_model(
    args: argparse.Namespace,
    *,
    catalog=None,
    opener=None,
    auth_opener=None,
    trace=None,
) -> Model:
    """Build the configured model.

    ``opener`` replaces the adapter's model HTTP opener. ``auth_opener``
    replaces the Codex OAuth refresh opener; other APIs make no auth requests.
    ``trace`` is invocation-only: it wraps HTTP openers or is passed directly
    to Claude Relay for native/MCP events, never included in a model binding.
    """
    args = prepare_namespace(args, catalog)
    binding = args.model_binding
    from .runtime_config import InteractionConfig
    from .runtime_config import resolve_compaction_mode
    from .runtime_config import resolve_request_timeout_seconds

    mode = resolve_compaction_mode(binding, getattr(args, "compaction_mode", None))
    for name in ("auto_compaction_context_tokens", "max_context_tokens"):
        value = getattr(args, name, None)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(
                f"--{name.replace('_', '-')} must be a positive integer"
            )
    keep = getattr(args, "compaction_pi_keep_recent_tokens", None)
    if keep is not None and (
        isinstance(keep, bool) or not isinstance(keep, int) or keep < 0
    ):
        raise ValueError(
            "--compaction-pi-keep-recent-tokens must be a nonnegative integer"
        )
    budget = getattr(args, "compaction_pi_max_output_tokens", None)
    if budget is not None and (
        isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0
    ):
        raise ValueError(
            "--compaction-pi-max-output-tokens must be a positive integer"
        )

    if args.model_api == "claude-relay":
        from .claude_relay import ClaudeRelayModel
        if opener is not None or auth_opener is not None:
            raise ValueError("Claude Relay does not use model HTTP openers")
        return ClaudeRelayModel(relay_endpoint(args, binding), trace=trace)

    # The adapter's default for calls without a per-call timeout: the launch
    # option, then the catalog's timeouts.request_seconds, then the default.
    request_timeout_seconds = resolve_request_timeout_seconds(
        binding, getattr(args, "request_timeout_seconds", None),
    )
    if trace is not None:
        from ._account_http import default_account_opener
        # The adapters' default underneath, so that traced requests stay
        # cancellable (see _request_gate).
        opener = trace.opener(opener if opener is not None else cancellable_urlopen)
        auth_opener = trace.opener(auth_opener if auth_opener is not None else default_account_opener(),
                                   op="auth_refresh")

    if args.model_api == "chat-completions":
        if args.codex_home is not None or args.codex_auth_file is not None:
            raise ValueError(
                "Endpoint auth paths require --endpoint-api codex"
            )
        endpoint = ChatCompletionsEndpoint(
            binding=binding,
            request_timeout_seconds=request_timeout_seconds,
            api_key=_endpoint_api_key(args, binding),
        )
        return ChatCompletionsModel(endpoint, opener=opener)

    if args.model_api == "messages":
        if args.codex_home is not None or args.codex_auth_file is not None:
            raise ValueError(
                "Endpoint auth paths require --endpoint-api codex"
            )
        if args.model is None or not args.model.strip():
            raise ValueError("--model is required with --endpoint-api messages")
        auto_compaction_context_tokens = getattr(args, "auto_compaction_context_tokens", None)
        server = compaction_methods(mode)[0] == "provider"
        if (
            server
            and auto_compaction_context_tokens is not None
            and auto_compaction_context_tokens < MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS
        ):
            raise ValueError(
                "--auto-compaction-context-tokens must be at least "
                f"{MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS} for "
                "--endpoint-api messages with --compaction-mode provider"
            )
        # In provider mode the endpoint always carries server compaction; the
        # config's per-call enable_auto_compaction suppresses it, so a later
        # /config change applies to the next turn. Pi mode attaches none.
        compaction_options = (
            MessagesServerCompaction() if server else None
        )
        # Validate required budgets before touching credential sources.
        output_budget = InteractionConfig.from_namespace(args).get("max_output_tokens")
        endpoint = MessagesEndpoint(
            binding=binding,
            max_output_tokens=output_budget,
            request_timeout_seconds=request_timeout_seconds,
            api_key=_endpoint_api_key(args, binding),
            server_compaction=compaction_options,
            prompt_caching=MessagesPromptCaching(),
        )
        return MessagesModel(endpoint, opener=opener)

    if args.model_api == "codex":
        if args.model is None or not args.model.strip():
            raise ValueError(
                "--model is required with --endpoint-api codex"
            )
        return CodexResponsesModel(
            request_timeout_seconds=request_timeout_seconds,
            binding=binding,
            auth=(CodexAuth(_endpoint_api_key(args, binding))
                  if binding.endpoint.auth == "supplied" else None),
            opener=opener,
            auth_opener=auth_opener,
        )

    if args.model_api == "responses":
        if args.codex_home is not None or args.codex_auth_file is not None:
            raise ValueError(
                "Endpoint auth paths require --endpoint-api codex"
            )
        if args.model is None or not args.model.strip():
            raise ValueError(
                "--model is required with --endpoint-api responses"
            )
        # API-key or anonymous auth only: there is no OAuth refresh for
        # auth_opener to carry, and env:NAME is reread by the model per sample.
        return ResponsesModel(
            binding=binding,
            api_key=(_endpoint_api_key(args, binding)
                     if binding.endpoint.auth == "supplied" else None),
            request_timeout_seconds=request_timeout_seconds,
            opener=opener,
        )

    raise ValueError(f"unsupported model API: {args.model_api!r}")


def _request_settings(spec) -> list:
    """A preset's typed Responses defaults, then its extra sample params as sent."""
    settings = []
    if spec.responses is not None:
        for label, value in (
            ("effort", spec.responses.reasoning_effort),
            ("summary", spec.responses.reasoning_summary),
            ("verbosity", spec.responses.text_verbosity),
        ):
            if value is not None:
                settings.append(f"{label}={value}")
    # Compact JSON, i.e. the value syntax of extra_sample_params.<key> and --extra-sample-params.
    for key, value in spec.extra_sample_params.items():
        settings.append(f"{key}={json.dumps(thaw_json(value), ensure_ascii=False, separators=(',', ':'))}")
    return settings


def _timeout_settings(spec) -> list:
    """A catalog entry's configured timeouts, in the catalog's key syntax."""
    return [f"timeouts.{name}={getattr(spec.timeouts, name):g}"
            for name in MODEL_TIMEOUT_FIELDS if getattr(spec.timeouts, name) is not None]


def _sample_param_settings(spec) -> list:
    """A catalog entry's typed sample params, in the catalog's key syntax."""
    return [f"sample_params.{name}={json.dumps(getattr(spec.sample_params, name))}"
            for name in SAMPLE_PARAM_DEFAULT_FIELDS if getattr(spec.sample_params, name) is not None]


def _model_argument_help() -> str:
    entries = []
    for spec in list_model_specs():
        details = [spec.endpoint.api, *_request_settings(spec)]
        if spec.endpoint.environment_variable is not None:
            details.append(spec.endpoint.environment_variable)
        if spec.aliases:
            details.append("aliases: " + ", ".join(spec.aliases))
        entries.append(f"{spec.name} ({', '.join(details)})")
    text = "model name; uncatalogued names pass through. Catalog presets: " + "; ".join(entries)
    return text.replace("%", "%%")  # argparse %-formats help strings


def build_parser(description: str, *, allow_prompt_file: bool = False,
                 save_prefix: bool = False) -> argparse.ArgumentParser:
    """The options that the CLI and the demo share.

    With ``save_prefix`` (the CLI), ``--save`` names a save prefix rather
    than a file.
    """
    parser = argparse.ArgumentParser(description=description)
    add_endpoint_arguments(parser)
    add_catalog_arguments(parser)
    parser.add_argument("--model", help=_model_argument_help())
    parser.add_argument("--cwd", default=".")
    parser.add_argument(
        "--enable-auto-compaction",
        nargs="?",
        const=True,
        default=True,
        type=_boolean_argument,
        metavar="{False,True}",
        help=(
            "enable automatic compaction: a pi summary, or Codex remote "
            "compaction in provider mode, when the context reaches "
            "auto_compaction_context_tokens, with one compact-and-retry when a sample "
            "exceeds the context window; Messages server edits in provider "
            "mode. A bare flag means True (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--enable-workspace",
        nargs="?",
        const=True,
        default=True,
        type=_boolean_argument,
        metavar="{False,True}",
        help=(
            "restrict exec_command workdir and apply_patch paths to --cwd; "
            "a bare flag means True (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--enable-experimental-media",
        nargs="?",
        const=True,
        default=False,
        type=_boolean_argument,
        metavar="{False,True}",
        help=(
            "experimental: expand leading @path-or-uri tokens in user "
            "prompts; images become media message content and local "
            ".txt/.md files are pasted as text; launch-only "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--auto-compaction-context-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "initial automatic-compaction token threshold; overrides the model "
            "catalog value and is tunable at runtime with "
            "/config auto_compaction_context_tokens (default: catalog value, if known; "
            "setting null restores that value)"
        ),
    )
    parser.add_argument(
        "--compaction-mode",
        type=compaction_mode_argument,
        default=None,
        metavar="SEQUENCE",
        help=(
            COMPACTION_MODE_HELP + "; tunable with /config compaction_mode "
            "(default: coldstart,warmpi:2; a session without exec_command "
            "drops coldstart)"
        ),
    )
    parser.add_argument(
        "--compaction-pi-keep-recent-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "estimated tokens of recent context that pi compaction keeps "
            "verbatim; 0 keeps nothing. Tunable with "
            "/config compaction_pi_keep_recent_tokens "
            f"(default: {DEFAULT_KEEP_RECENT_TOKENS})"
        ),
    )
    parser.add_argument(
        "--compaction-pi-max-output-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "output budget of each pi summary request; tunable with "
            "/config compaction_pi_max_output_tokens "
            "(default: the turn's max_output_tokens)"
        ),
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "initial context-window ceiling in tokens; informational only and "
            "tunable with /config max_context_tokens "
            "(default: catalog value, if known)"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="maximum model samples; unlimited when omitted",
    )
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=None,
        help=(
            "HTTP blocking-I/O timeout for model requests (default: "
            f"{DEFAULT_REQUEST_TIMEOUT_SECONDS} seconds unless the model catalog "
            "sets timeouts.request_seconds; not an overall deadline); tunable "
            "with /config request_timeout_seconds. /login and /quota keep "
            "their fixed timeout"
        ),
    )
    add_prompt_arguments(parser, allow_file=allow_prompt_file)
    parser.add_argument(
        "--instructions",
        default=None,
        help=(
            "Optional system instructions (Chat Completions system message). "
            "Empty string is preserved; omit to send none. "
            "On --resume, appends an override."
        ),
    )
    if save_prefix:
        parser.add_argument(
            "--save",
            dest="save_path",
            metavar="PREFIX",
            type=_save_path_argument,
            default=DEFAULT_SAVE_PREFIX,
            help=(
                "save prefix (default: %(default)s); relative to the launch directory, "
                "not --cwd; its directory must exist; a trailing .jsonl is ignored. "
                "PREFIX.jsonl is the save log, without reasoning, and PREFIX._private/ "
                "holds the full log, context.jsonl, which resuming reads, and debug files. "
                "--resume=False replaces the save"
            ),
        )
    else:
        parser.add_argument(
            "--save",
            dest="save_path",
            metavar="PATH",
            type=_save_path_argument,
            default=DEFAULT_SAVE_PATH,
            help=(
                "interaction JSONL file to read/write (default: %(default)s); "
                "relative to the launch directory, not --cwd; parent must exist; "
                "replaces the file when --resume is False"
            ),
        )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        type=_boolean_argument,
        metavar="{False,True}",
        help=(
            ("resume the save that --save names instead of starting a new one; "
             "a missing save starts a new one. ")
            if save_prefix else
            ("resume the selected --save file instead of starting a new save; "
             "a missing file starts a new one. ")
        ) + "A bare flag means True (default: %(default)s)",
    )
    add_renamed_options(parser)
    return parser

