"""Shared argument defaults and provider construction for interaction frontends."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path

from ._prompt import add_prompt_arguments
from .chat_completions import ChatCompletionsEndpoint
from .chat_completions import ChatCompletionsModel
from .messages import MessagesEndpoint
from .messages import MessagesModel
from .messages import MessagesPromptCaching
from .messages import MessagesServerCompaction
from .messages import MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS
from .model import Model
from .model_catalog import ANTHROPIC_MESSAGES_API_URL as ANTHROPIC_MESSAGES_API_URL
from .model_catalog import CODEX_RESPONSES_API_URL
from .model_catalog import get_model_route
from .model_catalog import list_model_specs
from .model_catalog import BUILTIN_MODEL_CATALOG, binding_from_namespace
from .model_catalog import parse_json_value, freeze_request_params, thaw_json
from .model_catalog import EndpointSpec
from .codex_auth import CodexAuth, _resolve_auth_file
from .model_catalog_config import load_model_catalog
from .responses import CodexResponsesModel
from .timeouts import DEFAULT_REQUEST_TIMEOUT_SECONDS


DEFAULT_SAVE_PATH = Path("interaction.jsonl")


def _boolean_argument(value: str) -> bool:
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError("expected True or False")
    normalized = value.casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected True or False")


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
    """Only the official ChatGPT route supports the initial login/quota tools."""
    if not getattr(args, "model", None):
        return False
    return binding_from_namespace(args).supports_account_services


def prepare_namespace(args, catalog=None):
    """Resolve without changing raw launch/saved input or rereading a catalog."""
    api = getattr(args, "model_api", None)
    if api is not None and api not in {"chat-completions", "messages", "codex", "codex-responses", "responses"}:
        raise ValueError(f"unsupported model API: {api!r}")
    binding = binding_from_namespace(args, catalog)
    endpoint = binding.endpoint
    if (endpoint.auth == "codex-login" and not endpoint.is_official_codex
            and not binding.api_explicit):
        raise ValueError("A custom Codex destination requires explicit --api codex; use --api chat-completions for a local chat model.")
    if binding.api not in {"chat-completions", "messages", "codex"}:
        raise ValueError(f"The frontend does not support the {binding.api} API.")
    if not getattr(args, "_endpoint_prepared", False):
        if getattr(args, "codex_home", None) is not None or getattr(args, "codex_auth_file", None) is not None:
            if endpoint.api != "codex":
                raise ValueError("--codex-home and --codex-auth-file require --model-api codex-responses")
            if endpoint.auth != "codex-login":
                raise ValueError("Codex credential files require endpoint-auth codex-login.")
    if endpoint.auth == "codex-login" and endpoint.auth_file is None:
        endpoint = replace(endpoint, auth_file=str(_resolve_auth_file(
            codex_home=getattr(args, "codex_home", None), auth_file=getattr(args, "codex_auth_file", None),
        ).resolve()))
        overrides = set(binding.endpoint_overrides)
        if getattr(args, "codex_home", None) is not None or getattr(args, "codex_auth_file", None) is not None:
            overrides.add("auth_file")
        binding = replace(binding, endpoint=endpoint, endpoint_overrides=frozenset(overrides))
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
        origin = catalog.origins[(spec.profile, spec.name)]
        lines.append(f"{spec.name}{aliases}: api={spec.profile}, model={spec.api_model}, "
                     f"url={spec.endpoint.url}, auth={spec.endpoint.auth}, source={origin}")
    return "\n".join(lines)


def _request_params_argument(text):
    try:
        value = parse_json_value(text)
        return None if value is None else thaw_json(freeze_request_params(value))
    except ValueError:
        raise argparse.ArgumentTypeError("expected a JSON object of request params or null") from None


def add_catalog_arguments(parser, *, suppress_request_params=False):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model-catalog", type=Path,
                       help="INI user catalog (default: ~/.pythia/model-catalog.ini)")
    group.add_argument("--no-user-model-catalog", action="store_true", help="use only the built-in model catalog")
    parser.add_argument("--list-models", action="store_true", help="list the selected catalog without loading credentials")
    parser.add_argument("--request-params", type=_request_params_argument,
                        default=argparse.SUPPRESS if suppress_request_params else None,
                        help="JSON object of model-specific request-body extensions; launch-only")


def add_endpoint_arguments(parser, *, auto=False):
    default = argparse.SUPPRESS if auto else None
    parser.add_argument("--endpoint-api", "--api", "--model-api", dest="model_api",
                        choices=("chat-completions", "messages", "codex", "codex-responses"),
                        default=default, help="endpoint API; omitted infers a unique catalog selection")
    urls = parser.add_mutually_exclusive_group()
    urls.add_argument("--endpoint-url", default=default, help="complete model POST URL (no suffix is appended)")
    urls.add_argument("--api-url", default=default, help="legacy URL prefix; retains the selected API's v1 append rule")
    parser.add_argument("--endpoint-model", default=default, help="wire model ID (not a catalog selector)")
    parser.add_argument("--endpoint-auth", default=default, help="none, env:NAME, codex-login, or supplied")
    if not auto:
        parser.add_argument("--endpoint-api-key", "--api-key", dest="api_key", default=None,
                            help="supplied credential; prefer an env:NAME reference")
    else:
        parser.add_argument("--endpoint-api-key-env", "--api-key-env", dest="api_key_env", default=default)
    parser.add_argument("--endpoint-auth-home", "--codex-home", dest="codex_home", default=default)
    parser.add_argument("--endpoint-auth-file", "--codex-auth-file", dest="codex_auth_file", default=default)


def _route_api_key(args, binding):
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


def build_model(args: argparse.Namespace, *, catalog=None) -> Model:
    args = prepare_namespace(args, catalog)
    binding = args.model_binding
    for name in ("auto_compact_tokens", "max_context_tokens"):
        value = getattr(args, name, None)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(
                f"--{name.replace('_', '-')} must be a positive integer"
            )

    if args.model_api == "chat-completions":
        if args.codex_home is not None or args.codex_auth_file is not None:
            raise ValueError(
                "--codex-home and --codex-auth-file require "
                "--model-api codex-responses"
            )
        endpoint = ChatCompletionsEndpoint(
            api_url=None,
            model=args.model,
            request_timeout_seconds=args.request_timeout_seconds,
            api_key=_route_api_key(args, binding),
            binding=binding,
        )
        return ChatCompletionsModel(endpoint)

    if args.model_api == "messages":
        if args.codex_home is not None or args.codex_auth_file is not None:
            raise ValueError(
                "--codex-home and --codex-auth-file require "
                "--model-api codex-responses"
            )
        if args.model is None or not args.model.strip():
            raise ValueError("--model is required with --model-api messages")
        auto_compact_tokens = getattr(args, "auto_compact_tokens", None)
        if (
            auto_compact_tokens is not None
            and auto_compact_tokens < MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS
        ):
            raise ValueError(
                "--auto-compact-tokens must be at least "
                f"{MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS} for "
                "--model-api messages"
            )
        compaction_options = (
            MessagesServerCompaction()
            if args.enable_auto_compaction
            else None
        )
        # Validate required budgets before touching credential sources.
        from .runtime_config import InteractionConfig
        output_budget = InteractionConfig.from_namespace(args).get("max_output_tokens")
        endpoint = MessagesEndpoint(
            api_url=None,
            model=args.model,
            max_output_tokens=output_budget,
            request_timeout_seconds=args.request_timeout_seconds,
            api_key=_route_api_key(args, binding),
            binding=binding,
            server_compaction=compaction_options,
            prompt_caching=MessagesPromptCaching(),
        )
        return MessagesModel(endpoint)

    if args.model_api in ("codex", "codex-responses"):
        if args.model is None or not args.model.strip():
            raise ValueError(
                "--model is required with --model-api codex-responses"
            )
        return CodexResponsesModel(
            model=args.model,
            request_timeout_seconds=args.request_timeout_seconds,
            binding=binding,
            auth=CodexAuth(_route_api_key(args, binding)) if binding.endpoint.auth == "supplied" else None,
        )

    raise ValueError(f"unsupported model API: {args.model_api!r}")


def _model_argument_help() -> str:
    entries = []
    for spec in list_model_specs():
        details = [f"{spec.profile}/{spec.provider_label}"]
        if spec.responses is not None:
            for label, value in (
                ("effort", spec.responses.reasoning_effort),
                ("summary", spec.responses.reasoning_summary),
                ("verbosity", spec.responses.text_verbosity),
            ):
                if value is not None:
                    details.append(f"{label}={value}")
        if spec.messages is not None:
            if spec.messages.output_effort is not None:
                details.append(
                    f"effort={spec.messages.output_effort}"
                )
        if spec.endpoint.environment_variable is not None:
            details.append(spec.endpoint.environment_variable)
        if spec.aliases:
            details.append("aliases: " + ", ".join(spec.aliases))
        entries.append(f"{spec.name} ({', '.join(details)})")
    return "model name; uncatalogued names pass through. Catalog presets: " + "; ".join(entries)


def build_parser(description: str, *, allow_prompt_file: bool = False) -> argparse.ArgumentParser:
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
            "enable automatic remote compaction (Responses threshold triggers "
            "and Messages server edits); a bare flag means True "
            "(default: %(default)s)"
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
            "experimental: convert leading @path-or-uri tokens in user "
            "prompts into media message content; launch-only "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--auto-compact-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "initial automatic-compaction token threshold; overrides the model "
            "catalog value and is tunable at runtime with "
            "/config auto_compact_tokens (default: catalog value, if known; "
            "setting null restores that value)"
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
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=(
            "HTTP blocking-I/O timeout for model and account requests "
            "(default: %(default)s seconds; not an overall deadline)"
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
    parser.add_argument(
        "--save",
        dest="save_path",
        metavar="PATH",
        type=_save_path_argument,
        default=DEFAULT_SAVE_PATH,
        help=(
            "interaction JSONL file to read/write (default: %(default)s); "
            "relative to the launch directory, not --cwd; parent must exist; "
            "replaces the file unless --resume is used"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the selected --save file instead of starting a new save",
    )
    return parser

