"""Shared argument defaults and provider construction for interaction frontends."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .chat_completions import ChatCompletionsEndpoint
from .chat_completions import ChatCompletionsModel
from .messages import MessagesEndpoint
from .messages import MessagesModel
from .messages import MessagesPromptCaching
from .messages import MessagesServerCompaction
from .model import Model
from .model_catalog import ANTHROPIC_MESSAGES_API_URL as ANTHROPIC_MESSAGES_API_URL
from .model_catalog import CODEX_RESPONSES_API_URL
from .model_catalog import get_model_route
from .model_catalog import list_model_specs
from .responses import CodexResponsesModel
from .timeouts import DEFAULT_REQUEST_TIMEOUT_SECONDS


DEFAULT_SAVE_PATH = Path("interaction.jsonl")


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
    if args.model_api not in {"codex", "codex-responses"} or not args.model:
        return False
    name = args.model.strip()
    route = get_model_route("codex", name)
    if route.provider != "chatgpt" or route.auth_source != "codex-login":
        return False
    url = args.api_url if args.api_url is not None else route.api_url
    return url.strip().rstrip("/") == CODEX_RESPONSES_API_URL


def build_model(args: argparse.Namespace) -> Model:
    messages_compaction_options_requested = any(
        (
            args.messages_server_compaction,
            args.messages_compaction_trigger_tokens is not None,
            args.messages_pause_after_compaction,
            args.messages_compaction_instructions is not None,
        )
    )
    if args.model_api != "messages" and messages_compaction_options_requested:
        raise ValueError(
            "Messages compaction options require --model-api messages"
        )

    if args.model_api == "chat-completions":
        if args.codex_home is not None or args.codex_auth_file is not None:
            raise ValueError(
                "--codex-home and --codex-auth-file require "
                "--model-api codex-responses"
            )
        endpoint = ChatCompletionsEndpoint(
            api_url=args.api_url or get_model_route("chat-completions", args.model).api_url,
            model=args.model,
            request_timeout_seconds=args.request_timeout_seconds,
            api_key=args.api_key,
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
        compaction_options = (
            MessagesServerCompaction(
                trigger_input_tokens=(
                    args.messages_compaction_trigger_tokens
                ),
                pause_after_compaction=(
                    args.messages_pause_after_compaction
                ),
                instructions=args.messages_compaction_instructions,
            )
            if args.messages_server_compaction
            else None
        )
        if compaction_options is None and any(
            (
                args.messages_compaction_trigger_tokens is not None,
                args.messages_pause_after_compaction,
                args.messages_compaction_instructions is not None,
            )
        ):
            raise ValueError(
                "Messages compaction options require "
                "--messages-server-compaction"
            )
        route = get_model_route("messages", args.model)
        endpoint = MessagesEndpoint(
            api_url=args.api_url or route.api_url,
            model=args.model,
            request_timeout_seconds=args.request_timeout_seconds,
            api_key=args.api_key or (
                os.environ.get(route.api_key_environment_variable)
                if route.api_key_environment_variable is not None else None
            ),
            server_compaction=compaction_options,
            prompt_caching=MessagesPromptCaching(),
        )
        return MessagesModel(endpoint)

    if args.model_api in ("codex", "codex-responses"):
        if args.api_key is not None:
            raise ValueError(
                "--api-key is not used with --model-api codex-responses; "
                "use an existing Codex login"
            )
        if args.model is None or not args.model.strip():
            raise ValueError(
                "--model is required with --model-api codex-responses"
            )
        return CodexResponsesModel(
            model=args.model,
            api_url=args.api_url,
            request_timeout_seconds=args.request_timeout_seconds,
            codex_home=args.codex_home,
            auth_file=args.codex_auth_file,
        )

    raise ValueError(f"unsupported model API: {args.model_api!r}")


def _model_argument_help() -> str:
    entries = []
    for spec in list_model_specs():
        details = [f"{spec.profile}/{spec.route.provider}"]
        if spec.responses is not None:
            for label, value in (
                ("effort", spec.responses.reasoning_effort),
                ("summary", spec.responses.reasoning_summary),
                ("verbosity", spec.responses.text_verbosity),
            ):
                if value is not None:
                    details.append(f"{label}={value}")
        if spec.route.api_key_environment_variable is not None:
            details.append(spec.route.api_key_environment_variable)
        if spec.aliases:
            details.append("aliases: " + ", ".join(spec.aliases))
        entries.append(f"{spec.name} ({', '.join(details)})")
    return "model name; uncatalogued names pass through. Catalog presets: " + "; ".join(entries)


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model-api",
        choices=("chat-completions", "messages", "codex", "codex-responses"),
        default="chat-completions",
        help=(
            "model API (messages uses automatic 5m prompt caching; "
            "codex is shorthand for codex-responses)"
        ),
    )
    parser.add_argument("--api-url")
    parser.add_argument(
        "--model",
        help=_model_argument_help(),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (Messages defaults to ANTHROPIC_API_KEY)",
    )
    parser.add_argument("--codex-home")
    parser.add_argument("--codex-auth-file")
    parser.add_argument(
        "--messages-server-compaction",
        action="store_true",
        help="enable Anthropic Messages server-side compaction",
    )
    parser.add_argument(
        "--messages-compaction-trigger-tokens",
        type=int,
        help=(
            "server compaction threshold (minimum 50000; defaults to the known "
            "model context maximum, otherwise the server default)"
        ),
    )
    parser.add_argument(
        "--messages-pause-after-compaction",
        action="store_true",
        help="pause and resample after the server creates a compaction block",
    )
    parser.add_argument("--messages-compaction-instructions")
    parser.add_argument("--cwd", default=".")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="maximum model samples; unlimited when omitted",
    )
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=(
            "HTTP blocking-I/O timeout for model and account requests "
            "(default: %(default)s seconds; not an overall deadline)"
        ),
    )
    parser.add_argument("--prompt")
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

