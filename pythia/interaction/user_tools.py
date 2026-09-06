"""User-only tool adapters. Never register these with the model environment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

from ._account_http import AccountServiceError
from .codex_auth import CodexAuthError, _resolve_auth_file, load_codex_auth
from .codex_login import login
from .codex_quota import query_quota
from .environment import Environment, Tool, ToolOutcome, ToolSpec
from .model_config import supports_account_services


@dataclass(frozen=True)
class UserToolIntent:
    name: str
    arguments_json: str


def parse_user_tool(text: str) -> UserToolIntent:
    words = text.split()
    if not words or words[0] not in {"/login", "/quota"}:
        raise ValueError("Unsupported command. Use /login, /quota, /quit, or /exit.")
    name = words[0][1:]
    if "\n" in text or "\r" in text:
        raise ValueError("User-tool commands must be a single line.")
    if name == "quota":
        if len(words) != 1:
            raise ValueError("Usage: /quota (no arguments).")
        arguments = {}
    else:
        if len(words) > 2 or (len(words) == 2 and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", words[1])):
            raise ValueError("Usage: /login [workspace-id]. Do not paste codes or tokens.")
        arguments = {"workspace_id": words[1]} if len(words) == 2 else {}
    return UserToolIntent(name, json.dumps(arguments, sort_keys=True))


def create_user_environment(args, *, notify, cancel, expected_account=None, provider_history=False):
    """Create invocation-scoped adapters with a safe error boundary and notice sink."""
    supported = supports_account_services(args)

    def guard(handler):
        def execute(arguments, *, timeout_seconds=None):
            try:
                if not supported:
                    return ToolOutcome("This configuration does not support ChatGPT account services.", False)
                if cancel.is_set():
                    return ToolOutcome("User tool cancelled before execution.", False)
                return handler(arguments, timeout_seconds)
            except CodexAuthError:
                return ToolOutcome("Model authentication needed; use /login.", False)
            except AccountServiceError as exc:
                return ToolOutcome(str(exc), False)
            except Exception:
                # The ordinary executor prints str(exc); do not let secrets reach it.
                return ToolOutcome("User tool failed; provider and credential details were withheld.", False)
        return execute

    def sign_in(arguments, timeout_seconds):
        if set(arguments) - {"workspace_id"}:
            return ToolOutcome("Unexpected login arguments.", False)
        workspace = arguments.get("workspace_id")
        if workspace is not None and (not isinstance(workspace, str) or
                                      not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", workspace)):
            return ToolOutcome("Invalid workspace ID. Do not supply authorization codes or tokens.", False)
        if provider_history and expected_account is None:
            return ToolOutcome("Cannot verify the account for saved provider state; start a fresh session to log in.", False)
        path = _resolve_auth_file(codex_home=args.codex_home, auth_file=args.codex_auth_file).resolve()
        login(path, notify=notify, cancel=cancel, workspace_id=workspace,
              expected_account=expected_account, timeout_seconds=timeout_seconds,
              request_timeout_seconds=args.request_timeout_seconds)
        return ToolOutcome("Codex credentials saved. Model activation is handled separately.")

    def quota(arguments, timeout_seconds):
        if arguments:
            return ToolOutcome("Quota takes no arguments.", False)
        auth = load_codex_auth(codex_home=args.codex_home, auth_file=args.codex_auth_file)
        if expected_account is not None and auth.account_id != expected_account:
            return ToolOutcome("Credential account changed; start a fresh session before using it.", False)
        return ToolOutcome(query_quota(auth, timeout_seconds=timeout_seconds))

    return Environment((
        Tool(ToolSpec("login", "Sign in to the selected ChatGPT/Codex account.", {
            "type": "object", "properties": {"workspace_id": {"type": "string"}},
            "additionalProperties": False,
        }), guard(sign_in), timeout_seconds=120.0),
        Tool(ToolSpec("quota", "Query a historical Codex account quota snapshot.", {
            "type": "object", "properties": {}, "additionalProperties": False,
        }), guard(quota), timeout_seconds=args.request_timeout_seconds),
    ))
