"""Small fixed-role configuration resolver. It never loads credential values."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

from ._auto_board import parse_json
from .model import SamplingOptions
from .runtime_config import InteractionConfig
from .timeouts import DEFAULT_REQUEST_TIMEOUT_SECONDS


NAMES = {1: "main", 2: "worker", -1: "watcher"}
DEFAULTS = {
    "model_api": "chat-completions", "model": None, "api_url": None,
    "api_key_env": None, "codex_home": None, "codex_auth_file": None,
    "cwd": ".", "max_samples": None, "max_output_tokens": None,
    "request_timeout_seconds": DEFAULT_REQUEST_TIMEOUT_SECONDS,
    "enable_workspace": True, "enable_auto_compaction": True,
    "instructions": None,
}
_APIS = {"chat-completions", "messages", "codex", "codex-responses"}
_PROVIDER_FIELDS = ("model", "api_url", "api_key_env", "codex_home", "codex_auth_file")
_PATHS = ("cwd", "codex_home", "codex_auth_file")


def _layer(value, base):
    if not isinstance(value, dict) or set(value) - (set(DEFAULTS) | {"name"}):
        raise ValueError("Invalid auto configuration fields (use credential references, not API keys).")
    value = dict(value)
    for key in _PATHS:
        if key == "cwd" and key in value and value[key] is None:
            raise ValueError("cwd must be an existing directory path.")
        if key in value and value[key] is not None:
            raw = value[key]
            if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
                raise ValueError(f"Invalid {key} path.")
            path = Path(raw).expanduser()
            value[key] = str((base / path).absolute())
    return value


def _merge(current, value):
    def canonical(api):
        return "codex" if api == "codex-responses" else api

    current = dict(current)
    api = value.get("model_api", current["model_api"])
    if canonical(api) != canonical(current["model_api"]):
        for key in _PROVIDER_FIELDS:
            current[key] = None
    current.update(value)
    return current


def resolve_config(path=None, overrides=None):
    document = {}
    base = Path.cwd()
    if path is not None:
        path = Path(path).expanduser().absolute()
        base = path.parent
        try:
            document = parse_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            raise ValueError("Could not load auto context configuration.") from None
        if (not isinstance(document, dict) or set(document) - {"version", "defaults", "contexts"}
                or type(document.get("version")) is not int or document["version"] != 1):
            raise ValueError("Expected auto configuration version 1.")
    contexts = document.get("contexts", {})
    if not isinstance(contexts, dict) or set(contexts) - {"1", "2", "-1"}:
        raise ValueError("Auto contexts must use keys 1, 2, and -1.")
    launch = _layer(overrides or {}, Path.cwd())
    if "name" in launch:
        raise ValueError("Names must be configured per context.")
    main_instructions = launch.pop("instructions", None)
    main_instruction_override = overrides is not None and "instructions" in overrides
    common = _merge(DEFAULTS, _layer(document.get("defaults", {}), base))
    if "name" in common:
        raise ValueError("Names must be configured per context.")
    common = _merge(common, launch)
    resolved = {}
    for index, name in NAMES.items():
        settings = _merge({**common, "name": name},
                          _layer(contexts.get(str(index), {}), base))
        if index == 1 and main_instruction_override and "instructions" not in contexts.get("1", {}):
            settings["instructions"] = main_instructions
        settings["cwd"] = str(Path(settings["cwd"]).expanduser().absolute())
        _validate(settings)
        resolved[index] = settings
    return resolved


def _validate(settings):
    api = settings["model_api"]
    if not isinstance(api, str) or api not in _APIS:
        raise ValueError("Unsupported auto model API.")
    model = settings["model"]
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model must be a nonempty string or null.")
    if api != "chat-completions" and model is None:
        raise ValueError("A model is required for Messages/Codex.")
    name = settings["name"]
    if (not isinstance(name, str) or not name.strip() or len(name) > 64 or
            any(ord(c) < 32 or ord(c) == 127 for c in name)):
        raise ValueError("Context name must be a short single-line label.")
    instructions = settings["instructions"]
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError("instructions must be text or null.")
    env = settings["api_key_env"]
    if env is not None and (not isinstance(env, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env) is None):
        raise ValueError("api_key_env must name an environment variable.")
    if api in {"codex", "codex-responses"}:
        if env is not None:
            raise ValueError("Codex uses codex_auth_file/codex_home, not api_key_env.")
    elif settings["codex_home"] is not None or settings["codex_auth_file"] is not None:
        raise ValueError("Codex credential paths require the Codex API.")
    timeout = settings["request_timeout_seconds"]
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
            not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("request_timeout_seconds must be positive and finite.")
    samples = settings["max_samples"]
    if samples is not None and (type(samples) is not int or samples <= 0):
        raise ValueError("max_samples must be a positive integer or null.")
    url = settings["api_url"]
    if url is not None:
        try:
            if not isinstance(url, str):
                raise ValueError()
            parsed = urlsplit(url)
            if (parsed.scheme not in {"http", "https"}
                    or not parsed.hostname or parsed.username is not None
                    or parsed.password is not None or parsed.query or parsed.fragment):
                raise ValueError()
            parsed.port  # Validate the numeric/range syntax without connecting.
        except (TypeError, ValueError):
            raise ValueError("api_url must be an HTTP(S) URL without credentials/query/fragment.") from None
    if not Path(settings["cwd"]).is_dir():
        raise ValueError("Context cwd must be an existing directory.")
    SamplingOptions(max_output_tokens=settings["max_output_tokens"])
    # Reuse provider-aware max-output-token and runtime setting validation.
    InteractionConfig.from_namespace(namespace(settings))


def namespace(settings):
    return argparse.Namespace(**settings, api_key=None)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Board-first three-context auto MVP (local tools are unsandboxed).",
        allow_abbrev=False,
    )
    parser.add_argument("--context-config", type=Path, help="Version-1 JSON configuration for contexts 1, 2, -1.")
    parser.add_argument("--save", type=Path, default=Path("interaction-auto"),
                        help="New save directory; existing paths are never overwritten (default: interaction-auto).")
    parser.add_argument("--prompt", help="Post one user task as a fresh board thread; run without a TTY.")
    parser.add_argument("--board-port", type=int, default=0, help="Loopback port (0 chooses an available port).")
    parser.add_argument("--model-api", choices=sorted(_APIS), default=argparse.SUPPRESS)
    for key in ("model", "api_url", "api_key_env", "codex_home", "codex_auth_file", "cwd", "instructions"):
        parser.add_argument("--" + key.replace("_", "-"), default=argparse.SUPPRESS)
    parser.add_argument("--max-samples", type=int, default=argparse.SUPPRESS,
                        help="Optional sample limit per turn; unlimited when unset.")
    parser.add_argument("--max-output-tokens", type=int, default=argparse.SUPPRESS,
                        help="Optional output-token limit; uses model/API defaults when unset.")
    parser.add_argument("--request-timeout-seconds", type=float, default=argparse.SUPPRESS)
    return parser
