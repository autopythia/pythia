"""Typed, non-secret configuration that lives for one interaction process."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
import json
import re
from threading import RLock
from typing import Dict
from typing import Optional
from typing import Union

from .messages import resolve_messages_max_output_tokens
from .model import SamplingOptions
from .model import ModelConfigurationError


ConfigValue = Union[bool, int, None]
CONFIG_KEYS = (
    "enable_workspace",
    "max_samples",
    "max_output_tokens",
    "enable_auto_compaction",
)
_BOOLEAN_KEYS = frozenset(("enable_workspace", "enable_auto_compaction"))
_OPTIONAL_POSITIVE_INTEGER_KEYS = frozenset((
    "max_samples",
    "max_output_tokens",
))
_INTEGER_LITERAL_RE = re.compile(r"^[+-]?[0-9]+$")


class ConfigError(ValueError):
    pass


def _require_key(key: object) -> str:
    if not isinstance(key, str) or key not in CONFIG_KEYS:
        raise ConfigError(
            "Unknown config key; use /config to list supported keys."
        )
    return key


def validate_config_value(key: str, value: object) -> ConfigValue:
    key = _require_key(key)
    if key in _BOOLEAN_KEYS:
        if not isinstance(value, bool):
            raise ConfigError(f"{key} requires True or False.")
        return value
    if key in _OPTIONAL_POSITIVE_INTEGER_KEYS:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(
                f"{key} requires a positive integer or None."
            )
        return value
    raise AssertionError(f"missing config validator for {key}")


def parse_config_literal(key: str, text: object) -> ConfigValue:
    """Parse the intentionally small JSON/Python scalar input grammar."""
    key = _require_key(key)
    if not isinstance(text, str):
        raise ConfigError(f"Invalid value for {key}.")
    literal = text.strip()
    aliases = {
        "null": None,
        "None": None,
        "false": False,
        "False": False,
        "true": True,
        "True": True,
    }
    if literal in aliases:
        value = aliases[literal]
    elif _INTEGER_LITERAL_RE.fullmatch(literal) is not None:
        value = int(literal)
    else:
        raise ConfigError(f"Invalid value for {key}.")
    return validate_config_value(key, value)


@dataclass(frozen=True)
class InteractionConfigSnapshot:
    enable_workspace: bool = True
    max_samples: Optional[int] = None
    max_output_tokens: Optional[int] = None
    enable_auto_compaction: bool = True
    _auto_compaction_override: Optional[bool] = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        for key in CONFIG_KEYS:
            validate_config_value(key, getattr(self, key))
        if (
            self._auto_compaction_override is not None
            and not isinstance(self._auto_compaction_override, bool)
        ):
            raise TypeError("_auto_compaction_override must be a bool or None")
        if (
            self._auto_compaction_override is not None
            and self._auto_compaction_override != self.enable_auto_compaction
        ):
            raise ValueError(
                "_auto_compaction_override must match enable_auto_compaction"
            )

    def as_dict(self) -> Dict[str, ConfigValue]:
        return {key: getattr(self, key) for key in CONFIG_KEYS}

    def sampling_options(self) -> Optional[SamplingOptions]:
        if (
            self.max_output_tokens is None
            and self._auto_compaction_override is None
        ):
            return None
        return SamplingOptions(
            max_output_tokens=self.max_output_tokens,
            enable_auto_compaction=self._auto_compaction_override,
        )


class InteractionConfig:
    """Thread-safe in-memory configuration with an immutable read snapshot."""

    def __init__(
        self,
        snapshot: InteractionConfigSnapshot = InteractionConfigSnapshot(),
        *,
        on_enable_workspace: Optional[Callable[[bool], None]] = None,
        require_max_output_tokens: bool = False,
        max_output_tokens_fallback: Optional[int] = None,
    ) -> None:
        if not isinstance(snapshot, InteractionConfigSnapshot):
            raise TypeError("snapshot must be InteractionConfigSnapshot")
        if on_enable_workspace is not None and not callable(
            on_enable_workspace
        ):
            raise TypeError("on_enable_workspace must be callable or None")
        if not isinstance(require_max_output_tokens, bool):
            raise TypeError("require_max_output_tokens must be a bool")
        if max_output_tokens_fallback is not None:
            max_output_tokens_fallback = validate_config_value(
                "max_output_tokens",
                max_output_tokens_fallback,
            )
        if require_max_output_tokens and snapshot.max_output_tokens is None:
            raise ConfigError(
                "max_output_tokens is required for Messages."
            )
        self._lock = RLock()
        self._snapshot = snapshot
        self._on_enable_workspace = on_enable_workspace
        self._require_max_output_tokens = require_max_output_tokens
        self._max_output_tokens_fallback = max_output_tokens_fallback

    @classmethod
    def from_namespace(
        cls,
        args,
        *,
        on_enable_workspace: Optional[Callable[[bool], None]] = None,
    ) -> "InteractionConfig":
        max_output_tokens = args.max_output_tokens
        max_output_tokens_fallback = None
        if args.model_api == "messages":
            try:
                max_output_tokens_fallback = (
                    resolve_messages_max_output_tokens(args.model, None)
                )
            except ModelConfigurationError:
                pass
            max_output_tokens = resolve_messages_max_output_tokens(
                args.model,
                max_output_tokens,
            )
        return cls(
            InteractionConfigSnapshot(
                enable_workspace=args.enable_workspace,
                max_samples=args.max_samples,
                max_output_tokens=max_output_tokens,
                enable_auto_compaction=args.enable_auto_compaction,
                _auto_compaction_override=(
                    None if args.enable_auto_compaction else False
                ),
            ),
            on_enable_workspace=on_enable_workspace,
            require_max_output_tokens=args.model_api == "messages",
            max_output_tokens_fallback=max_output_tokens_fallback,
        )

    def snapshot(self) -> InteractionConfigSnapshot:
        with self._lock:
            return self._snapshot

    def get(self, key: str) -> ConfigValue:
        key = _require_key(key)
        return getattr(self.snapshot(), key)

    def set(self, key: str, value: object) -> ConfigValue:
        key = _require_key(key)
        value = validate_config_value(key, value)
        if (
            key == "max_output_tokens"
            and value is None
            and self._require_max_output_tokens
        ):
            if self._max_output_tokens_fallback is None:
                raise ConfigError(
                    "max_output_tokens is required for this Messages model."
                )
            value = self._max_output_tokens_fallback
        with self._lock:
            current = self._snapshot
            if getattr(current, key) == value:
                return value
            changes = {key: value}
            if key == "enable_auto_compaction":
                changes["_auto_compaction_override"] = value
            candidate = replace(current, **changes)
            if key == "enable_workspace" and self._on_enable_workspace:
                self._on_enable_workspace(value)
            self._snapshot = candidate
        return value

    def values(self, key: Optional[str] = None) -> Dict[str, ConfigValue]:
        snapshot = self.snapshot()
        if key is not None:
            key = _require_key(key)
            return {key: getattr(snapshot, key)}
        return snapshot.as_dict()

    def render(self, key: Optional[str] = None, *, json_output: bool) -> str:
        values = self.values(key)
        if json_output:
            return json.dumps(values, ensure_ascii=False, indent=2)
        return "\n".join(
            f"{name} = {value!r}"
            for name, value in values.items()
        )


__all__ = [
    "CONFIG_KEYS",
    "ConfigError",
    "ConfigValue",
    "InteractionConfig",
    "InteractionConfigSnapshot",
    "parse_config_literal",
    "validate_config_value",
]
