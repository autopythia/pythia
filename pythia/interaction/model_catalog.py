"""Static, profile-scoped model facts and Pythia request presets.

This module has no adapter, credential, environment, or network dependencies.
Limits are metadata, not output budgets. Routes describe defaults, not resolved
credentials or permission to use account services. Unknown names remain valid
pass-through candidates for the owning adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from types import MappingProxyType
from typing import Iterable
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Tuple


OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1"
CODEX_RESPONSES_API_URL = "https://chatgpt.com/backend-api/codex"
META_RESPONSES_API_URL = "https://api.meta.ai/v1"
ANTHROPIC_MESSAGES_API_URL = "https://api.anthropic.com"

_PROFILES = frozenset(("codex", "responses", "messages", "chat-completions"))


def _normalize_profile(profile: str) -> str:
    if profile == "codex-responses":
        return "codex"
    if not isinstance(profile, str) or profile not in _PROFILES:
        raise ValueError("unknown model catalog profile")
    return profile


def _require_identifier(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{field_name} must be nonempty and contain no whitespace or controls")


@dataclass(frozen=True)
class ModelLimits:
    default_context_tokens: Optional[int] = None
    max_context_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("default_context_tokens", "max_context_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if (self.default_context_tokens is not None and self.max_context_tokens is not None
                and self.default_context_tokens > self.max_context_tokens):
            raise ValueError("default context must not exceed maximum context")


@dataclass(frozen=True)
class ModelRoute:
    """Non-secret provider defaults; credential loading stays in the caller."""

    provider: str
    api_url: str
    auth_source: Literal["codex-login", "environment", "explicit"]
    api_key_environment_variable: Optional[str] = None

    def __post_init__(self) -> None:
        _require_identifier(self.provider, "provider")
        _require_identifier(self.api_url, "api_url")
        if self.auth_source not in ("codex-login", "environment", "explicit"):
            raise ValueError("unsupported catalog authentication source")
        if self.auth_source == "environment":
            _require_identifier(self.api_key_environment_variable, "api_key_environment_variable")
        elif self.api_key_environment_variable is not None:
            raise ValueError("an environment variable requires environment authentication")


@dataclass(frozen=True)
class ResponsesDefaults:
    """Pythia request preferences, not claims about a model's native defaults."""

    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    text_verbosity: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("reasoning_effort", "reasoning_summary", "text_verbosity"):
            value = getattr(self, name)
            if value is not None:
                _require_identifier(value, name)


@dataclass(frozen=True)
class ModelSpec:
    """One selectable preset, plus any spellings with identical semantics."""

    profile: str
    name: str
    api_model: str
    route: ModelRoute
    limits: ModelLimits = field(default_factory=ModelLimits)
    responses: Optional[ResponsesDefaults] = None
    aliases: Tuple[str, ...] = ()
    source: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", _normalize_profile(self.profile))
        _require_identifier(self.name, "name")
        _require_identifier(self.api_model, "api_model")
        if not isinstance(self.route, ModelRoute):
            raise TypeError("route must be ModelRoute")
        if not isinstance(self.limits, ModelLimits):
            raise TypeError("limits must be ModelLimits")
        if self.responses is not None:
            if not isinstance(self.responses, ResponsesDefaults):
                raise TypeError("responses must be ResponsesDefaults or None")
            if self.profile not in {"codex", "responses"}:
                raise ValueError("Responses defaults require a Responses routing profile")
        if isinstance(self.aliases, (str, bytes)):
            raise TypeError("aliases must be an iterable of strings")
        aliases = tuple(self.aliases)
        for alias in aliases:
            _require_identifier(alias, "alias")
        object.__setattr__(self, "aliases", aliases)
        if self.source is not None and (not isinstance(self.source, str) or not self.source.strip()):
            raise ValueError("source must be a nonempty string or None")


_CHATGPT = ModelRoute("chatgpt", CODEX_RESPONSES_API_URL, "codex-login")
_META = ModelRoute("meta", META_RESPONSES_API_URL, "environment", "META_API_KEY")
_ANTHROPIC = ModelRoute(
    "anthropic", ANTHROPIC_MESSAGES_API_URL, "environment", "ANTHROPIC_API_KEY",
)
_PROFILE_DEFAULT_ROUTES = MappingProxyType({
    "codex": _CHATGPT,
    "responses": ModelRoute("api", OPENAI_RESPONSES_API_URL, "explicit"),
    "messages": _ANTHROPIC,
    "chat-completions": ModelRoute("api", "http://127.0.0.1:8000", "explicit"),
})

# Shared immutable facts, inherited by effort presets rather than copied.
_CODEX_LIMITS = ModelLimits(default_context_tokens=272_000, max_context_tokens=872_000)
_SOL = ModelSpec(
    profile="codex", name="gpt-5.6-sol", api_model="gpt-5.6-sol", route=_CHATGPT,
    limits=_CODEX_LIMITS, responses=ResponsesDefaults(),
    source="codex-latest-20260904/codex-rs/models-manager/models.json (context capacities)",
)
_ASTRA = replace(
    _SOL, name="gpt-6-astra", api_model="gpt-6-astra",
    # Pythia deliberately requests summaries; the bundled catalog default is none.
    responses=ResponsesDefaults(reasoning_summary="auto", text_verbosity="low"),
)
_SPARK = ModelSpec(
    profile="codex", name="muse-spark-1.3", api_model="muse-spark-1.3-contributor",
    route=_META, responses=ResponsesDefaults(),
    source="Existing Pythia Meta Responses integration presets; token capacities unknown",
)
_FABLE = ModelSpec(
    profile="messages", name="claude-fable-5-1", api_model="claude-fable-5-1",
    route=_ANTHROPIC, limits=ModelLimits(max_context_tokens=1_000_000, max_output_tokens=128_000),
    aliases=("claude-fable-5.1",),
    source="https://platform.claude.com/docs/en/models/fable-5-1/overview",
)


def _with_effort(base: ModelSpec, name: str, effort: str) -> ModelSpec:
    return replace(
        base, name=name, aliases=(),
        responses=replace(base.responses or ResponsesDefaults(), reasoning_effort=effort),
    )


_MODEL_SPECS = (
    _SOL,
    _with_effort(_SOL, "gpt-5.6-sol-medium", "medium"),
    _with_effort(_SOL, "gpt-5.6-sol-max", "max"),
    _ASTRA,
    _with_effort(_ASTRA, "gpt-6-astra-medium", "medium"),
    _with_effort(_ASTRA, "gpt-6-astra-max", "max"),
    _SPARK,
    _with_effort(_SPARK, "muse-spark-1.3-xhigh", "xhigh"),
    _FABLE,
)


def _build_index(specs: Iterable[ModelSpec]) -> Mapping[Tuple[str, str], ModelSpec]:
    index = {}
    for spec in specs:
        if not isinstance(spec, ModelSpec):
            raise TypeError("catalog entries must be ModelSpec")
        for name in (spec.name, *spec.aliases):
            key = (spec.profile, name)
            if key in index:
                raise ValueError(f"duplicate model selector in {spec.profile}: {name}")
            index[key] = spec
    return MappingProxyType(index)


_MODEL_INDEX = _build_index(_MODEL_SPECS)


def get_model_spec(profile: str, name: Optional[str]) -> Optional[ModelSpec]:
    """Look up an exact, case-sensitive selector; unknown models return None.

    Outer whitespace is stripped just as endpoint constructors strip it.
    ``codex-responses`` is a profile synonym for ``codex``. Generic Responses
    (``responses``) and Chat Completions do not inherit Codex presets.
    """
    profile = _normalize_profile(profile)
    if name is None:
        return None
    if not isinstance(name, str):
        raise TypeError("model name must be a string or None")
    return _MODEL_INDEX.get((profile, name.strip()))


def get_model_route(profile: str, name: Optional[str] = None) -> ModelRoute:
    """Return known model defaults or the profile's pass-through defaults."""
    profile = _normalize_profile(profile)
    spec = get_model_spec(profile, name)
    return spec.route if spec is not None else _PROFILE_DEFAULT_ROUTES[profile]


def list_model_specs(profile: Optional[str] = None) -> Tuple[ModelSpec, ...]:
    """List canonical presets in declaration order; aliases do not duplicate entries."""
    if profile is None:
        return _MODEL_SPECS
    profile = _normalize_profile(profile)
    return tuple(spec for spec in _MODEL_SPECS if spec.profile == profile)


__all__ = [
    "ANTHROPIC_MESSAGES_API_URL",
    "CODEX_RESPONSES_API_URL",
    "META_RESPONSES_API_URL",
    "OPENAI_RESPONSES_API_URL",
    "ModelLimits",
    "ModelRoute",
    "ModelSpec",
    "ResponsesDefaults",
    "get_model_route",
    "get_model_spec",
    "list_model_specs",
]
