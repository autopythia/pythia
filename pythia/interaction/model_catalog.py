"""Static, profile-scoped model facts and Pythia request presets.

This module has no adapter, credential, environment, or network dependencies.
Maximum/output limits are metadata rather than request budgets;
``auto_compact_context_tokens`` is caller policy consumed by interaction
frontends and server-compaction configuration. Routes describe defaults, not
resolved credentials or permission to use account services. Unknown names
remain valid pass-through candidates for the owning adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
import hashlib
import json
import math
import re
from types import MappingProxyType
from collections.abc import Mapping as MappingABC
from typing import Iterable
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Tuple
from urllib.parse import urlsplit


OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1"
CODEX_RESPONSES_API_URL = "https://chatgpt.com/backend-api/codex"
META_RESPONSES_API_URL = "https://api.meta.ai/v1"
ANTHROPIC_MESSAGES_API_URL = "https://api.anthropic.com"
# Shared protocol constraint for catalog validation and Messages request policy.
MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS = 50_000

_PROFILES = frozenset(("codex", "responses", "messages", "chat-completions"))

# Extensions are not a second path for typed policy or adapter-owned structure.
_RESERVED_REQUEST_PARAMS = frozenset((
    "model", "messages", "input", "instructions", "stream", "stream_options",
    "tools", "parallel_tool_calls", "max_tokens", "max_output_tokens",
    "max_completion_tokens", "max_new_tokens", "temperature", "top_p", "stop",
    "stop_sequences", "seed", "enable_auto_compaction", "auto_compact_tokens",
    "max_context_tokens", "context_management", "api", "model_api", "api_model",
    "api_url", "api_key", "api_key_env", "headers", "authorization",
    "bearer_token", "access_token", "account_id", "request_timeout_seconds",
))
MAX_REQUEST_PARAMS_BYTES = 65_536


def parse_json_value(text: str):
    """Strict JSON, also used for structured INI values; never evaluate Python."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("non-finite JSON number")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, RecursionError):
        raise ValueError("Invalid JSON value.") from None


def _freeze_json(value, depth=0):
    if depth > 32:
        raise ValueError("Request params exceed maximum nesting depth.")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, MappingABC):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Request param object keys must be strings.")
        return MappingProxyType({key: _freeze_json(item, depth + 1) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, depth + 1) for item in value)
    raise ValueError("Request params must contain only finite JSON values.")


def thaw_json(value):
    """Return fresh JSON containers; never expose a frozen config's internals."""
    if isinstance(value, MappingABC):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def freeze_request_params(value, profile=None):
    if not isinstance(value, MappingABC):
        raise ValueError("request_params must be an object.")
    if any(not isinstance(key, str) or not key or key.lower() in _RESERVED_REQUEST_PARAMS
           or any(ord(char) < 32 for char in key) for key in value):
        raise ValueError("Invalid or adapter-owned request parameter.")
    if profile is not None and _normalize_profile(profile) != "chat-completions" and value:
        raise ValueError("request_params currently requires the Chat Completions API.")
    frozen = _freeze_json(value)
    try:
        encoded = json.dumps(thaw_json(frozen), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("Request params are not valid JSON.") from None
    if len(encoded) > MAX_REQUEST_PARAMS_BYTES:
        raise ValueError("Request params exceed the size limit.")
    return frozen


def validate_route_url(value):
    """Validate external route data without constructing a credentialed adapter."""
    try:
        if not isinstance(value, str) or not value or any(char.isspace() or ord(char) < 32 for char in value):
            raise ValueError()
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or parsed.port == 0):
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("Route URL must be HTTP(S), without credentials, query, or fragment.") from None
    return value.rstrip("/")


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


_LEGACY_SUFFIXES = {
    "chat-completions": "/v1/chat/completions", "messages": "/v1/messages",
    "responses": "/responses", "codex": "/responses",
}


def legacy_endpoint_url(api, prefix):
    """Lossless v1 conversion; never guess whether a prefix already contains /v1."""
    return validate_route_url(prefix) + _LEGACY_SUFFIXES[_normalize_profile(api)]


@dataclass(frozen=True)
class EndpointSpec:
    """The single non-secret delivery authority. url is a complete POST URL."""

    api: str
    url: str
    model: Optional[str] = None
    auth: str = "none"
    # Resolved login reference, not file contents. Filled at the launch boundary.
    auth_file: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "api", _normalize_profile(self.api))
        validate_route_url(self.url)
        if self.model is not None:
            _require_identifier(self.model, "endpoint.model")
        if not isinstance(self.auth, str) or not (
            self.auth in {"none", "supplied", "codex-login"}
            or (self.auth.startswith("env:") and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.auth[4:]))
        ):
            raise ValueError("endpoint.auth must be none, supplied, codex-login, or env:NAME")
        if self.auth == "codex-login" and self.api != "codex":
            raise ValueError("codex-login requires the Codex API")
        if self.auth_file is not None:
            if self.auth != "codex-login" or not isinstance(self.auth_file, str) or not self.auth_file.strip() or "\x00" in self.auth_file:
                raise ValueError("auth_file requires a nonempty Codex login reference")

    @property
    def environment_variable(self):
        return self.auth[4:] if self.auth.startswith("env:") else None

    @property
    def connection_identity(self):
        return self.api, self.url, self.auth, self.auth_file

    @property
    def is_official_codex(self):
        return self.api == "codex" and self.url.rstrip("/") == CODEX_RESPONSES_API_URL + "/responses"

    @property
    def legacy_api_url(self):
        suffix = _LEGACY_SUFFIXES[self.api]
        if not self.url.endswith(suffix):
            raise ValueError("This exact endpoint URL has no legacy prefix; use EndpointSpec.url")
        return self.url[:-len(suffix)]

    def as_dict(self):
        return {"api": self.api, "url": self.url, "model": self.model,
                "auth": self.auth, "auth_file": self.auth_file}

    @classmethod
    def from_route(cls, api, model, route):
        api = _normalize_profile(api)
        auth = ("env:" + route.api_key_environment_variable if route.auth_source == "environment"
                else "codex-login" if route.auth_source == "codex-login"
                else "none" if api in {"chat-completions", "messages"} else "supplied")
        return cls(api, legacy_endpoint_url(api, route.api_url), model, auth)

    def legacy_route(self, provider="api"):
        return ModelRoute(provider, self.legacy_api_url,
                          "environment" if self.environment_variable else
                          "codex-login" if self.auth == "codex-login" else "explicit",
                          self.environment_variable)


@dataclass(frozen=True)
class ModelLimits:
    """Context policy and known ceilings for one catalogued model."""

    auto_compact_context_tokens: Optional[int] = None
    max_context_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        for name in (
            "auto_compact_context_tokens",
            "max_context_tokens",
            "max_output_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if (
            self.auto_compact_context_tokens is not None
            and self.max_context_tokens is not None
            and self.auto_compact_context_tokens > self.max_context_tokens
        ):
            raise ValueError(
                "auto-compact context must not exceed maximum context"
            )


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
class MessagesDefaults:
    """Pythia Messages request preferences, not native model defaults."""

    output_effort: Optional[str] = None

    def __post_init__(self) -> None:
        if self.output_effort is not None:
            _require_identifier(self.output_effort, "output_effort")


_UNSET_SPEC_FIELD = object()


@dataclass(frozen=True, init=False)
class ModelSpec:
    """A named policy and one endpoint. Legacy constructor fields are inputs only."""

    name: str
    endpoint: EndpointSpec
    limits: ModelLimits = field(default_factory=ModelLimits)
    responses: Optional[ResponsesDefaults] = None
    aliases: Tuple[str, ...] = ()
    source: Optional[str] = None
    messages: Optional[MessagesDefaults] = None
    request_params: Mapping = field(default_factory=dict)
    provider_label: str = "api"
    _legacy_route: Optional[ModelRoute] = field(default=None, init=False, repr=False, compare=False)

    def __init__(self, profile=_UNSET_SPEC_FIELD, name=None, api_model=_UNSET_SPEC_FIELD,
                 route=_UNSET_SPEC_FIELD, limits=_UNSET_SPEC_FIELD,
                 responses=None, aliases=(), source=None, messages=None,
                 request_params=_UNSET_SPEC_FIELD, *, endpoint=None, provider_label="api"):
        if endpoint is None:
            if not isinstance(route, ModelRoute):
                raise TypeError("route must be ModelRoute when endpoint is omitted")
            endpoint = EndpointSpec.from_route(profile, api_model, route)
            provider_label = route.provider
        else:
            if not isinstance(endpoint, EndpointSpec):
                raise TypeError("endpoint must be EndpointSpec")
            # dataclasses.replace with a legacy keyword remains supported.
            if profile is not _UNSET_SPEC_FIELD:
                endpoint = replace(endpoint, api=_normalize_profile(profile))
            if api_model is not _UNSET_SPEC_FIELD:
                endpoint = replace(endpoint, model=api_model)
            if route is not _UNSET_SPEC_FIELD:
                if not isinstance(route, ModelRoute):
                    raise TypeError("route must be ModelRoute")
                endpoint = EndpointSpec.from_route(endpoint.api, endpoint.model, route)
                provider_label = route.provider
        for key, value in (("name", name), ("endpoint", endpoint),
                           ("limits", ModelLimits() if limits is _UNSET_SPEC_FIELD else limits),
                           ("responses", responses), ("aliases", aliases), ("source", source),
                           ("messages", messages), ("request_params", {} if request_params is _UNSET_SPEC_FIELD else request_params),
                           ("provider_label", provider_label)):
            object.__setattr__(self, key, value)
        try:
            legacy = endpoint.legacy_route(provider_label)
        except ValueError:
            legacy = None
        object.__setattr__(self, "_legacy_route", legacy)
        self.__post_init__()

    @property
    def profile(self):
        return self.endpoint.api

    @property
    def api_model(self):
        return self.endpoint.model

    @property
    def route(self):
        if self._legacy_route is None:
            raise ValueError("Use spec.endpoint for an exact, non-legacy URL")
        return self._legacy_route

    def __post_init__(self) -> None:
        _require_identifier(self.name, "name")
        _require_identifier(self.api_model, "api_model")
        if not isinstance(self.limits, ModelLimits):
            raise TypeError("limits must be ModelLimits")
        if (self.profile == "messages" and self.limits.auto_compact_context_tokens is not None
                and self.limits.auto_compact_context_tokens < MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS):
            raise ValueError("Messages compaction threshold must be at least 50000.")
        if self.responses is not None:
            if not isinstance(self.responses, ResponsesDefaults):
                raise TypeError("responses must be ResponsesDefaults or None")
            if self.profile not in {"codex", "responses"}:
                raise ValueError("Responses defaults require a Responses routing profile")
        if self.messages is not None:
            if not isinstance(self.messages, MessagesDefaults):
                raise TypeError("messages must be MessagesDefaults or None")
            if self.profile != "messages":
                raise ValueError(
                    "Messages defaults require a Messages routing profile"
                )
        if isinstance(self.aliases, (str, bytes)):
            raise TypeError("aliases must be an iterable of strings")
        aliases = tuple(self.aliases)
        for alias in aliases:
            _require_identifier(alias, "alias")
        object.__setattr__(self, "aliases", aliases)
        if self.source is not None and (not isinstance(self.source, str) or not self.source.strip()):
            raise ValueError("source must be a nonempty string or None")
        object.__setattr__(self, "request_params", freeze_request_params(self.request_params, self.profile))


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
_CODEX_LIMITS = ModelLimits(
    auto_compact_context_tokens=872_000,
    max_context_tokens=1_000_000,
    max_output_tokens=128_000,
)
_SOL = ModelSpec(
    profile="codex", name="gpt-5.6-sol", api_model="gpt-5.6-sol", route=_CHATGPT,
    limits=_CODEX_LIMITS, responses=ResponsesDefaults(),
    source=(
        "codex-latest-20260904/codex-rs/models-manager/models.json; "
        "Pythia auto-compaction/max-context policy override"
    ),
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
    route=_ANTHROPIC,
    limits=ModelLimits(
        auto_compact_context_tokens=872_000,
        max_context_tokens=1_000_000,
        max_output_tokens=128_000,
    ),
    aliases=("claude-fable-5.1",),
    source=(
        "https://platform.claude.com/docs/en/models/fable-5-1/overview; "
        "https://platform.claude.com/docs/en/build-with-claude/effort"
    ),
)


def _with_effort(base: ModelSpec, name: str, effort: str) -> ModelSpec:
    return replace(
        base, name=name, aliases=(),
        responses=replace(base.responses or ResponsesDefaults(), reasoning_effort=effort),
    )


def _with_messages_effort(
    base: ModelSpec,
    name: str,
    effort: str,
    *,
    aliases: Tuple[str, ...] = (),
) -> ModelSpec:
    return replace(
        base,
        name=name,
        aliases=aliases,
        messages=MessagesDefaults(
            output_effort=effort,
        ),
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
    _with_messages_effort(
        _FABLE,
        "claude-fable-5-1-max",
        "max",
        aliases=("claude-fable-5.1-max",),
    ),
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


@dataclass(frozen=True)
class ModelBinding:
    """One resolved endpoint, including explicitly bound pass-through models."""

    selector: Optional[str]
    endpoint: EndpointSpec
    spec: Optional[ModelSpec] = None
    request_params: Mapping = field(default_factory=dict)
    origin: str = "builtin"
    api_explicit: bool = True
    endpoint_overrides: frozenset = field(default_factory=frozenset, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.endpoint, EndpointSpec):
            raise TypeError("binding endpoint must be EndpointSpec")
        if not isinstance(self.api_explicit, bool):
            raise TypeError("api_explicit must be a bool")
        if not isinstance(self.origin, str) or not self.origin:
            raise TypeError("binding origin must be a nonempty string")
        if self.selector is not None and not isinstance(self.selector, str):
            raise TypeError("selector must be a string or None")
        if self.spec is not None:
            if not isinstance(self.spec, ModelSpec) or self.spec.profile != self.api:
                raise ValueError("binding spec must match the selected API")
            if self.selector not in (self.spec.name, *self.spec.aliases):
                raise ValueError("binding selector does not select its spec")
        object.__setattr__(self, "request_params", freeze_request_params(self.request_params, self.api))
        object.__setattr__(self, "endpoint_overrides", frozenset(self.endpoint_overrides))

    @property
    def api(self):
        return self.endpoint.api

    @property
    def api_model(self):
        return self.endpoint.model

    @property
    def route(self):
        """Legacy read-only view; never used for dispatch or capability decisions."""
        return self.endpoint.legacy_route("api" if self.spec is None else self.spec.provider_label)

    @property
    def limits(self):
        return ModelLimits() if self.spec is None else self.spec.limits

    @property
    def supports_account_services(self):
        return self.endpoint.is_official_codex and self.endpoint.auth == "codex-login"

    @property
    def supports_remote_compaction(self):
        return self.endpoint.is_official_codex and self.endpoint.auth in {"codex-login", "supplied"}

    def with_request_params(self, overlay=None):
        params = dict(self.request_params)
        if overlay is not None:
            params.update(freeze_request_params(overlay, self.api))
        return replace(self, request_params=params)

    def manifest_entry(self):
        spec = self.spec
        facts = {
            "api": self.api, "selector": self.selector, "endpoint": self.endpoint.as_dict(),
            "limits": {name: getattr(self.limits, name) for name in (
                "auto_compact_context_tokens", "max_context_tokens", "max_output_tokens",
            )},
            "request_params": thaw_json(self.request_params),
            "responses": None if spec is None or spec.responses is None else vars(spec.responses),
            "messages": None if spec is None or spec.messages is None else vars(spec.messages),
        }
        fingerprint = hashlib.sha256(json.dumps(facts, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return {"api": self.api, "selector": self.selector, "api_model": self.api_model,
                "canonical": None if spec is None else spec.name, "source": self.origin,
                "fingerprint": fingerprint, "endpoint": self.endpoint.as_dict()}


@dataclass(frozen=True)
class ModelCatalog:
    """Immutable registry. File loading belongs to model_catalog_config, not here."""

    specs: Tuple[ModelSpec, ...] = ()
    origins: Mapping = field(default_factory=dict, repr=False)
    _index: Mapping = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        specs = tuple(self.specs)
        index = _build_index(specs)
        if not isinstance(self.origins, MappingABC) or any(
            not isinstance(value, str) or not value for value in self.origins.values()
        ):
            raise TypeError("catalog origins must be nonempty strings")
        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "origins", MappingProxyType({
            (spec.profile, spec.name): self.origins.get((spec.profile, spec.name), "python")
            for spec in specs
        }))

    def get_model_spec(self, api, name):
        api = _normalize_profile(api)
        if name is None:
            return None
        if not isinstance(name, str):
            raise TypeError("model name must be a string or None")
        return self._index.get((api, name.strip()))

    def get_model_route(self, api, name=None):
        api = _normalize_profile(api)
        spec = self.get_model_spec(api, name)
        return _PROFILE_DEFAULT_ROUTES[api] if spec is None else spec.route

    def list_model_specs(self, api=None):
        if api is None:
            return self.specs
        api = _normalize_profile(api)
        return tuple(spec for spec in self.specs if spec.profile == api)

    def matches(self, name, *, canonical_only=False):
        if name is None:
            return ()
        if not isinstance(name, str):
            raise TypeError("model name must be a string or None")
        name = name.strip()
        return tuple(spec for spec in self.specs if name == spec.name or (
            not canonical_only and name in spec.aliases
        ))

    def bind(self, api=None, name=None, *, api_url=None, request_params=None,
             endpoint_url=None, endpoint_model=None, endpoint_auth=None):
        if name is not None:
            if not isinstance(name, str):
                raise TypeError("model name must be a string or None")
            name = name.strip() or None
        explicit = api is not None
        if api is None:
            matches = self.matches(name)
            if len(matches) > 1:
                raise ValueError("Ambiguous catalog model; select its API with --endpoint-api.")
            api = matches[0].profile if matches else "chat-completions"
        api = _normalize_profile(api)
        spec = self.get_model_spec(api, name)
        endpoint = (EndpointSpec.from_route(api, name, _PROFILE_DEFAULT_ROUTES[api])
                    if spec is None else spec.endpoint)
        if api_url is not None and endpoint_url is not None:
            raise ValueError("Use --endpoint-url or legacy --api-url, not both.")
        changes = {}
        if api_url is not None:
            changes["url"] = legacy_endpoint_url(api, api_url)
        if endpoint_url is not None:
            if endpoint_url != endpoint.url and endpoint.auth != "none" and endpoint_auth is None:
                raise ValueError("Changing a credentialed endpoint URL requires explicit --endpoint-auth.")
            changes["url"] = endpoint_url
        if endpoint_model is not None:
            changes["model"] = endpoint_model
        if endpoint_auth is not None:
            changes.update(auth=endpoint_auth, auth_file=None)
        endpoint = replace(endpoint, **changes)
        binding = ModelBinding(
            name, endpoint, spec,
            {} if spec is None else spec.request_params,
            "builtin" if spec is None else self.origins[(spec.profile, spec.name)],
            explicit, frozenset(changes),
        )
        return binding.with_request_params(request_params)


BUILTIN_MODEL_CATALOG = ModelCatalog(
    _MODEL_SPECS, {(spec.profile, spec.name): "builtin" for spec in _MODEL_SPECS},
)


def binding_from_namespace(args, catalog=None):
    """Prepared bindings are authoritative; raw args cannot reconfigure them."""
    existing = getattr(args, "model_binding", None)
    if isinstance(existing, ModelBinding):
        return existing
    if existing is not None:
        raise TypeError("model_binding must be ModelBinding or None")
    if catalog is None and getattr(args, "model_catalog", None) is not None:
        raise ValueError("Load and bind the explicit model catalog before resolving model configuration.")
    catalog = BUILTIN_MODEL_CATALOG if catalog is None else catalog
    auth = getattr(args, "endpoint_auth", None)
    variable = getattr(args, "api_key_env", None)
    supplied = getattr(args, "api_key", None) is not None
    inferred_supplied = supplied and auth is None and variable is None
    if variable is not None:
        if auth is not None:
            raise ValueError("Use endpoint_auth or api_key_env, not both.")
        auth = "env:" + variable
    if supplied:
        if auth is not None and auth != "supplied":
            raise ValueError("A supplied API key requires endpoint-auth supplied.")
        auth = "supplied"
    binding = catalog.bind(
        getattr(args, "model_api", None), getattr(args, "model", None),
        api_url=getattr(args, "api_url", None),
        endpoint_url=getattr(args, "endpoint_url", None),
        endpoint_model=getattr(args, "endpoint_model", None),
        endpoint_auth=auth,
        request_params=getattr(args, "request_params", None),
    )
    if inferred_supplied and binding.api == "codex" and (
        binding.spec is None or binding.spec.endpoint.auth == "codex-login"
    ):
        raise ValueError("--api-key is not used with Codex login; select --endpoint-auth supplied explicitly.")
    return binding


def bind_endpoint(api, model, api_url, binding=None, request_params=None, *, auth=None):
    """Compatibility boundary for legacy transport constructors.

    With a binding, inputs are consistency assertions, never another override.
    Without one, old prefixes are converted exactly once.
    """
    api = _normalize_profile(api)
    if binding is None:
        result = BUILTIN_MODEL_CATALOG.bind(api, model, api_url=api_url, request_params=request_params)
        return result if auth is None else replace(result, endpoint=replace(result.endpoint, auth=auth, auth_file=None))
    if (not isinstance(binding, ModelBinding) or binding.api != api
            or model != (binding.selector if binding.selector is not None else binding.api_model)):
        raise ValueError("Endpoint identity does not match its model binding.")
    if api_url is not None and legacy_endpoint_url(api, api_url) != binding.endpoint.url:
        raise ValueError("URL conflicts with resolved endpoint; construct a new binding explicitly.")
    if request_params is not None and freeze_request_params(request_params, api) != binding.request_params:
        raise ValueError("Request params conflict with resolved binding.")
    return binding


def bound_transport_inputs(api, model, api_url, binding):
    """Project compatibility fields from an authoritative endpoint before validation."""
    if binding is None:
        return model, api_url
    if model is None:
        model = binding.selector if binding.selector is not None else binding.api_model
    bind_endpoint(api, model, api_url, binding)
    try:
        prefix = binding.endpoint.legacy_api_url
    except ValueError:
        prefix = None  # An arbitrary exact URL cannot be represented as an old prefix.
    return model, prefix


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
    "ModelCatalog",
    "ModelBinding",
    "BUILTIN_MODEL_CATALOG",
    "EndpointSpec",
    "ModelRoute",
    "ModelSpec",
    "MessagesDefaults",
    "ResponsesDefaults",
    "get_model_route",
    "get_model_spec",
    "list_model_specs",
]
