"""Resolved, non-secret configuration for one interaction context binding."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
import json
import re
from threading import RLock
from types import SimpleNamespace
from typing import Dict
from typing import Optional
from typing import Union
from typing import Mapping

from .compaction import compaction_methods
from .compaction import format_compaction_mode
from .compaction import normalize_compaction_mode
from .compaction import parse_compaction_mode
from .compaction import CompactionSettings
from .compaction import DEFAULT_COMPACTION_MODE
from .compaction import DEFAULT_KEEP_RECENT_TOKENS
from .messages import MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS
from .messages import resolve_messages_max_output_tokens
from .model import SampleParams
from .model_catalog import ModelBinding
from .model_catalog import CompactionDefaults
from .model_catalog import binding_from_namespace, freeze_extra_sample_params, thaw_json
from .timeouts import DEFAULT_REQUEST_TIMEOUT_SECONDS
from .timeouts import validate_timeout_seconds


ConfigValue = Union[bool, int, float, str, None, Mapping]
CONFIG_KEYS = (
    "enable_workspace",
    "max_samples",
    "max_output_tokens",
    "request_timeout_seconds",
    "enable_auto_compaction",
    "auto_compaction_context_tokens",
    "max_context_tokens",
    "compaction_mode",
    "compaction_pi_keep_recent_tokens",
    "compaction_pi_max_output_tokens",
    "compaction_coldstart_reboot",
    "extra_sample_params",
)
# Keys renamed in October 2026, from their old names. The old names keep
# working, with a warning that names the new one: as /config keys, as options,
# in auto's config files, and as demo.run keywords. Version-3 auto saves use
# them and load silently. Python fields and InteractionConfig's keyword
# arguments have only the new names.
RENAMED_CONFIG_KEYS = {
    "auto_compact_tokens": "auto_compaction_context_tokens",
    "compaction_keep_recent_tokens": "compaction_pi_keep_recent_tokens",
    "compaction_max_output_tokens": "compaction_pi_max_output_tokens",
}
_BOOLEAN_KEYS = frozenset((
    "enable_workspace", "enable_auto_compaction", "compaction_coldstart_reboot",
))
_OPTIONAL_POSITIVE_INTEGER_KEYS = frozenset((
    "max_samples",
    "max_output_tokens",
    "auto_compaction_context_tokens",
    "max_context_tokens",
    "compaction_pi_max_output_tokens",
))
# ``null`` restores the default through the config's fallback.
_OPTIONAL_NONNEGATIVE_INTEGER_KEYS = frozenset(("compaction_pi_keep_recent_tokens",))
# Optional seconds; ``null`` restores the default through the config's fallback.
_OPTIONAL_SECONDS_KEYS = frozenset(("request_timeout_seconds",))
# Readable in /config, but only set at launch.
_LAUNCH_ONLY_KEYS = frozenset(("extra_sample_params",))
_INTEGER_LITERAL_RE = re.compile(r"^[+-]?[0-9]+$")
# Seconds also accept plain decimals; no exponent, infinity, or NaN spellings.
_DECIMAL_LITERAL_RE = re.compile(r"^[+-]?[0-9]+\.[0-9]+$")
_NO_REQUEST_TIMEOUT = (
    "request_timeout_seconds does not apply to Claude Relay; use the model "
    "catalog's timeouts.generation_seconds or --claude-relay-generation-timeout."
)


class ConfigError(ValueError):
    pass


def _launch_only_error(key: str) -> ConfigError:
    return ConfigError(f"{key} is launch-only; use /config.json to inspect it.")


def _require_key(key: object) -> str:
    if not isinstance(key, str) or key not in CONFIG_KEYS:
        raise ConfigError(
            "Unknown config key; use /config to list supported keys."
        )
    return key


def validate_config_value(key: str, value: object) -> ConfigValue:
    key = _require_key(key)
    if key == "extra_sample_params":
        try:
            return freeze_extra_sample_params(value)
        except ValueError as exc:
            raise ConfigError(str(exc)) from None
    if key == "compaction_mode":
        # A compaction sequence, kept in its normal form. None is the route's
        # default, resolved when a config is bound.
        if value is None:
            return None
        try:
            return normalize_compaction_mode(value)
        except ValueError as exc:
            raise ConfigError(str(exc)) from None
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
    if key in _OPTIONAL_NONNEGATIVE_INTEGER_KEYS:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(
                f"{key} requires a nonnegative integer or None."
            )
        return value
    if key in _OPTIONAL_SECONDS_KEYS:
        if value is None:
            return None
        try:
            return validate_timeout_seconds(value, key)
        except ValueError as exc:
            raise ConfigError(str(exc)) from None
    raise AssertionError(f"missing config validator for {key}")


def parse_config_literal(key: str, text: object) -> ConfigValue:
    """Parse the intentionally small JSON/Python scalar input grammar.

    Integers, booleans, and null for every key; seconds keys also take plain
    decimals such as ``90.5``. ``compaction_mode`` takes a compaction
    sequence, such as ``coldstart,pi:2``, or null.
    """
    key = _require_key(key)
    if key in _LAUNCH_ONLY_KEYS:
        raise _launch_only_error(key)
    if not isinstance(text, str):
        raise ConfigError(f"Invalid value for {key}.")
    literal = text.strip()
    if key == "compaction_mode":
        return validate_config_value(key, None if literal in ("null", "None") else literal)
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
    elif key in _OPTIONAL_SECONDS_KEYS and _DECIMAL_LITERAL_RE.fullmatch(literal) is not None:
        value = float(literal)
    else:
        raise ConfigError(f"Invalid value for {key}.")
    return validate_config_value(key, value)


_NO_PROVIDER = "this route has no provider compaction; use --compaction-mode pi"


@dataclass(frozen=True)
class CompactionRoute:
    """What one session's route allows in ``compaction_mode``, fixed at launch.

    ``default`` is the route's own method, which ``legacydefault`` entries
    name. ``provider`` says whether the route has provider compaction. On
    Messages (``messages``), ``provider`` turns on server-side compaction,
    which runs inside samples, so it may only come first. ``server``, once
    known, is whether this session's Messages endpoint carries server-side
    compaction: set at launch, so a later change can't add or remove
    ``provider``.
    """

    default: str = "pi"
    provider: bool = False
    messages: bool = False
    server: Optional[bool] = None

    def resolve(self, value: object) -> str:
        """Resolve ``legacydefault`` entries and check the route; return the normal form."""
        try:
            entries = tuple(
                (self.default if method == "legacydefault" else method, attempts)
                for method, attempts in parse_compaction_mode(value)
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from None
        methods = [method for method, _attempts in entries]
        if "provider" in methods:
            if not self.provider:
                raise ConfigError(_NO_PROVIDER)
            if self.messages and "provider" in methods[1:]:
                raise ConfigError(
                    "On Messages, provider may only come first in compaction_mode: it "
                    "turns on server-side compaction, which runs inside samples."
                )
        if self.server is not None and (methods[0] == "provider") != self.server:
            raise ConfigError(
                "compaction_mode can't add or remove provider on this Messages endpoint: "
                f"its server-side compaction is {'on' if self.server else 'off'}, and that "
                "is fixed when the endpoint is built."
            )
        return format_compaction_mode(entries)

    @property
    def default_sequence(self) -> str:
        """What an unset ``compaction_mode`` is on this route, resolved.

        ``DEFAULT_COMPACTION_MODE``: coldstart, then warmpi up to twice, the
        official Codex route included. On a Messages endpoint with server-side
        compaction it is ``provider`` alone, since provider may only come
        first there.
        """
        if self.messages and self.default == "provider":
            return "provider"
        return self.resolve(DEFAULT_COMPACTION_MODE)


def _binding_route(binding, mode: Optional[str] = None) -> CompactionRoute:
    """A binding's route; with the launch ``mode``, also its Messages server state."""
    remote = bool(binding.supports_remote_compaction)
    messages = binding.api == "messages"
    server = None
    if messages and mode is not None:
        server = compaction_methods(mode)[0] == "provider"
    return CompactionRoute(
        default="provider" if remote or server else "pi",
        provider=messages or remote,
        messages=messages,
        server=server,
    )


def resolve_compaction_mode(binding, requested: Optional[str] = None) -> str:
    """Resolve ``--compaction-mode`` for one model binding.

    The value is a compaction sequence (``parse_compaction_mode``). Without a
    request, it is the model catalog's value (``ModelBinding.compaction``),
    else the default sequence: coldstart, then warmpi up to twice. A catalog
    value that can't run on this route fails, naming the catalog. The route's
    own method, which ``legacydefault`` entries name, is ``provider`` (remote
    compaction) on the official ChatGPT/Codex route and ``pi`` elsewhere.
    ``provider`` is also allowed for Messages, as
    Anthropic server-side compaction, but only first. A Codex-compatible
    third-party endpoint is not assumed to support remote compaction.
    """
    route = _binding_route(binding)
    requested = validate_config_value("compaction_mode", requested)
    if requested is not None:
        return route.resolve(requested)
    configured = binding.compaction.mode
    if configured is None:
        return route.default_sequence
    try:
        return route.resolve(configured)
    except ConfigError as exc:
        raise ConfigError(
            f"compaction_mode {configured} (from the model catalog): {exc}") from None


def _catalog_default(route: CompactionRoute, configured: Optional[str]) -> str:
    """The route's default compaction sequence: the catalog's, if it can run there.

    ``configured`` is the model catalog's value, in its normal form. On a
    Messages endpoint, server-side compaction is fixed when the endpoint is
    built, so a value that would add or remove ``provider`` can't run; the
    route's built-in default (``default_sequence``) applies instead.
    """
    if configured is not None:
        try:
            return route.resolve(configured)
        except ConfigError:
            pass
    return route.default_sequence


def resolve_request_timeout_seconds(binding, requested=None) -> Optional[float]:
    """Resolve ``--request-timeout-seconds`` for one model binding.

    An explicit request wins, then the bound model's catalog
    ``timeouts.request_seconds``, then ``DEFAULT_REQUEST_TIMEOUT_SECONDS``.
    Claude Relay sends no HTTP model requests: it resolves to ``None``, and an
    explicit request is an error.
    """
    if binding.api == "claude-relay":
        if requested is not None:
            raise ConfigError(_NO_REQUEST_TIMEOUT)
        return None
    if requested is not None:
        return validate_config_value("request_timeout_seconds", requested)
    configured = binding.timeouts.request_seconds
    return DEFAULT_REQUEST_TIMEOUT_SECONDS if configured is None else configured


def _model_route(model) -> CompactionRoute:
    """A Python-built model's route, from its adapter and endpoint."""
    from .messages import MessagesModel
    from .responses import _ResponsesModelBase

    if isinstance(model, MessagesModel):
        configured = model.endpoint.server_compaction is not None
        return CompactionRoute(default="provider" if configured else "pi",
                               provider=True, messages=True, server=configured)
    remote = (
        isinstance(model, _ResponsesModelBase)
        and model.supports_remote_compaction
    )
    return CompactionRoute(default="provider" if remote else "pi", provider=bool(remote))


def _model_compaction_defaults(model) -> CompactionDefaults:
    """A Python-built model's catalog compaction settings, from its binding."""
    binding = getattr(model, "binding", None)
    return binding.compaction if isinstance(binding, ModelBinding) else CompactionDefaults()


def _model_compaction_mode(model, requested: Optional[str]) -> str:
    """Infer a Python-built model's mode, or check a requested one.

    On Messages, ``provider`` comes first exactly when the endpoint has
    server-side compaction (``MessagesEndpoint(server_compaction=...)``).
    Without a request, the binding's catalog value applies where the model
    can run it; the model is already built, so one it can't run is ignored.
    """
    route = _model_route(model)
    requested = validate_config_value("compaction_mode", requested)
    if requested is not None:
        return route.resolve(requested)
    return _catalog_default(route, _model_compaction_defaults(model).mode)


@dataclass(frozen=True)
class InteractionConfigSnapshot:
    """Immutable context values; snapshots published by config are resolved.

    ``extra_sample_params`` is a launch-only, read-only view of the model
    binding's map. It is not projected into sample params: requests inherit
    the binding's map, so samples and compactions send the same extensions.
    ``compaction_mode`` is a compaction sequence (``parse_compaction_mode``),
    such as ``coldstart,pi:2``; ``None`` means the route's default until a
    config binds it.

    ``request_timeout_seconds`` is projected into sample params as each
    request's HTTP timeout. ``None`` leaves the model endpoint's own timeout;
    namespace-bound configs always resolve it except on Claude Relay, where it
    does not apply.

    ``compaction_coldstart_reboot`` has the CLI restart itself on the current
    code after each coldstart compaction, as ``/reboot`` does. Only the CLI
    reboots; other hosts leave it False.
    """

    enable_workspace: bool = True
    max_samples: Optional[int] = None
    max_output_tokens: Optional[int] = None
    request_timeout_seconds: Optional[float] = None
    enable_auto_compaction: bool = True
    auto_compaction_context_tokens: Optional[int] = None
    max_context_tokens: Optional[int] = None
    compaction_mode: Optional[str] = None
    compaction_pi_keep_recent_tokens: Optional[int] = DEFAULT_KEEP_RECENT_TOKENS
    compaction_pi_max_output_tokens: Optional[int] = None
    compaction_coldstart_reboot: bool = False
    extra_sample_params: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in CONFIG_KEYS:
            object.__setattr__(self, key, validate_config_value(key, getattr(self, key)))

    def as_dict(self) -> Dict[str, ConfigValue]:
        return {key: thaw_json(getattr(self, key)) for key in CONFIG_KEYS}

    def sample_params(
        self, base: Optional[SampleParams] = None,
    ) -> SampleParams:
        """Project config-owned fields over a caller's other per-call params.

        Budget, request timeout, and compaction policy come from this snapshot.
        Other fields, including a per-call ``extra`` (usually None), come from
        ``base``.
        """
        if base is not None and not isinstance(base, SampleParams):
            raise TypeError("base must be SampleParams or None")
        return replace(
            base or SampleParams(),
            max_output_tokens=self.max_output_tokens,
            request_timeout_seconds=self.request_timeout_seconds,
            enable_auto_compaction=self.enable_auto_compaction,
            auto_compaction_context_tokens=self.auto_compaction_context_tokens,
        )

    def compaction_settings(self) -> CompactionSettings:
        """Compactor selection and pi options for this snapshot."""
        keep = self.compaction_pi_keep_recent_tokens
        return CompactionSettings(
            mode=self.compaction_mode or "pi",
            keep_recent_tokens=DEFAULT_KEEP_RECENT_TOKENS if keep is None else keep,
            max_output_tokens=self.compaction_pi_max_output_tokens,
        )


class InteractionConfig:
    """Thread-safe effective policy with immutable launch and default baselines.

    Factories bind raw frontend inputs to defaults. ``set(key, None)`` resolves
    against that captured default, while ``reset`` restores the launch value.
    Reads and request projection never consult models or the catalog.
    ``request_timeout_applies=False`` (Claude Relay) rejects any request
    timeout.

    ``compaction_route`` (from the factories) resolves ``legacydefault`` in
    ``compaction_mode`` and checks every value against the route; an unset
    value is the model catalog's (``compaction_mode_fallback``) where the
    route can run it, else the route's default sequence
    (``CompactionRoute.default_sequence``), and ``pi`` without a route. The
    other ``compaction_*_fallback`` values are the catalog's too, else the
    built-in defaults. With ``coldstart_unavailable``, the reason this
    session can't run coldstart compaction, the launch value and its fallback
    lose coldstart, and a change that asks for coldstart is refused.
    ``coldstart_dropped`` records a launch value that lost it, as the value
    before and after, unless that value was the default, the catalog's or the
    built-in one, which loses it silently.
    """

    def __init__(
        self,
        snapshot: InteractionConfigSnapshot = InteractionConfigSnapshot(),
        *,
        initial: Optional[InteractionConfigSnapshot] = None,
        on_enable_workspace: Optional[Callable[[bool], None]] = None,
        require_max_output_tokens: bool = False,
        max_output_tokens_fallback: Optional[int] = None,
        auto_compaction_context_tokens_fallback: Optional[int] = None,
        max_context_tokens_fallback: Optional[int] = None,
        min_auto_compaction_context_tokens: Optional[int] = None,
        request_timeout_seconds_fallback: Optional[float] = None,
        request_timeout_applies: bool = True,
        compaction_route: Optional[CompactionRoute] = None,
        coldstart_unavailable: Optional[str] = None,
        compaction_mode_fallback: Optional[str] = None,
        compaction_pi_keep_recent_tokens_fallback: Optional[int] = None,
        compaction_pi_max_output_tokens_fallback: Optional[int] = None,
    ) -> None:
        if not isinstance(snapshot, InteractionConfigSnapshot):
            raise TypeError("snapshot must be InteractionConfigSnapshot")
        if initial is not None and not isinstance(
            initial, InteractionConfigSnapshot
        ):
            raise TypeError("initial must be InteractionConfigSnapshot or None")
        if on_enable_workspace is not None and not callable(
            on_enable_workspace
        ):
            raise TypeError("on_enable_workspace must be callable or None")
        if not isinstance(require_max_output_tokens, bool):
            raise TypeError("require_max_output_tokens must be a bool")
        if not isinstance(request_timeout_applies, bool):
            raise TypeError("request_timeout_applies must be a bool")
        if compaction_route is not None and not isinstance(compaction_route, CompactionRoute):
            raise TypeError("compaction_route must be CompactionRoute or None")
        if coldstart_unavailable is not None and (
            not isinstance(coldstart_unavailable, str) or not coldstart_unavailable.strip()
        ):
            raise TypeError("coldstart_unavailable must be a nonempty string or None")
        compaction_mode_fallback = validate_config_value(
            "compaction_mode", compaction_mode_fallback)
        if compaction_route is None:
            mode_default = "pi" if compaction_mode_fallback is None else compaction_mode_fallback
        else:
            mode_default = _catalog_default(compaction_route, compaction_mode_fallback)
        if compaction_pi_keep_recent_tokens_fallback is None:
            compaction_pi_keep_recent_tokens_fallback = DEFAULT_KEEP_RECENT_TOKENS
        # Bound defaults are used only at initialization/mutation, never on read.
        self._fallbacks = {
            key: validate_config_value(key, value)
            for key, value in (
                ("max_output_tokens", max_output_tokens_fallback),
                ("request_timeout_seconds", request_timeout_seconds_fallback),
                ("auto_compaction_context_tokens", auto_compaction_context_tokens_fallback),
                ("max_context_tokens", max_context_tokens_fallback),
                ("compaction_mode", mode_default),
                ("compaction_pi_keep_recent_tokens", compaction_pi_keep_recent_tokens_fallback),
                ("compaction_pi_max_output_tokens", compaction_pi_max_output_tokens_fallback),
            )
        }
        self._require_max_output_tokens = require_max_output_tokens
        self._request_timeout_applies = request_timeout_applies
        self._min_auto_compaction_context_tokens = validate_config_value(
            "auto_compaction_context_tokens", min_auto_compaction_context_tokens,
        )
        self._compaction_route = compaction_route
        # The default before coldstart is dropped, for _without_coldstart.
        self._mode_default = self._fallbacks["compaction_mode"]
        self._coldstart_unavailable = coldstart_unavailable
        self.coldstart_dropped: Optional[tuple] = None
        snapshot = self._resolve(snapshot)
        initial = snapshot if initial is None else self._resolve(initial)
        if compaction_route is not None:
            snapshot, initial = (
                replace(value, compaction_mode=compaction_route.resolve(value.compaction_mode))
                for value in (snapshot, initial)
            )
        if coldstart_unavailable is not None:
            default = self._fallbacks["compaction_mode"]
            self._fallbacks["compaction_mode"] = self._without_coldstart(default)
            before = snapshot.compaction_mode
            snapshot, initial = (
                replace(value, compaction_mode=self._without_coldstart(value.compaction_mode))
                for value in (snapshot, initial)
            )
            # Notices are for values that someone set: the default sequence,
            # or a value equal to it, loses coldstart silently.
            if snapshot.compaction_mode != before and before != default:
                self.coldstart_dropped = (before, snapshot.compaction_mode)
        self._validate(snapshot)
        self._validate(initial)
        self._lock = RLock()
        self._snapshot = snapshot
        # The resolved launch values remain fixed after live config changes.
        self._initial = initial
        self._on_enable_workspace = on_enable_workspace

    @property
    def coldstart_unavailable(self) -> Optional[str]:
        """Why this session can't run coldstart compaction, or None."""
        return self._coldstart_unavailable

    def _without_coldstart(self, mode: str) -> str:
        """``mode`` without coldstart.

        If nothing is left, this session's default without it: the catalog's
        value, else the built-in one. If that is coldstart alone too, the
        built-in default without it: warmpi twice.
        """
        method = "pi" if self._compaction_route is None else self._compaction_route.default
        for value in (mode, self._mode_default, DEFAULT_COMPACTION_MODE):
            entries = [(method if name == "legacydefault" else name, attempts)
                       for name, attempts in parse_compaction_mode(value)
                       if name != "coldstart"]
            if entries:
                return format_compaction_mode(entries)
        raise AssertionError("DEFAULT_COMPACTION_MODE names a method besides coldstart")

    def _changed_compaction_mode(self, value: str) -> str:
        """Check a /config change of ``compaction_mode`` (not null) for this session."""
        if self._compaction_route is not None:
            value = self._compaction_route.resolve(value)
        if (self._coldstart_unavailable is not None
                and "coldstart" in compaction_methods(value)):
            raise ConfigError(
                "compaction_mode can't include coldstart in this session: "
                f"{self._coldstart_unavailable}. It stays {self.get('compaction_mode')}."
            )
        return value

    def _resolve(
        self, snapshot: InteractionConfigSnapshot,
    ) -> InteractionConfigSnapshot:
        return replace(snapshot, **{
            key: value for key, value in self._fallbacks.items()
            if getattr(snapshot, key) is None
        })

    def _validate(self, snapshot: InteractionConfigSnapshot) -> None:
        if self._require_max_output_tokens and snapshot.max_output_tokens is None:
            raise ConfigError("max_output_tokens is required for this Messages model.")
        if not self._request_timeout_applies and snapshot.request_timeout_seconds is not None:
            raise ConfigError(_NO_REQUEST_TIMEOUT)
        threshold = snapshot.auto_compaction_context_tokens
        minimum = self._min_auto_compaction_context_tokens
        if threshold is not None and minimum is not None and threshold < minimum:
            raise ConfigError(
                f"auto_compaction_context_tokens must be at least {minimum} for this model."
            )

    def _publish(self, candidate: InteractionConfigSnapshot) -> None:
        """Validate and apply under the caller's lock; failed effects don't publish."""
        self._validate(candidate)
        if (candidate.enable_workspace != self._snapshot.enable_workspace
                and self._on_enable_workspace is not None):
            self._on_enable_workspace(candidate.enable_workspace)
        self._snapshot = candidate

    @classmethod
    def from_namespace(
        cls,
        args,
        *,
        on_enable_workspace: Optional[Callable[[bool], None]] = None,
        catalog=None,
        coldstart_unavailable: Optional[str] = None,
    ) -> "InteractionConfig":
        """Seed one final namespace; do not write resolved values back into it.

        ``coldstart_unavailable`` is why this session can't run coldstart
        compaction, if it can't (see the class docstring).
        Compaction settings without a launch value take the model catalog's
        (``ModelBinding.compaction``), which ``null`` also restores.
        """
        binding = binding_from_namespace(args, catalog)
        mode = resolve_compaction_mode(
            binding, getattr(args, "compaction_mode", None),
        )
        spec = binding.spec
        limits = None if spec is None else spec.limits
        max_output_tokens = args.max_output_tokens
        max_output_tokens_fallback = (
            limits.max_output_tokens
            if binding.api == "messages" and limits is not None else None
        )
        if binding.api == "messages":
            max_output_tokens = resolve_messages_max_output_tokens(
                binding,
                max_output_tokens,
            )
        # The launch value resolves now (flag, catalog, default); ``null``
        # later restores the catalog value or default, not the flag.
        request_timeout_seconds = resolve_request_timeout_seconds(
            binding, getattr(args, "request_timeout_seconds", None),
        )
        return cls(
            InteractionConfigSnapshot(
                enable_workspace=args.enable_workspace,
                max_samples=args.max_samples,
                max_output_tokens=max_output_tokens,
                request_timeout_seconds=request_timeout_seconds,
                enable_auto_compaction=args.enable_auto_compaction,
                auto_compaction_context_tokens=getattr(args, "auto_compaction_context_tokens", None),
                max_context_tokens=getattr(args, "max_context_tokens", None),
                compaction_mode=mode,
                compaction_pi_keep_recent_tokens=getattr(
                    args, "compaction_pi_keep_recent_tokens", None,
                ),
                compaction_pi_max_output_tokens=getattr(
                    args, "compaction_pi_max_output_tokens", None,
                ),
                # A CLI option: other frontends' namespaces have none.
                compaction_coldstart_reboot=getattr(args, "compaction_coldstart_reboot", False),
                extra_sample_params=binding.extra_sample_params,
            ),
            on_enable_workspace=on_enable_workspace,
            require_max_output_tokens=binding.api == "messages",
            max_output_tokens_fallback=max_output_tokens_fallback,
            request_timeout_seconds_fallback=resolve_request_timeout_seconds(binding),
            request_timeout_applies=binding.api != "claude-relay",
            auto_compaction_context_tokens_fallback=(
                None if limits is None else limits.auto_compaction_context_tokens
            ),
            max_context_tokens_fallback=(
                None if limits is None else limits.max_context_tokens
            ),
            # Anthropic's trigger minimum binds only its server compaction.
            min_auto_compaction_context_tokens=(
                MESSAGES_MIN_COMPACTION_TRIGGER_TOKENS
                if binding.api == "messages" and compaction_methods(mode)[0] == "provider"
                else None
            ),
            compaction_route=_binding_route(binding, mode),
            coldstart_unavailable=coldstart_unavailable,
            compaction_mode_fallback=binding.compaction.mode,
            compaction_pi_keep_recent_tokens_fallback=binding.compaction.pi_keep_recent_tokens,
            compaction_pi_max_output_tokens_fallback=binding.compaction.pi_max_output_tokens,
        )

    @classmethod
    def from_model(
        cls,
        model,
        snapshot: InteractionConfigSnapshot = InteractionConfigSnapshot(),
        *,
        coldstart_unavailable: Optional[str] = None,
    ) -> "InteractionConfig":
        """Bind Python frontend inputs once, without loading credentials.

        Endpoint budgets/triggers are initial preferences, not later overrides.
        Custom models may supply limit metadata at this boundary only. A
        built-in model's binding owns ``extra_sample_params``; a snapshot may
        repeat that map but not change it. The compaction mode is inferred
        from the adapter: ``provider`` for a Messages endpoint with server
        compaction; otherwise coldstart, then warmpi up to twice, on a Codex
        model with remote compaction too. A snapshot may request
        ``pi`` on Codex, but not a mode the model can't run. An HTTP
        endpoint's ``request_timeout_seconds`` seeds the launch
        timeout unless the snapshot sets one; ``null`` later restores the
        binding's catalog value or default. Claude Relay rejects a timeout.
        The binding's catalog compaction settings are what ``null`` restores;
        the snapshot's values are launch values, its default 20,000-token
        ``compaction_pi_keep_recent_tokens`` included.
        """
        from .chat_completions import ChatCompletionsModel
        from .messages import MessagesModel
        from .responses import _ResponsesModelBase

        if not isinstance(snapshot, InteractionConfigSnapshot):
            raise TypeError("snapshot must be InteractionConfigSnapshot")
        snapshot = replace(
            snapshot,
            compaction_mode=_model_compaction_mode(model, snapshot.compaction_mode),
        )
        if isinstance(model, MessagesModel):
            endpoint = model.endpoint
            compaction = endpoint.server_compaction
            snapshot = replace(
                snapshot,
                max_output_tokens=(
                    endpoint.max_output_tokens if snapshot.max_output_tokens is None
                    else snapshot.max_output_tokens
                ),
                auto_compaction_context_tokens=(
                    compaction.trigger_input_tokens
                    if snapshot.auto_compaction_context_tokens is None and compaction is not None
                    else snapshot.auto_compaction_context_tokens
                ),
            )
            profile = "messages"
        elif isinstance(model, _ResponsesModelBase):
            endpoint = model.endpoint
            profile = model.binding.endpoint.api
        elif isinstance(model, ChatCompletionsModel):
            endpoint = model.endpoint
            profile = "chat-completions"
        else:
            binding = getattr(model, "binding", None)
            catalog = _model_compaction_defaults(model)
            return cls(
                snapshot,
                auto_compaction_context_tokens_fallback=getattr(
                    model, "auto_compaction_context_tokens", None,
                ),
                max_context_tokens_fallback=getattr(model, "max_context_tokens", None),
                request_timeout_applies=not (
                    isinstance(binding, ModelBinding) and binding.api == "claude-relay"
                ),
                compaction_route=_model_route(model),
                coldstart_unavailable=coldstart_unavailable,
                compaction_mode_fallback=catalog.mode,
                compaction_pi_keep_recent_tokens_fallback=catalog.pi_keep_recent_tokens,
                compaction_pi_max_output_tokens_fallback=catalog.pi_max_output_tokens,
            )
        binding = model.binding
        if (snapshot.extra_sample_params
                and snapshot.extra_sample_params != binding.extra_sample_params):
            raise ConfigError(
                "extra_sample_params come from the model binding; bind them on "
                "the model, or pass SampleParams(extra=...) for per-call extras."
            )
        if snapshot.request_timeout_seconds is None:
            snapshot = replace(
                snapshot, request_timeout_seconds=endpoint.request_timeout_seconds,
            )
        return cls.from_namespace(SimpleNamespace(
            model_api=profile, model=endpoint.model,
            model_binding=binding,
            **snapshot.as_dict(),
        ), coldstart_unavailable=coldstart_unavailable)

    def snapshot(self) -> InteractionConfigSnapshot:
        with self._lock:
            return self._snapshot

    def initial_snapshot(self) -> InteractionConfigSnapshot:
        with self._lock:
            return self._initial

    def get(self, key: str) -> ConfigValue:
        key = _require_key(key)
        return getattr(self.snapshot(), key)

    def set(self, key: str, value: object) -> ConfigValue:
        key = _require_key(key)
        if key in _LAUNCH_ONLY_KEYS:
            raise _launch_only_error(key)
        value = validate_config_value(key, value)
        if value is None and key in self._fallbacks:
            value = self._fallbacks[key]
        elif key == "compaction_mode":
            value = self._changed_compaction_mode(value)
        with self._lock:
            current = self._snapshot
            if getattr(current, key) == value:
                return value
            self._publish(replace(current, **{key: value}))
        return value

    def reset(self, key: Optional[str] = None) -> InteractionConfigSnapshot:
        """Restore the resolved launch state (not the catalog fallback)."""
        if key is not None:
            key = _require_key(key)
        with self._lock:
            candidate = self._initial if key is None else replace(
                self._snapshot, **{key: getattr(self._initial, key)},
            )
            self._publish(candidate)
            return self._snapshot

    def values(self, key: Optional[str] = None) -> Dict[str, ConfigValue]:
        snapshot = self.snapshot()
        if key is not None:
            key = _require_key(key)
            return {key: snapshot.as_dict()[key]}
        return snapshot.as_dict()

    def initial_values(self, key: Optional[str] = None) -> Dict[str, ConfigValue]:
        initial = self.initial_snapshot()
        if key is not None:
            key = _require_key(key)
            return {key: initial.as_dict()[key]}
        return initial.as_dict()

    def render(self, key: Optional[str] = None, *, json_output: bool) -> str:
        values = self.values(key)
        initial = self.initial_values(key)
        if json_output:
            # ``__init__`` carries the launch-seeded defaults (a dict) alongside
            # the current values; `/config-reset` (TODO) will restore them.
            return json.dumps(
                {"__init__": initial, **values},
                ensure_ascii=False,
                indent=2,
            )
        lines = []
        for name, value in values.items():
            seeded = initial[name]
            if seeded != value:
                lines.append(f"# init: {name} = {seeded!r}")
            lines.append(f"{name} = {value!r}")
        return "\n".join(lines)


__all__ = [
    "CONFIG_KEYS",
    "CompactionRoute",
    "ConfigError",
    "ConfigValue",
    "InteractionConfig",
    "InteractionConfigSnapshot",
    "RENAMED_CONFIG_KEYS",
    "parse_config_literal",
    "resolve_compaction_mode",
    "resolve_request_timeout_seconds",
    "validate_config_value",
]
