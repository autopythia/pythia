"""Explicit file I/O for the user catalog. Importing model_catalog stays pure."""

from __future__ import annotations

import configparser
from dataclasses import replace
from pathlib import Path
import re
import warnings

from ._config_file import read_config_bytes
from .compaction import compaction_methods
from .model_catalog import BUILTIN_MODEL_CATALOG
from .model_catalog import COMPACTION_DEFAULT_FIELDS, CompactionDefaults
from .model_catalog import SECTION_PROVIDER_ERROR, compaction_route_error
from .model_catalog import ModelCatalog, ModelLimits, ModelSpec, ModelTimeouts
from .model_catalog import EndpointSpec
from .model_catalog import MODEL_TIMEOUT_FIELDS
from .model_catalog import PROMPT_CACHING_VALUES
from .model_catalog import RATE_LIMIT_RETRIES_UNBOUNDED
from .model_catalog import ResponsesDefaults
from .model_catalog import SAMPLE_PARAM_DEFAULT_FIELDS, SampleParamDefaults
from .model_catalog import TOKEN_TRACE_LEVELS
from .model_catalog import TOKEN_TRACE_FIELDS
from .model_catalog import sample_params_route_error
from .model_catalog import _normalize_profile, parse_json_value


MAX_CATALOG_BYTES = 1_048_576
MAX_CATALOG_MODELS = 1024
# Version 3 replaced messages.output_effort with request_params.output_config.
# Version 4 renamed request_params to extra_sample_params.
# Optional timeouts.* fields were added to version 4 without a version change.
# limits.auto_compact_context_tokens became limits.auto_compaction_context_tokens
# within version 4; the old name still loads, with a FutureWarning.
# The [compaction] section and compaction.* fields were added to version 4
# without a version change.
# Optional sample_params.* fields were added to version 4 without a version
# change.
LATEST_MODEL_CATALOG_VERSION = 4
_EXTRA_SAMPLE_PARAMS_PREFIX = "extra_sample_params."
_SAMPLE_PARAMS_PREFIX = "sample_params."
_LIMIT_FIELDS = frozenset(("auto_compaction_context_tokens", "max_context_tokens", "max_output_tokens"))
# Deprecated catalog keys and their current names.
_RENAMED_LIMIT_KEYS = {"limits.auto_compact_context_tokens": "limits.auto_compaction_context_tokens"}
_RESPONSES_FIELDS = frozenset(("reasoning_effort", "reasoning_summary", "text_verbosity"))
_TIMEOUT_FIELDS = frozenset(MODEL_TIMEOUT_FIELDS)
_INTEGER = re.compile(r"^[+-]?[0-9]+$")
# Plain decimal seconds: no exponent, infinity, or NaN spellings.
_DECIMAL = re.compile(r"^[+-]?[0-9]+(?:\.[0-9]+)?$")
# Roles whose default models the optional [auto] section may name.
AUTO_ROLES = ("main", "watcher", "worker")
_COMPACTION_PREFIX = "compaction."


class _FieldError(ValueError):
    """A model entry's field error whose message is safe to show.

    Other entry errors are reported without detail, since values from extra
    sample params or credentials must not appear in errors; compaction
    settings are neither.
    """


def _alternatives(names):
    """``a``, ``a or b``, or ``a, b, or c``."""
    if len(names) <= 2:
        return " or ".join(names)
    return f"{', '.join(names[:-1])}, or {names[-1]}"


def default_model_catalog_path():
    return Path.home() / ".pythia" / "model-catalog.ini"


def _text(value):
    if value == "null":
        return None
    if value.startswith('"'):
        value = parse_json_value(value)
        if not isinstance(value, str):
            raise ValueError("Expected text.")
    return value


def _compaction_value(name, raw):
    """One [compaction] or compaction.* value: None for null, else the setting's.

    Errors say what was expected; the caller says where.
    """
    if raw == "null":
        return None
    if name == "mode":
        return CompactionDefaults(mode=_text(raw)).mode
    kind, minimum = ("nonnegative", 0) if name == "pi_keep_recent_tokens" else ("positive", 1)
    if _INTEGER.fullmatch(raw) is None or int(raw) < minimum:
        raise ValueError(f"expected a {kind} integer or null")
    return int(raw)


def _sample_param_value(name, raw):
    """One sample_params.* value: None for null, else the field's validated number.

    Errors say what was expected; the caller says where.
    """
    if raw == "null":
        return None
    if _DECIMAL.fullmatch(raw) is None:
        raise ValueError("expected a plain decimal number, such as 0.7, or null")
    return getattr(SampleParamDefaults(**{name: float(raw)}), name)


def _typed_extras_error(key, params):
    """A _FieldError when extensions name typed sample params, or None.

    They are reserved like every typed field; this names the catalog field
    that sets them instead. Only names are shown, never values.
    """
    if not isinstance(params, dict):
        return None
    named = {name.lower() for name in params if isinstance(name, str)}
    typed = [name for name in SAMPLE_PARAM_DEFAULT_FIELDS if name in named]
    if not typed:
        return None
    fields = " and ".join(_SAMPLE_PARAMS_PREFIX + name for name in typed)
    what = (f"{typed[0]} is a typed sample param" if len(typed) == 1
            else f"{' and '.join(typed)} are typed sample params")
    return _FieldError(f"{key}: {what}; set {fields} instead")


def _entry(section, values, base):
    name = section.removeprefix("model.")
    values = dict(values)
    override = values.pop("override", "false").lower()
    if override not in configparser.ConfigParser.BOOLEAN_STATES:
        raise ValueError("override must be a boolean.")
    override = configparser.ConfigParser.BOOLEAN_STATES[override]
    api_key = "endpoint.api"
    has_api = api_key in values
    api = _text(values.pop(api_key)) if has_api else None
    if has_api and api is None:
        raise ValueError("endpoint.api cannot be null.")
    if api is not None:
        api = _normalize_profile(api)
    if override:
        candidates = tuple(spec for spec in base.matches(name, canonical_only=True)
                           if api is None or spec.endpoint.api == api)
        if len(candidates) != 1:
            raise ValueError("Override requires one existing canonical model; specify endpoint.api if ambiguous.")
        original = candidates[0]
        api = original.endpoint.api
    else:
        if api is None:
            raise ValueError("New entries require endpoint.api.")
        if base.get_model_spec(api, name) is not None:
            raise ValueError("Existing entries require override = true.")
        original = None

    fields = {}
    endpoint = {} if original is None else dict(vars(original.endpoint))
    if api is not None:
        endpoint["api"] = api
    limits = {} if original is None else dict(vars(original.limits))
    timeouts = {} if original is None else dict(vars(original.timeouts))
    compaction = {} if original is None else dict(vars(original.compaction))
    sample_params = {} if original is None else dict(vars(original.sample_params))
    responses = {} if original is None or original.responses is None else dict(vars(original.responses))
    params = {} if original is None else dict(original.extra_sample_params)
    if "extra_sample_params" in values and any(
        key.startswith(_EXTRA_SAMPLE_PARAMS_PREFIX) for key in values
    ):
        raise ValueError("Cannot combine whole-map and per-key extra sample params.")
    for key, value in values.items():
        # parse_model_catalog has rejected a section that sets both names.
        key = _RENAMED_LIMIT_KEYS.get(key, key)
        if key in {"endpoint.model", "endpoint.url", "endpoint.auth"}:
            endpoint[key[9:]] = _text(value)
            if key == "endpoint.auth":
                endpoint["auth_file"] = None
        elif key == "endpoint.prompt_caching":
            # null restores the default, 5m; none sends no cache_control.
            caching = _text(value)
            if caching is not None and caching not in PROMPT_CACHING_VALUES:
                raise _FieldError(f"{key} must be 5m, 1h, none, or null")
            endpoint["prompt_caching"] = caching
        elif key == "endpoint.rate_limit_retries":
            # null restores the default; unbounded retries until cancelled.
            retries = _text(value)
            if retries is not None and retries != RATE_LIMIT_RETRIES_UNBOUNDED:
                if _INTEGER.fullmatch(retries) is None or int(retries) < 0:
                    raise _FieldError(f"{key} must be a nonnegative integer, "
                                      f"{RATE_LIMIT_RETRIES_UNBOUNDED}, or null")
                retries = int(retries)
            endpoint["rate_limit_retries"] = retries
        elif key == "endpoint.token_trace":
            # null restores the default, none.
            trace = _text(value)
            if trace is not None and trace not in TOKEN_TRACE_LEVELS:
                raise _FieldError(f"{key} must be {', '.join(TOKEN_TRACE_LEVELS)}, or null")
            endpoint["token_trace"] = trace
        elif key == "aliases":
            fields[key] = parse_json_value(value)
            if not isinstance(fields[key], list):
                raise ValueError("aliases must be a JSON array.")
        elif key == "source":
            fields[key] = _text(value)
        elif key.startswith("limits.") and key[7:] in _LIMIT_FIELDS:
            if value != "null" and _INTEGER.fullmatch(value) is None:
                raise ValueError("Limits require integers or null.")
            limits[key[7:]] = None if value == "null" else int(value)
        elif key.startswith("timeouts.") and key[9:] in _TIMEOUT_FIELDS:
            if value != "null" and _DECIMAL.fullmatch(value) is None:
                raise ValueError("Timeouts require decimal seconds or null.")
            timeouts[key[9:]] = None if value == "null" else float(value)
        elif key.startswith("responses.") and key[10:] in _RESPONSES_FIELDS:
            responses[key[10:]] = _text(value)
        elif key.startswith(_COMPACTION_PREFIX):
            setting = key.removeprefix(_COMPACTION_PREFIX)
            if setting not in COMPACTION_DEFAULT_FIELDS:
                raise _FieldError(f"{key} is not a compaction field; use " + _alternatives(
                    [_COMPACTION_PREFIX + field for field in COMPACTION_DEFAULT_FIELDS]))
            try:
                # In an entry, null removes its own value: [compaction]'s applies.
                compaction[setting] = _compaction_value(setting, value)
            except ValueError as exc:
                raise _FieldError(f"{key}: {exc}") from None
        elif key.startswith(_SAMPLE_PARAMS_PREFIX):
            setting = key.removeprefix(_SAMPLE_PARAMS_PREFIX)
            if setting not in SAMPLE_PARAM_DEFAULT_FIELDS:
                raise _FieldError(f"{key} is not a sample param field; use " + _alternatives(
                    [_SAMPLE_PARAMS_PREFIX + field for field in SAMPLE_PARAM_DEFAULT_FIELDS]))
            try:
                # In an override, null removes the entry's value: no field is sent.
                sample_params[setting] = _sample_param_value(setting, value)
            except (TypeError, ValueError) as exc:
                raise _FieldError(f"{key}: {exc}") from None
        elif key == "extra_sample_params":
            params = parse_json_value(value)
            error = _typed_extras_error(key, params)
            if error is not None:
                raise error
        elif key.startswith(_EXTRA_SAMPLE_PARAMS_PREFIX):
            extension = key.removeprefix(_EXTRA_SAMPLE_PARAMS_PREFIX)
            error = _typed_extras_error(key, {extension: None})
            if error is not None:
                raise error
            params[extension] = parse_json_value(value)
        else:
            raise ValueError("Unknown model field.")

    if api == "claude-relay":
        endpoint.setdefault("url", None)
        endpoint.setdefault("auth", "runtime")
    if original is None and not {"api", "url", "model", "auth"} <= endpoint.keys():
        raise ValueError("New entries require endpoint.api, url, model, and auth.")
    if (original is not None and "endpoint.url" in values
            and endpoint["url"] != original.endpoint.url and original.endpoint.auth != "none"
            and "endpoint.auth" not in values):
        raise ValueError("Changing a credentialed endpoint URL requires endpoint.auth.")
    if endpoint.get("prompt_caching") is not None and endpoint.get("api") != "messages":
        raise _FieldError("endpoint.prompt_caching requires endpoint.api messages")
    if endpoint.get("rate_limit_retries") is not None and endpoint.get("api") == "claude-relay":
        raise _FieldError("endpoint.rate_limit_retries does not apply to endpoint.api claude-relay")
    if endpoint.get("token_trace") is not None and endpoint.get("api") != "chat-completions":
        raise _FieldError("endpoint.token_trace requires endpoint.api chat-completions")
    if endpoint.get("api") == "chat-completions":
        # The levels own these vLLM fields; name the field that sets them.
        traced = sorted(name for name in params if isinstance(name, str)
                        and name.lower() in TOKEN_TRACE_FIELDS)
        if traced:
            raise _FieldError(f"{_EXTRA_SAMPLE_PARAMS_PREFIX}{traced[0]}: {traced[0]} belongs "
                              "to endpoint.token_trace; set that instead")
    endpoint = EndpointSpec(**endpoint)
    if endpoint.auth == "codex-login" and not endpoint.is_official_codex:
        raise ValueError("Catalog Codex login requires the official Codex endpoint.")
    compaction = CompactionDefaults(**compaction)
    error = compaction_route_error(endpoint, compaction.mode)
    if error is not None:
        raise _FieldError(f"compaction.mode: {error}")
    sample_params = SampleParamDefaults(**sample_params)
    error = sample_params_route_error(endpoint.api, sample_params)
    if error is not None:
        raise _FieldError(f"sample_params: {error}")
    fields.update(endpoint=endpoint, limits=ModelLimits(**limits),
                  timeouts=ModelTimeouts(**timeouts), extra_sample_params=params,
                  compaction=compaction, sample_params=sample_params)
    if responses:
        fields["responses"] = ResponsesDefaults(**responses)
    spec = (ModelSpec(name=name, **fields) if original is None
            else replace(original, **fields))
    return spec


def parse_model_catalog(text, *, base=BUILTIN_MODEL_CATALOG, source="user"):
    """Parse transactionally; omitted version metadata uses the latest schema."""
    if not isinstance(base, ModelCatalog):
        raise TypeError("base must be ModelCatalog")
    if not isinstance(text, str):
        raise TypeError("catalog text must be a string")
    if len(text.encode("utf-8")) > MAX_CATALOG_BYTES:
        raise ValueError("Model catalog exceeds the size limit.")
    parser = configparser.ConfigParser(
        interpolation=None, strict=True, delimiters=("=",), allow_no_value=False,
        inline_comment_prefixes=None, empty_lines_in_values=False,
    )
    parser.optionxform = str
    assumed_version = False
    try:
        parser.read_string(text)
        if parser.defaults():
            raise ValueError()
        if parser.has_section("catalog"):
            catalog_fields = set(parser["catalog"])
            if catalog_fields - {"version"}:
                raise ValueError()
            if "version" in catalog_fields:
                version = parser.getint("catalog", "version")
            else:
                version = LATEST_MODEL_CATALOG_VERSION
                assumed_version = True
        else:
            version = LATEST_MODEL_CATALOG_VERSION
            assumed_version = True
        sections = [section for section in parser.sections() if section != "catalog"]
        sections = [section for section in sections if section not in ("auto", "compaction")]
        if len(sections) > MAX_CATALOG_MODELS or any(not section.startswith("model.") for section in sections):
            raise ValueError()
    except (configparser.Error, ValueError):
        raise ValueError(f"Invalid model catalog header/INI syntax: {source}") from None
    if version != LATEST_MODEL_CATALOG_VERSION:
        raise ValueError(
            f"Unsupported model catalog version {version} in {source}; "
            f"expected version {LATEST_MODEL_CATALOG_VERSION}"
        )
    specs = {(spec.endpoint.api, spec.name): spec for spec in base.specs}
    origins = dict(base.origins)
    seen = set()
    renamed = {}  # deprecated key -> sections that use it
    for section in sections:
        # Name the replacement rather than failing as a generic unknown field.
        if any(key.startswith("messages.") for key in parser[section]):
            raise ValueError(
                f"Invalid model catalog entry {section!r} in {source}: messages.* fields "
                'were removed in catalog version 3; use extra_sample_params.output_config = {"effort": ...}'
            )
        if any(key == "request_params" or key.startswith("request_params.")
               for key in parser[section]):
            raise ValueError(
                f"Invalid model catalog entry {section!r} in {source}: request_params "
                "was renamed to extra_sample_params in catalog version 4"
            )
        for old, new in _RENAMED_LIMIT_KEYS.items():
            if old in parser[section]:
                if new in parser[section]:
                    raise ValueError(
                        f"Invalid model catalog entry {section!r} in {source}: {old} is the "
                        f"deprecated name of {new}; set only the new name"
                    )
                renamed.setdefault(old, []).append(section)
        try:
            spec = _entry(section, parser[section], base)
            identity = (spec.endpoint.api, spec.name)
            if identity in seen:
                raise ValueError("Duplicate normalized model identity.")
            seen.add(identity)
            specs[identity] = spec
            origins[identity] = str(source)
        except _FieldError as exc:
            raise ValueError(f"Invalid model catalog entry {section!r} in {source}: {exc}") from None
        except (ValueError, TypeError, KeyError, RecursionError):
            # No values from extra sample params/credentials should appear in errors.
            raise ValueError(f"Invalid model catalog entry {section!r} in {source}") from None
    try:
        catalog = ModelCatalog(tuple(specs.values()), origins, compaction=base.compaction)
    except ValueError:
        raise ValueError(f"Model catalog selector/alias collision in {source}") from None
    if parser.has_section("auto"):
        catalog = replace(catalog, auto_models={
            **catalog.auto_models, **_auto_models(parser["auto"], catalog, source),
        })
    if parser.has_section("compaction"):
        catalog = replace(catalog, compaction=_compaction_section(
            parser["compaction"], catalog.compaction, source))
    if assumed_version:
        warnings.warn(
            f"Model catalog {source!r} does not specify a version; assuming latest "
            f"supported version {version}.",
            UserWarning,
            stacklevel=2,
        )
    for old, used in renamed.items():
        # FutureWarning, not DeprecationWarning: Python shows it by default.
        warnings.warn(
            f"Model catalog {source!r} uses the deprecated {old} in {_listing(used)}; "
            f"rename it to {_RENAMED_LIMIT_KEYS[old]}.",
            FutureWarning,
            stacklevel=2,
        )
    return catalog


def _listing(sections):
    """``[a]``, ``[a] and [b]``, or ``[a], [b] and [c]``."""
    names = [f"[{section}]" for section in sections]
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _auto_models(section, catalog, source):
    """[auto] <role>.model selectors: catalog names only, so typos fail at load."""
    models = {}
    for key, raw in section.items():
        role, dot, field = key.partition(".")
        if role not in AUTO_ROLES or not dot or field != "model":
            raise ValueError(f"Invalid model catalog [auto] key {key!r} in {source}; "
                             "expected main.model, watcher.model, or worker.model")
        try:
            name = _text(raw)
        except ValueError:
            raise ValueError(f"Invalid model catalog [auto] {key} in {source}") from None
        if name is None:
            continue
        name = name.strip()
        matches = catalog.matches(name) if name else ()
        if not matches:
            raise ValueError(f"Invalid model catalog [auto] {key} in {source}: "
                             f"{name!r} is not a catalog model")
        if len(matches) > 1:
            raise ValueError(f"Invalid model catalog [auto] {key} in {source}: "
                             f"{name!r} is in the catalog for several APIs")
        models[role] = name
    return models


def _compaction_section(section, base, source):
    """[compaction]: settings for every model, over ``base``'s; null unsets one."""
    values = dict(vars(base))
    for key, raw in section.items():
        if key not in COMPACTION_DEFAULT_FIELDS:
            raise ValueError(f"Invalid model catalog [compaction] key {key!r} in {source}; "
                             f"expected {_alternatives(COMPACTION_DEFAULT_FIELDS)}")
        try:
            values[key] = _compaction_value(key, raw)
            if (key == "mode" and values[key] is not None
                    and "provider" in compaction_methods(values[key])):
                raise ValueError(SECTION_PROVIDER_ERROR)
        except ValueError as exc:
            raise ValueError(f"Invalid model catalog [compaction] {key} in {source}: {exc}") from None
    return CompactionDefaults(**values)


def load_model_catalog(path=None, *, enabled=True, base=BUILTIN_MODEL_CATALOG):
    """Only this opt-in boundary reads the default home file; a missing default is OK."""
    if not enabled:
        if path is not None:
            raise ValueError("Cannot select and disable the user model catalog.")
        return base
    explicit = path is not None
    path = (default_model_catalog_path() if path is None else Path(path)).expanduser().absolute()
    try:
        data = read_config_bytes(path, MAX_CATALOG_BYTES)
        text = data.decode("utf-8")
    except FileNotFoundError:
        if not explicit:
            return base
        raise ValueError(f"Model catalog file not found: {path}") from None
    except (OSError, UnicodeError):
        raise ValueError(f"Could not read model catalog: {path}") from None
    return parse_model_catalog(text, base=base, source=str(path))


__all__ = [
    "AUTO_ROLES",
    "LATEST_MODEL_CATALOG_VERSION",
    "default_model_catalog_path",
    "load_model_catalog",
    "parse_model_catalog",
]
