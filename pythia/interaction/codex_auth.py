from __future__ import annotations

import json
import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Union


CodexAuthPath = Union[str, os.PathLike[str]]


class CodexAuthError(ValueError):
    pass


class CodexAuthUnavailable(CodexAuthError):
    """Credential loading failed, as distinct from invalid path/config options."""


@dataclass(frozen=True)
class CodexAuth:
    access_token: str = field(repr=False)
    account_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.access_token, str):
            raise TypeError("access_token must be a string")
        access_token = self.access_token.strip()
        if not access_token:
            raise CodexAuthError("access_token must not be empty")
        if any(character.isspace() for character in access_token):
            raise CodexAuthError("access_token must not contain whitespace")
        object.__setattr__(self, "access_token", access_token)

        if self.account_id is not None:
            if not isinstance(self.account_id, str):
                raise TypeError("account_id must be a string or None")
            account_id = self.account_id.strip()
            if not account_id:
                raise CodexAuthError("account_id must not be empty")
            if "\r" in account_id or "\n" in account_id:
                raise CodexAuthError("account_id must not contain newlines")
            object.__setattr__(self, "account_id", account_id)


@dataclass(frozen=True)
class CodexCredentials:
    """A complete auth-file snapshot; all token material is repr-redacted."""

    auth: CodexAuth
    auth_file: Path
    refresh_token: Optional[str] = field(default=None, repr=False)
    id_token: Optional[str] = field(default=None, repr=False)
    auth_mode: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.auth, CodexAuth):
            raise TypeError("auth must be CodexAuth")
        if not isinstance(self.auth_file, Path):
            raise TypeError("auth_file must be Path")
        for field_name in ("refresh_token", "id_token", "auth_mode"):
            value = getattr(self, field_name)
            if value is not None:
                _optional_nonempty_string(value, field_name)


def _normalize_path(value: CodexAuthPath, field_name: str) -> Path:
    if isinstance(value, os.PathLike):
        path = Path(value)
    elif isinstance(value, str):
        if not value.strip():
            raise CodexAuthError(f"{field_name} must not be empty")
        path = Path(value)
    else:
        raise TypeError(f"{field_name} must be a path-like value")
    return path.expanduser()


def _resolve_auth_file(
    *,
    codex_home: Optional[CodexAuthPath],
    auth_file: Optional[CodexAuthPath],
) -> Path:
    if auth_file is not None:
        return _normalize_path(auth_file, "auth_file")

    if codex_home is not None:
        home = _normalize_path(codex_home, "codex_home")
    else:
        configured_home = os.environ.get("CODEX_HOME")
        if configured_home is not None and configured_home.strip():
            home = Path(configured_home).expanduser()
        else:
            try:
                home = Path.home() / ".codex"
            except RuntimeError as exc:
                raise CodexAuthError(
                    "could not determine the Codex home directory"
                ) from exc
    return home / "auth.json"


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CodexAuthError(f"{field_name} must be a JSON object")
    return value


def _optional_nonempty_string(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CodexAuthError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise CodexAuthError(f"{field_name} must not be empty")
    return normalized


def _optional_credential_string(value: Any) -> Optional[str]:
    """Ignore malformed optional refresh metadata without hiding valid access auth."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or any(character.isspace() for character in normalized):
        return None
    return normalized


def load_codex_auth(
    *,
    codex_home: Optional[CodexAuthPath] = None,
    auth_file: Optional[CodexAuthPath] = None,
) -> CodexAuth:
    """Load an existing Codex CLI login without modifying credential state."""
    return load_codex_credentials(
        codex_home=codex_home,
        auth_file=auth_file,
    ).auth


def load_codex_credentials(
    *,
    codex_home: Optional[CodexAuthPath] = None,
    auth_file: Optional[CodexAuthPath] = None,
) -> CodexCredentials:
    """Load access and refresh material from an existing Codex auth file."""
    resolved_auth_file = _resolve_auth_file(
        codex_home=codex_home,
        auth_file=auth_file,
    )
    try:
        raw = resolved_auth_file.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CodexAuthUnavailable(
            f"Codex auth file not found: {resolved_auth_file}; "
            "run `codex login` to create or refresh it"
        ) from exc
    except OSError as exc:
        raise CodexAuthUnavailable(
            f"could not read Codex auth file {resolved_auth_file}: {exc}"
        ) from exc
    except UnicodeError:
        raise CodexAuthUnavailable("Codex credentials are not valid UTF-8") from None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CodexAuthUnavailable(
            f"could not parse Codex auth file {resolved_auth_file}: {exc}"
        ) from exc

    try:
        return _credentials_from_payload(payload, resolved_auth_file)
    except CodexAuthError as exc:
        raise CodexAuthUnavailable(str(exc)) from None


def _auth_from_payload(payload: Any, resolved_auth_file: Path) -> CodexAuth:
    return _credentials_from_payload(payload, resolved_auth_file).auth


def _credentials_from_payload(
    payload: Any,
    resolved_auth_file: Path,
) -> CodexCredentials:
    root = _require_mapping(payload, str(resolved_auth_file))
    tokens = _require_mapping(
        root.get("tokens"),
        f"{resolved_auth_file}:tokens",
    )
    access_token = _optional_nonempty_string(
        tokens.get("access_token"),
        f"{resolved_auth_file}:tokens.access_token",
    )
    if access_token is None:
        raise CodexAuthError(
            f"{resolved_auth_file} does not contain tokens.access_token; "
            "run `codex login` to create or refresh it"
        )
    account_id = _optional_nonempty_string(
        tokens.get("account_id"),
        f"{resolved_auth_file}:tokens.account_id",
    )
    return CodexCredentials(
        auth=CodexAuth(
            access_token=access_token,
            account_id=account_id,
        ),
        auth_file=resolved_auth_file,
        refresh_token=_optional_credential_string(tokens.get("refresh_token")),
        id_token=_optional_credential_string(tokens.get("id_token")),
        auth_mode=_optional_credential_string(root.get("auth_mode")),
    )


__all__ = [
    "CodexAuth",
    "CodexCredentials",
    "CodexAuthError",
    "CodexAuthUnavailable",
    "load_codex_auth",
    "load_codex_credentials",
]
