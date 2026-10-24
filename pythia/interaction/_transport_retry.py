"""Small shared retry-delay policy for synchronous interaction transports."""

from __future__ import annotations

from datetime import timezone
from email.utils import parsedate_to_datetime
import json
import math
import re
import time
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Tuple


DEFAULT_MAX_TRANSIENT_RETRIES = 2
INITIAL_RETRY_DELAY_SECONDS = 0.25
MAX_BACKOFF_DELAY_SECONDS = 2.0
MAX_RETRY_AFTER_SECONDS = 60.0
# A Retry-After over a year counts as malformed, as a negative delta does: no
# limit lasts that long, and a delay that size can't be waited out or shown.
MAX_PLAUSIBLE_RETRY_AFTER_SECONDS = 366 * 86400.0
# The error codes or types of a 429 that names an exhausted quota or plan
# (OpenAI's insufficient_quota; Codex's usage_limit_reached and
# usage_not_included, which last until the plan resets). The turn loop retries
# it like any other 429, since it may be lifted meanwhile, as by a top-up, a
# plan reset or another account; its failure says so (quota_error_code).
QUOTA_ERROR_CODES = frozenset(("insufficient_quota", "usage_limit_reached", "usage_not_included"))
_ERROR_CODE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")


def _header_value(headers: Any, name: str) -> Optional[str]:
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except Exception:
        return None
    if value is None:
        try:
            value = next(
                candidate
                for key, candidate in headers.items()
                if isinstance(key, str) and key.lower() == name.lower()
            )
        except (AttributeError, StopIteration, TypeError):
            return None
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value or "\r" in value or "\n" in value:
        return None
    return value


def retry_after_seconds(
    headers: Any,
    *,
    now_seconds: Optional[float] = None,
    cap: Optional[float] = MAX_RETRY_AFTER_SECONDS,
) -> Optional[float]:
    """Decode a Retry-After delta or HTTP date; malformed means absent.

    The result is at most ``cap`` (an adapter's own wait); ``cap=None`` keeps
    the provider's whole delay, as a rate limit's error reports it. A delay
    over MAX_PLAUSIBLE_RETRY_AFTER_SECONDS (a year) is malformed too.
    """
    value = _header_value(headers, "Retry-After")
    if value is None:
        return None
    parsed_date = False
    try:
        delay = float(value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            now = time.time() if now_seconds is None else float(now_seconds)
            delay = parsed.timestamp() - now
            parsed_date = True
        except (TypeError, ValueError, OverflowError, OSError):
            return None
    if not math.isfinite(delay) or delay > MAX_PLAUSIBLE_RETRY_AFTER_SECONDS:
        return None
    if delay < 0:
        return 0.0 if parsed_date else None
    return delay if cap is None else min(delay, cap)


def body_error_codes(detail: str) -> Tuple[str, ...]:
    """A JSON error body's ``error.code`` and ``error.type``, where each is an
    identifier (never free text, which may hold a secret)."""
    try:
        value = json.loads(detail)
    except (TypeError, ValueError, RecursionError):
        return ()
    error = value.get("error") if isinstance(value, Mapping) else None
    if not isinstance(error, Mapping):
        return ()
    return tuple(item for item in (error.get("code"), error.get("type"))
                 if isinstance(item, str) and _ERROR_CODE.fullmatch(item))


def quota_error_code(detail: str) -> Optional[str]:
    """The exhausted quota or plan that an error body names in its
    ``error.code`` or ``error.type`` (QUOTA_ERROR_CODES), or None."""
    return next((code for code in body_error_codes(detail) if code in QUOTA_ERROR_CODES), None)


def retry_delay_seconds(
    retry_count: int,
    headers: Any = None,
    *,
    now_seconds: Optional[float] = None,
) -> float:
    """Return a provider-directed delay or bounded exponential fallback.

    ``retry_count`` is one-based: 1 is the first replay after the initial
    request. The policy is deliberately deterministic; the small retry budget
    avoids a long hidden wait and leaves cancellation integration for the CLI's
    separate active-turn interruption work.
    """
    if (
        isinstance(retry_count, bool)
        or not isinstance(retry_count, int)
        or retry_count <= 0
    ):
        raise ValueError("retry_count must be a positive integer")
    provider_delay = retry_after_seconds(
        headers,
        now_seconds=now_seconds,
    )
    if provider_delay is not None:
        return provider_delay
    exponent = min(retry_count - 1, 32)
    return min(
        INITIAL_RETRY_DELAY_SECONDS * (2 ** exponent),
        MAX_BACKOFF_DELAY_SECONDS,
    )


__all__ = [
    "DEFAULT_MAX_TRANSIENT_RETRIES",
    "INITIAL_RETRY_DELAY_SECONDS",
    "MAX_BACKOFF_DELAY_SECONDS",
    "MAX_PLAUSIBLE_RETRY_AFTER_SECONDS",
    "MAX_RETRY_AFTER_SECONDS",
    "QUOTA_ERROR_CODES",
    "body_error_codes",
    "quota_error_code",
    "retry_after_seconds",
    "retry_delay_seconds",
]
