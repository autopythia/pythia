"""Small shared retry-delay policy for synchronous interaction transports."""

from __future__ import annotations

from datetime import timezone
from email.utils import parsedate_to_datetime
import math
import time
from typing import Any
from typing import Optional


DEFAULT_MAX_TRANSIENT_RETRIES = 2
INITIAL_RETRY_DELAY_SECONDS = 0.25
MAX_BACKOFF_DELAY_SECONDS = 2.0
MAX_RETRY_AFTER_SECONDS = 60.0


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
) -> Optional[float]:
    """Decode a bounded Retry-After delta or HTTP date; malformed means absent."""
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
    if not math.isfinite(delay):
        return None
    if delay < 0:
        return 0.0 if parsed_date else None
    return min(delay, MAX_RETRY_AFTER_SECONDS)


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
    "MAX_RETRY_AFTER_SECONDS",
    "retry_after_seconds",
    "retry_delay_seconds",
]
