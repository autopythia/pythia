"""A historical ChatGPT/Codex quota snapshot, not a context-window estimate."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Optional
from urllib.request import Request

from ._account_http import AccountServiceError, request_json


def _number(value):
    try:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        return False


def _window(label, value):
    if value is None:
        return f"{label}: unavailable"
    if not isinstance(value, dict):
        raise AccountServiceError("Quota response contained an invalid window.")
    used = value.get("used_percent")
    if not _number(used) or not 0 <= used <= 100:
        raise AccountServiceError("Quota response contained an invalid percentage.")
    seconds = value.get("limit_window_seconds")
    if seconds is not None and (not _number(seconds) or seconds <= 0):
        raise AccountServiceError("Quota response contained an invalid window duration.")
    reset = value.get("reset_at")
    timestamp = "unavailable"
    if reset is not None:
        if not _number(reset) or reset < 0:
            raise AccountServiceError("Quota response contained an invalid reset time.")
        try:
            timestamp = datetime.fromtimestamp(reset, timezone.utc).isoformat()
        except (ValueError, OverflowError, OSError):
            raise AccountServiceError("Quota reset time was out of range.") from None
    return f"{label}: used={used:g}% | window_seconds={seconds if seconds is not None else 'unavailable'} | resets_at={timestamp}"


def _normalize_plan_type(value: object) -> Optional[str]:
    """Preserve provider identifiers without accepting arbitrary display text."""
    if not isinstance(value, str):
        return None
    normalized = value.strip(" ")
    if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", normalized):
        return normalized
    return None


def format_quota(payload, *, queried_at=None):
    if not isinstance(payload, dict):
        raise AccountServiceError("Quota response was not an object.")
    lines = [f"Quota snapshot at {queried_at or datetime.now(timezone.utc).isoformat()}"]
    plan = _normalize_plan_type(payload.get("plan_type"))
    lines.append(f"plan: {plan or 'unavailable'}")

    def windows(prefix, limits):
        if limits is None:
            limits = {}
        if not isinstance(limits, dict):
            raise AccountServiceError("Quota response contained invalid limits.")
        for name in ("primary", "secondary"):
            lines.append(_window(prefix + name, limits.get(name + "_window")))

    windows("", payload.get("rate_limit"))
    credits = payload.get("credits")
    if isinstance(credits, dict):
        for key in ("has_credits", "unlimited"):
            value = credits.get(key)
            lines.append(f"credits.{key}: {value if isinstance(value, bool) else 'unavailable'}")
        try:
            raw_balance = str(credits.get("balance"))
            if len(raw_balance) > 64:
                raise InvalidOperation
            balance = Decimal(raw_balance)
            if not balance.is_finite() or balance.copy_abs() > 10**15:
                raise InvalidOperation
            lines.append(f"credits.balance: {balance.normalize()}")
        except (InvalidOperation, ValueError):
            lines.append("credits.balance: unavailable")
    else:
        lines.append("credits: unavailable")
    additional = payload.get("additional_rate_limits")
    if additional is None:
        additional = []
    if not isinstance(additional, list):
        raise AccountServiceError("Quota response contained invalid additional limits.")
    for index, limit in enumerate(additional[:64]):
        if not isinstance(limit, dict):
            raise AccountServiceError("Quota response contained an invalid additional limit.")
        label = limit.get("metered_feature")
        if not isinstance(label, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", label):
            label = f"additional-{index + 1}"
        windows(label + ".", limit.get("rate_limit"))
    if len(additional) > 64:
        lines.append("Additional limits truncated to 64 entries.")
    return "\n".join(lines)


def query_quota(auth, *, timeout_seconds=60.0, opener=None):
    headers = {"Authorization": f"Bearer {auth.access_token}", "Accept": "application/json"}
    if auth.account_id is not None:
        headers["ChatGPT-Account-ID"] = auth.account_id
    payload = request_json(Request("https://chatgpt.com/backend-api/wham/usage",
                                   headers=headers, method="GET"), timeout_seconds, opener=opener)
    # Do not persist credentials even if a provider reflects a token into a label.
    return format_quota(payload).replace(auth.access_token, "[redacted]")
