"""Bounded, non-redirecting account requests with secret-safe diagnostics."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class AccountServiceError(ValueError):
    """Only application-authored, credential-free messages belong here."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_json(request, timeout_seconds, *, opener=None):
    open_request = opener or urllib.request.build_opener(_NoRedirect()).open
    try:
        response = open_request(request, timeout=timeout_seconds)
        try:
            if not 200 <= getattr(response, "status", 200) < 300:
                raise AccountServiceError("Account service returned an unsuccessful response.")
            body = response.read(1_048_577)
        finally:
            response.close()
        if len(body) > 1_048_576:
            raise AccountServiceError("Account service response was too large.")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise AccountServiceError("Account service returned invalid data.")
        return value
    except urllib.error.HTTPError as exc:
        exc.close()
        raise AccountServiceError(f"Account service HTTP {exc.code}.") from None
    except AccountServiceError:
        raise
    except Exception:
        # Do not echo response bodies, request URLs, headers, or arbitrary errors.
        raise AccountServiceError("Account service request failed or returned invalid data.") from None
