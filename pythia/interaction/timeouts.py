"""Shared startup defaults for interaction HTTP requests and login.

HTTP timeouts apply to blocking I/O, not whole-operation wall-clock deadlines.
The login flow has a separate budget that also caps its token-exchange timeout.
Change these defaults before starting the process; use explicit call/endpoint
options for runtime overrides. Command waits, UI polling, and cleanup timings
are independent and intentionally do not use these defaults.
"""


DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_LOGIN_TIMEOUT_SECONDS = 120.0


__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "DEFAULT_LOGIN_TIMEOUT_SECONDS",
]
