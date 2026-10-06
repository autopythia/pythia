"""Shared timeout defaults and validation for interaction requests and login.

HTTP timeouts apply to blocking I/O, not whole-operation wall-clock deadlines:
each blocking operation of each attempt gets the whole budget. A model
request's timeout comes from, highest first: a per-call
``SampleParams.request_timeout_seconds`` (the CLI projects
``/config request_timeout_seconds``), ``--request-timeout-seconds``, the model
catalog's ``timeouts.request_seconds``, then
``DEFAULT_REQUEST_TIMEOUT_SECONDS``. Claude Relay deadlines come from their
launch options, then the catalog's ``timeouts.<name>_seconds``, then the
relay defaults below.

The CLI's account commands keep the fixed ``DEFAULT_LOGIN_TIMEOUT_SECONDS``:
it is the ``/login`` flow budget, which also caps the token exchange, and the
timeout of each ``/quota`` request. Model timeouts do not apply to them.

These constants are startup defaults; change them before starting the
process. Command waits, UI polling, and cleanup timings are independent and
intentionally do not use them.

``MAX_TIMEOUT_SECONDS`` and ``validate_timeout_seconds`` live with the model
catalog's dependency-free validators and are re-exported here.
"""

from .model_catalog import MAX_TIMEOUT_SECONDS
from .model_catalog import validate_timeout_seconds


DEFAULT_REQUEST_TIMEOUT_SECONDS = 1200.0
DEFAULT_LOGIN_TIMEOUT_SECONDS = 120.0
DEFAULT_CLAUDE_RELAY_GENERATION_TIMEOUT_SECONDS = 1200.0
DEFAULT_CLAUDE_RELAY_PARKED_TIMEOUT_SECONDS = 1800.0
DEFAULT_CLAUDE_RELAY_STARTUP_TIMEOUT_SECONDS = 30.0
DEFAULT_CLAUDE_RELAY_STOP_TIMEOUT_SECONDS = 5.0


__all__ = [
    "DEFAULT_CLAUDE_RELAY_GENERATION_TIMEOUT_SECONDS",
    "DEFAULT_CLAUDE_RELAY_PARKED_TIMEOUT_SECONDS",
    "DEFAULT_CLAUDE_RELAY_STARTUP_TIMEOUT_SECONDS",
    "DEFAULT_CLAUDE_RELAY_STOP_TIMEOUT_SECONDS",
    "DEFAULT_LOGIN_TIMEOUT_SECONDS",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "MAX_TIMEOUT_SECONDS",
    "validate_timeout_seconds",
]
