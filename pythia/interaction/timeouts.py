"""Shared timeout defaults and validation for interaction requests and login.

HTTP timeouts apply to blocking I/O, not whole-operation wall-clock deadlines:
each blocking operation of each attempt gets the whole budget. A model
request's timeout comes from, highest first: a per-call
``SampleParams.request_timeout_seconds`` (the CLI projects
``/config request_timeout_seconds``), ``--request-timeout-seconds``, the model
catalog's ``timeouts.request_seconds``, then
``DEFAULT_REQUEST_TIMEOUT_SECONDS``. Claude Relay deadlines come from their
launch options, then the catalog's ``timeouts.<name>_seconds``, then the
relay defaults below. The idle deadline after 5-minute cache writes has no
launch option.

The CLI's account commands keep the fixed ``DEFAULT_LOGIN_TIMEOUT_SECONDS``:
it is the ``/login`` flow budget, which also caps the token exchange, and the
timeout of each ``/quota`` request. Model timeouts do not apply to them.

These constants are startup defaults; change them before starting the
process. Command waits, UI polling, and cleanup timings are independent and
intentionally do not use them.

``MAX_TIMEOUT_SECONDS``, ``validate_timeout_seconds`` and
``validate_idle_seconds`` (which also accepts 0) live with the model catalog's
dependency-free validators and are re-exported here.
"""

from .model_catalog import MAX_TIMEOUT_SECONDS
from .model_catalog import validate_idle_seconds
from .model_catalog import validate_timeout_seconds


DEFAULT_REQUEST_TIMEOUT_SECONDS = 1200.0
DEFAULT_LOGIN_TIMEOUT_SECONDS = 120.0
DEFAULT_CLAUDE_RELAY_GENERATION_TIMEOUT_SECONDS = 1200.0
# How long the relay's MCP server holds a native tool call for its result: just
# past the CLI's own deadline for the call (DEFAULT_NATIVE_TOOL_TIMEOUT_SECONDS,
# an hour), so that a process parked after a yield waits as long as its prompt
# cache lives, and the CLI's timeout, which the relay recognizes, comes first if
# both are ever reached. doc/claude-relay-warm-yield-plan.md, section 4.2.
DEFAULT_CLAUDE_RELAY_PARKED_TIMEOUT_SECONDS = 3605.0
DEFAULT_CLAUDE_RELAY_STARTUP_TIMEOUT_SECONDS = 30.0
DEFAULT_CLAUDE_RELAY_STOP_TIMEOUT_SECONDS = 5.0
# How long the relay keeps a native process after a final answer, for the
# next turn, counted from the start of that answer's request, where its prompt
# cache lifetime starts: 59 minutes, before the 1-hour lifetime that Claude
# Code gets on a subscription ends. After a response that wrote 5-minute cache
# entries (on usage credits, or an API key), the shorter one applies.
# doc/claude-relay-warm-delivery-plan.md, section 3.3.
DEFAULT_CLAUDE_RELAY_IDLE_TIMEOUT_SECONDS = 3595.0
DEFAULT_CLAUDE_RELAY_SHORT_CACHE_IDLE_TIMEOUT_SECONDS = 305.0


__all__ = [
    "DEFAULT_CLAUDE_RELAY_GENERATION_TIMEOUT_SECONDS",
    "DEFAULT_CLAUDE_RELAY_IDLE_TIMEOUT_SECONDS",
    "DEFAULT_CLAUDE_RELAY_PARKED_TIMEOUT_SECONDS",
    "DEFAULT_CLAUDE_RELAY_SHORT_CACHE_IDLE_TIMEOUT_SECONDS",
    "DEFAULT_CLAUDE_RELAY_STARTUP_TIMEOUT_SECONDS",
    "DEFAULT_CLAUDE_RELAY_STOP_TIMEOUT_SECONDS",
    "DEFAULT_LOGIN_TIMEOUT_SECONDS",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "MAX_TIMEOUT_SECONDS",
    "validate_idle_seconds",
    "validate_timeout_seconds",
]
