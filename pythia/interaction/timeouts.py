"""Shared timeout defaults and validation for interaction requests and login.

HTTP timeouts apply to blocking I/O, not whole-operation wall-clock deadlines:
each blocking operation of each attempt gets the whole budget. A model
request's timeout comes from, highest first: a per-call
``SampleParams.request_timeout_seconds`` (the CLI projects
``/config timeouts_request_seconds``), ``--timeouts-request-seconds``, the model
catalog's ``timeouts.request_seconds``, then
``DEFAULT_REQUEST_TIMEOUT_SECONDS``. Claude Relay deadlines come from their
launch options, then the catalog's ``timeouts.<name>_seconds``, then the
relay defaults below. The idle deadline after 5-minute cache writes has no
launch option.

The CLI's account commands keep the fixed ``DEFAULT_LOGIN_TIMEOUT_SECONDS``:
it is the ``/login`` flow budget, which also caps the token exchange, and the
timeout of each ``/quota`` request. Model timeouts do not apply to them.

A model's tool call waits at most ``DEFAULT_TOOL_TIMEOUT_SECONDS``, on every
route: a command's wait for output (``yield_time_ms``, at most 59000 by
default) and icron's ``exec`` wait, and the timeout of the tools that wait.

These constants are startup defaults; change them before starting the
process. UI polling and cleanup timings are independent and intentionally do
not use them.

``MAX_TIMEOUT_SECONDS``, ``validate_timeout_seconds`` and
``validate_idle_seconds`` (which also accepts 0) live with the model catalog's
dependency-free validators and are re-exported here.
"""

from .model_catalog import MAX_TIMEOUT_SECONDS
from .model_catalog import validate_idle_seconds
from .model_catalog import validate_timeout_seconds


DEFAULT_REQUEST_TIMEOUT_SECONDS = 1200.0
DEFAULT_LOGIN_TIMEOUT_SECONDS = 120.0
# Claude Code 2.1.289 gives up on an MCP tool call 60 s after sending it unless
# told otherwise. The Claude relay raises that deadline
# (claude_relay._cli_protocol.DEFAULT_NATIVE_TOOL_TIMEOUT_SECONDS), so this is
# a backstop there, and the same limit on every other route.
# doc/claude-relay-tool-deadline-plan.md.
DEFAULT_TOOL_TIMEOUT_SECONDS = 59.0
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
# cache lifetime starts: an hour less 5 s, just before the 1-hour lifetime that
# Claude Code gets on a subscription ends. After a response that wrote 5-minute
# cache entries (on usage credits, or an API key), the shorter one applies:
# 305 s, 5 s past that lifetime.
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
    "DEFAULT_TOOL_TIMEOUT_SECONDS",
    "MAX_TIMEOUT_SECONDS",
    "validate_idle_seconds",
    "validate_timeout_seconds",
]
