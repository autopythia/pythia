"""Narrow native MCP-expiry recognition, not a generic retry policy."""
from __future__ import annotations
from dataclasses import dataclass

from ..model import ModelTransportError, RetryState


NATIVE_MCP_TIMEOUT = 'native_mcp_timeout'
# Why a kept native process went unused (continuation_abandoned): a delivery it
# didn't take up, which is also the sample's recovery label; it ended while
# idle, by exiting or with a record no idle CLI writes; its idle deadline; or,
# parked after a yield, the deadline of its wait for the batch's results.
DELIVERY_UNACKNOWLEDGED = 'delivery_unacknowledged'
IDLE_EXIT = 'idle_exit'
IDLE_RECORD = 'idle_record'
IDLE_DEADLINE = 'idle_deadline'
PARKED_DEADLINE = 'parked_deadline'
# What the CLI reports as a call's result when it gives up on the call: its
# 60 s HTTP abort while MCP_TOOL_TIMEOUT is unset, else its tool timeout,
# which names the relay's MCP server (the --mcp-config key) and the tool.
NATIVE_TIMEOUT_TEXT = 'The operation timed out.'
MCP_SERVER_NAME = 'pythia'


def final(error):
    """Mark a relay failure that no automatic retry can fix (ClaudeRelayModel
    never vouches for it): the host's own retirement of the continuation, or a
    startup or compatibility check, which a cold import fails the same way."""
    error.relay_final = True
    return error


def coded(error, code, retry_class=None):
    """``error``, with the code that its failure saves (its ``error_code``),
    and its retry class (ModelError.retry_class) where not its type's."""
    error.error_code = code
    if retry_class is not None:
        error.retry_class = retry_class
    return error


# The error types of a failed API request that the native CLI reports after
# its own retries: an overloaded or failing API, or a rate limit, which the
# relay refuses the rate-limit setting for (Claude Code retries those itself);
# and an invalid or unbillable request. Claude Code writes them bare
# (rate_limit, server_error, invalid_request, billing_error); the API's own
# types (overloaded_error...) come in an object or a stream's error event.
TRANSIENT_NATIVE_ERRORS = frozenset({'rate_limit', 'rate_limit_error', 'server_error',
                                     'overloaded_error', 'api_error'})
REQUEST_NATIVE_ERRORS = frozenset({'invalid_request', 'invalid_request_error', 'billing_error'})


def native_error_class(code):
    """The retry class of a failure that a native error type names: transient
    for an overloaded or failing API or a rate limit, error for an invalid or
    unbillable request; None for any other code."""
    return ('transient' if code in TRANSIENT_NATIVE_ERRORS
            else 'error' if code in REQUEST_NATIVE_ERRORS else None)


def result_text(content):
    """The observed CLI string / single MCP text-block representations only."""
    if isinstance(content, str):
        return content
    if (isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict)
            and content[0].get('type') == 'text' and isinstance(content[0].get('text'), str)):
        return content[0]['text']
    return None


def native_timeout_texts(tool, timeout_seconds=None):
    """The exact results the CLI writes when it gives up on a call to ``tool``.

    ``timeout_seconds`` is the MCP_TOOL_TIMEOUT the relay set, in whole seconds.
    """
    texts = (NATIVE_TIMEOUT_TEXT,)
    if timeout_seconds is not None:
        texts += (f'MCP server "{MCP_SERVER_NAME}" tool "{tool}" timed out after {timeout_seconds}s',)
    return texts


def is_native_timeout(block, texts):
    return (block.get('type') == 'tool_result' and block.get('is_error') is True
            and result_text(block.get('content')) in texts)


def is_host_echo(result, block):
    """The host's own error result, echoed unchanged: an ordinary tool failure."""
    text = result_text(block.get('content'))
    return (text is not None and isinstance(result, dict) and result.get('isError') is True
            and result_text(result.get('content')) == text)


def native_tool_rejection(record, streamed):
    """The CLI's own rejection of a streamed call to a tool it has not got, or None.

    Observed on CLI 2.1.289 (doc/claude-relay-tool-name-plan.md): right after a
    tool block's snapshot, before its message completes, the CLI answers the
    call with one error result naming the tool. ``record`` is a native ``user``
    record; ``streamed`` maps the open message's tool_use IDs to their names.
    Only that exact shape is recognized. It diagnoses a record that ordinary
    validation rejects anyway, and never permits accepting anything. Returns
    ``(native ID, tool name)``.
    """
    message = record.get('message')
    blocks = message.get('content') if isinstance(message, dict) else None
    if not isinstance(blocks, list) or len(blocks) != 1 or not isinstance(blocks[0], dict):
        return None
    block = blocks[0]
    ident = block.get('tool_use_id')
    name = streamed.get(ident) if isinstance(ident, str) else None
    if not isinstance(name, str) or block.get('type') != 'tool_result' or block.get('is_error') is not True:
        return None
    text = f'Error: No such tool available: {name}'
    if (result_text(block.get('content')) != f'<tool_use_error>{text}</tool_use_error>'
            or record.get('tool_use_result') != text):
        return None
    return ident, name


@dataclass(frozen=True)
class ExpiredCall:
    native_id: str
    released: bool
    returned: bool


class NativeMCPTimeout(ModelTransportError):
    """Private runtime cause; not permission for the harness to retry yet.

    It is raised ``RetryState.NONE``. ClaudeRelayModel.sample makes it ELIGIBLE
    once its checks pass, then either converts it to ModelContinuationExpired,
    once the teardown is confirmed, or raises it BLOCKED.
    """
    rpc_code = -32000

    def __init__(self, call: ExpiredCall, *, retry_state=RetryState.NONE):
        super().__init__('Native MCP operation expired before consuming the host reply',
                         retry_state=retry_state)
        self.call = call


class DeliveryUnacknowledged(ModelTransportError):
    """Messages delivered to a live process that the CLI didn't take up: it
    didn't report reading them in time (``queued``), answered without them (no
    ``started`` before its next reply, or no echo of a turn's record), or ended
    first. Or the process had already ended while idle (``reason``
    ``idle_exit`` or ``idle_record``).

    Nothing the process wrote after the delivery is adopted, and none of its
    calls is registered, so the saved context is unchanged: the model retires
    the process and answers the same sample from a cold import, and saves no
    failure (doc/claude-relay-warm-delivery-plan.md, section 4).
    """

    error_code = DELIVERY_UNACKNOWLEDGED

    def __init__(self, message='Native CLI did not take up the delivered messages', *,
                 reason=DELIVERY_UNACKNOWLEDGED, retry_state=RetryState.NONE):
        super().__init__(message, retry_state=retry_state)
        self.reason = reason
