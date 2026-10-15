"""Narrow native MCP-expiry recognition, not a generic retry policy."""
from __future__ import annotations
from dataclasses import dataclass

from ..model import ModelTransportError


NATIVE_MCP_TIMEOUT = 'native_mcp_timeout'
# What the CLI reports as a call's result when it gives up on the call: its
# 60 s HTTP abort while MCP_TOOL_TIMEOUT is unset, else its tool timeout,
# which names the relay's MCP server (the --mcp-config key) and the tool.
NATIVE_TIMEOUT_TEXT = 'The operation timed out.'
MCP_SERVER_NAME = 'pythia'


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
    """Private runtime cause; not permission for the harness to retry yet."""
    rpc_code = -32000

    def __init__(self, call: ExpiredCall):
        super().__init__('Native MCP operation expired before consuming the host reply')
        self.call = call
