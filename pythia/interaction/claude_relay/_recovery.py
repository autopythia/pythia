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
