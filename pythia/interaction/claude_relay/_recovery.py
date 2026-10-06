"""Narrow native MCP-expiry recognition, not a generic retry policy."""
from __future__ import annotations
from dataclasses import dataclass

from ..model import ModelTransportError


NATIVE_MCP_TIMEOUT = 'native_mcp_timeout'
NATIVE_TIMEOUT_TEXT = 'The operation timed out.'


def result_text(content):
    """The observed CLI string / single MCP text-block representations only."""
    if isinstance(content, str):
        return content
    if (isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict)
            and content[0].get('type') == 'text' and isinstance(content[0].get('text'), str)):
        return content[0]['text']
    return None


def is_native_timeout(block):
    return (block.get('type') == 'tool_result' and block.get('is_error') is True
            and result_text(block.get('content')) == NATIVE_TIMEOUT_TEXT)


def is_host_timeout_echo(result):
    return (isinstance(result, dict) and result.get('isError') is True
            and result_text(result.get('content')) == NATIVE_TIMEOUT_TEXT)


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
