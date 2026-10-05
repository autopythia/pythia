"""Stdlib-only model adapter for this repository's Claude Relay contract.

The standalone client/broker lives in claude-relay/claude_relay.py. This package
owns the model facade, native CLI codec/runtime, MCP mailbox and context mapping.
No native CLI/SDK is loaded or executed on import. Live compatibility is still
unverified; see claude-relay/SETUP.md for deployment and smoke testing.
"""

from ._model import ClaudeRelayEndpoint, ClaudeRelayModel

__all__ = ["ClaudeRelayEndpoint", "ClaudeRelayModel"]
