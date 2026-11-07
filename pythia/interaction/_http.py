"""Shared HTTP identity for interaction requests."""

from __future__ import annotations


# The default User-Agent. A model's requests send it unless the model's
# endpoint has its own (the model catalog's endpoint.user_agent, or
# --endpoint-user-agent), and the account requests always do. It names Pythia,
# with pyproject.toml's version, rather than urllib's default
# (Python-urllib/3.11): Cloudflare's browser-signature check refuses that one
# with error 1010 at some endpoints, such as Tinker's.
USER_AGENT = "pythia/0.1.0.dev0"


def user_agent_for(endpoint) -> str:
    """The User-Agent of a model's requests: its EndpointSpec's, else USER_AGENT."""
    return endpoint.user_agent or USER_AGENT
