"""Shared HTTP identity for interaction requests."""

from __future__ import annotations

import urllib.request


# TODO: Replace urllib's generic identity with a stable, distinctive Pythia
# User-Agent.
USER_AGENT = next(
    value
    for name, value in urllib.request.OpenerDirector().addheaders
    if name.lower() == "user-agent"
)
