from __future__ import annotations

from threading import Lock


class WorkspacePolicy:
    """Small shared switch read at the start of each path resolution."""

    def __init__(self, enabled: bool = True) -> None:
        self._lock = Lock()
        self._enabled = self._require_bool(enabled)

    @staticmethod
    def _require_bool(value: object) -> bool:
        if not isinstance(value, bool):
            raise TypeError("enable_workspace must be a bool")
        return value

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        enabled = self._require_bool(enabled)
        with self._lock:
            self._enabled = enabled


__all__ = [
    "WorkspacePolicy",
]
