"""
Lightweight per-server UI state and helpers.

This module provides a thread-safe, in-process counter store for:
- accepted requests (cumulative)
- last observed batch size

It is intentionally simple and dependency-free, so it can be extended later
to carry more metrics without impacting the serving path. A separate CLI
client can poll an HTTP endpoint that exposes these values.
"""

from __future__ import annotations

import threading
from typing import Dict


class _ServerUiState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._accepted_requests: int = 0
        self._last_batch_size: int = 0

    def inc_accepted(self, n: int = 1) -> None:
        if n <= 0:
            return
        with self._lock:
            self._accepted_requests += n

    def set_last_batch_size(self, n: int) -> None:
        if n < 0:
            n = 0
        with self._lock:
            self._last_batch_size = n

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "accepted_requests": self._accepted_requests,
                "last_batch_size": self._last_batch_size,
            }


# Singleton state used by the server
state = _ServerUiState()


def inc_accepted(n: int = 1) -> None:
    state.inc_accepted(n)


def set_last_batch_size(n: int) -> None:
    state.set_last_batch_size(n)


def snapshot() -> Dict[str, int]:
    return state.snapshot()

