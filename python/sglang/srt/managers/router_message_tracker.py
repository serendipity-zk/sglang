"""Utilities for tracking router message acknowledgments."""

from __future__ import annotations

import threading
from typing import Optional, Set, Tuple

import logging
logger = logging.getLogger(__name__)

class RouterMessageAckTracker:
    """Tracks highest contiguous router message IDs per generation for stats reporting."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_generation: Optional[int] = None
        self._last_contiguous_id: int = -1
        self._pending_ids: Set[int] = set()

    def record(self, generation: Optional[int], message_id: Optional[int]) -> None:
        """Record that a request with (generation, message_id) was received."""
        logger.warning(f"Recording message: generation={generation}, message_id={message_id}")
        if generation is None or message_id is None:
            return

        with self._lock:
            if self._current_generation is None or generation > self._current_generation:
                # Router restarted or first observation; reset state for new generation.
                self._current_generation = generation
                self._last_contiguous_id = -1
                self._pending_ids.clear()
            elif generation < self._current_generation:
                # Ignore stale generations.
                logger.warning(f"Ignoring stale generation: generation={generation}, current_generation={self._current_generation}")
                return

            if message_id <= self._last_contiguous_id:
                logger.warning(f"Ignoring duplicate message: message_id={message_id}, last_contiguous_id={self._last_contiguous_id}")
                return

            self._pending_ids.add(message_id)

            next_expected = self._last_contiguous_id + 1
            while next_expected in self._pending_ids:
                self._pending_ids.remove(next_expected)
                self._last_contiguous_id = next_expected
                next_expected += 1

    def get_state(self) -> Tuple[Optional[int], Optional[int]]:
        """Return (generation, last_contiguous_id) for stats reporting."""
        with self._lock:
            if self._current_generation is None:
                return None, None
            last_id = (
                None if self._last_contiguous_id < 0 else self._last_contiguous_id
            )
            return self._current_generation, last_id
