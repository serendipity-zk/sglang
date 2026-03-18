"""Utilities for tracking router message acknowledgments."""

from __future__ import annotations

import logging
import threading
from typing import Optional, Set, Tuple

logger = logging.getLogger(__name__)

MAX_SEEN_IDS = 10000


class RouterMessageAckTracker:
    """Track contiguous router ack state and drop duplicate router messages."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_generation: Optional[int] = None
        self._last_contiguous_id: int = -1
        self._pending_ids: Set[int] = set()
        self._seen_ids: Set[int] = set()

    def record(self, generation: Optional[int], message_id: Optional[int]) -> bool:
        """Record a router message and return whether it should be processed."""
        if generation is None or message_id is None:
            return True

        with self._lock:
            if self._current_generation is None or generation > self._current_generation:
                self._current_generation = generation
                self._last_contiguous_id = -1
                self._pending_ids.clear()
                self._seen_ids.clear()
            elif generation < self._current_generation:
                logger.info(
                    "Dropping stale generation message: generation=%s, current_generation=%s",
                    generation,
                    self._current_generation,
                )
                return False

            if message_id <= self._last_contiguous_id or message_id in self._seen_ids:
                logger.info(
                    "Dropping duplicate message: generation=%s, message_id=%s",
                    generation,
                    message_id,
                )
                return False

            self._seen_ids.add(message_id)
            if len(self._seen_ids) > MAX_SEEN_IDS:
                self._seen_ids = {
                    seen_id
                    for seen_id in self._seen_ids
                    if seen_id > self._last_contiguous_id
                }

            self._pending_ids.add(message_id)
            next_expected = self._last_contiguous_id + 1
            while next_expected in self._pending_ids:
                self._pending_ids.remove(next_expected)
                self._last_contiguous_id = next_expected
                next_expected += 1

            return True

    def get_state(self) -> Tuple[Optional[int], Optional[int]]:
        with self._lock:
            if self._current_generation is None:
                return None, None
            last_id = None if self._last_contiguous_id < 0 else self._last_contiguous_id
            return self._current_generation, last_id
