"""Utilities for tracking router message acknowledgments."""

from __future__ import annotations

import threading
from typing import Optional, Set, Tuple

import logging
logger = logging.getLogger(__name__)

# Maximum number of seen message IDs to track for deduplication
MAX_SEEN_IDS = 10000


class RouterMessageAckTracker:
    """Tracks highest contiguous router message IDs per generation for stats reporting.

    Also provides deduplication support for resent messages.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_generation: Optional[int] = None
        self._last_contiguous_id: int = -1
        self._pending_ids: Set[int] = set()
        self._seen_ids: Set[int] = set()  # Track all seen message IDs for deduplication
        self._last_gap_log_size: int = -1

    def record(self, generation: Optional[int], message_id: Optional[int]) -> bool:
        """Record that a request with (generation, message_id) was received.

        Returns:
            True if this is a new message that should be processed.
            False if this is a duplicate message that should be dropped.
        """
        if generation is None or message_id is None:
            return True  # Not tracked, allow through

        with self._lock:
            prev_generation = self._current_generation
            if self._current_generation is None or generation > self._current_generation:
                # Router restarted or first observation; reset state for new generation.
                self._current_generation = generation
                self._last_contiguous_id = -1
                self._pending_ids.clear()
                self._seen_ids.clear()
                self._last_gap_log_size = -1
                logger.info(
                    "[ACK_TRACKER_GEN_RESET] prev_gen=%s new_gen=%s",
                    prev_generation,
                    generation,
                )
            elif generation < self._current_generation:
                # Ignore stale generations.
                logger.info(f"Dropping stale generation message: generation={generation}, current_generation={self._current_generation}")
                return False

            # Check for duplicate - either already processed (contiguous) or already seen
            if message_id <= self._last_contiguous_id or message_id in self._seen_ids:
                logger.info(f"Dropping duplicate message: generation={generation}, message_id={message_id}")
                return False

            # Track this message ID for future deduplication
            self._seen_ids.add(message_id)

            # Memory management: clean up old seen IDs when limit exceeded
            if len(self._seen_ids) > MAX_SEEN_IDS:
                # Keep only IDs greater than last_contiguous_id (still relevant)
                self._seen_ids = {id for id in self._seen_ids if id > self._last_contiguous_id}

            # Track for contiguous acknowledgment
            self._pending_ids.add(message_id)

            prev_last_contiguous = self._last_contiguous_id
            next_expected = self._last_contiguous_id + 1
            while next_expected in self._pending_ids:
                self._pending_ids.remove(next_expected)
                self._last_contiguous_id = next_expected
                next_expected += 1

            advanced = self._last_contiguous_id - prev_last_contiguous
            pending_gaps = len(self._pending_ids)
            if advanced > 0:
                logger.debug(
                    "[ACK_TRACKER_PROGRESS] gen=%s msg=%s advanced=%s last_contiguous=%s pending_gaps=%s seen=%s",
                    generation,
                    message_id,
                    advanced,
                    self._last_contiguous_id,
                    pending_gaps,
                    len(self._seen_ids),
                )
                self._last_gap_log_size = pending_gaps
            elif pending_gaps != self._last_gap_log_size and (
                pending_gaps <= 8 or pending_gaps % 16 == 0
            ):
                logger.debug(
                    "[ACK_TRACKER_GAP] gen=%s msg=%s last_contiguous=%s pending_gaps=%s seen=%s",
                    generation,
                    message_id,
                    self._last_contiguous_id,
                    pending_gaps,
                    len(self._seen_ids),
                )
                self._last_gap_log_size = pending_gaps

            return True

    def get_state(self) -> Tuple[Optional[int], Optional[int]]:
        """Return (generation, last_contiguous_id) for stats reporting."""
        with self._lock:
            if self._current_generation is None:
                return None, None
            last_id = (
                None if self._last_contiguous_id < 0 else self._last_contiguous_id
            )
            return self._current_generation, last_id
