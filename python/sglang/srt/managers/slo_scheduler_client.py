"""ZMQ DEALER client for the external sidecar scheduler."""

from __future__ import annotations

import logging
import time

import zmq

logger = logging.getLogger(__name__)


class SLOSchedulerClient:
    """Send engine state to the sidecar and receive a correlated decision."""

    def __init__(self, addr: str, timeout_ms: int = 50):
        self.addr = addr
        self.timeout_ms = timeout_ms

        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.setsockopt(zmq.RCVHWM, 2)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(addr)

    def _serialize_engine_state(self, engine_state) -> bytes:
        from slo_scheduler.messages.serialization import serialize_engine_state

        return serialize_engine_state(engine_state)

    def _deserialize_decision(self, payload: bytes):
        from slo_scheduler.messages.serialization import deserialize_decision

        return deserialize_decision(payload)

    def send_and_recv(self, engine_state, current_iteration: int):
        """Return a matching sidecar decision or None on timeout/error/staleness."""
        try:
            payload = self._serialize_engine_state(engine_state)
            self.socket.send_multipart([b"", payload], flags=zmq.DONTWAIT)
        except zmq.Again:
            logger.warning("SLOSchedulerClient: send HWM reached, dropping message")
            return None
        except Exception:
            logger.exception("SLOSchedulerClient: send failed")
            return None

        start = time.monotonic()
        if not self.socket.poll(self.timeout_ms):
            return None

        drained = 0
        while True:
            try:
                while True:
                    frames = self.socket.recv_multipart(flags=zmq.DONTWAIT)
                    decision = self._deserialize_decision(frames[-1])
                    if decision.iteration_count == current_iteration:
                        if drained > 0:
                            logger.info(
                                "SLOSchedulerClient: drained %d stale responses "
                                "(expected iter=%d)",
                                drained,
                                current_iteration,
                            )
                        return decision
                    drained += 1
            except zmq.Again:
                pass
            except Exception:
                logger.exception("SLOSchedulerClient: recv/deserialize failed")
                return None

            elapsed_ms = (time.monotonic() - start) * 1000
            remaining_ms = int(self.timeout_ms - elapsed_ms)
            if remaining_ms <= 0 or not self.socket.poll(remaining_ms):
                if drained > 0:
                    logger.info(
                        "SLOSchedulerClient: drained %d stale responses "
                        "(expected iter=%d)",
                        drained,
                        current_iteration,
                    )
                return None

    def close(self):
        self.socket.close()
        self.ctx.term()
