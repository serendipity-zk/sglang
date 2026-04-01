"""ZMQ DEALER client for the external sidecar scheduler."""

from __future__ import annotations

import logging
import time

import zmq

logger = logging.getLogger(__name__)

CLIENT_RPC_BREAKDOWN_KEYS = (
    "serialize_send",
    "wait",
    "recv_deserialize",
)


def _empty_client_rpc_breakdown():
    return {key: 0.0 for key in CLIENT_RPC_BREAKDOWN_KEYS}


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
        """Return (decision, wait_time_ms, rpc_breakdown_ms, decision_payload)."""
        rpc_breakdown_ms = _empty_client_rpc_breakdown()
        try:
            serialize_send_start = time.perf_counter()
            payload = self._serialize_engine_state(engine_state)
            self.socket.send_multipart([b"", payload], flags=zmq.DONTWAIT)
            rpc_breakdown_ms["serialize_send"] = (
                time.perf_counter() - serialize_send_start
            ) * 1000.0
        except zmq.Again:
            logger.warning("SLOSchedulerClient: send HWM reached, dropping message")
            return None, 0.0, rpc_breakdown_ms, None
        except Exception:
            logger.exception("SLOSchedulerClient: send failed")
            return None, 0.0, rpc_breakdown_ms, None

        wait_start = time.perf_counter()
        poll_start = time.perf_counter()
        if not self.socket.poll(self.timeout_ms):
            rpc_breakdown_ms["wait"] += (time.perf_counter() - poll_start) * 1000.0
            return None, (time.perf_counter() - wait_start) * 1000.0, rpc_breakdown_ms, None
        rpc_breakdown_ms["wait"] += (time.perf_counter() - poll_start) * 1000.0

        drained = 0
        while True:
            try:
                while True:
                    recv_start = time.perf_counter()
                    frames = self.socket.recv_multipart(flags=zmq.DONTWAIT)
                    decision_payload = frames[-1]
                    decision = self._deserialize_decision(decision_payload)
                    rpc_breakdown_ms["recv_deserialize"] += (
                        time.perf_counter() - recv_start
                    ) * 1000.0
                    if decision.iteration_count == current_iteration:
                        if drained > 0:
                            logger.info(
                                "SLOSchedulerClient: drained %d stale responses "
                                "(expected iter=%d)",
                                drained,
                                current_iteration,
                            )
                        return (
                            decision,
                            (time.perf_counter() - wait_start) * 1000.0,
                            rpc_breakdown_ms,
                            decision_payload,
                        )
                    drained += 1
            except zmq.Again:
                pass
            except Exception:
                logger.exception("SLOSchedulerClient: recv/deserialize failed")
                return (
                    None,
                    (time.perf_counter() - wait_start) * 1000.0,
                    rpc_breakdown_ms,
                    None,
                )

            elapsed_ms = (time.perf_counter() - wait_start) * 1000
            remaining_ms = int(self.timeout_ms - elapsed_ms)
            poll_start = time.perf_counter()
            if remaining_ms <= 0 or not self.socket.poll(remaining_ms):
                rpc_breakdown_ms["wait"] += (time.perf_counter() - poll_start) * 1000.0
                if drained > 0:
                    logger.info(
                        "SLOSchedulerClient: drained %d stale responses "
                        "(expected iter=%d)",
                        drained,
                        current_iteration,
                    )
                return (
                    None,
                    (time.perf_counter() - wait_start) * 1000.0,
                    rpc_breakdown_ms,
                    None,
                )
            rpc_breakdown_ms["wait"] += (time.perf_counter() - poll_start) * 1000.0

    def close(self):
        self.socket.close()
        self.ctx.term()
