"""ZMQ DEALER client for communicating with the SLO scheduler sidecar."""

import logging
import pickle
import time
import zmq

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 3
FALLBACK_PROBE_INTERVAL_S = 5.0


class SLOSchedulerClient:
    """Sends EngineState to sidecar, receives SchedulingDecision.

    Uses DEALER socket to match the sidecar's ROUTER socket.
    Non-blocking sends with timeout-based receives.
    Enters fallback mode after consecutive failures and probes periodically.
    """

    def __init__(self, addr: str, timeout_ms: int = 50):
        self.addr = addr
        self.timeout_ms = timeout_ms

        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.setsockopt(zmq.RCVHWM, 2)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(addr)

        # Fallback state
        self.consecutive_failures = 0
        self.fallback_mode = False
        self.last_probe_time = 0.0

        logger.info(f"SLOSchedulerClient connected to {addr} (timeout={timeout_ms}ms)")

    def send_and_recv(self, engine_state, current_iteration: int):
        """Send engine state and receive scheduling decision.

        Args:
            engine_state: EngineState dataclass to send.
            current_iteration: Current iteration count for staleness check.

        Returns:
            SchedulingDecision if successful, None on timeout/error/stale.
        """
        # In fallback mode, only probe periodically
        if self.fallback_mode:
            now = time.monotonic()
            if now - self.last_probe_time < FALLBACK_PROBE_INTERVAL_S:
                return None
            self.last_probe_time = now
            logger.info("SLOSchedulerClient: probing sidecar recovery...")

        # Serialize and send
        try:
            payload = pickle.dumps(engine_state, protocol=pickle.HIGHEST_PROTOCOL)
            self.socket.send_multipart([b"", payload], flags=zmq.DONTWAIT)
        except zmq.Again:
            logger.warning("SLOSchedulerClient: send HWM reached, dropping message")
            self._record_failure()
            return None
        except Exception:
            logger.exception("SLOSchedulerClient: send failed")
            self._record_failure()
            return None

        # Poll for response
        if not self.socket.poll(self.timeout_ms):
            self._record_failure()
            return None

        # Receive and deserialize
        try:
            frames = self.socket.recv_multipart(flags=zmq.DONTWAIT)
            decision = pickle.loads(frames[-1])
        except Exception:
            logger.exception("SLOSchedulerClient: recv/deserialize failed")
            self._record_failure()
            return None

        # Staleness check
        # NOTE: stale decisions return None but do NOT call _record_failure().
        # This is intentional — transient sidecar lag (1-2 iterations behind) should
        # degrade gracefully to internal scheduling without triggering fallback mode.
        # If the sidecar is persistently stale, the engine silently uses internal
        # decisions (all returns are None) but remains ready to accept fresh ones.
        if decision.iteration_count != current_iteration:
            logger.debug(
                f"SLOSchedulerClient: stale decision "
                f"(got iter={decision.iteration_count}, expected={current_iteration})"
            )
            return None

        # Success — reset failure tracking
        self._record_success()
        return decision

    def _record_failure(self):
        self.consecutive_failures += 1
        if not self.fallback_mode and self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self.fallback_mode = True
            self.last_probe_time = time.monotonic()
            logger.warning(
                f"SLOSchedulerClient: entering fallback mode "
                f"after {self.consecutive_failures} consecutive failures"
            )

    def _record_success(self):
        if self.fallback_mode:
            logger.info("SLOSchedulerClient: sidecar recovered, exiting fallback mode")
        self.consecutive_failures = 0
        self.fallback_mode = False

    def close(self):
        self.socket.close()
        self.ctx.term()
        logger.info("SLOSchedulerClient closed")
