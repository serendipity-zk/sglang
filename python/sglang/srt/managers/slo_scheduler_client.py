"""ZMQ DEALER client for communicating with the SLO scheduler sidecar."""

import logging
import time
import zmq

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 3
FALLBACK_PROBE_INTERVAL_S = 0.5


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
        # Cache only for identical scheduler state. The engine can invoke sidecar
        # multiple times with the same iteration_count while queue/KV state is
        # changing; those calls must still be sent.
        self._last_iteration_attempted = -1
        self._last_state_signature = None
        self._last_iteration_result = None

        logger.info(f"SLOSchedulerClient connected to {addr} (timeout={timeout_ms}ms)")

    def send_and_recv(self, engine_state, current_iteration: int):
        """Send engine state and receive scheduling decision.

        Args:
            engine_state: EngineState dataclass to send.
            current_iteration: Current iteration count for staleness check.

        Returns:
            SchedulingDecision if successful, None on timeout/error/stale.
        """
        state_signature = self._compute_state_signature(engine_state)

        # Return cached answer only when both iteration and sidecar-relevant
        # state are unchanged.
        if (
            current_iteration == self._last_iteration_attempted
            and state_signature == self._last_state_signature
        ):
            return self._last_iteration_result

        if current_iteration < self._last_iteration_attempted:
            # Defensive logging for unexpected iteration regressions.
            logger.warning(
                "SLOSchedulerClient: iteration regressed from %d to %d; resetting per-iteration cache",
                self._last_iteration_attempted,
                current_iteration,
            )

        self._last_iteration_attempted = current_iteration
        self._last_state_signature = state_signature
        self._last_iteration_result = None

        # In fallback mode, only probe periodically
        if self.fallback_mode:
            now = time.monotonic()
            if now - self.last_probe_time < FALLBACK_PROBE_INTERVAL_S:
                return None
            self.last_probe_time = now
            logger.info("SLOSchedulerClient: probing sidecar recovery...")

        # Serialize and send
        try:
            from slo_scheduler.messages.serialization import serialize_engine_state
            payload = serialize_engine_state(engine_state)
            self.socket.send_multipart([b"", payload], flags=zmq.DONTWAIT)
        except zmq.Again:
            logger.warning("SLOSchedulerClient: send HWM reached, dropping message")
            self._record_failure()
            return None
        except Exception:
            logger.exception("SLOSchedulerClient: send failed")
            self._record_failure()
            return None

        t_start = time.monotonic()

        # Poll for response
        if not self.socket.poll(self.timeout_ms):
            self._record_failure()
            return None

        # Drain-and-retry loop: consume all buffered responses looking for an
        # exact match on current_iteration.  If the buffer only contained stale
        # responses (e.g. from a fallback period, or the sidecar being 1 iteration
        # behind), poll again with the remaining time budget.  Repeat until we
        # either find the match or exhaust the timeout.
        best = None
        drained = 0
        while True:
            # Drain all immediately available responses
            try:
                while True:
                    frames = self.socket.recv_multipart(flags=zmq.DONTWAIT)
                    from slo_scheduler.messages.serialization import deserialize_decision
                    candidate = deserialize_decision(frames[-1])
                    if candidate.iteration_count == current_iteration:
                        best = candidate
                        break  # Exact match
                    drained += 1
            except zmq.Again:
                pass  # Buffer empty
            except Exception:
                logger.exception("SLOSchedulerClient: recv/deserialize failed")
                self._record_failure()
                return None

            if best is not None:
                break

            # No match yet — poll again if time remains
            elapsed_ms = (time.monotonic() - t_start) * 1000
            remaining_ms = int(self.timeout_ms - elapsed_ms)
            if remaining_ms <= 0 or not self.socket.poll(remaining_ms):
                break  # Timeout exhausted

        if drained > 0:
            logger.info(
                f"SLOSchedulerClient: drained {drained} stale responses "
                f"(expected iter={current_iteration})"
            )

        if best is None:
            return None

        # Success — reset failure tracking
        self._last_iteration_result = best
        self._record_success()
        return best

    def _compute_state_signature(self, engine_state):
        """Compute a compact signature for sidecar-relevant scheduler state."""
        current = engine_state.current
        scheduling = engine_state.scheduling
        waiting = scheduling.waiting_requests
        decode = scheduling.decode_requests
        chunked = scheduling.chunked_requests

        # Include queue/KV/ack context and lightweight request identity hints.
        waiting_head = waiting[0].request_id if waiting else None
        waiting_tail = waiting[-1].request_id if waiting else None
        decode_head = decode[0].request_id if decode else None
        chunked_head = chunked[0].request_id if chunked else None

        return (
            current.num_running_requests,
            current.num_waiting_requests,
            current.kv_tokens_used,
            current.router_generation,
            current.router_last_ack_id,
            len(waiting),
            len(decode),
            len(chunked),
            waiting_head,
            waiting_tail,
            decode_head,
            chunked_head,
            engine_state.accepted_requests_count,
        )

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
