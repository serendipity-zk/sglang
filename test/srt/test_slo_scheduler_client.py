"""Tests for SLOSchedulerClient ↔ SLOSchedulerServer communication.

Tests the engine-side ZMQ DEALER client against a real sidecar server.
Covers: normal roundtrip, timeout fallback, fallback recovery, staleness rejection,
and multi-process latency benchmarking.
"""

import asyncio
import multiprocessing
import os
import signal
import threading
import time

import pytest
import zmq

from sglang.srt.managers.slo_scheduler_client import (
    SLOSchedulerClient,
    FALLBACK_PROBE_INTERVAL_S,
    MAX_CONSECUTIVE_FAILURES,
)
from slo_scheduler.config.settings import (
    SidecarConfig,
    ServerConfig,
    PredictorConfig,
    SchedulingConfig,
    PrefillScheduleMode,
)
from slo_scheduler.server.zmq_server import SLOSchedulerServer
from slo_scheduler.messages.engine_state import (
    EngineState,
    FinishedIterationData,
    CurrentSnapshot,
    RequestInfo,
    SchedulingContext,
)
from slo_scheduler.messages.scheduling_decision import SchedulingDecision


GRID_PATH = "/sgl-workspace/sglang/sglang_profile/mode_3d.json"
IPC_ADDR = "ipc:///tmp/sglang_slo_client_test.sock"


# ---------------------------------------------------------------------------
# State builders (reused from slo_scheduler/tests/test_zmq_roundtrip.py)
# ---------------------------------------------------------------------------

def _make_state(iteration: int = 1) -> EngineState:
    now_ms = time.time() * 1000
    decode_reqs = [
        RequestInfo(
            request_id=f"d-{i}",
            target_tpot_ms=50.0,
            target_ttft_ms=500.0,
            arrival_time_ms=now_ms - 2000,
            tokens_generated=80 + i * 10,
            prompt_tokens=512,
            remaining_prefill=0,
            max_new_tokens=256,
        )
        for i in range(4)
    ]
    kv_used = sum(r.prompt_tokens + r.tokens_generated for r in decode_reqs)
    return EngineState(
        protocol_version=1,
        min_sidecar_version=1,
        worker_id="test-engine",
        finished=FinishedIterationData(
            iteration_count=iteration - 1,
            batch_size_tokens=4,
            prefill_chunk_pairs=[],
            kv_tokens_used=kv_used,
            forward_mode="DECODE",
            actual_time_ms=12.0,
        ),
        current=CurrentSnapshot(
            iteration_count=iteration,
            timestamp_ms=now_ms,
            num_running_requests=4,
            num_waiting_requests=0,
            kv_tokens_used=kv_used,
            kv_capacity=100000,
            running_requests=decode_reqs,
        ),
        scheduling=SchedulingContext(
            iteration_count=iteration + 1,  # Always 1 ahead of current
            scheduling_time_ms=now_ms,
            decode_requests=decode_reqs,
            chunked_requests=[],
            waiting_requests=[],
            kv_available=100000 - kv_used,
            kv_capacity=100000,
        ),
    )


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def _make_config() -> SidecarConfig:
    return SidecarConfig(
        worker_id="test-server",
        server=ServerConfig(zmq_bind=IPC_ADDR),
        predictor=PredictorConfig(type="mode_aware", grid_path=GRID_PATH),
        scheduling=SchedulingConfig(
            default_tpot_ms=50.0,
            max_prefill_tokens=4096,
            schedule_mode=PrefillScheduleMode.PREDICTOR,
        ),
    )


def _run_server_loop(server: SLOSchedulerServer):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())
    loop.close()


def _start_server():
    config = _make_config()
    server = SLOSchedulerServer(config)
    server.bind()
    t = threading.Thread(target=_run_server_loop, args=(server,), daemon=True)
    t.start()
    time.sleep(0.1)  # let server bind
    return server, t


def _stop_server(server, thread):
    server.stop()
    thread.join(timeout=3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNormalRoundtrip:

    def test_send_and_recv_returns_decision(self):
        server, t = _start_server()
        try:
            client = SLOSchedulerClient(IPC_ADDR, timeout_ms=5000)
            try:
                state = _make_state(iteration=1)
                # current_iteration must match scheduling.iteration_count (= iteration + 1)
                decision = client.send_and_recv(state, current_iteration=2)

                assert isinstance(decision, SchedulingDecision)
                assert decision.iteration_count == 2
                assert decision.max_prefill_tokens >= 0
                assert decision.predicted_iteration_time_ms >= 0
            finally:
                client.close()
        finally:
            _stop_server(server, t)

    def test_multiple_iterations(self):
        server, t = _start_server()
        try:
            client = SLOSchedulerClient(IPC_ADDR, timeout_ms=5000)
            try:
                for i in range(1, 11):
                    state = _make_state(iteration=i)
                    # scheduling.iteration_count = i + 1
                    decision = client.send_and_recv(state, current_iteration=i + 1)
                    assert decision is not None
                    assert decision.iteration_count == i + 1

                assert server.messages_received == 10
                assert client.consecutive_failures == 0
                assert not client.fallback_mode
            finally:
                client.close()
        finally:
            _stop_server(server, t)

    def test_duplicate_iteration_uses_cached_result(self):
        """Repeated calls with same iteration should not re-send to sidecar."""
        server, t = _start_server()
        try:
            client = SLOSchedulerClient(IPC_ADDR, timeout_ms=5000)
            try:
                state = _make_state(iteration=7)
                decision1 = client.send_and_recv(state, current_iteration=8)
                assert decision1 is not None
                recv_after_first = server.messages_received

                decision2 = client.send_and_recv(state, current_iteration=8)
                decision3 = client.send_and_recv(state, current_iteration=8)

                assert decision2 is decision1
                assert decision3 is decision1
                assert server.messages_received == recv_after_first
                assert not client.fallback_mode
            finally:
                client.close()
        finally:
            _stop_server(server, t)

    def test_same_iteration_state_change_resends(self):
        """Same iteration with changed state should be sent again."""
        server, t = _start_server()
        try:
            client = SLOSchedulerClient(IPC_ADDR, timeout_ms=5000)
            try:
                state1 = _make_state(iteration=7)
                decision1 = client.send_and_recv(state1, current_iteration=8)
                assert decision1 is not None
                recv_after_first = server.messages_received

                state2 = _make_state(iteration=7)
                state2.current.num_waiting_requests = 1
                state2.scheduling.waiting_requests = [
                    RequestInfo(
                        request_id="w-x",
                        target_tpot_ms=50.0,
                        target_ttft_ms=800.0,
                        arrival_time_ms=time.time() * 1000 - 10,
                        tokens_generated=0,
                        prompt_tokens=256,
                        remaining_prefill=256,
                        max_new_tokens=128,
                    )
                ]
                decision2 = client.send_and_recv(state2, current_iteration=8)

                assert decision2 is not None
                assert server.messages_received == recv_after_first + 1
                assert not client.fallback_mode
            finally:
                client.close()
        finally:
            _stop_server(server, t)


class TestStalenessRejection:

    def test_stale_decision_returns_none(self):
        """If sidecar returns a decision for a different iteration, client returns None."""
        server, t = _start_server()
        try:
            client = SLOSchedulerClient(IPC_ADDR, timeout_ms=5000)
            try:
                # Send state for iteration 5
                state = _make_state(iteration=5)
                # Sidecar returns decision.iteration_count=6 (scheduling=5+1), claim 99 → stale
                decision = client.send_and_recv(state, current_iteration=99)

                assert decision is None
            finally:
                client.close()
        finally:
            _stop_server(server, t)


class TestFallbackMode:

    def test_enters_fallback_after_consecutive_failures(self):
        """No server running → client enters fallback after MAX_CONSECUTIVE_FAILURES."""
        # Use a bogus address that nothing listens on
        bogus_addr = "ipc:///tmp/sglang_slo_client_test_bogus.sock"
        client = SLOSchedulerClient(bogus_addr, timeout_ms=50)
        try:
            for i in range(MAX_CONSECUTIVE_FAILURES):
                assert not client.fallback_mode
                state = _make_state(iteration=i + 1)
                result = client.send_and_recv(state, current_iteration=i + 2)
                assert result is None

            # Now should be in fallback
            assert client.fallback_mode
            assert client.consecutive_failures == MAX_CONSECUTIVE_FAILURES
        finally:
            client.close()

    def test_duplicate_iteration_timeout_only_counts_once(self):
        """Repeated same-iteration timeouts should not escalate to fallback."""
        bogus_addr = "ipc:///tmp/sglang_slo_client_test_bogus_same_iter.sock"
        client = SLOSchedulerClient(bogus_addr, timeout_ms=50)
        try:
            state = _make_state(iteration=1)
            assert client.send_and_recv(state, current_iteration=2) is None
            assert client.consecutive_failures == 1
            assert not client.fallback_mode

            for _ in range(10):
                assert client.send_and_recv(state, current_iteration=2) is None

            assert client.consecutive_failures == 1
            assert not client.fallback_mode
        finally:
            client.close()

    def test_fallback_skips_without_probe_interval(self):
        """In fallback mode, send_and_recv returns None immediately (before probe interval)."""
        bogus_addr = "ipc:///tmp/sglang_slo_client_test_bogus2.sock"
        client = SLOSchedulerClient(bogus_addr, timeout_ms=50)
        try:
            # Force into fallback
            client.fallback_mode = True
            client.last_probe_time = time.monotonic()

            state = _make_state(iteration=1)
            result = client.send_and_recv(state, current_iteration=2)
            assert result is None
            # Should return instantly without even trying to send
        finally:
            client.close()

    def test_recovers_from_fallback(self):
        """Client exits fallback once sidecar responds successfully."""
        server, t = _start_server()
        try:
            client = SLOSchedulerClient(IPC_ADDR, timeout_ms=5000)
            try:
                # Force into fallback with expired probe time
                client.fallback_mode = True
                client.consecutive_failures = MAX_CONSECUTIVE_FAILURES
                client.last_probe_time = time.monotonic() - FALLBACK_PROBE_INTERVAL_S - 1

                state = _make_state(iteration=1)
                decision = client.send_and_recv(state, current_iteration=2)

                assert decision is not None
                assert not client.fallback_mode
                assert client.consecutive_failures == 0
            finally:
                client.close()
        finally:
            _stop_server(server, t)


# ---------------------------------------------------------------------------
# Multi-process test with latency measurement
# ---------------------------------------------------------------------------

MP_IPC_ADDR = "ipc:///tmp/sglang_slo_client_mp_test.sock"


def _sidecar_worker(addr: str, ready_event, grid_path: str, schedule_mode: str = "predictor"):
    """Sidecar server running in a separate process."""
    import asyncio
    from slo_scheduler.config.settings import (
        SidecarConfig, ServerConfig, PredictorConfig, SchedulingConfig, PrefillScheduleMode,
    )
    from slo_scheduler.server.zmq_server import SLOSchedulerServer

    mode = PrefillScheduleMode(schedule_mode)
    config = SidecarConfig(
        worker_id="mp-test-server",
        server=ServerConfig(zmq_bind=addr),
        predictor=PredictorConfig(type="mode_aware", grid_path=grid_path),
        scheduling=SchedulingConfig(
            default_tpot_ms=50.0,
            max_prefill_tokens=4096,
            schedule_mode=mode,
        ),
    )
    server = SLOSchedulerServer(config)
    server.bind()

    # Signal parent that we're ready
    ready_event.set()

    loop = asyncio.new_event_loop()

    def handle_sigterm(*_):
        server.stop()

    signal.signal(signal.SIGTERM, handle_sigterm)

    loop.run_until_complete(server.serve())
    loop.close()


def _run_latency_bench(addr, schedule_mode, n_iterations=100, n_warmup=5):
    """Spawn sidecar in child process, run client, return latency stats dict."""
    sock_path = addr.replace("ipc://", "")
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    ready = multiprocessing.Event()
    proc = multiprocessing.Process(
        target=_sidecar_worker,
        args=(addr, ready, GRID_PATH, schedule_mode),
        daemon=True,
    )
    proc.start()

    try:
        assert ready.wait(timeout=10), "Sidecar process failed to start"
        time.sleep(0.1)

        client = SLOSchedulerClient(addr, timeout_ms=5000)
        try:
            # Warmup
            for i in range(n_warmup):
                state = _make_state(iteration=i + 1)
                client.send_and_recv(state, current_iteration=i + 2)

            # Measured iterations
            latencies_ms = []
            for i in range(n_iterations):
                iteration = i + 100
                state = _make_state(iteration=iteration)

                t0 = time.perf_counter()
                decision = client.send_and_recv(state, current_iteration=iteration + 1)
                t1 = time.perf_counter()

                assert decision is not None, f"No decision at iteration {iteration}"
                latencies_ms.append((t1 - t0) * 1000)

            latencies_ms.sort()
            stats = {
                "avg": sum(latencies_ms) / len(latencies_ms),
                "p50": latencies_ms[len(latencies_ms) // 2],
                "p95": latencies_ms[int(len(latencies_ms) * 0.95)],
                "p99": latencies_ms[int(len(latencies_ms) * 0.99)],
                "min": latencies_ms[0],
                "max": latencies_ms[-1],
            }
            return stats
        finally:
            client.close()
    finally:
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()


def _print_latency_stats(label, mode, stats, n_iterations=100):
    print(
        f"\n{'='*60}\n"
        f"  {label}\n"
        f"  Mode: {mode} | Transport: IPC | Iterations: {n_iterations}\n"
        f"{'='*60}\n"
        f"  avg  = {stats['avg']:.3f} ms\n"
        f"  p50  = {stats['p50']:.3f} ms\n"
        f"  p95  = {stats['p95']:.3f} ms\n"
        f"  p99  = {stats['p99']:.3f} ms\n"
        f"  min  = {stats['min']:.3f} ms\n"
        f"  max  = {stats['max']:.3f} ms\n"
        f"{'='*60}"
    )


class TestMultiProcessLatency:
    """Real multi-process test: sidecar in a child process, client in the test process."""

    def test_roundtrip_latency_predictor(self):
        """Measure roundtrip latency with predictor mode (binary search + grid lookup)."""
        stats = _run_latency_bench(MP_IPC_ADDR, "predictor")
        _print_latency_stats("Predictor mode roundtrip", "predictor", stats)
        assert stats["avg"] < 50, f"Average latency {stats['avg']:.1f}ms too high"

    def test_roundtrip_latency_greedy_kv(self):
        """Measure roundtrip latency with greedy_kv mode (simplest policy, no predictor)."""
        addr = "ipc:///tmp/sglang_slo_client_mp_greedy_test.sock"
        stats = _run_latency_bench(addr, "greedy_kv")
        _print_latency_stats("Greedy KV mode roundtrip", "greedy_kv", stats)
        assert stats["avg"] < 50, f"Average latency {stats['avg']:.1f}ms too high"

    def test_raw_zmq_echo_latency(self):
        """Measure raw ZMQ IPC echo latency (no sidecar logic, just transport + serde)."""
        import pickle
        echo_addr = "ipc:///tmp/sglang_slo_zmq_echo_test.sock"
        sock_path = echo_addr.replace("ipc://", "")
        if os.path.exists(sock_path):
            os.unlink(sock_path)

        ready = multiprocessing.Event()

        def echo_server(addr, ready_event):
            """Minimal echo: recv → deserialize → serialize → send."""
            ctx = zmq.Context()
            sock = ctx.socket(zmq.ROUTER)
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind(addr)
            ready_event.set()
            while True:
                try:
                    if not sock.poll(1000):
                        continue
                    frames = sock.recv_multipart()
                    identity = frames[0]
                    payload = frames[-1]
                    obj = pickle.loads(payload)
                    response = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
                    sock.send_multipart([identity, b"", response], flags=zmq.DONTWAIT)
                except Exception:
                    break

        proc = multiprocessing.Process(
            target=echo_server, args=(echo_addr, ready), daemon=True,
        )
        proc.start()

        try:
            assert ready.wait(timeout=5)
            time.sleep(0.05)

            ctx = zmq.Context()
            sock = ctx.socket(zmq.DEALER)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(echo_addr)

            state = _make_state(iteration=1)
            n = 100

            # Warmup
            for _ in range(5):
                payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
                sock.send_multipart([b"", payload])
                sock.poll(5000)
                sock.recv_multipart()

            # Measure
            latencies_us = []
            for _ in range(n):
                payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
                t0 = time.perf_counter()
                sock.send_multipart([b"", payload])
                sock.poll(5000)
                sock.recv_multipart()
                t1 = time.perf_counter()
                latencies_us.append((t1 - t0) * 1_000_000)

            sock.close()
            ctx.term()

            latencies_us.sort()
            avg = sum(latencies_us) / len(latencies_us)
            p50 = latencies_us[len(latencies_us) // 2]
            p95 = latencies_us[int(len(latencies_us) * 0.95)]
            p99 = latencies_us[int(len(latencies_us) * 0.99)]

            # Also measure just serialization cost
            t0 = time.perf_counter()
            for _ in range(n):
                b = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.loads(b)
            t1 = time.perf_counter()
            serde_us = (t1 - t0) / n * 1_000_000
            payload_kb = len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)) / 1024

            print(
                f"\n{'='*60}\n"
                f"  ZMQ IPC latency breakdown ({n} iterations)\n"
                f"{'='*60}\n"
                f"  Raw echo roundtrip (transport + serde both sides):\n"
                f"    avg  = {avg:.0f} us\n"
                f"    p50  = {p50:.0f} us\n"
                f"    p95  = {p95:.0f} us\n"
                f"    p99  = {p99:.0f} us\n"
                f"  Client-side serde (dumps+loads once):\n"
                f"    avg  = {serde_us:.0f} us\n"
                f"  Payload size: {payload_kb:.1f} KB\n"
                f"{'='*60}"
            )

            assert avg < 5000, f"Echo latency {avg:.0f}us too high"
        finally:
            proc.terminate()
            proc.join(timeout=3)
