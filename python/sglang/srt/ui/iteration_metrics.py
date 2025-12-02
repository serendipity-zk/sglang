"""
Unified metrics state and reporting for scheduler and UI.

This module serves as the single source of truth for:
- Iteration-level metrics (batch size, KV cache, timing, queue status)
- UI display metrics (latest iteration data for dashboard)
- Prefill chunking metrics (tracking chunk pairs for prefill requests)

Supports dual output:
1. Log to standard logger (redirected to log file with other logs)
2. HTTP POST to router (optional, configured via router_metrics_url)
3. In-memory snapshot for UI endpoint (/ui_stats)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class _IterationMetricsState:
    """Thread-safe state for iteration metrics with dual output and UI snapshot."""

    def __init__(
        self,
        worker_id: str,
        router_url: Optional[str] = None,
    ):
        self.worker_id = worker_id
        self.router_url = router_url

        self._lock = threading.Lock()
        self._accepted_requests = 0  # Cumulative counter for accepted requests
        self._http_session = None
        self._http_executor = None

        # UI snapshot - latest metrics for dashboard
        self._latest_metrics: Dict = {}

        # Log metrics to standard logger (will be redirected to log file)
        logger.info("Iteration metrics will be logged to standard output")

        # Initialize HTTP client if URL provided
        if self.router_url:
            self._http_session = requests.Session()
            self._http_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="metrics-reporter"
            )
            logger.info(f"Iteration metrics will be reported to: {self.router_url}")

    def inc_accepted_requests(self, n: int = 1) -> None:
        """Increment the accepted requests counter."""
        if n <= 0:
            return
        with self._lock:
            self._accepted_requests += n

    def report_iteration(
        self,
        metrics: Dict,
        iteration_num: int,
        destinations: Optional[List[str]] = None,
    ) -> None:
        """Report metrics for one iteration (non-blocking).

        Args:
            metrics: Dictionary of metrics to report
            iteration_num: The iteration number (provided by scheduler)
            destinations: List of destinations - ["log", "ui", "router", "debug"]
                         Default: ["log", "ui", "router"] (all three)
        """
        if destinations is None:
            destinations = ["log", "ui", "router"]

        with self._lock:
            # Always update UI snapshot if requested
            if "ui" in destinations:
                self._latest_metrics = metrics.copy()
                self._latest_metrics["worker_id"] = self.worker_id
                self._latest_metrics["timestamp"] = time.time()
                self._latest_metrics["iteration_num"] = iteration_num
                self._latest_metrics["accepted_requests"] = self._accepted_requests

            # Add worker ID, timestamp, and accepted_requests to metrics
            metrics = metrics.copy()  # Don't modify caller's dict
            metrics["worker_id"] = self.worker_id
            metrics["timestamp"] = time.time()
            metrics["iteration_num"] = iteration_num
            metrics["accepted_requests"] = self._accepted_requests
            
            # Log to standard logger (will be redirected to log file)
            if "log" in destinations:
                min_decode_slack = metrics.get("min_decode_slack_ms")
                min_decode_slack_rid = metrics.get("min_decode_slack_rid")
                kv_usage_pct = metrics.get("kv_usage_pct")
                tpot_ms = metrics.get("tpot_ms")
                min_decode_slack_str = (
                    f"{min_decode_slack:.2f}ms"
                    if isinstance(min_decode_slack, (int, float))
                    else "n/a"
                )
                kv_usage_str = (
                    f"{kv_usage_pct:.2f}%"
                    if isinstance(kv_usage_pct, (int, float))
                    else "n/a"
                )
                kv_forecast_peak = metrics.get("kv_forecast_peak")
                kv_forecast_slack = metrics.get("kv_forecast_slack_ms")
                kv_forecast_peak_gt = metrics.get("kv_forecast_peak_gt")
                kv_forecast_slack_gt = metrics.get("kv_forecast_slack_ms_gt")
                kvf_pred_section = (
                    f"kvf:{kv_forecast_peak:.0f}/{kv_forecast_slack:.1f}ms"
                    if kv_forecast_peak is not None and kv_forecast_slack is not None
                    else "kvf:n/a"
                ).ljust(24)
                kvf_gt_section = (
                    f"gt:{kv_forecast_peak_gt:.0f}/{kv_forecast_slack_gt:.1f}ms"
                    if kv_forecast_peak_gt is not None and kv_forecast_slack_gt is not None
                    else "gt:n/a"
                ).ljust(18)
                tpot_str = (
                    f"tpot:{tpot_ms:.0f}ms" if isinstance(tpot_ms, (int, float)) else "n/a"
                )
                iter_section = f"Iter:{iteration_num}".ljust(12)
                token_section = (
                    f"Token:{metrics.get('prefill_tokens', 0)}P+"
                    f"{metrics.get('decode_tokens', 0)}D="
                    f"{metrics.get('token_batch_size', 0)}"
                ).ljust(22)
                req_section = (
                    f"Req:{metrics.get('num_requests', 0)}R+"
                    f"{metrics.get('queue_reqs', 0)}W"
                ).ljust(15)
                kv_section = (
                    f"KV:{metrics.get('kv_tokens_used', 0)} ({kv_usage_str})"
                ).ljust(22)
                slack_suffix = (
                    f" ({min_decode_slack_rid})" if min_decode_slack_rid else ""
                )
                slack_section = f"Slack:{min_decode_slack_str}{slack_suffix}".ljust(30)
                tpot_section = f"tpot:{tpot_str}".ljust(10)
                prefill_section = f"prefill:{metrics.get('prefill_chunk_pairs', [])}"
                log_line = (
                    f"{iter_section}| "
                    f"{tpot_section}| "
                    f"{token_section}| "
                    f"{req_section}| "
                    f"{kv_section}| "
                    f"{slack_section}| "
                    f"{kvf_pred_section}| "
                    f"{kvf_gt_section}| "
                    f"{prefill_section}"
                )
                logger.info(f"STAT_METRICS: {log_line}")

            # Debug log with different prefix
            if "debug" in destinations:
                json_line = json.dumps(metrics, separators=(",", ":"))
                logger.info(f"DEBUG_METRICS: {json_line}")

            # Send to router (asynchronous, non-blocking)
            if "router" in destinations and self.router_url and self._http_executor:
                self._http_executor.submit(self._send_to_router, metrics)

    def _send_to_router(self, metrics: Dict) -> None:
        """Send metrics to router via HTTP POST (runs in background thread)."""
        try:
            response = self._http_session.post(
                f"{self.router_url}/worker_stats",
                json=metrics,
                timeout=0.5,  # Fast timeout to avoid blocking
            )
            if not response.ok:
                logger.info(f"Router metrics HTTP error: {response.status_code}")
        except Exception as e:
            logger.info(f"Failed to send metrics to router: {e}")

    def get_ui_snapshot(self) -> Dict:
        """Get latest metrics snapshot for UI display."""
        with self._lock:
            return self._latest_metrics.copy()

    def close(self) -> None:
        """Clean shutdown of executors."""
        with self._lock:
            if self._http_executor:
                self._http_executor.shutdown(wait=False)


# Module-level singleton (similar to server_ui.py pattern)
_state: Optional[_IterationMetricsState] = None
_state_lock = threading.Lock()


def initialize(
    worker_id: str,
    router_url: Optional[str] = None,
) -> None:
    """Initialize the iteration metrics state (called once from scheduler __init__)."""
    global _state

    with _state_lock:
        if _state is not None:
            logger.warning("Iteration metrics already initialized")
            return
        logger.info("Initializing iteration metrics")
        _state = _IterationMetricsState(
            worker_id=worker_id,
            router_url=router_url,
        )


def inc_accepted_requests(n: int = 1) -> None:
    """Increment the accepted requests counter."""
    if _state is not None:
        _state.inc_accepted_requests(n)


def report_iteration(
    metrics: Dict,
    iteration_num: int,
    destinations: Optional[List[str]] = None,
) -> None:
    """Report metrics for one iteration.

    Args:
        metrics: Dictionary of metrics to report
        iteration_num: The iteration number (provided by scheduler)
        destinations: List of destinations - ["log", "ui", "router", "debug"]
                     Default: ["log", "ui", "router"] (all three)
    """
    if _state is not None:
        _state.report_iteration(metrics, iteration_num, destinations)


def get_ui_snapshot() -> Dict:
    """Get latest metrics snapshot for UI display."""
    if _state is not None:
        return _state.get_ui_snapshot()
    return {}


def shutdown() -> None:
    """Shutdown the metrics reporter."""
    global _state
    with _state_lock:
        if _state is not None:
            _state.close()
            _state = None
