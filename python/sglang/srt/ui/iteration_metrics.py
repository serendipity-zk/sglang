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
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


class _IterationMetricsState:
    """Thread-safe state for iteration metrics with dual output and UI snapshot."""

    def __init__(
        self,
        worker_id: str,
        stat_file_path: Optional[str] = None,
        router_url: Optional[str] = None,
        report_interval: int = 1,
    ):
        self.worker_id = worker_id
        self.stat_file_path = stat_file_path
        self.router_url = router_url
        self.report_interval = report_interval

        self._lock = threading.Lock()
        self._iteration_count = 0
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

    def report_iteration(self, metrics: Dict) -> None:
        """Report metrics for one iteration (non-blocking)."""
        logger.info(f"Reporting metrics: {metrics}")

        with self._lock:
            self._iteration_count += 1

            # Always update UI snapshot with latest metrics
            self._latest_metrics = metrics.copy()
            self._latest_metrics["worker_id"] = self.worker_id
            self._latest_metrics["timestamp"] = time.time()
            self._latest_metrics["iteration_num"] = self._iteration_count
            self._latest_metrics["accepted_requests"] = self._accepted_requests

            # Check if we should report based on interval
            if self._iteration_count % self.report_interval != 0:
                return

            # Add worker ID, timestamp, and accepted_requests to metrics for logging
            metrics["worker_id"] = self.worker_id
            metrics["timestamp"] = time.time()
            metrics["iteration_num"] = self._iteration_count
            metrics["accepted_requests"] = self._accepted_requests

            # Log to standard logger (will be redirected to log file)
            # Use root logger to ensure it's always logged
            json_line = json.dumps(metrics, separators=(",", ":"))
            # Also log with module logger for compatibility
            logger.info(f"STAT_METRICS: {json_line}")

            # Send to router (asynchronous, non-blocking)
            if self.router_url and self._http_executor:
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
            else:
                logger.info(f"Send metrics to router: {metrics}")
        except requests.exceptions.Timeout:
            logger.info("Router metrics request timeout (non-critical)")
        except Exception as e:
            logger.info(f"Failed to send metrics to router: {e}")

    def get_ui_snapshot(self) -> Dict:
        """Get latest metrics snapshot for UI display."""
        logger.info("Getting UI snapshot inside 22332")
        with self._lock:
            logger.info(f"Get UI snapshot: {self._latest_metrics}")
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
    stat_file_path: Optional[str] = None,
    router_url: Optional[str] = None,
    report_interval: int = 1,
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
            stat_file_path=stat_file_path,
            router_url=router_url,
            report_interval=report_interval,
        )


def inc_accepted_requests(n: int = 1) -> None:
    """Increment the accepted requests counter."""
    if _state is not None:
        _state.inc_accepted_requests(n)


def report_iteration(metrics: Dict) -> None:
    """Report metrics for one iteration."""
    if _state is not None:
        _state.report_iteration(metrics)


def get_ui_snapshot() -> Dict:
    """Get latest metrics snapshot for UI display."""
    logger.info("Getting UI snapshot inside")
    if _state is not None:
        logger.info("Getting UI snapshot inside not none")
        return _state.get_ui_snapshot()
    return {}


def shutdown() -> None:
    """Shutdown the metrics reporter."""
    global _state
    with _state_lock:
        if _state is not None:
            _state.close()
            _state = None
