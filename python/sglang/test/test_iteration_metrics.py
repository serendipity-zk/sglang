"""
Unit tests for iteration metrics module.

Tests the dual-output metrics system (file + HTTP) for reporting
scheduler iteration statistics.
"""

import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List

from sglang.srt.ui import iteration_metrics


class MockRouterHandler(BaseHTTPRequestHandler):
    """Mock HTTP handler that simulates the router's /worker_stats endpoint."""

    received_stats: List[Dict] = []
    lock = threading.Lock()

    def do_POST(self):
        if self.path == "/worker_stats":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            stats = json.loads(post_data.decode("utf-8"))

            with self.lock:
                MockRouterHandler.received_stats.append(stats)

            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Stats received")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress HTTP logs during testing
        pass


class TestIterationMetrics(unittest.TestCase):
    """Test suite for iteration_metrics module."""

    @classmethod
    def setUpClass(cls):
        """Start a mock router HTTP server."""
        cls.mock_server = HTTPServer(("127.0.0.1", 0), MockRouterHandler)
        cls.router_port = cls.mock_server.server_address[1]
        cls.router_url = f"http://127.0.0.1:{cls.router_port}"

        # Start server in background thread
        cls.server_thread = threading.Thread(
            target=cls.mock_server.serve_forever, daemon=True
        )
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        """Shutdown the mock router."""
        cls.mock_server.shutdown()
        cls.server_thread.join(timeout=2)

    def setUp(self):
        """Reset state before each test."""
        MockRouterHandler.received_stats.clear()
        # Shutdown any existing state
        iteration_metrics.shutdown()

    def tearDown(self):
        """Clean up after each test."""
        iteration_metrics.shutdown()

    def test_file_logging_only(self):
        """Test metrics logging to file without HTTP."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".stat", delete=False
        ) as stat_file:
            stat_file_path = stat_file.name

        try:
            # Initialize with file logging only
            iteration_metrics.initialize(
                worker_id="test_worker_1",
                stat_file_path=stat_file_path,
                router_url=None,
                report_interval=1,
            )

            # Report some metrics
            metrics = {
                "batch_size_tokens": 512,
                "num_requests": 4,
                "kv_cache_used_tokens": 1024,
                "iteration_time_ms": 15.5,
            }
            iteration_metrics.report_iteration(metrics)

            # Give file I/O time to complete
            time.sleep(0.1)

            # Verify file contents
            with open(stat_file_path, "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 1)

                logged_data = json.loads(lines[0])
                self.assertEqual(logged_data["worker_id"], "test_worker_1")
                self.assertEqual(logged_data["batch_size_tokens"], 512)
                self.assertEqual(logged_data["num_requests"], 4)
                self.assertEqual(logged_data["iteration_num"], 1)
                self.assertIn("timestamp", logged_data)

        finally:
            if os.path.exists(stat_file_path):
                os.remove(stat_file_path)

    def test_http_logging_only(self):
        """Test metrics reporting via HTTP without file."""
        # Initialize with HTTP logging only
        iteration_metrics.initialize(
            worker_id="test_worker_2",
            stat_file_path=None,
            router_url=self.router_url,
            report_interval=1,
        )

        # Report metrics
        metrics = {
            "batch_size_tokens": 256,
            "num_requests": 2,
            "kv_cache_usage_pct": 0.75,
            "iteration_time_ms": 10.2,
        }
        iteration_metrics.report_iteration(metrics)

        # Wait for HTTP request to complete
        time.sleep(0.5)

        # Verify HTTP endpoint received the data
        with MockRouterHandler.lock:
            self.assertEqual(len(MockRouterHandler.received_stats), 1)
            received = MockRouterHandler.received_stats[0]
            self.assertEqual(received["worker_id"], "test_worker_2")
            self.assertEqual(received["batch_size_tokens"], 256)
            self.assertEqual(received["kv_cache_usage_pct"], 0.75)

    def test_dual_output(self):
        """Test metrics reported to both file and HTTP."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".stat", delete=False
        ) as stat_file:
            stat_file_path = stat_file.name

        try:
            # Initialize with both outputs
            iteration_metrics.initialize(
                worker_id="test_worker_3",
                stat_file_path=stat_file_path,
                router_url=self.router_url,
                report_interval=1,
            )

            # Report metrics
            metrics = {
                "batch_size_tokens": 1024,
                "num_requests": 8,
                "waiting_queue_size": 5,
                "forward_mode": "EXTEND",
            }
            iteration_metrics.report_iteration(metrics)

            # Wait for both outputs
            time.sleep(0.5)

            # Verify file
            with open(stat_file_path, "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 1)
                file_data = json.loads(lines[0])
                self.assertEqual(file_data["batch_size_tokens"], 1024)

            # Verify HTTP
            with MockRouterHandler.lock:
                self.assertEqual(len(MockRouterHandler.received_stats), 1)
                http_data = MockRouterHandler.received_stats[0]
                self.assertEqual(http_data["batch_size_tokens"], 1024)
                self.assertEqual(http_data["forward_mode"], "EXTEND")

        finally:
            if os.path.exists(stat_file_path):
                os.remove(stat_file_path)

    def test_report_interval(self):
        """Test that report_interval controls reporting frequency."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".stat", delete=False
        ) as stat_file:
            stat_file_path = stat_file.name

        try:
            # Initialize with interval of 3
            iteration_metrics.initialize(
                worker_id="test_worker_4",
                stat_file_path=stat_file_path,
                router_url=None,
                report_interval=3,
            )

            # Report 5 times
            for i in range(5):
                metrics = {"batch_size_tokens": 128 * (i + 1), "num_requests": i + 1}
                iteration_metrics.report_iteration(metrics)

            time.sleep(0.1)

            # Should only log at iterations 3 (only one that's a multiple of 3 in range 1-5)
            with open(stat_file_path, "r") as f:
                lines = f.readlines()
                # Iterations 3 is reported (1, 2, 4, 5 are skipped)
                self.assertEqual(len(lines), 1)
                data = json.loads(lines[0])
                self.assertEqual(data["iteration_num"], 3)
                self.assertEqual(data["batch_size_tokens"], 128 * 3)

        finally:
            if os.path.exists(stat_file_path):
                os.remove(stat_file_path)

    def test_multiple_iterations(self):
        """Test logging multiple iterations with incrementing iteration numbers."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".stat", delete=False
        ) as stat_file:
            stat_file_path = stat_file.name

        try:
            iteration_metrics.initialize(
                worker_id="test_worker_5",
                stat_file_path=stat_file_path,
                router_url=None,
                report_interval=1,
            )

            # Report 3 iterations
            for i in range(3):
                metrics = {
                    "batch_size_tokens": 100 + i * 50,
                    "iteration_time_ms": 10.0 + i,
                }
                iteration_metrics.report_iteration(metrics)

            time.sleep(0.1)

            # Verify all 3 logged with correct iteration numbers
            with open(stat_file_path, "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 3)

                for i, line in enumerate(lines):
                    data = json.loads(line)
                    self.assertEqual(data["iteration_num"], i + 1)
                    self.assertEqual(data["batch_size_tokens"], 100 + i * 50)

        finally:
            if os.path.exists(stat_file_path):
                os.remove(stat_file_path)

    def test_shutdown_cleanup(self):
        """Test that shutdown properly closes resources."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".stat", delete=False
        ) as stat_file:
            stat_file_path = stat_file.name

        try:
            iteration_metrics.initialize(
                worker_id="test_worker_6",
                stat_file_path=stat_file_path,
                router_url=self.router_url,
                report_interval=1,
            )

            # Report a metric
            metrics = {"batch_size_tokens": 200}
            iteration_metrics.report_iteration(metrics)
            time.sleep(0.1)

            # Shutdown
            iteration_metrics.shutdown()

            # Try reporting after shutdown (should be no-op)
            iteration_metrics.report_iteration({"batch_size_tokens": 300})
            time.sleep(0.1)

            # Verify only the first metric was logged
            with open(stat_file_path, "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 1)

        finally:
            if os.path.exists(stat_file_path):
                os.remove(stat_file_path)

    def test_http_timeout_handling(self):
        """Test that HTTP timeouts don't block reporting."""
        # Use an unreachable URL to simulate timeout
        unreachable_url = "http://127.0.0.1:1"  # Port 1 is typically unreachable

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".stat", delete=False
        ) as stat_file:
            stat_file_path = stat_file.name

        try:
            iteration_metrics.initialize(
                worker_id="test_worker_7",
                stat_file_path=stat_file_path,
                router_url=unreachable_url,
                report_interval=1,
            )

            # Report metrics (HTTP should timeout but file should still work)
            start_time = time.perf_counter()
            metrics = {"batch_size_tokens": 400, "num_requests": 3}
            iteration_metrics.report_iteration(metrics)
            elapsed = time.perf_counter() - start_time

            # Should return quickly (not blocking on HTTP timeout)
            self.assertLess(elapsed, 0.5)

            # File should still be written
            time.sleep(0.1)
            with open(stat_file_path, "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 1)
                data = json.loads(lines[0])
                self.assertEqual(data["batch_size_tokens"], 400)

        finally:
            if os.path.exists(stat_file_path):
                os.remove(stat_file_path)

    def test_worker_id_in_metrics(self):
        """Test that worker_id is properly added to all metrics."""
        iteration_metrics.initialize(
            worker_id="server1:8000:tp0:dp1",
            stat_file_path=None,
            router_url=self.router_url,
            report_interval=1,
        )

        metrics = {"batch_size_tokens": 512}
        iteration_metrics.report_iteration(metrics)

        time.sleep(0.5)

        with MockRouterHandler.lock:
            self.assertEqual(len(MockRouterHandler.received_stats), 1)
            received = MockRouterHandler.received_stats[0]
            self.assertEqual(received["worker_id"], "server1:8000:tp0:dp1")

    def test_timestamp_added(self):
        """Test that timestamp is automatically added to metrics."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".stat", delete=False
        ) as stat_file:
            stat_file_path = stat_file.name

        try:
            iteration_metrics.initialize(
                worker_id="test_worker_8",
                stat_file_path=stat_file_path,
                router_url=None,
                report_interval=1,
            )

            before_time = time.time()
            metrics = {"batch_size_tokens": 100}
            iteration_metrics.report_iteration(metrics)
            after_time = time.time()

            time.sleep(0.1)

            with open(stat_file_path, "r") as f:
                data = json.loads(f.readline())
                self.assertIn("timestamp", data)
                self.assertGreaterEqual(data["timestamp"], before_time)
                self.assertLessEqual(data["timestamp"], after_time)

        finally:
            if os.path.exists(stat_file_path):
                os.remove(stat_file_path)

    def test_ui_snapshot(self):
        """Test that UI snapshot is updated with latest metrics."""
        iteration_metrics.initialize(
            worker_id="test_worker_9",
            stat_file_path=None,
            router_url=None,
            report_interval=1,
        )

        # Report first metric
        metrics1 = {"batch_size_tokens": 100, "num_requests": 2}
        iteration_metrics.report_iteration(metrics1)

        # Get UI snapshot
        snapshot = iteration_metrics.get_ui_snapshot()
        self.assertEqual(snapshot["batch_size_tokens"], 100)
        self.assertEqual(snapshot["num_requests"], 2)
        self.assertEqual(snapshot["worker_id"], "test_worker_9")
        self.assertEqual(snapshot["iteration_num"], 1)

        # Report second metric
        metrics2 = {"batch_size_tokens": 200, "num_requests": 4}
        iteration_metrics.report_iteration(metrics2)

        # Verify UI snapshot updated to latest
        snapshot2 = iteration_metrics.get_ui_snapshot()
        self.assertEqual(snapshot2["batch_size_tokens"], 200)
        self.assertEqual(snapshot2["num_requests"], 4)
        self.assertEqual(snapshot2["iteration_num"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
