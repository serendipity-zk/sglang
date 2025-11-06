"""
Unit tests for iteration metrics module.

Tests the metrics system for reporting scheduler iteration statistics.
"""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
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

    def test_http_logging_only(self):
        """Test metrics reporting via HTTP without file."""
        # Initialize with HTTP logging only
        iteration_metrics.initialize(
            worker_id="test_worker_2",
            router_url=self.router_url,
        )

        # Report metrics
        metrics = {
            "batch_size_tokens": 256,
            "num_requests": 2,
            "kv_cache_usage_pct": 0.75,
            "iteration_time_ms": 10.2,
        }
        iteration_metrics.report_iteration(metrics, iteration_num=1)

        # Wait for HTTP request to complete
        time.sleep(0.5)

        # Verify HTTP endpoint received the data
        with MockRouterHandler.lock:
            self.assertEqual(len(MockRouterHandler.received_stats), 1)
            received = MockRouterHandler.received_stats[0]
            self.assertEqual(received["worker_id"], "test_worker_2")
            self.assertEqual(received["batch_size_tokens"], 256)
            self.assertEqual(received["kv_cache_usage_pct"], 0.75)
            self.assertEqual(received["iteration_num"], 1)

    def test_dual_output(self):
        """Test metrics reported to both log and HTTP."""
        # Initialize with both outputs
        iteration_metrics.initialize(
            worker_id="test_worker_3",
            router_url=self.router_url,
        )

        # Report metrics
        metrics = {
            "batch_size_tokens": 1024,
            "num_requests": 8,
            "waiting_queue_size": 5,
            "forward_mode": "EXTEND",
        }
        iteration_metrics.report_iteration(metrics, iteration_num=1)

        # Wait for both outputs
        time.sleep(0.5)

        # Verify HTTP
        with MockRouterHandler.lock:
            self.assertEqual(len(MockRouterHandler.received_stats), 1)
            http_data = MockRouterHandler.received_stats[0]
            self.assertEqual(http_data["batch_size_tokens"], 1024)
            self.assertEqual(http_data["forward_mode"], "EXTEND")
            self.assertEqual(http_data["iteration_num"], 1)

    def test_multiple_iterations(self):
        """Test logging multiple iterations with incrementing iteration numbers."""
        iteration_metrics.initialize(
            worker_id="test_worker_5",
            router_url=self.router_url,
        )

        # Report 3 iterations
        for i in range(3):
            metrics = {
                "batch_size_tokens": 100 + i * 50,
                "iteration_time_ms": 10.0 + i,
            }
            iteration_metrics.report_iteration(metrics, iteration_num=i + 1)

        time.sleep(0.5)

        # Verify all 3 reported with correct iteration numbers
        with MockRouterHandler.lock:
            self.assertEqual(len(MockRouterHandler.received_stats), 3)
            for i in range(3):
                data = MockRouterHandler.received_stats[i]
                self.assertEqual(data["iteration_num"], i + 1)
                self.assertEqual(data["batch_size_tokens"], 100 + i * 50)

    def test_shutdown_cleanup(self):
        """Test that shutdown properly closes resources."""
        iteration_metrics.initialize(
            worker_id="test_worker_6",
            router_url=self.router_url,
        )

        # Report a metric
        metrics = {"batch_size_tokens": 200}
        iteration_metrics.report_iteration(metrics, iteration_num=1)
        time.sleep(0.5)

        # Shutdown
        iteration_metrics.shutdown()

        # Try reporting after shutdown (should be no-op)
        iteration_metrics.report_iteration({"batch_size_tokens": 300}, iteration_num=2)
        time.sleep(0.5)

        # Verify only the first metric was reported
        with MockRouterHandler.lock:
            self.assertEqual(len(MockRouterHandler.received_stats), 1)

    def test_http_timeout_handling(self):
        """Test that HTTP timeouts don't block reporting."""
        # Use an unreachable URL to simulate timeout
        unreachable_url = "http://127.0.0.1:1"  # Port 1 is typically unreachable

        iteration_metrics.initialize(
            worker_id="test_worker_7",
            router_url=unreachable_url,
        )

        # Report metrics (HTTP should timeout but should return quickly)
        start_time = time.perf_counter()
        metrics = {"batch_size_tokens": 400, "num_requests": 3}
        iteration_metrics.report_iteration(metrics, iteration_num=1)
        elapsed = time.perf_counter() - start_time

        # Should return quickly (not blocking on HTTP timeout)
        self.assertLess(elapsed, 0.5)

    def test_worker_id_in_metrics(self):
        """Test that worker_id is properly added to all metrics."""
        iteration_metrics.initialize(
            worker_id="server1:8000:tp0:dp1",
            router_url=self.router_url,
        )

        metrics = {"batch_size_tokens": 512}
        iteration_metrics.report_iteration(metrics, iteration_num=1)

        time.sleep(0.5)

        with MockRouterHandler.lock:
            self.assertEqual(len(MockRouterHandler.received_stats), 1)
            received = MockRouterHandler.received_stats[0]
            self.assertEqual(received["worker_id"], "server1:8000:tp0:dp1")

    def test_timestamp_added(self):
        """Test that timestamp is automatically added to metrics."""
        iteration_metrics.initialize(
            worker_id="test_worker_8",
            router_url=self.router_url,
        )

        before_time = time.time()
        metrics = {"batch_size_tokens": 100}
        iteration_metrics.report_iteration(metrics, iteration_num=1)
        after_time = time.time()

        time.sleep(0.5)

        with MockRouterHandler.lock:
            self.assertEqual(len(MockRouterHandler.received_stats), 1)
            data = MockRouterHandler.received_stats[0]
            self.assertIn("timestamp", data)
            self.assertGreaterEqual(data["timestamp"], before_time)
            self.assertLessEqual(data["timestamp"], after_time)

    def test_ui_snapshot(self):
        """Test that UI snapshot is updated with latest metrics."""
        iteration_metrics.initialize(
            worker_id="test_worker_9",
            router_url=None,
        )

        # Report first metric
        metrics1 = {"batch_size_tokens": 100, "num_requests": 2}
        iteration_metrics.report_iteration(metrics1, iteration_num=1)

        # Get UI snapshot
        snapshot = iteration_metrics.get_ui_snapshot()
        self.assertEqual(snapshot["batch_size_tokens"], 100)
        self.assertEqual(snapshot["num_requests"], 2)
        self.assertEqual(snapshot["worker_id"], "test_worker_9")
        self.assertEqual(snapshot["iteration_num"], 1)

        # Report second metric
        metrics2 = {"batch_size_tokens": 200, "num_requests": 4}
        iteration_metrics.report_iteration(metrics2, iteration_num=2)

        # Verify UI snapshot updated to latest
        snapshot2 = iteration_metrics.get_ui_snapshot()
        self.assertEqual(snapshot2["batch_size_tokens"], 200)
        self.assertEqual(snapshot2["num_requests"], 4)
        self.assertEqual(snapshot2["iteration_num"], 2)

    def test_prefill_chunk_pairs_empty_list(self):
        """Test that prefill_chunk_pairs can be an empty list for decode-only batches."""
        iteration_metrics.initialize(
            worker_id="test_worker_10",
            router_url=self.router_url,
        )

        # Report metrics for decode-only batch (no prefill chunks)
        metrics = {
            "batch_size_tokens": 128,
            "num_requests": 8,
            "forward_mode": "DECODE",
            "prefill_chunk_pairs": [],  # Empty for decode-only
        }
        iteration_metrics.report_iteration(metrics, iteration_num=1)

        time.sleep(0.5)

        # Verify HTTP received the data with empty prefill_chunk_pairs
        with MockRouterHandler.lock:
            self.assertEqual(len(MockRouterHandler.received_stats), 1)
            received = MockRouterHandler.received_stats[0]
            self.assertEqual(received["forward_mode"], "DECODE")
            self.assertEqual(received["prefill_chunk_pairs"], [])

    def test_prefill_chunk_pairs_non_chunked(self):
        """Test prefill_chunk_pairs for non-chunked prefill requests."""
        iteration_metrics.initialize(
            worker_id="test_worker_11",
            router_url=self.router_url,
        )

        # Report metrics with non-chunked prefill (seqlen, seqlen)
        metrics = {
            "batch_size_tokens": 512,
            "num_requests": 2,
            "forward_mode": "EXTEND",
            "prefill_chunk_pairs": [[512, 512], [1024, 1024]],  # Two non-chunked requests
        }
        iteration_metrics.report_iteration(metrics, iteration_num=1)

        time.sleep(0.5)

        # Verify HTTP logging
        with MockRouterHandler.lock:
            self.assertEqual(len(MockRouterHandler.received_stats), 1)
            received = MockRouterHandler.received_stats[0]
            self.assertEqual(len(received["prefill_chunk_pairs"]), 2)
            self.assertEqual(received["prefill_chunk_pairs"][0], [512, 512])

    def test_prefill_chunk_pairs_chunked(self):
        """Test prefill_chunk_pairs for chunked prefill requests."""
        iteration_metrics.initialize(
            worker_id="test_worker_12",
            router_url=self.router_url,
        )

        # Report metrics with chunked prefill
        # Format: [current_chunk, cumulative_prefill]
        # Example: request with 3 chunks of 256 tokens each
        metrics = {
            "batch_size_tokens": 256,
            "num_requests": 1,
            "forward_mode": "EXTEND",
            "prefill_chunk_pairs": [[256, 768]],  # Processing 3rd chunk (256) of total 768 tokens
        }
        iteration_metrics.report_iteration(metrics, iteration_num=1)

        time.sleep(0.5)

        # Verify the chunked prefill information was captured
        with MockRouterHandler.lock:
            self.assertEqual(len(MockRouterHandler.received_stats), 1)
            received = MockRouterHandler.received_stats[0]
            self.assertEqual(received["forward_mode"], "EXTEND")
            self.assertEqual(len(received["prefill_chunk_pairs"]), 1)
            # Current chunk is 256, cumulative is 768
            self.assertEqual(received["prefill_chunk_pairs"][0][0], 256)
            self.assertEqual(received["prefill_chunk_pairs"][0][1], 768)

    def test_prefill_chunk_pairs_mixed_mode(self):
        """Test prefill_chunk_pairs in mixed mode (prefill + decode)."""
        iteration_metrics.initialize(
            worker_id="test_worker_13",
            router_url=self.router_url,
        )

        # Report metrics in MIXED mode with multiple prefill requests
        metrics = {
            "batch_size_tokens": 1024,
            "num_requests": 5,  # 3 prefill + 2 decode
            "forward_mode": "MIXED",
            "prefill_chunk_pairs": [
                [512, 512],    # Non-chunked prefill
                [256, 768],    # Chunked prefill (3rd chunk)
                [128, 128],    # Non-chunked prefill
            ],  # Decode requests have no entries
        }
        iteration_metrics.report_iteration(metrics, iteration_num=1)

        time.sleep(0.5)

        # Verify mixed mode data
        with MockRouterHandler.lock:
            self.assertEqual(len(MockRouterHandler.received_stats), 1)
            received = MockRouterHandler.received_stats[0]
            self.assertEqual(received["forward_mode"], "MIXED")
            self.assertEqual(received["num_requests"], 5)
            # Should have 3 prefill chunk pairs (no entries for decode requests)
            self.assertEqual(len(received["prefill_chunk_pairs"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
