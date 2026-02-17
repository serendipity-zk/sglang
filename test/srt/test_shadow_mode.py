"""Unit tests for Phase 3 shadow mode decision logging.

Tests the shadow logging methods in SchedulerSidecarMixin and the
external analyzer script. Uses mock objects — no GPU needed.
"""

import json
import os
import tempfile
import types
import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.scheduler_sidecar_mixin import (
    SchedulerSidecarMixin,
    _InternalDecisionSnapshot,
)

# Save original cwd at import time to restore after tests that chdir
_ORIGINAL_CWD = os.getcwd()


def _bind_shadow_methods(mock):
    """Bind real mixin shadow methods to a MagicMock."""
    for name in (
        "_shadow_open_log",
        "_shadow_capture_decode_only",
        "_shadow_capture_pre_batch",
        "_shadow_capture_post_batch",
        "_shadow_log_decisions",
    ):
        method = getattr(SchedulerSidecarMixin, name)
        setattr(mock, name, types.MethodType(method, mock))


def _make_mock_scheduler(mode="shadow", worker_id="w0"):
    """Create a mock Scheduler with shadow mode attributes and bound methods."""
    mock = MagicMock()
    mock.slo_scheduler_mode = mode
    mock.worker_id = worker_id
    mock.iteration_count = 42

    # Shadow state from init_sidecar
    mock._shadow_log_file = None
    mock._shadow_internal_snapshot = None
    mock._last_sidecar_decision = None

    _bind_shadow_methods(mock)
    return mock


def _make_mock_sidecar_decision(iteration_count=43):
    """Create a mock SchedulingDecision."""
    decision = MagicMock()
    decision.iteration_count = iteration_count
    decision.decode_only_iteration = False
    decision.max_prefill_tokens = 1200
    decision.target_iteration_time_ms = 14.8
    decision.min_decode_slack_ms = 45.2
    decision.predicted_iteration_time_ms = 12.1
    decision.mode_used = "predictor"
    decision.scheduling_reason = "binary_search_budget"
    decision.prefill_chunk_budget = 0
    decision.execution_flow = []
    decision.simulation_success = True
    return decision


class _ShadowTestBase(unittest.TestCase):
    """Base class that manages tmpdir and cwd for shadow tests."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._saved_cwd = os.getcwd()
        os.chdir(self._tmpdir)

    def tearDown(self):
        os.chdir(self._saved_cwd)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @property
    def log_path(self):
        return os.path.join(self._tmpdir, "shadow_decisions_w0.jsonl")


class TestShadowLogWritesJSONL(_ShadowTestBase):
    """Verify JSONL line is written with correct structure."""

    def test_shadow_log_writes_jsonl(self):
        mock = _make_mock_scheduler()
        mock._last_sidecar_decision = _make_mock_sidecar_decision()
        mock._shadow_internal_snapshot = _InternalDecisionSnapshot(
            decode_only=False,
            effective_target=15.3,
            mode="predictor",
            actual_prefill_tokens=1024,
            num_requests_admitted=3,
        )

        mock._shadow_log_decisions()

        self.assertTrue(os.path.exists(self.log_path))

        with open(self.log_path) as f:
            record = json.loads(f.readline())

        self.assertIn("ts_ms", record)
        self.assertEqual(record["iteration"], 42)
        self.assertIsNotNone(record["internal"])
        self.assertIsNotNone(record["sidecar"])
        self.assertEqual(record["internal"]["decode_only"], False)
        self.assertEqual(record["internal"]["mode"], "predictor")
        self.assertEqual(record["internal"]["actual_prefill_tokens"], 1024)
        self.assertEqual(record["internal"]["num_requests_admitted"], 3)
        self.assertEqual(record["sidecar"]["max_prefill_tokens"], 1200)
        self.assertEqual(record["sidecar"]["mode"], "predictor")


class TestShadowDecodeOnlyCaptured(unittest.TestCase):
    """Verify decode-only snapshot sets fields correctly."""

    def test_shadow_decode_only_captured(self):
        mock = _make_mock_scheduler()
        mock._shadow_capture_decode_only("simulation")

        snapshot = mock._shadow_internal_snapshot
        self.assertIsInstance(snapshot, _InternalDecisionSnapshot)
        self.assertTrue(snapshot.decode_only)
        self.assertIsNone(snapshot.effective_target)
        self.assertEqual(snapshot.mode, "simulation")
        self.assertEqual(snapshot.actual_prefill_tokens, 0)
        self.assertEqual(snapshot.num_requests_admitted, 0)


class TestShadowPreAndPostBatchCaptured(unittest.TestCase):
    """Verify prefill snapshot updates with actual tokens."""

    def test_shadow_pre_and_post_batch_captured(self):
        mock = _make_mock_scheduler()
        mock._shadow_capture_pre_batch(
            effective_target=15.3, mode="slack", min_slack=45.2
        )

        snapshot = mock._shadow_internal_snapshot
        self.assertFalse(snapshot.decode_only)
        self.assertEqual(snapshot.effective_target, 15.3)
        self.assertEqual(snapshot.mode, "slack")
        self.assertEqual(snapshot.min_decode_slack_ms, 45.2)

        # Post-batch update
        mock._shadow_capture_post_batch(1024, 3)
        self.assertEqual(snapshot.actual_prefill_tokens, 1024)
        self.assertEqual(snapshot.num_requests_admitted, 3)


class TestShadowNullSidecarLogged(_ShadowTestBase):
    """Sidecar=None writes 'sidecar': null."""

    def test_shadow_null_sidecar_logged(self):
        mock = _make_mock_scheduler()
        mock._last_sidecar_decision = None
        mock._shadow_internal_snapshot = _InternalDecisionSnapshot(
            decode_only=True, effective_target=None, mode="predictor",
        )

        mock._shadow_log_decisions()

        with open(self.log_path) as f:
            record = json.loads(f.readline())

        self.assertIsNotNone(record["internal"])
        self.assertIsNone(record["sidecar"])


class TestShadowNullInternalLogged(_ShadowTestBase):
    """Resource constraint writes 'internal': null."""

    def test_shadow_null_internal_logged(self):
        mock = _make_mock_scheduler()
        mock._last_sidecar_decision = _make_mock_sidecar_decision()
        mock._shadow_internal_snapshot = None

        mock._shadow_log_decisions()

        with open(self.log_path) as f:
            record = json.loads(f.readline())

        self.assertIsNone(record["internal"])
        self.assertIsNotNone(record["sidecar"])


class TestShadowNoLogInInternalMode(_ShadowTestBase):
    """Verify no file I/O when mode is 'internal'."""

    def test_shadow_no_log_in_internal_mode(self):
        mock = _make_mock_scheduler(mode="internal")
        mock._shadow_internal_snapshot = _InternalDecisionSnapshot(
            decode_only=False, effective_target=15.0, mode="budget",
        )

        mock._shadow_log_decisions()

        # No log file should be created
        self.assertFalse(os.path.exists(self.log_path))


class TestShadowBothNullSkipped(_ShadowTestBase):
    """No log written when both internal and sidecar are None."""

    def test_both_null_skipped(self):
        mock = _make_mock_scheduler()
        mock._shadow_internal_snapshot = None
        mock._last_sidecar_decision = None

        mock._shadow_log_decisions()

        # Log file should not be created since nothing to log
        self.assertFalse(os.path.exists(self.log_path))


class TestShadowSnapshotClearedAfterLog(_ShadowTestBase):
    """Verify _shadow_internal_snapshot is cleared after logging."""

    def test_snapshot_cleared_after_log(self):
        mock = _make_mock_scheduler()
        mock._last_sidecar_decision = _make_mock_sidecar_decision()
        mock._shadow_internal_snapshot = _InternalDecisionSnapshot(
            decode_only=False, effective_target=15.0, mode="budget",
        )

        mock._shadow_log_decisions()
        self.assertIsNone(mock._shadow_internal_snapshot)


class TestAnalyzerBasicReport(unittest.TestCase):
    """Feed a small JSONL file, verify report has expected sections."""

    def test_analyzer_basic_report(self):
        from slo_scheduler.scripts.analyze_shadow_log import analyze

        records = [
            {
                "ts_ms": 1707000000000.0,
                "iteration": i,
                "internal": {
                    "decode_only": False,
                    "effective_target": 15.0,
                    "mode": "predictor",
                    "min_decode_slack_ms": None,
                    "actual_prefill_tokens": 1024,
                    "num_requests_admitted": 3,
                },
                "sidecar": {
                    "iteration_count": i + 1,
                    "decode_only": False,
                    "max_prefill_tokens": 1200,
                    "target_iteration_time_ms": 14.8,
                    "min_decode_slack_ms": None,
                    "predicted_iteration_time_ms": 12.1,
                    "mode": "predictor",
                    "scheduling_reason": "binary_search",
                    "prefill_chunk_budget": 0,
                    "execution_flow": [],
                    "simulation_success": True,
                },
            }
            for i in range(10)
        ]

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            analyze(records)

        output = buf.getvalue()
        self.assertIn("Shadow Mode Analysis", output)
        self.assertIn("Total records: 10", output)
        self.assertIn("Both present: 10", output)
        self.assertIn("Decode-only agreement", output)
        self.assertIn("Mode agreement", output)
        self.assertIn("Prefill tokens", output)


class TestAnalyzerToleranceComputation(unittest.TestCase):
    """Verify float tolerance logic."""

    def test_tolerance(self):
        from slo_scheduler.scripts.analyze_shadow_log import within_tolerance

        # Exact match
        self.assertTrue(within_tolerance(10.0, 10.0))
        # Within absolute tolerance
        self.assertTrue(within_tolerance(10.0, 10.5))
        # Within relative tolerance (5% of 100 = 5)
        self.assertTrue(within_tolerance(100.0, 104.0))
        # Outside both tolerances
        self.assertFalse(within_tolerance(10.0, 20.0))
        # None handling
        self.assertTrue(within_tolerance(None, None))
        self.assertFalse(within_tolerance(10.0, None))
        self.assertFalse(within_tolerance(None, 10.0))


if __name__ == "__main__":
    unittest.main()
