"""Unit tests for the VibeSim alignment record builders.

These records are the contract the alignment report reads, and it reads the
same one from the vLLM fork, so the field names and the guard behaviour are
pinned here rather than left to a live run to discover.
"""

import unittest

from sglang.srt.observability.vibesim_alignment import (
    API_REQUEST_TIMING_SCHEMA_VERSION,
    INPUT_ADAPTER,
    ITERATION_SCHEMA_VERSION,
    REQUEST_TIMING_SCHEMA_VERSION,
    build_api_request_timing_record,
    build_expert_load_record,
    build_iteration_record,
    build_request_timing_record,
    capture_iteration_geometry,
)
from sglang.test.ci.ci_register import register_cpu_ci

# Pure arithmetic over plain numbers: no GPU, no scheduler, no server.
register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class TestIterationRecord(unittest.TestCase):
    def test_extend_batch_keeps_the_prefix_and_extend_split(self):
        geometry = capture_iteration_geometry(
            launch_monotonic_ns=1_000,
            is_extend=True,
            prefix_lens=[0, 4096],
            extend_lens=[512, 128],
            seq_lens=[512, 4224],
        )

        # The pair is what attention cost depends on; the sum would lose the
        # distinction between a fresh 512-token prefill and a 128-token chunk
        # appended onto 4096 tokens of cached context.
        self.assertEqual(geometry.prefill_chunk_pairs, [[0, 512], [4096, 128]])
        # An extend batch reports no decode lengths even though seq_lens exists.
        self.assertEqual(geometry.decode_kv_lens, [])

    def test_extend_batch_drops_requests_scheduled_zero_tokens(self):
        geometry = capture_iteration_geometry(
            launch_monotonic_ns=0,
            is_extend=True,
            prefix_lens=[0, 100, 200],
            extend_lens=[16, 0, 32],
            seq_lens=None,
        )

        self.assertEqual(geometry.prefill_chunk_pairs, [[0, 16], [200, 32]])

    def test_decode_batch_reports_context_lengths_only(self):
        geometry = capture_iteration_geometry(
            launch_monotonic_ns=0,
            is_extend=False,
            prefix_lens=None,
            extend_lens=None,
            seq_lens=[128, 4096],
        )

        self.assertEqual(geometry.prefill_chunk_pairs, [])
        self.assertEqual(geometry.decode_kv_lens, [128, 4096])

    def test_record_totals_are_derived_not_passed_in(self):
        record = build_iteration_record(
            iteration_index=42,
            observed_start_monotonic_ns=1_000_000,
            observed_end_monotonic_ns=4_000_000,
            prefill_chunk_pairs=[[0, 512], [4096, 128]],
            decode_kv_lens=[7, 9, 11],
        )

        self.assertEqual(record["schema_version"], ITERATION_SCHEMA_VERSION)
        self.assertEqual(record["input_adapter"], INPUT_ADAPTER)
        self.assertEqual(record["iteration_index"], 42)
        self.assertEqual(record["prefill_tokens"], 640)
        self.assertEqual(record["decode_requests"], 3)
        self.assertEqual(record["decode_tokens_scheduled"], 3)
        self.assertAlmostEqual(record["observed_elapsed_ms"], 3.0)


class TestRequestTimingRecord(unittest.TestCase):
    def _record(self, **overrides):
        arguments = dict(
            request_id="rid-1",
            queued_monotonic=100.0,
            scheduled_monotonic=100.010,
            first_token_monotonic=100.025,
            last_token_monotonic=100.225,
            num_output_tokens=21,
        )
        arguments.update(overrides)
        return build_request_timing_record(**arguments)

    def test_ttft_splits_into_queue_wait_and_first_schedule_to_first_token(self):
        record = self._record()

        self.assertEqual(record["schema_version"], REQUEST_TIMING_SCHEMA_VERSION)
        self.assertAlmostEqual(record["engine_queue_wait_ms"], 10.0)
        self.assertAlmostEqual(record["engine_first_schedule_to_first_token_ms"], 15.0)
        # The two parts have to add back up to the whole.
        self.assertAlmostEqual(record["engine_core_ttft_ms"], 25.0)
        self.assertAlmostEqual(record["engine_core_decode_ms"], 200.0)
        self.assertAlmostEqual(record["engine_core_tpot_ms"], 10.0)

    def test_single_token_request_has_no_tpot_sample(self):
        record = self._record(num_output_tokens=1)

        # Null, not zero: there is no inter-token interval to have measured.
        self.assertIsNone(record["engine_core_tpot_ms"])

    def test_missing_or_reordered_timestamps_produce_no_record(self):
        # A request aborted before it was scheduled never stamped a boundary.
        self.assertIsNone(self._record(scheduled_monotonic=0.0))
        self.assertIsNone(self._record(num_output_tokens=0))
        # First token cannot precede the schedule that produced it.
        self.assertIsNone(self._record(first_token_monotonic=100.005))


class TestApiRequestTimingRecord(unittest.TestCase):
    def _record(self, **overrides):
        arguments = dict(
            request_id="rid-1",
            created_monotonic=100.0,
            tokenize_finish_monotonic=100.003,
            dispatch_monotonic=100.005,
            dispatch_finish_monotonic=100.007,
            first_token_monotonic=100.020,
            last_token_monotonic=100.220,
            finished_monotonic=100.221,
            response_sent_monotonic=100.222,
            output_tokens=32,
        )
        arguments.update(overrides)
        return build_api_request_timing_record(**arguments)

    def test_frontend_span_decomposes_into_adjacent_segments(self):
        record = self._record()

        self.assertEqual(record["schema_version"], API_REQUEST_TIMING_SCHEMA_VERSION)
        self.assertAlmostEqual(record["api_frontend_prepare_ms"], 3.0)
        self.assertAlmostEqual(record["api_stream_activation_ms"], 2.0)
        self.assertAlmostEqual(record["api_add_request_ms"], 2.0)
        self.assertAlmostEqual(record["api_first_output_wait_ms"], 13.0)
        self.assertAlmostEqual(record["api_token_output_receive_span_ms"], 200.0)
        self.assertAlmostEqual(record["api_terminal_tail_ms"], 1.0)
        self.assertAlmostEqual(record["api_e2e_ms"], 221.0)
        # The segments up to the first token must tile that whole span.
        self.assertAlmostEqual(
            record["api_frontend_prepare_ms"]
            + record["api_stream_activation_ms"]
            + record["api_add_request_ms"]
            + record["api_first_output_wait_ms"],
            20.0,
        )

    def test_non_streaming_request_omits_the_response_sent_segment(self):
        record = self._record(response_sent_monotonic=0.0)

        # Absent rather than zero: the non-streaming path never stamps it, and a
        # zero would read as "sent instantly".
        self.assertNotIn("api_response_sent_ms", record)

    def test_incomplete_timestamps_produce_no_record(self):
        self.assertIsNone(self._record(dispatch_monotonic=0.0))
        self.assertIsNone(self._record(first_token_monotonic=99.0))


class TestExpertLoadRecord(unittest.TestCase):
    def test_counts_are_kept_per_layer(self):
        record = build_expert_load_record(
            model="deepseek-ai/DeepSeek-V3",
            eplb_step=7,
            expert_parallel_size=8,
            experts_per_token=8,
            logical_expert_counts=[[1, 2, 3], [4, 5, 6]],
        )

        self.assertEqual(record["input_adapter"], INPUT_ADAPTER)
        self.assertEqual(record["logical_expert_counts"], [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(record["experts_per_token"], 8)


if __name__ == "__main__":
    unittest.main()
