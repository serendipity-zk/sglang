import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)


class _FakeEvent:
    def __init__(self, elapsed_ms: float | None = None):
        self.elapsed_ms = elapsed_ms
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1

    def elapsed_time(self, other):
        return float(other.elapsed_ms)


class _FakeProcessor(SchedulerOutputProcessorMixin):
    pass


class TestSchedulerGpuTiming(unittest.TestCase):
    def test_finalize_result_gpu_elapsed_ms_uses_events(self):
        processor = _FakeProcessor()
        result = SimpleNamespace(
            gpu_elapsed_ms=None,
            compute_start_event=_FakeEvent(),
            compute_end_event=_FakeEvent(elapsed_ms=12.5),
        )

        measured = processor._finalize_result_gpu_elapsed_ms(result)

        self.assertEqual(measured, 12.5)
        self.assertEqual(result.gpu_elapsed_ms, 12.5)
        self.assertEqual(result.compute_end_event.synchronize_calls, 1)

    def test_resolve_completed_iteration_time_ms_falls_back_to_host_time(self):
        processor = _FakeProcessor()
        result = SimpleNamespace(
            gpu_elapsed_ms=None,
            compute_start_event=None,
            compute_end_event=None,
        )

        measured = processor._resolve_completed_iteration_time_ms(
            result, fallback_time_ms=7.25
        )

        self.assertEqual(measured, 7.25)
