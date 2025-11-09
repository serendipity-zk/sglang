import unittest

from sglang.srt.managers.iteration_target import compute_iteration_target


class TestSchedulerIterationTarget(unittest.TestCase):
    def test_target_reduces_when_average_above_tpot(self):
        target = compute_iteration_target(tpot_slo=100.0, avg_iteration_ms=140.0)
        self.assertAlmostEqual(target, 20.0)

    def test_target_increases_when_average_below_tpot(self):
        target = compute_iteration_target(tpot_slo=120.0, avg_iteration_ms=90.0)
        self.assertAlmostEqual(target, 180.0)

    def test_target_clamped_to_minimum(self):
        target = compute_iteration_target(tpot_slo=50.0, avg_iteration_ms=500.0)
        self.assertAlmostEqual(target, 1.0)


if __name__ == "__main__":
    unittest.main()
