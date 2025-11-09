"""Helpers for deriving iteration timing targets from TPOT SLO data."""

from __future__ import annotations


def compute_iteration_target(
    tpot_slo: float, avg_iteration_ms: float, min_target_ms: float = 1.0
) -> float:
    """Return the next-iteration target based on TPOT and rolling average."""
    diff = avg_iteration_ms - tpot_slo
    target = tpot_slo - 2.0 * diff
    return max(min_target_ms, target)
