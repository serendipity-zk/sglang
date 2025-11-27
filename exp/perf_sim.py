import argparse
import glob
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.length_sim_algo import estimate_peak_memory_fast


def _parse_int_list(raw: str) -> List[int]:
    return [int(part) for part in raw.split(",") if part]


def build_bins(values: np.ndarray, bin_count: int) -> Tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=bin_count)
    probs = counts / counts.sum()
    return edges, probs


def exact_peak_memory(prefill_tokens: np.ndarray, current_decode: np.ndarray, total_decode: np.ndarray) -> float:
    """
    Exact peak memory forward from the current state (deterministic).
    """
    if len(prefill_tokens) == 0:
        return 0.0
    remaining_decode = np.maximum(total_decode - current_decode, 0)
    current_tokens = prefill_tokens + current_decode
    order = np.argsort(remaining_decode)
    remaining_sorted = remaining_decode[order]
    current_sorted = current_tokens[order]
    total_c_sum = float(current_sorted.sum())
    cumulative_finished = np.cumsum(current_sorted)
    finished_shifted = np.concatenate(([0.0], cumulative_finished[:-1]))
    sum_active_c = total_c_sum - finished_shifted
    active_counts = np.arange(len(prefill_tokens), 0, -1)
    memory = sum_active_c + active_counts * remaining_sorted
    peak_future = float(memory.max()) if len(memory) > 0 else 0.0
    return max(peak_future, total_c_sum)  # include time 0


def simulate_state(
    df: pd.DataFrame,
    submit_rate: float,
    sim_steps: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a tiny decode simulator and return a snapshot of active requests.
    """
    prefill_col = df["prefill"].to_numpy()
    decode_col = df["decode"].to_numpy()
    active = []
    target_step = int(rng.integers(0, max(sim_steps, 1)))
    snapshot = None

    for step in range(sim_steps):
        n_new = rng.poisson(submit_rate)
        if n_new > 0:
            indices = rng.choice(len(df), size=n_new, replace=len(df) < n_new)
            for idx in indices:
                prefill = max(prefill_col[idx], 0)
                decode_total = max(decode_col[idx], 0)
                active.append(
                    {"prefill": prefill, "decode_total": decode_total, "decode_progress": 0}
                )

        for req in active:
            req["decode_progress"] = min(req["decode_progress"] + 1, req["decode_total"])

        active = [req for req in active if req["decode_progress"] < req["decode_total"]]

        if step == target_step and active:
            snapshot = [req.copy() for req in active]
            break

    if snapshot is None:
        raise RuntimeError("No active requests captured; increase sim-steps or submit-rate.")

    prefill_tokens = np.array([req["prefill"] for req in snapshot])
    current_decode = np.array([req["decode_progress"] for req in snapshot])
    total_decode = np.array([req["decode_total"] for req in snapshot])
    return prefill_tokens, current_decode, total_decode


def sample_active_subset(
    prefill_tokens: np.ndarray,
    current_decode: np.ndarray,
    total_decode: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    replace = len(prefill_tokens) < count
    indices = rng.choice(len(prefill_tokens), size=count, replace=replace)
    return prefill_tokens[indices], current_decode[indices], total_decode[indices]


def benchmark_trace(
    trace_path: str,
    bin_counts: Iterable[int],
    req_counts: Iterable[int],
    sim_counts: Iterable[int],
    n_trials: int,
    submit_rate: float,
    sim_steps: int,
    max_retries: int,
    rng: np.random.Generator,
) -> List[dict]:
    df = pd.read_csv(trace_path)
    if not {"prefill", "decode"}.issubset(df.columns):
        raise ValueError(f"{trace_path} missing prefill/decode columns")
    decode_lengths = df["decode"].to_numpy()
    decode_mean = float(decode_lengths.mean())
    decode_rms = float(np.sqrt(np.mean(np.square(decode_lengths))))
    results = []
    trace_name = Path(trace_path).stem
    for bin_count in bin_counts:
        bin_edges, bin_probs = build_bins(decode_lengths, bin_count)
        for req_count in req_counts:
            for n_sim in sim_counts:
                runtimes = []
                rel_errors = []
                baseline_mean_errors = []
                baseline_rms_errors = []
                over_errors = 0
                under_errors = 0
                for _ in range(n_trials):
                    snapshot = None
                    for _ in range(max_retries):
                        try:
                            snapshot = simulate_state(
                                df,
                                submit_rate=submit_rate,
                                sim_steps=sim_steps,
                                rng=rng,
                            )
                            break
                        except RuntimeError:
                            continue
                    if snapshot is None:
                        raise RuntimeError("Failed to capture active snapshot; adjust sim parameters.")

                    prefill_tokens, current_decode, total_decode = snapshot
                    prefill_tokens, current_decode, total_decode = sample_active_subset(
                        prefill_tokens, current_decode, total_decode, req_count, rng
                    )

                    actual_peak = exact_peak_memory(prefill_tokens, current_decode, total_decode)
                    baseline_remaining_mean = np.maximum(decode_mean - current_decode, 0)
                    baseline_remaining_rms = np.maximum(decode_rms - current_decode, 0)
                    baseline_peak_mean = exact_peak_memory(
                        prefill_tokens, current_decode, current_decode + baseline_remaining_mean
                    )
                    baseline_peak_rms = exact_peak_memory(
                        prefill_tokens, current_decode, current_decode + baseline_remaining_rms
                    )
                    start = time.perf_counter()
                    estimate = estimate_peak_memory_fast(
                        prefill_tokens,
                        current_decode,
                        bin_edges,
                        bin_probs,
                        n_simulations=n_sim,
                    )
                    runtimes.append((time.perf_counter() - start) * 1000.0)
                    diff = estimate - actual_peak
                    rel_error = abs(diff) / actual_peak if actual_peak > 0 else 0.0
                    rel_errors.append(rel_error)
                    if diff > 0:
                        over_errors += 1
                    elif diff < 0:
                        under_errors += 1
                    baseline_mean_errors.append(
                        abs(baseline_peak_mean - actual_peak) / actual_peak if actual_peak > 0 else 0.0
                    )
                    baseline_rms_errors.append(
                        abs(baseline_peak_rms - actual_peak) / actual_peak if actual_peak > 0 else 0.0
                    )
                results.append(
                    {
                        "trace": trace_name,
                        "bin_count": bin_count,
                        "request_count": req_count,
                        "n_simulations": n_sim,
                        "runtime_ms_avg": float(np.mean(runtimes)),
                        "runtime_ms_std": float(np.std(runtimes)),
                        "rel_error_avg": float(np.mean(rel_errors)),
                        "rel_error_std": float(np.std(rel_errors)),
                        "over_error_pct": float(over_errors / n_trials * 100.0),
                        "under_error_pct": float(under_errors / n_trials * 100.0),
                        "baseline_mean_rel_error": float(np.mean(baseline_mean_errors)),
                        "baseline_rms_rel_error": float(np.mean(baseline_rms_errors)),
                    }
                )
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark estimate_peak_memory_fast over trace data.")
    parser.add_argument(
        "--trace",
        default=None,
        help="Optional single trace file to run. If set, overrides --trace-glob.",
    )
    parser.add_argument(
        "--trace-glob",
        default="SLO-CSim/trace/arxiv/*.csv",
        help="Glob for input traces.",
    )
    parser.add_argument(
        "--bin-counts",
        default="8,16,32",
        help="Comma separated bin counts to test.",
    )
    parser.add_argument(
        "--req-counts",
        default="8,32,128",
        help="Comma separated request counts to sample per run.",
    )
    parser.add_argument(
        "--sim-counts",
        default="500,2000,8000",
        help="Comma separated n_simulations values to test.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Number of repeats per parameter combination.",
    )
    parser.add_argument(
        "--submit-rate",
        type=float,
        default=1.0,
        help="Average number of new requests per simulation step (Poisson).",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=2000,
        help="Number of simulation iterations per snapshot attempt.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries to capture a non-empty snapshot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed for reproducibility.",
    )
    args = parser.parse_args()

    if args.trace:
        trace_paths = [args.trace]
    else:
        trace_paths = sorted(glob.glob(args.trace_glob))
    if not trace_paths:
        raise SystemExit(f"No traces found for glob: {args.trace_glob}")

    bin_counts = _parse_int_list(args.bin_counts)
    req_counts = _parse_int_list(args.req_counts)
    sim_counts = _parse_int_list(args.sim_counts)
    rng = np.random.default_rng(args.seed)

    all_results = []
    for trace_path in trace_paths:
        all_results.extend(
            benchmark_trace(
                trace_path,
                bin_counts=bin_counts,
                req_counts=req_counts,
                sim_counts=sim_counts,
                n_trials=args.n_trials,
                submit_rate=args.submit_rate,
                sim_steps=args.sim_steps,
                max_retries=args.max_retries,
                rng=rng,
            )
        )

    df = pd.DataFrame(all_results)
    df = df.round(
        {
            "runtime_ms_avg": 2,
            "runtime_ms_std": 2,
            "rel_error_avg": 2,
            "rel_error_std": 2,
            "over_error_pct": 2,
            "under_error_pct": 2,
            "baseline_mean_rel_error": 2,
            "baseline_rms_rel_error": 2,
        }
    )
    print(df.to_csv(index=False))


if __name__ == "__main__":
    main()
