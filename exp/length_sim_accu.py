import argparse
import glob
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

import sys
import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.length_sim_algo import MonteCarloOutputEst
from exp.length_monitor import RequestOutputLengthProfile
from sglang_profile.mode_aware_predictor import ModeAwarePredictor


def _parse_int_list(raw: str) -> List[int]:
    return [int(part) for part in raw.split(",") if part]


def simulate_state(
    df: pd.DataFrame,
    submit_rate: float,
    sim_steps: int,
    rng: np.random.Generator,
    warmup: int = 30000,
    sample_interval: int = 4000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a tiny decode simulator and return a snapshot of active requests.
    """
    prefill_col = df["prefill"].to_numpy()
    decode_col = df["decode"].to_numpy()
    active = []
    target_step = int(rng.integers(0, max(sim_steps, 1))) + warmup
    snapshot = None
    next_sample = warmup

    # Warmup + repeated sampling every interval; stop after we pass target_step.
    for step in tqdm.tqdm(range(target_step + sample_interval + 1)):
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

        if step >= warmup and step >= next_sample and active:
            snapshot = [req.copy() for req in active]
            next_sample += sample_interval

    if snapshot is None:
        raise RuntimeError("No active requests captured; increase sim-steps or submit-rate.")

    prefill_tokens = np.array([req["prefill"] for req in snapshot])
    current_decode = np.array([req["decode_progress"] for req in snapshot])
    total_decode = np.array([req["decode_total"] for req in snapshot])
    return prefill_tokens, current_decode, total_decode


def simulate_snapshots(
    df: pd.DataFrame,
    submit_rate: float,
    rng: np.random.Generator,
    n_snapshots: int,
    warmup: int = 30000,
    sample_interval: int = 4000,
    max_retries: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Run a single simulation with one warmup, then sample snapshots every interval.

    Captures the first snapshot right after warmup (if there is active traffic),
    then continues every `sample_interval` steps.
    """
    prefill_col = df["prefill"].to_numpy()
    decode_col = df["decode"].to_numpy()
    active: List[dict] = []
    snapshots: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    total_est = warmup + sample_interval * n_snapshots * (max_retries + 1)
    pbar = tqdm.tqdm(total=total_est)
    steps_run = 0

    def _step():
        nonlocal steps_run
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

        # Drop finished requests
        active[:] = [req for req in active if req["decode_progress"] < req["decode_total"]]

        steps_run += 1
        pbar.update(1)

    # Warmup once
    for _ in range(warmup):
        _step()

    # Helper to capture after optional stepping
    def _capture_with_retries(initial_check: bool = False):
        for attempt in range(max_retries + 1):
            if initial_check and active:
                return True
            for _ in range(sample_interval):
                _step()
            if active:
                return True
            initial_check = False
        return False

    # First snapshot: try immediately after warmup, then advance by intervals if needed
    captured = _capture_with_retries(initial_check=True)
    if captured:
        snapshots.append(
            (
                np.array([req["prefill"] for req in active]),
                np.array([req["decode_progress"] for req in active]),
                np.array([req["decode_total"] for req in active]),
            )
        )
    else:
        pbar.close()
        raise RuntimeError("Failed to capture active snapshot; increase submit-rate or retries.")

    # Remaining snapshots on a fixed cadence
    for _ in range(n_snapshots - 1):
        captured = _capture_with_retries(initial_check=False)
        if not captured:
            pbar.close()
            raise RuntimeError("Failed to capture active snapshot; increase submit-rate or retries.")
        snapshots.append(
            (
                np.array([req["prefill"] for req in active]),
                np.array([req["decode_progress"] for req in active]),
                np.array([req["decode_total"] for req in active]),
            )
        )

    if steps_run < total_est:
        pbar.update(total_est - steps_run)
    pbar.close()

    return snapshots


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
    warmup: int = 30000,
    sample_interval: int = 4000,
    grid_path: str = None,
    tpot: float = 20.0,
) -> List[dict]:
    def _collect_snapshots(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return simulate_snapshots(
            df,
            submit_rate=submit_rate,
            rng=rng,
            n_snapshots=n_trials,
            warmup=warmup,
            sample_interval=sample_interval,
            max_retries=max_retries,
        )

    df = pd.read_csv(trace_path)
    if not {"prefill", "decode"}.issubset(df.columns):
        raise ValueError(f"{trace_path} missing prefill/decode columns")
    decode_lengths = df["decode"].to_numpy()
    decode_mean = float(decode_lengths.mean())
    decode_rms = float(np.sqrt(np.mean(np.square(decode_lengths))))
    snapshots = _collect_snapshots(df)
    results = []
    trace_name = Path(trace_path).stem

    # Initialize predictor if grid_path provided
    predictor = None
    if grid_path:
        print(f"[Benchmark] Loading predictor from {grid_path}")
        predictor = ModeAwarePredictor(grid_path=grid_path)

    for bin_count in bin_counts:
        # Create Monte Carlo estimator with new API
        estimator = MonteCarloOutputEst(
            window_size_batches=256,
            bins_per_decade=bin_count,
            predictor=predictor
        )

        # Pre-populate the estimator with the trace's decode length distribution
        estimator.submit_decode_length_observation(decode_lengths)

        for req_count in req_counts:
            for n_sim in sim_counts:
                mc_runtimes = []
                gt_runtimes = []
                mc_peaks = []
                gt_peaks = []
                mc_slacks = []
                gt_slacks = []
                baseline_mean_peaks = []
                baseline_rms_peaks = []
                mc_sample_times = []
                mc_memory_times = []
                mc_predictor_times = []
                mc_slack_times = []
                predictor_original_queries = []
                predictor_unique_queries = []
                predictor_compression_ratios = []
                predictor_mean_batches = []
                predictor_max_batches = []
                predictor_quant_times = []
                predictor_dedup_times = []
                predictor_predict_times = []
                predictor_broadcast_times = []
                predictor_total_times = []
                for snapshot in snapshots:
                    prefill_tokens, current_decode, total_decode = snapshot
                    prefill_tokens, current_decode, total_decode = sample_active_subset(
                        prefill_tokens, current_decode, total_decode, req_count, rng
                    )

                    # Calculate remaining decode for ground truth
                    remaining_decode = np.maximum(total_decode - current_decode, 0)

                    # Baseline predictions using GT method
                    baseline_remaining_mean = np.maximum(decode_mean - current_decode, 0)
                    baseline_remaining_rms = np.maximum(decode_rms - current_decode, 0)
                    baseline_peak_mean, _ = estimator.estimate_peak_and_slack_with_ground_truth(
                        prefill_tokens, current_decode, baseline_remaining_mean, tpot=None
                    )
                    baseline_peak_rms, _ = estimator.estimate_peak_and_slack_with_ground_truth(
                        prefill_tokens, current_decode, baseline_remaining_rms, tpot=None
                    )
                    baseline_mean_peaks.append(baseline_peak_mean)
                    baseline_rms_peaks.append(baseline_peak_rms)
                    # Monte Carlo prediction
                    mc_stats = {}
                    start_mc = time.perf_counter()
                    mc_peak, mc_slack = estimator.estimate_peak_and_slack(
                        prefill_tokens,
                        current_decode,
                        n_simulations=n_sim,
                        tpot=tpot,
                        stats=mc_stats,
                    )
                    mc_runtimes.append((time.perf_counter() - start_mc) * 1000.0)
                    step_times = mc_stats.get("step_times_ms", {})
                    mc_sample_times.append(step_times.get("sample", 0.0))
                    mc_memory_times.append(step_times.get("memory_matrix", 0.0))
                    mc_predictor_times.append(step_times.get("predictor", 0.0))
                    mc_slack_times.append(step_times.get("slack", 0.0))
                    predictor_stats = mc_stats.get("predictor_batch_size_stats")
                    if predictor_stats:
                        predictor_original_queries.append(predictor_stats.get("original_queries", 0))
                        predictor_unique_queries.append(predictor_stats.get("unique_queries", 0))
                        predictor_compression_ratios.append(predictor_stats.get("compression_ratio", 0.0))
                        predictor_mean_batches.append(predictor_stats.get("mean_batch", 0.0))
                        predictor_max_batches.append(predictor_stats.get("max_batch", 0.0))
                        timings = predictor_stats.get("timings_ms") or {}
                        predictor_quant_times.append(timings.get("quantize", 0.0))
                        predictor_dedup_times.append(timings.get("dedup", 0.0))
                        predictor_predict_times.append(timings.get("predict", 0.0))
                        predictor_broadcast_times.append(timings.get("broadcast", 0.0))
                        predictor_total_times.append(timings.get("total", 0.0))
                    mc_peaks.append(mc_peak)
                    mc_slacks.append(mc_slack)

                    # Ground truth prediction (perfect information)
                    start_gt = time.perf_counter()
                    gt_peak, gt_slack = estimator.estimate_peak_and_slack_with_ground_truth(
                        prefill_tokens,
                        current_decode,
                        remaining_decode,
                        tpot=tpot
                    )
                    gt_runtimes.append((time.perf_counter() - start_gt) * 1000.0)
                    gt_peaks.append(gt_peak)
                    gt_slacks.append(gt_slack)
                # Calculate differences between MC and GT
                mc_peaks_arr = np.array(mc_peaks)
                gt_peaks_arr = np.array(gt_peaks)
                mc_slacks_arr = np.array(mc_slacks)
                gt_slacks_arr = np.array(gt_slacks)
                baseline_mean_arr = np.array(baseline_mean_peaks)
                baseline_rms_arr = np.array(baseline_rms_peaks)

                # Peak errors: MC vs GT
                peak_diffs = mc_peaks_arr - gt_peaks_arr
                peak_rel_errors = np.abs(peak_diffs) / np.maximum(gt_peaks_arr, 1e-6)

                # Slack errors: MC vs GT
                slack_diffs = mc_slacks_arr - gt_slacks_arr
                slack_abs_errors = np.abs(slack_diffs)

                # Baseline errors vs GT
                baseline_mean_errors = np.abs(baseline_mean_arr - gt_peaks_arr) / np.maximum(gt_peaks_arr, 1e-6)
                baseline_rms_errors = np.abs(baseline_rms_arr - gt_peaks_arr) / np.maximum(gt_peaks_arr, 1e-6)

                results.append(
                    {
                        "trace": trace_name,
                        "bin_count": bin_count,
                        "request_count": req_count,
                        "n_simulations": n_sim,
                        # Runtime comparison
                        "mc_runtime_ms_avg": float(np.mean(mc_runtimes)),
                        "mc_runtime_ms_std": float(np.std(mc_runtimes)),
                        "mc_sample_ms_avg": float(np.mean(mc_sample_times)),
                        "mc_memory_ms_avg": float(np.mean(mc_memory_times)),
                        "mc_predictor_ms_avg": float(np.mean(mc_predictor_times)),
                        "mc_slack_ms_avg": float(np.mean(mc_slack_times)),
                        "gt_runtime_ms_avg": float(np.mean(gt_runtimes)),
                        "gt_runtime_ms_std": float(np.std(gt_runtimes)),
                        # Peak: MC vs GT comparison
                        "mc_peak_avg": float(np.mean(mc_peaks_arr)),
                        "gt_peak_avg": float(np.mean(gt_peaks_arr)),
                        "peak_rel_error_avg": float(np.mean(peak_rel_errors)),
                        "peak_rel_error_std": float(np.std(peak_rel_errors)),
                        # Slack: MC vs GT comparison
                        "mc_slack_avg": float(np.mean(mc_slacks_arr)),
                        "gt_slack_avg": float(np.mean(gt_slacks_arr)),
                        "slack_abs_error_avg": float(np.mean(slack_abs_errors)),
                        "slack_abs_error_std": float(np.std(slack_abs_errors)),
                        # Baseline metrics
                        "baseline_mean_rel_error": float(np.mean(baseline_mean_errors)),
                        "baseline_rms_rel_error": float(np.mean(baseline_rms_errors)),
                        # Predictor batch stats
                        "predictor_original_queries": float(np.mean(predictor_original_queries)) if predictor_original_queries else 0.0,
                        "predictor_unique_queries": float(np.mean(predictor_unique_queries)) if predictor_unique_queries else 0.0,
                        "predictor_compression_ratio": float(np.mean(predictor_compression_ratios)) if predictor_compression_ratios else 0.0,
                        "predictor_mean_batch": float(np.mean(predictor_mean_batches)) if predictor_mean_batches else 0.0,
                        "predictor_max_batch": float(np.mean(predictor_max_batches)) if predictor_max_batches else 0.0,
                        "predictor_quantize_ms": float(np.mean(predictor_quant_times)) if predictor_quant_times else 0.0,
                        "predictor_dedup_ms": float(np.mean(predictor_dedup_times)) if predictor_dedup_times else 0.0,
                        "predictor_predict_ms": float(np.mean(predictor_predict_times)) if predictor_predict_times else 0.0,
                        "predictor_broadcast_ms": float(np.mean(predictor_broadcast_times)) if predictor_broadcast_times else 0.0,
                        "predictor_total_ms": float(np.mean(predictor_total_times)) if predictor_total_times else 0.0,
                    }
                )
    return results


def _render_table(df: pd.DataFrame) -> None:
    """Pretty-print results using rich if available; fallback to CSV."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print("rich is not installed; falling back to CSV output.", file=sys.stderr)
        print(df.to_csv(index=False))
        return

    console = Console()
    table = Table(show_lines=False, title="MC vs GT Estimation Benchmark")
    table.add_column("trace", style="cyan", no_wrap=True)
    table.add_column("bins", justify="right")
    table.add_column("reqs", justify="right")
    table.add_column("sims", justify="right")
    table.add_column("MC runtime", justify="right")
    table.add_column("MC sample", justify="right")
    table.add_column("MC memory", justify="right")
    table.add_column("MC predictor", justify="right")
    table.add_column("MC slack", justify="right")
    table.add_column("GT runtime", justify="right")
    table.add_column("MC peak", justify="right")
    table.add_column("GT peak", justify="right")
    table.add_column("peak err %", justify="right", style="bold yellow")
    table.add_column("MC slack", justify="right")
    table.add_column("GT slack", justify="right")
    table.add_column("slack err", justify="right", style="bold cyan")
    table.add_column("base mean", justify="right")
    table.add_column("base rms", justify="right")
    table.add_column("pred queries", justify="right")
    table.add_column("pred unique", justify="right")
    table.add_column("pred comp", justify="right")
    table.add_column("pred batch μ", justify="right")
    table.add_column("pred batch max", justify="right")
    table.add_column("pred quant ms", justify="right")
    table.add_column("pred dedup ms", justify="right")
    table.add_column("pred predict ms", justify="right")
    table.add_column("pred bcast ms", justify="right")
    table.add_column("pred total ms", justify="right")

    for _, row in df.iterrows():
        table.add_row(
            str(row["trace"]),
            f"{int(row['bin_count'])}",
            f"{int(row['request_count'])}",
            f"{int(row['n_simulations'])}",
            f"{row['mc_runtime_ms_avg']:.2f} ± {row['mc_runtime_ms_std']:.2f}",
            f"{row['mc_sample_ms_avg']:.2f}",
            f"{row['mc_memory_ms_avg']:.2f}",
            f"{row['mc_predictor_ms_avg']:.2f}",
            f"{row['mc_slack_ms_avg']:.2f}",
            f"{row['gt_runtime_ms_avg']:.2f} ± {row['gt_runtime_ms_std']:.2f}",
            f"{row['mc_peak_avg']:.0f}",
            f"{row['gt_peak_avg']:.0f}",
            f"{row['peak_rel_error_avg']*100:.2f} ± {row['peak_rel_error_std']*100:.2f}",
            f"{row['mc_slack_avg']:.2f}",
            f"{row['gt_slack_avg']:.2f}",
            f"{row['slack_abs_error_avg']:.2f} ± {row['slack_abs_error_std']:.2f}",
            f"{row['baseline_mean_rel_error']*100:.2f}",
            f"{row['baseline_rms_rel_error']*100:.2f}",
            f"{row['predictor_original_queries']:.1f}",
            f"{row['predictor_unique_queries']:.1f}",
            f"{row['predictor_compression_ratio']:.2f}",
            f"{row['predictor_mean_batch']:.1f}",
            f"{row['predictor_max_batch']:.1f}",
            f"{row['predictor_quantize_ms']:.2f}",
            f"{row['predictor_dedup_ms']:.2f}",
            f"{row['predictor_predict_ms']:.2f}",
            f"{row['predictor_broadcast_ms']:.2f}",
            f"{row['predictor_total_ms']:.2f}",
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Benchmark MonteCarloOutputEst peak memory estimation over trace data.")
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
        default="16,32,64",
        help="Comma separated bin counts to test.",
    )
    parser.add_argument(
        "--req-counts",
        default="8,32,128,256",
        help="Comma separated request counts to sample per run.",
    )
    parser.add_argument(
        "--sim-counts",
        default="100,200,400",
        help="Comma separated n_simulations values to test.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of repeats per parameter combination.",
    )
    parser.add_argument(
        "--submit-rate",
        type=float,
        default=4.0,
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
        default=0,
        help="Retries to capture a non-empty snapshot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12234,
        help="RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30000,
        help="Warmup steps before collecting snapshots.",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=4000,
        help="Step interval between snapshot samples after warmup.",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "csv"],
        default="table",
        help="Output results as a rich table (default) or CSV.",
    )
    parser.add_argument(
        "--grid-path",
        default="/sgl-workspace/sglang/sglang_profile/mode_3d.json",
        help="Path to grid3d.json for ModeAwarePredictor.",
    )
    parser.add_argument(
        "--tpot",
        type=float,
        default=20.0,
        help="Target time per output token in ms for slack calculation.",
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
                warmup=args.warmup,
                sample_interval=args.sample_interval,
                grid_path=args.grid_path,
                tpot=args.tpot,
            )
        )

    df = pd.DataFrame(all_results)
    df = df.round(
        {
            "mc_runtime_ms_avg": 2,
            "mc_runtime_ms_std": 2,
            "mc_sample_ms_avg": 2,
            "mc_memory_ms_avg": 2,
            "mc_predictor_ms_avg": 2,
            "mc_slack_ms_avg": 2,
            "gt_runtime_ms_avg": 2,
            "gt_runtime_ms_std": 2,
            "mc_peak_avg": 0,
            "gt_peak_avg": 0,
            "peak_rel_error_avg": 4,
            "peak_rel_error_std": 4,
            "mc_slack_avg": 2,
            "gt_slack_avg": 2,
            "slack_abs_error_avg": 2,
            "slack_abs_error_std": 2,
            "baseline_mean_rel_error": 4,
            "baseline_rms_rel_error": 4,
            "predictor_original_queries": 1,
            "predictor_unique_queries": 1,
            "predictor_compression_ratio": 2,
            "predictor_mean_batch": 1,
            "predictor_max_batch": 1,
            "predictor_quantize_ms": 2,
            "predictor_dedup_ms": 2,
            "predictor_predict_ms": 2,
            "predictor_broadcast_ms": 2,
            "predictor_total_ms": 2,
        }
    )
    if args.output_format == "csv":
        print(df.to_csv(index=False))
    else:
        _render_table(df)


if __name__ == "__main__":
    main()
