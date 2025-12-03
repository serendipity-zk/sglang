#!/usr/bin/env python3
"""
Examine predict_batch latency as a function of batch size.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sglang_profile.mode_aware_predictor import ModeAwarePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure predict_batch latency across batch sizes.")
    parser.add_argument(
        "--grid-path",
        default=str(ROOT / "sglang_profile" / "mode_3d.json"),
        help="Path to grid3d.json/mode_3d.json.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Mode to test (MIXED/DECODE/EXTEND). If omitted for multi-mode grids, all modes are tested separately.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,8,32,128,512,1024",
        help="Comma-separated batch sizes to test.",
    )
    parser.add_argument("--calls", type=int, default=50, help="Number of predict_batch calls per batch size (including warmup).")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup calls per batch size.")
    parser.add_argument("--max-prefill-tokens", type=int, default=8192, help="Max total prefill tokens in synthetic workload.")
    parser.add_argument("--max-prefill-chunks", type=int, default=4, help="Max chunk count in synthetic prefill pairs.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10**9,
        help="Override predictor logging interval to avoid console spam.",
    )
    parser.add_argument(
        "--csv-log-path",
        type=str,
        default=None,
        help="Path to CSV log file for predictor (default: auto-generated)",
    )
    return parser.parse_args()


def parse_batch_sizes(raw: str) -> List[int]:
    sizes = [int(part) for part in raw.split(",") if part]
    if any(sz <= 0 for sz in sizes):
        raise ValueError("batch sizes must be positive integers")
    return sizes


def select_modes(predictor: ModeAwarePredictor, requested_mode: str) -> List[str]:
    if predictor.is_multimode:
        available = list(predictor.grids.keys())
        if requested_mode is not None:
            if requested_mode not in available:
                raise ValueError(f"Requested mode '{requested_mode}' not in available modes: {available}")
            return [requested_mode]
        return available
    return [None]


def make_prefill_pairs(
    rng: np.random.Generator, total_prefill_tokens: int, max_chunks: int
) -> List[List[int]]:
    if total_prefill_tokens <= 0:
        return []
    n_chunks = int(rng.integers(1, min(max_chunks, total_prefill_tokens) + 1))
    if n_chunks == 1:
        return [[int(total_prefill_tokens), int(total_prefill_tokens)]]
    split_points = sorted(rng.choice(np.arange(1, total_prefill_tokens), size=n_chunks - 1, replace=False).tolist())
    split_points = [0] + split_points + [total_prefill_tokens]
    pairs = []
    cumulative = 0
    for idx in range(1, len(split_points)):
        chunk = split_points[idx] - split_points[idx - 1]
        cumulative += chunk
        pairs.append([int(chunk), int(cumulative)])
    return pairs


def build_inputs(
    rng: np.random.Generator,
    batch_size: int,
    mode: str,
    x_knots: Sequence[float],
    z_knots: Sequence[float],
    max_prefill_tokens: int,
    max_prefill_chunks: int,
) -> Tuple[List[int], List[List[List[int]]], List[int]]:
    x_min, x_max = int(np.min(x_knots)), int(np.max(x_knots))
    z_min, z_max = int(np.min(z_knots)), int(np.max(z_knots))
    batches = []
    pairs_list = []
    kv_list = []
    for _ in range(batch_size):
        batch_tokens = int(rng.integers(x_min, x_max + 1))
        kv_tokens = int(rng.integers(z_min, z_max + 1))
        if mode == "DECODE":
            prefill_pairs: List[List[int]] = []
        else:
            total_prefill = int(rng.integers(0, max_prefill_tokens + 1))
            prefill_pairs = make_prefill_pairs(rng, total_prefill, max_prefill_chunks)
        batches.append(batch_tokens)
        pairs_list.append(prefill_pairs)
        kv_list.append(kv_tokens)
    return batches, pairs_list, kv_list


def measure_latency(
    predictor: ModeAwarePredictor,
    mode: str,
    batches: List[int],
    pairs_list: List[List[List[int]]],
    kv_list: List[int],
    calls: int,
    warmup: int,
) -> Tuple[float, float]:
    if calls <= warmup:
        raise ValueError("calls must be greater than warmup")
    mode_arg = mode if predictor.is_multimode else None
    for _ in range(warmup):
        predictor.predict_batch(batches, pairs_list, kv_list, modes=[mode_arg] * len(batches) if predictor.is_multimode else None)
    timings = []
    for _ in range(calls - warmup):
        t0 = time.perf_counter()
        predictor.predict_batch(batches, pairs_list, kv_list, modes=[mode_arg] * len(batches) if predictor.is_multimode else None)
        timings.append(time.perf_counter() - t0)
    mean_s = float(np.mean(timings))
    p90_s = float(np.percentile(timings, 90)) if timings else 0.0
    return mean_s, p90_s


def run_for_mode(
    predictor: ModeAwarePredictor,
    mode: str,
    batch_sizes: Iterable[int],
    calls: int,
    warmup: int,
    rng: np.random.Generator,
    max_prefill_tokens: int,
    max_prefill_chunks: int,
) -> None:
    if predictor.is_multimode:
        x_knots = predictor.X_knots_dict[mode]
        z_knots = predictor.Z_knots_dict[mode]
    else:
        x_knots = predictor.X_knots
        z_knots = predictor.Z_knots

    print(f"\nMode: {mode or 'single-mode'}")
    print("batch_size,mean_ms,p90_ms")
    for bs in batch_sizes:
        batches, pairs_list, kv_list = build_inputs(
            rng, bs, mode, x_knots, z_knots, max_prefill_tokens, max_prefill_chunks
        )
        mean_s, p90_s = measure_latency(
            predictor, mode, batches, pairs_list, kv_list, calls, warmup
        )
        print(f"{bs},{mean_s*1000:.4f},{p90_s*1000:.4f}")


def main() -> None:
    args = parse_args()
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    predictor = ModeAwarePredictor(
        grid_path=args.grid_path,
        log_every=args.log_every,
        csv_log_path=args.csv_log_path
    )
    rng = np.random.default_rng(args.seed)

    modes = select_modes(predictor, args.mode)
    for mode in modes:
        run_for_mode(
            predictor,
            mode,
            batch_sizes,
            calls=args.calls,
            warmup=args.warmup,
            rng=rng,
            max_prefill_tokens=args.max_prefill_tokens,
            max_prefill_chunks=args.max_prefill_chunks,
        )


if __name__ == "__main__":
    main()
