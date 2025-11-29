#!/usr/bin/env python3
"""
Micro-benchmark for ModeAwarePredictor.predict_batch throughput.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sglang_profile.mode_aware_predictor import ModeAwarePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ModeAwarePredictor.predict_batch throughput.")
    parser.add_argument(
        "--grid-path",
        default=str(ROOT / "sglang_profile" / "mode_3d.json"),
        help="Path to grid3d.json/mode_3d.json.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Mode to test (MIXED/DECODE/EXTEND). If omitted for multi-mode grids, all modes are tested.",
    )
    parser.add_argument("--total", type=int, default=50000, help="Total predictions (including warmup).")
    parser.add_argument("--warmup", type=int, default=2000, help="Warmup predictions before timing.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size per predict_batch call.")
    parser.add_argument("--max-prefill-tokens", type=int, default=8192, help="Max total prefill tokens in synthetic workload.")
    parser.add_argument("--max-prefill-chunks", type=int, default=4, help="Max chunk count in synthetic prefill pairs.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=10**9,
        help="Override predictor logging interval to avoid console spam.",
    )
    return parser.parse_args()


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
    total: int,
    mode: str,
    x_knots: Sequence[float],
    z_knots: Sequence[float],
    max_prefill_tokens: int,
    max_prefill_chunks: int,
) -> List[Tuple[int, List[List[int]], int]]:
    x_min, x_max = int(np.min(x_knots)), int(np.max(x_knots))
    z_min, z_max = int(np.min(z_knots)), int(np.max(z_knots))
    inputs = []
    for _ in range(total):
        batch_tokens = int(rng.integers(x_min, x_max + 1))
        kv_tokens = int(rng.integers(z_min, z_max + 1))
        if mode == "DECODE":
            prefill_pairs: List[List[int]] = []
        else:
            total_prefill = int(rng.integers(0, max_prefill_tokens + 1))
            prefill_pairs = make_prefill_pairs(rng, total_prefill, max_prefill_chunks)
        inputs.append((batch_tokens, prefill_pairs, kv_tokens))
    return inputs


def run_batches(
    predictor: ModeAwarePredictor,
    mode_arg: str,
    inputs: Sequence[Tuple[int, List[List[int]], int]],
    batch_size: int,
) -> None:
    idx = 0
    total = len(inputs)
    while idx < total:
        end = min(idx + batch_size, total)
        batch = inputs[idx:end]
        predictor.predict_batch(
            [item[0] for item in batch],
            [item[1] for item in batch],
            [item[2] for item in batch],
            modes=[mode_arg] * len(batch) if predictor.is_multimode else None,
        )
        idx = end


def benchmark_mode(
    predictor: ModeAwarePredictor,
    mode: str,
    inputs: List[Tuple[int, List[List[int]], int]],
    warmup: int,
    batch_size: int,
) -> Tuple[int, float, float]:
    if warmup >= len(inputs):
        raise ValueError("Warmup must be smaller than total predictions.")
    mode_arg = mode if predictor.is_multimode else None
    run_batches(predictor, mode_arg, inputs[:warmup], batch_size)
    start = time.perf_counter()
    run_batches(predictor, mode_arg, inputs[warmup:], batch_size)
    elapsed = time.perf_counter() - start
    n_samples = len(inputs) - warmup
    pps = n_samples / elapsed if elapsed > 0 else float("inf")
    return n_samples, elapsed, pps


def main() -> None:
    args = parse_args()
    predictor = ModeAwarePredictor(args.grid_path, log_every=args.log_every)
    rng = np.random.default_rng(args.seed)

    if args.total <= args.warmup:
        raise SystemExit("total must be greater than warmup.")

    modes = select_modes(predictor, args.mode)
    for mode in modes:
        if predictor.is_multimode:
            x_knots = predictor.X_knots_dict[mode]
            z_knots = predictor.Z_knots_dict[mode]
        else:
            x_knots = predictor.X_knots
            z_knots = predictor.Z_knots

        inputs = build_inputs(
            rng,
            args.total,
            mode,
            x_knots,
            z_knots,
            args.max_prefill_tokens,
            args.max_prefill_chunks,
        )
        n_samples, elapsed, pps = benchmark_mode(
            predictor, mode, inputs, args.warmup, args.batch_size
        )
        print(
            f"[mode={mode or 'single-mode'}] {pps:,.0f} predictions/sec | "
            f"{n_samples} samples in {elapsed:.3f}s (warmup={args.warmup}, batch={args.batch_size})"
        )


if __name__ == "__main__":
    main()
