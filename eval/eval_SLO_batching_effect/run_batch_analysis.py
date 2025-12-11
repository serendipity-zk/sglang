#!/usr/bin/env python3
"""
Top-level controller to run batch throughput analysis across multiple (p, d) pairs and SLO targets.

Usage:
    python run_batch_analysis.py
"""

import sys
import os

sys.path.insert(0, '/sgl-workspace/sglang/sglang_profile')
from mode_aware_predictor import ModeAwarePredictor

# Import functions from estimate_batch_throughput
from estimate_batch_throughput import compute_batch_params_from_tokens, binary_search_max_batch, grid_search_max_throughput


# Configuration: (input_length, output_length) pairs
PD_PAIRS = [
    (28, 140),
    (36, 280),
    (1019, 130),
    (1024, 1024),
    (4096, 1024),
    (128, 1024),
]

# SLO targets in milliseconds
SLO_TARGETS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70]


def run_analysis():
    # Initialize predictor once
    print("Initializing ModeAwarePredictor...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    predictor = ModeAwarePredictor(
        grid_path="/sgl-workspace/sglang/sglang_profile/mode_3d.json",
        csv_log_path=os.path.join(script_dir, "batch_analysis_predictor_log.csv")
    )
    print()

    # Results storage
    results = []

    # Print header
    print("=" * 130)
    print(f"{'p':>6} {'d':>6} {'SLO(ms)':>8} {'opt_toks':>10} {'decode':>8} "
          f"{'iter_ms':>10} {'tok/s':>12} {'req/s':>10}")
    print("=" * 130)

    for p, d in PD_PAIRS:
        for slo_ms in SLO_TARGETS:
            # Run binary search and grid search (suppress internal prints)
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                # First find SLO-constrained max
                max_batch_tokens, _, _, _ = binary_search_max_batch(
                    p=p, d=d, slo_ms=slo_ms, predictor=predictor
                )
                # Then find optimal throughput within that range
                optimal_tokens, optimal_throughput, optimal_iter_time, optimal_decode_reqs = grid_search_max_throughput(
                    max_batch_tokens=max_batch_tokens,
                    p=p, d=d, slo_ms=slo_ms,
                    predictor=predictor,
                    num_points=50
                )
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout

            # Compute throughputs (using optimal values)
            token_throughput_per_sec = optimal_throughput * 1000  # convert from tokens/ms to tokens/sec
            requests_completed_per_iter = optimal_decode_reqs / d
            request_throughput_per_sec = (requests_completed_per_iter / optimal_iter_time * 1000) if optimal_iter_time > 0 else 0

            # Store result
            results.append({
                'p': p,
                'd': d,
                'slo_ms': slo_ms,
                'optimal_batch_tokens': optimal_tokens,
                'optimal_decode_reqs': optimal_decode_reqs,
                'optimal_iter_time': optimal_iter_time,
                'token_throughput_per_sec': token_throughput_per_sec,
                'request_throughput_per_sec': request_throughput_per_sec,
            })

            # Print row
            print(f"{p:>6} {d:>6} {slo_ms:>8} {optimal_tokens:>10} {optimal_decode_reqs:>8.1f} "
                  f"{optimal_iter_time:>10.3f} {token_throughput_per_sec:>12.1f} {request_throughput_per_sec:>10.2f}")

        # Separator between (p, d) pairs
        print("-" * 120)

    print("=" * 120)
    print()

    # Save results to CSV
    csv_path = os.path.join(script_dir, "batch_analysis_results.csv")
    with open(csv_path, 'w') as f:
        f.write("p,d,slo_ms,optimal_batch_tokens,optimal_decode_reqs,optimal_iter_time_ms,token_throughput_per_sec,request_throughput_per_sec\n")
        for r in results:
            f.write(f"{r['p']},{r['d']},{r['slo_ms']},{r['optimal_batch_tokens']},"
                    f"{r['optimal_decode_reqs']:.2f},{r['optimal_iter_time']:.3f},{r['token_throughput_per_sec']:.2f},"
                    f"{r['request_throughput_per_sec']:.4f}\n")

    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    run_analysis()
