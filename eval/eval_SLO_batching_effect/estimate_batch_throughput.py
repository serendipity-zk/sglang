#!/usr/bin/env python3
"""
Estimate max batch size and throughput for a given request type with SLO constraint.

Usage:
    python estimate_batch_throughput.py --input_length 512 --output_length 128 --slo_ms 50

The script uses binary search to find the largest batch size B where the predicted
iteration time stays within the SLO target.

Batch composition model:
- Given batch size B, assume d/(p+d)*B requests are in decode phase
- The remaining p/(p+d)*B requests are in prefill phase
- KV tokens = decode_count * (p + d/2) (decode requests at average position)
- Prefill pairs: whole requests get [p, p], fractional remainder gets [remaining, remaining]
"""

import sys
import os
import argparse

sys.path.insert(0, '/sgl-workspace/sglang/sglang_profile')
from mode_aware_predictor import ModeAwarePredictor


def compute_batch_params_from_tokens(batch_tokens: int, p: int, d: int):
    """
    Compute batch parameters for a given batch token count.

    Args:
        batch_tokens: Total tokens in the batch
        p: Input/prefill length per request
        d: Output/decode length per request

    Returns:
        batch_tokens: Total tokens in the batch (same as input)
        prefill_pairs: List of [chunk_size, cumulative] pairs for prefill requests
        kv_tokens: Total KV cache tokens used
        decode_req_count: Number of decode requests
        prefill_req_count: Number of prefill requests

    Note on token composition:
        In steady state, request ratio is 1:d (prefill:decode).
        Each decode request contributes 1 token, each prefill request contributes p tokens.
        So for B total requests:
            batch_tokens = decode_tokens + prefill_tokens
                        = (d/(1+d))*B * 1 + (1/(1+d))*B * p
                        = B * (d + p) / (1 + d)
        Therefore:
            B = batch_tokens * (1 + d) / (d + p)
    """
    # Compute number of requests from batch tokens
    B = batch_tokens * (1 + d) / (d + p)

    # Fraction of requests in each phase (based on request ratio 1:d)
    decode_req_count = d / (1 + d) * B
    prefill_req_count = 1 / (1 + d) * B

    # KV tokens from decode requests (each at average position p + d/2)
    kv_tokens = int(decode_req_count * (p + d / 2))

    # Actual token counts
    decode_tokens = int(decode_req_count)  # 1 token per decode request
    prefill_tokens = batch_tokens - decode_tokens  # Remaining tokens are prefill

    # Build prefill chunk pairs
    prefill_pairs = []
    whole_prefills = int(prefill_req_count)

    # Add pairs for whole prefill requests
    for _ in range(whole_prefills):
        prefill_pairs.append([p, p])

    # Handle fractional remainder
    remainder = prefill_req_count - whole_prefills
    if remainder > 0:
        remaining_tokens = int(remainder * p)
        if remaining_tokens > 0:
            prefill_pairs.append([remaining_tokens, remaining_tokens])

    return batch_tokens, prefill_pairs, kv_tokens, decode_req_count, prefill_req_count


def binary_search_max_batch(p: int, d: int, slo_ms: float, predictor: ModeAwarePredictor):
    """
    Binary search to find the maximum batch tokens that satisfies the SLO constraint.

    Args:
        p: Input/prefill length
        d: Output/decode length
        slo_ms: SLO target in milliseconds
        predictor: ModeAwarePredictor instance

    Returns:
        max_batch_tokens: Maximum batch tokens that satisfies SLO
        iter_time: Iteration time at max_batch_tokens
        request_count: Number of requests at max_batch_tokens
    """
    low, high = 1, 8192
    iteration = 0
    kv_tokens_limit = 1000000

    print("=" * 80)
    print(f"Binary Search for Max Batch Tokens")
    print(f"  Input length (p): {p}")
    print(f"  Output length (d): {d}")
    print(f"  SLO target: {slo_ms} ms")
    print(f"  Batch token search range: [{low}, {high}] tokens")
    print(f"  KV tokens limit: {kv_tokens_limit}")
    print("=" * 80)
    print()

    while low < high:
        iteration += 1
        mid = (low + high + 1) // 2

        batch_tokens, prefill_pairs, kv_tokens, decode_req_count, prefill_req_count = \
            compute_batch_params_from_tokens(mid, p, d)

        # Check KV tokens limit first
        if kv_tokens > kv_tokens_limit:
            iter_time = float('inf')
            status = "KV_EXCEED"
        else:
            iter_time = predictor.predict(batch_tokens, prefill_pairs, kv_tokens, mode="MIXED")
            status = "OK" if iter_time <= slo_ms else "EXCEED"

        # Compute detailed breakdown
        decode_tokens = int(decode_req_count)  # 1 token per decode request
        prefill_tokens = batch_tokens - decode_tokens
        total_requests = decode_req_count + prefill_req_count

        print(f"[Iter {iteration:2d}] batch_toks={mid:5d} | "
              f"reqs={total_requests:6.1f} (decode={decode_req_count:5.1f}, prefill={prefill_req_count:4.1f}) | "
              f"decode_toks={decode_tokens:5d}, prefill_toks={prefill_tokens:5d} | "
              f"kv_tokens={kv_tokens:9d} | "
              f"prefill_pairs={len(prefill_pairs):3d} | "
              f"iter_time={iter_time:8.3f}ms | "
              f"SLO={slo_ms}ms | [{status}]")

        if status == "OK":
            low = mid
            print(f"         -> {mid} tokens fits SLO, searching higher...")
        else:
            high = mid - 1
            if status == "KV_EXCEED":
                print(f"         -> {mid} tokens exceeds KV limit ({kv_tokens} > {kv_tokens_limit}), searching lower...")
            else:
                print(f"         -> {mid} tokens exceeds SLO, searching lower...")
        print()

    # Final result
    final_batch_tokens, final_prefill_pairs, final_kv_tokens, final_decode_reqs, final_prefill_reqs = \
        compute_batch_params_from_tokens(low, p, d)
    final_iter_time = predictor.predict(final_batch_tokens, final_prefill_pairs, final_kv_tokens, mode="MIXED")
    final_request_count = final_decode_reqs + final_prefill_reqs

    return low, final_iter_time, final_request_count, final_decode_reqs


def grid_search_max_throughput(max_batch_tokens: int, p: int, d: int, slo_ms: float,
                                predictor: ModeAwarePredictor, num_points: int = 50):
    """
    Grid search from 1 to max_batch_tokens to find batch size with max throughput.

    Args:
        max_batch_tokens: Upper bound from binary search (SLO-constrained max)
        p: Input/prefill length
        d: Output/decode length
        slo_ms: SLO target in milliseconds
        predictor: ModeAwarePredictor instance
        num_points: Number of points to evaluate in grid search

    Returns:
        optimal_batch_tokens: Batch size with highest throughput
        optimal_throughput_per_ms: The max throughput value (tokens/ms)
        optimal_iter_time: Iteration time at optimal point
        optimal_decode_reqs: Decode requests at optimal point
    """
    step = max(1, max_batch_tokens // num_points)
    kv_tokens_limit = 1000000

    best_tokens = 1
    best_throughput = 0.0
    best_iter_time = 0.0
    best_decode_reqs = 0.0

    print()
    print("=" * 80)
    print(f"Grid Search for Max Throughput (0 to {max_batch_tokens} tokens, step={step})")
    print("=" * 80)

    # Evaluate points from 1 to max_batch_tokens
    points_to_check = list(range(1, max_batch_tokens + 1, step))
    # Always include the max point
    if max_batch_tokens not in points_to_check:
        points_to_check.append(max_batch_tokens)

    for batch_tokens in points_to_check:
        batch_tokens_actual, prefill_pairs, kv_tokens, decode_req_count, prefill_req_count = \
            compute_batch_params_from_tokens(batch_tokens, p, d)

        # Skip if KV limit exceeded
        if kv_tokens > kv_tokens_limit:
            continue

        iter_time = predictor.predict(batch_tokens_actual, prefill_pairs, kv_tokens, mode="MIXED")

        # Skip if SLO exceeded
        if iter_time > slo_ms:
            continue

        throughput = batch_tokens / iter_time if iter_time > 0 else 0

        is_best = throughput > best_throughput
        marker = " *" if is_best else ""

        print(f"  batch_toks={batch_tokens:5d} | iter_time={iter_time:8.3f}ms | "
              f"throughput={throughput:8.3f} tokens/ms ({throughput*1000:10.1f} tokens/sec){marker}")

        if is_best:
            best_throughput = throughput
            best_tokens = batch_tokens
            best_iter_time = iter_time
            best_decode_reqs = decode_req_count

    print()
    print(f"  Best: batch_toks={best_tokens}, throughput={best_throughput:.3f} tokens/ms")
    print("=" * 80)

    return best_tokens, best_throughput, best_iter_time, best_decode_reqs


def main():
    parser = argparse.ArgumentParser(
        description="Estimate max batch size and throughput for given request type with SLO constraint"
    )
    parser.add_argument("--input_length", "-p", type=int, required=True,
                        help="Input/prefill length per request")
    parser.add_argument("--output_length", "-d", type=int, required=True,
                        help="Output/decode length per request")
    parser.add_argument("--slo_ms", type=float, required=True,
                        help="SLO target in milliseconds")
    parser.add_argument("--grid_path", type=str,
                        default="/sgl-workspace/sglang/sglang_profile/mode_3d.json",
                        help="Path to the 3D grid JSON file")

    args = parser.parse_args()

    # Initialize predictor with fixed log path (overwrites each run)
    print("Initializing ModeAwarePredictor...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    predictor = ModeAwarePredictor(
        grid_path=args.grid_path,
        csv_log_path=os.path.join(script_dir, "predictor_log.csv")
    )
    print()

    # Run binary search to find SLO-constrained max
    max_batch_tokens, iter_time, request_count, decode_reqs = binary_search_max_batch(
        p=args.input_length,
        d=args.output_length,
        slo_ms=args.slo_ms,
        predictor=predictor
    )

    # Run grid search to find optimal throughput within SLO constraint
    optimal_tokens, optimal_throughput, optimal_iter_time, optimal_decode_reqs = grid_search_max_throughput(
        max_batch_tokens=max_batch_tokens,
        p=args.input_length,
        d=args.output_length,
        slo_ms=args.slo_ms,
        predictor=predictor,
        num_points=50
    )

    # Compute throughput at SLO-constrained max
    token_throughput_per_ms = max_batch_tokens / iter_time if iter_time > 0 else 0
    token_throughput_per_sec = token_throughput_per_ms * 1000

    # Compute request throughput at optimal point
    requests_completed_per_iter = optimal_decode_reqs / args.output_length
    request_throughput_per_ms = requests_completed_per_iter / optimal_iter_time if optimal_iter_time > 0 else 0
    request_throughput_per_sec = request_throughput_per_ms * 1000

    # Print final results
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"  Input length (p):      {args.input_length}")
    print(f"  Output length (d):     {args.output_length}")
    print(f"  SLO target:            {args.slo_ms} ms")
    print()
    print(f"  SLO-Constrained Max:")
    print(f"    Max batch tokens:    {max_batch_tokens}")
    print(f"    Iteration time:      {iter_time:.3f} ms")
    print(f"    Token throughput:    {token_throughput_per_sec:.1f} tokens/sec")
    print()
    print(f"  Optimal for Throughput:")
    print(f"    Batch tokens:        {optimal_tokens}")
    print(f"    Iteration time:      {optimal_iter_time:.3f} ms")
    print(f"    Token throughput:    {optimal_throughput * 1000:.1f} tokens/sec")
    print()
    print(f"  Request throughput:    {request_throughput_per_sec:.2f} requests/sec")
    print("=" * 80)


if __name__ == "__main__":
    main()
