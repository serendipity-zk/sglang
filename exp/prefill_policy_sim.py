import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sglang_profile.mode_aware_predictor import ModeAwarePredictor

GRANULARITY = 128

# Predictor signature: returns latency (ms) for the provided workload.
PrefillPredictor = Callable[[int, List[List[int]], int, str], float]


def toy_predictor(token_batch: int, prefill_chunk_pairs: List[List[int]], kv_cache_size: int, mode: str) -> float:
    # A simplified model for demonstration purposes
    prefill_workload = sum(pair[0] * pair[1] for pair in prefill_chunk_pairs)
    # The return value is the predicted latency (ms)
    return 0.04 * token_batch + 0.00002 * kv_cache_size + 0.00005 * prefill_workload


def build_mode_predictor(grid_path: str) -> PrefillPredictor:
    model = ModeAwarePredictor(grid_path)

    def _predict(token_batch: int, prefill_chunk_pairs: List[List[int]], kv_cache_size: int, mode: str) -> float:
        return float(model.predict(token_batch, prefill_chunk_pairs, kv_cache_size, mode=mode))

    return _predict


@dataclass
class PrefillPlanResult:
    strategy: str  # "incremental" or "frontwait"
    initial_wait_cycles: int
    initial_wait_time_ms: float
    plan: List[int]
    success: bool
    total_time_ms: float
    wait_time_ms: float
    padded_total_time_ms: float
    num_iterations: int
    min_decode_slack_ms: float
    final_decode_slack_ms: float
    final_prefill_slack_ms: float
    final_kv_tokens: int
    reason: str
    timeline: List[Dict]


def sample_random_plan(prefill_len: int, granularity: int, rng: np.random.Generator, max_chunks: int) -> List[int]:
    if prefill_len % granularity != 0:
        raise ValueError(f"prefill_len must be a multiple of {granularity}")
    units = prefill_len // granularity
    max_chunks = max(1, min(max_chunks, units))
    n_chunks = int(rng.integers(1, max_chunks + 1))
    if n_chunks == 1:
        return [prefill_len]
    split_points = sorted(rng.choice(np.arange(1, units), size=n_chunks - 1, replace=False).tolist())
    split_points = [0] + split_points + [units]
    plan = []
    for i in range(1, len(split_points)):
        delta_units = split_points[i] - split_points[i - 1]
        plan.append(delta_units * granularity)
    return plan


def search_wait_cycles(
    slack_ms: float,
    tpot_ms: float,
    decode_batch: int,
    kv_cache_size: int,
    predictor: PrefillPredictor,
) -> Tuple[bool, int, float, float, int, float, str]:
    """
    Minimal decode-only waits so the next decode cycle respects slack.
    Returns (success, wait_cycles, wait_time, final_slack, final_kv, min_slack_seen, reason)
    """
    wait_cycles = 0
    wait_time = 0.0
    min_slack_seen = slack_ms
    kv = kv_cache_size
    slack = slack_ms
    reason = "ok"

    # Minimal waiting: wait until slack is non-negative
    while slack < 0:
        decode_time = predictor(decode_batch, [], kv, "DECODE")
        
        # Check for impossibility: if decode time > tpot and slack is negative, we can't recover.
        if decode_time >= tpot_ms and slack < 0:
            reason = "Decode-only cost exceeds tpot with negative slack; infeasible"
            return False, wait_cycles, wait_time, slack, kv, min_slack_seen, reason
        
        slack += tpot_ms - decode_time
        min_slack_seen = min(min_slack_seen, slack) # Prospective slack is the slack *after* this cycle
        
        wait_cycles += 1
        wait_time += decode_time
        kv += decode_batch
    
    # After the loop, slack is >= 0, meaning we have waited the minimum cycles.
    return True, wait_cycles, wait_time, slack, kv, min_slack_seen, reason


def simulate_interleave_wait(
    prefill_len: int,
    plan: Sequence[int],
    *,
    tpot_ms: float,
    initial_decode_slack_ms: float,
    initial_prefill_slack_ms: float,
    decode_batch: int,
    kv_cache_size: int,
    predictor: PrefillPredictor,
) -> PrefillPlanResult:
    if sum(plan) != prefill_len:
        raise ValueError("Plan must sum to prefill_len")
    
    # 1. Check for initial wait
    success_init, init_cycles, init_time, slack_decode, kv_size, min_slack_decode, reason_init = search_wait_cycles(
        initial_decode_slack_ms, tpot_ms, decode_batch, kv_cache_size, predictor
    )

    if not success_init and init_cycles == 0:
        # Initial slack is too negative and cannot be recovered in the first decode cycle
        return PrefillPlanResult(
            strategy="incremental", initial_wait_cycles=0, initial_wait_time_ms=0.0, plan=list(plan), success=False,
            total_time_ms=0.0, wait_time_ms=0.0, padded_total_time_ms=0.0, num_iterations=0,
            min_decode_slack_ms=min_slack_decode, final_decode_slack_ms=-math.inf, final_prefill_slack_ms=-math.inf,
            final_kv_tokens=kv_cache_size, reason=reason_init, timeline=[],
        )

    # State after minimal initial wait (if any)
    slack_prefill = float(initial_prefill_slack_ms)
    total_time = init_time
    wait_time = init_time
    iterations = init_cycles
    prefill_done = 0
    prefill_pairs: List[List[int]] = []
    timeline: List[Dict] = []
    
    # 2. Process prefill chunks interleaved with decode
    for chunk in plan:
        current_pair = [chunk, prefill_done + chunk]
        mode = "MIXED" if chunk > 0 else "DECODE"
        
        # Current cycle
        cycle_time = predictor(decode_batch + chunk, prefill_pairs + [current_pair], kv_size, mode)
        total_time += cycle_time
        iterations += 1

        # Check prefill slack
        slack_prefill -= cycle_time
        if slack_prefill < 0:
            return PrefillPlanResult(
                strategy="incremental", initial_wait_cycles=init_cycles, initial_wait_time_ms=init_time, plan=list(plan),
                success=False, total_time_ms=total_time, wait_time_ms=wait_time, padded_total_time_ms=total_time,
                num_iterations=iterations, min_decode_slack_ms=min_slack_decode, final_decode_slack_ms=slack_decode,
                final_prefill_slack_ms=slack_prefill, final_kv_tokens=kv_size, reason="Prefill slack below zero during plan",
                timeline=timeline,
            )

        # Update decode slack after cycle
        slack_decode += tpot_ms - cycle_time
        min_slack_decode = min(min_slack_decode, slack_decode)
        
        # Determine minimal guard wait cycles
        success_guard, g_cycles, g_time, slack_decode, kv_size_after_guard, guard_min_slack, reason = search_wait_cycles(
            slack_decode, tpot_ms, decode_batch, kv_size, predictor
        )

        # Update state with guard cycles
        total_time += g_time
        wait_time += g_time
        iterations += g_cycles
        min_slack_decode = min(min_slack_decode, guard_min_slack)
        
        if not success_guard:
            return PrefillPlanResult(
                strategy="incremental", initial_wait_cycles=init_cycles, initial_wait_time_ms=init_time, plan=list(plan),
                success=False, total_time_ms=total_time, wait_time_ms=wait_time, padded_total_time_ms=total_time,
                num_iterations=iterations, min_decode_slack_ms=min_slack_decode, final_decode_slack_ms=slack_decode,
                final_prefill_slack_ms=slack_prefill, final_kv_tokens=kv_size, reason=reason, timeline=timeline,
            )

        timeline.append(
            {
                "chunk": chunk, "prefill_pair": current_pair, "cycle_time_ms": cycle_time, 
                "prefill_done": prefill_done + chunk, "kv_cache_size": kv_size, 
                "decode_slack_after_ms": slack_decode, "prefill_slack_after_ms": slack_prefill, 
                "guard_cycles": g_cycles, "guard_time_ms": g_time,
            }
        )

        prefill_done += chunk
        prefill_pairs.append(current_pair)
        kv_size = kv_size_after_guard # kv_size is updated by the guard cycles

    success = (slack_decode >= 0) and (slack_prefill >= 0)
    reason = "ok" if success else "Slack below zero by end of plan"
    return PrefillPlanResult(
        strategy="incremental", initial_wait_cycles=init_cycles, initial_wait_time_ms=init_time, plan=list(plan),
        success=success, total_time_ms=total_time, wait_time_ms=wait_time, padded_total_time_ms=total_time,
        num_iterations=iterations, min_decode_slack_ms=min_slack_decode, final_decode_slack_ms=slack_decode,
        final_prefill_slack_ms=slack_prefill, final_kv_tokens=kv_size, reason=reason, timeline=timeline,
    )


def simulate_frontwait(
    prefill_len: int,
    plan: Sequence[int],
    *,
    tpot_ms: float,
    initial_decode_slack_ms: float,
    initial_prefill_slack_ms: float,
    decode_batch: int,
    kv_cache_size: int,
    predictor: PrefillPredictor,
    max_wait_cycles: int = 512,
) -> PrefillPlanResult:
    if sum(plan) != prefill_len:
        raise ValueError("Plan must sum to prefill_len")

    # The loop searches for the MINIMAL initial_wait that makes the plan successful.
    best_result: PrefillPlanResult = PrefillPlanResult(
        strategy="frontwait", initial_wait_cycles=0, initial_wait_time_ms=0.0, plan=list(plan), success=False,
        total_time_ms=0.0, wait_time_ms=0.0, padded_total_time_ms=0.0, num_iterations=0,
        min_decode_slack_ms=initial_decode_slack_ms, final_decode_slack_ms=-math.inf, final_prefill_slack_ms=-math.inf,
        final_kv_tokens=kv_cache_size, reason="Infeasible under frontwait", timeline=[],
    )
    
    # print(f"Init decode time: {predictor(decode_batch, [], kv_cache_size, 'DECODE'):.2f} ms")
    print(f"plan: {plan}")
    sum_history = 0
    for chunk in plan:
        sum_history += chunk
        print(f"chunk: {chunk} time: {predictor(chunk+decode_batch, [[chunk, sum_history]], kv_cache_size, 'Mixed'):.2f} ms")

    for initial_wait in range(0, max_wait_cycles + 1):
        slack_decode = float(initial_decode_slack_ms)
        slack_prefill = float(initial_prefill_slack_ms)
        total_time = 0.0
        wait_time = 0.0
        iterations = 0
        min_slack_decode = slack_decode
        kv_size = kv_cache_size
        timeline: List[Dict] = []
        prefill_done = 0
        prefill_pairs: List[List[int]] = []
        infeasible_wait = False

        # --- 1. Initial Wait Phase ---
        for _ in range(initial_wait):
            decode_time = predictor(decode_batch, [], kv_size, "DECODE")
            total_time += decode_time
            wait_time += decode_time
            iterations += 1
            slack_decode += tpot_ms - decode_time
            min_slack_decode = min(min_slack_decode, slack_decode)
            kv_size += decode_batch
            
            # Check for impossibility during wait cycles
            if slack_decode < 0 and decode_time >= tpot_ms:
                infeasible_wait = True
                break
        
        if infeasible_wait:
            # If the wait cycle itself becomes infeasible, further waiting won't help
            best_result.reason = "Initial wait sequence became infeasible"
            break

        success = True
        best_result.reason = "ok"

        # --- 2. Prefill Execution Phase ---
        for chunk in plan:
            current_pair = [chunk, prefill_done + chunk]
            mode = "MIXED" if chunk > 0 else "DECODE"
            
            # Current cycle
            cycle_time = predictor(decode_batch + chunk, prefill_pairs + [current_pair], kv_size, mode)
            total_time += cycle_time
            iterations += 1
            slack_prefill -= cycle_time
            slack_decode += tpot_ms - cycle_time
            min_slack_decode = min(min_slack_decode, slack_decode)

            timeline.append(
                {
                    "chunk": chunk, "prefill_pair": current_pair, "cycle_time_ms": cycle_time, 
                    "prefill_done": prefill_done + chunk, "kv_cache_size": kv_size, 
                    "decode_slack_after_ms": slack_decode, "prefill_slack_after_ms": slack_prefill, 
                    "guard_cycles": 0, "guard_time_ms": 0.0,
                }
            )

            prefill_done += chunk
            prefill_pairs.append(current_pair)
            kv_size += decode_batch

            if slack_prefill < 0:
                success = False
                best_result.reason = "Prefill slack below zero during plan"
                break
            if slack_decode < 0:
                success = False
                best_result.reason = "Decode slack below zero during plan (frontwait)"
                break
        
        # --- 3. Result Check ---
        if success:
            # Found the minimal successful initial_wait
            return PrefillPlanResult(
                strategy="frontwait", initial_wait_cycles=initial_wait, initial_wait_time_ms=wait_time,
                plan=list(plan), success=True, total_time_ms=total_time, wait_time_ms=wait_time,
                padded_total_time_ms=total_time, num_iterations=iterations, min_decode_slack_ms=min_slack_decode,
                final_decode_slack_ms=slack_decode, final_prefill_slack_ms=slack_prefill,
                final_kv_tokens=kv_size, reason="ok", timeline=timeline,
            )

    # If the loop completes without finding a successful plan, return the impossible result.
    return best_result


def enumerate_candidate_plans(prefill_len: int, granularity: int, rng: np.random.Generator, samples: int) -> List[List[int]]:
    units = prefill_len // granularity
    candidates: List[List[int]] = []
    candidates.append([prefill_len]) # One big chunk
    for k in (2, 4, 8):
        if units % k == 0:
            candidates.append([prefill_len // k] * k) # Even splits
    for _ in range(samples):
        candidates.append(sample_random_plan(prefill_len, granularity, rng, max_chunks=units)) # Random splits
    unique = []
    seen = set()
    for plan in candidates:
        key = tuple(plan)
        if key not in seen:
            seen.add(key)
            unique.append(plan)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Explore prefill chunking plans under slack constraints.")
    parser.add_argument("--prefill-len", type=int, default=512, help="Total prefill tokens for the new request.")
    parser.add_argument("--decode-batch", type=int, default=256, help="Decode batch size per cycle.")
    parser.add_argument("--kv-cache", type=int, default=50_000, help="Current KV cache size (tokens).")
    parser.add_argument("--tpot", type=float, default=30.0, help="Allowed time per decode iteration (ms).")
    parser.add_argument("--slack-decode", type=float, default=0.0, help="Initial decode slack (ms).")
    parser.add_argument("--slack-prefill", type=float, default=300.0, help="Initial prefill slack (ms), consumed by prefill work.")
    parser.add_argument("--samples", type=int, default=0, help="Random trajectory samples.")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument(
        "--predictor",
        choices=["grid", "toy"],
        default="grid", # Changed to toy for simple execution without external files
        help="Which predictor to use for cycle time estimation.",
    )
    parser.add_argument(
        "--grid-path",
        default="sglang_profile/grid3d.json",
        help="Path to grid3d.json for grid predictor (ignored if using 'toy').",
    )
    parser.add_argument(
        "--strategy",
        choices=["incremental", "frontwait", "both"],
        default="both", # Changed to both to compare the two strategies
        help="Wait handling strategy: incremental waits between chunks or frontload waits once.",
    )
    args = parser.parse_args()

    if args.prefill_len % GRANULARITY != 0:
        raise SystemExit(f"prefill_len must be a multiple of {GRANULARITY}")

    rng = np.random.default_rng(args.seed)
    plans = enumerate_candidate_plans(args.prefill_len, GRANULARITY, rng, args.samples)

    if args.predictor == "grid":
        try:
            predictor = build_mode_predictor(args.grid_path)
        except Exception:
            print("Warning: Grid predictor failed to load. Falling back to toy_predictor.")
            predictor = toy_predictor
    else:
        predictor = toy_predictor

    strategy_flags = []
    if args.strategy in ("incremental", "both"):
        strategy_flags.append("incremental")
    if args.strategy in ("frontwait", "both"):
        strategy_flags.append("frontwait")

    results: List[PrefillPlanResult] = []
    for plan in plans:
        if "incremental" in strategy_flags:
            results.append(
                simulate_interleave_wait(
                    args.prefill_len, plan, tpot_ms=args.tpot, initial_decode_slack_ms=args.slack_decode,
                    initial_prefill_slack_ms=args.slack_prefill, decode_batch=args.decode_batch,
                    kv_cache_size=args.kv_cache, predictor=predictor,
                )
            )
        if "frontwait" in strategy_flags:
            results.append(
                simulate_frontwait(
                    args.prefill_len, plan, tpot_ms=args.tpot, initial_decode_slack_ms=args.slack_decode,
                    initial_prefill_slack_ms=args.slack_prefill, decode_batch=args.decode_batch,
                    kv_cache_size=args.kv_cache, predictor=predictor,
                )
            )

    # Calculate padded time (Total time including decode cycles until MaxIterations is reached)
    successful_results = [r for r in results if r.success]
    if successful_results:
        max_iters_success = max(r.num_iterations for r in successful_results)
    else:
        max_iters_success = 0
    
    for res in results:
        padded = res.total_time_ms
        if res.success and res.num_iterations < max_iters_success:
            kv_size = res.final_kv_tokens
            missing = max_iters_success - res.num_iterations
            for _ in range(missing):
                cycle_time = predictor(args.decode_batch, [], kv_size, "DECODE")
                padded += cycle_time
                kv_size += args.decode_batch
            res.padded_total_time_ms = padded
        else:
            res.padded_total_time_ms = padded

    # Sort: Successful first, then by minimal padded time, then maximal min_decode_slack, then strategy
    results.sort(key=lambda r: (not r.success, r.padded_total_time_ms, -r.min_decode_slack_ms, r.strategy))
    
    if not results:
        print("No plans were generated or tested.")
        return

    best = results[0]

    print(f"Tested {len(results)} plan/strategy combinations (granularity={GRANULARITY}).")
    print("---")
    print(f"**Best Plan Found**")
    print(f"  Plan: {best.plan}")
    print(f"  Strategy: **{best.strategy}**")
    print(f"  Success: {best.success}")
    print(f"  Total Time (ms): {best.total_time_ms:.2f} (Padded: {best.padded_total_time_ms:.2f})")
    print(f"  Iterations: {best.num_iterations} | Wait Time (ms): {best.wait_time_ms:.2f}")
    print(f"  Min Decode Slack (ms): {best.min_decode_slack_ms:.2f} | Final Decode Slack (ms): {best.final_decode_slack_ms:.2f}")
    print(f"  Final Prefill Slack (ms): {best.final_prefill_slack_ms:.2f}")
    print(f"  Reason: {best.reason}")
    print("---")
    print("\nTop 5 Plans (Sorted by Success, Padded Time, Min Slack):")
    for res in results[:5]:
        print(
            f"  plan={res.plan} strategy={res.strategy} iterations={res.num_iterations} success={res.success} "
            f"total_ms={res.total_time_ms:.2f} padded_total_ms={res.padded_total_time_ms:.2f} "
            f"wait_ms={res.wait_time_ms:.2f} min_decode_slack_ms={res.min_decode_slack_ms:.2f} "
            f"final_decode_slack_ms={res.final_decode_slack_ms:.2f} "
            f"final_prefill_slack_ms={res.final_prefill_slack_ms:.2f} reason={res.reason}"
        )


if __name__ == "__main__":
    main()
