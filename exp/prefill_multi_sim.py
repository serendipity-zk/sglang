#!/usr/bin/env python3
"""
Refactored Batch Prefill Simulator.
Organized into domain classes: Scenario, ProposedPlan, TimelineSimulator, GlobalPredictor.
Features:
- Stateless execution.
- Super-batching (aggregates all scenarios into one GPU prediction).
- Distinct PASS (met all deadlines) vs LATE (feasible but missed deadline) statuses.
"""

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional
from pathlib import Path
import sys
import math
import time
import random

# --- Path Setup ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from sglang_profile.mode_aware_predictor import ModeAwarePredictor
except ImportError:
    print("Error: Could not import 'sglang_profile'. Make sure you are in the correct directory.")
    sys.exit(1)

GRANULARITY = 128


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass(frozen=True)
class SimulatorConfig:
    """Immutable hardware and system configuration."""
    decode_batch: int
    kv_cache: int
    tpot_ms: float
    slack_decode_ms: float
    grid_path: str


@dataclass
class BatchPlanResult:
    """Final output for a single plan simulation."""
    base_plan: List[int]
    success: bool
    decode_feasible: bool
    prefill_met: int
    total_time_ms: float
    padded_total_time_ms: float
    iterations: int
    min_decode_slack_ms: float
    min_prefill_slack_ms: float
    reason: str
    final_prefill_slacks_ms: List[float]  # Added back for reporting metrics
    execution_flow: List[int] = None  # Full execution timeline with padding (0 = decode-only)
    execution_times: List[float] = None  # Time (ms) consumed at each step in execution_flow


# ==============================================================================
# Domain Classes
# ==============================================================================

class ProposedPlan:
    """
    Represents a specific chunking strategy (e.g. [128, 128, 64]) for a specific set of requests.
    Responsibility: Calculate the token schedule and prepare raw inputs for prediction.
    """
    def __init__(self, chunks: List[int], req_lens: List[int]):
        self.chunks = chunks
        self.req_lens = req_lens

        # Derived attributes (computed in one pass)
        self.schedule_segments, self.cycle_inputs = self._build_schedule_and_inputs()

        # Filled later by the GlobalPredictor
        self.cycle_times_ms: List[float] = []

    def _build_schedule_and_inputs(self) -> Tuple[List[List[Tuple[int, int]]], List[Tuple[int, List[List[int]]]]]:
        """
        OPTIMIZED: Builds schedule segments AND predictor inputs in a single pass.
        Returns: (schedule_segments, cycle_inputs)
        """
        schedule = []
        inputs = []

        if not self.req_lens:
            return schedule, inputs

        req_idx = 0
        remaining = self.req_lens[req_idx]
        temp_progress = [0] * len(self.req_lens)

        for chunk in self.chunks:
            left = chunk
            chunk_segs = []
            pairs = []
            chunk_len = 0

            while left > 0 and req_idx < len(self.req_lens):
                alloc = min(left, remaining)

                # Build schedule segment
                chunk_segs.append((req_idx, alloc))

                # Build predictor input (reuse alloc, avoid recomputing)
                start_pos = temp_progress[req_idx]
                end_pos = start_pos + alloc
                pairs.append([alloc, end_pos])
                temp_progress[req_idx] = end_pos

                chunk_len += alloc
                remaining -= alloc
                left -= alloc

                if remaining == 0:
                    req_idx += 1
                    if req_idx < len(self.req_lens):
                        remaining = self.req_lens[req_idx]

            schedule.append(chunk_segs)
            if chunk_len > 0:
                inputs.append((chunk_len, pairs))

        return schedule, inputs


class SimulationScenario:
    """
    Represents a User Scenario (Set of requests and slacks).
    Responsibility: Normalization (Align/Sort) and Plan Generation.
    """
    def __init__(self, raw_lens: Sequence[int], raw_slacks: Sequence[float]):
        if len(raw_lens) != len(raw_slacks):
            raise ValueError("Lengths and slacks must match")
        
        # 1. Align to granularity and Sort by slack (Tightest first)
        zipped = []
        for l, s in zip(raw_lens, raw_slacks):
            aligned_len = (int(l) + GRANULARITY - 1) // GRANULARITY * GRANULARITY
            zipped.append((aligned_len, float(s)))
        zipped.sort(key=lambda x: x[1])
        
        self.req_lens = [z[0] for z in zipped]
        self.req_slacks = [z[1] for z in zipped]
        self.total_len = sum(self.req_lens)
        
        # Populated later
        self.plans: List[ProposedPlan] = []

    def _sample_random_plan(self, max_chunks: int) -> List[int]:
        """Generate a random plan by sampling split points."""
        if self.total_len == 0:
            return [0]

        units = self.total_len // GRANULARITY
        max_chunks = max(1, min(max_chunks, units))
        n_chunks = random.randint(1, max_chunks)

        if n_chunks == 1:
            return [self.total_len]

        # Sample n_chunks-1 split points
        split_points = sorted(random.sample(range(1, units), n_chunks - 1))
        split_points = [0] + split_points + [units]

        # Convert to chunk sizes
        plan = []
        for i in range(1, len(split_points)):
            delta_units = split_points[i] - split_points[i - 1]
            plan.append(delta_units * GRANULARITY)

        return plan

    def generate_plans(self) -> None:
        """Enumerates diverse plans: power-of-2, group-by-request, and random splits."""
        t_start = time.perf_counter()

        # Edge case: If total_len is 0, make a dummy plan
        if self.total_len == 0:
            self.plans = [ProposedPlan([0], self.req_lens)]
            return

        raw_plans = []

        # 1. Power-of-2 equal-division plans (existing)
        t_pow2_start = time.perf_counter()
        chunk_size = GRANULARITY
        while chunk_size < self.total_len:
            p = [chunk_size] * (self.total_len // chunk_size)
            rem = self.total_len % chunk_size
            if rem:
                p.append(rem)
            raw_plans.append(p)
            chunk_size *= 2
        raw_plans.append([self.total_len])
        t_pow2_end = time.perf_counter()

        # 2. Group-by-request plan
        t_group_start = time.perf_counter()
        raw_plans.append(self.req_lens)
        t_group_end = time.perf_counter()

        # 3. Random split plans
        t_random_start = time.perf_counter()
        num_random_samples = 5
        units = self.total_len // GRANULARITY
        for _ in range(num_random_samples):
            random_plan = self._sample_random_plan(max_chunks=units)
            raw_plans.append(random_plan)
        t_random_end = time.perf_counter()

        # Deduplication
        t_dedup_start = time.perf_counter()
        unique_plans = []
        seen = set()
        for plan in raw_plans:
            key = tuple(plan)
            if key not in seen:
                seen.add(key)
                unique_plans.append(plan)

        t_create_start = time.perf_counter()
        self.plans = [ProposedPlan(p, self.req_lens) for p in unique_plans]

        t_end = time.perf_counter()
        pow2_time = (t_pow2_end - t_pow2_start) * 1000
        group_time = (t_group_end - t_group_start) * 1000
        random_time = (t_random_end - t_random_start) * 1000
        dedup_time = (t_create_start - t_dedup_start) * 1000
        create_time = (t_end - t_create_start) * 1000
        # Debug: print number of plans generated (uncomment if needed)
        print(f"[PERF] generate_plans: {(t_end - t_start)*1000:.3f} ms (pow2: {pow2_time:.3f}, group: {group_time:.3f}, rnd: {random_time:.3f}, dedup: {dedup_time:.3f}, create: {create_time:.3f}) - {len(self.plans)} plans")


class GlobalBatchPredictor:
    """
    Wraps the ML Predictor.
    Responsibility: Aggregate requests from ALL plans in ALL scenarios, predict once, distribute results.
    """
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.predictor = ModeAwarePredictor(config.grid_path)
        self.is_multimode = getattr(self.predictor, "is_multimode", False)
        
        # Base decode latency is constant for a given batch/kv setup
        self.base_decode_ms = float(
            self.predictor.predict(config.decode_batch, [], config.kv_cache, mode="DECODE")
        )

    def predict_all(self, scenarios: List[SimulationScenario]):
        """
        1. Collects all cycle inputs from all plans.
        2. Runs one massive predict_batch.
        3. Assigns times back to ProposedPlan objects.
        """
        t_start = time.perf_counter()
        all_batches = []
        all_pairs = []
        all_kv = []
        all_modes = []

        # We need to map the flat results back to: [Scenario] -> [Plan] -> [Cycle]
        # Map structure: list of (plan_obj, num_cycles)
        plan_map: List[Tuple[ProposedPlan, int]] = []

        for sc in scenarios:
            for plan in sc.plans:
                inputs = plan.cycle_inputs # List of (len, pairs)
                count = len(inputs)
                plan_map.append((plan, count))
                
                for chunk_len, pairs in inputs:
                    all_batches.append(self.config.decode_batch + chunk_len)
                    all_pairs.append(pairs)
                    all_kv.append(self.config.kv_cache)
                    if self.is_multimode:
                        all_modes.append("MIXED" if chunk_len > 0 else "DECODE")

        t_collect = time.perf_counter()

        # The Big Prediction
        if all_batches:
            preds = self.predictor.predict_batch(
                all_batches, all_pairs, all_kv, 
                modes=all_modes if self.is_multimode else None
            )
            preds = [float(x) for x in preds]
        else:
            preds = []

        t_predict = time.perf_counter()

        # Distribute results back
        cursor = 0
        for plan_obj, count in plan_map:
            plan_obj.cycle_times_ms = preds[cursor : cursor + count]
            cursor += count

        t_end = time.perf_counter()
        print(f"[PERF] predict_all: {(t_end - t_start)*1000:.3f} ms (collect: {(t_collect - t_start)*1000:.3f} ms, predict: {(t_predict - t_collect)*1000:.3f} ms, distribute: {(t_end - t_predict)*1000:.3f} ms) - {len(all_batches)} predictions")


class TimelineSimulator:
    """
    Pure Logic Engine.
    Responsibility: Simulate time, enforce TPOT, handle slack.
    """
    @staticmethod
    def simulate(plan: ProposedPlan, slacks: List[float], config: SimulatorConfig, base_decode_ms: float) -> BatchPlanResult:
        current_slacks = list(slacks)
        min_prefill_slack = min(current_slacks) if current_slacks else 0.0

        decode_slack = config.slack_decode_ms
        min_decode_slack = decode_slack

        total_time = 0.0
        iterations = 0
        decode_feasible = True
        execution_flow = []  # Track actual execution timeline
        execution_times = []  # Track time at each step

        progress = [0] * len(plan.req_lens)
        done = [False] * len(plan.req_lens)

        # How much slack we recover per pure decode step
        decode_recovery_rate = config.tpot_ms - base_decode_ms

        def consume_time(delta_ms: float):
            nonlocal min_prefill_slack
            for idx, finished in enumerate(done):
                if not finished:
                    current_slacks[idx] -= delta_ms
                    min_prefill_slack = min(min_prefill_slack, current_slacks[idx])

        # Step through the plan
        for chunk_size, (chunk_segs, cycle_time) in zip(plan.chunks, zip(plan.schedule_segments, plan.cycle_times_ms)):
            # 1. Decode Constraint Check
            est_decode_slack = decode_slack - cycle_time + config.tpot_ms

            if est_decode_slack < 0:
                # Must inject wait cycles
                if decode_recovery_rate <= 1e-9:
                    decode_feasible = False
                    num_waits = 0
                else:
                    num_waits = math.ceil(-est_decode_slack / decode_recovery_rate)

                # Add wait cycles to execution flow
                execution_flow.extend([0] * num_waits)
                execution_times.extend([base_decode_ms] * num_waits)

                wait_time = num_waits * base_decode_ms
                consume_time(wait_time)
                total_time += wait_time
                iterations += num_waits
                decode_slack += num_waits * decode_recovery_rate
                min_decode_slack = min(min_decode_slack, decode_slack)

            # 2. Prefill Execution
            execution_flow.append(chunk_size)  # Add prefill chunk to flow
            execution_times.append(cycle_time)  # Add prefill time

            consume_time(cycle_time)
            total_time += cycle_time
            iterations += 1
            decode_slack += config.tpot_ms - cycle_time
            min_decode_slack = min(min_decode_slack, decode_slack)

            # 3. Update Progress
            for req_idx, alloc in chunk_segs:
                progress[req_idx] += alloc
                if progress[req_idx] >= plan.req_lens[req_idx]:
                    done[req_idx] = True

        prefill_met = sum(1 for s in current_slacks if s >= 0)
        
        if min_decode_slack < 0 or decode_slack < 0:
            decode_feasible = False
        
        success = decode_feasible and prefill_met == len(current_slacks)
        
        return BatchPlanResult(
            base_plan=plan.chunks,
            success=success,
            decode_feasible=decode_feasible,
            prefill_met=prefill_met,
            total_time_ms=total_time,
            padded_total_time_ms=total_time, # Will be padded by Selector
            iterations=iterations,
            min_decode_slack_ms=min_decode_slack,
            min_prefill_slack_ms=min_prefill_slack,
            reason="ok" if success else ("decode infeasible" if not decode_feasible else "prefill slack miss"),
            final_prefill_slacks_ms=current_slacks,
            execution_flow=execution_flow,
            execution_times=execution_times
        )


# ==============================================================================
# Orchestrator
# ==============================================================================

class PrefillSimulatorEngine:
    def __init__(self, grid_path: str):
        self.grid_path = grid_path
        self.config: Optional[SimulatorConfig] = None
        self.predictor: Optional[GlobalBatchPredictor] = None

    def update_decode(
        self,
        decode_batch: int,
        kv_cache: int,
        tpot_ms: float,
        slack_decode_ms: float,
    ) -> None:
        self.config = SimulatorConfig(
            decode_batch=decode_batch,
            kv_cache=kv_cache,
            tpot_ms=tpot_ms,
            slack_decode_ms=slack_decode_ms,
            grid_path=self.grid_path,
        )
        self.predictor = GlobalBatchPredictor(self.config)

    def _ensure_ready(self) -> None:
        if self.config is None or self.predictor is None:
            raise RuntimeError("Call update_decode before running simulations.")

    def evaluate_extras(
        self,
        base_lens: Sequence[int],
        base_slacks: Sequence[float],
        extra_lens: Sequence[int]
    ) -> List[BatchPlanResult]:
        self._ensure_ready()
        assert self.config is not None
        assert self.predictor is not None

        t_scenario_start = time.perf_counter()
        # 1. Create Scenarios
        scenarios: List[SimulationScenario] = []
        total_sc_init = 0.0
        total_plan_gen = 0.0

        for extra in extra_lens:
            t_sc_start = time.perf_counter()
            if extra == 0:
                # Base Case: Use original lists
                sc = SimulationScenario(base_lens, base_slacks)
            else:
                # Extra Case: Copy and append
                sc = SimulationScenario(
                    list(base_lens) + [extra],
                    list(base_slacks) + [float('inf')]
                )
            t_sc_end = time.perf_counter()
            total_sc_init += (t_sc_end - t_sc_start)

            t_gen_start = time.perf_counter()
            sc.generate_plans()
            t_gen_end = time.perf_counter()
            total_plan_gen += (t_gen_end - t_gen_start)

            scenarios.append(sc)

        t_scenario_end = time.perf_counter()
        print(f"[PERF] scenario_creation+plan_generation: {(t_scenario_end - t_scenario_start)*1000:.3f} ms (sc_init: {total_sc_init*1000:.3f} ms, plan_gen: {total_plan_gen*1000:.3f} ms)")

        # 2. Predict All (Super-Batch)
        self.predictor.predict_all(scenarios)

        # 3. Simulate & Select Best per Scenario
        t_sim_start = time.perf_counter()
        final_results = []
        base_decode = self.predictor.base_decode_ms

        total_sim_time = 0.0
        total_select_time = 0.0
        total_plans_simulated = 0

        for sc in scenarios:
            results = []
            t_sc_sim_start = time.perf_counter()
            for plan in sc.plans:
                res = TimelineSimulator.simulate(plan, sc.req_slacks, self.config, base_decode)
                results.append(res)
                total_plans_simulated += 1
            t_sc_sim_end = time.perf_counter()
            total_sim_time += (t_sc_sim_end - t_sc_sim_start)

            t_select_start = time.perf_counter()
            best = self._select_best(results, base_decode)
            t_select_end = time.perf_counter()
            total_select_time += (t_select_end - t_select_start)

            final_results.append(best)

        t_sim_end = time.perf_counter()
        print(f"[PERF] simulate+select: {(t_sim_end - t_sim_start)*1000:.3f} ms (simulate: {total_sim_time*1000:.3f} ms, select: {total_select_time*1000:.3f} ms) - {total_plans_simulated} plans")

        return final_results

    def _select_best(self, results: List[BatchPlanResult], base_decode_ms: float) -> BatchPlanResult:
        successful = [r for r in results if r.success]
        decode_feasible = [r for r in results if r.decode_feasible]

        def pad(r: BatchPlanResult, max_i: int):
            if r.iterations < max_i:
                num_padding = max_i - r.iterations
                r.padded_total_time_ms = r.total_time_ms + num_padding * base_decode_ms
                # Don't add padding to execution_flow - user only wants actual execution

        if successful:
            max_iters = max(r.iterations for r in successful)
            for r in successful: pad(r, max_iters)
            successful.sort(key=lambda r: (r.padded_total_time_ms, -r.min_decode_slack_ms))
            return successful[0]

        if decode_feasible:
            max_iters = max(r.iterations for r in decode_feasible)
            for r in decode_feasible: pad(r, max_iters)
            decode_feasible.sort(key=lambda r: (-r.prefill_met, r.padded_total_time_ms))
            return decode_feasible[0]

        return BatchPlanResult(
            base_plan=[], success=False, decode_feasible=False,
            prefill_met=0, total_time_ms=0.0, padded_total_time_ms=0.0, iterations=0,
            min_decode_slack_ms=-1, min_prefill_slack_ms=-1, reason="infeasible",
            final_prefill_slacks_ms=[],
            execution_flow=[],
            execution_times=[]
        )


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refactored Batch Simulator")
    parser.add_argument("--prefill-lens", type=int, nargs="+", required=True)
    parser.add_argument("--prefill-slacks", type=float, nargs="+", required=True)
    parser.add_argument("--decode-batch", type=int, default=256)
    parser.add_argument("--kv-cache", type=int, default=50_000)
    parser.add_argument("--tpot", type=float, default=30.0)
    parser.add_argument("--decode-slack", type=float, default=0.0)
    parser.add_argument("--grid-path", default="sglang_profile/grid3d.json")
    parser.add_argument("--test-extra-len", type=int, nargs="+", default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    engine = PrefillSimulatorEngine(
        grid_path=args.grid_path,
    )
    engine.update_decode(
        decode_batch=args.decode_batch,
        kv_cache=args.kv_cache,
        tpot_ms=args.tpot,
        slack_decode_ms=args.decode_slack,
    )

    # If no extras provided, just run base case (extra=0)
    extras = args.test_extra_len if args.test_extra_len else [0]

    t_start = time.perf_counter()
    results = engine.evaluate_extras(
        args.prefill_lens, 
        args.prefill_slacks, 
        extras
    )
    t_end = time.perf_counter()
    print(f"[PERF] TOTAL TIME: {(t_end - t_start)*1000:.3f} ms")
    print("=" * 85)
    # UPDATED TABLE HEADER
    print(f"{'Extra':<10} | {'Status':<10} | {'Time (ms)':<10} | {'PadTime':<10} | {'Req SLO':<10} | {'Execution Flow'}")
    print("-" * 85)

    base_time = results[0].total_time_ms if results else 0.0
    base_decode_ms = engine.predictor.base_decode_ms

    for extra, res in zip(extras, results):
        # DETAILED STATUS LOGIC
        if res.success:
            status = "PASS"
        elif res.decode_feasible:
            status = "LATE"  # Feasible, but missed deadlines
        else:
            status = "FAIL"  # System constraint (TPOT) violated

        delta = ""
        if extra > 0 and (res.success or res.decode_feasible):
             delta = f"(+{res.total_time_ms - base_time:.1f})"
        
        # Create binary string: 1=met SLO, 0=missed SLO
        if res.final_prefill_slacks_ms:
            slo_status = ''.join('1' if slack >= 0 else '0' for slack in res.final_prefill_slacks_ms)
        else:
            slo_status = ""

        # Print execution flow (with padding) instead of base plan
        flow_str = str(res.execution_flow) if res.execution_flow else str(res.base_plan)
        print(f"{extra:<10} | {status:<10} | {res.total_time_ms:<10.2f} | {res.padded_total_time_ms:<10.2f} | {slo_status:<10} | {flow_str}")

        if status == "FAIL":
            print(f"  Reason: {res.reason}")
        elif status == "LATE":
            print(f"  Reason: Missed Prefill SLO (Min Slack: {res.min_prefill_slack_ms:.2f} ms)")

        # Show detailed timeline at each step
        if res.execution_flow and res.execution_times:
            # Build timeline string showing each step with its time
            timeline_parts = []
            cumulative_time = 0.0
            for chunk, time_ms in zip(res.execution_flow, res.execution_times):
                cumulative_time += time_ms
                if chunk == 0:
                    timeline_parts.append(f"0({time_ms:.1f}ms)")
                else:
                    timeline_parts.append(f"{chunk}({time_ms:.1f}ms)")

            print(f"  Timeline: [{', '.join(timeline_parts)}]")
            print(f"  Total: {cumulative_time:.2f} ms over {len(res.execution_flow)} iterations")

if __name__ == "__main__":
    main()
