#!/usr/bin/env python3
"""
Top-level controller to run router scale experiments across different server counts and rates.

For each server count N, runs experiments at:
  - rate = N * 20
  - rate = N * 40

Usage:
    python run_scale_sweep.py --server-counts 2 4 8 16
    python run_scale_sweep.py --server-counts 4 8 --max-requests 5000
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()


RESULTS_DIR = SCRIPT_DIR / "results"


def kill_all_processes() -> None:
    """Kill all related processes."""
    # Be specific to avoid killing parent scripts
    for pattern in ["fake_server.py", "launch_router.sh", "launch_fake_servers.sh", "sglang_router.launch_router"]:
        try:
            subprocess.run(
                ["pkill", "-9", "-f", pattern],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass


def is_experiment_complete(name: str) -> bool:
    """Check if experiment already completed successfully."""
    complete_file = RESULTS_DIR / name / "complete.txt"
    return complete_file.exists()


def mark_experiment_complete(name: str) -> None:
    """Mark experiment as complete."""
    complete_file = RESULTS_DIR / name / "complete.txt"
    complete_file.parent.mkdir(parents=True, exist_ok=True)
    with open(complete_file, "w") as f:
        f.write(f"Completed at {datetime.now().isoformat()}\n")


def run_experiment(
    num_servers: int,
    rate: float,
    duration_secs: int,
    trace: str,
    tpot_buckets: list,
    max_retries: int = 3,
    retry_wait: int = 30,
) -> bool:
    """Run a single experiment with retry logic. Returns success status."""
    name = f"{num_servers}srv_{int(rate)}rate"
    # Compute max_requests from rate and duration
    max_requests = int(rate * duration_secs)

    # Check if already complete (resume logic)
    if is_experiment_complete(name):
        print(f"\n{'='*60}")
        print(f"SKIPPING (already complete): {num_servers} servers, rate={rate}")
        print(f"{'='*60}")
        return True

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_experiment.py"),
        "--num-servers", str(num_servers),
        "--rate", str(rate),
        "--max-requests", str(max_requests),
        "--name", name,
        "--trace", trace,
        "--tpot-buckets", *[str(b) for b in tpot_buckets],
    ]

    for attempt in range(1, max_retries + 1):
        print(f"\n{'='*60}")
        print(f"Running: {num_servers} servers, rate={rate} (attempt {attempt}/{max_retries})")
        print(f"{'='*60}")

        result = subprocess.run(cmd, cwd=SCRIPT_DIR)

        if result.returncode == 0:
            # Success - mark complete
            mark_experiment_complete(name)
            return True

        # Failed - kill processes and retry
        if attempt < max_retries:
            print(f"\nExperiment failed. Killing processes and waiting {retry_wait}s before retry...")
            kill_all_processes()
            time.sleep(retry_wait)

    print(f"\nExperiment failed after {max_retries} attempts")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run router scale sweep experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--server-counts",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="List of server counts to test",
    )
    parser.add_argument(
        "--rate-multipliers",
        type=float,
        nargs="+",
        default=[20, 40],
        help="Rate multipliers (rate = num_servers * multiplier)",
    )
    parser.add_argument(
        "--duration-secs",
        type=int,
        default=60,
        help="Duration per experiment in seconds (max_requests = rate * duration)",
    )
    parser.add_argument(
        "--trace",
        type=str,
        default="/sgl-workspace/sglang/SLO-CSim/trace/arxiv/uniform_512_512.csv",
        help="Path to trace file",
    )
    parser.add_argument(
        "--tpot-buckets",
        type=float,
        nargs="+",
        default=[10.0, 20.0, 40.0],
        help="TPOT buckets for SLO tiers (ms)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per experiment",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=30,
        help="Seconds to wait between retries",
    )

    args = parser.parse_args()

    # Summary
    experiments = []
    for num_servers in args.server_counts:
        for mult in args.rate_multipliers:
            rate = num_servers * mult
            experiments.append((num_servers, rate))

    print("="*60)
    print("Router Scale Sweep")
    print("="*60)
    print(f"Server counts: {args.server_counts}")
    print(f"Rate multipliers: {args.rate_multipliers}")
    print(f"Duration: {args.duration_secs}s per experiment")
    print(f"Total experiments: {len(experiments)}")
    print()
    print("Experiments to run:")
    for num_servers, rate in experiments:
        print(f"  - {num_servers} servers @ rate {rate}")
    print("="*60)

    # Run experiments
    results = []
    start_time = time.time()

    for i, (num_servers, rate) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}]", end="")
        success = run_experiment(
            num_servers=num_servers,
            rate=rate,
            duration_secs=args.duration_secs,
            trace=args.trace,
            tpot_buckets=args.tpot_buckets,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
        )
        results.append((num_servers, rate, success))

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("SWEEP COMPLETE")
    print("="*60)
    print(f"Total time: {elapsed:.1f}s")
    print()
    print("Results:")
    for num_servers, rate, success in results:
        status = "OK" if success else "FAILED"
        print(f"  {num_servers} servers @ rate {rate}: {status}")

    failed = sum(1 for _, _, s in results if not s)
    if failed:
        print(f"\n{failed} experiment(s) failed")
        return 1

    print(f"\nAll {len(results)} experiments completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
