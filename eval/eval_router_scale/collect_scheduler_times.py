#!/usr/bin/env python3
"""
Collect scheduler tick times from router logs.

Parses lines like:
    2025-12-10 10:39:29  INFO ... Scheduler tick completed after 182 us

Usage:
    python collect_scheduler_times.py                    # Auto-discover experiments in results/
    python collect_scheduler_times.py --results-dir /path/to/results
    python collect_scheduler_times.py results/4srv_80rate/router_log/router.log  # Explicit file
    python collect_scheduler_times.py --summary          # Summary table
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Default results directory (relative to script location)
DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"


def parse_experiment_name(name: str) -> tuple:
    """Parse experiment name to extract server count and rate.

    Handles patterns like:
        4srv_80rate -> (4, 80.0)
        exp_32srv_320.0rate -> (32, 320.0)
        16srv_320rate -> (16, 320.0)

    Returns (num_servers, rate) or (None, None) if parsing fails.
    """
    # Try pattern: {num}srv_{rate}rate
    match = re.search(r"(\d+)srv[_]?(\d+(?:\.\d+)?)rate", name)
    if match:
        return int(match.group(1)), float(match.group(2))

    # Fallback: extract any two numbers
    nums = re.findall(r"(\d+(?:\.\d+)?)", name)
    if len(nums) >= 2:
        return int(float(nums[0])), float(nums[1])

    return None, None


def extract_sort_key(exp_dir: Path) -> tuple:
    """Extract numeric sort key from experiment directory name."""
    servers, rate = parse_experiment_name(exp_dir.name)
    if servers is not None:
        return (servers, rate)
    return (exp_dir.name,)  # Fallback to alphabetical


def discover_experiments(results_dir: Path) -> List[Path]:
    """Auto-discover experiment directories and find their router logs."""
    log_paths = []

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        return log_paths

    # Collect experiment directories and sort numerically
    exp_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    exp_dirs.sort(key=extract_sort_key)

    for exp_dir in exp_dirs:
        # Check for router log in expected location
        router_log = exp_dir / "router_log" / "router.log"
        if router_log.exists():
            log_paths.append(router_log)
        else:
            # Try alternative locations
            for alt_path in [
                exp_dir / "router.log",
                exp_dir / "logs" / "router.log",
            ]:
                if alt_path.exists():
                    log_paths.append(alt_path)
                    break

    return log_paths


def find_last_worker_sel_line(log_path: Path) -> int:
    """Find the line number of the last [WORKER_SEL] entry.

    Returns the line number (1-indexed) or -1 if not found.
    """
    last_line = -1
    try:
        with open(log_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if "[WORKER_SEL]" in line:
                    last_line = line_num
    except Exception:
        pass
    return last_line


def parse_scheduler_times(log_path: Path) -> List[float]:
    """Parse scheduler tick times from a router log file.

    Only collects times from lines BEFORE the last [WORKER_SEL] entry,
    to exclude idle scheduler ticks after the workload finishes.

    Returns list of times in microseconds.
    """
    # First pass: find the last WORKER_SEL line
    last_worker_sel = find_last_worker_sel_line(log_path)

    pattern = re.compile(r"Scheduler tick completed after (\d+) us")
    times = []

    try:
        with open(log_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                # Stop if we've passed the last WORKER_SEL
                if last_worker_sel > 0 and line_num > last_worker_sel:
                    break

                match = pattern.search(line)
                if match:
                    times.append(float(match.group(1)))
    except Exception as e:
        print(f"Error reading {log_path}: {e}", file=sys.stderr)

    return times


def compute_stats(times: List[float]) -> dict:
    """Compute statistics for a list of times."""
    if not times:
        return {
            "count": 0,
            "mean": 0,
            "std": 0,
            "min": 0,
            "max": 0,
            "p50": 0,
            "p90": 0,
            "p99": 0,
        }

    arr = np.array(times)
    return {
        "count": len(arr),
        "mean": np.mean(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        "p99": np.percentile(arr, 99),
    }


def print_stats(name: str, stats: dict) -> None:
    """Print statistics in a formatted way."""
    print(f"\n{name}")
    print("-" * 40)
    print(f"  Count: {stats['count']}")
    print(f"  Mean:  {stats['mean']:.1f} us")
    print(f"  Std:   {stats['std']:.1f} us")
    print(f"  Min:   {stats['min']:.1f} us")
    print(f"  Max:   {stats['max']:.1f} us")
    print(f"  P50:   {stats['p50']:.1f} us")
    print(f"  P90:   {stats['p90']:.1f} us")
    print(f"  P99:   {stats['p99']:.1f} us")


def main():
    parser = argparse.ArgumentParser(
        description="Collect scheduler tick times from router logs"
    )
    parser.add_argument(
        "log_files",
        type=str,
        nargs="*",
        help="Router log file(s) to parse (supports glob patterns). If not provided, auto-discovers experiments.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=f"Results directory to scan for experiments (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary table for all files",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output as CSV to stdout (instead of table)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="scheduler_times.csv",
        help="Output CSV file path (default: scheduler_times.csv)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to CSV file",
    )

    args = parser.parse_args()

    # Determine log paths
    log_paths = []

    if args.log_files:
        # Explicit files provided - expand glob patterns
        for pattern in args.log_files:
            if "*" in pattern:
                log_paths.extend(Path(".").glob(pattern))
            else:
                log_paths.append(Path(pattern))
    else:
        # Auto-discover experiments
        results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
        log_paths = discover_experiments(results_dir)
        if log_paths:
            print(f"Auto-discovered {len(log_paths)} experiment(s) in {results_dir}", file=sys.stderr)

    if not log_paths:
        print("No log files found", file=sys.stderr)
        return 1

    # Collect stats for each file
    all_stats: List[Tuple[str, dict]] = []

    for log_path in log_paths:
        if not log_path.exists():
            print(f"File not found: {log_path}", file=sys.stderr)
            continue

        times = parse_scheduler_times(log_path)
        stats = compute_stats(times)

        # Extract experiment name from path (e.g., results/4srv_80rate/router_log/router.log -> 4srv_80rate)
        try:
            exp_name = log_path.parts[-3]
        except IndexError:
            exp_name = str(log_path)

        # Parse experiment name to get servers and rate
        servers, rate = parse_experiment_name(exp_name)
        all_stats.append((exp_name, servers, rate, stats))

    # Save to CSV file (default behavior)
    if not args.no_save:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.write("experiment,num_servers,rate,count,mean_us,std_us,min_us,max_us,p50_us,p90_us,p99_us\n")
            for name, servers, rate, stats in all_stats:
                srv_str = str(servers) if servers is not None else ""
                rate_str = f"{rate:.1f}" if rate is not None else ""
                f.write(f"{name},{srv_str},{rate_str},{stats['count']},{stats['mean']:.1f},{stats['std']:.1f},"
                        f"{stats['min']:.1f},{stats['max']:.1f},{stats['p50']:.1f},"
                        f"{stats['p90']:.1f},{stats['p99']:.1f}\n")
        print(f"Saved to: {output_path.absolute()}", file=sys.stderr)

    # Print output
    if args.csv:
        # CSV output to stdout
        print("experiment,num_servers,rate,count,mean_us,std_us,min_us,max_us,p50_us,p90_us,p99_us")
        for name, servers, rate, stats in all_stats:
            srv_str = str(servers) if servers is not None else ""
            rate_str = f"{rate:.1f}" if rate is not None else ""
            print(f"{name},{srv_str},{rate_str},{stats['count']},{stats['mean']:.1f},{stats['std']:.1f},"
                  f"{stats['min']:.1f},{stats['max']:.1f},{stats['p50']:.1f},"
                  f"{stats['p90']:.1f},{stats['p99']:.1f}")
    else:
        # Summary table (default)
        print("\nScheduler Tick Time Summary (microseconds)")
        print("=" * 95)
        print(f"{'Experiment':<20} {'Servers':>8} {'Rate':>8} {'Count':>8} {'Mean':>8} {'P50':>8} {'P90':>8} {'P99':>8} {'Max':>8}")
        print("-" * 95)
        for name, servers, rate, stats in all_stats:
            srv_str = str(servers) if servers is not None else "-"
            rate_str = f"{rate:.0f}" if rate is not None else "-"
            print(f"{name:<20} {srv_str:>8} {rate_str:>8} {stats['count']:>8} {stats['mean']:>8.1f} "
                  f"{stats['p50']:>8.1f} {stats['p90']:>8.1f} {stats['p99']:>8.1f} {stats['max']:>8.1f}")
        print("=" * 95)

    return 0


if __name__ == "__main__":
    sys.exit(main())
