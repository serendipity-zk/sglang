#!/usr/bin/env python3
"""
Parse STAT_METRICS from worker logs to extract iteration-level metrics.

Extracts:
- iteration_num: Iteration number
- timestamp: Log timestamp
- tpot_ms: Time per output token
- prefill_tokens: Tokens in prefill
- decode_tokens: Tokens in decode
- token_batch_size: Total tokens (prefill + decode)
- num_requests: Number of requests in batch
- queue_reqs: Requests waiting in queue
- kv_tokens_used: KV cache tokens used
- kv_usage_pct: KV cache usage percentage
- iteration_time_ms: Time between iterations (computed)

Output: CSV file with parsed metrics.

Usage:
    python parse_iteration_logs.py --input worker_log/worker_1_gpu0_p31001.ans --output parsed_metrics.csv
    python parse_iteration_logs.py --results-dir results/  # Parse all rate directories
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional


def parse_timestamp(line: str) -> Optional[datetime]:
    """Extract timestamp from log line."""
    # Format: [2025-12-09 11:22:12]
    match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return None


def parse_time_line(line: str) -> Optional[Dict]:
    """Parse a [TIME] line to extract gpu time and other timing metrics.

    Example line:
    [2025-12-11 02:04:46] [TIME]: since_last=11.027 gpu=8.784 loop=0.814 recv=0.035 ...
    """
    if "[TIME]:" not in line:
        return None

    # Strip ANSI escape codes
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    line = ansi_escape.sub('', line)

    result = {}

    # Parse gpu time (this is the actual iteration/forward pass time in ms)
    gpu_match = re.search(r'gpu=([0-9.]+)', line)
    if gpu_match:
        result["iteration_time_ms"] = float(gpu_match.group(1))

    # Parse since_last (time since last iteration)
    since_last_match = re.search(r'since_last=([0-9.]+)', line)
    if since_last_match:
        result["since_last_ms"] = float(since_last_match.group(1))

    # Parse loop time
    loop_match = re.search(r'loop=([0-9.]+)', line)
    if loop_match:
        result["loop_time_ms"] = float(loop_match.group(1))

    return result if result else None


def parse_stat_metrics_line(line: str) -> Optional[Dict]:
    """Parse a single STAT_METRICS line."""
    # Example line (with ANSI codes stripped):
    # [2025-12-09 11:22:12] STAT_METRICS: Iter:1      | tpot:n/a  | Token:7P+0D=7         | Req:1R+0W      | KV:8 (0.00%)          | Slack:n/a ...

    if "STAT_METRICS:" not in line:
        return None

    # Strip ANSI escape codes
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    line = ansi_escape.sub('', line)

    result = {}

    # Get timestamp
    ts = parse_timestamp(line)
    if ts:
        result["timestamp"] = ts.isoformat()
        result["timestamp_unix"] = ts.timestamp()

    # Parse iteration number
    iter_match = re.search(r'Iter:(\d+)', line)
    if iter_match:
        result["iteration_num"] = int(iter_match.group(1))

    # Parse tpot
    tpot_match = re.search(r'tpot:(\d+)ms', line)
    if tpot_match:
        result["tpot_ms"] = float(tpot_match.group(1))
    elif "tpot:n/a" in line:
        result["tpot_ms"] = None

    # Parse Token: XP+YD=Z
    token_match = re.search(r'Token:(\d+)P\+(\d+)D=(\d+)', line)
    if token_match:
        result["prefill_tokens"] = int(token_match.group(1))
        result["decode_tokens"] = int(token_match.group(2))
        result["token_batch_size"] = int(token_match.group(3))

    # Parse Req: XR+YW
    req_match = re.search(r'Req:(\d+)R\+(\d+)W', line)
    if req_match:
        result["num_requests"] = int(req_match.group(1))
        result["queue_reqs"] = int(req_match.group(2))

    # Parse KV: X (Y%)
    kv_match = re.search(r'KV:(\d+)\s+\(([0-9.]+)%\)', line)
    if kv_match:
        result["kv_tokens_used"] = int(kv_match.group(1))
        result["kv_usage_pct"] = float(kv_match.group(2))

    # Parse Slack
    slack_match = re.search(r'Slack:(-?[0-9.]+)ms', line)
    if slack_match:
        result["min_decode_slack_ms"] = float(slack_match.group(1))
    elif "Slack:n/a" in line:
        result["min_decode_slack_ms"] = None

    return result if result else None


def parse_log_file(log_path: str) -> List[Dict]:
    """Parse all STAT_METRICS and TIME lines from a log file.

    STAT_METRICS and TIME lines alternate - each iteration produces both.
    We merge them together based on their order in the file.
    """
    metrics = []
    pending_time = None  # TIME line waiting to be merged with next STAT_METRICS

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Try parsing as TIME line first
            time_parsed = parse_time_line(line)
            if time_parsed:
                pending_time = time_parsed
                continue

            # Try parsing as STAT_METRICS line
            stat_parsed = parse_stat_metrics_line(line)
            if stat_parsed and "iteration_num" in stat_parsed:
                # Merge with pending TIME data if available
                if pending_time:
                    stat_parsed.update(pending_time)
                    pending_time = None
                metrics.append(stat_parsed)

    # Fallback: compute iteration times from timestamps if not available from TIME lines
    for i in range(1, len(metrics)):
        curr = metrics[i]
        prev = metrics[i - 1]
        if "iteration_time_ms" not in curr or curr["iteration_time_ms"] is None:
            if "timestamp_unix" in curr and "timestamp_unix" in prev:
                curr["iteration_time_ms"] = (curr["timestamp_unix"] - prev["timestamp_unix"]) * 1000

    return metrics


def write_csv(metrics: List[Dict], output_path: str):
    """Write metrics to CSV file."""
    if not metrics:
        print(f"[warning] No metrics to write to {output_path}")
        return

    # Define column order
    columns = [
        "iteration_num",
        "timestamp",
        "iteration_time_ms",
        "since_last_ms",
        "loop_time_ms",
        "tpot_ms",
        "prefill_tokens",
        "decode_tokens",
        "token_batch_size",
        "num_requests",
        "queue_reqs",
        "kv_tokens_used",
        "kv_usage_pct",
        "min_decode_slack_ms",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)

    print(f"[done] Wrote {len(metrics)} rows to {output_path}")


def find_worker_log(rate_dir: str) -> Optional[str]:
    """Find worker log file in rate directory.

    Uses run_info.json to determine the correct log file based on gpu_id and server_port.
    Falls back to first matching file if run_info.json is not available.
    """
    worker_log_dir = os.path.join(rate_dir, "worker_log")
    if not os.path.isdir(worker_log_dir):
        return None

    # Try to read run_info.json to get the correct log file
    run_info_path = os.path.join(rate_dir, "run_info.json")
    if os.path.exists(run_info_path):
        try:
            with open(run_info_path, "r") as f:
                run_info = json.load(f)
            gpu_id = run_info.get("gpu_id")
            server_port = run_info.get("server_port")
            if gpu_id is not None and server_port is not None:
                # Construct expected filename: worker_{gpu_id+1}_gpu{gpu_id}_p{port}.ans
                expected_file = f"worker_{gpu_id + 1}_gpu{gpu_id}_p{server_port}.ans"
                expected_path = os.path.join(worker_log_dir, expected_file)
                if os.path.exists(expected_path):
                    return expected_path
                else:
                    print(f"[warning] Expected log file not found: {expected_path}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[warning] Could not parse run_info.json: {e}")

    # Fallback: return the first matching file
    for f in os.listdir(worker_log_dir):
        if f.startswith("worker_") and f.endswith(".ans"):
            return os.path.join(worker_log_dir, f)

    return None


def compute_rate_statistics(metrics: List[Dict], rate: float) -> Dict:
    """Compute average statistics for a single rate's metrics."""
    if not metrics:
        return {"rate": rate, "num_iterations": 0}

    # Filter valid values for each metric
    iteration_times = [m["iteration_time_ms"] for m in metrics if m.get("iteration_time_ms") is not None]
    num_requests = [m["num_requests"] for m in metrics if m.get("num_requests") is not None]
    kv_tokens = [m["kv_tokens_used"] for m in metrics if m.get("kv_tokens_used") is not None]
    token_batch_sizes = [m["token_batch_size"] for m in metrics if m.get("token_batch_size") is not None]
    prefill_tokens = [m["prefill_tokens"] for m in metrics if m.get("prefill_tokens") is not None]
    decode_tokens = [m["decode_tokens"] for m in metrics if m.get("decode_tokens") is not None]
    kv_usage_pcts = [m["kv_usage_pct"] for m in metrics if m.get("kv_usage_pct") is not None]

    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else None

    def safe_max(lst):
        return max(lst) if lst else None

    def safe_min(lst):
        return min(lst) if lst else None

    return {
        "rate": rate,
        "num_iterations": len(metrics),
        "avg_iteration_time_ms": safe_avg(iteration_times),
        "min_iteration_time_ms": safe_min(iteration_times),
        "max_iteration_time_ms": safe_max(iteration_times),
        "avg_num_requests": safe_avg(num_requests),
        "max_num_requests": safe_max(num_requests),
        "avg_kv_tokens_used": safe_avg(kv_tokens),
        "max_kv_tokens_used": safe_max(kv_tokens),
        "avg_token_batch_size": safe_avg(token_batch_sizes),
        "max_token_batch_size": safe_max(token_batch_sizes),
        "avg_prefill_tokens": safe_avg(prefill_tokens),
        "avg_decode_tokens": safe_avg(decode_tokens),
        "avg_kv_usage_pct": safe_avg(kv_usage_pcts),
        "max_kv_usage_pct": safe_max(kv_usage_pcts),
    }


def parse_trace_dir(trace_dir: str, trace_name: str) -> tuple:
    """Parse all rate directories within a trace directory.

    Returns:
        Tuple of (all_metrics, all_statistics) for this trace
    """
    # Find all rate_* directories
    rate_dirs = []
    for entry in os.listdir(trace_dir):
        if entry.startswith("rate_"):
            rate_path = os.path.join(trace_dir, entry)
            if os.path.isdir(rate_path):
                rate_dirs.append((entry, rate_path))

    rate_dirs.sort(key=lambda x: x[0])

    trace_metrics = []
    trace_statistics = []

    for rate_name, rate_path in rate_dirs:
        log_path = find_worker_log(rate_path)
        if log_path:
            print(f"[parse] Parsing {trace_name}/{rate_name}: {log_path}")
            metrics = parse_log_file(log_path)

            # Add trace and rate to each metric
            rate_value = float(rate_name.replace("rate_", "").replace("p", "."))
            for m in metrics:
                m["trace"] = trace_name
                m["rate"] = rate_value

            # Write per-rate CSV
            csv_path = os.path.join(rate_path, "parsed_metrics.csv")
            write_csv(metrics, csv_path)

            # Compute statistics for this rate
            stats = compute_rate_statistics(metrics, rate_value)
            stats["trace"] = trace_name
            trace_statistics.append(stats)

            trace_metrics.extend(metrics)
        else:
            print(f"[warning] No worker log found in {rate_path}")

    # Write per-trace aggregated CSV
    if trace_metrics:
        trace_csv_path = os.path.join(trace_dir, "all_metrics.csv")
        columns = ["trace", "rate"] + [
            "iteration_num",
            "timestamp",
            "iteration_time_ms",
            "tpot_ms",
            "prefill_tokens",
            "decode_tokens",
            "token_batch_size",
            "num_requests",
            "queue_reqs",
            "kv_tokens_used",
            "kv_usage_pct",
            "min_decode_slack_ms",
        ]
        with open(trace_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(trace_metrics)
        print(f"[done] Trace metrics: {trace_csv_path} ({len(trace_metrics)} rows)")

    # Write per-trace statistics JSON
    if trace_statistics:
        trace_stats_path = os.path.join(trace_dir, "statistics.json")
        with open(trace_stats_path, "w", encoding="utf-8") as f:
            json.dump(trace_statistics, f, indent=2)
        print(f"[done] Trace statistics: {trace_stats_path}")

    return trace_metrics, trace_statistics


def parse_results_dir(results_dir: str, output_dir: Optional[str] = None):
    """Parse all trace and rate directories in results.

    Supports both flat structure (rate_* directly in results_dir) and
    nested structure (trace_name/rate_* in results_dir).
    """
    if output_dir is None:
        output_dir = results_dir

    all_metrics = []
    all_statistics = []

    # Check for trace directories (directories that aren't rate_* or worker_log)
    trace_dirs = []
    rate_dirs = []

    for entry in os.listdir(results_dir):
        entry_path = os.path.join(results_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry.startswith("rate_"):
            rate_dirs.append((entry, entry_path))
        elif entry not in ["worker_log", "__pycache__"]:
            # Check if this directory contains rate_* subdirectories
            has_rates = any(
                e.startswith("rate_") and os.path.isdir(os.path.join(entry_path, e))
                for e in os.listdir(entry_path)
            )
            if has_rates:
                trace_dirs.append((entry, entry_path))

    # If we have trace directories, parse each trace
    if trace_dirs:
        trace_dirs.sort(key=lambda x: x[0])
        for trace_name, trace_path in trace_dirs:
            print(f"\n[trace] Processing trace: {trace_name}")
            trace_metrics, trace_stats = parse_trace_dir(trace_path, trace_name)
            all_metrics.extend(trace_metrics)
            all_statistics.extend(trace_stats)

    # Also handle flat structure (rate_* directly in results_dir)
    elif rate_dirs:
        rate_dirs.sort(key=lambda x: x[0])
        for rate_name, rate_path in rate_dirs:
            log_path = find_worker_log(rate_path)
            if log_path:
                print(f"[parse] Parsing {rate_name}: {log_path}")
                metrics = parse_log_file(log_path)

                rate_value = float(rate_name.replace("rate_", "").replace("p", "."))
                for m in metrics:
                    m["rate"] = rate_value

                csv_path = os.path.join(rate_path, "parsed_metrics.csv")
                write_csv(metrics, csv_path)

                stats = compute_rate_statistics(metrics, rate_value)
                all_statistics.append(stats)
                all_metrics.extend(metrics)
            else:
                print(f"[warning] No worker log found in {rate_path}")

    # Write global aggregated CSV
    if all_metrics:
        agg_csv_path = os.path.join(output_dir, "all_metrics.csv")
        # Check if we have trace column
        has_trace = "trace" in all_metrics[0] if all_metrics else False
        if has_trace:
            columns = ["trace", "rate"]
        else:
            columns = ["rate"]
        columns += [
            "iteration_num",
            "timestamp",
            "iteration_time_ms",
            "tpot_ms",
            "prefill_tokens",
            "decode_tokens",
            "token_batch_size",
            "num_requests",
            "queue_reqs",
            "kv_tokens_used",
            "kv_usage_pct",
            "min_decode_slack_ms",
        ]
        with open(agg_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\n[done] Aggregated metrics: {agg_csv_path} ({len(all_metrics)} rows)")

    # Write global statistics JSON
    if all_statistics:
        stats_path = os.path.join(output_dir, "statistics.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(all_statistics, f, indent=2)
        print(f"[done] Statistics: {stats_path} ({len(all_statistics)} entries)")

        # Print summary table
        has_trace = "trace" in all_statistics[0] if all_statistics else False
        print("\n[summary] Per-rate statistics:")
        if has_trace:
            print(f"{'Trace':>12} {'Rate':>8} {'Iters':>8} {'AvgIterTime':>12} {'AvgReqs':>10} {'AvgKVTokens':>12}")
            print("-" * 68)
            for s in all_statistics:
                iter_time = f"{s['avg_iteration_time_ms']:.2f}" if s.get('avg_iteration_time_ms') else "N/A"
                avg_reqs = f"{s['avg_num_requests']:.1f}" if s.get('avg_num_requests') else "N/A"
                avg_kv = f"{s['avg_kv_tokens_used']:.0f}" if s.get('avg_kv_tokens_used') else "N/A"
                print(f"{s.get('trace', 'N/A'):>12} {s['rate']:>8.1f} {s['num_iterations']:>8} {iter_time:>12} {avg_reqs:>10} {avg_kv:>12}")
        else:
            print(f"{'Rate':>8} {'Iters':>8} {'AvgIterTime':>12} {'AvgReqs':>10} {'AvgKVTokens':>12}")
            print("-" * 54)
            for s in all_statistics:
                iter_time = f"{s['avg_iteration_time_ms']:.2f}" if s.get('avg_iteration_time_ms') else "N/A"
                avg_reqs = f"{s['avg_num_requests']:.1f}" if s.get('avg_num_requests') else "N/A"
                avg_kv = f"{s['avg_kv_tokens_used']:.0f}" if s.get('avg_kv_tokens_used') else "N/A"
                print(f"{s['rate']:>8.1f} {s['num_iterations']:>8} {iter_time:>12} {avg_reqs:>10} {avg_kv:>12}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse STAT_METRICS from worker logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", help="Single worker log file to parse")
    parser.add_argument("--output", "-o", help="Output CSV file (for single file mode)")
    parser.add_argument("--results-dir", "-r",
                        help="Parse all rate directories in results folder")
    args = parser.parse_args()

    if args.results_dir:
        parse_results_dir(args.results_dir)
    elif args.input:
        output = args.output or args.input.replace(".ans", "_metrics.csv")
        metrics = parse_log_file(args.input)
        write_csv(metrics, output)
    else:
        parser.print_help()
        print("\nError: Must specify either --input or --results-dir")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
