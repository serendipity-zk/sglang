#!/usr/bin/env python3
"""
Adaptive rate sampler for issue_stream_aiohttp.

This script reuses the three-phase search idea from exp/smart_perf.py to
automatically probe request rates, find the knee where SLO attainment drops,
and emit a SLO attainment curve.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python <3.9
    ZoneInfo = None  # type: ignore

import matplotlib

# Headless-friendly backend for plot export
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from issue_stream_aiohttp import format_slo_tier_label, run_trace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RUN_HISTORY_NAME = "run_history.json"


def tier_sort_key(label: str) -> float:
    try:
        return float(label.split()[0])
    except (ValueError, IndexError):
        return float("inf")


def pst_now() -> datetime:
    """Return the current time in the America/Los_Angeles timezone."""
    try:
        tz = ZoneInfo("America/Los_Angeles") if ZoneInfo else timezone(timedelta(hours=-8))
    except Exception:
        tz = timezone(timedelta(hours=-8))
    return datetime.now(tz)


def default_output_dir() -> str:
    timestamp = pst_now().strftime("%Y%m%d_%H%M%S_%Z")
    return os.path.join(SCRIPT_DIR, "auto_slo", timestamp)


def capture_git_state(output_dir: str) -> None:
    """Persist the current git commit and diff for traceability."""
    commit_path = os.path.join(output_dir, "git_commit.txt")
    diff_path = os.path.join(output_dir, "git_diff.patch")

    commit_id: Optional[str] = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        commit_id = result.stdout.strip()
        with open(commit_path, "w", encoding="utf-8") as f:
            f.write(commit_id + "\n")
    except Exception as e:
        with open(commit_path, "w", encoding="utf-8") as f:
            f.write(f"Unable to read git commit: {e}\n")

    diff_cmd = ["git", "diff", commit_id] if commit_id else ["git", "diff"]
    try:
        diff_result = subprocess.run(
            diff_cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        with open(diff_path, "w", encoding="utf-8") as f:
            if diff_result.stdout:
                f.write(diff_result.stdout)
            if diff_result.stderr:
                f.write("\n# stderr\n")
                f.write(diff_result.stderr)
    except Exception as e:
        with open(diff_path, "w", encoding="utf-8") as f:
            f.write(f"Unable to capture git diff: {e}\n")


def write_run_scaffold(args, plot_path: str, full_log_dir: str, start_ts: datetime) -> None:
    """Write config, git info, and start time artifacts before tests run."""
    def read_script(path: str) -> Dict[str, str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            content = f"# Unable to read: {e}"
        return {"path": path, "content": content}

    config_payload = {
        "start_time_pst": start_ts.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "start_time_iso": start_ts.isoformat(),
        "output_dir": args.output_dir,
        "full_log_dir": full_log_dir,
        "plot_path": plot_path,
        "script_paths": {
            "launch_auto_test": read_script(os.path.join(SCRIPT_DIR, "launch_auto_test.sh")),
            "launch_router": read_script(os.path.join(SCRIPT_DIR, "launch_router.sh")),
            "launch_server": read_script(os.path.join(SCRIPT_DIR, "launch_server.sh")),
        },
        "cli_args": {
            "trace": args.trace,
            "text_file": args.text_file,
            "tokenizer": args.tokenizer,
            "base_url": args.base_url,
            "model": args.model,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_requests": args.max_requests,
            "timeout": args.timeout,
            "num_workers": args.num_workers,
            "concurrency_per_worker": args.concurrency_per_worker,
            "target_attainment": args.target_attainment,
            "start_rate": args.start_rate,
            "max_probes": args.max_probes,
            "precision": args.precision,
            "ui": args.ui,
        },
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    with open(os.path.join(args.output_dir, "start_time_pst.txt"), "w", encoding="utf-8") as f:
        f.write(config_payload["start_time_pst"] + "\n")

    capture_git_state(args.output_dir)


def load_run_history(folder: str) -> List[Dict[str, Optional[float]]]:
    path = os.path.join(folder, RUN_HISTORY_NAME)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
        # Normalize ordering to be sequential
        history_sorted = sorted(history, key=lambda h: h.get("order", 0))
        for idx, entry in enumerate(history_sorted, start=1):
            entry["order"] = idx
        return history_sorted
    except Exception:
        return []


def persist_run_history(path: str, history: List[Dict[str, Optional[float]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive rate sampler for issue_stream_aiohttp")
    p.add_argument("--trace", required=True, help="CSV trace file with arrival times (ms)")
    p.add_argument("--text-file", required=True, help="Large text file to build token pool from")
    p.add_argument("--tokenizer", required=True, help="Tokenizer path/name (HF) to use")
    p.add_argument("--base-url", default="http://0.0.0.0:40000/v1", help="Base URL")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    p.add_argument("--max-requests", type=int, default=0, help="Limit number of requests (0 for all)")
    p.add_argument("--timeout", type=float, default=0, help="Per-request timeout seconds (0 to disable)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Number of worker processes (0 for auto=min(256, cpu_count))")
    p.add_argument("--concurrency-per-worker", type=int, default=50,
                   help="Max concurrent requests per worker (default: 50)")
    p.add_argument("--target-attainment", type=float, default=0.99,
                   help="Desired SLO attainment (0-1) to bracket the knee")
    p.add_argument("--start-rate", type=float, default=1.0,
                   help="Initial rate guess (arrival scaling factor)")
    p.add_argument("--max-probes", type=int, default=12,
                   help="Maximum number of test runs during adaptive search (including tail samples)")
    p.add_argument("--precision", type=float, default=0.10,
                   help="Stop binary search when right-left <= precision*left")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store per-run logs and the curve image "
             "(default: <script-dir>/auto_slo/<timestamp_PST>)",
    )
    p.add_argument(
        "--resume-folder",
        default=None,
        help="Resume from an existing run folder; previously completed rates will be skipped",
    )
    p.add_argument("--plot-path",
                   help="Optional custom path for the generated SLOG curve image "
                        "(defaults to <output-dir>/slog_curve.png)")
    p.add_argument("--ui", action="store_true", help="Enable live dashboard UI for each run")
    return p.parse_args()


def summarize_slo(log_path: str) -> Dict[str, Optional[float]]:
    """Compute request-level and per-tier SLO attainment from a log file."""
    total = 0
    satisfied = 0
    slack_total = 0
    slack_satisfied = 0

    tier_stats: Dict[str, Dict[str, int]] = {}
    slack_tier_stats: Dict[str, Dict[str, int]] = {}

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("status") != "SUCCESS":
                    continue

                tier_label = format_slo_tier_label(record.get("target_tpot_ms"))

                if "slo_satisfied" in record:
                    total += 1
                    if record.get("slo_satisfied"):
                        satisfied += 1
                    if tier_label:
                        tier_stats.setdefault(tier_label, {"total": 0, "satisfied": 0})
                        tier_stats[tier_label]["total"] += 1
                        if record.get("slo_satisfied"):
                            tier_stats[tier_label]["satisfied"] += 1

                if "tpot_with_100_slack" in record:
                    slack_total += 1
                    if record.get("tpot_with_100_slack"):
                        slack_satisfied += 1
                    if tier_label:
                        slack_tier_stats.setdefault(tier_label, {"total": 0, "satisfied": 0})
                        slack_tier_stats[tier_label]["total"] += 1
                        if record.get("tpot_with_100_slack"):
                            slack_tier_stats[tier_label]["satisfied"] += 1
    except FileNotFoundError:
        return {
            "attainment": None,
            "slack_attainment": None,
            "tier_stats": {},
            "slack_tier_stats": {},
        }

    attainment = satisfied / total if total > 0 else None
    slack_attainment = slack_satisfied / slack_total if slack_total > 0 else None
    tier_attainment_values = [
        stats["satisfied"] / stats["total"] for stats in tier_stats.values() if stats["total"] > 0
    ]
    if tier_attainment_values:
        attainment = min(tier_attainment_values)
    slack_tier_attainment_values = [
        stats["satisfied"] / stats["total"] for stats in slack_tier_stats.values() if stats["total"] > 0
    ]
    if slack_tier_attainment_values:
        slack_attainment = min(slack_tier_attainment_values)
    return {
        "attainment": attainment,
        "slack_attainment": slack_attainment,
        "tier_stats": tier_stats,
        "slack_tier_stats": slack_tier_stats,
    }


def render_curve(history: List[Dict[str, float]], target: float, output_path: str) -> None:
    """Plot rate vs attainment and save to output_path."""
    if not history:
        return
    history_sorted = sorted(history, key=lambda h: h["rate"])
    rates = [h["rate"] for h in history_sorted]
    att = [h["attainment"] if h["attainment"] is not None else 0.0 for h in history_sorted]
    colors = [
        "green" if (h["attainment"] is not None and h["attainment"] >= target) else "red"
        for h in history_sorted
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(rates, att, "--", alpha=0.4, color="gray", label="Path")
    plt.scatter(rates, att, c=colors, s=80, zorder=3, label="Samples")
    for h in history_sorted:
        plt.annotate(str(h["order"]), (h["rate"], h["attainment"] if h["attainment"] is not None else 0.0),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    plt.axhline(target, color="orange", linestyle=":", label=f"Target {target:.2f}")
    plt.ylim(0, 1.05)
    plt.xlabel("Arrival rate scale (relative to trace)")
    plt.ylabel("Request-level SLO attainment")
    plt.title("Adaptive rate sweep (issue_stream_aiohttp)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def run_single_rate(args, rate: float, logs_dir: str) -> Dict[str, Optional[float]]:
    """Run issue_stream_aiohttp once at the given rate and return stats."""
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_rate = f"{rate:.4f}".replace(".", "p")
    log_path = os.path.join(logs_dir, f"rate_{safe_rate}_{timestamp}.jsonl")

    num_workers = args.num_workers if args.num_workers > 0 else min(256, mp.cpu_count())

    print(f"\n[auto-slog] Running rate={rate:.4f}, log={log_path}")
    submitted, completed, failed = run_trace(
        trace_csv=args.trace,
        text_file=args.text_file,
        tokenizer_path=args.tokenizer,
        base_url=args.base_url,
        model=args.model,
        rate=rate,
        temperature=args.temperature,
        seed=args.seed,
        max_requests=args.max_requests,
        timeout=args.timeout,
        num_workers=num_workers,
        concurrency_per_worker=args.concurrency_per_worker,
        enable_ui=args.ui,
        log_path=log_path,
    )

    slo_stats = summarize_slo(log_path)
    attainment = slo_stats["attainment"]
    slack_attainment = slo_stats["slack_attainment"]
    tier_stats = slo_stats["tier_stats"]
    slack_tier_stats = slo_stats["slack_tier_stats"]

    def fmt_tiers(ts: Dict[str, Dict[str, int]]) -> str:
        parts = []
        for tier in sorted(ts.keys(), key=tier_sort_key):
            total = ts[tier]["total"]
            ok = ts[tier]["satisfied"]
            pct = ok / total if total else 0.0
            parts.append(f"{tier}:{ok}/{total}({pct:.3f})")
        return "; ".join(parts) if parts else "N/A"

    print(
        f"[auto-slog] Submitted={submitted}, Completed={completed}, Failed={failed}, "
        f"SLO attainment={attainment if attainment is not None else 'N/A'}, "
        f"SLO+100ms={slack_attainment if slack_attainment is not None else 'N/A'}, "
        f"Tiers={fmt_tiers(tier_stats)}, Tiers+100ms={fmt_tiers(slack_tier_stats)}"
    )

    return {
        "rate": rate,
        "attainment": attainment,
        "slack_attainment": slack_attainment,
        "tier_stats": tier_stats,
        "slack_tier_stats": slack_tier_stats,
        "log_path": log_path,
        "submitted": submitted,
        "completed": completed,
        "failed": failed,
    }


def adaptive_rate_search(
    args,
    logs_dir: str,
    history_path: str,
    initial_history: List[Dict[str, Optional[float]]],
) -> Dict[str, Optional[float]]:
    """Three-phase adaptive search (bracket, binary search, tail sampling)."""
    history: List[Dict[str, Optional[float]]] = list(initial_history)
    target = args.target_attainment
    max_probes = max(1, args.max_probes)
    precision = max(0.0, args.precision)

    def can_probe() -> bool:
        return len(history) < max_probes

    def find_existing(rate: float) -> Optional[Dict[str, Optional[float]]]:
        for h in history:
            if abs(rate - h["rate"]) / max(rate, 1e-6) < 0.02:
                return h
        return None

    def probe(rate: float) -> float:
        if not can_probe():
            last = history[-1] if history else {"attainment": 0.0}
            return last["attainment"] or 0.0
        existing = find_existing(rate)
        if existing is not None:
            print(
                f"[auto-slog] Reusing existing run rate={rate:.4f} "
                f"(order={existing.get('order', 'N/A')}, attainment={existing.get('attainment')})"
            )
            return existing["attainment"] if existing["attainment"] is not None else 0.0
        result = run_single_rate(args, rate, logs_dir)
        result["order"] = len(history) + 1
        history.append(result)
        persist_run_history(history_path, history)
        return result["attainment"] if result["attainment"] is not None else 0.0

    estimate_rate = max(1e-6, args.start_rate)
    val = probe(estimate_rate)
    left = right = None

    # Bracketing
    if val >= target:
        left = estimate_rate
        curr = estimate_rate * 1.5
        while can_probe():
            v = probe(curr)
            if v < target:
                right = curr
                break
            left = curr
            curr *= 1.5
    else:
        right = estimate_rate
        curr = estimate_rate * 0.5
        while can_probe():
            v = probe(curr)
            if v >= target:
                left = curr
                break
            right = curr
            curr *= 0.5

    # Binary search within bracket
    while (
        can_probe()
        and left is not None
        and right is not None
        and (right - left) > (precision * max(left, 1e-6))
    ):
        mid = (left + right) / 2
        v = probe(mid)
        if v >= target:
            left = mid
        else:
            right = mid

    knee = left if left is not None else estimate_rate

    # Tail samples around the knee for plotting texture
    for factor in (0.8, 1.25, 1.6):
        if not can_probe():
            break
        candidate = knee * factor
        # Avoid near-duplicate rates
        if any(abs(candidate - h["rate"]) / max(candidate, 1e-6) < 0.02 for h in history):
            continue
        probe(candidate)

    return {"knee_rate": knee, "history": history}


def write_rate_csv(history: List[Dict[str, Optional[float]]], output_dir: str) -> Optional[str]:
    """Write detailed per-rate, per-tier attainment to CSV."""
    if not history:
        return None
    path = os.path.join(output_dir, "rate_tier_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "order",
                "rate",
                "attainment",
                "attainment_slack_100ms",
                "tier_label",
                "tier_attainment",
                "tier_attainment_slack_100ms",
                "log_path",
            ]
        )
        for entry in sorted(history, key=lambda h: h["order"]):
            base_row = [
                entry["order"],
                entry["rate"],
                entry["attainment"],
                entry["slack_attainment"],
                None,
                None,
                None,
                entry["log_path"],
            ]
            tier_stats = entry.get("tier_stats") or {}
            slack_stats = entry.get("slack_tier_stats") or {}

            if not tier_stats and not slack_stats:
                writer.writerow(base_row)
                continue

            all_tiers = set(tier_stats.keys()) | set(slack_stats.keys())
            for tier in sorted(all_tiers, key=tier_sort_key):
                t = tier_stats.get(tier, {"total": 0, "satisfied": 0})
                s = slack_stats.get(tier, {"total": 0, "satisfied": 0})
                t_att = t["satisfied"] / t["total"] if t["total"] else None
                s_att = s["satisfied"] / s["total"] if s["total"] else None
                writer.writerow(
                    [
                        entry["order"],
                        entry["rate"],
                        entry["attainment"],
                        entry["slack_attainment"],
                        tier,
                        t_att,
                        s_att,
                        entry["log_path"],
                    ]
                )
    return path


def main():
    args = parse_args()

    # Compute default output dir with timestamp if not provided
    args.output_dir = (
        os.path.abspath(args.resume_folder)
        if args.resume_folder
        else (os.path.abspath(args.output_dir) if args.output_dir else default_output_dir())
    )
    full_log_dir = os.path.join(args.output_dir, "full_log")
    history_path = os.path.join(args.output_dir, RUN_HISTORY_NAME)
    existing_history = load_run_history(args.output_dir)

    # Ensure output dir exists and determine plot path
    os.makedirs(full_log_dir, exist_ok=True)
    plot_path = args.plot_path or os.path.join(args.output_dir, "slog_curve.png")
    start_ts = pst_now()
    write_run_scaffold(args, plot_path, full_log_dir, start_ts)

    start = time.time()
    search_result = adaptive_rate_search(args, full_log_dir, history_path, existing_history)
    elapsed = time.time() - start

    history = search_result["history"]
    knee_rate = search_result["knee_rate"]

    # Print summary table
    print("\n=== Adaptive rate sweep summary ===")
    for entry in sorted(history, key=lambda h: h["order"]):
        att = entry["attainment"]
        slack_att = entry["slack_attainment"]
        att_str = f"{att:.3f}" if att is not None else "N/A"
        slack_str = f"{slack_att:.3f}" if slack_att is not None else "N/A"
        tier_stats = entry.get("tier_stats") or {}
        slack_tier_stats = entry.get("slack_tier_stats") or {}
        print(
            f"[{entry['order']:02d}] rate={entry['rate']:.4f} "
            f"SLO={att_str} "
            f"SLO+100ms={slack_str} "
            f"log={entry['log_path']}"
        )
        if tier_stats or slack_tier_stats:
            def fmt(ts):
                parts = []
                for tier in sorted(ts.keys(), key=tier_sort_key):
                    total = ts[tier]["total"]
                    ok = ts[tier]["satisfied"]
                    pct = ok / total if total else 0.0
                    parts.append(f"{tier}:{ok}/{total} ({pct:.3f})")
                return "; ".join(parts) if parts else "N/A"

            print(f"     tiers: {fmt(tier_stats)}")
            print(f"     tiers+100ms: {fmt(slack_tier_stats)}")

    # Render curve
    render_curve(history, args.target_attainment, plot_path)
    csv_path = write_rate_csv(history, args.output_dir)
    print(f"\nEstimated knee rate: {knee_rate:.4f}")
    print(f"SLOG curve saved to: {plot_path}")
    if csv_path:
        print(f"Per-rate tier summary CSV: {csv_path}")
    print(f"Total elapsed: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
