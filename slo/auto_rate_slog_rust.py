#!/usr/bin/env python3
"""
Adaptive rate sampler using the Rust streaming client backend.

This script reuses the three-phase search idea from auto_rate_slog.py but
invokes the high-performance Rust client instead of the Python aiohttp backend.

Key differences from auto_rate_slog.py:
- Uses Rust binary for request execution (lower overhead, no GIL)
- Subprocess-based invocation rather than direct function call
- Same JSONL output format for compatibility with summarize_slo()
"""

import argparse
import csv
import json
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RUST_CLIENT_DIR = os.path.join(SCRIPT_DIR, "rust_client")
RUST_BINARY = os.path.join(RUST_CLIENT_DIR, "target", "release", "slo_runner")
RUN_HISTORY_NAME = "run_history.json"


def format_slo_tier_label(tpot_ms: Optional[float]) -> Optional[str]:
    """Human-readable label for grouping requests by TPOT target."""
    if tpot_ms is None:
        return None
    try:
        value = float(tpot_ms)
    except (TypeError, ValueError):
        return None
    if abs(value - round(value)) < 1e-6:
        value_str = f"{int(round(value))}"
    else:
        value_str = f"{value:.1f}"
    return f"{value_str} ms"


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
    return os.path.join(SCRIPT_DIR, "auto_slo_rust", timestamp)


def ensure_rust_binary() -> str:
    """Build the Rust binary if needed and return its path."""
    if not os.path.exists(RUST_BINARY):
        print("[auto-slog-rust] Rust binary not found, building...")
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=RUST_CLIENT_DIR,
            check=True,
        )
    else:
        # Check if sources are newer than binary
        src_dir = os.path.join(RUST_CLIENT_DIR, "src")
        cargo_toml = os.path.join(RUST_CLIENT_DIR, "Cargo.toml")
        binary_mtime = os.path.getmtime(RUST_BINARY)
        
        needs_rebuild = False
        for root, _, files in os.walk(src_dir):
            for f in files:
                if os.path.getmtime(os.path.join(root, f)) > binary_mtime:
                    needs_rebuild = True
                    break
            if needs_rebuild:
                break
        
        if os.path.getmtime(cargo_toml) > binary_mtime:
            needs_rebuild = True
        
        if needs_rebuild:
            print("[auto-slog-rust] Source files changed, rebuilding...")
            subprocess.run(
                ["cargo", "build", "--release"],
                cwd=RUST_CLIENT_DIR,
                check=True,
            )
    
    return RUST_BINARY


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
        "backend": "rust",
        "script_paths": {
            "launch_auto_test_rust": read_script(os.path.join(SCRIPT_DIR, "launch_auto_test_rust.sh")),
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
            "max_requests": args.max_requests,
            "target_attainment": args.target_attainment,
            "start_rate": args.start_rate,
            "single_rate": args.single_rate,
            "rate_list": args.rate_list,
            "max_probes": args.max_probes,
            "precision": args.precision,
            "slo_use_detokenize_time": args.slo_use_detokenize_time,
            "elapsed_dump_path": args.elapsed_dump_path,
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
    p = argparse.ArgumentParser(description="Adaptive rate sampler using Rust streaming client")
    p.add_argument("--trace", required=True, help="CSV trace file with arrival times (ms)")
    p.add_argument("--text-file", required=True, help="Large text file to build token pool from")
    p.add_argument("--tokenizer", required=True, help="Tokenizer path/name (HF) to use")
    p.add_argument("--base-url", default="http://0.0.0.0:40000/v1", help="Base URL")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    p.add_argument("--max-requests", type=int, default=0, help="Limit number of requests (0 for all)")
    p.add_argument("--target-attainment", type=float, default=0.99,
                   help="Desired SLO attainment (0-1) to bracket the knee")
    p.add_argument("--start-rate", type=float, default=1.0,
                   help="Initial rate guess (arrival scaling factor)")
    p.add_argument("--single-rate", type=float, default=None,
                   help="Run at a fixed rate (skips adaptive search). "
                        "If not specified, uses adaptive rate search.")
    p.add_argument("--rate-list", type=str, default=None,
                   help="Comma-separated list of rates to test sequentially "
                        "(e.g., '100,150,200,250'). Skips adaptive search.")
    p.add_argument("--max-probes", type=int, default=12,
                   help="Maximum number of test runs during adaptive search (including tail samples)")
    p.add_argument("--precision", type=float, default=0.10,
                   help="Stop binary search when right-left <= precision*left")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store per-run logs and the curve image "
             "(default: <script-dir>/auto_slo_rust/<timestamp_PST>)",
    )
    p.add_argument(
        "--resume-folder",
        default=None,
        help="Resume from an existing run folder; previously completed rates will be skipped",
    )
    p.add_argument("--plot-path",
                   help="Optional custom path for the generated SLOG curve image "
                        "(defaults to <output-dir>/slog_curve.png)")
    p.add_argument("--slo-use-detokenize-time", action="store_true",
                   help="Use detokenize time for SLO calculation instead of default")
    p.add_argument("--elapsed-dump-path",
                   help="Path to dump elapsed timelines pickle file (optional)")
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

                # Check for slack attainment (tpot_with_100_slack field)
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
    plt.title("Adaptive rate sweep (Rust client)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def run_single_rate_rust(args, rate: float, logs_dir: str, rust_binary: str) -> Dict[str, Optional[float]]:
    """Run the Rust client once at the given rate and return stats."""
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_rate = f"{rate:.4f}".replace(".", "p")
    log_path = os.path.join(logs_dir, f"rate_{safe_rate}_{timestamp}.jsonl")
    ans_log_path = os.path.join(logs_dir, f"rate_{safe_rate}_{timestamp}.ans")

    print(f"\n[auto-slog-rust] Running rate={rate:.4f}, log={log_path}", flush=True)

    # Build command for Rust binary
    cmd = [
        rust_binary,
        "--trace", args.trace,
        "--text-file", args.text_file,
        "--tokenizer", args.tokenizer,
        "--base-url", args.base_url,
        "--model", args.model,
        "--rate", str(rate),
        "--temperature", str(args.temperature),
        "--log-path", log_path,
        "--ans-log-path", ans_log_path,
    ]

    if args.max_requests > 0:
        cmd.extend(["--max-requests", str(args.max_requests)])

    if args.slo_use_detokenize_time:
        cmd.append("--slo-use-detokenize-time")

    if args.elapsed_dump_path:
        # Use a per-run elapsed dump path
        elapsed_path = os.path.join(logs_dir, f"rate_{safe_rate}_{timestamp}_elapsed.pkl")
        cmd.extend(["--elapsed-dump-path", elapsed_path])

    # Run Rust binary
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[auto-slog-rust] Rust client exited with code {result.returncode}", flush=True)
        if result.stderr:
            print(f"[auto-slog-rust] stderr: {result.stderr[:500]}", flush=True)

    # Count completed/failed from log file
    submitted = 0
    completed = 0
    failed = 0
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    submitted += 1
                    if record.get("status") == "SUCCESS":
                        completed += 1
                    else:
                        failed += 1
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass

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
        f"[auto-slog-rust] Submitted={submitted}, Completed={completed}, Failed={failed}, "
        f"SLO attainment={attainment if attainment is not None else 'N/A'}, "
        f"SLO+100ms={slack_attainment if slack_attainment is not None else 'N/A'}, "
        f"Tiers={fmt_tiers(tier_stats)}, Tiers+100ms={fmt_tiers(slack_tier_stats)}",
        flush=True,
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
    rust_binary: str,
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
                f"[auto-slog-rust] Reusing existing run rate={rate:.4f} "
                f"(order={existing.get('order', 'N/A')}, attainment={existing.get('attainment')})"
            )
            return existing["attainment"] if existing["attainment"] is not None else 0.0
        result = run_single_rate_rust(args, rate, logs_dir, rust_binary)
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

    # Ensure Rust binary is built
    rust_binary = ensure_rust_binary()
    print(f"[auto-slog-rust] Using Rust binary: {rust_binary}")

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

    if args.rate_list is not None:
        # Rate list mode - iterate through provided rates
        rates = [float(r.strip()) for r in args.rate_list.split(",") if r.strip()]
        print(f"[auto-slog-rust] Rate list mode: testing {len(rates)} rates: {rates}")
        history = []
        for idx, rate in enumerate(rates, start=1):
            result = run_single_rate_rust(args, rate, full_log_dir, rust_binary)
            result["order"] = idx
            history.append(result)
            persist_run_history(history_path, history)
        # Use the rate with highest attainment >= target as knee, or max tested rate
        passing = [h for h in history if h["attainment"] is not None and h["attainment"] >= args.target_attainment]
        knee_rate = max(h["rate"] for h in passing) if passing else rates[-1]
    elif args.single_rate is not None:
        # Single rate mode - run once and exit
        print(f"[auto-slog-rust] Single rate mode: running at rate={args.single_rate}")
        result = run_single_rate_rust(args, args.single_rate, full_log_dir, rust_binary)
        result["order"] = 1
        history = [result]
        persist_run_history(history_path, history)
        knee_rate = args.single_rate
    else:
        # Adaptive search mode (existing behavior)
        search_result = adaptive_rate_search(args, full_log_dir, history_path, existing_history, rust_binary)
        history = search_result["history"]
        knee_rate = search_result["knee_rate"]

    elapsed = time.time() - start

    # Print summary table
    print("\n=== Adaptive rate sweep summary (Rust client) ===", flush=True)
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
            f"log={entry['log_path']}",
            flush=True,
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

            print(f"     tiers: {fmt(tier_stats)}", flush=True)
            print(f"     tiers+100ms: {fmt(slack_tier_stats)}", flush=True)

    # Render curve
    render_curve(history, args.target_attainment, plot_path)
    csv_path = write_rate_csv(history, args.output_dir)
    print(f"\nEstimated knee rate: {knee_rate:.4f}", flush=True)
    print(f"SLOG curve saved to: {plot_path}", flush=True)
    if csv_path:
        print(f"Per-rate tier summary CSV: {csv_path}", flush=True)
    print(f"Total elapsed: {elapsed/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
