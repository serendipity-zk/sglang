#!/usr/bin/env python3
"""
Aggregate evaluation results from rust_client_output.jsonl files.

Usage:
    python aggregate_results.py
    python aggregate_results.py --results-dir ./results --output ./results/aggregated_results.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_jsonl(jsonl_path: Path) -> List[Dict]:
    """Read JSONL file and return list of records."""
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_metrics(records: List[Dict]) -> Dict:
    """Compute SLO metrics from request records.

    Note: Failed requests count as SLO violations.
    """
    if not records:
        return None

    total = len(records)
    successful = [r for r in records if r.get("status") == "SUCCESS"]
    failed = [r for r in records if r.get("status") != "SUCCESS"]
    success_count = len(successful)
    fail_count = len(failed)

    # Overall SLO attainment: failed requests count as violations
    # attainment = (successful requests with slo_satisfied=true) / total_requests
    slo_satisfied = sum(1 for r in successful if r.get("slo_satisfied", False))
    attainment = slo_satisfied / total if total > 0 else 0.0

    # Per-tier attainment (group by target_tpot_ms)
    # Failed requests are counted in their tier as violations
    tier_data = {}

    # Count successful requests by tier
    for r in successful:
        tpot = r.get("target_tpot_ms")
        if tpot is not None:
            tier_key = f"tier_{int(tpot)}ms"
            if tier_key not in tier_data:
                tier_data[tier_key] = {"satisfied": 0, "total": 0}
            tier_data[tier_key]["total"] += 1
            if r.get("slo_satisfied", False):
                tier_data[tier_key]["satisfied"] += 1

    # Count failed requests by tier (as violations)
    for r in failed:
        tpot = r.get("target_tpot_ms")
        if tpot is not None:
            tier_key = f"tier_{int(tpot)}ms"
            if tier_key not in tier_data:
                tier_data[tier_key] = {"satisfied": 0, "total": 0}
            tier_data[tier_key]["total"] += 1
            # Failed requests don't increment satisfied count

    tier_attainment = {}
    for tier, data in tier_data.items():
        tier_attainment[tier] = data["satisfied"] / data["total"] if data["total"] > 0 else 0.0

    # TTFT statistics (from successful requests only)
    ttft_values = [r["ttft_ms"] for r in successful if r.get("ttft_ms") is not None]
    ttft_stats = {}
    if ttft_values:
        ttft_stats = {
            "ttft_mean": np.mean(ttft_values),
            "ttft_p50": np.percentile(ttft_values, 50),
            "ttft_p90": np.percentile(ttft_values, 90),
            "ttft_p99": np.percentile(ttft_values, 99),
        }

    return {
        "total_requests": total,
        "success_count": success_count,
        "fail_count": fail_count,
        "fail_rate": fail_count / total if total > 0 else 0.0,
        "attainment": attainment,
        **tier_attainment,
        **ttft_stats,
    }


def aggregate_results(results_dir: str, output_path: str) -> pd.DataFrame:
    """Aggregate all rust_client_output.jsonl files into a single DataFrame."""
    results_dir = Path(results_dir)
    all_data = []

    # Find all jsonl files: results/{trace}_rate{rate}/{config}/client_log/rust_client_output.jsonl
    jsonl_files = list(results_dir.glob("*/*/client_log/rust_client_output.jsonl"))
    print(f"Found {len(jsonl_files)} result files")

    for jsonl_path in sorted(jsonl_files):
        # Extract trace_rate and config from path
        # Path: results/{trace}_rate{rate}/{config}/client_log/rust_client_output.jsonl
        config = jsonl_path.parts[-3]
        trace_rate = jsonl_path.parts[-4]

        # Parse trace_rate: "sharegpt_rate250" -> trace="sharegpt", rate=250
        if "_rate" in trace_rate:
            trace, rate_str = trace_rate.rsplit("_rate", 1)
            rate = int(rate_str)
        else:
            trace = trace_rate
            rate = 0

        try:
            records = parse_jsonl(jsonl_path)
        except Exception as e:
            print(f"  Skipping {jsonl_path}: {e}")
            continue

        if not records:
            print(f"  Skipping {trace}/{config}: empty JSONL")
            continue

        metrics = compute_metrics(records)
        if metrics is None:
            continue

        metrics["trace"] = trace
        metrics["rate"] = rate
        metrics["config"] = config
        all_data.append(metrics)
        print(f"  Processed {trace}_rate{rate}/{config}: {metrics['total_requests']} requests, attainment={metrics['attainment']:.3f}")

    if not all_data:
        print("No data found!")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Reorder columns
    base_cols = ["trace", "rate", "config", "total_requests", "success_count", "fail_count", "attainment", "fail_rate"]
    tier_cols = sorted([c for c in df.columns if c.startswith("tier_")])
    ttft_cols = [c for c in df.columns if c.startswith("ttft_")]
    other_cols = [c for c in df.columns if c not in base_cols + tier_cols + ttft_cols]
    final_cols = base_cols + tier_cols + ttft_cols + other_cols
    df = df[[c for c in final_cols if c in df.columns]]

    # Sort by trace, rate, config
    df.sort_values(["trace", "rate", "config"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Write to CSV
    df.to_csv(output_path, index=False)
    print(f"\nWrote {len(df)} rows to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Aggregate SLO config evaluation results")
    parser.add_argument("--results-dir", default="results", help="Path to results directory")
    parser.add_argument("--output", default="results/aggregated_results.csv", help="Output CSV path")
    args = parser.parse_args()

    aggregate_results(args.results_dir, args.output)


if __name__ == "__main__":
    main()
