#!/usr/bin/env python3
"""
Aggregate evaluation results from all traces and methods into a single CSV.

Usage:
    python aggregate_results.py
    python aggregate_results.py --results-dir ./results --output ./results/aggregated.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def aggregate_results(results_dir: str, output_path: str) -> pd.DataFrame:
    """
    Aggregate all rate_tier_summary.csv files into a single DataFrame.

    Args:
        results_dir: Path to results directory
        output_path: Path to output CSV file

    Returns:
        Aggregated DataFrame
    """
    results_dir = Path(results_dir)
    all_data = []

    # Find all rate_tier_summary.csv files
    summary_files = list(results_dir.glob("*/*/client_log/rate_tier_summary.csv"))
    print(f"Found {len(summary_files)} result files")

    for summary_csv in summary_files:
        # Extract trace and method from path: results/{trace}/{method}/client_log/...
        trace = summary_csv.parts[-4]
        method = summary_csv.parts[-3]

        try:
            df = pd.read_csv(summary_csv)
        except Exception as e:
            print(f"  Skipping {summary_csv}: {e}")
            continue

        if df.empty:
            print(f"  Skipping {trace}/{method}: empty CSV")
            continue

        # Pivot tiers into columns (one row per rate)
        pivoted = df.pivot_table(
            index=["rate", "attainment"],
            columns="tier_label",
            values="tier_attainment",
        ).reset_index()

        # Flatten column names after pivot
        pivoted.columns.name = None

        pivoted["trace"] = trace
        pivoted["method"] = method
        all_data.append(pivoted)
        print(f"  Processed {trace}/{method}: {len(pivoted)} rates")

    if not all_data:
        print("No data found!")
        return pd.DataFrame()

    # Concatenate all data
    result = pd.concat(all_data, ignore_index=True)

    # Rename tier columns: "10 ms" -> "tier_10ms"
    rename_map = {}
    for col in result.columns:
        if "ms" in str(col):
            new_name = "tier_" + col.replace(" ", "")
            rename_map[col] = new_name
    result.rename(columns=rename_map, inplace=True)

    # Reorder columns
    tier_cols = sorted([c for c in result.columns if c.startswith("tier_")])
    final_cols = ["trace", "method", "rate", "attainment"] + tier_cols
    result = result[final_cols]

    # Sort by trace, method, rate
    result.sort_values(["trace", "method", "rate"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    # Write to CSV
    result.to_csv(output_path, index=False)
    print(f"\nWrote {len(result)} rows to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to results directory (default: results)",
    )
    parser.add_argument(
        "--output",
        default="results/aggregated_results.csv",
        help="Output CSV path (default: results/aggregated_results.csv)",
    )
    args = parser.parse_args()

    aggregate_results(args.results_dir, args.output)


if __name__ == "__main__":
    main()
