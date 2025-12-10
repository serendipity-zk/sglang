#!/usr/bin/env python3
"""
Generate all plots for evaluation results.

This is a convenience script that runs all plotting scripts.

Usage:
    python plot_all.py
    python plot_all.py --input results/aggregated_results.csv --output results/figures/
"""

import argparse
import os

from plot_utils import load_and_filter_data
from plot_attainment_grid import plot_all_traces_grid
from plot_attainment_single import plot_single_trace_figure
from plot_per_tier import plot_per_tier_comparison
from plot_goodput import plot_goodput_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Generate all evaluation plots"
    )
    parser.add_argument(
        "--input",
        default="results/aggregated_results.csv",
        help="Input CSV file (default: results/aggregated_results.csv)",
    )
    parser.add_argument(
        "--output",
        default="results/figures",
        help="Output directory for figures (default: results/figures)",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format (default: pdf)",
    )
    args = parser.parse_args()

    # Load data once
    df = load_and_filter_data(args.input)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    ext = args.format

    # 1. Grid of all traces
    plot_all_traces_grid(df, os.path.join(args.output, f"rate_vs_attainment_grid.{ext}"))

    # 2. Individual trace plots
    for trace in df["trace"].unique():
        safe_name = trace.replace("/", "_")
        plot_single_trace_figure(df, trace, os.path.join(args.output, f"rate_vs_attainment_{safe_name}.{ext}"))

    # 3. Per-tier breakdown
    plot_per_tier_comparison(df, os.path.join(args.output, f"per_tier_attainment.{ext}"))

    # 4. Goodput bar chart
    plot_goodput_comparison(df, os.path.join(args.output, f"goodput_comparison.{ext}"))

    print(f"\nAll figures saved to {args.output}/")


if __name__ == "__main__":
    main()
