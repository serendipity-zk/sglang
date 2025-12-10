#!/usr/bin/env python3
"""
Plot rate vs attainment for individual traces (one figure per trace).

Usage:
    python plot_attainment_single.py
    python plot_attainment_single.py --trace sharegpt
    python plot_attainment_single.py --input results/aggregated_results.csv --output results/figures/
"""

import argparse
import os

import matplotlib.pyplot as plt

from plot_utils import (
    load_and_filter_data,
    plot_single_trace,
    ensure_output_dir,
)


def plot_single_trace_figure(df, trace: str, output_path: str):
    """Create a single figure for one trace."""
    df_trace = df[df["trace"] == trace]

    fig, ax = plt.subplots(figsize=(4, 3))
    plot_single_trace(ax, df_trace, trace, show_legend=True)
    ax.legend(loc="lower left")

    plt.tight_layout()
    ensure_output_dir(output_path)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot rate vs attainment for individual traces")
    parser.add_argument(
        "--input",
        default="results/aggregated_results.csv",
        help="Input CSV file (default: results/aggregated_results.csv)",
    )
    parser.add_argument(
        "--output",
        default="results/figures",
        help="Output directory (default: results/figures)",
    )
    parser.add_argument(
        "--trace",
        default=None,
        help="Specific trace to plot (default: plot all traces)",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format (default: pdf)",
    )
    args = parser.parse_args()

    df = load_and_filter_data(args.input)
    os.makedirs(args.output, exist_ok=True)

    if args.trace:
        traces = [args.trace]
    else:
        traces = df["trace"].unique()

    for trace in traces:
        safe_name = trace.replace("/", "_")
        output_path = os.path.join(args.output, f"rate_vs_attainment_{safe_name}.{args.format}")
        plot_single_trace_figure(df, trace, output_path)


if __name__ == "__main__":
    main()
