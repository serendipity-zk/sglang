#!/usr/bin/env python3
"""
Plot rate vs attainment grid - all traces in one figure.

Usage:
    python plot_attainment_grid.py
    python plot_attainment_grid.py --input results/aggregated_results.csv --output results/figures/grid.pdf
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_utils import (
    load_and_filter_data,
    sort_traces,
    sort_methods,
    get_style,
    plot_single_trace,
    ensure_output_dir,
)


def plot_all_traces_grid(df, output_path: str):
    """Create a grid of subplots, one per trace."""
    traces = df["trace"].unique()
    traces = sort_traces(traces)

    n_traces = len(traces)
    n_cols = min(3, n_traces)
    n_rows = (n_traces + n_cols - 1) // n_cols

    # Figure size: ~3.3in per column for double-column paper
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 2.5 * n_rows))

    if n_traces == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, trace in enumerate(traces):
        df_trace = df[df["trace"] == trace]
        plot_single_trace(axes[idx], df_trace, trace, show_legend=False)

    # Hide unused subplots
    for idx in range(n_traces, len(axes)):
        axes[idx].set_visible(False)

    # Collect all unique methods across all traces for legend
    all_methods = df["method"].unique()
    all_methods = sort_methods(all_methods)

    # Create custom legend handles for all methods
    legend_handles = []
    for method in all_methods:
        style = get_style(method)
        handle = Line2D([0], [0], color=style["color"], marker=style["marker"],
                        linestyle=style["linestyle"], label=style["label"],
                        markerfacecolor="white", markeredgewidth=1.2, markersize=5)
        legend_handles.append(handle)

    fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles),
               bbox_to_anchor=(0.5, 1.02), frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_output_dir(output_path)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot rate vs attainment grid")
    parser.add_argument(
        "--input",
        default="results/aggregated_results.csv",
        help="Input CSV file (default: results/aggregated_results.csv)",
    )
    parser.add_argument(
        "--output",
        default="results/figures/rate_vs_attainment_grid.pdf",
        help="Output file path (default: results/figures/rate_vs_attainment_grid.pdf)",
    )
    parser.add_argument(
        "--format",
        default=None,
        choices=["pdf", "png", "svg"],
        help="Output format (overrides extension in --output)",
    )
    args = parser.parse_args()

    df = load_and_filter_data(args.input)

    output_path = args.output
    if args.format:
        base = output_path.rsplit(".", 1)[0]
        output_path = f"{base}.{args.format}"

    plot_all_traces_grid(df, output_path)


if __name__ == "__main__":
    main()
