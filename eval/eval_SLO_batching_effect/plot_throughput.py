#!/usr/bin/env python3
"""
Plot throughput vs SLO target for different (p, d) request types.

Each (p, d) pair is plotted as a separate line.

Usage:
    python plot_throughput.py
    python plot_throughput.py --input batch_analysis_results.csv --output figures/throughput.pdf
    python plot_throughput.py --metric request_throughput_per_sec
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Academic Plot Style Configuration (from eval_goodput/plot_utils.py)
# =============================================================================

plt.rcParams.update({
    # Font settings - larger
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 21,
    "legend.fontsize": 14,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,

    # Line settings
    "lines.linewidth": 2.0,
    "lines.markersize": 7,

    # Axes settings
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,

    # Figure settings
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    # Legend settings
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
})

# =============================================================================
# Style Configuration
# =============================================================================

# Colorblind-friendly palette (extended)
COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]

MARKERS = ["o", "s", "^", "D", "v", "p", "h", "*", "X", "P"]

LINESTYLES = ["-", "-", "-", "-", "-", "--", "--", "--", "--", "--"]

# Map (p, d) pairs to workload names for legend
TYPE_LABELS = {
    (28, 140): "LMSYS",
    (36, 280): "ShareGPT",
    (1019, 130): "Splitwise",
}


def plot_throughput(csv_path: str, output_path: str, metric: str = "token_throughput_per_sec"):
    """
    Plot throughput vs SLO target for different (p, d) request types.

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output figure
        metric: Column name for y-axis metric
    """
    df = pd.read_csv(csv_path)

    # Create type label - use workload name if available, otherwise p=X, d=Y
    def get_type_label(row):
        key = (int(row['p']), int(row['d']))
        if key in TYPE_LABELS:
            return TYPE_LABELS[key]
        return f"{int(row['p'])}P,{int(row['d'])}D"

    df["type"] = df.apply(get_type_label, axis=1)
    types = df["type"].unique()

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, t in enumerate(types):
        df_t = df[df["type"] == t].sort_values("slo_ms")
        # Convert to K tokens/sec for token throughput
        y_values = df_t[metric] / 1000 if metric == "token_throughput_per_sec" else df_t[metric]
        ax.plot(
            df_t["slo_ms"],
            y_values,
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            label=t,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
        )

    ax.set_xlabel("SLO Target (ms)")

    # Set y-axis label based on metric
    if metric == "token_throughput_per_sec":
        ax.set_ylabel("Throughput\n(K tokens/s)")
    elif metric == "request_throughput_per_sec":
        ax.set_ylabel("Request Throughput (req/sec)")
    elif metric == "optimal_batch_tokens":
        ax.set_ylabel("Optimal Batch Tokens")
    else:
        ax.set_ylabel(metric)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Set x ticks at interval of 10
    x_max = df["slo_ms"].max()
    ax.set_xticks(np.arange(0, x_max + 10, 10))

    # Legend outside the plot on the right
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def plot_throughput_with_batch(csv_path: str, output_path: str):
    """
    Plot throughput and batch size as two subfigures.

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output figure
    """
    df = pd.read_csv(csv_path)

    # Create type label - use workload name if available, otherwise p=X, d=Y
    def get_type_label(row):
        key = (int(row['p']), int(row['d']))
        if key in TYPE_LABELS:
            return TYPE_LABELS[key]
        return f"{int(row['p'])}P,{int(row['d'])}D"

    df["type"] = df.apply(get_type_label, axis=1)
    types = df["type"].unique()

    # Compute request batch size from decode_reqs (total requests = decode_reqs * (1+d)/d)
    df["request_batch_size"] = df["optimal_decode_reqs"] * (1 + df["d"]) / df["d"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Subplot 1: Token throughput
    ax1 = axes[0]
    for i, t in enumerate(types):
        df_t = df[df["type"] == t].sort_values("slo_ms")
        ax1.plot(
            df_t["slo_ms"],
            df_t["token_throughput_per_sec"] / 1000,  # Convert to K tokens/sec
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            label=t,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
        )

    ax1.set_xlabel("SLO Target (ms)")
    ax1.set_ylabel("Throughput\n(K tokens/s)")
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    x_max = df["slo_ms"].max()
    ax1.set_xticks(np.arange(0, x_max + 10, 10))

    # Subplot 2: Request batch size
    ax2 = axes[1]
    for i, t in enumerate(types):
        df_t = df[df["type"] == t].sort_values("slo_ms")
        ax2.plot(
            df_t["slo_ms"],
            df_t["request_batch_size"],
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            label=t,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
        )

    ax2.set_xlabel("SLO Target (ms)")
    ax2.set_ylabel("Request Batch Size")
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.set_xticks(np.arange(0, x_max + 10, 10))

    # Shared legend at the top
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(types), bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot throughput vs SLO target for different request types"
    )
    parser.add_argument(
        "--input",
        default="batch_analysis_results.csv",
        help="Input CSV file (default: batch_analysis_results.csv)",
    )
    parser.add_argument(
        "--output",
        default="figures/throughput.pdf",
        help="Output file path (default: figures/throughput.pdf)",
    )
    parser.add_argument(
        "--format",
        default=None,
        choices=["pdf", "png", "svg"],
        help="Output format (overrides extension in --output)",
    )
    parser.add_argument(
        "--metric",
        default="token_throughput_per_sec",
        choices=["token_throughput_per_sec", "request_throughput_per_sec", "optimal_batch_tokens"],
        help="Metric to plot on y-axis (default: token_throughput_per_sec)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Generate combined plot with throughput and batch size subfigures",
    )
    args = parser.parse_args()

    # Resolve input path relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(script_dir, input_path)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(script_dir, output_path)

    if args.format:
        base = output_path.rsplit(".", 1)[0]
        output_path = f"{base}.{args.format}"

    if args.combined:
        plot_throughput_with_batch(input_path, output_path)
    else:
        plot_throughput(input_path, output_path, metric=args.metric)


if __name__ == "__main__":
    main()
