#!/usr/bin/env python3
"""
Plot scheduler tick times vs number of servers.

X-axis: number of servers
Y-axis: mean scheduler tick time (log scale)
Lines: rate per server (req/s/server)

Usage:
    python plot_scheduler_times.py
    python plot_scheduler_times.py --input scheduler_times.csv --output scheduler_times.pdf
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# =============================================================================
# Academic Plot Style Configuration
# =============================================================================

plt.rcParams.update({
    # Font settings
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    # Line settings
    "lines.linewidth": 1.5,
    "lines.markersize": 5,

    # Axes settings
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,

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

# Colorblind-friendly palette
COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#17becf",  # Cyan
]

MARKERS = ["o", "s", "^", "D", "v", "p"]


def format_time_us(value, pos):
    """Format microseconds as human-readable time labels."""
    if value >= 1000:
        ms = value / 1000
        if ms >= 10:
            return f"{ms:.0f}ms"
        else:
            return f"{ms:.1f}ms"
    else:
        return f"{value:.0f}µs"


def plot_scheduler_times(df: pd.DataFrame, output_path: str, metric: str = "mean_us"):
    """Plot scheduler tick times vs number of servers."""

    # Compute rate per server
    df = df.copy()
    df["rate_per_server"] = df["rate"] / df["num_servers"]

    # Filter to 8-40 servers only
    df = df[(df["num_servers"] >= 8) & (df["num_servers"] <= 40)]

    # Get unique rate_per_server values and sort
    rates_per_server = sorted(df["rate_per_server"].unique())

    fig, ax = plt.subplots(figsize=(8, 3))

    for i, rps in enumerate(rates_per_server):
        df_rps = df[df["rate_per_server"] == rps].sort_values("num_servers")

        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        ax.plot(
            df_rps["num_servers"],
            df_rps[metric],
            color=color,
            marker=marker,
            linestyle="-",
            label=f"{rps:.0f} req/s/srv",
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=6,
        )

    # Set log scale for y-axis
    ax.set_yscale("log")

    # Custom y-axis formatter for nice labels
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_time_us))

    # Set explicit y-ticks for better labeling
    y_ticks = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]  # in microseconds
    y_ticks = [50, 100, 500, 1000, 5000, 10000, 20000]  # 7 ticks
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_time_us))

    # Minor ticks for sub-grid
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(1.5, 2, 3, 4, 5, 6, 7, 8, 9), numticks=20))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # Enable minor grid
    ax.grid(which='major', alpha=0.5, linestyle='-', linewidth=0.6)
    ax.grid(which='minor', alpha=0.3, linestyle='-', linewidth=0.4)

    # X-axis settings
    ax.set_xlabel("Number of Servers")
    ax.set_ylabel("Router Tick Time")

    # Set x-ticks to actual server counts
    servers = sorted(df["num_servers"].unique())
    ax.set_xticks(servers)
    ax.set_xticklabels([str(s) for s in servers])

    # Legend
    ax.legend(loc="upper left", title="Rate/Server")

    # Save
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot scheduler tick times")
    parser.add_argument(
        "--input", "-i",
        default="scheduler_times.csv",
        help="Input CSV file (default: scheduler_times.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="scheduler_times.pdf",
        help="Output file path (default: scheduler_times.pdf)",
    )
    parser.add_argument(
        "--metric",
        default="mean_us",
        choices=["mean_us", "p50_us", "p90_us", "p99_us", "max_us"],
        help="Metric to plot (default: mean_us)",
    )
    parser.add_argument(
        "--format",
        default=None,
        choices=["pdf", "png", "svg"],
        help="Output format (overrides extension in --output)",
    )
    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        # Try relative to script location
        input_path = Path(__file__).parent / args.input

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    output_path = args.output
    if args.format:
        base = output_path.rsplit(".", 1)[0]
        output_path = f"{base}.{args.format}"

    plot_scheduler_times(df, output_path, metric=args.metric)


if __name__ == "__main__":
    main()
