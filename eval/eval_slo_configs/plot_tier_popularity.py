#!/usr/bin/env python3
"""
Plot tier popularity across time from a trace file.

Shows the distribution of requests by TPOT tier over time windows.

Usage:
    python plot_tier_popularity.py --trace /path/to/trace.csv
    python plot_tier_popularity.py --trace /path/to/trace.csv --output figures/tier_popularity.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import shared style settings
from plot_utils import ensure_output_dir

# Academic plot style (consistent with other figures)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
})

# Use matplotlib Blues colormap for gradient (dark = tight SLO, light = relaxed SLO)
_blues = plt.colormaps.get_cmap("Blues")

TIER_COLORS = {
    10: _blues(0.8),   # Darkest blue - tightest SLO
    20: _blues(0.5),   # Medium blue
    40: _blues(0.3),   # Lightest blue - most relaxed SLO
}

TIER_LABELS = {
    10: "TPOT 10 ms",
    20: "TPOT 20 ms",
    40: "TPOT 40 ms",
}


def load_trace(trace_path: str) -> pd.DataFrame:
    """Load trace CSV file. Arrival times are in ms."""
    df = pd.read_csv(trace_path)
    print(f"Loaded {len(df)} requests from {trace_path}")
    print(f"Time range: {df['arrival'].min():.1f}ms - {df['arrival'].max():.1f}ms")
    print(f"TPOT tiers: {sorted(df['tpot'].unique())}")
    return df


def compute_tier_distribution(df: pd.DataFrame, window_size_ms: float = 10000.0,
                               smooth_window: int = 1) -> pd.DataFrame:
    """Compute tier distribution over time windows with optional smoothing.

    Args:
        df: Trace DataFrame with 'arrival' (ms) and 'tpot' columns
        window_size_ms: Time window size in milliseconds
        smooth_window: Number of windows for moving average smoothing (1 = no smoothing)

    Returns:
        DataFrame with columns: time_start, tier_10, tier_20, tier_40, total
    """
    max_time = df['arrival'].max()
    windows = []

    for start in np.arange(0, max_time, window_size_ms):
        end = start + window_size_ms
        window_df = df[(df['arrival'] >= start) & (df['arrival'] < end)]

        tier_counts = window_df['tpot'].value_counts().to_dict()
        total = len(window_df)

        windows.append({
            'time_start': start,
            'time_mid': start + window_size_ms / 2,
            'tier_10': tier_counts.get(10, 0),
            'tier_20': tier_counts.get(20, 0),
            'tier_40': tier_counts.get(40, 0),
            'total': total,
        })

    dist_df = pd.DataFrame(windows)

    # Apply moving average smoothing if requested
    if smooth_window > 1:
        for col in ['tier_10', 'tier_20', 'tier_40', 'total']:
            dist_df[col] = dist_df[col].rolling(window=smooth_window, center=True, min_periods=1).mean()

    return dist_df


def plot_tier_popularity_stacked(df: pd.DataFrame, trace_name: str, output_path: str,
                                  window_size_ms: float = 10000.0, smooth_window: int = 1):
    """Plot stacked area chart of tier popularity over time."""
    dist_df = compute_tier_distribution(df, window_size_ms, smooth_window)

    fig, ax = plt.subplots(figsize=(8, 3))

    # Compute percentages
    dist_df['pct_10'] = dist_df['tier_10'] / dist_df['total'] * 100
    dist_df['pct_20'] = dist_df['tier_20'] / dist_df['total'] * 100
    dist_df['pct_40'] = dist_df['tier_40'] / dist_df['total'] * 100

    # Handle NaN (windows with no requests)
    dist_df = dist_df.fillna(0)

    # Convert time from ms to seconds for display
    time_sec = dist_df['time_mid'] / 1000.0

    # Stacked area plot
    ax.stackplot(
        time_sec,
        dist_df['pct_10'],
        dist_df['pct_20'],
        dist_df['pct_40'],
        labels=[TIER_LABELS[10], TIER_LABELS[20], TIER_LABELS[40]],
        colors=[TIER_COLORS[10], TIER_COLORS[20], TIER_COLORS[40]],
        alpha=0.8,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tier Distribution (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(time_sec.min(), time_sec.max())

    # Legend
    ax.legend(loc="upper right", ncol=1)

    plt.tight_layout()
    ensure_output_dir(output_path)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def plot_tier_popularity_lines(df: pd.DataFrame, trace_name: str, output_path: str,
                                window_size_ms: float = 10000.0, smooth_window: int = 1):
    """Plot line chart of tier counts over time."""
    dist_df = compute_tier_distribution(df, window_size_ms, smooth_window)

    fig, ax = plt.subplots(figsize=(8, 3))

    # Convert time from ms to seconds for display
    time_sec = dist_df['time_mid'] / 1000.0
    window_size_sec = window_size_ms / 1000.0

    # Line plot for each tier
    for tier in [10, 20, 40]:
        ax.plot(
            time_sec,
            dist_df[f'tier_{tier}'],
            color=TIER_COLORS[tier],
            label=TIER_LABELS[tier],
            linewidth=1.5,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Requests per {window_size_sec:.0f}s Window")
    ax.set_xlim(time_sec.min(), time_sec.max())
    ax.set_ylim(0, None)

    # Legend
    ax.legend(loc="upper right", ncol=1)

    plt.tight_layout()
    ensure_output_dir(output_path)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def plot_tier_popularity_combined(df: pd.DataFrame, trace_name: str, output_path: str,
                                   window_size_ms: float = 10000.0, smooth_window: int = 1):
    """Plot combined figure with both percentage and absolute counts."""
    dist_df = compute_tier_distribution(df, window_size_ms, smooth_window)

    # Compute percentages
    dist_df['pct_10'] = dist_df['tier_10'] / dist_df['total'] * 100
    dist_df['pct_20'] = dist_df['tier_20'] / dist_df['total'] * 100
    dist_df['pct_40'] = dist_df['tier_40'] / dist_df['total'] * 100
    dist_df = dist_df.fillna(0)

    # Convert time from ms to seconds for display
    time_sec = dist_df['time_mid'] / 1000.0
    window_size_sec = window_size_ms / 1000.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Top: Stacked percentage
    ax1.stackplot(
        time_sec,
        dist_df['pct_10'],
        dist_df['pct_20'],
        dist_df['pct_40'],
        labels=[TIER_LABELS[10], TIER_LABELS[20], TIER_LABELS[40]],
        colors=[TIER_COLORS[10], TIER_COLORS[20], TIER_COLORS[40]],
        alpha=0.8,
    )
    ax1.set_ylabel("Distribution (%)")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right", ncol=3)

    # Bottom: Line plot of counts
    for tier in [10, 20, 40]:
        ax2.plot(
            time_sec,
            dist_df[f'tier_{tier}'],
            color=TIER_COLORS[tier],
            label=TIER_LABELS[tier],
            linewidth=1.5,
        )

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(f"Requests / {window_size_sec:.0f}s")
    ax2.set_xlim(time_sec.min(), time_sec.max())
    ax2.set_ylim(0, None)

    plt.tight_layout()
    ensure_output_dir(output_path)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot tier popularity from trace file")
    parser.add_argument("--trace", required=True, help="Path to trace CSV file")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--format", default="png", choices=["pdf", "png", "svg"],
                        help="Output format")
    parser.add_argument("--window", type=float, default=10.0,
                        help="Time window size in seconds (default: 10)")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Smoothing window size (number of windows for moving average, default: 5)")
    parser.add_argument("--style", default="stacked", choices=["stacked", "lines", "combined"],
                        help="Plot style: stacked area, lines, or combined")
    args = parser.parse_args()

    # Load trace
    df = load_trace(args.trace)

    # Extract trace name from path
    trace_name = os.path.basename(args.trace).replace(".csv", "")
    # Clean up name for title
    trace_name = trace_name.replace("time_shift_", "").replace("_", " ").title()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.basename(args.trace).replace(".csv", "")
        output_path = f"results/figures/tier_popularity_{base_name}.{args.format}"

    # Ensure correct extension
    if args.format and not output_path.endswith(f".{args.format}"):
        base = output_path.rsplit(".", 1)[0]
        output_path = f"{base}.{args.format}"

    # Convert window from seconds to ms
    window_size_ms = args.window * 1000.0

    # Plot
    if args.style == "stacked":
        plot_tier_popularity_stacked(df, trace_name, output_path, window_size_ms, args.smooth)
    elif args.style == "lines":
        plot_tier_popularity_lines(df, trace_name, output_path, window_size_ms, args.smooth)
    else:
        plot_tier_popularity_combined(df, trace_name, output_path, window_size_ms, args.smooth)


if __name__ == "__main__":
    main()
