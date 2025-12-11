#!/usr/bin/env python3
"""Plot GPU time vs scheduling time (kvf+psim) scatter plot."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Academic plot style (consistent with plot_utils.py)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
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
    "savefig.pad_inches": 0.02,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
})

SCRIPT_DIR = Path(__file__).parent
RESULTS_PATH = SCRIPT_DIR / "results"
INPUT_CSV = RESULTS_PATH / "polyserve_timing.csv"
OUTPUT_PDF = RESULTS_PATH / "gpu_vs_scheduling_scatter.pdf"
OUTPUT_PNG = RESULTS_PATH / "gpu_vs_scheduling_scatter.png"
OUTPUT_STATS = RESULTS_PATH / "gpu_vs_scheduling.stats"


def plot_scatter():
    # Load data
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} rows")

    # Calculate scheduling time
    df['scheduling'] = df['kvf'] + df['psim']

    # Sample for visualization
    sample_size = min(100000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    print(f"Sampled {sample_size:,} points for plotting")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 3))

    # Get axis limits
    x_max = 250
    y_max = 100

    # Fill light green region to the right of y=x line (where GPU time > scheduling time)
    # This is the "overlapped" region
    triangle_x = [0, y_max, x_max, x_max, 0]
    triangle_y = [0, y_max, y_max, 0, 0]
    ax.fill(triangle_x, triangle_y, color='#A5D6A7', alpha=0.5, zorder=0)

    # Scatter plot with transparency (larger dots)
    ax.scatter(df_sample['gpu'], df_sample['scheduling'],
               alpha=0.5, s=20, c='#1f77b4', edgecolors='none', rasterized=True)

    # Add y=x reference line (terminate at y=100)
    ax.plot([0, y_max], [0, y_max], color='#d62728', linestyle='--',
            linewidth=1.5, label='GPU Time = Scheduling Time', zorder=10)

    # Add "Overlapped" text in the green region (top right)
    ax.text(200, 60, 'Overlapped', fontsize=14, color='#4CAF50',
            fontweight='bold', ha='center', va='center')

    # Labels (font sizes aligned with plot_violation_breakdown.py)
    ax.set_xlabel('GPU Time (ms)', fontsize=14)
    ax.set_ylabel('Scheduling Time (ms)', fontsize=14)

    # Set axis limits
    ax.set_xlim(0, 250)
    ax.set_ylim(0, y_max)

    # Tick label sizes
    ax.tick_params(axis='both', labelsize=13)

    # Legend
    ax.legend(loc='upper right', fontsize=10)

    # Save plot
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF)
    plt.savefig(OUTPUT_PNG)
    print(f"Saved to {OUTPUT_PDF}")
    print(f"Saved to {OUTPUT_PNG}")

    # Compute and save statistics (on full dataset, not sample)
    total_count = len(df)
    overlapped_count = (df['gpu'] >= df['scheduling']).sum()
    not_overlapped_count = total_count - overlapped_count
    overlap_ratio = overlapped_count / total_count

    stats = {
        "total_count": int(total_count),
        "overlapped_count": int(overlapped_count),
        "not_overlapped_count": int(not_overlapped_count),
        "overlap_ratio": float(overlap_ratio),
        "scheduling_time": {
            "mean": float(df['scheduling'].mean()),
            "std": float(df['scheduling'].std()),
            "min": float(df['scheduling'].min()),
            "max": float(df['scheduling'].max()),
            "median": float(df['scheduling'].median()),
            "p95": float(df['scheduling'].quantile(0.95)),
            "p99": float(df['scheduling'].quantile(0.99)),
        },
        "gpu_time": {
            "mean": float(df['gpu'].mean()),
            "std": float(df['gpu'].std()),
            "min": float(df['gpu'].min()),
            "max": float(df['gpu'].max()),
            "median": float(df['gpu'].median()),
            "p95": float(df['gpu'].quantile(0.95)),
            "p99": float(df['gpu'].quantile(0.99)),
        },
        "kvf": {
            "mean": float(df['kvf'].mean()),
            "std": float(df['kvf'].std()),
        },
        "psim": {
            "mean": float(df['psim'].mean()),
            "std": float(df['psim'].std()),
        },
    }

    with open(OUTPUT_STATS, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved to {OUTPUT_STATS}")


if __name__ == '__main__':
    plot_scatter()
