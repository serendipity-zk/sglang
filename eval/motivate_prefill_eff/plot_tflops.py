#!/usr/bin/env python3
"""
Plot TFLOPS at different chunk sizes (qo_len) across various total lengths (kv_len).

X-axis: Chunk size (qo_len)
Different lines: Total sequence length (kv_len)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Academic Plot Style Configuration (from eval_goodput)
# =============================================================================

plt.rcParams.update({
    # Font settings
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
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

# Colorblind-friendly palette
COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#17becf",  # Cyan
    "#8c564b",  # Brown
    "#e377c2",  # Pink
]

MARKERS = ["o", "s", "^", "D", "v", "p", "<", ">"]


def plot_tflops_vs_chunk_size(df, output_path: str):
    """
    Plot TFLOPS vs kv_len, with different lines for each chunk size.
    Also plot time to complete prefill on the right.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # Filter to only show kv_len >= 512
    df = df[df["kv_len"] >= 512]

    # Get unique values
    kv_lens = sorted(df["kv_len"].unique())
    qo_lens = sorted(df["qo_len"].unique())

    # Left plot: TFLOPS vs kv_len, different lines for chunk size
    for i, qo_len in enumerate(qo_lens):
        df_qo = df[df["qo_len"] == qo_len].sort_values("kv_len")

        ax1.plot(
            df_qo["kv_len"],
            df_qo["tflops"],
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linestyle="-",
            label=f"{qo_len}",
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
        )

    ax1.set_xlabel("Prefill Length")
    ax1.set_ylabel("TFLOPS")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(kv_lens)
    ax1.set_xticklabels([str(x) for x in kv_lens])
    ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, min_n_ticks=5))

    # Right plot: Per-chunk time vs kv_len, different lines for chunk size
    for i, qo_len in enumerate(qo_lens):
        df_qo = df[df["qo_len"] == qo_len].sort_values("kv_len")

        ax2.plot(
            df_qo["kv_len"],
            df_qo["ms"],
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linestyle="-",
            label=f"{qo_len}",
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
        )

    ax2.set_xlabel("Prefill Length")
    ax2.set_ylabel("Per-Chunk Time (ms)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(kv_lens)
    ax2.set_xticklabels([str(x) for x in kv_lens])
    ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, min_n_ticks=5))

    # Single legend at top, one line
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(qo_lens),
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "perf.csv")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"QO Lens: {sorted(df['qo_len'].unique())}")
    print(f"KV Lens: {sorted(df['kv_len'].unique())}")

    # Plot PDF
    output_path = os.path.join(script_dir, "tflops_vs_chunk_size.pdf")
    plot_tflops_vs_chunk_size(df, output_path)

    # Also save PNG for quick viewing
    output_path_png = os.path.join(script_dir, "tflops_vs_chunk_size.png")
    plot_tflops_vs_chunk_size(df, output_path_png)


if __name__ == "__main__":
    main()
