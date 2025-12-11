#!/usr/bin/env python3
"""
Plot iteration metrics vs request rate to visualize non-linear load behavior.

Generates combined plots with multiple traces, similar to plot_throughput.py style.

Usage:
    python plot_results.py --results-dir results/
    python plot_results.py --csv all_metrics.csv
    python plot_results.py --results-dir results/ --combined
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install matplotlib: pip install matplotlib")
    sys.exit(1)

# =============================================================================
# Academic Plot Style Configuration (from eval_goodput/plot_utils.py)
# =============================================================================

plt.rcParams.update({
    # Font settings - larger
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 16,
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

# Colorblind-friendly palette
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

# Map trace names to display labels
TRACE_LABELS = {
    "lmsys": "LMSYS",
    "sharegpt": "ShareGPT",
    "splitwise": "Splitwise",
}


def get_trace_label(trace_name: str) -> str:
    """Get display label for trace name."""
    return TRACE_LABELS.get(trace_name, trace_name)


def load_metrics(results_dir: str) -> pd.DataFrame:
    """Load all_metrics.csv from results directory."""
    csv_path = os.path.join(results_dir, "all_metrics.csv")
    if not os.path.exists(csv_path):
        # Try to parse logs first
        print(f"[info] all_metrics.csv not found, running parse_iteration_logs.py...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parse_script = os.path.join(script_dir, "parse_iteration_logs.py")
        import subprocess
        subprocess.run([sys.executable, parse_script, "--results-dir", results_dir])

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find or create {csv_path}")

    return pd.read_csv(csv_path)


def compute_p93_p97_mean(group, column):
    """Compute mean of values between p93 and p97 percentiles."""
    values = group[column].dropna()
    if len(values) == 0:
        return np.nan
    p93 = values.quantile(0.93)
    p97 = values.quantile(0.97)
    filtered = values[(values >= p93) & (values <= p97)]
    if len(filtered) == 0:
        # Fallback to p93-p97 range even if empty, use boundary values
        return (p93 + p97) / 2
    return filtered.mean()


def plot_combined(df: pd.DataFrame, output_path: str):
    """
    Plot iteration time, token batch size, and KV cache usage as three subfigures, one line per trace.
    Uses p93-p97 percentile range for computing mean values.

    Args:
        df: DataFrame with columns: trace, rate, iteration_time_ms, token_batch_size, kv_tokens_used
        output_path: Path to output figure
    """
    df_valid = df[df["iteration_time_ms"].notna()].copy()

    # Check if we have trace column
    has_trace = "trace" in df_valid.columns
    if has_trace:
        traces = sorted(df_valid["trace"].unique())
    else:
        traces = ["all"]
        df_valid["trace"] = "all"

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    # Subplot 1: Iteration Time (p93-p97 mean)
    ax1 = axes[0]
    for i, trace in enumerate(traces):
        df_t = df_valid[df_valid["trace"] == trace]
        grouped = df_t.groupby("rate").apply(
            lambda g: compute_p93_p97_mean(g, "iteration_time_ms")
        ).reset_index(name="iteration_time_ms")
        grouped = grouped.sort_values("rate")

        ax1.plot(
            grouped["rate"],
            grouped["iteration_time_ms"],
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            label=get_trace_label(trace),
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
        )

    ax1.set_xlabel("Request Rate (req/s)")
    ax1.set_ylabel("Iteration Time (ms)")
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    # Set 5 y ticks
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))

    # Set x ticks - use step of 10 or 20 depending on range
    x_max = df_valid["rate"].max()
    x_step = 20 if x_max > 60 else 10
    ax1.set_xticks(np.arange(0, x_max + x_step, x_step))

    # Subplot 2: Token Batch Size (p93-p97 mean)
    ax2 = axes[1]
    df_batch = df[df["token_batch_size"].notna()].copy()
    if not has_trace:
        df_batch["trace"] = "all"

    for i, trace in enumerate(traces):
        df_t = df_batch[df_batch["trace"] == trace]
        grouped = df_t.groupby("rate").apply(
            lambda g: compute_p93_p97_mean(g, "token_batch_size")
        ).reset_index(name="token_batch_size")
        grouped = grouped.sort_values("rate")

        ax2.plot(
            grouped["rate"],
            grouped["token_batch_size"],
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            label=get_trace_label(trace),
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
        )

    ax2.set_xlabel("Request Rate (req/s)")
    ax2.set_ylabel("Token\nBatch Size")
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.set_xticks(np.arange(0, x_max + x_step, x_step))

    # Subplot 3: KV Cache Tokens Used (p93-p97 mean)
    ax3 = axes[2]
    df_kv = df[df["kv_tokens_used"].notna()].copy()
    if not has_trace:
        df_kv["trace"] = "all"

    for i, trace in enumerate(traces):
        df_t = df_kv[df_kv["trace"] == trace]
        grouped = df_t.groupby("rate").apply(
            lambda g: compute_p93_p97_mean(g, "kv_tokens_used")
        ).reset_index(name="kv_tokens_used")
        grouped = grouped.sort_values("rate")

        ax3.plot(
            grouped["rate"],
            grouped["kv_tokens_used"] / 1000,  # Convert to K tokens
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            label=get_trace_label(trace),
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
        )

    ax3.set_xlabel("Request Rate (req/s)")
    ax3.set_ylabel("KV Cache\n(K tokens)")
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0)
    ax3.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax3.set_xticks(np.arange(0, x_max + x_step, x_step))

    # Shared legend at the top
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(traces), bbox_to_anchor=(0.5, 1.08))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path)
    print(f"Saved: {output_path}")

    # Also save PDF
    pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    print(f"Saved: {pdf_path}")
    plt.close()


def plot_iteration_time(df: pd.DataFrame, output_path: str):
    """
    Plot iteration time vs rate, one line per trace.

    Args:
        df: DataFrame with columns: trace, rate, iteration_time_ms
        output_path: Path to output figure
    """
    df_valid = df[df["iteration_time_ms"].notna()].copy()

    has_trace = "trace" in df_valid.columns
    if has_trace:
        traces = sorted(df_valid["trace"].unique())
    else:
        traces = ["all"]
        df_valid["trace"] = "all"

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, trace in enumerate(traces):
        df_t = df_valid[df_valid["trace"] == trace]
        grouped = df_t.groupby("rate")["iteration_time_ms"].mean().reset_index()
        grouped = grouped.sort_values("rate")

        ax.plot(
            grouped["rate"],
            grouped["iteration_time_ms"],
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            label=get_trace_label(trace),
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
        )

    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("Iteration Time (ms)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    x_max = df_valid["rate"].max()
    x_step = 20 if x_max > 60 else 10
    ax.set_xticks(np.arange(0, x_max + x_step, x_step))

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path)
    print(f"Saved: {output_path}")

    pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    print(f"Saved: {pdf_path}")
    plt.close()


def plot_batch_size(df: pd.DataFrame, output_path: str):
    """
    Plot batch size (num_requests) vs rate, one line per trace.

    Args:
        df: DataFrame with columns: trace, rate, num_requests
        output_path: Path to output figure
    """
    df_valid = df[df["num_requests"].notna()].copy()

    has_trace = "trace" in df_valid.columns
    if has_trace:
        traces = sorted(df_valid["trace"].unique())
    else:
        traces = ["all"]
        df_valid["trace"] = "all"

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, trace in enumerate(traces):
        df_t = df_valid[df_valid["trace"] == trace]
        grouped = df_t.groupby("rate")["num_requests"].mean().reset_index()
        grouped = grouped.sort_values("rate")

        ax.plot(
            grouped["rate"],
            grouped["num_requests"],
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            label=get_trace_label(trace),
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
        )

    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("Batch Size (requests)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    x_max = df_valid["rate"].max()
    x_step = 20 if x_max > 60 else 10
    ax.set_xticks(np.arange(0, x_max + x_step, x_step))

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path)
    print(f"Saved: {output_path}")

    pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    print(f"Saved: {pdf_path}")
    plt.close()


def plot_token_batch_size(df: pd.DataFrame, output_path: str):
    """
    Plot token batch size vs rate, one line per trace.

    Args:
        df: DataFrame with columns: trace, rate, token_batch_size
        output_path: Path to output figure
    """
    df_valid = df[df["token_batch_size"].notna()].copy()

    has_trace = "trace" in df_valid.columns
    if has_trace:
        traces = sorted(df_valid["trace"].unique())
    else:
        traces = ["all"]
        df_valid["trace"] = "all"

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, trace in enumerate(traces):
        df_t = df_valid[df_valid["trace"] == trace]
        grouped = df_t.groupby("rate")["token_batch_size"].mean().reset_index()
        grouped = grouped.sort_values("rate")

        ax.plot(
            grouped["rate"],
            grouped["token_batch_size"],
            color=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            label=get_trace_label(trace),
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
        )

    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("Token Batch Size")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    x_max = df_valid["rate"].max()
    x_step = 20 if x_max > 60 else 10
    ax.set_xticks(np.arange(0, x_max + x_step, x_step))

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path)
    print(f"Saved: {output_path}")

    pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    print(f"Saved: {pdf_path}")
    plt.close()


def generate_summary_stats(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Generate summary statistics CSV with p93-p97 percentile means."""
    df_valid = df.copy()

    # Check if we have trace column
    has_trace = "trace" in df_valid.columns
    group_cols = ["trace", "rate"] if has_trace else ["rate"]

    # Compute p93-p97 mean for key metrics
    def agg_p93_p97(group):
        result = {}
        for col in ["iteration_time_ms", "token_batch_size", "num_requests", "kv_tokens_used"]:
            if col in group.columns:
                result[f"{col}_p93_p97"] = compute_p93_p97_mean(group, col)
        # Also compute regular percentiles
        for col in ["iteration_time_ms", "token_batch_size"]:
            if col in group.columns:
                values = group[col].dropna()
                if len(values) > 0:
                    result[f"{col}_p50"] = values.quantile(0.50)
                    result[f"{col}_p90"] = values.quantile(0.90)
                    result[f"{col}_p95"] = values.quantile(0.95)
                    result[f"{col}_p99"] = values.quantile(0.99)
        result["count"] = len(group)
        return pd.Series(result)

    stats = df_valid.groupby(group_cols).apply(agg_p93_p97).reset_index()

    output_path = os.path.join(output_dir, "summary_stats.csv")
    stats.to_csv(output_path, index=False)
    print(f"[stats] Saved: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Plot iteration metrics vs request rate",
    )
    parser.add_argument("--results-dir", "-r", default="results",
                        help="Results directory containing all_metrics.csv")
    parser.add_argument("--csv", "-c", help="Direct path to metrics CSV file")
    parser.add_argument("--output-dir", "-o", help="Output directory for plots (default: results-dir/figures)")
    parser.add_argument("--combined", action="store_true",
                        help="Generate combined plot with iteration time and batch size subfigures")
    parser.add_argument("--trace", "-t", help="Filter to plot only a specific trace (e.g., lmsys, sharegpt, splitwise)")
    args = parser.parse_args()

    # Load data
    if args.csv:
        df = pd.read_csv(args.csv)
        base_dir = os.path.dirname(args.csv) or "."
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = args.results_dir
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(script_dir, results_dir)
        df = load_metrics(results_dir)
        base_dir = results_dir

    # Filter by trace if specified
    if args.trace and "trace" in df.columns:
        available_traces = df["trace"].unique()
        if args.trace not in available_traces:
            print(f"[error] Trace '{args.trace}' not found. Available: {list(available_traces)}")
            return 1
        df = df[df["trace"] == args.trace].copy()
        print(f"[info] Filtered to trace: {args.trace}")

    output_dir = args.output_dir or os.path.join(base_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[info] Loaded {len(df)} rows")
    if "trace" in df.columns:
        print(f"[info] Traces: {sorted(df['trace'].unique())}")
    print(f"[info] Rates: {sorted(df['rate'].unique())}")
    print(f"[info] Output directory: {output_dir}")

    # Generate plots
    if args.combined:
        plot_combined(df, os.path.join(output_dir, "combined.png"))
    else:
        # Generate all individual plots
        plot_iteration_time(df, os.path.join(output_dir, "iteration_time.png"))
        plot_batch_size(df, os.path.join(output_dir, "batch_size.png"))
        plot_token_batch_size(df, os.path.join(output_dir, "token_batch_size.png"))
        # Also generate combined
        plot_combined(df, os.path.join(output_dir, "combined.png"))

    # Generate summary stats
    stats = generate_summary_stats(df, output_dir)

    # Print summary
    print("\n[done] Summary statistics (p93-p97 mean):")
    if "trace" in df.columns:
        summary_cols = ["trace", "rate", "iteration_time_ms_p93_p97", "token_batch_size_p93_p97", "kv_tokens_used_p93_p97"]
    else:
        summary_cols = ["rate", "iteration_time_ms_p93_p97", "token_batch_size_p93_p97", "kv_tokens_used_p93_p97"]
    print(stats[summary_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
