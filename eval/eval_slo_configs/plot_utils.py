#!/usr/bin/env python3
"""
Shared utilities for plotting SLO config evaluation results.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Academic plot style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
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
    "savefig.pad_inches": 0.02,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
})

# Style for configs
COLORS = {
    "no_autoscaling": "#d62728",   # Red - baseline
    "ttft": "#17becf",             # Cyan - TierServe (same as PolyServe)
    "ttft_steal": "#2ca02c",       # Green
    "ttft_promote": "#1f77b4",     # Blue - best
}

MARKERS = {
    "no_autoscaling": "o",
    "ttft": "s",
    "ttft_steal": "^",
    "ttft_promote": "D",
}

LINESTYLES = {
    "no_autoscaling": "--",
    "ttft": "-.",
    "ttft_steal": ":",
    "ttft_promote": "-",
}

CONFIG_LABELS = {
    "no_autoscaling": "Static",
    "ttft": "TierServe",
    "ttft_steal": "TierServe + Steal",
    "ttft_promote": "TierServe + Steal + Promote",
}

TRACE_LABELS = {
    "sharegpt": "ShareGPT",
    "lmsys": "LMSYS-Chat",
    "splitwise": "Splitwise",
}

# Filtering
CONFIGS_TO_PLOT = ["no_autoscaling", "ttft", "ttft_steal", "ttft_promote"]
TRACES_TO_PLOT = ["sharegpt", "lmsys", "splitwise"]
CONFIG_ORDER = ["no_autoscaling", "ttft", "ttft_steal", "ttft_promote"]
TRACE_ORDER = ["sharegpt", "lmsys", "splitwise"]
TARGET_ATTAINMENT = 0.90


def get_style(config: str):
    """Get plotting style for a config."""
    return {
        "color": COLORS.get(config, "#333333"),
        "marker": MARKERS.get(config, "o"),
        "linestyle": LINESTYLES.get(config, "-"),
        "label": CONFIG_LABELS.get(config, config),
    }


def compute_min_tier_attainment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute min attainment across all tiers for each row."""
    tier_cols = [c for c in df.columns if c.startswith("tier_")]
    if tier_cols:
        df = df.copy()
        df["min_tier_attainment"] = df[tier_cols].min(axis=1)
    return df


def load_and_filter_data(input_path: str) -> pd.DataFrame:
    """Load CSV and compute min tier attainment."""
    df = pd.read_csv(input_path)

    if CONFIGS_TO_PLOT is not None:
        df = df[df["config"].isin(CONFIGS_TO_PLOT)]
    if TRACES_TO_PLOT is not None:
        df = df[df["trace"].isin(TRACES_TO_PLOT)]

    # Compute min tier attainment
    df = compute_min_tier_attainment(df)

    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Traces: {df['trace'].unique().tolist()}")
    print(f"Configs: {df['config'].unique().tolist()}")

    return df


def sort_configs(configs):
    """Sort configs by predefined order."""
    return sorted(configs, key=lambda c: CONFIG_ORDER.index(c) if c in CONFIG_ORDER else 99)


def sort_traces(traces):
    """Sort traces by predefined order."""
    return sorted(traces, key=lambda t: TRACE_ORDER.index(t) if t in TRACE_ORDER else 99)


def plot_single_trace(ax, df_trace: pd.DataFrame, trace_name: str, show_legend: bool = False):
    """Plot rate vs min_tier_attainment for a single trace."""
    configs = df_trace["config"].unique()
    configs = sort_configs(configs)

    for config in configs:
        df_config = df_trace[df_trace["config"] == config].sort_values("rate")
        style = get_style(config)

        ax.plot(
            df_config["rate"],
            df_config["min_tier_attainment"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=style["label"],
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
        )

    # Set x-axis limits
    all_rates = df_trace["rate"]
    min_rate, max_rate = all_rates.min(), all_rates.max()
    x_padding = (max_rate - min_rate) * 0.05
    ax.set_xlim(max(0, min_rate - x_padding), max_rate + x_padding)

    # Add target line
    ax.axhline(y=TARGET_ATTAINMENT, color="gray", linestyle=":", linewidth=1, alpha=0.7, zorder=0)
    xlim = ax.get_xlim()
    ax.text(xlim[1] - (xlim[1] - xlim[0]) * 0.02, TARGET_ATTAINMENT + 0.01,
            f"{int(TARGET_ATTAINMENT*100)}% target", ha="right", va="bottom",
            fontsize=7, color="gray", alpha=0.8)

    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("Min Tier SLO Attainment")
    ax.set_title(TRACE_LABELS.get(trace_name, trace_name))
    ax.set_ylim(0, 1.05)

    if show_legend:
        ax.legend(loc="lower left", ncol=1)


def ensure_output_dir(output_path: str):
    """Ensure the output directory exists."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
