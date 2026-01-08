#!/usr/bin/env python3
"""
Shared utilities for plotting evaluation results.

Contains style configurations, filtering settings, and helper functions
used by all plotting scripts.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Academic Plot Style Configuration
# =============================================================================

# Use Type 1 fonts for camera-ready papers
plt.rcParams.update({
    # Font settings
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

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

# =============================================================================
# Style Configuration
# =============================================================================

# Colorblind-friendly palette (from ColorBrewer / Tableau)
COLORS = {
    "SGLang-PD": "#1f77b4",       # Blue
    "Niyama": "#2ca02c",          # Green
    "SGLang-Random": "#ff7f0e",   # Orange
    "Chunked-Prefill": "#d62728", # Red
    "SGLang-Round-Robin": "#9467bd",  # Purple
    "PolyServe": "#17becf",       # Cyan
}

MARKERS = {
    "SGLang-PD": "o",
    "Niyama": "s",
    "SGLang-Random": "^",
    "Chunked-Prefill": "D",
    "SGLang-Round-Robin": "v",
    "PolyServe": "p",
}

LINESTYLES = {
    "SGLang-PD": "-",
    "Niyama": "-",
    "SGLang-Random": "--",
    "Chunked-Prefill": "--",
    "SGLang-Round-Robin": "-.",
    "PolyServe": "-",
}

# Display names for methods (cleaner labels)
METHOD_LABELS = {
    "SGLang-PD": "SGLang-PD",
    "Niyama": "Niyama",
    "SGLang-Random": "SGLang-Random",
    "Chunked-Prefill": "Chunked-Prefill",
    "SGLang-Round-Robin": "SGLang-RR",
    "PolyServe": "TierServe",
}

# Display names for traces
TRACE_LABELS = {
    "sharegpt": "ShareGPT",
    "lmsys": "LMSYS-Chat",
    "splitwise": "Splitwise",
    "uniform_512_512": "Uniform (512/512)",
    "uniform_4096_1024": "Uniform (4K/1K)",
}

# =============================================================================
# Filtering Configuration - Edit these to control what gets plotted
# =============================================================================

# Methods to include in plots (set to None to include all)
METHODS_TO_PLOT = ["PolyServe", "Niyama", "SGLang-PD", "Chunked-Prefill", "SGLang-Round-Robin"]

# Traces to include in plots (set to None to include all)
TRACES_TO_PLOT = ["sharegpt", "lmsys", "splitwise"] # , "uniform_512_512", "uniform_4096_1024"

# Order for consistent plotting
METHOD_ORDER = ["SGLang-Round-Robin", "Chunked-Prefill", "Niyama", "SGLang-PD", "PolyServe" ]
TRACE_ORDER = ["sharegpt", "lmsys", "splitwise"] #, "uniform_512_512", "uniform_4096_1024"

# Target attainment threshold
TARGET_ATTAINMENT = 0.90

# =============================================================================
# Helper Functions
# =============================================================================

def get_style(method: str):
    """Get plotting style for a method."""
    return {
        "color": COLORS.get(method, "#333333"),
        "marker": MARKERS.get(method, "o"),
        "linestyle": LINESTYLES.get(method, "-"),
        "label": METHOD_LABELS.get(method, method),
    }


def load_and_filter_data(input_path: str) -> pd.DataFrame:
    """Load CSV data and apply filtering based on METHODS_TO_PLOT and TRACES_TO_PLOT."""
    df = pd.read_csv(input_path)

    if METHODS_TO_PLOT is not None:
        df = df[df["method"].isin(METHODS_TO_PLOT)]
    if TRACES_TO_PLOT is not None:
        df = df[df["trace"].isin(TRACES_TO_PLOT)]

    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Traces: {df['trace'].unique().tolist()}")
    print(f"Methods: {df['method'].unique().tolist()}")

    return df


def sort_methods(methods):
    """Sort methods by predefined order."""
    return sorted(methods, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)


def sort_traces(traces):
    """Sort traces by predefined order."""
    return sorted(traces, key=lambda t: TRACE_ORDER.index(t) if t in TRACE_ORDER else 99)


def plot_single_trace(ax, df_trace: pd.DataFrame, trace_name: str, show_legend: bool = False):
    """Plot rate vs attainment for a single trace on given axes."""
    methods = df_trace["method"].unique()
    methods = sort_methods(methods)

    for method in methods:
        df_method = df_trace[df_trace["method"] == method].sort_values("rate")
        style = get_style(method)

        ax.plot(
            df_method["rate"],
            df_method["attainment"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=style["label"],
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5,
        )

    # Set x-axis limits based on data range with padding
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
    ax.set_ylabel("SLO Attainment")
    ax.set_title(TRACE_LABELS.get(trace_name, trace_name))
    ax.set_ylim(0, 1.05)

    if show_legend:
        ax.legend(loc="lower left", ncol=1)


def ensure_output_dir(output_path: str):
    """Ensure the output directory exists."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def find_rate_jsonl(results_dir: str, trace: str, method: str, rate: float) -> str:
    """Find the JSONL file for a given trace, method, and rate.

    Rate is encoded in filename as e.g., rate_3p1250 for 3.125
    If exact rate not found, picks the closest smaller rate.
    """
    import glob
    import re

    # Search for all JSONL files
    pattern = os.path.join(
        results_dir, trace, method, "client_log", "full_log", "rate_*.jsonl"
    )
    all_files = glob.glob(pattern)

    if not all_files:
        return None

    # Parse rates from filenames
    rate_files = []
    for f in all_files:
        basename = os.path.basename(f)
        # Extract rate from filename like "rate_400p0000_20251209_141332.jsonl"
        match = re.match(r"rate_(\d+p\d+)_", basename)
        if match:
            rate_str = match.group(1).replace("p", ".")
            try:
                file_rate = float(rate_str)
                rate_files.append((file_rate, f))
            except ValueError:
                pass

    if not rate_files:
        return None

    # Sort by rate
    rate_files.sort(key=lambda x: x[0])

    # Try exact match first
    for file_rate, f in rate_files:
        if abs(file_rate - rate) < 0.001:
            return f

    # Find closest smaller rate
    smaller_rates = [(r, f) for r, f in rate_files if r < rate]
    if smaller_rates:
        return smaller_rates[-1][1]  # Largest rate that's still smaller

    # If no smaller rate, return the smallest available
    return rate_files[0][1]


def classify_violation(record: dict, threshold: float = 0.1) -> str:
    """Classify a request record into violation category.

    Args:
        record: Request record from JSONL
        threshold: Threshold for minor vs major TPOT violation (default 0.1 = 10%)

    Returns one of:
    - 'pass': Request satisfied SLO
    - 'ttft': TTFT violation (or failed request)
    - 'tpot_minor': TPOT violation with < threshold tokens violated
    - 'tpot_major': TPOT violation with >= threshold tokens violated
    """
    # Check if request passed
    if record.get("slo_satisfied", False):
        return "pass"

    # Check TTFT violation
    ttft_ms = record.get("ttft_ms")
    target_ttft_ms = record.get("target_ttft_ms")

    # Failed request or TTFT violation
    if ttft_ms is None or target_ttft_ms is None or ttft_ms > target_ttft_ms:
        return "ttft"

    # TTFT passed, so it's a TPOT violation
    # Check how many tokens violated
    slo_violations = record.get("slo_violations", 0)
    real_output_len = record.get("real_output_len", 1)

    if real_output_len > 0:
        violation_ratio = slo_violations / real_output_len
    else:
        violation_ratio = 0

    if violation_ratio < threshold:
        return "tpot_minor"
    else:
        return "tpot_major"
