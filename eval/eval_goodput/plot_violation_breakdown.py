#!/usr/bin/env python3
"""
Plot SLO violation breakdown as stacked bar chart.

Shows breakdown of violations by type (TTFT, TPOT minor, TPOT major) for each method
at a specific trace and rate.

Usage:
    python plot_violation_breakdown.py --trace sharegpt --rate 3.125
    python plot_violation_breakdown.py --trace lmsys --rate 5.0 --output results/figures/breakdown.pdf
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import (
    find_rate_jsonl,
    classify_violation,
    ensure_output_dir,
    sort_methods,
    METHOD_LABELS,
    METHOD_ORDER,
    TRACE_LABELS,
)


# Use matplotlib Blues colormap for gradient (dark to light = bad to good)
import matplotlib.cm as cm
_blues = cm.get_cmap("Blues")

VIOLATION_COLORS = {
    "ttft": _blues(0.9),        # Darkest blue (worst)
    "tpot_major": _blues(0.7),  # Medium blue
    "tpot_minor": _blues(0.4),  # Light blue
    "pass": "#b0b0b0",          # Medium gray (best)
}

def get_violation_labels(threshold: float) -> dict:
    """Get violation labels based on threshold percentage."""
    pct = int(threshold * 100)
    return {
        "ttft": "TTFT violation",
        "tpot_major": f"TPOT (≥{pct}%)",
        "tpot_minor": f"TPOT (<{pct}%)",
        "pass": "Pass",
    }


def load_and_classify_requests(jsonl_path: str, threshold: float = 0.1) -> dict:
    """Load JSONL file and classify each request.

    Returns dict with counts for each category.
    """
    counts = {"pass": 0, "ttft": 0, "tpot_minor": 0, "tpot_major": 0}

    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                category = classify_violation(record, threshold=threshold)
                counts[category] += 1

    return counts


def plot_violation_breakdown(
    results_dir: str,
    trace: str,
    rate: float,
    output_path: str,
    methods: list = None,
    threshold: float = 0.1,
):
    """Create stacked bar chart of violation breakdown."""
    if methods is None:
        methods = METHOD_ORDER

    # Filter to methods that exist
    methods = [m for m in methods if m in METHOD_ORDER or m in METHOD_LABELS]
    methods = sort_methods(methods)

    # Get labels based on threshold
    violation_labels = get_violation_labels(threshold)

    # Collect data for each method
    data = {}
    for method in methods:
        jsonl_path = find_rate_jsonl(results_dir, trace, method, rate)
        if jsonl_path and os.path.exists(jsonl_path):
            data[method] = load_and_classify_requests(jsonl_path, threshold=threshold)
        else:
            print(f"Warning: No data found for {method} at trace={trace}, rate={rate}")

    if not data:
        print("No data found for any method")
        return

    # Prepare data for plotting
    methods_with_data = [m for m in methods if m in data]
    n_methods = len(methods_with_data)

    # Calculate percentages (order: ttft at bottom, then tpot_major, tpot_minor, pass on top)
    categories = ["ttft", "tpot_major", "tpot_minor", "pass"]
    percentages = {cat: [] for cat in categories}

    for method in methods_with_data:
        counts = data[method]
        total = sum(counts.values())
        if total > 0:
            for cat in categories:
                percentages[cat].append(100 * counts[cat] / total)
        else:
            for cat in categories:
                percentages[cat].append(0)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 3))

    x = np.arange(n_methods)
    width = 0.6

    # Stack bars from bottom to top
    bottom = np.zeros(n_methods)
    for cat in categories:
        ax.bar(
            x,
            percentages[cat],
            width,
            bottom=bottom,
            label=violation_labels[cat],
            color=VIOLATION_COLORS[cat],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += np.array(percentages[cat])

    # Labels and formatting
    ax.set_xlabel("")
    ax.set_ylabel("Percentage of Requests", fontsize=14)
    ax.set_title(f"{TRACE_LABELS.get(trace, trace)} @ {rate} req/s", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_LABELS.get(m, m) for m in methods_with_data],
        fontsize=13,
    )
    ax.tick_params(axis='y', labelsize=13)
    ax.set_ylim(0, 100)

    # Legend (reverse order so Pass is at top, matching visual)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper right", fontsize=10)

    plt.tight_layout()
    ensure_output_dir(output_path)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot SLO violation breakdown")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Base results directory (default: results)",
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Trace name (e.g., sharegpt, lmsys)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Request rate to analyze (e.g., 3.125)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: results/figures/violation_breakdown_{trace}_{rate}.pdf)",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format (default: pdf)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold for minor vs major TPOT violation (default: 0.1 = 10%%)",
    )
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        rate_str = f"{args.rate}".replace(".", "p")
        args.output = f"results/figures/violation_breakdown_{args.trace}_{rate_str}.{args.format}"
    elif args.format:
        base, _ = os.path.splitext(args.output)
        args.output = f"{base}.{args.format}"

    plot_violation_breakdown(
        results_dir=args.results_dir,
        trace=args.trace,
        rate=args.rate,
        output_path=args.output,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
