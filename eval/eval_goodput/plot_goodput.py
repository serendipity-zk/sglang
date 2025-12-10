#!/usr/bin/env python3
"""
Plot goodput comparison bar chart.

Goodput = max sustainable rate at target attainment.

Usage:
    python plot_goodput.py
    python plot_goodput.py --target 0.95
    python plot_goodput.py --input results/aggregated_results.csv --output results/figures/goodput.pdf
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import (
    load_and_filter_data,
    sort_traces,
    sort_methods,
    get_style,
    ensure_output_dir,
    TRACE_LABELS,
    TARGET_ATTAINMENT,
)


def plot_goodput_comparison(df, output_path: str, target: float = None):
    """
    Bar chart comparing max sustainable rate at target attainment across methods.
    This is the key "goodput" metric.
    """
    if target is None:
        target = TARGET_ATTAINMENT

    traces = df["trace"].unique()
    traces = sort_traces(traces)

    # Find max rate where attainment >= target for each method/trace
    goodput_data = []
    for trace in traces:
        df_trace = df[df["trace"] == trace]
        for method in df_trace["method"].unique():
            df_m = df_trace[df_trace["method"] == method]
            passing = df_m[df_m["attainment"] >= target]
            max_rate = passing["rate"].max() if not passing.empty else 0
            goodput_data.append({
                "trace": trace,
                "method": method,
                "goodput": max_rate,
            })

    df_goodput = pd.DataFrame(goodput_data)

    methods = df_goodput["method"].unique()
    methods = sort_methods(methods)

    x = np.arange(len(traces))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(7, 3))

    for i, method in enumerate(methods):
        df_m = df_goodput[df_goodput["method"] == method]
        values = [df_m[df_m["trace"] == t]["goodput"].values[0] if t in df_m["trace"].values else 0
                  for t in traces]
        style = get_style(method)
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, values, width * 0.9, label=style["label"],
               color=style["color"], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Workload")
    ax.set_ylabel(f"Goodput (req/s @ {int(target*100)}% SLO)")
    ax.set_xticks(x)
    ax.set_xticklabels([TRACE_LABELS.get(t, t) for t in traces], rotation=15, ha="right")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    ensure_output_dir(output_path)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot goodput comparison bar chart")
    parser.add_argument(
        "--input",
        default="results/aggregated_results.csv",
        help="Input CSV file (default: results/aggregated_results.csv)",
    )
    parser.add_argument(
        "--output",
        default="results/figures/goodput_comparison.pdf",
        help="Output file path (default: results/figures/goodput_comparison.pdf)",
    )
    parser.add_argument(
        "--format",
        default=None,
        choices=["pdf", "png", "svg"],
        help="Output format (overrides extension in --output)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help=f"Target attainment threshold (default: {TARGET_ATTAINMENT})",
    )
    args = parser.parse_args()

    df = load_and_filter_data(args.input)

    output_path = args.output
    if args.format:
        base = output_path.rsplit(".", 1)[0]
        output_path = f"{base}.{args.format}"

    plot_goodput_comparison(df, output_path, target=args.target)


if __name__ == "__main__":
    main()
