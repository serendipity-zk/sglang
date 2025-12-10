#!/usr/bin/env python3
"""
Plot per-tier attainment comparison (TTFT, TPOT).

Usage:
    python plot_per_tier.py
    python plot_per_tier.py --trace sharegpt
    python plot_per_tier.py --input results/aggregated_results.csv --output results/figures/per_tier.pdf
"""

import argparse
import os

import matplotlib.pyplot as plt

from plot_utils import (
    load_and_filter_data,
    sort_traces,
    sort_methods,
    get_style,
    ensure_output_dir,
    TRACE_LABELS,
    TARGET_ATTAINMENT,
)


def plot_per_tier_comparison(df, output_path: str, trace: str = None):
    """Plot per-tier attainment comparison (TTFT, TPOT)."""
    if trace:
        traces = [trace]
    else:
        traces = df["trace"].unique()
        traces = sort_traces(traces)

    tier_cols = [c for c in df.columns if c.startswith("tier_")]
    if not tier_cols:
        print("No tier columns found, skipping per-tier plot")
        return

    n_traces = len(traces)
    n_tiers = min(3, len(tier_cols))
    fig, axes = plt.subplots(n_traces, n_tiers, figsize=(3.3 * n_tiers, 2.5 * n_traces), squeeze=False)

    # Label tiers as TTFT and TPOT
    tier_labels = {
        "tier_10ms": "TPOT=10ms",
        "tier_20ms": "TPOT=20ms",
        "tier_40ms": "TPOT=40ms",
    }

    for row, tr in enumerate(traces):
        df_trace = df[df["trace"] == tr]
        methods = df_trace["method"].unique()
        methods = sort_methods(methods)

        for col, tier in enumerate(tier_cols[:n_tiers]):
            ax = axes[row, col]

            for method in methods:
                df_method = df_trace[df_trace["method"] == method].sort_values("rate")
                style = get_style(method)

                if tier in df_method.columns:
                    ax.plot(
                        df_method["rate"],
                        df_method[tier],
                        color=style["color"],
                        marker=style["marker"],
                        linestyle=style["linestyle"],
                        label=style["label"] if row == 0 else None,
                        markerfacecolor="white",
                        markeredgewidth=1.2,
                        markersize=4,
                    )

            # Set x-axis limits based on data range
            all_rates = df_trace["rate"]
            min_rate, max_rate = all_rates.min(), all_rates.max()
            x_padding = (max_rate - min_rate) * 0.05
            ax.set_xlim(max(0, min_rate - x_padding), max_rate + x_padding)

            ax.axhline(y=TARGET_ATTAINMENT, color="gray", linestyle=":", linewidth=1, alpha=0.7)
            ax.set_ylim(0, 1.05)

            if row == n_traces - 1:
                ax.set_xlabel("Request Rate (req/s)")
            if col == 0:
                ax.set_ylabel(TRACE_LABELS.get(tr, tr))
            if row == 0:
                ax.set_title(tier_labels.get(tier, tier))

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                   bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_output_dir(output_path)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot per-tier attainment comparison")
    parser.add_argument(
        "--input",
        default="results/aggregated_results.csv",
        help="Input CSV file (default: results/aggregated_results.csv)",
    )
    parser.add_argument(
        "--output",
        default="results/figures/per_tier_attainment.pdf",
        help="Output file path (default: results/figures/per_tier_attainment.pdf)",
    )
    parser.add_argument(
        "--format",
        default=None,
        choices=["pdf", "png", "svg"],
        help="Output format (overrides extension in --output)",
    )
    parser.add_argument(
        "--trace",
        default=None,
        help="Specific trace to plot (default: plot all traces)",
    )
    args = parser.parse_args()

    df = load_and_filter_data(args.input)

    output_path = args.output
    if args.format:
        base = output_path.rsplit(".", 1)[0]
        output_path = f"{base}.{args.format}"

    # If specific trace, modify output filename
    if args.trace:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_{args.trace}{ext}"

    plot_per_tier_comparison(df, output_path, trace=args.trace)


if __name__ == "__main__":
    main()
