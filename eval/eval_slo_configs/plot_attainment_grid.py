#!/usr/bin/env python3
"""
Plot rate vs attainment grid - all traces in one figure.

Usage:
    python plot_attainment_grid.py
    python plot_attainment_grid.py --input results/aggregated_results.csv --output results/figures/grid.pdf
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_utils import (
    load_and_filter_data,
    sort_traces,
    sort_configs,
    get_style,
    plot_single_trace,
    ensure_output_dir,
    TARGET_ATTAINMENT,
)


def interpolate_rate_at_attainment(df_config, target_attainment):
    """
    Find the rate at which min_tier_attainment crosses the target threshold using linear interpolation.

    Returns the interpolated rate, or None if no crossing is found.
    """
    df_sorted = df_config.sort_values("rate")
    rates = df_sorted["rate"].values
    attainments = df_sorted["min_tier_attainment"].values

    # Find the crossing point (attainment goes from >= target to < target)
    for i in range(len(attainments) - 1):
        att_curr = attainments[i]
        att_next = attainments[i + 1]

        # Check if the target attainment is crossed between these two points
        if att_curr >= target_attainment > att_next:
            # Linear interpolation
            rate_curr = rates[i]
            rate_next = rates[i + 1]

            if att_next != att_curr:
                # Interpolate rate at target attainment
                rate_at_target = rate_curr + (target_attainment - att_curr) * (rate_next - rate_curr) / (att_next - att_curr)
                return rate_at_target
            else:
                return rate_curr

    # If attainment is always >= target, return the max rate
    if attainments[-1] >= target_attainment:
        return rates[-1]

    # If attainment is always < target, return None (cannot achieve target)
    if attainments[0] < target_attainment:
        return None

    return None


def print_trace_statistics(df, target_attainment=TARGET_ATTAINMENT):
    """Print statistics for each trace showing goodput at target attainment and gains."""
    traces = df["trace"].unique()
    traces = sort_traces(traces)

    print("\n" + "=" * 80)
    print(f"STATISTICS: Goodput (rate at {int(target_attainment * 100)}% min-tier attainment)")
    print("=" * 80)

    for trace in traces:
        print(f"\n--- Trace: {trace} ---")
        df_trace = df[df["trace"] == trace]
        configs = df_trace["config"].unique()
        configs = sort_configs(configs)

        # Calculate goodput for each config
        goodputs = {}
        for config in configs:
            df_config = df_trace[df_trace["config"] == config]
            goodput = interpolate_rate_at_attainment(df_config, target_attainment)
            goodputs[config] = goodput

            if goodput is not None:
                print(f"  {config:25s}: {goodput:8.2f} req/s")
            else:
                print(f"  {config:25s}: N/A (never achieves {int(target_attainment * 100)}%)")

        # Calculate gains relative to baseline (no_autoscaling / Static)
        baseline_config = "no_autoscaling"
        if baseline_config in goodputs and goodputs[baseline_config] is not None:
            baseline_goodput = goodputs[baseline_config]
            print(f"\n  Gains over {baseline_config} (Static):")
            for config in configs:
                if config != baseline_config and goodputs[config] is not None:
                    gain = (goodputs[config] - baseline_goodput) / baseline_goodput * 100
                    gain_abs = goodputs[config] - baseline_goodput
                    print(f"    {config:23s}: {gain:+6.1f}% ({gain_abs:+.2f} req/s)")

        # Also show gains relative to ttft (TierServe) if available
        compare_config = "ttft"
        if compare_config in goodputs and goodputs[compare_config] is not None:
            compare_goodput = goodputs[compare_config]
            print(f"\n  Gains over {compare_config} (TierServe):")
            for config in configs:
                if config != compare_config and goodputs[config] is not None:
                    gain = (goodputs[config] - compare_goodput) / compare_goodput * 100
                    gain_abs = goodputs[config] - compare_goodput
                    print(f"    {config:23s}: {gain:+6.1f}% ({gain_abs:+.2f} req/s)")

    print("\n" + "=" * 80)


def plot_all_traces_grid(df, output_path: str):
    """Create a grid of subplots, one per trace."""
    traces = df["trace"].unique()
    traces = sort_traces(traces)

    n_traces = len(traces)
    n_cols = min(3, n_traces)
    n_rows = (n_traces + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 2.5 * n_rows))

    if n_traces == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, trace in enumerate(traces):
        df_trace = df[df["trace"] == trace]
        plot_single_trace(axes[idx], df_trace, trace, show_legend=False)

    # Hide unused subplots
    for idx in range(n_traces, len(axes)):
        axes[idx].set_visible(False)

    # Create legend
    all_configs = df["config"].unique()
    all_configs = sort_configs(all_configs)

    legend_handles = []
    for config in all_configs:
        style = get_style(config)
        handle = Line2D([0], [0], color=style["color"], marker=style["marker"],
                        linestyle=style["linestyle"], label=style["label"],
                        markerfacecolor="white", markeredgewidth=1.2, markersize=5)
        legend_handles.append(handle)

    fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles),
               bbox_to_anchor=(0.5, 1.02), frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_output_dir(output_path)
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot rate vs attainment grid")
    parser.add_argument("--input", default="results/aggregated_results.csv",
                        help="Input CSV file")
    parser.add_argument("--output", default="results/figures/rate_vs_attainment_grid.pdf",
                        help="Output file path")
    parser.add_argument("--format", default=None, choices=["pdf", "png", "svg"],
                        help="Output format (overrides extension)")
    args = parser.parse_args()

    df = load_and_filter_data(args.input)

    output_path = args.output
    if args.format:
        base = output_path.rsplit(".", 1)[0]
        output_path = f"{base}.{args.format}"

    # Print statistics for each trace
    print_trace_statistics(df)

    plot_all_traces_grid(df, output_path)


if __name__ == "__main__":
    main()
