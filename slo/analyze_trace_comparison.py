#!/usr/bin/env python3
"""
Script to compare actual submit times with desired trace and analyze burstiness.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime

# File paths
ACTUAL_TRACE = '/sgl-workspace/sglang/slo/logs/rust_client_output.jsonl'
TARGET_TRACE = '/sgl-workspace/sglang/SLO-CSim/trace/1024_1024_152540.csv'
ROUTER_LOG = '/sgl-workspace/sglang/slo/logs/router-log-20251201-030820.txt'


def load_actual_trace(filepath):
    """Load actual execution trace from JSONL file."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Convert to relative timestamps (ms from start)
    if len(df) > 0:
        start_time = df['submit_timestamp'].min()
        df['actual_arrival_ms'] = (df['submit_timestamp'] - start_time) * 1000
        if 'post_timestamp' in df.columns:
            df['post_arrival_ms'] = (df['post_timestamp'] - start_time) * 1000
            df['post_submit_latency_ms'] = (df['post_timestamp'] - df['submit_timestamp']) * 1000

        # Extract request number from request_id (e.g., "req_000005" -> 5)
        # Pattern matches "req_" followed by digits, ignoring any suffix after underscore
        req_raw = df['request_id']
        req_num_numeric = pd.to_numeric(req_raw, errors='coerce')
        req_num_from_pattern = pd.to_numeric(
            req_raw.astype(str).str.extract(r'req_(\d+)')[0],
            errors='coerce'
        )
        # Fallback to any first digits if pattern doesn't match
        req_num_any_digits = pd.to_numeric(
            req_raw.astype(str).str.extract(r'(\d+)')[0],
            errors='coerce'
        )
        req_num = req_num_numeric.fillna(req_num_from_pattern).fillna(req_num_any_digits)
        # Use ordinal position as final fallback
        req_num = req_num.fillna(pd.Series(range(len(df)), index=df.index))
        df['req_num'] = req_num

    return df.sort_values('req_num').reset_index(drop=True)


def load_target_trace(filepath):
    """Load target trace from CSV file."""
    df = pd.read_csv(filepath)
    df = df.rename(columns={'arrival': 'target_arrival_ms'})
    df['req_num'] = range(len(df))
    return df


def load_router_log(filepath):
    """Load router log and extract 'Request added' timestamps."""
    timestamps = []
    queues = []
    tpots = []
    request_ids = []

    # Pattern to extract: Timestamp 1764548621.865, request_id=req_000005_622090, Request added to queue 2: target_tpot=Some(40.0)
    pattern = r'Timestamp\s+([\d.]+),\s+request_id=([^,]+),\s+Request added to queue\s+(\d+):\s+target_tpot=Some\(([\d.]+)\)'

    with open(filepath, 'r') as f:
        for line in f:
            if 'Request added' in line:
                match = re.search(pattern, line)
                if match:
                    timestamp = float(match.group(1))
                    request_id = match.group(2)
                    queue_idx = int(match.group(3))
                    tpot = float(match.group(4))
                    timestamps.append(timestamp)
                    request_ids.append(request_id)
                    queues.append(queue_idx)
                    tpots.append(tpot)

    df = pd.DataFrame({
        'router_timestamp': timestamps,
        'request_id': request_ids,
        'queue_idx': queues,
        'target_tpot': tpots
    })

    if len(df) > 0:
        # Convert to relative timestamps (ms from start)
        start_time = df['router_timestamp'].min()
        df['router_arrival_ms'] = (df['router_timestamp'] - start_time) * 1000
        # Extract request number from request_id (e.g., "req_000005" -> 5)
        req_raw = df['request_id']
        req_num_numeric = pd.to_numeric(req_raw, errors='coerce')
        req_num_from_pattern = pd.to_numeric(
            req_raw.astype(str).str.extract(r'req_(\d+)')[0],
            errors='coerce'
        )
        # Fallback to any first digits if pattern doesn't match
        req_num_any_digits = pd.to_numeric(
            req_raw.astype(str).str.extract(r'(\d+)')[0],
            errors='coerce'
        )
        req_num = req_num_numeric.fillna(req_num_from_pattern).fillna(req_num_any_digits)
        # Use ordinal position as final fallback
        req_num = req_num.fillna(pd.Series(range(len(df)), index=df.index))
        df['req_num'] = req_num

    return df


def calculate_inter_arrival_times(timestamps):
    """Calculate inter-arrival times from timestamps."""
    sorted_times = np.sort(timestamps)
    return np.diff(sorted_times)


def analyze_burstiness(inter_arrival_times, window_size=10):
    """
    Analyze burstiness in arrival pattern.

    Returns:
        - coefficient of variation (CV)
        - burstiness index (B)
        - rolling statistics
    """
    # Coefficient of Variation (CV = std/mean)
    mean_iat = np.mean(inter_arrival_times)
    std_iat = np.std(inter_arrival_times)
    cv = std_iat / mean_iat if mean_iat > 0 else 0

    # Burstiness index: B = (CV - 1) / (CV + 1)
    # B = -1: regular, B = 0: random (Poisson), B = 1: bursty
    burstiness_index = (cv - 1) / (cv + 1) if cv > 0 else 0

    # Rolling statistics
    rolling_mean = pd.Series(inter_arrival_times).rolling(window=window_size, min_periods=1).mean()
    rolling_std = pd.Series(inter_arrival_times).rolling(window=window_size, min_periods=1).std()

    return {
        'mean': mean_iat,
        'std': std_iat,
        'cv': cv,
        'burstiness_index': burstiness_index,
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std
    }


def find_burst_periods(inter_arrival_times, threshold_factor=0.5):
    """
    Identify burst periods where inter-arrival time is significantly below average.

    threshold_factor: multiplier for mean (e.g., 0.5 means bursts are < 50% of mean)
    """
    mean_iat = np.mean(inter_arrival_times)
    threshold = mean_iat * threshold_factor

    burst_indices = np.where(inter_arrival_times < threshold)[0]

    # Group consecutive bursts
    burst_periods = []
    if len(burst_indices) > 0:
        current_burst = [burst_indices[0]]
        for i in range(1, len(burst_indices)):
            if burst_indices[i] == burst_indices[i-1] + 1:
                current_burst.append(burst_indices[i])
            else:
                burst_periods.append(current_burst)
                current_burst = [burst_indices[i]]
        burst_periods.append(current_burst)

    return burst_periods, threshold


def summarize_time_series(series_name: str, time_ms: pd.Series):
    """Compute inter-arrival stats and burstiness for a time series."""
    inter_arrivals = calculate_inter_arrival_times(time_ms.values)
    burst = analyze_burstiness(inter_arrivals)
    return {
        'name': series_name,
        'count': len(time_ms),
        'duration_ms': (time_ms.max() - time_ms.min()) if len(time_ms) > 0 else 0,
        'mean_iat': burst['mean'],
        'std_iat': burst['std'],
        'cv': burst['cv'],
        'burstiness_index': burst['burstiness_index'],
        'inter_arrivals': inter_arrivals,
        'rolling_mean': burst['rolling_mean'],
        'rolling_std': burst['rolling_std'],
    }


def compare_series(left_df, right_df, left_col, right_col, label):
    """Align two series by req_num and compute arrival differences."""
    merged = pd.merge(
        left_df[['req_num', left_col]],
        right_df[['req_num', right_col]],
        on='req_num',
        how='inner'
    ).sort_values('req_num')
    if len(merged) == 0:
        return merged, None
    diff_col = f"{label}_diff_ms"
    merged[diff_col] = merged[left_col] - merged[right_col]
    stats = {
        'mean': merged[diff_col].mean(),
        'median': merged[diff_col].median(),
        'std': merged[diff_col].std(),
        'min': merged[diff_col].min(),
        'max': merged[diff_col].max(),
    }
    return merged, stats


def main():
    print("=" * 80)
    print("TRACE COMPARISON AND BURSTINESS ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    print("Loading traces...")
    actual_df = load_actual_trace(ACTUAL_TRACE)
    target_df = load_target_trace(TARGET_TRACE)
    router_df = load_router_log(ROUTER_LOG)

    print(f"Client trace: {len(actual_df)} requests")
    print(f"Target trace: {len(target_df)} requests")
    print(f"Router log: {len(router_df)} requests")
    print()

    post_duration_s = None
    post_latency_summary = None

    # Calculate actual request rate
    if len(actual_df) > 1:
        actual_duration_s = (actual_df['actual_arrival_ms'].max() - actual_df['actual_arrival_ms'].min()) / 1000
        actual_rate = (len(actual_df) - 1) / actual_duration_s  # requests per second
        print("=" * 80)
        print("ACTUAL TRACE RATE CALCULATION")
        print("=" * 80)
        print(f"Duration: {actual_duration_s:.2f} seconds")
        print(f"Total requests: {len(actual_df)}")
        print(f"Actual rate: {actual_rate:.3f} requests/second")
        print()

        # Scale target trace to match actual rate
        # Target trace is at some baseline rate, we need to scale it
        target_duration_s = target_df['target_arrival_ms'].max() / 1000
        target_baseline_rate = len(target_df) / target_duration_s if target_duration_s > 0 else 1
        rate_factor = actual_rate / target_baseline_rate

        print(f"Target baseline rate: {target_baseline_rate:.3f} requests/second")
        print(f"Rate scaling factor: {rate_factor:.3f}")
        print()

        # Apply scaling to target arrival times
        target_df['target_arrival_ms_scaled'] = target_df['target_arrival_ms'] / rate_factor
        target_df['target_arrival_ms'] = target_df['target_arrival_ms_scaled']

        print(f"Target trace scaled from {target_duration_s:.2f}s to {target_df['target_arrival_ms'].max()/1000:.2f}s")
        print()

        if 'post_arrival_ms' in actual_df.columns:
            post_duration_s = (actual_df['post_arrival_ms'].max() - actual_df['post_arrival_ms'].min()) / 1000
            print(f"Post-arrival duration : {post_duration_s:.2f} seconds")
            print()

        if 'post_submit_latency_ms' in actual_df.columns:
            lat_series = actual_df['post_submit_latency_ms'].dropna()
            if len(lat_series) > 0:
                post_latency_summary = {
                    'mean': lat_series.mean(),
                    'median': lat_series.median(),
                    'std': lat_series.std(),
                    'min': lat_series.min(),
                    'max': lat_series.max(),
                }
                print("Submit → Post latency (ms):")
                print(f"  Mean   : {post_latency_summary['mean']:.2f}")
                print(f"  Median : {post_latency_summary['median']:.2f}")
                print(f"  Std    : {post_latency_summary['std']:.2f}")
                print(f"  Min    : {post_latency_summary['min']:.2f}")
                print(f"  Max    : {post_latency_summary['max']:.2f}")
                print()
    else:
        print("Not enough data to calculate rate")
        return

    # Align by req_num so comparisons use the same ordinal request
    if len(actual_df) > 0:
        max_submit_req = actual_df['req_num'].max()
        target_df = target_df[target_df['req_num'] <= max_submit_req]
    target_series = target_df[['req_num', 'target_arrival_ms']].copy()
    submit_series = actual_df[['req_num', 'actual_arrival_ms']].copy()
    router_series = router_df[['req_num', 'router_arrival_ms']].copy()

    # Summaries per series
    series_summaries = [
        summarize_time_series("Target (trace)", target_series['target_arrival_ms']) if len(target_series) else None,
        summarize_time_series("Submit (client)", submit_series['actual_arrival_ms']) if len(submit_series) else None,
        summarize_time_series("Router", router_series['router_arrival_ms']) if len(router_series) else None,
    ]
    series_summaries = [s for s in series_summaries if s is not None]

    print("=" * 80)
    print("SERIES BURSTINESS (target / submit / router)")
    print("=" * 80)
    for s in series_summaries:
        print(f"{s['name']}: count={s['count']}, duration={s['duration_ms']:.2f} ms")
        print(f"  Mean IAT: {s['mean_iat']:.2f} ms, Std: {s['std_iat']:.2f} ms, CV: {s['cv']:.3f}, B: {s['burstiness_index']:.3f}")
    print()

    # Pairwise comparisons (aligned by req_num)
    target_submit_aligned, target_submit_stats = compare_series(
        target_series, submit_series, 'target_arrival_ms', 'actual_arrival_ms', 'submit_vs_target'
    )
    post_target_stats = None
    post_target_aligned = pd.DataFrame()
    submit_router_aligned, submit_router_stats = compare_series(
        submit_series, router_series, 'actual_arrival_ms', 'router_arrival_ms', 'router_vs_submit'
    )
    target_router_aligned, target_router_stats = compare_series(
        target_series, router_series, 'target_arrival_ms', 'router_arrival_ms', 'router_vs_target'
    )

    def print_comparison(label, stats):
        if not stats:
            print(f"{label}: no overlapping data")
            return
        print(f"{label}: mean={stats['mean']:.2f} ms, median={stats['median']:.2f} ms, "
              f"std={stats['std']:.2f} ms, min={stats['min']:.2f} ms, max={stats['max']:.2f} ms")

    print("=" * 80)
    print("PAIRWISE ARRIVAL DIFFERENCES (aligned by req_num)")
    print("=" * 80)
    print_comparison("Submit - Target", target_submit_stats)
    if 'post_arrival_ms' in actual_df.columns:
        post_series = actual_df[['req_num', 'post_arrival_ms']].copy()
        post_target_aligned, post_target_stats = compare_series(
            target_series, post_series, 'target_arrival_ms', 'post_arrival_ms', 'post_vs_target'
        )
        print_comparison("Post - Target", post_target_stats)
    print_comparison("Router - Submit", submit_router_stats)
    print_comparison("Router - Target", target_router_stats)
    print()

    # Calculate inter-arrival times for plotting/other analyses
    actual_iat = calculate_inter_arrival_times(submit_series['actual_arrival_ms'].values)
    target_iat = calculate_inter_arrival_times(target_series['target_arrival_ms'].values)
    router_iat = calculate_inter_arrival_times(router_series['router_arrival_ms'].values)

    # Calculate inter-arrival times
    actual_iat = calculate_inter_arrival_times(actual_df['actual_arrival_ms'].values)
    target_iat = calculate_inter_arrival_times(target_df['target_arrival_ms'].values)
    router_iat = calculate_inter_arrival_times(router_df['router_arrival_ms'].values)

    # Analyze burstiness - Client
    print("=" * 80)
    print("BURSTINESS ANALYSIS - CLIENT TRACE")
    print("=" * 80)
    actual_burst = analyze_burstiness(actual_iat)
    print(f"Mean inter-arrival time: {actual_burst['mean']:.2f} ms")
    print(f"Std inter-arrival time: {actual_burst['std']:.2f} ms")
    print(f"Coefficient of Variation (CV): {actual_burst['cv']:.3f}")
    print(f"Burstiness Index (B): {actual_burst['burstiness_index']:.3f}")
    print(f"  (-1 = regular, 0 = random/Poisson, 1 = highly bursty)")
    print()

    # Analyze burstiness - Router
    print("=" * 80)
    print("BURSTINESS ANALYSIS - ROUTER LOG")
    print("=" * 80)
    router_burst = analyze_burstiness(router_iat)
    print(f"Mean inter-arrival time: {router_burst['mean']:.2f} ms")
    print(f"Std inter-arrival time: {router_burst['std']:.2f} ms")
    print(f"Coefficient of Variation (CV): {router_burst['cv']:.3f}")
    print(f"Burstiness Index (B): {router_burst['burstiness_index']:.3f}")
    print()

    print("=" * 80)
    print("BURSTINESS ANALYSIS - TARGET TRACE")
    print("=" * 80)
    target_burst = analyze_burstiness(target_iat)
    print(f"Mean inter-arrival time: {target_burst['mean']:.2f} ms")
    print(f"Std inter-arrival time: {target_burst['std']:.2f} ms")
    print(f"Coefficient of Variation (CV): {target_burst['cv']:.3f}")
    print(f"Burstiness Index (B): {target_burst['burstiness_index']:.3f}")
    print()

    # Compare client and router timestamps
    print("=" * 80)
    print("CLIENT vs ROUTER TIMING")
    print("=" * 80)
    if len(actual_df) > 0 and len(router_df) > 0:
        # Check if logs are from the same run (time gap < 60 seconds)
        client_start = actual_df['submit_timestamp'].min()
        router_start = router_df['router_timestamp'].min()
        time_gap_s = abs(router_start - client_start)

        if time_gap_s > 60:  # More than 60 seconds apart
            print(f"⚠️  WARNING: Logs are from DIFFERENT test runs!")
            print(f"   Client start: {datetime.fromtimestamp(client_start)}")
            print(f"   Router start: {datetime.fromtimestamp(router_start)}")
            print(f"   Time gap: {time_gap_s/3600:.2f} hours ({time_gap_s:.1f} seconds)")
            print()
            print("Skipping client-router delay analysis (not comparable)")
            delays_ms = None
        else:
            # Match client and router logs by request_id
            if 'request_id' in actual_df.columns and 'request_id' in router_df.columns:
                # Merge by request_id
                matched = pd.merge(
                    actual_df[['request_id', 'submit_timestamp']],
                    router_df[['request_id', 'router_timestamp']],
                    on='request_id',
                    how='inner'
                )

                if len(matched) > 0:
                    delays_ms = (matched['router_timestamp'] - matched['submit_timestamp']) * 1000
                    print(f"✓ Logs are from the same run (gap: {time_gap_s:.2f}s)")
                    print(f"Number of matched requests (by ID): {len(matched)}")
                    print(f"Mean client->router delay: {np.mean(delays_ms):.2f} ms")
                    print(f"Median client->router delay: {np.median(delays_ms):.2f} ms")
                    print(f"Std client->router delay: {np.std(delays_ms):.2f} ms")
                    print(f"Min delay: {np.min(delays_ms):.2f} ms")
                    print(f"Max delay: {np.max(delays_ms):.2f} ms")
                else:
                    print(f"⚠️  No matching request IDs found between client and router logs")
                    delays_ms = None
            else:
                print(f"⚠️  Missing request_id field in logs - cannot match by ID")
                print(f"   Client has request_id: {'request_id' in actual_df.columns}")
                print(f"   Router has request_id: {'request_id' in router_df.columns}")
                delays_ms = None
        print()
    else:
        delays_ms = None
        print("Cannot compare - mismatched data")
        print()

    # Find burst periods in client trace
    burst_periods, threshold = find_burst_periods(actual_iat, threshold_factor=0.5)
    print("=" * 80)
    print("BURST PERIODS DETECTED - CLIENT (Inter-arrival < 50% of mean)")
    print("=" * 80)
    print(f"Threshold: {threshold:.2f} ms")
    print(f"Number of burst periods: {len(burst_periods)}")
    print()

    if burst_periods:
        print("Top 10 burst periods:")
        # Sort by length
        burst_periods_sorted = sorted(burst_periods, key=len, reverse=True)[:10]
        for i, period in enumerate(burst_periods_sorted, 1):
            start_idx = period[0]
            end_idx = period[-1]
            duration = len(period)
            avg_iat_in_burst = np.mean(actual_iat[period])
            print(f"{i}. Requests {start_idx}-{end_idx+1}: {duration} consecutive bursts, "
                  f"avg IAT: {avg_iat_in_burst:.2f} ms")
    print()

    # Find burst periods in router trace
    router_burst_periods, router_threshold = find_burst_periods(router_iat, threshold_factor=0.5)
    print("=" * 80)
    print("BURST PERIODS DETECTED - ROUTER (Inter-arrival < 50% of mean)")
    print("=" * 80)
    print(f"Threshold: {router_threshold:.2f} ms")
    print(f"Number of burst periods: {len(router_burst_periods)}")
    print()

    if router_burst_periods:
        print("Top 10 burst periods:")
        # Sort by length
        router_burst_sorted = sorted(router_burst_periods, key=len, reverse=True)[:10]
        for i, period in enumerate(router_burst_sorted, 1):
            start_idx = period[0]
            end_idx = period[-1]
            duration = len(period)
            avg_iat_in_burst = np.mean(router_iat[period])
            print(f"{i}. Requests {start_idx}-{end_idx+1}: {duration} consecutive bursts, "
                  f"avg IAT: {avg_iat_in_burst:.2f} ms")
    print()

    # Visualizations
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    fig, axes = plt.subplots(4, 2, figsize=(16, 16))

    # 1. Arrival times comparison
    ax = axes[0, 0]
    if len(target_submit_aligned) > 0:
        ax.plot(target_submit_aligned['req_num'], target_submit_aligned['target_arrival_ms'] / 1000,
                label='Target', alpha=0.7, marker='o', markersize=2)
        ax.plot(target_submit_aligned['req_num'], target_submit_aligned['actual_arrival_ms'] / 1000,
                label='Submit', alpha=0.7, marker='x', markersize=2)
    ax.set_xlabel('Request (req_num)')
    ax.set_ylabel('Arrival Time (seconds)')
    ax.set_title('Arrival Times: Target vs Submit')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Arrival time differences
    ax = axes[0, 1]
    if len(target_submit_aligned) > 0:
        ax.plot(
            target_submit_aligned['req_num'],
            target_submit_aligned['submit_vs_target_diff_ms'],
            marker='o',
            markersize=2,
            label='Submit - Target',
        )
    if 'post_arrival_ms' in actual_df.columns and len(target_submit_aligned) > 0:
        post_series = actual_df[['req_num', 'post_arrival_ms']].copy()
        post_target_aligned, _ = compare_series(
            target_series, post_series, 'target_arrival_ms', 'post_arrival_ms', 'post_vs_target'
        )
        if len(post_target_aligned) > 0:
            ax.plot(
                post_target_aligned['req_num'],
                post_target_aligned['post_vs_target_diff_ms'],
                marker='x',
                markersize=2,
                linewidth=1,
                linestyle='--',
                label='Post - Target',
                alpha=0.8,
            )
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Request (req_num)')
    ax.set_ylabel('Arrival Difference (ms)')
    ax.set_title('Arrival Time Differences')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Inter-arrival time comparison - Client vs Router
    ax = axes[1, 0]
    ax.plot(range(len(actual_iat)), actual_iat, label='Client', alpha=0.7, linewidth=1)
    ax.plot(range(len(router_iat)), router_iat, label='Router', alpha=0.7, linewidth=1)
    ax.axhline(y=actual_burst['mean'], color='blue', linestyle='--',
               alpha=0.5, label=f"Client Mean ({actual_burst['mean']:.1f}ms)")
    ax.axhline(y=router_burst['mean'], color='green', linestyle='--',
               alpha=0.5, label=f"Router Mean ({router_burst['mean']:.1f}ms)")
    ax.set_xlabel('Request Pair Index')
    ax.set_ylabel('Inter-Arrival Time (ms)')
    ax.set_title('Inter-Arrival Times: Client vs Router')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Inter-arrival time distribution
    ax = axes[1, 1]
    ax.hist(actual_iat, bins=50, alpha=0.5, label='Client', density=True)
    ax.hist(router_iat, bins=50, alpha=0.5, label='Router', density=True)
    ax.set_xlabel('Inter-Arrival Time (ms)')
    ax.set_ylabel('Density')
    ax.set_title('Inter-Arrival Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Client->Router delay
    ax = axes[2, 0]
    if delays_ms is not None:
        ax.plot(delays_ms, marker='o', markersize=2, linewidth=1)
        ax.axhline(y=np.mean(delays_ms), color='red', linestyle='--',
                   alpha=0.7, label=f'Mean ({np.mean(delays_ms):.1f}ms)')
        ax.axhline(y=np.median(delays_ms), color='green', linestyle='--',
                   alpha=0.7, label=f'Median ({np.median(delays_ms):.1f}ms)')
        ax.set_xlabel('Request Index')
        ax.set_ylabel('Client->Router Delay (ms)')
        ax.set_title('Network/Processing Delay (Client to Router)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No delay data', ha='center', va='center', transform=ax.transAxes)

    # 6. Router queue distribution
    ax = axes[2, 1]
    queue_counts = router_df['queue_idx'].value_counts().sort_index()
    ax.bar(queue_counts.index, queue_counts.values, alpha=0.7)
    ax.set_xlabel('Queue Index')
    ax.set_ylabel('Number of Requests')
    ax.set_title('Router Queue Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    # 7. Rolling statistics - Client
    ax = axes[3, 0]
    ax.plot(actual_burst['rolling_mean'], label='Client Rolling Mean', linewidth=2)
    ax.plot(router_burst['rolling_mean'], label='Router Rolling Mean', linewidth=2, alpha=0.7)
    ax.axhline(y=actual_burst['mean'], color='blue', linestyle='--',
               alpha=0.5, label=f"Client Mean ({actual_burst['mean']:.1f}ms)")
    ax.axhline(y=router_burst['mean'], color='green', linestyle='--',
               alpha=0.5, label=f"Router Mean ({router_burst['mean']:.1f}ms)")
    ax.set_xlabel('Request Pair Index')
    ax.set_ylabel('Inter-Arrival Time (ms)')
    ax.set_title('Rolling Mean Comparison (window=10)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. Cumulative arrival
    ax = axes[3, 1]
    if len(target_submit_aligned) > 0:
        ax.plot(target_submit_aligned['target_arrival_ms'] / 1000, target_submit_aligned.index,
                label='Target', alpha=0.7, linewidth=2)
        ax.plot(target_submit_aligned['actual_arrival_ms'] / 1000, target_submit_aligned.index,
                label='Submit', alpha=0.7, linewidth=2)
    if len(router_series) > 0:
        ax.plot(router_series['router_arrival_ms'] / 1000, router_series.index,
                label='Router', alpha=0.7, linewidth=2)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Cumulative Requests')
    ax.set_title('Cumulative Request Arrivals')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '/sgl-workspace/sglang/slo/trace_comparison_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to: {output_path}")
    print()

    # Save detailed statistics
    stats_output = '/sgl-workspace/sglang/slo/trace_comparison_stats.txt'
    with open(stats_output, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRACE COMPARISON AND BURSTINESS ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Actual trace: {ACTUAL_TRACE}\n")
        f.write(f"Target trace: {TARGET_TRACE}\n")
        f.write(f"Analysis time: {datetime.now()}\n\n")

        f.write("RATE INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Actual duration: {actual_duration_s:.2f} seconds\n")
        f.write(f"Actual rate: {actual_rate:.3f} requests/second\n")
        f.write(f"Target baseline rate: {target_baseline_rate:.3f} requests/second\n")
        f.write(f"Rate scaling factor: {rate_factor:.3f}\n\n")
        if post_duration_s is not None:
            f.write(f"Post arrival duration: {post_duration_s:.2f} seconds\n")
        if post_latency_summary is not None:
            f.write("Submit -> Post latency statistics (ms):\n")
            f.write(f"  Mean   : {post_latency_summary['mean']:.2f}\n")
            f.write(f"  Median : {post_latency_summary['median']:.2f}\n")
            f.write(f"  Std    : {post_latency_summary['std']:.2f}\n")
            f.write(f"  Min    : {post_latency_summary['min']:.2f}\n")
            f.write(f"  Max    : {post_latency_summary['max']:.2f}\n\n")

        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total requests (client): {len(actual_df)}\n")
        f.write(f"Total requests (router): {len(router_df)}\n")
        f.write(f"Total requests (target): {len(target_df)}\n")
        f.write(f"Matched requests (target-submit): {len(target_submit_aligned)}\n")
        f.write(f"Matched requests (submit-router): {len(submit_router_aligned)}\n")
        f.write(f"Matched requests (target-router): {len(target_router_aligned)}\n\n")

        f.write("CLIENT vs ROUTER TIMING\n")
        f.write("-" * 80 + "\n")
        if delays_ms is not None:
            f.write(f"Number of matched requests: {len(delays_ms)}\n")
            f.write(f"Mean client->router delay: {np.mean(delays_ms):.2f} ms\n")
            f.write(f"Median client->router delay: {np.median(delays_ms):.2f} ms\n")
            f.write(f"Std client->router delay: {np.std(delays_ms):.2f} ms\n")
            f.write(f"Min delay: {np.min(delays_ms):.2f} ms\n")
            f.write(f"Max delay: {np.max(delays_ms):.2f} ms\n\n")
        else:
            f.write("No matching data available\n\n")

        if target_submit_stats:
            f.write("ARRIVAL TIME DIFFERENCES (Submit vs Target)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean: {target_submit_stats['mean']:.2f} ms\n")
            f.write(f"Median: {target_submit_stats['median']:.2f} ms\n")
            f.write(f"Std: {target_submit_stats['std']:.2f} ms\n")
            f.write(f"Min: {target_submit_stats['min']:.2f} ms\n")
            f.write(f"Max: {target_submit_stats['max']:.2f} ms\n\n")
        if post_target_stats:
            f.write("ARRIVAL TIME DIFFERENCES (Post vs Target)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean: {post_target_stats['mean']:.2f} ms\n")
            f.write(f"Median: {post_target_stats['median']:.2f} ms\n")
            f.write(f"Std: {post_target_stats['std']:.2f} ms\n")
            f.write(f"Min: {post_target_stats['min']:.2f} ms\n")
            f.write(f"Max: {post_target_stats['max']:.2f} ms\n\n")
        if submit_router_stats:
            f.write("ARRIVAL TIME DIFFERENCES (Router vs Submit)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean: {submit_router_stats['mean']:.2f} ms\n")
            f.write(f"Median: {submit_router_stats['median']:.2f} ms\n")
            f.write(f"Std: {submit_router_stats['std']:.2f} ms\n")
            f.write(f"Min: {submit_router_stats['min']:.2f} ms\n")
            f.write(f"Max: {submit_router_stats['max']:.2f} ms\n\n")
        if target_router_stats:
            f.write("ARRIVAL TIME DIFFERENCES (Router vs Target)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean: {target_router_stats['mean']:.2f} ms\n")
            f.write(f"Median: {target_router_stats['median']:.2f} ms\n")
            f.write(f"Std: {target_router_stats['std']:.2f} ms\n")
            f.write(f"Min: {target_router_stats['min']:.2f} ms\n")
            f.write(f"Max: {target_router_stats['max']:.2f} ms\n\n")

        f.write("CLIENT TRACE BURSTINESS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean IAT: {actual_burst['mean']:.2f} ms\n")
        f.write(f"Std IAT: {actual_burst['std']:.2f} ms\n")
        f.write(f"CV: {actual_burst['cv']:.3f}\n")
        f.write(f"Burstiness Index: {actual_burst['burstiness_index']:.3f}\n\n")

        f.write("ROUTER TRACE BURSTINESS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean IAT: {router_burst['mean']:.2f} ms\n")
        f.write(f"Std IAT: {router_burst['std']:.2f} ms\n")
        f.write(f"CV: {router_burst['cv']:.3f}\n")
        f.write(f"Burstiness Index: {router_burst['burstiness_index']:.3f}\n\n")

        f.write("TARGET TRACE BURSTINESS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean IAT: {target_burst['mean']:.2f} ms\n")
        f.write(f"Std IAT: {target_burst['std']:.2f} ms\n")
        f.write(f"CV: {target_burst['cv']:.3f}\n")
        f.write(f"Burstiness Index: {target_burst['burstiness_index']:.3f}\n\n")

        f.write("ROUTER QUEUE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        queue_counts = router_df['queue_idx'].value_counts().sort_index()
        for queue_idx, count in queue_counts.items():
            f.write(f"Queue {queue_idx}: {count} requests ({count/len(router_df)*100:.1f}%)\n")
        f.write("\n")

        f.write("CLIENT BURST PERIODS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Threshold: {threshold:.2f} ms\n")
        f.write(f"Number of burst periods: {len(burst_periods)}\n\n")

        if burst_periods:
            f.write("Detailed burst periods:\n")
            for i, period in enumerate(sorted(burst_periods, key=len, reverse=True), 1):
                start_idx = period[0]
                end_idx = period[-1]
                duration = len(period)
                avg_iat = np.mean(actual_iat[period])
                f.write(f"{i}. Requests {start_idx}-{end_idx+1}: {duration} bursts, "
                       f"avg IAT: {avg_iat:.2f} ms\n")
        f.write("\n")

        f.write("ROUTER BURST PERIODS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Threshold: {router_threshold:.2f} ms\n")
        f.write(f"Number of burst periods: {len(router_burst_periods)}\n\n")

        if router_burst_periods:
            f.write("Detailed burst periods:\n")
            for i, period in enumerate(sorted(router_burst_periods, key=len, reverse=True), 1):
                start_idx = period[0]
                end_idx = period[-1]
                duration = len(period)
                avg_iat = np.mean(router_iat[period])
                f.write(f"{i}. Requests {start_idx}-{end_idx+1}: {duration} bursts, "
                       f"avg IAT: {avg_iat:.2f} ms\n")

    print(f"Detailed statistics saved to: {stats_output}")
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
