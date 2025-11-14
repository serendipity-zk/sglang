#!/usr/bin/env python3
"""Analyze interval patterns in issue_stream logs to diagnose 0.0 intervals."""

import json
import sys
from collections import Counter

def analyze_log(log_path):
    """Analyze a single log file for interval patterns."""

    zero_interval_requests = []
    interval_stats = {
        'zero_count': 0,
        'tiny_count': 0,  # < 1ms
        'small_count': 0,  # 1-10ms
        'normal_count': 0,  # 10-50ms
        'large_count': 0,  # > 50ms
    }

    with open(log_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get('status') != 'SUCCESS':
                continue

            intervals = record.get('intervals', [])
            if not intervals:
                continue

            # Count interval types
            has_zeros = False
            for interval in intervals:
                if interval == 0.0:
                    interval_stats['zero_count'] += 1
                    has_zeros = True
                elif interval < 1.0:
                    interval_stats['tiny_count'] += 1
                elif interval < 10.0:
                    interval_stats['small_count'] += 1
                elif interval < 50.0:
                    interval_stats['normal_count'] += 1
                else:
                    interval_stats['large_count'] += 1

            if has_zeros:
                zero_interval_requests.append({
                    'request_id': record.get('request_id'),
                    'real_output_len': record.get('real_output_len'),
                    'chunk_count': record.get('chunk_count'),
                    'avg_interval_ms': record.get('avg_interval_ms'),
                    'intervals': intervals[:20],  # First 20
                })

    # Print statistics
    print(f"\n=== Interval Statistics ===")
    print(f"Zero (0.0ms):     {interval_stats['zero_count']:6d}")
    print(f"Tiny (<1ms):      {interval_stats['tiny_count']:6d}")
    print(f"Small (1-10ms):   {interval_stats['small_count']:6d}")
    print(f"Normal (10-50ms): {interval_stats['normal_count']:6d}")
    print(f"Large (>50ms):    {interval_stats['large_count']:6d}")

    total = sum(interval_stats.values())
    if total > 0:
        zero_pct = interval_stats['zero_count'] / total * 100
        print(f"\nZero intervals: {zero_pct:.1f}% of all intervals")

    # Show examples with zeros
    print(f"\n=== Requests with Zero Intervals ({len(zero_interval_requests)} total) ===")
    for i, req in enumerate(zero_interval_requests[:5], 1):
        print(f"\n{i}. {req['request_id']}")
        print(f"   Output tokens: {req['real_output_len']}, Chunks: {req['chunk_count']}")
        print(f"   Avg interval: {req['avg_interval_ms']:.2f}ms")
        print(f"   Intervals: {req['intervals']}")

        # Calculate tokens per chunk
        if req['chunk_count'] > 0:
            tokens_per_chunk = req['real_output_len'] / req['chunk_count']
            print(f"   Avg tokens/chunk: {tokens_per_chunk:.1f}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_intervals.py <log_file.jsonl>")
        sys.exit(1)

    analyze_log(sys.argv[1])
