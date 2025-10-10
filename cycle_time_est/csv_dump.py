#!/usr/bin/env python3
"""
Simple CSV Dump for Cycle Time Estimation Data

This script parses a SGLang worker log file and dumps core cycle time metrics to CSV format.
Keeps only: iteration_time_ms, batch_size_tokens, kv_tokens_used, prefill_chunk_pairs.
"""

import argparse
import csv
import json
import sys
from typing import List

from log_parser import LogParser, MetricsRecord


def flatten_prefill_pairs(prefill_pairs: List[List[int]]) -> str:
    """
    Convert prefill_chunk_pairs to a string representation for CSV.

    Args:
        prefill_pairs: List of [current_chunk, cumulative_prefill] pairs

    Returns:
        JSON string representation of the pairs
    """
    return json.dumps(prefill_pairs)


def dump_to_csv(records: List[MetricsRecord], output_file: str):
    """
    Dump records to CSV file with only core cycle time fields.

    Args:
        records: List of MetricsRecord objects
        output_file: Path to output CSV file
    """
    if not records:
        print("No records to dump")
        return

    print(f"Dumping {len(records)} records to {output_file}")

    with open(output_file, 'w', newline='') as csvfile:
        # Define CSV headers - only core cycle time fields
        fieldnames = [
            'batch_size_tokens',
            'prefill_chunk_pairs',
            'kv_tokens_used',
            'iteration_time_ms'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            # Create row with only core fields
            row = {
                'batch_size_tokens': record.batch_size_tokens,
                'prefill_chunk_pairs': flatten_prefill_pairs(record.prefill_chunk_pairs),
                'kv_tokens_used': record.kv_tokens_used,
                'iteration_time_ms': record.iteration_time_ms
            }
            writer.writerow(row)

    print(f"✓ Successfully dumped {len(records)} records to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dump core SGLang cycle time metrics to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dump log file to CSV (core fields only)
  python csv_dump.py worker.log -o output.csv

  # Use default output name (input_file.csv)
  python csv_dump.py worker.log

  # Show statistics without dumping
  python csv_dump.py worker.log --stats-only

Output fields: batch_size_tokens, prefill_chunk_pairs, kv_tokens_used, iteration_time_ms
        """
    )

    parser.add_argument('log_file', type=str, help='Path to the SGLang worker log file')
    parser.add_argument('-o', '--output', type=str,
                       help='Output CSV file path (default: <log_file>.csv)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show statistics, do not dump CSV')

    args = parser.parse_args()

    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        # Replace .log extension with .csv, or just append .csv
        if args.log_file.endswith('.log'):
            output_file = args.log_file[:-4] + '.csv'
        else:
            output_file = args.log_file + '.csv'

    try:
        # Parse the log file
        print(f"Parsing log file: {args.log_file}")
        parser = LogParser()
        records = parser.parse_file(args.log_file)

        if not records:
            print("Error: No records found in log file")
            return 1

        print(f"✓ Parsed {len(records)} records")

        # Show statistics for core cycle time metrics only
        print("\nCore Cycle Time Metrics Statistics:")
        records = parser.records
        if records:
            iteration_times = [r.iteration_time_ms for r in records]
            batch_sizes = [r.batch_size_tokens for r in records]
            kv_usage = [r.kv_tokens_used for r in records]
            chunked_count = sum(1 for r in records if r.prefill_chunk_pairs)

            stats = {
                'total_records': len(records),
                'iteration_time_ms': {
                    'min': min(iteration_times),
                    'max': max(iteration_times),
                    'avg': sum(iteration_times) / len(iteration_times),
                },
                'batch_size_tokens': {
                    'min': min(batch_sizes),
                    'max': max(batch_sizes),
                    'avg': sum(batch_sizes) / len(batch_sizes),
                },
                'kv_tokens_used': {
                    'min': min(kv_usage),
                    'max': max(kv_usage),
                    'avg': sum(kv_usage) / len(kv_usage),
                },
                'chunked_prefill_operations': chunked_count,
            }
            print(json.dumps(stats, indent=2))

        # Dump to CSV if not stats-only
        if not args.stats_only:
            dump_to_csv(records, output_file)
        else:
            print(f"\nSkipping CSV dump (--stats-only mode)")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
