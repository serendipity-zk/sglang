#!/usr/bin/env python3
"""
Log Parser for SGLang Worker Metrics

This module parses STAT_METRICS logs from SGLang workers and extracts
relevant features for cycle time prediction.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime


@dataclass
class MetricsRecord:
    """Represents a single metrics record from the log."""

    timestamp: float
    iteration_num: int
    worker_id: str

    # Features for prediction
    batch_size_tokens: int
    prefill_chunk_pairs: List[List[int]]  # List of [current_chunk, cumulative_prefill]
    kv_tokens_used: int

    # Target variable
    iteration_time_ms: float

    # Additional context
    running_batch_size: int
    queue_reqs: int
    kv_usage_pct: float
    forward_mode: str
    prefill_tokens: int
    decode_tokens: int

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'iteration_num': self.iteration_num,
            'worker_id': self.worker_id,
            'batch_size_tokens': self.batch_size_tokens,
            'prefill_chunk_pairs': self.prefill_chunk_pairs,
            'kv_tokens_used': self.kv_tokens_used,
            'iteration_time_ms': self.iteration_time_ms,
            'running_batch_size': self.running_batch_size,
            'queue_reqs': self.queue_reqs,
            'kv_usage_pct': self.kv_usage_pct,
            'forward_mode': self.forward_mode,
            'prefill_tokens': self.prefill_tokens,
            'decode_tokens': self.decode_tokens,
        }


class LogParser:
    """Parser for SGLang worker metrics logs."""

    # Regex pattern to match STAT_METRICS lines
    STAT_METRICS_PATTERN = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] STAT_METRICS: (.+)$'
    )

    def __init__(self):
        self.records: List[MetricsRecord] = []

    def parse_file(self, filepath: str) -> List[MetricsRecord]:
        """
        Parse a log file and extract all STAT_METRICS records.

        Args:
            filepath: Path to the log file

        Returns:
            List of MetricsRecord objects
        """
        self.records = []

        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                match = self.STAT_METRICS_PATTERN.search(line)
                if match:
                    timestamp_str, json_str = match.groups()
                    try:
                        record = self._parse_metrics_json(json_str)
                        if record:
                            self.records.append(record)
                    except Exception as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue

        print(f"Parsed {len(self.records)} metrics records from {filepath}")
        return self.records

    def _parse_metrics_json(self, json_str: str) -> Optional[MetricsRecord]:
        """
        Parse the JSON metrics payload.

        Args:
            json_str: JSON string from the log line

        Returns:
            MetricsRecord object or None if parsing fails
        """
        try:
            data = json.loads(json_str)

            # Extract required fields
            record = MetricsRecord(
                timestamp=data.get('timestamp', 0.0),
                iteration_num=data.get('iteration_num', 0),
                worker_id=data.get('worker_id', 'unknown'),
                batch_size_tokens=data.get('batch_size_tokens', 0),
                prefill_chunk_pairs=data.get('prefill_chunk_pairs', []),
                kv_tokens_used=data.get('kv_tokens_used', 0),
                iteration_time_ms=data.get('iteration_time_ms', 0.0),
                running_batch_size=data.get('running_batch_size', 0),
                queue_reqs=data.get('queue_reqs', 0),
                kv_usage_pct=data.get('kv_usage_pct', 0.0),
                forward_mode=data.get('forward_mode', 'UNKNOWN'),
                prefill_tokens=data.get('prefill_tokens', 0),
                decode_tokens=data.get('decode_tokens', 0),
            )

            return record
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
        except KeyError as e:
            print(f"Missing required field: {e}")
            return None

    def split_train_test(
        self,
        train_ratio: float = 0.9
    ) -> Tuple[List[MetricsRecord], List[MetricsRecord]]:
        """
        Split records into training and test sets.

        Args:
            train_ratio: Ratio of training data (default 0.9 for 90/10 split)

        Returns:
            Tuple of (train_records, test_records)
        """
        if not self.records:
            return [], []

        # Sort by timestamp to maintain temporal order
        sorted_records = sorted(self.records, key=lambda x: x.timestamp)

        split_idx = int(len(sorted_records) * train_ratio)
        train_records = sorted_records[:split_idx]
        test_records = sorted_records[split_idx:]

        print(f"Split: {len(train_records)} training, {len(test_records)} test records")
        return train_records, test_records

    def get_statistics(self) -> dict:
        """
        Get statistics about the parsed records.

        Returns:
            Dictionary with statistics
        """
        if not self.records:
            return {}

        iteration_times = [r.iteration_time_ms for r in self.records]
        batch_sizes = [r.batch_size_tokens for r in self.records]
        kv_usage = [r.kv_tokens_used for r in self.records]

        # Count prefill vs decode
        prefill_count = sum(1 for r in self.records if r.forward_mode == 'EXTEND')
        decode_count = sum(1 for r in self.records if r.forward_mode == 'DECODE')
        mixed_count = sum(1 for r in self.records if r.forward_mode == 'MIXED')

        # Count chunked prefill instances
        chunked_count = sum(1 for r in self.records if r.prefill_chunk_pairs)

        return {
            'total_records': len(self.records),
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
            'forward_modes': {
                'EXTEND': prefill_count,
                'DECODE': decode_count,
                'MIXED': mixed_count,
            },
            'chunked_prefill_count': chunked_count,
        }


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python log_parser.py <log_file>")
        sys.exit(1)

    parser = LogParser()
    records = parser.parse_file(sys.argv[1])

    print("\nStatistics:")
    stats = parser.get_statistics()
    print(json.dumps(stats, indent=2))

    train, test = parser.split_train_test()
    print(f"\nTrain set: {len(train)} records")
    print(f"Test set: {len(test)} records")
