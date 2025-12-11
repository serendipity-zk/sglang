#!/usr/bin/env python3
"""Extract timing information (gpu, kvf, psim) from PolyServe worker logs."""

import re
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_PATH = SCRIPT_DIR / "results"
OUTPUT_CSV = RESULTS_PATH / "polyserve_timing.csv"

# Regex patterns
TIMESTAMP_PATTERN = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]'
TIMING_PATTERN = r'gpu=([0-9.]+).*kvf=([0-9.]+).*psim=([0-9.]+)'


def extract_timing():
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'experiment', 'worker', 'gpu', 'kvf', 'psim'])

        # Find all PolyServe worker .ans files
        ans_files = list(RESULTS_PATH.glob('*/PolyServe/worker_log/worker_*.ans'))
        print(f"Found {len(ans_files)} worker log files")

        for i, ans_file in enumerate(ans_files):
            experiment = ans_file.parents[2].name  # e.g., "uniform_4096_1024"
            worker = ans_file.stem  # e.g., "worker_1_gpu0_p31001"
            print(f"[{i+1}/{len(ans_files)}] Processing {experiment}/{worker}...")

            line_count = 0
            with open(ans_file, 'r', errors='ignore') as f:
                for line in f:
                    if '[TIME]:' not in line:
                        continue

                    # Extract timestamp
                    ts_match = re.search(TIMESTAMP_PATTERN, line)
                    timestamp = ts_match.group(1) if ts_match else ''

                    # Extract timing values
                    timing_match = re.search(TIMING_PATTERN, line)
                    if timing_match:
                        gpu, kvf, psim = timing_match.groups()
                        writer.writerow([timestamp, experiment, worker, gpu, kvf, psim])
                        line_count += 1

            print(f"    Extracted {line_count} timing entries")


if __name__ == '__main__':
    extract_timing()
    print(f"\nOutput written to {OUTPUT_CSV}")
