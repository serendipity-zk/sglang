#!/usr/bin/env python3
"""Analyze SLO satisfaction results from issue_stream logs."""

import json
import sys
from collections import Counter

def analyze_slo_log(log_path):
    """Analyze SLO satisfaction from a log file."""

    total_requests = 0
    slo_satisfied_count = 0
    slo_violated_count = 0
    no_slo_count = 0

    total_violations = 0
    total_tokens_checked = 0

    violation_distribution = Counter()  # Number of requests by violation count

    # Alternative SLO with 100ms slack
    slack_satisfied_count = 0
    slack_violated_count = 0
    no_slack_count = 0
    total_slack_violations = 0
    total_slack_tokens_checked = 0
    slack_violation_distribution = Counter()

    with open(log_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get('status') != 'SUCCESS':
                continue

            total_requests += 1

            # Check if SLO data is present
            if 'slo_satisfied' not in record:
                no_slo_count += 1
            else:
                slo_satisfied = record['slo_satisfied']
                violations = record.get('slo_violations', 0)
                tokens_checked = record.get('slo_tokens_checked', 0)

                total_violations += violations
                total_tokens_checked += tokens_checked
                violation_distribution[violations] += 1

                if slo_satisfied:
                    slo_satisfied_count += 1
                else:
                    slo_violated_count += 1

            # Check alternative SLO with 100ms slack
            if 'tpot_with_100_slack' not in record:
                no_slack_count += 1
            else:
                slack_satisfied = record['tpot_with_100_slack']
                slack_violations = record.get('tpot_slack_violations', 0)
                slack_tokens_checked = record.get('tpot_slack_tokens_checked', 0)

                total_slack_violations += slack_violations
                total_slack_tokens_checked += slack_tokens_checked
                slack_violation_distribution[slack_violations] += 1

                if slack_satisfied:
                    slack_satisfied_count += 1
                else:
                    slack_violated_count += 1

    # Print statistics
    print(f"\n=== SLO Analysis Results ===")
    print(f"Total successful requests: {total_requests}")

    # Standard SLO (start_time + ttft + i*tpot)
    if no_slo_count < total_requests:
        print(f"\n--- Standard SLO (start_time + ttft + i*tpot) ---")
        print(f"Requests with SLO data: {total_requests - no_slo_count}")
        print(f"  ✓ SLO satisfied:      {slo_satisfied_count} ({slo_satisfied_count/(total_requests-no_slo_count)*100:.1f}%)")
        print(f"  ✗ SLO violated:       {slo_violated_count} ({slo_violated_count/(total_requests-no_slo_count)*100:.1f}%)")

        if total_tokens_checked > 0:
            violation_rate = total_violations / total_tokens_checked * 100
            print(f"\nToken-level statistics:")
            print(f"  Total tokens checked: {total_tokens_checked}")
            print(f"  Total violations:     {total_violations}")
            print(f"  Violation rate:       {violation_rate:.2f}%")

        print(f"\nViolation Distribution:")
        print(f"{'Violations':<12} {'Requests':<10} {'Percentage':<12}")
        print("-" * 35)
        for violations in sorted(violation_distribution.keys())[:10]:  # Show top 10
            count = violation_distribution[violations]
            pct = count / (total_requests - no_slo_count) * 100
            print(f"{violations:<12} {count:<10} {pct:>6.1f}%")

    # Alternative SLO with 100ms slack (first_token_time + 100 + (i-1)*tpot)
    if no_slack_count < total_requests:
        print(f"\n--- Alternative SLO with 100ms slack (first_token_time + 100ms + (i-1)*tpot) ---")
        print(f"Requests with slack SLO data: {total_requests - no_slack_count}")
        print(f"  ✓ SLO satisfied:             {slack_satisfied_count} ({slack_satisfied_count/(total_requests-no_slack_count)*100:.1f}%)")
        print(f"  ✗ SLO violated:              {slack_violated_count} ({slack_violated_count/(total_requests-no_slack_count)*100:.1f}%)")

        if total_slack_tokens_checked > 0:
            slack_violation_rate = total_slack_violations / total_slack_tokens_checked * 100
            print(f"\nToken-level statistics (excluding first token):")
            print(f"  Total tokens checked: {total_slack_tokens_checked}")
            print(f"  Total violations:     {total_slack_violations}")
            print(f"  Violation rate:       {slack_violation_rate:.2f}%")

        print(f"\nViolation Distribution:")
        print(f"{'Violations':<12} {'Requests':<10} {'Percentage':<12}")
        print("-" * 35)
        for violations in sorted(slack_violation_distribution.keys())[:10]:  # Show top 10
            count = slack_violation_distribution[violations]
            pct = count / (total_requests - no_slack_count) * 100
            print(f"{violations:<12} {count:<10} {pct:>6.1f}%")

    if no_slo_count == total_requests and no_slack_count == total_requests:
        print(f"\n⚠️  No SLO data found in log file.")
        print(f"   Make sure the trace has 'ttft' and 'tpot' columns.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_slo.py <log_file.jsonl>")
        sys.exit(1)

    analyze_slo_log(sys.argv[1])
