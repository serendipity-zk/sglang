#!/usr/bin/env python3
"""Check consistency between engine and sidecar stats in shadow_stats.jsonl.

Usage:
    python check_shadow_stats.py [path/to/shadow_stats.jsonl]

Compares all fields, reports per-field match rates, and flags mismatches.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_records(path: str):
    """Load JSONL records, skipping malformed lines."""
    records = []
    errors = 0
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                errors += 1
    return records, errors


def compare_field(engine_val, sidecar_val, field: str):
    """Compare two values, returning (match: bool, detail: str)."""
    if engine_val is None and sidecar_val is None:
        return True, "both_null"
    if engine_val is None or sidecar_val is None:
        return False, f"one_null(e={engine_val}, s={sidecar_val})"

    # Numeric comparison with tolerance
    if isinstance(engine_val, (int, float)) and isinstance(sidecar_val, (int, float)):
        if engine_val == sidecar_val:
            return True, "exact"
        # Float tolerance for timing fields
        if isinstance(engine_val, float) or isinstance(sidecar_val, float):
            if abs(engine_val) < 1e-9 and abs(sidecar_val) < 1e-9:
                return True, "both_zero"
            if abs(engine_val) > 1e-9:
                pct = abs(engine_val - sidecar_val) / abs(engine_val) * 100
                if pct < 0.1:
                    return True, f"~equal({pct:.4f}%)"
                return False, f"diff={engine_val - sidecar_val:.4f} ({pct:.2f}%)"
        return False, f"e={engine_val} s={sidecar_val}"

    # String comparison
    if isinstance(engine_val, str) and isinstance(sidecar_val, str):
        if engine_val == sidecar_val:
            return True, "exact"
        return False, f"e='{engine_val}' s='{sidecar_val}'"

    # Dict comparison (e.g., waiting_queue_info, batch_size_by_tpot_tier)
    if isinstance(engine_val, dict) and isinstance(sidecar_val, dict):
        if engine_val == sidecar_val:
            return True, "exact"
        # Compare key by key for useful detail
        all_keys = set(engine_val.keys()) | set(sidecar_val.keys())
        diffs = []
        for k in sorted(all_keys):
            ev = engine_val.get(k)
            sv = sidecar_val.get(k)
            if ev != sv:
                diffs.append(f"{k}:e={ev}/s={sv}")
        if not diffs:
            return True, "equal"
        return False, "; ".join(diffs[:5])

    # List comparison (e.g., prefill_chunk_pairs)
    if isinstance(engine_val, list) and isinstance(sidecar_val, list):
        if engine_val == sidecar_val:
            return True, "exact"
        return False, f"len(e={len(engine_val)},s={len(sidecar_val)})"

    # Fallback
    if engine_val == sidecar_val:
        return True, "exact"
    return False, f"type_mismatch(e={type(engine_val).__name__},s={type(sidecar_val).__name__})"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/shadow_stats.jsonl"
    if not Path(path).exists():
        print(f"File not found: {path}")
        sys.exit(1)

    records, parse_errors = load_records(path)
    print(f"Loaded {len(records)} records ({parse_errors} parse errors)\n")

    # Categorize records
    both = [r for r in records if r.get("engine") and r.get("sidecar")]
    engine_null = [r for r in records if not r.get("engine")]
    sidecar_null = [r for r in records if not r.get("sidecar")]

    print(f"Both present:  {len(both)}")
    print(f"Engine null:   {len(engine_null)}")
    print(f"Sidecar null:  {len(sidecar_null)}")

    if not both:
        print("No records with both engine and sidecar stats.")
        return

    # Worker distribution
    workers = defaultdict(int)
    for r in records:
        workers[r.get("worker_id", "?")] += 1
    print(f"\nWorkers ({len(workers)}):")
    for w in sorted(workers):
        print(f"  {w}: {workers[w]} records")

    # Per-field comparison
    all_fields = set()
    for r in both:
        all_fields |= set(r["engine"].keys())
        all_fields |= set(r["sidecar"].keys())

    # Skip worker_id (always different by design — engine URL vs sidecar ID)
    skip_fields = {"worker_id"}
    compare_fields = sorted(all_fields - skip_fields)

    print(f"\n{'Field':<30} {'Match':>7} {'Mismatch':>8} {'Rate':>7}  First mismatch")
    print("-" * 95)

    field_mismatches = {}
    for field in compare_fields:
        match_count = 0
        mismatch_count = 0
        first_mismatch = None

        for r in both:
            ev = r["engine"].get(field)
            sv = r["sidecar"].get(field)
            matched, detail = compare_field(ev, sv, field)
            if matched:
                match_count += 1
            else:
                mismatch_count += 1
                if first_mismatch is None:
                    first_mismatch = detail

        total = match_count + mismatch_count
        rate = match_count / total * 100 if total > 0 else 0
        marker = " OK" if mismatch_count == 0 else " **"
        first_str = first_mismatch or ""
        if len(first_str) > 40:
            first_str = first_str[:37] + "..."
        print(f"  {field:<28} {match_count:>7} {mismatch_count:>8} {rate:>6.1f}%{marker} {first_str}")

        if mismatch_count > 0:
            field_mismatches[field] = mismatch_count

    # Iteration alignment check
    print(f"\n{'='*60}")
    print("Iteration alignment:")
    iter_diffs = defaultdict(int)
    for r in both:
        e_iter = r["engine"].get("iteration_num", 0)
        s_iter = r["sidecar"].get("iteration_num", 0)
        diff = s_iter - e_iter
        iter_diffs[diff] += 1
    for diff in sorted(iter_diffs):
        pct = iter_diffs[diff] / len(both) * 100
        label = "MATCH" if diff == 0 else f"sidecar ahead by {diff}" if diff > 0 else f"engine ahead by {-diff}"
        print(f"  offset={diff:+d}: {iter_diffs[diff]:>7} ({pct:.1f}%) — {label}")

    # Summary
    print(f"\n{'='*60}")
    if field_mismatches:
        print(f"MISMATCHES in {len(field_mismatches)} fields:")
        for f, c in sorted(field_mismatches.items(), key=lambda x: -x[1]):
            print(f"  {f}: {c} mismatches")
    else:
        print("ALL FIELDS MATCH")


if __name__ == "__main__":
    main()
