#!/usr/bin/env python3
"""
Concatenate all JSONL metrics files from profile runs into consolidated outputs.

This script walks through all subdirectories in profile_runs_full and:
1. Collects all metrics.jsonl files
2. Adds metadata (chunk_size, run_name) to each metric
3. Writes consolidated outputs
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def extract_metadata_from_path(jsonl_path: Path) -> Dict[str, str]:
    """
    Extract metadata from the file path.

    Example path:
    profile_runs_full/chunk512/chunk_prefill_sweep_chunk512_chunk512_20251023-065043/in1024_out1_metrics.jsonl

    Returns:
        dict with chunk_size, run_name, file_type, prompt_tokens, completion_tokens
    """
    parts = jsonl_path.parts
    metadata = {}

    # Extract chunk size from directory name (e.g., "chunk512" -> 512)
    for part in parts:
        if part.startswith("chunk") and part != "chunk_prefill_sweep":
            match = re.match(r"chunk(\d+)", part)
            if match:
                metadata["chunk_size"] = int(match.group(1))
                break

    # Extract run name (the timestamp directory)
    if len(parts) >= 2:
        metadata["run_name"] = parts[-2]

    # Extract file type and request pattern
    filename = jsonl_path.stem  # e.g., "in1024_out1_metrics" or "metrics"
    metadata["filename"] = filename

    if filename == "metrics":
        metadata["file_type"] = "aggregate"
    else:
        # Try to extract in/out pattern (e.g., "in1024_out1_metrics")
        match = re.match(r"in(\d+)_out(\d+)_metrics", filename)
        if match:
            metadata["file_type"] = "per_config"
            metadata["prompt_tokens"] = int(match.group(1))
            metadata["completion_tokens"] = int(match.group(2))
        else:
            metadata["file_type"] = "unknown"

    return metadata


def concat_jsonl_files(
    root_dir: Path,
    output_file: Path,
    file_pattern: str = "*.jsonl",
    add_metadata: bool = True,
) -> int:
    """
    Concatenate all JSONL files matching the pattern.

    Args:
        root_dir: Root directory to search
        output_file: Output file path
        file_pattern: Glob pattern for files to match
        add_metadata: Whether to add path metadata to each line

    Returns:
        Number of lines written
    """
    jsonl_files = sorted(root_dir.rglob(file_pattern))
    total_lines = 0

    print(f"Found {len(jsonl_files)} files matching '{file_pattern}'")

    with output_file.open("w", encoding="utf-8") as out_f:
        for jsonl_path in jsonl_files:
            metadata = extract_metadata_from_path(jsonl_path)
            print(f"Processing: {jsonl_path.relative_to(root_dir)} (chunk_size={metadata.get('chunk_size', 'N/A')})")

            try:
                with jsonl_path.open("r", encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue

                        if add_metadata:
                            # Parse, add metadata, re-serialize
                            try:
                                data = json.loads(line)
                                data.update(metadata)
                                out_f.write(json.dumps(data) + "\n")
                            except json.JSONDecodeError as e:
                                print(f"  WARNING: Skipping invalid JSON line: {e}")
                                continue
                        else:
                            # Just copy the line as-is
                            out_f.write(line + "\n")

                        total_lines += 1
            except Exception as e:
                print(f"  ERROR reading {jsonl_path}: {e}")

    return total_lines


def concat_by_chunk_size(root_dir: Path, output_dir: Path) -> None:
    """Create separate concatenated files for each chunk size."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group files by chunk size
    chunk_files: Dict[int, List[Path]] = {}

    for jsonl_path in root_dir.rglob("*_metrics.jsonl"):
        metadata = extract_metadata_from_path(jsonl_path)
        chunk_size = metadata.get("chunk_size")
        if chunk_size:
            if chunk_size not in chunk_files:
                chunk_files[chunk_size] = []
            chunk_files[chunk_size].append(jsonl_path)

    # Concatenate each chunk size
    for chunk_size in sorted(chunk_files.keys()):
        output_file = output_dir / f"chunk{chunk_size}_all_metrics.jsonl"
        print(f"\nCreating {output_file.name} from {len(chunk_files[chunk_size])} files...")

        total_lines = 0
        with output_file.open("w", encoding="utf-8") as out_f:
            for jsonl_path in sorted(chunk_files[chunk_size]):
                metadata = extract_metadata_from_path(jsonl_path)

                with jsonl_path.open("r", encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                            data.update(metadata)
                            out_f.write(json.dumps(data) + "\n")
                            total_lines += 1
                        except json.JSONDecodeError:
                            continue

        print(f"  Wrote {total_lines} lines to {output_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate JSONL metrics files from profile runs"
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("/sgl-workspace/sglang/profile/profile_runs_full"),
        help="Root directory containing profile runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/sgl-workspace/sglang/profile/all_metrics.jsonl"),
        help="Output file for concatenated metrics",
    )
    parser.add_argument(
        "--by-chunk",
        action="store_true",
        help="Also create separate files per chunk size",
    )
    parser.add_argument(
        "--chunk-output-dir",
        type=Path,
        default=Path("/sgl-workspace/sglang/profile/metrics_by_chunk"),
        help="Output directory for per-chunk-size files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_metrics.jsonl",
        help="File pattern to match (default: *_metrics.jsonl)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't add metadata to each line",
    )

    args = parser.parse_args()

    if not args.root_dir.exists():
        print(f"ERROR: Root directory does not exist: {args.root_dir}")
        return 1

    print(f"Scanning directory: {args.root_dir}")
    print(f"Output file: {args.output}")
    print(f"Add metadata: {not args.no_metadata}")
    print()

    # Create main concatenated file
    total_lines = concat_jsonl_files(
        root_dir=args.root_dir,
        output_file=args.output,
        file_pattern=args.pattern,
        add_metadata=not args.no_metadata,
    )

    print(f"\n✓ Wrote {total_lines} total lines to {args.output}")
    print(f"  File size: {args.output.stat().st_size / 1024 / 1024:.2f} MB")

    # Optionally create per-chunk-size files
    if args.by_chunk:
        print("\nCreating per-chunk-size files...")
        concat_by_chunk_size(args.root_dir, args.chunk_output_dir)
        print(f"\n✓ Per-chunk files written to {args.chunk_output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
