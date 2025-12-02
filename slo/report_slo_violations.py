#!/usr/bin/env python3
"""
Scan SLO client logs for violations and map them to server/iteration metadata.

Inputs:
- JSONL log file produced by the Rust SLO client (`--log`).
- Pickle stream of per-token timelines (`--timeline`), emitted when the client
  is run with `--elapsed-dump-path`.

Output: tabular summary to stdout of each request that missed SLO, including
the first violating token index, its delay, and the server/iteration that
produced it.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from rich.console import Console
from rich.table import Table


@dataclass
class LogEntry:
    request_id: str
    target_ttft_ms: Optional[float]
    target_tpot_ms: Optional[float]
    server_id: Optional[str]
    iteration_id: Optional[int]
    start_iteration: Optional[int]
    slo_satisfied: Optional[bool]
    slo_violations: Optional[int]
    slo_tokens_checked: Optional[int]


@dataclass
class TimelineEntry:
    request_id: str
    elapsed_ms: List[float]
    server_id: Optional[str]
    start_iteration: Optional[int]
    iteration_ids: List[Optional[int]]


def load_logs(path: Path) -> Dict[str, LogEntry]:
    failing: Dict[str, LogEntry] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            slo_satisfied = rec.get("slo_satisfied")
            slo_violations = rec.get("slo_violations")
            if slo_satisfied is True:
                continue
            if slo_satisfied is None and not slo_violations:
                continue  # no SLO data
            entry = LogEntry(
                request_id=rec["request_id"],
                target_ttft_ms=rec.get("target_ttft_ms"),
                target_tpot_ms=rec.get("target_tpot_ms"),
                server_id=rec.get("server_id"),
                iteration_id=rec.get("iteration_id"),
                start_iteration=rec.get("start_iteration"),
                slo_satisfied=slo_satisfied,
                slo_violations=slo_violations,
                slo_tokens_checked=rec.get("slo_tokens_checked"),
            )
            failing[entry.request_id] = entry
    return failing


def load_timelines(path: Path) -> Dict[str, TimelineEntry]:
    timelines: Dict[str, TimelineEntry] = {}
    with path.open("rb") as f:
        while True:
            try:
                rec = pickle.load(f)
            except EOFError:
                break
            if not isinstance(rec, dict):
                continue
            rid = rec.get("request_id")
            if not rid:
                continue
            timelines[rid] = TimelineEntry(
                request_id=rid,
                elapsed_ms=rec.get("elapsed_ms", []) or [],
                server_id=rec.get("server_id"),
                start_iteration=rec.get("start_iteration"),
                iteration_ids=rec.get("iteration_ids", []) or [],
            )
    return timelines


def find_first_violation(
    log_entry: LogEntry, timeline: TimelineEntry
) -> Optional[Tuple[int, float]]:
    ttft = log_entry.target_ttft_ms
    tpot = log_entry.target_tpot_ms
    if ttft is None or tpot is None:
        return None
    for idx, elapsed in enumerate(timeline.elapsed_ms):
        deadline = ttft + idx * tpot
        delay = elapsed - deadline
        if delay > 0:
            return idx, delay
    return None


def resolve_iter_id(
    timeline: TimelineEntry, token_idx: int, fallback_iter: Optional[int]
) -> Optional[int]:
    if token_idx < len(timeline.iteration_ids):
        iter_id = timeline.iteration_ids[token_idx]
        if iter_id is not None:
            return iter_id
    return fallback_iter


def iter_failures(
    logs: Dict[str, LogEntry], timelines: Dict[str, TimelineEntry]
) -> Iterable[Tuple[LogEntry, Optional[TimelineEntry], Optional[int], Optional[float]]]:
    for rid, log_entry in logs.items():
        timeline = timelines.get(rid)
        token_idx = delay_ms = None
        if timeline:
            violation = find_first_violation(log_entry, timeline)
            if violation:
                token_idx, delay_ms = violation
        yield log_entry, timeline, token_idx, delay_ms


def main():
    parser = argparse.ArgumentParser(
        description="Report SLO-violating requests with server/iteration context."
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to JSONL log produced by rust client (output.jsonl).",
    )
    parser.add_argument(
        "--timeline",
        type=Path,
        required=False,
        help="Path to pickle stream from --elapsed-dump-path.",
    )
    args = parser.parse_args()

    logs = load_logs(args.log)
    timelines = load_timelines(args.timeline) if args.timeline else {}

    table = Table(title="SLO Violations", show_lines=False)
    table.add_column("request_id", overflow="fold")
    table.add_column("token_idx", justify="right")
    table.add_column("delay_ms", justify="right")
    table.add_column("server_id", overflow="fold")
    table.add_column("violation_iter", justify="right")
    table.add_column("start_iteration", justify="right")
    table.add_column("target_ttft_ms", justify="right")
    table.add_column("target_tpot_ms", justify="right")

    missing_iter: List[str] = []

    for log_entry, timeline, token_idx, delay_ms in iter_failures(logs, timelines):
        server_id = (timeline.server_id if timeline else None) or log_entry.server_id
        violation_iter = (
            resolve_iter_id(timeline, token_idx, log_entry.iteration_id)
            if token_idx is not None
            else log_entry.iteration_id
        )
        if violation_iter is None:
            missing_iter.append(log_entry.request_id)
        table.add_row(
            log_entry.request_id,
            str(token_idx if token_idx is not None else ""),
            f"{delay_ms:.2f}" if delay_ms is not None else "",
            server_id or "",
            str(violation_iter) if violation_iter is not None else "",
            str(log_entry.start_iteration or ""),
            f"{log_entry.target_ttft_ms:.2f}"
            if log_entry.target_ttft_ms is not None
            else "",
            f"{log_entry.target_tpot_ms:.2f}"
            if log_entry.target_tpot_ms is not None
            else "",
        )

    console = Console()
    console.print(table)
    if missing_iter:
        console.print(
            f"[yellow]Warning:[/yellow] missing violation iteration_id for "
            f"{len(missing_iter)} request(s): {', '.join(missing_iter)}"
        )


if __name__ == "__main__":
    main()
