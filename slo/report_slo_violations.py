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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from rich.console import Console
from rich.table import Table


# Regex to parse [ARRIVAL_DEBUG] lines from server .ans logs
# Example: [ARRIVAL_DEBUG] rid=req_000012_0, recv_arrival=1764830892991.893, current_time=1764830893047.649, gap=55.756ms
ARRIVAL_DEBUG_RE = re.compile(
    r"\[ARRIVAL_DEBUG\]\s+rid=([^,]+),\s*recv_arrival=([^,]+),\s*current_time=([^,]+),\s*gap=([0-9.]+)ms"
)


@dataclass
class LogEntry:
    request_id: str
    submit_timestamp: Optional[float]
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
    detokenize_timestamps: List[Optional[float]]
    server_id: Optional[str]
    start_iteration: Optional[int]
    iteration_ids: List[Optional[int]]


@dataclass
class ArrivalDebugEntry:
    """Parsed [ARRIVAL_DEBUG] timing from server logs."""
    request_id: str
    recv_arrival_ms: Optional[float]  # Client submit time (may be None)
    scheduler_arrival_ms: float       # When scheduler received request
    gap_ms: float                     # Difference (queue time)


def find_server_log(log_dir: Path, server_id: str) -> Optional[Path]:
    """Find server log file matching server_id port.

    server_id: '0.0.0.0:31003' → looks for 'worker_*_p31003.ans'
    """
    if not log_dir or not server_id:
        return None
    port = server_id.split(":")[-1]
    matches = list(log_dir.glob(f"worker_*_p{port}.ans"))
    return matches[0] if matches else None


def load_arrival_debug_from_file(path: Path) -> Dict[str, ArrivalDebugEntry]:
    """Parse [ARRIVAL_DEBUG] lines from a single server .ans log."""
    entries: Dict[str, ArrivalDebugEntry] = {}
    with path.open() as f:
        for line in f:
            m = ARRIVAL_DEBUG_RE.search(line)
            if not m:
                continue
            rid = m.group(1)
            recv_arrival_str = m.group(2)
            current_time_str = m.group(3)
            gap_str = m.group(4)

            recv_arrival = None if recv_arrival_str == "None" else float(recv_arrival_str)
            scheduler_arrival = float(current_time_str)
            gap = float(gap_str)

            entries[rid] = ArrivalDebugEntry(
                request_id=rid,
                recv_arrival_ms=recv_arrival,
                scheduler_arrival_ms=scheduler_arrival,
                gap_ms=gap,
            )
    return entries


def load_arrival_debug(
    log_dir: Optional[Path], server_ids: Iterable[str]
) -> Dict[str, ArrivalDebugEntry]:
    """Load [ARRIVAL_DEBUG] entries for all unique server_ids.

    Auto-discovers server log files based on port in server_id.
    """
    if not log_dir:
        return {}

    all_entries: Dict[str, ArrivalDebugEntry] = {}
    loaded_files: set = set()

    for server_id in server_ids:
        if not server_id:
            continue
        log_path = find_server_log(log_dir, server_id)
        if not log_path or log_path in loaded_files:
            continue
        loaded_files.add(log_path)
        entries = load_arrival_debug_from_file(log_path)
        all_entries.update(entries)

    return all_entries


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
                submit_timestamp=rec.get("submit_timestamp"),
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
                detokenize_timestamps=rec.get("detokenize_timestamps", []) or [],
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


def count_violated_tokens(
    log_entry: LogEntry, timeline: Optional[TimelineEntry]
) -> Optional[int]:
    ttft = log_entry.target_ttft_ms
    tpot = log_entry.target_tpot_ms
    if ttft is None or tpot is None or not timeline:
        return log_entry.slo_violations
    count = 0
    for idx, elapsed in enumerate(timeline.elapsed_ms):
        deadline = ttft + idx * tpot
        if elapsed - deadline > 0:
            count += 1
    return count if count > 0 else log_entry.slo_violations


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


def auto_discover_logs(log_dir: Path) -> Tuple[Optional[Path], Optional[Path], Path]:
    """Auto-discover log files in the given directory.

    Returns: (client_log_path, timeline_path, server_log_dir)
    """
    # Find client JSONL log
    client_log = None
    for pattern in ["rust_client_output.jsonl", "*_output.jsonl", "*.jsonl"]:
        matches = list(log_dir.glob(pattern))
        if matches:
            client_log = matches[0]
            break

    # Find timeline pickle
    timeline = None
    for pattern in ["rust_elapsed_timelines.pkl", "*_timelines.pkl", "*.pkl"]:
        matches = list(log_dir.glob(pattern))
        if matches:
            timeline = matches[0]
            break

    return client_log, timeline, log_dir


def main():
    parser = argparse.ArgumentParser(
        description="Report SLO-violating requests with server/iteration context."
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        nargs="?",
        default=Path("logs"),
        help="Directory containing all logs (auto-discovers client log, timeline, server logs).",
    )
    args = parser.parse_args()

    # Auto-discover log files
    client_log, timeline_path, server_log_dir = auto_discover_logs(args.log_dir)

    if not client_log:
        print(f"Error: No client log (.jsonl) found in {args.log_dir}")
        return

    print(f"Client log: {client_log}")
    print(f"Timeline:   {timeline_path or 'not found'}")
    print(f"Server dir: {server_log_dir}")
    print()

    logs = load_logs(client_log)
    timelines = load_timelines(timeline_path) if timeline_path else {}

    # Collect unique server_ids for auto-discovery of server logs
    server_ids = set()
    for entry in logs.values():
        if entry.server_id:
            server_ids.add(entry.server_id)
    for entry in timelines.values():
        if entry.server_id:
            server_ids.add(entry.server_id)

    # Load arrival debug timing from server logs
    arrival_debug = load_arrival_debug(server_log_dir, server_ids)

    table = Table(title="SLO Viol.", show_lines=False)
    table.add_column("request_id", overflow="fold")
    table.add_column("token_idx", justify="right")
    table.add_column("delay_ms", justify="right")
    table.add_column("queue", justify="right")  # Time in router/tokenizer
    table.add_column("proc", justify="right")   # Time to first detokenize
    table.add_column("net", justify="right")    # Network latency
    table.add_column("server_id", overflow="fold")
    table.add_column("v_iter", justify="right")
    table.add_column("s_iter", justify="right")
    table.add_column("ttft", justify="right")
    table.add_column("tpot", justify="right")
    table.add_column("n_vio", justify="right")

    missing_iter: List[str] = []
    rows = []
    total_violated_tokens = 0
    for log_entry, timeline, token_idx, delay_ms in iter_failures(logs, timelines):
        server_id = (timeline.server_id if timeline else None) or log_entry.server_id
        violation_iter = (
            resolve_iter_id(timeline, token_idx, log_entry.iteration_id)
            if token_idx is not None
            else log_entry.iteration_id
        )

        # Compute timing breakdown: queue, proc, net
        queue_ms = None
        proc_ms = None
        net_ms = None
        submit_ms = log_entry.submit_timestamp * 1000 if log_entry.submit_timestamp else None

        # Try to find arrival_debug entry (server may append _0 suffix)
        rid = log_entry.request_id
        arr_entry = arrival_debug.get(rid) or arrival_debug.get(f"{rid}_0")

        if arr_entry and submit_ms:
            # queue = scheduler_arrival - submit
            queue_ms = arr_entry.scheduler_arrival_ms - submit_ms

        # Get first token detokenize timestamp for proc/net calculation
        first_detok_ts = None
        first_recv_ms = None
        if timeline and timeline.elapsed_ms:
            first_recv_ms = timeline.elapsed_ms[0]
            if timeline.detokenize_timestamps:
                first_detok_ts = timeline.detokenize_timestamps[0]
                if first_detok_ts is not None:
                    # Normalize to ms since epoch
                    first_detok_ts = first_detok_ts if first_detok_ts > 1e11 else first_detok_ts * 1000

        if arr_entry and first_detok_ts:
            # proc = first_detokenize - scheduler_arrival
            proc_ms = first_detok_ts - arr_entry.scheduler_arrival_ms

        if first_detok_ts and first_recv_ms and submit_ms:
            # net = first_recv (relative to submit) - (first_detok - submit)
            first_detok_rel = first_detok_ts - submit_ms
            net_ms = first_recv_ms - first_detok_rel

        violated_tokens = count_violated_tokens(log_entry, timeline)
        if violated_tokens is not None:
            total_violated_tokens += violated_tokens
        if violation_iter is None:
            missing_iter.append(log_entry.request_id)
        rows.append(
            (
                log_entry.request_id,
                token_idx,
                delay_ms,
                queue_ms,
                proc_ms,
                net_ms,
                server_id,
                violation_iter,
                log_entry.start_iteration,
                log_entry.target_ttft_ms,
                log_entry.target_tpot_ms,
                violated_tokens,
            )
        )

    rows.sort(
        key=lambda row: (
            row[6] or "",  # server_id
            row[7] if row[7] is not None else float("inf"),  # violation_iter
            row[2] if row[2] is not None else float("inf"),  # delay_ms
        )
    )

    for row in rows:
        table.add_row(
            row[0],   # request_id
            str(row[1] if row[1] is not None else ""),  # token_idx
            f"{row[2]:.2f}" if row[2] is not None else "",  # delay_ms
            f"{row[3]:.2f}" if row[3] is not None else "",  # queue_ms
            f"{row[4]:.2f}" if row[4] is not None else "",  # proc_ms
            f"{row[5]:.2f}" if row[5] is not None else "",  # net_ms
            row[6] or "",  # server_id
            str(row[7]) if row[7] is not None else "",  # violation_iter
            str(row[8] or ""),  # start_iteration
            f"{row[9]:.2f}" if row[9] is not None else "",  # ttft
            f"{row[10]:.2f}" if row[10] is not None else "",  # tpot
            str(row[11]) if row[11] is not None else "",  # n_vio
        )

    # Print to console
    console = Console()
    console.print(table)
    console.print(f"Total violated tokens: {total_violated_tokens}")
    if missing_iter:
        console.print(
            f"[yellow]Warning:[/yellow] missing violation iteration_id for "
            f"{len(missing_iter)} request(s): {', '.join(missing_iter)}"
        )

    # Save to file in log directory
    output_path = args.log_dir / "slo_violations_report.txt"
    with open(output_path, "w") as f:
        file_console = Console(file=f, force_terminal=True, width=200)
        file_console.print(table)
        file_console.print(f"Total violated tokens: {total_violated_tokens}")
        if missing_iter:
            file_console.print(
                f"Warning: missing violation iteration_id for "
                f"{len(missing_iter)} request(s): {', '.join(missing_iter)}"
            )
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
