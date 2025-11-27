"""
High-performance trace-driven streaming request issuer using aiohttp and multiprocessing.

Improvements over issue_stream.py:
- Uses aiohttp with readany() for minimal buffering latency
- Multiprocessing to utilize all available CPU cores (bypasses GIL)
- Manual SSE line parsing for fine-grained timing control
- Configurable worker processes and concurrency per worker

Expected improvements:
- Inter-token intervals closer to sender-side timing (4-5ms vs 30-80ms)
- Much higher throughput with 256 cores x concurrent requests per core
- Same JSONL output format for compatibility

Usage example:

python -m slo.issue_stream_aiohttp \
  --trace /sgl-workspace/sglang/slo/trace/uniform_4096_1024.csv \
  --text-file /path/to/large_corpus.txt \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://0.0.0.0:40000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --rate 1.0 \
  --num-workers 256 \
  --concurrency-per-worker 50
"""

import argparse
import asyncio
import csv
import fcntl
import json
import multiprocessing as mp
import os
import random
import signal
import sys
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
import tqdm

import aiohttp

from sglang.srt.hf_transformers_utils import get_tokenizer


def format_slo_tier_label(tpot_ms: Optional[float]) -> Optional[str]:
    """Human-readable label for grouping requests by TPOT target."""
    if tpot_ms is None:
        return None
    try:
        value = float(tpot_ms)
    except (TypeError, ValueError):
        return None
    if abs(value - round(value)) < 1e-6:
        value_str = f"{int(round(value))}"
    else:
        value_str = f"{value:.1f}"
    return f"{value_str} ms"


def build_token_pool(
    text_file: str,
    tokenizer_path: str,
    num_tokens: int = 100_000,
    seed: int = 1234,
    chunk_bytes: int = 1 << 20,
):
    """Stream-tokenize from the beginning and take the first `num_tokens` tokens."""
    tokenizer = get_tokenizer(tokenizer_path, tokenizer_mode="auto", trust_remote_code=True)
    token_pool: List[int] = []
    with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
        while len(token_pool) < num_tokens:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            chunk_tokens = tokenizer.encode(chunk)
            need = num_tokens - len(token_pool)
            if need <= 0:
                break
            if len(chunk_tokens) >= need:
                token_pool.extend(chunk_tokens[:need])
                break
            else:
                token_pool.extend(chunk_tokens)
    return token_pool, tokenizer


def sample_prompt_from_pool(token_pool: List[int], length: int, rng: random.Random) -> List[int]:
    if length <= 0:
        return []
    if length > len(token_pool):
        length = len(token_pool)
    max_start = max(0, len(token_pool) - length)
    start = rng.randint(0, max_start)
    return token_pool[start : start + length]


class SharedStats:
    """Multiprocessing-safe statistics tracker."""

    def __init__(self, expected_total: int, manager: mp.Manager):
        self.expected_total = expected_total
        self._manager = manager
        self._lock = manager.Lock()
        self._dict = manager.dict()
        self._dict["submitted"] = 0
        self._dict["completed"] = 0
        self._dict["failed"] = 0
        self._dict["active"] = 0
        self._slo_tier_stats = manager.dict()  # tier_label -> {"attained": count, "total": count}

    def record_submit(self):
        with self._lock:
            self._dict["submitted"] = self._dict["submitted"] + 1
            self._dict["active"] = self._dict["active"] + 1

    def record_complete(self, success: bool):
        with self._lock:
            self._dict["active"] = self._dict["active"] - 1
            if success:
                self._dict["completed"] = self._dict["completed"] + 1
            else:
                self._dict["failed"] = self._dict["failed"] + 1

    def record_slo_result(self, tier_label: Optional[str], satisfied: bool):
        if not tier_label:
            return
        with self._lock:
            # Get or create tier stats
            if tier_label not in self._slo_tier_stats:
                self._slo_tier_stats[tier_label] = self._manager.dict({"attained": 0, "total": 0})
            tier_stats = self._slo_tier_stats[tier_label]
            tier_stats["total"] = tier_stats.get("total", 0) + 1
            if satisfied:
                tier_stats["attained"] = tier_stats.get("attained", 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            slo_tiers = {tier: dict(stats) for tier, stats in self._slo_tier_stats.items()}
            return {
                "submitted": self._dict["submitted"],
                "completed": self._dict["completed"],
                "failed": self._dict["failed"],
                "active": self._dict["active"],
                "expected_total": self.expected_total,
                "slo_tiers": slo_tiers,
            }

    def all_done(self) -> bool:
        snap = self.snapshot()
        return (snap["completed"] + snap["failed"]) >= snap["expected_total"]


class CliStatusDisplay:
    """Minimal terminal dashboard."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def render(self, lines: List[str], final: bool = False) -> None:
        output = "\n".join(lines)
        if self.enabled:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(output)
            if final:
                sys.stdout.write("\n")
            sys.stdout.flush()
        elif final:
            sys.stdout.write(output + "\n")
            sys.stdout.flush()


async def async_send_streaming_request(
    session: aiohttp.ClientSession,
    endpoint_url: str,
    prompt_ids: List[int],
    decode_tokens: int,
    temperature: float,
    timeout: float,
    request_id: str,
    log_path: str,
    stats: SharedStats,
    ttft: Optional[float] = None,
    tpot: Optional[float] = None,
    enqueue_timestamp: Optional[float] = None,
):
    """
    Async streaming request with low-latency readany() approach.

    Uses manual line parsing with readany() to minimize buffering delays.
    Preserves exact timing methodology from original issue_stream.py.
    """

    # Use enqueue timestamp from main process if available, otherwise record now
    submit_timestamp = enqueue_timestamp if enqueue_timestamp is not None else time.time()
    start_time = time.perf_counter()

    sampling_params = {
        "max_new_tokens": max(0, int(decode_tokens)),
        "temperature": float(temperature),
        "ignore_eos": True,
    }
    data = {
        "input_ids": [prompt_ids],
        "sampling_params": sampling_params,
        "stream": True,
    }
    if ttft is not None:
        data["target_ttft_ms"] = ttft
    if tpot is not None:
        data["target_tpot_ms"] = tpot

    success = False
    first_token_time = None
    all_token_times = []  # Store ALL token arrival times
    chunk_count = 0  # Number of streaming chunks received
    output_token_count = 0  # Actual number of output tokens
    prev_token_count = 0  # Previous token count to detect new tokens
    output_text = ""  # Accumulated output text
    error_msg = None
    record = None  # Initialize record to avoid UnboundLocalError in finally block

    try:
        timeout_obj = aiohttp.ClientTimeout(total=timeout if timeout > 0 else None)

        async with session.post(endpoint_url, json=data, timeout=timeout_obj) as resp:
            resp.raise_for_status()

            # Low-latency streaming with readany() and manual line parsing
            buffer = b""

            while True:
                # Read whatever is available immediately (no waiting for lines)
                chunk = await resp.content.readany()

                if not chunk:
                    break

                # Record timestamp ONCE per readany() call, before parsing lines
                # This prevents multiple SSE events in the same buffer from getting identical timestamps
                chunk_arrival_time = time.perf_counter()

                buffer += chunk

                # Parse complete lines from buffer
                while b"\n" in buffer:
                    line_bytes, buffer = buffer.split(b"\n", 1)

                    if not line_bytes:
                        continue

                    try:
                        line = line_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        continue

                    # Skip SSE comments
                    if line.startswith(':'):
                        continue

                    # Parse SSE data
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix

                        # Check for done signal
                        if data_str.strip() == '[DONE]':
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            chunk_count += 1

                            # Extract actual token count from response
                            if 'meta_info' in chunk_data and 'completion_tokens' in chunk_data['meta_info']:
                                current_token_count = chunk_data['meta_info']['completion_tokens']

                                # Record time when we get NEW tokens
                                # Use chunk_arrival_time (recorded once per readany) not time.perf_counter()
                                if current_token_count > prev_token_count:
                                    token_time = chunk_arrival_time
                                    num_new_tokens = current_token_count - prev_token_count

                                    if len(all_token_times) == 0:
                                        # First token(s) - interpolate from start_time to token_time
                                        # This prevents all first tokens from having the same timestamp
                                        interval = token_time - start_time
                                        for i in range(1, num_new_tokens + 1):
                                            interpolated_time = start_time + (interval * i / num_new_tokens)
                                            all_token_times.append(interpolated_time)
                                            if first_token_time is None:
                                                first_token_time = interpolated_time
                                    else:
                                        # Uniformly distribute interval between last token and now
                                        last_time = all_token_times[-1]
                                        interval = token_time - last_time

                                        # Assign uniformly spaced times
                                        for i in range(1, num_new_tokens + 1):
                                            interpolated_time = last_time + (interval * i / num_new_tokens)
                                            all_token_times.append(interpolated_time)

                                    prev_token_count = current_token_count

                                output_token_count = current_token_count

                            # Get output text from server (cumulative)
                            if 'text' in chunk_data:
                                output_text = chunk_data['text']

                        except json.JSONDecodeError:
                            continue

        end_time = time.perf_counter()
        total_duration_ms = (end_time - start_time) * 1000

        # Calculate timing metrics
        ttft_ms = None
        if first_token_time is not None:
            ttft_ms = (first_token_time - start_time) * 1000

        # Calculate inter-token intervals over ALL tokens
        all_intervals = []
        if len(all_token_times) >= 2:
            for i in range(1, len(all_token_times)):
                interval_ms = (all_token_times[i] - all_token_times[i-1]) * 1000
                all_intervals.append(interval_ms)

        # Average over all tokens
        avg_interval = sum(all_intervals) / len(all_intervals) if all_intervals else None

        # Keep first 20 intervals for printing/logging
        first_20_intervals = [round(x, 2) for x in all_intervals[:20]]

        # SLO check: verify each token i arrives before start_time + ttft + i * tpot
        slo_satisfied = None
        slo_violations = 0
        slo_tokens_checked = 0
        if ttft is not None and tpot is not None and len(all_token_times) > 0:
            slo_tokens_checked = len(all_token_times)
            for i, token_time in enumerate(all_token_times):
                # Token i should arrive by: start_time + ttft + i * tpot (in seconds)
                deadline = start_time + (ttft / 1000.0) + (i * tpot / 1000.0)
                if token_time > deadline:
                    slo_violations += 1
            slo_satisfied = (slo_violations == 0)

        # Alternative SLO check with 100ms slack: ignore first token, then check
        # each token i (i >= 1) arrives before first_token_time + 100ms + (i-1) * tpot
        tpot_with_100_slack = None
        tpot_slack_violations = 0
        tpot_slack_tokens_checked = 0
        if tpot is not None and len(all_token_times) > 1 and first_token_time is not None:
            # Check tokens starting from index 1 (second token)
            tpot_slack_tokens_checked = len(all_token_times) - 1
            for i in range(1, len(all_token_times)):
                # Token i should arrive by: first_token_time + 100ms + (i-1) * tpot (in seconds)
                deadline = first_token_time + 0.1 + ((i - 1) * tpot / 1000.0)
                if all_token_times[i] > deadline:
                    tpot_slack_violations += 1
            tpot_with_100_slack = (tpot_slack_violations == 0)

        # Build log record with logical field ordering
        record = {
            "request_id": request_id,
            "input_len": len(prompt_ids),
            "trace_output_len": decode_tokens,  # Target decode length from trace
            "real_output_len": output_token_count,  # Actual output tokens generated
            "status": "SUCCESS",
            "submit_timestamp": submit_timestamp,
            "total_duration_ms": round(total_duration_ms, 2),
            "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
            "avg_interval_ms": round(avg_interval, 2) if avg_interval is not None else None,
            "chunk_count": chunk_count,  # Number of streaming chunks
        }
        if ttft is not None:
            record["target_ttft_ms"] = ttft
        slo_tier_label = format_slo_tier_label(tpot)
        if tpot is not None:
            record["target_tpot_ms"] = tpot
        # Add SLO check results
        if slo_satisfied is not None:
            record["slo_satisfied"] = slo_satisfied
            record["slo_violations"] = slo_violations
            record["slo_tokens_checked"] = slo_tokens_checked
            stats.record_slo_result(slo_tier_label, slo_satisfied)
        # Add alternative SLO check results
        if tpot_with_100_slack is not None:
            record["tpot_with_100_slack"] = tpot_with_100_slack
            record["tpot_slack_violations"] = tpot_slack_violations
            record["tpot_slack_tokens_checked"] = tpot_slack_tokens_checked
        # Append detailed timing and text at the end
        record["intervals"] = first_20_intervals
        record["output_text"] = output_text[:100]  # Only log first 100 chars

        success = True

    except Exception as e:
        end_time = time.perf_counter()
        total_duration_ms = (end_time - start_time) * 1000
        error_msg = str(e)

        record = {
            "request_id": request_id,
            "input_len": len(prompt_ids),
            "trace_output_len": decode_tokens,  # Target decode length from trace
            "real_output_len": output_token_count,  # Actual output tokens before failure
            "status": "FAILED",
            "error": error_msg,
            "exception_type": type(e).__name__,
            "submit_timestamp": submit_timestamp,
            "total_duration_ms": round(total_duration_ms, 2),
            "chunk_count": chunk_count,  # Number of streaming chunks before failure
        }
        if ttft is not None:
            record["target_ttft_ms"] = ttft
        if tpot is not None:
            record["target_tpot_ms"] = tpot
        # Append text at the end
        record["output_text"] = output_text[:100]  # Only log first 100 chars (partial output before failure)

    finally:
        # Log the result with file locking for concurrent writes (only if record was created)
        if record is not None:
            with open(log_path, "a", encoding="utf-8") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    lf.write(json.dumps(record, ensure_ascii=False) + "\n")
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

            # Update stats
            stats.record_complete(success)


async def worker_event_loop(
    worker_id: int,
    request_queue: mp.Queue,
    token_pool: List[int],
    endpoint_url: str,
    temperature: float,
    timeout: float,
    log_path: str,
    stats: SharedStats,
    concurrency: int,
    seed: int,
    ready_queue: mp.Queue,
):
    """
    Worker process event loop.

    Pulls requests from queue and processes them with bounded concurrency.
    Uses single aiohttp ClientSession for connection reuse.
    """

    rng = random.Random(seed + worker_id + 1000)

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)

    # Create single ClientSession for this worker (connection reuse)
    # Enable TCP_NODELAY to disable Nagle's algorithm and reduce batching
    connector = aiohttp.TCPConnector(
        limit=concurrency,
        limit_per_host=concurrency,
        force_close=False,
        enable_cleanup_closed=True,
    )

    # Configure TCP socket options for low latency
    import socket
    tcp_connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)

    # Custom connector that sets TCP_NODELAY
    class LowLatencyTCPConnector(aiohttp.TCPConnector):
        async def _create_connection(self, req, traces, timeout):
            conn = await super()._create_connection(req, traces, timeout)
            # Set TCP_NODELAY on the socket to disable Nagle's algorithm
            if conn.transport:
                sock = conn.transport.get_extra_info('socket')
                if sock:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return conn

    connector = LowLatencyTCPConnector(limit=concurrency, limit_per_host=concurrency)
    timeout_obj = aiohttp.ClientTimeout(total=None)  # Per-request timeout set individually

    async with aiohttp.ClientSession(connector=connector, timeout=timeout_obj) as session:
        # Signal that this worker is ready
        ready_queue.put(worker_id)

        async def process_request(req_data: Dict[str, Any]):
            """Process a single request with semaphore control."""
            async with semaphore:
                # Extract request parameters
                idx = req_data["idx"]
                prefill = req_data["prefill"]
                decode = req_data["decode"]
                ttft = req_data.get("ttft")
                tpot = req_data.get("tpot")
                enqueue_timestamp = req_data.get("enqueue_timestamp")  # Get enqueue time from main process

                # Sample prompt
                prompt_ids = sample_prompt_from_pool(token_pool, prefill, rng)
                request_id = f"req_{idx:06d}_{int(time.time() * 1000) % 1000000:06d}"

                # Record submission
                stats.record_submit()

                # Send request
                await async_send_streaming_request(
                    session=session,
                    endpoint_url=endpoint_url,
                    prompt_ids=prompt_ids,
                    decode_tokens=decode,
                    temperature=temperature,
                    timeout=timeout,
                    request_id=request_id,
                    log_path=log_path,
                    stats=stats,
                    ttft=ttft,
                    tpot=tpot,
                    enqueue_timestamp=enqueue_timestamp,
                )

        # Process requests from queue
        tasks = []

        while True:
            try:
                # Non-blocking get with timeout
                try:
                    req_data = request_queue.get(timeout=0.001)
                except Exception:
                    # Queue is empty or other error - check if there are pending tasks
                    if tasks:
                        # Wait for some tasks to complete
                        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        tasks = list(pending)
                    continue

                # Sentinel value to stop worker
                if req_data is None:
                    break

                # Create task for this request
                task = asyncio.create_task(process_request(req_data))
                tasks.append(task)

                # Prevent task list from growing unbounded
                if len(tasks) >= concurrency * 2:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = list(pending)

            except KeyboardInterrupt:
                break

        # Wait for all remaining tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def worker_process_main(
    worker_id: int,
    request_queue: mp.Queue,
    token_pool: List[int],
    endpoint_url: str,
    temperature: float,
    timeout: float,
    log_path: str,
    stats: SharedStats,
    concurrency: int,
    seed: int,
    ready_queue: mp.Queue,
):
    """Entry point for worker process - sets up event loop."""
    try:
        asyncio.run(worker_event_loop(
            worker_id=worker_id,
            request_queue=request_queue,
            token_pool=token_pool,
            endpoint_url=endpoint_url,
            temperature=temperature,
            timeout=timeout,
            log_path=log_path,
            stats=stats,
            concurrency=concurrency,
            seed=seed,
            ready_queue=ready_queue,
        ))
    except KeyboardInterrupt:
        pass


def status_display_thread(
    stats: SharedStats,
    start_time: float,
    stop_event: mp.Event,
    log_path: str,
):
    """Background thread to render the terminal dashboard."""

    ui = CliStatusDisplay(enabled=sys.stdout.isatty())
    log_display_path = os.path.relpath(log_path)

    def tier_sort_key(label: str) -> float:
        try:
            return float(label.split()[0])
        except (ValueError, IndexError):
            return float("inf")

    try:
        while not stop_event.is_set():
            for _ in range(5):
                if stop_event.is_set():
                    break
                time.sleep(0.1)

            current_time = time.perf_counter()
            elapsed_s = current_time - start_time

            # Get stats
            snapshot = stats.snapshot()
            submitted = snapshot["submitted"]
            completed = snapshot["completed"]
            failed = snapshot["failed"]
            active = snapshot["active"]
            expected_total = snapshot["expected_total"]
            slo_tiers = snapshot.get("slo_tiers") or {}

            # Calculate rates
            submit_speed = submitted / elapsed_s if elapsed_s > 0 else 0.0
            complete_speed = completed / elapsed_s if elapsed_s > 0 else 0.0

            percent_complete = (
                (completed + failed) / expected_total * 100.0 if expected_total > 0 else 0.0
            )

            lines = [
                "SLO Streaming Issue Runner (aiohttp + multiprocessing) - Live Stats",
                "=====================================================================",
                f"Total Requests       : {expected_total}",
                f"Submitted            : {submitted}",
                f"Active               : {active}",
                f"Completed / Failed   : {completed} / {failed}",
                f"Submit Rate (req/s)  : {submit_speed:.2f}",
                f"Complete Rate (req/s): {complete_speed:.2f}",
                f"Elapsed (s)          : {elapsed_s:.1f}",
                f"Progress             : {percent_complete:.1f}%",
                "",
                f"Log File             : {log_display_path}",
                "Ctrl+C to stop",
            ]

            if slo_tiers:
                lines.append("")
                lines.append("SLO Attainment (per TPOT tier)")
                for tier_label in sorted(slo_tiers.keys(), key=tier_sort_key):
                    tier_counts = slo_tiers[tier_label]
                    total = tier_counts.get("total", 0)
                    attained = tier_counts.get("attained", 0)
                    percent = (attained / total * 100.0) if total > 0 else 0.0
                    lines.append(f"  {tier_label:>8} : {attained}/{total} ({percent:.1f}%)")

            ui.render(lines)

            if stats.all_done():
                break

    except Exception:
        pass
    finally:
        try:
            # Final render
            elapsed_s = max(0.0, time.perf_counter() - start_time)
            snapshot = stats.snapshot()
            submitted = snapshot["submitted"]
            completed = snapshot["completed"]
            failed = snapshot["failed"]
            active = snapshot["active"]
            expected_total = snapshot["expected_total"]
            slo_tiers = snapshot.get("slo_tiers") or {}
            percent_complete = (
                (completed + failed) / expected_total * 100.0 if expected_total > 0 else 0.0
            )

            lines = [
                "SLO Streaming Issue Runner (aiohttp + multiprocessing) - Final Stats",
                "====================================================================",
                f"Total Requests       : {expected_total}",
                f"Submitted            : {submitted}",
                f"Active               : {active}",
                f"Completed / Failed   : {completed} / {failed}",
                f"Elapsed (s)          : {elapsed_s:.1f}",
                f"Progress             : {percent_complete:.1f}%",
                "",
                f"Log File             : {log_display_path}",
            ]
            if slo_tiers:
                lines.append("")
                lines.append("SLO Attainment (per TPOT tier)")
                for tier_label in sorted(slo_tiers.keys(), key=tier_sort_key):
                    tier_counts = slo_tiers[tier_label]
                    total = tier_counts.get("total", 0)
                    attained = tier_counts.get("attained", 0)
                    percent = (attained / total * 100.0) if total > 0 else 0.0
                    lines.append(f"  {tier_label:>8} : {attained}/{total} ({percent:.1f}%)")
            ui.render(lines, final=True)
        except Exception:
            pass


def run_trace(
    trace_csv: str,
    text_file: str,
    tokenizer_path: str,
    base_url: str,
    model: str,
    rate: float,
    temperature: float,
    seed: int,
    max_requests: int,
    timeout: float,
    num_workers: int,
    concurrency_per_worker: int,
    enable_ui: Optional[bool] = None,
    log_path: Optional[str] = None,
):
    """Run the trace-driven streaming load test with multiprocessing."""

    # Build token pool in main process
    print("Building token pool...")
    token_pool, tokenizer = build_token_pool(text_file, tokenizer_path, 100_000, seed)
    print(f"Token pool length: {len(token_pool)}")

    # Build endpoint URL
    native_base = base_url
    if native_base.endswith("/v1"):
        native_base = native_base[: -len("/v1")]
    endpoint_url = native_base.rstrip("/") + "/generate"

    # Read and sort trace by arrival time
    print("Loading trace...")
    rows = []
    with open(trace_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if 0 < max_requests <= len(rows):
                break
    rows.sort(key=lambda r: float(r["arrival"]))

    # Scale arrivals
    rate = float(rate) if rate and rate > 0 else 1.0
    for r in rows:
        r["scaled_arrival_ms"] = float(r["arrival"]) / rate

    total_requests = len(rows)
    print(f"Total requests: {total_requests}")
    print(f"Workers: {num_workers}")
    print(f"Concurrency per worker: {concurrency_per_worker}")
    print(f"Total concurrent capacity: {num_workers * concurrency_per_worker}")

    # Setup logging
    if log_path:
        logs_dir = os.path.dirname(log_path)
        if logs_dir:
            os.makedirs(logs_dir, exist_ok=True)
    else:
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(
            logs_dir, f"issue_stream_aiohttp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )

    # Create multiprocessing manager for shared state
    manager = mp.Manager()
    stats = SharedStats(expected_total=total_requests, manager=manager)
    request_queue = manager.Queue()
    stop_event = manager.Event()
    ready_queue = manager.Queue()  # Workers signal when ready

    # UI setup
    if enable_ui is None:
        enable_ui = sys.stdout.isatty()

    # Signal handler
    signal_count = [0]

    def signal_handler(signum, frame):
        """Handle SIGINT/SIGTERM gracefully."""
        signal_count[0] += 1
        print(f"\n[aiohttp] Received signal {signum}, shutting down gracefully...")
        stop_event.set()

        if signal_count[0] >= 2:
            print("[aiohttp] Multiple signals received, forcing immediate exit...")
            os._exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start worker processes
    print(f"Starting {num_workers} worker processes...")
    workers = []
    for worker_id in tqdm.tqdm(range(num_workers)):
        p = mp.Process(
            target=worker_process_main,
            args=(
                worker_id,
                request_queue,
                token_pool,
                endpoint_url,
                temperature,
                timeout,
                log_path,
                stats,
                concurrency_per_worker,
                seed,
                ready_queue,
            ),
            daemon=False,
        )
        p.start()
        workers.append(p)

    # Wait for all workers to signal readiness
    print("Waiting for all workers to be ready...")
    ready_workers = set()
    timeout_start = time.perf_counter()
    while len(ready_workers) < num_workers:
        try:
            worker_id = ready_queue.get(timeout=1.0)
            ready_workers.add(worker_id)
            if len(ready_workers) % 10 == 0 or len(ready_workers) == num_workers:
                print(f"  {len(ready_workers)}/{num_workers} workers ready...")
        except Exception:
            # Check if we've been waiting too long
            if time.perf_counter() - timeout_start > 30:
                print(f"  WARNING: Only {len(ready_workers)}/{num_workers} workers ready after 30s, proceeding anyway...")
                break
    print(f"All {len(ready_workers)} workers ready! Starting request submission...")

    # Start status display thread in main process
    import threading
    start_time = time.perf_counter()
    status_thread = None
    if enable_ui:
        status_thread = threading.Thread(
            target=status_display_thread,
            args=(stats, start_time, stop_event, log_path),
            daemon=False,
        )
        status_thread.start()

    start_time = time.perf_counter()
    try:
        # Submit requests to queue according to schedule
        print("Submitting requests to queue...")
        for idx, row in enumerate(rows):
            if stop_event.is_set():
                break

            # Wait until scheduled time
            target_s = row["scaled_arrival_ms"] / 1000.0
            while True:
                now_s = time.perf_counter() - start_time
                if now_s >= target_s or stop_event.is_set():
                    break
                sleep_time = target_s - now_s
                chunk_sleep = min(0.1, sleep_time)
                time.sleep(chunk_sleep)

            # Extract request parameters
            prefill = int(float(row["prefill"]))
            decode = int(float(row["decode"]))
            ttft_val = row.get("ttft")
            tpot_val = row.get("tpot")
            ttft = float(ttft_val) if ttft_val not in (None, "") else None
            tpot = float(tpot_val) if tpot_val not in (None, "") else None

            # Put request in queue (record enqueue time for accurate rate tracking)
            req_data = {
                "idx": idx,
                "prefill": prefill,
                "decode": decode,
                "ttft": ttft,
                "tpot": tpot,
                "enqueue_timestamp": time.time(),  # Record when main process enqueues
            }
            request_queue.put(req_data)

        print(f"\n[aiohttp] Finished queueing {total_requests} requests")
        print("[aiohttp] Waiting for workers to complete...")

        # Send sentinel values to stop workers
        for _ in range(num_workers):
            request_queue.put(None)

        # Wait for workers to finish
        for p in workers:
            p.join()

    except KeyboardInterrupt:
        print("\n[aiohttp] Interrupted, shutting down...")
        stop_event.set()
    except Exception as e:
        print(f"\n[aiohttp] Error: {e}")
        stop_event.set()
    finally:
        stop_event.set()

        # Send sentinel values to stop workers
        for _ in range(num_workers):
            try:
                request_queue.put(None)
            except Exception:
                pass

        # Give workers time to finish
        time.sleep(1.0)

        # Terminate workers if still alive
        for p in workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)

        # Wait for status thread
        if status_thread and status_thread.is_alive():
            status_thread.join(timeout=2.0)

    snapshot = stats.snapshot()
    return snapshot["submitted"], snapshot["completed"], snapshot["failed"]


def parse_args():
    p = argparse.ArgumentParser(
        description="High-performance trace-driven streaming request issuer using aiohttp + multiprocessing"
    )
    p.add_argument("--trace", required=True, help="CSV trace file with arrival times (ms)")
    p.add_argument("--text-file", required=True, help="Large text file to build token pool from")
    p.add_argument("--tokenizer", required=True, help="Tokenizer path/name (HF) to use")
    p.add_argument("--base-url", default="http://0.0.0.0:40000/v1", help="Base URL")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    p.add_argument("--rate", type=float, default=1.0, help="Arrival time scale (>1 speeds up)")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    p.add_argument("--max-requests", type=int, default=0, help="Limit number of requests (0 for all)")
    p.add_argument("--timeout", type=float, default=0, help="Per-request timeout seconds (0 to disable)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Number of worker processes (0 for auto=min(256, cpu_count))")
    p.add_argument("--concurrency-per-worker", type=int, default=50,
                   help="Max concurrent requests per worker (default: 50)")
    p.add_argument("--no-ui", action="store_true", help="Disable live dashboard UI")
    p.add_argument("--log-path", help="Optional path for the JSONL log output")
    return p.parse_args()


def main():
    # Set multiprocessing start method
    # Use 'fork' on Unix for fast process creation (0.1s vs 4s per process)
    # Fall back to 'spawn' on Windows or if fork is unavailable
    if sys.platform != 'win32' and 'fork' in mp.get_all_start_methods():
        mp.set_start_method('fork', force=True)
    else:
        mp.set_start_method('spawn', force=True)

    args = parse_args()

    # Auto-detect number of workers
    if args.num_workers <= 0:
        args.num_workers = min(256, mp.cpu_count())

    ui_enabled = (not args.no_ui) and sys.stdout.isatty()

    submitted, completed, failed = run_trace(
        trace_csv=args.trace,
        text_file=args.text_file,
        tokenizer_path=args.tokenizer,
        base_url=args.base_url,
        model=args.model,
        rate=args.rate,
        temperature=args.temperature,
        seed=args.seed,
        max_requests=args.max_requests,
        timeout=args.timeout,
        num_workers=args.num_workers,
        concurrency_per_worker=args.concurrency_per_worker,
        enable_ui=ui_enabled,
        log_path=args.log_path,
    )
    print(f"\nSubmitted={submitted}, Completed={completed}, Failed={failed}")


if __name__ == "__main__":
    main()
