"""
Trace-driven multi-threaded streaming request issuer.

Simpler than issue.py - each request runs in its own thread spawned at the scheduled time.
No thread pool, no async result processing - just direct threading.

Tracks timing for streaming responses:
- Start time (when request is submitted)
- Time to first token (TTFT)
- Inter-token delays (intervals between tokens 1->2, 2->3, ..., 9->10)
- Average inter-token delay

Usage example:

python -m slo.issue_stream \
  --trace /sgl-workspace/sglang/slo/trace/uniform_4096_1024.csv \
  --text-file /path/to/large_corpus.txt \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://0.0.0.0:40000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --rate 1.0
"""

import argparse
import csv
import random
import time
import threading
import json
import sys
import os
import signal
from typing import List, Optional
from datetime import datetime
from collections import deque

import requests

from sglang.srt.hf_transformers_utils import get_tokenizer

# Optional progress bars
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


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


class StreamStats:
    """Thread-safe stats for streaming requests."""

    def __init__(self, expected_total: int):
        self.expected_total = expected_total
        self._lock = threading.Lock()
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._active = 0

    def record_submit(self):
        with self._lock:
            self._submitted += 1
            self._active += 1

    def record_complete(self, success: bool):
        with self._lock:
            self._active -= 1
            if success:
                self._completed += 1
            else:
                self._failed += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "active": self._active,
                "expected_total": self.expected_total,
            }

    def all_done(self) -> bool:
        snap = self.snapshot()
        return (snap["completed"] + snap["failed"]) >= snap["expected_total"]


class CliStatusDisplay:
    """Minimal terminal dashboard."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._lock = threading.Lock()

    def render(self, lines: List[str], final: bool = False) -> None:
        output = "\n".join(lines)
        with self._lock:
            if self.enabled:
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.write(output)
                if final:
                    sys.stdout.write("\n")
                sys.stdout.flush()
            elif final:
                sys.stdout.write(output + "\n")
                sys.stdout.flush()


def send_streaming_request(
    endpoint_url: str,
    prompt_ids: List[int],
    decode_tokens: int,
    temperature: float,
    timeout: float,
    request_id: str,
    log_path: str,
    log_lock: threading.Lock,
    stats: StreamStats,
    ttft: Optional[float] = None,
    tpot: Optional[float] = None,
):
    """
    Send a streaming request and track timing:
    - Start time
    - Time to first token
    - Inter-token intervals (up to 10 tokens)
    - Average inter-token interval
    """

    submit_timestamp = time.time()
    start_time = time.perf_counter()

    sampling_params = {
        "max_new_tokens": max(0, int(decode_tokens)),
        "temperature": float(temperature),
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

    try:
        resp = requests.post(endpoint_url, json=data, stream=True, timeout=timeout or None)
        resp.raise_for_status()

        # Process streaming response
        for line in resp.iter_lines():
            if not line:
                continue

            line = line.decode('utf-8')

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
                    chunk = json.loads(data_str)
                    chunk_count += 1

                    # Extract actual token count from response
                    # SGLang native format has 'meta_info' with 'completion_tokens'
                    if 'meta_info' in chunk and 'completion_tokens' in chunk['meta_info']:
                        current_token_count = chunk['meta_info']['completion_tokens']

                        # Record time when we get NEW tokens
                        if current_token_count > prev_token_count:
                            token_time = time.perf_counter()
                            num_new_tokens = current_token_count - prev_token_count

                            if len(all_token_times) == 0:
                                # First token(s) - no previous time to interpolate from
                                for _ in range(num_new_tokens):
                                    all_token_times.append(token_time)
                                    if first_token_time is None:
                                        first_token_time = token_time
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
                    if 'text' in chunk:
                        output_text = chunk['text']

                except json.JSONDecodeError as e:
                    print(f"[DEBUG {request_id}] JSON decode error: {e}, data: {data_str[:100]}")
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

        # Keep first 10 intervals for printing/logging
        first_10_intervals = [round(x, 2) for x in all_intervals[:10]]

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
        if tpot is not None:
            record["target_tpot_ms"] = tpot
        # Append detailed timing and text at the end
        record["intervals"] = first_10_intervals
        record["output_text"] = output_text

        success = True

    except Exception as e:
        end_time = time.perf_counter()
        total_duration_ms = (end_time - start_time) * 1000
        error_msg = str(e)
        print(f"[stream] Error in request {request_id}: {error_msg}")

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
        record["output_text"] = output_text  # Partial output text before failure

    finally:
        # Log the result
        with log_lock:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Update stats
        stats.record_complete(success)


def status_display_thread(
    stats: StreamStats,
    submission_times: deque,
    submission_lock: threading.Lock,
    start_time: float,
    stop_event: threading.Event,
    log_path: str,
):
    """Background thread to render the terminal dashboard."""

    ui = CliStatusDisplay(enabled=sys.stdout.isatty())
    log_display_path = os.path.relpath(log_path)

    try:
        while not stop_event.is_set():
            for _ in range(5):
                if stop_event.is_set():
                    break
                time.sleep(0.1)

            current_time = time.perf_counter()
            elapsed_s = current_time - start_time

            # Calculate submission rate
            with submission_lock:
                cutoff_time = current_time - 10.0
                while submission_times and submission_times[0] < cutoff_time:
                    submission_times.popleft()
                submit_count_10s = len(submission_times)

            submit_speed = submit_count_10s / min(10.0, elapsed_s) if elapsed_s > 0 else 0.0

            # Get stats
            snapshot = stats.snapshot()
            submitted = snapshot["submitted"]
            completed = snapshot["completed"]
            failed = snapshot["failed"]
            active = snapshot["active"]
            expected_total = snapshot["expected_total"]

            percent_complete = (
                (completed + failed) / expected_total * 100.0 if expected_total > 0 else 0.0
            )

            lines = [
                "SLO Streaming Issue Runner - Live Stats",
                "========================================",
                f"Total Requests       : {expected_total}",
                f"Submitted            : {submitted}",
                f"Active Threads       : {active}",
                f"Completed / Failed   : {completed} / {failed}",
                f"Submit Rate (req/s)  : {submit_speed:.2f}",
                f"Elapsed (s)          : {elapsed_s:.1f}",
                f"Progress             : {percent_complete:.1f}%",
                "",
                f"Log File             : {log_display_path}",
                "Ctrl+C twice to force stop",
            ]
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
            percent_complete = (
                (completed + failed) / expected_total * 100.0 if expected_total > 0 else 0.0
            )

            lines = [
                "SLO Streaming Issue Runner - Final Stats",
                "=========================================",
                f"Total Requests       : {expected_total}",
                f"Submitted            : {submitted}",
                f"Active Threads       : {active}",
                f"Completed / Failed   : {completed} / {failed}",
                f"Elapsed (s)          : {elapsed_s:.1f}",
                f"Progress             : {percent_complete:.1f}%",
                "",
                f"Log File             : {log_display_path}",
            ]
            ui.render(lines, final=True)
        except Exception:
            # Suppress errors during shutdown
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
    enable_ui: Optional[bool] = None,
):
    """Run the trace-driven streaming load test."""

    # Build token pool
    token_pool, tokenizer = build_token_pool(text_file, tokenizer_path, 100_000, seed)
    print(f"Token pool length: {len(token_pool)}")

    # Setup stop event for signal handling
    stop_event = threading.Event()

    # Track signal count for aggressive shutdown
    signal_count = [0]

    def signal_handler(signum, frame):
        """Handle SIGINT/SIGTERM gracefully."""
        signal_count[0] += 1
        print(f"\n[stream] Received signal {signum}, shutting down gracefully... (signal #{signal_count[0]})")
        stop_event.set()

        # If multiple signals received, force exit
        if signal_count[0] >= 2:
            print("[stream] Multiple signals received, forcing immediate exit...")
            import os
            os._exit(1)

        # Start watchdog timer for emergency exit
        def emergency_exit():
            time.sleep(5.0)  # Give 5 seconds for graceful shutdown
            if signal_count[0] > 0:  # Only if we received a signal
                print("[stream] Emergency timeout reached, forcing exit...")
                import os
                os._exit(1)

        emergency_thread = threading.Thread(target=emergency_exit, daemon=True)
        emergency_thread.start()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Build endpoint URL
    native_base = base_url
    if native_base.endswith("/v1"):
        native_base = native_base[: -len("/v1")]
    endpoint_url = native_base.rstrip("/") + "/generate"

    rng = random.Random(seed + 1)

    # Read and sort trace by arrival time
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

    # Setup logging
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(
        logs_dir, f"issue_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    log_lock = threading.Lock()

    # Stats
    stats = StreamStats(expected_total=total_requests)

    # UI setup
    if enable_ui is None:
        enable_ui = sys.stdout.isatty()

    submission_times = deque()
    submission_lock = threading.Lock()

    # Start status display thread
    start_time = time.perf_counter()
    status_thread = None
    if enable_ui:
        status_thread = threading.Thread(
            target=status_display_thread,
            args=(stats, submission_times, submission_lock, start_time, stop_event, log_path),
            daemon=False,  # Changed to False for clean shutdown
        )
        status_thread.start()

    # Track threads
    threads = []

    try:
        # Submit requests according to schedule
        for idx, row in enumerate(rows):
            if stop_event.is_set():
                break

            # Wait until scheduled time
            target_s = row["scaled_arrival_ms"] / 1000.0
            now_s = time.perf_counter() - start_time
            if now_s < target_s:
                sleep_time = target_s - now_s
                while sleep_time > 0 and not stop_event.is_set():
                    chunk_sleep = min(0.1, sleep_time)
                    time.sleep(chunk_sleep)
                    sleep_time -= chunk_sleep

            if stop_event.is_set():
                break

            # Extract request parameters
            prefill = int(float(row["prefill"]))
            decode = int(float(row["decode"]))
            ttft_val = row.get("ttft")
            tpot_val = row.get("tpot")
            ttft = float(ttft_val) if ttft_val not in (None, "") else None
            tpot = float(tpot_val) if tpot_val not in (None, "") else None

            # Sample prompt
            prompt_ids = sample_prompt_from_pool(token_pool, prefill, rng)
            request_id = f"req_{idx:06d}_{int(time.time() * 1000) % 1000000:06d}"

            # Record submission
            stats.record_submit()
            with submission_lock:
                submission_times.append(time.perf_counter())

            # Spawn thread for this request
            t = threading.Thread(
                target=send_streaming_request,
                args=(
                    endpoint_url,
                    prompt_ids,
                    decode,
                    temperature,
                    timeout,
                    request_id,
                    log_path,
                    log_lock,
                    stats,
                    ttft,
                    tpot,
                ),
                daemon=False,
            )
            t.start()
            threads.append(t)

        print(f"\n[stream] Finished submitting {len(threads)} requests")
        print("[stream] Waiting for all requests to complete...")

        # Wait for all threads to complete
        for t in threads:
            if stop_event.is_set():
                break
            t.join()

    except KeyboardInterrupt:
        print("\n[stream] Interrupted, shutting down...")
        stop_event.set()
    except Exception as e:
        print(f"\n[stream] Error: {e}")
        stop_event.set()
    finally:
        stop_event.set()

        # Give threads a moment to finish
        if not stats.all_done():
            time.sleep(1.0)

        # Wait for status thread to finish cleanly
        if status_thread and status_thread.is_alive():
            status_thread.join(timeout=2.0)

    # Quick exit if interrupted
    if stop_event.is_set():
        print("[stream] Cleanup complete. Exiting...")
        snapshot = stats.snapshot()
        return snapshot["submitted"], snapshot["completed"], snapshot["failed"]

    snapshot = stats.snapshot()
    return snapshot["submitted"], snapshot["completed"], snapshot["failed"]


def parse_args():
    p = argparse.ArgumentParser()
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
    p.add_argument("--no-ui", action="store_true", help="Disable live dashboard UI")
    return p.parse_args()


def main():
    args = parse_args()
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
        timeout=(None if args.timeout == 0 else args.timeout),
        enable_ui=ui_enabled,
    )
    print(f"\nSubmitted={submitted}, Completed={completed}, Failed={failed}")


if __name__ == "__main__":
    main()
