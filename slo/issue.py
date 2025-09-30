"""
Trace-driven multi-threaded request issuer.

- Reads a CSV trace with columns: id,prefill,decode,tpot,ttft,arrival
  - prefill: number of prompt tokens to send
  - decode: number of tokens to generate (forced via max_tokens)
  - arrival: scheduled arrival time in milliseconds since start

- Reads a large text file, tokenizes it with the specified tokenizer, selects a
  fixed random 100,000-token window centered around the middle, and for each
  request randomly samples a contiguous prefill-length segment from this pool
  as the prompt.

- Releases requests according to arrival times (scaled by --rate). Each request
  is submitted individually to the `/generate` endpoint as soon as it becomes due,
  using a thread pool for concurrent processing. By default, waits for completion
  with dual progress bars (use --no-wait for fire-and-forget mode).

Usage examples:

# Default mode: Track both submission and completion with dual progress bars
python -m slo.issue \
  --trace /sgl-workspace/sglang/slo/trace/uniform_4096_1024.csv \
  --text-file /path/to/large_corpus.txt \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://0.0.0.0:40000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --rate 1.0 \
  --max-workers 256

# Fire and forget mode: Submit requests and exit immediately  
python -m slo.issue \
  --trace /sgl-workspace/sglang/slo/trace/uniform_4096_1024.csv \
  --text-file /path/to/large_corpus.txt \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://0.0.0.0:40000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --rate 1.0 \
  --max-workers 256 \
  --no-wait
"""

import argparse
import csv
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
import signal
from collections import deque

import requests
import os
import json
import threading
import sys
from datetime import datetime

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
) -> Tuple[List[int], object]:
    """Stream-tokenize from the beginning and take the first `num_tokens` tokens.

    Returns (token_pool, tokenizer). If the file has fewer than `num_tokens`
    tokens, returns all available tokens.
    """
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


def any_not_none(values: List[Optional[float]]) -> bool:
    return any(v is not None for v in values)



class IssuerStats:
    """Thread-safe counters for executor and request lifecycle metrics."""

    def __init__(self, max_workers: int, expected_total: int, wait_threshold_ms: float = 1.0):
        self.max_workers = max_workers
        self.expected_total = expected_total
        self._wait_threshold_ms = max(0.0, float(wait_threshold_ms))
        self._lock = threading.Lock()
        self._submitted = 0
        self._started = 0
        self._completed = 0
        self._failed = 0
        self._active_workers = 0
        self._total_queue_wait_ms = 0.0
        self._queue_wait_events = 0
        self._max_queue_wait_ms = 0.0
        self._last_queue_wait_ms = 0.0
        self._max_queue_depth = 0

    def record_submission(self) -> int:
        """Record a submission attempt and return current queue depth."""
        with self._lock:
            self._submitted += 1
            queued = max(0, self._submitted - self._started)
            if queued > self._max_queue_depth:
                self._max_queue_depth = queued
            return queued

    def record_start(self, queued_timestamp: Optional[float]) -> float:
        """Record when a worker thread starts handling the request."""
        wait_ms = 0.0
        if queued_timestamp is not None:
            wait_ms = max(0.0, (time.perf_counter() - queued_timestamp) * 1000.0)
        significant = wait_ms >= self._wait_threshold_ms
        with self._lock:
            self._started += 1
            self._active_workers += 1
            self._last_queue_wait_ms = (wait_ms if significant else 0.0)
            if significant:
                self._queue_wait_events += 1
                self._total_queue_wait_ms += wait_ms
                if wait_ms > self._max_queue_wait_ms:
                    self._max_queue_wait_ms = wait_ms
        return wait_ms

    def record_completion(self, success: bool) -> None:
        with self._lock:
            if self._active_workers > 0:
                self._active_workers -= 1
            if success:
                self._completed += 1
            else:
                self._failed += 1

    def snapshot(self) -> dict:
        with self._lock:
            queued = max(0, self._submitted - self._started)
            avg_wait = (
                self._total_queue_wait_ms / self._queue_wait_events
                if self._queue_wait_events > 0
                else 0.0
            )
            return {
                "submitted": self._submitted,
                "started": self._started,
                "completed": self._completed,
                "failed": self._failed,
                "active_workers": self._active_workers,
                "queued": queued,
                "max_queue_depth": self._max_queue_depth,
                "queue_wait_events": self._queue_wait_events,
                "avg_queue_wait_ms": avg_wait,
                "max_queue_wait_ms": self._max_queue_wait_ms,
                "last_queue_wait_ms": self._last_queue_wait_ms,
                "expected_total": self.expected_total,
            }

    def all_done(self) -> bool:
        snap = self.snapshot()
        target = snap["expected_total"] if snap["expected_total"] > 0 else snap["submitted"]
        if target <= 0:
            return False
        return (snap["completed"] + snap["failed"]) >= target


class CliStatusDisplay:
    """Minimal terminal dashboard that mirrors the router's live UI style."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._lock = threading.Lock()

    def render(self, lines: List[str], final: bool = False) -> None:
        output = "\n".join(lines)
        with self._lock:
            if self.enabled:
                sys.stdout.write("[2J[H")
                sys.stdout.write(output)
                if final:
                    sys.stdout.write("\n")
                sys.stdout.flush()
            elif final:
                sys.stdout.write(output + "\n")
                sys.stdout.flush()


def _format_dashboard(
    stats_snapshot: dict,
    total_requests: int,
    submit_speed: float,
    lag_s: float,
    elapsed_s: float,
    log_path: str,
    max_workers: int,
) -> List[str]:
    completed = stats_snapshot["completed"]
    failed = stats_snapshot["failed"]
    submitted = stats_snapshot["submitted"]
    started = stats_snapshot["started"]
    active_workers = stats_snapshot["active_workers"]
    queued = stats_snapshot["queued"]
    max_queue_depth = stats_snapshot["max_queue_depth"]
    avg_wait = stats_snapshot["avg_queue_wait_ms"]
    max_wait = stats_snapshot["max_queue_wait_ms"]
    last_wait = stats_snapshot["last_queue_wait_ms"]
    wait_events = stats_snapshot["queue_wait_events"]

    waiting = "YES" if queued > 0 or last_wait > 0 else "NO"
    percent_complete = (
        (completed + failed) / total_requests * 100.0 if total_requests > 0 else 0.0
    )

    lines = [
        "SLO Issue Runner - Live Stats",
        "==============================",
        f"Trace Requests       : {total_requests}",
        f"Submitted / Started  : {submitted} / {started}",
        f"Completed / Failed   : {completed} / {failed}",
        f"Active Threads       : {active_workers} / {max_workers}",
        f"Backlog (queued)     : {queued} (max {max_queue_depth})",
        f"Waiting On Executor  : {waiting}",
        f"Waited Requests      : {wait_events}",
        f"Queue Wait (ms)      : last {last_wait:.1f} | avg {avg_wait:.1f} | max {max_wait:.1f}",
        f"Submit Rate (req/s)  : {submit_speed:.2f}",
        f"Schedule Lag (s)     : {lag_s:+.2f}",
        f"Elapsed (s)          : {elapsed_s:.1f}",
        f"Progress             : {percent_complete:.1f}%",
        "",
        f"Log File             : {log_path}",
        "Ctrl+C twice to force stop",
    ]
    return lines



def send_request(
    endpoint_url: str,
    prompt_ids: List[int],
    decode_tokens: int,
    temperature: float,
    timeout: float,
    ttft: Optional[float] = None,
    tpot: Optional[float] = None,
    request_id: Optional[str] = None,
    stats: Optional[IssuerStats] = None,
    queued_timestamp: Optional[float] = None,
):
    """Send a single /generate request and return logging metadata."""

    submit_timestamp = time.time()

    sampling_params = {
        "max_new_tokens": max(0, int(decode_tokens)),
        "temperature": float(temperature),
    }
    data = {"input_ids": [prompt_ids], "sampling_params": sampling_params}
    if ttft is not None:
        data["target_ttft_ms"] = ttft
    if tpot is not None:
        data["target_tpot_ms"] = tpot

    wait_ms = 0.0
    if stats is not None:
        wait_ms = stats.record_start(queued_timestamp)

    request_start_time = time.perf_counter()
    resp = None
    success = False

    try:
        resp = requests.post(endpoint_url, json=data, timeout=timeout or None)
        request_end_time = time.perf_counter()
        request_duration_ms = (request_end_time - request_start_time) * 1000
        resp.raise_for_status()
        body = resp.json()
        record = {
            "request_id": request_id,
            "submit_timestamp": submit_timestamp,
            "queue_wait_ms": wait_ms,
            "request_duration_ms": request_duration_ms,
            "status": resp.status_code,
            "status_text": "SUCCESS",
            "decode": decode_tokens,
            "batch_size": 1,
            "response": body,
        }
        if ttft is not None:
            record["target_ttft_ms"] = ttft
        if tpot is not None:
            record["target_tpot_ms"] = tpot
        success = True
        return 1, 0, record
    except Exception as e:
        request_end_time = time.perf_counter()
        request_duration_ms = (request_end_time - request_start_time) * 1000
        status_code = getattr(resp, "status_code", 0) if resp is not None else 0
        message_source = resp.text[:500] if getattr(resp, "text", None) else str(e)
        print(f"[issue] /generate error status={status_code}: {message_source}")
        record = {
            "request_id": request_id,
            "submit_timestamp": submit_timestamp,
            "queue_wait_ms": wait_ms,
            "request_duration_ms": request_duration_ms,
            "status": status_code,
            "status_text": "FAILED",
            "decode": decode_tokens,
            "batch_size": 1,
            "error": message_source,
            "exception_type": type(e).__name__,
        }
        if ttft is not None:
            record["target_ttft_ms"] = ttft
        if tpot is not None:
            record["target_tpot_ms"] = tpot
        return 0, 1, record
    finally:
        if stats is not None:
            stats.record_completion(success)




def status_display_thread(
    stats: IssuerStats,
    submission_times: deque,
    submission_lock: threading.Lock,
    start_time: float,
    stop_event: threading.Event,
    submission_complete: threading.Event,
    ui_stop_event: threading.Event,
    actual_vs_expected: List[float],
    total_requests: int,
    log_path: str,
):
    """Background thread to render the terminal dashboard."""

    ui = CliStatusDisplay(enabled=sys.stdout.isatty())
    log_display_path = os.path.relpath(log_path)

    try:
        while not ui_stop_event.is_set():
            for _ in range(5):
                if ui_stop_event.is_set():
                    break
                time.sleep(0.1)

            current_time = time.perf_counter()
            elapsed_s = current_time - start_time

            with submission_lock:
                cutoff_time = current_time - 10.0
                while submission_times and submission_times[0] < cutoff_time:
                    submission_times.popleft()
                submit_count_10s = len(submission_times)

            submit_speed = submit_count_10s / min(10.0, elapsed_s) if elapsed_s > 0 else 0.0
            lag_s = actual_vs_expected[0]

            snapshot = stats.snapshot()
            lines = _format_dashboard(
                snapshot,
                total_requests,
                submit_speed,
                lag_s,
                elapsed_s,
                log_display_path,
                stats.max_workers,
            )
            ui.render(lines)

            if (stats.all_done() and submission_complete.is_set()) or stop_event.is_set():
                break
    except Exception:
        pass
    finally:
        snapshot = stats.snapshot()
        elapsed_s = max(0.0, time.perf_counter() - start_time)
        submit_speed = 0.0
        lag_s = actual_vs_expected[0]
        lines = _format_dashboard(
            snapshot,
            total_requests,
            submit_speed,
            lag_s,
            elapsed_s,
            log_display_path,
            stats.max_workers,
        )
        ui.render(lines, final=True)
        ui_stop_event.set()



def process_results_async(
    futures: List,
    log_path: str,
    log_lock: threading.Lock,
    pbar,
    completed_counter: List[int],  # Use list for mutable counter
    failed_counter: List[int],  # Use list for mutable counter
    stop_event: threading.Event,
    submission_complete: threading.Event,  # Signal when all requests submitted
):
    """Background thread to process request results as they complete."""
    completed = 0
    failed = 0
    
    try:
        # Wait for futures to start appearing (allows parallel submission)
        while len(futures) == 0 and not submission_complete.is_set() and not stop_event.is_set():
            time.sleep(0.05)
            
        # Process futures as they complete
        # Keep track of which futures we've already processed to avoid double-counting
        processed_futures = set()
        
        while not stop_event.is_set():
            # Early exit check
            if stop_event.is_set():
                break
                
            # Find any newly completed futures that we haven't processed yet
            newly_completed = []
            for fut in futures:
                # Check for early exit during iteration
                if stop_event.is_set():
                    break
                if fut not in processed_futures and fut.done():
                    newly_completed.append(fut)
                    processed_futures.add(fut)
            
            # Process newly completed futures
            for fut in newly_completed:
                # Check for early exit during processing
                if stop_event.is_set():
                    break
                    
                try:
                    batch_completed, batch_failed, log_record = fut.result(timeout=0.1)  # Don't wait long for results
                except Exception as e:
                    batch_completed, batch_failed, log_record = 0, 1, {"error": str(e)}
                    
                completed += batch_completed
                failed += batch_failed
                
                if pbar:
                    pbar.update(batch_completed + batch_failed)
                        
                # Skip logging if we're shutting down
                if not stop_event.is_set():
                    try:
                        with log_lock:
                            with open(log_path, "a", encoding="utf-8") as lf:
                                lf.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"[issue] Failed to write log: {e}")
                    
            # Check if we're completely done
            if submission_complete.is_set() and (completed + failed) >= len(futures):
                break
                
            # Short sleep to avoid busy waiting, but check stop_event frequently
            for _ in range(10):  # 10 * 0.01s = 0.1s total
                if stop_event.is_set():
                    break
                time.sleep(0.01)
                
    except Exception as e:
        print(f"[issue] Error in result processor: {e}")
    finally:
        completed_counter[0] = completed
        failed_counter[0] = failed
        if pbar:
            pbar.close()


def run_trace(
    trace_csv: str,
    text_file: str,
    tokenizer_path: str,
    base_url: str,
    model: str,
    rate: float,
    max_workers: int,
    temperature: float,
    seed: int,
    max_requests: int,
    timeout: float,
    wait_completion: bool = True,
    enable_ui: Optional[bool] = None,
    enable_progress: Optional[bool] = None,
    queue_wait_threshold_ms: float = 1.0,
):
    token_pool, tokenizer = build_token_pool(text_file, tokenizer_path, 100_000, seed)
    print(f"Token pool length: {len(token_pool)}")
    # Derive native base (strip trailing /v1 if present) and build /generate endpoint
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

    # Scale arrivals: higher rate compresses the timeline
    rate = float(rate) if rate and rate > 0 else 1.0

    # Convert arrival to scaled ms and keep pointer of next row to release
    for r in rows:
        r["scaled_arrival_ms"] = float(r["arrival"]) / rate

    submitted = 0
    completed = 0
    start_time = time.perf_counter()
    idx = 0

    # Decide UI/progress defaults if not provided
    if enable_ui is None:
        enable_ui = sys.stdout.isatty()
    if enable_progress is None:
        enable_progress = (tqdm is not None) and (not enable_ui)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        # Logging setup
        total_requests = len(rows)
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(
            logs_dir, f"issue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        log_lock = threading.Lock()
        # Stats for worker/queue tracking and UI
        stats = IssuerStats(
            max_workers=max_workers,
            expected_total=total_requests if wait_completion else 0,
            wait_threshold_ms=queue_wait_threshold_ms,
        )

        # Setup progress bars and synchronization
        completed_counter = [0]  # Mutable counter for background thread
        failed_counter = [0]  # Mutable counter for failed requests
        stop_event = threading.Event()
        submission_complete = threading.Event()  # Signal when all requests submitted
        ui_stop_event = threading.Event()
        
        # Setup status tracking for real-time display
        submission_times = deque()  # Track submission timestamps for speed calculation
        submission_lock = threading.Lock()
        current_idx = [0]  # Mutable current index for trace position
        actual_vs_expected = [0.0]  # Track cumulative lag in seconds
        
        if wait_completion:
            # Two progress bars: submission and completion
            if enable_progress and tqdm:
                submit_pbar = tqdm(total=total_requests, desc="Submitting", unit="req", position=0, disable=not enable_progress)
                complete_pbar = tqdm(total=total_requests, desc="Completed", unit="req", position=1, disable=not enable_progress)
            else:
                submit_pbar = None
                complete_pbar = None
            
            # Start background result processor immediately (in parallel)
            result_processor = threading.Thread(
                target=process_results_async,
                args=(futures, log_path, log_lock, complete_pbar, completed_counter, failed_counter, stop_event, submission_complete),
                daemon=True
            )
            result_processor.start()
            
            # Start status display thread
            if enable_ui:
                status_thread = threading.Thread(
                    target=status_display_thread,
                    args=(stats, submission_times, submission_lock, start_time, stop_event, submission_complete, ui_stop_event, actual_vs_expected, total_requests, log_path),
                    daemon=True
                )
                status_thread.start()
        else:
            # Single submission progress bar for fire-and-forget mode
            submit_pbar = tqdm(total=total_requests, desc="Submitting", unit="req", disable=not (enable_progress and tqdm)) if tqdm else None
            complete_pbar = None
            result_processor = None
            
            # Start status display thread for fire-and-forget mode too
            if enable_ui:
                status_thread = threading.Thread(
                    target=status_display_thread,
                    args=(stats, submission_times, submission_lock, start_time, stop_event, submission_complete, ui_stop_event, actual_vs_expected, total_requests, log_path),
                    daemon=True
                )
                status_thread.start()
        
        # Track signal count for aggressive shutdown
        signal_count = [0]
        
        # Setup signal handler for graceful shutdown
        def signal_handler(signum, frame):
            signal_count[0] += 1
            print(f"\n[issue] Received signal {signum}, shutting down gracefully... (signal #{signal_count[0]})")
            stop_event.set()
            
            # If multiple signals received, force exit
            if signal_count[0] >= 2:
                print("[issue] Multiple signals received, forcing immediate exit...")
                import os
                os._exit(1)
                
            # Start watchdog timer for emergency exit
            def emergency_exit():
                time.sleep(5.0)  # Give 5 seconds for graceful shutdown
                if signal_count[0] > 0:  # Only if we received a signal
                    print("[issue] Emergency timeout reached, forcing exit...")
                    import os
                    os._exit(1)
            
            emergency_thread = threading.Thread(target=emergency_exit, daemon=True)
            emergency_thread.start()
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Submit requests individually as they become due
            for idx in range(len(rows)):
                # Check for early termination
                if stop_event.is_set():
                    break
                    
                row = rows[idx]
                
                # Wait until this request's scheduled time
                target_s = row["scaled_arrival_ms"] / 1000.0
                now_s = time.perf_counter() - start_time
                if now_s < target_s:
                    sleep_time = target_s - now_s
                    # Sleep in small chunks to be responsive to stop_event
                    while sleep_time > 0 and not stop_event.is_set():
                        chunk_sleep = min(0.1, sleep_time)  # Sleep max 0.1s at a time
                        time.sleep(chunk_sleep)
                        sleep_time -= chunk_sleep
                
                # Calculate precise lag for this submission
                actual_submit_time = time.perf_counter() - start_time
                lag_for_this_request = actual_submit_time - target_s
                actual_vs_expected[0] = lag_for_this_request
                
                # Extract request parameters
                prefill = int(float(row["prefill"]))
                decode = int(float(row["decode"]))
                ttft_val = row.get("ttft")
                tpot_val = row.get("tpot")
                ttft = float(ttft_val) if ttft_val not in (None, "") else None
                tpot = float(tpot_val) if tpot_val not in (None, "") else None
                
                # Submit request to thread pool
                prompt_ids = sample_prompt_from_pool(token_pool, prefill, rng)
                request_id = f"req_{idx:06d}_{int(time.time() * 1000) % 1000000:06d}"
                # Record submission for stats and potential queueing
                queued_timestamp = time.perf_counter()
                stats.record_submission()

                fut = executor.submit(
                    send_request,
                    endpoint_url,
                    prompt_ids,
                    decode,
                    temperature,
                    timeout,
                    ttft,
                    tpot,
                    request_id,
                    stats,
                    queued_timestamp,
                )
                futures.append(fut)
                submitted += 1
                
                # Track submission time for speed calculation
                with submission_lock:
                    submission_times.append(time.perf_counter())
                
                # Update current index for lag calculation
                current_idx[0] = idx
                
                # Update submission progress bar
                if submit_pbar:
                    submit_pbar.update(1)

        
        except KeyboardInterrupt:
            print("\n[issue] Interrupted during submission")
            stop_event.set()
        except Exception as e:
            print(f"\n[issue] Error during submission: {e}")
            stop_event.set()
        
        print(f"[issue] Finished submitting {submitted} requests")
        
        # Signal that submission is complete
        submission_complete.set()
        
        # Close submission progress bar
        if submit_pbar:
            submit_pbar.close()
            
        # If interrupted, quickly shutdown the executor
        if stop_event.is_set():
            print("[issue] Shutting down thread pool...")
            # Cancel all pending futures
            cancelled_count = 0
            for fut in futures:
                if fut.cancel():
                    cancelled_count += 1
            if cancelled_count > 0:
                print(f"[issue] Cancelled {cancelled_count} pending requests")
            executor.shutdown(wait=False)  # Don't wait for running tasks
            
        if wait_completion and result_processor:

            
            if not stop_event.is_set():
                print("[issue] Waiting for all requests to complete...")
                # Wait for result processor with timeout to be responsive
                result_processor.join(timeout=1.0)
                if result_processor.is_alive():
                    print("[issue] Result processor still running, forcing shutdown...")
                    stop_event.set()
                    result_processor.join(timeout=1.0)  # Give it 1 more second
                    
                    # If still alive, abandon it and exit
                    if result_processor.is_alive():
                        print("[issue] Result processor unresponsive, abandoning...")
            else:
                print("[issue] Shutdown requested, skipping completion wait...")
            completed = completed_counter[0]
            failed = failed_counter[0]
        else:
            print("[issue] Exiting without waiting for request completion")
            completed = 0  # Unknown since we're not waiting
            failed = 0  # Unknown since we're not waiting
            # Close completion progress bar if it exists
            if complete_pbar:
                complete_pbar.close()

    # Final cleanup and quick exit if interrupted
    if stop_event.is_set():
        print("[issue] Cleanup complete. Exiting...")
        import sys
        sys.exit(0)

    # We consider 'submitted' as number of logical requests (prompts).
    return submitted, completed, failed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True, help="CSV trace file with arrival times (ms)")
    p.add_argument("--text-file", required=True, help="Large text file to build token pool from")
    p.add_argument("--tokenizer", required=True, help="Tokenizer path/name (HF) to use")
    p.add_argument("--base-url", default="http://0.0.0.0:40000/v1", help="OpenAI-compatible base URL")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name on server")
    p.add_argument("--rate", type=float, default=1.0, help="Arrival time scale (>1 speeds up, arrival_ms / rate)")
    p.add_argument("--max-workers", type=int, default=256, help="Max concurrent worker threads")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    p.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")
    p.add_argument("--max-requests", type=int, default=0, help="Limit number of requests (0 for all)")
    p.add_argument("--timeout", type=float, default=0, help="Per-request client timeout seconds (0 to disable)")
    p.add_argument("--no-wait", action="store_true", help="Exit immediately after submission (fire-and-forget mode)")
    p.add_argument("--no-ui", action="store_true", help="Disable live dashboard UI (use progress bars only)")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    p.add_argument("--queue-wait-threshold-ms", type=float, default=1.0, help="Minimum queue wait (ms) counted as waiting")
    return p.parse_args()


def main():
    args = parse_args()
    # Decide UI/progress based on flags and TTY
    ui_enabled = (not args.no_ui) and sys.stdout.isatty()
    progress_enabled = (not args.no_progress) and (tqdm is not None) and (not ui_enabled)
    submitted, completed, failed = run_trace(
        trace_csv=args.trace,
        text_file=args.text_file,
        tokenizer_path=args.tokenizer,
        base_url=args.base_url,
        model=args.model,
        rate=args.rate,
        max_workers=args.max_workers,
        temperature=args.temperature,
        seed=args.seed,
        max_requests=args.max_requests,
        timeout=(None if args.timeout == 0 else args.timeout),
        wait_completion=not args.no_wait,
        enable_ui=ui_enabled,
        enable_progress=progress_enabled,
        queue_wait_threshold_ms=args.queue_wait_threshold_ms,
    )
    print(f"Submitted={submitted}, Completed={completed}, Failed={failed}")


if __name__ == "__main__":
    main()
