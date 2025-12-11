#!/usr/bin/env python3
"""
Non-Linear Load Evaluation Runner (8-GPU Parallel)

Runs rate sweep on 8 SGLang servers (one per GPU) to measure iteration time vs load.
Collects worker traces (iteration metrics) for each rate.

8 servers run continuously on ports 31001-31008. Tests are distributed across
servers in parallel (up to 8 concurrent tests). After each test, log entries
are segmented per-server and copied to that test's worker_log directory.

Usage:
    python run_eval.py
    python run_eval.py --traces lmsys,sharegpt,splitwise --rates 5,10,15,20,25,30,35,40,45,50 --duration 60
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(SCRIPT_DIR)
SLO_DIR = os.path.join(os.path.dirname(EVAL_DIR), "slo")

# Default config
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
SERVER_HOST = "0.0.0.0"

# 8 servers on ports 31001-31008
NUM_SERVERS = 8
SERVER_PORTS = list(range(31001, 31001 + NUM_SERVERS))


class TeeOutput:
    """Tee stdout/stderr to both console and a log file."""

    def __init__(self, log_path: str):
        self.log_file = open(log_path, "w", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, data):
        self.stdout.write(data)
        self.stdout.flush()
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.log_file.close()


class LaunchError(Exception):
    """Exception raised when server launch fails."""
    pass


def pst_now() -> datetime:
    """Return the current time in the America/Los_Angeles timezone."""
    try:
        tz = ZoneInfo("America/Los_Angeles") if ZoneInfo else timezone(timedelta(hours=-8))
    except Exception:
        tz = timezone(timedelta(hours=-8))
    return datetime.now(tz)


def load_config() -> Dict:
    """Load configuration from config.json."""
    config_path = os.path.join(SCRIPT_DIR, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def expand_traces(trace_args: List[str], config: Dict) -> List[Dict]:
    """Expand trace shortcuts to actual trace configs."""
    all_traces = config.get("traces", [])
    trace_by_name = {t["name"]: t for t in all_traces}

    # Define shortcuts
    nanoflow_names = ["lmsys", "sharegpt", "splitwise"]
    uniform_names = ["uniform_512_512", "uniform_4096_1024"]

    expanded = []
    seen = set()

    for t in trace_args:
        if t == "all":
            for trace in all_traces:
                if trace["name"] not in seen:
                    expanded.append(trace)
                    seen.add(trace["name"])
        elif t == "nanoflow":
            for name in nanoflow_names:
                if name in trace_by_name and name not in seen:
                    expanded.append(trace_by_name[name])
                    seen.add(name)
        elif t == "uniform":
            for name in uniform_names:
                if name in trace_by_name and name not in seen:
                    expanded.append(trace_by_name[name])
                    seen.add(name)
        elif t in trace_by_name:
            if t not in seen:
                expanded.append(trace_by_name[t])
                seen.add(t)
        else:
            print(f"[warning] Unknown trace: {t}")

    return expanded


def create_output_structure(trace_name: str, rate: float, base_dir: str) -> str:
    """Create output directory structure: trace/rate_<X>/."""
    rate_str = f"{rate:.1f}".replace(".", "p")
    output_dir = os.path.join(base_dir, trace_name, f"rate_{rate_str}")

    subdirs = ["worker_log", "client_log"]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    return output_dir


def cleanup_processes():
    """Kill all sglang-related processes."""
    print("[cleanup] Killing existing sglang processes...")
    subprocess.run(["pkill", "-f", "-9", "sglang"], check=False, capture_output=True)
    time.sleep(2)


def launch_server(worker_log_dir: str) -> subprocess.Popen:
    """Launch server using launch_server.sh.

    Args:
        worker_log_dir: Top-level worker log directory for the entire run
    """
    script_path = os.path.join(SCRIPT_DIR, "launch_server.sh")
    if not os.path.exists(script_path):
        raise LaunchError(f"Server script not found: {script_path}")

    log_path = os.path.join(worker_log_dir, "server_stdout.log")

    env = os.environ.copy()
    env["EVAL_LOG_DIR"] = worker_log_dir

    print(f"[launch] Starting server from {script_path}")
    print(f"[launch] Server log: {log_path}")
    print(f"[launch] Worker logs: {worker_log_dir}")

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            ["bash", script_path],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
    return proc


def wait_for_server(timeout: int = 600) -> bool:
    """Wait for all 8 servers to respond to health check."""
    deadline = time.time() + timeout
    print(f"[health] Waiting for {NUM_SERVERS} servers on ports {SERVER_PORTS[0]}-{SERVER_PORTS[-1]}...")

    ready_servers = set()

    while time.time() < deadline:
        for port in SERVER_PORTS:
            if port in ready_servers:
                continue
            try:
                resp = requests.get(f"http://{SERVER_HOST}:{port}/health", timeout=2)
                if resp.status_code == 200:
                    ready_servers.add(port)
                    print(f"[health] Server on port {port} is ready! ({len(ready_servers)}/{NUM_SERVERS})")
            except requests.exceptions.RequestException:
                pass

        if len(ready_servers) == NUM_SERVERS:
            print(f"[health] All {NUM_SERVERS} servers are ready!")
            return True

        elapsed = int(time.time() - (deadline - timeout))
        if elapsed % 30 == 0 and elapsed > 0:
            print(f"[health] Still waiting... ({len(ready_servers)}/{NUM_SERVERS} ready, elapsed: {elapsed}s)")
        time.sleep(5)

    raise LaunchError(f"Only {len(ready_servers)}/{NUM_SERVERS} servers ready within {timeout}s")


def get_log_line_count(log_path: str) -> int:
    """Get the current number of lines in the log file."""
    if not os.path.exists(log_path):
        return 0
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def segment_log(source_log: str, dest_log: str, start_line: int, end_line: Optional[int] = None):
    """Copy a segment of the log file to destination.

    Args:
        source_log: Path to the full worker log
        dest_log: Path to write the segment
        start_line: Line number to start from (0-indexed)
        end_line: Line number to end at (exclusive), None for end of file
    """
    if not os.path.exists(source_log):
        print(f"[warning] Source log not found: {source_log}")
        return

    try:
        with open(source_log, "r", encoding="utf-8", errors="ignore") as src:
            lines = src.readlines()

        if end_line is None:
            end_line = len(lines)

        segment = lines[start_line:end_line]

        with open(dest_log, "w", encoding="utf-8") as dst:
            dst.writelines(segment)

        print(f"[segment] Copied lines {start_line}-{end_line} ({len(segment)} lines) to {dest_log}")

    except Exception as e:
        print(f"[error] Failed to segment log: {e}")


def run_rate_test(args, rate: float, output_dir: str, trace: Dict, server_port: int, print_lock: threading.Lock):
    """Run client at a fixed rate against a specific server.

    Args:
        args: Command line arguments
        rate: Request rate
        output_dir: Output directory for this test
        trace: Trace config dict
        server_port: Port of the server to use
        print_lock: Lock for thread-safe printing
    """
    client_log_dir = os.path.join(output_dir, "client_log")

    # Calculate max requests: rate * duration
    max_requests = int(rate * args.duration)

    cmd = [
        sys.executable, "-u", "-m", "auto_rate_slog_rust",
        "--trace", trace["path"],
        "--text-file", os.path.join(SLO_DIR, "text", "enwik8"),
        "--tokenizer", "meta-llama/Llama-3.1-8B-Instruct",
        "--base-url", f"http://{SERVER_HOST}:{server_port}/v1",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--max-requests", str(max_requests),
        "--single-rate", str(rate),
        "--output-dir", client_log_dir,
    ]

    with print_lock:
        print(f"\n[test:{server_port}] Running rate test: rate={rate}, requests={max_requests}, duration={args.duration}s")
        print(f"[test:{server_port}] Trace: {trace['name']} ({trace['path']})")
        print(f"[test:{server_port}] Output: {client_log_dir}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Capture output to a log file instead of printing (to avoid interleaving)
    client_stdout_log = os.path.join(client_log_dir, "client_stdout.log")
    with open(client_stdout_log, "w") as log_file:
        proc = subprocess.Popen(
            cmd, cwd=SLO_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        proc.wait()

    if proc.returncode != 0:
        with print_lock:
            print(f"[warning:{server_port}] Client exited with code {proc.returncode}")


def save_run_info(output_dir: str, trace: Dict, rate: float, args,
                  start_line: int, end_line: int, server_port: int, gpu_id: int):
    """Save run metadata to JSON file."""
    run_info = {
        "trace": trace["name"],
        "trace_path": trace["path"],
        "rate": rate,
        "duration_seconds": args.duration,
        "max_requests": int(rate * args.duration),
        "timestamp": pst_now().isoformat(),
        "server_port": server_port,
        "gpu_id": gpu_id,
        "log_segment": {
            "start_line": start_line,
            "end_line": end_line,
        }
    }

    info_path = os.path.join(output_dir, "run_info.json")
    with open(info_path, "w") as f:
        json.dump(run_info, f, indent=2)


class ServerLogTracker:
    """Thread-safe tracker for per-server log offsets."""

    def __init__(self, master_worker_log_dir: str):
        self.master_worker_log_dir = master_worker_log_dir
        self._locks = {port: threading.Lock() for port in SERVER_PORTS}
        self._offsets = {port: 0 for port in SERVER_PORTS}

    def get_log_path(self, gpu_id: int, port: int) -> str:
        """Get the master log path for a specific server."""
        return os.path.join(
            self.master_worker_log_dir,
            f"worker_{gpu_id + 1}_gpu{gpu_id}_p{port}.ans"
        )

    def get_and_update_offset(self, gpu_id: int, port: int) -> Tuple[int, str]:
        """Get current offset and log path for a server (thread-safe).

        Returns:
            Tuple of (start_offset, log_path)
        """
        with self._locks[port]:
            log_path = self.get_log_path(gpu_id, port)
            start_offset = self._offsets[port]
            return start_offset, log_path

    def update_offset(self, port: int, new_offset: int):
        """Update the offset for a server after test completes."""
        with self._locks[port]:
            self._offsets[port] = new_offset

    def initialize_offsets(self):
        """Initialize offsets from current log file sizes (after warmup)."""
        for i, port in enumerate(SERVER_PORTS):
            log_path = self.get_log_path(i, port)
            self._offsets[port] = get_log_line_count(log_path)
            print(f"[init] Server GPU{i}/port {port} log offset: {self._offsets[port]}")


def run_single_test(args, trace: Dict, rate: float, gpu_id: int, server_port: int,
                    log_tracker: ServerLogTracker, print_lock: threading.Lock) -> Dict:
    """Run evaluation for a single trace and rate on a specific GPU/server.

    Args:
        args: Command line arguments
        trace: Trace config dict
        rate: Request rate to test
        gpu_id: GPU index (0-7)
        server_port: Port of the server (31001-31008)
        log_tracker: ServerLogTracker for managing per-server log offsets
        print_lock: Lock for thread-safe printing

    Returns:
        Dict with trace, rate, status, output_dir, gpu_id, server_port
    """
    with print_lock:
        print(f"\n{'#'*60}")
        print(f"# GPU{gpu_id}:{server_port} | Trace: {trace['name']} | Rate: {rate} req/s ({int(rate * args.duration)} requests)")
        print(f"{'#'*60}")

    output_dir = create_output_structure(trace["name"], rate, args.output_dir)

    # Get starting offset and log path for this server
    start_line, master_log_path = log_tracker.get_and_update_offset(gpu_id, server_port)

    with print_lock:
        print(f"[init:{server_port}] Output directory: {output_dir}")
        print(f"[init:{server_port}] Log segment starts at line {start_line}")

    try:
        # Run client at this rate against the assigned server
        run_rate_test(args, rate, output_dir, trace, server_port, print_lock)

        # Get current line count (end of this segment)
        end_line = get_log_line_count(master_log_path)
        log_tracker.update_offset(server_port, end_line)

        with print_lock:
            print(f"[segment:{server_port}] Log segment ends at line {end_line}")

        # Segment the log for this rate (copy to rate-specific directory)
        rate_worker_log = os.path.join(
            output_dir, "worker_log",
            f"worker_{gpu_id + 1}_gpu{gpu_id}_p{server_port}.ans"
        )
        segment_log(master_log_path, rate_worker_log, start_line, end_line)

        # Save run info with segment info
        save_run_info(output_dir, trace, rate, args, start_line, end_line, server_port, gpu_id)

        with print_lock:
            print(f"[done:{server_port}] {trace['name']} @ rate {rate} completed. Results: {output_dir}")

        return {
            "trace": trace["name"],
            "rate": rate,
            "status": "completed",
            "output_dir": output_dir,
            "gpu_id": gpu_id,
            "server_port": server_port,
            "log_start_line": start_line,
            "log_end_line": end_line,
        }

    except Exception as e:
        end_line = get_log_line_count(master_log_path)
        log_tracker.update_offset(server_port, end_line)

        with print_lock:
            print(f"[error:{server_port}] {trace['name']} @ rate {rate} failed: {e}")

        return {
            "trace": trace["name"],
            "rate": rate,
            "status": "failed",
            "output_dir": output_dir,
            "gpu_id": gpu_id,
            "server_port": server_port,
            "error": str(e),
            "log_end_line": end_line,
        }


def print_summary(results: List[Dict]):
    """Print summary table of all runs."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # Group by trace
    by_trace = {}
    for r in results:
        trace = r["trace"]
        if trace not in by_trace:
            by_trace[trace] = []
        by_trace[trace].append(r)

    for trace, trace_results in by_trace.items():
        print(f"\n  {trace}:")
        for r in trace_results:
            if r["status"] == "completed":
                status_icon = "OK"
            else:
                status_icon = "FAIL"
            print(f"    [{status_icon}] Rate {r['rate']:.1f}")

    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    total = len(results)
    print(f"\nTotal: {completed} completed, {failed} failed (out of {total})")


def run_parallel_tests(args, test_tasks: List[Tuple[Dict, float]],
                       log_tracker: ServerLogTracker, print_lock: threading.Lock) -> List[Dict]:
    """Run test tasks in parallel across 8 servers.

    Each server (GPU) handles one test at a time. Tests are distributed round-robin.

    Args:
        args: Command line arguments
        test_tasks: List of (trace, rate) tuples to run
        log_tracker: ServerLogTracker for managing per-server log offsets
        print_lock: Lock for thread-safe printing

    Returns:
        List of result dicts from all tests
    """
    results = []

    # Create a queue of (trace, rate, gpu_id, server_port) assignments
    # Each GPU runs tests sequentially, but all 8 GPUs run in parallel
    gpu_task_queues = [[] for _ in range(NUM_SERVERS)]

    # Distribute tasks round-robin across GPUs
    for i, (trace, rate) in enumerate(test_tasks):
        gpu_idx = i % NUM_SERVERS
        gpu_task_queues[gpu_idx].append((trace, rate))

    # Show task distribution
    with print_lock:
        print(f"\n[parallel] Distributing {len(test_tasks)} tests across {NUM_SERVERS} servers")
        for i, queue in enumerate(gpu_task_queues):
            port = SERVER_PORTS[i]
            tasks_desc = ", ".join([f"{t['name']}@{r}" for t, r in queue])
            print(f"  GPU{i}/port {port}: {len(queue)} tests [{tasks_desc}]")

    def run_gpu_queue(gpu_id: int, task_queue: List[Tuple[Dict, float]]) -> List[Dict]:
        """Run all tasks assigned to a single GPU sequentially."""
        server_port = SERVER_PORTS[gpu_id]
        gpu_results = []
        for trace, rate in task_queue:
            result = run_single_test(
                args, trace, rate, gpu_id, server_port,
                log_tracker, print_lock
            )
            gpu_results.append(result)
        return gpu_results

    # Run all GPU queues in parallel
    with ThreadPoolExecutor(max_workers=NUM_SERVERS) as executor:
        futures = []
        for gpu_id, task_queue in enumerate(gpu_task_queues):
            if task_queue:  # Only submit if there are tasks
                future = executor.submit(run_gpu_queue, gpu_id, task_queue)
                futures.append(future)

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                gpu_results = future.result()
                results.extend(gpu_results)
            except Exception as e:
                with print_lock:
                    print(f"[error] GPU queue failed: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run non-linear load evaluation for rate sweep (8-GPU parallel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Trace shortcuts:
  nanoflow    lmsys, sharegpt, splitwise (default)
  uniform     uniform_512_512, uniform_4096_1024
  all         All traces

Examples:
  python run_eval.py
  python run_eval.py --traces sharegpt
  python run_eval.py --traces nanoflow --rates 5,10,15,20,25,30,35,40,45,50 --duration 60
  python run_eval.py --traces lmsys,sharegpt --rates 10,20,30 --duration 30
"""
    )
    parser.add_argument("--traces", default="nanoflow",
                        help="Comma-separated list of traces or shortcuts: nanoflow, uniform, all (default: nanoflow)")
    parser.add_argument("--rates", default=None,
                        help="Comma-separated list of rates to test (default: from config.json)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Duration in seconds for each rate test (default: from config.json)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Base output directory")
    parser.add_argument("--server-timeout", type=int, default=600,
                        help="Timeout for server startup in seconds")
    parser.add_argument("--skip-server", action="store_true",
                        help="Skip server launch (assume it's already running)")
    args = parser.parse_args()

    # Load config
    config = load_config()

    # Parse traces from args
    trace_args = [t.strip() for t in args.traces.split(",") if t.strip()]
    traces = expand_traces(trace_args, config)

    if not traces:
        print("[error] No valid traces specified")
        return 1

    # Parse rates from args or config
    if args.rates:
        rates = [float(r.strip()) for r in args.rates.split(",") if r.strip()]
    else:
        rates = config.get("rates", [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    # Get duration from args or config
    if args.duration is None:
        args.duration = config.get("duration_seconds", 60)

    # Ensure output directory exists and set up master log
    os.makedirs(args.output_dir, exist_ok=True)
    master_log_path = os.path.join(args.output_dir, "runner.log")
    master_tee = TeeOutput(master_log_path)
    print(f"[init] Master log: {master_log_path}")

    print(f"[init] Parallel mode: {NUM_SERVERS} GPUs on ports {SERVER_PORTS[0]}-{SERVER_PORTS[-1]}")
    print(f"[init] Traces ({len(traces)}): {[t['name'] for t in traces]}")
    print(f"[init] Rates ({len(rates)}): {rates}")
    print(f"[init] Duration per rate: {args.duration}s")
    print(f"[init] Total combinations: {len(traces) * len(rates)}")
    print(f"[init] Output directory: {args.output_dir}")

    # Create top-level worker_log directory for full server logs
    master_worker_log_dir = os.path.join(args.output_dir, "worker_log")
    os.makedirs(master_worker_log_dir, exist_ok=True)

    # Create log tracker for all 8 servers
    log_tracker = ServerLogTracker(master_worker_log_dir)
    print_lock = threading.Lock()

    server_proc = None
    results = []

    try:
        if not args.skip_server:
            # Cleanup any existing processes
            cleanup_processes()

            # Launch all 8 servers with top-level worker_log directory
            server_proc = launch_server(master_worker_log_dir)
            wait_for_server(timeout=args.server_timeout)

            # Initialize log offsets after warmup
            time.sleep(2)  # Let warmup logs settle
            log_tracker.initialize_offsets()

        # Build list of all test tasks (trace, rate)
        test_tasks = []
        for trace in traces:
            for rate in rates:
                test_tasks.append((trace, rate))

        print(f"\n[parallel] Starting {len(test_tasks)} tests across {NUM_SERVERS} GPUs...")

        # Run tests in parallel
        results = run_parallel_tests(args, test_tasks, log_tracker, print_lock)

    except KeyboardInterrupt:
        print("\n[interrupt] Stopping evaluation...")

    except LaunchError as e:
        print(f"\n[error] Launch failed: {e}")

    finally:
        if not args.skip_server:
            print("\n[cleanup] Shutting down servers...")
            cleanup_processes()

    # Sort results by trace then rate for consistent ordering
    results.sort(key=lambda r: (r.get("trace", ""), r.get("rate", 0)))

    # Print summary
    if results:
        print_summary(results)

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "num_servers": NUM_SERVERS,
            "server_ports": SERVER_PORTS,
            "traces": [t["name"] for t in traces],
            "rates": rates,
            "duration_seconds": args.duration,
            "results": results,
            "timestamp": pst_now().isoformat(),
        }, f, indent=2)
    print(f"\n[done] Summary saved to: {summary_path}")

    # Close master log
    master_tee.close()

    # Return non-zero if any failed
    failed = sum(1 for r in results if r["status"] == "failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
