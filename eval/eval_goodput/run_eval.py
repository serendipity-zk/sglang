#!/usr/bin/env python3
"""
Goodput Evaluation Runner

Runs one method and one trace, managing server/router lifecycle with retry logic.

Usage:
    python run_eval.py --method Niyama --trace sharegpt --output-dir ./results
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional


class TeeOutput:
    """Tee stdout/stderr to both console and a log file."""

    def __init__(self, log_path: str):
        self.log_file = open(log_path, "w", buffering=1)  # Line buffered
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
EVAL_UTILS_DIR = os.path.join(EVAL_DIR, "eval_utils")
SLO_DIR = os.path.join(os.path.dirname(EVAL_DIR), "slo")

# Default config
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
ROUTER_PORT = 40010
SERVER_PORTS = [31001, 31002, 31003, 31004, 31005, 31006, 31007, 31008]

# For PD disaggregation, servers bind to different loopback IPs
# Port 3100X binds to 127.0.0.X
def get_server_url(port: int, method_path: str = "") -> str:
    """Get the health check URL for a server port."""
    if "sglang_pd" in method_path.lower():
        # PD servers use specific loopback IPs: 127.0.0.X for port 3100X
        idx = port - 31000
        return f"http://127.0.0.{idx}:{port}/health"
    else:
        # Other methods bind to 0.0.0.0, reachable via 127.0.0.1
        return f"http://127.0.0.1:{port}/health"


class LaunchError(Exception):
    """Exception raised when server/router launch fails."""
    pass


def pst_now() -> datetime:
    """Return the current time in the America/Los_Angeles timezone."""
    try:
        tz = ZoneInfo("America/Los_Angeles") if ZoneInfo else timezone(timedelta(hours=-8))
    except Exception:
        tz = timezone(timedelta(hours=-8))
    return datetime.now(tz)


def load_method_config(method_name: str) -> Dict:
    """Load method configuration from method.json."""
    method_json_path = os.path.join(EVAL_UTILS_DIR, "method.json")
    with open(method_json_path, "r") as f:
        config = json.load(f)

    # Search in both baseline and polyserve categories
    for category in ["baseline", "polyserve"]:
        if category in config and method_name in config[category]:
            return {
                "name": method_name,
                "path": config[category][method_name],
                "category": category,
            }

    available = []
    for category in ["baseline", "polyserve"]:
        if category in config:
            available.extend(config[category].keys())
    raise ValueError(f"Method '{method_name}' not found. Available: {available}")


def load_trace_config(trace_name: str) -> Dict:
    """Load trace configuration from trace.json."""
    trace_json_path = os.path.join(EVAL_UTILS_DIR, "trace.json")
    with open(trace_json_path, "r") as f:
        config = json.load(f)

    for trace in config.get("traces", []):
        if trace["name"] == trace_name:
            return trace

    available = [t["name"] for t in config.get("traces", [])]
    raise ValueError(f"Trace '{trace_name}' not found. Available: {available}")


def expand_methods(method_args: List[str]) -> List[str]:
    """Expand method shortcuts to actual method names."""
    method_json_path = os.path.join(EVAL_UTILS_DIR, "method.json")
    with open(method_json_path, "r") as f:
        config = json.load(f)

    all_baseline = list(config.get("baseline", {}).keys())
    all_polyserve = list(config.get("polyserve", {}).keys())

    expanded = []
    for m in method_args:
        if m == "baseline":
            expanded.extend(all_baseline)
        elif m == "ours":
            expanded.extend(all_polyserve)
        elif m == "all":
            expanded.extend(all_baseline + all_polyserve)
        else:
            expanded.append(m)
    return list(dict.fromkeys(expanded))  # Remove duplicates, preserve order


def expand_traces(trace_args: List[str]) -> List[str]:
    """Expand trace shortcuts to actual trace names."""
    trace_json_path = os.path.join(EVAL_UTILS_DIR, "trace.json")
    with open(trace_json_path, "r") as f:
        config = json.load(f)

    all_traces = [t["name"] for t in config.get("traces", [])]
    uniform_traces = [t for t in all_traces if "uniform" in t]
    nanoflow_traces = [t for t in all_traces if "uniform" not in t]

    expanded = []
    for t in trace_args:
        if t == "all":
            expanded.extend(all_traces)
        elif t == "uniform":
            expanded.extend(uniform_traces)
        elif t == "nanoflow":
            expanded.extend(nanoflow_traces)
        else:
            expanded.append(t)
    return list(dict.fromkeys(expanded))  # Remove duplicates, preserve order


def create_output_structure(trace_name: str, method_name: str, base_dir: str) -> str:
    """Create output directory structure: trace/method/."""
    output_dir = os.path.join(base_dir, trace_name, method_name)

    # Create all subdirectories
    subdirs = ["worker_log", "pred_log", "router_log", "client_log"]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    return output_dir


def is_completed(output_dir: str) -> bool:
    """Check if this config has already been completed."""
    complete_file = os.path.join(output_dir, "complete.txt")
    return os.path.exists(complete_file)


def write_complete_marker(output_dir: str):
    """Write complete.txt marker file."""
    complete_file = os.path.join(output_dir, "complete.txt")
    with open(complete_file, "w") as f:
        f.write(f"Completed at: {pst_now().isoformat()}\n")
    print(f"[done] Wrote completion marker: {complete_file}")


def cleanup_processes():
    """Kill all sglang-related processes."""
    print("[cleanup] Killing existing sglang processes...")
    subprocess.run(["pkill", "-f", "-9", "sglang"], check=False, capture_output=True)
    subprocess.run(["pkill", "-f", "-9", "sglang_router"], check=False, capture_output=True)
    time.sleep(2)


def launch_router(method_path: str, output_dir: str) -> subprocess.Popen:
    """Launch router from method's launch_router.sh."""
    script_path = os.path.join(EVAL_UTILS_DIR, method_path, "launch_router.sh")
    if not os.path.exists(script_path):
        raise LaunchError(f"Router script not found: {script_path}")

    router_log_dir = os.path.join(output_dir, "router_log")
    log_path = os.path.join(router_log_dir, "router_stdout.log")

    # Set environment variables for log directories
    env = os.environ.copy()
    env["EVAL_LOG_DIR"] = router_log_dir

    print(f"[launch] Starting router from {script_path}")
    print(f"[launch] Router log: {log_path}")

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            ["bash", script_path],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
    return proc


def launch_server(method_path: str, output_dir: str) -> subprocess.Popen:
    """Launch server from method's launch_server.sh."""
    script_path = os.path.join(EVAL_UTILS_DIR, method_path, "launch_server.sh")
    if not os.path.exists(script_path):
        raise LaunchError(f"Server script not found: {script_path}")

    worker_log_dir = os.path.join(output_dir, "worker_log")
    pred_log_dir = os.path.join(output_dir, "pred_log")
    log_path = os.path.join(worker_log_dir, "server_stdout.log")

    # Set environment variables for log directories
    env = os.environ.copy()
    env["EVAL_LOG_DIR"] = worker_log_dir
    env["EVAL_PRED_LOG_DIR"] = pred_log_dir

    print(f"[launch] Starting server from {script_path}")
    print(f"[launch] Server log: {log_path}")
    print(f"[launch] Worker logs: {worker_log_dir}")
    print(f"[launch] Predictor logs: {pred_log_dir}")

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            ["bash", script_path],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
    return proc


def wait_for_router(timeout: int = 60) -> bool:
    """Wait for router to respond to health check."""
    router_url = f"http://0.0.0.0:{ROUTER_PORT}/health"
    deadline = time.time() + timeout
    print(f"[health] Waiting for router at {router_url}...")

    while time.time() < deadline:
        try:
            resp = requests.get(router_url, timeout=5)
            if resp.status_code == 200:
                print("[health] Router is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    raise LaunchError(f"Router failed to start within {timeout}s")


def wait_for_servers(timeout: int = 600, method_path: str = "") -> bool:
    """Wait for all worker servers to respond."""
    deadline = time.time() + timeout
    print(f"[health] Waiting for {len(SERVER_PORTS)} servers...")

    healthy = 0
    while time.time() < deadline:
        healthy = 0
        for port in SERVER_PORTS:
            try:
                url = get_server_url(port, method_path)
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    healthy += 1
            except requests.exceptions.RequestException:
                pass

        elapsed = int(time.time() - (deadline - timeout))
        print(f"[health] Healthy servers: {healthy}/{len(SERVER_PORTS)} (elapsed: {elapsed}s)")

        if healthy == len(SERVER_PORTS):
            print("[health] All servers are ready!")
            return True
        time.sleep(10)

    raise LaunchError(f"Only {healthy}/{len(SERVER_PORTS)} servers healthy within {timeout}s")


def check_server_log_for_errors(output_dir: str) -> bool:
    """Check if server log contains error indicators."""
    log_path = os.path.join(output_dir, "worker_log", "server_stdout.log")
    if not os.path.exists(log_path):
        return False
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if "错误" in content:
                print(f"[error] Found '错误' in server log")
                return True
    except Exception as e:
        print(f"[warning] Could not read server log: {e}")
    return False


def health_check_all(method_path: str = "") -> bool:
    """Final health check for router and all servers."""
    # Check router
    try:
        resp = requests.get(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=5)
        if resp.status_code != 200:
            raise LaunchError("Router health check failed")
    except requests.exceptions.RequestException as e:
        raise LaunchError(f"Router health check failed: {e}")

    # Check all servers
    for port in SERVER_PORTS:
        try:
            url = get_server_url(port, method_path)
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                raise LaunchError(f"Server {port} health check failed")
        except requests.exceptions.RequestException as e:
            raise LaunchError(f"Server {port} health check failed: {e}")

    print("[health] All components passed health check!")
    return True


def run_slo_test(args, trace_config: Dict, output_dir: str):
    """Run auto_rate_slog_rust test."""
    client_log_dir = os.path.join(output_dir, "client_log")

    cmd = [
        sys.executable, "-u", "-m", "auto_rate_slog_rust",
        "--trace", trace_config["path"],
        "--text-file", os.path.join(SLO_DIR, "text", "enwik8"),
        "--tokenizer", "meta-llama/Llama-3.1-8B-Instruct",
        "--base-url", f"http://0.0.0.0:{ROUTER_PORT}/v1",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--max-requests", str(trace_config["length"]),
        "--output-dir", client_log_dir,
    ]

    if args.single_rate is not None:
        # Single rate mode - run at fixed rate
        cmd.extend(["--single-rate", str(args.single_rate)])
        print(f"[test] Running SLO test (single rate mode)...")
        print(f"[test] Fixed rate: {args.single_rate}")
    elif args.use_range and "ranges" in trace_config:
        # Range mode - use predefined rate list from trace.json
        start, end, step = trace_config["ranges"]
        rates = list(range(int(start), int(end) + 1, int(step)))
        rate_list_str = ",".join(str(r) for r in rates)
        cmd.extend([
            "--rate-list", rate_list_str,
            "--target-attainment", str(args.target_attainment),
        ])
        print(f"[test] Running SLO test (range mode)...")
        print(f"[test] Rate list: {rates}")
    else:
        # Adaptive search mode
        cmd.extend([
            "--start-rate", str(trace_config["rate_est"]),
            "--target-attainment", str(args.target_attainment),
        ])
        print(f"[test] Running SLO test (adaptive mode)...")
        print(f"[test] Start rate: {trace_config['rate_est']}")

    print(f"[test] Trace: {trace_config['name']} ({trace_config['path']})")
    print(f"[test] Max requests: {trace_config['length']}")
    print(f"[test] Output: {client_log_dir}")

    # Run with real-time output through tee
    # Use PYTHONUNBUFFERED to force unbuffered output from subprocess
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd, cwd=SLO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
        env=env,
    )
    for line in proc.stdout:
        print(line, end='', flush=True)  # Goes through tee
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def save_run_info(output_dir: str, method_name: str, method_config: Dict,
                  trace_config: Dict, args, status: str, attempts: int,
                  error_msg: Optional[str] = None):
    """Save run metadata to JSON file."""
    run_info = {
        "method": method_name,
        "method_path": method_config["path"],
        "trace": trace_config["name"],
        "trace_path": trace_config["path"],
        "timestamp": pst_now().isoformat(),
        "max_requests": trace_config["length"],
        "status": status,
        "attempts": attempts,
        "max_retries": args.max_retries,
        "retry_wait": args.retry_wait,
    }
    if args.single_rate is not None:
        run_info["single_rate"] = args.single_rate
    else:
        run_info["start_rate"] = trace_config["rate_est"]
        run_info["target_attainment"] = args.target_attainment
    if error_msg:
        run_info["error"] = error_msg

    info_path = os.path.join(output_dir, "run_info.json")
    with open(info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"[info] Run info saved to {info_path}")


def run_single_eval(args, method_name: str, trace_name: str) -> Dict:
    """Run a single evaluation for one method and one trace."""
    print(f"\n{'#'*60}")
    print(f"# Evaluating: {method_name} x {trace_name}")
    print(f"{'#'*60}\n")

    # Load configs
    print(f"[init] Loading method config for '{method_name}'...")
    method_config = load_method_config(method_name)
    print(f"[init] Method path: {method_config['path']}")

    print(f"[init] Loading trace config for '{trace_name}'...")
    trace_config = load_trace_config(trace_name)
    print(f"[init] Trace path: {trace_config['path']}")

    # Create output directory
    output_dir = create_output_structure(trace_name, method_name, args.output_dir)
    print(f"[init] Output directory: {output_dir}")

    # Check if already completed
    if is_completed(output_dir) and not args.force:
        print(f"[skip] Already completed: {output_dir}")
        print(f"[skip] Use --force to re-run")
        return {
            "method": method_name,
            "trace": trace_name,
            "status": "skipped",
            "output_dir": output_dir,
        }

    # Set up tee to capture runner output
    runner_log_path = os.path.join(output_dir, "runner.log")
    tee = TeeOutput(runner_log_path)
    print(f"[init] Runner log: {runner_log_path}")

    status = "failed"
    error_msg = None
    attempt = 0

    for attempt in range(1, args.max_retries + 1):
        try:
            print(f"\n{'='*60}")
            print(f"[attempt {attempt}/{args.max_retries}] Starting...")
            print(f"{'='*60}\n")

            # Cleanup any existing processes
            cleanup_processes()

            # Launch router
            print("[launch] Starting router...")
            launch_router(method_config["path"], output_dir)
            wait_for_router(timeout=60)

            # Launch server
            print("[launch] Starting server...")
            launch_server(method_config["path"], output_dir)

            # Wait for servers with 60s timeout
            wait_for_servers(timeout=60, method_path=method_config["path"])

            # Check for errors in log
            if check_server_log_for_errors(output_dir):
                raise LaunchError("Server log contains errors")

            # Final health check
            health_check_all(method_path=method_config["path"])

            # Run SLO test
            run_slo_test(args, trace_config, output_dir)

            # Success!
            status = "completed"
            print("\n[success] Evaluation completed successfully!")
            break

        except LaunchError as e:
            error_msg = str(e)
            print(f"\n[error] Attempt {attempt} failed: {e}")

            if attempt < args.max_retries:
                cleanup_processes()
                print(f"[retry] Waiting {args.retry_wait}s before retry...")
                time.sleep(args.retry_wait)
            else:
                print(f"[error] Failed after {args.max_retries} attempts")

        except KeyboardInterrupt:
            print("\n[interrupt] Received Ctrl+C, cleaning up...")
            error_msg = "Interrupted by user"
            break

        except Exception as e:
            error_msg = str(e)
            print(f"\n[error] Unexpected error: {e}")
            break

    # Cleanup
    print("\n[cleanup] Shutting down processes...")
    cleanup_processes()

    # Save run info
    save_run_info(output_dir, method_name, method_config, trace_config, args,
                  status, attempt, error_msg)

    if status == "completed":
        write_complete_marker(output_dir)
        print(f"\n[done] Results saved to: {output_dir}")

    tee.close()

    return {
        "method": method_name,
        "trace": trace_name,
        "status": status,
        "output_dir": output_dir,
    }


def print_summary(results: List[Dict]):
    """Print summary table of all runs."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for r in results:
        if r["status"] == "completed":
            status_icon = "OK"
        elif r["status"] == "skipped":
            status_icon = "SKIP"
        else:
            status_icon = "FAIL"
        print(f"  [{status_icon}] {r['method']} x {r['trace']}")

    completed = sum(1 for r in results if r["status"] == "completed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    total = len(results)
    print(f"\nTotal: {completed} completed, {skipped} skipped, {failed} failed (out of {total})")


def main():
    parser = argparse.ArgumentParser(
        description="Run goodput evaluation for methods and traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Shortcuts:
  --method baseline    All baseline methods
  --method ours        PolyServe methods
  --method all         All methods

  --trace all          All traces
  --trace uniform      Traces with 'uniform' in name
  --trace nanoflow     Traces without 'uniform' (lmsys, sharegpt, splitwise)

Examples:
  python run_eval.py --method Niyama --trace sharegpt
  python run_eval.py --method baseline --trace all
  python run_eval.py --method all --trace nanoflow
"""
    )
    parser.add_argument("--method", required=True, nargs="+",
                        help="Method name(s) or shortcuts: baseline, ours, all")
    parser.add_argument("--trace", required=True, nargs="+",
                        help="Trace name(s) or shortcuts: all, uniform, nanoflow")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Base output directory")
    parser.add_argument("--max-retries", type=int, default=3, help="Max server launch retries")
    parser.add_argument("--retry-wait", type=int, default=30, help="Wait time between retries (seconds)")
    parser.add_argument("--target-attainment", type=float, default=0.95, help="Target SLO attainment")
    parser.add_argument("--single-rate", type=float, default=None,
                        help="Run at a fixed rate (skips adaptive search)")
    parser.add_argument("--use-range", action="store_true",
                        help="Use rate ranges from trace.json instead of adaptive search")
    parser.add_argument("--force", action="store_true",
                        help="Run even if already completed (ignore complete.txt)")
    args = parser.parse_args()

    # Ensure output directory exists and set up master log
    os.makedirs(args.output_dir, exist_ok=True)
    master_log_path = os.path.join(args.output_dir, "runner.log")
    master_tee = TeeOutput(master_log_path)
    print(f"[init] Master log: {master_log_path}")

    # Expand shortcuts
    methods = expand_methods(args.method)
    traces = expand_traces(args.trace)

    print(f"[init] Methods ({len(methods)}): {methods}")
    print(f"[init] Traces ({len(traces)}): {traces}")
    print(f"[init] Total combinations: {len(methods) * len(traces)}")

    # Run all combinations
    results = []
    try:
        for method_name in methods:
            for trace_name in traces:
                result = run_single_eval(args, method_name, trace_name)
                results.append(result)
    except KeyboardInterrupt:
        print("\n[interrupt] Stopping evaluation loop...")

    # Print summary
    if results:
        print_summary(results)

    # Close master log
    master_tee.close()

    # Return non-zero if any failed
    failed = sum(1 for r in results if r["status"] == "failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
