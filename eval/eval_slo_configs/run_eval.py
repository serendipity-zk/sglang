#!/usr/bin/env python3
"""
SLO Config Evaluation Runner

Runs different scheduler configs against time-shifted traces.
Rate is embedded in trace (arrival times), so client uses --rate 1.

Usage:
    python run_eval.py --config ttft_promote --trace sharegpt --rate 250
    python run_eval.py --config all --trace all --rate all
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

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
SLO_DIR = "/sgl-workspace/sglang/slo"
TRACE_BASE_DIR = "/sgl-workspace/sglang/SLO-CSim/trace/arxiv/time_shift_traces"
RUST_BINARY = "/sgl-workspace/sglang/slo/rust_client/target/release/slo_runner"

# Default config
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
ROUTER_PORT = 40010
SERVER_PORTS = [31001, 31002, 31003, 31004, 31005, 31006, 31007, 31008]

# Scheduler configs
CONFIGS = {
    "no_autoscaling": "/sgl-workspace/sglang/slo/scheduler_config_slo_aware_no_autoscaling.json",
    "ttft": "/sgl-workspace/sglang/slo/scheduler_config_slo_aware_ttft.json",
    "ttft_steal": "/sgl-workspace/sglang/slo/scheduler_config_slo_aware_ttft_steal.json",
    "ttft_promote": "/sgl-workspace/sglang/slo/scheduler_config_slo_aware_ttft_promote.json",
}

# Trace configurations
TRACES = {
    "lmsys": {
        "rates": [150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
        "base_path": os.path.join(TRACE_BASE_DIR, "lmsys"),
        "pattern": "time_shift_lmsys_rate_{rate}.csv",
    },
    "sharegpt": {
        "rates": [100, 130, 160, 190, 220, 250, 280, 310, 340, 370, 400],
        "base_path": os.path.join(TRACE_BASE_DIR, "sharegpt"),
        "pattern": "time_shift_sharegpt_rate_{rate}.csv",
    },
    "splitwise": {
        "rates": [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
        "base_path": os.path.join(TRACE_BASE_DIR, "splitwise"),
        "pattern": "time_shift_splitwise_rate_{rate}.csv",
    },
}

# Duration in seconds for computing max_requests
TEST_DURATION_SECONDS = 120


def get_max_requests(rate: int) -> int:
    """Compute max_requests as rate * TEST_DURATION_SECONDS."""
    return rate * TEST_DURATION_SECONDS


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
    """Exception raised when server/router launch fails."""
    pass


def pst_now() -> datetime:
    """Return the current time in the America/Los_Angeles timezone."""
    try:
        tz = ZoneInfo("America/Los_Angeles") if ZoneInfo else timezone(timedelta(hours=-8))
    except Exception:
        tz = timezone(timedelta(hours=-8))
    return datetime.now(tz)


def get_trace_path(trace_name: str, rate: int) -> str:
    """Construct trace file path from trace name and rate."""
    trace_config = TRACES[trace_name]
    filename = trace_config["pattern"].format(rate=rate)
    return os.path.join(trace_config["base_path"], filename)


def expand_configs(config_args: List[str]) -> List[str]:
    """Expand config shortcuts."""
    if "all" in config_args:
        return list(CONFIGS.keys())
    return config_args


def expand_traces(trace_args: List[str]) -> List[str]:
    """Expand trace shortcuts."""
    if "all" in trace_args:
        return list(TRACES.keys())
    return trace_args


def expand_rates(rate_args: List[str], trace_name: str) -> List[int]:
    """Expand rate shortcuts for a given trace."""
    if "all" in rate_args:
        return TRACES[trace_name]["rates"]
    return [int(r) for r in rate_args]


def create_output_structure(trace_name: str, rate: int, config_name: str, base_dir: str) -> str:
    """Create output directory structure."""
    output_dir = os.path.join(base_dir, f"{trace_name}_rate{rate}", config_name)

    subdirs = ["worker_log", "router_log", "client_log"]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    return output_dir


def is_completed(output_dir: str) -> bool:
    """Check if this config has already been completed."""
    return os.path.exists(os.path.join(output_dir, "complete.txt"))


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


def launch_server(output_dir: str) -> subprocess.Popen:
    """Launch server."""
    script_path = os.path.join(SCRIPT_DIR, "launch_server.sh")
    worker_log_dir = os.path.join(output_dir, "worker_log")
    log_path = os.path.join(worker_log_dir, "server_stdout.log")

    env = os.environ.copy()
    env["EVAL_LOG_DIR"] = worker_log_dir
    env["EVAL_PRED_LOG_DIR"] = os.path.join(output_dir, "pred_log")

    print(f"[launch] Starting server from {script_path}")
    print(f"[launch] Server log: {log_path}")

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            ["bash", script_path],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
    return proc


def launch_router(config_path: str, output_dir: str) -> subprocess.Popen:
    """Launch router with specified config."""
    script_path = os.path.join(SCRIPT_DIR, "launch_router.sh")
    router_log_dir = os.path.join(output_dir, "router_log")
    log_path = os.path.join(router_log_dir, "router_stdout.log")

    env = os.environ.copy()
    env["EVAL_LOG_DIR"] = router_log_dir

    print(f"[launch] Starting router with config: {config_path}")
    print(f"[launch] Router log: {log_path}")

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            ["bash", script_path, config_path],
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


def wait_for_servers(timeout: int = 600) -> bool:
    """Wait for all worker servers to respond."""
    deadline = time.time() + timeout
    print(f"[health] Waiting for {len(SERVER_PORTS)} servers...")

    healthy = 0
    while time.time() < deadline:
        healthy = 0
        for port in SERVER_PORTS:
            try:
                url = f"http://127.0.0.1:{port}/health"
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


def health_check_all() -> bool:
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
            url = f"http://127.0.0.1:{port}/health"
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                raise LaunchError(f"Server {port} health check failed")
        except requests.exceptions.RequestException as e:
            raise LaunchError(f"Server {port} health check failed: {e}")

    print("[health] All components passed health check!")
    return True


def run_client(trace_path: str, output_dir: str, max_requests: int):
    """Run Rust client with rate=1 (arrival times in trace)."""
    client_log_dir = os.path.join(output_dir, "client_log")

    # Build rust client if needed
    rust_client_dir = "/sgl-workspace/sglang/slo/rust_client"
    if not os.path.exists(RUST_BINARY):
        print("[build] Building Rust client...")
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=rust_client_dir,
            check=True,
        )

    cmd = [
        RUST_BINARY,
        "--trace", trace_path,
        "--text-file", os.path.join(SLO_DIR, "text", "enwik8"),
        "--tokenizer", "meta-llama/Llama-3.1-8B-Instruct",
        "--base-url", f"http://0.0.0.0:{ROUTER_PORT}/v1",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--rate", "1",  # Rate is embedded in trace
        "--max-requests", str(max_requests),
        "--log-path", os.path.join(client_log_dir, "rust_client_output.jsonl"),
        "--elapsed-dump-path", os.path.join(client_log_dir, "rust_elapsed_timelines.pkl"),
        "--ans-log-path", os.path.join(client_log_dir, "rust_client.ans"),
        "--slo-use-detokenize-time",
    ]

    print(f"[client] Running: {' '.join(cmd)}")
    print(f"[client] Trace: {trace_path}")
    print(f"[client] Max requests: {max_requests}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def save_run_info(output_dir: str, config_name: str, trace_name: str, rate: int,
                  status: str, attempts: int, error_msg: Optional[str] = None):
    """Save run metadata to JSON file."""
    run_info = {
        "config": config_name,
        "config_path": CONFIGS[config_name],
        "trace": trace_name,
        "rate": rate,
        "trace_path": get_trace_path(trace_name, rate),
        "timestamp": pst_now().isoformat(),
        "max_requests": get_max_requests(rate),
        "status": status,
        "attempts": attempts,
    }
    if error_msg:
        run_info["error"] = error_msg

    info_path = os.path.join(output_dir, "run_info.json")
    with open(info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"[info] Run info saved to {info_path}")


def run_single_eval(args, config_name: str, trace_name: str, rate: int) -> Dict:
    """Run a single evaluation for one config, trace, and rate."""
    print(f"\n{'#'*60}")
    print(f"# Config: {config_name} | Trace: {trace_name} | Rate: {rate}")
    print(f"{'#'*60}\n")

    # Validate inputs
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    if trace_name not in TRACES:
        raise ValueError(f"Unknown trace: {trace_name}. Available: {list(TRACES.keys())}")
    if rate not in TRACES[trace_name]["rates"]:
        raise ValueError(f"Invalid rate {rate} for {trace_name}. Available: {TRACES[trace_name]['rates']}")

    # Create output directory
    output_dir = create_output_structure(trace_name, rate, config_name, args.output_dir)
    print(f"[init] Output directory: {output_dir}")

    # Check if already completed
    if is_completed(output_dir) and not args.force:
        print(f"[skip] Already completed: {output_dir}")
        print(f"[skip] Use --force to re-run")
        return {
            "config": config_name,
            "trace": trace_name,
            "rate": rate,
            "status": "skipped",
            "output_dir": output_dir,
        }

    # Set up tee
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

            # Cleanup
            cleanup_processes()

            # Launch router with specific config
            config_path = CONFIGS[config_name]
            print(f"[launch] Starting router with config: {config_name}")
            launch_router(config_path, output_dir)
            wait_for_router(timeout=60)

            # Launch server
            print("[launch] Starting server...")
            launch_server(output_dir)
            wait_for_servers(timeout=60)

            # Check for errors in log
            if check_server_log_for_errors(output_dir):
                raise LaunchError("Server log contains errors")

            # Final health check
            health_check_all()

            # Run client
            trace_path = get_trace_path(trace_name, rate)
            max_requests = get_max_requests(rate)
            run_client(trace_path, output_dir, max_requests)

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
    save_run_info(output_dir, config_name, trace_name, rate, status, attempt, error_msg)

    if status == "completed":
        write_complete_marker(output_dir)
        print(f"\n[done] Results saved to: {output_dir}")

    tee.close()

    return {
        "config": config_name,
        "trace": trace_name,
        "rate": rate,
        "status": status,
        "output_dir": output_dir,
    }


def print_summary(results: List[Dict]):
    """Print summary table of all runs."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    for r in results:
        if r["status"] == "completed":
            status_icon = "OK"
        elif r["status"] == "skipped":
            status_icon = "SKIP"
        else:
            status_icon = "FAIL"
        print(f"  [{status_icon}] {r['config']} x {r['trace']} x rate{r['rate']}")

    completed = sum(1 for r in results if r["status"] == "completed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    total = len(results)
    print(f"\nTotal: {completed} completed, {skipped} skipped, {failed} failed (out of {total})")


def main():
    parser = argparse.ArgumentParser(
        description="Run SLO config evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_eval.py --config ttft_promote --trace sharegpt --rate 250
  python run_eval.py --config all --trace sharegpt --rate 250
  python run_eval.py --config ttft_promote --trace all --rate all
  python run_eval.py --config all --trace all --rate all

Available configs: no_autoscaling, ttft, ttft_steal, ttft_promote
Available traces: lmsys, sharegpt, splitwise
"""
    )
    parser.add_argument("--config", required=True, nargs="+",
                        help="Config name(s) or 'all'")
    parser.add_argument("--trace", required=True, nargs="+",
                        help="Trace name(s) or 'all'")
    parser.add_argument("--rate", required=True, nargs="+",
                        help="Rate(s) or 'all'")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Base output directory")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max server launch retries")
    parser.add_argument("--retry-wait", type=int, default=30,
                        help="Wait time between retries (seconds)")
    parser.add_argument("--force", action="store_true",
                        help="Run even if already completed")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Expand shortcuts
    configs = expand_configs(args.config)
    traces = expand_traces(args.trace)

    print(f"[init] Configs: {configs}")
    print(f"[init] Traces: {traces}")

    # Run all combinations
    results = []
    try:
        for config_name in configs:
            for trace_name in traces:
                rates = expand_rates(args.rate, trace_name)
                print(f"[init] Rates for {trace_name}: {rates}")

                for rate in rates:
                    result = run_single_eval(args, config_name, trace_name, rate)
                    results.append(result)
    except KeyboardInterrupt:
        print("\n[interrupt] Stopping evaluation loop...")

    # Print summary
    if results:
        print_summary(results)

    # Return non-zero if any failed
    failed = sum(1 for r in results if r["status"] == "failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
