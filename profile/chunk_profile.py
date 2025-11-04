#!/usr/bin/env python3
"""
Profile chunked prefill behaviour using iteration-metrics logs.

This script launches a local SGLang server with chunked prefill enabled,
cycles through user-specified prompt/completion length presets, issues
randomized requests, and captures the scheduler iteration metrics that are
emitted via the existing STAT_METRICS logging pipeline.

Outputs for each run include:
  - Raw server log with STDOUT/STDERR merged
  - JSONL dump of every STAT_METRICS event observed
  - Per-length summaries (average iteration time, throughput, etc.)
  - Sample request/response payloads
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import socket
import string
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, Iterable, List, Sequence, Tuple

import requests

STAT_PREFIX = "STAT_METRICS:"
DEFAULT_LENGTH_CONFIGS = ("512:64", "2048:128")
DEFAULT_REQUESTS_PER_LENGTH = 16
DEFAULT_OUTPUT_ROOT = Path("./profile_runs")
DEFAULT_POST_LENGTH_WAIT = 2.0
DEFAULT_IDLE_WAIT_TIMEOUT = 30.0
DEFAULT_IDLE_POLL_INTERVAL = 0.2
DEFAULT_MAX_CONCURRENT_REQUESTS = 32
HEALTH_ENDPOINT = "/health_generate"
GENERATE_ENDPOINT = "/generate"
TIMEOUT_FOR_SERVER_START = 600
REQUEST_TIMEOUT = 120


@dataclass(frozen=True)
class LengthPreset:
    prompt_tokens: int
    max_new_tokens: int

    @classmethod
    def parse(cls, token_spec: str) -> "LengthPreset":
        """Parse CLI token specification formatted as <prompt>:<completion>."""
        try:
            prompt, completion = token_spec.split(":")
            prompt_tokens = int(prompt)
            completion_tokens = int(completion)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid length specification '{token_spec}'. Expected <prompt>:<completion>."
            ) from exc

        if prompt_tokens <= 0 or completion_tokens <= 0:
            raise argparse.ArgumentTypeError(
                f"Prompt/completion tokens must be positive in '{token_spec}'."
            )

        return cls(prompt_tokens=prompt_tokens, max_new_tokens=completion_tokens)


class MetricsBuffer:
    """Thread-safe store for STAT_METRICS payloads."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._metrics: List[Dict] = []

    def append(self, metric: Dict) -> None:
        with self._lock:
            self._metrics.append(metric)

    def slice_from(self, start_index: int) -> Tuple[List[Dict], int]:
        with self._lock:
            snapshot = self._metrics[start_index:]
            end_index = len(self._metrics)
        return list(snapshot), end_index

    def snapshot(self) -> List[Dict]:
        with self._lock:
            return list(self._metrics)

    def get_latest(self) -> Dict | None:
        """Get the most recent metric, or None if buffer is empty."""
        with self._lock:
            if not self._metrics:
                return None
            return dict(self._metrics[-1])

    def __len__(self) -> int:
        with self._lock:
            return len(self._metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch SGLang with chunked prefill and capture iteration metrics."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Model identifier or checkpoint path to serve.",
    )
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        required=True,
        help="Chunked prefill token limit passed to the server.",
    )
    parser.add_argument(
        "--length-configs",
        nargs="+",
        type=LengthPreset.parse,
        default=[LengthPreset.parse(spec) for spec in DEFAULT_LENGTH_CONFIGS],
        help="Pairs of <prompt_tokens>:<max_new_tokens> to cycle through.",
    )
    parser.add_argument(
        "--requests-per-length",
        type=int,
        default=DEFAULT_REQUESTS_PER_LENGTH,
        help="Number of requests to issue for each length preset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that will contain logs, metrics, and summaries.",
    )
    parser.add_argument(
        "--profile-name",
        type=str,
        default=None,
        help="Optional logical name for this experiment. Used when naming run folders.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface for the local server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional fixed port for the server. Leave unset to auto-reserve.",
    )
    parser.add_argument(
        "--server-extra-args",
        nargs="*",
        default=(),
        help="Additional arguments forwarded verbatim to sglang.launch_server.",
    )
    parser.add_argument(
        "--enable-mixed-chunk",
        action="store_true",
        help="Toggle --enable-mixed-chunk for the launched server.",
    )
    parser.add_argument(
        "--post-length-wait",
        type=float,
        default=DEFAULT_POST_LENGTH_WAIT,
        help="Wait time (seconds) after finishing a length preset before sampling metrics. Set to 0 to use only --wait-for-idle.",
    )
    parser.add_argument(
        "--wait-for-idle",
        action="store_true",
        help="Wait for server to become idle (running_batch_size <= 1) before next preset.",
    )
    parser.add_argument(
        "--idle-wait-timeout",
        type=float,
        default=DEFAULT_IDLE_WAIT_TIMEOUT,
        help="Maximum time (seconds) to wait for server to become idle.",
    )
    parser.add_argument(
        "--idle-poll-interval",
        type=float,
        default=DEFAULT_IDLE_POLL_INTERVAL,
        help="Polling interval (seconds) when checking for server idle state.",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_REQUESTS,
        help="Maximum number of concurrent requests to send in parallel.",
    )
    parser.add_argument(
        "--iteration-metrics-interval",
        type=int,
        default=1,
        help="Report STAT_METRICS every N scheduler iterations.",
    )
    parser.add_argument(
        "--router-metrics-url",
        type=str,
        default=None,
        help="Optional router metrics URL to forward STAT_METRICS via HTTP.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=REQUEST_TIMEOUT,
        help="HTTP timeout in seconds for /generate requests.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Server log level (forwarded to --log-level).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Stream server logs to stdout for debugging.",
    )
    return parser.parse_args()


def ensure_output_root(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def build_run_directory(
    output_root: Path, chunk_size: int, profile_name: str | None
) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    name = profile_name or "chunk_prefill_profile"
    run_dir = output_root / f"{name}_chunk{chunk_size}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_server_command(args: argparse.Namespace, port: int) -> List[str]:
    command: List[str] = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(port),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--enable-iteration-metrics",
        "--iteration-metrics-interval",
        str(args.iteration_metrics_interval),
        "--log-level",
        args.log_level,
    ]

    if args.enable_mixed_chunk:
        command.append("--enable-mixed-chunk")

    if args.router_metrics_url:
        command.extend(["--router-metrics-url", args.router_metrics_url])

    command.extend(args.server_extra_args)
    return command


def wait_for_server(base_url: str, timeout: float) -> None:
    """Poll the health endpoint until the server is ready or we hit timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(base_url + HEALTH_ENDPOINT, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("Server failed to pass health check within the timeout window.")


def wait_for_idle_server(
    base_url: str,
    timeout: float,
    poll_interval: float,
    debug: bool = False,
) -> bool:
    """
    Wait for the server to become fully idle (no running or waiting requests).

    Polls the /check_idle endpoint to query the scheduler state directly.

    Returns True if server became idle, False if timeout occurred.
    """
    if debug:
        print("[profile] Waiting for server to become idle (no requests in batch or queue)...")

    deadline = time.time() + timeout
    last_status = None

    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/check_idle", timeout=5)
            if response.status_code == 200:
                data = response.json()
                is_idle = data.get("is_idle", False)

                if debug and is_idle != last_status:
                    loads = data.get("loads", [])
                    total_reqs = sum(load.get("num_reqs", 0) for load in loads)
                    total_waiting = sum(load.get("num_waiting_reqs", 0) for load in loads)
                    print(f"[profile] Server status: is_idle={is_idle}, total_reqs={total_reqs}, waiting={total_waiting}")
                    last_status = is_idle

                if is_idle:
                    if debug:
                        print("[profile] Server is now idle.")
                    return True
        except requests.RequestException as e:
            if debug:
                print(f"[profile] Error checking idle status: {e}")
            pass

        time.sleep(poll_interval)

    if debug:
        print(f"[profile] WARNING: Timeout waiting for server to become idle after {timeout}s")
    return False


def random_prompt(prompt_tokens: int, rng: random.Random) -> str:
    """Generate a whitespace-delimited prompt containing the requested number of tokens."""
    vocabulary = string.ascii_lowercase
    tokens: List[str] = []
    for _ in range(prompt_tokens):
        token_length = rng.randint(3, 8)
        token = "".join(rng.choice(vocabulary) for _ in range(token_length))
        tokens.append(token)
    return " ".join(tokens)


def issue_request(
    base_url: str,
    prompt: str,
    max_new_tokens: int,
    timeout: float,
) -> Dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
    }
    response = requests.post(
        base_url + GENERATE_ENDPOINT, json=payload, timeout=timeout
    )
    response.raise_for_status()
    return response.json()


def percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return float(sorted_vals[int(k)])
    lower_val = sorted_vals[lower]
    upper_val = sorted_vals[upper]
    return float(lower_val + (upper_val - lower_val) * (k - lower))


def summarize_metrics(metrics: List[Dict]) -> Dict:
    summary: Dict = {"iterations": len(metrics)}
    if not metrics:
        return summary

    iteration_nums = [m.get("iteration_num") for m in metrics if m.get("iteration_num") is not None]
    times_ms = [float(m.get("iteration_time_ms", 0.0)) for m in metrics]
    token_batch = [float(m.get("token_batch_size", 0.0)) for m in metrics]
    prefill_tokens = [float(m.get("prefill_tokens", 0.0)) for m in metrics]
    decode_tokens = [float(m.get("decode_tokens", 0.0)) for m in metrics]
    kv_usage_pct = [float(m.get("kv_usage_pct", 0.0)) for m in metrics]
    running_batch_size = [float(m.get("running_batch_size", 0.0)) for m in metrics]

    sum_time_ms = sum(times_ms)
    sum_tokens = sum(token_batch)

    summary.update(
        {
            "iteration_range": [
                min(iteration_nums) if iteration_nums else None,
                max(iteration_nums) if iteration_nums else None,
            ],
            "mean_iteration_time_ms": sum_time_ms / len(times_ms) if times_ms else None,
            "median_iteration_time_ms": percentile(times_ms, 50) if times_ms else None,
            "p90_iteration_time_ms": percentile(times_ms, 90) if times_ms else None,
            "mean_token_batch_size": sum_tokens / len(token_batch) if token_batch else None,
            "mean_prefill_tokens": sum(prefill_tokens) / len(prefill_tokens) if prefill_tokens else None,
            "mean_decode_tokens": sum(decode_tokens) / len(decode_tokens) if decode_tokens else None,
            "mean_kv_usage_pct": sum(kv_usage_pct) / len(kv_usage_pct) if kv_usage_pct else None,
            "mean_running_batch_size": sum(running_batch_size) / len(running_batch_size)
            if running_batch_size
            else None,
            "throughput_tokens_per_s": (sum_tokens / (sum_time_ms / 1000.0)) if sum_time_ms > 0 else None,
        }
    )
    return summary


def stream_process_output(
    process: subprocess.Popen,
    log_path: Path,
    metrics_buffer: MetricsBuffer,
    debug: bool,
) -> List[Thread]:
    """
    Start background threads that forward stdout/stderr to a log file while parsing STAT_METRICS lines.
    """
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    def pump(stream, tag: str) -> None:
        try:
            for raw in iter(stream.readline, b""):
                if not raw:
                    break
                text = raw.decode("utf-8", errors="replace")
                formatted = f"[{tag}] {text}"
                log_file.write(formatted)
                if debug:
                    sys.stdout.write(formatted)
                    sys.stdout.flush()

                idx = text.find(STAT_PREFIX)
                if idx >= 0:
                    payload = text[idx + len(STAT_PREFIX) :].strip()
                    try:
                        metric = json.loads(payload)
                        metrics_buffer.append(metric)
                    except json.JSONDecodeError:
                        continue
        finally:
            log_file.flush()

    threads: List[Thread] = []
    for name, stream in (("STDOUT", process.stdout), ("STDERR", process.stderr)):
        if stream is None:
            continue
        thread = Thread(target=pump, args=(stream, name), daemon=True)
        thread.start()
        threads.append(thread)

    return threads


def launch_server(
    command: Sequence[str],
    env: Dict[str, str],
    log_path: Path,
    metrics_buffer: MetricsBuffer,
    debug: bool,
) -> Tuple[subprocess.Popen, List[Thread]]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        preexec_fn=os.setsid,
    )
    threads = stream_process_output(process, log_path, metrics_buffer, debug)
    return process, threads


def terminate_process_tree(process: subprocess.Popen, timeout: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.1)


def reserve_port(host: str) -> Tuple[int, socket.socket]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    # Keep the socket open so the port stays reserved.
    return port, sock


def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    return obj


def write_json(path: Path, content: Dict) -> None:
    with path.open("w", encoding="utf-8") as fout:
        json.dump(_json_safe(content), fout, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(_json_safe(row)))
            fout.write("\n")


def main() -> None:
    args = parse_args()

    if args.chunked_prefill_size <= 0:
        raise ValueError("--chunked-prefill-size must be positive.")
    if args.requests_per_length <= 0:
        raise ValueError("--requests-per-length must be positive.")
    if args.iteration_metrics_interval <= 0:
        raise ValueError("--iteration-metrics-interval must be positive.")

    output_root = ensure_output_root(args.output_root)
    run_dir = build_run_directory(output_root, args.chunked_prefill_size, args.profile_name)
    log_path = run_dir / "server.log"
    metrics_path = run_dir / "metrics.jsonl"

    env = os.environ.copy()
    env.setdefault("SGLANG_LOG_DIR", str(run_dir))

    port = args.port
    if port is None:
        port, port_lock = reserve_port(args.host)
    else:
        port_lock = None

    base_url = f"http://{args.host}:{port}"
    server_command = build_server_command(args, port)

    rng = random.Random()
    metrics_buffer = MetricsBuffer()

    if args.debug:
        print(f"[profile] Run directory: {run_dir}")
        print(f"[profile] Launching server: {' '.join(server_command)}")

    process = None
    threads: List[Thread] = []
    try:
        process, threads = launch_server(
            server_command,
            env=env,
            log_path=log_path,
            metrics_buffer=metrics_buffer,
            debug=args.debug,
        )
        wait_for_server(base_url, timeout=TIMEOUT_FOR_SERVER_START)
        if args.debug:
            print("[profile] Server is ready.")

        current_metric_index = 0
        for preset in args.length_configs:
            label = f"in{preset.prompt_tokens}_out{preset.max_new_tokens}"
            if args.debug:
                print(f"[profile] Running preset {label}")

            prompts = [
                random_prompt(preset.prompt_tokens, rng)
                for _ in range(args.requests_per_length)
            ]

            def send_request(prompt: str) -> Dict:
                record = {
                    "prompt": prompt,
                    "prompt_tokens": preset.prompt_tokens,
                    "max_new_tokens": preset.max_new_tokens,
                    "issued_at": time.time(),
                }
                try:
                    response = issue_request(
                        base_url,
                        prompt,
                        preset.max_new_tokens,
                        args.request_timeout,
                    )
                    record["status"] = "ok"
                    record["response"] = response
                except requests.RequestException as exc:
                    record["status"] = "error"
                    record["error"] = str(exc)
                    if args.debug:
                        print(f"[profile] Request failed ({label}): {exc}")
                return record

            max_workers = max(1, min(len(prompts), args.max_concurrent_requests))
            length_start = time.time()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(send_request, prompt) for prompt in prompts]
                responses = [future.result() for future in futures]
            length_end = time.time()

            # Wait for server to become idle before capturing metrics
            if args.wait_for_idle:
                wait_for_idle_server(
                    base_url,
                    timeout=args.idle_wait_timeout,
                    poll_interval=args.idle_poll_interval,
                    debug=args.debug,
                )

            # Optional additional fixed delay
            if args.post_length_wait > 0:
                time.sleep(args.post_length_wait)

            metrics_slice, end_index = metrics_buffer.slice_from(current_metric_index)
            current_metric_index = end_index

            summary = summarize_metrics(metrics_slice)
            summary.update(
                {
                    "prompt_tokens": preset.prompt_tokens,
                    "max_new_tokens": preset.max_new_tokens,
                    "requests_sent": len(responses),
                    "responses_ok": sum(1 for r in responses if r.get("status") == "ok"),
                    "responses_error": sum(1 for r in responses if r.get("status") == "error"),
                    "length_start_timestamp": length_start,
                    "length_end_timestamp": length_end,
                }
            )

            summary_path = run_dir / f"{label}_summary.json"
            responses_path = run_dir / f"{label}_responses.json"
            metrics_slice_path = run_dir / f"{label}_metrics.jsonl"

            write_json(summary_path, summary)
            write_json(responses_path, {"requests": responses})
            write_jsonl(metrics_slice_path, metrics_slice)

            if args.debug:
                print(f"[profile] Completed preset {label} with {summary['iterations']} iterations logged.")

        # Write overall artifacts
        write_jsonl(metrics_path, metrics_buffer.snapshot())
        run_manifest = {
            "args": {
                **vars(args),
                "length_configs": [asdict(preset) for preset in args.length_configs],
            },
            "server_command": server_command,
            "base_url": base_url,
            "run_dir": str(run_dir),
        }
        write_json(run_dir / "run_manifest.json", run_manifest)

    finally:
        if process is not None:
            terminate_process_tree(process)
        for thread in threads:
            thread.join(timeout=5)
        if port_lock is not None:
            port_lock.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[profile] interrupted by user", file=sys.stderr)
