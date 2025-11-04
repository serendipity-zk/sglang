#!/usr/bin/env python3
"""
Coordinator for running chunk_profile.py sweeps defined in a JSON config.

The config lists chunk sizes and shared request options. At runtime you provide
the set of GPUs to use; each GPU pulls chunk jobs from a shared queue so that
no GPU runs more than one profile job at a time while multiple GPUs can run in
parallel.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Sequence
import signal

CONFIG_DEFAULT = Path(__file__).with_name("config.default.json")
PROFILE_SCRIPT = Path(__file__).with_name("chunk_profile.py")


@dataclass
class SweepConfig:
    model_path: str
    chunk_sizes: List[int]
    length_configs: List[str]
    requests_per_length: int = 8
    profile_name_prefix: Optional[str] = None
    output_root: str = "./runs"
    enable_mixed_chunk: bool = True
    wait_for_idle: bool = False
    idle_wait_timeout: float = 30.0
    idle_poll_interval: float = 0.2
    post_length_wait: float = 2.0
    max_concurrent_requests: int = 32
    extra_profile_args: Optional[List[str]] = None
    sleep_between_jobs: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict) -> "SweepConfig":
        chunk_sizes = data.get("chunk_sizes")
        if not chunk_sizes:
            raise ValueError("Config must provide a non-empty 'chunk_sizes' list.")

        length_configs = data.get("length_configs")
        if not length_configs:
            raise ValueError("Config must provide a non-empty 'length_configs' list.")

        return cls(
            model_path=data["model_path"],
            chunk_sizes=[int(c) for c in chunk_sizes],
            length_configs=[str(cfg) for cfg in length_configs],
            requests_per_length=int(data.get("requests_per_length", 8)),
            profile_name_prefix=data.get("profile_name_prefix"),
            output_root=data.get("output_root", "./runs"),
            enable_mixed_chunk=bool(data.get("enable_mixed_chunk", True)),
            wait_for_idle=bool(data.get("wait_for_idle", False)),
            idle_wait_timeout=float(data.get("idle_wait_timeout", 30.0)),
            idle_poll_interval=float(data.get("idle_poll_interval", 0.2)),
            post_length_wait=float(data.get("post_length_wait", 2.0)),
            max_concurrent_requests=int(data.get("max_concurrent_requests", 32)),
            extra_profile_args=[
                str(arg) for arg in data.get("extra_profile_args", [])
            ],
            sleep_between_jobs=float(data.get("sleep_between_jobs", 0.0)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple chunk_profile.py runs based on a JSON configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DEFAULT,
        help="Path to JSON config describing the sweep.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="Comma-separated GPU indices to use at runtime (e.g. '0,1,2').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be launched without running them.",
    )
    return parser.parse_args()


def load_config(path: Path) -> SweepConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    return SweepConfig.from_dict(data)


def parse_gpu_list(gpu_arg: str) -> List[str]:
    parts = [p.strip() for p in gpu_arg.split(",") if p.strip()]
    if not parts:
        raise ValueError("At least one GPU index must be provided via --gpus.")
    return parts


def build_command(cfg: SweepConfig, chunk_size: int) -> List[str]:
    profile_name = (
        f"{cfg.profile_name_prefix}_chunk{chunk_size}"
        if cfg.profile_name_prefix
        else f"chunk{chunk_size}"
    )
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    chunk_output_dir = output_root / f"chunk{chunk_size}"
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--model-path",
        cfg.model_path,
        "--chunked-prefill-size",
        str(chunk_size),
        "--length-configs",
        *cfg.length_configs,
        "--requests-per-length",
        str(cfg.requests_per_length),
        "--output-root",
        str(chunk_output_dir),
        "--profile-name",
        profile_name,
        "--post-length-wait",
        str(cfg.post_length_wait),
    ]

    if cfg.enable_mixed_chunk:
        cmd.append("--enable-mixed-chunk")

    if cfg.wait_for_idle:
        cmd.append("--wait-for-idle")
        cmd.extend(["--idle-wait-timeout", str(cfg.idle_wait_timeout)])
        cmd.extend(["--idle-poll-interval", str(cfg.idle_poll_interval)])

    cmd.extend(["--max-concurrent-requests", str(cfg.max_concurrent_requests)])

    if cfg.extra_profile_args:
        cmd.extend(cfg.extra_profile_args)

    return cmd


def launch_process(command: Sequence[str], gpu: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    process = subprocess.Popen(
        command,
        stdout=None,
        stderr=None,
        env=env,
        preexec_fn=os.setsid,
    )
    return process


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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    gpu_list = parse_gpu_list(args.gpus)

    print(f"[run] Loaded config from {args.config}")
    print(f"[run] Using GPUs: {', '.join(gpu_list)}")
    print(f"[run] Scheduling {len(cfg.chunk_sizes)} chunk job(s)")

    if args.dry_run:
        for chunk in cfg.chunk_sizes:
            cmd = build_command(cfg, chunk)
            print(f"[dry-run] chunk {chunk}: {' '.join(cmd)}")
        return

    chunk_queue: queue.Queue[int] = queue.Queue()
    for chunk in cfg.chunk_sizes:
        chunk_queue.put(chunk)

    processes: List[subprocess.Popen] = []
    processes_lock = Lock()

    def cleanup() -> None:
        with processes_lock:
            for proc in processes:
                terminate_process_tree(proc)
            processes.clear()

    def worker(gpu: str) -> None:
        try:
            while True:
                try:
                    chunk = chunk_queue.get_nowait()
                except queue.Empty:
                    break

                cmd = build_command(cfg, chunk)
                print(f"[run] GPU {gpu}: launching chunk {chunk}")
                proc = launch_process(cmd, gpu)
                with processes_lock:
                    processes.append(proc)

                proc.wait()

                with processes_lock:
                    if proc in processes:
                        processes.remove(proc)

                if cfg.sleep_between_jobs > 0 and not chunk_queue.empty():
                    time.sleep(cfg.sleep_between_jobs)
        except Exception as exc:
            print(f"[run] Worker for GPU {gpu} failed: {exc}")
            raise

    threads: List[threading.Thread] = []

    try:
        for gpu in gpu_list:
            thread = threading.Thread(target=worker, args=(gpu,), daemon=True)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\n[run] Caught KeyboardInterrupt. Terminating child processes ...")
    finally:
        cleanup()
        for thread in threads:
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
