#!/usr/bin/env python3
"""
Router scale evaluation controller.

This script orchestrates experiments by:
1. Launching fake servers
2. Launching the router
3. Running the client workload
4. Collecting all logs in an organized directory structure

Usage:
    python run_experiment.py --num-servers 8 --rate 100
    python run_experiment.py --num-servers 4 --rate 200 --name "4_servers_stress"
    python run_experiment.py --num-servers 8 --rate 50 --max-requests 10000
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent.absolute()
RESULTS_DIR = SCRIPT_DIR / "results"


class ExperimentRunner:
    """Manages the lifecycle of a router scale experiment."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.processes: List[Tuple[str, subprocess.Popen]] = []
        self.experiment_dir: Optional[Path] = None

    def setup_experiment_dir(self) -> Path:
        """Create experiment output directory with proper structure."""
        name = self.args.name or f"exp_{self.args.num_servers}srv_{self.args.rate}rate"
        self.experiment_dir = RESULTS_DIR / name

        # Create subdirectories
        (self.experiment_dir / "router_log").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "server_log").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "client_log").mkdir(parents=True, exist_ok=True)

        # Generate scheduler config with even tier allocation
        scheduler_config = self.generate_scheduler_config()
        self.scheduler_config_path = self.experiment_dir / "scheduler_config.json"
        with open(self.scheduler_config_path, "w") as f:
            json.dump(scheduler_config, f, indent=2)

        # Save experiment config
        config = {
            "num_servers": self.args.num_servers,
            "rate": self.args.rate,
            "max_requests": self.args.max_requests,
            "router_port": self.args.router_port,
            "base_port": self.args.base_port,
            "trace": self.args.trace,
            "scheduler_config": scheduler_config,
            "name": name,
        }
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Experiment directory: {self.experiment_dir}")
        return self.experiment_dir

    def generate_scheduler_config(self) -> dict:
        """Generate SLO-aware scheduler config with even tier allocation."""
        num_servers = self.args.num_servers
        tpot_buckets = self.args.tpot_buckets
        num_tiers = len(tpot_buckets)

        # Evenly allocate servers across tiers
        base_per_tier = num_servers // num_tiers
        remainder = num_servers % num_tiers

        # Distribute remainder to earlier tiers (tighter SLOs get more servers)
        tier_allocation = []
        for i in range(num_tiers):
            alloc = base_per_tier + (1 if i < remainder else 0)
            tier_allocation.append(alloc)

        return {
            "type": "slo_aware",
            "tpot_buckets": tpot_buckets,
            "initial_tier_allocation": tier_allocation,
            "send_tpot_updates": True,
            "worker_selection_policy": {
                "type": "TTFTAware",
                "margin_ms": 50.0,
            },
            "auto_scaling": {
                "enabled": False,
                "idle_tpot_ms": 1000.0,
            },
        }

    def launch_servers(self) -> subprocess.Popen:
        """Launch fake servers."""
        cmd = [
            "bash",
            str(SCRIPT_DIR / "launch_fake_servers.sh"),
            str(self.args.num_servers),
            f"http://0.0.0.0:{self.args.router_port}",
            str(self.args.base_port),
        ]

        env = os.environ.copy()
        # Override log directory to use experiment directory
        server_log_dir = self.experiment_dir / "server_log"
        env["SERVER_LOG_DIR"] = str(server_log_dir)

        log_file = open(server_log_dir / "launch_servers.log", "w")

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=SCRIPT_DIR,
        )
        self.processes.append(("servers", proc))
        return proc

    def launch_router(self) -> subprocess.Popen:
        """Launch the router."""
        router_log_dir = self.experiment_dir / "router_log"

        cmd = [
            "bash",
            str(SCRIPT_DIR / "launch_router.sh"),
            str(self.args.num_servers),
            str(self.args.router_port),
            str(router_log_dir),
            str(self.scheduler_config_path),
            str(self.args.prometheus_port),
        ]

        env = os.environ.copy()
        env["BASE_WORKER_PORT"] = str(self.args.base_port)

        log_file = open(router_log_dir / "router_stdout.log", "w")

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=SCRIPT_DIR,
        )
        self.processes.append(("router", proc))
        return proc

    def launch_client(self) -> subprocess.Popen:
        """Launch the client workload."""
        client_log_dir = self.experiment_dir / "client_log"

        cmd = [
            "bash",
            str(SCRIPT_DIR / "launch_client.sh"),
            str(self.args.rate),
            str(self.args.max_requests),
            str(client_log_dir),
            str(self.args.trace),
        ]

        env = os.environ.copy()
        env["BASE_URL"] = f"http://0.0.0.0:{self.args.router_port}/v1"
        env["MODEL"] = "fake/test-model"

        log_file = open(client_log_dir / "client_stdout.log", "w")

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=SCRIPT_DIR,
        )
        self.processes.append(("client", proc))
        return proc

    def wait_for_servers(self, timeout: int = 30) -> bool:
        """Wait for all fake servers to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            ready = 0
            for i in range(self.args.num_servers):
                port = self.args.base_port + i
                try:
                    urllib.request.urlopen(
                        f"http://0.0.0.0:{port}/health", timeout=1
                    )
                    ready += 1
                except Exception:
                    pass
            if ready == self.args.num_servers:
                # Extra wait to ensure servers are fully ready
                time.sleep(1)
                return True
            time.sleep(0.5)
        return False

    def wait_for_router(self, timeout: int = 30) -> bool:
        """Wait for router to be ready."""
        start = time.time()
        router_proc = None
        for name, proc in self.processes:
            if name == "router":
                router_proc = proc
                break

        while time.time() - start < timeout:
            # Check if router process crashed
            if router_proc and router_proc.poll() is not None:
                print(f"      Router process exited with code {router_proc.returncode}")
                return False

            try:
                urllib.request.urlopen(
                    f"http://0.0.0.0:{self.args.router_port}/health", timeout=1
                )
                return True
            except Exception:
                time.sleep(0.5)
        return False

    def kill_existing_processes(self) -> None:
        """Kill any existing fake servers and router processes."""
        print("Killing any existing processes...")
        # Be specific to avoid killing parent scripts
        for pattern in ["fake_server.py", "launch_router.sh", "launch_fake_servers.sh", "sglang_router.launch_router"]:
            try:
                subprocess.run(
                    ["pkill", "-9", "-f", pattern],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass

        # Kill any process using the router port or prometheus metrics port (default 28888)
        for port in [self.args.router_port, 28888]:
            try:
                subprocess.run(
                    ["fuser", "-k", f"{port}/tcp"],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass

        time.sleep(2)  # Give processes time to die

    def cleanup(self) -> None:
        """Kill all launched processes gracefully."""
        print("\nCleaning up processes...")

        # Kill in order: client, servers, router (servers before router to avoid router errors)
        kill_order = ["client", "servers", "router"]
        proc_dict = {name: proc for name, proc in self.processes}

        for name in kill_order:
            if name in proc_dict:
                proc = proc_dict[name]
                if proc.poll() is None:
                    print(f"  Terminating {name}...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"  Force killing {name}...")
                        proc.kill()
                        proc.wait()

        # Also kill any remaining fake servers and router processes
        self.kill_existing_processes()

        print("Cleanup complete.")

    def run(self) -> int:
        """Run the full experiment."""
        try:
            # Kill any existing processes first
            self.kill_existing_processes()

            # Setup
            self.setup_experiment_dir()

            # Launch servers
            print(f"\n[1/3] Launching {self.args.num_servers} fake servers...")
            self.launch_servers()
            if not self.wait_for_servers():
                print("ERROR: Servers did not start in time")
                return 1
            print(f"      Servers ready (ports {self.args.base_port}-{self.args.base_port + self.args.num_servers - 1})")

            # Launch router
            print(f"\n[2/3] Launching router on port {self.args.router_port}...")
            self.launch_router()
            if not self.wait_for_router():
                print("ERROR: Router did not start in time")
                return 1
            print("      Router ready")

            # Launch client
            print(f"\n[3/3] Running workload (rate={self.args.rate}, max_requests={self.args.max_requests})...")
            client_proc = self.launch_client()

            # Wait for client to complete
            client_proc.wait()
            print("      Workload complete")

            print(f"\n{'='*60}")
            print(f"Experiment complete!")
            print(f"Results: {self.experiment_dir}")
            print(f"{'='*60}")

            return 0

        except Exception as e:
            print(f"\nERROR: {e}")
            return 1

        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Router scale evaluation controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-servers",
        type=int,
        default=8,
        help="Number of fake servers to launch",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=100,
        help="Request rate (requests per second relative to trace)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=1000,
        help="Maximum number of requests to send",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--router-port",
        type=int,
        default=None,
        help="Router port (default: random 40000-49999)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=None,
        help="Base port for fake servers (default: random 30000-39999)",
    )
    parser.add_argument(
        "--trace",
        type=str,
        default="/sgl-workspace/sglang/SLO-CSim/trace/arxiv/uniform_512_512.csv",
        help="Path to trace file",
    )
    parser.add_argument(
        "--tpot-buckets",
        type=float,
        nargs="+",
        default=[10.0, 20.0, 40.0],
        help="TPOT buckets for SLO tiers (ms)",
    )

    args = parser.parse_args()

    # Assign random ports if not specified
    import random
    if args.router_port is None:
        args.router_port = random.randint(40000, 49999)
    if args.base_port is None:
        args.base_port = random.randint(30000, 39999)
    # Always use random prometheus port to avoid conflicts
    args.prometheus_port = random.randint(28000, 28999)

    print(f"Using router port: {args.router_port}")
    print(f"Using base port: {args.base_port}")
    print(f"Using prometheus port: {args.prometheus_port}")

    # Create runner
    runner = ExperimentRunner(args)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nInterrupted by user")
        runner.cleanup()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run experiment
    sys.exit(runner.run())


if __name__ == "__main__":
    main()
