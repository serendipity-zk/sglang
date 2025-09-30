#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import signal
import socket
import sys
import time
from datetime import datetime
from subprocess import Popen, PIPE
from threading import Thread

# -------- utilities --------

def ensure_dir(p: str):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def wait_port(host: str, port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.2)
    return False

def stream_output(proc: Popen, prefix: str, logfile_path: str):
    """
    将子进程的 stdout/stderr 同时写到终端（带前缀）和独立日志文件（追加）。
    """
    logf = open(logfile_path, "ab", buffering=0)

    def pump(stream, tag):
        for line in iter(stream.readline, b""):
            # 写终端
            sys.stdout.write(f"[{prefix}:{tag}] {line.decode(errors='replace')}")
            sys.stdout.flush()
            # 写文件
            try:
                logf.write(line)
            except Exception:
                pass

    Thread(target=pump, args=(proc.stdout, "O"), daemon=True).start()
    Thread(target=pump, args=(proc.stderr, "E"), daemon=True).start()

    # 返回文件句柄以便退出时关闭
    return logf

def start_proc(cmd, env=None, prefix="proc", logfile_path="worker.log"):
    """
    启动子进程并置于独立进程组（便于整体信号退出）。
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = Popen(
        cmd,
        stdout=PIPE,
        stderr=PIPE,
        env=full_env,
        preexec_fn=os.setsid  # Linux: 新建会话/进程组
    )
    log_handle = stream_output(proc, prefix, logfile_path)
    return proc, log_handle

def kill_proc_tree(proc: Popen, grace=10):
    """优雅退出：SIGINT -> 等待 -> SIGTERM -> 等待 -> SIGKILL"""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    deadline = time.time() + grace
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(25):
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

# -------- main --------

def main():
    parser = argparse.ArgumentParser(
        description="Launch multiple SGLang workers (one GPU each) with per-worker logs; Ctrl-C to stop all."
    )
    parser.add_argument("--model-path", required=True,
                        help="HF 名称或本地路径，如 meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--ports", default="31001,31002,31003,31004",
                        help="逗号分隔端口列表")
    parser.add_argument("--gpus", default="0,1,2,3",
                        help="逗号分隔 GPU 列表（与端口一一对应）")
    parser.add_argument("--log-dir", default="logs",
                        help="日志目录（每个 worker 一个文件）")
    parser.add_argument("--extra-worker-args", default="",
                        help="附加给 sglang worker 的参数串，例如: \"--max-concurrent-requests 16\"")
    parser.add_argument("--no-wait", action="store_true",
                        help="不等待端口就绪")
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
    gpus  = [g.strip() for g in args.gpus.split(",") if g.strip()]
    assert len(ports) == len(gpus), "ports 与 gpus 数量必须一致"

    ensure_dir(args.log_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    procs = []
    logs = []
    try:
        # 启动 workers
        for idx, (gpu, port) in enumerate(zip(gpus, ports), start=1):
            env = {"CUDA_VISIBLE_DEVICES": gpu}
            cmd = [
                sys.executable, "-m", "sglang.launch_server",
                "--model-path", args.model_path,
                "--host", args.host, "--port", str(port),
            ]
            if args.extra_worker_args:
                cmd.extend(args.extra_worker_args.split())

            log_path = os.path.join(
                args.log_dir, f"worker_{idx}_gpu{gpu}_p{port}_{stamp}.log"
            )
            prefix = f"w{idx}@gpu{gpu}:{port}"
            print(f"Starting worker {idx} on GPU {gpu} port {port} ... log -> {log_path}")
            proc, logf = start_proc(cmd, env=env, prefix=prefix, logfile_path=log_path)
            procs.append(proc)
            logs.append(logf)

        # 等待端口监听
        if not args.no_wait:
            for port in ports:
                ok = wait_port(args.host, port, timeout=600)
                print(f"Wait {args.host}:{port} -> {'READY' if ok else 'TIMEOUT'}")

        print("\nWorkers started. 按 Ctrl-C 结束所有 worker。\n")

        # 主循环：若任一子进程退出则结束（可按需改成忽略）
        while True:
            alive = [p for p in procs if p.poll() is None]
            if not alive:
                print("All workers exited.")
                break
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nCtrl-C received; stopping all workers ...")
    finally:
        # 关闭与回收
        for p in procs:
            kill_proc_tree(p)
        for lf in logs:
            try:
                lf.close()
            except Exception:
                pass
        print("All workers terminated.")

if __name__ == "__main__":
    main()
