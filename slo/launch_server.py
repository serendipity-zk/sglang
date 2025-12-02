#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import signal
import socket
import sys
import time
import errno
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

def is_port_available(host: str, port: int) -> bool:
    """检查端口是否可用（能否绑定）。可在启动前快速失败。

    注意：仍然存在竞态条件，无法完全避免并发时端口被其他进程抢占。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        s.bind((host, port))
        # 监听后立即释放（仅用于检测）
        s.listen(1)
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def who_uses_port(port: int, timeout_s: float = 5.0):
    """返回占用指定端口的进程信息列表（尽力而为）。

    优先使用 psutil；否则回退到 `ss` 或 `lsof`。返回元素形如：
    {"pid": 1234, "name": "python", "cmdline": "/usr/bin/python ..."}
    """
    results = []
    deadline = time.time() + max(0.1, float(timeout_s))
    # Try psutil first
    try:
        import psutil  # type: ignore

        conns = psutil.net_connections(kind="inet")
        for conn in conns:
            if time.time() > deadline:
                break
            try:
                if not conn.laddr:
                    continue
                if int(getattr(conn, "laddr").port) != int(port):
                    continue
                if conn.status != psutil.CONN_LISTEN:
                    continue
                pid = conn.pid
                if pid:
                    try:
                        p = psutil.Process(pid)
                        name = p.name()
                        cmdline = " ".join(p.cmdline())
                    except Exception:
                        name = "unknown"
                        cmdline = ""
                    results.append({"pid": pid, "name": name, "cmdline": cmdline})
            except Exception:
                continue
    except Exception:
        pass

    if results:
        return results

    # Fallback to `ss -ltnp`
    try:
        import subprocess, re

        remaining = max(0.0, deadline - time.time())
        if remaining > 0.05:
            out = subprocess.check_output(
                ["ss", "-ltnp"], stderr=subprocess.DEVNULL, timeout=min(5.0, remaining)
            ).decode()
            for line in out.splitlines():
                if f":{port} " in line or line.rstrip().endswith(f":{port}"):
                    m = re.search(r'users:\\(\\(([^,\\)]+),pid=(\\d+)', line)
                    if m:
                        name = m.group(2) if m.group(2) else m.group(1)
                        try:
                            pid = int(m.group(3))
                        except Exception:
                            pid = None
                        results.append({"pid": pid, "name": name.strip('"'), "cmdline": ""})
    except Exception:
        pass

    if results:
        return results

    # Fallback to `lsof`
    try:
        import subprocess

        remaining = max(0.0, deadline - time.time())
        if remaining > 0.05:
            out = subprocess.check_output(
                ["lsof", "-nP", "-i", f"TCP:{port}", "-sTCP:LISTEN"],
                stderr=subprocess.DEVNULL,
                timeout=min(5.0, remaining),
            ).decode()
            lines = out.splitlines()
            for line in lines[1:]:  # skip header
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    try:
                        pid = int(parts[1])
                    except Exception:
                        pid = None
                    results.append({"pid": pid, "name": name, "cmdline": ""})
    except Exception:
        pass

    return results


def _read_cmdline_from_proc(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            data = f.read().replace(b"\x00", b" ").decode(errors="replace").strip()
            return data
    except Exception:
        return ""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError as e:
        return e.errno != errno.ESRCH


def _kill_pid_gracefully(pid: int, label: str = "", grace: float = 8.0):
    try:
        os.kill(pid, signal.SIGINT)
    except Exception:
        pass
    deadline = time.time() + grace
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.1)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
        deadline = time.time() + 4.0
        while time.time() < deadline and _pid_alive(pid):
            time.sleep(0.1)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass

def stream_output(proc: Popen, prefix: str, logfile_path: str, echo_console: bool = False):
    """
    持续读取子进程 stdout/stderr，并将日志写入独立日志文件（追加）。
    若 echo_console=True，则同时打印到当前终端（带前缀）。
    """
    logf = open(logfile_path, "ab", buffering=0)

    def pump(stream, tag):
        for line in iter(stream.readline, b""):
            if echo_console:
                try:
                    sys.stdout.write(f"[{prefix}:{tag}] {line.decode(errors='replace')}")
                    sys.stdout.flush()
                except Exception:
                    pass
            try:
                logf.write(line)
            except Exception:
                pass

    Thread(target=pump, args=(proc.stdout, "O"), daemon=True).start()
    Thread(target=pump, args=(proc.stderr, "E"), daemon=True).start()

    # 返回文件句柄以便退出时关闭
    return logf

def start_proc(cmd, env=None, prefix="proc", logfile_path="worker.log", echo_console: bool = False):
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
    log_handle = stream_output(proc, prefix, logfile_path, echo_console=echo_console)
    return proc, log_handle

def kill_proc_tree(proc: Popen, grace=1):
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
    for _ in range(5):
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
    parser.add_argument("--tmux-ui", action="store_true",
                        help="使用 tmux 打开4个纵向pane，分别显示每个server的UI (不自动attach)" )
    parser.add_argument("--tmux-attach", action="store_true",
                        help="创建 tmux UI 后自动 attach（Ctrl-C 将不会被本进程接收）")
    args = parser.parse_args()

    ports = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
    gpus  = [g.strip() for g in args.gpus.split(",") if g.strip()]
    assert len(ports) == len(gpus), "ports 与 gpus 数量必须一致"

    # 启动前检查端口可用性与重复项，失败则立即退出
    dup_ports = [p for p in set(ports) if ports.count(p) > 1]
    if dup_ports:
        print(f"错误：端口列表包含重复项: {sorted(dup_ports)}")
        sys.exit(1)

    unavailable = [p for p in ports if not is_port_available(args.host, p)]
    if unavailable:
        # 尝试自动结束已存在的 sglang.launch_server 进程占用
        recovered = []
        for p in list(unavailable):
            holders = who_uses_port(p)
            killed_any = False
            for h in holders:
                pid = h.get("pid")
                if not pid:
                    continue
                cmd = h.get("cmdline") or _read_cmdline_from_proc(pid)
                if "sglang.launch_server" in cmd:
                    print(f"检测到已有 sglang.launch_server (PID={pid}) 占用 {args.host}:{p}，尝试结束...")
                    _kill_pid_gracefully(pid, label=f"{p}")
                    killed_any = True
            # 重新检查端口
            if killed_any and is_port_available(args.host, p):
                print(f"已释放端口 {args.host}:{p}")
                recovered.append(p)
        # 从不可用列表移除已恢复的端口
        if recovered:
            unavailable = [p for p in unavailable if p not in recovered]

        if unavailable:
            print("错误：以下端口不可用（已被占用或无权限）：")
            for p in unavailable:
                print(f"  - {args.host}:{p}")
                holders = who_uses_port(p)
                if holders:
                    for h in holders:
                        pid = h.get("pid")
                        name = h.get("name")
                        cmd = h.get("cmdline") or (pid and _read_cmdline_from_proc(pid)) or ""
                        if pid is not None:
                            print(f"      PID={pid} NAME={name} CMD={cmd}")
                else:
                    print("      (无法识别占用进程；可能需要安装 psutil/ss/lsof 或更高权限)")
            print("请更换端口或释放占用后重试。")
            sys.exit(1)

    ensure_dir(args.log_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    procs = []
    logs = []
    tmux_session = None
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
                args.log_dir, f"worker_{idx}_gpu{gpu}_p{port}_{stamp}.ans"
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

        # Optionally open a tmux UI window stacking four panes
        if args.tmux_ui:
            try:
                session = f"sgl-ui-{stamp}"
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "server_ui.py"))
                # First pane
                first_cmd = (
                    f"{sys.executable} -u {script_path} --url http://{args.host}:{ports[0]} "
                    f"--name w1@{ports[0]}"
                )
                os.system(
                    f"tmux new-session -d -s {session} '{first_cmd}' >/dev/null 2>&1"
                )
                # Remaining panes stacked vertically
                for i, port in enumerate(ports[1:], start=2):
                    cmd = (
                        f"{sys.executable} -u {script_path} --url http://{args.host}:{port} "
                        f"--name w{i}@{port}"
                    )
                    os.system(
                        f"tmux split-window -v -t {session} '{cmd}' >/dev/null 2>&1"
                    )
                    # Ensure even layout after each split
                    os.system(
                        f"tmux select-layout -t {session} even-vertical >/dev/null 2>&1"
                    )
                os.system(f"tmux set-option -t {session} remain-on-exit on >/dev/null 2>&1")
                os.system(f"tmux select-layout -t {session} even-vertical >/dev/null 2>&1")
                tmux_session = session
                print(f"Launching tmux UI session: {session}")
                print(f"  Attach: tmux attach -t {session}")
                if args.tmux_attach:
                    # Foreground attach: user Ctrl-b d to detach; Ctrl-C won't reach this process
                    os.system(f"tmux attach -t {session}")
            except Exception as e:
                print(f"[WARN] Failed to launch tmux UI: {e}")

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
        if tmux_session is not None:
            # Kill the UI session on exit for clean-up convenience
            os.system(f"tmux kill-session -t {tmux_session} >/dev/null 2>&1")
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
