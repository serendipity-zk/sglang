#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from typing import Dict, Optional

import requests


def clear(full: bool = True):
    # Move cursor to home; optionally clear full screen
    if full:
        sys.stdout.write("\x1b[2J\x1b[H")
    else:
        sys.stdout.write("\x1b[H")
    sys.stdout.flush()


def render(name: str, stats: Dict[str, object], url: str, stale: bool = False) -> None:
    model = stats.get("model", "-")
    pid = stats.get("pid", "-")
    accepted = stats.get("accepted_requests", 0)
    running_bs = stats.get("running_batch_size")
    queue_reqs = stats.get("queue_reqs")
    token_usage_pct = stats.get("kv_usage_pct")
    kv_tokens_used = stats.get("kv_tokens_used")
    token_capacity = stats.get("token_capacity")
    prefill_tokens = stats.get("prefill_tokens")
    decode_tokens = stats.get("decode_tokens")
    token_batch = stats.get("token_batch_size")
    input_tokens = stats.get("input_tokens")
    gen_tps = stats.get("gen_throughput_tps")
    last_bs = stats.get("last_batch_size", 0)

    lines = []
    title = f"SGLang Server - {name}"
    if stale:
        title += "  [STALE]"
    lines.append(title)
    lines.append("===========================")
    lines.append(f"URL: {url}")
    lines.append(f"Model: {model}")
    lines.append(f"PID: {pid}")
    lines.append("")
    lines.append(f"Accepted Requests:   {accepted}")
    if running_bs is not None:
        lines.append(f"Running Batch Size:  {running_bs}")
    else:
        lines.append(f"Last Batch Size:     {last_bs}")
    if queue_reqs is not None:
        lines.append(f"Queue Reqs:          {queue_reqs}")
    if kv_tokens_used is not None and token_capacity is not None:
        try:
            used_k = kv_tokens_used / 1000.0
            cap_k = token_capacity / 1000.0
            lines.append(f"KV Tokens:           {used_k:.1f}k / {cap_k:.1f}k")
        except Exception:
            lines.append(f"KV Tokens:           {kv_tokens_used} / {token_capacity}")
    elif token_usage_pct is not None:
        lines.append(f"KV Usage:            {token_usage_pct:.2f}%")
    # Prefill/Decode breakdown with clearer units
    def _fmt_tokens(n: Optional[object]) -> Optional[str]:
        try:
            v = int(n) if n is not None else None
        except Exception:
            return None
        if v is None:
            return None
        if v < 1000:
            return str(v)
        return f"{v/1000.0:.1f}k"

    ft = _fmt_tokens(prefill_tokens)
    dt = _fmt_tokens(decode_tokens)
    tt = _fmt_tokens(token_batch)
    if ft is not None:
        lines.append(f"Prefill Tokens:      {ft}")
    if dt is not None:
        # Decode tokens activated in current step (or combined when mixed-chunk enabled)
        lines.append(f"Decode Tokens:       {dt}")
    if tt is not None and ft is not None and dt is not None:
        lines.append(f"Token Batch:         {ft} + {dt} = {tt}")
    # Low-level embedding input size from runner (most recent pass)
    it = _fmt_tokens(input_tokens)
    if it is not None:
        lines.append(f"Input Tokens:        {it}")
    if gen_tps is not None:
        lines.append(f"Gen Throughput:      {gen_tps:.2f} tok/s")

    # Render in one write to reduce flicker
    sys.stdout.write("\x1b[?25l")  # hide cursor
    clear(full=True)
    sys.stdout.write("\n".join(lines) + "\n\x1b[0J")  # clear below
    sys.stdout.flush()


def fetch_stats(url: str, timeout: float = 0.5, session: Optional[requests.Session] = None) -> Optional[Dict[str, object]]:
    try:
        sess = session or requests
        r = sess.get(url.rstrip("/") + "/ui_stats", timeout=timeout)
        if r.ok:
            return r.json()
        return None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Tiny per-server TUI for SGLang server")
    ap.add_argument("--url", required=True, help="Base URL, e.g. http://127.0.0.1:31001")
    ap.add_argument("--name", default=None, help="Pane title or friendly name")
    ap.add_argument("--refresh", type=float, default=0.5, help="Refresh interval seconds")
    args = ap.parse_args()

    name = args.name or args.url
    last_stats: Optional[Dict[str, object]] = None
    session = requests.Session()
    # initial hide cursor
    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()
    try:
        while True:
            # Fetch first to avoid blank screen while waiting for network
            stats = fetch_stats(args.url, timeout=max(0.15, args.refresh * 0.6), session=session)
            if stats is None and last_stats is None:
                # Nothing yet; render minimal message
                clear(full=True)
                sys.stdout.write(f"SGLang Server - {name}\n")
                sys.stdout.write("===========================\n")
                sys.stdout.write(f"URL: {args.url}\n\n")
                sys.stdout.write("Waiting for /ui_stats ...\n")
                sys.stdout.flush()
            else:
                if stats is not None:
                    last_stats = stats
                    stale = False
                else:
                    stale = True
                render(name, last_stats or {}, args.url, stale=stale)
            time.sleep(args.refresh)
    finally:
        # show cursor back on exit
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
