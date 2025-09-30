#!/usr/bin/env python3
"""
Ultra-fast fake \"/generate\" endpoint for isolating client-side bottlenecks.

Run this server locally and point ``issue.py`` at it to confirm whether
completion lag is caused by the request issuer or the actual router backend.
The server answers immediately (optionally with a configurable latency) and
returns a minimal OpenAI-compatible JSON payload.
"""

import argparse
import json
import random
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Tuple


class _GenerateHandler(BaseHTTPRequestHandler):
    """Handle POST /generate requests with an immediate canned response."""

    server_version = "FakeGenerate/0.1"
    protocol_version = "HTTP/1.1"

    def _read_request_json(self) -> Tuple[dict, bytes]:
        length = int(self.headers.get("content-length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            body = {}
        return body, raw

    def log_message(self, format: str, *args):  # noqa: N802  (signature fixed upstream)
        # Silence default access logs to keep test runs tidy.
        return

    def do_POST(self):  # noqa: D401  (BaseHTTPRequestHandler signature requirement)
        if self.path not in {"/generate", "/v1/generate"}:
            self.send_error(404, "Unknown endpoint")
            return

        payload, raw = self._read_request_json()
        server = self.server  # type: ignore[assignment]
        assert isinstance(server, FakeGenerateServer)

        # Optional artificial latency + jitter to emulate various router speeds.
        delay_s = server.fixed_delay
        if server.jitter > 0:
            delay_s = max(0.0, delay_s + random.uniform(-server.jitter, server.jitter))
        if delay_s:
            time.sleep(delay_s)

        prompt = payload.get("input_ids", [[]])
        prompt_tokens = len(prompt[0]) if prompt and prompt[0] else 0
        sampling_params = payload.get("sampling_params", {})
        max_new_tokens = int(sampling_params.get("max_new_tokens", 0) or 0)
        completion_tokens = min(max_new_tokens, server.max_completion_tokens)

        if completion_tokens:
            completion_text = " ".join(server.completion_words[:completion_tokens])
        else:
            completion_text = ""

        response_body = {
            "id": payload.get("request_id") or f"fake-{int(time.time() * 1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": server.model_name,
            "choices": [
                {
                    "index": 0,
                    "text": completion_text,
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "echoed_request": payload if server.echo_request else None,
        }

        body = json.dumps(response_body, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeGenerateServer(ThreadingHTTPServer):
    """Threaded HTTP server with a few knobs to simulate latency."""

    def __init__(self, host: str, port: int, *, latency_ms: float, jitter_ms: float,
                 max_completion_tokens: int, model_name: str, echo_request: bool):
        super().__init__((host, port), _GenerateHandler)
        self.fixed_delay = max(0.0, latency_ms) / 1000.0
        self.jitter = max(0.0, jitter_ms) / 1000.0
        self.max_completion_tokens = max_completion_tokens
        self.model_name = model_name
        self.echo_request = echo_request
        # Pre-generate words to avoid per-request allocations.
        self.completion_words = ["tok"] * max_completion_tokens

        # Allow clean shutdown on Ctrl+C without hanging threads.
        self._shutdown_requested = threading.Event()

    def serve_forever(self) -> None:  # noqa: D401
        try:
            super().serve_forever()
        finally:
            self._shutdown_requested.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightning-fast fake /generate server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=45000, help="Bind port")
    parser.add_argument("--latency-ms", type=float, default=10000.0, help="Fixed response delay in milliseconds")
    parser.add_argument("--jitter-ms", type=float, default=10.0, help="Uniform jitter (+/-) in milliseconds")
    parser.add_argument("--max-completion-tokens", type=int, default=32,
                        help="Cap fake completion length to avoid large payloads")
    parser.add_argument("--model-name", default="fake/test-model", help="Model name to echo in responses")
    parser.add_argument("--echo-request", action="store_true", help="Include the decoded request payload in responses")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = FakeGenerateServer(
        host=args.host,
        port=args.port,
        latency_ms=args.latency_ms,
        jitter_ms=args.jitter_ms,
        max_completion_tokens=max(1, args.max_completion_tokens),
        model_name=args.model_name,
        echo_request=args.echo_request,
    )
    addr, port = server.server_address
    print(f"[fake-server] Listening on http://{addr}:{port}/generate")
    print("[fake-server] Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[fake-server] Caught Ctrl+C, shutting down...")
    finally:
        server.shutdown()
        server.server_close()
        print("[fake-server] Stopped")


if __name__ == "__main__":
    main()
