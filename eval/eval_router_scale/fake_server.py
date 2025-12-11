"""
Fake server for SLO-aware scheduling testing.

This server:
1. Accepts requests and returns one token immediately (streaming)
2. Reports iteration metrics to the router
3. Simulates processing at configurable tokens/second rate
4. Accepts /set_tpot to receive TPOT targets from router

Usage:
    python fake_server.py --port 31001 --router-url http://0.0.0.0:40010
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fake_server")


@dataclass
class PendingRequest:
    """Tracks a pending request being processed."""

    request_id: str
    input_tokens: int
    output_tokens: int
    tokens_remaining: int
    future: asyncio.Future
    start_time: float = field(default_factory=time.time)


@dataclass
class ServerState:
    """Global server state for fake metrics simulation."""

    # Configuration
    max_tokens_per_second: int = 24576  # 8192*3
    iteration_interval_ms: float = 10.0  # ~100 iterations/sec
    model_name: str = "fake/test-model"
    worker_type: str = "regular"
    dp_size: int = 1

    # Router connection
    router_url: Optional[str] = None
    worker_id: str = "fake-worker-1"

    # TPOT configuration (set by router via /set_tpot)
    tpot_target_ms: float = 30.0

    # State tracking
    iteration_num: int = 0
    pending_requests: Dict[str, PendingRequest] = field(default_factory=dict)
    completed_requests: int = 0
    total_tokens_processed: int = 0

    # Metrics
    current_batch_tokens: int = 0
    kv_tokens_used: int = 0
    last_iteration_time_ms: float = 10.0

    # Control
    running: bool = True
    request_counter: int = 0

    # Message tracking for router acknowledgments
    router_generation: Optional[int] = None
    last_received_message_id: int = 0

    # Rolling window for token tracking (timestamp_ms, tokens)
    token_history: Deque[Tuple[float, int]] = field(default_factory=deque)
    rolling_window_ms: float = 200.0  # Track tokens in last 200ms

    def record_tokens(self, tokens: int) -> None:
        """Record tokens received at current time."""
        now_ms = time.time() * 1000
        self.token_history.append((now_ms, tokens))
        self._prune_old_tokens(now_ms)

    def _prune_old_tokens(self, now_ms: float) -> None:
        """Remove tokens older than rolling window."""
        cutoff = now_ms - self.rolling_window_ms
        while self.token_history and self.token_history[0][0] < cutoff:
            self.token_history.popleft()

    def get_tokens_in_window(self) -> int:
        """Get total tokens received in the rolling window."""
        now_ms = time.time() * 1000
        self._prune_old_tokens(now_ms)
        return sum(tokens for _, tokens in self.token_history)

    def get_remaining_budget(self) -> int:
        """Get remaining token budget for the rolling window."""
        # Budget = max_tokens_per_second * (window_ms / 1000)
        window_budget = int(self.max_tokens_per_second * (self.rolling_window_ms / 1000.0))
        used = self.get_tokens_in_window()
        return max(0, window_budget - used)

    @property
    def num_requests(self) -> int:
        return len(self.pending_requests)

    @property
    def waiting_queue_size(self) -> int:
        # For simplicity, we don't distinguish queue vs active
        return 0

    @property
    def tokens_per_iteration(self) -> float:
        return self.max_tokens_per_second * (self.iteration_interval_ms / 1000.0)


# Global state
state: Optional[ServerState] = None


def create_app(server_state: ServerState) -> FastAPI:
    """Create FastAPI application with all endpoints."""

    app = FastAPI(title="Fake SGLang Server")

    # --- Health endpoints ---

    @app.get("/health")
    async def health():
        return PlainTextResponse("ok", status_code=200)

    @app.get("/health_generate")
    async def health_generate():
        return PlainTextResponse("ok", status_code=200)

    # --- Server info endpoints ---

    @app.get("/get_server_info")
    async def get_server_info():
        return JSONResponse(
            {
                "worker_id": server_state.worker_id,
                "model_id": server_state.model_name,
                "model_path": server_state.model_name,
                "priority": 1,
                "cost": 1.0,
                "worker_type": server_state.worker_type,
                "dp_size": server_state.dp_size,
                "load_in_flight": server_state.num_requests,
                "cache": {"size": 0, "hit_rate": 0.0},
            }
        )

    @app.get("/get_model_info")
    async def get_model_info():
        return JSONResponse(
            {
                "model": server_state.model_name,
                "vocab_size": 128256,  # Llama-3 vocab size
            }
        )

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse(
            {
                "data": [
                    {
                        "id": server_state.model_name,
                        "object": "model",
                        "owned_by": "fake",
                    }
                ]
            }
        )

    # --- TPOT configuration ---

    @app.api_route("/set_tpot", methods=["POST", "PUT"])
    async def set_tpot(request: Request):
        try:
            data = await request.json()
            tpot = data.get("tpot", 30.0)
            server_state.tpot_target_ms = float(tpot)
            logger.info(f"[FakeServer] TPOT target set to {tpot}ms")
            return JSONResponse({"success": True, "tpot": tpot})
        except Exception as e:
            logger.error(f"[FakeServer] set_tpot error: {e}")
            return JSONResponse({"success": False, "error": str(e)}, status_code=400)

    # --- Cache endpoint ---

    @app.post("/flush_cache")
    async def flush_cache():
        return PlainTextResponse("ok", status_code=200)

    # --- Generation endpoints ---

    async def generate_stream(request_id: str, output_tokens: int):
        """Generate SSE stream with tokens."""
        # Return tokens one by one
        for i in range(output_tokens):
            chunk = {
                "id": request_id,
                "object": "text_completion",
                "choices": [
                    {
                        "text": "x",  # Single token
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
                "x-server-id": server_state.worker_id,
                "x-iteration-id": server_state.iteration_num,
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            # Small delay to simulate token generation
            await asyncio.sleep(0.001)

        # Final chunk with finish reason
        final_chunk = {
            "id": request_id,
            "object": "text_completion",
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "x-server-id": server_state.worker_id,
            "x-iteration-id": server_state.iteration_num,
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    async def handle_generate_request(request: Request, is_chat: bool = False):
        """Handle generate/completion request."""
        try:
            data = await request.json()
        except Exception:
            data = {}

        # Extract router tracking info
        router_gen = data.get("router_generation")
        msg_id = data.get("router_message_id")
        if router_gen is not None:
            server_state.router_generation = router_gen
        if msg_id is not None:
            server_state.last_received_message_id = max(
                server_state.last_received_message_id, msg_id
            )

        # Parse request parameters
        server_state.request_counter += 1
        request_id = f"cmpl-fake-{server_state.request_counter}"

        # Determine output tokens
        max_tokens = data.get(
            "max_new_tokens", data.get("max_tokens", data.get("max_completion_tokens", 1))
        )
        output_tokens = min(max_tokens, 1)  # Return only 1 token as per spec

        # Check if streaming
        stream = data.get("stream", False)

        # Update metrics - add to pending
        # Handle different request formats:
        # 1. input_ids: [[token_ids]] - array of token arrays (Rust client)
        # 2. text/prompt: string content
        # 3. messages: chat format
        input_ids = data.get("input_ids")
        if input_ids and isinstance(input_ids, list) and len(input_ids) > 0:
            # input_ids is [[token_ids]] - count tokens from first array
            if isinstance(input_ids[0], list):
                input_tokens = len(input_ids[0])
            else:
                input_tokens = len(input_ids)
        else:
            text_content = data.get("text") or data.get("prompt") or data.get("input") or ""
            input_tokens = len(text_content) // 4  # rough estimate
            if is_chat:
                messages = data.get("messages", [])
                input_tokens = sum(len(str(m)) for m in messages) // 4

        server_state.kv_tokens_used += input_tokens
        server_state.record_tokens(input_tokens)
        logger.info(
            f"[FakeServer:{server_state.worker_id}] REQ {request_id} input_tokens={input_tokens} "
            f"tokens_in_window={server_state.get_tokens_in_window()} remaining_budget={server_state.get_remaining_budget()}"
        )

        if stream:
            headers = {"X-Server-Id": server_state.worker_id}
            return StreamingResponse(
                generate_stream(request_id, output_tokens),
                media_type="text/event-stream",
                headers=headers,
            )
        else:
            # Non-streaming response
            response = {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": server_state.model_name,
                "choices": [
                    {
                        "text": "x",
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                "x-server-id": server_state.worker_id,
                "x-iteration-id": server_state.iteration_num,
            }
            resp = JSONResponse(response)
            resp.headers["X-Server-Id"] = server_state.worker_id
            return resp

    @app.post("/generate")
    async def generate(request: Request):
        return await handle_generate_request(request, is_chat=False)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await handle_generate_request(request, is_chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await handle_generate_request(request, is_chat=True)

    return app


async def metrics_reporter(server_state: ServerState):
    """Background task to report metrics to router."""
    if not server_state.router_url:
        logger.info("[FakeServer] No router URL configured, skipping metrics reporting")
        return

    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0.5))

    try:
        while server_state.running:
            # Compute prefill_sim_results based on remaining budget
            # The server can only handle (remaining_budget) more tokens within the window
            # If a request needs more tokens than remaining budget, TTFT will be higher
            tokens_per_ms = server_state.max_tokens_per_second / 1000.0
            remaining_budget = server_state.get_remaining_budget()

            prefill_sim_results = {}
            for tokens in [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
                if tokens <= remaining_budget:
                    # Can handle within current capacity
                    time_ms = tokens / tokens_per_ms if tokens_per_ms > 0 else 0
                else:
                    # Exceeds remaining budget - set to very large value to prevent selection
                    time_ms = 999999.0
                prefill_sim_results[str(tokens)] = round(time_ms, 2)

            # Debug log: show budget and prefill_sim for key token counts
            tokens_in_window = server_state.get_tokens_in_window()
            window_budget = int(server_state.max_tokens_per_second * (server_state.rolling_window_ms / 1000.0))
            logger.info(
                f"[FakeServer:{server_state.worker_id}] iter={server_state.iteration_num} "
                f"tokens_in_window={tokens_in_window} budget={window_budget} remaining={remaining_budget} "
                f"prefill_sim[512]={prefill_sim_results.get('512', 'N/A')}ms "
                f"prefill_sim[1024]={prefill_sim_results.get('1024', 'N/A')}ms "
                f"prefill_sim[2048]={prefill_sim_results.get('2048', 'N/A')}ms"
            )

            # Compute fake metrics
            metrics = {
                "worker_id": server_state.worker_id,
                "batch_size_tokens": server_state.current_batch_tokens,
                "num_requests": server_state.num_requests,
                "kv_tokens_used": server_state.kv_tokens_used,
                "waiting_queue_size": server_state.waiting_queue_size,
                "forward_mode": "DECODE",
                "iteration_num": server_state.iteration_num,
                "last_iteration_time_ms": server_state.last_iteration_time_ms,
                # Prefill simulation metrics: maps token count to estimated TTFT in ms
                # Used by SLO-aware scheduler for TTFT estimation
                "prefill_sim_results": prefill_sim_results,
            }

            # Add router acknowledgment fields if available
            if server_state.router_generation is not None:
                metrics["router_generation"] = server_state.router_generation
                metrics["last_received_message_id"] = server_state.last_received_message_id

            # POST to router
            try:
                url = f"{server_state.router_url}/worker_stats"
                async with session.post(url, json=metrics) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"[FakeServer] Metrics POST failed: {resp.status}"
                        )
            except asyncio.TimeoutError:
                pass  # Timeout is expected, don't spam logs
            except Exception as e:
                logger.debug(f"[FakeServer] Metrics POST error: {e}")

            # Advance iteration
            server_state.iteration_num += 1

            # Simulate some batch activity
            server_state.current_batch_tokens = int(
                server_state.tokens_per_iteration * 0.8
            )  # 80% utilization

            await asyncio.sleep(server_state.iteration_interval_ms / 1000.0)
    finally:
        await session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fake SGLang server for testing")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, required=True, help="Bind port")
    parser.add_argument(
        "--router-url",
        default=None,
        help="Router URL for metrics reporting (e.g., http://0.0.0.0:40010)",
    )
    parser.add_argument(
        "--worker-id", default=None, help="Worker ID (default: fake-worker-{port})"
    )
    parser.add_argument(
        "--model-name", default="fake/test-model", help="Model name to report"
    )
    parser.add_argument(
        "--max-tokens-per-second",
        type=int,
        default=12288,
        help="Max tokens per second (default: 12288 = 4096*3)",
    )
    parser.add_argument(
        "--iteration-interval-ms",
        type=float,
        default=10.0,
        help="Iteration interval in ms (default: 10)",
    )
    parser.add_argument(
        "--worker-type",
        default="regular",
        choices=["regular", "prefill", "decode"],
        help="Worker type (default: regular)",
    )
    parser.add_argument("--dp-size", type=int, default=1, help="DP size (default: 1)")
    return parser.parse_args()


def main():
    global state

    args = parse_args()

    # Create server state
    worker_id = args.worker_id or f"http://0.0.0.0:{args.port}"
    state = ServerState(
        max_tokens_per_second=args.max_tokens_per_second,
        iteration_interval_ms=args.iteration_interval_ms,
        model_name=args.model_name,
        worker_type=args.worker_type,
        dp_size=args.dp_size,
        router_url=args.router_url,
        worker_id=worker_id,
    )

    logger.info(f"[FakeServer] Starting on {args.host}:{args.port}")
    logger.info(f"[FakeServer] Worker ID: {worker_id}")
    logger.info(f"[FakeServer] Max tokens/sec: {args.max_tokens_per_second}")
    logger.info(f"[FakeServer] Router URL: {args.router_url}")

    # Create app
    app = create_app(state)

    # Setup graceful shutdown
    def shutdown_handler(*_):
        state.running = False
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    # Start metrics reporter in background
    @app.on_event("startup")
    async def start_metrics():
        asyncio.create_task(metrics_reporter(state))

    # Run server
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
