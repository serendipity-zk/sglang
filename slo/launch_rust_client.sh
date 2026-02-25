#!/bin/bash

# Launcher for the high-performance Rust streaming issue client
#
# Key features:
# - Native Rust implementation with tokio async runtime
# - Low-latency streaming with reqwest (minimal buffering)
# - High-performance single-process async (no GIL, no multiprocessing overhead)
# - Manual SSE line parsing for fine-grained timing control
# - Massive concurrency via async tasks (spawns all requests immediately)
#
# Expected characteristics vs Python aiohttp client:
# - Lower CPU overhead (compiled Rust vs interpreted Python)
# - Simpler concurrency model (async/await vs multiprocessing)
# - Similar latency characteristics for streaming
# - No per-tier SLO attainment tracking (logs per-request only)
#
# IMPORTANT NOTE:
# - Rust version now supports both HuggingFace tokenizer names AND file paths
# - HF model names (e.g., "meta-llama/Llama-3.1-8B-Instruct") will be auto-downloaded
# - File paths (e.g., "/path/to/tokenizer.json") are also supported
# - First run will cache the tokenizer in ~/.cache/huggingface/

# Path to the compiled Rust binary (release build for performance)
RUST_BINARY="/sgl-workspace/sglang/slo/rust_client/target/release/slo_runner"

# Build if missing or sources changed
cd /sgl-workspace/sglang/slo/rust_client
if [ ! -f "$RUST_BINARY" ] || find src Cargo.toml -newer "$RUST_BINARY" | read; then
    echo "Building release binary..."
    cargo build --release
fi
cd /sgl-workspace/sglang

# Run the Rust client with HuggingFace tokenizer (matches Python version)
# The tokenizer will be automatically downloaded from HuggingFace on first run
# $RUST_BINARY \
#   --trace SLO-CSim/trace/1024_1024_152540.csv \
#   --text-file /sgl-workspace/sglang/slo/text/enwik8 \
#   --tokenizer meta-llama/Llama-3.1-8B-Instruct \
#   --base-url http://0.0.0.0:40010/v1 \
#   --model meta-llama/Llama-3.1-8B-Instruct \
#   --rate 40 \
#   --max-requests 2000 \
#   --log-path /sgl-workspace/sglang/slo/logs/rust_client_output.jsonl \
#   --elapsed-dump-path /sgl-workspace/sglang/slo/logs/rust_elapsed_timelines.pkl \
#   --ans-log-path /sgl-workspace/sglang/slo/logs/rust_client.ans \
#   --slo-use-detokenize-time

# $RUST_BINARY \
#   --trace SLO-CSim/trace/arxiv/sharegpt.csv \
#   --text-file /sgl-workspace/sglang/slo/text/enwik8 \
#   --tokenizer meta-llama/Llama-3.1-8B-Instruct \
#   --base-url http://0.0.0.0:40010/v1 \
#   --model meta-llama/Llama-3.1-8B-Instruct \
#   --rate 130 \
#   --max-requests 5000 \
#   --log-path /sgl-workspace/sglang/slo/logs/rust_client_output.jsonl \
#   --elapsed-dump-path /sgl-workspace/sglang/slo/logs/rust_elapsed_timelines.pkl \
#   --ans-log-path /sgl-workspace/sglang/slo/logs/rust_client.ans \
#   --slo-use-detokenize-time

$RUST_BINARY \
  --trace SLO-CSim/trace/arxiv/lmsys.csv \
  --text-file /sgl-workspace/sglang/slo/text/enwik8 \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://0.0.0.0:40010/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --rate 450 \
  --max-requests 20000 \
  --log-path /sgl-workspace/sglang/slo/logs/rust_client_output.jsonl \
  --elapsed-dump-path /sgl-workspace/sglang/slo/logs/rust_elapsed_timelines.pkl \
  --ans-log-path /sgl-workspace/sglang/slo/logs/rust_client.ans \
  --slo-use-detokenize-time