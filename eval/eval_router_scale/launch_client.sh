#!/bin/bash
# Launch Rust client for eval_router_scale testing
#
# Usage: ./launch_client.sh [rate] [max_requests] [output_dir] [trace_file]
#
# Examples:
#   ./launch_client.sh 100 1000                     # Rate 100, 1000 requests
#   ./launch_client.sh 200 5000 ./logs/client_log   # Custom output dir

set -e

RATE=${1:-100}
MAX_REQUESTS=${2:-1000}
OUTPUT_DIR=${3:-"./logs/client_log"}
TRACE_FILE=${4:-"/sgl-workspace/sglang/SLO-CSim/trace/arxiv/uniform_512_512.csv"}
BASE_URL=${BASE_URL:-"http://0.0.0.0:40010/v1"}
MODEL=${MODEL:-"fake/test-model"}
TEXT_FILE=${TEXT_FILE:-"/sgl-workspace/sglang/slo/text/enwik8"}
TOKENIZER=${TOKENIZER:-"meta-llama/Llama-3.1-8B-Instruct"}

RUST_CLIENT_DIR="/sgl-workspace/sglang/slo/rust_client"
RUST_BINARY="$RUST_CLIENT_DIR/target/release/slo_runner"

echo "=========================================="
echo "Client Launcher"
echo "=========================================="
echo "Rate:         $RATE"
echo "Max requests: $MAX_REQUESTS"
echo "Output dir:   $OUTPUT_DIR"
echo "Trace file:   $TRACE_FILE"
echo "Base URL:     $BASE_URL"
echo "Model:        $MODEL"
echo "=========================================="

# Build Rust client if needed
if [ ! -f "$RUST_BINARY" ]; then
    echo "Building Rust client..."
    cd "$RUST_CLIENT_DIR"
    cargo build --release
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run client
$RUST_BINARY \
    --trace "$TRACE_FILE" \
    --text-file "$TEXT_FILE" \
    --tokenizer "$TOKENIZER" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --rate "$RATE" \
    --max-requests "$MAX_REQUESTS" \
    --log-path "$OUTPUT_DIR/client_output.jsonl" \
    --ans-log-path "$OUTPUT_DIR/client.ans"
