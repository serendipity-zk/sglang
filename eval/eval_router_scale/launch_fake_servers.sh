#!/bin/bash
# Launch multiple fake servers for SLO-aware scheduling testing
#
# Usage:
#   ./launch_fake_servers.sh [num_servers] [router_url] [base_port]
#
# Examples:
#   ./launch_fake_servers.sh                    # 8 servers, default router, ports 31001-31008
#   ./launch_fake_servers.sh 4                  # 4 servers
#   ./launch_fake_servers.sh 8 http://0.0.0.0:40010 31001

set -e

# Configuration
NUM_SERVERS=${1:-8}
ROUTER_URL=${2:-"http://0.0.0.0:40010"}
BASE_PORT=${3:-31001}
MODEL_NAME=${MODEL_NAME:-"fake/test-model"}
MAX_TOKENS_PER_SEC=${MAX_TOKENS_PER_SEC:-24576}
ITERATION_INTERVAL_MS=${ITERATION_INTERVAL_MS:-10}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Fake Server Launcher"
echo "=========================================="
echo "Servers:          $NUM_SERVERS"
echo "Router URL:       $ROUTER_URL"
echo "Base port:        $BASE_PORT"
echo "Model:            $MODEL_NAME"
echo "Tokens/sec:       $MAX_TOKENS_PER_SEC"
echo "Iteration (ms):   $ITERATION_INTERVAL_MS"
echo "=========================================="

# Create logs directory (use env var if set, otherwise default)
LOG_DIR="${SERVER_LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_DIR"

# Array to store PIDs
PIDS=()

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down fake servers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Wait a moment then force kill if needed
    sleep 1
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo "All servers stopped."
    exit 0
}

# Register cleanup handler
trap cleanup SIGINT SIGTERM

# Launch servers
for i in $(seq 1 $NUM_SERVERS); do
    PORT=$((BASE_PORT + i - 1))
    WORKER_ID="http://0.0.0.0:$PORT"
    LOG_FILE="$LOG_DIR/fake_server_$PORT.log"

    echo "Starting fake server $i on port $PORT..."

    python "$SCRIPT_DIR/fake_server.py" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --router-url "$ROUTER_URL" \
        --worker-id "$WORKER_ID" \
        --model-name "$MODEL_NAME" \
        --max-tokens-per-second "$MAX_TOKENS_PER_SEC" \
        --iteration-interval-ms "$ITERATION_INTERVAL_MS" \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    PIDS+=($PID)
    echo "  PID: $PID, Log: $LOG_FILE"
done

echo ""
echo "=========================================="
echo "All $NUM_SERVERS fake servers started!"
echo "Ports: $BASE_PORT - $((BASE_PORT + NUM_SERVERS - 1))"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "=========================================="

# Wait for all servers
wait
