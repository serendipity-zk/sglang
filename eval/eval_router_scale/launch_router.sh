#!/bin/bash
# Launch router for eval_router_scale testing
#
# Usage: ./launch_router.sh [num_workers] [port] [log_dir] [scheduler_config] [prometheus_port]
#
# Examples:
#   ./launch_router.sh 8                    # 8 workers, default settings
#   ./launch_router.sh 4 40010 ./logs       # 4 workers, custom port and log dir

set -e

NUM_WORKERS=${1:-8}
ROUTER_PORT=${2:-40010}
LOG_DIR=${3:-"./logs/router_log"}
SCHEDULER_CONFIG=${4:-""}
PROMETHEUS_PORT=${5:-29000}
BASE_WORKER_PORT=${BASE_WORKER_PORT:-31001}

echo "=========================================="
echo "Router Launcher"
echo "=========================================="
echo "Workers:          $NUM_WORKERS"
echo "Router port:      $ROUTER_PORT"
echo "Log dir:          $LOG_DIR"
echo "Base worker port: $BASE_WORKER_PORT"
echo "=========================================="

# Create log directory
mkdir -p "$LOG_DIR"

# Build worker URLs dynamically
WORKER_URLS=""
for i in $(seq 1 $NUM_WORKERS); do
    PORT=$((BASE_WORKER_PORT + i - 1))
    WORKER_URLS="$WORKER_URLS http://0.0.0.0:$PORT"
done

echo "Worker URLs: $WORKER_URLS"

# Install sgl-router if needed
cd /sgl-workspace/sglang/sgl-router
pip install . -q 2>/dev/null || pip install .

# Launch router
PYTHONPATH=/sgl-workspace/sglang/sgl-router/py_src \
TOKIO_WORKER_THREADS=64 \
python -m sglang_router.launch_router \
    --host 0.0.0.0 \
    --port $ROUTER_PORT \
    --policy gated_round_robin \
    --log-dir "$LOG_DIR" \
    --max-concurrent-requests 8192 \
    --queue-size 10000 \
    --prometheus-port $PROMETHEUS_PORT \
    --worker-urls $WORKER_URLS \
    ${SCHEDULER_CONFIG:+--scheduler-config-file "$SCHEDULER_CONFIG"}
