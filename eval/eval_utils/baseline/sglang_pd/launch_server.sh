#!/bin/bash
# SGLang-PD: Prefill-Decode Disaggregation (6 prefill + 2 decode servers)

LOG_DIR="${EVAL_LOG_DIR:-/sgl-workspace/sglang/slo/logs}"
PRED_LOG_DIR="${EVAL_PRED_LOG_DIR:-/sgl-workspace/sglang/slo/logs/predictor}"

MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

mkdir -p "$LOG_DIR" "$PRED_LOG_DIR"

# Find active IB device for RDMA transfer
find_active_ib_device() {
    for device in mlx5_{0..11}; do
        if ibv_devinfo $device >/dev/null 2>&1; then
            state=$(ibv_devinfo $device | grep "state:" | head -1 | awk '{print $2}')
            if [[ "$state" == "PORT_ACTIVE" ]]; then
                echo "$device"
                return 0
            fi
        fi
    done
    echo "mooncake"  # fallback
    return 0
}

IB_DEVICE=$(find_active_ib_device)
echo "Using IB device: $IB_DEVICE"
echo "Log directory: $LOG_DIR"
echo "Predictor log directory: $PRED_LOG_DIR"

echo "========================================="
echo "Launching 6 PREFILL servers on GPUs 0-5"
echo "========================================="

# Launch prefill servers on GPUs 0-5 (ports 31001-31006, bootstrap ports 9001-9006)
for i in {0..5}; do
    PORT=$((31001 + i))
    BOOTSTRAP_PORT=$((9001 + i))
    GPU=$i
    HOST="127.0.0.$((i + 1))"
    LOG_FILE="$LOG_DIR/worker_${i}_gpu${GPU}_p${PORT}.ans"
    PRED_FILE="$PRED_LOG_DIR/predictor_w${i}_gpu${GPU}_p${PORT}.csv"

    echo "PREFILL $((i+1)): GPU $GPU at $HOST:$PORT (bootstrap: $BOOTSTRAP_PORT)"

    CUDA_VISIBLE_DEVICES=$GPU \
    python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend nixl \
        --disaggregation-ib-device "$IB_DEVICE" \
        --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
        --predictor-log-path "$PRED_FILE" \
        > "$LOG_FILE" 2>&1 &
done

echo ""
echo "========================================="
echo "Launching 2 DECODE servers on GPUs 6-7"
echo "========================================="

# Launch decode servers on GPUs 6-7 (ports 31007-31008)
for i in {0..1}; do
    GPU=$((6 + i))
    PORT=$((31007 + i))
    HOST="127.0.0.$((7 + i))"
    IDX=$((6 + i))
    LOG_FILE="$LOG_DIR/worker_${IDX}_gpu${GPU}_p${PORT}.ans"
    PRED_FILE="$PRED_LOG_DIR/predictor_w${IDX}_gpu${GPU}_p${PORT}.csv"

    echo "DECODE $((i+1)): GPU $GPU at $HOST:$PORT"

    CUDA_VISIBLE_DEVICES=$GPU \
    python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend nixl \
        --disaggregation-ib-device "$IB_DEVICE" \
        --base-gpu-id 0 \
        --predictor-log-path "$PRED_FILE" \
        > "$LOG_FILE" 2>&1 &
done

echo ""
echo "========================================="
echo "Waiting for all 8 servers to be healthy"
echo "========================================="

# Health check with timeout
TIMEOUT=300
START_TIME=$(date +%s)

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "Timeout: Servers did not become healthy within 5 minutes"
        exit 1
    fi

    HEALTHY_PREFILL=0
    HEALTHY_DECODE=0

    # Check prefill servers (127.0.0.1-6:31001-31006)
    for i in {1..6}; do
        if curl -s -f "http://127.0.0.$i:$((31000 + i))/health" >/dev/null 2>&1; then
            HEALTHY_PREFILL=$((HEALTHY_PREFILL + 1))
        fi
    done

    # Check decode servers (127.0.0.7-8:31007-31008)
    for i in {7..8}; do
        if curl -s -f "http://127.0.0.$i:$((31000 + i))/health" >/dev/null 2>&1; then
            HEALTHY_DECODE=$((HEALTHY_DECODE + 1))
        fi
    done

    echo "Healthy: Prefill=$HEALTHY_PREFILL/6, Decode=$HEALTHY_DECODE/2 (elapsed: ${ELAPSED}s)"

    if [ $HEALTHY_PREFILL -eq 6 ] && [ $HEALTHY_DECODE -eq 2 ]; then
        echo ""
        echo "All 8 servers are healthy!"
        echo ""
        echo "Server endpoints:"
        echo "  Prefill: http://127.0.0.1:31001, http://127.0.0.2:31002, http://127.0.0.3:31003, http://127.0.0.4:31004, http://127.0.0.5:31005, http://127.0.0.6:31006"
        echo "  Decode:  http://127.0.0.7:31007, http://127.0.0.8:31008"
        echo ""
        echo "Now launch the router with: ./launch_router.sh"
        break
    fi

    sleep 10
done

echo ""
echo "Servers running. Press Ctrl+C to stop all."
wait
