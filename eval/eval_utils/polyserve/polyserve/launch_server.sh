#!/bin/bash
# PolyServe: 8 workers with predictor-based scheduling

LOG_DIR="${EVAL_LOG_DIR:-/sgl-workspace/sglang/slo/logs}"
PRED_LOG_DIR="${EVAL_PRED_LOG_DIR:-/sgl-workspace/sglang/slo/logs/predictor}"

mkdir -p "$LOG_DIR" "$PRED_LOG_DIR"

python /sgl-workspace/sglang/slo/launch_server.py \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --ports 31001,31002,31003,31004,31005,31006,31007,31008 \
  --gpus 0,1,2,3,4,5,6,7 \
  --log-dir "$LOG_DIR" \
  --predictor-log-dir "$PRED_LOG_DIR" \
  --extra-worker-args " --prefill-schedule-mode simulation --enable-mixed-chunk --chunked-prefill-size 4096 --router-metrics-url http://0.0.0.0:40010 --enable-iteration-metrics --iteration-metrics-interval 1 --predictor-type mode_aware --predictor-grid-path /sgl-workspace/sglang/sglang_profile/mode_3d.json" \
  --tmux-ui \