#!/bin/bash
# 8-GPU server for non-linear load evaluation
# Launches 8 independent SGLang servers (one per GPU) with iteration metrics enabled

LOG_DIR="${EVAL_LOG_DIR:-/sgl-workspace/sglang/eval/motivate_non_linear_load/results/worker_log}"

python /sgl-workspace/sglang/slo/launch_server.py \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --ports 31001,31002,31003,31004,31005,31006,31007,31008 \
  --gpus 0,1,2,3,4,5,6,7 \
  --log-dir "$LOG_DIR" \
  --extra-worker-args " --prefill-schedule-mode budget --enable-mixed-chunk --chunked-prefill-size 1024   --enable-iteration-metrics --iteration-metrics-interval 1"
