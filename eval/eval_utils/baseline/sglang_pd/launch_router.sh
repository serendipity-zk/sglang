#!/bin/bash
# Launch router with PD disaggregation mode (6 prefill + 2 decode)

LOG_DIR="${EVAL_LOG_DIR:-/sgl-workspace/sglang/slo/logs}"

cd /sgl-workspace/sglang/sgl-router
pip install .

cd /sgl-workspace/sglang/slo

PYTHONPATH=/sgl-workspace/sglang/sgl-router/py_src \
TOKIO_WORKER_THREADS=64 \
python -m sglang_router.launch_router \
  --host 0.0.0.0 --port 40010 \
  --pd-disaggregation \
  --policy round_robin \
  --log-dir "$LOG_DIR" \
  --max-concurrent-requests 8192 \
  --rate-limit-tokens-per-second 600 \
  --prefill http://127.0.0.1:31001 9001 \
  --prefill http://127.0.0.2:31002 9002 \
  --prefill http://127.0.0.3:31003 9003 \
  --prefill http://127.0.0.4:31004 9004 \
  --prefill http://127.0.0.5:31005 9005 \
  --prefill http://127.0.0.6:31006 9006 \
  --decode http://127.0.0.7:31007 \
  --decode http://127.0.0.8:31008 \
  --scheduler-config-file /sgl-workspace/sglang/eval/eval_utils/baseline/sglang_pd/scheduler_config_eager.json
