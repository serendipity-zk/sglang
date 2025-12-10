#!/bin/bash
# Niyama (SGLang Slack) router - round robin policy

LOG_DIR="${EVAL_LOG_DIR:-/sgl-workspace/sglang/slo/logs}"

cd /sgl-workspace/sglang/sgl-router
pip install .
cd /sgl-workspace/sglang/slo
PYTHONPATH=/sgl-workspace/sglang/sgl-router/py_src \
TOKIO_WORKER_THREADS=64 \
python -m sglang_router.launch_router \
  --host 0.0.0.0 --port 40010 \
  --policy round_robin \
  --log-dir "$LOG_DIR" \
  --max-concurrent-requests 8192 \
  --rate-limit-tokens-per-second 600 \
  --worker-urls \
    http://0.0.0.0:31001 \
    http://0.0.0.0:31002 \
    http://0.0.0.0:31003 \
    http://0.0.0.0:31004 \
    http://0.0.0.0:31005 \
    http://0.0.0.0:31006 \
    http://0.0.0.0:31007 \
    http://0.0.0.0:31008 \
  --scheduler-config-file /sgl-workspace/sglang/eval/eval_utils/baseline/sglang_slack/scheduler_config_eager.json
