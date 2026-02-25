#!/bin/bash
# Launch router with sidecar awareness (TPOT routing + stats mode).
# Usage: bash launch_router_with_sidecar.sh [shadow|shadow-sidecar|sidecar]
#   shadow          - (default) engine stats for routing; sidecar stats logged to shadow_stats.jsonl
#   shadow-sidecar  - sidecar stats for routing; engine stats logged to shadow_stats.jsonl
#   sidecar         - sidecar stats for routing; engine stats suppressed (internal SLO disabled)

MODE="${1:-shadow}"
if [[ "$MODE" != "shadow" && "$MODE" != "shadow-sidecar" && "$MODE" != "sidecar" ]]; then
  echo "Usage: $0 [shadow|shadow-sidecar|sidecar]"
  echo "  shadow          - engine authoritative, sidecar logged for comparison (default)"
  echo "  shadow-sidecar  - sidecar authoritative, engine logged for comparison"
  echo "  sidecar         - sidecar authoritative, engine stats ignored"
  exit 1
fi
echo "Starting router with stats mode: $MODE"

cd /sgl-workspace/sglang/slo

PYTHONPATH=/sgl-workspace/sglang/sgl-router/py_src \
TOKIO_WORKER_THREADS=64 \
python -m sglang_router.launch_router \
  --host 0.0.0.0 --port 40010 \
  --policy gated_round_robin \
  --log-dir /sgl-workspace/sglang/slo/logs \
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
  --scheduler-config-file /sgl-workspace/sglang/slo/scheduler_config_slo_aware_ttft_promote.json \
  --sidecar-urls \
    http://0.0.0.0:18100 \
    http://0.0.0.0:18101 \
    http://0.0.0.0:18102 \
    http://0.0.0.0:18103 \
    http://0.0.0.0:18104 \
    http://0.0.0.0:18105 \
    http://0.0.0.0:18106 \
    http://0.0.0.0:18107 \
  --stats-mode "$MODE"
