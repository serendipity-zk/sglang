#!/bin/bash
# Launch server with SLO scheduler sidecar.
# Usage: bash launch_server_with_sidecar.sh [shadow|shadow-sidecar|sidecar]
#   shadow          - (default) both internal and sidecar run, decisions logged for comparison
#   shadow-sidecar  - sidecar decisions applied, both stats logged for comparison
#   sidecar         - sidecar decisions applied, internal runs for metrics/fallback

MODE="${1:-shadow}"
if [[ "$MODE" != "shadow" && "$MODE" != "shadow-sidecar" && "$MODE" != "sidecar" ]]; then
  echo "Usage: $0 [shadow|shadow-sidecar|sidecar]"
  echo "  shadow          - log both decisions for comparison (default)"
  echo "  shadow-sidecar  - sidecar authoritative, both stats logged"
  echo "  sidecar         - apply sidecar decisions"
  exit 1
fi

# Two independent axes derived from MODE:
#   --sidecar-mode (engine: who makes scheduling decisions)
#   --stats-mode   (router: whose stats to trust for routing)
# shadow-sidecar uses sidecar decisions (engine axis=sidecar) but logs both stats (router axis=shadow-sidecar)
if [[ "$MODE" == "shadow-sidecar" ]]; then
  SIDECAR_MODE="sidecar"
else
  SIDECAR_MODE="$MODE"
fi
STATS_MODE="$MODE"
echo "Starting: sidecar-mode=$SIDECAR_MODE, stats-mode=$STATS_MODE"

# Cleanup stale IPC sockets
rm -f /tmp/sglang_slo_scheduler_0.sock /tmp/sglang_slo_scheduler_1.sock /tmp/sglang_slo_scheduler_2.sock /tmp/sglang_slo_scheduler_3.sock /tmp/sglang_slo_scheduler_4.sock /tmp/sglang_slo_scheduler_5.sock /tmp/sglang_slo_scheduler_6.sock /tmp/sglang_slo_scheduler_7.sock 2>/dev/null

# Cleanup stale shadow decision logs
rm -f shadow_decisions_0.0.0.0:31001.jsonl shadow_decisions_0.0.0.0:31002.jsonl shadow_decisions_0.0.0.0:31003.jsonl shadow_decisions_0.0.0.0:31004.jsonl shadow_decisions_0.0.0.0:31005.jsonl shadow_decisions_0.0.0.0:31006.jsonl shadow_decisions_0.0.0.0:31007.jsonl shadow_decisions_0.0.0.0:31008.jsonl 2>/dev/null

# NOTE: When using --stats-mode shadow or sidecar, add these args to router launch:
#   --sidecar-urls http://0.0.0.0:18100 http://0.0.0.0:18101 ... http://0.0.0.0:18107
#   --stats-mode $MODE
# The launch_server.py script will print the exact args to use.

python launch_server.py \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --ports 31001,31002,31003,31004,31005,31006,31007,31008 \
  --gpus 0,1,2,3,4,5,6,7 \
  --log-dir /sgl-workspace/sglang/slo/logs \
  --predictor-log-dir /sgl-workspace/sglang/slo/logs/predictor \
  --extra-worker-args " --prefill-schedule-mode simulation --enable-mixed-chunk --chunked-prefill-size 4096 --router-metrics-url http://0.0.0.0:40010 --enable-iteration-metrics --iteration-metrics-interval 1 --predictor-type mode_aware --predictor-grid-path /sgl-workspace/sglang/sglang_profile/mode_3d.json" \
  --with-sidecar \
  --sidecar-mode "$SIDECAR_MODE" \
  --stats-mode "$STATS_MODE" \
  --tmux-ui \
