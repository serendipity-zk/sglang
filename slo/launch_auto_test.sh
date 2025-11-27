#!/usr/bin/env bash
set -euo pipefail

resume_flag=()
if [[ $# -gt 0 && -n "${1:-}" ]]; then
  resume_flag=(--resume-folder "$1")
fi

python -m auto_rate_slog \
  --trace /sgl-workspace/sglang/slo/trace/1024_1024_152540.csv \
  --text-file /sgl-workspace/sglang/slo/text/enwik8 \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://0.0.0.0:40010/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --start-rate 20.0 \
  --target-attainment 0.99 \
  --num-workers 64 \
  --concurrency-per-worker 500 \
  --max-requests 2000 \
  "${resume_flag[@]}"
