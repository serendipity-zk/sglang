#!/usr/bin/env bash
set -euo pipefail

# Launch auto rate slog using the high-performance Rust client backend
#
# Key differences from launch_auto_test.sh:
# - Uses Rust streaming client instead of Python aiohttp
# - Lower CPU overhead (compiled Rust vs interpreted Python)
# - Simpler concurrency model (async/await vs multiprocessing)
#
# Usage:
#   ./launch_auto_test_rust.sh                    # Start fresh
#   ./launch_auto_test_rust.sh <resume_folder>    # Resume existing run

resume_flag=()
if [[ $# -gt 0 && -n "${1:-}" ]]; then
  resume_flag=(--resume-folder "$1")
fi

python -m auto_rate_slog_rust \
  --trace /sgl-workspace/sglang/SLO-CSim/trace/arxiv/sharegpt.csv \
  --text-file /sgl-workspace/sglang/slo/text/enwik8 \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://0.0.0.0:40010/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --start-rate 50.0 \
  --target-attainment 0.95 \
  --max-requests 5000 \
  "${resume_flag[@]}"
