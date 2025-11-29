#!/bin/bash

# Launcher for the high-performance aiohttp + multiprocessing streaming issue client
#
# Key features:
# - Low-latency streaming with aiohttp readany() (minimal buffering)
# - Multiprocessing to utilize all CPU cores (bypasses GIL)
# - Configurable worker processes and concurrency per worker
#
# Expected improvements over launch_issue_stream.sh:
# - Inter-token intervals: ~4-5ms vs 30-80ms
# - Higher throughput: supports thousands of concurrent requests
# - Better CPU utilization across all cores

# Auto-detect number of workers (defaults to min(256, cpu_count))
# Override with --num-workers N if needed

python3 -m issue_stream_aiohttp \
  --trace /sgl-workspace/sglang/slo/trace/1024_1024_152540.csv \
  --text-file /sgl-workspace/sglang/slo/text/enwik8 \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://0.0.0.0:40010/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --rate 40.0 \
  --max-requests 0 \
  --num-workers 64 \
  --concurrency-per-worker 500
