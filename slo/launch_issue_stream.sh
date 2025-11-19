#!/bin/bash

# Simple launcher for the streaming issue client

python3 -m issue_stream \
  --trace /sgl-workspace/sglang/slo/trace/1024_1024_152540.csv \
  --text-file /sgl-workspace/sglang/slo/text/enwik8 \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://0.0.0.0:40000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --rate 20.0 \
  --max-requests 0
