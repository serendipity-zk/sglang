#!/bin/bash

# Simple launcher for the streaming issue client

python3 -m issue_stream \
  --trace /sgl-workspace/sglang/slo/trace/uniform_4096_1024.csv \
  --text-file /sgl-workspace/sglang/slo/enwik8.zip \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://0.0.0.0:40000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --rate 1.0 \
  --max-requests 0
