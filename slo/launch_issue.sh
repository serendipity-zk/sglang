python -m issue \
  --trace /sgl-workspace/sglang/slo/trace/short.csv \
  --text-file ./text/enwik8 \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://0.0.0.0:40000/v1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --rate 200.0 \
  --max-workers 2560