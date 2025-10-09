python launch_server.py \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --ports 31001,31002,31003,31004 \
  --gpus 0,1,2,3 \
  --log-dir /sgl-workspace/sglang/slo/logs \
  --tmux-ui
