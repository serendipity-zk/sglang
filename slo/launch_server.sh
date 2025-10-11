python launch_server.py \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --ports 31001,31002,31003,31004 \
  --gpus 0,1,2,3 \
  --log-dir /sgl-workspace/sglang/slo/logs \
  --extra-worker-args "  --enable-mixed-chunk --chunked-prefill-size 4096 --router-metrics-url http://0.0.0.0:40000 --enable-iteration-metrics --iteration-metrics-interval 1" \
  --tmux-ui \
  
