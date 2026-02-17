# Cleanup stale IPC sockets
rm -f /tmp/sglang_slo_scheduler_0.sock /tmp/sglang_slo_scheduler_1.sock /tmp/sglang_slo_scheduler_2.sock /tmp/sglang_slo_scheduler_3.sock /tmp/sglang_slo_scheduler_4.sock /tmp/sglang_slo_scheduler_5.sock /tmp/sglang_slo_scheduler_6.sock /tmp/sglang_slo_scheduler_7.sock 2>/dev/null

# Cleanup stale shadow decision logs
rm -f shadow_decisions_0.0.0.0:31001.jsonl shadow_decisions_0.0.0.0:31002.jsonl shadow_decisions_0.0.0.0:31003.jsonl shadow_decisions_0.0.0.0:31004.jsonl shadow_decisions_0.0.0.0:31005.jsonl shadow_decisions_0.0.0.0:31006.jsonl shadow_decisions_0.0.0.0:31007.jsonl shadow_decisions_0.0.0.0:31008.jsonl 2>/dev/null

python launch_server.py \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --ports 31001,31002,31003,31004,31005,31006,31007,31008 \
  --gpus 0,1,2,3,4,5,6,7 \
  --log-dir /sgl-workspace/sglang/slo/logs \
  --predictor-log-dir /sgl-workspace/sglang/slo/logs/predictor \
  --extra-worker-args " --prefill-schedule-mode simulation --enable-mixed-chunk --chunked-prefill-size 4096 --router-metrics-url http://0.0.0.0:40010 --enable-iteration-metrics --iteration-metrics-interval 1 --predictor-type mode_aware --predictor-grid-path /sgl-workspace/sglang/sglang_profile/mode_3d.json" \
  --with-sidecar \
  --sidecar-mode shadow \
  --tmux-ui \
