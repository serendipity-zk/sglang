cd /sgl-workspace/sglang/sgl-router
pip install .
cd /sgl-workspace/sglang/slo
PYTHONPATH=/sgl-workspace/sglang/sgl-router/py_src \
TOKIO_WORKER_THREADS=64 \
python -m sglang_router.launch_router \
  --host 0.0.0.0 --port 40010 \
  --policy gated_round_robin \
  --log-dir /sgl-workspace/sglang/slo/logs \
  --max-concurrent-requests 8192 \
  --rate-limit-tokens-per-second 600 \
  --worker-urls \
    http://0.0.0.0:31001 \
    http://0.0.0.0:31002 \
    http://0.0.0.0:31003 \
    http://0.0.0.0:31004 \
    http://0.0.0.0:31005 \
    http://0.0.0.0:31006 \
    http://0.0.0.0:31007 \
    http://0.0.0.0:31008 \
  --scheduler-config-file /sgl-workspace/sglang/slo/scheduler_config_slo_aware_ttft_promote.json