Online Cycle Time Estimator (RLS, Linear, Online)

- Predict-first online learning with bounded memory. Linear model updated via Recursive Least Squares (RLS) with optional forgetting.
- Online normalization (running mean/std). Periodic logs show window MAE, P90/P99, and cumulative MAE.

Quick CLI
- Replay: `python cycle_time_est_online/frontend.py replay <log> [--forgetting 0.99] [--feature-preset hybrid5] [--epochs 2] [--predictions preds.json]`
- Perf: `python cycle_time_est_online/frontend.py perf <log> --warmup 1000 --max-records 5000 [--feature-preset hybrid5]`
- Interactive: `python cycle_time_est_online/frontend.py interactive [--forgetting 0.99] [--feature-preset basic]`

Feature Selection
- Presets: `--feature-preset {all,basic,hybrid5,prod_only}`
- Custom: `--feature-indices 0,1,8,16,17` (0-based; see `PredictionInput.to_features()` ordering).

Python API
- from cycle_time_est_online import OnlineLinearCycleTime
- `est = OnlineLinearCycleTime(forgetting=0.99, feature_preset="hybrid5")`
- `pred = est.predict(batch_size_tokens, prefill_chunk_pairs, kv_tokens_used)`
- `pred_before, abs_err = est.submit(batch_size_tokens, prefill_chunk_pairs, kv_tokens_used, iteration_time_ms)`

Notes
- `submit()` predicts before updating and logs every `log_every` samples.
- Optional bounded history is off by default; enable with `store_history=True` (max 100000).
