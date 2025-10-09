Server UI Metrics: What They Mean and Where They Come From

This document explains each metric shown by the per‑server TUI (`slo/server_ui.py`), how it is computed, and where it is collected in the codebase. Use this as a maintenance guide and a map for adding new metrics.

Overview
- Endpoint: The UI polls `GET /ui_stats` from the Python server to fetch a compact JSON of live metrics.
- Renderer: `slo/server_ui.py` redraws the terminal with anti‑flicker and cached stale states.
- Sources: Metrics are aggregated from lightweight admission counters, scheduler internals, and the model runner boundary.

Files Involved
- UI client: `slo/server_ui.py`
- UI stats endpoint: `python/sglang/srt/entrypoints/http_server.py` (`/ui_stats`)
- Lightweight server counters: `python/sglang/srt/ui/server_ui.py`
- Admission hook: `python/sglang/srt/managers/tokenizer_manager.py` (`TokenizerManager.generate_request`)
- Scheduler internals and metrics: `python/sglang/srt/managers/scheduler.py` (+ `scheduler_metrics_mixin.py`)
- Model runner boundary: `python/sglang/srt/model_executor/model_runner.py`

Metrics, Semantics, and Collection Path

1) Accepted Requests
- Meaning: Total accepted requests at admission (batch‑expanded), excluding health checks.
- Where:
  - Counter store: `python/sglang/srt/ui/server_ui.py` (`inc_accepted`, `snapshot`).
  - Increment site: `python/sglang/srt/managers/tokenizer_manager.py::TokenizerManager.generate_request` — increments only when `obj.log_metrics` is True.
- Exposed: `/ui_stats` copies from the counter store.

2) Last Batch Size
- Meaning: Last observed request batch size at admission (not the engine’s micro‑batch). Quick sanity signal.
- Where:
  - Set at admission: `server_ui.set_last_batch_size(obj.batch_size)`.
  - Stored in `python/sglang/srt/ui/server_ui.py`.
- Exposed: `/ui_stats` copies from the counter store.

3) Running Batch Size
- Meaning: Current number of decoding sequences executing (aka “#running‑req”), aggregated across DP ranks.
- Where:
  - Source: `get_load()` returns `num_reqs` and `num_waiting_reqs` per DP rank from the scheduler.
  - Compute `running = num_reqs - num_waiting_reqs`, sum across ranks.
  - Files: `python/sglang/srt/managers/tokenizer_communicator_mixin.py` (`get_load`), `python/sglang/srt/managers/scheduler.py` (`get_load`).
- Exposed: `/ui_stats` emits `running_batch_size`.

4) Queue Reqs
- Meaning: Total number of waiting requests across DP ranks.
- Where: Sum `num_waiting_reqs` from `get_load()` per rank.
- Exposed: `/ui_stats` emits `queue_reqs`.

5) KV Tokens Used and Token Capacity
- Meaning: Actual KV cache tokens currently allocated (used by attention), and total capacity, across DP ranks.
- Where (per‑rank occupancy): `python/sglang/srt/managers/scheduler.py::get_internal_state` sets `kv_tokens_used` using:
  - Non‑hybrid: `_get_token_info()` → `used = capacity - (available + evictable)`.
  - Hybrid (SWA): `_get_swa_token_info()` → `max(full_used, swa_used)`.
- Where (capacity): `memory_usage.token_capacity` from the scheduler.
- Aggregation: `/ui_stats` sums `kv_tokens_used` and `token_capacity` across DP ranks; computes `kv_usage_pct`.
- Exposed: `kv_tokens_used`, `token_capacity`, `kv_usage_pct`.
- UI: “KV Tokens: <used_k>k / <cap_k>k”.

6) Generate Throughput (tok/s)
- Meaning: Recent generation throughput over the scheduler’s decode stats window.
- Where:
  - Per rank: `python/sglang/srt/managers/scheduler_metrics_mixin.py::log_decode_stats` updates `self.last_gen_throughput`.
  - Exposed per rank via `scheduler.get_internal_state` → `last_gen_throughput`.
  - Aggregation: `/ui_stats` sums `last_gen_throughput` across DP ranks.
- Exposed: `gen_throughput_tps`.

7) Prefill Tokens
- Meaning: New prompt tokens admitted in the most recent prefill step, aggregated across DP ranks.
- Where:
  - Per rank: `scheduler_metrics_mixin.py::log_prefill_stats` sets `self.last_prefill_tokens`.
  - Exposed per rank: `scheduler.get_internal_state` includes `last_prefill_tokens`.
  - Aggregation/Mode logic: `/ui_stats`:
    - If `enable_mixed_chunk` is True: show both prefill and decode simultaneously (combined pass possible).
    - Else: compare `last_prefill_tic` vs `last_decode_tic` and show only the most recent step’s tokens (no overlap).
- Exposed: `prefill_tokens`.

8) Decode Tokens
- Meaning: Activated decode tokens for the most recent step. In decode steps this equals the number of active decode sequences (one token per sequence). In mixed chunk mode, both prefill and decode tokens can be non‑zero.
- Where:
  - Per rank proxy: `num_running_reqs` (from scheduler) represents concurrent decode sequences.
  - Exposed per rank: `scheduler.get_internal_state` includes `num_running_reqs` and timestamps.
  - Aggregation/Mode logic: `/ui_stats` sums `num_running_reqs` subject to `enable_mixed_chunk` and recency rules noted above.
- Exposed: `decode_tokens`.

9) Token Batch Size
- Meaning: `prefill_tokens + decode_tokens` as a recent load proxy.
  - Mixed on: often matches a single pass input size.
  - Mixed off: cross‑step proxy (not a single pass M).
- Where: `/ui_stats` computes `token_batch_size = prefill_tokens + decode_tokens`.

10) Input Tokens (Runner‑Derived)
- Meaning: Exact number of tokens fed into embedding/attention for the most recent forward pass (per server, aggregated across DP ranks).
- Why: Mode‑agnostic, the most fundamental M of the current pass; independent of logging hooks.
- Where:
  - At runner boundary: `python/sglang/srt/model_executor/model_runner.py::_forward_raw` records:
    - Prefer `ForwardBatch.global_num_tokens_cpu` (sum across partitions) when available.
    - Else if decode: `ForwardBatch.batch_size` (one token per sequence).
    - Else if prefill/split: `ForwardBatch.extend_num_tokens` if set; fallback to `batch_size`.
    - Sets `last_input_tokens`, `last_input_step_type`, `last_input_tic` on `ModelRunner`.
  - Exposed per rank: `python/sglang/srt/managers/scheduler.py::get_internal_state` reads from `model_runner` and emits `input_tokens`, `input_step_type`, `last_input_tic`.
  - Aggregation: `/ui_stats` sums `input_tokens` across DP ranks and emits `input_tokens` (and `last_input_tic` for recency if needed).
- UI: “Input Tokens: N” (integer below 1000; k‑format above).

11) Model and PID
- Meaning: Convenience identifiers for the pane.
- Where: `/ui_stats` sets `model` from `TokenizerManager.served_model_name`, and `pid` from `os.getpid()`.

Demand vs Occupancy Notes
- `used_tokens` from `get_load().num_tokens` may include waiting/prealloc tokens (demand). The UI relies on `kv_tokens_used` + `token_capacity` to reflect actual KV occupancy (allocated tokens used by attention), which never exceeds capacity.

Mixed vs Non‑Mixed Chunking
- Mixed on (`server_args.enable_mixed_chunk=True`): Prefill and Decode tokens can both be non‑zero for the same pass; Token Batch often matches the single GEMM input size.
- Mixed off: Prefill and Decode occur in separate micro‑steps. `/ui_stats` uses recency (last prefill vs last decode timestamp) so only one component is non‑zero. `Input Tokens` always reflects the most recent pass exactly.

UI Rendering Behavior
- `slo/server_ui.py` fetches stats first, then redraws with one write to reduce flicker; it hides the cursor during updates and shows a [STALE] tag if the last fetch failed (shows cached stats instead of blanking).
- Small values (<1000) render as integers; larger values use `k` with one decimal.

Quick Pointers (by function/class)
- UI client: `slo/server_ui.py`.
- UI endpoint: `python/sglang/srt/entrypoints/http_server.py` (`ui_stats`).
- Counters at admission: `python/sglang/srt/ui/server_ui.py` and `python/sglang/srt/managers/tokenizer_manager.py::generate_request`.
- Scheduler load: `python/sglang/srt/managers/tokenizer_communicator_mixin.py::get_load`, `python/sglang/srt/managers/scheduler.py::get_load`.
- Scheduler internal state: `python/sglang/srt/managers/scheduler.py::get_internal_state`.
- Scheduler metrics hooks: `python/sglang/srt/managers/scheduler_metrics_mixin.py::log_prefill_stats`, `::log_decode_stats`.
- Runner boundary metric: `python/sglang/srt/model_executor/model_runner.py::_forward_raw`.

How to Extend
- Add new counters at the closest source of truth.
- Expose via `get_internal_state()` (per‑rank), aggregate in `/ui_stats`, and render in `slo/server_ui.py`.
- Keep updates lightweight; never allow UI bookkeeping to affect serving.

