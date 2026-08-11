# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""VibeSim alignment records, emitted in the same schema the vLLM fork emits.

The alignment report compares a measured serving run against a VibeSim
simulation of the same workload. It reads four record kinds, each one JSON
object on a `VibeSimAlignment<Kind> {...}` log line:

* `Iteration` (v2) -- per forward pass: the batch geometry the simulator has to
  reproduce, plus a scheduler-side cadence that does not depend on any profiler
  being attached.
* `RequestTiming` (v2) -- per finished request, scheduler-side: TTFT split into
  queue wait and first-schedule-to-first-token, decode span, and TPOT.
* `ApiRequestTiming` (v3) -- per finished request, API-server-side: the frontend
  span the client actually waits on, which the scheduler never sees.
* `ExpertLoad` (v2) -- per EPLB rebalance: routed load per logical expert.

**The schema is vLLM's**, deliberately, down to the field names: one analyzer
reads both engines, and a field that means something slightly different under
one of them would be worse than a field that is absent. Where SGLang has no
equivalent the field is omitted rather than filled with a lookalike, and
`input_adapter` says which engine produced the row.

Two structural differences from the vLLM fork are worth knowing:

* SGLang already carries `ScheduleBatch.forward_iter`, so there is no need to
  add a dispatch-order index -- that field *is* vLLM's
  `alignment_iteration_index`, and the NVTX ranges key off the same value.
* SGLang's `ReqTimeStats` is already split by process (`APIServerReqTimeStats`
  vs `SchedulerReqTimeStats`) and calibrates the two clocks against each other.
  The vLLM fork instead refuses to subtract across domains. Both records here
  stay within one domain anyway, so the two engines' numbers stay comparable
  without relying on that calibration being exact.

Every builder below is a pure function over plain numbers so the arithmetic is
testable without a GPU, a scheduler, or a server.
"""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Optional, Sequence

import msgspec

from sglang.srt.environ import envs
from sglang.srt.utils.nvtx_utils import (
    NVTX_SCHEDULER_ENABLED,
    _nvtx_module,
    profile_range,
)

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    return envs.SGLANG_ENABLE_VIBESIM_ALIGNMENT.get()


class IterationGeometry(msgspec.Struct, omit_defaults=True):
    """A batch's shape, snapshotted before the forward pass mutates it.

    Taken at dispatch rather than at result-processing time because the decode
    path advances `seq_lens` during the forward: read afterwards, every context
    length would be one token too long, and in overlap mode the next batch may
    already be in flight.

    Not frozen only because `nvtx_range_id` is a handle to a range that is
    opened at dispatch and closed once the result is processed; the measured
    fields are written once at capture and never touched again.
    """

    launch_monotonic_ns: int = 0
    prefill_chunk_pairs: list[list[int]] = []
    decode_kv_lens: list[int] = []
    nvtx_range_id: Optional[tuple] = None


def capture_iteration_geometry(
    *,
    launch_monotonic_ns: int,
    is_extend: bool,
    prefix_lens: Optional[Sequence[int]],
    extend_lens: Optional[Sequence[int]],
    seq_lens: Optional[Sequence[int]],
) -> IterationGeometry:
    """Snapshot the axes attention cost depends on, for one batch.

    An extend batch contributes `(prefix, extend)` pairs and no decode lengths;
    a decode batch contributes context lengths and no pairs. SGLang keeps the
    two modes in separate batches, so a batch is one or the other -- unlike
    vLLM, where a single step can carry both and the record has to hold both
    lists at once. The record shape is vLLM's either way, so a reader does not
    need to know which engine produced it.
    """
    if is_extend:
        if prefix_lens is None or extend_lens is None:
            return IterationGeometry(launch_monotonic_ns=launch_monotonic_ns)
        pairs = [
            [int(prefix), int(extend)]
            for prefix, extend in zip(prefix_lens, extend_lens)
            if int(extend) > 0
        ]
        return IterationGeometry(
            launch_monotonic_ns=launch_monotonic_ns, prefill_chunk_pairs=pairs
        )

    if seq_lens is None:
        return IterationGeometry(launch_monotonic_ns=launch_monotonic_ns)
    return IterationGeometry(
        launch_monotonic_ns=launch_monotonic_ns,
        decode_kv_lens=[int(length) for length in seq_lens],
    )


#: Names the engine that produced a row. The analyzer keys per-engine quirks off
#: this rather than guessing from which fields happen to be present.
INPUT_ADAPTER = "sglang_text"

WORKER_SCHEMA_VERSION = 1
ITERATION_SCHEMA_VERSION = 2
REQUEST_TIMING_SCHEMA_VERSION = 2
EXPERT_LOAD_SCHEMA_VERSION = 2
API_REQUEST_TIMING_SCHEMA_VERSION = 3

_WORKER_RECORD_NAME = "VibeSimAlignmentWorker"
_ITERATION_RECORD_NAME = "VibeSimAlignmentIteration"
_REQUEST_TIMING_RECORD_NAME = "VibeSimAlignmentRequestTiming"
_API_REQUEST_TIMING_RECORD_NAME = "VibeSimAlignmentApiRequestTiming"
_EXPERT_LOAD_RECORD_NAME = "VibeSimAlignmentExpertLoad"


def _emit(record_name: str, record: dict[str, Any]) -> None:
    logger.info("%s %s", record_name, json.dumps(record, separators=(",", ":")))


def open_iteration_forward_range(iteration_index: Optional[int]) -> Optional[tuple]:
    """Open `sglang_iteration(N): forward`, to be closed after the result.

    A start/end pair rather than a `with` block because the span has to survive
    the return from `run_batch`: the forward it covers is only complete once the
    result is processed, and in overlap mode the next batch is dispatched before
    that happens. Overlapping spans are exactly what `start_range`/`end_range`
    are for -- a push/pop `annotate` would mis-nest them.
    """
    if not NVTX_SCHEDULER_ENABLED:
        return None
    return _nvtx_module.start_range(
        message=f"sglang_iteration({iteration_index}): forward", color="green"
    )


def close_iteration_forward_range(range_id: Optional[tuple]) -> None:
    if range_id is not None:
        _nvtx_module.end_range(range_id)


def iteration_profile_method(stage: str):
    """Name a span `sglang_iteration(N): <stage>`, mirroring the vLLM fork.

    The point of putting the index in the name is that an nsys range then joins
    to that iteration's records with no timestamp matching, which is what makes
    a measured GPU timeline comparable to a simulated one span by span.

    Only usable where `batch.forward_iter` is already assigned -- so not on
    entry to `run_batch`, which is the method that assigns it. The forward span
    is opened explicitly there instead, by `open_iteration_forward_range`.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, batch, *args: Any, **kwargs: Any):
            if not is_enabled():
                return func(self, batch, *args, **kwargs)
            with profile_range(
                f"sglang_iteration({batch.forward_iter}): {stage}",
                nvtx_enabled=NVTX_SCHEDULER_ENABLED,
            ):
                return func(self, batch, *args, **kwargs)

        return wrapper

    return decorator


# ── iteration ────────────────────────────────────────────────────────────────


def build_iteration_record(
    *,
    iteration_index: Optional[int],
    observed_start_monotonic_ns: int,
    observed_end_monotonic_ns: int,
    prefill_chunk_pairs: Sequence[Sequence[int]],
    decode_kv_lens: Sequence[int],
) -> dict[str, Any]:
    """Describe one forward pass by the geometry a simulator has to reproduce.

    `prefill_chunk_pairs` is `[already_computed, appended_this_step]` per
    prefill-carrying request -- SGLang's `(prefix_lens[i], extend_lens[i])`, and
    vLLM's `(num_computed_tokens, num_scheduled_tokens)`. The pair matters
    rather than the sum because attention cost depends on both the query length
    and the context it attends over.

    `decode_kv_lens` is the context length of each single-token request, which
    is the other axis of that same attention cost.
    """
    prefill_pairs = [
        [int(prefix), int(extend)] for prefix, extend in prefill_chunk_pairs
    ]
    decode_lens = [int(length) for length in decode_kv_lens]
    elapsed_ms = (observed_end_monotonic_ns - observed_start_monotonic_ns) / 1e6
    return {
        "schema_version": ITERATION_SCHEMA_VERSION,
        "input_adapter": INPUT_ADAPTER,
        "iteration_index": iteration_index,
        "observed_start_monotonic_ns": int(observed_start_monotonic_ns),
        "observed_end_monotonic_ns": int(observed_end_monotonic_ns),
        "observed_elapsed_ms": elapsed_ms,
        "prefill_tokens": sum(pair[1] for pair in prefill_pairs),
        "decode_requests": len(decode_lens),
        # One scheduled token per decode request; speculative drafts are counted
        # by the verify step that admits them, not here.
        "decode_tokens_scheduled": len(decode_lens),
        "prefill_chunk_pairs": prefill_pairs,
        "decode_kv_lens": decode_lens,
    }


# ── worker identity ──────────────────────────────────────────────────────────


def build_worker_record(
    *,
    pid: int,
    device_id: int,
    visible_devices: Optional[str],
    tp_rank: int,
    dp_rank: Optional[int],
    pp_rank: int,
    tp_size: int,
    dp_size: int,
) -> dict[str, Any]:
    """State which device this scheduler owns and which rank it holds.

    A multi-rank capture is a set of per-device kernel streams, and turning it
    back into per-rank evidence needs someone to say which device ran which
    rank. The vLLM fork leaves that to be recovered indirectly -- its banner
    states pid <-> rank, and the profiler separately knows pid <-> device -- but
    a scheduler process here is handed its `gpu_id` outright, so it can state
    the thing the analyzer actually wants and skip the join.

    `visible_devices` is recorded because that directness has one precondition:
    `gpu_id` is an index into the process's visible set, and it only equals the
    device ordinal the profiler reports while every rank shares one visible set.
    Under `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS` each rank sees a single device
    and calls it 0, so every rank would claim device 0. Recording what each rank
    could see lets the reader detect that and refuse, instead of silently
    folding the whole run onto one device.
    """
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "input_adapter": INPUT_ADAPTER,
        "pid": int(pid),
        "device_id": int(device_id),
        "visible_devices": visible_devices,
        "tp_rank": int(tp_rank),
        "dp_rank": int(dp_rank or 0),
        "pp_rank": int(pp_rank),
        "tp_size": int(tp_size),
        "dp_size": int(dp_size),
    }


def emit_worker_record(**kwargs: Any) -> None:
    _emit(_WORKER_RECORD_NAME, build_worker_record(**kwargs))


# ── iteration ────────────────────────────────────────────────────────────────


def emit_iteration_record(**kwargs: Any) -> None:
    _emit(_ITERATION_RECORD_NAME, build_iteration_record(**kwargs))


def emit_iteration_from_geometry(
    *,
    iteration_index: Optional[int],
    geometry: Optional[IterationGeometry],
    observed_end_monotonic_ns: int,
) -> None:
    """Close the record opened by `capture_iteration_geometry` at dispatch.

    A batch with no geometry was dispatched while the flag was off, so it has no
    start timestamp either and is skipped rather than reported with a zero one.
    """
    if geometry is None:
        return
    emit_iteration_record(
        iteration_index=iteration_index,
        observed_start_monotonic_ns=geometry.launch_monotonic_ns,
        observed_end_monotonic_ns=observed_end_monotonic_ns,
        prefill_chunk_pairs=geometry.prefill_chunk_pairs,
        decode_kv_lens=geometry.decode_kv_lens,
    )


# ── request timing, scheduler side ───────────────────────────────────────────


def build_request_timing_record(
    *,
    request_id: str,
    queued_monotonic: float,
    scheduled_monotonic: float,
    first_token_monotonic: float,
    last_token_monotonic: float,
    num_output_tokens: int,
) -> Optional[dict[str, Any]]:
    """Split a finished request's scheduler-side latency.

    Returns `None` -- rather than a record full of negative or zero durations --
    when a timestamp is missing or the four are out of order. A request that was
    aborted, or finished during a phase that never stamped its boundary, has no
    meaningful split, and a partial row would quietly skew the aggregate.
    """
    boundaries = (
        queued_monotonic,
        scheduled_monotonic,
        first_token_monotonic,
        last_token_monotonic,
    )
    if any(boundary <= 0.0 for boundary in boundaries) or num_output_tokens <= 0:
        return None
    if (
        not queued_monotonic
        <= scheduled_monotonic
        <= first_token_monotonic
        <= (last_token_monotonic)
    ):
        return None

    decode_ms = (last_token_monotonic - first_token_monotonic) * 1000.0
    return {
        "schema_version": REQUEST_TIMING_SCHEMA_VERSION,
        "input_adapter": INPUT_ADAPTER,
        "engine_request_id": request_id,
        "engine_core_ttft_ms": (first_token_monotonic - queued_monotonic) * 1000.0,
        "engine_queue_wait_ms": (scheduled_monotonic - queued_monotonic) * 1000.0,
        "engine_first_schedule_to_first_token_ms": (
            first_token_monotonic - scheduled_monotonic
        )
        * 1000.0,
        "engine_core_decode_ms": decode_ms,
        "num_output_tokens": int(num_output_tokens),
        # A one-token request has no inter-token interval, so it contributes no
        # TPOT sample. JSON null keeps that distinct from a measured zero.
        "engine_core_tpot_ms": (
            decode_ms / (num_output_tokens - 1) if num_output_tokens > 1 else None
        ),
    }


def emit_request_timing_record(**kwargs: Any) -> None:
    record = build_request_timing_record(**kwargs)
    if record is not None:
        _emit(_REQUEST_TIMING_RECORD_NAME, record)


# ── request timing, API server side ──────────────────────────────────────────


def build_api_request_timing_record(
    *,
    request_id: str,
    created_monotonic: float,
    tokenize_finish_monotonic: float,
    dispatch_monotonic: float,
    dispatch_finish_monotonic: float,
    first_token_monotonic: float,
    last_token_monotonic: float,
    finished_monotonic: float,
    output_tokens: int,
) -> Optional[dict[str, Any]]:
    """Split the API-server span the client waits on, ahead of the scheduler.

    Every subtraction here is between two API-process timestamps. SGLang does
    calibrate its API and scheduler clocks against each other, but this record
    does not lean on that: it reports one domain, and the scheduler-side
    `RequestTiming` record reports the other, exactly as the vLLM fork does.

    The field set is a subset of the vLLM fork's, and deliberately so: vLLM
    splits the wait for the first output across its output collector and its
    per-request generator, handoffs SGLang does not have -- it moves outputs over
    its own IPC and the phases are not separable from outside. Reporting the
    parent span and the two dispatch segments states what is true here rather
    than inventing a split to fill the schema.
    """
    boundaries = (
        created_monotonic,
        tokenize_finish_monotonic,
        dispatch_monotonic,
        dispatch_finish_monotonic,
        first_token_monotonic,
        last_token_monotonic,
        finished_monotonic,
    )
    if any(boundary <= 0.0 for boundary in boundaries):
        return None
    if not created_monotonic <= first_token_monotonic <= last_token_monotonic:
        return None

    record = {
        "schema_version": API_REQUEST_TIMING_SCHEMA_VERSION,
        "input_adapter": INPUT_ADAPTER,
        "api_request_id": request_id,
        "output_tokens": int(output_tokens),
        # Tokenization is the frontend work vLLM calls "prepare"; naming it the
        # same keeps the two engines' waterfalls stackable.
        "api_frontend_prepare_ms": (tokenize_finish_monotonic - created_monotonic)
        * 1000.0,
        # The whole wait for the first output, prepare excluded -- the parent of
        # the two dispatch segments below, not the residual left after them.
        # That nesting is the vLLM fork's, and the report checks it: the parts
        # an engine reports must fit inside this span.
        "api_first_output_wait_ms": (first_token_monotonic - tokenize_finish_monotonic)
        * 1000.0,
        "api_stream_activation_ms": (dispatch_monotonic - tokenize_finish_monotonic)
        * 1000.0,
        "api_add_request_ms": (dispatch_finish_monotonic - dispatch_monotonic) * 1000.0,
        "api_token_output_receive_span_ms": (
            last_token_monotonic - first_token_monotonic
        )
        * 1000.0,
        "api_terminal_tail_ms": (finished_monotonic - last_token_monotonic) * 1000.0,
    }
    return record


def emit_api_request_timing_record(**kwargs: Any) -> None:
    record = build_api_request_timing_record(**kwargs)
    if record is not None:
        _emit(_API_REQUEST_TIMING_RECORD_NAME, record)


# ── expert load ──────────────────────────────────────────────────────────────


def build_expert_load_record(
    *,
    model: str,
    eplb_step: int,
    expert_parallel_size: int,
    experts_per_token: int,
    logical_expert_counts: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Routed load per *logical* expert, per layer.

    Logical rather than physical: the physical placement is what EPLB is about
    to change, so a physical histogram is only meaningful next to the mapping
    that produced it. The simulator reasons about logical experts.
    """
    return {
        "schema_version": EXPERT_LOAD_SCHEMA_VERSION,
        "input_adapter": INPUT_ADAPTER,
        "model": model,
        "eplb_step": int(eplb_step),
        "expert_parallel_size": int(expert_parallel_size),
        "experts_per_token": int(experts_per_token),
        "logical_expert_counts": [
            [int(count) for count in layer_counts]
            for layer_counts in logical_expert_counts
        ],
    }


def emit_expert_load_record(**kwargs: Any) -> None:
    _emit(_EXPERT_LOAD_RECORD_NAME, build_expert_load_record(**kwargs))


# ── bulk dumps ───────────────────────────────────────────────────────────────
#
# The records above are one small JSON object each, on the log. These two are a
# different kind of thing: a single iteration's token IDs or per-layer expert
# histograms are far too large for a log line, and are only wanted for a handful
# of hand-picked iterations. So each writes JSONL to its own path, is off unless
# that path is set, and takes an iteration selector (`3`, `10-20`, `0,5,100-110`)
# that defaults to every iteration.
#
# Rows are appended under an exclusive lock so TP ranks sharing one path
# interleave whole rows rather than half-lines.


def parse_iterations(raw: str) -> Optional[set[int]]:
    """Parse an iteration selector. `None` means "every iteration"."""
    raw = raw.strip()
    if not raw:
        return None

    selected: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            selected.update(range(int(start_text), int(end_text) + 1))
        else:
            selected.add(int(part))
    return selected


def _iteration_selected(iteration_index: Optional[int], raw_selector: str) -> bool:
    selected = parse_iterations(raw_selector)
    if selected is None:
        return True
    return iteration_index in selected


def should_trace_token_iteration(iteration_index: Optional[int]) -> bool:
    if not envs.SGLANG_VIBESIM_TOKEN_TRACE_PATH.get():
        return False
    return _iteration_selected(
        iteration_index, envs.SGLANG_VIBESIM_TOKEN_TRACE_ITERS.get()
    )


def should_trace_routing_iteration(iteration_index: Optional[int]) -> bool:
    if not envs.SGLANG_VIBESIM_ROUTING_TRACE_PATH.get():
        return False
    return _iteration_selected(
        iteration_index, envs.SGLANG_VIBESIM_ROUTING_TRACE_ITERS.get()
    )


def _append_jsonl(path_text: str, row: dict[str, Any]) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _sequence_int_list(values: Any) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def _device_provenance() -> dict[str, Any]:
    import torch

    return {
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "local_cuda_device": (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        ),
    }


def build_token_input_row(
    *,
    iteration_index: Optional[int],
    token_ids: list[int],
    request_ids: list[str],
    num_scheduled_tokens: list[int],
) -> dict[str, Any]:
    """Split a flat token buffer back into per-request spans.

    Kept free of torch and of the environment: the span arithmetic is the only
    part that can silently attribute tokens to the wrong request, so it is the
    part worth testing directly.
    """
    requests: list[dict[str, Any]] = []
    offset = 0
    for request_id, count in zip(request_ids, num_scheduled_tokens):
        end = offset + count
        requests.append(
            {
                "req_id": request_id,
                "start": offset,
                "end": end,
                "num_scheduled_tokens": count,
                "token_ids": token_ids[offset:end],
            }
        )
        offset = end

    return {
        "schema_version": 1,
        "input_adapter": INPUT_ADAPTER,
        "iteration": iteration_index,
        "tokens": len(token_ids),
        "num_reqs": len(request_ids),
        "req_ids": request_ids,
        "num_scheduled_tokens": num_scheduled_tokens,
        "token_ids_flat": token_ids,
        "requests": requests,
    }


def dump_token_inputs(
    *,
    iteration_index: Optional[int],
    input_ids: Any,
    request_ids: list[str],
    num_scheduled_tokens: Sequence[int],
) -> None:
    """Dump the scheduled input token IDs for alignment-only reruns.

    Nothing else recovers the exact token stream the engine saw; a replay would
    otherwise have to re-tokenize prompt text and hope it matches.
    """
    if not should_trace_token_iteration(iteration_index):
        return
    trace_path = envs.SGLANG_VIBESIM_TOKEN_TRACE_PATH.get()
    if input_ids is None or not trace_path:
        return

    counts = [int(count) for count in num_scheduled_tokens]
    token_ids = _sequence_int_list(input_ids)[: sum(counts)]
    row = build_token_input_row(
        iteration_index=iteration_index,
        token_ids=token_ids,
        request_ids=list(request_ids),
        num_scheduled_tokens=counts,
    )
    row.update(_device_provenance())
    _append_jsonl(trace_path, row)


def grouped_gemm_block_stats(
    *,
    local_counts: list[int],
    total_assignments: int,
    global_num_experts: int,
    block_m: int,
) -> dict[str, Any]:
    """Padding cost of one grouped GEMM launch, given a per-expert load vector.

    The kernel sizes its launch for the worst case -- `total_assignments` plus
    one short block per expert -- while only `local_padded` rows carry work, so
    the ratio between them is how much of the launch is spent on padding. Pure
    arithmetic, so the simulator's grouped-GEMM cost model can be checked
    against it directly.
    """
    nonzero_counts = [count for count in local_counts if count > 0]
    local_padded = sum(
        int(math.ceil(count / block_m) * block_m) for count in nonzero_counts
    )
    sorted_token_ids_len = total_assignments + global_num_experts * (block_m - 1)
    launch_m_blocks = int(math.ceil(sorted_token_ids_len / block_m))
    effective_m_blocks = int(math.ceil(local_padded / block_m)) if local_padded else 0
    return {
        "block_m_assumed": block_m,
        "local_padded": local_padded,
        "sorted_token_ids_len": sorted_token_ids_len,
        "launch_m_blocks": launch_m_blocks,
        "effective_m_blocks": effective_m_blocks,
        "m_block_overlaunch": (
            launch_m_blocks / effective_m_blocks if effective_m_blocks else None
        ),
    }


def dump_routing_summary(
    *,
    iteration_index: Optional[int],
    per_layer_counts: Sequence[Sequence[int]],
    global_num_experts: int,
    top_k: int,
    tokens: int,
    block_m: int = 64,
) -> None:
    """Dump per-layer routed expert histograms for one nominated iteration.

    Complements the `ExpertLoad` record, which reads EPLB's accumulated load and
    therefore needs EPLB enabled and only fires on its rebalance steps. This one
    is per-iteration and carries the grouped-GEMM padding arithmetic with it.
    """
    if not should_trace_routing_iteration(iteration_index):
        return
    trace_path = envs.SGLANG_VIBESIM_ROUTING_TRACE_PATH.get()
    if not trace_path:
        return

    provenance = _device_provenance()
    for layer_id, layer_counts in enumerate(per_layer_counts):
        counts = [int(count) for count in layer_counts]
        total_assignments = sum(counts)
        row: dict[str, Any] = {
            "schema_version": 1,
            "input_adapter": INPUT_ADAPTER,
            "iteration": iteration_index,
            "layer_id": layer_id,
            "tokens": int(tokens),
            "top_k": int(top_k),
            "total_assignments": total_assignments,
            "global_num_experts": int(global_num_experts),
            "local_counts": counts,
            "local_assignments": total_assignments,
            "local_count_min": min(counts) if counts else 0,
            "local_count_max": max(counts) if counts else 0,
            "local_nonzero_experts": len([count for count in counts if count > 0]),
        }
        row.update(
            grouped_gemm_block_stats(
                local_counts=counts,
                total_assignments=total_assignments,
                global_num_experts=int(global_num_experts),
                block_m=block_m,
            )
        )
        row.update(provenance)
        _append_jsonl(trace_path, row)
