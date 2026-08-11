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

import json
import logging
from functools import wraps
from typing import Any, Optional, Sequence

import msgspec

from sglang.srt.environ import envs
from sglang.srt.utils.nvtx_utils import NVTX_SCHEDULER_ENABLED, profile_range

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    return envs.SGLANG_ENABLE_VIBESIM_ALIGNMENT.get()


class IterationGeometry(msgspec.Struct, frozen=True, omit_defaults=True):
    """A batch's shape, snapshotted before the forward pass mutates it.

    Taken at dispatch rather than at result-processing time because the decode
    path advances `seq_lens` during the forward: read afterwards, every context
    length would be one token too long, and in overlap mode the next batch may
    already be in flight.
    """

    launch_monotonic_ns: int = 0
    prefill_chunk_pairs: list[list[int]] = []
    decode_kv_lens: list[int] = []


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

ITERATION_SCHEMA_VERSION = 2
REQUEST_TIMING_SCHEMA_VERSION = 2
EXPERT_LOAD_SCHEMA_VERSION = 2
API_REQUEST_TIMING_SCHEMA_VERSION = 3

_ITERATION_RECORD_NAME = "VibeSimAlignmentIteration"
_REQUEST_TIMING_RECORD_NAME = "VibeSimAlignmentRequestTiming"
_API_REQUEST_TIMING_RECORD_NAME = "VibeSimAlignmentApiRequestTiming"
_EXPERT_LOAD_RECORD_NAME = "VibeSimAlignmentExpertLoad"


def _emit(record_name: str, record: dict[str, Any]) -> None:
    logger.info("%s %s", record_name, json.dumps(record, separators=(",", ":")))


def iteration_profile_method(stage: str):
    """Name a span `sglang_iteration(N): <stage>`, mirroring the vLLM fork.

    The point of putting the index in the name is that an nsys range then joins
    to that iteration's records with no timestamp matching, which is what makes
    a measured GPU timeline comparable to a simulated one span by span.

    Only applicable where `batch.forward_iter` is already assigned. It is not,
    on entry to `run_batch` -- that method assigns it -- so the forward keeps
    SGLang's own `scheduler.run_batch` span, and the iteration record's
    `observed_start/end_monotonic_ns` is what windows the forward on a timeline.
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
    response_sent_monotonic: float,
    output_tokens: int,
) -> Optional[dict[str, Any]]:
    """Split the API-server span the client waits on, ahead of the scheduler.

    Every subtraction here is between two API-process timestamps. SGLang does
    calibrate its API and scheduler clocks against each other, but this record
    does not lean on that: it reports one domain, and the scheduler-side
    `RequestTiming` record reports the other, exactly as the vLLM fork does.
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
        "api_stream_activation_ms": (dispatch_monotonic - tokenize_finish_monotonic)
        * 1000.0,
        "api_add_request_ms": (dispatch_finish_monotonic - dispatch_monotonic) * 1000.0,
        "api_first_output_wait_ms": (first_token_monotonic - dispatch_finish_monotonic)
        * 1000.0,
        "api_token_output_receive_span_ms": (
            last_token_monotonic - first_token_monotonic
        )
        * 1000.0,
        "api_terminal_tail_ms": (finished_monotonic - last_token_monotonic) * 1000.0,
        "api_e2e_ms": (finished_monotonic - created_monotonic) * 1000.0,
    }
    # Only the streaming path stamps the moment the last chunk left the server;
    # the non-streaming path has no such boundary, so the field is absent rather
    # than folded into the tail.
    if response_sent_monotonic > 0.0:
        record["api_response_sent_ms"] = (
            response_sent_monotonic - finished_monotonic
        ) * 1000.0
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
