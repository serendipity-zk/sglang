"""Mixin for SLO scheduler sidecar communication.

Implements the 3-position temporal drain pattern for sending EngineState
to an external SLO scheduling sidecar over ZMQ. The three temporal sections
(FinishedIterationData, CurrentSnapshot, SchedulingContext) are captured at
distinct code positions in the event loop where their data is most accurate.

Call sites remain in scheduler.py (event loop orchestration); this mixin
holds only the method implementations and initialization.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from sglang.srt.managers.schedule_batch import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerSidecarMixin:

    def init_sidecar(self: "Scheduler", server_args):
        """Initialize SLO scheduler sidecar client and drain buffers."""
        if server_args.slo_scheduler_addr:
            from sglang.srt.managers.slo_scheduler_client import SLOSchedulerClient
            self.slo_client = SLOSchedulerClient(
                server_args.slo_scheduler_addr,
                server_args.slo_scheduler_timeout_ms,
            )
            self.slo_scheduler_mode = server_args.slo_scheduler_mode
        else:
            self.slo_client = None
            self.slo_scheduler_mode = "internal"

        # Sidecar state drain buffers (3-position temporal drain)
        self._pending_finished = None   # FinishedIterationData from last completed iteration
        self._pending_current = None    # CurrentSnapshot of what GPU is executing
        self._pending_scheduling = None # SchedulingContext for next batch planning
        self._last_sidecar_decision = None
        # In overlap loop, _pre_batch_kv_used is overwritten for the NEW batch before
        # the PREVIOUS batch's FinishedIterationData is drained. We keep two slots:
        # _prev_pre_batch_kv_used holds the value for the batch being finished.
        self._pre_batch_kv_used: int = 0          # KV before current run_batch
        self._prev_pre_batch_kv_used: int = 0     # KV before previous run_batch (overlap)
        self._accepted_since_last_send: int = 0   # accepted request counter

    def _req_to_request_info(self: "Scheduler", req):
        """Convert engine Req to sidecar RequestInfo."""
        from slo_scheduler.messages.engine_state import RequestInfo
        return RequestInfo(
            request_id=req.rid,
            target_tpot_ms=req.target_tpot_ms,
            target_ttft_ms=req.target_ttft_ms,
            arrival_time_ms=req.arrival_time_ms,
            tokens_generated=len(req.output_ids),
            prompt_tokens=len(req.origin_input_ids),
            remaining_prefill=req.extend_input_len,
            max_new_tokens=req.sampling_params.max_new_tokens,
            slo_violated=req.slo_violated,
            prefix_len=len(req.prefix_indices),
            extend_input_len=req.extend_input_len,
            evicted_seqlen_local=req.evicted_seqlen_local,
            router_generation=req.router_generation,
            router_message_id=req.router_message_id,
        )

    def _drain_current_snapshot(self: "Scheduler", batch, iteration_count: int):
        """Drain position: Capture what GPU is actively executing NOW.

        Called immediately after run_batch() launches a batch on the GPU.

        Args:
            batch: The batch just launched on GPU.
            iteration_count: The iteration number for this batch.
                In overlap loop: self.iteration_count + 1 (not yet incremented).
                In normal loop: self.iteration_count (already incremented).

        NOTE (overlap warmup): On the second pass of the overlap loop, the
        previous _drain_current_snapshot set iteration_count = N+1 but
        self.iteration_count has not yet been incremented (happens later when
        processing the first real batch result).  This means the next
        _assemble_and_send_engine_state sees scheduling.iteration_count ==
        current.iteration_count (both N+1) instead of scheduling = current + 1.
        This is a one-time startup transient that self-corrects after the first
        batch result is processed and iteration_count increments.
        """
        from slo_scheduler.messages.engine_state import CurrentSnapshot
        num_used = self._get_token_info()[0]
        now_ms = time.time() * 1000
        self._pending_current = CurrentSnapshot(
            iteration_count=iteration_count,
            timestamp_ms=now_ms,
            num_running_requests=len(batch.reqs) if batch.reqs else 0,
            num_waiting_requests=len(self.waiting_queue),
            kv_tokens_used=num_used,
            kv_capacity=self.max_total_num_tokens,
            running_requests=[self._req_to_request_info(r) for r in (batch.reqs or [])],
        )

    def _drain_finished_iteration(self: "Scheduler", tmp_batch, actual_time_ms, kv_tokens_used: int):
        """Drain position: Capture data from just-completed iteration.

        Called after process_batch_result() and iteration_count increment.

        Args:
            tmp_batch: The batch that just completed.
            actual_time_ms: GPU elapsed or wall-clock iteration time.
            kv_tokens_used: KV tokens used BEFORE this batch ran.
                Normal loop: self._pre_batch_kv_used
                Overlap loop: self._prev_pre_batch_kv_used (from before this batch launched)
        """
        from slo_scheduler.messages.engine_state import (
            FinishedIterationData, PrefillChunkPair,
        )

        # Build prefill_chunk_pairs
        prefill_pairs = []
        if tmp_batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            prefix_lens = getattr(tmp_batch, "prefix_lens", None)
            extend_lens = getattr(tmp_batch, "extend_lens", None)
            decoding_reqs = (
                set(tmp_batch.decoding_reqs)
                if getattr(tmp_batch, "decoding_reqs", None)
                else None
            )
            if (
                prefix_lens is not None
                and extend_lens is not None
                and tmp_batch.reqs is not None
            ):
                for i, req in enumerate(tmp_batch.reqs):
                    if (
                        tmp_batch.forward_mode == ForwardMode.MIXED
                        and decoding_reqs
                        and req in decoding_reqs
                    ):
                        continue
                    chunk = extend_lens[i] if i < len(extend_lens) else 0
                    if chunk and chunk > 0:
                        cumulative = (
                            (prefix_lens[i] if i < len(prefix_lens) else 0) + chunk
                        )
                        prefill_pairs.append(PrefillChunkPair(
                            request_id=req.rid,
                            chunk_tokens=int(chunk),
                            cumulative_prefill=int(cumulative),
                        ))

        # Compute batch_size_tokens
        prefill_tokens = tmp_batch.extend_num_tokens or 0
        if tmp_batch.forward_mode == ForwardMode.MIXED and tmp_batch.decoding_reqs:
            prefill_tokens -= len(tmp_batch.decoding_reqs)
        decode_tokens = (
            len(tmp_batch.decoding_reqs)
            if getattr(tmp_batch, "decoding_reqs", None)
            else (
                len(tmp_batch.reqs)
                if tmp_batch.forward_mode == ForwardMode.DECODE
                else 0
            )
        )

        # Completed decode lengths from finished requests
        completed_lengths = [
            len(req.output_ids)
            for req in (tmp_batch.reqs or [])
            if req.finished() and not getattr(req, 'is_retracted', False)
        ]

        self._pending_finished = FinishedIterationData(
            iteration_count=self.iteration_count,
            batch_size_tokens=prefill_tokens + decode_tokens,
            prefill_chunk_pairs=prefill_pairs,
            kv_tokens_used=kv_tokens_used,
            forward_mode=tmp_batch.forward_mode.name,
            actual_time_ms=actual_time_ms,
            completed_decode_lengths=completed_lengths,
        )

    def _drain_scheduling_context(self: "Scheduler"):
        """Drain position: Capture context for next scheduling decision.

        Called in get_next_batch_to_run() after merge, before predictions.
        """
        from slo_scheduler.messages.engine_state import SchedulingContext
        _, _, available, _ = self._get_token_info()
        now_ms = time.time() * 1000

        # Decode requests: running batch reqs that finished prefill
        running_reqs = self.running_batch.reqs if self.running_batch else None
        decode_reqs = [
            self._req_to_request_info(r)
            for r in (running_reqs or [])
            if r.extend_input_len == 0
        ]

        # Chunked request (separate list for slack observability)
        chunked = []
        if self.chunked_req is not None:
            chunked = [self._req_to_request_info(self.chunked_req)]

        # ENGINE CONTRACT: chunked_req prepended to waiting_requests
        waiting = []
        if self.chunked_req is not None:
            waiting.append(self._req_to_request_info(self.chunked_req))
        waiting.extend([self._req_to_request_info(r) for r in self.waiting_queue])

        self._pending_scheduling = SchedulingContext(
            iteration_count=self.iteration_count + 1,
            scheduling_time_ms=now_ms,
            decode_requests=decode_reqs,
            chunked_requests=chunked,
            waiting_requests=waiting,
            kv_available=available,
            kv_capacity=self.max_total_num_tokens,
        )

    def _assemble_and_send_engine_state(self: "Scheduler"):
        """Assemble 3 temporal sections into EngineState and send to sidecar."""
        from slo_scheduler.messages.engine_state import (
            EngineState, FinishedIterationData, CurrentSnapshot,
        )

        # Use zero-valued defaults if pending sections not yet populated (first iteration
        # or idle loops where no batch ran). actual_time_ms=0 tells sidecar to skip
        # predictor training. Live KV/queue state is captured for idle snapshots.
        finished = self._pending_finished or FinishedIterationData(
            iteration_count=self.iteration_count, batch_size_tokens=0,
            prefill_chunk_pairs=[], kv_tokens_used=0,
            forward_mode="DECODE", actual_time_ms=0.0,
        )
        if self._pending_current is not None:
            current = self._pending_current
        else:
            num_used = self._get_token_info()[0]
            current = CurrentSnapshot(
                iteration_count=self.iteration_count,
                timestamp_ms=time.time() * 1000,
                num_running_requests=len(self.running_batch.reqs) if self.running_batch and self.running_batch.reqs else 0,
                num_waiting_requests=len(self.waiting_queue),
                kv_tokens_used=num_used,
                kv_capacity=self.max_total_num_tokens,
            )
        scheduling = self._pending_scheduling  # always just drained

        state = EngineState(
            protocol_version=1,
            min_sidecar_version=1,
            worker_id=self.worker_id,
            finished=finished,
            current=current,
            scheduling=scheduling,
            accepted_requests_count=self._accepted_since_last_send,
        )

        self._last_sidecar_decision = self.slo_client.send_and_recv(
            state, current_iteration=scheduling.iteration_count
        )
        self._accepted_since_last_send = 0

        # Clear pending sections after sending to prevent re-sending stale data.
        # On the next call, if no new data was drained, defaults (with actual_time_ms=0)
        # are used, which the sidecar safely ignores for predictor training.
        self._pending_finished = None
        self._pending_current = None
