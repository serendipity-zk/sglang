"""Snapshot helpers for sidecar scheduler integration."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerSidecarMixin:
    """Build sidecar snapshot state without wiring it into the live loop yet."""

    def _get_router_ack_state(self: "Scheduler"):
        tracker = getattr(self, "router_ack_tracker", None)
        if tracker is None:
            return None, None
        return tracker.get_state()

    def init_sidecar(self: "Scheduler", _server_args) -> None:
        if getattr(_server_args, "slo_scheduler_addr", None):
            from sglang.srt.managers.slo_scheduler_client import SLOSchedulerClient

            self.slo_client = SLOSchedulerClient(
                _server_args.slo_scheduler_addr,
                _server_args.slo_scheduler_timeout_ms,
            )
        else:
            self.slo_client = None
        self._pending_finished = None
        self._pending_current = None
        self._pending_scheduling = None
        self._last_sidecar_decision = None
        self._pre_batch_kv_used = 0
        self._prev_pre_batch_kv_used = 0
        self._accepted_since_last_send = 0

    def _assemble_and_send_engine_state(self: "Scheduler") -> None:
        from slo_scheduler.messages.engine_state import (
            CurrentSnapshot,
            EngineState,
            FinishedIterationData,
        )

        if self.slo_client is None or self._pending_scheduling is None:
            return

        finished = self._pending_finished or FinishedIterationData(
            iteration_count=self.iteration_count,
            batch_size_tokens=0,
            prefill_chunk_pairs=[],
            kv_tokens_used=0,
            forward_mode="DECODE",
            actual_time_ms=0.0,
        )
        ack_gen, ack_last_id = self._get_router_ack_state()
        current = self._pending_current or CurrentSnapshot(
            iteration_count=self.iteration_count,
            timestamp_ms=time.time() * 1000,
            num_running_requests=(
                len(self.running_batch.reqs)
                if self.running_batch is not None and self.running_batch.reqs is not None
                else 0
            ),
            num_waiting_requests=len(self.waiting_queue),
            kv_tokens_used=self._get_token_info()[0],
            kv_capacity=self.max_total_num_tokens,
            running_requests=[],
            router_generation=ack_gen,
            router_last_ack_id=ack_last_id,
        )

        state = EngineState(
            protocol_version=1,
            min_sidecar_version=1,
            worker_id=self.worker_id,
            finished=finished,
            current=current,
            scheduling=self._pending_scheduling,
            accepted_requests_count=self._accepted_since_last_send,
        )

        self._last_sidecar_decision = self.slo_client.send_and_recv(
            state, current_iteration=self._pending_scheduling.iteration_count
        )
        self._accepted_since_last_send = 0
        self._pending_finished = None
        self._pending_current = None

    def _req_to_request_info(self: "Scheduler", req):
        from slo_scheduler.messages.engine_state import RequestInfo

        is_decoding = len(req.output_ids) > 0
        remaining_prefill = 0 if is_decoding else req.extend_input_len
        return RequestInfo(
            request_id=req.rid,
            target_tpot_ms=req.target_tpot_ms,
            target_ttft_ms=req.target_ttft_ms,
            arrival_time_ms=req.arrival_time_ms,
            tokens_generated=len(req.output_ids),
            prompt_tokens=len(req.origin_input_ids),
            remaining_prefill=remaining_prefill,
            max_new_tokens=req.sampling_params.max_new_tokens,
            slo_violated=req.slo_violated,
            prefix_len=len(req.prefix_indices),
            extend_input_len=req.extend_input_len,
            evicted_seqlen_local=getattr(req, "swa_evicted_seqlen", 0),
            router_generation=getattr(req, "router_generation", None),
            router_message_id=getattr(req, "router_message_id", None),
        )

    def _build_prefill_chunk_pairs(self: "Scheduler", batch):
        pairs = []
        if batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.MIXED):
            return pairs

        prefix_lens = getattr(batch, "prefix_lens", None)
        extend_lens = getattr(batch, "extend_lens", None)
        decoding_reqs = (
            set(batch.decoding_reqs) if getattr(batch, "decoding_reqs", None) else None
        )
        if prefix_lens is None or extend_lens is None or batch.reqs is None:
            return pairs

        for i, req in enumerate(batch.reqs):
            if (
                batch.forward_mode == ForwardMode.MIXED
                and decoding_reqs
                and req in decoding_reqs
            ):
                continue
            chunk = extend_lens[i] if i < len(extend_lens) else 0
            if chunk and chunk > 0:
                cumulative = (prefix_lens[i] if i < len(prefix_lens) else 0) + chunk
                pairs.append((req, int(chunk), int(cumulative)))
        return pairs

    def _compute_batch_size_tokens(self: "Scheduler", batch):
        prefill_tokens = 0
        decode_tokens = 0
        if batch.forward_mode == ForwardMode.EXTEND:
            prefill_tokens = batch.extend_num_tokens if batch.extend_num_tokens else 0
        elif batch.forward_mode == ForwardMode.DECODE:
            decode_tokens = len(batch.reqs) if batch.reqs else 0
        elif batch.forward_mode == ForwardMode.MIXED:
            prefill_tokens = (
                batch.extend_num_tokens - len(batch.decoding_reqs)
                if batch.extend_num_tokens
                else 0
            )
            decode_tokens = len(batch.decoding_reqs) if batch.decoding_reqs else 0
        return prefill_tokens + decode_tokens

    def _drain_current_snapshot(self: "Scheduler", batch, iteration_count: int) -> None:
        from slo_scheduler.messages.engine_state import CurrentSnapshot

        num_used = self._get_token_info()[0]
        pairs = self._build_prefill_chunk_pairs(batch)
        ack_gen, ack_last_id = self._get_router_ack_state()

        self._pending_current = CurrentSnapshot(
            iteration_count=iteration_count,
            timestamp_ms=time.time() * 1000,
            num_running_requests=len(batch.reqs) if batch.reqs else 0,
            num_waiting_requests=len(self.waiting_queue),
            kv_tokens_used=num_used,
            kv_capacity=self.max_total_num_tokens,
            running_requests=[self._req_to_request_info(r) for r in (batch.reqs or [])],
            forward_mode=batch.forward_mode.name,
            batch_size_tokens=self._compute_batch_size_tokens(batch),
            prefill_chunk_pairs=[[chunk, cumulative] for _, chunk, cumulative in pairs],
            router_generation=ack_gen,
            router_last_ack_id=ack_last_id,
        )

    def _drain_finished_iteration(
        self: "Scheduler", tmp_batch, actual_time_ms: float, kv_tokens_used: int
    ) -> None:
        from slo_scheduler.messages.engine_state import (
            FinishedIterationData,
            PrefillChunkPair,
        )

        pairs = [
            PrefillChunkPair(
                request_id=req.rid,
                chunk_tokens=chunk,
                cumulative_prefill=cumulative,
            )
            for req, chunk, cumulative in self._build_prefill_chunk_pairs(tmp_batch)
        ]
        completed_lengths = [
            len(req.output_ids)
            for req in (tmp_batch.reqs or [])
            if req.finished() and not getattr(req, "is_retracted", False)
        ]

        self._pending_finished = FinishedIterationData(
            iteration_count=self.iteration_count,
            batch_size_tokens=self._compute_batch_size_tokens(tmp_batch),
            prefill_chunk_pairs=pairs,
            kv_tokens_used=kv_tokens_used,
            forward_mode=tmp_batch.forward_mode.name,
            actual_time_ms=actual_time_ms,
            completed_decode_lengths=completed_lengths,
        )

    def _drain_scheduling_context(self: "Scheduler") -> None:
        from slo_scheduler.messages.engine_state import SchedulingContext

        _num_used, _token_usage, available, evictable = self._get_token_info()
        kv_available = available + evictable

        if self.chunked_req is not None:
            try:
                self.chunked_req.init_next_round_input(self.tree_cache)
            except TypeError:
                self.chunked_req.init_next_round_input()

        for req in self.waiting_queue:
            extend_len = max(int(getattr(req, "extend_input_len", 0)), 0)
            if extend_len == 0 and not req.finished():
                req.init_next_round_input(self.tree_cache)

        running_reqs = self.running_batch.reqs if self.running_batch else []
        decode_reqs = [
            self._req_to_request_info(req)
            for req in running_reqs
            if len(req.output_ids) > 0
        ]
        chunked_reqs = (
            [self._req_to_request_info(self.chunked_req)]
            if self.chunked_req is not None
            else []
        )
        waiting_reqs = []
        if self.chunked_req is not None:
            waiting_reqs.append(self._req_to_request_info(self.chunked_req))
        waiting_reqs.extend(self._req_to_request_info(req) for req in self.waiting_queue)

        last_batch_size = None
        if self.last_batch is not None and getattr(self.last_batch, "reqs", None) is not None:
            last_batch_size = len(self.last_batch.reqs)

        self._pending_scheduling = SchedulingContext(
            iteration_count=self.iteration_count + 1,
            scheduling_time_ms=time.time() * 1000,
            decode_requests=decode_reqs,
            chunked_requests=chunked_reqs,
            waiting_requests=waiting_reqs,
            kv_available=kv_available,
            kv_capacity=self.max_total_num_tokens,
            last_batch_size=last_batch_size,
        )
