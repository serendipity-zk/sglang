"""Mixin for SLO scheduler sidecar communication.

Implements the 3-position temporal drain pattern for sending EngineState
to an external SLO scheduling sidecar over ZMQ. The three temporal sections
(FinishedIterationData, CurrentSnapshot, SchedulingContext) are captured at
distinct code positions in the event loop where their data is most accurate.

Call sites remain in scheduler.py (event loop orchestration); this mixin
holds only the method implementations and initialization.

Shadow mode (Phase 3): logs both internal and sidecar scheduling decisions
to a JSONL file for offline comparison. No online comparison in the engine.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional

from sglang.srt.managers.schedule_batch import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


@dataclass
class _InternalDecisionSnapshot:
    """Captures internal scheduling decision for shadow logging. Temporary; removed in Phase 5."""
    decode_only: bool
    effective_target: Optional[float]
    mode: str  # "simulation" | "predictor" | "slack" | "budget" | "resource_constraint"
    min_decode_slack_ms: Optional[float] = None
    actual_prefill_tokens: int = 0  # Filled post-batch
    num_requests_admitted: int = 0  # Filled post-batch
    # Debug context — common state
    tpot_ms: Optional[float] = None
    pred_last: Optional[float] = None  # self.last_cycle_time_prediction
    decode_batch_size: int = 0
    num_waiting: int = 0
    kv_used: int = 0
    kv_capacity: int = 0
    # Simulation-specific
    sim_budget: Optional[int] = None
    sim_execution_flow: Optional[list] = None
    sim_decode_slack_ms: Optional[float] = None
    sim_safety_margin_ms: Optional[float] = None
    # Predictor/Slack-specific
    pred_decode_time: Optional[float] = None
    target_iteration_time_ms: Optional[float] = None


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

        # Shadow mode decision logging (file opened lazily in _shadow_open_log)
        self._shadow_log_file = None
        self._shadow_internal_snapshot = None
        self._shadow_last_logged_iteration = -1  # dedup idle loops

    def _req_to_request_info(self: "Scheduler", req):
        """Convert engine Req to sidecar RequestInfo."""
        from slo_scheduler.messages.engine_state import RequestInfo
        # remaining_prefill: how many prefill tokens still need to be processed.
        # Use tokens_generated > 0 as the authoritative decode indicator:
        # once a request has generated ANY output token, its prefill is complete.
        # We do NOT rely on extend_input_len because it can be stale (retains
        # the original prefill chunk size even after the request transitions
        # to decode mode).
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
                Both loops pass self.iteration_count (the engine's counter).
                This ensures current.iteration_count matches the engine's
                router metrics report (iteration_num = self.iteration_count).
        """
        from slo_scheduler.messages.engine_state import CurrentSnapshot
        num_used = self._get_token_info()[0]
        now_ms = time.time() * 1000
        # Capture batch forward_mode so sidecar can correctly classify
        # DECODE batches (where req.extend_input_len may be stale from prefill).
        fwd_mode = getattr(batch, "forward_mode", None)
        fwd_mode_str = None
        if fwd_mode is not None:
            try:
                fwd_mode_str = fwd_mode.name  # e.g. "DECODE", "EXTEND", "MIXED"
            except AttributeError:
                fwd_mode_str = str(fwd_mode)

        # Batch composition for router reporting. In the overlap loop the router
        # receives metrics from the RUNNING batch (scheduler.py:1530), so these
        # must come from the batch just launched, not the completed one.
        prefill_tokens = 0
        decode_tokens = 0
        prefill_chunk_pairs = []
        if fwd_mode == ForwardMode.EXTEND:
            prefill_tokens = batch.extend_num_tokens if batch.extend_num_tokens else 0
        elif fwd_mode == ForwardMode.DECODE:
            decode_tokens = len(batch.reqs) if batch.reqs else 0
        elif fwd_mode == ForwardMode.MIXED:
            prefill_tokens = (
                batch.extend_num_tokens - len(batch.decoding_reqs)
                if batch.extend_num_tokens else 0
            )
            decode_tokens = len(batch.decoding_reqs) if batch.decoding_reqs else 0

        if fwd_mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            prefix_lens = getattr(batch, "prefix_lens", None)
            extend_lens = getattr(batch, "extend_lens", None)
            decoding_reqs = (
                set(batch.decoding_reqs)
                if getattr(batch, "decoding_reqs", None) else None
            )
            if prefix_lens is not None and extend_lens is not None and batch.reqs is not None:
                for i, req in enumerate(batch.reqs):
                    if fwd_mode == ForwardMode.MIXED and decoding_reqs and req in decoding_reqs:
                        continue
                    chunk = extend_lens[i] if i < len(extend_lens) else 0
                    if chunk and chunk > 0:
                        cumulative = (prefix_lens[i] if i < len(prefix_lens) else 0) + chunk
                        prefill_chunk_pairs.append([int(chunk), int(cumulative)])

        # Router ack tracking
        ack_gen, ack_last_id = self.router_ack_tracker.get_state()

        self._pending_current = CurrentSnapshot(
            iteration_count=iteration_count,
            timestamp_ms=now_ms,
            num_running_requests=len(batch.reqs) if batch.reqs else 0,
            num_waiting_requests=len(self.waiting_queue),
            kv_tokens_used=num_used,
            kv_capacity=self.max_total_num_tokens,
            running_requests=[self._req_to_request_info(r) for r in (batch.reqs or [])],
            forward_mode=fwd_mode_str,
            batch_size_tokens=prefill_tokens + decode_tokens,
            prefill_chunk_pairs=prefill_chunk_pairs,
            router_generation=ack_gen,
            router_last_ack_id=ack_last_id,
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

        # Compute batch_size_tokens — must match engine's
        # _collect_and_report_iteration_metrics exactly:
        #   EXTEND: prefill = extend_num_tokens, decode = 0
        #   DECODE: prefill = 0, decode = len(reqs)
        #   MIXED:  prefill = extend_num_tokens - len(decoding_reqs), decode = len(decoding_reqs)
        prefill_tokens = 0
        decode_tokens = 0
        if tmp_batch.forward_mode == ForwardMode.EXTEND:
            prefill_tokens = tmp_batch.extend_num_tokens if tmp_batch.extend_num_tokens else 0
        elif tmp_batch.forward_mode == ForwardMode.DECODE:
            decode_tokens = len(tmp_batch.reqs) if tmp_batch.reqs else 0
        elif tmp_batch.forward_mode == ForwardMode.MIXED:
            prefill_tokens = (
                tmp_batch.extend_num_tokens - len(tmp_batch.decoding_reqs)
                if tmp_batch.extend_num_tokens
                else 0
            )
            decode_tokens = len(tmp_batch.decoding_reqs) if tmp_batch.decoding_reqs else 0

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

        # # Log what we're sending to sidecar for predictor mismatch debugging
        # logger.warning(
        #     "[ENGINE_DRAIN_FINISHED] iter=%d batch_size_tokens=%d n_prefill_pairs=%d "
        #     "prefill_pairs=%s kv_tokens_used=%d actual_time_ms=%.3f forward_mode=%s",
        #     self.iteration_count, prefill_tokens + decode_tokens,
        #     len(prefill_pairs),
        #     [(p.chunk_tokens, p.cumulative_prefill) for p in prefill_pairs[:5]],
        #     kv_tokens_used, actual_time_ms,
        #     tmp_batch.forward_mode.name,
        # )

    def _drain_scheduling_context(self: "Scheduler"):
        """Drain position: Capture context for next scheduling decision.

        Called in get_next_batch_to_run() after merge, before predictions.
        """
        from slo_scheduler.messages.engine_state import SchedulingContext
        num_used, _, available, evictable = self._get_token_info()
        # kv_available must include evictable cache entries so that the sidecar
        # computes kv_used = kv_capacity - kv_available = num_used (matching
        # the engine's _get_token_info()[0] which excludes evictable tokens).
        kv_available = available + evictable
        now_ms = time.time() * 1000

        # Decode requests: running batch reqs that finished prefill and are
        # generating output tokens.  The old check (extend_input_len == 0) was
        # wrong because decode requests have extend_input_len = 1 after
        # process_batch_result.  Using output_ids > 0 matches the engine's
        # semantic: a request is "decoding" once it has produced at least one
        # output token.  Requests still in chunked prefill have output_ids = [].
        running_reqs = self.running_batch.reqs if self.running_batch else None
        n_total = len(running_reqs) if running_reqs else 0
        decode_reqs = [
            self._req_to_request_info(r)
            for r in (running_reqs or [])
            if len(r.output_ids) > 0
        ]
        # # Temporary debug: log when decode filter mismatches running batch size
        # if n_total > 0 and len(decode_reqs) == 0:
        #     sample = [(len(r.output_ids), r.extend_input_len, getattr(r, 'is_chunked', None)) for r in (running_reqs or [])[:5]]
        #     logger.warning("[DRAIN-DEBUG] n_running=%d n_decode=%d sample(output_ids_len, extend_input_len, is_chunked)=%s",
        #                    n_total, len(decode_reqs), sample)

        # Initialize tree-cache-aware prefix/extend for waiting requests
        # BEFORE converting to RequestInfo.  The engine's simulation
        # (_maybe_run_prefill_simulation, scheduler.py:2816) calls
        # req.init_next_round_input(tree_cache) to resolve tree cache hits,
        # which updates prefix_indices and extend_input_len in-place.
        # Without this, the sidecar sees stale total_prefill_lens
        # (e.g. 1846 instead of 1124 after tree cache hit).
        if self.chunked_req is not None:
            try:
                self.chunked_req.init_next_round_input(self.tree_cache)
            except Exception:
                pass
        for req in self.waiting_queue:
            extend_len = max(int(getattr(req, "extend_input_len", 0)), 0)
            if extend_len == 0 and not req.finished():
                try:
                    req.init_next_round_input(self.tree_cache)
                except Exception:
                    pass

        # Chunked request (separate list for slack observability)
        chunked = []
        if self.chunked_req is not None:
            chunked = [self._req_to_request_info(self.chunked_req)]

        # ENGINE CONTRACT: chunked_req prepended to waiting_requests
        waiting = []
        if self.chunked_req is not None:
            waiting.append(self._req_to_request_info(self.chunked_req))
        waiting.extend([self._req_to_request_info(r) for r in self.waiting_queue])

        # Engine uses max(len(last_batch.reqs), 1) as decode_batch for simulation.
        # last_batch is the batch currently executing on GPU (set at end of previous loop).
        last_batch_size = None
        if self.last_batch is not None and getattr(self.last_batch, 'reqs', None) is not None:
            last_batch_size = len(self.last_batch.reqs)

        self._pending_scheduling = SchedulingContext(
            iteration_count=self.iteration_count + 1,
            scheduling_time_ms=now_ms,
            decode_requests=decode_reqs,
            chunked_requests=chunked,
            waiting_requests=waiting,
            kv_available=kv_available,
            kv_capacity=self.max_total_num_tokens,
            last_batch_size=last_batch_size,
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
            ack_gen, ack_last_id = self.router_ack_tracker.get_state()
            current = CurrentSnapshot(
                iteration_count=self.iteration_count,
                timestamp_ms=time.time() * 1000,
                num_running_requests=len(self.running_batch.reqs) if self.running_batch and self.running_batch.reqs else 0,
                num_waiting_requests=len(self.waiting_queue),
                kv_tokens_used=num_used,
                kv_capacity=self.max_total_num_tokens,
                router_generation=ack_gen,
                router_last_ack_id=ack_last_id,
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
        logger.debug(
            "[ENGINE_SIDECAR_ACK_STATE] worker=%s iter=%s ack_gen=%s ack_last_id=%s running=%s waiting=%s",
            self.worker_id,
            scheduling.iteration_count,
            current.router_generation,
            current.router_last_ack_id,
            current.num_running_requests,
            current.num_waiting_requests,
        )

        # # Debug: log every message sent to sidecar
        # logger.warning(
        #     "[ENGINE→SIDECAR] iter=%d n_decode=%d n_waiting=%d n_chunked=%d kv_avail=%d",
        #     scheduling.iteration_count,
        #     len(scheduling.decode_requests),
        #     len(scheduling.waiting_requests),
        #     len(scheduling.chunked_requests),
        #     scheduling.kv_available,
        # )

        self._last_sidecar_decision = self.slo_client.send_and_recv(
            state, current_iteration=scheduling.iteration_count
        )
        self._accepted_since_last_send = 0

        # Clear pending sections after sending to prevent re-sending stale data.
        # On the next call, if no new data was drained, defaults (with actual_time_ms=0)
        # are used, which the sidecar safely ignores for predictor training.
        self._pending_finished = None
        self._pending_current = None

    # ── Shadow mode methods (Phase 3) ──────────────────────────────────

    def _shadow_open_log(self: "Scheduler"):
        """Lazily open shadow decision log file.

        Deferred because worker_id isn't set during init_sidecar().
        """
        if self._shadow_log_file is not None:
            return
        if self.slo_scheduler_mode not in ("shadow", "shadow-sidecar", "sidecar"):
            return
        import os
        fname = f"shadow_decisions_{self.worker_id}.jsonl"
        self._shadow_log_file = open(fname, "a", buffering=1)  # line-buffered
        logger.info("Shadow mode: logging decisions to %s", os.path.abspath(fname))

    def _shadow_common_context(self: "Scheduler") -> dict:
        """Capture common scheduler state for shadow logging debug fields."""
        if self.slo_scheduler_mode not in ("shadow", "shadow-sidecar", "sidecar"):
            return {}
        num_used = self._get_token_info()[0]
        running_bs = len(self.running_batch.reqs) if self.running_batch and self.running_batch.reqs else 0
        return dict(
            tpot_ms=self.tpot,
            pred_last=self.last_cycle_time_prediction,
            decode_batch_size=running_bs,
            num_waiting=len(self.waiting_queue),
            kv_used=num_used,
            kv_capacity=self.max_total_num_tokens,
        )

    def _shadow_capture_decode_only(self: "Scheduler", mode: str, **kwargs):
        """Capture internal decode-only decision for shadow logging."""
        if self.slo_scheduler_mode not in ("shadow", "shadow-sidecar", "sidecar"):
            return
        self._shadow_internal_snapshot = _InternalDecisionSnapshot(
            decode_only=True,
            effective_target=None,
            mode=mode,
            **kwargs,
        )

    def _shadow_capture_pre_batch(self: "Scheduler", effective_target, mode: str, min_slack=None, **kwargs):
        """Capture internal pre-batch scheduling decision for shadow logging."""
        if self.slo_scheduler_mode not in ("shadow", "shadow-sidecar", "sidecar"):
            return
        self._shadow_internal_snapshot = _InternalDecisionSnapshot(
            decode_only=False,
            effective_target=effective_target,
            mode=mode,
            min_decode_slack_ms=min_slack,
            **kwargs,
        )

    def _shadow_capture_post_batch(self: "Scheduler", prefill_tokens: int, num_admitted: int):
        """Update shadow snapshot with actual post-batch numbers."""
        if self.slo_scheduler_mode not in ("shadow", "shadow-sidecar", "sidecar"):
            return
        if self._shadow_internal_snapshot is not None:
            self._shadow_internal_snapshot.actual_prefill_tokens = prefill_tokens
            self._shadow_internal_snapshot.num_requests_admitted = num_admitted

    def _shadow_log_decisions(self: "Scheduler"):
        """Log both internal and sidecar decisions to JSONL file."""
        if self.slo_scheduler_mode not in ("shadow", "shadow-sidecar", "sidecar"):
            return

        import json

        internal = self._shadow_internal_snapshot
        sidecar = self._last_sidecar_decision

        # Nothing to log if both are absent
        if internal is None and sidecar is None:
            return

        # Deduplicate idle loops: only log once per iteration.
        # iteration_count only increments when a batch actually runs,
        # so idle spins produce the same iteration_count repeatedly.
        if self.iteration_count == self._shadow_last_logged_iteration:
            self._shadow_internal_snapshot = None
            return
        self._shadow_last_logged_iteration = self.iteration_count

        self._shadow_open_log()

        record = {
            "ts_ms": time.time() * 1000,
            "iteration": self.iteration_count,
            "applied": "sidecar" if self.slo_scheduler_mode in ("shadow-sidecar", "sidecar") and sidecar is not None else "internal",
            "internal": asdict(internal) if internal else None,
            "sidecar": {
                "iteration_count": sidecar.iteration_count,
                "decode_only": sidecar.decode_only_iteration,
                "max_prefill_tokens": sidecar.max_prefill_tokens,
                "target_iteration_time_ms": sidecar.target_iteration_time_ms,
                "min_decode_slack_ms": sidecar.min_decode_slack_ms,
                "predicted_iteration_time_ms": sidecar.predicted_iteration_time_ms,
                "mode": sidecar.mode_used,
                "scheduling_reason": sidecar.scheduling_reason,
                "prefill_chunk_budget": sidecar.prefill_chunk_budget,
                "execution_flow": sidecar.execution_flow,
                "simulation_success": sidecar.simulation_success,
            } if sidecar else None,
        }
        try:
            self._shadow_log_file.write(json.dumps(record) + "\n")
        except Exception:
            logger.warning("Shadow mode: failed to write decision log", exc_info=True)
        self._shadow_internal_snapshot = None
