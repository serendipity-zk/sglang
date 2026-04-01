"""Snapshot helpers for sidecar scheduler integration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import time
from typing import TYPE_CHECKING, Dict, Optional

from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

CPU_PHASE_BREAKDOWN_KEYS = (
    "recv",
    "input",
    "schedule",
    "launch",
    "current",
    "result",
    "sample",
    "idle",
)
CPU_SCHEDULE_BREAKDOWN_KEYS = (
    "merge",
    "sidecar_prep",
    "sidecar_rpc",
    "prefill",
    "decode",
    "finalize",
)
SIDECAR_RPC_BREAKDOWN_KEYS = (
    "build_state",
    "serialize_send",
    "wait",
    "recv_deserialize",
    "tp_sync",
    "other",
)


def _empty_breakdown(keys) -> Dict[str, float]:
    return {key: 0.0 for key in keys}


@dataclass(frozen=True)
class RuntimeSidecarDecision:
    """Minimal sidecar decision fields needed by every TP rank."""

    iteration_count: int
    max_prefill_tokens: int
    decode_only_iteration: bool


@dataclass(frozen=True)
class _FinishedObservabilitySnapshot:
    """Launch-time batch composition cached until the same batch finishes."""

    num_running_requests: int
    num_waiting_requests: int
    kv_tokens_used: int
    kv_capacity: int
    batch_size_tokens: int
    prefill_chunk_pairs: list[list[int]]
    batch_size_by_tpot_tier: Dict[str, int]
    forward_mode: Optional[str] = None


class SchedulerSidecarMixin:
    """Build sidecar snapshot state without wiring it into the live loop yet."""

    def _sidecar_enabled_for_scheduling(self: "Scheduler") -> bool:
        return bool(
            getattr(self, "_sidecar_enabled", False)
            or getattr(self, "slo_client", None) is not None
        )

    def _sidecar_owner_on_rank(self: "Scheduler") -> bool:
        return bool(
            getattr(self, "_sidecar_is_owner", False)
            or getattr(self, "slo_client", None) is not None
        )

    def _build_sidecar_worker_id(self: "Scheduler", server_args) -> str:
        host = getattr(server_args, "host", None)
        port = getattr(server_args, "port", None)
        if host is not None and port is not None:
            worker_id = f"{host}:{port}"
        else:
            worker_id = getattr(self, "worker_id", "unknown-worker")

        if not getattr(self, "_sidecar_sync_across_tp", False) and getattr(
            self, "tp_size", 1
        ) > 1:
            worker_id += f":tp{self.tp_rank}"
        if getattr(self, "dp_size", 1) > 1 and getattr(self, "dp_rank", None) is not None:
            worker_id += f":dp{self.dp_rank}"
        return worker_id

    def _build_runtime_sidecar_decision(
        self: "Scheduler", decision
    ) -> Optional[RuntimeSidecarDecision]:
        if decision is None:
            return None
        return RuntimeSidecarDecision(
            iteration_count=int(decision.iteration_count),
            max_prefill_tokens=int(decision.max_prefill_tokens),
            decode_only_iteration=bool(
                getattr(decision, "decode_only_iteration", False)
            ),
        )

    def _broadcast_sidecar_runtime_decision(
        self: "Scheduler", decision: Optional[RuntimeSidecarDecision]
    ) -> Optional[RuntimeSidecarDecision]:
        if not getattr(self, "_sidecar_sync_across_tp", False):
            return decision

        tp_group = getattr(self, "tp_group", None)
        if tp_group is None:
            return decision

        outbound_decision = (
            decision if getattr(self, "_sidecar_is_owner", False) else None
        )
        # GroupCoordinator.broadcast_object expects the source rank within the TP
        # group, not the global rank. The TP group's shared-memory-backed message
        # queue fast path only supports src=0.
        return tp_group.broadcast_object(outbound_decision, src=0)

    def _sync_sidecar_decision(
        self: "Scheduler", decision, decision_payload: Optional[bytes] = None
    ):
        runtime_decision = self._build_runtime_sidecar_decision(decision)
        # Keep the full decision owner-local for observability/debugging, but only
        # synchronize the hot-path fields needed by follower TP ranks.
        _ = decision_payload
        return self._broadcast_sidecar_runtime_decision(runtime_decision)

    def init_sidecar(self: "Scheduler", _server_args) -> None:
        self._sidecar_enabled = bool(getattr(_server_args, "slo_scheduler_addr", None))
        tp_group = getattr(self, "tp_group", None)
        self._sidecar_sync_across_tp = bool(
            self._sidecar_enabled
            and getattr(self, "tp_size", 1) > 1
            and tp_group is not None
            and getattr(tp_group, "world_size", 1) > 1
        )
        self._sidecar_is_owner = bool(
            self._sidecar_enabled
            and (
                not self._sidecar_sync_across_tp
                or getattr(tp_group, "is_first_rank", False)
            )
        )
        self.sidecar_worker_id = self._build_sidecar_worker_id(_server_args)

        if self._sidecar_is_owner:
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
        self._pending_finished_observability_snapshots = {}
        self._last_sidecar_decision = None
        self._last_sidecar_full_decision = None
        self._pre_batch_kv_used = 0
        self._prev_pre_batch_kv_used = 0
        self._accepted_since_last_send = 0
        self._accepted_request_ids_since_last_send = []
        self._pending_launch_sidecar_wait_time_ms = None
        self._pending_launch_sidecar_rpc_breakdown_ms = None
        self._launched_batch_sidecar_wait_times_ms = deque()
        self._launched_batch_sidecar_rpc_breakdowns_ms = deque()
        self._last_finished_iteration_ts = None
        self._cpu_phase_breakdown_ms = _empty_breakdown(CPU_PHASE_BREAKDOWN_KEYS)
        self._cpu_schedule_breakdown_ms = _empty_breakdown(
            CPU_SCHEDULE_BREAKDOWN_KEYS
        )

    def _ensure_cpu_timing_breakdowns(self: "Scheduler") -> None:
        if getattr(self, "_cpu_phase_breakdown_ms", None) is None:
            self._cpu_phase_breakdown_ms = _empty_breakdown(CPU_PHASE_BREAKDOWN_KEYS)
        if getattr(self, "_cpu_schedule_breakdown_ms", None) is None:
            self._cpu_schedule_breakdown_ms = _empty_breakdown(
                CPU_SCHEDULE_BREAKDOWN_KEYS
            )

    def _record_cpu_phase_time(
        self: "Scheduler", phase: str, elapsed_ms: Optional[float]
    ) -> None:
        if elapsed_ms is None:
            return
        SchedulerSidecarMixin._ensure_cpu_timing_breakdowns(self)
        self._cpu_phase_breakdown_ms[phase] = self._cpu_phase_breakdown_ms.get(
            phase, 0.0
        ) + max(float(elapsed_ms), 0.0)

    def _record_cpu_schedule_time(
        self: "Scheduler", phase: str, elapsed_ms: Optional[float]
    ) -> None:
        if elapsed_ms is None:
            return
        SchedulerSidecarMixin._ensure_cpu_timing_breakdowns(self)
        self._cpu_schedule_breakdown_ms[phase] = self._cpu_schedule_breakdown_ms.get(
            phase, 0.0
        ) + max(float(elapsed_ms), 0.0)

    def _consume_cpu_timing_breakdowns(
        self: "Scheduler", cpu_iteration_time_ms: Optional[float]
    ) -> tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        SchedulerSidecarMixin._ensure_cpu_timing_breakdowns(self)

        phase_breakdown = {
            key: float(self._cpu_phase_breakdown_ms.get(key, 0.0))
            for key in CPU_PHASE_BREAKDOWN_KEYS
        }
        schedule_breakdown = {
            key: float(self._cpu_schedule_breakdown_ms.get(key, 0.0))
            for key in CPU_SCHEDULE_BREAKDOWN_KEYS
        }

        self._cpu_phase_breakdown_ms = _empty_breakdown(CPU_PHASE_BREAKDOWN_KEYS)
        self._cpu_schedule_breakdown_ms = _empty_breakdown(
            CPU_SCHEDULE_BREAKDOWN_KEYS
        )

        phase_has_values = any(abs(value) > 1e-9 for value in phase_breakdown.values())
        schedule_has_values = any(
            abs(value) > 1e-9 for value in schedule_breakdown.values()
        )

        if cpu_iteration_time_ms is not None:
            phase_total_ms = sum(phase_breakdown.values())
            phase_breakdown["other"] = float(cpu_iteration_time_ms) - phase_total_ms
            phase_has_values = True

        return (
            phase_breakdown if phase_has_values else None,
            schedule_breakdown if schedule_has_values else None,
        )

    def _assemble_and_send_engine_state(self: "Scheduler"):
        from slo_scheduler.messages.engine_state import (
            CurrentSnapshot,
            EngineState,
            FinishedIterationData,
        )

        decision = None
        decision_payload = None
        sidecar_wait_time_ms = None
        sidecar_rpc_breakdown_ms = _empty_breakdown(SIDECAR_RPC_BREAKDOWN_KEYS)
        self._pending_launch_sidecar_wait_time_ms = None
        self._pending_launch_sidecar_rpc_breakdown_ms = None
        assemble_total_start = time.perf_counter()
        if self._sidecar_owner_on_rank() and self.slo_client is not None:
            if self._pending_scheduling is None:
                decision = None
            else:
                build_state_start = time.perf_counter()
                finished = self._pending_finished or FinishedIterationData(
                    iteration_count=self.iteration_count,
                    batch_size_tokens=0,
                    prefill_chunk_pairs=[],
                    kv_tokens_used=0,
                    forward_mode="DECODE",
                    actual_time_ms=0.0,
                )
                current = self._pending_current or CurrentSnapshot(
                    iteration_count=self.iteration_count,
                    timestamp_ms=time.time() * 1000,
                    num_running_requests=(
                        len(self.running_batch.reqs)
                        if self.running_batch is not None
                        and self.running_batch.reqs is not None
                        else 0
                    ),
                    num_waiting_requests=len(self.waiting_queue),
                    kv_tokens_used=self._get_token_info()[0],
                    kv_capacity=self.max_total_num_tokens,
                    running_requests=[],
                )
                accepted_request_ids = list(
                    self._accepted_request_ids_since_last_send
                )
                accepted_requests_count = self._accepted_since_last_send
                if accepted_request_ids or current.num_running_requests > 0:
                    logger.info(
                        "[SIDECAR_ACK_SEND] worker=%s iter=%d accepted_count=%d "
                        "accepted_sample=%s running=%d waiting=%d",
                        self.sidecar_worker_id,
                        self._pending_scheduling.iteration_count,
                        accepted_requests_count,
                        ",".join(accepted_request_ids[:3]),
                        current.num_running_requests,
                        current.num_waiting_requests,
                    )

                state = EngineState(
                    protocol_version=1,
                    min_sidecar_version=1,
                    worker_id=self.sidecar_worker_id,
                    finished=finished,
                    current=current,
                    scheduling=self._pending_scheduling,
                    accepted_requests_count=accepted_requests_count,
                    accepted_request_ids=accepted_request_ids,
                )

                sidecar_rpc_breakdown_ms["build_state"] = (
                    time.perf_counter() - build_state_start
                ) * 1000.0

                (
                    decision,
                    sidecar_wait_time_ms,
                    client_rpc_breakdown_ms,
                    decision_payload,
                ) = self.slo_client.send_and_recv(
                    state, current_iteration=self._pending_scheduling.iteration_count
                )
                for key in ("serialize_send", "wait", "recv_deserialize"):
                    sidecar_rpc_breakdown_ms[key] = client_rpc_breakdown_ms.get(
                        key, 0.0
                    )
            self._accepted_since_last_send = 0
            self._accepted_request_ids_since_last_send = []
            self._pending_finished = None
            self._pending_current = None

        self._pending_launch_sidecar_wait_time_ms = sidecar_wait_time_ms
        self._last_sidecar_full_decision = decision
        sync_start = time.perf_counter()
        decision = self._sync_sidecar_decision(decision, decision_payload)
        sidecar_rpc_breakdown_ms["tp_sync"] = (
            time.perf_counter() - sync_start
        ) * 1000.0
        tracked_rpc_ms = sum(
            sidecar_rpc_breakdown_ms[key]
            for key in SIDECAR_RPC_BREAKDOWN_KEYS
            if key != "other"
        )
        sidecar_rpc_breakdown_ms["other"] = (
            (time.perf_counter() - assemble_total_start) * 1000.0 - tracked_rpc_ms
        )
        if any(abs(value) > 1e-9 for value in sidecar_rpc_breakdown_ms.values()):
            self._pending_launch_sidecar_rpc_breakdown_ms = sidecar_rpc_breakdown_ms
        self._last_sidecar_decision = decision
        return decision

    def _record_sidecar_batch_launch(self: "Scheduler") -> None:
        """Associate the latest sidecar wait time with the batch being launched."""
        if not self._sidecar_owner_on_rank():
            return

        self._launched_batch_sidecar_wait_times_ms.append(
            self._pending_launch_sidecar_wait_time_ms
        )
        self._launched_batch_sidecar_rpc_breakdowns_ms.append(
            self._pending_launch_sidecar_rpc_breakdown_ms
        )
        self._pending_launch_sidecar_wait_time_ms = None
        self._pending_launch_sidecar_rpc_breakdown_ms = None

    def _req_to_request_info(
        self: "Scheduler", req, include_prefill_metadata: bool = True
    ):
        from slo_scheduler.messages.engine_state import RequestInfo

        is_decoding = len(req.output_ids) > 0
        remaining_prefill = 0 if is_decoding else req.extend_input_len
        info = RequestInfo(
            request_id=req.rid,
            target_tpot_ms=req.target_tpot_ms,
            target_ttft_ms=req.target_ttft_ms,
            arrival_time_ms=req.arrival_time_ms,
            tokens_generated=len(req.output_ids),
            remaining_prefill=remaining_prefill,
            slo_violated=req.slo_violated,
        )
        if include_prefill_metadata:
            info.prefix_len = len(req.prefix_indices)
            info.extend_input_len = req.extend_input_len
        return info

    def _req_to_running_request_info(self: "Scheduler", req):
        from slo_scheduler.messages.engine_state import RunningRequestInfo

        is_decoding = len(req.output_ids) > 0
        remaining_prefill = 0 if is_decoding else req.extend_input_len
        return RunningRequestInfo(
            request_id=req.rid,
            arrival_time_ms=req.arrival_time_ms,
            target_tpot_ms=req.target_tpot_ms,
            target_ttft_ms=req.target_ttft_ms,
            tokens_generated=len(req.output_ids),
            remaining_prefill=remaining_prefill,
            slo_violated=req.slo_violated,
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

    def _build_batch_size_by_tpot_tier(self: "Scheduler", batch) -> Dict[str, int]:
        tier_counts: Dict[str, int] = {}
        for req in (getattr(batch, "reqs", None) or []):
            target_tpot_ms = getattr(req, "target_tpot_ms", None)
            tier_key = str(target_tpot_ms) if target_tpot_ms is not None else "none"
            tier_counts[tier_key] = tier_counts.get(tier_key, 0) + 1
        return tier_counts

    def _cache_finished_observability_snapshot(
        self: "Scheduler",
        *,
        iteration_count: int,
        num_running_requests: int,
        num_waiting_requests: int,
        kv_tokens_used: int,
        kv_capacity: int,
        batch_size_tokens: int,
        prefill_chunk_pairs: list[list[int]],
        batch_size_by_tpot_tier: Dict[str, int],
        forward_mode: Optional[str],
    ) -> None:
        snapshots = getattr(self, "_pending_finished_observability_snapshots", None)
        if snapshots is None:
            snapshots = {}
            self._pending_finished_observability_snapshots = snapshots

        snapshots[int(iteration_count)] = _FinishedObservabilitySnapshot(
            num_running_requests=int(num_running_requests),
            num_waiting_requests=int(num_waiting_requests),
            kv_tokens_used=int(kv_tokens_used),
            kv_capacity=int(kv_capacity),
            batch_size_tokens=int(batch_size_tokens),
            prefill_chunk_pairs=[list(pair) for pair in prefill_chunk_pairs],
            batch_size_by_tpot_tier=dict(batch_size_by_tpot_tier),
            forward_mode=forward_mode,
        )
        while len(snapshots) > 4:
            oldest_iteration = min(snapshots)
            snapshots.pop(oldest_iteration, None)

    def _drain_current_snapshot(self: "Scheduler", batch, iteration_count: int) -> None:
        from slo_scheduler.messages.engine_state import CurrentSnapshot

        num_used = self._get_token_info()[0]
        pairs = self._build_prefill_chunk_pairs(batch)
        prefill_chunk_pairs = [[chunk, cumulative] for _, chunk, cumulative in pairs]
        batch_size_tokens = self._compute_batch_size_tokens(batch)
        self._pending_current = CurrentSnapshot(
            iteration_count=iteration_count,
            timestamp_ms=time.time() * 1000,
            num_running_requests=len(batch.reqs) if batch.reqs else 0,
            num_waiting_requests=len(self.waiting_queue),
            kv_tokens_used=num_used,
            kv_capacity=self.max_total_num_tokens,
            running_requests=[
                self._req_to_running_request_info(r) for r in (batch.reqs or [])
            ],
            forward_mode=batch.forward_mode.name,
            batch_size_tokens=batch_size_tokens,
            prefill_chunk_pairs=prefill_chunk_pairs,
        )
        self._cache_finished_observability_snapshot(
            iteration_count=iteration_count,
            num_running_requests=len(batch.reqs) if batch.reqs else 0,
            num_waiting_requests=len(self.waiting_queue),
            kv_tokens_used=num_used,
            kv_capacity=self.max_total_num_tokens,
            batch_size_tokens=batch_size_tokens,
            prefill_chunk_pairs=prefill_chunk_pairs,
            batch_size_by_tpot_tier=self._build_batch_size_by_tpot_tier(batch),
            forward_mode=batch.forward_mode.name,
        )

    def _drain_finished_iteration(
        self: "Scheduler",
        tmp_batch,
        actual_time_ms: float,
        kv_tokens_used: int,
        launch_time_breakdown_ms: Optional[Dict[str, float]] = None,
    ) -> None:
        from slo_scheduler.messages.engine_state import (
            FinishedIterationData,
            ObservabilityBatchSnapshot,
            PrefillChunkPair,
        )

        finished_ts = time.perf_counter()
        cpu_iteration_time_ms = None
        if self._last_finished_iteration_ts is not None:
            cpu_iteration_time_ms = (
                finished_ts - self._last_finished_iteration_ts
            ) * 1000.0
        self._last_finished_iteration_ts = finished_ts

        sidecar_wait_time_ms = None
        if self._launched_batch_sidecar_wait_times_ms:
            sidecar_wait_time_ms = self._launched_batch_sidecar_wait_times_ms.popleft()
        sidecar_rpc_breakdown_ms = None
        if self._launched_batch_sidecar_rpc_breakdowns_ms:
            sidecar_rpc_breakdown_ms = (
                self._launched_batch_sidecar_rpc_breakdowns_ms.popleft()
            )

        (
            cpu_time_breakdown_ms,
            schedule_time_breakdown_ms,
        ) = SchedulerSidecarMixin._consume_cpu_timing_breakdowns(
            self, cpu_iteration_time_ms
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
        cached_observability = None
        observability_snapshots = getattr(
            self, "_pending_finished_observability_snapshots", None
        )
        if observability_snapshots is not None:
            cached_observability = observability_snapshots.pop(
                int(self.iteration_count), None
            )
        observability_snapshot = None
        if cached_observability is not None:
            observability_snapshot = ObservabilityBatchSnapshot(
                num_running_requests=cached_observability.num_running_requests,
                num_waiting_requests=cached_observability.num_waiting_requests,
                kv_tokens_used=cached_observability.kv_tokens_used,
                kv_capacity=cached_observability.kv_capacity,
                batch_size_tokens=cached_observability.batch_size_tokens,
                prefill_chunk_pairs=[
                    list(pair) for pair in cached_observability.prefill_chunk_pairs
                ],
                batch_size_by_tpot_tier=dict(
                    cached_observability.batch_size_by_tpot_tier
                ),
                forward_mode=cached_observability.forward_mode,
            )

        self._pending_finished = FinishedIterationData(
            iteration_count=self.iteration_count,
            batch_size_tokens=self._compute_batch_size_tokens(tmp_batch),
            prefill_chunk_pairs=pairs,
            kv_tokens_used=kv_tokens_used,
            forward_mode=tmp_batch.forward_mode.name,
            actual_time_ms=actual_time_ms,
            cpu_iteration_time_ms=cpu_iteration_time_ms,
            sidecar_wait_time_ms=sidecar_wait_time_ms,
            cpu_time_breakdown_ms=cpu_time_breakdown_ms,
            schedule_time_breakdown_ms=schedule_time_breakdown_ms,
            launch_time_breakdown_ms=launch_time_breakdown_ms,
            sidecar_rpc_breakdown_ms=sidecar_rpc_breakdown_ms,
            observability_snapshot=observability_snapshot,
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
            self._req_to_request_info(req, include_prefill_metadata=False)
            for req in running_reqs
            if len(req.output_ids) > 0
        ]
        chunked_info = (
            self._req_to_request_info(self.chunked_req)
            if self.chunked_req is not None
            else None
        )
        chunked_reqs = [chunked_info] if chunked_info is not None else []
        waiting_reqs = []
        if chunked_info is not None:
            waiting_reqs.append(chunked_info)
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
            page_size=self.page_size,
            last_batch_size=last_batch_size,
        )
