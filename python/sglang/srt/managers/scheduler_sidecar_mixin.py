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
PREFILL_SCHEDULE_BREAKDOWN_KEYS = (
    "ready",
    "priority",
    "budget",
    "scan",
    "queue",
    "batch",
    "prepare",
    "mix",
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
        self._pending_launch_prefill_schedule_breakdown_ms = None
        self._launched_batch_sidecar_wait_times_ms = deque()
        self._launched_batch_sidecar_rpc_breakdowns_ms = deque()
        self._launched_batch_prefill_schedule_breakdowns_ms = deque()
        self._last_finished_iteration_ts = None
        self._cpu_phase_breakdown_ms = _empty_breakdown(CPU_PHASE_BREAKDOWN_KEYS)
        self._cpu_schedule_breakdown_ms = _empty_breakdown(
            CPU_SCHEDULE_BREAKDOWN_KEYS
        )
        self._request_info_pool = []
        self._request_info_pool_cursor = 0
        self._running_request_info_pool = []
        self._running_request_info_pool_cursor = 0
        self._prefill_chunk_pair_pool = []
        self._prefill_chunk_pair_pool_cursor = 0
        self._current_running_requests_buffer = []
        self._current_prefill_chunk_pairs_buffer = []
        self._scheduling_decode_requests_buffer = []
        self._scheduling_chunked_requests_buffer = []
        self._scheduling_waiting_requests_buffer = []
        self._finished_prefill_chunk_pairs_buffer = []
        self._finished_completed_decode_lengths_buffer = []
        self._observability_prefill_chunk_pairs_buffer = []
        self._idle_running_requests_buffer = []
        self._idle_prefill_chunk_pairs_buffer = []
        self._current_snapshot_buffer = None
        self._idle_current_snapshot_buffer = None
        self._scheduling_context_buffer = None
        self._finished_iteration_buffer = None
        self._observability_snapshot_buffer = None
        self._engine_state_buffer = None
        self._sidecar_RequestInfo = None
        self._sidecar_RunningRequestInfo = None
        self._sidecar_PrefillChunkPair = None
        self._sidecar_CurrentSnapshot = None
        self._sidecar_FinishedIterationData = None
        self._sidecar_ObservabilityBatchSnapshot = None
        self._sidecar_SchedulingContext = None
        self._sidecar_EngineState = None

    def _ensure_sidecar_message_types(self: "Scheduler") -> None:
        if self._sidecar_RequestInfo is not None:
            return

        from slo_scheduler.messages import engine_state as engine_state_module

        self._sidecar_RequestInfo = getattr(engine_state_module, "RequestInfo", None)
        self._sidecar_RunningRequestInfo = getattr(
            engine_state_module, "RunningRequestInfo", None
        )
        self._sidecar_PrefillChunkPair = getattr(
            engine_state_module, "PrefillChunkPair", None
        )
        self._sidecar_CurrentSnapshot = getattr(
            engine_state_module, "CurrentSnapshot", None
        )
        self._sidecar_FinishedIterationData = getattr(
            engine_state_module, "FinishedIterationData", None
        )
        self._sidecar_ObservabilityBatchSnapshot = getattr(
            engine_state_module, "ObservabilityBatchSnapshot", None
        )
        self._sidecar_SchedulingContext = getattr(
            engine_state_module, "SchedulingContext", None
        )
        self._sidecar_EngineState = getattr(engine_state_module, "EngineState", None)

    @staticmethod
    def _fill_int_pair_buffer(buffer: list[list[int]], pairs) -> None:
        target_len = len(pairs)
        if len(buffer) > target_len:
            del buffer[target_len:]

        for idx, pair in enumerate(pairs):
            left = int(pair[0])
            right = int(pair[1])
            if idx >= len(buffer):
                buffer.append([left, right])
            else:
                item = buffer[idx]
                item[0] = left
                item[1] = right

    def _acquire_request_info(self: "Scheduler"):
        SchedulerSidecarMixin._ensure_sidecar_message_types(self)
        idx = self._request_info_pool_cursor
        if idx >= len(self._request_info_pool):
            self._request_info_pool.append(
                self._sidecar_RequestInfo("", None, None, 0.0, 0, 0)
            )
        self._request_info_pool_cursor = idx + 1
        return self._request_info_pool[idx]

    def _acquire_running_request_info(self: "Scheduler"):
        SchedulerSidecarMixin._ensure_sidecar_message_types(self)
        idx = self._running_request_info_pool_cursor
        if idx >= len(self._running_request_info_pool):
            self._running_request_info_pool.append(
                self._sidecar_RunningRequestInfo("", 0.0)
            )
        self._running_request_info_pool_cursor = idx + 1
        return self._running_request_info_pool[idx]

    def _acquire_prefill_chunk_pair(self: "Scheduler"):
        SchedulerSidecarMixin._ensure_sidecar_message_types(self)
        idx = self._prefill_chunk_pair_pool_cursor
        if idx >= len(self._prefill_chunk_pair_pool):
            self._prefill_chunk_pair_pool.append(
                self._sidecar_PrefillChunkPair("", 0, 0)
            )
        self._prefill_chunk_pair_pool_cursor = idx + 1
        return self._prefill_chunk_pair_pool[idx]

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
        SchedulerSidecarMixin._ensure_sidecar_message_types(self)

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
                finished = self._pending_finished
                if finished is None:
                    finished = self._sidecar_FinishedIterationData(
                        iteration_count=self.iteration_count,
                        batch_size_tokens=0,
                        prefill_chunk_pairs=[],
                        kv_tokens_used=0,
                        forward_mode="DECODE",
                        actual_time_ms=0.0,
                    )
                accepted_request_ids = self._accepted_request_ids_since_last_send
                accepted_requests_count = self._accepted_since_last_send
                current = self._pending_current
                if current is None:
                    current = self._idle_current_snapshot_buffer
                    if current is None:
                        current = self._sidecar_CurrentSnapshot(
                            iteration_count=0,
                            timestamp_ms=0.0,
                            num_running_requests=0,
                            num_waiting_requests=0,
                            kv_tokens_used=0,
                            kv_capacity=0,
                            running_requests=self._idle_running_requests_buffer,
                            prefill_chunk_pairs=self._idle_prefill_chunk_pairs_buffer,
                        )
                        self._idle_current_snapshot_buffer = current
                    current.iteration_count = self.iteration_count
                    current.timestamp_ms = time.time() * 1000
                    current.num_running_requests = (
                        len(self.running_batch.reqs)
                        if self.running_batch is not None
                        and self.running_batch.reqs is not None
                        else 0
                    )
                    current.num_waiting_requests = len(self.waiting_queue)
                    current.kv_tokens_used = self._get_token_info()[0]
                    current.kv_capacity = self.max_total_num_tokens
                    current.forward_mode = None
                    current.batch_size_tokens = 0
                    current.running_requests.clear()
                    current.prefill_chunk_pairs.clear()
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

                state = self._engine_state_buffer
                if state is None:
                    state = self._sidecar_EngineState(
                        protocol_version=1,
                        min_sidecar_version=1,
                        worker_id=self.sidecar_worker_id,
                        finished=finished,
                        current=current,
                        scheduling=self._pending_scheduling,
                        accepted_requests_count=accepted_requests_count,
                        accepted_request_ids=accepted_request_ids,
                    )
                    self._engine_state_buffer = state
                else:
                    state.protocol_version = 1
                    state.min_sidecar_version = 1
                    state.worker_id = self.sidecar_worker_id
                    state.finished = finished
                    state.current = current
                    state.scheduling = self._pending_scheduling
                    state.accepted_requests_count = accepted_requests_count
                    state.accepted_request_ids = accepted_request_ids

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

    def _set_pending_prefill_schedule_breakdown(
        self: "Scheduler", breakdown_ms: Optional[Dict[str, float]]
    ) -> None:
        if not breakdown_ms:
            self._pending_launch_prefill_schedule_breakdown_ms = None
            return

        normalized = {
            key: max(float(breakdown_ms.get(key, 0.0)), 0.0)
            for key in PREFILL_SCHEDULE_BREAKDOWN_KEYS
        }
        other = max(float(breakdown_ms.get("other", 0.0)), 0.0)
        if other > 0.0:
            normalized["other"] = other

        if not any(abs(value) > 1e-9 for value in normalized.values()):
            self._pending_launch_prefill_schedule_breakdown_ms = None
            return

        self._pending_launch_prefill_schedule_breakdown_ms = normalized

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
        self._launched_batch_prefill_schedule_breakdowns_ms.append(
            self._pending_launch_prefill_schedule_breakdown_ms
        )
        self._pending_launch_sidecar_wait_time_ms = None
        self._pending_launch_sidecar_rpc_breakdown_ms = None
        self._pending_launch_prefill_schedule_breakdown_ms = None

    def _req_to_request_info(
        self: "Scheduler", req, include_prefill_metadata: bool = True
    ):
        SchedulerSidecarMixin._ensure_sidecar_message_types(self)
        is_decoding = len(req.output_ids) > 0
        remaining_prefill = 0 if is_decoding else req.extend_input_len
        info = SchedulerSidecarMixin._acquire_request_info(self)
        info.request_id = req.rid
        info.target_tpot_ms = req.target_tpot_ms
        info.target_ttft_ms = req.target_ttft_ms
        info.arrival_time_ms = req.arrival_time_ms
        info.tokens_generated = len(req.output_ids)
        info.remaining_prefill = remaining_prefill
        info.slo_violated = req.slo_violated
        info.prefix_len = len(req.prefix_indices) if include_prefill_metadata else 0
        info.extend_input_len = req.extend_input_len if include_prefill_metadata else 0
        return info

    def _req_to_running_request_info(self: "Scheduler", req):
        SchedulerSidecarMixin._ensure_sidecar_message_types(self)
        is_decoding = len(req.output_ids) > 0
        remaining_prefill = 0 if is_decoding else req.extend_input_len
        info = SchedulerSidecarMixin._acquire_running_request_info(self)
        info.request_id = req.rid
        info.arrival_time_ms = req.arrival_time_ms
        info.target_tpot_ms = req.target_tpot_ms
        info.target_ttft_ms = req.target_ttft_ms
        info.tokens_generated = len(req.output_ids)
        info.remaining_prefill = remaining_prefill
        info.slo_violated = req.slo_violated
        return info

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
        SchedulerSidecarMixin._ensure_sidecar_message_types(self)
        num_used = self._get_token_info()[0]
        pairs = self._build_prefill_chunk_pairs(batch)
        self._running_request_info_pool_cursor = 0
        running_requests = self._current_running_requests_buffer
        running_requests.clear()
        for req in (batch.reqs or []):
            running_requests.append(self._req_to_running_request_info(req))

        prefill_chunk_pairs = self._current_prefill_chunk_pairs_buffer
        SchedulerSidecarMixin._fill_int_pair_buffer(
            prefill_chunk_pairs,
            [(chunk, cumulative) for _, chunk, cumulative in pairs],
        )
        batch_size_tokens = self._compute_batch_size_tokens(batch)
        batch_size_by_tpot_tier = self._build_batch_size_by_tpot_tier(batch)
        current = self._current_snapshot_buffer
        if current is None:
            current = self._sidecar_CurrentSnapshot(
                iteration_count=iteration_count,
                timestamp_ms=time.time() * 1000,
                num_running_requests=len(batch.reqs) if batch.reqs else 0,
                num_waiting_requests=len(self.waiting_queue),
                kv_tokens_used=num_used,
                kv_capacity=self.max_total_num_tokens,
                running_requests=running_requests,
                forward_mode=batch.forward_mode.name,
                batch_size_tokens=batch_size_tokens,
                prefill_chunk_pairs=prefill_chunk_pairs,
            )
            self._current_snapshot_buffer = current
        else:
            current.iteration_count = iteration_count
            current.timestamp_ms = time.time() * 1000
            current.num_running_requests = len(batch.reqs) if batch.reqs else 0
            current.num_waiting_requests = len(self.waiting_queue)
            current.kv_tokens_used = num_used
            current.kv_capacity = self.max_total_num_tokens
            current.running_requests = running_requests
            current.forward_mode = batch.forward_mode.name
            current.batch_size_tokens = batch_size_tokens
            current.prefill_chunk_pairs = prefill_chunk_pairs
        self._pending_current = current
        self._cache_finished_observability_snapshot(
            iteration_count=iteration_count,
            num_running_requests=len(batch.reqs) if batch.reqs else 0,
            num_waiting_requests=len(self.waiting_queue),
            kv_tokens_used=num_used,
            kv_capacity=self.max_total_num_tokens,
            batch_size_tokens=batch_size_tokens,
            prefill_chunk_pairs=prefill_chunk_pairs,
            batch_size_by_tpot_tier=batch_size_by_tpot_tier,
            forward_mode=batch.forward_mode.name,
        )

    def _drain_finished_iteration(
        self: "Scheduler",
        tmp_batch,
        actual_time_ms: float,
        kv_tokens_used: int,
        launch_time_breakdown_ms: Optional[Dict[str, float]] = None,
        launch_forward_breakdown_ms: Optional[Dict[str, float]] = None,
    ) -> None:
        SchedulerSidecarMixin._ensure_sidecar_message_types(self)

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
        prefill_schedule_breakdown_ms = None
        if self._launched_batch_prefill_schedule_breakdowns_ms:
            prefill_schedule_breakdown_ms = (
                self._launched_batch_prefill_schedule_breakdowns_ms.popleft()
            )

        (
            cpu_time_breakdown_ms,
            schedule_time_breakdown_ms,
        ) = SchedulerSidecarMixin._consume_cpu_timing_breakdowns(
            self, cpu_iteration_time_ms
        )

        self._prefill_chunk_pair_pool_cursor = 0
        pairs = self._finished_prefill_chunk_pairs_buffer
        pairs.clear()
        for req, chunk, cumulative in self._build_prefill_chunk_pairs(tmp_batch):
            pair = SchedulerSidecarMixin._acquire_prefill_chunk_pair(self)
            pair.request_id = req.rid
            pair.chunk_tokens = chunk
            pair.cumulative_prefill = cumulative
            pairs.append(pair)
        completed_lengths = self._finished_completed_decode_lengths_buffer
        completed_lengths.clear()
        for req in (tmp_batch.reqs or []):
            if req.finished() and not getattr(req, "is_retracted", False):
                completed_lengths.append(len(req.output_ids))
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
            SchedulerSidecarMixin._fill_int_pair_buffer(
                self._observability_prefill_chunk_pairs_buffer,
                cached_observability.prefill_chunk_pairs,
            )
            observability_snapshot = self._observability_snapshot_buffer
            if observability_snapshot is None:
                observability_snapshot = self._sidecar_ObservabilityBatchSnapshot(
                    num_running_requests=cached_observability.num_running_requests,
                    num_waiting_requests=cached_observability.num_waiting_requests,
                    kv_tokens_used=cached_observability.kv_tokens_used,
                    kv_capacity=cached_observability.kv_capacity,
                    batch_size_tokens=cached_observability.batch_size_tokens,
                    prefill_chunk_pairs=self._observability_prefill_chunk_pairs_buffer,
                    batch_size_by_tpot_tier=dict(
                        cached_observability.batch_size_by_tpot_tier
                    ),
                    forward_mode=cached_observability.forward_mode,
                )
                self._observability_snapshot_buffer = observability_snapshot
            else:
                observability_snapshot.num_running_requests = (
                    cached_observability.num_running_requests
                )
                observability_snapshot.num_waiting_requests = (
                    cached_observability.num_waiting_requests
                )
                observability_snapshot.kv_tokens_used = cached_observability.kv_tokens_used
                observability_snapshot.kv_capacity = cached_observability.kv_capacity
                observability_snapshot.batch_size_tokens = (
                    cached_observability.batch_size_tokens
                )
                observability_snapshot.prefill_chunk_pairs = (
                    self._observability_prefill_chunk_pairs_buffer
                )
                observability_snapshot.batch_size_by_tpot_tier.clear()
                observability_snapshot.batch_size_by_tpot_tier.update(
                    cached_observability.batch_size_by_tpot_tier
                )
                observability_snapshot.forward_mode = cached_observability.forward_mode

        finished = self._finished_iteration_buffer
        if finished is None:
            finished = self._sidecar_FinishedIterationData(
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
                prefill_schedule_breakdown_ms=prefill_schedule_breakdown_ms,
                launch_time_breakdown_ms=launch_time_breakdown_ms,
                launch_forward_breakdown_ms=launch_forward_breakdown_ms,
                sidecar_rpc_breakdown_ms=sidecar_rpc_breakdown_ms,
                observability_snapshot=observability_snapshot,
                completed_decode_lengths=completed_lengths,
            )
            self._finished_iteration_buffer = finished
        else:
            finished.iteration_count = self.iteration_count
            finished.batch_size_tokens = self._compute_batch_size_tokens(tmp_batch)
            finished.prefill_chunk_pairs = pairs
            finished.kv_tokens_used = kv_tokens_used
            finished.forward_mode = tmp_batch.forward_mode.name
            finished.actual_time_ms = actual_time_ms
            finished.cpu_iteration_time_ms = cpu_iteration_time_ms
            finished.sidecar_wait_time_ms = sidecar_wait_time_ms
            finished.cpu_time_breakdown_ms = cpu_time_breakdown_ms
            finished.schedule_time_breakdown_ms = schedule_time_breakdown_ms
            finished.prefill_schedule_breakdown_ms = prefill_schedule_breakdown_ms
            finished.launch_time_breakdown_ms = launch_time_breakdown_ms
            finished.launch_forward_breakdown_ms = launch_forward_breakdown_ms
            finished.sidecar_rpc_breakdown_ms = sidecar_rpc_breakdown_ms
            finished.observability_snapshot = observability_snapshot
            finished.completed_decode_lengths = completed_lengths
        self._pending_finished = finished

    def _drain_scheduling_context(self: "Scheduler") -> None:
        SchedulerSidecarMixin._ensure_sidecar_message_types(self)
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

        self._request_info_pool_cursor = 0
        running_reqs = self.running_batch.reqs if self.running_batch else []
        decode_reqs = self._scheduling_decode_requests_buffer
        decode_reqs.clear()
        for req in running_reqs:
            if len(req.output_ids) > 0:
                decode_reqs.append(
                    self._req_to_request_info(req, include_prefill_metadata=False)
                )
        chunked_info = (
            self._req_to_request_info(self.chunked_req)
            if self.chunked_req is not None
            else None
        )
        chunked_reqs = self._scheduling_chunked_requests_buffer
        chunked_reqs.clear()
        if chunked_info is not None:
            chunked_reqs.append(chunked_info)
        waiting_reqs = self._scheduling_waiting_requests_buffer
        waiting_reqs.clear()
        if chunked_info is not None:
            waiting_reqs.append(chunked_info)
        for req in self.waiting_queue:
            waiting_reqs.append(self._req_to_request_info(req))

        last_batch_size = None
        if self.last_batch is not None and getattr(self.last_batch, "reqs", None) is not None:
            last_batch_size = len(self.last_batch.reqs)

        scheduling = self._scheduling_context_buffer
        if scheduling is None:
            scheduling = self._sidecar_SchedulingContext(
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
            self._scheduling_context_buffer = scheduling
        else:
            scheduling.iteration_count = self.iteration_count + 1
            scheduling.scheduling_time_ms = time.time() * 1000
            scheduling.decode_requests = decode_reqs
            scheduling.chunked_requests = chunked_reqs
            scheduling.waiting_requests = waiting_reqs
            scheduling.kv_available = kv_available
            scheduling.kv_capacity = self.max_total_num_tokens
            scheduling.page_size = self.page_size
            scheduling.last_batch_size = last_batch_size
        self._pending_scheduling = scheduling
