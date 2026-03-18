import sys
import types
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_sidecar_mixin import SchedulerSidecarMixin
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.disaggregation.utils import DisaggregationMode


@dataclass
class PrefillChunkPair:
    request_id: str
    chunk_tokens: int
    cumulative_prefill: int = 0


@dataclass
class FinishedIterationData:
    iteration_count: int
    batch_size_tokens: int
    prefill_chunk_pairs: list[PrefillChunkPair]
    kv_tokens_used: int
    forward_mode: str
    actual_time_ms: float
    completed_decode_lengths: list[int] = field(default_factory=list)


@dataclass
class CurrentSnapshot:
    iteration_count: int
    timestamp_ms: float
    num_running_requests: int
    num_waiting_requests: int
    kv_tokens_used: int
    kv_capacity: int
    running_requests: list = field(default_factory=list)
    forward_mode: str | None = None
    batch_size_tokens: int = 0
    prefill_chunk_pairs: list[list[int]] = field(default_factory=list)
    router_generation: int | None = None
    router_last_ack_id: int | None = None


@dataclass
class SchedulingContext:
    iteration_count: int
    scheduling_time_ms: float
    decode_requests: list = field(default_factory=list)
    chunked_requests: list = field(default_factory=list)
    waiting_requests: list = field(default_factory=list)
    kv_available: int = 0
    kv_capacity: int = 0
    last_batch_size: int | None = None


@dataclass
class EngineState:
    protocol_version: int
    min_sidecar_version: int
    worker_id: str
    finished: FinishedIterationData
    current: CurrentSnapshot
    scheduling: SchedulingContext
    accepted_requests_count: int = 0


class FakeClient:
    def __init__(self, decision=None):
        self.decision = decision
        self.calls = []

    def send_and_recv(self, state, current_iteration):
        self.calls.append((state, current_iteration))
        return self.decision


class FakeRunningBatch:
    def __init__(self, reqs=None):
        self.reqs = reqs or []

    def is_empty(self):
        return len(self.reqs) == 0


class FakeScheduler(SchedulerSidecarMixin):
    def __init__(self):
        self.waiting_queue = []
        self.chunked_req = None
        self.running_batch = FakeRunningBatch()
        self.max_total_num_tokens = 128
        self.iteration_count = 4
        self.worker_id = "127.0.0.1:30000"
        self.init_sidecar(SimpleNamespace(slo_scheduler_addr=None))

    def _get_token_info(self):
        return 33, 0.25, 60, 20


class TestSchedulerSidecarIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_modules = {
            name: sys.modules.get(name)
            for name in (
                "slo_scheduler",
                "slo_scheduler.messages",
                "slo_scheduler.messages.engine_state",
            )
        }
        engine_state_module = types.ModuleType("slo_scheduler.messages.engine_state")
        engine_state_module.PrefillChunkPair = PrefillChunkPair
        engine_state_module.FinishedIterationData = FinishedIterationData
        engine_state_module.CurrentSnapshot = CurrentSnapshot
        engine_state_module.SchedulingContext = SchedulingContext
        engine_state_module.EngineState = EngineState
        sys.modules["slo_scheduler"] = types.ModuleType("slo_scheduler")
        sys.modules["slo_scheduler.messages"] = types.ModuleType("slo_scheduler.messages")
        sys.modules["slo_scheduler.messages.engine_state"] = engine_state_module

    @classmethod
    def tearDownClass(cls):
        for name, module in cls._old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_build_worker_id_matches_old_format(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(host="127.0.0.1", port=30000)
        scheduler.tp_size = 2
        scheduler.tp_rank = 1
        scheduler.dp_size = 2
        scheduler.dp_rank = 0

        worker_id = Scheduler._build_worker_id(scheduler)

        self.assertEqual(worker_id, "127.0.0.1:30000:tp1:dp0")

    def test_assemble_and_send_engine_state_uses_defaults_and_resets_buffers(self):
        scheduler = FakeScheduler()
        scheduler.slo_client = FakeClient(
            decision=SimpleNamespace(iteration_count=5, max_prefill_tokens=77)
        )
        scheduler._pending_scheduling = SchedulingContext(
            iteration_count=5,
            scheduling_time_ms=1.0,
        )
        scheduler._accepted_since_last_send = 3
        scheduler.waiting_queue = [object()]

        scheduler._assemble_and_send_engine_state()

        self.assertEqual(len(scheduler.slo_client.calls), 1)
        state, current_iteration = scheduler.slo_client.calls[0]
        self.assertEqual(current_iteration, 5)
        self.assertEqual(state.worker_id, "127.0.0.1:30000")
        self.assertEqual(state.protocol_version, 1)
        self.assertEqual(state.min_sidecar_version, 1)
        self.assertEqual(state.accepted_requests_count, 3)
        self.assertEqual(state.finished.actual_time_ms, 0.0)
        self.assertEqual(state.current.kv_tokens_used, 33)
        self.assertEqual(scheduler._accepted_since_last_send, 0)
        self.assertIsNone(scheduler._pending_finished)
        self.assertIsNone(scheduler._pending_current)
        self.assertEqual(scheduler._last_sidecar_decision.max_prefill_tokens, 77)

    def test_assemble_and_send_engine_state_reports_idle_state(self):
        scheduler = FakeScheduler()
        scheduler.slo_client = FakeClient(
            decision=SimpleNamespace(iteration_count=5, max_prefill_tokens=64)
        )
        scheduler._pending_scheduling = SchedulingContext(
            iteration_count=5,
            scheduling_time_ms=1.0,
        )

        scheduler._assemble_and_send_engine_state()

        self.assertEqual(len(scheduler.slo_client.calls), 1)
        state, current_iteration = scheduler.slo_client.calls[0]
        self.assertEqual(current_iteration, 5)
        self.assertEqual(state.finished.actual_time_ms, 0.0)
        self.assertEqual(state.current.num_running_requests, 0)
        self.assertEqual(state.current.num_waiting_requests, 0)
        self.assertEqual(state.current.kv_tokens_used, 33)

    def test_get_effective_max_prefill_tokens_prefers_matching_sidecar_decision(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_prefill_tokens = 16384
        scheduler._last_sidecar_decision = SimpleNamespace(
            iteration_count=9, max_prefill_tokens=512
        )

        self.assertEqual(scheduler._get_effective_max_prefill_tokens(9), 512)
        self.assertEqual(scheduler._get_effective_max_prefill_tokens(10), 16384)

    def test_should_skip_prefill_for_sidecar_decision_abandons_chunked_request(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._last_sidecar_decision = SimpleNamespace(
            iteration_count=9,
            max_prefill_tokens=0,
            decode_only_iteration=True,
        )
        scheduler.tree_cache = SimpleNamespace(dec_lock_ref=lambda node, params=None: None)
        requeued = []
        scheduler._add_request_to_queue = (
            lambda req, is_retracted=False: requeued.append((req, is_retracted))
        )

        req = Req(
            rid="chunked",
            origin_input_text="hello",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(max_new_tokens=4),
            time_stats=SchedulerReqTimeStats(),
        )
        req.last_node = object()
        req.is_chunked = 1
        scheduler.chunked_req = req

        skipped = Scheduler._should_skip_prefill_for_sidecar_decision(scheduler, 9)

        self.assertTrue(skipped)
        self.assertIsNone(scheduler.chunked_req)
        self.assertTrue(req.is_retracted)
        self.assertEqual(requeued, [(req, True)])

    def test_normalize_sidecar_arrival_time_fills_missing_value(self):
        scheduler = Scheduler.__new__(Scheduler)
        recv_req = SimpleNamespace(arrival_time_ms=None)

        with patch("sglang.srt.managers.scheduler.time.time", return_value=12.5):
            Scheduler._normalize_sidecar_arrival_time(scheduler, recv_req)

        self.assertEqual(recv_req.arrival_time_ms, 12500.0)

    def test_normalize_sidecar_arrival_time_keeps_existing_value(self):
        scheduler = Scheduler.__new__(Scheduler)
        recv_req = SimpleNamespace(arrival_time_ms=321.0)

        with patch("sglang.srt.managers.scheduler.time.time", return_value=12.5):
            Scheduler._normalize_sidecar_arrival_time(scheduler, recv_req)

        self.assertEqual(recv_req.arrival_time_ms, 321.0)

    def test_add_request_to_queue_counts_marked_generate_request(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.slo_client = object()
        scheduler._accepted_since_last_send = 0
        scheduler.waiting_queue = []
        scheduler._set_or_validate_priority = lambda req: True
        scheduler._abort_on_queued_limit = lambda req: False
        scheduler._prefetch_kvcache = lambda req: None

        req = Req(
            rid="r1",
            origin_input_text="hello",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(max_new_tokens=4),
            time_stats=SchedulerReqTimeStats(),
        )
        req._sidecar_count_as_accepted = True

        Scheduler._add_request_to_queue(scheduler, req)

        self.assertEqual(len(scheduler.waiting_queue), 1)
        self.assertEqual(scheduler._accepted_since_last_send, 1)

    def test_add_request_to_queue_does_not_count_unmarked_request(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.slo_client = object()
        scheduler._accepted_since_last_send = 0
        scheduler.waiting_queue = []
        scheduler._set_or_validate_priority = lambda req: True
        scheduler._abort_on_queued_limit = lambda req: False
        scheduler._prefetch_kvcache = lambda req: None

        req = Req(
            rid="embed-1",
            origin_input_text="hello",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(max_new_tokens=0),
            time_stats=SchedulerReqTimeStats(),
        )
        req._sidecar_count_as_accepted = False

        Scheduler._add_request_to_queue(scheduler, req)

        self.assertEqual(len(scheduler.waiting_queue), 1)
        self.assertEqual(scheduler._accepted_since_last_send, 0)

    def test_event_loop_normal_drains_current_then_finished(self):
        batch = SimpleNamespace(reqs=[object()])
        scheduler = SimpleNamespace()
        scheduler._engine_paused = False
        scheduler.slo_client = object()
        scheduler.iteration_count = 0
        scheduler._pre_batch_kv_used = 0
        scheduler.cur_batch = None
        scheduler.last_batch = None
        scheduler.recv_calls = 0
        scheduler.call_order = []
        scheduler.recv_requests = lambda: (
            [] if (setattr(scheduler, "recv_calls", scheduler.recv_calls + 1) or scheduler.recv_calls) == 1 else (_ for _ in ()).throw(StopIteration())
        )
        scheduler.process_input_requests = lambda recv_reqs: None
        scheduler.get_next_batch_to_run = lambda: batch
        scheduler._get_token_info = lambda: (11, 0.0, 0, 0)
        scheduler.run_batch = lambda b: "result"
        scheduler.process_batch_result = (
            lambda b, r: scheduler.call_order.append(("process", scheduler.iteration_count))
        )
        scheduler._drain_current_snapshot = (
            lambda b, it: scheduler.call_order.append(("current", it))
        )
        scheduler._drain_finished_iteration = (
            lambda b, elapsed, kv: scheduler.call_order.append(("finished", scheduler.iteration_count, kv))
        )
        scheduler.self_check_during_idle = lambda: None
        scheduler.self_check_during_busy = lambda: None
        scheduler.cancel_bubble_timer = lambda: None

        with patch("sglang.srt.managers.scheduler.envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get", return_value=False):
            with self.assertRaises(StopIteration):
                Scheduler.event_loop_normal(scheduler)

        self.assertEqual(scheduler.iteration_count, 1)
        self.assertEqual(
            scheduler.call_order,
            [("current", 1), ("process", 1), ("finished", 1, 11)],
        )

    def test_event_loop_overlap_uses_prev_kv_for_finished_drain(self):
        class Batch:
            def __init__(self):
                self.reqs = [object()]

            def copy(self):
                return self

        batch = Batch()
        scheduler = SimpleNamespace()
        scheduler._engine_paused = False
        scheduler.slo_client = object()
        scheduler.is_generation = False
        scheduler.iteration_count = 0
        scheduler._pre_batch_kv_used = 0
        scheduler._prev_pre_batch_kv_used = 0
        scheduler.last_batch = None
        scheduler.cur_batch = None
        scheduler.recv_index = 0
        scheduler.batch_index = 0
        scheduler.call_order = []

        def recv_requests():
            scheduler.recv_index += 1
            if scheduler.recv_index <= 2:
                return []
            raise StopIteration()

        def get_next_batch_to_run():
            scheduler.batch_index += 1
            return batch if scheduler.batch_index == 1 else None

        scheduler.recv_requests = recv_requests
        scheduler.process_input_requests = lambda recv_reqs: None
        scheduler.get_next_batch_to_run = get_next_batch_to_run
        scheduler.is_disable_overlap_for_batch = lambda batch_obj: False
        scheduler._get_token_info = lambda: (17, 0.0, 0, 0)
        scheduler.run_batch = lambda b: "result"
        scheduler.process_batch_result = (
            lambda b, r: scheduler.call_order.append(("process", scheduler.iteration_count))
        )
        scheduler._drain_current_snapshot = (
            lambda b, it: scheduler.call_order.append(("current", it))
        )
        scheduler._drain_finished_iteration = (
            lambda b, elapsed, kv: scheduler.call_order.append(("finished", scheduler.iteration_count, kv))
        )
        scheduler.cancel_bubble_timer = lambda: None
        scheduler.self_check_during_idle = lambda: None
        scheduler.launch_batch_sample_if_needed = lambda batch_result: None
        scheduler.self_check_during_busy = lambda: None

        with patch("sglang.srt.managers.scheduler.envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get", return_value=False):
            with self.assertRaises(StopIteration):
                Scheduler.event_loop_overlap(scheduler)

        self.assertEqual(scheduler.iteration_count, 1)
        self.assertEqual(
            scheduler.call_order,
            [("current", 1), ("process", 0), ("finished", 1, 17)],
        )


if __name__ == "__main__":
    unittest.main()
