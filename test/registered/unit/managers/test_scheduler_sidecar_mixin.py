import sys
import types
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace

from sglang.srt.managers.scheduler_sidecar_mixin import SchedulerSidecarMixin
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams


@dataclass
class RequestInfo:
    request_id: str
    target_tpot_ms: float | None
    target_ttft_ms: float | None
    arrival_time_ms: float | None
    tokens_generated: int
    prompt_tokens: int
    remaining_prefill: int
    max_new_tokens: int
    slo_violated: bool = False
    prefix_len: int = 0
    extend_input_len: int = 0
    evicted_seqlen_local: int = 0
    router_generation: int | None = None
    router_message_id: int | None = None


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
    running_requests: list[RequestInfo] = field(default_factory=list)
    forward_mode: str | None = None
    batch_size_tokens: int = 0
    prefill_chunk_pairs: list[list[int]] = field(default_factory=list)
    router_generation: int | None = None
    router_last_ack_id: int | None = None


@dataclass
class SchedulingContext:
    iteration_count: int
    scheduling_time_ms: float
    decode_requests: list[RequestInfo] = field(default_factory=list)
    chunked_requests: list[RequestInfo] = field(default_factory=list)
    waiting_requests: list[RequestInfo] = field(default_factory=list)
    kv_available: int = 0
    kv_capacity: int = 0
    last_batch_size: int | None = None


class FakeReq:
    def __init__(
        self,
        rid: str,
        *,
        output_ids=None,
        origin_input_ids=None,
        extend_input_len=0,
        prefix_indices=None,
        target_ttft_ms=None,
        target_tpot_ms=None,
        arrival_time_ms=None,
        slo_violated=False,
        swa_evicted_seqlen=0,
        finished=False,
    ):
        self.rid = rid
        self.output_ids = list(output_ids or [])
        self.origin_input_ids = list(origin_input_ids or [])
        self.extend_input_len = extend_input_len
        self.prefix_indices = list(prefix_indices or [])
        self.target_ttft_ms = target_ttft_ms
        self.target_tpot_ms = target_tpot_ms
        self.arrival_time_ms = arrival_time_ms
        self.slo_violated = slo_violated
        self.swa_evicted_seqlen = swa_evicted_seqlen
        self.is_retracted = False
        self._finished = finished
        self.init_next_round_calls = []
        self.sampling_params = SamplingParams(max_new_tokens=8)

    def finished(self):
        return self._finished

    def init_next_round_input(self, tree_cache=None):
        self.init_next_round_calls.append(tree_cache)
        if self.extend_input_len == 0:
            self.extend_input_len = 6
        if not self.prefix_indices:
            self.prefix_indices = [10, 11]


class FakeScheduler(SchedulerSidecarMixin):
    def __init__(self):
        self.waiting_queue = []
        self.running_batch = SimpleNamespace(reqs=[])
        self.chunked_req = None
        self.last_batch = None
        self.iteration_count = 7
        self.max_total_num_tokens = 128
        self.tree_cache = object()
        self.init_sidecar(SimpleNamespace())

    def _get_token_info(self):
        return 40, 0.3125, 70, 18


class TestSchedulerSidecarMixin(unittest.TestCase):
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
        engine_state_module.RequestInfo = RequestInfo
        engine_state_module.PrefillChunkPair = PrefillChunkPair
        engine_state_module.FinishedIterationData = FinishedIterationData
        engine_state_module.CurrentSnapshot = CurrentSnapshot
        engine_state_module.SchedulingContext = SchedulingContext
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

    def test_req_to_request_info_uses_fresh_runtime_fields(self):
        scheduler = FakeScheduler()
        req = FakeReq(
            "r1",
            output_ids=[5, 6],
            origin_input_ids=[1, 2, 3],
            extend_input_len=4,
            prefix_indices=[0, 1],
            target_ttft_ms=100.0,
            target_tpot_ms=20.0,
            arrival_time_ms=123.0,
            slo_violated=True,
            swa_evicted_seqlen=9,
        )

        info = scheduler._req_to_request_info(req)

        self.assertEqual(info.request_id, "r1")
        self.assertEqual(info.target_ttft_ms, 100.0)
        self.assertEqual(info.target_tpot_ms, 20.0)
        self.assertEqual(info.arrival_time_ms, 123.0)
        self.assertEqual(info.tokens_generated, 2)
        self.assertEqual(info.prompt_tokens, 3)
        self.assertEqual(info.remaining_prefill, 0)
        self.assertTrue(info.slo_violated)
        self.assertEqual(info.prefix_len, 2)
        self.assertEqual(info.extend_input_len, 4)
        self.assertEqual(info.evicted_seqlen_local, 9)
        self.assertIsNone(info.router_generation)

    def test_drain_current_snapshot_mixed_batch(self):
        scheduler = FakeScheduler()
        decode_req = FakeReq(
            "decode", output_ids=[1], origin_input_ids=[1, 2], extend_input_len=1
        )
        prefill_req = FakeReq(
            "prefill", origin_input_ids=[1, 2, 3], extend_input_len=4
        )
        batch = SimpleNamespace(
            reqs=[decode_req, prefill_req],
            forward_mode=ForwardMode.MIXED,
            extend_num_tokens=5,
            decoding_reqs=[decode_req],
            prefix_lens=[2, 3],
            extend_lens=[1, 4],
        )
        scheduler.waiting_queue = [FakeReq("w1")]

        scheduler._drain_current_snapshot(batch, iteration_count=8)

        snapshot = scheduler._pending_current
        self.assertEqual(snapshot.iteration_count, 8)
        self.assertEqual(snapshot.num_running_requests, 2)
        self.assertEqual(snapshot.num_waiting_requests, 1)
        self.assertEqual(snapshot.kv_tokens_used, 40)
        self.assertEqual(snapshot.kv_capacity, 128)
        self.assertEqual(snapshot.forward_mode, "MIXED")
        self.assertEqual(snapshot.batch_size_tokens, 5)
        self.assertEqual(snapshot.prefill_chunk_pairs, [[4, 7]])
        self.assertEqual(
            [r.request_id for r in snapshot.running_requests], ["decode", "prefill"]
        )
        self.assertIsNone(snapshot.router_generation)

    def test_drain_finished_iteration_collects_prefill_and_completed_lengths(self):
        scheduler = FakeScheduler()
        finished_req = FakeReq(
            "finished",
            output_ids=[1, 2, 3],
            origin_input_ids=[9],
            extend_input_len=4,
            finished=True,
        )
        other_req = FakeReq("other", origin_input_ids=[1, 2], extend_input_len=2)
        batch = SimpleNamespace(
            reqs=[finished_req, other_req],
            forward_mode=ForwardMode.EXTEND,
            extend_num_tokens=6,
            decoding_reqs=[],
            prefix_lens=[2, 1],
            extend_lens=[4, 2],
        )

        scheduler._drain_finished_iteration(batch, actual_time_ms=12.5, kv_tokens_used=33)

        finished = scheduler._pending_finished
        self.assertEqual(finished.iteration_count, 7)
        self.assertEqual(finished.batch_size_tokens, 6)
        self.assertEqual(finished.kv_tokens_used, 33)
        self.assertEqual(finished.forward_mode, "EXTEND")
        self.assertEqual(finished.actual_time_ms, 12.5)
        self.assertEqual(finished.completed_decode_lengths, [3])
        self.assertEqual(
            [
                (p.request_id, p.chunk_tokens, p.cumulative_prefill)
                for p in finished.prefill_chunk_pairs
            ],
            [("finished", 4, 6), ("other", 2, 3)],
        )

    def test_drain_scheduling_context_refreshes_waiting_and_chunked_requests(self):
        scheduler = FakeScheduler()
        decode_req = FakeReq(
            "decode", output_ids=[1], origin_input_ids=[1], extend_input_len=1
        )
        waiting_req = FakeReq("waiting", origin_input_ids=[1, 2], extend_input_len=0)
        chunked_req = FakeReq("chunked", origin_input_ids=[1, 2, 3], extend_input_len=0)
        scheduler.running_batch = SimpleNamespace(reqs=[decode_req])
        scheduler.waiting_queue = [waiting_req]
        scheduler.chunked_req = chunked_req
        scheduler.last_batch = SimpleNamespace(reqs=[decode_req, chunked_req])

        scheduler._drain_scheduling_context()

        context = scheduler._pending_scheduling
        self.assertEqual(context.iteration_count, 8)
        self.assertEqual(context.kv_available, 88)
        self.assertEqual(context.kv_capacity, 128)
        self.assertEqual(context.last_batch_size, 2)
        self.assertEqual([r.request_id for r in context.decode_requests], ["decode"])
        self.assertEqual([r.request_id for r in context.chunked_requests], ["chunked"])
        self.assertEqual(
            [r.request_id for r in context.waiting_requests], ["chunked", "waiting"]
        )
        self.assertEqual(chunked_req.init_next_round_calls, [scheduler.tree_cache])
        self.assertEqual(waiting_req.init_next_round_calls, [scheduler.tree_cache])
        self.assertEqual(context.waiting_requests[0].extend_input_len, 6)


if __name__ == "__main__":
    unittest.main()
