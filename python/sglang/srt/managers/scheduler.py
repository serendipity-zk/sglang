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
"""A scheduler that manages a tensor parallel GPU worker."""

import faulthandler
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from concurrent import futures
from dataclasses import dataclass
from http import HTTPStatus
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union
import copy

from jwt import decode
import psutil
import setproctitle
import torch
import zmq
from torch.distributed import barrier

PREFILL_SIM_EXTRAS = [128, 256, 384, 512, 768, 1024, 2048, 4096, 8192]

from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    create_grammar_backend,
)
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    prepare_abort,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    DestroyWeightsUpdateGroupReqInput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    FreezeGCReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadReqInput,
    GetLoadReqOutput,
    GetUIMetricsReqInput,
    GetUIMetricsReqOutput,
    GetWeightsByNameReqInput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    MultiTokenizerRegisterReq,
    MultiTokenizerWrapper,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SetTPOTReqInput,
    SetTPOTReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.mm_utils import init_embedding_cache
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    MultimodalInputs,
    Req,
    RequestStage,
    ScheduleBatch,
    global_server_args_dict,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    PrefillScheduleMode,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.scheduler_metrics_mixin import (
    RECORD_STEP_TIME,
    SchedulerMetricsMixin,
)
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin
from sglang.srt.managers.scheduler_sidecar_mixin import SchedulerSidecarMixin
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
from sglang.srt.managers.iteration_target import compute_iteration_target
from sglang.srt.managers.scheduler_update_weights_mixin import (
    SchedulerUpdateWeightsMixin,
)
from sglang.srt.managers.session_controller import Session
from sglang.srt.managers.router_message_tracker import RouterMessageAckTracker
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.tp_worker_overlap_thread import TpModelWorkerClient
from sglang.srt.managers.utils import DPBalanceMeta, validate_input_length
from sglang.srt.mem_cache.chunk_cache import ChunkCache, SWAChunkCache
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter

# Import cycle time predictors for SLO-aware scheduling
sys.path.insert(0, "/sgl-workspace/sglang")
sys.path.insert(0, "/sgl-workspace/sglang/rust_predictor")
from sglang_profile.grid_predictor import GridBasedCycleTimePredictor
from sglang_profile.mode_aware_predictor import ModeAwarePredictor
from exp.prefill_multi_sim import PrefillSimulatorEngine
from exp.length_sim_algo import MonteCarloOutputEst
from sglang.srt.tracing.trace import (
    process_tracing_init,
    trace_event,
    trace_set_proc_propagate_context,
    trace_set_thread_info,
    trace_slice,
    trace_slice_end,
    trace_slice_start,
)
from sglang.srt.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.utils import (
    DynamicGradMode,
    broadcast_pyobj,
    configure_gc_logger,
    configure_logger,
    disable_request_logging,
    freeze_gc,
    get_available_gpu_memory,
    get_bool_env_var,
    get_int_env_var,
    get_zmq_socket,
    is_cpu,
    kill_itself_when_parent_died,
    numa_bind_to_node,
    point_to_point_pyobj,
    pyspy_dump_schedulers,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)

# Test retract decode for debugging purposes
TEST_RETRACT = get_bool_env_var("SGLANG_TEST_RETRACT")
GRAMMAR_TIMEOUT = float(os.environ.get("SGLANG_GRAMMAR_TIMEOUT", 300))

_is_cpu = is_cpu()


def _forward_mode_to_string(forward_mode: ForwardMode) -> Optional[str]:
    """Convert ForwardMode enum to string for mode-aware predictor."""
    mode_map = {
        ForwardMode.DECODE: "DECODE",
        ForwardMode.EXTEND: "EXTEND",
        ForwardMode.MIXED: "MIXED",
    }
    return mode_map.get(forward_mode, None)


@dataclass
class GenerationBatchResult:
    logits_output: Optional[LogitsProcessorOutput]
    pp_hidden_states_proxy_tensors: Optional[torch.Tensor]
    next_token_ids: Optional[List[int]]
    extend_input_len_per_req: List[int]
    extend_logprob_start_len_per_req: List[int]
    bid: int
    can_run_cuda_graph: bool


@dataclass
class EmbeddingBatchResult:
    embeddings: torch.Tensor
    bid: int


class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerUpdateWeightsMixin,
    SchedulerProfilerMixin,
    SchedulerMetricsMixin,
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerSidecarMixin,
):
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        dp_balance_meta: Optional[DPBalanceMeta] = None,
    ):
        # Parse args
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.tp_size = server_args.tp_size
        self.moe_ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.dp_size = server_args.dp_size
        self.schedule_policy = server_args.schedule_policy
        self.enable_priority_scheduling = server_args.enable_priority_scheduling
        self.schedule_low_priority_values_first = (
            server_args.schedule_low_priority_values_first
        )
        self.priority_scheduling_preemption_threshold = (
            server_args.priority_scheduling_preemption_threshold
        )
        self.enable_lora = server_args.enable_lora
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = not server_args.disable_overlap_schedule
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.enable_metrics = server_args.enable_metrics
        self.enable_metrics_for_all_schedulers = (
            server_args.enable_metrics_for_all_schedulers
        )
        self.enable_kv_cache_events = server_args.kv_events_config is not None
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.gpu_id = gpu_id
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        self.enable_hicache_storage = server_args.hicache_storage_backend is not None
        self.page_size = server_args.page_size
        self.router_ack_tracker = RouterMessageAckTracker()
        # Track idle reporting cadence
        self._last_idle_router_report_time: Optional[float] = None
        self._idle_router_report_interval = 0.02  # seconds
        self.last_cycle_time_prediction = 0

        # Timestamp of the last completed process_batch_result call
        self._last_process_result_end_time: Optional[float] = None
        # Timestamp of the last payload we submitted to the detokenizer
        self._last_detokenizer_submit_time: Optional[float] = None

        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
            )
        )

        # Init model config
        self.model_config = ModelConfig.from_server_args(server_args)

        # Init inter-process communication
        context = zmq.Context(2)
        self.idle_sleeper = None
        if self.pp_rank == 0 and self.attn_tp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            self.recv_from_rpc = get_zmq_socket(
                context, zmq.DEALER, port_args.rpc_ipc_name, False
            )

            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )
            if server_args.skip_tokenizer_init:
                # Directly send to the TokenizerManager
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )

            if self.server_args.sleep_on_idle:
                self.idle_sleeper = IdleSleeper(
                    [
                        self.recv_from_tokenizer,
                        self.recv_from_rpc,
                    ]
                )
        else:
            self.recv_from_tokenizer = None
            self.recv_from_rpc = None
            self.send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)

        if self.current_scheduler_metrics_enabled():
            self.send_metrics_from_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.metrics_ipc_name, False
            )

        # Init SLO scheduler sidecar client and drain buffers
        self.init_sidecar(server_args)

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config
        self.init_moe_config()

        # Set reasoning_parser and think_end_id if --reasoning_parser is enabled
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=self.server_args.reasoning_parser, stream_reasoning=False
            )
            self.tokenizer.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )[0]

        # Check whether overlap can be enabled
        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")

        # Launch a tensor parallel worker
        if self.enable_overlap:
            TpWorkerClass = TpModelWorkerClient
        else:
            TpWorkerClass = TpModelWorker

        self.tp_worker = TpWorkerClass(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            nccl_port=port_args.nccl_port,
        )

        # Launch a draft worker for speculative decoding
        if self.spec_algorithm.is_eagle():
            from sglang.srt.speculative.eagle_worker import EAGLEWorker

            self.draft_worker = EAGLEWorker(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                server_args=server_args,
                nccl_port=port_args.nccl_port,
                target_worker=self.tp_worker,
                dp_rank=dp_rank,
            )
        elif self.spec_algorithm.is_standalone():
            from sglang.srt.speculative.standalone_worker import StandaloneWorker

            self.draft_worker = StandaloneWorker(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                server_args=server_args,
                nccl_port=port_args.nccl_port,
                target_worker=self.tp_worker,
                dp_rank=dp_rank,
            )
        elif self.spec_algorithm.is_lookahead():
            from sglang.srt.speculative.lookahead_worker import LOOKAHEADWorker

            self.draft_worker = LOOKAHEADWorker(
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                moe_ep_rank=moe_ep_rank,
                server_args=server_args,
                nccl_port=port_args.nccl_port,
                target_worker=self.tp_worker,
                dp_rank=dp_rank,
            )
        else:
            self.draft_worker = None

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            worker_global_server_args_dict,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        if global_server_args_dict["max_micro_batch_size"] is None:
            global_server_args_dict["max_micro_batch_size"] = max(
                self.max_running_requests // server_args.pp_size, 1
            )

        self.tp_group = self.tp_worker.get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = self.tp_worker.get_attention_tp_group()
        self.attn_tp_cpu_group = self.tp_worker.get_attention_tp_cpu_group()
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        global_server_args_dict.update(worker_global_server_args_dict)
        set_random_seed(self.random_seed)

        # Hybrid memory pool
        self.is_hybrid = self.tp_worker.is_hybrid
        if self.is_hybrid:
            self.sliding_window_size = self.tp_worker.sliding_window_size
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()
            )

        # Print debug info
        if tp_rank == 0:
            avail_mem = get_available_gpu_memory(
                self.device, self.gpu_id, empty_cache=False
            )
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={server_args.chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}={avail_mem:.2f} GB"
            )

        # Init memory pool and cache
        self.init_memory_pool_and_cache()

        # Init running status
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_prefill_tokens = 0
        self.last_decode_stats_tic = time.perf_counter()
        self.last_prefill_stats_tic = time.perf_counter()
        self.return_health_check_ct = 0
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
        self.kv_transfer_speed_gb_s: float = 0.0
        self.kv_transfer_latency_ms: float = 0.0
        self.sessions: Dict[str, Session] = {}
        self.current_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.current_stream.synchronize = lambda: None  # No-op for CPU
        self.forward_sleep_time = None

        # Init chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:  # -1 means disable
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )

        # Init the grammar backend for constrained generation
        self.grammar_queue: List[Req] = []
        if not server_args.skip_tokenizer_init:
            self.grammar_backend = create_grammar_backend(
                server_args,
                self.tokenizer,
                self.model_config.vocab_size,
                self.model_config.hf_eos_token_id,
            )
        else:
            self.grammar_backend = None

        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
            self.enable_priority_scheduling,
            self.schedule_low_priority_values_first,
        )
        # Enable preemption for priority scheduling.
        self.try_preemption = self.enable_priority_scheduling

        assert (
            server_args.schedule_conservativeness >= 0
        ), "Invalid schedule_conservativeness"
        self.init_new_token_ratio = min(
            global_config.default_init_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio
            * global_config.default_min_new_token_ratio_factor,
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / global_config.default_new_token_ratio_decay_steps
        self.new_token_ratio = self.init_new_token_ratio

        # Init watchdog thread
        self.watchdog_timeout = server_args.watchdog_timeout
        t = threading.Thread(target=self.watchdog_thread, daemon=True)
        t.start()
        self.parent_process = psutil.Process().parent()

        # Init memory saver, profiler and metric stats
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        self.offload_tags = set()
        self.init_profiler()

        self.recv_skipper = SchedulerRecvSkipper.maybe_create(server_args)
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Init metrics stats
        self.init_metrics(tp_rank, pp_rank, dp_rank)
        self.init_kv_events(server_args.kv_events_config)
        self.init_dp_balance(dp_balance_meta)

        # Init disaggregation
        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.init_disaggregation()

        if get_bool_env_var("SGLANG_GC_LOG"):
            configure_gc_logger()

        # Init prefill kv split size when deterministic inference is enabled with various attention backends
        self.init_deterministic_inference_config()

        # Global TPOT (cycle time) regulator; set via /set_tpot or --default-tpot-ms
        self.tpot: Optional[float] = self.server_args.default_tpot_ms

        # Iteration counter - tracks completed GPU iterations
        self.iteration_count = 0

        # Rolling history for iteration durations (ms) + derived target for next iteration
        self.iteration_time_history: deque = deque(maxlen=10)
        self.target_iteration_time_ms: Optional[float] = None
        self.last_iteration_time_ms: Optional[float] = None

        # Cycle time predictor for SLO-aware scheduling (optional)
        self.cycle_time_predictor = None
        if self.server_args.enable_iteration_metrics:
            if self.server_args.predictor_type == "rust":
                try:
                    from rust_predictor.rust_mode_aware_wrapper import RustModeAwarePredictor
                    logger.info(
                        f"Initializing RustModeAwarePredictor with grid: {self.server_args.predictor_grid_path}"
                    )
                    if self.server_args.predictor_log_path:
                        logger.info(
                            f"Predictor CSV logging to: {self.server_args.predictor_log_path}"
                        )
                    self.cycle_time_predictor = RustModeAwarePredictor(
                        grid_path=self.server_args.predictor_grid_path,
                        csv_log_path=self.server_args.predictor_log_path,
                    )
                except ImportError as e:
                    logger.warning(
                        f"Failed to import RustModeAwarePredictor: {e}. "
                        "Install knn_workload_predictor wheel built by maturin. "
                        "Prediction disabled."
                    )
            elif self.server_args.predictor_type == "mode_aware":
                logger.info(
                    f"Initializing ModeAwarePredictor with grid: {self.server_args.predictor_grid_path}"
                )
                if self.server_args.predictor_log_path:
                    logger.info(
                        f"Predictor CSV logging to: {self.server_args.predictor_log_path}"
                    )
                self.cycle_time_predictor = ModeAwarePredictor(
                    grid_path=self.server_args.predictor_grid_path,
                    log_every=100,
                    alpha=0.6,
                    k=64,
                    radius=0.30,
                    bandwidth=0.20,
                    half_life=50.0,
                    W0=4.0,
                    csv_log_path=self.server_args.predictor_log_path,
                )
            else:  # "grid" (old single-mode predictor)
                logger.info(
                    f"Initializing GridBasedCycleTimePredictor with grid: {self.server_args.predictor_grid_path}"
                )
                self.cycle_time_predictor = GridBasedCycleTimePredictor(
                    grid_path=self.server_args.predictor_grid_path,
                    log_every=100,
                    alpha=0.6,
                    k=64,
                    radius=0.30,
                    bandwidth=0.20,
                    half_life=50.0,
                    W0=4.0,
                )

        # Prefill schedule mode for SLO-aware scheduling
        self.prefill_schedule_mode = PrefillScheduleMode(
            server_args.prefill_schedule_mode
        )
        logger.info(f"Prefill schedule mode: {self.prefill_schedule_mode.value}")
        if self.tpot is not None:
            logger.info(f"Default TPOT target: {self.tpot}ms")

        # Track output length observations + forecast future KV peak/slack
        self.output_estimator = (
            MonteCarloOutputEst(predictor=self.cycle_time_predictor)
            if self.is_generation
            else None
        )
        self.prefill_sim_engine: Optional[PrefillSimulatorEngine] = None
        self._last_prefill_sim_results: Optional[List] = None
        self._prefill_sim_runtime_ms: Optional[float] = None  # Algorithm execution time
        self._predicted_ttft_for_new_admits: Optional[Dict[int, float]] = None  # Maps extra_len -> predicted TTFT (ms) for router
        # Track last scenario to avoid duplicate logging
        self._last_prefill_sim_decode_batch: Optional[int] = None
        self._last_prefill_sim_kv_cache: Optional[int] = None
        self._last_prefill_sim_prefill_lens: Optional[List[int]] = None
        self._last_kv_forecast: Optional[Tuple[float, float]] = None
        self._last_kv_forecast_gt: Optional[Tuple[float, float]] = None
        self._last_kv_forecast_time_ms: Optional[float] = None

        # Init request dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.flush_cache_wrapped),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
                (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
                (DestroyWeightsUpdateGroupReqInput, self.destroy_weights_update_group),
                (
                    InitWeightsSendGroupForRemoteInstanceReqInput,
                    self.init_weights_send_group_for_remote_instance,
                ),
                (
                    SendWeightsToRemoteInstanceReqInput,
                    self.send_weights_to_remote_instance,
                ),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.update_weights_from_distributed,
                ),
                (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
                (GetWeightsByNameReqInput, self.get_weights_by_name),
                (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),
                (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),
                (SlowDownReqInput, self.slow_down),
                (ProfileReq, self.profile),
                (FreezeGCReq, self.handle_freeze_gc),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (SetTPOTReqInput, self.set_tpot),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
                (MultiTokenizerRegisterReq, self.register_multi_tokenizer),
                (GetLoadReqInput, self.get_load),
                (GetUIMetricsReqInput, self.get_ui_metrics),
            ]
        )

        # Initialize iteration metrics reporting
        self.worker_id = self._build_worker_id()
        if server_args.enable_iteration_metrics:
            self._init_iteration_metrics()

    def _build_worker_id(self) -> str:
        """Build unique worker identifier."""
        worker_id = f"{self.server_args.host}:{self.server_args.port}"
        if self.tp_size > 1:
            worker_id += f":tp{self.tp_rank}"
        if self.dp_size > 1 and self.dp_rank is not None:
            worker_id += f":dp{self.dp_rank}"
        return worker_id

    def _get_stat_file_path(self) -> Optional[str]:
        """
        Get the .stat file path based on main log configuration.

        Pattern: If main log is /path/to/server.log,
                 stat file is /path/to/server.log.stat
        """
        import tempfile
        from pathlib import Path

        # Try to determine the main log file path from Python logging
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                base_log_path = handler.baseFilename
                stat_path = f"{base_log_path}.stat"
                return stat_path

        # Fallback: use temp directory
        stat_path = (
            Path(tempfile.gettempdir())
            / f"sglang_server_{self.worker_id.replace(':', '_')}.stat"
        )
        return str(stat_path)

    def _init_iteration_metrics(self) -> None:
        """Initialize iteration metrics reporting."""
        from sglang.srt.ui import iteration_metrics

        iteration_metrics.initialize(
            worker_id=self.worker_id,
            router_url=self.server_args.router_metrics_url,
        )
        logger.info(f"Iteration metrics enabled for worker: {self.worker_id}")

    def _collect_and_report_iteration_metrics(
        self,
        batch: ScheduleBatch,
        iteration_time_ms: float = None,
        destinations: list = None,
    ) -> None:
        """
        Collect iteration metrics and report via iteration_metrics module.

        Args:
            batch: The batch that was just executed
            iteration_time_ms: Actual execution time in milliseconds (None for running batch)
            destinations: List of destinations ["log", "ui", "router", "debug"]
                         (None defaults to ["log", "ui", "router"])
        """
        if not self.server_args.enable_iteration_metrics:
            return

        # In sidecar mode, engine does not report to router — sidecar handles it
        if self.slo_scheduler_mode == "sidecar":
            if destinations is None:
                destinations = ["log", "ui"]
            elif "router" in destinations:
                destinations = [d for d in destinations if d != "router"]

        # Auto-include debug destination for completed iterations when flag is set
        if (getattr(self.server_args, "enable_debug_metrics", False)
                and iteration_time_ms is not None):
            if destinations is None:
                destinations = ["log", "ui", "router", "debug"]
            elif "debug" not in destinations:
                destinations = list(destinations) + ["debug"]

        num_batch_reqs = len(batch.reqs) if batch.reqs is not None else 0
        if num_batch_reqs > 0:
            # Activity observed; allow the next idle transition to report immediately
            self._last_idle_router_report_time = None

        from sglang.srt.ui import iteration_metrics

        # Get KV cache stats
        num_used, token_usage, available_size, evictable_size = self._get_token_info()

        # Determine prefill/decode token counts based on forward mode
        from sglang.srt.managers.schedule_batch import ForwardMode

        prefill_tokens = 0
        decode_tokens = 0
        decoding_reqs = set(batch.decoding_reqs) if getattr(batch, "decoding_reqs", None) else None
        if batch.forward_mode == ForwardMode.EXTEND:
            prefill_tokens = batch.extend_num_tokens if batch.extend_num_tokens else 0
        elif batch.forward_mode == ForwardMode.DECODE:
            decode_tokens = len(batch.reqs)
        elif batch.forward_mode == ForwardMode.MIXED:
            # Mixed mode: has both prefill and decode, where prefill should exclude decode tokens
            prefill_tokens = batch.extend_num_tokens - len(batch.decoding_reqs) if batch.extend_num_tokens else 0
            decode_tokens = len(batch.decoding_reqs) if batch.decoding_reqs else 0

        total_tokens = prefill_tokens + decode_tokens

        # Collect prefill chunk pairs for chunked prefill requests
        # Each pair is (current_chunk_length, cumulative_prefill_length)
        # For non-chunked requests, this becomes (seqlen, seqlen)
        # For decode-only batches, this is an empty list
        prefill_chunk_pairs = []
        if batch.forward_mode == ForwardMode.EXTEND or batch.forward_mode == ForwardMode.MIXED:
            # Prefer stable lengths captured in the batch at preparation time
            prefix_lens = getattr(batch, "prefix_lens", None)
            extend_lens = getattr(batch, "extend_lens", None)
            if prefix_lens is not None and extend_lens is not None and batch.reqs is not None:
                for i, req in enumerate(batch.reqs):
                    if batch.forward_mode == ForwardMode.MIXED and decoding_reqs and req in decoding_reqs:
                        continue
                    current_chunk = extend_lens[i] if i < len(extend_lens) else 0
                    if current_chunk and current_chunk > 0:
                        cumulative_prefill = (prefix_lens[i] if i < len(prefix_lens) else 0) + current_chunk
                        prefill_chunk_pairs.append([int(current_chunk), int(cumulative_prefill)])
            else:
                # Fallback: derive from possibly mutable reqs (non-overlap paths)
                for req in batch.reqs or []:
                    if batch.forward_mode == ForwardMode.MIXED and decoding_reqs and req in decoding_reqs:
                        continue
                    current_chunk = getattr(req, "extend_input_len", 0)
                    if current_chunk and current_chunk > 0:
                        prefix_len = len(getattr(req, "prefix_indices", []))
                        cumulative_prefill = prefix_len + current_chunk
                        prefill_chunk_pairs.append([current_chunk, cumulative_prefill])

        # Count batch size per TPOT tier
        batch_size_by_tpot_tier = {}
        for req in batch.reqs or []:
            tpot_key = str(req.target_tpot_ms) if req.target_tpot_ms is not None else "none"
            batch_size_by_tpot_tier[tpot_key] = batch_size_by_tpot_tier.get(tpot_key, 0) + 1

        # Build metrics payload (field names match router and UI expectations)
        metrics = {
            # UI expects: running_batch_size, queue_reqs, kv_tokens_used, token_capacity
            "running_batch_size": num_batch_reqs,
            "queue_reqs": len(self.waiting_queue),
            "waiting_queue_size": len(self.waiting_queue),  # Router expects this name
            "kv_tokens_used": num_used,
            "token_capacity": self.max_total_num_tokens,
            "kv_usage_pct": round(token_usage * 100, 2),  # For UI (as percentage)
            "kv_cache_usage_pct": round(token_usage, 4),  # For router (as fraction 0.0-1.0)
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "token_batch_size": total_tokens,
            "forward_mode": batch.forward_mode.name if batch.forward_mode else "UNKNOWN",
            # KV forecast (predicted + ground truth upper bound) captured during scheduling
            "kv_forecast_peak": self._last_kv_forecast[0] if self._last_kv_forecast else None,
            "kv_forecast_slack_ms": self._last_kv_forecast[1] if self._last_kv_forecast else None,
            "kv_forecast_peak_gt": self._last_kv_forecast_gt[0] if self._last_kv_forecast_gt else None,
            "kv_forecast_slack_ms_gt": self._last_kv_forecast_gt[1] if self._last_kv_forecast_gt else None,
            "kv_forecast_time_ms": round(self._last_kv_forecast_time_ms, 3) if self._last_kv_forecast_time_ms is not None else None,
            "prefill_sim_time_ms": round(self._prefill_sim_runtime_ms, 3) if self._prefill_sim_runtime_ms is not None else None,
            # Prefill chunk pairs: list of [current_chunk, cumulative_prefill]
            "prefill_chunk_pairs": prefill_chunk_pairs,
            "prefill_sim_results": self._predicted_ttft_for_new_admits,  # Predicted TTFT for different admission sizes
            # Keep some old names for compatibility
            "batch_size_tokens": total_tokens,
            "num_requests": num_batch_reqs,
            "input_id_len": batch.input_ids.shape[0] if batch.input_ids is not None else 0,
            # Per-TPOT-tier batch sizes
            "batch_size_by_tpot_tier": batch_size_by_tpot_tier,
        }

        # Compute slack for running batch and waiting queue using router arrival time and SLO hints.
        now_ms = time.time() * 1000.0
        running_slacks: List[Dict[str, Union[str, float]]] = []
        queue_slacks: List[Dict[str, Union[str, float]]] = []
        min_decode_slack_val: Optional[float] = None
        min_decode_slack_rid: Optional[str] = None

        for req in batch.reqs or []:
            slack = self._compute_req_slack_ms(req, now_ms=now_ms)
            if slack is not None:
                running_slacks.append({
                    "rid": req.rid,
                    "slack": slack,
                    "output_len": len(req.output_ids),
                    "arrival_time_ms": req.arrival_time_ms,
                    "current_time_ms": now_ms,
                    "target_ttft_ms": req.target_ttft_ms,
                    "target_tpot_ms": req.target_tpot_ms,
                })
                is_decode = (
                    batch.forward_mode == ForwardMode.DECODE
                    or (batch.forward_mode == ForwardMode.MIXED and decoding_reqs and req in decoding_reqs)
                )
                if is_decode:
                    if min_decode_slack_val is None or slack < min_decode_slack_val:
                        min_decode_slack_val = slack
                        min_decode_slack_rid = req.rid

        for req in self.waiting_queue:
            slack = self._compute_req_slack_ms(req, now_ms=now_ms)
            if slack is not None:
                queue_slacks.append({
                    "rid": req.rid,
                    "slack": slack,
                    "output_len": len(req.output_ids),
                    "arrival_time_ms": req.arrival_time_ms,
                    "current_time_ms": now_ms,
                    "target_ttft_ms": req.target_ttft_ms,
                    "target_tpot_ms": req.target_tpot_ms,
                })

        def _slack_stats(entries: List[Dict[str, Union[str, float]]]):
            if not entries:
                return None
            values = [entry["slack"] for entry in entries]
            return {
                "min_ms": round(min(values), 2),
                "max_ms": round(max(values), 2),
                "avg_ms": round(sum(values) / len(values), 2),
            }

        batch_slack_stats = _slack_stats(running_slacks)
        queue_slack_stats = _slack_stats(queue_slacks)
        if batch_slack_stats:
            metrics["running_slack_ms"] = batch_slack_stats
        if queue_slack_stats:
            metrics["queue_slack_ms"] = queue_slack_stats
        if min_decode_slack_val is not None:
            metrics["min_decode_slack_ms"] = round(min_decode_slack_val, 2)
            if min_decode_slack_rid is not None:
                metrics["min_decode_slack_rid"] = min_decode_slack_rid

        if running_slacks or queue_slacks:
            combined = [
                ("running", entry["rid"], entry["slack"], entry["arrival_time_ms"], entry["current_time_ms"], entry["output_len"], entry["target_ttft_ms"], entry["target_tpot_ms"])
                for entry in running_slacks
            ] + [
                ("waiting", entry["rid"], entry["slack"], entry["arrival_time_ms"], entry["current_time_ms"], entry["output_len"], entry["target_ttft_ms"], entry["target_tpot_ms"])
                for entry in queue_slacks
            ]
            combined.sort(key=lambda x: x[2])  # most negative / worst slack first
            worst_samples = [
                {"rid": rid, "stage": stage, "arrival_time_ms": arrival_time_ms, "current_time_ms": current_time_ms, "diff": current_time_ms - arrival_time_ms, "slack_ms": round(slack, 2), "output_len": output_len, "target_ttft_ms": target_ttft_ms, "target_tpot_ms": target_tpot_ms}
                for stage, rid, slack, arrival_time_ms, current_time_ms, output_len, target_ttft_ms, target_tpot_ms in combined[:1]
            ]
            metrics["slack_worst_samples"] = worst_samples

        # Add timing fields only if iteration_time_ms is provided (finishing batch)
        if iteration_time_ms is not None:
            # iteration_time_ms semantically represents elapsed iteration time; we now pass gpu_elapsed_ms here
            metrics["iteration_time_ms"] = round(iteration_time_ms, 2)
            self.last_iteration_time_ms = float(iteration_time_ms)
            if self.slo_scheduler_mode != "sidecar":
                self._record_iteration_time(iteration_time_ms)

        if self.last_iteration_time_ms is not None:
            metrics["last_iteration_time_ms"] = round(
                self.last_iteration_time_ms, 2
            )

        avg_iteration = self._get_iteration_time_average()
        if avg_iteration is not None:
            metrics["iteration_time_avg_ms"] = round(avg_iteration, 2)

        if self.target_iteration_time_ms is not None:
            metrics["target_iteration_time_ms"] = round(
                self.target_iteration_time_ms, 2
            )
        if self.tpot is not None:
            metrics["tpot_ms"] = round(self.tpot, 2)
        # Optional scheduler interval between last two process_batch_result completions (overlap only)
        scheduler_interval = getattr(batch, "scheduler_interval_ms", None) or getattr(batch, "since_last_process_ms", None)
        if scheduler_interval is not None:
            metrics["scheduler_interval_ms"] = round(scheduler_interval, 2)

        # Collect waiting queue information
        waiting_queue_requests = []
        total_extend_len = 0

        for i, req in enumerate(self.waiting_queue):
            extend_len = getattr(req, "extend_input_len", 0)

            if i < 10:  # Limit to first 10 requests
                prefix_len = len(getattr(req, "prefix_indices", []))
                waiting_queue_requests.append({
                    "id": req.rid,
                    "prefix_len": prefix_len,
                    "extend_len": extend_len,
                })

            # Calculate total for all requests (not just first 10)
            total_extend_len += extend_len

        metrics["waiting_queue_info"] = {
            "pending_req_num": len(self.waiting_queue),
            "total_extend_len": total_extend_len,
            "requests": waiting_queue_requests,
        }
        if queue_slack_stats:
            metrics["waiting_queue_info"]["slack_ms"] = queue_slack_stats

        metrics["stats_source"] = "engine"
        ack_generation, ack_last_id = self.router_ack_tracker.get_state()
        if ack_generation is not None:
            metrics["router_generation"] = ack_generation
            metrics["last_received_message_id"] = (
                ack_last_id if ack_last_id is not None else -1
            )
            logger.debug(
                "[ROUTER_ACK_EXPORT] worker=%s iter=%s mode=%s ack_gen=%s ack_last_id=%s destinations=%s",
                self.worker_id,
                self.iteration_count,
                self.slo_scheduler_mode,
                ack_generation,
                ack_last_id,
                destinations,
            )

        # Report (non-blocking)
        iteration_metrics.report_iteration(
            metrics, iteration_num=self.iteration_count, destinations=destinations
        )

        # Train the online predictor with this iteration's data
        # In sidecar mode, the sidecar owns predictor training
        if self.cycle_time_predictor is not None and iteration_time_ms is not None and self.slo_scheduler_mode != "sidecar":
            # Get mode for mode-aware predictor
            mode_str = _forward_mode_to_string(batch.forward_mode)

            # # Log predictor inputs for debugging sidecar mismatch
            # logger.warning(
            #     "[ENGINE_PREDICTOR_SUBMIT] iter=%d batch_size_tokens=%d n_prefill_pairs=%d "
            #     "prefill_pairs=%s kv_tokens_used=%d actual_time_ms=%.3f mode=%s forward_mode=%s",
            #     self.iteration_count, total_tokens, len(prefill_chunk_pairs),
            #     prefill_chunk_pairs[:5],  # first 5 pairs to avoid log spam
            #     num_used, iteration_time_ms, mode_str,
            #     batch.forward_mode.name if batch.forward_mode else "NONE",
            # )

            # Call submit with mode parameter (for multi-mode predictor) or without (for old predictor)
            if getattr(self.cycle_time_predictor, "is_multimode", False):
                pred, err = self.cycle_time_predictor.submit(
                    batch_size_tokens=total_tokens,
                    prefill_chunk_pairs=prefill_chunk_pairs,
                    kv_tokens_used=num_used,
                    iteration_time_ms=iteration_time_ms,
                    mode=mode_str,
                )
                # logger.warning(
                #     "[ENGINE_PREDICTOR_RESULT] iter=%d pred=%.3f err=%.3f n_seen=%d",
                #     self.iteration_count, pred, err,
                #     getattr(self.cycle_time_predictor, '_n_seen', -1),
                # )
            else:
                # Old predictor without mode parameter
                self.cycle_time_predictor.submit(
                    batch_size_tokens=total_tokens,
                    prefill_chunk_pairs=prefill_chunk_pairs,
                    kv_tokens_used=num_used,
                    iteration_time_ms=iteration_time_ms,
                )

    def _report_idle_metrics_if_needed(self):
        """Emit idle metrics updates to the router at a fixed cadence while idle."""
        if not self.server_args.enable_iteration_metrics:
            return
        # In sidecar mode, engine does not report to router — sidecar handles it
        if self.slo_scheduler_mode == "sidecar":
            return

        now = time.perf_counter()
        if self._last_idle_router_report_time is not None:
            elapsed = now - self._last_idle_router_report_time
            if elapsed < self._idle_router_report_interval:
                return

        idle_batch = self.get_idle_batch()
        self._collect_and_report_iteration_metrics(
            idle_batch, iteration_time_ms=None, destinations=["router"]
        )
        self._last_idle_router_report_time = now

    def _record_iteration_time(self, iteration_time_ms: float):
        """Store the latest iteration and refresh derived targets."""
        self.iteration_time_history.append(float(iteration_time_ms))
        self._refresh_iteration_time_target()

    def _get_iteration_time_average(self) -> Optional[float]:
        if not self.iteration_time_history:
            return None
        return sum(self.iteration_time_history) / len(self.iteration_time_history)

    def _refresh_iteration_time_target(self):
        avg = self._get_iteration_time_average()
        if avg is None or self.tpot is None:
            if self.target_iteration_time_ms is not None:
                logger.debug(
                    "[Scheduler] clearing iteration target; avg=%s tpot=%s",
                    avg,
                    self.tpot,
                )
            self.target_iteration_time_ms = None
            return

        new_target = compute_iteration_target(self.tpot, avg)
        if (
            self.target_iteration_time_ms is None
            or abs(self.target_iteration_time_ms - new_target) > 1e-3
        ):
            logger.info(
                "[Scheduler] iteration target updated: avg=%.3fms tpot=%.3fms target=%.3fms",
                avg,
                self.tpot,
                new_target,
            )
        self.target_iteration_time_ms = new_target

    def _compute_req_slack_ms(self, req: Req, now_ms: Optional[float] = None) -> Optional[float]:
        """Compute slack for a request based on router arrival and SLO hints."""
        arrival_ms = getattr(req, "arrival_time_ms", None)
        if arrival_ms is None:
            return None

        if now_ms is None:
            now_ms = time.time() * 1000.0

        expected_ms = 0.0
        if req.target_ttft_ms is not None:
            expected_ms += float(req.target_ttft_ms)
        if req.target_tpot_ms is not None:
            expected_ms += float(len(req.output_ids) * req.target_tpot_ms)

        slack_ms = expected_ms - (now_ms - arrival_ms)

        # Mark request as permanently violated once slack goes negative
        if slack_ms < 0 and not getattr(req, 'slo_violated', False):
            req.slo_violated = True

        # Cache on the request for reuse in scheduling decisions.
        req.last_slack_ms = slack_ms
        req.last_slack_computed_at_ms = now_ms

        return slack_ms

    def init_deterministic_inference_config(self):
        """Initialize deterministic inference configuration for different attention backends."""
        if not self.server_args.enable_deterministic_inference:
            self.truncation_align_size = None
            return

        backend_sizes = {
            "flashinfer": ("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096),
            "triton": ("SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE", 4096),
        }
        env_var, default_size = backend_sizes.get(
            self.server_args.attention_backend, (None, None)
        )
        self.truncation_align_size = (
            get_int_env_var(env_var, default_size) if env_var else None
        )

    def init_tokenizer(self):
        server_args = self.server_args
        self.is_generation = self.model_config.is_generation

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    use_fast=not server_args.disable_fast_image_processor,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )

    def init_memory_pool_and_cache(self):
        server_args = self.server_args

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self.tp_worker.get_memory_pool()
        )

        if (
            server_args.chunked_prefill_size is not None
            and server_args.disable_radix_cache
        ):
            if self.is_hybrid:
                ChunkCacheClass = SWAChunkCache
            else:
                ChunkCacheClass = ChunkCache
            self.tree_cache = ChunkCacheClass(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=self.page_size,
            )
        else:
            if os.environ.get("SGLANG_EXPERIMENTAL_CPP_RADIX_TREE") == "1":
                # lazy import to avoid JIT overhead
                from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp

                self.tree_cache = RadixCacheCpp(
                    disable=False,
                    use_hicache=self.enable_hierarchical_cache,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool=self.token_to_kv_pool_allocator,
                    tp_cache_group=self.tp_cpu_group,
                    page_size=self.page_size,
                    hicache_ratio=server_args.hicache_ratio,
                    hicache_size=server_args.hicache_size,
                    hicache_write_policy=server_args.hicache_write_policy,
                    enable_kv_cache_events=self.enable_kv_cache_events,
                )
            elif self.enable_hierarchical_cache:
                self.tree_cache = HiRadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    tp_cache_group=(
                        self.attn_tp_cpu_group
                        if self.server_args.enable_dp_attention
                        else self.tp_cpu_group
                    ),
                    page_size=self.page_size,
                    eviction_policy=server_args.radix_eviction_policy,
                    hicache_ratio=server_args.hicache_ratio,
                    hicache_size=server_args.hicache_size,
                    hicache_write_policy=server_args.hicache_write_policy,
                    hicache_io_backend=server_args.hicache_io_backend,
                    hicache_mem_layout=server_args.hicache_mem_layout,
                    enable_metrics=self.enable_metrics,
                    hicache_storage_backend=server_args.hicache_storage_backend,
                    hicache_storage_prefetch_policy=server_args.hicache_storage_prefetch_policy,
                    model_name=server_args.served_model_name,
                    storage_backend_extra_config=server_args.hicache_storage_backend_extra_config,
                )
                self.tp_worker.register_hicache_layer_transfer_counter(
                    self.tree_cache.cache_controller.layer_done_counter
                )
            elif self.is_hybrid:
                assert (
                    self.server_args.disaggregation_mode == "null"
                ), "Hybrid mode does not support disaggregation yet"
                self.tree_cache = SWARadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    sliding_window_size=self.sliding_window_size,
                    page_size=self.page_size,
                    disable=server_args.disable_radix_cache,
                )
            elif server_args.enable_lmcache:
                from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
                    LMCRadixCache,
                )

                self.tree_cache = LMCRadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    page_size=self.page_size,
                    disable=server_args.disable_radix_cache,
                    model_config=self.model_config,
                    tp_size=self.tp_size,
                    rank=self.tp_rank,
                    tp_group=self.tp_group,
                    eviction_policy=server_args.radix_eviction_policy,
                )
            else:
                self.tree_cache = RadixCache(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    page_size=self.page_size,
                    disable=server_args.disable_radix_cache,
                    enable_kv_cache_events=self.enable_kv_cache_events,
                    eviction_policy=server_args.radix_eviction_policy,
                )

        self.decode_mem_cache_buf_multiplier = (
            1
            if self.spec_algorithm.is_none()
            else (
                server_args.speculative_num_draft_tokens
                + (
                    (server_args.speculative_eagle_topk or 1)
                    * (server_args.speculative_num_steps or 1)
                )
            )
        )

        embedding_cache_size = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "100"))
        init_embedding_cache(embedding_cache_size * 1024 * 1024)

    def init_disaggregation(self):
        self.transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ):  # *2 for the headroom.
            buffer_size = (self.req_to_token_pool.size) * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=self.model_config.hf_text_config.hidden_size,
                dtype=self.model_config.dtype,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                tp_rank=self.tp_rank,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                tree_cache=self.tree_cache,
            )

            # The decode requests pending for pre-allocation
            self.disagg_decode_prealloc_queue = DecodePreallocQueue(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                draft_token_to_kv_pool=(
                    None
                    if self.draft_worker is None or self.spec_algorithm.is_lookahead()
                    else self.draft_worker.model_runner.token_to_kv_pool
                ),
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                transfer_queue=self.disagg_decode_transfer_queue,
                tree_cache=self.tree_cache,
                gloo_group=self.attn_tp_cpu_group,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                dp_size=self.server_args.dp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                max_total_num_tokens=self.max_total_num_tokens,
                prefill_pp_size=self.server_args.disaggregation_prefill_pp,
                num_reserved_decode_tokens=self.server_args.num_reserved_decode_tokens,
                transfer_backend=self.transfer_backend,
            )

        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # *2 for the headroom.
            buffer_size = self.max_running_requests * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=self.model_config.hf_text_config.hidden_size,
                dtype=self.model_config.dtype,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
                draft_token_to_kv_pool=(
                    None
                    if self.draft_worker is None or self.spec_algorithm.is_lookahead()
                    else self.draft_worker.model_runner.token_to_kv_pool
                ),
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                gpu_id=self.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                gloo_group=self.attn_tp_cpu_group,
                max_total_num_tokens=self.max_total_num_tokens,
                decode_tp_size=self.server_args.disaggregation_decode_tp,
                decode_dp_size=self.server_args.disaggregation_decode_dp,
                scheduler=self,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                transfer_backend=self.transfer_backend,
            )
            # The prefill requests that are in the middle of kv sending
            self.disagg_prefill_inflight_queue: List[Req] = []

    def init_moe_config(self):
        if hasattr(self.model_config.hf_config, "num_experts_per_tok"):
            initialize_moe_config(self.server_args)

    @DynamicGradMode()
    def event_loop_normal(self):
        
        """A normal scheduler loop."""
        while True:
            logger.info("Event loop normal")
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                for req in batch.reqs:
                    trace_event("schedule", req.rid)

            if batch:
                # Snapshot KV before run_batch (for FinishedIterationData)
                if self.slo_client is not None:
                    self._pre_batch_kv_used = self._get_token_info()[0]
                # Measure iteration time
                batch_start_time = time.perf_counter()
                result = self.run_batch(batch)
                batch_end_time = time.perf_counter()
                iteration_time_ms = (batch_end_time - batch_start_time) * 1000
                logger.info(f"before report metrics: {iteration_time_ms}")
                # Increment iteration counter for completed iteration
                self.iteration_count += 1
                batch.iteration_id = self.iteration_count
                # Report metrics
                self._collect_and_report_iteration_metrics(batch, iteration_time_ms)
                # Drain 🟢 CurrentSnapshot (what GPU just executed)
                # Normal loop: iteration_count already incremented above
                if self.slo_client is not None:
                    self._drain_current_snapshot(batch, self.iteration_count)

                self.process_batch_result(batch, result)
                # Drain 🔴 FinishedIterationData (iteration just completed)
                # Normal loop: _pre_batch_kv_used is correct (no overlap overwrite)
                if self.slo_client is not None:
                    self._drain_finished_iteration(batch, iteration_time_ms, self._pre_batch_kv_used)
            else:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()
                self._report_idle_metrics_if_needed()

            self.last_batch = batch

    @DynamicGradMode()
    def event_loop_overlap(self):
        logger.info("Event loop overlap")
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        self.result_queue = deque()

        while True:
            recv_time_ms = process_input_time_ms = 0.0
            run_batch_time_ms = 0.0
            metrics_running_time_ms = 0.0
            metrics_debug_time_ms = 0.0
            metrics_complete_time_ms = 0.0
            process_result_dummy_time_ms = 0.0
            process_result_main_time_ms = 0.0
            get_batch_time_ms = 0.0
            gpu_elapsed_ms = 0.0
            since_last_ms = 0.0

            recv_start = time.perf_counter()
            recv_reqs = self.recv_requests()
            recv_time_ms = (time.perf_counter() - recv_start) * 1000

            process_input_start = time.perf_counter()
            self.process_input_requests(recv_reqs)
            process_input_time_ms = (time.perf_counter() - process_input_start) * 1000

            get_batch_start = time.perf_counter()
            batch = self.get_next_batch_to_run()
            get_batch_time_ms = (time.perf_counter() - get_batch_start) * 1000
            self.cur_batch = batch

            if batch:
                for req in batch.reqs:
                    trace_event("schedule", req.rid)

            # Shift KV snapshot BEFORE the if-batch block so that even when
            # batch=None (pipeline flush), _prev holds the value from before
            # the PREVIOUS batch ran — which is what FinishedIterationData needs.
            if self.slo_client is not None:
                self._prev_pre_batch_kv_used = self._pre_batch_kv_used

            if batch:
                batch.launch_done = threading.Event()
                # Mark the start time for this batch
                batch.iteration_start_time = time.perf_counter()
                # Snapshot KV before run_batch (for FinishedIterationData of THIS batch)
                if self.slo_client is not None:
                    self._pre_batch_kv_used = self._get_token_info()[0]
                run_start = time.perf_counter()
                result = self.run_batch(batch)
                run_batch_time_ms = (time.perf_counter() - run_start) * 1000
                self.result_queue.append((batch.copy(), result))

                # Send running batch state to router (no timing information)
                metrics_running_start = time.perf_counter()
                self._collect_and_report_iteration_metrics(
                    batch,
                    iteration_time_ms=None,  # No timing for running batch
                    destinations=["router"]   # Router only
                )
                metrics_running_time_ms = (
                    time.perf_counter() - metrics_running_start
                ) * 1000

                # Drain 🟢 CurrentSnapshot: what GPU is actively executing NOW
                # Overlap loop: iteration_count not yet incremented (happens after
                # process_batch_result below).  Use +1 to maintain the temporal
                # invariant: scheduling.iteration_count = current.iteration_count + 1.
                if self.slo_client is not None:
                    _t_cs = time.perf_counter()
                    self._drain_current_snapshot(batch, self.iteration_count + 1)
                    self._last_drain_cs_ms = (time.perf_counter() - _t_cs) * 1000

                # Optional: Debug log running batch snapshot
                if getattr(self.server_args, "enable_debug_metrics", False):
                    metrics_debug_start = time.perf_counter()
                    self._collect_and_report_iteration_metrics(
                        batch,
                        iteration_time_ms=None,
                        destinations=["debug"]  # Debug log only
                    )
                    metrics_debug_time_ms = (
                        time.perf_counter() - metrics_debug_start
                    ) * 1000

                if self.last_batch is None:
                    # Create a dummy first batch to start the pipeline for overlap schedule.
                    # It is now used for triggering the sampling_info_done event.
                    tmp_batch = ScheduleBatch(
                        reqs=None,
                        forward_mode=ForwardMode.DUMMY_FIRST,
                        next_batch_sampling_info=self.tp_worker.cur_sampling_info,
                    )
                    tmp_batch.iteration_start_time = time.perf_counter()
                    process_result_dummy_start = time.perf_counter()
                    self.process_batch_result(tmp_batch, None, batch.launch_done)
                    process_result_dummy_time_ms = (
                        time.perf_counter() - process_result_dummy_start
                    ) * 1000

            if self.last_batch:
                # Process the results of the last batch
                tmp_batch, tmp_result = self.result_queue.popleft()
                tmp_batch.next_batch_sampling_info = (
                    self.tp_worker.cur_sampling_info if batch else None
                )
                # NOTE: we should use current launched batch's launch_done event Instead of the last batch's
                process_result_main_start = time.perf_counter()
                self.process_batch_result(
                    tmp_batch, tmp_result, batch.launch_done if batch else None
                )
                process_result_main_time_ms = (
                    time.perf_counter() - process_result_main_start
                ) * 1000

                # Report metrics for the batch that just completed using precise GPU time if available
                # Send to log/UI only (router already got running batch state)
                gpu_elapsed = getattr(tmp_batch, 'gpu_elapsed_ms', None)
                if gpu_elapsed is not None:
                    gpu_elapsed_ms = gpu_elapsed
                    # Increment iteration counter for completed iteration
                    self.iteration_count += 1
                    tmp_batch.iteration_id = self.iteration_count
                    # Attach scheduler interval since last process result if available
                    since_last = getattr(tmp_batch, 'since_last_process_ms', None)
                    if since_last is not None:
                        since_last_ms = since_last
                        tmp_batch.scheduler_interval_ms = since_last
                    metrics_complete_start = time.perf_counter()
                    self._collect_and_report_iteration_metrics(
                        tmp_batch,
                        gpu_elapsed_ms,
                        destinations=["log", "ui"]  # Skip router
                    )
                    metrics_complete_time_ms = (
                        time.perf_counter() - metrics_complete_start
                    ) * 1000
                    # Drain 🔴 FinishedIterationData (gpu_elapsed path)
                    # Overlap loop: use _prev_pre_batch_kv_used (from before THIS batch ran)
                    if self.slo_client is not None:
                        _t_fi = time.perf_counter()
                        self._drain_finished_iteration(tmp_batch, gpu_elapsed_ms, self._prev_pre_batch_kv_used)
                        self._last_drain_fi_ms = (time.perf_counter() - _t_fi) * 1000
                elif hasattr(tmp_batch, 'iteration_start_time'):
                    # Increment iteration counter for completed iteration
                    self.iteration_count += 1
                    tmp_batch.iteration_id = self.iteration_count
                    # Fallback to wall-clock time if GPU elapsed is unavailable
                    batch_end_time = time.perf_counter()
                    iteration_time_ms = (batch_end_time - tmp_batch.iteration_start_time) * 1000
                    since_last = getattr(tmp_batch, 'since_last_process_ms', None)
                    if since_last is not None:
                        since_last_ms = since_last
                        tmp_batch.scheduler_interval_ms = since_last
                    metrics_complete_start = time.perf_counter()
                    self._collect_and_report_iteration_metrics(
                        tmp_batch,
                        iteration_time_ms,
                        destinations=["log", "ui"]  # Skip router
                    )
                    metrics_complete_time_ms = (
                        time.perf_counter() - metrics_complete_start
                    ) * 1000
                    # Drain 🔴 FinishedIterationData (wall-clock fallback path)
                    # Overlap loop: use _prev_pre_batch_kv_used (from before THIS batch ran)
                    if self.slo_client is not None:
                        _t_fi = time.perf_counter()
                        self._drain_finished_iteration(tmp_batch, iteration_time_ms, self._prev_pre_batch_kv_used)
                        self._last_drain_fi_ms = (time.perf_counter() - _t_fi) * 1000
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()
                self._report_idle_metrics_if_needed()

            self.last_batch = batch
            if self.slo_scheduler_mode != "sidecar":
                self.last_cycle_time_prediction = self.predict_batch(batch)
            else:
                self.last_cycle_time_prediction = 0.0

            loop_total_ms = (
                recv_time_ms
                + process_input_time_ms
                + get_batch_time_ms
                + run_batch_time_ms
                + metrics_running_time_ms
                + metrics_debug_time_ms
                + metrics_complete_time_ms
                + process_result_dummy_time_ms
            )
            kv_forecast_time_ms = (
                self._last_kv_forecast_time_ms
                if self._last_kv_forecast_time_ms is not None
                else 0.0
            )
            prefill_sim_time_ms = (
                self._prefill_sim_runtime_ms
                if self._prefill_sim_runtime_ms is not None
                else 0.0
            )
            # Sub-breakdown of sidecar overhead
            _bb = getattr(self, '_last_batch_breakdown', (0.0, 0.0, 0.0, 0.0, 0.0))
            _dcs = getattr(self, '_last_drain_cs_ms', 0.0)
            _dfi = getattr(self, '_last_drain_fi_ms', 0.0)
            self._last_drain_cs_ms = 0.0
            self._last_drain_fi_ms = 0.0
            if self.last_batch is not None or batch is not None:
                logger.info(
                    "\033[94m[TIME]: since_last=%.3f gpu=%.3f loop=%.3f "
                    "recv=%.3f input=%.3f batch=%.3f run=%.3f mr=%.3f md=%.3f "
                    "mc=%.3f pd=%.3f pm=%.3f kvf=%.3f psim=%.3f "
                    "merge=%.3f drain=%.3f zmq=%.3f pfill=%.3f shlog=%.3f "
                    "dcs=%.3f dfi=%.3f\033[0m",
                    since_last_ms,
                    gpu_elapsed_ms,
                    loop_total_ms,
                    recv_time_ms,
                    process_input_time_ms,
                    get_batch_time_ms,
                    run_batch_time_ms,
                    metrics_running_time_ms,
                    metrics_debug_time_ms,
                    metrics_complete_time_ms,
                    process_result_dummy_time_ms,
                    process_result_main_time_ms,
                    kv_forecast_time_ms,
                    prefill_sim_time_ms,
                    _bb[0], _bb[1], _bb[2], _bb[3], _bb[4],
                    _dcs, _dfi,
                )



    @DynamicGradMode()
    def event_loop_pp(self):

        """A non-overlap scheduler loop for pipeline parallelism."""
        mbs = [None] * self.pp_size
        last_mbs = [None] * self.pp_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False) for _ in range(self.pp_size)
        ]
        bids = [None] * self.pp_size
        pp_outputs: Optional[PPProxyTensors] = None
        while True:
            server_is_idle = True
            for mb_id in range(self.pp_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = last_mbs[mb_id]

                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)
                mbs[mb_id] = self.get_next_batch_to_run()
                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch = mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    # Measure iteration time for this micro-batch
                    batch_start_time = time.perf_counter()
                    result = self.run_batch(self.cur_batch)
                    batch_end_time = time.perf_counter()
                    iteration_time_ms = (batch_end_time - batch_start_time) * 1000
                    logger.info(f"before report metrics3: {iteration_time_ms}")
                    # Increment iteration counter for completed micro-batch
                    self.iteration_count += 1
                    self.cur_batch.iteration_id = self.iteration_count
                    # Report metrics for this micro-batch
                    self._collect_and_report_iteration_metrics(self.cur_batch, iteration_time_ms)

                # (last rank) send the outputs to the next step
                if self.pp_group.is_last_rank:
                    if self.cur_batch:
                        next_token_ids, bids[mb_id] = (
                            result.next_token_ids,
                            result.bid,
                        )
                        if self.cur_batch.return_logprob:
                            pp_outputs = PPProxyTensors(
                                {
                                    "next_token_ids": next_token_ids,
                                    "extend_input_len_per_req": result.extend_input_len_per_req,
                                    "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
                                }
                                | (
                                    {
                                        f"logits_output.{k}": v
                                        for k, v in result.logits_output.__dict__.items()
                                    }
                                    if result.logits_output is not None
                                    else {}
                                )
                            )
                        else:
                            pp_outputs = PPProxyTensors(
                                {
                                    "next_token_ids": next_token_ids,
                                }
                            )
                        # send the output from the last round to let the next stage worker run post processing
                        self.pp_group.send_tensor_dict(
                            pp_outputs.tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                # receive outputs and post-process (filter finished reqs) the coming microbatch
                next_mb_id = (mb_id + 1) % self.pp_size
                next_pp_outputs = None
                if mbs[next_mb_id] is not None:
                    next_pp_outputs: Optional[PPProxyTensors] = PPProxyTensors(
                        self.pp_group.recv_tensor_dict(
                            all_gather_group=self.attn_tp_group
                        )
                    )
                    mbs[next_mb_id].output_ids = next_pp_outputs["next_token_ids"]
                    logits_output_args = {
                        k[len("logits_output.") :]: v
                        for k, v in next_pp_outputs.tensors.items()
                        if k.startswith("logits_output.")
                    }
                    if len(logits_output_args) > 0:
                        logits_output = LogitsProcessorOutput(**logits_output_args)
                    else:
                        logits_output = None
                    output_result = GenerationBatchResult(
                        logits_output=logits_output,
                        pp_hidden_states_proxy_tensors=None,
                        next_token_ids=next_pp_outputs["next_token_ids"],
                        extend_input_len_per_req=next_pp_outputs.tensors.get(
                            "extend_input_len_per_req", None
                        ),
                        extend_logprob_start_len_per_req=next_pp_outputs.tensors.get(
                            "extend_logprob_start_len_per_req", None
                        ),
                        bid=bids[next_mb_id],
                        can_run_cuda_graph=result.can_run_cuda_graph,
                    )
                    self.process_batch_result(mbs[next_mb_id], output_result)
                    last_mbs[next_mb_id] = mbs[next_mb_id]

                # (not last rank)
                if not self.pp_group.is_last_rank:
                    if self.cur_batch:
                        bids[mb_id] = result.bid
                    # carry the outputs to the next stage
                    # send the outputs from the last round to let the next stage worker run post processing
                    if pp_outputs:
                        self.pp_group.send_tensor_dict(
                            pp_outputs.tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                    # send out reqs to the next stage
                    dp_offset = self.attn_dp_rank * self.attn_tp_size
                    if self.attn_tp_rank == 0:
                        point_to_point_pyobj(
                            recv_reqs,
                            self.pp_rank * self.tp_size + dp_offset,
                            self.world_group.device_group,
                            self.pp_rank * self.tp_size + dp_offset,
                            (self.pp_rank + 1) * self.tp_size + dp_offset,
                        )

                    # send out proxy tensors to the next stage
                    if self.cur_batch:
                        self.pp_group.send_tensor_dict(
                            result.pp_hidden_states_proxy_tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                pp_outputs = next_pp_outputs

            # When the server is idle, self-check and re-init some states
            if server_is_idle:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()
                self._report_idle_metrics_if_needed()

    def recv_requests(self) -> List[Req]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""

        if self.recv_skipper is not None:
            last_forward_mode = (
                self.last_batch.forward_mode if self.last_batch is not None else None
            )
            if not self.recv_skipper.handle(last_forward_mode):
                return []

        if self.pp_rank == 0:
            if self.attn_tp_rank == 0:
                recv_reqs = []

                while True:
                    try:
                        recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)

                while True:
                    try:
                        recv_rpc = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_rpc)
            else:
                recv_reqs = None
        else:
            if self.attn_tp_rank == 0:
                dp_offset = self.attn_dp_rank * self.attn_tp_size
                recv_reqs = point_to_point_pyobj(
                    [],
                    self.pp_rank * self.tp_size + dp_offset,
                    self.world_group.device_group,
                    (self.pp_rank - 1) * self.tp_size + dp_offset,
                    self.pp_rank * self.tp_size + dp_offset,
                )
            else:
                recv_reqs = None

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        if self.server_args.enable_dp_attention:
            if self.attn_tp_rank == 0:
                work_reqs = [
                    req
                    for req in recv_reqs
                    if isinstance(
                        req,
                        (
                            TokenizedGenerateReqInput,
                            TokenizedEmbeddingReqInput,
                            BatchTokenizedGenerateReqInput,
                            BatchTokenizedEmbeddingReqInput,
                        ),
                    )
                ]
                control_reqs = [
                    req
                    for req in recv_reqs
                    if not isinstance(
                        req,
                        (
                            TokenizedGenerateReqInput,
                            TokenizedEmbeddingReqInput,
                            BatchTokenizedGenerateReqInput,
                            BatchTokenizedEmbeddingReqInput,
                        ),
                    )
                ]
            else:
                work_reqs = None
                control_reqs = None

            if self.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
            if self.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif self.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )

        for req in recv_reqs:
            if isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
                trace_set_proc_propagate_context(req.rid, req.trace_context)
                trace_slice_start("", req.rid, anonymous=True)

        return recv_reqs

    def process_input_requests(self, recv_reqs: List):
        for recv_req in recv_reqs:
            # If it is a health check generation request and there are running requests, ignore it.
            if is_health_check_generate_req(recv_req) and (
                self.chunked_req is not None
                or not self.running_batch.is_empty()
                or len(self.offload_tags) > 0
            ):
                self.return_health_check_ct += 1
                continue

            # If it is a MultiTokenizerWrapper, unwrap it and handle the inner request.
            if isinstance(recv_req, MultiTokenizerWrapper):
                worker_id = recv_req.worker_id
                recv_req = recv_req.obj
                output = self._request_dispatcher(recv_req)
                if output is not None:
                    output = MultiTokenizerWrapper(worker_id, output)
                    self.send_to_tokenizer.send_pyobj(output)
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                if isinstance(output, RpcReqOutput):
                    if self.recv_from_rpc is not None:
                        self.recv_from_rpc.send_pyobj(output)
                else:
                    self.send_to_tokenizer.send_pyobj(output)

    def init_req_max_new_tokens(self, req):
        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # Check for duplicate messages early (before creating Req object)
        if not self.router_ack_tracker.record(
            recv_req.router_generation, recv_req.router_message_id
        ):
            logger.info(
                f"Dropping duplicate request: rid={recv_req.rid}, "
                f"generation={recv_req.router_generation}, message_id={recv_req.router_message_id}"
            )
            return

        self.maybe_update_dp_balance_data(recv_req)

        recv_arrival = getattr(recv_req, "arrival_time_ms", None)
        current_time = time.time() * 1000.0
        if recv_arrival is None:
            arrival_time_ms = current_time
            logger.info(f"[ARRIVAL_DEBUG] rid={recv_req.rid}, recv_arrival=None, using current_time={current_time:.3f}")
        else:
            arrival_time_ms = recv_arrival
            logger.info(f"[ARRIVAL_DEBUG] rid={recv_req.rid}, recv_arrival={recv_arrival:.3f}, current_time={current_time:.3f}, gap={(current_time - recv_arrival):.3f}ms")

        # Create a new request
        if (
            recv_req.session_params is None
            or recv_req.session_params.id is None
            or recv_req.session_params.id not in self.sessions
        ):
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                fake_input_ids = [1] * seq_length
                recv_req.input_ids = fake_input_ids

            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port

            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                stream=recv_req.stream,
                lora_id=recv_req.lora_id,
                input_embeds=recv_req.input_embeds,
                custom_logit_processor=recv_req.custom_logit_processor,
                return_hidden_states=recv_req.return_hidden_states,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                data_parallel_rank=recv_req.data_parallel_rank,
                vocab_size=self.model_config.vocab_size,
                priority=recv_req.priority,
                metrics_collector=(
                    self.metrics_collector if self.enable_metrics else None
                ),
                target_ttft_ms=recv_req.target_ttft_ms,
                target_tpot_ms=recv_req.target_tpot_ms,
                router_generation=recv_req.router_generation,
                router_message_id=recv_req.router_message_id,
                arrival_time_ms=arrival_time_ms,
                start_iteration=self.iteration_count,
            )
            req.tokenizer = self.tokenizer
            logger.info(f"Received new request: rid={req.rid}, time={arrival_time_ms}, recv_time={time.time() * 1000.0}")

            if self.disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                if recv_req.bootstrap_room is None:
                    error_msg = (
                        f"Invalid request: Disaggregated request received without "
                        f"boostrap room id. {req.rid=}"
                    )
                    logger.error(error_msg)
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.stream_output([req], req.return_logprob)
                    return

            if (
                recv_req.session_params is not None
                and recv_req.session_params.id is not None
            ):
                req.set_finish_with_abort(
                    f"Invalid request: session id {recv_req.session_params.id} does not exist"
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return
        else:
            # Create a new request from a previous session
            session = self.sessions[recv_req.session_params.id]
            req = session.create_req(recv_req, self.tokenizer)
            # Align session-generated requests with current scheduler iteration counter
            req.start_iteration = self.iteration_count
            if isinstance(req.finished_reason, FINISH_ABORT):
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # Note: router_ack_tracker.record() is called at the start of handle_generate_request()

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            image_inputs = MultimodalInputs.from_dict(recv_req.mm_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            req.origin_input_ids = self.pad_input_ids_func(
                req.origin_input_ids, image_inputs
            )
            req.extend_image_inputs(image_inputs)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # initialize before returning
        self.init_req_max_new_tokens(req)

        # Validate prompt length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        if recv_req.logprob_start_len == -1 or not recv_req.return_logprob:
            # By default, only return the logprobs for output tokens
            # For prefill-only requests with logprob_start_len == -1, set logprob_start_len beyond input sequence
            # to skip input logprob computation entirely
            if req.is_prefill_only:
                req.logprob_start_len = len(req.origin_input_ids)
            else:
                # TODO: For text generation, evaluate setting logprob_start_len to len(req.origin_input_ids) as well
                req.logprob_start_len = len(req.origin_input_ids) - 1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if not req.is_prefill_only and req.logprob_start_len >= len(
            req.origin_input_ids
        ):
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = len(req.origin_input_ids) - 1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            assert self.grammar_backend is not None
            if req.sampling_params.json_schema is not None:
                key = ("json", req.sampling_params.json_schema)
            elif req.sampling_params.regex is not None:
                key = ("regex", req.sampling_params.regex)
            elif req.sampling_params.ebnf is not None:
                key = ("ebnf", req.sampling_params.ebnf)
            elif req.sampling_params.structural_tag:
                key = ("structural_tag", req.sampling_params.structural_tag)

            value, cache_hit = self.grammar_backend.get_cached_or_future_value(key)
            req.grammar = value

            if not cache_hit:
                req.grammar_key = key
                add_to_grammar_queue = True
            else:
                if value is INVALID_GRAMMAR_OBJ:  # We hit a cached invalid grammar.
                    error_msg = f"Invalid grammar request with cache hit: {key=}"
                    req.set_finish_with_abort(error_msg)

        if add_to_grammar_queue:
            req.queue_time_start = time.perf_counter()
            self.grammar_queue.append(req)
        else:
            self._add_request_to_queue(req)

    def handle_batch_generate_request(
        self,
        recv_req: BatchTokenizedGenerateReqInput,
    ):
        """Handle optimized batch generate request."""
        logger.debug(f"Processing batch generate request with {len(recv_req)} requests")

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_generate_request(tokenized_req)

    def _add_request_to_queue(self, req: Req):
        req.queue_time_start = time.perf_counter()

        # Increment accepted requests counter for metrics
        if self.server_args.enable_iteration_metrics:
            from sglang.srt.ui import iteration_metrics
            iteration_metrics.inc_accepted_requests(1)

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(
                req, self.model_config.num_key_value_heads
            )
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.add(req)
        else:
            self._set_or_validate_priority(req)
            if self._abort_on_queued_limit(req):
                return
            self._prefetch_kvcache(req)
            self.waiting_queue.append(req)
            trace_slice_end("process req", req.rid, auto_next_anon=True)

        # Track accepted requests for sidecar (after successful enqueue)
        if self.slo_client is not None:
            self._accepted_since_last_send += 1

    def _prefetch_kvcache(self, req: Req):
        if self.enable_hicache_storage:
            req.init_next_round_input(self.tree_cache)
            if req.last_node.backuped:
                # only to initiate the prefetch if the last node is backuped
                # otherwise, the allocated GPU memory must be locked for integrity
                last_hash = req.last_host_node.get_last_hash_value()
                matched_len = len(req.prefix_indices) + req.host_hit_length
                new_input_tokens = req.fill_ids[matched_len:]
                self.tree_cache.prefetch_from_storage(
                    req.rid, req.last_host_node, new_input_tokens, last_hash
                )

    def _extend_requests_to_queue(self, reqs: List[Req], is_retracted: bool = False):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.disagg_prefill_bootstrap_queue.extend(
                reqs, self.model_config.num_key_value_heads
            )
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # If this is a decode server, we put the request to the decode pending prealloc queue
            self.disagg_decode_prealloc_queue.extend(reqs, is_retracted)
        else:
            for req in reqs:
                self._set_or_validate_priority(req)
                if not self._abort_on_queued_limit(req):
                    self.waiting_queue.append(req)

    def _set_or_validate_priority(self, req: Req):
        """Set the default priority value, or abort the request based on the priority scheduling mode."""
        if self.enable_priority_scheduling and req.priority is None:
            if self.schedule_low_priority_values_first:
                req.priority = sys.maxsize
            else:
                req.priority = -sys.maxsize - 1
        elif not self.enable_priority_scheduling and req.priority is not None:
            abort_req = AbortReq(
                req.rid,
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": "Using priority is disabled for this server. Please send a new request without a priority.",
                },
            )
            self.send_to_tokenizer.send_pyobj(abort_req)

    def _abort_on_queued_limit(self, recv_req: Req) -> bool:
        """Abort an incoming or existing request if the waiting queue is full. Returns True if the incoming request is aborted."""
        if (
            self.max_queued_requests is None
            or len(self.waiting_queue) + 1 <= self.max_queued_requests
        ):
            return False

        # Reject the incoming request by default.
        req_to_abort = recv_req
        message = "The request queue is full."
        if self.enable_priority_scheduling:
            # With priority scheduling, consider aboritng an existing request based on the priority.
            # direction = 1  => smaller number = higher priority; -1 => larger number = higher priority.
            # max(...) + (direction * priority, queue_time_start) picks the least-preferred request.
            # Tie: later queue_time_start (newer) is evicted first. Preempt only if strictly better.
            direction = 1 if self.schedule_low_priority_values_first else -1
            key_fn = lambda item: (
                direction * item[1].priority,
                item[1].queue_time_start,
            )
            idx, candidate_req = max(enumerate(self.waiting_queue), key=key_fn)
            abort_existing_req = (
                direction * recv_req.priority < direction * candidate_req.priority
            )
            if abort_existing_req:
                self.waiting_queue.pop(idx)
                req_to_abort = candidate_req
                message = "The request is aborted by a higher priority request."

        self.send_to_tokenizer.send_pyobj(
            AbortReq(
                req_to_abort.rid,
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": message,
                },
            )
        )
        return req_to_abort.rid == recv_req.rid

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            token_type_ids=recv_req.token_type_ids,
            priority=recv_req.priority,
            start_iteration=self.iteration_count,
        )
        req.tokenizer = self.tokenizer

        # Handle multimodal inputs
        if recv_req.image_inputs is not None:
            image_inputs = MultimodalInputs.from_dict(recv_req.image_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            req.origin_input_ids = self.pad_input_ids_func(
                req.origin_input_ids, image_inputs
            )
            req.extend_image_inputs(image_inputs)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self._add_request_to_queue(req)
                return

        # Validate prompts length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        req.logprob_start_len = len(req.origin_input_ids) - 1
        self._add_request_to_queue(req)

    def handle_batch_embedding_request(
        self,
        recv_req: BatchTokenizedEmbeddingReqInput,
    ):
        """Handle optimized batch embedding request."""
        logger.debug(
            f"Processing batch embedding request with {len(recv_req)} requests"
        )

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_embedding_request(tokenized_req)

    def self_check_during_idle(self):
        return
        self.check_memory()
        self.check_tree_cache()
        self.new_token_ratio = self.init_new_token_ratio
        self.maybe_sleep_on_idle()

    def check_memory(self):
        if self.is_hybrid:
            (
                full_num_used,
                swa_num_used,
                _,
                _,
                full_available_size,
                full_evictable_size,
                swa_available_size,
                swa_evictable_size,
            ) = self._get_swa_token_info()
            memory_leak = full_num_used != 0 or swa_num_used != 0
            token_msg = (
                f"{self.full_tokens_per_layer=}, {full_available_size=}, {full_evictable_size=}, {self.tree_cache.full_protected_size()=}\n"
                f"{self.swa_tokens_per_layer=}, {swa_available_size=}, {swa_evictable_size=}, {self.tree_cache.swa_protected_size()=}\n"
            )
        else:
            _, _, available_size, evictable_size = self._get_token_info()
            protected_size = self.tree_cache.protected_size()
            memory_leak = (available_size + evictable_size) != (
                # self.max_total_num_tokens
                # if not self.enable_hierarchical_cache
                # else self.max_total_num_tokens - protected_size
                self.max_total_num_tokens
                - protected_size
            )
            token_msg = f"{self.max_total_num_tokens=}, {available_size=}, {evictable_size=}, {protected_size=}\n"

        if memory_leak > 0.02 * self.max_total_num_tokens:
            msg = "token_to_kv_pool_allocator memory leak detected! " f"{token_msg}"
            raise ValueError(msg)

        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        if len(self.req_to_token_pool.free_slots) != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise ValueError(msg)

        if (
            self.enable_metrics
            and self.current_scheduler_metrics_enabled()
            and time.perf_counter() > self.metrics_collector.last_log_time + 30
        ):
            # During idle time, also collect metrics every 30 seconds.
            if self.is_hybrid:
                (
                    full_num_used,
                    swa_num_used,
                    full_token_usage,
                    swa_token_usage,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_swa_token_info()
                num_used = max(full_num_used, swa_num_used)
                token_usage = max(full_token_usage, swa_token_usage)
            else:
                num_used, token_usage, _, _ = self._get_token_info()
            num_running_reqs = len(self.running_batch.reqs)
            self.stats.num_running_reqs = num_running_reqs
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = round(token_usage, 2)
            self.stats.gen_throughput = 0
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.num_grammar_queue_reqs = len(self.grammar_queue)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )
            self.metrics_collector.log_stats(self.stats)
        self._publish_kv_events()

    def check_tree_cache(self):
        if self.is_hybrid and isinstance(self.tree_cache, SWARadixCache):
            self.tree_cache.sanity_check()

    def _get_token_info(self):
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return num_used, token_usage, available_size, evictable_size

    def _get_swa_token_info(self):
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        full_evictable_size = self.tree_cache.full_evictable_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        swa_evictable_size = self.tree_cache.swa_evictable_size()
        full_num_used = self.full_tokens_per_layer - (
            full_available_size + full_evictable_size
        )
        swa_num_used = self.swa_tokens_per_layer - (
            swa_available_size + swa_evictable_size
        )
        full_token_usage = full_num_used / self.full_tokens_per_layer
        swa_token_usage = swa_num_used / self.swa_tokens_per_layer
        return (
            full_num_used,
            swa_num_used,
            full_token_usage,
            swa_token_usage,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        )

    def _submit_decode_length_observations(self, finished_reqs: List[Req]) -> None:
        """Push finished decode lengths into the estimator for future sampling."""
        if self.output_estimator is None or not finished_reqs:
            return

        lengths = [
            len(req.output_ids)
            for req in finished_reqs
            if not getattr(req, "is_retracted", False)
            and not isinstance(req.finished_reason, FINISH_ABORT)
        ]
        # Skip empty observations to avoid log bucket edge cases.
        lengths = [l for l in lengths if l > 0]
        if lengths:
            self.output_estimator.submit_decode_length_observation(lengths)

    def _predict_future_kv_usage(self) -> None:
        """Estimate future KV peak/slack for active decode batch (currently unused)."""
        if (
            self.output_estimator is None
            or self.running_batch is None
            or self.running_batch.is_empty()
            or self.running_batch.forward_mode is None
            or not self.running_batch.forward_mode.is_decode()
        ):
            return

        active_reqs = [req for req in self.running_batch.reqs if not req.finished()]
        if not active_reqs:
            return

        t0 = time.perf_counter()
        # Prefill/context length should exclude locally evicted KV.
        prefill_tokens = [
            max(len(req.origin_input_ids) - getattr(req, "evicted_seqlen_local", 0), 0)
            for req in active_reqs
        ]
        current_decode_tokens = [len(req.output_ids) for req in active_reqs]
        ground_truth_remaining = [
            max(getattr(req.sampling_params, "max_new_tokens", 0) - len(req.output_ids), 0)
            for req in active_reqs
        ]

        try:
            self._last_kv_forecast = self.output_estimator.estimate_peak_and_slack(
                prefill_tokens,
                current_decode_tokens,
                tpot=self.tpot,
            )
            # Ground truth using max_new_tokens upper bounds; can be used to assess predictor bias.
            self._last_kv_forecast_gt = (
                self.output_estimator.estimate_peak_and_slack_with_ground_truth(
                    prefill_tokens,
                    current_decode_tokens,
                    ground_truth_remaining,
                    tpot=self.tpot,
                )
            )
            self._last_kv_forecast_time_ms = (time.perf_counter() - t0) * 1000.0
        except Exception:
            logger.exception("Failed to predict future KV usage/slack")

    def _get_simulation_chunk_budget(self) -> int:
        """Extract first chunk from simulation execution_flow for this iteration.
        
        Returns:
            Chunk budget in tokens:
            - 0: Decode-only iteration (simulation says skip prefill)
            - >0: Prefill budget for this iteration
            - chunked_prefill_size: Fallback when no simulation results
        """
        if self._last_prefill_sim_results is None:
            logger.info("No execution flow found, falling back to decode")
            return self.chunked_prefill_size  # Fallback if no simulation results   
        
        # Use the base case result (extra=0)
        base_result = self._last_prefill_sim_results[0]
        
        # Trust the simulation's execution_flow - it provides a reasonable plan
        # even when decode_feasible=False (marked as "LATE" but still executable)
        execution_flow = getattr(base_result, 'execution_flow', None) or []
        if not execution_flow:
            logger.info("No execution flow found, falling back to decode")
            return self.chunked_prefill_size  # Fallback if no execution flow
        
        # First chunk determines this iteration's budget
        # Can be 0 (decode-only) or >0 (prefill budget)
        first_chunk = execution_flow[0]
        logger.info(f"[SIM SCHEDULE] execution_flow={execution_flow}, using first_chunk={first_chunk}")
        return first_chunk

    def _compute_min_valid_decode_slack(self) -> Optional[float]:
        """Compute minimum valid decode slack from active decode requests.

        Filters out:
        - Requests with slo_violated flag set
        - Requests with slack < 0
        - Requests with fixed_slack < 0 (after overlap adjustment)

        Returns None if no valid decode requests found.
        """
        tpot_effective = self.tpot if self.tpot not in (None, 0) else 1000.0
        pred_last = self.last_cycle_time_prediction

        min_decode_slack = None

        if self.last_batch is not None and getattr(self.last_batch, "reqs", None):
            decoding_reqs = (
                set(self.last_batch.decoding_reqs)
                if getattr(self.last_batch, "decoding_reqs", None)
                else None
            )
            for req in self.last_batch.reqs:
                is_decode = (
                    self.last_batch.forward_mode == ForwardMode.DECODE
                    or (
                        self.last_batch.forward_mode == ForwardMode.MIXED
                        and decoding_reqs
                        and req in decoding_reqs
                    )
                )
                if not is_decode:
                    continue

                # Filter: skip already violated requests
                if getattr(req, 'slo_violated', False):
                    continue

                slack_val = getattr(req, "last_slack_ms", None)
                if slack_val is None:
                    slack_val = self._compute_req_slack_ms(req, now_ms=time.time() * 1000.0)

                # Filter: skip requests with no slack or negative slack
                if slack_val is None or slack_val < 0:
                    continue

                # Compute fixed slack: slack - pred_last + tpot
                effective_req_tpot = req.target_tpot_ms if req.target_tpot_ms is not None else tpot_effective
                fixed_slack = slack_val - pred_last + effective_req_tpot

                # Filter: skip if fixed slack is negative
                if fixed_slack < 0:
                    continue

                if min_decode_slack is None or fixed_slack < min_decode_slack:
                    min_decode_slack = fixed_slack

        return min_decode_slack

    def _maybe_run_prefill_simulation(self) -> None:
        """Run prefill simulator for chunked + queued requests to guide batch formation.

        Overlap Model (Conservative Assumptions):
        ---------------------------------------
        This function simulates prefill scheduling while accounting for an overlapping
        batch execution. The model makes these conservative assumptions:

        1. Mixed-chunk mode is enabled (--enable-mixed-chunk):
           - Either DECODE or MIXED forward mode for batches
           - Decode requests participate and are serviced in every iteration
           - Pure EXTEND batches should not exist with mixed-chunk enabled

        2. Conservative timing:
           - Assume last_batch just started execution (use full predicted time)
           - This provides safety margin for scheduling decisions

        Overlap Slack Adjustments:
        --------------------------
        - Decode slack: min_decode_slack - pred_last + TPOT
          * Decode requests lose pred_last ms waiting for last_batch to complete
          * But recover TPOT budget from last_batch's iteration
          * Net effect: -(pred_last - TPOT)

        - Prefill slack: max(raw_slack - pred_last, 0)
          * Waiting requests simply lose pred_last ms (no TPOT recovery)
          * They're not being served during overlap period

        Why last_batch (not running_batch):
        -----------------------------------
        last_batch = the batch currently executing (or just finished) on GPU
                     (can be DECODE, EXTEND, or MIXED mode)

        running_batch = merged batch for future iterations (after prefill merge)
        We must predict last_batch (current GPU batch) to get correct overlap time.
        Predicting merged running_batch would include future requests, overcounting overlap.

        Results Stored:
        --------------
        - self._last_prefill_sim_results: List[BatchPlanResult] (base case results)
        - self._predicted_ttft_for_new_admits: Dict[int, float] (extra_len -> predicted TTFT in ms for router)
        """
        t_sim_start = time.perf_counter()
        self._prefill_sim_runtime_ms = 0.0
        try:
            grid_path = getattr(self.server_args, "predictor_grid_path", None)
            if grid_path is None:
                self._prefill_sim_runtime_ms = (time.perf_counter() - t_sim_start) * 1000.0
                return

            # Lazily create engine and update dynamic config each iteration.
            if self.prefill_sim_engine is None:
                self.prefill_sim_engine = PrefillSimulatorEngine(predictor=self.cycle_time_predictor, safety_margin_ms=100.0)

            num_used, _, _, _ = self._get_token_info()

            # Decode batch uses the most active batch available.
            if self.last_batch is not None and self.last_batch.reqs is not None:
                decode_batch = max(len(self.last_batch.reqs), 1)
            elif self.running_batch is not None and self.running_batch.reqs is not None:
                decode_batch = max(len(self.running_batch.reqs), 1)
            else:
                decode_batch = 1

            # Overlap adjustment: predict last_batch execution time.
            # IMPORTANT: Use last_batch (the batch currently on GPU), NOT running_batch.
            # last_batch = current GPU batch (can be DECODE, EXTEND, or MIXED mode)
            # running_batch = merged batch for future iterations (after prefill merge)
            # We need the current GPU batch time for correct overlap calculation.
            tpot_effective = self.tpot if self.tpot not in (None, 0) else 1000.0
    
            pred_last = self.last_cycle_time_prediction


            min_decode_slack = None
            min_slack_req_tpot = tpot_effective  # Track TPOT of request with minimum slack
            if self.last_batch is not None and getattr(self.last_batch, "reqs", None):
                decoding_reqs = (
                    set(self.last_batch.decoding_reqs)
                    if getattr(self.last_batch, "decoding_reqs", None)
                    else None
                )
                for req in self.last_batch.reqs:
                    is_decode = (
                        self.last_batch.forward_mode == ForwardMode.DECODE
                        or (
                            self.last_batch.forward_mode == ForwardMode.MIXED
                            and decoding_reqs
                            and req in decoding_reqs
                        )
                    )
                    if not is_decode:
                        continue
                    slack_val = getattr(req, "last_slack_ms", None)
                    if slack_val is None:
                        slack_val = self._compute_req_slack_ms(req, now_ms=time.time() * 1000.0)
                    if slack_val is None or slack_val < 0 or getattr(req, 'slo_violated', False): # if a request already violates slack, skip it
                        continue
                    fixed_slack = slack_val - pred_last + tpot_effective
                    if fixed_slack < 0: # if a request already violates slack, skip it
                        continue
                    # Update min slack and track the TPOT of that request
                    if min_decode_slack is None or fixed_slack < min_decode_slack:
                        min_decode_slack = fixed_slack
                        # Use per-request TPOT if available, otherwise fall back to global
                        req_tpot = req.target_tpot_ms
                        min_slack_req_tpot = float(req_tpot) if req_tpot not in (None, 0) else tpot_effective

            min_decode_slack = 9999999 if min_decode_slack is None else float(min_decode_slack) # if not request, then slack is inf

            # Decode slack formula: min_slack - overlap_time + TPOT_budget
            # - Subtract pred_last: time consumed waiting for last_batch
            # - Add min_slack_req_tpot: TPOT budget recovered from last_batch iteration (using the TPOT of the request with minimum slack)
            # Assumes mixed-chunk mode: decode requests are serviced during last_batch

            decode_slack_ms = min_decode_slack 

            # Edge cases handled by this formula:
            # 1. pred_last = 0 (no overlap): decode_slack_ms = min_decode_slack + tpot
            #    Correct: TPOT budget for the first iteration is included
            # 2. pred_last >> TPOT (slow batch): decode_slack_ms may be negative
            #    Correct: signals overspending, simulator will inject wait cycles
            # 3. pred_last << TPOT (fast batch): decode_slack_ms increases
            #    Correct: underspending creates slack budget for next iteration

            # Only log update_decode if decode parameters changed
            if (
                self._last_prefill_sim_decode_batch != decode_batch
                or self._last_prefill_sim_kv_cache != num_used
            ):
                logger.info(
                    "[PREFILL-SIM] update_decode: decode_batch=%d kv_cache=%d tpot_ms=%.2f slack_decode_ms=%.2f (pred_last=%.2f min_decode_slack=%.2f min_slack_req_tpot=%.2f)",
                    decode_batch,
                    num_used,
                    tpot_effective,
                    decode_slack_ms,
                    pred_last,
                    min_decode_slack,
                    min_slack_req_tpot,
                )
            self.prefill_sim_engine.update_decode(
                decode_batch=decode_batch,
                kv_cache=num_used,
                tpot_ms=tpot_effective,
                slack_decode_ms=decode_slack_ms,
            )

            now_ms = time.time() * 1000.0

            # Build candidates in FIFO order: chunked req first (if any), then current queue as-is.
            candidates: List[Req] = []
            if self.chunked_req is not None:
                candidates.append(self.chunked_req)
            if self.waiting_queue:
                candidates.extend(self.waiting_queue)

            total_prefill_lens: List[int] = []
            already_prefilled_lens: List[int] = []
            prefill_slacks: List[float] = []

            skipped_zero_len = 0
            for req in candidates:
                # For chunked request, extend_input_len may be stale from previous chunk.
                # Always reinitialize to get accurate remaining tokens.
                is_chunked_req = (req is self.chunked_req)
                extend_len = max(int(getattr(req, "extend_input_len", 0)), 0)
                if (extend_len == 0 or is_chunked_req) and not req.finished():
                    try:
                        req.init_next_round_input(self.tree_cache)
                        extend_len = max(int(getattr(req, "extend_input_len", 0)), 0)
                    except Exception as e:
                        logger.warning("[PREFILL-SIM] init_next_round_input failed for req %s: %s", req.rid, e)
                        extend_len = 0

                prefetched = max(len(getattr(req, "prefix_indices", [])), 0)
                total_len = prefetched + extend_len

                # For SLO-violated requests, assign infinite slack so they don't constrain scheduling
                # but still get included in the prefill plan
                if getattr(req, 'slo_violated', False):
                    slack_adj = 9999999.0
                else:
                    slack_raw = getattr(req, "last_slack_ms", None)
                    if slack_raw is None:
                        slack_raw = self._compute_req_slack_ms(req, now_ms=now_ms)
                    slack_raw = 0.0 if slack_raw is None else float(slack_raw)
                    # Prefill slack: subtract overlap time; do not add TPOT here.
                    # Waiting requests are not serviced during overlap, so they just lose time.
                    slack_adj = max(slack_raw - pred_last, 0.0)

                if total_len <= 0:
                    skipped_zero_len += 1
                    logger.warning("[PREFILL-SIM] Skipping req %s: total_len=%d (prefetched=%d, extend_len=%d, origin_input_ids=%d, output_ids=%d)",
                                   req.rid, total_len, prefetched, extend_len,
                                   len(getattr(req, 'origin_input_ids', [])),
                                   len(getattr(req, 'output_ids', [])))
                    continue

                total_prefill_lens.append(total_len)
                already_prefilled_lens.append(prefetched)
                prefill_slacks.append(slack_adj)

            # Check if scenario changed from last iteration to avoid duplicate logging
            scenario_changed = (
                self._last_prefill_sim_decode_batch != decode_batch
                or self._last_prefill_sim_kv_cache != num_used
                or self._last_prefill_sim_prefill_lens != total_prefill_lens
            )

            if scenario_changed:
                skip_info = f" (skipped_zero_len={skipped_zero_len})" if skipped_zero_len > 0 else ""
                logger.info(
                    "[PREFILL-SIM] evaluate_extras: raw_candidates=%d valid=%d total_lens=%s prefill_slacks=%s already_prefilled=%s extras=%s%s%s",
                    len(candidates),
                    len(total_prefill_lens),
                    total_prefill_lens if total_prefill_lens else "[]",
                    [f"{s:.1f}" for s in prefill_slacks] if prefill_slacks else "[]",
                    already_prefilled_lens if already_prefilled_lens else "[]",
                    PREFILL_SIM_EXTRAS,
                    " (empty base)" if not total_prefill_lens else "",
                    skip_info,
                )
                # Update last scenario
                self._last_prefill_sim_decode_batch = decode_batch
                self._last_prefill_sim_kv_cache = num_used
                self._last_prefill_sim_prefill_lens = total_prefill_lens.copy()
            
            # # take a look how should we handle this iteration
            # logger.warning(
            #     "[ENGINE-SIM-INPUT] decode_batch=%d kv_cache=%d tpot_ms=%.2f slack_decode_ms=%.2f "
            #     "safety_margin_ms=%.1f n_candidates=%d total_prefill_lens=%s already_prefilled=%s "
            #     "prefill_slacks=%s",
            #     decode_batch, num_used, tpot_effective, decode_slack_ms,
            #     self.prefill_sim_engine.safety_margin_ms,
            #     len(total_prefill_lens), total_prefill_lens[:10],
            #     already_prefilled_lens[:10],
            #     [round(s, 1) for s in prefill_slacks[:10]],
            # )
            t0 = time.time()
            self._last_prefill_sim_results = self.prefill_sim_engine.evaluate_extras(
                total_prefill_lens,
                prefill_slacks,
                [0],
                already_prefilled_lens=already_prefilled_lens,
            )
            t1 = time.time()

            # # Log simulation output for comparison with sidecar
            # if self._last_prefill_sim_results and len(self._last_prefill_sim_results) > 0:
            #     _r = self._last_prefill_sim_results[0]
            #     logger.warning(
            #         "[ENGINE-SIM-OUTPUT] execution_flow=%s execution_times=%s "
            #         "base_plan=%s success=%s decode_feasible=%s min_decode_slack=%.2f",
            #         _r.execution_flow, [round(t, 2) for t in (_r.execution_times or [])],
            #         _r.base_plan, _r.success, _r.decode_feasible, _r.min_decode_slack_ms,
            #     )

            # Part 1: Handle base case (extra_len = 0)
            res = None  # Initialize to avoid scope issues
            try:
                if self._last_prefill_sim_results and len(self._last_prefill_sim_results) > 0:
                    res = self._last_prefill_sim_results[0]  # Base case only
                    status = "PASS" if res.success else ("LATE" if res.decode_feasible else "FAIL")
                    # Since another batch is running, include pred_last in reported times for visibility.
                    # reported_time = time from NOW until new_batch completes
                    #               = pred_last (last_batch completes) + res.total_time_ms (new_batch executes)
                    reported_time = res.total_time_ms + pred_last
                    reported_padded = res.padded_total_time_ms + pred_last
                    execution_flow = res.execution_flow if res.execution_flow else res.base_plan
                    if scenario_changed:
                        logger.info(
                            "[PREFILL-SIM] base extra=0 status=%s time=%.2f + %.2f = %.2fms padded=%.2fms execution_flow=%s, expected run time=%s",
                            status,
                            res.total_time_ms,
                            pred_last,
                            reported_time,
                            reported_padded,
                            execution_flow,
                            res.execution_times,
                        )
            except Exception:
                logger.exception("Failed to log base case prefill simulation results")

            # Part 2: Simulate next iteration after first chunk
            # Skip if base case failed or returned no results
            if res is None or not hasattr(res, 'execution_flow') or not res.execution_flow:
                logger.warning("[PREFILL-SIM] Skipping next iteration simulation - no base case results")
                self._predicted_ttft_for_new_admits = {}
                self._prefill_sim_runtime_ms = (time.perf_counter() - t_sim_start) * 1000.0
                return
        
            if(t1 - t0) > 0.001 or len(total_prefill_lens) > 10:
                logger.warning("[PREFILL-SIM] Taking too long to do simulation: %.4f s", (t1 - t0))
                self._predicted_ttft_for_new_admits = {128: 9999999, 8192: 99999999}
                self._prefill_sim_runtime_ms = (time.perf_counter() - t_sim_start) * 1000.0
                return


            # Extract chunk0 execution details
            tokens_left_to_process = res.execution_flow[0]  # Can be 0 for decode-only iteration
            time_consumed = res.execution_times[0] if res.execution_times and len(res.execution_times) > 0 else 0.0

            # Remove scheduled-to-be-finished prefills and update remaining state
            remaining_total_lens = []
            remaining_already_prefilled = []
            remaining_slacks = []

            for idx in range(len(total_prefill_lens)):
                total_len = total_prefill_lens[idx]
                already_prefilled = already_prefilled_lens[idx]
                slack = prefill_slacks[idx]

                remaining_len = total_len - already_prefilled  # Tokens still to process for this request

                if tokens_left_to_process >= remaining_len:
                    # This request will be fully processed, remove it
                    tokens_left_to_process -= remaining_len
                elif tokens_left_to_process > 0:
                    # This is the borderline request - partially processed
                    new_already_prefilled = already_prefilled + tokens_left_to_process
                    new_slack = max(slack - time_consumed, 0.0)  # Update slack

                    remaining_total_lens.append(total_len)
                    remaining_already_prefilled.append(new_already_prefilled)
                    remaining_slacks.append(new_slack)

                    tokens_left_to_process = 0
                else:
                    # This request hasn't been touched yet, update slack only
                    new_slack = max(slack - time_consumed, 0.0)

                    remaining_total_lens.append(total_len)
                    remaining_already_prefilled.append(already_prefilled)
                    remaining_slacks.append(new_slack)

            # Update the lists for the next evaluation
            total_prefill_lens = remaining_total_lens
            already_prefilled_lens = remaining_already_prefilled
            prefill_slacks = remaining_slacks
            
            predicted_decode_slack = decode_slack_ms - time_consumed + tpot_effective 
            if predicted_decode_slack < 0:
                predicted_decode_slack = 0.0
            
            self.prefill_sim_engine.update_decode(
                decode_batch=decode_batch,
                kv_cache=num_used,
                tpot_ms=tpot_effective,
                slack_decode_ms=predicted_decode_slack,
            )
            
            if scenario_changed:
                logger.info(f"[PREFILL-SIM] after first chunk, total_prefill_lens={total_prefill_lens}, already_prefilled_lens={already_prefilled_lens}, prefill_slacks={prefill_slacks}, predicted_decode_slack={predicted_decode_slack:.2f}ms")
            
            # Now lets predict for next iteration, if we can admit more prefill
            self._pred_prefill_results = self.prefill_sim_engine.evaluate_extras(
                total_prefill_lens,
                prefill_slacks,
                PREFILL_SIM_EXTRAS,
                already_prefilled_lens=already_prefilled_lens,
            )

            # Process prediction results for router reporting
            try:
                self._predicted_ttft_for_new_admits = {}  # Maps extra_len -> predicted TTFT (ms)
                if self._pred_prefill_results:
                    summaries = []

                    for idx, res in enumerate(self._pred_prefill_results):
                        extra_len = PREFILL_SIM_EXTRAS[idx]
                        status = "PASS" if res.success else ("LATE" if res.decode_feasible else "FAIL")
                        # Time for new admits = wait for GPU + wait for chunk0 + their batch execution
                        # 1) pred_last: ongoing batch on GPU
                        # 2) time_consumed: chunk0 execution time
                        # 3) res.total_time_ms: new batch with extras
                        reported_time = pred_last + time_consumed + res.total_time_ms
                        if status == "FAIL":
                            summaries.append(f"({extra_len}, FAIL)")
                            self._predicted_ttft_for_new_admits[extra_len] = 9999999
                        else:
                            summaries.append(f"({extra_len}, {round(reported_time)}ms)")
                            self._predicted_ttft_for_new_admits[extra_len] = reported_time

                        # Log detailed plan and timeline for each extra
                        # if scenario_changed:
                        #     execution_flow = res.execution_flow if res.execution_flow else res.base_plan
                        #     execution_times = res.execution_times if res.execution_times else []
                            # # Build timeline string: chunk(time_ms)
                            # timeline_parts = []
                            # for chunk, t_ms in zip(execution_flow, execution_times):
                            #     timeline_parts.append(f"{chunk}({t_ms:.1f}ms)")
                            # timeline_str = "[" + ", ".join(timeline_parts) + "]" if timeline_parts else "[]"
                            # logger.info(
                            #     "[PREFILL-SIM] extra=%d status=%s reported_time=%.2fms (pred_last=%.2f + chunk0=%.2f + sim=%.2f) "
                            #     "min_decode_slack=%.2fms min_prefill_slack=%.2fms iterations=%d flow=%s timeline=%s",
                            #     extra_len, status, reported_time, pred_last, time_consumed, res.total_time_ms,
                            #     res.min_decode_slack_ms, res.min_prefill_slack_ms, res.iterations,
                            #     execution_flow, timeline_str
                            # )
                    if summaries and scenario_changed:
                        logger.info("[PREFILL-SIM] extras summary: %s", ", ".join(summaries))
            except Exception:
                logger.exception("Failed to log extra prefill simulation results")

            # Record total runtime for the entire simulation process
            self._prefill_sim_runtime_ms = (time.perf_counter() - t_sim_start) * 1000.0
        except Exception:
            self._prefill_sim_runtime_ms = (time.perf_counter() - t_sim_start) * 1000.0
            logger.exception("Failed to run prefill simulator")

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        _t_merge_start = time.perf_counter()
        # Gather finished requests before they are filtered out of batches.
        finished_for_observation: List[Req] = []
        if self.last_batch is not None:
            finished_for_observation.extend(
                [req for req in self.last_batch.reqs if req.finished()]
            )
        if self.running_batch is not None:
            finished_for_observation.extend(
                [req for req in self.running_batch.reqs if req.finished()]
            )
        # In sidecar mode, KV forecast is skipped so decode-length observations are unnecessary
        if self.slo_scheduler_mode != "sidecar":
            self._submit_decode_length_observations(finished_for_observation)

        # Merge the prefill batch into the running batch
        chunked_req_to_exclude = set()
        if self.chunked_req:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            chunked_req_to_exclude.add(self.chunked_req)
            self.tree_cache.cache_unfinished_req(self.chunked_req, chunked=True)
            # chunked request keeps its rid but will get a new req_pool_idx
            if self.tp_worker.worker.model_runner.is_hybrid_gdn:
                self.req_to_token_pool.free(
                    self.chunked_req.req_pool_idx, free_mamba_cache=False
                )
            else:
                self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req is not None:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            # Filter batch
            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

            # Merge the new batch into the running batch.
            # For prefill-only batch, we can avoid going through decoding step.
            if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    # Merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)

        _merge_ms = (time.perf_counter() - _t_merge_start) * 1000

        # Drain 🔵 SchedulingContext + Assemble + Send to sidecar
        _t_drain = time.perf_counter()
        _drain_ms = 0.0
        _zmq_ms = 0.0
        if self.slo_client is not None and self.slo_scheduler_mode != "internal":
            self._drain_scheduling_context()
            _drain_ms = (time.perf_counter() - _t_drain) * 1000
            _t_zmq = time.perf_counter()
            self._assemble_and_send_engine_state()
            _zmq_ms = (time.perf_counter() - _t_zmq) * 1000

        # Run predictions BEFORE batch formation to guide scheduling decisions
        # Only run simulation-related predictions in SIMULATION mode
        # In sidecar mode, internal SLO logic is disabled — sidecar handles this
        if self.prefill_schedule_mode == PrefillScheduleMode.SIMULATION and self.slo_scheduler_mode != "sidecar":
            self._maybe_run_prefill_simulation()
            self._predict_future_kv_usage()

        _t_prefill = time.perf_counter()
        new_batch = self.get_new_batch_prefill()
        _prefill_ms = (time.perf_counter() - _t_prefill) * 1000
        _t_shadow = time.perf_counter()
        self._shadow_log_decisions()
        _shadow_ms = (time.perf_counter() - _t_shadow) * 1000
        self._last_batch_breakdown = (_merge_ms, _drain_ms, _zmq_ms, _prefill_ms, _shadow_ms)

        need_dp_attn_preparation = require_mlp_sync(self.server_args)

        if need_dp_attn_preparation and not self.spec_algorithm.is_none():
            # In speculative decoding, prefill batches and decode batches cannot be processed in the same DP attention group.
            # We prepare idle batches in advance to skip preparing decode batches when there are prefill batches in the group.
            new_batch = self.prepare_mlp_sync_batch(new_batch)
            need_dp_attn_preparation = new_batch is None

        if new_batch is not None:
            # Run prefill first if possible
            ret = new_batch
        else:
            # Run decode
            if not self.running_batch.is_empty():
                self.running_batch = self.update_running_batch(self.running_batch)
                ret = self.running_batch if not self.running_batch.is_empty() else None
            else:
                ret = None

        # Handle DP attention
        if need_dp_attn_preparation:
            self.maybe_handle_dp_balance_data()
            ret = self.prepare_mlp_sync_batch(ret)

        # log the batch infomation
        # if ret is not None:
        #     logger.info(f"running batch len reqs: {len(ret.reqs)}")
        #     # concate all extend input len in a row separate by comma
        #     extend_input_len_str = ", ".join([str(req.extend_input_len) for req in ret.reqs])
        #     logger.info(f"extend input len: {extend_input_len_str}")
        #     logger.info(f"total extend input len: {sum([req.extend_input_len for req in ret.reqs])}")
        #     # get current kv size
        #     num_used, token_usage, available_size, evictable_size = self._get_token_info()
        #     kv_size = num_used
        #     logger.info(f"kv size: {kv_size}")
        #     # only for those extend len > 1
        #     prefill_pairs = []
        #     for req in ret.reqs:
        #         if req.extend_input_len > 1:
        #             prefill_pairs.append([req.extend_input_len, len(req.prefix_indices) + req.extend_input_len])
        #     logger.info(f"prefill pairs: {prefill_pairs}")
        #     pred = self.cycle_time_predictor.predict(batch_size_tokens=sum([req.extend_input_len for req in ret.reqs]), prefill_chunk_pairs=prefill_pairs, kv_tokens_used=kv_size)
        #     logger.info(f"pred: {pred}")
                            
        return ret

    def get_num_allocatable_reqs(self, running_bs):
        res = global_server_args_dict["max_micro_batch_size"] - running_bs
        if self.pp_size > 1:
            res = min(res, self.req_to_token_pool.available_size())
        return res

    def predict_batch(self, batch: Optional[ScheduleBatch], mode: Optional[str] = None) -> float:
        """Predict cycle time for a batch; return 0 for edge cases."""
        if (
            batch is None
            or batch.is_empty()
            or self.cycle_time_predictor is None
        ):
            return 0.0

        num_used, token_usage, available_size, evictable_size = self._get_token_info()

        # Respect the batch's forward_mode when available instead of inferring from token shapes.
        batch_mode = getattr(batch, "forward_mode", None)
        inferred_mode = mode or _forward_mode_to_string(batch_mode)
        is_decode_mode = False
        if batch_mode is not None:
            try:
                is_decode_mode = batch_mode.is_decode()
            except AttributeError:
                is_decode_mode = batch_mode == ForwardMode.DECODE

        total_tokens = 0
        prefill_pairs: List[List[int]] = []
        for req in batch.reqs:
            extend_len = getattr(req, "extend_input_len", 0)
            if is_decode_mode:
                # Decode batches effectively process one token per request.
                extend_len = 1
            total_tokens += extend_len
            if not is_decode_mode and extend_len > 1:
                prefix_len = len(getattr(req, "prefix_indices", []))
                prefill_pairs.append([extend_len, prefix_len + extend_len])

        if total_tokens <= 0:
            return 0.0

        predictor = self.cycle_time_predictor
        kwargs = dict(
            batch_size_tokens=total_tokens,
            prefill_chunk_pairs=prefill_pairs,
            kv_tokens_used=num_used,
        )
        if getattr(predictor, "is_multimode", False) and inferred_mode is not None:
            kwargs["mode"] = inferred_mode

        return predictor.predict(**kwargs)

    def _abandon_chunked_prefill(self, requeue: bool = True) -> None:
        """Safely abandon the current chunked prefill request.

        Called when we need to stop a chunked prefill mid-way (e.g., SLO constraints).
        At this point, cache_unfinished_req has already been called in get_next_batch_to_run:
        - KV tokens are in radix cache (with lock_ref on last_node)
        - req_pool_idx has already been freed

        We need to:
        1. Release the radix cache lock
        2. Reset request state
        3. Optionally re-queue to retry from scratch
        4. Clear self.chunked_req
        """
        if self.chunked_req is None:
            return

        req = self.chunked_req
        logger.info(
            f"Abandoning chunked prefill for req {req.rid}, "
            f"is_chunked={req.is_chunked}, prefix_len={len(req.prefix_indices)}"
        )

        # Release the radix cache lock (acquired in cache_unfinished_req)
        if req.last_node is not None:
            if self.is_hybrid:
                self.tree_cache.dec_lock_ref(req.last_node, req.swa_uuid_for_lock)
            else:
                self.tree_cache.dec_lock_ref(req.last_node)

        # Reset request state for retry
        req.reset_for_retract()

        # Re-queue to retry from scratch (will re-match prefix from radix cache)
        if requeue:
            self._extend_requests_to_queue([req])

        self.chunked_req = None

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        # Check if the grammar is ready in the grammar queue
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        if self.try_preemption:
            # Reset batch_is_full to try preemption with a prefill adder.
            self.running_batch.batch_is_full = False

        # # SLO-aware scheduling: block new prefills if last iteration exceeded TPOT
        # if self.tpot is not None and self.server_args.enable_iteration_metrics:
        #     from sglang.srt.ui import iteration_metrics
        #     latest_metrics = iteration_metrics.get_ui_snapshot()
        #     last_iteration_time = latest_metrics.get("iteration_time_ms")
        #     if last_iteration_time is not None and last_iteration_time > self.tpot:
        #         # Don't schedule new prefill requests when we're over SLO
        #         logger.info(
        #             f"Blocking new prefill: last_iteration_time={last_iteration_time:.2f}ms > tpot={self.tpot:.2f}ms"
        #         )
        #         return None
        # estimate the time for running batch

        
        # Handle the cases where prefill is not allowed
        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            self._shadow_capture_decode_only("resource_constraint", **self._shadow_common_context())
            return None

        running_bs = len(self.running_batch.reqs)
        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked request has just been released.
        # In PP case, a chunked req can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked request to be added, otherwise, there will be a memory leak.
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and not self.chunked_req
            and not self.try_preemption
        ):
            self.running_batch.batch_is_full = True
            self._shadow_capture_decode_only("resource_constraint", **self._shadow_common_context())
            return None

        if self.enable_hierarchical_cache:
            self.tree_cache.check_hicache_events()

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue)

        # Decode tokens for mixed chunk mode
        decode_tokens = running_bs if self.is_mixed_chunk else 0

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
        # ── Step 1: Compute internal scheduling decision (no early returns) ──
        internal_decode_only = False
        effective_chunk_size = self.chunked_prefill_size
        effective_predictor = None
        effective_tpot = None
        effective_target = None

        # In sidecar mode, skip all internal SLO computation — sidecar handles scheduling
        if self.slo_scheduler_mode != "sidecar":
            if self.prefill_schedule_mode == PrefillScheduleMode.SIMULATION:
                sim_budget = self._get_simulation_chunk_budget()
                if sim_budget == 0:
                    logger.debug("SIMULATION mode: decode-only iteration per simulation plan")
                    internal_decode_only = True
                else:
                    # sim_budget is pure prefill tokens; add decode_tokens back because
                    # PrefillAdder will subtract them (it expects total token budget)
                    effective_chunk_size = sim_budget + decode_tokens
            elif self.prefill_schedule_mode == PrefillScheduleMode.PREDICTOR:
                pred = self.predict_batch(self.running_batch)
                if self.target_iteration_time_ms is not None and pred > self.target_iteration_time_ms:
                    internal_decode_only = True
                elif self.tpot is not None and self.cycle_time_predictor is not None:
                    effective_predictor = self.cycle_time_predictor
                    effective_tpot = self.tpot
                    effective_target = self.target_iteration_time_ms
                else:
                    logger.debug("PREDICTOR mode: TPOT not set, falling back to BUDGET behavior")
            elif self.prefill_schedule_mode == PrefillScheduleMode.SLACK:
                min_slack = self._compute_min_valid_decode_slack()
                if min_slack is not None and self.cycle_time_predictor is not None:
                    pred = self.predict_batch(self.running_batch)
                    if pred > min_slack:
                        internal_decode_only = True
                    else:
                        effective_predictor = self.cycle_time_predictor
                        effective_tpot = self.tpot
                        effective_target = min_slack
                        logger.info(
                            "SLACK mode: using min_decode_slack=%.2fms as target_iteration_time",
                            min_slack,
                        )
                else:
                    if min_slack is None:
                        logger.debug("SLACK mode: no valid decode slack, falling back to BUDGET behavior")
                    else:
                        logger.info("SLACK mode: cycle_time_predictor unavailable, falling back to BUDGET behavior")

        # ── Step 2: Shadow/sidecar logging (always captures internal decision) ──
        if self.slo_scheduler_mode in ("shadow", "shadow-sidecar", "sidecar"):
            if self.slo_scheduler_mode == "sidecar":
                # Internal SLO skipped; log simplified snapshot
                _ctx = self._shadow_common_context()
                self._shadow_capture_pre_batch(None, "sidecar_only", None, **_ctx)
            else:
                # shadow / shadow-sidecar: full internal capture
                _mode = self.prefill_schedule_mode.value
                _slack = min_slack if self.prefill_schedule_mode == PrefillScheduleMode.SLACK else None
                _ctx = self._shadow_common_context()
                if self.prefill_schedule_mode == PrefillScheduleMode.SIMULATION:
                    _base = self._last_prefill_sim_results[0] if self._last_prefill_sim_results else None
                    _ctx.update(
                        sim_budget=sim_budget,
                        sim_execution_flow=list(getattr(_base, 'execution_flow', []) or []) if _base else None,
                        sim_decode_slack_ms=self.prefill_sim_engine.config.slack_decode_ms if self.prefill_sim_engine and self.prefill_sim_engine.config else None,
                        sim_safety_margin_ms=getattr(self.prefill_sim_engine, 'safety_margin_ms', None),
                    )
                elif self.prefill_schedule_mode == PrefillScheduleMode.PREDICTOR:
                    _ctx.update(pred_decode_time=pred, target_iteration_time_ms=self.target_iteration_time_ms)
                elif self.prefill_schedule_mode == PrefillScheduleMode.SLACK:
                    _ctx.update(min_decode_slack_ms=min_slack)
                    # pred exists only when min_slack and predictor were both available
                    if min_slack is not None and self.cycle_time_predictor is not None:
                        _ctx.update(pred_decode_time=pred)
                if internal_decode_only:
                    self._shadow_capture_decode_only(_mode, **_ctx)
                else:
                    self._shadow_capture_pre_batch(effective_target, _mode, _slack, **_ctx)

        # ── Step 3: Apply decision (sidecar overrides internal if available) ──
        if self.slo_scheduler_mode in ("shadow-sidecar", "sidecar") and self._last_sidecar_decision is not None:
            decision = self._last_sidecar_decision
            if decision.decode_only_iteration or decision.max_prefill_tokens <= 0:
                self._abandon_chunked_prefill()
                return None
            sidecar_budget = max(0, decision.max_prefill_tokens)
            if self.chunked_prefill_size is not None:
                sidecar_budget = min(sidecar_budget, self.chunked_prefill_size)
            effective_chunk_size = sidecar_budget + decode_tokens
            effective_target = decision.target_iteration_time_ms
            effective_predictor = None
            effective_tpot = None
        elif self.slo_scheduler_mode in ("shadow-sidecar", "sidecar"):
            # Sidecar unavailable — fallback to chunked_prefill_size (no SLO awareness)
            pass
        elif internal_decode_only:
            self._abandon_chunked_prefill()
            return None

        # Prefill policy
        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            effective_chunk_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            cycle_time_predictor=effective_predictor,
            tpot_slo=effective_tpot,
            target_iteration_time_ms=effective_target,
            max_total_num_tokens=self.max_total_num_tokens,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        # Pre-compute slack for waiting requests so scheduling logic can reuse it.
        now_ms = time.time() * 1000.0
        for req in self.waiting_queue:
            self._compute_req_slack_ms(req, now_ms=now_ms)

        if self.enable_lora:
            lora_set = set([req.lora_id for req in self.running_batch.reqs])

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:

            if self.enable_lora and not self.tp_worker.can_run_lora_batch(
                lora_set
                | set([req.lora_id for req in adder.can_run_list])
                | set([req.lora_id])
            ):
                self.running_batch.batch_is_full = True
                break

            running_bs = len(self.running_batch.reqs) - len(adder.preempt_list)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if not self.try_preemption:
                    break
                if not adder.preempt_to_schedule(req, self.server_args):
                    break

            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=(self.chunked_req is not None),
                truncation_align_size=self.truncation_align_size,
            )

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (not self.running_batch.is_empty())
                    else:
                        self.running_batch.batch_is_full = True
                break

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        if self.enable_metrics:
            # only record queue time when enable_metrics is True to avoid overhead
            for req in can_run_list:
                req.queue_time_end = time.perf_counter()
                req.add_latency(RequestStage.PREFILL_WAITING)

        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]
        if adder.preempt_list:
            self._extend_requests_to_queue(adder.preempt_list)

        if adder.new_chunked_req is not None:
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req:
            self.chunked_req.is_chunked += 1

        # Print stats
        if self.current_scheduler_metrics_enabled():
            self.log_prefill_stats(adder, can_run_list, running_bs)

        self._shadow_capture_post_batch(adder.log_input_tokens, len(can_run_list))

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=self.chunked_req,
        )
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_batch.prepare_for_extend()

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            self.running_batch.filter_batch()
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=self.running_batch.batch_is_full
            )
        else:
            new_batch.decoding_reqs = None

        return new_batch

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # Check if decode out of memory
        if not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier) or (
            TEST_RETRACT and batch.batch_size() > 10
        ):
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio = batch.retract_decode(self.server_args)
            num_retracted_reqs = len(retracted_reqs)
            self.new_token_ratio = new_token_ratio

            logger.info(
                "KV cache pool is full. Retract requests. "
                f"#retracted_reqs: {num_retracted_reqs}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )

            self._extend_requests_to_queue(retracted_reqs, is_retracted=True)
            self.total_retracted_reqs += num_retracted_reqs
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        # Update batch tensors
        batch.prepare_for_decode()
        return batch

    def run_batch(
        self, batch: ScheduleBatch
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.forward_ct += 1

        # Whether to run the profiler
        self._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # Run forward
        if self.is_generation:
            if self.spec_algorithm.is_none():
                model_worker_batch = batch.get_model_worker_batch()

                if self.pp_group.is_last_rank:
                    logits_output, next_token_ids, can_run_cuda_graph = (
                        self.tp_worker.forward_batch_generation(model_worker_batch)
                    )
                else:
                    pp_hidden_states_proxy_tensors, _, can_run_cuda_graph = (
                        self.tp_worker.forward_batch_generation(model_worker_batch)
                    )
                bid = model_worker_batch.bid
            else:
                (
                    logits_output,
                    next_token_ids,
                    bid,
                    num_accepted_tokens,
                    can_run_cuda_graph,
                ) = self.draft_worker.forward_batch_speculative_generation(batch)
                bs = batch.batch_size()
                self.spec_num_total_accepted_tokens += num_accepted_tokens + bs
                self.spec_num_total_forward_ct += bs
                self.num_generated_tokens += num_accepted_tokens

            if self.pp_group.is_last_rank:
                batch.output_ids = next_token_ids

            # These 2 values are needed for processing the output, but the values can be
            # modified by overlap schedule. So we have to copy them here so that
            # we can use the correct values in output processing.
            if batch.return_logprob or self.spec_algorithm.is_eagle():
                extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]
            else:
                extend_input_len_per_req = None
            if batch.return_logprob:
                extend_logprob_start_len_per_req = [
                    req.extend_logprob_start_len for req in batch.reqs
                ]
            else:
                extend_logprob_start_len_per_req = None

            ret = GenerationBatchResult(
                logits_output=logits_output if self.pp_group.is_last_rank else None,
                pp_hidden_states_proxy_tensors=(
                    pp_hidden_states_proxy_tensors
                    if not self.pp_group.is_last_rank
                    else None
                ),
                next_token_ids=next_token_ids if self.pp_group.is_last_rank else None,
                extend_input_len_per_req=extend_input_len_per_req,
                extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
                bid=bid,
                can_run_cuda_graph=can_run_cuda_graph,
            )
        else:  # embedding or reward model
            model_worker_batch = batch.get_model_worker_batch()
            embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
            ret = EmbeddingBatchResult(
                embeddings=embeddings, bid=model_worker_batch.bid
            )
        return ret

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        launch_done: Optional[threading.Event] = None,
    ):
        

        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result, launch_done)
            for req in batch.reqs:
                trace_slice(
                    "decode loop",
                    req.rid,
                    auto_next_anon=not req.finished(),
                    thread_finish_flag=req.finished(),
                )

        elif batch.forward_mode.is_extend():
            self.process_batch_result_prefill(batch, result, launch_done)
            for req in batch.reqs:
                trace_slice(
                    "prefill",
                    req.rid,
                    auto_next_anon=not req.finished(),
                    thread_finish_flag=req.finished(),
                )
        elif batch.forward_mode.is_idle():
            if self.enable_overlap:
                self.tp_worker.resolve_last_batch_result(launch_done)
                self.set_next_batch_sampling_info_done(batch)
        elif batch.forward_mode.is_dummy_first():
            self.set_next_batch_sampling_info_done(batch)

        self.maybe_send_health_check_signal()

        # Update the last completed timestamp and record interval on batch for metrics
        # Measure time interval since the last completed process_batch_result
        last_interval_ms = None
        now = time.perf_counter()
        if self._last_process_result_end_time is not None:
            last_interval_ms = (now - self._last_process_result_end_time) * 1000
        
        self._last_process_result_end_time = now
        if last_interval_ms is not None:
            setattr(batch, "since_last_process_ms", last_interval_ms)

    def maybe_send_health_check_signal(self):
        if self.return_health_check_ct:
            # Return some signal for the health check.
            # This is used to prevent the health check signal being blocked by long context prefill.
            # However, one minor issue is that this code path does not check the status of detokenizer manager.
            self.return_health_check_ct -= 1
            self.send_to_tokenizer.send_pyobj(HealthCheckOutput())

    def prepare_mlp_sync_batch(self, local_batch: ScheduleBatch):
        return self.prepare_mlp_sync_batch_raw(
            local_batch,
            dp_size=self.server_args.dp_size,
            attn_tp_size=self.attn_tp_size,
            tp_group=self.tp_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=self.server_args.disable_cuda_graph,
            spec_algorithm=self.spec_algorithm,
            speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
            require_mlp_tp_gather=require_mlp_tp_gather(self.server_args),
            disable_overlap_schedule=self.server_args.disable_overlap_schedule,
        )

    @staticmethod
    def prepare_mlp_sync_batch_raw(
        local_batch: ScheduleBatch,
        dp_size,
        attn_tp_size: int,
        tp_group,
        get_idle_batch,
        disable_cuda_graph: bool,
        spec_algorithm,
        speculative_num_draft_tokens,
        require_mlp_tp_gather: bool,
        disable_overlap_schedule: bool,
    ):
        # Check if other DP workers have running batches
        if local_batch is None:
            num_tokens = 0
            num_tokens_for_logprob = 0
        elif local_batch.forward_mode.is_decode():
            num_tokens = local_batch.batch_size()
            num_tokens_for_logprob = num_tokens
        else:
            num_tokens = local_batch.extend_num_tokens
            num_tokens_for_logprob = sum(
                [
                    # We should have at least 1 token for sample in every case.
                    max(extend_len - logprob_start_len, 1)
                    for logprob_start_len, extend_len in zip(
                        local_batch.extend_logprob_start_lens, local_batch.extend_lens
                    )
                ]
            )

        if local_batch is None or local_batch.forward_mode.is_decode_or_idle():
            can_cuda_graph = 1
        else:
            can_cuda_graph = 0

        is_extend_in_batch = (
            local_batch.forward_mode.is_extend() if local_batch else False
        )

        tbo_preparer = TboDPAttentionPreparer()
        if disable_overlap_schedule:
            group = tp_group.device_group
            device = tp_group.device
        else:
            group = tp_group.cpu_group
            device = "cpu"

        local_info = torch.tensor(
            [
                num_tokens,
                can_cuda_graph,
                num_tokens_for_logprob,
                is_extend_in_batch,
                *tbo_preparer.prepare_all_gather(
                    local_batch,
                ),
            ],
            dtype=torch.int64,
            device=device,
        )
        global_info = torch.empty(
            (dp_size, attn_tp_size, 6),
            dtype=torch.int64,
            device=device,
        )
        torch.distributed.all_gather_into_tensor(
            global_info.flatten(),
            local_info,
            group=group,
        )
        global_num_tokens = global_info[:, 0, 0].tolist()
        can_cuda_graph = min(global_info[:, 0, 1].tolist())
        global_num_tokens_for_logprob = global_info[:, 0, 2].tolist()
        is_extend_in_batch = global_info[:, 0, 3].tolist()

        tbo_split_seq_index, global_forward_mode = tbo_preparer.compute_output(
            global_info[:, :, 4:6]
        )

        if local_batch is None and max(global_num_tokens) > 0:
            local_batch = get_idle_batch()

        if local_batch is not None:
            # TODO: handle the case when moe_dense_tp_size != 1
            if not require_mlp_tp_gather:
                local_batch.global_num_tokens = [num_tokens]
                local_batch.global_num_tokens_for_logprob = [num_tokens_for_logprob]
            else:
                local_batch.global_num_tokens = global_num_tokens
                local_batch.global_num_tokens_for_logprob = (
                    global_num_tokens_for_logprob
                )
            local_batch.is_extend_in_batch = any(is_extend_in_batch)
            local_batch.tbo_split_seq_index = tbo_split_seq_index
            local_batch.global_forward_mode = global_forward_mode

            # Check forward mode for cuda graph
            if not disable_cuda_graph:
                local_batch.can_run_dp_cuda_graph = can_cuda_graph

        return local_batch

    def get_idle_batch(self):
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch

    def move_ready_grammar_requests(self):
        """Move requests whose grammar objects are ready from grammar_queue to waiting_queue."""

        num_ready_reqs = 0
        num_timeout_reqs = 0
        for req in self.grammar_queue:
            try:
                if req.finished():  # It is aborted by AbortReq
                    num_ready_reqs += 1
                    continue
                req.grammar = req.grammar.result(timeout=0.03)
                self.grammar_backend.set_cache(req.grammar_key, req.grammar.copy())
                if req.grammar is INVALID_GRAMMAR_OBJ:
                    req.set_finish_with_abort(
                        f"Invalid grammar request: {req.grammar_key=}"
                    )
                num_ready_reqs += 1
            except futures._base.TimeoutError:
                req.grammar_wait_ct += 1
                # NOTE(lianmin): this timeout is the waiting time of the above line. It is
                # not the waiting time from it enters the grammar queue.
                if req.grammar_wait_ct > GRAMMAR_TIMEOUT / 0.03:
                    num_timeout_reqs = 1
                break

        if self.server_args.enable_dp_attention:
            tp_size = self.attn_tp_size
            tp_group = self.attn_tp_cpu_group
        else:
            tp_size = self.tp_size
            tp_group = self.tp_cpu_group

        if tp_size > 1:
            # Sync across TP ranks to make sure they have the same number of ready requests
            tensor = torch.tensor([num_ready_reqs, num_timeout_reqs], dtype=torch.int32)
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MAX, group=tp_group
            )
            num_ready_reqs_max, num_timeout_reqs_max = tensor.tolist()

            for i in range(num_ready_reqs, num_ready_reqs_max):
                req = self.grammar_queue[i]
                if req.finished():  # It is aborted by AbortReq
                    continue
                req.grammar = req.grammar.result()
                self.grammar_backend.set_cache(req.grammar_key, req.grammar.copy())
                if req.grammar is INVALID_GRAMMAR_OBJ:
                    req.set_finish_with_abort(
                        f"Invalid grammar request: {req.grammar_key=}"
                    )
        else:
            num_ready_reqs_max = num_ready_reqs
            num_timeout_reqs_max = num_timeout_reqs

        for i in range(num_ready_reqs, num_ready_reqs + num_timeout_reqs_max):
            req = self.grammar_queue[i]
            req.grammar.cancel()
            error_msg = f"Grammar preprocessing timed out for {req.grammar_key=}"
            req.set_finish_with_abort(error_msg)
            self.grammar_backend.set_cache(req.grammar_key, INVALID_GRAMMAR_OBJ)
        num_ready_reqs = num_ready_reqs_max + num_timeout_reqs_max

        self._extend_requests_to_queue(self.grammar_queue[:num_ready_reqs])
        self.grammar_queue = self.grammar_queue[num_ready_reqs:]

    def set_next_batch_sampling_info_done(self, batch: ScheduleBatch):
        if batch.next_batch_sampling_info:
            if batch.next_batch_sampling_info.grammars is not None:
                batch.next_batch_sampling_info.update_regex_vocab_mask()
                self.current_stream.synchronize()
            batch.next_batch_sampling_info.sampling_info_done.set()

    def watchdog_thread(self):
        """A watch dog thread that will try to kill the server itself if one forward batch takes too long."""
        self.watchdog_last_forward_ct = 0
        self.watchdog_last_time = time.perf_counter()

        while True:
            current = time.perf_counter()
            if self.cur_batch is not None:
                if self.watchdog_last_forward_ct == self.forward_ct:
                    if current > self.watchdog_last_time + self.watchdog_timeout:
                        break
                else:
                    self.watchdog_last_forward_ct = self.forward_ct
                    self.watchdog_last_time = current
            time.sleep(self.watchdog_timeout // 2)

        if not disable_request_logging():
            # Print batch size and memory pool info to check whether there are de-sync issues.
            if self.is_hybrid:
                (
                    _,
                    _,
                    _,
                    _,
                    full_available_size,
                    full_evictable_size,
                    swa_available_size,
                    swa_evictable_size,
                ) = self._get_swa_token_info()
                info_msg = (
                    f"{full_available_size=}, "
                    f"{full_evictable_size=}, "
                    f"{swa_available_size=}, "
                    f"{swa_evictable_size=}, "
                )
            else:
                _, _, available_size, evictable_size = self._get_token_info()
                info_msg = f"{available_size=}, " f"{evictable_size=}, "
            logger.error(
                f"{self.cur_batch.batch_size()=}, "
                f"{self.cur_batch.reqs=}, "
                f"{info_msg}"
            )

        pyspy_dump_schedulers()
        logger.error(f"Watchdog timeout ({self.watchdog_timeout=})")
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)

        # Wait for some time so that the parent process can print the error.
        time.sleep(5)
        self.parent_process.send_signal(signal.SIGQUIT)

    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        success = self.flush_cache()
        return FlushCacheReqOutput(success=success)

    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        if self.enable_hierarchical_cache:
            self.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    def flush_cache(self):
        """Flush the memory pool and cache."""
        if (
            len(self.waiting_queue) == 0
            and self.running_batch.is_empty()
            and (self.pp_size == 1 or all(x.is_empty() for x in self.running_mbs))
        ):
            self.cur_batch = None
            self.last_batch = None
            self.tree_cache.reset()
            if self.grammar_backend:
                self.grammar_backend.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()

            self.num_generated_tokens = 0
            self.forward_ct_decode = 0
            self.spec_num_total_accepted_tokens = 0
            self.spec_num_total_forward_ct = 0
            self.cum_spec_accept_length = 0
            self.cum_spec_accept_count = 0
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
            if_success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {len(self.running_batch.reqs)}"
            )
            if_success = False
        return if_success

    def get_load(self, recv_req: GetLoadReqInput = None) -> GetLoadReqOutput:
        # TODO(lsyin): use dynamically maintained num_waiting_tokens

        if self.is_hybrid:
            num_tokens_full = (
                self.full_tokens_per_layer
                - self.token_to_kv_pool_allocator.full_available_size()
                - self.tree_cache.full_evictable_size()
            )
            num_tokens_swa = (
                self.swa_tokens_per_layer
                - self.token_to_kv_pool_allocator.swa_available_size()
                - self.tree_cache.swa_evictable_size()
            )
            num_tokens = max(num_tokens_full, num_tokens_swa)
        else:
            num_tokens = (
                self.max_total_num_tokens
                - self.token_to_kv_pool_allocator.available_size()
                - self.tree_cache.evictable_size()
            )

        # Tokens in waiting queue, bootstrap queue, prealloc queue
        num_tokens += sum(len(req.origin_input_ids) for req in self.waiting_queue)
        num_waiting_reqs = len(self.waiting_queue)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            num_tokens += sum(
                len(req.origin_input_ids)
                for req in self.disagg_prefill_bootstrap_queue.queue
            )
            num_waiting_reqs += len(self.disagg_prefill_bootstrap_queue.queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            num_tokens += sum(
                len(req.req.origin_input_ids)
                for req in self.disagg_decode_prealloc_queue.queue
            )
            num_waiting_reqs += len(self.disagg_decode_prealloc_queue.queue)

        return GetLoadReqOutput(
            dp_rank=self.dp_rank,
            num_reqs=len(self.running_batch.reqs) + num_waiting_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_tokens=num_tokens,
        )

    def get_ui_metrics(self, recv_req: GetUIMetricsReqInput = None) -> GetUIMetricsReqOutput:
        """Get UI metrics from the iteration_metrics module."""
        from sglang.srt.ui import iteration_metrics

        metrics = iteration_metrics.get_ui_snapshot()
        return GetUIMetricsReqOutput(metrics=metrics)

    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = dict(global_server_args_dict)
        ret["last_gen_throughput"] = self.last_gen_throughput
        # Expose recent scheduler batch signals for UIs/monitoring
        try:
            ret["last_prefill_tokens"] = int(self.last_prefill_tokens)
        except Exception:
            ret["last_prefill_tokens"] = 0
        # Expose last stats timestamps to infer most recent step type
        try:
            ret["last_prefill_tic"] = float(self.last_prefill_stats_tic)
        except Exception:
            pass
        try:
            ret["last_decode_tic"] = float(self.last_decode_stats_tic)
        except Exception:
            pass
        try:
            # Use stats.num_running_reqs if set by metrics mixin; fallback to current running batch
            if hasattr(self, "stats") and getattr(self.stats, "num_running_reqs", None) is not None:
                num_running = int(self.stats.num_running_reqs)
            else:
                num_running = len(self.running_batch.reqs)
        except Exception:
            num_running = len(self.running_batch.reqs)
        ret["num_running_reqs"] = int(num_running)
        # KV occupancy (exclude waiting queues)
        try:
            if self.is_hybrid:
                (
                    full_num_used,
                    swa_num_used,
                    _full_token_usage,
                    _swa_token_usage,
                    _full_available_size,
                    _full_evictable_size,
                    _swa_available_size,
                    _swa_evictable_size,
                ) = self._get_swa_token_info()
                kv_used = max(int(full_num_used), int(swa_num_used))
            else:
                num_used, _token_usage, _avail, _evict = self._get_token_info()
                kv_used = int(num_used)
            ret["kv_tokens_used"] = kv_used
        except Exception:
            pass
        ret["memory_usage"] = {
            "weight": round(
                self.tp_worker.worker.model_runner.weight_load_mem_usage, 2
            ),
            "kvcache": round(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2
            ),
            "token_capacity": int(self.max_total_num_tokens),
        }
        # Per-pass embedding input size observed at runner boundary
        try:
            mr = self.tp_worker.worker.model_runner
            ret["input_tokens"] = int(getattr(mr, "last_input_tokens", 0) or 0)
            ret["input_step_type"] = getattr(mr, "last_input_step_type", "")
            ret["last_input_tic"] = float(getattr(mr, "last_input_tic", 0.0) or 0.0)
        except Exception:
            pass

        ret["memory_usage"]["graph"] = round(
            self.tp_worker.worker.model_runner.graph_mem_usage, 2
        )

        if not self.spec_algorithm.is_none() and self.cum_spec_accept_count > 0:
            ret["avg_spec_accept_length"] = (
                self.cum_spec_accept_length / self.cum_spec_accept_count
            )
        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.step_time_dict

        return GetInternalStateReqOutput(internal_state=ret)

    def set_internal_state(self, recv_req: SetInternalStateReq):
        server_args_dict = recv_req.server_args
        args_allow_update = set(
            [
                "max_micro_batch_size",
                "speculative_accept_threshold_single",
                "speculative_accept_threshold_acc",
            ]
        )
        if_success = True
        for k, v in server_args_dict.items():
            if k not in args_allow_update:
                logging.warning(f"Updating {k} is not supported.")
                if_success = False
                break
            elif k == "max_micro_batch_size" and (
                v > self.max_running_requests // self.pp_size or v < 1
            ):
                logging.warning(
                    f"Updating {k} to {v} is rejected because it is out of the valid range [1, {self.max_running_requests // self.pp_size}]."
                )
                if_success = False
                break
        if if_success:
            if not self.spec_algorithm.is_none() and self.cum_spec_accept_count > 0:
                avg_spec_accept_length = (
                    self.cum_spec_accept_length / self.cum_spec_accept_count
                )
                logger.info(f"{avg_spec_accept_length=}")
            self.cum_spec_accept_length = self.cum_spec_accept_count = 0
            for k, v in server_args_dict.items():
                global_server_args_dict[k] = v
            logger.info(f"Global server args updated! {global_server_args_dict=}")
        return SetInternalStateReqOutput(
            updated=True,
            server_args=global_server_args_dict,
        )

    def set_tpot(self, recv_req: SetTPOTReqInput) -> SetTPOTReqOutput:
        """Set a global TPOT value on the scheduler. Logs for visibility."""
        try:
            self.tpot = float(recv_req.tpot) * 0.97 # LEAVE SOME ROOM
            logger.info(f"[Scheduler] set_tpot received: tpot={self.tpot}")
            if self.slo_scheduler_mode != "sidecar":
                self._refresh_iteration_time_target()
            return SetTPOTReqOutput(success=True, tpot=self.tpot, message="ok")
        except Exception as e:
            logger.error(f"[Scheduler] set_tpot error: {e}")
            current = self.tpot if hasattr(self, "tpot") and self.tpot is not None else 0.0
            return SetTPOTReqOutput(success=False, tpot=float(current), message=str(e))

    def handle_rpc_request(self, recv_req: RpcReqInput):
        # Handle RPC requests
        logger.info(
            f"handle_rpc_request: {recv_req.method}, param: {recv_req.parameters}"
        )

        success = True
        exec = None
        try:
            func = getattr(self, recv_req.method)
            func(recv_req.parameters)
        except Exception as e:
            success = False
            exec = e
            logger.error(f"Failed to call rpc {recv_req.method}: {str(e)}")

        barrier()
        return RpcReqOutput(success, "" if not exec else str(exec))

    def abort_request(self, recv_req: AbortReq):
        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            req = self.waiting_queue.pop(i)
            if self.enable_hicache_storage:
                # to release prefetch events associated with the request
                self.tree_cache.release_aborted_request(req.rid)
            self.send_to_tokenizer.send_pyobj(AbortReq(req.rid))
            # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.tree_cache.cache_finished_req(req)

            logger.debug(f"Abort queued request. {req.rid=}")

        # Delete the requests in the grammar queue
        for req in self.grammar_queue:
            # Abort method 2: call `set_finish_with_abort`
            # The request will still run one prefill forward pass.
            # In this case, we change the input_ids to be only one token to make this prefill cheap.
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                logger.debug(f"Abort grammar queue request. {req.rid=}")
                if req.grammar:
                    req.grammar.cancel()
                req.set_finish_with_abort("Aborted by AbortReq.")

        # Delete requests not in the waiting queue when PD disaggregation is enabled
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Abort requests that have not yet been bootstrapped
            for i, req in enumerate(self.disagg_prefill_bootstrap_queue.queue):
                logger.debug(f"Abort bootstrap queue request. {req.rid=}")
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

            # Abort in-flight requests
            for i, req in enumerate(self.disagg_prefill_inflight_queue):
                logger.debug(f"Abort inflight queue request. {req.rid=}")
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Abort requests that have not yet finished preallocation
            for i, decode_req in enumerate(self.disagg_decode_prealloc_queue.queue):
                logger.debug(f"Abort prealloc queue request. {decode_req.req.rid=}")
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    if hasattr(decode_req.kv_receiver, "abort"):
                        decode_req.kv_receiver.abort()

            # Abort requests waiting for kvcache to release tree cache
            for i, decode_req in enumerate(self.disagg_decode_transfer_queue.queue):
                logger.debug(f"Abort transfer queue request. {decode_req.req.rid=}")
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    if hasattr(decode_req.kv_receiver, "abort"):
                        decode_req.kv_receiver.abort()

        # Delete requests in the running batch
        if self.cur_batch is self.running_batch or self.cur_batch is None:
            reqs = self.running_batch.reqs
        else:
            reqs = self.running_batch.reqs + self.cur_batch.reqs

        for req in reqs:
            if not req.finished() and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # Abort method 3: set `to_abort=True`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_abort = True

    def _pause_engine(self) -> Tuple[List[Req], int]:
        raise NotImplementedError()

    def load_lora_adapter(
        self, recv_req: LoadLoRAAdapterReqInput
    ) -> LoadLoRAAdapterReqOutput:
        """In-place loading a new lora adapter from disk or huggingface."""

        result = self.tp_worker.load_lora_adapter(recv_req)
        return result

    def unload_lora_adapter(
        self, recv_req: UnloadLoRAAdapterReqInput
    ) -> UnloadLoRAAdapterReqOutput:
        """Unload the lora adapter."""

        result = self.tp_worker.unload_lora_adapter(recv_req)
        return result

    def register_multi_tokenizer(self, recv_req: MultiTokenizerRegisterReq):
        self.send_to_detokenizer.send_pyobj(recv_req)
        return recv_req

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        """Init the seed and client instance communication group."""
        success, message = self.tp_worker.init_weights_send_group_for_remote_instance(
            recv_req
        )
        return InitWeightsSendGroupForRemoteInstanceReqOutput(success, message)

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        """Send the seed instance weights to the destination instance."""
        success, message = self.tp_worker.send_weights_to_remote_instance(recv_req)
        return SendWeightsToRemoteInstanceReqOutput(success, message)

    def slow_down(self, recv_req: SlowDownReqInput):
        t = recv_req.forward_sleep_time
        if t is not None and t <= 0:
            t = None
        self.forward_sleep_time = t
        return SlowDownReqOutput()

    def expert_distribution_handle(self, recv_req: ExpertDistributionReq):
        if recv_req == ExpertDistributionReq.START_RECORD:
            get_global_expert_distribution_recorder().start_record()
        elif recv_req == ExpertDistributionReq.STOP_RECORD:
            get_global_expert_distribution_recorder().stop_record()
        elif recv_req == ExpertDistributionReq.DUMP_RECORD:
            get_global_expert_distribution_recorder().dump_record()
        else:
            raise ValueError(f"Unrecognized ExpertDistributionReq value: {recv_req=}")
        return ExpertDistributionReqOutput()

    def open_session(self, recv_req: OpenSessionReqInput):
        # handle error
        session_id = recv_req.session_id
        if session_id in self.sessions:
            logger.warning(f"session id {session_id} already exist, cannot open.")
            return OpenSessionReqOutput(session_id, False)
        elif session_id is None:
            logger.warning("session id is None, cannot open.")
            return OpenSessionReqOutput(session_id, False)
        else:
            self.sessions[session_id] = Session(
                recv_req.capacity_of_str_len, session_id
            )
            return OpenSessionReqOutput(session_id, True)

    def close_session(self, recv_req: CloseSessionReqInput):
        # handle error
        session_id = recv_req.session_id
        if session_id not in self.sessions:
            logger.warning(f"session id {session_id} does not exist, cannot delete.")
        else:
            del self.sessions[session_id]

    def get_print_prefix(self):
        prefix = ""
        if self.attn_dp_rank is not None:
            prefix += f" DP{self.attn_dp_rank}"
        if self.server_args.tp_size > 1:
            prefix += f" TP{self.tp_rank}"
        if self.pp_size > 1:
            prefix += f" PP{self.pp_rank}"
        return prefix

    def current_scheduler_metrics_enabled(self):
        return self.attn_tp_rank == 0 or self.enable_metrics_for_all_schedulers

    def maybe_sleep_on_idle(self):
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()

    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")
        self.send_to_detokenizer.send_pyobj(recv_req)
        return None


class IdleSleeper:
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when sglang does nothing. This would lead not only
    to power savings, but also to more CPU thermal headroom when a request
    eventually comes. This is important in cases when multiple GPUs are connected
    as each GPU would otherwise pin one thread at 100% CPU usage.

    The simplest solution is to use zmq.Poller on all sockets that may receive
    data that needs handling immediately.
    """

    def __init__(self, sockets):
        self.poller = zmq.Poller()
        self.last_empty_time = time.time()
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

    def maybe_sleep(self):
        self.poller.poll(1000)
        if (
            global_config.torch_empty_cache_interval > 0
            and time.time() - self.last_empty_time
            > global_config.torch_empty_cache_interval
        ):
            self.last_empty_time = time.time()
            torch.cuda.empty_cache()


def is_health_check_generate_req(recv_req):
    return getattr(recv_req, "rid", "").startswith("HEALTH_CHECK")


def is_work_request(recv_req):
    return isinstance(
        recv_req,
        (
            TokenizedGenerateReqInput,
            TokenizedEmbeddingReqInput,
            BatchTokenizedGenerateReqInput,
            BatchTokenizedEmbeddingReqInput,
        ),
    )


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    balance_meta: Optional[DPBalanceMeta] = None,
):
    if server_args.enable_trace:
        process_tracing_init(server_args.oltp_traces_endpoint, "sglang")
        if server_args.disaggregation_mode == "null":
            thread_label = "Scheduler"
            trace_set_thread_info(thread_label, tp_rank, dp_rank)

    if (numa_node := server_args.numa_node) is not None:
        numa_bind_to_node(numa_node[gpu_id])

    # Generate the prefix
    prefix = ""
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            dp_balance_meta=balance_meta,
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )

        disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
        if disaggregation_mode == DisaggregationMode.NULL:
            if server_args.pp_size > 1:
                scheduler.event_loop_pp()
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap()
            else:
                scheduler.event_loop_normal()
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_prefill()
            else:
                if server_args.pp_size > 1:
                    scheduler.event_loop_pp_disagg_prefill()
                else:
                    scheduler.event_loop_normal_disagg_prefill()

        elif disaggregation_mode == DisaggregationMode.DECODE:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_decode()
            else:
                scheduler.event_loop_normal_disagg_decode()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
