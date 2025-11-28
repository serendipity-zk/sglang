import time
import numpy as np
from typing import Tuple, Optional, List, Any, Union, Dict

# Assumed import based on your snippet
from exp.length_monitor import RequestOutputLengthProfile

class MonteCarloOutputEst:
    """
    Monte Carlo Estimator for LLM Memory and Scheduling Slack.
    
    This class predicts future system state by simulating thousands of potential 
    request completion trajectories in parallel using vectorized NumPy operations.
    
    Key Physical Model: "The Sawtooth"
        - Memory grows linearly (+1 token per step) for all active requests.
        - Memory drops discretely (instant free) when a request finishes.
        - Therefore, local maxima for memory usage ALWAYS occur exactly at 
          the moment before a request finishes.
          
    Key Statistical Model: "Conditional Sampling"
        - Requests that have already generated many tokens are statistically likely 
          to continue generating more (heavy-tailed distributions).
        - We sample from the tail of the distribution: P(Total | Total > Current).
    """

    def __init__(
        self,
        window_size_batches: int = 256,
        bins_per_decade: int = 10,
        predictor: Optional[Any] = None
    ):
        """
        Initialize the Monte Carlo estimator with adaptive output length tracking.

        Args:
            window_size_batches: EMA window size for distribution tracking.
            bins_per_decade: Log-bin resolution for the histogram.
            predictor: Optional cycle time predictor (required for Slack).
        """
        # Construct RequestOutputLengthProfile internally for adaptive tracking
        self.output_length_profile = RequestOutputLengthProfile(
            window_size_batches=window_size_batches,
            bins_per_decade=bins_per_decade
        )

        # Initialize with default distribution (will adapt as observations arrive)
        edges, probs = self.output_length_profile.get_distribution()
        self.decode_bin_edges = edges
        self.cdf_values = self._build_cdf(probs)

        self.predictor = predictor

    def _build_cdf(self, probs: np.ndarray) -> np.ndarray:
        """Build CDF from probability mass function."""
        cdf_values = np.concatenate(([0.0], np.cumsum(probs)))
        cdf_values[-1] = 1.0  # Ensure numerical stability at the tail
        return cdf_values

    def submit_decode_length_observation(
        self, decode_lengths: Union[float, np.ndarray, List[float]]
    ) -> None:
        """
        Update the output length profile with newly observed decode lengths.
        Call this after each batch completes to keep predictions current.
        """
        # Add observations to the profile (uses EMA internally)
        self.output_length_profile.add(decode_lengths)

        # Refresh the CDF with updated distribution
        edges, probs = self.output_length_profile.get_distribution()
        self.decode_bin_edges = edges
        self.cdf_values = self._build_cdf(probs)

    def sample_trajectories(
        self,
        prefill_tokens: np.ndarray,
        current_decode_tokens: np.ndarray,
        n_simulations: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates N parallel futures for the current batch of requests.

        Logic:
            1. Calculate F(current) for each request.
            2. Sample u uniform in [F(current), 1.0].
            3. Invert CDF to find Total Length.
        """
        prefill_tokens = np.array(prefill_tokens, dtype=float)
        current_decode_tokens = np.array(current_decode_tokens, dtype=float)

        n_requests = len(prefill_tokens)
        if n_requests == 0:
            return np.zeros((n_simulations, 0)), np.zeros((n_simulations, 0))

        current_tokens = prefill_tokens + current_decode_tokens

        # --- Conditional Sampling ---
        
        # 1. Determine current percentile: P(L <= current)
        current_cdfs = np.interp(current_decode_tokens, self.decode_bin_edges, self.cdf_values)

        # 2. Sample from the remaining tail
        u_random = np.random.random((n_simulations, n_requests))
        
        # Scale u to be within [F(current), 1.0]
        target_cdfs = current_cdfs + u_random * (1.0 - current_cdfs)

        # 3. Inverse CDF Transform
        decode_total_samples = np.interp(
            target_cdfs.ravel(),
            self.cdf_values,
            self.decode_bin_edges
        ).reshape(n_simulations, n_requests)

        # Numerical safety: Clamp to ensure remaining is never negative
        decode_total_samples = np.maximum(decode_total_samples, current_decode_tokens)
        remaining_decode = decode_total_samples - current_decode_tokens

        # Broadcast current totals to (N_sim, N_req) for later matrix math
        current_tokens_matrix = np.broadcast_to(current_tokens, (n_simulations, n_requests))

        return remaining_decode, current_tokens_matrix

    def calculate_memory_matrix(
        self,
        remaining_decode: np.ndarray,
        current_tokens: np.ndarray,
        extract_periods: bool = False
    ) -> Tuple:
        """
        Core Vectorized Calculation Engine.

        Calculates memory usage at every "drop event" and extracts workload intervals.

        Optimization (Implicit pool_size = -1):
            - Treats time between any two completion events as one interval.
            - Computes all intervals for all simulations in one vectorized pass.
            - Returns flat NumPy arrays for optimal downstream processing.

        Returns:
            memory_matrix: (S, N) Memory usage just before each request finishes.
            (If extract_periods=True):
                batch_size_list: (M,) Batch sizes for valid intervals.
                kv_size_list: (M,) Avg KV sizes.
                cycle_counts_list: (M,) Duration (cycles) of intervals.
                sim_indices: (M,) Which simulation each interval belongs to.
        """
        n_simulations, n_requests = remaining_decode.shape
        
        if n_requests == 0:
            empty = np.zeros((n_simulations, 0))
            if extract_periods:
                return empty, np.array([]), np.array([]), np.array([]), np.array([])
            return (empty,)

        # 1. Sort Events by Time
        sort_indices = np.argsort(remaining_decode, axis=1)
        R_sorted = np.take_along_axis(remaining_decode, sort_indices, axis=1)
        C_sorted = np.take_along_axis(current_tokens, sort_indices, axis=1)

        # 2. Calculate Memory "Sawtooth" Profile
        # Active Count at event k: N, N-1, ... 1
        active_counts = np.arange(n_requests, 0, -1)

        # Base Memory: Sum(Starts) - Sum(Starts_Finished)
        total_C_sum = np.sum(current_tokens, axis=1, keepdims=True)
        cum_finished_C = np.cumsum(C_sorted, axis=1)
        
        # Shift cumsum right to get sum of PREVIOUSLY finished requests
        sum_active_C = np.empty_like(cum_finished_C)
        sum_active_C[:, 0] = total_C_sum[:, 0]
        sum_active_C[:, 1:] = total_C_sum[:, 0:1] - cum_finished_C[:, :-1]

        # Final Matrix: Memory state just before each request k finishes
        memory_matrix = sum_active_C + (active_counts * R_sorted)

        if not extract_periods:
            return (memory_matrix,)

        # 3. Vectorized Interval Extraction
        # Calculate duration of every interval (Prepend 0s for start at t=0)
        zeros = np.zeros((n_simulations, 1))
        R_padded = np.hstack([zeros, R_sorted])
        intervals = np.diff(R_padded, axis=1)  # Shape (S, N)

        # Convert float time to integer cycles (ceiling)
        n_cycles_mat = np.ceil(intervals).astype(int)

        # Filter: Only intervals where time actually passed (> 0 cycles)
        mask = n_cycles_mat > 0

        # 4. Calculate Interval Properties via Broadcasting
        
        # Batch Size: During interval k, batch size is N-k
        batch_sizes_mat = np.tile(np.arange(n_requests, 0, -1), (n_simulations, 1))
        
        # Avg KV Size: (Start + End) / 2
        mem_end = memory_matrix
        mem_start = np.hstack([total_C_sum, memory_matrix[:, :-1]])
        avg_kv_mat = ((mem_start + mem_end) // 2).astype(int)

        # 5. Flatten using Mask (Zero-Copy where possible)
        # We return NumPy arrays directly to keep data on C-side for Slack Calc
        batch_size_arr = batch_sizes_mat[mask]
        kv_size_arr = avg_kv_mat[mask]
        cycle_counts_arr = n_cycles_mat[mask]

        # 6. Create Simulation Index Map
        # sim_indices[i, j] = i. We mask this to know which sim an interval belongs to.
        sim_indices_mat = np.broadcast_to(np.arange(n_simulations)[:, None], (n_simulations, n_requests))
        valid_sims = sim_indices_mat[mask]

        return memory_matrix, batch_size_arr, kv_size_arr, cycle_counts_arr, valid_sims

    def calculate_cycle_times(
        self,
        batch_sizes: Union[List[int], np.ndarray],
        avg_kv_sizes: Union[List[int], np.ndarray],
        batch_step: int = 32,
        kv_step: int = 1024,
        return_stats: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Batch queries the predictor using GRID QUANTIZATION (Pooling).
        
        Optimizes predictor calls by:
        1. Quantizing (Batch, KV) pairs to a grid (e.g., 32, 1024).
        2. Bit-packing (batch, kv) into int64 so np.unique runs on 1D keys.
        3. Predicting only for unique points.
        4. Mapping results back to original array.
        If `return_stats` is True, returns per-cycle times alongside a dict
        describing the quantization/compression (and timings) that occurred.
        """
        if self.predictor is None:
            raise ValueError("Predictor is required for slack calculation")

        total_start = time.perf_counter()
        timings = {} if return_stats else None

        def _mark(name: str, start_time: float):
            if timings is not None:
                timings[name] = (time.perf_counter() - start_time) * 1000.0
        
        # Ensure we work with numpy arrays (int64 for bit packing safety)
        b_arr = np.array(batch_sizes, dtype=np.int64)
        k_arr = np.array(avg_kv_sizes, dtype=np.int64)
        
        if b_arr.size == 0:
            empty = np.array([])
            if return_stats:
                _mark("total", total_start)
                return empty, {
                    "original_queries": 0,
                    "unique_queries": 0,
                    "compression_ratio": 0.0,
                    "mean_batch": 0.0,
                    "max_batch": 0.0,
                    "timings_ms": timings if timings is not None else {},
                }
            return empty

        # 1. Quantize (Ceiling Strategy for Safety)
        # E.g., batch 1..32 -> 32
        quant_start = time.perf_counter()
        b_quant = (np.ceil(b_arr / batch_step) * batch_step).astype(np.int64)
        k_quant = (np.ceil(k_arr / kv_step) * kv_step).astype(np.int64)
        _mark("quantize", quant_start)

        # 2. Identify Unique Grid Points (Deduplication via bit packing)
        dedup_start = time.perf_counter()

        # Pack into a single int64: [batch (high 32) | kv (low 32)]
        packed_keys = (b_quant << 32) | k_quant

        # unique_packed: reduced workload; inverse_indices maps back
        unique_packed, inverse_indices = np.unique(packed_keys, return_inverse=True)
        _mark("dedup", dedup_start)

        # 3. Predict for Unique Points Only
        predict_start = time.perf_counter()
        unique_batches = (unique_packed >> 32).tolist()
        unique_kvs = (unique_packed & 0xFFFFFFFF).tolist()
        
        prefill_chunk_pairs_list = [[] for _ in range(len(unique_batches))]
        modes_list = ["DECODE"] * len(unique_batches)

        # Expensive Predictor Call (Reduced by ~10-100x)
        unique_predictions = self.predictor.predict_batch(
            batch_size_tokens_list=unique_batches,
            prefill_chunk_pairs_list=prefill_chunk_pairs_list,
            kv_tokens_used_list=unique_kvs,
            modes=modes_list
        )
        unique_predictions_arr = np.array(unique_predictions)
        _mark("predict", predict_start)

        # 4. Reconstruct Full Array via Broadcasting
        broadcast_start = time.perf_counter()
        per_cycle_times = unique_predictions_arr[inverse_indices]
        _mark("broadcast", broadcast_start)

        if not return_stats:
            return per_cycle_times

        unique_count = len(unique_packed)
        original_queries = int(b_quant.size)
        stats = {
            "original_queries": original_queries,
            "unique_queries": int(unique_count),
            "compression_ratio": float(original_queries / unique_count) if unique_count > 0 else 0.0,
            "mean_batch": float(b_arr.mean()),
            "max_batch": float(b_arr.max()),
        }
        _mark("total", total_start)
        stats["timings_ms"] = timings if timings is not None else {}
        return per_cycle_times, stats

    def calculate_slack(
        self,
        per_cycle_times: np.ndarray,
        cycle_counts: np.ndarray,
        sim_indices: np.ndarray,
        n_simulations: int,
        tpot: float
    ) -> float:
        """
        Vectorized Slack Calculation using `reduceat`.
        
        Optimization:
            1. Calculates global cumulative metrics for the entire flattened timeline.
            2. Uses `reduceat` to efficiently find the minimum slack per simulation chunk.
            3. Subtracts offsets to isolate simulations from each other.
        
        Complexity: O(Total_Intervals) in C (vs Python Loop).
        """
        if len(per_cycle_times) == 0:
            return 0.0

        # 1. Compute Global Cumulative State
        # Treat all simulations as one continuous sequence
        durations = per_cycle_times * cycle_counts
        cum_durations = np.cumsum(durations)
        cum_cycles = np.cumsum(cycle_counts)

        # 2. Identify Simulation Boundaries
        # sim_indices is sorted (0,0,0, 1,1, ...). Find where it changes.
        # change_mask is True at index 0 and wherever sim_id changes.
        change_mask = np.concatenate(([True], sim_indices[1:] != sim_indices[:-1]))
        start_indices = np.flatnonzero(change_mask)
        unique_sims = sim_indices[start_indices]

        # 3. Calculate Simulation Offsets
        # We need the cumulative values *just before* each simulation started to zero-base them.
        # Shift global arrays right by 1 (insert 0 at start)
        padded_dur = np.insert(cum_durations, 0, 0.0)
        offsets_dur = padded_dur[start_indices]

        padded_cyc = np.insert(cum_cycles, 0, 0)
        offsets_cyc = padded_cyc[start_indices]

        # 4. Calculate Global Slack Curve
        global_slack = (cum_cycles * tpot) - cum_durations

        # 5. Find Minimum per Group (The Optimization)
        # reduceat finds min in ranges [start[i] : start[i+1]]
        min_global_slack = np.minimum.reduceat(global_slack, start_indices)

        # 6. Convert to Local Slack
        # LocalSlack = GlobalSlack - OffsetSlack
        # OffsetSlack = (OffsetCycles * TPOT) - OffsetDuration
        offset_slack = (offsets_cyc * tpot) - offsets_dur
        min_local_slack = min_global_slack - offset_slack

        # 7. Handle t=0 Constraint
        # Slack starts at 0. If min_local > 0, it means we never dipped below schedule.
        # We clamp to 0.0 because "ahead of schedule" implies 0 risk.
        final_mins = np.minimum(0.0, min_local_slack)

        # 8. Map back to full simulation list
        # (Handles simulations that might have been filtered out due to empty intervals)
        results = np.zeros(n_simulations)
        results[unique_sims] = final_mins

        return float(np.mean(results))

    def estimate_peak_and_slack_with_ground_truth(
        self,
        prefill_tokens: np.ndarray,
        current_decode_tokens: np.ndarray,
        ground_truth_remaining_decode: np.ndarray,
        tpot: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Deterministic calculation using known remaining lengths.
        """
        prefill_tokens = np.array(prefill_tokens, dtype=float)
        current_decode_tokens = np.array(current_decode_tokens, dtype=float)
        ground_truth_remaining_decode = np.array(ground_truth_remaining_decode, dtype=float)

        if len(prefill_tokens) == 0:
            return 0.0, 0.0

        # reshape to (1, N) for the vectorized engine
        remaining_decode = ground_truth_remaining_decode.reshape(1, -1)
        current_tokens = (prefill_tokens + current_decode_tokens).reshape(1, -1)

        calc_slack = (self.predictor is not None) and (tpot is not None)

        result = self.calculate_memory_matrix(
            remaining_decode,
            current_tokens,
            extract_periods=calc_slack
        )

        if calc_slack:
            memory_matrix, batch_sizes, kv_sizes, cycle_counts, sim_indices = result
        else:
            (memory_matrix,) = result

        if memory_matrix.shape[1] == 0:
            return 0.0, 0.0

        # Exact Peak
        peak = float(np.max(np.maximum(memory_matrix, np.sum(current_tokens, axis=1))))

        if not calc_slack or len(batch_sizes) == 0:
            return peak, 0.0

        # Exact Slack
        per_cycle_times = self.calculate_cycle_times(batch_sizes, kv_sizes, batch_step=32, kv_step=1024)
        minimal_slack = self.calculate_slack(
            per_cycle_times,
            cycle_counts,
            sim_indices,
            n_simulations=1,
            tpot=tpot
        )

        return peak, minimal_slack

    def estimate_peak_and_slack(
        self,
        prefill_tokens: np.ndarray,
        current_decode_tokens: np.ndarray,
        n_simulations: int = 100,
        tpot: Optional[float] = None,
        stats: Optional[Dict] = None
    ) -> Tuple[float, float]:
        """
        Main Entry Point: Estimates Peak Memory and Slack.

        Args:
            prefill_tokens: Fixed context lengths.
            current_decode_tokens: Tokens generated so far.
            n_simulations: Accuracy vs Speed tradeoff parameter.
            tpot: Target time per output token (ms).
            stats: Optional dict to populate with timing metrics.
        """
        step_times = stats.setdefault("step_times_ms", {}) if stats is not None else None
        
        def _record_step(name: str, start_time: float):
            if step_times is not None:
                step_times[name] = (time.perf_counter() - start_time) * 1000.0

        total_start = time.perf_counter()
        if len(prefill_tokens) == 0:
            return 0.0, 0.0

        # 1. Sample Futures
        sample_start = time.perf_counter()
        remaining_decode, current_tokens = self.sample_trajectories(
            prefill_tokens, current_decode_tokens, n_simulations
        )
        _record_step("sample", sample_start)

        # 2. Determine if we need Slack calculation
        calc_slack = (self.predictor is not None) and (tpot is not None)

        # 3. Calculate Memory Matrix (Optimized Single Pass)
        memory_start = time.perf_counter()
        result = self.calculate_memory_matrix(
            remaining_decode,
            current_tokens,
            extract_periods=calc_slack
        )
        _record_step("memory_matrix", memory_start)

        if calc_slack:
            memory_matrix, batch_sizes, kv_sizes, cycle_counts, sim_indices = result
        else:
            (memory_matrix,) = result

        if memory_matrix.shape[1] == 0:
            return 0.0, 0.0

        # 4. Estimate Peak Memory
        # Peak is max over time (axis 1), then mean over simulations (axis 0)
        peak_per_sim = np.max(memory_matrix, axis=1)

        # Correction: Check against current state (t=0)
        current_total_size = np.sum(current_tokens, axis=1)
        peak_per_sim = np.maximum(peak_per_sim, current_total_size)

        peak_kv_cache = float(np.mean(peak_per_sim))

        if not calc_slack or len(batch_sizes) == 0:
            if stats is not None:
                stats["predictor_batch_size_stats"] = None
                _record_step("total", total_start)
            return peak_kv_cache, 0.0

        # 5. Estimate Slack (Vectorized & Quantized)
        predictor_start = time.perf_counter()
        
        # Calculate Cycle Times using Grid Quantization
        cycle_result = self.calculate_cycle_times(
            batch_sizes, kv_sizes, batch_step=32, kv_step=1024, return_stats=stats is not None
        )
        if stats is not None:
            per_cycle_times, predictor_stats = cycle_result
            stats["predictor_batch_size_stats"] = predictor_stats
        else:
            per_cycle_times = cycle_result
        _record_step("predictor", predictor_start)
        
        slack_start = time.perf_counter()
        minimal_slack = self.calculate_slack(
            per_cycle_times,
            cycle_counts,
            sim_indices,
            n_simulations=memory_matrix.shape[0],
            tpot=tpot
        )
        _record_step("slack", slack_start)

        if stats is not None:
            _record_step("total", total_start)

        return peak_kv_cache, minimal_slack
