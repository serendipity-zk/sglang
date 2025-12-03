"""
Thin Python wrapper around the Rust `knn_workload_predictor` bindings.

The API mirrors `mode_aware_predictor.ModeAwarePredictor` so it can be swapped in:
- predict(batch_size_tokens, prefill_chunk_pairs, kv_tokens_used, mode=None) -> float
- submit(batch_size_tokens, prefill_chunk_pairs, kv_tokens_used, actual_time_ms, mode=None) -> (prediction, abs_error)
- predict_batch(batch_size_tokens_list, prefill_chunk_pairs_list, kv_tokens_used_list, modes=None) -> List[float]

Mode parameter is accepted for compatibility but currently not used; the Rust side
chooses DECODE vs MIXED based on whether prefill_chunk_pairs is empty.

prefill_chunk_pairs should be a List[List[int]] of [chunk_len, cumulative_len] pairs,
matching the scheduler's native format.
"""
import csv
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    from knn_workload_predictor import KNNPredictor
except Exception as exc:  # pragma: no cover - import-time failure path
    raise ImportError(
        "knn_workload_predictor module not found. Install the wheel built by maturin."
    ) from exc


class RustModeAwarePredictor:
    """Rust-backed KNN predictor with API compatible with ModeAwarePredictor."""
    
    # Mark as multi-mode predictor for scheduler compatibility
    is_multimode: bool = True
    
    def __init__(
        self,
        grid_path: str = "/sgl-workspace/sglang/sglang_profile/mode_3d.json",
        k_neighbors: int = 10,
        max_history: int = 5000,
        csv_log_path: Optional[str] = None,
        log_every: int = 100,
    ):
        self.inner = KNNPredictor(k_neighbors, max_history)
        try:
            self.inner.load_grid(grid_path)
            self._grid_loaded = True
        except Exception:
            # Keep going; KNN will still function and fall back to InsufficientData until fitted
            self._grid_loaded = False

        # Logging setup (roughly mirrors mode_aware_predictor)
        if csv_log_path is None:
            csv_log_path = f"rust_mode_predictor_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_log_path = csv_log_path
        self.csv_file = open(self.csv_log_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "timestamp",
                "operation",
                "mode",
                "batch_size_tokens",
                "prefill_chunk_pairs",
                "kv_tokens_used",
                "prediction_ms",
                "actual_time_ms",
                "abs_error_ms",
                "status",
            ]
        )
        self.csv_file.flush()
        self.log_every = max(1, log_every)
        self._n_seen = 0
        self._window_errs: List[float] = []

    def predict(
        self,
        batch_size_tokens: int,
        prefill_chunk_pairs: List[List[int]],
        kv_tokens_used: int,
        mode: Optional[str] = None,
    ) -> float:
        preds = self.predict_batch(
            [batch_size_tokens],
            [prefill_chunk_pairs],
            [kv_tokens_used],
            modes=[mode] if mode is not None else None,
        )
        return preds[0]

    def submit(
        self,
        batch_size_tokens: int,
        prefill_chunk_pairs: List[List[int]],
        kv_tokens_used: int,
        iteration_time_ms: float,
        mode: Optional[str] = None,
    ) -> Tuple[float, float]:
        prediction = self.predict(
            batch_size_tokens, prefill_chunk_pairs, kv_tokens_used, mode=mode
        )
        # Convert List[List[int]] to List[[int, int]] for Rust
        pairs_for_rust = [[p[0], p[1]] for p in prefill_chunk_pairs] if prefill_chunk_pairs else []
        self.inner.update(batch_size_tokens, pairs_for_rust, kv_tokens_used, iteration_time_ms)
        self._log(
            op="submit",
            mode=mode,
            batch=batch_size_tokens,
            prefill=prefill_chunk_pairs,
            kv=kv_tokens_used,
            prediction=prediction,
            actual=iteration_time_ms,
            status="UPDATE",
        )
        return prediction, abs(prediction - iteration_time_ms)

    def predict_batch(
        self,
        batch_size_tokens_list: Sequence[int],
        prefill_chunk_pairs_list: Sequence[List[List[int]]],
        kv_tokens_used_list: Sequence[int],
        modes: Optional[Iterable[Optional[str]]] = None,
    ) -> List[float]:
        # modes are accepted for compatibility but ignored by the Rust core
        # Convert List[List[int]] to List[[int, int]] for each sample
        prefills_for_rust = [
            [[p[0], p[1]] for p in pairs] if pairs else []
            for pairs in prefill_chunk_pairs_list
        ]
        predictions_and_status = self.inner.predict_batch(
            list(batch_size_tokens_list),
            prefills_for_rust,
            list(kv_tokens_used_list),
        )
        preds = [pred for pred, status in predictions_and_status]

        # # Log (without actuals)
        # for (b, p, k, (pred, status), mode) in zip(
        #     batch_size_tokens_list,
        #     prefill_chunk_pairs_list,
        #     kv_tokens_used_list,
        #     predictions_and_status,
        #     modes if modes is not None else [None] * len(batch_size_tokens_list),
        # ):
        #     self._log(
        #         op="predict",
        #         mode=mode,
        #         batch=b,
        #         prefill=p,
        #         kv=k,
        #         prediction=pred,
        #         actual=None,
        #         status=status,
        #     )

        return preds

    def _log(
        self,
        op: str,
        mode: Optional[str],
        batch: int,
        prefill: List[List[int]],
        kv: int,
        prediction: float,
        actual: Optional[float],
        status: str,
    ) -> None:
        ts = datetime.now().isoformat()
        abs_err = None if actual is None else abs(prediction - actual)
        self.csv_writer.writerow(
            [
                ts,
                op,
                mode or "",
                batch,
                prefill,
                kv,
                prediction,
                actual if actual is not None else "",
                abs_err if abs_err is not None else "",
                status,
            ]
        )
        self.csv_file.flush()

        if abs_err is not None:
            self._n_seen += 1
            self._window_errs.append(abs_err)
            if len(self._window_errs) > self.log_every:
                self._window_errs.pop(0)
            if self._n_seen % self.log_every == 0:
                window_avg = sum(self._window_errs) / len(self._window_errs)
                print(
                    f"[rust-mode-predictor] n={self._n_seen}, "
                    f"last_abs_err={abs_err:.3f} ms, "
                    f"avg_abs_err_window({self.log_every})={window_avg:.3f} ms"
                )


__all__ = ["RustModeAwarePredictor"]
