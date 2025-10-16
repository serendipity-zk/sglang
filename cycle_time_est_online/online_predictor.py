#!/usr/bin/env python3
"""
Online Linear Cycle Time Estimator

- Online-only learning: predict first, then update model with the new measurement.
- Linear model trained incrementally using Recursive Least Squares (RLS).
- Feature engineering mirrors cycle_time_est/predictor.py's PredictionInput.to_features.
- Online normalization via Welford running mean/std for stability.
- Bounded memory: no full sample storage; optional limited history (default off).
"""

from dataclasses import dataclass
from collections import deque
from typing import List, Optional, Tuple, Dict, Sequence

import numpy as np


@dataclass
class PredictionInput:
    """Input features for cycle time prediction."""
    batch_size_tokens: int
    prefill_chunk_pairs: List[List[int]]  # List of [current_chunk, cumulative_prefill]
    kv_tokens_used: int

    def to_features(self) -> np.ndarray:
        """
        Convert to feature vector compatible with cycle_time_est/predictor.py.

        Features (19 total, index 0..18):
          0: batch_size_tokens
          1: kv_tokens_used
          2: num_prefill_requests
          3: total_prefill_chunks (sum current)
          4: total_cumulative_prefill (sum cumulative)
          5: avg_chunk_size
          6: max_chunk_size
          7: avg_cumulative_size
          8: sum_chunk_history_product (sum current*history)
          9: max_chunk_history_product
         10: avg_chunk_history_product
         11: sum_chunk_squared
         12: sum_cumulative_squared
         13: min_chunk_size
         14: std_chunk_size
         15: kv_to_batch_ratio
         16: batch_squared
         17: kv_squared
         18: batch_times_kv
        """
        features: List[float] = []

        # Basic features
        batch_size = float(self.batch_size_tokens)
        kv_used = float(self.kv_tokens_used)
        features.append(batch_size)
        features.append(kv_used)

        # Prefill chunk features
        num_prefill = len(self.prefill_chunk_pairs)
        features.append(float(num_prefill))

        if num_prefill > 0:
            chunks = np.array([pair[0] for pair in self.prefill_chunk_pairs], dtype=np.float32)
            cumulative = np.array([pair[1] for pair in self.prefill_chunk_pairs], dtype=np.float32)

            features.append(float(np.sum(chunks)))  # total_prefill_chunks
            features.append(float(np.sum(cumulative)))  # total_cumulative_prefill
            features.append(float(np.mean(chunks)))  # avg_chunk_size
            features.append(float(np.max(chunks)))  # max_chunk_size
            features.append(float(np.mean(cumulative)))  # avg_cumulative_size

            # Product features
            chunk_history_products = chunks * cumulative
            features.append(float(np.sum(chunk_history_products)))  # sum_chunk_history_product
            features.append(float(np.max(chunk_history_products)))  # max_chunk_history_product
            features.append(float(np.mean(chunk_history_products)))  # avg_chunk_history_product

            # Squared features
            features.append(float(np.sum(chunks ** 2)))  # sum_chunk_squared
            features.append(float(np.sum(cumulative ** 2)))  # sum_cumulative_squared

            # Additional stats
            features.append(float(np.min(chunks)))  # min_chunk_size
            features.append(float(np.std(chunks)))  # std_chunk_size
        else:
            # No prefill → zeros for the 12 prefill-related features
            features.extend([0.0] * 12)

        # Derived ratios and interactions
        kv_to_batch_ratio = kv_used / batch_size if batch_size > 0 else 0.0
        features.append(kv_to_batch_ratio)
        features.append(batch_size ** 2)  # batch_squared
        features.append(kv_used ** 2)  # kv_squared
        features.append(batch_size * kv_used)  # batch_times_kv

        return np.array(features, dtype=np.float64)


class OnlineLinearCycleTime:
    """
    Online-only linear cycle time estimator with RLS updates.

    - Predicts first, then updates average deviation and model state.
    - Maintains running mean/std for feature normalization.
    - Uses Recursive Least Squares (with optional forgetting) for weight updates.

    Memory notes:
      - Does not store per-sample data by default.
      - Keeps a small error ring buffer for last-window logging.
      - Optional history storage can be enabled but is disabled by default.
    """

    def __init__(
        self,
        *,
        log_every: int = 1000,
        forgetting: float = 1.0,
        init_cov: float = 1e3,
        eps_std: float = 1e-6,
        store_history: bool = False,
        max_history: int = 100000,
        feature_indices: Optional[Sequence[int]] = None,
        feature_preset: Optional[str] = None,
    ) -> None:
        if forgetting <= 0 or forgetting > 1.0:
            raise ValueError("forgetting must be in (0, 1], e.g. 1.0 for no forgetting")
        if log_every <= 0:
            raise ValueError("log_every must be positive")
        if max_history <= 0:
            raise ValueError("max_history must be positive")

        self.log_every = int(log_every)
        self.forgetting = float(forgetting)
        self.init_cov = float(init_cov)
        self.eps_std = float(eps_std)
        self.store_history = bool(store_history)
        self.max_history = int(max_history)
        # Feature selection configuration
        self.feature_indices_cfg: Optional[List[int]] = list(feature_indices) if feature_indices is not None else None
        self.feature_preset: Optional[str] = str(feature_preset) if feature_preset is not None else None
        self._feat_idx_runtime: Optional[np.ndarray] = None  # resolved after first sample

        # Model parameters (initialized on first sample)
        self._w: Optional[np.ndarray] = None  # shape (d+1,), includes bias
        self._P: Optional[np.ndarray] = None  # shape (d+1, d+1)

        # Online normalization via Welford per-feature
        self._count: int = 0
        self._mean: Optional[np.ndarray] = None
        self._M2: Optional[np.ndarray] = None

        # Error tracking
        self._n_seen: int = 0
        self._sum_abs_err: float = 0.0
        self._window_errs: deque = deque(maxlen=self.log_every)

        # Optional bounded sample history (off by default)
        self._history: Optional[deque] = deque(maxlen=self.max_history) if self.store_history else None

    # ---------------------- Feature selection ------------------
    PRESET_FEATURES: Dict[str, List[int]] = {
        # All features as produced by PredictionInput.to_features()
        "all": None,  # special-case meaning no selection
        # A compact subset emphasizing core scale and quadratic terms
        "basic": [0, 1, 2, 3, 4, 15, 16, 17, 18],
        # Subset used by the hybrid in offline code (5-D here)
        "hybrid5": [0, 1, 8, 16, 17],
        # Only interaction-like term capturing attention cost proxy
        "prod_only": [8],
        "key": [0,1,8],
    }

    def _resolve_feature_indices(self, d_total: int) -> Optional[np.ndarray]:
        if self._feat_idx_runtime is not None:
            return self._feat_idx_runtime
        idx: Optional[List[int]] = None
        if self.feature_indices_cfg is not None:
            idx = list(self.feature_indices_cfg)
        elif self.feature_preset is not None:
            preset = self.feature_preset.lower()
            if preset in self.PRESET_FEATURES and self.PRESET_FEATURES[preset] is not None:
                idx = list(self.PRESET_FEATURES[preset])
            else:
                # unknown preset or 'all' → no selection
                idx = None
        if idx is None:
            self._feat_idx_runtime = None
            return None
        # Validate and deduplicate while preserving order
        seen = set()
        clean = []
        for i in idx:
            if not isinstance(i, (int, np.integer)):
                raise ValueError("feature_indices must be integers")
            ii = int(i)
            if 0 <= ii < d_total and ii not in seen:
                seen.add(ii)
                clean.append(ii)
        if not clean:
            self._feat_idx_runtime = None
            return None
        self._feat_idx_runtime = np.array(clean, dtype=np.int64)
        return self._feat_idx_runtime

    def _select_features(self, x: np.ndarray) -> np.ndarray:
        idx = self._resolve_feature_indices(x.shape[0])
        if idx is None:
            return x
        return x[idx]

    # ---------------------- Feature utils ----------------------
    def _ensure_stats(self, d: int) -> None:
        if self._mean is None or self._M2 is None:
            self._mean = np.zeros(d, dtype=np.float64)
            self._M2 = np.zeros(d, dtype=np.float64)
            self._count = 0

    def _current_std(self) -> np.ndarray:
        if self._count <= 1:
            # Not enough samples to compute variance; fallback to ones
            return np.ones_like(self._mean, dtype=np.float64)
        var = self._M2 / (self._count - 1)
        std = np.sqrt(np.maximum(var, self.eps_std))
        std[std < self.eps_std] = self.eps_std
        return std

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        # Use current mean/std (do not update stats here)
        self._ensure_stats(x.shape[0])
        mean = self._mean
        std = self._current_std()
        return (x - mean) / std

    def _update_stats(self, x: np.ndarray) -> None:
        self._ensure_stats(x.shape[0])
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._M2 += delta * delta2

    # ---------------------- Model utils ------------------------
    def _ensure_model(self, d: int) -> None:
        if self._w is None or self._P is None:
            # d features + 1 bias
            self._w = np.zeros(d + 1, dtype=np.float64)
            self._P = self.init_cov * np.eye(d + 1, dtype=np.float64)

    def _rls_predict_raw(self, z: np.ndarray) -> float:
        # z is normalized features (shape (d,)), append bias
        self._ensure_model(z.shape[0])
        phi = np.append(z, 1.0)
        y_hat = float(phi @ self._w)
        return max(0.0, y_hat)

    def _rls_update(self, z: np.ndarray, y: float) -> None:
        # Recursive Least Squares update with forgetting factor
        self._ensure_model(z.shape[0])
        phi = np.append(z, 1.0)  # (d+1,)
        P = self._P
        w = self._w
        lam = self.forgetting

        # Gain
        # k = P phi / (lam + phi^T P phi)
        P_phi = P @ phi
        denom = lam + float(phi.T @ P_phi)
        k = P_phi / denom

        # Update weights: w <- w + k (y - phi^T w)
        err = y - float(phi.T @ w)
        w_new = w + k * err

        # Update covariance: P <- (P - k phi^T P) / lam
        P_new = (P - (np.outer(k, phi) @ P)) / lam

        self._w = w_new
        self._P = P_new

    # --------------------------- API ---------------------------
    def predict(self, batch_size_tokens: int, prefill_chunk_pairs: List[List[int]], kv_tokens_used: int) -> float:
        inp = PredictionInput(batch_size_tokens, prefill_chunk_pairs, kv_tokens_used)
        x = inp.to_features()
        x = self._select_features(x)
        # If model not initialized yet, prediction is 0.0
        if self._w is None:
            return 0.0
        z = self._normalize(x)
        return self._rls_predict_raw(z)

    def submit(
        self,
        batch_size_tokens: int,
        prefill_chunk_pairs: List[List[int]],
        kv_tokens_used: int,
        iteration_time_ms: float,
    ) -> Tuple[float, float]:
        """
        Predict first, then update metrics and model online.

        Returns: (prediction_ms_before_update, absolute_error_ms)
        """
        # Features
        inp = PredictionInput(batch_size_tokens, prefill_chunk_pairs, kv_tokens_used)
        x = inp.to_features()
        x = self._select_features(x)

        # Predict BEFORE any updates
        if self._w is None:
            y_pred = 0.0
        else:
            z_pred = self._normalize(x)
            y_pred = self._rls_predict_raw(z_pred)

        # Track deviation
        abs_err = abs(float(iteration_time_ms) - y_pred)
        self._n_seen += 1
        self._sum_abs_err += abs_err
        self._window_errs.append(abs_err)

        # Logging every N
        if self._n_seen % self.log_every == 0:
            window_avg = float(np.mean(self._window_errs)) if len(self._window_errs) > 0 else 0.0
            cum_avg = self._sum_abs_err / self._n_seen
            # Percentiles over the recent window
            if len(self._window_errs) > 0:
                window_vals = np.fromiter(self._window_errs, dtype=np.float64)
                p90 = float(np.percentile(window_vals, 90))
                p99 = float(np.percentile(window_vals, 99))
            else:
                p90 = 0.0
                p99 = 0.0
            print(
                f"[OnlineLinearCycleTime] N={self._n_seen} | "
                f"avg_abs_err_window({self.log_every})={window_avg:.2f} ms | "
                f"p90_window={p90:.2f} ms | p99_window={p99:.2f} ms | "
                f"avg_abs_err_cum={cum_avg:.2f} ms"
            )

        # Optional bounded history (OFF by default)
        if self._history is not None:
            self._history.append((x, float(iteration_time_ms)))

        # Update model: first update RLS using current normalization, then update stats
        # Note: normalization for this sample uses pre-update stats by design.
        z = self._normalize(x)
        self._rls_update(z, float(iteration_time_ms))

        # Update feature stats AFTER model update to keep predict-first semantics
        self._update_stats(x)

        return y_pred, abs_err

    # ------------------------ Introspection --------------------
    @property
    def n_seen(self) -> int:
        return self._n_seen

    @property
    def mean_abs_err_cum(self) -> float:
        return self._sum_abs_err / self._n_seen if self._n_seen > 0 else 0.0

    def get_weights(self) -> Optional[np.ndarray]:
        """Return current weight vector (including bias) or None if uninitialized."""
        return None if self._w is None else self._w.copy()

    def get_norm_stats(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """Return (mean, std, count) used for normalization."""
        if self._mean is None:
            return None, None, 0
        return self._mean.copy(), self._current_std().copy(), self._count

    def reset(self) -> None:
        """Reset model and statistics while keeping configuration."""
        self._w = None
        self._P = None
        self._count = 0
        self._mean = None
        self._M2 = None
        self._n_seen = 0
        self._sum_abs_err = 0.0
        self._window_errs.clear()
        if self._history is not None:
            self._history.clear()
        self._feat_idx_runtime = None

    def update_feature_selection(self, *, feature_indices: Optional[Sequence[int]] = None, feature_preset: Optional[str] = None, reset: bool = True) -> None:
        """Change the feature subset. Optionally reset the model/statistics to avoid shape mismatch.

        If reset=False and the new subset changes dimensionality, an error will be raised upon next use.
        """
        self.feature_indices_cfg = list(feature_indices) if feature_indices is not None else None
        self.feature_preset = str(feature_preset) if feature_preset is not None else None
        self._feat_idx_runtime = None
        if reset:
            self.reset()
