#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mode-aware cycle time predictor with local bias correction.

Features:
- Multi-mode support (DECODE, EXTEND, MIXED) with separate grids per mode
- Offline 3D trilinear grid lookup (from grid3d.json)
- Online local bias-only correction using BiasLocalCorrector (per mode)
- Backward compatible with single-mode grids

Interface compatible with OnlineLinearCycleTime:
- predict(batch_size_tokens, prefill_chunk_pairs, kv_tokens_used, mode) -> float
- submit(batch_size_tokens, prefill_chunk_pairs, kv_tokens_used, iteration_time_ms, mode) -> (prediction, abs_error)
- predict_batch(batch_size_tokens_list, prefill_chunk_pairs_list, kv_tokens_used_list, modes) -> List[float]
"""

import json
from pathlib import Path
from typing import List, Tuple
import numpy as np


# ---------- Trilinear interpolation ----------
def trilinear_predict(x, y, z, X, Y, Z, G):
    """
    Trilinear interpolation on a 3D grid.

    Args:
        x, y, z: Query points (scalars or arrays)
        X, Y, Z: Knot arrays for each axis
        G: 3D grid array of shape (len(X), len(Y), len(Z))

    Returns:
        Interpolated values at (x, y, z)
    """
    ix = np.searchsorted(X, x, side="right") - 1
    iy = np.searchsorted(Y, y, side="right") - 1
    iz = np.searchsorted(Z, z, side="right") - 1
    ix = np.clip(ix, 0, len(X) - 2)
    iy = np.clip(iy, 0, len(Y) - 2)
    iz = np.clip(iz, 0, len(Z) - 2)

    x0, x1 = X[ix], X[ix + 1]
    y0, y1 = Y[iy], Y[iy + 1]
    z0, z1 = Z[iz], Z[iz + 1]

    tx = np.divide(x - x0, x1 - x0, out=np.zeros_like(x), where=(x1 > x0))
    ty = np.divide(y - y0, y1 - y0, out=np.zeros_like(y), where=(y1 > y0))
    tz = np.divide(z - z0, z1 - z0, out=np.zeros_like(z), where=(z1 > z0))

    g000 = G[ix, iy, iz]
    g100 = G[ix+1, iy, iz]
    g010 = G[ix, iy+1, iz]
    g110 = G[ix+1, iy+1, iz]
    g001 = G[ix, iy, iz+1]
    g101 = G[ix+1, iy, iz+1]
    g011 = G[ix, iy+1, iz+1]
    g111 = G[ix+1, iy+1, iz+1]

    return ((1-tx)*(1-ty)*(1-tz)*g000 + tx*(1-ty)*(1-tz)*g100 +
            (1-tx)*ty*(1-tz)*g010 + tx*ty*(1-tz)*g110 +
            (1-tx)*(1-ty)*tz*g001 + tx*(1-ty)*tz*g101 +
            (1-tx)*ty*tz*g011 + tx*ty*tz*g111)


# ---------- Online bias-only corrector ----------
class BiasLocalCorrector:
    """
    Online local bias correction using k-NN weighted averaging.

    Maintains a circular buffer of recent (x, y, z, residual) observations.
    For each prediction, computes a local bias estimate from nearby points
    using spatial and temporal weighting.
    """
    def __init__(self, buffer_size=10000, k=64, radius=0.30, bandwidth=0.20,
                 half_life=50.0, alpha=0.6, W0=4.0, max_correction=np.inf,
                 x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1):
        self.N, self.k, self.radius, self.h2 = int(buffer_size), int(k), radius, bandwidth**2
        self.tau, self.alpha, self.W0, self.max_corr = half_life / np.log(2.0), alpha, W0, max_correction
        self.xmin, self.xmax = x_min, x_max
        self.ymin, self.ymax = y_min, y_max
        self.zmin, self.zmax = z_min, z_max
        self.xspan = max(x_max - x_min, 1e-12)
        self.yspan = max(y_max - y_min, 1e-12)
        self.zspan = max(z_max - z_min, 1e-12)

        self.buf_x = np.empty(self.N, np.float32)
        self.buf_y = np.empty(self.N, np.float32)
        self.buf_z = np.empty(self.N, np.float32)
        self.buf_r = np.empty(self.N, np.float32)
        self.buf_t = np.empty(self.N, np.int32)
        self.size, self.head, self.t = 0, 0, 0

    def _norm(self, x, y, z):
        """Normalize coordinates to [0, 1] range."""
        return (x - self.xmin) / self.xspan, (y - self.ymin) / self.yspan, (z - self.zmin) / self.zspan

    def update(self, x, y, z, residual):
        """Add a new observation to the buffer."""
        xn, yn, zn = self._norm(x, y, z)
        self.buf_x[self.head] = xn
        self.buf_y[self.head] = yn
        self.buf_z[self.head] = zn
        self.buf_r[self.head] = residual
        self.buf_t[self.head] = self.t
        self.head = (self.head + 1) % self.N
        self.size = min(self.size + 1, self.N)
        self.t += 1

    def correction(self, x, y, z):
        """
        Compute local bias correction for point (x, y, z).

        Returns:
            (correction, num_neighbors): Correction value and number of neighbors used
        """
        if self.size == 0:
            return 0.0, 0

        xn, yn, zn = self._norm(x, y, z)
        Xb = self.buf_x[:self.size]
        Yb = self.buf_y[:self.size]
        Zb = self.buf_z[:self.size]
        Rb = self.buf_r[:self.size]
        Tb = self.buf_t[:self.size]

        # Find neighbors within radius
        d2 = (Xb - xn)**2 + (Yb - yn)**2 + (Zb - zn)**2
        mask = d2 <= self.radius**2
        if not np.any(mask):
            return 0.0, 0

        d2 = d2[mask]
        Rb = Rb[mask]
        Tb = Tb[mask]

        # Keep only k nearest
        if d2.size > self.k:
            idx = np.argpartition(d2, self.k)[:self.k]
            d2 = d2[idx]
            Rb = Rb[idx]
            Tb = Tb[idx]

        # Compute weights: spatial (Gaussian) * temporal (exponential decay)
        w = np.exp(-d2 / (2 * self.h2)) * np.exp(-(self.t - Tb) / self.tau)
        Wsum = float(w.sum())

        if Wsum <= 1e-9:
            return 0.0, len(Rb)

        # Local bias estimate
        local_bias = float(np.sum(w * Rb) / Wsum)

        # Adaptive alpha based on confidence (total weight)
        alpha_eff = self.alpha * (Wsum / (Wsum + self.W0))

        return float(np.clip(alpha_eff * local_bias, -self.max_corr, self.max_corr)), len(Rb)


# ---------- Mode-aware predictor with bias correction ----------
class ModeAwarePredictor:
    """
    Mode-aware cycle time predictor using offline 3D grids + online local bias correction.

    Supports multi-mode grids (DECODE, EXTEND, MIXED) with mode-specific predictions
    and bias correction. Provides the same interface as OnlineLinearCycleTime for
    drop-in replacement.
    """

    def __init__(
        self,
        grid_path: str,
        buffer_size: int = 10000,
        k: int = 64,
        radius: float = 0.30,
        bandwidth: float = 0.20,
        half_life: float = 50.0,
        alpha: float = 0.6,
        W0: float = 4.0,
        max_correction: float = np.inf,
        log_every: int = 100,
    ):
        """
        Args:
            grid_path: Path to grid3d.json file (single or multi-mode format)
            buffer_size: Size of circular buffer for bias correction (per mode)
            k: Number of nearest neighbors for local correction
            radius: Spatial radius for neighbor search (normalized)
            bandwidth: Gaussian kernel bandwidth for spatial weighting
            half_life: Temporal decay half-life for weighting
            alpha: Strength of bias correction (0-1)
            W0: Confidence threshold for adaptive alpha
            max_correction: Maximum absolute correction value
            log_every: Log statistics every N samples
        """
        # Load grid model
        self.grid_path = grid_path
        model = json.loads(Path(grid_path).read_text())

        # Detect format: multi-mode or single-mode
        self.is_multimode = "modes" in model

        if self.is_multimode:
            # Multi-mode format: load all grids
            self.grids = {}
            self.X_knots_dict = {}
            self.Y_knots_dict = {}
            self.Z_knots_dict = {}

            print(f"[ModeAwarePredictor] Loaded multi-mode grid from {grid_path}")
            print(f"  Available modes: {list(model['modes'].keys())}")

            for mode_name, mode_data in model["modes"].items():
                self.X_knots_dict[mode_name] = np.array(mode_data["knots"]["X_knots"])
                self.Y_knots_dict[mode_name] = np.array(mode_data["knots"]["Y_knots"])
                self.Z_knots_dict[mode_name] = np.array(mode_data["knots"]["Z_knots"])
                self.grids[mode_name] = np.array(mode_data["grid"])

                print(f"  Mode '{mode_name}': grid shape {self.grids[mode_name].shape}, "
                      f"X:[{self.X_knots_dict[mode_name].min():.0f}, {self.X_knots_dict[mode_name].max():.0f}], "
                      f"Y:[{self.Y_knots_dict[mode_name].min():.0f}, {self.Y_knots_dict[mode_name].max():.0f}], "
                      f"Z:[{self.Z_knots_dict[mode_name].min():.0f}, {self.Z_knots_dict[mode_name].max():.0f}]")

            # Compute global bounds for bias corrector initialization
            all_x_knots = np.concatenate([k for k in self.X_knots_dict.values()])
            all_y_knots = np.concatenate([k for k in self.Y_knots_dict.values()])
            all_z_knots = np.concatenate([k for k in self.Z_knots_dict.values()])

            x_min, x_max = all_x_knots.min(), all_x_knots.max()
            y_min, y_max = all_y_knots.min(), all_y_knots.max()
            z_min, z_max = all_z_knots.min(), all_z_knots.max()

            # Initialize separate bias corrector per mode
            self.bias_correctors = {}
            for mode_name in self.grids.keys():
                self.bias_correctors[mode_name] = BiasLocalCorrector(
                    buffer_size=buffer_size,
                    k=k,
                    radius=radius,
                    bandwidth=bandwidth,
                    half_life=half_life,
                    alpha=alpha,
                    W0=W0,
                    max_correction=max_correction,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    z_min=z_min,
                    z_max=z_max,
                )

        else:
            # Single-mode format (backward compatible)
            self.X_knots = np.array(model["knots"]["X_knots"])
            self.Y_knots = np.array(model["knots"]["Y_knots"])
            self.Z_knots = np.array(model["knots"]["Z_knots"])
            self.grid = np.array(model["grid"])

            print(f"[ModeAwarePredictor] Loaded single-mode grid from {grid_path}")
            print(f"  Grid shape: {self.grid.shape}")
            print(f"  X range: [{self.X_knots.min():.0f}, {self.X_knots.max():.0f}]")
            print(f"  Y range: [{self.Y_knots.min():.0f}, {self.Y_knots.max():.0f}]")
            print(f"  Z range: [{self.Z_knots.min():.0f}, {self.Z_knots.max():.0f}]")

            # Initialize single bias corrector
            self.bias_corrector = BiasLocalCorrector(
                buffer_size=buffer_size,
                k=k,
                radius=radius,
                bandwidth=bandwidth,
                half_life=half_life,
                alpha=alpha,
                W0=W0,
                max_correction=max_correction,
                x_min=self.X_knots.min(),
                x_max=self.X_knots.max(),
                y_min=self.Y_knots.min(),
                y_max=self.Y_knots.max(),
                z_min=self.Z_knots.min(),
                z_max=self.Z_knots.max(),
            )

        # Logging
        self.log_every = log_every
        self._n_seen = 0
        self._sum_abs_err = 0.0
        self._window_errs = []

        print(f"  Bias correction: α={alpha}, k={k}, r={radius}, h={bandwidth}, half_life={half_life}, W0={W0}")

    def _compute_y_scalar(self, prefill_chunk_pairs: List[List[int]]) -> float:
        """
        Convert prefill_chunk_pairs to Y-axis scalar for grid lookup.

        Uses sum of chunk*cumulative products (feature 8 from online predictor).
        """
        if not prefill_chunk_pairs:
            return 0.0
        return float(sum(pair[0] * pair[1] for pair in prefill_chunk_pairs))

    def predict(self, batch_size_tokens: int, prefill_chunk_pairs: List[List[int]], kv_tokens_used: int, mode: str = None) -> float:
        """
        Predict iteration time with grid + local bias correction.

        Args:
            batch_size_tokens: Total tokens in batch
            prefill_chunk_pairs: List of [current_chunk, cumulative_prefill] pairs
            kv_tokens_used: Number of KV cache tokens used
            mode: Mode for multi-mode grids (e.g., "MIXED", "DECODE", "EXTEND").
                  Required for multi-mode grids, ignored for single-mode grids.

        Returns:
            Predicted iteration time in milliseconds
        """
        # Map to grid coordinates
        x = float(batch_size_tokens)
        y = self._compute_y_scalar(prefill_chunk_pairs)
        z = float(kv_tokens_used)

        if self.is_multimode:
            # Multi-mode: select grid based on mode parameter
            if mode is None:
                raise ValueError(
                    f"Mode parameter required for multi-mode predictor. "
                    f"Available modes: {list(self.grids.keys())}"
                )
            if mode not in self.grids:
                raise ValueError(
                    f"Mode '{mode}' not found. Available modes: {list(self.grids.keys())}"
                )

            X_knots = self.X_knots_dict[mode]
            Y_knots = self.Y_knots_dict[mode]
            Z_knots = self.Z_knots_dict[mode]
            grid = self.grids[mode]
            bias_corrector = self.bias_correctors[mode]
        else:
            # Single-mode: use default grid
            X_knots = self.X_knots
            Y_knots = self.Y_knots
            Z_knots = self.Z_knots
            grid = self.grid
            bias_corrector = self.bias_corrector

        # Grid base prediction
        grid_pred = float(trilinear_predict(x, y, z, X_knots, Y_knots, Z_knots, grid))

        # Local bias correction
        correction, _ = bias_corrector.correction(x, y, z)

        # Final prediction (ensure non-negative)
        final_pred = max(0.0, grid_pred + correction)

        return final_pred

    def predict_batch(
        self,
        batch_size_tokens_list: List[int],
        prefill_chunk_pairs_list: List[List[List[int]]],
        kv_tokens_used_list: List[int],
        modes: List[str] = None,
    ) -> List[float]:
        """
        Batched variant of predict with identical semantics per element.

        Args:
            batch_size_tokens_list: List of batch sizes
            prefill_chunk_pairs_list: List of prefill_chunk_pairs per sample
            kv_tokens_used_list: List of KV tokens used per sample
            modes: List of modes (required for multi-mode grids; ignored otherwise)

        Returns:
            List of predicted iteration times (ms), one per input
        """
        if not (
            len(batch_size_tokens_list)
            == len(prefill_chunk_pairs_list)
            == len(kv_tokens_used_list)
        ):
            raise ValueError("All input lists must have the same length")

        if len(batch_size_tokens_list) == 0:
            return []

        preds: List[float] = [0.0] * len(batch_size_tokens_list)

        if self.is_multimode:
            if modes is None or len(modes) != len(batch_size_tokens_list):
                raise ValueError(
                    "modes must be provided and match input length for multi-mode predictor"
                )

            available_modes = set(self.grids.keys())
            invalid_modes = [m for m in modes if m not in available_modes]
            if invalid_modes:
                raise ValueError(
                    f"Mode(s) {invalid_modes} not found. Available modes: {list(self.grids.keys())}"
                )

            # Group indices by mode to reuse vectorized grid lookup per mode.
            mode_to_indices = {}
            for idx, mode in enumerate(modes):
                mode_to_indices.setdefault(mode, []).append(idx)

            for mode_name, indices in mode_to_indices.items():
                X_knots = self.X_knots_dict[mode_name]
                Y_knots = self.Y_knots_dict[mode_name]
                Z_knots = self.Z_knots_dict[mode_name]
                grid = self.grids[mode_name]
                bias_corrector = self.bias_correctors[mode_name]

                xs = np.array([float(batch_size_tokens_list[i]) for i in indices])
                ys = np.array(
                    [self._compute_y_scalar(prefill_chunk_pairs_list[i]) for i in indices]
                )
                zs = np.array([float(kv_tokens_used_list[i]) for i in indices])

                grid_preds = trilinear_predict(xs, ys, zs, X_knots, Y_knots, Z_knots, grid)

                for offset, idx in enumerate(indices):
                    correction, _ = bias_corrector.correction(
                        float(xs[offset]), float(ys[offset]), float(zs[offset])
                    )
                    preds[idx] = max(0.0, float(grid_preds[offset]) + correction)

        else:
            X_knots = self.X_knots
            Y_knots = self.Y_knots
            Z_knots = self.Z_knots
            grid = self.grid
            bias_corrector = self.bias_corrector

            xs = np.array([float(x) for x in batch_size_tokens_list])
            ys = np.array([self._compute_y_scalar(pairs) for pairs in prefill_chunk_pairs_list])
            zs = np.array([float(z) for z in kv_tokens_used_list])

            grid_preds = trilinear_predict(xs, ys, zs, X_knots, Y_knots, Z_knots, grid)

            for idx in range(len(batch_size_tokens_list)):
                correction, _ = bias_corrector.correction(
                    float(xs[idx]), float(ys[idx]), float(zs[idx])
                )
                preds[idx] = max(0.0, float(grid_preds[idx]) + correction)

        return preds

    def submit(
        self,
        batch_size_tokens: int,
        prefill_chunk_pairs: List[List[int]],
        kv_tokens_used: int,
        iteration_time_ms: float,
        mode: str = None,
    ) -> Tuple[float, float]:
        """
        Update predictor with actual measurement.

        Predicts first (using current state), then updates bias corrector.

        Args:
            batch_size_tokens: Total tokens in batch
            prefill_chunk_pairs: List of [current_chunk, cumulative_prefill] pairs
            kv_tokens_used: Number of KV cache tokens used
            iteration_time_ms: Actual measured iteration time
            mode: Mode for multi-mode grids (e.g., "MIXED", "DECODE", "EXTEND").
                  Required for multi-mode grids, ignored for single-mode grids.

        Returns:
            (prediction_before_update, absolute_error): Prediction and error in milliseconds
        """
        # Map to grid coordinates
        x = float(batch_size_tokens)
        y = self._compute_y_scalar(prefill_chunk_pairs)
        z = float(kv_tokens_used)

        # Predict BEFORE update
        y_pred = self.predict(batch_size_tokens, prefill_chunk_pairs, kv_tokens_used, mode=mode)

        if self.is_multimode:
            # Multi-mode: select grid based on mode parameter
            if mode is None:
                raise ValueError(
                    f"Mode parameter required for multi-mode predictor. "
                    f"Available modes: {list(self.grids.keys())}"
                )
            if mode not in self.grids:
                raise ValueError(
                    f"Mode '{mode}' not found. Available modes: {list(self.grids.keys())}"
                )

            X_knots = self.X_knots_dict[mode]
            Y_knots = self.Y_knots_dict[mode]
            Z_knots = self.Z_knots_dict[mode]
            grid = self.grids[mode]
            bias_corrector = self.bias_correctors[mode]
        else:
            # Single-mode: use default grid
            X_knots = self.X_knots
            Y_knots = self.Y_knots
            Z_knots = self.Z_knots
            grid = self.grid
            bias_corrector = self.bias_corrector

        # Compute base grid prediction (without bias correction)
        grid_pred = float(trilinear_predict(x, y, z, X_knots, Y_knots, Z_knots, grid))

        # Residual relative to base grid (this is what we correct)
        residual = float(iteration_time_ms) - grid_pred

        # Update bias corrector for this mode
        bias_corrector.update(x, y, z, residual)

        # Track error for logging
        abs_err = abs(float(iteration_time_ms) - y_pred)
        self._n_seen += 1
        self._sum_abs_err += abs_err
        self._window_errs.append(abs_err)
        if len(self._window_errs) > self.log_every:
            self._window_errs.pop(0)

        # Logging
        if self._n_seen % self.log_every == 0:
            window_avg = float(np.mean(self._window_errs)) if self._window_errs else 0.0
            cum_avg = self._sum_abs_err / self._n_seen
            if self._window_errs:
                p90 = float(np.percentile(self._window_errs, 90))
                p99 = float(np.percentile(self._window_errs, 99))
            else:
                p90 = p99 = 0.0

            print(
                f"[ModeAwarePredictor] N={self._n_seen} | "
                f"avg_abs_err_window({self.log_every})={window_avg:.2f} ms | "
                f"p90_window={p90:.2f} ms | p99_window={p99:.2f} ms | "
                f"avg_abs_err_cum={cum_avg:.2f} ms"
            )

        return y_pred, abs_err

    @property
    def n_seen(self) -> int:
        """Number of samples seen."""
        return self._n_seen

    @property
    def mean_abs_err_cum(self) -> float:
        """Cumulative mean absolute error."""
        return self._sum_abs_err / self._n_seen if self._n_seen > 0 else 0.0
