#!/usr/bin/env python3
"""
Cycle Time Predictor - Smooth Transition Model

Model: y_hat = c0 + w^T x + s * softplus((tau - m) / gamma)
where:
  - c0: baseline/startup overhead
  - w: linear coefficients (main trend)
  - alpha: scale weights (m = alpha^T x determines magnitude)
  - tau: transition point
  - gamma: smoothness
  - s: small-value correction magnitude
"""

import pickle
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class PredictionInput:
    """Input features for cycle time prediction."""
    batch_size_tokens: int
    prefill_chunk_pairs: List[List[int]]
    kv_tokens_used: int

    def to_features(self) -> np.ndarray:
        """
        Simplified 6-feature set:
        1. batch_size_tokens
        2. kv_tokens_used
        3. sum_chunk_history_product
        4. batch_squared
        5. kv_squared
        6. batch_times_kv
        """
        batch_size = float(self.batch_size_tokens)
        kv_used = float(self.kv_tokens_used)

        features = [batch_size, kv_used]

        # Prefill attention cost
        if self.prefill_chunk_pairs:
            chunks = np.array([p[0] for p in self.prefill_chunk_pairs], dtype=np.float32)
            cumulative = np.array([p[1] for p in self.prefill_chunk_pairs], dtype=np.float32)
            features.append(float(np.sum(chunks * cumulative)))
        else:
            features.append(0.0)

        # Non-linear terms
        features.extend([batch_size ** 2, kv_used ** 2, batch_size * kv_used])

        return np.array(features, dtype=np.float32)


def softplus(x: np.ndarray) -> np.ndarray:
    """Smooth ReLU: log(1 + exp(x))"""
    return np.log1p(np.exp(np.clip(x, -20, 20)))


class CycleTimePredictor:
    """
    Linear backbone + smooth small-value correction.

    Two-stage training:
      1. Initialize with linear regression (OLS)
      2. Refine transition parameters (alpha, tau, gamma, s) with gradient descent
    """

    def __init__(self):
        self.training_data: List[Tuple[PredictionInput, float]] = []
        self.is_trained = False

        # Normalization
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

        # Model parameters
        self.c0: float = 0.0
        self.w: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.tau: float = 1.0
        self.gamma: float = 1.0
        self.s: float = 0.0

    def submit(self, batch_size_tokens: int, prefill_chunk_pairs: List[List[int]],
               kv_tokens_used: int, iteration_time_ms: float) -> None:
        """Submit training data."""
        inp = PredictionInput(batch_size_tokens, prefill_chunk_pairs, kv_tokens_used)
        self.training_data.append((inp, iteration_time_ms))
        self.is_trained = False

    def _compute_scale(self, X_norm: np.ndarray) -> np.ndarray:
        """m = alpha^T x"""
        return X_norm @ self.alpha

    def _predict_batch(self, X_norm: np.ndarray) -> np.ndarray:
        """y = c0 + w^T x + s * softplus((tau - m) / gamma)"""
        y_linear = self.c0 + X_norm @ self.w
        m = self._compute_scale(X_norm)
        correction = self.s * softplus((self.tau - m) / self.gamma)
        return y_linear + correction

    def train(self) -> dict:
        """Two-stage training: linear init + gradient descent refinement."""
        if not self.training_data:
            raise ValueError("No training data")

        print(f"Training smooth transition model on {len(self.training_data)} samples...")

        # Extract and normalize
        X = np.array([inp.to_features() for inp, _ in self.training_data], dtype=np.float32)
        y = np.array([target for _, target in self.training_data], dtype=np.float32)

        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0)
        self.feature_std[self.feature_std == 0] = 1.0
        X_norm = (X - self.feature_mean) / self.feature_std

        n_features = X_norm.shape[1]

        # Stage 1: Linear initialization
        print("  Stage 1: Linear initialization...")
        X_aug = np.hstack([X_norm, np.ones((X_norm.shape[0], 1))])
        try:
            params = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        except:
            params = np.linalg.pinv(X_aug) @ y

        self.w = params[:-1]
        self.c0 = params[-1]

        # Initialize alpha as normalized absolute linear coefficients
        self.alpha = np.abs(self.w) / (np.sum(np.abs(self.w)) + 1e-8)

        # Initialize transition parameters
        m_values = X_norm @ self.alpha
        self.tau = float(np.percentile(m_values, 20))  # 20th percentile
        self.gamma = 1.0
        self.s = 0.0

        y_linear = self.c0 + X_norm @ self.w
        mae_linear = np.mean(np.abs(y - y_linear))
        print(f"  Linear baseline: MAE={mae_linear:.2f}ms")

        # Stage 2: Refine transition parameters
        print("  Stage 2: Refining transition parameters...")

        # Simple grid search for tau, gamma, s
        best_mae = mae_linear
        best_params = (self.tau, self.gamma, self.s)

        for tau_scale in [0.1, 0.2, 0.3, 0.5]:
            tau_try = float(np.percentile(m_values, tau_scale * 100))
            for gamma_try in [0.5, 1.0, 2.0, 5.0]:
                for s_try in [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0]:
                    self.tau, self.gamma, self.s = tau_try, gamma_try, s_try
                    y_pred = self._predict_batch(X_norm)
                    mae = np.mean(np.abs(y - y_pred))
                    if mae < best_mae:
                        best_mae = mae
                        best_params = (tau_try, gamma_try, s_try)

        self.tau, self.gamma, self.s = best_params
        print(f"  Best transition params: tau={self.tau:.3f}, gamma={self.gamma:.3f}, s={self.s:.3f}")

        self.is_trained = True

        # Final evaluation
        y_pred = self._predict_batch(X_norm)
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(mse)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        print(f"Training complete: MAE={mae:.2f}ms, RMSE={rmse:.2f}ms, R²={r2:.4f}")

        return {
            'num_samples': len(self.training_data),
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
        }

    def predict(self, batch_size_tokens: int, prefill_chunk_pairs: List[List[int]],
                kv_tokens_used: int) -> float:
        """Predict iteration time."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        inp = PredictionInput(batch_size_tokens, prefill_chunk_pairs, kv_tokens_used)
        features = inp.to_features()
        features_norm = (features - self.feature_mean) / self.feature_std

        prediction = float(self._predict_batch(features_norm.reshape(1, -1))[0])
        return max(0.0, prediction)

    def evaluate(self, test_data: List[Tuple[PredictionInput, float]]) -> dict:
        """Evaluate on test set."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        predictions = []
        targets = []
        for inp, target in test_data:
            pred = self.predict(inp.batch_size_tokens, inp.prefill_chunk_pairs, inp.kv_tokens_used)
            predictions.append(pred)
            targets.append(target)

        predictions = np.array(predictions)
        targets = np.array(targets)

        mse = np.mean((targets - predictions) ** 2)
        mae = np.mean(np.abs(targets - predictions))
        rmse = np.sqrt(mse)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        ss_res = np.sum((targets - predictions) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100

        return {
            'num_samples': len(test_data),
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'mape': float(mape),
        }

    def save(self, filepath: str) -> None:
        """Save model."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        state = {
            'c0': self.c0,
            'w': self.w,
            'alpha': self.alpha,
            'tau': self.tau,
            'gamma': self.gamma,
            's': self.s,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'training_data': self.training_data,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.c0 = state['c0']
        self.w = state['w']
        self.alpha = state['alpha']
        self.tau = state['tau']
        self.gamma = state['gamma']
        self.s = state['s']
        self.feature_mean = state['feature_mean']
        self.feature_std = state['feature_std']
        self.training_data = state['training_data']
        self.is_trained = True

        print(f"Model loaded from {filepath}")
