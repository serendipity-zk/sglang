#!/usr/bin/env python3
"""
Cycle Time Predictor

A machine learning model for predicting iteration cycle time based on
batch characteristics and system state.
"""

import pickle
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class PredictionInput:
    """Input features for cycle time prediction."""
    batch_size_tokens: int
    prefill_chunk_pairs: List[List[int]]  # List of [current_chunk, cumulative_prefill]
    kv_tokens_used: int

    def to_features(self) -> np.ndarray:
        """
        Convert to feature vector for the model.

        Feature engineering:
        1. batch_size_tokens (total tokens in batch)
        2. kv_tokens_used (KV cache usage)
        3. num_prefill_requests (number of prefill requests in batch)
        4. total_prefill_chunks (sum of current chunks)
        5. total_cumulative_prefill (sum of cumulative prefill)
        6. avg_chunk_size (average chunk size, 0 if no prefill)
        7. max_chunk_size (max chunk size, 0 if no prefill)
        8. avg_cumulative_size (average cumulative size, 0 if no prefill)
        9. kv_to_batch_ratio (KV usage / batch size)
        10. sum_chunk_history_product (sum of current * history for each prefill pair)
        11. max_chunk_history_product (max of current * history)
        12. avg_chunk_history_product (average of current * history, 0 if no prefill)
        13. sum_chunk_squared (sum of current^2 for attention computation estimate)
        14. sum_cumulative_squared (sum of cumulative^2)
        15. min_chunk_size (minimum chunk size, 0 if no prefill)
        16. std_chunk_size (standard deviation of chunk sizes, 0 if no prefill)
        17. batch_squared (batch_size_tokens^2)
        18. kv_squared (kv_tokens_used^2)
        19. batch_times_kv (batch_size_tokens * kv_tokens_used)
        """
        features = []

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

            # Basic aggregations (5 features)
            features.append(float(np.sum(chunks)))  # total_prefill_chunks
            features.append(float(np.sum(cumulative)))  # total_cumulative_prefill
            features.append(float(np.mean(chunks)))  # avg_chunk_size
            features.append(float(np.max(chunks)))  # max_chunk_size
            features.append(float(np.mean(cumulative)))  # avg_cumulative_size

            # NEW: Product features (current * history) - captures attention computation cost (3 features)
            # The cost of attention is proportional to current_chunk * cumulative_length
            chunk_history_products = chunks * cumulative
            features.append(float(np.sum(chunk_history_products)))  # sum_chunk_history_product
            features.append(float(np.max(chunk_history_products)))  # max_chunk_history_product
            features.append(float(np.mean(chunk_history_products)))  # avg_chunk_history_product

            # NEW: Squared features - for modeling quadratic complexity (2 features)
            features.append(float(np.sum(chunks ** 2)))  # sum_chunk_squared
            features.append(float(np.sum(cumulative ** 2)))  # sum_cumulative_squared

            # NEW: Additional statistics (2 features)
            features.append(float(np.min(chunks)))  # min_chunk_size
            features.append(float(np.std(chunks)))  # std_chunk_size
        else:
            # No prefill, use zeros for all prefill-related features (5+3+2+2 = 12 features)
            features.extend([0.0] * 12)

        # Derived features from basic inputs
        kv_to_batch_ratio = kv_used / batch_size if batch_size > 0 else 0.0
        features.append(kv_to_batch_ratio)

        # NEW: Quadratic and interaction terms for basic features
        features.append(batch_size ** 2)  # batch_squared
        features.append(kv_used ** 2)  # kv_squared
        features.append(batch_size * kv_used)  # batch_times_kv

        return np.array(features, dtype=np.float32)


class CycleTimePredictor:
    """
    Cycle time predictor supporting multiple model types.

    Models:
    - 'linear': Simple linear regression (fast, interpretable)
    - 'polynomial': Polynomial regression with degree 2 (better accuracy)
    - 'ensemble': Simple gradient boosting ensemble (best accuracy)
    - 'hybrid': Global Ridge on normalized features + KNN residual smoothing
    """

    def __init__(self, model_type: str = 'ensemble',
                 # Hybrid model hyperparameters
                 ridge_alpha: float = 1.0,
                 n_neighbors: int = 25,
                 knn_algorithm: str = 'auto',
                 distance_metric: str = 'minkowski',
                 distance_p: int = 2,
                 gate_mode: str = 'exp',  # 'exp' or 'inv'
                 gate_bandwidth: Optional[float] = None,
                 eps: float = 1e-6,
                 random_state: Optional[int] = 42):
        """
        Initialize the predictor.

        Args:
            model_type: Type of model to use ('linear', 'polynomial', or 'ensemble')
        """
        self.model_type = model_type
        self.model = None
        self.training_data: List[Tuple[PredictionInput, float]] = []
        self.is_trained = False

        # Feature statistics for normalization
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

        # For ensemble models
        self.models: List[np.ndarray] = []
        self.learning_rate: float = 0.1
        self.n_estimators: int = 200

        # Hybrid model state
        self.hybrid_w: Optional[np.ndarray] = None  # Ridge weights on normalized features
        self.hybrid_b: Optional[float] = None       # Intercept
        self.hybrid_X_norm_train: Optional[np.ndarray] = None
        self.hybrid_residuals_train: Optional[np.ndarray] = None
        self.hybrid_feature_mean: Optional[np.ndarray] = None
        self.hybrid_feature_std: Optional[np.ndarray] = None
        self.hybrid_gate_bandwidth: Optional[float] = gate_bandwidth
        # Hybrid hyperparams
        self.ridge_alpha = float(ridge_alpha)
        self.n_neighbors = int(n_neighbors)
        self.knn_algorithm = str(knn_algorithm)
        self.distance_metric = str(distance_metric)
        self.distance_p = int(distance_p)
        self.gate_mode = str(gate_mode)
        self.gate_bandwidth = gate_bandwidth
        self.eps = float(eps)
        self.random_state = random_state
        # KNN index (sklearn-like if available)
        self._knn_index = None

    def _select_hybrid_features(self, X: np.ndarray) -> np.ndarray:
        """
        Select the 6-D feature subset for the hybrid model.

        Assumes full feature order from PredictionInput.to_features().
        Hybrid 6-D subset:
        [0] batch_size_tokens
        [1] kv_tokens_used
        [8] sum_chunk_history_product
        [16] batch_squared
        [17] kv_squared
        [18] batch_times_kv
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        idx = [0, 1, 8, 16, 17]
        idx = [i for i in idx if i < X.shape[1]]
        return X[:, idx].astype(np.float32)

    def _fit_ridge_with_intercept(self, X_norm: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
        """Closed-form ridge with intercept on normalized features.

        Returns (w, b) where y ≈ X_norm @ w + b
        """
        # Center X_norm and y to compute weights without intercept
        Xc = X_norm
        y_mean = float(np.mean(y))
        y_c = y - y_mean
        n_features = Xc.shape[1]
        I = np.eye(n_features, dtype=np.float64)
        # Solve (X^T X + alpha I) w = X^T y
        XtX = (Xc.T @ Xc).astype(np.float64)
        Xty = (Xc.T @ y_c).astype(np.float64)
        try:
            w = np.linalg.solve(XtX + alpha * I, Xty)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(XtX + alpha * I) @ Xty
        # Intercept for original (non-centered y)
        b = y_mean
        return w.astype(np.float32), float(b)

    def _fit_knn_index(self, X_norm: np.ndarray) -> None:
        """Fit KNN index. Uses sklearn if available, falls back to brute-force."""
        try:
            from sklearn.neighbors import NearestNeighbors  # type: ignore
            self._knn_index = NearestNeighbors(
                n_neighbors=max(1, self.n_neighbors),
                algorithm=self.knn_algorithm,
                metric=self.distance_metric,
                p=self.distance_p,
            )
            self._knn_index.fit(X_norm)
            self._knn_is_sklearn = True
        except Exception:
            # Fallback: store training data for brute-force neighbor search
            self._knn_index = X_norm.astype(np.float32)
            self._knn_is_sklearn = False

    def _kneighbors(self, X_query: np.ndarray, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
        """Query KNN index and return (distances, indices)."""
        n_neighbors = max(1, n_neighbors)
        if getattr(self, '_knn_is_sklearn', False):
            distances, indices = self._knn_index.kneighbors(X_query, n_neighbors=n_neighbors)
            return distances, indices
        # Brute-force fallback
        X_train = self.hybrid_X_norm_train
        if X_train is None or len(X_train) == 0:
            raise RuntimeError("KNN index is not fitted")
        # Compute L2 distances
        dists = np.sqrt(np.maximum(0.0, ((X_query[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)))
        # Argpartition to get top-k
        idx = np.argpartition(dists, kth=min(n_neighbors-1, dists.shape[1]-1), axis=1)[:, :n_neighbors]
        # Sort each row
        row_sorted = np.take_along_axis(dists, idx, axis=1)
        order = np.argsort(row_sorted, axis=1)
        sorted_idx = np.take_along_axis(idx, order, axis=1)
        sorted_d = np.take_along_axis(row_sorted, order, axis=1)
        return sorted_d, sorted_idx

    def _kth_neighbor_distances(self, X_norm: np.ndarray) -> np.ndarray:
        """Compute kth neighbor distances for each training point (excluding self)."""
        k = max(1, self.n_neighbors)
        # Query k+1 to account for self-distance 0
        try:
            if getattr(self, '_knn_is_sklearn', False):
                distances, _ = self._knn_index.kneighbors(X_norm, n_neighbors=min(k+1, len(X_norm)))
            else:
                distances, _ = self._kneighbors(X_norm, n_neighbors=min(k+1, len(X_norm)))
            if distances.shape[1] >= k+1:
                kth = distances[:, k]  # 0 is self, kth is the k-th neighbor
            else:
                kth = distances[:, -1]
            return kth.astype(np.float32)
        except Exception:
            return np.array([], dtype=np.float32)

    def _hybrid_predict_batch(self, X_h: np.ndarray) -> np.ndarray:
        """Predict for a batch using hybrid model.

        X_h: raw features restricted to 6-D subset; will be normalized internally
        """
        if self.hybrid_w is None or self.hybrid_b is None or self.hybrid_X_norm_train is None or self.hybrid_residuals_train is None:
            raise RuntimeError("Hybrid model artifacts missing. Train with model_type='hybrid'.")

        if X_h.ndim == 1:
            X_h = X_h.reshape(1, -1)
        if self.hybrid_feature_mean is None or self.hybrid_feature_std is None:
            raise RuntimeError("Hybrid normalization stats not found. Train with model_type='hybrid'.")

        # Normalize using stored stats
        X_norm = (X_h - self.hybrid_feature_mean) / self.hybrid_feature_std

        # Global linear prediction
        y_lin = (X_norm @ self.hybrid_w) + self.hybrid_b

        # Residual correction via KNN
        try:
            distances, indices = self._kneighbors(X_norm, n_neighbors=min(self.n_neighbors, len(self.hybrid_X_norm_train)))
            # Extract residuals of neighbors
            r_neighbors = self.hybrid_residuals_train[indices]
            d = distances
            # Inverse distance weights
            w = 1.0 / (d + self.eps)
            # Weighted residual average
            r_knn = (w * r_neighbors).sum(axis=1) / w.sum(axis=1)

            # Gate computation
            if self.gate_mode == 'exp':
                # k-th neighbor distance per query
                d_k = d[:, -1]
                s = self.hybrid_gate_bandwidth if (self.hybrid_gate_bandwidth is not None and self.hybrid_gate_bandwidth > 0) else 1.0
                gate = np.exp(- (d_k / s) ** 2)
            else:
                gate = np.ones_like(r_knn)

            y_hat = y_lin.reshape(-1) + gate * r_knn
        except Exception:
            # Fallback to linear prediction
            y_hat = y_lin.reshape(-1)

        # Ensure non-negative
        return np.maximum(0.0, y_hat)

    def submit(
        self,
        batch_size_tokens: int,
        prefill_chunk_pairs: List[List[int]],
        kv_tokens_used: int,
        iteration_time_ms: float
    ) -> None:
        """
        Submit a known data pair for training.

        Args:
            batch_size_tokens: Total tokens in the batch
            prefill_chunk_pairs: List of [current_chunk, cumulative_prefill] pairs
            kv_tokens_used: KV cache tokens used
            iteration_time_ms: Actual iteration time in milliseconds (target)
        """
        input_data = PredictionInput(
            batch_size_tokens=batch_size_tokens,
            prefill_chunk_pairs=prefill_chunk_pairs,
            kv_tokens_used=kv_tokens_used
        )

        self.training_data.append((input_data, iteration_time_ms))

        # Mark as untrained since we have new data
        self.is_trained = False

    def _add_polynomial_features(self, X: np.ndarray, degree: int = 2) -> np.ndarray:
        """Add polynomial features up to specified degree."""
        X_poly = X.copy()
        n_features = X.shape[1]

        if degree >= 2:
            # Add all pairwise interactions (degree 2)
            for i in range(n_features):
                for j in range(i, n_features):
                    X_poly = np.column_stack([X_poly, X[:, i] * X[:, j]])

        return X_poly

    def train(self) -> dict:
        """
        Train the model on submitted data.

        Returns:
            Dictionary with training statistics
        """
        if not self.training_data:
            raise ValueError("No training data available. Use submit() to add data.")

        print(f"Training {self.model_type} model on {len(self.training_data)} samples...")

        # Extract features and targets
        X_list = []
        y_list = []

        for input_data, target in self.training_data:
            features = input_data.to_features()
            X_list.append(features)
            y_list.append(target)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        # Compute normalization statistics
        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0)
        self.feature_std[self.feature_std == 0] = 1.0  # Avoid division by zero

        # Normalize features
        X_norm = (X - self.feature_mean) / self.feature_std

        if self.model_type == 'hybrid':
            # Use a compact 6-D feature subset for hybrid
            X_h = self._select_hybrid_features(X)
            # Normalize based on subset stats
            feat_mean_h = np.mean(X_h, axis=0)
            feat_std_h = np.std(X_h, axis=0)
            feat_std_h[feat_std_h == 0] = 1.0
            X_norm_h = (X_h - feat_mean_h) / feat_std_h
            # Store subset stats
            self.hybrid_feature_mean = feat_mean_h.astype(np.float32)
            self.hybrid_feature_std = feat_std_h.astype(np.float32)

            # Fit global Ridge with intercept on normalized features
            self.hybrid_w, self.hybrid_b = self._fit_ridge_with_intercept(X_norm_h, y, alpha=self.ridge_alpha)
            y_lin = X_norm_h @ self.hybrid_w + self.hybrid_b
            residuals = y - y_lin

            # Fit KNN index on normalized features
            self._fit_knn_index(X_norm_h)
            self.hybrid_X_norm_train = X_norm_h.astype(np.float32)
            self.hybrid_residuals_train = residuals.astype(np.float32)

            # Fit gate bandwidth s if not provided: median of kth neighbor distance per training sample
            if self.gate_bandwidth is None:
                kth_d = self._kth_neighbor_distances(X_norm_h)
                self.hybrid_gate_bandwidth = float(np.median(kth_d)) if len(kth_d) > 0 else 1.0
            else:
                self.hybrid_gate_bandwidth = float(self.gate_bandwidth)

            # Training predictions for metrics
            y_pred = self._hybrid_predict_batch(X_h)

        elif self.model_type == 'ensemble':
            # For ensemble, we'll train multiple weak learners
            # Apply model-specific transformations
            # Add bias term
            Xn = X_norm
            Xn = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
            # Simple gradient boosting
            self.models = []
            predictions = np.zeros_like(y)

            for i in range(self.n_estimators):
                # Compute residuals
                residuals = y - predictions

                # Train a weak learner on residuals
                try:
                    weak_model = np.linalg.lstsq(Xn, residuals, rcond=None)[0]
                except np.linalg.LinAlgError:
                    weak_model = np.linalg.pinv(Xn) @ residuals

                self.models.append(weak_model)

                # Update predictions
                predictions += self.learning_rate * (Xn @ weak_model)

                if (i + 1) % 10 == 0:
                    mae = np.mean(np.abs(y - predictions))
                    print(f"  Iteration {i+1}/{self.n_estimators}: MAE={mae:.2f}ms")

            y_pred = predictions
        else:
            # Linear or polynomial regression using least squares
            Xn = X_norm
            if self.model_type == 'polynomial':
                print("Adding polynomial features (degree=2)...")
                Xn = self._add_polynomial_features(Xn, degree=2)
            # Add bias term
            Xn = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
            try:
                self.model = np.linalg.lstsq(Xn, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix, using pseudo-inverse")
                self.model = np.linalg.pinv(Xn) @ y

            y_pred = Xn @ self.model

        self.is_trained = True

        # Compute training statistics
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(mse)

        # R^2 score
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        stats = {
            'num_samples': len(self.training_data),
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
        }

        print(f"Training complete: MAE={mae:.2f}ms, RMSE={rmse:.2f}ms, R²={r2:.4f}")

        return stats

    def predict(
        self,
        batch_size_tokens: int,
        prefill_chunk_pairs: List[List[int]],
        kv_tokens_used: int,
        use_model_type: Optional[str] = None
    ) -> float:
        """
        Predict cycle time for given input features.

        Args:
            batch_size_tokens: Total tokens in the batch
            prefill_chunk_pairs: List of [current_chunk, cumulative_prefill] pairs
            kv_tokens_used: KV cache tokens used

        Returns:
            Predicted iteration time in milliseconds
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        input_data = PredictionInput(
            batch_size_tokens=batch_size_tokens,
            prefill_chunk_pairs=prefill_chunk_pairs,
            kv_tokens_used=kv_tokens_used
        )

        # Choose model
        model_type = use_model_type if use_model_type is not None else self.model_type

        # Extract features
        features_full = input_data.to_features()

        if model_type == 'hybrid':
            prediction = float(self._hybrid_predict_batch(self._select_hybrid_features(features_full.reshape(1, -1)))[0])
        else:
            # Extract and normalize features
            features_norm = (features_full - self.feature_mean) / self.feature_std
            # Apply model-specific transformations
            if model_type == 'polynomial':
                features_norm = self._add_polynomial_features(features_norm.reshape(1, -1), degree=2)[0]
            # Add bias term
            features_norm = np.append(features_norm, 1.0)
            # Predict based on model type
            if model_type == 'ensemble':
                prediction = 0.0
                for weak_model in self.models:
                    prediction += self.learning_rate * float(features_norm @ weak_model)
            else:
                prediction = float(features_norm @ self.model)

        # Ensure non-negative prediction
        prediction = max(0.0, prediction)

        return prediction

    def evaluate(self, test_data: List[Tuple[PredictionInput, float]]) -> dict:
        """
        Evaluate the model on test data.

        Args:
            test_data: List of (input, target) tuples

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        if not test_data:
            raise ValueError("No test data provided")

        predictions = []
        targets = []

        for input_data, target in test_data:
            pred = self.predict(
                input_data.batch_size_tokens,
                input_data.prefill_chunk_pairs,
                input_data.kv_tokens_used
            )
            predictions.append(pred)
            targets.append(target)

        predictions = np.array(predictions)
        targets = np.array(targets)

        # Compute metrics
        mse = np.mean((targets - predictions) ** 2)
        mae = np.mean(np.abs(targets - predictions))
        rmse = np.sqrt(mse)

        # R^2 score
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        ss_res = np.sum((targets - predictions) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Percentage errors
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
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        state = {
            'model_type': self.model_type,
            'model': self.model,
            'models': self.models if self.model_type == 'ensemble' else [],
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'training_data': self.training_data,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            # Hybrid
            'hybrid_w': self.hybrid_w,
            'hybrid_b': self.hybrid_b,
            'hybrid_X_norm_train': self.hybrid_X_norm_train,
            'hybrid_residuals_train': self.hybrid_residuals_train,
            'hybrid_feature_mean': self.hybrid_feature_mean,
            'hybrid_feature_std': self.hybrid_feature_std,
            'hybrid_gate_bandwidth': self.hybrid_gate_bandwidth,
            'ridge_alpha': self.ridge_alpha,
            'n_neighbors': self.n_neighbors,
            'knn_algorithm': self.knn_algorithm,
            'distance_metric': self.distance_metric,
            'distance_p': self.distance_p,
            'gate_mode': self.gate_mode,
            'gate_bandwidth': self.gate_bandwidth,
            'eps': self.eps,
            'random_state': self.random_state,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.model_type = state.get('model_type', 'linear')  # Backward compatibility
        self.model = state['model']
        self.models = state.get('models', [])
        self.feature_mean = state['feature_mean']
        self.feature_std = state['feature_std']
        self.training_data = state['training_data']
        self.learning_rate = state.get('learning_rate', 0.1)
        self.n_estimators = state.get('n_estimators', 50)
        # Hybrid
        self.hybrid_w = state.get('hybrid_w', None)
        self.hybrid_b = state.get('hybrid_b', None)
        self.hybrid_X_norm_train = state.get('hybrid_X_norm_train', None)
        self.hybrid_residuals_train = state.get('hybrid_residuals_train', None)
        self.hybrid_gate_bandwidth = state.get('hybrid_gate_bandwidth', None)
        self.hybrid_feature_mean = state.get('hybrid_feature_mean', None)
        self.hybrid_feature_std = state.get('hybrid_feature_std', None)
        self.ridge_alpha = state.get('ridge_alpha', 1.0)
        self.n_neighbors = state.get('n_neighbors', 25)
        self.knn_algorithm = state.get('knn_algorithm', 'auto')
        self.distance_metric = state.get('distance_metric', 'minkowski')
        self.distance_p = state.get('distance_p', 2)
        self.gate_mode = state.get('gate_mode', 'exp')
        self.gate_bandwidth = state.get('gate_bandwidth', None)
        self.eps = state.get('eps', 1e-6)
        self.random_state = state.get('random_state', 42)

        # Rebuild KNN index for hybrid if applicable
        if self.model_type == 'hybrid' and self.hybrid_X_norm_train is not None:
            self._fit_knn_index(self.hybrid_X_norm_train)

        self.is_trained = True

        print(f"Model loaded from {filepath}")


if __name__ == '__main__':
    # Example usage
    predictor = CycleTimePredictor(model_type='hybrid')

    # Submit some example data
    predictor.submit(
        batch_size_tokens=186,
        prefill_chunk_pairs=[],
        kv_tokens_used=897717,
        iteration_time_ms=43.16
    )

    predictor.submit(
        batch_size_tokens=512,
        prefill_chunk_pairs=[[256, 256], [512, 512]],
        kv_tokens_used=450000,
        iteration_time_ms=35.2
    )

    # Train
    stats = predictor.train()
    print(f"Training stats: {stats}")

    # Predict
    pred_time = predictor.predict(
        batch_size_tokens=200,
        prefill_chunk_pairs=[],
        kv_tokens_used=900000,
        use_model_type='hybrid'
    )
    print(f"Predicted time: {pred_time:.2f}ms")

    # Also demonstrate using current model type implicitly
