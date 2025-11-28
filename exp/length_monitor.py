import numpy as np
from collections import defaultdict
from typing import Tuple, Union

# --- 1. The Base Class (Handles the Log-Binning Logic) ---
class DynamicLogHistogram:
    def __init__(self, bins_per_decade: int = 10):
        self.bins_per_decade = bins_per_decade
        self.scale_factor = bins_per_decade
        # Sparse storage: Key=BucketIndex, Value=Count/Prob
        self.counts = defaultdict(float) 
        self.min_val_seen = float('inf')
        self.max_val_seen = float('-inf')

    def _get_bucket_index(self, value: float) -> int:
        return int(np.floor(np.log10(value) * self.scale_factor))

    # We leave .add() empty here because the subclass will define 
    # HOW to add data (Count vs Decay vs EMA)
    def add(self, value):
        raise NotImplementedError

# --- 2. The EMA Subclass (Handles the "Last 256 Batches" Logic) ---
class RequestOutputLengthProfile(DynamicLogHistogram):
    def __init__(self, window_size_batches: int = 256, bins_per_decade: int = 10):
        super().__init__(bins_per_decade)
        # Calculate Alpha automatically based on window size
        # Formula: 2 / (N + 1)
        self.alpha = 2.0 / (window_size_batches + 1.0)

    def add(self, values: Union[float, np.ndarray, list]):
        # Handle input types
        if isinstance(values, (int, float)):
            values = np.array([values])
        else:
            values = np.array(values)
            
        # Filter valid data
        values = values[values > 0]
        if len(values) == 0:
            return

        # 1. Update global stats
        self.min_val_seen = min(self.min_val_seen, np.min(values))
        self.max_val_seen = max(self.max_val_seen, np.max(values))

        # 2. Calculate the "Shape" of the CURRENT batch
        #    (Convert this batch into a probability distribution)
        batch_indices = np.floor(np.log10(values) * self.scale_factor).astype(int)
        unique_indices, counts = np.unique(batch_indices, return_counts=True)
        
        # Normalize batch to sum=1.0
        batch_total = counts.sum()
        batch_probs = {k: v / batch_total for k, v in zip(unique_indices, counts)}

        # 3. Apply EMA Formula:
        #    NewState = (OldState * Decay) + (BatchState * Alpha)
        decay_mult = 1.0 - self.alpha
        
        # A. Decay existing history
        keys_to_remove = []
        for k in self.counts:
            self.counts[k] *= decay_mult
            if self.counts[k] < 1e-6: # Cleanup tiny residuals
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self.counts[k]

        # B. Mix in the new batch
        for idx, prob in batch_probs.items():
            self.counts[idx] += (prob * self.alpha)

    def get_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.counts:
            return np.array([0.0, 1.0]), np.array([1.0])

        sorted_indices = sorted(self.counts.keys())
        min_idx, max_idx = sorted_indices[0], sorted_indices[-1]
        
        # Fill dense array
        dense_probs = np.zeros(max_idx - min_idx + 1)
        for idx, prob in self.counts.items():
            dense_probs[idx - min_idx] = prob
            
        # Reconstruct Edges
        indices_arr = np.arange(min_idx, max_idx + 2)
        edges = np.power(10.0, indices_arr / self.scale_factor)
        
        # Final safety normalize
        total = dense_probs.sum()
        if total > 0: dense_probs /= total
            
        return edges, dense_probs