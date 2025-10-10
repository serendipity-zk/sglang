
# Cycle Time Estimation

A machine learning framework for predicting SGLang worker iteration cycle times based on batch characteristics and system state.

## Overview

This tool parses SGLang worker metrics logs, extracts relevant features, and trains a predictive model to estimate iteration cycle times. It can be used for:

- Performance analysis and optimization
- SLO-aware scheduling
- Resource planning and capacity estimation

## Components

### 1. `log_parser.py`
Parses STAT_METRICS logs and extracts:
- **Features**: `batch_size_tokens`, `prefill_chunk_pairs`, `kv_tokens_used`
- **Target**: `iteration_time_ms`
- **Context**: forward mode, queue status, KV usage percentage

### 2. `predictor.py`
Implements the `CycleTimePredictor` class with:
- **`submit(batch_size_tokens, prefill_chunk_pairs, kv_tokens_used, iteration_time_ms)`**
  - Submit known data pairs for training
- **`predict(batch_size_tokens, prefill_chunk_pairs, kv_tokens_used)`**
  - Predict cycle time for given inputs
- **`train()`** - Train the model on submitted data
- **`evaluate(test_data)`** - Evaluate on test set
- **`save(filepath)`** / **`load(filepath)`** - Persist trained models

### 3. `frontend.py`
Main CLI interface with three commands:
- **`train`** - Parse logs, train model, evaluate
- **`predict`** - Interactive prediction mode
- **`parse`** - Parse logs and show statistics

## Installation

No additional dependencies required beyond standard Python libraries:
```bash
cd /sgl-workspace/sglang/cycle_time_est
```

## Usage

### 1. Parse a log file and show statistics
```bash
python frontend.py parse /path/to/worker.log
```

### 2. Train a model from log file
```bash
# Default ensemble model with 90/10 train/test split
python frontend.py train /path/to/worker.log --output model.pkl

# Use linear regression model
python frontend.py train /path/to/worker.log --output model.pkl --model-type linear

# Custom 80/20 split with polynomial model
python frontend.py train /path/to/worker.log --output model.pkl --train-ratio 0.8 --model-type polynomial

# Save test set predictions to a file
python frontend.py train /path/to/worker.log --output model.pkl --predictions preds.json
```

**Model Types:**
- `linear`: Simple linear regression (fast, interpretable, R²≈0.32)
- `polynomial`: Polynomial regression degree=2 (may overfit, not recommended)
- `ensemble`: Gradient boosting with 50 weak learners (best accuracy, R²≈0.32, default) ✅
- `hybrid`: Global Ridge on normalized features + KNN residual smoothing (adds local corrections)

**Output:**
- Parses STAT_METRICS lines
- Splits into 90% training, 10% test (or custom ratio)
- Trains selected model type
- Reports MAE, RMSE, R² on both train and test sets
- Saves model to specified file
- Dumps test set predictions to JSON file (includes actual vs predicted values, errors, and statistics)

### 3. Make predictions with trained model
```bash
# Default (uses model type saved in the model)
python frontend.py predict model.pkl

# Override model type at prediction time (must match trained artifacts)
python frontend.py predict model.pkl --model-type hybrid
```

**Interactive mode example:**
```
batch_size_tokens: 186
kv_tokens_used: 897717
prefill_chunk_pairs (JSON format, e.g., [[256,256]]): []
➜ Predicted iteration time: 43.28 ms

batch_size_tokens: 512
kv_tokens_used: 450000
prefill_chunk_pairs (JSON format, e.g., [[256,256]]): [[256,256],[512,512]]
➜ Predicted iteration time: 37.45 ms
```

## Feature Engineering

The predictor extracts the following 19 features from inputs:

### Basic Features (3):
1. **batch_size_tokens** - Total tokens in the batch
2. **kv_tokens_used** - KV cache memory usage
3. **num_prefill_requests** - Number of prefill requests in batch

### Prefill Chunk Statistics (5):
4. **total_prefill_chunks** - Sum of current chunk sizes
5. **total_cumulative_prefill** - Sum of cumulative prefill lengths
6. **avg_chunk_size** - Average chunk size
7. **max_chunk_size** - Maximum chunk size
8. **avg_cumulative_size** - Average cumulative size

### Attention Computation Features (3):
9. **sum_chunk_history_product** - Sum of (current × cumulative) for each prefill pair
10. **max_chunk_history_product** - Maximum (current × cumulative)
11. **avg_chunk_history_product** - Average (current × cumulative)

### Quadratic Complexity Features (2):
12. **sum_chunk_squared** - Sum of current²
13. **sum_cumulative_squared** - Sum of cumulative²

### Additional Prefill Statistics (2):
14. **min_chunk_size** - Minimum chunk size
15. **std_chunk_size** - Standard deviation of chunk sizes

### Derived Interaction Features (4):
16. **kv_to_batch_ratio** - KV usage / batch size ratio
17. **batch_squared** - batch_size_tokens²
18. **kv_squared** - kv_tokens_used²
19. **batch_times_kv** - batch_size_tokens × kv_tokens_used

The attention computation features (9-11) are particularly important as they capture the O(n×m) complexity of attention computation, where n is the current chunk size and m is the cumulative context length.

## Example Log Format

The tool expects log lines in this format:
```
[2025-10-10 11:43:51] STAT_METRICS: {"running_batch_size":186,"queue_reqs":24,"kv_tokens_used":897717,"token_capacity":915780,"kv_usage_pct":98.03,"prefill_tokens":0,"decode_tokens":186,"token_batch_size":186,"iteration_time_ms":43.16,"forward_mode":"DECODE","prefill_chunk_pairs":[],"batch_size_tokens":186,"num_requests":186,"input_id_len":186,"worker_id":"0.0.0.0:31004","timestamp":1760096631.8051252,"iteration_num":1603,"accepted_requests":282}
```

## API Example

```python
from predictor import CycleTimePredictor

# Create predictor
predictor = CycleTimePredictor()

# Submit training data
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

# Train the model
stats = predictor.train()
print(f"Training MAE: {stats['mae']:.2f} ms")

# Make predictions
pred_time = predictor.predict(
    batch_size_tokens=200,
    prefill_chunk_pairs=[],
    kv_tokens_used=900000
)
print(f"Predicted time: {pred_time:.2f} ms")

# Save for later use
predictor.save('my_model.pkl')
```

## Model Architecture

The predictor now supports three model types with no external dependencies:

### 1. Linear Regression (model_type='linear')
- Simple least squares regression
- Fast training and prediction
- Interpretable coefficients
- Best for: Quick prototyping, understanding feature importance

### 2. Polynomial Regression (model_type='polynomial')
- Adds all pairwise feature interactions (degree=2)
- Captures non-linear relationships
- Risk of overfitting on small datasets
- Best for: Large datasets (>5000 samples)

### 3. Ensemble (model_type='ensemble', default)
- Gradient boosting with 50 weak learners
- Learning rate: 0.1
- Combines multiple linear models trained on residuals

### 4. Hybrid Ridge + KNN Residuals (model_type='hybrid')
- Global trend via Ridge regression on 6 normalized features: `[batch_size_tokens, kv_tokens_used, sum_chunk_history_product, batch_squared, kv_squared, batch_times_kv]`
- Local correction via distance-weighted KNN residual averaging with gating:
  - `weights = 1/(d + eps)` (default), `gate = exp(-(d_k/s)^2)` with `s` as median k-th neighbor distance
  - Falls back to linear part if neighbors insufficient
- Configurable hyperparameters (defaults in parentheses): `ridge_alpha(1.0)`, `n_neighbors(25)`, `knn_algorithm('auto')`, `distance_metric('minkowski')`, `distance_p(2)`, `gate_mode('exp'|'inv')`, `gate_bandwidth(None)`, `eps(1e-6)`

Train with:
```bash
python frontend.py train worker.log --model-type hybrid --output model.pkl
```
- Best generalization on test data
- Best for: Production use, best accuracy

**Recommendation**: Use `ensemble` (default) for best results.

## Extending the Model

To add external libraries like XGBoost:

```python
from xgboost import XGBRegressor

def train(self):
    X = np.array([inp.to_features() for inp, _ in self.training_data])
    y = np.array([target for _, target in self.training_data])

    self.model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8
    )
    self.model.fit(X, y)
    self.is_trained = True
```

## Performance Metrics

The tool reports:
- **MAE** (Mean Absolute Error) - Average prediction error in ms
- **RMSE** (Root Mean Square Error) - Penalizes large errors
- **R²** (R-squared) - Proportion of variance explained (0-1)
- **MAPE** (Mean Absolute Percentage Error) - Percentage error

## License

Part of the SGLang project.
