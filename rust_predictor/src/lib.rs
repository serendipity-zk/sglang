#![allow(non_local_definitions)]

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use rayon::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[derive(Debug, Clone, Copy)]
pub struct PredictionResult {
    pub predicted_time: f32,
    pub neighbors_found: usize,
    pub status: PredictionStatus,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredictionStatus {
    Success,          // KNN prediction
    GridFallback,     // Grid prediction
    InsufficientData, // No prediction possible
}

/// Helper struct to manage reuseable memory buffers per thread.
/// Separating distances (f32) from indices (usize) allows for
/// SIMD-friendly contiguous memory access during the heavy math phase.
pub struct PredictBuffer {
    pub distances: Vec<f32>,
    pub indices: Vec<usize>,
}

impl PredictBuffer {
    pub fn new(capacity: usize) -> Self {
        // Pre-fill indices [0, 1, 2, ..., capacity]
        // We will just slice this array during prediction rather than re-creating it.
        let indices: Vec<usize> = (0..capacity).collect();
        Self {
            distances: Vec::with_capacity(capacity),
            indices,
        }
    }
}

// --- Grid Structures ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridKnots {
    #[serde(rename = "X_knots")]
    pub x_knots: Vec<f32>,
    #[serde(rename = "Y_knots")]
    pub y_knots: Vec<f32>,
    #[serde(rename = "Z_knots")]
    pub z_knots: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridModeData {
    pub knots: GridKnots,
    pub grid: Vec<Vec<Vec<f32>>>, 
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridFileFormat {
    pub modes: Option<std::collections::HashMap<String, GridModeData>>,
    pub knots: Option<GridKnots>,
    pub grid: Option<Vec<Vec<Vec<f32>>>>,
}

#[derive(Debug, Clone)]
struct FlattenedGrid {
    x_knots: Vec<f32>,
    y_knots: Vec<f32>,
    z_knots: Vec<f32>,
    data: Vec<f32>, 
    dims: (usize, usize, usize),
}

impl FlattenedGrid {
    fn from_nested(knots: GridKnots, grid_3d: Vec<Vec<Vec<f32>>>) -> Self {
        let dim_x = grid_3d.len();
        let dim_y = if dim_x > 0 { grid_3d[0].len() } else { 0 };
        let dim_z = if dim_y > 0 { grid_3d[0][0].len() } else { 0 };
        
        let mut data = Vec::with_capacity(dim_x * dim_y * dim_z);
        for x in &grid_3d {
            for y in x {
                for z in y {
                    data.push(*z);
                }
            }
        }

        Self {
            x_knots: knots.x_knots,
            y_knots: knots.y_knots,
            z_knots: knots.z_knots,
            data,
            dims: (dim_x, dim_y, dim_z),
        }
    }

    fn predict(&self, x: f32, y: f32, z: f32) -> f32 {
        let ix = match self.x_knots.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let iy = match self.y_knots.binary_search_by(|v| v.partial_cmp(&y).unwrap()) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let iz = match self.z_knots.binary_search_by(|v| v.partial_cmp(&z).unwrap()) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };

        let ix = ix.min(self.x_knots.len().saturating_sub(2));
        let iy = iy.min(self.y_knots.len().saturating_sub(2));
        let iz = iz.min(self.z_knots.len().saturating_sub(2));

        let x0 = self.x_knots[ix]; let x1 = self.x_knots[ix+1];
        let y0 = self.y_knots[iy]; let y1 = self.y_knots[iy+1];
        let z0 = self.z_knots[iz]; let z1 = self.z_knots[iz+1];

        let tx = if x1 > x0 { (x - x0) / (x1 - x0) } else { 0.0 };
        let ty = if y1 > y0 { (y - y0) / (y1 - y0) } else { 0.0 };
        let tz = if z1 > z0 { (z - z0) / (z1 - z0) } else { 0.0 };

        let get = |i, j, k| {
            let idx = i * self.dims.1 * self.dims.2 + j * self.dims.2 + k;
            self.data.get(idx).copied().unwrap_or(0.0)
        };

        let g000 = get(ix, iy, iz);
        let g100 = get(ix+1, iy, iz);
        let g010 = get(ix, iy+1, iz);
        let g110 = get(ix+1, iy+1, iz);
        let g001 = get(ix, iy, iz+1);
        let g101 = get(ix+1, iy, iz+1);
        let g011 = get(ix, iy+1, iz+1);
        let g111 = get(ix+1, iy+1, iz+1);

        let c00 = g000 * (1.0 - tx) + g100 * tx;
        let c01 = g001 * (1.0 - tx) + g101 * tx;
        let c10 = g010 * (1.0 - tx) + g110 * tx;
        let c11 = g011 * (1.0 - tx) + g111 * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;

        c0 * (1.0 - tz) + c1 * tz
    }
}

// --- Main Predictor ---

pub struct KNNWorkloadPredictor {
    k_neighbors: usize,
    max_history: usize,

    // --- Structure of Arrays (SoA) for Raw History ---
    raw_x: Vec<f32>,
    raw_y: Vec<f32>,
    raw_z: Vec<f32>,
    raw_w: Vec<f32>,
    time_history: Vec<f32>,

    // --- SoA for Scaled History (Hot Cache) ---
    scaled_x: Vec<f32>,
    scaled_y: Vec<f32>,
    scaled_z: Vec<f32>,
    scaled_w: Vec<f32>,

    // Scaler state
    mean: [f32; 4],
    scale: [f32; 4],
    is_fitted: bool,

    // Grid Fallback
    grids: std::collections::HashMap<String, FlattenedGrid>,
    has_grid_fallback: bool,
}

impl KNNWorkloadPredictor {
    pub fn new(k_neighbors: usize, max_history: usize) -> Self {
        Self {
            k_neighbors,
            max_history,
            raw_x: Vec::with_capacity(max_history),
            raw_y: Vec::with_capacity(max_history),
            raw_z: Vec::with_capacity(max_history),
            raw_w: Vec::with_capacity(max_history),
            time_history: Vec::with_capacity(max_history),
            scaled_x: Vec::with_capacity(max_history),
            scaled_y: Vec::with_capacity(max_history),
            scaled_z: Vec::with_capacity(max_history),
            scaled_w: Vec::with_capacity(max_history),
            mean: [0.0; 4],
            scale: [1.0; 4],
            is_fitted: false,
            grids: std::collections::HashMap::new(),
            has_grid_fallback: false,
        }
    }

    pub fn load_grid_from_file(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let model: GridFileFormat = serde_json::from_reader(reader)?;

        self.grids.clear();
        if let Some(modes) = model.modes {
            for (mode_name, mode_data) in modes {
                let flat_grid = FlattenedGrid::from_nested(mode_data.knots, mode_data.grid);
                self.grids.insert(mode_name, flat_grid);
            }
        } else if let (Some(knots), Some(grid)) = (model.knots, model.grid) {
            let flat_grid = FlattenedGrid::from_nested(knots, grid);
            self.grids.insert("DECODE".to_string(), flat_grid.clone());
            self.grids.insert("MIXED".to_string(), flat_grid);
        }
        self.has_grid_fallback = !self.grids.is_empty();
        Ok(())
    }

    fn determine_mode(&self, prefill_chunk_pairs: &[[i32; 2]]) -> &str {
        if prefill_chunk_pairs.is_empty() {
            "DECODE"
        } else {
            "MIXED"
        }
    }

    fn compute_prefill_features(&self, prefill_chunk_pairs: &[[i32; 2]]) -> (f32, f32) {
        if prefill_chunk_pairs.is_empty() {
            return (0.0, 0.0);
        }
        
        let mut score_sq: f32 = 0.0;
        for pair in prefill_chunk_pairs {
            score_sq += (pair[0] as f32) * (pair[1] as f32);
        }
        (score_sq.sqrt(), prefill_chunk_pairs.len() as f32)
    }

    /// Parse JSON string to array pairs (for CSV/CLI compatibility)
    fn parse_prefill_string(&self, prefill_str: &str) -> Vec<[i32; 2]> {
        if prefill_str == "[]" || prefill_str.is_empty() {
            return Vec::new();
        }
        let pairs: Vec<[i32; 2]> = match serde_json::from_str(prefill_str) {
            Ok(p) => p,
            Err(_) => Vec::new(),
        };
        pairs
    }

    fn refit_and_cache(&mut self) {
        let n = self.raw_x.len();
        if n == 0 { self.is_fitted = false; return; }
        let n_f32 = n as f32;
        
        let sum_x: f32 = self.raw_x.iter().sum();
        let sum_y: f32 = self.raw_y.iter().sum();
        let sum_z: f32 = self.raw_z.iter().sum();
        let sum_w: f32 = self.raw_w.iter().sum();

        self.mean = [sum_x / n_f32, sum_y / n_f32, sum_z / n_f32, sum_w / n_f32];

        let mut sq_diff_x = 0.0;
        let mut sq_diff_y = 0.0;
        let mut sq_diff_z = 0.0;
        let mut sq_diff_w = 0.0;

        for i in 0..n {
            sq_diff_x += (self.raw_x[i] - self.mean[0]).powi(2);
            sq_diff_y += (self.raw_y[i] - self.mean[1]).powi(2);
            sq_diff_z += (self.raw_z[i] - self.mean[2]).powi(2);
            sq_diff_w += (self.raw_w[i] - self.mean[3]).powi(2);
        }

        self.scale = [
            (sq_diff_x / n_f32).sqrt(), (sq_diff_y / n_f32).sqrt(),
            (sq_diff_z / n_f32).sqrt(), (sq_diff_w / n_f32).sqrt(),
        ];
        for i in 0..4 { if self.scale[i] == 0.0 { self.scale[i] = 1.0; } }

        self.scaled_x.clear(); self.scaled_y.clear();
        self.scaled_z.clear(); self.scaled_w.clear();

        for i in 0..n {
            self.scaled_x.push((self.raw_x[i] - self.mean[0]) / self.scale[0]);
            self.scaled_y.push((self.raw_y[i] - self.mean[1]) / self.scale[1]);
            self.scaled_z.push((self.raw_z[i] - self.mean[2]) / self.scale[2]);
            self.scaled_w.push((self.raw_w[i] - self.mean[3]) / self.scale[3]);
        }
        if n >= self.k_neighbors { self.is_fitted = true; }
    }

    pub fn update(&mut self, batch: i32, prefill: &[[i32; 2]], kv: i32, time: f32) {
        let (int_score, n_pre) = self.compute_prefill_features(prefill);
        self.raw_x.push(batch as f32);
        self.raw_y.push(int_score);
        self.raw_z.push(kv as f32);
        self.raw_w.push(n_pre);
        self.time_history.push(time);

        if self.raw_x.len() > self.max_history {
            self.raw_x.remove(0); self.raw_y.remove(0);
            self.raw_z.remove(0); self.raw_w.remove(0);
            self.time_history.remove(0);
        }
        self.refit_and_cache();
    }

    /// Update from JSON string (for CSV/CLI compatibility)
    pub fn update_from_str(&mut self, batch: i32, prefill_str: &str, kv: i32, time: f32) {
        let prefill = self.parse_prefill_string(prefill_str);
        self.update(batch, &prefill, kv, time);
    }

    pub fn update_batch(&mut self, batches: &[i32], prefills: &[Vec<[i32; 2]>], kvs: &[i32], times: &[f32]) {
        let count = batches.len();
        if count == 0 { return; }
        for i in 0..count {
            let (int_score, n_pre) = self.compute_prefill_features(&prefills[i]);
            self.raw_x.push(batches[i] as f32);
            self.raw_y.push(int_score);
            self.raw_z.push(kvs[i] as f32);
            self.raw_w.push(n_pre);
            self.time_history.push(times[i]);
        }
        if self.raw_x.len() > self.max_history {
            let excess = self.raw_x.len() - self.max_history;
            self.raw_x.drain(0..excess); self.raw_y.drain(0..excess);
            self.raw_z.drain(0..excess); self.raw_w.drain(0..excess);
            self.time_history.drain(0..excess);
        }
        self.refit_and_cache();
    }

    /// Optimized Predict Function:
    /// Uses split buffers (f32 array + index array) to allow for SIMD autovectorization.
    pub fn predict(
        &self,
        batch: i32,
        prefill: &[[i32; 2]],
        kv: i32,
        buffer: &mut PredictBuffer
    ) -> PredictionResult {
        let (int_score, n_pre) = self.compute_prefill_features(prefill);

        if self.is_fitted {
            let qx = (batch as f32 - self.mean[0]) / self.scale[0];
            let qy = (int_score - self.mean[1]) / self.scale[1];
            let qz = (kv as f32 - self.mean[2]) / self.scale[2];
            let qw = (n_pre - self.mean[3]) / self.scale[3];

            let len = self.scaled_x.len();
            
            // 1. Vectorized Distance Calculation (Pure Float Math)
            // We write directly to the float buffer.
            // Safety: We reserve enough capacity. Setting len unsafely skips checks for the loop.
            // This loop should compile to 8-wide or 16-wide SIMD instructions.
            buffer.distances.clear();
            unsafe { buffer.distances.set_len(len); }

            for i in 0..len {
                let dx = qx - self.scaled_x[i];
                let dy = qy - self.scaled_y[i];
                let dz = qz - self.scaled_z[i];
                let dw = qw - self.scaled_w[i];
                buffer.distances[i] = dx*dx + dy*dy + dz*dz + dw*dw;
            }

            // 2. Select Top K using Indices
            // We use the persistent indices buffer (0..N) and only sort the active slice.
            // Note: Make sure indices buffer is large enough if history grew.
            if buffer.indices.len() < len {
                buffer.indices = (0..self.max_history).collect();
            }
            
            let active_indices = &mut buffer.indices[0..len];
            let k = self.k_neighbors.min(len);
            
            // Partial sort on INDICES by looking up DISTANCES
            let partition_idx = k.saturating_sub(1);
            active_indices.select_nth_unstable_by(partition_idx, |&i, &j| {
                buffer.distances[i].partial_cmp(&buffer.distances[j]).unwrap()
            });
            
            let nearest_indices = &active_indices[..k];

            // 3. Weighted Average
            let mut num = 0.0;
            let mut den = 0.0;
            
            for &idx in nearest_indices {
                let dist = buffer.distances[idx].sqrt();
                if dist < 1e-6 {
                    return PredictionResult { 
                        predicted_time: self.time_history[idx], 
                        neighbors_found: self.k_neighbors, 
                        status: PredictionStatus::Success 
                    };
                }
                let w = 1.0 / dist;
                num += w * self.time_history[idx];
                den += w;
            }
            return PredictionResult {
                predicted_time: (if den == 0.0 { 0.0 } else { num/den }).max(0.0),
                neighbors_found: self.k_neighbors,
                status: PredictionStatus::Success,
            };
        }

        if self.has_grid_fallback {
            let mode = self.determine_mode(prefill);
            if let Some(grid) = self.grids.get(mode) {
                let gp = grid.predict(batch as f32, int_score, kv as f32);
                return PredictionResult {
                    predicted_time: gp.max(0.0),
                    neighbors_found: 0,
                    status: PredictionStatus::GridFallback,
                };
            }
        }

        PredictionResult {
            predicted_time: 0.0,
            neighbors_found: 0,
            status: PredictionStatus::InsufficientData,
        }
    }

    pub fn predict_simple(&self, batch: i32, prefill: &[[i32; 2]], kv: i32) -> PredictionResult {
        let mut buffer = PredictBuffer::new(self.max_history);
        self.predict(batch, prefill, kv, &mut buffer)
    }

    /// Predict from JSON string (for CSV/CLI compatibility)
    pub fn predict_from_str(
        &self,
        batch: i32,
        prefill_str: &str,
        kv: i32,
        buffer: &mut PredictBuffer
    ) -> PredictionResult {
        let prefill = self.parse_prefill_string(prefill_str);
        self.predict(batch, &prefill, kv, buffer)
    }

    pub fn predict_batch_parallel(
        &self,
        batches: &[i32],
        prefills: &[Vec<[i32; 2]>],
        kvs: &[i32],
    ) -> Vec<PredictionResult> {
        (batches, prefills, kvs)
            .into_par_iter()
            .map_init(
                || PredictBuffer::new(self.max_history),
                |buffer, (b, p, k)| {
                    self.predict(*b, p, *k, buffer)
                }
            )
            .collect()
    }
}



// =========================================================================
//  PYTHON BINDINGS
// =========================================================================

/// This struct wraps the pure Rust implementation for Python
#[pyclass(name = "KNNPredictor")]
pub struct PyKNNPredictor {
    inner: KNNWorkloadPredictor,
}

#[pymethods]
impl PyKNNPredictor {
    #[new]
    #[pyo3(signature = (k_neighbors=10, max_history=5000))]
    fn new(k_neighbors: usize, max_history: usize) -> Self {
        PyKNNPredictor {
            inner: KNNWorkloadPredictor::new(k_neighbors, max_history),
        }
    }

    fn load_grid(&mut self, path: String) -> PyResult<()> {
        self.inner.load_grid_from_file(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
        })
    }

    /// Update with native Python list of [chunk, cumulative] pairs
    fn update(&mut self, batch: i32, prefill: Vec<[i32; 2]>, kv: i32, time: f32) {
        self.inner.update(batch, &prefill, kv, time);
    }

    /// Optimized batch update for Python lists
    fn update_batch(&mut self, batches: Vec<i32>, prefills: Vec<Vec<[i32; 2]>>, kvs: Vec<i32>, times: Vec<f32>) {
        self.inner.update_batch(&batches, &prefills, &kvs, &times);
    }

    /// Parallel Batch Prediction for Python.
    /// Releases the GIL to allow true multi-core processing.
    fn predict_batch(&self, py: Python<'_>, batches: Vec<i32>, prefills: Vec<Vec<[i32; 2]>>, kvs: Vec<i32>) -> Vec<(f32, String)> {
        // We assume lists are equal length; simple check recommended in prod
        
        // Release GIL and run Rayon parallel predictor
        let results = py.allow_threads(move || {
            self.inner.predict_batch_parallel(&batches, &prefills, &kvs)
        });

        // Convert back to Python-friendly format (Tuple: (Time, Status))
        results.into_iter().map(|r| {
            let status_str = match r.status {
                PredictionStatus::Success => "KNN",
                PredictionStatus::GridFallback => "GRID",
                PredictionStatus::InsufficientData => "NONE",
            };
            (r.predicted_time, status_str.to_string())
        }).collect()
    }
}

/// The Python module definition
#[pymodule]
fn knn_workload_predictor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyKNNPredictor>()?;
    Ok(())
}
