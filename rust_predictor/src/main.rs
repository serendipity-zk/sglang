use knn_workload_predictor::{KNNWorkloadPredictor, PredictBuffer, PredictionStatus};
use serde::Deserialize;
use std::error::Error;
use std::time::Instant;
use rand::Rng;

#[derive(Debug, Deserialize)]
struct TraceRow {
    batch_size_tokens: i32,
    prefill_chunk_pairs: String,
    kv_tokens_used: i32,
    actual_time_ms: Option<f32>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && args[1] == "bench" {
        run_benchmark();
    } else {
        // Default to simulation
        let csv_path = "/sgl-workspace/sglang/slo/logs/predictor/predictor_w3_gpu6_p31003.csv";
        println!("Starting trace simulation on: {}", csv_path);
        if let Err(e) = run_simulation(csv_path) {
            eprintln!("Simulation error: {}", e);
            eprintln!("Usage: cargo run --release -- [bench]");
        }
    }

    Ok(())
}

fn run_simulation(file_path: &str) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(file_path)?;

    let k = 10;
    let max_history = 5000;
    let mut predictor = KNNWorkloadPredictor::new(k, max_history);
    
    // Load grid fallback model if available
    let grid_path = "/sgl-workspace/sglang/sglang_profile/mode_3d.json";
    if let Err(e) = predictor.load_grid_from_file(grid_path) {
        eprintln!("Warning: could not load grid fallback '{}': {}", grid_path, e);
    } else {
        println!("Loaded grid fallback model from {}", grid_path);
    }
    
    // Allocate ONE reusable buffer structure
    let mut buffer = PredictBuffer::new(max_history);

    let mut errors: Vec<f32> = Vec::new();
    let mut count = 0;

    let start_time = Instant::now();

    for result in rdr.deserialize() {
        let row: TraceRow = result?;

        if let Some(actual_time) = row.actual_time_ms {
            let prediction = predictor.predict_from_str(
                row.batch_size_tokens,
                &row.prefill_chunk_pairs,
                row.kv_tokens_used,
                &mut buffer
            );

            predictor.update_from_str(
                row.batch_size_tokens,
                &row.prefill_chunk_pairs,
                row.kv_tokens_used,
                actual_time,
            );

            let abs_error = match prediction.status {
                PredictionStatus::InsufficientData => None,
                _ => {
                    let err = (prediction.predicted_time - actual_time).abs();
                    errors.push(err);
                    Some(err)
                }
            };

            println!(
                "Event {:>6}: predicted {:.4} ms (status: {:?}), actual {:.4} ms{}",
                count + 1,
                prediction.predicted_time,
                prediction.status,
                actual_time,
                match abs_error {
                    Some(e) => format!(", abs error {:.4} ms", e),
                    None => ", abs error N/A (insufficient data)".to_string(),
                }
            );
            count += 1;
        }
    }

    let duration = start_time.elapsed();
    let mean_mae: f32 = if errors.is_empty() { 0.0 } else { errors.iter().sum::<f32>() / errors.len() as f32 };

    println!("--- Simulation Complete ---");
    println!("Total Events: {}", count);
    println!("Time Elapsed: {:.2?}", duration);
    println!("Final MAE:    {:.4} ms", mean_mae);
    Ok(())
}

fn run_benchmark() {
    let max_history = 1000;  // Set to 1000 to match your target scenario
    let k = 10;
    let mut predictor = KNNWorkloadPredictor::new(k, max_history);
    let mut rng = rand::thread_rng();

    println!("--- Starting Benchmark (N={}, K={}, Parallel) ---", max_history, k);
    println!("Phase 1: Warming up history...");

    // 1. Fill History
    for _ in 0..max_history {
        let batch = rng.gen_range(1..128);
        let kv = rng.gen_range(0..8000);
        let time = rng.gen_range(5.0..100.0);
        predictor.update(batch, &[], kv, time);
    }

    println!("History filled. Preparing large batch...");

    // 2. Prepare Data
    for iter in 0..5 {
        println!("Benchmark Iteration {}", iter + 1);
        let batch_size = 1_000;
        let mut b_vec = Vec::with_capacity(batch_size);
        let mut p_vec: Vec<Vec<[i32; 2]>> = Vec::with_capacity(batch_size);
        let mut k_vec = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            b_vec.push(rng.gen_range(1..128));
            p_vec.push(vec![]);  // empty prefill pairs for decode-only benchmark
            k_vec.push(rng.gen_range(0..8000));
        }

        println!("Phase 2: Running parallel batch prediction on {} items...", batch_size);
        
        let start = Instant::now();

        let results = predictor.predict_batch_parallel(&b_vec, &p_vec, &k_vec);

        let duration = start.elapsed();
        let rps = batch_size as f64 / duration.as_secs_f64();

        assert_eq!(results.len(), batch_size);

        println!("Processed {} queries in {:.2?}", batch_size, duration);
        println!("Throughput: {:.2} requests/sec", rps);
        println!("Avg Latency: {:.2} μs", (duration.as_micros() as f64) / (batch_size as f64));
    }
}
