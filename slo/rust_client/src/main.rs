use anyhow::Result;
use bytes::BytesMut;
use clap::Parser;
use futures::StreamExt;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_pickle as pickle;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokenizers::Tokenizer;

// --- Configuration ---

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    trace: String,

    #[arg(long)]
    text_file: String,

    #[arg(long)]
    tokenizer: String,

    #[arg(long, default_value = "http://0.0.0.0:40000/v1")]
    base_url: String,

    #[arg(long, default_value = "meta-llama/Llama-3.1-8B-Instruct")]
    model: String,

    #[arg(long, default_value_t = 1.0)]
    rate: f64,

    #[arg(long, default_value_t = 0.2)]
    temperature: f64,

    #[arg(long)]
    max_requests: Option<usize>,

    #[arg(long, default_value = "output.jsonl")]
    log_path: String,

    #[arg(long)]
    elapsed_dump_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TraceRow {
    arrival: f64, // ms
    prefill: usize,
    decode: usize,
    #[serde(default)]
    ttft: Option<f64>,
    #[serde(default)]
    tpot: Option<f64>,
}

#[derive(Serialize)]
struct LogRecord {
    request_id: String,
    input_len: usize,
    trace_output_len: usize,
    real_output_len: usize,
    status: String,
    submit_timestamp: f64,
    post_timestamp: Option<f64>,
    total_duration_ms: f64,
    ttft_ms: Option<f64>,
    avg_interval_ms: Option<f64>,
    chunk_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_ttft_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_tpot_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    slo_satisfied: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    slo_violations: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    slo_tokens_checked: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    slo_badness_ms: Option<f64>,
    intervals: Vec<f64>,
    output_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
struct TokenElapsedTimeline {
    request_id: String,
    elapsed_ms: Vec<f64>,
}

// --- Global Shared State ---

#[derive(Default)]
struct SloTierStats {
    attained: usize,
    total: usize,
}

struct SharedStats {
    submitted: AtomicUsize,
    completed: AtomicUsize,
    failed: AtomicUsize,
    slo_tiers: RwLock<HashMap<String, SloTierStats>>,
}

impl SharedStats {
    fn new() -> Self {
        Self {
            submitted: AtomicUsize::new(0),
            completed: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
            slo_tiers: RwLock::new(HashMap::new()),
        }
    }

    fn record_submit(&self) {
        self.submitted.fetch_add(1, Ordering::Relaxed);
    }

    fn record_complete(&self, success: bool) {
        if success {
            self.completed.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed.fetch_add(1, Ordering::Relaxed);
        }
    }

    async fn record_slo_result(&self, tier_label: Option<String>, satisfied: bool) {
        if let Some(label) = tier_label {
            let mut tiers = self.slo_tiers.write().await;
            let stats = tiers.entry(label).or_insert_with(SloTierStats::default);
            stats.total += 1;
            if satisfied {
                stats.attained += 1;
            }
        }
    }

    fn snapshot(&self) -> (usize, usize, usize) {
        (
            self.submitted.load(Ordering::Relaxed),
            self.completed.load(Ordering::Relaxed),
            self.failed.load(Ordering::Relaxed),
        )
    }

    async fn snapshot_slo_tiers(&self) -> HashMap<String, (usize, usize)> {
        let tiers = self.slo_tiers.read().await;
        tiers.iter()
            .map(|(k, v)| (k.clone(), (v.attained, v.total)))
            .collect()
    }
}

struct AppState {
    client: reqwest::Client,
    token_pool: Vec<u32>,
    args: Args,
    endpoint: String,
    stats: Arc<SharedStats>,
    elapsed_dump_tx: Option<tokio::sync::mpsc::Sender<TokenElapsedTimeline>>,
}

// --- Helper Functions ---

fn format_slo_tier_label(tpot_ms: Option<f64>) -> Option<String> {
    tpot_ms.map(|value| {
        if (value - value.round()).abs() < 1e-6 {
            format!("{} ms", value.round() as i32)
        } else {
            format!("{:.1} ms", value)
        }
    })
}

async fn status_display_task(
    stats: Arc<SharedStats>,
    total_requests: usize,
    log_path: String,
    start_time: Instant,
) {
    use std::io::{self, Write};

    let is_tty = atty::is(atty::Stream::Stdout);

    loop {
        tokio::time::sleep(Duration::from_millis(500)).await;

        let elapsed_s = start_time.elapsed().as_secs_f64();
        let (submitted, completed, failed) = stats.snapshot();
        let active = submitted.saturating_sub(completed + failed);
        let slo_tiers = stats.snapshot_slo_tiers().await;

        let submit_speed = submitted as f64 / elapsed_s;
        let complete_speed = completed as f64 / elapsed_s;
        let percent_complete = if total_requests > 0 {
            (completed + failed) as f64 / total_requests as f64 * 100.0
        } else {
            0.0
        };

        let mut lines = vec![
            "SLO Streaming Issue Runner (Rust) - Live Stats".to_string(),
            "=================================================".to_string(),
            format!("Total Requests       : {}", total_requests),
            format!("Submitted            : {}", submitted),
            format!("Active               : {}", active),
            format!("Completed / Failed   : {} / {}", completed, failed),
            format!("Submit Rate (req/s)  : {:.2}", submit_speed),
            format!("Complete Rate (req/s): {:.2}", complete_speed),
            format!("Elapsed (s)          : {:.1}", elapsed_s),
            format!("Progress             : {:.1}%", percent_complete),
            "".to_string(),
            format!("Log File             : {}", log_path),
            "Ctrl+C to stop".to_string(),
        ];

        if !slo_tiers.is_empty() {
            lines.push("".to_string());
            lines.push("SLO Attainment (per TPOT tier)".to_string());

            // Sort tiers by numeric value
            let mut tier_vec: Vec<_> = slo_tiers.iter().collect();
            tier_vec.sort_by(|a, b| {
                let a_val: f64 = a.0.split_whitespace().next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(f64::INFINITY);
                let b_val: f64 = b.0.split_whitespace().next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(f64::INFINITY);
                a_val.partial_cmp(&b_val).unwrap()
            });

            for (tier_label, (attained, total)) in tier_vec {
                let percent = if *total > 0 {
                    *attained as f64 / *total as f64 * 100.0
                } else {
                    0.0
                };
                lines.push(format!("  {:>8} : {}/{} ({:.1}%)", tier_label, attained, total, percent));
            }
        }

        if is_tty {
            // Clear screen and render
            print!("\x1B[2J\x1B[H");
            for line in &lines {
                println!("{}", line);
            }
            io::stdout().flush().ok();
        }

        // Check if done
        if completed + failed >= total_requests {
            if is_tty {
                println!();
            }
            break;
        }
    }
}

// --- Main Execution ---

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Optimize OS Limits (Important for 10k connections)
    match fdlimit::raise_fd_limit() {
        Ok(_) => println!("OS File Descriptor limit raised successfully"),
        Err(e) => println!("Failed to raise FD limit: {}. Ensure `ulimit -n` is high.", e),
    }

    let args = Args::parse();
    
    // 2. Prepare Logger (Dedicated Thread)
    let (log_tx, mut log_rx) = tokio::sync::mpsc::channel::<LogRecord>(100_000);
    let log_path = args.log_path.clone();
    
    tokio::spawn(async move {
        let file = File::create(&log_path).expect("Failed to create log file");
        let mut writer = std::io::BufWriter::with_capacity(1024 * 1024, file);
        while let Some(record) = log_rx.recv().await {
            if let Ok(json) = serde_json::to_string(&record) {
                writeln!(writer, "{}", json).ok();
            }
        }
        writer.flush().ok();
    });

    // Optional: high-performance elapsed timeline dump (pickle stream)
    let elapsed_dump_tx = if let Some(path) = args.elapsed_dump_path.clone() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<TokenElapsedTimeline>(100_000);
        tokio::spawn(async move {
            let file = File::create(&path).expect("Failed to create elapsed dump file");
            let mut writer = std::io::BufWriter::with_capacity(1024 * 1024, file);
            let opts = pickle::SerOptions::new();
            while let Some(timeline) = rx.recv().await {
                if let Err(e) = pickle::to_writer(&mut writer, &timeline, opts.clone()) {
                    eprintln!("Failed to write elapsed timeline: {}", e);
                }
            }
            writer.flush().ok();
        });
        Some(tx)
    } else {
        None
    };

    // 3. Build Components
    println!("Loading tokenizer and building token pool...");
    let token_pool = build_token_pool(&args.text_file, &args.tokenizer, 200_000)?;
    println!("Token pool size: {}", token_pool.len());

    let mut base = args.base_url.clone();
    if base.ends_with("/v1") {
        base = base.replace("/v1", "");
    }
    let endpoint = format!("{}/generate", base.trim_end_matches('/'));

    // High-performance Client Configuration
    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(20_000) // Allow massive concurrency
        .tcp_nodelay(true)              // CRITICAL: Disable Nagle's algo for <5ms latency
        .timeout(Duration::from_secs(3600)) // Long timeout, handle internally
        .build()?;

    let stats = Arc::new(SharedStats::new());

    let state = Arc::new(AppState {
        client,
        token_pool,
        args: args.clone(), // Clone for thread safety
        endpoint,
        stats: stats.clone(),
        elapsed_dump_tx,
    });

    // 4. Load and Sort Trace
    println!("Loading trace...");
    let mut rdr = csv::Reader::from_path(&args.trace)?;
    let mut rows: Vec<TraceRow> = rdr.deserialize().collect::<Result<_, _>>()?;
    
    // Sort by arrival time to ensure chronological dispatch
    rows.sort_by(|a, b| a.arrival.partial_cmp(&b.arrival).unwrap());

    if let Some(max) = args.max_requests {
        if max < rows.len() {
            rows.truncate(max);
        }
    }
    
    let total_reqs = rows.len();
    println!("Starting trace replay: {} requests", total_reqs);

    // 5. The Dispatch Loop
    let start_time = Instant::now();

    // Spawn status display task
    let stats_clone = stats.clone();
    let log_path_clone = args.log_path.clone();
    tokio::spawn(async move {
        status_display_task(stats_clone, total_reqs, log_path_clone, start_time).await;
    });
    let epoch_start = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
    let rate = args.rate;

    // We use a JoinSet to ensure we wait for everyone at the end, 
    // but we don't await individual tasks in the loop.
    let mut join_set = tokio::task::JoinSet::new();

    for (i, row) in rows.into_iter().enumerate() {
        let state_ref = state.clone();
        let log_tx_ref = log_tx.clone();

        // Calculate precise dispatch time
        let arrival_ms = row.arrival / rate;
        let target_time = start_time + Duration::from_secs_f64(arrival_ms / 1000.0);

        // Wait precisely
        tokio::time::sleep_until(tokio::time::Instant::from_std(target_time)).await;

        // Spawn Worker
        join_set.spawn(async move {
            let req_id = format!("req_{:06}", i);
            process_request(state_ref, row, req_id, log_tx_ref, epoch_start).await;
        });
    }

    // Wait for all requests to drain
    while join_set.join_next().await.is_some() {}

    println!("\nAll requests completed. Log saved.");

    Ok(())
}

// --- Worker Logic ---

async fn process_request(
    state: Arc<AppState>,
    row: TraceRow,
    req_id: String,
    logger: tokio::sync::mpsc::Sender<LogRecord>,
    _epoch_start: f64,
) {
    // Record submission
    state.stats.record_submit();

    // Record submit timestamp at the very start (wall clock time)
    let submit_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    // Use perf_counter for accurate duration measurements
    let start_ts = Instant::now();

    let mut rng = rand::rngs::StdRng::from_entropy();

    // 1. Sample Prompt
    let pool_len = state.token_pool.len();
    let start_idx = rng.gen_range(0..pool_len.saturating_sub(row.prefill));
    let prompt_ids = &state.token_pool[start_idx..start_idx + row.prefill];

    // 2. Build Payload
    let mut payload = serde_json::json!({
        "input_ids": [prompt_ids],
        "sampling_params": {
            "max_new_tokens": row.decode,
            "temperature": state.args.temperature,
            "ignore_eos": true
        },
        "stream": true
    });

    // Add SLO targets if present (matching Python behavior)
    if let Some(ttft) = row.ttft {
        payload["target_ttft_ms"] = serde_json::json!(ttft);
    }
    if let Some(tpot) = row.tpot {
        payload["target_tpot_ms"] = serde_json::json!(tpot);
    }

    let mut chunk_count: usize = 0;

    // 3. Send Request - record post timestamp right before sending
    let post_timestamp = Some(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64());
    let response_result = state.client.post(&state.endpoint)
        .header("x-request-id", &req_id)
        .json(&payload)
        .send()
        .await;

    let mut record = LogRecord {
        request_id: req_id.clone(),
        input_len: prompt_ids.len(),
        trace_output_len: row.decode,
        real_output_len: 0,
        status: "PENDING".to_string(),
        submit_timestamp,
        post_timestamp,
        total_duration_ms: 0.0,
        ttft_ms: None,
        avg_interval_ms: None,
        chunk_count: 0,
        target_ttft_ms: row.ttft,
        target_tpot_ms: row.tpot,
        slo_satisfied: None,
        slo_violations: None,
        slo_tokens_checked: None,
        slo_badness_ms: None,
        intervals: Vec::new(),
        output_text: String::new(),
        error: None,
    };

    match response_result {
        Ok(response) => {
            if !response.status().is_success() {
                record.status = "FAILED".to_string();
                record.error = Some(format!("HTTP {}", response.status()));
            } else {
                // 4. Stream Processing (The Critical Path)
                let mut stream = response.bytes_stream();
                let mut buffer = BytesMut::with_capacity(8192);

                let mut first_token_ts: Option<Instant> = None;
                let mut token_times: Vec<Instant> = Vec::with_capacity(row.decode);
                let mut output_text = String::new();
                let mut prev_token_count = 0;

                // Read stream chunks
                while let Some(chunk_res) = stream.next().await {
                    let chunk_arrival_time = Instant::now(); // Capture time IMMEDIATELY on packet arrival

                    match chunk_res {
                        Ok(chunk) => {
                            buffer.extend_from_slice(&chunk);

                            // Parse SSE lines manually
                            while let Some(idx) = buffer.iter().position(|&b| b == b'\n') {
                                let line_bytes = buffer.split_to(idx + 1);
                                let line = String::from_utf8_lossy(&line_bytes);

                                if line.starts_with("data: ") {
                                    let data_str = line.trim_start_matches("data: ").trim();
                                    if data_str == "[DONE]" {
                                        break;
                                    }

                                    // Try parse JSON
                                    if let Ok(json) = serde_json::from_str::<Value>(data_str) {
                                        chunk_count += 1;

                                        // Extract actual token count from response (matching Python)
                                        if let Some(meta_info) = json.get("meta_info") {
                                            if let Some(completion_tokens) = meta_info.get("completion_tokens") {
                                                if let Some(current_token_count) = completion_tokens.as_u64() {
                                                    let current_token_count = current_token_count as usize;

                                                    // Record time when we get NEW tokens
                                                    if current_token_count > prev_token_count {
                                                        let num_new_tokens = current_token_count - prev_token_count;

                                                        // Add token times for each new token
                                                        for _ in 0..num_new_tokens {
                                                            token_times.push(chunk_arrival_time);
                                                            if first_token_ts.is_none() {
                                                                first_token_ts = Some(chunk_arrival_time);
                                                            }
                                                        }

                                                        prev_token_count = current_token_count;
                                                    }
                                                }
                                            }
                                        }

                                        // Get output text from server (cumulative)
                                        if let Some(text) = json.get("text") {
                                            if let Some(text_str) = text.as_str() {
                                                output_text = text_str.to_string();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            record.error = Some(e.to_string());
                            break;
                        }
                    }
                }

                // Store output text (first 100 chars like Python)
                record.output_text = output_text.chars().take(100).collect();

                // 5. Calculate Stats
                let end_ts = Instant::now();
                record.total_duration_ms = (end_ts.duration_since(start_ts).as_secs_f64() * 1000.0 * 100.0).round() / 100.0;
                record.status = "SUCCESS".to_string();
                record.real_output_len = token_times.len();
                record.chunk_count = chunk_count;

                if let Some(ft) = first_token_ts {
                    record.ttft_ms = Some((ft.duration_since(start_ts).as_secs_f64() * 1000.0 * 100.0).round() / 100.0);
                }

                // Calculate inter-token intervals over ALL tokens (matching Python)
                let mut all_intervals: Vec<f64> = Vec::new();
                if token_times.len() >= 2 {
                    for i in 1..token_times.len() {
                        let interval_ms = token_times[i].duration_since(token_times[i-1]).as_secs_f64() * 1000.0;
                        all_intervals.push(interval_ms);
                    }
                }

                // Average over all tokens
                let avg_interval = if !all_intervals.is_empty() {
                    Some((all_intervals.iter().sum::<f64>() / all_intervals.len() as f64 * 100.0).round() / 100.0)
                } else {
                    None
                };
                record.avg_interval_ms = avg_interval;

                // Keep first 20 intervals for logging (matching Python)
                record.intervals = all_intervals.iter().take(20).map(|x| (x * 100.0).round() / 100.0).collect();

                // Precompute per-token elapsed times relative to request start
                let elapsed_ms_per_token: Vec<f64> = token_times
                    .iter()
                    .map(|t| t.duration_since(start_ts).as_secs_f64() * 1000.0)
                    .collect();

                // SLO check: verify each token i arrives before start_time + ttft + i * tpot
                if let (Some(ttft_limit), Some(tpot_limit)) = (row.ttft, row.tpot) {
                    if !elapsed_ms_per_token.is_empty() {
                        let slo_tokens_checked = elapsed_ms_per_token.len();
                        let mut violation_indices = Vec::new();
                        let mut max_delay_ms = 0.0; // Track maximum delay beyond deadline (in ms)
                        let mut delays_ms = Vec::with_capacity(slo_tokens_checked);

                        for (i, &elapsed_ms) in elapsed_ms_per_token.iter().enumerate() {
                            // Token i should arrive by: ttft + i * tpot (in ms since request start)
                            let deadline_ms = ttft_limit + (i as f64 * tpot_limit);
                            let delay_ms = elapsed_ms - deadline_ms;
                            if delay_ms > 0.0 {
                                violation_indices.push(i);
                                if delay_ms > max_delay_ms {
                                    max_delay_ms = delay_ms;
                                }
                            }
                            delays_ms.push(delay_ms);
                        }

                        record.slo_satisfied = Some(violation_indices.is_empty());
                        record.slo_violations = Some(violation_indices.len());
                        record.slo_tokens_checked = Some(slo_tokens_checked);
                        record.slo_badness_ms = Some((max_delay_ms * 100.0).round() / 100.0);

                        // Record SLO result to stats
                        let tier_label = format_slo_tier_label(row.tpot);
                        state.stats.record_slo_result(tier_label, violation_indices.is_empty()).await;
                    }
                }

                // Emit elapsed timeline for post-hoc analysis if configured
                if let Some(dump_tx) = state.elapsed_dump_tx.as_ref() {
                    if !elapsed_ms_per_token.is_empty() {
                        let timeline = TokenElapsedTimeline {
                            request_id: req_id.clone(),
                            elapsed_ms: elapsed_ms_per_token,
                        };
                        let _ = dump_tx.send(timeline).await;
                    }
                }
            }
        }
        Err(e) => {
            let end_ts = Instant::now();
            record.status = "FAILED".to_string();
            record.error = Some(e.to_string());
            record.total_duration_ms = (end_ts.duration_since(start_ts).as_secs_f64() * 1000.0 * 100.0).round() / 100.0;
        }
    }

    // Record completion
    let success = record.status == "SUCCESS";
    state.stats.record_complete(success);

    // Send to logger
    let _ = logger.send(record).await;
}

// --- Utilities ---

fn build_token_pool(text_file: &str, model_path: &str, limit: usize) -> Result<Vec<u32>> {
    use std::path::Path;

    // Try to load tokenizer - support both file paths and HuggingFace model names
    let tokenizer = if Path::new(model_path).exists() {
        // Load from file path
        Tokenizer::from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from file: {}", e))?
    } else if model_path.contains('/') {
        // Looks like a HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        // Download tokenizer.json from HuggingFace Hub
        println!("Downloading tokenizer from HuggingFace: {}", model_path);
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| anyhow::anyhow!("Failed to create HF API client: {}", e))?;
        let repo = api.model(model_path.to_string());
        let tokenizer_path = repo.get("tokenizer.json")
            .map_err(|e| anyhow::anyhow!("Failed to download tokenizer.json from HuggingFace: {}", e))?;

        Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load downloaded tokenizer: {}", e))?
    } else {
        // Try as file path anyway
        Tokenizer::from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer (not a file and not a HF model name): {}", e))?
    };

    let file = File::open(text_file)?;
    let reader = BufReader::new(file);
    let mut pool = Vec::with_capacity(limit);

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }

        let encoding = tokenizer.encode(line, false)
            .map_err(|e| anyhow::anyhow!("Encoding error: {}", e))?;

        pool.extend(encoding.get_ids());

        if pool.len() >= limit {
            pool.truncate(limit);
            break;
        }
    }
    Ok(pool)
}
