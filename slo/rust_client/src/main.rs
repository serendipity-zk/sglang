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
use tokio::sync::{mpsc, RwLock};
use tokio::time::timeout;
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

    #[arg(long, default_value = "output.ans")]
    ans_log_path: String,

    /// Use server detokenize timestamps instead of client receive timestamps for SLO judgment
    #[arg(long, default_value_t = false)]
    slo_use_detokenize_time: bool,
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
    server_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    iteration_id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    start_iteration: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    iteration_id_missing: Option<bool>,
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
    detokenize_timestamps: Vec<Option<f64>>,
    server_id: Option<String>,
    start_iteration: Option<usize>,
    iteration_ids: Vec<Option<usize>>,
}

async fn log_ans(ans_tx: &Option<mpsc::Sender<String>>, msg: impl Into<String>) {
    if let Some(tx) = ans_tx {
        let _ = tx.send(msg.into()).await;
    }
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
    ans_tx: Option<tokio::sync::mpsc::Sender<String>>,
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
    ans_tx: Option<mpsc::Sender<String>>,
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
        } else if let Some(tx) = ans_tx.as_ref() {
            // Non-TTY: emit periodic summary to ans log instead of stdout
            let summary = lines.join(" | ");
            let _ = tx.try_send(summary);
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
    // Prepare ans log channel early (stdout may be hidden)
    let args = Args::parse();
    let (ans_tx, mut ans_rx) = mpsc::channel::<String>(10_000);
    let ans_log_path = args.ans_log_path.clone();
    tokio::spawn(async move {
        let file = File::create(&ans_log_path).expect("Failed to create ans log file");
        let mut writer = std::io::BufWriter::with_capacity(256 * 1024, file);
        while let Some(line) = ans_rx.recv().await {
            writeln!(writer, "{}", line).ok();
        }
        writer.flush().ok();
    });

    // 1. Optimize OS Limits (Important for 10k connections)
    match fdlimit::raise_fd_limit() {
        Ok(_) => log_ans(&Some(ans_tx.clone()), "OS File Descriptor limit raised successfully").await,
        Err(e) => log_ans(&Some(ans_tx.clone()), format!("Failed to raise FD limit: {}. Ensure `ulimit -n` is high.", e)).await,
    }

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
        let ans_tx_for_timeline = Some(ans_tx.clone());
        tokio::spawn(async move {
            let file = File::create(&path).expect("Failed to create elapsed dump file");
            let mut writer = std::io::BufWriter::with_capacity(1024 * 1024, file);
            let opts = pickle::SerOptions::new();
            while let Some(timeline) = rx.recv().await {
                if let Err(e) = pickle::to_writer(&mut writer, &timeline, opts.clone()) {
                    if let Some(tx) = ans_tx_for_timeline.as_ref() {
                        let _ = tx.send(format!("Failed to write elapsed timeline: {}", e)).await;
                    }
                }
            }
            writer.flush().ok();
        });
        Some(tx)
    } else {
        None
    };

    // 3. Build Components
    log_ans(&Some(ans_tx.clone()), "Loading tokenizer and building token pool...").await;
    let token_pool = build_token_pool(&args.text_file, &args.tokenizer, 200_000)?;
    log_ans(&Some(ans_tx.clone()), format!("Token pool size: {}", token_pool.len())).await;

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
        ans_tx: Some(ans_tx.clone()),
    });

    // 4. Load and Sort Trace
    log_ans(&Some(ans_tx.clone()), "Loading trace...").await;
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
    log_ans(&Some(ans_tx.clone()), format!("Starting trace replay: {} requests", total_reqs)).await;

    // 5. The Dispatch Loop
    let start_time = Instant::now();

    // Spawn status display task
    let stats_clone = stats.clone();
    let log_path_clone = args.log_path.clone();
    let ans_tx_clone = state.ans_tx.clone();
    tokio::spawn(async move {
        status_display_task(stats_clone, total_reqs, log_path_clone, start_time, ans_tx_clone).await;
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

    log_ans(&Some(ans_tx.clone()), "All requests completed. Log saved.").await;

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
    // This will be our single source of truth for all time measurements
    let submit_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    // Use SystemTime for all measurements to ensure consistency
    let start_ts = SystemTime::now();

    let mut rng = rand::rngs::StdRng::from_entropy();

    // 1. Sample Prompt
    let pool_len = state.token_pool.len();
    let start_idx = rng.gen_range(0..pool_len.saturating_sub(row.prefill));
    let prompt_ids = &state.token_pool[start_idx..start_idx + row.prefill];

    // Use submit_timestamp as arrival_time_ms for consistency
    // This ensures client and server use the same time base
    let arrival_time_ms = submit_timestamp * 1000.0;

    // 2. Build Payload
    let mut payload = serde_json::json!({
        "input_ids": [prompt_ids],
        "sampling_params": {
            "max_new_tokens": row.decode,
            "temperature": state.args.temperature,
            "ignore_eos": true
        },
        "stream": true,
        "arrival_time_ms": arrival_time_ms
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
        server_id: None,
        iteration_id: None,
        start_iteration: None,
        iteration_id_missing: None,
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
                // Count failed requests as SLO violations if SLO targets were set
                if row.tpot.is_some() {
                    let tier_label = format_slo_tier_label(row.tpot);
                    record.slo_satisfied = Some(false);
                    record.slo_violations = Some(1);
                    record.slo_tokens_checked = Some(0);
                    state.stats.record_slo_result(tier_label, false).await;
                }
            } else {
                // 4. Stream Processing (The Critical Path)
                let mut stream = response.bytes_stream();
                let mut buffer = BytesMut::with_capacity(8192);

                let mut first_token_ts: Option<SystemTime> = None;
                let mut token_times: Vec<SystemTime> = Vec::with_capacity(row.decode);
                let mut token_iteration_ids: Vec<Option<usize>> = Vec::with_capacity(row.decode);
                let mut token_detokenize_timestamps: Vec<Option<f64>> = Vec::with_capacity(row.decode);
                let mut server_id: Option<String> = None;
                let mut start_iteration: Option<usize> = None;
                let mut last_iteration_id: Option<usize> = None;
                let mut iteration_seen = false;
                let mut output_text = String::new();
                let mut prev_token_count = 0;

                // Read stream chunks with idle timeout (10 seconds)
                const IDLE_TIMEOUT_SECS: u64 = 10;
                let idle_timeout = Duration::from_secs(IDLE_TIMEOUT_SECS);

                loop {
                    match timeout(idle_timeout, stream.next()).await {
                        Ok(Some(chunk_res)) => {
                            let chunk_arrival_time = SystemTime::now(); // Capture time IMMEDIATELY on packet arrival

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
                                                    if server_id.is_none() {
                                                        if let Some(id_val) = meta_info.get("server_id").and_then(|v| v.as_str()) {
                                                            server_id = Some(id_val.to_string());
                                                        }
                                                    }
                                                    if start_iteration.is_none() {
                                                        if let Some(start_iter) = meta_info.get("start_iteration").and_then(|v| v.as_u64()) {
                                                            start_iteration = Some(start_iter as usize);
                                                        }
                                                    }
                                                    let iteration_id = meta_info.get("iteration_id").and_then(|v| v.as_u64()).map(|v| v as usize);
                                                    if iteration_id.is_some() {
                                                        iteration_seen = true;
                                                        last_iteration_id = iteration_id;
                                                    }
                                                    let token_iteration_tag = iteration_id.or(last_iteration_id);

                                                    // Extract detokenize timestamp
                                                    let detokenize_timestamp = meta_info.get("detokenize_timestamp").and_then(|v| v.as_f64());

                                                    if let Some(completion_tokens) = meta_info.get("completion_tokens") {
                                                        if let Some(current_token_count) = completion_tokens.as_u64() {
                                                            let current_token_count = current_token_count as usize;

                                                            // Record time when we get NEW tokens
                                                            if current_token_count > prev_token_count {
                                                                let num_new_tokens = current_token_count - prev_token_count;

                                                                // Add token times for each new token
                                                                for _ in 0..num_new_tokens {
                                                                    token_times.push(chunk_arrival_time);
                                                                    token_iteration_ids.push(token_iteration_tag);
                                                                    token_detokenize_timestamps.push(detokenize_timestamp);
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
                                    record.status = "FAILED".to_string();
                                    record.error = Some(e.to_string());
                                    break;
                                }
                            }
                        }
                        Ok(None) => {
                            // Stream ended normally
                            break;
                        }
                        Err(_) => {
                            // Timeout - no data received for 10 seconds
                            record.status = "FAILED".to_string();
                            record.error = Some("Token idle timeout after 10s".to_string());
                            break;
                        }
                    }
                }

                // Store output text (first 100 chars like Python)
                record.output_text = output_text.chars().take(100).collect();

                // 5. Calculate Stats
                let end_ts = SystemTime::now();
                record.total_duration_ms = (end_ts.duration_since(start_ts).unwrap_or_default().as_secs_f64() * 1000.0 * 100.0).round() / 100.0;
                // Only mark SUCCESS if no error occurred (timeout or stream error)
                if record.error.is_none() {
                    record.status = "SUCCESS".to_string();
                }
                record.real_output_len = token_times.len();
                record.chunk_count = chunk_count;

                if let Some(ft) = first_token_ts {
                    record.ttft_ms = Some((ft.duration_since(start_ts).unwrap_or_default().as_secs_f64() * 1000.0 * 100.0).round() / 100.0);
                }

                // Calculate inter-token intervals over ALL tokens (matching Python)
                let mut all_intervals: Vec<f64> = Vec::new();
                if token_times.len() >= 2 {
                    for i in 1..token_times.len() {
                        let interval_ms = token_times[i].duration_since(token_times[i-1]).unwrap_or_default().as_secs_f64() * 1000.0;
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
                // Use detokenize timestamps if flag is set, otherwise use receive timestamps
                let submit_ms = submit_timestamp * 1000.0;
                let elapsed_ms_per_token: Vec<f64> = if state.args.slo_use_detokenize_time {
                    // Use server detokenize timestamps for SLO calculation
                    // Fall back to receive time if detokenize timestamp is missing
                    token_detokenize_timestamps
                        .iter()
                        .zip(token_times.iter())
                        .map(|(opt_detok_ts, recv_time)| {
                            if let Some(detok_ts) = opt_detok_ts {
                                // detok_ts is in ms since epoch, submit_ms is also in ms since epoch
                                detok_ts - submit_ms
                            } else {
                                // Fallback to receive time
                                recv_time.duration_since(start_ts).unwrap_or_default().as_secs_f64() * 1000.0
                            }
                        })
                        .collect()
                } else {
                    // Use client receive timestamps (existing behavior)
                    token_times
                        .iter()
                        .map(|t| t.duration_since(start_ts).unwrap_or_default().as_secs_f64() * 1000.0)
                        .collect()
                };

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

                record.server_id = server_id.clone();
                record.iteration_id = last_iteration_id;
                record.start_iteration = start_iteration;

                if !iteration_seen {
                    record.iteration_id_missing = Some(true);
                    log_ans(
                        &state.ans_tx,
                        format!(
                            "[warn] request {} missing iteration_id (server_id={}, start_iteration={:?}, tokens={})",
                            req_id,
                            server_id.clone().unwrap_or_else(|| "<unknown>".to_string()),
                            start_iteration,
                            token_times.len()
                        ),
                    )
                    .await;
                }

                // Emit elapsed timeline for post-hoc analysis if configured
                if let Some(dump_tx) = state.elapsed_dump_tx.as_ref() {
                    if !elapsed_ms_per_token.is_empty() {
                        let timeline = TokenElapsedTimeline {
                            request_id: req_id.clone(),
                            elapsed_ms: elapsed_ms_per_token,
                            detokenize_timestamps: token_detokenize_timestamps,
                            server_id: server_id.clone(),
                            start_iteration,
                            iteration_ids: token_iteration_ids,
                        };
                        let _ = dump_tx.send(timeline).await;
                    }
                }
            }
        }
        Err(e) => {
            let end_ts = SystemTime::now();
            record.status = "FAILED".to_string();
            record.error = Some(e.to_string());
            record.total_duration_ms = (end_ts.duration_since(start_ts).unwrap_or_default().as_secs_f64() * 1000.0 * 100.0).round() / 100.0;
            // Count failed requests as SLO violations if SLO targets were set
            if row.tpot.is_some() {
                let tier_label = format_slo_tier_label(row.tpot);
                record.slo_satisfied = Some(false);
                record.slo_violations = Some(1);
                record.slo_tokens_checked = Some(0);
                state.stats.record_slo_result(tier_label, false).await;
            }
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
