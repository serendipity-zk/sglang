use crate::schedulers::SloAwareScheduler;
use crate::server::AppState;
use dashmap::DashMap;
use std::collections::HashMap;
use std::io::{stdout, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// Centralized, lightweight CLI UI for live router stats.
///
/// This module maintains global counters and periodically redraws a
/// compact terminal UI. It avoids external dependencies and keeps
/// the code in a single place to allow future expansion.
pub struct RouterUi;

#[derive(Default)]
struct UiState {
    // Global counters
    generate_attempts: AtomicU64,
    total_generate: AtomicU64,
    finished_generate: AtomicU64,
    failed_generate: AtomicU64,
    mid_to_generate_immediate: AtomicU64,
    mid_not_immediate: AtomicU64,
    mid_to_generate_struggle: AtomicU64,
    pending_queue: AtomicU64,
    // Token bucket budget (stored as f64 bits)
    token_bucket_tokens_bits: AtomicU64,

    // Per-worker issued request counts
    worker_issued: DashMap<String, AtomicU64>,

    // Control flags
    running: AtomicBool,
}

static UI_STATE: OnceLock<Arc<UiState>> = OnceLock::new();
static UI_APPSTATE: OnceLock<Arc<AppState>> = OnceLock::new();

impl RouterUi {
    /// Initialize and start the UI render loop once.
    pub fn start(app_state: Arc<AppState>) {
        let state = UI_STATE
            .get_or_init(|| Arc::new(UiState::default()))
            .clone();

        // Only set and start once
        if state
            .running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            // Store app state and seed worker list so UI shows workers immediately
            let _ = UI_APPSTATE.set(app_state.clone());
            if let Some(ui_state) = UI_STATE.get() {
                for w in app_state.context.worker_registry.get_all() {
                    let url = w.url().to_string();
                    ui_state
                        .worker_issued
                        .entry(url)
                        .or_insert_with(|| AtomicU64::new(0));
                }
            }

            std::thread::spawn(move || {
                Self::render_loop_blocking();
            });
        }
    }

    /// Increment total generate requests
    pub fn inc_total_generate() {
        if let Some(state) = UI_STATE.get() {
            state.total_generate.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Increment generate attempts (before rate limiting)
    pub fn inc_generate_attempt() {
        if let Some(state) = UI_STATE.get() {
            state.generate_attempts.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Increment failed generate requests (rate limited/queue errors)
    pub fn inc_failed_generate() {
        if let Some(state) = UI_STATE.get() {
            state.failed_generate.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Increment requests that passed middleware to generate endpoint immediately
    pub fn inc_mid_to_generate_immediate() {
        if let Some(state) = UI_STATE.get() {
            state
                .mid_to_generate_immediate
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Increment requests that could not be processed immediately
    pub fn inc_mid_not_immediate() {
        if let Some(state) = UI_STATE.get() {
            state.mid_not_immediate.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Increment requests that struggled through queue to generate endpoint
    pub fn inc_mid_to_generate_struggle() {
        if let Some(state) = UI_STATE.get() {
            state
                .mid_to_generate_struggle
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Increment finished generate requests
    pub fn inc_finished_generate() {
        if let Some(state) = UI_STATE.get() {
            state.finished_generate.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Increment pending queue size (on enqueue)
    pub fn inc_queue() {
        if let Some(state) = UI_STATE.get() {
            state.pending_queue.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Decrement pending queue size (on dequeue or queue error)
    pub fn dec_queue() {
        if let Some(state) = UI_STATE.get() {
            state
                .pending_queue
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |curr| {
                    curr.checked_sub(1)
                })
                .ok();
        }
    }

    /// Increment per-worker issued counter
    pub fn inc_worker_issued(worker_url: &str) {
        if let Some(state) = UI_STATE.get() {
            let entry = state
                .worker_issued
                .entry(worker_url.to_string())
                .or_insert_with(|| AtomicU64::new(0));
            entry.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Set current available tokens in the rate limiter's token bucket
    pub fn set_token_bucket_tokens_available(tokens: f64) {
        if let Some(state) = UI_STATE.get() {
            let bits = tokens.to_bits();
            state
                .token_bucket_tokens_bits
                .store(bits, Ordering::Relaxed);
        }
    }

    /// Get SLO queue information from SLO-aware scheduler
    /// Returns Vec<(tier_idx, tpot_boundary, queue_size)> sorted by tier
    fn get_slo_queue_info(app_state: &AppState) -> Option<Vec<(usize, Option<f32>, usize)>> {
        // Get the scheduler (non-blocking)
        let scheduler = app_state.context.scheduler_registry.try_get_scheduler()?;

        // Check if it's SLO-aware
        if scheduler.name() != "slo_aware" {
            return None;
        }

        // Downcast to SloAwareScheduler
        let slo_scheduler = scheduler.as_any().downcast_ref::<SloAwareScheduler>()?;

        let tpot_buckets = &slo_scheduler.tpot_buckets;
        let queue_sizes = slo_scheduler.get_tier_queue_sizes();

        // Build list of (tier_idx, boundary, queue_size)
        let mut queue_info: Vec<(usize, Option<f32>, usize)> = queue_sizes
            .into_iter()
            .map(|(tier_idx, queue_size)| {
                let boundary = if tier_idx < tpot_buckets.len() {
                    Some(tpot_buckets[tier_idx])
                } else {
                    None // Idle tier
                };
                (tier_idx, boundary, queue_size)
            })
            .collect();

        // Sort by tier index
        queue_info.sort_by_key(|(tier_idx, _, _)| *tier_idx);

        Some(queue_info)
    }

    /// Get last scheduling iteration duration from SLO-aware scheduler
    /// Returns duration in microseconds, or None if not using SLO-aware scheduler
    fn get_schedule_duration_us(app_state: &AppState) -> Option<u64> {
        let scheduler = app_state.context.scheduler_registry.try_get_scheduler()?;

        if scheduler.name() != "slo_aware" {
            return None;
        }

        let slo_scheduler = scheduler.as_any().downcast_ref::<SloAwareScheduler>()?;
        Some(slo_scheduler.get_last_schedule_duration_us())
    }

    /// Get worker tier mapping from SLO-aware scheduler
    /// Returns HashMap<worker_url, tpot_boundary>
    fn get_worker_tier_map(app_state: &AppState) -> Option<HashMap<String, Option<f32>>> {
        // Get the scheduler (non-blocking)
        let scheduler = app_state.context.scheduler_registry.try_get_scheduler()?;

        // Check if it's SLO-aware
        if scheduler.name() != "slo_aware" {
            return None;
        }

        // Downcast to SloAwareScheduler
        let slo_scheduler = scheduler.as_any().downcast_ref::<SloAwareScheduler>()?;

        // Build reverse map: worker_url -> tpot_boundary
        let mut tier_map = HashMap::new();
        let tpot_buckets = &slo_scheduler.tpot_buckets;
        let worker_registry = &app_state.context.worker_registry;

        for tier_entry in slo_scheduler.tier_workers.iter() {
            let tier_idx = *tier_entry.key();
            let worker_ids = tier_entry.value();

            // Get TPOT boundary for this tier
            let tpot_boundary = if tier_idx < tpot_buckets.len() {
                Some(tpot_buckets[tier_idx])
            } else {
                None // Idle tier
            };

            // Map each worker ID to worker URL
            for worker_id in worker_ids {
                if let Some(worker) = worker_registry.get(worker_id) {
                    tier_map.insert(worker.url().to_string(), tpot_boundary);
                }
            }
        }

        Some(tier_map)
    }

    /// Extract port from worker URL (e.g., "http://localhost:31001" -> "31001")
    fn extract_port(url: &str) -> String {
        url.rsplit(':')
            .next()
            .and_then(|s| s.trim_end_matches('/').parse::<u16>().ok())
            .map(|p| p.to_string())
            .unwrap_or_else(|| "?".to_string())
    }

    /// Build worker ID mapping (A, B, C, ...) based on scheduler's internal traversal order
    /// Returns HashMap<worker_url, (worker_display_id, tier_position)>
    /// Display ID format: "A(31001)" - letter + port
    /// If SLO-aware scheduler is active, uses tier_workers order; otherwise uses alphabetical order
    fn build_worker_id_map(app_state: &AppState) -> HashMap<String, (String, usize)> {
        let mut worker_id_map = HashMap::new();
        let mut global_idx: u8 = 0;

        // Try to get SLO scheduler's tier order
        if let Some(scheduler) = app_state.context.scheduler_registry.try_get_scheduler() {
            if scheduler.name() == "slo_aware" {
                if let Some(slo_scheduler) = scheduler.as_any().downcast_ref::<SloAwareScheduler>()
                {
                    // Get all tiers in order
                    let mut tier_indices: Vec<usize> = slo_scheduler
                        .tier_workers
                        .iter()
                        .map(|entry| *entry.key())
                        .collect();
                    tier_indices.sort();

                    // For each tier, assign IDs in the order workers appear in tier_workers
                    for tier_idx in tier_indices {
                        if let Some(worker_ids) = slo_scheduler.tier_workers.get(&tier_idx) {
                            for (position, worker_id) in worker_ids.iter().enumerate() {
                                if let Some(worker) =
                                    app_state.context.worker_registry.get(worker_id)
                                {
                                    let letter = (b'A' + (global_idx % 26)) as char;
                                    let port = Self::extract_port(worker.url());
                                    worker_id_map.insert(
                                        worker.url().to_string(),
                                        (format!("{}({})", letter, port), position),
                                    );
                                    global_idx += 1;
                                }
                            }
                        }
                    }
                    return worker_id_map;
                }
            }
        }

        // Fallback: alphabetical order for non-SLO schedulers
        let mut worker_urls: Vec<String> = app_state
            .context
            .worker_registry
            .get_all_stats()
            .keys()
            .cloned()
            .collect();
        worker_urls.sort();

        for (idx, url) in worker_urls.into_iter().enumerate() {
            let letter = (b'A' + ((idx % 26) as u8)) as char;
            let port = Self::extract_port(&url);
            worker_id_map.insert(url, (format!("{}({})", letter, port), idx));
        }

        worker_id_map
    }

    /// Format worker metrics for display with aligned fields
    /// Format: "A(31001) [xx ms]  B:batch  T:tokens  KV:kv_tokens  P:prefill  L:iter_time  N:tier_counts" (SLO-aware)
    ///     or: "A(31001)          B:batch  T:tokens  KV:kv_tokens  P:prefill  L:iter_time" (non-SLO)
    /// tpot_buckets: list of all TPOT tier values (e.g., [20.0, 40.0, 80.0]) for showing all tier counts
    fn format_worker_metrics(
        worker_id: &str,
        stats: &crate::core::WorkerStats,
        tpot_boundary: Option<Option<f32>>,
        tpot_buckets: Option<&[f32]>,
    ) -> String {
        // Extract metrics
        let batch_size = stats.num_requests;
        let tokens = stats.batch_size_tokens;
        let kv_tokens = stats.kv_tokens_used.unwrap_or(0);

        // Format prefill chunk pairs as "current/cumulative"
        let prefill_str = match &stats.prefill_chunk_pairs {
            Some(pairs) if !pairs.is_empty() => {
                let (current, cumulative) = pairs[0];
                format!("{}/{}", current, cumulative)
            }
            _ => "0/0".to_string(),
        };

        let last_iter_str = match stats.last_iteration_time_ms {
            Some(ms) => format!("L:{:>7.2}ms", ms),
            None => "L:   --".to_string(),
        };

        // Format batch_size_by_tpot_tier as "N:count1,count2,..." for all tiers
        // Show counts for all tpot_buckets, defaulting to 0 for missing tiers
        let tier_counts_str = match (tpot_buckets, &stats.batch_size_by_tpot_tier) {
            (Some(buckets), Some(tier_map)) => {
                let counts: Vec<String> = buckets
                    .iter()
                    .map(|&tpot| {
                        let key = (tpot as i64).to_string();
                        // Try exact match first, then nearest key
                        let count = tier_map.get(&key).copied().unwrap_or_else(|| {
                            // Find nearest numeric key within 0.8x-1.2x range
                            let mut nearest: Option<(f64, i64)> = None;
                            for (k, &v) in tier_map.iter() {
                                if k == "none" {
                                    continue;
                                }
                                if let Ok(k_val) = k.parse::<f64>() {
                                    let ratio = k_val / tpot as f64;
                                    // Only consider if within 0.8x-1.2x range (20% tolerance)
                                    if ratio >= 0.8 && ratio <= 1.2 {
                                        let diff = (k_val - tpot as f64).abs();
                                        match nearest {
                                            None => nearest = Some((diff, v)),
                                            Some((best_diff, _)) if diff < best_diff => {
                                                nearest = Some((diff, v));
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            nearest.map(|(_, v)| v).unwrap_or(0)
                        });
                        count.to_string()
                    })
                    .collect();
                format!("N:{}", counts.join(","))
            }
            (Some(buckets), None) => {
                // No tier map, show all zeros
                let zeros: Vec<&str> = buckets.iter().map(|_| "0").collect();
                format!("N:{}", zeros.join(","))
            }
            _ => "N:-".to_string(),
        };

        // Build the formatted string with aligned columns
        // Worker ID is now format like "A(31001)" - about 8-9 chars
        match tpot_boundary {
            Some(boundary_opt) => {
                // SLO-aware scheduler
                let boundary_str = match boundary_opt {
                    Some(ms) => format!("[{:>6.1} ms]", ms),
                    None => "[  idle   ]".to_string(),
                };
                format!(
                    "{:<10} {}  B:{:<4} T:{:<7} KV:{:<7} P:{}  {}  {}",
                    worker_id,
                    boundary_str,
                    batch_size,
                    tokens,
                    kv_tokens,
                    prefill_str,
                    last_iter_str,
                    tier_counts_str
                )
            }
            None => {
                // Non-SLO scheduler (pad to match SLO spacing)
                format!(
                    "{:<10}              B:{:<4} T:{:<7} KV:{:<7} P:{}  {}",
                    worker_id, batch_size, tokens, kv_tokens, prefill_str, last_iter_str
                )
            }
        }
    }

    fn render_loop_blocking() {
        let interval = Duration::from_millis(500);
        loop {
            Self::draw_once();
            std::thread::sleep(interval);
        }
    }

    fn draw_once() {
        let state = match UI_STATE.get() {
            Some(s) => s,
            None => return,
        };

        // Clear screen and move cursor to home; lock stdout for the entire frame
        let out = stdout();
        let mut handle = out.lock();
        let _ = write!(handle, "\x1b[2J\x1b[H");

        let total = state.total_generate.load(Ordering::Relaxed);
        let finished = state.finished_generate.load(Ordering::Relaxed);
        let inflight = total.saturating_sub(finished);
        let failed = state.failed_generate.load(Ordering::Relaxed);

        let _ = writeln!(handle, "SGLang Router - Live Stats");
        let _ = writeln!(handle, "===========================");

        // Display scheduling iteration time if using SLO-aware scheduler
        if let Some(app_state) = UI_APPSTATE.get() {
            if let Some(duration_us) = Self::get_schedule_duration_us(app_state) {
                // Convert microseconds to milliseconds for display
                let duration_ms = duration_us as f64 / 1000.0;
                let _ = writeln!(handle, "Schedule Iteration Time: {:.3} ms", duration_ms);
            }
        }

        // Display per-SLO queue states if using SLO-aware scheduler
        if let Some(app_state) = UI_APPSTATE.get() {
            if let Some(slo_queue_info) = Self::get_slo_queue_info(app_state) {
                let _ = writeln!(handle, "");
                let _ = writeln!(handle, "Per-SLO Queue States:");
                let num_slo_tiers = slo_queue_info
                    .iter()
                    .filter(|(_, boundary, _)| boundary.is_some())
                    .count();

                // Collect all SLO boundaries for range calculation
                let mut slo_boundaries: Vec<f32> = slo_queue_info
                    .iter()
                    .filter_map(|(_, boundary, _)| *boundary)
                    .collect();
                slo_boundaries
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                for (tier_idx, boundary, queue_size) in &slo_queue_info {
                    if let Some(ms) = boundary {
                        // SLO tier - format with boundary range
                        let range_desc = if *tier_idx == 0 {
                            format!("≤{:.1} ms", ms)
                        } else {
                            // Find previous boundary by position in sorted list
                            let pos = slo_boundaries
                                .iter()
                                .position(|&b| (b - ms).abs() < 0.01)
                                .unwrap_or(0);
                            let prev_boundary = if pos > 0 {
                                slo_boundaries[pos - 1]
                            } else {
                                0.0
                            };
                            format!(">{:.1} ms, ≤{:.1} ms", prev_boundary, ms)
                        };
                        let _ = writeln!(
                            handle,
                            "  Queue {} ({}): {} requests",
                            tier_idx, range_desc, queue_size
                        );
                    } else if *tier_idx == num_slo_tiers {
                        // Idle tier
                        let _ = writeln!(
                            handle,
                            "  Queue {} (idle/autoscaling): {} requests",
                            tier_idx, queue_size
                        );
                    }
                }
                let _ = writeln!(handle, "");
            }
        }

        let _ = writeln!(handle, "Accepted Generate: {}", total);
        let _ = writeln!(handle, "Finished Generate: {}", finished);
        let _ = writeln!(handle, "Failed: {}", failed);
        let _ = writeln!(handle, "In-Flight: {}", inflight);
        let _ = writeln!(handle, "");

        // Get app state and worker stats
        let app_state = match UI_APPSTATE.get() {
            Some(s) => s,
            None => {
                let _ = writeln!(handle, "Per-Worker Stats:");
                let _ = writeln!(handle, "------------------");
                let _ = writeln!(handle, "(app state not initialized)");
                let _ = handle.flush();
                return;
            }
        };

        // Get tier mapping (if using SLO-aware scheduler)
        let tier_map = Self::get_worker_tier_map(app_state);

        // Get all worker stats
        let worker_stats = app_state.context.worker_registry.get_all_stats();

        // Build worker ID mapping (S0, S1, S2, ...) based on scheduler's traversal order
        let worker_id_map = Self::build_worker_id_map(app_state);

        // Build reverse tier map for sorting: worker_url -> tier_index
        // Also extract tpot_buckets for displaying tier counts
        let (worker_to_tier, tpot_buckets): (HashMap<String, usize>, Option<Vec<f32>>) =
            if let Some(scheduler) = app_state.context.scheduler_registry.try_get_scheduler() {
                if scheduler.name() == "slo_aware" {
                    if let Some(slo_scheduler) =
                        scheduler.as_any().downcast_ref::<SloAwareScheduler>()
                    {
                        let mut map = HashMap::new();
                        for tier_entry in slo_scheduler.tier_workers.iter() {
                            let tier_idx = *tier_entry.key();
                            for worker_id in tier_entry.value() {
                                if let Some(worker) =
                                    app_state.context.worker_registry.get(worker_id)
                                {
                                    map.insert(worker.url().to_string(), tier_idx);
                                }
                            }
                        }
                        (map, Some(slo_scheduler.tpot_buckets.clone()))
                    } else {
                        (HashMap::new(), None)
                    }
                } else {
                    (HashMap::new(), None)
                }
            } else {
                (HashMap::new(), None)
            };

        // Build rows with formatted metrics and sorting keys
        let mut rows: Vec<(String, usize, usize)> = Vec::new(); // (metrics_str, tier_index, tier_position)
        for (worker_url, stats) in worker_stats.iter() {
            let (worker_id, tier_position) = worker_id_map
                .get(worker_url)
                .map(|(id, pos)| (id.as_str(), *pos))
                .unwrap_or(("?", usize::MAX));
            let tpot_boundary = tier_map
                .as_ref()
                .and_then(|map| map.get(worker_url).copied());
            let metrics_str = Self::format_worker_metrics(
                worker_id,
                stats,
                tpot_boundary,
                tpot_buckets.as_deref(),
            );
            let tier_idx = worker_to_tier
                .get(worker_url)
                .copied()
                .unwrap_or(usize::MAX); // Unknown tier goes last
            rows.push((metrics_str, tier_idx, tier_position));
        }

        // Sort by tier first, then by position within tier (matching scheduler's traversal order)
        rows.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2)));

        let _ = writeln!(handle, "Per-Worker Stats:");
        let _ = writeln!(handle, "------------------");
        if rows.is_empty() {
            let _ = writeln!(handle, "(no workers registered)");
        } else {
            for (metrics, _tier_idx, _worker_id) in rows.iter().take(50) {
                let _ = writeln!(handle, "{}", metrics);
            }
        }

        let _ = handle.flush();
    }
}
