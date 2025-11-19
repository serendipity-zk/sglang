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

    /// Build worker ID mapping (S0, S1, S2, ...) based on alphabetical URL order
    /// Returns HashMap<worker_url, worker_display_id>
    fn build_worker_id_map(worker_urls: &[String]) -> HashMap<String, String> {
        let mut urls = worker_urls.to_vec();
        urls.sort();

        urls.into_iter()
            .enumerate()
            .map(|(idx, url)| (url, format!("S{}", idx)))
            .collect()
    }

    /// Format worker metrics for display with aligned fields
    /// Format: "Sx [xx ms]  B:batch  T:tokens  KV:kv_tokens  P:prefill  L:iter_time" (SLO-aware)
    ///     or: "Sx          B:batch  T:tokens  KV:kv_tokens  P:prefill  L:iter_time" (non-SLO)
    fn format_worker_metrics(
        worker_id: &str,
        stats: &crate::core::WorkerStats,
        tpot_boundary: Option<Option<f32>>,
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

        // Build the formatted string with aligned columns
        match tpot_boundary {
            Some(boundary_opt) => {
                // SLO-aware scheduler
                let boundary_str = match boundary_opt {
                    Some(ms) => format!("[{:>6.1} ms]", ms),
                    None => "[  idle   ]".to_string(),
                };
                format!(
                    "{:<4} {}  B:{:<4} T:{:<7} KV:{:<7} P:{}  {}",
                    worker_id, boundary_str, batch_size, tokens, kv_tokens, prefill_str, last_iter_str
                )
            }
            None => {
                // Non-SLO scheduler (pad to match SLO spacing)
                format!(
                    "{:<4}              B:{:<4} T:{:<7} KV:{:<7} P:{}  {}",
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
        let attempts = state.generate_attempts.load(Ordering::Relaxed);
        let failed = state.failed_generate.load(Ordering::Relaxed);
        let mid_immediate = state.mid_to_generate_immediate.load(Ordering::Relaxed);
        let mid_not_immediate = state.mid_not_immediate.load(Ordering::Relaxed);
        let mid_struggle = state.mid_to_generate_struggle.load(Ordering::Relaxed);
        let pending = state.pending_queue.load(Ordering::Relaxed);
        let token_budget = f64::from_bits(state.token_bucket_tokens_bits.load(Ordering::Relaxed));

        let _ = writeln!(handle, "SGLang Router - Live Stats");
        let _ = writeln!(handle, "===========================");
        let _ = writeln!(handle, "Pending Queue: {}", pending);
        let _ = writeln!(handle, "Token Budget: {:.2}", token_budget);
        let _ = writeln!(handle, "Generate Attempts: {}", attempts);
        let _ = writeln!(handle, "Mid Immediate: {}", mid_immediate);
        let _ = writeln!(handle, "Mid Not Immediate: {}", mid_not_immediate);
        let _ = writeln!(handle, "Mid Struggle: {}", mid_struggle);
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

        // Build worker ID mapping (S0, S1, S2, ...)
        let worker_urls: Vec<String> = worker_stats.keys().cloned().collect();
        let worker_id_map = Self::build_worker_id_map(&worker_urls);

        // Build reverse tier map for sorting: worker_url -> tier_index
        let worker_to_tier: HashMap<String, usize> = if let Some(scheduler) = app_state.context.scheduler_registry.try_get_scheduler() {
            if scheduler.name() == "slo_aware" {
                if let Some(slo_scheduler) = scheduler.as_any().downcast_ref::<SloAwareScheduler>() {
                    let mut map = HashMap::new();
                    for tier_entry in slo_scheduler.tier_workers.iter() {
                        let tier_idx = *tier_entry.key();
                        for worker_id in tier_entry.value() {
                            if let Some(worker) = app_state.context.worker_registry.get(worker_id) {
                                map.insert(worker.url().to_string(), tier_idx);
                            }
                        }
                    }
                    map
                } else {
                    HashMap::new()
                }
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        };

        // Build rows with formatted metrics and sorting keys
        let mut rows: Vec<(String, usize, String)> = Vec::new(); // (metrics_str, tier_index, worker_id)
        for (worker_url, stats) in worker_stats.iter() {
            let worker_id = worker_id_map.get(worker_url).map(|s| s.as_str()).unwrap_or("S?");
            let tpot_boundary = tier_map.as_ref().and_then(|map| map.get(worker_url).copied());
            let metrics_str = Self::format_worker_metrics(worker_id, stats, tpot_boundary);
            let tier_idx = worker_to_tier.get(worker_url).copied().unwrap_or(usize::MAX); // Unknown tier goes last
            rows.push((metrics_str, tier_idx, worker_id.to_string()));
        }

        // Sort by tier first, then by server ID (S0, S1, S2, ...)
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
