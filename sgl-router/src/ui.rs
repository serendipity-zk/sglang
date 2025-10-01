use crate::server::AppState;
use dashmap::DashMap;
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
            state.mid_to_generate_immediate.fetch_add(1, Ordering::Relaxed);
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
            state.mid_to_generate_struggle.fetch_add(1, Ordering::Relaxed);
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
        let mut out = stdout();
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

        // Build rows only from counters to minimize locks/work per frame
        let mut rows: Vec<(String, u64)> = Vec::new();
        for kv in state.worker_issued.iter() {
            rows.push((kv.key().clone(), kv.value().load(Ordering::Relaxed)));
        }

        // Sort by count desc, then URL
        rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let _ = writeln!(handle, "Per-Worker Issued Requests:");
        let _ = writeln!(handle, "---------------------------");
        if rows.is_empty() {
            let _ = writeln!(handle, "(no workers registered)");
        } else {
            for (url, count) in rows.iter().take(50) {
                let _ = writeln!(handle, "{:>8}  {}", count, url);
            }
        }

        let _ = handle.flush();
    }
}
