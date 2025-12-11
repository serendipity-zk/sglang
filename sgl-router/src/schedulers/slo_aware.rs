use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use rand::seq::SliceRandom;
use tokio::sync::mpsc;
use tokio::time::{interval, MissedTickBehavior};
use tracing::{info, warn};

use crate::config::types::{AutoScalingConfig, WorkerSelectionPolicy};
use crate::core::{Worker, WorkerId, WorkerStats};
use crate::routers::http::scheduler::{PendingRequest, SchedulerConfig};
use crate::ui::RouterUi;

use super::{
    available_workers_for_request, has_request_timed_out, Scheduler, SchedulerBase,
    SCHEDULER_TICK_INTERVAL_MS,
};

#[derive(Debug)]
pub struct SloAwareScheduler {
    pub tpot_buckets: Vec<f32>,
    /// Worker assignments to SLO tiers + idle tier
    /// Tiers 0..tpot_buckets.len() are SLO tiers, tier tpot_buckets.len() is idle tier
    pub tier_workers: Arc<DashMap<usize, Vec<WorkerId>>>,
    /// Cached worker stats for admission control
    worker_stats: RwLock<HashMap<String, WorkerStats>>,
    /// Per-tier queue sizes for UI display
    tier_queue_sizes: Arc<DashMap<usize, AtomicUsize>>,
    /// Last scheduling iteration duration in microseconds
    last_schedule_duration_us: AtomicU64,
    /// Worker selection policy configuration
    policy: WorkerSelectionPolicy,
    /// Auto-scaling configuration (if enabled)
    auto_scaling: Option<AutoScalingConfig>,
    /// Initial tier allocation configuration
    initial_tier_allocation: Option<Vec<usize>>,
    /// Track last known health state of workers (worker_url -> is_healthy)
    worker_health_state: Arc<DashMap<String, bool>>,
    /// Enable periodic TPOT updates (500ms interval)
    send_tpot_updates: bool,
    /// Shared HTTP client for TPOT updates (reused to avoid connection pool overhead)
    tpot_client: reqwest::Client,
    /// Persistent worker URL -> letter mapping for logging (A, B, C, ...)
    worker_letters: Arc<DashMap<String, char>>,
    /// Next letter to assign to new workers
    next_letter: Arc<std::sync::atomic::AtomicU8>,
    /// Track when each worker last had a request scheduled to it
    last_scheduled_time: Arc<DashMap<String, std::time::Instant>>,
    /// Track TTFT violations per tier (for steal decision)
    tier_violations: Arc<DashMap<usize, AtomicUsize>>,
}

impl SloAwareScheduler {
    pub fn new(
        _policy_registry: Arc<crate::policies::PolicyRegistry>,
        tpot_buckets: Vec<f32>,
        policy: WorkerSelectionPolicy,
        auto_scaling: Option<AutoScalingConfig>,
        initial_tier_allocation: Option<Vec<usize>>,
        send_tpot_updates: bool,
    ) -> Self {
        let tier_workers = Arc::new(DashMap::new());
        let tier_queue_sizes = Arc::new(DashMap::new());
        let tier_violations = Arc::new(DashMap::new());

        // Initialize empty tier assignments and queue size counters
        // SLO tiers: 0..tpot_buckets.len()-1
        // Idle tier: tpot_buckets.len()
        // Example: [10, 50] creates tiers 0, 1 (SLO) and 2 (idle)
        for tier_idx in 0..=tpot_buckets.len() {
            tier_workers.insert(tier_idx, Vec::new());
            tier_queue_sizes.insert(tier_idx, AtomicUsize::new(0));
            tier_violations.insert(tier_idx, AtomicUsize::new(0));
        }

        Self {
            tpot_buckets,
            tier_workers,
            worker_stats: RwLock::new(HashMap::new()),
            tier_queue_sizes,
            last_schedule_duration_us: AtomicU64::new(0),
            policy,
            auto_scaling,
            initial_tier_allocation,
            worker_health_state: Arc::new(DashMap::new()),
            send_tpot_updates,
            tpot_client: reqwest::Client::new(),
            worker_letters: Arc::new(DashMap::new()),
            next_letter: Arc::new(std::sync::atomic::AtomicU8::new(0)),
            last_scheduled_time: Arc::new(DashMap::new()),
            tier_violations,
        }
    }

    /// Record that a request was scheduled to this worker
    fn record_scheduled(&self, worker_url: &str) {
        self.last_scheduled_time.insert(worker_url.to_string(), std::time::Instant::now());
    }

    /// Record a TTFT violation for a tier
    fn record_violation(&self, tier_idx: usize) {
        if let Some(counter) = self.tier_violations.get(&tier_idx) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Check if tier has recent violations and reset counter
    fn has_violations_and_reset(&self, tier_idx: usize) -> bool {
        if let Some(counter) = self.tier_violations.get(&tier_idx) {
            let violations = counter.swap(0, Ordering::Relaxed);
            violations > 0
        } else {
            false
        }
    }

    /// Check if worker has been idle longer than threshold
    fn is_worker_idle_for(&self, worker_url: &str, threshold_ms: u64) -> bool {
        if let Some(last_time) = self.last_scheduled_time.get(worker_url) {
            last_time.elapsed().as_millis() as u64 > threshold_ms
        } else {
            true // Never scheduled = considered idle
        }
    }

    /// Get or assign a letter for a worker URL (A, B, C, ... Z, then wraps)
    fn get_worker_letter(&self, worker_url: &str) -> char {
        if let Some(letter) = self.worker_letters.get(worker_url) {
            return *letter;
        }
        // Assign new letter
        let idx = self.next_letter.fetch_add(1, Ordering::Relaxed);
        let letter = (b'A' + (idx % 26)) as char;
        self.worker_letters.insert(worker_url.to_string(), letter);
        letter
    }

    fn num_buckets(&self) -> usize {
        self.tpot_buckets.len()
    }

    /// Get queue index for a request based on its target TPOT
    /// Returns None if request should be rejected (no target or exceeds max boundary)
    fn get_queue_index(&self, tpot: Option<f32>) -> Option<usize> {
        let Some(tpot_value) = tpot else {
            // No target_tpot_ms specified - reject request
            return None;
        };

        // Find the first boundary where tpot <= boundary
        for (idx, boundary) in self.tpot_buckets.iter().enumerate() {
            if tpot_value <= *boundary {
                return Some(idx);
            }
        }

        // Request exceeds all boundaries - reject
        None
    }

    /// Initialize workers into SLO tiers
    /// Uses initial_tier_allocation from config if provided, otherwise distributes evenly via round-robin
    /// Idle tier (tpot_buckets.len()) is left empty for future autoscaling
    fn initialize_workers(&self, worker_registry: &Arc<crate::core::WorkerRegistry>) {
        let all_workers = worker_registry.get_all_with_ids();
        let num_slo_tiers = self.tpot_buckets.len();

        if all_workers.is_empty() {
            warn!("No workers available for tier initialization");
            return;
        }

        // Use configured tier allocation if provided, otherwise use round-robin
        match &self.initial_tier_allocation {
            Some(tier_allocation) => {
                info!(
                    "Initializing {} workers into {} SLO tiers using configured allocation {:?}",
                    all_workers.len(),
                    num_slo_tiers,
                    tier_allocation
                );

                // Assign workers to tiers based on configured allocation
                // If not enough workers, fill from beginning and leave rest empty
                let mut worker_iter = all_workers.iter();

                for (tier_idx, &count) in tier_allocation.iter().enumerate() {
                    if tier_idx >= num_slo_tiers {
                        break; // Don't exceed available tiers
                    }
                    for _ in 0..count {
                        if let Some((worker_id, _worker)) = worker_iter.next() {
                            self.tier_workers
                                .entry(tier_idx)
                                .or_insert_with(Vec::new)
                                .push(worker_id.clone());
                        } else {
                            // No more workers available
                            break;
                        }
                    }
                }
            }
            None => {
                info!(
                    "Initializing {} workers into {} SLO tiers using round-robin distribution",
                    all_workers.len(),
                    num_slo_tiers
                );

                // Round-robin assignment across SLO tiers
                for (idx, (worker_id, _worker)) in all_workers.iter().enumerate() {
                    let tier_idx = idx % num_slo_tiers;
                    self.tier_workers
                        .entry(tier_idx)
                        .or_insert_with(Vec::new)
                        .push(worker_id.clone());
                }
            }
        }

        // Log tier assignments (each tier is an upper bound)
        for tier_idx in 0..num_slo_tiers {
            if let Some(workers) = self.tier_workers.get(&tier_idx) {
                let boundary = self.tpot_buckets[tier_idx];
                let range_desc = if tier_idx == 0 {
                    format!("≤{} ms", boundary)
                } else {
                    let prev_boundary = self.tpot_buckets[tier_idx - 1];
                    format!(">{} ms, ≤{} ms", prev_boundary, boundary)
                };
                info!(
                    "  Tier {} ({}): {} workers",
                    tier_idx,
                    range_desc,
                    workers.len()
                );
            }
        }

        // Log idle tier (should be empty initially)
        if let Some(idle_workers) = self.tier_workers.get(&num_slo_tiers) {
            info!(
                "  Tier {} (idle/autoscaling): {} workers",
                num_slo_tiers,
                idle_workers.len()
            );
        }
    }

    /// Update worker statistics for admission control
    fn update_worker_stats(&self, stats: &HashMap<String, WorkerStats>) {
        if let Ok(mut cached) = self.worker_stats.write() {
            *cached = stats.clone();
        }
    }

    /// Check if worker has pending work (queued + pending dispatches)
    /// Returns true if worker is busy and should not receive new requests
    fn has_pending_work(&self, worker: &dyn Worker) -> bool {
        // Check worker-reported queue size
        let queue_size = if let Ok(stats) = self.worker_stats.read() {
            if let Some(worker_stats) = stats.get(worker.url()) {
                worker_stats.waiting_queue_size
            } else {
                0
            }
        } else {
            0
        };

        // Check router-tracked pending messages
        let pending_messages = worker.pending_message_count();

        // Worker is busy if either queue has work or pending messages exist
        (queue_size + pending_messages as i64) > 0
    }

    /// Check if worker is truly idle (no running requests, no queued, no pending)
    /// Used for deciding whether to move worker to idle tier
    fn is_worker_truly_idle(&self, worker: &dyn Worker) -> bool {
        if let Ok(stats) = self.worker_stats.read() {
            if let Some(worker_stats) = stats.get(worker.url()) {
                let num_requests = worker_stats.num_requests;
                let queue_size = worker_stats.waiting_queue_size;
                let pending_messages = worker.pending_message_count() as i64;

                // Worker is truly idle only if ALL are zero
                return num_requests == 0 && queue_size == 0 && pending_messages == 0;
            }
        }
        false // If we can't get stats, assume not idle
    }

    /// Update queue size for a specific tier
    fn set_tier_queue_size(&self, tier_idx: usize, size: usize) {
        if let Some(counter) = self.tier_queue_sizes.get(&tier_idx) {
            counter.store(size, Ordering::Relaxed);
        }
    }

    /// Get all tier queue sizes for UI display
    /// Returns HashMap<tier_idx, queue_size>
    pub fn get_tier_queue_sizes(&self) -> HashMap<usize, usize> {
        self.tier_queue_sizes
            .iter()
            .map(|entry| (*entry.key(), entry.value().load(Ordering::Relaxed)))
            .collect()
    }

    /// Get last scheduling iteration duration in microseconds
    pub fn get_last_schedule_duration_us(&self) -> u64 {
        self.last_schedule_duration_us.load(Ordering::Relaxed)
    }

    /// Calculate TPOT value for a given tier index
    /// - For SLO tiers (0..tpot_buckets.len()): returns the upper bound of that tier
    /// - For idle tier (tpot_buckets.len()): returns idle_tpot_ms from config (default 1000.0)
    fn calculate_tpot_for_tier(&self, tier_idx: usize) -> f64 {
        let idle_tier_idx = self.tpot_buckets.len();

        if tier_idx < idle_tier_idx {
            // SLO tier - use the upper bound
            self.tpot_buckets[tier_idx] as f64
        } else {
            // Idle tier - use configured idle TPOT (default 1000.0)
            self.auto_scaling
                .as_ref()
                .map(|config| config.idle_tpot_ms)
                .unwrap_or(1000.0)
        }
    }

    /// Send TPOT update to a worker (fire-and-forget HTTP POST to /set_tpot)
    /// Only sends if send_tpot_updates is true
    fn send_tpot_update(&self, worker_url: &str, tpot_ms: f64) {
        if !self.send_tpot_updates {
            return;
        }

        // Fire-and-forget HTTP POST to /set_tpot endpoint
        let url = format!("{}/set_tpot", worker_url.trim_end_matches('/'));
        let tpot_value = tpot_ms;
        let client = self.tpot_client.clone(); // Cheap clone - just Arc increment

        tokio::spawn(async move {
            let payload = serde_json::json!({
                "tpot": tpot_value
            });

            match client
                .post(&url)
                .json(&payload)
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await
            {
                Ok(response) => {
                    if response.status().is_success() {
                        tracing::debug!("Successfully sent TPOT update to {}: {} ms", url, tpot_value);
                    } else {
                        tracing::warn!(
                            "Failed to send TPOT update to {}: HTTP {}",
                            url,
                            response.status()
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!("Error sending TPOT update to {}: {}", url, e);
                }
            }
        });
    }

    /// Dynamic worker reclassification based on load
    /// - Move workers with batch_size=0 to idle tier
    /// - Assign idle workers to tiers with pending queues
    fn schedule_worker(&self, worker_registry: &Arc<crate::core::WorkerRegistry>) {
        // Check if auto-scaling is enabled
        let auto_scaling_enabled = self.auto_scaling
            .as_ref()
            .map(|config| config.enabled)
            .unwrap_or(false);

        if !auto_scaling_enabled {
            // Auto-scaling disabled - skip dynamic tier reassignment
            return;
        }

        let idle_tier_idx = self.tpot_buckets.len();

        // Get current worker stats
        let stats = if let Ok(cached_stats) = self.worker_stats.read() {
            cached_stats.clone()
        } else {
            return;
        };

        // Step 1: Collect idle workers from SLO tiers and move them to idle tier
        // Use is_worker_truly_idle() which checks num_requests + queue + pending
        let mut workers_to_move_to_idle: Vec<(usize, WorkerId)> = Vec::new(); // (from_tier, worker_id)

        for tier_idx in 0..self.tpot_buckets.len() {
            if let Some(worker_ids) = self.tier_workers.get(&tier_idx) {
                for worker_id in worker_ids.iter() {
                    if let Some(worker) = worker_registry.get(worker_id) {
                        // Worker is idle only if num_requests=0, queue=0, pending=0
                        if self.is_worker_truly_idle(worker.as_ref()) {
                            workers_to_move_to_idle.push((tier_idx, worker_id.clone()));
                        }
                    }
                }
            }
        }

        // Move idle workers to idle tier
        for (from_tier, worker_id) in workers_to_move_to_idle {
            // Remove from source tier
            if let Some(mut worker_ids) = self.tier_workers.get_mut(&from_tier) {
                worker_ids.retain(|id| id != &worker_id);
            }
            // Add to idle tier
            self.tier_workers
                .entry(idle_tier_idx)
                .or_insert_with(Vec::new)
                .push(worker_id.clone());

            // Send TPOT update and log
            if let Some(worker) = worker_registry.get(&worker_id) {
                let letter = self.get_worker_letter(worker.url());
                let new_tpot = self.calculate_tpot_for_tier(idle_tier_idx);
                self.send_tpot_update(worker.url(), new_tpot);
                info!(
                    "[IDLE] {} moved from tier {} to idle (TPOT {}ms)",
                    letter,
                    self.tpot_buckets[from_tier] as u32,
                    new_tpot as u64
                );
            }
        }

        // Step 2: Find tiers with pending queues and assign idle workers
        let mut tiers_with_queue: Vec<(usize, usize)> = Vec::new(); // (tier_idx, queue_size)
        for tier_idx in 0..self.tpot_buckets.len() {
            if let Some(queue_size_atomic) = self.tier_queue_sizes.get(&tier_idx) {
                let queue_size = queue_size_atomic.load(Ordering::Relaxed);
                if queue_size > 0 {
                    tiers_with_queue.push((tier_idx, queue_size));
                }
            }
        }

        // Sort tiers by queue size (descending) to prioritize busiest tiers
        tiers_with_queue.sort_by(|a, b| b.1.cmp(&a.1));

        // Step 3: Assign idle workers to tiers with pending queues
        // Only assign workers that are truly idle (no pending work)
        if !tiers_with_queue.is_empty() {
            let idle_worker_ids: Vec<WorkerId> = self.tier_workers
                .get(&idle_tier_idx)
                .map(|workers| workers.clone())
                .unwrap_or_default();

            // Filter to only truly idle workers (num_requests=0, queue=0, pending=0)
            let truly_idle: Vec<&WorkerId> = idle_worker_ids.iter()
                .filter(|worker_id| {
                    if let Some(worker) = worker_registry.get(worker_id) {
                        self.is_worker_truly_idle(worker.as_ref())
                    } else {
                        false
                    }
                })
                .collect();

            // Assign one idle worker to each tier with pending queue (round-robin)
            for (idle_worker_id, (target_tier_idx, _queue_size)) in
                truly_idle.iter().zip(tiers_with_queue.iter().cycle()) {

                // Remove from idle tier
                if let Some(mut worker_ids) = self.tier_workers.get_mut(&idle_tier_idx) {
                    worker_ids.retain(|id| id != *idle_worker_id);
                }

                // Add to target tier
                self.tier_workers
                    .entry(*target_tier_idx)
                    .or_insert_with(Vec::new)
                    .push((*idle_worker_id).clone());

                // Send TPOT update if auto-scaling is enabled
                if let Some(worker) = worker_registry.get(*idle_worker_id) {
                    let letter = self.get_worker_letter(worker.url());
                    let new_tpot = self.calculate_tpot_for_tier(*target_tier_idx);
                    self.send_tpot_update(worker.url(), new_tpot);
                    info!(
                        "[ASSIGN] {} moved from idle to tier {} (TPOT {}ms)",
                        letter,
                        self.tpot_buckets[*target_tier_idx] as u32,
                        new_tpot as u64
                    );
                }
            }
        }

        // Step 4: Steal from lower tiers if enabled
        let steal_enabled = self.auto_scaling
            .as_ref()
            .map(|config| config.steal_from_lower_tier)
            .unwrap_or(false);

        let steal_idle_threshold_ms = self.auto_scaling
            .as_ref()
            .map(|config| config.steal_idle_threshold_ms)
            .unwrap_or(1000);

        if steal_enabled {

            // Re-check which tiers still have queues AND have violations
            let mut tiers_needing_workers: Vec<usize> = Vec::new();
            for tier_idx in 0..self.tpot_buckets.len() {
                let has_queue = self.tier_queue_sizes
                    .get(&tier_idx)
                    .map(|q| q.load(Ordering::Relaxed) > 0)
                    .unwrap_or(false);
                let has_violations = self.has_violations_and_reset(tier_idx);

                if has_queue && has_violations {
                    tiers_needing_workers.push(tier_idx);
                    info!(
                        "[STEAL] tier {} has pending queue and violations, considering steal",
                        self.tpot_buckets[tier_idx] as u32
                    );
                }
            }

            // For each tier needing workers (with violations), try to steal from lower tiers
            for target_tier_idx in tiers_needing_workers {
                let target_tpot = self.tpot_buckets[target_tier_idx] as f64;

                // Look at lower tiers (tier_idx > target_tier_idx means higher TPOT)
                for source_tier_idx in (target_tier_idx + 1)..self.tpot_buckets.len() {
                    // Check if source tier has no queue
                    let source_has_queue = self.tier_queue_sizes
                        .get(&source_tier_idx)
                        .map(|q| q.load(Ordering::Relaxed) > 0)
                        .unwrap_or(false);

                    if source_has_queue {
                        continue; // Don't steal from tiers that have queued work
                    }

                    // Get workers in source tier
                    let source_workers: Vec<WorkerId> = self.tier_workers
                        .get(&source_tier_idx)
                        .map(|w| w.clone())
                        .unwrap_or_default();

                    if source_workers.len() <= 1 {
                        continue; // Don't leave tier empty, keep at least 1 worker
                    }

                    // Check last worker's iteration time and idle status
                    if let Some(last_worker_id) = source_workers.last() {
                        if let Some(worker) = worker_registry.get(last_worker_id) {
                            // Check if worker has been idle long enough
                            if !self.is_worker_idle_for(worker.url(), steal_idle_threshold_ms) {
                                continue; // Worker recently had requests, don't steal
                            }

                            let iter_time = stats.get(worker.url())
                                .and_then(|ws| ws.last_iteration_time_ms)
                                .unwrap_or(f64::MAX);

                            if iter_time < target_tpot {
                                // Steal this worker!
                                let letter = self.get_worker_letter(worker.url());

                                // Remove from source tier
                                if let Some(mut worker_ids) = self.tier_workers.get_mut(&source_tier_idx) {
                                    worker_ids.retain(|id| id != last_worker_id);
                                }

                                // Add to target tier
                                self.tier_workers
                                    .entry(target_tier_idx)
                                    .or_insert_with(Vec::new)
                                    .push(last_worker_id.clone());

                                // Update TPOT
                                let new_tpot = self.calculate_tpot_for_tier(target_tier_idx);
                                self.send_tpot_update(worker.url(), new_tpot);

                                info!(
                                    "[STEAL] {} (iter={}ms, idle>{}ms) moved from tier {} to tier {} (TPOT {}ms)",
                                    letter, iter_time as u64, steal_idle_threshold_ms,
                                    self.tpot_buckets[source_tier_idx] as u32,
                                    self.tpot_buckets[target_tier_idx] as u32,
                                    new_tpot as u64
                                );

                                break; // Only steal one worker per tier per tick
                            }
                        }
                    }
                }
            }
        }
    }

    /// Monitor worker health state changes and react accordingly
    /// - When worker becomes healthy: set TPOT based on tier assignment
    /// - When worker becomes unhealthy: log and update UI
    fn monitor_worker_health(&self, worker_registry: &Arc<crate::core::WorkerRegistry>) {
        let all_workers = worker_registry.get_all_with_ids();

        for (worker_id, worker) in all_workers {
            let worker_url = worker.url();
            let current_health = worker.is_healthy();

            // Check if we have previous health state
            let previous_health = self.worker_health_state.get(worker_url).map(|entry| *entry);

            match (previous_health, current_health) {
                // Worker just became healthy (unhealthy -> healthy or first time seeing it healthy)
                (Some(false), true) | (None, true) => {
                    info!("Worker {} is now healthy", worker_url);

                    // Find which tier this worker belongs to
                    let mut worker_tier: Option<usize> = None;
                    for tier_idx in 0..=self.tpot_buckets.len() {
                        if let Some(tier_workers) = self.tier_workers.get(&tier_idx) {
                            if tier_workers.contains(&worker_id) {
                                worker_tier = Some(tier_idx);
                                break;
                            }
                        }
                    }

                    // If worker is assigned to a tier and TPOT updates are enabled, send TPOT
                    if let Some(tier_idx) = worker_tier {
                        if self.send_tpot_updates {
                            let tpot_ms = self.calculate_tpot_for_tier(tier_idx);
                            info!(
                                "Setting TPOT for newly healthy worker {} (tier {}): {} ms",
                                worker_url, tier_idx, tpot_ms
                            );
                            self.send_tpot_update(worker_url, tpot_ms);
                        }

                        // Log tier assignment for UI visibility
                        let boundary_desc = if tier_idx < self.tpot_buckets.len() {
                            let boundary = self.tpot_buckets[tier_idx];
                            if tier_idx == 0 {
                                format!("≤{} ms", boundary)
                            } else {
                                let prev_boundary = self.tpot_buckets[tier_idx - 1];
                                format!(">{} ms, ≤{} ms", prev_boundary, boundary)
                            }
                        } else {
                            "idle/autoscaling".to_string()
                        };
                        info!(
                            "Worker {} assigned to tier {} ({})",
                            worker_url, tier_idx, boundary_desc
                        );
                    }

                    // Update health state
                    self.worker_health_state.insert(worker_url.to_string(), true);
                }
                // Worker just became unhealthy (healthy -> unhealthy)
                (Some(true), false) => {
                    warn!("Worker {} is now UNHEALTHY - will be excluded from scheduling", worker_url);

                    // Find which tier this worker belongs to for UI/logging
                    for tier_idx in 0..=self.tpot_buckets.len() {
                        if let Some(tier_workers) = self.tier_workers.get(&tier_idx) {
                            if tier_workers.contains(&worker_id) {
                                warn!(
                                    "Unhealthy worker {} in tier {} will not receive new requests",
                                    worker_url, tier_idx
                                );
                                break;
                            }
                        }
                    }

                    // Update health state
                    self.worker_health_state.insert(worker_url.to_string(), false);
                }
                // Worker still healthy or still unhealthy - no change needed
                (Some(true), true) | (Some(false), false) => {
                    // No health state change, do nothing
                }
                // First time seeing worker and it's unhealthy
                (None, false) => {
                    // Initialize health state as unhealthy
                    self.worker_health_state.insert(worker_url.to_string(), false);
                }
            }
        }
    }

    /// Send periodic TPOT updates to all workers (every 500ms)
    /// This ensures workers always have the correct TPOT value even if they miss
    /// tier change notifications or restart
    fn send_periodic_tpot_updates(&self, worker_registry: &Arc<crate::core::WorkerRegistry>) {
        if !self.send_tpot_updates {
            return;
        }

        // Iterate through all tiers including idle (0..=tpot_buckets.len())
        for tier_idx in 0..=self.tpot_buckets.len() {
            if let Some(worker_ids) = self.tier_workers.get(&tier_idx) {
                let tpot_ms = self.calculate_tpot_for_tier(tier_idx);

                for worker_id in worker_ids.iter() {
                    if let Some(worker) = worker_registry.get(worker_id) {
                        if worker.is_healthy() {
                            self.send_tpot_update(worker.url(), tpot_ms);
                        }
                    }
                }
            }
        }
    }

    fn select_worker_first_available(&self, workers: &[Arc<dyn Worker>]) -> Option<Arc<dyn Worker>> {
        if workers.is_empty() {
            return None;
        }

        // Check servers from first to last, find the first one with no pending work
        for worker in workers {
            if !self.has_pending_work(worker.as_ref()) {
                return Some(Arc::clone(worker));
            }
        }

        // All workers have pending work - defer scheduling
        None
    }

    /// Log status output
    /// Format: [SLO] 10: Q=5 W=[A1, B6] | 20: Q=2 W=[C3] | idle: W=[D0] | 150us
    fn log_status(
        &self,
        worker_registry: &Arc<crate::core::WorkerRegistry>,
        queues: &[VecDeque<crate::routers::http::scheduler::PendingRequest>],
        tick_us: u64,
    ) {
        // Get worker stats for iteration times
        let stats = self.worker_stats.read().ok();

        let mut parts = Vec::new();

        // Log each SLO tier
        for (tier_idx, boundary) in self.tpot_buckets.iter().enumerate() {
            let queue_size = queues.get(tier_idx).map(|q| q.len()).unwrap_or(0);

            // Get workers in this tier with their iteration times
            let mut worker_strs = Vec::new();
            if let Some(worker_ids) = self.tier_workers.get(&tier_idx) {
                for worker_id in worker_ids.iter() {
                    if let Some(worker) = worker_registry.get(worker_id) {
                        let letter = self.get_worker_letter(worker.url());
                        let iter_time = stats.as_ref()
                            .and_then(|s| s.get(worker.url()))
                            .and_then(|ws| ws.last_iteration_time_ms)
                            .map(|t| t as u64)
                            .unwrap_or(0);
                        worker_strs.push(format!("{}{}", letter, iter_time));
                    }
                }
            }

            parts.push(format!(
                "{}: Q={} W=[{}]",
                *boundary as u32,
                queue_size,
                worker_strs.join(", ")
            ));
        }

        // Log idle tier
        let idle_tier_idx = self.tpot_buckets.len();
        let mut idle_worker_strs = Vec::new();
        if let Some(worker_ids) = self.tier_workers.get(&idle_tier_idx) {
            for worker_id in worker_ids.iter() {
                if let Some(worker) = worker_registry.get(worker_id) {
                    let letter = self.get_worker_letter(worker.url());
                    idle_worker_strs.push(format!("{}", letter));
                }
            }
        }
        parts.push(format!("idle: W=[{}]", idle_worker_strs.join(", ")));

        // Add tick duration
        parts.push(format!("{}us", tick_us));

        info!("[SLO] {}", parts.join(" | "));
    }

    /// Linear interpolation to estimate prefill time for a given token count
    /// Uses prefill simulation metrics which map token counts to execution times
    fn interpolate_prefill_time(
        &self,
        sim_metrics: &HashMap<i64, f64>,
        target_tokens: i64,
        pending_tokens: i64,
    ) -> Option<f64> {
        // Convert to sorted vector
        let mut points: Vec<(i64, f64)> = sim_metrics.iter().map(|(&k, &v)| (k, v)).collect();
        points.sort_by_key(|&(k, _)| k);

        if points.is_empty() {
            return None;
        }

        // Exact match
        if let Some(&(_, time)) = points.iter().find(|(k, _)| *k == target_tokens) {
            return Some(time);
        }

        // Interpolation between two points
        for window in points.windows(2) {
            let (k1, t1) = window[0];
            let (k2, t2) = window[1];

            if k1 < target_tokens && target_tokens < k2 {
                let ratio = (target_tokens - k1) as f64 / (k2 - k1) as f64;
                return Some(t1 + ratio * (t2 - t1));
            }
        }

        // Extrapolation (use closest point)
        if target_tokens < points[0].0 {
            return Some(points[0].1);
        }
        if target_tokens > points.last().unwrap().0 {
            // If there are pending tokens, don't extrapolate - block admission
            if pending_tokens > 0 {
                return None;
            }
            // No pending tokens - extrapolate using linear fit of last two points
            if points.len() >= 2 {
                let (k1, t1) = points[points.len() - 2];
                let (k2, t2) = points[points.len() - 1];
                let slope = (t2 - t1) / (k2 - k1) as f64;
                let extrapolated = t2 + slope * (target_tokens - k2) as f64;
                info!(
                    "[INTERPOLATE] extrapolating: target_tokens={} > max={}, slope={:.4}, result={:.1}ms",
                    target_tokens, k2, slope, extrapolated
                );
                return Some(extrapolated);
            }
            // Only one point - can't extrapolate
            return None;
        }

        None
    }

    /// Calculate total pending tokens for a worker (queue + pending messages)
    fn calculate_pending_tokens(&self, worker: &dyn Worker, worker_stats: &WorkerStats) -> i64 {
        // Sum tokens from waiting queue (accurate from worker)
        let queue_tokens = worker_stats
            .waiting_queue_info
            .as_ref()
            .map(|info| info.total_extend_len)
            .unwrap_or(0);

        // Sum actual tokens from pending messages tracked by the worker
        let pending_msg_tokens = worker.pending_message_tokens();

        queue_tokens + pending_msg_tokens
    }

    /// Estimate TTFT for a worker given a new request
    fn estimate_ttft(
        &self,
        worker: &dyn Worker,
        new_request_tokens: i64,
        margin_ms: f64,
    ) -> Option<f64> {
        // Get worker stats
        let stats = self.worker_stats.read().ok()?;
        let worker_stats = stats.get(worker.url());
        if worker_stats.is_none() {
            // info!("[TTFT_EST] worker={} no worker_stats", worker.url());
            return None;
        }
        let worker_stats = worker_stats.unwrap();

        // Get prefill sim metrics
        let sim_metrics = worker_stats.prefill_sim_metrics.as_ref();
        if sim_metrics.is_none() {
            // info!("[TTFT_EST] worker={} no prefill_sim_metrics", worker.url());
            return None;
        }
        let sim_metrics = sim_metrics.unwrap();

        // Calculate pending tokens
        let pending_tokens = self.calculate_pending_tokens(worker, worker_stats);

        // Total tokens = pending + new request
        let total_tokens = pending_tokens + new_request_tokens;

        // Interpolate to get estimated time
        let estimated_ms = self.interpolate_prefill_time(sim_metrics, total_tokens, pending_tokens);
        if estimated_ms.is_none() {
            // info!(
            //     "[TTFT_EST] worker={} interpolate failed total_tokens={} pending={}",
            //     worker.url(), total_tokens, pending_tokens
            // );
            return None;
        }
        let estimated_ms = estimated_ms.unwrap();

        // info!(
        //     "[TTFT_EST] worker={} pending={} new={} total={} est={:.1}ms margin={:.1}ms",
        //     worker.url(), pending_tokens, new_request_tokens, total_tokens, estimated_ms, margin_ms
        // );

        // Add safety margin
        Some(estimated_ms + margin_ms)
    }

    /// Check if request's TTFT deadline has been violated
    fn is_ttft_violated(&self, req: &PendingRequest) -> bool {
        let Some(target_ttft) = req.target_ttft_ms else {
            return false; // No TTFT target specified, consider not violated
        };

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as f64;
        let elapsed_ms = now_ms - req.arrival_time_ms;
        elapsed_ms > target_ttft as f64
    }

    /// TTFT-aware worker selection (without fallback)
    /// Returns worker that can meet remaining TTFT slack, or None to retry later
    fn select_worker_ttft_aware(
        &self,
        workers: &[Arc<dyn Worker>],
        request_tokens: i64,
        remaining_slack_ms: f64,
        margin_ms: f64,
    ) -> Option<Arc<dyn Worker>> {
        if workers.is_empty() {
            info!("[TTFT_SEL] no workers available");
            return None;
        }

        // Phase 1: Find first worker that can meet remaining TTFT slack
        // This maintains a load gradient similar to first-available policy
        for worker in workers {
            if let Some(estimated_ttft) = self.estimate_ttft(worker.as_ref(), request_tokens, margin_ms) {
                if estimated_ttft <= remaining_slack_ms {
                    // info!(
                    //     "[TTFT_SEL] worker={} ACCEPT est={:.1}ms <= slack={:.1}ms",
                    //     worker.url(), estimated_ttft, remaining_slack_ms
                    // );
                    return Some(Arc::clone(worker));
                } else {
                    // info!(
                    //     "[TTFT_SEL] worker={} REJECT est={:.1}ms > slack={:.1}ms",
                    //     worker.url(), estimated_ttft, remaining_slack_ms
                    // );
                }
            }
        }

        // Phase 2: No worker can meet remaining slack -> return None (retry next tick)
        None
    }

    /// TTFT-aware worker selection with fallback strategy
    #[allow(dead_code)]
    fn select_worker_ttft_aware_with_fallback(
        &self,
        workers: &[Arc<dyn Worker>],
        request_tokens: i64,
        remaining_slack_ms: f64,
        ttft_violated: bool,
        margin_ms: f64,
    ) -> Option<Arc<dyn Worker>> {
        if workers.is_empty() {
            return None;
        }

        // If TTFT already violated, use fallback strategy
        if ttft_violated {
            // Phase 3: Select worker with minimal estimated latency
            let mut all_with_estimates: Vec<(Arc<dyn Worker>, f64)> = workers
                .iter()
                .filter_map(|w| {
                    let estimated = self.estimate_ttft(w.as_ref(), request_tokens, margin_ms)?;
                    Some((Arc::clone(w), estimated))
                })
                .collect();

            if !all_with_estimates.is_empty() {
                all_with_estimates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                return Some(all_with_estimates[0].0.clone());
            }

            // Phase 4: No metrics available -> random selection
            if !workers.is_empty() {
                use rand::Rng;
                let mut rng = rand::rng();
                let idx = rng.random_range(0..workers.len());
                return Some(workers[idx].clone());
            }
            return None;
        }

        // Normal case: try to meet remaining slack
        if let Some(worker) = self.select_worker_ttft_aware(workers, request_tokens, remaining_slack_ms, margin_ms) {
            return Some(worker);
        }

        // If TTFT-aware selection failed (typically because prefill_sim_metrics unavailable),
        // fall back to FirstAvailable logic to preserve admission control
        self.select_worker_first_available(workers)
    }

    async fn drain_all_queues(
        self: &Arc<Self>,
        config: &Arc<SchedulerConfig>,
        queues: &mut [VecDeque<PendingRequest>],
    ) {
        for (queue_idx, queue) in queues.iter_mut().enumerate() {
            self.drain_single_queue(config, queue, queue_idx).await;
        }
    }

    async fn drain_single_queue(
        self: &Arc<Self>,
        config: &Arc<SchedulerConfig>,
        queue: &mut VecDeque<PendingRequest>,
        queue_idx: usize,
    ) {
        // First pass: remove timed out and TTFT-violated requests from anywhere in queue
        let mut i = 0;
        while i < queue.len() {
            let request = &queue[i];
            if has_request_timed_out(config, request) {
                let timed_out = queue.remove(i).unwrap();
                self.handle_timeout(timed_out);
                // Don't increment i, next element shifts into current position
            } else if self.is_ttft_violated(request) {
                let violated = queue.remove(i).unwrap();
                warn!(
                    "Rejecting request {}: TTFT target {:?} ms violated",
                    violated.request_id,
                    violated.target_ttft_ms
                );
                self.record_violation(queue_idx);
                self.handle_timeout(violated);
                // Don't increment i
            } else {
                i += 1;
            }
        }

        // Get tier workers for this queue
        let tier_worker_ids = match self.tier_workers.get(&queue_idx) {
            Some(ids) => ids.clone(),
            None => {
                warn!("No tier workers found for queue index {}", queue_idx);
                return;
            }
        };

        // Second pass: try to schedule requests, skipping those that don't fit
        // If queue is very long, only check a random sample to limit runtime
        const MAX_QUEUE_SCAN: usize = 100;
        let indices_to_check: Vec<usize> = if queue.len() > MAX_QUEUE_SCAN {
            let mut indices: Vec<usize> = (0..queue.len()).collect();
            indices.shuffle(&mut rand::rng());
            let mut sampled: Vec<usize> = indices.into_iter().take(MAX_QUEUE_SCAN).collect();
            sampled.sort(); // Preserve arrival order
            sampled
        } else {
            (0..queue.len()).collect()
        };

        // First pass: read-only, decide which indices to schedule and to which worker
        let mut schedule_decisions: Vec<(usize, Arc<dyn Worker>)> = Vec::new();

        for &i in &indices_to_check {
            let request = &queue[i];

            let available =
                available_workers_for_request(&config.worker_registry, request.model_id.as_deref());
            if available.is_empty() {
                continue;
            }

            let tier_filtered: Vec<Arc<dyn Worker>> = available
                .into_iter()
                .filter(|worker| {
                    if let Some(worker_id) = config.worker_registry.get_worker_id_by_url(worker.url()) {
                        tier_worker_ids.contains(&worker_id)
                    } else {
                        false
                    }
                })
                .collect();

            if tier_filtered.is_empty() {
                continue;
            }

            let worker = match &self.policy {
                WorkerSelectionPolicy::FirstAvailable => {
                    self.select_worker_first_available(&tier_filtered)
                }
                WorkerSelectionPolicy::TTFTAware { margin_ms } => {
                    // Use actual token count if available (input_ids), otherwise estimate from text
                    let request_tokens = request
                        .input_token_count
                        .unwrap_or_else(|| (request.text.len() / 4) as i64);
                    let target_ttft_ms = request.target_ttft_ms.unwrap_or(1000.0) as f64;
                    let now_ms = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as f64;
                    let elapsed_ms = now_ms - request.arrival_time_ms;
                    let remaining_slack_ms = target_ttft_ms - elapsed_ms;

                    self.select_worker_ttft_aware(
                        &tier_filtered,
                        request_tokens,
                        remaining_slack_ms,
                        *margin_ms,
                    )
                }
            };

            if let Some(w) = worker {
                schedule_decisions.push((i, w));
            }
        }

        // Build index -> worker map for O(1) lookup and count per-worker scheduling
        let mut per_worker_count: HashMap<String, usize> = HashMap::new();
        let schedule_map: HashMap<usize, Arc<dyn Worker>> = schedule_decisions
            .into_iter()
            .map(|(idx, worker)| {
                *per_worker_count.entry(worker.url().to_string()).or_insert(0) += 1;
                (idx, worker)
            })
            .collect();

        let scheduled_count = schedule_map.len();
        let remaining_count = queue.len() - scheduled_count;

        // Build new queue with remaining requests, dispatch scheduled ones
        let old_queue = std::mem::take(queue);
        for (i, request) in old_queue.into_iter().enumerate() {
            if let Some(worker) = schedule_map.get(&i) {
                RouterUi::dec_queue();
                // Record that this worker received a request
                self.record_scheduled(worker.url());
                let dispatcher = Arc::clone(self);
                let cfg = Arc::clone(config);
                dispatcher.dispatch_to_worker(cfg, request, Arc::clone(worker)).await;
            } else {
                queue.push_back(request);
            }
        }

        // Log scheduling stats if there was activity or pending requests
        if scheduled_count > 0 || remaining_count > 0 {
            let tier_tpot = self.tpot_buckets.get(queue_idx).map(|b| *b as u32).unwrap_or(0);

            // Build per-worker stats string
            let mut worker_stats_str = Vec::new();
            for worker_id in tier_worker_ids.iter() {
                if let Some(worker) = config.worker_registry.get(worker_id) {
                    let letter = self.get_worker_letter(worker.url());
                    let count = per_worker_count.get(worker.url()).copied().unwrap_or(0);
                    worker_stats_str.push(format!("{}={}", letter, count));

                    // If worker got no requests but tier has pending, print diagnostics
                    if count == 0 && remaining_count > 0 {
                        if let Ok(stats) = self.worker_stats.read() {
                            if let Some(ws) = stats.get(worker.url()) {
                                let has_pending = self.has_pending_work(worker.as_ref());
                                let prefill_map = ws.prefill_sim_metrics.as_ref()
                                    .map(|m| {
                                        let mut entries: Vec<_> = m.iter().collect();
                                        entries.sort_by_key(|(k, _)| *k);
                                        entries.iter()
                                            .map(|(k, v)| format!("{}:{:.0}", k, v))
                                            .collect::<Vec<_>>()
                                            .join(",")
                                    })
                                    .unwrap_or_else(|| "none".to_string());
                                info!(
                                    "[DIAG] tier={} worker={} pending_work={} prefill_map=[{}]",
                                    tier_tpot, letter, has_pending, prefill_map
                                );
                            }
                        }
                    }
                }
            }

            info!(
                "[SCHED] tier={}: scheduled={} remaining={} workers=[{}]",
                tier_tpot, scheduled_count, remaining_count, worker_stats_str.join(" ")
            );
        }
    }
}

impl SchedulerBase for SloAwareScheduler {}

impl Scheduler for SloAwareScheduler {
    fn spawn(
        self: Arc<Self>,
        config: SchedulerConfig,
        mut pending_rx: mpsc::Receiver<PendingRequest>,
    ) -> tokio::task::JoinHandle<()> {
        let config = Arc::new(config);
        let bucket_count = self.num_buckets();

        tokio::spawn(async move {
            if self.tpot_buckets.is_empty() {
                info!("SLO-aware scheduler started with no TPOT buckets (all requests rejected)");
            } else {
                info!(
                    "SLO-aware scheduler started with {} SLO tiers",
                    self.tpot_buckets.len()
                );
                for (i, boundary) in self.tpot_buckets.iter().enumerate() {
                    let range_desc = if i == 0 {
                        format!("≤{} ms", boundary)
                    } else {
                        let prev_boundary = self.tpot_buckets[i - 1];
                        format!(">{} ms, ≤{} ms", prev_boundary, boundary)
                    };
                    info!("  Queue {}: {}", i, range_desc);
                }
                if let Some(max_boundary) = self.tpot_buckets.last() {
                    info!("  Requests with TPOT > {} ms will be rejected", max_boundary);
                }
            }

            let mut queues: Vec<VecDeque<PendingRequest>> =
                (0..bucket_count).map(|_| VecDeque::new()).collect();

            // Initialize workers into tiers
            self.initialize_workers(&config.worker_registry);

            let mut ticker = interval(Duration::from_millis(SCHEDULER_TICK_INTERVAL_MS));
            ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

            // Separate ticker for periodic TPOT updates (500ms interval)
            let mut tpot_ticker = interval(Duration::from_millis(500));
            tpot_ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

            let mut receiver_closed = false;
            let mut tick_count: u64 = 0;

            loop {
                tokio::select! {
                    maybe_request = pending_rx.recv(), if !receiver_closed => {
                        match maybe_request {
                            Some(request) => {
                                match self.get_queue_index(request.target_tpot_ms) {
                                    Some(queue_idx) => {
                                        queues[queue_idx].push_back(request);
                                        self.set_tier_queue_size(queue_idx, queues[queue_idx].len());
                                    }
                                    None => {
                                        // Request rejected - no valid tier
                                        self.handle_timeout(request);
                                    }
                                }
                            }
                            None => receiver_closed = true,
                        }
                    }
                    _ = ticker.tick() => {
                        tick_count += 1;
                        let tick_start = std::time::Instant::now();

                        // Update worker stats for admission control
                        let stats = config.worker_registry.get_all_stats();
                        self.update_worker_stats(&stats);

                        // Drain queues and dispatch requests
                        self.drain_all_queues(&config, &mut queues).await;

                        // Update queue sizes for UI
                        for (queue_idx, queue) in queues.iter().enumerate() {
                            self.set_tier_queue_size(queue_idx, queue.len());
                        }

                        // Dynamic worker reclassification
                        self.schedule_worker(&config.worker_registry);

                        // Monitor worker health
                        self.monitor_worker_health(&config.worker_registry);

                        let tick_us = tick_start.elapsed().as_micros() as u64;
                        self.last_schedule_duration_us.store(tick_us, Ordering::Relaxed);

                        // Log status every 100 ticks (~500ms) with colored output
                        if tick_count % 100 == 0 {
                            self.log_status(&config.worker_registry, &queues, tick_us);
                        }

                        if receiver_closed && queues.iter().all(|queue| queue.is_empty()) {
                            break;
                        }
                    }
                    _ = tpot_ticker.tick() => {
                        self.send_periodic_tpot_updates(&config.worker_registry);
                    }
                }
            }

            warn!("SLO-aware scheduler loop exiting - pending channel closed and queues drained");
        })
    }

    fn name(&self) -> &'static str {
        "slo_aware"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
