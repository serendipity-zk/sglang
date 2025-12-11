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

        // Initialize empty tier assignments and queue size counters
        // SLO tiers: 0..tpot_buckets.len()-1
        // Idle tier: tpot_buckets.len()
        // Example: [10, 50] creates tiers 0, 1 (SLO) and 2 (idle)
        for tier_idx in 0..=tpot_buckets.len() {
            tier_workers.insert(tier_idx, Vec::new());
            tier_queue_sizes.insert(tier_idx, AtomicUsize::new(0));
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
        }
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

        tokio::spawn(async move {
            let client = reqwest::Client::new();
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
        let mut workers_to_move_to_idle: Vec<(usize, WorkerId)> = Vec::new(); // (from_tier, worker_id)

        for tier_idx in 0..self.tpot_buckets.len() {
            if let Some(worker_ids) = self.tier_workers.get(&tier_idx) {
                for worker_id in worker_ids.iter() {
                    if let Some(worker) = worker_registry.get(worker_id) {
                        if let Some(worker_stats) = stats.get(worker.url()) {
                            // Worker is idle if it has no requests
                            if worker_stats.num_requests == 0 {
                                workers_to_move_to_idle.push((tier_idx, worker_id.clone()));
                            }
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

            // Send TPOT update if auto-scaling is enabled
            if let Some(worker) = worker_registry.get(&worker_id) {
                let new_tpot = self.calculate_tpot_for_tier(idle_tier_idx);
                self.send_tpot_update(worker.url(), new_tpot);
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
        if !tiers_with_queue.is_empty() {
            let idle_worker_ids: Vec<WorkerId> = self.tier_workers
                .get(&idle_tier_idx)
                .map(|workers| workers.clone())
                .unwrap_or_default();

            // Assign one idle worker to each tier with pending queue (round-robin)
            for (idle_worker_id, (target_tier_idx, _queue_size)) in
                idle_worker_ids.iter().zip(tiers_with_queue.iter().cycle()) {

                // Remove from idle tier
                if let Some(mut worker_ids) = self.tier_workers.get_mut(&idle_tier_idx) {
                    worker_ids.retain(|id| id != idle_worker_id);
                }

                // Add to target tier
                self.tier_workers
                    .entry(*target_tier_idx)
                    .or_insert_with(Vec::new)
                    .push(idle_worker_id.clone());

                // Send TPOT update if auto-scaling is enabled
                if let Some(worker) = worker_registry.get(idle_worker_id) {
                    let new_tpot = self.calculate_tpot_for_tier(*target_tier_idx);
                    self.send_tpot_update(worker.url(), new_tpot);
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
        // Sum tokens from waiting queue
        let queue_tokens = worker_stats
            .waiting_queue_info
            .as_ref()
            .map(|info| info.total_extend_len)
            .unwrap_or(0);

        // Add tokens from pending router messages (estimate)
        let pending_msg_count = worker.pending_message_count() as i64;
        // Estimate: assume average request has ~512 tokens
        let pending_msg_tokens = pending_msg_count * 512;

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
                    let request_tokens = (request.text.len() / 4) as i64;
                    let target_ttft_ms = request.target_ttft_ms.unwrap_or(1000.0) as f64;
                    let now_ms = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as f64;
                    let elapsed_ms = now_ms - request.arrival_time_ms;
                    let remaining_slack_ms = target_ttft_ms - elapsed_ms;

                    info!(
                        "[WORKER_SEL] req={} target_ttft={:.1}ms elapsed={:.1}ms remaining_slack={:.1}ms tokens={} tier_workers={}",
                        request.request_id, target_ttft_ms, elapsed_ms, remaining_slack_ms, request_tokens, tier_filtered.len()
                    );

                    let result = self.select_worker_ttft_aware(
                        &tier_filtered,
                        request_tokens,
                        remaining_slack_ms,
                        *margin_ms,
                    );

                    if result.is_none() {
                        info!(
                            "[WORKER_SEL] req={} NO worker selected (remaining_slack={:.1}ms)",
                            request.request_id, remaining_slack_ms
                        );
                    } else {
                        info!(
                            "[WORKER_SEL] req={} selected worker={}",
                            request.request_id, result.as_ref().unwrap().url()
                        );
                    }

                    result
                }
            };

            if let Some(w) = worker {
                schedule_decisions.push((i, w));
            }
        }

        // Build index -> worker map for O(1) lookup
        let schedule_map: HashMap<usize, Arc<dyn Worker>> = schedule_decisions.into_iter().collect();

        // Build new queue with remaining requests, dispatch scheduled ones
        let old_queue = std::mem::take(queue);
        for (i, request) in old_queue.into_iter().enumerate() {
            if let Some(worker) = schedule_map.get(&i) {
                RouterUi::dec_queue();
                let dispatcher = Arc::clone(self);
                let cfg = Arc::clone(config);
                dispatcher.dispatch_to_worker(cfg, request, Arc::clone(worker)).await;
            } else {
                queue.push_back(request);
            }
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

            loop {
                tokio::select! {
                    maybe_request = pending_rx.recv(), if !receiver_closed => {
                        match maybe_request {
                            Some(request) => {
                                match self.get_queue_index(request.target_tpot_ms) {
                                    Some(queue_idx) => {
                                        // Valid request - add to appropriate queue
                                        let target_tpot_ms = request.target_tpot_ms;
                                        let target_ttft_ms = request.target_ttft_ms.unwrap_or(1000.0) as f64;
                                        let request_id = request.request_id.clone();
                                        let now_ms = SystemTime::now()
                                            .duration_since(UNIX_EPOCH)
                                            .unwrap()
                                            .as_millis() as f64;
                                        let elapsed_ms = now_ms - request.arrival_time_ms;
                                        let remaining_slack_ms = target_ttft_ms - elapsed_ms;

                                        queues[queue_idx].push_back(request);
                                        self.set_tier_queue_size(queue_idx, queues[queue_idx].len());
                                        info!("Timestamp {}, request_id={}, Request added to queue {}: target_tpot={:?}, elapsed={:.1}ms, remaining_slack={:.1}ms",
                                              chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
                                              request_id,
                                              queue_idx,
                                              target_tpot_ms,
                                              elapsed_ms,
                                              remaining_slack_ms);
                                    }
                                    None => {
                                        // Request should be rejected
                                        let reason = if request.target_tpot_ms.is_none() {
                                            "no SLO target specified"
                                        } else if let Some(max_boundary) = self.tpot_buckets.last() {
                                            if request.target_tpot_ms.unwrap() > *max_boundary {
                                                "exceeds maximum SLO boundary"
                                            } else {
                                                "invalid SLO target"
                                            }
                                        } else {
                                            "no SLO tiers configured"
                                        };
                                        warn!(
                                            "Rejecting request: {} (target_tpot={:?}, max_boundary={:?})",
                                            reason,
                                            request.target_tpot_ms,
                                            self.tpot_buckets.last()
                                        );
                                        // Handle timeout for rejected request
                                        self.handle_timeout(request);
                                    }
                                }
                            }
                            None => receiver_closed = true,
                        }
                    }
                    _ = ticker.tick() => {
                        info!("Timestamp {}, Scheduler tick started", chrono::Utc::now().timestamp_millis() as f64 / 1000.0);
                        // Update worker stats for admission control
                        let stats = config.worker_registry.get_all_stats();
                        self.update_worker_stats(&stats);

                        // Time the scheduling decision
                        let start = std::time::Instant::now();
                        self.drain_all_queues(&config, &mut queues).await;
                        let duration_us = start.elapsed().as_micros() as u64;
                        self.last_schedule_duration_us.store(duration_us, Ordering::Relaxed);

                        // Update queue sizes for UI after draining
                        for (queue_idx, queue) in queues.iter().enumerate() {
                            self.set_tier_queue_size(queue_idx, queue.len());
                        }

                        // Dynamic worker reclassification based on load
                        self.schedule_worker(&config.worker_registry);

                        // Monitor worker health state changes and update TPOT/UI accordingly
                        self.monitor_worker_health(&config.worker_registry);

                        if receiver_closed && queues.iter().all(|queue| queue.is_empty()) {
                            break;
                        }

                        // Only log if tick took meaningful time (≥10μs)
                        if duration_us >= 10 {
                            info!("Timestamp {}, Scheduler tick completed after {} us", chrono::Utc::now().timestamp_millis() as f64 / 1000.0, duration_us);
                        }
                    }
                    _ = tpot_ticker.tick() => {
                        // Periodic TPOT updates every 500ms
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
