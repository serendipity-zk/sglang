use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use dashmap::DashMap;
use tokio::sync::mpsc;
use tokio::time::{interval, MissedTickBehavior};
use tracing::{info, warn};

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
}

impl SloAwareScheduler {
    pub fn new(
        _policy_registry: Arc<crate::policies::PolicyRegistry>,
        tpot_buckets: Vec<f32>,
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

    /// Initialize workers into SLO tiers using round-robin assignment
    /// Idle tier (tpot_buckets.len()) is left empty for future autoscaling
    fn initialize_workers(&self, worker_registry: &Arc<crate::core::WorkerRegistry>) {
        let all_workers = worker_registry.get_all_with_ids();
        let num_slo_tiers = self.tpot_buckets.len();

        if all_workers.is_empty() {
            warn!("No workers available for tier initialization");
            return;
        }

        info!(
            "Initializing {} workers into {} SLO tiers using round-robin assignment",
            all_workers.len(),
            num_slo_tiers
        );

        // Round-robin assign workers to SLO tiers (not idle tier)
        for (worker_idx, (worker_id, _worker)) in all_workers.iter().enumerate() {
            let tier_idx = worker_idx % num_slo_tiers;

            self.tier_workers
                .entry(tier_idx)
                .or_insert_with(Vec::new)
                .push(worker_id.clone());
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

    /// Dynamic worker reclassification based on load
    /// - Move workers with batch_size=0 to idle tier
    /// - Assign idle workers to tiers with pending queues
    fn schedule_worker(&self, worker_registry: &Arc<crate::core::WorkerRegistry>) {
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
                .push(worker_id);
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
        loop {
            let front = match queue.front() {
                Some(request) => request,
                None => break,
            };

            if has_request_timed_out(config, front) {
                let timed_out = queue.pop_front().unwrap();
                self.handle_timeout(timed_out);
                continue;
            }

            // Get available workers for the request (filtered by model)
            let available =
                available_workers_for_request(&config.worker_registry, front.model_id.as_deref());

            if available.is_empty() {
                break;
            }

            // Strict tier-based filtering: only use workers from the matching tier
            let tier_worker_ids = match self.tier_workers.get(&queue_idx) {
                Some(ids) => ids.clone(),
                None => {
                    warn!("No tier workers found for queue index {}", queue_idx);
                    break;
                }
            };

            // Filter available workers to only include those in the current tier
            let tier_filtered: Vec<Arc<dyn Worker>> = available
                .into_iter()
                .filter(|worker| {
                    // Get worker ID from registry by URL
                    if let Some(worker_id) = config.worker_registry.get_worker_id_by_url(worker.url()) {
                        tier_worker_ids.contains(&worker_id)
                    } else {
                        false
                    }
                })
                .collect();

            if tier_filtered.is_empty() {
                // No workers available in this tier - leave request in queue (strict matching)
                info!(
                    "Queue {}: No tier-assigned workers available, keeping request in queue",
                    queue_idx
                );
                break;
            }

            let Some(worker) = self.select_worker_first_available(&tier_filtered) else {
                // All workers in this tier have pending work - defer scheduling (admission control)
                info!(
                    "Queue {}: All {} tier workers have pending work, keeping request in queue (admission control)",
                    queue_idx,
                    tier_filtered.len()
                );
                break;
            };

            let request = queue.pop_front().unwrap();
            RouterUi::dec_queue();
            let dispatcher = Arc::clone(self);
            let cfg = Arc::clone(config);
            dispatcher.dispatch_to_worker(cfg, request, worker).await;
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
                                        queues[queue_idx].push_back(request);
                                        self.set_tier_queue_size(queue_idx, queues[queue_idx].len());
                                        info!("Request added to queue {}: target_tpot={:?}", queue_idx, target_tpot_ms);
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

                        // Placeholder for dynamic worker reclassification (currently empty)
                        self.schedule_worker(&config.worker_registry);

                        if receiver_closed && queues.iter().all(|queue| queue.is_empty()) {
                            break;
                        }
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
