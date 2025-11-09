use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use rand::Rng;
use tokio::sync::mpsc;
use tokio::time::{interval, MissedTickBehavior};
use tracing::{info, warn};

use crate::core::Worker;
use crate::routers::http::scheduler::{PendingRequest, SchedulerConfig};
use crate::ui::RouterUi;

use super::{
    available_workers_for_request, has_request_timed_out, Scheduler, SchedulerBase,
    SCHEDULER_TICK_INTERVAL_MS,
};

#[derive(Debug)]
pub struct SloAwareScheduler {
    pub tpot_buckets: Vec<f32>,
}

impl SloAwareScheduler {
    pub fn new(
        _policy_registry: Arc<crate::policies::PolicyRegistry>,
        tpot_buckets: Vec<f32>,
    ) -> Self {
        Self { tpot_buckets }
    }

    fn num_buckets(&self) -> usize {
        self.tpot_buckets.len() + 1
    }

    fn get_queue_index(&self, tpot: Option<f32>) -> usize {
        let Some(tpot) = tpot else {
            return self.tpot_buckets.len();
        };

        for (idx, boundary) in self.tpot_buckets.iter().enumerate() {
            if tpot <= *boundary {
                return idx;
            }
        }

        self.tpot_buckets.len()
    }

    fn select_worker_random(&self, workers: &[Arc<dyn Worker>]) -> Option<Arc<dyn Worker>> {
        if workers.is_empty() {
            return None;
        }

        let mut rng = rand::rng();
        let idx = rng.random_range(0..workers.len());
        workers.get(idx).cloned()
    }

    async fn drain_all_queues(
        self: &Arc<Self>,
        config: &Arc<SchedulerConfig>,
        queues: &mut [VecDeque<PendingRequest>],
    ) {
        for queue in queues.iter_mut() {
            self.drain_single_queue(config, queue).await;
        }
    }

    async fn drain_single_queue(
        self: &Arc<Self>,
        config: &Arc<SchedulerConfig>,
        queue: &mut VecDeque<PendingRequest>,
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

            let available =
                available_workers_for_request(&config.worker_registry, front.model_id.as_deref());

            if available.is_empty() {
                break;
            }

            let Some(worker) = self.select_worker_random(&available) else {
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
                info!("SLO-aware scheduler started with single queue (no TPOT buckets)");
            } else {
                info!(
                    "SLO-aware scheduler started with {} TPOT buckets",
                    self.tpot_buckets.len()
                );
                for (i, boundary) in self.tpot_buckets.iter().enumerate() {
                    info!("  Queue {}: TPOT <= {} ms", i, boundary);
                }
                if let Some(last) = self.tpot_buckets.last() {
                    info!("  Queue {}: TPOT > {} ms", self.tpot_buckets.len(), last);
                }
            }

            let mut queues: Vec<VecDeque<PendingRequest>> =
                (0..bucket_count).map(|_| VecDeque::new()).collect();

            let mut ticker = interval(Duration::from_millis(SCHEDULER_TICK_INTERVAL_MS));
            ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
            let mut receiver_closed = false;

            loop {
                tokio::select! {
                    maybe_request = pending_rx.recv(), if !receiver_closed => {
                        match maybe_request {
                            Some(request) => {
                                let idx = self.get_queue_index(request.target_tpot_ms);
                                let target_queue = idx.min(queues.len().saturating_sub(1));
                                let target_tpot_ms = request.target_tpot_ms;
                                queues[target_queue].push_back(request);
                                info!("Request added to queue {}: {:?}", target_queue, target_tpot_ms);
                            }
                            None => receiver_closed = true,
                        }
                    }
                    _ = ticker.tick() => {
                        self.drain_all_queues(&config, &mut queues).await;

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
}
