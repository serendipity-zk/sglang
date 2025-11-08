use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time::{interval, MissedTickBehavior};
use tracing::warn;

use crate::routers::http::scheduler::{PendingRequest, SchedulerConfig};
use crate::ui::RouterUi;

use super::{
    available_workers_for_request, has_request_timed_out, Scheduler, SchedulerBase,
    SCHEDULER_TICK_INTERVAL_MS,
};

#[derive(Debug, Default)]
pub struct EagerScheduler;

impl EagerScheduler {
    pub fn new() -> Self {
        Self
    }

    async fn drain_queue(
        self: &Arc<Self>,
        config: &Arc<SchedulerConfig>,
        pending_queue: &mut VecDeque<PendingRequest>,
    ) {
        loop {
            let front = match pending_queue.front() {
                Some(request) => request,
                None => break,
            };

            if has_request_timed_out(config, front) {
                let timed_out = pending_queue.pop_front().unwrap();
                self.handle_timeout(timed_out);
                continue;
            }

            let available = available_workers_for_request(
                &config.worker_registry,
                front.model_id.as_deref(),
            );

            if available.is_empty() {
                break;
            }

            let policy = match front.model_id.as_deref() {
                Some(model) => config.policy_registry.get_policy_or_default(model),
                None => config.policy_registry.get_default_policy(),
            };

            let worker = policy
                .select_worker(&available, Some(&front.text))
                .and_then(|idx| available.get(idx).cloned())
                .or_else(|| available.first().cloned());

            let Some(worker) = worker else {
                break;
            };

            let request = pending_queue.pop_front().unwrap();
            RouterUi::dec_queue();
            let dispatcher = Arc::clone(self);
            let cfg = Arc::clone(config);
            dispatcher.dispatch_to_worker(cfg, request, worker).await;
        }
    }
}

impl SchedulerBase for EagerScheduler {}

impl Scheduler for EagerScheduler {
    fn spawn(
        self: Arc<Self>,
        config: SchedulerConfig,
        mut pending_rx: mpsc::Receiver<PendingRequest>,
    ) -> tokio::task::JoinHandle<()> {
        let config = Arc::new(config);
        tokio::spawn(async move {
            let mut pending_queue = VecDeque::new();
            let mut ticker = interval(Duration::from_millis(SCHEDULER_TICK_INTERVAL_MS));
            ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
            let mut receiver_closed = false;

            loop {
                tokio::select! {
                    maybe_request = pending_rx.recv(), if !receiver_closed => {
                        match maybe_request {
                            Some(request) => pending_queue.push_back(request),
                            None => receiver_closed = true,
                        }
                    }
                    _ = ticker.tick() => {
                        self.drain_queue(&config, &mut pending_queue).await;

                        if receiver_closed && pending_queue.is_empty() {
                            break;
                        }
                    }
                }
            }

            warn!("Eager scheduler loop exiting - pending channel closed and queue drained");
        })
    }

    fn name(&self) -> &'static str {
        "eager"
    }
}
