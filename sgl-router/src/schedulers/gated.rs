use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time::{interval, MissedTickBehavior};
use tracing::warn;

use crate::routers::http::scheduler::{PendingRequest, SchedulerConfig};
use crate::ui::RouterUi;

use super::{
    has_request_timed_out, select_worker_for_request, Scheduler, SchedulerBase, WorkerSelection,
    SCHEDULER_TICK_INTERVAL_MS,
};

#[derive(Debug, Default)]
pub struct GatedScheduler;

impl GatedScheduler {
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

            match select_worker_for_request(config, front) {
                WorkerSelection::Selected(worker) => {
                    let request = pending_queue.pop_front().unwrap();
                    RouterUi::dec_queue();
                    let dispatcher = Arc::clone(self);
                    let cfg = Arc::clone(config);
                    dispatcher.dispatch_to_worker(cfg, request, worker).await;
                }
                WorkerSelection::NoWorkers | WorkerSelection::PolicyDeferred => break,
            }
        }
    }
}

impl SchedulerBase for GatedScheduler {}

impl Scheduler for GatedScheduler {
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

            warn!("Gated scheduler loop exiting - pending channel closed and queue drained");
        })
    }

    fn name(&self) -> &'static str {
        "gated"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
