use crate::core::WorkerRegistry;
use crate::policies::PolicyRegistry;
use crate::schedulers::{Scheduler, SchedulerRegistry};
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tracing::info;

/// Pending request metadata shared between the router entrypoint and scheduler loop.
pub struct PendingRequest {
    pub headers: Option<axum::http::HeaderMap>,
    pub body_json: serde_json::Value,
    pub route: String,
    pub model_id: Option<String>,
    pub is_stream: bool,
    pub text: String,
    pub enqueue_started: std::time::Instant,
    /// Arrival time recorded at router ingress (ms since Unix epoch)
    pub arrival_time_ms: f64,
    pub response_tx: oneshot::Sender<axum::response::Response>,
    /// Target time to first token (SLO) in milliseconds
    #[allow(dead_code)]
    pub target_ttft_ms: Option<f32>,
    /// Target time per output token (SLO) in milliseconds
    pub target_tpot_ms: Option<f32>,
}

/// Configuration and shared state required by scheduler implementations.
pub struct SchedulerConfig {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub dp_aware: bool,
    pub queue_timeout: Duration,
}

/// Launch the configured scheduler to drain pending requests.
pub(crate) fn spawn_scheduler(
    scheduler_registry: Arc<SchedulerRegistry>,
    config: SchedulerConfig,
    pending_rx: mpsc::Receiver<PendingRequest>,
) -> tokio::task::JoinHandle<()> {
    let scheduler = scheduler_registry.get_scheduler_blocking();
    info!("Starting '{}' scheduler", scheduler.name());
    scheduler.clone().spawn(config, pending_rx)
}

// Helper extension trait to allow blocking read from scheduler registry in sync context
trait SchedulerRegistryExt {
    fn get_scheduler_blocking(&self) -> Arc<dyn Scheduler>;
}

impl SchedulerRegistryExt for Arc<SchedulerRegistry> {
    fn get_scheduler_blocking(&self) -> Arc<dyn Scheduler> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.get_scheduler())
        })
    }
}
