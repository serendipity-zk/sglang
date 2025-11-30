use std::sync::Arc;

use crate::config::types::{SchedulerConfig, WorkerSelectionPolicy};
use crate::policies::PolicyRegistry;

use super::{EagerScheduler, GatedScheduler, Scheduler, SloAwareScheduler};

/// Factory for creating scheduler instances from configuration
pub struct SchedulerFactory;

impl SchedulerFactory {
    /// Creates a scheduler from the given configuration
    ///
    /// # Arguments
    /// * `config` - The scheduler configuration
    /// * `policy_registry` - The policy registry for worker selection
    ///
    /// # Returns
    /// Arc-wrapped scheduler instance implementing Scheduler
    pub fn create_from_config(
        config: &SchedulerConfig,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Arc<dyn Scheduler> {
        match config {
            SchedulerConfig::Eager => Arc::new(EagerScheduler::new()),
            SchedulerConfig::Gated => Arc::new(GatedScheduler::new()),
            SchedulerConfig::SloAware { tpot_buckets, worker_selection_policy } => {
                let policy = worker_selection_policy.clone()
                    .unwrap_or(WorkerSelectionPolicy::FirstAvailable);
                Arc::new(SloAwareScheduler::new(
                    policy_registry,
                    tpot_buckets.clone(),
                    policy,
                ))
            },
        }
    }

    /// Creates a scheduler by name (useful for defaults or testing)
    pub fn create_by_name(name: &str, policy_registry: Arc<PolicyRegistry>) -> Arc<dyn Scheduler> {
        match name {
            "eager" => Arc::new(EagerScheduler::new()),
            "gated" => Arc::new(GatedScheduler::new()),
            "slo_aware" => {
                // Default buckets: <10ms, 10-50ms, >50ms
                // Default policy: FirstAvailable
                Arc::new(SloAwareScheduler::new(
                    policy_registry,
                    vec![10.0, 50.0],
                    WorkerSelectionPolicy::FirstAvailable,
                ))
            }
            _ => {
                tracing::warn!("Unknown scheduler name '{}', defaulting to eager", name);
                Arc::new(EagerScheduler::new())
            }
        }
    }
}
