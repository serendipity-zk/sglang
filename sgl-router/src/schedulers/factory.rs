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
            SchedulerConfig::SloAware { tpot_buckets, worker_selection_policy, auto_scaling, initial_tier_allocation, send_tpot_updates } => {
                let policy = worker_selection_policy.clone()
                    .unwrap_or(WorkerSelectionPolicy::FirstAvailable);
                Arc::new(SloAwareScheduler::new(
                    policy_registry,
                    tpot_buckets.clone(),
                    policy,
                    auto_scaling.clone(),
                    initial_tier_allocation.clone(),
                    *send_tpot_updates,
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
                // Default auto-scaling: disabled
                // Default tier allocation: None (round-robin)
                // Default TPOT updates: disabled
                Arc::new(SloAwareScheduler::new(
                    policy_registry,
                    vec![10.0, 50.0],
                    WorkerSelectionPolicy::FirstAvailable,
                    None,
                    None,
                    false,
                ))
            }
            _ => {
                tracing::warn!("Unknown scheduler name '{}', defaulting to eager", name);
                Arc::new(EagerScheduler::new())
            }
        }
    }
}
