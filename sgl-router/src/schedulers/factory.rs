use std::sync::Arc;

use crate::config::types::SchedulerConfig;
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
            SchedulerConfig::SloAware { tpot_buckets } => Arc::new(SloAwareScheduler::new(
                policy_registry,
                tpot_buckets.clone(),
            )),
        }
    }

    /// Creates a scheduler by name (useful for defaults or testing)
    pub fn create_by_name(name: &str, policy_registry: Arc<PolicyRegistry>) -> Arc<dyn Scheduler> {
        match name {
            "eager" => Arc::new(EagerScheduler::new()),
            "gated" => Arc::new(GatedScheduler::new()),
            "slo_aware" => {
                // Default buckets: <10ms, 10-50ms, >50ms
                Arc::new(SloAwareScheduler::new(policy_registry, vec![10.0, 50.0]))
            }
            _ => {
                tracing::warn!("Unknown scheduler name '{}', defaulting to eager", name);
                Arc::new(EagerScheduler::new())
            }
        }
    }
}
