use std::sync::Arc;
use tokio::sync::RwLock;

use super::Scheduler;

/// Registry for managing the active scheduling strategy
#[derive(Debug)]
pub struct SchedulerRegistry {
    scheduler: RwLock<Arc<dyn Scheduler>>,
}

impl SchedulerRegistry {
    /// Creates a new scheduler registry with the given initial scheduler
    pub fn new(scheduler: Arc<dyn Scheduler>) -> Self {
        Self {
            scheduler: RwLock::new(scheduler),
        }
    }

    /// Gets the current active scheduler
    pub async fn get_scheduler(&self) -> Arc<dyn Scheduler> {
        self.scheduler.read().await.clone()
    }

    /// Sets a new active scheduler
    pub async fn set_scheduler(&self, scheduler: Arc<dyn Scheduler>) {
        let mut guard = self.scheduler.write().await;
        *guard = scheduler;
    }
}
