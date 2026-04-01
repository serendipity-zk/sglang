//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing

use crate::core::{resend_message, ConnectionMode, Worker, WorkerStats, WorkerType};
use crate::metrics::RouterMetrics;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use url::Url;
use uuid::Uuid;

const PENDING_MESSAGE_TTL: Duration = Duration::from_secs(1);
const PENDING_DIAG_LOG_INTERVAL: Duration = Duration::from_secs(2);
/// Interval for checking and resending unacknowledged messages (50ms)
const RESEND_CHECK_INTERVAL_MS: u64 = 50;

/// Unique identifier for a worker
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct WorkerId(String);

impl WorkerId {
    /// Create a new worker ID
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a worker ID from a string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get the ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for the model index to reduce complexity
type ModelIndex = Arc<DashMap<String, Arc<RwLock<Vec<Arc<dyn Worker>>>>>>;

/// Worker registry with model-based indexing
#[derive(Debug)]
pub struct WorkerRegistry {
    /// All workers indexed by ID
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// Workers indexed by model ID (stores WorkerId for reference)
    model_workers: Arc<DashMap<String, Vec<WorkerId>>>,

    /// Optimized model index for O(1) lookups (stores Arc<dyn Worker> directly)
    model_index: ModelIndex,

    /// Workers indexed by worker type
    type_workers: Arc<DashMap<WorkerType, Vec<WorkerId>>>,

    /// Workers indexed by connection mode
    connection_workers: Arc<DashMap<ConnectionMode, Vec<WorkerId>>>,

    /// URL to worker ID mapping (for backward compatibility)
    url_to_id: Arc<DashMap<String, WorkerId>>,

    /// Real-time worker stats (worker_url -> stats)
    worker_stats: Arc<DashMap<String, WorkerStats>>,

    /// Per-worker/category rate limiting for pending-message diagnostics
    pending_diag_last_log: Arc<DashMap<String, Instant>>,
}

impl WorkerRegistry {
    /// Create a new worker registry
    pub fn new() -> Self {
        Self {
            workers: Arc::new(DashMap::new()),
            model_workers: Arc::new(DashMap::new()),
            model_index: Arc::new(DashMap::new()),
            type_workers: Arc::new(DashMap::new()),
            connection_workers: Arc::new(DashMap::new()),
            url_to_id: Arc::new(DashMap::new()),
            worker_stats: Arc::new(DashMap::new()),
            pending_diag_last_log: Arc::new(DashMap::new()),
        }
    }

    fn should_log_pending_diag(&self, worker_url: &str, category: &str) -> bool {
        let key = format!("{worker_url}::{category}");
        let now = Instant::now();

        if let Some(mut last) = self.pending_diag_last_log.get_mut(&key) {
            if now.saturating_duration_since(*last) < PENDING_DIAG_LOG_INTERVAL {
                return false;
            }
            *last = now;
            return true;
        }

        self.pending_diag_last_log.insert(key, now);
        true
    }

    /// Register a new worker
    pub fn register(&self, worker: Arc<dyn Worker>) -> WorkerId {
        let worker_id = if let Some(existing_id) = self.url_to_id.get(worker.url()) {
            // Worker with this URL already exists, update it
            existing_id.clone()
        } else {
            WorkerId::new()
        };

        // Store worker
        self.workers.insert(worker_id.clone(), worker.clone());

        // Update URL mapping
        self.url_to_id
            .insert(worker.url().to_string(), worker_id.clone());

        // Update model index (both ID-based and optimized)
        let model_id = worker.model_id().to_string();
        self.model_workers
            .entry(model_id.clone())
            .or_default()
            .push(worker_id.clone());

        // Update optimized model index for O(1) lookups
        self.model_index
            .entry(model_id)
            .or_insert_with(|| Arc::new(RwLock::new(Vec::new())))
            .write()
            .expect("RwLock for model_index is poisoned")
            .push(worker.clone());

        // Update type index
        self.type_workers
            .entry(worker.worker_type())
            .or_default()
            .push(worker_id.clone());

        // Update connection mode index
        self.connection_workers
            .entry(worker.connection_mode())
            .or_default()
            .push(worker_id.clone());

        worker_id
    }

    /// Remove a worker by ID
    pub fn remove(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker)) = self.workers.remove(worker_id) {
            // Remove from URL mapping
            self.url_to_id.remove(worker.url());

            // Remove from model index (both ID-based and optimized)
            if let Some(mut model_workers) = self.model_workers.get_mut(worker.model_id()) {
                model_workers.retain(|id| id != worker_id);
            }

            // Remove from optimized model index
            if let Some(model_index_entry) = self.model_index.get(worker.model_id()) {
                let worker_url = worker.url();
                model_index_entry
                    .write()
                    .expect("RwLock for model_index is poisoned")
                    .retain(|w| w.url() != worker_url);
            }

            // Remove from type index
            if let Some(mut type_workers) = self.type_workers.get_mut(&worker.worker_type()) {
                type_workers.retain(|id| id != worker_id);
            }

            // Remove from connection mode index
            if let Some(mut conn_workers) =
                self.connection_workers.get_mut(&worker.connection_mode())
            {
                conn_workers.retain(|id| id != worker_id);
            }

            // Remove worker stats
            self.worker_stats.remove(worker.url());

            Some(worker)
        } else {
            None
        }
    }

    /// Remove a worker by URL
    pub fn remove_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker_id)) = self.url_to_id.remove(url) {
            self.remove(&worker_id)
        } else {
            None
        }
    }

    /// Get a worker by ID
    pub fn get(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        self.workers.get(worker_id).map(|entry| entry.clone())
    }

    /// Get a worker by URL
    pub fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        self.url_to_id.get(url).and_then(|id| self.get(&id))
    }

    /// Get worker ID by URL
    pub fn get_worker_id_by_url(&self, url: &str) -> Option<WorkerId> {
        self.url_to_id.get(url).map(|id| id.clone())
    }

    /// Get all workers for a model
    pub fn get_by_model(&self, model_id: &str) -> Vec<Arc<dyn Worker>> {
        self.model_workers
            .get(model_id)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all workers for a model (O(1) optimized version)
    /// This method uses the pre-indexed model_index for fast lookups
    pub fn get_by_model_fast(&self, model_id: &str) -> Vec<Arc<dyn Worker>> {
        self.model_index
            .get(model_id)
            .map(|workers| {
                workers
                    .read()
                    .expect("RwLock for model_index is poisoned")
                    .clone()
            })
            .unwrap_or_default()
    }

    /// Get all workers by worker type
    pub fn get_by_type(&self, worker_type: &WorkerType) -> Vec<Arc<dyn Worker>> {
        self.type_workers
            .get(worker_type)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all prefill workers (regardless of bootstrap_port)
    pub fn get_prefill_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .filter_map(|entry| {
                let worker = entry.value();
                match worker.worker_type() {
                    WorkerType::Prefill { .. } => Some(worker.clone()),
                    _ => None,
                }
            })
            .collect()
    }

    /// Get all decode workers
    pub fn get_decode_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.get_by_type(&WorkerType::Decode)
    }

    /// Get all workers by connection mode
    pub fn get_by_connection(&self, connection_mode: &ConnectionMode) -> Vec<Arc<dyn Worker>> {
        self.connection_workers
            .get(connection_mode)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all workers
    pub fn get_all(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get all workers with their IDs
    pub fn get_all_with_ids(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.workers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Get all worker URLs
    pub fn get_all_urls(&self) -> Vec<String> {
        self.workers
            .iter()
            .map(|entry| entry.value().url().to_string())
            .collect()
    }

    pub fn get_all_urls_with_api_key(&self) -> Vec<(String, Option<String>)> {
        self.workers
            .iter()
            .map(|entry| {
                (
                    entry.value().url().to_string(),
                    entry.value().api_key().clone(),
                )
            })
            .collect()
    }

    /// Get all model IDs with workers
    pub fn get_models(&self) -> Vec<String> {
        self.model_workers
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get workers filtered by multiple criteria
    ///
    /// This method allows flexible filtering of workers based on:
    /// - model_id: Filter by specific model
    /// - worker_type: Filter by worker type (Regular, Prefill, Decode)
    /// - connection_mode: Filter by connection mode (Http, Grpc)
    /// - healthy_only: Only return healthy workers
    pub fn get_workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
        healthy_only: bool,
    ) -> Vec<Arc<dyn Worker>> {
        // Start with the most efficient collection based on filters
        // Use model index when possible as it's O(1) lookup
        let workers = if let Some(model) = model_id {
            self.get_by_model_fast(model)
        } else {
            self.get_all()
        };

        // Apply remaining filters
        workers
            .into_iter()
            .filter(|w| {
                // Check worker_type if specified
                if let Some(ref wtype) = worker_type {
                    if w.worker_type() != *wtype {
                        return false;
                    }
                }

                // Check connection_mode if specified
                if let Some(ref conn) = connection_mode {
                    if w.connection_mode() != *conn {
                        return false;
                    }
                }

                // Check health if required
                if healthy_only && !w.is_healthy() {
                    return false;
                }

                true
            })
            .collect()
    }

    /// Get worker statistics
    pub fn stats(&self) -> WorkerRegistryStats {
        let total_workers = self.workers.len();
        let total_models = self.get_models().len();

        let mut healthy_count = 0;
        let mut total_load = 0;
        let mut regular_count = 0;
        let mut prefill_count = 0;
        let mut decode_count = 0;

        for worker in self.get_all() {
            if worker.is_healthy() {
                healthy_count += 1;
            }
            total_load += worker.load();

            match worker.worker_type() {
                WorkerType::Regular => regular_count += 1,
                WorkerType::Prefill { .. } => prefill_count += 1,
                WorkerType::Decode => decode_count += 1,
            }
        }

        WorkerRegistryStats {
            total_workers,
            total_models,
            healthy_workers: healthy_count,
            total_load,
            regular_workers: regular_count,
            prefill_workers: prefill_count,
            decode_workers: decode_count,
        }
    }

    /// Start a health checker for all workers in the registry
    /// This should be called once after the registry is populated with workers
    pub fn start_health_checker(&self, check_interval_secs: u64) -> crate::core::HealthChecker {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let workers_ref = self.workers.clone();

        let handle = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(check_interval_secs));

            // Counter for periodic load reset (every 10 health check cycles)
            let mut check_count = 0u64;
            const LOAD_RESET_INTERVAL: u64 = 10;

            loop {
                interval.tick().await;

                // Check for shutdown signal
                if shutdown_clone.load(Ordering::Acquire) {
                    tracing::debug!("Registry health checker shutting down");
                    break;
                }

                // Get all workers from registry
                let workers: Vec<Arc<dyn crate::core::Worker>> = workers_ref
                    .iter()
                    .map(|entry| entry.value().clone())
                    .collect();

                // Perform health checks
                for worker in &workers {
                    let _ = worker.check_health_async().await; // Use async version directly
                }

                Self::cleanup_pending_for_workers(&workers, PENDING_MESSAGE_TTL);

                // Reset loads periodically
                check_count += 1;
                if check_count.is_multiple_of(LOAD_RESET_INTERVAL) {
                    tracing::debug!("Resetting worker loads (cycle {})", check_count);
                    for worker in &workers {
                        worker.reset_load();
                    }
                }
            }
        });

        crate::core::HealthChecker::new(handle, shutdown)
    }

    /// Start a background task that checks for unacknowledged messages and resends them.
    /// This task runs every 50ms and resends messages that haven't been acknowledged
    /// within the RESEND_TIMEOUT_MS threshold.
    pub fn start_resend_checker(&self) -> crate::core::HealthChecker {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let workers_ref = self.workers.clone();

        let handle = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_millis(RESEND_CHECK_INTERVAL_MS));

            loop {
                interval.tick().await;

                // Check for shutdown signal
                if shutdown_clone.load(Ordering::Acquire) {
                    tracing::debug!("Registry resend checker shutting down");
                    break;
                }

                // Get all workers from registry
                let workers: Vec<Arc<dyn crate::core::Worker>> = workers_ref
                    .iter()
                    .map(|entry| entry.value().clone())
                    .collect();

                // Check each worker for messages to resend
                for worker in &workers {
                    let messages_to_resend = worker.get_messages_to_resend();
                    for message in &messages_to_resend {
                        // Resend the message
                        resend_message(worker.url(), message).await;
                        // Mark it as resent (increment counter and update timestamp)
                        worker.mark_resent(message.request_id.as_str());
                    }
                }
            }
        });

        crate::core::HealthChecker::new(handle, shutdown)
    }

    /// Update worker stats for a given worker URL
    pub fn update_stats(&self, worker_url: &str, stats: WorkerStats) {
        if let Some(worker) = self.get_by_url(worker_url) {
            let pending_before = worker.pending_message_debug_info();
            let accepted_request_ids = stats.accepted_request_ids.clone().unwrap_or_default();
            let accepted_request_count = accepted_request_ids.len();
            let accepted_request_sample = accepted_request_ids
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(",");
            let mut removed_count = 0usize;
            let mut removed_messages = Vec::new();

            if !accepted_request_ids.is_empty() {
                let now = Instant::now();
                let removed = worker.remove_pending_requests_by_id(&accepted_request_ids);
                removed_count = removed.len();
                for message in removed.iter() {
                    let latency = now.saturating_duration_since(message.timestamp);
                    RouterMetrics::record_message_ack(worker.url(), latency);
                }
                removed_messages = removed;
            }

            RouterMetrics::set_pending_messages(worker.url(), worker.pending_message_count());

            let pending_after = worker.pending_message_debug_info();
            let is_idle_from_stats = stats.num_requests == 0 && stats.waiting_queue_size == 0;

            if removed_count > 0 && pending_after.count == 0 {
                let removed_request_ids = removed_messages
                    .iter()
                    .map(|msg| msg.request_id.as_str())
                    .take(5)
                    .collect::<Vec<_>>()
                    .join(",");

                tracing::info!(
                    worker = worker.url(),
                    iter = stats.iteration_num,
                    num_requests = stats.num_requests,
                    waiting_queue = stats.waiting_queue_size,
                    batch_tokens = stats.batch_size_tokens,
                    accepted_request_count = accepted_request_count,
                    accepted_request_sample = accepted_request_sample.as_str(),
                    removed = removed_count,
                    removed_request_ids = removed_request_ids,
                    pending_before = pending_before.compact_string(),
                    pending_after = pending_after.compact_string(),
                    "[PENDING_TRACE] worker ack advanced router pending ledger"
                );
            }

            if pending_before.count > 0
                && removed_count == 0
                && accepted_request_count > 0
                && self.should_log_pending_diag(worker.url(), "ack_no_progress")
            {
                tracing::warn!(
                    worker = worker.url(),
                    iter = stats.iteration_num,
                    num_requests = stats.num_requests,
                    waiting_queue = stats.waiting_queue_size,
                    batch_tokens = stats.batch_size_tokens,
                    accepted_request_count = accepted_request_count,
                    accepted_request_sample = accepted_request_sample.as_str(),
                    pending_before = pending_before.compact_string(),
                    pending_after = pending_after.compact_string(),
                    "[ACK_NO_PROGRESS] Worker stats carried accepted_request_ids but router pending ledger did not advance"
                );
            }

            if pending_after.count > 0
                && accepted_request_count == 0
                && self.should_log_pending_diag(worker.url(), "ack_missing")
            {
                tracing::warn!(
                    worker = worker.url(),
                    iter = stats.iteration_num,
                    num_requests = stats.num_requests,
                    waiting_queue = stats.waiting_queue_size,
                    batch_tokens = stats.batch_size_tokens,
                    accepted_request_count = accepted_request_count,
                    pending = pending_after.compact_string(),
                    "[ACK_MISSING] Worker still has router pending messages but sidecar stats did not include accepted_request_ids"
                );
            }

            if pending_after.count > 0
                && is_idle_from_stats
                && self.should_log_pending_diag(worker.url(), "pending_stall")
            {
                tracing::warn!(
                    worker = worker.url(),
                    iter = stats.iteration_num,
                    forward_mode = stats.forward_mode.as_str(),
                    batch_tokens = stats.batch_size_tokens,
                    accepted_request_count = accepted_request_count,
                    accepted_request_sample = accepted_request_sample.as_str(),
                    pending = pending_after.compact_string(),
                    "[PENDING_STALL] Worker stats are idle (num_requests=0, waiting_queue=0) but router pending ledger is still non-empty"
                );
            }
        }

        self.worker_stats.insert(worker_url.to_string(), stats);
    }

    fn cleanup_pending_for_workers(workers: &[Arc<dyn Worker>], ttl: Duration) {
        for worker in workers {
            worker.cleanup_pending_messages(ttl);
        }
    }

    /// Try to resolve a worker identifier (ID, URL, or `<host>:<port>[:dp#]`) to a registered URL
    pub fn resolve_worker_url(&self, identifier: &str) -> Option<String> {
        if identifier.is_empty() {
            return None;
        }

        if self.get_by_url(identifier).is_some() {
            return Some(identifier.to_string());
        }

        let (host_port, dp_rank) = Self::parse_worker_identifier(identifier)?;
        for worker in self.get_all() {
            if let Some(expected_dp) = dp_rank {
                if worker.dp_rank() != Some(expected_dp) {
                    continue;
                }
            } else if worker.dp_rank().is_some() {
                // Avoid matching DP-aware workers when stats omit dp rank, since multiple entries share host:port
                continue;
            }

            if let Some(candidate_host_port) = Self::host_port_from_url(worker.base_url()) {
                if candidate_host_port == host_port {
                    return Some(worker.url().to_string());
                }
            }
        }

        None
    }

    /// Get worker stats for a given worker URL
    pub fn get_stats(&self, worker_url: &str) -> Option<WorkerStats> {
        self.worker_stats.get(worker_url).map(|s| s.clone())
    }

    /// Get all worker stats as a HashMap
    pub fn get_all_stats(&self) -> HashMap<String, WorkerStats> {
        self.worker_stats
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Remove stats for a given worker URL
    pub fn remove_stats(&self, worker_url: &str) {
        self.worker_stats.remove(worker_url);
    }

    fn parse_worker_identifier(identifier: &str) -> Option<(String, Option<usize>)> {
        let mut base = identifier.trim();
        let mut dp_rank = None;

        while let Some((rest, suffix)) = base.rsplit_once(':') {
            if let Some(rank) = suffix.strip_prefix("dp") {
                if dp_rank.is_none() {
                    if let Ok(num) = rank.parse::<usize>() {
                        dp_rank = Some(num);
                        base = rest;
                        continue;
                    }
                }
            }

            if let Some(rank) = suffix.strip_prefix("tp") {
                if rank.parse::<usize>().is_ok() {
                    base = rest;
                    continue;
                }
            }

            break;
        }

        let host_port = if base.contains("://") {
            Self::host_port_from_url(base)?
        } else {
            Self::host_port_from_host_port(base)?
        };

        Some((host_port, dp_rank))
    }

    fn host_port_from_url(url_str: &str) -> Option<String> {
        if let Ok(parsed) = Url::parse(url_str) {
            let host = parsed.host_str()?.to_ascii_lowercase();
            let port = parsed.port_or_known_default()?;
            return Some(format!("{host}:{port}"));
        }

        if let Some((base, _)) = url_str.rsplit_once('@') {
            if let Ok(parsed) = Url::parse(base) {
                let host = parsed.host_str()?.to_ascii_lowercase();
                let port = parsed.port_or_known_default()?;
                return Some(format!("{host}:{port}"));
            }
        }

        None
    }

    fn host_port_from_host_port(input: &str) -> Option<String> {
        let trimmed = input.trim().trim_matches('/');
        if trimmed.is_empty() {
            return None;
        }

        let (host_part, port_part) = trimmed.rsplit_once(':')?;
        let port = port_part.parse::<u16>().ok()?;
        let host = host_part
            .trim()
            .trim_matches(|c| c == '[' || c == ']')
            .to_ascii_lowercase();

        Some(format!("{host}:{port}"))
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the worker registry
#[derive(Debug, Clone)]
pub struct WorkerRegistryStats {
    pub total_workers: usize,
    pub total_models: usize,
    pub healthy_workers: usize,
    pub total_load: usize,
    pub regular_workers: usize,
    pub prefill_workers: usize,
    pub decode_workers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        BasicWorkerBuilder, CircuitBreakerConfig, DPAwareWorkerBuilder, PendingMessage,
    };
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    #[test]
    fn test_worker_registry() {
        let registry = WorkerRegistry::new();

        // Create a worker with labels
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "llama-3-8b".to_string());
        labels.insert("priority".to_string(), "50".to_string());
        labels.insert("cost".to_string(), "0.8".to_string());

        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register worker (WorkerFactory returns Box<dyn Worker>, convert to Arc)
        let worker_id = registry.register(Arc::from(worker));

        // Verify registration
        assert!(registry.get(&worker_id).is_some());
        assert!(registry.get_by_url("http://worker1:8080").is_some());
        assert_eq!(registry.get_by_model("llama-3-8b").len(), 1);
        assert_eq!(registry.get_by_type(&WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(&ConnectionMode::Http).len(), 1);

        // Test stats
        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.total_models, 1);

        // Remove worker
        registry.remove(&worker_id);
        assert!(registry.get(&worker_id).is_none());
    }

    #[test]
    fn test_model_index_fast_lookup() {
        let registry = WorkerRegistry::new();

        // Create workers for different models
        let mut labels1 = HashMap::new();
        labels1.insert("model_id".to_string(), "llama-3".to_string());
        let worker1: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels1)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels2 = HashMap::new();
        labels2.insert("model_id".to_string(), "llama-3".to_string());
        let worker2: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker2:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels2)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels3 = HashMap::new();
        labels3.insert("model_id".to_string(), "gpt-4".to_string());
        let worker3: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker3:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels3)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register workers
        registry.register(Arc::from(worker1));
        registry.register(Arc::from(worker2));
        registry.register(Arc::from(worker3));

        // Test get_by_model_fast for llama-3
        let llama_workers = registry.get_by_model_fast("llama-3");
        assert_eq!(llama_workers.len(), 2);
        let urls: Vec<String> = llama_workers.iter().map(|w| w.url().to_string()).collect();
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));

        // Test get_by_model_fast for gpt-4
        let gpt_workers = registry.get_by_model_fast("gpt-4");
        assert_eq!(gpt_workers.len(), 1);
        assert_eq!(gpt_workers[0].url(), "http://worker3:8080");

        // Test get_by_model_fast for non-existent model
        let unknown_workers = registry.get_by_model_fast("unknown-model");
        assert_eq!(unknown_workers.len(), 0);

        // Test that both get_by_model and get_by_model_fast return same results
        let llama_workers_slow = registry.get_by_model("llama-3");
        assert_eq!(llama_workers.len(), llama_workers_slow.len());

        // Test removal updates the model index
        registry.remove_by_url("http://worker1:8080");
        let llama_workers_after = registry.get_by_model_fast("llama-3");
        assert_eq!(llama_workers_after.len(), 1);
        assert_eq!(llama_workers_after[0].url(), "http://worker2:8080");
    }

    #[test]
    fn test_resolve_worker_url_basic_identifier() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker-basic:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        registry.register(worker);

        let resolved = registry
            .resolve_worker_url("worker-basic:8080")
            .expect("should resolve host:port identifier");
        assert_eq!(resolved, "http://worker-basic:8080");

        let resolved_url = registry
            .resolve_worker_url("http://worker-basic:8080")
            .expect("should accept existing url");
        assert_eq!(resolved_url, "http://worker-basic:8080");
    }

    #[test]
    fn test_resolve_worker_url_dp_identifier() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> =
            Arc::new(DPAwareWorkerBuilder::new("http://worker-dp:9090", 1, 2).build());
        registry.register(worker);

        let resolved = registry
            .resolve_worker_url("worker-dp:9090:dp1")
            .expect("should resolve dp-aware worker");
        assert_eq!(resolved, "http://worker-dp:9090@1");

        // Missing dp rank is ambiguous across DP workers and should not resolve
        assert!(registry.resolve_worker_url("worker-dp:9090").is_none());
    }

    #[test]
    fn test_update_stats_clears_acknowledged_messages() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker-ack:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        registry.register(worker.clone());

        let pending = PendingMessage::new(
            "/generate",
            "req-1",
            serde_json::json!({"test": true}),
            512, // default token count for tests
        );
        worker.add_pending_message(pending);
        assert_eq!(worker.pending_message_count(), 1);

        let stats = WorkerStats {
            worker_id: worker.url().to_string(),
            batch_size_tokens: 0,
            kv_tokens_used: Some(0),
            num_requests: 0,
            waiting_queue_size: 0,
            waiting_queue_info: None,
            forward_mode: "UNKNOWN".to_string(),
            iteration_num: 0,
            worker_iteration_id: None,
            report_send_time_ms: None,
            last_iteration_time_ms: None,
            prefill_chunk_pairs: None,
            prefill_sim_metrics: None,
            accepted_request_ids: Some(vec!["req-1".to_string()]),
            batch_size_by_tpot_tier: None,
            timestamp: Instant::now(),
        };

        registry.update_stats(worker.url(), stats);
        assert_eq!(worker.pending_message_count(), 0);
    }

    #[test]
    fn test_update_stats_clears_acknowledged_messages_with_parallel_sample_suffix() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker-ack-batch:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        registry.register(worker.clone());

        let pending =
            PendingMessage::new("/generate", "req-1", serde_json::json!({"test": true}), 512);
        worker.add_pending_message(pending);
        assert_eq!(worker.pending_message_count(), 1);

        let stats = WorkerStats {
            worker_id: worker.url().to_string(),
            batch_size_tokens: 0,
            kv_tokens_used: Some(0),
            num_requests: 0,
            waiting_queue_size: 0,
            waiting_queue_info: None,
            forward_mode: "UNKNOWN".to_string(),
            iteration_num: 0,
            worker_iteration_id: None,
            report_send_time_ms: None,
            last_iteration_time_ms: None,
            prefill_chunk_pairs: None,
            prefill_sim_metrics: None,
            accepted_request_ids: Some(vec!["req-1_0".to_string()]),
            batch_size_by_tpot_tier: None,
            timestamp: Instant::now(),
        };

        registry.update_stats(worker.url(), stats);
        assert_eq!(worker.pending_message_count(), 0);
    }

    #[test]
    fn test_cleanup_pending_messages_removes_stale_entries() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker-ttl:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        registry.register(worker.clone());

        let mut pending = PendingMessage::new(
            "/generate",
            "req-stale".to_string(),
            serde_json::json!({"test": true}),
            512, // default token count for tests
        );
        pending.timestamp = Instant::now() - Duration::from_secs(2);
        worker.add_pending_message(pending);
        assert_eq!(worker.pending_message_count(), 1);

        worker.cleanup_pending_messages(Duration::from_secs(1));
        assert_eq!(worker.pending_message_count(), 0);
    }
}
