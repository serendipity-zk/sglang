//! Gated round-robin policy that defers when all workers have queued requests
//!
//! This policy is designed for use with the gated scheduler. It returns None
//! (defers scheduling) when ALL workers have pending work in their queues,
//! preventing over-subscription. It accounts for both:
//! - Worker-reported queue size (from WorkerStats)
//! - Router-tracked pending dispatches (not yet in stats)

use super::{get_healthy_worker_indices, LoadBalancingPolicy};
use crate::core::{Worker, WorkerStats};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Gated round-robin policy
///
/// This policy implements round-robin worker selection with a key difference:
/// it defers scheduling (returns None) when all workers have pending work.
/// This prevents over-subscribing workers and works optimally with the gated scheduler.
#[derive(Debug)]
pub struct GatedRoundRobinPolicy {
    counter: AtomicUsize,
    worker_stats: RwLock<HashMap<String, WorkerStats>>,
}

impl GatedRoundRobinPolicy {
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
            worker_stats: RwLock::new(HashMap::new()),
        }
    }

    /// Check if worker has pending work (queued + pending dispatches)
    fn has_pending_work(&self, worker: &dyn Worker) -> bool {
        if worker.has_send_gap() {
            return true;
        }

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

        tracing::warn!(
            "worker {} has pending work: queue_size={}, pending_messages={}",
            worker.url(),
            queue_size,
            pending_messages
        );
        // Worker is busy if either queue has work or pending messages exist
        (queue_size + pending_messages as i64) > 0
    }
}

impl Default for GatedRoundRobinPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingPolicy for GatedRoundRobinPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Check if ANY worker is available (no pending work)
        let has_available = healthy_indices
            .iter()
            .any(|&idx| !self.has_pending_work(workers[idx].as_ref()));

        if !has_available {
            // All workers busy - defer scheduling
            tracing::warn!("[GATED_RR] All workers busy, deferring");
            return None;
        }

        // Round-robin search for available worker
        let start_count = self.counter.fetch_add(1, Ordering::Relaxed);

        for i in 0..healthy_indices.len() {
            let idx = (start_count + i) % healthy_indices.len();
            let worker_idx = healthy_indices[idx];

            if !self.has_pending_work(workers[worker_idx].as_ref()) {
                tracing::warn!(
                    "[GATED_RR] Selected worker {} (index {})",
                    workers[worker_idx].url(),
                    worker_idx
                );
                return Some(worker_idx);
            }
        }

        // Shouldn't reach here if has_available was true
        tracing::warn!("[GATED_RR] No available worker found despite has_available=true");
        None
    }

    fn update_worker_stats(&self, stats: &HashMap<String, WorkerStats>) {
        if let Ok(mut cached) = self.worker_stats.write() {
            *cached = stats.clone();
        }
    }

    fn name(&self) -> &'static str {
        "gated_round_robin"
    }

    fn reset(&self) {
        self.counter.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, CircuitBreakerConfig};
    use std::collections::HashMap;

    #[test]
    fn test_gated_round_robin_basic() {
        let policy = GatedRoundRobinPolicy::new();
        assert_eq!(policy.name(), "gated_round_robin");
    }

    #[test]
    fn test_defers_when_all_workers_busy() {
        let policy = GatedRoundRobinPolicy::new();

        // Create workers
        let worker1 = Arc::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );
        let worker2 = Arc::new(
            BasicWorkerBuilder::new("http://worker2:8080")
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );

        let workers: Vec<Arc<dyn Worker>> = vec![worker1.clone(), worker2.clone()];

        // Simulate both workers having queue
        let mut stats = HashMap::new();
        stats.insert(
            "http://worker1:8080".to_string(),
            WorkerStats {
                worker_id: "http://worker1:8080".to_string(),
                batch_size_tokens: 100,
                kv_tokens_used: None,
                num_requests: 2,
                waiting_queue_size: 3,
                waiting_queue_info: None,
                forward_mode: "DECODE".to_string(),
                iteration_num: 10,
                last_iteration_time_ms: None,
                prefill_chunk_pairs: None,
                prefill_sim_metrics: None,
                router_generation: None,
                last_received_message_id: None,
                batch_size_by_tpot_tier: None,
                timestamp: std::time::Instant::now(),
            },
        );
        stats.insert(
            "http://worker2:8080".to_string(),
            WorkerStats {
                worker_id: "http://worker2:8080".to_string(),
                batch_size_tokens: 100,
                kv_tokens_used: None,
                num_requests: 1,
                waiting_queue_size: 2,
                waiting_queue_info: None,
                forward_mode: "DECODE".to_string(),
                iteration_num: 10,
                last_iteration_time_ms: None,
                prefill_chunk_pairs: None,
                prefill_sim_metrics: None,
                router_generation: None,
                last_received_message_id: None,
                batch_size_by_tpot_tier: None,
                timestamp: std::time::Instant::now(),
            },
        );

        policy.update_worker_stats(&stats);

        // Should defer when all workers busy
        let result = policy.select_worker(&workers, None);
        assert_eq!(result, None, "Should defer when all workers have queue");
    }

    #[test]
    fn test_selects_available_worker() {
        let policy = GatedRoundRobinPolicy::new();

        let worker1 = Arc::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );
        let worker2 = Arc::new(
            BasicWorkerBuilder::new("http://worker2:8080")
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );

        let workers: Vec<Arc<dyn Worker>> = vec![worker1.clone(), worker2.clone()];

        // Worker1 busy, Worker2 available
        let mut stats = HashMap::new();
        stats.insert(
            "http://worker1:8080".to_string(),
            WorkerStats {
                worker_id: "http://worker1:8080".to_string(),
                batch_size_tokens: 100,
                kv_tokens_used: None,
                num_requests: 2,
                waiting_queue_size: 3,
                waiting_queue_info: None,
                forward_mode: "DECODE".to_string(),
                iteration_num: 10,
                last_iteration_time_ms: None,
                prefill_chunk_pairs: None,
                prefill_sim_metrics: None,
                router_generation: None,
                last_received_message_id: None,
                batch_size_by_tpot_tier: None,
                timestamp: std::time::Instant::now(),
            },
        );
        stats.insert(
            "http://worker2:8080".to_string(),
            WorkerStats {
                worker_id: "http://worker2:8080".to_string(),
                batch_size_tokens: 0,
                kv_tokens_used: None,
                num_requests: 0,
                waiting_queue_size: 0,
                waiting_queue_info: None,
                forward_mode: "DECODE".to_string(),
                iteration_num: 10,
                last_iteration_time_ms: None,
                prefill_chunk_pairs: None,
                prefill_sim_metrics: None,
                router_generation: None,
                last_received_message_id: None,
                batch_size_by_tpot_tier: None,
                timestamp: std::time::Instant::now(),
            },
        );

        policy.update_worker_stats(&stats);

        // Should select worker2 (available)
        let result = policy.select_worker(&workers, None);
        assert_eq!(result, Some(1), "Should select available worker");
    }

    #[test]
    fn test_pending_message_prevents_reselection() {
        // This test verifies that the race condition fix works:
        // Once a worker is selected and pending message is added,
        // the next selection should skip that worker
        let policy = GatedRoundRobinPolicy::new();

        let worker1 = Arc::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );
        let worker2 = Arc::new(
            BasicWorkerBuilder::new("http://worker2:8080")
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );

        let workers: Vec<Arc<dyn Worker>> = vec![worker1.clone(), worker2.clone()];

        // Initially both workers should have no pending work
        assert_eq!(worker1.pending_message_count(), 0);
        assert_eq!(worker2.pending_message_count(), 0);

        // First selection should return worker1 (index 0)
        let result1 = policy.select_worker(&workers, None);
        assert_eq!(result1, Some(0), "First selection should return worker1");

        // Simulate adding pending message to worker1 (what dispatch_to_worker does now)
        let msg_id = worker1.next_message_id();
        let gen = worker1.generation();
        worker1.add_pending_message(crate::core::PendingMessage::new(
            msg_id,
            gen,
            "/generate".to_string(),
            Some("test-req".to_string()),
            serde_json::json!({"test": true}),
            512, // default token count for tests
        ));

        // Verify worker1 now has pending work
        assert_eq!(worker1.pending_message_count(), 1);

        // Second selection should return worker2 (index 1) because worker1 is busy
        let result2 = policy.select_worker(&workers, None);
        assert_eq!(
            result2,
            Some(1),
            "Second selection should return worker2, not worker1"
        );

        // Add pending message to worker2 as well
        let msg_id = worker2.next_message_id();
        let gen = worker2.generation();
        worker2.add_pending_message(crate::core::PendingMessage::new(
            msg_id,
            gen,
            "/generate".to_string(),
            Some("test-req-2".to_string()),
            serde_json::json!({"test": true}),
            512, // default token count for tests
        ));

        // Now both workers are busy, should defer
        let result3 = policy.select_worker(&workers, None);
        assert_eq!(
            result3, None,
            "Should defer when all workers have pending work"
        );
    }
}
