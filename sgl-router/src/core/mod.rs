//! Core abstractions for the SGLang router
//!
//! This module contains the fundamental types and traits used throughout the router:
//! - Worker trait and implementations
//! - Error types
//! - Circuit breaker for reliability
//! - Common utilities

pub mod circuit_breaker;
pub mod error;
pub mod retry;
pub mod token_bucket;
pub mod worker;
pub mod worker_builder;
pub mod worker_registry;
pub mod worker_stats;

// Re-export commonly used types at the module level
pub use circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerStats, CircuitState,
};
pub use error::{WorkerError, WorkerResult};
pub use retry::{is_retryable_status, BackoffCalculator, RetryError, RetryExecutor};
pub use worker::{
    resend_message, send_gap_message_once, start_health_checker, BasicWorker, ConnectionMode,
    DPAwareWorker, HealthChecker, HealthConfig, PendingMessage, SendGapStatus, Worker,
    WorkerFactory, WorkerLoadGuard, WorkerType, MAX_RESEND_ATTEMPTS, RESEND_TIMEOUT_MS,
};
pub use worker_builder::{BasicWorkerBuilder, DPAwareWorkerBuilder};
pub use worker_registry::{WorkerId, WorkerRegistry, WorkerRegistryStats};
pub use worker_stats::WorkerStats;
