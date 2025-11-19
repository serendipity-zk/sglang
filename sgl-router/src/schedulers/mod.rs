use std::fmt::Debug;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, warn};

use crate::core::{PendingMessage, Worker, WorkerRegistry};
use crate::metrics::RouterMetrics;
use crate::routers::header_utils;
use crate::routers::http::scheduler::{PendingRequest, SchedulerConfig};
use crate::ui::RouterUi;
use serde_json::json;

mod eager;
mod factory;
mod gated;
mod registry;
mod slo_aware;

pub use eager::EagerScheduler;
pub use factory::SchedulerFactory;
pub use gated::GatedScheduler;
pub use registry::SchedulerRegistry;
pub use slo_aware::SloAwareScheduler;

pub(crate) const SCHEDULER_TICK_INTERVAL_MS: u64 = 5;

#[async_trait]
pub trait SchedulerBase: Send + Sync + Debug + 'static {
    async fn dispatch_to_worker(
        self: Arc<Self>,
        config: Arc<SchedulerConfig>,
        request: PendingRequest,
        worker: Arc<dyn Worker>,
    ) {
        // Generate message ID and add pending message BEFORE async spawn
        // This prevents race conditions where multiple requests select the same worker
        let (message_id, generation) = if request.route == "/generate" {
            let msg_id = worker.next_message_id();
            let gen = worker.generation();

            // Extract request_id from body for tracking
            let request_id = request.body_json
                .get("request_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            // Add pending message NOW, before spawning async task
            worker.add_pending_message(PendingMessage::new(
                msg_id,
                gen,
                request.route.clone(),
                request_id,
            ));

            (Some(msg_id), Some(gen))
        } else {
            (None, None)
        };

        tokio::spawn(async move {
            process_pending(self, config, request, worker, message_id, generation).await;
        });
    }

    async fn send_http_request(
        &self,
        config: &Arc<SchedulerConfig>,
        headers: Option<&HeaderMap>,
        body_json: &serde_json::Value,
        route: &str,
        worker_url: &str,
        is_stream: bool,
        load_incremented: bool,
    ) -> Response {
        send_http_request_impl(
            config,
            headers,
            body_json,
            route,
            worker_url,
            is_stream,
            load_incremented,
        )
        .await
    }

    fn handle_timeout(&self, request: PendingRequest) {
        handle_timeout_impl(request);
    }
}

pub trait Scheduler: SchedulerBase {
    fn spawn(
        self: Arc<Self>,
        config: SchedulerConfig,
        pending_rx: mpsc::Receiver<PendingRequest>,
    ) -> JoinHandle<()>;

    fn name(&self) -> &'static str;

    /// Downcast support for accessing concrete scheduler types
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Result of attempting to select a worker for a pending request.
pub(crate) enum WorkerSelection {
    Selected(Arc<dyn Worker>),
    NoWorkers,
    PolicyDeferred,
}

pub(crate) fn available_workers_for_request(
    worker_registry: &Arc<WorkerRegistry>,
    model_id: Option<&str>,
) -> Vec<Arc<dyn Worker>> {
    let workers = match model_id {
        Some(model) => worker_registry.get_by_model_fast(model),
        None => worker_registry.get_all(),
    };

    workers
        .into_iter()
        .filter(|worker| worker.is_available())
        .collect()
}

pub(crate) fn select_worker_for_request(
    config: &Arc<SchedulerConfig>,
    request: &PendingRequest,
) -> WorkerSelection {
    let available =
        available_workers_for_request(&config.worker_registry, request.model_id.as_deref());

    if available.is_empty() {
        return WorkerSelection::NoWorkers;
    }

    let policy = match request.model_id.as_deref() {
        Some(model) => config.policy_registry.get_policy_or_default(model),
        None => config.policy_registry.get_default_policy(),
    };

    match policy.select_worker(&available, Some(&request.text)) {
        Some(idx) => WorkerSelection::Selected(available[idx].clone()),
        None => WorkerSelection::PolicyDeferred,
    }
}

pub(crate) fn has_request_timed_out(
    config: &Arc<SchedulerConfig>,
    request: &PendingRequest,
) -> bool {
    if config.queue_timeout.is_zero() {
        return false;
    }

    request.enqueue_started.elapsed() > config.queue_timeout
}

async fn process_pending<S: SchedulerBase + ?Sized>(
    scheduler: Arc<S>,
    config: Arc<SchedulerConfig>,
    pending: PendingRequest,
    worker: Arc<dyn Worker>,
    message_id: Option<i64>,
    generation: Option<i64>,
) {
    let PendingRequest {
        headers,
        body_json,
        route,
        model_id,
        is_stream,
        text: _,
        enqueue_started: _,
        response_tx,
        target_ttft_ms: _,
        target_tpot_ms: _,
    } = pending;

    let start = Instant::now();

    let response = dispatch_request(
        &scheduler,
        &config,
        headers.as_ref(),
        &body_json,
        &route,
        model_id.as_deref(),
        is_stream,
        worker,
        message_id,
        generation,
    )
    .await;

    if response.status().is_success() {
        RouterMetrics::record_request(&route);
        RouterMetrics::record_generate_duration(start.elapsed());
    } else {
        RouterMetrics::record_request_error(&route, "request_failed");
    }

    if response_tx.send(response).is_err() {
        warn!(
            route = route,
            "Response channel closed before scheduler could reply"
        );
    }
}

enum RequestPayload<'a> {
    Borrowed(&'a serde_json::Value),
    Owned(serde_json::Value),
}

impl<'a> RequestPayload<'a> {
    fn as_ref(&self) -> &serde_json::Value {
        match self {
            RequestPayload::Borrowed(value) => value,
            RequestPayload::Owned(value) => value,
        }
    }
}

fn prepare_request_payload<'a>(
    route: &str,
    body_json: &'a serde_json::Value,
    message_id: Option<i64>,
    generation: Option<i64>,
) -> RequestPayload<'a> {
    if route != "/generate" || message_id.is_none() || generation.is_none() {
        return RequestPayload::Borrowed(body_json);
    }

    if !body_json.is_object() {
        warn!(
            route = route,
            "Expected JSON object for /generate, skipping router message metadata injection"
        );
        return RequestPayload::Borrowed(body_json);
    }

    let mut owned = body_json.clone();

    if let Some(map) = owned.as_object_mut() {
        map.insert("router_generation".to_string(), json!(generation.unwrap()));
        map.insert("router_message_id".to_string(), json!(message_id.unwrap()));

        RequestPayload::Owned(owned)
    } else {
        RequestPayload::Borrowed(body_json)
    }
}

async fn dispatch_request<S: SchedulerBase + ?Sized>(
    scheduler: &Arc<S>,
    config: &Arc<SchedulerConfig>,
    headers: Option<&HeaderMap>,
    body_json: &serde_json::Value,
    route: &str,
    model_id: Option<&str>,
    is_stream: bool,
    worker: Arc<dyn Worker>,
    message_id: Option<i64>,
    generation: Option<i64>,
) -> Response {
    info!(
        "Selected worker for model: {} worker_url={}",
        model_id.unwrap_or("default"),
        worker.url()
    );

    // Increment pending dispatch counter - will be reset when worker stats arrive

    let policy = match model_id {
        Some(model) => config.policy_registry.get_policy_or_default(model),
        None => config.policy_registry.get_default_policy(),
    };

    let load_incremented = if policy.name() == "cache_aware" {
        worker.increment_load();
        RouterMetrics::set_running_requests(worker.url(), worker.load());
        true
    } else {
        false
    };

    RouterUi::inc_worker_issued(worker.url());

    let request_payload = prepare_request_payload(route, body_json, message_id, generation);

    let response = scheduler
        .send_http_request(
            config,
            headers,
            request_payload.as_ref(),
            route,
            worker.url(),
            is_stream,
            load_incremented,
        )
        .await;

    worker.record_outcome(response.status().is_success());

    response
}

async fn send_http_request_impl(
    config: &Arc<SchedulerConfig>,
    headers: Option<&HeaderMap>,
    body_json: &serde_json::Value,
    route: &str,
    worker_url: &str,
    is_stream: bool,
    load_incremented: bool,
) -> Response {
    let mut request_builder = if config.dp_aware {
        let (worker_url_prefix, dp_rank) =
            match crate::routers::http::router::Router::extract_dp_rank(worker_url) {
                Ok(parts) => parts,
                Err(e) => {
                    error!("Failed to extract dp_rank: {}", e);
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to extract dp_rank: {}", e),
                    )
                        .into_response();
                }
            };

        let mut json_val = body_json.clone();
        if let Some(map) = json_val.as_object_mut() {
            map.insert(
                String::from("data_parallel_rank"),
                serde_json::json!(dp_rank),
            );
            debug!(
                "Modified request body with data_parallel_rank: {}",
                serde_json::to_string(&json_val).unwrap_or_else(|_| String::from("ERR"))
            );
        } else {
            return (
                StatusCode::BAD_REQUEST,
                "Failed to insert the data_parallel_rank field into the request body",
            )
                .into_response();
        }

        config
            .client
            .post(format!("{}{}", worker_url_prefix, route))
            .json(&json_val)
    } else {
        config
            .client
            .post(format!("{}{}", worker_url, route))
            .json(body_json)
    };

    if let Some(headers) = headers {
        for (name, value) in headers {
            if *name != CONTENT_TYPE && *name != CONTENT_LENGTH {
                request_builder = request_builder.header(name, value);
            }
        }
    }

    let res = match request_builder.send().await {
        Ok(res) => res,
        Err(e) => {
            error!(
                "Failed to send typed request worker_url={} route={} error={}",
                worker_url, route, e
            );

            if load_incremented {
                if let Some(worker) = config.worker_registry.get_by_url(worker_url) {
                    worker.decrement_load();
                    RouterMetrics::set_running_requests(worker_url, worker.load());
                }
            }

            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Request failed: {}", e),
            )
                .into_response();
        }
    };

    let status =
        StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    if !is_stream {
        let response_headers = header_utils::preserve_response_headers(res.headers());

        let response = match res.bytes().await {
            Ok(body) => {
                let mut response = Response::new(axum::body::Body::from(body));
                *response.status_mut() = status;
                *response.headers_mut() = response_headers;
                response
            }
            Err(e) => {
                if load_incremented {
                    if let Some(worker) = config.worker_registry.get_by_url(worker_url) {
                        worker.decrement_load();
                        RouterMetrics::set_running_requests(worker_url, worker.load());
                    }
                }

                let error_msg = format!("Failed to get response body: {}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, error_msg).into_response()
            }
        };

        if load_incremented {
            if let Some(worker) = config.worker_registry.get_by_url(worker_url) {
                worker.decrement_load();
                RouterMetrics::set_running_requests(worker_url, worker.load());
            }
        }

        response
    } else if load_incremented {
        let registry = Arc::clone(&config.worker_registry);
        let worker_url = worker_url.to_string();

        let mut response_headers = header_utils::preserve_response_headers(res.headers());
        response_headers.insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("text/event-stream"),
        );

        let stream = res.bytes_stream();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let mut stream = stream;
            let mut decremented = false;
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        if bytes
                            .as_ref()
                            .windows(12)
                            .any(|window| window == b"data: [DONE]")
                        {
                            if let Some(worker) = registry.get_by_url(&worker_url) {
                                worker.decrement_load();
                                RouterMetrics::set_running_requests(&worker_url, worker.load());
                                decremented = true;
                            }
                        }
                        if tx.send(Ok(bytes)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(format!("Stream error: {}", e)));
                        break;
                    }
                }
            }
            if !decremented {
                if let Some(worker) = registry.get_by_url(&worker_url) {
                    worker.decrement_load();
                    RouterMetrics::set_running_requests(&worker_url, worker.load());
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let body = axum::body::Body::from_stream(stream);

        let mut response = Response::new(body);
        *response.status_mut() = status;
        *response.headers_mut() = response_headers;
        response
    } else {
        let mut response_headers = header_utils::preserve_response_headers(res.headers());
        response_headers.insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("text/event-stream"),
        );

        let stream = res.bytes_stream();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let mut stream = stream;
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        if tx.send(Ok(bytes)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(format!("Stream error: {}", e)));
                        break;
                    }
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let body = axum::body::Body::from_stream(stream);

        let mut response = Response::new(body);
        *response.status_mut() = status;
        *response.headers_mut() = response_headers;
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BasicWorkerBuilder;

    #[test]
    fn prepare_request_payload_injects_metadata_and_tracks_pending() {
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://worker:8080").build());
        let body = json!({
            "prompt": "hello",
            "request_id": "req-test"
        });

        // Simulate what dispatch_to_worker does: generate ID and add pending message
        let message_id = worker.next_message_id();
        let generation = worker.generation();
        worker.add_pending_message(PendingMessage::new(
            message_id,
            generation,
            "/generate".to_string(),
            Some("req-test".to_string()),
        ));

        let payload = prepare_request_payload("/generate", &body, Some(message_id), Some(generation));
        let value = payload.as_ref();

        // First message from this worker should have ID 0
        assert_eq!(
            value.get("router_message_id").and_then(|v| v.as_i64()),
            Some(0)
        );
        // Should have a valid generation (ROUTER_GENERATION)
        assert!(value.get("router_generation").and_then(|v| v.as_i64()).is_some());
        assert_eq!(worker.pending_message_count(), 1);
    }

    #[test]
    fn prepare_request_payload_skips_non_generate_routes() {
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://worker:8080").build());
        let body = json!({
            "prompt": "hello",
            "request_id": "req-test"
        });

        // For non-/generate routes, no message ID should be passed
        let payload = prepare_request_payload("/v1/chat/completions", &body, None, None);
        let value = payload.as_ref();

        assert!(value.get("router_message_id").is_none());
        assert!(value.get("router_generation").is_none());
        assert_eq!(worker.pending_message_count(), 0);
    }

    #[test]
    fn prepare_request_payload_logs_warning_for_non_object_body() {
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://worker:8080").build());
        let body = serde_json::Value::String("not-an-object".to_string());

        // Even with message_id/generation, non-object bodies should be rejected
        let message_id = worker.next_message_id();
        let generation = worker.generation();
        let payload = prepare_request_payload("/generate", &body, Some(message_id), Some(generation));
        let value = payload.as_ref();

        assert!(value.get("router_message_id").is_none());
        assert_eq!(worker.pending_message_count(), 0);
    }
}

fn handle_timeout_impl(request: PendingRequest) {
    RouterUi::dec_queue();
    RouterMetrics::record_request_error(&request.route, "queue_timeout");
    let _ = request.response_tx.send(
        (
            StatusCode::REQUEST_TIMEOUT,
            "Request timed out while waiting to be scheduled",
        )
            .into_response(),
    );
}
