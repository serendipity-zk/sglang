use std::fmt::Debug;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, warn};

use crate::core::{Worker, WorkerRegistry};
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
        let dispatch_fn_start = std::time::Instant::now();

        if request.route == "/generate" {
            let input_token_count = request
                .input_token_count
                .unwrap_or_else(|| (request.text.len() / 4) as i64);
            let add_pending_start = std::time::Instant::now();
            worker.allocate_pending_message(
                request.route.clone(),
                request.request_id.clone(),
                request.body_json.clone(),
                input_token_count,
            );
            let add_pending_us = add_pending_start.elapsed().as_micros() as u64;
            if add_pending_us > 500 {
                tracing::warn!("add_pending_message took {}us", add_pending_us);
            }
        }

        let spawn_start = std::time::Instant::now();
        tokio::spawn(async move {
            process_pending(self, config, request, worker).await;
        });
        let spawn_us = spawn_start.elapsed().as_micros() as u64;

        let total_us = dispatch_fn_start.elapsed().as_micros() as u64;
        if total_us > 500 {
            tracing::warn!(
                "dispatch_to_worker total={}us (spawn={}us)",
                total_us,
                spawn_us
            );
        }
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
    ) -> DispatchResult {
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

pub struct DispatchResult {
    response: Response,
    delivered: bool,
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
) {
    let PendingRequest {
        headers,
        body_json,
        route,
        model_id,
        is_stream,
        text: _,
        request_id,
        enqueue_started,
        arrival_time_ms,
        response_tx,
        target_ttft_ms,
        target_tpot_ms,
        input_token_count,
    } = pending;

    let start = Instant::now();
    let submit_time_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let arrival_time_ms_u64 = arrival_time_ms.max(0.0) as u64;
    let router_queue_wait_ms = enqueue_started.elapsed().as_millis() as u64;
    let ingress_wait_ms = submit_time_ms.saturating_sub(arrival_time_ms_u64);

    info!(
        request_id = request_id,
        worker = worker.url(),
        route = route,
        arrival_time_ms = arrival_time_ms_u64,
        submit_time_ms = submit_time_ms,
        router_queue_wait_ms = router_queue_wait_ms,
        ingress_wait_ms = ingress_wait_ms,
        target_ttft_ms = ?target_ttft_ms,
        target_tpot_ms = ?target_tpot_ms,
        input_token_count = ?input_token_count,
        is_stream = is_stream,
        "[ROUTER_SUBMIT] dispatching request to worker"
    );

    let dispatch_result = dispatch_request(
        &scheduler,
        &config,
        headers.as_ref(),
        &body_json,
        &route,
        model_id.as_deref(),
        is_stream,
        worker,
        request_id.as_str(),
        arrival_time_ms,
    )
    .await;

    if dispatch_result.response.status().is_success() {
        RouterMetrics::record_request(&route);
        RouterMetrics::record_generate_duration(start.elapsed());
    } else {
        RouterMetrics::record_request_error(&route, "request_failed");
    }

    if response_tx.send(dispatch_result.response).is_err() {
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
    arrival_time_ms: f64,
) -> RequestPayload<'a> {
    if !body_json.is_object() {
        warn!(
            route = route,
            "Expected JSON object, skipping router metadata injection"
        );
        return RequestPayload::Borrowed(body_json);
    }

    let mut owned = body_json.clone();

    if let Some(map) = owned.as_object_mut() {
        if route == "/generate" {
            // Only inject arrival_time_ms if not already present in payload
            // (New clients provide it; old payloads need router injection)
            if !map.contains_key("arrival_time_ms") {
                map.insert("arrival_time_ms".to_string(), json!(arrival_time_ms));
            }
        }

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
    request_id: &str,
    arrival_time_ms: f64,
) -> DispatchResult {
    tracing::debug!(
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

    let request_payload = prepare_request_payload(route, body_json, arrival_time_ms);

    let dispatch_result = scheduler
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

    if !dispatch_result.delivered {
        if let Some(removed) = worker.remove_pending_request_by_id(request_id) {
            RouterMetrics::set_pending_messages(worker.url(), worker.pending_message_count());
            warn!(
                worker = worker.url(),
                request_id = request_id,
                route = route,
                status = %dispatch_result.response.status(),
                removed_request_id = removed.request_id.as_str(),
                pending_ledger = worker.pending_message_debug_info().compact_string(),
                "[PENDING_TRACE] removed undelivered pending request after dispatch transport failure"
            );
        }
    }

    worker.record_outcome(dispatch_result.response.status().is_success());

    dispatch_result
}

async fn send_http_request_impl(
    config: &Arc<SchedulerConfig>,
    headers: Option<&HeaderMap>,
    body_json: &serde_json::Value,
    route: &str,
    worker_url: &str,
    is_stream: bool,
    load_incremented: bool,
) -> DispatchResult {
    let request_id = headers
        .and_then(|header_map| header_map.get("x-request-id"))
        .and_then(|value| value.to_str().ok())
        .unwrap_or("-");
    let rid = body_json
        .get("rid")
        .and_then(|value| value.as_str())
        .unwrap_or("-");
    let arrival_time_ms = body_json
        .get("arrival_time_ms")
        .and_then(|value| value.as_f64());
    let (mut request_builder, request_url) = if config.dp_aware {
        let (worker_url_prefix, dp_rank) =
            match crate::routers::http::router::Router::extract_dp_rank(worker_url) {
                Ok(parts) => parts,
                Err(e) => {
                    error!("Failed to extract dp_rank: {}", e);
                    return DispatchResult {
                        response: (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to extract dp_rank: {}", e),
                        )
                            .into_response(),
                        delivered: false,
                    };
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
            return DispatchResult {
                response: (
                    StatusCode::BAD_REQUEST,
                    "Failed to insert the data_parallel_rank field into the request body",
                )
                    .into_response(),
                delivered: false,
            };
        }

        let request_url = format!("{}{}", worker_url_prefix, route);
        (
            config.client.post(&request_url).json(&json_val),
            request_url,
        )
    } else {
        let request_url = format!("{}{}", worker_url, route);
        (
            config.client.post(&request_url).json(body_json),
            request_url,
        )
    };

    if let Some(headers) = headers {
        for (name, value) in headers {
            if *name != CONTENT_TYPE && *name != CONTENT_LENGTH {
                request_builder = request_builder.header(name, value);
            }
        }
    }

    let http_send_time_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    info!(
        request_id = request_id,
        rid = rid,
        worker = worker_url,
        url = request_url,
        route = route,
        http_send_time_ms = http_send_time_ms,
        arrival_time_ms = ?arrival_time_ms,
        is_stream = is_stream,
        "[ROUTER_HTTP_SEND] sending request to worker over http"
    );

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

            return DispatchResult {
                response: (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Request failed: {}", e),
                )
                    .into_response(),
                delivered: false,
            };
        }
    };

    let http_headers_recv_time_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    info!(
        request_id = request_id,
        rid = rid,
        worker = worker_url,
        url = request_url,
        route = route,
        http_send_time_ms = http_send_time_ms,
        http_headers_recv_time_ms = http_headers_recv_time_ms,
        http_send_to_headers_ms = http_headers_recv_time_ms.saturating_sub(http_send_time_ms),
        status = %res.status(),
        is_stream = is_stream,
        "[ROUTER_HTTP_HEADERS_RECV] received worker response headers"
    );

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

        DispatchResult {
            response,
            delivered: true,
        }
    } else if load_incremented {
        let registry = Arc::clone(&config.worker_registry);
        let worker_url = worker_url.to_string();

        let mut response_headers = header_utils::preserve_response_headers(res.headers());
        response_headers.insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("text/event-stream"),
        );
        // Add anti-buffering headers to prevent nginx/proxy buffering and reduce latency
        response_headers.insert(
            axum::http::header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-cache, no-store, must-revalidate"),
        );
        response_headers.insert(
            axum::http::header::HeaderName::from_static("x-accel-buffering"),
            axum::http::HeaderValue::from_static("no"),
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
        DispatchResult {
            response,
            delivered: true,
        }
    } else {
        let mut response_headers = header_utils::preserve_response_headers(res.headers());
        response_headers.insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("text/event-stream"),
        );
        // Add anti-buffering headers to prevent nginx/proxy buffering and reduce latency
        response_headers.insert(
            axum::http::header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-cache, no-store, must-revalidate"),
        );
        response_headers.insert(
            axum::http::header::HeaderName::from_static("x-accel-buffering"),
            axum::http::HeaderValue::from_static("no"),
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
        DispatchResult {
            response,
            delivered: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::PolicyConfig;
    use crate::core::{BasicWorkerBuilder, PendingMessage};
    use crate::policies::PolicyRegistry;
    use reqwest::Client;
    use std::time::Duration;

    #[derive(Debug)]
    struct TransportFailScheduler;

    #[async_trait]
    impl SchedulerBase for TransportFailScheduler {
        async fn send_http_request(
            &self,
            _config: &Arc<SchedulerConfig>,
            _headers: Option<&HeaderMap>,
            _body_json: &serde_json::Value,
            _route: &str,
            _worker_url: &str,
            _is_stream: bool,
            _load_incremented: bool,
        ) -> DispatchResult {
            DispatchResult {
                response: (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Request failed: synthetic transport error",
                )
                    .into_response(),
                delivered: false,
            }
        }
    }

    #[tokio::test]
    async fn dispatch_request_clears_pending_message_on_transport_failure() {
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://worker:8080").build());
        let body = json!({
            "prompt": "hello",
            "request_id": "req-test"
        });
        worker.add_pending_message(PendingMessage::new(
            "/generate".to_string(),
            "req-test".to_string(),
            body.clone(),
            512,
        ));
        assert_eq!(worker.pending_message_count(), 1);

        let config = Arc::new(SchedulerConfig {
            worker_registry: Arc::new(WorkerRegistry::new()),
            policy_registry: Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            client: Client::builder().build().unwrap(),
            dp_aware: false,
            queue_timeout: Duration::from_secs(60),
        });

        let scheduler = Arc::new(TransportFailScheduler);
        let dispatch_result = dispatch_request(
            &scheduler,
            &config,
            None,
            &body,
            "/generate",
            None,
            false,
            worker.clone(),
            "req-test",
            0.0,
        )
        .await;

        assert_eq!(
            dispatch_result.response.status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(worker.pending_message_count(), 0);
    }

    #[test]
    fn prepare_request_payload_keeps_generate_body_pure() {
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://worker:8080").build());
        let body = json!({
            "prompt": "hello",
            "request_id": "req-test"
        });

        worker.add_pending_message(PendingMessage::new(
            "/generate".to_string(),
            "req-test".to_string(),
            body.clone(),
            512, // default token count for tests
        ));

        let payload = prepare_request_payload("/generate", &body, 0.0);
        let value = payload.as_ref();

        assert_eq!(
            value.get("request_id").and_then(|v| v.as_str()),
            Some("req-test")
        );
        assert_eq!(
            value.get("arrival_time_ms").and_then(|v| v.as_f64()),
            Some(0.0)
        );
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
        let payload = prepare_request_payload("/v1/chat/completions", &body, 0.0);
        let value = payload.as_ref();

        assert_eq!(
            value.get("request_id").and_then(|v| v.as_str()),
            Some("req-test")
        );
        assert!(value.get("arrival_time_ms").is_none());
        assert_eq!(worker.pending_message_count(), 0);
    }

    #[test]
    fn prepare_request_payload_logs_warning_for_non_object_body() {
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://worker:8080").build());
        let body = serde_json::Value::String("not-an-object".to_string());

        let payload = prepare_request_payload("/generate", &body, 0.0);
        let value = payload.as_ref();

        assert!(value.is_string());
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
