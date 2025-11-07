use crate::core::{Worker, WorkerRegistry};
use crate::metrics::RouterMetrics;
use crate::policies::PolicyRegistry;
use crate::routers::header_utils;
use crate::routers::http::router::Router;
use crate::ui::RouterUi;
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use reqwest::Client;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{interval, MissedTickBehavior};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, warn };

const SCHEDULER_TICK_INTERVAL_MS: u64 = 5;

/// Pending request metadata shared between the router entrypoint and scheduler loop.
pub(crate) struct PendingRequest {
    pub headers: Option<HeaderMap>,
    pub body_json: serde_json::Value,
    pub route: String,
    pub model_id: Option<String>,
    pub is_stream: bool,
    pub text: String,
    pub enqueue_started: Instant,
    pub response_tx: oneshot::Sender<Response>,
}

/// Configuration and shared state required by the scheduler loop.
pub(crate) struct SchedulerConfig {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub dp_aware: bool,
    pub queue_timeout: Duration,
}

/// Launch the background scheduler responsible for draining the pending queue.
pub(crate) fn spawn_scheduler(
    config: SchedulerConfig,
    mut pending_rx: mpsc::Receiver<PendingRequest>,
) -> tokio::task::JoinHandle<()> {
    let config = Arc::new(config);
    tokio::spawn(async move {
        let mut pending_queue: VecDeque<PendingRequest> = VecDeque::new();
        let mut interval = interval(Duration::from_millis(SCHEDULER_TICK_INTERVAL_MS));
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut receiver_closed = false;

        loop {
            tokio::select! {
                maybe_request = pending_rx.recv(), if !receiver_closed => {
                    match maybe_request {
                        Some(request) => pending_queue.push_back(request),
                        None => receiver_closed = true,
                    }
                }
                _ = interval.tick() => {
                    drain_pending_queue(&config, &mut pending_queue);
                    if receiver_closed && pending_queue.is_empty() {
                        break;
                    }
                }
            }
        }

        warn!("Router scheduler loop exiting - pending channel closed and queue drained");
    })
}

fn drain_pending_queue(
    config: &Arc<SchedulerConfig>,
    pending_queue: &mut VecDeque<PendingRequest>,
) {
    while let Some(pending) = pending_queue.pop_front() {
        RouterUi::dec_queue();

        if !config.queue_timeout.is_zero() {
            let waited = pending.enqueue_started.elapsed();
            if waited > config.queue_timeout {
                RouterMetrics::record_request_error(&pending.route, "queue_timeout");
                let _ = pending.response_tx.send(
                    (
                        StatusCode::REQUEST_TIMEOUT,
                        "Request timed out while waiting to be scheduled",
                    )
                        .into_response(),
                );
                continue;
            }
        }
        
        let cfg = Arc::clone(config);
        tokio::spawn(async move {
            process_pending(cfg, pending).await;
        });
    }
}

async fn process_pending(config: Arc<SchedulerConfig>, pending: PendingRequest) {
    let PendingRequest {
        headers,
        body_json,
        route,
        model_id,
        is_stream,
        text,
        enqueue_started: _,
        response_tx,
    } = pending;

    let start = Instant::now();

    let response = dispatch_request(
        &config,
        headers.as_ref(),
        &body_json,
        &route,
        model_id.as_deref(),
        &text,
        is_stream,
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

async fn dispatch_request(
    config: &Arc<SchedulerConfig>,
    headers: Option<&HeaderMap>,
    body_json: &serde_json::Value,
    route: &str,
    model_id: Option<&str>,
    text: &str,
    is_stream: bool,
) -> Response {
    let worker = match select_worker_for_model(
        &config.worker_registry,
        &config.policy_registry,
        model_id,
        Some(text),
    ) {
        Some(worker) => worker,
        None => {
            RouterMetrics::record_request_error(route, "no_available_workers");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "No available workers (all circuits open or unhealthy)",
            )
                .into_response();
        }
    };

    info!(
        "Selected worker for model: {} worker_url={}",
        model_id.unwrap_or("default"),
        worker.url()
    );

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

    let response = send_json_request(
        config,
        headers,
        body_json,
        route,
        worker.url(),
        is_stream,
        load_incremented,
    )
    .await;

    worker.record_outcome(response.status().is_success());

    response
}

fn select_worker_for_model(
    worker_registry: &Arc<WorkerRegistry>,
    policy_registry: &Arc<PolicyRegistry>,
    model_id: Option<&str>,
    text: Option<&str>,
) -> Option<Arc<dyn Worker>> {
    let workers = match model_id {
        Some(model) => worker_registry.get_by_model_fast(model),
        None => worker_registry.get_all(),
    };

    let available: Vec<Arc<dyn Worker>> = workers
        .iter()
        .filter(|w| w.is_available())
        .cloned()
        .collect();

    if available.is_empty() {
        return None;
    }

    let policy = match model_id {
        Some(model) => policy_registry.get_policy_or_default(model),
        None => policy_registry.get_default_policy(),
    };

    let idx = policy.select_worker(&available, text)?;
    Some(available[idx].clone())
}

async fn send_json_request(
    config: &Arc<SchedulerConfig>,
    headers: Option<&HeaderMap>,
    body_json: &serde_json::Value,
    route: &str,
    worker_url: &str,
    is_stream: bool,
    load_incremented: bool,
) -> Response {
    let mut request_builder = if config.dp_aware {
        let (worker_url_prefix, dp_rank) = match Router::extract_dp_rank(worker_url) {
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
