use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Individual waiting queue request information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaitingQueueRequest {
    pub id: String,
    pub prefix_len: i64,
    pub extend_len: i64,
}

/// Waiting queue information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaitingQueueInfo {
    pub pending_req_num: i64,
    pub total_extend_len: i64,
    pub requests: Vec<WaitingQueueRequest>,
}

/// Statistics reported by workers via /worker_stats endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStats {
    /// Worker identifier (URL)
    pub worker_id: String,

    /// Current batch size in tokens
    pub batch_size_tokens: i64,

    /// Number of KV cache tokens currently used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_tokens_used: Option<i64>,

    /// Number of requests currently being processed
    pub num_requests: i64,

    /// Size of the waiting queue
    pub waiting_queue_size: i64,

    /// Detailed waiting queue information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waiting_queue_info: Option<WaitingQueueInfo>,

    /// Current forward mode (e.g., "PREFILL", "DECODE", "UNKNOWN")
    pub forward_mode: String,

    /// Iteration counter
    pub iteration_num: i64,

    /// Duration of the last completed iteration in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_iteration_time_ms: Option<f64>,

    /// Prefill chunk pairs: [(chunk_size, cumulative_prefill), ...]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_chunk_pairs: Option<Vec<(i64, i64)>>,

    /// Prefill simulation metrics: maps extra token count to estimated execution time in ms
    /// Python sends this as "prefill_sim_results" with integer keys (0, 128, 256, ..., 4096)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_sim_metrics: Option<HashMap<i64, f64>>,

    /// Router generation reported by the worker (for message acknowledgments)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_generation: Option<i64>,

    /// Highest contiguous message ID received for the reported generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_received_message_id: Option<i64>,

    /// Batch size breakdown by TPOT tier (e.g., {"40": 5, "80": 3, "none": 2})
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size_by_tpot_tier: Option<HashMap<String, i64>>,

    /// Timestamp when stats were received (not serialized)
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

impl WorkerStats {
    /// Calculate total load as num_requests + waiting_queue_size
    pub fn total_load(&self) -> i64 {
        self.num_requests + self.waiting_queue_size
    }

    /// Create WorkerStats from JSON value
    pub fn from_json(stats: &serde_json::Value) -> Result<Self, String> {
        let worker_id = stats
            .get("worker_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing worker_id")?
            .to_string();

        let batch_size_tokens = stats
            .get("batch_size_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        let num_requests = stats
            .get("num_requests")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        let waiting_queue_size = stats
            .get("waiting_queue_size")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        let waiting_queue_info = stats
            .get("waiting_queue_info")
            .and_then(|v| v.as_object())
            .and_then(|obj| {
                let pending_req_num = obj.get("pending_req_num")?.as_i64()?;
                let total_extend_len = obj.get("total_extend_len")?.as_i64()?;
                let requests = obj
                    .get("requests")?
                    .as_array()?
                    .iter()
                    .filter_map(|req| {
                        let req_obj = req.as_object()?;
                        Some(WaitingQueueRequest {
                            id: req_obj.get("id")?.as_str()?.to_string(),
                            prefix_len: req_obj.get("prefix_len")?.as_i64()?,
                            extend_len: req_obj.get("extend_len")?.as_i64()?,
                        })
                    })
                    .collect();

                Some(WaitingQueueInfo {
                    pending_req_num,
                    total_extend_len,
                    requests,
                })
            });

        let forward_mode = stats
            .get("forward_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();

        let iteration_num = stats
            .get("iteration_num")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        let last_iteration_time_ms = stats.get("last_iteration_time_ms").and_then(|v| v.as_f64());

        let prefill_chunk_pairs = stats
            .get("prefill_chunk_pairs")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|pair| {
                        pair.as_array().and_then(|p| {
                            if p.len() == 2 {
                                Some((p[0].as_i64()?, p[1].as_i64()?))
                            } else {
                                None
                            }
                        })
                    })
                    .collect()
            });

        let prefill_sim_metrics = stats
            .get("prefill_sim_results")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| {
                        // Parse key as i64 (Python sends integer keys as strings in JSON)
                        let key = k.parse::<i64>().ok()?;
                        // Get value as f64 (skip null values)
                        let value = v.as_f64()?;
                        Some((key, value))
                    })
                    .collect()
            });

        let kv_tokens_used = stats.get("kv_tokens_used").and_then(|v| v.as_i64());

        let router_generation = stats.get("router_generation").and_then(|v| v.as_i64());
        let last_received_message_id = stats
            .get("last_received_message_id")
            .and_then(|v| v.as_i64());

        // Parse batch_size_by_tpot_tier: {"40": 5, "80": 3, "none": 2}
        // Keys are TPOT values as strings (or "none"), values are request counts
        let batch_size_by_tpot_tier = stats
            .get("batch_size_by_tpot_tier")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| {
                        let value = v.as_i64()?;
                        Some((k.clone(), value))
                    })
                    .collect()
            });

        Ok(WorkerStats {
            worker_id,
            batch_size_tokens,
            kv_tokens_used,
            num_requests,
            waiting_queue_size,
            waiting_queue_info,
            forward_mode,
            iteration_num,
            last_iteration_time_ms,
            prefill_chunk_pairs,
            prefill_sim_metrics,
            router_generation,
            last_received_message_id,
            batch_size_by_tpot_tier,
            timestamp: Instant::now(),
        })
    }
}
