# Streaming Issue Client

## Overview

`issue_stream.py` is a simplified streaming client for SLO testing. Unlike `issue.py`, it:

1. **Spawns one thread per request** - No thread pool, no async result processing
2. **Streams responses** - Uses SSE streaming endpoint to receive tokens incrementally
3. **Tracks detailed token timing**:
   - Start time (when request submitted)
   - Time to First Token (TTFT)
   - Inter-token intervals (1→2, 2→3, ..., 9→10)
   - Average inter-token interval

## Key Differences from issue.py

| Feature | issue.py | issue_stream.py |
|---------|----------|-----------------|
| Request type | Non-streaming | Streaming (SSE) |
| Threading | ThreadPoolExecutor | Direct thread spawning |
| Result processing | Async background thread | Inline per request |
| Timing metrics | End-to-end only | Token-level timing |
| Complexity | High (queue management, async) | Low (simple threading) |

## Usage

```bash
python -m slo.issue_stream \
  --trace /path/to/trace.csv \
  --text-file /path/to/corpus.txt \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://0.0.0.0:40000/v1 \
  --rate 1.0 \
  --max-requests 10
```

Or use the launch script:
```bash
./slo/launch_issue_stream.sh
```

## Log Format

Each request logs a JSON object with:

```json
{
  "request_id": "req_000001_123456",
  "submit_timestamp": 1234567890.123,
  "total_duration_ms": 1234.5,
  "status": "SUCCESS",
  "token_count": 100,
  "ttft_ms": 45.2,
  "intervals": [12.3, 11.8, 12.1, 11.9, 12.0, 11.7, 12.2, 11.8, 12.0],
  "avg_interval_ms": 11.98,
  "decode": 100,
  "target_ttft_ms": 50.0,
  "target_tpot_ms": 12.0
}
```

## Timing Metrics Explained

- **ttft_ms**: Time from request start to first token received
- **intervals**: List of inter-token delays in milliseconds
  - intervals[0] = time from token 1 to token 2
  - intervals[1] = time from token 2 to token 3
  - ...
  - intervals[8] = time from token 9 to token 10
- **avg_interval_ms**: Average of all intervals (for tokens beyond the first)

## Live UI

Similar to `issue.py`, displays:
- Total requests
- Active threads
- Completed/Failed counts
- Submit rate
- Progress percentage
- Log file location

Disable with `--no-ui` flag.
