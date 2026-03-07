# RouterModelCustom - AI Agent Documentation

> **Project:** RouterModelCustom  
> **Description:** A robust model router and manager for Local LLM inference  
> **Language:** English (code comments mixed with Indonesian)  
> **Last Updated:** 2026-03-08

---

## ⚠️ Important Notice

**`API_DOCUMENTATION.md` is OUTDATED.**  
Do not rely on it for API reference. Use this `AGENTS.md` file and the actual source code (`app/main.py`, `run.py`) as the single source of truth for API endpoints and behavior.

---

## Project Overview

RouterModelCustom is middleware between client applications and `llama-server` instances, providing **dynamic model loading**, **resource management**, **request queuing**, and an **OpenAI-compatible API**. It acts as a gateway/router that manages multiple LLM models running on llama.cpp servers.

### Key Features

- **Dynamic Model Management:** Auto-load/unload models based on demand and VRAM availability
- **Priority Queue System:** HIGH/NORMAL/LOW priority with heap-based scheduling
- **OpenAI Compatible API:** Drop-in replacement for `/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`
- **Prometheus + Grafana:** Built-in monitoring with pre-configured dashboards
- **Model Preloading:** Warmup models on startup with `preload_models: ["*"]`
- **VRAM Management:** Real-time tracking and guards to prevent OOM
- **Health Monitoring:** Auto-restart crashed models, health checks every 30s
- **Idempotency Support:** Request deduplication with `X-Idempotency-Key` header
- **Streaming Support:** Server-Sent Events for both inference and status updates
- **Status Server:** Lightweight aiohttp server on port 80 for early health checks

---

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **Status Server** | aiohttp (runs on separate thread, port 80) |
| **LLM Inference** | llama.cpp (llama-server binary) |
| **GPU Monitoring** | pynvml (NVIDIA), psutil |
| **HTTP Client** | httpx (async, HTTP/1.1 for local connections) |
| **Queue** | heapq (priority heap), asyncio |
| **Monitoring** | Prometheus, Grafana, prometheus-client |
| **Validation** | Pydantic v2 |
| **Logging** | Python logging + structured JSON logs |

---

## Project Structure

```
.
├── app/                          # Main application code
│   ├── __init__.py
│   ├── check_validate_config.py  # Config validation CLI tool
│   ├── main.py                   # FastAPI application entry point (~2200 lines)
│   ├── core/                     # Core business logic
│   │   ├── config.py            # Configuration loading & validation (Pydantic models)
│   │   ├── manager.py           # RunnerProcess & ModelManager (llama-server lifecycle)
│   │   ├── queue.py             # Priority queue system with heapq
│   │   ├── warmup.py            # Model preloading and keep-warm strategy
│   │   ├── vram_tracker.py      # GPU VRAM monitoring and tracking
│   │   ├── health_monitor.py    # Health checks and auto-restart
│   │   ├── prometheus_metrics.py # Prometheus metrics collection
│   │   ├── model_status.py      # Model status tracking with file persistence
│   │   ├── telemetry.py         # Request telemetry collection
│   │   ├── metrics.py           # Basic metrics storage
│   │   ├── errors.py            # Custom exceptions (InsufficientVRAMError)
│   │   ├── gguf_utils.py        # GGUF file metadata parsing
│   │   ├── limit_request.py     # Request size limiting middleware
│   │   └── logging_server.py    # Structured logging setup
│   └── http/                     # HTTP layer
│       ├── __init__.py
│       └── schema.py            # Pydantic request schemas (RerankRequest)
├── monitoring/                   # Docker monitoring stack
│   ├── prometheus.yml           # Prometheus scrape configuration
│   └── grafana/
│       └── provisioning/        # Auto-configured dashboards and datasources
├── logs/                        # Runtime logs (gitignored)
│   ├── runners/                 # Per-model llama-server logs
│   └── model_status.json        # Persistent model status file
├── run.py                       # Server entry point with status server thread
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Multi-stage build with CUDA support
├── docker-compose.monitoring.yml # Prometheus + Grafana stack
├── .example.config.json         # Example configuration template
├── config.json                  # Active configuration (gitignored)
├── API_DOCUMENTATION.md         # ⚠️ OUTDATED - DO NOT USE
├── README.md                    # User-facing documentation
└── AGENTS.md                    # This file - accurate technical reference
```

---

## Configuration System

The project uses a JSON configuration file validated by Pydantic. Config path resolution:
1. CLI argument: `--config` or `-c`
2. Environment variable: `CONFIG_PATH`
3. Default: `config.json`

### Key Configuration Sections

```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "cors_origins": ["http://localhost:3000"]
  },
  "system": {
    "llama_server_path": "/path/to/llama-server",
    "base_models_path": "/path/to/models",
    "max_concurrent_models": 3,
    "preload_models": ["*"],
    "enable_idle_timeout": true,
    "idle_timeout_sec": 300,
    "gpu_devices": [0],
    "flash_attention": "on",
    "queue_timeout_sec": 300,
    "queue_processor_idle_sec": 60,
    "http_max_keepalive": 10,
    "http_max_connections": 100,
    "request_timeout_sec": 300
  },
  "models": {
    "model-alias": {
      "model_path": "relative-or-absolute.gguf",
      "params": {
        "n_gpu_layers": 99,
        "n_ctx": 4096,
        "embedding": false,
        "reranker": false,
        "parallel_requests": 2,
        "type_k": "f16",
        "type_v": "f16"
      }
    }
  }
}
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CONFIG_PATH` | Path to config JSON file |
| `LLAMA_SERVER_PATH` | Override llama-server binary path |
| `BASE_MODELS_PATH` | Override base models directory |
| `STATUS_SERVER_HOST` | Status server bind host (default: 0.0.0.0) |
| `STATUS_SERVER_PORT` | Status server port (default: 80) |
| `DEBUG` | Enable debug mode (true/false) |
| `STRUCTURED_LOGS` | Enable structured JSON logging |

---

## Running the Application

### Prerequisites

- Python 3.10+
- NVIDIA GPU (optional but recommended)
- `llama-server` binary from llama.cpp
- GGUF model files

### Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# OR: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Validate configuration
python -c "from app.check_validate_config import validate_config_file; validate_config_file('config.json')"

# Run the server
python run.py
# OR with custom config:
python run.py --config custom_config.json
```

### Docker Deployment

```bash
# Build image
docker build -t router-model-custom .

# Run with mounted config and models
docker run -d \
  -p 8000:8000 \
  -p 80:80 \
  -v $(pwd)/config.json:/app/config.json:ro \
  -v /path/to/models:/models:ro \
  -e CONFIG_PATH=/app/config.json \
  router-model-custom
```

### Monitoring Stack (Prometheus + Grafana)

```bash
# Start monitoring
docker compose -f docker-compose.monitoring.yml up -d

# Access:
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

---

## Core Architecture Components

### 1. ModelManager (`app/core/manager.py`)

Manages llama-server process lifecycle:
- **RunnerProcess:** Wrapper for subprocess llama-server
- Port allocation from pool (8085-8584)
- VRAM estimation before loading
- Sequential loading with load_lock
- Idle timeout watchdog
- Retry logic for failed loads (max 3 attempts)
- Failed models tracking (prevents infinite restart loops)

### 2. VRAMTracker (`app/core/vram_tracker.py`)

GPU memory management:
- Real-time VRAM monitoring via NVML
- Per-model VRAM usage tracking with snapshots
- Sequential loading coordination (load_lock)
- VRAM estimation: `file_size * multiplier + KV cache + overhead`
- Safety buffer (default 200MB) before rejecting loads
- Load duration tracking

### 3. QueueManager (`app/core/queue.py`)

Request queuing system:
- Priority-based: HIGH (1), NORMAL (2), LOW (3)
- Heap-based priority queue: O(log n) operations
- Backpressure: max queue size per model (configurable)
- Timeout handling for queued requests
- Idle processor shutdown (configurable timeout)
- Per-model queue statistics

### 4. ModelWarmupManager (`app/core/warmup.py`)

Preloading and keep-warm strategy:
- Preload models on startup (`preload_models: ["*"]`)
- Keep popular models warm (prevent idle timeout)
- Popularity tracking based on request count
- VRAM-aware loading (skip if insufficient)
- Configurable warmup delay and max models

### 5. HealthMonitor (`app/core/health_monitor.py`)

Health checking and auto-recovery:
- Periodic health checks every 30s
- Response time tracking per model
- Consecutive failure counting
- Auto-restart for crashed models (5+ failures triggers permanent failure)
- HTTP health endpoint for each model

### 6. PrometheusMetricsCollector (`app/core/prometheus_metrics.py`)

Metrics collection and export:
- Request counters, latency histograms
- GPU VRAM gauges
- Queue depth tracking
- Token throughput metrics
- SLO tracking (99.5% availability target)
- 5-minute aggregated reports
- Real-time metrics snapshots

### 7. ModelStatusTracker (`app/core/model_status.py`)

Persistent model status tracking:
- File-based status persistence (`logs/model_status.json`)
- Real-time status updates via SSE
- Status types: OFF, STARTING, LOADING, READY, STOPPING, CRASHED, FAILED, UNKNOWN
- Pub/sub pattern for status changes
- Server status tracking (initializing, ready, shutting_down)

### 8. Idempotency System

Request deduplication:
- `X-Idempotency-Key` header support
- In-flight request tracking per endpoint type
- Completed response caching (30 min TTL)
- Separate locks for chat, embeddings, and rerank
- Automatic cleanup of stale entries

---

## API Endpoints (FastAPI - Port 8000)

### Health & System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Serverless ping check - simple alive check |
| `/health` | GET | Detailed health check with GPU, manager, HTTP client status |

### OpenAI Compatible

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models with n_ctx info |
| `/v1/chat/completions` | POST | Chat completion (streaming supported). Headers: `X-Request-Priority`, `X-Idempotency-Key` |
| `/v1/embeddings` | POST | Generate embeddings (model must have `embedding: true`) |
| `/v1/rerank` | POST | Rerank documents (model must have `reranker: true`). Applies sigmoid activation to scores |

### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models/eject` | POST | Unload model from VRAM (body: `{"model": "alias"}`) |
| `/v1/models/{alias}/reset` | POST | Reset failed model status to allow retry |
| `/v1/models/{alias}/status` | GET | Get loading status for specific model |
| `/v1/models/failed` | GET | List models that failed to start |
| `/v1/models/status` | GET | Get all model statuses with summary |
| `/v1/models/status/stream` | GET | SSE stream for real-time status updates |

### Queue

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/queue/stats` | GET | Queue statistics per model with summary |

### VRAM

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vram` | GET | Full VRAM report with per-model usage |
| `/vram/models/{alias}` | GET | Detailed VRAM info for specific model (snapshots, load history) |
| `/vram/summary` | GET | Quick VRAM summary for dashboards |

### Metrics & Telemetry

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/metrics` | GET | Prometheus exposition format |
| `/metrics/legacy` | GET | Legacy metrics format (old) |
| `/metrics/stream` | GET | SSE stream for real-time metrics |
| `/metrics/report` | GET | 5-minute aggregated metrics report |
| `/v1/telemetry/summary` | GET | Request telemetry summary |
| `/v1/health/models` | GET | Health status for all active models |

---

## Status Server Endpoints (aiohttp - Port 80)

Runs on a separate thread, available immediately on startup (before FastAPI is ready).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Get current model status from file |
| `/status/stream` | GET | SSE stream for status updates (file polling based) |
| `/ping` | GET | Health check for load balancers (checks FastAPI port 8000) |

**SSE Event Types:**
- `full_status`: Complete status on connect
- `server_update`: Server status changes
- `model_update`: Single model status change
- `models_update`: Multiple models changed
- `model_removed`: Model removed from tracking
- `heartbeat`: Keep-alive every 30 seconds

---

## Request Flow

### Chat Completions Flow

```
1. Client POST /v1/chat/completions
   Headers: X-Idempotency-Key (optional), X-Request-Priority (optional)

2. _proxy_request_with_queue()
   - Parse request body
   - Determine priority (high/normal/low)
   - Record request for warmup tracking

3. _process_request_via_queue()
   - Check idempotency cache
   - Enqueue request (heapq)
   - Wait for queue processor

4. _queue_processor() (background task)
   - Dequeue by priority
   - Get runner via manager.get_runner_for_request()
   - Process request with retry logic

5. _process_queued_request()
   - Send to llama-server
   - Handle streaming or JSON response
   - Detect context shift warnings
   - Return result

6. Response to client
   - Streaming: SSE format
   - Non-streaming: JSON with metadata
```

### Idempotency Flow

```
1. Extract X-Idempotency-Key header (or generate UUID)
2. Check completed cache -> return cached response
3. Check in-flight requests -> wait for completion
4. Create asyncio.Event for new request
5. Process request
6. Store result in completed cache
7. Signal waiting requests
8. Schedule cleanup after 30 minutes
```

---

## Code Style Guidelines

### Language and Comments

- **Code:** Python 3.10+ with type hints
- **Comments:** Mixed English and Indonesian (existing convention)
- **Docstrings:** Google-style or descriptive blocks

### Naming Conventions

- **Classes:** PascalCase (`ModelManager`, `VRAMTracker`)
- **Functions/Methods:** snake_case (`get_runner_for_request`)
- **Variables:** snake_case (`model_alias`, `vram_used_mb`)
- **Constants:** UPPER_SNAKE_CASE (`MAX_QUEUE_SIZE`)
- **Private:** Leading underscore (`_init_nvml`, `_load_single_model`)

### Async Patterns

- Use `async`/`await` for I/O operations
- Use `asyncio.Lock()` for thread-safe state
- Background tasks stored in `background_tasks: Set[asyncio.Task]`
- Graceful shutdown with task cancellation
- Use `assign_to_background()` helper for background tasks

### Error Handling

```python
# Custom exceptions from app.core.errors
class InsufficientVRAMError(Exception):
    def __init__(self, model_alias, required_mb, available_mb, loaded_models):
        ...

# Retry logic for transient errors
max_retries = 2
for attempt in range(max_retries + 1):
    try:
        result = await operation()
        break
    except (httpx.ReadTimeout, httpx.ConnectError) as e:
        if attempt < max_retries:
            await asyncio.sleep(retry_delay)
            continue
        raise
```

### Logging

```python
import logging
logger = logging.getLogger(__name__)

# Use structured logging
logger.info(f"[{alias}] Model loaded successfully")
logger.warning(f"[{alias}] VRAM running low: {free_mb} MB remaining")
logger.error(f"[{alias}] Failed to start: {error}")
```

---

## Testing Strategy

The project currently does not have automated tests. For manual testing:

```bash
# 1. Validate config
python -c "from app.check_validate_config import validate_config_file; validate_config_file('config.json')"

# 2. Test chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "your-model", "messages": [{"role": "user", "content": "Hello"}]}'

# 3. Test streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "your-model", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'

# 4. Test embeddings
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "embedding-model", "input": "test text"}'

# 5. Test rerank
curl http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model": "reranker-model", "query": "test", "documents": ["doc1", "doc2"], "top_n": 2}'

# 6. Check status server
curl http://localhost:80/status
curl http://localhost:80/ping

# 7. Check metrics
curl http://localhost:8000/metrics
curl http://localhost:8000/vram

# 8. Check queue stats
curl http://localhost:8000/v1/queue/stats
```

---

## Security Considerations

1. **No Built-in Authentication:** The API does not require authentication. For production:
   - Place behind reverse proxy (nginx, Traefik)
   - Add authentication middleware
   - Use API keys or OAuth

2. **CORS Configuration:** Configure `cors_origins` in config.json to restrict origins

3. **Model Path Validation:** Config validates model paths exist and are `.gguf` files

4. **Request Size Limits:** `RequestSizeLimitMiddleware` prevents oversized requests

5. **VRAM Guards:** Prevents OOM by checking available VRAM before loading models

6. **Idempotency Keys:** Client-provided keys should be unique per request

---

## Common Development Tasks

### Adding a New Model

1. Add to `config.json`:
```json
"new-model": {
  "model_path": "model.gguf",
  "params": {
    "n_gpu_layers": 99,
    "n_ctx": 4096,
    "embedding": false,
    "reranker": false
  }
}
```

2. Validate: `python run.py --config config.json` (will validate on startup)

3. Test: `curl http://localhost:8000/v1/models`

### Adding a New Endpoint

1. Add route in `app/main.py`:
```python
@app.get("/new-endpoint")
async def new_endpoint():
    return {"status": "ok"}
```

2. Add to middleware skip lists if needed (telemetry, metrics)

### Modifying Queue Behavior

- Queue logic is in `app/core/queue.py`
- Queue processor is in `app/main.py` (`_queue_processor`)
- Queue integration is in `_process_request_via_queue`
- Config options: `queue_timeout_sec`, `queue_processor_idle_sec`

### Adding New Metrics

1. Define metric in `app/core/prometheus_metrics.py`:
```python
NEW_METRIC = Counter(
    'router_new_metric_total',
    'Description',
    ['label1', 'label2']
)
```

2. Record in `PrometheusMetricsCollector` methods

3. Export in `get_prometheus_metrics()`

---

## Troubleshooting

### Model Fails to Load

1. Check runner logs: `logs/runners/{alias}_{port}.log`
2. Verify llama-server path in config
3. Check VRAM availability: `curl http://localhost:8000/vram`
4. Verify model file exists and is valid GGUF
5. Check failed models: `curl http://localhost:8000/v1/models/failed`

### High VRAM Usage

1. Reduce `n_ctx` (context window)
2. Reduce `n_gpu_layers` (offload fewer layers to GPU)
3. Enable KV cache quantization: `type_k: "q4_0"`, `type_v: "q4_0"`
4. Reduce `parallel_requests` per model
5. Enable idle timeout to auto-unload unused models

### Queue Timeouts

1. Increase `queue_timeout_sec` in config
2. Increase `max_queue_size_per_model`
3. Enable more parallel requests (if VRAM permits)
4. Check if model is stuck in loading state

### Port Conflicts

- Port pool: 8085-8584 for llama-server instances
- API port: Configurable via `api.port` (default 8000)
- Status server: Configurable via `STATUS_SERVER_PORT` (default 80)

### Status Server Not Responding

1. Check if port 80 is available (may need root/admin on some systems)
2. Verify aiohttp is installed: `pip install aiohttp`
3. Check logs for "Status server running on..."
4. Status server runs in separate thread - check for thread startup errors

---

## Additional Resources

- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Prometheus Querying](https://prometheus.io/docs/prometheus/latest/querying/basics/)

---

## Changelog

### 2026-03-08
- Updated AGENTS.md with accurate endpoint documentation
- Added warning about outdated API_DOCUMENTATION.md
- Documented status server endpoints
- Added idempotency system documentation
- Documented rerank endpoint

---

*This documentation is intended for AI coding agents. For user-facing documentation, see README.md (but verify against this file for accuracy).*
