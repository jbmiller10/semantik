# GPU Memory Management

Semantik uses a sophisticated GPU memory management system to efficiently handle ML models on GPUs with varying amounts of VRAM. The system includes intelligent eviction, CPU offloading for "warm" models, and background pressure monitoring.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GPUMemoryGovernor                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Memory Budget Tracking                        │   │
│  │  - GPU usable = total * max_percent                             │   │
│  │  - CPU warm pool capacity                                        │   │
│  │  - Current allocations per model                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐   │
│  │   LRU Tracking    │  │ Pressure Monitor  │  │  Eviction Logic   │   │
│  │  (OrderedDict)    │  │ (Background Task) │  │ (Offload/Unload)  │   │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘   │
└───────────────────────────┬────────────────────────────────────────────┘
                            │ Callbacks
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    GovernedModelManager                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Extends ModelManager with:                                      │   │
│  │  - Memory request before load                                    │   │
│  │  - Governor tracking on load/unload                              │   │
│  │  - Callback handlers for offload/restore                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       CPUOffloader                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - Offload model weights to pinned CPU memory                    │   │
│  │  - Fast restore to GPU (2-5s vs 10-30s disk reload)              │   │
│  │  - Metadata tracking (offload time, memory usage)                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### GPUMemoryGovernor

The central coordinator for GPU memory allocation. It:

- Tracks all loaded models with LRU ordering
- Enforces memory budgets
- Triggers eviction when memory is needed
- Monitors memory pressure and responds preemptively

### MemoryBudget

Configurable memory limits:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_max_percent` | 0.90 | Maximum GPU memory the application can use |
| `cpu_max_percent` | 0.50 | Maximum CPU memory for warm models |

**Usable memory calculation:**
```
usable_gpu = total_gpu * gpu_max_percent
usable_cpu = total_cpu * cpu_max_percent
```

### Pressure Levels

The governor continuously monitors GPU memory usage and responds based on pressure level:

| Level | Threshold | Action |
|-------|-----------|--------|
| LOW | <60% | No action |
| MODERATE | 60-80% | Preemptively offload models idle >120s |
| HIGH | 80-90% | Aggressively offload models idle >30s |
| CRITICAL | >90% | Force unload all idle models |

## Eviction Strategy

When a new model needs to load but there's insufficient memory:

1. **Offload LRU models to CPU** (if CPU offloading enabled and room available)
   - Preserves "warm" state for fast restoration
   - Ordered by last-used time (oldest first)

2. **Fully unload LRU models** (if CPU offload not possible)
   - Removes model from memory entirely
   - Requires full reload from disk on next use

3. **Grace period protection**
   - Models used within last 5 seconds are never evicted
   - Prevents evicting a model currently serving a request

4. **Fail gracefully**
   - Only fails if requested model won't fit even with empty GPU
   - Returns clear error with memory statistics

## CPU Offloading (Warm Pool)

When enabled, models are moved to CPU RAM instead of being fully unloaded:

**Benefits:**
- Restoration time: 2-5 seconds (vs 10-30s disk reload)
- Model stays in process memory
- Useful for GPUs with less VRAM

**How it works:**
1. Model weights are moved to pinned CPU memory
2. GPU memory is freed
3. On restore, weights are transferred back via fast PCIe path
4. Uses pinned memory for optimal transfer speed

## Configuration

### Environment Variables

```bash
# Enable/disable the memory governor (default: true)
ENABLE_MEMORY_GOVERNOR=true

# Memory Limits
GPU_MEMORY_MAX_PERCENT=0.90        # Maximum GPU memory usage (default: 90%)
CPU_MEMORY_MAX_PERCENT=0.50        # Maximum CPU memory for warm models (default: 50%)

# CPU Offloading
ENABLE_CPU_OFFLOAD=true            # Enable warm model pool (default: true)

# Eviction Behavior
EVICTION_IDLE_THRESHOLD_SECONDS=120    # Idle time before eligible (default: 120s)
PRESSURE_CHECK_INTERVAL_SECONDS=15     # Monitor interval (default: 15s)
```

### Recommended Configurations by VRAM Size

#### 24GB+ VRAM (RTX 3090, RTX 4090, A100)
```bash
# Default settings work well
# Both embedding + reranker can coexist
GPU_MEMORY_MAX_PERCENT=0.90
```

#### 16GB VRAM (RTX 4080, A4000)
```bash
# Default settings, may need occasional eviction
GPU_MEMORY_MAX_PERCENT=0.90
ENABLE_CPU_OFFLOAD=true
```

#### 8GB VRAM (RTX 3060, RTX 4060)
```bash
# More aggressive eviction
GPU_MEMORY_MAX_PERCENT=0.85
ENABLE_CPU_OFFLOAD=true
EVICTION_IDLE_THRESHOLD_SECONDS=60
```

#### 4GB VRAM
```bash
# Very constrained - may need single model at a time
GPU_MEMORY_MAX_PERCENT=0.80
ENABLE_CPU_OFFLOAD=true
EVICTION_IDLE_THRESHOLD_SECONDS=30
```

## API Endpoints

### GET /memory/stats

Returns comprehensive memory statistics:

```json
{
  "cuda_available": true,
  "total_mb": 24576,
  "free_mb": 18432,
  "used_mb": 6144,
  "used_percent": 25.0,
  "allocated_mb": 3200,
  "budget_total_mb": 24576,
  "budget_usable_mb": 22118,
  "cpu_budget_total_mb": 65536,
  "cpu_budget_usable_mb": 32768,
  "cpu_used_mb": 1600,
  "models_loaded": 2,
  "models_offloaded": 1,
  "pressure_level": "LOW",
  "total_evictions": 15,
  "total_offloads": 12,
  "total_restorations": 8,
  "total_unloads": 3
}
```

### GET /memory/models

Returns list of tracked models:

```json
[
  {
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "model_type": "embedding",
    "quantization": "float16",
    "location": "gpu",
    "memory_mb": 1200,
    "idle_seconds": 45.2,
    "use_count": 127
  },
  {
    "model_name": "Qwen/Qwen3-Reranker-0.6B",
    "model_type": "reranker",
    "quantization": "float16",
    "location": "cpu",
    "memory_mb": 1200,
    "idle_seconds": 320.5,
    "use_count": 23
  }
]
```

### GET /memory/evictions

Returns recent eviction history:

```json
[
  {
    "model_name": "Qwen/Qwen3-Reranker-0.6B",
    "model_type": "reranker",
    "quantization": "float16",
    "reason": "memory_pressure",
    "action": "offloaded",
    "memory_freed_mb": 1200,
    "timestamp": 1704067200.123
  }
]
```

### GET /memory/fragmentation

Returns CUDA memory fragmentation analysis. High fragmentation can cause OOM even when
total free memory seems sufficient.

```json
{
  "cuda_available": true,
  "allocated_mb": 6144,
  "reserved_mb": 8192,
  "fragmentation_mb": 2048,
  "fragmentation_percent": 25.0,
  "num_alloc_retries": 3,
  "num_ooms": 0
}
```

### GET /memory/offloaded

Returns list of models currently offloaded to CPU RAM (warm pool):

```json
[
  {
    "model_key": "reranker:Qwen/Qwen3-Reranker-0.6B:float16",
    "original_device": "cuda:0",
    "offload_time": 1704067200.123,
    "seconds_offloaded": 45.2
  }
]
```

### GET /memory/health

Quick health check based on current memory pressure:

```json
{
  "healthy": true,
  "pressure": "LOW",
  "used_percent": 25.0,
  "message": "Memory usage normal"
}
```

Returns `healthy: false` when pressure level is CRITICAL.

### POST /memory/defragment

Triggers CUDA memory cache cleanup. Useful when experiencing fragmentation issues:

```json
{
  "status": "defragmentation_triggered"
}
```

### POST /memory/evict/{model_type}

Manually evict a model to free GPU memory. `model_type` must be "embedding" or "reranker":

```bash
curl -X POST http://localhost:8001/memory/evict/reranker
```

```json
{
  "status": "evicted",
  "model_type": "reranker"
}
```

### POST /memory/preload

Preload models for expected requests. Useful for warming up before peak traffic:

```json
{
  "models": [
    {
      "name": "Qwen/Qwen3-Embedding-0.6B",
      "model_type": "embedding",
      "quantization": "float16"
    }
  ]
}
```

Response:

```json
{
  "results": {
    "embedding:Qwen/Qwen3-Embedding-0.6B:float16": true
  }
}
```

## Troubleshooting

### OOM Errors Still Occurring

1. **Check pressure level**: If consistently HIGH/CRITICAL, reduce `GPU_MEMORY_MAX_PERCENT`
2. **Lower memory limit**: Try `GPU_MEMORY_MAX_PERCENT=0.85` or `0.80`
3. **Reduce batch sizes**: Large batches consume activation memory beyond weights
4. **Check for memory leaks**: Monitor `/memory/stats` over time

### Slow Model Switching

1. **Enable CPU offloading**: `ENABLE_CPU_OFFLOAD=true`
2. **Increase CPU warm pool**: Raise `CPU_MEMORY_MAX_PERCENT` if RAM available
3. **Check restore times**: Monitor logs for "Restored X from CPU to GPU" messages

### High Memory Pressure Warnings

1. **Normal under load**: MODERATE pressure is expected during heavy use
2. **Persistent HIGH**: Consider upgrading GPU or using smaller models
3. **Reduce idle threshold**: `EVICTION_IDLE_THRESHOLD_SECONDS=60` for faster eviction

### Models Evicted Too Aggressively

1. **Increase idle threshold**: `EVICTION_IDLE_THRESHOLD_SECONDS=180` or higher
2. **Check pressure levels**: LOW pressure shouldn't trigger eviction
3. **Review model sizes**: Larger models may need more headroom

## Monitoring

### Logs to Watch

```
INFO - GovernedModelManager initialized with memory governor (gpu_budget=22118MB)
INFO - Memory request approved: embedding:Qwen/Qwen3-Embedding-0.6B:float16 needs 1440MB
INFO - Offloaded reranker:Qwen/Qwen3-Reranker-0.6B to CPU (freed 1200MB GPU)
INFO - Restored embedding:Qwen/Qwen3-Embedding-0.6B from CPU to GPU
WARNING - HIGH memory pressure: 85.2%. Offloading idle models.
ERROR - CRITICAL memory pressure! Usage: 94.1%. Forcing aggressive eviction.
```

### Memory Monitoring

Memory statistics are available via REST API endpoints (see API Reference below).
Use the `/memory/stats` endpoint to get comprehensive memory metrics including:

- GPU/CPU memory usage
- Models loaded and offloaded counts
- Pressure level
- Eviction/offload/restoration statistics

For production monitoring, you can poll the `/memory/health` endpoint which returns
a quick health status based on current memory pressure.
