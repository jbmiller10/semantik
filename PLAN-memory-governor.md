# Plan: Implement GPUMemoryGovernor for Dynamic Memory Management

## Summary
Implement the missing `GPUMemoryGovernor` class to enable sophisticated, dynamic GPU memory management with CPU offloading support. The system will keep models "warm" in CPU RAM for fast restoration while gracefully handling both large (16-24+ GB) and constrained (4-8 GB) GPU memory configurations.

## Problem
- `GovernedModelManager` exists but imports from non-existent `memory_governor.py`
- `cpu_offloader.py` and `memory_api.py` are complete but unused without the governor
- Current `ModelManager` uses simple timeout-based unloading (300s) with no memory awareness
- OOM errors occur because there's no intelligent eviction or memory pressure response

## Solution
Create `/home/john/semantik/packages/vecpipe/memory_governor.py` with:

### Core Components

**1. Enums & Data Classes**
- `ModelLocation`: GPU | CPU | UNLOADED
- `ModelType`: EMBEDDING | RERANKER
- `PressureLevel`: LOW (<60%) | MODERATE (60-80%) | HIGH (80-90%) | CRITICAL (>90%)
- `MemoryBudget`: GPU + CPU limits (reserve %, max %) - configurable independently
- `TrackedModel`: Model state tracking (location, memory, last_used, use_count)
- `EvictionRecord`: Eviction history for debugging

**2. GPUMemoryGovernor Class**
- LRU-based model tracking using `OrderedDict`
- Single unified budget (no per-type percentages - dynamic eviction handles allocation)
- Small system buffer (10%) reserved for activation memory spikes
- Callback-based lifecycle management (integrates with existing GovernedModelManager)
- Background pressure monitoring task

### Eviction Strategy (Priority Order)
When model needs to load but insufficient memory:
1. Offload LRU idle models to CPU first (preserves warm state for fast restore)
2. If still not enough, fully unload LRU models
3. Continue until enough space or no models left to evict
4. Never evict model currently being used (5s grace period)
5. Fail only if requested model won't fit even with empty GPU

### Memory Pressure Response

| Level | Threshold | Action |
|-------|-----------|--------|
| LOW | <60% | No action |
| MODERATE | 60-80% | Preemptively offload models idle >120s |
| HIGH | 80-90% | Aggressively offload models idle >30s |
| CRITICAL | >90% | Force unload all idle models |

### VRAM-Adaptive Behavior

**Large VRAM (16+ GB):**
- Warm model pool enabled (CPU offloading preferred)
- Both embedding + reranker can coexist
- Longer idle thresholds before eviction

**Small VRAM (4-8 GB):**
- More aggressive eviction (shorter idle thresholds)
- May need mutual exclusion (one model type at a time)
- CPU offloading still beneficial if sufficient RAM

## Files to Create/Modify

### New File
`/home/john/semantik/packages/vecpipe/memory_governor.py`
- ~400 lines
- Implements GPUMemoryGovernor and supporting classes
- Exports: GPUMemoryGovernor, MemoryBudget, ModelLocation, ModelType, PressureLevel, get_memory_governor, initialize_memory_governor

### Integration (Minor Updates)
`/home/john/semantik/packages/vecpipe/search/lifespan.py`
- Switch from `ModelManager` to `GovernedModelManager`
- Call `governor.start_monitor()` on startup

### Tests
`/home/john/semantik/tests/unit/test_memory_governor.py`
- Budget calculations
- LRU eviction ordering
- Pressure level responses
- Callback execution

`/home/john/semantik/tests/integration/test_governed_model_manager.py`
- End-to-end with real models
- Offload/restore cycles
- Concurrent requests

### Documentation Updates

**`/home/john/semantik/.env.docker.example`**
- Add new environment variables:
  - `GPU_MEMORY_RESERVE_PERCENT` (default: 0.10)
  - `GPU_MEMORY_MAX_PERCENT` (default: 0.90)
  - `CPU_MEMORY_RESERVE_PERCENT` (default: 0.20)
  - `CPU_MEMORY_MAX_PERCENT` (default: 0.50)
  - `ENABLE_CPU_OFFLOAD` (default: true)
  - `EVICTION_IDLE_THRESHOLD_SECONDS` (default: 120)

**`/home/john/semantik/packages/vecpipe/CLAUDE.md`**
- Update memory-management section with new governor behavior
- Document warm model pool / CPU offloading
- Document pressure levels and eviction strategy

**`/home/john/semantik/README.md`**
- Add section on memory configuration for different GPU sizes
- Document recommended settings for 8GB / 16GB / 24GB+ configurations

**`/home/john/semantik/docs/` (new or existing)**
- Create `memory-management.md` with:
  - Architecture overview (governor, offloader, pressure monitoring)
  - Configuration reference table
  - Troubleshooting OOM issues
  - API endpoints for monitoring (`/memory/stats`, `/memory/health`, etc.)

## Implementation Sequence

1. **Create memory_governor.py** with enums, data classes, and GPUMemoryGovernor skeleton
2. **Implement core methods**: request_memory, mark_loaded/unloaded, touch
3. **Implement eviction logic**: LRU ordering, offload vs unload decisions
4. **Implement pressure monitoring**: background task, threshold responses
5. **Add unit tests** for governor logic
6. **Update lifespan.py** to use GovernedModelManager
7. **Integration testing** with actual models
8. **Update configuration**: Add env vars to `.env.docker.example` and settings
9. **Update documentation**: CLAUDE.md, README.md, create docs/memory-management.md

## Key API Methods (matching GovernedModelManager expectations)

```python
# Memory allocation
async def request_memory(model_name, model_type, quantization, required_mb) -> bool

# Lifecycle tracking
async def mark_loaded(model_name, model_type, quantization, model_ref)
async def mark_unloaded(model_name, model_type, quantization)
async def touch(model_name, model_type, quantization)

# Callbacks
def register_callbacks(model_type, unload_fn, offload_fn)

# Monitoring
async def start_monitor()
async def shutdown()

# Status
def get_memory_stats() -> dict
def get_loaded_models() -> list
def get_eviction_history() -> list
```

## Configuration Defaults

```python
# GPU Memory Limits
gpu_reserve_percent = 0.10       # Always keep 10% VRAM free (safety buffer)
gpu_max_percent = 0.90           # Never use more than 90% of VRAM

# CPU Memory Limits (for offloaded/warm models)
cpu_reserve_percent = 0.20       # Always keep 20% RAM free for system
cpu_max_percent = 0.50           # Never use more than 50% of RAM for warm models

# Behavior
eviction_idle_threshold_seconds = 120  # Idle time before eligible for preemptive eviction
pressure_check_interval_seconds = 15   # Background pressure check interval
activation_overhead_factor = 1.2       # 20% overhead for activations during inference
```

**Memory limit logic:**
- `usable_gpu = total_gpu * min(gpu_max_percent, 1.0 - gpu_reserve_percent)`
- `usable_cpu = total_cpu * min(cpu_max_percent, 1.0 - cpu_reserve_percent)`
- If CPU warm pool is full, fall back to full unload instead of offload

**No per-model-type budgets** - LRU eviction naturally allocates memory based on actual usage.

## Success Criteria
- No OOM errors under normal operation
- Models offload to CPU instead of full unload when possible
- Restoration from CPU faster than reload from disk (~2-5s vs 10-30s)
- Graceful degradation on small VRAM systems
- Memory API endpoints return accurate stats
