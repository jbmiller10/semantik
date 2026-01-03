"""
Memory management API endpoints for VecPipe.

Provides endpoints for monitoring and managing GPU memory,
model lifecycle, and memory pressure.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from vecpipe.cpu_offloader import (
    defragment_cuda_memory,
    get_cuda_memory_fragmentation,
    get_offloader,
)
from vecpipe.search.state import get_resources

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryStatsResponse(BaseModel):
    """Response model for memory statistics."""
    cuda_available: bool
    total_mb: int = 0
    free_mb: int = 0
    used_mb: int = 0
    used_percent: float = 0.0
    allocated_mb: int = 0
    reserved_mb: int = 0
    budget_total_mb: int = 0
    budget_usable_mb: int = 0
    cpu_budget_total_mb: int = 0
    cpu_budget_usable_mb: int = 0
    cpu_used_mb: int = 0
    models_loaded: int = 0
    models_offloaded: int = 0
    pressure_level: str = "UNKNOWN"
    total_evictions: int = 0
    total_offloads: int = 0
    total_restorations: int = 0
    total_unloads: int = 0


class LoadedModelInfo(BaseModel):
    """Info about a loaded model."""
    model_name: str
    model_type: str
    quantization: str
    location: str
    memory_mb: int
    idle_seconds: float
    use_count: int


class EvictionInfo(BaseModel):
    """Info about a model eviction."""
    model_name: str
    model_type: str
    quantization: str
    reason: str
    action: str  # "offloaded" or "unloaded"
    memory_freed_mb: int
    timestamp: float


class FragmentationInfo(BaseModel):
    """CUDA memory fragmentation info."""
    cuda_available: bool = False
    allocated_mb: int = 0
    reserved_mb: int = 0
    fragmentation_mb: int = 0
    fragmentation_percent: float = 0.0
    num_alloc_retries: int = 0
    num_ooms: int = 0


class PreloadModelSpec(BaseModel):
    """Specification for a model to preload."""
    name: str
    model_type: str  # "embedding" or "reranker"
    quantization: str

    def model_post_init(self, __context: Any) -> None:
        """Validate and normalize model_type after initialization."""
        valid_types = {"embedding", "reranker"}
        # Case-insensitive comparison
        normalized = self.model_type.lower().strip()
        if normalized not in valid_types:
            raise ValueError(
                f"model_type must be one of {valid_types} (case-insensitive), "
                f"got '{self.model_type}'"
            )
        # Normalize to lowercase
        object.__setattr__(self, "model_type", normalized)


class PreloadRequest(BaseModel):
    """Request to preload models."""
    models: list[PreloadModelSpec]


class PreloadResponse(BaseModel):
    """Response from preload request."""
    results: dict[str, bool]


@router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats() -> dict[str, Any]:
    """
    Get current GPU memory statistics.

    Returns detailed memory usage including:
    - Total, free, used memory
    - Budget allocations
    - Pressure level
    - Loaded model count
    """
    resources = get_resources()
    model_mgr = resources.get("model_mgr")

    if model_mgr is None:
        return {"cuda_available": False}

    # Check if using governed manager
    if hasattr(model_mgr, "_governor"):
        return model_mgr._governor.get_memory_stats()

    # Fallback for non-governed manager
    import torch
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_mb = free_bytes // (1024 * 1024)
    total_mb = total_bytes // (1024 * 1024)
    used_mb = total_mb - free_mb

    return {
        "cuda_available": True,
        "total_mb": total_mb,
        "free_mb": free_mb,
        "used_mb": used_mb,
        "used_percent": (used_mb / total_mb * 100) if total_mb > 0 else 0,
        "models_loaded": 1 if model_mgr.current_model_key else 0,
        "pressure_level": "UNKNOWN",
    }


@router.get("/models", response_model=list[LoadedModelInfo])
async def get_loaded_models() -> list[dict[str, Any]]:
    """
    Get information about currently loaded models.

    Returns list of models with their location (GPU/CPU), memory usage,
    and idle time.
    """
    resources = get_resources()
    model_mgr = resources.get("model_mgr")

    if model_mgr is None:
        return []

    # Check if using governed manager
    if hasattr(model_mgr, "_governor"):
        return model_mgr._governor.get_loaded_models()

    # Fallback for non-governed manager
    models = []
    if model_mgr.current_model_key:
        parts = model_mgr.current_model_key.rsplit("_", 1)
        if len(parts) == 2:
            models.append({
                "model_name": parts[0],
                "model_type": "embedding",
                "quantization": parts[1],
                "location": "gpu",
                "memory_mb": 0,
                "idle_seconds": model_mgr.last_used,
                "use_count": 0,
            })

    if model_mgr.current_reranker_key:
        parts = model_mgr.current_reranker_key.rsplit("_", 1)
        if len(parts) == 2:
            models.append({
                "model_name": parts[0],
                "model_type": "reranker",
                "quantization": parts[1],
                "location": "gpu",
                "memory_mb": 0,
                "idle_seconds": model_mgr.last_reranker_used,
                "use_count": 0,
            })

    return models


@router.get("/evictions", response_model=list[EvictionInfo])
async def get_eviction_history() -> list[dict[str, Any]]:
    """
    Get recent model eviction history.

    Useful for understanding memory pressure patterns and model lifecycle.
    """
    resources = get_resources()
    model_mgr = resources.get("model_mgr")

    if model_mgr is None or not hasattr(model_mgr, "_governor"):
        return []

    return model_mgr._governor.get_eviction_history()


@router.get("/fragmentation", response_model=FragmentationInfo)
async def get_fragmentation() -> dict[str, Any]:
    """
    Get CUDA memory fragmentation analysis.

    High fragmentation can cause OOM even with free memory.
    """
    return get_cuda_memory_fragmentation()


@router.post("/defragment")
async def trigger_defragment() -> dict[str, str]:
    """
    Trigger CUDA memory defragmentation.

    This clears caches and resets memory stats.
    """
    defragment_cuda_memory()
    return {"status": "defragmentation_triggered"}


@router.post("/evict/{model_type}")
async def evict_model(model_type: str) -> dict[str, Any]:
    """
    Manually evict a model to free GPU memory.

    Args:
        model_type: "embedding" or "reranker"

    Returns:
        Status dict with 'evicted' (bool) and details about what was unloaded.
    """
    resources = get_resources()
    model_mgr = resources.get("model_mgr")

    if model_mgr is None:
        raise HTTPException(status_code=503, detail="Model manager not available")

    if model_type == "embedding":
        # Check if there's actually a model to evict
        current_key = getattr(model_mgr, "current_model_key", None)
        if not current_key:
            return {
                "status": "no_action",
                "model_type": "embedding",
                "message": "No embedding model currently loaded",
            }

        if hasattr(model_mgr, "unload_model_async"):
            await model_mgr.unload_model_async()
        else:
            model_mgr.unload_model()

        # Verify eviction succeeded
        new_key = getattr(model_mgr, "current_model_key", None)
        if new_key is None:
            return {
                "status": "evicted",
                "model_type": "embedding",
                "evicted_model": current_key,
            }
        logger.error("Eviction failed: embedding model still loaded as %s", new_key)
        raise HTTPException(
            status_code=500,
            detail=f"Eviction failed: model still loaded as {new_key}",
        )

    if model_type == "reranker":
        # Check if there's actually a reranker to evict
        current_key = getattr(model_mgr, "current_reranker_key", None)
        if not current_key:
            return {
                "status": "no_action",
                "model_type": "reranker",
                "message": "No reranker model currently loaded",
            }

        model_mgr.unload_reranker()

        # Verify eviction succeeded
        new_key = getattr(model_mgr, "current_reranker_key", None)
        if new_key is None:
            return {
                "status": "evicted",
                "model_type": "reranker",
                "evicted_model": current_key,
            }
        logger.error("Eviction failed: reranker still loaded as %s", new_key)
        raise HTTPException(
            status_code=500,
            detail=f"Eviction failed: reranker still loaded as {new_key}",
        )

    raise HTTPException(status_code=400, detail="Invalid model_type. Use 'embedding' or 'reranker'")


@router.post("/preload", response_model=PreloadResponse)
async def preload_models(request: PreloadRequest) -> dict[str, Any]:
    """
    Preload models for expected requests.

    This allows warming up models before they're needed.
    """
    resources = get_resources()
    model_mgr = resources.get("model_mgr")

    if model_mgr is None:
        raise HTTPException(status_code=503, detail="Model manager not available")

    if not hasattr(model_mgr, "preload_models"):
        raise HTTPException(
            status_code=501,
            detail="Preloading not supported (requires GovernedModelManager)"
        )

    # Convert from PreloadModelSpec to tuples expected by preload_models
    model_tuples = [
        (spec.name, spec.model_type, spec.quantization)
        for spec in request.models
    ]
    results = await model_mgr.preload_models(model_tuples)
    return {"results": results}


@router.get("/offloaded")
async def get_offloaded_models() -> list[dict[str, Any]]:
    """
    Get list of models currently offloaded to CPU.
    """
    offloader = get_offloader()
    offloaded_keys = offloader.get_offloaded_models()

    return [
        {
            "model_key": key,
            **(offloader.get_offload_info(key) or {}),
        }
        for key in offloaded_keys
    ]


@router.get("/health")
async def memory_health_check() -> dict[str, Any]:
    """
    Memory health check endpoint.

    Returns overall health status based on memory pressure.
    """
    stats = await get_memory_stats()

    if not stats.get("cuda_available"):
        return {
            "healthy": True,
            "mode": "cpu",
            "message": "Running in CPU mode",
        }

    pressure = stats.get("pressure_level", "UNKNOWN")

    if pressure == "CRITICAL":
        return {
            "healthy": False,
            "pressure": pressure,
            "used_percent": stats.get("used_percent", 0),
            "message": "Critical memory pressure - OOM risk",
        }
    if pressure == "HIGH":
        return {
            "healthy": True,
            "pressure": pressure,
            "used_percent": stats.get("used_percent", 0),
            "message": "High memory pressure - eviction active",
        }
    return {
        "healthy": True,
        "pressure": pressure,
        "used_percent": stats.get("used_percent", 0),
        "message": "Memory usage normal",
    }
