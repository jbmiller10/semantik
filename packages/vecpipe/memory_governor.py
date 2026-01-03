"""
GPU Memory Governor - Dynamic memory management with LRU eviction and CPU offloading.

This module provides intelligent GPU memory management for ML models:
- LRU-based eviction when memory is needed
- CPU offloading to keep models "warm" for fast restoration
- Background pressure monitoring with preemptive eviction
- Configurable memory limits for both GPU and CPU
"""

import asyncio
import contextlib
import gc
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import psutil

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ModelLocation(Enum):
    """Where a model's weights currently reside."""

    GPU = auto()  # Model is loaded on GPU, ready for inference
    CPU = auto()  # Model offloaded to CPU RAM (warm state, fast restore)
    UNLOADED = auto()  # Model not loaded, must be loaded from disk


class ModelType(Enum):
    """Type of model for tracking purposes."""

    EMBEDDING = auto()
    RERANKER = auto()


class PressureLevel(Enum):
    """Memory pressure levels for monitoring."""

    LOW = auto()  # <60% used - no action needed
    MODERATE = auto()  # 60-80% - preemptive offloading of idle models
    HIGH = auto()  # 80-90% - aggressive offloading
    CRITICAL = auto()  # >90% - force unload to prevent OOM


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MemoryBudget:
    """Memory budget configuration for GPU and CPU."""

    # GPU limits (auto-detected if 0)
    total_gpu_mb: int = 0
    gpu_reserve_percent: float = 0.10  # Always keep 10% VRAM free
    gpu_max_percent: float = 0.90  # Never use more than 90% of VRAM

    # CPU limits (for offloaded/warm models)
    total_cpu_mb: int = 0  # Auto-detected if 0
    cpu_reserve_percent: float = 0.20  # Always keep 20% RAM free
    cpu_max_percent: float = 0.50  # Never use more than 50% for warm models

    def __post_init__(self) -> None:
        """Auto-detect GPU and CPU memory if not provided."""
        # Auto-detect GPU memory
        if self.total_gpu_mb == 0:
            try:
                import torch

                if torch.cuda.is_available():
                    _, total_bytes = torch.cuda.mem_get_info()
                    self.total_gpu_mb = total_bytes // (1024 * 1024)
            except ImportError:
                pass  # No torch, leave at 0 (CPU-only mode)

        # Auto-detect CPU memory
        if self.total_cpu_mb == 0:
            self.total_cpu_mb = psutil.virtual_memory().total // (1024 * 1024)

    @property
    def usable_gpu_mb(self) -> int:
        """Usable GPU memory after reserves."""
        effective_max = min(self.gpu_max_percent, 1.0 - self.gpu_reserve_percent)
        return int(self.total_gpu_mb * effective_max)

    @property
    def usable_cpu_mb(self) -> int:
        """Usable CPU memory for warm models."""
        effective_max = min(self.cpu_max_percent, 1.0 - self.cpu_reserve_percent)
        return int(self.total_cpu_mb * effective_max)


@dataclass
class TrackedModel:
    """Tracks state of a single loaded/offloaded model."""

    model_name: str
    model_type: ModelType
    quantization: str
    location: ModelLocation
    memory_mb: int
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    model_ref: Any = None  # Reference to actual model object for offloading

    @property
    def model_key(self) -> str:
        """Unique key for this model configuration."""
        return f"{self.model_type.name.lower()}:{self.model_name}:{self.quantization}"

    @property
    def idle_seconds(self) -> float:
        """Seconds since last use."""
        return time.time() - self.last_used


@dataclass
class EvictionRecord:
    """Record of a model eviction for history tracking."""

    model_name: str
    model_type: str
    quantization: str
    reason: str
    action: str  # "offloaded" or "unloaded"
    memory_freed_mb: int
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# GPUMemoryGovernor
# =============================================================================


class GPUMemoryGovernor:
    """
    Central coordinator for GPU memory allocation across models.

    Implements:
    - LRU-based eviction with CPU offloading preference
    - Memory budget enforcement
    - Background memory pressure monitoring
    - Callback-based model lifecycle management
    """

    def __init__(
        self,
        budget: MemoryBudget | None = None,
        enable_cpu_offload: bool = True,
        eviction_idle_threshold_seconds: int = 120,
        pressure_check_interval_seconds: int = 15,
    ):
        """
        Initialize the memory governor.

        Args:
            budget: Memory budget configuration (auto-detected if None)
            enable_cpu_offload: Whether to offload to CPU before unloading
            eviction_idle_threshold_seconds: Idle time before model eligible for eviction
            pressure_check_interval_seconds: Interval for background pressure checks
        """
        self._budget = budget or self._detect_budget()
        self._enable_cpu_offload = enable_cpu_offload
        self._eviction_idle_threshold = eviction_idle_threshold_seconds
        self._pressure_check_interval = pressure_check_interval_seconds

        # Model tracking: OrderedDict maintains insertion order for LRU
        # Key: model_key (e.g., "embedding:Qwen/Qwen3-Embedding-0.6B:float16")
        self._models: OrderedDict[str, TrackedModel] = OrderedDict()

        # Eviction history (bounded circular buffer)
        self._eviction_history: list[EvictionRecord] = []
        self._max_history_size = 100

        # Callbacks for model lifecycle operations
        # Dict[ModelType, Dict[str, Callable]]
        self._callbacks: dict[ModelType, dict[str, Callable[..., Awaitable[None]]]] = {
            ModelType.EMBEDDING: {},
            ModelType.RERANKER: {},
        }

        # Thread safety
        self._lock = asyncio.Lock()

        # Background monitor task
        self._monitor_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

        # Metrics
        self._total_evictions = 0
        self._total_offloads = 0
        self._total_restorations = 0
        self._total_unloads = 0

        logger.info(
            "GPUMemoryGovernor initialized: gpu_budget=%dMB (usable=%dMB), "
            "cpu_budget=%dMB (usable=%dMB), cpu_offload=%s",
            self._budget.total_gpu_mb,
            self._budget.usable_gpu_mb,
            self._budget.total_cpu_mb,
            self._budget.usable_cpu_mb,
            enable_cpu_offload,
        )

    # -------------------------------------------------------------------------
    # Budget Detection
    # -------------------------------------------------------------------------

    def _detect_budget(self) -> MemoryBudget:
        """Auto-detect GPU and CPU memory budget."""
        try:
            import torch

            if torch.cuda.is_available():
                _, total_bytes = torch.cuda.mem_get_info()
                total_gpu_mb = total_bytes // (1024 * 1024)
            else:
                total_gpu_mb = 0
        except ImportError:
            total_gpu_mb = 0

        return MemoryBudget(total_gpu_mb=total_gpu_mb)

    # -------------------------------------------------------------------------
    # Core Memory Management Methods
    # -------------------------------------------------------------------------

    async def request_memory(
        self,
        model_name: str,
        model_type: ModelType,
        quantization: str,
        required_mb: int,
    ) -> bool:
        """
        Request memory allocation for a model.

        This method:
        1. Checks if model is already loaded/offloaded
        2. Checks available budget
        3. Evicts/offloads other models if needed
        4. Returns True if allocation can proceed

        Args:
            model_name: Model identifier
            model_type: EMBEDDING or RERANKER
            quantization: Quantization type (float32, float16, int8)
            required_mb: Memory required in MB

        Returns:
            True if memory can be allocated, False otherwise
        """
        async with self._lock:
            # CPU-only mode: no GPU budget to enforce, always allow
            if self._budget.total_gpu_mb == 0:
                logger.debug(
                    "CPU-only mode: allowing %s load without GPU budget check",
                    model_name,
                )
                return True

            model_key = self._make_key(model_name, model_type, quantization)

            # Case 1: Model already on GPU
            if model_key in self._models:
                tracked = self._models[model_key]
                if tracked.location == ModelLocation.GPU:
                    self._touch_model(model_key)
                    return True
                if tracked.location == ModelLocation.CPU:
                    # Model is offloaded, need to restore
                    return await self._restore_from_cpu(model_key, required_mb)

            # Case 2: Need to allocate new memory
            # Note: required_mb already includes overhead from get_model_memory_requirement
            current_gpu_usage = self._get_gpu_usage()

            if current_gpu_usage + required_mb <= self._budget.usable_gpu_mb:
                # Fits within budget
                logger.debug(
                    "Memory request approved: %s needs %dMB, current=%dMB, budget=%dMB",
                    model_key,
                    required_mb,
                    current_gpu_usage,
                    self._budget.usable_gpu_mb,
                )
                return True

            # Case 3: Need to make room
            needed_mb = (current_gpu_usage + required_mb) - self._budget.usable_gpu_mb
            logger.info(
                "Memory request requires eviction: %s needs %dMB, must free %dMB",
                model_key,
                required_mb,
                needed_mb,
            )
            freed_mb = await self._make_room(needed_mb, exclude_key=model_key)

            return freed_mb >= needed_mb

    async def mark_loaded(
        self,
        model_name: str,
        model_type: ModelType,
        quantization: str,
        model_ref: Any = None,
    ) -> None:
        """
        Mark a model as loaded on GPU.

        Called after successful model load to register with governor.
        """
        async with self._lock:
            model_key = self._make_key(model_name, model_type, quantization)
            required_mb = self._get_model_memory(model_name, quantization)

            tracked = TrackedModel(
                model_name=model_name,
                model_type=model_type,
                quantization=quantization,
                location=ModelLocation.GPU,
                memory_mb=required_mb,
                model_ref=model_ref,
            )

            # Add to models dict (moves to end for LRU ordering)
            self._models[model_key] = tracked
            self._models.move_to_end(model_key)

            logger.info("Model registered: %s (%dMB) on GPU", model_key, required_mb)

    async def mark_unloaded(
        self,
        model_name: str,
        model_type: ModelType,
        quantization: str,
    ) -> None:
        """Mark a model as unloaded (fully removed from memory)."""
        async with self._lock:
            model_key = self._make_key(model_name, model_type, quantization)
            if model_key in self._models:
                del self._models[model_key]
                logger.info("Model unregistered: %s", model_key)

    async def touch(
        self,
        model_name: str,
        model_type: ModelType,
        quantization: str,
    ) -> None:
        """Update last_used timestamp and move to end of LRU."""
        async with self._lock:
            model_key = self._make_key(model_name, model_type, quantization)
            self._touch_model(model_key)

    # -------------------------------------------------------------------------
    # Eviction and Offloading
    # -------------------------------------------------------------------------

    async def _make_room(
        self,
        needed_mb: int,
        exclude_key: str | None = None,
    ) -> int:
        """
        Free memory by offloading/unloading models.

        Strategy:
        1. Offload idle models to CPU first (preserves warm state)
        2. Unload if CPU offload disabled or CPU is full
        3. Evict LRU models first

        Returns:
            Amount of memory freed in MB
        """
        freed_mb = 0

        # Get candidates sorted by last_used (oldest first = LRU)
        candidates = self._get_eviction_candidates(exclude_key)

        for tracked in candidates:
            if freed_mb >= needed_mb:
                break

            # Skip models in active use (recently touched)
            if tracked.idle_seconds < 5:  # 5 second grace period
                logger.debug("Skipping %s - recently used (%.1fs ago)", tracked.model_key, tracked.idle_seconds)
                continue

            # Try CPU offload first if enabled and there's room
            if (
                self._enable_cpu_offload
                and tracked.location == ModelLocation.GPU
                and self._can_offload_to_cpu(tracked.memory_mb)
            ):
                success = await self._offload_model(tracked)
                if success:
                    freed_mb += tracked.memory_mb
                    self._total_offloads += 1
                    continue

            # Fall back to full unload
            if tracked.location == ModelLocation.GPU:
                success = await self._unload_model(tracked)
                if success:
                    freed_mb += tracked.memory_mb
                    self._total_unloads += 1

        logger.info("Eviction complete: freed %dMB (needed %dMB)", freed_mb, needed_mb)
        return freed_mb

    def _get_eviction_candidates(self, exclude_key: str | None) -> list[TrackedModel]:
        """
        Get models sorted by eviction priority (LRU - oldest first).
        """
        candidates = []

        for key, tracked in self._models.items():
            if key == exclude_key:
                continue
            if tracked.location == ModelLocation.GPU:
                candidates.append(tracked)

        # Sort by last_used (oldest first = LRU)
        candidates.sort(key=lambda m: m.last_used)

        return candidates

    def _can_offload_to_cpu(self, memory_mb: int) -> bool:
        """Check if there's room in CPU warm pool."""
        current_cpu_usage = self._get_cpu_usage()
        return (current_cpu_usage + memory_mb) <= self._budget.usable_cpu_mb

    async def _offload_model(self, tracked: TrackedModel) -> bool:
        """Offload a model to CPU via callback."""
        callback = self._callbacks.get(tracked.model_type, {}).get("offload")
        if not callback:
            logger.warning("No offload callback for %s", tracked.model_type)
            return False

        try:
            await callback(tracked.model_name, tracked.quantization, "cpu")
            tracked.location = ModelLocation.CPU

            self._record_eviction(tracked, reason="memory_pressure", action="offloaded")

            logger.info(
                "Offloaded %s to CPU (freed %dMB GPU, idle %.1fs)",
                tracked.model_key,
                tracked.memory_mb,
                tracked.idle_seconds,
            )
            return True
        except Exception as e:
            logger.error("Failed to offload %s: %s", tracked.model_key, e)
            return False

    async def _unload_model(self, tracked: TrackedModel) -> bool:
        """Fully unload a model via callback."""
        callback = self._callbacks.get(tracked.model_type, {}).get("unload")
        if not callback:
            logger.warning("No unload callback for %s", tracked.model_type)
            return False

        try:
            await callback(tracked.model_name, tracked.quantization)

            self._record_eviction(tracked, reason="memory_pressure", action="unloaded")

            # Remove from tracking
            model_key = tracked.model_key
            if model_key in self._models:
                del self._models[model_key]

            logger.info(
                "Unloaded %s (freed %dMB, idle %.1fs)",
                model_key,
                tracked.memory_mb,
                tracked.idle_seconds,
            )
            return True
        except Exception as e:
            logger.error("Failed to unload %s: %s", tracked.model_key, e)
            return False

    async def _restore_from_cpu(self, model_key: str, required_mb: int) -> bool:
        """Restore an offloaded model from CPU to GPU."""
        tracked = self._models.get(model_key)
        if not tracked or tracked.location != ModelLocation.CPU:
            return False

        # Ensure we have room on GPU
        # Note: required_mb already includes overhead from get_model_memory_requirement
        current_gpu_usage = self._get_gpu_usage()

        if current_gpu_usage + required_mb > self._budget.usable_gpu_mb:
            needed = (current_gpu_usage + required_mb) - self._budget.usable_gpu_mb
            freed = await self._make_room(needed, exclude_key=model_key)
            if freed < needed:
                logger.warning("Cannot restore %s - insufficient GPU memory after eviction", model_key)
                return False

        # Restore via callback
        callback = self._callbacks.get(tracked.model_type, {}).get("offload")
        if callback:
            try:
                await callback(tracked.model_name, tracked.quantization, "cuda")
                tracked.location = ModelLocation.GPU
                tracked.last_used = time.time()
                self._models.move_to_end(model_key)
                self._total_restorations += 1

                logger.info("Restored %s from CPU to GPU", model_key)
                return True
            except Exception as e:
                logger.error("Failed to restore %s: %s", model_key, e)

        return False

    # -------------------------------------------------------------------------
    # Memory Pressure Monitoring
    # -------------------------------------------------------------------------

    async def start_monitor(self) -> None:
        """Start the background memory pressure monitor."""
        if self._monitor_task is not None:
            return

        self._shutdown_event.clear()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "Memory pressure monitor started (interval=%ds)",
            self._pressure_check_interval,
        )

    async def shutdown(self) -> None:
        """Shutdown the governor and stop monitoring."""
        self._shutdown_event.set()

        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
            self._monitor_task = None

        logger.info("GPUMemoryGovernor shutdown complete")

    async def _monitor_loop(self) -> None:
        """Background loop for memory pressure monitoring."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self._pressure_check_interval)

                pressure = self._calculate_pressure_level()

                if pressure == PressureLevel.CRITICAL:
                    logger.error(
                        "CRITICAL memory pressure! Usage: %.1f%%. Forcing aggressive eviction.",
                        self._get_usage_percent(),
                    )
                    await self._handle_critical_pressure()

                elif pressure == PressureLevel.HIGH:
                    logger.warning(
                        "HIGH memory pressure: %.1f%%. Offloading idle models.",
                        self._get_usage_percent(),
                    )
                    await self._handle_high_pressure()

                elif pressure == PressureLevel.MODERATE:
                    logger.debug(
                        "MODERATE memory pressure: %.1f%%. Preemptive offloading.",
                        self._get_usage_percent(),
                    )
                    await self._handle_moderate_pressure()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in memory monitor: %s", e)

    def _calculate_pressure_level(self) -> PressureLevel:
        """Calculate current memory pressure level."""
        usage_percent = self._get_usage_percent()

        if usage_percent >= 90:
            return PressureLevel.CRITICAL
        if usage_percent >= 80:
            return PressureLevel.HIGH
        if usage_percent >= 60:
            return PressureLevel.MODERATE
        return PressureLevel.LOW

    async def _handle_critical_pressure(self) -> None:
        """Handle critical memory pressure - force unload all idle models."""
        async with self._lock:
            for tracked in list(self._models.values()):
                if tracked.location == ModelLocation.GPU and tracked.idle_seconds > 5:
                    await self._unload_model(tracked)
            gc.collect()

    async def _handle_high_pressure(self) -> None:
        """Handle high memory pressure - aggressive offloading."""
        async with self._lock:
            for tracked in list(self._models.values()):
                if tracked.location == ModelLocation.GPU and tracked.idle_seconds > 30:
                    if self._enable_cpu_offload and self._can_offload_to_cpu(tracked.memory_mb):
                        await self._offload_model(tracked)
                    else:
                        await self._unload_model(tracked)

    async def _handle_moderate_pressure(self) -> None:
        """Handle moderate pressure - preemptive offloading of idle models."""
        async with self._lock:
            for tracked in list(self._models.values()):
                is_idle_on_gpu = (
                    tracked.location == ModelLocation.GPU
                    and tracked.idle_seconds >= self._eviction_idle_threshold
                )
                can_offload = self._enable_cpu_offload and self._can_offload_to_cpu(tracked.memory_mb)
                if is_idle_on_gpu and can_offload:
                    await self._offload_model(tracked)

    # -------------------------------------------------------------------------
    # Callback Registration
    # -------------------------------------------------------------------------

    def register_callbacks(
        self,
        model_type: ModelType,
        unload_fn: Callable[[str, str], Awaitable[None]],
        offload_fn: Callable[[str, str, str], Awaitable[None]] | None = None,
    ) -> None:
        """
        Register callbacks for model lifecycle operations.

        Args:
            model_type: Type of model these callbacks handle
            unload_fn: async fn(model_name, quantization) to fully unload
            offload_fn: async fn(model_name, quantization, target_device) to offload/restore
        """
        self._callbacks[model_type] = {
            "unload": unload_fn,
        }
        if offload_fn:
            self._callbacks[model_type]["offload"] = offload_fn
        logger.debug("Registered callbacks for %s", model_type.name)

    # -------------------------------------------------------------------------
    # Status and Metrics
    # -------------------------------------------------------------------------

    def get_memory_stats(self) -> dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_mb = free_bytes // (1024 * 1024)
                total_mb = total_bytes // (1024 * 1024)
                used_mb = total_mb - free_mb
            else:
                free_mb = total_mb = used_mb = 0
        except ImportError:
            cuda_available = False
            free_mb = total_mb = used_mb = 0

        # Governor tracking
        gpu_models = [m for m in self._models.values() if m.location == ModelLocation.GPU]
        cpu_models = [m for m in self._models.values() if m.location == ModelLocation.CPU]

        return {
            "cuda_available": cuda_available,
            "total_mb": total_mb,
            "free_mb": free_mb,
            "used_mb": used_mb,
            "used_percent": (used_mb / total_mb * 100) if total_mb > 0 else 0,
            "allocated_mb": sum(m.memory_mb for m in gpu_models),
            "reserved_mb": self._budget.total_gpu_mb,
            "budget_total_mb": self._budget.total_gpu_mb,
            "budget_usable_mb": self._budget.usable_gpu_mb,
            "cpu_budget_total_mb": self._budget.total_cpu_mb,
            "cpu_budget_usable_mb": self._budget.usable_cpu_mb,
            "cpu_used_mb": sum(m.memory_mb for m in cpu_models),
            "models_loaded": len(gpu_models),
            "models_offloaded": len(cpu_models),
            "pressure_level": self._calculate_pressure_level().name,
            "total_evictions": self._total_evictions,
            "total_offloads": self._total_offloads,
            "total_restorations": self._total_restorations,
            "total_unloads": self._total_unloads,
        }

    def get_loaded_models(self) -> list[dict[str, Any]]:
        """Get list of all tracked models with details."""
        return [
            {
                "model_name": m.model_name,
                "model_type": m.model_type.name.lower(),
                "quantization": m.quantization,
                "location": m.location.name.lower(),
                "memory_mb": m.memory_mb,
                "idle_seconds": m.idle_seconds,
                "use_count": m.use_count,
            }
            for m in self._models.values()
        ]

    def get_eviction_history(self) -> list[dict[str, Any]]:
        """Get recent eviction history."""
        return [
            {
                "model_name": r.model_name,
                "model_type": r.model_type,
                "quantization": r.quantization,
                "reason": r.reason,
                "action": r.action,
                "memory_freed_mb": r.memory_freed_mb,
                "timestamp": r.timestamp,
            }
            for r in self._eviction_history
        ]

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _make_key(self, model_name: str, model_type: ModelType, quantization: str) -> str:
        """Create unique model key."""
        return f"{model_type.name.lower()}:{model_name}:{quantization}"

    def _touch_model(self, model_key: str) -> None:
        """Update model's last_used and move to end of LRU."""
        if model_key in self._models:
            self._models[model_key].last_used = time.time()
            self._models[model_key].use_count += 1
            self._models.move_to_end(model_key)

    def _get_gpu_usage(self) -> int:
        """Get current GPU memory usage (tracked models only)."""
        return sum(m.memory_mb for m in self._models.values() if m.location == ModelLocation.GPU)

    def _get_cpu_usage(self) -> int:
        """Get current CPU memory usage for offloaded models."""
        return sum(m.memory_mb for m in self._models.values() if m.location == ModelLocation.CPU)

    def _get_usage_percent(self) -> float:
        """Get GPU usage as percentage of usable budget."""
        gpu_usage = self._get_gpu_usage()
        return (gpu_usage / self._budget.usable_gpu_mb * 100) if self._budget.usable_gpu_mb > 0 else 0

    def _get_model_memory(self, model_name: str, quantization: str) -> int:
        """Get memory requirement for a model."""
        try:
            from .memory_utils import get_model_memory_requirement

            return get_model_memory_requirement(model_name, quantization)
        except ImportError:
            # Fallback estimate
            logger.warning("Could not import memory_utils, using fallback estimate")
            return 2000  # 2GB default

    def _record_eviction(self, tracked: TrackedModel, reason: str, action: str) -> None:
        """Record an eviction event."""
        record = EvictionRecord(
            model_name=tracked.model_name,
            model_type=tracked.model_type.name.lower(),
            quantization=tracked.quantization,
            reason=reason,
            action=action,
            memory_freed_mb=tracked.memory_mb,
        )
        self._eviction_history.append(record)
        self._total_evictions += 1

        # Trim history if too large
        if len(self._eviction_history) > self._max_history_size:
            self._eviction_history = self._eviction_history[-self._max_history_size :]


# =============================================================================
# Singleton Pattern
# =============================================================================

_governor: GPUMemoryGovernor | None = None
_governor_lock = threading.Lock()


def get_memory_governor() -> GPUMemoryGovernor:
    """Get the singleton governor instance."""
    global _governor
    if _governor is None:
        raise RuntimeError("Memory governor not initialized. Call initialize_memory_governor() first.")
    return _governor


def initialize_memory_governor(
    budget: MemoryBudget | None = None,
    enable_cpu_offload: bool = True,
    **kwargs: Any,
) -> GPUMemoryGovernor:
    """Initialize the singleton governor instance."""
    global _governor
    with _governor_lock:
        if _governor is not None:
            logger.warning("Memory governor already initialized, returning existing instance")
            return _governor

        _governor = GPUMemoryGovernor(
            budget=budget,
            enable_cpu_offload=enable_cpu_offload,
            **kwargs,
        )
        return _governor


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "GPUMemoryGovernor",
    "MemoryBudget",
    "ModelLocation",
    "ModelType",
    "PressureLevel",
    "TrackedModel",
    "EvictionRecord",
    "get_memory_governor",
    "initialize_memory_governor",
]
