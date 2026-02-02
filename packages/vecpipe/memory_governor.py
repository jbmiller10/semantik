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
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    from torch import nn

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
    SPARSE = auto()  # SPLADE sparse indexer models
    LLM = auto()  # Local LLM models


class PressureLevel(Enum):
    """Memory pressure levels for monitoring."""

    LOW = auto()  # <60% used - no action needed
    MODERATE = auto()  # 60-80% - preemptive offloading of idle models
    HIGH = auto()  # 80-90% - aggressive offloading
    CRITICAL = auto()  # >90% - force unload to prevent OOM


class EvictionAction(Enum):
    """Type of eviction action taken."""

    OFFLOADED = auto()  # Model moved to CPU RAM (warm state)
    UNLOADED = auto()  # Model fully unloaded from memory


class ProbeMode(str, Enum):
    """GPU memory probe mode for accuracy/speed tradeoff."""

    FAST = "fast"  # mem_get_info() only - lowest latency
    SAFE = "safe"  # synchronize() + mem_get_info() - moderate latency
    AGGRESSIVE = "aggressive"  # gc + sync + empty_cache + mem_get_info - highest accuracy


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class MemoryBudget:
    """Memory budget configuration for GPU and CPU.

    This is a frozen (immutable) dataclass to prevent post-validation mutation.
    Use create_memory_budget() factory for auto-detection, or pass explicit values.
    """

    # GPU limits
    total_gpu_mb: int
    gpu_max_percent: float = 0.90  # Maximum GPU memory the application can use

    # CPU limits (for offloaded/warm models)
    total_cpu_mb: int = 0
    cpu_max_percent: float = 0.50  # Maximum CPU memory for warm models

    def __post_init__(self) -> None:
        """Validate all fields."""
        self._validate_percentages()
        self._validate_memory_values()

    def _validate_percentages(self) -> None:
        """Validate percentage fields are in valid range [0.0, 1.0]."""
        for field_name in ("gpu_max_percent", "cpu_max_percent"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value}")

    def _validate_memory_values(self) -> None:
        """Validate memory values are non-negative."""
        if self.total_gpu_mb < 0:
            raise ValueError(f"total_gpu_mb must be non-negative, got {self.total_gpu_mb}")
        if self.total_cpu_mb < 0:
            raise ValueError(f"total_cpu_mb must be non-negative, got {self.total_cpu_mb}")

    @property
    def usable_gpu_mb(self) -> int:
        """Usable GPU memory based on max percent."""
        return int(self.total_gpu_mb * self.gpu_max_percent)

    @property
    def usable_cpu_mb(self) -> int:
        """Usable CPU memory for warm models."""
        return int(self.total_cpu_mb * self.cpu_max_percent)


def create_memory_budget(
    total_gpu_mb: int | None = None,
    total_cpu_mb: int | None = None,
    gpu_max_percent: float = 0.90,
    cpu_max_percent: float = 0.50,
) -> MemoryBudget:
    """
    Factory function to create MemoryBudget with auto-detection.

    Args:
        total_gpu_mb: GPU memory in MB (auto-detected if None)
        total_cpu_mb: CPU memory in MB (auto-detected if None)
        gpu_max_percent: Maximum GPU memory percentage (default 0.90)
        cpu_max_percent: Maximum CPU memory percentage (default 0.50)

    Returns:
        Configured MemoryBudget instance
    """
    # Auto-detect GPU memory
    if total_gpu_mb is None:
        total_gpu_mb = 0
        try:
            import torch

            if torch.cuda.is_available():
                _, total_bytes = torch.cuda.mem_get_info()
                total_gpu_mb = total_bytes // (1024 * 1024)
            else:
                logger.info("CUDA not available, running in CPU-only mode (total_gpu_mb=0)")
        except ImportError:
            logger.info("PyTorch not installed, running in CPU-only mode (total_gpu_mb=0)")

    # Auto-detect CPU memory
    if total_cpu_mb is None:
        total_cpu_mb = psutil.virtual_memory().total // (1024 * 1024)

    return MemoryBudget(
        total_gpu_mb=total_gpu_mb,
        total_cpu_mb=total_cpu_mb,
        gpu_max_percent=gpu_max_percent,
        cpu_max_percent=cpu_max_percent,
    )


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
    model_ref: "nn.Module | None" = None  # Reference to actual model for offloading

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if self.memory_mb < 0:
            raise ValueError(f"memory_mb must be non-negative, got {self.memory_mb}")
        if self.use_count < 0:
            raise ValueError(f"use_count must be non-negative, got {self.use_count}")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.quantization:
            raise ValueError("quantization cannot be empty")

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
    model_type: ModelType
    quantization: str
    reason: str
    action: EvictionAction
    memory_freed_mb: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for API responses."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.name.lower(),
            "quantization": self.quantization,
            "reason": self.reason,
            "action": self.action.name.lower(),
            "memory_freed_mb": self.memory_freed_mb,
            "timestamp": self.timestamp,
        }


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
        probe_mode: str = "fast",
        probe_safe_threshold: float = 0.85,
    ):
        """
        Initialize the memory governor.

        Args:
            budget: Memory budget configuration (auto-detected if None)
            enable_cpu_offload: Whether to offload to CPU before unloading
            eviction_idle_threshold_seconds: Idle time before model eligible for eviction
            pressure_check_interval_seconds: Interval for background pressure checks
            probe_mode: GPU probe mode (fast|safe|aggressive) for memory checks
            probe_safe_threshold: Usage fraction (0-1) above which to upgrade probe to SAFE
        """
        self._budget = budget or self._detect_budget()
        self._enable_cpu_offload = enable_cpu_offload
        self._eviction_idle_threshold = eviction_idle_threshold_seconds
        self._pressure_check_interval = pressure_check_interval_seconds
        self._probe_mode = ProbeMode(probe_mode)
        self._probe_safe_threshold = probe_safe_threshold

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
            ModelType.SPARSE: {},
            ModelType.LLM: {},
        }

        # Async coordination - prevents concurrent coroutine access to shared state
        self._lock = asyncio.Lock()

        # Background monitor task
        self._monitor_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

        # Metrics
        self._total_evictions = 0
        self._total_offloads = 0
        self._total_restorations = 0
        self._total_unloads = 0

        # Circuit breaker degraded state tracking
        self._degraded_state: bool = False
        self._circuit_breaker_triggers: int = 0
        self._max_circuit_breaker_triggers: int = 3

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
        return create_memory_budget()

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
        2. Checks both tracked AND actual GPU memory availability
        3. Evicts/offloads other models if needed (force=True for explicit requests)
        4. Returns True if allocation can proceed

        Args:
            model_name: Model identifier
            model_type: EMBEDDING or RERANKER
            quantization: Quantization type (float32, float16, int8)
            required_mb: Memory required in MB

        Returns:
            True if memory can be allocated, False otherwise
        """
        from vecpipe.search.metrics import memory_request_latency_seconds

        start = time.time()
        outcome = "failed"
        try:
            async with self._lock:
                # CPU-only mode: no GPU budget to enforce, always allow
                if self._budget.total_gpu_mb == 0:
                    logger.debug(
                        "CPU-only mode: allowing %s load without GPU budget check",
                        model_name,
                    )
                    outcome = "success"
                    return True

                model_key = self._make_key(model_name, model_type, quantization)
                reserve_free_mb = max(0, self._budget.total_gpu_mb - self._budget.usable_gpu_mb)

                # Case 1: Model already on GPU
                if model_key in self._models:
                    tracked = self._models[model_key]
                    if tracked.location == ModelLocation.GPU:
                        # Even if a model is already on GPU, we may need to evict other
                        # models to restore our configured free-memory reserve. This
                        # avoids sequential pipelines (embed -> rerank) ending up with
                        # both models resident and too little headroom for activations.
                        actual_free_mb = self._probe_free_gpu_mb(self._select_probe_mode())
                        if reserve_free_mb > 0 and actual_free_mb < reserve_free_mb:
                            needed_mb = reserve_free_mb - actual_free_mb
                            logger.info(
                                "Already-loaded model %s below reserve (free=%dMB, reserve=%dMB). Evicting %dMB.",
                                model_key,
                                actual_free_mb,
                                reserve_free_mb,
                                needed_mb,
                            )
                            try:
                                await self._make_room(needed_mb, exclude_key=model_key, force=True)
                            except RuntimeError as e:
                                # If callbacks are not registered we cannot evict anything; treat
                                # this as a failed allocation request rather than crashing.
                                logger.warning("Eviction unavailable while enforcing reserve for %s: %s", model_key, e)
                                outcome = "failed"
                                return False

                            # Use SAFE mode after eviction for accurate reading
                            actual_free_mb = self._probe_free_gpu_mb(ProbeMode.SAFE)
                            if actual_free_mb < reserve_free_mb:
                                logger.warning(
                                    "Reserve enforcement failed for %s (free=%dMB, reserve=%dMB).",
                                    model_key,
                                    actual_free_mb,
                                    reserve_free_mb,
                                )
                                outcome = "failed"
                                return False

                        self._touch_model(model_key)
                        outcome = "already_loaded"
                        return True
                    if tracked.location == ModelLocation.CPU:
                        # Model is offloaded, need to restore
                        result = await self._restore_from_cpu(model_key, required_mb)
                        outcome = "success" if result else "failed"
                        return result

                # Case 2: Check if we need to make room
                # Use BOTH tracked estimates AND actual GPU memory to catch discrepancies
                current_gpu_usage = self._get_gpu_usage()
                actual_free_mb = self._probe_free_gpu_mb(self._select_probe_mode())

                # Check tracked budget first
                tracked_fits = current_gpu_usage + required_mb <= self._budget.usable_gpu_mb
                # Also check actual GPU memory (the ground truth)
                actual_fits = actual_free_mb >= required_mb

                if tracked_fits and actual_fits:
                    # Both checks pass - memory is available
                    logger.debug(
                        "Memory request approved: %s needs %dMB, tracked=%dMB, actual_free=%dMB, budget=%dMB",
                        model_key,
                        required_mb,
                        current_gpu_usage,
                        actual_free_mb,
                        self._budget.usable_gpu_mb,
                    )
                    outcome = "success"
                    return True

                # Case 3: Need to make room
                # Calculate how much we need to free based on the tighter constraint
                needed_from_tracked = max(0, (current_gpu_usage + required_mb) - self._budget.usable_gpu_mb)
                needed_from_actual = max(0, required_mb - actual_free_mb)
                needed_mb = max(needed_from_tracked, needed_from_actual)

                logger.info(
                    "Memory request requires eviction: %s needs %dMB, must free %dMB "
                    "(tracked_usage=%dMB, actual_free=%dMB, budget=%dMB)",
                    model_key,
                    required_mb,
                    needed_mb,
                    current_gpu_usage,
                    actual_free_mb,
                    self._budget.usable_gpu_mb,
                )
                # Use force=True for explicit allocation requests - this bypasses
                # the 5-second grace period since we're not doing background eviction,
                # we're responding to an explicit request that needs memory NOW
                freed_mb = await self._make_room(needed_mb, exclude_key=model_key, force=True)

                result = freed_mb >= needed_mb
                outcome = "success" if result else "failed"
                return result
        finally:
            memory_request_latency_seconds.labels(outcome=outcome).observe(time.time() - start)

    async def mark_loaded(
        self,
        model_name: str,
        model_type: ModelType,
        quantization: str,
        model_ref: Any = None,
        memory_mb: int | None = None,
    ) -> None:
        """
        Mark a model as loaded on GPU.

        Called after successful model load to register with governor.

        Args:
            model_name: Model identifier
            model_type: Type of model (EMBEDDING, RERANKER, SPARSE, LLM)
            quantization: Quantization type
            model_ref: Reference to actual model for offloading
            memory_mb: Optional explicit memory size (used for LLMs where memory_utils
                       doesn't have size data). If None, uses _get_model_memory().
        """
        async with self._lock:
            model_key = self._make_key(model_name, model_type, quantization)
            required_mb = memory_mb if memory_mb is not None else self._get_model_memory(model_name, quantization)

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
        force: bool = False,
    ) -> int:
        """
        Free memory by offloading/unloading models.

        Strategy:
        1. Clear CUDA cache first to reclaim stale memory (critical when models_loaded=0)
        2. Offload idle models to CPU first (preserves warm state)
        3. Unload if CPU offload disabled or CPU is full
        4. Evict LRU models first
        5. Clear CUDA cache again after evictions to reclaim fragmented memory

        Args:
            needed_mb: Amount of memory to free
            exclude_key: Model key to exclude from eviction (the one we're making room for)
            force: If True, bypass the 5-second grace period. Use this for explicit
                   allocation requests where we need memory NOW. The grace period is
                   meant to protect models during active inference, but in a sequential
                   pipeline (embed query -> load reranker), the embedding model has
                   finished its work and can be safely evicted.

        Returns:
            Amount of memory freed in MB
        """
        # ALWAYS clear CUDA cache first to reclaim stale/fragmented memory.
        # This is critical when models_loaded=0 but PyTorch's cache still
        # holds memory from previously unloaded models. Without this, we
        # would skip _clear_cuda_cache() entirely if there are no candidates.
        self._clear_cuda_cache()

        freed_mb = 0
        evictions_occurred = False

        # Get candidates sorted by last_used (oldest first = LRU)
        candidates = self._get_eviction_candidates(exclude_key)

        for tracked in candidates:
            if freed_mb >= needed_mb:
                break

            # Skip models in active use (recently touched) - unless force=True
            # The grace period protects models during active inference, but for
            # explicit allocation requests, we need the memory NOW
            if not force and tracked.idle_seconds < 5:  # 5 second grace period
                logger.debug("Skipping %s - recently used (%.1fs ago)", tracked.model_key, tracked.idle_seconds)
                continue

            if force and tracked.idle_seconds < 5:
                logger.info(
                    "Force evicting %s (idle %.1fs) - explicit allocation request",
                    tracked.model_key,
                    tracked.idle_seconds,
                )

            # Try CPU offload first if enabled, there's room, and the model type has an offload callback
            # (LLM models with bitsandbytes quantization don't support CPU offload in Phase 1)
            offload_cb = self._callbacks.get(tracked.model_type, {}).get("offload")
            if (
                offload_cb is not None
                and self._enable_cpu_offload
                and tracked.location == ModelLocation.GPU
                and self._can_offload_to_cpu(tracked.memory_mb)
            ):
                success = await self._offload_model(tracked)
                if success:
                    freed_mb += tracked.memory_mb
                    self._total_offloads += 1
                    evictions_occurred = True
                    continue

            # Fall back to full unload
            if tracked.location == ModelLocation.GPU:
                success = await self._unload_model(tracked)
                if success:
                    freed_mb += tracked.memory_mb
                    self._total_unloads += 1
                    evictions_occurred = True

        # Clear CUDA cache after evictions to reclaim fragmented memory
        if evictions_occurred:
            self._clear_cuda_cache()

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

    async def _offload_model(self, tracked: TrackedModel, retries: int = 1) -> bool:
        """Offload a model to CPU via callback.

        Args:
            tracked: Model to offload
            retries: Number of retry attempts for transient failures

        Raises:
            RuntimeError: If no offload callback is registered for the model type
        """
        from vecpipe.search.metrics import eviction_latency_seconds, evictions_total

        callback = self._callbacks.get(tracked.model_type, {}).get("offload")
        if not callback:
            raise RuntimeError(
                f"No offload callback registered for {tracked.model_type.name}. "
                f"Call register_callbacks() before memory operations."
            )

        start = time.time()
        last_error: Exception | None = None
        try:
            for attempt in range(retries + 1):
                try:
                    await callback(tracked.model_name, tracked.quantization, "cpu")
                    tracked.location = ModelLocation.CPU

                    self._record_eviction(tracked, reason="memory_pressure", action=EvictionAction.OFFLOADED)

                    evictions_total.labels(
                        model_type=tracked.model_type.name.lower(),
                        action="offload",
                    ).inc()

                    logger.info(
                        "Offloaded %s to CPU (freed %dMB GPU, idle %.1fs)",
                        tracked.model_key,
                        tracked.memory_mb,
                        tracked.idle_seconds,
                    )
                    return True
                except Exception as e:
                    last_error = e
                    if attempt < retries:
                        logger.warning(
                            "Offload attempt %d/%d failed for %s: %s, retrying...",
                            attempt + 1,
                            retries + 1,
                            tracked.model_key,
                            e,
                        )
                        await asyncio.sleep(0.1)  # Short delay before retry

            logger.error("Failed to offload %s after %d attempts: %s", tracked.model_key, retries + 1, last_error)
            return False
        finally:
            eviction_latency_seconds.labels(action="offload").observe(time.time() - start)

    async def _unload_model(self, tracked: TrackedModel, retries: int = 1) -> bool:
        """Fully unload a model via callback.

        Args:
            tracked: Model to unload
            retries: Number of retry attempts for transient failures

        Raises:
            RuntimeError: If no unload callback is registered for the model type
        """
        from vecpipe.search.metrics import eviction_latency_seconds, evictions_total

        callback = self._callbacks.get(tracked.model_type, {}).get("unload")
        if not callback:
            raise RuntimeError(
                f"No unload callback registered for {tracked.model_type.name}. "
                f"Call register_callbacks() before memory operations."
            )

        start = time.time()
        last_error: Exception | None = None
        try:
            for attempt in range(retries + 1):
                try:
                    await callback(tracked.model_name, tracked.quantization)

                    self._record_eviction(tracked, reason="memory_pressure", action=EvictionAction.UNLOADED)

                    evictions_total.labels(
                        model_type=tracked.model_type.name.lower(),
                        action="unload",
                    ).inc()

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
                    last_error = e
                    if attempt < retries:
                        logger.warning(
                            "Unload attempt %d/%d failed for %s: %s, retrying...",
                            attempt + 1,
                            retries + 1,
                            tracked.model_key,
                            e,
                        )
                        await asyncio.sleep(0.1)  # Short delay before retry

            logger.error("Failed to unload %s after %d attempts: %s", tracked.model_key, retries + 1, last_error)
            return False
        finally:
            eviction_latency_seconds.labels(action="unload").observe(time.time() - start)

    async def _restore_from_cpu(self, model_key: str, required_mb: int) -> bool:
        """Restore an offloaded model from CPU to GPU.

        Args:
            model_key: Key of the model to restore
            required_mb: Memory required for the model

        Returns:
            True if restored successfully, False with logged error details otherwise

        Raises:
            RuntimeError: If no offload callback is registered for the model type
        """
        from vecpipe.search.metrics import restore_latency_seconds

        start = time.time()
        try:
            tracked = self._models.get(model_key)
            if not tracked:
                logger.warning("Cannot restore %s: model not found in tracking", model_key)
                return False
            if tracked.location != ModelLocation.CPU:
                logger.warning("Cannot restore %s: model is on %s, not CPU", model_key, tracked.location.name)
                return False

            # Ensure we have room on GPU
            # Check both tracked budget AND actual GPU memory
            current_gpu_usage = self._get_gpu_usage()
            actual_free_mb = self._probe_free_gpu_mb(self._select_probe_mode())

            # Calculate needed memory from both perspectives
            needed_from_tracked = max(0, (current_gpu_usage + required_mb) - self._budget.usable_gpu_mb)
            needed_from_actual = max(0, required_mb - actual_free_mb)
            needed = max(needed_from_tracked, needed_from_actual)

            if needed > 0:
                # Use force=True since this is an explicit restore request
                freed = await self._make_room(needed, exclude_key=model_key, force=True)
                if freed < needed:
                    # Re-probe for accurate post-eviction diagnostics
                    post_gpu_usage = self._get_gpu_usage()
                    post_free_mb = self._probe_free_gpu_mb(ProbeMode.SAFE)
                    logger.warning(
                        "Cannot restore %s: insufficient GPU memory after eviction "
                        "(needed=%dMB, freed=%dMB, post_usage=%dMB, post_free=%dMB, usable=%dMB)",
                        model_key,
                        needed,
                        freed,
                        post_gpu_usage,
                        post_free_mb,
                        self._budget.usable_gpu_mb,
                    )
                    return False

            # Restore via callback
            callback = self._callbacks.get(tracked.model_type, {}).get("offload")
            if not callback:
                raise RuntimeError(
                    f"No offload callback registered for {tracked.model_type.name}. "
                    f"Call register_callbacks() before memory operations."
                )

            try:
                await callback(tracked.model_name, tracked.quantization, "cuda")
                tracked.location = ModelLocation.GPU
                tracked.last_used = time.time()
                self._models.move_to_end(model_key)
                self._total_restorations += 1

                logger.info("Restored %s from CPU to GPU", model_key)
                return True
            except Exception as e:
                logger.error("Failed to restore %s from CPU to GPU: %s (type: %s)", model_key, e, type(e).__name__)
                return False
        finally:
            restore_latency_seconds.observe(time.time() - start)

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
        """Background loop for memory pressure monitoring.

        Uses exponential backoff for circuit breaker:
        - After max_consecutive_failures, pause for base_backoff_seconds
        - Each subsequent trigger doubles the backoff (up to max_backoff_seconds)
        - Backoff resets after successful_iterations_to_reset successful cycles
        """
        from vecpipe.search.metrics import governor_degraded

        consecutive_failures = 0
        max_consecutive_failures = 5
        base_backoff_seconds = 30  # Initial backoff: 30 seconds
        max_backoff_seconds = 300  # Cap at 5 minutes
        current_backoff = base_backoff_seconds
        successful_iterations = 0
        successful_iterations_to_reset = 10  # Reset backoff after 10 successes

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

                # Reset failure counter on success
                consecutive_failures = 0
                successful_iterations += 1

                # Reset exponential backoff after sustained success
                if successful_iterations >= successful_iterations_to_reset:
                    if current_backoff > base_backoff_seconds or self._degraded_state:
                        logger.info(
                            "Memory monitor stable for %d iterations, resetting backoff from %ds to %ds%s",
                            successful_iterations,
                            current_backoff,
                            base_backoff_seconds,
                            " and clearing degraded state" if self._degraded_state else "",
                        )
                    current_backoff = base_backoff_seconds
                    self._circuit_breaker_triggers = 0
                    self._degraded_state = False
                    governor_degraded.set(0)
                    successful_iterations = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                consecutive_failures += 1
                successful_iterations = 0  # Reset success counter on failure
                logger.exception(
                    "Error in memory monitor (failure %d/%d): %s", consecutive_failures, max_consecutive_failures, e
                )

                if consecutive_failures >= max_consecutive_failures:
                    self._circuit_breaker_triggers += 1
                    logger.error(
                        "Memory monitor circuit breaker triggered after %d failures "
                        "(trigger %d/%d). Pausing for %ds before resuming (exponential backoff).",
                        consecutive_failures,
                        self._circuit_breaker_triggers,
                        self._max_circuit_breaker_triggers,
                        current_backoff,
                    )

                    # Check for degraded state after repeated triggers
                    if self._circuit_breaker_triggers >= self._max_circuit_breaker_triggers:
                        self._degraded_state = True
                        governor_degraded.set(1)
                        logger.critical(
                            "Memory monitor in DEGRADED STATE after %d circuit breaker triggers. "
                            "Manual intervention may be required. Memory pressure handling is unreliable.",
                            self._circuit_breaker_triggers,
                        )

                    await asyncio.sleep(current_backoff)
                    consecutive_failures = 0

                    # Exponential backoff for next trigger
                    current_backoff = min(current_backoff * 2, max_backoff_seconds)

    def _calculate_pressure_level(self) -> PressureLevel:
        """Calculate current memory pressure level.

        Uses MAX of tracked and actual GPU usage to be conservative.
        This catches cases where CUDA cache holds fragmented memory
        that isn't tracked by the governor (e.g., after LLM unload).
        """
        tracked_percent = self._get_usage_percent()
        actual_percent = self._get_actual_usage_percent()
        usage_percent = max(tracked_percent, actual_percent)

        if usage_percent >= 90:
            return PressureLevel.CRITICAL
        if usage_percent >= 80:
            return PressureLevel.HIGH
        if usage_percent >= 60:
            return PressureLevel.MODERATE
        return PressureLevel.LOW

    async def _handle_critical_pressure(self) -> None:
        """Handle critical memory pressure - force unload all idle models.

        Continues processing remaining models even if individual unloads fail.
        Clears CUDA cache after evictions to reclaim fragmented memory.
        """
        from vecpipe.search.metrics import models_evicted_per_event, pressure_events_total

        pressure_events_total.labels(level="critical").inc()
        evicted_count = 0
        async with self._lock:
            failed_models: list[str] = []
            for tracked in list(self._models.values()):
                if tracked.location == ModelLocation.GPU and tracked.idle_seconds > 5:
                    try:
                        result = await self._unload_model(tracked)
                        if result:
                            evicted_count += 1
                        else:
                            failed_models.append(tracked.model_key)
                            logger.warning(
                                "Failed to unload %s during critical pressure (callback returned False)",
                                tracked.model_key,
                            )
                    except Exception as e:
                        failed_models.append(tracked.model_key)
                        logger.error("Failed to unload %s during critical pressure: %s", tracked.model_key, e)
            gc.collect()
            # Clear CUDA cache after evictions to reclaim fragmented memory
            if evicted_count > 0:
                self._clear_cuda_cache()
            if failed_models:
                logger.warning(
                    "Critical pressure handler completed with %d failures: %s", len(failed_models), failed_models
                )
        models_evicted_per_event.labels(level="critical").observe(evicted_count)

    async def _handle_high_pressure(self) -> None:
        """Handle high memory pressure - aggressive offloading.

        Continues processing remaining models even if individual operations fail.
        Clears CUDA cache after evictions to reclaim fragmented memory.
        """
        from vecpipe.search.metrics import models_evicted_per_event, pressure_events_total

        pressure_events_total.labels(level="high").inc()
        evicted_count = 0
        async with self._lock:
            failed_models: list[str] = []
            for tracked in list(self._models.values()):
                if tracked.location == ModelLocation.GPU and tracked.idle_seconds > 30:
                    try:
                        # Check if offload callback exists before attempting offload
                        offload_cb = self._callbacks.get(tracked.model_type, {}).get("offload")
                        if offload_cb and self._enable_cpu_offload and self._can_offload_to_cpu(tracked.memory_mb):
                            result = await self._offload_model(tracked)
                        else:
                            result = await self._unload_model(tracked)
                        if result:
                            evicted_count += 1
                        else:
                            failed_models.append(tracked.model_key)
                            logger.warning(
                                "Failed to evict %s during high pressure (callback returned False)",
                                tracked.model_key,
                            )
                    except Exception as e:
                        failed_models.append(tracked.model_key)
                        logger.error("Failed to evict %s during high pressure: %s", tracked.model_key, e)
            # Clear CUDA cache after evictions to reclaim fragmented memory
            if evicted_count > 0:
                self._clear_cuda_cache()
            if failed_models:
                logger.warning(
                    "High pressure handler completed with %d failures: %s", len(failed_models), failed_models
                )
        models_evicted_per_event.labels(level="high").observe(evicted_count)

    async def _handle_moderate_pressure(self) -> None:
        """Handle moderate pressure - preemptive offloading of idle models.

        Continues processing remaining models even if individual offloads fail.
        Clears CUDA cache after evictions to reclaim fragmented memory.
        """
        from vecpipe.search.metrics import models_evicted_per_event, pressure_events_total

        pressure_events_total.labels(level="moderate").inc()
        evicted_count = 0
        async with self._lock:
            failed_models: list[str] = []
            for tracked in list(self._models.values()):
                is_idle_on_gpu = (
                    tracked.location == ModelLocation.GPU and tracked.idle_seconds >= self._eviction_idle_threshold
                )
                # Check if offload callback exists before attempting offload
                offload_cb = self._callbacks.get(tracked.model_type, {}).get("offload")
                can_offload = offload_cb and self._enable_cpu_offload and self._can_offload_to_cpu(tracked.memory_mb)
                if is_idle_on_gpu and can_offload:
                    try:
                        result = await self._offload_model(tracked)
                        if result:
                            evicted_count += 1
                        else:
                            failed_models.append(tracked.model_key)
                            logger.warning(
                                "Failed to offload %s during moderate pressure (callback returned False)",
                                tracked.model_key,
                            )
                    except Exception as e:
                        failed_models.append(tracked.model_key)
                        logger.error("Failed to offload %s during moderate pressure: %s", tracked.model_key, e)
            # Clear CUDA cache after evictions to reclaim fragmented memory
            if evicted_count > 0:
                self._clear_cuda_cache()
            if failed_models:
                logger.warning(
                    "Moderate pressure handler completed with %d failures: %s", len(failed_models), failed_models
                )
        models_evicted_per_event.labels(level="moderate").observe(evicted_count)

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
            "degraded_state": self._degraded_state,
            "circuit_breaker_triggers": self._circuit_breaker_triggers,
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
        return [r.to_dict() for r in self._eviction_history]

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _make_key(self, model_name: str, model_type: ModelType, quantization: str) -> str:
        """Create unique model key."""
        return f"{model_type.name.lower()}:{model_name}:{quantization}"

    def _touch_model(self, model_key: str) -> None:
        """Update model's last_used and move to end of LRU."""
        try:
            tracked = self._models[model_key]
            tracked.last_used = time.time()
            tracked.use_count += 1
            self._models.move_to_end(model_key)
        except KeyError:
            # Model was removed between check and access - this is fine
            logger.debug("Model %s no longer tracked during touch", model_key)

    def _get_gpu_usage(self) -> int:
        """Get current GPU memory usage (tracked models only)."""
        return sum(m.memory_mb for m in self._models.values() if m.location == ModelLocation.GPU)

    def _get_cpu_usage(self) -> int:
        """Get current CPU memory usage for offloaded models."""
        return sum(m.memory_mb for m in self._models.values() if m.location == ModelLocation.CPU)

    def _probe_free_gpu_mb(self, mode: ProbeMode | None = None) -> int:
        """Probe actual free GPU memory with configurable accuracy/speed tradeoff.

        Args:
            mode: Override probe mode. If None, uses configured default.

        Probe modes:
            FAST: torch.cuda.mem_get_info() only - fastest, may be slightly stale
            SAFE: synchronize() then mem_get_info() - accurate for async ops
            AGGRESSIVE: gc + sync + empty_cache + mem_get_info - most accurate, slowest

        Returns:
            Actual free GPU memory in MB, or usable_gpu_mb if CUDA unavailable
        """
        from vecpipe.search.metrics import gpu_free_probe_latency

        effective_mode = mode or self._probe_mode
        start_time = time.time()
        try:
            import torch

            if not torch.cuda.is_available():
                # Fall back to budget-based estimate
                return self._budget.usable_gpu_mb - self._get_gpu_usage()

            if effective_mode == ProbeMode.AGGRESSIVE:
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            elif effective_mode == ProbeMode.SAFE:
                torch.cuda.synchronize()
            # FAST mode: just call mem_get_info directly

            free_bytes, _ = torch.cuda.mem_get_info()
            return free_bytes // (1024 * 1024)
        except Exception as e:
            logger.warning("Failed to probe GPU memory: %s, using tracked estimate", e)
            return self._budget.usable_gpu_mb - self._get_gpu_usage()
        finally:
            gpu_free_probe_latency.observe(time.time() - start_time)

    def _get_actual_free_gpu_mb(self) -> int:
        """Compatibility wrapper - uses configured probe mode."""
        return self._probe_free_gpu_mb()

    def _select_probe_mode(self, force_accurate: bool = False) -> ProbeMode:
        """Select appropriate probe mode based on current state.

        Args:
            force_accurate: If True, always return AGGRESSIVE mode.

        Returns:
            ProbeMode to use for the next probe call.
        """
        if force_accurate:
            return ProbeMode.AGGRESSIVE
        usage_percent = self._get_usage_percent() / 100.0
        if usage_percent > self._probe_safe_threshold:
            return ProbeMode.SAFE
        return self._probe_mode

    def _clear_cuda_cache(self) -> None:
        """Clear CUDA cache to reclaim fragmented memory.

        CRITICAL: torch.cuda.synchronize() must be called before empty_cache()
        because CUDA operations are asynchronous. Without sync, memory checks
        and cache clearing may see inconsistent state.

        This should be called after evicting models to ensure PyTorch's CUDA
        allocator releases fragmented memory back to the system.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return
            torch.cuda.synchronize()  # CRITICAL: sync before empty_cache
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
        except Exception as e:
            logger.warning("Failed to clear CUDA cache: %s", e)

    def _get_usage_percent(self) -> float:
        """Get GPU usage as percentage of usable budget."""
        gpu_usage = self._get_gpu_usage()
        return (gpu_usage / self._budget.usable_gpu_mb * 100) if self._budget.usable_gpu_mb > 0 else 0

    def _get_actual_usage_percent(self) -> float:
        """Get actual GPU usage from CUDA (not tracked models).

        This provides ground truth for GPU memory usage, catching cases where
        CUDA cache holds fragmented memory that isn't tracked by the governor.
        """
        if self._budget.total_gpu_mb == 0:
            return 0.0
        try:
            import torch

            if not torch.cuda.is_available():
                return 0.0
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_bytes = total_bytes - free_bytes
            return (used_bytes / total_bytes * 100) if total_bytes > 0 else 0.0
        except Exception:
            return 0.0

    def _get_model_memory(self, model_name: str, quantization: str) -> int:
        """Get memory requirement for a model.

        Raises:
            RuntimeError: If memory_utils cannot be imported (installation issue)
        """
        try:
            from .memory_utils import get_model_memory_requirement

            return get_model_memory_requirement(model_name, quantization)
        except ImportError as e:
            # Don't use fallback - unknown model sizes can cause OOM
            raise RuntimeError(
                f"Cannot determine memory for model {model_name}:{quantization}. "
                f"memory_utils import failed: {e}. "
                f"Check that vecpipe.memory_utils is properly installed."
            ) from e

    def _record_eviction(self, tracked: TrackedModel, reason: str, action: EvictionAction) -> None:
        """Record an eviction event."""
        record = EvictionRecord(
            model_name=tracked.model_name,
            model_type=tracked.model_type,
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
    "ProbeMode",
    "TrackedModel",
    "EvictionRecord",
    "get_memory_governor",
    "initialize_memory_governor",
]
