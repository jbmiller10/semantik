"""
Governed Model Manager - ModelManager with integrated memory governance.

This module wraps the standard ModelManager with GPU Memory Governor
capabilities for sophisticated, dynamic memory management.
"""

import asyncio
import logging
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, TypeVar

from shared.config import settings

from .cpu_offloader import get_offloader
from .memory_governor import (
    GPUMemoryGovernor,
    MemoryBudget,
    ModelType,
)
from .memory_utils import get_model_memory_requirement
from .model_manager import ModelManager

if TYPE_CHECKING:
    from shared.embedding.plugin_base import BaseEmbeddingPlugin

T = TypeVar("T")

logger = logging.getLogger(__name__)


class GovernedModelManager(ModelManager):
    """
    ModelManager with integrated GPU Memory Governor.

    Extends the standard ModelManager with:
    - Memory budget enforcement
    - LRU-based model eviction
    - CPU offloading for warm models
    - Preemptive memory pressure handling
    - Request-aware model preloading
    """

    def __init__(
        self,
        unload_after_seconds: int = 300,
        budget: MemoryBudget | None = None,
        enable_cpu_offload: bool = True,
        enable_preemptive_eviction: bool = True,
        eviction_idle_threshold_seconds: int = 120,
    ):
        # Initialize parent ModelManager
        super().__init__(unload_after_seconds=unload_after_seconds)

        # Initialize governor
        self._governor = GPUMemoryGovernor(
            budget=budget,
            enable_cpu_offload=enable_cpu_offload,
            eviction_idle_threshold_seconds=eviction_idle_threshold_seconds,
        )

        # Initialize offloader
        self._offloader = get_offloader()

        self._enable_preemptive_eviction = enable_preemptive_eviction
        self._governor_initialized = False

        # Register callbacks with governor
        self._governor.register_callbacks(
            ModelType.EMBEDDING,
            unload_fn=self._governor_unload_embedding,
            offload_fn=self._governor_offload_embedding,
        )
        self._governor.register_callbacks(
            ModelType.RERANKER,
            unload_fn=self._governor_unload_reranker,
            offload_fn=self._governor_offload_reranker,
        )

        logger.info(
            "GovernedModelManager initialized with governor "
            "(cpu_offload=%s, preemptive_eviction=%s)",
            enable_cpu_offload,
            enable_preemptive_eviction,
        )

    async def start(self) -> None:
        """Start the governed model manager."""
        if self._enable_preemptive_eviction:
            await self._governor.start_monitor()
        self._governor_initialized = True

    def _run_governor_coro(
        self, coro: Coroutine[Any, Any, T], timeout: float = 30.0
    ) -> T:
        """
        Safely run a governor coroutine from sync context.

        Handles both cases:
        - No running loop: creates new loop and runs
        - Running loop on different thread: schedules and waits
        - Running loop on same thread: runs in new thread to avoid deadlock
        """
        import concurrent.futures

        try:
            # Check if there's a running loop
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No running loop - create one and run
                return asyncio.run(coro)

            # There's a running loop - we need to be careful
            # Run in a separate thread to avoid blocking the event loop
            def run_in_thread() -> T:
                return asyncio.run(coro)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=timeout)

        except Exception as e:
            logger.error("Error running governor coroutine: %s", e)
            raise

    def _schedule_governor_coro(
        self, coro: Coroutine[Any, Any, Any], *, critical: bool = False
    ) -> None:
        """
        Schedule a governor coroutine without blocking.

        Used for non-critical operations like touch and unload tracking
        where we don't need to wait for the result.

        Args:
            coro: Coroutine to schedule
            critical: If True, log failures as ERROR (state drift risk).
                      If False, log as WARNING (acceptable for touch operations).
        """
        import concurrent.futures

        def _handle_future_error(future: concurrent.futures.Future[Any]) -> None:
            """Log errors from scheduled coroutines."""
            try:
                future.result()
            except Exception as e:
                if critical:
                    logger.error(
                        "Critical governor coroutine failed (may cause state drift): %s", e
                    )
                else:
                    logger.warning("Scheduled governor coroutine failed: %s", e)

        try:
            try:
                loop = asyncio.get_running_loop()
                # Schedule without waiting, but add callback for error handling
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                future.add_done_callback(_handle_future_error)
            except RuntimeError:
                # No running loop - run synchronously
                asyncio.run(coro)
        except Exception as e:
            if critical:
                logger.error(
                    "Failed to schedule critical governor coroutine (state drift risk): %s", e
                )
            else:
                logger.warning("Failed to schedule governor coroutine: %s", e)

    async def _ensure_provider_initialized(
        self,
        model_name: str,
        quantization: str,
    ) -> "BaseEmbeddingPlugin":
        """
        Ensure provider is initialized with governor memory management.

        Overrides parent to add:
        - Memory budget checking
        - Preemptive eviction if needed
        - Governor tracking
        """
        model_key = self._get_model_key(model_name, quantization)

        # Calculate memory requirement
        required_mb = get_model_memory_requirement(model_name, quantization)

        # Request memory from governor (handles restoration from CPU if offloaded)
        can_allocate = await self._governor.request_memory(
            model_name=model_name,
            model_type=ModelType.EMBEDDING,
            quantization=quantization,
            required_mb=required_mb,
        )

        if not can_allocate:
            raise RuntimeError(
                f"Cannot allocate memory for model {model_name} ({required_mb}MB required). "
                f"Memory stats: {self._governor.get_memory_stats()}"
            )

        # Fast path: already initialized with correct model and on GPU
        if self._provider is not None and self.current_model_key == model_key:
            # Touch to update LRU (request_memory may have already touched,
            # but explicit touch ensures LRU is always updated on inference)
            await self._governor.touch(model_name, ModelType.EMBEDDING, quantization)
            self._update_last_used()
            return self._provider

        # Call parent implementation to actually load
        try:
            provider = await super()._ensure_provider_initialized(model_name, quantization)

            # Mark as loaded in governor (also sets initial last_used)
            await self._governor.mark_loaded(
                model_name=model_name,
                model_type=ModelType.EMBEDDING,
                quantization=quantization,
                model_ref=getattr(provider, "model", None),
            )

            return provider

        except Exception:
            # Remove from governor tracking on failure
            await self._governor.mark_unloaded(model_name, ModelType.EMBEDDING, quantization)
            raise

    async def unload_model_async(self) -> None:
        """Unload model with governor tracking."""
        if self.current_model_key:
            parsed = self._parse_model_key(self.current_model_key)
            if parsed:
                model_name, quantization = parsed
                await self._governor.mark_unloaded(
                    model_name, ModelType.EMBEDDING, quantization
                )

        await super().unload_model_async()

    def ensure_reranker_loaded(self, model_name: str, quantization: str) -> bool:
        """
        Load reranker with governor memory management.

        Note: This is sync in parent, so we use sync governor operations.
        """
        if self.is_mock_mode:
            return True

        reranker_key = self._get_model_key(model_name, quantization)

        with self.reranker_lock:
            # Calculate memory requirement
            required_mb = get_model_memory_requirement(model_name, quantization)

            # For sync context, we need to run governor request in event loop
            # This handles restoration from CPU if offloaded
            can_allocate = self._run_governor_coro(
                self._governor.request_memory(
                    model_name=model_name,
                    model_type=ModelType.RERANKER,
                    quantization=quantization,
                    required_mb=required_mb,
                )
            )

            if not can_allocate:
                from .memory_utils import InsufficientMemoryError
                raise InsufficientMemoryError(
                    f"Cannot allocate memory for reranker {model_name} ({required_mb}MB required)"
                )

            # Fast path: already loaded and on GPU (after governor check for restoration)
            if self.current_reranker_key == reranker_key and self.reranker is not None:
                # Touch to update LRU (request_memory may have already touched,
                # but explicit touch ensures LRU is always updated on inference)
                self._schedule_governor_coro(
                    self._governor.touch(model_name, ModelType.RERANKER, quantization)
                )
                self._update_last_reranker_used()
                return True

            # Call parent to actually load
            try:
                result = super().ensure_reranker_loaded(model_name, quantization)

                if result:
                    # Mark as loaded in governor
                    self._run_governor_coro(
                        self._governor.mark_loaded(
                            model_name, ModelType.RERANKER, quantization,
                            model_ref=self.reranker.model if self.reranker else None,
                        )
                    )

                return result

            except Exception as e:
                # Remove from governor tracking on failure (critical - state drift risk)
                logger.warning("Failed to load reranker %s: %s", model_name, e)
                self._schedule_governor_coro(
                    self._governor.mark_unloaded(model_name, ModelType.RERANKER, quantization),
                    critical=True,
                )
                raise

    def unload_reranker(self) -> None:
        """Unload reranker with governor tracking."""
        if self.current_reranker_key:
            parsed = self._parse_model_key(self.current_reranker_key)
            if parsed:
                model_name, quantization = parsed
                # Notify governor (critical - state drift risk if this fails)
                self._schedule_governor_coro(
                    self._governor.mark_unloaded(model_name, ModelType.RERANKER, quantization),
                    critical=True,
                )

        super().unload_reranker()

    async def _governor_unload_embedding(
        self,
        model_name: str,
        quantization: str,
    ) -> None:
        """Callback for governor to unload embedding model.

        Note: Calls parent's unload directly to avoid deadlock.
        The governor already handles removal from tracking in _unload_model.
        """
        expected_key = self._get_model_key(model_name, quantization)
        if self.current_model_key == expected_key:
            # Call parent directly - governor already removed from tracking
            await super().unload_model_async()

    async def _governor_unload_reranker(
        self,
        model_name: str,
        quantization: str,
    ) -> None:
        """Callback for governor to unload reranker.

        Note: Calls parent's unload directly to avoid deadlock.
        The governor already handles removal from tracking in _unload_model.
        """
        expected_key = self._get_model_key(model_name, quantization)
        if self.current_reranker_key == expected_key:
            # Call parent directly - governor already removed from tracking
            super().unload_reranker()

    async def _governor_offload_embedding(
        self,
        model_name: str,
        quantization: str,
        target_device: str,
    ) -> None:
        """Callback for governor to offload/restore embedding model.

        Raises:
            RuntimeError: If offload/restore operation cannot be completed
        """
        model_key = f"embedding:{model_name}:{quantization}"

        if target_device == "cpu":
            if self._provider is None:
                raise RuntimeError(
                    f"Cannot offload {model_key} to CPU: provider is None "
                    "(model may have been unloaded)"
                )
            model = getattr(self._provider, "model", None)
            if model is None:
                raise RuntimeError(
                    f"Cannot offload {model_key} to CPU: provider has no model attribute"
                )
            self._offloader.offload_to_cpu(model_key, model)

        elif target_device == "cuda":
            if self._provider is None:
                raise RuntimeError(
                    f"Cannot restore {model_key} to GPU: provider is None "
                    "(model may have been unloaded)"
                )
            if self._offloader.is_offloaded(model_key):
                self._offloader.restore_to_gpu(model_key)
            else:
                logger.warning(
                    "Cannot restore %s: not found in offloaded models", model_key
                )

    async def _governor_offload_reranker(
        self,
        model_name: str,
        quantization: str,
        target_device: str,
    ) -> None:
        """Callback for governor to offload/restore reranker.

        Raises:
            RuntimeError: If offload/restore operation cannot be completed
        """
        model_key = f"reranker:{model_name}:{quantization}"

        if target_device == "cpu":
            if self.reranker is None:
                raise RuntimeError(
                    f"Cannot offload {model_key} to CPU: reranker is None "
                    "(model may have been unloaded)"
                )
            model = getattr(self.reranker, "model", None)
            if model is None:
                raise RuntimeError(
                    f"Cannot offload {model_key} to CPU: reranker has no model attribute"
                )
            self._offloader.offload_to_cpu(model_key, model)

        elif target_device == "cuda":
            if self.reranker is None:
                raise RuntimeError(
                    f"Cannot restore {model_key} to GPU: reranker is None "
                    "(model may have been unloaded)"
                )
            if self._offloader.is_offloaded(model_key):
                self._offloader.restore_to_gpu(model_key)
            else:
                logger.warning(
                    "Cannot restore %s: not found in offloaded models", model_key
                )

    def get_status(self) -> dict[str, Any]:
        """Get status including governor information."""
        status = super().get_status()

        # Add governor stats
        status["governor"] = {
            "memory_stats": self._governor.get_memory_stats(),
            "loaded_models": self._governor.get_loaded_models(),
            "eviction_history_count": len(self._governor.get_eviction_history()),
            "recent_evictions": self._governor.get_eviction_history()[-5:],
        }

        # Add offloader stats
        offloaded = self._offloader.get_offloaded_models()
        status["offloaded_models"] = [
            self._offloader.get_offload_info(key)
            for key in offloaded
        ]

        return status

    async def preload_models(
        self,
        models: list[tuple[str, str, str]],  # (name, type, quantization)
    ) -> dict[str, bool | str]:
        """
        Preload models for expected requests.

        Note: Currently only embedding models are actually loaded. Rerankers
        have memory reserved but are not actually loaded until first use.
        This is because reranker loading requires different initialization.

        Args:
            models: List of (model_name, "embedding"|"reranker", quantization)

        Returns:
            Dict of model_key -> success (True), or error message (str) on failure
        """
        results: dict[str, bool | str] = {}

        for model_name, model_type_str, quantization in models:
            model_type = (
                ModelType.EMBEDDING
                if model_type_str == "embedding"
                else ModelType.RERANKER
            )

            required_mb = get_model_memory_requirement(model_name, quantization)
            model_key = f"{model_type_str}:{model_name}:{quantization}"

            success = await self._governor.request_memory(
                model_name, model_type, quantization, required_mb
            )

            if not success:
                error_msg = (
                    f"Memory allocation failed: {required_mb}MB required, "
                    f"stats: {self._governor.get_memory_stats()}"
                )
                logger.warning("Preload failed for %s: %s", model_key, error_msg)
                results[model_key] = error_msg
                continue

            if model_type == ModelType.EMBEDDING:
                # Actually load embedding models
                try:
                    await self._ensure_provider_initialized(model_name, quantization)
                    results[model_key] = True
                except Exception as e:
                    error_msg = f"Load failed: {type(e).__name__}: {e}"
                    logger.error("Failed to preload %s: %s", model_key, e)
                    results[model_key] = error_msg
            else:
                # Rerankers: memory reserved but not loaded until first use
                logger.info(
                    "Preload %s: memory reserved (%dMB), will load on first use",
                    model_key, required_mb
                )
                results[model_key] = True

        return results

    async def shutdown_async(self) -> None:
        """Async shutdown - use this from async contexts like FastAPI lifespan."""
        try:
            await self._governor.shutdown()
        except Exception as e:
            logger.error("Error shutting down governor: %s", e)

        # Clear offloader
        self._offloader.clear()

        # Parent shutdown
        super().shutdown()

    def shutdown(self) -> None:
        """Sync shutdown - use from non-async contexts only.

        For async contexts (FastAPI lifespan, etc.), use shutdown_async() instead
        to avoid deadlocks.
        """
        try:
            # Check if there's a running loop
            try:
                loop = asyncio.get_running_loop()
                # We're inside a running loop - schedule without blocking
                asyncio.run_coroutine_threadsafe(
                    self._governor.shutdown(),
                    loop,
                )
                logger.warning(
                    "shutdown() called from running event loop - "
                    "governor shutdown scheduled but not awaited. "
                    "Use shutdown_async() in async contexts."
                )
            except RuntimeError:
                # No running loop - safe to create one
                asyncio.run(self._governor.shutdown())
        except Exception as e:
            logger.error("Error shutting down governor: %s", e)

        # Clear offloader
        self._offloader.clear()

        # Parent shutdown
        super().shutdown()


def create_governed_model_manager(
    unload_after_seconds: int | None = None,
    total_gpu_memory_mb: int | None = None,
    enable_cpu_offload: bool = True,
    enable_preemptive_eviction: bool = True,
) -> GovernedModelManager:
    """
    Factory function to create a governed model manager.

    Args:
        unload_after_seconds: Idle timeout (default from settings)
        total_gpu_memory_mb: Override GPU memory detection
        enable_cpu_offload: Enable CPU offloading
        enable_preemptive_eviction: Enable background memory monitor

    Returns:
        Configured GovernedModelManager
    """
    import torch

    # Get timeout from settings if not provided
    if unload_after_seconds is None:
        unload_after_seconds = settings.MODEL_UNLOAD_AFTER_SECONDS

    # Build memory budget
    if total_gpu_memory_mb is not None:
        budget = MemoryBudget(total_gpu_mb=total_gpu_memory_mb)
    elif torch.cuda.is_available():
        _, total_bytes = torch.cuda.mem_get_info()
        total_mb = total_bytes // (1024 * 1024)
        budget = MemoryBudget(total_gpu_mb=total_mb)
    else:
        # CPU-only mode
        budget = MemoryBudget(total_gpu_mb=0)

    return GovernedModelManager(
        unload_after_seconds=unload_after_seconds,
        budget=budget,
        enable_cpu_offload=enable_cpu_offload,
        enable_preemptive_eviction=enable_preemptive_eviction,
    )
