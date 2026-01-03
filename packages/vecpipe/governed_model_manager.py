"""
Governed Model Manager - ModelManager with integrated memory governance.

This module wraps the standard ModelManager with GPU Memory Governor
capabilities for sophisticated, dynamic memory management.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from shared.config import settings

from .cpu_offloader import ModelOffloader, get_offloader
from .memory_governor import (
    GPUMemoryGovernor,
    MemoryBudget,
    ModelLocation,
    ModelType,
    get_memory_governor,
    initialize_memory_governor,
)
from .memory_utils import get_model_memory_requirement
from .model_manager import ModelManager

if TYPE_CHECKING:
    from shared.embedding.plugin_base import BaseEmbeddingPlugin

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
            self._update_last_used()
            return self._provider

        # Call parent implementation to actually load
        try:
            provider = await super()._ensure_provider_initialized(model_name, quantization)

            # Mark as loaded in governor
            await self._governor.mark_loaded(
                model_name=model_name,
                model_type=ModelType.EMBEDDING,
                quantization=quantization,
                model_ref=getattr(provider, "model", None),
            )

            return provider

        except Exception as e:
            # Remove from governor tracking on failure
            await self._governor.mark_unloaded(model_name, ModelType.EMBEDDING, quantization)
            raise

    async def unload_model_async(self) -> None:
        """Unload model with governor tracking."""
        if self.current_model_key:
            # Parse model key
            parts = self.current_model_key.rsplit("_", 1)
            if len(parts) == 2:
                model_name, quantization = parts
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
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a future for the async operation
                future = asyncio.run_coroutine_threadsafe(
                    self._governor.request_memory(
                        model_name=model_name,
                        model_type=ModelType.RERANKER,
                        quantization=quantization,
                        required_mb=required_mb,
                    ),
                    loop,
                )
                can_allocate = future.result(timeout=30)
            else:
                can_allocate = loop.run_until_complete(
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
                self._update_last_reranker_used()
                return True

            # Call parent to actually load
            try:
                result = super().ensure_reranker_loaded(model_name, quantization)

                if result:
                    # Mark as loaded (sync workaround)
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self._governor.mark_loaded(
                                model_name, ModelType.RERANKER, quantization,
                                model_ref=self.reranker.model if self.reranker else None,
                            ),
                            loop,
                        )

                return result

            except Exception:
                # Remove from governor tracking on failure
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._governor.mark_unloaded(
                            model_name, ModelType.RERANKER, quantization
                        ),
                        loop,
                    )
                raise

    def unload_reranker(self) -> None:
        """Unload reranker with governor tracking."""
        if self.current_reranker_key:
            parts = self.current_reranker_key.rsplit("_", 1)
            if len(parts) == 2:
                model_name, quantization = parts
                # Sync workaround
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self._governor.mark_unloaded(
                                model_name, ModelType.RERANKER, quantization
                            ),
                            loop,
                        )
                except Exception:
                    pass

        super().unload_reranker()

    async def _governor_unload_embedding(
        self,
        model_name: str,
        quantization: str,
    ) -> None:
        """Callback for governor to unload embedding model."""
        # Check if this is the currently loaded model
        expected_key = self._get_model_key(model_name, quantization)
        if self.current_model_key == expected_key:
            await self.unload_model_async()

    async def _governor_unload_reranker(
        self,
        model_name: str,
        quantization: str,
    ) -> None:
        """Callback for governor to unload reranker."""
        expected_key = self._get_model_key(model_name, quantization)
        if self.current_reranker_key == expected_key:
            self.unload_reranker()

    async def _governor_offload_embedding(
        self,
        model_name: str,
        quantization: str,
        target_device: str,
    ) -> None:
        """Callback for governor to offload/restore embedding model."""
        if target_device == "cpu" and self._provider is not None:
            model = getattr(self._provider, "model", None)
            if model is not None:
                model_key = f"embedding:{model_name}:{quantization}"
                self._offloader.offload_to_cpu(model_key, model)

        elif target_device == "cuda" and self._provider is not None:
            model_key = f"embedding:{model_name}:{quantization}"
            if self._offloader.is_offloaded(model_key):
                self._offloader.restore_to_gpu(model_key)

    async def _governor_offload_reranker(
        self,
        model_name: str,
        quantization: str,
        target_device: str,
    ) -> None:
        """Callback for governor to offload/restore reranker."""
        if target_device == "cpu" and self.reranker is not None:
            model = getattr(self.reranker, "model", None)
            if model is not None:
                model_key = f"reranker:{model_name}:{quantization}"
                self._offloader.offload_to_cpu(model_key, model)

        elif target_device == "cuda" and self.reranker is not None:
            model_key = f"reranker:{model_name}:{quantization}"
            if self._offloader.is_offloaded(model_key):
                self._offloader.restore_to_gpu(model_key)

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
    ) -> dict[str, bool]:
        """
        Preload models for expected requests.

        Args:
            models: List of (model_name, "embedding"|"reranker", quantization)

        Returns:
            Dict of model_key -> success status
        """
        results = {}

        for model_name, model_type_str, quantization in models:
            model_type = (
                ModelType.EMBEDDING
                if model_type_str == "embedding"
                else ModelType.RERANKER
            )

            required_mb = get_model_memory_requirement(model_name, quantization)

            success = await self._governor.request_memory(
                model_name, model_type, quantization, required_mb
            )

            model_key = f"{model_type_str}:{model_name}:{quantization}"
            results[model_key] = success

            if success and model_type == ModelType.EMBEDDING:
                # Actually load the model
                try:
                    await self._ensure_provider_initialized(model_name, quantization)
                except Exception as e:
                    logger.error("Failed to preload %s: %s", model_key, e)
                    results[model_key] = False

        return results

    def shutdown(self) -> None:
        """Shutdown with governor cleanup."""
        # Run governor shutdown in event loop if possible
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._governor.shutdown(),
                    loop,
                ).result(timeout=10)
            else:
                loop.run_until_complete(self._governor.shutdown())
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
