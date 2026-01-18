"""
CPU Offloading for PyTorch models.

Allows moving model weights to CPU RAM when GPU memory is needed,
and restoring them when the model is needed again.
"""

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class OffloadMetadata:
    """Typed metadata for an offloaded model.

    Note: This holds a strong reference to the model to keep it alive
    while offloaded. Use ModelOffloader.discard() to explicitly release
    models that are no longer needed.
    """

    original_device: str
    offload_time: float
    model_ref: nn.Module
    keep_on_gpu: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate metadata fields."""
        if not self.original_device:
            raise ValueError("original_device cannot be empty")
        if self.offload_time <= 0:
            raise ValueError(f"offload_time must be positive, got {self.offload_time}")

    @property
    def seconds_offloaded(self) -> float:
        """Time in seconds since model was offloaded."""
        return time.time() - self.offload_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "original_device": self.original_device,
            "offload_time": self.offload_time,
            "seconds_offloaded": self.seconds_offloaded,
        }


class ModelOffloader:
    """
    Handles CPU <-> GPU offloading of PyTorch models.

    This preserves model state (weights) while freeing GPU memory,
    allowing faster "warm" restoration compared to full model reload.
    """

    def __init__(self, pin_memory: bool = True):
        """
        Args:
            pin_memory: Whether to use pinned memory for faster transfers
        """
        self.pin_memory = pin_memory
        self._offloaded_models: dict[str, OffloadMetadata] = {}

    def offload_to_cpu(
        self,
        model_key: str,
        model: nn.Module,
        keep_on_gpu: list[str] | None = None,
    ) -> OffloadMetadata:
        """
        Move model weights to CPU.

        Args:
            model_key: Unique identifier for the model
            model: PyTorch model to offload
            keep_on_gpu: Reserved for future use - partial offloading not yet implemented

        Returns:
            Offload metadata for restoration
        """
        if keep_on_gpu:
            logger.warning("keep_on_gpu parameter is not yet implemented - entire model will be offloaded to CPU")
        keep_on_gpu = keep_on_gpu or []
        start_time = time.time()

        # Track original device
        try:
            original_device = next(model.parameters()).device
        except StopIteration:
            logger.warning("Model %s has no parameters, using 'cpu' as original device", model_key)
            original_device = torch.device("cpu")

        # Move to CPU
        try:
            model.to("cpu")
        except Exception as e:
            logger.error(
                "Failed to move model %s to CPU: %s (type: %s). Model remains on %s.",
                model_key,
                e,
                type(e).__name__,
                original_device,
            )
            raise RuntimeError(f"Failed to offload {model_key} to CPU: {e}") from e

        # Optional: pin memory for faster transfers back
        if self.pin_memory and torch.cuda.is_available():
            for param in model.parameters():
                if param.data.is_pinned():
                    continue
                try:
                    param.data = param.data.pin_memory()
                except RuntimeError as e:
                    # Some tensors can't be pinned (e.g., sparse tensors, certain dtypes)
                    # Log at INFO so it's visible in production - may affect restore speed
                    logger.info("Could not pin tensor for faster restore: %s", e)

        # Synchronize CUDA and clear GPU cache
        # CRITICAL: CUDA operations are asynchronous. We must synchronize before
        # empty_cache() to ensure all tensors are actually moved to CPU before
        # we try to release GPU memory. Without this, empty_cache() may not
        # release all memory because transfers are still in progress.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        # Store typed metadata
        metadata = OffloadMetadata(
            original_device=str(original_device),
            offload_time=time.time(),
            model_ref=model,
            keep_on_gpu=keep_on_gpu,
        )
        self._offloaded_models[model_key] = metadata

        elapsed = time.time() - start_time
        logger.info("Offloaded model %s to CPU in %.2fs", model_key, elapsed)

        return metadata

    def restore_to_gpu(
        self,
        model_key: str,
        device: str | torch.device = "cuda",
    ) -> nn.Module:
        """
        Restore model from CPU to GPU.

        Args:
            model_key: Model identifier
            device: Target GPU device

        Returns:
            Restored model

        Raises:
            KeyError: If model_key not found in offloaded models
            RuntimeError: If GPU transfer fails (model remains on CPU in offloaded state)
        """
        if model_key not in self._offloaded_models:
            raise KeyError(
                f"Model {model_key} not found in offloaded models. Available: {list(self._offloaded_models.keys())}"
            )

        metadata = self._offloaded_models[model_key]
        model = metadata.model_ref
        start_time = time.time()

        try:
            # Move back to GPU
            model.to(device)
        except Exception as e:
            # Keep model in offloaded state on failure - don't remove from tracking
            logger.error("Failed to restore model %s to %s: %s. Model remains offloaded on CPU.", model_key, device, e)
            raise RuntimeError(f"GPU restore failed for {model_key}: {e}") from e

        # Only clean up metadata after successful transfer
        del self._offloaded_models[model_key]

        elapsed = time.time() - start_time
        logger.info("Restored model %s to %s in %.2fs", model_key, device, elapsed)

        return model

    def is_offloaded(self, model_key: str) -> bool:
        """Check if a model is currently offloaded."""
        return model_key in self._offloaded_models

    def get_offloaded_models(self) -> list[str]:
        """Get list of offloaded model keys."""
        return list(self._offloaded_models.keys())

    def get_offload_info(self, model_key: str) -> dict[str, Any] | None:
        """Get info about an offloaded model."""
        if model_key not in self._offloaded_models:
            return None

        return self._offloaded_models[model_key].to_dict()

    def discard(self, model_key: str) -> bool:
        """
        Discard an offloaded model without restoring it.

        Use this to release memory for models that are no longer needed.
        This prevents memory leaks when models are offloaded but never restored.

        Args:
            model_key: Model identifier

        Returns:
            True if model was discarded, False if not found
        """
        if model_key in self._offloaded_models:
            del self._offloaded_models[model_key]
            gc.collect()
            logger.info("Discarded offloaded model %s", model_key)
            return True
        return False

    def clear(self) -> None:
        """Clear all offloaded models (for shutdown)."""
        count = len(self._offloaded_models)
        self._offloaded_models.clear()
        gc.collect()
        if count > 0:
            logger.info("Cleared %d offloaded models", count)


def get_cuda_memory_fragmentation() -> dict[str, Any]:
    """
    Analyze CUDA memory fragmentation.

    High fragmentation can cause OOM even when total free memory seems sufficient.
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    stats = torch.cuda.memory_stats()

    allocated = stats.get("allocated_bytes.all.current", 0)
    reserved = stats.get("reserved_bytes.all.current", 0)

    # Fragmentation = reserved but not allocated
    fragmentation_bytes = reserved - allocated
    fragmentation_mb = fragmentation_bytes // (1024 * 1024)

    return {
        "allocated_mb": allocated // (1024 * 1024),
        "reserved_mb": reserved // (1024 * 1024),
        "fragmentation_mb": fragmentation_mb,
        "fragmentation_percent": ((fragmentation_bytes / reserved * 100) if reserved > 0 else 0),
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
        "num_ooms": stats.get("num_ooms", 0),
    }


def defragment_cuda_memory() -> None:
    """
    Attempt to defragment CUDA memory.

    This is a best-effort operation that may help with fragmentation.
    """
    if not torch.cuda.is_available():
        return

    gc.collect()
    # Synchronize before empty_cache to ensure all async operations complete
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Reset peak memory stats for better tracking
    torch.cuda.reset_peak_memory_stats()

    logger.info("CUDA memory defragmentation attempted")


# Singleton offloader instance with thread-safe initialization
_offloader: ModelOffloader | None = None
_offloader_lock = threading.Lock()


def get_offloader() -> ModelOffloader:
    """Get the singleton offloader instance (thread-safe)."""
    global _offloader
    if _offloader is not None:
        return _offloader
    with _offloader_lock:
        # Double-checked locking pattern
        if _offloader is None:
            _offloader = ModelOffloader()
        return _offloader
