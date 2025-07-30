"""
Adaptive batch size management for embedding operations.

This module provides an AdaptiveBatchSizeManager that dynamically adjusts batch sizes
based on available GPU memory and model requirements to optimize throughput while
preventing out-of-memory errors.
"""

import logging
import threading
from typing import Dict, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

from .models import get_model_config

logger = logging.getLogger(__name__)


class AdaptiveBatchSizeManager:
    """
    Manages batch sizes adaptively based on GPU memory and model characteristics.
    
    This class tracks optimal batch sizes for different model/quantization combinations
    and provides safe initial estimates based on available GPU memory.
    
    Attributes:
        _batch_sizes: Dictionary storing batch sizes per model/quantization combo
        _lock: Thread lock for thread-safe operations
        _default_safety_margin: Default safety margin for memory calculations
    """
    
    def __init__(self, default_safety_margin: float = 0.2):
        """
        Initialize the AdaptiveBatchSizeManager.
        
        Args:
            default_safety_margin: Safety margin for memory calculations (default: 0.2 or 20%)
        """
        if not (0.0 <= default_safety_margin <= 0.5):
            raise ValueError("Safety margin must be between 0.0 and 0.5")
            
        self._batch_sizes: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._default_safety_margin = default_safety_margin
        
        logger.info(
            f"Initialized AdaptiveBatchSizeManager with safety margin: {default_safety_margin}"
        )
    
    def _get_key(self, model_name: str, quantization: str) -> str:
        """Generate a unique key for model/quantization combination."""
        return f"{model_name}:{quantization}"
    
    def _get_available_gpu_memory(self) -> Optional[int]:
        """
        Get available GPU memory in MB.
        
        Returns:
            Available GPU memory in MB, or None if no GPU is available
        """
        if torch is None or not torch.cuda.is_available():
            logger.warning("CUDA not available, cannot determine GPU memory")
            return None
            
        try:
            free_memory, total_memory = torch.cuda.mem_get_info()
            available_mb = free_memory // (1024 * 1024)
            logger.debug(f"Available GPU memory: {available_mb} MB")
            return available_mb
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            return None
    
    def calculate_initial_batch_size(
        self,
        model_name: str,
        quantization: str = "float32",
        text_length: int = 512,
        safety_margin: Optional[float] = None
    ) -> int:
        """
        Calculate a safe initial batch size based on available GPU memory.
        
        Args:
            model_name: Name of the model
            quantization: Quantization type (e.g., "float32", "float16", "int8")
            text_length: Average length of text sequences
            safety_margin: Override for default safety margin
            
        Returns:
            Recommended initial batch size
        """
        if safety_margin is None:
            safety_margin = self._default_safety_margin
            
        # Get model configuration
        model_config = get_model_config(model_name)
        if model_config is None:
            logger.warning(f"Unknown model: {model_name}, using conservative batch size")
            return 1
            
        # Get available GPU memory
        available_memory = self._get_available_gpu_memory()
        if available_memory is None:
            logger.warning("Cannot determine GPU memory, using conservative batch size")
            return 1
            
        # Get memory estimate for the model
        memory_estimates = model_config.memory_estimate or {}
        model_memory_mb = memory_estimates.get(quantization, 1000)  # Default to 1GB
        
        # Estimate memory per batch item
        # This is a rough estimate based on model size and sequence length
        sequence_factor = min(text_length / model_config.max_sequence_length, 1.0)
        memory_per_item = model_memory_mb * 0.1 * sequence_factor  # 10% of model size per item
        
        # Account for temporary tensors and gradients (if training)
        overhead_factor = 2.0  # Conservative estimate for intermediate tensors
        memory_per_item *= overhead_factor
        
        # Apply safety margin
        usable_memory = available_memory * (1 - safety_margin)
        
        # Calculate batch size
        batch_size = max(1, int(usable_memory / memory_per_item))
        
        # Cap at reasonable maximum
        max_batch_size = 256
        batch_size = min(batch_size, max_batch_size)
        
        logger.info(
            f"Calculated initial batch size for {model_name}/{quantization}: {batch_size} "
            f"(available: {available_memory}MB, per-item: {memory_per_item:.1f}MB)"
        )
        
        return batch_size
    
    def get_current_batch_size(
        self,
        model_name: str,
        quantization: str = "float32"
    ) -> Optional[int]:
        """
        Get the current batch size for a model/quantization combination.
        
        Args:
            model_name: Name of the model
            quantization: Quantization type
            
        Returns:
            Current batch size if set, None otherwise
        """
        key = self._get_key(model_name, quantization)
        with self._lock:
            return self._batch_sizes.get(key)
    
    def update_batch_size(
        self,
        model_name: str,
        quantization: str,
        new_size: int
    ) -> None:
        """
        Update the batch size for a model/quantization combination.
        
        Args:
            model_name: Name of the model
            quantization: Quantization type
            new_size: New batch size
            
        Raises:
            ValueError: If new_size is not positive
        """
        if new_size <= 0:
            raise ValueError("Batch size must be positive")
            
        key = self._get_key(model_name, quantization)
        with self._lock:
            old_size = self._batch_sizes.get(key)
            self._batch_sizes[key] = new_size
            
        logger.info(
            f"Updated batch size for {model_name}/{quantization}: "
            f"{old_size} -> {new_size}"
        )
    
    def reset_batch_size(
        self,
        model_name: str,
        quantization: str
    ) -> None:
        """
        Reset (remove) the batch size for a model/quantization combination.
        
        Args:
            model_name: Name of the model
            quantization: Quantization type
        """
        key = self._get_key(model_name, quantization)
        with self._lock:
            if key in self._batch_sizes:
                old_size = self._batch_sizes[key]
                del self._batch_sizes[key]
                logger.info(
                    f"Reset batch size for {model_name}/{quantization} "
                    f"(was: {old_size})"
                )
            else:
                logger.debug(
                    f"No batch size to reset for {model_name}/{quantization}"
                )
    
    def get_all_batch_sizes(self) -> Dict[str, int]:
        """
        Get all currently stored batch sizes.
        
        Returns:
            Dictionary of model:quantization -> batch_size mappings
        """
        with self._lock:
            return self._batch_sizes.copy()
    
    def clear_all(self) -> None:
        """Clear all stored batch sizes."""
        with self._lock:
            count = len(self._batch_sizes)
            self._batch_sizes.clear()
            
        logger.info(f"Cleared all batch sizes ({count} entries)")
    
    def get_or_calculate_batch_size(
        self,
        model_name: str,
        quantization: str = "float32",
        text_length: int = 512,
        safety_margin: Optional[float] = None
    ) -> int:
        """
        Get current batch size or calculate initial if not set.
        
        This is a convenience method that first checks for an existing batch size
        and calculates a new one if none exists.
        
        Args:
            model_name: Name of the model
            quantization: Quantization type
            text_length: Average length of text sequences
            safety_margin: Override for default safety margin
            
        Returns:
            Batch size to use
        """
        current_size = self.get_current_batch_size(model_name, quantization)
        if current_size is not None:
            return current_size
            
        # Calculate and store initial batch size
        initial_size = self.calculate_initial_batch_size(
            model_name, quantization, text_length, safety_margin
        )
        self.update_batch_size(model_name, quantization, initial_size)
        return initial_size