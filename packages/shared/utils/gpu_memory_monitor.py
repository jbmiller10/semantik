#!/usr/bin/env python3
"""
GPU memory monitoring and management utilities.

This module provides tools for monitoring GPU memory usage and implementing
intelligent memory management for chunking operations.
"""

import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class GPUMemoryMonitor:
    """Monitor and manage GPU memory usage during chunking operations."""
    
    def __init__(self):
        """Initialize GPU memory monitor."""
        self.start_memory = 0
        self.peak_memory = 0
        self.monitoring = False
        self._has_gpu = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.debug("PyTorch not available, GPU monitoring disabled")
            return False
            
    def start_monitoring(self):
        """Start monitoring GPU memory usage."""
        if not self._has_gpu:
            return
            
        try:
            import torch
            torch.cuda.empty_cache()  # Clean up before starting
            self.start_memory = torch.cuda.memory_allocated(0)
            self.peak_memory = self.start_memory
            self.monitoring = True
            logger.debug(f"Started GPU memory monitoring: {self.start_memory // (1024*1024)}MB allocated")
        except Exception as e:
            logger.warning(f"Failed to start GPU monitoring: {e}")
            self.monitoring = False
    
    def memory_usage(self) -> float:
        """Get current memory usage as percentage of total.
        
        Returns:
            Float between 0.0 and 1.0 representing memory usage percentage
        """
        if not self._has_gpu or not self.monitoring:
            return 0.0
            
        try:
            import torch
            current = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            usage = current / total
            
            # Update peak
            self.peak_memory = max(self.peak_memory, current)
            return usage
            
        except Exception as e:
            logger.debug(f"Failed to get GPU memory usage: {e}")
            return 0.0
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed GPU memory information.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self._has_gpu:
            return {
                "available": False,
                "total_mb": 0,
                "allocated_mb": 0,
                "free_mb": 0,
                "usage_percent": 0.0
            }
            
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            
            return {
                "available": True,
                "device_name": props.name,
                "total_mb": props.total_memory // (1024 * 1024),
                "allocated_mb": allocated // (1024 * 1024),
                "reserved_mb": reserved // (1024 * 1024),
                "free_mb": (props.total_memory - allocated) // (1024 * 1024),
                "usage_percent": (allocated / props.total_memory) * 100,
                "compute_capability": f"{props.major}.{props.minor}"
            }
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return {
                "available": False,
                "error": str(e)
            }
    
    def cleanup(self):
        """Clean up GPU memory after operation."""
        if not self.monitoring:
            return
            
        try:
            import torch
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated(0)
                memory_used = (self.peak_memory - self.start_memory) // (1024 * 1024)
                
                logger.info(f"GPU memory usage: {memory_used}MB peak during operation")
                
                # Clean up
                torch.cuda.empty_cache()
                
                # Check for memory leaks
                post_cleanup = torch.cuda.memory_allocated(0)
                if post_cleanup > self.start_memory * 1.1:  # 10% tolerance
                    leaked_mb = (post_cleanup - self.start_memory) // (1024 * 1024)
                    logger.warning(f"Possible GPU memory leak detected: {leaked_mb}MB not freed")
                
        except Exception as e:
            logger.error(f"GPU cleanup failed: {e}")
        finally:
            self.monitoring = False
            
    def calculate_optimal_batch_size(self, model_name: str = "all-MiniLM-L6-v2", 
                                   safety_factor: float = 0.7) -> int:
        """Calculate optimal batch size based on available GPU memory.
        
        Args:
            model_name: Name of the embedding model
            safety_factor: Fraction of free memory to use (0.0-1.0)
            
        Returns:
            Optimal batch size for current GPU memory state
        """
        if not self._has_gpu:
            return 8  # Conservative CPU batch size
            
        try:
            import torch
            
            # Get GPU memory info
            props = torch.cuda.get_device_properties(0)
            total_memory_mb = props.total_memory // (1024 * 1024)
            allocated_mb = torch.cuda.memory_allocated(0) // (1024 * 1024)
            free_mb = total_memory_mb - allocated_mb
            
            # Estimate memory per batch based on model
            model_memory_per_batch = self._estimate_model_memory(model_name)
            
            # Calculate safe batch size
            safe_batch_size = max(4, min(128, int((free_mb * safety_factor) // model_memory_per_batch)))
            
            logger.info(
                f"GPU memory: {total_memory_mb}MB total, {free_mb}MB free, "
                f"model: {model_name}, batch_size: {safe_batch_size}"
            )
            
            return safe_batch_size
            
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}, using default")
            return 16  # Safe default
            
    def _estimate_model_memory(self, model_name: str) -> int:
        """Estimate memory usage per batch for different models.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            Estimated MB per batch
        """
        # Based on empirical measurements
        if "large" in model_name.lower() or "instructor" in model_name.lower():
            return 150  # Large models like instructor-large
        elif "base" in model_name.lower():
            return 100  # Base models
        elif "mini" in model_name.lower():
            return 50   # Mini models like all-MiniLM-L6-v2
        elif "mpnet" in model_name.lower():
            return 75   # MPNet models
        else:
            return 75   # Conservative default


def test_gpu_memory():
    """Test GPU memory monitoring functionality."""
    monitor = GPUMemoryMonitor()
    
    print("GPU Memory Information:")
    info = monitor.get_memory_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    if info["available"]:
        print(f"\nOptimal batch sizes:")
        for model in ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "instructor-large"]:
            batch_size = monitor.calculate_optimal_batch_size(model)
            print(f"  {model}: {batch_size}")
    else:
        print("\nNo GPU available for testing")


if __name__ == "__main__":
    test_gpu_memory()