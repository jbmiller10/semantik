#!/usr/bin/env python3
"""
Unit tests for SemanticChunker GPU memory management.

Tests the dynamic batch sizing and GPU memory monitoring features.
"""

import os
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from packages.shared.text_processing.strategies.semantic_chunker import SemanticChunker
from packages.shared.utils.gpu_memory_monitor import GPUMemoryMonitor


class TestSemanticChunkerGPU:
    """Test GPU memory management in SemanticChunker."""

    def setup_method(self):
        """Set up test environment."""
        os.environ["TESTING"] = "true"

    def teardown_method(self):
        """Clean up test environment."""
        if "TESTING" in os.environ:
            del os.environ["TESTING"]

    @patch("packages.shared.text_processing.strategies.semantic_chunker.torch")
    def test_calculate_optimal_batch_size_with_gpu(self, mock_torch):
        """Test batch size calculation with GPU available."""
        # Mock GPU availability
        mock_torch.cuda.is_available.return_value = True
        
        # Mock GPU properties
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024 * 1024 * 1024  # 2GB allocated
        
        chunker = SemanticChunker()
        
        # Expected: (8GB - 2GB) * 0.7 / 50MB = ~84 batches
        # But capped at 128 max
        assert 80 <= chunker.embed_batch_size <= 90

    @patch("packages.shared.text_processing.strategies.semantic_chunker.torch")
    def test_calculate_optimal_batch_size_without_gpu(self, mock_torch):
        """Test batch size calculation without GPU."""
        mock_torch.cuda.is_available.return_value = False
        
        chunker = SemanticChunker()
        
        # Should use CPU default
        assert chunker.embed_batch_size == 8

    @patch("packages.shared.text_processing.strategies.semantic_chunker.torch")
    def test_calculate_optimal_batch_size_with_large_model(self, mock_torch):
        """Test batch size calculation with large model."""
        # Mock GPU availability
        mock_torch.cuda.is_available.return_value = True
        
        # Mock GPU properties
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024 * 1024 * 1024  # 16GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_allocated.return_value = 4 * 1024 * 1024 * 1024  # 4GB allocated
        
        # Set environment for large model
        os.environ["EMBEDDING_MODEL"] = "instructor-large"
        
        chunker = SemanticChunker()
        
        # Expected: (16GB - 4GB) * 0.7 / 150MB = ~56 batches
        assert 50 <= chunker.embed_batch_size <= 60
        
        # Clean up
        del os.environ["EMBEDDING_MODEL"]

    @patch("packages.shared.text_processing.strategies.semantic_chunker.torch")
    def test_calculate_optimal_batch_size_low_memory(self, mock_torch):
        """Test batch size calculation with low GPU memory."""
        # Mock GPU availability
        mock_torch.cuda.is_available.return_value = True
        
        # Mock GPU properties - low memory GPU
        mock_props = MagicMock()
        mock_props.total_memory = 2 * 1024 * 1024 * 1024  # 2GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_allocated.return_value = 1.5 * 1024 * 1024 * 1024  # 1.5GB allocated
        
        chunker = SemanticChunker()
        
        # Should use minimum batch size of 4
        assert chunker.embed_batch_size == 4

    @pytest.mark.asyncio
    @patch("packages.shared.text_processing.strategies.semantic_chunker.GPUMemoryMonitor")
    async def test_memory_pressure_handling(self, mock_gpu_monitor_class):
        """Test handling of high GPU memory pressure."""
        # Mock GPU monitor instance
        mock_monitor = MagicMock()
        mock_monitor.memory_usage.return_value = 0.9  # 90% memory usage
        mock_gpu_monitor_class.return_value = mock_monitor
        
        # Create chunker with mocked embeddings
        chunker = SemanticChunker(embed_batch_size=32)
        chunker._gpu_memory_monitor = mock_monitor
        
        # Test text
        test_text = "This is a test document for semantic chunking. " * 100
        
        # Process with high memory pressure
        chunks = await chunker.chunk_text_async(test_text, "test_doc")
        
        # Verify memory monitoring was used
        mock_monitor.start_monitoring.assert_called_once()
        mock_monitor.memory_usage.assert_called()
        mock_monitor.cleanup.assert_called_once()
        
        # Should produce chunks despite memory pressure
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_gpu_memory_cleanup_on_error(self):
        """Test that GPU memory is cleaned up even on error."""
        with patch("packages.shared.text_processing.strategies.semantic_chunker.GPUMemoryMonitor") as mock_monitor_class:
            mock_monitor = MagicMock()
            mock_monitor_class.return_value = mock_monitor
            
            chunker = SemanticChunker()
            chunker._gpu_memory_monitor = mock_monitor
            
            # Force an error in chunking
            with patch.object(chunker, '_chunk_with_retry', side_effect=Exception("Test error")):
                try:
                    await chunker.chunk_text_async("test text", "test_doc")
                except:
                    pass
            
            # Cleanup should still be called
            mock_monitor.cleanup.assert_called_once()

    def test_batch_size_override(self):
        """Test that explicit batch size overrides automatic calculation."""
        chunker = SemanticChunker(embed_batch_size=64)
        assert chunker.embed_batch_size == 64

    @patch("packages.shared.text_processing.strategies.semantic_chunker.torch")
    def test_batch_size_calculation_error_handling(self, mock_torch):
        """Test graceful handling of errors during batch size calculation."""
        # Mock torch to raise an exception
        mock_torch.cuda.is_available.side_effect = Exception("CUDA error")
        
        chunker = SemanticChunker()
        
        # Should fall back to default
        assert chunker.embed_batch_size == 16


class TestGPUMemoryMonitor:
    """Test GPU memory monitor functionality."""

    @patch("packages.shared.utils.gpu_memory_monitor.torch")
    def test_memory_info_with_gpu(self, mock_torch):
        """Test getting memory info with GPU available."""
        mock_torch.cuda.is_available.return_value = True
        
        # Mock GPU properties
        mock_props = MagicMock()
        mock_props.name = "NVIDIA GeForce RTX 3090"
        mock_props.total_memory = 24 * 1024 * 1024 * 1024  # 24GB
        mock_props.major = 8
        mock_props.minor = 6
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_allocated.return_value = 4 * 1024 * 1024 * 1024  # 4GB
        mock_torch.cuda.memory_reserved.return_value = 6 * 1024 * 1024 * 1024  # 6GB
        
        monitor = GPUMemoryMonitor()
        info = monitor.get_memory_info()
        
        assert info["available"] is True
        assert info["device_name"] == "NVIDIA GeForce RTX 3090"
        assert info["total_mb"] == 24 * 1024
        assert info["allocated_mb"] == 4 * 1024
        assert info["reserved_mb"] == 6 * 1024
        assert info["compute_capability"] == "8.6"

    @patch("packages.shared.utils.gpu_memory_monitor.torch")
    def test_memory_info_without_gpu(self, mock_torch):
        """Test getting memory info without GPU."""
        mock_torch.cuda.is_available.return_value = False
        
        monitor = GPUMemoryMonitor()
        info = monitor.get_memory_info()
        
        assert info["available"] is False
        assert info["total_mb"] == 0

    @patch("packages.shared.utils.gpu_memory_monitor.torch")
    def test_memory_leak_detection(self, mock_torch):
        """Test memory leak detection."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.side_effect = [
            1 * 1024 * 1024 * 1024,  # Start: 1GB
            3 * 1024 * 1024 * 1024,  # Peak: 3GB
            1.2 * 1024 * 1024 * 1024,  # After cleanup: 1.2GB (leak!)
        ]
        
        monitor = GPUMemoryMonitor()
        monitor.start_monitoring()
        
        # Simulate some work
        monitor.memory_usage()
        
        # Cleanup should detect leak
        with patch("packages.shared.utils.gpu_memory_monitor.logger") as mock_logger:
            monitor.cleanup()
            
            # Should log warning about memory leak
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "memory leak" in warning_call.lower()

    @patch("packages.shared.utils.gpu_memory_monitor.torch")
    def test_optimal_batch_size_calculation(self, mock_torch):
        """Test optimal batch size calculation in monitor."""
        mock_torch.cuda.is_available.return_value = True
        
        # Mock GPU properties
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024 * 1024 * 1024  # 2GB allocated
        
        monitor = GPUMemoryMonitor()
        
        # Test different models
        batch_size_mini = monitor.calculate_optimal_batch_size("all-MiniLM-L6-v2")
        batch_size_base = monitor.calculate_optimal_batch_size("all-mpnet-base-v2")
        batch_size_large = monitor.calculate_optimal_batch_size("instructor-large")
        
        # Large models should have smaller batch sizes
        assert batch_size_large < batch_size_base < batch_size_mini
        
        # All should be within reasonable bounds
        assert 4 <= batch_size_large <= 128
        assert 4 <= batch_size_base <= 128
        assert 4 <= batch_size_mini <= 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])