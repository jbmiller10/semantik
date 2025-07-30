#!/usr/bin/env python3
"""
Simplified unit tests for EmbeddingService that work with the new architecture
"""
# Mock the metrics module before importing
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.modules["packages.shared.metrics.prometheus"] = MagicMock()


class TestEmbeddingService(unittest.TestCase):
    """Test cases for EmbeddingService"""

    @patch("torch.cuda.is_available")
    def test_basic_functionality(self, mock_cuda) -> None:
        """Test basic embedding service functionality"""
        mock_cuda.return_value = False  # Use CPU for simplicity

        from packages.shared.embedding import EmbeddingService

        # Create service
        service = EmbeddingService()

        # Verify basic properties
        assert hasattr(service, "load_model")
        assert hasattr(service, "generate_embeddings")
        assert hasattr(service, "device")
        assert service.device == "cpu"

    @patch("torch.cuda.is_available")
    def test_quantization_fallback_property(self, mock_cuda) -> None:
        """Test quantization fallback can be configured"""
        mock_cuda.return_value = False

        from packages.shared.embedding import EmbeddingService

        service = EmbeddingService()

        # Default should allow fallback
        assert service.allow_quantization_fallback is True

        # Should be configurable
        service.allow_quantization_fallback = False
        assert service.allow_quantization_fallback is False

    def test_model_info_export(self) -> None:
        """Test that model info is exported"""
        from packages.shared.embedding import POPULAR_MODELS, QUANTIZED_MODEL_INFO

        # Should have model info
        assert len(QUANTIZED_MODEL_INFO) > 0
        assert POPULAR_MODELS == QUANTIZED_MODEL_INFO

        # Check a known model
        assert "Qwen/Qwen3-Embedding-0.6B" in QUANTIZED_MODEL_INFO
        assert QUANTIZED_MODEL_INFO["Qwen/Qwen3-Embedding-0.6B"]["dimension"] == 1024

    @patch("torch.cuda.is_available")
    def test_adaptive_batch_size_configuration(self, mock_cuda) -> None:
        """Test adaptive batch size configuration"""
        mock_cuda.return_value = True

        from packages.shared.embedding import EmbeddingService
        from packages.shared.config.vecpipe import VecpipeConfig

        # Test with config
        config = VecpipeConfig()
        service = EmbeddingService(config=config)

        # The sync wrapper delegates to the internal async service
        internal_service = service._service

        # Should load adaptive batch size settings from config
        assert hasattr(internal_service, "enable_adaptive_batch_size")
        assert hasattr(internal_service, "min_batch_size")
        assert hasattr(internal_service, "batch_size_increase_threshold")
        assert internal_service.enable_adaptive_batch_size == config.ENABLE_ADAPTIVE_BATCH_SIZE
        assert internal_service.min_batch_size == config.MIN_BATCH_SIZE
        assert internal_service.batch_size_increase_threshold == config.BATCH_SIZE_INCREASE_THRESHOLD

        # Test without config (defaults)
        service2 = EmbeddingService()
        internal_service2 = service2._service
        assert internal_service2.enable_adaptive_batch_size is True
        assert internal_service2.min_batch_size == 1
        assert internal_service2.batch_size_increase_threshold == 10


if __name__ == "__main__":
    unittest.main()
