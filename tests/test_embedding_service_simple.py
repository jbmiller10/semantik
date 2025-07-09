#!/usr/bin/env python3
"""
Simplified unit tests for EmbeddingService that work with the new architecture
"""
# Mock the metrics module before importing
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.modules["shared.metrics.prometheus"] = MagicMock()


class TestEmbeddingService(unittest.TestCase):
    """Test cases for EmbeddingService"""

    @patch("torch.cuda.is_available")
    def test_basic_functionality(self, mock_cuda):
        """Test basic embedding service functionality"""
        mock_cuda.return_value = False  # Use CPU for simplicity

        from shared.embedding import EmbeddingService

        # Create service
        service = EmbeddingService()

        # Verify basic properties
        assert hasattr(service, "load_model")
        assert hasattr(service, "generate_embeddings")
        assert hasattr(service, "device")
        assert service.device == "cpu"

    @patch("torch.cuda.is_available")
    def test_quantization_fallback_property(self, mock_cuda):
        """Test quantization fallback can be configured"""
        mock_cuda.return_value = False

        from shared.embedding import EmbeddingService

        service = EmbeddingService()

        # Default should allow fallback
        assert service.allow_quantization_fallback is True

        # Should be configurable
        service.allow_quantization_fallback = False
        assert service.allow_quantization_fallback is False

    def test_model_info_export(self):
        """Test that model info is exported"""
        from shared.embedding import POPULAR_MODELS, QUANTIZED_MODEL_INFO

        # Should have model info
        assert len(QUANTIZED_MODEL_INFO) > 0
        assert POPULAR_MODELS == QUANTIZED_MODEL_INFO

        # Check a known model
        assert "Qwen/Qwen3-Embedding-0.6B" in QUANTIZED_MODEL_INFO
        assert QUANTIZED_MODEL_INFO["Qwen/Qwen3-Embedding-0.6B"]["dimension"] == 1024


if __name__ == "__main__":
    unittest.main()
