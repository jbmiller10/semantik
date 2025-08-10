#!/usr/bin/env python3

"""
Comprehensive tests for OOM handling functionality in the EmbeddingService.

This test suite covers:
- Batch size reduction on OOM errors
- Minimum batch size enforcement
- Batch size recovery after successful batches
- Adaptive sizing toggle behavior
- CPU vs GPU adaptive sizing behavior
- Multiple OOM reductions and batch size progression

Note: The metrics module is mocked at the top of the file to prevent import errors.
While the current implementation doesn't use metrics for OOM tracking, this mock
ensures compatibility if metrics are added in the future.
"""

import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from sentence_transformers import SentenceTransformer

from packages.shared.embedding.dense import DenseEmbeddingService

# Mock the metrics module before importing
sys.modules["packages.shared.metrics.prometheus"] = MagicMock()


class TestEmbeddingOOMHandling(unittest.TestCase):
    """Test cases for OOM handling in the EmbeddingService"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        # Clear any existing patches
        self.patches = []

    def tearDown(self) -> None:
        """Clean up patches"""
        for p in self.patches:
            p.stop()

    @patch("packages.shared.embedding.dense.isinstance", side_effect=lambda _obj, _cls: True)
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.empty_cache")
    def test_oom_batch_size_reduction(self, mock_empty_cache, mock_cuda_available, mock_isinstance) -> None:
        """Test that batch size is reduced on OOM error"""
        mock_cuda_available.return_value = True

        service = DenseEmbeddingService(mock_mode=False)
        service._initialized = True
        service.device = "cuda"
        service.model_name = "test-model"
        service.dimension = 384
        service.quantization = "float32"
        service.dtype = torch.float32
        service.enable_adaptive_batch_size = True
        service.original_batch_size = 32
        service.current_batch_size = 32
        service.min_batch_size = 1

        # Mock the model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        service.model = mock_model
        service.tokenizer = mock_tokenizer

        # Set up tokenizer to return mock tensors
        mock_batch_dict = {"input_ids": torch.zeros((32, 10)), "attention_mask": torch.ones((32, 10))}
        mock_tokenizer.return_value = mock_batch_dict

        # Mock model to raise OOM on first call, then succeed
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(16, 10, 384)

        oom_count = 0

        def model_forward(**_kwargs) -> None:
            nonlocal oom_count
            if oom_count == 0:
                oom_count += 1
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return mock_outputs

        mock_model.side_effect = model_forward

        # Test with SentenceTransformer path
        service.is_qwen_model = False
        # Create a mock that passes isinstance check by using spec with SentenceTransformer

        mock_st_model = Mock(spec=SentenceTransformer)

        # First call raises OOM, second succeeds
        encode_call_count = 0

        def encode_with_oom(texts, batch_size, **_kwargs) -> None:
            nonlocal encode_call_count
            if encode_call_count == 0 and batch_size == 32:
                encode_call_count += 1
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            # Return embeddings with reduced batch size
            return np.random.randn(len(texts), 384).astype(np.float32)

        mock_st_model.encode.side_effect = encode_with_oom
        service.model = mock_st_model

        # Call embedding method
        texts = ["test text"] * 64
        embeddings = service._embed_sentence_transformer_texts(
            texts, batch_size=32, normalize=True, show_progress=False
        )

        # Verify batch size was reduced
        assert service.current_batch_size == 16
        assert embeddings is not None
        assert embeddings.shape == (64, 384)

        # Verify empty_cache was called
        mock_empty_cache.assert_called()

    @patch("packages.shared.embedding.dense.isinstance", side_effect=lambda _obj, _cls: True)
    @patch("torch.cuda.is_available")
    def test_minimum_batch_size_enforcement(self, mock_cuda_available, mock_isinstance) -> None:
        """Test that batch size doesn't go below min_batch_size"""
        mock_cuda_available.return_value = True

        service = DenseEmbeddingService(mock_mode=False)
        service._initialized = True
        service.device = "cuda"
        service.model_name = "test-model"
        service.dimension = 384
        service.quantization = "float32"
        service.enable_adaptive_batch_size = True
        service.original_batch_size = 32
        service.current_batch_size = 2  # Start with small batch size
        service.min_batch_size = 1

        # Mock SentenceTransformer model

        mock_model = Mock(spec=SentenceTransformer)

        # Always raise OOM to test minimum batch size enforcement
        mock_model.encode.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")
        service.model = mock_model
        service.is_qwen_model = False

        # Test that RuntimeError is raised when min batch size still causes OOM
        texts = ["test"] * 10
        with pytest.raises(RuntimeError) as context:
            service._embed_sentence_transformer_texts(texts, batch_size=2, normalize=True, show_progress=False)

        assert "minimum batch size" in str(context.value)

    @patch("packages.shared.embedding.dense.isinstance", side_effect=lambda _obj, _cls: True)
    @patch("torch.cuda.is_available")
    def test_batch_size_recovery(self, mock_cuda_available, mock_isinstance) -> None:
        """Test that batch size increases after successful batches"""
        mock_cuda_available.return_value = True

        service = DenseEmbeddingService(mock_mode=False)
        service._initialized = True
        service.device = "cuda"
        service.model_name = "test-model"
        service.dimension = 384
        service.enable_adaptive_batch_size = True
        service.original_batch_size = 32
        service.current_batch_size = 8  # Reduced from original
        service.min_batch_size = 1
        service.successful_batches = 0
        service.batch_size_increase_threshold = 3  # Reduce threshold for testing

        # Mock model that always succeeds

        mock_model = Mock(spec=SentenceTransformer)
        mock_model.encode.return_value = np.random.randn(10, 384).astype(np.float32)
        service.model = mock_model
        service.is_qwen_model = False

        # Run multiple successful batches
        for _ in range(3):
            embeddings = service._embed_sentence_transformer_texts(
                ["test"] * 10, batch_size=8, normalize=True, show_progress=False
            )
            assert embeddings is not None

        # Verify batch size increased
        assert service.current_batch_size == 16
        assert service.successful_batches == 0  # Reset after increase

    @patch("packages.shared.embedding.dense.isinstance", side_effect=lambda _obj, _cls: True)
    @patch("torch.cuda.is_available")
    def test_adaptive_sizing_disabled(self, mock_cuda_available, mock_isinstance) -> None:
        """Test behavior when adaptive sizing is disabled"""
        mock_cuda_available.return_value = True

        service = DenseEmbeddingService(mock_mode=False)
        service._initialized = True
        service.device = "cuda"
        service.model_name = "test-model"
        service.dimension = 384
        service.enable_adaptive_batch_size = False  # Disabled
        service.original_batch_size = 32
        service.current_batch_size = 32

        # Mock model that raises OOM

        mock_model = Mock(spec=SentenceTransformer)
        mock_model.encode.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")
        service.model = mock_model
        service.is_qwen_model = False

        # Should raise OOM without retry when adaptive sizing is disabled
        with pytest.raises(torch.cuda.OutOfMemoryError):
            service._embed_sentence_transformer_texts(["test"] * 10, batch_size=32, normalize=True, show_progress=False)

    @patch("packages.shared.embedding.dense.isinstance", side_effect=lambda _obj, _cls: True)
    @patch("torch.cuda.is_available")
    def test_cpu_no_adaptive_sizing(self, mock_cuda_available, mock_isinstance) -> None:
        """Test that CPU doesn't use adaptive sizing"""
        mock_cuda_available.return_value = False

        service = DenseEmbeddingService(mock_mode=False)
        service._initialized = True
        service.device = "cpu"
        service.model_name = "test-model"
        service.dimension = 384
        service.enable_adaptive_batch_size = True  # Enabled but should be ignored on CPU

        # Mock model - on CPU, OOM errors are unlikely but test the logic

        mock_model = Mock(spec=SentenceTransformer)
        mock_model.encode.side_effect = RuntimeError("Out of memory")  # Generic OOM for CPU
        service.model = mock_model
        service.is_qwen_model = False

        # Should raise error without retry on CPU
        with pytest.raises(RuntimeError):
            service._embed_sentence_transformer_texts(["test"] * 10, batch_size=32, normalize=True, show_progress=False)

    @patch("packages.shared.embedding.dense.isinstance", side_effect=lambda _obj, _cls: True)
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.empty_cache")
    def test_qwen_model_oom_handling(self, mock_empty_cache, mock_cuda_available, mock_isinstance) -> None:
        """Test OOM handling for Qwen models"""
        mock_cuda_available.return_value = True

        service = DenseEmbeddingService(mock_mode=False)
        service._initialized = True
        service.device = "cuda"
        service.model_name = "Qwen/Qwen3-Embedding-0.6B"
        service.dimension = 1024
        service.quantization = "float32"
        service.dtype = torch.float32
        service.is_qwen_model = True
        service.enable_adaptive_batch_size = True
        service.original_batch_size = 16
        service.current_batch_size = 16
        service.min_batch_size = 1
        service.max_sequence_length = 512

        # Mock tokenizer
        mock_tokenizer = Mock()
        batch_size_from_tokenizer = []

        def tokenizer_side_effect(texts, **_kwargs) -> None:
            batch_size_from_tokenizer.append(len(texts))
            # Create a mock object that has a .to() method
            mock_batch = Mock()
            mock_batch.to.return_value = {
                "input_ids": torch.zeros((len(texts), 10)),
                "attention_mask": torch.ones((len(texts), 10)),
            }
            return mock_batch

        mock_tokenizer.side_effect = tokenizer_side_effect
        service.tokenizer = mock_tokenizer

        # Mock model
        mock_model = Mock()
        call_count = 0

        def model_forward(**_kwargs) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call with batch size 16 raises OOM
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            # Subsequent calls succeed
            batch_size = _kwargs["input_ids"].shape[0]
            mock_output = Mock()
            mock_output.last_hidden_state = torch.randn(batch_size, 10, 1024)
            return mock_output

        mock_model.side_effect = model_forward
        service.model = mock_model

        # Test embedding
        texts = ["test text"] * 32
        embeddings = service._embed_qwen_texts(texts, batch_size=16, normalize=True, instruction=None)

        # Verify results
        assert embeddings is not None
        assert embeddings.shape == (32, 1024)

        # Verify batch size was reduced
        assert service.current_batch_size == 8

        # Verify empty_cache was called
        mock_empty_cache.assert_called()

    @patch("packages.shared.embedding.dense.isinstance", side_effect=lambda _obj, _cls: True)
    @patch("torch.cuda.is_available")
    def test_batch_size_increase_with_original_limit(self, mock_cuda_available, mock_isinstance) -> None:
        """Test that batch size doesn't exceed original when recovering"""
        mock_cuda_available.return_value = True

        service = DenseEmbeddingService(mock_mode=False)
        service._initialized = True
        service.device = "cuda"
        service.model_name = "test-model"
        service.dimension = 384
        service.enable_adaptive_batch_size = True
        service.original_batch_size = 32
        service.current_batch_size = 16
        service.min_batch_size = 1
        service.successful_batches = 0
        service.batch_size_increase_threshold = 2

        # Mock model

        mock_model = Mock(spec=SentenceTransformer)
        mock_model.encode.return_value = np.random.randn(10, 384).astype(np.float32)
        service.model = mock_model
        service.is_qwen_model = False

        # Run successful batches to trigger increase
        for _ in range(2):
            service._embed_sentence_transformer_texts(["test"] * 10, batch_size=16, normalize=True, show_progress=False)

        # Should increase to 32 (original), not beyond
        assert service.current_batch_size == 32

        # Run more successful batches
        for _ in range(2):
            service._embed_sentence_transformer_texts(["test"] * 10, batch_size=32, normalize=True, show_progress=False)

        # Should stay at 32
        assert service.current_batch_size == 32

    @patch("packages.shared.embedding.dense.isinstance", side_effect=lambda _obj, _cls: True)
    @patch("torch.cuda.is_available")
    def test_adaptive_batch_size_initialization(self, mock_cuda_available, mock_isinstance) -> None:
        """Test that adaptive batch size is properly initialized on first use"""
        mock_cuda_available.return_value = True

        service = DenseEmbeddingService(mock_mode=False)
        service._initialized = True
        service.device = "cuda"
        service.model_name = "test-model"
        service.dimension = 384
        service.enable_adaptive_batch_size = True
        service.is_qwen_model = False

        # Initially, these should be None
        assert service.original_batch_size is None
        assert service.current_batch_size is None

        # Mock model

        mock_model = Mock(spec=SentenceTransformer)
        mock_model.encode.return_value = np.random.randn(10, 384).astype(np.float32)
        service.model = mock_model

        # First embedding call should initialize batch sizes
        service._embed_sentence_transformer_texts(["test"] * 10, batch_size=64, normalize=True, show_progress=False)

        # Verify initialization
        assert service.original_batch_size == 64
        assert service.current_batch_size == 64

    @patch("packages.shared.embedding.dense.isinstance", side_effect=lambda _obj, _cls: True)
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.empty_cache")
    def test_multiple_oom_reductions(self, mock_empty_cache, mock_cuda_available, mock_isinstance) -> None:
        """Test multiple OOM errors lead to progressive batch size reduction"""
        mock_cuda_available.return_value = True

        service = DenseEmbeddingService(mock_mode=False)
        service._initialized = True
        service.device = "cuda"
        service.model_name = "test-model"
        service.dimension = 384
        service.enable_adaptive_batch_size = True
        service.original_batch_size = 64
        service.current_batch_size = 64
        service.min_batch_size = 4
        service.is_qwen_model = False

        # Mock model that raises OOM for large batch sizes

        mock_model = Mock(spec=SentenceTransformer)

        def encode_with_size_limit(texts, batch_size, **_kwargs) -> None:
            if batch_size > 16:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return np.random.randn(len(texts), 384).astype(np.float32)

        mock_model.encode.side_effect = encode_with_size_limit
        service.model = mock_model

        # Run embedding - should reduce from 64 -> 32 -> 16
        embeddings = service._embed_sentence_transformer_texts(
            ["test"] * 100, batch_size=64, normalize=True, show_progress=False
        )

        # Verify final batch size
        assert service.current_batch_size == 16
        assert embeddings is not None

        # Verify empty_cache was called multiple times
        assert mock_empty_cache.call_count == 2  # For 64->32 and 32->16


if __name__ == "__main__":
    unittest.main()
