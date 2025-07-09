#!/usr/bin/env python3
"""
Unit tests for EmbeddingService
Tests model loading, quantization, adaptive batching, and Qwen3 features
"""
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Mock the metrics module before importing embedding_service
sys.modules["vecpipe.metrics"] = MagicMock()


class TestEmbeddingService(unittest.TestCase):
    """Test cases for EmbeddingService"""

    def setUp(self):
        """Set up test fixtures"""
        # Patch at module level to avoid import errors
        self.mock_sentence_transformers = patch("packages.webui.embedding_service.SentenceTransformer")
        self.mock_auto_model = patch("packages.webui.embedding_service.AutoModel")
        self.mock_auto_tokenizer = patch("packages.webui.embedding_service.AutoTokenizer")
        # BitsAndBytesConfig is imported inside methods, so patch transformers directly
        self.mock_bitsandbytes_config = patch("transformers.BitsAndBytesConfig")
        self.mock_check_int8 = patch("packages.webui.embedding_service.check_int8_compatibility")

        # Start all patches
        self.mock_st = self.mock_sentence_transformers.start()
        self.mock_am = self.mock_auto_model.start()
        self.mock_at = self.mock_auto_tokenizer.start()
        self.mock_bnb = self.mock_bitsandbytes_config.start()
        self.mock_check = self.mock_check_int8.start()

        # Import after patching
        from packages.webui.embedding_service import EmbeddingService

        self.EmbeddingService = EmbeddingService

    def tearDown(self):
        """Clean up patches"""
        self.mock_sentence_transformers.stop()
        self.mock_auto_model.stop()
        self.mock_auto_tokenizer.stop()
        self.mock_bitsandbytes_config.stop()
        self.mock_check_int8.stop()

    @patch("torch.cuda.is_available")
    def test_load_model_quantization(self, mock_cuda_available):
        """Test loading models with different quantization settings"""
        mock_cuda_available.return_value = True

        # Create service instance
        service = self.EmbeddingService()

        # Mock model instance
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_model.half.return_value = mock_model
        self.mock_st.return_value = mock_model

        # Test INT8 quantization
        # Mock int8 compatibility check to pass
        self.mock_check.return_value = (True, "INT8 quantization is available")

        # Mock BitsAndBytesConfig instance
        mock_bnb_config = MagicMock()
        self.mock_bnb.return_value = mock_bnb_config

        # Test loading with int8
        result = service.load_model("test-model", quantization="int8")
        self.assertTrue(result)

        # Verify BitsAndBytesConfig was created with correct parameters
        self.mock_bnb.assert_called_once_with(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
        )

        # Verify SentenceTransformer was called with quantization config
        self.mock_st.assert_called_with(
            "test-model", device="cuda", model_kwargs={"quantization_config": mock_bnb_config}
        )

        # Reset mocks for float16 test
        self.mock_st.reset_mock()
        self.mock_bnb.reset_mock()

        # Test float16 quantization
        result = service.load_model("test-model", quantization="float16")
        self.assertTrue(result)

        # Verify model was loaded and converted to half precision
        self.mock_st.assert_called_with("test-model", device="cuda")
        mock_model.half.assert_called_once()

        # Verify correct quantization type was set
        self.assertEqual(service.current_quantization, "float16")

    @patch("torch.cuda.is_available")
    def test_int8_compatibility_fallback(self, mock_cuda_available):
        """Test graceful fallback when int8 compatibility check fails"""
        mock_cuda_available.return_value = True

        # Create service instance with fallback enabled (default)
        service = self.EmbeddingService()

        # Mock model instance
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        self.mock_st.return_value = mock_model

        # Mock int8 compatibility check to fail
        self.mock_check.return_value = (False, "INT8 not supported - missing dependencies")

        # Test loading with int8 quantization
        result = service.load_model("test-model", quantization="int8")

        # Should succeed by falling back to float32
        self.assertTrue(result)

        # Verify it fell back to float32
        self.assertEqual(service.current_quantization, "float32")

        # Verify SentenceTransformer was called without quantization config
        self.mock_st.assert_called_with("test-model", device="cuda")

        # Reset mocks
        self.mock_st.reset_mock()

        # Test with fallback disabled
        service2 = self.EmbeddingService()
        service2.allow_quantization_fallback = False

        # Mock model for service2
        mock_model2 = MagicMock()
        mock_model2.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        self.mock_st.return_value = mock_model2

        # Mock int8 compatibility check to fail
        self.mock_check.return_value = (False, "INT8 not supported")

        # When fallback is disabled, load_model should return False (not raise exception)
        # Looking at the code, the exception is caught and False is returned
        result = service2.load_model("test-model-2", quantization="int8")
        self.assertFalse(result)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.empty_cache")
    def test_adaptive_batching_oom_recovery(self, mock_empty_cache, mock_cuda_available):
        """Test adaptive batching recovers from OOM errors"""
        mock_cuda_available.return_value = True

        # Create service instance
        service = self.EmbeddingService()

        # Mock model instance
        mock_model = MagicMock()

        # Create a side effect that raises OOM on first call, then succeeds
        oom_count = 0

        def encode_with_oom(*_args, **kwargs):
            nonlocal oom_count
            batch_size = kwargs.get("batch_size", 32)
            if oom_count == 0 and batch_size > 16:
                oom_count += 1
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            # Return mock embeddings
            return np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        mock_model.encode.side_effect = encode_with_oom
        self.mock_st.return_value = mock_model

        # Load model - this will fail due to OOM during test embedding
        # So we need to mock the test embedding generation too
        with patch.object(service, "_generate_test_embedding", return_value=np.array([0.1, 0.2, 0.3])):
            result = service.load_model("test-model", quantization="float32")
            self.assertTrue(result)

        # Generate embeddings with initial batch size of 32
        texts = ["text1", "text2"]
        embeddings = service.generate_embeddings(texts, "test-model", batch_size=32, show_progress=False)

        # Should succeed after reducing batch size
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape, (2, 3))

        # Verify encode was called twice - once with 32 (failed), once with 16 (succeeded)
        calls = mock_model.encode.call_args_list
        self.assertEqual(len(calls), 2)
        # Check batch_size in kwargs
        self.assertEqual(calls[0].kwargs.get("batch_size"), 32)
        self.assertEqual(calls[1].kwargs.get("batch_size"), 16)

        # Verify batch size was reduced
        self.assertEqual(service.current_batch_size, 16)

        # Verify empty_cache was called after OOM
        mock_empty_cache.assert_called()

    @patch("torch.cuda.is_available")
    def test_qwen3_instruction_formatting(self, mock_cuda_available):
        """Test Qwen3 models correctly format instructions"""
        mock_cuda_available.return_value = True

        # Create service instance
        service = self.EmbeddingService()

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_batch_dict = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        mock_tokenizer.return_value = MagicMock(to=lambda device: mock_batch_dict)
        self.mock_at.from_pretrained.return_value = mock_tokenizer

        # Mock Qwen3 model
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 3, 768)
        mock_model.return_value = mock_outputs
        self.mock_am.from_pretrained.return_value = mock_model

        # Load Qwen3 model
        result = service.load_model("Qwen/Qwen3-Embedding-0.6B", quantization="float32")
        self.assertTrue(result)
        self.assertTrue(service.is_qwen3_model)

        # Generate embeddings with instruction
        texts = ["How to cook pasta?"]
        instruction = "Represent this cooking question for semantic search"

        with patch("packages.webui.embedding_service.last_token_pool") as mock_pool, patch("torch.nn.functional.normalize") as mock_normalize:
            # Mock the pooling and normalization
            mock_pool.return_value = torch.tensor([[0.1, 0.2, 0.3]])
            mock_normalize.return_value = torch.tensor([[0.1, 0.2, 0.3]])

            embeddings = service.generate_embeddings(
                texts, "Qwen/Qwen3-Embedding-0.6B", instruction=instruction, show_progress=False
            )

        # Verify tokenizer was called with correctly formatted text
        expected_text = f"Instruct: {instruction}\nQuery:{texts[0]}"
        mock_tokenizer.assert_called_with(
            [expected_text], padding=True, truncation=True, max_length=32768, return_tensors="pt"
        )

        # Verify embeddings were generated
        self.assertIsNotNone(embeddings)


if __name__ == "__main__":
    unittest.main()
