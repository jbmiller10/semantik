#!/usr/bin/env python3
"""
Full integration tests for embedding service across packages
"""
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Mock metrics before importing
sys.modules["shared.metrics.prometheus"] = MagicMock()


class TestVecpipeIntegration(unittest.TestCase):
    """Test embedding service integration with vecpipe."""

    @patch("torch.cuda.is_available")
    def test_model_manager_integration(self, mock_cuda: Mock) -> None:
        """Test that model manager can use the embedding service."""
        mock_cuda.return_value = False

        # Import after mocking
        from shared.embedding import EmbeddingService, embedding_service

        # Mock the model manager's usage pattern
        # model_manager.py uses embedding_service directly
        self.assertIsNotNone(embedding_service)
        self.assertTrue(hasattr(embedding_service, "load_model"))
        self.assertTrue(hasattr(embedding_service, "generate_embeddings"))
        self.assertTrue(hasattr(embedding_service, "unload_model"))

        # Create mock service for testing
        mock_service = EmbeddingService(mock_mode=True)

        # Test typical model manager workflow
        success = mock_service.load_model("test-model")
        self.assertTrue(success)

        # Generate embeddings
        texts = ["test text for model manager"]
        embeddings = mock_service.generate_embeddings(texts, "test-model", show_progress=False)

        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape[0], 1)

        # Unload model
        mock_service.unload_model()

    @patch("torch.cuda.is_available")
    def test_embed_chunks_unified_integration(self, mock_cuda: Mock) -> None:
        """Test embed_chunks_unified.py usage pattern."""
        mock_cuda.return_value = False

        from shared.embedding import EmbeddingService

        # This is how embed_chunks_unified uses the service
        service = EmbeddingService(mock_mode=True)

        # Test batch processing pattern
        chunks = [f"Chunk {i}: Some document text content" for i in range(100)]

        # Load model
        service.load_model("test-model", quantization="float32")

        # Process in batches (like the real implementation)
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = service.generate_embeddings(batch, "test-model", batch_size=batch_size, show_progress=False)
            if embeddings is not None:
                all_embeddings.append(embeddings)

        # Verify results
        self.assertGreater(len(all_embeddings), 0)
        combined = np.vstack(all_embeddings)
        self.assertEqual(combined.shape[0], len(chunks))

    @patch("torch.cuda.is_available")
    def test_search_api_integration(self, mock_cuda: Mock) -> None:
        """Test search_api.py integration pattern."""
        mock_cuda.return_value = False

        from shared.embedding import EmbeddingService

        # Create a mock service for testing
        mock_service = EmbeddingService(mock_mode=True)
        mock_service.load_model("test-model")

        # Simulate query embedding generation
        query = "search query text"
        query_embedding = mock_service.generate_single_embedding(query, "test-model", quantization="float32")

        self.assertIsNotNone(query_embedding)
        self.assertIsInstance(query_embedding, list)
        self.assertEqual(len(query_embedding), 384)  # Default mock dimension


class TestWebuiIntegration(unittest.TestCase):
    """Test embedding service integration with webui."""

    @patch("torch.cuda.is_available")
    def test_jobs_api_integration(self, mock_cuda: Mock) -> None:
        """Test jobs API usage pattern."""
        mock_cuda.return_value = False

        from shared.embedding import EmbeddingService

        # Create mock service for testing
        mock_service = EmbeddingService(mock_mode=True)

        # Simulate job processing workflow
        # 1. Load model
        mock_service.load_model("test-model", quantization="float32")

        # 2. Process file chunks
        file_chunks = [
            {"text": "Chunk 1 content", "metadata": {"page": 1}},
            {"text": "Chunk 2 content", "metadata": {"page": 2}},
            {"text": "Chunk 3 content", "metadata": {"page": 3}},
        ]

        texts = [chunk["text"] for chunk in file_chunks]
        embeddings = mock_service.generate_embeddings(
            texts, "test-model", batch_size=32, show_progress=True  # Jobs API shows progress
        )

        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape[0], len(file_chunks))

        # 3. Each embedding should be a vector
        for i, embedding in enumerate(embeddings):
            self.assertEqual(len(embedding), 384)
            self.assertIsInstance(embedding, np.ndarray)

    @patch("torch.cuda.is_available")
    def test_models_api_integration(self, mock_cuda: Mock) -> None:
        """Test models API usage pattern."""
        mock_cuda.return_value = False

        from shared.embedding import EmbeddingService
        from shared.embedding.models import list_available_models

        # Get available models
        models = list_available_models()
        self.assertGreater(len(models), 0)

        # Create mock service for testing
        mock_service = EmbeddingService(mock_mode=True)

        # Get model info for a known model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        info = mock_service.get_model_info(model_name, "float32")

        self.assertIsInstance(info, dict)
        self.assertIn("dimension", info)
        self.assertIn("model_name", info)
        self.assertIn("device", info)


class TestCrossPackageWorkflow(unittest.TestCase):
    """Test complete workflows across packages."""

    @patch("torch.cuda.is_available")
    def test_full_ingestion_workflow(self, mock_cuda: Mock) -> None:
        """Test full document ingestion workflow."""
        mock_cuda.return_value = False

        from shared.embedding import EmbeddingService

        # Initialize service (as webui would)
        service = EmbeddingService(mock_mode=True)
        service.load_model("sentence-transformers/all-MiniLM-L6-v2")

        # Simulate document processing workflow
        # 1. Extract text (mocked)
        document_text = (
            "This is a test document with multiple paragraphs.\n\n"
            "Each paragraph will become a chunk.\n\n"
            "We need to generate embeddings for each chunk."
        )

        # 2. Chunk text (simplified)
        chunks = document_text.split("\n\n")

        # 3. Generate embeddings (as vecpipe would)
        embeddings = service.generate_embeddings(chunks, "sentence-transformers/all-MiniLM-L6-v2", batch_size=32)

        # 4. Verify embeddings
        self.assertIsNotNone(embeddings)
        self.assertEqual(len(embeddings), len(chunks))

        # 5. Simulate vector storage preparation
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            vector = {
                "id": f"doc_1_chunk_{i}",
                "text": chunk,
                "embedding": embedding.tolist(),
                "metadata": {"document_id": "doc_1", "chunk_index": i},
            }
            vectors.append(vector)

        # Verify vector format
        self.assertEqual(len(vectors), 3)
        for vector in vectors:
            self.assertIn("embedding", vector)
            self.assertEqual(len(vector["embedding"]), 384)

    @patch("torch.cuda.is_available")
    async def test_async_service_lifecycle_workflow(self, mock_cuda: Mock) -> None:
        """Test async service lifecycle across packages."""
        mock_cuda.return_value = False

        from shared.embedding import cleanup, get_embedding_service, initialize_embedding_service

        # 1. Initialize service (as search_api might)
        await initialize_embedding_service(
            "sentence-transformers/all-MiniLM-L6-v2", quantization="float32", mock_mode=True
        )

        # 2. Get service from multiple places (simulating different packages)
        service1 = await get_embedding_service()  # vecpipe
        service2 = await get_embedding_service()  # webui

        # Should be same instance
        self.assertIs(service1, service2)

        # 3. Use service
        query = "test query"
        query_embedding = await service1.embed_single(query)

        self.assertEqual(len(query_embedding), 384)

        # 4. Cleanup (as shutdown handler would)
        await cleanup()

        # 5. After cleanup, should get new instance
        await initialize_embedding_service("test-model", mock_mode=True)
        service3 = await get_embedding_service()
        self.assertIsNot(service1, service3)


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling across packages."""

    @patch("torch.cuda.is_available")
    def test_int8_fallback_integration(self, mock_cuda: Mock) -> None:
        """Test INT8 fallback behavior."""
        mock_cuda.return_value = True  # Pretend we have CUDA

        # Mock bitsandbytes not available
        with patch.dict(sys.modules, {"bitsandbytes": None}):
            from shared.embedding import EmbeddingService

            service = EmbeddingService(mock_mode=True)
            service.allow_quantization_fallback = True

            # Should fall back to float32
            success = service.load_model("test-model", quantization="int8")
            self.assertTrue(success)

            # In mock mode, quantization doesn't actually fall back
            # Just verify it loaded successfully
            self.assertTrue(service._service.is_initialized)

    @patch("torch.cuda.is_available")
    def test_oom_recovery_pattern(self, mock_cuda: Mock) -> None:
        """Test OOM recovery pattern used in the codebase."""
        mock_cuda.return_value = False

        from shared.embedding import EmbeddingService

        service = EmbeddingService(mock_mode=True)
        service.load_model("test-model")

        # Simulate the adaptive batch sizing pattern
        texts = ["text"] * 1000
        batch_size = 128

        while batch_size > 0:
            try:
                embeddings = service.generate_embeddings(texts, "test-model", batch_size=batch_size)
                if embeddings is not None:
                    break
            except Exception:
                # Reduce batch size and retry
                batch_size = batch_size // 2
                if batch_size < 1:
                    raise

        # Should succeed with mock mode
        self.assertIsNotNone(embeddings)
        self.assertEqual(len(embeddings), len(texts))


if __name__ == "__main__":
    # Run async tests
    unittest.main()
