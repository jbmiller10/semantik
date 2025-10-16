#!/usr/bin/env python3
"""
Manual unittest suite covering embedding workflows across packages.

Moved from pytest on 2025-10-16. Execute directly to sanity check embedding
integration patterns without running the full automated test suite.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from packages.shared.embedding import (
    EmbeddingService,
    cleanup,
    embedding_service,
    get_embedding_service,
    initialize_embedding_service,
)
from packages.shared.embedding.models import list_available_models

# Mock metrics before importing embedding internals
sys.modules["packages.shared.metrics.prometheus"] = MagicMock()


class TestVecpipeIntegration(unittest.TestCase):
    """Test embedding service integration with vecpipe components."""

    @patch("torch.cuda.is_available")
    def test_model_manager_integration(self, mock_cuda: Mock) -> None:
        mock_cuda.return_value = False
        assert embedding_service is not None
        mock_service = EmbeddingService(mock_mode=True)
        success = mock_service.load_model("test-model")
        assert success
        texts = ["test text for model manager"]
        embeddings = mock_service.generate_embeddings(texts, "test-model", show_progress=False)
        assert embeddings is not None
        assert embeddings.shape[0] == 1
        mock_service.unload_model()

    @patch("torch.cuda.is_available")
    def test_embed_chunks_unified_integration(self, mock_cuda: Mock) -> None:
        mock_cuda.return_value = False
        service = EmbeddingService(mock_mode=True)
        chunks = [f"Chunk {i}: Some document text content" for i in range(100)]
        service.load_model("test-model", quantization="float32")
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = service.generate_embeddings(batch, "test-model", batch_size=batch_size, show_progress=False)
            if embeddings is not None:
                all_embeddings.append(embeddings)

        assert len(all_embeddings) > 0
        combined = np.vstack(all_embeddings)
        assert combined.shape[0] == len(chunks)

    @patch("torch.cuda.is_available")
    def test_search_api_integration(self, mock_cuda: Mock) -> None:
        mock_cuda.return_value = False
        service = EmbeddingService(mock_mode=True)
        service.load_model("test-model")
        query_embedding = service.generate_single_embedding("search query text", "test-model", quantization="float32")
        assert isinstance(query_embedding, list)
        assert len(query_embedding) == 384


class TestWebuiIntegration(unittest.TestCase):
    """Test embedding service integration with webui usage patterns."""

    @patch("torch.cuda.is_available")
    def test_operations_api_integration(self, mock_cuda: Mock) -> None:
        mock_cuda.return_value = False
        service = EmbeddingService(mock_mode=True)
        service.load_model("test-model", quantization="float32")
        file_chunks: list[dict[str, Any]] = [
            {"text": "Chunk 1 content", "metadata": {"page": 1}},
            {"text": "Chunk 2 content", "metadata": {"page": 2}},
            {"text": "Chunk 3 content", "metadata": {"page": 3}},
        ]
        texts = [chunk["text"] for chunk in file_chunks]
        embeddings = service.generate_embeddings(texts, "test-model", batch_size=32, show_progress=True)
        assert embeddings is not None
        assert embeddings.shape[0] == len(file_chunks)
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == 384

    @patch("torch.cuda.is_available")
    def test_models_api_integration(self, mock_cuda: Mock) -> None:
        mock_cuda.return_value = False
        models = list_available_models()
        assert len(models) > 0
        service = EmbeddingService(mock_mode=True)
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        info = service.get_model_info(model_name, "float32")
        assert isinstance(info, dict)
        assert "dimension" in info
        assert "model_name" in info
        assert "device" in info


class TestCrossPackageWorkflow(unittest.TestCase):
    """Test workflows that span multiple packages."""

    @patch("torch.cuda.is_available")
    def test_full_ingestion_workflow(self, mock_cuda: Mock) -> None:
        mock_cuda.return_value = False
        service = EmbeddingService(mock_mode=True)
        service.load_model("sentence-transformers/all-MiniLM-L6-v2")
        document_text = (
            "This is a test document with multiple paragraphs.\n\n"
            "Each paragraph will become a chunk.\n\n"
            "We need to generate embeddings for each chunk."
        )
        chunks = document_text.split("\n\n")
        embeddings = service.generate_embeddings(chunks, "sentence-transformers/all-MiniLM-L6-v2", batch_size=32)
        assert embeddings is not None
        assert len(embeddings) == len(chunks)
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            vectors.append(
                {
                    "id": f"doc_1_chunk_{i}",
                    "text": chunk,
                    "embedding": embedding.tolist(),
                    "metadata": {"document_id": "doc_1", "chunk_index": i},
                }
            )
        assert len(vectors) == 3
        for vector in vectors:
            assert "embedding" in vector
            assert len(vector["embedding"]) == 384

    @patch("torch.cuda.is_available")
    def test_async_service_lifecycle_workflow(self, mock_cuda: Mock) -> None:
        mock_cuda.return_value = False

        async def async_test() -> None:
            await initialize_embedding_service(
                "sentence-transformers/all-MiniLM-L6-v2", quantization="float32", mock_mode=True
            )
            service1 = await get_embedding_service()
            service2 = await get_embedding_service()
            assert service1 is service2
            query_embedding = await service1.embed_single("test query")
            assert len(query_embedding) == 384
            await cleanup()
            await initialize_embedding_service("test-model", mock_mode=True)
            service3 = await get_embedding_service()
            assert service1 is not service3

        asyncio.run(async_test())


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling patterns in embedding workflows."""

    @patch("torch.cuda.is_available")
    def test_int8_fallback_integration(self, mock_cuda: Mock) -> None:
        mock_cuda.return_value = True
        with patch.dict(sys.modules, {"bitsandbytes": None}):
            service = EmbeddingService(mock_mode=True)
            service.allow_quantization_fallback = True
            success = service.load_model("test-model", quantization="int8")
            assert success
            assert service._service.is_initialized

    @patch("torch.cuda.is_available")
    def test_oom_recovery_pattern(self, mock_cuda: Mock) -> None:
        mock_cuda.return_value = False
        service = EmbeddingService(mock_mode=True)
        service.load_model("test-model")
        texts = ["text"] * 1000
        batch_size = 128

        while batch_size > 0:
            try:
                embeddings = service.generate_embeddings(texts, "test-model", batch_size=batch_size)
                if embeddings is not None:
                    break
            except Exception:
                batch_size //= 2
                if batch_size < 1:
                    raise

        assert embeddings is not None
        assert len(embeddings) == len(texts)


if __name__ == "__main__":
    unittest.main()
