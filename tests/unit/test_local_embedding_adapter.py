#!/usr/bin/env python3
"""Unit tests for LocalEmbeddingAdapter.

This module tests the LocalEmbeddingAdapter to ensure:
1. Proper exception handling (no random fallbacks)
2. Correct raising of EmbeddingError and EmbeddingServiceNotInitializedError
3. Dynamic dimension handling
4. Proper event loop management
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from packages.shared.text_processing.embedding_adapter import LocalEmbeddingAdapter
from packages.shared.text_processing.exceptions import EmbeddingError, EmbeddingServiceNotInitializedError


class TestLocalEmbeddingAdapter:
    """Test suite for LocalEmbeddingAdapter."""

    @pytest.fixture()
    def adapter(self):
        """Create a LocalEmbeddingAdapter instance."""
        return LocalEmbeddingAdapter()

    def test_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter._embed_dim is None
        assert hasattr(adapter, "_get_query_embedding")
        assert hasattr(adapter, "_aget_query_embedding")

    def test_embed_dim_not_initialized(self, adapter):
        """Test embed_dim property when service is not initialized."""
        with patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service:
            # Simulate uninitialized service
            mock_service._service = None
            mock_service._instance = None

            with pytest.raises(EmbeddingServiceNotInitializedError) as exc_info:
                _ = adapter.embed_dim

            assert "Embedding service is not initialized" in str(exc_info.value)

    def test_embed_dim_dynamic_retrieval(self, adapter):
        """Test dynamic dimension retrieval from service."""
        with patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service:
            # Mock the service structure
            mock_inner_service = MagicMock()
            mock_inner_service.get_dimension.return_value = 384
            mock_service._service = mock_inner_service

            # First access should retrieve and cache
            assert adapter.embed_dim == 384
            assert adapter._embed_dim == 384

            # Second access should use cached value
            assert adapter.embed_dim == 384
            mock_inner_service.get_dimension.assert_called_once()

    def test_embed_dim_lazy_loading_case(self, adapter):
        """Test dimension retrieval with lazy loading structure."""
        with patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service:
            # Mock lazy loading structure
            mock_instance = MagicMock()
            mock_inner_service = MagicMock()
            mock_inner_service.get_dimension.return_value = 768
            mock_instance._service = mock_inner_service

            mock_service._service = None  # Not directly initialized
            mock_service._instance = mock_instance

            assert adapter.embed_dim == 768

    def test_get_query_embedding_success(self, adapter):
        """Test successful query embedding generation."""
        mock_embedding = np.array([0.1, 0.2, 0.3])

        with (
            patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service,
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
        ):
            # Simulate no running loop
            mock_service.embed_single = AsyncMock(return_value=mock_embedding)

            result = adapter._get_query_embedding("test query")

            assert result == [0.1, 0.2, 0.3]
            mock_service.embed_single.assert_called_once_with("test query")

    def test_get_query_embedding_failure(self, adapter):
        """Test query embedding failure raises EmbeddingError."""
        with (
            patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service,
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
        ):
            mock_service.embed_single = AsyncMock(side_effect=Exception("Embedding failed"))

            with pytest.raises(EmbeddingError) as exc_info:
                adapter._get_query_embedding("test query")

            assert "Failed to generate embedding for query" in str(exc_info.value)
            assert "Embedding failed" in str(exc_info.value.__cause__)

    def test_get_query_embedding_with_running_loop(self, adapter):
        """Test embedding generation when there's already a running event loop."""
        mock_embedding = np.array([0.4, 0.5, 0.6])
        mock_loop = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = mock_embedding

        with (
            patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service,
            patch("asyncio.get_running_loop", return_value=mock_loop),
            patch("asyncio.run_coroutine_threadsafe", return_value=mock_future) as mock_run,
        ):
            mock_service.embed_single = AsyncMock(return_value=mock_embedding)

            result = adapter._get_query_embedding("test query")

            assert result == [0.4, 0.5, 0.6]
            mock_run.assert_called_once()

    @pytest.mark.asyncio()
    async def test_aget_query_embedding_success(self, adapter):
        """Test async query embedding generation."""
        mock_embedding = np.array([0.7, 0.8, 0.9])

        with patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service:
            mock_service.embed_single = AsyncMock(return_value=mock_embedding)

            result = await adapter._aget_query_embedding("async query")

            assert result == [0.7, 0.8, 0.9]
            mock_service.embed_single.assert_called_once_with("async query")

    @pytest.mark.asyncio()
    async def test_aget_query_embedding_failure(self, adapter):
        """Test async query embedding failure."""
        with patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service:
            mock_service.embed_single = AsyncMock(side_effect=Exception("Async failed"))

            with pytest.raises(EmbeddingError) as exc_info:
                await adapter._aget_query_embedding("async query")

            assert "Failed to generate embedding for query" in str(exc_info.value)

    def test_get_text_embeddings_success(self, adapter):
        """Test batch text embedding generation."""
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        with (
            patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service,
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
        ):
            mock_service.embed_texts = AsyncMock(return_value=mock_embeddings)

            result = adapter._get_text_embeddings(["text1", "text2", "text3"])

            assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            mock_service.embed_texts.assert_called_once_with(["text1", "text2", "text3"])

    def test_get_text_embeddings_failure(self, adapter):
        """Test batch embedding failure raises EmbeddingError."""
        with (
            patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service,
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
        ):
            mock_service.embed_texts = AsyncMock(side_effect=Exception("Batch failed"))

            with pytest.raises(EmbeddingError) as exc_info:
                adapter._get_text_embeddings(["text1", "text2"])

            assert "Failed to generate embeddings for texts" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_aget_text_embeddings_success(self, adapter):
        """Test async batch embedding generation."""
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])

        with patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service:
            mock_service.embed_texts = AsyncMock(return_value=mock_embeddings)

            result = await adapter._aget_text_embeddings(["async1", "async2"])

            assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_no_random_fallback(self, adapter):
        """Verify there's no random embedding fallback on failure."""
        with (
            patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service,
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
        ):
            mock_service.embed_single = AsyncMock(side_effect=Exception("Service unavailable"))

            # Should raise exception, not return random embeddings
            with pytest.raises(EmbeddingError):
                adapter._get_query_embedding("test")

            # Verify no random generation occurred
            # (No numpy.random calls or similar fallback logic)

    def test_event_loop_cleanup(self, adapter):
        """Test proper event loop cleanup after sync operations."""
        mock_embedding = np.array([0.1, 0.2, 0.3])

        with (
            patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service,
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.new_event_loop") as mock_new_loop,
            patch("asyncio.set_event_loop") as mock_set_loop,
        ):
            mock_loop = MagicMock()
            mock_new_loop.return_value = mock_loop
            mock_loop.run_until_complete.return_value = mock_embedding
            mock_service.embed_single = AsyncMock(return_value=mock_embedding)

            adapter._get_query_embedding("test")

            # Verify cleanup
            mock_loop.close.assert_called_once()
            assert mock_set_loop.call_count == 2  # Once to set, once to clear
            mock_set_loop.assert_called_with(None)  # Last call should clear

    def test_concurrent_safety(self, adapter):
        """Test that adapter handles concurrent requests safely."""
        mock_embeddings = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        call_count = 0

        def side_effect(*args):  # noqa: ARG001
            nonlocal call_count
            result = mock_embeddings[call_count % 2]
            call_count += 1
            return result

        with (
            patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service,
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
        ):
            mock_service.embed_single = AsyncMock(side_effect=side_effect)

            # Multiple calls should work independently
            result1 = adapter._get_query_embedding("query1")
            result2 = adapter._get_query_embedding("query2")

            assert result1 == [0.1, 0.2]
            assert result2 == [0.3, 0.4]

    def test_service_structure_validation(self, adapter):
        """Test proper error when service structure is unexpected."""
        with patch("packages.shared.text_processing.embedding_adapter.embedding_service") as mock_service:
            # Mock unexpected structure
            mock_instance = MagicMock()
            mock_instance._service = None  # Missing expected attribute
            mock_service._service = None
            mock_service._instance = mock_instance

            with pytest.raises(EmbeddingServiceNotInitializedError) as exc_info:
                _ = adapter.embed_dim

            assert "Embedding service is not initialized" in str(exc_info.value)
