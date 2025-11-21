#!/usr/bin/env python3

"""Integration tests for dimension validation in search API and tasks.

This module tests the integration of dimension validation utilities with:
1. Search API endpoints
2. Background tasks (indexing/re-indexing)
3. End-to-end embedding dimension handling
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from shared.database.exceptions import DimensionMismatchError
from shared.embedding.validation import (
    adjust_embeddings_dimension,
    get_model_dimension,
    validate_dimension_compatibility,
    validate_embedding_dimensions,
)

# Mock imports for tests that are marked as skipped pending new architecture
# These are placeholder imports as the actual modules don't exist yet
SearchService = MagicMock
index_collection_task = AsyncMock
reindex_collection_task = AsyncMock
search_collection = AsyncMock


class TestDimensionValidationIntegration:
    """Integration tests for dimension validation across the system."""

    @pytest.fixture()
    def mock_embedding_service(self) -> None:
        """Mock embedding service with configurable dimensions."""
        service = MagicMock()
        service._service = MagicMock()
        service._service.get_dimension = MagicMock(return_value=384)
        service.embed_single = AsyncMock(return_value=np.random.rand(384))
        service.embed_texts = AsyncMock(return_value=np.random.rand(5, 384))
        return service

    @pytest.fixture()
    def mock_collection(self) -> None:
        """Mock collection with dimension info."""
        collection = MagicMock()
        collection.id = "test-collection"
        collection.expected_embedding_dimension = 384
        collection.embedding_model = "test-model"
        return collection

    @pytest.mark.asyncio()
    async def test_search_api_query_dimension_validation(self, mock_embedding_service) -> None:
        """Test that search API validates query embedding dimensions."""
        pytest.skip("Search service integration test needs updating for new architecture")

        with patch("shared.embedding.dense.embedding_service", mock_embedding_service):
            # Create search service
            search_service = SearchService()

            # Mock Qdrant client
            mock_qdrant = MagicMock()
            mock_qdrant.search = AsyncMock(return_value=[])
            search_service._client = mock_qdrant

            # Test with matching dimensions
            query = "test query"
            collection_id = "test-collection"

            # Mock get_collection_info to return expected dimension
            mock_qdrant.get_collection = AsyncMock(
                return_value=MagicMock(config=MagicMock(params=MagicMock(vectors=MagicMock(size=384))))
            )

            # Should not raise any errors
            await search_service.search(collection_id=collection_id, query=query, limit=10)

            # Verify embedding was called
            mock_embedding_service.embed_single.assert_called_with(query)

    @pytest.mark.asyncio()
    async def test_search_api_dimension_mismatch_handling(self, mock_embedding_service) -> None:
        """Test search API handles dimension mismatches gracefully."""
        pytest.skip("Search service integration test needs updating for new architecture")

        # Set up mismatched dimensions
        mock_embedding_service._service.get_dimension.return_value = 512
        mock_embedding_service.embed_single.return_value = np.random.rand(512)

        with patch("shared.embedding.dense.embedding_service", mock_embedding_service):
            search_service = SearchService()

            # Mock Qdrant with different dimension
            mock_qdrant = MagicMock()
            mock_qdrant.get_collection = AsyncMock(
                return_value=MagicMock(config=MagicMock(params=MagicMock(vectors=MagicMock(size=384))))
            )
            search_service._client = mock_qdrant

            # Test query with dimension adjustment
            with patch("shared.embedding.validation.adjust_embeddings_dimension") as mock_adjust:
                mock_adjust.return_value = [np.random.rand(384).tolist()]

                query = "test query"
                collection_id = "test-collection"

                # Should handle dimension mismatch
                await search_service.search(collection_id=collection_id, query=query, limit=10)

                # Verify dimension adjustment was called
                mock_adjust.assert_called_once()

    @pytest.mark.asyncio()
    async def test_indexing_task_dimension_validation(self, mock_collection, mock_embedding_service) -> None:
        """Test that indexing tasks validate embedding dimensions."""
        pytest.skip("Indexing task test needs updating for new architecture")

        with (
            patch("shared.embedding.dense.embedding_service", mock_embedding_service),
            patch("packages.worker.tasks.indexing_tasks.get_collection") as mock_get_collection,
        ):
            mock_get_collection.return_value = mock_collection

            # Mock document processing
            with patch("packages.worker.tasks.indexing_tasks.process_documents") as mock_process:
                mock_process.return_value = AsyncMock()

                # Run indexing task
                await index_collection_task(collection_id=mock_collection.id, operation_id="test-op")

                # Verify dimension was checked
                assert mock_embedding_service._service.get_dimension.called

    @pytest.mark.asyncio()
    async def test_reindexing_task_dimension_validation(self, mock_collection, mock_embedding_service) -> None:
        """Test that re-indexing tasks handle dimension changes."""
        pytest.skip("Reindexing task test needs updating for new architecture")

        # Simulate dimension change: old=384, new=512
        mock_collection.expected_embedding_dimension = 384
        mock_embedding_service._service.get_dimension.return_value = 512

        with (
            patch("shared.embedding.dense.embedding_service", mock_embedding_service),
            patch("packages.worker.tasks.reindexing_tasks.get_collection") as mock_get_collection,
        ):
            mock_get_collection.return_value = mock_collection

            # Mock Qdrant operations
            with patch("packages.worker.tasks.reindexing_tasks.recreate_collection") as mock_recreate:
                mock_recreate.return_value = AsyncMock()

                with patch("packages.worker.tasks.reindexing_tasks.process_documents") as mock_process:
                    mock_process.return_value = AsyncMock()

                    # Run re-indexing task
                    await reindex_collection_task(collection_id=mock_collection.id, operation_id="test-op")

                    # Verify collection was recreated with new dimension
                    mock_recreate.assert_called_once()
                    call_args = mock_recreate.call_args
                    assert call_args is not None

    def test_get_model_dimension(self) -> None:
        """Test getting model dimensions."""
        # Test known models
        dim = get_model_dimension("text-embedding-ada-002")
        if dim is not None:
            assert dim == 1536

        # Test unknown model
        dim = get_model_dimension("unknown-model-xyz")
        assert dim is None

    def test_validate_embedding_dimensions(self) -> None:
        """Test embedding dimensions validation function."""
        # Test valid embeddings
        embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        # Should not raise
        validate_embedding_dimensions(embeddings, expected_dimension=3)

        # Test invalid dimensions

        with pytest.raises(DimensionMismatchError):
            validate_embedding_dimensions(embeddings, expected_dimension=4)

    def test_adjust_embeddings_dimension_truncation(self) -> None:
        """Test embedding dimension adjustment with truncation."""
        # Create embeddings with dimension 512
        embeddings = [
            list(range(512)),
            list(range(512, 1024))[:512],
        ]

        # Adjust to dimension 384
        adjusted = adjust_embeddings_dimension(embeddings, 384, normalize=False)

        assert len(adjusted) == 2
        assert all(len(emb) == 384 for emb in adjusted)
        assert adjusted[0] == list(range(384))

    def test_adjust_embeddings_dimension_padding(self) -> None:
        """Test embedding dimension adjustment with padding."""
        # Create embeddings with dimension 384
        embeddings = [
            list(range(384)),
            [1.0] * 384,
        ]

        # Adjust to dimension 512
        adjusted = adjust_embeddings_dimension(embeddings, 512, normalize=False)

        assert len(adjusted) == 2
        assert all(len(emb) == 512 for emb in adjusted)
        # Check padding with zeros
        assert adjusted[0][384:] == [0.0] * 128

    def test_adjust_embeddings_dimension_normalization(self) -> None:
        """Test embedding dimension adjustment with normalization."""
        # Create embeddings
        embeddings = [[3.0, 4.0]]  # Norm = 5

        # Adjust with normalization
        adjusted = adjust_embeddings_dimension(embeddings, 2, normalize=True)

        # Check normalization to unit length
        norm = sum(v**2 for v in adjusted[0]) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    @pytest.mark.asyncio()
    async def test_end_to_end_search_with_dimension_validation(self) -> None:
        """Test end-to-end search flow with dimension validation."""
        # This would be a full integration test with actual services
        # For now, we mock the key components

        pytest.skip("End-to-end test needs updating for new architecture")

        # Mock dependencies
        with patch("vecpipe.api.search_api.SearchService") as mock_search_service_class:
            mock_service = MagicMock()
            mock_search_service_class.return_value = mock_service

            # Mock successful search
            mock_service.search = AsyncMock(return_value={"results": [], "total": 0, "query_embedding_dimension": 384})

            # Perform search
            await search_collection(collection_id="test-collection", query="test query", limit=10, offset=0)

            # Verify search was called
            mock_service.search.assert_called_once()

    def test_dimension_validation_error_messages(self) -> None:
        """Test that dimension validation provides helpful error messages."""
        # Test dimension mismatch error
        embeddings = [[1.0] * 512]

        with pytest.raises(DimensionMismatchError) as exc_info:
            validate_embedding_dimensions(embeddings, expected_dimension=384)

        error_msg = str(exc_info.value)
        assert "512" in error_msg or "dimension" in error_msg.lower()

    @pytest.mark.asyncio()
    async def test_concurrent_dimension_validation(self) -> None:
        """Test dimension validation under concurrent load."""

        # Create multiple concurrent validation tasks

        async def validate_task(embeddings, dimension) -> None:
            try:
                validate_embedding_dimensions(embeddings, dimension)
                return True
            except DimensionMismatchError:
                return False

        # Run multiple validations concurrently
        test_cases = [
            ([[1.0] * 384], 384),  # Valid
            ([[1.0] * 512], 512),  # Valid
            ([[1.0] * 384], 512),  # Invalid
        ] * 5

        tasks = [validate_task(emb, dim) for emb, dim in test_cases]
        results = await asyncio.gather(*tasks)

        # Check expected results
        expected = [True, True, False] * 5
        assert results == expected

    def test_dimension_compatibility_matrix(self) -> None:
        """Test dimension compatibility between different models."""
        # Common embedding model dimensions
        model_dimensions = {
            "openai/text-embedding-ada-002": 1536,
            "openai/text-embedding-3-small": 1536,
            "openai/text-embedding-3-large": 3072,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
        }

        # Test compatibility checks
        for model1, dim1 in model_dimensions.items():
            for _, dim2 in model_dimensions.items():
                # Test compatibility

                try:
                    validate_dimension_compatibility(expected_dimension=dim2, actual_dimension=dim1, model_name=model1)
                    is_compatible = True
                except DimensionMismatchError:
                    is_compatible = False

                if dim1 == dim2:
                    assert is_compatible
                else:
                    assert not is_compatible

    def test_dimension_validation_performance(self) -> None:
        """Test performance of dimension validation operations."""

        # Create large batch of embeddings
        num_embeddings = 1000
        embedding_dim = 768
        embeddings = [list(range(embedding_dim)) for _ in range(num_embeddings)]

        # Time the adjustment operation
        start_time = time.time()
        adjusted = adjust_embeddings_dimension(embeddings, 512, normalize=False)
        elapsed_time = time.time() - start_time

        # Should be fast even for large batches
        assert elapsed_time < 1.0  # Should complete in under 1 second
        assert len(adjusted) == num_embeddings
        assert all(len(emb) == 512 for emb in adjusted)
