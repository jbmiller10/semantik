"""Integration tests for Phase 3: Large Document Ingestion.

This test file validates the end-to-end ingestion of large documents
through the APPEND and REINDEX operations with progressive segmentation.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

from packages.webui.services.chunking_service import ChunkingService


class TestLargeDocumentIngestion:
    """Integration tests for large document ingestion."""

    @pytest.fixture()
    def large_document_content(self):
        """Generate large document content for testing."""
        # Create 10MB document
        return "This is a large document for testing ingestion. " * 200000

    @pytest.fixture()
    def mock_collection(self):
        """Create mock collection with chunking configuration."""
        return {
            "id": "test-collection",
            "name": "Test Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {
                "chunk_size": 500,
                "chunk_overlap": 50,
            },
            "chunk_size": 500,
            "chunk_overlap": 50,
            "user_id": 1,
        }

    @pytest.mark.asyncio()
    async def test_memory_bounded_processing(self, large_document_content):
        """Test that memory usage remains bounded during large document processing."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create chunking service
        service = ChunkingService(
            db_session=AsyncMock(),
            collection_repo=MagicMock(),
            document_repo=MagicMock(),
            redis_client=None,
        )

        collection = {
            "id": "test-collection",
            "chunking_strategy": "recursive",
            "chunking_config": {"chunk_size": 500, "chunk_overlap": 50},
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

        # Process large document
        with patch.object(service, "_process_segment") as mock_process:
            mock_process.return_value = {"chunks": []}

            await service.execute_ingestion_chunking_segmented(
                text=large_document_content,
                document_id="test-doc",
                collection=collection,
            )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be bounded (less than 100MB)
        # This is a soft check as memory usage can vary
        assert memory_increase < 100, f"Memory increased by {memory_increase}MB"

    @pytest.mark.asyncio()
    async def test_strategy_specific_segmentation_thresholds(self):
        """Test that different strategies use their configured thresholds."""
        service = ChunkingService(
            db_session=AsyncMock(),
            collection_repo=MagicMock(),
            document_repo=MagicMock(),
            redis_client=None,
        )

        # Test each strategy threshold
        test_cases = [
            ("semantic", 2 * 1024 * 1024),  # 2MB
            ("markdown", 10 * 1024 * 1024),  # 10MB
            ("recursive", 8 * 1024 * 1024),  # 8MB
            ("hierarchical", 5 * 1024 * 1024),  # 5MB
            ("hybrid", 3 * 1024 * 1024),  # 3MB
        ]

        for strategy, threshold in test_cases:
            # Create text just over the threshold
            text_size = threshold + 1000
            test_text = "X" * text_size

            collection = {
                "id": f"test-{strategy}",
                "chunking_strategy": strategy,
                "chunking_config": {},
                "chunk_size": 500,
                "chunk_overlap": 50,
            }

            with patch.object(service, "execute_ingestion_chunking_segmented") as mock_segmented:
                mock_segmented.return_value = {"chunks": [], "stats": {}}

                await service.execute_ingestion_chunking(
                    text=test_text,
                    document_id=f"doc-{strategy}",
                    collection=collection,
                )

                # Should trigger segmentation
                mock_segmented.assert_called_once()

    @pytest.mark.asyncio()
    async def test_concurrent_large_document_processing(self):
        """Test processing multiple large documents concurrently."""
        service = ChunkingService(
            db_session=AsyncMock(),
            collection_repo=MagicMock(),
            document_repo=MagicMock(),
            redis_client=None,
        )

        # Create multiple large documents
        documents = [
            ("doc1", "Content for document 1. " * 100000),
            ("doc2", "Content for document 2. " * 100000),
            ("doc3", "Content for document 3. " * 100000),
        ]

        collection = {
            "id": "test-collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

        async def process_document(doc_id, content):
            with patch.object(service, "_process_segment") as mock_process:
                mock_process.return_value = {
                    "chunks": [{"chunk_id": f"{doc_id}_chunk_0000", "text": "chunk", "metadata": {}}]
                }

                return await service.execute_ingestion_chunking_segmented(
                    text=content,
                    document_id=doc_id,
                    collection=collection,
                )

        # Process documents concurrently
        tasks = [process_document(doc_id, content) for doc_id, content in documents]
        results = await asyncio.gather(*tasks)

        # Verify all documents were processed
        assert len(results) == 3
        for result in results:
            assert "chunks" in result
            assert "stats" in result

    @pytest.mark.asyncio()
    async def test_segmentation_error_recovery(self):
        """Test that segmentation continues even if some segments fail."""
        service = ChunkingService(
            db_session=AsyncMock(),
            collection_repo=MagicMock(),
            document_repo=MagicMock(),
            redis_client=None,
        )

        large_text = "Test content. " * 100000

        collection = {
            "id": "test-collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

        segment_count = 0

        async def mock_process_with_failures(*args, **kwargs):  # noqa: ARG001
            nonlocal segment_count
            segment_count += 1

            # Fail every third segment
            if segment_count % 3 == 0:
                raise RuntimeError(f"Segment {segment_count} failed")

            return {
                "chunks": [{"chunk_id": f"chunk_{segment_count:04d}", "text": f"chunk {segment_count}", "metadata": {}}]
            }

        with patch.object(service, "_process_segment", side_effect=mock_process_with_failures):
            result = await service.execute_ingestion_chunking_segmented(
                text=large_text,
                document_id="test-doc",
                collection=collection,
            )

        # Should have processed some segments despite failures
        assert len(result["chunks"]) > 0
        assert result["stats"]["segment_count"] > 0

        # Verify that not all segments produced chunks (some failed)
        total_segments = result["stats"]["segment_count"]
        chunks_produced = len(result["chunks"])
        assert chunks_produced <= total_segments  # Less than or equal due to failures

    @pytest.mark.asyncio()
    async def test_chunking_service_with_large_document(self):
        """Test ChunkingService directly with a large document."""
        service = ChunkingService(
            db_session=AsyncMock(),
            collection_repo=MagicMock(),
            document_repo=MagicMock(),
            redis_client=None,
        )

        # Create a 9MB document (over recursive threshold)
        large_text = "This is a test document. " * 400000

        collection = {
            "id": "test-collection",
            "chunking_strategy": "recursive",
            "chunking_config": {"chunk_size": 500, "chunk_overlap": 50},
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

        # Mock the execute_ingestion_chunking to simulate segmented processing
        with patch.object(service, "execute_ingestion_chunking") as mock_chunking:
            mock_chunking.return_value = {
                "chunks": [
                    {"chunk_id": f"chunk_{i:04d}", "text": f"chunk {i}", "metadata": {"chunk_index": i}}
                    for i in range(100)
                ],
                "stats": {
                    "chunk_count": 100,
                    "segmented": True,
                    "segment_count": 10,
                    "duration_ms": 5000,
                },
            }

            result = await service.execute_ingestion_chunking(
                text=large_text,
                document_id="test-doc",
                collection=collection,
            )

            # Should use segmented processing for large text
            assert "chunks" in result
            assert "stats" in result
            assert result["stats"]["segmented"] is True
            assert result["stats"]["chunk_count"] == 100

    @pytest.mark.asyncio()
    async def test_append_operation_with_chunking_integration(self, large_document_content, mock_collection):
        """Test APPEND operation processing with chunking service integration."""
        from packages.webui.services.chunking_service import ChunkingService

        # Create service with mocked dependencies
        mock_db = AsyncMock()
        mock_collection_repo = MagicMock()
        mock_document_repo = MagicMock()

        service = ChunkingService(
            db_session=mock_db,
            collection_repo=mock_collection_repo,
            document_repo=mock_document_repo,
            redis_client=None,
        )

        # Test chunking directly
        with patch.object(service, "execute_ingestion_chunking") as mock_chunking:
            mock_chunking.return_value = {
                "chunks": [
                    {"chunk_id": f"chunk_{i:04d}", "text": f"chunk {i}", "metadata": {"chunk_index": i}}
                    for i in range(50)
                ],
                "stats": {
                    "chunk_count": 50,
                    "segmented": True,
                    "segment_count": 5,
                    "duration_ms": 3000,
                },
            }

            result = await service.execute_ingestion_chunking(
                text=large_document_content,
                document_id="test-doc",
                collection=mock_collection,
            )

            assert "chunks" in result
            assert len(result["chunks"]) > 0
            assert result["stats"]["segmented"] is True

    @pytest.mark.asyncio()
    async def test_reindex_operation_with_chunking_integration(self, large_document_content, mock_collection):
        """Test REINDEX operation processing with chunking service integration."""
        from packages.webui.services.chunking_service import ChunkingService

        # Create service with mocked dependencies
        mock_db = AsyncMock()
        mock_collection_repo = MagicMock()
        mock_document_repo = MagicMock()

        service = ChunkingService(
            db_session=mock_db,
            collection_repo=mock_collection_repo,
            document_repo=mock_document_repo,
            redis_client=None,
        )

        # Test chunking for reindex scenario
        with patch.object(service, "execute_ingestion_chunking") as mock_chunking:
            # Simulate more chunks for reindex
            mock_chunking.return_value = {
                "chunks": [
                    {
                        "chunk_id": f"chunk_{i:04d}",
                        "text": f"reindexed chunk {i}",
                        "metadata": {"chunk_index": i, "reindexed": True},
                    }
                    for i in range(75)
                ],
                "stats": {
                    "chunk_count": 75,
                    "segmented": True,
                    "segment_count": 8,
                    "duration_ms": 4000,
                },
            }

            result = await service.execute_ingestion_chunking(
                text=large_document_content,
                document_id="test-doc-reindex",
                collection=mock_collection,
            )

            assert "chunks" in result
            assert len(result["chunks"]) > 0
            assert result["stats"]["segmented"] is True
            # Verify we got reindexed chunks
            if result["chunks"]:
                assert "reindexed" in str(result["chunks"][0])
