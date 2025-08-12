"""Integration tests for Phase 3: Large Document Ingestion.

This test file validates the end-to-end ingestion of large documents
through the APPEND and REINDEX operations with progressive segmentation.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from packages.webui.tasks import append_operation_task, reindex_operation_task


class TestLargeDocumentIngestion:
    """Integration tests for large document ingestion."""

    @pytest.fixture()
    def large_document_content(self):
        """Generate large document content for testing."""
        # Create 10MB document
        return "This is a large document for testing ingestion. " * 200000

    @pytest.fixture()
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock(spec=AsyncSession)

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

    @pytest.fixture()
    def mock_document(self):
        """Create mock document."""
        doc = MagicMock()
        doc.id = "test-doc-001"
        doc.collection_id = "test-collection"
        doc.file_path = "/test/path/document.txt"
        doc.status = "pending"
        doc.chunk_count = 0
        return doc

    @pytest.mark.asyncio()
    async def test_append_large_document_with_segmentation(
        self, large_document_content, mock_collection, mock_document
    ):
        """Test APPEND operation with large document triggers segmentation."""

        with (
            patch("packages.webui.tasks.get_async_session_context") as mock_session_ctx,
            patch("packages.webui.tasks.CollectionRepository") as mock_collection_repo_class,
            patch("packages.webui.tasks.DocumentRepository") as mock_document_repo_class,
            patch("packages.webui.tasks.OperationRepository") as mock_operation_repo_class,
            patch("packages.webui.tasks.create_celery_chunking_service_with_repos") as mock_create_service,
            patch("packages.webui.tasks.extract_text") as mock_extract,
            patch("packages.webui.tasks.embed_and_upsert") as mock_embed,
        ):

            # Setup mocks
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            mock_col_repo = MagicMock()
            mock_col_repo.get_by_id.return_value = mock_collection
            mock_collection_repo_class.return_value = mock_col_repo

            mock_doc_repo = MagicMock()
            mock_doc_repo.get_documents_for_operation.return_value = [mock_document]
            mock_doc_repo.update_document_counts = AsyncMock()
            mock_document_repo_class.return_value = mock_doc_repo

            mock_op_repo = MagicMock()
            mock_op = MagicMock()
            mock_op.id = "test-op-001"
            mock_op.collection_id = "test-collection"
            mock_op.status = "processing"
            mock_op_repo.get_by_id.return_value = mock_op
            mock_operation_repo_class.return_value = mock_op_repo

            # Mock chunking service
            mock_chunking_service = MagicMock()
            mock_chunking_service.execute_ingestion_chunking = AsyncMock()
            mock_chunking_service.execute_ingestion_chunking.return_value = {
                "chunks": [
                    {"chunk_id": f"doc_chunk_{i:04d}", "text": f"chunk {i}", "metadata": {}}
                    for i in range(1000)  # Many chunks from segmentation
                ],
                "stats": {
                    "duration_ms": 5000,
                    "strategy_used": "recursive",
                    "chunk_count": 1000,
                    "segment_count": 10,
                    "segmented": True,
                },
            }
            mock_create_service.return_value = mock_chunking_service

            # Mock extraction to return large content
            mock_extract.return_value = [
                {
                    "text": large_document_content,
                    "metadata": {"page": 1},
                }
            ]

            # Mock embedding
            mock_embed.return_value = (1000, 0)  # 1000 successful, 0 failed

            # Execute APPEND operation
            await append_operation_task("test-op-001")

            # Verify chunking was called
            mock_chunking_service.execute_ingestion_chunking.assert_called_once()
            call_args = mock_chunking_service.execute_ingestion_chunking.call_args[1]
            assert len(call_args["text"]) > 5000000  # Large text

            # Verify document chunk count was updated
            assert mock_document.chunk_count == 1000

    @pytest.mark.asyncio()
    async def test_reindex_large_document_with_segmentation(
        self, large_document_content, mock_collection, mock_document
    ):
        """Test REINDEX operation with large document triggers segmentation."""

        with (
            patch("packages.webui.tasks.get_async_session_context") as mock_session_ctx,
            patch("packages.webui.tasks.CollectionRepository") as mock_collection_repo_class,
            patch("packages.webui.tasks.DocumentRepository") as mock_document_repo_class,
            patch("packages.webui.tasks.OperationRepository") as mock_operation_repo_class,
            patch("packages.webui.tasks.create_celery_chunking_service_with_repos") as mock_create_service,
            patch("packages.webui.tasks.extract_text") as mock_extract,
            patch("packages.webui.tasks.embed_and_upsert") as mock_embed,
            patch("packages.webui.tasks.QdrantManager") as mock_qdrant_manager_class,
        ):

            # Setup mocks
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            mock_col_repo = MagicMock()
            mock_col_repo.get_by_id.return_value = mock_collection
            mock_collection_repo_class.return_value = mock_col_repo

            mock_doc_repo = MagicMock()
            mock_doc_repo.get_documents_for_operation.return_value = [mock_document]
            mock_doc_repo.update_document_counts = AsyncMock()
            mock_document_repo_class.return_value = mock_doc_repo

            mock_op_repo = MagicMock()
            mock_op = MagicMock()
            mock_op.id = "test-reindex-op"
            mock_op.collection_id = "test-collection"
            mock_op.status = "processing"
            mock_op_repo.get_by_id.return_value = mock_op
            mock_operation_repo_class.return_value = mock_op_repo

            # Mock Qdrant for staging collection
            mock_qdrant = MagicMock()
            mock_qdrant.ensure_staging_collection = AsyncMock()
            mock_qdrant.swap_collections = AsyncMock()
            mock_qdrant_manager_class.return_value = mock_qdrant

            # Mock chunking service
            mock_chunking_service = MagicMock()
            mock_chunking_service.execute_ingestion_chunking = AsyncMock()
            mock_chunking_service.execute_ingestion_chunking.return_value = {
                "chunks": [
                    {"chunk_id": f"doc_chunk_{i:04d}", "text": f"chunk {i}", "metadata": {}}
                    for i in range(1500)  # Even more chunks from large doc
                ],
                "stats": {
                    "duration_ms": 7000,
                    "strategy_used": "recursive",
                    "chunk_count": 1500,
                    "segment_count": 15,
                    "segmented": True,
                },
            }
            mock_create_service.return_value = mock_chunking_service

            # Mock extraction to return large content
            mock_extract.return_value = [
                {
                    "text": large_document_content,
                    "metadata": {"page": 1},
                }
            ]

            # Mock embedding
            mock_embed.return_value = (1500, 0)  # 1500 successful, 0 failed

            # Execute REINDEX operation
            await reindex_operation_task("test-reindex-op")

            # Verify chunking was called with large text
            mock_chunking_service.execute_ingestion_chunking.assert_called()

            # Verify staging collection was created
            mock_qdrant.ensure_staging_collection.assert_called()

            # Verify collections were swapped after successful indexing
            mock_qdrant.swap_collections.assert_called()

    @pytest.mark.asyncio()
    async def test_memory_bounded_processing(self, large_document_content):
        """Test that memory usage remains bounded during large document processing."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create chunking service
        from packages.webui.services.chunking_service import ChunkingService

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
        from packages.webui.services.chunking_service import ChunkingService

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
        from packages.webui.services.chunking_service import ChunkingService

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
        from packages.webui.services.chunking_service import ChunkingService

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
        assert chunks_produced < total_segments  # Some segments failed
