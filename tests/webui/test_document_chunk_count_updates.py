"""Test suite for Document.chunk_count updates during ingestion operations.

This test suite ensures that Document.chunk_count is properly updated
during APPEND and REINDEX operations with various chunking strategies.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import (
    Collection,
    CollectionStatus,
    Document,
    DocumentStatus,
    Operation,
    OperationStatus,
    OperationType,
)
from packages.webui.tasks import (
    _process_append_operation,
    _process_reindex_operation,
)


class TestDocumentChunkCountUpdates:
    """Test Document.chunk_count updates during task processing."""

    @pytest.fixture()
    def mock_db(self):
        """Create a mock database session."""
        db = AsyncMock(spec=AsyncSession)
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        db.rollback = AsyncMock()
        db.refresh = AsyncMock()
        db.flush = AsyncMock()
        return db

    @pytest.fixture()
    def mock_updater(self):
        """Create a mock operation updater."""
        updater = AsyncMock()
        updater.send_update = AsyncMock()
        return updater

    @pytest.fixture()
    def create_mock_document(self):
        """Factory fixture to create mock documents."""

        def _create(doc_id, file_path, chunk_count=0, status=DocumentStatus.PENDING):
            doc = MagicMock(spec=Document)
            doc.id = doc_id
            doc.file_path = file_path
            doc.file_size = 1024
            doc.chunk_count = chunk_count
            doc.status = status
            doc.created_at = datetime.now(UTC)
            doc.updated_at = datetime.now(UTC)
            return doc

        return _create

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("packages.webui.tasks.httpx")
    @patch("packages.webui.tasks.ChunkingService")
    async def test_append_updates_chunk_count_for_new_documents(
        self,
        mock_chunking_service_class,
        mock_httpx,
        mock_extract_and_serialize_thread_safe,
        mock_db,
        mock_updater,
        create_mock_document,
    ):
        """Test that APPEND correctly updates chunk_count for new documents."""
        # Setup operation
        operation = MagicMock(spec=Operation)
        operation.id = "op-append-1"
        operation.collection_id = "coll-1"
        operation.type = OperationType.APPEND
        operation.status = OperationStatus.PROCESSING
        operation.config = {"source_id": "source-1"}

        # Setup collection
        collection = MagicMock(spec=Collection)
        collection.id = "coll-1"
        collection.name = "Test Collection"
        collection.path = "/test/path"
        collection.vector_collection_id = "vc-1"
        collection.status = CollectionStatus.READY
        collection.chunking_strategy = "recursive"
        collection.chunking_config = {"chunk_size": 100}
        collection.chunk_size = 100
        collection.chunk_overlap = 20

        # Create documents with initial chunk_count = 0
        docs = [create_mock_document(f"doc-{i}", f"/test/file{i}.txt", chunk_count=0) for i in range(3)]

        # Setup database mocks
        # First call returns operation via scalar_one()
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        # Second call returns collection via scalar_one_or_none()
        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = collection

        # Third call returns documents via scalars().all()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = docs
        mock_result_3 = MagicMock()
        mock_result_3.scalars.return_value = mock_scalars

        # Set up execute to return different results for each call
        mock_db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3]

        # Setup text extraction
        mock_extract_and_serialize_thread_safe.return_value = [("Sample text content", {})]

        # Setup ChunkingService to return different chunk counts
        mock_chunking_service = MagicMock()
        mock_chunking_service_class.return_value = mock_chunking_service

        chunk_results = [
            {
                "chunks": [
                    {"chunk_id": f"doc-0_chunk_{i:04d}", "text": f"chunk {i}", "metadata": {}} for i in range(5)
                ],
                "stats": {"chunk_count": 5, "strategy_used": "recursive", "fallback": False},
            },
            {
                "chunks": [
                    {"chunk_id": f"doc-1_chunk_{i:04d}", "text": f"chunk {i}", "metadata": {}} for i in range(3)
                ],
                "stats": {"chunk_count": 3, "strategy_used": "recursive", "fallback": False},
            },
            {
                "chunks": [
                    {"chunk_id": f"doc-2_chunk_{i:04d}", "text": f"chunk {i}", "metadata": {}} for i in range(8)
                ],
                "stats": {"chunk_count": 8, "strategy_used": "recursive", "fallback": False},
            },
        ]
        mock_chunking_service.execute_ingestion_chunking = AsyncMock(side_effect=chunk_results)

        # Setup httpx mock for vecpipe endpoints
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

        # Run APPEND operation
        await _process_append_operation(mock_db, mock_updater, "op-append-1")

        # Verify chunk counts were updated
        assert docs[0].chunk_count == 5
        assert docs[1].chunk_count == 3
        assert docs[2].chunk_count == 8

        # Verify documents were marked as completed
        for doc in docs:
            assert doc.status == DocumentStatus.COMPLETED

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("packages.webui.tasks.ChunkingService")
    async def test_append_preserves_chunk_count_on_failure(
        self,
        mock_chunking_service_class,
        mock_extract_and_serialize_thread_safe,
        mock_db,
        mock_updater,
        create_mock_document,
    ):
        """Test that chunk_count is not updated when chunking fails."""
        # Setup operation and collection
        operation = MagicMock(spec=Operation)
        operation.id = "op-fail-1"
        operation.collection_id = "coll-1"
        operation.type = OperationType.APPEND
        operation.status = OperationStatus.PROCESSING
        operation.config = {"source_id": "source-1"}

        collection = MagicMock(spec=Collection)
        collection.id = "coll-1"
        collection.vector_collection_id = "vc-1"
        collection.status = CollectionStatus.READY
        collection.chunk_size = 100
        collection.chunk_overlap = 20

        # Create document with existing chunk_count
        doc = create_mock_document("doc-fail", "/test/fail.txt", chunk_count=10)

        # Setup database mocks with proper side_effect
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = collection

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [doc]
        mock_result_3 = MagicMock()
        mock_result_3.scalars.return_value = mock_scalars

        mock_db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3]

        mock_extract_and_serialize_thread_safe.return_value = [("Text content", {})]

        # Make chunking service raise an exception
        mock_chunking_service = MagicMock()
        mock_chunking_service_class.return_value = mock_chunking_service
        mock_chunking_service.execute_ingestion_chunking = AsyncMock(side_effect=RuntimeError("Chunking failed"))

        # Run APPEND operation (should fail)
        with pytest.raises(RuntimeError):
            await _process_append_operation(mock_db, mock_updater, "op-fail-1")

        # Verify chunk_count was not changed
        assert doc.chunk_count == 10
        assert doc.status == DocumentStatus.FAILED

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("packages.webui.tasks.httpx")
    @patch("packages.webui.tasks.ChunkingService")
    async def test_reindex_updates_chunk_count_correctly(
        self,
        mock_chunking_service_class,
        mock_httpx,
        mock_extract_and_serialize_thread_safe,
        mock_db,
        mock_updater,
        create_mock_document,
    ):
        """Test that REINDEX correctly updates chunk_count with new strategy."""
        # Setup operation with new chunking config
        operation = MagicMock(spec=Operation)
        operation.id = "op-reindex-1"
        operation.collection_id = "coll-1"
        operation.type = OperationType.REINDEX
        operation.status = OperationStatus.PROCESSING
        operation.config = {
            "chunking_strategy": "semantic",
            "chunking_config": {"buffer_size": 1},
            "chunk_size": 200,
            "chunk_overlap": 40,
        }

        # Setup source collection
        source_collection = MagicMock(spec=Collection)
        source_collection.id = "coll-1"
        source_collection.name = "Source Collection"
        source_collection.vector_collection_id = "vc-source"
        source_collection.status = CollectionStatus.READY
        source_collection.chunking_strategy = "recursive"
        source_collection.chunk_size = 100
        source_collection.chunk_overlap = 20

        # Setup staging collection
        staging_collection = MagicMock(spec=Collection)
        staging_collection.id = "coll-staging-1"
        staging_collection.name = "Source Collection (staging)"
        staging_collection.vector_collection_id = "vc-staging"
        staging_collection.status = CollectionStatus.PROCESSING
        staging_collection.parent_collection_id = "coll-1"

        # Create documents with old chunk counts
        docs = [
            create_mock_document("doc-1", "/test/file1.txt", chunk_count=10, status=DocumentStatus.COMPLETED),
            create_mock_document("doc-2", "/test/file2.txt", chunk_count=15, status=DocumentStatus.COMPLETED),
        ]

        # Setup database mocks for reindex (4 calls to execute)
        # First call returns operation via scalar_one()
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        # Second call returns source_collection via scalar_one_or_none()
        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = source_collection

        # Third call returns staging_collection via scalar_one_or_none()
        mock_result_3 = MagicMock()
        mock_result_3.scalar_one_or_none.return_value = staging_collection

        # Fourth call returns docs via scalars().all()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = docs
        mock_result_4 = MagicMock()
        mock_result_4.scalars.return_value = mock_scalars

        mock_db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3, mock_result_4]

        mock_extract_and_serialize_thread_safe.return_value = [("Reindexed content", {})]

        # Setup ChunkingService with new chunk counts (semantic strategy produces fewer chunks)
        mock_chunking_service = MagicMock()
        mock_chunking_service_class.return_value = mock_chunking_service

        chunk_results = [
            {
                "chunks": [
                    {"chunk_id": f"doc-1_chunk_{i:04d}", "text": f"semantic chunk {i}", "metadata": {}}
                    for i in range(3)
                ],
                "stats": {"chunk_count": 3, "strategy_used": "semantic", "fallback": False},
            },
            {
                "chunks": [
                    {"chunk_id": f"doc-2_chunk_{i:04d}", "text": f"semantic chunk {i}", "metadata": {}}
                    for i in range(4)
                ],
                "stats": {"chunk_count": 4, "strategy_used": "semantic", "fallback": False},
            },
        ]
        mock_chunking_service.execute_ingestion_chunking = AsyncMock(side_effect=chunk_results)

        # Setup httpx mock for vecpipe endpoints
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

        # Run REINDEX operation
        await _process_reindex_operation(mock_db, mock_updater, "op-reindex-1")

        # Verify chunk counts were updated with new values
        assert docs[0].chunk_count == 3  # Was 10, now 3
        assert docs[1].chunk_count == 4  # Was 15, now 4

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("packages.webui.tasks.httpx")
    @patch("packages.webui.tasks.ChunkingService")
    async def test_chunk_count_zero_for_empty_documents(
        self,
        mock_chunking_service_class,
        mock_httpx,
        mock_extract_and_serialize_thread_safe,
        mock_db,
        mock_updater,
        create_mock_document,
    ):
        """Test that documents with no extractable text have chunk_count = 0."""
        operation = MagicMock(spec=Operation)
        operation.id = "op-empty-1"
        operation.collection_id = "coll-1"
        operation.type = OperationType.APPEND
        operation.status = OperationStatus.PROCESSING
        operation.config = {"source_id": "source-1"}

        collection = MagicMock(spec=Collection)
        collection.id = "coll-1"
        collection.vector_collection_id = "vc-1"
        collection.status = CollectionStatus.READY
        collection.chunk_size = 100
        collection.chunk_overlap = 20

        doc = create_mock_document("doc-empty", "/test/empty.txt", chunk_count=5)

        # Setup database mocks with proper side_effect
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = collection

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [doc]
        mock_result_3 = MagicMock()
        mock_result_3.scalars.return_value = mock_scalars

        mock_db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3]

        # Return empty text blocks
        mock_extract_and_serialize_thread_safe.return_value = []

        # Setup httpx mock for vecpipe endpoints
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

        # Run APPEND operation
        await _process_append_operation(mock_db, mock_updater, "op-empty-1")

        # Verify chunk_count is 0 for empty document
        assert doc.chunk_count == 0
        assert doc.status == DocumentStatus.COMPLETED

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("packages.webui.tasks.httpx")
    @patch("packages.webui.tasks.ChunkingService")
    async def test_chunk_count_with_fallback_strategy(
        self,
        mock_chunking_service_class,
        mock_httpx,
        mock_extract_and_serialize_thread_safe,
        mock_db,
        mock_updater,
        create_mock_document,
    ):
        """Test that chunk_count is correctly updated even when fallback to TokenChunker occurs."""
        operation = MagicMock(spec=Operation)
        operation.id = "op-fallback-1"
        operation.collection_id = "coll-1"
        operation.type = OperationType.APPEND
        operation.status = OperationStatus.PROCESSING
        operation.config = {"source_id": "source-1"}

        collection = MagicMock(spec=Collection)
        collection.id = "coll-1"
        collection.vector_collection_id = "vc-1"
        collection.status = CollectionStatus.READY
        collection.chunking_strategy = "invalid_strategy"
        collection.chunk_size = 100
        collection.chunk_overlap = 20

        doc = create_mock_document("doc-fallback", "/test/fallback.txt", chunk_count=0)

        # Setup database mocks with proper side_effect
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = collection

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [doc]
        mock_result_3 = MagicMock()
        mock_result_3.scalars.return_value = mock_scalars

        mock_db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3]

        mock_extract_and_serialize_thread_safe.return_value = [("Text for fallback chunking", {})]

        # Setup ChunkingService to simulate fallback
        mock_chunking_service = MagicMock()
        mock_chunking_service_class.return_value = mock_chunking_service
        mock_chunking_service.execute_ingestion_chunking = AsyncMock(
            return_value={
                "chunks": [
                    {"chunk_id": f"doc-fallback_chunk_{i:04d}", "text": f"fallback chunk {i}", "metadata": {}}
                    for i in range(7)
                ],
                "stats": {"chunk_count": 7, "strategy_used": "TokenChunker", "fallback": True},
            }
        )

        # Setup httpx mock for vecpipe endpoints
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

        # Run APPEND operation
        await _process_append_operation(mock_db, mock_updater, "op-fallback-1")

        # Verify chunk_count was updated correctly even with fallback
        assert doc.chunk_count == 7
        assert doc.status == DocumentStatus.COMPLETED

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("packages.webui.tasks.httpx")
    @patch("packages.webui.tasks.ChunkingService")
    async def test_batch_document_chunk_count_updates(
        self,
        mock_chunking_service_class,
        mock_httpx,
        mock_extract_and_serialize_thread_safe,
        mock_db,
        mock_updater,
        create_mock_document,
    ):
        """Test that chunk_count is updated correctly for large batches of documents."""
        operation = MagicMock(spec=Operation)
        operation.id = "op-batch-1"
        operation.collection_id = "coll-1"
        operation.type = OperationType.APPEND
        operation.status = OperationStatus.PROCESSING
        operation.config = {"source_id": "source-1"}

        collection = MagicMock(spec=Collection)
        collection.id = "coll-1"
        collection.vector_collection_id = "vc-1"
        collection.status = CollectionStatus.READY
        collection.chunking_strategy = "recursive"
        collection.chunk_size = 100
        collection.chunk_overlap = 20

        # Create a large batch of documents
        num_docs = 50
        docs = [create_mock_document(f"doc-{i}", f"/test/file{i}.txt", chunk_count=0) for i in range(num_docs)]

        # Setup database mocks with proper side_effect
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = collection

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = docs
        mock_result_3 = MagicMock()
        mock_result_3.scalars.return_value = mock_scalars

        mock_db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3]

        mock_extract_and_serialize_thread_safe.return_value = [("Batch document content", {})]

        # Setup ChunkingService to return varying chunk counts
        mock_chunking_service = MagicMock()
        mock_chunking_service_class.return_value = mock_chunking_service

        chunk_results = []
        total_chunks = 0
        for i in range(num_docs):
            chunk_count = (i % 10) + 1  # 1 to 10 chunks per document
            chunks = [
                {"chunk_id": f"doc-{i}_chunk_{j:04d}", "text": f"chunk {j}", "metadata": {}} for j in range(chunk_count)
            ]
            chunk_results.append(
                {
                    "chunks": chunks,
                    "stats": {"chunk_count": chunk_count, "strategy_used": "recursive", "fallback": False},
                }
            )
            total_chunks += chunk_count

        mock_chunking_service.execute_ingestion_chunking = AsyncMock(side_effect=chunk_results)

        # Setup httpx mock for vecpipe endpoints
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

        # Run APPEND operation
        await _process_append_operation(mock_db, mock_updater, "op-batch-1")

        # Verify all documents have correct chunk counts
        for i, doc in enumerate(docs):
            expected_chunk_count = (i % 10) + 1
            assert doc.chunk_count == expected_chunk_count
            assert doc.status == DocumentStatus.COMPLETED

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("packages.webui.tasks.ChunkingService")
    async def test_chunk_count_persistence_across_retries(
        self,
        mock_chunking_service_class,
        mock_extract_and_serialize_thread_safe,
        mock_db,
        mock_updater,
        create_mock_document,
    ):
        """Test that chunk_count updates are persisted even if later documents fail."""
        operation = MagicMock(spec=Operation)
        operation.id = "op-retry-1"
        operation.collection_id = "coll-1"
        operation.type = OperationType.APPEND
        operation.status = OperationStatus.PROCESSING
        operation.config = {"source_id": "source-1"}

        collection = MagicMock(spec=Collection)
        collection.id = "coll-1"
        collection.vector_collection_id = "vc-1"
        collection.status = CollectionStatus.READY
        collection.chunk_size = 100
        collection.chunk_overlap = 20

        # Create multiple documents
        docs = [create_mock_document(f"doc-{i}", f"/test/file{i}.txt", chunk_count=0) for i in range(3)]

        # Setup database mocks with proper side_effect
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = collection

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = docs
        mock_result_3 = MagicMock()
        mock_result_3.scalars.return_value = mock_scalars

        mock_db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3]

        mock_extract_and_serialize_thread_safe.return_value = [("Content", {})]

        # Setup ChunkingService - succeed for first two, fail for third
        mock_chunking_service = MagicMock()
        mock_chunking_service_class.return_value = mock_chunking_service

        chunk_results = [
            {
                "chunks": [{"chunk_id": "doc-0_chunk_0000", "text": "chunk", "metadata": {}}],
                "stats": {"chunk_count": 1, "strategy_used": "recursive", "fallback": False},
            },
            {
                "chunks": [
                    {"chunk_id": "doc-1_chunk_0000", "text": "chunk 0", "metadata": {}},
                    {"chunk_id": "doc-1_chunk_0001", "text": "chunk 1", "metadata": {}}
                ],
                "stats": {"chunk_count": 2, "strategy_used": "recursive", "fallback": False},
            },
            RuntimeError("Failed to chunk third document"),
        ]

        call_count = 0

        async def chunking_side_effect(*args, **kwargs):  # noqa: ARG001
            nonlocal call_count
            result = chunk_results[call_count]
            call_count += 1
            if isinstance(result, RuntimeError):
                raise result
            return result

        mock_chunking_service.execute_ingestion_chunking = AsyncMock(side_effect=chunking_side_effect)

        with patch("packages.webui.tasks.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=200))
            mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

            # Run APPEND operation (should partially fail)
            with pytest.raises(RuntimeError):
                await _process_append_operation(mock_db, mock_updater, "op-retry-1")

        # Verify first two documents have updated chunk counts
        assert docs[0].chunk_count == 1
        assert docs[1].chunk_count == 2
        # Third document should retain original count due to failure
        assert docs[2].chunk_count == 0

        # Verify status updates
        assert docs[0].status == DocumentStatus.COMPLETED
        assert docs[1].status == DocumentStatus.COMPLETED
        assert docs[2].status == DocumentStatus.FAILED
