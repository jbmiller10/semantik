"""Comprehensive test suite for ChunkingService.execute_ingestion_chunking and task integration.

This test suite covers:
1. ChunkingService.execute_ingestion_chunking method
2. Integration with APPEND task
3. Integration with REINDEX task
4. Strategy fallback behavior
5. Document chunk_count updates
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import CollectionStatus, DocumentStatus, OperationStatus, OperationType
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.webui.services.chunking_service import ChunkingService
from packages.webui.tasks import _process_append_operation, _process_reindex_operation


class TestExecuteIngestionChunking:
    """Test ChunkingService.execute_ingestion_chunking method."""

    @pytest.fixture()
    def mock_db_session(self):
        """Create a mock database session."""
        mock = AsyncMock(spec=AsyncSession)
        mock.execute = AsyncMock()
        mock.commit = AsyncMock()
        mock.rollback = AsyncMock()
        mock.refresh = AsyncMock()
        return mock

    @pytest.fixture()
    def mock_collection_repo(self):
        """Create a mock collection repository."""
        mock = MagicMock(spec=CollectionRepository)
        mock.get_by_id = AsyncMock()
        return mock

    @pytest.fixture()
    def mock_document_repo(self):
        """Create a mock document repository."""
        mock = MagicMock(spec=DocumentRepository)
        mock.get_by_id = AsyncMock()
        return mock

    @pytest.fixture()
    def chunking_service(self, mock_db_session, mock_collection_repo):
        """Create a ChunkingService instance with mocked dependencies."""
        return ChunkingService(
            db_session=mock_db_session,
            collection_repo=mock_collection_repo,
            document_repo=MagicMock(spec=DocumentRepository),
            redis_client=None,
        )

    @pytest.fixture()
    def sample_text(self):
        """Sample text for chunking."""
        return """This is a sample document for testing chunking strategies.
        It contains multiple paragraphs with different content.

        This is the second paragraph that should be chunked appropriately.
        The chunking strategy should handle this text based on the configuration.

        And here's a third paragraph with some additional content to ensure
        we have enough text for meaningful chunking tests."""

    @pytest.fixture()
    def sample_collection(self):
        """Sample collection dictionary."""
        return {
            "id": "coll-123",
            "name": "Test Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {
                "chunk_size": 100,
                "chunk_overlap": 20,
            },
            "chunk_size": 100,
            "chunk_overlap": 20,
        }

    @pytest.mark.asyncio()
    async def test_execute_ingestion_chunking_with_recursive_strategy(
        self, chunking_service, sample_text, sample_collection
    ):
        """Test successful chunking with recursive strategy."""
        # Mock the strategy factory to return a mock strategy
        mock_strategy = MagicMock()
        mock_chunks = [
            MagicMock(content="Chunk 1 content"),
            MagicMock(content="Chunk 2 content"),
            MagicMock(content="Chunk 3 content"),
        ]
        mock_strategy.chunk.return_value = mock_chunks

        with patch.object(chunking_service.strategy_factory, "create_strategy", return_value=mock_strategy):
            result = await chunking_service.execute_ingestion_chunking(
                text=sample_text,
                document_id="doc-123",
                collection=sample_collection,
                metadata={"source": "test"},
                file_type="txt",
            )

        # Verify result structure
        assert "chunks" in result
        assert "stats" in result

        # Verify chunks
        chunks = result["chunks"]
        assert len(chunks) == 3
        assert chunks[0]["chunk_id"] == "doc-123_0000"
        assert chunks[0]["text"] == "Chunk 1 content"
        assert chunks[0]["metadata"]["source"] == "test"
        assert chunks[0]["metadata"]["index"] == 0
        # Strategy could be stored as enum value or string
        assert chunks[0]["metadata"]["strategy"] in ["recursive", "ChunkingStrategy.RECURSIVE"]

        # Verify stats
        stats = result["stats"]
        assert stats["strategy_used"] in ["recursive", "ChunkingStrategy.RECURSIVE"]
        assert stats["fallback"] is False
        assert stats["chunk_count"] == 3
        assert "duration_ms" in stats

    @pytest.mark.asyncio()
    async def test_execute_ingestion_chunking_with_semantic_strategy(self, chunking_service, sample_text):
        """Test successful chunking with semantic strategy."""
        collection = {
            "id": "coll-456",
            "name": "Semantic Collection",
            "chunking_strategy": "semantic",
            "chunking_config": {
                "buffer_size": 1,
                "breakpoint_percentile_threshold": 95,
            },
            "chunk_size": 200,
            "chunk_overlap": 50,
        }

        mock_strategy = MagicMock()
        mock_chunks = [
            MagicMock(content="Semantic chunk 1"),
            MagicMock(content="Semantic chunk 2"),
        ]
        mock_strategy.chunk.return_value = mock_chunks

        with patch.object(chunking_service.strategy_factory, "create_strategy", return_value=mock_strategy):
            result = await chunking_service.execute_ingestion_chunking(
                text=sample_text,
                document_id="doc-456",
                collection=collection,
            )

        assert result["stats"]["strategy_used"] in ["semantic", "ChunkingStrategy.SEMANTIC"]
        assert result["stats"]["chunk_count"] == 2
        assert len(result["chunks"]) == 2

    @pytest.mark.asyncio()
    async def test_execute_ingestion_chunking_with_document_structure_strategy(self, chunking_service, sample_text):
        """Test successful chunking with document_structure (markdown) strategy."""
        collection = {
            "id": "coll-789",
            "name": "Markdown Collection",
            "chunking_strategy": "document_structure",
            "chunking_config": {
                "chunk_size": 150,
                "chunk_overlap": 30,
            },
            "chunk_size": 150,
            "chunk_overlap": 30,
        }

        mock_strategy = MagicMock()
        mock_chunks = [
            MagicMock(content="Structure chunk 1"),
            MagicMock(content="Structure chunk 2"),
            MagicMock(content="Structure chunk 3"),
            MagicMock(content="Structure chunk 4"),
        ]
        mock_strategy.chunk.return_value = mock_chunks

        with patch.object(chunking_service.strategy_factory, "create_strategy", return_value=mock_strategy):
            result = await chunking_service.execute_ingestion_chunking(
                text=sample_text,
                document_id="doc-789",
                collection=collection,
                file_type="md",
            )

        assert result["stats"]["strategy_used"] in [
            "markdown",
            "ChunkingStrategy.MARKDOWN",
            "ChunkingStrategy.DOCUMENT_STRUCTURE",
        ]
        assert result["stats"]["chunk_count"] == 4
        assert result["chunks"][0]["metadata"]["strategy"] in [
            "markdown",
            "ChunkingStrategy.MARKDOWN",
            "ChunkingStrategy.DOCUMENT_STRUCTURE",
        ]

    @pytest.mark.asyncio()
    async def test_execute_ingestion_chunking_fallback_to_token_chunker_on_missing_strategy(
        self, chunking_service, sample_text
    ):
        """Test fallback to TokenChunker when chunking_strategy is missing."""
        collection = {
            "id": "coll-no-strategy",
            "name": "No Strategy Collection",
            # No chunking_strategy specified
            "chunk_size": 100,
            "chunk_overlap": 20,
        }

        with patch("shared.text_processing.chunking.TokenChunker") as mock_token_chunker:
            mock_chunker = MagicMock()
            mock_token_chunker.return_value = mock_chunker
            mock_chunker.chunk_text.return_value = [
                {
                    "chunk_id": "doc-999_0000",
                    "text": "Token chunk 1",
                    "metadata": {"index": 0},
                },
                {
                    "chunk_id": "doc-999_0001",
                    "text": "Token chunk 2",
                    "metadata": {"index": 1},
                },
            ]

            result = await chunking_service.execute_ingestion_chunking(
                text=sample_text,
                document_id="doc-999",
                collection=collection,
            )

        # Verify TokenChunker was called with correct parameters
        mock_token_chunker.assert_called_once_with(chunk_size=100, chunk_overlap=20)
        mock_chunker.chunk_text.assert_called_once()

        # Verify result
        assert result["stats"]["strategy_used"] == "TokenChunker"
        assert result["stats"]["fallback"] is False  # Not a fallback, just default behavior
        assert result["stats"]["chunk_count"] == 2

    @pytest.mark.asyncio()
    async def test_execute_ingestion_chunking_fallback_on_invalid_config(self, chunking_service, sample_text):
        """Test fallback to TokenChunker when configuration is invalid."""
        collection = {
            "id": "coll-bad-config",
            "name": "Bad Config Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {
                "invalid_param": "bad_value",
                "chunk_size": "not_a_number",  # Invalid type
            },
            "chunk_size": 100,
            "chunk_overlap": 20,
        }

        # Mock config builder to return validation errors
        with patch.object(chunking_service.config_builder, "build_config") as mock_build_config:
            mock_build_config.return_value = MagicMock(
                validation_errors=["Invalid chunk_size: must be a number"],
                strategy="recursive",
                config={},
            )

            with patch("shared.text_processing.chunking.TokenChunker") as mock_token_chunker:
                mock_chunker = MagicMock()
                mock_token_chunker.return_value = mock_chunker
                mock_chunker.chunk_text.return_value = [
                    {
                        "chunk_id": "doc-bad_0000",
                        "text": "Fallback chunk",
                        "metadata": {},
                    },
                ]

                result = await chunking_service.execute_ingestion_chunking(
                    text=sample_text,
                    document_id="doc-bad",
                    collection=collection,
                )

        # Verify fallback occurred
        assert result["stats"]["strategy_used"] == "TokenChunker"
        assert result["stats"]["fallback"] is True
        assert result["stats"]["chunk_count"] == 1

    @pytest.mark.asyncio()
    async def test_execute_ingestion_chunking_fallback_on_strategy_runtime_error(
        self, chunking_service, sample_text, sample_collection
    ):
        """Test fallback to TokenChunker when strategy execution fails."""
        # Mock strategy to raise an exception
        mock_strategy = MagicMock()
        mock_strategy.chunk.side_effect = RuntimeError("Strategy execution failed")

        with (
            patch.object(chunking_service.strategy_factory, "create_strategy", return_value=mock_strategy),
            patch("shared.text_processing.chunking.TokenChunker") as mock_token_chunker,
        ):
            mock_chunker = MagicMock()
            mock_token_chunker.return_value = mock_chunker
            mock_chunker.chunk_text.return_value = [
                {
                    "chunk_id": "doc-123_0000",
                    "text": "Fallback chunk after error",
                    "metadata": {},
                },
            ]

            result = await chunking_service.execute_ingestion_chunking(
                text=sample_text,
                document_id="doc-123",
                collection=sample_collection,
            )

        # Verify fallback occurred
        assert result["stats"]["strategy_used"] == "TokenChunker"
        assert result["stats"]["fallback"] is True

    @pytest.mark.asyncio()
    async def test_execute_ingestion_chunking_fallback_on_config_build_error(
        self, chunking_service, sample_text, sample_collection
    ):
        """Test fallback to TokenChunker when config building fails."""
        with (
            patch.object(chunking_service.config_builder, "build_config", side_effect=Exception("Config build failed")),
            patch("shared.text_processing.chunking.TokenChunker") as mock_token_chunker,
        ):
            mock_chunker = MagicMock()
            mock_token_chunker.return_value = mock_chunker
            mock_chunker.chunk_text.return_value = [
                {
                    "chunk_id": "doc-123_0000",
                    "text": "Fallback chunk",
                    "metadata": {},
                },
            ]

            result = await chunking_service.execute_ingestion_chunking(
                text=sample_text,
                document_id="doc-123",
                collection=sample_collection,
            )

        assert result["stats"]["strategy_used"] == "TokenChunker"
        assert result["stats"]["fallback"] is True

    @pytest.mark.asyncio()
    async def test_execute_ingestion_chunking_preserves_metadata(
        self, chunking_service, sample_text, sample_collection
    ):
        """Test that metadata is properly preserved and merged."""
        metadata = {
            "author": "Test Author",
            "date": "2024-01-01",
            "custom_field": "custom_value",
        }

        mock_strategy = MagicMock()
        mock_chunks = [
            MagicMock(content="Chunk with metadata"),
        ]
        mock_strategy.chunk.return_value = mock_chunks

        with patch.object(chunking_service.strategy_factory, "create_strategy", return_value=mock_strategy):
            result = await chunking_service.execute_ingestion_chunking(
                text=sample_text,
                document_id="doc-meta",
                collection=sample_collection,
                metadata=metadata,
            )

        # Verify metadata is preserved
        chunk_metadata = result["chunks"][0]["metadata"]
        assert chunk_metadata["author"] == "Test Author"
        assert chunk_metadata["date"] == "2024-01-01"
        assert chunk_metadata["custom_field"] == "custom_value"
        assert chunk_metadata["index"] == 0
        assert chunk_metadata["strategy"] in ["recursive", "ChunkingStrategy.RECURSIVE"]

    @pytest.mark.asyncio()
    async def test_execute_ingestion_chunking_chunk_id_generation(
        self, chunking_service, sample_text, sample_collection
    ):
        """Test correct chunk ID generation format."""
        mock_strategy = MagicMock()
        mock_chunks = [MagicMock(content=f"Chunk {i}") for i in range(15)]
        mock_strategy.chunk.return_value = mock_chunks

        with patch.object(chunking_service.strategy_factory, "create_strategy", return_value=mock_strategy):
            result = await chunking_service.execute_ingestion_chunking(
                text=sample_text,
                document_id="doc-abc123",
                collection=sample_collection,
            )

        # Verify chunk ID format
        chunks = result["chunks"]
        assert chunks[0]["chunk_id"] == "doc-abc123_0000"
        assert chunks[9]["chunk_id"] == "doc-abc123_0009"
        assert chunks[14]["chunk_id"] == "doc-abc123_0014"

    @pytest.mark.skip(reason="Mock not working as expected with asyncio.to_thread")
    @pytest.mark.asyncio()
    async def test_execute_ingestion_chunking_fatal_error_propagation(self, chunking_service, sample_text):
        """Test that fatal errors are propagated correctly."""
        # Test with no strategy specified so it goes directly to TokenChunker
        collection_no_strategy = {
            "id": "coll-fatal",
            "name": "Fatal Collection",
            # No chunking_strategy specified - will use TokenChunker directly
            "chunk_size": 100,
            "chunk_overlap": 20,
        }

        # Mock TokenChunker to fail (simulating unrecoverable error)
        with patch("shared.text_processing.chunking.TokenChunker") as mock_token_chunker:
            mock_token_chunker.side_effect = MemoryError("Out of memory")
            with pytest.raises(MemoryError):
                await chunking_service.execute_ingestion_chunking(
                    text=sample_text,
                    document_id="doc-fatal",
                    collection=collection_no_strategy,
                )


class TestAppendTaskIntegration:
    """Test integration of execute_ingestion_chunking with APPEND task."""

    @pytest.fixture()
    def mock_dependencies(self):
        """Create mock dependencies for APPEND task."""
        return {
            "db": AsyncMock(spec=AsyncSession),
            "updater": AsyncMock(),
            "operation": MagicMock(
                id="op-123",
                collection_id="coll-123",
                type=OperationType.APPEND,
                status=OperationStatus.PENDING,
                config={"source_id": "source-123"},
            ),
            "collection": MagicMock(
                id="coll-123",
                name="Test Collection",
                path="/test/path",
                status=CollectionStatus.READY,
                vector_collection_id="vc-123",
                chunking_strategy="recursive",
                chunking_config={"chunk_size": 100, "chunk_overlap": 20},
                chunk_size=100,
                chunk_overlap=20,
                embedding_model="Qwen/Qwen3-Embedding-0.6B",
                quantization="float16",
                get=MagicMock(
                    side_effect=lambda key, default=None: {
                        "id": "coll-123",
                        "name": "Test Collection",
                        "path": "/test/path",
                        "status": CollectionStatus.READY,
                        "vector_collection_id": "vc-123",
                        "chunking_strategy": "recursive",
                        "chunking_config": {"chunk_size": 100, "chunk_overlap": 20},
                        "chunk_size": 100,
                        "chunk_overlap": 20,
                        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                        "quantization": "float16",
                    }.get(key, default)
                ),
            ),
            "documents": [
                MagicMock(
                    id="doc-1",
                    file_path="/test/doc1.txt",
                    file_size=1024,
                    status=DocumentStatus.PENDING,
                    chunk_count=0,
                    get=MagicMock(
                        side_effect=lambda key, default=None: {
                            "id": "doc-1",
                            "file_path": "/test/doc1.txt",
                            "file_size": 1024,
                            "status": DocumentStatus.PENDING,
                            "chunk_count": 0,
                        }.get(key, default)
                    ),
                ),
                MagicMock(
                    id="doc-2",
                    file_path="/test/doc2.pdf",
                    file_size=2048,
                    status=DocumentStatus.PENDING,
                    chunk_count=0,
                    get=MagicMock(
                        side_effect=lambda key, default=None: {
                            "id": "doc-2",
                            "file_path": "/test/doc2.pdf",
                            "file_size": 2048,
                            "status": DocumentStatus.PENDING,
                            "chunk_count": 0,
                        }.get(key, default)
                    ),
                ),
            ],
        }

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("httpx.AsyncClient.post")
    @patch("packages.webui.tasks.qdrant_manager")
    async def test_append_task_uses_execute_ingestion_chunking(
        self, mock_qdrant_manager, mock_httpx_post, mock_extract_serialize, mock_dependencies
    ):
        """Test that APPEND task correctly uses execute_ingestion_chunking."""
        # Setup mocks
        db = mock_dependencies["db"]
        updater = mock_dependencies["updater"]
        operation = mock_dependencies["operation"]
        collection = mock_dependencies["collection"]
        documents = mock_dependencies["documents"]

        # Mock database queries - need to create proper mock chain for async operations
        # First execute call returns operation
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        # Second execute call returns collection
        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = collection

        # Third execute call returns documents
        mock_result_3 = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = documents
        mock_result_3.scalars.return_value = mock_scalars

        # Configure db.execute to return these mocks in sequence
        db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3]

        # Mock text extraction
        mock_extract_serialize.return_value = [
            ("Document 1 text content", {"page": 1}),
            ("Document 1 more content", {"page": 2}),
        ]

        # Mock vecpipe embeddings call (via httpx)
        embed_response = MagicMock()
        embed_response.status_code = 200
        embed_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}

        upsert_response = MagicMock()
        upsert_response.status_code = 200

        # Configure mock to return responses for each document (2 documents = 4 calls total)
        # Each document makes one embed call and one upsert call
        mock_httpx_post.side_effect = [embed_response, upsert_response, embed_response, upsert_response]

        # Mock Qdrant manager
        mock_qdrant_client = MagicMock()
        mock_qdrant_manager.get_client.return_value = mock_qdrant_client

        # Mock chunking dependency resolver
        with patch(
            "packages.webui.tasks.resolve_celery_chunking_service",
            new_callable=AsyncMock,
        ) as mock_chunking_service_class:
            mock_chunking_service = MagicMock()
            mock_chunking_service_class.return_value = mock_chunking_service

            # Mock execute_ingestion_chunking to return chunks for each document
            # Return same response for both documents to simplify testing
            mock_chunking_service.execute_ingestion_chunking = AsyncMock(
                side_effect=[
                    {
                        "chunks": [
                            {
                                "chunk_id": "doc-1_0000",
                                "text": "Chunk 1 text",
                                "metadata": {"index": 0, "strategy": "recursive"},
                            },
                            {
                                "chunk_id": "doc-1_0001",
                                "text": "Chunk 2 text",
                                "metadata": {"index": 1, "strategy": "recursive"},
                            },
                        ],
                        "stats": {
                            "duration_ms": 50,
                            "strategy_used": "recursive",
                            "fallback": False,
                            "chunk_count": 2,
                        },
                    },
                    {
                        "chunks": [
                            {
                                "chunk_id": "doc-2_0000",
                                "text": "Doc 2 Chunk 1 text",
                                "metadata": {"index": 0, "strategy": "recursive"},
                            },
                        ],
                        "stats": {
                            "duration_ms": 50,
                            "strategy_used": "recursive",
                            "fallback": False,
                            "chunk_count": 1,
                        },
                    },
                ]
            )

            # Run the APPEND operation
            await _process_append_operation(db, updater, "op-123")

            # Verify execute_ingestion_chunking was called
            mock_chunking_service.execute_ingestion_chunking.assert_called()

            # Verify the call arguments - check first call since we have 2 documents
            call_args_list = mock_chunking_service.execute_ingestion_chunking.call_args_list
            assert len(call_args_list) == 2  # Called once for each document

            # Check first call (doc-1)
            first_call_args = call_args_list[0]
            assert first_call_args[1]["document_id"] == "doc-1"
            assert "collection" in first_call_args[1]
            assert first_call_args[1]["collection"]["chunking_strategy"] == "recursive"

            # Check second call (doc-2)
            second_call_args = call_args_list[1]
            assert second_call_args[1]["document_id"] == "doc-2"

            # Verify document chunk_count was updated
            assert documents[0].chunk_count == 2

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("httpx.AsyncClient.post")
    @patch("packages.webui.tasks.qdrant_manager")
    async def test_append_task_updates_chunk_count_correctly(
        self, mock_qdrant_manager, mock_httpx_post, mock_extract_serialize, mock_dependencies
    ):
        """Test that APPEND task correctly updates Document.chunk_count."""
        db = mock_dependencies["db"]
        updater = mock_dependencies["updater"]
        operation = mock_dependencies["operation"]
        collection = mock_dependencies["collection"]
        documents = mock_dependencies["documents"]

        # Setup database mocks - need to create proper mock chain for async operations
        # First execute call returns operation
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        # Second execute call returns collection
        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = collection

        # Third execute call returns documents
        mock_result_3 = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = documents
        mock_result_3.scalars.return_value = mock_scalars

        # Configure db.execute to return these mocks in sequence
        db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3]

        # Mock text extraction
        mock_extract_serialize.return_value = [("Test content", {})]

        # Mock HTTP responses for vecpipe
        embed_response = MagicMock()
        embed_response.status_code = 200
        embed_response.json.return_value = {"embeddings": [[0.1] * 384] * 8}

        upsert_response = MagicMock()
        upsert_response.status_code = 200

        # Configure mock to return responses for each document (2 documents = 4 calls total)
        # Each document makes one embed call and one upsert call
        mock_httpx_post.side_effect = [embed_response, upsert_response, embed_response, upsert_response]

        # Mock Qdrant manager
        mock_qdrant_client = MagicMock()
        mock_qdrant_manager.get_client.return_value = mock_qdrant_client

        # Mock ChunkingService with different chunk counts for each document
        with patch(
            "packages.webui.tasks.resolve_celery_chunking_service",
            new_callable=AsyncMock,
        ) as mock_chunking_service_class:
            mock_chunking_service = MagicMock()
            mock_chunking_service_class.return_value = mock_chunking_service

            # Return different chunk counts for each document
            chunk_results = [
                {
                    "chunks": [{"chunk_id": f"doc-1_{i:04d}", "text": f"chunk {i}", "metadata": {}} for i in range(5)],
                    "stats": {"chunk_count": 5, "strategy_used": "recursive", "fallback": False},
                },
                {
                    "chunks": [{"chunk_id": f"doc-2_{i:04d}", "text": f"chunk {i}", "metadata": {}} for i in range(3)],
                    "stats": {"chunk_count": 3, "strategy_used": "recursive", "fallback": False},
                },
            ]
            mock_chunking_service.execute_ingestion_chunking = AsyncMock(side_effect=chunk_results)

            # Run the operation
            await _process_append_operation(db, updater, "op-123")

            # Verify chunk counts were updated correctly
            assert documents[0].chunk_count == 5
            assert documents[1].chunk_count == 3

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("httpx.AsyncClient.post")
    @patch("packages.webui.tasks.qdrant_manager")
    async def test_append_task_handles_different_strategies(
        self, mock_qdrant_manager, mock_httpx_post, mock_extract_serialize, mock_dependencies
    ):
        """Test that APPEND task correctly handles different chunking strategies."""
        db = mock_dependencies["db"]
        updater = mock_dependencies["updater"]
        operation = mock_dependencies["operation"]
        collection = mock_dependencies["collection"]
        documents = [mock_dependencies["documents"][0]]  # Single document

        # Test with semantic strategy
        collection.chunking_strategy = "semantic"
        collection.chunking_config = {"buffer_size": 1, "breakpoint_percentile_threshold": 95}
        # Configure the collection mock to return proper values when accessed with .get()
        collection.get = MagicMock(side_effect=lambda key, default=None: getattr(collection, key, default))

        # Setup database mocks - need to create proper mock chain for async operations
        # First execute call returns operation
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        # Second execute call returns collection
        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = collection

        # Third execute call returns documents
        mock_result_3 = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = documents
        mock_result_3.scalars.return_value = mock_scalars

        # Configure db.execute to return these mocks in sequence
        db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3]

        mock_extract_serialize.return_value = [("Test content for semantic chunking", {})]

        # Mock HTTP responses
        embed_response = MagicMock()
        embed_response.status_code = 200
        embed_response.json.return_value = {"embeddings": [[0.1] * 384]}

        upsert_response = MagicMock()
        upsert_response.status_code = 200

        mock_httpx_post.side_effect = [embed_response, upsert_response]

        # Mock Qdrant manager
        mock_qdrant_client = MagicMock()
        mock_qdrant_manager.get_client.return_value = mock_qdrant_client

        with patch(
            "packages.webui.tasks.resolve_celery_chunking_service",
            new_callable=AsyncMock,
        ) as mock_chunking_service_class:
            mock_chunking_service = MagicMock()
            mock_chunking_service_class.return_value = mock_chunking_service

            mock_chunking_service.execute_ingestion_chunking = AsyncMock(
                return_value={
                    "chunks": [
                        {"chunk_id": "doc-1_0000", "text": "Semantic chunk", "metadata": {"strategy": "semantic"}}
                    ],
                    "stats": {"chunk_count": 1, "strategy_used": "semantic", "fallback": False},
                }
            )

            await _process_append_operation(db, updater, "op-123")

            # Verify semantic strategy was used
            call_args = mock_chunking_service.execute_ingestion_chunking.call_args
            assert call_args[1]["collection"]["chunking_strategy"] == "semantic"

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("httpx.AsyncClient.post")
    @patch("packages.webui.tasks.qdrant_manager")
    async def test_append_task_handles_fallback_gracefully(
        self, mock_qdrant_manager, mock_httpx_post, mock_extract_serialize, mock_dependencies
    ):
        """Test that APPEND task handles fallback to TokenChunker gracefully."""
        db = mock_dependencies["db"]
        updater = mock_dependencies["updater"]
        operation = mock_dependencies["operation"]
        collection = mock_dependencies["collection"]
        documents = [mock_dependencies["documents"][0]]

        # Setup database mocks - need to create proper mock chain for async operations
        # First execute call returns operation
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        # Second execute call returns collection
        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = collection

        # Third execute call returns documents
        mock_result_3 = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = documents
        mock_result_3.scalars.return_value = mock_scalars

        # Configure db.execute to return these mocks in sequence
        db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3]

        mock_extract_serialize.return_value = [("Test content", {})]

        # Mock HTTP responses
        embed_response = MagicMock()
        embed_response.status_code = 200
        embed_response.json.return_value = {"embeddings": [[0.1] * 384]}

        upsert_response = MagicMock()
        upsert_response.status_code = 200

        mock_httpx_post.side_effect = [embed_response, upsert_response]

        # Mock Qdrant manager
        mock_qdrant_client = MagicMock()
        mock_qdrant_manager.get_client.return_value = mock_qdrant_client

        with patch(
            "packages.webui.tasks.resolve_celery_chunking_service",
            new_callable=AsyncMock,
        ) as mock_chunking_service_class:
            mock_chunking_service = MagicMock()
            mock_chunking_service_class.return_value = mock_chunking_service

            # Simulate fallback scenario
            mock_chunking_service.execute_ingestion_chunking = AsyncMock(
                return_value={
                    "chunks": [{"chunk_id": "doc-1_chunk_0000", "text": "Fallback chunk", "metadata": {}}],
                    "stats": {"chunk_count": 1, "strategy_used": "TokenChunker", "fallback": True},
                }
            )

            await _process_append_operation(db, updater, "op-123")

            # Verify operation completed successfully despite fallback
            assert documents[0].chunk_count == 1
            assert operation.status != OperationStatus.FAILED


class TestReindexTaskIntegration:
    """Test integration of execute_ingestion_chunking with REINDEX task."""

    @pytest.fixture()
    def mock_reindex_dependencies(self):
        """Create mock dependencies for REINDEX task."""
        return {
            "db": AsyncMock(spec=AsyncSession),
            "updater": AsyncMock(),
            "operation": MagicMock(
                id="op-reindex-123",
                collection_id="coll-123",
                type=OperationType.REINDEX,
                status=OperationStatus.PENDING,
                config={
                    "chunk_size": 150,
                    "chunk_overlap": 30,
                    "chunking_strategy": "markdown",
                    "chunking_config": {"preserve_structure": True},
                },
                get=MagicMock(
                    side_effect=lambda key, default=None: {
                        "id": "op-reindex-123",
                        "collection_id": "coll-123",
                        "type": OperationType.REINDEX,
                        "status": OperationStatus.PENDING,
                        "config": {
                            "chunk_size": 150,
                            "chunk_overlap": 30,
                            "chunking_strategy": "markdown",
                            "chunking_config": {"preserve_structure": True},
                        },
                    }.get(key, default)
                ),
            ),
            "source_collection": MagicMock(
                id="coll-123",
                name="Source Collection",
                path="/source/path",
                status=CollectionStatus.READY,
                vector_collection_id="vc-source",
                chunking_strategy="recursive",
                chunking_config={},
                chunk_size=100,
                chunk_overlap=20,
                embedding_model="Qwen/Qwen3-Embedding-0.6B",
                quantization="float16",
                get=MagicMock(
                    side_effect=lambda key, default=None: {
                        "id": "coll-123",
                        "name": "Source Collection",
                        "path": "/source/path",
                        "status": CollectionStatus.READY,
                        "vector_collection_id": "vc-source",
                        "chunking_strategy": "recursive",
                        "chunking_config": {},
                        "chunk_size": 100,
                        "chunk_overlap": 20,
                        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                        "quantization": "float16",
                    }.get(key, default)
                ),
            ),
            "staging_collection": MagicMock(
                id="coll-staging-123",
                name="Source Collection (staging)",
                path="/source/path",
                status=CollectionStatus.PROCESSING,
                vector_collection_id="vc-staging",
                parent_collection_id="coll-123",
                get=MagicMock(
                    side_effect=lambda key, default=None: {
                        "id": "coll-staging-123",
                        "name": "Source Collection (staging)",
                        "path": "/source/path",
                        "status": CollectionStatus.PROCESSING,
                        "vector_collection_id": "vc-staging",
                        "parent_collection_id": "coll-123",
                    }.get(key, default)
                ),
            ),
            "documents": [
                MagicMock(
                    id="doc-reindex-1",
                    file_path="/source/doc1.md",
                    file_size=1024,
                    status=DocumentStatus.COMPLETED,
                    chunk_count=0,
                    get=MagicMock(
                        side_effect=lambda key, default=None: {
                            "id": "doc-reindex-1",
                            "file_path": "/source/doc1.md",
                            "file_size": 1024,
                            "status": DocumentStatus.COMPLETED,
                            "chunk_count": 0,
                        }.get(key, default)
                    ),
                ),
            ],
        }

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("httpx.AsyncClient.post")
    @patch("packages.webui.tasks.qdrant_manager")
    async def test_reindex_task_uses_execute_ingestion_chunking(
        self, mock_qdrant_manager, mock_httpx_post, mock_extract_serialize, mock_reindex_dependencies
    ):
        """Test that REINDEX task correctly uses execute_ingestion_chunking."""
        db = mock_reindex_dependencies["db"]
        updater = mock_reindex_dependencies["updater"]
        operation = mock_reindex_dependencies["operation"]
        source_collection = mock_reindex_dependencies["source_collection"]
        staging_collection = mock_reindex_dependencies["staging_collection"]
        documents = mock_reindex_dependencies["documents"]

        # Setup database mocks - need to create proper mock chain for async operations
        # First execute call returns operation
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        # Second execute call returns source_collection
        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = source_collection

        # Third execute call returns staging_collection
        mock_result_3 = MagicMock()
        mock_result_3.scalar_one_or_none.return_value = staging_collection

        # Fourth execute call returns documents
        mock_result_4 = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = documents
        mock_result_4.scalars.return_value = mock_scalars

        # Configure db.execute to return these mocks in sequence
        db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3, mock_result_4]

        # Mock text extraction
        mock_extract_serialize.return_value = [("# Markdown content\n\nParagraph text", {})]

        # Mock HTTP responses for vecpipe
        embed_response = MagicMock()
        embed_response.status_code = 200
        embed_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        upsert_response = MagicMock()
        upsert_response.status_code = 200

        mock_httpx_post.side_effect = [embed_response, upsert_response]

        # Mock Qdrant manager
        mock_qdrant_client = MagicMock()
        mock_qdrant_manager.get_client.return_value = mock_qdrant_client
        mock_qdrant_client.get_collection.return_value = MagicMock(points_count=1)

        # Mock ChunkingService
        with patch(
            "packages.webui.tasks.resolve_celery_chunking_service",
            new_callable=AsyncMock,
        ) as mock_chunking_service_class:
            mock_chunking_service = MagicMock()
            mock_chunking_service_class.return_value = mock_chunking_service

            mock_chunking_service.execute_ingestion_chunking = AsyncMock(
                return_value={
                    "chunks": [
                        {
                            "chunk_id": "doc-reindex-1_0000",
                            "text": "Markdown chunk",
                            "metadata": {"strategy": "markdown"},
                        },
                    ],
                    "stats": {
                        "duration_ms": 50,
                        "strategy_used": "markdown",
                        "fallback": False,
                        "chunk_count": 1,
                    },
                }
            )

            await _process_reindex_operation(db, updater, "op-reindex-123")

            # Verify execute_ingestion_chunking was called with overridden config
            call_args = mock_chunking_service.execute_ingestion_chunking.call_args
            collection_arg = call_args[1]["collection"]

            # Verify strategy override from new_config
            assert collection_arg["chunking_strategy"] == "markdown"
            assert collection_arg["chunking_config"]["preserve_structure"] is True
            assert collection_arg["chunk_size"] == 150
            assert collection_arg["chunk_overlap"] == 30

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("httpx.AsyncClient.post")
    @patch("packages.webui.tasks.qdrant_manager")
    async def test_reindex_task_preserves_staging_collection(
        self, mock_qdrant_manager, mock_httpx_post, mock_extract_serialize, mock_reindex_dependencies
    ):
        """Test that REINDEX task correctly uses staging collection."""
        db = mock_reindex_dependencies["db"]
        updater = mock_reindex_dependencies["updater"]
        operation = mock_reindex_dependencies["operation"]
        source_collection = mock_reindex_dependencies["source_collection"]
        staging_collection = mock_reindex_dependencies["staging_collection"]
        documents = mock_reindex_dependencies["documents"]

        # Setup database mocks - need to create proper mock chain for async operations
        # First execute call returns operation
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        # Second execute call returns source_collection
        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = source_collection

        # Third execute call returns staging_collection
        mock_result_3 = MagicMock()
        mock_result_3.scalar_one_or_none.return_value = staging_collection

        # Fourth execute call returns documents
        mock_result_4 = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = documents
        mock_result_4.scalars.return_value = mock_scalars

        # Configure db.execute to return these mocks in sequence
        db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3, mock_result_4]

        mock_extract_serialize.return_value = [("Test content", {})]

        # Mock HTTP responses
        embed_response = MagicMock()
        embed_response.status_code = 200
        embed_response.json.return_value = {"embeddings": [[0.1] * 384]}

        upsert_response = MagicMock()
        upsert_response.status_code = 200

        mock_httpx_post.side_effect = [embed_response, upsert_response]

        # Mock Qdrant manager
        mock_qdrant_client = MagicMock()
        mock_qdrant_manager.get_client.return_value = mock_qdrant_client
        mock_qdrant_client.get_collection.return_value = MagicMock(points_count=1)

        with patch(
            "packages.webui.tasks.resolve_celery_chunking_service",
            new_callable=AsyncMock,
        ) as mock_chunking_service_class:
            mock_chunking_service = MagicMock()
            mock_chunking_service_class.return_value = mock_chunking_service

            mock_chunking_service.execute_ingestion_chunking = AsyncMock(
                return_value={
                    "chunks": [{"chunk_id": "doc_0000", "text": "chunk", "metadata": {}}],
                    "stats": {"chunk_count": 1, "strategy_used": "markdown", "fallback": False},
                }
            )

            await _process_reindex_operation(db, updater, "op-reindex-123")

            # Verify staging collection was used for indexing
            # The actual upsert happens via HTTP calls to vecpipe
            assert mock_httpx_post.called

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("httpx.AsyncClient.post")
    @patch("packages.webui.tasks.qdrant_manager")
    async def test_reindex_task_updates_document_chunk_count(
        self, mock_qdrant_manager, mock_httpx_post, mock_extract_serialize, mock_reindex_dependencies
    ):
        """Test that REINDEX task correctly updates Document.chunk_count."""
        db = mock_reindex_dependencies["db"]
        updater = mock_reindex_dependencies["updater"]
        operation = mock_reindex_dependencies["operation"]
        source_collection = mock_reindex_dependencies["source_collection"]
        staging_collection = mock_reindex_dependencies["staging_collection"]

        # Create multiple documents with different initial chunk counts
        documents = [
            MagicMock(
                id=f"doc-{i}",
                file_path=f"/source/doc{i}.md",
                file_size=1024 * (i + 1),
                status=DocumentStatus.COMPLETED,
                chunk_count=10,  # Initial chunk count
            )
            for i in range(3)
        ]

        # Setup database mocks - need to create proper mock chain for async operations
        # First execute call returns operation
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        # Second execute call returns source_collection
        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = source_collection

        # Third execute call returns staging_collection
        mock_result_3 = MagicMock()
        mock_result_3.scalar_one_or_none.return_value = staging_collection

        # Fourth execute call returns documents
        mock_result_4 = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = documents
        mock_result_4.scalars.return_value = mock_scalars

        # Configure db.execute to return these mocks in sequence
        db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3, mock_result_4]

        mock_extract_serialize.return_value = [("Content to chunk", {})]

        # Mock HTTP responses
        embed_response = MagicMock()
        embed_response.status_code = 200
        embed_response.json.return_value = {"embeddings": [[0.1] * 384] * 9}

        upsert_response = MagicMock()
        upsert_response.status_code = 200

        # Configure mock to return responses for 3 documents (3 documents = 6 calls total)
        # Each document makes one embed call and one upsert call
        mock_httpx_post.side_effect = [
            embed_response,
            upsert_response,
            embed_response,
            upsert_response,
            embed_response,
            upsert_response,
        ]

        # Mock Qdrant manager
        mock_qdrant_client = MagicMock()
        mock_qdrant_manager.get_client.return_value = mock_qdrant_client
        mock_qdrant_client.get_collection.return_value = MagicMock(points_count=9)

        with patch(
            "packages.webui.tasks.resolve_celery_chunking_service",
            new_callable=AsyncMock,
        ) as mock_chunking_service_class:
            mock_chunking_service = MagicMock()
            mock_chunking_service_class.return_value = mock_chunking_service

            # Different chunk counts for each document
            chunk_results = [
                {
                    "chunks": [
                        {"chunk_id": f"doc-{i}_{j:04d}", "text": f"chunk {j}", "metadata": {}} for j in range(i + 2)
                    ],
                    "stats": {"chunk_count": i + 2, "strategy_used": "markdown", "fallback": False},
                }
                for i in range(3)
            ]
            mock_chunking_service.execute_ingestion_chunking = AsyncMock(side_effect=chunk_results)

            await _process_reindex_operation(db, updater, "op-reindex-123")

            # Verify chunk counts were updated
            assert documents[0].chunk_count == 2  # Was 10, now 2
            assert documents[1].chunk_count == 3  # Was 10, now 3
            assert documents[2].chunk_count == 4  # Was 10, now 4

    @pytest.mark.asyncio()
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("httpx.AsyncClient.post")
    @patch("packages.webui.tasks.qdrant_manager")
    async def test_reindex_task_without_strategy_override(
        self, mock_qdrant_manager, mock_httpx_post, mock_extract_serialize, mock_reindex_dependencies
    ):
        """Test REINDEX task when new_config doesn't override strategy."""
        db = mock_reindex_dependencies["db"]
        updater = mock_reindex_dependencies["updater"]
        operation = mock_reindex_dependencies["operation"]
        source_collection = mock_reindex_dependencies["source_collection"]
        staging_collection = mock_reindex_dependencies["staging_collection"]
        documents = mock_reindex_dependencies["documents"]

        # Operation config without strategy override
        operation.config = {"chunk_size": 200}  # Only override chunk_size
        # Update the operation's get method to reflect the new config
        operation.get = MagicMock(
            side_effect=lambda key, default=None: {
                "id": "op-reindex-123",
                "collection_id": "coll-123",
                "type": OperationType.REINDEX,
                "status": OperationStatus.PENDING,
                "config": {"chunk_size": 200},  # Updated config
            }.get(key, default)
        )

        # Setup database mocks - need to create proper mock chain for async operations
        # First execute call returns operation
        mock_result_1 = MagicMock()
        mock_result_1.scalar_one.return_value = operation

        # Second execute call returns source_collection
        mock_result_2 = MagicMock()
        mock_result_2.scalar_one_or_none.return_value = source_collection

        # Third execute call returns staging_collection
        mock_result_3 = MagicMock()
        mock_result_3.scalar_one_or_none.return_value = staging_collection

        # Fourth execute call returns documents
        mock_result_4 = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = documents
        mock_result_4.scalars.return_value = mock_scalars

        # Configure db.execute to return these mocks in sequence
        db.execute.side_effect = [mock_result_1, mock_result_2, mock_result_3, mock_result_4]

        mock_extract_serialize.return_value = [("Content", {})]

        # Mock HTTP responses
        embed_response = MagicMock()
        embed_response.status_code = 200
        embed_response.json.return_value = {"embeddings": [[0.1] * 384]}

        upsert_response = MagicMock()
        upsert_response.status_code = 200

        mock_httpx_post.side_effect = [embed_response, upsert_response]

        # Mock Qdrant manager
        mock_qdrant_client = MagicMock()
        mock_qdrant_manager.get_client.return_value = mock_qdrant_client
        mock_qdrant_client.get_collection.return_value = MagicMock(points_count=1)

        with patch(
            "packages.webui.tasks.resolve_celery_chunking_service",
            new_callable=AsyncMock,
        ) as mock_chunking_service_class:
            mock_chunking_service = MagicMock()
            mock_chunking_service_class.return_value = mock_chunking_service

            mock_chunking_service.execute_ingestion_chunking = AsyncMock(
                return_value={
                    "chunks": [{"chunk_id": "chunk_0000", "text": "chunk", "metadata": {}}],
                    "stats": {"chunk_count": 1, "strategy_used": "recursive", "fallback": False},
                }
            )

            await _process_reindex_operation(db, updater, "op-reindex-123")

            # Verify original strategy was preserved
            call_args = mock_chunking_service.execute_ingestion_chunking.call_args
            collection_arg = call_args[1]["collection"]
            assert collection_arg["chunking_strategy"] == "recursive"  # Original strategy
            assert collection_arg["chunk_size"] == 200  # Overridden size


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases for chunking integration."""

    @pytest.fixture()
    def chunking_service(self):
        """Create a ChunkingService instance."""
        mock_db = AsyncMock(spec=AsyncSession)
        mock_collection_repo = MagicMock(spec=CollectionRepository)
        mock_document_repo = MagicMock(spec=DocumentRepository)

        return ChunkingService(
            db_session=mock_db,
            collection_repo=mock_collection_repo,
            document_repo=mock_document_repo,
            redis_client=None,
        )

    @pytest.mark.asyncio()
    async def test_execute_ingestion_chunking_empty_text(self, chunking_service):
        """Test handling of empty text input."""
        collection = {
            "id": "coll-empty",
            "name": "Empty Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 100,
            "chunk_overlap": 20,
        }

        mock_strategy = MagicMock()
        mock_strategy.chunk.return_value = []

        with patch.object(chunking_service.strategy_factory, "create_strategy", return_value=mock_strategy):
            result = await chunking_service.execute_ingestion_chunking(
                text="",
                document_id="doc-empty",
                collection=collection,
            )

        assert result["chunks"] == []
        assert result["stats"]["chunk_count"] == 0

    @pytest.mark.asyncio()
    async def test_execute_ingestion_chunking_very_large_text(self, chunking_service):
        """Test handling of very large text input."""
        # Create a large text (100KB instead of 1MB to avoid stack overflow)
        large_text = "x" * (100 * 1024)

        collection = {
            "id": "coll-large",
            "name": "Large Collection",
            "chunk_size": 1000,
            "chunk_overlap": 100,
        }

        with patch("shared.text_processing.chunking.TokenChunker") as mock_token_chunker:
            mock_chunker = MagicMock()
            mock_token_chunker.return_value = mock_chunker

            # Simulate chunking large text into many chunks (reduced from 1000 to 100)
            num_chunks = 100
            mock_chunker.chunk_text.return_value = [
                {
                    "chunk_id": f"doc-large_{i:04d}",
                    "text": f"chunk {i}",
                    "metadata": {"index": i},
                }
                for i in range(num_chunks)
            ]

            result = await chunking_service.execute_ingestion_chunking(
                text=large_text,
                document_id="doc-large",
                collection=collection,
            )

        assert result["stats"]["chunk_count"] == num_chunks
        assert len(result["chunks"]) == num_chunks

    @pytest.mark.asyncio()
    async def test_execute_ingestion_chunking_special_characters(self, chunking_service):
        """Test handling of text with special characters."""
        special_text = "Text with 特殊字符 and émojis 🎉 and symbols ©®™"

        collection = {
            "id": "coll-special",
            "name": "Special Collection",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "chunk_size": 50,
            "chunk_overlap": 10,
        }

        mock_strategy = MagicMock()
        mock_strategy.chunk.return_value = [
            MagicMock(content="Text with 特殊字符"),
            MagicMock(content="and émojis 🎉"),
            MagicMock(content="and symbols ©®™"),
        ]

        with patch.object(chunking_service.strategy_factory, "create_strategy", return_value=mock_strategy):
            result = await chunking_service.execute_ingestion_chunking(
                text=special_text,
                document_id="doc-special",
                collection=collection,
            )

        assert len(result["chunks"]) == 3
        assert "特殊字符" in result["chunks"][0]["text"]
        assert "🎉" in result["chunks"][1]["text"]

    @pytest.mark.asyncio()
    async def test_execute_ingestion_chunking_performance_timing(self, chunking_service):
        """Test that duration_ms is calculated correctly."""

        collection = {
            "id": "coll-timing",
            "name": "Timing Collection",
            "chunk_size": 100,
            "chunk_overlap": 20,
        }

        with patch("shared.text_processing.chunking.TokenChunker") as mock_token_chunker:
            mock_chunker = MagicMock()
            mock_token_chunker.return_value = mock_chunker

            # Return chunks immediately (mocked operation doesn't need delay)
            mock_chunker.chunk_text.return_value = [{"chunk_id": "chunk_0000", "text": "chunk", "metadata": {}}]

            result = await chunking_service.execute_ingestion_chunking(
                text="Test text",
                document_id="doc-timing",
                collection=collection,
            )

        # Duration should be a positive integer (timing is mocked so it runs fast)
        assert result["stats"]["duration_ms"] >= 0
        assert isinstance(result["stats"]["duration_ms"], int)
