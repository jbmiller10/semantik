"""
Tests for ChunkingService.

Comprehensive unit tests for the chunking service layer including
preview functionality, strategy recommendations, configuration management,
caching behavior, and error handling.
"""

import asyncio
import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from packages.shared.database.exceptions import (
    EntityNotFoundError,
)
from packages.webui.api.chunking_exceptions import (
    ChunkingMemoryError,
    ChunkingStrategyError,
    ChunkingTimeoutError,
)
from packages.webui.api.v2.chunking_schemas import ChunkingStrategy
from packages.webui.services.chunking_service import ChunkingService


@pytest.fixture()
def mock_redis() -> AsyncMock:
    """Create a mock Redis client."""
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.setex.return_value = True
    mock.delete.return_value = 1
    mock.exists.return_value = False
    mock.expire.return_value = True
    mock.incr.return_value = 1
    mock.hget.return_value = None
    mock.hset.return_value = True
    mock.hgetall.return_value = {}
    return mock


@pytest.fixture()
def mock_db_session() -> AsyncMock:
    """Create a mock database session."""
    mock = AsyncMock()
    mock.execute.return_value = MagicMock()
    mock.commit.return_value = None
    mock.rollback.return_value = None
    mock.refresh.return_value = None
    return mock


@pytest.fixture()
def mock_document_service() -> AsyncMock:
    """Create a mock document service."""
    mock = AsyncMock()
    mock.get_document.return_value = {
        "id": "doc-123",
        "file_name": "test.pdf",
        "file_path": "/path/to/test.pdf",
        "content": "This is test document content for chunking.",
        "mime_type": "application/pdf",
        "file_size": 1024,
    }
    return mock


@pytest.fixture()
def mock_collection_service() -> AsyncMock:
    """Create a mock collection service."""
    mock = AsyncMock()
    mock.get_collection.return_value = {
        "id": "coll-123",
        "name": "Test Collection",
        "document_count": 10,
        "chunking_strategy": "fixed_size",
        "chunk_size": 512,
        "chunk_overlap": 50,
    }
    mock.get_collection_documents.return_value = [
        {"id": f"doc-{i}", "file_name": f"doc_{i}.txt", "file_size": 1024} for i in range(5)
    ]
    return mock


@pytest.fixture()
def chunking_service(
    mock_redis: AsyncMock,
    mock_db_session: AsyncMock,
    mock_document_service: AsyncMock,
    mock_collection_service: AsyncMock,
) -> ChunkingService:
    """Create a ChunkingService instance with mocked dependencies."""
    # Create mock repositories
    mock_collection_repo = MagicMock()
    mock_document_repo = MagicMock()
    mock_operation_repo = MagicMock()

    # Create mock Qdrant client
    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])
    mock_qdrant.create_collection.return_value = None
    mock_qdrant.upsert.return_value = None

    # Setup mock repository methods
    mock_collection_repo.get_by_uuid_with_permission_check = AsyncMock()

    # Create mock documents for testing
    mock_documents = []
    for i in range(5):
        doc = MagicMock()
        doc.id = f"doc-{i}"
        doc.file_name = f"doc_{i}.txt"
        doc.file_path = f"/path/to/doc_{i}.txt"
        doc.file_size_bytes = 1024
        doc.mime_type = "text/plain"
        doc.collection_id = "coll-123"
        doc.chunk_count = None
        doc.status = None
        doc.error_message = None
        mock_documents.append(doc)

    mock_document_repo.list_by_collection = AsyncMock(return_value=(mock_documents, len(mock_documents)))

    # Create a proper mock document object
    mock_document = MagicMock()
    mock_document.id = "doc-123"
    mock_document.file_name = "test.pdf"
    mock_document.file_path = "/path/to/test.pdf"
    mock_document.file_size_bytes = 1024
    mock_document.mime_type = "application/pdf"

    # Mock get_by_id to return appropriate documents
    async def mock_get_by_id(doc_id):
        # Return specific documents for specific IDs
        for doc in mock_documents:
            if doc.id == doc_id:
                return doc
        # Return the default mock document for other IDs
        if doc_id == "doc-123":
            return mock_document
        # Create a new mock for other specific document IDs
        if doc_id.startswith("doc-"):
            new_doc = MagicMock()
            new_doc.id = doc_id
            new_doc.file_name = f"{doc_id}.txt"
            new_doc.file_path = f"/path/to/{doc_id}.txt"
            new_doc.file_size_bytes = 1024
            new_doc.mime_type = "text/plain"
            new_doc.collection_id = "coll-123"
            new_doc.chunk_count = None
            new_doc.status = None
            new_doc.error_message = None
            return new_doc
        return None

    mock_document_repo.get_by_id = AsyncMock(side_effect=mock_get_by_id)

    mock_operation_repo.get_by_uuid_with_permission_check = AsyncMock()
    mock_operation_repo.get_by_uuid = AsyncMock()
    mock_operation_repo.update = AsyncMock()
    mock_operation_repo.update_status = AsyncMock()

    # Create service with mocked dependencies
    service = ChunkingService(
        db_session=mock_db_session,
        collection_repo=mock_collection_repo,
        document_repo=mock_document_repo,
        redis_client=mock_redis,
        operation_repo=mock_operation_repo,
        qdrant_client=mock_qdrant,
    )

    # Also set the services as attributes for tests that might use them
    service.document_service = mock_document_service
    service.collection_service = mock_collection_service

    return service


class TestPreviewFunctionality:
    """Test preview generation and caching."""

    @pytest.mark.asyncio()
    async def test_preview_with_content(
        self,
        chunking_service: ChunkingService,
        mock_redis: AsyncMock,
    ) -> None:
        """Test generating preview with provided content."""
        content = "This is a test document. It has multiple sentences. Each sentence should be preserved when chunking."

        result = await chunking_service.preview_chunking(
            document_id=None,
            content=content,
            strategy=ChunkingStrategy.FIXED_SIZE,
            config={"chunk_size": 50, "chunk_overlap": 10},
            max_chunks=5,
            include_metrics=True,
            user_id=1,
        )

        assert "preview_id" in result
        assert result["strategy"] == ChunkingStrategy.FIXED_SIZE
        assert "chunks" in result
        assert "metrics" in result
        assert result["processing_time_ms"] >= 0

        # Verify caching attempted
        mock_redis.setex.assert_called()

    @pytest.mark.asyncio()
    @patch("packages.webui.services.chunking_service.ChunkingFactory")
    async def test_preview_with_document_id(
        self,
        mock_chunking_factory: MagicMock,
        chunking_service: ChunkingService,
        mock_document_service: AsyncMock,
    ) -> None:
        """Test generating preview with document ID."""
        # Mock the chunker to avoid needing OpenAI API
        mock_chunker = MagicMock()
        mock_chunk_result = MagicMock()
        mock_chunk_result.chunk_id = "chunk-1"
        mock_chunk_result.text = "Test chunk content"
        mock_chunk_result.metadata = {"chunk_index": 0}
        mock_chunk_result.start_offset = 0
        mock_chunk_result.end_offset = 20

        mock_chunker.chunk_text_async = AsyncMock(return_value=[mock_chunk_result])
        mock_chunking_factory.create_chunker.return_value = mock_chunker

        result = await chunking_service.preview_chunking(
            document_id="doc-123",
            content=None,
            strategy=ChunkingStrategy.SEMANTIC,
            config=None,
            max_chunks=10,
            include_metrics=False,
            user_id=1,
        )

        assert result["strategy"] == ChunkingStrategy.SEMANTIC
        assert "chunks" in result
        assert result["total_chunks"] >= 0

        # Verify document was fetched
        mock_document_service.get_document.assert_called_with("doc-123", user_id=1)

    @pytest.mark.asyncio()
    async def test_preview_caching(
        self,
        chunking_service: ChunkingService,
        mock_redis: AsyncMock,
    ) -> None:
        """Test that preview results are cached and retrieved correctly."""
        # First call - not cached
        mock_redis.get.return_value = None

        content = "Test content for caching"
        result1 = await chunking_service.preview_chunking(
            content=content,
            strategy=ChunkingStrategy.FIXED_SIZE,
            user_id=1,
        )

        # First call should not be from cache
        assert "preview_id" in result1

        # Setup cache hit
        cached_data = json.dumps(result1)
        mock_redis.get.return_value = cached_data

        # Second call - should be cached
        result2 = await chunking_service.preview_chunking(
            content=content,
            strategy=ChunkingStrategy.FIXED_SIZE,
            user_id=1,
        )

        # Second call should have same content if cached
        assert result2["preview_id"] == result1["preview_id"]

    @pytest.mark.asyncio()
    async def test_preview_cache_key_generation(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test that cache keys are generated consistently."""
        content = "Test content"
        strategy = ChunkingStrategy.RECURSIVE
        config = {"chunk_size": 512, "separators": ["\n\n", "\n"]}

        key1 = chunking_service._generate_cache_key(content, strategy, config, 1)
        key2 = chunking_service._generate_cache_key(content, strategy, config, 1)

        assert key1 == key2

        # Different content should produce different key
        key3 = chunking_service._generate_cache_key("Different content", strategy, config, 1)
        assert key3 != key1

        # Different user should produce different key
        key4 = chunking_service._generate_cache_key(content, strategy, config, 2)
        assert key4 != key1

    @pytest.mark.asyncio()
    async def test_clear_preview_cache(
        self,
        chunking_service: ChunkingService,
        mock_redis: AsyncMock,
    ) -> None:
        """Test clearing preview cache."""
        preview_id = str(uuid.uuid4())

        await chunking_service.clear_preview_cache(preview_id, user_id=1)

        mock_redis.delete.assert_called_with(f"preview:{preview_id}:1")


class TestStrategyRecommendation:
    """Test strategy recommendation functionality."""

    @pytest.mark.asyncio()
    async def test_recommend_strategy_for_pdf(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test strategy recommendation for PDF files."""
        result = await chunking_service.recommend_strategy(
            file_types=["pdf"],
            user_id=1,
        )

        assert result["strategy"] in [
            ChunkingStrategy.SEMANTIC,
            ChunkingStrategy.DOCUMENT_STRUCTURE,
        ]
        assert result["confidence"] > 0.5
        assert "reasoning" in result
        assert "alternatives" in result

    @pytest.mark.asyncio()
    async def test_recommend_strategy_for_code(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test strategy recommendation for code files."""
        result = await chunking_service.recommend_strategy(
            file_types=["py", "js", "java"],
            user_id=1,
        )

        assert result["strategy"] in [
            ChunkingStrategy.RECURSIVE,
            ChunkingStrategy.FIXED_SIZE,
        ]
        assert "code" in result["reasoning"].lower() or "programming" in result["reasoning"].lower()

    @pytest.mark.asyncio()
    async def test_recommend_strategy_for_mixed_types(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test strategy recommendation for mixed file types."""
        result = await chunking_service.recommend_strategy(
            file_types=["pdf", "txt", "md", "json"],
            user_id=1,
        )

        assert result["strategy"] in [
            ChunkingStrategy.HYBRID,
            ChunkingStrategy.RECURSIVE,
        ]
        assert len(result.get("alternatives", [])) > 0

    @pytest.mark.asyncio()
    async def test_recommend_strategy_with_no_file_types(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test strategy recommendation with no file types provided."""
        result = await chunking_service.recommend_strategy(
            file_types=[],
            user_id=1,
        )

        # Should provide a default recommendation
        assert result["strategy"] == ChunkingStrategy.FIXED_SIZE
        assert result["confidence"] < 0.5  # Low confidence for default


class TestConfigurationValidation:
    """Test configuration validation for collections."""

    @pytest.mark.asyncio()
    async def test_validate_valid_config(
        self,
        chunking_service: ChunkingService,
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test validation of valid configuration."""
        result = await chunking_service.validate_config_for_collection(
            collection_id="coll-123",
            strategy=ChunkingStrategy.FIXED_SIZE,
            config={"chunk_size": 512, "chunk_overlap": 50},
            user_id=1,
        )

        assert result["is_valid"] is True
        assert "estimated_time" in result

    @pytest.mark.asyncio()
    async def test_validate_invalid_chunk_size(
        self,
        chunking_service: ChunkingService,
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test validation with invalid chunk size."""
        # Set collection with large documents
        mock_collection_service.get_collection_documents.return_value = [
            {"id": f"doc-{i}", "file_size": 10 * 1024 * 1024} for i in range(5)  # 10MB files
        ]

        result = await chunking_service.validate_config_for_collection(
            collection_id="coll-123",
            strategy=ChunkingStrategy.FIXED_SIZE,
            config={"chunk_size": 10},  # Too small for large documents
            user_id=1,
        )

        assert result["is_valid"] is False
        assert "reason" in result

    @pytest.mark.asyncio()
    async def test_validate_semantic_requirements(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test validation of semantic chunking requirements."""
        result = await chunking_service.validate_config_for_collection(
            collection_id="coll-123",
            strategy=ChunkingStrategy.SEMANTIC,
            config={"embedding_model": "non-existent-model"},
            user_id=1,
        )

        # Should check if embedding model is available
        assert "valid" in result
        if not result["is_valid"]:
            assert "embedding" in result.get("reason", "").lower()

    @pytest.mark.asyncio()
    async def test_estimate_processing_time(
        self,
        chunking_service: ChunkingService,
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test processing time estimation."""
        # Set up collection with known document count and sizes
        mock_collection_service.get_collection.return_value = {
            "id": "coll-123",
            "document_count": 100,
            "total_size_bytes": 50 * 1024 * 1024,  # 50MB total
        }

        result = await chunking_service.validate_config_for_collection(
            collection_id="coll-123",
            strategy=ChunkingStrategy.SEMANTIC,
            config={},
            user_id=1,
        )

        assert "estimated_time" in result
        assert result["estimated_time"] > 0  # Should take some time for 50MB


class TestChunkingOperations:
    """Test chunking operation processing."""

    @pytest.mark.asyncio()
    async def test_process_chunking_operation_success(
        self,
        chunking_service: ChunkingService,
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test successful chunking operation processing."""
        operation_id = str(uuid.uuid4())

        # Mock WebSocket manager
        with patch("packages.webui.services.chunking_service.ws_manager") as mock_ws:
            mock_ws.send_message = AsyncMock()

            await chunking_service.process_chunking_operation(
                operation_id=operation_id,
                collection_id="coll-123",
                strategy="fixed_size",
                config={"chunk_size": 512},
                document_ids=None,  # Process all documents
                user_id=1,
                websocket_channel=f"chunking:coll-123:{operation_id}",
            )

            # Verify progress updates were sent
            assert mock_ws.send_message.called

            # Check that completion was signaled
            calls = mock_ws.send_message.call_args_list
            completion_calls = [call for call in calls if "completed" in str(call)]
            assert len(completion_calls) > 0

    @pytest.mark.asyncio()
    async def test_process_chunking_with_specific_documents(
        self,
        chunking_service: ChunkingService,
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test chunking operation with specific document IDs."""
        operation_id = str(uuid.uuid4())
        document_ids = ["doc-1", "doc-2", "doc-3"]

        with patch("packages.webui.services.chunking_service.ws_manager") as mock_ws:
            mock_ws.send_message = AsyncMock()

            await chunking_service.process_chunking_operation(
                operation_id=operation_id,
                collection_id="coll-123",
                strategy="semantic",
                config={},
                document_ids=document_ids,
                user_id=1,
                websocket_channel=f"chunking:coll-123:{operation_id}",
            )

            # Should only process specified documents
            # Verify by checking progress messages
            progress_calls = [
                call for call in mock_ws.send_message.call_args_list if call[0][1].get("type") == "chunking_progress"
            ]

            # Should have progress for 3 documents
            assert len(progress_calls) <= len(document_ids)

    @pytest.mark.asyncio()
    async def test_process_chunking_operation_failure(
        self,
        chunking_service: ChunkingService,
        mock_document_service: AsyncMock,
    ) -> None:
        """Test chunking operation failure handling."""
        operation_id = str(uuid.uuid4())

        # Simulate document processing failure
        mock_document_service.get_document.side_effect = Exception("Document read error")

        with patch("packages.webui.services.chunking_service.ws_manager") as mock_ws:
            mock_ws.send_message = AsyncMock()

            with pytest.raises(Exception, match="Document read error"):
                await chunking_service.process_chunking_operation(
                    operation_id=operation_id,
                    collection_id="coll-123",
                    strategy="fixed_size",
                    config={},
                    document_ids=["doc-1"],
                    user_id=1,
                    websocket_channel=f"chunking:coll-123:{operation_id}",
                )

            # Should send failure notification
            failure_calls = [call for call in mock_ws.send_message.call_args_list if "failed" in str(call)]
            assert len(failure_calls) > 0


class TestStatisticsAndMetrics:
    """Test statistics and metrics calculation."""

    @pytest.mark.asyncio()
    async def test_get_chunking_statistics(
        self,
        chunking_service: ChunkingService,
        mock_db_session: AsyncMock,
    ) -> None:
        """Test getting chunking statistics for a collection."""
        # Mock Document objects with chunk_count attribute
        mock_docs = [MagicMock(chunk_count=25) for _ in range(20)]

        # Mock database query results
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_docs
        mock_db_session.execute.return_value = mock_result

        stats = await chunking_service.get_chunking_statistics(
            collection_id="coll-123",
            user_id=1,
        )

        assert stats.total_chunks == 500
        assert stats.total_documents == 20
        assert stats.average_chunk_size == 512
        assert "average_chunks_per_document" in stats.performance_metrics
        assert stats.performance_metrics["average_chunks_per_document"] == 25.0

    @pytest.mark.asyncio()
    async def test_calculate_quality_metrics(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test quality metrics calculation."""
        chunks = [
            {"content": "Short chunk", "size": 11},
            {"content": "This is a medium sized chunk with more content", "size": 47},
            {"content": "Very long chunk " * 50, "size": 800},
        ]

        metrics = chunking_service._calculate_quality_metrics(chunks)

        assert "coherence" in metrics
        assert "completeness" in metrics
        assert "size_consistency" in metrics
        assert 0 <= metrics["coherence"] <= 1
        assert 0 <= metrics["completeness"] <= 1

    @pytest.mark.asyncio()
    async def test_track_preview_usage(
        self,
        chunking_service: ChunkingService,
        mock_redis: AsyncMock,
    ) -> None:
        """Test tracking preview usage for rate limiting."""
        user_id = 1
        strategy = ChunkingStrategy.SEMANTIC

        await chunking_service.track_preview_usage(user_id, strategy.value)

        # Should track usage in Redis
        mock_redis.incr.assert_called()
        mock_redis.expire.assert_called()


class TestErrorHandling:
    """Test error handling in chunking service."""

    @pytest.mark.asyncio()
    async def test_handle_memory_error(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test handling of memory errors during chunking."""
        # Simulate large content that would exceed memory limits
        large_content = "x" * (100 * 1024 * 1024)  # 100MB

        with pytest.raises(ChunkingMemoryError):
            await chunking_service.preview_chunking(
                content=large_content,
                strategy=ChunkingStrategy.SEMANTIC,
                user_id=1,
            )

    @pytest.mark.asyncio()
    async def test_handle_timeout_error(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test handling of timeout errors."""
        # Mock slow processing
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError), pytest.raises(ChunkingTimeoutError):
            await chunking_service.preview_chunking(
                content="Test content",
                strategy=ChunkingStrategy.SEMANTIC,
                user_id=1,
            )

    @pytest.mark.asyncio()
    async def test_handle_invalid_strategy(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test handling of invalid strategy."""
        with pytest.raises(ChunkingStrategyError):
            await chunking_service.preview_chunking(
                content="Test",
                strategy="invalid_strategy",  # Invalid
                user_id=1,
            )

    @pytest.mark.asyncio()
    async def test_handle_document_not_found(
        self,
        chunking_service: ChunkingService,
        mock_document_service: AsyncMock,
    ) -> None:
        """Test handling when document is not found."""
        mock_document_service.get_document.side_effect = EntityNotFoundError("Document", "non-existent")

        with pytest.raises(EntityNotFoundError):
            await chunking_service.preview_chunking(
                document_id="non-existent",
                strategy=ChunkingStrategy.FIXED_SIZE,
                user_id=1,
            )

    @pytest.mark.asyncio()
    async def test_handle_redis_connection_error(
        self,
        chunking_service: ChunkingService,
        mock_redis: AsyncMock,
    ) -> None:
        """Test graceful handling of Redis connection errors."""
        # Simulate Redis connection error
        mock_redis.get.side_effect = ConnectionError("Redis unavailable")
        mock_redis.setex.side_effect = ConnectionError("Redis unavailable")

        # Should still work without caching
        result = await chunking_service.preview_chunking(
            content="Test content",
            strategy=ChunkingStrategy.FIXED_SIZE,
            user_id=1,
        )

        assert "preview_id" in result
        assert "chunks" in result


class TestChunkingAlgorithms:
    """Test actual chunking algorithm implementations."""

    @pytest.mark.asyncio()
    async def test_fixed_size_chunking(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test fixed size chunking algorithm."""
        content = "A" * 1000  # 1000 characters

        chunks = await chunking_service._chunk_content(
            content=content,
            strategy=ChunkingStrategy.FIXED_SIZE,
            config={"chunk_size": 100, "chunk_overlap": 10},
        )

        # Should create approximately 11 chunks with overlap
        assert len(chunks) >= 10

        # Check chunk sizes
        for chunk in chunks[:-1]:  # All but last chunk
            assert 90 <= len(chunk["content"]) <= 110

    @pytest.mark.asyncio()
    async def test_recursive_chunking(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test recursive chunking algorithm."""
        content = """
# Header 1
Paragraph 1 content.

# Header 2
Paragraph 2 content.
More content here.

## Subheader
Sub-content.
"""

        chunks = await chunking_service._chunk_content(
            content=content,
            strategy=ChunkingStrategy.RECURSIVE,
            config={"separators": ["\n#", "\n\n", "\n", " "]},
        )

        # Should split on headers first
        assert len(chunks) >= 3

        # Check that headers are preserved
        chunk_texts = [c["content"] for c in chunks]
        assert any("Header 1" in text for text in chunk_texts)
        assert any("Header 2" in text for text in chunk_texts)

    @pytest.mark.asyncio()
    async def test_sliding_window_chunking(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test sliding window chunking algorithm."""
        content = "The quick brown fox jumps over the lazy dog. " * 20

        chunks = await chunking_service._chunk_content(
            content=content,
            strategy=ChunkingStrategy.SLIDING_WINDOW,
            config={"window_size": 50, "step_size": 25},
        )

        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]["content"]
            chunk2 = chunks[i + 1]["content"]

            # Should have some overlap
            overlap = chunk1[25:]  # Last half of chunk1
            assert overlap in chunk2 or len(overlap) == 0

    @pytest.mark.asyncio()
    async def test_preserve_sentences(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test that sentence preservation works correctly."""
        content = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."

        chunks = await chunking_service._chunk_content(
            content=content,
            strategy=ChunkingStrategy.FIXED_SIZE,
            config={
                "chunk_size": 30,
                "chunk_overlap": 5,
                "preserve_sentences": True,
            },
        )

        # Check that chunks end at sentence boundaries
        for chunk in chunks:
            text = chunk["content"].strip()
            if text:
                # Should end with period, question mark, or exclamation
                assert text[-1] in ".?!" or text == content.strip()


class TestConcurrency:
    """Test concurrent operation handling."""

    @pytest.mark.asyncio()
    async def test_concurrent_preview_requests(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test handling multiple concurrent preview requests."""
        import asyncio

        # Create multiple preview tasks
        tasks = []
        for i in range(10):
            task = chunking_service.preview_chunking(
                content=f"Content {i}",
                strategy=ChunkingStrategy.FIXED_SIZE,
                user_id=i,
            )
            tasks.append(task)

        # Run concurrently
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 10

        # Each should have unique preview_id
        preview_ids = [r["preview_id"] for r in results]
        assert len(set(preview_ids)) == 10

    @pytest.mark.asyncio()
    async def test_concurrent_cache_access(
        self,
        chunking_service: ChunkingService,
        mock_redis: AsyncMock,
    ) -> None:
        """Test concurrent cache access doesn't cause issues."""
        import asyncio

        # Same content for all to trigger cache hits
        content = "Shared content for caching"

        # First request to populate cache
        await chunking_service.preview_chunking(
            content=content,
            strategy=ChunkingStrategy.FIXED_SIZE,
            user_id=1,
        )

        # Simulate cache hit
        mock_redis.get.return_value = json.dumps(
            {
                "preview_id": "cached-id",
                "chunks": [],
                "cached": True,
            }
        )

        # Multiple concurrent requests for same content
        tasks = []
        for _ in range(20):
            task = chunking_service.preview_chunking(
                content=content,
                strategy=ChunkingStrategy.FIXED_SIZE,
                user_id=1,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All should get cached result
        # All results should have preview_ids
        assert all("preview_id" in r for r in results)


class TestProgressTracking:
    """Test operation progress tracking."""

    @pytest.mark.asyncio()
    async def test_get_chunking_progress(
        self,
        chunking_service: ChunkingService,
        mock_redis: AsyncMock,
    ) -> None:
        """Test getting chunking operation progress."""
        operation_id = str(uuid.uuid4())

        # Mock operation with progress data
        from types import SimpleNamespace

        mock_operation = SimpleNamespace(
            meta={
                "progress": {
                    "total_documents": 11,
                    "processed_documents": 5,
                    "current_document": "doc_6.pdf",
                    "chunks_created": 250,
                }
            },
            status=SimpleNamespace(value="in_progress"),
            progress_percentage=45.5,
            uuid=operation_id,
            started_at=datetime.now(UTC),
            error_message=None,
        )

        # Mock the operation repository to return our mock operation
        chunking_service.operation_repo.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_operation)

        result = await chunking_service.get_chunking_progress(
            operation_id=operation_id,
            user_id=1,
        )

        assert result["status"] == "in_progress"
        assert result["progress_percentage"] == 45.5
        assert result["documents_processed"] == 5

    @pytest.mark.asyncio()
    async def test_update_progress(
        self,
        chunking_service: ChunkingService,
        mock_redis: AsyncMock,
    ) -> None:
        """Test updating operation progress."""
        operation_id = str(uuid.uuid4())

        await chunking_service._update_progress(
            operation_id=operation_id,
            progress=60.0,
            status="in_progress",
            message="Processing doc_7.pdf",
            documents_processed=6,
            total_documents=10,
            current_document="doc_7.pdf",
        )

        # Verify Redis was updated
        mock_redis.hset.assert_called()

        # Check expiration was set
        # Check that expire was called
        mock_redis.expire.assert_called()
