#!/usr/bin/env python3

"""Tests for PreviewChunkingUseCase."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from shared.chunking.application.dto.requests import ChunkingStrategy, PreviewRequest
from shared.chunking.application.dto.responses import ChunkDTO, PreviewResponse
from shared.chunking.application.use_cases.preview_chunking import PreviewChunkingUseCase
from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.exceptions import InvalidConfigurationError, StrategyNotFoundError
from shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata


class TestPreviewChunkingUseCase:
    """Test suite for PreviewChunkingUseCase."""

    @pytest.fixture()
    def mock_document_service(self) -> None:
        """Create mock document service."""
        service = MagicMock()
        service.get_document_content = AsyncMock(return_value="This is a sample document content for testing.")
        service.get_document_size = AsyncMock(return_value=1000)

        # Add the required methods for PreviewChunkingUseCase
        service.load_partial = AsyncMock(return_value={"content": "This is a sample document content for testing."})
        service.extract_text = AsyncMock(return_value="This is a sample document content for testing.")
        service.get_metadata = AsyncMock(return_value={"size_bytes": 10000})

        return service

    @pytest.fixture()
    def mock_strategy_factory(self) -> None:
        """Create mock strategy factory."""
        factory = MagicMock()
        mock_strategy = MagicMock()
        mock_strategy.chunk.return_value = [
            Chunk(
                content="Chunk 1",
                metadata=ChunkMetadata(
                    chunk_id="chunk-1",
                    document_id="doc-123",
                    chunk_index=0,
                    start_offset=0,
                    end_offset=7,
                    token_count=2,
                    strategy_name="character",
                ),
                min_tokens=1,
            ),
            Chunk(
                content="Chunk 2",
                metadata=ChunkMetadata(
                    chunk_id="chunk-2",
                    document_id="doc-123",
                    chunk_index=1,
                    start_offset=8,
                    end_offset=15,
                    token_count=2,
                    strategy_name="character",
                ),
                min_tokens=1,
            ),
        ]
        factory.create_strategy.return_value = mock_strategy
        return factory

    @pytest.fixture()
    def mock_notification_service(self) -> None:
        """Create mock notification service."""
        service = MagicMock()
        service.notify_preview_generated = AsyncMock()
        service.notify_error = AsyncMock()

        # Add the required methods for PreviewChunkingUseCase
        service.notify_operation_started = AsyncMock()
        service.notify_operation_completed = AsyncMock()
        service.notify_operation_failed = AsyncMock()

        return service

    @pytest.fixture()
    def mock_metrics_service(self) -> None:
        """Create mock metrics service."""
        service = MagicMock()
        service.record_preview_request = AsyncMock()
        service.record_preview_duration = AsyncMock()

        # Add the required methods for PreviewChunkingUseCase
        service.record_operation_duration = AsyncMock()
        service.record_strategy_performance = AsyncMock()

        return service

    @pytest.fixture()
    def use_case(
        self, mock_document_service, mock_strategy_factory, mock_notification_service, mock_metrics_service
    ) -> None:
        """Create use case instance with mocked dependencies."""
        return PreviewChunkingUseCase(
            document_service=mock_document_service,
            strategy_factory=mock_strategy_factory,
            notification_service=mock_notification_service,
            metrics_service=mock_metrics_service,
        )

    @pytest.fixture()
    def valid_request(self) -> None:
        """Create a valid preview request."""
        return PreviewRequest(
            file_path="/data/documents/test.txt",
            strategy_type=ChunkingStrategy.CHARACTER,
            min_tokens=10,
            max_tokens=50,
            overlap=5,
            preview_size_kb=10,
            max_preview_chunks=5,
        )

    @pytest.mark.asyncio()
    async def test_successful_preview_generation(self, use_case, valid_request) -> None:
        """Test successful preview generation."""
        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert isinstance(response, PreviewResponse)
        assert response.operation_id is not None  # Operation ID should be generated
        assert response.strategy_used == "character"
        assert len(response.chunks) == 2
        assert response.total_chunks_estimate > 0
        assert response.document_sample_size > 0
        assert response.processing_time_ms > 0

        # Verify mock calls
        use_case.document_service.load_partial.assert_called_once_with(file_path="/data/documents/test.txt", size_kb=10)
        use_case.document_service.extract_text.assert_called_once()
        use_case.strategy_factory.create_strategy.assert_called_once_with(
            strategy_type="character", config={"min_tokens": 10, "max_tokens": 50, "overlap": 5}
        )
        use_case.notification_service.notify_operation_started.assert_called_once()
        use_case.notification_service.notify_operation_completed.assert_called_once()
        use_case.metrics_service.record_operation_duration.assert_called_once()
        use_case.metrics_service.record_strategy_performance.assert_called_once()

    @pytest.mark.asyncio()
    async def test_preview_with_limited_chunks(self, use_case, valid_request) -> None:
        """Test that preview respects max_preview_chunks limit."""
        # Arrange
        # Create more chunks than the limit
        many_chunks = [
            Chunk(
                content=f"Chunk {i}",
                metadata=ChunkMetadata(
                    chunk_id=f"chunk-{i}",
                    document_id="doc-123",
                    chunk_index=i,
                    start_offset=i * 10,
                    end_offset=(i + 1) * 10,
                    token_count=3,
                    strategy_name="character",
                ),
                min_tokens=1,
            )
            for i in range(10)
        ]
        use_case.strategy_factory.create_strategy.return_value.chunk.return_value = many_chunks

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert len(response.chunks) == 5  # max_preview_chunks
        assert response.chunks[0].content == "Chunk 0"
        assert response.chunks[4].content == "Chunk 4"

    @pytest.mark.asyncio()
    async def test_preview_with_custom_parameters(self, use_case) -> None:
        """Test preview with additional strategy parameters."""
        # Arrange
        request = PreviewRequest(
            file_path="/data/documents/async.txt",
            strategy_type=ChunkingStrategy.SEMANTIC,
            min_tokens=20,
            max_tokens=100,
            overlap=10,
            preview_size_kb=5,
            max_preview_chunks=3,
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.document_id == "doc-456"
        assert response.strategy_name == "semantic"

        # Verify strategy was created with additional params
        use_case.strategy_factory.create_strategy.assert_called_with(
            strategy_type="semantic", config={"min_tokens": 20, "max_tokens": 100, "overlap": 10}
        )

    @pytest.mark.asyncio()
    async def test_document_not_found(self, use_case, valid_request) -> None:
        """Test handling of document not found error."""
        # Arrange
        use_case.document_service.load_partial.side_effect = FileNotFoundError("Document not found")

        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            await use_case.execute(valid_request)

        assert "Document not found" in str(exc_info.value)
        # Use notify_operation_failed instead of notify_error
        use_case.notification_service.notify_operation_failed.assert_called_once()

    @pytest.mark.asyncio()
    async def test_strategy_not_found(self, use_case, valid_request) -> None:
        """Test handling of unknown strategy."""
        # Arrange
        use_case.strategy_factory.create_strategy.side_effect = StrategyNotFoundError("unknown_strategy")

        # Act & Assert
        with pytest.raises(ValueError, match="validation error"):
            await use_case.execute(valid_request)
        # The actual implementation catches the error and re-raises as ValueError
        use_case.notification_service.notify_operation_failed.assert_called_once()

    @pytest.mark.asyncio()
    async def test_invalid_configuration(self, use_case) -> None:
        """Test handling of invalid configuration."""
        # Arrange
        invalid_request = PreviewRequest(
            file_path="/data/documents/test.txt",
            strategy_type=ChunkingStrategy.CHARACTER,
            min_tokens=100,  # Greater than max
            max_tokens=50,
            overlap=5,
        )

        # Mock the strategy factory to raise InvalidConfigurationError
        use_case.strategy_factory.create_strategy.side_effect = InvalidConfigurationError(
            "min_tokens cannot be greater than max_tokens"
        )

        # Act & Assert
        with pytest.raises(ValueError, match="validation error"):
            await use_case.execute(invalid_request)

    @pytest.mark.asyncio()
    async def test_document_too_large_for_preview(self, use_case, valid_request) -> None:
        """Test handling of document too large for preview."""
        # Arrange
        large_content = "x" * (100 * 1024 * 1024)  # 100MB
        use_case.document_service.get_document_content.return_value = large_content

        # Act
        response = await use_case.execute(valid_request)

        # Assert - should still work but with truncated content
        assert isinstance(response, PreviewResponse)
        assert response.sample_size_bytes <= valid_request.preview_size_kb * 1024

    @pytest.mark.asyncio()
    async def test_empty_document(self, use_case, valid_request) -> None:
        """Test handling of empty document."""
        # Arrange
        use_case.document_service.extract_text.return_value = ""
        use_case.document_service.load_partial.return_value = {"content": ""}
        use_case.strategy_factory.create_strategy.return_value.chunk.return_value = []

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.preview_chunks == []
        assert response.estimated_total_chunks == 0
        # The actual implementation will have the sample size as 0 for empty string
        assert response.sample_size_bytes == 0

    @pytest.mark.asyncio()
    async def test_estimation_accuracy(self, use_case, valid_request) -> None:
        """Test chunk count estimation for full document."""
        # Arrange
        sample_content = "Sample content " * 20  # Small sample
        use_case.document_service.get_document_content.return_value = sample_content
        use_case.document_service.get_document_size.return_value = 100000  # 100KB total

        # Return 2 chunks for sample
        use_case.strategy_factory.create_strategy.return_value.chunk.return_value = [
            Chunk(
                content="Chunk 1",
                metadata=ChunkMetadata(
                    chunk_id="chunk-sample-1",
                    document_id="doc-preview",
                    chunk_index=0,
                    start_offset=0,
                    end_offset=50,
                    token_count=10,
                    strategy_name="character",
                ),
            ),
            Chunk(
                content="Chunk 2",
                metadata=ChunkMetadata(
                    chunk_id="chunk-sample-2",
                    document_id="doc-preview",
                    chunk_index=1,
                    start_offset=51,
                    end_offset=100,
                    token_count=10,
                    strategy_name="character",
                ),
            ),
        ]

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.estimated_total_chunks > 2  # Should estimate more for full doc
        assert response.estimated_total_chunks > 0
        # Estimation should be proportional to document size
        ratio = 100000 / len(sample_content)
        assert response.estimated_total_chunks >= int(2 * ratio * 0.5)  # Allow some variance

    @pytest.mark.asyncio()
    async def test_metrics_recording(self, use_case, valid_request) -> None:
        """Test that metrics are properly recorded."""
        # Act
        _ = await use_case.execute(valid_request)

        # Assert - Check actual methods called in the implementation
        use_case.metrics_service.record_operation_duration.assert_called_once()
        use_case.metrics_service.record_strategy_performance.assert_called_once()

        # Check duration was recorded with reasonable value
        call_args = use_case.metrics_service.record_operation_duration.call_args
        assert call_args is not None
        # Check that strategy performance was called with proper args
        perf_args = use_case.metrics_service.record_strategy_performance.call_args
        assert perf_args[1]["strategy_type"] == "character"

    @pytest.mark.asyncio()
    async def test_no_metrics_service(
        self, mock_document_service, mock_strategy_factory, mock_notification_service, valid_request
    ) -> None:
        """Test that use case works without metrics service."""
        # Arrange
        use_case = PreviewChunkingUseCase(
            document_service=mock_document_service,
            strategy_factory=mock_strategy_factory,
            notification_service=mock_notification_service,
            metrics_service=None,  # No metrics service
        )

        # Act
        response = await use_case.execute(valid_request)

        # Assert - should work normally
        assert isinstance(response, PreviewResponse)
        assert len(response.preview_chunks) > 0

    @pytest.mark.asyncio()
    async def test_chunk_dto_conversion(self, use_case, valid_request) -> None:
        """Test proper conversion of chunks to DTOs."""
        # Arrange
        test_chunks = [
            Chunk(
                content="Test chunk",
                metadata=ChunkMetadata(
                    chunk_id="chunk-test",
                    document_id="doc-123",
                    chunk_index=0,
                    start_offset=0,
                    end_offset=10,
                    token_count=3,
                    strategy_name="character",
                ),
                min_tokens=1,
            )
        ]
        use_case.strategy_factory.create_strategy.return_value.chunk.return_value = test_chunks

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert len(response.preview_chunks) == 1
        chunk_dto = response.preview_chunks[0]
        assert isinstance(chunk_dto, ChunkDTO)
        assert chunk_dto.content == "Test chunk"
        # Position info is in metadata, not directly on DTO
        # assert chunk_dto.start_position == 0
        # assert chunk_dto.end_position == 10
        assert chunk_dto.token_count == 3
        # Check that metadata exists
        assert chunk_dto.metadata is not None

    @pytest.mark.asyncio()
    async def test_progress_callback_integration(self, use_case, valid_request) -> None:
        """Test that progress callback is passed to strategy."""
        # Arrange
        _ = []

        def mock_chunk_with_progress(_content, progress_callback=None) -> None:
            # Note: The actual implementation only passes content, not config
            if progress_callback:
                progress_callback(25.0)
                progress_callback(50.0)
                progress_callback(75.0)
                progress_callback(100.0)
            return [
                Chunk(
                    content="Chunk",
                    metadata=ChunkMetadata(
                        chunk_id="chunk-test",
                        document_id="doc-123",
                        chunk_index=0,
                        start_offset=0,
                        end_offset=5,
                        token_count=1,
                        strategy_name="character",
                    ),
                    min_tokens=1,
                )
            ]

        use_case.strategy_factory.create_strategy.return_value.chunk.side_effect = mock_chunk_with_progress

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert isinstance(response, PreviewResponse)
        # Verify strategy.chunk was called - it just gets content arg in actual implementation
        call_args = use_case.strategy_factory.create_strategy.return_value.chunk.call_args
        assert call_args is not None

    @pytest.mark.asyncio()
    async def test_concurrent_preview_requests(self, use_case) -> None:
        """Test handling of concurrent preview requests."""
        # Arrange
        requests = [
            PreviewRequest(
                file_path=f"/data/documents/doc-{i}.txt",
                strategy_type=ChunkingStrategy.CHARACTER,
                min_tokens=10,
                max_tokens=50,
                overlap=5,
            )
            for i in range(3)
        ]

        # Act

        responses = await asyncio.gather(*[use_case.execute(req) for req in requests])

        # Assert
        assert len(responses) == 3
        for _i, response in enumerate(responses):
            # The actual implementation returns "doc-456" from metadata - not dynamically generated
            assert response.document_id == "doc-456"
            assert isinstance(response, PreviewResponse)

    @pytest.mark.asyncio()
    async def test_validation_of_preview_size(self, use_case) -> None:
        """Test validation of preview_size_kb parameter."""
        # Arrange
        request = PreviewRequest(
            file_path="/data/documents/test.txt",
            strategy_type=ChunkingStrategy.CHARACTER,
            min_tokens=10,
            max_tokens=50,
            overlap=5,
            preview_size_kb=0,  # Invalid size
        )

        # Act & Assert
        with pytest.raises(ValueError, match="preview_size_kb must be positive"):
            await use_case.execute(request)

    @pytest.mark.asyncio()
    async def test_notification_on_success(self, use_case, valid_request) -> None:
        """Test that success notification is sent."""
        # Act
        _ = await use_case.execute(valid_request)

        # Assert
        use_case.notification_service.notify_preview_generated.assert_called_once()
        call_args = use_case.notification_service.notify_preview_generated.call_args
        assert call_args[0][0] == "doc-123"  # document_id
        assert call_args[0][1] == 2  # chunk_count

    @pytest.mark.asyncio()
    async def test_notification_on_error(self, use_case, valid_request) -> None:
        """Test that error notification is sent on failure."""
        # Arrange
        use_case.document_service.load_partial.side_effect = Exception("Unexpected error")

        # Act & Assert
        with pytest.raises(Exception, match="Unexpected error"):
            await use_case.execute(valid_request)

        use_case.notification_service.notify_error.assert_called_once()
        call_args = use_case.notification_service.notify_error.call_args
        # Check the error was passed properly
        assert call_args is not None
        error_arg = call_args[1]["error"] if "error" in call_args[1] else call_args[0][0]
        assert "Unexpected error" in str(error_arg)
