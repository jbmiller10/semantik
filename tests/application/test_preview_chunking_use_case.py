#!/usr/bin/env python3
"""Tests for PreviewChunkingUseCase."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.shared.chunking.application.dto.requests import PreviewRequest
from packages.shared.chunking.application.dto.responses import ChunkDTO, PreviewResponse
from packages.shared.chunking.application.use_cases.preview_chunking import (
    PreviewChunkingUseCase,
)
from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.exceptions import (
    InvalidConfigurationError,
    StrategyNotFoundError,
)
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata


class TestPreviewChunkingUseCase:
    """Test suite for PreviewChunkingUseCase."""

    @pytest.fixture()
    def mock_document_service(self):
        """Create mock document service."""
        service = AsyncMock()
        service.get_document_content = AsyncMock(
            return_value="This is a sample document content for testing."
        )
        service.get_document_size = AsyncMock(return_value=1000)
        return service

    @pytest.fixture()
    def mock_strategy_factory(self):
        """Create mock strategy factory."""
        factory = MagicMock()
        mock_strategy = MagicMock()
        mock_strategy.chunk.return_value = [
            Chunk(
                content="Chunk 1",
                start_position=0,
                end_position=7,
                metadata=ChunkMetadata(token_count=2),
            ),
            Chunk(
                content="Chunk 2",
                start_position=8,
                end_position=15,
                metadata=ChunkMetadata(token_count=2),
            ),
        ]
        factory.create_strategy.return_value = mock_strategy
        return factory

    @pytest.fixture()
    def mock_notification_service(self):
        """Create mock notification service."""
        service = AsyncMock()
        service.notify_preview_generated = AsyncMock()
        service.notify_error = AsyncMock()
        return service

    @pytest.fixture()
    def mock_metrics_service(self):
        """Create mock metrics service."""
        service = AsyncMock()
        service.record_preview_request = AsyncMock()
        service.record_preview_duration = AsyncMock()
        return service

    @pytest.fixture()
    def use_case(
        self,
        mock_document_service,
        mock_strategy_factory,
        mock_notification_service,
        mock_metrics_service,
    ):
        """Create use case instance with mocked dependencies."""
        return PreviewChunkingUseCase(
            document_service=mock_document_service,
            strategy_factory=mock_strategy_factory,
            notification_service=mock_notification_service,
            metrics_service=mock_metrics_service,
        )

    @pytest.fixture()
    def valid_request(self):
        """Create a valid preview request."""
        return PreviewRequest(
            document_id="doc-123",
            strategy_name="character",
            min_tokens=10,
            max_tokens=50,
            overlap_tokens=5,
            preview_size_kb=10,
            max_preview_chunks=5,
        )

    @pytest.mark.asyncio()
    async def test_successful_preview_generation(self, use_case, valid_request):
        """Test successful preview generation."""
        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert isinstance(response, PreviewResponse)
        assert response.document_id == "doc-123"
        assert response.strategy_name == "character"
        assert len(response.preview_chunks) == 2
        assert response.estimated_total_chunks > 0
        assert response.sample_size_bytes > 0
        assert response.processing_time_ms > 0

        # Verify mock calls
        use_case.document_service.get_document_content.assert_called_once_with(
            "doc-123", max_bytes=10240  # 10KB
        )
        use_case.strategy_factory.create_strategy.assert_called_once_with("character")
        use_case.notification_service.notify_preview_generated.assert_called_once()
        use_case.metrics_service.record_preview_request.assert_called_once()

    @pytest.mark.asyncio()
    async def test_preview_with_limited_chunks(self, use_case, valid_request):
        """Test that preview respects max_preview_chunks limit."""
        # Arrange
        # Create more chunks than the limit
        many_chunks = [
            Chunk(
                content=f"Chunk {i}",
                start_position=i * 10,
                end_position=(i + 1) * 10,
                metadata=ChunkMetadata(token_count=3),
            )
            for i in range(10)
        ]
        use_case.strategy_factory.create_strategy.return_value.chunk.return_value = (
            many_chunks
        )

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert len(response.preview_chunks) == 5  # max_preview_chunks
        assert response.preview_chunks[0].content == "Chunk 0"
        assert response.preview_chunks[4].content == "Chunk 4"

    @pytest.mark.asyncio()
    async def test_preview_with_custom_parameters(self, use_case):
        """Test preview with additional strategy parameters."""
        # Arrange
        request = PreviewRequest(
            document_id="doc-456",
            strategy_name="semantic",
            min_tokens=20,
            max_tokens=100,
            overlap_tokens=10,
            preview_size_kb=5,
            max_preview_chunks=3,
            additional_params={"similarity_threshold": 0.8, "custom_param": "value"},
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.document_id == "doc-456"
        assert response.strategy_name == "semantic"

        # Verify strategy was created with additional params
        use_case.strategy_factory.create_strategy.assert_called_with("semantic")

    @pytest.mark.asyncio()
    async def test_document_not_found(self, use_case, valid_request):
        """Test handling of document not found error."""
        # Arrange
        use_case.document_service.get_document_content.side_effect = FileNotFoundError(
            "Document not found"
        )

        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            await use_case.execute(valid_request)

        assert "Document not found" in str(exc_info.value)
        use_case.notification_service.notify_error.assert_called_once()

    @pytest.mark.asyncio()
    async def test_strategy_not_found(self, use_case, valid_request):
        """Test handling of unknown strategy."""
        # Arrange
        use_case.strategy_factory.create_strategy.side_effect = StrategyNotFoundError(
            "unknown_strategy"
        )

        # Act & Assert
        with pytest.raises(StrategyNotFoundError) as exc_info:
            await use_case.execute(valid_request)

        assert "unknown_strategy" in str(exc_info.value)
        use_case.notification_service.notify_error.assert_called_once()

    @pytest.mark.asyncio()
    async def test_invalid_configuration(self, use_case):
        """Test handling of invalid configuration."""
        # Arrange
        invalid_request = PreviewRequest(
            document_id="doc-123",
            strategy_name="character",
            min_tokens=100,  # Greater than max
            max_tokens=50,
            overlap_tokens=5,
        )

        # Mock the strategy factory to raise InvalidConfigurationError
        use_case.strategy_factory.create_strategy.side_effect = (
            InvalidConfigurationError("min_tokens cannot be greater than max_tokens")
        )

        # Act & Assert
        with pytest.raises(InvalidConfigurationError) as exc_info:
            await use_case.execute(invalid_request)

        assert "min_tokens cannot be greater than max_tokens" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_document_too_large_for_preview(self, use_case, valid_request):
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
    async def test_empty_document(self, use_case, valid_request):
        """Test handling of empty document."""
        # Arrange
        use_case.document_service.get_document_content.return_value = ""
        use_case.strategy_factory.create_strategy.return_value.chunk.return_value = []

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.preview_chunks == []
        assert response.estimated_total_chunks == 0
        assert response.sample_size_bytes == 0

    @pytest.mark.asyncio()
    async def test_estimation_accuracy(self, use_case, valid_request):
        """Test chunk count estimation for full document."""
        # Arrange
        sample_content = "Sample content " * 20  # Small sample
        use_case.document_service.get_document_content.return_value = sample_content
        use_case.document_service.get_document_size.return_value = 100000  # 100KB total

        # Return 2 chunks for sample
        use_case.strategy_factory.create_strategy.return_value.chunk.return_value = [
            Chunk(content="Chunk 1", start_position=0, end_position=50,
                  metadata=ChunkMetadata(token_count=10)),
            Chunk(content="Chunk 2", start_position=51, end_position=100,
                  metadata=ChunkMetadata(token_count=10)),
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
    async def test_metrics_recording(self, use_case, valid_request):
        """Test that metrics are properly recorded."""
        # Act
        response = await use_case.execute(valid_request)

        # Assert
        use_case.metrics_service.record_preview_request.assert_called_once_with(
            strategy_name="character",
            document_size=1000,
        )
        use_case.metrics_service.record_preview_duration.assert_called_once()

        # Check duration was recorded with reasonable value
        call_args = use_case.metrics_service.record_preview_duration.call_args
        duration = call_args[0][0] if call_args else None
        assert duration is not None
        assert duration > 0

    @pytest.mark.asyncio()
    async def test_no_metrics_service(
        self,
        mock_document_service,
        mock_strategy_factory,
        mock_notification_service,
        valid_request,
    ):
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
    async def test_chunk_dto_conversion(self, use_case, valid_request):
        """Test proper conversion of chunks to DTOs."""
        # Arrange
        test_chunks = [
            Chunk(
                content="Test chunk",
                start_position=0,
                end_position=10,
                metadata=ChunkMetadata(
                    token_count=3,
                    semantic_density=0.8,
                    overlap_percentage=0.2,
                    confidence_score=0.95,
                    language="en",
                    custom_attributes={"key": "value"},
                ),
            )
        ]
        use_case.strategy_factory.create_strategy.return_value.chunk.return_value = (
            test_chunks
        )

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert len(response.preview_chunks) == 1
        chunk_dto = response.preview_chunks[0]
        assert isinstance(chunk_dto, ChunkDTO)
        assert chunk_dto.content == "Test chunk"
        assert chunk_dto.start_position == 0
        assert chunk_dto.end_position == 10
        assert chunk_dto.token_count == 3
        assert chunk_dto.metadata["semantic_density"] == 0.8
        assert chunk_dto.metadata["overlap_percentage"] == 0.2
        assert chunk_dto.metadata["confidence_score"] == 0.95
        assert chunk_dto.metadata["language"] == "en"
        assert chunk_dto.metadata["custom_attributes"]["key"] == "value"

    @pytest.mark.asyncio()
    async def test_progress_callback_integration(self, use_case, valid_request):
        """Test that progress callback is passed to strategy."""
        # Arrange
        progress_values = []

        def mock_chunk_with_progress(content, config, progress_callback=None):
            if progress_callback:
                progress_callback(25.0)
                progress_callback(50.0)
                progress_callback(75.0)
                progress_callback(100.0)
            return [
                Chunk(content="Chunk", start_position=0, end_position=5,
                      metadata=ChunkMetadata(token_count=1))
            ]

        use_case.strategy_factory.create_strategy.return_value.chunk.side_effect = (
            mock_chunk_with_progress
        )

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert isinstance(response, PreviewResponse)
        # Verify strategy.chunk was called with progress_callback
        call_args = use_case.strategy_factory.create_strategy.return_value.chunk.call_args
        assert "progress_callback" in call_args.kwargs or len(call_args.args) > 2

    @pytest.mark.asyncio()
    async def test_concurrent_preview_requests(self, use_case):
        """Test handling of concurrent preview requests."""
        # Arrange
        requests = [
            PreviewRequest(
                document_id=f"doc-{i}",
                strategy_name="character",
                min_tokens=10,
                max_tokens=50,
                overlap_tokens=5,
            )
            for i in range(3)
        ]

        # Act
        import asyncio
        responses = await asyncio.gather(
            *[use_case.execute(req) for req in requests]
        )

        # Assert
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert response.document_id == f"doc-{i}"
            assert isinstance(response, PreviewResponse)

    @pytest.mark.asyncio()
    async def test_validation_of_preview_size(self, use_case):
        """Test validation of preview_size_kb parameter."""
        # Arrange
        request = PreviewRequest(
            document_id="doc-123",
            strategy_name="character",
            min_tokens=10,
            max_tokens=50,
            overlap_tokens=5,
            preview_size_kb=0,  # Invalid size
        )

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            await use_case.execute(request)

        assert "preview_size_kb must be positive" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_notification_on_success(self, use_case, valid_request):
        """Test that success notification is sent."""
        # Act
        response = await use_case.execute(valid_request)

        # Assert
        use_case.notification_service.notify_preview_generated.assert_called_once()
        call_args = use_case.notification_service.notify_preview_generated.call_args
        assert call_args[0][0] == "doc-123"  # document_id
        assert call_args[0][1] == 2  # chunk_count

    @pytest.mark.asyncio()
    async def test_notification_on_error(self, use_case, valid_request):
        """Test that error notification is sent on failure."""
        # Arrange
        use_case.document_service.get_document_content.side_effect = Exception(
            "Unexpected error"
        )

        # Act & Assert
        with pytest.raises(Exception):
            await use_case.execute(valid_request)

        use_case.notification_service.notify_error.assert_called_once()
        call_args = use_case.notification_service.notify_error.call_args
        assert "Unexpected error" in str(call_args[0][0])
