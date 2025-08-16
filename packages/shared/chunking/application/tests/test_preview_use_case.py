"""
Example test for Preview Chunking Use Case.

Demonstrates testability with mocked dependencies.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from packages.shared.chunking.application.dto.requests import ChunkingStrategy, PreviewRequest
from packages.shared.chunking.application.dto.responses import PreviewResponse
from packages.shared.chunking.application.use_cases.preview_chunking import PreviewChunkingUseCase


@pytest.mark.asyncio()
async def test_preview_success() -> None:
    """Test successful preview generation with mocked dependencies."""
    # Arrange - Create mocked dependencies
    mock_doc_service = Mock()
    mock_doc_service.load_partial = AsyncMock()
    mock_doc_service.extract_text = AsyncMock()
    mock_doc_service.get_metadata = AsyncMock()

    # Mock document loading
    mock_document = Mock()
    mock_doc_service.load_partial.return_value = mock_document
    mock_doc_service.extract_text.return_value = "This is test content for chunking."
    mock_doc_service.get_metadata.return_value = {"size_bytes": 1000}

    # Mock strategy factory
    mock_strategy_factory = Mock()
    mock_strategy = Mock()
    mock_chunk = Mock()
    mock_chunk.content = "This is test"
    # Add metadata attribute with proper structure
    mock_chunk.metadata = Mock()
    mock_chunk.metadata.start_offset = 0
    mock_chunk.metadata.end_offset = 12
    mock_chunk.metadata.token_count = 3
    mock_strategy.chunk = Mock(return_value=[mock_chunk] * 10)
    mock_strategy_factory.create_strategy.return_value = mock_strategy

    # Mock notification service
    mock_notification_service = Mock()
    mock_notification_service.notify_operation_started = AsyncMock()
    mock_notification_service.notify_operation_completed = AsyncMock()
    mock_notification_service.notify_operation_failed = AsyncMock()
    mock_notification_service.notify_error = AsyncMock()
    mock_notification_service.notify_preview_generated = AsyncMock()

    # Create use case
    use_case = PreviewChunkingUseCase(
        document_service=mock_doc_service,
        strategy_factory=mock_strategy_factory,
        notification_service=mock_notification_service,
        metrics_service=None,
    )

    # Create request
    request = PreviewRequest(
        file_path="/test/document.txt",
        strategy_type=ChunkingStrategy.CHARACTER,
        min_tokens=100,
        max_tokens=1000,
        overlap=50,
    )

    # Act - Execute use case
    response = await use_case.execute(request)

    # Assert - Verify response
    assert isinstance(response, PreviewResponse)
    assert response.operation_id is not None
    assert len(response.chunks) == 5  # Should return max_preview_chunks
    assert response.strategy_used == ChunkingStrategy.CHARACTER.value
    assert response.total_chunks_estimate > 0
    assert response.processing_time_ms > 0

    # Verify mocks were called correctly
    mock_doc_service.load_partial.assert_called_once_with(file_path="/test/document.txt", size_kb=10)
    mock_doc_service.extract_text.assert_called_once_with(mock_document)
    mock_strategy_factory.create_strategy.assert_called_once()
    mock_notification_service.notify_operation_started.assert_called_once()
    mock_notification_service.notify_operation_completed.assert_called_once()


@pytest.mark.asyncio()
async def test_preview_validation_error() -> None:
    """Test preview with invalid request parameters."""
    # Arrange
    mock_doc_service = Mock()
    mock_strategy_factory = Mock()
    mock_notification_service = Mock()
    mock_notification_service.notify_operation_failed = AsyncMock()
    mock_notification_service.notify_error = AsyncMock()
    mock_notification_service.notify_preview_generated = AsyncMock()

    use_case = PreviewChunkingUseCase(
        document_service=mock_doc_service,
        strategy_factory=mock_strategy_factory,
        notification_service=mock_notification_service,
    )

    # Invalid request - min_tokens > max_tokens
    request = PreviewRequest(
        file_path="/test/document.txt",
        strategy_type=ChunkingStrategy.CHARACTER,
        min_tokens=1000,
        max_tokens=100,
        overlap=50,
    )

    # Act & Assert
    with pytest.raises(ValueError, match="min_tokens cannot be greater than max_tokens"):
        await use_case.execute(request)
    mock_notification_service.notify_operation_failed.assert_called_once()


@pytest.mark.asyncio()
async def test_preview_file_not_found() -> None:
    """Test preview when document file doesn't exist."""
    # Arrange
    mock_doc_service = Mock()
    mock_doc_service.load_partial = AsyncMock(side_effect=FileNotFoundError("File not found"))

    mock_strategy_factory = Mock()
    mock_notification_service = Mock()
    mock_notification_service.notify_operation_started = AsyncMock()
    mock_notification_service.notify_operation_failed = AsyncMock()
    mock_notification_service.notify_error = AsyncMock()
    mock_notification_service.notify_preview_generated = AsyncMock()

    use_case = PreviewChunkingUseCase(
        document_service=mock_doc_service,
        strategy_factory=mock_strategy_factory,
        notification_service=mock_notification_service,
    )

    request = PreviewRequest(file_path="/nonexistent/document.txt", strategy_type=ChunkingStrategy.CHARACTER)

    # Act & Assert
    with pytest.raises(FileNotFoundError) as exc_info:
        await use_case.execute(request)

    assert "Document not found" in str(exc_info.value)
    mock_notification_service.notify_operation_failed.assert_called_once()


@pytest.mark.asyncio()
async def test_preview_with_metrics_service() -> None:
    """Test preview with metrics collection."""
    # Arrange
    mock_doc_service = Mock()
    mock_doc_service.load_partial = AsyncMock()
    mock_doc_service.extract_text = AsyncMock(return_value="Test content")
    mock_doc_service.get_metadata = AsyncMock(return_value={"size_bytes": 1000})

    mock_document = Mock()
    mock_doc_service.load_partial.return_value = mock_document

    mock_strategy_factory = Mock()
    mock_strategy = Mock()
    mock_chunk = Mock()
    mock_chunk.content = "Test"
    # Add metadata attribute with proper structure
    mock_chunk.metadata = Mock()
    mock_chunk.metadata.start_offset = 0
    mock_chunk.metadata.end_offset = 4
    mock_chunk.metadata.token_count = 1
    mock_strategy.chunk = Mock(return_value=[mock_chunk] * 3)
    mock_strategy_factory.create_strategy.return_value = mock_strategy

    mock_notification_service = Mock()
    mock_notification_service.notify_operation_started = AsyncMock()
    mock_notification_service.notify_operation_completed = AsyncMock()
    mock_notification_service.notify_operation_failed = AsyncMock()
    mock_notification_service.notify_error = AsyncMock()
    mock_notification_service.notify_preview_generated = AsyncMock()

    # Mock metrics service
    mock_metrics_service = Mock()
    mock_metrics_service.record_operation_duration = AsyncMock()
    mock_metrics_service.record_strategy_performance = AsyncMock()

    use_case = PreviewChunkingUseCase(
        document_service=mock_doc_service,
        strategy_factory=mock_strategy_factory,
        notification_service=mock_notification_service,
        metrics_service=mock_metrics_service,
    )

    request = PreviewRequest(file_path="/test/document.txt", strategy_type=ChunkingStrategy.SEMANTIC)

    # Act
    response = await use_case.execute(request)

    # Assert
    assert response is not None

    # Verify metrics were recorded
    mock_metrics_service.record_operation_duration.assert_called_once()
    mock_metrics_service.record_strategy_performance.assert_called_once()

    # Verify strategy performance metrics
    call_args = mock_metrics_service.record_strategy_performance.call_args
    assert call_args.kwargs["strategy_type"] == ChunkingStrategy.SEMANTIC.value
    assert call_args.kwargs["chunks_created"] == 3
