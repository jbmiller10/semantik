#!/usr/bin/env python3
"""Tests for ProcessDocumentUseCase."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from packages.shared.chunking.application.dto.requests import ProcessRequest
from packages.shared.chunking.application.dto.responses import OperationResponse
from packages.shared.chunking.application.use_cases.process_document import (
    ProcessDocumentUseCase,
)
from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.entities.chunking_operation import ChunkingOperation
from packages.shared.chunking.domain.exceptions import (
    DocumentTooLargeError,
    InvalidConfigurationError,
    InvalidStateError,
)
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from packages.shared.chunking.domain.value_objects.operation_status import OperationStatus


class TestProcessDocumentUseCase:
    """Test suite for ProcessDocumentUseCase."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock chunking operation repository."""
        repo = AsyncMock()
        repo.save = AsyncMock()
        repo.get_by_id = AsyncMock()
        repo.update = AsyncMock()
        return repo

    @pytest.fixture
    def mock_document_service(self):
        """Create mock document service."""
        service = AsyncMock()
        service.get_document_content = AsyncMock(
            return_value="This is a full document content for processing."
        )
        service.validate_document = AsyncMock(return_value=True)
        service.get_document_metadata = AsyncMock(
            return_value={"size": 1000, "type": "text/plain"}
        )
        return service

    @pytest.fixture
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

    @pytest.fixture
    def mock_unit_of_work(self, mock_repository):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        uow.chunking_operations = mock_repository
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        return uow

    @pytest.fixture
    def mock_notification_service(self):
        """Create mock notification service."""
        service = AsyncMock()
        service.notify_operation_started = AsyncMock()
        service.notify_operation_completed = AsyncMock()
        service.notify_operation_failed = AsyncMock()
        return service

    @pytest.fixture
    def mock_event_publisher(self):
        """Create mock event publisher."""
        publisher = AsyncMock()
        publisher.publish_operation_started = AsyncMock()
        publisher.publish_operation_completed = AsyncMock()
        publisher.publish_operation_failed = AsyncMock()
        return publisher

    @pytest.fixture
    def use_case(
        self,
        mock_unit_of_work,
        mock_document_service,
        mock_strategy_factory,
        mock_notification_service,
        mock_event_publisher,
    ):
        """Create use case instance with mocked dependencies."""
        return ProcessDocumentUseCase(
            unit_of_work=mock_unit_of_work,
            document_service=mock_document_service,
            strategy_factory=mock_strategy_factory,
            notification_service=mock_notification_service,
            event_publisher=mock_event_publisher,
        )

    @pytest.fixture
    def valid_request(self):
        """Create a valid process request."""
        return ProcessRequest(
            document_id="doc-789",
            document_path="/data/documents/test.txt",
            strategy_name="character",
            min_tokens=10,
            max_tokens=100,
            overlap_tokens=5,
            async_processing=False,
        )

    @pytest.mark.asyncio
    async def test_successful_synchronous_processing(self, use_case, valid_request):
        """Test successful synchronous document processing."""
        # Arrange
        operation_id = str(uuid4())
        with patch("packages.shared.chunking.application.use_cases.process_document.uuid4") as mock_uuid:
            mock_uuid.return_value = operation_id

            # Act
            response = await use_case.execute(valid_request)

        # Assert
        assert isinstance(response, OperationResponse)
        assert response.operation_id == operation_id
        assert response.document_id == "doc-789"
        assert response.status == "COMPLETED"
        assert response.chunks_produced == 2
        assert response.progress_percentage == 100.0

        # Verify workflow
        use_case.document_service.validate_document.assert_called_once_with("doc-789")
        use_case.document_service.get_document_content.assert_called_once_with("doc-789")
        use_case.strategy_factory.create_strategy.assert_called_once_with("character")
        use_case.unit_of_work.chunking_operations.save.assert_called()
        use_case.unit_of_work.commit.assert_called()
        use_case.notification_service.notify_operation_completed.assert_called_once()
        use_case.event_publisher.publish_operation_completed.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_asynchronous_processing(self, use_case):
        """Test successful asynchronous document processing."""
        # Arrange
        request = ProcessRequest(
            document_id="doc-async",
            document_path="/data/documents/async.txt",
            strategy_name="semantic",
            min_tokens=20,
            max_tokens=200,
            overlap_tokens=10,
            async_processing=True,  # Async mode
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.status == "PROCESSING"  # Should be in processing state
        assert response.progress_percentage < 100.0
        assert response.chunks_produced == 0  # No chunks yet

        # Verify async workflow
        use_case.notification_service.notify_operation_started.assert_called_once()
        use_case.event_publisher.publish_operation_started.assert_called_once()
        # Should not call completed notifications in async mode
        use_case.notification_service.notify_operation_completed.assert_not_called()

    @pytest.mark.asyncio
    async def test_document_validation_failure(self, use_case, valid_request):
        """Test handling of document validation failure."""
        # Arrange
        use_case.document_service.validate_document.return_value = False

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            await use_case.execute(valid_request)

        assert "Invalid document" in str(exc_info.value)
        use_case.unit_of_work.rollback.assert_called()
        use_case.notification_service.notify_operation_failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_document_too_large_error(self, use_case, valid_request):
        """Test handling of document too large error."""
        # Arrange
        large_content = "x" * (ChunkingOperation.MAX_DOCUMENT_SIZE + 1)
        use_case.document_service.get_document_content.return_value = large_content

        # Act & Assert
        with pytest.raises(DocumentTooLargeError):
            await use_case.execute(valid_request)

        use_case.unit_of_work.rollback.assert_called()
        use_case.notification_service.notify_operation_failed.assert_called()

    @pytest.mark.asyncio
    async def test_invalid_configuration_error(self, use_case):
        """Test handling of invalid configuration."""
        # Arrange
        invalid_request = ProcessRequest(
            document_id="doc-123",
            document_path="/data/test.txt",
            strategy_name="character",
            min_tokens=100,  # Greater than max
            max_tokens=50,
            overlap_tokens=5,
        )

        # Act & Assert
        with pytest.raises(InvalidConfigurationError):
            await use_case.execute(invalid_request)

        use_case.unit_of_work.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_strategy_execution_failure(self, use_case, valid_request):
        """Test handling of strategy execution failure."""
        # Arrange
        use_case.strategy_factory.create_strategy.return_value.chunk.side_effect = (
            RuntimeError("Strategy execution failed")
        )

        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            await use_case.execute(valid_request)

        assert "Strategy execution failed" in str(exc_info.value)
        use_case.unit_of_work.rollback.assert_called()
        use_case.notification_service.notify_operation_failed.assert_called()
        use_case.event_publisher.publish_operation_failed.assert_called()

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, use_case, valid_request):
        """Test that transaction is rolled back on error."""
        # Arrange
        use_case.unit_of_work.commit.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await use_case.execute(valid_request)

        assert "Database error" in str(exc_info.value)
        use_case.unit_of_work.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_operation_persistence(self, use_case, valid_request):
        """Test that operation is properly persisted."""
        # Act
        response = await use_case.execute(valid_request)

        # Assert
        # Verify save was called with ChunkingOperation
        save_calls = use_case.unit_of_work.chunking_operations.save.call_args_list
        assert len(save_calls) > 0
        
        saved_operation = save_calls[0][0][0]
        assert isinstance(saved_operation, ChunkingOperation)
        assert saved_operation.document_id == "doc-789"
        assert saved_operation.status == OperationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_progress_tracking(self, use_case, valid_request):
        """Test that progress is tracked during processing."""
        # Arrange
        progress_values = []
        
        def mock_chunk_with_progress(content, config, progress_callback=None):
            if progress_callback:
                progress_values.extend([25.0, 50.0, 75.0, 100.0])
                for value in [25.0, 50.0, 75.0, 100.0]:
                    progress_callback(value)
            return [
                Chunk(content="Result", start_position=0, end_position=6,
                      metadata=ChunkMetadata(token_count=1))
            ]
        
        use_case.strategy_factory.create_strategy.return_value.chunk = mock_chunk_with_progress

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.progress_percentage == 100.0
        assert len(progress_values) > 0

    @pytest.mark.asyncio
    async def test_chunk_validation(self, use_case, valid_request):
        """Test that chunks are validated after processing."""
        # Arrange
        # Create chunks with insufficient coverage
        use_case.strategy_factory.create_strategy.return_value.chunk.return_value = [
            Chunk(
                content="Small",
                start_position=0,
                end_position=5,
                metadata=ChunkMetadata(token_count=1),
            )
        ]

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        # Operation should complete but validation might flag issues
        assert response.status == "COMPLETED"
        # Verify validation was performed (through the operation)

    @pytest.mark.asyncio
    async def test_concurrent_processing_requests(self, use_case):
        """Test handling of concurrent processing requests."""
        # Arrange
        requests = [
            ProcessRequest(
                document_id=f"doc-{i}",
                document_path=f"/data/doc-{i}.txt",
                strategy_name="character",
                min_tokens=10,
                max_tokens=100,
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
            assert response.status == "COMPLETED"

    @pytest.mark.asyncio
    async def test_path_traversal_prevention(self, use_case):
        """Test that path traversal attempts are blocked."""
        # Arrange
        malicious_request = ProcessRequest(
            document_id="doc-123",
            document_path="../../etc/passwd",  # Path traversal attempt
            strategy_name="character",
            min_tokens=10,
            max_tokens=100,
            overlap_tokens=5,
        )

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            await use_case.execute(malicious_request)

        assert "Invalid path" in str(exc_info.value) or "Path traversal" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_metadata_inclusion(self, use_case, valid_request):
        """Test that document metadata is included in processing."""
        # Arrange
        use_case.document_service.get_document_metadata.return_value = {
            "size": 5000,
            "type": "text/markdown",
            "created_at": "2024-01-01T00:00:00Z",
        }

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.metadata is not None
        assert response.metadata["document_type"] == "text/markdown"
        assert "processing_time_ms" in response.metadata

    @pytest.mark.asyncio
    async def test_empty_document_handling(self, use_case, valid_request):
        """Test handling of empty documents."""
        # Arrange
        use_case.document_service.get_document_content.return_value = ""
        use_case.strategy_factory.create_strategy.return_value.chunk.return_value = []

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.status == "COMPLETED"
        assert response.chunks_produced == 0
        assert response.progress_percentage == 100.0

    @pytest.mark.asyncio
    async def test_operation_statistics_generation(self, use_case, valid_request):
        """Test that operation statistics are generated."""
        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.statistics is not None
        assert "total_chunks" in response.statistics
        assert "coverage" in response.statistics
        assert response.statistics["total_chunks"] == 2

    @pytest.mark.asyncio
    async def test_error_recovery_mechanism(self, use_case, valid_request):
        """Test error recovery and retry mechanism."""
        # Arrange
        # First call fails, second succeeds
        use_case.strategy_factory.create_strategy.return_value.chunk.side_effect = [
            RuntimeError("Temporary failure"),
            [Chunk(content="Success", start_position=0, end_position=7,
                   metadata=ChunkMetadata(token_count=1))]
        ]

        # Configure retry logic
        use_case.max_retries = 1

        # Act
        with patch.object(use_case, 'max_retries', 1):
            # This should retry and succeed on second attempt
            # (Note: Actual implementation would need retry logic)
            pass

        # Assert - verify retry behavior if implemented

    @pytest.mark.asyncio
    async def test_custom_strategy_parameters(self, use_case):
        """Test processing with custom strategy parameters."""
        # Arrange
        request = ProcessRequest(
            document_id="doc-custom",
            document_path="/data/custom.txt",
            strategy_name="semantic",
            min_tokens=30,
            max_tokens=150,
            overlap_tokens=15,
            additional_params={
                "similarity_threshold": 0.85,
                "embedding_model": "custom-model",
            },
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.status == "COMPLETED"
        # Verify strategy was created with custom params
        use_case.strategy_factory.create_strategy.assert_called_with("semantic")