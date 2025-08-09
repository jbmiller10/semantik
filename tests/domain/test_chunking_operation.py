#!/usr/bin/env python3
"""Tests for ChunkingOperation entity."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.entities.chunking_operation import ChunkingOperation
from packages.shared.chunking.domain.exceptions import (
    DocumentTooLargeError,
    InvalidStateError)
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from packages.shared.chunking.domain.value_objects.operation_status import OperationStatus


class TestChunkingOperation:
    """Test suite for ChunkingOperation entity."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid chunk configuration."""
        return ChunkConfig(
            strategy_name="character",
            min_tokens=10,
            max_tokens=100,
            overlap_tokens=5)

    @pytest.fixture
    def sample_document(self):
        """Create sample document content."""
        return "This is a sample document content for testing chunking operations."

    @pytest.fixture
    def chunking_operation(self, valid_config, sample_document):
        """Create a chunking operation instance."""
        return ChunkingOperation(
            operation_id="test-op-123",
            document_id="doc-456",
            document_content=sample_document,
            config=valid_config)

    def test_initialization_success(self, valid_config, sample_document):
        """Test successful initialization of ChunkingOperation."""
        # Arrange & Act
        operation = ChunkingOperation(
            operation_id="test-op-123",
            document_id="doc-456",
            document_content=sample_document,
            config=valid_config)

        # Assert
        assert operation.id == "test-op-123"
        assert operation.document_id == "doc-456"
        assert operation.status == OperationStatus.PENDING
        assert operation.config == valid_config
        assert operation.progress_percentage == 0.0
        assert operation.error_message is None
        assert operation.chunk_collection.chunk_count == 0

    def test_initialization_with_large_document_fails(self, valid_config):
        """Test that initialization fails with document exceeding size limit."""
        # Arrange
        large_content = "x" * (ChunkingOperation.MAX_DOCUMENT_SIZE + 1)

        # Act & Assert
        with pytest.raises(DocumentTooLargeError) as exc_info:
            ChunkingOperation(
                operation_id="test-op",
                document_id="doc-1",
                document_content=large_content,
                config=valid_config)
        
        assert str(exc_info.value).startswith("Document size")
        assert str(ChunkingOperation.MAX_DOCUMENT_SIZE) in str(exc_info.value)

    def test_start_operation_success(self, chunking_operation):
        """Test starting a chunking operation."""
        # Arrange
        assert chunking_operation.status == OperationStatus.PENDING

        # Act
        chunking_operation.start()

        # Assert
        assert chunking_operation.status == OperationStatus.PROCESSING
        assert chunking_operation._started_at is not None
        assert chunking_operation.progress_percentage == 0.0

    def test_start_operation_from_invalid_state_fails(self, chunking_operation):
        """Test that starting from invalid state raises error."""
        # Arrange
        chunking_operation._status = OperationStatus.COMPLETED

        # Act & Assert
        with pytest.raises(InvalidStateError) as exc_info:
            chunking_operation.start()
        
        assert "Cannot start operation in COMPLETED state" in str(exc_info.value)

    def test_execute_with_strategy_success(self, chunking_operation):
        """Test executing chunking with a strategy."""
        # Arrange
        mock_strategy = MagicMock()
        mock_chunks = [
            Chunk(
                content="Chunk 1", metadata=ChunkMetadata(
                    chunk_id="chunk-1",
                    document_id="doc-456",
                    chunk_index=0,
                    start_offset=0,
                    end_offset=7,
                    token_count=2,
                    strategy_name="character",
                    semantic_score=0.8), min_tokens=1),
            Chunk(
                content="Chunk 2", metadata=ChunkMetadata(
                    chunk_id="chunk-2",
                    document_id="doc-456",
                    chunk_index=1,
                    start_offset=8,
                    end_offset=15,
                    token_count=2,
                    strategy_name="character",
                    semantic_score=0.7), min_tokens=1),
        ]
        mock_strategy.chunk.return_value = mock_chunks
        
        chunking_operation.start()

        # Act
        chunking_operation.execute(mock_strategy)

        # Assert
        assert chunking_operation.status == OperationStatus.COMPLETED
        assert chunking_operation.progress_percentage == 100.0
        assert chunking_operation.chunk_collection.chunk_count == 2
        assert chunking_operation._completed_at is not None
        assert "duration_seconds" in chunking_operation.metrics
        
        # Verify strategy was called
        mock_strategy.chunk.assert_called_once()

    def test_execute_without_processing_state_fails(self, chunking_operation):
        """Test that execute fails if not in PROCESSING state."""
        # Arrange
        mock_strategy = MagicMock()

        # Act & Assert
        with pytest.raises(InvalidStateError) as exc_info:
            chunking_operation.execute(mock_strategy)
        
        assert "Cannot execute operation in PENDING state" in str(exc_info.value)

    def test_execute_with_too_many_chunks_fails(self, chunking_operation):
        """Test that execute fails when producing too many chunks."""
        # Arrange
        mock_strategy = MagicMock()
        # Create more chunks than allowed
        excessive_chunks = [
            Chunk(
                content=f"Chunk {i}",
                metadata=ChunkMetadata(
                    chunk_id=f"chunk-{i}",
                    document_id="doc-456",
                    chunk_index=i,
                    start_offset=i * 2,
                    end_offset=(i + 1) * 2,
                    token_count=2,
                    strategy_name="character"), min_tokens=1)
            for i in range(ChunkingOperation.MAX_CHUNKS_PER_OPERATION + 1)
        ]
        mock_strategy.chunk.return_value = excessive_chunks
        
        chunking_operation.start()

        # Act & Assert
        with pytest.raises(InvalidStateError) as exc_info:
            chunking_operation.execute(mock_strategy)
        
        assert "exceeding limit" in str(exc_info.value)
        assert chunking_operation.status == OperationStatus.FAILED

    def test_execute_with_strategy_exception_propagates(self, chunking_operation):
        """Test that strategy exceptions are properly handled."""
        # Arrange
        mock_strategy = MagicMock()
        mock_strategy.chunk.side_effect = ValueError("Strategy error")
        
        chunking_operation.start()

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            chunking_operation.execute(mock_strategy)
        
        assert "Strategy error" in str(exc_info.value)
        assert chunking_operation.status == OperationStatus.FAILED
        assert chunking_operation.error_message == "Strategy error"
        assert chunking_operation._error_details["exception_type"] == "ValueError"

    def test_add_chunk_success(self, chunking_operation):
        """Test adding chunks to an operation."""
        # Arrange
        chunk = Chunk(
            content="Test chunk", metadata=ChunkMetadata(
                chunk_id="chunk-test",
                document_id="doc-456",
                chunk_index=0,
                start_offset=0,
                end_offset=5,
                token_count=3,
                strategy_name="character"), min_tokens=1)
        chunking_operation.start()

        # Act
        chunking_operation.add_chunk(chunk)

        # Assert
        assert chunking_operation.chunk_collection.chunk_count == 1
        assert chunking_operation.progress_percentage > 0

    def test_add_chunk_invalid_state_fails(self, chunking_operation):
        """Test that adding chunks fails in invalid state."""
        # Arrange
        chunk = Chunk(
            content="Test chunk", metadata=ChunkMetadata(
                chunk_id="chunk-test",
                document_id="doc-456",
                chunk_index=0,
                start_offset=0,
                end_offset=5,
                token_count=3,
                strategy_name="character"), min_tokens=1)

        # Act & Assert
        with pytest.raises(InvalidStateError) as exc_info:
            chunking_operation.add_chunk(chunk)
        
        assert "Cannot add chunks to operation in PENDING state" in str(exc_info.value)

    def test_cancel_operation_success(self, chunking_operation):
        """Test cancelling an operation."""
        # Arrange
        chunking_operation.start()

        # Act
        chunking_operation.cancel("User requested cancellation")

        # Assert
        assert chunking_operation.status == OperationStatus.CANCELLED
        assert chunking_operation._completed_at is not None
        assert chunking_operation.error_message == "User requested cancellation"

    def test_cancel_from_invalid_state_fails(self, chunking_operation):
        """Test that cancelling from invalid state raises error."""
        # Arrange
        chunking_operation._status = OperationStatus.COMPLETED

        # Act & Assert
        with pytest.raises(InvalidStateError) as exc_info:
            chunking_operation.cancel()
        
        assert "Cannot cancel operation in COMPLETED state" in str(exc_info.value)

    def test_validate_results_with_valid_chunks(self, chunking_operation):
        """Test validation with valid chunking results."""
        # Arrange
        # Document is "Test document content for testing operations" (44 chars)
        doc_len = len(chunking_operation._document_content)
        chunks = [
            Chunk(
                content=chunking_operation._document_content[0:30], 
                metadata=ChunkMetadata(
                    chunk_id="chunk-1",
                    document_id="doc-456",
                    chunk_index=0,
                    start_offset=0,
                    end_offset=30,  # First 30 characters
                    token_count=6,  # Approximately 6 words
                    strategy_name="character"), min_tokens=1),
            Chunk(
                content=chunking_operation._document_content[25:doc_len],  # Overlap from 25-30, then 30-44
                metadata=ChunkMetadata(
                    chunk_id="chunk-2",
                    document_id="doc-456",
                    chunk_index=1,
                    start_offset=25,  # Overlaps with first chunk
                    end_offset=doc_len,  # To the end of document (44)
                    token_count=4,  # Approximately 4 words
                    strategy_name="character"), min_tokens=1),
        ]
        
        chunking_operation.start()
        for chunk in chunks:
            chunking_operation._chunk_collection.add_chunk(chunk)

        # Act
        is_valid, issues = chunking_operation.validate_results()

        # Assert
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_results_with_no_chunks(self, chunking_operation):
        """Test validation fails when no chunks produced."""
        # Arrange
        chunking_operation.start()

        # Act
        is_valid, issues = chunking_operation.validate_results()

        # Assert
        assert is_valid is False
        assert "No chunks were produced" in issues

    def test_validate_results_with_insufficient_coverage(self, chunking_operation):
        """Test validation fails with insufficient coverage."""
        # Arrange
        # Add a chunk that covers only a small portion
        chunk = Chunk(
            content="This", metadata=ChunkMetadata(
                chunk_id="chunk-error",
                document_id="doc-456",
                chunk_index=0,
                start_offset=0,
                end_offset=5,
                token_count=1,
                strategy_name="character"), min_tokens=1)
        
        chunking_operation.start()
        chunking_operation._chunk_collection.add_chunk(chunk)

        # Act
        is_valid, issues = chunking_operation.validate_results()

        # Assert
        assert is_valid is False
        assert any("Insufficient coverage" in issue for issue in issues)

    @patch("packages.shared.chunking.domain.entities.chunking_operation.datetime")
    def test_validate_results_with_timeout(self, mock_datetime, chunking_operation):
        """Test validation fails when operation times out."""
        # Arrange
        start_time = datetime.utcnow()
        timeout_time = start_time + timedelta(
            seconds=ChunkingOperation.MAX_OPERATION_DURATION_SECONDS + 1
        )
        
        mock_datetime.utcnow.side_effect = [start_time, timeout_time]
        
        chunking_operation.start()
        chunking_operation._started_at = start_time

        # Act
        is_valid, issues = chunking_operation.validate_results()

        # Assert
        assert is_valid is False
        assert any("Operation exceeded timeout" in issue for issue in issues)

    def test_get_statistics_pending(self, chunking_operation):
        """Test getting statistics for pending operation."""
        # Act
        stats = chunking_operation.get_statistics()

        # Assert
        assert stats["operation_id"] == "test-op-123"
        assert stats["document_id"] == "doc-456"
        assert stats["status"] == "PENDING"
        assert stats["chunks"]["total"] == 0
        assert stats["progress"] == 0.0
        assert "timing" not in stats

    def test_get_statistics_processing(self, chunking_operation):
        """Test getting statistics for processing operation."""
        # Arrange
        chunking_operation.start()
        chunk = Chunk(
            content="Test chunk", metadata=ChunkMetadata(
                chunk_id="chunk-test",
                document_id="doc-456",
                chunk_index=0,
                start_offset=0,
                end_offset=5,
                token_count=3,
                strategy_name="character"), min_tokens=1)
        chunking_operation.add_chunk(chunk)

        # Act
        stats = chunking_operation.get_statistics()

        # Assert
        assert stats["status"] == "PROCESSING"
        assert stats["chunks"]["total"] == 1
        assert stats["progress"] > 0
        assert "timing" in stats
        assert "started_at" in stats["timing"]
        assert "duration_seconds" in stats["timing"]

    def test_get_statistics_completed(self, chunking_operation):
        """Test getting statistics for completed operation."""
        # Arrange
        mock_strategy = MagicMock()
        mock_chunks = [
            Chunk(
                content="Chunk 1", metadata=ChunkMetadata(
                    chunk_id="chunk-0",
                    document_id="doc-456",
                    chunk_index=0,
                    start_offset=0,
                    end_offset=2,
                    token_count=2,
                    strategy_name="character"), min_tokens=1),
        ]
        mock_strategy.chunk.return_value = mock_chunks
        
        chunking_operation.start()
        chunking_operation.execute(mock_strategy)

        # Act
        stats = chunking_operation.get_statistics()

        # Assert
        assert stats["status"] == "COMPLETED"
        assert stats["chunks"]["total"] == 1
        assert stats["progress"] == 100.0
        assert "timing" in stats
        assert "completed_at" in stats["timing"]
        assert "metrics" in stats
        assert "chunk_stats" in stats
        assert "overlap_stats" in stats

    def test_get_statistics_failed(self, chunking_operation):
        """Test getting statistics for failed operation."""
        # Arrange
        chunking_operation.start()
        chunking_operation._fail("Test error", {"detail": "Error detail"})

        # Act
        stats = chunking_operation.get_statistics()

        # Assert
        assert stats["status"] == "FAILED"
        assert "error" in stats
        assert stats["error"]["message"] == "Test error"
        assert stats["error"]["details"]["detail"] == "Error detail"

    def test_estimate_chunks(self, chunking_operation):
        """Test chunk estimation."""
        # Act
        estimated = chunking_operation._estimate_chunks()

        # Assert
        # For the sample document "This is a sample document content for testing chunking operations."
        # ~67 chars / 4 = ~16 tokens
        # With max_tokens=100, min estimate would be 1 chunk
        assert estimated > 0
        assert estimated == chunking_operation._estimated_total_chunks

    def test_calculate_duration_not_started(self, chunking_operation):
        """Test duration calculation when not started."""
        # Act
        duration = chunking_operation._calculate_duration()

        # Assert
        assert duration == 0.0

    def test_calculate_duration_in_progress(self, chunking_operation):
        """Test duration calculation when in progress."""
        # Arrange
        chunking_operation.start()

        # Act
        duration = chunking_operation._calculate_duration()

        # Assert
        assert duration >= 0.0

    def test_calculate_duration_completed(self, chunking_operation):
        """Test duration calculation when completed."""
        # Arrange
        mock_strategy = MagicMock()
        mock_strategy.chunk.return_value = []
        
        chunking_operation.start()
        start_time = chunking_operation._started_at
        chunking_operation.execute(mock_strategy)
        
        # Act
        duration = chunking_operation._calculate_duration()

        # Assert
        assert duration >= 0.0
        expected_duration = (chunking_operation._completed_at - start_time).total_seconds()
        assert abs(duration - expected_duration) < 0.01

    def test_progress_update_clamping(self, chunking_operation):
        """Test that progress is clamped between 0 and 100."""
        # Arrange
        chunking_operation.start()

        # Act & Assert - test lower bound
        chunking_operation._update_progress(-10.0)
        assert chunking_operation.progress_percentage == 0.0

        # Act & Assert - test upper bound
        chunking_operation._update_progress(150.0)
        assert chunking_operation.progress_percentage == 100.0

        # Act & Assert - test normal value
        chunking_operation._update_progress(50.0)
        assert chunking_operation.progress_percentage == 50.0

    def test_repr(self, chunking_operation):
        """Test string representation of operation."""
        # Act
        repr_str = repr(chunking_operation)

        # Assert
        assert "ChunkingOperation" in repr_str
        assert "id=test-op-123" in repr_str
        assert "status=PENDING" in repr_str
        assert "chunks=0" in repr_str
        assert "progress=0.0%" in repr_str

    def test_metrics_calculation(self, chunking_operation):
        """Test performance metrics calculation."""
        # Arrange
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(seconds=2)
        chunk_count = 10

        # Act
        chunking_operation._update_metrics(start_time, end_time, chunk_count)

        # Assert
        metrics = chunking_operation.metrics
        assert metrics["duration_seconds"] == 2.0
        assert metrics["chunks_per_second"] == 5.0
        assert metrics["characters_per_second"] > 0
        assert metrics["average_chunk_size"] > 0

    def test_state_transitions_enforcement(self):
        """Test that state transitions are properly enforced."""
        # This test verifies the business rule that states can only transition
        # according to the allowed paths defined in OperationStatus
        
        # Test all valid transitions
        valid_transitions = [
            (OperationStatus.PENDING, OperationStatus.PROCESSING),
            (OperationStatus.PENDING, OperationStatus.CANCELLED),
            (OperationStatus.PROCESSING, OperationStatus.COMPLETED),
            (OperationStatus.PROCESSING, OperationStatus.FAILED),
            (OperationStatus.PROCESSING, OperationStatus.CANCELLED),
        ]
        
        for from_status, to_status in valid_transitions:
            assert from_status.can_transition_to(to_status), \
                f"Transition from {from_status} to {to_status} should be valid"
        
        # Test invalid transitions
        invalid_transitions = [
            (OperationStatus.COMPLETED, OperationStatus.PROCESSING),
            (OperationStatus.FAILED, OperationStatus.COMPLETED),
            (OperationStatus.CANCELLED, OperationStatus.PROCESSING),
        ]
        
        for from_status, to_status in invalid_transitions:
            assert not from_status.can_transition_to(to_status), \
                f"Transition from {from_status} to {to_status} should be invalid"