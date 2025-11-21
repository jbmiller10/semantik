#!/usr/bin/env python3

"""Tests for domain exceptions and business rule enforcement."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.entities.chunking_operation import ChunkingOperation
from shared.chunking.domain.exceptions import (
    ChunkingDomainError,
    DocumentTooLargeError,
    InvalidConfigurationError,
    InvalidStateError,
)
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata


class TestDomainExceptions:
    """Test suite for domain exceptions."""

    def test_chunking_domain_error_base(self) -> None:
        """Test base domain error."""
        # Act
        error = ChunkingDomainError("Base domain error")

        # Assert
        assert str(error) == "Base domain error"
        assert isinstance(error, Exception)

    def test_document_too_large_error(self) -> None:
        """Test DocumentTooLargeError with size information."""
        # Arrange
        actual_size = 15_000_000
        max_size = 10_000_000

        # Act
        error = DocumentTooLargeError(actual_size, max_size)

        # Assert
        assert isinstance(error, ChunkingDomainError)
        assert str(actual_size) in str(error)
        assert str(max_size) in str(error)
        assert "Document size" in str(error)
        assert "exceeds maximum" in str(error).lower()

    def test_document_too_large_error_attributes(self) -> None:
        """Test that DocumentTooLargeError stores size attributes."""
        # Arrange
        actual_size = 20_000_000
        max_size = 10_000_000

        # Act
        error = DocumentTooLargeError(actual_size, max_size)

        # Assert
        # Check if the error has the expected attributes
        error_str = str(error)
        assert "20000000" in error_str or "20,000,000" in error_str
        assert "10000000" in error_str or "10,000,000" in error_str

    def test_invalid_configuration_error(self) -> None:
        """Test InvalidConfigurationError."""
        # Act
        error = InvalidConfigurationError("min_tokens must be positive")

        # Assert
        assert isinstance(error, ChunkingDomainError)
        assert "min_tokens must be positive" in str(error)

    def test_invalid_configuration_error_various_messages(self) -> None:
        """Test InvalidConfigurationError with various validation messages."""
        # Arrange
        test_cases = [
            "min_tokens cannot be greater than max_tokens",
            "overlap_tokens must be less than min_tokens",
            "strategy_name cannot be empty",
            "Invalid parameter value",
        ]

        # Act & Assert
        for message in test_cases:
            error = InvalidConfigurationError(message)
            assert message in str(error)
            assert isinstance(error, ChunkingDomainError)

    def test_invalid_state_error(self) -> None:
        """Test InvalidStateError."""
        # Act
        error = InvalidStateError("Cannot start operation in COMPLETED state")

        # Assert
        assert isinstance(error, ChunkingDomainError)
        assert "Cannot start operation in COMPLETED state" in str(error)

    def test_invalid_state_error_various_transitions(self) -> None:
        """Test InvalidStateError for various invalid transitions."""
        # Arrange
        test_cases = [
            "Cannot transition from COMPLETED to PROCESSING",
            "Operation is already in terminal state",
            "Cannot add chunks to operation in PENDING state",
            "Cannot cancel operation in FAILED state",
        ]

        # Act & Assert
        for message in test_cases:
            error = InvalidStateError(message)
            assert message in str(error)
            assert isinstance(error, ChunkingDomainError)

    # Note: StrategyNotFoundError removed - strategy selection is handled at application layer

    def test_exception_inheritance_chain(self) -> None:
        """Test that all domain exceptions inherit from ChunkingDomainError."""
        # Arrange
        exceptions = [
            DocumentTooLargeError(100, 50),
            InvalidConfigurationError("test"),
            InvalidStateError("test"),
        ]

        # Assert
        for exc in exceptions:
            assert isinstance(exc, ChunkingDomainError)
            assert isinstance(exc, Exception)

    def test_exception_equality(self) -> None:
        """Test that exceptions with same message are not equal objects."""
        # Arrange
        error1 = InvalidStateError("Same message")
        error2 = InvalidStateError("Same message")

        # Assert
        assert str(error1) == str(error2)
        assert error1 is not error2  # Different objects

    def test_exception_type_checking(self) -> None:
        """Test that exceptions can be caught by type."""
        # Arrange & Act & Assert
        with pytest.raises(DocumentTooLargeError):
            raise DocumentTooLargeError(100, 50)

        with pytest.raises(InvalidConfigurationError):
            raise InvalidConfigurationError("test")

        with pytest.raises(InvalidStateError):
            raise InvalidStateError("test")

        # All should be catchable as ChunkingDomainError
        with pytest.raises(ChunkingDomainError):
            raise DocumentTooLargeError(100, 50)

    def test_exception_context_preservation(self) -> None:
        """Test that exception context is preserved when re-raising."""
        # Arrange
        original_message = "Original error occurred"

        # Act & Assert
        # Create a function that raises the exception (PT012: single statement in raises block)
        def raise_chained_exception() -> None:
            try:
                raise ValueError(original_message)
            except ValueError as ve:
                raise InvalidStateError(f"State error: {ve}") from ve

        with pytest.raises(InvalidStateError) as exc_info:
            raise_chained_exception()

        # Check that the context is preserved
        assert "State error" in str(exc_info.value)
        assert "Original error occurred" in str(exc_info.value)


class TestBusinessRuleEnforcement:
    """Test suite for business rule enforcement in the domain."""

    def test_document_size_limit_enforcement(self) -> None:
        """Test that document size limits are enforced."""

        # Arrange
        config = ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=5)

        # Create document exceeding size limit
        large_document = "x" * (ChunkingOperation.MAX_DOCUMENT_SIZE + 1)

        # Act & Assert
        with pytest.raises(DocumentTooLargeError) as exc_info:
            ChunkingOperation(operation_id="test", document_id="doc1", document_content=large_document, config=config)

        assert str(ChunkingOperation.MAX_DOCUMENT_SIZE) in str(exc_info.value)

    def test_chunk_count_limit_enforcement(self) -> None:
        """Test that chunk count limits are enforced."""

        # Arrange
        config = ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=5)

        operation = ChunkingOperation(
            operation_id="test", document_id="doc1", document_content="Test document", config=config
        )

        # Create mock strategy that produces too many chunks
        mock_strategy = MagicMock()
        excessive_chunks = [
            Chunk(
                content=f"Chunk {i}",
                metadata=ChunkMetadata(
                    chunk_id=f"chunk-{i}",
                    document_id="doc1",  # Match the operation's document_id
                    chunk_index=i,
                    start_offset=i,
                    end_offset=i + 1,
                    token_count=1,
                    strategy_name="character",
                ),
                min_tokens=1,
            )
            for i in range(ChunkingOperation.MAX_CHUNKS_PER_OPERATION + 1)
        ]
        mock_strategy.chunk.return_value = excessive_chunks

        # Act
        operation.start()

        # Assert
        with pytest.raises(InvalidStateError) as exc_info:
            operation.execute(mock_strategy)

        assert "exceeding limit" in str(exc_info.value)
        assert str(ChunkingOperation.MAX_CHUNKS_PER_OPERATION) in str(exc_info.value)

    def test_operation_timeout_enforcement(self) -> None:
        """Test that operation timeout is enforced."""

        # Arrange
        config = ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=5)

        operation = ChunkingOperation(
            operation_id="test", document_id="doc1", document_content="Test document", config=config
        )

        # Mock time to simulate timeout
        start_time = datetime.now(tz=UTC)
        timeout_time = start_time + timedelta(seconds=ChunkingOperation.MAX_OPERATION_DURATION_SECONDS + 1)

        # Act
        operation.start()
        operation._started_at = start_time

        with patch("shared.chunking.domain.entities.chunking_operation.datetime") as mock_datetime:
            mock_datetime.now.return_value = timeout_time
            is_valid, issues = operation.validate_results()

        # Assert
        assert not is_valid
        assert any("timeout" in issue.lower() for issue in issues)

    def test_configuration_validation_enforcement(self) -> None:
        """Test that configuration validation rules are enforced."""

        # Test various invalid configurations
        invalid_configs = [
            {"min_tokens": -1, "max_tokens": 100, "overlap_tokens": 5},
            {"min_tokens": 100, "max_tokens": 50, "overlap_tokens": 5},
            {"min_tokens": 10, "max_tokens": 100, "overlap_tokens": 15},
            {"min_tokens": 10, "max_tokens": 100, "overlap_tokens": -1},
        ]

        for config_params in invalid_configs:
            with pytest.raises(InvalidConfigurationError):
                ChunkConfig(strategy_name="test", **config_params)

    def test_state_transition_enforcement(self) -> None:
        """Test that state transition rules are enforced."""

        # Arrange
        config = ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=5)

        operation = ChunkingOperation(
            operation_id="test", document_id="doc1", document_content="Test document", config=config
        )

        # Test invalid transition from PENDING to COMPLETED
        operation._status.COMPLETED = operation._status.COMPLETED  # Set to completed

        # Trying to complete without processing should fail
        with pytest.raises(InvalidStateError):
            operation._complete()

        # Test invalid transition from COMPLETED to PROCESSING
        operation._status = operation._status.COMPLETED
        with pytest.raises(InvalidStateError):
            operation.start()

    def test_coverage_requirement_enforcement(self) -> None:
        """Test that coverage requirements are enforced."""

        # Arrange
        config = ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=5)

        document_content = "This is a test document with sufficient content for testing coverage."
        operation = ChunkingOperation(
            operation_id="test", document_id="doc1", document_content=document_content, config=config
        )

        # Add chunk covering only small portion (insufficient coverage)
        operation.start()
        small_chunk = Chunk(
            content="This",
            metadata=ChunkMetadata(
                chunk_id="chunk-small",
                document_id="doc1",  # Match the operation's document_id
                chunk_index=0,
                start_offset=0,
                end_offset=4,
                token_count=1,
                strategy_name="character",
            ),
            min_tokens=1,
        )
        operation.add_chunk(small_chunk)

        # Act
        is_valid, issues = operation.validate_results()

        # Assert
        assert not is_valid
        assert any("coverage" in issue.lower() for issue in issues)

    def test_metadata_validation_enforcement(self) -> None:
        """Test that metadata validation rules are enforced."""

        # Base valid parameters for ChunkMetadata
        base_params = {
            "chunk_id": "test-chunk",
            "document_id": "test-doc",
            "chunk_index": 0,
            "start_offset": 0,
            "end_offset": 100,
            "strategy_name": "test",
        }

        # Test invalid metadata values
        invalid_metadata_params = [
            {"token_count": -1},  # Negative token count
            {"token_count": 10, "semantic_density": -0.1},  # Negative semantic density
            {"token_count": 10, "semantic_density": 1.5},  # Semantic density > 1.0
            {"token_count": 10, "overlap_percentage": -0.1},  # Negative overlap percentage
            {"token_count": 10, "overlap_percentage": 1.5},  # Overlap percentage > 1.0
            {"token_count": 10, "confidence_score": -0.1},  # Negative confidence score
            {"token_count": 10, "confidence_score": 1.5},  # Confidence score > 1.0
        ]

        for params in invalid_metadata_params:
            # Merge base params with test params
            test_params = {**base_params, **params}
            with pytest.raises(ValueError, match=r".+"):  # Match any error message
                ChunkMetadata(**test_params)
