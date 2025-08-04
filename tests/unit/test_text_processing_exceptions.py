#!/usr/bin/env python3
"""Unit tests for text processing exception hierarchy.

This module tests the new exception hierarchy introduced in
packages/shared/text_processing/exceptions.py to ensure:
1. Proper inheritance relationships
2. Exception instantiation and string representation
3. Exception catching and handling patterns
"""

import pytest

from packages.shared.text_processing.exceptions import (
    # Factory errors
    ChunkerCreationError,
    ChunkingError,
    # Specific chunking errors
    ChunkSizeError,
    # Specific validation errors
    ConfigValidationError,
    DimensionMismatchError,
    EmbeddingError,
    EmbeddingServiceNotInitializedError,
    HierarchyDepthError,
    PermanentEmbeddingError,
    RegexTimeoutError,
    TextLengthError,
    # Base exceptions
    TextProcessingError,
    # Specific embedding errors
    TransientEmbeddingError,
    UnknownStrategyError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Test the exception hierarchy structure and inheritance."""

    def test_base_exception_hierarchy(self):
        """Test that all exceptions inherit from TextProcessingError."""
        # Base level exceptions
        assert issubclass(ChunkingError, TextProcessingError)
        assert issubclass(EmbeddingError, TextProcessingError)
        assert issubclass(ValidationError, TextProcessingError)

        # All exceptions should ultimately inherit from TextProcessingError
        all_exceptions = [
            ChunkSizeError,
            HierarchyDepthError,
            TextLengthError,
            TransientEmbeddingError,
            PermanentEmbeddingError,
            DimensionMismatchError,
            EmbeddingServiceNotInitializedError,
            ConfigValidationError,
            RegexTimeoutError,
            ChunkerCreationError,
            UnknownStrategyError,
        ]

        for exc_class in all_exceptions:
            assert issubclass(exc_class, TextProcessingError)

    def test_chunking_error_hierarchy(self):
        """Test chunking error inheritance."""
        assert issubclass(ChunkSizeError, ChunkingError)
        assert issubclass(HierarchyDepthError, ChunkingError)
        assert issubclass(TextLengthError, ChunkingError)
        assert issubclass(ChunkerCreationError, ChunkingError)
        assert issubclass(UnknownStrategyError, ChunkerCreationError)

    def test_embedding_error_hierarchy(self):
        """Test embedding error inheritance."""
        assert issubclass(TransientEmbeddingError, EmbeddingError)
        assert issubclass(PermanentEmbeddingError, EmbeddingError)
        assert issubclass(DimensionMismatchError, EmbeddingError)
        assert issubclass(EmbeddingServiceNotInitializedError, EmbeddingError)

    def test_validation_error_hierarchy(self):
        """Test validation error inheritance."""
        assert issubclass(ConfigValidationError, ValidationError)
        assert issubclass(RegexTimeoutError, ValidationError)

    def test_exception_instantiation(self):
        """Test that all exceptions can be instantiated with a message."""
        exceptions = [
            TextProcessingError("Base text processing error"),
            ChunkingError("Base chunking error"),
            EmbeddingError("Base embedding error"),
            ValidationError("Base validation error"),
            ChunkSizeError("Chunk too large"),
            HierarchyDepthError("Hierarchy too deep"),
            TextLengthError("Text too long"),
            TransientEmbeddingError("Temporary failure"),
            PermanentEmbeddingError("Model not found"),
            DimensionMismatchError("Expected 384, got 512"),
            EmbeddingServiceNotInitializedError("Service not started"),
            ConfigValidationError("Invalid config"),
            RegexTimeoutError("Regex took too long"),
            ChunkerCreationError("Failed to create chunker"),
            UnknownStrategyError("Unknown strategy: foo"),
        ]

        for exc in exceptions:
            assert isinstance(exc, Exception)
            assert isinstance(exc, TextProcessingError)
            assert str(exc) == exc.args[0]

    def test_exception_catching_patterns(self):
        """Test various exception catching patterns."""
        # Test catching specific exceptions
        with pytest.raises(ChunkSizeError):
            raise ChunkSizeError("Chunk exceeds 1000 tokens")

        # Test catching by category
        with pytest.raises(ChunkingError):
            raise HierarchyDepthError("Max depth exceeded")

        with pytest.raises(EmbeddingError):
            raise DimensionMismatchError("Wrong dimensions")

        # Test catching all text processing errors
        with pytest.raises(TextProcessingError):
            raise RegexTimeoutError("Regex timeout")

    def test_transient_vs_permanent_embedding_errors(self):
        """Test distinction between transient and permanent embedding errors."""
        # Transient errors (retryable)
        transient_errors = [
            TransientEmbeddingError("GPU OOM"),
            TransientEmbeddingError("API rate limit"),
            TransientEmbeddingError("Temporary network error"),
        ]

        # Permanent errors (not retryable)
        permanent_errors = [
            PermanentEmbeddingError("Invalid model name"),
            PermanentEmbeddingError("Corrupted model file"),
            PermanentEmbeddingError("Unsupported input format"),
        ]

        for err in transient_errors:
            assert isinstance(err, TransientEmbeddingError)
            assert isinstance(err, EmbeddingError)
            assert not isinstance(err, PermanentEmbeddingError)

        for err in permanent_errors:
            assert isinstance(err, PermanentEmbeddingError)
            assert isinstance(err, EmbeddingError)
            assert not isinstance(err, TransientEmbeddingError)

    def test_factory_error_hierarchy(self):
        """Test factory-related error hierarchy."""
        # UnknownStrategyError is a specific type of ChunkerCreationError
        with pytest.raises(ChunkerCreationError):
            raise UnknownStrategyError("Strategy 'quantum' not found")

        # Both are ChunkingErrors
        with pytest.raises(ChunkingError):
            raise ChunkerCreationError("Factory initialization failed")

    def test_error_context_preservation(self):
        """Test that exceptions preserve context when chained."""
        # Set up the low-level error outside the pytest.raises block
        try:
            # Simulate a low-level error
            raise ValueError("Invalid dimension: -1")
        except ValueError as e:
            # Store the exception for later use
            original_error = e

        # Only the statement that triggers the exception should be in pytest.raises
        with pytest.raises(DimensionMismatchError) as exc_info:
            raise DimensionMismatchError("Dimension must be positive") from original_error

        assert str(exc_info.value) == "Dimension must be positive"
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "Invalid dimension: -1"

    def test_exception_type_checking(self):
        """Test type checking for exception handling decisions."""

        def handle_embedding_error(error: EmbeddingError) -> str:
            """Determine how to handle an embedding error."""
            if isinstance(error, TransientEmbeddingError):
                return "retry"
            if isinstance(error, PermanentEmbeddingError):
                return "fail"
            if isinstance(error, EmbeddingServiceNotInitializedError):
                return "initialize"
            if isinstance(error, DimensionMismatchError):
                return "resize"
            return "unknown"

        assert handle_embedding_error(TransientEmbeddingError("OOM")) == "retry"
        assert handle_embedding_error(PermanentEmbeddingError("Bad model")) == "fail"
        assert handle_embedding_error(EmbeddingServiceNotInitializedError()) == "initialize"
        assert handle_embedding_error(DimensionMismatchError("384 != 512")) == "resize"

    def test_exception_messages(self):
        """Test exception message formatting."""
        # Test with single argument
        exc1 = ChunkSizeError("Chunk size 1500 exceeds maximum 1000")
        assert str(exc1) == "Chunk size 1500 exceeds maximum 1000"

        # Test with multiple arguments (creates tuple)
        exc2 = TextLengthError("Text length:", 50000, "exceeds max:", 10000)
        assert len(exc2.args) == 4

        # Test empty message
        exc3 = ValidationError()
        assert str(exc3) == ""

    def test_regex_timeout_error_specifics(self):
        """Test RegexTimeoutError for ReDoS protection."""
        pattern = r"(a+)+"
        text = "a" * 100
        timeout = 1.0

        error = RegexTimeoutError(f"Regex pattern '{pattern}' timed out after {timeout}s on text of length {len(text)}")

        assert isinstance(error, ValidationError)
        assert isinstance(error, TextProcessingError)
        assert "timed out" in str(error)
        assert str(timeout) in str(error)

    def test_security_related_exceptions(self):
        """Test exceptions related to security validations."""
        # Text length validation
        max_length = 5_000_000
        actual_length = 10_000_000

        error = TextLengthError(f"Text length {actual_length:,} exceeds maximum {max_length:,}")
        assert "10,000,000" in str(error)
        assert "5,000,000" in str(error)

        # Hierarchy depth validation
        max_depth = 5
        error = HierarchyDepthError(f"Hierarchy depth {max_depth + 1} exceeds maximum {max_depth}")
        assert "6" in str(error)
        assert "5" in str(error)
