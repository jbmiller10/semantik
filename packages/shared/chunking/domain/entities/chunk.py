#!/usr/bin/env python3
"""
Chunk entity representing a single text chunk.

This module defines the core chunk entity with its business logic and invariants.
"""


from packages.shared.chunking.domain.exceptions import (
    ChunkSizeViolationError,
    InvalidChunkError,
)
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata


class Chunk:
    """
    Entity representing a single text chunk.

    A chunk is a segment of text extracted from a document according to a
    specific chunking strategy. It maintains its own invariants and business rules.
    """

    def __init__(
        self,
        content: str,
        metadata: ChunkMetadata,
        min_tokens: int = 10,
        max_tokens: int = 10000,
    ) -> None:
        """
        Initialize a chunk with content and metadata.

        Args:
            content: The text content of the chunk
            metadata: Immutable metadata for the chunk
            min_tokens: Minimum allowed tokens
            max_tokens: Maximum allowed tokens

        Raises:
            InvalidChunkError: If chunk violates business rules
        """
        self._validate_content(content)
        self._validate_size_constraints(metadata.token_count, min_tokens, max_tokens)

        self._content = content
        self._metadata = metadata
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens

        # Mutable properties
        self._embedding: list[float] | None = None
        self._quality_score: float | None = None
        self._processing_time_ms: float | None = None

    @property
    def content(self) -> str:
        """Get the text content of the chunk."""
        return self._content

    @property
    def metadata(self) -> ChunkMetadata:
        """Get the immutable metadata of the chunk."""
        return self._metadata

    @property
    def embedding(self) -> list[float] | None:
        """Get the embedding vector if computed."""
        return self._embedding

    @property
    def quality_score(self) -> float | None:
        """Get the quality score if evaluated."""
        return self._quality_score

    @property
    def processing_time_ms(self) -> float | None:
        """Get the processing time in milliseconds."""
        return self._processing_time_ms

    def set_embedding(self, embedding: list[float]) -> None:
        """
        Set the embedding vector for the chunk.

        Args:
            embedding: The embedding vector

        Raises:
            InvalidChunkError: If embedding is invalid
        """
        if not embedding:
            raise InvalidChunkError("Embedding cannot be empty")

        if not all(isinstance(x, (int, float)) for x in embedding):
            raise InvalidChunkError("Embedding must contain only numeric values")

        self._embedding = embedding

    def set_quality_score(self, score: float) -> None:
        """
        Set the quality score for the chunk.

        Args:
            score: Quality score between 0.0 and 1.0

        Raises:
            InvalidChunkError: If score is out of range
        """
        if not 0.0 <= score <= 1.0:
            raise InvalidChunkError(
                f"Quality score must be between 0.0 and 1.0, got {score}"
            )
        self._quality_score = score

    def set_processing_time(self, time_ms: float) -> None:
        """
        Set the processing time for the chunk.

        Args:
            time_ms: Processing time in milliseconds

        Raises:
            InvalidChunkError: If time is negative
        """
        if time_ms < 0:
            raise InvalidChunkError(f"Processing time cannot be negative, got {time_ms}")
        self._processing_time_ms = time_ms

    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """
        Check if the chunk meets quality standards.

        Args:
            threshold: Quality threshold (default 0.7)

        Returns:
            True if quality score meets or exceeds threshold
        """
        if self._quality_score is None:
            return False
        return self._quality_score >= threshold

    def has_embedding(self) -> bool:
        """Check if the chunk has an embedding."""
        return self._embedding is not None

    def calculate_token_density(self) -> float:
        """
        Calculate the token density (tokens per character).

        Returns:
            Token density ratio
        """
        char_count = len(self._content)
        if char_count == 0:
            return 0.0
        return self._metadata.token_count / char_count

    def estimate_memory_usage(self) -> int:
        """
        Estimate memory usage of the chunk in bytes.

        Returns:
            Estimated memory usage
        """
        # Text content (approximate UTF-8 encoding)
        text_size = len(self._content.encode('utf-8'))

        # Embedding (if present) - 4 bytes per float
        embedding_size = len(self._embedding) * 4 if self._embedding else 0

        # Metadata overhead (rough estimate)
        metadata_size = 500  # Approximate size of metadata object

        return text_size + embedding_size + metadata_size

    def _validate_content(self, content: str) -> None:
        """
        Validate chunk content meets business requirements.

        Args:
            content: Content to validate

        Raises:
            InvalidChunkError: If content is invalid
        """
        if not content:
            raise InvalidChunkError("Chunk content cannot be empty")

        if not content.strip():
            raise InvalidChunkError("Chunk content cannot be only whitespace")

        # Check for reasonable content size (prevent memory issues)
        if len(content) > 1_000_000:  # 1MB limit per chunk
            raise InvalidChunkError(
                f"Chunk content exceeds 1MB limit: {len(content)} characters"
            )

    def _validate_size_constraints(
        self, token_count: int, min_tokens: int, max_tokens: int
    ) -> None:
        """
        Validate chunk size meets constraints.

        Args:
            token_count: Number of tokens in chunk
            min_tokens: Minimum allowed tokens
            max_tokens: Maximum allowed tokens

        Raises:
            ChunkSizeViolationError: If size constraints are violated
        """
        # Allow chunks smaller than min_tokens if they are likely the last chunk
        # or part of a small document (relaxed validation for edge cases)
        if token_count > max_tokens:
            raise ChunkSizeViolationError(token_count, min_tokens, max_tokens)
        
        # Only enforce minimum for chunks that are clearly too small
        # (less than 50% of minimum and not a trivially small chunk)
        if token_count < min_tokens * 0.5 and token_count > 0 and min_tokens > 5:
            raise ChunkSizeViolationError(token_count, min_tokens, max_tokens)

    def __repr__(self) -> str:
        """String representation of the chunk."""
        preview = self._content[:50] + "..." if len(self._content) > 50 else self._content
        return (
            f"Chunk(id={self._metadata.chunk_id}, "
            f"index={self._metadata.chunk_index}, "
            f"tokens={self._metadata.token_count}, "
            f"content='{preview}')"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality based on chunk ID."""
        if not isinstance(other, Chunk):
            return False
        return self._metadata.chunk_id == other._metadata.chunk_id

    def __hash__(self) -> int:
        """Hash based on chunk ID."""
        return hash(self._metadata.chunk_id)
