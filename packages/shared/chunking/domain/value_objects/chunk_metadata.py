#!/usr/bin/env python3
"""
Chunk metadata value object.

This module defines the immutable metadata associated with each chunk,
including position, size, and semantic information.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ChunkMetadata:
    """
    Immutable metadata for a text chunk.

    This value object captures all the properties and context of a chunk
    within the source document.
    """

    chunk_id: str
    document_id: str
    chunk_index: int  # 0-based index within document
    start_offset: int  # Character position in original text
    end_offset: int  # Character position in original text
    token_count: int
    strategy_name: str

    # Optional semantic information
    semantic_score: float | None = None  # Similarity to adjacent chunks
    semantic_density: float = 0.5  # Density of semantic information (0.0-1.0)
    confidence_score: float = 1.0  # Confidence in chunk quality (0.0-1.0)
    overlap_percentage: float = 0.0  # Percentage of overlap with adjacent chunks (0.0-1.0)
    hierarchy_level: int | None = None  # For hierarchical chunking
    section_title: str | None = None  # For document structure chunking

    # Strategy-specific metadata
    custom_attributes: dict[str, Any] = field(default_factory=dict)  # For strategy-specific data

    # Temporal information
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        # Validate offsets
        if self.start_offset < 0:
            raise ValueError(f"Start offset must be non-negative, got {self.start_offset}")

        if self.end_offset <= self.start_offset:
            raise ValueError(f"End offset ({self.end_offset}) must be greater than start offset ({self.start_offset})")

        # Validate chunk index
        if self.chunk_index < 0:
            raise ValueError(f"Chunk index must be non-negative, got {self.chunk_index}")

        # Validate token count
        if self.token_count <= 0:
            raise ValueError(f"Token count must be positive, got {self.token_count}")

        # Validate semantic score if provided
        if self.semantic_score is not None:
            if not 0.0 <= self.semantic_score <= 1.0:
                raise ValueError(f"Semantic score must be between 0.0 and 1.0, got {self.semantic_score}")

        # Validate semantic density
        if not 0.0 <= self.semantic_density <= 1.0:
            raise ValueError(f"Semantic density must be between 0.0 and 1.0, got {self.semantic_density}")

        # Validate confidence score
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}")

        # Validate overlap percentage
        if not 0.0 <= self.overlap_percentage <= 1.0:
            raise ValueError(f"Overlap percentage must be between 0.0 and 1.0, got {self.overlap_percentage}")

        # Validate hierarchy level if provided
        if self.hierarchy_level is not None:
            if self.hierarchy_level < 0:
                raise ValueError(f"Hierarchy level must be non-negative, got {self.hierarchy_level}")

    @property
    def character_count(self) -> int:
        """Calculate the number of characters in the chunk."""
        return self.end_offset - self.start_offset

    @property
    def average_token_length(self) -> float:
        """Calculate the average characters per token."""
        if self.token_count == 0:
            return 0.0
        return self.character_count / self.token_count

    def overlaps_with(self, other: "ChunkMetadata") -> bool:
        """
        Check if this chunk overlaps with another chunk.

        Args:
            other: Another chunk metadata to compare

        Returns:
            True if chunks overlap in the original document
        """
        if self.document_id != other.document_id:
            return False

        return not (self.end_offset <= other.start_offset or other.end_offset <= self.start_offset)

    def overlap_size(self, other: "ChunkMetadata") -> int:
        """
        Calculate the size of overlap with another chunk.

        Args:
            other: Another chunk metadata to compare

        Returns:
            Number of overlapping characters, or 0 if no overlap
        """
        if not self.overlaps_with(other):
            return 0

        overlap_start = max(self.start_offset, other.start_offset)
        overlap_end = min(self.end_offset, other.end_offset)

        return overlap_end - overlap_start
