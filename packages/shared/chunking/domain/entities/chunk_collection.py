#!/usr/bin/env python3
"""
ChunkCollection entity representing a collection of chunks from a document.

This module defines the collection entity that manages multiple chunks and
enforces collection-level business rules.
"""

from collections.abc import Iterator

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.exceptions import InvalidChunkError


class ChunkCollection:
    """
    Entity representing a collection of chunks from a single document.

    This entity manages the relationship between chunks and enforces
    collection-level invariants such as ordering and coverage.
    """

    def __init__(self, document_id: str, source_text: str) -> None:
        """
        Initialize a chunk collection for a document.

        Args:
            document_id: Unique identifier for the source document
            source_text: The original text that was chunked
        """
        self._document_id = document_id
        self._source_text = source_text
        self._chunks: list[Chunk] = []
        self._chunk_index: dict[str, Chunk] = {}

    @property
    def document_id(self) -> str:
        """Get the document ID."""
        return self._document_id

    @property
    def source_text(self) -> str:
        """Get the source text."""
        return self._source_text

    @property
    def chunk_count(self) -> int:
        """Get the number of chunks in the collection."""
        return len(self._chunks)

    def add_chunk(self, chunk: Chunk) -> None:
        """
        Add a chunk to the collection.

        Args:
            chunk: The chunk to add

        Raises:
            InvalidChunkError: If chunk doesn't belong to this document or is duplicate
        """
        # Validate chunk belongs to this document
        if chunk.metadata.document_id != self._document_id:
            raise InvalidChunkError(
                f"Chunk {chunk.metadata.chunk_id} belongs to document "
                f"{chunk.metadata.document_id}, not {self._document_id}"
            )

        # Check for duplicates
        if chunk.metadata.chunk_id in self._chunk_index:
            raise InvalidChunkError(f"Chunk {chunk.metadata.chunk_id} already exists in collection")

        # Validate chunk boundaries
        if chunk.metadata.end_offset > len(self._source_text):
            raise InvalidChunkError(
                f"Chunk {chunk.metadata.chunk_id} end offset ({chunk.metadata.end_offset}) "
                f"exceeds source text length ({len(self._source_text)})"
            )

        self._chunks.append(chunk)
        self._chunk_index[chunk.metadata.chunk_id] = chunk

        # Keep chunks sorted by index
        self._chunks.sort(key=lambda c: c.metadata.chunk_index)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """
        Get a chunk by its ID.

        Args:
            chunk_id: The chunk ID to retrieve

        Returns:
            The chunk if found, None otherwise
        """
        return self._chunk_index.get(chunk_id)

    def get_chunk_by_index(self, index: int) -> Chunk | None:
        """
        Get a chunk by its index.

        Args:
            index: The chunk index (0-based)

        Returns:
            The chunk if found, None otherwise
        """
        for chunk in self._chunks:
            if chunk.metadata.chunk_index == index:
                return chunk
        return None

    def get_chunks(self) -> list[Chunk]:
        """Get all chunks in order."""
        return self._chunks.copy()

    def __iter__(self) -> Iterator[Chunk]:
        """Iterate over chunks in order."""
        return iter(self._chunks)

    def calculate_coverage(self) -> float:
        """
        Calculate text coverage percentage.

        Returns:
            Percentage of source text covered by chunks (0.0 to 1.0)
        """
        if not self._source_text:
            return 0.0

        # Track covered character positions
        covered_positions = set()

        for chunk in self._chunks:
            for pos in range(chunk.metadata.start_offset, chunk.metadata.end_offset):
                covered_positions.add(pos)

        total_chars = len(self._source_text)
        covered_chars = len(covered_positions)

        return covered_chars / total_chars if total_chars > 0 else 0.0

    def calculate_overlap_statistics(self) -> dict[str, float]:
        """
        Calculate statistics about chunk overlaps.

        Returns:
            Dictionary with overlap statistics
        """
        if len(self._chunks) < 2:
            return {
                "average_overlap": 0.0,
                "max_overlap": 0.0,
                "min_overlap": 0.0,
                "total_overlap_chars": 0,
            }

        overlaps = []
        total_overlap_chars = 0

        # Check consecutive chunks for overlap
        for i in range(len(self._chunks) - 1):
            current = self._chunks[i]
            next_chunk = self._chunks[i + 1]

            overlap_size = current.metadata.overlap_size(next_chunk.metadata)
            if overlap_size > 0:
                overlaps.append(overlap_size)
                total_overlap_chars += overlap_size

        if not overlaps:
            return {
                "average_overlap": 0.0,
                "max_overlap": 0.0,
                "min_overlap": 0.0,
                "total_overlap_chars": 0,
            }

        return {
            "average_overlap": sum(overlaps) / len(overlaps),
            "max_overlap": max(overlaps),
            "min_overlap": min(overlaps),
            "total_overlap_chars": total_overlap_chars,
        }

    def calculate_size_statistics(self) -> dict[str, float]:
        """
        Calculate statistics about chunk sizes.

        Returns:
            Dictionary with size statistics
        """
        if not self._chunks:
            return {
                "average_tokens": 0.0,
                "max_tokens": 0,
                "min_tokens": 0,
                "total_tokens": 0,
                "average_chars": 0.0,
                "max_chars": 0,
                "min_chars": 0,
            }

        token_counts = [c.metadata.token_count for c in self._chunks]
        char_counts = [c.metadata.character_count for c in self._chunks]

        return {
            "average_tokens": sum(token_counts) / len(token_counts),
            "max_tokens": max(token_counts),
            "min_tokens": min(token_counts),
            "total_tokens": sum(token_counts),
            "average_chars": sum(char_counts) / len(char_counts),
            "max_chars": max(char_counts),
            "min_chars": min(char_counts),
        }

    def find_gaps(self) -> list[tuple[int, int]]:
        """
        Find gaps in text coverage.

        Returns:
            List of (start, end) tuples representing uncovered regions
        """
        if not self._chunks:
            return [(0, len(self._source_text))] if self._source_text else []

        gaps = []
        last_end = 0

        for chunk in self._chunks:
            if chunk.metadata.start_offset > last_end:
                gaps.append((last_end, chunk.metadata.start_offset))
            last_end = max(last_end, chunk.metadata.end_offset)

        # Check for gap at the end
        if last_end < len(self._source_text):
            gaps.append((last_end, len(self._source_text)))

        return gaps

    def validate_completeness(self) -> tuple[bool, list[str]]:
        """
        Validate the collection for completeness and consistency.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check for gaps
        gaps = self.find_gaps()
        if gaps:
            total_gap_chars = sum(end - start for start, end in gaps)
            issues.append(f"Found {len(gaps)} gaps totaling {total_gap_chars} characters")

        # Check for missing indices
        expected_indices = set(range(len(self._chunks)))
        actual_indices = {c.metadata.chunk_index for c in self._chunks}
        missing_indices = expected_indices - actual_indices

        if missing_indices:
            issues.append(f"Missing chunk indices: {sorted(missing_indices)}")

        # Check coverage
        coverage = self.calculate_coverage()
        if coverage < 0.95:  # Allow 5% uncovered for edge cases
            issues.append(f"Low coverage: {coverage:.1%}")

        return len(issues) == 0, issues

    def get_chunks_in_range(self, start_offset: int, end_offset: int) -> list[Chunk]:
        """
        Get all chunks that overlap with the given character range.

        Args:
            start_offset: Start position in source text
            end_offset: End position in source text

        Returns:
            List of chunks that overlap the range
        """
        result = []

        for chunk in self._chunks:
            if chunk.metadata.start_offset < end_offset and chunk.metadata.end_offset > start_offset:
                result.append(chunk)

        return result

    def merge_adjacent_chunks(self, max_tokens: int) -> list[Chunk]:
        """
        Merge adjacent small chunks up to max token limit.

        Args:
            max_tokens: Maximum tokens for merged chunks

        Returns:
            List of merged chunks
        """
        if not self._chunks:
            return []

        merged = []
        current_group = [self._chunks[0]]
        current_tokens = self._chunks[0].metadata.token_count

        for chunk in self._chunks[1:]:
            # Check if we can merge with current group
            if current_tokens + chunk.metadata.token_count <= max_tokens:
                current_group.append(chunk)
                current_tokens += chunk.metadata.token_count
            else:
                # Create merged chunk from current group
                if current_group:
                    merged.append(self._create_merged_chunk(current_group))

                # Start new group
                current_group = [chunk]
                current_tokens = chunk.metadata.token_count

        # Don't forget the last group
        if current_group:
            merged.append(self._create_merged_chunk(current_group))

        return merged

    def _create_merged_chunk(self, chunks: list[Chunk]) -> Chunk:
        """Create a single chunk from multiple chunks."""
        # This is a simplified merge - in practice, you'd want more sophisticated logic
        from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata

        first = chunks[0]
        last = chunks[-1]

        merged_content = " ".join(c.content for c in chunks)
        total_tokens = sum(c.metadata.token_count for c in chunks)

        metadata = ChunkMetadata(
            chunk_id=f"{first.metadata.chunk_id}_merged",
            document_id=self._document_id,
            chunk_index=first.metadata.chunk_index,
            start_offset=first.metadata.start_offset,
            end_offset=last.metadata.end_offset,
            token_count=total_tokens,
            strategy_name="merged",
        )

        return Chunk(merged_content, metadata)

    def __repr__(self) -> str:
        """String representation of the collection."""
        return (
            f"ChunkCollection(document_id={self._document_id}, "
            f"chunks={self.chunk_count}, "
            f"coverage={self.calculate_coverage():.1%})"
        )

    def __len__(self) -> int:
        """Get the number of chunks."""
        return len(self._chunks)
