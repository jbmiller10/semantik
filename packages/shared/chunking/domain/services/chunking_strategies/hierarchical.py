#!/usr/bin/env python3
"""
Hierarchical chunking strategy.

This strategy creates multi-level chunks with parent-child relationships,
useful for maintaining context at different granularities.
"""

from collections.abc import Callable
from datetime import datetime

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.services.chunking_strategies.base import (
    ChunkingStrategy,
)
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata


class HierarchicalChunkingStrategy(ChunkingStrategy):
    """
    Hierarchical chunking strategy.

    Creates multi-level chunks where larger parent chunks contain
    references to smaller child chunks, maintaining context at
    different levels of detail.
    """

    def __init__(self) -> None:
        """Initialize the hierarchical chunking strategy."""
        super().__init__("hierarchical")

    def chunk(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Create hierarchical chunks at multiple levels.

        Args:
            content: The text content to chunk
            config: Configuration parameters
            progress_callback: Optional progress callback

        Returns:
            List of chunks at all hierarchy levels
        """
        if not content:
            return []

        # Create chunks at different hierarchy levels
        all_chunks = []

        # Calculate chunk sizes for each level
        levels = min(config.hierarchy_levels, 3)  # Max 3 levels for practicality
        level_configs = self._create_level_configs(config, levels)

        # Process each level
        total_operations = levels
        parent_chunks = []

        for level in range(levels):
            level_config = level_configs[level]

            # Create chunks for this level
            level_chunks = self._create_level_chunks(
                content,
                level,
                level_config,
                config.strategy_name,
                parent_chunks if level > 0 else None,
            )

            # Store chunks from level 0 as parent chunks for next level
            if level == 0:
                parent_chunks = level_chunks

            all_chunks.extend(level_chunks)

            # Report progress
            if progress_callback:
                progress = ((level + 1) / total_operations) * 100
                progress_callback(min(progress, 100.0))

        return all_chunks

    def _create_level_configs(self, base_config: ChunkConfig, levels: int) -> list[dict]:
        """
        Create configuration for each hierarchy level.

        Args:
            base_config: Base configuration
            levels: Number of hierarchy levels

        Returns:
            List of level configurations
        """
        configs = []

        for level in range(levels):
            # Each level has progressively smaller chunks
            scale_factor = 2**level

            level_max_tokens = max(
                base_config.min_tokens,
                base_config.max_tokens // scale_factor,
            )

            level_min_tokens = max(
                10,  # Minimum viable chunk size
                base_config.min_tokens // scale_factor,
            )

            # Ensure min_tokens is never greater than max_tokens
            if level_min_tokens > level_max_tokens:
                level_min_tokens = level_max_tokens

            level_overlap = max(
                0,
                base_config.overlap_tokens // scale_factor,
            )

            # Ensure overlap doesn't exceed min_tokens
            if level_overlap >= level_min_tokens:
                level_overlap = max(0, level_min_tokens - 1)

            configs.append(
                {
                    "max_tokens": level_max_tokens,
                    "min_tokens": level_min_tokens,
                    "overlap_tokens": level_overlap,
                }
            )

        return configs

    def _create_level_chunks(
        self,
        content: str,
        level: int,
        level_config: dict,
        base_strategy: str,
        parent_chunks: list[Chunk] | None = None,
    ) -> list[Chunk]:
        """
        Create chunks for a specific hierarchy level.

        Args:
            content: Content to chunk
            level: Hierarchy level (0 = top level)
            level_config: Configuration for this level
            base_strategy: Base strategy name

        Returns:
            List of chunks for this level
        """
        chunks = []

        # Calculate chunk boundaries
        chars_per_token = 4
        chunk_size_chars = level_config["max_tokens"] * chars_per_token
        overlap_chars = level_config["overlap_tokens"] * chars_per_token

        position = 0
        chunk_index = 0

        while position < len(content):
            # Determine chunk boundaries
            start = position
            end = min(position + chunk_size_chars, len(content))

            # Adjust to natural boundaries
            if end < len(content):
                # For higher levels, prefer paragraph boundaries
                if level == 0:
                    end = self._find_paragraph_boundary(content, end)
                # For lower levels, prefer sentence boundaries
                elif level == 1:
                    end = self.find_sentence_boundary(content, end, prefer_before=True)
                # For lowest level, prefer word boundaries
                else:
                    end = self.find_word_boundary(content, end, prefer_before=True)

            # Extract and clean chunk text
            chunk_text = content[start:end]
            chunk_text = self.clean_chunk_text(chunk_text)

            if not chunk_text:
                position = end
                continue

            # Calculate token count
            token_count = self.count_tokens(chunk_text)

            # Create metadata with hierarchy info in custom_attributes
            chunk_id = f"{base_strategy}_L{level}_{chunk_index:04d}"
            custom_attrs = {
                "hierarchy_level": level,
                "chunk_id": chunk_id,  # Store ID for parent references
            }

            # Add parent_id for child chunks
            if level > 0 and parent_chunks:
                # Find parent chunk that contains this position
                for parent in parent_chunks:
                    if parent.metadata.start_offset <= start <= parent.metadata.end_offset:
                        custom_attrs["parent_id"] = parent.metadata.custom_attributes.get(
                            "chunk_id", parent.metadata.chunk_id
                        )
                        break

            # Add summary for parent chunks
            if level == 0:
                custom_attrs["summary"] = self._generate_summary(chunk_text)

            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_id="doc",
                chunk_index=chunk_index,
                start_offset=start,
                end_offset=end,
                token_count=token_count,
                strategy_name=self.name,
                hierarchy_level=level,
                custom_attributes=custom_attrs,
                semantic_density=0.7,  # Higher for hierarchical organization
                confidence_score=0.85,
                created_at=datetime.utcnow(),
            )

            # Create chunk
            chunk = Chunk(
                content=chunk_text,
                metadata=metadata,
                min_tokens=level_config["min_tokens"],
                max_tokens=level_config["max_tokens"],
            )

            chunks.append(chunk)
            chunk_index += 1

            # Move position with overlap
            if end >= len(content):
                break

            position = end - overlap_chars

        return chunks

    def _find_paragraph_boundary(self, text: str, target_position: int) -> int:
        """
        Find nearest paragraph boundary.

        Args:
            text: Text to search in
            target_position: Target position

        Returns:
            Position of paragraph boundary
        """
        if target_position >= len(text):
            return len(text)

        # Look for double newline (paragraph break)
        # Search backwards first
        for i in range(target_position, max(0, target_position - 500), -1):
            if i > 0 and text[i - 1 : i + 1] == "\n\n":
                return i + 1

        # If not found, search forwards
        for i in range(target_position, min(len(text), target_position + 500)):
            if i > 0 and text[i - 1 : i + 1] == "\n\n":
                return i + 1

        # Fall back to sentence boundary
        return self.find_sentence_boundary(text, target_position, prefer_before=True)

    def _establish_relationships(
        self,
        child_chunks: list[Chunk],
        all_chunks: list[Chunk],
        parent_level: int,
    ) -> None:
        """
        Establish parent-child relationships between chunks.

        Args:
            child_chunks: Child chunks to link
            all_chunks: All existing chunks
            parent_level: Level of parent chunks
        """
        # Get parent chunks
        parent_chunks = [c for c in all_chunks if c.metadata.hierarchy_level == parent_level]

        # For each child, find overlapping parent
        for child in child_chunks:
            best_parent = None
            max_overlap = 0

            for parent in parent_chunks:
                # Calculate overlap
                overlap_start = max(
                    child.metadata.start_offset,
                    parent.metadata.start_offset,
                )
                overlap_end = min(
                    child.metadata.end_offset,
                    parent.metadata.end_offset,
                )

                if overlap_end > overlap_start:
                    overlap = overlap_end - overlap_start
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_parent = parent

            # Store relationship (in a real implementation, this would be
            # stored in a proper relationship structure)
            if best_parent:
                # We can't modify frozen metadata, but in a real system
                # relationships would be stored separately
                pass

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """
        Validate content for hierarchical chunking.

        Args:
            content: Content to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not content:
            return False, "Content cannot be empty"

        if len(content) > 50_000_000:  # 50MB limit
            return False, f"Content too large: {len(content)} characters"

        # Hierarchical chunking needs sufficient content
        if len(content) < 100:
            return False, "Content too short for hierarchical chunking"

        return True, None

    def estimate_chunks(self, content_length: int, config: ChunkConfig) -> int:
        """
        Estimate total chunks across all levels.

        Args:
            content_length: Length of content in characters
            config: Chunking configuration

        Returns:
            Estimated total chunk count
        """
        if content_length == 0:
            return 0

        estimated_tokens = content_length // 4
        total_estimate = 0

        # Calculate estimate for each level
        levels = min(config.hierarchy_levels, 3)

        for level in range(levels):
            scale_factor = 2**level
            level_max_tokens = max(
                config.min_tokens,
                config.max_tokens // scale_factor,
            )

            level_min_tokens = max(
                10,  # Minimum viable chunk size
                config.min_tokens // scale_factor,
            )

            # Ensure min_tokens is never greater than max_tokens
            if level_min_tokens > level_max_tokens:
                level_min_tokens = level_max_tokens

            level_overlap = max(
                0,
                config.overlap_tokens // scale_factor,
            )

            # Ensure overlap doesn't exceed min_tokens
            if level_overlap >= level_min_tokens:
                level_overlap = max(0, level_min_tokens - 1)

            # Create temporary config for estimation
            level_config = ChunkConfig(
                strategy_name=config.strategy_name,
                min_tokens=level_min_tokens,
                max_tokens=level_max_tokens,
                overlap_tokens=level_overlap,
            )

            level_estimate = level_config.estimate_chunks(estimated_tokens)
            total_estimate += level_estimate

        return total_estimate

    def _generate_summary(self, text: str) -> str:
        """
        Generate a summary for a text chunk.

        This is a simplified version that extracts key sentences.
        In a production system, this could use more sophisticated
        summarization techniques.

        Args:
            text: Text to summarize

        Returns:
            Summary of the text
        """
        sentences = text.split(". ")
        if len(sentences) <= 2:
            return text

        # Simple heuristic: take first and last sentence
        summary = f"{sentences[0]}. {sentences[-1]}"
        if not summary.endswith("."):
            summary += "."

        return summary
