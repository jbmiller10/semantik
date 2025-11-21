#!/usr/bin/env python3
"""
Hierarchical chunking strategy wrapper for domain interface.

This module provides a wrapper around the unified hierarchical chunking strategy
to make it compatible with domain tests that expect specific methods to patch.
"""

from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.unified.factory import DomainStrategyAdapter, UnifiedChunkingFactory


class HierarchicalChunkingStrategy(DomainStrategyAdapter):
    """
    Hierarchical chunking strategy that creates multi-level chunks.

    This wrapper provides patchable methods for testing while delegating
    actual chunking to the unified strategy.
    """

    def __init__(self) -> None:
        """Initialize the hierarchical chunking strategy."""
        unified_strategy = UnifiedChunkingFactory.create_strategy("hierarchical", use_llama_index=False)
        super().__init__(unified_strategy)

    def _generate_summary(self, _text: str) -> str:
        """
        Generate a summary for a chunk of text.

        This method exists for test patching compatibility.

        Args:
            text: Text to summarize

        Returns:
            Summary of the text
        """
        # This is a placeholder for test patching
        # Actual summary generation could be implemented here or in unified strategy
        return "Summary of content"

    def chunk(self, content: str, config: ChunkConfig) -> list[Chunk]:
        """
        Create hierarchical chunks with parent-child relationships.

        Args:
            content: Text to chunk
            config: Chunking configuration

        Returns:
            List of hierarchical chunks
        """
        # Delegate to the wrapped strategy
        chunks = self.strategy.chunk(content, config)

        # Fix metadata placement - move hierarchy-related fields to custom_attributes
        for chunk in chunks:
            # Move hierarchy_level to custom_attributes if it exists
            if hasattr(chunk.metadata, "hierarchy_level") and chunk.metadata.hierarchy_level is not None:
                chunk.metadata.custom_attributes["hierarchy_level"] = chunk.metadata.hierarchy_level

            # Add parent_id if it's a child chunk (map from parent_chunk_id)
            if chunk.metadata.custom_attributes.get("parent_chunk_id"):
                chunk.metadata.custom_attributes["parent_id"] = chunk.metadata.custom_attributes["parent_chunk_id"]

            # Add chunk_id for parent-child linking
            chunk.metadata.custom_attributes["chunk_id"] = chunk.metadata.chunk_id

            # Add child_chunk_ids if it's a parent chunk
            if hasattr(chunk.metadata, "child_chunk_ids") and chunk.metadata.child_chunk_ids:
                chunk.metadata.custom_attributes["child_chunk_ids"] = chunk.metadata.child_chunk_ids

            # Generate summary for parent chunks (level 0)
            if chunk.metadata.custom_attributes.get("hierarchy_level") == 0:
                chunk.metadata.custom_attributes["summary"] = self._generate_summary(chunk.content)

        return chunks
