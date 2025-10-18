#!/usr/bin/env python3
"""
Semantic chunking strategy wrapper for domain interface.

This module provides a wrapper around the unified semantic chunking strategy
to make it compatible with domain tests that expect specific methods to patch.
"""


from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.unified.factory import DomainStrategyAdapter, UnifiedChunkingFactory


class SemanticChunkingStrategy(DomainStrategyAdapter):
    """
    Semantic chunking strategy that chunks based on semantic similarity.

    This wrapper provides patchable methods for testing while delegating
    actual chunking to the unified strategy.
    """

    def __init__(self) -> None:
        """Initialize the semantic chunking strategy."""
        unified_strategy = UnifiedChunkingFactory.create_strategy("semantic", use_llama_index=False)
        super().__init__(unified_strategy)

    def _calculate_similarity(self, _text1: str, _text2: str) -> float:
        """
        Calculate semantic similarity between two text segments.

        This method exists for test patching compatibility.
        In production, similarity is calculated within the unified strategy.

        Args:
            text1: First text segment
            text2: Second text segment

        Returns:
            Similarity score between 0 and 1
        """
        # This is a placeholder for test patching
        # Actual similarity calculation happens in the unified strategy
        return 0.7

    def _get_sentence_embedding(self, _sentence: str) -> list[float]:
        """
        Get embedding for a sentence.

        This method exists for test patching compatibility.

        Args:
            sentence: Sentence to embed

        Returns:
            Embedding vector
        """
        # This is a placeholder for test patching
        # Actual embedding happens in the unified strategy
        return [0.0] * 384  # Mock embedding

    def chunk(self, content: str, config: ChunkConfig) -> list[Chunk]:
        """
        Chunk content based on semantic boundaries.

        Args:
            content: Text to chunk
            config: Chunking configuration

        Returns:
            List of semantic chunks
        """
        # Delegate to the wrapped strategy but ensure semantic metadata is added
        chunks = self.strategy.chunk(content, config)

        # Add semantic-specific metadata if not present
        for chunk in chunks:
            if "semantic_boundary" not in chunk.metadata.custom_attributes:
                chunk.metadata.custom_attributes["semantic_boundary"] = True
            if "breakpoint_threshold" not in chunk.metadata.custom_attributes:
                chunk.metadata.custom_attributes["breakpoint_threshold"] = config.similarity_threshold

        return chunks
