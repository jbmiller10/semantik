#!/usr/bin/env python3
"""
Unified hybrid chunking strategy.

This module provides an intelligent chunking strategy that analyzes content
characteristics and selects the most appropriate unified chunking strategy.
"""

import asyncio
import logging
from collections.abc import Callable
from enum import Enum
from typing import Any, TypedDict

from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.unified.base import UnifiedChunkingStrategy
from shared.chunking.unified.character_strategy import CharacterChunkingStrategy
from shared.chunking.unified.hierarchical_strategy import HierarchicalChunkingStrategy
from shared.chunking.unified.markdown_strategy import MarkdownChunkingStrategy
from shared.chunking.unified.recursive_strategy import RecursiveChunkingStrategy
from shared.chunking.unified.semantic_strategy import SemanticChunkingStrategy
from shared.chunking.utils.safe_regex import SafeRegex

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Enumeration of content types."""

    MARKDOWN = "markdown"
    CODE = "code"
    STRUCTURED = "structured"
    NARRATIVE = "narrative"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ContentAnalysis(TypedDict):
    """Type definition for content analysis results."""

    total_chars: int
    total_lines: int
    has_markdown: bool
    has_code: bool
    has_structure: bool
    sentence_count: int
    avg_sentence_length: float
    is_mixed: bool
    content_type: ContentType
    recommended_strategy: str


class HybridChunkingStrategy(UnifiedChunkingStrategy):
    """
    Unified hybrid chunking strategy.

    This strategy intelligently combines different chunking methods based on
    content characteristics to achieve optimal results.
    """

    def __init__(self, use_llama_index: bool = False, embed_model: Any = None) -> None:
        """
        Initialize the hybrid chunking strategy.

        Args:
            use_llama_index: Whether to use LlamaIndex implementations
            embed_model: Optional embedding model for semantic chunking
        """
        super().__init__("hybrid")
        self._use_llama_index = use_llama_index
        self._embed_model = embed_model

        # Initialize component strategies (all unified)
        self._character_strategy = CharacterChunkingStrategy(use_llama_index=use_llama_index)
        self._recursive_strategy = RecursiveChunkingStrategy(use_llama_index=use_llama_index)
        self._semantic_strategy = SemanticChunkingStrategy(use_llama_index=use_llama_index, embed_model=embed_model)
        self._markdown_strategy = MarkdownChunkingStrategy(use_llama_index=use_llama_index)
        self._hierarchical_strategy = HierarchicalChunkingStrategy(use_llama_index=use_llama_index)

        # Initialize safe regex for content analysis
        self.safe_regex = SafeRegex(timeout=1.0)

    def chunk(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Apply hybrid chunking using the best strategy for the content.

        Args:
            content: The text content to chunk
            config: Configuration parameters
            progress_callback: Optional progress callback

        Returns:
            List of chunks
        """
        if not content:
            return []

        # Analyze content to determine best strategy
        analysis = self._analyze_content(content)

        # Get strategy weights from config or use defaults
        weights = config.weights if hasattr(config, "weights") else None
        adaptive = config.adaptive_weights if hasattr(config, "adaptive_weights") else True

        # Select strategy based on content analysis or use configured weights
        strategy = self._select_strategy(analysis, weights) if adaptive else self._get_weighted_strategy(config)

        # Log strategy selection
        logger.info(f"Hybrid chunking selected {strategy.name} strategy for {analysis['content_type'].value} content")

        # Apply selected strategy
        chunks = strategy.chunk(content, config, progress_callback)

        # Add hybrid metadata by creating new chunks with updated metadata
        updated_chunks = []
        for chunk in chunks:
            # Create new metadata with hybrid-specific attributes
            from dataclasses import replace

            updated_metadata = replace(
                chunk.metadata,
                custom_attributes={
                    **chunk.metadata.custom_attributes,
                    "hybrid_strategy": strategy.name,
                    "content_type": analysis["content_type"].value,
                },
            )

            # Create new chunk with updated metadata
            from shared.chunking.domain.entities.chunk import Chunk

            updated_chunk = Chunk(
                content=chunk.content,
                metadata=updated_metadata,
                min_tokens=chunk.min_tokens,
                max_tokens=chunk.max_tokens,
            )
            updated_chunks.append(updated_chunk)

        return updated_chunks

    async def chunk_async(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Asynchronous chunking.

        Args:
            content: The text content to chunk
            config: Configuration parameters
            progress_callback: Optional progress callback

        Returns:
            List of chunks
        """
        if not content:
            return []

        # Run synchronous method in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.chunk,
            content,
            config,
            progress_callback,
        )

    def _analyze_content(self, content: str) -> ContentAnalysis:
        """
        Analyze content characteristics.

        Args:
            content: Content to analyze

        Returns:
            Content analysis results
        """
        lines = content.split("\n")
        total_chars = len(content)
        total_lines = len(lines)

        # Check for markdown headers
        has_markdown = self._has_markdown_structure(content)

        # Check for code blocks
        has_code = self._has_code_blocks(content)

        # Check for structured content (lists, tables, etc.)
        has_structure = self._has_structure(content)

        # Count sentences
        sentence_count = content.count(".") + content.count("!") + content.count("?")
        avg_sentence_length = total_chars / max(1, sentence_count)

        # Determine content type
        content_type = self._determine_content_type(
            has_markdown,
            has_code,
            has_structure,
            avg_sentence_length,
        )

        # Determine if content is mixed
        is_mixed = sum([has_markdown, has_code, has_structure]) > 1

        # Recommend strategy
        recommended_strategy = self._recommend_strategy(content_type, is_mixed)

        return ContentAnalysis(
            total_chars=total_chars,
            total_lines=total_lines,
            has_markdown=has_markdown,
            has_code=has_code,
            has_structure=has_structure,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            is_mixed=is_mixed,
            content_type=content_type,
            recommended_strategy=recommended_strategy,
        )

    def _has_markdown_structure(self, content: str) -> bool:
        """Check if content has markdown structure."""
        try:
            # Look for markdown headers
            pattern = r"^#{1,6}\s+\S.*$"
            match = self.safe_regex.search_with_timeout(pattern, content, timeout=0.5)
            if match:
                return True
        except Exception as e:
            logger.debug(f"Failed to check for markdown: {e}")

        # Fallback: simple check
        return any(line.strip().startswith("#") and len(line.strip()) > 1 for line in content.split("\n")[:50])

    def _has_code_blocks(self, content: str) -> bool:
        """Check if content has code blocks."""
        # Look for triple backticks or indented code
        return "```" in content or "\n    " in content

    def _has_structure(self, content: str) -> bool:
        """Check if content has structural elements."""
        # Look for lists, tables, etc.
        indicators = [
            "\n- ",  # Unordered list
            "\n* ",  # Alternative unordered list
            "\n1. ",  # Ordered list
            "\n| ",  # Table
            "\n> ",  # Blockquote
        ]

        return any(indicator in content for indicator in indicators)

    def _determine_content_type(
        self,
        has_markdown: bool,
        has_code: bool,
        has_structure: bool,
        avg_sentence_length: float,
    ) -> ContentType:
        """
        Determine the primary content type.

        Args:
            has_markdown: Whether content has markdown
            has_code: Whether content has code blocks
            has_structure: Whether content has structure
            avg_sentence_length: Average sentence length

        Returns:
            Content type
        """
        # Priority order for content type determination
        if has_markdown:
            return ContentType.MARKDOWN
        if has_code:
            return ContentType.CODE
        if has_structure:
            return ContentType.STRUCTURED
        if avg_sentence_length > 50:  # Long sentences indicate narrative
            return ContentType.NARRATIVE
        if sum([has_markdown, has_code, has_structure]) > 1:
            return ContentType.MIXED

        return ContentType.UNKNOWN

    def _recommend_strategy(self, content_type: ContentType, is_mixed: bool) -> str:
        """
        Recommend a chunking strategy based on content type.

        Args:
            content_type: Type of content
            is_mixed: Whether content is mixed

        Returns:
            Recommended strategy name
        """
        if is_mixed:
            return "hybrid"  # Use hybrid for mixed content

        strategy_map = {
            ContentType.MARKDOWN: "markdown",
            ContentType.CODE: "markdown",  # Markdown handles code blocks well
            ContentType.STRUCTURED: "hierarchical",
            ContentType.NARRATIVE: "semantic",
            ContentType.MIXED: "recursive",
            ContentType.UNKNOWN: "character",
        }

        return strategy_map.get(content_type, "recursive")

    def _select_strategy(
        self,
        analysis: ContentAnalysis,
        weights: dict[str, float] | None = None,
    ) -> UnifiedChunkingStrategy:
        """
        Select the best strategy based on content analysis.

        Args:
            analysis: Content analysis results
            weights: Optional strategy weights

        Returns:
            Selected chunking strategy
        """
        recommended = analysis["recommended_strategy"]

        # Map strategy names to instances
        strategy_map = {
            "character": self._character_strategy,
            "recursive": self._recursive_strategy,
            "semantic": self._semantic_strategy,
            "markdown": self._markdown_strategy,
            "hierarchical": self._hierarchical_strategy,
            "hybrid": self._recursive_strategy,  # Default to recursive for hybrid recommendation
        }

        # Apply weights if provided
        if weights:
            # Calculate weighted scores
            scores = {}
            for name, _strategy in strategy_map.items():
                base_score = 1.0 if name == recommended else 0.5
                weight = weights.get(name, 1.0)
                scores[name] = base_score * weight

            # Select highest scoring strategy
            best_strategy = max(scores.items(), key=lambda x: x[1])[0]
            return strategy_map[best_strategy]

        # Use recommended strategy
        return strategy_map.get(recommended, self._recursive_strategy)

    def _get_weighted_strategy(self, config: ChunkConfig) -> UnifiedChunkingStrategy:
        """
        Get strategy based on configured weights.

        Args:
            config: Chunking configuration

        Returns:
            Selected strategy
        """
        # Get configured strategies
        strategies = config.strategies if hasattr(config, "strategies") else ["recursive"]

        # For simplicity, use the first configured strategy
        strategy_map = {
            "character": self._character_strategy,
            "recursive": self._recursive_strategy,
            "semantic": self._semantic_strategy,
            "markdown": self._markdown_strategy,
            "hierarchical": self._hierarchical_strategy,
        }

        strategy_name = strategies[0] if strategies else "recursive"
        return strategy_map.get(strategy_name, self._recursive_strategy)

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """
        Validate content for hybrid chunking.

        Args:
            content: Content to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not content:
            return False, "Content cannot be empty"

        if len(content) > 50_000_000:  # 50MB limit
            return False, f"Content too large: {len(content)} characters"

        return True, None

    def estimate_chunks(self, content_length: int, config: ChunkConfig) -> int:
        """
        Estimate the number of chunks.

        Args:
            content_length: Length of content in characters
            config: Chunking configuration

        Returns:
            Estimated chunk count
        """
        if content_length == 0:
            return 0

        # Hybrid strategy estimate depends on which strategy would be selected
        # Use recursive strategy estimate as default
        return int(self._recursive_strategy.estimate_chunks(content_length, config))
