#!/usr/bin/env python3
"""
Unified chunking strategy base class.

This module provides the single abstract base class for all chunking strategies,
combining domain-driven design with optional LlamaIndex integration.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig


@dataclass
class UnifiedChunkResult:
    """Unified chunk result that works with both approaches."""

    chunk: Chunk
    chunk_id: str
    text: str
    start_offset: int
    end_offset: int
    metadata: dict[str, Any]
    embedding: list[float] | None = None


class UnifiedChunkingStrategy(ABC):
    """
    Single abstract base class for all chunking strategies.

    This class provides a unified interface that can leverage both domain-level
    implementations and LlamaIndex-based implementations where appropriate.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the strategy with a name.

        Args:
            name: The name of the strategy
        """
        self._name = name
        self._use_llama_index = False

    @property
    def name(self) -> str:
        """Get the strategy name."""
        return self._name

    @property
    def use_llama_index(self) -> bool:
        """Check if this strategy should use LlamaIndex implementation."""
        return self._use_llama_index

    @use_llama_index.setter
    def use_llama_index(self, value: bool) -> None:
        """Set whether to use LlamaIndex implementation."""
        self._use_llama_index = value

    @abstractmethod
    def chunk(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Apply the chunking strategy to break content into chunks.

        Args:
            content: The text content to chunk
            config: Configuration parameters for chunking
            progress_callback: Optional callback to report progress (0-100)

        Returns:
            List of chunks created from the content
        """

    @abstractmethod
    async def chunk_async(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Asynchronous chunking for I/O bound operations.

        Args:
            content: The text content to chunk
            config: Configuration parameters for chunking
            progress_callback: Optional callback to report progress (0-100)

        Returns:
            List of chunks created from the content
        """

    @abstractmethod
    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """
        Validate that content is suitable for this strategy.

        Args:
            content: The content to validate

        Returns:
            Tuple of (is_valid, error_message)
        """

    @abstractmethod
    def estimate_chunks(self, content_length: int, config: ChunkConfig) -> int:
        """
        Estimate the number of chunks that will be produced.

        Args:
            content_length: Length of content in characters
            config: Configuration parameters

        Returns:
            Estimated number of chunks
        """

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using a simple approximation.

        This is a pure business logic function that doesn't depend on
        any external tokenizer libraries.

        Args:
            text: Text to count tokens in

        Returns:
            Approximate token count
        """
        # Business rule: approximate 1 token ≈ 4 characters for English text
        # This is a domain-level approximation that doesn't require external dependencies

        # Adjust for different text characteristics
        word_count = len(text.split())
        char_count = len(text)

        # Use a weighted average of word-based and character-based estimates
        # Typically, 1 word ≈ 1.3 tokens, and 4 characters ≈ 1 token
        word_based_estimate = word_count * 1.3
        char_based_estimate = char_count / 4

        # Weight character-based estimate more heavily for consistency
        # Ensure at least 1 token for any non-empty text
        token_count = int(0.3 * word_based_estimate + 0.7 * char_based_estimate)
        return max(1, token_count) if text else 0

    def calculate_overlap_size(self, chunk_size: int, overlap_percentage: float) -> int:
        """
        Calculate overlap size in tokens.

        Args:
            chunk_size: Size of chunk in tokens
            overlap_percentage: Overlap as percentage (0-100)

        Returns:
            Number of overlapping tokens
        """
        return int(chunk_size * (overlap_percentage / 100))

    def find_sentence_boundary(self, text: str, target_position: int, prefer_before: bool = True) -> int:
        """
        Find the nearest sentence boundary to a target position.

        Args:
            text: The text to search in
            target_position: Target character position
            prefer_before: If True, prefer boundary before target; otherwise after

        Returns:
            Position of nearest sentence boundary
        """
        if not text or target_position < 0:
            return 0

        if target_position >= len(text):
            return len(text)

        # Sentence ending markers
        sentence_endings = {".", "!", "?"}

        # Search for nearest sentence boundary
        if prefer_before:
            # Search backwards
            for i in range(target_position, -1, -1):
                if i < len(text) and text[i] in sentence_endings and i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1
        else:
            # Search forwards
            for i in range(target_position, len(text)):
                if text[i] in sentence_endings and i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1

        # If no boundary found, return target position
        return target_position

    def find_word_boundary(self, text: str, target_position: int, prefer_before: bool = True) -> int:
        """
        Find the nearest word boundary to a target position.

        Args:
            text: The text to search in
            target_position: Target character position
            prefer_before: If True, prefer boundary before target; otherwise after

        Returns:
            Position of nearest word boundary
        """
        if not text or target_position < 0:
            return 0

        if target_position >= len(text):
            return len(text)

        # If we're already at a space, we're at a boundary
        if text[target_position].isspace():
            return target_position

        if prefer_before:
            # Search backwards for space
            for i in range(target_position, -1, -1):
                if i < len(text) and text[i].isspace():
                    return i + 1
            return 0
        # Search forwards for space
        for i in range(target_position, len(text)):
            if text[i].isspace():
                return i
        return len(text)

    def clean_chunk_text(self, text: str) -> str:
        """
        Clean and normalize chunk text.

        Args:
            text: Raw chunk text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Preserve paragraph structure but clean up spacing
            cleaned_line = " ".join(line.split())
            if cleaned_line:
                cleaned_lines.append(cleaned_line)

        # Rejoin with single newlines
        result = "\n".join(cleaned_lines)

        # Ensure text doesn't start or end with whitespace
        return result.strip()

    def __repr__(self) -> str:
        """String representation of the strategy."""
        return f"{self.__class__.__name__}(name='{self._name}')"
