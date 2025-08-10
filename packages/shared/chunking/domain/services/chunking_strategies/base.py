#!/usr/bin/env python3
"""
Abstract base class for all chunking strategies.

This module defines the interface that all chunking strategies must implement,
ensuring consistency across different chunking approaches.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig


class ChunkingStrategy(ABC):
    """
    Abstract base class for all chunking strategies.

    Each strategy implements a specific approach to breaking text into chunks,
    such as character-based, semantic, or structural chunking.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the strategy with a name.

        Args:
            name: The name of the strategy
        """
        self._name = name

    @property
    def name(self) -> str:
        """Get the strategy name."""
        return self._name

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
        return int(0.3 * word_based_estimate + 0.7 * char_based_estimate)

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
