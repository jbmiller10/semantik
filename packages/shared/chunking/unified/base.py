#!/usr/bin/env python3
"""
Unified chunking strategy base class.

This module provides the single abstract base class for all chunking strategies,
combining domain-driven design with optional LlamaIndex integration.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tiktoken

from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig


@lru_cache(maxsize=1)
def _get_tokenizer() -> tiktoken.Encoding:
    """Get cached tiktoken encoder."""
    return tiktoken.get_encoding("cl100k_base")


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
        Count tokens in text using tiktoken.

        Uses cl100k_base encoding (GPT-4/ChatGPT tokenizer) for accurate
        token counting that aligns with most modern embedding models.

        Args:
            text: Text to count tokens in

        Returns:
            Exact token count
        """
        if not text:
            return 0
        return len(_get_tokenizer().encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within a maximum token count.

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens

        Returns:
            Truncated text that fits within max_tokens
        """
        if not text or max_tokens <= 0:
            return ""
        tokenizer = _get_tokenizer()
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return tokenizer.decode(tokens[:max_tokens])

    def split_to_token_limit(self, text: str, max_tokens: int) -> list[str]:
        """
        Split text into chunks that fit within max_tokens, breaking at word boundaries.

        This preserves all content by creating multiple chunks rather than truncating.
        Used when a chunk exceeds max_tokens and needs to be re-split.

        If a single "word" (text without whitespace) exceeds max_tokens, it will be
        split at the token level to guarantee all chunks fit within the limit.

        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk

        Returns:
            List of text chunks, each within max_tokens
        """
        if not text or max_tokens <= 0:
            return []

        if self.count_tokens(text) <= max_tokens:
            return [text]

        chunks = []
        words = text.split()
        current_words: list[str] = []
        current_tokens = 0

        for word in words:
            word_tokens = self.count_tokens(word)

            # Handle oversized words by splitting at token level
            if word_tokens > max_tokens:
                # Flush current buffer first
                if current_words:
                    chunks.append(" ".join(current_words))
                    current_words = []
                    current_tokens = 0

                # Split the oversized word at token boundaries
                word_chunks = self._split_word_by_tokens(word, max_tokens)
                chunks.extend(word_chunks)
                continue

            # Account for space between words
            effective_tokens = word_tokens + (1 if current_words else 0)

            if current_tokens + effective_tokens > max_tokens and current_words:
                chunks.append(" ".join(current_words))
                current_words = [word]
                current_tokens = word_tokens
            else:
                current_words.append(word)
                current_tokens += effective_tokens

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

    def _split_word_by_tokens(self, word: str, max_tokens: int) -> list[str]:
        """
        Split a single word (text without whitespace) into chunks at token boundaries.

        This handles cases like very long URLs, base64 data, or minified code
        that cannot be split at word boundaries.

        Args:
            word: The word to split (should not contain whitespace)
            max_tokens: Maximum tokens per chunk

        Returns:
            List of text chunks, each within max_tokens
        """
        if not word or max_tokens <= 0:
            return []

        tokenizer = _get_tokenizer()
        tokens = tokenizer.encode(word)

        if len(tokens) <= max_tokens:
            return [word]

        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk_text = tokenizer.decode(chunk_tokens)
            if chunk_text:  # Only add non-empty chunks
                chunks.append(chunk_text)

        return chunks

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

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for plugin configuration.

        Returns the common configuration options for all chunking strategies.
        Subclasses should override to add strategy-specific options.

        Returns:
            JSON Schema dictionary for configuration.
        """
        return {
            "type": "object",
            "properties": {
                "min_tokens": {
                    "type": "integer",
                    "title": "Minimum Tokens",
                    "description": "Minimum number of tokens per chunk",
                    "default": 100,
                    "minimum": 1,
                },
                "max_tokens": {
                    "type": "integer",
                    "title": "Maximum Tokens",
                    "description": "Maximum number of tokens per chunk",
                    "default": 1000,
                    "minimum": 1,
                },
                "overlap_tokens": {
                    "type": "integer",
                    "title": "Overlap Tokens",
                    "description": "Number of overlapping tokens between chunks",
                    "default": 50,
                    "minimum": 0,
                },
            },
        }

    def __repr__(self) -> str:
        """String representation of the strategy."""
        return f"{self.__class__.__name__}(name='{self._name}')"
