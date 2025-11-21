#!/usr/bin/env python3
"""
Abstract base class for streaming chunking strategies.

This module defines the interface that all streaming chunking strategies must implement,
ensuring consistent stream processing across different chunking approaches.
"""

from abc import ABC, abstractmethod
from typing import Any

from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.infrastructure.streaming.window import StreamingWindow


class StreamingChunkingStrategy(ABC):
    """
    Base class for streaming chunking strategies.

    Each strategy processes streaming windows of data to produce chunks
    while maintaining bounded memory usage and preserving output consistency
    with non-streaming versions.
    """

    def __init__(self, name: str):
        """
        Initialize the streaming strategy.

        Args:
            name: The name of the strategy
        """
        self._name = name
        self._buffer_size = 0
        self._state: dict[str, Any] = {}  # Strategy-specific state
        self._is_finalized = False

    @property
    def name(self) -> str:
        """Get the strategy name."""
        return self._name

    @abstractmethod
    async def process_window(self, window: StreamingWindow, config: ChunkConfig, is_final: bool = False) -> list[Chunk]:
        """
        Process a window of streaming data to produce chunks.

        Args:
            window: StreamingWindow containing the data to process
            config: Chunk configuration parameters
            is_final: Whether this is the final window (EOF reached)

        Returns:
            List of chunks produced from this window
        """

    @abstractmethod
    def get_buffer_size(self) -> int:
        """
        Return the current buffer size in bytes.

        Returns:
            Size of internal buffers in bytes
        """

    @abstractmethod
    def get_max_buffer_size(self) -> int:
        """
        Return the maximum allowed buffer size for this strategy.

        Returns:
            Maximum buffer size in bytes
        """

    def reset(self) -> None:
        """
        Reset the strategy state for processing a new document.

        This should be called before processing a new document
        to clear any internal state from previous processing.
        """
        self._buffer_size = 0
        self._state = {}
        self._is_finalized = False

    def validate_memory_constraint(self) -> bool:
        """
        Validate that the strategy is within its memory constraints.

        Returns:
            True if within memory limits, False otherwise
        """
        return self.get_buffer_size() <= self.get_max_buffer_size()

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using a simple approximation.

        Args:
            text: Text to count tokens in

        Returns:
            Approximate token count
        """
        # Use the same approximation as non-streaming strategies
        word_count = len(text.split())
        char_count = len(text)

        word_based_estimate = word_count * 1.3
        char_based_estimate = char_count / 4

        return int(0.3 * word_based_estimate + 0.7 * char_based_estimate)

    def find_sentence_boundary(self, text: str, target_position: int, prefer_before: bool = True) -> int:
        """
        Find the nearest sentence boundary to a target position.

        Args:
            text: The text to search in
            target_position: Target character position
            prefer_before: If True, prefer boundary before target

        Returns:
            Position of nearest sentence boundary
        """
        if not text or target_position < 0:
            return 0

        if target_position >= len(text):
            return len(text)

        if prefer_before:
            for i in range(target_position, -1, -1):
                if i < len(text) and text[i].isspace():
                    return i + 1
            return 0

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
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            cleaned_line = " ".join(line.split())
            if cleaned_line:
                cleaned_lines.append(cleaned_line)

        result = "\n".join(cleaned_lines)
        return result.strip()

    async def finalize(self, config: ChunkConfig) -> list[Chunk]:  # noqa: ARG002
        """
        Finalize processing and return any remaining chunks.

        This should be called after all windows have been processed
        to flush any remaining buffered data.

        Args:
            config: Chunk configuration parameters

        Returns:
            List of final chunks
        """
        self._is_finalized = True
        return []

    def __repr__(self) -> str:
        """String representation of the strategy."""
        return (
            f"{self.__class__.__name__}(name='{self._name}', "
            f"buffer_size={self.get_buffer_size()}, "
            f"max_buffer={self.get_max_buffer_size()})"
        )
