#!/usr/bin/env python3
"""
Abstract base class for all chunking strategies.

This module defines the interface that all chunking strategies must implement,
ensuring consistency across different chunking approaches.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from typing import Any, ClassVar

import tiktoken

from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig


@lru_cache(maxsize=1)
def _get_tokenizer() -> tiktoken.Encoding:
    """Get cached tiktoken encoder."""
    return tiktoken.get_encoding("cl100k_base")


class ChunkingStrategy(ABC):
    """
    Abstract base class for all chunking strategies.

    Each strategy implements a specific approach to breaking text into chunks,
    such as character-based, semantic, or structural chunking.

    Plugin Configuration:
        External chunking plugins can receive global configuration via the
        `configure()` method. This is called by the factory with settings
        from the plugin state file (e.g., API keys, model settings).
    """

    # Protocol-required class variables for plugin identification
    PLUGIN_ID: ClassVar[str] = ""
    """Unique plugin identifier - must be set by subclass."""

    PLUGIN_TYPE: ClassVar[str] = "chunking"
    """Plugin type identifier."""

    PLUGIN_VERSION: ClassVar[str] = "0.0.0"
    """Semantic version string."""

    def __init__(self, name: str) -> None:
        """
        Initialize the strategy with a name.

        Args:
            name: The name of the strategy
        """
        self._name = name
        self._plugin_config: dict[str, Any] = {}

    def configure(self, config: dict[str, Any]) -> None:
        """
        Configure the strategy with plugin-level settings.

        This method is called by the factory when creating strategies from
        external plugins. It allows plugins to receive configuration from
        the shared plugin state file (e.g., API keys, default settings).

        Note: This is different from ChunkConfig passed to chunk() which
        contains per-operation settings like chunk_size and overlap.

        Args:
            config: Plugin configuration dictionary (secrets already resolved)
        """
        self._plugin_config = config

    @property
    def plugin_config(self) -> dict[str, Any]:
        """Get the plugin configuration."""
        return self._plugin_config

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
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin manifest for discovery and UI.

        Returns a dictionary to maintain compatibility with the protocol
        interface (external plugins return dicts, not PluginManifest dataclasses).

        Returns:
            Dictionary with plugin metadata.
        """
        metadata = getattr(cls, "METADATA", {})
        return {
            "id": cls.PLUGIN_ID,
            "type": cls.PLUGIN_TYPE,
            "version": cls.PLUGIN_VERSION,
            "display_name": metadata.get("display_name", cls.PLUGIN_ID or cls.__name__),
            "description": metadata.get("description", ""),
        }

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
