#!/usr/bin/env python3
"""
Base chunker interface for all chunking strategies.

This module provides the abstract base class and type definitions for
implementing different text chunking strategies.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Type-safe chunk result."""

    chunk_id: str
    text: str
    start_offset: int
    end_offset: int
    metadata: dict[str, Any]
    embedding: list[float] | None = None


class BaseChunker(ABC):
    """Base class for all chunking strategies."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize chunker with strategy-specific parameters."""
        self.strategy_name: str = self.__class__.__name__.replace("Chunker", "").lower()
        logger.info(f"Initializing {self.strategy_name} chunker with params: {kwargs}")

    @abstractmethod
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Synchronous chunking method.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks

        Returns:
            List of ChunkResult objects
        """

    @abstractmethod
    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Asynchronous chunking for I/O bound operations.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks

        Returns:
            List of ChunkResult objects
        """

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate strategy-specific configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid, False otherwise
        """

    @abstractmethod
    def estimate_chunks(self, text_length: int, config: dict[str, Any]) -> int:
        """Estimate number of chunks for capacity planning.

        Args:
            text_length: Length of text in characters
            config: Configuration parameters

        Returns:
            Estimated number of chunks
        """

    def _create_chunk_result(
        self,
        doc_id: str,
        chunk_index: int,
        text: str,
        start_offset: int,
        end_offset: int,
        metadata: dict[str, Any] | None = None,
    ) -> ChunkResult:
        """Helper method to create consistent ChunkResult objects."""
        chunk_metadata = {
            "strategy": self.strategy_name,
            "chunk_index": chunk_index,
        }

        if metadata:
            chunk_metadata.update(metadata)

        return ChunkResult(
            chunk_id=f"{doc_id}_{chunk_index:04d}",
            text=text.strip(),
            start_offset=start_offset,
            end_offset=end_offset,
            metadata=chunk_metadata,
        )
