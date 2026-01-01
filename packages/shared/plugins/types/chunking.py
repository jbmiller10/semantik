"""Chunking plugin base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig

from ..base import SemanticPlugin


class ChunkingPlugin(SemanticPlugin, ABC):
    """Base class for chunking strategy plugins."""

    PLUGIN_TYPE = "chunking"

    @abstractmethod
    def chunk(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """Chunk content into list of chunks."""

    @abstractmethod
    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """Validate that content is suitable for this strategy."""

    @abstractmethod
    def estimate_chunks(self, content_length: int, config: ChunkConfig) -> int:
        """Estimate number of chunks produced for given length/config."""
