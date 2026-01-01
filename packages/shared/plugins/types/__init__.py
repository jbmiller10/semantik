"""Type-specific plugin base classes."""

from .chunking import ChunkingPlugin
from .connector import ConnectorPlugin
from .embedding import EmbeddingPlugin

__all__ = ["EmbeddingPlugin", "ChunkingPlugin", "ConnectorPlugin"]
