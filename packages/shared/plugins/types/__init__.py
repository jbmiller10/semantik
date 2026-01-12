"""Type-specific plugin base classes."""

from .chunking import ChunkingPlugin
from .connector import ConnectorPlugin
from .embedding import EmbeddingPlugin
from .extractor import (
    Entity,
    ExtractionResult,
    ExtractionType,
    ExtractorPlugin,
)
from .reranker import RerankerCapabilities, RerankerPlugin, RerankResult
from .sparse_indexer import (
    SparseIndexerCapabilities,
    SparseIndexerPlugin,
    SparseQueryVector,
    SparseVector,
)

__all__ = [
    "ChunkingPlugin",
    "ConnectorPlugin",
    "EmbeddingPlugin",
    "Entity",
    "ExtractionResult",
    "ExtractionType",
    "ExtractorPlugin",
    "RerankerCapabilities",
    "RerankerPlugin",
    "RerankResult",
    "SparseIndexerCapabilities",
    "SparseIndexerPlugin",
    "SparseQueryVector",
    "SparseVector",
]
