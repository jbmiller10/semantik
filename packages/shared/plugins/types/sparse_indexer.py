"""Sparse indexer plugin base class for BM25/SPLADE indexing.

This module defines the SparseIndexerPlugin ABC, which provides a unified interface
for sparse vector indexing implementations in the Semantik plugin system.

Sparse indexers generate sparse vector representations for documents:
- BM25: Classic term-frequency based retrieval (indices=term_ids, values=tf-idf)
- SPLADE: Learned sparse representations (indices=token_ids, values=learned_weights)

The plugin is responsible ONLY for vector generation - persistence to Qdrant
is handled by vecpipe infrastructure.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

from shared.plugins.base import SemanticPlugin
from shared.plugins.manifest import PluginManifest
from shared.plugins.typed_dicts import SPARSE_TYPES


@dataclass(frozen=True)
class SparseVector:
    """Sparse vector representation for document indexing.

    For BM25: indices are term IDs, values are TF-IDF scores.
    For SPLADE: indices are token IDs, values are learned weights.

    Uses chunk_id (not document_id) to align 1:1 with dense vectors for RRF fusion.

    Attributes:
        indices: Sparse vector indices (term/token IDs). Must be sorted ascending.
        values: Corresponding weights/scores. Same length as indices.
        chunk_id: Chunk identifier (aligns with dense vectors).
        metadata: Additional indexing metadata.
    """

    indices: tuple[int, ...]
    """Sparse vector indices (term/token IDs)."""
    values: tuple[float, ...]
    """Corresponding weights/scores."""
    chunk_id: str
    """Chunk identifier (aligns with dense vectors)."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional indexing metadata."""

    def __post_init__(self) -> None:
        """Validate indices and values have matching lengths."""
        if len(self.indices) != len(self.values):
            msg = f"indices and values must have same length: {len(self.indices)} != {len(self.values)}"
            raise ValueError(msg)


@dataclass(frozen=True)
class SparseQueryVector:
    """Sparse vector representation for query encoding.

    Attributes:
        indices: Sparse vector indices (term/token IDs). Must be sorted ascending.
        values: Corresponding weights/scores. Same length as indices.
    """

    indices: tuple[int, ...]
    """Sparse vector indices (term/token IDs)."""
    values: tuple[float, ...]
    """Corresponding weights/scores."""

    def __post_init__(self) -> None:
        """Validate indices and values have matching lengths."""
        if len(self.indices) != len(self.values):
            msg = f"indices and values must have same length: {len(self.indices)} != {len(self.values)}"
            raise ValueError(msg)


@dataclass(frozen=True)
class SparseIndexerCapabilities:
    """Capabilities and limits of a sparse indexer plugin.

    Attributes:
        sparse_type: Type of sparse representation ('bm25' or 'splade').
        max_tokens: Maximum tokens per document.
        vocabulary_handling: How vocabulary maps to sparse dimensions
            ('direct' or 'hashed').
        supports_batching: Whether batch encoding is supported.
        max_batch_size: Maximum documents per batch.
        requires_corpus_stats: Whether indexer needs corpus statistics (e.g., BM25 IDF).
        max_terms_per_vector: Maximum non-zero terms in output sparse vector.
            None means no limit.
        vocabulary_size: Vocabulary size if fixed. None for open vocabulary.
        supports_filters: Whether metadata filters are supported during search.
        idf_storage: IDF storage backend ('file' or 'qdrant_point').
        supported_languages: Supported languages (ISO 639-1 codes). None for all.
    """

    sparse_type: str
    """Type: 'bm25' or 'splade'."""
    max_tokens: int
    """Maximum tokens per document."""
    vocabulary_handling: str = "direct"
    """How vocabulary maps to sparse dimensions: 'direct' or 'hashed'."""
    supports_batching: bool = True
    """Whether batch encoding is supported."""
    max_batch_size: int = 64
    """Maximum documents per batch."""
    requires_corpus_stats: bool = False
    """Whether indexer needs corpus statistics (e.g., BM25 IDF)."""
    max_terms_per_vector: int | None = None
    """Maximum non-zero terms in output sparse vector. None = no limit."""
    vocabulary_size: int | None = None
    """Vocabulary size (if fixed). None = open vocabulary."""
    supports_filters: bool = False
    """Whether metadata filters are supported during search."""
    idf_storage: str = "file"
    """IDF storage backend: 'file' or 'qdrant_point'."""
    supported_languages: tuple[str, ...] | None = None
    """Supported languages (ISO 639-1 codes). None for all languages."""

    def __post_init__(self) -> None:
        """Validate sparse_type is valid."""
        if self.sparse_type not in SPARSE_TYPES:
            msg = f"Invalid sparse_type: '{self.sparse_type}'. Must be one of: {sorted(SPARSE_TYPES)}"
            raise ValueError(msg)


class SparseIndexerPlugin(SemanticPlugin, ABC):
    """Base class for sparse indexer plugins (BM25, SPLADE).

    Sparse indexers generate sparse vector representations for documents.
    The plugin is responsible ONLY for vector generation - persistence to
    Qdrant is handled by vecpipe infrastructure.

    Constraint: One sparse indexer per collection. A collection can have
    either BM25 or SPLADE enabled, but not both simultaneously.

    Class Variables:
        PLUGIN_TYPE: Always "sparse_indexer" for sparse indexer plugins.
        PLUGIN_ID: Unique identifier for the plugin (must be set by subclass).
        PLUGIN_VERSION: Semantic version of the plugin (must be set by subclass).
        SPARSE_TYPE: Type of sparse representation ('bm25' or 'splade').

    Example implementation:

        class MyBM25Plugin(SparseIndexerPlugin):
            PLUGIN_ID = "my-bm25"
            PLUGIN_VERSION = "1.0.0"
            SPARSE_TYPE = "bm25"

            async def encode_documents(
                self, documents: list[dict[str, Any]]
            ) -> list[SparseVector]:
                results = []
                for doc in documents:
                    indices, values = self._compute_bm25(doc["content"])
                    results.append(SparseVector(
                        indices=tuple(indices),
                        values=tuple(values),
                        chunk_id=doc["chunk_id"],
                    ))
                return results

            async def encode_query(self, query: str) -> SparseQueryVector:
                indices, values = self._compute_query_vector(query)
                return SparseQueryVector(
                    indices=tuple(indices),
                    values=tuple(values),
                )

            async def remove_documents(self, chunk_ids: list[str]) -> None:
                # Update IDF statistics when chunks are removed
                await self._update_idf_for_removal(chunk_ids)

            @classmethod
            def get_capabilities(cls) -> SparseIndexerCapabilities:
                return SparseIndexerCapabilities(
                    sparse_type="bm25",
                    max_tokens=8192,
                    requires_corpus_stats=True,
                )
    """

    PLUGIN_TYPE: ClassVar[str] = "sparse_indexer"

    # Subclasses must define these
    PLUGIN_ID: ClassVar[str]
    PLUGIN_VERSION: ClassVar[str]
    SPARSE_TYPE: ClassVar[str]
    """Sparse representation type: 'bm25' or 'splade'."""

    @abstractmethod
    async def encode_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[SparseVector]:
        """Generate sparse vectors for documents.

        NOTE: This method generates vectors only. Persistence to Qdrant
        is handled by vecpipe infrastructure, not the plugin.

        Args:
            documents: List of documents with keys:
                - content (str): Document/chunk text
                - chunk_id (str): Unique chunk identifier (aligns with dense vectors)
                - metadata (dict, optional): Additional metadata

        Returns:
            List of SparseVector instances, one per input document.
            Order must match input document order.
        """

    @abstractmethod
    async def encode_query(self, query: str) -> SparseQueryVector:
        """Generate sparse vector for a search query.

        Args:
            query: Search query text.

        Returns:
            SparseQueryVector with indices and values.
        """

    @abstractmethod
    async def remove_documents(self, chunk_ids: list[str]) -> None:
        """Clean up any plugin-internal state for removed chunks.

        Called when chunks are deleted from the collection.
        For stateless plugins (like SPLADE), this may be a no-op.
        For BM25, this updates IDF statistics.

        Args:
            chunk_ids: List of chunk IDs being removed (aligns with dense vectors).
        """

    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> SparseIndexerCapabilities:
        """Return sparse indexer capabilities and limits.

        This allows the system to make informed decisions about batching,
        IDF storage, and to warn users when limits may be exceeded.
        """

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin manifest for discovery and UI.

        Builds a PluginManifest from the plugin's class variables and capabilities.
        Subclasses may override for custom manifest generation.

        Returns:
            PluginManifest with sparse indexer metadata.
        """
        metadata = getattr(cls, "METADATA", {})
        capabilities: dict[str, Any] = {}

        # Include capabilities if available (not abstract on concrete class)
        with contextlib.suppress(TypeError, NotImplementedError):
            caps = cls.get_capabilities()
            capabilities = asdict(caps)

        # Ensure sparse_type is in capabilities even if get_capabilities() failed
        if not capabilities:
            sparse_type = getattr(cls, "SPARSE_TYPE", None)
            if sparse_type:
                capabilities = {"sparse_type": sparse_type}

        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name=metadata.get("display_name", cls.PLUGIN_ID),
            description=metadata.get("description", ""),
            author=metadata.get("author"),
            homepage=metadata.get("homepage"),
            capabilities=capabilities,
        )

    def get_sparse_collection_name(self, base_name: str) -> str:
        """Generate Qdrant collection name for this sparse indexer.

        Args:
            base_name: Base collection name (dense collection).

        Returns:
            Sparse collection name (e.g., "work_docs_sparse_bm25").
        """
        return f"{base_name}_sparse_{self.SPARSE_TYPE}"

    async def encode_documents_batch(
        self,
        document_batches: list[list[dict[str, Any]]],
    ) -> list[list[SparseVector]]:
        """Batch encode multiple document lists.

        Override for optimized batch processing. Default implementation
        calls encode_documents() for each batch sequentially.

        Args:
            document_batches: List of document lists to encode.

        Returns:
            List of sparse vector lists, one per input batch.
        """
        results = []
        for batch in document_batches:
            result = await self.encode_documents(batch)
            results.append(result)
        return results
