"""Reranker plugin base class for document reranking."""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from shared.plugins.base import SemanticPlugin
from shared.plugins.manifest import PluginManifest


@dataclass(frozen=True)
class RerankResult:
    """Result of reranking a document."""

    index: int
    """Original index in input list."""
    score: float
    """Relevance score (higher = more relevant)."""
    document: str
    """The document text."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional metadata associated with the document."""


@dataclass(frozen=True)
class RerankerCapabilities:
    """Capabilities and limits of a reranker plugin."""

    max_documents: int
    """Maximum documents per request."""
    max_query_length: int
    """Maximum query length in characters."""
    max_doc_length: int
    """Maximum document length in characters."""
    supports_batching: bool
    """Whether the reranker can process multiple queries in one call."""
    models: list[str] = field(default_factory=list)
    """Available model variants (if applicable)."""


class RerankerPlugin(SemanticPlugin, ABC):
    """Base class for document reranker plugins.

    Rerankers improve search quality by re-scoring initial retrieval results
    using more sophisticated models (typically cross-encoders).

    Example usage:
        reranker = MyRerankerPlugin(config={...})
        await reranker.initialize()

        results = await reranker.rerank(
            query="quantum computing applications",
            documents=["doc1", "doc2", "doc3"],
            top_k=2
        )
        # Returns top 2 documents sorted by relevance
    """

    PLUGIN_TYPE = "reranker"

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of results to return. If None, return all documents.
            metadata: Optional metadata for each document (same length as documents).

        Returns:
            List of RerankResult sorted by relevance (highest score first).
            Length is min(top_k, len(documents)) if top_k specified.
        """

    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> RerankerCapabilities:
        """Return reranker capabilities and limits.

        This allows the system to make informed decisions about batching
        and to warn users when limits may be exceeded.
        """

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin manifest for discovery and UI.

        Builds a PluginManifest from the plugin's class variables and capabilities.
        Subclasses may override for custom manifest generation.

        Returns:
            PluginManifest with reranker metadata.
        """
        metadata = getattr(cls, "METADATA", {})
        capabilities = None
        # Include capabilities if available (not abstract on concrete class)
        with contextlib.suppress(TypeError, NotImplementedError):
            capabilities = asdict(cls.get_capabilities())

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

    async def rerank_batch(
        self,
        queries: list[str],
        documents_per_query: list[list[str]],
        top_k: int | None = None,
    ) -> list[list[RerankResult]]:
        """Batch rerank multiple queries.

        Override for optimized batch processing. Default implementation
        calls rerank() for each query sequentially.

        Args:
            queries: List of search queries.
            documents_per_query: List of document lists, one per query.
            top_k: Number of results to return per query.

        Returns:
            List of rerank results, one list per query.
        """
        results = []
        for query, docs in zip(queries, documents_per_query, strict=True):
            result = await self.rerank(query, docs, top_k)
            results.append(result)
        return results
