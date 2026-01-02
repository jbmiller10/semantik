"""Mock implementations for plugin testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shared.plugins.types.extractor import Entity, ExtractionResult, ExtractionType
from shared.plugins.types.reranker import RerankResult


@dataclass
class MockDocument:
    """Mock document for testing connectors and extractors."""

    content: str
    """Document content text."""
    source_id: str = "test-source"
    """Source identifier."""
    file_path: str = "test/document.txt"
    """File path."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""


class MockEmbeddingService:
    """Mock embedding service for testing.

    Returns deterministic embeddings based on text content.
    """

    def __init__(self, dimension: int = 384) -> None:
        """Initialize mock embedding service.

        Args:
            dimension: Embedding vector dimension.
        """
        self.dimension = dimension
        self.embed_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding from text.

        Uses hash of text to create reproducible but varied embeddings.
        """
        # Simple hash-based deterministic embedding
        text_hash = hash(text)
        return [(text_hash >> i) % 100 / 100.0 for i in range(self.dimension)]

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        self.embed_calls.append(text)
        return self._generate_embedding(text)

    async def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,  # noqa: ARG002
    ) -> list[list[float]]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size (ignored in mock).

        Returns:
            List of embedding vectors.
        """
        self.batch_calls.append(texts)
        return [self._generate_embedding(text) for text in texts]

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimension


class MockReranker:
    """Mock reranker for testing.

    Returns documents sorted by simple text similarity to query.
    """

    def __init__(self) -> None:
        """Initialize mock reranker."""
        self.rerank_calls: list[tuple[str, list[str]]] = []

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[RerankResult]:
        """Rerank documents by simulated relevance.

        Args:
            query: Search query.
            documents: Documents to rerank.
            top_k: Number of results to return.
            metadata: Optional metadata for each document.

        Returns:
            List of RerankResult objects.
        """
        self.rerank_calls.append((query, documents))

        # Simple scoring: count common words between query and doc
        query_words = set(query.lower().split())
        scored = []
        for i, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            score = len(query_words & doc_words) / max(len(query_words), 1)
            scored.append((i, score, doc))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        results: list[RerankResult] = []
        for index, score, doc in scored:
            doc_metadata = metadata[index] if metadata and index < len(metadata) else {}
            results.append(RerankResult(index=index, score=score, document=doc, metadata=doc_metadata))
        return results


class MockExtractor:
    """Mock extractor for testing.

    Returns simple mock extraction results.
    """

    def __init__(self) -> None:
        """Initialize mock extractor."""
        self.extract_calls: list[str] = []

    async def extract(
        self,
        text: str,
        extraction_types: list[ExtractionType] | None = None,  # noqa: ARG002
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> ExtractionResult:
        """Extract mock metadata from text.

        Args:
            text: Text to extract from.
            extraction_types: Types of extraction to perform (ignored in mock).
            options: Plugin-specific options (ignored in mock).

        Returns:
            Mock extraction result.
        """
        self.extract_calls.append(text)

        # Simple mock extraction
        words = text.split()
        entities: list[Entity] = []
        cursor = 0
        for word in words[:3]:
            start = text.find(word, cursor)
            if start == -1:
                start = cursor
            end = start + len(word)
            cursor = end
            entities.append(Entity(text=word, type="MOCK", start=start, end=end, confidence=0.9))

        return ExtractionResult(
            entities=entities,
            keywords=words[:5],
            language="en",
            language_confidence=0.95,
            custom={"options": options or {}},
        )


class MockChunker:
    """Mock chunker for testing.

    Splits text into fixed-size chunks.
    """

    def __init__(self, chunk_size: int = 100, overlap: int = 20) -> None:
        """Initialize mock chunker.

        Args:
            chunk_size: Target chunk size in characters.
            overlap: Overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_calls: list[str] = []

    def chunk(self, content: str) -> list[dict[str, Any]]:
        """Chunk content into pieces.

        Args:
            content: Content to chunk.

        Returns:
            List of chunk dictionaries.
        """
        self.chunk_calls.append(content)

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_text = content[start:end]

            chunks.append(
                {
                    "content": chunk_text,
                    "index": chunk_index,
                    "start_offset": start,
                    "end_offset": end,
                    "metadata": {},
                }
            )

            chunk_index += 1
            start = end - self.overlap if end < len(content) else end

        return chunks
