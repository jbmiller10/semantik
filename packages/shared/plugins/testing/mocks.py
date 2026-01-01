"""Mock implementations for plugin testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        self, texts: list[str], batch_size: int = 32  # noqa: ARG002
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
    ) -> list[tuple[int, float, str]]:
        """Rerank documents by simulated relevance.

        Args:
            query: Search query.
            documents: Documents to rerank.
            top_k: Number of results to return.

        Returns:
            List of (original_index, score, document) tuples.
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

        return scored


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
        extraction_types: list[str] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Extract mock metadata from text.

        Args:
            text: Text to extract from.
            extraction_types: Types of extraction to perform (ignored in mock).

        Returns:
            Mock extraction result.
        """
        self.extract_calls.append(text)

        # Simple mock extraction
        words = text.split()
        return {
            "entities": [{"text": w, "type": "MOCK"} for w in words[:3]],
            "keywords": words[:5],
            "language": "en",
            "language_confidence": 0.95,
        }


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
