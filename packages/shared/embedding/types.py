"""Type definitions for the embedding module."""

from enum import Enum


class EmbeddingMode(str, Enum):
    """Mode for embedding generation - affects prefix/instruction handling.

    Many retrieval models are asymmetric, meaning they need different processing
    for queries (what users search for) vs documents (what gets indexed).

    Examples:
        - E5 models: "query: {text}" vs "passage: {text}"
        - BGE models: instruction prefix for queries, none for documents
        - Qwen models: "Instruct: {task}\\nQuery:{text}" vs raw text

    Usage:
        # For search queries
        embeddings = await provider.embed_texts(texts, mode=EmbeddingMode.QUERY)

        # For document indexing
        embeddings = await provider.embed_texts(texts, mode=EmbeddingMode.DOCUMENT)
    """

    QUERY = "query"
    """Embedding a search query - applies query-specific prefixes/instructions."""

    DOCUMENT = "document"
    """Embedding a document for indexing - typically no prefix or document-specific prefix."""


__all__ = ["EmbeddingMode"]
