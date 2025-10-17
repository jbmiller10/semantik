"""Adapter to use local embedding service with LlamaIndex.

This module provides an adapter that bridges between the local DenseEmbeddingService
and LlamaIndex's BaseEmbedding interface, enabling use of local embeddings in
LlamaIndex-based chunking strategies.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from llama_index.core.embeddings import BaseEmbedding

from packages.shared.embedding.dense import embedding_service
from packages.shared.text_processing.exceptions import EmbeddingError, EmbeddingServiceNotInitializedError

logger = logging.getLogger(__name__)


class LocalEmbeddingAdapter(BaseEmbedding):
    """Adapter to use local embedding service with LlamaIndex components."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the adapter.

        Args:
            **kwargs: Additional arguments (for compatibility)
        """
        super().__init__(**kwargs)
        self._embed_dim: int | None = None

    @property
    def embed_dim(self) -> int:
        """Get embedding dimension.

        Returns:
            Embedding dimension

        Raises:
            EmbeddingError: If embedding service is not initialized or dimension cannot be determined
        """
        if self._embed_dim is None:
            # Get dimension from the actual embedding service
            try:
                # Access the underlying DenseEmbeddingService instance
                if hasattr(embedding_service, "_service") and hasattr(embedding_service._service, "get_dimension"):
                    self._embed_dim = embedding_service._service.get_dimension()
                elif hasattr(embedding_service, "_instance") and embedding_service._instance is not None:
                    # For lazy loading case
                    if hasattr(embedding_service._instance, "_service"):
                        self._embed_dim = embedding_service._instance._service.get_dimension()
                    else:
                        raise RuntimeError("Embedding service structure not as expected")
                else:
                    raise RuntimeError("Embedding service not properly initialized")
            except (RuntimeError, AttributeError) as e:
                raise EmbeddingServiceNotInitializedError(
                    "Embedding service is not initialized. Cannot determine embedding dimension."
                ) from e
        return self._embed_dim

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a single query.

        Args:
            query: Query text

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # Check if there's already a running event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, we need to use run_coroutine_threadsafe
                future = asyncio.run_coroutine_threadsafe(embedding_service.embed_single(query), loop)
                embedding = future.result()
            except RuntimeError:
                # No running loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    embedding = loop.run_until_complete(embedding_service.embed_single(query))
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)

            return cast(list[float], embedding.tolist())
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            raise EmbeddingError(f"Failed to generate embedding for query: {e}") from e

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a single query asynchronously.

        Args:
            query: Query text

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            embedding = await embedding_service.embed_single(query)
            return cast(list[float], embedding.tolist())
        except Exception as e:
            logger.error(f"Error getting async query embedding: {e}")
            raise EmbeddingError(f"Failed to generate embedding for query: {e}") from e

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        return self._get_query_embedding(text)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text asynchronously.

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        return await self._aget_query_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # Check if there's already a running event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, we need to use run_coroutine_threadsafe
                future = asyncio.run_coroutine_threadsafe(embedding_service.embed_texts(texts), loop)
                embeddings = future.result()
            except RuntimeError:
                # No running loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    embeddings = loop.run_until_complete(embedding_service.embed_texts(texts))
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)

            return cast(list[list[float]], embeddings.tolist())
        except Exception as e:
            logger.error(f"Error getting text embeddings: {e}")
            raise EmbeddingError(f"Failed to generate embeddings for texts: {e}") from e

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts asynchronously.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            embeddings = await embedding_service.embed_texts(texts)
            return cast(list[list[float]], embeddings.tolist())
        except Exception as e:
            logger.error(f"Error getting async text embeddings: {e}")
            raise EmbeddingError(f"Failed to generate embeddings for texts: {e}") from e
