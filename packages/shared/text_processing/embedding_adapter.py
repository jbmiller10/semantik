"""Adapter to use local embedding service with LlamaIndex.

This module provides an adapter that bridges between the local DenseEmbeddingService
and LlamaIndex's BaseEmbedding interface, enabling use of local embeddings in
LlamaIndex-based chunking strategies.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from llama_index.core.embeddings import BaseEmbedding

from packages.shared.embedding.dense import embedding_service

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
            RuntimeError: If embedding service is not initialized
        """
        if self._embed_dim is None:
            # Try to get from the global embedding service
            try:
                self._embed_dim = embedding_service.get_dimension()
            except RuntimeError:
                # Default to common dimension if service not initialized
                logger.warning("Embedding service not initialized, using default dimension 384")
                self._embed_dim = 384
        return self._embed_dim

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a single query.

        Args:
            query: Query text

        Returns:
            Embedding vector as list of floats
        """
        try:
            # Run async method in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                embedding = loop.run_until_complete(
                    embedding_service.embed_single(query)
                )
                return embedding.tolist()
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            # Return random embedding as fallback
            return np.random.randn(self.embed_dim).tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a single query asynchronously.

        Args:
            query: Query text

        Returns:
            Embedding vector as list of floats
        """
        try:
            embedding = await embedding_service.embed_single(query)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error getting async query embedding: {e}")
            # Return random embedding as fallback
            return np.random.randn(self.embed_dim).tolist()

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
        """
        try:
            # Run async method in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                embeddings = loop.run_until_complete(
                    embedding_service.embed_texts(texts)
                )
                return embeddings.tolist()
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error getting text embeddings: {e}")
            # Return random embeddings as fallback
            return [np.random.randn(self.embed_dim).tolist() for _ in texts]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts asynchronously.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = await embedding_service.embed_texts(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error getting async text embeddings: {e}")
            # Return random embeddings as fallback
            return [np.random.randn(self.embed_dim).tolist() for _ in texts]

