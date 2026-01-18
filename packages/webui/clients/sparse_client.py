"""Client for VecPipe sparse encoding API.

This module provides an HTTP client for encoding documents and queries
to sparse vectors via the VecPipe service. This allows the Celery worker
to offload GPU-based sparse encoding (SPLADE) to VecPipe for centralized
memory management.

Usage:
    client = SparseEncodingClient()
    vectors = await client.encode_documents(
        texts=["hello world"],
        chunk_ids=["chunk-1"],
        plugin_id="splade-local",
    )
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from shared.config import settings
from webui.tasks.utils import _build_internal_api_headers

logger = logging.getLogger(__name__)

# Timeout for sparse encoding requests (large batches may take time)
SPARSE_ENCODE_TIMEOUT_SECONDS = 300.0  # 5 minutes
SPARSE_QUERY_TIMEOUT_SECONDS = 30.0  # 30 seconds for single query


def _get_vecpipe_base_url() -> str:
    """Get the VecPipe service base URL."""
    host = getattr(settings, "SEARCH_API_HOST", "vecpipe")
    port = getattr(settings, "SEARCH_API_PORT", 8001)
    return f"http://{host}:{port}"


class SparseEncodingClient:
    """HTTP client for VecPipe sparse encoding endpoints.

    This client is used by Celery workers to encode documents and queries
    to sparse vectors via VecPipe, which manages SPLADE models with GPU
    memory governance.

    Example:
        client = SparseEncodingClient()

        # Encode documents
        vectors = await client.encode_documents(
            texts=["Document text here"],
            chunk_ids=["chunk-123"],
            plugin_id="splade-local",
            model_config={"batch_size": 32},
        )

        # Encode query
        query_vector = await client.encode_query(
            query="search query",
            plugin_id="splade-local",
        )
    """

    def __init__(self, base_url: str | None = None) -> None:
        """Initialize the sparse encoding client.

        Args:
            base_url: Optional base URL for VecPipe service.
                     Defaults to http://{SEARCH_API_HOST}:{SEARCH_API_PORT}.
        """
        self._base_url = base_url or _get_vecpipe_base_url()

    async def encode_documents(
        self,
        texts: list[str],
        chunk_ids: list[str],
        plugin_id: str,
        model_config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Encode documents to sparse vectors via VecPipe.

        Args:
            texts: Document texts to encode.
            chunk_ids: Chunk IDs corresponding to each text.
            plugin_id: Sparse indexer plugin ID (e.g., "splade-local").
            model_config: Optional plugin-specific configuration.

        Returns:
            List of sparse vector dicts with keys:
                - chunk_id: str
                - indices: list[int]
                - values: list[float]

        Raises:
            httpx.HTTPStatusError: If the request fails.
            httpx.TimeoutException: If the request times out.
        """
        headers = _build_internal_api_headers()

        request_data = {
            "texts": texts,
            "chunk_ids": chunk_ids,
            "plugin_id": plugin_id,
            "model_config_data": model_config,
        }

        async with httpx.AsyncClient(base_url=self._base_url) as client:
            response = await client.post(
                "/sparse/encode",
                json=request_data,
                headers=headers,
                timeout=SPARSE_ENCODE_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()

        logger.debug(
            "Encoded %d documents via VecPipe in %.1fms",
            len(texts),
            data.get("encoding_time_ms", 0),
        )

        vectors: list[dict[str, Any]] = data["vectors"]
        return vectors

    async def encode_query(
        self,
        query: str,
        plugin_id: str,
        model_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Encode a query to sparse vector via VecPipe.

        Args:
            query: Query text to encode.
            plugin_id: Sparse indexer plugin ID.
            model_config: Optional plugin-specific configuration.

        Returns:
            Dict with sparse query vector:
                - indices: list[int]
                - values: list[float]
                - encoding_time_ms: float

        Raises:
            httpx.HTTPStatusError: If the request fails.
            httpx.TimeoutException: If the request times out.
        """
        headers = _build_internal_api_headers()

        request_data = {
            "query": query,
            "plugin_id": plugin_id,
            "model_config_data": model_config,
        }

        async with httpx.AsyncClient(base_url=self._base_url) as client:
            response = await client.post(
                "/sparse/query",
                json=request_data,
                headers=headers,
                timeout=SPARSE_QUERY_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()

        logger.debug(
            "Encoded query via VecPipe in %.1fms (vector size: %d)",
            data.get("encoding_time_ms", 0),
            len(data.get("indices", [])),
        )

        result: dict[str, Any] = data
        return result

    async def list_plugins(self) -> list[dict[str, Any]]:
        """List available sparse indexer plugins from VecPipe.

        Returns:
            List of plugin info dicts with keys:
                - plugin_id: str
                - display_name: str
                - sparse_type: str ("bm25" or "splade")
                - requires_gpu: bool

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        headers = _build_internal_api_headers()

        async with httpx.AsyncClient(base_url=self._base_url) as client:
            response = await client.get(
                "/sparse/plugins",
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

        plugins: list[dict[str, Any]] = data.get("plugins", [])
        return plugins
