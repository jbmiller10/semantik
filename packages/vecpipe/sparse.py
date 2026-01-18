"""Sparse vector operations for Qdrant.

This module handles all sparse vector persistence to Qdrant.
Plugins generate sparse vectors; this module handles storage and retrieval.
"""

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    CollectionInfo,
    NamedSparseVector,
    PointIdsList,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
)

logger = logging.getLogger(__name__)


async def ensure_sparse_collection(
    sparse_collection_name: str,
    qdrant_client: AsyncQdrantClient,
) -> bool:
    """Create sparse Qdrant collection if it doesn't exist.

    Args:
        sparse_collection_name: Name for the sparse collection
        qdrant_client: Qdrant async client

    Returns:
        True if collection was created, False if it already existed
    """
    try:
        # Check if collection exists
        collections = await qdrant_client.get_collections()
        existing_names = {c.name for c in collections.collections}

        if sparse_collection_name in existing_names:
            logger.debug("Sparse collection '%s' already exists", sparse_collection_name)
            return False

        # Create collection with sparse vector configuration
        await qdrant_client.create_collection(
            collection_name=sparse_collection_name,
            vectors_config={},  # No dense vectors
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,  # Keep in memory for fast search
                    ),
                )
            },
        )

        logger.info("Created sparse collection '%s'", sparse_collection_name)
        return True

    except Exception as e:
        logger.error("Failed to create sparse collection '%s': %s", sparse_collection_name, e)
        raise


async def upsert_sparse_vectors(
    sparse_collection_name: str,
    vectors: list[dict[str, Any]],
    qdrant_client: AsyncQdrantClient,
) -> int:
    """Upsert sparse vectors to Qdrant.

    Args:
        sparse_collection_name: Target sparse collection
        vectors: List of SparseVectorDict with keys:
            - chunk_id (str): Point ID (aligns with dense vectors)
            - indices (list[int]): Sparse vector indices
            - values (list[float]): Sparse vector values
            - metadata (dict, optional): Additional metadata
        qdrant_client: Qdrant async client

    Returns:
        Number of points upserted
    """
    if not vectors:
        return 0

    points = []
    for vec in vectors:
        chunk_id = vec["chunk_id"]
        sparse_vec = SparseVector(
            indices=vec["indices"],
            values=vec["values"],
        )
        payload = vec.get("metadata", {})
        # Store chunk_id in payload for retrieval
        payload["chunk_id"] = chunk_id

        points.append(
            PointStruct(
                id=chunk_id,
                vector={"sparse": sparse_vec},
                payload=payload,
            )
        )

    await qdrant_client.upsert(
        collection_name=sparse_collection_name,
        points=points,
        wait=True,
    )

    logger.debug("Upserted %d sparse vectors to '%s'", len(points), sparse_collection_name)
    return len(points)


async def search_sparse_collection(
    sparse_collection_name: str,
    query_indices: list[int],
    query_values: list[float],
    limit: int,
    qdrant_client: AsyncQdrantClient,
) -> list[dict[str, Any]]:
    """Search sparse collection with sparse query vector.

    Args:
        sparse_collection_name: Sparse collection to search
        query_indices: Sparse query vector indices
        query_values: Sparse query vector values
        limit: Maximum results to return
        qdrant_client: Qdrant async client

    Returns:
        List of results with chunk_id and score
    """
    query_vector = NamedSparseVector(
        name="sparse",
        vector=SparseVector(indices=query_indices, values=query_values),
    )

    results = await qdrant_client.search(
        collection_name=sparse_collection_name,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
    )

    return [
        {
            "chunk_id": str(point.id),
            "score": point.score,
            "payload": point.payload,
        }
        for point in results
    ]


async def delete_sparse_vectors(
    sparse_collection_name: str,
    chunk_ids: list[str],
    qdrant_client: AsyncQdrantClient,
) -> int:
    """Delete sparse vectors by chunk_id.

    Args:
        sparse_collection_name: Sparse collection to delete from
        chunk_ids: List of chunk IDs to delete
        qdrant_client: Qdrant async client

    Returns:
        Number of points deleted
    """
    if not chunk_ids:
        return 0

    await qdrant_client.delete(
        collection_name=sparse_collection_name,
        points_selector=PointIdsList(points=chunk_ids),
        wait=True,
    )

    logger.debug("Deleted %d sparse vectors from '%s'", len(chunk_ids), sparse_collection_name)
    return len(chunk_ids)


async def delete_sparse_collection(
    sparse_collection_name: str,
    qdrant_client: AsyncQdrantClient,
) -> bool:
    """Delete entire sparse collection.

    Args:
        sparse_collection_name: Sparse collection to delete
        qdrant_client: Qdrant async client

    Returns:
        True if collection was deleted, False if it didn't exist
    """
    try:
        # Check if collection exists first
        collections = await qdrant_client.get_collections()
        existing_names = {c.name for c in collections.collections}

        if sparse_collection_name not in existing_names:
            logger.debug("Sparse collection '%s' does not exist, nothing to delete", sparse_collection_name)
            return False

        await qdrant_client.delete_collection(collection_name=sparse_collection_name)
        logger.info("Deleted sparse collection '%s'", sparse_collection_name)
        return True

    except Exception as e:
        logger.error("Failed to delete sparse collection '%s': %s", sparse_collection_name, e)
        raise


async def get_sparse_collection_info(
    sparse_collection_name: str,
    qdrant_client: AsyncQdrantClient,
) -> CollectionInfo | None:
    """Get info about a sparse collection.

    Args:
        sparse_collection_name: Sparse collection to query
        qdrant_client: Qdrant async client

    Returns:
        CollectionInfo if collection exists, None otherwise
    """
    try:
        collections = await qdrant_client.get_collections()
        existing_names = {c.name for c in collections.collections}

        if sparse_collection_name not in existing_names:
            return None

        return await qdrant_client.get_collection(collection_name=sparse_collection_name)
    except Exception as e:
        logger.error("Failed to get sparse collection info for '%s': %s", sparse_collection_name, e)
        return None


def generate_sparse_collection_name(base_collection_name: str, sparse_type: str) -> str:
    """Generate sparse collection name from base collection and sparse type.

    Args:
        base_collection_name: Name of the dense collection
        sparse_type: Type of sparse indexer ('bm25' or 'splade')

    Returns:
        Sparse collection name (e.g., 'work_docs_sparse_bm25')
    """
    return f"{base_collection_name}_sparse_{sparse_type}"
