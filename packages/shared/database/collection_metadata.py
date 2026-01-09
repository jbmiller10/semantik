"""
Collection metadata management for tracking model information
"""

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)

METADATA_COLLECTION = "_collection_metadata"


def ensure_metadata_collection(qdrant: QdrantClient) -> None:
    """Ensure the metadata collection exists"""
    try:
        collections = qdrant.get_collections().collections
        if not any(c.name == METADATA_COLLECTION for c in collections):
            qdrant.create_collection(
                collection_name=METADATA_COLLECTION,
                vectors_config=VectorParams(
                    size=4,
                    distance=Distance.COSINE,  # Small vector size, we're not using it for search
                ),
            )
            logger.info(f"Created metadata collection: {METADATA_COLLECTION}")
    except Exception as e:
        logger.error("Failed to ensure metadata collection: %s", e, exc_info=True)


def store_collection_metadata(
    qdrant: QdrantClient,
    collection_name: str,
    model_name: str,
    quantization: str,
    vector_dim: int,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    instruction: str | None = None,
    *,
    ensure: bool = True,
) -> None:
    """Store metadata about a collection.

    The ``ensure`` flag allows callers that have already provisioned the metadata
    collection to skip the additional existence check. This keeps downstream
    mocks from observing an unexpected second ``create_collection`` invocation
    while preserving the safety net for production code paths.
    """

    if ensure:
        ensure_metadata_collection(qdrant)

    metadata = {
        "collection_name": collection_name,
        "model_name": model_name,
        "quantization": quantization,
        "vector_dim": vector_dim,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "instruction": instruction,
    }

    try:
        # Use UUID for point ID to avoid validation errors
        qdrant.upsert(
            collection_name=METADATA_COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),  # Use UUID instead of collection name
                    vector=[0.0] * 4,
                    payload={**metadata, "id": collection_name},
                )
            ],  # Store collection name in payload
        )
        logger.info(f"Stored metadata for collection {collection_name}: {metadata}")
    except Exception as e:
        logger.error("Failed to store collection metadata: %s", e, exc_info=True)


def get_collection_metadata(qdrant: QdrantClient, collection_name: str) -> dict[str, Any] | None:
    """Get metadata for a collection by querying payload field.

    Uses scroll with a filter on the collection_name payload field rather than
    point ID lookup, since metadata points are stored with random UUID IDs.
    """
    try:
        points, _ = qdrant.scroll(
            collection_name=METADATA_COLLECTION,
            limit=1,
            with_payload=True,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="collection_name",
                        match=MatchValue(value=collection_name),
                    )
                ]
            ),
        )
        if points:
            return dict(points[0].payload) if points[0].payload else {}
    except Exception as e:
        logger.warning("Failed to get metadata for collection %s: %s", collection_name, e, exc_info=True)
    return None


async def get_collection_metadata_async(
    qdrant: AsyncQdrantClient,
    collection_name: str,
) -> dict[str, Any] | None:
    """Async version: Get metadata for a collection by querying payload field.

    Uses scroll with a filter on the collection_name payload field rather than
    point ID lookup, since metadata points are stored with random UUID IDs.
    """
    try:
        points, _ = await qdrant.scroll(
            collection_name=METADATA_COLLECTION,
            limit=1,
            with_payload=True,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="collection_name",
                        match=MatchValue(value=collection_name),
                    )
                ]
            ),
        )
        if points:
            return dict(points[0].payload) if points[0].payload else {}
    except Exception as e:
        logger.warning("Failed to get metadata for collection %s: %s", collection_name, e, exc_info=True)
    return None


# Sparse Index Config Functions


async def store_sparse_index_config(
    qdrant: AsyncQdrantClient,
    collection_name: str,
    sparse_config: dict[str, Any],
) -> bool:
    """Store sparse index configuration for a collection.

    Updates the existing metadata point with sparse_index_config field.

    Args:
        qdrant: Async Qdrant client
        collection_name: Name of the collection
        sparse_config: Sparse index configuration with keys:
            - enabled (bool): Whether sparse indexing is enabled
            - plugin_id (str): Sparse indexer plugin ID
            - sparse_collection_name (str): Name of the sparse Qdrant collection
            - model_config (dict): Plugin-specific configuration
            - created_at (str): ISO datetime of creation
            - document_count (int): Number of indexed documents
            - last_indexed_at (str | None): ISO datetime of last indexing

    Returns:
        True if successful, False otherwise
    """
    try:
        # Find the existing metadata point
        points, _ = await qdrant.scroll(
            collection_name=METADATA_COLLECTION,
            limit=1,
            with_payload=True,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="collection_name",
                        match=MatchValue(value=collection_name),
                    )
                ]
            ),
        )

        if not points:
            logger.warning(
                "No metadata found for collection '%s', cannot store sparse config",
                collection_name,
            )
            return False

        # Get existing point and update it
        point = points[0]
        existing_payload = dict(point.payload) if point.payload else {}
        existing_payload["sparse_index_config"] = sparse_config

        # Upsert with the same point ID
        await qdrant.upsert(
            collection_name=METADATA_COLLECTION,
            points=[
                PointStruct(
                    id=point.id,
                    vector=[0.0] * 4,
                    payload=existing_payload,
                )
            ],
        )
        logger.info("Stored sparse index config for collection '%s'", collection_name)
        return True

    except Exception as e:
        logger.error(
            "Failed to store sparse index config for collection '%s': %s",
            collection_name,
            e,
            exc_info=True,
        )
        return False


async def get_sparse_index_config(
    qdrant: AsyncQdrantClient,
    collection_name: str,
) -> dict[str, Any] | None:
    """Get sparse index configuration for a collection.

    Args:
        qdrant: Async Qdrant client
        collection_name: Name of the collection

    Returns:
        Sparse index config dict if exists, None otherwise
    """
    metadata = await get_collection_metadata_async(qdrant, collection_name)
    if metadata:
        return metadata.get("sparse_index_config")
    return None


async def delete_sparse_index_config(
    qdrant: AsyncQdrantClient,
    collection_name: str,
) -> bool:
    """Delete sparse index configuration for a collection.

    Removes the sparse_index_config field from the metadata.

    Args:
        qdrant: Async Qdrant client
        collection_name: Name of the collection

    Returns:
        True if successful, False otherwise
    """
    try:
        # Find the existing metadata point
        points, _ = await qdrant.scroll(
            collection_name=METADATA_COLLECTION,
            limit=1,
            with_payload=True,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="collection_name",
                        match=MatchValue(value=collection_name),
                    )
                ]
            ),
        )

        if not points:
            logger.warning(
                "No metadata found for collection '%s', cannot delete sparse config",
                collection_name,
            )
            return False

        # Get existing point and remove sparse_index_config
        point = points[0]
        existing_payload = dict(point.payload) if point.payload else {}

        if "sparse_index_config" not in existing_payload:
            logger.debug("No sparse config to delete for collection '%s'", collection_name)
            return True  # Nothing to delete, consider it success

        del existing_payload["sparse_index_config"]

        # Upsert with the same point ID
        await qdrant.upsert(
            collection_name=METADATA_COLLECTION,
            points=[
                PointStruct(
                    id=point.id,
                    vector=[0.0] * 4,
                    payload=existing_payload,
                )
            ],
        )
        logger.info("Deleted sparse index config for collection '%s'", collection_name)
        return True

    except Exception as e:
        logger.error(
            "Failed to delete sparse index config for collection '%s': %s",
            collection_name,
            e,
            exc_info=True,
        )
        return False


async def update_sparse_index_stats(
    qdrant: AsyncQdrantClient,
    collection_name: str,
    document_count: int,
    last_indexed_at: str,
) -> bool:
    """Update sparse index statistics (document count and last indexed time).

    Args:
        qdrant: Async Qdrant client
        collection_name: Name of the collection
        document_count: Updated document/chunk count
        last_indexed_at: ISO datetime of last indexing

    Returns:
        True if successful, False otherwise
    """
    sparse_config = await get_sparse_index_config(qdrant, collection_name)
    if not sparse_config:
        logger.warning(
            "No sparse config found for collection '%s', cannot update stats",
            collection_name,
        )
        return False

    sparse_config["document_count"] = document_count
    sparse_config["last_indexed_at"] = last_indexed_at

    return await store_sparse_index_config(qdrant, collection_name, sparse_config)
