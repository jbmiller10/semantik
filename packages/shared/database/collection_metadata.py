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
