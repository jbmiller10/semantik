"""Collection metadata management for tracking model information."""

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

METADATA_COLLECTION = "_collection_metadata"
METADATA_VECTOR_DIM = 4


def ensure_metadata_collection(qdrant: QdrantClient) -> None:
    """Ensure the metadata collection exists"""
    try:
        collections = qdrant.get_collections().collections
        if not any(c.name == METADATA_COLLECTION for c in collections):
            qdrant.create_collection(
                collection_name=METADATA_COLLECTION,
                vectors_config=VectorParams(
                    size=METADATA_VECTOR_DIM, distance=Distance.COSINE  # Small vector size, metadata only
                ),
            )
            logger.info(f"Created metadata collection: {METADATA_COLLECTION}")
    except Exception as e:
        logger.error(f"Failed to ensure metadata collection: {e}")


def store_collection_metadata(
    qdrant: QdrantClient,
    collection_name: str,
    model_name: str,
    quantization: str,
    vector_dim: int,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    instruction: str | None = None,
) -> None:
    """Store metadata about a collection"""
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
        qdrant.upsert(
            collection_name=METADATA_COLLECTION,
            points=[
                PointStruct(
                    id=collection_name,
                    vector=[0.0] * METADATA_VECTOR_DIM,
                    payload={**metadata, "id": collection_name},
                )
            ],  # Store collection name in payload
        )
        logger.info(f"Stored metadata for collection {collection_name}: {metadata}")
    except Exception as e:
        logger.error(f"Failed to store collection metadata: {e}")


def get_collection_metadata(qdrant: QdrantClient, collection_name: str) -> dict[str, Any] | None:
    """Get metadata for a collection"""
    try:
        result = qdrant.retrieve(collection_name=METADATA_COLLECTION, ids=[collection_name])
        if result:
            payload = result[0].payload
            return dict(payload) if payload else {}

        # Fall back to scanning existing entries (supports legacy random point IDs)
        offset = None
        while True:
            points, offset = qdrant.scroll(
                collection_name=METADATA_COLLECTION,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            if not points:
                break
            for point in points:
                payload = point.payload or {}
                name_match = payload.get("collection_name") or payload.get("id")
                if name_match == collection_name:
                    return dict(payload)
            if offset is None:
                break
    except Exception as e:
        logger.warning(f"Failed to get metadata for collection {collection_name}: {e}")
    return None


def restamp_collection_metadata(qdrant: QdrantClient, batch_size: int = 256) -> int:
    """Restamp legacy metadata entries so their point IDs match the collection name.

    Args:
        qdrant: Qdrant client instance.
        batch_size: Number of records to scroll per request.

    Returns:
        Number of metadata records that were restamped.
    """

    ensure_metadata_collection(qdrant)

    migrated = 0
    offset = None
    processed: set[str] = set()

    try:
        while True:
            points, offset = qdrant.scroll(
                collection_name=METADATA_COLLECTION,
                with_payload=True,
                with_vectors=False,
                limit=batch_size,
                offset=offset,
            )
            if not points:
                break

            upsert_points: list[PointStruct] = []
            for point in points:
                payload = point.payload or {}
                desired_id = payload.get("collection_name") or payload.get("id")
                if not desired_id or desired_id in processed:
                    continue
                processed.add(desired_id)

                if str(point.id) == desired_id:
                    continue

                upsert_points.append(
                    PointStruct(
                        id=desired_id,
                        vector=[0.0] * METADATA_VECTOR_DIM,
                        payload={**payload, "collection_name": desired_id, "id": desired_id},
                    )
                )
                migrated += 1

            if upsert_points:
                qdrant.upsert(collection_name=METADATA_COLLECTION, points=upsert_points)

            if offset is None:
                break

        if migrated:
            logger.info(f"Restamped {migrated} metadata entries with deterministic point IDs")
        return migrated
    except Exception as e:  # pragma: no cover - network errors
        logger.error(f"Failed to restamp metadata identifiers: {e}")
        return migrated
