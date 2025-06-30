"""
Collection metadata management for tracking model information
"""

import json
import logging
from typing import Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

logger = logging.getLogger(__name__)

METADATA_COLLECTION = "_collection_metadata"


def ensure_metadata_collection(qdrant: QdrantClient):
    """Ensure the metadata collection exists"""
    try:
        collections = qdrant.get_collections().collections
        if not any(c.name == METADATA_COLLECTION for c in collections):
            qdrant.create_collection(
                collection_name=METADATA_COLLECTION,
                vectors_config=VectorParams(
                    size=4, distance=Distance.COSINE  # Small vector size, we're not using it for search
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
    chunk_size: int = None,
    chunk_overlap: int = None,
    instruction: str = None,
):
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
        # Use collection name as ID for easy lookup
        qdrant.upsert(
            collection_name=METADATA_COLLECTION,
            points=[PointStruct(id=collection_name, vector=[0.0] * 4, payload=metadata)],  # Dummy vector
        )
        logger.info(f"Stored metadata for collection {collection_name}: {metadata}")
    except Exception as e:
        logger.error(f"Failed to store collection metadata: {e}")


def get_collection_metadata(qdrant: QdrantClient, collection_name: str) -> Optional[Dict]:
    """Get metadata for a collection"""
    try:
        result = qdrant.retrieve(collection_name=METADATA_COLLECTION, ids=[collection_name])
        if result:
            return result[0].payload
    except Exception as e:
        logger.warning(f"Failed to get metadata for collection {collection_name}: {e}")
    return None
