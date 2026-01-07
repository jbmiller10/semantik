"""Shared utilities for Qdrant operations in indexing tasks."""

import uuid
from typing import Any

from qdrant_client.models import PointStruct


def build_chunk_point(
    *,
    collection_id: str,
    doc_id: str,
    chunk: dict[str, Any],
    chunk_index: int,
    total_chunks: int,
    path: str,
    embedding: list[float],
) -> PointStruct:
    """Build a PointStruct for a document chunk.

    This is the single source of truth for document chunk payloads.
    All indexing paths (ingestion, parallel_ingestion, reindex) should use this.

    Args:
        collection_id: UUID of the collection
        doc_id: Document identifier
        chunk: Chunk dictionary with chunk_id, text/content, metadata
        chunk_index: 0-based index of this chunk within the document
        total_chunks: Total number of chunks in the document
        path: File path or document identifier
        embedding: Vector embedding for this chunk

    Returns:
        PointStruct ready for Qdrant upsert
    """
    return PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={
            "collection_id": collection_id,
            "doc_id": doc_id,
            "chunk_id": chunk["chunk_id"],
            "path": path,
            "content": chunk.get("text") or chunk.get("content") or "",
            "metadata": chunk.get("metadata", {}),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
        },
    )
