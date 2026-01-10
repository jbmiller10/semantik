"""Celery tasks for sparse indexing operations.

This module provides tasks for:
- Reindexing collections with sparse vectors (BM25/SPLADE)
- Background sparse index management

Progress is tracked via Celery's built-in task state mechanism.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from qdrant_client import AsyncQdrantClient

from shared.config import settings
from shared.database.collection_metadata import (
    get_sparse_index_config,
    store_sparse_index_config,
)
from shared.database.database import AsyncSessionLocal
from shared.database.repositories.chunk_repository import ChunkRepository
from shared.database.repositories.collection_repository import CollectionRepository
from shared.plugins.loader import load_plugins
from shared.plugins.registry import plugin_registry
from vecpipe.sparse import ensure_sparse_collection, upsert_sparse_vectors
from webui.celery_app import celery_app

if TYPE_CHECKING:
    from celery import Task

logger = logging.getLogger(__name__)

# Batch size for processing chunks
SPARSE_REINDEX_BATCH_SIZE = 100


def _load_sparse_indexer_plugin(plugin_id: str) -> Any:
    """Load and instantiate a sparse indexer plugin.

    Args:
        plugin_id: The plugin ID to load.

    Returns:
        Instantiated sparse indexer plugin.

    Raises:
        ValueError: If plugin not found or not a sparse indexer.
    """
    # Ensure sparse_indexer plugins are loaded
    load_plugins(plugin_types={"sparse_indexer"})

    # Get the plugin class from registry
    record = plugin_registry.find_by_id(plugin_id)
    if record is None:
        raise ValueError(f"Sparse indexer plugin '{plugin_id}' not found")

    if record.plugin_type != "sparse_indexer":
        raise ValueError(f"Plugin '{plugin_id}' is not a sparse indexer (type: {record.plugin_type})")

    # Instantiate the plugin
    plugin_cls = record.plugin_class
    return plugin_cls()


async def _reindex_collection_async(
    task: Task,
    collection_uuid: str,
    plugin_id: str,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Async implementation of sparse reindexing.

    Args:
        task: Celery task instance for progress updates.
        collection_uuid: UUID of the collection to reindex.
        plugin_id: Sparse indexer plugin ID.
        model_config: Plugin-specific configuration.

    Returns:
        Dict with reindex results.
    """
    logger.info(
        "Starting sparse reindex for collection %s with plugin %s",
        collection_uuid,
        plugin_id,
    )

    # Load the sparse indexer plugin
    try:
        indexer = _load_sparse_indexer_plugin(plugin_id)
    except ValueError as e:
        logger.error("Failed to load sparse indexer: %s", e)
        raise

    # Initialize plugin with config if it has an initialize method
    if hasattr(indexer, "initialize"):
        await indexer.initialize(model_config)

    # Determine batch size from plugin capabilities or use default
    batch_size = SPARSE_REINDEX_BATCH_SIZE
    if hasattr(indexer, "get_capabilities"):
        try:
            capabilities = indexer.get_capabilities()
            if hasattr(capabilities, "max_batch_size") and capabilities.max_batch_size:
                batch_size = min(batch_size, capabilities.max_batch_size)
        except Exception:
            pass  # Use default if capabilities unavailable

    # Initialize qdrant client reference for cleanup
    async_qdrant: AsyncQdrantClient | None = None

    try:
        # Get collection info
        async with AsyncSessionLocal() as session:
            collection_repo = CollectionRepository(session)
            collection = await collection_repo.get_by_uuid(collection_uuid)
            if collection is None:
                raise ValueError(f"Collection '{collection_uuid}' not found")

            vector_store_name = collection.vector_store_name

            # Get total chunks count using paginated query
            chunk_repo = ChunkRepository(session)
            _, total_chunks = await chunk_repo.get_chunks_paginated(
                collection_id=collection_uuid, page=1, page_size=1
            )

        if total_chunks == 0:
            logger.info("Collection %s has no chunks to reindex", collection_uuid)
            return {
                "status": "completed",
                "documents_processed": 0,
                "total_documents": 0,
            }

        # Create Qdrant client
        async_qdrant = AsyncQdrantClient(
            url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
            api_key=settings.QDRANT_API_KEY,
        )

        # Generate sparse collection name
        sparse_collection_name = indexer.get_sparse_collection_name(vector_store_name)

        # Ensure sparse collection exists
        await ensure_sparse_collection(sparse_collection_name, async_qdrant)

        # Process chunks in batches
        processed = 0
        offset = 0

        while offset < total_chunks:
            # Fetch batch of chunks
            async with AsyncSessionLocal() as session:
                chunk_repo = ChunkRepository(session)
                chunks = await chunk_repo.get_chunks_by_collection(
                    collection_id=collection_uuid,
                    offset=offset,
                    limit=batch_size,
                )

            if not chunks:
                break

            # Prepare documents for encoding
            documents = [
                {
                    "content": chunk.content,
                    "chunk_id": chunk.id,
                    "metadata": chunk.metadata or {},
                }
                for chunk in chunks
            ]

            # Encode documents
            sparse_vectors = await indexer.encode_documents(documents)

            # Convert to Qdrant format and upsert
            qdrant_vectors = [
                {
                    "chunk_id": sv.chunk_id,
                    "indices": list(sv.indices),
                    "values": list(sv.values),
                }
                for sv in sparse_vectors
            ]

            await upsert_sparse_vectors(
                sparse_collection_name=sparse_collection_name,
                vectors=qdrant_vectors,
                qdrant_client=async_qdrant,
            )

            processed += len(chunks)
            offset += batch_size

            # Update progress
            progress = (processed / total_chunks) * 100
            task.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "documents_processed": processed,
                    "total_documents": total_chunks,
                },
            )

            logger.debug(
                "Sparse reindex progress: %d/%d (%.1f%%)",
                processed,
                total_chunks,
                progress,
            )

        # Update sparse config with completion info
        existing_config = await get_sparse_index_config(async_qdrant, vector_store_name)
        if existing_config:
            existing_config["document_count"] = processed
            existing_config["last_indexed_at"] = datetime.now(UTC).isoformat()
            await store_sparse_index_config(async_qdrant, vector_store_name, existing_config)

        logger.info(
            "Completed sparse reindex for collection %s: %d chunks",
            collection_uuid,
            processed,
        )

        return {
            "status": "completed",
            "documents_processed": processed,
            "total_documents": total_chunks,
        }

    finally:
        # Close Qdrant client if it was created
        if async_qdrant is not None:
            await async_qdrant.close()

        # Cleanup plugin if it has a cleanup method
        if hasattr(indexer, "cleanup"):
            await indexer.cleanup()


@celery_app.task(
    bind=True,
    name="sparse.reindex_collection",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    soft_time_limit=3600,  # 1 hour soft limit
    time_limit=3660,  # 1 hour + 1 minute hard limit
)
def reindex_sparse_collection(
    self: Task,
    collection_uuid: str,
    plugin_id: str,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Celery task to reindex a collection with sparse vectors.

    This task loads all chunks from the collection, encodes them using
    the specified sparse indexer plugin (BM25/SPLADE), and upserts the
    sparse vectors to Qdrant.

    Progress is tracked via Celery's built-in task state mechanism:
    - State: "PROGRESS" with meta containing progress %, docs processed, total docs
    - State: "SUCCESS" with result containing final stats
    - State: "FAILURE" with error info

    Args:
        collection_uuid: UUID of the collection to reindex.
        plugin_id: ID of the sparse indexer plugin to use.
        model_config: Optional plugin-specific configuration.

    Returns:
        Dict with status, documents_processed, and total_documents.

    Raises:
        ValueError: If collection or plugin not found.
        Exception: On processing errors (will retry).
    """
    if model_config is None:
        model_config = {}

    try:
        return asyncio.run(
            _reindex_collection_async(
                task=self,
                collection_uuid=collection_uuid,
                plugin_id=plugin_id,
                model_config=model_config,
            )
        )
    except Exception as exc:
        logger.exception(
            "Sparse reindex failed for collection %s: %s", collection_uuid, exc
        )
        # Let Celery handle retries
        raise
