#!/usr/bin/env python3
"""
Repository for chunk operations with partition awareness.

This repository handles all database operations for the chunks table,
ensuring efficient use of PostgreSQL partitioning by collection_id.
"""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.models import Chunk
from shared.database.partition_utils import ChunkPartitionHelper, PartitionAwareMixin, PartitionValidation

logger = logging.getLogger(__name__)


class ChunkRepository(PartitionAwareMixin):
    """Repository for chunk operations with partition optimization.

    All operations ensure the partition key (collection_id) is included
    in queries for optimal performance through partition pruning.
    """

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.

        Args:
            session: AsyncSession for database operations
        """
        self.session = session

    async def create_chunk(self, chunk_data: dict[str, Any]) -> Chunk:
        """Create a single chunk.

        Args:
            chunk_data: Dictionary with chunk fields

        Returns:
            Created chunk instance

        Raises:
            ValueError: If collection_id is missing or data is invalid
            TypeError: If data types are incorrect
        """
        # Comprehensive validation
        chunk_data = PartitionValidation.validate_chunk_data(chunk_data)

        # Remove id from chunk_data if present - let database generate it
        # The id is auto-generated via BIGSERIAL sequence
        if "id" in chunk_data:
            del chunk_data["id"]

        # Ensure partition key is present (tests may not have triggers/generation)
        if chunk_data.get("partition_key") is None:
            partition_key = await self.compute_partition_key(self.session, chunk_data["collection_id"])
            chunk_data["partition_key"] = partition_key

        chunk = Chunk(**chunk_data)
        self.session.add(chunk)
        await self.session.flush()

        logger.debug(
            f"Created chunk {chunk.id} for collection {chunk.collection_id} in partition {chunk.partition_key}"
        )
        return chunk

    async def create_chunks_bulk(self, chunks_data: list[dict[str, Any]]) -> int:
        """Bulk create chunks with partition-aware optimization.

        Groups chunks by collection_id before insertion for efficiency.

        Args:
            chunks_data: List of chunk data dictionaries

        Returns:
            Number of chunks created

        Raises:
            ValueError: If any chunk is missing collection_id
        """
        if not chunks_data:
            return 0

        partition_cache: dict[str, int] = {}
        # Remove id from chunks - let database handle them
        for chunk_data in chunks_data:
            if "id" in chunk_data:
                del chunk_data["id"]

            if chunk_data.get("partition_key") is None:
                collection_id = chunk_data.get("collection_id")
                if not collection_id:
                    raise ValueError("collection_id is required for chunks")
                if collection_id not in partition_cache:
                    partition_cache[collection_id] = await self.compute_partition_key(self.session, collection_id)
                chunk_data["partition_key"] = partition_cache[collection_id]

        # Use partition-aware bulk insert
        await self.bulk_insert_partitioned(self.session, Chunk, chunks_data, partition_key_field="collection_id")

        logger.info(f"Bulk created {len(chunks_data)} chunks")
        return len(chunks_data)

    async def get_chunk_by_id(
        self, chunk_id: int, collection_id: str, partition_key: int | None = None
    ) -> Chunk | None:
        """Get a chunk by ID with partition pruning.

        IMPORTANT: collection_id is required for partition pruning.
        partition_key can be computed if not provided.

        Args:
            chunk_id: Chunk ID (integer from database sequence)
            collection_id: Collection ID (used to compute partition_key if not provided)
            partition_key: Optional partition key (0-99). If not provided, will be computed from collection_id

        Returns:
            Chunk instance or None if not found

        Raises:
            ValueError: If IDs are invalid
            TypeError: If IDs have wrong types
        """
        # Validate IDs
        if not isinstance(chunk_id, int):
            raise TypeError(f"chunk_id must be an integer, got {type(chunk_id).__name__}")
        if chunk_id < 0:
            raise ValueError("chunk_id must be non-negative")

        collection_id = PartitionValidation.validate_partition_key(collection_id, "collection_id")

        # Compute partition_key if not provided
        if partition_key is None:
            # This mimics the database trigger: abs(hashtext(collection_id)) % 100
            # We can't exactly replicate PostgreSQL's hashtext in Python, but we can query with collection_id
            query = select(Chunk).where(and_(Chunk.id == chunk_id, Chunk.collection_id == collection_id))
        else:
            # If partition_key is provided, use all three parts of the composite key
            if not isinstance(partition_key, int) or partition_key < 0 or partition_key > 99:
                raise ValueError("partition_key must be an integer between 0 and 99")
            query = select(Chunk).where(
                and_(Chunk.id == chunk_id, Chunk.collection_id == collection_id, Chunk.partition_key == partition_key)
            )

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_chunks_by_document(
        self, document_id: str, collection_id: str, limit: int | None = None, offset: int = 0
    ) -> list[Chunk]:
        """Get chunks for a document with partition pruning.

        Args:
            document_id: Document ID
            collection_id: Collection ID (partition key)
            limit: Maximum chunks to return
            offset: Number of chunks to skip

        Returns:
            List of chunks ordered by chunk_index

        Raises:
            ValueError: If IDs are invalid or limit/offset are negative
            TypeError: If parameters have wrong types
        """
        # Validate IDs
        document_id = PartitionValidation.validate_uuid(document_id, "document_id")
        collection_id = PartitionValidation.validate_partition_key(collection_id, "collection_id")

        # Validate pagination parameters
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        query = ChunkPartitionHelper.create_chunk_query_with_partition(
            collection_id, [Chunk.document_id == document_id]
        ).order_by(Chunk.chunk_index)

        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_chunks_by_collection(
        self, collection_id: str, limit: int | None = None, offset: int = 0, created_after: datetime | None = None
    ) -> list[Chunk]:
        """Get all chunks for a collection from its partition.

        Args:
            collection_id: Collection ID (partition key)
            limit: Maximum chunks to return
            offset: Number of chunks to skip
            created_after: Only return chunks created after this time

        Returns:
            List of chunks
        """
        filters = []
        if created_after:
            filters.append(Chunk.created_at > created_after)

        query = ChunkPartitionHelper.create_chunk_query_with_partition(collection_id, filters).order_by(
            Chunk.created_at
        )

        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update_chunk_embeddings(self, chunk_updates: list[dict[str, str]]) -> int:
        """Update embedding vector IDs for chunks.

        Groups updates by collection_id for efficient partition access.

        Args:
            chunk_updates: List of dicts with 'id', 'collection_id',
                         and 'embedding_vector_id'

        Returns:
            Number of chunks updated

        Raises:
            ValueError: If update data is invalid
            TypeError: If update data has wrong types
        """
        if not chunk_updates:
            return 0

        # Validate batch size
        PartitionValidation.validate_batch_size(chunk_updates, "chunk embedding updates")

        # Validate each update
        validated_updates: list[dict[str, Any]] = []
        for chunk_update in chunk_updates:
            if not isinstance(chunk_update, dict):
                raise TypeError("Each chunk update must be a dictionary")

            # Validate required fields
            if not all(key in chunk_update for key in ["id", "collection_id", "embedding_vector_id"]):
                raise ValueError("Each update must have 'id', 'collection_id', and 'embedding_vector_id'")

            # Validate chunk id as integer
            if not isinstance(chunk_update["id"], int):
                raise TypeError(f"chunk id must be an integer, got {type(chunk_update['id']).__name__}")

            validated_update = {
                "id": chunk_update["id"],
                "collection_id": PartitionValidation.validate_partition_key(chunk_update["collection_id"]),
                "embedding_vector_id": PartitionValidation.validate_uuid(
                    chunk_update["embedding_vector_id"], "embedding_vector_id"
                ),
            }
            validated_updates.append(validated_update)

        # Group by collection for partition efficiency
        updates_by_collection = self.group_by_partition_key(validated_updates, lambda u: u["collection_id"])

        total_updated = 0
        for collection_id, updates in updates_by_collection.items():
            vector_map = {u["id"]: u["embedding_vector_id"] for u in updates}

            # Update in batches per partition
            for chunk_id, vector_id in vector_map.items():
                stmt = (
                    update(Chunk)
                    .where(and_(Chunk.collection_id == collection_id, Chunk.id == chunk_id))
                    .values(embedding_vector_id=vector_id)
                )
                result = await self.session.execute(stmt)
                total_updated += result.rowcount or 0

        logger.info(f"Updated embeddings for {total_updated} chunks")
        return total_updated

    async def delete_chunks_by_document(self, document_id: str, collection_id: str) -> int:
        """Delete all chunks for a document with partition pruning.

        Args:
            document_id: Document ID
            collection_id: Collection ID (partition key)

        Returns:
            Number of chunks deleted
        """
        stmt = delete(Chunk).where(and_(Chunk.collection_id == collection_id, Chunk.document_id == document_id))

        result = await self.session.execute(stmt)
        deleted_count = result.rowcount or 0

        logger.info(f"Deleted {deleted_count} chunks for document {document_id} in collection {collection_id}")
        return deleted_count

    async def delete_chunks_by_collection(self, collection_id: str) -> int:
        """Delete all chunks for a collection from its partition.

        This is efficient as it only touches one partition.

        Args:
            collection_id: Collection ID

        Returns:
            Number of chunks deleted
        """
        stmt = delete(Chunk).where(Chunk.collection_id == collection_id)

        result = await self.session.execute(stmt)
        deleted_count = result.rowcount or 0

        logger.info(f"Deleted {deleted_count} chunks for collection {collection_id}")
        return deleted_count

    async def get_chunk_statistics(self, collection_id: str) -> dict[str, Any]:
        """Get statistics for chunks in a collection partition.

        Args:
            collection_id: Collection ID

        Returns:
            Dictionary with statistics
        """
        return await ChunkPartitionHelper.get_partition_statistics(self.session, collection_id)

    async def count_chunks_by_document(self, document_id: str, collection_id: str) -> int:
        """Count chunks for a document with partition pruning.

        Args:
            document_id: Document ID
            collection_id: Collection ID (partition key)

        Returns:
            Number of chunks
        """
        query = (
            select(func.count())
            .select_from(Chunk)
            .where(and_(Chunk.collection_id == collection_id, Chunk.document_id == document_id))
        )

        result = await self.session.execute(query)
        return result.scalar() or 0

    async def get_chunks_without_embeddings(self, collection_id: str, limit: int = 1000) -> list[Chunk]:
        """Get chunks that need embedding generation.

        Args:
            collection_id: Collection ID (partition key)
            limit: Maximum chunks to return

        Returns:
            List of chunks without embeddings
        """
        query = ChunkPartitionHelper.create_chunk_query_with_partition(
            collection_id, [Chunk.embedding_vector_id.is_(None)]
        ).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_chunk_by_embedding_vector_id(self, embedding_vector_id: str, collection_id: str) -> Chunk | None:
        """Get a chunk by its embedding_vector_id with partition pruning.

        This maps back from a vector-store point identifier (for example,
        a Qdrant point ID) to the associated chunk row.

        Args:
            embedding_vector_id: Vector-store point identifier (UUID v4).
            collection_id: Collection ID (partition key).

        Returns:
            Chunk instance or None if not found.

        Raises:
            ValueError: If identifiers are invalid.
            TypeError: If identifiers have wrong types.
        """
        # Validate identifiers
        embedding_vector_id = PartitionValidation.validate_uuid(embedding_vector_id, "embedding_vector_id")
        collection_id = PartitionValidation.validate_partition_key(collection_id, "collection_id")

        query = select(Chunk).where(
            and_(Chunk.collection_id == collection_id, Chunk.embedding_vector_id == embedding_vector_id)
        )

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def chunk_exists(self, document_id: str, collection_id: str, chunk_index: int) -> bool:
        """Check if a specific chunk exists.

        Args:
            document_id: Document ID
            collection_id: Collection ID (partition key)
            chunk_index: Chunk index

        Returns:
            True if chunk exists
        """
        query = (
            select(func.count())
            .select_from(Chunk)
            .where(
                and_(
                    Chunk.collection_id == collection_id,
                    Chunk.document_id == document_id,
                    Chunk.chunk_index == chunk_index,
                )
            )
        )

        result = await self.session.execute(query)
        count = result.scalar() or 0
        return count > 0

    async def get_chunks_batch(self, collection_id: str, document_ids: list[str], limit: int = 1000) -> list[Chunk]:
        """Batch fetch chunks for multiple documents.

        Uses IN clause for efficient batch fetching.

        Args:
            collection_id: Collection ID (partition key)
            document_ids: List of document IDs to fetch chunks for
            limit: Maximum chunks to return

        Returns:
            List of chunks ordered by document_id and chunk_index
        """
        if not document_ids:
            return []

        # Validate inputs
        collection_id = PartitionValidation.validate_partition_key(collection_id, "collection_id")
        validated_doc_ids = [
            PartitionValidation.validate_uuid(doc_id, f"document_id[{i}]") for i, doc_id in enumerate(document_ids)
        ]

        # Use IN clause for batch fetching
        query = (
            select(Chunk)
            .where(and_(Chunk.collection_id == collection_id, Chunk.document_id.in_(validated_doc_ids)))
            .order_by(Chunk.document_id, Chunk.chunk_index)
            .limit(limit)
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_chunks_paginated(
        self, collection_id: str, page: int = 1, page_size: int = 100
    ) -> tuple[list[Chunk], int]:
        """Get paginated chunks with total count.

        Uses window function for efficient pagination.

        Args:
            collection_id: Collection ID (partition key)
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            Tuple of (chunks, total_count)
        """
        if page < 1:
            raise ValueError("page must be >= 1")
        if page_size < 1:
            raise ValueError("page_size must be >= 1")

        collection_id = PartitionValidation.validate_partition_key(collection_id, "collection_id")

        # Use window function for efficient pagination with count
        query = (
            select(Chunk, func.count(Chunk.id).over().label("total_count"))
            .where(Chunk.collection_id == collection_id)
            .order_by(Chunk.created_at.desc())
            .limit(page_size)
            .offset((page - 1) * page_size)
        )

        result = await self.session.execute(query)
        rows = result.all()

        if not rows:
            return [], 0

        chunks = [row[0] for row in rows]
        total_count = rows[0][1] if rows else 0

        return chunks, total_count

    async def get_chunk_statistics_optimized(self, collection_id: str) -> dict[str, Any]:
        """Get optimized statistics for chunks in a collection.

        Uses aggregation functions for efficient statistics calculation.

        Args:
            collection_id: Collection ID

        Returns:
            Dictionary with detailed statistics
        """
        collection_id = PartitionValidation.validate_partition_key(collection_id, "collection_id")

        # Single aggregation query for all statistics
        stats_query = select(
            func.count(Chunk.id).label("total_chunks"),
            func.avg(func.length(Chunk.content)).label("avg_chunk_size"),
            func.min(func.length(Chunk.content)).label("min_chunk_size"),
            func.max(func.length(Chunk.content)).label("max_chunk_size"),
            func.count(func.distinct(Chunk.document_id)).label("unique_documents"),
            func.min(Chunk.created_at).label("first_chunk_created"),
            func.max(Chunk.created_at).label("last_chunk_created"),
        ).where(Chunk.collection_id == collection_id)

        result = await self.session.execute(stats_query)
        stats = result.one()

        return {
            "total_chunks": stats.total_chunks or 0,
            "avg_chunk_size": float(stats.avg_chunk_size or 0),
            "min_chunk_size": stats.min_chunk_size or 0,
            "max_chunk_size": stats.max_chunk_size or 0,
            "unique_documents": stats.unique_documents or 0,
            "first_chunk_created": stats.first_chunk_created.isoformat() if stats.first_chunk_created else None,
            "last_chunk_created": stats.last_chunk_created.isoformat() if stats.last_chunk_created else None,
        }

    async def update_chunks_batch(self, updates: list[dict[str, Any]]) -> int:
        """Batch update chunks with various fields.

        Groups updates by collection for partition efficiency.

        Args:
            updates: List of dicts with 'id', 'collection_id', and fields to update

        Returns:
            Number of chunks updated
        """
        if not updates:
            return 0

        # Validate batch size
        PartitionValidation.validate_batch_size(updates, "chunk updates")

        # Group by collection for partition efficiency
        updates_by_collection = self.group_by_partition_key(updates, lambda u: u["collection_id"])

        total_updated = 0
        for collection_id, collection_updates in updates_by_collection.items():
            for update_data in collection_updates:
                chunk_id = update_data.pop("id")
                update_data.pop("collection_id")  # Remove from update fields

                if update_data:  # Only update if there are fields to update
                    stmt = (
                        update(Chunk)
                        .where(and_(Chunk.collection_id == collection_id, Chunk.id == chunk_id))
                        .values(**update_data)
                    )
                    result = await self.session.execute(stmt)
                    total_updated += result.rowcount or 0

        logger.info(f"Batch updated {total_updated} chunks")
        return total_updated
