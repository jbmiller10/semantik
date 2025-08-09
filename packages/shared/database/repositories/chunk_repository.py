#!/usr/bin/env python3
"""
Repository for chunk operations with partition awareness.

This repository handles all database operations for the chunks table,
ensuring efficient use of PostgreSQL partitioning by collection_id.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import Chunk
from packages.shared.database.partition_utils import ChunkPartitionHelper, PartitionAwareMixin, PartitionValidation

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

        # Generate ID if not provided
        if "id" not in chunk_data:
            chunk_data["id"] = str(uuid.uuid4())

        chunk = Chunk(**chunk_data)
        self.session.add(chunk)
        await self.session.flush()

        logger.debug(f"Created chunk {chunk.id} for collection {chunk.collection_id}")
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

        # Generate IDs for chunks without them
        for chunk_data in chunks_data:
            if "id" not in chunk_data:
                chunk_data["id"] = str(uuid.uuid4())

        # Use partition-aware bulk insert
        await self.bulk_insert_partitioned(self.session, Chunk, chunks_data, partition_key_field="collection_id")

        logger.info(f"Bulk created {len(chunks_data)} chunks")
        return len(chunks_data)

    async def get_chunk_by_id(self, chunk_id: str, collection_id: str) -> Chunk | None:
        """Get a chunk by ID with partition pruning.

        IMPORTANT: collection_id is required for partition pruning.
        Without it, the query would scan all partitions.

        Args:
            chunk_id: Chunk ID
            collection_id: Collection ID (partition key)

        Returns:
            Chunk instance or None if not found

        Raises:
            ValueError: If IDs are invalid
            TypeError: If IDs are not strings
        """
        # Validate both IDs
        chunk_id = PartitionValidation.validate_uuid(chunk_id, "chunk_id")
        collection_id = PartitionValidation.validate_partition_key(collection_id, "collection_id")

        query = ChunkPartitionHelper.create_chunk_query_with_partition(collection_id, [Chunk.id == chunk_id])

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
        validated_updates = []
        for chunk_update in chunk_updates:
            if not isinstance(chunk_update, dict):
                raise TypeError("Each chunk update must be a dictionary")

            # Validate required fields
            if not all(key in chunk_update for key in ["id", "collection_id", "embedding_vector_id"]):
                raise ValueError("Each update must have 'id', 'collection_id', and 'embedding_vector_id'")

            validated_update = {
                "id": PartitionValidation.validate_uuid(chunk_update["id"], "chunk id"),
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
