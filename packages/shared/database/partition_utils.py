#!/usr/bin/env python3
"""
Utilities for working with partitioned tables in PostgreSQL.

This module provides helper classes and functions for efficient operations
on partitioned tables, particularly the chunks table which is partitioned
by collection_id.
"""

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

from sqlalchemy import Select, and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import Chunk

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PartitionAwareMixin:
    """Mixin for repositories that work with partitioned tables.

    This mixin provides common patterns and utilities for efficient
    operations on partitioned tables, ensuring partition pruning is
    enabled and bulk operations are optimized.
    """

    @staticmethod
    def ensure_partition_key_in_filter(
        query: Select[tuple[Any, ...]], partition_key_column: Any, partition_key_value: str | None
    ) -> Select[tuple[Any, ...]]:
        """Ensure partition key is included in query filter for pruning.

        Args:
            query: SQLAlchemy select query
            partition_key_column: The partition key column (e.g., Chunk.collection_id)
            partition_key_value: The partition key value to filter by

        Returns:
            Modified query with partition key filter

        Example:
            query = select(Chunk).where(Chunk.document_id == doc_id)
            query = self.ensure_partition_key_in_filter(
                query, Chunk.collection_id, collection_id
            )
        """
        if partition_key_value is not None:
            return query.where(partition_key_column == partition_key_value)

        logger.warning("Query on partitioned table without partition key - this will scan all partitions")
        return query

    @staticmethod
    def group_by_partition_key(items: Iterable[T], key_getter: Callable[[T], str]) -> dict[str, list[T]]:
        """Group items by partition key for efficient bulk operations.

        Args:
            items: Items to group
            key_getter: Function to extract partition key from item

        Returns:
            Dictionary mapping partition keys to lists of items

        Example:
            chunks_by_collection = self.group_by_partition_key(
                chunks, lambda c: c.collection_id
            )
        """
        grouped: dict[str, list[T]] = {}
        for item in items:
            key = key_getter(item)
            grouped.setdefault(key, []).append(item)
        return grouped

    async def bulk_insert_partitioned(
        self,
        session: AsyncSession,
        model_class: type[T],
        items: Sequence[dict[str, Any]],
        partition_key_field: str = "collection_id",
    ) -> None:
        """Efficiently bulk insert items into a partitioned table.

        Groups items by partition key before insertion to minimize
        partition switching overhead.

        Args:
            session: Database session
            model_class: SQLAlchemy model class
            items: List of item dictionaries to insert
            partition_key_field: Name of the partition key field
        """
        if not items:
            return

        # Group by partition key
        items_by_partition: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            key = item.get(partition_key_field)
            if key is None:
                raise ValueError(f"Partition key '{partition_key_field}' is required for all items")
            items_by_partition.setdefault(key, []).append(item)

        # Insert in batches by partition
        for partition_key, partition_items in items_by_partition.items():
            logger.debug(
                f"Inserting {len(partition_items)} items into partition for {partition_key_field}={partition_key}"
            )

            # Using bulk_insert_mappings for efficiency
            # Create a closure to capture partition_items properly
            def make_bulk_insert(items: list[dict[str, Any]]) -> Callable[[Any], None]:
                def bulk_insert(sync_session: Any) -> None:
                    sync_session.bulk_insert_mappings(model_class, items)

                return bulk_insert

            await session.run_sync(make_bulk_insert(partition_items))

    async def delete_by_partition_filter(
        self,
        session: AsyncSession,
        model_class: type[T],
        partition_key_column: Any,
        partition_key_value: str,
        additional_filters: list[Any] | None = None,
    ) -> int:
        """Delete records from a partitioned table with partition pruning.

        Args:
            session: Database session
            model_class: SQLAlchemy model class
            partition_key_column: The partition key column
            partition_key_value: The partition key value
            additional_filters: Additional WHERE conditions

        Returns:
            Number of deleted records
        """
        filters = [partition_key_column == partition_key_value]
        if additional_filters:
            filters.extend(additional_filters)

        # Execute delete with proper filters
        result = await session.execute(select(model_class).where(and_(*filters)))
        records = result.scalars().all()

        for record in records:
            await session.delete(record)

        return len(records)


class ChunkPartitionHelper:
    """Helper class specifically for chunk table partition operations."""

    @staticmethod
    def create_chunk_query_with_partition(
        collection_id: str, additional_filters: list[Any] | None = None
    ) -> Select[tuple[Chunk]]:
        """Create a chunk query with partition key for optimal performance.

        Args:
            collection_id: Collection ID (partition key)
            additional_filters: Additional WHERE conditions

        Returns:
            SQLAlchemy select query

        Example:
            query = ChunkPartitionHelper.create_chunk_query_with_partition(
                collection_id,
                [Chunk.document_id == doc_id]
            )
        """
        query = select(Chunk).where(Chunk.collection_id == collection_id)

        if additional_filters:
            query = query.where(and_(*additional_filters))

        return query

    @staticmethod
    def validate_chunk_partition_key(chunk_data: dict[str, Any]) -> None:
        """Validate that chunk data includes required partition key.

        Args:
            chunk_data: Chunk data dictionary

        Raises:
            ValueError: If collection_id is missing
        """
        if "collection_id" not in chunk_data or not chunk_data["collection_id"]:
            raise ValueError("collection_id is required for chunks table (partition key)")

    @staticmethod
    async def get_partition_statistics(session: AsyncSession, collection_id: str) -> dict[str, Any]:
        """Get statistics for a specific partition.

        Args:
            session: Database session
            collection_id: Collection ID to get stats for

        Returns:
            Dictionary with partition statistics
        """
        from sqlalchemy import func

        # Count chunks in this partition
        chunk_count = await session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.collection_id == collection_id)
        )

        # Get size statistics
        stats_query = select(
            func.count().label("count"),
            func.avg(func.length(Chunk.content)).label("avg_content_length"),
            func.sum(func.length(Chunk.content)).label("total_content_length"),
            func.min(Chunk.created_at).label("oldest_chunk"),
            func.max(Chunk.created_at).label("newest_chunk"),
        ).where(Chunk.collection_id == collection_id)

        result = await session.execute(stats_query)
        stats = result.one()

        return {
            "collection_id": collection_id,
            "chunk_count": chunk_count or 0,
            "avg_content_length": float(stats.avg_content_length or 0),
            "total_content_length": stats.total_content_length or 0,
            "oldest_chunk": stats.oldest_chunk,
            "newest_chunk": stats.newest_chunk,
        }


# Usage example for bulk operations
async def example_bulk_chunk_insert(session: AsyncSession, chunks_data: list[dict[str, Any]]) -> None:
    """Example of efficient bulk insert for chunks.

    This demonstrates the recommended pattern for inserting many chunks
    efficiently into the partitioned table.
    """
    helper = PartitionAwareMixin()

    # Validate all chunks have collection_id
    for chunk in chunks_data:
        ChunkPartitionHelper.validate_chunk_partition_key(chunk)

    # Perform partitioned bulk insert
    await helper.bulk_insert_partitioned(session, Chunk, chunks_data, partition_key_field="collection_id")
