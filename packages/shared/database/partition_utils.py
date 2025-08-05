#!/usr/bin/env python3
"""
Utilities for working with partitioned tables in PostgreSQL.

This module provides helper classes and functions for efficient operations
on partitioned tables, particularly the chunks table which is partitioned
by collection_id.
"""

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

from sqlalchemy import Select, and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import Chunk

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PartitionValidation:
    """Validation utilities for partition operations."""

    # UUID v4 pattern
    UUID_PATTERN = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$", re.IGNORECASE)

    # Limits for validation
    MAX_BATCH_SIZE = 10000  # Maximum items in a single batch operation
    MAX_STRING_LENGTH = 1000000  # 1MB max for content fields

    @classmethod
    def validate_uuid(cls, value: Any, field_name: str = "id") -> str:
        """Validate UUID format.

        Args:
            value: Value to validate
            field_name: Name of the field for error messages

        Returns:
            Validated UUID string

        Raises:
            ValueError: If value is not a valid UUID
            TypeError: If value is not a string
        """
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")

        if not value:
            raise ValueError(f"{field_name} cannot be empty")

        if not cls.UUID_PATTERN.match(value):
            raise ValueError(f"{field_name} must be a valid UUID v4, got: {value}")

        return value.lower()  # Normalize to lowercase

    @classmethod
    def validate_partition_key(cls, value: Any, field_name: str = "collection_id") -> str:
        """Validate partition key value.

        Args:
            value: Partition key value
            field_name: Name of the field

        Returns:
            Validated partition key

        Raises:
            ValueError: If invalid
            TypeError: If wrong type
        """
        # For now, partition keys are UUIDs
        return cls.validate_uuid(value, field_name)

    @classmethod
    def validate_chunk_data(cls, chunk_data: dict[str, Any]) -> dict[str, Any]:
        """Validate chunk data before database operations.

        Args:
            chunk_data: Chunk data dictionary

        Returns:
            Validated chunk data

        Raises:
            ValueError: If data is invalid
            TypeError: If types are incorrect
        """
        if not isinstance(chunk_data, dict):
            raise TypeError(f"chunk_data must be a dictionary, got {type(chunk_data).__name__}")

        # Required fields
        if "collection_id" not in chunk_data:
            raise ValueError("collection_id is required for chunks (partition key)")

        # Validate collection_id
        chunk_data["collection_id"] = cls.validate_partition_key(chunk_data["collection_id"], "collection_id")

        # Validate document_id if present
        if "document_id" in chunk_data and chunk_data["document_id"] is not None:
            chunk_data["document_id"] = cls.validate_uuid(chunk_data["document_id"], "document_id")

        # Validate id if present
        if "id" in chunk_data and chunk_data["id"] is not None:
            chunk_data["id"] = cls.validate_uuid(chunk_data["id"], "id")

        # Validate chunk_index
        if "chunk_index" in chunk_data:
            if not isinstance(chunk_data["chunk_index"], int):
                raise TypeError("chunk_index must be an integer")
            if chunk_data["chunk_index"] < 0:
                raise ValueError("chunk_index must be non-negative")

        # Validate content length
        if "content" in chunk_data:
            if not isinstance(chunk_data["content"], str):
                raise TypeError("content must be a string")
            if len(chunk_data["content"]) > cls.MAX_STRING_LENGTH:
                raise ValueError(f"content exceeds maximum length of {cls.MAX_STRING_LENGTH} characters")

        # Validate metadata if present
        if (
            "metadata" in chunk_data
            and chunk_data["metadata"] is not None
            and not isinstance(chunk_data["metadata"], dict)
        ):
            raise TypeError("metadata must be a dictionary")

        return chunk_data

    @classmethod
    def validate_batch_size(cls, items: Sequence[Any], operation: str = "bulk operation") -> None:
        """Validate batch size for bulk operations.

        Args:
            items: Items to validate
            operation: Operation name for error message

        Raises:
            ValueError: If batch size exceeds limit
        """
        if len(items) > cls.MAX_BATCH_SIZE:
            raise ValueError(
                f"{operation} batch size ({len(items)}) exceeds maximum allowed "
                f"({cls.MAX_BATCH_SIZE}). Please split into smaller batches."
            )

    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 255) -> str:
        """Sanitize string input for database operations.

        Args:
            value: String to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return str(value)

        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]

        # Remove null bytes which PostgreSQL doesn't like
        return value.replace("\x00", "")


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
            # Validate partition key format
            validated_key = PartitionValidation.validate_partition_key(partition_key_value)
            return query.where(partition_key_column == validated_key)

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

        # Validate batch size
        PartitionValidation.validate_batch_size(items, "bulk insert")

        # Validate and group by partition key
        items_by_partition: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            key = item.get(partition_key_field)
            if key is None:
                raise ValueError(f"Partition key '{partition_key_field}' is required for all items")

            # Validate partition key format
            validated_key = PartitionValidation.validate_partition_key(key, partition_key_field)
            item[partition_key_field] = validated_key  # Update with validated key

            # Additional validation for chunk data
            if model_class.__name__ == "Chunk":
                item = PartitionValidation.validate_chunk_data(item)

            items_by_partition.setdefault(validated_key, []).append(item)

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
        # Validate partition key
        validated_key = PartitionValidation.validate_partition_key(partition_key_value)

        filters = [partition_key_column == validated_key]
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
        # Validate collection_id
        validated_id = PartitionValidation.validate_partition_key(collection_id, "collection_id")

        query = select(Chunk).where(Chunk.collection_id == validated_id)

        if additional_filters:
            query = query.where(and_(*additional_filters))

        return query

    @staticmethod
    def validate_chunk_partition_key(chunk_data: dict[str, Any]) -> None:
        """Validate that chunk data includes required partition key.

        Args:
            chunk_data: Chunk data dictionary

        Raises:
            ValueError: If collection_id is missing or invalid
            TypeError: If chunk_data is not a dictionary
        """
        # Delegate to comprehensive validation
        PartitionValidation.validate_chunk_data(chunk_data)

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

        # Validate collection_id
        validated_id = PartitionValidation.validate_partition_key(collection_id, "collection_id")

        try:
            # Count chunks in this partition
            chunk_count = await session.scalar(
                select(func.count()).select_from(Chunk).where(Chunk.collection_id == validated_id)
            )

            # Get size statistics
            stats_query = select(
                func.count().label("count"),
                func.avg(func.length(Chunk.content)).label("avg_content_length"),
                func.sum(func.length(Chunk.content)).label("total_content_length"),
                func.min(Chunk.created_at).label("oldest_chunk"),
                func.max(Chunk.created_at).label("newest_chunk"),
            ).where(Chunk.collection_id == validated_id)

            result = await session.execute(stats_query)
            stats = result.one()

            # Handle null/None values safely
            avg_length = 0.0
            if stats.avg_content_length is not None:
                try:
                    avg_length = float(stats.avg_content_length)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid avg_content_length value: {stats.avg_content_length}")
                    avg_length = 0.0

            return {
                "collection_id": validated_id,
                "chunk_count": int(chunk_count or 0),
                "avg_content_length": avg_length,
                "total_content_length": int(stats.total_content_length or 0),
                "oldest_chunk": stats.oldest_chunk,
                "newest_chunk": stats.newest_chunk,
            }
        except Exception as e:
            logger.error(f"Error getting partition statistics for collection {validated_id}: {e}")
            # Return safe defaults on error
            return {
                "collection_id": validated_id,
                "chunk_count": 0,
                "avg_content_length": 0.0,
                "total_content_length": 0,
                "oldest_chunk": None,
                "newest_chunk": None,
            }


# Usage example for bulk operations
async def example_bulk_chunk_insert(session: AsyncSession, chunks_data: list[dict[str, Any]]) -> None:
    """Example of efficient bulk insert for chunks.

    This demonstrates the recommended pattern for inserting many chunks
    efficiently into the partitioned table.
    """
    helper = PartitionAwareMixin()

    # Validation is now handled internally by bulk_insert_partitioned
    # which includes batch size validation, partition key validation,
    # and comprehensive chunk data validation

    # Perform partitioned bulk insert
    await helper.bulk_insert_partitioned(session, Chunk, chunks_data, partition_key_field="collection_id")
