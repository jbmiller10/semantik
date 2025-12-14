"""Repository for Entity model operations.

This repository handles all database operations for the entities table,
ensuring efficient use of PostgreSQL partitioning by collection_id.
All queries include partition_key for optimal partition pruning.
"""

import hashlib
import logging
from typing import Any

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.exceptions import (
    DatabaseOperationError,
    EntityNotFoundError,
)
from shared.database.models import Entity
from shared.database.partition_utils import PartitionAwareMixin

logger = logging.getLogger(__name__)


class EntityRepository:
    """Repository for Entity model operations.

    All queries that filter by collection_id automatically include partition_key
    for PostgreSQL partition pruning. This ensures queries only scan the relevant
    partition rather than the entire table.

    Partition Key Computation:
        partition_key = abs(hashtext(collection_id::text)) % 100
        (resolved via database-side logic when available)

    Name Hash Computation (for deduplication):
        name_hash = sha256(f"{entity_type}:{name.lower().strip()}")
    """

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session
        self._partition_key_cache: dict[str, int] = {}

    async def _compute_partition_key(self, collection_id: str) -> int:
        """Compute (and cache) partition key from collection_id."""
        cached = self._partition_key_cache.get(collection_id)
        if cached is not None:
            return cached

        partition_key = await PartitionAwareMixin.compute_partition_key(self.session, collection_id)
        self._partition_key_cache[collection_id] = partition_key
        return partition_key

    @staticmethod
    def _compute_name_hash(name: str, entity_type: str) -> str:
        """Compute hash for entity deduplication.

        Args:
            name: The entity name.
            entity_type: The entity type (PERSON, ORG, etc.).

        Returns:
            SHA256 hash of normalized "{entity_type}:{name_normalized}".
        """
        normalized = f"{entity_type}:{name.lower().strip()}"
        return hashlib.sha256(normalized.encode()).hexdigest()

    async def create(
        self,
        collection_id: str,
        document_id: str,
        name: str,
        entity_type: str,
        chunk_id: int | None = None,
        start_offset: int | None = None,
        end_offset: int | None = None,
        confidence: float = 0.85,
        metadata: dict[str, Any] | None = None,
    ) -> Entity:
        """Create a new entity.

        Args:
            collection_id: ID of the collection
            document_id: ID of the source document
            name: Entity name as extracted
            entity_type: Entity type (PERSON, ORG, GPE, etc.)
            chunk_id: Optional ID of the source chunk
            start_offset: Character offset where entity starts in chunk
            end_offset: Character offset where entity ends in chunk
            confidence: NER confidence score (0-1)
            metadata: Additional metadata

        Returns:
            Created Entity instance

        Raises:
            DatabaseOperationError: If creation fails
        """
        try:
            partition_key = await self._compute_partition_key(collection_id)
            name_normalized = name.lower().strip()
            name_hash = self._compute_name_hash(name, entity_type)

            entity = Entity(
                collection_id=collection_id,
                partition_key=partition_key,
                document_id=document_id,
                chunk_id=chunk_id,
                name=name,
                name_normalized=name_normalized,
                name_hash=name_hash,
                entity_type=entity_type,
                start_offset=start_offset,
                end_offset=end_offset,
                confidence=confidence,
                metadata_=metadata or {},
            )

            self.session.add(entity)
            await self.session.flush()
            await self.session.refresh(entity)

            logger.debug(
                f"Created entity {entity.id} '{name}' ({entity_type}) "
                f"in collection {collection_id}"
            )
            return entity

        except Exception as e:
            logger.error(f"Failed to create entity: {e}")
            raise DatabaseOperationError("create", "entity", str(e)) from e

    async def bulk_create(
        self,
        entities: list[dict[str, Any]],
        collection_id: str,
    ) -> int:
        """Bulk create entities efficiently.

        Uses PostgreSQL's INSERT with multiple VALUES for optimal performance.
        All entities are assigned to the same partition based on collection_id.

        Args:
            entities: List of entity dicts with keys: name, entity_type, document_id,
                     and optionally: chunk_id, start_offset, end_offset, confidence, metadata
            collection_id: Collection ID for all entities

        Returns:
            Number of entities created

        Raises:
            DatabaseOperationError: If bulk creation fails
        """
        if not entities:
            return 0

        try:
            partition_key = await self._compute_partition_key(collection_id)

            # Prepare records with computed fields
            records = []
            for entity in entities:
                name = entity["name"]
                entity_type = entity["entity_type"]
                records.append({
                    "collection_id": collection_id,
                    "partition_key": partition_key,
                    "document_id": entity["document_id"],
                    "chunk_id": entity.get("chunk_id"),
                    "name": name,
                    "name_normalized": name.lower().strip(),
                    "name_hash": self._compute_name_hash(name, entity_type),
                    "entity_type": entity_type,
                    "start_offset": entity.get("start_offset"),
                    "end_offset": entity.get("end_offset"),
                    "confidence": entity.get("confidence", 0.85),
                    "metadata_": entity.get("metadata", {}),
                })

            # Bulk insert using PostgreSQL's efficient multi-row INSERT
            stmt = insert(Entity).values(records)
            await self.session.execute(stmt)
            await self.session.flush()

            logger.info(f"Bulk created {len(records)} entities in collection {collection_id}")
            return len(records)

        except Exception as e:
            logger.error(f"Failed to bulk create entities: {e}")
            raise DatabaseOperationError("bulk_create", "entities", str(e)) from e

    async def get_by_id(
        self,
        entity_id: int,
        collection_id: str,
    ) -> Entity:
        """Get entity by ID.

        Args:
            entity_id: Entity ID
            collection_id: Collection ID (required for partition pruning)

        Returns:
            Entity instance

        Raises:
            EntityNotFoundError: If entity not found
        """
        partition_key = await self._compute_partition_key(collection_id)

        result = await self.session.execute(
            select(Entity).where(
                and_(
                    Entity.id == entity_id,
                    Entity.collection_id == collection_id,
                    Entity.partition_key == partition_key,
                )
            )
        )
        entity = result.scalar_one_or_none()

        if not entity:
            raise EntityNotFoundError("Entity", str(entity_id))

        return entity

    async def get_by_document(
        self,
        document_id: str,
        collection_id: str,
    ) -> list[Entity]:
        """Get all entities for a document.

        Args:
            document_id: Document ID
            collection_id: Collection ID

        Returns:
            List of Entity instances ordered by creation time
        """
        partition_key = await self._compute_partition_key(collection_id)

        result = await self.session.execute(
            select(Entity).where(
                and_(
                    Entity.document_id == document_id,
                    Entity.collection_id == collection_id,
                    Entity.partition_key == partition_key,
                )
            ).order_by(Entity.created_at)
        )
        return list(result.scalars().all())

    async def get_by_chunk(
        self,
        chunk_id: int,
        collection_id: str,
    ) -> list[Entity]:
        """Get all entities for a specific chunk.

        Args:
            chunk_id: Chunk ID
            collection_id: Collection ID

        Returns:
            List of Entity instances
        """
        partition_key = await self._compute_partition_key(collection_id)

        result = await self.session.execute(
            select(Entity).where(
                and_(
                    Entity.chunk_id == chunk_id,
                    Entity.collection_id == collection_id,
                    Entity.partition_key == partition_key,
                )
            )
        )
        return list(result.scalars().all())

    async def get_by_type(
        self,
        collection_id: str,
        entity_type: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        """Get entities by type within a collection.

        Only returns canonical entities (those without a canonical_id set),
        excluding entities that have been marked as aliases/duplicates.

        Args:
            collection_id: Collection ID
            entity_type: Entity type to filter by (e.g., PERSON, ORG, GPE)
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            List of Entity instances ordered by normalized name
        """
        partition_key = await self._compute_partition_key(collection_id)

        result = await self.session.execute(
            select(Entity).where(
                and_(
                    Entity.collection_id == collection_id,
                    Entity.partition_key == partition_key,
                    Entity.entity_type == entity_type,
                    Entity.canonical_id.is_(None),  # Only canonical entities
                )
            ).order_by(Entity.name_normalized)
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def search_by_name(
        self,
        collection_id: str,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[Entity]:
        """Search entities by name (substring match on normalized name).

        Only returns canonical entities (those without a canonical_id set).

        Args:
            collection_id: Collection ID
            query: Substring to search for (case-insensitive)
            entity_types: Optional list of entity types to filter by
            limit: Maximum results to return

        Returns:
            List of matching Entity instances ordered by normalized name
        """
        partition_key = await self._compute_partition_key(collection_id)
        query_normalized = query.lower().strip()

        conditions = [
            Entity.collection_id == collection_id,
            Entity.partition_key == partition_key,
            Entity.name_normalized.contains(query_normalized),
            Entity.canonical_id.is_(None),  # Only canonical entities
        ]

        if entity_types:
            conditions.append(Entity.entity_type.in_(entity_types))

        result = await self.session.execute(
            select(Entity)
            .where(and_(*conditions))
            .order_by(Entity.name_normalized)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def find_duplicates(
        self,
        collection_id: str,
        name: str,
        entity_type: str,
    ) -> list[Entity]:
        """Find potential duplicate entities by name hash.

        This method is used during entity extraction to detect and merge
        duplicates. It finds all entities with the same normalized name
        and type combination.

        Args:
            collection_id: Collection ID
            name: Entity name
            entity_type: Entity type

        Returns:
            List of entities with matching name_hash
        """
        partition_key = await self._compute_partition_key(collection_id)
        name_hash = self._compute_name_hash(name, entity_type)

        result = await self.session.execute(
            select(Entity).where(
                and_(
                    Entity.collection_id == collection_id,
                    Entity.partition_key == partition_key,
                    Entity.name_hash == name_hash,
                )
            )
        )
        return list(result.scalars().all())

    async def set_canonical(
        self,
        entity_id: int,
        canonical_id: int,
        collection_id: str,
    ) -> None:
        """Mark an entity as an alias of another (canonical) entity.

        When entities are determined to be duplicates, one is designated
        as the "canonical" entity and others point to it via canonical_id.

        Args:
            entity_id: ID of the entity to mark as alias
            canonical_id: ID of the canonical entity
            collection_id: Collection ID

        Raises:
            DatabaseOperationError: If update fails
        """
        try:
            partition_key = await self._compute_partition_key(collection_id)

            await self.session.execute(
                update(Entity)
                .where(
                    and_(
                        Entity.id == entity_id,
                        Entity.collection_id == collection_id,
                        Entity.partition_key == partition_key,
                    )
                )
                .values(canonical_id=canonical_id)
            )
            await self.session.flush()

            logger.debug(f"Set entity {entity_id} canonical to {canonical_id}")

        except Exception as e:
            logger.error(f"Failed to set canonical entity: {e}")
            raise DatabaseOperationError("update", "entity", str(e)) from e

    async def count_by_collection(self, collection_id: str) -> int:
        """Count total entities in a collection.

        Counts all entities including aliases (those with canonical_id set).

        Args:
            collection_id: Collection ID

        Returns:
            Total entity count
        """
        partition_key = await self._compute_partition_key(collection_id)

        result = await self.session.execute(
            select(func.count(Entity.id)).where(
                and_(
                    Entity.collection_id == collection_id,
                    Entity.partition_key == partition_key,
                )
            )
        )
        return result.scalar() or 0

    async def count_by_type(self, collection_id: str) -> dict[str, int]:
        """Count entities by type in a collection.

        Only counts canonical entities (those without canonical_id set),
        excluding aliases/duplicates.

        Args:
            collection_id: Collection ID

        Returns:
            Dict mapping entity_type to count
        """
        partition_key = await self._compute_partition_key(collection_id)

        result = await self.session.execute(
            select(Entity.entity_type, func.count(Entity.id))
            .where(
                and_(
                    Entity.collection_id == collection_id,
                    Entity.partition_key == partition_key,
                    Entity.canonical_id.is_(None),  # Only canonical entities
                )
            )
            .group_by(Entity.entity_type)
        )
        return {row[0]: row[1] for row in result.all()}

    async def delete_by_document(
        self,
        document_id: str,
        collection_id: str,
    ) -> int:
        """Delete all entities for a document.

        This is typically called when a document is re-processed or deleted.

        Args:
            document_id: Document ID
            collection_id: Collection ID

        Returns:
            Number of entities deleted
        """
        partition_key = await self._compute_partition_key(collection_id)

        result = await self.session.execute(
            delete(Entity).where(
                and_(
                    Entity.document_id == document_id,
                    Entity.collection_id == collection_id,
                    Entity.partition_key == partition_key,
                )
            )
        )
        await self.session.flush()

        deleted_count = result.rowcount or 0
        logger.info(
            f"Deleted {deleted_count} entities for document {document_id} "
            f"in collection {collection_id}"
        )
        return deleted_count

    async def delete_by_collection(self, collection_id: str) -> int:
        """Delete all entities in a collection.

        This is typically called when a collection is deleted or re-indexed.

        Args:
            collection_id: Collection ID

        Returns:
            Number of entities deleted
        """
        partition_key = await self._compute_partition_key(collection_id)

        result = await self.session.execute(
            delete(Entity).where(
                and_(
                    Entity.collection_id == collection_id,
                    Entity.partition_key == partition_key,
                )
            )
        )
        await self.session.flush()

        deleted_count = result.rowcount or 0
        logger.info(f"Deleted {deleted_count} entities for collection {collection_id}")
        return deleted_count
