"""Repository for Relationship model operations.

This repository handles all database operations for the relationships table,
enabling graph-enhanced search through efficient relationship lookups and
multi-hop traversals. The table is partitioned by collection_id for scalability.

All queries include partition_key for optimal PostgreSQL partition pruning.
"""

import logging
from collections import deque
from typing import Any

from sqlalchemy import and_, delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.exceptions import (
    DatabaseOperationError,
    EntityNotFoundError,
)
from shared.database.models import Entity, Relationship

logger = logging.getLogger(__name__)

# Maximum allowed hops for graph traversal to prevent runaway queries
MAX_HOPS_LIMIT = 5


class RelationshipRepository:
    """Repository for Relationship model operations.

    All queries that filter by collection_id automatically include partition_key
    for PostgreSQL partition pruning. This ensures queries only scan the relevant
    partition rather than the entire table.

    Partition Key Computation:
        partition_key = abs(hash(collection_id)) % 100

    Graph Traversal:
        - get_neighbors: BFS traversal with configurable max_hops
        - get_subgraph: Returns data formatted for React Flow visualization
    """

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session

    @staticmethod
    def _compute_partition_key(collection_id: str) -> int:
        """Compute partition key from collection_id.

        Must match the algorithm used in migrations and other repositories.

        Args:
            collection_id: The collection UUID string.

        Returns:
            Partition key in range 0-99.
        """
        return abs(hash(collection_id)) % 100

    @staticmethod
    def _normalize_relationship_type(relationship_type: str) -> str:
        """Normalize relationship type to UPPERCASE.

        Args:
            relationship_type: The relationship type string.

        Returns:
            Uppercase normalized relationship type.
        """
        return relationship_type.upper().strip()

    async def create(
        self,
        collection_id: str,
        source_entity_id: int,
        target_entity_id: int,
        relationship_type: str,
        confidence: float = 0.7,
        extraction_method: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Relationship:
        """Create a new relationship.

        Args:
            collection_id: ID of the collection
            source_entity_id: ID of the source entity
            target_entity_id: ID of the target entity
            relationship_type: Type of relationship (e.g., WORKS_FOR, LOCATED_IN)
            confidence: Confidence score for the relationship (0-1)
            extraction_method: How the relationship was extracted (dependency, pattern, llm)
            metadata: Additional metadata

        Returns:
            Created Relationship instance

        Raises:
            DatabaseOperationError: If creation fails
        """
        try:
            partition_key = self._compute_partition_key(collection_id)
            normalized_type = self._normalize_relationship_type(relationship_type)

            relationship = Relationship(
                collection_id=collection_id,
                partition_key=partition_key,
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                relationship_type=normalized_type,
                confidence=confidence,
                extraction_method=extraction_method,
                metadata_=metadata or {},
            )

            self.session.add(relationship)
            await self.session.flush()
            await self.session.refresh(relationship)

            logger.debug(
                f"Created relationship {relationship.id} "
                f"({source_entity_id} -{normalized_type}-> {target_entity_id}) "
                f"in collection {collection_id}"
            )
            return relationship

        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            raise DatabaseOperationError("create", "relationship", str(e)) from e

    async def bulk_create(
        self,
        relationships: list[dict[str, Any]],
        collection_id: str,
    ) -> int:
        """Bulk create relationships efficiently.

        Uses PostgreSQL's INSERT with multiple VALUES for optimal performance.
        All relationships are assigned to the same partition based on collection_id.

        Args:
            relationships: List of relationship dicts with keys:
                - source_entity_id: int
                - target_entity_id: int
                - relationship_type: str
                - confidence: float (optional, default 0.7)
                - extraction_method: str (optional)
                - metadata: dict (optional)
            collection_id: Collection ID for all relationships

        Returns:
            Number of relationships created

        Raises:
            DatabaseOperationError: If bulk creation fails
        """
        if not relationships:
            return 0

        try:
            partition_key = self._compute_partition_key(collection_id)

            # Prepare records with computed fields
            records = []
            for rel in relationships:
                records.append({
                    "collection_id": collection_id,
                    "partition_key": partition_key,
                    "source_entity_id": rel["source_entity_id"],
                    "target_entity_id": rel["target_entity_id"],
                    "relationship_type": self._normalize_relationship_type(
                        rel["relationship_type"]
                    ),
                    "confidence": rel.get("confidence", 0.7),
                    "extraction_method": rel.get("extraction_method"),
                    "metadata": rel.get("metadata", {}),
                })

            # Bulk insert using PostgreSQL's efficient multi-row INSERT
            stmt = insert(Relationship).values(records)
            await self.session.execute(stmt)
            await self.session.flush()

            logger.info(
                f"Bulk created {len(records)} relationships in collection {collection_id}"
            )
            return len(records)

        except Exception as e:
            logger.error(f"Failed to bulk create relationships: {e}")
            raise DatabaseOperationError("bulk_create", "relationships", str(e)) from e

    async def get_by_id(
        self,
        relationship_id: int,
        collection_id: str,
    ) -> Relationship:
        """Get relationship by ID.

        Args:
            relationship_id: Relationship ID
            collection_id: Collection ID (required for partition pruning)

        Returns:
            Relationship instance

        Raises:
            EntityNotFoundError: If relationship not found
        """
        partition_key = self._compute_partition_key(collection_id)

        result = await self.session.execute(
            select(Relationship).where(
                and_(
                    Relationship.id == relationship_id,
                    Relationship.collection_id == collection_id,
                    Relationship.partition_key == partition_key,
                )
            )
        )
        relationship = result.scalar_one_or_none()

        if not relationship:
            raise EntityNotFoundError("Relationship", str(relationship_id))

        return relationship

    async def get_by_source_entity(
        self,
        entity_id: int,
        collection_id: str,
        relationship_types: list[str] | None = None,
    ) -> list[Relationship]:
        """Get all relationships where the given entity is the source.

        Args:
            entity_id: Source entity ID
            collection_id: Collection ID
            relationship_types: Optional list of relationship types to filter by

        Returns:
            List of Relationship instances
        """
        partition_key = self._compute_partition_key(collection_id)

        conditions = [
            Relationship.source_entity_id == entity_id,
            Relationship.collection_id == collection_id,
            Relationship.partition_key == partition_key,
        ]

        if relationship_types:
            normalized_types = [
                self._normalize_relationship_type(t) for t in relationship_types
            ]
            conditions.append(Relationship.relationship_type.in_(normalized_types))

        result = await self.session.execute(
            select(Relationship).where(and_(*conditions))
        )
        return list(result.scalars().all())

    async def get_by_target_entity(
        self,
        entity_id: int,
        collection_id: str,
        relationship_types: list[str] | None = None,
    ) -> list[Relationship]:
        """Get all relationships where the given entity is the target.

        Args:
            entity_id: Target entity ID
            collection_id: Collection ID
            relationship_types: Optional list of relationship types to filter by

        Returns:
            List of Relationship instances
        """
        partition_key = self._compute_partition_key(collection_id)

        conditions = [
            Relationship.target_entity_id == entity_id,
            Relationship.collection_id == collection_id,
            Relationship.partition_key == partition_key,
        ]

        if relationship_types:
            normalized_types = [
                self._normalize_relationship_type(t) for t in relationship_types
            ]
            conditions.append(Relationship.relationship_type.in_(normalized_types))

        result = await self.session.execute(
            select(Relationship).where(and_(*conditions))
        )
        return list(result.scalars().all())

    async def get_by_entity(
        self,
        entity_id: int,
        collection_id: str,
        relationship_types: list[str] | None = None,
    ) -> list[Relationship]:
        """Get all relationships where the given entity is either source or target.

        Args:
            entity_id: Entity ID (can be source or target)
            collection_id: Collection ID
            relationship_types: Optional list of relationship types to filter by

        Returns:
            List of Relationship instances (deduplicated)
        """
        # Get both source and target relationships
        source_rels = await self.get_by_source_entity(
            entity_id, collection_id, relationship_types
        )
        target_rels = await self.get_by_target_entity(
            entity_id, collection_id, relationship_types
        )

        # Deduplicate by relationship ID
        seen_ids: set[int] = set()
        result: list[Relationship] = []
        for rel in source_rels + target_rels:
            if rel.id not in seen_ids:
                seen_ids.add(rel.id)
                result.append(rel)

        return result

    async def get_neighbors(
        self,
        entity_ids: list[int],
        collection_id: str,
        max_hops: int = 1,
        relationship_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get neighboring entities via BFS traversal.

        Performs breadth-first search from the given entity IDs, returning
        all entities reachable within max_hops.

        Args:
            entity_ids: Starting entity IDs
            collection_id: Collection ID
            max_hops: Maximum number of hops (capped at 5)
            relationship_types: Optional list of relationship types to traverse

        Returns:
            Dict with:
                - entities: Dict mapping entity_id to hop distance
                - relationships: List of relationship dicts traversed
        """
        # Cap max_hops to prevent runaway queries
        max_hops = min(max_hops, MAX_HOPS_LIMIT)

        partition_key = self._compute_partition_key(collection_id)

        # BFS state
        visited: dict[int, int] = {}  # entity_id -> hop distance
        queue: deque[tuple[int, int]] = deque()  # (entity_id, current_hop)
        traversed_relationships: list[dict[str, Any]] = []
        seen_rel_ids: set[int] = set()

        # Initialize with starting entities at hop 0
        for eid in entity_ids:
            if eid not in visited:
                visited[eid] = 0
                queue.append((eid, 0))

        # Normalize relationship types once
        normalized_types: list[str] | None = None
        if relationship_types:
            normalized_types = [
                self._normalize_relationship_type(t) for t in relationship_types
            ]

        # BFS traversal
        while queue:
            current_id, current_hop = queue.popleft()

            # Don't explore beyond max_hops
            if current_hop >= max_hops:
                continue

            # Build query conditions
            conditions = [
                Relationship.collection_id == collection_id,
                Relationship.partition_key == partition_key,
            ]

            if normalized_types:
                conditions.append(Relationship.relationship_type.in_(normalized_types))

            # Get outgoing relationships (current entity is source)
            outgoing_result = await self.session.execute(
                select(Relationship).where(
                    and_(*conditions, Relationship.source_entity_id == current_id)
                )
            )
            outgoing = outgoing_result.scalars().all()

            # Get incoming relationships (current entity is target)
            incoming_result = await self.session.execute(
                select(Relationship).where(
                    and_(*conditions, Relationship.target_entity_id == current_id)
                )
            )
            incoming = incoming_result.scalars().all()

            # Process all relationships
            for rel in outgoing:
                if rel.id not in seen_rel_ids:
                    seen_rel_ids.add(rel.id)
                    traversed_relationships.append({
                        "id": rel.id,
                        "source_entity_id": rel.source_entity_id,
                        "target_entity_id": rel.target_entity_id,
                        "relationship_type": rel.relationship_type,
                        "confidence": rel.confidence,
                    })

                neighbor_id = rel.target_entity_id
                if neighbor_id not in visited:
                    visited[neighbor_id] = current_hop + 1
                    queue.append((neighbor_id, current_hop + 1))

            for rel in incoming:
                if rel.id not in seen_rel_ids:
                    seen_rel_ids.add(rel.id)
                    traversed_relationships.append({
                        "id": rel.id,
                        "source_entity_id": rel.source_entity_id,
                        "target_entity_id": rel.target_entity_id,
                        "relationship_type": rel.relationship_type,
                        "confidence": rel.confidence,
                    })

                neighbor_id = rel.source_entity_id
                if neighbor_id not in visited:
                    visited[neighbor_id] = current_hop + 1
                    queue.append((neighbor_id, current_hop + 1))

        return {
            "entities": visited,
            "relationships": traversed_relationships,
        }

    async def get_subgraph(
        self,
        entity_ids: list[int],
        collection_id: str,
        max_hops: int = 2,
        relationship_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get subgraph data formatted for React Flow visualization.

        Performs BFS traversal and returns graph data suitable for rendering
        with React Flow or similar graph visualization libraries.

        Args:
            entity_ids: Starting entity IDs (focal points)
            collection_id: Collection ID
            max_hops: Maximum number of hops (capped at 5)
            relationship_types: Optional list of relationship types to traverse

        Returns:
            Dict with:
                - nodes: List of entity dicts with id, name, type, hop
                - edges: List of relationship dicts with id, source, target, type, confidence
        """
        # Cap max_hops
        max_hops = min(max_hops, MAX_HOPS_LIMIT)

        partition_key = self._compute_partition_key(collection_id)

        # Get neighbors using BFS
        neighbor_data = await self.get_neighbors(
            entity_ids, collection_id, max_hops, relationship_types
        )

        visited_entities = neighbor_data["entities"]
        traversed_relationships = neighbor_data["relationships"]

        # Fetch entity details for all visited entities
        entity_ids_to_fetch = list(visited_entities.keys())
        nodes: list[dict[str, Any]] = []

        if entity_ids_to_fetch:
            result = await self.session.execute(
                select(Entity).where(
                    and_(
                        Entity.id.in_(entity_ids_to_fetch),
                        Entity.collection_id == collection_id,
                        Entity.partition_key == partition_key,
                    )
                )
            )
            entities = result.scalars().all()

            for entity in entities:
                nodes.append({
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "hop": visited_entities.get(entity.id, 0),
                })

        # Format edges for React Flow
        edges: list[dict[str, Any]] = []
        for rel in traversed_relationships:
            edges.append({
                "id": rel["id"],
                "source": rel["source_entity_id"],
                "target": rel["target_entity_id"],
                "type": rel["relationship_type"],
                "confidence": rel["confidence"],
            })

        return {
            "nodes": nodes,
            "edges": edges,
        }

    async def count_by_collection(self, collection_id: str) -> int:
        """Count total relationships in a collection.

        Args:
            collection_id: Collection ID

        Returns:
            Total relationship count
        """
        partition_key = self._compute_partition_key(collection_id)

        result = await self.session.execute(
            select(func.count(Relationship.id)).where(
                and_(
                    Relationship.collection_id == collection_id,
                    Relationship.partition_key == partition_key,
                )
            )
        )
        return result.scalar() or 0

    async def count_by_type(self, collection_id: str) -> dict[str, int]:
        """Count relationships by type in a collection.

        Args:
            collection_id: Collection ID

        Returns:
            Dict mapping relationship_type to count
        """
        partition_key = self._compute_partition_key(collection_id)

        result = await self.session.execute(
            select(Relationship.relationship_type, func.count(Relationship.id))
            .where(
                and_(
                    Relationship.collection_id == collection_id,
                    Relationship.partition_key == partition_key,
                )
            )
            .group_by(Relationship.relationship_type)
        )
        return {row[0]: row[1] for row in result.all()}

    async def delete_by_entity(
        self,
        entity_id: int,
        collection_id: str,
    ) -> int:
        """Delete all relationships involving an entity (as source or target).

        This is typically called when an entity is deleted.

        Args:
            entity_id: Entity ID
            collection_id: Collection ID

        Returns:
            Number of relationships deleted
        """
        partition_key = self._compute_partition_key(collection_id)

        # Delete relationships where entity is source or target
        result = await self.session.execute(
            delete(Relationship).where(
                and_(
                    Relationship.collection_id == collection_id,
                    Relationship.partition_key == partition_key,
                    (Relationship.source_entity_id == entity_id)
                    | (Relationship.target_entity_id == entity_id),
                )
            )
        )
        await self.session.flush()

        deleted_count = result.rowcount or 0
        logger.info(
            f"Deleted {deleted_count} relationships for entity {entity_id} "
            f"in collection {collection_id}"
        )
        return deleted_count

    async def delete_by_collection(self, collection_id: str) -> int:
        """Delete all relationships in a collection.

        This is typically called when a collection is deleted or re-indexed.

        Args:
            collection_id: Collection ID

        Returns:
            Number of relationships deleted
        """
        partition_key = self._compute_partition_key(collection_id)

        result = await self.session.execute(
            delete(Relationship).where(
                and_(
                    Relationship.collection_id == collection_id,
                    Relationship.partition_key == partition_key,
                )
            )
        )
        await self.session.flush()

        deleted_count = result.rowcount or 0
        logger.info(
            f"Deleted {deleted_count} relationships for collection {collection_id}"
        )
        return deleted_count
