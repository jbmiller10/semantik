"""Tests to verify graph testing fixtures work correctly.

These tests validate that the entity_factory, relationship_factory, graph_factory,
and collection_with_graph fixtures are properly configured and functional.

Note: These tests require a database connection with the graph tables migrated.
They are marked as integration tests and will be skipped if the database is unavailable.
"""

import hashlib

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.models import Collection, Entity, Relationship


# Mark all tests in this module as integration tests requiring database
pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestEntityFactory:
    """Tests for entity_factory fixture."""

    async def test_entity_factory_creates_entity(
        self,
        db_session: AsyncSession,
        entity_factory,
    ) -> None:
        """Entity factory should create a valid Entity instance."""
        entity = await entity_factory(
            name="John Smith",
            entity_type="PERSON",
        )

        assert entity is not None
        assert entity.id is not None
        assert entity.name == "John Smith"
        assert entity.entity_type == "PERSON"

    async def test_entity_factory_computes_partition_key(
        self,
        db_session: AsyncSession,
        entity_factory,
    ) -> None:
        """Entity factory should compute partition_key from collection_id."""
        entity = await entity_factory()

        expected_partition_key = abs(hash(entity.collection_id)) % 100
        assert entity.partition_key == expected_partition_key

    async def test_entity_factory_computes_name_hash(
        self,
        db_session: AsyncSession,
        entity_factory,
    ) -> None:
        """Entity factory should compute name_hash using SHA256."""
        entity = await entity_factory(
            name="Test Entity",
            entity_type="ORG",
        )

        expected_hash_input = "ORG:test entity"  # lowercase and stripped
        expected_hash = hashlib.sha256(expected_hash_input.encode()).hexdigest()
        assert entity.name_hash == expected_hash

    async def test_entity_factory_normalizes_name(
        self,
        db_session: AsyncSession,
        entity_factory,
    ) -> None:
        """Entity factory should normalize entity name to lowercase and stripped."""
        entity = await entity_factory(
            name="  ACME Corp  ",
            entity_type="ORG",
        )

        assert entity.name == "  ACME Corp  "  # Original preserved
        assert entity.name_normalized == "acme corp"  # Normalized

    async def test_entity_factory_creates_collection_if_not_provided(
        self,
        db_session: AsyncSession,
        entity_factory,
    ) -> None:
        """Entity factory should create a graph-enabled collection if not provided."""
        entity = await entity_factory()

        assert entity.collection_id is not None
        # Verify collection exists and has graph_enabled
        from sqlalchemy import select

        result = await db_session.execute(
            select(Collection).where(Collection.id == entity.collection_id)
        )
        collection = result.scalar_one()
        assert collection.graph_enabled is True

    async def test_entity_factory_creates_document_if_not_provided(
        self,
        db_session: AsyncSession,
        entity_factory,
    ) -> None:
        """Entity factory should create a document if not provided."""
        entity = await entity_factory()

        assert entity.document_id is not None

    async def test_entity_factory_with_custom_confidence(
        self,
        db_session: AsyncSession,
        entity_factory,
    ) -> None:
        """Entity factory should accept custom confidence score."""
        entity = await entity_factory(confidence=0.95)

        assert entity.confidence == 0.95

    async def test_entity_factory_reuses_cached_collection(
        self,
        db_session: AsyncSession,
        entity_factory,
    ) -> None:
        """Entity factory should reuse the same collection for multiple entities."""
        entity1 = await entity_factory(name="Entity 1")
        entity2 = await entity_factory(name="Entity 2")

        assert entity1.collection_id == entity2.collection_id


class TestRelationshipFactory:
    """Tests for relationship_factory fixture."""

    async def test_relationship_factory_creates_relationship(
        self,
        db_session: AsyncSession,
        relationship_factory,
    ) -> None:
        """Relationship factory should create a valid Relationship instance."""
        relationship = await relationship_factory(
            relationship_type="WORKS_FOR",
        )

        assert relationship is not None
        assert relationship.id is not None
        assert relationship.relationship_type == "WORKS_FOR"

    async def test_relationship_factory_creates_entities_if_not_provided(
        self,
        db_session: AsyncSession,
        relationship_factory,
    ) -> None:
        """Relationship factory should create source and target entities if not provided."""
        relationship = await relationship_factory()

        assert relationship.source_entity_id is not None
        assert relationship.target_entity_id is not None

    async def test_relationship_factory_computes_partition_key(
        self,
        db_session: AsyncSession,
        relationship_factory,
    ) -> None:
        """Relationship factory should compute partition_key from collection_id."""
        relationship = await relationship_factory()

        expected_partition_key = abs(hash(relationship.collection_id)) % 100
        assert relationship.partition_key == expected_partition_key

    async def test_relationship_factory_normalizes_type_to_uppercase(
        self,
        db_session: AsyncSession,
        relationship_factory,
    ) -> None:
        """Relationship factory should normalize relationship_type to uppercase."""
        relationship = await relationship_factory(
            relationship_type="works_for",  # lowercase
        )

        assert relationship.relationship_type == "WORKS_FOR"

    async def test_relationship_factory_with_custom_confidence(
        self,
        db_session: AsyncSession,
        relationship_factory,
    ) -> None:
        """Relationship factory should accept custom confidence score."""
        relationship = await relationship_factory(confidence=0.9)

        assert relationship.confidence == 0.9

    async def test_relationship_factory_with_extraction_method(
        self,
        db_session: AsyncSession,
        relationship_factory,
    ) -> None:
        """Relationship factory should accept extraction_method."""
        relationship = await relationship_factory(extraction_method="llm")

        assert relationship.extraction_method == "llm"

    async def test_relationship_factory_with_existing_entities(
        self,
        db_session: AsyncSession,
        entity_factory,
        relationship_factory,
    ) -> None:
        """Relationship factory should accept existing entity IDs."""
        entity1 = await entity_factory(name="Alice", entity_type="PERSON")
        entity2 = await entity_factory(
            name="Acme Corp",
            entity_type="ORG",
            collection_id=entity1.collection_id,
        )

        relationship = await relationship_factory(
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            collection_id=entity1.collection_id,
            relationship_type="EMPLOYED_BY",
        )

        assert relationship.source_entity_id == entity1.id
        assert relationship.target_entity_id == entity2.id
        assert relationship.collection_id == entity1.collection_id


class TestCollectionWithGraph:
    """Tests for collection_with_graph fixture."""

    async def test_collection_with_graph_creates_graph_enabled_collection(
        self,
        db_session: AsyncSession,
        collection_with_graph,
    ) -> None:
        """collection_with_graph should create a collection with graph_enabled=True."""
        collection = await collection_with_graph()

        assert collection is not None
        assert collection.id is not None
        assert collection.graph_enabled is True

    async def test_collection_with_graph_accepts_custom_name(
        self,
        db_session: AsyncSession,
        collection_with_graph,
    ) -> None:
        """collection_with_graph should accept custom collection name."""
        collection = await collection_with_graph(name="Custom Graph Collection")

        assert collection.name == "Custom Graph Collection"


class TestGraphFactory:
    """Tests for graph_factory fixture."""

    async def test_graph_factory_creates_complete_graph(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """Graph factory should create a complete graph with entities and relationships."""
        graph = await graph_factory()

        assert graph is not None
        assert graph.collection is not None
        assert graph.collection.graph_enabled is True
        assert len(graph.entities) == 5  # Default
        assert len(graph.relationships) == 4  # Default is num_entities - 1

    async def test_graph_factory_with_custom_entity_count(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """Graph factory should accept custom entity count."""
        graph = await graph_factory(num_entities=10)

        assert len(graph.entities) == 10
        assert len(graph.relationships) == 9  # num_entities - 1

    async def test_graph_factory_with_custom_relationship_count(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """Graph factory should accept custom relationship count."""
        graph = await graph_factory(num_entities=5, num_relationships=2)

        assert len(graph.entities) == 5
        assert len(graph.relationships) == 2

    async def test_graph_factory_cycles_through_entity_types(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """Graph factory should cycle through entity types."""
        graph = await graph_factory(num_entities=6)

        # Default types are ["PERSON", "ORG", "GPE"]
        entity_types = [e.entity_type for e in graph.entities]
        assert entity_types == ["PERSON", "ORG", "GPE", "PERSON", "ORG", "GPE"]

    async def test_graph_factory_cycles_through_relationship_types(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """Graph factory should cycle through relationship types."""
        graph = await graph_factory(num_entities=5)

        # Default types are ["WORKS_FOR", "LOCATED_IN", "AFFILIATED_WITH"]
        rel_types = [r.relationship_type for r in graph.relationships]
        assert rel_types == ["WORKS_FOR", "LOCATED_IN", "AFFILIATED_WITH", "WORKS_FOR"]

    async def test_graph_factory_creates_chain_relationships(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """Graph factory should create relationships in a chain structure."""
        graph = await graph_factory(num_entities=4)

        # Relationships should be: 0->1, 1->2, 2->3
        assert graph.relationships[0].source_entity_id == graph.entities[0].id
        assert graph.relationships[0].target_entity_id == graph.entities[1].id
        assert graph.relationships[1].source_entity_id == graph.entities[1].id
        assert graph.relationships[1].target_entity_id == graph.entities[2].id
        assert graph.relationships[2].source_entity_id == graph.entities[2].id
        assert graph.relationships[2].target_entity_id == graph.entities[3].id

    async def test_graph_factory_entity_count_property(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """GraphTestData should have working entity_count property."""
        graph = await graph_factory(num_entities=7)

        assert graph.entity_count == 7

    async def test_graph_factory_relationship_count_property(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """GraphTestData should have working relationship_count property."""
        graph = await graph_factory(num_entities=5, num_relationships=3)

        assert graph.relationship_count == 3

    async def test_graph_factory_get_entity_by_name(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """GraphTestData should find entity by name (case-insensitive)."""
        graph = await graph_factory(num_entities=3)

        entity = graph.get_entity_by_name("Entity_0_PERSON")
        assert entity is not None
        assert entity.entity_type == "PERSON"

        # Case-insensitive
        entity_lower = graph.get_entity_by_name("entity_0_person")
        assert entity_lower is not None
        assert entity_lower.id == entity.id

    async def test_graph_factory_get_entity_by_name_not_found(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """GraphTestData should return None for non-existent entity name."""
        graph = await graph_factory(num_entities=3)

        entity = graph.get_entity_by_name("NonExistent")
        assert entity is None

    async def test_graph_factory_get_entities_by_type(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """GraphTestData should filter entities by type."""
        graph = await graph_factory(num_entities=6)

        persons = graph.get_entities_by_type("PERSON")
        assert len(persons) == 2
        assert all(e.entity_type == "PERSON" for e in persons)

    async def test_graph_factory_with_custom_types(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """Graph factory should accept custom entity and relationship types."""
        graph = await graph_factory(
            num_entities=4,
            entity_types=["PRODUCT", "COMPANY"],
            relationship_types=["MANUFACTURED_BY", "SOLD_BY"],
        )

        entity_types = [e.entity_type for e in graph.entities]
        assert entity_types == ["PRODUCT", "COMPANY", "PRODUCT", "COMPANY"]

        rel_types = [r.relationship_type for r in graph.relationships]
        assert rel_types == ["MANUFACTURED_BY", "SOLD_BY", "MANUFACTURED_BY"]

    async def test_graph_factory_no_relationships_with_single_entity(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """Graph factory should handle single entity (no relationships)."""
        graph = await graph_factory(num_entities=1)

        assert len(graph.entities) == 1
        assert len(graph.relationships) == 0

    async def test_graph_factory_zero_entities(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """Graph factory should handle zero entities."""
        graph = await graph_factory(num_entities=0)

        assert len(graph.entities) == 0
        assert len(graph.relationships) == 0
        assert graph.collection is not None  # Collection still created
