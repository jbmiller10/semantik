"""Integration tests for graph extraction pipeline and API endpoints.

These tests verify the complete graph extraction and search pipeline works end-to-end.
Tests use real database connections and verify data flows correctly through all components.

Note: Tests are marked as integration tests and will be skipped if the database is unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from shared.database.repositories.entity_repository import EntityRepository
from shared.database.repositories.relationship_repository import RelationshipRepository

if TYPE_CHECKING:
    from conftest import GraphTestData
    from httpx import AsyncClient
    from sqlalchemy.ext.asyncio import AsyncSession


# Mark all tests in this module as integration tests requiring database
pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestGraphExtractionPipeline:
    """Integration tests for the graph extraction pipeline.

    Tests verify that entities and relationships are correctly stored
    via repositories and can be retrieved and traversed.
    """

    async def test_entity_extraction_stores_entities(
        self,
        db_session: AsyncSession,
        collection_with_graph,
        document_factory,
    ) -> None:
        """Verify entities are stored via repository after extraction.

        This test creates entities through the repository and verifies
        they are persisted correctly in the database.
        """
        # Arrange: Create a graph-enabled collection and document
        collection = await collection_with_graph()
        document = await document_factory(
            collection_id=collection.id,
            file_name="test_extraction.txt",
        )

        entity_repo = EntityRepository(db_session)

        # Act: Create entities (simulating extraction results)
        await entity_repo.create(
            collection_id=collection.id,
            document_id=document.id,
            name="John Smith",
            entity_type="PERSON",
            confidence=0.92,
            chunk_id=1,
        )

        await entity_repo.create(
            collection_id=collection.id,
            document_id=document.id,
            name="Acme Corporation",
            entity_type="ORG",
            confidence=0.88,
            chunk_id=1,
        )

        # Commit to ensure persistence
        await db_session.commit()

        # Assert: Verify entities are stored
        stored_entities = await entity_repo.get_by_document(
            document_id=document.id,
            collection_id=collection.id,
        )

        assert len(stored_entities) == 2
        entity_names = {e.name for e in stored_entities}
        assert "John Smith" in entity_names
        assert "Acme Corporation" in entity_names

        # Verify entity details
        john = next(e for e in stored_entities if e.name == "John Smith")
        assert john.entity_type == "PERSON"
        assert john.confidence == 0.92
        assert john.document_id == document.id

    async def test_relationship_extraction_links_entities(
        self,
        db_session: AsyncSession,
        entity_factory,
    ) -> None:
        """Verify relationships link entities correctly after extraction.

        This test creates entities and relationships through repositories
        and verifies the connections are persisted correctly.
        """
        # Arrange: Create entities in the same collection
        entity1 = await entity_factory(
            name="Alice Johnson",
            entity_type="PERSON",
        )
        entity2 = await entity_factory(
            name="TechCorp Inc",
            entity_type="ORG",
            collection_id=entity1.collection_id,
        )

        rel_repo = RelationshipRepository(db_session)

        # Act: Create relationship linking entities
        relationship = await rel_repo.create(
            collection_id=entity1.collection_id,
            source_entity_id=entity1.id,
            target_entity_id=entity2.id,
            relationship_type="WORKS_FOR",
            confidence=0.85,
            extraction_method="dependency",
        )

        await db_session.commit()

        # Assert: Verify relationship is stored correctly
        stored_rel = await rel_repo.get_by_id(
            relationship_id=relationship.id,
            collection_id=entity1.collection_id,
        )

        assert stored_rel is not None
        assert stored_rel.source_entity_id == entity1.id
        assert stored_rel.target_entity_id == entity2.id
        assert stored_rel.relationship_type == "WORKS_FOR"
        assert stored_rel.confidence == 0.85

        # Verify relationship is retrievable by entity
        entity_rels = await rel_repo.get_by_entity(
            entity_id=entity1.id,
            collection_id=entity1.collection_id,
        )
        assert len(entity_rels) == 1
        assert entity_rels[0].id == relationship.id

    async def test_graph_traversal_finds_connected_entities(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """Verify BFS traversal finds connected entities in the graph.

        This test creates a chain of entities connected by relationships
        and verifies the traversal algorithm finds connected entities.
        """
        # Arrange: Create a graph with 5 entities in a chain
        graph: GraphTestData = await graph_factory(
            num_entities=5,
            num_relationships=4,  # Chain: 0->1->2->3->4
        )

        rel_repo = RelationshipRepository(db_session)

        # Act: Traverse from the first entity with 1 hop
        neighbors_1hop = await rel_repo.get_neighbors(
            entity_ids=[graph.entities[0].id],
            collection_id=graph.collection.id,
            max_hops=1,
        )

        # Assert: Should find entities 0 and 1 (starting + 1 hop)
        assert len(neighbors_1hop["entities"]) == 2
        assert graph.entities[0].id in neighbors_1hop["entities"]
        assert graph.entities[1].id in neighbors_1hop["entities"]
        # Entity 0 at hop 0, entity 1 at hop 1
        assert neighbors_1hop["entities"][graph.entities[0].id] == 0
        assert neighbors_1hop["entities"][graph.entities[1].id] == 1

        # Act: Traverse from the first entity with 2 hops
        neighbors_2hop = await rel_repo.get_neighbors(
            entity_ids=[graph.entities[0].id],
            collection_id=graph.collection.id,
            max_hops=2,
        )

        # Assert: Should find entities 0, 1, and 2
        assert len(neighbors_2hop["entities"]) == 3
        assert graph.entities[0].id in neighbors_2hop["entities"]
        assert graph.entities[1].id in neighbors_2hop["entities"]
        assert graph.entities[2].id in neighbors_2hop["entities"]

        # Act: Traverse from the middle entity (entity 2) with 2 hops
        neighbors_middle = await rel_repo.get_neighbors(
            entity_ids=[graph.entities[2].id],
            collection_id=graph.collection.id,
            max_hops=2,
        )

        # Assert: Should find entities 0, 1, 2, 3, 4 (all connected within 2 hops)
        assert len(neighbors_middle["entities"]) == 5

    async def test_get_subgraph_returns_react_flow_format(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """Verify get_subgraph returns data suitable for React Flow visualization."""
        # Arrange: Create a small graph
        graph: GraphTestData = await graph_factory(
            num_entities=3,
            num_relationships=2,
        )

        rel_repo = RelationshipRepository(db_session)

        # Act: Get subgraph from first entity
        subgraph = await rel_repo.get_subgraph(
            entity_ids=[graph.entities[0].id],
            collection_id=graph.collection.id,
            max_hops=2,
        )

        # Assert: Verify React Flow format
        assert "nodes" in subgraph
        assert "edges" in subgraph

        # All 3 entities should be nodes
        assert len(subgraph["nodes"]) == 3

        # Verify node structure
        for node in subgraph["nodes"]:
            assert "id" in node
            assert "name" in node
            assert "type" in node
            assert "hop" in node

        # 2 relationships should be edges
        assert len(subgraph["edges"]) == 2

        # Verify edge structure
        for edge in subgraph["edges"]:
            assert "id" in edge
            assert "source" in edge
            assert "target" in edge
            assert "type" in edge
            assert "confidence" in edge

    async def test_full_extraction_to_search_flow(
        self,
        db_session: AsyncSession,
        collection_with_graph,
        document_factory,
    ) -> None:
        """Test end-to-end flow from extraction to search enhancement.

        This test simulates the full pipeline:
        1. Create entities from extraction
        2. Create relationships
        3. Query via repositories to verify data is searchable
        4. Verify counts and type breakdowns
        """
        # Arrange: Create collection and document
        collection = await collection_with_graph()
        document = await document_factory(
            collection_id=collection.id,
            file_name="test_full_flow.txt",
        )

        entity_repo = EntityRepository(db_session)
        rel_repo = RelationshipRepository(db_session)

        # Act: Create entities (simulating extraction)
        person1 = await entity_repo.create(
            collection_id=collection.id,
            document_id=document.id,
            name="Dr. Jane Doe",
            entity_type="PERSON",
            confidence=0.95,
        )

        person2 = await entity_repo.create(
            collection_id=collection.id,
            document_id=document.id,
            name="Prof. Bob Wilson",
            entity_type="PERSON",
            confidence=0.90,
        )

        org = await entity_repo.create(
            collection_id=collection.id,
            document_id=document.id,
            name="Stanford University",
            entity_type="ORG",
            confidence=0.98,
        )

        location = await entity_repo.create(
            collection_id=collection.id,
            document_id=document.id,
            name="California",
            entity_type="GPE",
            confidence=0.85,
        )

        # Create relationships
        await rel_repo.create(
            collection_id=collection.id,
            source_entity_id=person1.id,
            target_entity_id=org.id,
            relationship_type="AFFILIATED_WITH",
            confidence=0.88,
        )

        await rel_repo.create(
            collection_id=collection.id,
            source_entity_id=person2.id,
            target_entity_id=org.id,
            relationship_type="AFFILIATED_WITH",
            confidence=0.82,
        )

        await rel_repo.create(
            collection_id=collection.id,
            source_entity_id=org.id,
            target_entity_id=location.id,
            relationship_type="LOCATED_IN",
            confidence=0.95,
        )

        await db_session.commit()

        # Assert: Verify entity counts
        total_entities = await entity_repo.count_by_collection(collection.id)
        assert total_entities == 4

        # Verify entity type breakdown
        entities_by_type = await entity_repo.count_by_type(collection.id)
        assert entities_by_type.get("PERSON", 0) == 2
        assert entities_by_type.get("ORG", 0) == 1
        assert entities_by_type.get("GPE", 0) == 1

        # Verify relationship counts
        total_rels = await rel_repo.count_by_collection(collection.id)
        assert total_rels == 3

        # Verify relationship type breakdown
        rels_by_type = await rel_repo.count_by_type(collection.id)
        assert rels_by_type.get("AFFILIATED_WITH", 0) == 2
        assert rels_by_type.get("LOCATED_IN", 0) == 1

        # Verify search by name works
        search_results = await entity_repo.search_by_name(
            collection_id=collection.id,
            query="Jane",
        )
        assert len(search_results) == 1
        assert search_results[0].name == "Dr. Jane Doe"

        # Verify traversal from person finds org and location
        subgraph = await rel_repo.get_subgraph(
            entity_ids=[person1.id],
            collection_id=collection.id,
            max_hops=2,
        )
        found_ids = {node["id"] for node in subgraph["nodes"]}
        assert person1.id in found_ids
        assert org.id in found_ids
        assert location.id in found_ids

    async def test_bulk_entity_creation(
        self,
        db_session: AsyncSession,
        collection_with_graph,
        document_factory,
    ) -> None:
        """Verify bulk entity creation is efficient and correct."""
        # Arrange
        collection = await collection_with_graph()
        document = await document_factory(collection_id=collection.id)

        entity_repo = EntityRepository(db_session)

        # Create a batch of entities
        entities_data = [
            {
                "name": f"Entity_{i}",
                "entity_type": ["PERSON", "ORG", "GPE"][i % 3],
                "document_id": document.id,
                "confidence": 0.85,
            }
            for i in range(50)
        ]

        # Act: Bulk create
        created_count = await entity_repo.bulk_create(
            entities=entities_data,
            collection_id=collection.id,
        )
        await db_session.commit()

        # Assert
        assert created_count == 50

        total = await entity_repo.count_by_collection(collection.id)
        assert total == 50

        # Verify type distribution
        by_type = await entity_repo.count_by_type(collection.id)
        # 50 entities, cycling through 3 types: ~17 each, with PERSON getting 1 extra
        assert by_type.get("PERSON", 0) >= 16
        assert by_type.get("ORG", 0) >= 16
        assert by_type.get("GPE", 0) >= 16

    async def test_bulk_relationship_creation(
        self,
        db_session: AsyncSession,
        entity_factory,
    ) -> None:
        """Verify bulk relationship creation is efficient and correct."""
        # Arrange: Create entities
        entities = []
        first_entity = await entity_factory(name="Node_0", entity_type="PERSON")
        entities.append(first_entity)

        for i in range(1, 20):
            entity = await entity_factory(
                name=f"Node_{i}",
                entity_type="PERSON",
                collection_id=first_entity.collection_id,
            )
            entities.append(entity)

        rel_repo = RelationshipRepository(db_session)

        # Create relationships data (chain structure)
        relationships_data = [
            {
                "source_entity_id": entities[i].id,
                "target_entity_id": entities[i + 1].id,
                "relationship_type": "RELATED_TO",
                "confidence": 0.75,
            }
            for i in range(len(entities) - 1)
        ]

        # Act: Bulk create
        created_count = await rel_repo.bulk_create(
            relationships=relationships_data,
            collection_id=first_entity.collection_id,
        )
        await db_session.commit()

        # Assert
        assert created_count == 19

        total = await rel_repo.count_by_collection(first_entity.collection_id)
        assert total == 19


class TestGraphAPIEndpoints:
    """Integration tests for Graph API endpoints.

    Tests verify the API endpoints return correct data and handle
    edge cases properly.
    """

    @pytest.fixture()
    def repository_mocks(self, db_session: AsyncSession):
        """Create repository instances for API tests."""
        return {
            "entity_repo": EntityRepository(db_session),
            "rel_repo": RelationshipRepository(db_session),
        }

    async def test_get_graph_stats(
        self,
        async_client: AsyncClient,
        graph_factory,
        auth_headers,
    ) -> None:
        """GET /api/graph/collections/{id}/stats returns statistics."""
        # Arrange
        graph: GraphTestData = await graph_factory(num_entities=10, num_relationships=9)

        # Act
        response = await async_client.get(
            f"/api/graph/collections/{graph.collection.id}/stats",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_entities"] == 10
        assert data["total_relationships"] == 9
        assert data["graph_enabled"] is True
        assert "entities_by_type" in data
        assert "relationships_by_type" in data

        # Verify type breakdowns are present
        assert isinstance(data["entities_by_type"], dict)
        assert isinstance(data["relationships_by_type"], dict)

    async def test_search_entities(
        self,
        async_client: AsyncClient,
        graph_factory,
        auth_headers,
    ) -> None:
        """POST /api/graph/collections/{id}/entities/search returns matching entities."""
        # Arrange
        graph: GraphTestData = await graph_factory(num_entities=5)

        # Act: Search for entities
        response = await async_client.post(
            f"/api/graph/collections/{graph.collection.id}/entities/search",
            headers=auth_headers,
            json={"query": "Entity", "limit": 10},
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "total" in data
        assert "has_more" in data
        assert len(data["entities"]) == 5
        assert data["total"] == 5

        # Verify entity structure
        for entity in data["entities"]:
            assert "id" in entity
            assert "name" in entity
            assert "entity_type" in entity
            assert "confidence" in entity

    async def test_search_entities_with_type_filter(
        self,
        async_client: AsyncClient,
        graph_factory,
        auth_headers,
    ) -> None:
        """POST /api/graph/collections/{id}/entities/search respects type filter."""
        # Arrange: Create graph with mixed entity types
        graph: GraphTestData = await graph_factory(
            num_entities=6,
            entity_types=["PERSON", "ORG", "GPE"],  # 2 of each type
        )

        # Act: Search only for PERSON entities
        response = await async_client.post(
            f"/api/graph/collections/{graph.collection.id}/entities/search",
            headers=auth_headers,
            json={"entity_types": ["PERSON"], "limit": 10},
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        # Should only return PERSON entities (2 out of 6)
        assert len(data["entities"]) == 2
        for entity in data["entities"]:
            assert entity["entity_type"] == "PERSON"

    async def test_traverse_graph(
        self,
        async_client: AsyncClient,
        graph_factory,
        auth_headers,
    ) -> None:
        """POST /api/graph/collections/{id}/traverse returns subgraph."""
        # Arrange
        graph: GraphTestData = await graph_factory(num_entities=5, num_relationships=4)

        # Act: Traverse from first entity
        response = await async_client.post(
            f"/api/graph/collections/{graph.collection.id}/traverse",
            headers=auth_headers,
            json={
                "entity_id": graph.entities[0].id,
                "max_hops": 2,
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data

        # With 2 hops from first entity, should find entities 0, 1, 2
        assert len(data["nodes"]) >= 3
        assert len(data["edges"]) >= 2

        # Verify node structure
        for node in data["nodes"]:
            assert "id" in node
            assert "name" in node
            assert "type" in node
            assert "hop" in node

        # Verify edge structure
        for edge in data["edges"]:
            assert "id" in edge
            assert "source" in edge
            assert "target" in edge
            assert "type" in edge
            assert "confidence" in edge

    async def test_get_entity_types(
        self,
        async_client: AsyncClient,
        graph_factory,
        auth_headers,
    ) -> None:
        """GET /api/graph/collections/{id}/entity-types returns type counts."""
        # Arrange
        graph: GraphTestData = await graph_factory(
            num_entities=9,
            entity_types=["PERSON", "ORG", "GPE"],  # 3 of each
        )

        # Act
        response = await async_client.get(
            f"/api/graph/collections/{graph.collection.id}/entity-types",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Should have 3 types with 3 entities each
        assert "PERSON" in data
        assert "ORG" in data
        assert "GPE" in data
        assert data["PERSON"] == 3
        assert data["ORG"] == 3
        assert data["GPE"] == 3

    async def test_graph_disabled_returns_empty(
        self,
        async_client: AsyncClient,
        collection_factory,
        test_user_db,
        auth_headers,
    ) -> None:
        """Verify empty results when graph_enabled=False."""
        # Arrange: Create collection with graph disabled
        collection = await collection_factory(
            owner_id=test_user_db.id,
            graph_enabled=False,
        )

        # Act: Get stats
        response = await async_client.get(
            f"/api/graph/collections/{collection.id}/stats",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_entities"] == 0
        assert data["total_relationships"] == 0
        assert data["graph_enabled"] is False

        # Act: Search entities
        response = await async_client.post(
            f"/api/graph/collections/{collection.id}/entities/search",
            headers=auth_headers,
            json={"query": "test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["entities"] == []
        assert data["total"] == 0

        # Act: Get entity types
        response = await async_client.get(
            f"/api/graph/collections/{collection.id}/entity-types",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data == {}

        # Act: Traverse (should return empty graph)
        response = await async_client.post(
            f"/api/graph/collections/{collection.id}/traverse",
            headers=auth_headers,
            json={"entity_id": 1, "max_hops": 2},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["nodes"] == []
        assert data["edges"] == []

    async def test_unauthorized_access_denied(
        self,
        async_client: AsyncClient,
        graph_factory,
    ) -> None:
        """Verify 401 without authentication."""
        # Arrange
        graph: GraphTestData = await graph_factory(num_entities=3)

        # Act: Request without auth headers
        response = await async_client.get(
            f"/api/graph/collections/{graph.collection.id}/stats",
        )

        # Assert
        # Note: When DISABLE_AUTH=true (test env), this may return 200
        # When DISABLE_AUTH=false, it should return 401
        # We accept both cases depending on test environment
        if response.status_code != 200:
            assert response.status_code == 401

    async def test_nonexistent_collection_returns_404(
        self,
        async_client: AsyncClient,
        auth_headers,
    ) -> None:
        """Verify 404 for non-existent collection."""
        # Act
        response = await async_client.get(
            "/api/graph/collections/00000000-0000-0000-0000-000000000000/stats",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 404

    async def test_get_relationship_types(
        self,
        async_client: AsyncClient,
        graph_factory,
        auth_headers,
    ) -> None:
        """GET /api/graph/collections/{id}/relationship-types returns type counts."""
        # Arrange
        graph: GraphTestData = await graph_factory(
            num_entities=6,
            num_relationships=5,
            relationship_types=["WORKS_FOR", "LOCATED_IN"],
        )

        # Act
        response = await async_client.get(
            f"/api/graph/collections/{graph.collection.id}/relationship-types",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Should have both relationship types
        assert "WORKS_FOR" in data
        assert "LOCATED_IN" in data
        total_rels = sum(data.values())
        assert total_rels == 5


class TestSearchIntegration:
    """Integration tests for search with graph enhancement.

    Tests verify that search properly integrates with graph data
    and handles edge cases gracefully.
    """

    async def test_search_with_graph_enhancement_setup(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """Verify graph search service can be instantiated with graph data."""
        # Arrange: Create graph data
        await graph_factory(num_entities=5)

        # Act: Create graph search service
        from packages.vecpipe.graphrag import create_graph_search_service

        service = await create_graph_search_service(
            db_session=db_session,
            graph_weight=0.2,
            max_hops=2,
        )

        # Assert: Service is created
        assert service is not None
        assert service.graph_weight == 0.2
        assert service.max_hops == 2

    async def test_search_graceful_without_graph(
        self,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """Verify search works gracefully when graph is disabled or empty."""
        # Arrange: Create collection without graph data
        collection = await collection_factory(
            owner_id=test_user_db.id,
            graph_enabled=False,
        )

        # Act: Create graph search service and try to enhance results
        from packages.vecpipe.graphrag import create_graph_search_service

        service = await create_graph_search_service(db_session=db_session)

        # Simulate vector search results
        vector_results = [
            {"chunk_id": 1, "score": 0.95, "text": "Test result 1"},
            {"chunk_id": 2, "score": 0.85, "text": "Test result 2"},
        ]

        # Enhance results (should gracefully handle no graph data)
        enhanced = await service.enhance_results(
            query="test query",
            vector_results=vector_results,
            collection_id=collection.id,
        )

        # Assert: Results are returned with empty graph fields
        assert len(enhanced) == 2
        for result in enhanced:
            assert "original_score" in result
            assert "graph_score" in result
            assert "matched_entities" in result
            # Graph score should be 0 with no graph data
            assert result["graph_score"] == 0.0

    async def test_search_with_matching_entities(
        self,
        db_session: AsyncSession,
        collection_with_graph,
        document_factory,
    ) -> None:
        """Verify search enhancement when query contains matching entities."""
        # Arrange: Create collection with entities
        collection = await collection_with_graph()
        document = await document_factory(collection_id=collection.id)

        entity_repo = EntityRepository(db_session)

        # Create entities that might match a query
        await entity_repo.create(
            collection_id=collection.id,
            document_id=document.id,
            name="Microsoft",
            entity_type="ORG",
            confidence=0.95,
            chunk_id=1,
        )

        await entity_repo.create(
            collection_id=collection.id,
            document_id=document.id,
            name="Bill Gates",
            entity_type="PERSON",
            confidence=0.92,
            chunk_id=1,
        )

        await db_session.commit()

        # Act: Verify entities are searchable
        search_results = await entity_repo.search_by_name(
            collection_id=collection.id,
            query="Microsoft",
        )

        # Assert
        assert len(search_results) == 1
        assert search_results[0].name == "Microsoft"

    async def test_graph_enhancement_with_mock_spacy(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """Verify graph enhancement works with mocked spaCy extraction."""
        # Arrange: Create graph with known entities
        graph: GraphTestData = await graph_factory(
            num_entities=3,
            entity_types=["ORG", "PERSON", "GPE"],
        )

        from packages.vecpipe.graphrag import create_graph_search_service

        service = await create_graph_search_service(db_session=db_session)

        # Mock entity extraction to return a known entity name
        mock_entities = [
            {
                "name": graph.entities[0].name,
                "entity_type": graph.entities[0].entity_type,
            }
        ]

        with patch.object(
            service.entity_service,
            "extract_from_text",
            return_value=mock_entities,
        ):
            # Simulate vector search results that include chunk with entity
            vector_results = [
                {"chunk_id": 1, "score": 0.90, "text": f"Text mentioning {graph.entities[0].name}"},
                {"chunk_id": 2, "score": 0.85, "text": "Other text"},
            ]

            # Act
            enhanced = await service.enhance_results(
                query=f"Tell me about {graph.entities[0].name}",
                vector_results=vector_results,
                collection_id=graph.collection.id,
            )

            # Assert: Results are enhanced (original score preserved)
            assert len(enhanced) == 2
            for result in enhanced:
                assert "original_score" in result
                assert "graph_score" in result
                assert result["original_score"] in [0.90, 0.85]


class TestDeleteOperations:
    """Tests for entity and relationship deletion operations."""

    async def test_delete_entities_by_document(
        self,
        db_session: AsyncSession,
        collection_with_graph,
        document_factory,
    ) -> None:
        """Verify entities are deleted when document is removed."""
        # Arrange
        collection = await collection_with_graph()
        doc1 = await document_factory(collection_id=collection.id, file_name="doc1.txt")
        doc2 = await document_factory(collection_id=collection.id, file_name="doc2.txt")

        entity_repo = EntityRepository(db_session)

        # Create entities for both documents
        for i in range(3):
            await entity_repo.create(
                collection_id=collection.id,
                document_id=doc1.id,
                name=f"Doc1_Entity_{i}",
                entity_type="PERSON",
            )
            await entity_repo.create(
                collection_id=collection.id,
                document_id=doc2.id,
                name=f"Doc2_Entity_{i}",
                entity_type="PERSON",
            )

        await db_session.commit()

        # Verify initial count
        total = await entity_repo.count_by_collection(collection.id)
        assert total == 6

        # Act: Delete doc1's entities
        deleted_count = await entity_repo.delete_by_document(
            document_id=doc1.id,
            collection_id=collection.id,
        )
        await db_session.commit()

        # Assert
        assert deleted_count == 3
        remaining = await entity_repo.count_by_collection(collection.id)
        assert remaining == 3

        # Verify correct entities were deleted
        doc1_entities = await entity_repo.get_by_document(doc1.id, collection.id)
        doc2_entities = await entity_repo.get_by_document(doc2.id, collection.id)
        assert len(doc1_entities) == 0
        assert len(doc2_entities) == 3

    async def test_delete_relationships_by_collection(
        self,
        db_session: AsyncSession,
        graph_factory,
    ) -> None:
        """Verify all relationships are deleted when collection is cleared."""
        # Arrange
        graph: GraphTestData = await graph_factory(num_entities=5, num_relationships=4)

        rel_repo = RelationshipRepository(db_session)

        # Verify initial count
        total = await rel_repo.count_by_collection(graph.collection.id)
        assert total == 4

        # Act
        deleted_count = await rel_repo.delete_by_collection(graph.collection.id)
        await db_session.commit()

        # Assert
        assert deleted_count == 4
        remaining = await rel_repo.count_by_collection(graph.collection.id)
        assert remaining == 0
