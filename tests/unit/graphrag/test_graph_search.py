"""Unit tests for GraphEnhancedSearchService.

Tests graph-enhanced search service logic with mocked repositories.
All tests use mocks - no real database or spaCy required.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestGraphEnhancedSearchService:
    """Tests for GraphEnhancedSearchService."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        return AsyncMock()

    @pytest.fixture
    def mock_entity_repo(self):
        """Create a mock EntityRepository."""
        repo = AsyncMock()
        repo.search_by_name = AsyncMock(return_value=[])
        repo.get_by_chunk = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def mock_rel_repo(self):
        """Create a mock RelationshipRepository."""
        repo = AsyncMock()
        repo.get_neighbors = AsyncMock(return_value={
            "entities": {},
            "relationships": [],
        })
        return repo

    @pytest.fixture
    def mock_entity_service(self):
        """Create a mock EntityExtractionService."""
        service = AsyncMock()
        service.extract_from_text = AsyncMock(return_value=[])
        return service

    @pytest.fixture
    def search_service(
        self, mock_db_session, mock_entity_repo, mock_rel_repo, mock_entity_service
    ):
        """Create GraphEnhancedSearchService with mocked dependencies."""
        with patch(
            "packages.vecpipe.graphrag.search.EntityRepository",
            return_value=mock_entity_repo,
        ), patch(
            "packages.vecpipe.graphrag.search.RelationshipRepository",
            return_value=mock_rel_repo,
        ), patch(
            "packages.vecpipe.graphrag.search.EntityExtractionService",
            return_value=mock_entity_service,
        ):
            from packages.vecpipe.graphrag.search import GraphEnhancedSearchService

            service = GraphEnhancedSearchService(
                db_session=mock_db_session,
                graph_weight=0.2,
                max_hops=2,
            )
            # Attach mocks for test access
            service.entity_repo = mock_entity_repo
            service.rel_repo = mock_rel_repo
            service.entity_service = mock_entity_service
            return service

    @pytest.mark.asyncio
    async def test_enhance_results_empty_input_returns_empty(self, search_service):
        """Should return empty list when vector_results is empty."""
        results = await search_service.enhance_results(
            query="test query",
            vector_results=[],
            collection_id="col-1",
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_enhance_results_no_query_entities(self, search_service):
        """Should return original results with empty graph fields when no entities in query."""
        # Entity service returns no entities
        search_service.entity_service.extract_from_text.return_value = []

        vector_results = [
            {"chunk_id": 1, "score": 0.9, "text": "Test chunk 1"},
            {"chunk_id": 2, "score": 0.8, "text": "Test chunk 2"},
        ]

        results = await search_service.enhance_results(
            query="some query without entities",
            vector_results=vector_results,
            collection_id="col-1",
        )

        assert len(results) == 2
        # Check graph fields are added
        assert results[0]["original_score"] == 0.9
        assert results[0]["graph_score"] == 0.0
        assert results[0]["matched_entities"] == []
        assert results[1]["original_score"] == 0.8
        assert results[1]["graph_score"] == 0.0
        assert results[1]["matched_entities"] == []

    @pytest.mark.asyncio
    async def test_enhance_results_no_matching_entities_in_collection(
        self, search_service
    ):
        """Should return original results when query entities don't match any in collection."""
        # Query has entities
        search_service.entity_service.extract_from_text.return_value = [
            {"name": "John Smith", "entity_type": "PERSON"}
        ]
        # But no matches in collection
        search_service.entity_repo.search_by_name.return_value = []

        vector_results = [
            {"chunk_id": 1, "score": 0.9, "text": "Test chunk"},
        ]

        results = await search_service.enhance_results(
            query="John Smith",
            vector_results=vector_results,
            collection_id="col-1",
        )

        assert len(results) == 1
        assert results[0]["original_score"] == 0.9
        assert results[0]["graph_score"] == 0.0
        assert results[0]["matched_entities"] == []

    @pytest.mark.asyncio
    async def test_enhance_results_with_matching_entities(self, search_service):
        """Should boost results containing entities matching query entities."""
        # Query has entities
        search_service.entity_service.extract_from_text.return_value = [
            {"name": "John Smith", "entity_type": "PERSON"}
        ]

        # Collection has matching entity
        mock_entity = MagicMock()
        mock_entity.id = 1
        mock_entity.name = "John Smith"
        mock_entity.entity_type = "PERSON"
        search_service.entity_repo.search_by_name.return_value = [mock_entity]

        # Graph expansion returns same entity
        search_service.rel_repo.get_neighbors.return_value = {
            "entities": {1: 0},  # entity_id: hop_distance
            "relationships": [],
        }

        # Chunk contains the entity
        search_service.entity_repo.get_by_chunk.return_value = [mock_entity]

        vector_results = [
            {"chunk_id": 1, "score": 0.8, "text": "John Smith is mentioned here"},
        ]

        results = await search_service.enhance_results(
            query="John Smith",
            vector_results=vector_results,
            collection_id="col-1",
        )

        assert len(results) == 1
        # Graph score should be positive
        assert results[0]["graph_score"] > 0
        # Combined score should reflect boost
        assert results[0]["original_score"] == 0.8
        # matched_entities should contain the entity
        assert len(results[0]["matched_entities"]) >= 1

    def test_names_match_exact(self, search_service):
        """Should match identical names (case-insensitive)."""
        assert search_service._names_match("John Smith", "John Smith")
        assert search_service._names_match("john smith", "John Smith")
        assert search_service._names_match("JOHN SMITH", "john smith")

    def test_names_match_with_whitespace(self, search_service):
        """Should handle whitespace in names."""
        assert search_service._names_match("John Smith", "  John Smith  ")
        assert search_service._names_match("  john  ", "John")

    def test_names_match_partial_contains(self, search_service):
        """Should match when one name contains the other."""
        assert search_service._names_match("John", "John Smith")
        assert search_service._names_match("John Smith", "John")
        assert search_service._names_match("Microsoft", "Microsoft Corporation")
        assert search_service._names_match("Microsoft Corporation", "Microsoft")

    def test_names_no_match_different(self, search_service):
        """Should not match unrelated names."""
        assert not search_service._names_match("John", "Jane")
        assert not search_service._names_match("Apple", "Microsoft")
        assert not search_service._names_match("Seattle", "Boston")

    def test_score_results_no_graph_matches(self, search_service):
        """Should calculate combined score with zero graph contribution."""
        vector_results = [
            {"chunk_id": 1, "score": 0.8, "text": "Test"},
        ]

        results = search_service._score_results(
            vector_results=vector_results,
            related_entity_ids=set(),
            direct_match_ids=set(),
            chunk_entity_map={1: []},
            weight=0.2,
        )

        assert len(results) == 1
        # With no graph matches: (1 - 0.2) * 0.8 + 0.2 * 0 = 0.64
        expected_score = 0.8 * 0.8  # (1 - weight) * vector_score
        assert abs(results[0]["score"] - expected_score) < 0.01
        assert results[0]["graph_score"] == 0.0
        assert results[0]["original_score"] == 0.8

    def test_score_results_with_direct_matches(self, search_service):
        """Should boost score for chunks with directly matched entities."""
        vector_results = [
            {"chunk_id": 1, "score": 0.8, "text": "Test"},
        ]

        # Chunk contains entity 1 which is a direct match
        chunk_entity_map = {
            1: [{"id": 1, "name": "John", "type": "PERSON"}]
        }

        results = search_service._score_results(
            vector_results=vector_results,
            related_entity_ids={1},
            direct_match_ids={1},
            chunk_entity_map=chunk_entity_map,
            weight=0.2,
        )

        assert len(results) == 1
        # Direct match adds 0.3 to graph_score
        assert results[0]["graph_score"] == 0.3
        # matched_entities should include the entity
        assert len(results[0]["matched_entities"]) == 1
        assert results[0]["matched_entities"][0]["name"] == "John"
        assert results[0]["matched_entities"][0]["direct_match"] is True

    def test_score_results_with_related_matches(self, search_service):
        """Should boost score for chunks with related (but not direct) entities."""
        vector_results = [
            {"chunk_id": 1, "score": 0.8, "text": "Test"},
        ]

        # Chunk contains entity 2 which is related but not direct
        chunk_entity_map = {
            1: [{"id": 2, "name": "Microsoft", "type": "ORG"}]
        }

        results = search_service._score_results(
            vector_results=vector_results,
            related_entity_ids={1, 2},  # 2 is related
            direct_match_ids={1},  # But 1 is direct, not 2
            chunk_entity_map=chunk_entity_map,
            weight=0.2,
        )

        assert len(results) == 1
        # Related match adds 0.1 to graph_score
        assert results[0]["graph_score"] == 0.1
        assert len(results[0]["matched_entities"]) == 1
        assert results[0]["matched_entities"][0]["direct_match"] is False

    def test_score_results_combined_matches(self, search_service):
        """Should correctly combine direct and related match scores."""
        vector_results = [
            {"chunk_id": 1, "score": 0.8, "text": "Test"},
        ]

        # Chunk contains both direct and related entities
        chunk_entity_map = {
            1: [
                {"id": 1, "name": "John", "type": "PERSON"},
                {"id": 2, "name": "Microsoft", "type": "ORG"},
            ]
        }

        results = search_service._score_results(
            vector_results=vector_results,
            related_entity_ids={1, 2},
            direct_match_ids={1},  # Only entity 1 is direct
            chunk_entity_map=chunk_entity_map,
            weight=0.2,
        )

        assert len(results) == 1
        # Direct match: 0.3, Related match (excluding direct): 0.1
        assert results[0]["graph_score"] == 0.4
        assert len(results[0]["matched_entities"]) == 2

    def test_score_results_caps_at_one(self, search_service):
        """Graph score should be capped at 1.0."""
        vector_results = [
            {"chunk_id": 1, "score": 0.8, "text": "Test"},
        ]

        # Many direct matches
        chunk_entity_map = {
            1: [
                {"id": i, "name": f"Entity{i}", "type": "PERSON"}
                for i in range(10)
            ]
        }

        results = search_service._score_results(
            vector_results=vector_results,
            related_entity_ids=set(range(10)),
            direct_match_ids=set(range(10)),
            chunk_entity_map=chunk_entity_map,
            weight=0.2,
        )

        # 10 direct matches would give 3.0 but should cap at 1.0
        assert results[0]["graph_score"] == 1.0

    def test_add_empty_graph_fields(self, search_service):
        """Should add empty graph fields to all results."""
        results = [
            {"chunk_id": 1, "score": 0.9, "text": "Test 1"},
            {"chunk_id": 2, "score": 0.8, "text": "Test 2"},
        ]

        enhanced = search_service._add_empty_graph_fields(results)

        assert len(enhanced) == 2

        assert enhanced[0]["original_score"] == 0.9
        assert enhanced[0]["graph_score"] == 0.0
        assert enhanced[0]["matched_entities"] == []
        assert enhanced[0]["chunk_id"] == 1  # Original fields preserved

        assert enhanced[1]["original_score"] == 0.8
        assert enhanced[1]["graph_score"] == 0.0
        assert enhanced[1]["matched_entities"] == []

    def test_add_empty_graph_fields_preserves_original(self, search_service):
        """Should preserve all original result fields."""
        results = [
            {
                "chunk_id": 1,
                "score": 0.9,
                "text": "Test content",
                "custom_field": "value",
            },
        ]

        enhanced = search_service._add_empty_graph_fields(results)

        assert enhanced[0]["chunk_id"] == 1
        assert enhanced[0]["text"] == "Test content"
        assert enhanced[0]["custom_field"] == "value"

    def test_constructor_defaults(self, mock_db_session):
        """Should have sensible default values."""
        with patch(
            "packages.vecpipe.graphrag.search.EntityRepository"
        ), patch(
            "packages.vecpipe.graphrag.search.RelationshipRepository"
        ), patch(
            "packages.vecpipe.graphrag.search.EntityExtractionService"
        ):
            from packages.vecpipe.graphrag.search import GraphEnhancedSearchService

            service = GraphEnhancedSearchService(db_session=mock_db_session)

            assert service.graph_weight == 0.2
            assert service.max_hops == 2

    def test_constructor_custom_values(self, mock_db_session):
        """Should accept custom graph_weight and max_hops."""
        with patch(
            "packages.vecpipe.graphrag.search.EntityRepository"
        ), patch(
            "packages.vecpipe.graphrag.search.RelationshipRepository"
        ), patch(
            "packages.vecpipe.graphrag.search.EntityExtractionService"
        ):
            from packages.vecpipe.graphrag.search import GraphEnhancedSearchService

            service = GraphEnhancedSearchService(
                db_session=mock_db_session,
                graph_weight=0.5,
                max_hops=3,
            )

            assert service.graph_weight == 0.5
            assert service.max_hops == 3

    @pytest.mark.asyncio
    async def test_enhance_results_override_weight(self, search_service):
        """Should allow overriding graph_weight in enhance_results call."""
        search_service.entity_service.extract_from_text.return_value = []

        vector_results = [
            {"chunk_id": 1, "score": 0.9, "text": "Test"},
        ]

        # Call with custom weight override
        results = await search_service.enhance_results(
            query="test",
            vector_results=vector_results,
            collection_id="col-1",
            graph_weight=0.5,
        )

        # Results should still be returned (weight doesn't affect empty graph case)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_enhance_results_override_max_hops(self, search_service):
        """Should allow overriding max_hops in enhance_results call."""
        search_service.entity_service.extract_from_text.return_value = []

        vector_results = [
            {"chunk_id": 1, "score": 0.9, "text": "Test"},
        ]

        # Call with custom max_hops override
        results = await search_service.enhance_results(
            query="test",
            vector_results=vector_results,
            collection_id="col-1",
            max_hops=5,
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_enhance_results_graceful_degradation(self, search_service):
        """Should return original results if graph enhancement fails."""
        # Make entity extraction fail
        search_service.entity_service.extract_from_text.side_effect = Exception(
            "Extraction failed"
        )

        vector_results = [
            {"chunk_id": 1, "score": 0.9, "text": "Test"},
        ]

        results = await search_service.enhance_results(
            query="test",
            vector_results=vector_results,
            collection_id="col-1",
        )

        # Should return original results with empty graph fields
        assert len(results) == 1
        assert results[0]["original_score"] == 0.9
        assert results[0]["graph_score"] == 0.0

    @pytest.mark.asyncio
    async def test_enhance_results_sorts_by_combined_score(self, search_service):
        """Should sort results by combined score in descending order."""
        # Setup for boosting second result
        search_service.entity_service.extract_from_text.return_value = [
            {"name": "Target", "entity_type": "PERSON"}
        ]

        mock_entity = MagicMock()
        mock_entity.id = 1
        mock_entity.name = "Target"
        mock_entity.entity_type = "PERSON"
        search_service.entity_repo.search_by_name.return_value = [mock_entity]

        search_service.rel_repo.get_neighbors.return_value = {
            "entities": {1: 0},
            "relationships": [],
        }

        # Only chunk 2 contains the entity
        async def mock_get_by_chunk(chunk_id, collection_id):
            if chunk_id == 2:
                return [mock_entity]
            return []

        search_service.entity_repo.get_by_chunk.side_effect = mock_get_by_chunk

        vector_results = [
            {"chunk_id": 1, "score": 0.9, "text": "No entity here"},
            {"chunk_id": 2, "score": 0.7, "text": "Target is here"},
        ]

        results = await search_service.enhance_results(
            query="Target",
            vector_results=vector_results,
            collection_id="col-1",
        )

        assert len(results) == 2
        # Second result (originally 0.7) should be boosted and potentially higher
        # Results should be sorted by combined score
        assert results[0]["score"] >= results[1]["score"]


class TestCreateGraphSearchService:
    """Tests for the factory function."""

    @pytest.mark.asyncio
    async def test_create_graph_search_service(self):
        """Should create a GraphEnhancedSearchService with defaults."""
        mock_session = AsyncMock()

        with patch(
            "packages.vecpipe.graphrag.search.EntityRepository"
        ), patch(
            "packages.vecpipe.graphrag.search.RelationshipRepository"
        ), patch(
            "packages.vecpipe.graphrag.search.EntityExtractionService"
        ):
            from packages.vecpipe.graphrag.search import create_graph_search_service

            service = await create_graph_search_service(db_session=mock_session)

            assert service.graph_weight == 0.2
            assert service.max_hops == 2

    @pytest.mark.asyncio
    async def test_create_graph_search_service_custom(self):
        """Should create a GraphEnhancedSearchService with custom values."""
        mock_session = AsyncMock()

        with patch(
            "packages.vecpipe.graphrag.search.EntityRepository"
        ), patch(
            "packages.vecpipe.graphrag.search.RelationshipRepository"
        ), patch(
            "packages.vecpipe.graphrag.search.EntityExtractionService"
        ):
            from packages.vecpipe.graphrag.search import create_graph_search_service

            service = await create_graph_search_service(
                db_session=mock_session,
                graph_weight=0.4,
                max_hops=4,
            )

            assert service.graph_weight == 0.4
            assert service.max_hops == 4
