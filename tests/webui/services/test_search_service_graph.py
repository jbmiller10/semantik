"""
Tests for SearchService graph enhancement functionality.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.database.models import Collection, CollectionStatus
from webui.services.search_service import SearchService


@pytest.fixture()
def search_service(mock_db_session: AsyncMock, mock_collection_repo: AsyncMock) -> SearchService:
    """Create a SearchService instance with mocked dependencies."""
    return SearchService(db_session=mock_db_session, collection_repo=mock_collection_repo)


@pytest.fixture()
def mock_collection_graph_enabled() -> MagicMock:
    """Mock collection with graph_enabled=True."""
    collection = MagicMock(spec=Collection)
    collection.id = "123e4567-e89b-12d3-a456-426614174000"
    collection.name = "Graph Enabled Collection"
    collection.vector_store_name = "qdrant_graph_collection"
    collection.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
    collection.quantization = "float16"
    collection.status = CollectionStatus.READY
    collection.graph_enabled = True
    return collection


@pytest.fixture()
def mock_collection_graph_disabled() -> MagicMock:
    """Mock collection with graph_enabled=False."""
    collection = MagicMock(spec=Collection)
    collection.id = "456e7890-e89b-12d3-a456-426614174001"
    collection.name = "Standard Collection"
    collection.vector_store_name = "qdrant_standard_collection"
    collection.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
    collection.quantization = "float16"
    collection.status = CollectionStatus.READY
    collection.graph_enabled = False
    return collection


class TestSearchWithGraphEnhancement:
    """Test SearchService graph enhancement functionality."""

    @pytest.mark.asyncio()
    async def test_graph_enhancement_skipped_when_disabled(
        self,
        search_service: SearchService,
        mock_collection_graph_disabled: MagicMock,
    ) -> None:
        """Graph enhancement should be skipped when collection has graph_enabled=False."""
        results = [
            {"chunk_id": 1, "score": 0.9, "text": "Test content 1"},
            {"chunk_id": 2, "score": 0.8, "text": "Test content 2"},
        ]

        enhanced = await search_service.search_with_graph_enhancement(
            results=results,
            query="test query",
            collection=mock_collection_graph_disabled,
        )

        # Should return original results unchanged
        assert enhanced == results
        assert len(enhanced) == 2
        assert enhanced[0]["score"] == 0.9
        # No graph fields added when enhancement skipped
        assert "original_score" not in enhanced[0]
        assert "graph_score" not in enhanced[0]

    @pytest.mark.asyncio()
    async def test_graph_enhancement_skipped_for_empty_results(
        self,
        search_service: SearchService,
        mock_collection_graph_enabled: MagicMock,
    ) -> None:
        """Graph enhancement should be skipped when there are no results."""
        results: list[dict] = []

        enhanced = await search_service.search_with_graph_enhancement(
            results=results,
            query="test query",
            collection=mock_collection_graph_enabled,
        )

        assert enhanced == []

    @pytest.mark.asyncio()
    async def test_graph_enhancement_applied_when_enabled(
        self,
        search_service: SearchService,
        mock_collection_graph_enabled: MagicMock,
    ) -> None:
        """Graph enhancement should be applied when collection has graph_enabled=True."""
        results = [
            {"chunk_id": 1, "score": 0.9, "text": "John works at Microsoft"},
            {"chunk_id": 2, "score": 0.8, "text": "The project was completed"},
        ]

        with patch("webui.services.search_service.GraphEnhancedSearchService") as mock_graph_service:
            mock_instance = AsyncMock()
            mock_instance.enhance_results.return_value = [
                {
                    "chunk_id": 1,
                    "score": 0.92,
                    "original_score": 0.9,
                    "graph_score": 0.3,
                    "text": "John works at Microsoft",
                    "matched_entities": [
                        {"name": "John", "type": "PERSON", "direct_match": True},
                        {"name": "Microsoft", "type": "ORGANIZATION", "direct_match": True},
                    ],
                },
                {
                    "chunk_id": 2,
                    "score": 0.8,
                    "original_score": 0.8,
                    "graph_score": 0.0,
                    "text": "The project was completed",
                    "matched_entities": [],
                },
            ]
            mock_graph_service.return_value = mock_instance

            enhanced = await search_service.search_with_graph_enhancement(
                results=results,
                query="John Microsoft",
                collection=mock_collection_graph_enabled,
            )

            # Should have called the graph service
            mock_graph_service.assert_called_once()
            mock_instance.enhance_results.assert_called_once_with(
                query="John Microsoft",
                vector_results=results,
                collection_id=mock_collection_graph_enabled.id,
            )

            # Should have graph enhancement fields
            assert len(enhanced) == 2
            assert enhanced[0]["original_score"] == 0.9
            assert enhanced[0]["graph_score"] == 0.3
            assert enhanced[0]["score"] == 0.92
            assert len(enhanced[0]["matched_entities"]) == 2

    @pytest.mark.asyncio()
    async def test_graph_enhancement_graceful_degradation_on_error(
        self,
        search_service: SearchService,
        mock_collection_graph_enabled: MagicMock,
    ) -> None:
        """Enhancement failures should not break search - return original results."""
        results = [
            {"chunk_id": 1, "score": 0.9, "text": "Test content"},
        ]

        with patch("webui.services.search_service.GraphEnhancedSearchService") as mock_graph_service:
            mock_instance = AsyncMock()
            mock_instance.enhance_results.side_effect = Exception("Database connection failed")
            mock_graph_service.return_value = mock_instance

            enhanced = await search_service.search_with_graph_enhancement(
                results=results,
                query="test query",
                collection=mock_collection_graph_enabled,
            )

            # Should return original results on error
            assert enhanced == results
            assert len(enhanced) == 1
            assert enhanced[0]["score"] == 0.9

    @pytest.mark.asyncio()
    async def test_graph_enhancement_with_custom_parameters(
        self,
        search_service: SearchService,
        mock_collection_graph_enabled: MagicMock,
    ) -> None:
        """Graph enhancement should respect custom graph_weight and max_hops."""
        results = [{"chunk_id": 1, "score": 0.9}]

        with patch("webui.services.search_service.GraphEnhancedSearchService") as mock_graph_service:
            mock_instance = AsyncMock()
            mock_instance.enhance_results.return_value = results
            mock_graph_service.return_value = mock_instance

            await search_service.search_with_graph_enhancement(
                results=results,
                query="test",
                collection=mock_collection_graph_enabled,
                graph_weight=0.4,
                max_hops=3,
            )

            # Should have been called with custom parameters
            mock_graph_service.assert_called_once()
            call_kwargs = mock_graph_service.call_args.kwargs
            assert call_kwargs["graph_weight"] == 0.4
            assert call_kwargs["max_hops"] == 3


class TestSearchIntegrationWithGraph:
    """Test graph enhancement integration with search methods."""

    @pytest.mark.asyncio()
    async def test_single_collection_search_applies_graph_enhancement(
        self,
        search_service: SearchService,
        mock_collection_repo: AsyncMock,
        mock_collection_graph_enabled: MagicMock,
    ) -> None:
        """Single collection search should apply graph enhancement when enabled."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection_graph_enabled

        # Mock vecpipe response
        mock_response = {
            "results": [
                {"chunk_id": 1, "score": 0.9, "text": "Test content"},
            ],
        }

        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("webui.services.search_service.GraphEnhancedSearchService") as mock_graph_service,
        ):
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj

            # Mock graph enhancement
            mock_instance = AsyncMock()
            mock_instance.enhance_results.return_value = [
                {
                    "chunk_id": 1,
                    "score": 0.95,
                    "original_score": 0.9,
                    "graph_score": 0.25,
                    "text": "Test content",
                    "matched_entities": [],
                },
            ]
            mock_graph_service.return_value = mock_instance

            result = await search_service.single_collection_search(
                user_id=1,
                collection_uuid=mock_collection_graph_enabled.id,
                query="test query",
                k=10,
            )

            # Should have called graph enhancement
            mock_instance.enhance_results.assert_called_once()

            # Result should have graph fields
            assert result["results"][0]["original_score"] == 0.9
            assert result["results"][0]["graph_score"] == 0.25

    @pytest.mark.asyncio()
    async def test_single_collection_search_skips_graph_when_disabled(
        self,
        search_service: SearchService,
        mock_collection_repo: AsyncMock,
        mock_collection_graph_disabled: MagicMock,
    ) -> None:
        """Single collection search should skip graph enhancement when disabled."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection_graph_disabled

        mock_response = {
            "results": [
                {"chunk_id": 1, "score": 0.9, "text": "Test content"},
            ],
        }

        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("webui.services.search_service.GraphEnhancedSearchService") as mock_graph_service,
        ):
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj

            result = await search_service.single_collection_search(
                user_id=1,
                collection_uuid=mock_collection_graph_disabled.id,
                query="test query",
                k=10,
            )

            # Graph service should NOT have been called
            mock_graph_service.assert_not_called()

            # Result should not have graph fields
            assert "original_score" not in result["results"][0]
            assert "graph_score" not in result["results"][0]

    @pytest.mark.asyncio()
    async def test_multi_collection_search_applies_graph_per_collection(
        self,
        search_service: SearchService,
        mock_collection_repo: AsyncMock,
        mock_collection_graph_enabled: MagicMock,
        mock_collection_graph_disabled: MagicMock,
    ) -> None:
        """Multi-collection search should apply graph enhancement per-collection."""
        # First collection has graph enabled, second has it disabled
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = [
            mock_collection_graph_enabled,
            mock_collection_graph_disabled,
        ]

        mock_responses = [
            {"results": [{"chunk_id": 1, "score": 0.9}]},
            {"results": [{"chunk_id": 2, "score": 0.85}]},
        ]

        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("webui.services.search_service.GraphEnhancedSearchService") as mock_graph_service,
        ):
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            response_objs = []
            for resp in mock_responses:
                obj = MagicMock()
                obj.json.return_value = resp
                obj.raise_for_status = MagicMock()
                response_objs.append(obj)
            mock_client.post.side_effect = response_objs

            # Graph enhancement only for first collection
            mock_instance = AsyncMock()
            mock_instance.enhance_results.return_value = [
                {"chunk_id": 1, "score": 0.95, "original_score": 0.9, "graph_score": 0.25, "matched_entities": []},
            ]
            mock_graph_service.return_value = mock_instance

            result = await search_service.multi_collection_search(
                user_id=1,
                collection_uuids=[mock_collection_graph_enabled.id, mock_collection_graph_disabled.id],
                query="test query",
                k=10,
            )

            # Graph service should have been called exactly once (for the enabled collection)
            assert mock_instance.enhance_results.call_count == 1

            # Results should be sorted by score
            assert len(result["results"]) == 2
            # First result (from graph-enabled collection) should have higher score after enhancement
            assert result["results"][0]["score"] == 0.95
            assert result["results"][1]["score"] == 0.85

    @pytest.mark.asyncio()
    async def test_search_continues_when_graph_enhancement_fails(
        self,
        search_service: SearchService,
        mock_collection_repo: AsyncMock,
        mock_collection_graph_enabled: MagicMock,
    ) -> None:
        """Search should continue and return vector results when graph enhancement fails."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection_graph_enabled

        mock_response = {
            "results": [
                {"chunk_id": 1, "score": 0.9, "text": "Test content"},
            ],
        }

        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("webui.services.search_service.GraphEnhancedSearchService") as mock_graph_service,
        ):
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj

            # Graph enhancement fails
            mock_instance = AsyncMock()
            mock_instance.enhance_results.side_effect = Exception("Graph service unavailable")
            mock_graph_service.return_value = mock_instance

            result = await search_service.single_collection_search(
                user_id=1,
                collection_uuid=mock_collection_graph_enabled.id,
                query="test query",
                k=10,
            )

            # Search should still succeed with original results
            assert len(result["results"]) == 1
            assert result["results"][0]["score"] == 0.9
