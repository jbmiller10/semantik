"""Tests for SemanticSearchTool."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.agents.tools.builtins.search import SemanticSearchTool
from shared.agents.types import AgentContext


@asynccontextmanager
async def mock_db_session():
    """Mock async context manager for database session."""
    mock_session = AsyncMock()
    yield mock_session


class TestSemanticSearchToolDefinition:
    """Tests for SemanticSearchTool definition."""

    def test_tool_name(self) -> None:
        """Test tool has correct name."""
        tool = SemanticSearchTool()
        assert tool.name == "semantic_search"

    def test_tool_category(self) -> None:
        """Test tool has correct category."""
        tool = SemanticSearchTool()
        assert tool.definition.category == "search"

    def test_tool_requires_context(self) -> None:
        """Test tool requires context."""
        tool = SemanticSearchTool()
        assert tool.definition.requires_context is True

    def test_tool_is_not_destructive(self) -> None:
        """Test tool is not destructive."""
        tool = SemanticSearchTool()
        assert tool.definition.is_destructive is False

    def test_tool_parameters(self) -> None:
        """Test tool has expected parameters."""
        tool = SemanticSearchTool()
        param_names = [p.name for p in tool.definition.parameters]
        assert "query" in param_names
        assert "top_k" in param_names
        assert "collection_ids" in param_names
        assert "score_threshold" in param_names
        assert "search_type" in param_names

    def test_query_is_required(self) -> None:
        """Test query parameter is required."""
        tool = SemanticSearchTool()
        query_param = next(p for p in tool.definition.parameters if p.name == "query")
        assert query_param.required is True

    def test_default_search_type_from_config(self) -> None:
        """Test default search type comes from configuration."""
        tool = SemanticSearchTool(default_search_type="hybrid")
        search_type_param = next(p for p in tool.definition.parameters if p.name == "search_type")
        assert search_type_param.default == "hybrid"


class TestSemanticSearchToolExecution:
    """Tests for SemanticSearchTool execution."""

    @pytest.fixture()
    def tool(self) -> SemanticSearchTool:
        """Create a tool instance."""
        return SemanticSearchTool()

    @pytest.fixture()
    def context(self) -> AgentContext:
        """Create a test context."""
        return AgentContext(
            request_id="test-request",
            user_id="123",
            collection_id="550e8400-e29b-41d4-a716-446655440000",
        )

    @pytest.mark.asyncio()
    async def test_empty_query_returns_error(self, tool: SemanticSearchTool, context: AgentContext) -> None:
        """Test empty query returns error."""
        result = await tool.execute({"query": ""}, context)
        assert "error" in result
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_no_collection_returns_error(self, tool: SemanticSearchTool) -> None:
        """Test no collection returns error."""
        context = AgentContext(request_id="test", user_id="123")
        result = await tool.execute({"query": "test"}, context)
        assert "error" in result
        assert "collection" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_no_context_returns_error(self, tool: SemanticSearchTool) -> None:
        """Test no context returns error."""
        result = await tool.execute({"query": "test"}, None)
        assert "error" in result
        assert "context" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_invalid_collection_uuid_returns_error(self, tool: SemanticSearchTool, context: AgentContext) -> None:
        """Test invalid collection UUID returns error."""
        result = await tool.execute(
            {"query": "test", "collection_ids": ["not-a-uuid"]},
            context,
        )
        assert "error" in result
        assert "uuid" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_successful_search(self, tool: SemanticSearchTool, context: AgentContext) -> None:
        """Test successful search execution."""
        mock_results = {
            "results": [
                {
                    "content": "Test content",
                    "score": 0.95,
                    "doc_id": "doc1",
                    "chunk_id": "chunk1",
                    "collection_id": "550e8400-e29b-41d4-a716-446655440000",
                    "collection_name": "Test Collection",
                    "metadata": {"source": "test"},
                },
            ],
        }

        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        mock_service = AsyncMock()
        mock_service.multi_collection_search.return_value = mock_results

        with patch(
            "shared.agents.tools.builtins.search.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.search.CollectionRepository",
            ):
                with patch(
                    "webui.services.search_service.SearchService",
                    return_value=mock_service,
                ):
                    result = await tool.execute({"query": "test query"}, context)

        assert "error" not in result
        assert result["query"] == "test query"
        assert result["total"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == "Test content"
        assert result["results"][0]["score"] == 0.95

    @pytest.mark.asyncio()
    async def test_top_k_clamped(self, tool: SemanticSearchTool, context: AgentContext) -> None:
        """Test top_k is clamped to valid range."""
        mock_results = {"results": []}

        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        mock_service = AsyncMock()
        mock_service.multi_collection_search.return_value = mock_results

        with patch(
            "shared.agents.tools.builtins.search.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.search.CollectionRepository",
            ):
                with patch(
                    "webui.services.search_service.SearchService",
                    return_value=mock_service,
                ):
                    # Test top_k > 50 is clamped
                    await tool.execute({"query": "test", "top_k": 100}, context)
                    call_args = mock_service.multi_collection_search.call_args
                    assert call_args.kwargs["k"] == 50

                    # Test top_k < 1 is clamped
                    await tool.execute({"query": "test", "top_k": 0}, context)
                    call_args = mock_service.multi_collection_search.call_args
                    assert call_args.kwargs["k"] == 1

    @pytest.mark.asyncio()
    async def test_reranker_config_passed(self, context: AgentContext) -> None:
        """Test reranker configuration is passed to service."""
        tool = SemanticSearchTool(use_reranker=True, reranker_id="test-reranker")
        mock_results = {"results": []}

        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        mock_service = AsyncMock()
        mock_service.multi_collection_search.return_value = mock_results

        with patch(
            "shared.agents.tools.builtins.search.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.search.CollectionRepository",
            ):
                with patch(
                    "webui.services.search_service.SearchService",
                    return_value=mock_service,
                ):
                    await tool.execute({"query": "test"}, context)
                    call_args = mock_service.multi_collection_search.call_args
                    assert call_args.kwargs["use_reranker"] is True
                    assert call_args.kwargs["reranker_id"] == "test-reranker"

    @pytest.mark.asyncio()
    async def test_collection_ids_from_args(self, tool: SemanticSearchTool, context: AgentContext) -> None:
        """Test collection_ids from args takes precedence."""
        mock_results = {"results": []}
        custom_collection = "660e8400-e29b-41d4-a716-446655440001"

        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        mock_service = AsyncMock()
        mock_service.multi_collection_search.return_value = mock_results

        with patch(
            "shared.agents.tools.builtins.search.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.search.CollectionRepository",
            ):
                with patch(
                    "webui.services.search_service.SearchService",
                    return_value=mock_service,
                ):
                    await tool.execute(
                        {"query": "test", "collection_ids": [custom_collection]},
                        context,
                    )
                    call_args = mock_service.multi_collection_search.call_args
                    assert custom_collection in call_args.kwargs["collection_uuids"]


class TestSemanticSearchToolValidation:
    """Tests for SemanticSearchTool validation."""

    def test_validate_args_missing_query(self) -> None:
        """Test validation fails for missing query."""
        tool = SemanticSearchTool()
        is_valid, errors = tool.validate_args({})
        assert is_valid is False
        assert any("query" in e.lower() for e in errors)

    def test_validate_args_valid_input(self) -> None:
        """Test validation passes for valid input."""
        tool = SemanticSearchTool()
        is_valid, errors = tool.validate_args({"query": "test"})
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_args_invalid_search_type(self) -> None:
        """Test validation fails for invalid search_type."""
        tool = SemanticSearchTool()
        is_valid, errors = tool.validate_args({"query": "test", "search_type": "invalid"})
        assert is_valid is False
        assert any("search_type" in e.lower() for e in errors)
