"""Tests for ListCollectionsTool."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from shared.agents.tools.builtins.collections import ListCollectionsTool
from shared.agents.types import AgentContext


@asynccontextmanager
async def mock_db_session():
    """Mock async context manager for database session."""
    mock_session = AsyncMock()
    yield mock_session


class TestListCollectionsToolDefinition:
    """Tests for ListCollectionsTool definition."""

    def test_tool_name(self) -> None:
        """Test tool has correct name."""
        tool = ListCollectionsTool()
        assert tool.name == "list_collections"

    def test_tool_category(self) -> None:
        """Test tool has correct category."""
        tool = ListCollectionsTool()
        assert tool.definition.category == "search"

    def test_tool_requires_context(self) -> None:
        """Test tool requires context."""
        tool = ListCollectionsTool()
        assert tool.definition.requires_context is True

    def test_tool_is_not_destructive(self) -> None:
        """Test tool is not destructive."""
        tool = ListCollectionsTool()
        assert tool.definition.is_destructive is False

    def test_tool_parameters(self) -> None:
        """Test tool has expected parameters."""
        tool = ListCollectionsTool()
        param_names = [p.name for p in tool.definition.parameters]
        assert "include_public" in param_names
        assert "status_filter" in param_names
        assert "limit" in param_names

    def test_all_parameters_optional(self) -> None:
        """Test all parameters are optional."""
        tool = ListCollectionsTool()
        for param in tool.definition.parameters:
            assert param.required is False


class TestListCollectionsToolExecution:
    """Tests for ListCollectionsTool execution."""

    @pytest.fixture
    def tool(self) -> ListCollectionsTool:
        """Create a tool instance."""
        return ListCollectionsTool()

    @pytest.fixture
    def context(self) -> AgentContext:
        """Create a test context."""
        return AgentContext(
            request_id="test-request",
            user_id="123",
        )

    @pytest.mark.asyncio
    async def test_no_context_returns_error(self, tool: ListCollectionsTool) -> None:
        """Test no context returns error."""
        result = await tool.execute({}, None)
        assert "error" in result
        assert "context" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_user_id_returns_error(self, tool: ListCollectionsTool) -> None:
        """Test invalid user_id returns error."""
        context = AgentContext(request_id="test", user_id="not-a-number")
        result = await tool.execute({}, context)
        assert "error" in result
        assert "user_id" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_listing(self, tool: ListCollectionsTool, context: AgentContext) -> None:
        """Test successful collection listing."""
        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        # Create mock collections
        mock_coll1 = MagicMock()
        mock_coll1.id = UUID("550e8400-e29b-41d4-a716-446655440001")
        mock_coll1.name = "Test Collection 1"
        mock_coll1.description = "First test collection"
        mock_coll1.status = MagicMock(value="ready")
        mock_coll1.embedding_model = "test-model"
        mock_coll1.is_public = False
        mock_coll1.owner_id = 123
        mock_coll1.created_at = datetime.now()

        mock_coll2 = MagicMock()
        mock_coll2.id = UUID("550e8400-e29b-41d4-a716-446655440002")
        mock_coll2.name = "Test Collection 2"
        mock_coll2.description = "Second test collection"
        mock_coll2.status = MagicMock(value="ready")
        mock_coll2.embedding_model = "test-model"
        mock_coll2.is_public = True
        mock_coll2.owner_id = 456
        mock_coll2.created_at = datetime.now()

        mock_repo = AsyncMock()
        mock_repo.list_for_user.return_value = ([mock_coll1, mock_coll2], 2)

        with patch(
            "shared.agents.tools.builtins.collections.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.collections.CollectionRepository",
                return_value=mock_repo,
            ):
                result = await tool.execute({}, context)

        assert "error" not in result
        assert result["total"] == 2
        assert len(result["collections"]) == 2
        assert result["collections"][0]["name"] == "Test Collection 1"
        assert result["collections"][0]["is_owned"] is True
        assert result["collections"][1]["is_owned"] is False

    @pytest.mark.asyncio
    async def test_status_filter(self, tool: ListCollectionsTool, context: AgentContext) -> None:
        """Test status filter is applied."""
        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        # Create collections with different statuses
        mock_ready = MagicMock()
        mock_ready.id = UUID("550e8400-e29b-41d4-a716-446655440001")
        mock_ready.name = "Ready Collection"
        mock_ready.description = ""
        mock_ready.status = MagicMock(value="ready")
        mock_ready.embedding_model = "test-model"
        mock_ready.is_public = False
        mock_ready.owner_id = 123
        mock_ready.created_at = datetime.now()

        mock_processing = MagicMock()
        mock_processing.id = UUID("550e8400-e29b-41d4-a716-446655440002")
        mock_processing.name = "Processing Collection"
        mock_processing.description = ""
        mock_processing.status = MagicMock(value="processing")
        mock_processing.embedding_model = "test-model"
        mock_processing.is_public = False
        mock_processing.owner_id = 123
        mock_processing.created_at = datetime.now()

        mock_repo = AsyncMock()
        mock_repo.list_for_user.return_value = ([mock_ready, mock_processing], 2)

        with patch(
            "shared.agents.tools.builtins.collections.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.collections.CollectionRepository",
                return_value=mock_repo,
            ):
                result = await tool.execute({"status_filter": "ready"}, context)

        assert "error" not in result
        assert result["total"] == 1
        assert result["collections"][0]["name"] == "Ready Collection"

    @pytest.mark.asyncio
    async def test_limit_clamped(self, tool: ListCollectionsTool, context: AgentContext) -> None:
        """Test limit is clamped to valid range."""
        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        mock_repo = AsyncMock()
        mock_repo.list_for_user.return_value = ([], 0)

        with patch(
            "shared.agents.tools.builtins.collections.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.collections.CollectionRepository",
                return_value=mock_repo,
            ):
                # Test limit > 100 is clamped
                await tool.execute({"limit": 200}, context)
                call_args = mock_repo.list_for_user.call_args
                assert call_args.kwargs["limit"] == 100

                # Test limit < 1 is clamped
                await tool.execute({"limit": 0}, context)
                call_args = mock_repo.list_for_user.call_args
                assert call_args.kwargs["limit"] == 1

    @pytest.mark.asyncio
    async def test_include_public_passed(self, tool: ListCollectionsTool, context: AgentContext) -> None:
        """Test include_public is passed to repository."""
        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        mock_repo = AsyncMock()
        mock_repo.list_for_user.return_value = ([], 0)

        with patch(
            "shared.agents.tools.builtins.collections.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.collections.CollectionRepository",
                return_value=mock_repo,
            ):
                await tool.execute({"include_public": False}, context)
                call_args = mock_repo.list_for_user.call_args
                assert call_args.kwargs["include_public"] is False


class TestListCollectionsToolValidation:
    """Tests for ListCollectionsTool validation."""

    def test_validate_args_empty_input(self) -> None:
        """Test validation passes for empty input (all optional)."""
        tool = ListCollectionsTool()
        is_valid, errors = tool.validate_args({})
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_args_invalid_status(self) -> None:
        """Test validation fails for invalid status_filter."""
        tool = ListCollectionsTool()
        is_valid, errors = tool.validate_args({"status_filter": "invalid_status"})
        assert is_valid is False
        assert any("status_filter" in e.lower() for e in errors)
