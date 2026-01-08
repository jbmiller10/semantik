"""Tests for GetChunkTool."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from shared.agents.tools.builtins.chunks import GetChunkTool
from shared.agents.types import AgentContext


@asynccontextmanager
async def mock_db_session():
    """Mock async context manager for database session."""
    mock_session = AsyncMock()
    yield mock_session


class TestGetChunkToolDefinition:
    """Tests for GetChunkTool definition."""

    def test_tool_name(self) -> None:
        """Test tool has correct name."""
        tool = GetChunkTool()
        assert tool.name == "get_chunk"

    def test_tool_category(self) -> None:
        """Test tool has correct category."""
        tool = GetChunkTool()
        assert tool.definition.category == "search"

    def test_tool_requires_context(self) -> None:
        """Test tool requires context."""
        tool = GetChunkTool()
        assert tool.definition.requires_context is True

    def test_tool_is_not_destructive(self) -> None:
        """Test tool is not destructive."""
        tool = GetChunkTool()
        assert tool.definition.is_destructive is False

    def test_tool_parameters(self) -> None:
        """Test tool has expected parameters."""
        tool = GetChunkTool()
        param_names = [p.name for p in tool.definition.parameters]
        assert "chunk_id" in param_names
        assert "collection_id" in param_names

    def test_chunk_id_is_required(self) -> None:
        """Test chunk_id parameter is required."""
        tool = GetChunkTool()
        chunk_param = next(p for p in tool.definition.parameters if p.name == "chunk_id")
        assert chunk_param.required is True

    def test_collection_id_is_required(self) -> None:
        """Test collection_id parameter is required."""
        tool = GetChunkTool()
        coll_param = next(p for p in tool.definition.parameters if p.name == "collection_id")
        assert coll_param.required is True


class TestGetChunkToolExecution:
    """Tests for GetChunkTool execution."""

    @pytest.fixture
    def tool(self) -> GetChunkTool:
        """Create a tool instance."""
        return GetChunkTool()

    @pytest.fixture
    def context(self) -> AgentContext:
        """Create a test context."""
        return AgentContext(
            request_id="test-request",
            user_id="123",
        )

    @pytest.fixture
    def valid_chunk_id(self) -> str:
        """Return a valid chunk ID format."""
        return "550e8400-e29b-41d4-a716-446655440001_0001"

    @pytest.fixture
    def valid_collection_id(self) -> str:
        """Return a valid collection UUID."""
        return "550e8400-e29b-41d4-a716-446655440000"

    @pytest.mark.asyncio
    async def test_missing_chunk_id_returns_error(self, tool: GetChunkTool, context: AgentContext) -> None:
        """Test missing chunk_id returns error."""
        result = await tool.execute(
            {"collection_id": "550e8400-e29b-41d4-a716-446655440000"},
            context,
        )
        assert "error" in result
        assert "chunk_id" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_collection_id_returns_error(self, tool: GetChunkTool, context: AgentContext) -> None:
        """Test missing collection_id returns error."""
        result = await tool.execute(
            {"chunk_id": "doc_0001"},
            context,
        )
        assert "error" in result
        assert "collection_id" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_collection_uuid_returns_error(self, tool: GetChunkTool, context: AgentContext) -> None:
        """Test invalid collection UUID returns error."""
        result = await tool.execute(
            {"chunk_id": "doc_0001", "collection_id": "not-a-uuid"},
            context,
        )
        assert "error" in result
        assert "uuid" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_no_context_returns_error(self, tool: GetChunkTool) -> None:
        """Test no context returns error."""
        result = await tool.execute(
            {
                "chunk_id": "doc_0001",
                "collection_id": "550e8400-e29b-41d4-a716-446655440000",
            },
            None,
        )
        assert "error" in result
        assert "context" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_collection_not_found_returns_error(
        self,
        tool: GetChunkTool,
        context: AgentContext,
        valid_chunk_id: str,
        valid_collection_id: str,
    ) -> None:
        """Test collection not found returns error."""
        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        mock_coll_repo = AsyncMock()
        mock_coll_repo.get_by_uuid_with_permission_check.return_value = None

        with patch(
            "shared.agents.tools.builtins.chunks.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.chunks.CollectionRepository",
                return_value=mock_coll_repo,
            ):
                result = await tool.execute(
                    {"chunk_id": valid_chunk_id, "collection_id": valid_collection_id},
                    context,
                )

        assert "error" in result
        assert "not found" in result["error"].lower() or "access denied" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_chunk_not_found_returns_error(
        self,
        tool: GetChunkTool,
        context: AgentContext,
        valid_chunk_id: str,
        valid_collection_id: str,
    ) -> None:
        """Test chunk not found returns error."""
        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        mock_coll_repo = AsyncMock()
        mock_coll_repo.get_by_uuid_with_permission_check.return_value = MagicMock()

        mock_chunk_repo = AsyncMock()
        mock_chunk_repo.get_chunk_by_metadata_chunk_id.return_value = None

        with patch(
            "shared.agents.tools.builtins.chunks.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.chunks.CollectionRepository",
                return_value=mock_coll_repo,
            ):
                with patch(
                    "shared.agents.tools.builtins.chunks.ChunkRepository",
                    return_value=mock_chunk_repo,
                ):
                    result = await tool.execute(
                        {"chunk_id": valid_chunk_id, "collection_id": valid_collection_id},
                        context,
                    )

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_retrieval(
        self,
        tool: GetChunkTool,
        context: AgentContext,
        valid_chunk_id: str,
        valid_collection_id: str,
    ) -> None:
        """Test successful chunk retrieval."""
        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        mock_coll_repo = AsyncMock()
        mock_coll_repo.get_by_uuid_with_permission_check.return_value = MagicMock()

        mock_chunk = MagicMock()
        mock_chunk.id = 12345
        mock_chunk.content = "This is the chunk content."
        mock_chunk.document_id = UUID("550e8400-e29b-41d4-a716-446655440001")
        mock_chunk.collection_id = UUID(valid_collection_id)
        mock_chunk.chunk_index = 1
        mock_chunk.metadata = {"chunk_id": valid_chunk_id, "source": "test"}

        mock_chunk_repo = AsyncMock()
        mock_chunk_repo.get_chunk_by_metadata_chunk_id.return_value = mock_chunk

        with patch(
            "shared.agents.tools.builtins.chunks.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.chunks.CollectionRepository",
                return_value=mock_coll_repo,
            ):
                with patch(
                    "shared.agents.tools.builtins.chunks.ChunkRepository",
                    return_value=mock_chunk_repo,
                ):
                    result = await tool.execute(
                        {"chunk_id": valid_chunk_id, "collection_id": valid_collection_id},
                        context,
                    )

        assert "error" not in result
        assert result["id"] == "12345"
        assert result["chunk_id"] == valid_chunk_id
        assert result["content"] == "This is the chunk content."
        assert result["chunk_index"] == 1
        assert result["metadata"]["source"] == "test"


class TestGetChunkToolValidation:
    """Tests for GetChunkTool validation."""

    def test_validate_args_missing_chunk_id(self) -> None:
        """Test validation fails for missing chunk_id."""
        tool = GetChunkTool()
        is_valid, errors = tool.validate_args({"collection_id": "test"})
        assert is_valid is False
        assert any("chunk_id" in e.lower() for e in errors)

    def test_validate_args_missing_collection_id(self) -> None:
        """Test validation fails for missing collection_id."""
        tool = GetChunkTool()
        is_valid, errors = tool.validate_args({"chunk_id": "test"})
        assert is_valid is False
        assert any("collection_id" in e.lower() for e in errors)

    def test_validate_args_valid_input(self) -> None:
        """Test validation passes for valid input."""
        tool = GetChunkTool()
        is_valid, errors = tool.validate_args({"chunk_id": "test", "collection_id": "test"})
        assert is_valid is True
        assert len(errors) == 0
