"""Tests for DocumentRetrieveTool."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from shared.agents.tools.builtins.retrieve import DocumentRetrieveTool
from shared.agents.types import AgentContext


@asynccontextmanager
async def mock_db_session():
    """Mock async context manager for database session."""
    mock_session = AsyncMock()
    yield mock_session


class TestDocumentRetrieveToolDefinition:
    """Tests for DocumentRetrieveTool definition."""

    def test_tool_name(self) -> None:
        """Test tool has correct name."""
        tool = DocumentRetrieveTool()
        assert tool.name == "retrieve_document"

    def test_tool_category(self) -> None:
        """Test tool has correct category."""
        tool = DocumentRetrieveTool()
        assert tool.definition.category == "search"

    def test_tool_requires_context(self) -> None:
        """Test tool requires context."""
        tool = DocumentRetrieveTool()
        assert tool.definition.requires_context is True

    def test_tool_is_not_destructive(self) -> None:
        """Test tool is not destructive."""
        tool = DocumentRetrieveTool()
        assert tool.definition.is_destructive is False

    def test_tool_parameters(self) -> None:
        """Test tool has expected parameters."""
        tool = DocumentRetrieveTool()
        param_names = [p.name for p in tool.definition.parameters]
        assert "document_id" in param_names
        assert "collection_id" in param_names
        assert "include_chunks" in param_names

    def test_document_id_is_required(self) -> None:
        """Test document_id parameter is required."""
        tool = DocumentRetrieveTool()
        doc_param = next(p for p in tool.definition.parameters if p.name == "document_id")
        assert doc_param.required is True

    def test_collection_id_is_required(self) -> None:
        """Test collection_id parameter is required."""
        tool = DocumentRetrieveTool()
        coll_param = next(p for p in tool.definition.parameters if p.name == "collection_id")
        assert coll_param.required is True


class TestDocumentRetrieveToolExecution:
    """Tests for DocumentRetrieveTool execution."""

    @pytest.fixture()
    def tool(self) -> DocumentRetrieveTool:
        """Create a tool instance."""
        return DocumentRetrieveTool()

    @pytest.fixture()
    def context(self) -> AgentContext:
        """Create a test context."""
        return AgentContext(
            request_id="test-request",
            user_id="123",
        )

    @pytest.fixture()
    def valid_doc_id(self) -> str:
        """Return a valid document UUID."""
        return "550e8400-e29b-41d4-a716-446655440001"

    @pytest.fixture()
    def valid_collection_id(self) -> str:
        """Return a valid collection UUID."""
        return "550e8400-e29b-41d4-a716-446655440000"

    @pytest.mark.asyncio()
    async def test_missing_document_id_returns_error(self, tool: DocumentRetrieveTool, context: AgentContext) -> None:
        """Test missing document_id returns error."""
        result = await tool.execute(
            {"collection_id": "550e8400-e29b-41d4-a716-446655440000"},
            context,
        )
        assert "error" in result
        assert "document_id" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_missing_collection_id_returns_error(self, tool: DocumentRetrieveTool, context: AgentContext) -> None:
        """Test missing collection_id returns error."""
        result = await tool.execute(
            {"document_id": "550e8400-e29b-41d4-a716-446655440001"},
            context,
        )
        assert "error" in result
        assert "collection_id" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_invalid_document_uuid_returns_error(self, tool: DocumentRetrieveTool, context: AgentContext) -> None:
        """Test invalid document UUID returns error."""
        result = await tool.execute(
            {
                "document_id": "not-a-uuid",
                "collection_id": "550e8400-e29b-41d4-a716-446655440000",
            },
            context,
        )
        assert "error" in result
        assert "uuid" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_no_context_returns_error(self, tool: DocumentRetrieveTool) -> None:
        """Test no context returns error."""
        result = await tool.execute(
            {
                "document_id": "550e8400-e29b-41d4-a716-446655440001",
                "collection_id": "550e8400-e29b-41d4-a716-446655440000",
            },
            None,
        )
        assert "error" in result
        assert "context" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_collection_not_found_returns_error(
        self,
        tool: DocumentRetrieveTool,
        context: AgentContext,
        valid_doc_id: str,
        valid_collection_id: str,
    ) -> None:
        """Test collection not found returns error."""
        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        mock_coll_repo = AsyncMock()
        mock_coll_repo.get_by_uuid_with_permission_check.return_value = None

        with patch(
            "shared.agents.tools.builtins.retrieve.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.retrieve.CollectionRepository",
                return_value=mock_coll_repo,
            ):
                result = await tool.execute(
                    {"document_id": valid_doc_id, "collection_id": valid_collection_id},
                    context,
                )

        assert "error" in result
        assert "not found" in result["error"].lower() or "access denied" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_document_not_found_returns_error(
        self,
        tool: DocumentRetrieveTool,
        context: AgentContext,
        valid_doc_id: str,
        valid_collection_id: str,
    ) -> None:
        """Test document not found returns error."""
        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        mock_coll_repo = AsyncMock()
        mock_coll_repo.get_by_uuid_with_permission_check.return_value = MagicMock()

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = None

        with patch(
            "shared.agents.tools.builtins.retrieve.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.retrieve.CollectionRepository",
                return_value=mock_coll_repo,
            ):
                with patch(
                    "shared.agents.tools.builtins.retrieve.DocumentRepository",
                    return_value=mock_doc_repo,
                ):
                    result = await tool.execute(
                        {"document_id": valid_doc_id, "collection_id": valid_collection_id},
                        context,
                    )

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_successful_retrieval(
        self,
        tool: DocumentRetrieveTool,
        context: AgentContext,
        valid_doc_id: str,
        valid_collection_id: str,
    ) -> None:
        """Test successful document retrieval."""
        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        mock_coll_repo = AsyncMock()
        mock_coll_repo.get_by_uuid_with_permission_check.return_value = MagicMock()

        mock_document = MagicMock()
        mock_document.id = UUID(valid_doc_id)
        mock_document.collection_id = UUID(valid_collection_id)
        mock_document.file_name = "test.txt"
        mock_document.file_path = "/path/to/test.txt"
        mock_document.mime_type = "text/plain"
        mock_document.status = MagicMock(value="completed")
        mock_document.chunk_count = 5
        mock_document.file_size = 1024
        mock_document.created_at = datetime.now(tz=UTC)
        mock_document.updated_at = datetime.now(tz=UTC)
        mock_document.meta = {"source": "test"}

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = mock_document

        with patch(
            "shared.agents.tools.builtins.retrieve.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.retrieve.CollectionRepository",
                return_value=mock_coll_repo,
            ):
                with patch(
                    "shared.agents.tools.builtins.retrieve.DocumentRepository",
                    return_value=mock_doc_repo,
                ):
                    result = await tool.execute(
                        {"document_id": valid_doc_id, "collection_id": valid_collection_id},
                        context,
                    )

        assert "error" not in result
        assert result["id"] == valid_doc_id
        assert result["file_name"] == "test.txt"
        assert result["mime_type"] == "text/plain"
        assert result["chunk_count"] == 5

    @pytest.mark.asyncio()
    async def test_include_chunks(
        self,
        tool: DocumentRetrieveTool,
        context: AgentContext,
        valid_doc_id: str,
        valid_collection_id: str,
    ) -> None:
        """Test retrieval with include_chunks=True."""
        mock_manager = MagicMock()
        mock_manager.get_session = mock_db_session

        mock_coll_repo = AsyncMock()
        mock_coll_repo.get_by_uuid_with_permission_check.return_value = MagicMock()

        mock_document = MagicMock()
        mock_document.id = UUID(valid_doc_id)
        mock_document.collection_id = UUID(valid_collection_id)
        mock_document.file_name = "test.txt"
        mock_document.file_path = "/path/to/test.txt"
        mock_document.mime_type = "text/plain"
        mock_document.status = MagicMock(value="completed")
        mock_document.chunk_count = 2
        mock_document.file_size = 1024
        mock_document.created_at = datetime.now(tz=UTC)
        mock_document.updated_at = datetime.now(tz=UTC)
        mock_document.meta = {}

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_by_id.return_value = mock_document

        mock_chunk1 = MagicMock()
        mock_chunk1.id = 1
        mock_chunk1.chunk_index = 0
        mock_chunk1.metadata = {"chunk_id": f"{valid_doc_id}_0000"}

        mock_chunk2 = MagicMock()
        mock_chunk2.id = 2
        mock_chunk2.chunk_index = 1
        mock_chunk2.metadata = {"chunk_id": f"{valid_doc_id}_0001"}

        mock_chunk_repo = AsyncMock()
        mock_chunk_repo.get_chunks_by_document.return_value = [mock_chunk1, mock_chunk2]

        with patch(
            "shared.agents.tools.builtins.retrieve.pg_connection_manager",
            mock_manager,
        ):
            with patch(
                "shared.agents.tools.builtins.retrieve.CollectionRepository",
                return_value=mock_coll_repo,
            ):
                with patch(
                    "shared.agents.tools.builtins.retrieve.DocumentRepository",
                    return_value=mock_doc_repo,
                ):
                    with patch(
                        "shared.agents.tools.builtins.retrieve.ChunkRepository",
                        return_value=mock_chunk_repo,
                    ):
                        result = await tool.execute(
                            {
                                "document_id": valid_doc_id,
                                "collection_id": valid_collection_id,
                                "include_chunks": True,
                            },
                            context,
                        )

        assert "error" not in result
        assert "chunks" in result
        assert len(result["chunks"]) == 2
        assert result["chunks"][0]["index"] == 0
        assert result["chunks"][1]["index"] == 1


class TestDocumentRetrieveToolValidation:
    """Tests for DocumentRetrieveTool validation."""

    def test_validate_args_missing_document_id(self) -> None:
        """Test validation fails for missing document_id."""
        tool = DocumentRetrieveTool()
        is_valid, errors = tool.validate_args({"collection_id": "test"})
        assert is_valid is False
        assert any("document_id" in e.lower() for e in errors)

    def test_validate_args_missing_collection_id(self) -> None:
        """Test validation fails for missing collection_id."""
        tool = DocumentRetrieveTool()
        is_valid, errors = tool.validate_args({"document_id": "test"})
        assert is_valid is False
        assert any("collection_id" in e.lower() for e in errors)

    def test_validate_args_valid_input(self) -> None:
        """Test validation passes for valid input."""
        tool = DocumentRetrieveTool()
        is_valid, errors = tool.validate_args({"document_id": "test", "collection_id": "test"})
        assert is_valid is True
        assert len(errors) == 0
