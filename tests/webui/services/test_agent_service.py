"""Comprehensive tests for AgentService covering all methods and edge cases."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from shared.agents.exceptions import SessionNotFoundError
from shared.agents.types import AgentCapabilities, AgentContext, AgentUseCase
from webui.services.agent_service import AgentService


@pytest.fixture()
def mock_db_session() -> AsyncMock:
    """Mock database session with proper async methods."""
    mock = AsyncMock(spec=AsyncSession)
    mock.commit = AsyncMock()
    mock.rollback = AsyncMock()
    mock.close = AsyncMock()
    mock.execute = AsyncMock()
    mock.scalar = AsyncMock()
    mock.scalars = AsyncMock()
    return mock


@pytest.fixture()
def mock_session_repo() -> AsyncMock:
    """Mock agent session repository with all async methods."""
    mock = AsyncMock()
    mock.create = AsyncMock()
    mock.get_by_id = AsyncMock()
    mock.get_by_external_id = AsyncMock()
    mock.get_by_sdk_session_id = AsyncMock()
    mock.list_by_user = AsyncMock()
    mock.add_message = AsyncMock()
    mock.get_messages = AsyncMock()
    mock.update_sdk_session_id = AsyncMock()
    mock.update_title = AsyncMock()
    mock.archive = AsyncMock()
    mock.delete = AsyncMock()
    mock.hard_delete = AsyncMock()
    mock.cleanup_old_sessions = AsyncMock()
    mock.increment_fork_count = AsyncMock()
    return mock


@pytest.fixture()
def mock_agent_session() -> MagicMock:
    """Mock an agent session object with all required attributes."""
    session = MagicMock()
    session.id = "123e4567-e89b-12d3-a456-426614174000"
    session.external_id = "abc123"
    session.title = "Test Session"
    session.user_id = "user-1"
    session.agent_plugin_id = "claude-agent"
    session.agent_config = {"model": "claude-sonnet-4-20250514"}
    session.collection_id = None
    session.messages = []
    session.message_count = 0
    session.sdk_session_id = None
    session.parent_session_id = None
    session.fork_count = 0
    session.total_input_tokens = 0
    session.total_output_tokens = 0
    session.total_cost_usd = 0
    session.status = "active"
    session.created_at = MagicMock()
    session.updated_at = MagicMock()
    session.last_activity_at = MagicMock()
    session.archived_at = None

    def session_to_dict() -> dict[str, Any]:
        return {
            "id": session.id,
            "external_id": session.external_id,
            "title": session.title,
            "user_id": session.user_id,
            "agent_plugin_id": session.agent_plugin_id,
            "agent_config": session.agent_config,
            "collection_id": session.collection_id,
            "message_count": session.message_count,
            "status": session.status,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

    session.to_dict = MagicMock(side_effect=session_to_dict)
    return session


@pytest.fixture()
def mock_session_message() -> MagicMock:
    """Mock a session message object."""
    msg = MagicMock()
    msg.id = "msg-123"
    msg.message_id = "msg-123"
    msg.session_id = "123e4567-e89b-12d3-a456-426614174000"
    msg.sequence_number = 0
    msg.role = "user"
    msg.type = "text"
    msg.content = "Hello"
    msg.tool_name = None
    msg.tool_call_id = None
    msg.tool_input = None
    msg.tool_output = None
    msg.model = None
    msg.input_tokens = None
    msg.output_tokens = None
    msg.cost_usd = None
    msg.created_at = MagicMock()
    return msg


@pytest.fixture()
def mock_plugin_record() -> MagicMock:
    """Mock a plugin registry record."""
    record = MagicMock()
    record.plugin_id = "claude-agent"
    record.plugin_version = "1.0.0"
    record.manifest = MagicMock()
    record.manifest.to_dict.return_value = {
        "id": "claude-agent",
        "type": "agent",
        "version": "1.0.0",
        "display_name": "Claude Agent",
        "description": "LLM agent powered by Claude",
    }
    record.plugin_class = MagicMock()
    record.plugin_class.get_capabilities.return_value = AgentCapabilities(
        supports_streaming=True,
        supports_tools=True,
        supported_models=("claude-sonnet-4-20250514",),
        default_model="claude-sonnet-4-20250514",
    )
    record.plugin_class.supported_use_cases.return_value = [
        AgentUseCase.ASSISTANT,
        AgentUseCase.TOOL_USE,
    ]
    record.plugin_class.get_config_schema.return_value = {"type": "object"}
    return record


@pytest.fixture()
def agent_service(
    mock_db_session: AsyncMock,
    mock_session_repo: AsyncMock,
) -> AgentService:
    """Create an AgentService instance with mocked dependencies."""
    return AgentService(db=mock_db_session, session_repo=mock_session_repo)


class TestAgentServiceInit:
    """Test AgentService initialization."""

    def test_init(
        self,
        mock_db_session: AsyncMock,
        mock_session_repo: AsyncMock,
    ) -> None:
        """Test service initialization."""
        service = AgentService(db=mock_db_session, session_repo=mock_session_repo)

        assert service._db == mock_db_session
        assert service._session_repo == mock_session_repo
        assert service._instances == {}
        assert service._active_executions == {}

    def test_init_creates_default_repo(
        self,
        mock_db_session: AsyncMock,
    ) -> None:
        """Test service creates default repo if not provided."""
        with patch("webui.services.agent_service.AgentSessionRepository") as mock_repo_class:
            mock_repo_instance = AsyncMock()
            mock_repo_class.return_value = mock_repo_instance

            service = AgentService(db=mock_db_session)

            mock_repo_class.assert_called_once_with(mock_db_session)
            assert service._session_repo == mock_repo_instance


class TestComputeConfigHash:
    """Test _compute_config_hash method."""

    def test_empty_config_returns_default(self) -> None:
        """Test empty config returns 'default'."""
        assert AgentService._compute_config_hash(None) == "default"
        assert AgentService._compute_config_hash({}) == "default"

    def test_same_config_same_hash(self) -> None:
        """Test same config produces same hash."""
        config = {"model": "claude-sonnet-4-20250514", "temperature": 0.7}
        hash1 = AgentService._compute_config_hash(config)
        hash2 = AgentService._compute_config_hash(config)
        assert hash1 == hash2

    def test_different_config_different_hash(self) -> None:
        """Test different config produces different hash."""
        config1 = {"model": "claude-sonnet-4-20250514"}
        config2 = {"model": "claude-opus-4-20250514"}
        hash1 = AgentService._compute_config_hash(config1)
        hash2 = AgentService._compute_config_hash(config2)
        assert hash1 != hash2

    def test_key_order_independent(self) -> None:
        """Test hash is independent of key order."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}
        hash1 = AgentService._compute_config_hash(config1)
        hash2 = AgentService._compute_config_hash(config2)
        assert hash1 == hash2


class TestBuildExecutionContext:
    """Test _build_execution_context method."""

    def test_sets_external_session_id_on_existing_context(
        self,
        agent_service: AgentService,
        mock_agent_session: MagicMock,
    ) -> None:
        """Uses external_id for context.session_id."""
        context = AgentContext(request_id="req-1")

        result = agent_service._build_execution_context(context=context, db_session=mock_agent_session)

        assert result.session_id == mock_agent_session.external_id

    def test_creates_context_and_sets_external_session_id(
        self,
        agent_service: AgentService,
        mock_agent_session: MagicMock,
    ) -> None:
        """Creates a context when none is provided."""
        result = agent_service._build_execution_context(context=None, db_session=mock_agent_session)

        assert result.request_id == str(mock_agent_session.id)
        assert result.session_id == mock_agent_session.external_id


class TestListAgents:
    """Test list_agents method."""

    @pytest.mark.asyncio()
    async def test_list_agents_returns_registered(
        self,
        agent_service: AgentService,
        mock_plugin_record: MagicMock,
    ) -> None:
        """Test listing all available agent plugins."""
        with (
            patch("webui.services.agent_service.load_plugins") as mock_load,
            patch("webui.services.agent_service.plugin_registry") as mock_registry,
        ):
            mock_registry.list_records.return_value = [mock_plugin_record]

            result = await agent_service.list_agents()

            mock_load.assert_called_once_with(plugin_types={"agent"})
            mock_registry.list_records.assert_called_once_with(plugin_type="agent")
            assert len(result) == 1
            assert result[0]["id"] == "claude-agent"
            assert result[0]["version"] == "1.0.0"
            assert "capabilities" in result[0]
            assert "use_cases" in result[0]

    @pytest.mark.asyncio()
    async def test_list_agents_empty_registry(
        self,
        agent_service: AgentService,
    ) -> None:
        """Test listing when no agents are registered."""
        with (
            patch("webui.services.agent_service.load_plugins"),
            patch("webui.services.agent_service.plugin_registry") as mock_registry,
        ):
            mock_registry.list_records.return_value = []

            result = await agent_service.list_agents()

            assert result == []


class TestGetAgent:
    """Test get_agent method."""

    @pytest.mark.asyncio()
    async def test_get_agent_returns_details(
        self,
        agent_service: AgentService,
        mock_plugin_record: MagicMock,
    ) -> None:
        """Test getting agent details by ID."""
        with (
            patch("webui.services.agent_service.load_plugins"),
            patch("webui.services.agent_service.plugin_registry") as mock_registry,
        ):
            mock_registry.get.return_value = mock_plugin_record

            result = await agent_service.get_agent("claude-agent")

            mock_registry.get.assert_called_once_with("agent", "claude-agent")
            assert result is not None
            assert result["id"] == "claude-agent"
            assert "config_schema" in result

    @pytest.mark.asyncio()
    async def test_get_agent_not_found(
        self,
        agent_service: AgentService,
    ) -> None:
        """Test getting non-existent agent returns None."""
        with (
            patch("webui.services.agent_service.load_plugins"),
            patch("webui.services.agent_service.plugin_registry") as mock_registry,
        ):
            mock_registry.get.return_value = None

            result = await agent_service.get_agent("non-existent")

            assert result is None


class TestFindAgentForUseCase:
    """Test find_agent_for_use_case method."""

    @pytest.mark.asyncio()
    async def test_find_agent_matching_use_case(
        self,
        agent_service: AgentService,
        mock_plugin_record: MagicMock,
    ) -> None:
        """Test finding agent for matching use case."""
        with (
            patch("webui.services.agent_service.load_plugins"),
            patch("webui.services.agent_service.plugin_registry") as mock_registry,
        ):
            mock_registry.list_records.return_value = [mock_plugin_record]

            result = await agent_service.find_agent_for_use_case(AgentUseCase.ASSISTANT)

            assert result == "claude-agent"

    @pytest.mark.asyncio()
    async def test_find_agent_no_match(
        self,
        agent_service: AgentService,
        mock_plugin_record: MagicMock,
    ) -> None:
        """Test finding agent with no matching use case."""
        with (
            patch("webui.services.agent_service.load_plugins"),
            patch("webui.services.agent_service.plugin_registry") as mock_registry,
        ):
            mock_registry.list_records.return_value = [mock_plugin_record]

            # HYDE is not in the supported use cases
            result = await agent_service.find_agent_for_use_case(AgentUseCase.HYDE)

            assert result is None


class TestGetCapabilities:
    """Test get_capabilities method."""

    @pytest.mark.asyncio()
    async def test_get_capabilities(
        self,
        agent_service: AgentService,
        mock_plugin_record: MagicMock,
    ) -> None:
        """Test getting agent capabilities."""
        with (
            patch("webui.services.agent_service.load_plugins"),
            patch("webui.services.agent_service.plugin_registry") as mock_registry,
        ):
            mock_registry.get.return_value = mock_plugin_record

            result = await agent_service.get_capabilities("claude-agent")

            assert result is not None
            assert result.supports_streaming is True
            assert result.supports_tools is True

    @pytest.mark.asyncio()
    async def test_get_capabilities_not_found(
        self,
        agent_service: AgentService,
    ) -> None:
        """Test getting capabilities for non-existent agent."""
        with (
            patch("webui.services.agent_service.load_plugins"),
            patch("webui.services.agent_service.plugin_registry") as mock_registry,
        ):
            mock_registry.get.return_value = None

            result = await agent_service.get_capabilities("non-existent")

            assert result is None


class TestGetSession:
    """Test get_session method."""

    @pytest.mark.asyncio()
    async def test_get_session_found(
        self,
        agent_service: AgentService,
        mock_session_repo: AsyncMock,
        mock_agent_session: MagicMock,
    ) -> None:
        """Test getting existing session."""
        mock_session_repo.get_by_external_id.return_value = mock_agent_session

        result = await agent_service.get_session("abc123")

        mock_session_repo.get_by_external_id.assert_called_once_with("abc123")
        assert result is not None
        assert result["external_id"] == "abc123"

    @pytest.mark.asyncio()
    async def test_get_session_not_found(
        self,
        agent_service: AgentService,
        mock_session_repo: AsyncMock,
    ) -> None:
        """Test getting non-existent session returns None."""
        mock_session_repo.get_by_external_id.return_value = None

        result = await agent_service.get_session("non-existent")

        assert result is None


class TestGetMessages:
    """Test get_messages method."""

    @pytest.mark.asyncio()
    async def test_get_messages(
        self,
        agent_service: AgentService,
        mock_session_repo: AsyncMock,
        mock_agent_session: MagicMock,
        mock_session_message: MagicMock,
    ) -> None:
        """Test getting session messages."""
        mock_session_repo.get_by_external_id.return_value = mock_agent_session
        mock_session_repo.get_messages.return_value = [mock_session_message]

        result = await agent_service.get_messages("abc123", limit=50, offset=0)

        assert len(result) == 1
        assert result[0]["content"] == "Hello"
        mock_session_repo.get_messages.assert_called_once()

    @pytest.mark.asyncio()
    async def test_get_messages_session_not_found(
        self,
        agent_service: AgentService,
        mock_session_repo: AsyncMock,
    ) -> None:
        """Test getting messages for non-existent session raises error."""
        mock_session_repo.get_by_external_id.return_value = None

        with pytest.raises(SessionNotFoundError):
            await agent_service.get_messages("non-existent")


class TestListSessions:
    """Test list_sessions method."""

    @pytest.mark.asyncio()
    async def test_list_sessions(
        self,
        agent_service: AgentService,
        mock_session_repo: AsyncMock,
        mock_agent_session: MagicMock,
    ) -> None:
        """Test listing user sessions."""
        mock_session_repo.list_by_user.return_value = ([mock_agent_session], 1)

        sessions, total = await agent_service.list_sessions(user_id=1)

        assert len(sessions) == 1
        assert total == 1
        mock_session_repo.list_by_user.assert_called_once_with(1, status=None, collection_id=None, limit=50, offset=0)


class TestUpdateSessionTitle:
    """Test update_session_title method."""

    @pytest.mark.asyncio()
    async def test_update_title(
        self,
        agent_service: AgentService,
        mock_db_session: AsyncMock,
        mock_session_repo: AsyncMock,
        mock_agent_session: MagicMock,
    ) -> None:
        """Test updating session title."""
        mock_session_repo.get_by_external_id.return_value = mock_agent_session
        mock_session_repo.get_by_id.return_value = mock_agent_session

        result = await agent_service.update_session_title("abc123", "New Title")

        mock_session_repo.update_title.assert_called_once()
        mock_db_session.commit.assert_called()
        assert result is not None

    @pytest.mark.asyncio()
    async def test_update_title_not_found(
        self,
        agent_service: AgentService,
        mock_session_repo: AsyncMock,
    ) -> None:
        """Test updating title for non-existent session raises error."""
        mock_session_repo.get_by_external_id.return_value = None

        with pytest.raises(SessionNotFoundError):
            await agent_service.update_session_title("non-existent", "New Title")


class TestForkSession:
    """Test fork_session method."""

    @pytest.mark.asyncio()
    async def test_fork_session(
        self,
        agent_service: AgentService,
        mock_db_session: AsyncMock,
        mock_session_repo: AsyncMock,
        mock_agent_session: MagicMock,
    ) -> None:
        """Test forking a session."""
        forked_session = MagicMock()
        forked_session.external_id = "forked123"
        forked_session.messages = None
        forked_session.message_count = 0
        forked_session.sdk_session_id = None

        mock_session_repo.get_by_external_id.return_value = mock_agent_session
        mock_session_repo.create.return_value = forked_session

        result = await agent_service.fork_session("abc123")

        assert result == "forked123"
        mock_session_repo.create.assert_called_once()
        mock_session_repo.increment_fork_count.assert_called_once()
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio()
    async def test_fork_session_not_found(
        self,
        agent_service: AgentService,
        mock_session_repo: AsyncMock,
    ) -> None:
        """Test forking non-existent session raises error."""
        mock_session_repo.get_by_external_id.return_value = None

        with pytest.raises(SessionNotFoundError):
            await agent_service.fork_session("non-existent")


class TestArchiveSession:
    """Test archive_session method."""

    @pytest.mark.asyncio()
    async def test_archive_session(
        self,
        agent_service: AgentService,
        mock_db_session: AsyncMock,
        mock_session_repo: AsyncMock,
        mock_agent_session: MagicMock,
    ) -> None:
        """Test archiving a session."""
        mock_session_repo.get_by_external_id.return_value = mock_agent_session

        await agent_service.archive_session("abc123")

        mock_session_repo.archive.assert_called_once()
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio()
    async def test_archive_session_not_found(
        self,
        agent_service: AgentService,
        mock_session_repo: AsyncMock,
    ) -> None:
        """Test archiving non-existent session raises error."""
        mock_session_repo.get_by_external_id.return_value = None

        with pytest.raises(SessionNotFoundError):
            await agent_service.archive_session("non-existent")


class TestDeleteSession:
    """Test delete_session method."""

    @pytest.mark.asyncio()
    async def test_delete_session(
        self,
        agent_service: AgentService,
        mock_db_session: AsyncMock,
        mock_session_repo: AsyncMock,
        mock_agent_session: MagicMock,
    ) -> None:
        """Test soft-deleting a session."""
        mock_session_repo.get_by_external_id.return_value = mock_agent_session

        await agent_service.delete_session("abc123")

        mock_session_repo.delete.assert_called_once()
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio()
    async def test_hard_delete_session(
        self,
        agent_service: AgentService,
        mock_db_session: AsyncMock,
        mock_session_repo: AsyncMock,
        mock_agent_session: MagicMock,
    ) -> None:
        """Test permanently deleting a session."""
        mock_session_repo.get_by_external_id.return_value = mock_agent_session

        await agent_service.hard_delete_session("abc123")

        mock_session_repo.hard_delete.assert_called_once()
        mock_db_session.commit.assert_called()


class TestCleanupOldSessions:
    """Test cleanup_old_sessions method."""

    @pytest.mark.asyncio()
    async def test_cleanup_old_sessions(
        self,
        agent_service: AgentService,
        mock_db_session: AsyncMock,
        mock_session_repo: AsyncMock,
    ) -> None:
        """Test cleaning up old deleted sessions."""
        mock_session_repo.cleanup_old_sessions.return_value = 5

        result = await agent_service.cleanup_old_sessions(older_than_days=90)

        assert result == 5
        mock_session_repo.cleanup_old_sessions.assert_called_once_with(older_than_days=90, status="deleted")
        mock_db_session.commit.assert_called()


class TestInterrupt:
    """Test interrupt method."""

    @pytest.mark.asyncio()
    async def test_interrupt_active_execution(
        self,
        agent_service: AgentService,
        mock_session_repo: AsyncMock,
        mock_agent_session: MagicMock,
    ) -> None:
        """Test interrupting an active execution."""
        mock_instance = AsyncMock()
        mock_instance.interrupt = AsyncMock()
        agent_service._active_executions[mock_agent_session.id] = mock_instance
        mock_session_repo.get_by_external_id.return_value = mock_agent_session

        await agent_service.interrupt("abc123")

        mock_instance.interrupt.assert_called_once()

    @pytest.mark.asyncio()
    async def test_interrupt_no_active_execution(
        self,
        agent_service: AgentService,
        mock_session_repo: AsyncMock,
        mock_agent_session: MagicMock,
    ) -> None:
        """Test interrupting when no active execution exists (no-op)."""
        mock_session_repo.get_by_external_id.return_value = mock_agent_session

        # Should not raise
        await agent_service.interrupt("abc123")

    @pytest.mark.asyncio()
    async def test_interrupt_session_not_found(
        self,
        agent_service: AgentService,
        mock_session_repo: AsyncMock,
    ) -> None:
        """Test interrupting non-existent session raises error."""
        mock_session_repo.get_by_external_id.return_value = None

        with pytest.raises(SessionNotFoundError):
            await agent_service.interrupt("non-existent")


class TestResolveTools:
    """Test _resolve_tools method."""

    def test_resolve_tools_none(
        self,
        agent_service: AgentService,
    ) -> None:
        """Test resolve tools with no tools requested."""
        result = agent_service._resolve_tools(None)
        assert result is None

    def test_resolve_tools_empty_list(
        self,
        agent_service: AgentService,
    ) -> None:
        """Test resolve tools with empty list."""
        result = agent_service._resolve_tools([])
        assert result is None

    def test_resolve_tools_found(
        self,
        agent_service: AgentService,
    ) -> None:
        """Test resolve tools when tools exist."""
        mock_tool = MagicMock()
        mock_tool.name = "search"

        with patch("webui.services.agent_service.get_tool_registry") as mock_registry:
            mock_registry.return_value.get_by_names.return_value = [mock_tool]

            result = agent_service._resolve_tools(["search"])

            assert result == [mock_tool]
            mock_registry.return_value.get_by_names.assert_called_once_with(["search"])

    def test_resolve_tools_partial_match(
        self,
        agent_service: AgentService,
    ) -> None:
        """Test resolve tools with partial match logs warning."""
        mock_tool = MagicMock()
        mock_tool.name = "search"

        with patch("webui.services.agent_service.get_tool_registry") as mock_registry:
            mock_registry.return_value.get_by_names.return_value = [mock_tool]

            result = agent_service._resolve_tools(["search", "non_existent"])

            assert result == [mock_tool]
