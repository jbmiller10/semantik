"""Unit tests for AgentSessionRepository."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.database.agent_session import AgentSession, AgentSessionMessage
from shared.database.exceptions import EntityNotFoundError
from shared.database.repositories.agent_session_repository import AgentSessionRepository


class TestAgentSessionRepository:
    """Tests for AgentSessionRepository."""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.flush = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.fixture()
    def repo(self, mock_session):
        """Create a repository with mocked session."""
        return AgentSessionRepository(mock_session)

    @pytest.fixture()
    def mock_agent_session(self):
        """Create a mock AgentSession."""
        session = MagicMock(spec=AgentSession)
        session.id = "test-session-id"
        session.external_id = "abc12345"
        session.title = "Test Session"
        session.user_id = 1
        session.agent_plugin_id = "claude-agent"
        session.agent_config = {"model": "claude-sonnet-4"}
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
        session.created_at = datetime.now(UTC)
        session.updated_at = datetime.now(UTC)
        session.last_activity_at = datetime.now(UTC)
        session.archived_at = None
        return session

    @pytest.fixture()
    def mock_agent_message(self):
        """Create a mock AgentMessage."""
        message = MagicMock()
        message.id = "msg-123"
        message.role = MagicMock()
        message.role.value = "assistant"
        message.type = MagicMock()
        message.type.value = "text"
        message.content = "Hello, world!"
        message.tool_name = None
        message.tool_call_id = None
        message.tool_input = None
        message.tool_output = None
        message.model = "claude-sonnet-4"
        message.usage = None
        message.cost_usd = None
        message.sequence_number = 0
        message.to_dict = MagicMock(
            return_value={
                "id": "msg-123",
                "role": "assistant",
                "type": "text",
                "content": "Hello, world!",
            }
        )
        return message

    @pytest.mark.asyncio()
    async def test_create_session(self, repo, mock_session):
        """Test create creates a new session."""
        result = await repo.create(
            agent_plugin_id="claude-agent",
            user_id=1,
            title="Test Session",
        )

        assert result is not None
        assert result.agent_plugin_id == "claude-agent"
        assert result.user_id == 1
        assert result.title == "Test Session"
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_create_session_with_parent(self, repo, mock_session):
        """Test create session with parent (fork)."""
        result = await repo.create(
            agent_plugin_id="claude-agent",
            user_id=1,
            parent_session_id="parent-session-id",
        )

        assert result.parent_session_id == "parent-session-id"

    @pytest.mark.asyncio()
    async def test_get_by_id_found(self, repo, mock_session, mock_agent_session):
        """Test get_by_id returns session when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_agent_session
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_by_id("test-session-id")

        assert result is not None
        assert result.id == "test-session-id"

    @pytest.mark.asyncio()
    async def test_get_by_id_not_found(self, repo, mock_session):
        """Test get_by_id returns None when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_by_id("nonexistent")

        assert result is None

    @pytest.mark.asyncio()
    async def test_get_by_external_id_found(self, repo, mock_session, mock_agent_session):
        """Test get_by_external_id returns session when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_agent_session
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_by_external_id("abc12345")

        assert result is not None
        assert result.external_id == "abc12345"

    @pytest.mark.asyncio()
    async def test_get_by_sdk_session_id_found(self, repo, mock_session, mock_agent_session):
        """Test get_by_sdk_session_id returns session when found."""
        mock_agent_session.sdk_session_id = "sdk-session-123"
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_agent_session
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_by_sdk_session_id("sdk-session-123")

        assert result is not None

    @pytest.mark.asyncio()
    async def test_list_by_user(self, repo, mock_session, mock_agent_session):
        """Test list_by_user returns sessions with pagination."""
        # Mock the list query result
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value.all.return_value = [mock_agent_session]

        mock_session.execute = AsyncMock(return_value=mock_list_result)
        mock_session.scalar = AsyncMock(return_value=1)

        sessions, total = await repo.list_by_user(user_id=1)

        assert len(sessions) == 1
        assert total == 1

    @pytest.mark.asyncio()
    async def test_list_by_user_with_status_filter(self, repo, mock_session, mock_agent_session):
        """Test list_by_user with status filter."""
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value.all.return_value = [mock_agent_session]
        mock_session.execute = AsyncMock(return_value=mock_list_result)
        mock_session.scalar = AsyncMock(return_value=1)

        sessions, total = await repo.list_by_user(user_id=1, status="active")

        assert len(sessions) == 1
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_list_by_user_with_collection_filter(self, repo, mock_session, mock_agent_session):
        """Test list_by_user with collection filter."""
        mock_agent_session.collection_id = "collection-123"
        mock_list_result = MagicMock()
        mock_list_result.scalars.return_value.all.return_value = [mock_agent_session]
        mock_session.execute = AsyncMock(return_value=mock_list_result)
        mock_session.scalar = AsyncMock(return_value=1)

        sessions, total = await repo.list_by_user(user_id=1, collection_id="collection-123")

        assert len(sessions) == 1

    @pytest.mark.asyncio()
    async def test_add_message(self, repo, mock_session, mock_agent_session, mock_agent_message):
        """Test add_message adds message to session."""
        # Mock get_by_id to return the session
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_agent_session
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repo.add_message("test-session-id", mock_agent_message)

        assert result is not None
        mock_agent_session.add_message.assert_called_once()
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_add_message_session_not_found(self, repo, mock_session, mock_agent_message):
        """Test add_message raises error when session not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(EntityNotFoundError):
            await repo.add_message("nonexistent", mock_agent_message)

    @pytest.mark.asyncio()
    async def test_add_message_with_usage(self, repo, mock_session, mock_agent_session, mock_agent_message):
        """Test add_message updates stats when usage provided."""
        # Add usage to the message
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 200
        mock_agent_message.usage = mock_usage
        mock_agent_message.cost_usd = 0.001

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_agent_session
        mock_session.execute = AsyncMock(return_value=mock_result)

        await repo.add_message("test-session-id", mock_agent_message)

        mock_agent_session.update_stats.assert_called_once_with(
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
        )

    @pytest.mark.asyncio()
    async def test_get_messages(self, repo, mock_session):
        """Test get_messages returns messages for session."""
        mock_message = MagicMock(spec=AgentSessionMessage)
        mock_message.id = "msg-1"
        mock_message.sequence_number = 0

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_message]
        mock_session.execute = AsyncMock(return_value=mock_result)

        messages = await repo.get_messages("test-session-id")

        assert len(messages) == 1
        assert messages[0].id == "msg-1"

    @pytest.mark.asyncio()
    async def test_get_messages_with_after_sequence(self, repo, mock_session):
        """Test get_messages with after_sequence filter."""
        mock_message = MagicMock(spec=AgentSessionMessage)
        mock_message.id = "msg-2"
        mock_message.sequence_number = 1

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_message]
        mock_session.execute = AsyncMock(return_value=mock_result)

        messages = await repo.get_messages("test-session-id", after_sequence=0)

        assert len(messages) == 1
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_update_sdk_session_id(self, repo, mock_session):
        """Test update_sdk_session_id updates the SDK session ID."""
        mock_session.execute = AsyncMock()

        await repo.update_sdk_session_id("test-session-id", "new-sdk-session")

        mock_session.execute.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_update_title(self, repo, mock_session):
        """Test update_title updates the session title."""
        mock_session.execute = AsyncMock()

        await repo.update_title("test-session-id", "New Title")

        mock_session.execute.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_archive(self, repo, mock_session):
        """Test archive sets status to archived."""
        mock_session.execute = AsyncMock()

        await repo.archive("test-session-id")

        mock_session.execute.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_delete(self, repo, mock_session):
        """Test delete sets status to deleted (soft delete)."""
        mock_session.execute = AsyncMock()

        await repo.delete("test-session-id")

        mock_session.execute.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_hard_delete(self, repo, mock_session):
        """Test hard_delete permanently deletes session and messages."""
        mock_session.execute = AsyncMock()

        await repo.hard_delete("test-session-id")

        # Should be called twice: once for messages, once for session
        assert mock_session.execute.call_count == 2
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_cleanup_old_sessions(self, repo, mock_session):
        """Test cleanup_old_sessions deletes old deleted sessions."""
        # Mock finding sessions to delete
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("session-1",), ("session-2",)]
        mock_session.execute = AsyncMock(return_value=mock_result)

        count = await repo.cleanup_old_sessions(older_than_days=90, status="deleted")

        assert count == 2
        # Called 3 times: select, delete messages, delete sessions
        assert mock_session.execute.call_count == 3

    @pytest.mark.asyncio()
    async def test_cleanup_old_sessions_none_found(self, repo, mock_session):
        """Test cleanup_old_sessions returns 0 when no sessions found."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        count = await repo.cleanup_old_sessions(older_than_days=90)

        assert count == 0
        # Only called once for the select
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_increment_fork_count(self, repo, mock_session):
        """Test increment_fork_count increments the fork count."""
        mock_session.execute = AsyncMock()

        await repo.increment_fork_count("test-session-id")

        mock_session.execute.assert_called_once()
        mock_session.flush.assert_called_once()


class TestAgentSessionModel:
    """Tests for AgentSession model methods."""

    def test_add_message(self):
        """Test add_message adds to messages list."""
        session = AgentSession(
            id="test-id",
            external_id="abc123",
            agent_plugin_id="claude-agent",
        )
        session.messages = []

        session.add_message({"id": "msg-1", "content": "Hello"})

        assert len(session.messages) == 1
        assert session.message_count == 1

    def test_add_message_initializes_empty_list(self):
        """Test add_message initializes messages if None."""
        session = AgentSession(
            id="test-id",
            external_id="abc123",
            agent_plugin_id="claude-agent",
        )
        session.messages = None

        session.add_message({"id": "msg-1", "content": "Hello"})

        assert session.messages is not None
        assert len(session.messages) == 1

    def test_update_stats(self):
        """Test update_stats accumulates statistics."""
        session = AgentSession(
            id="test-id",
            external_id="abc123",
            agent_plugin_id="claude-agent",
        )
        session.total_input_tokens = 0
        session.total_output_tokens = 0
        session.total_cost_usd = 0

        session.update_stats(input_tokens=100, output_tokens=200, cost_usd=0.001)

        assert session.total_input_tokens == 100
        assert session.total_output_tokens == 200
        assert session.total_cost_usd == 10  # 0.001 * 10000

    def test_to_dict(self):
        """Test to_dict serializes session."""
        session = AgentSession(
            id="test-id",
            external_id="abc123",
            agent_plugin_id="claude-agent",
            title="Test Session",
            user_id=1,
            message_count=5,
            total_cost_usd=1000,  # $0.10 stored as cents * 100
        )
        session.created_at = datetime(2026, 1, 7, 10, 0, 0, tzinfo=UTC)
        session.updated_at = datetime(2026, 1, 7, 10, 0, 0, tzinfo=UTC)
        session.last_activity_at = datetime(2026, 1, 7, 10, 0, 0, tzinfo=UTC)

        result = session.to_dict()

        assert result["id"] == "test-id"
        assert result["external_id"] == "abc123"
        assert result["title"] == "Test Session"
        assert result["message_count"] == 5
        assert result["total_cost_usd"] == 0.10


class TestAgentSessionMessageModel:
    """Tests for AgentSessionMessage model methods."""

    def test_to_dict(self):
        """Test to_dict serializes message."""
        message = AgentSessionMessage(
            id="msg-id",
            session_id="session-id",
            message_id="msg-123",
            sequence_number=0,
            role="assistant",
            type="text",
            content="Hello, world!",
            cost_usd=100,  # $0.01 stored as cents * 100
        )
        message.created_at = datetime(2026, 1, 7, 10, 0, 0, tzinfo=UTC)

        result = message.to_dict()

        assert result["id"] == "msg-id"
        assert result["session_id"] == "session-id"
        assert result["role"] == "assistant"
        assert result["content"] == "Hello, world!"
        assert result["cost_usd"] == 0.01
