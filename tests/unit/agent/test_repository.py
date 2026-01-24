"""Unit tests for AgentConversationRepository."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from shared.database.exceptions import EntityNotFoundError
from webui.services.agent.models import (
    AgentConversation,
    ConversationStatus,
    ConversationUncertainty,
    UncertaintySeverity,
)
from webui.services.agent.repository import AgentConversationRepository


@pytest.fixture()
def mock_session():
    """Create a mock async session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture()
def repository(mock_session):
    """Create a repository with mock session."""
    return AgentConversationRepository(mock_session)


class TestAgentConversationRepository:
    """Tests for AgentConversationRepository."""

    @pytest.mark.asyncio()
    async def test_create_conversation(self, repository, mock_session):
        """Test creating a new conversation."""
        await repository.create(user_id=1)

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

        # Check the added conversation
        added = mock_session.add.call_args[0][0]
        assert isinstance(added, AgentConversation)
        assert added.user_id == 1
        assert added.status == ConversationStatus.ACTIVE
        assert added.source_id is None

    @pytest.mark.asyncio()
    async def test_create_conversation_with_source(self, repository, mock_session):
        """Test creating a conversation with a source."""
        await repository.create(user_id=1, source_id=42)

        added = mock_session.add.call_args[0][0]
        assert added.source_id == 42

    @pytest.mark.asyncio()
    async def test_get_by_id_found(self, repository, mock_session):
        """Test getting a conversation by ID when it exists."""
        conv_id = str(uuid4())
        mock_conv = AgentConversation(
            id=conv_id,
            user_id=1,
            status=ConversationStatus.ACTIVE,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_conv
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(conv_id)

        assert result == mock_conv
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_get_by_id_not_found(self, repository, mock_session):
        """Test getting a conversation by ID when it doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(str(uuid4()))

        assert result is None

    @pytest.mark.asyncio()
    async def test_get_by_id_for_user(self, repository, mock_session):
        """Test getting a conversation by ID with user check."""
        conv_id = str(uuid4())
        mock_conv = AgentConversation(
            id=conv_id,
            user_id=1,
            status=ConversationStatus.ACTIVE,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_conv
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id_for_user(conv_id, user_id=1)

        assert result == mock_conv

    @pytest.mark.asyncio()
    async def test_get_by_id_for_wrong_user(self, repository, mock_session):
        """Test that wrong user gets None."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id_for_user(str(uuid4()), user_id=999)

        assert result is None

    @pytest.mark.asyncio()
    async def test_list_for_user(self, repository, mock_session):
        """Test listing conversations for a user."""
        conversations = [
            AgentConversation(id=str(uuid4()), user_id=1, status=ConversationStatus.ACTIVE),
            AgentConversation(id=str(uuid4()), user_id=1, status=ConversationStatus.APPLIED),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = conversations
        mock_session.execute.return_value = mock_result

        result = await repository.list_for_user(user_id=1)

        assert len(result) == 2
        assert all(c.user_id == 1 for c in result)

    @pytest.mark.asyncio()
    async def test_update_status(self, repository, mock_session):
        """Test updating conversation status."""
        conv_id = str(uuid4())
        mock_conv = AgentConversation(
            id=conv_id,
            user_id=1,
            status=ConversationStatus.ACTIVE,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_conv
        mock_session.execute.return_value = mock_result

        result = await repository.update_status(conv_id, user_id=1, status=ConversationStatus.ABANDONED)

        assert result.status == ConversationStatus.ABANDONED
        mock_session.flush.assert_called()

    @pytest.mark.asyncio()
    async def test_update_status_not_found(self, repository, mock_session):
        """Test updating status of non-existent conversation."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(EntityNotFoundError):
            await repository.update_status(str(uuid4()), user_id=1, status=ConversationStatus.ABANDONED)

    @pytest.mark.asyncio()
    async def test_update_pipeline(self, repository, mock_session):
        """Test updating pipeline configuration."""
        conv_id = str(uuid4())
        mock_conv = AgentConversation(
            id=conv_id,
            user_id=1,
            status=ConversationStatus.ACTIVE,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_conv
        mock_session.execute.return_value = mock_result

        pipeline = {"stages": [{"id": "chunk", "type": "recursive"}]}
        result = await repository.update_pipeline(conv_id, user_id=1, pipeline_config=pipeline)

        assert result.current_pipeline == pipeline

    @pytest.mark.asyncio()
    async def test_add_uncertainty(self, repository, mock_session):
        """Test adding an uncertainty to a conversation."""
        conv_id = str(uuid4())
        mock_conv = AgentConversation(
            id=conv_id,
            user_id=1,
            status=ConversationStatus.ACTIVE,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_conv
        mock_session.execute.return_value = mock_result

        await repository.add_uncertainty(
            conversation_id=conv_id,
            user_id=1,
            severity=UncertaintySeverity.BLOCKING,
            message="Test uncertainty",
            context={"key": "value"},
        )

        mock_session.add.assert_called()
        added = mock_session.add.call_args[0][0]
        assert isinstance(added, ConversationUncertainty)
        assert added.severity == UncertaintySeverity.BLOCKING
        assert added.message == "Test uncertainty"

    @pytest.mark.asyncio()
    async def test_resolve_uncertainty(self, repository, mock_session):
        """Test resolving an uncertainty."""
        uncertainty_id = str(uuid4())
        mock_uncertainty = ConversationUncertainty(
            id=uncertainty_id,
            conversation_id=str(uuid4()),
            severity=UncertaintySeverity.BLOCKING,
            message="Test",
            resolved=False,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_uncertainty
        mock_session.execute.return_value = mock_result

        result = await repository.resolve_uncertainty(
            uncertainty_id=uncertainty_id,
            resolved_by="user_confirmed",
        )

        assert result.resolved is True
        assert result.resolved_by == "user_confirmed"

    @pytest.mark.asyncio()
    async def test_get_blocking_uncertainties(self, repository, mock_session):
        """Test getting unresolved blocking uncertainties."""
        conv_id = str(uuid4())
        uncertainties = [
            ConversationUncertainty(
                id=str(uuid4()),
                conversation_id=conv_id,
                severity=UncertaintySeverity.BLOCKING,
                message="Issue 1",
                resolved=False,
            ),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = uncertainties
        mock_session.execute.return_value = mock_result

        result = await repository.get_blocking_uncertainties(conv_id)

        assert len(result) == 1
        assert result[0].severity == UncertaintySeverity.BLOCKING
        assert result[0].resolved is False

    @pytest.mark.asyncio()
    async def test_set_collection(self, repository, mock_session):
        """Test linking conversation to created collection."""
        conv_id = str(uuid4())
        collection_id = str(uuid4())
        mock_conv = AgentConversation(
            id=conv_id,
            user_id=1,
            status=ConversationStatus.ACTIVE,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_conv
        mock_session.execute.return_value = mock_result

        result = await repository.set_collection(conv_id, user_id=1, collection_id=collection_id)

        assert result.collection_id == collection_id
        assert result.status == ConversationStatus.APPLIED
