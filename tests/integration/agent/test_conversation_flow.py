"""Integration tests for end-to-end conversation flows.

These tests verify complete conversation workflows from start to finish.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from shared.database.models import Collection, CollectionSource, CollectionStatus, LLMProviderConfig
from shared.database.repositories.llm_provider_config_repository import LLMProviderConfigRepository
from shared.utils.encryption import SecretEncryption, generate_fernet_key
from webui.services.agent.message_store import ConversationMessage, MessageStore
from webui.services.agent.models import AgentConversation, ConversationStatus, UncertaintySeverity
from webui.services.agent.orchestrator import AgentOrchestrator
from webui.services.agent.repository import AgentConversationRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.models import User


@pytest.fixture(autouse=True)
def _initialize_encryption():
    """Initialize encryption for all tests."""
    test_key = generate_fernet_key()
    SecretEncryption.initialize(test_key)
    yield
    SecretEncryption.reset()


@pytest_asyncio.fixture
async def test_source(db_session: AsyncSession, test_user_db: User) -> CollectionSource:
    """Create a test collection source with required parent collection."""
    # First create a Collection (CollectionSource requires collection_id)
    # Use UUID in name to avoid unique constraint violations in parallel tests
    collection_uuid = str(uuid4())
    collection = Collection(
        id=collection_uuid,
        name=f"Test Collection {collection_uuid[:8]}",
        description="Test collection for agent tests",
        vector_store_name=f"col_{uuid4().hex[:16]}",
        embedding_model="test-model",
        owner_id=test_user_db.id,
        status=CollectionStatus.READY,
    )
    db_session.add(collection)
    await db_session.flush()

    # Create the source with correct field names
    source = CollectionSource(
        collection_id=collection.id,
        source_path="/test/path",
        source_type="local",
        source_config={"path": "/test/path"},
    )
    db_session.add(source)
    await db_session.flush()
    return source


@pytest_asyncio.fixture
async def llm_config_with_key(db_session: AsyncSession, test_user_db: User) -> LLMProviderConfig:
    """Create LLM config with API key."""
    config = LLMProviderConfig(
        user_id=test_user_db.id,
        high_quality_provider="anthropic",
        high_quality_model="claude-opus-4-5-20251101",
        low_quality_provider="anthropic",
        low_quality_model="claude-sonnet-4-5-20250929",
    )
    db_session.add(config)
    await db_session.flush()

    repo = LLMProviderConfigRepository(db_session)
    await repo.set_api_key(config.id, "anthropic", "sk-ant-test-key")
    await db_session.flush()
    return config


@pytest_asyncio.fixture
async def test_conversation(
    db_session: AsyncSession, test_user_db: User, test_source: CollectionSource
) -> AgentConversation:
    """Create a test conversation."""
    repo = AgentConversationRepository(db_session)
    conv = await repo.create(
        user_id=test_user_db.id,
        source_id=test_source.id,
    )
    await db_session.commit()
    return conv


class TestConversationLifecycle:
    """Tests for conversation lifecycle management."""

    @pytest.mark.asyncio()
    async def test_conversation_creation_and_status(self, db_session, test_user_db, test_source):
        """Conversation is created with active status."""
        repo = AgentConversationRepository(db_session)

        conv = await repo.create(
            user_id=test_user_db.id,
            source_id=test_source.id,
        )
        await db_session.commit()

        # Verify initial state
        assert conv.status == ConversationStatus.ACTIVE
        assert conv.source_id == test_source.id
        assert conv.current_pipeline is None
        assert conv.source_analysis is None

    @pytest.mark.asyncio()
    async def test_conversation_status_transitions(self, db_session, test_user_db, test_conversation):
        """Conversation can transition through statuses."""
        repo = AgentConversationRepository(db_session)

        # Transition to abandoned
        updated = await repo.update_status(
            conversation_id=test_conversation.id,
            user_id=test_user_db.id,
            status=ConversationStatus.ABANDONED,
        )
        await db_session.commit()

        assert updated.status == ConversationStatus.ABANDONED

    @pytest.mark.asyncio()
    async def test_pipeline_persistence(self, db_session, test_user_db, test_conversation):
        """Pipeline configuration is persisted correctly."""
        repo = AgentConversationRepository(db_session)

        pipeline_config = {
            "id": "test-pipeline",
            "version": "1.0",
            "nodes": [{"id": "chunker", "type": "chunker", "plugin_id": "semantic"}],
            "edges": [],
        }

        await repo.update_pipeline(
            conversation_id=test_conversation.id,
            user_id=test_user_db.id,
            pipeline_config=pipeline_config,
        )
        await db_session.commit()

        # Fetch fresh
        updated = await repo.get_by_id(test_conversation.id)
        assert updated.current_pipeline == pipeline_config


class TestUncertaintyManagement:
    """Tests for uncertainty tracking and resolution."""

    @pytest.mark.asyncio()
    async def test_adds_uncertainty(self, db_session, test_user_db, test_conversation):
        """Uncertainties can be added to conversations."""
        repo = AgentConversationRepository(db_session)

        uncertainty = await repo.add_uncertainty(
            conversation_id=test_conversation.id,
            user_id=test_user_db.id,
            severity=UncertaintySeverity.BLOCKING,
            message="Document format not recognized",
            context={"file": "test.xyz"},
        )
        await db_session.commit()

        assert uncertainty.severity == UncertaintySeverity.BLOCKING
        assert not uncertainty.resolved

    @pytest.mark.asyncio()
    async def test_resolves_uncertainty(self, db_session, test_user_db, test_conversation):
        """Uncertainties can be resolved."""
        repo = AgentConversationRepository(db_session)

        uncertainty = await repo.add_uncertainty(
            conversation_id=test_conversation.id,
            user_id=test_user_db.id,
            severity=UncertaintySeverity.NOTABLE,
            message="Large file warning",
        )
        await db_session.flush()

        resolved = await repo.resolve_uncertainty(
            uncertainty_id=uncertainty.id,
            resolved_by="user_confirmed",
        )
        await db_session.commit()

        assert resolved.resolved
        assert resolved.resolved_by == "user_confirmed"

    @pytest.mark.asyncio()
    async def test_gets_blocking_uncertainties(self, db_session, test_user_db, test_conversation):
        """Can fetch only blocking uncertainties."""
        repo = AgentConversationRepository(db_session)

        # Add blocking and non-blocking
        await repo.add_uncertainty(
            conversation_id=test_conversation.id,
            user_id=test_user_db.id,
            severity=UncertaintySeverity.BLOCKING,
            message="Critical issue",
        )
        await repo.add_uncertainty(
            conversation_id=test_conversation.id,
            user_id=test_user_db.id,
            severity=UncertaintySeverity.INFO,
            message="FYI",
        )
        await db_session.commit()

        blocking = await repo.get_blocking_uncertainties(test_conversation.id)

        assert len(blocking) == 1
        assert blocking[0].severity == UncertaintySeverity.BLOCKING


class TestMessageStorePersistence:
    """Tests for Redis message store integration."""

    @pytest_asyncio.fixture
    async def message_store(self):
        """Create a message store with fake Redis."""
        import fakeredis.aioredis

        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        return MessageStore(redis_client)

    @pytest.mark.asyncio()
    async def test_message_roundtrip(self, message_store):
        """Messages can be stored and retrieved."""
        store = message_store
        conv_id = "test-conv-123"

        # Add messages
        msg1 = ConversationMessage.create("user", "Hello")
        msg2 = ConversationMessage.create("assistant", "Hi there!")

        await store.append_message(conv_id, msg1)
        await store.append_message(conv_id, msg2)

        # Retrieve
        messages = await store.get_messages(conv_id)

        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there!"

    @pytest.mark.asyncio()
    async def test_message_count(self, message_store):
        """Message count is tracked correctly."""
        store = message_store
        conv_id = "test-conv-456"

        for i in range(5):
            await store.append_message(
                conv_id,
                ConversationMessage.create("user", f"Message {i}"),
            )

        count = await store.get_message_count(conv_id)
        assert count == 5

    @pytest.mark.asyncio()
    async def test_conversation_lock(self, message_store):
        """Conversation locking works correctly."""
        store = message_store
        conv_id = "test-conv-789"

        # Acquire lock
        acquired = await store.acquire_lock(conv_id)
        assert acquired

        # Verify lock is held
        is_locked = await store.is_locked(conv_id)
        assert is_locked

        # Second acquire should fail
        acquired2 = await store.acquire_lock(conv_id)
        assert not acquired2

        # Release lock
        await store.release_lock(conv_id)

        # Verify lock is released
        is_locked_after = await store.is_locked(conv_id)
        assert not is_locked_after

        # Now reacquire should work
        acquired3 = await store.acquire_lock(conv_id)
        assert acquired3


class TestConversationRecovery:
    """Tests for conversation recovery from summaries."""

    @pytest.mark.asyncio()
    async def test_summary_persistence(self, db_session, test_user_db, test_conversation):
        """Conversation summary is persisted correctly."""
        repo = AgentConversationRepository(db_session)

        summary = "User configured semantic chunking with 512 token limit."
        await repo.update_summary(
            conversation_id=test_conversation.id,
            user_id=test_user_db.id,
            summary=summary,
        )
        await db_session.commit()

        updated = await repo.get_by_id(test_conversation.id)
        assert updated.summary == summary

    @pytest.mark.asyncio()
    async def test_source_analysis_persistence(self, db_session, test_user_db, test_conversation):
        """Source analysis results are persisted correctly."""
        repo = AgentConversationRepository(db_session)

        analysis = {
            "file_types": {"pdf": 10, "txt": 5},
            "total_size_mb": 25.5,
            "recommendations": ["Use PDF extractor"],
        }
        await repo.update_source_analysis(
            conversation_id=test_conversation.id,
            user_id=test_user_db.id,
            source_analysis=analysis,
        )
        await db_session.commit()

        updated = await repo.get_by_id(test_conversation.id)
        assert updated.source_analysis == analysis


class TestOrchestratorIntegration:
    """Tests for orchestrator with real database."""

    @pytest.mark.asyncio()
    async def test_orchestrator_tool_context(
        self, db_session, test_user_db, test_conversation, llm_config_with_key, use_fakeredis, fake_redis_client
    ):
        """Orchestrator builds correct tool context."""
        from shared.llm.factory import LLMServiceFactory

        store = MessageStore(fake_redis_client)
        factory = LLMServiceFactory(db_session)

        with patch("webui.services.agent.orchestrator.ORCHESTRATOR_TOOL_CLASSES", []):
            orch = AgentOrchestrator(
                conversation=test_conversation,
                session=db_session,
                llm_factory=factory,
                message_store=store,
            )

        context = orch._build_tool_context()

        assert context["session"] == db_session
        assert context["user_id"] == test_user_db.id
        assert context["conversation"] == test_conversation

    @pytest.mark.asyncio()
    async def test_orchestrator_handles_simple_response(
        self, db_session, test_user_db, test_conversation, llm_config_with_key, use_fakeredis, fake_redis_client
    ):
        """Orchestrator handles simple LLM response without tools."""
        from shared.llm.factory import LLMServiceFactory

        store = MessageStore(fake_redis_client)
        factory = LLMServiceFactory(db_session)

        # Mock LLM provider
        mock_response = MagicMock()
        mock_response.content = "I can help you configure your pipeline. What would you like to do?"

        mock_provider = AsyncMock()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=None)
        mock_provider.generate = AsyncMock(return_value=mock_response)

        with (
            patch("webui.services.agent.orchestrator.ORCHESTRATOR_TOOL_CLASSES", []),
            patch.object(factory, "create_provider_for_tier", return_value=mock_provider),
        ):
            orch = AgentOrchestrator(
                conversation=test_conversation,
                session=db_session,
                llm_factory=factory,
                message_store=store,
            )

            response = await orch.handle_message("Hello, I need help")

        assert "help you configure" in response.content
        assert response.pipeline_updated is False
