"""Integration tests for SSE streaming endpoint.

These tests verify the streaming API endpoint returns properly formatted
Server-Sent Events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from shared.database import get_db
from shared.database.models import Collection, CollectionSource, CollectionStatus, LLMProviderConfig
from shared.database.repositories.llm_provider_config_repository import LLMProviderConfigRepository
from shared.utils.encryption import SecretEncryption, generate_fernet_key
from webui.auth import create_access_token, get_current_user
from webui.main import app
from webui.services.agent.models import AgentConversation, ConversationStatus

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

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
    collection = Collection(
        id=str(uuid4()),
        name="Test Collection",
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
async def llm_config(db_session: AsyncSession, test_user_db: User) -> LLMProviderConfig:
    """Create LLM config for testing."""
    config = LLMProviderConfig(
        user_id=test_user_db.id,
        high_quality_provider="anthropic",
        high_quality_model="claude-opus-4-5-20251101",
        low_quality_provider="anthropic",
        low_quality_model="claude-sonnet-4-5-20250929",
    )
    db_session.add(config)
    await db_session.flush()
    return config


@pytest_asyncio.fixture
async def llm_config_with_key(db_session: AsyncSession, llm_config: LLMProviderConfig) -> LLMProviderConfig:
    """LLM config with an API key."""
    repo = LLMProviderConfigRepository(db_session)
    await repo.set_api_key(llm_config.id, "anthropic", "sk-ant-test-key")
    await db_session.flush()
    return llm_config


@pytest_asyncio.fixture
async def test_conversation(
    db_session: AsyncSession, test_user_db: User, test_source: CollectionSource
) -> AgentConversation:
    """Create a test conversation."""
    from webui.services.agent.repository import AgentConversationRepository

    repo = AgentConversationRepository(db_session)
    conv = await repo.create(
        user_id=test_user_db.id,
        source_id=test_source.id,
    )
    await db_session.flush()
    return conv


@pytest.fixture()
def api_auth_headers(test_user_db) -> dict[str, str]:
    """Create auth headers for API requests."""
    token = create_access_token(data={"sub": test_user_db.username})
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def api_client(
    db_session,
    test_user_db,
    use_fakeredis,
) -> AsyncGenerator[AsyncClient, None]:
    """Create an async client for API testing."""
    _ = use_fakeredis

    async def override_get_db() -> AsyncGenerator[Any, None]:
        yield db_session

    async def override_get_current_user() -> dict[str, Any]:
        return {
            "id": test_user_db.id,
            "username": test_user_db.username,
            "email": test_user_db.email,
            "full_name": test_user_db.full_name,
        }

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


class TestStreamingEndpoint:
    """Tests for the SSE streaming endpoint."""

    @pytest.mark.asyncio()
    async def test_returns_sse_content_type(self, api_client, api_auth_headers, test_conversation, llm_config_with_key):
        """Streaming endpoint returns text/event-stream content type."""
        # Mock the LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Hello, I can help you with that."

        mock_provider = AsyncMock()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=None)
        mock_provider.generate = AsyncMock(return_value=mock_llm_response)

        with patch(
            "webui.services.agent.orchestrator.LLMServiceFactory.create_provider_for_tier",
            new_callable=lambda: AsyncMock(return_value=mock_provider),
        ):
            response = await api_client.post(
                f"/api/v2/agent/conversations/{test_conversation.id}/messages/stream",
                json={"message": "Hello"},
                headers=api_auth_headers,
            )

        # Should return 200 with event stream type
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    @pytest.mark.asyncio()
    async def test_stream_events_are_properly_formatted(
        self, api_client, api_auth_headers, test_conversation, llm_config_with_key
    ):
        """SSE events follow the event:type/data:json format."""
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Test response"

        mock_provider = AsyncMock()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=None)
        mock_provider.generate = AsyncMock(return_value=mock_llm_response)

        with patch(
            "webui.services.agent.orchestrator.LLMServiceFactory.create_provider_for_tier",
            new_callable=lambda: AsyncMock(return_value=mock_provider),
        ):
            response = await api_client.post(
                f"/api/v2/agent/conversations/{test_conversation.id}/messages/stream",
                json={"message": "Test"},
                headers=api_auth_headers,
            )

        # Parse SSE events from response
        text = response.text
        events = []
        current_event = {}

        for line in text.split("\n"):
            if line.startswith("event:"):
                current_event["type"] = line[6:].strip()
            elif line.startswith("data:"):
                current_event["data"] = line[5:].strip()
            elif line == "" and current_event:
                events.append(current_event)
                current_event = {}

        # Should have at least some events
        assert len(events) >= 1

        # Each event should have type and data
        for event in events:
            assert "type" in event
            assert "data" in event

    @pytest.mark.asyncio()
    async def test_stream_includes_done_event(
        self, api_client, api_auth_headers, test_conversation, llm_config_with_key
    ):
        """Stream ends with a done event."""
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Done!"

        mock_provider = AsyncMock()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=None)
        mock_provider.generate = AsyncMock(return_value=mock_llm_response)

        with patch(
            "webui.services.agent.orchestrator.LLMServiceFactory.create_provider_for_tier",
            new_callable=lambda: AsyncMock(return_value=mock_provider),
        ):
            response = await api_client.post(
                f"/api/v2/agent/conversations/{test_conversation.id}/messages/stream",
                json={"message": "Hello"},
                headers=api_auth_headers,
            )

        text = response.text
        assert "event: done" in text or "event:done" in text

    @pytest.mark.asyncio()
    async def test_returns_404_for_missing_conversation(self, api_client, api_auth_headers, llm_config_with_key):
        """Returns 404 when conversation doesn't exist."""
        response = await api_client.post(
            "/api/v2/agent/conversations/nonexistent-uuid/messages/stream",
            json={"message": "Hello"},
            headers=api_auth_headers,
        )

        assert response.status_code == 404

    @pytest.mark.asyncio()
    async def test_stream_headers_disable_buffering(
        self, api_client, api_auth_headers, test_conversation, llm_config_with_key
    ):
        """Response headers disable nginx buffering for proper streaming."""
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Hello"

        mock_provider = AsyncMock()
        mock_provider.__aenter__ = AsyncMock(return_value=mock_provider)
        mock_provider.__aexit__ = AsyncMock(return_value=None)
        mock_provider.generate = AsyncMock(return_value=mock_llm_response)

        with patch(
            "webui.services.agent.orchestrator.LLMServiceFactory.create_provider_for_tier",
            new_callable=lambda: AsyncMock(return_value=mock_provider),
        ):
            response = await api_client.post(
                f"/api/v2/agent/conversations/{test_conversation.id}/messages/stream",
                json={"message": "Hello"},
                headers=api_auth_headers,
            )

        # Check streaming-related headers
        assert response.headers.get("cache-control") == "no-cache"
        assert response.headers.get("x-accel-buffering") == "no"


class TestStreamingWithInactiveConversation:
    """Tests for streaming with inactive conversations."""

    @pytest_asyncio.fixture
    async def inactive_conversation(
        self, db_session: AsyncSession, test_user_db: User, test_source: CollectionSource
    ) -> AgentConversation:
        """Create an inactive conversation."""
        from webui.services.agent.repository import AgentConversationRepository

        repo = AgentConversationRepository(db_session)
        conv = await repo.create(
            user_id=test_user_db.id,
            source_id=test_source.id,
        )
        conv.status = ConversationStatus.ABANDONED
        await db_session.flush()
        return conv

    @pytest.mark.asyncio()
    async def test_stream_returns_error_for_inactive(
        self, api_client, api_auth_headers, inactive_conversation, llm_config_with_key
    ):
        """Stream returns error event for inactive conversation."""
        response = await api_client.post(
            f"/api/v2/agent/conversations/{inactive_conversation.id}/messages/stream",
            json={"message": "Hello"},
            headers=api_auth_headers,
        )

        # Should still return 200 (SSE sends errors in-band)
        assert response.status_code == 200
        assert "error" in response.text.lower()
