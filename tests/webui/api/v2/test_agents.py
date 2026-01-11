"""Unit tests for the v2 agents API.

These tests mock the service layer to avoid database dependencies.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from slowapi.errors import RateLimitExceeded

from webui.api.v2.agents import router
from webui.auth import get_current_user
from webui.rate_limiter import limiter
from webui.services.factory import get_agent_service


def _rate_limit_exceeded_handler(
    request: Request,  # noqa: ARG001
    exc: RateLimitExceeded,
) -> JSONResponse:
    """Handle rate limit exceeded errors."""
    return JSONResponse(status_code=429, content={"detail": str(exc)})


# Create a minimal app for testing
def create_test_app():
    """Create a minimal FastAPI app for testing."""
    app = FastAPI()
    # Set up rate limiter (needed for @limiter.limit decorator)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(router)
    return app


@pytest.fixture()
def mock_user():
    """Create a mock user."""
    return {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
    }


@pytest.fixture()
def mock_other_user():
    """Create another mock user."""
    return {
        "id": 2,
        "username": "otheruser",
        "email": "other@example.com",
        "full_name": "Other User",
    }


@pytest.fixture()
def mock_agent_service():
    """Create a mock agent service."""
    return AsyncMock()


@pytest.fixture()
def test_app(mock_user, mock_agent_service):
    """Create test app with mocked auth and service."""
    app = create_test_app()

    async def override_get_current_user():
        return mock_user

    async def override_get_agent_service():
        return mock_agent_service

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_agent_service] = override_get_agent_service
    return app


@pytest_asyncio.fixture
async def test_client(test_app):
    """Create async test client."""
    transport = ASGITransport(app=test_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# =============================================================================
# Agent Discovery Tests
# =============================================================================


@pytest.mark.asyncio()
async def test_list_agents_returns_registered_agents(test_client, mock_agent_service) -> None:
    """GET /api/v2/agents should return list of registered agent plugins."""
    mock_agents = [
        {
            "id": "test-agent",
            "version": "1.0.0",
            "manifest": {
                "display_name": "Test Agent",
                "description": "A test agent for testing",
            },
            "capabilities": {
                "supports_streaming": True,
                "supports_tools": True,
            },
            "use_cases": ["assistant", "tool_use"],
        }
    ]
    mock_agent_service.list_agents = AsyncMock(return_value=mock_agents)

    response = await test_client.get("/api/v2/agents")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert len(body["agents"]) == 1
    assert body["agents"][0]["id"] == "test-agent"
    assert body["agents"][0]["display_name"] == "Test Agent"


@pytest.mark.asyncio()
async def test_list_agents_empty_when_none_registered(test_client, mock_agent_service) -> None:
    """GET /api/v2/agents should return empty list when no agents registered."""
    mock_agent_service.list_agents = AsyncMock(return_value=[])

    response = await test_client.get("/api/v2/agents")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["agents"] == []


@pytest.mark.asyncio()
async def test_get_agent_returns_details(test_client, mock_agent_service) -> None:
    """GET /api/v2/agents/{agent_id} should return agent details."""
    mock_agent = {
        "id": "test-agent",
        "version": "1.0.0",
        "manifest": {
            "display_name": "Test Agent",
            "description": "A test agent",
        },
        "capabilities": {
            "supports_streaming": True,
        },
        "use_cases": ["assistant"],
        "config_schema": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
            },
        },
    }
    mock_agent_service.get_agent = AsyncMock(return_value=mock_agent)

    response = await test_client.get("/api/v2/agents/test-agent")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "test-agent"
    assert body["config_schema"] is not None


@pytest.mark.asyncio()
async def test_get_agent_not_found(test_client, mock_agent_service) -> None:
    """GET /api/v2/agents/{agent_id} should return 404 for unknown agent."""
    mock_agent_service.get_agent = AsyncMock(return_value=None)

    response = await test_client.get("/api/v2/agents/nonexistent")

    assert response.status_code == 404


# =============================================================================
# Session Management Tests
# =============================================================================


@pytest.mark.asyncio()
async def test_list_sessions_returns_user_sessions(test_client, mock_user, mock_agent_service) -> None:
    """GET /api/v2/agents/sessions should return user's sessions."""
    mock_sessions = [
        {
            "id": "session-uuid-1",
            "external_id": "abc12345",
            "title": "Test Session",
            "agent_plugin_id": "test-agent",
            "message_count": 5,
            "total_input_tokens": 100,
            "total_output_tokens": 200,
            "total_cost_usd": 0.01,
            "status": "active",
            "created_at": "2025-01-06T10:00:00Z",
            "last_activity_at": "2025-01-06T10:05:00Z",
            "user_id": mock_user["id"],
        }
    ]
    mock_agent_service.list_sessions = AsyncMock(return_value=(mock_sessions, 1))

    response = await test_client.get("/api/v2/agents/sessions")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert len(body["sessions"]) == 1
    assert body["sessions"][0]["external_id"] == "abc12345"


@pytest.mark.asyncio()
async def test_get_session_returns_details(test_client, mock_user, mock_agent_service) -> None:
    """GET /api/v2/agents/sessions/{session_id} should return session details."""
    mock_session = {
        "id": "session-uuid-1",
        "external_id": "abc12345",
        "title": "Test Session",
        "agent_plugin_id": "test-agent",
        "message_count": 5,
        "total_input_tokens": 100,
        "total_output_tokens": 200,
        "total_cost_usd": 0.01,
        "status": "active",
        "created_at": "2025-01-06T10:00:00Z",
        "last_activity_at": "2025-01-06T10:05:00Z",
        "user_id": mock_user["id"],
    }
    mock_agent_service.get_session = AsyncMock(return_value=mock_session)

    response = await test_client.get("/api/v2/agents/sessions/abc12345")

    assert response.status_code == 200
    body = response.json()
    assert body["external_id"] == "abc12345"
    assert body["message_count"] == 5


@pytest.mark.asyncio()
async def test_get_session_not_found(test_client, mock_agent_service) -> None:
    """GET /api/v2/agents/sessions/{session_id} should return 404 if not found."""
    mock_agent_service.get_session = AsyncMock(return_value=None)

    response = await test_client.get("/api/v2/agents/sessions/nonexistent")

    assert response.status_code == 404


@pytest.mark.asyncio()
async def test_get_session_forbidden_for_other_user(test_client, mock_other_user, mock_agent_service) -> None:
    """GET /api/v2/agents/sessions/{session_id} should return 403 for other user's session."""
    mock_session = {
        "id": "session-uuid-1",
        "external_id": "abc12345",
        "title": "Other User Session",
        "agent_plugin_id": "test-agent",
        "message_count": 5,
        "total_input_tokens": 100,
        "total_output_tokens": 200,
        "total_cost_usd": 0.01,
        "status": "active",
        "created_at": "2025-01-06T10:00:00Z",
        "last_activity_at": "2025-01-06T10:05:00Z",
        "user_id": mock_other_user["id"],  # Different user
    }
    mock_agent_service.get_session = AsyncMock(return_value=mock_session)

    response = await test_client.get("/api/v2/agents/sessions/abc12345")

    assert response.status_code == 403


@pytest.mark.asyncio()
async def test_get_session_messages_returns_messages(test_client, mock_user, mock_agent_service) -> None:
    """GET /api/v2/agents/sessions/{session_id}/messages should return messages."""
    mock_session = {
        "id": "session-uuid-1",
        "external_id": "abc12345",
        "user_id": mock_user["id"],
    }
    mock_messages = [
        {
            "id": "msg-001",
            "sequence": 0,
            "role": "user",
            "type": "text",
            "content": "Hello",
            "created_at": "2025-01-06T10:00:00Z",
        },
        {
            "id": "msg-002",
            "sequence": 1,
            "role": "assistant",
            "type": "text",
            "content": "Hello! How can I help?",
            "created_at": "2025-01-06T10:00:01Z",
        },
    ]
    mock_agent_service.get_session = AsyncMock(return_value=mock_session)
    mock_agent_service.get_messages = AsyncMock(return_value=mock_messages)

    response = await test_client.get("/api/v2/agents/sessions/abc12345/messages")

    assert response.status_code == 200
    body = response.json()
    assert len(body["messages"]) == 2
    assert body["messages"][0]["content"] == "Hello"


@pytest.mark.asyncio()
async def test_update_session_title(test_client, mock_user, mock_agent_service) -> None:
    """PATCH /api/v2/agents/sessions/{session_id} should update session title."""
    mock_session = {
        "id": "session-uuid-1",
        "external_id": "abc12345",
        "title": "Original Title",
        "agent_plugin_id": "test-agent",
        "message_count": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost_usd": 0.0,
        "status": "active",
        "created_at": "2025-01-06T10:00:00Z",
        "last_activity_at": "2025-01-06T10:00:00Z",
        "user_id": mock_user["id"],
    }
    updated_session = {**mock_session, "title": "New Title"}
    mock_agent_service.get_session = AsyncMock(return_value=mock_session)
    mock_agent_service.update_session_title = AsyncMock(return_value=updated_session)

    response = await test_client.patch(
        "/api/v2/agents/sessions/abc12345",
        json={"title": "New Title"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "New Title"


@pytest.mark.asyncio()
async def test_fork_session_creates_new_session(test_client, mock_user, mock_agent_service) -> None:
    """POST /api/v2/agents/sessions/{session_id}/fork should create a forked session."""
    mock_session = {
        "id": "session-uuid-1",
        "external_id": "abc12345",
        "user_id": mock_user["id"],
    }
    mock_agent_service.get_session = AsyncMock(return_value=mock_session)
    mock_agent_service.fork_session = AsyncMock(return_value="forked123")

    response = await test_client.post("/api/v2/agents/sessions/abc12345/fork")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "forked123"


@pytest.mark.asyncio()
async def test_interrupt_session(test_client, mock_user, mock_agent_service) -> None:
    """POST /api/v2/agents/sessions/{session_id}/interrupt should interrupt execution."""
    mock_session = {
        "id": "session-uuid-1",
        "external_id": "abc12345",
        "user_id": mock_user["id"],
    }
    mock_agent_service.get_session = AsyncMock(return_value=mock_session)
    mock_agent_service.interrupt = AsyncMock()

    response = await test_client.post("/api/v2/agents/sessions/abc12345/interrupt")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "interrupted"


@pytest.mark.asyncio()
async def test_delete_session(test_client, mock_user, mock_agent_service) -> None:
    """DELETE /api/v2/agents/sessions/{session_id} should soft-delete session."""
    mock_session = {
        "id": "session-uuid-1",
        "external_id": "abc12345",
        "user_id": mock_user["id"],
    }
    mock_agent_service.get_session = AsyncMock(return_value=mock_session)
    mock_agent_service.delete_session = AsyncMock()

    response = await test_client.delete("/api/v2/agents/sessions/abc12345")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "deleted"


# =============================================================================
# Tool Endpoints Tests
# =============================================================================


@pytest.mark.asyncio()
async def test_list_tools_returns_available_tools(test_client) -> None:
    """GET /api/v2/agents/tools should return available tools."""
    mock_tool = MagicMock()
    mock_tool.name = "semantic_search"
    mock_tool.description = "Search documents semantically"
    mock_tool.definition.parameters = []
    mock_tool.definition.category = "search"
    mock_tool.definition.requires_context = True
    mock_tool.definition.is_destructive = False

    mock_record = MagicMock()
    mock_record.tool = mock_tool
    mock_record.enabled = True

    with patch("webui.api.v2.agents.get_tool_registry") as mock_get_registry:
        mock_registry = MagicMock()
        mock_registry.list_all = MagicMock(return_value=[mock_record])
        mock_get_registry.return_value = mock_registry

        response = await test_client.get("/api/v2/agents/tools")

    assert response.status_code == 200
    body = response.json()
    assert len(body["tools"]) == 1
    assert body["tools"][0]["name"] == "semantic_search"


@pytest.mark.asyncio()
async def test_get_tool_returns_details(test_client) -> None:
    """GET /api/v2/agents/tools/{tool_name} should return tool details."""
    mock_tool = MagicMock()
    mock_tool.name = "semantic_search"
    mock_tool.description = "Search documents semantically"
    mock_tool.definition.parameters = []
    mock_tool.definition.category = "search"
    mock_tool.definition.requires_context = True
    mock_tool.definition.is_destructive = False

    with patch("webui.api.v2.agents.get_tool_registry") as mock_get_registry:
        mock_registry = MagicMock()
        mock_registry.get = MagicMock(return_value=mock_tool)
        mock_get_registry.return_value = mock_registry

        response = await test_client.get("/api/v2/agents/tools/semantic_search")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "semantic_search"
    assert body["category"] == "search"


@pytest.mark.asyncio()
async def test_get_tool_not_found(test_client) -> None:
    """GET /api/v2/agents/tools/{tool_name} should return 404 for unknown tool."""
    with patch("webui.api.v2.agents.get_tool_registry") as mock_get_registry:
        mock_registry = MagicMock()
        mock_registry.get = MagicMock(return_value=None)
        mock_get_registry.return_value = mock_registry

        response = await test_client.get("/api/v2/agents/tools/nonexistent")

    assert response.status_code == 404


# =============================================================================
# Authentication Tests
# =============================================================================


@pytest.mark.asyncio()
async def test_list_agents_requires_auth() -> None:
    """GET /api/v2/agents should require authentication."""
    from fastapi import HTTPException

    app = create_test_app()

    # Override auth to raise 401 (simulating missing/invalid token)
    async def raise_unauthorized() -> None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_current_user] = raise_unauthorized

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v2/agents")

    assert response.status_code == 401


@pytest.mark.asyncio()
async def test_list_sessions_requires_auth() -> None:
    """GET /api/v2/agents/sessions should require authentication."""
    from fastapi import HTTPException

    app = create_test_app()

    # Override auth to raise 401 (simulating missing/invalid token)
    async def raise_unauthorized() -> None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_current_user] = raise_unauthorized

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v2/agents/sessions")

    assert response.status_code == 401
