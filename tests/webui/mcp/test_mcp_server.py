from __future__ import annotations

import json
import types
from unittest.mock import AsyncMock

import pytest

from webui.mcp.server import SemantikMCPServer


@pytest.mark.asyncio()
async def test_list_tools_exposes_profile_search_tools() -> None:
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        return_value=[
            {
                "name": "coding",
                "description": "Coding docs",
                "enabled": True,
                "search_type": "semantic",
                "result_count": 5,
                "use_reranker": True,
                "collections": [{"id": "c1", "name": "Collection 1"}],
            },
            {
                "name": "personal",
                "description": "Personal notes",
                "enabled": True,
                "search_type": "semantic",
                "result_count": 10,
                "use_reranker": True,
                "collections": [{"id": "c2", "name": "Collection 2"}],
            },
        ]
    )

    tools = await server.list_tools()
    names = {t.name for t in tools}

    assert "search_coding" in names
    assert "search_personal" in names
    assert "search" not in names

    assert "get_document" in names
    assert "get_document_content" in names
    assert "get_chunk" in names
    assert "list_documents" in names

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_profile_filter_limits_exposed_search_tools() -> None:
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token", profile_filter=["coding"])
    server.api_client.get_profiles = AsyncMock(
        return_value=[
            {
                "name": "coding",
                "description": "Coding docs",
                "enabled": True,
                "search_type": "semantic",
                "result_count": 5,
                "use_reranker": True,
                "collections": [{"id": "c1", "name": "Collection 1"}],
            },
            {
                "name": "personal",
                "description": "Personal notes",
                "enabled": True,
                "search_type": "semantic",
                "result_count": 10,
                "use_reranker": True,
                "collections": [{"id": "c2", "name": "Collection 2"}],
            },
        ]
    )

    tools = await server.list_tools()
    names = {t.name for t in tools}

    assert "search_coding" in names
    assert "search_personal" not in names

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_search_tool_formats_results_as_text() -> None:
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        return_value=[
            {
                "name": "coding",
                "description": "Coding docs",
                "enabled": True,
                "search_type": "semantic",
                "result_count": 5,
                "use_reranker": True,
                "collections": [{"id": "c1", "name": "Collection 1"}],
            }
        ]
    )
    server.api_client.search = AsyncMock(
        return_value={
            "query": "hello",
            "total_results": 1,
            "partial_failure": False,
            "results": [
                {
                    "collection_id": "c1",
                    "collection_name": "Collection 1",
                    "document_id": "d1",
                    "chunk_id": "d1_0001",
                    "score": 0.9,
                    "file_path": "/docs/readme.md",
                    "file_name": "readme.md",
                    "text": "x" * 1000,
                    "metadata": {"page": 1},
                }
            ],
        }
    )

    result = await server.call_tool("search_coding", {"query": "hello"})
    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"

    payload = json.loads(result["content"][0]["text"])
    assert payload["query"] == "hello"
    assert payload["total_results"] == 1
    assert payload["results"][0]["chunk_id"] == "d1_0001"
    assert payload["results"][0]["text"].endswith("…")

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_utility_tools_are_scope_checked() -> None:
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        return_value=[
            {
                "name": "coding",
                "description": "Coding docs",
                "enabled": True,
                "search_type": "semantic",
                "result_count": 5,
                "use_reranker": True,
                "collections": [{"id": "c1", "name": "Collection 1"}],
            }
        ]
    )
    server.api_client.get_document = AsyncMock(return_value={"id": "d1"})

    denied = await server.call_tool("get_document", {"collection_id": "c2", "document_id": "d1"})
    assert denied["isError"] is True
    assert "Collection is not accessible" in denied["content"][0]["text"]

    allowed = await server.call_tool("get_document", {"collection_id": "c1", "document_id": "d1"})
    assert allowed["isError"] is False

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_profiles_cache_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    # First call: cache miss; second call: cache hit; third call: after TTL, miss again.
    values = [0.0, 0.0, 1.0, 20.0, 20.0]

    def fake_monotonic() -> float:
        return values.pop(0) if values else 20.0

    monkeypatch.setattr("webui.mcp.server.time.monotonic", fake_monotonic)

    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        return_value=[
            {
                "name": "coding",
                "description": "Coding docs",
                "enabled": True,
                "search_type": "semantic",
                "result_count": 5,
                "use_reranker": True,
                "collections": [{"id": "c1", "name": "Collection 1"}],
            }
        ]
    )

    await server.list_tools()
    await server.list_tools()
    await server.list_tools()

    assert server.api_client.get_profiles.call_count == 2

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_call_tool_unknown_tool_returns_error() -> None:
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(return_value=[])

    result = await server.call_tool("nope", {})
    assert result["isError"] is True
    assert "Unknown tool" in result["content"][0]["text"]

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_call_tool_unknown_profile_returns_error() -> None:
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        return_value=[
            {
                "name": "coding",
                "description": "Coding docs",
                "enabled": True,
                "search_type": "semantic",
                "result_count": 5,
                "use_reranker": True,
                "collections": [{"id": "c1", "name": "Collection 1"}],
            }
        ]
    )

    result = await server.call_tool("search_missing", {"query": "x"})
    assert result["isError"] is True
    assert "Profile not found" in result["content"][0]["text"]

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_search_missing_query_returns_error() -> None:
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        return_value=[
            {
                "name": "coding",
                "description": "Coding docs",
                "enabled": True,
                "search_type": "semantic",
                "result_count": 5,
                "use_reranker": True,
                "collections": [{"id": "c1", "name": "Collection 1"}],
            }
        ]
    )

    result = await server.call_tool("search_coding", {})
    assert result["isError"] is True
    assert "Missing required argument: query" in result["content"][0]["text"]

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_search_profile_with_no_collections_returns_error() -> None:
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        return_value=[
            {
                "name": "coding",
                "description": "Coding docs",
                "enabled": True,
                "search_type": "semantic",
                "result_count": 5,
                "use_reranker": True,
                "collections": [],
            }
        ]
    )

    result = await server.call_tool("search_coding", {"query": "x"})
    assert result["isError"] is True
    assert "no collections" in result["content"][0]["text"].lower()

    await server.api_client.close()


def test_format_search_results_handles_weird_inputs() -> None:
    payload = SemantikMCPServer._format_search_results(  # noqa: SLF001 - unit test
        {
            "query": "q",
            "results": [None, {"text": 123, "metadata": None}, {"text": " okay "}],
            "partial_failure": True,
        },
        max_snippet_chars=2,
    )
    assert payload["partial_failure"] is True
    assert payload["results"][0]["text"] == ""
    assert payload["results"][1]["text"].endswith("…")


@pytest.mark.asyncio()
async def test_get_document_content_binary_detection() -> None:
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        return_value=[
            {"name": "coding", "description": "d", "enabled": True, "collections": [{"id": "c1", "name": "c1"}]}
        ]
    )
    server.api_client.get_document_content = AsyncMock(return_value=(b"\x00pdf", "application/pdf"))

    result = await server.call_tool("get_document_content", {"collection_id": "c1", "document_id": "d1"})
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert "binary" in payload["error"].lower()

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_get_document_content_truncates_large_text() -> None:
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        return_value=[
            {"name": "coding", "description": "d", "enabled": True, "collections": [{"id": "c1", "name": "c1"}]}
        ]
    )
    server.api_client.get_document_content = AsyncMock(return_value=(b"a" * 200_001, "text/plain"))

    result = await server.call_tool("get_document_content", {"collection_id": "c1", "document_id": "d1"})
    payload = json.loads(result["content"][0]["text"])
    assert payload["truncated"] is True
    assert len(payload["text"]) == 200_000

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_run_closes_api_client(monkeypatch: pytest.MonkeyPatch) -> None:
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.server.run = AsyncMock()
    server.api_client.close = AsyncMock()

    class FakeStdio:
        async def __aenter__(self):
            return (types.SimpleNamespace(), types.SimpleNamespace())

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("webui.mcp.server.stdio_server", lambda: FakeStdio())

    await server.run()
    server.api_client.close.assert_awaited()


# --------------------------------------------------------------------------
# Auth Middleware Tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_auth_middleware_health_endpoint_bypasses_auth() -> None:
    """Test that /health endpoint bypasses auth middleware."""
    from unittest.mock import MagicMock
    from webui.mcp.server import MCPAuthMiddleware

    mock_api_client = MagicMock()
    mock_app = MagicMock()
    middleware = MCPAuthMiddleware(mock_app, mock_api_client)

    mock_request = MagicMock()
    mock_request.url.path = "/health"
    mock_response = MagicMock()

    async def mock_call_next(request):
        return mock_response

    result = await middleware.dispatch(mock_request, mock_call_next)
    assert result == mock_response
    # validate_api_key should not be called for health endpoint
    mock_api_client.validate_api_key.assert_not_called()


@pytest.mark.asyncio()
async def test_auth_middleware_missing_auth_header_returns_401() -> None:
    """Test that missing Authorization header returns 401."""
    from unittest.mock import MagicMock
    from webui.mcp.server import MCPAuthMiddleware
    from starlette.responses import JSONResponse

    mock_api_client = MagicMock()
    mock_app = MagicMock()
    middleware = MCPAuthMiddleware(mock_app, mock_api_client)

    mock_request = MagicMock()
    mock_request.url.path = "/mcp"
    mock_request.headers = {}

    async def mock_call_next(request):
        return MagicMock()

    result = await middleware.dispatch(mock_request, mock_call_next)
    assert isinstance(result, JSONResponse)
    assert result.status_code == 401


@pytest.mark.asyncio()
async def test_auth_middleware_invalid_bearer_format_returns_401() -> None:
    """Test that invalid bearer format returns 401."""
    from unittest.mock import MagicMock
    from webui.mcp.server import MCPAuthMiddleware
    from starlette.responses import JSONResponse

    mock_api_client = MagicMock()
    mock_app = MagicMock()
    middleware = MCPAuthMiddleware(mock_app, mock_api_client)

    mock_request = MagicMock()
    mock_request.url.path = "/mcp"
    mock_request.headers = {"authorization": "Basic abc123"}

    async def mock_call_next(request):
        return MagicMock()

    result = await middleware.dispatch(mock_request, mock_call_next)
    assert isinstance(result, JSONResponse)
    assert result.status_code == 401


@pytest.mark.asyncio()
async def test_auth_middleware_empty_api_key_returns_401() -> None:
    """Test that empty API key returns 401."""
    from unittest.mock import MagicMock
    from webui.mcp.server import MCPAuthMiddleware
    from starlette.responses import JSONResponse

    mock_api_client = MagicMock()
    mock_app = MagicMock()
    middleware = MCPAuthMiddleware(mock_app, mock_api_client)

    mock_request = MagicMock()
    mock_request.url.path = "/mcp"
    mock_request.headers = {"authorization": "Bearer "}

    async def mock_call_next(request):
        return MagicMock()

    result = await middleware.dispatch(mock_request, mock_call_next)
    assert isinstance(result, JSONResponse)
    assert result.status_code == 401


@pytest.mark.asyncio()
async def test_auth_middleware_invalid_api_key_returns_401() -> None:
    """Test that invalid API key returns 401."""
    from unittest.mock import MagicMock
    from webui.mcp.server import MCPAuthMiddleware
    from starlette.responses import JSONResponse

    mock_api_client = MagicMock()
    mock_api_client.validate_api_key = AsyncMock(return_value={"valid": False, "error": "Invalid key"})
    mock_app = MagicMock()
    middleware = MCPAuthMiddleware(mock_app, mock_api_client)

    mock_request = MagicMock()
    mock_request.url.path = "/mcp"
    mock_request.headers = {"authorization": "Bearer invalid-key"}

    async def mock_call_next(request):
        return MagicMock()

    result = await middleware.dispatch(mock_request, mock_call_next)
    assert isinstance(result, JSONResponse)
    assert result.status_code == 401


@pytest.mark.asyncio()
async def test_auth_middleware_valid_api_key_sets_request_state() -> None:
    """Test that valid API key sets request state and context."""
    from unittest.mock import MagicMock
    from webui.mcp.server import MCPAuthMiddleware, _mcp_auth_context

    mock_api_client = MagicMock()
    mock_api_client.validate_api_key = AsyncMock(
        return_value={"valid": True, "user_id": 42, "username": "testuser"}
    )
    mock_app = MagicMock()
    middleware = MCPAuthMiddleware(mock_app, mock_api_client)

    mock_request = MagicMock()
    mock_request.url.path = "/mcp"
    mock_request.headers = {"authorization": "Bearer valid-key"}
    mock_request.state = types.SimpleNamespace()

    mock_response = MagicMock()
    captured_context = None

    async def mock_call_next(request):
        nonlocal captured_context
        # Capture the context during request processing
        captured_context = _mcp_auth_context.get()
        return mock_response

    result = await middleware.dispatch(mock_request, mock_call_next)
    assert result == mock_response
    assert mock_request.state.mcp_user_id == 42
    assert mock_request.state.mcp_username == "testuser"
    assert captured_context is not None
    assert captured_context.user_id == 42
    assert captured_context.username == "testuser"
    # Context should be cleared after request
    assert _mcp_auth_context.get() is None


@pytest.mark.asyncio()
async def test_auth_middleware_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that cache hit returns cached validation result."""
    from unittest.mock import MagicMock
    from webui.mcp.server import MCPAuthMiddleware

    time_values = [0.0, 30.0]  # 30 seconds later still within 60s TTL

    def fake_monotonic() -> float:
        return time_values.pop(0) if time_values else 30.0

    monkeypatch.setattr("webui.mcp.server.time.monotonic", fake_monotonic)

    mock_api_client = MagicMock()
    mock_api_client.validate_api_key = AsyncMock(
        return_value={"valid": True, "user_id": 42, "username": "testuser"}
    )
    mock_app = MagicMock()
    middleware = MCPAuthMiddleware(mock_app, mock_api_client)

    mock_request = MagicMock()
    mock_request.url.path = "/mcp"
    mock_request.headers = {"authorization": "Bearer test-key"}
    mock_request.state = types.SimpleNamespace()

    mock_response = MagicMock()

    async def mock_call_next(request):
        return mock_response

    # First request - cache miss
    await middleware.dispatch(mock_request, mock_call_next)
    # Second request - should be cache hit
    mock_request.state = types.SimpleNamespace()
    await middleware.dispatch(mock_request, mock_call_next)

    # Only one API call should have been made
    assert mock_api_client.validate_api_key.call_count == 1


@pytest.mark.asyncio()
async def test_auth_middleware_cache_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that cache expires after TTL (60s)."""
    from unittest.mock import MagicMock
    from webui.mcp.server import MCPAuthMiddleware

    time_values = [0.0, 61.0]  # 61 seconds later, cache expired

    def fake_monotonic() -> float:
        return time_values.pop(0) if time_values else 61.0

    monkeypatch.setattr("webui.mcp.server.time.monotonic", fake_monotonic)

    mock_api_client = MagicMock()
    mock_api_client.validate_api_key = AsyncMock(
        return_value={"valid": True, "user_id": 42, "username": "testuser"}
    )
    mock_app = MagicMock()
    middleware = MCPAuthMiddleware(mock_app, mock_api_client)

    mock_request = MagicMock()
    mock_request.url.path = "/mcp"
    mock_request.headers = {"authorization": "Bearer test-key"}
    mock_request.state = types.SimpleNamespace()

    mock_response = MagicMock()

    async def mock_call_next(request):
        return mock_response

    # First request - cache miss
    await middleware.dispatch(mock_request, mock_call_next)
    # Second request - cache expired, should re-validate
    mock_request.state = types.SimpleNamespace()
    await middleware.dispatch(mock_request, mock_call_next)

    # Two API calls should have been made
    assert mock_api_client.validate_api_key.call_count == 2


@pytest.mark.asyncio()
async def test_auth_middleware_cache_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that cache cleanup removes expired entries when > 100 entries."""
    from unittest.mock import MagicMock
    from webui.mcp.server import MCPAuthMiddleware

    monkeypatch.setattr("webui.mcp.server.time.monotonic", lambda: 100.0)

    mock_api_client = MagicMock()
    mock_api_client.validate_api_key = AsyncMock(
        return_value={"valid": True, "user_id": 42, "username": "testuser"}
    )
    mock_app = MagicMock()
    middleware = MCPAuthMiddleware(mock_app, mock_api_client)

    # Pre-populate cache with > 100 expired entries
    for i in range(105):
        key_hash = f"hash_{i}"
        middleware._validation_cache[key_hash] = ({"valid": True}, 50.0)  # noqa: SLF001 - unit test

    mock_request = MagicMock()
    mock_request.url.path = "/mcp"
    mock_request.headers = {"authorization": "Bearer new-key"}
    mock_request.state = types.SimpleNamespace()

    async def mock_call_next(request):
        return MagicMock()

    await middleware.dispatch(mock_request, mock_call_next)

    # All expired entries should be cleaned up
    assert len(middleware._validation_cache) <= 2  # noqa: SLF001 - unit test


@pytest.mark.asyncio()
async def test_auth_middleware_api_error_returns_service_unavailable() -> None:
    """Test that API error returns graceful error response."""
    from unittest.mock import MagicMock
    from webui.mcp.server import MCPAuthMiddleware
    from webui.mcp.client import SemantikAPIError
    from starlette.responses import JSONResponse

    mock_api_client = MagicMock()
    mock_api_client.validate_api_key = AsyncMock(side_effect=SemantikAPIError("Connection failed"))
    mock_app = MagicMock()
    middleware = MCPAuthMiddleware(mock_app, mock_api_client)

    mock_request = MagicMock()
    mock_request.url.path = "/mcp"
    mock_request.headers = {"authorization": "Bearer test-key"}

    async def mock_call_next(request):
        return MagicMock()

    result = await middleware.dispatch(mock_request, mock_call_next)
    assert isinstance(result, JSONResponse)
    assert result.status_code == 401


# --------------------------------------------------------------------------
# Service Mode Tests
# --------------------------------------------------------------------------


def test_server_requires_auth_credentials() -> None:
    """Test that server raises ValueError if neither auth_token nor internal_api_key is provided."""
    with pytest.raises(ValueError, match="Either auth_token or internal_api_key must be provided"):
        SemantikMCPServer(webui_url="http://example.invalid")


def test_service_mode_is_set_correctly() -> None:
    """Test that service_mode is correctly determined based on auth method."""
    # User mode
    user_server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    assert user_server.service_mode is False

    # Service mode
    service_server = SemantikMCPServer(webui_url="http://example.invalid", internal_api_key="internal-key")
    assert service_server.service_mode is True


@pytest.mark.asyncio()
async def test_service_mode_filters_profiles_by_user_id() -> None:
    """Test that service mode filters profiles by authenticated user_id."""
    from webui.mcp.server import _mcp_auth_context, MCPAuthContext

    server = SemantikMCPServer(webui_url="http://example.invalid", internal_api_key="internal-key")
    server.api_client.get_all_profiles = AsyncMock(
        return_value=[
            {"name": "user1-profile", "owner_id": 1, "enabled": True, "collections": [{"id": "c1"}]},
            {"name": "user2-profile", "owner_id": 2, "enabled": True, "collections": [{"id": "c2"}]},
            {"name": "user1-other", "owner_id": 1, "enabled": True, "collections": [{"id": "c3"}]},
        ]
    )

    # Set auth context for user 1
    _mcp_auth_context.set(MCPAuthContext(user_id=1, username="user1"))

    tools = await server.list_tools()
    tool_names = {t.name for t in tools}

    # Should only see user 1's profiles
    assert "search_user1-profile" in tool_names
    assert "search_user1-other" in tool_names
    assert "search_user2-profile" not in tool_names

    # Clean up
    _mcp_auth_context.set(None)
    await server.api_client.close()


@pytest.mark.asyncio()
async def test_service_mode_no_auth_context_returns_empty_profiles() -> None:
    """Test that service mode with no auth context returns empty profiles list."""
    server = SemantikMCPServer(webui_url="http://example.invalid", internal_api_key="internal-key")
    server.api_client.get_all_profiles = AsyncMock(
        return_value=[
            {"name": "profile1", "owner_id": 1, "enabled": True, "collections": [{"id": "c1"}]},
        ]
    )

    # Don't set auth context
    tools = await server.list_tools()
    search_tools = [t for t in tools if t.name.startswith("search_")]

    # Should have no search tools (only utility tools)
    assert len(search_tools) == 0

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_all_profiles_cache_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that all-profiles cache has 10s TTL in service mode."""
    from webui.mcp.server import _mcp_auth_context, MCPAuthContext

    values = [0.0, 0.0, 5.0, 15.0, 15.0]  # First two calls within 10s, third after

    def fake_monotonic() -> float:
        return values.pop(0) if values else 15.0

    monkeypatch.setattr("webui.mcp.server.time.monotonic", fake_monotonic)

    server = SemantikMCPServer(webui_url="http://example.invalid", internal_api_key="internal-key")
    server.api_client.get_all_profiles = AsyncMock(
        return_value=[{"name": "profile", "owner_id": 1, "enabled": True, "collections": [{"id": "c1"}]}]
    )

    _mcp_auth_context.set(MCPAuthContext(user_id=1, username="user1"))

    await server.list_tools()  # Cache miss
    await server.list_tools()  # Cache hit
    await server.list_tools()  # Cache expired, refresh

    assert server.api_client.get_all_profiles.call_count == 2

    _mcp_auth_context.set(None)
    await server.api_client.close()


# --------------------------------------------------------------------------
# Diagnostics Tool Tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_diagnostics_tool_returns_server_info() -> None:
    """Test that diagnostics tool returns server status information."""
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        return_value=[
            {
                "name": "coding",
                "description": "Coding docs",
                "enabled": True,
                "search_type": "semantic",
                "use_reranker": True,
                "collections": [{"id": "c1"}, {"id": "c2"}],
            }
        ]
    )

    result = await server.call_tool("diagnostics", {})
    assert result["isError"] is False

    payload = json.loads(result["content"][0]["text"])
    assert payload["server_name"] == "semantik"
    assert "connection" in payload
    assert payload["connection"]["connected"] is True
    assert payload["connection"]["authenticated"] is True
    assert len(payload["profiles"]) == 1
    assert payload["profiles"][0]["name"] == "coding"
    assert payload["profiles"][0]["collection_count"] == 2
    assert payload["available_tools"] == 6  # 1 search + 5 utility

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_diagnostics_tool_handles_connection_error() -> None:
    """Test that diagnostics tool handles connection errors gracefully."""
    from webui.mcp.client import SemantikAPIError

    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        side_effect=SemantikAPIError("connection refused")
    )

    result = await server.call_tool("diagnostics", {})
    assert result["isError"] is False

    payload = json.loads(result["content"][0]["text"])
    assert payload["connection"]["connected"] is False
    assert "error" in payload["connection"]

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_diagnostics_tool_handles_auth_error() -> None:
    """Test that diagnostics tool handles auth errors gracefully."""
    from webui.mcp.client import SemantikAPIError

    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        side_effect=SemantikAPIError("GET /api/v2/mcp/profiles failed (401): unauthorized")
    )

    result = await server.call_tool("diagnostics", {})
    assert result["isError"] is False

    payload = json.loads(result["content"][0]["text"])
    assert payload["connection"]["authenticated"] is False
    assert "error" in payload["connection"]

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_diagnostics_tool_with_profile_filter() -> None:
    """Test that diagnostics tool respects profile filter."""
    server = SemantikMCPServer(
        webui_url="http://example.invalid",
        auth_token="token",
        profile_filter=["coding"],
    )
    server.api_client.get_profiles = AsyncMock(
        return_value=[
            {"name": "coding", "enabled": True, "collections": []},
            {"name": "personal", "enabled": True, "collections": []},
        ]
    )

    result = await server.call_tool("diagnostics", {})
    payload = json.loads(result["content"][0]["text"])

    assert payload["profile_filter"] == ["coding"]
    assert len(payload["profiles"]) == 1
    assert payload["profiles"][0]["name"] == "coding"

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_diagnostics_shows_cache_status() -> None:
    """Test that diagnostics tool shows cache status."""
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(return_value=[])

    # First call - no cache
    result1 = await server.call_tool("diagnostics", {})
    payload1 = json.loads(result1["content"][0]["text"])
    assert payload1["cache"]["cached"] is False

    # Second call - should show cache info (diagnostics populates it)
    result2 = await server.call_tool("diagnostics", {})
    payload2 = json.loads(result2["content"][0]["text"])
    assert payload2["cache"]["cached"] is True

    await server.api_client.close()


# --------------------------------------------------------------------------
# Additional Edge Case Tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_get_chunk_tool_scope_checked() -> None:
    """Test that get_chunk tool is scope checked."""
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        return_value=[{"name": "p", "enabled": True, "collections": [{"id": "c1"}]}]
    )
    server.api_client.get_chunk = AsyncMock(return_value={"chunk_id": "chunk1", "text": "content"})

    # Denied - wrong collection
    denied = await server.call_tool("get_chunk", {"collection_id": "c2", "chunk_id": "chunk1"})
    assert denied["isError"] is True
    assert "not accessible" in denied["content"][0]["text"]

    # Allowed - correct collection
    allowed = await server.call_tool("get_chunk", {"collection_id": "c1", "chunk_id": "chunk1"})
    assert allowed["isError"] is False

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_list_documents_tool_scope_checked() -> None:
    """Test that list_documents tool is scope checked."""
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        return_value=[{"name": "p", "enabled": True, "collections": [{"id": "c1"}]}]
    )
    server.api_client.list_documents = AsyncMock(return_value={"documents": [], "total": 0})

    # Denied - wrong collection
    denied = await server.call_tool("list_documents", {"collection_id": "c2"})
    assert denied["isError"] is True

    # Allowed - correct collection
    allowed = await server.call_tool("list_documents", {"collection_id": "c1"})
    assert allowed["isError"] is False

    await server.api_client.close()


@pytest.mark.asyncio()
async def test_list_documents_with_pagination_params() -> None:
    """Test that list_documents passes pagination parameters correctly."""
    server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
    server.api_client.get_profiles = AsyncMock(
        return_value=[{"name": "p", "enabled": True, "collections": [{"id": "c1"}]}]
    )
    server.api_client.list_documents = AsyncMock(return_value={"documents": [], "total": 0})

    await server.call_tool(
        "list_documents",
        {"collection_id": "c1", "page": 2, "per_page": 25, "status": "completed"},
    )

    server.api_client.list_documents.assert_awaited_once_with(
        collection_id="c1",
        page=2,
        per_page=25,
        status="completed",
    )

    await server.api_client.close()
