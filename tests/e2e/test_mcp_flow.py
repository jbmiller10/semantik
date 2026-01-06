"""End-to-end tests for MCP (Model Context Protocol) integration.

These tests verify the complete MCP server flow using mocked API responses
for faster iteration without Docker dependencies.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from webui.mcp.server import SemantikMCPServer


def _make_profile(
    name: str,
    description: str = "Test profile",
    collections: list[dict[str, str]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a mock profile with sensible defaults.

    Pass collections=[] explicitly to create a profile with no collections.
    Pass collections=None to use the default collection.
    """
    if collections is None:
        # Default: create a collection for this profile
        default_collections = [{"id": f"col-{name}", "name": f"Collection for {name}"}]
    else:
        # Use exactly what was passed (including empty list)
        default_collections = collections

    return {
        "name": name,
        "description": description,
        "enabled": True,
        "search_type": kwargs.get("search_type", "semantic"),
        "result_count": kwargs.get("result_count", 10),
        "use_reranker": kwargs.get("use_reranker", True),
        "score_threshold": kwargs.get("score_threshold"),
        "hybrid_alpha": kwargs.get("hybrid_alpha"),
        "collections": default_collections,
    }


def _make_search_results(
    query: str,
    results: list[dict[str, Any]] | None = None,
    total_results: int | None = None,
) -> dict[str, Any]:
    """Create mock search results."""
    if results is None:
        results = [
            {
                "collection_id": "col-1",
                "collection_name": "Test Collection",
                "document_id": "doc-1",
                "chunk_id": "doc-1_0001",
                "score": 0.95,
                "file_path": "/docs/readme.md",
                "file_name": "readme.md",
                "text": "This is a test document with relevant content.",
                "metadata": {"page": 1},
            }
        ]
    return {
        "query": query,
        "total_results": total_results if total_results is not None else len(results),
        "partial_failure": False,
        "results": results,
    }


@pytest.mark.e2e()
class TestMCPSearchFlow:
    """End-to-end tests for MCP search workflow."""

    @pytest.mark.asyncio()
    async def test_full_search_flow(self) -> None:
        """Test complete search flow: list tools -> search -> get results."""
        # Setup server with mocked API client
        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                _make_profile(
                    "coding",
                    "Search coding documentation and API references",
                    collections=[
                        {"id": "col-docs", "name": "Documentation"},
                        {"id": "col-api", "name": "API Reference"},
                    ],
                )
            ]
        )
        server.api_client.search = AsyncMock(
            return_value=_make_search_results(
                "how to use async functions",
                results=[
                    {
                        "collection_id": "col-docs",
                        "collection_name": "Documentation",
                        "document_id": "doc-async",
                        "chunk_id": "doc-async_0001",
                        "score": 0.92,
                        "file_path": "/docs/async-guide.md",
                        "file_name": "async-guide.md",
                        "text": "Async functions allow you to write non-blocking code...",
                        "metadata": {},
                    },
                    {
                        "collection_id": "col-api",
                        "collection_name": "API Reference",
                        "document_id": "doc-asyncio",
                        "chunk_id": "doc-asyncio_0001",
                        "score": 0.88,
                        "file_path": "/api/asyncio.md",
                        "file_name": "asyncio.md",
                        "text": "The asyncio module provides infrastructure for async programming.",
                        "metadata": {},
                    },
                ],
            )
        )

        try:
            # Step 1: List available tools
            tools = await server.list_tools()
            tool_names = {t.name for t in tools}

            assert "search_coding" in tool_names, "Profile search tool should be exposed"
            assert "get_document" in tool_names, "Utility tool should be available"
            assert "get_document_content" in tool_names
            assert "get_chunk" in tool_names
            assert "list_documents" in tool_names

            # Step 2: Execute search
            result = await server.call_tool("search_coding", {"query": "how to use async functions"})

            assert result["isError"] is False, f"Search should succeed: {result}"
            payload = json.loads(result["content"][0]["text"])

            assert payload["query"] == "how to use async functions"
            assert payload["total_results"] == 2
            assert len(payload["results"]) == 2

            # Verify search was called with correct parameters
            server.api_client.search.assert_awaited_once()
            call_kwargs = server.api_client.search.call_args.kwargs
            assert call_kwargs["query"] == "how to use async functions"
            assert "col-docs" in call_kwargs["collection_uuids"]
            assert "col-api" in call_kwargs["collection_uuids"]

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_search_with_custom_parameters(self) -> None:
        """Test search with overridden parameters."""
        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                _make_profile(
                    "docs",
                    "Documentation search",
                    search_type="semantic",
                    result_count=10,
                    use_reranker=True,
                )
            ]
        )
        server.api_client.search = AsyncMock(return_value=_make_search_results("test query"))

        try:
            # Override profile defaults
            result = await server.call_tool(
                "search_docs",
                {
                    "query": "test query",
                    "k": 5,
                    "search_type": "hybrid",
                    "use_reranker": False,
                    "hybrid_alpha": 0.3,
                },
            )

            assert result["isError"] is False

            # Verify overrides were applied
            call_kwargs = server.api_client.search.call_args.kwargs
            assert call_kwargs["k"] == 5
            assert call_kwargs["search_type"] == "hybrid"
            assert call_kwargs["use_reranker"] is False
            assert call_kwargs["hybrid_alpha"] == 0.3

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_get_document_flow(self) -> None:
        """Test getting document metadata after search."""
        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[_make_profile("docs", collections=[{"id": "col-1", "name": "Docs"}])]
        )
        server.api_client.get_document = AsyncMock(
            return_value={
                "id": "doc-1",
                "file_name": "readme.md",
                "file_path": "/docs/readme.md",
                "file_size": 2048,
                "mime_type": "text/markdown",
                "status": "completed",
                "chunk_count": 5,
                "created_at": "2024-01-01T00:00:00Z",
            }
        )

        try:
            result = await server.call_tool(
                "get_document",
                {"collection_id": "col-1", "document_id": "doc-1"},
            )

            assert result["isError"] is False
            payload = json.loads(result["content"][0]["text"])
            assert payload["id"] == "doc-1"
            assert payload["file_name"] == "readme.md"
            assert payload["chunk_count"] == 5

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_get_document_content_flow(self) -> None:
        """Test getting full document content."""
        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[_make_profile("docs", collections=[{"id": "col-1", "name": "Docs"}])]
        )
        content = b"# README\n\nThis is the full document content."
        server.api_client.get_document_content = AsyncMock(return_value=(content, "text/markdown"))

        try:
            result = await server.call_tool(
                "get_document_content",
                {"collection_id": "col-1", "document_id": "doc-1"},
            )

            assert result["isError"] is False
            payload = json.loads(result["content"][0]["text"])
            assert "# README" in payload["text"]
            assert payload["content_type"] == "text/markdown"
            assert payload["truncated"] is False

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_get_chunk_flow(self) -> None:
        """Test getting a specific chunk."""
        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[_make_profile("docs", collections=[{"id": "col-1", "name": "Docs"}])]
        )
        server.api_client.get_chunk = AsyncMock(
            return_value={
                "id": "chunk-1",
                "document_id": "doc-1",
                "text": "This is the full chunk text without truncation.",
                "metadata": {"page": 1, "section": "introduction"},
            }
        )

        try:
            result = await server.call_tool(
                "get_chunk",
                {"collection_id": "col-1", "chunk_id": "chunk-1"},
            )

            assert result["isError"] is False
            payload = json.loads(result["content"][0]["text"])
            assert payload["id"] == "chunk-1"
            assert "full chunk text" in payload["text"]

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_list_documents_flow(self) -> None:
        """Test listing documents in a collection."""
        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[_make_profile("docs", collections=[{"id": "col-1", "name": "Docs"}])]
        )
        server.api_client.list_documents = AsyncMock(
            return_value={
                "documents": [
                    {"id": "doc-1", "file_name": "readme.md", "status": "completed"},
                    {"id": "doc-2", "file_name": "guide.md", "status": "completed"},
                ],
                "total": 2,
                "page": 1,
                "per_page": 50,
            }
        )

        try:
            result = await server.call_tool(
                "list_documents",
                {"collection_id": "col-1", "page": 1, "per_page": 50},
            )

            assert result["isError"] is False
            payload = json.loads(result["content"][0]["text"])
            assert len(payload["documents"]) == 2
            assert payload["total"] == 2

        finally:
            await server.api_client.close()


@pytest.mark.e2e()
class TestMCPProfileFiltering:
    """Tests for profile filtering via --profile flag."""

    @pytest.mark.asyncio()
    async def test_profile_filter_limits_exposed_tools(self) -> None:
        """Test that --profile flag limits which search tools are exposed."""
        # Server with profile filter
        server = SemantikMCPServer(
            webui_url="http://localhost:8080",
            auth_token="test-token",
            profile_filter=["coding"],
        )
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                _make_profile("coding", "Coding docs"),
                _make_profile("personal", "Personal notes"),
                _make_profile("work", "Work documents"),
            ]
        )

        try:
            tools = await server.list_tools()
            tool_names = {t.name for t in tools}

            # Only coding profile should be exposed
            assert "search_coding" in tool_names
            assert "search_personal" not in tool_names
            assert "search_work" not in tool_names

            # Utility tools should still be available
            assert "get_document" in tool_names
            assert "get_chunk" in tool_names

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_multiple_profile_filters(self) -> None:
        """Test filtering with multiple profiles."""
        server = SemantikMCPServer(
            webui_url="http://localhost:8080",
            auth_token="test-token",
            profile_filter=["coding", "work"],
        )
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                _make_profile("coding", "Coding docs", collections=[{"id": "col-code", "name": "Code"}]),
                _make_profile("personal", "Personal notes", collections=[{"id": "col-personal", "name": "Personal"}]),
                _make_profile("work", "Work documents", collections=[{"id": "col-work", "name": "Work"}]),
            ]
        )

        try:
            tools = await server.list_tools()
            tool_names = {t.name for t in tools}

            assert "search_coding" in tool_names
            assert "search_work" in tool_names
            assert "search_personal" not in tool_names

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_no_filter_exposes_all_enabled_profiles(self) -> None:
        """Test that without filter, all enabled profiles are exposed."""
        server = SemantikMCPServer(
            webui_url="http://localhost:8080",
            auth_token="test-token",
            # No profile_filter
        )
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                _make_profile("coding", "Coding docs"),
                _make_profile("personal", "Personal notes"),
            ]
        )

        try:
            tools = await server.list_tools()
            tool_names = {t.name for t in tools}

            assert "search_coding" in tool_names
            assert "search_personal" in tool_names

        finally:
            await server.api_client.close()


@pytest.mark.e2e()
class TestMCPScopeEnforcement:
    """Tests for collection scope enforcement."""

    @pytest.mark.asyncio()
    async def test_utility_tool_rejects_out_of_scope_collection(self) -> None:
        """Test that utility tools reject collections not in profile scope."""
        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                _make_profile(
                    "docs",
                    "Documentation",
                    collections=[{"id": "col-allowed", "name": "Allowed Collection"}],
                )
            ]
        )

        try:
            # Try to access a collection not in the profile
            result = await server.call_tool(
                "get_document",
                {"collection_id": "col-not-allowed", "document_id": "doc-1"},
            )

            assert result["isError"] is True
            error_text = result["content"][0]["text"]
            assert "not accessible" in error_text.lower()

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_utility_tool_allows_in_scope_collection(self) -> None:
        """Test that utility tools work for collections in profile scope."""
        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                _make_profile(
                    "docs",
                    "Documentation",
                    collections=[{"id": "col-allowed", "name": "Allowed Collection"}],
                )
            ]
        )
        server.api_client.get_document = AsyncMock(
            return_value={"id": "doc-1", "file_name": "test.md"}
        )

        try:
            # Access allowed collection
            result = await server.call_tool(
                "get_document",
                {"collection_id": "col-allowed", "document_id": "doc-1"},
            )

            assert result["isError"] is False
            payload = json.loads(result["content"][0]["text"])
            assert payload["id"] == "doc-1"

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_scope_enforcement_with_multiple_profiles(self) -> None:
        """Test scope enforcement spans all exposed profiles' collections."""
        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                _make_profile("coding", collections=[{"id": "col-code", "name": "Code"}]),
                _make_profile("docs", collections=[{"id": "col-docs", "name": "Docs"}]),
            ]
        )
        server.api_client.get_document = AsyncMock(return_value={"id": "doc-1"})

        try:
            # Both collections should be accessible
            result1 = await server.call_tool(
                "get_document",
                {"collection_id": "col-code", "document_id": "doc-1"},
            )
            assert result1["isError"] is False

            result2 = await server.call_tool(
                "get_document",
                {"collection_id": "col-docs", "document_id": "doc-1"},
            )
            assert result2["isError"] is False

            # Unknown collection should be rejected
            result3 = await server.call_tool(
                "get_document",
                {"collection_id": "col-unknown", "document_id": "doc-1"},
            )
            assert result3["isError"] is True

        finally:
            await server.api_client.close()


@pytest.mark.e2e()
class TestMCPErrorHandling:
    """Tests for MCP error handling scenarios."""

    @pytest.mark.asyncio()
    async def test_search_with_missing_query(self) -> None:
        """Test that search fails gracefully without query."""
        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[_make_profile("docs")]
        )

        try:
            result = await server.call_tool("search_docs", {})

            assert result["isError"] is True
            error_text = result["content"][0]["text"]
            assert "query" in error_text.lower()

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_unknown_tool_returns_error(self) -> None:
        """Test that calling unknown tool returns error."""
        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(return_value=[])

        try:
            result = await server.call_tool("nonexistent_tool", {})

            assert result["isError"] is True
            error_text = result["content"][0]["text"]
            assert "unknown" in error_text.lower()

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_search_nonexistent_profile(self) -> None:
        """Test searching with a profile that doesn't exist."""
        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[_make_profile("coding")]
        )

        try:
            result = await server.call_tool("search_nonexistent", {"query": "test"})

            assert result["isError"] is True
            error_text = result["content"][0]["text"]
            assert "profile" in error_text.lower() or "not found" in error_text.lower()

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_profile_with_no_collections(self) -> None:
        """Test searching a profile with no collections configured."""
        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[_make_profile("empty", collections=[])]
        )

        try:
            result = await server.call_tool("search_empty", {"query": "test"})

            assert result["isError"] is True
            error_text = result["content"][0]["text"]
            assert "collection" in error_text.lower()

        finally:
            await server.api_client.close()


@pytest.mark.e2e()
class TestMCPDiagnostics:
    """Tests for the diagnostics tool."""

    @pytest.mark.asyncio()
    async def test_diagnostics_returns_server_info(self) -> None:
        """Test that diagnostics tool returns server information."""
        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                _make_profile("coding", collections=[{"id": "col-1", "name": "Code"}]),
                _make_profile("docs", collections=[{"id": "col-2", "name": "Docs"}, {"id": "col-3", "name": "API"}]),
            ]
        )

        try:
            # Verify diagnostics tool is listed
            tools = await server.list_tools()
            tool_names = {t.name for t in tools}
            assert "diagnostics" in tool_names

            # Call diagnostics tool
            result = await server.call_tool("diagnostics", {})

            assert result["isError"] is False
            payload = json.loads(result["content"][0]["text"])

            # Verify server info
            assert payload["server_name"] == "semantik"
            assert payload["webui_url"] == "http://localhost:8080"

            # Verify connection status
            assert payload["connection"]["connected"] is True
            assert payload["connection"]["authenticated"] is True
            assert payload["connection"]["profile_count"] == 2

            # Verify profile summaries
            assert len(payload["profiles"]) == 2
            profile_names = {p["name"] for p in payload["profiles"]}
            assert profile_names == {"coding", "docs"}

            # Verify collection counts
            coding_profile = next(p for p in payload["profiles"] if p["name"] == "coding")
            docs_profile = next(p for p in payload["profiles"] if p["name"] == "docs")
            assert coding_profile["collection_count"] == 1
            assert docs_profile["collection_count"] == 2

            # Verify tool count (2 search tools + 5 utility tools)
            assert payload["available_tools"] == 7

            # Verify cache status
            assert payload["cache"]["cached"] is True
            assert payload["cache"]["cached_profile_count"] == 2
            assert payload["cache"]["ttl_remaining_seconds"] >= 0

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_diagnostics_with_profile_filter(self) -> None:
        """Test diagnostics with profile filter active."""
        server = SemantikMCPServer(
            webui_url="http://localhost:8080",
            auth_token="test-token",
            profile_filter=["coding"],
        )
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                _make_profile("coding"),
                _make_profile("docs"),
            ]
        )

        try:
            result = await server.call_tool("diagnostics", {})

            assert result["isError"] is False
            payload = json.loads(result["content"][0]["text"])

            # Verify filter is shown
            assert payload["profile_filter"] == ["coding"]

            # Only filtered profiles shown
            assert len(payload["profiles"]) == 1
            assert payload["profiles"][0]["name"] == "coding"

            # Tool count reflects filtered profiles
            assert payload["available_tools"] == 6  # 1 search + 5 utility

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_diagnostics_shows_connection_error(self) -> None:
        """Test diagnostics when API connection fails."""
        from webui.mcp.client import SemantikAPIError

        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(side_effect=SemantikAPIError("Connection refused"))

        try:
            result = await server.call_tool("diagnostics", {})

            assert result["isError"] is False  # Diagnostics itself succeeds
            payload = json.loads(result["content"][0]["text"])

            # Connection error should be reported
            assert payload["connection"]["connected"] is False
            assert "Connection refused" in payload["connection"]["error"]

            # No profiles available
            assert payload["profiles"] == []

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_diagnostics_shows_auth_error(self) -> None:
        """Test diagnostics when auth token is invalid."""
        from webui.mcp.client import SemantikAPIError

        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="invalid-token")
        server.api_client.get_profiles = AsyncMock(side_effect=SemantikAPIError("401 Unauthorized"))

        try:
            result = await server.call_tool("diagnostics", {})

            assert result["isError"] is False
            payload = json.loads(result["content"][0]["text"])

            # Auth error should be reported
            assert payload["connection"]["authenticated"] is False
            assert "401" in payload["connection"]["error"]

        finally:
            await server.api_client.close()


@pytest.mark.e2e()
class TestMCPCaching:
    """Tests for profile caching behavior."""

    @pytest.mark.asyncio()
    async def test_profile_cache_reduces_api_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that profile caching reduces API calls."""
        # Mock time.monotonic to control cache behavior
        times = [0.0, 0.0, 1.0, 1.0, 2.0]  # Within 10-second TTL

        def fake_monotonic() -> float:
            return times.pop(0) if times else 2.0

        monkeypatch.setattr("webui.mcp.server.time.monotonic", fake_monotonic)

        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[_make_profile("docs")]
        )

        try:
            # Multiple list_tools calls should reuse cache
            await server.list_tools()
            await server.list_tools()
            await server.list_tools()

            # Only one API call should be made (cache hit for subsequent calls)
            assert server.api_client.get_profiles.call_count == 1

        finally:
            await server.api_client.close()

    @pytest.mark.asyncio()
    async def test_profile_cache_expires_after_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that profile cache expires after TTL."""
        times = [0.0, 0.0, 15.0, 15.0]  # 15 seconds > 10-second TTL

        def fake_monotonic() -> float:
            return times.pop(0) if times else 15.0

        monkeypatch.setattr("webui.mcp.server.time.monotonic", fake_monotonic)

        server = SemantikMCPServer(webui_url="http://localhost:8080", auth_token="test-token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[_make_profile("docs")]
        )

        try:
            await server.list_tools()  # First call - cache miss
            await server.list_tools()  # Second call - cache expired, another miss

            # Two API calls due to cache expiry
            assert server.api_client.get_profiles.call_count == 2

        finally:
            await server.api_client.close()
