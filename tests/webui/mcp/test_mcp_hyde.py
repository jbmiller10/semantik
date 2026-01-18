"""Tests for HyDE (Hypothetical Document Embeddings) support in MCP server."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from webui.mcp.server import SemantikMCPServer
from webui.mcp.tools import build_search_tool


class TestMCPSearchToolHyDESchema:
    """Tests for HyDE parameter in MCP search tool schema."""

    def test_search_tool_includes_use_hyde_parameter(self) -> None:
        """Search tool schema includes use_hyde boolean parameter."""
        profile = {
            "name": "test",
            "description": "Test profile",
            "result_count": 10,
            "search_type": "semantic",
            "use_reranker": True,
            "use_hyde": False,
        }
        tool = build_search_tool(name="search_test", description="Test", profile=profile)

        properties = tool.inputSchema["properties"]
        assert "use_hyde" in properties
        assert properties["use_hyde"]["type"] == "boolean"

    def test_search_tool_hyde_description_includes_profile_default(self) -> None:
        """Search tool use_hyde description shows profile default value."""
        profile_hyde_enabled = {"name": "test", "use_hyde": True}
        tool_enabled = build_search_tool(name="search_test", description="Test", profile=profile_hyde_enabled)

        profile_hyde_disabled = {"name": "test", "use_hyde": False}
        tool_disabled = build_search_tool(name="search_test", description="Test", profile=profile_hyde_disabled)

        assert "True" in tool_enabled.inputSchema["properties"]["use_hyde"]["description"]
        assert "False" in tool_disabled.inputSchema["properties"]["use_hyde"]["description"]

    def test_search_tool_hyde_default_when_profile_missing(self) -> None:
        """Search tool use_hyde defaults to False when profile omits it."""
        profile = {"name": "test"}  # No use_hyde field
        tool = build_search_tool(name="search_test", description="Test", profile=profile)

        # Default should be False
        assert "False" in tool.inputSchema["properties"]["use_hyde"]["description"]


@pytest.mark.asyncio()
class TestMCPServerHyDEExecution:
    """Tests for HyDE parameter handling in MCP server search execution."""

    @pytest.fixture()
    def server_with_profile(self) -> SemantikMCPServer:
        """Create MCP server with a test profile."""
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
                    "use_hyde": True,  # Profile default: HyDE enabled
                    "collections": [{"id": "c1", "name": "Collection 1"}],
                }
            ]
        )
        return server

    @pytest.fixture()
    def mock_search_response(self) -> dict:
        """Standard mock search response."""
        return {
            "query": "test query",
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
                    "text": "Sample result text",
                    "metadata": {},
                }
            ],
        }

    async def test_search_uses_profile_hyde_default(
        self, server_with_profile: SemantikMCPServer, mock_search_response: dict
    ) -> None:
        """Search uses profile's use_hyde setting when not specified in arguments."""
        server_with_profile.api_client.search = AsyncMock(return_value=mock_search_response)

        await server_with_profile.call_tool("search_coding", {"query": "test"})

        # Verify use_hyde was passed with profile default (True)
        call_kwargs = server_with_profile.api_client.search.call_args.kwargs
        assert call_kwargs["use_hyde"] is True

        await server_with_profile.api_client.close()

    async def test_search_argument_overrides_profile_hyde_true(
        self, server_with_profile: SemantikMCPServer, mock_search_response: dict
    ) -> None:
        """Explicit use_hyde=False argument overrides profile's True default."""
        server_with_profile.api_client.search = AsyncMock(return_value=mock_search_response)

        await server_with_profile.call_tool("search_coding", {"query": "test", "use_hyde": False})

        call_kwargs = server_with_profile.api_client.search.call_args.kwargs
        assert call_kwargs["use_hyde"] is False

        await server_with_profile.api_client.close()

    async def test_search_argument_overrides_profile_hyde_false(self) -> None:
        """Explicit use_hyde=True argument overrides profile's False default."""
        server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                {
                    "name": "coding",
                    "description": "Coding docs",
                    "enabled": True,
                    "use_hyde": False,  # Profile default: HyDE disabled
                    "collections": [{"id": "c1", "name": "Collection 1"}],
                }
            ]
        )
        server.api_client.search = AsyncMock(
            return_value={
                "query": "test",
                "total_results": 0,
                "results": [],
            }
        )

        await server.call_tool("search_coding", {"query": "test", "use_hyde": True})

        call_kwargs = server.api_client.search.call_args.kwargs
        assert call_kwargs["use_hyde"] is True

        await server.api_client.close()

    async def test_search_passes_all_hyde_related_params(
        self, server_with_profile: SemantikMCPServer, mock_search_response: dict
    ) -> None:
        """Search passes use_hyde along with other parameters to API client."""
        server_with_profile.api_client.search = AsyncMock(return_value=mock_search_response)

        await server_with_profile.call_tool(
            "search_coding",
            {
                "query": "test query",
                "k": 10,
                "search_type": "semantic",
                "search_mode": "hybrid",
                "use_reranker": True,
                "use_hyde": True,
            },
        )

        call_kwargs = server_with_profile.api_client.search.call_args.kwargs
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["k"] == 10
        assert call_kwargs["use_hyde"] is True
        assert call_kwargs["search_mode"] == "hybrid"
        assert call_kwargs["use_reranker"] is True

        await server_with_profile.api_client.close()

    async def test_search_response_formatting_with_hyde_info(self) -> None:
        """Search response includes HyDE-related information if present."""
        server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                {
                    "name": "coding",
                    "description": "Coding docs",
                    "enabled": True,
                    "use_hyde": True,
                    "collections": [{"id": "c1", "name": "Collection 1"}],
                }
            ]
        )
        # Response with HyDE metadata (if API includes it)
        server.api_client.search = AsyncMock(
            return_value={
                "query": "test",
                "total_results": 1,
                "results": [
                    {
                        "collection_id": "c1",
                        "collection_name": "Collection 1",
                        "document_id": "d1",
                        "chunk_id": "d1_0001",
                        "score": 0.95,
                        "text": "Result text",
                        "metadata": {},
                    }
                ],
                "hyde_used": True,
                "hyde_expanded_query": "Hypothetical document about test...",
            }
        )

        result = await server.call_tool("search_coding", {"query": "test"})

        assert result["isError"] is False
        payload = json.loads(result["content"][0]["text"])
        assert payload["query"] == "test"
        assert payload["total_results"] == 1

        await server.api_client.close()


@pytest.mark.asyncio()
class TestMCPServerHyDEEdgeCases:
    """Edge case tests for HyDE in MCP server."""

    async def test_profile_without_use_hyde_defaults_to_false(self) -> None:
        """Profile missing use_hyde field defaults to False."""
        server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                {
                    "name": "legacy",
                    "description": "Legacy profile without HyDE field",
                    "enabled": True,
                    # Note: no use_hyde field
                    "collections": [{"id": "c1", "name": "Collection 1"}],
                }
            ]
        )
        server.api_client.search = AsyncMock(
            return_value={
                "query": "test",
                "total_results": 0,
                "results": [],
            }
        )

        await server.call_tool("search_legacy", {"query": "test"})

        call_kwargs = server.api_client.search.call_args.kwargs
        assert call_kwargs["use_hyde"] is False

        await server.api_client.close()

    async def test_use_hyde_boolean_coercion(self) -> None:
        """Truthy/falsy values are coerced to boolean for use_hyde."""
        server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                {
                    "name": "test",
                    "description": "Test",
                    "enabled": True,
                    "use_hyde": False,
                    "collections": [{"id": "c1", "name": "Collection 1"}],
                }
            ]
        )
        server.api_client.search = AsyncMock(return_value={"query": "q", "total_results": 0, "results": []})

        # Test truthy value (1 should be coerced to True)
        await server.call_tool("search_test", {"query": "test", "use_hyde": 1})
        call_kwargs = server.api_client.search.call_args.kwargs
        assert call_kwargs["use_hyde"] is True

        await server.api_client.close()

    async def test_list_tools_includes_hyde_in_schema(self) -> None:
        """list_tools returns tools with use_hyde in their schema."""
        server = SemantikMCPServer(webui_url="http://example.invalid", auth_token="token")
        server.api_client.get_profiles = AsyncMock(
            return_value=[
                {
                    "name": "docs",
                    "description": "Documentation search",
                    "enabled": True,
                    "use_hyde": True,
                    "collections": [{"id": "c1", "name": "Docs"}],
                }
            ]
        )

        tools = await server.list_tools()

        search_tool = next((t for t in tools if t.name == "search_docs"), None)
        assert search_tool is not None
        assert "use_hyde" in search_tool.inputSchema["properties"]

        await server.api_client.close()
