from __future__ import annotations

import json
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
