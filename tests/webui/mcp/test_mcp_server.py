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
