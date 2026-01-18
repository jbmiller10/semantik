from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from webui.mcp.client import SemantikAPIClient, SemantikAPIError


def _httpx_response(*, status_code: int, json_data=None, text_data: str | None = None) -> httpx.Response:
    request = httpx.Request("GET", "http://example.invalid/test")
    if json_data is not None:
        return httpx.Response(status_code, json=json_data, request=request)
    return httpx.Response(status_code, text=text_data or "", request=request)


@pytest.mark.asyncio()
async def test_get_profiles_success_parses_list() -> None:
    client = SemantikAPIClient("http://example.invalid", "token")
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.get = AsyncMock(return_value=_httpx_response(status_code=200, json_data={"profiles": []}))
    client._client.aclose = AsyncMock()

    profiles = await client.get_profiles()
    assert profiles == []

    await client.close()


@pytest.mark.asyncio()
async def test_get_profiles_unexpected_shape_raises() -> None:
    client = SemantikAPIClient("http://example.invalid", "token")
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.get = AsyncMock(return_value=_httpx_response(status_code=200, json_data={"profiles": "nope"}))
    client._client.aclose = AsyncMock()

    with pytest.raises(SemantikAPIError, match="missing 'profiles' list"):
        await client.get_profiles()

    await client.close()


@pytest.mark.asyncio()
async def test_search_unexpected_shape_raises() -> None:
    client = SemantikAPIClient("http://example.invalid", "token")
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.post = AsyncMock(return_value=_httpx_response(status_code=200, json_data=["not", "a", "dict"]))
    client._client.aclose = AsyncMock()

    with pytest.raises(SemantikAPIError, match="expected object"):
        await client.search(query="hi")

    await client.close()


@pytest.mark.asyncio()
async def test_list_documents_unexpected_shape_raises() -> None:
    client = SemantikAPIClient("http://example.invalid", "token")
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.get = AsyncMock(return_value=_httpx_response(status_code=200, json_data=["not", "a", "dict"]))
    client._client.aclose = AsyncMock()

    with pytest.raises(SemantikAPIError, match="Unexpected response from list documents"):
        await client.list_documents("00000000-0000-4000-8000-000000000000")

    await client.close()


@pytest.mark.asyncio()
async def test_get_document_unexpected_shape_raises() -> None:
    client = SemantikAPIClient("http://example.invalid", "token")
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.get = AsyncMock(return_value=_httpx_response(status_code=200, json_data=["nope"]))
    client._client.aclose = AsyncMock()

    with pytest.raises(SemantikAPIError, match="Unexpected response from get document"):
        await client.get_document(
            "00000000-0000-4000-8000-000000000000",
            "11111111-1111-4111-8111-111111111111",
        )

    await client.close()


@pytest.mark.asyncio()
async def test_get_chunk_unexpected_shape_raises() -> None:
    client = SemantikAPIClient("http://example.invalid", "token")
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.get = AsyncMock(return_value=_httpx_response(status_code=200, json_data=["nope"]))
    client._client.aclose = AsyncMock()

    with pytest.raises(SemantikAPIError, match="Unexpected response from get chunk"):
        await client.get_chunk("00000000-0000-4000-8000-000000000000", "chunk_1")

    await client.close()


@pytest.mark.asyncio()
async def test_raise_for_status_http_error_with_json_detail() -> None:
    response = _httpx_response(status_code=403, json_data={"detail": "nope"})

    with pytest.raises(SemantikAPIError, match=r"GET /x failed \(403\):"):
        SemantikAPIClient._raise_for_status(response, "GET", "/x")  # noqa: SLF001 - unit test


@pytest.mark.asyncio()
async def test_raise_for_status_http_error_with_text_detail() -> None:
    request = httpx.Request("GET", "http://example.invalid/test")
    response = httpx.Response(500, text="boom", request=request)

    with pytest.raises(SemantikAPIError, match=r"POST /y failed \(500\): boom"):
        SemantikAPIClient._raise_for_status(response, "POST", "/y")  # noqa: SLF001 - unit test


def test_raise_for_status_request_error_branch_is_wrapped() -> None:
    class FakeResponse:
        status_code = 0
        text = ""

        def json(self):
            raise AssertionError("json() should not be called for request errors")

        def raise_for_status(self) -> None:
            raise httpx.RequestError("network down", request=httpx.Request("GET", "http://example.invalid/test"))

    with pytest.raises(SemantikAPIError, match=r"GET /z failed \(request error\):"):
        SemantikAPIClient._raise_for_status(FakeResponse(), "GET", "/z")  # type: ignore[arg-type]  # noqa: SLF001
