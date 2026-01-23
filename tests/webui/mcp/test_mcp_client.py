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


# --------------------------------------------------------------------------
# Retry logic tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_retry_on_502_status_code(monkeypatch) -> None:
    """Test that 502 Bad Gateway triggers retry."""
    sleep_calls = []

    async def mock_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    client = SemantikAPIClient("http://example.invalid", "token", max_retries=3, base_delay=0.1)
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()

    # First two calls return 502, third returns success
    client._client.get = AsyncMock(
        side_effect=[
            _httpx_response(status_code=502, text_data="Bad Gateway"),
            _httpx_response(status_code=502, text_data="Bad Gateway"),
            _httpx_response(status_code=200, json_data={"profiles": [{"id": "1"}]}),
        ]
    )

    profiles = await client.get_profiles()
    assert profiles == [{"id": "1"}]
    assert len(sleep_calls) == 2  # Two retries
    assert sleep_calls == [0.1, 0.2]  # Exponential backoff: 0.1 * 2^0, 0.1 * 2^1

    await client.close()


@pytest.mark.asyncio()
async def test_retry_on_503_status_code(monkeypatch) -> None:
    """Test that 503 Service Unavailable triggers retry."""
    sleep_calls = []

    async def mock_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    client = SemantikAPIClient("http://example.invalid", "token", max_retries=2, base_delay=0.5)
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()

    # First call returns 503, second returns success
    client._client.post = AsyncMock(
        side_effect=[
            _httpx_response(status_code=503, text_data="Service Unavailable"),
            _httpx_response(status_code=200, json_data={"results": []}),
        ]
    )

    result = await client.search(query="test")
    assert result == {"results": []}
    assert len(sleep_calls) == 1
    assert sleep_calls[0] == 0.5  # First retry: base_delay * 2^0

    await client.close()


@pytest.mark.asyncio()
async def test_retry_on_504_status_code(monkeypatch) -> None:
    """Test that 504 Gateway Timeout triggers retry."""
    sleep_calls = []

    async def mock_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    client = SemantikAPIClient("http://example.invalid", "token", max_retries=1, base_delay=1.0)
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()

    # First call returns 504, second returns success
    client._client.get = AsyncMock(
        side_effect=[
            _httpx_response(status_code=504, text_data="Gateway Timeout"),
            _httpx_response(status_code=200, json_data={"document": {"id": "doc1"}}),
        ]
    )

    result = await client.get_document("col-id", "doc-id")
    assert result == {"document": {"id": "doc1"}}
    assert len(sleep_calls) == 1

    await client.close()


@pytest.mark.asyncio()
async def test_retry_exhausted_on_retryable_status_code() -> None:
    """Test that max retries are exhausted and final error is raised."""
    client = SemantikAPIClient("http://example.invalid", "token", max_retries=2, base_delay=0.01)
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()

    # All calls return 503
    client._client.get = AsyncMock(
        return_value=_httpx_response(status_code=503, text_data="Service Unavailable")
    )

    # After max_retries, the response is returned and _raise_for_status handles it
    with pytest.raises(SemantikAPIError, match="503"):
        await client.get_profiles()

    # Should have made 3 calls: initial + 2 retries
    assert client._client.get.call_count == 3  # noqa: SLF001 - unit test

    await client.close()


@pytest.mark.asyncio()
async def test_retry_on_timeout_exception(monkeypatch) -> None:
    """Test that TimeoutException triggers retry with proper error."""
    sleep_calls = []

    async def mock_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    client = SemantikAPIClient("http://example.invalid", "token", max_retries=2, base_delay=0.1)
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()

    # First two calls timeout, third succeeds
    request = httpx.Request("GET", "http://example.invalid/test")
    client._client.get = AsyncMock(
        side_effect=[
            httpx.TimeoutException("Connection timed out", request=request),
            httpx.TimeoutException("Connection timed out", request=request),
            _httpx_response(status_code=200, json_data={"profiles": []}),
        ]
    )

    profiles = await client.get_profiles()
    assert profiles == []
    assert len(sleep_calls) == 2
    assert sleep_calls == [0.1, 0.2]

    await client.close()


@pytest.mark.asyncio()
async def test_timeout_exception_exhausts_retries() -> None:
    """Test that timeout after max retries raises SemantikAPIError."""
    client = SemantikAPIClient("http://example.invalid", "token", max_retries=1, base_delay=0.01)
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()

    request = httpx.Request("GET", "http://example.invalid/test")
    client._client.get = AsyncMock(
        side_effect=httpx.TimeoutException("Connection timed out", request=request)
    )

    with pytest.raises(SemantikAPIError, match=r"failed \(timeout\)"):
        await client.get_profiles()

    await client.close()


@pytest.mark.asyncio()
async def test_retry_on_connection_error(monkeypatch) -> None:
    """Test that ConnectError triggers retry."""
    sleep_calls = []

    async def mock_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    client = SemantikAPIClient("http://example.invalid", "token", max_retries=2, base_delay=0.1)
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()

    # First call gets connection error, second succeeds
    request = httpx.Request("POST", "http://example.invalid/test")
    client._client.post = AsyncMock(
        side_effect=[
            httpx.ConnectError("Connection refused", request=request),
            _httpx_response(status_code=200, json_data={"results": []}),
        ]
    )

    result = await client.search(query="test")
    assert result == {"results": []}
    assert len(sleep_calls) == 1

    await client.close()


@pytest.mark.asyncio()
async def test_connection_error_exhausts_retries() -> None:
    """Test that connection error after max retries raises SemantikAPIError."""
    client = SemantikAPIClient("http://example.invalid", "token", max_retries=1, base_delay=0.01)
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()

    request = httpx.Request("GET", "http://example.invalid/test")
    client._client.get = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused", request=request)
    )

    with pytest.raises(SemantikAPIError, match=r"failed \(connection error\)"):
        await client.get_profiles()

    await client.close()


# --------------------------------------------------------------------------
# Service mode tests
# --------------------------------------------------------------------------


def test_client_requires_auth_credentials() -> None:
    """Test that client raises ValueError if neither auth_token nor internal_api_key is provided."""
    with pytest.raises(ValueError, match="Either auth_token or internal_api_key must be provided"):
        SemantikAPIClient("http://example.invalid")


def test_user_mode_sets_bearer_header() -> None:
    """Test that user mode sets Authorization header correctly."""
    client = SemantikAPIClient("http://example.invalid", "my-token")
    assert client._is_service_mode is False  # noqa: SLF001 - unit test
    assert client.is_service_mode is False


def test_service_mode_sets_internal_api_key_header() -> None:
    """Test that service mode sets X-Internal-Api-Key header."""
    client = SemantikAPIClient("http://example.invalid", internal_api_key="internal-key")
    assert client._is_service_mode is True  # noqa: SLF001 - unit test
    assert client.is_service_mode is True


@pytest.mark.asyncio()
async def test_get_all_profiles_requires_service_mode() -> None:
    """Test that get_all_profiles raises error in user mode."""
    client = SemantikAPIClient("http://example.invalid", "token")
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()

    with pytest.raises(SemantikAPIError, match="get_all_profiles\\(\\) requires service mode"):
        await client.get_all_profiles()

    await client.close()


@pytest.mark.asyncio()
async def test_get_all_profiles_success_in_service_mode() -> None:
    """Test that get_all_profiles works in service mode."""
    client = SemantikAPIClient("http://example.invalid", internal_api_key="internal-key")
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()
    client._client.get = AsyncMock(
        return_value=_httpx_response(
            status_code=200,
            json_data={"profiles": [{"id": "1", "owner_id": 100}]},
        )
    )

    profiles = await client.get_all_profiles()
    assert profiles == [{"id": "1", "owner_id": 100}]

    await client.close()


@pytest.mark.asyncio()
async def test_get_all_profiles_unexpected_shape_raises() -> None:
    """Test that get_all_profiles raises on invalid response shape."""
    client = SemantikAPIClient("http://example.invalid", internal_api_key="internal-key")
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()
    client._client.get = AsyncMock(
        return_value=_httpx_response(status_code=200, json_data={"profiles": "not-a-list"})
    )

    with pytest.raises(SemantikAPIError, match="missing 'profiles' list"):
        await client.get_all_profiles()

    await client.close()


@pytest.mark.asyncio()
async def test_validate_api_key_requires_service_mode() -> None:
    """Test that validate_api_key raises error in user mode."""
    client = SemantikAPIClient("http://example.invalid", "token")
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()

    with pytest.raises(SemantikAPIError, match="validate_api_key\\(\\) requires service mode"):
        await client.validate_api_key("some-api-key")

    await client.close()


@pytest.mark.asyncio()
async def test_validate_api_key_success_in_service_mode() -> None:
    """Test that validate_api_key works in service mode."""
    client = SemantikAPIClient("http://example.invalid", internal_api_key="internal-key")
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()
    client._client.post = AsyncMock(
        return_value=_httpx_response(
            status_code=200,
            json_data={"valid": True, "user_id": 42, "username": "testuser"},
        )
    )

    result = await client.validate_api_key("smtk_test_key")
    assert result == {"valid": True, "user_id": 42, "username": "testuser"}

    await client.close()


@pytest.mark.asyncio()
async def test_validate_api_key_unexpected_shape_raises() -> None:
    """Test that validate_api_key raises on invalid response shape."""
    client = SemantikAPIClient("http://example.invalid", internal_api_key="internal-key")
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()
    client._client.post = AsyncMock(
        return_value=_httpx_response(status_code=200, json_data=["not", "a", "dict"])
    )

    with pytest.raises(SemantikAPIError, match="expected object"):
        await client.validate_api_key("some-key")

    await client.close()


# --------------------------------------------------------------------------
# Other API methods tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_get_document_content_success() -> None:
    """Test that get_document_content returns content and content-type."""
    client = SemantikAPIClient("http://example.invalid", "token")
    client._client = MagicMock()  # noqa: SLF001 - unit test
    client._client.aclose = AsyncMock()

    request = httpx.Request("GET", "http://example.invalid/test")
    response = httpx.Response(
        200,
        content=b"Hello World",
        headers={"content-type": "text/plain"},
        request=request,
    )
    client._client.get = AsyncMock(return_value=response)

    content, content_type = await client.get_document_content("col-id", "doc-id")
    assert content == b"Hello World"
    assert content_type == "text/plain"

    await client.close()
