from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from vecpipe.search.errors import extract_qdrant_error, maybe_raise_for_status, response_json


@pytest.mark.asyncio()
async def test_response_json_supports_sync_and_async_json_methods() -> None:
    class SyncResponse:
        def json(self):  # type: ignore[no-untyped-def]
            return {"ok": True}

    class AsyncResponse:
        async def json(self):  # type: ignore[no-untyped-def]
            return {"ok": True}

    assert await response_json(SyncResponse()) == {"ok": True}
    assert await response_json(AsyncResponse()) == {"ok": True}


@pytest.mark.asyncio()
async def test_maybe_raise_for_status_handles_missing_sync_and_async_methods() -> None:
    class NoRaiseForStatus:
        pass

    class SyncRaiseForStatus:
        def __init__(self) -> None:
            self.raise_for_status = Mock()

    class AsyncRaiseForStatus:
        def __init__(self) -> None:
            self.raise_for_status = AsyncMock()

    await maybe_raise_for_status(NoRaiseForStatus())

    sync = SyncRaiseForStatus()
    await maybe_raise_for_status(sync)
    sync.raise_for_status.assert_called_once()

    async_resp = AsyncRaiseForStatus()
    await maybe_raise_for_status(async_resp)
    async_resp.raise_for_status.assert_awaited_once()


def test_extract_qdrant_error_prefers_payload_status_error_then_error_key() -> None:
    request = httpx.Request("GET", "http://qdrant.local")

    resp_status_error = Mock()
    resp_status_error.json.return_value = {"status": {"error": "status boom"}}
    exc1 = httpx.HTTPStatusError("boom", request=request, response=resp_status_error)
    assert extract_qdrant_error(exc1) == "status boom"

    resp_error = Mock()
    resp_error.json.return_value = {"error": "plain boom"}
    exc2 = httpx.HTTPStatusError("boom", request=request, response=resp_error)
    assert extract_qdrant_error(exc2) == "plain boom"


def test_extract_qdrant_error_returns_default_on_unexpected_payloads() -> None:
    request = httpx.Request("GET", "http://qdrant.local")

    resp_non_dict = Mock()
    resp_non_dict.json.return_value = ["not-a-dict"]
    exc1 = httpx.HTTPStatusError("boom", request=request, response=resp_non_dict)
    assert extract_qdrant_error(exc1) == "Vector database error"

    resp_parse_error = Mock()
    resp_parse_error.json.side_effect = ValueError("bad json")
    exc2 = httpx.HTTPStatusError("boom", request=request, response=resp_parse_error)
    assert extract_qdrant_error(exc2) == "Vector database error"
