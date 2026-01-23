from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from vecpipe.search.cache import clear_cache
from vecpipe.search.collection_info import (
    get_cached_collection_metadata,
    get_collection_info,
    lookup_collection_from_operation,
    resolve_collection_name,
)


@pytest.mark.asyncio()
async def test_resolve_collection_name_priority() -> None:
    assert await resolve_collection_name("explicit", None, "default") == "explicit"

    with patch("vecpipe.search.collection_info.lookup_collection_from_operation", new_callable=AsyncMock) as lookup:
        lookup.return_value = "from-op"
        assert await resolve_collection_name(None, "op-uuid", "default") == "from-op"


@pytest.mark.asyncio()
async def test_resolve_collection_name_raises_when_operation_missing() -> None:
    with patch("vecpipe.search.collection_info.lookup_collection_from_operation", new_callable=AsyncMock) as lookup:
        lookup.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            await resolve_collection_name(None, "op-uuid", "default")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio()
async def test_resolve_collection_name_falls_back_to_default() -> None:
    assert await resolve_collection_name(None, None, "default") == "default"


class _FakeHttpxClient:
    def __init__(self, *, base_url: str, timeout: object, headers: dict[str, str]):  # type: ignore[no-untyped-def]
        self.base_url = base_url
        self.timeout = timeout
        self.headers = headers
        self.closed = False
        self._next_response: object | None = None
        self._next_error: Exception | None = None

    async def get(self, _path: str) -> object:  # type: ignore[no-untyped-def]
        if self._next_error:
            raise self._next_error
        assert self._next_response is not None
        return self._next_response

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio()
async def test_get_collection_info_uses_cache_when_available() -> None:
    clear_cache()
    from vecpipe.search.cache import set_collection_info

    set_collection_info("col", 123, {"config": {"params": {"vectors": {"size": 123}}}})

    qdrant_http = AsyncMock()
    dim, info = await get_collection_info(collection_name="col", cfg=Mock(), qdrant_http=qdrant_http)
    assert dim == 123
    assert info is not None
    qdrant_http.get.assert_not_called()


@pytest.mark.asyncio()
async def test_get_collection_info_fetches_and_caches_with_provided_client() -> None:
    clear_cache()
    qdrant_http = AsyncMock()
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 768}}}}}
    qdrant_http.get = AsyncMock(return_value=resp)

    dim1, _info1 = await get_collection_info(collection_name="col", cfg=Mock(), qdrant_http=qdrant_http)
    dim2, _info2 = await get_collection_info(collection_name="col", cfg=Mock(), qdrant_http=qdrant_http)

    assert dim1 == 768
    assert dim2 == 768
    qdrant_http.get.assert_awaited_once()


@pytest.mark.asyncio()
async def test_get_collection_info_creates_and_closes_ad_hoc_client(monkeypatch) -> None:
    clear_cache()
    fake_metrics = Mock()
    fake_metrics.labels.return_value.inc = Mock()
    monkeypatch.setattr("vecpipe.search.collection_info.qdrant_ad_hoc_client_total", fake_metrics)

    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 512}}}}}

    # Rebind the created instance so we can inspect it.
    created: list[_FakeHttpxClient] = []

    def _ctor(**kw):  # type: ignore[no-untyped-def]
        inst = _FakeHttpxClient(**kw)
        inst._next_response = resp
        created.append(inst)
        return inst

    monkeypatch.setattr("vecpipe.search.collection_info.httpx.AsyncClient", _ctor)

    dim, info = await get_collection_info(
        collection_name="col",
        cfg=Mock(QDRANT_HOST="h", QDRANT_PORT=1, QDRANT_API_KEY="k"),
        qdrant_http=None,
    )

    assert dim == 512
    assert info is not None
    assert created
    assert created[0].closed is True
    fake_metrics.labels.assert_called_with(location="collection_info")


@pytest.mark.asyncio()
async def test_get_collection_info_returns_default_on_error_and_closes_client(monkeypatch) -> None:
    clear_cache()
    fake_metrics = Mock()
    fake_metrics.labels.return_value.inc = Mock()
    monkeypatch.setattr("vecpipe.search.collection_info.qdrant_ad_hoc_client_total", fake_metrics)

    created: list[_FakeHttpxClient] = []

    def _ctor(**kw):  # type: ignore[no-untyped-def]
        inst = _FakeHttpxClient(**kw)
        inst._next_error = RuntimeError("boom")
        created.append(inst)
        return inst

    monkeypatch.setattr("vecpipe.search.collection_info.httpx.AsyncClient", _ctor)

    dim, info = await get_collection_info(
        collection_name="col",
        cfg=Mock(QDRANT_HOST="h", QDRANT_PORT=1, QDRANT_API_KEY=None),
        qdrant_http=None,
    )

    assert dim == 1024
    assert info is None
    assert created
    assert created[0].closed is True


@pytest.mark.asyncio()
async def test_lookup_collection_from_operation_returns_none_on_exception() -> None:
    with patch(
        "shared.database.database.ensure_async_sessionmaker", new_callable=AsyncMock, side_effect=RuntimeError("db")
    ):
        assert await lookup_collection_from_operation("op-uuid") is None


@pytest.mark.asyncio()
async def test_get_cached_collection_metadata_creates_and_closes_sdk_client(monkeypatch) -> None:
    clear_cache()
    cache_miss_sentinel = {"__cache_miss__": True}
    metadata = {"model_name": "m", "quantization": "q"}

    fake_metrics = Mock()
    fake_metrics.labels.return_value.inc = Mock()
    monkeypatch.setattr("vecpipe.search.collection_info.qdrant_ad_hoc_client_total", fake_metrics)

    mock_sdk = AsyncMock()
    mock_sdk.close = AsyncMock()

    with (
        patch("vecpipe.search.collection_info.get_collection_metadata", return_value=cache_miss_sentinel),
        patch("vecpipe.search.collection_info.is_cache_miss", return_value=True),
        patch("vecpipe.search.collection_info.set_collection_metadata") as set_cache,
        patch("qdrant_client.AsyncQdrantClient", return_value=mock_sdk) as sdk_ctor,
        patch(
            "shared.database.collection_metadata.get_collection_metadata_async",
            new_callable=AsyncMock,
            return_value=metadata,
        ),
    ):
        out = await get_cached_collection_metadata(
            collection_name="col", cfg=Mock(QDRANT_HOST="h", QDRANT_PORT=1, QDRANT_API_KEY="k"), qdrant_sdk=None
        )

    assert out == metadata
    sdk_ctor.assert_called_once_with(url="http://h:1", api_key="k")
    mock_sdk.close.assert_awaited_once()
    set_cache.assert_called_once()
    fake_metrics.labels.assert_called_with(location="metadata_fetch")


@pytest.mark.asyncio()
async def test_get_cached_collection_metadata_returns_none_on_fetch_error() -> None:
    clear_cache()
    cache_miss_sentinel = {"__cache_miss__": True}

    with (
        patch("vecpipe.search.collection_info.get_collection_metadata", return_value=cache_miss_sentinel),
        patch("vecpipe.search.collection_info.is_cache_miss", return_value=True),
        patch(
            "shared.database.collection_metadata.get_collection_metadata_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ),
    ):
        out = await get_cached_collection_metadata(
            collection_name="col", cfg=Mock(QDRANT_HOST="h", QDRANT_PORT=1, QDRANT_API_KEY=None), qdrant_sdk=AsyncMock()
        )

    assert out is None
