from __future__ import annotations

from unittest.mock import Mock

import pytest


class FakeResponse:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):  # type: ignore[no-untyped-def]
        return self._payload


class FakeAsyncClient:
    def __init__(self, *, base_url: str, timeout: object, headers: dict[str, str]):  # type: ignore[no-untyped-def]
        self.base_url = base_url
        self.timeout = timeout
        self.headers = headers
        self.posts: list[tuple[str, dict]] = []
        self.closed = False
        self.next_response: FakeResponse | None = None

    async def post(self, path: str, *, json: dict):  # type: ignore[no-untyped-def]
        self.posts.append((path, json))
        assert self.next_response is not None
        return self.next_response

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio()
async def test_fetch_payloads_builds_scroll_filter_and_closes_created_client(monkeypatch) -> None:
    from vecpipe.search.payloads import fetch_payloads_for_chunk_ids

    fake_client = FakeAsyncClient(base_url="http://h:1", timeout=object(), headers={"api-key": "k"})
    fake_client.next_response = FakeResponse(
        200,
        {
            "result": {
                "points": [
                    {"payload": {"chunk_id": "c1", "doc_id": "d1"}},
                    {"payload": {"chunk_id": "c2", "doc_id": "d2"}},
                ]
            }
        },
    )

    monkeypatch.setattr("vecpipe.search.payloads.httpx.AsyncClient", lambda **kw: FakeAsyncClient(**kw))

    # Re-bind the instance created by our patched constructor so we can inspect it.
    created: list[FakeAsyncClient] = []

    def _ctor(**kw):  # type: ignore[no-untyped-def]
        inst = FakeAsyncClient(**kw)
        inst.next_response = fake_client.next_response
        created.append(inst)
        return inst

    monkeypatch.setattr("vecpipe.search.payloads.httpx.AsyncClient", _ctor)

    filters = {"must": [{"key": "tenant", "match": {"value": "t1"}}]}
    payloads = await fetch_payloads_for_chunk_ids(
        collection_name="dense",
        chunk_ids=["c1", "c2"],
        cfg=Mock(QDRANT_HOST="h", QDRANT_PORT=1, QDRANT_API_KEY="k"),
        qdrant_http=None,
        filters=filters,
    )
    assert payloads["c1"]["doc_id"] == "d1"
    assert payloads["c2"]["doc_id"] == "d2"

    assert len(created) == 1
    inst = created[0]
    assert inst.headers == {"api-key": "k"}
    assert inst.closed is True

    assert inst.posts
    path, body = inst.posts[0]
    assert path == "/collections/dense/points/scroll"
    # Filter should be flattened: chunk_id condition first, then user conditions
    assert body["filter"]["must"][0]["key"] == "chunk_id"
    # User filter conditions are merged (not nested)
    assert body["filter"]["must"][1] == {"key": "tenant", "match": {"value": "t1"}}


@pytest.mark.asyncio()
async def test_fetch_payloads_returns_empty_on_http_error() -> None:
    from vecpipe.search.payloads import fetch_payloads_for_chunk_ids

    client = FakeAsyncClient(base_url="http://h:1", timeout=object(), headers={})
    client.next_response = FakeResponse(500, {"status": {"error": "boom"}}, text="boom")

    payloads = await fetch_payloads_for_chunk_ids(
        collection_name="dense",
        chunk_ids=["c1"],
        cfg=Mock(QDRANT_HOST="h", QDRANT_PORT=1, QDRANT_API_KEY=None),
        qdrant_http=client,  # Not a created client, should not be closed.
        filters=None,
    )
    assert payloads == {}
    assert client.closed is False  # not a created client
