"""Integration tests for SearchService with real repositories and lightweight HTTP stubs."""

from __future__ import annotations

import json
from collections import deque
from uuid import uuid4

import httpx
import pytest

from shared.database.exceptions import (
    AccessDeniedError as PackagesAccessDeniedError,
    AccessDeniedError as SharedAccessDeniedError,
    EntityNotFoundError,
)
from shared.database.models import CollectionStatus
from shared.database.repositories.collection_repository import CollectionRepository
from webui.services.search_service import SearchService


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestSearchServiceIntegration:
    """Validate search flows using the real collection repository."""

    @pytest.fixture()
    def service(self, db_session):
        return SearchService(db_session, CollectionRepository(db_session))

    async def test_validate_collection_access_allows_authorized_user(self, service, collection_factory, test_user_db):
        collection = await collection_factory(owner_id=test_user_db.id)
        result = await service.validate_collection_access([collection.id], test_user_db.id)
        assert [col.id for col in result] == [collection.id]

    async def test_validate_collection_access_denies_other_user(
        self, service, collection_factory, test_user_db, other_user_db
    ):
        collection = await collection_factory(owner_id=other_user_db.id)
        with pytest.raises((PackagesAccessDeniedError, SharedAccessDeniedError)):
            await service.validate_collection_access([collection.id], test_user_db.id)

    async def test_multi_collection_search_merges_results(
        self,
        service,
        monkeypatch,
        collection_factory,
        test_user_db,
    ):
        ready_one = await collection_factory(owner_id=test_user_db.id, name=f"Ready One {uuid4().hex[:4]}")
        ready_two = await collection_factory(owner_id=test_user_db.id, name=f"Ready Two {uuid4().hex[:4]}")

        responses = deque(
            [
                httpx.Response(
                    status_code=200,
                    content=json.dumps({"results": [{"score": 0.9, "document_id": "doc-1"}]}).encode(),
                    request=httpx.Request("POST", "http://search.test/search"),
                ),
                httpx.Response(
                    status_code=200,
                    content=json.dumps({"results": [{"score": 0.7, "document_id": "doc-2"}]}).encode(),
                    request=httpx.Request("POST", "http://search.test/search"),
                ),
            ]
        )

        def fake_async_client(*_args, **_kwargs):  # noqa: ANN001
            class _Client:
                async def __aenter__(self):  # noqa: ANN001
                    return self

                async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
                    return False

                async def post(self, url, json):  # noqa: ANN001
                    response = responses.popleft()
                    response.request = httpx.Request("POST", url, json=json)
                    return response

            return _Client()

        monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)

        result = await service.multi_collection_search(
            user_id=test_user_db.id,
            collection_uuids=[ready_one.id, ready_two.id],
            query="integration",
            k=5,
        )

        assert result["metadata"]["collections_searched"] == 2
        scores = [item["score"] for item in result["results"]]
        assert scores == sorted(scores, reverse=True)

    async def test_single_collection_search_handles_not_found(
        self,
        service,
        monkeypatch,
        collection_factory,
        test_user_db,
    ):
        collection = await collection_factory(owner_id=test_user_db.id, status=CollectionStatus.READY)

        def failing_client(*_args, **_kwargs):  # noqa: ANN001
            class _Client:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def post(self, url, json):
                    response = httpx.Response(
                        status_code=404,
                        content=b"{}",
                        request=httpx.Request("POST", url, json=json),
                    )
                    raise httpx.HTTPStatusError("not found", request=response.request, response=response)

            return _Client()

        monkeypatch.setattr(httpx, "AsyncClient", failing_client)

        with pytest.raises(EntityNotFoundError):
            await service.single_collection_search(
                user_id=test_user_db.id,
                collection_uuid=collection.id,
                query="missing",
            )
