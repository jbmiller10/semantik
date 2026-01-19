"""Integration tests for the v2 benchmark datasets endpoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from shared.database.models import Document, DocumentStatus

if TYPE_CHECKING:
    from httpx import AsyncClient


def _dataset_payload(*, doc_uri: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "queries": [
            {
                "query_key": "q1",
                "query_text": "hello",
                "relevant_docs": [{"doc_ref": {"uri": doc_uri}, "relevance_grade": 2}],
            }
        ],
    }


@pytest.mark.asyncio()
async def test_upload_list_get_delete_dataset(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    dataset_name = f"dataset-{uuid4().hex[:8]}"
    payload = _dataset_payload(doc_uri="file:///tmp/bench-doc.txt")

    upload = await api_client.post(
        "/api/v2/benchmark-datasets",
        headers=api_auth_headers,
        data={"name": dataset_name, "description": "test dataset"},
        files={"file": ("dataset.json", json.dumps(payload).encode("utf-8"), "application/json")},
    )
    assert upload.status_code == 201, upload.text
    created = upload.json()
    assert created["name"] == dataset_name
    assert created["query_count"] == 1
    assert created["schema_version"] == "1.0"

    dataset_id = created["id"]

    listing = await api_client.get("/api/v2/benchmark-datasets", headers=api_auth_headers)
    assert listing.status_code == 200, listing.text
    listing_payload = listing.json()
    ids = {item["id"] for item in listing_payload["datasets"]}
    assert dataset_id in ids

    get_dataset = await api_client.get(f"/api/v2/benchmark-datasets/{dataset_id}", headers=api_auth_headers)
    assert get_dataset.status_code == 200, get_dataset.text
    assert get_dataset.json()["id"] == dataset_id

    deleted = await api_client.delete(f"/api/v2/benchmark-datasets/{dataset_id}", headers=api_auth_headers)
    assert deleted.status_code == 204, deleted.text

    missing = await api_client.get(f"/api/v2/benchmark-datasets/{dataset_id}", headers=api_auth_headers)
    assert missing.status_code == 404, missing.text


@pytest.mark.asyncio()
async def test_dataset_isolated_across_users(
    api_client_other_user: AsyncClient,
    other_user_auth_headers: dict[str, str],
    test_user_db,
    db_session,
) -> None:
    from shared.database.models import BenchmarkDataset

    dataset_id = str(uuid4())
    dataset = BenchmarkDataset(
        id=dataset_id,
        name=f"dataset-{uuid4().hex[:8]}",
        description=None,
        owner_id=test_user_db.id,
        query_count=1,
        raw_file_path=None,
        schema_version="1.0",
        meta=None,
    )
    db_session.add(dataset)
    await db_session.commit()

    other_get = await api_client_other_user.get(
        f"/api/v2/benchmark-datasets/{dataset_id}",
        headers=other_user_auth_headers,
    )
    assert other_get.status_code == 403, other_get.text

    other_list = await api_client_other_user.get("/api/v2/benchmark-datasets", headers=other_user_auth_headers)
    assert other_list.status_code == 200, other_list.text
    other_ids = {item["id"] for item in other_list.json()["datasets"]}
    assert dataset_id not in other_ids


@pytest.mark.asyncio()
async def test_create_mapping_and_resolve_mapping_sync(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    db_session,
) -> None:
    collection = await collection_factory(owner_id=test_user_db.id)
    doc_uri = "file:///tmp/bench-doc-sync.txt"
    doc = Document(
        id=str(uuid4()),
        collection_id=collection.id,
        file_path="/tmp/bench-doc-sync.txt",
        file_name="bench-doc-sync.txt",
        file_size=1,
        mime_type="text/plain",
        content_hash=uuid4().hex,
        status=DocumentStatus.COMPLETED,
        chunk_count=0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        uri=doc_uri,
    )
    db_session.add(doc)
    await db_session.commit()

    dataset_name = f"dataset-{uuid4().hex[:8]}"
    payload = _dataset_payload(doc_uri=doc_uri)

    upload = await api_client.post(
        "/api/v2/benchmark-datasets",
        headers=api_auth_headers,
        data={"name": dataset_name},
        files={"file": ("dataset.json", json.dumps(payload).encode("utf-8"), "application/json")},
    )
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["id"]

    mapping = await api_client.post(
        f"/api/v2/benchmark-datasets/{dataset_id}/mappings",
        headers=api_auth_headers,
        json={"collection_id": collection.id},
    )
    assert mapping.status_code == 201, mapping.text
    mapping_payload = mapping.json()
    assert mapping_payload["dataset_id"] == dataset_id
    assert mapping_payload["collection_id"] == collection.id

    mapping_id = mapping_payload["id"]

    list_mappings = await api_client.get(
        f"/api/v2/benchmark-datasets/{dataset_id}/mappings",
        headers=api_auth_headers,
    )
    assert list_mappings.status_code == 200, list_mappings.text
    assert any(m["id"] == mapping_id for m in list_mappings.json())

    get_mapping = await api_client.get(
        f"/api/v2/benchmark-datasets/{dataset_id}/mappings/{mapping_id}",
        headers=api_auth_headers,
    )
    assert get_mapping.status_code == 200, get_mapping.text
    assert get_mapping.json()["id"] == mapping_id

    resolve = await api_client.post(
        f"/api/v2/benchmark-datasets/{dataset_id}/mappings/{mapping_id}/resolve",
        headers=api_auth_headers,
    )
    assert resolve.status_code == 200, resolve.text
    resolve_payload = resolve.json()
    assert resolve_payload["id"] == mapping_id
    assert resolve_payload["operation_uuid"] is None
    assert resolve_payload["mapped_count"] == 1
    assert resolve_payload["total_count"] == 1
    assert resolve_payload["mapping_status"] == "resolved"
    assert resolve_payload["unresolved"] == []


@pytest.mark.asyncio()
async def test_resolve_mapping_routes_large_jobs_to_async_contract(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    stub_celery_send_task,
    monkeypatch,
) -> None:
    # Force async routing.
    monkeypatch.setattr("shared.config.settings.BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_REFS", 0)
    monkeypatch.setattr("shared.config.settings.BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_DOCS", 0)

    collection = await collection_factory(owner_id=test_user_db.id)

    dataset_name = f"dataset-{uuid4().hex[:8]}"
    payload = _dataset_payload(doc_uri="file:///tmp/bench-doc-async.txt")

    upload = await api_client.post(
        "/api/v2/benchmark-datasets",
        headers=api_auth_headers,
        data={"name": dataset_name},
        files={"file": ("dataset.json", json.dumps(payload).encode("utf-8"), "application/json")},
    )
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["id"]

    mapping = await api_client.post(
        f"/api/v2/benchmark-datasets/{dataset_id}/mappings",
        headers=api_auth_headers,
        json={"collection_id": collection.id},
    )
    assert mapping.status_code == 201, mapping.text
    mapping_id = mapping.json()["id"]

    stub_celery_send_task.reset_mock()
    resolve = await api_client.post(
        f"/api/v2/benchmark-datasets/{dataset_id}/mappings/{mapping_id}/resolve",
        headers=api_auth_headers,
    )
    assert resolve.status_code == 202, resolve.text
    payload = resolve.json()
    assert payload["id"] == mapping_id
    assert payload["operation_uuid"]

    stub_celery_send_task.assert_called_once()
    name_arg = stub_celery_send_task.call_args.args[0]
    assert name_arg == "webui.tasks.benchmark_mapping.resolve_mapping"
