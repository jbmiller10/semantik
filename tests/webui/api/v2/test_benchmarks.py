"""Integration tests for the v2 benchmarks endpoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from shared.database.models import Document, DocumentStatus

if TYPE_CHECKING:
    from httpx import AsyncClient


async def _create_resolved_mapping(
    *,
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    collection_id: str,
    db_session,
) -> tuple[str, int]:
    """Create a dataset + mapping and resolve it synchronously (small job)."""
    doc_uri = f"file:///tmp/bench-{uuid4().hex}.txt"
    doc = Document(
        id=str(uuid4()),
        collection_id=collection_id,
        file_path="/tmp/bench.txt",
        file_name="bench.txt",
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

    dataset_payload = {
        "schema_version": "1.0",
        "queries": [
            {
                "query_key": "q1",
                "query_text": "hello",
                "relevant_docs": [{"doc_ref": {"uri": doc_uri}, "relevance_grade": 2}],
            }
        ],
    }

    dataset_name = f"dataset-{uuid4().hex[:8]}"
    upload = await api_client.post(
        "/api/v2/benchmark-datasets",
        headers=api_auth_headers,
        data={"name": dataset_name},
        files={"file": ("dataset.json", json.dumps(dataset_payload).encode("utf-8"), "application/json")},
    )
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["id"]

    mapping = await api_client.post(
        f"/api/v2/benchmark-datasets/{dataset_id}/mappings",
        headers=api_auth_headers,
        json={"collection_id": collection_id},
    )
    assert mapping.status_code == 201, mapping.text
    mapping_id = int(mapping.json()["id"])

    resolve = await api_client.post(
        f"/api/v2/benchmark-datasets/{dataset_id}/mappings/{mapping_id}/resolve",
        headers=api_auth_headers,
    )
    assert resolve.status_code == 200, resolve.text
    assert resolve.json()["mapping_status"] == "resolved"

    return dataset_id, mapping_id


@pytest.mark.asyncio()
async def test_create_list_get_results_query_results_and_delete_benchmark(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    db_session,
) -> None:
    collection = await collection_factory(owner_id=test_user_db.id)
    _, mapping_id = await _create_resolved_mapping(
        api_client=api_client,
        api_auth_headers=api_auth_headers,
        collection_id=collection.id,
        db_session=db_session,
    )

    create = await api_client.post(
        "/api/v2/benchmarks",
        headers=api_auth_headers,
        json={
            "name": f"bench-{uuid4().hex[:8]}",
            "description": "benchmark test",
            "mapping_id": mapping_id,
            "config_matrix": {
                "search_modes": ["dense"],
                "use_reranker": [False],
                "top_k_values": [10],
                "rrf_k_values": [60],
                "score_thresholds": [None],
            },
            "top_k": 10,
            "metrics_to_compute": ["precision", "recall", "mrr", "ndcg"],
        },
    )
    assert create.status_code == 201, create.text
    created = create.json()
    benchmark_id = created["id"]
    assert created["status"] == "pending"
    assert created["total_runs"] == 1

    listing = await api_client.get("/api/v2/benchmarks", headers=api_auth_headers)
    assert listing.status_code == 200, listing.text
    ids = {item["id"] for item in listing.json()["benchmarks"]}
    assert benchmark_id in ids

    get_benchmark = await api_client.get(f"/api/v2/benchmarks/{benchmark_id}", headers=api_auth_headers)
    assert get_benchmark.status_code == 200, get_benchmark.text
    assert get_benchmark.json()["id"] == benchmark_id

    results = await api_client.get(f"/api/v2/benchmarks/{benchmark_id}/results", headers=api_auth_headers)
    assert results.status_code == 200, results.text
    results_payload = results.json()
    assert results_payload["benchmark_id"] == benchmark_id
    assert results_payload["total_runs"] == 1
    assert len(results_payload["runs"]) == 1
    run = results_payload["runs"][0]
    assert run["config"]["search_mode"] == "dense"
    assert "metrics" in run
    assert "precision" in run["metrics"]
    assert "recall" in run["metrics"]
    assert "ndcg" in run["metrics"]

    run_id = run["id"]
    query_results = await api_client.get(
        f"/api/v2/benchmarks/{benchmark_id}/runs/{run_id}/queries",
        headers=api_auth_headers,
    )
    assert query_results.status_code == 200, query_results.text
    query_payload = query_results.json()
    assert query_payload["run_id"] == run_id
    assert query_payload["results"] == []
    assert query_payload["total"] == 0

    deleted = await api_client.delete(f"/api/v2/benchmarks/{benchmark_id}", headers=api_auth_headers)
    assert deleted.status_code == 204, deleted.text

    missing = await api_client.get(f"/api/v2/benchmarks/{benchmark_id}", headers=api_auth_headers)
    assert missing.status_code == 404, missing.text


@pytest.mark.asyncio()
async def test_start_and_cancel_benchmark_dispatches_celery(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    db_session,
    stub_celery_send_task,
) -> None:
    collection = await collection_factory(owner_id=test_user_db.id)
    _, mapping_id = await _create_resolved_mapping(
        api_client=api_client,
        api_auth_headers=api_auth_headers,
        collection_id=collection.id,
        db_session=db_session,
    )

    create = await api_client.post(
        "/api/v2/benchmarks",
        headers=api_auth_headers,
        json={
            "name": f"bench-{uuid4().hex[:8]}",
            "mapping_id": mapping_id,
            "config_matrix": {
                "search_modes": ["dense"],
                "use_reranker": [False],
                "top_k_values": [10],
                "rrf_k_values": [60],
                "score_thresholds": [None],
            },
        },
    )
    assert create.status_code == 201, create.text
    benchmark_id = create.json()["id"]

    stub_celery_send_task.reset_mock()
    start = await api_client.post(f"/api/v2/benchmarks/{benchmark_id}/start", headers=api_auth_headers)
    assert start.status_code == 200, start.text
    start_payload = start.json()
    assert start_payload["id"] == benchmark_id
    assert start_payload["status"] == "running"
    assert start_payload["operation_uuid"]

    stub_celery_send_task.assert_called_once()
    assert stub_celery_send_task.call_args.args[0] == "webui.tasks.benchmark.run_benchmark"
    assert stub_celery_send_task.call_args.kwargs["kwargs"]["benchmark_id"] == benchmark_id
    assert stub_celery_send_task.call_args.kwargs["kwargs"]["operation_uuid"] == start_payload["operation_uuid"]

    cancel = await api_client.post(f"/api/v2/benchmarks/{benchmark_id}/cancel", headers=api_auth_headers)
    assert cancel.status_code == 200, cancel.text
    assert cancel.json()["status"] == "cancelled"


@pytest.mark.asyncio()
async def test_benchmark_isolated_across_users(
    api_client_other_user: AsyncClient,
    other_user_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    db_session,
) -> None:
    from shared.database.models import (
        Benchmark,
        BenchmarkDataset,
        BenchmarkDatasetMapping,
        BenchmarkStatus,
        MappingStatus,
    )

    collection = await collection_factory(owner_id=test_user_db.id)

    dataset = BenchmarkDataset(
        id=str(uuid4()),
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

    mapping = BenchmarkDatasetMapping(
        dataset_id=dataset.id,
        collection_id=collection.id,
        mapping_status=MappingStatus.RESOLVED.value,
        mapped_count=1,
        total_count=1,
    )
    db_session.add(mapping)
    await db_session.commit()
    await db_session.refresh(mapping)

    benchmark_id = str(uuid4())
    benchmark = Benchmark(
        id=benchmark_id,
        name=f"bench-{uuid4().hex[:8]}",
        description=None,
        owner_id=test_user_db.id,
        mapping_id=int(mapping.id),
        operation_uuid=None,
        evaluation_unit="query",
        config_matrix={"search_modes": ["dense"], "use_reranker": [False], "top_k_values": [10], "rrf_k_values": [60]},
        config_matrix_hash="hash",
        limits=None,
        collection_snapshot_hash=None,
        reproducibility_metadata=None,
        top_k=10,
        metrics_to_compute=["precision"],
        status=BenchmarkStatus.PENDING.value,
        total_runs=0,
        completed_runs=0,
        failed_runs=0,
    )
    db_session.add(benchmark)
    await db_session.commit()

    other_get = await api_client_other_user.get(f"/api/v2/benchmarks/{benchmark_id}", headers=other_user_auth_headers)
    assert other_get.status_code == 403, other_get.text

    other_results = await api_client_other_user.get(
        f"/api/v2/benchmarks/{benchmark_id}/results", headers=other_user_auth_headers
    )
    assert other_results.status_code == 403, other_results.text
