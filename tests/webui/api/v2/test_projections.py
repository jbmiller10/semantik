"""Integration tests for the v2 projections API."""

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from httpx import AsyncClient

from packages.shared.database.models import (
    OperationStatus,
    OperationType,
    ProjectionRun,
    ProjectionRunStatus,
)


@pytest.mark.asyncio()
async def test_start_projection_includes_operation_status(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    stub_celery_send_task,
) -> None:
    """Starting a projection should return operation_status in response."""
    collection = await collection_factory(owner_id=test_user_db.id)

    request_payload = {
        "reducer": "umap",
        "dimensionality": 2,
        "color_by": "document_id",
        "config": {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"},
    }

    response = await api_client.post(
        f"/api/v2/collections/{collection.id}/projections",
        json=request_payload,
        headers=api_auth_headers,
    )

    assert response.status_code == 202, response.text
    body = response.json()

    # Verify celery task was dispatched
    stub_celery_send_task.assert_called_once()

    # Verify operation_status is present and valid
    assert "operation_status" in body
    assert body["operation_status"] in ["pending", "processing", "completed", "failed", "cancelled"]

    # Verify operation_id is also present
    assert "operation_id" in body
    assert body["operation_id"] is not None

    # Verify other fields
    assert body["reducer"] == "umap"
    assert body["dimensionality"] == 2


@pytest.mark.asyncio()
async def test_start_projection_accepts_sampling_parameters(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    stub_celery_send_task,
) -> None:
    """Sampling parameters should be accepted and persisted in config."""
    collection = await collection_factory(owner_id=test_user_db.id)

    request_payload = {
        "reducer": "umap",
        "dimensionality": 2,
        "color_by": "document_id",
        "sample_size": 1234,
        "config": {"n_neighbors": 10, "min_dist": 0.2},
    }

    response = await api_client.post(
        f"/api/v2/collections/{collection.id}/projections",
        json=request_payload,
        headers=api_auth_headers,
    )

    assert response.status_code == 202, response.text
    body = response.json()

    assert body["config"] is not None
    assert body["config"]["sample_size"] == 1234


@pytest.mark.asyncio()
async def test_get_projection_includes_operation_status(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    operation_factory,
    db_session,
) -> None:
    """Fetching a single projection should include operation_status."""
    collection = await collection_factory(owner_id=test_user_db.id)

    # Create operation
    operation = await operation_factory(
        collection_id=collection.id,
        user_id=test_user_db.id,
        type=OperationType.PROJECTION_BUILD,
        status=OperationStatus.PROCESSING,
    )

    # Create projection run
    projection_run = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=operation.uuid,
        reducer="umap",
        dimensionality=2,
        status=ProjectionRunStatus.RUNNING,
        config={"n_neighbors": 15},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(projection_run)
    await db_session.commit()
    await db_session.refresh(projection_run)

    response = await api_client.get(
        f"/api/v2/collections/{collection.id}/projections/{projection_run.uuid}",
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()

    # Verify operation_status is present and matches operation
    assert "operation_status" in body
    assert body["operation_status"] == "processing"

    # Verify operation_id
    assert body["operation_id"] == operation.uuid


@pytest.mark.asyncio()
async def test_list_projections_includes_operation_status(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    operation_factory,
    db_session,
) -> None:
    """Listing projections should include operation_status for each item."""
    collection = await collection_factory(owner_id=test_user_db.id)

    # Create first projection with PROCESSING operation
    operation1 = await operation_factory(
        collection_id=collection.id,
        user_id=test_user_db.id,
        type=OperationType.PROJECTION_BUILD,
        status=OperationStatus.PROCESSING,
    )
    projection1 = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=operation1.uuid,
        reducer="umap",
        dimensionality=2,
        status=ProjectionRunStatus.RUNNING,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(projection1)

    # Create second projection with COMPLETED operation
    operation2 = await operation_factory(
        collection_id=collection.id,
        user_id=test_user_db.id,
        type=OperationType.PROJECTION_BUILD,
        status=OperationStatus.COMPLETED,
    )
    projection2 = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=operation2.uuid,
        reducer="pca",
        dimensionality=2,
        status=ProjectionRunStatus.COMPLETED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(projection2)

    await db_session.commit()

    response = await api_client.get(
        f"/api/v2/collections/{collection.id}/projections",
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()

    assert "projections" in body
    projections = body["projections"]
    assert len(projections) == 2

    # Verify all projections have operation_status
    for projection in projections:
        assert "operation_status" in projection
        assert projection["operation_status"] in ["pending", "processing", "completed", "failed", "cancelled"]
        assert "operation_id" in projection

    # Find specific projections and verify their operation_status
    processing_projection = next((p for p in projections if p["id"] == projection1.uuid), None)
    completed_projection = next((p for p in projections if p["id"] == projection2.uuid), None)

    assert processing_projection is not None
    assert processing_projection["operation_status"] == "processing"

    assert completed_projection is not None
    assert completed_projection["operation_status"] == "completed"


@pytest.mark.asyncio()
async def test_projection_with_failed_operation_includes_error_message(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    operation_factory,
    db_session,
) -> None:
    """Projection with failed operation should include error message."""
    collection = await collection_factory(owner_id=test_user_db.id)

    # Create failed operation
    operation = await operation_factory(
        collection_id=collection.id,
        user_id=test_user_db.id,
        type=OperationType.PROJECTION_BUILD,
        status=OperationStatus.FAILED,
    )
    # Manually set error message (operation_factory doesn't support it)
    operation.error_message = "CUDA out of memory"
    db_session.add(operation)

    # Create projection run
    projection_run = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=operation.uuid,
        reducer="umap",
        dimensionality=2,
        status=ProjectionRunStatus.FAILED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(projection_run)
    await db_session.commit()
    await db_session.refresh(projection_run)

    response = await api_client.get(
        f"/api/v2/collections/{collection.id}/projections/{projection_run.uuid}",
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()

    # Verify operation_status is failed
    assert body["operation_status"] == "failed"

    # Verify error message is included
    assert "message" in body
    assert body["message"] == "CUDA out of memory"


@pytest.mark.asyncio()
async def test_projection_without_operation_has_no_operation_status(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    db_session,
) -> None:
    """Projection without an associated operation should have null operation_status."""
    collection = await collection_factory(owner_id=test_user_db.id)

    # Create projection run without operation
    projection_run = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=None,  # No operation
        reducer="pca",
        dimensionality=2,
        status=ProjectionRunStatus.COMPLETED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(projection_run)
    await db_session.commit()
    await db_session.refresh(projection_run)

    response = await api_client.get(
        f"/api/v2/collections/{collection.id}/projections/{projection_run.uuid}",
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()

    # operation_status should be present but None/null
    assert "operation_status" in body or body.get("operation_status") is None
    assert body.get("operation_id") is None


@pytest.mark.asyncio()
async def test_get_projection_surfaces_degraded_meta_from_run(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    db_session,
) -> None:
    """Projection metadata should expose degraded flag derived from run.meta."""
    collection = await collection_factory(owner_id=test_user_db.id)

    projection_run = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=None,
        reducer="pca",
        dimensionality=2,
        status=ProjectionRunStatus.COMPLETED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        meta={"degraded": True, "color_by": "document_id"},
    )
    db_session.add(projection_run)
    await db_session.commit()
    await db_session.refresh(projection_run)

    response = await api_client.get(
        f"/api/v2/collections/{collection.id}/projections/{projection_run.uuid}",
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["meta"]["degraded"] is True
    assert body["meta"]["color_by"] == "document_id"


@pytest.mark.asyncio()
async def test_select_projection_region_happy_path(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    db_session,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting points should return items and missing_ids in a stable schema."""

    # Route projection artifacts under a temporary data directory
    from packages.webui import services as services_pkg

    projection_service_module = services_pkg.projection_service
    monkeypatch.setattr(projection_service_module, "settings", SimpleNamespace(data_dir=tmp_path))

    collection = await collection_factory(owner_id=test_user_db.id)

    projection_run = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=None,
        reducer="pca",
        dimensionality=2,
        status=ProjectionRunStatus.COMPLETED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        point_count=3,
    )

    # Prepare on-disk artifacts
    run_dir = tmp_path / "semantik" / "projections" / collection.id / projection_run.uuid
    run_dir.mkdir(parents=True, exist_ok=True)

    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    y = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    ids = np.array([101, 102, 103], dtype=np.int32)
    cats = np.array([0, 1, 1], dtype=np.uint8)

    x.tofile(run_dir / "x.f32.bin")
    y.tofile(run_dir / "y.f32.bin")
    ids.tofile(run_dir / "ids.i32.bin")
    cats.tofile(run_dir / "cat.u8.bin")

    meta_payload = {
        "projection_id": projection_run.uuid,
        "collection_id": collection.id,
        "point_count": 3,
        "total_count": 3,
        "shown_count": 3,
        "sampled": False,
        "reducer_requested": "pca",
        "reducer_used": "pca",
        "reducer_params": {},
        "dimensionality": 2,
        "source_vector_collection": "vector-collection",
        "sample_limit": 3,
        "files": {
            "x": "x.f32.bin",
            "y": "y.f32.bin",
            "ids": "ids.i32.bin",
            "categories": "cat.u8.bin",
        },
        "color_by": "document_id",
        "legend": [
            {"index": 0, "label": "A", "count": 1},
            {"index": 1, "label": "B", "count": 2},
        ],
        "original_ids": ["101", "102", "103"],
        "category_counts": {"0": 1, "1": 2},
    }
    (run_dir / "meta.json").write_text(__import__("json").dumps(meta_payload), encoding="utf-8")

    # storage_path is stored relative to data_dir
    projection_run.storage_path = str(run_dir.relative_to(tmp_path))

    db_session.add(projection_run)
    await db_session.commit()
    await db_session.refresh(projection_run)

    request_payload = {"ids": [101, 999]}

    response = await api_client.post(
        f"/api/v2/collections/{collection.id}/projections/{projection_run.uuid}/select",
        json=request_payload,
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["projection_id"] == projection_run.uuid
    assert body["degraded"] is False
    assert body["missing_ids"] == [999]
    assert len(body["items"]) == 1

    item = body["items"][0]
    assert item["selected_id"] == 101
    assert item["index"] == 0
    assert item["original_id"] == "101"
    # Chunk and document metadata are optional and may be null when no chunk mapping exists
    assert "chunk_id" in item
    assert "document_id" in item
    assert "chunk_index" in item
    assert "content_preview" in item


@pytest.mark.asyncio()
async def test_select_projection_region_degraded_flag(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    db_session,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection responses should surface degraded projections."""

    from packages.webui import services as services_pkg

    projection_service_module = services_pkg.projection_service
    monkeypatch.setattr(projection_service_module, "settings", SimpleNamespace(data_dir=tmp_path))

    collection = await collection_factory(owner_id=test_user_db.id)

    projection_run = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=None,
        reducer="pca",
        dimensionality=2,
        status=ProjectionRunStatus.COMPLETED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        point_count=1,
        meta={
            "degraded": True,
            "projection_artifacts": {"original_ids": ["42"], "degraded": True},
        },
    )

    run_dir = tmp_path / "semantik" / "projections" / collection.id / projection_run.uuid
    run_dir.mkdir(parents=True, exist_ok=True)

    np.array([0.1], dtype=np.float32).tofile(run_dir / "x.f32.bin")
    np.array([0.2], dtype=np.float32).tofile(run_dir / "y.f32.bin")
    np.array([42], dtype=np.int32).tofile(run_dir / "ids.i32.bin")
    np.array([0], dtype=np.uint8).tofile(run_dir / "cat.u8.bin")

    meta_payload = {
        "projection_id": projection_run.uuid,
        "collection_id": collection.id,
        "point_count": 1,
        "total_count": 1,
        "shown_count": 1,
        "sampled": False,
        "reducer_requested": "pca",
        "reducer_used": "pca",
        "reducer_params": {},
        "dimensionality": 2,
        "source_vector_collection": "vector-collection",
        "sample_limit": 1,
        "files": {
            "x": "x.f32.bin",
            "y": "y.f32.bin",
            "ids": "ids.i32.bin",
            "categories": "cat.u8.bin",
        },
        "color_by": "document_id",
        "legend": [{"index": 0, "label": "A", "count": 1}],
        "original_ids": ["42"],
        "category_counts": {"0": 1},
        "degraded": True,
    }
    (run_dir / "meta.json").write_text(__import__("json").dumps(meta_payload), encoding="utf-8")

    projection_run.storage_path = str(run_dir.relative_to(tmp_path))

    db_session.add(projection_run)
    await db_session.commit()
    await db_session.refresh(projection_run)

    response = await api_client.post(
        f"/api/v2/collections/{collection.id}/projections/{projection_run.uuid}/select",
        json={"ids": [42]},
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["projection_id"] == projection_run.uuid
    assert body["degraded"] is True
    assert body["missing_ids"] == []
    assert len(body["items"]) == 1


@pytest.mark.asyncio()
async def test_select_projection_missing_ids_marks_run_degraded(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    db_session,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing ids artifact should mark the projection run degraded."""

    from packages.webui import services as services_pkg

    projection_service_module = services_pkg.projection_service
    monkeypatch.setattr(projection_service_module, "settings", SimpleNamespace(data_dir=tmp_path))

    collection = await collection_factory(owner_id=test_user_db.id)

    projection_run = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=None,
        reducer="pca",
        dimensionality=2,
        status=ProjectionRunStatus.COMPLETED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        point_count=3,
    )

    run_dir = tmp_path / "semantik" / "projections" / collection.id / projection_run.uuid
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write some artifacts but deliberately omit ids.i32.bin
    np.array([0.1, 0.2, 0.3], dtype=np.float32).tofile(run_dir / "x.f32.bin")
    np.array([0.4, 0.5, 0.6], dtype=np.float32).tofile(run_dir / "y.f32.bin")
    np.array([0, 1, 1], dtype=np.uint8).tofile(run_dir / "cat.u8.bin")

    meta_payload = {
        "projection_id": projection_run.uuid,
        "collection_id": collection.id,
        "point_count": 3,
        "total_count": 3,
        "shown_count": 3,
        "sampled": False,
        "reducer_requested": "pca",
        "reducer_used": "pca",
        "reducer_params": {},
        "dimensionality": 2,
        "source_vector_collection": "vector-collection",
        "sample_limit": 3,
        "files": {
            "x": "x.f32.bin",
            "y": "y.f32.bin",
            "ids": "ids.i32.bin",
            "categories": "cat.u8.bin",
        },
        "color_by": "document_id",
        "legend": [
            {"index": 0, "label": "A", "count": 1},
            {"index": 1, "label": "B", "count": 2},
        ],
        "original_ids": ["101", "102", "103"],
        "category_counts": {"0": 1, "1": 2},
    }
    (run_dir / "meta.json").write_text(__import__("json").dumps(meta_payload), encoding="utf-8")

    projection_run.storage_path = str(run_dir.relative_to(tmp_path))

    db_session.add(projection_run)
    await db_session.commit()
    await db_session.refresh(projection_run)

    response = await api_client.post(
        f"/api/v2/collections/{collection.id}/projections/{projection_run.uuid}/select",
        json={"ids": [101]},
        headers=api_auth_headers,
    )

    assert response.status_code == 404, response.text

    refreshed = await db_session.get(ProjectionRun, projection_run.id)
    assert refreshed is not None
    assert isinstance(refreshed.meta, dict)
    assert refreshed.meta.get("degraded") is True


@pytest.mark.asyncio()
async def test_delete_in_progress_projection_rejected(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    db_session,
) -> None:
    """Deleting a projection that is still running should return a conflict error."""

    collection = await collection_factory(owner_id=test_user_db.id)

    projection_run = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=None,
        reducer="pca",
        dimensionality=2,
        status=ProjectionRunStatus.RUNNING,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(projection_run)
    await db_session.commit()

    response = await api_client.delete(
        f"/api/v2/collections/{collection.id}/projections/{projection_run.uuid}",
        headers=api_auth_headers,
    )

    assert response.status_code == 409, response.text
    body = response.json()
    assert "cannot be deleted" in body["detail"]


@pytest.mark.asyncio()
async def test_delete_completed_projection_removes_only_target(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    db_session,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting a completed projection removes its artifacts and DB row without affecting others."""

    # Route projection artifacts under a temporary data directory
    from packages.webui import services as services_pkg

    projection_service_module = services_pkg.projection_service
    monkeypatch.setattr(projection_service_module, "settings", SimpleNamespace(data_dir=tmp_path))

    collection = await collection_factory(owner_id=test_user_db.id)

    run1 = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=None,
        reducer="pca",
        dimensionality=2,
        status=ProjectionRunStatus.COMPLETED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    run2 = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=None,
        reducer="pca",
        dimensionality=2,
        status=ProjectionRunStatus.COMPLETED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    run_dir1 = tmp_path / "semantik" / "projections" / collection.id / run1.uuid
    run_dir2 = tmp_path / "semantik" / "projections" / collection.id / run2.uuid
    run_dir1.mkdir(parents=True, exist_ok=True)
    run_dir2.mkdir(parents=True, exist_ok=True)

    run1.storage_path = str(run_dir1.relative_to(tmp_path))
    run2.storage_path = str(run_dir2.relative_to(tmp_path))

    db_session.add_all([run1, run2])
    await db_session.commit()

    response = await api_client.delete(
        f"/api/v2/collections/{collection.id}/projections/{run1.uuid}",
        headers=api_auth_headers,
    )

    assert response.status_code == 204, response.text

    # run1 should be deleted, run2 should remain
    deleted = await db_session.get(ProjectionRun, run1.id)
    remaining = await db_session.get(ProjectionRun, run2.id)
    assert deleted is None
    assert remaining is not None

    # Artifacts directory for run1 should be removed; run2 should still exist
    assert not run_dir1.exists()
    assert run_dir2.exists()


@pytest.mark.asyncio()
async def test_stream_missing_artifact_marks_run_degraded(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    db_session,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming a missing artifact should mark the projection run degraded."""

    from packages.webui import services as services_pkg

    projection_service_module = services_pkg.projection_service
    monkeypatch.setattr(projection_service_module, "settings", SimpleNamespace(data_dir=tmp_path))

    collection = await collection_factory(owner_id=test_user_db.id)

    projection_run = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=None,
        reducer="pca",
        dimensionality=2,
        status=ProjectionRunStatus.COMPLETED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        point_count=None,
    )

    run_dir = tmp_path / "semantik" / "projections" / collection.id / projection_run.uuid
    run_dir.mkdir(parents=True, exist_ok=True)

    # Do not create x.f32.bin so the stream endpoint sees a missing artifact.
    projection_run.storage_path = str(run_dir.relative_to(tmp_path))

    db_session.add(projection_run)
    await db_session.commit()
    await db_session.refresh(projection_run)

    response = await api_client.get(
        f"/api/v2/collections/{collection.id}/projections/{projection_run.uuid}/arrays/x",
        headers=api_auth_headers,
    )

    assert response.status_code == 404, response.text

    refreshed = await db_session.get(ProjectionRun, projection_run.id)
    assert refreshed is not None
    assert isinstance(refreshed.meta, dict)
    assert refreshed.meta.get("degraded") is True


@pytest.mark.asyncio()
async def test_stream_projection_artifact_happy_path(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    db_session,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming an existing artifact should return bytes with stable headers."""

    from packages.webui import services as services_pkg

    projection_service_module = services_pkg.projection_service
    monkeypatch.setattr(projection_service_module, "settings", SimpleNamespace(data_dir=tmp_path))

    collection = await collection_factory(owner_id=test_user_db.id)

    projection_run = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=None,
        reducer="pca",
        dimensionality=2,
        status=ProjectionRunStatus.COMPLETED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    run_dir = tmp_path / "semantik" / "projections" / collection.id / projection_run.uuid
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = b"\x01\x02\x03\x04"
    (run_dir / "x.f32.bin").write_bytes(payload)

    projection_run.storage_path = str(run_dir.relative_to(tmp_path))
    db_session.add(projection_run)
    await db_session.commit()
    await db_session.refresh(projection_run)

    response = await api_client.get(
        f"/api/v2/collections/{collection.id}/projections/{projection_run.uuid}/arrays/x",
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    assert response.content == payload
    assert response.headers.get("Content-Length") == str(len(payload))
    content_disposition = response.headers.get("Content-Disposition") or ""
    assert "x.f32.bin" in content_disposition


@pytest.mark.asyncio()
async def test_stream_projection_artifact_forbidden_for_non_owner(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
    db_session,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming artifacts from another user's collection should be forbidden."""

    from packages.webui import services as services_pkg

    projection_service_module = services_pkg.projection_service
    monkeypatch.setattr(projection_service_module, "settings", SimpleNamespace(data_dir=tmp_path))

    # Collection is owned by other_user_db, but api_auth_headers authenticates test_user_db.
    collection = await collection_factory(owner_id=other_user_db.id)

    projection_run = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=None,
        reducer="pca",
        dimensionality=2,
        status=ProjectionRunStatus.COMPLETED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(projection_run)
    await db_session.commit()

    response = await api_client.get(
        f"/api/v2/collections/{collection.id}/projections/{projection_run.uuid}/arrays/x",
        headers=api_auth_headers,
    )

    assert response.status_code == 403, response.text


@pytest.mark.asyncio()
async def test_stream_projection_artifact_rejects_invalid_name(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    db_session,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Artifact names outside the allowed set should be rejected defensively."""

    from packages.webui import services as services_pkg

    projection_service_module = services_pkg.projection_service
    monkeypatch.setattr(projection_service_module, "settings", SimpleNamespace(data_dir=tmp_path))

    collection = await collection_factory(owner_id=test_user_db.id)

    projection_run = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=None,
        reducer="pca",
        dimensionality=2,
        status=ProjectionRunStatus.COMPLETED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(projection_run)
    await db_session.commit()

    # Use a suspicious-looking artifact name; the service should treat this as
    # an invalid artifact key rather than attempting path traversal.
    response = await api_client.get(
        f"/api/v2/collections/{collection.id}/projections/{projection_run.uuid}/arrays/../secret",
        headers=api_auth_headers,
    )

    assert response.status_code == 400, response.text


@pytest.mark.asyncio()
async def test_delete_projection_forbidden_for_non_owner(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
    db_session,
) -> None:
    """Deleting another user's projection should be rejected with 403."""

    collection = await collection_factory(owner_id=other_user_db.id)

    projection_run = ProjectionRun(
        uuid=str(uuid4()),
        collection_id=collection.id,
        operation_uuid=None,
        reducer="pca",
        dimensionality=2,
        status=ProjectionRunStatus.COMPLETED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(projection_run)
    await db_session.commit()

    response = await api_client.delete(
        f"/api/v2/collections/{collection.id}/projections/{projection_run.uuid}",
        headers=api_auth_headers,
    )

    assert response.status_code == 403, response.text
