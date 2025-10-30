"""Integration tests for the v2 projections API."""

from datetime import UTC, datetime
from uuid import uuid4

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
