"""Unit tests for ProjectionService and helpers."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from shared.database.exceptions import AccessDeniedError
from shared.database.models import OperationStatus, ProjectionRunStatus

from packages.webui.services import projection_service as projection_module
from packages.webui.services.projection_service import (
    ProjectionService,
    compute_projection_metadata_hash,
)


def test_encode_projection_without_operation() -> None:
    """Test _encode_projection returns basic projection metadata without operation_status."""
    # Create mock projection run
    run = MagicMock()
    run.collection_id = "coll-123"
    run.uuid = "proj-456"
    run.status = ProjectionRunStatus.COMPLETED
    run.reducer = "umap"
    run.dimensionality = 2
    run.created_at = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
    run.operation_uuid = "op-789"
    run.config = {"n_neighbors": 15, "min_dist": 0.1, "color_by": "document_id"}
    run.meta = {"legend": [{"index": 0, "label": "Doc A"}]}

    result = ProjectionService._encode_projection(run)

    assert result["collection_id"] == "coll-123"
    assert result["projection_id"] == "proj-456"
    assert result["status"] == "completed"
    assert result["reducer"] == "umap"
    assert result["dimensionality"] == 2
    assert result["created_at"] == datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
    assert result["operation_id"] == "op-789"
    assert result["config"] == {"n_neighbors": 15, "min_dist": 0.1, "color_by": "document_id"}
    assert result["meta"] == {"legend": [{"index": 0, "label": "Doc A"}], "color_by": "document_id"}
    assert result["message"] is None
    # operation_status should NOT be present when operation is not provided
    assert "operation_status" not in result


def test_encode_projection_with_operation() -> None:
    """Test _encode_projection includes operation_status when operation provided."""
    # Create mock projection run
    run = MagicMock()
    run.collection_id = "coll-123"
    run.uuid = "proj-456"
    run.status = ProjectionRunStatus.RUNNING
    run.reducer = "umap"
    run.dimensionality = 2
    run.created_at = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
    run.operation_uuid = "op-789"
    run.config = {"n_neighbors": 15}
    run.meta = {}

    # Create mock operation
    operation = MagicMock()
    operation.uuid = "op-789"
    operation.status = OperationStatus.PROCESSING
    operation.error_message = None

    result = ProjectionService._encode_projection(run, operation=operation)

    assert result["operation_id"] == "op-789"
    assert result["operation_status"] == "processing"
    assert result["message"] is None


def test_encode_projection_with_failed_operation() -> None:
    """Test _encode_projection includes error message from failed operation."""
    # Create mock projection run
    run = MagicMock()
    run.collection_id = "coll-123"
    run.uuid = "proj-456"
    run.status = ProjectionRunStatus.FAILED
    run.reducer = "umap"
    run.dimensionality = 2
    run.created_at = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
    run.operation_uuid = "op-789"
    run.config = None
    run.meta = None

    # Create mock failed operation
    operation = MagicMock()
    operation.uuid = "op-789"
    operation.status = OperationStatus.FAILED
    operation.error_message = "CUDA out of memory"

    result = ProjectionService._encode_projection(run, operation=operation)

    assert result["operation_status"] == "failed"
    assert result["message"] == "CUDA out of memory"


def test_encode_projection_custom_message_overrides_error() -> None:
    """Test that custom message parameter takes precedence over operation error_message."""
    # Create mock projection run
    run = MagicMock()
    run.collection_id = "coll-123"
    run.uuid = "proj-456"
    run.status = ProjectionRunStatus.FAILED
    run.reducer = "umap"
    run.dimensionality = 2
    run.created_at = None
    run.operation_uuid = "op-789"
    run.config = None
    run.meta = None

    # Create mock failed operation
    operation = MagicMock()
    operation.uuid = "op-789"
    operation.status = OperationStatus.FAILED
    operation.error_message = "CUDA out of memory"

    result = ProjectionService._encode_projection(run, operation=operation, message="Custom error message")

    assert result["operation_status"] == "failed"
    # Custom message should take precedence
    assert result["message"] == "Custom error message"


def test_encode_projection_color_by_in_config() -> None:
    """Test that color_by from config is propagated to meta."""
    # Create mock projection run
    run = MagicMock()
    run.collection_id = "coll-123"
    run.uuid = "proj-456"
    run.status = ProjectionRunStatus.COMPLETED
    run.reducer = "umap"
    run.dimensionality = 2
    run.created_at = None
    run.operation_uuid = None
    run.config = {"color_by": "filetype", "n_neighbors": 15}
    run.meta = {}

    result = ProjectionService._encode_projection(run)

    # color_by should be added to meta from config
    assert result["meta"]["color_by"] == "filetype"


def test_encode_projection_empty_meta_returns_none() -> None:
    """Test that empty meta dict becomes None in response."""
    # Create mock projection run
    run = MagicMock()
    run.collection_id = "coll-123"
    run.uuid = "proj-456"
    run.status = ProjectionRunStatus.PENDING
    run.reducer = "pca"
    run.dimensionality = 2
    run.created_at = None
    run.operation_uuid = None
    run.config = None
    run.meta = {}

    result = ProjectionService._encode_projection(run)

    # Empty meta should become None
    assert result["meta"] is None


def test_encode_projection_with_completed_operation() -> None:
    """Test _encode_projection with completed operation status."""
    # Create mock projection run
    run = MagicMock()
    run.collection_id = "coll-123"
    run.uuid = "proj-456"
    run.status = ProjectionRunStatus.COMPLETED
    run.reducer = "umap"
    run.dimensionality = 2
    run.created_at = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
    run.operation_uuid = "op-789"
    run.config = None
    run.meta = None

    # Create mock completed operation
    operation = MagicMock()
    operation.uuid = "op-789"
    operation.status = OperationStatus.COMPLETED
    operation.error_message = None

    result = ProjectionService._encode_projection(run, operation=operation)

    assert result["status"] == "completed"
    assert result["operation_status"] == "completed"
    assert result["message"] is None


def test_encode_projection_includes_degraded_flag_from_projection_meta() -> None:
    """degraded=True inside projection_artifacts should surface on meta."""
    run = MagicMock()
    run.collection_id = "coll-123"
    run.uuid = "proj-456"
    run.status = ProjectionRunStatus.COMPLETED
    run.reducer = "umap"
    run.dimensionality = 2
    run.created_at = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
    run.operation_uuid = None
    run.config = {"color_by": "document_id"}
    run.meta = {
        "projection_artifacts": {
            "degraded": True,
            "color_by": "document_id",
        }
    }

    result = ProjectionService._encode_projection(run)

    assert result["meta"]["degraded"] is True
    assert result["meta"]["color_by"] == "document_id"


def test_encode_projection_includes_degraded_flag_from_run_meta() -> None:
    """degraded=True on the run meta should surface in encoded meta."""
    run = MagicMock()
    run.collection_id = "coll-123"
    run.uuid = "proj-456"
    run.status = ProjectionRunStatus.COMPLETED
    run.reducer = "pca"
    run.dimensionality = 2
    run.created_at = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
    run.operation_uuid = None
    run.config = {"color_by": "document_id"}
    run.meta = {"degraded": True, "legend": []}

    result = ProjectionService._encode_projection(run)

    assert result["meta"]["degraded"] is True
    # colour mode should still be propagated from config
    assert result["meta"]["color_by"] == "document_id"


def test_compute_projection_metadata_hash_deterministic_and_case_insensitive() -> None:
    """Hash should be stable across config key order and case-only differences."""
    updated_at = datetime(2025, 1, 1, 12, 0, 0, 123456, tzinfo=UTC)

    base_kwargs = {
        "collection_id": "c1",
        "embedding_model": "Model/Embedding",
        "collection_vector_count": 10,
        "collection_updated_at": updated_at,
        "reducer": "UMAP",
        "dimensionality": 2,
        "color_by": "DOCUMENT_ID",
        "sample_limit": 100,
    }

    hash1 = compute_projection_metadata_hash(config={"b": 2, "a": 1}, **base_kwargs)
    hash2 = compute_projection_metadata_hash(config={"a": 1, "b": 2}, **base_kwargs)

    assert hash1 == hash2


def test_compute_projection_metadata_hash_changes_with_inputs() -> None:
    """Changing key parameters should produce a different hash."""
    base_kwargs = {
        "collection_id": "c1",
        "embedding_model": "Model/Embedding",
        "collection_vector_count": 10,
        "collection_updated_at": None,
        "reducer": "umap",
        "dimensionality": 2,
        "color_by": "document_id",
    }

    hash_default = compute_projection_metadata_hash(config=None, sample_limit=None, **base_kwargs)
    hash_with_limit = compute_projection_metadata_hash(config=None, sample_limit=50, **base_kwargs)

    assert hash_default != hash_with_limit


def test_extract_sample_limit_handles_aliases_and_validation() -> None:
    config: dict[str, Any] = {"sample_size": "10", "sample_limit": 20}

    result = ProjectionService._extract_sample_limit(config)

    assert result == 10


def test_extract_sample_limit_ignores_invalid_values() -> None:
    assert ProjectionService._extract_sample_limit(None) is None
    assert ProjectionService._extract_sample_limit({"sample_size": "not-int"}) is None
    assert ProjectionService._extract_sample_limit({"sample_limit": -5}) is None


def test_is_run_degraded_checks_meta_and_projection_artifacts() -> None:
    run = MagicMock()
    run.meta = None
    assert ProjectionService._is_run_degraded(run) is False

    run.meta = {"degraded": True}
    assert ProjectionService._is_run_degraded(run) is True

    run.meta = {"projection_artifacts": {"degraded": True}}
    assert ProjectionService._is_run_degraded(run) is True

    run.meta = {"projection_artifacts": {"degraded": False}}
    assert ProjectionService._is_run_degraded(run) is False


def test_is_metadata_compatible_matches_expected_fields() -> None:
    run = MagicMock()
    run.reducer = "UMAP"
    run.dimensionality = 2
    run.config = {"color_by": "document_id", "sample_limit": 100}

    assert ProjectionService._is_metadata_compatible(
        run,
        reducer="umap",
        dimensionality=2,
        color_by="document_id",
        sample_limit=100,
    )

    # Mismatched reducer
    assert not ProjectionService._is_metadata_compatible(
        run,
        reducer="tsne",
        dimensionality=2,
        color_by="document_id",
        sample_limit=100,
    )

    # Mismatched colour mode
    assert not ProjectionService._is_metadata_compatible(
        run,
        reducer="umap",
        dimensionality=2,
        color_by="collection",
        sample_limit=100,
    )

    # Mismatched sample limit
    assert not ProjectionService._is_metadata_compatible(
        run,
        reducer="umap",
        dimensionality=2,
        color_by="document_id",
        sample_limit=50,
    )


def test_normalise_reducer_config_umap_defaults_and_validation() -> None:
    cfg = ProjectionService._normalise_reducer_config("umap", None)
    assert cfg == {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"}

    cfg_custom = ProjectionService._normalise_reducer_config(
        "umap",
        {"n_neighbors": "25", "min_dist": "0.2", "metric": "manhattan"},
    )
    assert cfg_custom == {"n_neighbors": 25, "min_dist": 0.2, "metric": "manhattan"}

    with pytest.raises(HTTPException):
        ProjectionService._normalise_reducer_config("umap", {"n_neighbors": 1})


def test_normalise_reducer_config_tsne_defaults_and_init_normalisation() -> None:
    cfg = ProjectionService._normalise_reducer_config("tsne", None)
    assert cfg["perplexity"] == 30.0
    assert cfg["learning_rate"] == 200.0
    assert cfg["n_iter"] == 1000
    assert cfg["metric"] == "euclidean"
    assert cfg["init"] == "pca"

    cfg_custom = ProjectionService._normalise_reducer_config(
        "tsne",
        {"perplexity": "10", "learning_rate": "100", "n_iter": "500", "metric": "cosine", "init": "something"},
    )
    assert cfg_custom["perplexity"] == 10.0
    assert cfg_custom["learning_rate"] == 100.0
    assert cfg_custom["n_iter"] == 500
    assert cfg_custom["metric"] == "cosine"
    # Unknown init should normalise to "pca"
    assert cfg_custom["init"] == "pca"


def test_normalise_reducer_config_other_reducer_passthrough() -> None:
    cfg_none = ProjectionService._normalise_reducer_config("pca", None)
    assert cfg_none is None

    cfg = ProjectionService._normalise_reducer_config("pca", {"whiten": True})
    assert cfg == {"whiten": True}


def test_normalise_reducer_config_rejects_non_dict_config() -> None:
    with pytest.raises(HTTPException):
        ProjectionService._normalise_reducer_config("umap", "not-a-dict")  # type: ignore[arg-type]

    with pytest.raises(HTTPException):
        ProjectionService._normalise_reducer_config("tsne", "not-a-dict")  # type: ignore[arg-type]

    with pytest.raises(HTTPException):
        ProjectionService._normalise_reducer_config("pca", "not-a-dict")  # type: ignore[arg-type]


@pytest.mark.asyncio()
async def test_start_projection_build_rejects_non_2d(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
) -> None:
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.start_projection_build(
            collection_id="coll-123",
            user_id=1,
            parameters={"reducer": "umap", "dimensionality": 3},
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio()
async def test_start_projection_build_creates_run_and_operation(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
    mock_projection_repo.find_latest_completed_by_metadata_hash.return_value = None

    run = MagicMock()
    run.uuid = "proj-123"
    run.collection_id = mock_collection.id
    run.reducer = "umap"
    run.dimensionality = 2
    run.config = {}
    run.meta = {}
    run.operation_uuid = None
    run.status = MagicMock(value="pending")
    run.created_at = None
    mock_projection_repo.create.return_value = run

    operation = MagicMock()
    operation.uuid = "op-123"
    operation.status = MagicMock(value="pending")
    operation.error_message = None
    mock_operation_repo.create.return_value = operation

    send_task_mock = MagicMock()
    monkeypatch.setattr(projection_module.celery_app, "send_task", send_task_mock)

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    result = await service.start_projection_build(
        collection_id=mock_collection.id,
        user_id=1,
        parameters={"reducer": "umap", "dimensionality": 2, "color_by": "document_id", "sample_size": "10"},
    )

    mock_projection_repo.create.assert_awaited_once()
    mock_operation_repo.create.assert_awaited_once()
    send_task_mock.assert_called_once()
    assert result["projection_id"] == "proj-123"
    assert result["operation_id"] == "op-123"
    assert "idempotent_reuse" not in result


@pytest.mark.asyncio()
async def test_start_projection_build_reuses_existing_run(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    existing_run = MagicMock()
    existing_run.uuid = "proj-existing"
    existing_run.collection_id = mock_collection.id
    existing_run.reducer = "umap"
    existing_run.dimensionality = 2
    existing_run.config = {"color_by": "document_id", "sample_limit": 10}
    existing_run.meta = {}
    existing_run.status = ProjectionRunStatus.COMPLETED
    existing_run.operation = MagicMock()
    existing_run.operation.status = OperationStatus.COMPLETED
    existing_run.operation.error_message = None

    mock_projection_repo.find_latest_completed_by_metadata_hash.return_value = existing_run

    send_task_mock = MagicMock()
    monkeypatch.setattr(projection_module.celery_app, "send_task", send_task_mock)

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    result = await service.start_projection_build(
        collection_id=mock_collection.id,
        user_id=1,
        parameters={
            "reducer": "umap",
            "dimensionality": 2,
            "color_by": "document_id",
            "sample_limit": 10,
        },
    )

    mock_projection_repo.create.assert_not_awaited()
    mock_operation_repo.create.assert_not_awaited()
    send_task_mock.assert_not_called()
    assert result["projection_id"] == "proj-existing"
    assert result["idempotent_reuse"] is True


@pytest.mark.asyncio()
async def test_start_projection_build_handles_celery_failure(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
    mock_projection_repo.find_latest_completed_by_metadata_hash.return_value = None

    run = MagicMock()
    run.uuid = "proj-123"
    run.collection_id = mock_collection.id
    run.reducer = "umap"
    run.dimensionality = 2
    run.config = {}
    run.meta = {}
    run.operation_uuid = None
    run.status = MagicMock(value="pending")
    run.created_at = None
    mock_projection_repo.create.return_value = run

    operation = MagicMock()
    operation.uuid = "op-123"
    operation.status = MagicMock(value="pending")
    operation.error_message = None
    mock_operation_repo.create.return_value = operation

    async def update_status_side_effect(*_: Any, **__: Any) -> None:
        return None

    mock_operation_repo.update_status = AsyncMock(side_effect=update_status_side_effect)
    mock_projection_repo.update_status = AsyncMock(side_effect=update_status_side_effect)

    def failing_send_task(*_: Any, **__: Any) -> None:
        raise RuntimeError("broker failure")

    monkeypatch.setattr(projection_module.celery_app, "send_task", failing_send_task)

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.start_projection_build(
            collection_id=mock_collection.id,
            user_id=1,
            parameters={"reducer": "umap", "dimensionality": 2},
        )

    assert exc_info.value.status_code == 503
    mock_operation_repo.update_status.assert_awaited_once()
    mock_projection_repo.update_status.assert_awaited_once()


@pytest.mark.asyncio()
async def test_list_projections_encodes_runs(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
) -> None:
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    run = MagicMock()
    run.collection_id = mock_collection.id
    run.uuid = "proj-1"
    run.status = MagicMock(value="completed")
    run.reducer = "umap"
    run.dimensionality = 2
    run.created_at = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
    run.operation_uuid = "op-1"
    run.config = {}
    run.meta = {}
    run.operation = MagicMock()
    run.operation.status = OperationStatus.COMPLETED
    run.operation.error_message = None

    mock_projection_repo.list_for_collection.return_value = ([run], 1)

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    results = await service.list_projections(collection_id=mock_collection.id, user_id=1)

    assert len(results) == 1
    assert results[0]["projection_id"] == "proj-1"
    assert results[0]["operation_status"] == "completed"


@pytest.mark.asyncio()
async def test_get_projection_metadata_enforces_owner_permissions(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
) -> None:
    mock_collection.owner_id = 2
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    with pytest.raises(AccessDeniedError):
        await service.get_projection_metadata(collection_id="coll-123", projection_id="proj-1", user_id=1)


@pytest.mark.asyncio()
async def test_get_projection_metadata_returns_encoded_projection(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
) -> None:
    mock_collection.owner_id = 1
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    run = MagicMock()
    run.collection_id = mock_collection.id
    run.uuid = "proj-1"
    run.status = MagicMock(value="completed")
    run.reducer = "umap"
    run.dimensionality = 2
    run.created_at = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
    run.operation_uuid = "op-1"
    run.config = {}
    run.meta = {}
    mock_projection_repo.get_by_uuid.return_value = run

    operation = MagicMock()
    operation.uuid = "op-1"
    operation.status = OperationStatus.COMPLETED
    operation.error_message = None
    mock_operation_repo.get_by_uuid.return_value = operation

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    result = await service.get_projection_metadata(
        collection_id=mock_collection.id,
        projection_id="proj-1",
        user_id=1,
    )

    assert result["projection_id"] == "proj-1"
    assert result["operation_status"] == "completed"


@pytest.mark.asyncio()
async def test_resolve_storage_directory_normalises_relative_paths(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
) -> None:
    data_dir = projection_module.settings.data_dir
    artifacts_dir = data_dir / "test_projection_run1"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run = MagicMock()
    run.storage_path = "./run1"

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    resolved = await service._resolve_storage_directory(run, "./test_projection_run1")

    assert resolved == artifacts_dir
    assert run.storage_path == "test_projection_run1"
    mock_db_session.flush.assert_awaited()


@pytest.mark.asyncio()
async def test_resolve_artifact_path_returns_existing_file(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
) -> None:
    data_dir = projection_module.settings.data_dir

    artifacts_dir = data_dir / "test_projection_artifacts1"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_file = artifacts_dir / "x.f32.bin"
    artifact_file.write_bytes(b"\x00" * 4)

    run = MagicMock()
    run.uuid = "proj-1"
    run.collection_id = "coll-1"
    run.storage_path = "./test_projection_artifacts1"
    mock_projection_repo.get_by_uuid.return_value = run
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    resolved = await service.resolve_artifact_path(
        collection_id="coll-1",
        projection_id="proj-1",
        artifact_name="x",
        user_id=1,
    )

    assert resolved == artifact_file


@pytest.mark.asyncio()
async def test_resolve_artifact_path_marks_degraded_when_directory_missing(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
) -> None:
    run = MagicMock()
    run.uuid = "proj-1"
    run.collection_id = "coll-1"
    run.storage_path = "./missing_artifacts_dir"
    mock_projection_repo.get_by_uuid.return_value = run
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
    mock_projection_repo.update_metadata = AsyncMock()

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    with pytest.raises(FileNotFoundError):
        await service.resolve_artifact_path(
            collection_id="coll-1",
            projection_id="proj-1",
            artifact_name="x",
            user_id=1,
        )

    mock_projection_repo.update_metadata.assert_awaited_once()


@pytest.mark.asyncio()
async def test_select_projection_region_with_valid_ids_returns_items(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
) -> None:
    data_dir = projection_module.settings.data_dir

    artifacts_dir = data_dir / "test_projection_selection1"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ids_file = artifacts_dir / "ids.i32.bin"
    ids_file.write_bytes((5).to_bytes(4, byteorder="little", signed=True))

    run = MagicMock()
    run.uuid = "proj-1"
    run.collection_id = "coll-1"
    run.storage_path = "./test_projection_selection1"
    run.meta = {"projection_artifacts": {"original_ids": ["5"]}}
    mock_projection_repo.get_by_uuid.return_value = run
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    # Avoid hitting real Qdrant by omitting source_vector_collection
    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    result = await service.select_projection_region(
        collection_id="coll-1",
        projection_id="proj-1",
        selection={"ids": [5]},
        user_id=1,
    )

    assert result["missing_ids"] == []
    assert len(result["items"]) == 1
    assert result["items"][0]["selected_id"] == 5


@pytest.mark.asyncio()
async def test_select_projection_region_rejects_invalid_ids() -> None:
    service = ProjectionService(
        db_session=AsyncMock(),
        projection_repo=AsyncMock(),
        operation_repo=AsyncMock(),
        collection_repo=AsyncMock(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.select_projection_region(
            collection_id="coll-1",
            projection_id="proj-1",
            selection={"ids": "not-a-list"},
            user_id=1,
        )

    assert exc_info.value.status_code == 400
