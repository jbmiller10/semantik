"""Unit tests for ProjectionService and helpers."""

import json
from array import array
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from shared.database.models import OperationStatus, ProjectionRunStatus

from webui.services import projection_service as projection_module
from webui.services.projection_service import (
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

    # Mismatched dimensionality
    assert not ProjectionService._is_metadata_compatible(
        run,
        reducer="umap",
        dimensionality=3,
        color_by="document_id",
        sample_limit=100,
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


@pytest.mark.parametrize(
    ("config", "expected_detail"),
    [
        ({"n_neighbors": "oops"}, "n_neighbors must be an integer"),
        ({"n_neighbors": 15, "min_dist": "oops"}, "min_dist must be a number"),
        ({"n_neighbors": 15, "min_dist": 2}, "min_dist must be between 0 and 1"),
        ({"n_neighbors": 15, "min_dist": 0.1, "metric": ""}, "metric must be a non-empty string"),
    ],
)
def test_normalise_reducer_config_umap_invalid_inputs(config: dict[str, Any], expected_detail: str) -> None:
    with pytest.raises(HTTPException) as exc_info:
        ProjectionService._normalise_reducer_config("umap", config)

    assert exc_info.value.detail == expected_detail


@pytest.mark.parametrize(
    ("config", "expected_detail"),
    [
        ({"perplexity": "oops"}, "perplexity must be a number"),
        ({"perplexity": 0}, "perplexity must be > 0"),
        ({"perplexity": 10, "learning_rate": "oops"}, "learning_rate must be a number"),
        ({"perplexity": 10, "learning_rate": 0}, "learning_rate must be > 0"),
        ({"perplexity": 10, "learning_rate": 100, "n_iter": "oops"}, "n_iter must be an integer"),
        ({"perplexity": 10, "learning_rate": 100, "n_iter": 100}, "n_iter must be >= 250"),
        ({"perplexity": 10, "learning_rate": 100, "n_iter": 300, "metric": ""}, "metric must be a non-empty string"),
    ],
)
def test_normalise_reducer_config_tsne_invalid_inputs(config: dict[str, Any], expected_detail: str) -> None:
    with pytest.raises(HTTPException) as exc_info:
        ProjectionService._normalise_reducer_config("tsne", config)

    assert exc_info.value.detail == expected_detail


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
@pytest.mark.parametrize(
    ("param_name", "param_value", "expected_detail"),
    [
        ("sample_size", "oops", "sample_size must be an integer"),
        ("sample_n", 0, "sample_n must be > 0"),
    ],
)
async def test_start_projection_build_validates_sampling_aliases(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
    param_name: str,
    param_value: Any,
    expected_detail: str,
) -> None:
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    parameters = {"reducer": "umap", "dimensionality": 2, param_name: param_value}

    with pytest.raises(HTTPException) as exc_info:
        await service.start_projection_build(
            collection_id=mock_collection.id,
            user_id=1,
            parameters=parameters,
        )

    assert exc_info.value.detail == expected_detail


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
async def test_start_projection_build_ignores_mismatched_metadata_hash(
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
    run.uuid = "proj-hash"
    run.collection_id = mock_collection.id
    run.reducer = "umap"
    run.dimensionality = 2
    run.config = {}
    run.meta = {}
    run.operation_uuid = None
    run.status = ProjectionRunStatus.PENDING
    mock_projection_repo.create.return_value = run

    operation = MagicMock()
    operation.uuid = "op-hash"
    operation.status = OperationStatus.PENDING
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
        parameters={"reducer": "umap", "dimensionality": 2, "metadata_hash": "client-supplied"},
    )

    create_kwargs = mock_projection_repo.create.await_args.kwargs
    expected_hash = compute_projection_metadata_hash(
        collection_id=mock_collection.id,
        embedding_model=getattr(mock_collection, "embedding_model", ""),
        collection_vector_count=getattr(mock_collection, "vector_count", 0),
        collection_updated_at=getattr(mock_collection, "updated_at", None),
        reducer="umap",
        dimensionality=2,
        color_by="document_id",
        config=create_kwargs["config"],
        sample_limit=None,
    )

    assert create_kwargs["metadata_hash"] == expected_hash
    assert result["projection_id"] == "proj-hash"
    send_task_mock.assert_called_once()


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
async def test_resolve_storage_directory_handles_absolute_projection_suffix(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(projection_module.settings, "DATA_DIR", tmp_path)
    artifacts_dir = tmp_path / "semantik" / "projections" / "abs-run"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run = MagicMock()
    run.storage_path = str(artifacts_dir)

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    resolved = await service._resolve_storage_directory(run, str(artifacts_dir))

    assert resolved == artifacts_dir
    assert run.storage_path == "semantik/projections/abs-run"
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
async def test_resolve_artifact_path_rejects_unknown_artifact() -> None:
    service = ProjectionService(
        db_session=AsyncMock(),
        projection_repo=AsyncMock(),
        operation_repo=AsyncMock(),
        collection_repo=AsyncMock(),
    )

    with pytest.raises(ValueError, match="Unsupported projection artifact 'unknown'"):
        await service.resolve_artifact_path("c", "p", "unknown", 1)


@pytest.mark.asyncio()
async def test_resolve_artifact_path_requires_matching_projection(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
) -> None:
    mock_projection_repo.get_by_uuid.return_value = None

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    with pytest.raises(EntityNotFoundError):
        await service.resolve_artifact_path("c", "p", "x", 1)


@pytest.mark.asyncio()
async def test_resolve_artifact_path_requires_storage_path(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
) -> None:
    run = MagicMock()
    run.uuid = "proj-1"
    run.collection_id = "coll-1"
    run.storage_path = None
    mock_projection_repo.get_by_uuid.return_value = run
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.resolve_artifact_path("coll-1", "proj-1", "x", 1)

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio()
async def test_resolve_artifact_path_blocks_symlink_escape(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(projection_module.settings, "DATA_DIR", tmp_path)
    artifacts_dir = tmp_path / "symlink_proj"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    outside_file = tmp_path / "outside.bin"
    outside_file.write_bytes(b"forbidden")
    symlink_path = artifacts_dir / "x.f32.bin"
    symlink_path.symlink_to(outside_file)

    run = MagicMock()
    run.uuid = "proj-1"
    run.collection_id = "coll-1"
    run.storage_path = "symlink_proj"
    mock_projection_repo.get_by_uuid.return_value = run
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    with pytest.raises(PermissionError):
        await service.resolve_artifact_path("coll-1", "proj-1", "x", 1)


@pytest.mark.asyncio()
async def test_resolve_artifact_path_marks_degraded_when_file_missing(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(projection_module.settings, "DATA_DIR", tmp_path)
    artifacts_dir = tmp_path / "missing_file_proj"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run = MagicMock()
    run.uuid = "proj-1"
    run.collection_id = "coll-1"
    run.storage_path = "missing_file_proj"
    mock_projection_repo.get_by_uuid.return_value = run
    mock_projection_repo.update_metadata = AsyncMock()
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    with pytest.raises(FileNotFoundError):
        await service.resolve_artifact_path("coll-1", "proj-1", "x", 1)

    mock_projection_repo.update_metadata.assert_awaited_once()


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


@pytest.mark.asyncio()
async def test_select_projection_region_requires_integer_ids() -> None:
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
            selection={"ids": [1, "abc"]},
            user_id=1,
        )

    assert exc_info.value.detail == "ids must be integers"


@pytest.mark.asyncio()
async def test_select_projection_region_enforces_max_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    service = ProjectionService(
        db_session=AsyncMock(),
        projection_repo=AsyncMock(),
        operation_repo=AsyncMock(),
        collection_repo=AsyncMock(),
    )

    monkeypatch.setattr(ProjectionService, "_MAX_SELECTION_IDS", 1)

    with pytest.raises(HTTPException) as exc_info:
        await service.select_projection_region(
            collection_id="coll-1",
            projection_id="proj-1",
            selection={"ids": [1, 2]},
            user_id=1,
        )

    assert exc_info.value.status_code == 413


@pytest.mark.asyncio()
async def test_select_projection_region_requires_existing_projection(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
) -> None:
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
    mock_projection_repo.get_by_uuid.return_value = None

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    with pytest.raises(EntityNotFoundError):
        await service.select_projection_region(
            collection_id=mock_collection.id,
            projection_id="missing",
            selection={"ids": [1]},
            user_id=1,
        )


@pytest.mark.asyncio()
async def test_select_projection_region_requires_artifacts_path(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
) -> None:
    run = MagicMock()
    run.collection_id = mock_collection.id
    run.storage_path = None
    mock_projection_repo.get_by_uuid.return_value = run
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    with pytest.raises(FileNotFoundError):
        await service.select_projection_region(
            collection_id=mock_collection.id,
            projection_id="proj-1",
            selection={"ids": [1]},
            user_id=1,
        )


@pytest.mark.asyncio()
async def test_select_projection_region_marks_degraded_when_directory_missing(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
) -> None:
    run = MagicMock()
    run.collection_id = mock_collection.id
    run.storage_path = "proj-missing"
    mock_projection_repo.get_by_uuid.return_value = run
    mock_projection_repo.update_metadata = AsyncMock()
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )
    service._resolve_storage_directory = AsyncMock(side_effect=FileNotFoundError("missing"))

    with pytest.raises(HTTPException) as exc_info:
        await service.select_projection_region(
            collection_id=mock_collection.id,
            projection_id="proj-1",
            selection={"ids": [1]},
            user_id=1,
        )

    assert exc_info.value.status_code == 404
    mock_projection_repo.update_metadata.assert_awaited_once()


@pytest.mark.asyncio()
async def test_select_projection_region_marks_degraded_when_ids_file_missing(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(projection_module.settings, "DATA_DIR", tmp_path)
    artifacts_dir = tmp_path / "missing_ids"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run = MagicMock()
    run.collection_id = mock_collection.id
    run.storage_path = "missing_ids"
    run.meta = {"projection_artifacts": {"original_ids": ["1"]}}
    mock_projection_repo.get_by_uuid.return_value = run
    mock_projection_repo.update_metadata = AsyncMock()
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.select_projection_region(
            collection_id=mock_collection.id,
            projection_id="proj-ids",
            selection={"ids": [1]},
            user_id=1,
        )

    assert exc_info.value.detail == "Projection ids artifact is missing"
    mock_projection_repo.update_metadata.assert_awaited_once()


@pytest.mark.asyncio()
async def test_select_projection_region_handles_invalid_meta_json(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(projection_module.settings, "DATA_DIR", tmp_path)
    artifacts_dir = tmp_path / "invalid_meta"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ids_file = artifacts_dir / "ids.i32.bin"
    ids_file.write_bytes(array("i", [5]).tobytes())
    (artifacts_dir / "meta.json").write_text("{not json")

    run = MagicMock()
    run.collection_id = mock_collection.id
    run.storage_path = "invalid_meta"
    run.meta = {"projection_artifacts": {"original_ids": ["5"]}}
    mock_projection_repo.get_by_uuid.return_value = run
    mock_projection_repo.update_metadata = AsyncMock()
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    result = await service.select_projection_region(
        collection_id=mock_collection.id,
        projection_id="proj-invalid-meta",
        selection={"ids": [5]},
        user_id=1,
    )

    assert result["items"][0]["selected_id"] == 5
    mock_projection_repo.update_metadata.assert_awaited_once()


@pytest.mark.asyncio()
async def test_select_projection_region_enriches_chunk_and_qdrant_metadata(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(projection_module.settings, "DATA_DIR", tmp_path)
    artifacts_dir = tmp_path / "projection_enrich"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ids_file = artifacts_dir / "ids.i32.bin"
    ids_file.write_bytes(array("i", [5, 6, 7, 8, 9]).tobytes())
    meta_payload = {
        "original_ids": ["10", "vector-1", "not-int", 77],
        "source_vector_collection": "collection-q",
        "degraded": True,
    }
    (artifacts_dir / "meta.json").write_text(json.dumps(meta_payload))

    run = MagicMock()
    run.collection_id = mock_collection.id
    run.storage_path = "projection_enrich"
    run.meta = {}
    mock_projection_repo.get_by_uuid.return_value = run
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    class FakeChunk:
        def __init__(self, chunk_id: int, document_id: str, content: str, chunk_index: int = 0) -> None:
            self.id = chunk_id
            self.document_id = document_id
            self.content = content
            self.chunk_index = chunk_index

    class FakeChunkRepository:
        def __init__(self, session: Any) -> None:
            self.session = session

        async def get_chunk_by_id(self, chunk_id: int, collection_id: str) -> FakeChunk | None:
            if chunk_id == 10:
                return FakeChunk(10, "doc-10", "A" * 300)
            if chunk_id == 77:
                return FakeChunk(77, "doc-77", "chunk77", chunk_index=5)
            raise Exception("missing")

        async def get_chunk_by_embedding_vector_id(self, vector_id: str, collection_id: str) -> FakeChunk | None:
            raise Exception("missing")

    class FakeDocument:
        def __init__(self, doc_id: str) -> None:
            self.id = doc_id
            self.file_name = f"{doc_id}.txt"
            self.source_id = f"src-{doc_id}"
            self.mime_type = "text/plain"

    class FakeDocumentRepository:
        def __init__(self, session: Any) -> None:
            self.session = session

        async def get_by_id(self, doc_id: str) -> FakeDocument | None:
            if str(doc_id) in {"doc-10", "doc-77", "doc-qdrant"}:
                return FakeDocument(str(doc_id))
            return None

    class FakeQdrantClient:
        def retrieve(self, collection_name: str, ids: list[str], with_payload: bool) -> list[SimpleNamespace]:
            if ids == ["vector-1"]:
                chunk_id_value = "chunk-" + "\\" + "dddd"
                payload = {
                    "doc_id": "doc-qdrant",
                    "chunk_id": chunk_id_value,
                    "content": "payload-text" * 20,
                }
                return [SimpleNamespace(payload=payload)]
            return []

    monkeypatch.setattr(projection_module, "ChunkRepository", FakeChunkRepository)
    monkeypatch.setattr(projection_module, "DocumentRepository", FakeDocumentRepository)
    monkeypatch.setattr(
        projection_module.qdrant_manager,
        "get_client",
        MagicMock(return_value=FakeQdrantClient()),
    )

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    result = await service.select_projection_region(
        collection_id=mock_collection.id,
        projection_id="proj-rich",
        selection={"ids": [5, 6, 7, 8, 9, 42]},
        user_id=1,
    )

    assert result["degraded"] is True
    assert result["missing_ids"] == [42]
    assert len(result["items"]) == 5

    first, second, third, fourth, fifth = result["items"]
    assert first["chunk_id"] == 10
    assert first["document"]["document_id"] == "doc-10"
    assert second["document"]["document_id"] == "doc-qdrant"
    assert second["chunk_index"] is None
    assert third["document_id"] is None
    assert fourth["chunk_id"] == 77
    assert fifth["original_id"] is None


@pytest.mark.asyncio()
async def test_delete_projection_rejects_pending_run(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
) -> None:
    run = MagicMock()
    run.collection_id = mock_collection.id
    run.status = ProjectionRunStatus.PENDING
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
    mock_projection_repo.get_by_uuid.return_value = run

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.delete_projection(mock_collection.id, "proj", 1)

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio()
async def test_delete_projection_removes_artifacts_and_commits(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(projection_module.settings, "DATA_DIR", tmp_path)
    artifacts_dir = tmp_path / "proj_delete"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "dummy.bin").write_bytes(b"x")

    run = MagicMock()
    run.uuid = "proj-delete"
    run.collection_id = mock_collection.id
    run.storage_path = "proj_delete"
    run.status = ProjectionRunStatus.COMPLETED
    mock_projection_repo.get_by_uuid.return_value = run
    mock_projection_repo.delete = AsyncMock()
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    await service.delete_projection(mock_collection.id, "proj-delete", 1)

    assert not artifacts_dir.exists()
    mock_projection_repo.delete.assert_awaited_once_with("proj-delete")
    mock_db_session.commit.assert_awaited_once()


@pytest.mark.asyncio()
async def test_delete_projection_handles_missing_artifacts_directory(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
) -> None:
    run = MagicMock()
    run.uuid = "proj-delete"
    run.collection_id = mock_collection.id
    run.storage_path = "missing"
    run.status = ProjectionRunStatus.COMPLETED
    mock_projection_repo.get_by_uuid.return_value = run
    mock_projection_repo.delete = AsyncMock()
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )
    service._resolve_storage_directory = AsyncMock(side_effect=FileNotFoundError("missing"))

    await service.delete_projection(mock_collection.id, "proj-delete", 1)

    mock_projection_repo.delete.assert_awaited_once_with("proj-delete")
    mock_db_session.commit.assert_awaited_once()


@pytest.mark.asyncio()
async def test_delete_projection_ignores_rmtree_file_not_found(
    mock_db_session: AsyncMock,
    mock_projection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_collection: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(projection_module.settings, "DATA_DIR", tmp_path)
    artifacts_dir = tmp_path / "proj_delete_missing"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run = MagicMock()
    run.uuid = "proj-delete-missing"
    run.collection_id = mock_collection.id
    run.storage_path = "proj_delete_missing"
    run.status = ProjectionRunStatus.COMPLETED
    mock_projection_repo.get_by_uuid.return_value = run
    mock_projection_repo.delete = AsyncMock()
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

    monkeypatch.setattr(projection_module.shutil, "rmtree", MagicMock(side_effect=FileNotFoundError()))

    service = ProjectionService(
        db_session=mock_db_session,
        projection_repo=mock_projection_repo,
        operation_repo=mock_operation_repo,
        collection_repo=mock_collection_repo,
    )

    await service.delete_projection(mock_collection.id, "proj-delete-missing", 1)

    mock_projection_repo.delete.assert_awaited_once_with("proj-delete-missing")
    mock_db_session.commit.assert_awaited_once()
