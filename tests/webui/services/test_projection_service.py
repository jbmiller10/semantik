"""Unit tests for ProjectionService._encode_projection method."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from packages.shared.database.models import OperationStatus, ProjectionRunStatus
from packages.webui.services.projection_service import ProjectionService


def test_encode_projection_without_operation():
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


def test_encode_projection_with_operation():
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


def test_encode_projection_with_failed_operation():
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


def test_encode_projection_custom_message_overrides_error():
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


def test_encode_projection_color_by_in_config():
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


def test_encode_projection_empty_meta_returns_none():
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


def test_encode_projection_with_completed_operation():
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


def test_encode_projection_includes_degraded_flag_from_projection_meta():
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


def test_encode_projection_includes_degraded_flag_from_run_meta():
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
