"""Unit tests for Collections API v2 endpoints.

These tests mock the CollectionService to test error handling and
HTTP status code mappings in the API layer.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared.database.exceptions import (
    AccessDeniedError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidStateError,
    ValidationError,
)
from shared.database.models import Collection, CollectionStatus, Operation, OperationStatus, OperationType
from webui.api.v2.collections import router

# Create test app
app = FastAPI()
app.include_router(router)


@pytest.fixture()
def mock_current_user():
    """Mock current user dependency."""
    return {"id": "1", "username": "testuser"}


@pytest.fixture()
def mock_service():
    """Mock CollectionService."""
    return AsyncMock()


@pytest.fixture()
def mock_collection():
    """Create a mock Collection with all required attributes."""
    # Use MagicMock with spec to prevent auto-creating attributes
    mock = MagicMock()
    # Set all required attributes explicitly
    mock.id = str(uuid4())
    mock.name = "Test Collection"
    mock.description = "Test description"
    mock.owner_id = 1
    mock.vector_store_name = "test_collection"
    mock.embedding_model = "test-model"
    mock.quantization = "float16"
    mock.chunk_size = 512
    mock.chunk_overlap = 64
    mock.chunking_strategy = "recursive"
    mock.chunking_config = {}
    mock.is_public = False
    mock.meta = {}
    mock.document_count = 0
    mock.vector_count = 0
    mock.total_size_bytes = 0
    mock.status = CollectionStatus.READY
    mock.status_message = None
    mock.sync_mode = "one_time"
    mock.sync_interval_minutes = None
    mock.sync_paused_at = None
    mock.sync_next_run_at = None
    mock.sync_last_run_started_at = None
    mock.sync_last_run_completed_at = None
    mock.sync_last_run_status = None
    mock.sync_last_error = None
    mock.created_at = datetime.now(UTC)
    mock.updated_at = datetime.now(UTC)
    return mock


@pytest.fixture()
def mock_operation():
    """Create a mock Operation."""
    return MagicMock(
        id=1,
        uuid=str(uuid4()),
        collection_id=str(uuid4()),
        type=OperationType.APPEND,
        status=OperationStatus.PENDING,
        config={},
        created_at=datetime.now(UTC),
        started_at=None,
        completed_at=None,
        error_message=None,
    )


@pytest.fixture()
def mock_sync_run():
    """Create a mock CollectionSyncRun."""
    return MagicMock(
        id=1,
        collection_id=str(uuid4()),
        triggered_by="manual",
        started_at=datetime.now(UTC),
        completed_at=None,
        status="running",
        expected_sources=2,
        completed_sources=0,
        failed_sources=0,
        partial_sources=0,
        error_summary=None,
    )


@pytest.fixture()
def test_client(mock_current_user, mock_service):
    """Create test client with mocked dependencies."""
    from webui.api.v2 import collections
    from webui.auth import get_current_user
    from webui.services.factory import get_collection_service

    # Override dependencies
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    app.dependency_overrides[get_collection_service] = lambda: mock_service

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


# =============================================================================
# POST /api/v2/collections - Create Collection
# =============================================================================


class TestCreateCollectionEndpoint:
    """Tests for POST /api/v2/collections."""

    def test_create_collection_passes_sync_policy_to_service(self, test_client, mock_service):
        """Sync policy fields should be forwarded to CollectionService on create."""
        now = datetime.now(UTC)
        operation_uuid = str(uuid4())

        mock_service.create_collection.return_value = (
            {
                "id": str(uuid4()),
                "name": "Test Collection",
                "description": None,
                "owner_id": 1,
                "vector_store_name": "col_test",
                "embedding_model": "test-model",
                "quantization": "float16",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "chunking_strategy": None,
                "chunking_config": None,
                "is_public": False,
                "metadata": {},
                "created_at": now,
                "updated_at": now,
                "document_count": 0,
                "vector_count": 0,
                "total_size_bytes": 0,
                "status": "pending",
                "status_message": None,
                "sync_mode": "continuous",
                "sync_interval_minutes": 15,
                "sync_paused_at": None,
                "sync_next_run_at": None,
                "sync_last_run_started_at": None,
                "sync_last_run_completed_at": None,
                "sync_last_run_status": None,
                "sync_last_error": None,
            },
            {"uuid": operation_uuid},
        )

        response = test_client.post(
            "/api/v2/collections",
            json={
                "name": "Test Collection",
                "sync_mode": "continuous",
                "sync_interval_minutes": 15,
            },
        )

        assert response.status_code == 201

        _args, kwargs = mock_service.create_collection.call_args
        assert kwargs["config"]["sync_mode"] == "continuous"
        assert kwargs["config"]["sync_interval_minutes"] == 15

    def test_create_collection_duplicate_name_returns_409(self, test_client, mock_service):
        """Return 409 when collection name already exists."""
        mock_service.create_collection.side_effect = EntityAlreadyExistsError(
            "collection", "Test Collection"
        )

        response = test_client.post(
            "/api/v2/collections",
            json={"name": "Test Collection", "description": "Test"},
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_create_collection_invalid_data_returns_400(self, test_client, mock_service):
        """Return 400 when request data is invalid."""
        mock_service.create_collection.side_effect = ValueError("Invalid embedding model")

        response = test_client.post(
            "/api/v2/collections",
            json={"name": "Test Collection", "embedding_model": "invalid-model"},
        )

        assert response.status_code == 400
        assert "Invalid embedding model" in response.json()["detail"]


# =============================================================================
# DELETE /api/v2/collections/{id} - Delete Collection
# =============================================================================


class TestDeleteCollectionEndpoint:
    """Tests for DELETE /api/v2/collections/{collection_uuid}."""

    def test_delete_collection_active_operation_returns_409(self, test_client, mock_service):
        """Return 409 when collection has active operation."""
        mock_service.delete_collection.side_effect = InvalidStateError(
            "Cannot delete collection with active operations"
        )

        response = test_client.delete(f"/api/v2/collections/{uuid4()}")

        assert response.status_code == 409
        assert "active operations" in response.json()["detail"]


# =============================================================================
# POST /api/v2/collections/{id}/sources - Add Source
# =============================================================================


class TestAddSourceEndpoint:
    """Tests for POST /api/v2/collections/{collection_uuid}/sources."""

    def test_add_source_invalid_state_returns_409(self, test_client, mock_service):
        """Return 409 when collection is in invalid state."""
        mock_service.add_source.side_effect = InvalidStateError(
            "Collection has active operation"
        )

        response = test_client.post(
            f"/api/v2/collections/{uuid4()}/sources",
            json={"source_type": "directory", "source_config": {"path": "/data"}},
        )

        assert response.status_code == 409
        assert "active operation" in response.json()["detail"]

    def test_add_source_validation_error_returns_400(self, test_client, mock_service):
        """Return 400 when source config is invalid."""
        mock_service.add_source.side_effect = ValidationError("Invalid path", "source_config")

        response = test_client.post(
            f"/api/v2/collections/{uuid4()}/sources",
            json={"source_type": "directory", "source_config": {"path": ""}},
        )

        assert response.status_code == 400


# =============================================================================
# DELETE /api/v2/collections/{id}/sources - Remove Source
# =============================================================================


class TestRemoveSourceEndpoint:
    """Tests for DELETE /api/v2/collections/{collection_uuid}/sources."""

    def test_remove_source_success(self, test_client, mock_service, mock_operation):
        """Successfully remove a source."""
        mock_service.remove_source.return_value = {
            "uuid": mock_operation.uuid,
            "collection_id": mock_operation.collection_id,
            "type": "remove_source",
            "status": "pending",
            "config": {},
            "created_at": datetime.now(UTC),
        }

        response = test_client.delete(
            f"/api/v2/collections/{uuid4()}/sources",
            params={"source_path": "/data/test"},
        )

        assert response.status_code == 202
        assert response.json()["type"] == "remove_source"

    def test_remove_source_invalid_state_returns_409(self, test_client, mock_service):
        """Return 409 when collection has active operation."""
        mock_service.remove_source.side_effect = InvalidStateError(
            "Collection has active operation"
        )

        response = test_client.delete(
            f"/api/v2/collections/{uuid4()}/sources",
            params={"source_path": "/data/test"},
        )

        assert response.status_code == 409


# =============================================================================
# POST /api/v2/collections/{id}/reindex - Reindex Collection
# =============================================================================


class TestReindexCollectionEndpoint:
    """Tests for POST /api/v2/collections/{collection_uuid}/reindex."""

    def test_reindex_invalid_state_returns_409(self, test_client, mock_service):
        """Return 409 when collection is in invalid state."""
        mock_service.reindex_collection.side_effect = InvalidStateError(
            "Cannot reindex while operation in progress"
        )

        response = test_client.post(f"/api/v2/collections/{uuid4()}/reindex")

        assert response.status_code == 409


# =============================================================================
# GET /api/v2/collections/{id}/operations - List Operations
# =============================================================================


class TestListOperationsEndpoint:
    """Tests for GET /api/v2/collections/{collection_uuid}/operations."""

    def test_list_operations_success(self, test_client, mock_service, mock_operation):
        """Successfully list operations."""
        mock_service.list_operations_filtered.return_value = ([mock_operation], 1)

        response = test_client.get(f"/api/v2/collections/{uuid4()}/operations")

        assert response.status_code == 200
        assert len(response.json()) == 1

    def test_list_operations_not_found_returns_404(self, test_client, mock_service):
        """Return 404 when collection not found."""
        mock_service.list_operations_filtered.side_effect = EntityNotFoundError(
            "collection", "fake-uuid"
        )

        response = test_client.get(f"/api/v2/collections/{uuid4()}/operations")

        assert response.status_code == 404

    def test_list_operations_forbidden_returns_403(self, test_client, mock_service):
        """Return 403 when user doesn't have access."""
        mock_service.list_operations_filtered.side_effect = AccessDeniedError(
            "1", "collection", "fake-uuid"
        )

        response = test_client.get(f"/api/v2/collections/{uuid4()}/operations")

        assert response.status_code == 403

    def test_list_operations_invalid_filter_returns_400(self, test_client, mock_service):
        """Return 400 when filter is invalid."""
        mock_service.list_operations_filtered.side_effect = ValueError(
            "Invalid status filter: invalid"
        )

        response = test_client.get(
            f"/api/v2/collections/{uuid4()}/operations",
            params={"status": "invalid"},
        )

        assert response.status_code == 400
        assert "Invalid status filter" in response.json()["detail"]


# =============================================================================
# GET /api/v2/collections/{id}/documents - List Documents
# =============================================================================


class TestListDocumentsEndpoint:
    """Tests for GET /api/v2/collections/{collection_uuid}/documents."""

    def test_list_documents_invalid_filter_returns_400(self, test_client, mock_service):
        """Return 400 when filter is invalid."""
        mock_service.list_documents_filtered.side_effect = ValueError(
            "Invalid status filter: invalid"
        )

        response = test_client.get(
            f"/api/v2/collections/{uuid4()}/documents",
            params={"status": "invalid"},
        )

        assert response.status_code == 400


# =============================================================================
# POST /api/v2/collections/{id}/sync/run - Run Sync
# =============================================================================


class TestRunSyncEndpoint:
    """Tests for POST /api/v2/collections/{collection_uuid}/sync/run."""

    def test_run_sync_success(self, test_client, mock_service, mock_sync_run):
        """Successfully trigger sync run."""
        mock_service.run_collection_sync.return_value = mock_sync_run

        response = test_client.post(f"/api/v2/collections/{uuid4()}/sync/run")

        assert response.status_code == 202
        assert response.json()["status"] == "running"

    def test_run_sync_invalid_state_returns_409(self, test_client, mock_service):
        """Return 409 when collection has active operation."""
        mock_service.run_collection_sync.side_effect = InvalidStateError(
            "Collection has active operation"
        )

        response = test_client.post(f"/api/v2/collections/{uuid4()}/sync/run")

        assert response.status_code == 409


# =============================================================================
# POST /api/v2/collections/{id}/sync/pause - Pause Sync
# =============================================================================


class TestPauseSyncEndpoint:
    """Tests for POST /api/v2/collections/{collection_uuid}/sync/pause."""

    def test_pause_sync_success(self, test_client, mock_service, mock_collection):
        """Successfully pause sync."""
        mock_service.pause_collection_sync.return_value = mock_collection

        response = test_client.post(f"/api/v2/collections/{uuid4()}/sync/pause")

        assert response.status_code == 200

    def test_pause_sync_not_continuous_returns_400(self, test_client, mock_service):
        """Return 400 when collection is not in continuous sync mode."""
        mock_service.pause_collection_sync.side_effect = ValidationError(
            "Can only pause continuous sync collections", "sync_mode"
        )

        response = test_client.post(f"/api/v2/collections/{uuid4()}/sync/pause")

        assert response.status_code == 400
        assert "continuous sync" in response.json()["detail"]


# =============================================================================
# POST /api/v2/collections/{id}/sync/resume - Resume Sync
# =============================================================================


class TestResumeSyncEndpoint:
    """Tests for POST /api/v2/collections/{collection_uuid}/sync/resume."""

    def test_resume_sync_success(self, test_client, mock_service, mock_collection):
        """Successfully resume sync."""
        mock_service.resume_collection_sync.return_value = mock_collection

        response = test_client.post(f"/api/v2/collections/{uuid4()}/sync/resume")

        assert response.status_code == 200

    def test_resume_sync_not_paused_returns_400(self, test_client, mock_service):
        """Return 400 when collection is not paused."""
        mock_service.resume_collection_sync.side_effect = ValidationError(
            "Can only resume continuous sync collections", "sync_mode"
        )

        response = test_client.post(f"/api/v2/collections/{uuid4()}/sync/resume")

        assert response.status_code == 400
