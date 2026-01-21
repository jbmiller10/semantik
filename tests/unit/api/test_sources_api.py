"""Unit tests for Sources API v2 endpoints.

Note: Sync policy (mode, interval, pause/resume) is now managed at collection level.
Sources only track per-source telemetry (last_run_* fields).
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared.database.exceptions import AccessDeniedError, EntityNotFoundError, InvalidStateError
from shared.database.models import CollectionSource
from shared.utils.encryption import EncryptionNotConfiguredError
from webui.api.v2.sources import router
from webui.middleware.exception_handlers import register_global_exception_handlers

# Create test app with global exception handlers
app = FastAPI()
register_global_exception_handlers(app)
app.include_router(router)


@pytest.fixture()
def mock_current_user():
    """Mock current user dependency."""
    return {"id": "1", "username": "testuser"}


@pytest.fixture()
def mock_service():
    """Mock SourceService."""
    return AsyncMock()


@pytest.fixture()
def mock_source():
    """Create a mock CollectionSource.

    Note: sync_mode, interval_minutes, paused_at, next_run_at are no longer
    on CollectionSource - they're at collection level now.
    """
    return CollectionSource(
        id=1,
        collection_id=str(uuid4()),
        source_type="directory",
        source_path="/data/test",
        source_config={"path": "/data/test"},
        document_count=10,
        size_bytes=1024,
        last_run_started_at=None,
        last_run_completed_at=None,
        last_run_status=None,
        last_error=None,
        last_indexed_at=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


@pytest.fixture()
def test_client(mock_current_user, mock_service):
    """Create test client with mocked dependencies."""
    from webui.api.v2 import sources

    # Override dependencies
    app.dependency_overrides[sources.get_current_user] = lambda: mock_current_user
    app.dependency_overrides[sources.get_source_service] = lambda: mock_service

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


class TestListSourcesEndpoint:
    """Tests for GET /api/v2/collections/{collection_id}/sources."""

    def test_list_sources_success(self, test_client, mock_service, mock_source):
        """Test successful source listing."""
        # API expects list of (source, secret_types) tuples
        mock_service.list_sources.return_value = ([(mock_source, [])], 1)

        response = test_client.get(f"/api/v2/collections/{mock_source.collection_id}/sources")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["source_type"] == "directory"

    def test_list_sources_collection_not_found(self, test_client, mock_service):
        """Test listing sources for non-existent collection."""
        mock_service.list_sources.side_effect = EntityNotFoundError("collection", "test-id")

        response = test_client.get("/api/v2/collections/test-id/sources")

        assert response.status_code == 404

    def test_list_sources_access_denied(self, test_client, mock_service):
        """Test listing sources without access."""
        mock_service.list_sources.side_effect = AccessDeniedError("1", "collection", "test-id")

        response = test_client.get("/api/v2/collections/test-id/sources")

        assert response.status_code == 403


class TestGetSourceEndpoint:
    """Tests for GET /api/v2/collections/{collection_id}/sources/{source_id}."""

    def test_get_source_success(self, test_client, mock_service, mock_source):
        """Test successful source retrieval."""
        # API expects (source, secret_types) tuple when include_secret_types=True
        mock_service.get_source.return_value = (mock_source, [])

        response = test_client.get(f"/api/v2/collections/{mock_source.collection_id}/sources/{mock_source.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == mock_source.id

    def test_get_source_not_found(self, test_client, mock_service):
        """Test getting non-existent source."""
        mock_service.get_source.side_effect = EntityNotFoundError("collection_source", "999")

        response = test_client.get("/api/v2/collections/test-id/sources/999")

        assert response.status_code == 404


class TestUpdateSourceEndpoint:
    """Tests for PATCH /api/v2/collections/{collection_id}/sources/{source_id}."""

    def test_update_source_success(self, test_client, mock_service, mock_source):
        """Test successful source update."""
        mock_source.source_config = {"path": "/data/updated"}
        # API expects (source, secret_types) tuple
        mock_service.update_source.return_value = (mock_source, [])

        response = test_client.patch(
            f"/api/v2/collections/{mock_source.collection_id}/sources/{mock_source.id}",
            json={"source_config": {"path": "/data/updated"}},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["source_config"]["path"] == "/data/updated"

    def test_update_source_not_found(self, test_client, mock_service):
        """Test updating non-existent source."""
        mock_service.update_source.side_effect = EntityNotFoundError("collection_source", "999")

        response = test_client.patch(
            "/api/v2/collections/test-id/sources/999",
            json={"source_config": {}},
        )

        assert response.status_code == 404

    def test_update_source_encryption_not_configured(self, test_client, mock_service):
        """Test source secrets update when encryption is not configured."""
        mock_service.update_source.side_effect = EncryptionNotConfiguredError(
            "Encryption not configured - set CONNECTOR_SECRETS_KEY environment variable"
        )

        response = test_client.patch(
            "/api/v2/collections/test-id/sources/1",
            json={"secrets": {"password": "super-secret"}},
        )

        assert response.status_code == 400


class TestDeleteSourceEndpoint:
    """Tests for DELETE /api/v2/collections/{collection_id}/sources/{source_id}."""

    def test_delete_source_success(self, test_client, mock_service, mock_source):
        """Test successful source deletion."""
        mock_service.delete_source.return_value = {
            "id": 1,
            "uuid": str(uuid4()),
            "type": "remove_source",
            "status": "pending",
        }

        response = test_client.delete(f"/api/v2/collections/{mock_source.collection_id}/sources/{mock_source.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "remove_source"

    def test_delete_source_active_operation(self, test_client, mock_service, mock_source):
        """Test deleting source blocked by active operation."""
        mock_service.delete_source.side_effect = InvalidStateError("Collection has active operation(s)")

        response = test_client.delete(f"/api/v2/collections/{mock_source.collection_id}/sources/{mock_source.id}")

        assert response.status_code == 409
