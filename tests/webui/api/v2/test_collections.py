"""
Tests for v2 collections API endpoints.

Comprehensive test coverage for all collection management endpoints including
CRUD operations, source management, and reindexing functionality.
"""

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from packages.shared.database.exceptions import (
    AccessDeniedError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidStateError,
)
from packages.shared.database.models import Collection, CollectionStatus
from packages.webui.auth import get_current_user
from packages.webui.dependencies import get_collection_for_user
from packages.webui.main import app
from packages.webui.services.collection_service import CollectionService
from packages.webui.services.factory import get_collection_service


@pytest.fixture()
def mock_user() -> dict[str, Any]:
    """Mock authenticated user."""
    return {"id": 1, "username": "testuser"}


@pytest.fixture()
def mock_collection() -> MagicMock:
    """Mock collection object."""
    collection = MagicMock(spec=Collection)
    collection.id = "123e4567-e89b-12d3-a456-426614174000"
    collection.name = "Test Collection"
    collection.description = "A test collection"
    collection.owner_id = 1
    collection.vector_store_name = "test_vector_store"
    collection.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
    collection.quantization = "float16"
    collection.chunk_size = 1000
    collection.chunk_overlap = 200
    collection.is_public = False
    collection.meta = {"test": "metadata"}
    collection.created_at = datetime.now(UTC)
    collection.updated_at = datetime.now(UTC)
    collection.document_count = 10
    collection.vector_count = 100
    collection.status = CollectionStatus.READY
    collection.status_message = None
    return collection


@pytest.fixture()
def mock_collection_service() -> AsyncMock:
    """Mock CollectionService."""
    return AsyncMock(spec=CollectionService)


@pytest.fixture()
def client(mock_user: dict[str, Any], mock_collection_service: AsyncMock) -> TestClient:
    """Create a test client with mocked dependencies."""

    # Override dependencies
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_collection_service] = lambda: mock_collection_service

    # Create test client
    client = TestClient(app)

    yield client

    # Clean up
    app.dependency_overrides.clear()


class TestCreateCollection:
    """Test create_collection endpoint."""

    def test_create_collection_success(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test successful collection creation."""
        # Setup
        create_request = {
            "name": "New Collection",
            "description": "Test description",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "is_public": False,
            "metadata": {"test": "value"},
        }

        collection_data = {
            "id": str(uuid.uuid4()),
            "name": create_request["name"],
            "description": create_request["description"],
            "owner_id": 1,
            "vector_store_name": "qdrant_collection_name",
            "embedding_model": create_request["embedding_model"],
            "quantization": create_request["quantization"],
            "chunk_size": create_request["chunk_size"],
            "chunk_overlap": create_request["chunk_overlap"],
            "is_public": create_request["is_public"],
            "metadata": create_request["metadata"],
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "document_count": 0,
            "vector_count": 0,
            "status": "pending",
            "status_message": None,
        }

        operation_data = {
            "uuid": str(uuid.uuid4()),
            "collection_id": collection_data["id"],
            "type": "index",
            "status": "pending",
            "config": {},
            "created_at": datetime.now(UTC),
        }

        mock_collection_service.create_collection.return_value = (collection_data, operation_data)

        # Execute
        response = client.post("/api/v2/collections", json=create_request)

        # Verify
        assert response.status_code == 201
        result = response.json()
        assert result["name"] == create_request["name"]
        assert result["description"] == create_request["description"]
        assert result["embedding_model"] == create_request["embedding_model"]
        assert result["initial_operation_id"] == operation_data["uuid"]

        mock_collection_service.create_collection.assert_called_once_with(
            user_id=1,
            name=create_request["name"],
            description=create_request["description"],
            config={
                "embedding_model": create_request["embedding_model"],
                "quantization": create_request["quantization"],
                "chunk_size": create_request["chunk_size"],
                "chunk_overlap": create_request["chunk_overlap"],
                "chunking_strategy": None,
                "chunking_config": None,
                "is_public": create_request["is_public"],
                "metadata": create_request["metadata"],
            },
        )

    def test_create_collection_duplicate_name(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test 409 error when collection name already exists."""
        create_request = {"name": "Existing Collection"}

        mock_collection_service.create_collection.side_effect = EntityAlreadyExistsError(
            "Collection", "Existing Collection"
        )

        response = client.post("/api/v2/collections", json=create_request)

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_create_collection_invalid_data(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test 400 error for invalid collection data."""
        create_request = {"name": "Invalid Collection", "chunk_size": 10000}

        mock_collection_service.create_collection.side_effect = ValueError("Invalid chunk size")

        response = client.post("/api/v2/collections", json=create_request)

        assert response.status_code == 400
        assert "Invalid chunk size" in response.json()["detail"]

    def test_create_collection_service_error(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test 500 error for service failures."""
        create_request = {"name": "Test Collection"}

        mock_collection_service.create_collection.side_effect = Exception("Database error")

        response = client.post("/api/v2/collections", json=create_request)

        assert response.status_code == 500
        assert "Failed to create collection" in response.json()["detail"]

    def test_create_collection_omits_null_chunk_fields(
        self, client: TestClient, mock_collection_service: AsyncMock
    ) -> None:
        """When chunk fields are null, API should omit them in config."""
        # Setup request with explicit nulls
        create_request = {
            "name": "Null Chunks",
            "description": None,
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "chunk_size": None,
            "chunk_overlap": None,
            "is_public": False,
            "metadata": None,
        }

        # Minimal service return payload

        collection_data = {
            "id": str(uuid.uuid4()),
            "name": create_request["name"],
            "description": None,
            "owner_id": 1,
            "vector_store_name": "qdrant_collection_name",
            "embedding_model": create_request["embedding_model"],
            "quantization": create_request["quantization"],
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "is_public": create_request["is_public"],
            "metadata": None,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "document_count": 0,
            "vector_count": 0,
            "status": "pending",
            "status_message": None,
        }
        operation_data = {
            "uuid": str(uuid.uuid4()),
            "collection_id": collection_data["id"],
            "type": "index",
            "status": "pending",
            "config": {},
            "created_at": datetime.now(UTC),
        }

        mock_collection_service.create_collection.return_value = (collection_data, operation_data)

        # Execute
        response = client.post("/api/v2/collections", json=create_request)

        # Verify
        assert response.status_code == 201

        # Capture call to service
        assert mock_collection_service.create_collection.call_count == 1
        _, kwargs = mock_collection_service.create_collection.call_args
        cfg = kwargs["config"]
        # Should not include null chunk fields
        assert "chunk_size" not in cfg
        assert "chunk_overlap" not in cfg
        # Should include non-null values
        assert cfg["embedding_model"] == create_request["embedding_model"]
        assert cfg["quantization"] == create_request["quantization"]
        assert cfg["is_public"] is False


class TestListCollections:
    """Test list_collections endpoint."""

    def test_list_collections_success(
        self, client: TestClient, mock_collection: MagicMock, mock_collection_service: AsyncMock
    ) -> None:
        """Test successful collection listing."""
        collections = [mock_collection]
        total = 1

        mock_collection_service.list_for_user.return_value = (collections, total)

        response = client.get("/api/v2/collections?page=1&per_page=50&include_public=true")

        assert response.status_code == 200
        result = response.json()
        assert len(result["collections"]) == 1
        assert result["total"] == 1
        assert result["page"] == 1
        assert result["per_page"] == 50

        mock_collection_service.list_for_user.assert_called_once_with(
            user_id=1, offset=0, limit=50, include_public=True
        )

    def test_list_collections_pagination(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test collection listing with pagination."""
        mock_collection_service.list_for_user.return_value = ([], 0)

        response = client.get("/api/v2/collections?page=3&per_page=20&include_public=false")

        assert response.status_code == 200

        # Verify offset calculation
        mock_collection_service.list_for_user.assert_called_once_with(
            user_id=1, offset=40, limit=20, include_public=False  # (page-1) * per_page = 2 * 20
        )

    def test_list_collections_service_error(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test 500 error for service failures."""
        mock_collection_service.list_for_user.side_effect = Exception("Database error")

        response = client.get("/api/v2/collections")

        assert response.status_code == 500
        assert "Failed to list collections" in response.json()["detail"]


class TestGetCollection:
    """Test get_collection endpoint."""

    def test_get_collection_success(self, client: TestClient, mock_collection: MagicMock) -> None:
        """Test successful collection retrieval."""

        app.dependency_overrides[get_collection_for_user] = lambda: mock_collection

        collection_uuid = mock_collection.id
        response = client.get(f"/api/v2/collections/{collection_uuid}")

        assert response.status_code == 200
        result = response.json()
        assert result["id"] == mock_collection.id
        assert result["name"] == mock_collection.name
        assert result["description"] == mock_collection.description


class TestUpdateCollection:
    """Test update_collection endpoint."""

    def test_update_collection_success(
        self, client: TestClient, mock_collection: MagicMock, mock_collection_service: AsyncMock
    ) -> None:
        """Test successful collection update."""
        collection_uuid = str(uuid.uuid4())
        update_request = {
            "name": "Updated Name",
            "description": "Updated description",
            "is_public": True,
            "metadata": {"updated": "metadata"},
        }

        mock_collection_service.update.return_value = mock_collection

        response = client.put(f"/api/v2/collections/{collection_uuid}", json=update_request)

        assert response.status_code == 200

        mock_collection_service.update.assert_called_once_with(
            collection_id=collection_uuid,
            user_id=1,
            updates={
                "name": update_request["name"],
                "description": update_request["description"],
                "is_public": update_request["is_public"],
                "meta": update_request["metadata"],
            },
        )

    def test_update_collection_partial_update(
        self, client: TestClient, mock_collection: MagicMock, mock_collection_service: AsyncMock
    ) -> None:
        """Test partial collection update with only some fields."""
        collection_uuid = str(uuid.uuid4())
        update_request = {"name": "New Name Only"}

        mock_collection_service.update.return_value = mock_collection

        response = client.put(f"/api/v2/collections/{collection_uuid}", json=update_request)

        assert response.status_code == 200

        # Verify only name is included in updates
        mock_collection_service.update.assert_called_once_with(
            collection_id=collection_uuid, user_id=1, updates={"name": "New Name Only"}
        )

    def test_update_collection_not_found(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test 404 error when collection not found."""
        collection_uuid = str(uuid.uuid4())
        update_request = {"name": "Updated Name"}

        mock_collection_service.update.side_effect = EntityNotFoundError("Collection", collection_uuid)

        response = client.put(f"/api/v2/collections/{collection_uuid}", json=update_request)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_update_collection_access_denied(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test 403 error when user lacks permission."""
        collection_uuid = str(uuid.uuid4())
        update_request = {"name": "Updated Name"}

        mock_collection_service.update.side_effect = AccessDeniedError("1", "Collection", collection_uuid)

        response = client.put(f"/api/v2/collections/{collection_uuid}", json=update_request)

        assert response.status_code == 403
        assert "Only the collection owner" in response.json()["detail"]

    def test_update_collection_duplicate_name(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test 409 error when new name already exists."""
        collection_uuid = str(uuid.uuid4())
        update_request = {"name": "Existing Name"}

        mock_collection_service.update.side_effect = EntityAlreadyExistsError("Collection", "Existing Name")

        response = client.put(f"/api/v2/collections/{collection_uuid}", json=update_request)

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_update_collection_validation_error(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test 422 error for pydantic validation failures."""
        collection_uuid = str(uuid.uuid4())
        update_request = {"name": "Invalid/Name"}

        # No need to mock service - Pydantic validation happens first
        response = client.put(f"/api/v2/collections/{collection_uuid}", json=update_request)

        assert response.status_code == 422
        # Pydantic validation errors have a specific structure
        assert "string_pattern_mismatch" in str(response.json())


class TestDeleteCollection:
    """Test delete_collection endpoint."""

    def test_delete_collection_success(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test successful collection deletion."""
        collection_uuid = str(uuid.uuid4())

        mock_collection_service.delete_collection.return_value = None

        response = client.delete(f"/api/v2/collections/{collection_uuid}")

        assert response.status_code == 204

        mock_collection_service.delete_collection.assert_called_once_with(collection_id=collection_uuid, user_id=1)

    def test_delete_collection_not_found(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test 404 error when collection not found."""
        collection_uuid = str(uuid.uuid4())

        mock_collection_service.delete_collection.side_effect = EntityNotFoundError("Collection", collection_uuid)

        response = client.delete(f"/api/v2/collections/{collection_uuid}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_delete_collection_access_denied(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test 403 error when user lacks permission."""
        collection_uuid = str(uuid.uuid4())

        mock_collection_service.delete_collection.side_effect = AccessDeniedError("1", "Collection", collection_uuid)

        response = client.delete(f"/api/v2/collections/{collection_uuid}")

        assert response.status_code == 403
        assert "Only the collection owner" in response.json()["detail"]

    def test_delete_collection_operation_in_progress(
        self, client: TestClient, mock_collection_service: AsyncMock
    ) -> None:
        """Test 409 error when operation is in progress."""
        collection_uuid = str(uuid.uuid4())

        mock_collection_service.delete_collection.side_effect = InvalidStateError(
            "Cannot delete collection with active operations"
        )

        response = client.delete(f"/api/v2/collections/{collection_uuid}")

        assert response.status_code == 409
        assert "Cannot delete collection" in response.json()["detail"]


class TestAddSource:
    """Test add_source endpoint."""

    def test_add_source_success(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test successful source addition."""
        collection_uuid = str(uuid.uuid4())
        add_source_request = {
            "source_path": "/data/documents",
            "config": {"recursive": True},
        }

        operation_data = {
            "uuid": str(uuid.uuid4()),
            "collection_id": collection_uuid,
            "type": "add_source",
            "status": "pending",
            "config": {
                "source_path": add_source_request["source_path"],
                **add_source_request["config"],
            },
            "created_at": datetime.now(UTC),
        }

        mock_collection_service.add_source.return_value = operation_data

        response = client.post(f"/api/v2/collections/{collection_uuid}/sources", json=add_source_request)

        assert response.status_code == 202
        result = response.json()
        assert result["id"] == operation_data["uuid"]
        assert result["type"] == "add_source"

        mock_collection_service.add_source.assert_called_once_with(
            collection_id=collection_uuid,
            user_id=1,
            source_path=add_source_request["source_path"],
            source_config=add_source_request["config"],
        )

    def test_add_source_collection_not_found(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test 404 error when collection not found."""
        collection_uuid = str(uuid.uuid4())
        add_source_request = {"source_path": "/data/documents"}

        mock_collection_service.add_source.side_effect = EntityNotFoundError("Collection", collection_uuid)

        response = client.post(f"/api/v2/collections/{collection_uuid}/sources", json=add_source_request)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_add_source_invalid_state(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test 409 error when collection is in invalid state."""
        collection_uuid = str(uuid.uuid4())
        add_source_request = {"source_path": "/data/documents"}

        mock_collection_service.add_source.side_effect = InvalidStateError("Collection is currently being reindexed")

        response = client.post(f"/api/v2/collections/{collection_uuid}/sources", json=add_source_request)

        assert response.status_code == 409
        assert "Collection is currently being reindexed" in response.json()["detail"]


class TestRemoveSource:
    """Test remove_source endpoint."""

    def test_remove_source_success(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test successful source removal."""
        collection_uuid = str(uuid.uuid4())
        source_path = "/data/documents"

        operation_data = {
            "uuid": str(uuid.uuid4()),
            "collection_id": collection_uuid,
            "type": "remove_source",
            "status": "pending",
            "config": {"source_path": source_path},
            "created_at": datetime.now(UTC),
        }

        mock_collection_service.remove_source.return_value = operation_data

        response = client.delete(f"/api/v2/collections/{collection_uuid}/sources?source_path={source_path}")

        assert response.status_code == 202
        result = response.json()
        assert result["id"] == operation_data["uuid"]
        assert result["type"] == "remove_source"

        mock_collection_service.remove_source.assert_called_once_with(
            collection_id=collection_uuid, user_id=1, source_path=source_path
        )


class TestReindexCollection:
    """Test reindex_collection endpoint."""

    def test_reindex_collection_success(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test successful collection reindexing."""
        collection_uuid = str(uuid.uuid4())
        config_updates = {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "chunk_size": 500,
        }

        operation_data = {
            "uuid": str(uuid.uuid4()),
            "collection_id": collection_uuid,
            "type": "reindex",
            "status": "pending",
            "config": config_updates,
            "created_at": datetime.now(UTC),
        }

        mock_collection_service.reindex_collection.return_value = operation_data

        response = client.post(f"/api/v2/collections/{collection_uuid}/reindex", json=config_updates)

        assert response.status_code == 202
        result = response.json()
        assert result["id"] == operation_data["uuid"]
        assert result["type"] == "reindex"

        mock_collection_service.reindex_collection.assert_called_once_with(
            collection_id=collection_uuid, user_id=1, config_updates=config_updates
        )

    def test_reindex_collection_without_updates(self, client: TestClient, mock_collection_service: AsyncMock) -> None:
        """Test reindexing without configuration updates."""
        collection_uuid = str(uuid.uuid4())

        operation_data = {
            "uuid": str(uuid.uuid4()),
            "collection_id": collection_uuid,
            "type": "reindex",
            "status": "pending",
            "config": {},
            "created_at": datetime.now(UTC),
        }

        mock_collection_service.reindex_collection.return_value = operation_data

        response = client.post(f"/api/v2/collections/{collection_uuid}/reindex")

        assert response.status_code == 202
        result = response.json()
        assert result["id"] == operation_data["uuid"]

        mock_collection_service.reindex_collection.assert_called_once_with(
            collection_id=collection_uuid, user_id=1, config_updates=None
        )


# Additional test classes for operations and documents would follow the same pattern...
