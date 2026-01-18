"""Shared fixtures for service tests."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.models import Collection, CollectionSource, CollectionStatus, Operation, OperationType
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.collection_source_repository import CollectionSourceRepository
from shared.database.repositories.document_repository import DocumentRepository
from shared.database.repositories.operation_repository import OperationRepository
from shared.database.repositories.projection_run_repository import ProjectionRunRepository


def _create_mock_user_prefs() -> MagicMock:
    """Create a mock UserPreferences with HyDE settings."""
    prefs = MagicMock()
    prefs.search_use_hyde = False
    prefs.search_hyde_quality_tier = "LOW"
    prefs.search_hyde_timeout_seconds = 30
    return prefs


@pytest.fixture(autouse=True)
def mock_user_prefs_repo():
    """Mock UserPreferencesRepository.get_or_create to return proper prefs object.

    This is autouse=True because SearchService now depends on UserPreferencesRepository
    for HyDE settings, and tests that don't mock it will fail.
    """
    with patch("webui.services.search_service.UserPreferencesRepository") as mock_repo_class:
        mock_repo = AsyncMock()
        mock_repo.get_or_create = AsyncMock(return_value=_create_mock_user_prefs())
        mock_repo_class.return_value = mock_repo
        yield mock_repo_class


@pytest.fixture()
def mock_db_session() -> AsyncMock:
    """Mock database session with proper async methods."""
    mock = AsyncMock(spec=AsyncSession)
    mock.commit = AsyncMock()
    mock.rollback = AsyncMock()
    mock.close = AsyncMock()
    mock.execute = AsyncMock()
    mock.scalar = AsyncMock()
    mock.scalars = AsyncMock()
    return mock


@pytest.fixture()
def mock_collection_repo() -> AsyncMock:
    """Mock collection repository with all async methods properly configured."""
    mock = AsyncMock(spec=CollectionRepository)

    # Configure all async methods with AsyncMock
    mock.create = AsyncMock()
    mock.get_by_id = AsyncMock()
    mock.get_by_uuid = AsyncMock()
    mock.get_by_uuid_with_permission_check = AsyncMock()
    mock.list_for_user = AsyncMock()
    mock.update = AsyncMock()
    mock.delete = AsyncMock()
    mock.exists_by_name = AsyncMock()
    mock.get_by_name = AsyncMock()
    mock.list_ready_collections = AsyncMock()
    mock.get_accessible_collections = AsyncMock()

    return mock


@pytest.fixture()
def mock_operation_repo() -> AsyncMock:
    """Mock operation repository with all async methods properly configured."""
    mock = AsyncMock(spec=OperationRepository)

    # Configure all async methods with AsyncMock
    mock.create = AsyncMock()
    mock.get_by_id = AsyncMock()
    mock.get_by_uuid = AsyncMock()
    mock.list_for_collection = AsyncMock()
    mock.update = AsyncMock()
    mock.delete = AsyncMock()
    mock.get_active_operations = AsyncMock()
    mock.get_active_operations_count = AsyncMock()
    mock.cancel_operation = AsyncMock()

    return mock


@pytest.fixture()
def mock_document_repo() -> AsyncMock:
    """Mock document repository with all async methods properly configured."""
    mock = AsyncMock(spec=DocumentRepository)

    # Configure all async methods with AsyncMock
    mock.create = AsyncMock()
    mock.create_many = AsyncMock()
    mock.get_by_id = AsyncMock()
    mock.get_by_uuid = AsyncMock()
    mock.list_for_collection = AsyncMock()
    mock.list_by_collection = AsyncMock()
    mock.update = AsyncMock()
    mock.delete = AsyncMock()
    mock.delete_by_collection = AsyncMock()
    mock.delete_by_source = AsyncMock()
    mock.count_for_collection = AsyncMock()
    mock.exists_by_hash = AsyncMock()
    mock.get_by_file_path = AsyncMock()
    mock.list_by_source_id = AsyncMock()

    return mock


@pytest.fixture()
def mock_collection_source_repo() -> AsyncMock:
    """Mock collection source repository with all async methods properly configured."""
    mock = AsyncMock(spec=CollectionSourceRepository)

    # Configure all async methods with AsyncMock
    mock.create = AsyncMock()
    mock.get_by_id = AsyncMock()
    mock.get_by_collection_and_path = AsyncMock()
    mock.get_or_create = AsyncMock()
    mock.list_by_collection = AsyncMock()
    mock.update_stats = AsyncMock()
    mock.delete = AsyncMock()

    return mock


@pytest.fixture()
def mock_projection_repo() -> AsyncMock:
    """Mock projection run repository with async methods."""
    mock = AsyncMock(spec=ProjectionRunRepository)

    mock.create = AsyncMock()
    mock.get_by_uuid = AsyncMock()
    mock.list_for_collection = AsyncMock()
    mock.update_status = AsyncMock()
    mock.update_metadata = AsyncMock()
    mock.set_operation_uuid = AsyncMock()
    mock.delete = AsyncMock()
    mock.find_latest_completed_by_metadata_hash = AsyncMock()

    return mock


@pytest.fixture()
def mock_collection() -> MagicMock:
    """Mock a collection object with all required attributes."""
    collection = MagicMock(spec=Collection)
    collection.id = "123e4567-e89b-12d3-a456-426614174000"  # UUID string
    collection.uuid = "123e4567-e89b-12d3-a456-426614174000"  # Same as id
    collection.name = "Test Collection"
    collection.description = "Test description"
    collection.owner_id = 1
    collection.vector_store_name = "col_123e4567_e89b_12d3_a456_426614174000"
    collection.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
    collection.quantization = "float16"
    collection.chunk_size = 1000
    collection.chunk_overlap = 200
    collection.is_public = False
    collection.meta = {"key": "value"}
    collection.status = CollectionStatus.READY
    collection.created_at = MagicMock()
    collection.updated_at = MagicMock()
    # Add document and vector counts - these are expected by the service
    collection.document_count = 0
    collection.vector_count = 0

    # Add to_dict method
    def collection_to_dict() -> dict[str, Any]:
        return {
            "id": collection.id,
            "uuid": collection.uuid,
            "name": collection.name,
            "description": collection.description,
            "owner_id": collection.owner_id,
            "vector_store_name": collection.vector_store_name,
            "embedding_model": collection.embedding_model,
            "quantization": collection.quantization,
            "chunk_size": collection.chunk_size,
            "chunk_overlap": collection.chunk_overlap,
            "is_public": collection.is_public,
            "meta": collection.meta,
            "status": collection.status.value,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "document_count": collection.document_count,
            "vector_count": collection.vector_count,
        }

    collection.to_dict = MagicMock(side_effect=collection_to_dict)

    return collection


@pytest.fixture()
def mock_collection_source() -> MagicMock:
    """Mock a collection source object with all required attributes."""
    source = MagicMock(spec=CollectionSource)
    source.id = 1
    source.collection_id = "123e4567-e89b-12d3-a456-426614174000"
    source.source_type = "directory"
    source.source_path = "/path/to/source"
    source.source_config = {"path": "/path/to/source"}
    source.document_count = 0
    source.size_bytes = 0
    source.last_indexed_at = None
    source.created_at = MagicMock()
    source.updated_at = MagicMock()
    source.meta = {}

    return source


@pytest.fixture()
def mock_operation() -> MagicMock:
    """Mock an operation object with all required attributes."""
    operation = MagicMock(spec=Operation)
    operation.id = 1
    operation.uuid = "456e7890-e89b-12d3-a456-426614174001"
    operation.collection_id = "123e4567-e89b-12d3-a456-426614174000"  # UUID string
    operation.type = OperationType.INDEX
    operation.status = MagicMock(value="pending")
    operation.config = {}
    operation.created_at = MagicMock()
    operation.started_at = None
    operation.completed_at = None
    operation.error_message = None
    operation.created_by = 1

    # Add to_dict method
    def operation_to_dict() -> dict[str, Any]:
        return {
            "id": operation.id,
            "uuid": operation.uuid,
            "collection_id": operation.collection_id,
            "type": operation.type.value,
            "status": operation.status.value,
            "config": operation.config,
            "created_at": "2024-01-01T00:00:00",
            "started_at": None,
            "completed_at": None,
            "error_message": operation.error_message,
            "created_by": operation.created_by,
        }

    operation.to_dict = MagicMock(side_effect=operation_to_dict)

    return operation
