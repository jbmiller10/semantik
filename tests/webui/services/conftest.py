"""Shared fixtures for service tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.shared.database.models import Collection, CollectionStatus, Operation, OperationType


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
    
    # Add to_dict method
    def collection_to_dict():
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
        }
    collection.to_dict = MagicMock(side_effect=collection_to_dict)
    
    return collection


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
    def operation_to_dict():
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