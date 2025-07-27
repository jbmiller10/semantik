"""
Tests for v2 collections API endpoints.

Comprehensive test coverage for all collection management endpoints including
CRUD operations, source management, and reindexing functionality.
"""

import uuid
from datetime import datetime, UTC
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.exceptions import (
    AccessDeniedError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidStateError,
    ValidationError,
)
from packages.shared.database.models import (
    Collection,
    CollectionStatus,
    Document,
    DocumentStatus,
    Operation,
    OperationStatus,
    OperationType,
)
from packages.webui.api.schemas import (
    AddSourceRequest,
    CollectionCreate,
    CollectionListResponse,
    CollectionResponse,
    CollectionUpdate,
    DocumentListResponse,
    OperationResponse,
)
from packages.webui.api.v2.collections import (
    add_source,
    create_collection,
    delete_collection,
    get_collection,
    list_collection_documents,
    list_collection_operations,
    list_collections,
    reindex_collection,
    remove_source,
    update_collection,
)
from packages.webui.services.collection_service import CollectionService


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
def mock_request() -> MagicMock:
    """Mock FastAPI Request object."""
    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    return request


class TestCreateCollection:
    """Test create_collection endpoint."""

    @pytest.mark.asyncio()
    async def test_create_collection_success(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
        """Test successful collection creation."""
        # Setup
        create_request = CollectionCreate(
            name="New Collection",
            description="Test description",
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            quantization="float16",
            chunk_size=1000,
            chunk_overlap=200,
            is_public=False,
            metadata={"test": "value"},
        )

        collection_data = {
            "id": str(uuid.uuid4()),
            "name": create_request.name,
            "description": create_request.description,
            "owner_id": mock_user["id"],
            "vector_store_name": "qdrant_collection_name",
            "embedding_model": create_request.embedding_model,
            "quantization": create_request.quantization,
            "chunk_size": create_request.chunk_size,
            "chunk_overlap": create_request.chunk_overlap,
            "is_public": create_request.is_public,
            "metadata": create_request.metadata,
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
        result = await create_collection(
            request=mock_request,
            create_request=create_request,
            current_user=mock_user,
            service=mock_collection_service,
        )

        # Verify
        assert isinstance(result, CollectionResponse)
        assert result.name == create_request.name
        assert result.description == create_request.description
        assert result.embedding_model == create_request.embedding_model
        assert result.initial_operation_id == operation_data["uuid"]

        mock_collection_service.create_collection.assert_called_once_with(
            user_id=mock_user["id"],
            name=create_request.name,
            description=create_request.description,
            config={
                "embedding_model": create_request.embedding_model,
                "quantization": create_request.quantization,
                "chunk_size": create_request.chunk_size,
                "chunk_overlap": create_request.chunk_overlap,
                "is_public": create_request.is_public,
                "metadata": create_request.metadata,
            },
        )

    @pytest.mark.asyncio()
    async def test_create_collection_duplicate_name(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
        """Test 409 error when collection name already exists."""
        create_request = CollectionCreate(name="Existing Collection")

        mock_collection_service.create_collection.side_effect = EntityAlreadyExistsError(
            "Collection", "Existing Collection"
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_collection(
                request=mock_request,
                create_request=create_request,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 409
        assert "already exists" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_create_collection_invalid_data(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
        """Test 400 error for invalid collection data."""
        create_request = CollectionCreate(name="Invalid Collection", chunk_size=10000)

        mock_collection_service.create_collection.side_effect = ValueError("Invalid chunk size")

        with pytest.raises(HTTPException) as exc_info:
            await create_collection(
                request=mock_request,
                create_request=create_request,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 400
        assert "Invalid chunk size" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_create_collection_service_error(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
        """Test 500 error for service failures."""
        create_request = CollectionCreate(name="Test Collection")

        mock_collection_service.create_collection.side_effect = Exception("Database error")

        with pytest.raises(HTTPException) as exc_info:
            await create_collection(
                request=mock_request,
                create_request=create_request,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 500
        assert "Failed to create collection" in str(exc_info.value.detail)


class TestListCollections:
    """Test list_collections endpoint."""

    @pytest.mark.asyncio()
    async def test_list_collections_success(
        self,
        mock_user: dict[str, Any],
        mock_collection: MagicMock,
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test successful collection listing."""
        collections = [mock_collection]
        total = 1

        mock_collection_service.list_for_user.return_value = (collections, total)

        result = await list_collections(
            page=1,
            per_page=50,
            include_public=True,
            current_user=mock_user,
            service=mock_collection_service,
        )

        assert isinstance(result, CollectionListResponse)
        assert len(result.collections) == 1
        assert result.total == 1
        assert result.page == 1
        assert result.per_page == 50

        mock_collection_service.list_for_user.assert_called_once_with(
            user_id=mock_user["id"],
            offset=0,
            limit=50,
            include_public=True,
        )

    @pytest.mark.asyncio()
    async def test_list_collections_pagination(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test collection listing with pagination."""
        mock_collection_service.list_for_user.return_value = ([], 0)

        result = await list_collections(
            page=3,
            per_page=20,
            include_public=False,
            current_user=mock_user,
            service=mock_collection_service,
        )

        # Verify offset calculation
        mock_collection_service.list_for_user.assert_called_once_with(
            user_id=mock_user["id"],
            offset=40,  # (page-1) * per_page = 2 * 20
            limit=20,
            include_public=False,
        )

    @pytest.mark.asyncio()
    async def test_list_collections_service_error(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test 500 error for service failures."""
        mock_collection_service.list_for_user.side_effect = Exception("Database error")

        with pytest.raises(HTTPException) as exc_info:
            await list_collections(
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 500
        assert "Failed to list collections" in str(exc_info.value.detail)


class TestGetCollection:
    """Test get_collection endpoint."""

    @pytest.mark.asyncio()
    async def test_get_collection_success(self, mock_collection: MagicMock) -> None:
        """Test successful collection retrieval."""
        result = await get_collection(collection=mock_collection)

        assert isinstance(result, CollectionResponse)
        assert result.id == mock_collection.id
        assert result.name == mock_collection.name
        assert result.description == mock_collection.description


class TestUpdateCollection:
    """Test update_collection endpoint."""

    @pytest.mark.asyncio()
    async def test_update_collection_success(
        self,
        mock_user: dict[str, Any],
        mock_collection: MagicMock,
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test successful collection update."""
        collection_uuid = str(uuid.uuid4())
        update_request = CollectionUpdate(
            name="Updated Name",
            description="Updated description",
            is_public=True,
            metadata={"updated": "metadata"},
        )

        mock_collection_service.update.return_value = mock_collection

        result = await update_collection(
            collection_uuid=collection_uuid,
            request=update_request,
            current_user=mock_user,
            service=mock_collection_service,
        )

        assert isinstance(result, CollectionResponse)
        
        mock_collection_service.update.assert_called_once_with(
            collection_id=collection_uuid,
            user_id=mock_user["id"],
            updates={
                "name": update_request.name,
                "description": update_request.description,
                "is_public": update_request.is_public,
                "meta": update_request.metadata,
            },
        )

    @pytest.mark.asyncio()
    async def test_update_collection_partial_update(
        self,
        mock_user: dict[str, Any],
        mock_collection: MagicMock,
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test partial collection update with only some fields."""
        collection_uuid = str(uuid.uuid4())
        update_request = CollectionUpdate(name="New Name Only")

        mock_collection_service.update.return_value = mock_collection

        await update_collection(
            collection_uuid=collection_uuid,
            request=update_request,
            current_user=mock_user,
            service=mock_collection_service,
        )

        # Verify only name is included in updates
        mock_collection_service.update.assert_called_once_with(
            collection_id=collection_uuid,
            user_id=mock_user["id"],
            updates={"name": "New Name Only"},
        )

    @pytest.mark.asyncio()
    async def test_update_collection_not_found(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test 404 error when collection not found."""
        collection_uuid = str(uuid.uuid4())
        update_request = CollectionUpdate(name="Updated Name")

        mock_collection_service.update.side_effect = EntityNotFoundError("Collection", collection_uuid)

        with pytest.raises(HTTPException) as exc_info:
            await update_collection(
                collection_uuid=collection_uuid,
                request=update_request,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_update_collection_access_denied(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test 403 error when user lacks permission."""
        collection_uuid = str(uuid.uuid4())
        update_request = CollectionUpdate(name="Updated Name")

        mock_collection_service.update.side_effect = AccessDeniedError("Not the owner")

        with pytest.raises(HTTPException) as exc_info:
            await update_collection(
                collection_uuid=collection_uuid,
                request=update_request,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 403
        assert "Only the collection owner" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_update_collection_duplicate_name(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test 409 error when new name already exists."""
        collection_uuid = str(uuid.uuid4())
        update_request = CollectionUpdate(name="Existing Name")

        mock_collection_service.update.side_effect = EntityAlreadyExistsError("Collection", "Existing Name")

        with pytest.raises(HTTPException) as exc_info:
            await update_collection(
                collection_uuid=collection_uuid,
                request=update_request,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 409
        assert "already exists" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_update_collection_validation_error(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test 400 error for validation failures."""
        collection_uuid = str(uuid.uuid4())
        update_request = CollectionUpdate(name="Invalid/Name")

        mock_collection_service.update.side_effect = ValidationError("Invalid name format")

        with pytest.raises(HTTPException) as exc_info:
            await update_collection(
                collection_uuid=collection_uuid,
                request=update_request,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 400
        assert "Invalid name format" in str(exc_info.value.detail)


class TestDeleteCollection:
    """Test delete_collection endpoint."""

    @pytest.mark.asyncio()
    async def test_delete_collection_success(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
        """Test successful collection deletion."""
        collection_uuid = str(uuid.uuid4())

        mock_collection_service.delete_collection.return_value = None

        # Should not raise any exception
        await delete_collection(
            request=mock_request,
            collection_uuid=collection_uuid,
            current_user=mock_user,
            service=mock_collection_service,
        )

        mock_collection_service.delete_collection.assert_called_once_with(
            collection_id=collection_uuid,
            user_id=mock_user["id"],
        )

    @pytest.mark.asyncio()
    async def test_delete_collection_not_found(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
        """Test 404 error when collection not found."""
        collection_uuid = str(uuid.uuid4())

        mock_collection_service.delete_collection.side_effect = EntityNotFoundError("Collection", collection_uuid)

        with pytest.raises(HTTPException) as exc_info:
            await delete_collection(
                request=mock_request,
                collection_uuid=collection_uuid,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_delete_collection_access_denied(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
        """Test 403 error when user lacks permission."""
        collection_uuid = str(uuid.uuid4())

        mock_collection_service.delete_collection.side_effect = AccessDeniedError("Not the owner")

        with pytest.raises(HTTPException) as exc_info:
            await delete_collection(
                request=mock_request,
                collection_uuid=collection_uuid,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 403
        assert "Only the collection owner" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_delete_collection_operation_in_progress(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
        """Test 409 error when operation is in progress."""
        collection_uuid = str(uuid.uuid4())

        mock_collection_service.delete_collection.side_effect = InvalidStateError(
            "Cannot delete collection with active operations"
        )

        with pytest.raises(HTTPException) as exc_info:
            await delete_collection(
                request=mock_request,
                collection_uuid=collection_uuid,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 409
        assert "Cannot delete collection" in str(exc_info.value.detail)


class TestAddSource:
    """Test add_source endpoint."""

    @pytest.mark.asyncio()
    async def test_add_source_success(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
        """Test successful source addition."""
        collection_uuid = str(uuid.uuid4())
        add_source_request = AddSourceRequest(
            source_path="/data/documents",
            config={"recursive": True},
        )

        operation_data = {
            "uuid": str(uuid.uuid4()),
            "collection_id": collection_uuid,
            "type": "add_source",
            "status": "pending",
            "config": {
                "source_path": add_source_request.source_path,
                **add_source_request.config,
            },
            "created_at": datetime.now(UTC),
        }

        mock_collection_service.add_source.return_value = operation_data

        result = await add_source(
            request=mock_request,
            collection_uuid=collection_uuid,
            add_source_request=add_source_request,
            current_user=mock_user,
            service=mock_collection_service,
        )

        assert isinstance(result, OperationResponse)
        assert result.id == operation_data["uuid"]
        assert result.type == "add_source"

        mock_collection_service.add_source.assert_called_once_with(
            collection_id=collection_uuid,
            user_id=mock_user["id"],
            source_path=add_source_request.source_path,
            source_config=add_source_request.config,
        )

    @pytest.mark.asyncio()
    async def test_add_source_collection_not_found(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
        """Test 404 error when collection not found."""
        collection_uuid = str(uuid.uuid4())
        add_source_request = AddSourceRequest(source_path="/data/documents")

        mock_collection_service.add_source.side_effect = EntityNotFoundError("Collection", collection_uuid)

        with pytest.raises(HTTPException) as exc_info:
            await add_source(
                request=mock_request,
                collection_uuid=collection_uuid,
                add_source_request=add_source_request,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_add_source_invalid_state(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
        """Test 409 error when collection is in invalid state."""
        collection_uuid = str(uuid.uuid4())
        add_source_request = AddSourceRequest(source_path="/data/documents")

        mock_collection_service.add_source.side_effect = InvalidStateError(
            "Collection is currently being reindexed"
        )

        with pytest.raises(HTTPException) as exc_info:
            await add_source(
                request=mock_request,
                collection_uuid=collection_uuid,
                add_source_request=add_source_request,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 409
        assert "Collection is currently being reindexed" in str(exc_info.value.detail)


class TestRemoveSource:
    """Test remove_source endpoint."""

    @pytest.mark.asyncio()
    async def test_remove_source_success(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
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

        result = await remove_source(
            request=mock_request,
            collection_uuid=collection_uuid,
            source_path=source_path,
            current_user=mock_user,
            service=mock_collection_service,
        )

        assert isinstance(result, OperationResponse)
        assert result.id == operation_data["uuid"]
        assert result.type == "remove_source"

        mock_collection_service.remove_source.assert_called_once_with(
            collection_id=collection_uuid,
            user_id=mock_user["id"],
            source_path=source_path,
        )


class TestReindexCollection:
    """Test reindex_collection endpoint."""

    @pytest.mark.asyncio()
    async def test_reindex_collection_success(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
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

        result = await reindex_collection(
            request=mock_request,
            collection_uuid=collection_uuid,
            config_updates=config_updates,
            current_user=mock_user,
            service=mock_collection_service,
        )

        assert isinstance(result, OperationResponse)
        assert result.id == operation_data["uuid"]
        assert result.type == "reindex"

        mock_collection_service.reindex_collection.assert_called_once_with(
            collection_id=collection_uuid,
            user_id=mock_user["id"],
            config_updates=config_updates,
        )

    @pytest.mark.asyncio()
    async def test_reindex_collection_without_updates(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
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

        result = await reindex_collection(
            request=mock_request,
            collection_uuid=collection_uuid,
            config_updates=None,
            current_user=mock_user,
            service=mock_collection_service,
        )

        assert isinstance(result, OperationResponse)

        mock_collection_service.reindex_collection.assert_called_once_with(
            collection_id=collection_uuid,
            user_id=mock_user["id"],
            config_updates=None,
        )


class TestListCollectionOperations:
    """Test list_collection_operations endpoint."""

    @pytest.mark.asyncio()
    async def test_list_operations_success(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test successful operation listing."""
        collection_uuid = str(uuid.uuid4())

        mock_operation = MagicMock(spec=Operation)
        mock_operation.uuid = str(uuid.uuid4())
        mock_operation.collection_id = collection_uuid
        mock_operation.type = OperationType.INDEX
        mock_operation.status = OperationStatus.COMPLETED
        mock_operation.config = {}
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = datetime.now(UTC)
        mock_operation.completed_at = datetime.now(UTC)
        mock_operation.error_message = None

        operations = [mock_operation]
        total = 1

        mock_collection_service.list_operations.return_value = (operations, total)

        result = await list_collection_operations(
            collection_uuid=collection_uuid,
            status=None,
            operation_type=None,
            page=1,
            per_page=50,
            current_user=mock_user,
            service=mock_collection_service,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], OperationResponse)
        assert result[0].id == mock_operation.uuid
        assert result[0].type == "index"
        assert result[0].status == "completed"

    @pytest.mark.asyncio()
    async def test_list_operations_with_filters(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test operation listing with status and type filters."""
        collection_uuid = str(uuid.uuid4())

        # Create operations with different statuses and types
        operation1 = MagicMock(spec=Operation)
        operation1.uuid = str(uuid.uuid4())
        operation1.type = OperationType.INDEX
        operation1.status = OperationStatus.COMPLETED
        operation1.config = {}
        operation1.created_at = datetime.now(UTC)
        operation1.started_at = None
        operation1.completed_at = None
        operation1.error_message = None

        operation2 = MagicMock(spec=Operation)
        operation2.uuid = str(uuid.uuid4())
        operation2.type = OperationType.REINDEX
        operation2.status = OperationStatus.PROCESSING
        operation2.config = {}
        operation2.created_at = datetime.now(UTC)
        operation2.started_at = datetime.now(UTC)
        operation2.completed_at = None
        operation2.error_message = None

        operations = [operation1, operation2]
        mock_collection_service.list_operations.return_value = (operations, 2)

        # Filter by status
        result = await list_collection_operations(
            collection_uuid=collection_uuid,
            status="completed",
            operation_type=None,
            page=1,
            per_page=50,
            current_user=mock_user,
            service=mock_collection_service,
        )

        assert len(result) == 1
        assert result[0].status == "completed"

        # Filter by type
        result = await list_collection_operations(
            collection_uuid=collection_uuid,
            status=None,
            operation_type="reindex",
            page=1,
            per_page=50,
            current_user=mock_user,
            service=mock_collection_service,
        )

        assert len(result) == 1
        assert result[0].type == "reindex"

    @pytest.mark.asyncio()
    async def test_list_operations_invalid_status(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test 400 error for invalid status filter."""
        collection_uuid = str(uuid.uuid4())
        mock_collection_service.list_operations.return_value = ([], 0)

        with pytest.raises(HTTPException) as exc_info:
            await list_collection_operations(
                collection_uuid=collection_uuid,
                status="invalid_status",
                operation_type=None,
                page=1,
                per_page=50,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 400
        assert "Invalid status" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_list_operations_invalid_type(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test 400 error for invalid operation type filter."""
        collection_uuid = str(uuid.uuid4())
        mock_collection_service.list_operations.return_value = ([], 0)

        with pytest.raises(HTTPException) as exc_info:
            await list_collection_operations(
                collection_uuid=collection_uuid,
                status=None,
                operation_type="invalid_type",
                page=1,
                per_page=50,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 400
        assert "Invalid operation type" in str(exc_info.value.detail)


class TestListCollectionDocuments:
    """Test list_collection_documents endpoint."""

    @pytest.mark.asyncio()
    async def test_list_documents_success(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test successful document listing."""
        collection_uuid = str(uuid.uuid4())

        mock_document = MagicMock(spec=Document)
        mock_document.id = str(uuid.uuid4())
        mock_document.collection_id = collection_uuid
        mock_document.file_name = "test.pdf"
        mock_document.file_path = "/data/test.pdf"
        mock_document.file_size = 1024
        mock_document.mime_type = "application/pdf"
        mock_document.content_hash = "abc123"
        mock_document.status = DocumentStatus.COMPLETED
        mock_document.error_message = None
        mock_document.chunk_count = 10
        mock_document.meta = {}
        mock_document.created_at = datetime.now(UTC)
        mock_document.updated_at = datetime.now(UTC)

        documents = [mock_document]
        total = 1

        mock_collection_service.list_documents.return_value = (documents, total)

        result = await list_collection_documents(
            collection_uuid=collection_uuid,
            page=1,
            per_page=50,
            status=None,
            current_user=mock_user,
            service=mock_collection_service,
        )

        assert isinstance(result, DocumentListResponse)
        assert len(result.documents) == 1
        assert result.total == 1
        assert result.documents[0].file_name == "test.pdf"

    @pytest.mark.asyncio()
    async def test_list_documents_with_status_filter(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test document listing with status filter."""
        collection_uuid = str(uuid.uuid4())

        # Create documents with different statuses
        doc1 = MagicMock(spec=Document)
        doc1.id = str(uuid.uuid4())
        doc1.collection_id = collection_uuid
        doc1.file_name = "doc1.pdf"
        doc1.file_path = "/data/doc1.pdf"
        doc1.file_size = 1024
        doc1.mime_type = "application/pdf"
        doc1.content_hash = "abc123"
        doc1.status = DocumentStatus.COMPLETED
        doc1.error_message = None
        doc1.chunk_count = 10
        doc1.meta = {}
        doc1.created_at = datetime.now(UTC)
        doc1.updated_at = datetime.now(UTC)

        doc2 = MagicMock(spec=Document)
        doc2.id = str(uuid.uuid4())
        doc2.collection_id = collection_uuid
        doc2.file_name = "doc2.pdf"
        doc2.file_path = "/data/doc2.pdf"
        doc2.file_size = 2048
        doc2.mime_type = "application/pdf"
        doc2.content_hash = "def456"
        doc2.status = DocumentStatus.FAILED
        doc2.error_message = "Processing error"
        doc2.chunk_count = 0
        doc2.meta = {}
        doc2.created_at = datetime.now(UTC)
        doc2.updated_at = datetime.now(UTC)

        documents = [doc1, doc2]
        mock_collection_service.list_documents.return_value = (documents, 2)

        result = await list_collection_documents(
            collection_uuid=collection_uuid,
            page=1,
            per_page=50,
            status="completed",
            current_user=mock_user,
            service=mock_collection_service,
        )

        assert len(result.documents) == 1
        assert result.documents[0].status == "completed"
        assert result.total == 1  # Total is updated after filtering

    @pytest.mark.asyncio()
    async def test_list_documents_invalid_status(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test 400 error for invalid status filter."""
        collection_uuid = str(uuid.uuid4())
        mock_collection_service.list_documents.return_value = ([], 0)

        with pytest.raises(HTTPException) as exc_info:
            await list_collection_documents(
                collection_uuid=collection_uuid,
                page=1,
                per_page=50,
                status="invalid_status",
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 400
        assert "Invalid status" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_list_documents_collection_not_found(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test 404 error when collection not found."""
        collection_uuid = str(uuid.uuid4())

        mock_collection_service.list_documents.side_effect = EntityNotFoundError("Collection", collection_uuid)

        with pytest.raises(HTTPException) as exc_info:
            await list_collection_documents(
                collection_uuid=collection_uuid,
                page=1,
                per_page=50,
                status=None,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_list_documents_access_denied(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test 403 error when user lacks access."""
        collection_uuid = str(uuid.uuid4())

        mock_collection_service.list_documents.side_effect = AccessDeniedError("No access to collection")

        with pytest.raises(HTTPException) as exc_info:
            await list_collection_documents(
                collection_uuid=collection_uuid,
                page=1,
                per_page=50,
                status=None,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 403
        assert "don't have access" in str(exc_info.value.detail)


# Edge case tests
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio()
    async def test_create_collection_with_special_characters(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
        """Test collection creation with special but valid characters."""
        create_request = CollectionCreate(
            name="Test-Collection_2025 (v1.0)",
            description="Collection with special chars: !@#$%^&*()",
        )

        collection_data = {
            "id": str(uuid.uuid4()),
            "name": create_request.name,
            "description": create_request.description,
            "owner_id": mock_user["id"],
            "vector_store_name": "test_collection",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "is_public": False,
            "metadata": None,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "document_count": 0,
            "vector_count": 0,
            "status": "pending",
        }

        operation_data = {"uuid": str(uuid.uuid4()), "type": "index"}

        mock_collection_service.create_collection.return_value = (collection_data, operation_data)

        result = await create_collection(
            request=mock_request,
            create_request=create_request,
            current_user=mock_user,
            service=mock_collection_service,
        )

        assert result.name == create_request.name
        assert result.description == create_request.description

    @pytest.mark.asyncio()
    async def test_list_operations_empty_results(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test listing operations when none exist."""
        collection_uuid = str(uuid.uuid4())

        mock_collection_service.list_operations.return_value = ([], 0)

        result = await list_collection_operations(
            collection_uuid=collection_uuid,
            status=None,
            operation_type=None,
            page=1,
            per_page=50,
            current_user=mock_user,
            service=mock_collection_service,
        )

        assert result == []

    @pytest.mark.asyncio()
    async def test_update_collection_no_changes(
        self,
        mock_user: dict[str, Any],
        mock_collection: MagicMock,
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test update with empty request (no changes)."""
        collection_uuid = str(uuid.uuid4())
        update_request = CollectionUpdate()  # All fields None

        mock_collection_service.update.return_value = mock_collection

        await update_collection(
            collection_uuid=collection_uuid,
            request=update_request,
            current_user=mock_user,
            service=mock_collection_service,
        )

        # Should be called with empty updates dict
        mock_collection_service.update.assert_called_once_with(
            collection_id=collection_uuid,
            user_id=mock_user["id"],
            updates={},
        )

    @pytest.mark.asyncio()
    async def test_pagination_boundary_conditions(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
    ) -> None:
        """Test pagination with edge values."""
        # Test with page=1, per_page=1
        mock_collection_service.list_for_user.return_value = ([], 0)

        await list_collections(
            page=1,
            per_page=1,
            include_public=True,
            current_user=mock_user,
            service=mock_collection_service,
        )

        mock_collection_service.list_for_user.assert_called_with(
            user_id=mock_user["id"],
            offset=0,
            limit=1,
            include_public=True,
        )

        # Test with maximum per_page
        await list_collections(
            page=1,
            per_page=100,
            include_public=True,
            current_user=mock_user,
            service=mock_collection_service,
        )

        mock_collection_service.list_for_user.assert_called_with(
            user_id=mock_user["id"],
            offset=0,
            limit=100,
            include_public=True,
        )

    @pytest.mark.asyncio()
    async def test_concurrent_operation_handling(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock,
        mock_request: MagicMock,
    ) -> None:
        """Test handling of concurrent operation errors."""
        collection_uuid = str(uuid.uuid4())

        # Simulate concurrent operation error during delete
        mock_collection_service.delete_collection.side_effect = InvalidStateError(
            "Another operation is currently running on this collection"
        )

        with pytest.raises(HTTPException) as exc_info:
            await delete_collection(
                request=mock_request,
                collection_uuid=collection_uuid,
                current_user=mock_user,
                service=mock_collection_service,
            )

        assert exc_info.value.status_code == 409
        assert "Another operation is currently running" in str(exc_info.value.detail)