"""Tests for chunking strategy validation in CollectionService."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import Collection
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.webui.services.collection_service import CollectionService


@pytest.fixture()
def mock_db_session() -> Any:
    """Create a mock database session."""
    session = MagicMock(spec=AsyncSession)
    session.commit = AsyncMock()
    return session


@pytest.fixture()
def mock_collection_repo() -> Any:
    """Create a mock collection repository."""
    repo = MagicMock(spec=CollectionRepository)
    repo.create = AsyncMock()
    repo.get_by_uuid_with_permission_check = AsyncMock()
    return repo


@pytest.fixture()
def mock_operation_repo() -> Any:
    """Create a mock operation repository."""
    repo = MagicMock(spec=OperationRepository)
    repo.create = AsyncMock()
    return repo


@pytest.fixture()
def mock_document_repo() -> Any:
    """Create a mock document repository."""
    return MagicMock(spec=DocumentRepository)


@pytest.fixture()
def collection_service(
    mock_db_session: Any,
    mock_collection_repo: Any,
    mock_operation_repo: Any,
    mock_document_repo: Any,
) -> CollectionService:
    """Create a CollectionService instance with mocked dependencies."""
    return CollectionService(
        db_session=mock_db_session,
        collection_repo=mock_collection_repo,
        operation_repo=mock_operation_repo,
        document_repo=mock_document_repo,
    )


class TestCreateCollectionValidation:
    """Test validation in create_collection method."""

    @pytest.mark.asyncio()
    async def test_valid_chunking_strategy_accepted(
        self, collection_service: CollectionService, mock_collection_repo: Any
    ) -> None:
        """Test that valid chunking strategies are accepted and normalized."""
        # Mock successful collection creation
        mock_collection = MagicMock(spec=Collection)
        mock_collection.uuid = "test-uuid"
        mock_collection.name = "Test Collection"
        mock_collection_repo.create.return_value = mock_collection

        # Test with a valid strategy
        with patch("packages.webui.services.collection_service.celery_app"):
            await collection_service.create_collection(
                user_id=1,
                name="Test Collection",
                config={"chunking_strategy": "recursive", "chunking_config": {"chunk_size": 500}},
            )

        # Verify the strategy was normalized (recursive -> recursive)
        mock_collection_repo.create.assert_called_once()
        call_args = mock_collection_repo.create.call_args.kwargs
        assert call_args["chunking_strategy"] == "recursive"

    @pytest.mark.asyncio()
    async def test_invalid_chunking_strategy_rejected(self, collection_service: CollectionService) -> None:
        """Test that invalid chunking strategies are rejected with helpful error."""
        with pytest.raises(ValueError, match="Invalid chunking_strategy") as exc_info:
            await collection_service.create_collection(
                user_id=1, name="Test Collection", config={"chunking_strategy": "invalid_strategy"}
            )

        error_msg = str(exc_info.value)
        assert "Invalid chunking_strategy" in error_msg
        assert "Available strategies:" in error_msg  # Check that available strategies are listed

    @pytest.mark.asyncio()
    async def test_chunking_config_without_strategy_rejected(self, collection_service: CollectionService) -> None:
        """Test that chunking_config without strategy is rejected."""
        with pytest.raises(ValueError, match="chunking_config requires chunking_strategy") as exc_info:
            await collection_service.create_collection(
                user_id=1, name="Test Collection", config={"chunking_config": {"chunk_size": 500}}
            )

        error_msg = str(exc_info.value)
        assert "chunking_config requires chunking_strategy" in error_msg

    @pytest.mark.asyncio()
    async def test_invalid_chunking_config_rejected(self, collection_service: CollectionService) -> None:
        """Test that invalid chunking configurations are rejected."""
        with pytest.raises(ValueError, match="Invalid chunking_config") as exc_info:
            await collection_service.create_collection(
                user_id=1,
                name="Test Collection",
                config={
                    "chunking_strategy": "recursive",
                    "chunking_config": {"chunk_size": -100},  # Invalid: negative size
                },
            )

        error_msg = str(exc_info.value)
        assert "Invalid chunking_config" in error_msg
        assert "chunk_size must be at least 10" in error_msg

    @pytest.mark.asyncio()
    async def test_strategy_name_normalization(
        self, collection_service: CollectionService, mock_collection_repo: Any
    ) -> None:
        """Test that strategy names are normalized to internal format."""
        mock_collection = MagicMock(spec=Collection)
        mock_collection.uuid = "test-uuid"
        mock_collection.name = "Test Collection"
        mock_collection_repo.create.return_value = mock_collection

        # Test with a strategy that needs normalization
        with patch("packages.webui.services.collection_service.celery_app"):
            await collection_service.create_collection(
                user_id=1,
                name="Test Collection",
                config={"chunking_strategy": "fixed_size"},  # Should normalize to "character"
            )

        # Verify the strategy was normalized
        mock_collection_repo.create.assert_called_once()
        call_args = mock_collection_repo.create.call_args.kwargs
        assert call_args["chunking_strategy"] == "character"


class TestUpdateCollectionValidation:
    """Test validation in update_collection method."""

    @pytest.mark.asyncio()
    async def test_valid_strategy_update_accepted(
        self,
        collection_service: CollectionService,
        mock_collection_repo: Any,
        mock_db_session: Any,
    ) -> None:
        """Test that valid strategy updates are accepted."""
        # Mock existing collection
        mock_collection = MagicMock(spec=Collection)
        mock_collection.owner_id = 1  # Set owner_id to match user_id
        mock_collection.chunking_strategy = "recursive"
        mock_collection.chunking_config = {"chunk_size": 500}
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # Update with valid strategy
        await collection_service.update_collection(
            collection_id="test-uuid", updates={"chunking_strategy": "semantic"}, user_id=1
        )

        # Verify commit was called (indicates successful update)
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio()
    async def test_invalid_strategy_update_rejected(
        self, collection_service: CollectionService, mock_collection_repo: Any
    ) -> None:
        """Test that invalid strategy updates are rejected."""
        mock_collection = MagicMock(spec=Collection)
        mock_collection.owner_id = 1  # Set owner_id to match user_id
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(ValueError, match="Invalid chunking strategy") as exc_info:
            await collection_service.update_collection(
                collection_id="test-uuid", updates={"chunking_strategy": "nonexistent"}, user_id=1
            )

        error_msg = str(exc_info.value)
        assert "Invalid chunking strategy" in error_msg
        assert "Available" in error_msg

    @pytest.mark.asyncio()
    async def test_config_update_with_existing_strategy(
        self,
        collection_service: CollectionService,
        mock_collection_repo: Any,
        mock_db_session: Any,
    ) -> None:
        """Test updating config when collection already has a strategy."""
        mock_collection = MagicMock(spec=Collection)
        mock_collection.owner_id = 1  # Set owner_id to match user_id
        mock_collection.chunking_strategy = "recursive"
        mock_collection.chunking_config = {"chunk_size": 500}
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # Update config only
        await collection_service.update_collection(
            collection_id="test-uuid",
            updates={"chunking_config": {"chunk_size": 1000, "chunk_overlap": 100}},
            user_id=1,
        )

        # Verify commit was called (indicates successful update)
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio()
    async def test_config_update_without_strategy_rejected(
        self, collection_service: CollectionService, mock_collection_repo: Any
    ) -> None:
        """Test that config updates without strategy are rejected."""
        mock_collection = MagicMock(spec=Collection)
        mock_collection.owner_id = 1  # Set owner_id to match user_id
        mock_collection.chunking_strategy = None  # No existing strategy
        mock_collection.chunking_config = None
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(ValueError, match="chunking_config requires chunking_strategy") as exc_info:
            await collection_service.update_collection(
                collection_id="test-uuid", updates={"chunking_config": {"chunk_size": 500}}, user_id=1
            )

        error_msg = str(exc_info.value)
        assert "chunking_config requires chunking_strategy" in error_msg

    @pytest.mark.asyncio()
    async def test_simultaneous_strategy_and_config_update(
        self,
        collection_service: CollectionService,
        mock_collection_repo: Any,
        mock_db_session: Any,
    ) -> None:
        """Test updating both strategy and config together."""
        mock_collection = MagicMock(spec=Collection)
        mock_collection.owner_id = 1  # Set owner_id to match user_id
        mock_collection.chunking_strategy = "recursive"
        mock_collection.chunking_config = {"chunk_size": 500}
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # Update both strategy and config
        await collection_service.update_collection(
            collection_id="test-uuid",
            updates={
                "chunking_strategy": "semantic",
                "chunking_config": {"chunk_size": 512, "similarity_threshold": 0.7},
            },
            user_id=1,
        )

        # Verify commit was called (indicates successful update)
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio()
    async def test_invalid_config_for_strategy_rejected(
        self, collection_service: CollectionService, mock_collection_repo: Any
    ) -> None:
        """Test that invalid configs for specific strategies are rejected."""
        mock_collection = MagicMock(spec=Collection)
        mock_collection.owner_id = 1  # Set owner_id to match user_id
        mock_collection.chunking_strategy = "semantic"
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(ValueError, match="Invalid chunking config") as exc_info:
            await collection_service.update_collection(
                collection_id="test-uuid",
                updates={"chunking_config": {"similarity_threshold": 2.0}},  # Invalid: > 1.0
                user_id=1,
            )

        error_msg = str(exc_info.value)
        assert "Invalid chunking config" in error_msg
        assert "similarity_threshold must be between 0 and 1" in error_msg
