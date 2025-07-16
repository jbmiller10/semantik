"""
Unit tests for FastAPI dependencies in webui.dependencies
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from packages.shared.database.exceptions import (
    AccessDeniedError,
    EntityNotFoundError,
)
from packages.shared.database.models import Collection
from packages.webui.dependencies import get_collection_for_user


@pytest.mark.asyncio()
class TestGetCollectionForUser:
    """Test cases for get_collection_for_user dependency."""

    @pytest.fixture()
    def mock_db(self) -> AsyncMock:
        """Mock async database session."""
        return AsyncMock()

    @pytest.fixture()
    def mock_user(self) -> dict[str, Any]:
        """Mock current user dictionary."""
        return {
            "id": "123",  # String ID that can be converted to int
            "username": "testuser",
            "email": "test@example.com",
            "is_active": True,
            "is_superuser": False,
        }

    @pytest.fixture()
    def mock_collection(self) -> MagicMock:
        """Mock Collection object."""
        collection = MagicMock(spec=Collection)
        collection.id = "collection-456"
        collection.name = "Test Collection"
        collection.owner_id = "user-123"
        collection.is_public = False
        return collection

    async def test_get_collection_for_user_success(
        self, mock_db: AsyncMock, mock_user: dict[str, Any], mock_collection: MagicMock
    ) -> None:
        """Test successful retrieval of collection with proper permissions."""
        # Arrange
        with patch("packages.webui.dependencies.CollectionRepository") as mock_repo:
            mock_repo_instance = mock_repo.return_value
            mock_repo_instance.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_collection)

            # Act
            result = await get_collection_for_user(collection_uuid="collection-456", current_user=mock_user, db=mock_db)

            # Assert
            assert result == mock_collection
            mock_repo.assert_called_once_with(mock_db)
            mock_repo_instance.get_by_uuid_with_permission_check.assert_called_once_with(
                collection_uuid="collection-456", user_id=123  # Now expects int, not string
            )

    async def test_get_collection_for_user_not_found(self, mock_db: AsyncMock, mock_user: dict[str, Any]) -> None:
        """Test HTTPException 404 when collection is not found."""
        # Arrange
        with patch("packages.webui.dependencies.CollectionRepository") as mock_repo:
            mock_repo_instance = mock_repo.return_value
            mock_repo_instance.get_by_uuid_with_permission_check = AsyncMock(
                side_effect=EntityNotFoundError("collection", "nonexistent-uuid")
            )

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await get_collection_for_user(collection_uuid="nonexistent-uuid", current_user=mock_user, db=mock_db)

            assert exc_info.value.status_code == 404
            assert "Collection with UUID 'nonexistent-uuid' not found" in str(exc_info.value.detail)

    async def test_get_collection_for_user_access_denied(self, mock_db: AsyncMock, mock_user: dict[str, Any]) -> None:
        """Test HTTPException 403 when user lacks permission."""
        # Arrange
        with patch("packages.webui.dependencies.CollectionRepository") as mock_repo:
            mock_repo_instance = mock_repo.return_value
            mock_repo_instance.get_by_uuid_with_permission_check = AsyncMock(
                side_effect=AccessDeniedError("123", "collection", "private-collection")
            )

            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await get_collection_for_user(collection_uuid="private-collection", current_user=mock_user, db=mock_db)

            assert exc_info.value.status_code == 403
            assert "You do not have permission to access this collection" in str(exc_info.value.detail)

    async def test_get_collection_for_user_with_different_user_id_format(
        self, mock_db: AsyncMock, mock_collection: MagicMock
    ) -> None:
        """Test that user ID is properly converted to int."""
        # Arrange
        mock_user_with_str_id = {
            "id": "456",  # String ID that needs conversion
            "username": "testuser",
            "email": "test@example.com",
            "is_active": True,
            "is_superuser": False,
        }

        with patch("packages.webui.dependencies.CollectionRepository") as mock_repo:
            mock_repo_instance = mock_repo.return_value
            mock_repo_instance.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_collection)

            # Act
            result = await get_collection_for_user(
                collection_uuid="collection-456", current_user=mock_user_with_str_id, db=mock_db
            )

            # Assert
            assert result == mock_collection
            mock_repo_instance.get_by_uuid_with_permission_check.assert_called_once_with(
                collection_uuid="collection-456", user_id=456  # Should be converted to int
            )
