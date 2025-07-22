"""Base repository interfaces for database access.

This module defines abstract base classes for repository pattern implementation,
preparing for future database migrations and ensuring clean separation of concerns.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository interface.

    This defines the common operations that all repositories should support,
    regardless of the underlying storage mechanism.
    """

    @abstractmethod
    async def get(self, id: str) -> T | None:
        """Get an entity by ID."""

    @abstractmethod
    async def list(self, **filters: Any) -> list[T]:
        """List entities with optional filters."""

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create a new entity."""

    @abstractmethod
    async def update(self, id: str, updates: dict[str, Any]) -> T | None:
        """Update an entity by ID."""

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete an entity by ID. Returns True if deleted."""




class UserRepository(ABC):
    """Abstract interface for user data access."""

    @abstractmethod
    async def create_user(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new user."""

    @abstractmethod
    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Get a user by ID."""

    @abstractmethod
    async def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Get a user by username."""

    @abstractmethod
    async def update_user(self, user_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update a user."""

    @abstractmethod
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""

    @abstractmethod
    async def list_users(self, **filters: Any) -> list[dict[str, Any]]:
        """List all users with optional filters."""




class CollectionRepository(ABC):
    """Abstract interface for collection data access."""

    @abstractmethod
    async def list_collections(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """List all collections with optional user filter."""

    @abstractmethod
    async def get_collection_details(self, collection_name: str, user_id: str) -> dict[str, Any] | None:
        """Get detailed information for a collection."""

    @abstractmethod
    async def get_collection_files(
        self, collection_name: str, user_id: str, page: int = 1, limit: int = 50
    ) -> dict[str, Any]:
        """Get paginated files in a collection."""

    @abstractmethod
    async def rename_collection(self, old_name: str, new_name: str, user_id: str) -> bool:
        """Rename a collection."""

    @abstractmethod
    async def delete_collection(self, collection_name: str, user_id: str) -> dict[str, Any]:
        """Delete a collection and return deletion info."""

    @abstractmethod
    async def get_collection_metadata(self, collection_name: str) -> dict[str, Any] | None:
        """Get metadata from the first job of a collection."""


class AuthRepository(ABC):
    """Abstract interface for authentication data access."""

    @abstractmethod
    async def save_refresh_token(self, user_id: str, token_hash: str, expires_at: Any) -> None:
        """Save a refresh token for a user."""

    @abstractmethod
    async def verify_refresh_token(self, token: str) -> str | None:
        """Verify a refresh token and return user_id if valid."""

    @abstractmethod
    async def revoke_refresh_token(self, token: str) -> None:
        """Revoke a refresh token."""

    @abstractmethod
    async def update_user_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp."""


class ApiKeyRepository(ABC):
    """Abstract interface for API key data access."""

    @abstractmethod
    async def create_api_key(
        self, user_id: str, name: str, permissions: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create a new API key for a user."""

    @abstractmethod
    async def get_api_key(self, api_key_id: str) -> dict[str, Any] | None:
        """Get an API key by ID."""

    @abstractmethod
    async def get_api_key_by_hash(self, key_hash: str) -> dict[str, Any] | None:
        """Get an API key by its hash."""

    @abstractmethod
    async def list_user_api_keys(self, user_id: str) -> list[dict[str, Any]]:
        """List all API keys for a user."""

    @abstractmethod
    async def update_api_key(self, api_key_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update an API key."""

    @abstractmethod
    async def delete_api_key(self, api_key_id: str) -> bool:
        """Delete an API key."""

    @abstractmethod
    async def verify_api_key(self, api_key: str) -> dict[str, Any] | None:
        """Verify an API key and return associated data if valid."""

    @abstractmethod
    async def update_last_used(self, api_key_id: str) -> None:
        """Update the last used timestamp for an API key."""
