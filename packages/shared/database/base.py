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
    regardless of the underlying storage mechanism (SQLite, PostgreSQL, etc).
    """

    @abstractmethod
    async def get(self, id: str) -> T | None:
        """Get an entity by ID."""
        pass

    @abstractmethod
    async def list(self, **filters: Any) -> list[T]:
        """List entities with optional filters."""
        pass

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create a new entity."""
        pass

    @abstractmethod
    async def update(self, id: str, updates: dict[str, Any]) -> T | None:
        """Update an entity by ID."""
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete an entity by ID. Returns True if deleted."""
        pass


class JobRepository(ABC):
    """Abstract interface for job data access.

    This will be implemented by SQLiteJobRepository initially,
    and can be replaced with PostgreSQLJobRepository in the future.
    """

    @abstractmethod
    async def create_job(self, job_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new job."""
        pass

    @abstractmethod
    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by ID."""
        pass

    @abstractmethod
    async def update_job(self, job_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update a job."""
        pass

    @abstractmethod
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        pass

    @abstractmethod
    async def list_jobs(self, user_id: str | None = None, **filters: Any) -> list[dict[str, Any]]:
        """List jobs with optional filters."""
        pass

    @abstractmethod
    async def get_all_job_ids(self) -> list[str]:
        """Get all job IDs (for maintenance tasks)."""
        pass


class UserRepository(ABC):
    """Abstract interface for user data access."""

    @abstractmethod
    async def create_user(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new user."""
        pass

    @abstractmethod
    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Get a user by ID."""
        pass

    @abstractmethod
    async def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Get a user by username."""
        pass

    @abstractmethod
    async def update_user(self, user_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update a user."""
        pass

    @abstractmethod
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        pass
