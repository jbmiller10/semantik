"""SQLite implementation of repository interfaces.

This module provides SQLite-specific implementations of the repository interfaces.
These implementations wrap the existing database functions to provide a clean
abstraction layer that can be replaced in the future.
"""

from typing import Any

from .base import JobRepository, UserRepository


class SQLiteJobRepository(JobRepository):
    """SQLite implementation of JobRepository.

    This is a wrapper around the existing database functions,
    providing an async interface that matches the repository pattern.
    """

    def __init__(self, database_module: Any):
        """Initialize with the webui database module.

        Args:
            database_module: The webui.database module (injected to avoid circular imports)
        """
        self.db = database_module

    async def create_job(self, job_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new job."""
        # Convert async to sync for now (database module is sync)
        result: dict[str, Any] = self.db.create_job(**job_data)
        return result

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by ID."""
        result: dict[str, Any] | None = self.db.get_job(job_id)
        return result

    async def update_job(self, job_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update a job."""
        result: dict[str, Any] | None = self.db.update_job(job_id, updates)
        return result

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        result: bool = self.db.delete_job(job_id)
        return result

    async def list_jobs(self, user_id: str | None = None, **filters: Any) -> list[dict[str, Any]]:
        """List jobs with optional filters."""
        # Note: filters are accepted for interface compatibility but not used in SQLite implementation
        _ = filters
        result: list[dict[str, Any]] = self.db.list_jobs(user_id=user_id)
        return result

    async def get_all_job_ids(self) -> list[str]:
        """Get all job IDs."""
        jobs = self.db.list_jobs()
        return [job["id"] for job in jobs]


class SQLiteUserRepository(UserRepository):
    """SQLite implementation of UserRepository."""

    def __init__(self, database_module: Any):
        """Initialize with the webui database module."""
        self.db = database_module

    async def create_user(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new user."""
        result: dict[str, Any] = self.db.create_user(**user_data)
        return result

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Get a user by ID."""
        result: dict[str, Any] | None = self.db.get_user(user_id)
        return result

    async def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Get a user by username."""
        result: dict[str, Any] | None = self.db.get_user_by_username(username)
        return result

    async def update_user(self, user_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update a user."""
        # Note: Current database module might not have this method
        # This is a placeholder for future implementation
        user: dict[str, Any] | None = self.db.get_user(user_id)
        if not user:
            return None
        # In a real implementation, this would update the database
        user.update(updates)
        return user

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        # Note: Current database module might not have this method
        # This is a placeholder for future implementation
        _ = user_id  # Mark as used for linting
        return True  # Placeholder
