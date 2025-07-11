"""SQLite implementation of repository interfaces.

This module provides SQLite-specific implementations of the repository interfaces.
These implementations wrap the existing database functions to provide a clean
abstraction layer that can be replaced in the future.
"""

import logging
from typing import Any

from . import sqlite_implementation as db_impl
from .base import JobRepository, UserRepository

logger = logging.getLogger(__name__)


class SQLiteJobRepository(JobRepository):
    """SQLite implementation of JobRepository.

    This is a wrapper around the existing database functions,
    providing an async interface that matches the repository pattern.
    """

    def __init__(self) -> None:
        """Initialize with the local database implementation."""
        self.db = db_impl

    async def create_job(self, job_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new job."""
        try:
            # Convert async to sync for now (database module is sync)
            # Note: create_job expects job_data as a dict, not unpacked
            job_id = self.db.create_job(job_data)
            # create_job returns just the ID, but the interface expects the full job object
            job = self.db.get_job(job_id)
            if job is None:
                raise ValueError(f"Failed to retrieve created job with ID: {job_id}")
            return job
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            raise

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by ID."""
        try:
            result: dict[str, Any] | None = self.db.get_job(job_id)
            return result
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            raise

    async def update_job(self, job_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update a job."""
        try:
            self.db.update_job(job_id, updates)
            # update_job returns None, but the interface expects the updated job object
            updated_job = self.db.get_job(job_id)
            if updated_job is None:
                logger.warning(f"Job {job_id} not found after update")
            return updated_job
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")
            raise

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        try:
            # Check if job exists before deletion
            job = self.db.get_job(job_id)
            if job is None:
                logger.warning(f"Attempted to delete non-existent job: {job_id}")
                return False
            
            self.db.delete_job(job_id)
            
            # Verify deletion was successful
            deleted_job = self.db.get_job(job_id)
            success = deleted_job is None
            if not success:
                logger.error(f"Job {job_id} still exists after deletion attempt")
            return success
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            raise

    async def list_jobs(self, user_id: str | None = None, **filters: Any) -> list[dict[str, Any]]:
        """List jobs with optional filters."""
        try:
            # Note: filters are accepted for interface compatibility but not used in SQLite implementation
            _ = filters
            
            # Convert string user_id to int if provided
            user_id_int: int | None = None
            if user_id is not None:
                try:
                    user_id_int = int(user_id)
                except ValueError:
                    logger.error(f"Invalid user_id format: {user_id}")
                    raise ValueError(f"user_id must be a valid integer, got: {user_id}")
            
            result: list[dict[str, Any]] = self.db.list_jobs(user_id=user_id_int)
            return result
        except ValueError:
            # Re-raise ValueError for invalid user_id
            raise
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise

    async def get_all_job_ids(self) -> list[str]:
        """Get all job IDs."""
        try:
            jobs = self.db.list_jobs()
            return [job["id"] for job in jobs]
        except Exception as e:
            logger.error(f"Failed to get all job IDs: {e}")
            raise


class SQLiteUserRepository(UserRepository):
    """SQLite implementation of UserRepository."""

    def __init__(self) -> None:
        """Initialize with the local database implementation."""
        self.db = db_impl

    async def create_user(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new user."""
        try:
            result: dict[str, Any] = self.db.create_user(**user_data)
            return result
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Get a user by ID."""
        try:
            result: dict[str, Any] | None = self.db.get_user(user_id)
            return result
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            raise

    async def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Get a user by username."""
        try:
            # The sqlite implementation uses get_user for username lookup
            result: dict[str, Any] | None = self.db.get_user(username)
            return result
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            raise

    async def update_user(self, user_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update a user."""
        try:
            # Note: Current database module doesn't have an update_user method
            # This is a placeholder implementation
            user: dict[str, Any] | None = self.db.get_user(user_id)
            if not user:
                logger.warning(f"User {user_id} not found for update")
                return None
            # TODO: Implement actual database update when method is available
            # For now, return the existing user
            logger.warning("User update not yet implemented in database layer")
            return user
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            raise

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        try:
            # Note: Current database module doesn't have a delete_user method
            # This is a placeholder implementation
            user = self.db.get_user(user_id)
            if user is None:
                logger.warning(f"Attempted to delete non-existent user: {user_id}")
                return False
            # TODO: Implement actual database deletion when method is available
            logger.warning("User deletion not yet implemented in database layer")
            return False  # Return False since we can't actually delete
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            raise
