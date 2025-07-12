"""SQLite implementation of repository interfaces.

This module provides SQLite-specific implementations of the repository interfaces.
These implementations wrap the existing database functions to provide a clean
abstraction layer that can be replaced in the future.
"""

import logging
from typing import Any

from . import sqlite_implementation as db_impl
from .base import AuthRepository, CollectionRepository, FileRepository, JobRepository, UserRepository
from .utils import parse_user_id

logger = logging.getLogger(__name__)


class SQLiteJobRepository(JobRepository):
    """SQLite implementation of JobRepository.

    This is a wrapper around the existing database functions,
    providing an async interface that matches the repository pattern.

    Note on Error Handling:
    - Methods that modify data (create, update, delete) will raise exceptions on failure
    - Methods that retrieve data (get, list) return None or empty list when not found
    - All database errors are logged and re-raised for proper error propagation
    """

    def __init__(self) -> None:
        """Initialize with the local database implementation."""
        self.db = db_impl

    async def create_job(self, job_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new job.

        Args:
            job_data: Dictionary containing job fields

        Returns:
            The created job object

        Raises:
            ValueError: If the job cannot be created or retrieved
            Exception: For database errors
        """
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
        """Update a job.

        Note: This differs from the original database function which returned None.
        The repository pattern returns the updated object for consistency.

        Args:
            job_id: ID of the job to update
            updates: Dictionary of fields to update

        Returns:
            The updated job object, or None if job doesn't exist

        Raises:
            Exception: For database errors
        """
        try:
            # Check if job exists first
            existing_job = self.db.get_job(job_id)
            if existing_job is None:
                logger.warning(f"Attempted to update non-existent job: {job_id}")
                return None

            self.db.update_job(job_id, updates)
            # update_job returns None, but the interface expects the updated job object
            updated_job = self.db.get_job(job_id)
            if updated_job is None:
                raise ValueError(f"Job {job_id} disappeared during update operation")
            return updated_job
        except ValueError:
            raise
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
        """List jobs with optional filters.

        Note on user_id type: The repository interface uses string IDs for consistency
        across different storage backends (some databases use UUIDs, others use integers).
        The SQLite implementation converts to int internally.

        Args:
            user_id: Optional user ID as string
            **filters: Additional filters (not used in SQLite implementation)

        Returns:
            List of job dictionaries

        Raises:
            ValueError: If user_id is not a valid integer string
            Exception: For database errors
        """
        try:
            # Note: filters are accepted for interface compatibility but not used in SQLite implementation
            _ = filters

            # Convert string user_id to int if provided
            # This conversion is necessary because the repository interface uses strings
            # for IDs to support different backend storage systems
            user_id_int: int | None = None
            if user_id is not None:
                user_id_int = parse_user_id(user_id)

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
            # Convert string user_id to int for SQLite
            user_id_int = parse_user_id(user_id)

            result: dict[str, Any] | None = self.db.get_user_by_id(user_id_int)
            return result
        except ValueError:
            raise
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
        """Update a user.

        TODO: Implement in sqlite_implementation.py by Q1 2025
        Currently the database layer only supports create_user and get_user.
        This method will be implemented when user profile editing is added to the UI.

        Args:
            user_id: ID of the user to update
            updates: Dictionary of fields to update

        Returns:
            The existing user object (updates not applied)

        Raises:
            NotImplementedError: When proper implementation is needed
        """
        try:
            # Note: Current database module doesn't have an update_user method
            # This is a placeholder implementation
            _ = updates  # Mark as intentionally unused until implementation is added
            user: dict[str, Any] | None = self.db.get_user(user_id)
            if not user:
                logger.warning(f"User {user_id} not found for update")
                return None
            # For now, return the existing user without modifications
            logger.warning("User update not yet implemented in database layer - returning unmodified user")
            return user
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            raise

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user.

        TODO: Implement in sqlite_implementation.py by Q1 2025
        Currently not implemented as user deletion has GDPR implications
        and requires careful handling of related data (jobs, tokens, etc).

        Args:
            user_id: ID of the user to delete

        Returns:
            False (deletion not implemented)

        Raises:
            NotImplementedError: When proper implementation is needed
        """
        try:
            # Note: Current database module doesn't have a delete_user method
            # This is a placeholder implementation
            user = self.db.get_user(user_id)
            if user is None:
                logger.warning(f"Attempted to delete non-existent user: {user_id}")
                return False
            # User deletion requires careful handling of related data
            logger.warning("User deletion not yet implemented - requires GDPR compliance review")
            return False  # Return False since we can't actually delete
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            raise


class SQLiteFileRepository(FileRepository):
    """SQLite implementation of FileRepository."""

    def __init__(self) -> None:
        """Initialize with the local database implementation."""
        self.db = db_impl

    async def add_files_to_job(self, job_id: str, files: list[dict[str, Any]]) -> None:
        """Add files to a job."""
        try:
            self.db.add_files_to_job(job_id, files)
        except Exception as e:
            logger.error(f"Failed to add files to job {job_id}: {e}")
            raise

    async def get_job_files(self, job_id: str, status: str | None = None) -> list[dict[str, Any]]:
        """Get files for a job with optional status filter."""
        try:
            result: list[dict[str, Any]] = self.db.get_job_files(job_id, status)
            return result
        except Exception as e:
            logger.error(f"Failed to get files for job {job_id}: {e}")
            raise

    async def update_file_status(
        self,
        job_id: str,
        file_path: str,
        status: str,
        error: str | None = None,
        chunks_created: int = 0,
        vectors_created: int = 0,
    ) -> None:
        """Update file processing status."""
        try:
            self.db.update_file_status(job_id, file_path, status, error, chunks_created, vectors_created)
        except Exception as e:
            logger.error(f"Failed to update file status for {file_path} in job {job_id}: {e}")
            raise

    async def get_job_total_vectors(self, job_id: str) -> int:
        """Get total vectors created for all files in a job."""
        try:
            result: int = self.db.get_job_total_vectors(job_id)
            return result
        except Exception as e:
            logger.error(f"Failed to get total vectors for job {job_id}: {e}")
            raise

    async def get_duplicate_files_in_collection(self, collection_name: str, content_hashes: list[str]) -> set[str]:
        """Check which content hashes already exist in a collection."""
        try:
            result: set[str] = self.db.get_duplicate_files_in_collection(collection_name, content_hashes)
            return result
        except Exception as e:
            logger.error(f"Failed to check duplicate files in collection {collection_name}: {e}")
            raise


class SQLiteCollectionRepository(CollectionRepository):
    """SQLite implementation of CollectionRepository."""

    def __init__(self) -> None:
        """Initialize with the local database implementation."""
        self.db = db_impl

    async def list_collections(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """List all collections with optional user filter.

        Note: Converts string user_id to int for SQLite compatibility.
        """
        try:
            user_id_int: int | None = None
            if user_id is not None:
                user_id_int = parse_user_id(user_id)

            result: list[dict[str, Any]] = self.db.list_collections(user_id=user_id_int)
            return result
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise

    async def get_collection_details(self, collection_name: str, user_id: str) -> dict[str, Any] | None:
        """Get detailed information for a collection.

        Note: Converts string user_id to int for SQLite compatibility.
        """
        try:
            user_id_int = parse_user_id(user_id)

            result: dict[str, Any] | None = self.db.get_collection_details(collection_name, user_id_int)
            return result
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to get collection details for {collection_name}: {e}")
            raise

    async def get_collection_files(
        self, collection_name: str, user_id: str, page: int = 1, limit: int = 50
    ) -> dict[str, Any]:
        """Get paginated files in a collection.

        Note: Converts string user_id to int for SQLite compatibility.
        """
        try:
            user_id_int = parse_user_id(user_id)

            result: dict[str, Any] = self.db.get_collection_files(collection_name, user_id_int, page, limit)
            return result
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to get files for collection {collection_name}: {e}")
            raise

    async def rename_collection(self, old_name: str, new_name: str, user_id: str) -> bool:
        """Rename a collection.

        Note: Converts string user_id to int for SQLite compatibility.
        """
        try:
            user_id_int = parse_user_id(user_id)

            result: bool = self.db.rename_collection(old_name, new_name, user_id_int)
            return result
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to rename collection from {old_name} to {new_name}: {e}")
            raise

    async def delete_collection(self, collection_name: str, user_id: str) -> dict[str, Any]:
        """Delete a collection and return deletion info.

        Note: Converts string user_id to int for SQLite compatibility.
        """
        try:
            user_id_int = parse_user_id(user_id)

            result: dict[str, Any] = self.db.delete_collection(collection_name, user_id_int)
            return result
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise

    async def get_collection_metadata(self, collection_name: str) -> dict[str, Any] | None:
        """Get metadata from the first job of a collection."""
        try:
            result: dict[str, Any] | None = self.db.get_collection_metadata(collection_name)
            return result
        except Exception as e:
            logger.error(f"Failed to get metadata for collection {collection_name}: {e}")
            raise


class SQLiteAuthRepository(AuthRepository):
    """SQLite implementation of AuthRepository."""

    def __init__(self) -> None:
        """Initialize with the local database implementation."""
        self.db = db_impl

    async def save_refresh_token(self, user_id: str, token_hash: str, expires_at: Any) -> None:
        """Save a refresh token for a user.

        Note: Converts string user_id to int for SQLite compatibility.
        """
        try:
            # Import datetime to handle the type
            from datetime import datetime

            user_id_int = parse_user_id(user_id)

            if isinstance(expires_at, str):
                # Convert string to datetime if needed
                expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))

            self.db.save_refresh_token(user_id_int, token_hash, expires_at)
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to save refresh token: {e}")
            raise

    async def verify_refresh_token(self, token: str) -> str | None:
        """Verify a refresh token and return user_id if valid.

        Note: Returns string user_id for repository interface consistency.
        """
        try:
            user_id_int: int | None = self.db.verify_refresh_token(token)
            return str(user_id_int) if user_id_int is not None else None
        except Exception as e:
            logger.error(f"Failed to verify refresh token: {e}")
            raise

    async def revoke_refresh_token(self, token: str) -> None:
        """Revoke a refresh token."""
        try:
            self.db.revoke_refresh_token(token)
        except Exception as e:
            logger.error(f"Failed to revoke refresh token: {e}")
            raise

    async def update_user_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp.

        Note: Converts string user_id to int for SQLite compatibility.
        """
        try:
            user_id_int = parse_user_id(user_id)

            self.db.update_user_last_login(user_id_int)
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
            raise
