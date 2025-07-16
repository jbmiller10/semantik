"""SQLite implementation of repository interfaces.

This module provides SQLite-specific implementations of the repository interfaces.
These implementations wrap the existing database functions to provide a clean
abstraction layer that can be replaced in the future.
"""

import logging
from typing import Any

from . import sqlite_implementation as db_impl
from .base import AuthRepository, CollectionRepository, FileRepository, JobRepository, UserRepository
from .exceptions import (
    AccessDeniedError,
    DatabaseOperationError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidUserIdError,
)
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
        except InvalidUserIdError:
            # Re-raise InvalidUserIdError (which is also a ValueError)
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
        except InvalidUserIdError:
            # Re-raise InvalidUserIdError (which is also a ValueError)
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
    """SQLite implementation of UserRepository.

    This is a wrapper around the existing database functions,
    providing an async interface that matches the repository pattern.
    """

    def __init__(self) -> None:
        """Initialize with the local database implementation."""
        self.db = db_impl

    async def create_user(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new user.

        Args:
            user_data: Dictionary containing user fields (username, email, hashed_password, full_name)

        Returns:
            The created user object including the generated ID

        Raises:
            EntityAlreadyExistsError: If username already exists
            DatabaseOperationError: For database errors
        """
        try:
            # Extract fields with defaults
            username = user_data["username"]
            email = user_data.get("email", "")
            hashed_password = user_data["hashed_password"]
            full_name = user_data.get("full_name")

            # The create_user function returns the full user object
            return self.db.create_user(username, email, hashed_password, full_name)
        except ValueError:
            # Re-raise ValueError to maintain backward compatibility with auth API
            raise
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise DatabaseOperationError("create", "user", str(e)) from e

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Get a user by ID."""
        return await self.get_user_by_id(user_id)

    async def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        """Get a user by ID.

        Args:
            user_id: Numeric user ID as a string (e.g., "123")

        Returns:
            User dictionary or None if not found

        Raises:
            InvalidUserIdError: If user_id is not numeric
            DatabaseOperationError: For database errors
        """
        try:
            # Convert string user_id to int for SQLite
            user_id_int = parse_user_id(user_id)

            result: dict[str, Any] | None = self.db.get_user_by_id(user_id_int)
            return result
        except InvalidUserIdError:
            # Re-raise InvalidUserIdError directly (which is also a ValueError)
            raise
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            raise DatabaseOperationError("retrieve", "user", str(e)) from e

    async def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Get a user by username."""
        try:
            # The sqlite implementation uses get_user for username lookup
            result: dict[str, Any] | None = self.db.get_user(username)
            return result
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            raise

    async def update_user(self, user_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:  # noqa: ARG002
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
        # For now, just return the existing user without modifications
        return await self.get_user_by_id(user_id)

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user.

        Args:
            user_id: ID of the user to delete

        Returns:
            True if user was deleted, False if not found

        Raises:
            InvalidUserIdError: If user_id is not numeric
            NotImplementedError: Always (not supported in SQLite backend)
        """
        # Validate user_id format
        parse_user_id(user_id)

        # SQLite backend doesn't support user deletion
        raise NotImplementedError("User deletion is not supported in SQLite backend")

    async def verify_password(self, username: str, password: str) -> dict[str, Any] | None:
        """Verify user password and return user data if valid.

        Args:
            username: The username to check
            password: The plain text password to verify

        Returns:
            User dictionary if credentials are valid, None otherwise

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            # Get user by username
            user = await self.get_user_by_username(username)
            if not user:
                return None

            # Verify password using pwd_context
            if self.db.pwd_context.verify(password, user["hashed_password"]):
                return user
            return None
        except Exception as e:
            logger.error(f"Failed to verify password for {username}: {e}")
            raise DatabaseOperationError("verify", "password", str(e)) from e

    async def list_users(
        self,
        **filters: Any,  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        """List all users.

        Note: SQLite implementation doesn't have a list_users method.
        This is a stub that returns an empty list.

        Args:
            **filters: Filter parameters (all ignored in SQLite implementation)

        Returns:
            Empty list (not implemented in SQLite backend)
        """
        # SQLite backend doesn't support listing users
        return []

    async def update_last_login(self, user_id: str) -> None:
        """Update the last login timestamp for a user.

        Args:
            user_id: ID of the user

        Raises:
            InvalidUserIdError: If user_id is not numeric
            DatabaseOperationError: For database errors
        """
        try:
            user_id_int = parse_user_id(user_id)

            self.db.update_user_last_login(user_id_int)
        except InvalidUserIdError:
            # Re-raise InvalidUserIdError (which is also a ValueError)
            raise
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
            raise


class SQLiteAuthRepository(AuthRepository):
    """SQLite implementation of AuthRepository.

    This handles authentication tokens and related operations.
    Note: The SQLite backend uses refresh tokens instead of regular tokens.
    """

    def __init__(self) -> None:
        """Initialize with the local database implementation."""
        self.db = db_impl

    async def save_refresh_token(self, user_id: str, token_hash: str, expires_at: Any) -> None:
        """Save a refresh token for a user.

        Args:
            user_id: ID of the user
            token_hash: Hashed token
            expires_at: Expiration datetime

        Raises:
            InvalidUserIdError: If user_id is not numeric
            DatabaseOperationError: For database errors
        """
        try:
            user_id_int = parse_user_id(user_id)
            self.db.save_refresh_token(user_id_int, token_hash, expires_at)
        except InvalidUserIdError:
            raise
        except Exception as e:
            logger.error(f"Failed to save refresh token: {e}")
            raise DatabaseOperationError("save", "refresh token", str(e)) from e

    async def verify_refresh_token(self, token: str) -> str | None:
        """Verify a refresh token and return user_id if valid.

        Args:
            token: The refresh token

        Returns:
            User ID as string or None if invalid

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            user_id = self.db.verify_refresh_token(token)
            return str(user_id) if user_id else None
        except Exception as e:
            logger.error(f"Failed to verify refresh token: {e}")
            raise DatabaseOperationError("verify", "refresh token", str(e)) from e

    async def revoke_refresh_token(self, token: str) -> None:
        """Revoke a refresh token.

        Args:
            token: The token to revoke

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            self.db.revoke_refresh_token(token)
        except Exception as e:
            logger.error(f"Failed to revoke refresh token: {e}")
            raise DatabaseOperationError("revoke", "refresh token", str(e)) from e

    async def update_user_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp.

        Args:
            user_id: ID of the user

        Raises:
            InvalidUserIdError: If user_id is not numeric
            DatabaseOperationError: For database errors
        """
        try:
            user_id_int = parse_user_id(user_id)
            self.db.update_user_last_login(user_id_int)
        except InvalidUserIdError:
            raise
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
            raise DatabaseOperationError("update", "last login", str(e)) from e

    async def create_token(self, user_id: str, token: str, expires_at: str) -> None:
        """Store an authentication token.

        Note: SQLite backend doesn't support this directly.
        Tokens are managed differently in the SQLite implementation.

        Args:
            user_id: ID of the user
            token: The token string
            expires_at: ISO format expiration timestamp

        Raises:
            NotImplementedError: Always (not supported in SQLite backend)
        """
        raise NotImplementedError("Token storage is not supported in SQLite backend")

    async def get_token_user_id(self, token: str) -> str | None:
        """Get the user ID associated with a token.

        Args:
            token: The token string

        Returns:
            User ID as string or None if token not found/expired

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            # Try to verify as refresh token
            user_id: int | None = self.db.verify_refresh_token(token)
            return str(user_id) if user_id else None
        except Exception as e:
            logger.error(f"Failed to get token user ID: {e}")
            raise DatabaseOperationError("retrieve", "token", str(e)) from e

    async def delete_token(self, token: str) -> None:
        """Delete a token (logout).

        Args:
            token: The token to delete

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            # Revoke refresh token
            self.db.revoke_refresh_token(token)
        except Exception as e:
            logger.error(f"Failed to delete token: {e}")
            raise DatabaseOperationError("delete", "token", str(e)) from e

    async def delete_user_tokens(self, user_id: str) -> None:
        """Delete all tokens for a user.

        Args:
            user_id: ID of the user

        Raises:
            InvalidUserIdError: If user_id is not numeric
            NotImplementedError: Always (not supported in SQLite backend)
        """
        # Validate user_id format
        parse_user_id(user_id)

        # SQLite backend doesn't have this method
        raise NotImplementedError("Deleting all user tokens is not supported in SQLite backend")


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
    """SQLite implementation of CollectionRepository.

    Note on Transactions:
    The underlying SQLite implementation already handles transactions for
    operations that modify multiple tables (like delete_collection).

    For custom transaction handling, you can use the transaction support:

    Example:
        from shared.database import async_sqlite_transaction

        async with async_sqlite_transaction() as conn:
            cursor = conn.cursor()
            # Perform multiple operations atomically
            cursor.execute("DELETE FROM files WHERE job_id = ?", (job_id,))
            cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            # Both deletes are committed together
    """

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
        except InvalidUserIdError:
            # Re-raise InvalidUserIdError (which is also a ValueError)
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
        except InvalidUserIdError:
            # Re-raise InvalidUserIdError (which is also a ValueError)
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
        except InvalidUserIdError:
            # Re-raise InvalidUserIdError (which is also a ValueError)
            raise
        except Exception as e:
            logger.error(f"Failed to get files for collection {collection_name}: {e}")
            raise

    async def rename_collection(self, old_name: str, new_name: str, user_id: str) -> bool:
        """Rename a collection.

        Note: Converts string user_id to int for SQLite compatibility.
        The underlying implementation already validates that:
        - The user owns at least one job in the collection
        - The new name doesn't already exist

        Args:
            old_name: Current collection name
            new_name: New collection name
            user_id: User ID as string

        Returns:
            True if successful, False if validation failed

        Raises:
            InvalidUserIdError: If user_id is not numeric
            EntityAlreadyExistsError: If new_name already exists
            AccessDeniedError: If user doesn't own the collection
            DatabaseOperationError: For database errors
        """
        try:
            user_id_int = parse_user_id(user_id)

            # First check if the collection exists and user has access
            collection_details = self.db.get_collection_details(old_name, user_id_int)
            if collection_details is None:
                # Either collection doesn't exist or user doesn't have access
                # Check if collection exists at all
                all_collections = self.db.list_collections()
                collection_exists = any(c["name"] == old_name for c in all_collections)

                if collection_exists:
                    raise AccessDeniedError(user_id, "collection", old_name)
                raise EntityNotFoundError("collection", old_name)

            # The rename_collection method returns False if new name already exists
            # or if user doesn't have access. We need to distinguish these cases.
            result: bool = self.db.rename_collection(old_name, new_name, user_id_int)

            if not result:
                # Check if it failed due to duplicate name
                all_collections = self.db.list_collections()
                name_exists = any(c["name"] == new_name for c in all_collections)

                if name_exists:
                    raise EntityAlreadyExistsError("collection", new_name)
                # This shouldn't happen since we already checked access above
                raise AccessDeniedError(user_id, "collection", old_name)

            return result
        except (InvalidUserIdError, EntityAlreadyExistsError, AccessDeniedError, EntityNotFoundError):
            # Re-raise our domain exceptions
            raise
        except Exception as e:
            logger.error(f"Failed to rename collection from {old_name} to {new_name}: {e}")
            raise DatabaseOperationError("rename", "collection", str(e)) from e

    async def delete_collection(self, collection_name: str, user_id: str) -> dict[str, Any]:
        """Delete a collection and return deletion info.

        Note: Converts string user_id to int for SQLite compatibility.
        """
        try:
            user_id_int = parse_user_id(user_id)

            result: dict[str, Any] = self.db.delete_collection(collection_name, user_id_int)
            return result
        except InvalidUserIdError:
            # Re-raise InvalidUserIdError (which is also a ValueError)
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
