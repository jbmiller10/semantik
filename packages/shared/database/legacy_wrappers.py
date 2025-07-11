"""
Wrapper functions for legacy database functions with deprecation warnings.

These wrappers add deprecation warnings to encourage migration to the repository pattern.
"""

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

from . import sqlite_implementation as db_impl

# Type variable for preserving function signatures
F = TypeVar("F", bound=Callable[..., Any])


def deprecated(message: str) -> Callable[[F], F]:
    """Decorator to mark functions as deprecated."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(f"{func.__name__} is deprecated. {message}", DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# Job operations with deprecation warnings
@deprecated("Use create_job_repository().create_job() instead")
def create_job(job_data: dict[str, Any]) -> str:
    """Create a new job in the database."""
    return db_impl.create_job(job_data)


@deprecated("Use create_job_repository().update_job() instead")
def update_job(job_id: str, updates: dict[str, Any]) -> None:
    """Update a job in the database."""
    return db_impl.update_job(job_id, updates)


@deprecated("Use create_job_repository().get_job() instead")
def get_job(job_id: str) -> dict[str, Any] | None:
    """Get a job by ID."""
    return db_impl.get_job(job_id)


@deprecated("Use create_job_repository().list_jobs() instead")
def list_jobs(user_id: int | None = None) -> list[dict[str, Any]]:
    """List all jobs."""
    return db_impl.list_jobs(user_id)


@deprecated("Use create_job_repository().delete_job() instead")
def delete_job(job_id: str) -> None:
    """Delete a job."""
    return db_impl.delete_job(job_id)


# User operations with deprecation warnings
@deprecated("Use create_user_repository().create_user() instead")
def create_user(**kwargs: Any) -> dict[str, Any]:
    """Create a new user."""
    return db_impl.create_user(**kwargs)


@deprecated("Use create_user_repository().get_user() instead")
def get_user(username: str) -> dict[str, Any] | None:
    """Get a user by username."""
    return db_impl.get_user(username)


@deprecated("Use create_user_repository().get_user() instead")
def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    """Get a user by ID."""
    return db_impl.get_user_by_id(user_id)


# Collection operations with deprecation warnings
@deprecated("Use repository pattern for collection operations")
def list_collections(user_id: int | None = None) -> list[dict[str, Any]]:
    """List all collections."""
    return db_impl.list_collections(user_id)


@deprecated("Use repository pattern for collection operations")
def get_collection_details(collection_name: str, user_id: int) -> dict[str, Any] | None:
    """Get detailed info for a collection."""
    return db_impl.get_collection_details(collection_name, user_id)


@deprecated("Use repository pattern for collection operations")
def get_collection_files(collection_name: str, user_id: int, page: int = 1, limit: int = 10) -> dict[str, Any]:
    """Get files in a collection."""
    return db_impl.get_collection_files(collection_name, user_id, page, limit)


@deprecated("Use repository pattern for collection operations")
def rename_collection(old_name: str, new_name: str, user_id: int) -> bool:
    """Rename a collection."""
    return db_impl.rename_collection(old_name, new_name, user_id)


@deprecated("Use repository pattern for collection operations")
def delete_collection(collection_name: str, user_id: int) -> dict[str, Any]:
    """Delete a collection."""
    return db_impl.delete_collection(collection_name, user_id)


# File operations with deprecation warnings
@deprecated("Use repository pattern for file operations")
def add_files_to_job(job_id: str, files: list[dict[str, Any]]) -> None:
    """Add files to a job."""
    return db_impl.add_files_to_job(job_id, files)


@deprecated("Use repository pattern for file operations")
def get_job_files(job_id: str, status: str | None = None) -> list[dict[str, Any]]:
    """Get files for a job."""
    return db_impl.get_job_files(job_id, status)


@deprecated("Use repository pattern for file operations")
def update_file_status(
    job_id: str,
    file_path: str,
    status: str,
    error: str | None = None,
    chunks_created: int = 0,
    vectors_created: int = 0,
) -> None:
    """Update file status."""
    return db_impl.update_file_status(job_id, file_path, status, error, chunks_created, vectors_created)


@deprecated("Use repository pattern for file operations")
def get_job_total_vectors(job_id: str) -> int:
    """Get total vectors for a job."""
    return db_impl.get_job_total_vectors(job_id)


@deprecated("Use repository pattern for file operations")
def get_duplicate_files_in_collection(collection_name: str, content_hashes: list[str]) -> set[str]:
    """Check which content hashes already exist in a collection."""
    return db_impl.get_duplicate_files_in_collection(collection_name, content_hashes)


@deprecated("Use repository pattern for collection operations")
def get_collection_metadata(collection_name: str) -> dict[str, Any] | None:
    """Get the metadata from the first job of a collection."""
    return db_impl.get_collection_metadata(collection_name)


# Auth operations with deprecation warnings
@deprecated("Use repository pattern for auth operations")
def update_user_last_login(user_id: int) -> None:
    """Update user's last login time."""
    return db_impl.update_user_last_login(user_id)


@deprecated("Use repository pattern for auth operations")
def save_refresh_token(user_id: int, token_hash: str, expires_at: Any) -> None:
    """Save a refresh token."""
    # Import datetime to handle the type
    from datetime import datetime

    if isinstance(expires_at, str):
        # Convert string to datetime if needed
        expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    return db_impl.save_refresh_token(user_id, token_hash, expires_at)


@deprecated("Use repository pattern for auth operations")
def verify_refresh_token(token: str) -> int | None:
    """Verify a refresh token."""
    return db_impl.verify_refresh_token(token)


@deprecated("Use repository pattern for auth operations")
def revoke_refresh_token(token: str) -> None:
    """Revoke a refresh token."""
    return db_impl.revoke_refresh_token(token)


# Database management operations (these might not need deprecation)
def init_db() -> None:
    """Initialize the database."""
    return db_impl.init_db()


def reset_database() -> None:
    """Reset the database."""
    return db_impl.reset_database()


def get_database_stats() -> dict[str, Any]:
    """Get database statistics."""
    return db_impl.get_database_stats()
