"""Repository factory for dependency injection.

This module provides factory functions to create repository instances,
allowing for easy switching between different implementations.
"""

from .base import AuthRepository, CollectionRepository, FileRepository, JobRepository, UserRepository
from .sqlite_repository import SQLiteAuthRepository, SQLiteUserRepository


def create_job_repository() -> JobRepository:
    """Create a job repository instance.

    Returns:
        JobRepository instance

    Note:
        Jobs have been replaced by operations in the new schema.
        This function is deprecated and will be removed.
    """
    raise NotImplementedError("JobRepository is deprecated. Use OperationRepository instead.")


def create_user_repository() -> UserRepository:
    """Create a user repository instance.

    Returns:
        UserRepository instance
    """
    return SQLiteUserRepository()


def create_file_repository() -> FileRepository:
    """Create a file repository instance.

    Returns:
        FileRepository instance

    Note:
        Files have been replaced by documents in the new schema.
        This function is deprecated and will be removed.
    """
    raise NotImplementedError("FileRepository is deprecated. Use DocumentRepository instead.")


def create_collection_repository() -> CollectionRepository:
    """Create a collection repository instance.

    Returns:
        CollectionRepository instance

    Note:
        The new CollectionRepository uses SQLAlchemy.
        The old SQLite implementation has been removed.
    """
    raise NotImplementedError(
        "SQLite CollectionRepository is deprecated. Use packages.shared.database.repositories.CollectionRepository instead."
    )


def create_auth_repository() -> AuthRepository:
    """Create an auth repository instance.

    Returns:
        AuthRepository instance
    """
    return SQLiteAuthRepository()


def create_all_repositories() -> dict[str, object]:
    """Create all repository instances.

    Returns:
        Dictionary mapping repository names to instances

    Note:
        Some repositories are deprecated and will raise NotImplementedError.
    """
    return {
        "user": create_user_repository(),
        "auth": create_auth_repository(),
        # Job, File, and Collection repositories are deprecated
    }
