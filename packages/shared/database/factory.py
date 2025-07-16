"""Repository factory for dependency injection.

This module provides factory functions to create repository instances,
allowing for easy switching between different implementations.
"""

from .base import AuthRepository, CollectionRepository, FileRepository, JobRepository, UserRepository
from .sqlite_repository import (
    SQLiteAuthRepository,
    SQLiteCollectionRepository,
    SQLiteFileRepository,
    SQLiteJobRepository,
    SQLiteUserRepository,
)


def create_job_repository() -> JobRepository:
    """Create a job repository instance.

    Returns:
        JobRepository instance

    Note:
        This returns the SQLite implementation for backward compatibility.
        Jobs have been replaced by operations in the new schema.
        This function will be removed in a future phase.
    """
    return SQLiteJobRepository()


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
        This returns the SQLite implementation for backward compatibility.
        Files have been replaced by documents in the new schema.
        This function will be removed in a future phase.
    """
    return SQLiteFileRepository()


def create_collection_repository() -> CollectionRepository:
    """Create a collection repository instance.

    Returns:
        CollectionRepository instance

    Note:
        This returns the SQLite implementation for backward compatibility.
        The new CollectionRepository uses SQLAlchemy and should be used for new code.
        This function will be removed in a future phase.
    """
    return SQLiteCollectionRepository()


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
        Job, File, and Collection repositories are using the old SQLite implementation
        for backward compatibility. They will be replaced in a future phase.
    """
    return {
        "job": create_job_repository(),
        "user": create_user_repository(),
        "file": create_file_repository(),
        "collection": create_collection_repository(),
        "auth": create_auth_repository(),
    }
