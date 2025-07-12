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
        In the future, this can check configuration to decide whether to
        return SQLiteJobRepository or PostgreSQLJobRepository.
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
    """
    return SQLiteFileRepository()


def create_collection_repository() -> CollectionRepository:
    """Create a collection repository instance.

    Returns:
        CollectionRepository instance
    """
    return SQLiteCollectionRepository()


def create_auth_repository() -> AuthRepository:
    """Create an auth repository instance.

    Returns:
        AuthRepository instance
    """
    return SQLiteAuthRepository()


# Future implementation example:
# def create_repositories(config: Config) -> tuple[JobRepository, UserRepository, FileRepository, CollectionRepository]:
#     """Create all repositories based on configuration."""
#     if config.DATABASE_TYPE == "postgresql":
#         return (
#             PostgreSQLJobRepository(),
#             PostgreSQLUserRepository(),
#             PostgreSQLFileRepository(),
#             PostgreSQLCollectionRepository()
#         )
#     else:
#         return (
#             SQLiteJobRepository(),
#             SQLiteUserRepository(),
#             SQLiteFileRepository(),
#             SQLiteCollectionRepository()
#         )
