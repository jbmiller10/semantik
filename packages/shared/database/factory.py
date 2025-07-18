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


def create_all_repositories() -> (
    dict[str, JobRepository | UserRepository | FileRepository | CollectionRepository | AuthRepository]
):
    """Create all repository instances at once.

    This function provides a centralized way to create all repositories,
    ensuring consistency and making it easier to manage dependencies.

    Returns:
        Dictionary containing all repository instances with keys:
        - 'job': JobRepository instance
        - 'user': UserRepository instance
        - 'file': FileRepository instance
        - 'collection': CollectionRepository instance
        - 'auth': AuthRepository instance

    Example:
        repos = create_all_repositories()
        job_repo = repos['job']
        user_repo = repos['user']
    """
    return {
        "job": SQLiteJobRepository(),
        "user": SQLiteUserRepository(),
        "file": SQLiteFileRepository(),
        "collection": SQLiteCollectionRepository(),
        "auth": SQLiteAuthRepository(),
    }


# Future implementation with configuration support:
# def create_all_repositories(config: Config | None = None) -> dict[str, Any]:
#     """Create all repositories based on configuration."""
#     config = config or get_default_config()
#
#     if config.DATABASE_TYPE == "postgresql":
#         return {
#             'job': PostgreSQLJobRepository(config.DATABASE_URL),
#             'user': PostgreSQLUserRepository(config.DATABASE_URL),
#             'file': PostgreSQLFileRepository(config.DATABASE_URL),
#             'collection': PostgreSQLCollectionRepository(config.DATABASE_URL),
#             'auth': PostgreSQLAuthRepository(config.DATABASE_URL),
#         }
#     else:
#         return {
#             'job': SQLiteJobRepository(),
#             'user': SQLiteUserRepository(),
#             'file': SQLiteFileRepository(),
#             'collection': SQLiteCollectionRepository(),
#             'auth': SQLiteAuthRepository(),
#         }
