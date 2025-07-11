"""Repository factory for dependency injection.

This module provides factory functions to create repository instances,
allowing for easy switching between different implementations.
"""

from .base import JobRepository, UserRepository
from .sqlite_repository import SQLiteJobRepository, SQLiteUserRepository


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


# Future implementation example:
# def create_repositories(config: Config) -> tuple[JobRepository, UserRepository]:
#     """Create all repositories based on configuration."""
#     if config.DATABASE_TYPE == "postgresql":
#         return PostgreSQLJobRepository(), PostgreSQLUserRepository()
#     else:
#         return SQLiteJobRepository(), SQLiteUserRepository()
