"""Repository factory for dependency injection.

This module provides factory functions to create repository instances,
allowing for easy switching between different implementations.
"""

from typing import Any

from .base import JobRepository, UserRepository
from .sqlite_repository import SQLiteJobRepository, SQLiteUserRepository


def create_job_repository(database_module: Any | None = None) -> JobRepository:
    """Create a job repository instance.

    Args:
        database_module: The database module to use (for SQLite implementation)

    Returns:
        JobRepository instance

    Note:
        In the future, this can check configuration to decide whether to
        return SQLiteJobRepository or PostgreSQLJobRepository.
    """
    if database_module is None:
        # Lazy import to avoid circular dependency
        from webui import database

        database_module = database

    return SQLiteJobRepository(database_module)


def create_user_repository(database_module: Any | None = None) -> UserRepository:
    """Create a user repository instance.

    Args:
        database_module: The database module to use (for SQLite implementation)

    Returns:
        UserRepository instance
    """
    if database_module is None:
        # Lazy import to avoid circular dependency
        from webui import database

        database_module = database

    return SQLiteUserRepository(database_module)


# Future implementation example:
# def create_repositories(config: Config) -> tuple[JobRepository, UserRepository]:
#     """Create all repositories based on configuration."""
#     if config.DATABASE_TYPE == "postgresql":
#         return PostgreSQLJobRepository(), PostgreSQLUserRepository()
#     else:
#         return SQLiteJobRepository(), SQLiteUserRepository()
