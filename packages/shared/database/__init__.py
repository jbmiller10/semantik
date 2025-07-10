"""
Shared database utilities and abstractions.

This module provides database-related functionality that's shared across
the vecpipe and webui packages.

The repository pattern provides an abstraction layer over the database,
making it easier to switch between different storage backends (SQLite, PostgreSQL, etc).
"""

from .base import BaseRepository, JobRepository, UserRepository
from .collection_metadata import ensure_metadata_collection, get_collection_metadata, store_collection_metadata
from .factory import create_job_repository, create_user_repository
from .sqlite_repository import SQLiteJobRepository, SQLiteUserRepository

__all__ = [
    # Repository interfaces
    "BaseRepository",
    "JobRepository",
    "UserRepository",
    # Repository implementations
    "SQLiteJobRepository",
    "SQLiteUserRepository",
    # Factory functions
    "create_job_repository",
    "create_user_repository",
    # Collection metadata functions
    "ensure_metadata_collection",
    "get_collection_metadata",
    "store_collection_metadata",
]
