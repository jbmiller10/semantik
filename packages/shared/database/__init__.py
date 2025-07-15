"""
Shared database utilities and abstractions.

This module provides database-related functionality that's shared across
the vecpipe and webui packages.

The repository pattern provides an abstraction layer over the database,
making it easier to switch between different storage backends (SQLite, PostgreSQL, etc).

Import Organization:
- Repository Pattern (Recommended): Interfaces, implementations, and factories
- Legacy Functions (Deprecated): Direct database functions with deprecation warnings
- Utilities: Database initialization, password hashing, and metadata management
"""

from .base import BaseRepository, JobRepository, UserRepository
from .collection_metadata import ensure_metadata_collection, store_collection_metadata
from .collection_metadata import get_collection_metadata as get_collection_metadata_qdrant
from .factory import create_job_repository, create_user_repository

# Import legacy database functions with deprecation warnings
from .legacy_wrappers import (
    # File operations
    add_files_to_job,
    # Job operations
    create_job,
    # User operations
    create_user,
    delete_collection,
    delete_job,
    get_collection_details,
    get_collection_files,
    get_collection_metadata,
    get_database_stats,
    get_duplicate_files_in_collection,
    get_job,
    get_job_files,
    get_job_total_vectors,
    get_user,
    get_user_by_id,
    # Database management
    init_db,
    # Collection operations
    list_collections,
    list_jobs,
    rename_collection,
    reset_database,
    revoke_refresh_token,
    # Token operations
    save_refresh_token,
    update_file_status,
    update_job,
    update_user_last_login,
    verify_refresh_token,
)

# Import database constants directly
from .sqlite_implementation import DB_PATH, pwd_context
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
    "get_collection_metadata_qdrant",
    "store_collection_metadata",
    # Legacy database functions (to be removed)
    "init_db",
    "reset_database",
    "get_database_stats",
    "DB_PATH",
    "pwd_context",
    "create_job",
    "update_job",
    "get_job",
    "list_jobs",
    "delete_job",
    "get_job_total_vectors",
    "add_files_to_job",
    "get_job_files",
    "update_file_status",
    "list_collections",
    "get_collection_details",
    "get_collection_files",
    "rename_collection",
    "delete_collection",
    "get_duplicate_files_in_collection",
    "get_collection_metadata",
    "create_user",
    "get_user",
    "get_user_by_id",
    "update_user_last_login",
    "save_refresh_token",
    "verify_refresh_token",
    "revoke_refresh_token",
]
