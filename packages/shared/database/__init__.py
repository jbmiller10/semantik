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

from .base import (
    AuthRepository,
    BaseRepository,
    CollectionRepository,
    FileRepository,
    JobRepository,
    UserRepository,
)
from .collection_metadata import ensure_metadata_collection, store_collection_metadata
from .collection_metadata import get_collection_metadata as get_collection_metadata_qdrant
from .factory import (
    create_auth_repository,
    create_collection_repository,
    create_file_repository,
    create_job_repository,
    create_user_repository,
)

# Import database management functions directly (these don't need the repository pattern)
# Import database constants directly
from .sqlite_implementation import DB_PATH, get_database_stats, init_db, pwd_context, reset_database
from .sqlite_repository import (
    SQLiteAuthRepository,
    SQLiteCollectionRepository,
    SQLiteFileRepository,
    SQLiteJobRepository,
    SQLiteUserRepository,
)
from .utils import parse_user_id

__all__ = [
    # Repository interfaces
    "BaseRepository",
    "JobRepository",
    "UserRepository",
    "FileRepository",
    "CollectionRepository",
    "AuthRepository",
    # Repository implementations
    "SQLiteJobRepository",
    "SQLiteUserRepository",
    "SQLiteFileRepository",
    "SQLiteCollectionRepository",
    "SQLiteAuthRepository",
    # Factory functions
    "create_job_repository",
    "create_user_repository",
    "create_file_repository",
    "create_collection_repository",
    "create_auth_repository",
    # Collection metadata functions
    "ensure_metadata_collection",
    "get_collection_metadata_qdrant",
    "store_collection_metadata",
    # Database management functions (for admin operations)
    "init_db",
    "reset_database",
    "get_database_stats",
    "DB_PATH",
    "pwd_context",
    # Utility functions
    "parse_user_id",
]
