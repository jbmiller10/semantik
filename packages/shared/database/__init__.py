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

from .base import AuthRepository, BaseRepository, CollectionRepository, FileRepository, JobRepository, UserRepository
from .collection_metadata import ensure_metadata_collection, store_collection_metadata
from .collection_metadata import get_collection_metadata as get_collection_metadata_qdrant
from .exceptions import (
    AccessDeniedError,
    ConcurrencyError,
    DatabaseOperationError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidUserIdError,
    RepositoryError,
    TransactionError,
    ValidationError,
)
from .factory import (
    create_all_repositories,
    create_auth_repository,
    create_collection_repository,
    create_file_repository,
    create_job_repository,
    create_user_repository,
)

# Import database management functions directly (these don't need the repository pattern)
# Import database constants directly
# Legacy function imports for backward compatibility
# TODO: Migrate tests and code to use repository pattern instead
from .sqlite_implementation import (
    DB_PATH,
    create_user,
    get_database_stats,
    get_user,
    get_user_by_id,
    init_db,
    pwd_context,
    reset_database,
    revoke_refresh_token,
    save_refresh_token,
    update_user_last_login,
    verify_refresh_token,
)
from .sqlite_repository import SQLiteAuthRepository, SQLiteUserRepository
from .transaction import RepositoryTransaction, async_sqlite_transaction, sqlite_transaction
from .utils import parse_user_id

# Connection pooling for workers
try:
    from .connection_pool import get_connection_pool, get_db_connection
except ImportError:
    # Connection pool is optional
    get_connection_pool = None  # type: ignore
    get_db_connection = None  # type: ignore

__all__ = [
    # Repository interfaces
    "BaseRepository",
    "JobRepository",
    "UserRepository",
    "FileRepository",
    "CollectionRepository",
    "AuthRepository",
    # Repository implementations
    "SQLiteUserRepository",
    "SQLiteAuthRepository",
    # Factory functions
    "create_all_repositories",
    "create_job_repository",
    "create_user_repository",
    "create_file_repository",
    "create_collection_repository",
    "create_auth_repository",
    # Domain exceptions
    "AccessDeniedError",
    "ConcurrencyError",
    "DatabaseOperationError",
    "EntityAlreadyExistsError",
    "EntityNotFoundError",
    "InvalidUserIdError",
    "RepositoryError",
    "TransactionError",
    "ValidationError",
    # Transaction support
    "RepositoryTransaction",
    "async_sqlite_transaction",
    "sqlite_transaction",
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
    # Connection pooling (optional, for workers)
    "get_connection_pool",
    "get_db_connection",
    # Legacy function exports (for backward compatibility)
    # NOTE: Job/file/collection functions have been removed as part of collections refactor
    # Only user-related functions remain
    "create_user",
    "get_user",
    "get_user_by_id",
    "update_user_last_login",
    "save_refresh_token",
    "verify_refresh_token",
    "revoke_refresh_token",
]
