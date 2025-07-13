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
    add_files_to_job,
    create_job,
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
    init_db,
    list_collections,
    list_jobs,
    pwd_context,
    rename_collection,
    reset_database,
    revoke_refresh_token,
    save_refresh_token,
    update_file_status,
    update_job,
    update_user_last_login,
    verify_refresh_token,
)
from .sqlite_repository import (
    SQLiteAuthRepository,
    SQLiteCollectionRepository,
    SQLiteFileRepository,
    SQLiteJobRepository,
    SQLiteUserRepository,
)
from .transaction import RepositoryTransaction, async_sqlite_transaction, sqlite_transaction
from .utils import parse_user_id

# Connection pooling for workers
try:
    from .connection_pool import get_connection_pool, get_db_connection
except ImportError:
    # Connection pool is optional
    get_connection_pool = None
    get_db_connection = None

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
    # TODO: Remove these once all code is migrated to repository pattern
    "create_job",
    "get_job",
    "update_job",
    "delete_job",
    "list_jobs",
    "add_files_to_job",
    "update_file_status",
    "get_job_files",
    "get_job_total_vectors",
    "get_duplicate_files_in_collection",
    "create_user",
    "get_user",
    "get_user_by_id",
    "update_user_last_login",
    "get_collection_metadata",
    "list_collections",
    "get_collection_details",
    "get_collection_files",
    "rename_collection",
    "delete_collection",
    "save_refresh_token",
    "verify_refresh_token",
    "revoke_refresh_token",
]
