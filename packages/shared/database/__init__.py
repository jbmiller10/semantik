"""
Shared database utilities and abstractions using PostgreSQL.

This module provides database-related functionality that's shared across
the vecpipe and webui packages.

The repository pattern provides an abstraction layer over the database,
using PostgreSQL as the primary storage backend.

Import Organization:
- Repository Pattern: Interfaces, implementations, and factories
- Database Session Management: PostgreSQL async session support
- Utilities: Password hashing, user ID parsing, and metadata management
"""

# Password hashing context
from passlib.context import CryptContext

from .base import ApiKeyRepository, AuthRepository, BaseRepository, CollectionRepository, UserRepository
from .collection_metadata import ensure_metadata_collection, store_collection_metadata
from .collection_metadata import get_collection_metadata as get_collection_metadata_qdrant

# Async database session management
from .database import AsyncSessionLocal, get_db
from .exceptions import (
    AccessDeniedError,
    ConcurrencyError,
    DatabaseOperationError,
    DimensionMismatchError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidStateError,
    InvalidUserIdError,
    RepositoryError,
    TransactionError,
    ValidationError,
)
from .factory import (
    create_all_repositories,
    create_api_key_repository,
    create_auth_repository,
    create_chunk_repository,
    create_collection_repository,
    create_document_repository,
    create_operation_repository,
    create_user_repository,
    get_db_session,
)

# Partition utilities for working with partitioned tables
from .partition_utils import ChunkPartitionHelper, PartitionAwareMixin

# PostgreSQL database management
from .postgres_database import check_postgres_connection, get_postgres_db, pg_connection_manager

# Utilities
from .utils import parse_user_id

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

__all__ = [
    # Repository interfaces
    "BaseRepository",
    "UserRepository",
    "CollectionRepository",
    "AuthRepository",
    "ApiKeyRepository",
    # Factory functions
    "create_all_repositories",
    "create_user_repository",
    "create_collection_repository",
    "create_chunk_repository",
    "create_auth_repository",
    "create_api_key_repository",
    "create_operation_repository",
    "create_document_repository",
    "get_db_session",
    # Domain exceptions
    "AccessDeniedError",
    "ConcurrencyError",
    "DatabaseOperationError",
    "DimensionMismatchError",
    "EntityAlreadyExistsError",
    "EntityNotFoundError",
    "InvalidStateError",
    "InvalidUserIdError",
    "RepositoryError",
    "TransactionError",
    "ValidationError",
    # Collection metadata functions
    "ensure_metadata_collection",
    "get_collection_metadata_qdrant",
    "store_collection_metadata",
    # Utility functions
    "parse_user_id",
    "pwd_context",
    # Partition utilities
    "ChunkPartitionHelper",
    "PartitionAwareMixin",
    # Async database session management
    "AsyncSessionLocal",
    "get_db",
    "get_postgres_db",
    # PostgreSQL connection management
    "pg_connection_manager",
    "check_postgres_connection",
]
