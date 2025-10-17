#!/usr/bin/env python3
"""
SQLAlchemy declarative models for the Semantik database.

This module defines the database schema using SQLAlchemy's declarative mapping.
These models are used by Alembic for migrations and can be used for ORM operations.

Note on Timestamps:
All DateTime fields use timezone=True to ensure consistent timezone-aware datetime
handling across the application. This allows proper storage and retrieval of
timezones with datetime values, preventing timezone-related bugs.

We use DateTime columns for new tables but maintain String columns for existing
user-related tables for backward compatibility. A future migration could convert
these to proper DateTime columns.

Partitioned Tables:
The chunks table is partitioned by HASH(collection_id) to improve performance and
scalability. When working with partitioned tables:

1. ALWAYS include the partition key (collection_id) in WHERE clauses
2. Group bulk operations by partition key for efficiency
3. Be aware that cross-partition queries are expensive
4. The partition key must be part of any unique constraint or primary key

See the Chunk model and packages.shared.database.partition_utils for detailed
examples and utilities for working with partitioned tables.
"""

import enum
from typing import Any, cast

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# Create the declarative base
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""


# Enums
class DocumentStatus(str, enum.Enum):
    """Status of document processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class PermissionType(str, enum.Enum):
    """Types of permissions for collections."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class CollectionStatus(str, enum.Enum):
    """Status of a collection."""

    PENDING = "pending"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    DEGRADED = "degraded"


class OperationStatus(str, enum.Enum):
    """Status of an operation."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def _missing_(cls, value: Any) -> "OperationStatus | None":
        """Provide case-insensitive lookup for enum values.

        This allows constructing OperationStatus from values like "PROCESSING"
        that appear in some test helpers, while keeping canonical values
        lowercase in the database.
        """
        if isinstance(value, str):
            result = cls.__members__.get(value.upper()) or cls._value2member_map_.get(value.lower())
            return cast("OperationStatus | None", result)
        return None


class OperationType(str, enum.Enum):
    """Types of collection operations."""

    INDEX = "index"
    APPEND = "append"
    REINDEX = "reindex"
    REMOVE_SOURCE = "remove_source"
    DELETE = "delete"


class User(Base):
    """User model for authentication."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now())
    last_login = Column(DateTime(timezone=True))

    # Relationships
    collections = relationship("Collection", back_populates="owner", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    permissions = relationship("CollectionPermission", back_populates="user", cascade="all, delete-orphan")
    operations = relationship("Operation", back_populates="user")
    audit_logs = relationship("CollectionAuditLog", back_populates="user")


class Collection(Base):
    """Collection model for organizing documents."""

    __tablename__ = "collections"

    id = Column(String, primary_key=True)  # UUID as string
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    vector_store_name = Column(String, unique=True, nullable=False)  # Qdrant collection name
    embedding_model = Column(String, nullable=False)
    quantization = Column(String, nullable=False, default="float16")  # float32, float16, int8
    chunk_size = Column(Integer, nullable=False, default=1000)
    chunk_overlap = Column(Integer, nullable=False, default=200)
    chunking_strategy = Column(String, nullable=True)  # New field for chunking strategy type
    chunking_config = Column(JSON, nullable=True)  # New field for strategy-specific configuration
    is_public = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    meta = Column(JSON)

    # New fields from second migration
    status = Column(
        Enum(CollectionStatus, name="collection_status", native_enum=True, create_constraint=False),
        nullable=False,
        default=CollectionStatus.PENDING,
        index=True,
    )  # type: ignore[var-annotated]
    status_message = Column(Text)
    qdrant_collections = Column(JSON)  # List of Qdrant collection names
    qdrant_staging = Column(JSON)  # Staging collections during reindex
    document_count = Column(Integer, nullable=False, default=0)
    vector_count = Column(Integer, nullable=False, default=0)
    total_size_bytes = Column(BigInteger, nullable=False, default=0)

    # New chunking-related fields
    default_chunking_config_id = Column(Integer, ForeignKey("chunking_configs.id"), nullable=True, index=True)
    chunks_total_count = Column(Integer, nullable=False, default=0)
    chunking_completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    owner = relationship("User", back_populates="collections")
    documents = relationship("Document", back_populates="collection", cascade="all, delete-orphan")
    permissions = relationship("CollectionPermission", back_populates="collection", cascade="all, delete-orphan")
    sources = relationship("CollectionSource", back_populates="collection", cascade="all, delete-orphan")
    operations = relationship("Operation", back_populates="collection", cascade="all, delete-orphan")
    audit_logs = relationship("CollectionAuditLog", back_populates="collection", cascade="all, delete-orphan")
    resource_limits = relationship(
        "CollectionResourceLimits", back_populates="collection", uselist=False, cascade="all, delete-orphan"
    )
    default_chunking_config = relationship(
        "ChunkingConfig", foreign_keys=[default_chunking_config_id], back_populates="collections"
    )
    chunks = relationship("Chunk", back_populates="collection", cascade="all, delete-orphan")


class Document(Base):
    """Document model representing documents in collections."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True)  # UUID as string
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id = Column(Integer, ForeignKey("collection_sources.id"), nullable=True, index=True)
    file_path = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String)
    content_hash = Column(String, nullable=False, index=True)
    status = Column(
        Enum(DocumentStatus, name="document_status", native_enum=True, create_constraint=False),
        nullable=False,
        default=DocumentStatus.PENDING,
        index=True,
    )  # type: ignore[var-annotated]
    error_message = Column(Text)
    chunk_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    meta = Column(JSON)

    # New chunking-related fields
    chunking_config_id = Column(Integer, ForeignKey("chunking_configs.id"), nullable=True, index=True)
    chunks_count = Column(Integer, nullable=False, default=0)
    chunking_started_at = Column(DateTime(timezone=True), nullable=True)
    chunking_completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    collection = relationship("Collection", back_populates="documents")
    source = relationship("CollectionSource", back_populates="documents")
    chunking_config = relationship("ChunkingConfig", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("ix_documents_collection_content_hash", "collection_id", "content_hash", unique=True),
        Index("ix_documents_collection_id_chunking_completed_at", "collection_id", "chunking_completed_at"),
    )


class ApiKey(Base):
    """API key model for programmatic access."""

    __tablename__ = "api_keys"

    id = Column(String, primary_key=True)  # UUID as string
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)
    key_hash = Column(String, unique=True, nullable=False, index=True)
    permissions = Column(JSON)  # Store collection access rights
    last_used_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    is_active = Column(Boolean, nullable=False, default=True, index=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")
    collection_permissions = relationship(
        "CollectionPermission", back_populates="api_key", cascade="all, delete-orphan"
    )


class CollectionPermission(Base):
    """Permission model for fine-grained access control."""

    __tablename__ = "collection_permissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    api_key_id = Column(String, ForeignKey("api_keys.id", ondelete="CASCADE"), index=True)
    permission = Column(Enum(PermissionType, name="permission_type"), nullable=False)  # type: ignore[var-annotated]
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

    # Relationships
    collection = relationship("Collection", back_populates="permissions")
    user = relationship("User", back_populates="permissions")
    api_key = relationship("ApiKey", back_populates="collection_permissions")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "(user_id IS NOT NULL AND api_key_id IS NULL) OR (user_id IS NULL AND api_key_id IS NOT NULL)",
            name="check_user_or_api_key",
        ),
        Index(
            "ix_collection_permissions_unique_user",
            "collection_id",
            "user_id",
            unique=True,
            postgresql_where="user_id IS NOT NULL",
        ),
        Index(
            "ix_collection_permissions_unique_api_key",
            "collection_id",
            "api_key_id",
            unique=True,
            postgresql_where="api_key_id IS NOT NULL",
        ),
    )


class RefreshToken(Base):
    """Refresh token model for JWT authentication."""

    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token_hash = Column(String, unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    is_revoked = Column(Boolean, default=False)

    # Relationships
    user = relationship("User", back_populates="refresh_tokens")


class CollectionSource(Base):
    """Source model for tracking collection data sources."""

    __tablename__ = "collection_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    source_path = Column(String, nullable=False)
    source_type = Column(String, nullable=False, default="directory")  # directory, file, url, etc.
    document_count = Column(Integer, nullable=False, default=0)
    size_bytes = Column(Integer, nullable=False, default=0)
    last_indexed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    meta = Column(JSON)

    # Relationships
    collection = relationship("Collection", back_populates="sources")
    documents = relationship("Document", back_populates="source")

    # Constraints
    __table_args__ = (UniqueConstraint("collection_id", "source_path", name="uq_collection_source_path"),)


class Operation(Base):
    """Operation model for tracking async collection operations."""

    __tablename__ = "operations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String, unique=True, nullable=False)  # For external reference
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    type = Column(
        Enum(OperationType, name="operation_type", native_enum=True, create_constraint=False),
        nullable=False,
        index=True,
    )  # type: ignore[var-annotated]
    status = Column(
        Enum(
            OperationStatus,
            name="operation_status",
            native_enum=True,
            create_constraint=False,
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
        default=OperationStatus.PENDING,
        index=True,
    )  # type: ignore[var-annotated]
    task_id = Column(String)  # Celery task ID
    config = Column(JSON, nullable=False)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    meta = Column(JSON)

    # Relationships
    collection = relationship("Collection", back_populates="operations")
    user = relationship("User")
    audit_logs = relationship("CollectionAuditLog", back_populates="operation")
    metrics = relationship("OperationMetrics", back_populates="operation", cascade="all, delete-orphan")


class CollectionAuditLog(Base):
    """Audit log model for tracking collection actions."""

    __tablename__ = "collection_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    operation_id = Column(Integer, ForeignKey("operations.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    action = Column(String, nullable=False)  # created, updated, deleted, reindexed, etc.
    details = Column(JSON)
    ip_address = Column(String)
    user_agent = Column(String)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)

    # Relationships
    collection = relationship("Collection", back_populates="audit_logs")
    operation = relationship("Operation", back_populates="audit_logs")
    user = relationship("User")


class CollectionResourceLimits(Base):
    """Resource limits model for collection quotas."""

    __tablename__ = "collection_resource_limits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, unique=True)
    max_documents = Column(Integer, default=100000)
    max_storage_gb = Column(Float, default=50.0)
    max_operations_per_hour = Column(Integer, default=10)
    max_sources = Column(Integer, default=10)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

    # Relationships
    collection = relationship("Collection", back_populates="resource_limits")


class OperationMetrics(Base):
    """Metrics model for tracking operation performance."""

    __tablename__ = "operation_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    operation_id = Column(Integer, ForeignKey("operations.id", ondelete="CASCADE"), nullable=False, index=True)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)

    # Relationships
    operation = relationship("Operation", back_populates="metrics")


class ChunkingStrategy(Base):
    """Chunking strategy model for document processing."""

    __tablename__ = "chunking_strategies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    version = Column(String, nullable=False, default="1.0.0")
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    meta = Column(JSON)

    # Relationships
    configs = relationship("ChunkingConfig", back_populates="strategy")


class ChunkingConfig(Base):
    """Chunking configuration model (deduplicated)."""

    __tablename__ = "chunking_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey("chunking_strategies.id"), nullable=False, index=True)
    config_hash = Column(String(64), unique=True, nullable=False, index=True)
    config_data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    use_count = Column(Integer, nullable=False, default=0, index=True)
    last_used_at = Column(DateTime(timezone=True))

    # Relationships
    strategy = relationship("ChunkingStrategy", back_populates="configs")
    chunks = relationship("Chunk", back_populates="chunking_config")
    collections = relationship(
        "Collection", foreign_keys="Collection.default_chunking_config_id", back_populates="default_chunking_config"
    )
    documents = relationship("Document", back_populates="chunking_config")


class Chunk(Base):
    """Chunk model for partitioned document storage.

    IMPORTANT: This table is partitioned by LIST(partition_key) in PostgreSQL with 100 partitions.

    Partition Awareness:
    - The table uses LIST partitioning on partition_key (0-99)
    - partition_key is computed via get_partition_key(collection_id) before insert
    - Primary key is (id, collection_id, partition_key) to support partitioning
    - Always include collection_id in WHERE clauses for optimal partition pruning
    - Bulk operations should be grouped by collection_id for efficiency
    - Cross-collection queries will scan multiple partitions (use sparingly)

    Usage Examples:
        # Good - partition pruning enabled
        chunks = session.query(Chunk).filter(
            Chunk.collection_id == collection_id,
            Chunk.document_id == document_id
        ).all()

        # Bad - scans all partitions
        chunks = session.query(Chunk).filter(
            Chunk.document_id == document_id
        ).all()

    Bulk Insert Example:
        # Group chunks by collection_id for efficient partition routing
        chunks_by_collection = {}
        for chunk in chunks_to_insert:
            chunks_by_collection.setdefault(chunk.collection_id, []).append(chunk)

        for collection_id, chunks in chunks_by_collection.items():
            session.bulk_insert_mappings(Chunk, chunks)
    """

    __tablename__ = "chunks"

    # Primary key includes partition key for partitioned table support
    # Note: id is BigInteger with auto-incrementing sequence
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True, nullable=False)
    partition_key = Column(Integer, primary_key=True, nullable=False)

    # Foreign keys and data columns
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=True)  # Can be NULL
    chunking_config_id = Column(Integer, ForeignKey("chunking_configs.id"), nullable=True)  # Can be NULL
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    start_offset = Column(Integer)  # Can be NULL
    end_offset = Column(Integer)  # Can be NULL
    token_count = Column(Integer)
    embedding_vector_id = Column(String)  # Reference to Qdrant
    meta = Column(
        "metadata", JSON
    )  # Column name is 'metadata' in DB, but 'meta' in Python to avoid SQLAlchemy reserved word
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

    # Relationships
    collection = relationship("Collection", back_populates="chunks")
    document = relationship("Document", back_populates="chunks")
    chunking_config = relationship("ChunkingConfig", back_populates="chunks")

    # Composite indexes optimized for partition pruning
    __table_args__ = (
        # Indexes that exist at the database level
        Index("idx_chunks_part_collection", "collection_id"),  # Per-partition index
        Index("idx_chunks_part_created", "created_at"),  # Per-partition index
        Index("idx_chunks_part_chunk_index", "collection_id", "chunk_index"),  # Per-partition index
        Index("idx_chunks_part_document", "document_id"),  # Per-partition conditional index
        {
            "comment": "Partitioned by LIST(partition_key) with 100 partitions. partition_key is computed via database function.",
            "info": {
                "partition_key": "partition_key",
                "partition_method": "LIST",
                "partition_count": 100,
            },
        },
    )
