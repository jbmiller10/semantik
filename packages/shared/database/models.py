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

See the Chunk model and shared.database.partition_utils for detailed
examples and utilities for working with partitioned tables.
"""

import enum
import sys
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
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, relationship

# Ensure both ``shared.database.models`` and ``packages.shared.database.models``
# resolve to the same module instance. This prevents SQLAlchemy from creating
# duplicate model classes for the same tables when mixed import prefixes are
# used across the codebase or in downstream scripts.
if __name__ in {"shared.database.models", "packages.shared.database.models"}:
    sys.modules.setdefault("shared.database.models", sys.modules[__name__])
    sys.modules.setdefault("packages.shared.database.models", sys.modules[__name__])


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
    PROJECTION_BUILD = "projection_build"
    RETRY_DOCUMENTS = "retry_documents"

    @classmethod
    def _missing_(cls, value: Any) -> "OperationType | None":
        """Provide case-insensitive lookup for enum values."""

        if isinstance(value, str):
            normalized = value.lower()
            resolved = cls._value2member_map_.get(normalized) or cls.__members__.get(value.upper())
            return cast("OperationType | None", resolved)
        return None

    def __str__(self) -> str:  # pragma: no cover - simple helper for driver bindings
        """Return the canonical string value for the enum member."""

        return self.value


class ProjectionRunStatus(str, enum.Enum):
    """Lifecycle states for embedding projection runs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
    mcp_profiles = relationship("MCPProfile", back_populates="owner", cascade="all, delete-orphan")
    llm_provider_config = relationship(
        "LLMProviderConfig",
        back_populates="user",
        uselist=False,  # One-to-one
        cascade="all, delete-orphan",
    )
    preferences = relationship(
        "UserPreferences",
        back_populates="user",
        uselist=False,  # One-to-one
        cascade="all, delete-orphan",
    )


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
    total_size_bytes = Column(Integer, nullable=False, default=0)

    # New chunking-related fields
    default_chunking_config_id = Column(Integer, ForeignKey("chunking_configs.id"), nullable=True, index=True)
    chunks_total_count = Column(Integer, nullable=False, default=0)
    chunking_completed_at = Column(DateTime(timezone=True), nullable=True)

    # Sync policy fields (collection-level sync configuration)
    sync_mode = Column(String(20), nullable=False, default="one_time")  # 'one_time' or 'continuous'
    sync_interval_minutes = Column(Integer, nullable=True)  # Sync interval for continuous mode (min 15)
    sync_paused_at = Column(DateTime(timezone=True), nullable=True)  # NULL = not paused
    sync_next_run_at = Column(DateTime(timezone=True), nullable=True)  # When next sync should run

    # Sync run tracking fields
    sync_last_run_started_at = Column(DateTime(timezone=True), nullable=True)
    sync_last_run_completed_at = Column(DateTime(timezone=True), nullable=True)
    sync_last_run_status = Column(String(20), nullable=True)  # 'running', 'success', 'failed', 'partial'
    sync_last_error = Column(Text, nullable=True)  # Error summary from last failed sync

    # Reranker and extraction config (Phase 2 plugin extensibility)
    default_reranker_id = Column(String, nullable=True)  # Default reranker plugin ID
    extraction_config = Column(
        JSON, nullable=True
    )  # {"enabled": bool, "extractor_ids": [], "types": [], "options": {}}

    # Relationships
    owner = relationship("User", back_populates="collections")
    documents = relationship("Document", back_populates="collection", cascade="all, delete-orphan")
    permissions = relationship("CollectionPermission", back_populates="collection", cascade="all, delete-orphan")
    sources = relationship("CollectionSource", back_populates="collection", cascade="all, delete-orphan")
    operations = relationship("Operation", back_populates="collection", cascade="all, delete-orphan")
    projection_runs = relationship("ProjectionRun", back_populates="collection", cascade="all, delete-orphan")
    audit_logs = relationship("CollectionAuditLog", back_populates="collection", cascade="all, delete-orphan")
    resource_limits = relationship(
        "CollectionResourceLimits", back_populates="collection", uselist=False, cascade="all, delete-orphan"
    )
    default_chunking_config = relationship(
        "ChunkingConfig", foreign_keys=[default_chunking_config_id], back_populates="collections"
    )
    chunks = relationship("Chunk", back_populates="collection", cascade="all, delete-orphan")
    sync_runs = relationship("CollectionSyncRun", back_populates="collection", cascade="all, delete-orphan")
    mcp_profiles = relationship("MCPProfile", secondary="mcp_profile_collections", back_populates="collections")


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

    # Flexible source fields (for non-file-based sources like web, Slack, etc.)
    uri = Column(String, nullable=True)  # Logical identifier (URL, file path, message ID, etc.)
    source_metadata = Column(JSON)  # Connector-specific metadata (headers, timestamps, etc.)

    # New chunking-related fields
    chunking_config_id = Column(Integer, ForeignKey("chunking_configs.id"), nullable=True, index=True)
    chunks_count = Column(Integer, nullable=False, default=0)
    chunking_started_at = Column(DateTime(timezone=True), nullable=True)
    chunking_completed_at = Column(DateTime(timezone=True), nullable=True)

    # Sync tracking fields (for continuous sync "keep last-known" behavior)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)  # When document was last seen during sync
    is_stale = Column(Boolean, nullable=False, default=False)  # Marks documents not seen in recent sync

    # Retry tracking fields (for failed document retry functionality)
    retry_count = Column(Integer, nullable=False, default=0)  # Number of retry attempts
    last_retry_at = Column(DateTime(timezone=True), nullable=True)  # When last retry was attempted
    error_category = Column(String(50), nullable=True)  # 'transient', 'permanent', or 'unknown'

    # Relationships
    collection = relationship("Collection", back_populates="documents")
    source = relationship("CollectionSource", back_populates="documents")
    chunking_config = relationship("ChunkingConfig", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    artifacts = relationship("DocumentArtifact", back_populates="document", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("ix_documents_collection_content_hash", "collection_id", "content_hash", unique=True),
        Index("ix_documents_collection_id_chunking_completed_at", "collection_id", "chunking_completed_at"),
        Index(
            "ix_documents_collection_uri_unique",
            "collection_id",
            "uri",
            unique=True,
            postgresql_where=text("uri IS NOT NULL"),
        ),
        # Index for querying retryable failed documents
        # Note: DocumentStatus enum uses member NAMES (FAILED) not VALUES (failed) in PostgreSQL
        Index(
            "ix_documents_collection_failed_retryable",
            "collection_id",
            "status",
            "error_category",
            "retry_count",
            postgresql_where=text("status = 'FAILED'"),
        ),
    )


class ArtifactKind(str, enum.Enum):
    """Types of document artifacts."""

    PRIMARY = "primary"
    PREVIEW = "preview"
    THUMBNAIL = "thumbnail"


class DocumentArtifact(Base):
    """Artifact model for storing document content in database.

    Used for non-file sources (Git, IMAP, web) where content cannot be
    served from the filesystem. The content endpoint checks for artifacts
    first, then falls back to file serving.

    Artifacts can be:
    - primary: The canonical document content
    - preview: A preview/summary version
    - thumbnail: A thumbnail image (for visual documents)
    """

    __tablename__ = "document_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(
        String,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    collection_id = Column(
        String,
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    artifact_kind = Column(String(20), nullable=False, default="primary")
    mime_type = Column(String(255), nullable=False)
    charset = Column(String(50), nullable=True)
    content_text = Column(Text, nullable=True)  # For text-based content
    content_bytes = Column(LargeBinary, nullable=True)  # For binary content
    content_hash = Column(String(64), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    is_truncated = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

    # Relationships
    document = relationship("Document", back_populates="artifacts")
    collection = relationship("Collection")

    __table_args__ = (
        UniqueConstraint("document_id", "artifact_kind", name="uq_document_artifact_kind"),
        CheckConstraint(
            "content_text IS NOT NULL OR content_bytes IS NOT NULL",
            name="ck_content_present",
        ),
        CheckConstraint(
            "artifact_kind IN ('primary', 'preview', 'thumbnail')",
            name="ck_artifact_kind_values",
        ),
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
    source_type = Column(String, nullable=False)  # directory, web, slack, etc.
    source_config = Column(JSON)  # Connector-specific configuration (e.g. {"path": "..."})
    document_count = Column(Integer, nullable=False, default=0)
    size_bytes = Column(Integer, nullable=False, default=0)
    last_indexed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    meta = Column(JSON)

    # Sync telemetry fields (per-source status tracking)
    # Note: Sync policy (mode, interval, pause) is now at collection level
    last_run_started_at = Column(DateTime(timezone=True), nullable=True)
    last_run_completed_at = Column(DateTime(timezone=True), nullable=True)
    last_run_status = Column(String(20), nullable=True)  # 'success', 'failed', 'partial'
    last_error = Column(Text, nullable=True)  # Error message from last failed sync

    # Relationships
    collection = relationship("Collection", back_populates="sources")
    documents = relationship("Document", back_populates="source")
    secrets = relationship("ConnectorSecret", back_populates="source", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (UniqueConstraint("collection_id", "source_path", name="uq_collection_source_path"),)


class ConnectorSecret(Base):
    """Encrypted secrets for connector authentication.

    Stores encrypted credentials (passwords, tokens, SSH keys) for
    connector sources. Secrets are encrypted using Fernet symmetric
    encryption and are never returned via API responses.

    Secret types:
    - password: IMAP password, generic password
    - token: HTTPS access token, API key
    - ssh_key: SSH private key content
    - ssh_passphrase: Passphrase for encrypted SSH key
    """

    __tablename__ = "connector_secrets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    collection_source_id = Column(
        Integer,
        ForeignKey("collection_sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    secret_type = Column(String(50), nullable=False)  # 'password', 'token', 'ssh_key', etc.
    ciphertext = Column(LargeBinary, nullable=False)  # Fernet-encrypted data
    key_id = Column(String(64), nullable=False)  # Identifies which key encrypted this
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

    # Relationships
    source = relationship("CollectionSource", back_populates="secrets")

    __table_args__ = (UniqueConstraint("collection_source_id", "secret_type", name="uq_source_secret_type"),)


class PluginConfig(Base):
    """Configuration and status for external plugins."""

    __tablename__ = "plugin_configs"

    id = Column(String, primary_key=True)  # plugin_id
    type = Column(String, nullable=False, index=True)  # embedding, chunking, connector, etc.
    enabled = Column(Boolean, nullable=False, default=True)
    config = Column(JSON, nullable=False, default=dict)
    last_health_check = Column(DateTime(timezone=True), nullable=True)
    health_status = Column(String(20), nullable=True)  # healthy, unhealthy, unknown
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())


class CollectionSyncRun(Base):
    """Tracks a single sync run across all sources in a collection.

    Each sync run fans out APPEND operations to all sources and tracks
    their completion status for aggregated reporting.

    Status values:
    - running: Sync run in progress
    - success: All sources completed successfully
    - partial: Some sources failed or had partial success
    - failed: All sources failed

    The triggered_by field indicates how the sync was initiated:
    - scheduler: Automatic dispatch from Celery Beat
    - manual: User triggered via API
    """

    __tablename__ = "collection_sync_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    collection_id = Column(
        String,
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    triggered_by = Column(String(50), nullable=False)  # 'scheduler', 'manual'
    started_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), nullable=False, default="running")  # running, success, failed, partial

    # Source completion tracking
    expected_sources = Column(Integer, nullable=False, default=0)
    completed_sources = Column(Integer, nullable=False, default=0)
    failed_sources = Column(Integer, nullable=False, default=0)
    partial_sources = Column(Integer, nullable=False, default=0)

    # Error summary
    error_summary = Column(Text, nullable=True)
    meta = Column(JSON, nullable=True)

    # Relationships
    collection = relationship("Collection", back_populates="sync_runs")


class Operation(Base):
    """Operation model for tracking async collection operations."""

    __tablename__ = "operations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String, unique=True, nullable=False)  # For external reference
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    type = Column(
        Enum(
            OperationType,
            name="operation_type",
            native_enum=True,
            create_constraint=False,
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
            validate_strings=True,
        ),
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
    projection_run = relationship("ProjectionRun", back_populates="operation", uselist=False)


class ProjectionRun(Base):
    """Dimensionality reduction run persisted for visualization."""

    __tablename__ = "projection_runs"
    __table_args__ = (
        CheckConstraint("dimensionality > 0", name="ck_projection_runs_dimensionality_positive"),
        CheckConstraint("point_count IS NULL OR point_count >= 0", name="ck_projection_runs_point_count_non_negative"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String, unique=True, nullable=False, index=True)
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    operation_uuid = Column(
        String,
        ForeignKey("operations.uuid", ondelete="SET NULL"),
        unique=True,
        nullable=True,
        index=True,
    )
    status = Column(
        Enum(
            ProjectionRunStatus,
            name="projection_run_status",
            native_enum=True,
            create_constraint=False,
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
        default=ProjectionRunStatus.PENDING,
        index=True,
    )  # type: ignore[var-annotated]
    dimensionality = Column(Integer, nullable=False)
    reducer = Column(String, nullable=False)
    storage_path = Column(String, nullable=True)
    point_count = Column(Integer, nullable=True)
    config = Column(JSON, nullable=True)
    meta = Column(JSON, nullable=True)
    metadata_hash = Column(String, nullable=True, index=True)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    collection = relationship("Collection", back_populates="projection_runs")
    operation = relationship("Operation", back_populates="projection_run")


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


class ChunkingConfigProfile(Base):
    """User-scoped saved chunking configuration.

    Replaces the legacy JSON file store so configurations persist across
    replicas and can be audited. Records are scoped to a user and may be
    marked as default. Tags are stored as JSON list to avoid extension
    dependencies while keeping flexibility.
    """

    __tablename__ = "chunking_config_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    strategy = Column(String, nullable=False, index=True)
    config = Column(JSON, nullable=False)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    is_default = Column(Boolean, nullable=False, default=False, index=True)
    usage_count = Column(Integer, nullable=False, default=0)
    tags = Column(JSON)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

    __table_args__ = (UniqueConstraint("created_by", "name", name="uq_chunking_config_profiles_user_name"),)

    # Relationships
    user = relationship("User", backref="chunking_config_profiles")


class Chunk(Base):
    """Chunk model for partitioned document storage.

    IMPORTANT: This table is partitioned by LIST(partition_key) in PostgreSQL with 100 partitions.

    Partition Awareness:
    - The table uses LIST partitioning on partition_key (0-99)
    - partition_key is computed automatically via trigger: abs(hashtext(collection_id)) % 100
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
    partition_key = Column(Integer, primary_key=True, nullable=False, server_default="0")  # Computed via trigger

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
            "comment": "Partitioned by LIST(partition_key) with 100 partitions. partition_key is computed via trigger.",
            "info": {
                "partition_key": "partition_key",
                "partition_method": "LIST",
                "partition_count": 100,
                "partition_trigger": "compute_partition_key()",
            },
        },
    )


class MCPProfile(Base):
    """MCP search profile configuration.

    Defines a named search profile that exposes collections to MCP clients
    like Claude Desktop. Each profile has scoped collection access and
    configurable search defaults.

    Profile names must be unique per user and follow the pattern [a-z][a-z0-9_-]*
    to ensure valid tool naming in MCP clients.
    """

    __tablename__ = "mcp_profiles"

    id = Column(String, primary_key=True)  # UUID as string
    name = Column(String(64), nullable=False)  # Tool name: lowercase, no spaces
    description = Column(Text, nullable=False)  # Shown to LLM as tool description
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    enabled = Column(Boolean, nullable=False, default=True)

    # Search defaults
    search_type = Column(String(32), nullable=False, default="semantic")  # semantic, hybrid, keyword, question, code
    result_count = Column(Integer, nullable=False, default=10)
    use_reranker = Column(Boolean, nullable=False, default=True)
    score_threshold = Column(Float, nullable=True)
    hybrid_alpha = Column(Float, nullable=True)  # Only used when search_type=hybrid
    search_mode = Column(String(16), nullable=False, default="dense")  # dense, sparse, hybrid
    rrf_k = Column(Integer, nullable=True)  # RRF constant for hybrid mode (default: 60)
    hyde_enabled = Column(Boolean, nullable=False, default=False)  # New field for HyDE

    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

    # Relationships
    owner = relationship("User", back_populates="mcp_profiles")
    collections = relationship("Collection", secondary="mcp_profile_collections", back_populates="mcp_profiles")

    __table_args__ = (
        UniqueConstraint("owner_id", "name", name="uq_mcp_profiles_owner_name"),
        CheckConstraint(
            "search_type IN ('semantic', 'hybrid', 'keyword', 'question', 'code')",
            name="ck_mcp_profiles_search_type",
        ),
        CheckConstraint("result_count >= 1 AND result_count <= 100", name="ck_mcp_profiles_result_count"),
        CheckConstraint(
            "score_threshold IS NULL OR (score_threshold >= 0 AND score_threshold <= 1)",
            name="ck_mcp_profiles_score_threshold",
        ),
        CheckConstraint(
            "hybrid_alpha IS NULL OR (hybrid_alpha >= 0 AND hybrid_alpha <= 1)",
            name="ck_mcp_profiles_hybrid_alpha",
        ),
        CheckConstraint(
            "search_mode IN ('dense', 'sparse', 'hybrid')",
            name="ck_mcp_profiles_search_mode",
        ),
        CheckConstraint(
            "rrf_k IS NULL OR (rrf_k >= 1 AND rrf_k <= 1000)",
            name="ck_mcp_profiles_rrf_k",
        ),
    )


class MCPProfileCollection(Base):
    """Junction table for MCP profile to collection mapping.

    Maps collections to MCP profiles with an ordering field that determines
    search priority. Lower order values are searched first and may affect
    result ranking when searching across multiple collections.
    """

    __tablename__ = "mcp_profile_collections"

    profile_id = Column(
        String,
        ForeignKey("mcp_profiles.id", ondelete="CASCADE"),
        primary_key=True,
    )
    collection_id = Column(
        String,
        ForeignKey("collections.id", ondelete="CASCADE"),
        primary_key=True,
    )
    order = Column(Integer, nullable=False, default=0)  # Lower values = higher priority


class LLMProviderConfig(Base):
    """Per-user LLM provider configuration.

    Stores quality tier settings (high/low) for LLM model selection.
    Each user has at most one config row (one-to-one with users).
    API keys are stored separately in LLMProviderApiKey for security.
    """

    __tablename__ = "llm_provider_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # High-quality tier (complex tasks: summaries, entity extraction)
    # NULL means "use application default from model registry"
    high_quality_provider = Column(String(32))  # 'anthropic', 'openai', etc.
    high_quality_model = Column(String(128))  # Model ID

    # Low-quality tier (simple tasks: HyDE, keywords)
    low_quality_provider = Column(String(32))
    low_quality_model = Column(String(128))

    # Optional defaults (can be overridden per-call)
    default_temperature = Column(Float)
    default_max_tokens = Column(Integer)
    provider_config = Column(JSON)  # Provider-specific config

    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint(
            "default_temperature IS NULL OR (default_temperature >= 0 AND default_temperature <= 2)",
            name="ck_llm_provider_configs_temperature",
        ),
    )

    # Relationships
    user = relationship("User", back_populates="llm_provider_config")
    api_keys = relationship(
        "LLMProviderApiKey",
        back_populates="config",
        cascade="all, delete-orphan",
    )


class LLMProviderApiKey(Base):
    """Encrypted API keys for LLM providers.

    Keys are stored per-provider (not per-tier). If both tiers use the
    same provider, they share the key. Uses Fernet encryption via
    SecretEncryption class.

    The key_id field stores SHA-256 fingerprint of the encryption key
    used, enabling future key rotation support.
    """

    __tablename__ = "llm_provider_api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    config_id = Column(
        Integer,
        ForeignKey("llm_provider_configs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider = Column(String(32), nullable=False)  # 'anthropic', 'openai', etc.
    ciphertext = Column(LargeBinary, nullable=False)  # Fernet-encrypted API key
    key_id = Column(String(64), nullable=False)  # Encryption key fingerprint
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    last_used_at = Column(DateTime(timezone=True))

    __table_args__ = (UniqueConstraint("config_id", "provider", name="uq_llm_api_keys_config_provider"),)

    # Relationships
    config = relationship("LLMProviderConfig", back_populates="api_keys")


class LLMUsageEvent(Base):
    """LLM token usage tracking.

    Records usage per-request for both interactive (HyDE search) and
    background operations (summarization). Uses provider-reported token
    counts, not approximations.

    Stored in a dedicated table (not OperationMetrics) because:
    - HyDE runs in request path with no Operation record
    - Enables per-user usage dashboards without joining Operations
    """

    __tablename__ = "llm_usage_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    # What was called
    provider = Column(String(32), nullable=False)  # 'anthropic', 'openai', etc.
    model = Column(String(128), nullable=False)  # Model ID
    quality_tier = Column(String(16), nullable=False)  # 'high' or 'low'
    feature = Column(String(50), nullable=False)  # 'hyde', 'summary', 'extraction'

    # Token counts (provider-reported)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)

    # Optional context
    operation_id = Column(Integer)  # NULL for interactive requests (HyDE)
    collection_id = Column(String(36))
    request_metadata = Column(JSON)

    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

    __table_args__ = (
        Index("idx_llm_usage_user_created", "user_id", "created_at"),
        Index("idx_llm_usage_feature", "user_id", "feature"),
    )


class UserPreferences(Base):
    """Per-user preferences for search, collection defaults, and interface settings.

    Stores user-specific settings for search behavior, collection creation, and UI.
    Each user has at most one preferences row (one-to-one with users).
    Missing preferences use application defaults via get_or_create pattern.

    Search preferences:
    - search_top_k: Number of results to return (1-250, default 10)
    - search_mode: 'dense', 'sparse', or 'hybrid' (default 'dense')
    - search_use_reranker: Enable reranking (default false)
    - search_rrf_k: RRF constant for hybrid fusion (1-100, default 60)
    - search_similarity_threshold: Minimum similarity score (0-1, NULL for no threshold)

    Collection defaults:
    - default_embedding_model: Model ID or NULL for system default
    - default_quantization: 'float32', 'float16', 'int8' (default 'float16')
    - default_chunking_strategy: 'character', 'recursive', 'markdown', 'semantic'
    - default_chunk_size: 256-4096 (default 1024)
    - default_chunk_overlap: 0-512 (default 200)
    - default_enable_sparse: Enable sparse indexing (default false)
    - default_sparse_type: 'bm25' or 'splade' (default 'bm25')
    - default_enable_hybrid: Enable hybrid search (requires sparse, default false)

    Interface preferences:
    - data_refresh_interval_ms: Data polling interval in ms (10000-60000, default 30000)
    - visualization_sample_limit: Max points for UMAP/PCA (10000-500000, default 200000)
    - animation_enabled: Enable UI animations (default true)
    """

    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Search preferences
    search_top_k = Column(Integer, nullable=False, default=10)
    search_mode = Column(String(16), nullable=False, default="dense")
    search_use_reranker = Column(Boolean, nullable=False, default=False)
    search_rrf_k = Column(Integer, nullable=False, default=60)
    search_similarity_threshold = Column(Float, nullable=True)

    # Collection defaults
    default_embedding_model = Column(String(128), nullable=True)
    default_quantization = Column(String(16), nullable=False, default="float16")
    default_chunking_strategy = Column(String(32), nullable=False, default="recursive")
    default_chunk_size = Column(Integer, nullable=False, default=1024)
    default_chunk_overlap = Column(Integer, nullable=False, default=200)
    default_enable_sparse = Column(Boolean, nullable=False, default=False)
    default_sparse_type = Column(String(16), nullable=False, default="bm25")
    default_enable_hybrid = Column(Boolean, nullable=False, default=False)

    # Interface preferences
    data_refresh_interval_ms = Column(Integer, nullable=False, default=30000)
    visualization_sample_limit = Column(Integer, nullable=False, default=200000)
    animation_enabled = Column(Boolean, nullable=False, default=True)
    hyde_enabled_default = Column(Boolean, nullable=False, default=False)
    hyde_llm_tier = Column(String(16), nullable=False, default="low")

    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint(
            "search_top_k >= 1 AND search_top_k <= 250",
            name="ck_user_preferences_search_top_k",
        ),
        CheckConstraint(
            "search_mode IN ('dense', 'sparse', 'hybrid')",
            name="ck_user_preferences_search_mode",
        ),
        CheckConstraint(
            "search_rrf_k >= 1 AND search_rrf_k <= 100",
            name="ck_user_preferences_search_rrf_k",
        ),
        CheckConstraint(
            "search_similarity_threshold IS NULL OR (search_similarity_threshold >= 0.0 AND search_similarity_threshold <= 1.0)",
            name="ck_user_preferences_search_similarity_threshold",
        ),
        CheckConstraint(
            "default_quantization IN ('float32', 'float16', 'int8')",
            name="ck_user_preferences_default_quantization",
        ),
        CheckConstraint(
            "default_chunking_strategy IN ('character', 'recursive', 'markdown', 'semantic')",
            name="ck_user_preferences_default_chunking_strategy",
        ),
        CheckConstraint(
            "default_chunk_size >= 256 AND default_chunk_size <= 4096",
            name="ck_user_preferences_default_chunk_size",
        ),
        CheckConstraint(
            "default_chunk_overlap >= 0 AND default_chunk_overlap <= 512",
            name="ck_user_preferences_default_chunk_overlap",
        ),
        CheckConstraint(
            "default_sparse_type IN ('bm25', 'splade')",
            name="ck_user_preferences_default_sparse_type",
        ),
        CheckConstraint(
            "default_enable_hybrid = false OR default_enable_sparse = true",
            name="ck_user_preferences_hybrid_requires_sparse",
        ),
        CheckConstraint(
            "data_refresh_interval_ms >= 10000 AND data_refresh_interval_ms <= 60000",
            name="ck_user_preferences_data_refresh_interval_ms",
        ),
        CheckConstraint(
            "visualization_sample_limit >= 10000 AND visualization_sample_limit <= 500000",
            name="ck_user_preferences_visualization_sample_limit",
        ),
    )

    # Relationships
    user = relationship("User", back_populates="preferences")


class SystemSettings(Base):
    """Key-value store for system-wide admin settings.

    This table stores configurable system parameters that can be modified
    by administrators through the UI instead of requiring environment variables.
    Values are stored as JSON and support any JSON-serializable type.

    A JSON null value means "use environment variable fallback".

    Example:
        >>> setting = SystemSettings(
        ...     key="max_collections_per_user",
        ...     value=20,
        ...     updated_by=admin_user_id,
        ... )
    """

    __tablename__ = "system_settings"

    key = Column(String(64), primary_key=True)
    value = Column(JSON, nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    updated_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    # Relationships
    user = relationship("User", foreign_keys=[updated_by])
