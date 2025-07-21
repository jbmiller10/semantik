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
"""

import enum

from sqlalchemy import (
    JSON,
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
    is_public = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    meta = Column(JSON)

    # New fields from second migration
    status = Column(Enum(CollectionStatus, name='collection_status', values_callable=lambda obj: [e.value for e in obj]), nullable=False, default=CollectionStatus.PENDING, index=True)  # type: ignore[var-annotated]
    status_message = Column(Text)
    qdrant_collections = Column(JSON)  # List of Qdrant collection names
    qdrant_staging = Column(JSON)  # Staging collections during reindex
    document_count = Column(Integer, nullable=False, default=0)
    vector_count = Column(Integer, nullable=False, default=0)
    total_size_bytes = Column(Integer, nullable=False, default=0)

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


class Document(Base):
    """Document model representing files in collections."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True)  # UUID as string
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id = Column(Integer, ForeignKey("collection_sources.id"), nullable=True, index=True)
    file_path = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String)
    content_hash = Column(String, nullable=False, index=True)
    status = Column(Enum(DocumentStatus, name='document_status', values_callable=lambda obj: [e.value for e in obj]), nullable=False, default=DocumentStatus.PENDING, index=True)  # type: ignore[var-annotated]
    error_message = Column(Text)
    chunk_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    meta = Column(JSON)

    # Relationships
    collection = relationship("Collection", back_populates="documents")
    source = relationship("CollectionSource", back_populates="documents")

    # Indexes
    __table_args__ = (Index("ix_documents_collection_content_hash", "collection_id", "content_hash", unique=True),)


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
    permission = Column(Enum(PermissionType, name='permission_type'), nullable=False)  # type: ignore[var-annotated]
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
    type = Column(Enum(OperationType, name='operation_type', values_callable=lambda obj: [e.value for e in obj]), nullable=False, index=True)  # type: ignore[var-annotated]
    status = Column(Enum(OperationStatus, name='operation_status', values_callable=lambda obj: [e.value for e in obj]), nullable=False, default=OperationStatus.PENDING, index=True)  # type: ignore[var-annotated]
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
