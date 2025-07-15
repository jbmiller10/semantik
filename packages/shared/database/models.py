#!/usr/bin/env python3
"""
SQLAlchemy declarative models for the Semantik database.

This module defines the database schema using SQLAlchemy's declarative mapping.
These models are used by Alembic for migrations and can be used for ORM operations.

Note on Timestamps:
We use DateTime columns for new tables but maintain String columns for existing
user-related tables for backward compatibility. A future migration could convert
these to proper DateTime columns.
"""

import enum
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
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
    created_at = Column(String, nullable=False)  # Kept as String for compatibility
    updated_at = Column(DateTime)
    last_login = Column(String)  # Kept as String for compatibility

    # Relationships
    collections = relationship("Collection", back_populates="owner", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    permissions = relationship("CollectionPermission", back_populates="user", cascade="all, delete-orphan")


class Collection(Base):
    """Collection model for organizing documents."""

    __tablename__ = "collections"

    id = Column(String, primary_key=True)  # UUID as string
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    vector_store_name = Column(String, unique=True, nullable=False)  # Qdrant collection name
    embedding_model = Column(String, nullable=False)
    chunk_size = Column(Integer, nullable=False, default=1000)
    chunk_overlap = Column(Integer, nullable=False, default=200)
    is_public = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    meta = Column(JSON)

    # Relationships
    owner = relationship("User", back_populates="collections")
    documents = relationship("Document", back_populates="collection", cascade="all, delete-orphan")
    permissions = relationship("CollectionPermission", back_populates="collection", cascade="all, delete-orphan")


class Document(Base):
    """Document model representing files in collections."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True)  # UUID as string
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    file_path = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String)
    content_hash = Column(String, nullable=False, index=True)
    status = Column(Enum(DocumentStatus), nullable=False, default=DocumentStatus.PENDING, index=True)
    error_message = Column(Text)
    chunk_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    meta = Column(JSON)

    # Relationships
    collection = relationship("Collection", back_populates="documents")

    # Indexes
    __table_args__ = (
        Index("ix_documents_collection_content_hash", "collection_id", "content_hash", unique=True),
    )


class ApiKey(Base):
    """API key model for programmatic access."""

    __tablename__ = "api_keys"

    id = Column(String, primary_key=True)  # UUID as string
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)
    key_hash = Column(String, unique=True, nullable=False, index=True)
    permissions = Column(JSON)  # Store collection access rights
    last_used_at = Column(DateTime)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True, index=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")
    collection_permissions = relationship("CollectionPermission", back_populates="api_key", cascade="all, delete-orphan")


class CollectionPermission(Base):
    """Permission model for fine-grained access control."""

    __tablename__ = "collection_permissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    api_key_id = Column(String, ForeignKey("api_keys.id", ondelete="CASCADE"), index=True)
    permission = Column(Enum(PermissionType), nullable=False)
    created_at = Column(DateTime, nullable=False)

    # Relationships
    collection = relationship("Collection", back_populates="permissions")
    user = relationship("User", back_populates="permissions")
    api_key = relationship("ApiKey", back_populates="collection_permissions")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "(user_id IS NOT NULL AND api_key_id IS NULL) OR (user_id IS NULL AND api_key_id IS NOT NULL)",
            name="check_user_or_api_key"
        ),
        Index("ix_collection_permissions_unique_user", "collection_id", "user_id", unique=True, postgresql_where="user_id IS NOT NULL"),
        Index("ix_collection_permissions_unique_api_key", "collection_id", "api_key_id", unique=True, postgresql_where="api_key_id IS NOT NULL"),
    )


class RefreshToken(Base):
    """Refresh token model for JWT authentication."""

    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token_hash = Column(String, unique=True, nullable=False, index=True)
    expires_at = Column(String, nullable=False)  # Kept as String for compatibility
    created_at = Column(String, nullable=False)  # Kept as String for compatibility
    is_revoked = Column(Boolean, default=False)

    # Relationships
    user = relationship("User", back_populates="refresh_tokens")
