#!/usr/bin/env python3
"""
SQLAlchemy declarative models for the Semantik database.

This module defines the database schema using SQLAlchemy's declarative mapping.
These models are used by Alembic for migrations and can be used for ORM operations.

Note on Timestamps:
We use String columns for timestamps to maintain backward compatibility with existing
databases that store ISO format timestamp strings. A future migration could convert
these to proper DateTime columns, but that would require careful data migration.
"""


from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# Create the declarative base
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""


class Job(Base):
    """Job model representing processing jobs."""

    __tablename__ = "jobs"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    status = Column(String, nullable=False)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)
    directory_path = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    chunk_size = Column(Integer)
    chunk_overlap = Column(Integer)
    batch_size = Column(Integer)
    vector_dim = Column(Integer)
    quantization = Column(String, default="float32")
    instruction = Column(Text)
    total_files = Column(Integer, default=0)
    processed_files = Column(Integer, default=0)
    failed_files = Column(Integer, default=0)
    current_file = Column(String)
    start_time = Column(String)
    error = Column(Text)
    user_id = Column(Integer, ForeignKey("users.id"))
    parent_job_id = Column(String)
    mode = Column(String, default="create")

    # Relationships
    files = relationship("File", back_populates="job", cascade="all, delete-orphan")
    user = relationship("User", back_populates="jobs")


class File(Base):
    """File model representing files processed in jobs."""

    __tablename__ = "files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    path = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    modified = Column(String, nullable=False)
    extension = Column(String, nullable=False)
    hash = Column(String)
    doc_id = Column(String)
    content_hash = Column(String)
    status = Column(String, default="pending")
    error = Column(Text)
    chunks_created = Column(Integer, default=0)
    vectors_created = Column(Integer, default=0)

    # Relationships
    job = relationship("Job", back_populates="files")

    # Indexes
    __table_args__ = (
        Index("idx_files_job_id", "job_id"),
        Index("idx_files_status", "status"),
        Index("idx_files_doc_id", "doc_id"),
        Index("idx_files_content_hash", "content_hash"),
        Index("idx_files_job_content_hash", "job_id", "content_hash"),
    )


class User(Base):
    """User model for authentication."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(String, nullable=False)
    last_login = Column(String)

    # Relationships
    jobs = relationship("Job", back_populates="user")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")


class RefreshToken(Base):
    """Refresh token model for JWT authentication."""

    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token_hash = Column(String, unique=True, nullable=False, index=True)
    expires_at = Column(String, nullable=False)
    created_at = Column(String, nullable=False)
    is_revoked = Column(Boolean, default=False)

    # Relationships
    user = relationship("User", back_populates="refresh_tokens")
