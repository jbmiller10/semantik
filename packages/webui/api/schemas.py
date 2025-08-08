"""
Pydantic schemas for the collections-based API.

This module defines request/response models for the WebUI API endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Enums
class DocumentStatusEnum(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class PermissionTypeEnum(str, Enum):
    """Collection permission types."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


# User schemas
class UserBase(BaseModel):
    """Base user schema."""

    username: str
    email: str
    full_name: str | None = None
    is_active: bool = True
    is_superuser: bool = False


class UserCreate(UserBase):
    """Schema for creating a user."""

    password: str


class UserUpdate(BaseModel):
    """Schema for updating a user."""

    username: str | None = None
    email: str | None = None
    full_name: str | None = None
    password: str | None = None
    is_active: bool | None = None
    is_superuser: bool | None = None


class UserResponse(UserBase):
    """User response schema."""

    id: int
    created_at: str
    updated_at: datetime | None = None
    last_login: str | None = None

    model_config = ConfigDict(from_attributes=True)


# Collection schemas
class CollectionBase(BaseModel):
    """Base collection schema."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        pattern=r"^[^/\\*?<>|:\"]+$",
        description='Collection name (cannot contain / \\ * ? < > | : ")',
    )
    description: str | None = None
    embedding_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    quantization: str = Field(
        default="float16",
        pattern="^(float32|float16|int8)$",
        description="Model quantization level (float32, float16, or int8)",
    )
    # Deprecated fields for backward compatibility
    chunk_size: int | None = Field(default=None, ge=100, le=10000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)
    # New chunking strategy fields
    chunking_strategy: str | None = Field(default=None, description="Chunking strategy type")
    chunking_config: dict[str, Any] | None = Field(default=None, description="Strategy-specific configuration")
    is_public: bool = False
    metadata: dict[str, Any] | None = None

    @field_validator("name", mode="after")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        """Normalize collection name by stripping whitespace."""
        return v.strip()


class CollectionCreate(CollectionBase):
    """Schema for creating a collection."""


class CollectionUpdate(BaseModel):
    """Schema for updating a collection."""

    name: str | None = Field(
        None,
        min_length=1,
        max_length=255,
        pattern=r"^[^/\\*?<>|:\"]+$",
        description='Collection name (cannot contain / \\ * ? < > | : ")',
    )
    description: str | None = None
    is_public: bool | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("name", mode="after")
    @classmethod
    def normalize_name(cls, v: str | None) -> str | None:
        """Normalize collection name by stripping whitespace."""
        if v is None:
            return v
        return v.strip()


class AddSourceRequest(BaseModel):
    """Schema for adding a source to a collection."""

    source_path: str = Field(..., description="Path to the source file or directory")
    config: dict[str, Any] | None = Field(
        None,
        description="Optional configuration for the source",
        json_schema_extra={
            "example": {"chunk_size": 1000, "chunk_overlap": 200, "metadata": {"department": "engineering"}}
        },
    )


class CollectionResponse(CollectionBase):
    """Collection response schema."""

    id: str
    owner_id: int
    vector_store_name: str
    created_at: datetime
    updated_at: datetime
    document_count: int | None = 0
    vector_count: int | None = 0
    status: str  # Collection status: pending, ready, processing, error, degraded
    status_message: str | None = None
    initial_operation_id: str | None = None  # ID of the initial INDEX operation

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_collection(cls, collection: Any) -> "CollectionResponse":
        """Create response from ORM Collection object."""
        # Safely coerce new chunking fields to expected types or None to avoid
        # MagicMock leakage in tests where attributes exist but aren't set.
        raw_strategy = getattr(collection, "chunking_strategy", None)
        chunking_strategy = raw_strategy if isinstance(raw_strategy, str | type(None)) else None

        raw_config = getattr(collection, "chunking_config", None)
        chunking_config = raw_config if isinstance(raw_config, dict | type(None)) else None

        return cls(
            id=collection.id,
            name=collection.name,
            description=collection.description,
            owner_id=collection.owner_id,
            vector_store_name=collection.vector_store_name,
            embedding_model=collection.embedding_model,
            quantization=collection.quantization,
            chunk_size=getattr(collection, "chunk_size", None),
            chunk_overlap=getattr(collection, "chunk_overlap", None),
            chunking_strategy=chunking_strategy,
            chunking_config=chunking_config,
            is_public=collection.is_public,
            metadata=collection.meta,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
            document_count=collection.document_count,
            vector_count=collection.vector_count,
            status=collection.status.value if hasattr(collection.status, "value") else collection.status,
            status_message=getattr(collection, "status_message", None),
        )


class CollectionListResponse(BaseModel):
    """Response for listing collections."""

    collections: list[CollectionResponse]
    total: int
    page: int
    per_page: int


# Document schemas
class DocumentBase(BaseModel):
    """Base document schema."""

    file_name: str
    metadata: dict[str, Any] | None = None


class DocumentCreate(DocumentBase):
    """Schema for creating a document."""

    file_path: str
    file_size: int
    mime_type: str | None = None
    content_hash: str


class DocumentUpdate(BaseModel):
    """Schema for updating a document."""

    metadata: dict[str, Any] | None = None


class DocumentResponse(DocumentBase):
    """Document response schema."""

    id: str
    collection_id: str
    file_path: str
    file_size: int
    mime_type: str | None = None
    content_hash: str
    status: DocumentStatusEnum
    error_message: str | None = None
    chunk_count: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """Response for listing documents."""

    documents: list[DocumentResponse]
    total: int
    page: int
    per_page: int


# API Key schemas
class ApiKeyBase(BaseModel):
    """Base API key schema."""

    name: str = Field(..., min_length=1, max_length=255)
    permissions: dict[str, Any] | None = None
    expires_at: datetime | None = None


class ApiKeyCreate(ApiKeyBase):
    """Schema for creating an API key."""


class ApiKeyResponse(ApiKeyBase):
    """API key response schema."""

    id: str
    user_id: int
    key_hash: str
    last_used_at: datetime | None = None
    created_at: datetime
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class ApiKeyCreateResponse(ApiKeyResponse):
    """Response when creating an API key, includes the actual key."""

    api_key: str  # Only returned on creation


# Permission schemas
class CollectionPermissionBase(BaseModel):
    """Base collection permission schema."""

    permission: PermissionTypeEnum


class CollectionPermissionCreate(CollectionPermissionBase):
    """Schema for creating a collection permission."""

    user_id: int | None = None
    api_key_id: str | None = None

    model_config = ConfigDict(json_schema_extra={"example": {"user_id": 2, "permission": "read"}})


class CollectionPermissionResponse(CollectionPermissionBase):
    """Collection permission response schema."""

    id: int
    collection_id: str
    user_id: int | None = None
    api_key_id: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Search schemas
class SearchRequest(BaseModel):
    """Search request schema."""

    collection_id: str
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata_filter: dict[str, Any] | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "collection_id": "123e4567-e89b-12d3-a456-426614174000",
                "query": "How to implement authentication?",
                "top_k": 10,
                "score_threshold": 0.7,
            }
        }
    )


class SearchResult(BaseModel):
    """Individual search result."""

    document_id: str
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]
    file_name: str
    file_path: str


class SearchResponse(BaseModel):
    """Search response schema."""

    results: list[SearchResult]
    query: str
    total_results: int
    search_time_ms: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "document_id": "123e4567-e89b-12d3-a456-426614174000",
                        "chunk_id": "chunk_001",
                        "score": 0.95,
                        "text": "To implement authentication, you can use JWT tokens...",
                        "metadata": {"page": 1, "section": "Authentication"},
                        "file_name": "auth_guide.md",
                        "file_path": "/docs/auth_guide.md",
                    }
                ],
                "query": "How to implement authentication?",
                "total_results": 1,
                "search_time_ms": 125.5,
            }
        }
    )


# Batch operations
class BatchDocumentUpload(BaseModel):
    """Schema for batch document upload."""

    collection_id: str
    directory_path: str
    file_patterns: list[str] | None = Field(default=["*"])
    recursive: bool = True
    metadata: dict[str, Any] | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "collection_id": "123e4567-e89b-12d3-a456-426614174000",
                "directory_path": "/data/documents",
                "file_patterns": ["*.pdf", "*.md", "*.txt"],
                "recursive": True,
            }
        }
    )


# Directory scan schemas
class DirectoryScanRequest(BaseModel):
    """Request to scan a directory for documents."""

    path: str = Field(
        ...,
        description="Path to the directory to scan",
    )
    scan_id: str = Field(
        ...,
        pattern="^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
        description="UUID for tracking this scan session",
    )
    recursive: bool = Field(
        default=True,
        description="Whether to scan subdirectories recursively",
    )
    include_patterns: list[str] | None = Field(
        default=None,
        description="File patterns to include (e.g., '*.pdf', '*.docx')",
    )
    exclude_patterns: list[str] | None = Field(
        default=None,
        description="File patterns to exclude",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "path": "/mnt/shared/documents/project-alpha",
                "scan_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                "recursive": True,
                "include_patterns": ["*.pdf", "*.docx"],
                "exclude_patterns": ["*.tmp", "~*"],
            }
        }
    )


class DirectoryScanFile(BaseModel):
    """Information about a scanned file."""

    file_path: str
    file_name: str
    file_size: int
    mime_type: str | None
    content_hash: str
    modified_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DirectoryScanResponse(BaseModel):
    """Response from directory scan."""

    scan_id: str
    path: str
    files: list[DirectoryScanFile]
    total_files: int
    total_size: int
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "scan_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                "path": "/mnt/shared/documents/project-alpha",
                "files": [
                    {
                        "file_path": "/mnt/shared/documents/project-alpha/spec.pdf",
                        "file_name": "spec.pdf",
                        "file_size": 1048576,
                        "mime_type": "application/pdf",
                        "content_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                        "modified_at": "2025-07-15T10:00:00Z",
                    }
                ],
                "total_files": 150,
                "total_size": 536870912,
                "warnings": ["Permission denied: /mnt/shared/documents/project-alpha/private"],
            }
        }
    )


class DirectoryScanProgress(BaseModel):
    """WebSocket message for directory scan progress."""

    type: str = Field(default="progress", pattern="^(started|counting|progress|completed|error|warning)$")
    scan_id: str
    data: dict[str, Any]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "progress",
                "scan_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                "data": {
                    "files_scanned": 520,
                    "total_files": 1250,
                    "current_path": "/mnt/shared/documents/project-alpha/specs/spec-v2.pdf",
                    "percentage": 41.6,
                },
            }
        }
    )


# Error schemas
class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    code: str | None = None

    model_config = ConfigDict(
        json_schema_extra={"example": {"detail": "Collection not found", "code": "COLLECTION_NOT_FOUND"}}
    )


# Operation schemas
class OperationResponse(BaseModel):
    """Operation response schema."""

    id: str
    collection_id: str
    type: str
    status: str
    config: dict[str, Any]
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "collection_id": "123e4567-e89b-12d3-a456-426614174000",
                "type": "index",
                "status": "processing",
                "config": {"source_path": "/data/documents"},
                "created_at": "2025-07-15T10:00:00Z",
            }
        },
    )
