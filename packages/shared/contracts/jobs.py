"""Job management API contracts."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class JobStatus(str, Enum):
    """Job status enumeration."""

    CREATED = "created"
    SCANNING = "scanning"
    PROCESSING = "processing"
    PENDING = "pending"  # Alternative to CREATED
    RUNNING = "running"  # Alternative to PROCESSING
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CreateJobRequest(BaseModel):
    """Create job request model."""

    name: str = Field(..., min_length=1, max_length=255, description="Job name")
    description: str = Field("", max_length=1000, description="Job description")
    directory_path: str = Field(..., min_length=1, max_length=4096, description="Directory path to process")
    model_name: str = Field("Qwen/Qwen3-Embedding-0.6B", max_length=500, description="Embedding model to use")
    chunk_size: int = Field(600, ge=100, le=50000, description="Chunk size in tokens")
    chunk_overlap: int = Field(200, ge=0, description="Chunk overlap in tokens")
    batch_size: int = Field(96, ge=1, description="Batch size for processing")
    vector_dim: int | None = Field(None, description="Vector dimension (auto-detected if not provided)")
    quantization: str = Field("float32", max_length=20, description="Model quantization: float32, float16, or int8")
    instruction: str | None = Field(None, max_length=1000, description="Custom instruction for embeddings")
    job_id: str | None = Field(None, max_length=200, description="Pre-generated job ID (for WebSocket connection)")
    scan_subdirs: bool = Field(True, description="Scan subdirectories")
    file_extensions: list[str] | None = Field(None, description="File extensions to process")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        """Validate chunk size is within reasonable bounds."""
        if v < 100:
            raise ValueError("chunk_size must be at least 100 tokens")
        if v > 50000:
            raise ValueError("chunk_size must not exceed 50000 tokens")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info: ValidationInfo) -> int:
        """Validate chunk overlap is less than chunk size."""
        if v < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if info.data.get("chunk_size") and v >= info.data["chunk_size"]:
            raise ValueError(f'chunk_overlap ({v}) must be less than chunk_size ({info.data["chunk_size"]})')
        return v

    @field_validator("directory_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Clean and validate directory path with security checks.

        Note: This validator resolves symbolic links to their real paths,
        which may change the expected path but provides better security."""
        from pathlib import Path

        # Strip whitespace
        cleaned_path = v.strip()

        # Check for empty path
        if not cleaned_path:
            raise ValueError("Directory path cannot be empty")

        # Check for path traversal attempts
        if ".." in cleaned_path or cleaned_path.startswith("~"):
            raise ValueError("Path traversal not allowed")

        # Check if it's a relative path before resolving
        path_obj = Path(cleaned_path)
        if not path_obj.is_absolute():
            raise ValueError("Only absolute paths are allowed")

        # Normalize the path and resolve any symbolic links
        try:
            resolved_path = path_obj.resolve()
        except (ValueError, RuntimeError) as e:
            raise ValueError(f"Invalid directory path: {e}") from e

        return str(resolved_path)

    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v: str) -> str:
        """Validate quantization type."""
        # Accept both original and shorthand formats
        accepted_values = {"float32", "float16", "int8", "fp32", "fp16"}
        # Normalize fp32/fp16 to float32/float16
        if v == "fp32":
            return "float32"
        if v == "fp16":
            return "float16"
        if v not in accepted_values:
            raise ValueError(
                f"Invalid quantization: {v}. Must be one of: float32, float16, int8 (also accepts fp32, fp16)"
            )
        return v

    @field_validator("file_extensions")
    @classmethod
    def validate_file_extensions(cls, v: list[str] | None) -> list[str] | None:
        """Validate file extensions format."""
        if v is None:
            return v

        validated_extensions = []
        for ext in v:
            # Ensure extension starts with dot
            if not ext.startswith("."):
                ext = f".{ext}"

            # Validate extension format
            if len(ext) < 2:  # At least . and one character
                raise ValueError(f"Invalid file extension: {ext}")
            if len(ext) > 10:  # Reasonable max length
                raise ValueError(f"File extension too long: {ext}")
            if not ext[1:].replace("_", "").isalnum():
                raise ValueError(f"Invalid characters in file extension: {ext}")

            validated_extensions.append(ext.lower())

        return validated_extensions


class AddToCollectionRequest(BaseModel):
    """Request to add documents to an existing collection."""

    collection_name: str = Field(..., min_length=1, max_length=200, description="Collection name")
    directory_path: str = Field(..., min_length=1, max_length=4096, description="Directory path to process")
    description: str = Field("", max_length=1000, description="Description of the addition")
    job_id: str | None = Field(None, max_length=200, description="Pre-generated job ID (for WebSocket connection)")


class JobResponse(BaseModel):
    """Job information response."""

    id: str = Field(..., max_length=200, description="Job ID", alias="job_id")
    name: str = Field(..., max_length=255, description="Job name")
    status: str = Field(
        ..., max_length=50, description="Job status"
    )  # Using str to support both enum and legacy string values
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    completed_at: datetime | None = Field(None, description="Completion timestamp")
    directory_path: str = Field(..., max_length=4096, description="Directory being processed")
    error: str | None = Field(None, max_length=2000, description="Error message if failed", alias="error_message")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress percentage")
    total_files: int = Field(0, description="Total number of files to process")
    processed_files: int = Field(0, description="Number of processed files")
    failed_files: int = Field(0, description="Number of failed files")
    current_file: str | None = Field(None, max_length=4096, description="Currently processing file")
    model_name: str = Field(..., max_length=500, description="Embedding model used")
    quantization: str | None = Field(None, max_length=20, description="Model quantization")
    batch_size: int | None = Field(None, description="Batch size")
    chunk_size: int | None = Field(None, description="Chunk size")
    chunk_overlap: int | None = Field(None, description="Chunk overlap")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    user_id: str | None = Field(None, description="User ID who created the job")
    api_version: str = Field(default="1.0", description="API version")

    class Config:
        populate_by_name = True  # Allow both 'id' and 'job_id', 'error' and 'error_message'

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with legacy field names."""
        data = self.model_dump()
        # Ensure job_id field exists for backward compatibility
        if "job_id" not in data:
            data["job_id"] = data.get("id")
        if "error_message" not in data:
            data["error_message"] = data.get("error")
        return data


class JobListResponse(BaseModel):
    """List of jobs response."""

    jobs: list[JobResponse]
    total: int = Field(description="Total number of jobs")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(100, description="Number of items per page")
    has_more: bool = Field(False, description="Whether more pages are available")
    api_version: str = Field(default="1.0", description="API version")


class JobMetrics(BaseModel):
    """Job processing metrics."""

    embeddings_generated: int = Field(0, description="Total embeddings generated")
    tokens_processed: int = Field(0, description="Total tokens processed")
    processing_time_seconds: float = Field(0.0, description="Total processing time")
    average_chunk_size: float | None = Field(None, description="Average chunk size in tokens")
    files_per_second: float | None = Field(None, description="Processing speed")


class JobUpdateRequest(BaseModel):
    """Request to update job status or progress."""

    status: str | None = Field(None, max_length=50, description="New status")
    progress: float | None = Field(None, ge=0.0, le=1.0, description="Progress percentage")
    processed_files: int | None = Field(None, ge=0, description="Number of processed files")
    failed_files: int | None = Field(None, ge=0, description="Number of failed files")
    current_file: str | None = Field(None, max_length=4096, description="Currently processing file")
    error: str | None = Field(None, max_length=2000, description="Error message")
    metrics: JobMetrics | None = Field(None, description="Processing metrics")


class JobFilter(BaseModel):
    """Filter criteria for listing jobs."""

    status: str | None = Field(None, max_length=50, description="Filter by status")
    user_id: str | None = Field(None, max_length=200, description="Filter by user ID")
    created_after: datetime | None = Field(None, description="Filter by creation date")
    created_before: datetime | None = Field(None, description="Filter by creation date")
    name_contains: str | None = Field(None, max_length=255, description="Filter by name substring")
    model_name: str | None = Field(None, max_length=500, description="Filter by model name")
