"""Job management API contracts."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, validator


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

    name: str = Field(..., min_length=1, description="Job name")
    description: str = Field("", description="Job description")
    directory_path: str = Field(..., min_length=1, description="Directory path to process")
    model_name: str = Field("Qwen/Qwen3-Embedding-0.6B", description="Embedding model to use")
    chunk_size: int = Field(600, ge=100, le=50000, description="Chunk size in tokens")
    chunk_overlap: int = Field(200, ge=0, description="Chunk overlap in tokens")
    batch_size: int = Field(96, ge=1, description="Batch size for processing")
    vector_dim: Optional[int] = Field(None, description="Vector dimension (auto-detected if not provided)")
    quantization: str = Field("float32", description="Model quantization: float32, float16, or int8")
    instruction: Optional[str] = Field(None, description="Custom instruction for embeddings")
    job_id: Optional[str] = Field(None, description="Pre-generated job ID (for WebSocket connection)")
    scan_subdirs: bool = Field(True, description="Scan subdirectories")
    file_extensions: Optional[list[str]] = Field(None, description="File extensions to process")
    metadata: Optional[dict[str, Any]] = Field(None, description="Additional metadata")

    @validator("chunk_size")
    def validate_chunk_size(cls, v: int) -> int:
        """Validate chunk size is within reasonable bounds."""
        if v < 100:
            raise ValueError("chunk_size must be at least 100 tokens")
        if v > 50000:
            raise ValueError("chunk_size must not exceed 50000 tokens")
        return v

    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v: int, values: dict[str, Any]) -> int:
        """Validate chunk overlap is less than chunk size."""
        if v < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError(f'chunk_overlap ({v}) must be less than chunk_size ({values["chunk_size"]})')
        return v

    @validator("directory_path")
    def validate_path(cls, v: str) -> str:
        """Clean and validate directory path."""
        return v.strip()

    @validator("quantization")
    def validate_quantization(cls, v: str) -> str:
        """Validate quantization type."""
        valid_types = {"float32", "float16", "int8", "fp32", "fp16"}
        # Normalize fp32/fp16 to float32/float16
        if v == "fp32":
            return "float32"
        if v == "fp16":
            return "float16"
        if v not in valid_types:
            raise ValueError(f"Invalid quantization: {v}. Must be one of {valid_types}")
        return v


class AddToCollectionRequest(BaseModel):
    """Request to add documents to an existing collection."""

    collection_name: str = Field(..., min_length=1, description="Collection name")
    directory_path: str = Field(..., min_length=1, description="Directory path to process")
    description: str = Field("", description="Description of the addition")
    job_id: Optional[str] = Field(None, description="Pre-generated job ID (for WebSocket connection)")


class JobResponse(BaseModel):
    """Job information response."""

    id: str = Field(description="Job ID", alias="job_id")
    name: str = Field(description="Job name")
    status: str = Field(description="Job status")  # Using str to support both enum and legacy string values
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    directory_path: str = Field(description="Directory being processed")
    error: Optional[str] = Field(None, description="Error message if failed", alias="error_message")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress percentage")
    total_files: int = Field(0, description="Total number of files to process")
    processed_files: int = Field(0, description="Number of processed files")
    failed_files: int = Field(0, description="Number of failed files")
    current_file: Optional[str] = Field(None, description="Currently processing file")
    model_name: str = Field(description="Embedding model used")
    quantization: Optional[str] = Field(None, description="Model quantization")
    batch_size: Optional[int] = Field(None, description="Batch size")
    chunk_size: Optional[int] = Field(None, description="Chunk size")
    chunk_overlap: Optional[int] = Field(None, description="Chunk overlap")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    user_id: Optional[str] = Field(None, description="User ID who created the job")

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


class JobMetrics(BaseModel):
    """Job processing metrics."""

    embeddings_generated: int = Field(0, description="Total embeddings generated")
    tokens_processed: int = Field(0, description="Total tokens processed")
    processing_time_seconds: float = Field(0.0, description="Total processing time")
    average_chunk_size: Optional[float] = Field(None, description="Average chunk size in tokens")
    files_per_second: Optional[float] = Field(None, description="Processing speed")


class JobUpdateRequest(BaseModel):
    """Request to update job status or progress."""

    status: Optional[str] = Field(None, description="New status")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Progress percentage")
    processed_files: Optional[int] = Field(None, ge=0, description="Number of processed files")
    failed_files: Optional[int] = Field(None, ge=0, description="Number of failed files")
    current_file: Optional[str] = Field(None, description="Currently processing file")
    error: Optional[str] = Field(None, description="Error message")
    metrics: Optional[JobMetrics] = Field(None, description="Processing metrics")


class JobFilter(BaseModel):
    """Filter criteria for listing jobs."""

    status: Optional[str] = Field(None, description="Filter by status")
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date")
    name_contains: Optional[str] = Field(None, description="Filter by name substring")
    model_name: Optional[str] = Field(None, description="Filter by model name")
