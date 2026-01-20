"""Pydantic schemas for model manager endpoints."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ModelType(str, Enum):
    """Type of model in the curated registry."""

    EMBEDDING = "embedding"
    LLM = "llm"
    RERANKER = "reranker"
    SPLADE = "splade"


class ConflictType(str, Enum):
    """Types of 409 Conflict responses."""

    CROSS_OP_EXCLUSION = "cross_op_exclusion"  # Another op is active
    IN_USE_BLOCK = "in_use_block"  # Collections reference model
    REQUIRES_CONFIRMATION = "requires_confirmation"  # Warnings exist, need confirm=true


class TaskStatus(str, Enum):
    """Task status values including idempotent no-op statuses."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    # Idempotent "no-op" statuses
    ALREADY_INSTALLED = "already_installed"
    NOT_INSTALLED = "not_installed"


class CacheSizeInfo(BaseModel):
    """HuggingFace cache size breakdown."""

    total_cache_size_mb: int = Field(..., description="Entire HF hub cache directory size in MB")
    managed_cache_size_mb: int = Field(..., description="Sum of curated model repo sizes in MB")
    unmanaged_cache_size_mb: int = Field(..., description="Difference: total - managed")
    unmanaged_repo_count: int = Field(..., description="Number of repos not in curated list")

    model_config = ConfigDict(extra="forbid")


class EmbeddingModelDetails(BaseModel):
    """Embedding-specific model details."""

    dimension: int | None = Field(default=None, description="Embedding dimension")
    max_sequence_length: int | None = Field(default=None, description="Maximum input sequence length")
    pooling_method: str | None = Field(default=None, description="Pooling method (mean, cls, last_token)")
    is_asymmetric: bool = Field(
        default=False, description="Whether model uses different handling for queries vs documents"
    )
    query_prefix: str = Field(default="", description="Prefix for query texts")
    document_prefix: str = Field(default="", description="Prefix for document texts")
    default_query_instruction: str = Field(default="", description="Default instruction for query embedding")

    model_config = ConfigDict(extra="forbid")


class LLMModelDetails(BaseModel):
    """LLM-specific model details."""

    context_window: int | None = Field(default=None, description="Context window size in tokens")

    model_config = ConfigDict(extra="forbid")


class CuratedModelResponse(BaseModel):
    """Response schema for a curated model in the model list."""

    id: str = Field(..., description="HuggingFace model ID")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Model description")
    model_type: ModelType = Field(..., description="Type of model")
    memory_mb: dict[str, int] = Field(
        default_factory=dict, description="Memory estimates per quantization (e.g., float16, int8)"
    )
    is_installed: bool = Field(..., description="Whether the model is installed in HF cache")
    size_on_disk_mb: int | None = Field(default=None, description="Size on disk in MB (if installed)")
    used_by_collections: list[str] = Field(
        default_factory=list, description="Collection names using this model (embedding models only)"
    )

    # Placeholders for Phase 1B
    active_download_task_id: str | None = Field(default=None, description="Active download task ID (Phase 1B)")
    active_delete_task_id: str | None = Field(default=None, description="Active delete task ID (Phase 1B)")

    # Type-specific details
    embedding_details: EmbeddingModelDetails | None = Field(default=None, description="Embedding-specific details")
    llm_details: LLMModelDetails | None = Field(default=None, description="LLM-specific details")

    model_config = ConfigDict(extra="forbid")


class ModelListResponse(BaseModel):
    """Response schema for the model list endpoint."""

    models: list[CuratedModelResponse] = Field(..., description="List of curated models")
    cache_size: CacheSizeInfo | None = Field(default=None, description="Cache size breakdown (if requested)")

    model_config = ConfigDict(extra="forbid")


class ModelManagerConflictResponse(BaseModel):
    """409 Conflict response for model manager operations."""

    conflict_type: ConflictType = Field(..., description="Type of conflict encountered")
    detail: str = Field(..., description="Human-readable error message")
    model_id: str = Field(..., description="HuggingFace model ID")

    # For cross_op_exclusion
    active_operation: str | None = Field(default=None, description="Active operation type (download/delete)")
    active_task_id: str | None = Field(default=None, description="ID of the active task")

    # For in_use_block
    blocked_by_collections: list[str] = Field(default_factory=list, description="Collection names using this model")

    # For requires_confirmation
    requires_confirmation: bool = Field(default=False, description="Whether confirmation is required")
    warnings: list[str] = Field(default_factory=list, description="Warning messages to display")

    model_config = ConfigDict(extra="forbid")


class TaskResponse(BaseModel):
    """Response for download/delete operations."""

    task_id: str | None = Field(default=None, description="Task ID (None for idempotent no-ops)")
    model_id: str = Field(..., description="HuggingFace model ID")
    operation: str = Field(..., description="Operation type (download/delete)")
    status: TaskStatus = Field(..., description="Current task status")
    warnings: list[str] = Field(default_factory=list, description="Warning messages (for delete with confirm=true)")

    model_config = ConfigDict(extra="forbid")


class TaskProgressResponse(BaseModel):
    """Progress response for download/delete tasks.

    Used by GET /api/v2/models/tasks/{task_id} to poll task progress.
    """

    task_id: str = Field(..., description="Unique task identifier")
    model_id: str = Field(..., description="HuggingFace model ID")
    operation: str = Field(..., description="Operation type (download/delete)")
    status: TaskStatus = Field(..., description="Current task status")
    bytes_downloaded: int = Field(default=0, description="Bytes downloaded so far (download only)")
    bytes_total: int = Field(default=0, description="Total bytes to download (download only)")
    error: str | None = Field(default=None, description="Error message if failed")
    updated_at: float = Field(..., description="Last update timestamp (epoch seconds)")

    model_config = ConfigDict(extra="forbid")
