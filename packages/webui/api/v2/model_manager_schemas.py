"""Pydantic schemas for model manager endpoints."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class OperationType(str, Enum):
    """Operation types for model manager tasks."""

    DOWNLOAD = "download"
    DELETE = "delete"


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

    @model_validator(mode="after")
    def _validate_type_specific_details(self) -> "CuratedModelResponse":
        if self.model_type == ModelType.EMBEDDING:
            if self.embedding_details is None:
                raise ValueError("embedding_details is required when model_type='embedding'")
            if self.llm_details is not None:
                raise ValueError("llm_details must be None when model_type='embedding'")
            return self

        if self.model_type == ModelType.LLM:
            if self.llm_details is None:
                raise ValueError("llm_details is required when model_type='llm'")
            if self.embedding_details is not None:
                raise ValueError("embedding_details must be None when model_type='llm'")
            return self

        if self.embedding_details is not None or self.llm_details is not None:
            raise ValueError("embedding_details and llm_details must be None for non-embedding/non-llm models")
        return self


class ModelListResponse(BaseModel):
    """Response schema for the model list endpoint."""

    models: list[CuratedModelResponse] = Field(..., description="List of curated models")
    cache_size: CacheSizeInfo | None = Field(default=None, description="Cache size breakdown (if requested)")
    hf_cache_scan_error: str | None = Field(
        default=None,
        description="Error message if HuggingFace cache scan failed (installed status may be inaccurate)",
    )

    model_config = ConfigDict(extra="forbid")


class ModelManagerConflictResponse(BaseModel):
    """409 Conflict response for model manager operations."""

    conflict_type: ConflictType = Field(..., description="Type of conflict encountered")
    detail: str = Field(..., description="Human-readable error message")
    model_id: str = Field(..., description="HuggingFace model ID")

    # For cross_op_exclusion
    active_operation: OperationType | None = Field(default=None, description="Active operation type (download/delete)")
    active_task_id: str | None = Field(default=None, description="ID of the active task")

    # For in_use_block
    blocked_by_collections: list[str] = Field(default_factory=list, description="Collection names using this model")

    # For requires_confirmation
    requires_confirmation: bool = Field(default=False, description="Whether confirmation is required")
    warnings: list[str] = Field(default_factory=list, description="Warning messages to display")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_conflict_fields(self) -> "ModelManagerConflictResponse":
        if self.conflict_type == ConflictType.CROSS_OP_EXCLUSION:
            if not self.active_operation or not self.active_task_id:
                raise ValueError(
                    "active_operation and active_task_id are required when conflict_type='cross_op_exclusion'"
                )
            if self.blocked_by_collections:
                raise ValueError("blocked_by_collections must be empty when conflict_type='cross_op_exclusion'")
            if self.requires_confirmation:
                raise ValueError("requires_confirmation must be false when conflict_type='cross_op_exclusion'")
            if self.warnings:
                raise ValueError("warnings must be empty when conflict_type='cross_op_exclusion'")
            return self

        if self.conflict_type == ConflictType.IN_USE_BLOCK:
            if not self.blocked_by_collections:
                raise ValueError("blocked_by_collections is required when conflict_type='in_use_block'")
            if self.active_operation is not None or self.active_task_id is not None:
                raise ValueError("active_operation and active_task_id must be None when conflict_type='in_use_block'")
            if self.requires_confirmation:
                raise ValueError("requires_confirmation must be false when conflict_type='in_use_block'")
            if self.warnings:
                raise ValueError("warnings must be empty when conflict_type='in_use_block'")
            return self

        if self.conflict_type == ConflictType.REQUIRES_CONFIRMATION:
            if not self.requires_confirmation:
                raise ValueError("requires_confirmation must be true when conflict_type='requires_confirmation'")
            if not self.warnings:
                raise ValueError("warnings is required when conflict_type='requires_confirmation'")
            if self.active_operation is not None or self.active_task_id is not None:
                raise ValueError(
                    "active_operation and active_task_id must be None when conflict_type='requires_confirmation'"
                )
            if self.blocked_by_collections:
                raise ValueError("blocked_by_collections must be empty when conflict_type='requires_confirmation'")
            return self

        return self


class TaskResponse(BaseModel):
    """Response for download/delete operations."""

    task_id: str | None = Field(default=None, description="Task ID (None for idempotent no-ops)")
    model_id: str = Field(..., description="HuggingFace model ID")
    operation: OperationType = Field(..., description="Operation type (download/delete)")
    status: TaskStatus = Field(..., description="Current task status")
    warnings: list[str] = Field(default_factory=list, description="Warning messages (for delete with confirm=true)")

    model_config = ConfigDict(extra="forbid")


class TaskProgressResponse(BaseModel):
    """Progress response for download/delete tasks.

    Used by GET /api/v2/models/tasks/{task_id} to poll task progress.
    """

    task_id: str = Field(..., description="Unique task identifier")
    model_id: str = Field(..., description="HuggingFace model ID")
    operation: OperationType = Field(..., description="Operation type (download/delete)")
    status: TaskStatus = Field(..., description="Current task status")
    bytes_downloaded: int = Field(default=0, description="Bytes downloaded so far (download only)")
    bytes_total: int = Field(default=0, description="Total bytes to download (download only)")
    error: str | None = Field(default=None, description="Error message if failed")
    updated_at: float = Field(..., description="Last update timestamp (epoch seconds)")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_error_field(self) -> "TaskProgressResponse":
        if self.status != TaskStatus.FAILED and self.error is not None:
            raise ValueError("error must be None unless status='failed'")
        return self


class ModelDownloadRequest(BaseModel):
    """Request body for model download endpoint."""

    model_id: str = Field(..., description="HuggingFace model ID to download")

    model_config = ConfigDict(extra="forbid")


class ModelUsageResponse(BaseModel):
    """Response for model usage preflight check before deletion.

    Provides information about model usage across the system to help users
    understand the impact of deleting a model.
    """

    model_id: str = Field(..., description="HuggingFace model ID")
    is_installed: bool = Field(..., description="Whether the model is installed")
    size_on_disk_mb: int | None = Field(default=None, description="Size on disk in MB (if installed)")
    estimated_freed_size_mb: int | None = Field(
        default=None, description="Estimated space freed after deletion (same as size_on_disk_mb)"
    )

    # Blocking conditions (prevent deletion)
    blocked_by_collections: list[str] = Field(
        default_factory=list, description="Collection names using this model (blocks deletion)"
    )

    # Warning conditions (require confirmation)
    user_preferences_count: int = Field(default=0, description="Number of users with this as default_embedding_model")
    llm_config_count: int = Field(
        default=0, description="Number of LLM configs referencing this model (for local LLMs)"
    )
    is_default_embedding_model: bool = Field(
        default=False, description="Whether this is the system default embedding model"
    )

    # VecPipe state (best-effort, may fail)
    loaded_in_vecpipe: bool = Field(default=False, description="Whether model is loaded in VecPipe GPU memory")
    loaded_vecpipe_model_types: list[str] = Field(
        default_factory=list, description="Model types loaded in VecPipe (embedding, reranker, etc.)"
    )
    hf_cache_scan_error: str | None = Field(
        default=None,
        description="Error message if HuggingFace cache scan failed (installed status may be inaccurate)",
    )
    vecpipe_query_error: str | None = Field(
        default=None,
        description="Error message if VecPipe loaded-model query failed (loaded status is unknown)",
    )

    # Computed fields
    warnings: list[str] = Field(default_factory=list, description="Human-readable warning messages")
    can_delete: bool = Field(..., description="Whether deletion is allowed (no blocking conditions)")
    requires_confirmation: bool = Field(
        ..., description="Whether deletion requires explicit confirmation (warnings exist)"
    )

    model_config = ConfigDict(extra="forbid")
