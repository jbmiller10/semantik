"""Pydantic schemas for model manager endpoints."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


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


class CacheSizeInfo(BaseModel):
    """HuggingFace cache size breakdown."""

    total_cache_size_mb: int = Field(..., description="Entire HF hub cache directory size in MB")
    managed_cache_size_mb: int = Field(..., description="Sum of curated model repo sizes in MB")
    unmanaged_cache_size_mb: int = Field(..., description="Difference: total - managed")
    unmanaged_repo_count: int = Field(..., description="Number of repos not in curated list")

    model_config = ConfigDict(extra="forbid")
