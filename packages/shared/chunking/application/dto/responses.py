"""
Response DTOs for chunking application layer.

These DTOs define the output contracts for use cases.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class OperationStatus(str, Enum):
    """Status of a chunking operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"


@dataclass
class ChunkDTO:
    """DTO representing a text chunk."""

    chunk_id: str
    content: str
    position: int
    start_offset: int
    end_offset: int
    token_count: int
    metadata: dict[str, Any]


@dataclass
class PreviewResponse:
    """Output DTO for preview chunking use case."""

    operation_id: str
    chunks: list[ChunkDTO]
    total_chunks_estimate: int
    strategy_used: str
    document_sample_size: int  # Bytes of document that were processed
    processing_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_id": self.operation_id,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "position": c.position,
                    "start_offset": c.start_offset,
                    "end_offset": c.end_offset,
                    "token_count": c.token_count,
                    "metadata": c.metadata
                }
                for c in self.chunks
            ],
            "total_chunks_estimate": self.total_chunks_estimate,
            "strategy_used": self.strategy_used,
            "document_sample_size": self.document_sample_size,
            "processing_time_ms": self.processing_time_ms
        }


@dataclass
class ProcessDocumentResponse:
    """Output DTO for document processing use case."""

    operation_id: str
    document_id: str
    collection_id: str
    status: OperationStatus
    total_chunks: int
    chunks_processed: int
    chunks_saved: int
    processing_started_at: datetime
    processing_completed_at: datetime | None
    error_message: str | None
    checkpoints_created: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_id": self.operation_id,
            "document_id": self.document_id,
            "collection_id": self.collection_id,
            "status": self.status.value,
            "total_chunks": self.total_chunks,
            "chunks_processed": self.chunks_processed,
            "chunks_saved": self.chunks_saved,
            "processing_started_at": self.processing_started_at.isoformat(),
            "processing_completed_at": self.processing_completed_at.isoformat() if self.processing_completed_at else None,
            "error_message": self.error_message,
            "checkpoints_created": self.checkpoints_created
        }


@dataclass
class StrategyMetrics:
    """Metrics for a single chunking strategy."""

    strategy_name: str
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    avg_token_count: float
    processing_time_ms: float
    overlap_effectiveness: float  # 0-1 score for overlap quality
    semantic_coherence: float  # 0-1 score for semantic boundaries


@dataclass
class CompareStrategiesResponse:
    """Output DTO for strategy comparison use case."""

    operation_id: str
    strategies_compared: list[str]
    document_sample_size: int
    metrics: list[StrategyMetrics]
    recommended_strategy: str
    recommendation_reason: str
    sample_chunks: dict[str, list[ChunkDTO]]  # First 3 chunks from each strategy

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_id": self.operation_id,
            "strategies_compared": self.strategies_compared,
            "document_sample_size": self.document_sample_size,
            "metrics": [
                {
                    "strategy_name": m.strategy_name,
                    "total_chunks": m.total_chunks,
                    "avg_chunk_size": m.avg_chunk_size,
                    "min_chunk_size": m.min_chunk_size,
                    "max_chunk_size": m.max_chunk_size,
                    "avg_token_count": m.avg_token_count,
                    "processing_time_ms": m.processing_time_ms,
                    "overlap_effectiveness": m.overlap_effectiveness,
                    "semantic_coherence": m.semantic_coherence
                }
                for m in self.metrics
            ],
            "recommended_strategy": self.recommended_strategy,
            "recommendation_reason": self.recommendation_reason,
            "sample_chunks": {
                strategy: [
                    {
                        "chunk_id": c.chunk_id,
                        "content": c.content,
                        "position": c.position,
                        "token_count": c.token_count
                    }
                    for c in chunks
                ]
                for strategy, chunks in self.sample_chunks.items()
            }
        }


@dataclass
class OperationMetrics:
    """Detailed metrics for an operation."""

    chunks_per_second: float
    avg_chunk_processing_time_ms: float
    memory_usage_mb: float
    checkpoint_recovery_count: int
    error_count: int
    retry_count: int


@dataclass
class GetOperationStatusResponse:
    """Output DTO for operation status query."""

    operation_id: str
    status: OperationStatus
    progress_percentage: float
    chunks_processed: int
    total_chunks: int | None
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    error_message: str | None
    error_details: dict[str, Any] | None
    chunks: list[ChunkDTO] | None
    metrics: OperationMetrics | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "operation_id": self.operation_id,
            "status": self.status.value,
            "progress_percentage": self.progress_percentage,
            "chunks_processed": self.chunks_processed,
            "total_chunks": self.total_chunks,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "error_details": self.error_details
        }

        if self.chunks:
            result["chunks"] = [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content[:100] + "..." if len(c.content) > 100 else c.content,
                    "position": c.position,
                    "token_count": c.token_count
                }
                for c in self.chunks
            ]

        if self.metrics:
            result["metrics"] = {
                "chunks_per_second": self.metrics.chunks_per_second,
                "avg_chunk_processing_time_ms": self.metrics.avg_chunk_processing_time_ms,
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "checkpoint_recovery_count": self.metrics.checkpoint_recovery_count,
                "error_count": self.metrics.error_count,
                "retry_count": self.metrics.retry_count
            }

        return result


@dataclass
class CancelOperationResponse:
    """Output DTO for operation cancellation."""

    operation_id: str
    previous_status: OperationStatus
    new_status: OperationStatus
    chunks_deleted: int
    cancellation_time: datetime
    cancellation_reason: str | None
    cleanup_performed: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_id": self.operation_id,
            "previous_status": self.previous_status.value,
            "new_status": self.new_status.value,
            "chunks_deleted": self.chunks_deleted,
            "cancellation_time": self.cancellation_time.isoformat(),
            "cancellation_reason": self.cancellation_reason,
            "cleanup_performed": self.cleanup_performed
        }
