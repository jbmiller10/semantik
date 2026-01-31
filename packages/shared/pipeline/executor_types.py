"""Type definitions for the pipeline executor.

This module defines data structures used by the pipeline executor for tracking
execution state, progress events, and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from shared.pipeline.types import FileReference, PipelineNode


class ExecutionMode(str, Enum):
    """Execution mode for the pipeline.

    Attributes:
        FULL: Production mode - writes documents and vectors to storage
        DRY_RUN: Validation mode - processes files but does not write to storage
    """

    FULL = "full"
    DRY_RUN = "dry_run"


@dataclass
class ProgressEvent:
    """Progress event emitted during pipeline execution.

    These events allow callers to track the progress of pipeline execution
    for UI updates and logging.

    Attributes:
        event_type: Type of event that occurred
        file_uri: URI of the file being processed (if applicable)
        stage_id: Pipeline stage ID (if applicable)
        details: Additional event-specific details
    """

    event_type: Literal[
        "file_started",
        "file_completed",
        "file_failed",
        "file_skipped",
        "stage_completed",
        "pipeline_started",
        "pipeline_completed",
        "pipeline_halted",
    ]
    file_uri: str | None = None
    stage_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "event_type": self.event_type,
            "file_uri": self.file_uri,
            "stage_id": self.stage_id,
            "details": dict(self.details),
        }


@dataclass
class ChunkStats:
    """Statistics about chunks created during pipeline execution.

    Attributes:
        total_chunks: Total number of chunks created
        avg_tokens: Average tokens per chunk
        min_tokens: Minimum tokens in any chunk
        max_tokens: Maximum tokens in any chunk
    """

    total_chunks: int
    avg_tokens: float
    min_tokens: int
    max_tokens: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "total_chunks": self.total_chunks,
            "avg_tokens": self.avg_tokens,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_token_counts(cls, token_counts: list[int]) -> ChunkStats | None:
        """Create ChunkStats from a list of token counts.

        Args:
            token_counts: List of token counts per chunk

        Returns:
            ChunkStats instance or None if no chunks
        """
        if not token_counts:
            return None

        return cls(
            total_chunks=len(token_counts),
            avg_tokens=sum(token_counts) / len(token_counts),
            min_tokens=min(token_counts),
            max_tokens=max(token_counts),
        )


@dataclass
class PathState:
    """State for a single execution path through the pipeline DAG.

    When a document is processed through multiple parallel paths (e.g., chunking
    path AND summarization path), each path maintains its own state. This allows
    different processing strategies to produce path-specific outputs that are
    tagged for search filtering.

    Attributes:
        path_id: Unique identifier for this path (from edge path_name or node id)
        current_node: The node currently being or to be processed (None when complete)
        parsed_text: Text content after parsing (may be shared across paths)
        parse_metadata: Parser-extracted metadata (may be shared across paths)
        chunks: Chunks produced by this path's chunker
        token_counts: Token counts for each chunk
        completed: Whether this path has finished processing
        error: Exception if this path encountered an error
    """

    path_id: str
    current_node: PipelineNode | None = None
    parsed_text: str = ""
    parse_metadata: dict[str, Any] = field(default_factory=dict)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    completed: bool = False
    error: Exception | None = None


@dataclass
class SampleOutput:
    """Sample output from DRY_RUN mode for preview/validation.

    Attributes:
        file_ref: The file reference that was processed
        chunks: List of chunk data (text, metadata, etc.)
        parse_metadata: Metadata from the parser
        path_id: Path identifier for parallel fan-out (default: "default")
    """

    file_ref: FileReference
    chunks: list[dict[str, Any]]
    parse_metadata: dict[str, Any]
    path_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "file_ref": self.file_ref.to_dict(),
            "chunks": list(self.chunks),
            "parse_metadata": dict(self.parse_metadata),
            "path_id": self.path_id,
        }


@dataclass
class StageFailure:
    """Details of a failure at a specific pipeline stage.

    Attributes:
        file_uri: URI of the file that failed
        stage_id: Pipeline stage ID where failure occurred
        stage_type: Type of the pipeline stage
        error_type: Category of the error
        error_message: Human-readable error description
        error_traceback: Optional stack trace
    """

    file_uri: str
    stage_id: str
    stage_type: str
    error_type: str
    error_message: str
    error_traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "file_uri": self.file_uri,
            "stage_id": self.stage_id,
            "stage_type": self.stage_type,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
        }


@dataclass
class ExecutionResult:
    """Result of pipeline execution.

    This is returned after executing the pipeline over a set of files,
    containing summary statistics and any failures encountered.

    Attributes:
        mode: The execution mode used
        files_processed: Total files processed (including failed)
        files_succeeded: Files that completed successfully
        files_failed: Files that failed processing
        files_skipped: Files skipped due to unchanged content
        chunks_created: Total chunks created during execution
        chunk_stats: Statistics about chunks (if available)
        failures: List of failures encountered
        stage_timings: Time spent in each stage (stage_id -> milliseconds)
        total_duration_ms: Total execution time in milliseconds
        sample_outputs: Sample outputs in DRY_RUN mode (None in FULL mode)
        halted: Whether execution was halted due to consecutive failures
        halt_reason: Reason for halt (if halted)
        callback_failures: Number of times progress callbacks failed
        warnings: Non-fatal issues encountered (e.g., sniff failures, skipped files)
    """

    mode: ExecutionMode
    files_processed: int
    files_succeeded: int
    files_failed: int
    files_skipped: int
    chunks_created: int
    chunk_stats: ChunkStats | None
    failures: list[StageFailure]
    stage_timings: dict[str, float]
    total_duration_ms: float
    sample_outputs: list[SampleOutput] | None = None
    halted: bool = False
    halt_reason: str | None = None
    callback_failures: int = 0
    warnings: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        outcome_sum = self.files_succeeded + self.files_failed + self.files_skipped
        if outcome_sum > self.files_processed:
            raise ValueError(f"Sum of outcomes ({outcome_sum}) cannot exceed files_processed ({self.files_processed})")
        if self.halt_reason and not self.halted:
            raise ValueError("halt_reason requires halted=True")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "mode": self.mode.value,
            "files_processed": self.files_processed,
            "files_succeeded": self.files_succeeded,
            "files_failed": self.files_failed,
            "files_skipped": self.files_skipped,
            "chunks_created": self.chunks_created,
            "chunk_stats": self.chunk_stats.to_dict() if self.chunk_stats else None,
            "failures": [f.to_dict() for f in self.failures],
            "stage_timings": dict(self.stage_timings),
            "total_duration_ms": self.total_duration_ms,
            "sample_outputs": [s.to_dict() for s in self.sample_outputs] if self.sample_outputs else None,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "callback_failures": self.callback_failures,
            "warnings": self.warnings,
        }


__all__ = [
    "ExecutionMode",
    "ProgressEvent",
    "ChunkStats",
    "PathState",
    "SampleOutput",
    "StageFailure",
    "ExecutionResult",
]
