"""Pydantic schemas for pipeline route preview API.

This module defines the request/response schemas for the route preview endpoint
that allows testing how files would be routed through a pipeline DAG.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FieldEvaluationResult(BaseModel):
    """Result of evaluating a single predicate field."""

    field: str = Field(
        ..., description="Field path that was evaluated (e.g., 'mime_type', 'metadata.detected.is_code')"
    )
    pattern: Any = Field(..., description="Pattern that was tested")
    value: Any = Field(None, description="Actual value from the file reference")
    matched: bool = Field(..., description="Whether the value matched the pattern")


class EdgeEvaluationResult(BaseModel):
    """Result of evaluating a single edge predicate."""

    from_node: str = Field(..., description="Source node ID")
    to_node: str = Field(..., description="Target node ID")
    predicate: dict[str, Any] | None = Field(None, description="The 'when' clause predicate")
    matched: bool = Field(..., description="Whether the predicate matched")
    status: Literal["matched", "not_matched", "skipped"] = Field(
        ...,
        description="Status: 'matched' if this edge was selected, 'not_matched' if evaluated but didn't match, 'skipped' if not evaluated (earlier edge matched)",
    )
    field_evaluations: list[FieldEvaluationResult] | None = Field(
        None, description="Detailed field-by-field evaluation results"
    )


class StageEvaluationResult(BaseModel):
    """Result of evaluating routing at a pipeline stage."""

    stage: str = Field(..., description="Stage identifier (e.g., 'entry_routing', 'parser_to_chunker')")
    from_node: str = Field(..., description="Node from which routing occurs")
    evaluated_edges: list[EdgeEvaluationResult] = Field(
        default_factory=list, description="All edges that were evaluated at this stage"
    )
    selected_node: str | None = Field(None, description="The node that was selected for this stage")
    metadata_snapshot: dict[str, Any] = Field(
        default_factory=dict, description="Metadata state at this point in routing"
    )


class RoutePreviewResponse(BaseModel):
    """Response from the route preview endpoint."""

    # File information
    file_info: dict[str, Any] = Field(
        ..., description="Basic file information: filename, extension, mime_type, size_bytes"
    )

    # Sniff results
    sniff_result: dict[str, Any] | None = Field(None, description="Content detection results (detected.* fields)")

    # Routing evaluation
    routing_stages: list[StageEvaluationResult] = Field(
        default_factory=list, description="Detailed evaluation results for each routing stage"
    )

    # Final path
    path: list[str] = Field(
        default_factory=list,
        description="Ordered list of node IDs in the selected path (e.g., ['_source', 'pdf_parser', 'recursive_chunker', 'embedder'])",
    )

    # Parser metadata (if parser was run)
    parsed_metadata: dict[str, Any] | None = Field(
        None, description="Metadata extracted by the parser (parsed.* fields)"
    )

    # Timing
    total_duration_ms: float = Field(..., description="Total time taken for route preview in milliseconds")

    # Warnings
    warnings: list[str] = Field(default_factory=list, description="Any warnings encountered during preview")


class RoutePreviewRequest(BaseModel):
    """Request body for route preview (when not using multipart form)."""

    dag: dict[str, Any] = Field(..., description="Pipeline DAG definition as JSON")
    include_parser_metadata: bool = Field(True, description="Whether to run the parser and include parsed metadata")


__all__ = [
    "FieldEvaluationResult",
    "EdgeEvaluationResult",
    "StageEvaluationResult",
    "RoutePreviewResponse",
    "RoutePreviewRequest",
]
