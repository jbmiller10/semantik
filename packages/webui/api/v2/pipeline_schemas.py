"""Pydantic schemas for pipeline route preview API.

This module defines the request/response schemas for the route preview endpoint
that allows testing how files would be routed through a pipeline DAG.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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
    status: Literal["matched", "matched_parallel", "not_matched", "skipped"] = Field(
        ...,
        description="Status: 'matched' if this edge was selected (exclusive), 'matched_parallel' if selected as part of parallel fan-out, 'not_matched' if evaluated but didn't match, 'skipped' if not evaluated (earlier exclusive edge matched)",
    )
    field_evaluations: list[FieldEvaluationResult] | None = Field(
        None, description="Detailed field-by-field evaluation results"
    )
    is_parallel: bool = Field(False, description="Whether edge has parallel=True")
    path_name: str | None = Field(None, description="Path name tag from edge")

    @model_validator(mode="after")
    def validate_matched_status_consistency(self) -> EdgeEvaluationResult:
        """Ensure matched and status fields are consistent."""
        # matched should be True for matched/matched_parallel, False otherwise
        expected_matched = self.status in ("matched", "matched_parallel")
        if self.matched != expected_matched:
            # Log warning but don't fail - prefer status as source of truth
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "EdgeEvaluationResult inconsistency: matched=%s but status=%s",
                self.matched,
                self.status,
            )
        return self


class StageEvaluationResult(BaseModel):
    """Result of evaluating routing at a pipeline stage."""

    stage: str = Field(..., description="Stage identifier (e.g., 'entry_routing', 'parser_to_chunker')")
    from_node: str = Field(..., description="Node from which routing occurs")
    evaluated_edges: list[EdgeEvaluationResult] = Field(
        default_factory=list, description="All edges that were evaluated at this stage"
    )
    selected_node: str | None = Field(None, description="The node that was selected for this stage (first/primary)")
    selected_nodes: list[str] | None = Field(
        None, description="All selected nodes for parallel fan-out (None if single path)"
    )
    metadata_snapshot: dict[str, Any] = Field(
        default_factory=dict, description="Metadata state at this point in routing"
    )


class PathInfo(BaseModel):
    """Information about a single execution path through the pipeline."""

    path_name: str = Field(..., description="Path identifier (from edge path_name or auto-generated)")
    nodes: list[str] = Field(..., description="Ordered list of node IDs in this path")


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

    # Final path (primary path for backward compatibility)
    path: list[str] = Field(
        default_factory=list,
        description="Ordered list of node IDs in the primary path (e.g., ['_source', 'pdf_parser', 'recursive_chunker', 'embedder'])",
    )

    # All execution paths (for parallel fan-out)
    paths: list[PathInfo] | None = Field(
        None, description="All execution paths for parallel fan-out (None for single-path DAGs)"
    )

    # Parser metadata (if parser was run)
    parsed_metadata: dict[str, Any] | None = Field(
        None, description="Metadata extracted by the parser (parsed.* fields)"
    )

    # Timing
    total_duration_ms: float = Field(..., description="Total time taken for route preview in milliseconds")

    # Warnings
    warnings: list[str] = Field(default_factory=list, description="Any warnings encountered during preview")


class AvailablePredicateFieldsRequest(BaseModel):
    """Request for available predicate fields based on DAG and source node."""

    dag: dict[str, Any] = Field(..., description="The pipeline DAG configuration")
    from_node: str = Field(..., description="The source node ID for the edge (e.g., '_source', 'text_parser')")


class PredicateField(BaseModel):
    """A single predicate field available for routing."""

    value: str = Field(..., description="Full field path (e.g., 'metadata.parsed.has_tables')")
    label: str = Field(..., description="Human-readable label (e.g., 'Has Tables')")
    category: Literal["source", "detected", "parsed"] = Field(..., description="Field category for UI grouping")


class AvailablePredicateFieldsResponse(BaseModel):
    """Response containing available predicate fields for an edge."""

    fields: list[PredicateField] = Field(default_factory=list, description="List of available predicate fields")


__all__ = [
    "FieldEvaluationResult",
    "EdgeEvaluationResult",
    "StageEvaluationResult",
    "PathInfo",
    "RoutePreviewResponse",
    "AvailablePredicateFieldsRequest",
    "PredicateField",
    "AvailablePredicateFieldsResponse",
]
