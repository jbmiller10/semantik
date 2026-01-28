"""Pipeline template discovery and listing API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from webui.api.schemas import ErrorResponse
from webui.auth import get_current_user

router = APIRouter(prefix="/api/v2/templates", tags=["templates-v2"])


# --- Response Models ---


class TunableParameterSchema(BaseModel):
    """Schema for a tunable parameter in a template."""

    path: str = Field(..., description="Dot-notation path to the parameter")
    description: str = Field(..., description="Human-readable description")
    default: Any = Field(..., description="Default value for this parameter")
    range: list[int] | None = Field(None, description="[min, max] range for numeric parameters")
    options: list[str] | None = Field(None, description="Valid options for enum-like parameters")


class PipelineNodeSchema(BaseModel):
    """Schema for a node in the pipeline DAG."""

    id: str = Field(..., description="Unique node identifier")
    type: str = Field(..., description="Node type (parser, chunker, extractor, embedder)")
    plugin_id: str = Field(..., description="Plugin ID that implements this node")
    config: dict[str, Any] = Field(default_factory=dict, description="Node configuration")


class PipelineEdgeSchema(BaseModel):
    """Schema for an edge in the pipeline DAG."""

    from_node: str = Field(..., description="Source node ID (or '_source')")
    to_node: str = Field(..., description="Target node ID")
    when: dict[str, Any] | None = Field(None, description="Optional predicate for routing")


class PipelineDAGSchema(BaseModel):
    """Schema for a complete pipeline DAG."""

    id: str = Field(..., description="DAG identifier")
    version: str = Field(..., description="Schema version")
    nodes: list[PipelineNodeSchema] = Field(..., description="Processing nodes")
    edges: list[PipelineEdgeSchema] = Field(..., description="Data flow edges")


class TemplateSummary(BaseModel):
    """Summary information for a template (used in list responses)."""

    id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description")
    suggested_for: list[str] = Field(..., description="Use case hints")


class TemplateListResponse(BaseModel):
    """Response for listing all templates."""

    templates: list[TemplateSummary]
    total: int


class TemplateDetailResponse(BaseModel):
    """Full template details including pipeline DAG."""

    id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description")
    suggested_for: list[str] = Field(..., description="Use case hints")
    pipeline: PipelineDAGSchema = Field(..., description="Pre-configured pipeline DAG")
    tunable: list[TunableParameterSchema] = Field(default_factory=list, description="Adjustable parameters")


# --- Endpoints ---


@router.get(
    "",
    response_model=TemplateListResponse,
    responses={401: {"model": ErrorResponse, "description": "Unauthorized"}},
)
async def list_templates(
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> TemplateListResponse:
    """List all available pipeline templates.

    Returns summary information for each template, suitable for display
    in a selection UI. Use GET /api/v2/templates/{id} to retrieve
    the full template with pipeline DAG.
    """
    from shared.pipeline.templates import list_templates as get_all_templates

    templates = get_all_templates()
    summaries = [
        TemplateSummary(
            id=t.id,
            name=t.name,
            description=t.description,
            suggested_for=list(t.suggested_for),
        )
        for t in templates
    ]

    return TemplateListResponse(templates=summaries, total=len(summaries))


@router.get(
    "/{template_id}",
    response_model=TemplateDetailResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Template not found"},
    },
)
async def get_template(
    template_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> TemplateDetailResponse:
    """Get full details for a specific template.

    Returns the complete template including the pre-configured pipeline DAG
    and any tunable parameters that can be adjusted.
    """
    from shared.pipeline.templates import load_template

    template = load_template(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail=f"Template not found: {template_id}")

    # Convert to response schema
    pipeline_dict = template.pipeline.to_dict()
    pipeline = PipelineDAGSchema(
        id=pipeline_dict["id"],
        version=pipeline_dict["version"],
        nodes=[
            PipelineNodeSchema(
                id=n["id"],
                type=n["type"],
                plugin_id=n["plugin_id"],
                config=n.get("config", {}),
            )
            for n in pipeline_dict["nodes"]
        ],
        edges=[
            PipelineEdgeSchema(
                from_node=e["from_node"],
                to_node=e["to_node"],
                when=e.get("when"),
            )
            for e in pipeline_dict["edges"]
        ],
    )

    tunable = [
        TunableParameterSchema(
            path=t.path,
            description=t.description,
            default=t.default,
            range=list(t.range) if t.range else None,
            options=list(t.options) if t.options else None,
        )
        for t in template.tunable
    ]

    return TemplateDetailResponse(
        id=template.id,
        name=template.name,
        description=template.description,
        suggested_for=list(template.suggested_for),
        pipeline=pipeline,
        tunable=tunable,
    )
