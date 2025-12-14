"""Pydantic models for Graph API."""

from pydantic import BaseModel, Field


class EntityResponse(BaseModel):
    """Response model for a single entity."""

    id: int
    name: str
    entity_type: str
    document_id: str
    chunk_id: int | None = None
    confidence: float
    created_at: str

    class Config:
        from_attributes = True


class EntitySearchRequest(BaseModel):
    """Request model for entity search."""

    query: str | None = Field(None, description="Search query (name prefix)")
    entity_types: list[str] | None = Field(None, description="Filter by entity types")
    limit: int = Field(50, ge=1, le=200, description="Maximum results")
    offset: int = Field(0, ge=0, description="Offset for pagination")


class EntitySearchResponse(BaseModel):
    """Response model for entity search."""

    entities: list[EntityResponse]
    total: int
    has_more: bool


class GraphTraversalRequest(BaseModel):
    """Request model for graph traversal."""

    entity_id: int = Field(..., description="Starting entity ID")
    max_hops: int = Field(2, ge=1, le=5, description="Maximum traversal depth")
    relationship_types: list[str] | None = Field(None, description="Filter by relationship types")


class GraphNode(BaseModel):
    """A node in the graph visualization."""

    id: int
    name: str
    type: str
    hop: int = Field(0, description="Distance from starting entity")


class GraphEdge(BaseModel):
    """An edge in the graph visualization."""

    id: int
    source: int
    target: int
    type: str
    confidence: float


class GraphResponse(BaseModel):
    """Response model for graph traversal."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]


class GraphStatsResponse(BaseModel):
    """Response model for graph statistics."""

    total_entities: int
    entities_by_type: dict[str, int]
    total_relationships: int
    relationships_by_type: dict[str, int]
    graph_enabled: bool
