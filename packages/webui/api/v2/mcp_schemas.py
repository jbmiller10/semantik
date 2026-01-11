"""Pydantic schemas for MCP profile management APIs."""

from __future__ import annotations

import uuid as uuid_module
from datetime import datetime  # noqa: TCH003 - Required at runtime for Pydantic
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CollectionSummary(BaseModel):
    """Minimal collection info for MCP profile responses."""

    id: str
    name: str

    model_config = ConfigDict(from_attributes=True)


class MCPProfileCreate(BaseModel):
    """Request schema for creating an MCP profile."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
        description="Profile name used as MCP tool name (lowercase, no spaces)",
        json_schema_extra={"example": "coding"},
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Description shown to LLM to explain what this profile searches",
        json_schema_extra={"example": "Search coding documentation, API references, and technical guides"},
    )
    collection_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Collection UUIDs ordered by search priority (first = highest)",
    )
    enabled: bool = Field(default=True, description="Whether the profile is active")
    search_type: Literal["semantic", "hybrid", "keyword", "question", "code"] = Field(
        default="semantic",
        description="Default search type for this profile",
    )
    result_count: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default number of results to return",
    )
    use_reranker: bool = Field(
        default=True,
        description="Whether to apply reranking to search results",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold for results (0-1)",
    )
    hybrid_alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Hybrid search alpha value (0=keyword, 1=semantic). Only used when search_type=hybrid",
    )
    search_mode: Literal["dense", "sparse", "hybrid"] = Field(
        default="dense",
        description="Default search mode: 'dense' (vector only), 'sparse' (BM25/SPLADE), 'hybrid' (combined with RRF)",
    )
    rrf_k: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="RRF constant k for hybrid search mode (default: 60 if not specified)",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "name": "coding",
                "description": "Search coding documentation and API references",
                "collection_ids": ["550e8400-e29b-41d4-a716-446655440000"],
                "enabled": True,
                "search_type": "semantic",
                "search_mode": "dense",
                "result_count": 10,
                "use_reranker": True,
            }
        },
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is valid for MCP tool naming."""
        if v != v.lower():
            raise ValueError("Profile name must be lowercase")
        return v

    @field_validator("collection_ids")
    @classmethod
    def validate_collection_ids(cls, v: list[str]) -> list[str]:
        """Ensure all collection IDs are valid UUIDs."""
        for cid in v:
            try:
                uuid_module.UUID(cid)
            except ValueError as e:
                raise ValueError(f"Invalid UUID: {cid}") from e
        return v


class MCPProfileUpdate(BaseModel):
    """Request schema for updating an MCP profile. All fields optional."""

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
        description="Profile name used as MCP tool name",
    )
    description: str | None = Field(
        default=None,
        min_length=1,
        max_length=1000,
        description="Description shown to LLM",
    )
    collection_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Collection UUIDs ordered by search priority",
    )
    enabled: bool | None = Field(default=None, description="Whether the profile is active")
    search_type: Literal["semantic", "hybrid", "keyword", "question", "code"] | None = Field(
        default=None,
        description="Default search type",
    )
    result_count: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Default number of results",
    )
    use_reranker: bool | None = Field(default=None, description="Apply reranking")
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold",
    )
    hybrid_alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Hybrid alpha value",
    )
    search_mode: Literal["dense", "sparse", "hybrid"] | None = Field(
        default=None,
        description="Default search mode",
    )
    rrf_k: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="RRF constant k for hybrid mode",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        """Ensure name is valid for MCP tool naming."""
        if v is not None and v != v.lower():
            raise ValueError("Profile name must be lowercase")
        return v

    @field_validator("collection_ids")
    @classmethod
    def validate_collection_ids(cls, v: list[str] | None) -> list[str] | None:
        """Ensure all collection IDs are valid UUIDs."""
        if v is not None:
            for cid in v:
                try:
                    uuid_module.UUID(cid)
                except ValueError as e:
                    raise ValueError(f"Invalid UUID: {cid}") from e
        return v


class MCPProfileResponse(BaseModel):
    """Response schema for MCP profile."""

    id: str
    name: str
    description: str
    enabled: bool
    search_type: Literal["semantic", "hybrid", "keyword", "question", "code"]
    result_count: int
    use_reranker: bool
    score_threshold: float | None
    hybrid_alpha: float | None
    search_mode: Literal["dense", "sparse", "hybrid"]
    rrf_k: int | None
    collections: list[CollectionSummary]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MCPProfileListResponse(BaseModel):
    """Response for listing MCP profiles."""

    profiles: list[MCPProfileResponse]
    total: int

    model_config = ConfigDict(extra="forbid")


class MCPClientConfig(BaseModel):
    """Claude Desktop / MCP client configuration snippet."""

    server_name: str = Field(
        ...,
        description="Suggested name for the MCP server entry",
        json_schema_extra={"example": "semantik-coding"},
    )
    command: str = Field(
        ...,
        description="Command to run the MCP server",
        json_schema_extra={"example": "semantik-mcp"},
    )
    args: list[str] = Field(
        ...,
        description="Arguments for the MCP server command",
        json_schema_extra={"example": ["serve", "--profile", "coding"]},
    )
    env: dict[str, str] = Field(
        ...,
        description="Environment variables for the MCP server",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "server_name": "semantik-coding",
                "command": "semantik-mcp",
                "args": ["serve", "--profile", "coding"],
                "env": {
                    "SEMANTIK_WEBUI_URL": "http://localhost:8080",
                    "SEMANTIK_AUTH_TOKEN": "<your-access-token-or-api-key>",
                },
            }
        },
    )
