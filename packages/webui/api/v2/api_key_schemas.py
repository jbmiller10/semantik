"""Pydantic schemas for API key management endpoints."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 - Required at runtime for Pydantic
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiKeyCreate(BaseModel):
    """Request schema for creating an API key."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable name for the API key",
        json_schema_extra={"example": "Production MCP Server"},
    )
    expires_in_days: int | None = Field(
        default=None,
        ge=1,
        le=3650,
        description="Days until expiration (1-3650). Uses default if not specified.",
        json_schema_extra={"example": 365},
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "name": "Production MCP Server",
                "expires_in_days": 365,
            }
        },
    )


class ApiKeyUpdate(BaseModel):
    """Request schema for updating an API key (soft revoke/reactivate)."""

    is_active: bool = Field(
        ...,
        description="Set to false to revoke the key, true to reactivate",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "is_active": False,
            }
        },
    )


class ApiKeyResponse(BaseModel):
    """Response schema for API key details (excludes raw key and hash)."""

    id: str = Field(..., description="Unique identifier (UUID)")
    name: str = Field(..., description="Human-readable name")
    is_active: bool = Field(..., description="Whether the key is currently active")
    permissions: dict[str, Any] | None = Field(
        default=None,
        description="Collection access permissions (reserved for future use)",
    )
    last_used_at: datetime | None = Field(
        default=None,
        description="Last time the key was used for authentication",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Expiration timestamp (null = never expires)",
    )
    created_at: datetime = Field(..., description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True)


class ApiKeyCreateResponse(ApiKeyResponse):
    """Response schema for newly created API key (includes raw key once)."""

    api_key: str = Field(
        ...,
        description="The full API key (only shown once at creation)",
        json_schema_extra={"example": "smtk_550e8400_Wq3xY5pZ..."},
    )

    model_config = ConfigDict(from_attributes=True)


class ApiKeyListResponse(BaseModel):
    """Response schema for listing API keys."""

    api_keys: list[ApiKeyResponse] = Field(
        ...,
        description="List of API keys owned by the user",
    )
    total: int = Field(
        ...,
        description="Total number of API keys",
    )

    model_config = ConfigDict(extra="forbid")
