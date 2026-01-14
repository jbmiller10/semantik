"""Pydantic schemas for LLM settings API endpoints."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 - Required at runtime for Pydantic
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMSettingsUpdate(BaseModel):
    """Request body for updating LLM settings."""

    # Tier configuration (NULL = use defaults from model registry)
    high_quality_provider: Literal["anthropic", "openai", "local"] | None = None
    high_quality_model: str | None = Field(default=None, max_length=128)
    low_quality_provider: Literal["anthropic", "openai", "local"] | None = None
    low_quality_model: str | None = Field(default=None, max_length=128)

    # API keys per PROVIDER (write-only, shared across tiers)
    anthropic_api_key: str | None = Field(default=None, min_length=1)
    openai_api_key: str | None = Field(default=None, min_length=1)

    # Local model quantization (stored in provider_config JSON)
    local_high_quantization: Literal["int4", "int8"] | None = Field(default=None)
    local_low_quantization: Literal["int4", "int8"] | None = Field(default=None)

    # Optional defaults
    default_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    default_max_tokens: int | None = Field(default=None, ge=1, le=200000)

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "high_quality_provider": "anthropic",
                "high_quality_model": "claude-opus-4-5-20251101",
                "low_quality_provider": "anthropic",
                "low_quality_model": "claude-sonnet-4-5-20250929",
                "anthropic_api_key": "sk-ant-...",
                "default_temperature": 0.7,
            }
        },
    )


class LLMSettingsResponse(BaseModel):
    """Response for GET /llm/settings."""

    high_quality_provider: str | None
    high_quality_model: str | None
    low_quality_provider: str | None
    low_quality_model: str | None

    # Per-provider key status (never return actual keys)
    anthropic_has_key: bool
    openai_has_key: bool

    # Local model quantization settings
    local_high_quantization: str | None = None
    local_low_quantization: str | None = None

    default_temperature: float | None
    default_max_tokens: int | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "high_quality_provider": "anthropic",
                "high_quality_model": "claude-opus-4-5-20251101",
                "low_quality_provider": "anthropic",
                "low_quality_model": "claude-sonnet-4-5-20250929",
                "anthropic_has_key": True,
                "openai_has_key": False,
                "default_temperature": 0.7,
                "default_max_tokens": None,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            }
        },
    )


class AvailableModel(BaseModel):
    """Model info from curated registry or provider API."""

    id: str
    name: str
    display_name: str
    provider: str
    tier_recommendation: str
    context_window: int
    description: str
    is_curated: bool = True  # True for curated registry, False for API-fetched
    memory_mb: dict[str, int] | None = None  # Memory per quantization for local models

    model_config = ConfigDict(extra="forbid")


class AvailableModelsResponse(BaseModel):
    """Response for GET /llm/models."""

    models: list[AvailableModel]

    model_config = ConfigDict(extra="forbid")


class LLMTestRequest(BaseModel):
    """Request to test API key validity."""

    provider: Literal["anthropic", "openai", "local"]
    api_key: str = Field(..., min_length=1, description="API key to test")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "provider": "anthropic",
                "api_key": "sk-ant-...",
            }
        },
    )


class LLMTestResponse(BaseModel):
    """Response for POST /llm/test."""

    success: bool
    message: str
    model_tested: str | None = None

    model_config = ConfigDict(extra="forbid")


class TokenUsageResponse(BaseModel):
    """Response for GET /llm/usage."""

    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    by_feature: dict[str, dict[str, int]]
    by_provider: dict[str, dict[str, int]]
    event_count: int
    period_days: int

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "total_input_tokens": 15000,
                "total_output_tokens": 5000,
                "total_tokens": 20000,
                "by_feature": {
                    "hyde": {"input_tokens": 10000, "output_tokens": 3000, "total_tokens": 13000},
                    "summary": {"input_tokens": 5000, "output_tokens": 2000, "total_tokens": 7000},
                },
                "by_provider": {
                    "anthropic": {"input_tokens": 15000, "output_tokens": 5000, "total_tokens": 20000},
                },
                "event_count": 150,
                "period_days": 30,
            }
        },
    )
