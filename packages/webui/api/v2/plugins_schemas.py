"""Pydantic schemas for plugin management APIs."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 - Required at runtime for Pydantic
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PluginManifestSchema(BaseModel):
    id: str
    type: str
    version: str
    display_name: str
    description: str
    author: str | None = None
    license: str | None = None
    homepage: str | None = None
    requires: list[str] = Field(default_factory=list)
    semantik_version: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class PluginInfo(BaseModel):
    id: str
    type: str
    version: str
    manifest: PluginManifestSchema
    enabled: bool
    config: dict[str, Any] = Field(default_factory=dict)
    health_status: str | None = None
    last_health_check: datetime | None = None
    error_message: str | None = None
    requires_restart: bool | None = None

    model_config = ConfigDict(extra="allow")


class PluginListResponse(BaseModel):
    plugins: list[PluginInfo]

    model_config = ConfigDict(extra="forbid")


class PluginConfigUpdateRequest(BaseModel):
    config: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class PluginStatusResponse(BaseModel):
    plugin_id: str
    enabled: bool
    requires_restart: bool = True

    model_config = ConfigDict(extra="forbid")


class PluginHealthResponse(BaseModel):
    plugin_id: str
    health_status: str | None
    last_health_check: datetime | None
    error_message: str | None

    model_config = ConfigDict(extra="forbid")
