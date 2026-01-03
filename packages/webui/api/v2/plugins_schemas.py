"""Pydantic schemas for plugin management APIs."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 - Required at runtime for Pydantic
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.plugins.validation import (
    PLUGIN_ID_MAX_LENGTH,
    PLUGIN_ID_REGEX,
    validate_git_ref,
    validate_plugin_id,
)


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
    sync_warning: str | None = None

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
    sync_warning: str | None = None

    model_config = ConfigDict(extra="forbid")


class PluginHealthResponse(BaseModel):
    plugin_id: str
    health_status: str | None
    last_health_check: datetime | None
    error_message: str | None

    model_config = ConfigDict(extra="forbid")


# --- Available Plugins (from registry) ---


class AvailablePluginInfo(BaseModel):
    """Information about an available (not installed) plugin from the registry."""

    id: str
    type: str
    name: str
    description: str
    author: str
    repository: str
    pypi: str | None = None
    verified: bool
    min_semantik_version: str | None = None
    tags: list[str] = Field(default_factory=list)
    is_compatible: bool = True
    compatibility_message: str | None = None
    is_installed: bool = False
    pending_restart: bool = False  # True if installed in plugins dir but not yet loaded
    install_command: str = ""

    model_config = ConfigDict(extra="forbid")


class PluginInstallRequest(BaseModel):
    """Request to install a plugin from the registry."""

    plugin_id: str = Field(..., max_length=PLUGIN_ID_MAX_LENGTH, pattern=PLUGIN_ID_REGEX)
    """Registry plugin ID, e.g., "openai-embeddings"."""

    version: str | None = Field(default=None, max_length=128)
    """Optional git tag/branch or package version, e.g., "v1.0.0"."""

    model_config = ConfigDict(extra="forbid")

    @field_validator("plugin_id")
    @classmethod
    def _validate_plugin_id(cls, value: str) -> str:
        validate_plugin_id(value)
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str | None) -> str | None:
        if value is None:
            return value
        validate_git_ref(value)
        return value


class PluginInstallResponse(BaseModel):
    """Response from install/uninstall operations."""

    success: bool
    message: str
    restart_required: bool = True

    model_config = ConfigDict(extra="forbid")


class AvailablePluginsListResponse(BaseModel):
    """Response for listing available plugins from the registry."""

    plugins: list[AvailablePluginInfo]
    registry_version: str | None = None
    last_updated: str | None = None
    registry_source: str | None = None  # "remote" or "bundled"
    semantik_version: str

    model_config = ConfigDict(extra="forbid")


# --- Error Response Models ---


class PluginErrorDetail(BaseModel):
    """Detailed error information for a specific field or issue."""

    field: str | None = None
    """Field path where the error occurred (e.g., 'config.api_key')."""

    message: str
    """Human-readable error message."""

    suggestion: str | None = None
    """Optional suggestion for how to fix the error."""

    model_config = ConfigDict(extra="forbid")


class PluginErrorResponse(BaseModel):
    """Structured error response for plugin API endpoints.

    This model provides machine-parseable error information with:
    - A standardized error code from PluginErrorCode enum
    - Human-readable detail message
    - Optional list of field-specific errors
    - Plugin ID for context when applicable
    """

    detail: str
    """Human-readable error summary."""

    code: str
    """Error code from PluginErrorCode enum."""

    errors: list[PluginErrorDetail] = Field(default_factory=list)
    """List of detailed errors (useful for validation failures)."""

    plugin_id: str | None = None
    """Plugin ID if the error is related to a specific plugin."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "detail": "Plugin configuration validation failed",
                "code": "PLUGIN_CONFIG_INVALID",
                "errors": [
                    {
                        "field": "config.api_key",
                        "message": "field is required",
                        "suggestion": "Add 'api_key' to the config object",
                    }
                ],
                "plugin_id": "openai-embeddings",
            }
        },
    )
