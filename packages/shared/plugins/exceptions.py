"""Plugin-specific exceptions."""

from __future__ import annotations

from typing import Any


class PluginError(Exception):
    """Base error for plugin-related issues."""


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load."""


class PluginConfigError(PluginError):
    """Raised when plugin configuration is invalid."""


class PluginCompatibilityError(PluginError):
    """Raised when a plugin is incompatible with the current Semantik version."""


class PluginRegistrationError(PluginError):
    """Raised when plugin registration fails.

    This exception provides structured information about registration failures,
    including the plugin ID, type, error code, and additional context.
    """

    def __init__(
        self,
        message: str,
        plugin_id: str | None = None,
        plugin_type: str | None = None,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.plugin_id = plugin_id
        self.plugin_type = plugin_type
        self.error_code = error_code
        self.details = details or {}


class PluginDuplicateError(PluginRegistrationError):
    """Raised when a plugin ID is already registered.

    This can occur when:
    - The same plugin ID exists in a different plugin type
    - A different plugin class is registered with the same ID and type
    """


class PluginContractError(PluginRegistrationError):
    """Raised when a plugin fails contract validation.

    This occurs when a plugin class doesn't implement required methods
    or returns invalid values from contract methods.
    """


class PluginMetadataError(PluginRegistrationError):
    """Raised when plugin metadata is invalid or missing.

    This occurs when required class attributes (like PLUGIN_ID, PLUGIN_TYPE)
    are missing or have invalid values.
    """


class PluginConfigValidationError(PluginError):
    """Raised when plugin configuration validation fails.

    Carries structured validation errors with field paths and suggestions.
    """

    def __init__(
        self,
        message: str,
        plugin_id: str,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.plugin_id = plugin_id
        self.errors = errors or []

    def to_response_dict(self) -> dict[str, Any]:
        """Convert to API response format."""
        return {
            "detail": str(self),
            "plugin_id": self.plugin_id,
            "errors": self.errors,
        }
