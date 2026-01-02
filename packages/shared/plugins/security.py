"""Plugin security utilities.

This module provides:
1. Audit logging for plugin operations
2. Cooperative environment filtering for "good citizen" plugins

IMPORTANT: Since plugins run in-process, they can always access os.environ
directly. The env filtering is "defense in depth" for plugins that cooperate
with the API - it is NOT a security boundary.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Patterns that indicate sensitive environment variables
SENSITIVE_ENV_PATTERNS = frozenset(
    {
        "PASSWORD",
        "SECRET",
        "KEY",
        "TOKEN",
        "CREDENTIAL",
        "API_KEY",
        "PRIVATE",
        "AUTH",
    }
)


def get_sanitized_environment() -> dict[str, str]:
    """Return environment with sensitive values removed.

    This utility is for plugins that want to be good citizens and avoid
    accidentally logging or exposing sensitive values. It filters out
    environment variables whose names contain sensitive patterns.

    NOTE: This is cooperative only. Plugins can still access os.environ
    directly if they choose to. This is NOT a security boundary.

    Returns:
        Dictionary of non-sensitive environment variables.

    Example:
        >>> env = get_sanitized_environment()
        >>> "PATH" in env  # Non-sensitive vars included
        True
        >>> "JWT_SECRET_KEY" in env  # Sensitive vars excluded
        False
    """
    return {
        key: value
        for key, value in os.environ.items()
        if not any(pattern in key.upper() for pattern in SENSITIVE_ENV_PATTERNS)
    }


def audit_log(
    plugin_id: str,
    action: str,
    details: dict[str, Any] | None = None,
    *,
    level: int = logging.INFO,
) -> None:
    """Log plugin action for security auditing.

    Creates a structured log entry with plugin_id, action, and optional
    details. All audit logs use the "PLUGIN_AUDIT" prefix for easy filtering.

    This function never raises exceptions - logging failures are caught
    and logged at WARNING level.

    Args:
        plugin_id: Unique identifier of the plugin
        action: Action being audited (e.g., "plugin.registered.external")
        details: Optional dictionary of additional context
        level: Logging level (default: INFO)

    Example:
        >>> audit_log("my-plugin", "plugin.config.updated", {"keys": ["api_url"]})
        # Logs: PLUGIN_AUDIT: my-plugin - plugin.config.updated
    """
    try:
        sanitized_details = _sanitize_audit_details(details)
        extra = {
            "plugin_id": plugin_id,
            "audit_action": action,
            "audit_timestamp": datetime.now(UTC).isoformat(),
            "audit_details": sanitized_details,
        }
        logger.log(level, "PLUGIN_AUDIT: %s - %s", plugin_id, action, extra=extra)
    except Exception as exc:
        # Never let audit logging break plugin operations
        logger.warning("Failed to log plugin audit: %s", exc)


def _sanitize_audit_details(
    details: dict[str, Any] | None,
    _seen: set[int] | None = None,
) -> dict[str, Any] | None:
    """Sanitize audit details to remove potentially sensitive values.

    Removes keys that match sensitive patterns and recursively sanitizes
    nested dictionaries.

    Args:
        details: Dictionary to sanitize
        _seen: Set of object IDs to detect circular references

    Returns:
        Sanitized dictionary or None
    """
    if not details:
        return details

    if _seen is None:
        _seen = set()

    obj_id = id(details)
    if obj_id in _seen:
        return {"__circular_reference__": True}
    _seen.add(obj_id)

    sanitized: dict[str, Any] = {}

    for key, value in details.items():
        # Skip keys that look sensitive
        if any(pattern in key.upper() for pattern in SENSITIVE_ENV_PATTERNS):
            continue

        if isinstance(value, dict):
            sanitized_value = _sanitize_audit_details(value, _seen)
            if sanitized_value is not None:
                sanitized[key] = sanitized_value
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_audit_details(item, _seen) if isinstance(item, dict) else item for item in value
            ]
        else:
            sanitized[key] = value

    _seen.discard(obj_id)
    return sanitized
