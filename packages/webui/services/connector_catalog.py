"""Connector catalog for schema-driven UI.

This module provides definitions for all available connectors, including their
configuration fields, secrets, and UI metadata. The catalog enables the frontend
to dynamically render appropriate forms for each connector type.
"""

from __future__ import annotations

from typing import Any, TypedDict


class FieldOption(TypedDict, total=False):
    """Option for select fields."""

    value: str
    label: str


class ShowWhen(TypedDict, total=False):
    """Conditional visibility rules."""

    field: str
    equals: str | list[str]


class FieldDefinition(TypedDict, total=False):
    """Definition for a connector configuration field."""

    name: str
    type: str  # text, number, select, multiselect, textarea, boolean, glob_list
    label: str
    description: str
    required: bool
    default: Any
    placeholder: str
    options: list[FieldOption]  # For select/multiselect
    show_when: ShowWhen  # Conditional visibility
    min: int | float  # For number fields
    max: int | float
    step: int | float


class SecretDefinition(TypedDict, total=False):
    """Definition for a connector secret field."""

    name: str
    label: str
    description: str
    required: bool
    show_when: ShowWhen  # Conditional visibility
    is_multiline: bool  # For SSH keys, etc.


class ConnectorDefinition(TypedDict, total=False):
    """Complete definition for a connector type."""

    name: str
    description: str
    icon: str
    fields: list[FieldDefinition]
    secrets: list[SecretDefinition]
    supports_sync: bool
    preview_endpoint: str


# =============================================================================
# Connector Catalog
# =============================================================================

CONNECTOR_CATALOG: dict[str, ConnectorDefinition] = {
    "directory": {
        "name": "Local Directory",
        "description": "Index files from a local directory on the server",
        "icon": "folder",
        "fields": [
            {
                "name": "path",
                "type": "text",
                "label": "Directory Path",
                "description": "Absolute path to the directory to index",
                "required": True,
                "placeholder": "/path/to/documents",
            },
            {
                "name": "recursive",
                "type": "boolean",
                "label": "Recursive",
                "description": "Include files from subdirectories",
                "default": True,
            },
            {
                "name": "include_patterns",
                "type": "glob_list",
                "label": "Include Patterns",
                "description": "Glob patterns to include (e.g., *.md, *.py)",
                "placeholder": "*.md, *.txt",
            },
            {
                "name": "exclude_patterns",
                "type": "glob_list",
                "label": "Exclude Patterns",
                "description": "Glob patterns to exclude",
                "placeholder": "*.log, __pycache__/**",
            },
        ],
        "secrets": [],
        "supports_sync": True,
    },
    "git": {
        "name": "Git Repository",
        "description": "Clone and index files from a remote Git repository",
        "icon": "git-branch",
        "fields": [
            {
                "name": "repo_url",
                "type": "text",
                "label": "Repository URL",
                "description": "HTTPS or SSH URL of the Git repository",
                "required": True,
                "placeholder": "https://github.com/user/repo.git",
            },
            {
                "name": "ref",
                "type": "text",
                "label": "Branch/Tag",
                "description": "Branch, tag, or commit to checkout",
                "default": "main",
                "placeholder": "main",
            },
            {
                "name": "auth_method",
                "type": "select",
                "label": "Authentication",
                "description": "How to authenticate with the repository",
                "default": "none",
                "options": [
                    {"value": "none", "label": "None (Public)"},
                    {"value": "https_token", "label": "HTTPS Token"},
                    {"value": "ssh_key", "label": "SSH Key"},
                ],
            },
            {
                "name": "include_globs",
                "type": "glob_list",
                "label": "Include Patterns",
                "description": "Glob patterns for files to include",
                "placeholder": "*.md, docs/**",
            },
            {
                "name": "exclude_globs",
                "type": "glob_list",
                "label": "Exclude Patterns",
                "description": "Glob patterns for files to exclude",
                "placeholder": "*.min.js, node_modules/**",
            },
            {
                "name": "max_file_size_mb",
                "type": "number",
                "label": "Max File Size (MB)",
                "description": "Maximum file size to index",
                "default": 10,
                "min": 1,
                "max": 100,
            },
            {
                "name": "shallow_depth",
                "type": "number",
                "label": "Clone Depth",
                "description": "Shallow clone depth (0 for full history)",
                "default": 1,
                "min": 0,
                "max": 100,
            },
        ],
        "secrets": [
            {
                "name": "token",
                "label": "Personal Access Token",
                "description": "GitHub/GitLab personal access token",
                "show_when": {"field": "auth_method", "equals": "https_token"},
            },
            {
                "name": "ssh_key",
                "label": "SSH Private Key",
                "description": "SSH private key content",
                "is_multiline": True,
                "show_when": {"field": "auth_method", "equals": "ssh_key"},
            },
            {
                "name": "ssh_passphrase",
                "label": "SSH Key Passphrase",
                "description": "Passphrase for the SSH key (if encrypted)",
                "show_when": {"field": "auth_method", "equals": "ssh_key"},
            },
        ],
        "supports_sync": True,
        "preview_endpoint": "/api/v2/connectors/preview/git",
    },
    "imap": {
        "name": "Email (IMAP)",
        "description": "Connect to an IMAP mailbox and index emails",
        "icon": "mail",
        "fields": [
            {
                "name": "host",
                "type": "text",
                "label": "IMAP Server",
                "description": "IMAP server hostname",
                "required": True,
                "placeholder": "imap.gmail.com",
            },
            {
                "name": "port",
                "type": "number",
                "label": "Port",
                "description": "IMAP server port",
                "default": 993,
                "min": 1,
                "max": 65535,
            },
            {
                "name": "use_ssl",
                "type": "boolean",
                "label": "Use SSL",
                "description": "Connect using SSL/TLS",
                "default": True,
            },
            {
                "name": "username",
                "type": "text",
                "label": "Username",
                "description": "IMAP username or email address",
                "required": True,
                "placeholder": "user@example.com",
            },
            {
                "name": "mailboxes",
                "type": "multiselect",
                "label": "Mailboxes",
                "description": "Mailboxes to sync (leave empty for INBOX only)",
                "placeholder": "INBOX, Sent",
            },
            {
                "name": "since_days",
                "type": "number",
                "label": "Initial Days",
                "description": "Days to look back for initial sync",
                "default": 30,
                "min": 1,
                "max": 365,
            },
            {
                "name": "max_messages",
                "type": "number",
                "label": "Max Messages",
                "description": "Maximum messages per sync",
                "default": 1000,
                "min": 10,
                "max": 10000,
            },
        ],
        "secrets": [
            {
                "name": "password",
                "label": "Password",
                "description": "IMAP password or app password",
                "required": True,
            },
        ],
        "supports_sync": True,
        "preview_endpoint": "/api/v2/connectors/preview/imap",
    },
}


def get_connector_catalog() -> dict[str, ConnectorDefinition]:
    """Get the complete connector catalog.

    Returns:
        Dict mapping connector type to its definition
    """
    return CONNECTOR_CATALOG.copy()


def get_connector_definition(connector_type: str) -> ConnectorDefinition | None:
    """Get the definition for a specific connector type.

    Args:
        connector_type: The connector type identifier

    Returns:
        ConnectorDefinition or None if not found
    """
    return CONNECTOR_CATALOG.get(connector_type.lower())


def list_connector_types() -> list[str]:
    """List all available connector types.

    Returns:
        List of connector type identifiers
    """
    return list(CONNECTOR_CATALOG.keys())
