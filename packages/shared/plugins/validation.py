"""Validation helpers for plugin management inputs.

These helpers are used by the WebUI plugin management API and the in-app
installer/uninstaller to validate user- and registry-supplied values.

This is not a sandbox: plugins are trusted and run in-process. The goal here is
to prevent obvious path traversal and option-like injection into tools like pip.
"""

from __future__ import annotations

import re

# Plugin IDs are used as stable identifiers across APIs, DB rows, and state files.
# Allow lowercase + digits with separators, but require alnum at both ends.
PLUGIN_ID_MAX_LENGTH = 64
PLUGIN_ID_REGEX = r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$"
_PLUGIN_ID_RE = re.compile(PLUGIN_ID_REGEX)

# Package names are used for pip installations and to locate installed directories.
_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Git refs are appended to VCS URLs after an '@'. Allow common tag/branch formats
# (including slashes), but disallow whitespace/control characters.
_GIT_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+-]{0,127}$")


def validate_plugin_id(plugin_id: str) -> None:
    """Validate a plugin ID.

    Raises:
        ValueError: If the ID is invalid.
    """
    if not plugin_id:
        raise ValueError("plugin_id is required")
    if len(plugin_id) > PLUGIN_ID_MAX_LENGTH:
        raise ValueError(f"plugin_id too long (max {PLUGIN_ID_MAX_LENGTH})")
    if not _PLUGIN_ID_RE.fullmatch(plugin_id):
        raise ValueError("Invalid plugin_id format")


def validate_git_ref(ref: str) -> None:
    """Validate a git ref/tag/branch name used for VCS installs.

    Notes:
        We intentionally allow common branch formats like "feature/foo" and tags
        like "v1.2.3". This does not aim to be a complete git-ref validator.

    Raises:
        ValueError: If the ref is invalid.
    """
    if not ref:
        raise ValueError("version is required")
    if any(ch.isspace() for ch in ref):
        raise ValueError("version must not contain whitespace")
    if ref.startswith("-"):
        raise ValueError("version must not start with '-'")
    if not _GIT_REF_RE.fullmatch(ref):
        raise ValueError("Invalid version/ref format")
    # Defensive: git forbids these sequences; treat as invalid here too.
    if ".." in ref or "@{" in ref:
        raise ValueError("Invalid version/ref format")


def validate_package_name(package_name: str) -> None:
    """Validate a package name used for install/uninstall operations.

    Raises:
        ValueError: If the package name is invalid.
    """
    if not package_name:
        raise ValueError("package_name is required")
    if any(ch.isspace() for ch in package_name):
        raise ValueError("package_name must not contain whitespace")
    if "/" in package_name or "\\" in package_name:
        raise ValueError("package_name must not contain path separators")
    if not _PACKAGE_NAME_RE.fullmatch(package_name):
        raise ValueError("Invalid package_name format")


def validate_pip_install_target(target: str) -> None:
    """Validate a pip install target passed as a single argv element.

    This function validates the *final* value passed to `pip install ... <target>`.

    Raises:
        ValueError: If the target looks unsafe or malformed.
    """
    if not target:
        raise ValueError("install target is required")
    if any(ch.isspace() for ch in target):
        raise ValueError("install target must not contain whitespace")
    # pip treats option-like args anywhere on the command line as flags.
    if target.startswith("-"):
        raise ValueError("install target must not start with '-'")
    if len(target) > 1024:
        raise ValueError("install target too long")
