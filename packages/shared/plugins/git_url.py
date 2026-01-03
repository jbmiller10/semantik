"""Git URL parsing utilities for plugin installation.

This module provides utilities for parsing and manipulating git URLs
for pip install commands, supporting HTTPS, SSH, and git+ prefixed URLs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GitUrl:
    """Parsed git URL with optional ref/version.

    Attributes:
        original: The original URL string as provided.
        scheme: URL scheme (e.g., "https", "git+https", "git+ssh", "ssh").
        host: The hostname (e.g., "github.com").
        path: The repository path (e.g., "org/repo.git").
        repo_name: The repository name extracted from path.
        ref: Optional git ref (tag, branch, commit) if present in URL.
    """

    original: str
    scheme: str
    host: str
    path: str
    repo_name: str
    ref: str | None = None

    def with_ref(self, ref: str) -> str:
        """Return URL with the specified version/ref appended.

        Handles ref replacement if one already exists.

        Args:
            ref: Git ref to append (e.g., "v1.0.0", "main", commit hash).

        Returns:
            The URL with the ref appended in pip-compatible format.

        Examples:
            >>> url = parse_git_url("git+https://github.com/org/repo.git")
            >>> url.with_ref("v1.0.0")
            'git+https://github.com/org/repo.git@v1.0.0'
        """
        base = self._normalize_for_pip()

        # Remove existing ref if present (after .git@ or after path@)
        if self.ref:
            # Remove the existing @ref portion
            if f".git@{self.ref}" in base:
                base = base.replace(f"@{self.ref}", "")
            elif f"@{self.ref}" in base and ".git" not in base:
                base = base.rsplit("@", 1)[0]

        # Ensure the base ends properly before adding ref
        return f"{base}@{ref}"

    def _normalize_for_pip(self) -> str:
        """Normalize URL to pip-compatible format.

        Converts SSH shorthand (git@host:path) to git+ssh:// format.

        Returns:
            Pip-compatible git URL.
        """
        # Handle SSH shorthand: git@host:path -> git+ssh://git@host/path
        if self.scheme == "ssh" and self.original.startswith("git@"):
            match = re.match(r"git@([^:]+):(.+)", self.original)
            if match:
                host, path = match.groups()
                # Remove any existing ref from path for normalization
                if self.ref and f"@{self.ref}" in path:
                    path = path.replace(f"@{self.ref}", "")
                return f"git+ssh://git@{host}/{path}"
        return self.original if not self.ref else self.original.replace(f"@{self.ref}", "")


# Regex patterns for git URL detection
_HTTPS_GIT_PATTERN = re.compile(r"^(git\+)?(https?)://([^/]+)/(.+?)(?:\.git)?(?:@([^@/]+))?$")

_SSH_SHORTHAND_PATTERN = re.compile(r"^git@([^:]+):(.+?)(?:\.git)?(?:@([^@]+))?$")

_GIT_SSH_PATTERN = re.compile(r"^git\+ssh://([^/]+)/(.+?)(?:\.git)?(?:@([^@/]+))?$")


def parse_git_url(url: str) -> GitUrl | None:
    """Parse a git URL for pip installation.

    Supports the following formats:
        - https://github.com/org/repo.git
        - https://github.com/org/repo
        - git+https://github.com/org/repo.git
        - git+https://github.com/org/repo.git@v1.0.0
        - git@github.com:org/repo.git
        - git@github.com:org/repo.git@v1.0.0
        - git+ssh://git@github.com/org/repo.git
        - git+ssh://git@github.com/org/repo.git@v1.0.0

    Args:
        url: The git URL to parse.

    Returns:
        A GitUrl object if the URL is a valid git URL, None otherwise.

    Examples:
        >>> result = parse_git_url("git+https://github.com/org/repo.git@v1.0.0")
        >>> result.host
        'github.com'
        >>> result.repo_name
        'repo'
        >>> result.ref
        'v1.0.0'
    """
    url = url.strip()

    # Try SSH shorthand: git@host:path
    match = _SSH_SHORTHAND_PATTERN.match(url)
    if match:
        host, path, ref = match.groups()
        # Extract repo name from path
        repo_name = path.rstrip(".git").split("/")[-1]
        return GitUrl(
            original=url,
            scheme="ssh",
            host=host,
            path=path,
            repo_name=repo_name,
            ref=ref,
        )

    # Try git+ssh:// format
    match = _GIT_SSH_PATTERN.match(url)
    if match:
        host, path, ref = match.groups()
        repo_name = path.rstrip(".git").split("/")[-1]
        return GitUrl(
            original=url,
            scheme="git+ssh",
            host=host,
            path=path,
            repo_name=repo_name,
            ref=ref,
        )

    # Try HTTPS format (with or without git+ prefix)
    match = _HTTPS_GIT_PATTERN.match(url)
    if match:
        git_prefix, protocol, host, path, ref = match.groups()
        scheme = f"git+{protocol}" if git_prefix else protocol
        repo_name = path.rstrip(".git").split("/")[-1]
        return GitUrl(
            original=url,
            scheme=scheme,
            host=host,
            path=path,
            repo_name=repo_name,
            ref=ref,
        )

    return None


def is_git_url(url: str) -> bool:
    """Check if URL is a valid git URL for pip installation.

    Args:
        url: The URL to check.

    Returns:
        True if the URL is a valid git URL, False otherwise.

    Examples:
        >>> is_git_url("git+https://github.com/org/repo.git")
        True
        >>> is_git_url("requests==2.28.0")
        False
    """
    return parse_git_url(url) is not None


def append_version_to_git_url(url: str, version: str) -> str:
    """Append a version/ref to a git URL, handling various formats.

    If the URL already has a ref, it will be replaced with the new version.

    Args:
        url: A git URL (may or may not have an existing ref).
        version: The version/ref to append (e.g., "v1.0.0", "main").

    Returns:
        The URL with the version appended.

    Raises:
        ValueError: If the URL is not a valid git URL.

    Examples:
        >>> append_version_to_git_url("git+https://github.com/org/repo.git", "v1.0.0")
        'git+https://github.com/org/repo.git@v1.0.0'

        >>> append_version_to_git_url("git+https://github.com/org/repo.git@main", "v2.0.0")
        'git+https://github.com/org/repo.git@v2.0.0'
    """
    parsed = parse_git_url(url)
    if parsed is None:
        raise ValueError(f"Invalid git URL: {url}")

    return parsed.with_ref(version)
