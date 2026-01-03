#!/usr/bin/env python3
"""Bump version across all version files in the project.

Usage:
    ./scripts/bump_version.py patch     # 0.7.7 -> 0.7.8
    ./scripts/bump_version.py minor     # 0.7.7 -> 0.8.0
    ./scripts/bump_version.py major     # 0.7.7 -> 1.0.0
    ./scripts/bump_version.py 1.2.3     # Set explicit version
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent

# Files that contain version strings to update
VERSION_FILES = [
    ("VERSION", r"^(.*)$", r"{version}"),
    ("pyproject.toml", r'^version = "([^"]+)"', 'version = "{version}"'),
    ("packages/cli/pyproject.toml", r'^version = "([^"]+)"', 'version = "{version}"'),
]


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse a semver string into (major, minor, patch)."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version.strip())
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def format_version(major: int, minor: int, patch: int) -> str:
    """Format version tuple as string."""
    return f"{major}.{minor}.{patch}"


def get_current_version() -> str:
    """Read current version from VERSION file."""
    version_file = PROJECT_ROOT / "VERSION"
    return version_file.read_text().strip()


def bump_version(current: str, bump_type: str) -> str:
    """Calculate new version based on bump type."""
    major, minor, patch = parse_version(current)

    if bump_type == "patch":
        return format_version(major, minor, patch + 1)
    elif bump_type == "minor":
        return format_version(major, minor + 1, 0)
    elif bump_type == "major":
        return format_version(major + 1, 0, 0)
    else:
        # Assume it's an explicit version
        parse_version(bump_type)  # Validate format
        return bump_type


def update_file(filepath: Path, pattern: str, replacement: str, new_version: str) -> bool:
    """Update version in a single file. Returns True if file was modified."""
    if not filepath.exists():
        print(f"  Warning: {filepath} not found, skipping")
        return False

    content = filepath.read_text()
    new_replacement = replacement.format(version=new_version)

    # For VERSION file, just replace the whole content
    if filepath.name == "VERSION":
        new_content = new_version + "\n"
    else:
        new_content = re.sub(pattern, new_replacement, content, count=1, flags=re.MULTILINE)

    if content == new_content:
        print(f"  {filepath.relative_to(PROJECT_ROOT)}: no change needed")
        return False

    filepath.write_text(new_content)
    print(f"  {filepath.relative_to(PROJECT_ROOT)}: updated")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    bump_type = sys.argv[1]

    if bump_type in ("-h", "--help"):
        print(__doc__)
        return 0

    try:
        current_version = get_current_version()
        new_version = bump_version(current_version, bump_type)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"Bumping version: {current_version} -> {new_version}\n")

    updated_count = 0
    for filename, pattern, replacement in VERSION_FILES:
        filepath = PROJECT_ROOT / filename
        if update_file(filepath, pattern, replacement, new_version):
            updated_count += 1

    print(f"\nUpdated {updated_count} file(s)")
    print(f"\nNext steps:")
    print(f"  git add -u && git commit -m 'chore: bump version to {new_version}'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
