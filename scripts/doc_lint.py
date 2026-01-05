#!/usr/bin/env python3
"""Lightweight doc linting for path references.

Rules:
- Scan markdown files in docs/.
- Find code-literal references like `packages/...` or `docs/...`.
- Warn when a referenced path does not exist on disk.
- Warnings only; exit code is always 0.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"

CODE_PATH_RE = re.compile(r"`((?:packages|apps|docs|scripts|tests|alembic|webui|shared|vecpipe)/[^`\s]+)`")

# Patterns that indicate template/example paths (not meant to exist)
# Matches: <name>, *, YYYYMMDD-style timestamps
TEMPLATE_INDICATORS = re.compile(r"<[^>]+>|\*|YYYY|HHMI|MMDD")

# Comment to ignore entire file from doc-lint
FILE_IGNORE_COMMENT = "<!-- doc-lint-ignore-file -->"


def iter_markdown_files() -> list[Path]:
    return [p for p in DOCS_DIR.rglob("*.md") if p.is_file()]


def normalize_path(raw: str) -> Path:
    # Strip trailing punctuation commonly adjacent to code ticks.
    cleaned = raw.rstrip(".,:;")
    # Remove line number references like :123 or :123-456
    cleaned = re.sub(r":\d+(-\d+)?$", "", cleaned)
    return (ROOT / cleaned).resolve()


def is_template_path(raw: str) -> bool:
    """Check if path contains template placeholders like <name> or glob patterns like *."""
    return bool(TEMPLATE_INDICATORS.search(raw))


def main() -> int:
    warnings = []
    for md_file in iter_markdown_files():
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as exc:
            warnings.append(f"WARN {md_file}: failed to read ({exc})")
            continue

        # Skip files with ignore comment
        if FILE_IGNORE_COMMENT in content:
            continue

        for match in CODE_PATH_RE.finditer(content):
            raw_path = match.group(1)
            # Skip template/example paths with placeholders or globs
            if is_template_path(raw_path):
                continue
            path = normalize_path(raw_path)
            if not path.exists():
                rel = os.path.relpath(md_file, ROOT)
                warnings.append(f"WARN {rel}: missing path `{raw_path}`")

    if warnings:
        print("Doc lint warnings:")
        for warn in warnings:
            print(warn)
    else:
        print("Doc lint: no missing paths detected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
