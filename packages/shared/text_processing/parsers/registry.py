from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BaseParser

# Registry of available parsers (self-initializing via ensure_registered()).
PARSER_REGISTRY: dict[str, type[BaseParser]] = {}
_REGISTERED = False


# Shared default parser map - connectors can override selectively
DEFAULT_PARSER_MAP: dict[str, str] = {
    # Binary document formats - use unstructured
    ".pdf": "unstructured",
    ".docx": "unstructured",
    ".doc": "unstructured",
    ".pptx": "unstructured",
    ".ppt": "unstructured",
    ".eml": "unstructured",
    ".msg": "unstructured",
    ".html": "unstructured",
    ".htm": "unstructured",
    # Text/code formats - use lightweight text parser
    ".txt": "text",
    ".text": "text",
    ".md": "text",
    ".markdown": "text",
    ".mdown": "text",
    ".mkd": "text",
    ".mdx": "text",
    ".rst": "text",
    ".adoc": "text",
    ".py": "text",
    ".js": "text",
    ".ts": "text",
    ".tsx": "text",
    ".jsx": "text",
    ".java": "text",
    ".go": "text",
    ".rs": "text",
    ".rb": "text",
    ".php": "text",
    ".c": "text",
    ".cpp": "text",
    ".h": "text",
    ".hpp": "text",
    ".cs": "text",
    ".sh": "text",
    ".bash": "text",
    ".zsh": "text",
    ".sql": "text",
    ".graphql": "text",
    ".json": "text",
    ".yaml": "text",
    ".yml": "text",
    ".toml": "text",
    ".ini": "text",
    ".cfg": "text",
    ".xml": "text",
    ".csv": "text",
    ".css": "text",
    ".scss": "text",
    ".less": "text",
}


def ensure_registered() -> None:
    """Ensure built-in parsers are registered exactly once.

    This avoids import-order pitfalls where importing registry.py directly
    would otherwise yield an empty PARSER_REGISTRY.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    # Local imports to avoid circular dependencies at module import time.
    from .text import TextParser
    from .unstructured import UnstructuredParser

    register_parser("text", TextParser)
    register_parser("unstructured", UnstructuredParser)
    _REGISTERED = True


def register_parser(name: str, cls: type[BaseParser]) -> None:
    """Register a parser class.

    Called explicitly (built-ins via ensure_registered()) rather than via
    decorators to avoid import side-effect issues.

    Args:
        name: Parser name for lookup.
        cls: Parser class.
    """
    PARSER_REGISTRY[name] = cls


def get_parser(name: str, config: dict[str, Any] | None = None) -> BaseParser:
    """Get parser instance by name.

    Args:
        name: Registered parser name.
        config: Parser configuration.

    Returns:
        Configured parser instance.

    Raises:
        ValueError: If parser name not found.
    """
    ensure_registered()
    if name not in PARSER_REGISTRY:
        available = ", ".join(sorted(PARSER_REGISTRY.keys()))
        raise ValueError(f"Unknown parser: {name}. Available: {available}")
    return PARSER_REGISTRY[name](config)


# Extensions that should try unstructured first, then fall back.
# Includes HTML by default (can change later).
UNSTRUCTURED_FIRST_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".eml", ".msg",
    ".html", ".htm",
})

# Extensions that should try TextParser first, then fall back.
TEXT_FIRST_EXTENSIONS: frozenset[str] = frozenset({
    ext for ext, parser in DEFAULT_PARSER_MAP.items() if parser == "text"
})


def parser_candidates_for_extension(
    ext: str,
    *,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    """Return parser names in priority order for an extension.

    Notes:
    - Overrides (if present) are tried first.
    - Unknown/no extension defaults to unstructured-first.
    - Actual fallback occurs when a parser raises UnsupportedFormatError.
    """
    ensure_registered()
    ext_norm = ext.lower() if ext.startswith(".") else (f".{ext.lower()}" if ext else "")

    override = (overrides or {}).get(ext_norm)
    default = DEFAULT_PARSER_MAP.get(ext_norm)

    # text-first for known text extensions; unstructured-first otherwise (binary/unknown)
    base = ["text", "unstructured"] if ext_norm in TEXT_FIRST_EXTENSIONS else ["unstructured", "text"]

    candidates: list[str] = []
    for name in (override, default, *base):
        if name and name not in candidates:
            candidates.append(name)
    return candidates


def list_parsers() -> list[str]:
    """List all registered parser names."""
    ensure_registered()
    return sorted(PARSER_REGISTRY.keys())


def list_parsers_for_extension(ext: str) -> list[str]:
    """List parser names that support a file extension.

    Useful for connector authors to see available options.

    Args:
        ext: File extension (with or without leading dot).

    Returns:
        List of parser names supporting this extension.
    """
    ensure_registered()
    ext = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
    return [
        name for name, cls in PARSER_REGISTRY.items()
        if ext in cls.supported_extensions()
    ]
