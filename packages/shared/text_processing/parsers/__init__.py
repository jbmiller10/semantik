"""Built-in document parsers.

Parsers extract text from documents. Most callers should use `parse_content()`
to parse `bytes | str` inputs with built-in selection and fallback behavior.

Example:
    from shared.text_processing.parsers import parse_content

    result = parse_content(
        b"...",
        filename="README.md",
        file_extension=".md",
        metadata={"source_type": "git", "source_path": "README.md"},
    )
    print(result.text)
"""

from typing import Any

from .base import BaseParser, ParsedElement, ParseResult
from .exceptions import ExtractionFailedError, ParserConfigError, ParserError, UnsupportedFormatError
from .normalization import (
    PROTECTED_METADATA_KEYS,
    build_parser_metadata,
    normalize_extension,
    normalize_file_type,
    normalize_mime_type,
)
from .registry import (
    ensure_registered,
    get_default_parser_map,
    get_parser,
    get_parser_registry,
    list_parsers,
    list_parsers_for_extension,
    parser_candidates_for_extension,
    register_parser,
)
from .text import TextParser
from .unstructured import UnstructuredParser

PARSER_REGISTRY = get_parser_registry()
DEFAULT_PARSER_MAP = get_default_parser_map()


def parse_content(
    content: bytes | str,
    *,
    filename: str | None = None,
    file_extension: str | None = None,
    mime_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    include_elements: bool = False,
    parser_overrides: dict[str, str] | None = None,
    parser_configs: dict[str, dict[str, Any]] | None = None,
) -> ParseResult:
    """Parse content with built-in selection and fallback.

    Args:
        content: Document content as bytes or str.
        filename: Optional filename hint for format detection.
        file_extension: Optional extension hint (e.g., ".pdf").
        mime_type: Optional MIME type hint.
        metadata: Optional metadata to include in result.
        include_elements: Whether to populate ParseResult.elements.
        parser_overrides: Extension-to-parser overrides (e.g., {".html": "text"}).
        parser_configs: Per-parser configs (e.g., {"unstructured": {"strategy": "fast"}}).

    Returns:
        ParseResult with extracted text and metadata.

    Raises:
        UnsupportedFormatError: If no parser can handle the content.
        ExtractionFailedError: If extraction fails.

    Notes:
    - For `str` inputs, this uses TextParser semantics (already-decoded text).
    - For `bytes` inputs, this tries candidate parsers in priority order and
      falls back only on UnsupportedFormatError.
    """
    ensure_registered()

    # If it's already text, treat it as TextParser output.
    if isinstance(content, str):
        ext_norm = normalize_extension(file_extension)
        base_meta = build_parser_metadata(
            parser_name="text",
            filename=filename,
            file_extension=ext_norm,
            mime_type=mime_type,
            caller_metadata=metadata,
        )
        return ParseResult(
            text=content,
            elements=[ParsedElement(text=content, metadata=base_meta)] if include_elements and content.strip() else [],
            metadata=base_meta,
        )

    ext_norm = normalize_extension(file_extension)
    candidates = parser_candidates_for_extension(ext_norm, overrides=parser_overrides)
    last_unsupported: UnsupportedFormatError | None = None
    for parser_name in candidates:
        parser = get_parser(parser_name, (parser_configs or {}).get(parser_name, {}))
        try:
            return parser.parse_bytes(
                content,
                filename=filename,
                file_extension=ext_norm,
                mime_type=mime_type,
                metadata=metadata,
                include_elements=include_elements,
            )
        except UnsupportedFormatError as exc:
            last_unsupported = exc
            continue

    raise last_unsupported or UnsupportedFormatError(f"No parser found for {ext_norm or '(no extension)'}")


__all__ = [
    # Base types
    "BaseParser",
    "ParsedElement",
    "ParseResult",
    # Exceptions
    "ParserError",
    "ParserConfigError",
    "UnsupportedFormatError",
    "ExtractionFailedError",
    # Normalization helpers
    "PROTECTED_METADATA_KEYS",
    "build_parser_metadata",
    "normalize_extension",
    "normalize_file_type",
    "normalize_mime_type",
    # Registry (prefer safe accessors over raw dicts)
    "get_parser_registry",
    "get_default_parser_map",
    "ensure_registered",
    "get_parser",
    "list_parsers",
    "list_parsers_for_extension",
    "register_parser",
    "parser_candidates_for_extension",
    "parse_content",
    # Built-in parsers
    "TextParser",
    "UnstructuredParser",
    # Legacy exports (prefer get_parser_registry/get_default_parser_map)
    "PARSER_REGISTRY",
    "DEFAULT_PARSER_MAP",
]
