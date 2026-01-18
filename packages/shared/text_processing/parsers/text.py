from __future__ import annotations

import codecs
from pathlib import Path
from typing import Any, ClassVar

from .base import BaseParser, ParsedElement, ParseResult
from .exceptions import ExtractionFailedError, UnsupportedFormatError
from .normalization import build_parser_metadata, normalize_extension

# BOM signatures in detection order (longer first to avoid UTF-16 matching UTF-32).
# UTF-32-LE BOM (FF FE 00 00) starts with UTF-16-LE BOM (FF FE), so check 4-byte first.
_BOM_ENCODINGS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),  # FF FE 00 00 (4 bytes)
    (codecs.BOM_UTF32_BE, "utf-32-be"),  # 00 00 FE FF (4 bytes)
    (codecs.BOM_UTF16_LE, "utf-16-le"),  # FF FE (2 bytes)
    (codecs.BOM_UTF16_BE, "utf-16-be"),  # FE FF (2 bytes)
    (codecs.BOM_UTF8, "utf-8-sig"),  # EF BB BF (3 bytes)
)


def _detect_bom(content: bytes) -> tuple[str, int] | None:
    """Detect BOM and return (encoding, bom_length) or None.

    Checks BOMs in order from longest to shortest to avoid
    misdetecting UTF-32-LE as UTF-16-LE.
    """
    for bom, encoding in _BOM_ENCODINGS:
        if content.startswith(bom):
            return (encoding, len(bom))
    return None


def _is_binary_content(content: bytes) -> bool:
    """Check if content appears to be binary (BOM-aware).

    Returns True if content is binary and should be rejected.
    BOM-marked UTF-16/32 files are NOT binary, even though they contain NUL bytes.
    """
    # First check for BOM - if present, this is text, not binary
    bom_info = _detect_bom(content)
    if bom_info is not None:
        # BOM detected - this is encoded text, not binary
        return False

    # No BOM: check for NUL bytes (common in binary files)
    if b"\x00" in content:
        return True

    # Check non-printable ratio in first 8KB
    sample = content[:8192]
    if not sample:
        return False

    # Count bytes that are non-printable (exclude tab=9, LF=10, CR=13)
    non_printable = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return (non_printable / len(sample)) > 0.30


class TextParser(BaseParser):
    """Lightweight parser for plain text files.

    No external dependencies. Fast and simple. Use for text, markdown,
    code files, JSON, YAML, etc.

    Config options:
        encoding: str - Text encoding (default: "utf-8")
        errors: str - Encoding error handling (default: "replace")
    """

    SUPPORTED_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
        {
            # Plain text
            ".txt",
            ".text",
            # Markdown
            ".md",
            ".markdown",
            ".mdown",
            ".mkd",
            ".mdx",
            # Code
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".java",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".cs",
            ".sh",
            ".bash",
            ".zsh",
            ".sql",
            ".graphql",
            # Config/data
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".xml",
            ".csv",
            # Web
            ".css",
            ".scss",
            ".less",
            # Documentation
            ".rst",
            ".adoc",
        }
    )

    @classmethod
    def supported_extensions(cls) -> frozenset[str]:
        return cls.SUPPORTED_EXTENSIONS

    @classmethod
    def get_config_options(cls) -> list[dict[str, Any]]:
        return [
            {
                "name": "encoding",
                "type": "text",
                "label": "Text Encoding",
                "default": "utf-8",
            },
            {
                "name": "errors",
                "type": "select",
                "label": "Encoding Error Handling",
                "options": [
                    {"value": "replace", "label": "Replace (recommended)"},
                    {"value": "ignore", "label": "Ignore"},
                    {"value": "strict", "label": "Strict (raise error)"},
                ],
                "default": "replace",
            },
        ]

    def parse_file(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
        *,
        include_elements: bool = False,
    ) -> ParseResult:
        """Parse text file.

        Delegates to parse_bytes() for consistent BOM detection and binary rejection.

        Raises:
            UnsupportedFormatError: If file content looks like binary.
            ExtractionFailedError: If file cannot be read or decoded.
        """
        path = Path(file_path)

        try:
            content = path.read_bytes()
        except Exception as e:
            raise ExtractionFailedError(f"Failed to read {path.name}: {e}", cause=e) from e

        import mimetypes

        mime_type, _ = mimetypes.guess_type(str(path))

        return self.parse_bytes(
            content,
            filename=path.name,
            file_extension=path.suffix,
            mime_type=mime_type,
            metadata=metadata,
            include_elements=include_elements,
        )

    def parse_bytes(
        self,
        content: bytes,
        *,
        filename: str | None = None,
        file_extension: str | None = None,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        include_elements: bool = False,
    ) -> ParseResult:
        """Parse text content from bytes.

        Supports BOM-marked text files (UTF-8, UTF-16, UTF-32). For non-BOM
        files, uses the configured encoding (default: utf-8).

        Args:
            content: Raw bytes to decode.
            filename: Optional filename hint.
            file_extension: Optional extension hint.
            mime_type: Optional MIME type hint.
            metadata: Optional metadata to include.
            include_elements: Whether to populate ParseResult.elements.

        Raises:
            UnsupportedFormatError: If content looks like binary.
            ExtractionFailedError: If decoding fails with strict error handling.
        """
        # BOM-aware binary detection
        if _is_binary_content(content):
            raise UnsupportedFormatError("TextParser cannot decode binary content")

        errors = str(self.config.get("errors", "replace"))

        # Detect BOM for encoding selection
        bom_info = _detect_bom(content)
        if bom_info is not None:
            encoding, bom_length = bom_info
            # For UTF-8-sig, Python's codec handles BOM stripping automatically.
            # For UTF-16/32 variants, we manually skip the BOM bytes before decode.
            # utf-8-sig codec strips BOM automatically; UTF-16/32 need manual skip
            bytes_to_decode = content if encoding == "utf-8-sig" else content[bom_length:]
        else:
            # No BOM: use configured encoding (default: utf-8)
            encoding = str(self.config.get("encoding", "utf-8"))
            bytes_to_decode = content

        try:
            text = bytes_to_decode.decode(encoding, errors=errors)
        except Exception as e:
            raise ExtractionFailedError(f"Failed to decode content: {e}", cause=e) from e

        ext_norm = normalize_extension(file_extension)
        base_metadata = build_parser_metadata(
            parser_name="text",
            filename=filename,
            file_extension=ext_norm,
            mime_type=mime_type,
            caller_metadata=metadata,
        )

        return ParseResult(
            text=text,
            elements=[ParsedElement(text=text, metadata=base_metadata)] if include_elements and text.strip() else [],
            metadata=base_metadata,
        )
