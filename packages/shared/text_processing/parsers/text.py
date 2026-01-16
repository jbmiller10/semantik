from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from .base import BaseParser, ParsedElement, ParseResult
from .exceptions import ExtractionFailedError, UnsupportedFormatError


class TextParser(BaseParser):
    """Lightweight parser for plain text files.

    No external dependencies. Fast and simple. Use for text, markdown,
    code files, JSON, YAML, etc.

    Config options:
        encoding: str - Text encoding (default: "utf-8")
        errors: str - Encoding error handling (default: "replace")
    """

    SUPPORTED_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({
        # Plain text
        ".txt", ".text",
        # Markdown
        ".md", ".markdown", ".mdown", ".mkd", ".mdx",
        # Code
        ".py", ".js", ".ts", ".tsx", ".jsx",
        ".java", ".go", ".rs", ".rb", ".php",
        ".c", ".cpp", ".h", ".hpp", ".cs",
        ".sh", ".bash", ".zsh",
        ".sql", ".graphql",
        # Config/data
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".xml", ".csv",
        # Web
        ".css", ".scss", ".less",
        # Documentation
        ".rst", ".adoc",
    })

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

        Raises:
            ExtractionFailedError: If file cannot be read or decoded.
        """
        path = Path(file_path)
        encoding = str(self.config.get("encoding", "utf-8"))
        errors = str(self.config.get("errors", "replace"))

        try:
            content = path.read_text(encoding=encoding, errors=errors)
        except Exception as e:
            raise ExtractionFailedError(f"Failed to read {path.name}: {e}", cause=e) from e

        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(path))

        file_metadata = {
            "filename": path.name,
            "file_extension": path.suffix.lower(),
            "file_type": path.suffix.lstrip(".").lower(),
            "mime_type": mime_type,
            "parser": "text",
            **(metadata or {}),
        }

        return ParseResult(
            text=content,
            elements=[ParsedElement(text=content, metadata=file_metadata)] if include_elements and content.strip() else [],
            metadata=file_metadata,
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
        encoding = str(self.config.get("encoding", "utf-8"))
        errors = str(self.config.get("errors", "replace"))

        # Strict binary sniff (avoid indexing garbage mojibake)
        if b"\x00" in content:
            raise UnsupportedFormatError("TextParser cannot decode binary content (NUL byte found)")
        sample = content[:8192]
        non_printable = sum(1 for b in sample if b < 9 or (13 < b < 32))
        if sample and (non_printable / len(sample)) > 0.30:
            raise UnsupportedFormatError("TextParser cannot decode binary content (non-printable ratio)")

        try:
            text = content.decode(encoding, errors=errors)
        except Exception as e:
            raise ExtractionFailedError(f"Failed to decode content: {e}", cause=e) from e

        ext_norm = (file_extension or "").lower()
        base_metadata = {
            "filename": filename or (metadata or {}).get("filename", "document"),
            "file_extension": ext_norm,
            "file_type": ext_norm.lstrip("."),
            "mime_type": mime_type,
            "parser": "text",
            **(metadata or {}),
        }

        return ParseResult(
            text=text,
            elements=[ParsedElement(text=text, metadata=base_metadata)] if include_elements and text.strip() else [],
            metadata=base_metadata,
        )
