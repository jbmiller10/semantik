"""UnstructuredParserPlugin - Full-featured parser using the unstructured library.

This plugin handles complex document formats like PDF, DOCX, PPTX, HTML, and email
files using the unstructured library. It supports multiple parsing strategies and
element-level metadata extraction.

The unstructured library is imported lazily to avoid loading heavy dependencies
until actually needed.
"""

from __future__ import annotations

import io
import logging
import mimetypes
from pathlib import Path
from typing import Any, ClassVar

from shared.plugins.manifest import AgentHints
from shared.plugins.types.parser import (
    ExtractionFailedError,
    ParsedElement,
    ParserOutput,
    ParserPlugin,
    UnsupportedFormatError,
)

logger = logging.getLogger(__name__)


def _normalize_extension(ext: str | None) -> str:
    """Normalize a file extension to lowercase with leading dot."""
    if not ext:
        return ""
    ext = ext.strip().lower()
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    return ext


def _normalize_file_type(ext: str) -> str:
    """Convert extension to file type (extension without leading dot)."""
    return ext.lstrip(".").lower()


def _build_parser_metadata(
    *,
    parser_name: str,
    filename: str | None = None,
    file_extension: str | None = None,
    mime_type: str | None = None,
    caller_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata dict with protected keys."""
    ext_norm = _normalize_extension(file_extension)

    # Start with caller metadata (or empty dict)
    result = dict(caller_metadata or {})

    # Determine filename: explicit > default
    resolved_filename = filename or "document"

    # Overwrite with protected/required keys
    result["filename"] = resolved_filename
    result["file_extension"] = ext_norm
    result["file_type"] = _normalize_file_type(ext_norm)
    result["parser"] = parser_name

    # Determine MIME type: explicit > guess > default
    mime_norm = mime_type.strip().lower() if mime_type else None
    if mime_norm is None:
        guessed_mime: str | None = None
        if resolved_filename and resolved_filename != "document":
            guessed_mime, _ = mimetypes.guess_type(resolved_filename)
        if guessed_mime is None and ext_norm:
            guessed_mime, _ = mimetypes.guess_type(f"document{ext_norm}")
        result["mime_type"] = guessed_mime.lower() if guessed_mime else "application/octet-stream"
    else:
        result["mime_type"] = mime_norm

    return result


class UnstructuredParserPlugin(ParserPlugin):
    """Full-featured parser using the unstructured library.

    Supports PDF, DOCX, PPTX, HTML, EML, and more. Heavier dependency
    but handles complex document formats well.

    Dependency Policy:
        The unstructured library is imported lazily to avoid loading heavy
        dependencies until actually needed. If the library is missing or
        fails to import, ExtractionFailedError is raised (NOT UnsupportedFormatError).

        This distinction is intentional:
        - ExtractionFailedError: Pipeline will NOT fall back to TextParser
        - UnsupportedFormatError: Pipeline WILL fall back to TextParser

        Callers that want graceful fallback on missing dependencies should
        catch ExtractionFailedError explicitly and handle accordingly.

    Config options:
        strategy: "auto" | "fast" | "hi_res" | "ocr_only" (default: "auto")
        include_page_breaks: bool - Track page numbers (default: True)
        infer_table_structure: bool - Preserve tables (default: True)
    """

    PLUGIN_ID: ClassVar[str] = "unstructured"
    PLUGIN_TYPE: ClassVar[str] = "parser"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"

    METADATA: ClassVar[dict[str, Any]] = {
        "display_name": "Unstructured Parser",
        "description": "Full-featured parser for PDF, DOCX, PPTX, HTML, and email files.",
        "author": "Semantik",
        "requires": ["unstructured"],
    }

    SUPPORTED_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
        {
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".html",
            ".htm",
            ".eml",
            ".msg",
            ".txt",
            ".md",
            ".rst",
        }
    )

    SUPPORTED_MIME_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "text/html",
            "message/rfc822",
            "text/plain",
            "text/markdown",
        }
    )

    AGENT_HINTS: ClassVar[AgentHints] = AgentHints(
        purpose="Parse complex documents (PDF, DOCX, PPTX, HTML, email) with layout awareness.",
        best_for=[
            "PDF documents (including scanned with OCR)",
            "Microsoft Office files (DOCX, PPTX)",
            "HTML pages",
            "Email files (.eml, .msg)",
            "Documents requiring table extraction",
        ],
        not_recommended_for=[
            "Simple text files (use TextParser instead - faster)",
            "Large codebases (TextParser is more appropriate)",
            "JSON/YAML config files",
        ],
        input_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "text/html",
            "message/rfc822",
        ],
        output_type="text",
        tradeoffs=(
            "More accurate for complex documents but slower and requires additional dependencies. "
            "Use 'fast' strategy for speed, 'hi_res' for accuracy, 'ocr_only' for scanned docs."
        ),
        examples=[
            {
                "name": "Fast parsing",
                "description": "Quick extraction without OCR",
                "config": {"strategy": "fast"},
            },
            {
                "name": "High resolution",
                "description": "Best accuracy for complex layouts",
                "config": {"strategy": "hi_res", "infer_table_structure": True},
            },
            {
                "name": "OCR only",
                "description": "For scanned documents",
                "config": {"strategy": "ocr_only"},
            },
        ],
    )

    @classmethod
    def supported_extensions(cls) -> frozenset[str]:
        return cls.SUPPORTED_EXTENSIONS

    @classmethod
    def supported_mime_types(cls) -> frozenset[str]:
        return cls.SUPPORTED_MIME_TYPES

    @classmethod
    def get_config_options(cls) -> list[dict[str, Any]]:
        return [
            {
                "name": "strategy",
                "type": "select",
                "label": "Parsing Strategy",
                "options": [
                    {"value": "auto", "label": "Auto (recommended)"},
                    {"value": "fast", "label": "Fast (less accurate)"},
                    {"value": "hi_res", "label": "High Resolution (slower)"},
                    {"value": "ocr_only", "label": "OCR Only (scanned docs)"},
                ],
                "default": "auto",
            },
            {
                "name": "include_page_breaks",
                "type": "boolean",
                "label": "Track Page Numbers",
                "default": True,
            },
            {
                "name": "infer_table_structure",
                "type": "boolean",
                "label": "Preserve Table Structure",
                "default": True,
            },
        ]

    def parse_file(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
        *,
        include_elements: bool = False,
    ) -> ParserOutput:
        """Parse document from file path.

        Note: Reads entire file into memory before parsing.
        """
        path = Path(file_path)
        content = path.read_bytes()
        file_metadata = {"filename": path.name, **(metadata or {})}

        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        mime_type = mime_type or "application/octet-stream"

        return self.parse_bytes(
            content,
            filename=path.name,
            file_extension=path.suffix.lower(),
            mime_type=mime_type,
            metadata=file_metadata,
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
    ) -> ParserOutput:
        """Parse document content using unstructured.

        Raises:
            UnsupportedFormatError: If file extension/MIME type is not supported.
            ExtractionFailedError: If unstructured parsing fails.
        """
        ext_norm = _normalize_extension(file_extension)
        if (
            ext_norm
            and ext_norm not in self.supported_extensions()
            and (not mime_type or mime_type not in self.supported_mime_types())
        ):
            raise UnsupportedFormatError(f"UnstructuredParser does not support {ext_norm}")

        # Lazy import to avoid loading heavy dependency until needed
        try:
            from unstructured.partition.auto import partition
        except Exception as e:
            raise ExtractionFailedError("unstructured is not installed or failed to import", cause=e) from e

        strategy = str(self.config.get("strategy", "auto"))
        include_page_breaks = bool(self.config.get("include_page_breaks", True))
        infer_table_structure = bool(self.config.get("infer_table_structure", True))

        resolved_filename = filename or (metadata or {}).get("filename", "document")

        try:
            file_obj = io.BytesIO(content)
            elements = partition(
                file=file_obj,
                metadata_filename=resolved_filename,
                content_type=mime_type or "application/octet-stream",
                strategy=strategy,
                include_page_breaks=include_page_breaks,
                infer_table_structure=infer_table_structure,
            )
        except Exception as e:
            logger.error("Unstructured parsing failed for %s: %s", resolved_filename, e)
            raise ExtractionFailedError(f"Failed to parse {resolved_filename}: {e}", cause=e) from e

        parsed_elements: list[ParsedElement] = []
        text_parts: list[str] = []
        current_page = 1
        base_metadata = _build_parser_metadata(
            parser_name="unstructured",
            filename=resolved_filename,
            file_extension=ext_norm,
            mime_type=mime_type,
            caller_metadata=metadata,
        )

        for element in elements:
            text = str(element)
            if not text.strip():
                continue

            text_parts.append(text)
            elem_metadata: dict[str, Any] = {**base_metadata}

            if hasattr(element, "metadata"):
                elem_meta = element.metadata
                if hasattr(elem_meta, "page_number") and elem_meta.page_number:
                    elem_metadata["page_number"] = elem_meta.page_number
                    current_page = elem_meta.page_number
                else:
                    elem_metadata["page_number"] = current_page

                if hasattr(elem_meta, "category"):
                    elem_metadata["element_type"] = str(elem_meta.category)

            if include_elements:
                parsed_elements.append(ParsedElement(text=text, metadata=elem_metadata))

        return ParserOutput(
            text="\n\n".join(text_parts),
            elements=parsed_elements,
            metadata=base_metadata,
        )


__all__ = ["UnstructuredParserPlugin"]
