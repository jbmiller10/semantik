from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, ClassVar

from .base import BaseParser, ParsedElement, ParseResult
from .exceptions import ExtractionFailedError, UnsupportedFormatError
from .normalization import build_parser_metadata, normalize_extension

logger = logging.getLogger(__name__)


class UnstructuredParser(BaseParser):
    """Full-featured parser using the unstructured library.

    Supports PDF, DOCX, PPTX, HTML, EML, and more. Heavier dependency
    but handles complex document formats well.

    Dependency Policy:
        The unstructured library is imported lazily to avoid loading heavy
        dependencies until actually needed. If the library is missing or
        fails to import, ExtractionFailedError is raised (NOT UnsupportedFormatError).

        This distinction is intentional:
        - ExtractionFailedError: parse_content() will NOT fall back to TextParser
        - UnsupportedFormatError: parse_content() WILL fall back to TextParser

        Callers that want graceful fallback on missing dependencies should
        catch ExtractionFailedError explicitly and handle accordingly.

    Config options:
        strategy: "auto" | "fast" | "hi_res" | "ocr_only" (default: "auto")
        include_page_breaks: bool - Track page numbers (default: True)
        infer_table_structure: bool - Preserve tables (default: True)
    """

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
    ) -> ParseResult:
        """Parse document from file path.

        Note: Reads entire file into memory before parsing.
        """
        path = Path(file_path)
        content = path.read_bytes()
        file_metadata = {"filename": path.name, **(metadata or {})}

        # Detect MIME type
        import mimetypes

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
    ) -> ParseResult:
        """Parse document content using unstructured.

        Raises:
            UnsupportedFormatError: If file extension/MIME type is not supported.
            ExtractionFailedError: If unstructured parsing fails.
        """
        ext_norm = normalize_extension(file_extension)
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
            logger.error(f"Unstructured parsing failed for {resolved_filename}: {e}")
            raise ExtractionFailedError(f"Failed to parse {resolved_filename}: {e}", cause=e) from e

        parsed_elements: list[ParsedElement] = []
        text_parts: list[str] = []
        current_page = 1
        base_metadata = build_parser_metadata(
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

        return ParseResult(
            text="\n\n".join(text_parts),
            elements=parsed_elements,
            metadata=base_metadata,
        )
