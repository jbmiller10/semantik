#!/usr/bin/env python3
"""
Document extraction module - handles text extraction from various file formats.
Uses unstructured library for unified document parsing.
"""

import io
import logging
from pathlib import Path
from typing import Any

# Unstructured for document parsing
from unstructured.partition.auto import partition

from .file_type_detector import FileTypeDetector

logger = logging.getLogger(__name__)


def _extension_to_content_type(ext: str) -> str | None:
    """Map file extension to MIME content type for unstructured.

    Args:
        ext: File extension with leading dot (e.g., ".txt")

    Returns:
        MIME content type string or None if not mapped
    """
    mapping = {
        ".txt": "text/plain",
        ".text": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".mdown": "text/markdown",
        ".mkd": "text/markdown",
        ".mdx": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
    }
    return mapping.get(ext)


def _decode_text_content(content: bytes) -> str:
    """Best-effort decode of text content bytes."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1", errors="ignore")


def parse_document_content(
    content: bytes | str,
    file_extension: str,
    metadata: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Parse document content from memory without file I/O.

    This function enables parsing of documents from any source (web, Slack, etc.)
    without requiring the content to be written to disk first.

    Args:
        content: Raw document content (bytes for binary formats, str for text)
        file_extension: File extension with dot (e.g., ".pdf", ".txt")
        metadata: Optional metadata to merge into each element's metadata

    Returns:
        List of (text, metadata) tuples, same format as extract_and_serialize

    Raises:
        Exception: If parsing fails
    """
    # Normalize extension
    ext = file_extension.lower() if file_extension.startswith(".") else f".{file_extension.lower()}"

    try:
        # Use file parameter for bytes, text parameter for strings
        if isinstance(content, bytes):
            file_obj = io.BytesIO(content)
            elements = partition(
                file=file_obj,
                metadata_filename=f"document{ext}",  # Helps unstructured detect type
                strategy="auto",
                include_page_breaks=True,
                infer_table_structure=True,
            )
        else:
            # For text content, use partition with text parameter
            elements = partition(
                text=content,
                content_type=_extension_to_content_type(ext),
                strategy="auto",
                include_page_breaks=True,
                infer_table_structure=True,
            )

        results = []
        current_page = 1
        base_metadata = metadata or {}

        for element in elements:
            text = str(element)
            if not text.strip():
                continue

            # Build metadata with file_type and merge provided metadata
            elem_metadata: dict[str, Any] = {
                "file_type": ext[1:] if ext else "unknown",
                **base_metadata,
            }

            # Add element-specific metadata
            if hasattr(element, "metadata"):
                elem_meta = element.metadata
                if hasattr(elem_meta, "page_number") and elem_meta.page_number:
                    elem_metadata["page_number"] = elem_meta.page_number
                    current_page = elem_meta.page_number
                else:
                    elem_metadata["page_number"] = current_page

                if hasattr(elem_meta, "category"):
                    elem_metadata["element_type"] = str(elem_meta.category)

                if hasattr(elem_meta, "coordinates"):
                    elem_metadata["has_coordinates"] = "True"

            results.append((text, elem_metadata))

        return results

    except Exception as e:
        logger.error(f"Failed to parse document content: {e}")
        raise


def extract_and_serialize(filepath: str) -> list[tuple[str, dict[str, Any]]]:
    """Uses unstructured to partition a file and serializes structured data.

    This function reads the file from disk and delegates parsing to
    parse_document_content for the actual extraction.

    Args:
        filepath: Path to the file to extract

    Returns:
        List of (text, metadata) tuples

    Raises:
        Exception: If file cannot be read or parsed
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    try:
        # Read file content
        with path.open("rb") as f:
            content = f.read()

        # For code/markdown/plain text, bypass unstructured to avoid heavy parsing
        if FileTypeDetector.is_code_file(ext) or FileTypeDetector.is_markdown_file(ext) or ext in {".txt", ".text"}:
            text = _decode_text_content(content)
            if not text.strip():
                return []
            return [(text, {"filename": path.name, "file_type": ext[1:] if ext else "unknown"})]

        # Parse using shared function
        return parse_document_content(
            content=content,
            file_extension=ext,
            metadata={"filename": path.name},
        )

    except Exception as e:
        logger.error(f"Failed to extract from {filepath}: {e}")
        raise


def extract_text(filepath: str, timeout: int = 300) -> str:  # noqa: ARG001
    """Legacy function for backward compatibility - extracts text without metadata
    Note: timeout parameter is kept for backward compatibility but not used"""
    try:
        results = extract_and_serialize(filepath)
        # Concatenate all text parts
        text_parts = [text for text, _ in results]
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to extract text from {filepath}: {e}")
        raise
