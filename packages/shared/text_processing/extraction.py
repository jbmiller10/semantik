#!/usr/bin/env python3
"""
Document extraction module - handles text extraction from various file formats.
Uses unstructured library for unified document parsing.
"""

import logging
from pathlib import Path
from typing import Any

# Unstructured for document parsing
from unstructured.partition.auto import partition

logger = logging.getLogger(__name__)


def extract_and_serialize(filepath: str) -> list[tuple[str, dict[str, Any]]]:
    """Uses unstructured to partition a file and serializes structured data.
    Returns list of (text, metadata) tuples."""
    ext = Path(filepath).suffix.lower()

    # Use unstructured for all file types
    try:
        elements = partition(
            filename=filepath,
            strategy="auto",  # Let unstructured determine the best strategy
            include_page_breaks=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
        )

        results = []
        current_page = 1

        for element in elements:
            # Extract text content
            text = str(element)
            if not text.strip():
                continue

            # Build metadata
            metadata: dict[str, Any] = {"filename": Path(filepath).name, "file_type": ext[1:] if ext else "unknown"}

            # Add element-specific metadata
            if hasattr(element, "metadata"):
                elem_meta = element.metadata
                if hasattr(elem_meta, "page_number") and elem_meta.page_number:
                    metadata["page_number"] = elem_meta.page_number
                    current_page = elem_meta.page_number
                else:
                    metadata["page_number"] = current_page

                if hasattr(elem_meta, "category"):
                    metadata["element_type"] = str(elem_meta.category)

                # Add any coordinates if available
                if hasattr(elem_meta, "coordinates"):
                    metadata["has_coordinates"] = "True"

            results.append((text, metadata))

        return results

    except Exception as e:
        logger.error(f"Failed to extract from {filepath} using unstructured: {e}")
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
