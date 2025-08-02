#!/usr/bin/env python3
"""
File type detection and optimal chunking configuration.

This module provides utilities for detecting file types and recommending
optimal chunking strategies.
"""

from pathlib import Path
from typing import Any


class FileTypeDetector:
    """Detect file types and recommend chunking configurations."""

    # Code file extensions
    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp",
        ".cs", ".rb", ".go", ".rs", ".php", ".swift", ".kt", ".scala",
        ".r", ".m", ".mm", ".lua", ".dart", ".jsx", ".tsx", ".vue",
        ".sql", ".sh", ".bash", ".zsh", ".ps1", ".yaml", ".yml",
        ".json", ".xml", ".html", ".css", ".scss", ".sass", ".less",
    }

    # Markdown file extensions
    MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd", ".mdx"}

    # Document file extensions
    DOCUMENT_EXTENSIONS = {
        ".txt", ".pdf", ".doc", ".docx", ".odt", ".rtf",
        ".tex", ".latex", ".rst", ".asciidoc", ".org",
    }

    # All supported extensions
    SUPPORTED_EXTENSIONS = CODE_EXTENSIONS | MARKDOWN_EXTENSIONS | DOCUMENT_EXTENSIONS

    @classmethod
    def get_file_type(cls, file_path: str) -> str | None:
        """Get the file type based on extension.

        Args:
            file_path: Path to the file

        Returns:
            File extension or None if not recognized
        """
        ext = Path(file_path.lower()).suffix
        return ext if ext in cls.SUPPORTED_EXTENSIONS else None

    @classmethod
    def is_code_file(cls, file_path: str) -> bool:
        """Check if file is a code file.

        Args:
            file_path: Path to the file or just the extension

        Returns:
            True if file is a code file
        """
        # Handle both full paths and just extensions
        if file_path.startswith('.') and '/' not in file_path:
            ext = file_path.lower()
        else:
            ext = Path(file_path.lower()).suffix
        return ext in cls.CODE_EXTENSIONS

    @classmethod
    def is_markdown_file(cls, file_path: str) -> bool:
        """Check if file is a markdown file.

        Args:
            file_path: Path to the file or just the extension

        Returns:
            True if file is a markdown file
        """
        # Handle both full paths and just extensions
        if file_path.startswith('.') and '/' not in file_path:
            ext = file_path.lower()
        else:
            ext = Path(file_path.lower()).suffix
        return ext in cls.MARKDOWN_EXTENSIONS

    @classmethod
    def get_optimal_config(cls, file_path: str) -> dict[str, Any]:
        """Get optimal chunking configuration for a file type.

        Args:
            file_path: Path to the file or just the extension

        Returns:
            Optimal chunking configuration
        """
        # Handle both full paths and just extensions
        if file_path.startswith('.') and '/' not in file_path:
            # It's just an extension
            ext = file_path.lower()
        else:
            # It's a path, extract extension
            ext = Path(file_path.lower()).suffix

        # Markdown files
        if ext in cls.MARKDOWN_EXTENSIONS:
            return {
                "strategy": "markdown",
                "params": {},
            }

        # Code files - use recursive with optimized parameters
        if ext in cls.CODE_EXTENSIONS:
            return {
                "strategy": "recursive",
                "params": {
                    "chunk_size": 400,
                    "chunk_overlap": 50,
                },
            }

        # Default for documents and unknown files
        return {
                "strategy": "recursive",
                "params": {
                    "chunk_size": 300,
                    "chunk_overlap": 50,
                },
            }

    @classmethod
    def get_file_category(cls, file_path: str) -> str:
        """Get the category of a file.

        Args:
            file_path: Path to the file

        Returns:
            File category: 'code', 'markdown', 'document', or 'unknown'
        """
        ext = Path(file_path.lower()).suffix

        if ext in cls.CODE_EXTENSIONS:
            return "code"
        if ext in cls.MARKDOWN_EXTENSIONS:
            return "markdown"
        if ext in cls.DOCUMENT_EXTENSIONS:
            return "document"
        return "unknown"
