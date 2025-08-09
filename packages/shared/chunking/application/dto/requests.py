"""
Request DTOs for chunking application layer.

These DTOs define the input contracts for use cases.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def validate_file_path_security(file_path: str) -> None:
    """
    Validate file path for security issues.

    Args:
        file_path: Path to validate

    Raises:
        ValueError: If path contains security issues
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    # Convert to Path object for proper validation
    path = Path(file_path)

    # Check for path traversal attempts
    if ".." in path.parts:
        raise ValueError("Path traversal detected: '..' not allowed in file paths")

    # Check if path is absolute and trying to access sensitive locations
    if path.is_absolute():
        # Define sensitive directories that should not be accessed
        sensitive_prefixes = [
            "/etc", "/sys", "/proc", "/boot", "/root",
            "/usr/bin", "/usr/sbin", "/bin", "/sbin",
            "C:\\Windows", "C:\\Program Files", "C:\\ProgramData"
        ]

        path_str = str(path).replace("\\", "/")
        for prefix in sensitive_prefixes:
            if path_str.startswith(prefix.replace("\\", "/")):
                raise ValueError(f"Access to system directory not allowed: {prefix}")

    # Check for null bytes which could be used for path injection
    if "\x00" in file_path:
        raise ValueError("Null bytes not allowed in file paths")

    # Check for suspicious patterns
    suspicious_patterns = ["../", "..\\", "~", "${", "$(", "`"]
    for pattern in suspicious_patterns:
        if pattern in file_path:
            raise ValueError(f"Suspicious pattern '{pattern}' detected in file path")


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    CHARACTER = "character"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"


@dataclass
class PreviewRequest:
    """Input DTO for preview chunking use case."""

    file_path: str
    strategy_type: ChunkingStrategy
    min_tokens: int = 100
    max_tokens: int = 1000
    overlap: int = 50
    preview_size_kb: int = 10  # Size of document to preview (in KB)
    max_preview_chunks: int = 5  # Maximum chunks to return in preview

    def validate(self) -> None:
        """Validate request parameters."""
        # Validate file path security
        validate_file_path_security(self.file_path)

        if self.min_tokens <= 0:
            raise ValueError("min_tokens must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens cannot be greater than max_tokens")
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")
        if self.overlap >= self.min_tokens:
            raise ValueError("overlap must be less than min_tokens")
        if self.preview_size_kb <= 0:
            raise ValueError("preview_size_kb must be positive")
        if self.max_preview_chunks <= 0:
            raise ValueError("max_preview_chunks must be positive")


@dataclass
class ProcessDocumentRequest:
    """Input DTO for document processing use case."""

    document_id: str
    file_path: str
    collection_id: str
    strategy_type: ChunkingStrategy
    min_tokens: int = 100
    max_tokens: int = 1000
    overlap: int = 50
    metadata: dict[str, Any] | None = None
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100  # Checkpoint every N chunks

    def validate(self) -> None:
        """Validate request parameters."""
        if not self.document_id:
            raise ValueError("document_id is required")
        if not self.file_path:
            raise ValueError("file_path is required")
        if not self.collection_id:
            raise ValueError("collection_id is required")

        # Validate file path security
        validate_file_path_security(self.file_path)

        if self.min_tokens <= 0:
            raise ValueError("min_tokens must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens cannot be greater than max_tokens")
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")
        if self.overlap >= self.min_tokens:
            raise ValueError("overlap must be less than min_tokens")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")


@dataclass
class CompareStrategiesRequest:
    """Input DTO for strategy comparison use case."""

    file_path: str
    strategies: list[ChunkingStrategy]
    min_tokens: int = 100
    max_tokens: int = 1000
    overlap: int = 50
    sample_size_kb: int = 50  # Size of document sample for comparison

    def validate(self) -> None:
        """Validate request parameters."""
        if not self.file_path:
            raise ValueError("file_path is required")

        # Validate file path security
        validate_file_path_security(self.file_path)

        if not self.strategies:
            raise ValueError("At least one strategy must be specified")
        # Allow single strategy for analysis (not comparison)
        # if len(self.strategies) < 2:
        #     raise ValueError("At least two strategies required for comparison")
        if self.min_tokens <= 0:
            raise ValueError("min_tokens must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens cannot be greater than max_tokens")
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")
        if self.overlap >= self.min_tokens:
            raise ValueError("overlap must be less than min_tokens")
        if self.sample_size_kb <= 0:
            raise ValueError("sample_size_kb must be positive")


@dataclass
class GetOperationStatusRequest:
    """Input DTO for operation status query."""

    operation_id: str | None = None  # Operation ID (optional if document_id provided)
    include_chunks: bool = False  # Whether to include chunk details
    include_metrics: bool = True  # Whether to include processing metrics
    document_id: str | None = None  # Optional document ID for finding operations

    def validate(self) -> None:
        """Validate request parameters."""
        if not self.operation_id and not self.document_id:
            raise ValueError("Either operation_id or document_id is required")
        
        # If operation_id is provided as empty string, treat as None
        if self.operation_id == "":
            self.operation_id = None


@dataclass
class CancelOperationRequest:
    """Input DTO for operation cancellation."""

    operation_id: str
    reason: str | None = None
    force: bool = False  # Force cancellation even if partially complete
    cleanup_chunks: bool = True  # Whether to delete already created chunks

    def validate(self) -> None:
        """Validate request parameters."""
        if not self.operation_id:
            raise ValueError("operation_id is required")
