"""Data transfer objects for document ingestion pipeline."""

from dataclasses import dataclass
from typing import Any


@dataclass
class IngestedDocument:
    """Represents a document ingested from any source (file, web, Slack, etc.).

    This is the unified data contract that all connectors must produce.
    The DocumentRegistryService consumes these objects for registration.

    Attributes:
        content: Fully parsed text content (post-extraction)
        unique_id: Logical identifier (URI, file path, message ID, etc.)
        source_type: Source type identifier (e.g., "directory", "web", "slack")
        metadata: Raw/original metadata from the connector
        content_hash: SHA-256 hash of content (64 lowercase hex chars)
        file_path: Optional local file path for file-based sources
    """

    content: str
    unique_id: str
    source_type: str
    metadata: dict[str, Any]
    content_hash: str
    file_path: str | None = None

    def __post_init__(self) -> None:
        """Validate hash format after initialization."""
        if len(self.content_hash) != 64 or not all(c in "0123456789abcdef" for c in self.content_hash):
            raise ValueError(f"content_hash must be a 64-character lowercase hex string, got: {self.content_hash!r}")
