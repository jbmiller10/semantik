"""Base connector interface for document ingestion sources."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from shared.dtos.ingestion import IngestedDocument


class BaseConnector(ABC):
    """Abstract base class for all document source connectors.

    Connectors are responsible for:
    1. Authenticating with the source (if required)
    2. Discovering documents from the source
    3. Yielding IngestedDocument DTOs for registration

    Subclasses must implement:
    - authenticate(): Perform any auth/handshake
    - load_documents(): Yield IngestedDocument objects

    Example:
        ```python
        connector = LocalFileConnector({"source_path": "/data/docs"})
        if await connector.authenticate():
            async for doc in connector.load_documents():
                result = await registry.register(collection_id, doc)
        ```
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the connector with configuration.

        Args:
            config: Connector-specific configuration dictionary.
                    Each connector defines its required/optional keys.
        """
        self._config = config
        self.validate_config()

    @property
    def config(self) -> dict[str, Any]:
        """Return the connector configuration."""
        return self._config

    def validate_config(self) -> None:  # noqa: B027
        """Validate the configuration dictionary.

        Override in subclasses to validate required keys.
        Raises ValueError if configuration is invalid.

        Default implementation does nothing (all configs valid).
        """

    @abstractmethod
    async def authenticate(self) -> bool:
        """Perform authentication or connection handshake.

        Returns:
            True if authentication succeeded, False otherwise.

        Raises:
            Exception: If authentication fails with an error.
        """
        ...

    @abstractmethod
    def load_documents(self) -> AsyncIterator[IngestedDocument]:
        """Yield documents from the source.

        This is an async generator that yields IngestedDocument objects.
        Each document should have:
        - content: Parsed text content
        - unique_id: Source-specific identifier
        - source_type: Connector type identifier
        - metadata: Source-specific metadata
        - content_hash: SHA-256 hash of content

        Yields:
            IngestedDocument: Documents discovered from the source.

        Raises:
            Exception: If document loading fails.
        """
        ...
