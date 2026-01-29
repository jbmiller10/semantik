"""Base connector interface for document ingestion sources."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from shared.dtos.ingestion import IngestedDocument
from shared.pipeline.types import FileReference
from shared.plugins.manifest import PluginManifest


class BaseConnector(ABC):
    """Abstract base class for all document source connectors.

    Connectors are responsible for:
    1. Authenticating with the source (if required)
    2. Enumerating files from the source (yielding FileReference objects)

    Connectors do NOT load or parse content - they only enumerate files.
    Content loading and parsing is handled by the pipeline executor.

    Subclasses must implement:
    - authenticate(): Perform any auth/handshake
    - enumerate(): Yield FileReference objects describing available files

    Example:
        ```python
        connector = LocalFileConnector({"path": "/data/docs"})
        if await connector.authenticate():
            async for file_ref in connector.enumerate():
                print(file_ref.uri, file_ref.change_hint)
        ```

    Note:
        load_documents() is deprecated and will be removed in a future release.
        Use enumerate() instead.
    """

    PLUGIN_ID: ClassVar[str] = ""
    PLUGIN_TYPE: ClassVar[str] = "connector"
    PLUGIN_VERSION: ClassVar[str] = "0.0.0"
    METADATA: ClassVar[dict[str, Any]] = {}

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

    @classmethod
    def get_config_fields(cls) -> list[dict[str, Any]]:
        """Return list of config field definitions for UI consumption."""
        return []

    @classmethod
    def get_secret_fields(cls) -> list[dict[str, Any]]:
        """Return list of secret field definitions for UI consumption."""
        return []

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin manifest for discovery and UI.

        Builds a PluginManifest from the connector's class variables.
        Subclasses may override for custom manifest generation.

        Returns:
            PluginManifest with connector metadata.
        """
        metadata = cls.METADATA or {}
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name=metadata.get("display_name", cls.PLUGIN_ID),
            description=metadata.get("description", ""),
            author=metadata.get("author"),
            homepage=metadata.get("homepage"),
        )

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
    def enumerate(
        self,
        source_id: int | None = None,
    ) -> AsyncIterator[FileReference]:
        """Yield file references from the source.

        This is an async generator that yields FileReference objects
        describing files available from the source. Enumeration only -
        no content loading or parsing should occur.

        Must use lazy iteration (no upfront collection into memory).

        Each FileReference should have:
        - uri: Unique identifier (file://, git://, imap://, etc.)
        - source_type: Connector plugin ID
        - content_type: Semantic type (document, message, code)
        - filename, extension, mime_type, size_bytes
        - change_hint: For change detection (mtime+size, blob SHA, UID)
        - source_metadata: Connector-specific data for content loading later

        Args:
            source_id: Optional source ID for caching/uniqueness

        Yields:
            FileReference: File references discovered from the source.

        Raises:
            Exception: If enumeration fails.
        """
        ...

    @abstractmethod
    async def load_content(self, file_ref: FileReference) -> bytes:
        """Load raw content bytes for a file reference.

        This method is called by the pipeline executor to load the actual
        content of files enumerated by this connector. The connector knows
        how to retrieve content based on the source_metadata it populated
        during enumeration.

        For local files, this typically reads from source_metadata["local_path"].
        For remote sources like IMAP, this may require network requests.

        Args:
            file_ref: File reference previously yielded by enumerate()

        Returns:
            Raw content bytes

        Raises:
            ValueError: If the file reference is invalid
            OSError: If content cannot be loaded
        """
        ...

    def get_skipped_files(self) -> list[tuple[str, str]]:
        """Get list of files that were skipped during enumeration.

        This is an optional method that connectors may implement to report
        files that couldn't be enumerated (e.g., permission denied, broken
        symlinks). The default implementation returns an empty list.

        Returns:
            List of (path, reason) tuples for skipped files.
        """
        return []

    def load_documents(
        self,
        source_id: int | None = None,
    ) -> AsyncIterator[IngestedDocument]:
        """Yield documents from the source.

        .. deprecated::
            This method is deprecated and will be removed in a future release.
            Use enumerate() instead. Content loading and parsing has been
            moved to the pipeline executor.

        Raises:
            NotImplementedError: Always raises as this method is deprecated.
        """
        raise NotImplementedError(
            "load_documents() is deprecated. Use enumerate() instead. "
            "Content loading and parsing has been moved to the pipeline executor."
        )
