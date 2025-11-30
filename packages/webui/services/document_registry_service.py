"""Service for registering documents from any connector."""

import logging
import mimetypes
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.repositories.document_repository import DocumentRepository
from shared.dtos.ingestion import IngestedDocument

logger = logging.getLogger(__name__)


class DocumentRegistryService:
    """Centralized service for document registration and deduplication.

    Consumes IngestedDocument DTOs from any connector (directory, web, Slack)
    and handles database registration with content-hash-based deduplication.

    This service is designed to be used by:
    - DocumentScanningService (local files)
    - Future web connectors
    - Future Slack connectors
    - Any other document source

    Example:
        ```python
        registry = DocumentRegistryService(db_session, document_repo)
        result = await registry.register(
            collection_id="uuid",
            ingested=IngestedDocument(
                content="Document text",
                unique_id="https://example.com/doc",
                source_type="web",
                metadata={"url": "https://example.com/doc"},
                content_hash="abc123...",
            ),
        )
        if result["is_new"]:
            # Process new document
            pass
        ```
    """

    def __init__(self, db_session: AsyncSession, document_repo: DocumentRepository):
        """Initialize the registry service.

        Args:
            db_session: Database session for transactions
            document_repo: Document repository for DB operations
        """
        self.db_session = db_session
        self.document_repo = document_repo

    async def register(
        self,
        collection_id: str,
        ingested: IngestedDocument,
        source_id: int | None = None,
    ) -> dict[str, Any]:
        """Register a document in the collection with deduplication.

        This method checks if a document with the same content hash already
        exists in the collection. If so, it returns the existing document's
        info. Otherwise, it creates a new document record.

        Args:
            collection_id: UUID of the collection
            ingested: IngestedDocument DTO from connector
            source_id: Optional source ID to associate

        Returns:
            Dictionary with:
                - is_new: bool - Whether this is a new document
                - document_id: str - ID of the created/existing document
                - file_size: int - Size in bytes

        Raises:
            ValueError: If ingested document has invalid data
            EntityNotFoundError: If collection doesn't exist
            DatabaseOperationError: For database errors
        """
        # Derive file_path and file_name
        file_path = ingested.file_path or ingested.unique_id
        file_name = self._derive_file_name(ingested)

        # Get file_size from metadata, or estimate from content
        file_size = self._get_file_size(ingested)

        # Get MIME type if available
        mime_type = self._get_mime_type(ingested)

        # Check for existing document with same content hash
        existing_doc = await self.document_repo.get_by_content_hash(collection_id, ingested.content_hash)
        if existing_doc is not None:
            logger.debug(
                f"Document with hash {ingested.content_hash[:16]}... already exists " f"in collection {collection_id}"
            )
            return {
                "is_new": False,
                "document_id": existing_doc.id,
                "file_size": existing_doc.file_size or file_size,
            }

        # Register new document via repository
        # Note: DocumentRepository.create handles race conditions (IntegrityError)
        document = await self.document_repo.create(
            collection_id=collection_id,
            file_path=file_path,
            file_name=file_name,
            file_size=file_size,
            content_hash=ingested.content_hash,
            mime_type=mime_type,
            source_id=source_id,
            meta=ingested.metadata,
            uri=ingested.unique_id,
            source_metadata=ingested.metadata,
        )

        await self.document_repo.session.refresh(document)

        logger.info(
            f"Registered new document {document.id} for collection {collection_id} "
            f"(source_type={ingested.source_type})"
        )

        return {
            "is_new": True,
            "document_id": document.id,
            "file_size": file_size,
        }

    def _derive_file_name(self, ingested: IngestedDocument) -> str:
        """Derive file name from IngestedDocument.

        Args:
            ingested: The IngestedDocument to extract file name from

        Returns:
            File name string
        """
        if ingested.file_path:
            return Path(ingested.file_path).name

        # For web/slack sources, extract from unique_id
        unique_id: str = ingested.unique_id
        if "/" in unique_id:
            # Take the last path component
            last_segment: str = unique_id.rsplit("/", 1)[-1]
            return last_segment if last_segment else unique_id
        return unique_id

    def _get_file_size(self, ingested: IngestedDocument) -> int:
        """Get file size from metadata or estimate from content.

        Checks common metadata keys first, then falls back to
        estimating size from content byte length.

        Args:
            ingested: The IngestedDocument to get size from

        Returns:
            File size in bytes
        """
        # Check common metadata keys
        for key in ("file_size", "size", "content_length"):
            if key in ingested.metadata:
                try:
                    return int(ingested.metadata[key])
                except (ValueError, TypeError):
                    pass

        # Estimate from content (UTF-8 byte length)
        return len(ingested.content.encode("utf-8"))

    def _get_mime_type(self, ingested: IngestedDocument) -> str | None:
        """Get MIME type from metadata or infer from file path.

        Args:
            ingested: The IngestedDocument to get MIME type from

        Returns:
            MIME type string or None if cannot be determined
        """
        # Check metadata first
        for key in ("mime_type", "content_type", "mimetype"):
            if key in ingested.metadata:
                return str(ingested.metadata[key])

        # Infer from file_path if available
        if ingested.file_path:
            mime_type, _ = mimetypes.guess_type(ingested.file_path)
            return mime_type

        return None
