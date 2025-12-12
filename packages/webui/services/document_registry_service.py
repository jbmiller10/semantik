"""Service for registering documents from any connector."""

import logging
import mimetypes
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.repositories.chunk_repository import ChunkRepository
from shared.database.repositories.document_artifact_repository import DocumentArtifactRepository
from shared.database.repositories.document_repository import DocumentRepository
from shared.dtos.ingestion import IngestedDocument

logger = logging.getLogger(__name__)

# Source types that require artifact storage (no local file exists)
NON_FILE_SOURCE_TYPES = {"git", "imap", "web", "slack", "email"}


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

    def __init__(
        self,
        db_session: AsyncSession,
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository | None = None,
        artifact_repo: DocumentArtifactRepository | None = None,
    ):
        """Initialize the registry service.

        Args:
            db_session: Database session for transactions
            document_repo: Document repository for DB operations
            chunk_repo: Optional chunk repository for sync operations
            artifact_repo: Optional artifact repository for non-file sources
        """
        self.db_session = db_session
        self.document_repo = document_repo
        self.chunk_repo = chunk_repo
        self.artifact_repo = artifact_repo

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
                f"Document with hash {ingested.content_hash[:16]}... already exists in collection {collection_id}"
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

    async def register_or_update(
        self,
        collection_id: str,
        ingested: IngestedDocument,
        source_id: int | None = None,
    ) -> dict[str, Any]:
        """Register or update a document using URI-based identity.

        This method implements sync-aware document management:
        1. Look up document by URI (stable identity across syncs)
        2. If found and content unchanged: update last_seen_at, skip processing
        3. If found and content changed: update content, delete old chunks
        4. If not found: create new document

        This enables:
        - Efficient incremental syncs (skip unchanged documents)
        - In-place updates (same document ID, re-chunked)
        - Stale detection (documents not seen can be marked stale)

        Args:
            collection_id: UUID of the collection
            ingested: IngestedDocument DTO from connector
            source_id: Optional source ID to associate

        Returns:
            Dictionary with:
                - is_new: bool - Whether this is a new document
                - is_updated: bool - Whether content was updated
                - document_id: str - ID of the document
                - file_size: int - Size in bytes

        Raises:
            ValueError: If ingested document has invalid data
            EntityNotFoundError: If collection doesn't exist
            DatabaseOperationError: For database errors
        """
        # Look up document by URI (stable identity)
        existing = await self.document_repo.get_by_uri(collection_id, ingested.unique_id)

        if existing is not None:
            # Document exists - update last_seen_at
            await self.document_repo.update_last_seen(existing.id)

            if existing.content_hash == ingested.content_hash:
                # Content unchanged - skip processing
                logger.debug(f"Document {existing.id} unchanged (hash {ingested.content_hash[:16]}...), skipping")
                return {
                    "is_new": False,
                    "is_updated": False,
                    "document_id": existing.id,
                    "file_size": existing.file_size or 0,
                }

            # Content changed - update in place
            logger.info(
                f"Document {existing.id} content changed "
                f"({existing.content_hash[:16]}... -> {ingested.content_hash[:16]}...), updating"
            )

            # Get new values
            file_size = self._get_file_size(ingested)
            file_path = ingested.file_path or ingested.unique_id
            mime_type = self._get_mime_type(ingested)

            # Update document content
            await self.document_repo.update_content(
                document_id=existing.id,
                content_hash=ingested.content_hash,
                file_size=file_size,
                file_path=file_path,
                mime_type=mime_type,
                source_metadata=ingested.metadata,
            )

            # Delete old chunks for re-chunking
            if self.chunk_repo is not None:
                deleted_count = await self.chunk_repo.delete_chunks_by_document(
                    document_id=existing.id,
                    collection_id=collection_id,
                )
                logger.debug(f"Deleted {deleted_count} old chunks for document {existing.id}")

            # Store/update artifact for non-file sources
            await self._store_artifact(existing.id, collection_id, ingested)

            return {
                "is_new": False,
                "is_updated": True,
                "document_id": existing.id,
                "file_size": file_size,
            }

        # New document - create via standard register flow
        # But also update last_seen_at for the new document
        result = await self.register(
            collection_id=collection_id,
            ingested=ingested,
            source_id=source_id,
        )

        # Update last_seen_at for the newly created document
        await self.document_repo.update_last_seen(result["document_id"])

        # Store artifact for non-file sources (new documents)
        if result["is_new"]:
            await self._store_artifact(result["document_id"], collection_id, ingested)

        return {
            "is_new": result["is_new"],
            "is_updated": False,
            "document_id": result["document_id"],
            "file_size": result["file_size"],
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

    def _should_store_artifact(self, ingested: IngestedDocument) -> bool:
        """Determine if document content should be stored as an artifact.

        Artifacts are stored for non-file sources where content cannot be
        served from the filesystem (Git, IMAP, web, etc.).

        Args:
            ingested: The IngestedDocument to check

        Returns:
            True if artifact should be stored, False otherwise
        """
        # Check if source type requires artifact storage
        if ingested.source_type in NON_FILE_SOURCE_TYPES:
            return True

        # Check if file_path is missing or is a URI (not a real file)
        if not ingested.file_path:
            return True

        # Check if file_path looks like a URI scheme (git://, imap://, etc.)
        if "://" in ingested.file_path:
            return True

        # Check if the file_path exists on disk
        # If it doesn't exist, we need to store the content as artifact
        try:
            if not Path(ingested.file_path).exists():
                return True
        except (OSError, ValueError):
            # Path is invalid or inaccessible
            return True

        return False

    async def _store_artifact(
        self,
        document_id: str,
        collection_id: str,
        ingested: IngestedDocument,
    ) -> bool:
        """Store document content as an artifact in the database.

        Args:
            document_id: UUID of the document
            collection_id: UUID of the collection
            ingested: The IngestedDocument with content

        Returns:
            True if artifact was stored, False if skipped or failed
        """
        if self.artifact_repo is None:
            logger.debug(f"Artifact repo not configured, skipping artifact storage for {document_id}")
            return False

        if not self._should_store_artifact(ingested):
            logger.debug(f"Document {document_id} has file source, skipping artifact storage")
            return False

        if not ingested.content:
            logger.warning(f"Document {document_id} has no content, skipping artifact storage")
            return False

        try:
            mime_type = self._get_mime_type(ingested) or "text/plain"

            await self.artifact_repo.create_or_replace(
                document_id=document_id,
                collection_id=collection_id,
                content=ingested.content,
                mime_type=mime_type,
                content_hash=ingested.content_hash,
            )

            logger.debug(
                f"Stored artifact for document {document_id} "
                f"(source_type={ingested.source_type}, size={len(ingested.content)})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to store artifact for document {document_id}: {e}")
            # Don't fail the registration - artifact is optional
            return False
