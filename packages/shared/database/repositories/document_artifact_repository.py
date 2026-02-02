"""Repository implementation for DocumentArtifact model."""

import inspect
import logging
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.db_retry import with_db_retry
from shared.database.exceptions import DatabaseOperationError, ValidationError
from shared.database.models import DocumentArtifact
from shared.utils.hashing import compute_content_hash

logger = logging.getLogger(__name__)

# Default max artifact size (50 MB)
DEFAULT_MAX_ARTIFACT_BYTES = 50 * 1024 * 1024


class DocumentArtifactRepository:
    """Repository for DocumentArtifact model operations.

    This repository manages document artifacts - stored content for non-file
    sources (Git, IMAP, web). Artifacts allow the content endpoint to serve
    documents that don't exist on the local filesystem.

    Example:
        ```python
        artifact_repo = DocumentArtifactRepository(session)

        # Store artifact
        artifact = await artifact_repo.create_or_replace(
            document_id="uuid",
            collection_id="uuid",
            content="Document text content",
            mime_type="text/plain",
            content_hash="abc123...",
        )

        # Retrieve for serving
        content_data = await artifact_repo.get_content(document_id)
        if content_data:
            content, mime_type, charset = content_data
        ```
    """

    def __init__(self, session: AsyncSession, max_artifact_bytes: int | None = None):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
            max_artifact_bytes: Maximum artifact size in bytes (default: 50 MB)
        """
        self.session = session
        self._max_bytes = max_artifact_bytes or DEFAULT_MAX_ARTIFACT_BYTES

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def create_or_replace(
        self,
        document_id: str,
        collection_id: str,
        content: str | bytes,
        mime_type: str,
        content_hash: str,
        artifact_kind: str = "primary",
        charset: str | None = None,
    ) -> DocumentArtifact:
        """Create or replace an artifact for a document.

        If an artifact of the same kind already exists for the document,
        it is replaced with the new content.

        Args:
            document_id: UUID of the parent document
            collection_id: UUID of the collection
            content: Text or binary content to store
            mime_type: MIME type of the content
            content_hash: SHA-256 hash of the content
            artifact_kind: Type of artifact ('primary', 'preview', 'thumbnail')
            charset: Character encoding for text content (default: utf-8 for strings)

        Returns:
            Created/updated DocumentArtifact instance

        Raises:
            ValidationError: If content exceeds max size or artifact_kind is invalid
            DatabaseOperationError: For database errors
        """
        try:
            # Calculate size
            size_bytes = len(content.encode("utf-8")) if isinstance(content, str) else len(content)

            # Check size limit
            is_truncated = False
            if size_bytes > self._max_bytes:
                logger.warning(
                    f"Artifact for document {document_id} exceeds size limit "
                    f"({size_bytes} > {self._max_bytes}), truncating"
                )
                is_truncated = True
                if isinstance(content, str):
                    # Truncate to max bytes (handling UTF-8 boundary)
                    content = content.encode("utf-8")[: self._max_bytes].decode("utf-8", errors="ignore")
                    size_bytes = len(content.encode("utf-8"))
                else:
                    content = content[: self._max_bytes]
                    size_bytes = self._max_bytes
                # Recompute hash for truncated content
                content_hash = compute_content_hash(content)

            # Validate artifact_kind
            valid_kinds = ("primary", "preview", "thumbnail")
            if artifact_kind not in valid_kinds:
                raise ValidationError(
                    f"artifact_kind must be one of {valid_kinds}",
                    "artifact_kind",
                )

            # Delete existing artifact with same kind
            await self.session.execute(
                delete(DocumentArtifact).where(
                    DocumentArtifact.document_id == document_id,
                    DocumentArtifact.artifact_kind == artifact_kind,
                )
            )

            # Create new artifact
            if isinstance(content, str):
                artifact = DocumentArtifact(
                    document_id=document_id,
                    collection_id=collection_id,
                    artifact_kind=artifact_kind,
                    mime_type=mime_type,
                    charset=charset or "utf-8",
                    content_text=content,
                    content_bytes=None,
                    content_hash=content_hash,
                    size_bytes=size_bytes,
                    is_truncated=is_truncated,
                )
            else:
                artifact = DocumentArtifact(
                    document_id=document_id,
                    collection_id=collection_id,
                    artifact_kind=artifact_kind,
                    mime_type=mime_type,
                    charset=charset,
                    content_text=None,
                    content_bytes=content,
                    content_hash=content_hash,
                    size_bytes=size_bytes,
                    is_truncated=is_truncated,
                )

            self.session.add(artifact)
            await self.session.flush()

            logger.debug(
                f"Created artifact for document {document_id} "
                f"(kind={artifact_kind}, size={size_bytes}, truncated={is_truncated})"
            )

            return artifact

        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                "Failed to create/replace artifact for document %s: %s",
                document_id,
                e,
                exc_info=True,
            )
            raise DatabaseOperationError("create", "DocumentArtifact", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_primary(self, document_id: str) -> DocumentArtifact | None:
        """Get the primary artifact for a document.

        Args:
            document_id: UUID of the document

        Returns:
            DocumentArtifact instance or None if not found
        """
        try:
            result = await self.session.execute(
                select(DocumentArtifact).where(
                    DocumentArtifact.document_id == document_id,
                    DocumentArtifact.artifact_kind == "primary",
                )
            )
            artifact = result.scalar_one_or_none()
            if inspect.isawaitable(artifact):
                artifact = await artifact
            return artifact
        except Exception as e:
            logger.error("Failed to get primary artifact for document %s: %s", document_id, e, exc_info=True)
            raise DatabaseOperationError("get", "DocumentArtifact", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_by_kind(self, document_id: str, artifact_kind: str) -> DocumentArtifact | None:
        """Get an artifact by document ID and kind.

        Args:
            document_id: UUID of the document
            artifact_kind: Type of artifact to retrieve

        Returns:
            DocumentArtifact instance or None if not found
        """
        try:
            result = await self.session.execute(
                select(DocumentArtifact).where(
                    DocumentArtifact.document_id == document_id,
                    DocumentArtifact.artifact_kind == artifact_kind,
                )
            )
            artifact = result.scalar_one_or_none()
            if inspect.isawaitable(artifact):
                artifact = await artifact
            return artifact
        except Exception as e:
            logger.error("Failed to get artifact for document %s: %s", document_id, e, exc_info=True)
            raise DatabaseOperationError("get", "DocumentArtifact", str(e)) from e

    async def get_content(
        self,
        document_id: str,
        artifact_kind: str = "primary",
    ) -> tuple[str | bytes, str, str | None] | None:
        """Get artifact content with metadata for serving.

        This is the primary method used by the content endpoint to retrieve
        document content from the database.

        Args:
            document_id: UUID of the document
            artifact_kind: Type of artifact (default: 'primary')

        Returns:
            Tuple of (content, mime_type, charset) or None if not found.
            - content: Text string or binary bytes
            - mime_type: MIME type of the content
            - charset: Character encoding (None for binary)
        """
        artifact = await self.get_by_kind(document_id, artifact_kind)

        if artifact is None:
            return None

        content = artifact.content_text if artifact.content_text is not None else artifact.content_bytes

        return (content, artifact.mime_type, artifact.charset)

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def has_artifact(self, document_id: str, artifact_kind: str = "primary") -> bool:
        """Check if an artifact exists without loading content.

        Args:
            document_id: UUID of the document
            artifact_kind: Type of artifact to check

        Returns:
            True if artifact exists, False otherwise
        """
        try:
            result = await self.session.execute(
                select(DocumentArtifact.id).where(
                    DocumentArtifact.document_id == document_id,
                    DocumentArtifact.artifact_kind == artifact_kind,
                )
            )
            value = result.scalar_one_or_none()
            if inspect.isawaitable(value):
                value = await value
            return value is not None
        except Exception as e:
            logger.error(
                "Failed to check artifact existence for document %s: %s",
                document_id,
                e,
                exc_info=True,
            )
            raise DatabaseOperationError("check", "DocumentArtifact", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def delete_for_document(self, document_id: str) -> int:
        """Delete all artifacts for a document.

        Args:
            document_id: UUID of the document

        Returns:
            Number of artifacts deleted
        """
        try:
            result = await self.session.execute(
                delete(DocumentArtifact).where(DocumentArtifact.document_id == document_id)
            )
            count = result.rowcount or 0
            logger.debug(f"Deleted {count} artifacts for document {document_id}")
            return count
        except Exception as e:
            logger.error("Failed to delete artifacts for document %s: %s", document_id, e, exc_info=True)
            raise DatabaseOperationError("delete", "DocumentArtifact", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def delete_for_collection(self, collection_id: str) -> int:
        """Delete all artifacts for a collection.

        Args:
            collection_id: UUID of the collection

        Returns:
            Number of artifacts deleted
        """
        try:
            result = await self.session.execute(
                delete(DocumentArtifact).where(DocumentArtifact.collection_id == collection_id)
            )
            count = result.rowcount or 0
            logger.debug(f"Deleted {count} artifacts for collection {collection_id}")
            return count
        except Exception as e:
            logger.error("Failed to delete artifacts for collection %s: %s", collection_id, e, exc_info=True)
            raise DatabaseOperationError("delete", "DocumentArtifact", str(e)) from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_stats_for_collection(self, collection_id: str) -> dict[str, Any]:
        """Get artifact statistics for a collection.

        Args:
            collection_id: UUID of the collection

        Returns:
            Dictionary with artifact statistics
        """
        try:
            from sqlalchemy import func as sql_func

            result = await self.session.execute(
                select(
                    sql_func.count(DocumentArtifact.id).label("count"),
                    sql_func.sum(DocumentArtifact.size_bytes).label("total_bytes"),
                    sql_func.count(DocumentArtifact.id).filter(DocumentArtifact.is_truncated).label("truncated_count"),
                ).where(DocumentArtifact.collection_id == collection_id)
            )
            row = result.one()

            return {
                "artifact_count": row.count or 0,
                "total_bytes": row.total_bytes or 0,
                "truncated_count": row.truncated_count or 0,
            }
        except Exception as e:
            logger.error("Failed to get artifact stats for collection %s: %s", collection_id, e, exc_info=True)
            raise DatabaseOperationError("get_stats", "DocumentArtifact", str(e)) from e
