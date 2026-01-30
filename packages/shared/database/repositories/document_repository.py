"""Repository implementation for Document model."""

import logging
import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, case, delete, desc, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from shared.database.db_retry import with_db_retry
from shared.database.exceptions import DatabaseOperationError, EntityNotFoundError, ValidationError
from shared.database.models import Collection, Document, DocumentStatus

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for Document model operations."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session

    @with_db_retry(retries=5, delay=0.5, backoff=2.0, max_delay=10.0)
    async def create(
        self,
        collection_id: str,
        file_path: str,
        file_name: str,
        file_size: int,
        content_hash: str,
        mime_type: str | None = None,
        source_id: int | None = None,
        meta: dict[str, Any] | None = None,
        uri: str | None = None,
        source_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Create a new document with deduplication.

        If a document with the same content_hash already exists in the collection,
        returns the existing document instead of creating a new one.

        Args:
            collection_id: UUID of the collection
            file_path: Path to the document file
            file_name: Name of the document
            file_size: Size of the file in bytes
            content_hash: SHA256 hash of the file content
            mime_type: Optional MIME type
            source_id: Optional source ID
            meta: Optional metadata
            uri: Optional logical identifier (URL, file path, message ID, etc.)
            source_metadata: Optional connector-specific metadata

        Returns:
            Created or existing Document instance

        Raises:
            EntityNotFoundError: If collection not found
            ValidationError: If required fields are invalid
            DatabaseOperationError: For database errors
        """
        try:
            # Validate required fields
            if not file_path or not file_name:
                raise ValidationError("File path and name are required", "file_path/file_name")
            if file_size < 0:
                raise ValidationError("File size cannot be negative", "file_size")
            if not content_hash:
                raise ValidationError("Content hash is required", "content_hash")
            # Validate SHA-256 hash format (64 hex characters)
            if not re.match(r"^[a-f0-9]{64}$", content_hash.lower()):
                raise ValidationError("Invalid SHA-256 hash format", "content_hash")

            # Check if collection exists
            collection_result = await self.session.execute(select(Collection).where(Collection.id == collection_id))
            collection = collection_result.scalar_one_or_none()
            if not collection:
                raise EntityNotFoundError("collection", collection_id)

            # Check for existing document with same content hash in this collection
            existing_doc = await self.get_by_content_hash(collection_id, content_hash)
            if existing_doc:
                logger.info(
                    f"Document with content_hash {content_hash} already exists "
                    f"in collection {collection_id}, returning existing document {existing_doc.id}"
                )
                return existing_doc

            # Create new document
            document_id = str(uuid4())
            document = Document(
                id=document_id,
                collection_id=collection_id,
                source_id=source_id,
                file_path=file_path,
                file_name=file_name,
                file_size=file_size,
                mime_type=mime_type,
                content_hash=content_hash,
                status=DocumentStatus.PENDING.value,  # Use .value for PostgreSQL compatibility
                meta=meta or {},
                uri=uri,
                source_metadata=source_metadata,
            )

            self.session.add(document)
            await self.session.flush()

            logger.info(
                f"Created document {document.id} for collection {collection_id} with content_hash {content_hash}"
            )
            return document

        except IntegrityError as e:
            # Handle race condition where another process created the same document
            logger.warning("Integrity error creating document, checking for existing: %s", e, exc_info=True)
            existing_doc = await self.get_by_content_hash(collection_id, content_hash)
            if existing_doc:
                return existing_doc
            raise DatabaseOperationError("create", "document", str(e)) from e
        except (EntityNotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error("Failed to create document: %s", e, exc_info=True)
            raise DatabaseOperationError("create", "document", str(e)) from e

    async def get_by_id(self, document_id: str) -> Document | None:
        """Get a document by ID.

        Args:
            document_id: UUID of the document

        Returns:
            Document instance or None if not found
        """
        try:
            result = await self.session.execute(
                select(Document)
                .where(Document.id == document_id)
                .options(selectinload(Document.collection), selectinload(Document.source))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Failed to get document %s: %s", document_id, e, exc_info=True)
            raise DatabaseOperationError("get", "document", str(e)) from e

    async def get_by_content_hash(self, collection_id: str, content_hash: str) -> Document | None:
        """Get a document by content hash within a collection.

        Args:
            collection_id: UUID of the collection
            content_hash: SHA256 hash of the content

        Returns:
            Document instance or None if not found
        """
        try:
            result = await self.session.execute(
                select(Document).where(
                    and_(
                        Document.collection_id == collection_id,
                        Document.content_hash == content_hash,
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(
                "Failed to get document by content_hash %s in collection %s: %s",
                content_hash,
                collection_id,
                e,
                exc_info=True,
            )
            raise DatabaseOperationError("get", "document", str(e)) from e

    async def get_by_uri(self, collection_id: str, uri: str) -> Document | None:
        """Get a document by URI within a collection.

        Args:
            collection_id: UUID of the collection
            uri: Logical identifier (URL, file path, message ID, etc.)

        Returns:
            Document instance or None if not found
        """
        try:
            result = await self.session.execute(
                select(Document).where(
                    and_(
                        Document.collection_id == collection_id,
                        Document.uri == uri,
                    )
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(
                "Failed to get document by uri %s in collection %s: %s",
                uri,
                collection_id,
                e,
                exc_info=True,
            )
            raise DatabaseOperationError("get", "document", str(e)) from e

    async def list_by_collection(
        self,
        collection_id: str,
        status: DocumentStatus | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[Document], int]:
        """List documents in a collection.

        Args:
            collection_id: UUID of the collection
            status: Optional status filter
            offset: Pagination offset
            limit: Maximum number of results

        Returns:
            Tuple of (documents list, total count)
        """
        try:
            # Build base query
            query = select(Document).where(Document.collection_id == collection_id)

            # Apply status filter if provided
            if status:
                query = query.where(Document.status == status)

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total = await self.session.scalar(count_query)

            # Get paginated results - sort failed documents first, then by created_at desc
            query = (
                query.order_by(
                    case(
                        (Document.status == DocumentStatus.FAILED, 0),
                        else_=1,
                    ),
                    desc(Document.created_at),
                )
                .offset(offset)
                .limit(limit)
            )
            result = await self.session.execute(query)
            documents = result.scalars().all()

            return list(documents), total or 0

        except Exception as e:
            logger.error("Failed to list documents for collection %s: %s", collection_id, e, exc_info=True)
            raise DatabaseOperationError("list", "documents", str(e)) from e

    async def list_by_source_id(
        self,
        collection_id: str,
        source_id: int,
        status: DocumentStatus | None = None,
    ) -> list[Document]:
        """List documents by collection and source ID.

        Args:
            collection_id: UUID of the collection
            source_id: Integer ID of the collection source
            status: Optional status filter

        Returns:
            List of Document instances
        """
        try:
            query = select(Document).where(
                and_(
                    Document.collection_id == collection_id,
                    Document.source_id == source_id,
                )
            )

            if status:
                query = query.where(Document.status == status)

            result = await self.session.execute(query)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("Failed to list documents by source_id %s: %s", source_id, e, exc_info=True)
            raise DatabaseOperationError("list", "documents", str(e)) from e

    async def list_duplicates(self, collection_id: str) -> list[tuple[str, int, list[Document]]]:
        """List duplicate documents in a collection.

        Returns documents grouped by content_hash where there are multiple
        documents with the same hash.

        Args:
            collection_id: UUID of the collection

        Returns:
            List of tuples containing (content_hash, count, documents)
        """
        try:
            # Subquery to find content hashes with duplicates
            duplicate_hashes = (
                select(Document.content_hash, func.count(Document.id).label("count"))
                .where(Document.collection_id == collection_id)
                .group_by(Document.content_hash)
                .having(func.count(Document.id) > 1)
                .subquery()
            )

            # Get all documents with duplicate hashes
            query = (
                select(Document)
                .where(
                    and_(
                        Document.collection_id == collection_id,
                        Document.content_hash.in_(select(duplicate_hashes.c.content_hash)),
                    )
                )
                .order_by(Document.content_hash, Document.created_at)
            )

            result = await self.session.execute(query)
            documents = result.scalars().all()

            # Group by content_hash
            duplicates: dict[str, list[Document]] = {}
            for doc in documents:
                if doc.content_hash not in duplicates:
                    duplicates[doc.content_hash] = []
                duplicates[doc.content_hash].append(doc)

            # Return as list of tuples
            return [(content_hash, len(docs), docs) for content_hash, docs in duplicates.items()]

        except Exception as e:
            logger.error("Failed to list duplicate documents: %s", e, exc_info=True)
            raise DatabaseOperationError("list", "duplicate documents", str(e)) from e

    async def count_by_collection(self, collection_id: str) -> int:
        """Return the total number of documents in a collection."""
        try:
            result = await self.session.scalar(
                select(func.count()).select_from(Document).where(Document.collection_id == collection_id)
            )
            return int(result or 0)
        except Exception as exc:
            logger.error("Failed to count documents for collection %s: %s", collection_id, exc, exc_info=True)
            raise DatabaseOperationError("count", "documents", str(exc)) from exc

    async def get_existing_ids_in_collection(self, collection_id: str, document_ids: set[str]) -> set[str]:
        """Return the subset of document IDs that exist in a collection."""
        if not document_ids:
            return set()

        try:
            rows = await self.session.execute(
                select(Document.id).where(
                    and_(
                        Document.collection_id == collection_id,
                        Document.id.in_(document_ids),
                    )
                )
            )
            return {str(row[0]) for row in rows.all()}
        except Exception as exc:
            logger.error(
                "Failed to bulk-check document ids for collection %s: %s",
                collection_id,
                exc,
                exc_info=True,
            )
            raise DatabaseOperationError("get", "document", str(exc)) from exc

    async def get_doc_ids_by_uri_bulk(self, collection_id: str, uris: set[str]) -> dict[str, str]:
        """Return a mapping from uri -> document_id for URIs present in the collection."""
        if not uris:
            return {}

        try:
            rows = await self.session.execute(
                select(Document.uri, Document.id).where(
                    and_(
                        Document.collection_id == collection_id,
                        Document.uri.in_(uris),
                    )
                )
            )
            result: dict[str, str] = {}
            for uri, doc_id in rows.all():
                if uri is None:
                    continue
                result[str(uri)] = str(doc_id)
            return result
        except Exception as exc:
            logger.error(
                "Failed to bulk-get documents by uri for collection %s: %s",
                collection_id,
                exc,
                exc_info=True,
            )
            raise DatabaseOperationError("get", "document", str(exc)) from exc

    async def get_doc_ids_by_content_hash_bulk(self, collection_id: str, hashes: set[str]) -> dict[str, list[str]]:
        """Return a mapping from content_hash -> [document_id, ...] for hashes present in the collection."""
        if not hashes:
            return {}

        try:
            rows = await self.session.execute(
                select(Document.content_hash, Document.id).where(
                    and_(
                        Document.collection_id == collection_id,
                        Document.content_hash.in_(hashes),
                    )
                )
            )
            result: dict[str, list[str]] = {}
            for content_hash, doc_id in rows.all():
                result.setdefault(str(content_hash), []).append(str(doc_id))
            return result
        except Exception as exc:
            logger.error(
                "Failed to bulk-get documents by content_hash for collection %s: %s",
                collection_id,
                exc,
                exc_info=True,
            )
            raise DatabaseOperationError("get", "document", str(exc)) from exc

    async def get_doc_ids_by_file_name_bulk(self, collection_id: str, file_names: set[str]) -> dict[str, list[str]]:
        """Return a mapping from file_name -> [document_id, ...] for file names present in the collection."""
        if not file_names:
            return {}

        try:
            rows = await self.session.execute(
                select(Document.file_name, Document.id).where(
                    and_(
                        Document.collection_id == collection_id,
                        Document.file_name.in_(file_names),
                    )
                )
            )
            result: dict[str, list[str]] = {}
            for file_name, doc_id in rows.all():
                result.setdefault(str(file_name), []).append(str(doc_id))
            return result
        except Exception as exc:
            logger.error(
                "Failed to bulk-get documents by file_name for collection %s: %s",
                collection_id,
                exc,
                exc_info=True,
            )
            raise DatabaseOperationError("get", "document", str(exc)) from exc

    async def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: str | None = None,
        chunk_count: int | None = None,
        error_category: str | None = None,
    ) -> Document:
        """Update document status.

        Args:
            document_id: UUID of the document
            status: New status
            error_message: Optional error message
            chunk_count: Optional chunk count
            error_category: Optional error category ('transient', 'permanent', 'unknown')

        Returns:
            Updated Document instance

        Raises:
            EntityNotFoundError: If document not found
        """
        try:
            document = await self.get_by_id(document_id)
            if not document:
                raise EntityNotFoundError("document", document_id)

            document.status = status.value  # Use .value for PostgreSQL compatibility
            document.error_message = error_message
            if chunk_count is not None:
                document.chunk_count = chunk_count
            if error_category is not None:
                document.error_category = error_category
            document.updated_at = datetime.now(UTC)

            await self.session.flush()

            logger.info(f"Updated document {document_id} status to {status}")
            return document

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update document status: %s", e, exc_info=True)
            raise DatabaseOperationError("update", "document", str(e)) from e

    async def bulk_update_status(
        self,
        document_ids: list[str],
        status: DocumentStatus,
        error_message: str | None = None,
    ) -> int:
        """Update status for multiple documents.

        Args:
            document_ids: List of document UUIDs
            status: New status
            error_message: Optional error message

        Returns:
            Number of documents updated
        """
        try:
            if not document_ids:
                return 0

            stmt = (
                update(Document)
                .where(Document.id.in_(document_ids))
                .values(
                    status=status.value,  # Use .value for PostgreSQL compatibility
                    error_message=error_message,
                    updated_at=datetime.now(UTC),
                )
            )

            result = await self.session.execute(stmt)
            count = result.rowcount or 0

            logger.info(f"Updated {count} documents to status {status}")
            return count

        except Exception as e:
            logger.error("Failed to bulk update document status: %s", e, exc_info=True)
            raise DatabaseOperationError("bulk update", "documents", str(e)) from e

    async def delete(self, document_id: str) -> None:
        """Delete a document.

        Args:
            document_id: UUID of the document

        Raises:
            EntityNotFoundError: If document not found
        """
        try:
            document = await self.get_by_id(document_id)
            if not document:
                raise EntityNotFoundError("document", document_id)

            # Use async pattern for deletion
            await self.session.execute(delete(Document).where(Document.id == document.id))
            await self.session.flush()

            logger.info(f"Deleted document {document_id}")

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to delete document: %s", e, exc_info=True)
            raise DatabaseOperationError("delete", "document", str(e)) from e

    async def delete_duplicates(self, collection_id: str, keep_oldest: bool = True) -> int:
        """Delete duplicate documents in a collection.

        For each set of documents with the same content_hash, keeps one and
        deletes the rest.

        Args:
            collection_id: UUID of the collection
            keep_oldest: If True, keeps the oldest document; if False, keeps the newest

        Returns:
            Number of documents deleted
        """
        try:
            duplicates = await self.list_duplicates(collection_id)
            deleted_count = 0

            for content_hash, _, docs in duplicates:
                # Sort by created_at
                sorted_docs = sorted(docs, key=lambda d: d.created_at, reverse=not keep_oldest)

                # Keep the first one, delete the rest using bulk operation
                if len(sorted_docs) > 1:
                    delete_ids = [doc.id for doc in sorted_docs[1:]]
                    result = await self.session.execute(delete(Document).where(Document.id.in_(delete_ids)))
                    deleted_count += result.rowcount

                    logger.info(f"Deleted {len(delete_ids)} duplicate documents with content_hash {content_hash}")

            await self.session.flush()

            logger.info(f"Deleted {deleted_count} duplicate documents from collection {collection_id}")
            return deleted_count

        except Exception as e:
            logger.error("Failed to delete duplicate documents: %s", e, exc_info=True)
            raise DatabaseOperationError("delete", "duplicate documents", str(e)) from e

    async def get_stats_by_collection(self, collection_id: str) -> dict[str, Any]:
        """Get document statistics for a collection.

        Args:
            collection_id: UUID of the collection

        Returns:
            Dictionary with statistics
        """
        try:
            # Get counts by status
            status_counts = await self.session.execute(
                select(Document.status, func.count(Document.id))
                .where(Document.collection_id == collection_id)
                .group_by(Document.status)
            )

            # Get total size
            total_size = await self.session.scalar(
                select(func.sum(Document.file_size)).where(Document.collection_id == collection_id)
            )

            # Get total chunk count
            total_chunks = await self.session.scalar(
                select(func.sum(Document.chunk_count)).where(Document.collection_id == collection_id)
            )

            # Get duplicate count
            duplicate_query = (
                select(func.count(Document.id))
                .where(Document.collection_id == collection_id)
                .group_by(Document.content_hash)
                .having(func.count(Document.id) > 1)
            )
            duplicate_groups = await self.session.scalar(select(func.count()).select_from(duplicate_query.subquery()))

            stats: dict[str, Any] = {
                "total_documents": 0,
                "by_status": {},
                "total_size_bytes": total_size or 0,
                "total_chunks": total_chunks or 0,
                "duplicate_groups": duplicate_groups or 0,
            }

            for status, count in status_counts:
                stats["by_status"][status.value] = count
                stats["total_documents"] += count

            return stats

        except Exception as e:
            logger.error("Failed to get document stats: %s", e, exc_info=True)
            raise DatabaseOperationError("get stats", "documents", str(e)) from e

    # -------------------------------------------------------------------------
    # Sync tracking methods (for continuous sync support)
    # -------------------------------------------------------------------------

    async def update_last_seen(self, document_id: str) -> Document:
        """Update the last_seen_at timestamp for a document.

        Called when a document is seen during a sync run.

        Args:
            document_id: UUID of the document

        Returns:
            Updated Document instance

        Raises:
            EntityNotFoundError: If document not found
        """
        try:
            document = await self.get_by_id(document_id)
            if not document:
                raise EntityNotFoundError("document", document_id)

            document.last_seen_at = datetime.now(UTC)
            document.is_stale = False  # Reset stale flag when seen
            document.updated_at = datetime.now(UTC)

            await self.session.flush()

            logger.debug(f"Updated last_seen_at for document {document_id}")
            return document

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update last_seen_at: %s", e, exc_info=True)
            raise DatabaseOperationError("update_last_seen", "document", str(e)) from e

    async def update_content(
        self,
        document_id: str,
        content_hash: str,
        file_size: int | None = None,
        file_path: str | None = None,
        mime_type: str | None = None,
        source_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Update a document's content fields (for in-place update during sync).

        Called when a document's content has changed and needs to be re-indexed.

        Args:
            document_id: UUID of the document
            content_hash: New content hash
            file_size: New file size (optional)
            file_path: New file path (optional)
            mime_type: New MIME type (optional)
            source_metadata: New source metadata (optional)

        Returns:
            Updated Document instance

        Raises:
            EntityNotFoundError: If document not found
        """
        try:
            document = await self.get_by_id(document_id)
            if not document:
                raise EntityNotFoundError("document", document_id)

            document.content_hash = content_hash
            if file_size is not None:
                document.file_size = file_size
            if file_path is not None:
                document.file_path = file_path
            if mime_type is not None:
                document.mime_type = mime_type
            if source_metadata is not None:
                document.source_metadata = source_metadata

            # Reset document status to PENDING for re-processing
            document.status = DocumentStatus.PENDING.value
            document.error_message = None
            document.chunk_count = 0
            document.chunks_count = 0
            document.chunking_started_at = None
            document.chunking_completed_at = None

            # Update sync tracking
            document.last_seen_at = datetime.now(UTC)
            document.is_stale = False
            document.updated_at = datetime.now(UTC)

            await self.session.flush()

            logger.info(f"Updated content for document {document_id}, new hash: {content_hash}")
            return document

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update document content: %s", e, exc_info=True)
            raise DatabaseOperationError("update_content", "document", str(e)) from e

    async def mark_unseen_as_stale(
        self,
        collection_id: str,
        source_id: int,
        since: datetime,
    ) -> int:
        """Mark documents not seen since a timestamp as stale.

        Called after a sync run to mark documents that were not encountered
        as potentially deleted from the source.

        Args:
            collection_id: UUID of the collection
            source_id: ID of the source
            since: Timestamp threshold (documents with last_seen_at before this are stale)

        Returns:
            Number of documents marked as stale
        """
        try:
            stmt = (
                update(Document)
                .where(
                    and_(
                        Document.collection_id == collection_id,
                        Document.source_id == source_id,
                        Document.is_stale == False,  # Only update non-stale docs  # noqa: E712
                        (Document.last_seen_at < since) | (Document.last_seen_at.is_(None)),
                    )
                )
                .values(
                    is_stale=True,
                    updated_at=datetime.now(UTC),
                )
            )

            result = await self.session.execute(stmt)
            count = result.rowcount or 0

            if count > 0:
                logger.info(f"Marked {count} documents as stale for source {source_id} in collection {collection_id}")

            return count

        except Exception as e:
            logger.error("Failed to mark documents as stale: %s", e, exc_info=True)
            raise DatabaseOperationError("mark_unseen_as_stale", "documents", str(e)) from e

    async def get_stale_documents(
        self,
        collection_id: str,
        source_id: int | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[Document], int]:
        """Get stale documents for a collection.

        Args:
            collection_id: UUID of the collection
            source_id: Optional source ID filter
            offset: Pagination offset
            limit: Maximum results

        Returns:
            Tuple of (documents list, total count)
        """
        try:
            # Build base query
            query = select(Document).where(
                and_(
                    Document.collection_id == collection_id,
                    Document.is_stale == True,  # noqa: E712
                )
            )

            if source_id is not None:
                query = query.where(Document.source_id == source_id)

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total = await self.session.scalar(count_query)

            # Get paginated results
            query = query.order_by(desc(Document.updated_at)).offset(offset).limit(limit)
            result = await self.session.execute(query)
            documents = result.scalars().all()

            return list(documents), total or 0

        except Exception as e:
            logger.error("Failed to get stale documents: %s", e, exc_info=True)
            raise DatabaseOperationError("get_stale_documents", "documents", str(e)) from e

    async def clear_stale_flag(
        self,
        document_id: str,
    ) -> Document:
        """Clear the stale flag for a document.

        Args:
            document_id: UUID of the document

        Returns:
            Updated Document instance

        Raises:
            EntityNotFoundError: If document not found
        """
        try:
            document = await self.get_by_id(document_id)
            if not document:
                raise EntityNotFoundError("document", document_id)

            document.is_stale = False
            document.updated_at = datetime.now(UTC)

            await self.session.flush()

            logger.debug(f"Cleared stale flag for document {document_id}")
            return document

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to clear stale flag: %s", e, exc_info=True)
            raise DatabaseOperationError("clear_stale_flag", "document", str(e)) from e

    # ========== Retry Methods ==========

    async def reset_for_retry(self, document_id: str) -> Document:
        """Reset a failed document for manual retry.

        Resets the document status to PENDING and increments the retry count.
        Clears error message and category so the document can be reprocessed.

        Args:
            document_id: UUID of the document

        Returns:
            Updated Document instance

        Raises:
            EntityNotFoundError: If document not found
            ValidationError: If document is not in FAILED status
        """
        try:
            document = await self.get_by_id(document_id)
            if not document:
                raise EntityNotFoundError("document", document_id)

            status_value = document.status
            if isinstance(status_value, DocumentStatus):
                normalized_status = status_value.value
            elif isinstance(status_value, str):
                normalized_status = status_value.lower()
            else:
                normalized_status = str(status_value).lower()

            if normalized_status != DocumentStatus.FAILED.value:
                raise ValidationError(f"Document {document_id} is not in FAILED status (current: {document.status})")

            document.status = DocumentStatus.PENDING.value
            document.retry_count = (document.retry_count or 0) + 1
            document.last_retry_at = datetime.now(UTC)
            document.error_message = None
            document.error_category = None
            document.chunk_count = 0
            document.updated_at = datetime.now(UTC)

            await self.session.flush()

            logger.info(f"Reset document {document_id} for retry (attempt {document.retry_count})")
            return document

        except (EntityNotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error("Failed to reset document for retry: %s", e, exc_info=True)
            raise DatabaseOperationError("reset_for_retry", "document", str(e)) from e

    async def bulk_reset_failed_for_retry(
        self,
        collection_id: str,
        max_retry_count: int = 3,
        error_categories: list[str] | None = None,
        retry_at: datetime | None = None,
    ) -> int:
        """Reset all retryable failed documents in a collection for retry.

        Only resets documents that:
        - Are in FAILED status
        - Have retry_count < max_retry_count
        - Have error_category in the specified list (or transient/unknown if not specified)

        Args:
            collection_id: UUID of the collection
            max_retry_count: Maximum retry attempts (documents with more won't be reset)
            error_categories: List of error categories to include (default: transient, unknown, None)
            retry_at: Timestamp to record for last_retry_at (defaults to now)

        Returns:
            Number of documents reset
        """
        try:
            # Default to retryable categories
            if error_categories is None:
                error_categories = ["transient", "unknown"]

            retry_time = retry_at or datetime.now(UTC)

            # Build conditions for retryable documents
            conditions = [
                Document.collection_id == collection_id,
                Document.status == DocumentStatus.FAILED.value,
                Document.retry_count < max_retry_count,
            ]

            # Include documents with specified categories or NULL category
            category_conditions = [Document.error_category.in_(error_categories)]
            if None not in error_categories:
                # Also include NULL categories as they're likely retryable
                category_conditions.append(Document.error_category.is_(None))

            from sqlalchemy import or_

            conditions.append(or_(*category_conditions))

            stmt = (
                update(Document)
                .where(and_(*conditions))
                .values(
                    status=DocumentStatus.PENDING.value,
                    retry_count=Document.retry_count + 1,
                    last_retry_at=retry_time,
                    error_message=None,
                    error_category=None,
                    chunk_count=0,
                    updated_at=datetime.now(UTC),
                )
            )

            result = await self.session.execute(stmt)
            count = result.rowcount or 0

            logger.info(f"Reset {count} failed documents for retry in collection {collection_id}")
            return count

        except Exception as e:
            logger.error("Failed to bulk reset documents for retry: %s", e, exc_info=True)
            raise DatabaseOperationError("bulk_reset_for_retry", "documents", str(e)) from e

    async def bulk_mark_retry_dispatch_failed(
        self,
        collection_id: str,
        retry_at: datetime,
        error_message: str,
        error_category: str | None = "transient",
    ) -> int:
        """Revert retry reset when task dispatch fails.

        Args:
            collection_id: UUID of the collection
            retry_at: Timestamp used for the retry reset
            error_message: Message to store on reverted documents
            error_category: Category to store (default: transient)

        Returns:
            Number of documents reverted to FAILED
        """
        try:
            stmt = (
                update(Document)
                .where(
                    and_(
                        Document.collection_id == collection_id,
                        Document.status == DocumentStatus.PENDING.value,
                        Document.last_retry_at == retry_at,
                    )
                )
                .values(
                    status=DocumentStatus.FAILED.value,
                    retry_count=case(
                        (Document.retry_count > 0, Document.retry_count - 1),
                        else_=0,
                    ),
                    error_message=error_message,
                    error_category=error_category,
                    updated_at=datetime.now(UTC),
                )
            )

            result = await self.session.execute(stmt)
            count = result.rowcount or 0
            if count:
                logger.warning(
                    "Reverted %d documents to FAILED after retry dispatch failure in collection %s",
                    count,
                    collection_id,
                )
            return count

        except Exception as e:
            logger.error("Failed to revert documents after retry dispatch failure: %s", e, exc_info=True)
            raise DatabaseOperationError("bulk_revert_retry_dispatch", "documents", str(e)) from e

    async def count_stuck_pending_documents(
        self,
        collection_id: str,
        stuck_threshold_minutes: int = 5,
    ) -> int:
        """Count documents stuck in PENDING status for longer than threshold.

        These are documents that were registered but never processed, likely due to
        an interrupted operation or worker crash.

        Args:
            collection_id: UUID of the collection
            stuck_threshold_minutes: How long a document must be in PENDING to be
                considered "stuck" (default: 5 minutes)

        Returns:
            Number of stuck pending documents
        """
        from datetime import timedelta

        try:
            threshold_time = datetime.now(UTC) - timedelta(minutes=stuck_threshold_minutes)

            stmt = select(func.count(Document.id)).where(
                and_(
                    Document.collection_id == collection_id,
                    Document.status == DocumentStatus.PENDING.value,
                    Document.created_at < threshold_time,
                )
            )

            result = await self.session.execute(stmt)
            count = result.scalar() or 0

            if count > 0:
                logger.info(
                    "Found %d stuck pending documents in collection %s (threshold: %d min)",
                    count,
                    collection_id,
                    stuck_threshold_minutes,
                )

            return count

        except Exception as e:
            logger.error("Failed to count stuck pending documents: %s", e, exc_info=True)
            raise DatabaseOperationError("count_stuck_pending", "documents", str(e)) from e

    async def list_failed_documents(
        self,
        collection_id: str,
        error_category: str | None = None,
        retryable_only: bool = False,
        max_retry_count: int = 3,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[Document], int]:
        """List failed documents in a collection with optional filtering.

        Args:
            collection_id: UUID of the collection
            error_category: Filter by specific error category (transient, permanent, unknown)
            retryable_only: If True, only return documents that can be retried
            max_retry_count: Max retry attempts (used when retryable_only=True)
            offset: Pagination offset
            limit: Maximum results

        Returns:
            Tuple of (documents list, total count)
        """
        try:
            # Build base query
            conditions = [
                Document.collection_id == collection_id,
                Document.status == DocumentStatus.FAILED.value,
            ]

            if error_category is not None:
                conditions.append(Document.error_category == error_category)

            if retryable_only:
                conditions.append(Document.retry_count < max_retry_count)
                # Exclude permanent errors
                from sqlalchemy import or_

                conditions.append(
                    or_(
                        Document.error_category != "permanent",
                        Document.error_category.is_(None),
                    )
                )

            query = select(Document).where(and_(*conditions))

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total = await self.session.scalar(count_query)

            # Get paginated results
            query = query.order_by(desc(Document.updated_at)).offset(offset).limit(limit)
            result = await self.session.execute(query)
            documents = result.scalars().all()

            return list(documents), total or 0

        except Exception as e:
            logger.error("Failed to list failed documents: %s", e, exc_info=True)
            raise DatabaseOperationError("list_failed_documents", "documents", str(e)) from e

    async def get_failed_document_count(
        self,
        collection_id: str,
        retryable_only: bool = False,
        max_retry_count: int = 3,
    ) -> dict[str, int]:
        """Get count of failed documents by error category.

        Args:
            collection_id: UUID of the collection
            retryable_only: If True, only count documents that can be retried
            max_retry_count: Max retry attempts (used when retryable_only=True)

        Returns:
            Dictionary with counts by category: {transient: N, permanent: N, unknown: N, total: N}
        """
        try:
            conditions = [
                Document.collection_id == collection_id,
                Document.status == DocumentStatus.FAILED.value,
            ]

            if retryable_only:
                conditions.append(Document.retry_count < max_retry_count)
                from sqlalchemy import or_

                conditions.append(
                    or_(
                        Document.error_category != "permanent",
                        Document.error_category.is_(None),
                    )
                )

            # Count by category using group by
            query = (
                select(
                    Document.error_category,
                    func.count(Document.id).label("count"),
                )
                .where(and_(*conditions))
                .group_by(Document.error_category)
            )

            result = await self.session.execute(query)
            rows = result.all()

            counts = {"transient": 0, "permanent": 0, "unknown": 0, "total": 0}
            for category, count in rows:
                key = category if category in counts else "unknown"
                counts[key] += count
                counts["total"] += count

            return counts

        except Exception as e:
            logger.error("Failed to get failed document count: %s", e, exc_info=True)
            raise DatabaseOperationError("get_failed_document_count", "documents", str(e)) from e

    async def count_failed_by_collection(self, collection_id: str) -> int:
        """Count documents with FAILED status for a collection.

        Args:
            collection_id: Collection UUID

        Returns:
            Count of failed documents
        """
        result = await self.session.execute(
            select(func.count(Document.id)).where(
                Document.collection_id == collection_id,
                Document.status == DocumentStatus.FAILED,
            )
        )
        return result.scalar() or 0

    async def count_failed_by_collections(self, collection_ids: list[str]) -> dict[str, int]:
        """Count documents with FAILED status for multiple collections.

        Args:
            collection_ids: List of collection UUIDs

        Returns:
            Dict mapping collection_id to failed count
        """
        if not collection_ids:
            return {}

        result = await self.session.execute(
            select(Document.collection_id, func.count(Document.id))
            .where(
                Document.collection_id.in_(collection_ids),
                Document.status == DocumentStatus.FAILED,
            )
            .group_by(Document.collection_id)
        )
        return {row[0]: row[1] for row in result.all()}
