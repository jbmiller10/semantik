"""Repository implementation for Document model."""

import logging
import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from shared.database.exceptions import DatabaseOperationError, EntityNotFoundError, ValidationError
from shared.database.models import Collection, Document, DocumentStatus
from shared.database.db_retry import with_db_retry
from sqlalchemy import and_, delete, desc, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

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
            )

            self.session.add(document)
            await self.session.flush()

            logger.info(
                f"Created document {document.id} for collection {collection_id} with content_hash {content_hash}"
            )
            return document

        except IntegrityError as e:
            # Handle race condition where another process created the same document
            logger.warning(f"Integrity error creating document, checking for existing: {e}")
            existing_doc = await self.get_by_content_hash(collection_id, content_hash)
            if existing_doc:
                return existing_doc
            raise DatabaseOperationError("create", "document", str(e)) from e
        except (EntityNotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Failed to create document: {e}")
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
            logger.error(f"Failed to get document {document_id}: {e}")
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
            logger.error(f"Failed to get document by content_hash {content_hash} in collection {collection_id}: {e}")
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

            # Get paginated results
            query = query.order_by(desc(Document.created_at)).offset(offset).limit(limit)
            result = await self.session.execute(query)
            documents = result.scalars().all()

            return list(documents), total or 0

        except Exception as e:
            logger.error(f"Failed to list documents for collection {collection_id}: {e}")
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
            logger.error(f"Failed to list duplicate documents: {e}")
            raise DatabaseOperationError("list", "duplicate documents", str(e)) from e

    async def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: str | None = None,
        chunk_count: int | None = None,
    ) -> Document:
        """Update document status.

        Args:
            document_id: UUID of the document
            status: New status
            error_message: Optional error message
            chunk_count: Optional chunk count

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
            document.updated_at = datetime.now(UTC)

            await self.session.flush()

            logger.info(f"Updated document {document_id} status to {status}")
            return document

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update document status: {e}")
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
            count = result.rowcount

            logger.info(f"Updated {count} documents to status {status}")
            return count

        except Exception as e:
            logger.error(f"Failed to bulk update document status: {e}")
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

            self.session.delete(document)
            await self.session.flush()

            logger.info(f"Deleted document {document_id}")

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
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
            logger.error(f"Failed to delete duplicate documents: {e}")
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
            logger.error(f"Failed to get document stats: {e}")
            raise DatabaseOperationError("get stats", "documents", str(e)) from e
