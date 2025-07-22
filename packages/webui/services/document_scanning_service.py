"""Document scanning service for discovering and registering documents in collections."""

import hashlib
import logging
import mimetypes
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shared.database.repositories.document_repository import DocumentRepository
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Supported file extensions for document scanning
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".text", ".pptx", ".eml", ".md", ".html"}

# Maximum file size (500 MB)
MAX_FILE_SIZE = 500 * 1024 * 1024

# Chunk size for file reading (for hash calculation)
HASH_CHUNK_SIZE = 8192


class DocumentScanningService:
    """Service for scanning directories and registering documents with deduplication."""

    def __init__(self, db_session: AsyncSession, document_repo: DocumentRepository):
        """Initialize the document scanning service.

        Args:
            db_session: Database session for transactions
            document_repo: Document repository for creating/checking documents
        """
        self.db_session = db_session
        self.document_repo = document_repo

    async def scan_directory_and_register_documents(
        self,
        collection_id: str,
        source_path: str,
        source_id: int | None = None,
        recursive: bool = True,
        batch_size: int = 100,
        progress_callback: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Scan a directory and register all supported documents with deduplication.

        This method scans the specified directory for supported file types,
        calculates content hashes, and registers them in the database.
        Duplicate files (same content hash) are automatically skipped.

        Args:
            collection_id: UUID of the collection to add documents to
            source_path: Path to directory to scan
            source_id: Optional source ID to associate with documents
            recursive: Whether to scan subdirectories recursively
            batch_size: Number of files to process before committing (default 100)
            progress_callback: Optional async callback for progress updates (documents_processed, total_documents)

        Returns:
            Dictionary with scan statistics:
                - total_documents_found: Total number of supported files found
                - new_documents_registered: Number of new files registered
                - duplicate_documents_skipped: Number of duplicate files skipped
                - errors: List of files that couldn't be processed
                - total_size_bytes: Total size of all processed files

        Raises:
            ValueError: If source_path doesn't exist or is not a directory
        """
        path = Path(source_path)
        if not path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")
        if not path.is_dir():
            raise ValueError(f"Source path is not a directory: {source_path}")

        # Initialize statistics
        stats: dict[str, Any] = {
            "total_documents_found": 0,
            "new_documents_registered": 0,
            "duplicate_documents_skipped": 0,
            "errors": [],
            "total_size_bytes": 0,
        }

        # Track scan start time for duplicate detection
        scan_start_time = datetime.now(UTC)

        # Track batch processing
        batch_count = 0
        documents_processed = 0

        # Use os.walk for memory-efficient directory traversal
        try:
            if recursive:
                # Recursive walk through all subdirectories
                for root, _, files in os.walk(source_path):
                    for filename in files:
                        file_path = Path(root) / filename

                        # Check if file extension is supported
                        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                            continue

                        stats["total_documents_found"] += 1

                        # Process individual file
                        try:
                            result = await self._register_file(
                                collection_id=collection_id,
                                file_path=file_path,
                                source_id=source_id,
                                scan_start_time=scan_start_time,
                            )

                            if result["is_new"]:
                                stats["new_documents_registered"] += 1
                            else:
                                stats["duplicate_documents_skipped"] += 1

                            stats["total_size_bytes"] += result["file_size"]
                            batch_count += 1
                            documents_processed += 1

                            # Commit batch if needed
                            if batch_count >= batch_size:
                                await self.db_session.commit()
                                batch_count = 0

                            # Call progress callback if provided
                            if progress_callback:
                                await progress_callback(documents_processed, stats["total_documents_found"])

                        except Exception as e:
                            logger.error(f"Failed to register document {file_path}: {e}")
                            stats["errors"].append(
                                {
                                    "document": str(file_path),
                                    "error": str(e),
                                }
                            )
            else:
                # Non-recursive - only scan immediate directory
                for filename in os.listdir(source_path):
                    file_path = Path(source_path) / filename

                    if not file_path.is_file():
                        continue

                    # Check if file extension is supported
                    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                        continue

                    stats["total_documents_found"] += 1

                    # Process individual file
                    try:
                        result = await self._register_file(
                            collection_id=collection_id,
                            file_path=file_path,
                            source_id=source_id,
                            scan_start_time=scan_start_time,
                        )

                        if result["is_new"]:
                            stats["new_documents_registered"] += 1
                        else:
                            stats["duplicate_documents_skipped"] += 1

                        stats["total_size_bytes"] += result["file_size"]
                        batch_count += 1
                        files_processed += 1

                        # Commit batch if needed
                        if batch_count >= batch_size:
                            await self.db_session.commit()
                            batch_count = 0

                        # Call progress callback if provided
                        if progress_callback:
                            await progress_callback(files_processed, stats["total_documents_found"])

                    except Exception as e:
                        logger.error(f"Failed to register document {file_path}: {e}")
                        stats["errors"].append(
                            {
                                "document": str(file_path),
                                "error": str(e),
                            }
                        )

            # Commit any remaining files in the batch
            if batch_count > 0:
                await self.db_session.commit()

        except Exception as e:
            logger.error(f"Error scanning directory {source_path}: {e}")
            raise

        logger.info(
            f"Scan completed for {source_path}: "
            f"{stats['total_documents_found']} documents found, "
            f"{stats['new_documents_registered']} new, "
            f"{stats['duplicate_documents_skipped']} duplicates"
        )

        return stats

    async def _register_file(
        self,
        collection_id: str,
        file_path: Path,
        source_id: int | None = None,
        scan_start_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Register a single file in the collection.

        Args:
            collection_id: UUID of the collection
            file_path: Path to the file
            source_id: Optional source ID
            scan_start_time: Start time of scan for duplicate detection

        Returns:
            Dictionary with:
                - is_new: Whether this is a new file (not duplicate)
                - document_id: ID of the created/existing document
                - file_size: Size of the file in bytes

        Raises:
            ValueError: If file is too large or cannot be accessed
        """
        # Get file stats
        try:
            stat = file_path.stat()
            file_size = stat.st_size
        except Exception as e:
            raise ValueError(f"Cannot access document at {file_path}: {e}") from e

        # Check file size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"Document too large: {file_size} bytes (max {MAX_FILE_SIZE} bytes)")

        # Calculate content hash
        content_hash = await self._calculate_file_hash(file_path)

        # Detect MIME type
        mime_type = self._get_mime_type(file_path)

        # Register document (handles deduplication internally)
        document = await self.document_repo.create(
            collection_id=collection_id,
            file_path=str(file_path),
            file_name=file_path.name,
            file_size=file_size,
            content_hash=content_hash,
            mime_type=mime_type,
            source_id=source_id,
        )

        # Check if this was a new document or existing one
        # If scan_start_time is provided, compare with document creation time
        # Documents created before scan start are considered duplicates
        is_new = True
        if scan_start_time and hasattr(document, "created_at"):
            is_new = document.created_at >= scan_start_time

        # Log duplicate detection
        if not is_new:
            logger.debug(f"Duplicate document detected: {file_path} (existing document created at {document.created_at})")

        return {
            "is_new": is_new,
            "document_id": document.id,
            "file_size": file_size,
        }

    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file contents.

        Args:
            file_path: Path to the file

        Returns:
            Hex string of SHA-256 hash (64 characters)

        Raises:
            IOError: If file cannot be read
        """
        sha256_hash = hashlib.sha256()

        try:
            with file_path.open("rb") as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b""):
                    sha256_hash.update(chunk)

            return sha256_hash.hexdigest()

        except Exception as e:
            raise OSError(f"Failed to calculate hash for {file_path}: {e}") from e

    def _get_mime_type(self, file_path: Path) -> str | None:
        """Get MIME type for a file.

        Args:
            file_path: Path to the file

        Returns:
            MIME type string or None if cannot be determined
        """
        # Try to guess MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))

        # Provide defaults for known extensions if mimetypes fails
        if not mime_type:
            mime_map = {
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".doc": "application/msword",
                ".txt": "text/plain",
                ".text": "text/plain",
                ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ".eml": "message/rfc822",
                ".md": "text/markdown",
                ".html": "text/html",
            }
            mime_type = mime_map.get(file_path.suffix.lower())

        return mime_type

    async def scan_document(
        self,
        collection_id: str,
        file_path: str,
        source_id: int | None = None,
    ) -> dict[str, Any]:
        """Scan and register a single document.

        This is a convenience method for registering individual documents.

        Args:
            collection_id: UUID of the collection
            file_path: Path to the file
            source_id: Optional source ID

        Returns:
            Dictionary with file registration result

        Raises:
            ValueError: If file doesn't exist or is not supported
        """
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"Document does not exist at: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a document: {file_path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported document type: {path.suffix}")

        result = await self._register_file(
            collection_id=collection_id,
            file_path=path,
            source_id=source_id,
            scan_start_time=datetime.now(UTC),
        )

        return {
            "document_id": result["document_id"],
            "is_new": result["is_new"],
            "file_size": result["file_size"],
            "file_name": path.name,
            "mime_type": self._get_mime_type(path),
        }
