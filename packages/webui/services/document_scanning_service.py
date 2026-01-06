"""Document scanning service for discovering and registering documents in collections."""

import asyncio
import logging
import mimetypes
import os
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import settings
from shared.database.repositories.document_repository import DocumentRepository
from shared.dtos.ingestion import IngestedDocument
from shared.utils.hashing import compute_file_hash
from webui.services.document_registry_service import DocumentRegistryService

logger = logging.getLogger(__name__)

# Supported file extensions for document scanning
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".text", ".pptx", ".eml", ".md", ".html"}

# Maximum file size (500 MB)
MAX_FILE_SIZE = 500 * 1024 * 1024

# Parallel registration settings
DEFAULT_HASH_WORKERS = 4
MAX_HASH_WORKERS = 8
PARALLEL_REGISTRATION_BATCH_SIZE = 50


class DocumentScanningService:
    """Service for scanning directories and registering documents with deduplication.

    This service handles filesystem traversal and delegates document registration
    to the DocumentRegistryService for consistent handling across all connectors.
    """

    def __init__(self, db_session: AsyncSession, document_repo: DocumentRepository):
        """Initialize the document scanning service.

        Args:
            db_session: Database session for transactions
            document_repo: Document repository for creating/checking documents
        """
        self.db_session = db_session
        self.document_repo = document_repo
        # Create registry service for document registration
        self._registry: DocumentRegistryService = DocumentRegistryService(db_session, document_repo)

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

        # Check if parallel registration is enabled
        use_parallel = getattr(settings, "PARALLEL_INGESTION_ENABLED", True)

        if use_parallel:
            return await self._scan_directory_parallel(
                collection_id=collection_id,
                source_path=path,
                source_id=source_id,
                recursive=recursive,
                batch_size=batch_size,
                progress_callback=progress_callback,
                stats=stats,
                scan_start_time=scan_start_time,
            )

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
                            logger.error("Failed to register document %s: %s", file_path, e, exc_info=True)
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
                        documents_processed += 1

                        # Commit batch if needed
                        if batch_count >= batch_size:
                            await self.db_session.commit()
                            batch_count = 0

                        # Call progress callback if provided
                        if progress_callback:
                            await progress_callback(documents_processed, stats["total_documents_found"])

                    except Exception as e:
                        logger.error("Failed to register document %s: %s", file_path, e, exc_info=True)
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
            logger.error("Error scanning directory %s: %s", source_path, e, exc_info=True)
            raise

        logger.info(
            "Scan completed for %s: %d documents found, %d new, %d duplicates",
            source_path,
            stats["total_documents_found"],
            stats["new_documents_registered"],
            stats["duplicate_documents_skipped"],
        )

        return stats

    async def _register_file(  # noqa: ARG002
        self,
        collection_id: str,
        file_path: Path,
        source_id: int | None = None,
        scan_start_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Register a single file in the collection.

        This method builds an IngestedDocument DTO from the file and delegates
        registration to the DocumentRegistryService.

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
        # Retain parameter for future duplicate detection logic without dropping caller compatibility.
        _ = scan_start_time

        # Get file stats
        try:
            stat = file_path.stat()
            file_size = stat.st_size
        except Exception as e:
            raise ValueError(f"Cannot access document at {file_path}: {e}") from e

        # Check file size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"Document too large: {file_size} bytes (max {MAX_FILE_SIZE} bytes)")

        # Calculate content hash (uses chunked reading for large files)
        content_hash = await self._calculate_file_hash(file_path)

        # Detect MIME type
        mime_type = self._get_mime_type(file_path)

        # Build IngestedDocument DTO
        ingested = IngestedDocument(
            content="",  # Content not needed for registration phase
            unique_id=f"file://{file_path}",
            source_type="directory",
            metadata={
                "file_size": file_size,
                "mime_type": mime_type,
            },
            content_hash=content_hash,
            file_path=str(file_path),
        )

        # Delegate to registry service for registration and deduplication
        result: dict[str, Any] = await self._registry.register(
            collection_id=collection_id,
            ingested=ingested,
            source_id=source_id,
        )
        return result

    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file contents.

        Delegates to shared.utils.hashing.compute_file_hash for consistent
        hashing across the codebase. Streaming behavior is preserved for
        memory-efficient handling of large files.

        Args:
            file_path: Path to the file

        Returns:
            Hex string of SHA-256 hash (64 characters)

        Raises:
            IOError: If file cannot be read
        """
        result: str = compute_file_hash(file_path)
        return result

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

    async def _scan_directory_parallel(
        self,
        collection_id: str,
        source_path: Path,
        source_id: int | None,
        recursive: bool,
        batch_size: int,
        progress_callback: Callable[[int, int], Awaitable[None]] | None,
        stats: dict[str, Any],
        scan_start_time: datetime,
    ) -> dict[str, Any]:
        """Parallel version of directory scanning with concurrent hash computation.

        This method:
        1. Collects all file paths first (fast os.walk)
        2. Computes file hashes in parallel using ThreadPoolExecutor
        3. Registers documents in batches

        Args:
            collection_id: UUID of the collection
            source_path: Path to directory
            source_id: Optional source ID
            recursive: Whether to scan subdirectories
            batch_size: Batch size for commits
            progress_callback: Optional progress callback
            stats: Statistics dict to update
            scan_start_time: Scan start time

        Returns:
            Updated statistics dict
        """
        # Retain parameter for future duplicate detection logic
        _ = scan_start_time

        # Step 1: Collect all file paths (fast)
        logger.info("Collecting files from %s (recursive=%s)", source_path, recursive)
        file_paths: list[Path] = []

        if recursive:
            for root, _, files in os.walk(source_path):
                for filename in files:
                    file_path = Path(root) / filename
                    if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        file_paths.append(file_path)
        else:
            for filename in os.listdir(source_path):
                file_path = source_path / filename
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    file_paths.append(file_path)

        stats["total_documents_found"] = len(file_paths)
        logger.info("Found %d documents to process", len(file_paths))

        if not file_paths:
            return stats

        # Step 2: Compute file metadata (size, hash) in parallel
        num_workers = min(
            getattr(settings, "PARALLEL_INGESTION_WORKERS", DEFAULT_HASH_WORKERS) or DEFAULT_HASH_WORKERS,
            MAX_HASH_WORKERS,
            len(file_paths),
        )

        logger.info("Computing file hashes with %d parallel workers", num_workers)

        # Use ThreadPoolExecutor for I/O-bound hash computation
        loop = asyncio.get_running_loop()

        def compute_file_metadata(file_path: Path) -> dict[str, Any] | None:
            """Compute file size and hash. Returns None on error."""
            try:
                stat = file_path.stat()
                file_size = stat.st_size

                if file_size > MAX_FILE_SIZE:
                    return {"error": f"File too large: {file_size} bytes", "path": str(file_path)}

                content_hash = compute_file_hash(file_path)
                mime_type = mimetypes.guess_type(str(file_path))[0]

                return {
                    "path": file_path,
                    "size": file_size,
                    "hash": content_hash,
                    "mime_type": mime_type,
                }
            except Exception as e:
                return {"error": str(e), "path": str(file_path)}

        # Process files in parallel
        file_metadata: list[dict[str, Any] | None] = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [loop.run_in_executor(executor, compute_file_metadata, fp) for fp in file_paths]
            file_metadata = await asyncio.gather(*futures)

        logger.info("Hash computation complete, registering documents")

        # Step 3: Register documents (database operations, sequential but batched)
        documents_processed = 0
        batch_count = 0

        for meta in file_metadata:
            if meta is None:
                continue

            if "error" in meta:
                stats["errors"].append({"document": meta["path"], "error": meta["error"]})
                continue

            try:
                file_path = meta["path"]

                # Build IngestedDocument DTO
                ingested = IngestedDocument(
                    content="",  # Content not needed for registration phase
                    unique_id=f"file://{file_path}",
                    source_type="directory",
                    metadata={
                        "file_size": meta["size"],
                        "mime_type": meta["mime_type"],
                    },
                    content_hash=meta["hash"],
                    file_path=str(file_path),
                )

                # Register with deduplication
                result = await self._registry.register(
                    collection_id=collection_id,
                    ingested=ingested,
                    source_id=source_id,
                )

                if result["is_new"]:
                    stats["new_documents_registered"] += 1
                else:
                    stats["duplicate_documents_skipped"] += 1

                stats["total_size_bytes"] += meta["size"]
                batch_count += 1
                documents_processed += 1

                # Commit batch if needed
                if batch_count >= batch_size:
                    await self.db_session.commit()
                    batch_count = 0
                    logger.debug("Committed batch of %d documents", batch_size)

                # Progress callback
                if progress_callback:
                    await progress_callback(documents_processed, stats["total_documents_found"])

            except Exception as e:
                logger.error("Failed to register document %s: %s", meta.get("path"), e, exc_info=True)
                stats["errors"].append({"document": str(meta.get("path")), "error": str(e)})

        # Final commit
        if batch_count > 0:
            await self.db_session.commit()

        logger.info(
            "Parallel registration complete: %d new, %d duplicates, %d errors",
            stats["new_documents_registered"],
            stats["duplicate_documents_skipped"],
            len(stats["errors"]),
        )

        return stats
