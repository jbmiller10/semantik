#!/usr/bin/env python3
"""
Maintenance service for Qdrant vector database
Removes vectors for deleted documents from all collections
"""

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared.config import settings

from .document_tracker import DocumentChangeTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
# Using settings for some values, keeping others as constants for now


class QdrantMaintenanceService:
    """Service to maintain vectors and clean up deleted documents"""

    def __init__(
        self,
        qdrant_host: str | None = None,
        qdrant_port: int | None = None,
        webui_host: str = "localhost",
        webui_port: int = 5555,
    ):
        if qdrant_host is None:
            qdrant_host = settings.QDRANT_HOST
        if qdrant_port is None:
            qdrant_port = settings.QDRANT_PORT
        self.client = QdrantClient(url=f"http://{qdrant_host}:{qdrant_port}")
        self.tracker = DocumentChangeTracker()
        self.webui_base_url = f"http://{webui_host}:{webui_port}"

    def _retry_request(self, func: Any, max_attempts: int = 3, base_delay: float = 1.0) -> Any:
        """Execute a request with exponential backoff retry logic."""
        last_exception: Exception | None = None

        for attempt in range(max_attempts):
            try:
                return func()
            except httpx.HTTPStatusError as e:
                # Don't retry on 4xx errors (client errors)
                if 400 <= e.response.status_code < 500:
                    raise
                last_exception = e
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exception = e

            if attempt < max_attempts - 1:
                delay = base_delay * (2**attempt)  # Exponential backoff
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_attempts}), retrying in {delay}s: {last_exception}"
                )
                time.sleep(delay)

        logger.error(f"Request failed after {max_attempts} attempts")
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Request failed after {max_attempts} attempts")

    def get_current_documents(self, file_list_path: str) -> list[str]:
        """Read current file list from null-delimited file"""
        file_path = Path(file_list_path)
        if not file_path.exists():
            logger.error(f"File list not found: {file_list_path}")
            return []

        with file_path.open("rb") as f:
            content = f.read()
            files = content.decode("utf-8").split("\0")
            # Filter out empty strings
            files = [f for f in files if f.strip()]

        logger.info(f"Found {len(files)} current files")
        return files

    def get_operation_collections(self) -> list[str]:
        """Get all collection names from webui API
        
        Note: This now returns only the default collection. In the new collection-centric
        architecture, collections have their own Qdrant collection names stored in the
        qdrant_collections field, not operation-based naming.
        """
        collections = [settings.DEFAULT_COLLECTION]
        # Legacy operation-based collections (operation_{uuid}) are no longer used
        logger.info(f"Found {len(collections)} collections to check")
        return collections

    def get_active_collections(self) -> list[str]:
        """Get all active collection names from the database"""
        collections = [settings.DEFAULT_COLLECTION]

        try:
            # Call the internal API endpoint to get all collection vector_store_names
            headers = {}
            if settings.INTERNAL_API_KEY:
                headers["X-Internal-Api-Key"] = settings.INTERNAL_API_KEY

            def make_request() -> list[str]:
                response = httpx.get(
                    f"{self.webui_base_url}/api/internal/collections/vector-store-names", headers=headers, timeout=30.0
                )
                response.raise_for_status()
                result: list[str] = response.json()
                return result

            vector_store_names = self._retry_request(make_request)
            collections.extend(vector_store_names)

            # Remove duplicates
            collections = list(set(collections))

            logger.info(f"Found {len(collections)} active collections: {collections}")
            return collections

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch active collections from API: {e}")
            return collections
        except Exception as e:
            logger.error(f"Unexpected error while fetching active collections: {e}")
            return collections

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in Qdrant"""
        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            # Collection doesn't exist or other error
            return False

    def delete_points_by_doc_id(self, collection_name: str, doc_id: str) -> int:
        """Delete all points with given doc_id from collection"""
        if not self.collection_exists(collection_name):
            logger.info(f"Collection {collection_name} does not exist, skipping")
            return 0

        try:
            # First, check how many points we're about to delete
            count_filter = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])

            count_result = self.client.count(collection_name=collection_name, count_filter=count_filter)

            count = count_result.count

            if count == 0:
                return 0

            # Delete the points
            self.client.delete(collection_name=collection_name, points_selector=FilterSelector(filter=count_filter))

            logger.info(f"Deleted {count} points with doc_id={doc_id} from {collection_name}")
            return int(count)

        except Exception as e:
            logger.error(f"Error deleting points from {collection_name}: {e}")
            return 0

    def cleanup_removed_documents(self, current_documents: list[str], dry_run: bool = False) -> dict:
        """Main cleanup logic"""
        # Get removed files
        removed_documents = self.tracker.get_removed_documents(current_documents)

        if not removed_documents:
            logger.info("No removed documents detected")
            return {"removed_documents": 0, "deleted_points": 0}

        logger.info(f"Found {len(removed_documents)} removed documents")

        # Get all collections to clean
        collections = self.get_active_collections()

        # Track statistics
        total_deleted = 0
        deleted_by_collection = {}

        # Process each removed file
        for removed_file in removed_documents:
            doc_id = removed_file["doc_id"]
            file_path = removed_file["path"]
            logger.info(f"Processing removed document: {file_path} (doc_id: {doc_id})")

            if dry_run:
                logger.info(f"[DRY RUN] Would delete points with doc_id={doc_id}")
                continue

            # Delete from each collection
            for collection in collections:
                deleted_count = self.delete_points_by_doc_id(collection, doc_id)

                if deleted_count > 0:
                    total_deleted += deleted_count
                    if collection not in deleted_by_collection:
                        deleted_by_collection[collection] = 0
                    deleted_by_collection[collection] += deleted_count

        # Update tracker to remove these files from tracking
        if not dry_run:
            for removed_file in removed_documents:
                self.tracker.remove_file(removed_file["path"])
            self.tracker.save()

        # Log summary
        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "removed_documents": len(removed_documents),
            "deleted_points": total_deleted,
            "by_collection": deleted_by_collection,
            "dry_run": dry_run,
        }

        logger.info(f"Cleanup summary: {json.dumps(summary, indent=2)}")

        # Write to cleanup log
        try:
            with Path(settings.cleanup_log).open("a") as f:
                f.write(json.dumps(summary) + "\n")
        except Exception as e:
            logger.error(f"Failed to write cleanup log: {e}")

        return summary

    def cleanup_orphaned_collections(self, dry_run: bool = False, grace_period_minutes: int = 60) -> dict:
        """Clean up Qdrant collections that don't have corresponding entries in the database

        Args:
            dry_run: If True, only log what would be deleted without actually deleting
            grace_period_minutes: Don't delete collections created within this many minutes (default: 60)
        """
        # Note: grace_period_minutes parameter reserved for future implementation
        _ = grace_period_minutes  # Acknowledge unused parameter

        try:
            # Get all valid collection names from the database
            valid_collections = set(self.get_active_collections())

            # Get all collections from Qdrant
            all_collections = self.client.get_collections().collections
            qdrant_collections = {col.name for col in all_collections}

            # Find orphaned collections (those in Qdrant but not in database)
            orphaned = []
            for col_name in qdrant_collections:
                # Check if collection is not in the valid list and matches known patterns
                if col_name not in valid_collections:
                    # Only consider collections with known patterns as orphaned
                    if col_name.startswith(("operation_", "col_")) and col_name != settings.DEFAULT_COLLECTION:
                        # Add to orphaned list
                        orphaned.append(col_name)
                        logger.info(f"Found orphaned collection: {col_name}")
                    else:
                        logger.debug(f"Skipping unknown collection pattern: {col_name}")

            deleted_collections = []
            if orphaned:
                logger.info(f"Found {len(orphaned)} orphaned collections: {orphaned}")
                for col_name in orphaned:
                    if dry_run:
                        logger.info(f"[DRY RUN] Would delete collection: {col_name}")
                    else:
                        try:
                            self.client.delete_collection(col_name)
                            logger.info(f"Deleted orphaned collection: {col_name}")
                            deleted_collections.append(col_name)
                        except Exception as e:
                            logger.error(f"Failed to delete collection {col_name}: {e}")
            else:
                logger.info("No orphaned collections found")

            # Log summary of valid vs orphaned collections
            logger.info(f"Valid collections in database: {len(valid_collections)}")
            logger.info(f"Total collections in Qdrant: {len(qdrant_collections)}")

            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "orphaned_collections": orphaned,
                "deleted_collections": deleted_collections,
                "valid_collections_count": len(valid_collections),
                "qdrant_collections_count": len(qdrant_collections),
                "dry_run": dry_run,
            }

        except Exception as e:
            logger.error(f"Failed to cleanup orphaned collections: {e}")
            return {"error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Maintenance service for Qdrant vector database")
    parser.add_argument(
        "--file-list", "-f", default=str(settings.manifest_file), help="Path to null-delimited file list"
    )
    parser.add_argument("--dry-run", "-n", action="store_true", help="Perform dry run without deleting")
    parser.add_argument("--qdrant-host", default=settings.QDRANT_HOST, help="Qdrant host address")
    parser.add_argument("--qdrant-port", type=int, default=settings.QDRANT_PORT, help="Qdrant port")
    parser.add_argument("--webui-host", default="localhost", help="WebUI host address")
    parser.add_argument("--webui-port", type=int, default=5555, help="WebUI port")
    parser.add_argument(
        "--cleanup-orphaned",
        action="store_true",
        help="Clean up orphaned Qdrant collections that don't have corresponding operations",
    )

    args = parser.parse_args()

    # Create maintenance service
    service = QdrantMaintenanceService(args.qdrant_host, args.qdrant_port, args.webui_host, args.webui_port)

    try:
        if args.cleanup_orphaned:
            # Run orphaned collection cleanup
            logger.info("Running orphaned collection cleanup")
            result = service.cleanup_orphaned_collections(dry_run=args.dry_run)
            logger.info(f"Orphaned collection cleanup result: {result}")
        else:
            # Get current files
            current_documents = service.get_current_documents(args.file_list)

            if not current_documents:
                logger.error("No current files found, exiting")
                return 1

            # Run file cleanup
            service.cleanup_removed_documents(current_documents, dry_run=args.dry_run)

        # Exit with success if no errors
        return 0

    except Exception as e:
        logger.error(f"Maintenance operation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
