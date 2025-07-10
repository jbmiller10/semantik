#!/usr/bin/env python3
"""
Maintenance service for Qdrant vector database
Removes vectors for deleted documents from all collections
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared.config import settings

from .extract_chunks import FileChangeTracker

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
        self.tracker = FileChangeTracker()
        self.webui_base_url = f"http://{webui_host}:{webui_port}"

    def get_current_files(self, file_list_path: str) -> list[str]:
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

    def get_job_collections(self) -> list[str]:
        """Get all job collection names from webui API"""
        collections = [settings.DEFAULT_COLLECTION]

        try:
            # Call the internal API endpoint to get all job IDs
            response = httpx.get(f"{self.webui_base_url}/api/internal/jobs/all-ids", timeout=30.0)
            response.raise_for_status()

            job_ids = response.json()

            for job_id in job_ids:
                collections.append(f"job_{job_id}")

            logger.info(f"Found {len(collections)} collections to check")
            return collections

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch job collections from API: {e}")
            return collections
        except Exception as e:
            logger.error(f"Failed to read job collections: {e}")
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

    def cleanup_removed_files(self, current_files: list[str], dry_run: bool = False) -> dict:
        """Main cleanup logic"""
        # Get removed files
        removed_files = self.tracker.get_removed_files(current_files)

        if not removed_files:
            logger.info("No removed files detected")
            return {"removed_files": 0, "deleted_points": 0}

        logger.info(f"Found {len(removed_files)} removed files")

        # Get all collections to clean
        collections = self.get_job_collections()

        # Track statistics
        total_deleted = 0
        deleted_by_collection = {}

        # Process each removed file
        for removed_file in removed_files:
            doc_id = removed_file["doc_id"]
            file_path = removed_file["path"]
            logger.info(f"Processing removed file: {file_path} (doc_id: {doc_id})")

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
            for removed_file in removed_files:
                self.tracker.remove_file(removed_file["path"])
            self.tracker.save()

        # Log summary
        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "removed_files": len(removed_files),
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

    def cleanup_orphaned_collections(self, dry_run: bool = False) -> dict:
        """Clean up Qdrant collections that don't have corresponding jobs in webui"""
        try:
            # Get all valid job IDs from webui
            valid_collections = set(self.get_job_collections())

            # Get all collections from Qdrant
            all_collections = self.client.get_collections().collections
            qdrant_collections = {col.name for col in all_collections}

            # Find orphaned collections (those that start with "job_" but aren't in valid list)
            orphaned = []
            for col_name in qdrant_collections:
                if col_name.startswith("job_") and col_name not in valid_collections:
                    orphaned.append(col_name)

            deleted_collections = []
            if orphaned:
                logger.info(f"Found {len(orphaned)} orphaned collections")
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

            summary = {
                "timestamp": datetime.now(UTC).isoformat(),
                "orphaned_collections": orphaned,
                "deleted_collections": deleted_collections,
                "dry_run": dry_run,
            }

            return summary

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
        help="Clean up orphaned Qdrant collections that don't have corresponding jobs",
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
            current_files = service.get_current_files(args.file_list)

            if not current_files:
                logger.error("No current files found, exiting")
                return 1

            # Run file cleanup
            service.cleanup_removed_files(current_files, dry_run=args.dry_run)

        # Exit with success if no errors
        return 0

    except Exception as e:
        logger.error(f"Maintenance operation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
