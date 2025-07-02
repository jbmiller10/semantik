#!/usr/bin/env python3
"""
Document cleanup service for Qdrant vector database
Removes vectors for deleted documents from all collections
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import settings
from .extract_chunks import FileChangeTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
# Using settings for some values, keeping others as constants for now


class QdrantCleanupService:
    """Service to clean up vectors for deleted documents"""

    def __init__(self, qdrant_host: str = None, qdrant_port: int = None):
        if qdrant_host is None:
            qdrant_host = settings.QDRANT_HOST
        if qdrant_port is None:
            qdrant_port = settings.QDRANT_PORT
        self.client = QdrantClient(url=f"http://{qdrant_host}:{qdrant_port}")
        self.tracker = FileChangeTracker()

    def get_current_files(self, file_list_path: str) -> list[str]:
        """Read current file list from null-delimited file"""
        if not os.path.exists(file_list_path):
            logger.error(f"File list not found: {file_list_path}")
            return []

        with open(file_list_path, "rb") as f:
            content = f.read()
            files = content.decode("utf-8").split("\0")
            # Filter out empty strings
            files = [f for f in files if f.strip()]

        logger.info(f"Found {len(files)} current files")
        return files

    def get_job_collections(self) -> list[str]:
        """Get all job collection names from webui database"""
        collections = [settings.DEFAULT_COLLECTION]

        if not os.path.exists(settings.WEBUI_DB):
            logger.warning(f"WebUI database not found: {settings.WEBUI_DB}")
            return collections

        try:
            conn = sqlite3.connect(str(settings.WEBUI_DB))
            c = conn.cursor()

            # Get all job IDs
            job_ids = c.execute("SELECT id FROM jobs").fetchall()

            for (job_id,) in job_ids:
                collections.append(f"job_{job_id}")

            conn.close()

            logger.info(f"Found {len(collections)} collections to check")
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
            return count

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
            "timestamp": datetime.now().isoformat(),
            "removed_files": len(removed_files),
            "deleted_points": total_deleted,
            "by_collection": deleted_by_collection,
            "dry_run": dry_run,
        }

        logger.info(f"Cleanup summary: {json.dumps(summary, indent=2)}")

        # Write to cleanup log
        try:
            with open(settings.CLEANUP_LOG, "a") as f:
                f.write(json.dumps(summary) + "\n")
        except Exception as e:
            logger.error(f"Failed to write cleanup log: {e}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Clean up vectors for deleted documents")
    parser.add_argument(
        "--file-list", "-f", default=str(settings.MANIFEST_FILE), help="Path to null-delimited file list"
    )
    parser.add_argument("--dry-run", "-n", action="store_true", help="Perform dry run without deleting")
    parser.add_argument("--qdrant-host", default=QDRANT_HOST, help="Qdrant host address")
    parser.add_argument("--qdrant-port", type=int, default=QDRANT_PORT, help="Qdrant port")

    args = parser.parse_args()

    # Create cleanup service
    service = QdrantCleanupService(args.qdrant_host, args.qdrant_port)

    try:
        # Get current files
        current_files = service.get_current_files(args.file_list)

        if not current_files:
            logger.error("No current files found, exiting")
            return 1

        # Run cleanup
        result = service.cleanup_removed_files(current_files, dry_run=args.dry_run)

        # Exit with success if no errors
        return 0

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
