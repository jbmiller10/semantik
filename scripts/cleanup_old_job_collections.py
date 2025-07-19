#!/usr/bin/env python3
"""
One-time cleanup script to remove old job_* collections from Qdrant.
These are from the legacy job-based architecture and are no longer needed.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from shared.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def cleanup_old_job_collections(dry_run: bool = True) -> None:
    """Remove old job_* collections from Qdrant."""
    try:
        # Connect to Qdrant
        qdrant_url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
        client = QdrantClient(url=qdrant_url)
        logger.info(f"Connected to Qdrant at {qdrant_url}")

        # Get all collections
        all_collections = client.get_collections().collections
        collection_names = [col.name for col in all_collections]
        logger.info(f"Found {len(collection_names)} total collections in Qdrant")

        # Find job_* collections
        job_collections = [name for name in collection_names if name.startswith("job_")]
        logger.info(f"Found {len(job_collections)} job_* collections to clean up")

        if not job_collections:
            logger.info("No job_* collections found, nothing to clean up")
            return

        # Delete each job collection
        deleted_count = 0
        for collection_name in job_collections:
            if dry_run:
                logger.info(f"[DRY RUN] Would delete collection: {collection_name}")
            else:
                try:
                    client.delete_collection(collection_name)
                    logger.info(f"Deleted collection: {collection_name}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete collection {collection_name}: {e}")

        # Summary
        if dry_run:
            logger.info(f"\n[DRY RUN] Would have deleted {len(job_collections)} job_* collections")
            logger.info("Run with --no-dry-run to actually delete collections")
        else:
            logger.info(f"\nSuccessfully deleted {deleted_count} job_* collections")

    except Exception as e:
        logger.error(f"Failed to cleanup collections: {e}")
        raise


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Clean up old job_* collections from Qdrant")
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually delete collections (default is dry run)",
    )
    args = parser.parse_args()

    # Run cleanup
    asyncio.run(cleanup_old_job_collections(dry_run=not args.no_dry_run))


if __name__ == "__main__":
    main()
