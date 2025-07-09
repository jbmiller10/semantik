"""
Settings and database management routes for the Web UI
"""

import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from qdrant_client import AsyncQdrantClient

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from shared.config import settings

from webui import database
from webui.auth import get_current_user

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = str(settings.output_dir)

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.post("/reset-database")
async def reset_database_endpoint(
    current_user: dict[str, Any] = Depends(get_current_user)  # noqa: ARG001
) -> dict[str, str]:
    """Reset the database"""
    try:
        # Get all job IDs before reset
        jobs = database.list_jobs()
        job_ids = [job["id"] for job in jobs]

        # Delete Qdrant collections for all jobs
        async_client = AsyncQdrantClient(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        for job_id in job_ids:
            collection_name = f"job_{job_id}"
            try:
                await async_client.delete_collection(collection_name)
            except Exception as e:
                logger.warning(f"Failed to delete collection {collection_name}: {e}")

        # Also delete the metadata collection
        try:
            await async_client.delete_collection("_collection_metadata")
            logger.info("Deleted metadata collection")
        except Exception as e:
            logger.warning(f"Failed to delete metadata collection: {e}")

        # Delete all parquet files
        try:
            output_path = Path(OUTPUT_DIR)
            parquet_files = list(output_path.glob("*.parquet"))
            for pf in parquet_files:
                pf.unlink()
                logger.info(f"Deleted parquet file: {pf}")
        except Exception as e:
            logger.warning(f"Failed to delete parquet files: {e}")

        # Reset database
        database.reset_database()

        return {"status": "success", "message": "Database reset successfully"}
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/stats")
async def get_database_stats(
    current_user: dict[str, Any] = Depends(get_current_user)  # noqa: ARG001
) -> dict[str, Any]:
    """Get database statistics"""
    # Get stats from database module
    stats = database.get_database_stats()

    # Get database file size
    db_path = Path(database.DB_PATH)
    db_size = db_path.stat().st_size if db_path.exists() else 0

    # Get total parquet files size
    output_path = Path(OUTPUT_DIR)
    parquet_files = list(output_path.glob("*.parquet"))
    parquet_size = sum(f.stat().st_size for f in parquet_files)

    return {
        "job_count": stats["jobs"]["total"],
        "file_count": stats["files"]["total"],
        "database_size_mb": round(db_size / 1024 / 1024, 2),
        "parquet_files_count": len(parquet_files),
        "parquet_size_mb": round(parquet_size / 1024 / 1024, 2),
    }
