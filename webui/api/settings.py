"""
Settings and database management routes for the Web UI
"""

import os
import sys
import glob
import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from qdrant_client import AsyncQdrantClient

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vecpipe.config import settings
from webui import database
from webui.auth import get_current_user

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = str(settings.OUTPUT_DIR)

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.post("/reset-database")
async def reset_database_endpoint(current_user: Dict[str, Any] = Depends(get_current_user)):
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
            parquet_files = glob.glob(os.path.join(OUTPUT_DIR, "*.parquet"))
            for pf in parquet_files:
                os.remove(pf)
                logger.info(f"Deleted parquet file: {pf}")
        except Exception as e:
            logger.warning(f"Failed to delete parquet files: {e}")

        # Reset database
        database.reset_database()

        return {"status": "success", "message": "Database reset successfully"}
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_database_stats(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get database statistics"""
    # Get stats from database module
    stats = database.get_database_stats()

    # Get database file size
    db_size = os.path.getsize(database.DB_PATH) if os.path.exists(database.DB_PATH) else 0

    # Get total parquet files size
    parquet_files = glob.glob(os.path.join(OUTPUT_DIR, "*.parquet"))
    parquet_size = sum(os.path.getsize(f) for f in parquet_files)

    return {
        "job_count": stats["jobs"]["total"],
        "file_count": stats["files"]["total"],
        "database_size_mb": round(db_size / 1024 / 1024, 2),
        "parquet_files_count": len(parquet_files),
        "parquet_size_mb": round(parquet_size / 1024 / 1024, 2),
    }
