"""
Settings and database management routes for the Web UI
"""

import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from qdrant_client import AsyncQdrantClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from shared import database
from shared.config import settings
from shared.database import get_db
from shared.database.models import Collection, Document
from webui.auth import get_current_user

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = str(settings.output_dir)

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.post("/reset-database")
async def reset_database_endpoint(
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Reset the database"""
    try:
        # Get all collections before reset
        # We need to get all collections, not just for the current user
        # Since this is an admin function, we'll query all collections
        result = await db.execute(select(Collection))
        collections = result.scalars().all()

        # Delete Qdrant collections for all collections
        async_client = AsyncQdrantClient(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        for collection in collections:
            collection_name = collection.vector_store_name
            try:
                await async_client.delete_collection(collection_name)
                logger.info(f"Deleted Qdrant collection: {collection_name}")
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
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get database statistics"""
    # Get collection and document counts by querying directly
    # Get collection count
    collection_count_query = select(func.count()).select_from(Collection)
    collection_count = await db.scalar(collection_count_query) or 0

    # Get document count
    document_count_query = select(func.count()).select_from(Document)
    document_count = await db.scalar(document_count_query) or 0

    # Get database file size
    db_path = Path(database.DB_PATH)
    db_size = db_path.stat().st_size if db_path.exists() else 0

    # Get total parquet files size
    output_path = Path(OUTPUT_DIR)
    parquet_files = list(output_path.glob("*.parquet"))
    parquet_size = sum(f.stat().st_size for f in parquet_files)

    return {
        "collection_count": collection_count,
        "file_count": document_count,
        "database_size_mb": round(db_size / 1024 / 1024, 2),
        "parquet_files_count": len(parquet_files),
        "parquet_size_mb": round(parquet_size / 1024 / 1024, 2),
    }
