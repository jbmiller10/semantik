"""
Settings and database management routes for the Web UI
"""

import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from qdrant_client import AsyncQdrantClient
from sqlalchemy import delete, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from packages.shared.config import settings
from packages.shared.database import get_db
from packages.shared.database.models import (
    Collection,
    CollectionAuditLog,
    CollectionPermission,
    CollectionResourceLimits,
    CollectionSource,
    Document,
    Operation,
    OperationMetrics,
)
from packages.webui.auth import get_current_user

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = str(settings.output_dir)

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.post("/reset-database")
async def reset_database_endpoint(
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Reset the database - ADMIN ONLY"""
    # Check if user is admin/superuser
    if not current_user.get("is_superuser", False):
        raise HTTPException(status_code=403, detail="Only administrators can reset the database")

    try:
        # Get all collections before reset
        # We need to get all collections, not just for the current user
        # Since this is an admin function, we'll query all collections
        result = await db.execute(select(Collection))
        collections = result.scalars().all()

        # Delete Qdrant collections for all collections
        async_client = AsyncQdrantClient(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        for collection in collections:
            collection_name = str(collection.vector_store_name)
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

        # Clear all tables in the database
        # Note: This is a destructive operation and should be protected
        # We'll delete all records from the tables in the correct order to respect foreign keys
        try:
            # Delete in order of dependencies (most dependent first)
            await db.execute(delete(OperationMetrics))
            await db.execute(delete(CollectionAuditLog))
            await db.execute(delete(CollectionResourceLimits))
            await db.execute(delete(CollectionPermission))
            await db.execute(delete(Operation))
            await db.execute(delete(Document))
            await db.execute(delete(CollectionSource))
            await db.execute(delete(Collection))
            # Note: We don't delete Users, ApiKeys, or RefreshTokens as they're auth-related

            await db.commit()
            logger.info("Database tables cleared successfully")
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to clear database tables: {e}")
            raise

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

    # Get database size from PostgreSQL when available
    database_size_bytes = 0
    size_query = select(func.pg_database_size(func.current_database()))

    try:
        size_result = await db.scalar(size_query)
        if size_result is not None:
            database_size_bytes = int(size_result)
    except OperationalError as exc:
        logger.warning("Failed querying database size: %s", exc)
    except Exception as exc:  # pragma: no cover - unexpected error path
        logger.warning("Unexpected error while querying database size: %s", exc)

    # Get total parquet files size
    output_path = Path(OUTPUT_DIR)
    parquet_files = list(output_path.glob("*.parquet"))
    parquet_size = sum(f.stat().st_size for f in parquet_files)

    return {
        "collection_count": collection_count,
        "file_count": document_count,
        "database_size_mb": round(database_size_bytes / 1024 / 1024, 2),
        "parquet_files_count": len(parquet_files),
        "parquet_size_mb": round(parquet_size / 1024 / 1024, 2),
    }
