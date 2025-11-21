"""
Settings and database management routes for the Web UI
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from qdrant_client import AsyncQdrantClient
from shared.config import settings
from shared.database import get_db
from shared.database.models import (
    Collection,
    CollectionAuditLog,
    CollectionPermission,
    CollectionResourceLimits,
    CollectionSource,
    Document,
    Operation,
    OperationMetrics,
)
from sqlalchemy import delete, func, select
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from webui.auth import get_current_user

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
    collection_count = 0
    collection_count_query = select(func.count()).select_from(Collection)

    try:
        collection_count = await db.scalar(collection_count_query) or 0
    except SQLAlchemyError as exc:
        logger.warning("Failed querying collection count: %s", exc)
        if db.in_transaction():  # pragma: no branch - defensive rollback
            await db.rollback()

    document_count = 0
    document_count_query = select(func.count()).select_from(Document)

    try:
        document_count = await db.scalar(document_count_query) or 0
    except SQLAlchemyError as exc:
        logger.warning("Failed querying document count: %s", exc)
        if db.in_transaction():  # pragma: no branch - defensive rollback
            await db.rollback()

    # Get database size from PostgreSQL when available
    database_size_mb: float | None = None
    size_query = select(func.pg_database_size(func.current_database()))

    try:
        size_result = await db.scalar(size_query)
        if size_result is not None:
            database_size_mb = round(int(size_result) / 1024 / 1024, 2)
    except OperationalError as exc:
        logger.warning("Failed querying database size: %s", exc)
        if db.in_transaction():
            await db.rollback()
    except SQLAlchemyError as exc:
        logger.warning("Unexpected SQL error while querying database size: %s", exc)
        if db.in_transaction():
            await db.rollback()
    except Exception as exc:  # pragma: no cover - unexpected error path
        logger.warning("Unexpected error while querying database size: %s", exc)
        if db.in_transaction():
            await db.rollback()

    # Get total parquet files size
    output_path = Path(OUTPUT_DIR)
    parquet_files: list[Path] = []
    parquet_size_bytes = 0

    try:
        if output_path.exists():
            parquet_files = list(output_path.glob("*.parquet"))
            for parquet_file in parquet_files:
                try:
                    parquet_size_bytes += parquet_file.stat().st_size
                except OSError as exc:
                    logger.warning("Failed to stat parquet file %s: %s", parquet_file, exc)
        else:
            logger.debug("Parquet output directory does not exist: %s", output_path)
    except Exception as exc:  # pragma: no cover - filesystem errors are unexpected
        logger.warning("Unexpected error while scanning parquet files: %s", exc)

    parquet_size_mb = round(parquet_size_bytes / 1024 / 1024, 2) if parquet_size_bytes else 0.0

    return {
        "collection_count": collection_count,
        "file_count": document_count,
        "database_size_mb": database_size_mb,
        "parquet_files_count": len(parquet_files),
        "parquet_size_mb": parquet_size_mb,
    }
