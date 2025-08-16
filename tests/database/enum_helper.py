"""Helper module for safely creating PostgreSQL enum types in tests.

This module provides utilities to handle concurrent enum type creation
when running tests in parallel with pytest-xdist.
"""

import asyncio
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.orm import Session


async def create_enum_types_if_not_exist(conn: AsyncConnection) -> None:
    """Create enum types if they don't already exist.
    
    This function safely creates enum types by checking for their existence first,
    preventing duplicate key violations when multiple test workers run concurrently.
    
    Args:
        conn: Async SQLAlchemy connection
    """
    # Define all enum types used in the application
    enum_definitions = [
        ("document_status", ["pending", "processing", "completed", "failed", "deleted"]),
        ("permission_type", ["read", "write", "admin"]),
        ("collection_status", ["pending", "ready", "processing", "error", "degraded"]),
        ("operation_type", ["index", "append", "reindex", "remove_source", "delete"]),
        ("operation_status", ["pending", "processing", "completed", "failed", "cancelled"]),
    ]
    
    # Use advisory lock to prevent concurrent enum creation
    # This is a PostgreSQL-specific feature that provides application-level locking
    await conn.execute(text("SELECT pg_advisory_lock(12345)"))
    
    try:
        for enum_name, values in enum_definitions:
            # Check if enum type exists
            result = await conn.execute(
                text("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_type 
                        WHERE typname = :enum_name
                    )
                """),
                {"enum_name": enum_name}
            )
            exists = result.scalar()
            
            if not exists:
                # Create enum type
                values_str = ", ".join(f"'{v}'" for v in values)
                try:
                    await conn.execute(
                        text(f"CREATE TYPE {enum_name} AS ENUM ({values_str})")
                    )
                except Exception as e:
                    # Check if it's a duplicate key error (race condition despite lock)
                    if "duplicate key value violates unique constraint" in str(e) or "already exists" in str(e):
                        # That's fine, the enum was created by another connection
                        continue
                    else:
                        # Re-raise other errors
                        raise
    finally:
        # Always release the advisory lock
        await conn.execute(text("SELECT pg_advisory_unlock(12345)"))


def create_enum_types_if_not_exist_sync(conn: Any) -> None:
    """Synchronous version of create_enum_types_if_not_exist.
    
    Args:
        conn: Sync SQLAlchemy connection
    """
    # Define all enum types used in the application
    enum_definitions = [
        ("document_status", ["pending", "processing", "completed", "failed", "deleted"]),
        ("permission_type", ["read", "write", "admin"]),
        ("collection_status", ["pending", "ready", "processing", "error", "degraded"]),
        ("operation_type", ["index", "append", "reindex", "remove_source", "delete"]),
        ("operation_status", ["pending", "processing", "completed", "failed", "cancelled"]),
    ]
    
    # Use advisory lock to prevent concurrent enum creation
    conn.execute(text("SELECT pg_advisory_lock(12345)"))
    
    try:
        for enum_name, values in enum_definitions:
            # Check if enum type exists
            result = conn.execute(
                text("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_type 
                        WHERE typname = :enum_name
                    )
                """),
                {"enum_name": enum_name}
            )
            exists = result.scalar()
            
            if not exists:
                # Create enum type
                values_str = ", ".join(f"'{v}'" for v in values)
                try:
                    conn.execute(
                        text(f"CREATE TYPE {enum_name} AS ENUM ({values_str})")
                    )
                except Exception as e:
                    # Check if it's a duplicate key error (race condition despite lock)
                    if "duplicate key value violates unique constraint" in str(e) or "already exists" in str(e):
                        # That's fine, the enum was created by another connection
                        continue
                    else:
                        # Re-raise other errors
                        raise
    finally:
        # Always release the advisory lock
        conn.execute(text("SELECT pg_advisory_unlock(12345)"))


async def ensure_enums_exist_with_retry(engine: AsyncEngine, max_retries: int = 3) -> None:
    """Ensure enum types exist with retry logic for concurrent access.
    
    This function handles the race condition when multiple test workers
    try to create enum types at the same time.
    
    Args:
        engine: Async SQLAlchemy engine
        max_retries: Maximum number of retry attempts
    """
    for attempt in range(max_retries):
        try:
            async with engine.begin() as conn:
                await create_enum_types_if_not_exist(conn)
            break  # Success, exit retry loop
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed, re-raise
                raise
            # Wait a bit before retrying
            await asyncio.sleep(0.1 * (attempt + 1))