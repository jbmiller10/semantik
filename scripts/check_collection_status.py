#!/usr/bin/env python3
"""
Diagnostic script to check collection and operation status in Semantik.
This helps identify why embeddings might not be generated.
"""

import asyncio
import sys
from datetime import UTC, datetime

# Add the packages directory to the Python path
sys.path.insert(0, "/home/dockertest/semantik/packages")

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from shared.database import pg_connection_manager
from shared.database.database import AsyncSessionLocal
from shared.database.models import Collection, Document, DocumentStatus, Operation, OperationStatus


async def check_collections() -> None:
    """Check all collections and their status."""
    await pg_connection_manager.initialize()

    async with AsyncSessionLocal() as db:
        # Get all collections with their operations
        stmt = select(Collection).options(selectinload(Collection.operations))
        result = await db.execute(stmt)
        collections = result.scalars().all()

        if not collections:
            print("No collections found in the database.")
            return

        print(f"\n{'='*80}")
        print("COLLECTIONS STATUS")
        print(f"{'='*80}\n")

        for collection in collections:
            print(f"Collection: {collection.name}")
            print(f"  ID: {collection.id}")
            print(f"  Status: {collection.status.value if collection.status else 'None'}")
            print(f"  Status Message: {collection.status_message or 'None'}")
            print(f"  Vector Store Name: {collection.vector_store_name}")
            print(f"  Document Count: {collection.document_count}")
            print(f"  Vector Count: {collection.vector_count}")
            print(f"  Created: {collection.created_at}")
            print(f"  Updated: {collection.updated_at}")

            # Check operations for this collection
            operations = sorted(collection.operations, key=lambda x: x.created_at, reverse=True)
            if operations:
                print("\n  Recent Operations:")
                for op in operations[:5]:  # Show last 5 operations
                    print(f"    - Type: {op.type.value}, Status: {op.status.value}")
                    print(f"      ID: {op.uuid}")
                    print(f"      Created: {op.created_at}")
                    if op.started_at:
                        print(f"      Started: {op.started_at}")
                    if op.completed_at:
                        print(f"      Completed: {op.completed_at}")
                    if op.error_message:
                        print(f"      ERROR: {op.error_message}")
                    if op.task_id:
                        print(f"      Task ID: {op.task_id}")
            print()


async def check_failed_operations() -> None:
    """Check for any failed operations."""
    async with AsyncSessionLocal() as db:
        stmt = select(Operation).where(Operation.status == OperationStatus.FAILED).order_by(Operation.created_at.desc())
        result = await db.execute(stmt)
        failed_ops = result.scalars().all()

        if failed_ops:
            print(f"\n{'='*80}")
            print("FAILED OPERATIONS")
            print(f"{'='*80}\n")

            for op in failed_ops:
                print(f"Operation ID: {op.uuid}")
                print(f"  Collection ID: {op.collection_id}")
                print(f"  Type: {op.type.value}")
                print(f"  Created: {op.created_at}")
                print(f"  Error: {op.error_message}")
                print(f"  Config: {op.config}")
                print()


async def check_document_status(collection_id: str | None = None) -> None:
    """Check document processing status."""
    async with AsyncSessionLocal() as db:
        # Build query
        stmt = select(Document.collection_id, Document.status, func.count(Document.id).label("count")).group_by(
            Document.collection_id, Document.status
        )

        if collection_id:
            stmt = stmt.where(Document.collection_id == collection_id)

        result = await db.execute(stmt)
        doc_stats = result.all()

        if doc_stats:
            print(f"\n{'='*80}")
            print("DOCUMENT STATUS SUMMARY")
            print(f"{'='*80}\n")

            # Group by collection
            collection_stats: dict[str, dict[str, int]] = {}
            for row in doc_stats:
                coll_id = row.collection_id
                status = row.status
                count = row.count

                if coll_id not in collection_stats:
                    collection_stats[coll_id] = {}

                collection_stats[coll_id][status.value if status else "None"] = count

            for coll_id, stats in collection_stats.items():
                # Get collection name
                coll_stmt = select(Collection.name).where(Collection.id == coll_id)
                coll_result = await db.execute(coll_stmt)
                coll_name = coll_result.scalar_one_or_none() or "Unknown"

                print(f"Collection: {coll_name} ({coll_id})")
                for status, count in stats.items():
                    print(f"  {status}: {count}")
                print()


async def check_recent_errors() -> None:
    """Check for recent document processing errors."""
    async with AsyncSessionLocal() as db:
        stmt = (
            select(Document)
            .where(Document.status == DocumentStatus.FAILED)
            .where(Document.error_message.isnot(None))
            .order_by(Document.updated_at.desc())
            .limit(10)
        )
        result = await db.execute(stmt)
        failed_docs = result.scalars().all()

        if failed_docs:
            print(f"\n{'='*80}")
            print("RECENT DOCUMENT FAILURES")
            print(f"{'='*80}\n")

            for doc in failed_docs:
                print(f"Document: {doc.file_path}")
                print(f"  Collection ID: {doc.collection_id}")
                print(f"  Error: {doc.error_message}")
                print(f"  Updated: {doc.updated_at}")
                print()


async def check_pending_operations() -> None:
    """Check for operations stuck in pending or processing state."""
    async with AsyncSessionLocal() as db:
        stmt = (
            select(Operation)
            .where(Operation.status.in_([OperationStatus.PENDING, OperationStatus.PROCESSING]))
            .order_by(Operation.created_at.desc())
        )
        result = await db.execute(stmt)
        pending_ops = result.scalars().all()

        if pending_ops:
            print(f"\n{'='*80}")
            print("PENDING/PROCESSING OPERATIONS")
            print(f"{'='*80}\n")

            for op in pending_ops:
                print(f"Operation ID: {op.uuid}")
                print(f"  Collection ID: {op.collection_id}")
                print(f"  Type: {op.type.value}")
                print(f"  Status: {op.status.value}")
                print(f"  Created: {op.created_at}")
                if op.started_at:
                    print(f"  Started: {op.started_at}")
                    duration = datetime.now(tz=UTC) - op.started_at.replace(tzinfo=None)
                    print(f"  Running for: {duration}")
                print(f"  Task ID: {op.task_id or 'None'}")
                print(f"  Config: {op.config}")
                print()


async def main() -> None:
    """Run all diagnostic checks."""
    print("Semantik Collection & Operation Status Check")
    print("=" * 80)

    try:
        await check_collections()
        await check_failed_operations()
        await check_document_status()
        await check_recent_errors()
        await check_pending_operations()

        print("\nDiagnostic check complete.")

    except Exception as e:
        print(f"Error during diagnostic check: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await pg_connection_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
