#!/usr/bin/env python3
"""
Example demonstrating proper usage of partition-aware chunk operations.

This file shows best practices for working with the partitioned chunks table,
ensuring optimal performance through partition pruning and efficient bulk operations.
"""

import asyncio
import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.repositories import ChunkRepository


class ChunkingServiceExample:
    """Example service demonstrating partition-aware chunk operations."""

    def __init__(self, db_session: AsyncSession):
        """Initialize with database session."""
        self.db_session = db_session
        self.chunk_repo = ChunkRepository(db_session)

    async def process_document_chunks(
        self, collection_id: str, document_id: str, text_content: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> list[str]:
        """Process a document into chunks with partition optimization.

        This example shows:
        1. Efficient bulk insertion grouped by collection_id
        2. Proper use of partition key in all operations
        3. Transaction management for consistency
        """
        # Simple chunking logic (in reality, use proper text splitters)
        chunks_data = []
        chunk_ids = []

        for i in range(0, len(text_content), chunk_size - chunk_overlap):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)

            chunks_data.append(
                {
                    "id": chunk_id,
                    "collection_id": collection_id,  # CRITICAL: Always include partition key
                    "document_id": document_id,
                    "chunk_index": i // (chunk_size - chunk_overlap),
                    "content": text_content[i : i + chunk_size],
                    "start_offset": i,
                    "end_offset": min(i + chunk_size, len(text_content)),
                    "chunking_config_id": 1,  # Example config ID
                    "created_at": datetime.now(UTC),
                }
            )

        # Bulk insert - automatically grouped by collection_id
        await self.chunk_repo.create_chunks_bulk(chunks_data)

        print(f"Created {len(chunks_data)} chunks for document {document_id}")
        return chunk_ids

    async def update_chunk_embeddings_example(
        self, collection_id: str, chunk_ids: list[str], embeddings: list[str]
    ) -> None:
        """Update embeddings for chunks with partition awareness.

        Shows how to structure updates to include partition key.
        """
        # Structure updates with partition key
        chunk_updates = [
            {
                "id": chunk_id,
                "collection_id": collection_id,  # Required for partition routing
                "embedding_vector_id": embedding_id,
            }
            for chunk_id, embedding_id in zip(chunk_ids, embeddings, strict=False)
        ]

        updated_count = await self.chunk_repo.update_chunk_embeddings(chunk_updates)
        print(f"Updated embeddings for {updated_count} chunks")

    async def search_chunks_example(self, collection_id: str, document_id: str | None = None) -> None:
        """Demonstrate efficient chunk queries with partition pruning."""

        # GOOD: Query with collection_id for partition pruning
        if document_id:
            chunks = await self.chunk_repo.get_chunks_by_document(
                document_id=document_id,
                collection_id=collection_id,
                limit=10,  # Always include!
            )
            print(f"Found {len(chunks)} chunks for document {document_id}")
        else:
            # Get recent chunks from collection
            chunks = await self.chunk_repo.get_chunks_by_collection(
                collection_id=collection_id,
                limit=10,
                created_after=datetime.now(UTC).replace(hour=0, minute=0, second=0),
            )
            print(f"Found {len(chunks)} recent chunks in collection")

        # Get statistics for the partition
        stats = await self.chunk_repo.get_chunk_statistics(collection_id)
        print(f"Collection partition stats: {stats}")

    async def cleanup_chunks_example(self, collection_id: str, document_id: str) -> None:
        """Delete chunks with partition-aware operations."""

        # Delete by document - efficient because it includes collection_id
        deleted_count = await self.chunk_repo.delete_chunks_by_document(
            document_id=document_id, collection_id=collection_id
        )
        print(f"Deleted {deleted_count} chunks for document {document_id}")

    async def cross_collection_query_example(self) -> None:
        """Example of what NOT to do - cross-partition queries.

        This demonstrates inefficient patterns to avoid.
        """
        # BAD: This would scan ALL partitions (avoid if possible)
        # chunks = await self.db_session.execute(
        #     select(Chunk).where(Chunk.content.contains("search term"))
        # )

        # GOOD: If you must search across collections, do it collection by collection
        collection_ids = ["coll1", "coll2", "coll3"]
        for collection_id in collection_ids:
            # Each query hits only one partition
            chunks = await self.chunk_repo.get_chunks_by_collection(collection_id=collection_id, limit=100)
            # Process chunks...
            _ = chunks  # In a real implementation, you would process the chunks here


async def main() -> None:
    """Run the examples."""
    # This is just for demonstration - in real code, get session from dependency injection
    from packages.shared.database import get_db

    async for session in get_db():
        service = ChunkingServiceExample(session)

        # Example collection and document IDs
        collection_id = str(uuid.uuid4())
        document_id = str(uuid.uuid4())

        # 1. Create chunks
        chunk_ids = await service.process_document_chunks(
            collection_id=collection_id,
            document_id=document_id,
            text_content="This is a long document text that needs to be chunked..." * 100,
        )

        # 2. Update embeddings
        fake_embeddings = [f"vector_{i}" for i in range(len(chunk_ids))]
        await service.update_chunk_embeddings_example(
            collection_id=collection_id, chunk_ids=chunk_ids, embeddings=fake_embeddings
        )

        # 3. Query chunks
        await service.search_chunks_example(collection_id=collection_id, document_id=document_id)

        # 4. Cleanup
        await service.cleanup_chunks_example(collection_id=collection_id, document_id=document_id)

        # Commit the transaction
        await session.commit()
        break  # We only need one session


# Key Takeaways:
# 1. ALWAYS include collection_id in chunk queries for partition pruning
# 2. Group bulk operations by collection_id for efficiency
# 3. Avoid cross-partition queries when possible
# 4. Use the ChunkRepository methods which handle partition optimization
# 5. Design your application logic to work within partition boundaries

if __name__ == "__main__":
    asyncio.run(main())
