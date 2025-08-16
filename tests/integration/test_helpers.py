"""
Helper functions for integration tests to avoid code duplication.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Dict, List

from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.models import Chunk, Document, Operation

fake = Faker()


class ChunkingTestHelper:
    """Helper for chunking-related test operations."""
    
    @staticmethod
    async def simulate_chunking_process(
        session: AsyncSession,
        redis_client: Any,
        operation_id: str,
        collection_id: str,
        documents: List[Dict[str, Any]],
        strategy: str = "recursive",
        chunk_size: int = 400,
        overlap: int = 100,
    ) -> int:
        """
        Simulate the worker processing chunks.
        
        Returns:
            Number of chunks created
        """
        total_chunks_created = 0
        step = chunk_size - overlap
        
        for doc_idx, doc in enumerate(documents):
            # Skip empty documents
            if not doc.get("content"):
                continue
            
            chunks_to_create = []
            
            # Create chunks for document
            for i, start in enumerate(range(0, len(doc["content"]), step)):
                end = min(start + chunk_size, len(doc["content"]))
                chunk_content = doc["content"][start:end]
                
                if chunk_content.strip():
                    chunk = Chunk(
                        collection_id=collection_id,
                        document_id=doc["id"],
                        content=chunk_content,
                        chunk_index=i,
                        start_offset=start,
                        end_offset=end,
                        token_count=len(chunk_content.split()),
                        created_at=datetime.now(UTC),
                    )
                    chunks_to_create.append(chunk)
            
            # Batch insert chunks
            if chunks_to_create:
                session.add_all(chunks_to_create)
                total_chunks_created += len(chunks_to_create)
            
            # Update progress in Redis
            if redis_client:
                progress = ((doc_idx + 1) / len(documents)) * 100
                await redis_client.hset(
                    f"operation:{operation_id}",
                    mapping={
                        "status": "processing",
                        "progress": str(progress),
                        "chunks_created": str(total_chunks_created),
                        "documents_processed": str(doc_idx + 1),
                    },
                )
        
        await session.commit()
        
        # Mark operation as completed
        operation = await session.get(Operation, operation_id)
        if operation:
            operation.status = "completed"
            operation.completed_at = datetime.now(UTC)
            operation.progress_percentage = 100.0
            await session.commit()
        
        return total_chunks_created
    
    @staticmethod
    async def create_test_documents(
        session: AsyncSession,
        collection_id: str,
        count: int = 5,
        content_size: int = 1000,
    ) -> List[Document]:
        """Create test documents efficiently."""
        documents = []
        
        for i in range(count):
            doc = Document(
                id=str(uuid.uuid4()),
                collection_id=collection_id,
                name=f"test_doc_{i}.txt",
                content=fake.text(max_nb_chars=content_size),
                file_type="text",
                file_size=content_size,
                created_at=datetime.now(UTC),
            )
            documents.append(doc)
            session.add(doc)
        
        await session.commit()
        return documents
    
    @staticmethod
    async def search_chunks(
        session: AsyncSession,
        collection_id: str,
        query: str,
        limit: int = 10,
    ) -> List[Chunk]:
        """Simulate searching across chunks."""
        from sqlalchemy import select
        
        result = await session.execute(
            select(Chunk).where(
                Chunk.collection_id == collection_id,
                Chunk.content.ilike(f"%{query}%"),
            ).limit(limit)
        )
        return result.scalars().all()


class TestDataGenerator:
    """Generate test data efficiently with configurable sizes."""
    
    SIZES = {
        "tiny": 100,      # 100 bytes
        "small": 500,     # 500 bytes
        "medium": 5000,   # 5KB
        "large": 50000,   # 50KB
        "huge": 500000,   # 500KB
    }
    
    @classmethod
    def generate_documents(
        cls,
        count: int = 10,
        size: str = "small",
        varied: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate test documents.
        
        Args:
            count: Number of documents to generate
            size: Size category (tiny, small, medium, large, huge)
            varied: If True, generate documents of varied sizes
        
        Returns:
            List of document dictionaries
        """
        documents = []
        
        for i in range(count):
            if varied:
                # Rotate through sizes
                size_key = list(cls.SIZES.keys())[i % len(cls.SIZES)]
                max_chars = cls.SIZES[size_key]
            else:
                max_chars = cls.SIZES.get(size, cls.SIZES["small"])
            
            content = fake.text(max_nb_chars=max_chars)
            
            documents.append({
                "id": str(uuid.uuid4()),
                "name": f"doc_{i}_{size}.txt",
                "content": content,
                "type": "text",
                "size": len(content),
            })
        
        return documents
    
    @staticmethod
    def generate_special_documents() -> List[Dict[str, Any]]:
        """Generate documents with special cases for edge testing."""
        return [
            {
                "id": str(uuid.uuid4()),
                "name": "empty.txt",
                "content": "",
                "type": "text",
                "size": 0,
            },
            {
                "id": str(uuid.uuid4()),
                "name": "whitespace.txt",
                "content": "   \n\n\t\t   ",
                "type": "text",
                "size": 10,
            },
            {
                "id": str(uuid.uuid4()),
                "name": "unicode.txt",
                "content": "Unicode: ñáéíóú 中文 日本語 한국어 🚀 ✨",
                "type": "text",
                "size": 50,
            },
            {
                "id": str(uuid.uuid4()),
                "name": "special_chars.txt",
                "content": "Special: \x00\x01\x02 <script>alert('xss')</script>",
                "type": "text",
                "size": 60,
            },
        ]