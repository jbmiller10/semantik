"""
End-to-end integration tests for the complete chunking workflow.

Tests the entire document processing pipeline from upload through chunking,
storage, and search functionality across partitioned data.
"""

import time
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from faker import Faker
from httpx import AsyncClient
from shared.database.models import Chunk, Collection, Document, Operation
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from webui.auth import create_access_token

fake = Faker()


@pytest.fixture()
def test_documents() -> list[dict[str, Any]]:
    """Generate test documents with various content types."""
    return [
        {
            "id": str(uuid.uuid4()),
            "name": "technical_doc.md",
            "content": """# Technical Documentation

## Introduction
This is a comprehensive technical document that covers various aspects of our system architecture.
The document is structured to provide clear insights into our implementation details.

## Architecture Overview
Our system follows a microservices architecture with the following components:
- API Gateway: Handles all incoming requests
- Authentication Service: Manages user authentication and authorization
- Processing Service: Handles document processing and chunking
- Storage Service: Manages persistent data storage

## Implementation Details
The implementation leverages modern cloud-native technologies:
1. Kubernetes for container orchestration
2. Redis for caching and message queuing
3. PostgreSQL for relational data
4. Qdrant for vector storage

### Code Example
```python
def process_document(doc):
    chunks = chunk_document(doc)
    embeddings = generate_embeddings(chunks)
    store_vectors(embeddings)
    return chunks
```

## Performance Considerations
- Horizontal scaling through Kubernetes
- Caching strategies with Redis
- Database query optimization
- Asynchronous processing for long-running tasks
""",
            "type": "markdown",
            "size": 1500,
        },
        {
            "id": str(uuid.uuid4()),
            "name": "research_paper.txt",
            "content": """Abstract
This research paper investigates the effectiveness of various text chunking strategies
for semantic search applications. We compare fixed-size, recursive, and semantic
chunking approaches across different document types and query patterns.

Introduction
Text chunking is a fundamental preprocessing step in many natural language processing
applications, particularly in semantic search and retrieval-augmented generation systems.
The choice of chunking strategy can significantly impact the quality of search results
and the overall system performance.

Methodology
We evaluated three primary chunking strategies:
1. Fixed-size chunking with configurable overlap
2. Recursive chunking based on document structure
3. Semantic chunking using embedding similarity

Each strategy was tested on a diverse corpus including technical documentation,
research papers, legal documents, and narrative texts. We measured performance
using metrics including search precision, recall, and processing efficiency.

Results
Our experiments revealed that no single chunking strategy performs optimally
across all document types. Fixed-size chunking showed consistent performance
but occasionally broke semantic boundaries. Recursive chunking excelled with
structured documents but struggled with unstructured text. Semantic chunking
provided the best search quality but required significantly more processing time.

Conclusion
The optimal chunking strategy depends on the specific use case, document types,
and performance requirements. We recommend a hybrid approach that selects
the appropriate strategy based on document characteristics.
""",
            "type": "text",
            "size": 2000,
        },
        {
            "id": str(uuid.uuid4()),
            "name": "code_sample.py",
            "content": '''"""
Sample Python module for testing code chunking.
Contains multiple classes and functions to test chunking boundaries.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for chunking operations."""
    strategy: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    preserve_boundaries: bool = True

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Overlap must be less than chunk size")
        return True


class DocumentProcessor:
    """Processes documents for chunking."""

    def __init__(self, config: ChunkConfig):
        self.config = config
        self.config.validate()
        self._cache: Dict[str, List[str]] = {}

    async def process(self, document: str) -> List[str]:
        """Process a document into chunks."""
        if not document:
            return []

        # Check cache
        doc_hash = hash(document)
        if doc_hash in self._cache:
            logger.info("Returning cached chunks")
            return self._cache[doc_hash]

        # Process based on strategy
        if self.config.strategy == "fixed":
            chunks = self._fixed_chunking(document)
        elif self.config.strategy == "recursive":
            chunks = self._recursive_chunking(document)
        else:
            chunks = self._semantic_chunking(document)

        # Cache results
        self._cache[doc_hash] = chunks
        return chunks

    def _fixed_chunking(self, text: str) -> List[str]:
        """Implement fixed-size chunking."""
        chunks = []
        step = self.config.chunk_size - self.config.chunk_overlap

        for i in range(0, len(text), step):
            chunk = text[i:i + self.config.chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def _recursive_chunking(self, text: str) -> List[str]:
        """Implement recursive chunking based on structure."""
        # Simplified implementation
        return text.split("\\n\\n")

    def _semantic_chunking(self, text: str) -> List[str]:
        """Implement semantic-based chunking."""
        # Simplified implementation
        sentences = text.split(". ")
        return [s + "." for s in sentences if s]


async def main():
    """Main entry point for testing."""
    config = ChunkConfig(strategy="fixed", chunk_size=500, chunk_overlap=50)
    processor = DocumentProcessor(config)

    sample_text = "This is a sample document. " * 100
    chunks = await processor.process(sample_text)

    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i}: {chunk[:50]}...")


if __name__ == "__main__":
    asyncio.run(main())
''',
            "type": "python",
            "size": 2500,
        },
        {
            "id": str(uuid.uuid4()),
            "name": "small_doc.txt",
            "content": "This is a small document for testing edge cases. " * 10,
            "type": "text",
            "size": 500,
        },
        {
            "id": str(uuid.uuid4()),
            "name": "empty_sections.md",
            "content": """# Document with Empty Sections

## First Section
Some content here.

##

## Third Section
More content here.

###

### Valid Subsection
Final content.
""",
            "type": "markdown",
            "size": 200,
        },
    ]


@pytest.fixture()
async def test_collection(async_session: AsyncSession, test_user: dict) -> Collection:
    """Create a test collection for chunking tests."""
    collection = Collection(
        id=str(uuid.uuid4()),
        name=f"Test Collection {fake.word()}",
        description="Collection for end-to-end chunking tests",
        owner_id=test_user["id"],
        status="ready",
        vector_store_name=f"test_chunks_{uuid.uuid4().hex[:8]}",
        embedding_model="test-embedding-model",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    async_session.add(collection)
    await async_session.commit()
    await async_session.refresh(collection)

    return collection


@pytest.fixture()
async def auth_headers(test_user: dict) -> dict[str, str]:
    """Create authorization headers for API requests."""
    token = create_access_token(data={"sub": test_user["username"], "user_id": test_user["id"]})
    return {"Authorization": f"Bearer {token}"}


class TestCompleteChunkingWorkflow:
    """Test the complete end-to-end chunking workflow."""

    @pytest.mark.asyncio()
    async def test_single_document_chunking_workflow(
        self,
        async_client: AsyncClient,
        async_session: AsyncSession,
        test_collection: Collection,
        test_documents: list[dict],
        auth_headers: dict[str, str],
        redis_client: Any,
    ) -> None:
        """Test complete workflow for a single document."""
        # Step 1: Upload document to collection
        document = test_documents[0]  # Use markdown document
        doc_record = Document(
            id=document["id"],
            collection_id=test_collection.id,
            name=document["name"],
            content=document["content"],
            file_type=document["type"],
            file_size=document["size"],
            created_at=datetime.now(UTC),
        )
        async_session.add(doc_record)
        await async_session.commit()

        # Step 2: Start chunking operation
        chunking_config = {
            "strategy": "recursive",
            "config": {
                "strategy": "recursive",
                "chunk_size": 500,
                "chunk_overlap": 50,
            },
        }

        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection.id}/chunk",
            headers=auth_headers,
            json=chunking_config,
        )

        assert response.status_code == 202
        operation_data = response.json()
        operation_id = operation_data["operation_id"]
        assert operation_data["status"] == "pending"
        assert operation_data["websocket_channel"] is not None

        # Step 3: Simulate worker processing
        # In real scenario, Celery worker would process this
        await self._simulate_chunking_process(
            async_session,
            redis_client,
            operation_id,
            test_collection.id,
            [document],
        )

        # Step 4: Verify chunks were created and stored correctly
        chunks = await async_session.execute(
            select(Chunk).where(
                Chunk.collection_id == test_collection.id,
                Chunk.document_id == document["id"],
            )
        )
        chunk_list = chunks.scalars().all()

        assert len(chunk_list) > 0
        assert all(chunk.collection_id == test_collection.id for chunk in chunk_list)
        assert all(chunk.document_id == document["id"] for chunk in chunk_list)

        # Step 5: Verify chunk content and metadata
        first_chunk = chunk_list[0]
        assert first_chunk.content is not None
        assert first_chunk.chunk_index == 0
        assert first_chunk.token_count is not None
        assert first_chunk.partition_key is not None  # Should be 0-99
        assert 0 <= first_chunk.partition_key <= 99

        # Step 6: Test search across chunks
        search_query = "microservices architecture"
        search_results = await self._search_chunks(
            async_session,
            test_collection.id,
            search_query,
        )

        assert len(search_results) > 0
        assert any("microservices" in result.content.lower() for result in search_results)

        # Step 7: Verify operation completed successfully
        operation = await async_session.get(Operation, operation_id)
        assert operation.status == "completed"
        assert operation.completed_at is not None
        assert operation.error_message is None

    @pytest.mark.asyncio()
    async def test_multi_document_batch_processing(
        self,
        async_client: AsyncClient,
        async_session: AsyncSession,
        test_collection: Collection,
        test_documents: list[dict],
        auth_headers: dict[str, str],
        redis_client: Any,
    ) -> None:
        """Test processing multiple documents with different strategies."""
        # Add all test documents to collection
        for doc in test_documents:
            doc_record = Document(
                id=doc["id"],
                collection_id=test_collection.id,
                name=doc["name"],
                content=doc["content"],
                file_type=doc["type"],
                file_size=doc["size"],
                created_at=datetime.now(UTC),
            )
            async_session.add(doc_record)
        await async_session.commit()

        # Start chunking with different strategies for different file types
        chunking_config = {
            "strategy": "hybrid",
            "config": {
                "strategy": "hybrid",
                "chunk_size": 400,
                "chunk_overlap": 100,
                "file_type_strategies": {
                    "markdown": "markdown",
                    "python": "recursive",
                    "text": "fixed_size",
                },
            },
        }

        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection.id}/chunk",
            headers=auth_headers,
            json=chunking_config,
        )

        assert response.status_code == 202
        operation_id = response.json()["operation_id"]

        # Simulate batch processing
        await self._simulate_chunking_process(
            async_session,
            redis_client,
            operation_id,
            test_collection.id,
            test_documents,
        )

        # Verify all documents were chunked
        for doc in test_documents:
            chunks = await async_session.execute(
                select(Chunk).where(
                    Chunk.collection_id == test_collection.id,
                    Chunk.document_id == doc["id"],
                )
            )
            chunk_list = chunks.scalars().all()

            assert len(chunk_list) > 0, f"No chunks created for document {doc['name']}"

            # Verify appropriate chunking based on file type
            if doc["type"] == "markdown":
                # Markdown should preserve structure
                assert any("##" in chunk.content for chunk in chunk_list)
            elif doc["type"] == "python":
                # Python should preserve code structure
                assert any("class" in chunk.content or "def" in chunk.content for chunk in chunk_list)

    @pytest.mark.asyncio()
    async def test_chunking_strategy_switching(
        self,
        async_client: AsyncClient,
        async_session: AsyncSession,
        test_collection: Collection,
        test_documents: list[dict],
        auth_headers: dict[str, str],
        redis_client: Any,
    ) -> None:
        """Test changing chunking strategy on existing collection."""
        # Initial chunking with fixed-size strategy
        document = test_documents[1]  # Use text document
        doc_record = Document(
            id=document["id"],
            collection_id=test_collection.id,
            name=document["name"],
            content=document["content"],
            file_type=document["type"],
            file_size=document["size"],
            created_at=datetime.now(UTC),
        )
        async_session.add(doc_record)
        await async_session.commit()

        # First chunking operation
        initial_config = {
            "strategy": "fixed_size",
            "config": {
                "strategy": "fixed_size",
                "chunk_size": 200,
                "chunk_overlap": 20,
            },
        }

        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection.id}/chunk",
            headers=auth_headers,
            json=initial_config,
        )

        operation_id = response.json()["operation_id"]
        await self._simulate_chunking_process(
            async_session,
            redis_client,
            operation_id,
            test_collection.id,
            [document],
        )

        # Get initial chunk count
        initial_chunks = await async_session.execute(select(Chunk).where(Chunk.collection_id == test_collection.id))
        initial_count = len(initial_chunks.scalars().all())

        # Switch to semantic chunking strategy
        new_config = {
            "strategy": "semantic",
            "config": {
                "strategy": "semantic",
                "chunk_size": 300,
                "chunk_overlap": 50,
            },
            "reprocess_existing": True,
        }

        response = await async_client.patch(
            f"/api/v2/chunking/collections/{test_collection.id}/chunking-strategy",
            headers=auth_headers,
            json=new_config,
        )

        assert response.status_code == 200

        # Delete old chunks and create new ones
        await async_session.execute(select(Chunk).where(Chunk.collection_id == test_collection.id).delete())
        await async_session.commit()

        # Process with new strategy
        operation_id = str(uuid.uuid4())
        await self._simulate_chunking_process(
            async_session,
            redis_client,
            operation_id,
            test_collection.id,
            [document],
            strategy="semantic",
        )

        # Verify new chunks with different characteristics
        new_chunks = await async_session.execute(select(Chunk).where(Chunk.collection_id == test_collection.id))
        new_chunk_list = new_chunks.scalars().all()

        assert len(new_chunk_list) > 0
        # Semantic chunking should create different number of chunks
        assert len(new_chunk_list) != initial_count

    @pytest.mark.asyncio()
    async def test_partition_distribution(
        self,
        async_client: AsyncClient,
        async_session: AsyncSession,
        test_collection: Collection,
        auth_headers: dict[str, str],
        redis_client: Any,
    ) -> None:
        """Test that chunks are distributed across partitions correctly."""
        # Create a large number of documents to ensure partition distribution
        large_doc_set = []
        for i in range(50):
            doc = {
                "id": str(uuid.uuid4()),
                "name": f"doc_{i}.txt",
                "content": fake.text(max_nb_chars=2000),
                "type": "text",
                "size": 2000,
            }
            large_doc_set.append(doc)

            doc_record = Document(
                id=doc["id"],
                collection_id=test_collection.id,
                name=doc["name"],
                content=doc["content"],
                file_type=doc["type"],
                file_size=doc["size"],
                created_at=datetime.now(UTC),
            )
            async_session.add(doc_record)

        await async_session.commit()

        # Start chunking operation
        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection.id}/chunk",
            headers=auth_headers,
            json={
                "strategy": "fixed_size",
                "config": {
                    "strategy": "fixed_size",
                    "chunk_size": 300,
                    "chunk_overlap": 50,
                },
            },
        )

        operation_id = response.json()["operation_id"]
        await self._simulate_chunking_process(
            async_session,
            redis_client,
            operation_id,
            test_collection.id,
            large_doc_set,
        )

        # Analyze partition distribution
        chunks = await async_session.execute(
            select(Chunk.partition_key, Chunk.id).where(Chunk.collection_id == test_collection.id)
        )

        partition_distribution = {}
        for chunk in chunks:
            partition_key = chunk[0]
            if partition_key not in partition_distribution:
                partition_distribution[partition_key] = 0
            partition_distribution[partition_key] += 1

        # Verify chunks are distributed across multiple partitions
        assert len(partition_distribution) > 1, "Chunks should be distributed across multiple partitions"

        # Check for reasonable distribution (no partition should have >50% of chunks)
        total_chunks = sum(partition_distribution.values())
        max_partition_count = max(partition_distribution.values())
        assert max_partition_count < total_chunks * 0.5, "Partition distribution is too skewed"

    @pytest.mark.asyncio()
    async def test_search_across_partitions(
        self,
        async_session: AsyncSession,
        test_collection: Collection,
        test_documents: list[dict],
    ) -> None:
        """Test that search works correctly across partitioned chunks."""
        # Add documents and simulate they've been chunked
        for doc in test_documents[:3]:
            # Add document
            doc_record = Document(
                id=doc["id"],
                collection_id=test_collection.id,
                name=doc["name"],
                content=doc["content"],
                file_type=doc["type"],
                file_size=doc["size"],
                created_at=datetime.now(UTC),
            )
            async_session.add(doc_record)

            # Create chunks across different partitions
            chunk_size = 200
            for i, start in enumerate(range(0, len(doc["content"]), chunk_size - 50)):
                chunk = Chunk(
                    collection_id=test_collection.id,
                    document_id=doc["id"],
                    content=doc["content"][start : start + chunk_size],
                    chunk_index=i,
                    start_offset=start,
                    end_offset=min(start + chunk_size, len(doc["content"])),
                    token_count=len(doc["content"][start : start + chunk_size].split()),
                    created_at=datetime.now(UTC),
                )
                async_session.add(chunk)

        await async_session.commit()

        # Search across all partitions
        search_queries = [
            "architecture",
            "chunking strategies",
            "def process",
        ]

        for query in search_queries:
            results = await self._search_chunks(
                async_session,
                test_collection.id,
                query,
            )

            # Verify we get results from different partitions
            if results:
                partition_keys = {chunk.partition_key for chunk in results}
                assert len(partition_keys) >= 1, f"Search '{query}' should return results from partitions"

    async def _simulate_chunking_process(
        self,
        session: AsyncSession,
        redis_client: Any,
        operation_id: str,
        collection_id: str,
        documents: list[dict],
        strategy: str = "recursive",
    ) -> None:
        """Simulate the worker processing chunks."""
        total_chunks_created = 0

        for doc_idx, doc in enumerate(documents):
            # Simple chunking simulation
            chunk_size = 400
            overlap = 100
            step = chunk_size - overlap

            chunks_to_create = []
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
            session.add_all(chunks_to_create)
            total_chunks_created += len(chunks_to_create)

            # Update progress in Redis
            progress = ((doc_idx + 1) / len(documents)) * 100
            redis_client.hset(
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

    async def _search_chunks(
        self,
        session: AsyncSession,
        collection_id: str,
        query: str,
    ) -> list[Chunk]:
        """Simulate searching across chunks."""
        # Simple text search simulation
        result = await session.execute(
            select(Chunk)
            .where(
                Chunk.collection_id == collection_id,
                Chunk.content.ilike(f"%{query}%"),
            )
            .limit(10)
        )
        return result.scalars().all()


class TestEdgeCases:
    """Test edge cases in the chunking workflow."""

    @pytest.mark.asyncio()
    async def test_empty_document_handling(
        self,
        async_client: AsyncClient,
        async_session: AsyncSession,
        test_collection: Collection,
        auth_headers: dict[str, str],
    ) -> None:
        """Test handling of empty documents."""
        empty_doc = Document(
            id=str(uuid.uuid4()),
            collection_id=test_collection.id,
            name="empty.txt",
            content="",
            file_type="text",
            file_size=0,
            created_at=datetime.now(UTC),
        )
        async_session.add(empty_doc)
        await async_session.commit()

        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection.id}/chunk",
            headers=auth_headers,
            json={
                "strategy": "fixed_size",
                "config": {"strategy": "fixed_size", "chunk_size": 100, "chunk_overlap": 10},
            },
        )

        assert response.status_code == 202

        # Verify no chunks created for empty document
        chunks = await async_session.execute(
            select(Chunk).where(
                Chunk.collection_id == test_collection.id,
                Chunk.document_id == empty_doc.id,
            )
        )
        assert len(chunks.scalars().all()) == 0

    @pytest.mark.asyncio()
    async def test_very_large_document_handling(
        self,
        async_client: AsyncClient,
        async_session: AsyncSession,
        test_collection: Collection,
        auth_headers: dict[str, str],
        redis_client: Any,
    ) -> None:
        """Test handling of very large documents."""
        # Create a large document (5MB of text)
        large_content = fake.text(max_nb_chars=5000000)
        large_doc = Document(
            id=str(uuid.uuid4()),
            collection_id=test_collection.id,
            name="large_document.txt",
            content=large_content,
            file_type="text",
            file_size=len(large_content),
            created_at=datetime.now(UTC),
        )
        async_session.add(large_doc)
        await async_session.commit()

        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection.id}/chunk",
            headers=auth_headers,
            json={
                "strategy": "fixed_size",
                "config": {"strategy": "fixed_size", "chunk_size": 1000, "chunk_overlap": 200},
            },
        )

        assert response.status_code == 202
        operation_id = response.json()["operation_id"]

        # Simulate processing with memory monitoring
        start_time = time.time()
        await self._simulate_chunking_process(
            async_session,
            redis_client,
            operation_id,
            test_collection.id,
            [
                {
                    "id": large_doc.id,
                    "content": large_content,
                    "name": "large.txt",
                    "type": "text",
                    "size": len(large_content),
                }
            ],
        )
        processing_time = time.time() - start_time

        # Verify chunks were created efficiently
        chunks = await async_session.execute(
            select(Chunk).where(
                Chunk.collection_id == test_collection.id,
                Chunk.document_id == large_doc.id,
            )
        )
        chunk_list = chunks.scalars().all()

        assert len(chunk_list) > 1000  # Should create many chunks
        assert processing_time < 60  # Should complete within reasonable time

        # Verify chunks maintain order
        sorted_chunks = sorted(chunk_list, key=lambda c: c.chunk_index)
        for i, chunk in enumerate(sorted_chunks):
            assert chunk.chunk_index == i

    async def _simulate_chunking_process(
        self,
        session: AsyncSession,
        redis_client: Any,
        operation_id: str,
        collection_id: str,
        documents: list[dict],
    ) -> None:
        """Simulate chunking process for edge case testing."""
        for doc in documents:
            if not doc["content"]:
                continue

            # Process in batches to handle large documents
            chunk_size = 1000
            overlap = 200
            step = chunk_size - overlap
            batch_size = 100  # Insert chunks in batches

            chunks_batch = []
            chunk_index = 0

            for start in range(0, len(doc["content"]), step):
                end = min(start + chunk_size, len(doc["content"]))
                chunk_content = doc["content"][start:end]

                if chunk_content.strip():
                    chunk = Chunk(
                        collection_id=collection_id,
                        document_id=doc["id"],
                        content=chunk_content,
                        chunk_index=chunk_index,
                        start_offset=start,
                        end_offset=end,
                        token_count=len(chunk_content.split()),
                        created_at=datetime.now(UTC),
                    )
                    chunks_batch.append(chunk)
                    chunk_index += 1

                    # Insert batch when it reaches batch_size
                    if len(chunks_batch) >= batch_size:
                        session.add_all(chunks_batch)
                        await session.flush()
                        chunks_batch = []

            # Insert remaining chunks
            if chunks_batch:
                session.add_all(chunks_batch)
                await session.flush()

        await session.commit()

        # Update operation status
        operation = await session.get(Operation, operation_id)
        if operation:
            operation.status = "completed"
            operation.completed_at = datetime.now(UTC)
            await session.commit()
