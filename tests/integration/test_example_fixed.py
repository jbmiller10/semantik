"""
Example integration test demonstrating the fixed infrastructure.

This test shows how to properly use the fixtures and helpers
for efficient and isolated testing.
"""

import pytest
import pytest_asyncio
from sqlalchemy import select

from shared.database.models import Chunk, Collection, Document
from test_helpers import ChunkingTestHelper, TestDataGenerator


class TestFixedExample:
    """Example tests with proper fixture usage and optimization."""
    
    @pytest.mark.integration
    async def test_basic_chunking_with_fixtures(
        self,
        async_session,
        test_user,
        redis_client,
        test_data_factory,
    ):
        """Test basic chunking with proper fixtures."""
        # Create a test collection
        collection = Collection(
            id="test-collection-1",
            name="Test Collection",
            owner_id=test_user["id"],
            status="ready",
        )
        async_session.add(collection)
        await async_session.commit()
        
        # Generate small test documents (optimized for speed)
        documents = TestDataGenerator.generate_documents(
            count=3,  # Small number for fast tests
            size="small",  # 500 bytes each
        )
        
        # Add documents to database
        for doc_data in documents:
            doc = Document(
                id=doc_data["id"],
                collection_id=collection.id,
                name=doc_data["name"],
                content=doc_data["content"],
                file_type=doc_data["type"],
                file_size=doc_data["size"],
            )
            async_session.add(doc)
        
        await async_session.commit()
        
        # Simulate chunking process using helper
        operation_id = "test-operation-1"
        chunks_created = await ChunkingTestHelper.simulate_chunking_process(
            session=async_session,
            redis_client=redis_client,
            operation_id=operation_id,
            collection_id=collection.id,
            documents=documents,
            chunk_size=200,  # Small chunks for testing
            overlap=20,
        )
        
        # Verify chunks were created
        assert chunks_created > 0
        
        # Query chunks to verify
        result = await async_session.execute(
            select(Chunk).where(Chunk.collection_id == collection.id)
        )
        chunks = result.scalars().all()
        
        assert len(chunks) == chunks_created
        
        # Verify Redis progress was tracked
        progress = await redis_client.hget(f"operation:{operation_id}", "progress")
        assert progress == "100.0"
        
        # Cleanup happens automatically via fixtures
    
    @pytest.mark.integration
    async def test_edge_cases_with_special_documents(
        self,
        async_session,
        test_user,
        redis_client,
    ):
        """Test edge cases with special documents."""
        # Create collection
        collection = Collection(
            id="test-collection-2",
            name="Edge Case Collection",
            owner_id=test_user["id"],
            status="ready",
        )
        async_session.add(collection)
        await async_session.commit()
        
        # Generate special case documents
        special_docs = TestDataGenerator.generate_special_documents()
        
        # Add to database
        for doc_data in special_docs:
            doc = Document(
                id=doc_data["id"],
                collection_id=collection.id,
                name=doc_data["name"],
                content=doc_data["content"] or "",  # Handle None
                file_type=doc_data["type"],
                file_size=doc_data["size"],
            )
            async_session.add(doc)
        
        await async_session.commit()
        
        # Process with chunking
        operation_id = "test-operation-2"
        chunks_created = await ChunkingTestHelper.simulate_chunking_process(
            session=async_session,
            redis_client=redis_client,
            operation_id=operation_id,
            collection_id=collection.id,
            documents=special_docs,
        )
        
        # Empty documents should not create chunks
        empty_doc = next(d for d in special_docs if d["name"] == "empty.txt")
        result = await async_session.execute(
            select(Chunk).where(
                Chunk.collection_id == collection.id,
                Chunk.document_id == empty_doc["id"],
            )
        )
        empty_chunks = result.scalars().all()
        assert len(empty_chunks) == 0
        
        # Unicode document should create chunks
        unicode_doc = next(d for d in special_docs if d["name"] == "unicode.txt")
        result = await async_session.execute(
            select(Chunk).where(
                Chunk.collection_id == collection.id,
                Chunk.document_id == unicode_doc["id"],
            )
        )
        unicode_chunks = result.scalars().all()
        assert len(unicode_chunks) > 0
    
    @pytest.mark.slow
    @pytest.mark.integration
    async def test_large_dataset_chunking(
        self,
        async_session,
        test_user,
        redis_client,
    ):
        """Test with larger dataset (marked as slow)."""
        # This test only runs when --slow flag is used
        collection = Collection(
            id="test-collection-large",
            name="Large Dataset Collection",
            owner_id=test_user["id"],
            status="ready",
        )
        async_session.add(collection)
        await async_session.commit()
        
        # Generate more documents for comprehensive testing
        documents = TestDataGenerator.generate_documents(
            count=20,  # More documents
            size="medium",  # 5KB each
            varied=True,  # Mix of sizes
        )
        
        # Add all documents
        for doc_data in documents:
            doc = Document(
                id=doc_data["id"],
                collection_id=collection.id,
                name=doc_data["name"],
                content=doc_data["content"],
                file_type=doc_data["type"],
                file_size=doc_data["size"],
            )
            async_session.add(doc)
        
        await async_session.commit()
        
        # Process chunking
        operation_id = "test-operation-large"
        chunks_created = await ChunkingTestHelper.simulate_chunking_process(
            session=async_session,
            redis_client=redis_client,
            operation_id=operation_id,
            collection_id=collection.id,
            documents=documents,
        )
        
        # Should create many chunks
        assert chunks_created > 50
        
        # Test search functionality
        search_results = await ChunkingTestHelper.search_chunks(
            session=async_session,
            collection_id=collection.id,
            query="the",
            limit=5,
        )
        
        assert len(search_results) <= 5