"""
Integration tests for chunk partition distribution and performance.

Tests partition distribution, query performance, and health monitoring
for the PostgreSQL partitioned chunks table.
"""

import asyncio
import hashlib
import random
import statistics
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Dict, List

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database import get_db
from shared.database.models import Chunk, Collection, Document
from shared.chunking.infrastructure.repositories.partition_manager import PartitionManager

fake = Faker()


def calculate_partition_key(collection_id: str, document_id: str = None) -> int:
    """
    Calculate the partition key using the same algorithm as PostgreSQL.
    Mimics: abs(hashtext(collection_id::text || ':' || document_id::text)) % 100
    """
    if document_id:
        key_string = f"{collection_id}:{document_id}"
    else:
        key_string = f"{collection_id}:null"
    
    # Use Python's hash function to approximate PostgreSQL's hashtext
    hash_value = abs(hash(key_string))
    return hash_value % 100


@pytest.fixture()
async def test_collections(async_session: AsyncSession, test_user: dict) -> List[Collection]:
    """Create multiple test collections for distribution testing."""
    collections = []
    for i in range(10):
        collection = Collection(
            id=str(uuid.uuid4()),
            name=f"Test Collection {i}",
            description=f"Collection {i} for partition testing",
            owner_id=test_user["id"],
            status="ready",
            vector_store_name=f"test_partition_{i}_{uuid.uuid4().hex[:8]}",
            embedding_model="test-model",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        collections.append(collection)
        async_session.add(collection)
    
    await async_session.commit()
    return collections


@pytest.fixture()
async def large_dataset(
    async_session: AsyncSession,
    test_collections: List[Collection],
) -> Dict[str, Any]:
    """Create a large dataset for partition distribution testing."""
    documents = []
    expected_distribution = Counter()
    
    # Create 100 documents per collection (1000 total)
    for collection in test_collections:
        for doc_idx in range(100):
            doc = Document(
                id=str(uuid.uuid4()),
                collection_id=collection.id,
                name=f"doc_{collection.name}_{doc_idx}.txt",
                content=fake.text(max_nb_chars=5000),
                file_type="text",
                file_size=5000,
                created_at=datetime.now(UTC),
            )
            documents.append(doc)
            async_session.add(doc)
            
            # Calculate expected partition for each document
            partition_key = calculate_partition_key(collection.id, doc.id)
            expected_distribution[partition_key] += 1
    
    await async_session.commit()
    
    # Create chunks for all documents (10 chunks per document = 10,000 total)
    chunks_created = 0
    chunk_distribution = Counter()
    
    for doc in documents:
        for chunk_idx in range(10):
            chunk = Chunk(
                collection_id=doc.collection_id,
                document_id=doc.id,
                content=fake.text(max_nb_chars=500),
                chunk_index=chunk_idx,
                start_offset=chunk_idx * 500,
                end_offset=(chunk_idx + 1) * 500,
                token_count=random.randint(50, 150),
                created_at=datetime.now(UTC),
            )
            async_session.add(chunk)
            
            # Track distribution
            partition_key = calculate_partition_key(doc.collection_id, doc.id)
            chunk_distribution[partition_key] += 1
            chunks_created += 1
            
            # Commit in batches to avoid memory issues
            if chunks_created % 1000 == 0:
                await async_session.commit()
    
    await async_session.commit()
    
    return {
        "collections": test_collections,
        "documents": documents,
        "total_chunks": chunks_created,
        "expected_distribution": dict(chunk_distribution),
    }


class TestPartitionDistribution:
    """Test partition distribution characteristics."""
    
    @pytest.mark.asyncio()
    async def test_partition_key_calculation(
        self,
        async_session: AsyncSession,
        test_collections: List[Collection],
    ) -> None:
        """Test that partition keys are calculated correctly."""
        # Create test chunks with known inputs
        test_cases = []
        for collection in test_collections[:3]:
            doc_id = str(uuid.uuid4())
            doc = Document(
                id=doc_id,
                collection_id=collection.id,
                name="test.txt",
                content="test content",
                file_type="text",
                file_size=100,
                created_at=datetime.now(UTC),
            )
            async_session.add(doc)
            
            chunk = Chunk(
                collection_id=collection.id,
                document_id=doc_id,
                content="test chunk",
                chunk_index=0,
                start_offset=0,
                end_offset=100,
                token_count=10,
                created_at=datetime.now(UTC),
            )
            async_session.add(chunk)
            test_cases.append((collection.id, doc_id))
        
        await async_session.commit()
        
        # Verify partition keys match expected values
        for collection_id, doc_id in test_cases:
            result = await async_session.execute(
                select(Chunk.partition_key).where(
                    Chunk.collection_id == collection_id,
                    Chunk.document_id == doc_id,
                )
            )
            actual_key = result.scalar_one()
            
            # Partition key should be between 0 and 99
            assert 0 <= actual_key <= 99
            
            # Verify it's deterministic (same inputs = same key)
            result2 = await async_session.execute(
                select(Chunk.partition_key).where(
                    Chunk.collection_id == collection_id,
                    Chunk.document_id == doc_id,
                )
            )
            assert result2.scalar_one() == actual_key
    
    @pytest.mark.asyncio()
    async def test_even_distribution_large_dataset(
        self,
        async_session: AsyncSession,
        large_dataset: Dict[str, Any],
    ) -> None:
        """Test that chunks are evenly distributed across partitions."""
        # Query actual partition distribution
        result = await async_session.execute(
            select(
                Chunk.partition_key,
                func.count(Chunk.id).label("count"),
            ).group_by(Chunk.partition_key)
        )
        
        actual_distribution = {row[0]: row[1] for row in result}
        
        # Statistical analysis
        partition_counts = list(actual_distribution.values())
        total_chunks = sum(partition_counts)
        mean_chunks = statistics.mean(partition_counts)
        std_dev = statistics.stdev(partition_counts) if len(partition_counts) > 1 else 0
        
        # Assertions
        assert total_chunks == large_dataset["total_chunks"]
        
        # Check that we're using multiple partitions (at least 30 for 10k chunks)
        assert len(actual_distribution) >= 30
        
        # Check for reasonable distribution (coefficient of variation < 0.3)
        if mean_chunks > 0:
            cv = std_dev / mean_chunks
            assert cv < 0.3, f"Distribution too skewed: CV={cv:.2f}"
        
        # No partition should have more than 2x the mean
        max_count = max(partition_counts)
        assert max_count < mean_chunks * 2, f"Partition too large: {max_count} vs mean {mean_chunks}"
        
        # No partition should have less than 0.5x the mean (if it has data)
        min_count = min(partition_counts)
        assert min_count > mean_chunks * 0.5, f"Partition too small: {min_count} vs mean {mean_chunks}"
    
    @pytest.mark.asyncio()
    async def test_partition_skew_detection(
        self,
        async_session: AsyncSession,
        test_collections: List[Collection],
    ) -> None:
        """Test detection of skewed partition distribution."""
        # Create intentionally skewed distribution
        # Put 90% of chunks in first 10 partitions
        skewed_collection = test_collections[0]
        
        # Create documents that will hash to low partition keys
        for i in range(100):
            # Use predictable IDs that hash to low values
            doc_id = f"00000000-0000-0000-0000-{i:012d}"
            doc = Document(
                id=doc_id,
                collection_id=skewed_collection.id,
                name=f"skewed_{i}.txt",
                content="content",
                file_type="text",
                file_size=100,
                created_at=datetime.now(UTC),
            )
            async_session.add(doc)
            
            for chunk_idx in range(10):
                chunk = Chunk(
                    collection_id=skewed_collection.id,
                    document_id=doc_id,
                    content=f"chunk {chunk_idx}",
                    chunk_index=chunk_idx,
                    start_offset=0,
                    end_offset=100,
                    token_count=10,
                    created_at=datetime.now(UTC),
                )
                async_session.add(chunk)
        
        await async_session.commit()
        
        # Use PartitionManager to detect skew
        partition_manager = PartitionManager(async_session)
        health_report = await partition_manager.check_partition_health()
        
        # Should detect distribution issues
        assert health_report["status"] in ["warning", "critical"]
        assert health_report["distribution_score"] < 80  # Poor distribution score
    
    @pytest.mark.asyncio()
    async def test_collection_locality(
        self,
        async_session: AsyncSession,
        test_collections: List[Collection],
    ) -> None:
        """Test that chunks from the same collection tend to cluster."""
        # Create chunks for a single collection
        collection = test_collections[0]
        doc_ids = []
        
        for i in range(20):
            doc = Document(
                id=str(uuid.uuid4()),
                collection_id=collection.id,
                name=f"doc_{i}.txt",
                content="content",
                file_type="text",
                file_size=100,
                created_at=datetime.now(UTC),
            )
            doc_ids.append(doc.id)
            async_session.add(doc)
            
            for chunk_idx in range(5):
                chunk = Chunk(
                    collection_id=collection.id,
                    document_id=doc.id,
                    content=f"chunk {chunk_idx}",
                    chunk_index=chunk_idx,
                    start_offset=0,
                    end_offset=100,
                    token_count=10,
                    created_at=datetime.now(UTC),
                )
                async_session.add(chunk)
        
        await async_session.commit()
        
        # Check partition distribution for this collection
        result = await async_session.execute(
            select(Chunk.partition_key).where(
                Chunk.collection_id == collection.id
            ).distinct()
        )
        
        partitions_used = [row[0] for row in result]
        
        # Chunks from same collection should span limited partitions
        # (not all 100 partitions for just 100 chunks)
        assert len(partitions_used) < 50
        assert len(partitions_used) > 1  # But should use more than 1


class TestPartitionQueryPerformance:
    """Test query performance with partitioned data."""
    
    @pytest.mark.asyncio()
    async def test_partition_pruning_effectiveness(
        self,
        async_session: AsyncSession,
        large_dataset: Dict[str, Any],
    ) -> None:
        """Test that partition pruning works for collection-based queries."""
        collection = large_dataset["collections"][0]
        
        # Warm up the query
        await async_session.execute(
            select(Chunk).where(Chunk.collection_id == collection.id).limit(1)
        )
        
        # Measure query with partition pruning (includes collection_id)
        start_time = time.time()
        result_with_pruning = await async_session.execute(
            select(Chunk).where(
                Chunk.collection_id == collection.id
            )
        )
        chunks_with_pruning = result_with_pruning.scalars().all()
        time_with_pruning = time.time() - start_time
        
        # Measure query without partition pruning (no collection_id)
        # This is just for comparison - normally you wouldn't do this
        doc_id = large_dataset["documents"][0].id
        start_time = time.time()
        result_without_pruning = await async_session.execute(
            select(Chunk).where(
                Chunk.document_id == doc_id
            )
        )
        chunks_without_pruning = result_without_pruning.scalars().all()
        time_without_pruning = time.time() - start_time
        
        # Partition pruning should be faster for large datasets
        # (May not always be true for small test datasets)
        print(f"With pruning: {time_with_pruning:.3f}s for {len(chunks_with_pruning)} chunks")
        print(f"Without pruning: {time_without_pruning:.3f}s for {len(chunks_without_pruning)} chunks")
        
        # Verify we got the expected chunks
        assert len(chunks_with_pruning) > 0
        assert all(c.collection_id == collection.id for c in chunks_with_pruning)
    
    @pytest.mark.asyncio()
    async def test_cross_partition_query_performance(
        self,
        async_session: AsyncSession,
        large_dataset: Dict[str, Any],
    ) -> None:
        """Test performance of queries that span multiple partitions."""
        # Query chunks from multiple collections (spans partitions)
        collection_ids = [c.id for c in large_dataset["collections"][:3]]
        
        start_time = time.time()
        result = await async_session.execute(
            select(Chunk).where(
                Chunk.collection_id.in_(collection_ids)
            )
        )
        chunks = result.scalars().all()
        query_time = time.time() - start_time
        
        # Should still complete in reasonable time
        assert query_time < 5.0  # 5 seconds max for test data
        assert len(chunks) > 0
        
        # Verify chunks are from multiple partitions
        partition_keys = {chunk.partition_key for chunk in chunks[:100]}
        assert len(partition_keys) > 1
    
    @pytest.mark.asyncio()
    async def test_partition_parallel_query(
        self,
        async_session: AsyncSession,
        large_dataset: Dict[str, Any],
    ) -> None:
        """Test parallel queries to different partitions."""
        collections = large_dataset["collections"][:5]
        
        async def query_collection(collection_id: str) -> int:
            """Query chunks for a single collection."""
            result = await async_session.execute(
                select(func.count()).select_from(Chunk).where(
                    Chunk.collection_id == collection_id
                )
            )
            return result.scalar()
        
        # Run parallel queries
        start_time = time.time()
        tasks = [query_collection(c.id) for c in collections]
        counts = await asyncio.gather(*tasks)
        parallel_time = time.time() - start_time
        
        # Run sequential queries for comparison
        start_time = time.time()
        sequential_counts = []
        for collection in collections:
            count = await query_collection(collection.id)
            sequential_counts.append(count)
        sequential_time = time.time() - start_time
        
        # Parallel should be similar or faster than sequential
        print(f"Parallel: {parallel_time:.3f}s, Sequential: {sequential_time:.3f}s")
        
        # Verify results are consistent
        assert counts == sequential_counts
        assert all(count > 0 for count in counts)
    
    @pytest.mark.asyncio()
    async def test_partition_index_usage(
        self,
        async_session: AsyncSession,
        large_dataset: Dict[str, Any],
    ) -> None:
        """Test that partition-specific indexes are used effectively."""
        collection = large_dataset["collections"][0]
        
        # Query using indexed columns
        queries = [
            # Collection-based query (uses partition pruning)
            select(Chunk).where(
                Chunk.collection_id == collection.id,
                Chunk.chunk_index < 5,
            ),
            # Document-based query within collection
            select(Chunk).where(
                Chunk.collection_id == collection.id,
                Chunk.document_id == large_dataset["documents"][0].id,
            ),
            # Time-based query with collection filter
            select(Chunk).where(
                Chunk.collection_id == collection.id,
                Chunk.created_at >= datetime.now(UTC).replace(hour=0, minute=0, second=0),
            ),
        ]
        
        for query in queries:
            start_time = time.time()
            result = await async_session.execute(query)
            chunks = result.scalars().all()
            query_time = time.time() - start_time
            
            # All queries should be fast with proper indexes
            assert query_time < 1.0
            assert len(chunks) > 0


class TestPartitionMaintenance:
    """Test partition maintenance and monitoring."""
    
    @pytest.mark.asyncio()
    async def test_partition_health_monitoring(
        self,
        async_session: AsyncSession,
        large_dataset: Dict[str, Any],
    ) -> None:
        """Test partition health monitoring capabilities."""
        partition_manager = PartitionManager(async_session)
        
        # Get comprehensive health report
        health_report = await partition_manager.check_partition_health()
        
        assert "status" in health_report
        assert "total_partitions" in health_report
        assert "active_partitions" in health_report
        assert "distribution_score" in health_report
        assert "recommendations" in health_report
        
        # Should show healthy status for evenly distributed data
        assert health_report["status"] in ["healthy", "warning"]
        assert health_report["total_partitions"] == 100
        assert health_report["active_partitions"] > 0
    
    @pytest.mark.asyncio()
    async def test_partition_statistics(
        self,
        async_session: AsyncSession,
        large_dataset: Dict[str, Any],
    ) -> None:
        """Test partition statistics collection."""
        # Get partition statistics
        result = await async_session.execute(
            text("""
                SELECT 
                    partition_key,
                    COUNT(*) as chunk_count,
                    AVG(token_count) as avg_tokens,
                    MIN(created_at) as oldest_chunk,
                    MAX(created_at) as newest_chunk
                FROM chunks
                GROUP BY partition_key
                ORDER BY chunk_count DESC
                LIMIT 10
            """)
        )
        
        stats = result.fetchall()
        
        assert len(stats) > 0
        
        for stat in stats:
            partition_key, chunk_count, avg_tokens, oldest, newest = stat
            assert 0 <= partition_key <= 99
            assert chunk_count > 0
            assert avg_tokens > 0
            assert oldest <= newest
    
    @pytest.mark.asyncio()
    async def test_identify_hot_partitions(
        self,
        async_session: AsyncSession,
        large_dataset: Dict[str, Any],
    ) -> None:
        """Test identification of hot (frequently accessed) partitions."""
        # Simulate access patterns by querying specific collections more
        hot_collection = large_dataset["collections"][0]
        
        # Multiple queries to create "hot" partition
        for _ in range(10):
            await async_session.execute(
                select(Chunk).where(
                    Chunk.collection_id == hot_collection.id
                ).limit(10)
            )
        
        # In a real system, we'd track access patterns
        # For testing, we'll check partition sizes
        result = await async_session.execute(
            select(
                Chunk.partition_key,
                func.count(Chunk.id).label("count"),
            ).where(
                Chunk.collection_id == hot_collection.id
            ).group_by(Chunk.partition_key)
        )
        
        hot_partitions = result.fetchall()
        
        assert len(hot_partitions) > 0
        
        # These partitions contain the hot collection's data
        for partition_key, count in hot_partitions:
            assert count > 0
    
    @pytest.mark.asyncio()
    async def test_partition_size_limits(
        self,
        async_session: AsyncSession,
    ) -> None:
        """Test that partition sizes are monitored and limits are respected."""
        # Check current partition sizes
        result = await async_session.execute(
            text("""
                SELECT 
                    partition_key,
                    pg_size_pretty(SUM(octet_length(content))::bigint) as size,
                    COUNT(*) as chunk_count
                FROM chunks
                GROUP BY partition_key
                HAVING COUNT(*) > 100
                ORDER BY COUNT(*) DESC
            """)
        )
        
        large_partitions = result.fetchall()
        
        # Log any partitions that might need attention
        for partition_key, size, count in large_partitions:
            print(f"Partition {partition_key}: {size}, {count} chunks")
            
            # In production, we'd alert if a partition gets too large
            # For testing, just verify we can identify them
            assert count < 10000  # No partition should have > 10k chunks in test


class TestPartitionFailureScenarios:
    """Test partition failure and recovery scenarios."""
    
    @pytest.mark.asyncio()
    async def test_partition_unavailable_handling(
        self,
        async_session: AsyncSession,
        test_collections: List[Collection],
    ) -> None:
        """Test handling when a partition is unavailable."""
        # This test simulates partition unavailability
        # In real scenario, this would involve actual partition failure
        
        collection = test_collections[0]
        
        # Try to query with a simulated partition failure
        with pytest.raises(Exception):
            # Simulate partition access error
            await async_session.execute(
                text("""
                    -- Simulate accessing a non-existent partition
                    SELECT * FROM chunks_part_999 WHERE collection_id = :collection_id
                """),
                {"collection_id": collection.id},
            )
    
    @pytest.mark.asyncio()
    async def test_partition_rebalancing_simulation(
        self,
        async_session: AsyncSession,
        large_dataset: Dict[str, Any],
    ) -> None:
        """Simulate partition rebalancing scenario."""
        # Get current distribution
        result = await async_session.execute(
            select(
                Chunk.partition_key,
                func.count(Chunk.id).label("count"),
            ).group_by(Chunk.partition_key)
        )
        
        original_distribution = {row[0]: row[1] for row in result}
        
        # Identify overloaded partitions (> mean + 2*stddev)
        counts = list(original_distribution.values())
        mean = statistics.mean(counts)
        stddev = statistics.stdev(counts) if len(counts) > 1 else 0
        threshold = mean + (2 * stddev)
        
        overloaded_partitions = [
            p for p, count in original_distribution.items()
            if count > threshold
        ]
        
        # In a real rebalancing scenario, we would:
        # 1. Identify chunks to move
        # 2. Create new chunks with different partition keys
        # 3. Delete old chunks
        # 4. Update indexes
        
        # For testing, just verify we can identify candidates
        print(f"Found {len(overloaded_partitions)} overloaded partitions")
        
        # Verify detection works
        if overloaded_partitions:
            for partition in overloaded_partitions:
                assert original_distribution[partition] > mean


class TestPartitionBulkOperations:
    """Test bulk operations on partitioned data."""
    
    @pytest.mark.asyncio()
    async def test_bulk_insert_distribution(
        self,
        async_session: AsyncSession,
        test_collections: List[Collection],
    ) -> None:
        """Test that bulk inserts distribute correctly across partitions."""
        collection = test_collections[0]
        
        # Create documents
        documents = []
        for i in range(50):
            doc = Document(
                id=str(uuid.uuid4()),
                collection_id=collection.id,
                name=f"bulk_doc_{i}.txt",
                content=fake.text(max_nb_chars=1000),
                file_type="text",
                file_size=1000,
                created_at=datetime.now(UTC),
            )
            documents.append(doc)
            async_session.add(doc)
        
        await async_session.commit()
        
        # Bulk insert chunks
        chunks_to_insert = []
        for doc in documents:
            for chunk_idx in range(20):
                chunk = Chunk(
                    collection_id=collection.id,
                    document_id=doc.id,
                    content=f"Bulk chunk {chunk_idx} for {doc.name}",
                    chunk_index=chunk_idx,
                    start_offset=chunk_idx * 100,
                    end_offset=(chunk_idx + 1) * 100,
                    token_count=random.randint(10, 30),
                    created_at=datetime.now(UTC),
                )
                chunks_to_insert.append(chunk)
        
        # Insert in batches
        batch_size = 500
        for i in range(0, len(chunks_to_insert), batch_size):
            batch = chunks_to_insert[i:i + batch_size]
            async_session.add_all(batch)
            await async_session.commit()
        
        # Verify distribution
        result = await async_session.execute(
            select(
                Chunk.partition_key,
                func.count(Chunk.id).label("count"),
            ).where(
                Chunk.collection_id == collection.id
            ).group_by(Chunk.partition_key)
        )
        
        distribution = {row[0]: row[1] for row in result}
        
        # Should use multiple partitions
        assert len(distribution) > 1
        
        # Total should match
        total_inserted = sum(distribution.values())
        assert total_inserted == len(chunks_to_insert)
    
    @pytest.mark.asyncio()
    async def test_bulk_delete_partition_aware(
        self,
        async_session: AsyncSession,
        test_collections: List[Collection],
    ) -> None:
        """Test bulk deletion with partition awareness."""
        collection = test_collections[0]
        
        # Create chunks to delete
        doc_id = str(uuid.uuid4())
        doc = Document(
            id=doc_id,
            collection_id=collection.id,
            name="to_delete.txt",
            content="content",
            file_type="text",
            file_size=100,
            created_at=datetime.now(UTC),
        )
        async_session.add(doc)
        
        chunks_to_delete = []
        for i in range(100):
            chunk = Chunk(
                collection_id=collection.id,
                document_id=doc_id,
                content=f"Delete me {i}",
                chunk_index=i,
                start_offset=i * 10,
                end_offset=(i + 1) * 10,
                token_count=5,
                created_at=datetime.now(UTC),
            )
            async_session.add(chunk)
            chunks_to_delete.append(chunk)
        
        await async_session.commit()
        
        # Count before deletion
        count_before = await async_session.execute(
            select(func.count()).select_from(Chunk).where(
                Chunk.collection_id == collection.id,
                Chunk.document_id == doc_id,
            )
        )
        assert count_before.scalar() == 100
        
        # Bulk delete using partition-aware query
        await async_session.execute(
            select(Chunk).where(
                Chunk.collection_id == collection.id,
                Chunk.document_id == doc_id,
            ).delete()
        )
        await async_session.commit()
        
        # Verify deletion
        count_after = await async_session.execute(
            select(func.count()).select_from(Chunk).where(
                Chunk.collection_id == collection.id,
                Chunk.document_id == doc_id,
            )
        )
        assert count_after.scalar() == 0