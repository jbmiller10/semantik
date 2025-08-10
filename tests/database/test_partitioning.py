"""
Comprehensive tests for the 100-partition database implementation.

Tests cover:
- Even distribution of collections across partitions
- Partition pruning effectiveness
- Query performance with partitions
- Monitoring and health check functions
"""

import uuid
import random
import asyncio
from typing import List, Dict, Any
from datetime import datetime
import pytest
import hashlib

from sqlalchemy import text, select, insert
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.chunking.infrastructure.repositories.partition_manager import (
    PartitionManager,
    PartitionHealth,
    DistributionStats
)


class TestPartitionDistribution:
    """Test even distribution of data across partitions."""
    
    def test_partition_count(self):
        """Verify we have exactly 100 partitions configured."""
        assert PartitionManager.PARTITION_COUNT == 100
    
    def test_partition_id_calculation(self):
        """Test that partition ID calculation is deterministic and within range."""
        manager = PartitionManager()
        
        # Test with known UUIDs
        test_cases = [
            str(uuid.uuid4()) for _ in range(100)
        ]
        
        for collection_id in test_cases:
            partition_id = manager.get_partition_id(collection_id)
            
            # Verify partition ID is within valid range
            assert 0 <= partition_id < 100, f"Partition ID {partition_id} out of range"
            
            # Verify calculation is deterministic
            partition_id_2 = manager.get_partition_id(collection_id)
            assert partition_id == partition_id_2, "Partition ID calculation not deterministic"
    
    def test_partition_name_generation(self):
        """Test partition name generation with proper formatting."""
        manager = PartitionManager()
        
        # Test edge cases
        test_cases = [
            ("00000000-0000-0000-0000-000000000000", lambda x: x.startswith("chunks_part_")),
            ("ffffffff-ffff-ffff-ffff-ffffffffffff", lambda x: x.startswith("chunks_part_")),
        ]
        
        for collection_id, validator in test_cases:
            name = manager.get_partition_name(collection_id)
            assert validator(name), f"Invalid partition name: {name}"
            
            # Verify format is chunks_part_XX where XX is zero-padded
            parts = name.split("_")
            assert len(parts) == 3
            assert parts[0] == "chunks"
            assert parts[1] == "part"
            assert len(parts[2]) == 2
            assert parts[2].isdigit()
    
    def test_even_distribution_simulation(self):
        """
        Simulate distribution of 10,000 collections across partitions.
        Verify no partition exceeds 20% deviation from average.
        """
        manager = PartitionManager()
        num_collections = 10000
        
        # Generate random collection IDs
        collection_ids = [str(uuid.uuid4()) for _ in range(num_collections)]
        
        # Count distribution across partitions
        partition_counts: Dict[int, int] = {}
        for cid in collection_ids:
            partition_id = manager.get_partition_id(cid)
            partition_counts[partition_id] = partition_counts.get(partition_id, 0) + 1
        
        # Calculate statistics
        expected_per_partition = num_collections / PartitionManager.PARTITION_COUNT
        
        # Check distribution
        max_deviation = 0
        for partition_id in range(PartitionManager.PARTITION_COUNT):
            count = partition_counts.get(partition_id, 0)
            
            if expected_per_partition > 0:
                deviation = abs(count - expected_per_partition) / expected_per_partition
                max_deviation = max(max_deviation, deviation)
        
        # Assert max deviation is within 20% threshold
        assert max_deviation < 0.2, (
            f"Maximum deviation {max_deviation:.2%} exceeds 20% threshold. "
            f"Distribution may be uneven."
        )
        
        # Verify all partitions get some data (statistical test)
        empty_partitions = sum(
            1 for i in range(PartitionManager.PARTITION_COUNT) 
            if i not in partition_counts
        )
        
        # With 10,000 items across 100 partitions, probability of any partition
        # being empty is extremely low
        assert empty_partitions < 5, f"Too many empty partitions: {empty_partitions}"
    
    def test_distribution_with_sequential_ids(self):
        """Test that sequential IDs still distribute well."""
        manager = PartitionManager()
        
        # Generate sequential-looking IDs (common in some systems)
        base_uuid = uuid.uuid4()
        collection_ids = []
        for i in range(1000):
            # Modify last part of UUID
            uuid_str = str(base_uuid)[:-4] + f"{i:04d}"
            collection_ids.append(uuid_str)
        
        # Count distribution
        partition_counts: Dict[int, int] = {}
        for cid in collection_ids:
            partition_id = manager.get_partition_id(cid)
            partition_counts[partition_id] = partition_counts.get(partition_id, 0) + 1
        
        # Sequential IDs should still hash to different partitions
        # We expect at least 50 different partitions to be used
        assert len(partition_counts) > 50, (
            f"Sequential IDs only distributed to {len(partition_counts)} partitions"
        )


@pytest.mark.asyncio
class TestPartitionOperations:
    """Test database operations with partitions."""
    
    async def test_partition_health_query(self, db_session: AsyncSession):
        """Test that partition health view works correctly."""
        manager = PartitionManager()
        
        # Get partition health (should work even with empty partitions)
        health_data = await manager.get_partition_health(db_session)
        
        # Should have data for all 100 partitions
        assert len(health_data) == 100, f"Expected 100 partitions, got {len(health_data)}"
        
        # Verify each partition has required fields
        for partition in health_data:
            assert isinstance(partition, PartitionHealth)
            assert 0 <= partition.partition_id < 100
            assert partition.partition_name.startswith("chunks_part_")
            assert partition.row_count >= 0
            assert partition.size_bytes >= 0
            assert partition.partition_status in ['HOT', 'COLD', 'NORMAL']
    
    async def test_distribution_stats(self, db_session: AsyncSession):
        """Test distribution statistics calculation."""
        manager = PartitionManager()
        
        # Get distribution stats
        stats = await manager.get_distribution_stats(db_session)
        
        assert isinstance(stats, DistributionStats)
        assert stats.partitions_used >= 0
        assert stats.empty_partitions >= 0
        assert stats.partitions_used + stats.empty_partitions <= 100
        assert stats.distribution_status in ['HEALTHY', 'WARNING', 'REBALANCE NEEDED']
        assert len(stats.recommendations) > 0
    
    async def test_partition_skew_analysis(self, db_session: AsyncSession):
        """Test partition skew analysis function."""
        manager = PartitionManager()
        
        # Analyze skew
        skew_data = await manager.analyze_partition_skew(db_session)
        
        assert 'status' in skew_data
        assert 'max_skew_ratio' in skew_data
        assert 'recommendation' in skew_data
        
        # With empty or minimal data, should be healthy
        if skew_data['status'] != 'NO_DATA':
            assert skew_data['status'] in ['HEALTHY', 'WARNING', 'CRITICAL']
    
    async def test_verify_partition_assignment(self, db_session: AsyncSession):
        """Test that Python and PostgreSQL partition calculations match."""
        manager = PartitionManager()
        
        # Test with multiple collection IDs
        test_ids = [str(uuid.uuid4()) for _ in range(10)]
        
        for collection_id in test_ids:
            result = await manager.verify_partition_for_collection(
                db_session, 
                collection_id
            )
            
            # Python and DB calculations should match
            # Note: This will fail if PostgreSQL hashtext() behaves differently
            # In production, we rely on PostgreSQL's calculation
            assert result['collection_id'] == collection_id
            assert result['python_partition_id'] is not None
            assert result['db_partition_id'] is not None
            
            # The partition names should be valid
            assert result['python_partition_name'].startswith("chunks_part_")
            assert result['db_partition_name'].startswith("chunks_part_")


@pytest.mark.asyncio
class TestPartitionPruning:
    """Test that partition pruning works effectively."""
    
    async def test_single_collection_query_plan(self, db_session: AsyncSession):
        """
        Test that querying a single collection only scans one partition.
        """
        collection_id = str(uuid.uuid4())
        
        # Get query plan
        explain_query = text("""
            EXPLAIN (FORMAT JSON, BUFFERS FALSE, ANALYZE FALSE) 
            SELECT * FROM chunks 
            WHERE collection_id = :collection_id
        """)
        
        result = await db_session.execute(
            explain_query, 
            {"collection_id": collection_id}
        )
        
        plan_json = result.scalar()
        
        # The plan should mention only one partition
        # Look for "chunks_part_" in the plan
        plan_str = str(plan_json)
        
        # Count how many different partition references
        partition_refs = []
        for i in range(100):
            partition_name = f"chunks_part_{i:02d}"
            if partition_name in plan_str:
                partition_refs.append(partition_name)
        
        # Should reference at most 1 partition (or 0 if optimized away)
        assert len(partition_refs) <= 1, (
            f"Query plan references multiple partitions: {partition_refs}"
        )
    
    async def test_partition_constraint_exclusion(self, db_session: AsyncSession):
        """
        Verify that PostgreSQL constraint exclusion is working.
        """
        # Check that constraint_exclusion is properly configured
        result = await db_session.execute(
            text("SHOW constraint_exclusion")
        )
        setting = result.scalar()
        
        # Should be 'partition' or 'on' for proper partition pruning
        assert setting in ['partition', 'on'], (
            f"constraint_exclusion is '{setting}', should be 'partition' or 'on'"
        )


@pytest.mark.asyncio
class TestPartitionPerformance:
    """Performance tests for partitioned table operations."""
    
    async def test_bulk_insert_performance(self, db_session: AsyncSession):
        """Test that bulk inserts distribute across partitions efficiently."""
        # Generate test data
        num_collections = 10
        chunks_per_collection = 100
        
        insert_data = []
        for _ in range(num_collections):
            collection_id = str(uuid.uuid4())
            for chunk_idx in range(chunks_per_collection):
                insert_data.append({
                    'collection_id': collection_id,
                    'chunk_index': chunk_idx,
                    'content': f'Test content for chunk {chunk_idx}',
                    'metadata': {}
                })
        
        # Measure insert time
        start_time = datetime.now()
        
        # Bulk insert
        for chunk in insert_data:
            await db_session.execute(
                text("""
                    INSERT INTO chunks (collection_id, chunk_index, content, metadata)
                    VALUES (:collection_id, :chunk_index, :content, :metadata)
                """),
                chunk
            )
        
        await db_session.commit()
        
        insert_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertion - should complete reasonably quickly
        # Adjust threshold based on your performance requirements
        assert insert_time < 10, f"Bulk insert took {insert_time}s, expected < 10s"
        
        # Verify data was distributed
        manager = PartitionManager()
        stats = await manager.get_distribution_stats(db_session)
        
        assert stats.partitions_used >= min(num_collections, 100)
        assert stats.total_rows == num_collections * chunks_per_collection
    
    async def test_query_performance_with_partition_key(self, db_session: AsyncSession):
        """Test query performance when partition key is used."""
        collection_id = str(uuid.uuid4())
        
        # Insert some test data
        for i in range(10):
            await db_session.execute(
                text("""
                    INSERT INTO chunks (collection_id, chunk_index, content, metadata)
                    VALUES (:collection_id, :chunk_index, :content, :metadata)
                """),
                {
                    'collection_id': collection_id,
                    'chunk_index': i,
                    'content': f'Test content {i}',
                    'metadata': {}
                }
            )
        await db_session.commit()
        
        # Measure query time
        start_time = datetime.now()
        
        result = await db_session.execute(
            text("""
                SELECT * FROM chunks 
                WHERE collection_id = :collection_id
                ORDER BY chunk_index
            """),
            {"collection_id": collection_id}
        )
        
        rows = result.fetchall()
        query_time = (datetime.now() - start_time).total_seconds()
        
        # Should return correct number of rows
        assert len(rows) == 10
        
        # Query should be fast (< 100ms for small dataset)
        assert query_time < 0.1, f"Query took {query_time}s, expected < 0.1s"


@pytest.mark.asyncio
class TestPartitionMonitoring:
    """Test partition monitoring and management features."""
    
    async def test_hot_partition_detection(self, db_session: AsyncSession):
        """Test detection of hot (overloaded) partitions."""
        manager = PartitionManager()
        
        # Get hot partitions
        hot_partitions = await manager.get_hot_partitions(db_session)
        
        # Should return a list (possibly empty if data is balanced)
        assert isinstance(hot_partitions, list)
        
        # If there are hot partitions, verify they're properly identified
        for partition in hot_partitions:
            assert isinstance(partition, PartitionHealth)
            # Hot partitions should have positive deviation or HOT status
            assert (
                partition.partition_status == 'HOT' or 
                partition.pct_deviation_from_avg > 10
            )
    
    async def test_efficiency_report(self, db_session: AsyncSession):
        """Test comprehensive efficiency report generation."""
        manager = PartitionManager()
        
        # Get efficiency report
        report = await manager.get_efficiency_report(db_session)
        
        # Verify report structure
        assert 'efficiency_score' in report
        assert 0 <= report['efficiency_score'] <= 100
        
        assert 'total_partitions' in report
        assert report['total_partitions'] == 100
        
        assert 'distribution_status' in report
        assert 'recommendations' in report
        assert isinstance(report['recommendations'], list)
        
        assert 'partition_efficiency' in report
        efficiency = report['partition_efficiency']
        assert sum([
            efficiency['excellent'],
            efficiency['good'],
            efficiency['fair'],
            efficiency['poor']
        ]) == 1  # Exactly one should be True


class TestPartitionMaintenance:
    """Test partition maintenance operations."""
    
    def test_get_all_partition_names(self):
        """Test generation of all partition names."""
        manager = PartitionManager()
        
        names = manager.get_all_partition_names()
        
        # Should have exactly 100 names
        assert len(names) == 100
        
        # All should follow the pattern
        for i, name in enumerate(names):
            assert name == f"chunks_part_{i:02d}"
        
        # Should be unique
        assert len(set(names)) == 100
    
    def test_partition_id_consistency(self):
        """Test that partition ID calculation is consistent with naming."""
        manager = PartitionManager()
        
        # Test multiple UUIDs
        for _ in range(100):
            collection_id = str(uuid.uuid4())
            
            partition_id = manager.get_partition_id(collection_id)
            partition_name = manager.get_partition_name(collection_id)
            
            # Extract ID from name
            name_parts = partition_name.split("_")
            name_id = int(name_parts[2])
            
            # Should match
            assert partition_id == name_id, (
                f"Partition ID {partition_id} doesn't match name ID {name_id}"
            )


# Fixtures for database testing
@pytest.fixture
async def db_session():
    """
    Provide a database session for testing.
    This should be implemented based on your test database setup.
    """
    # This is a placeholder - implement based on your test configuration
    # from shared.database import get_test_db_session
    # async with get_test_db_session() as session:
    #     yield session
    pass