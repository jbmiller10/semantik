# BE-005: Optimize Database Queries and Add Indexes

## Ticket Information
- **Priority**: HIGH
- **Estimated Time**: 3 hours
- **Dependencies**: BE-001, BE-002
- **Risk Level**: MEDIUM - Performance bottleneck
- **Affected Files**:
  - `packages/webui/services/chunking_service.py`
  - `packages/shared/database/repositories/chunk_repository.py`
  - New migration file for indexes
  - `packages/webui/services/cache_manager.py` (new)

## Context

The statistics endpoint has an N+1 query problem, loading all operations to calculate simple counts. Missing database indexes cause full table scans. No caching layer for frequently accessed data.

### Current Problems

```python
# packages/webui/services/chunking_service.py
# BAD: Loading all records to count them
result = await self.db_session.execute(
    select(Operation)
    .where(Operation.collection_id == collection_id)
    .where(Operation.type == "chunking")
    .order_by(Operation.created_at.desc())
)
operations = result.scalars().all()  # Loads ALL operations!
completed = len([op for op in operations if op.status == 'completed'])
```

## Requirements

1. Fix N+1 query problems with aggregation queries
2. Add composite indexes for common query patterns
3. Implement query result caching with Redis
4. Add query performance monitoring
5. Optimize chunk retrieval queries
6. Implement batch fetching where appropriate

## Technical Details

### 1. Fix N+1 Query in Statistics

```python
# packages/webui/services/chunking_service.py

from sqlalchemy import func, select, case, and_
from sqlalchemy.sql import text

class ChunkingService:
    async def get_statistics(self, collection_id: str) -> Dict[str, Any]:
        """Get statistics with optimized queries"""
        
        # Single aggregation query instead of loading all records
        stats_query = select(
            func.count(Operation.id).label('total_operations'),
            func.count(
                case(
                    (Operation.status == 'completed', Operation.id),
                    else_=None
                )
            ).label('completed_operations'),
            func.count(
                case(
                    (Operation.status == 'failed', Operation.id),
                    else_=None
                )
            ).label('failed_operations'),
            func.count(
                case(
                    (Operation.status == 'processing', Operation.id),
                    else_=None
                )
            ).label('processing_operations'),
            func.avg(
                case(
                    (Operation.status == 'completed', Operation.processing_time),
                    else_=None
                )
            ).label('avg_processing_time'),
            func.max(Operation.created_at).label('last_operation_at'),
            func.min(Operation.created_at).label('first_operation_at')
        ).where(
            and_(
                Operation.collection_id == collection_id,
                Operation.type == 'chunking'
            )
        )
        
        result = await self.db_session.execute(stats_query)
        stats = result.one()
        
        # Get chunk statistics with single query
        chunk_stats_query = select(
            func.count(Chunk.id).label('total_chunks'),
            func.avg(func.length(Chunk.content)).label('avg_chunk_size'),
            func.min(func.length(Chunk.content)).label('min_chunk_size'),
            func.max(func.length(Chunk.content)).label('max_chunk_size'),
            func.count(func.distinct(Chunk.document_id)).label('unique_documents')
        ).where(
            Chunk.collection_id == collection_id
        )
        
        chunk_result = await self.db_session.execute(chunk_stats_query)
        chunk_stats = chunk_result.one()
        
        # Get strategy distribution with single query
        strategy_query = select(
            Operation.metadata['strategy'].label('strategy'),
            func.count(Operation.id).label('count')
        ).where(
            and_(
                Operation.collection_id == collection_id,
                Operation.type == 'chunking',
                Operation.metadata.isnot(None)
            )
        ).group_by(
            Operation.metadata['strategy']
        )
        
        strategy_result = await self.db_session.execute(strategy_query)
        strategy_distribution = {
            row.strategy: row.count
            for row in strategy_result
        }
        
        return {
            "operations": {
                "total": stats.total_operations or 0,
                "completed": stats.completed_operations or 0,
                "failed": stats.failed_operations or 0,
                "processing": stats.processing_operations or 0,
                "success_rate": (
                    (stats.completed_operations / stats.total_operations * 100)
                    if stats.total_operations > 0 else 0
                ),
                "avg_processing_time": float(stats.avg_processing_time or 0),
                "last_operation": stats.last_operation_at,
                "first_operation": stats.first_operation_at
            },
            "chunks": {
                "total": chunk_stats.total_chunks or 0,
                "avg_size": float(chunk_stats.avg_chunk_size or 0),
                "min_size": chunk_stats.min_chunk_size or 0,
                "max_size": chunk_stats.max_chunk_size or 0,
                "unique_documents": chunk_stats.unique_documents or 0
            },
            "strategies": strategy_distribution
        }
```

### 2. Create Database Indexes Migration

```python
# alembic/versions/xxx_add_chunking_indexes.py

"""Add indexes for chunking performance

Revision ID: xxx_add_chunking_indexes
Revises: xxx_generated_partition_key
Create Date: 2024-xx-xx
"""

from alembic import op
import sqlalchemy as sa

def upgrade():
    # Composite indexes for common queries
    
    # Operations table indexes
    op.create_index(
        'idx_operations_collection_type_status',
        'operations',
        ['collection_id', 'type', 'status'],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_operations_created_desc',
        'operations',
        [sa.text('created_at DESC')],
        postgresql_using='btree'
    )
    
    op.create_index(
        'idx_operations_user_status',
        'operations',
        ['user_id', 'status'],
        postgresql_where=sa.text("status IN ('processing', 'pending')"),
        postgresql_using='btree'
    )
    
    # Chunks table indexes (per partition)
    for i in range(100):
        # Collection + document index
        op.create_index(
            f'idx_chunks_part_{i}_collection_document',
            f'chunks_part_{i}',
            ['collection_id', 'document_id'],
            postgresql_using='btree'
        )
        
        # Document + index for ordering
        op.create_index(
            f'idx_chunks_part_{i}_document_index',
            f'chunks_part_{i}',
            ['document_id', 'chunk_index'],
            postgresql_using='btree'
        )
        
        # Created at for time-based queries
        op.create_index(
            f'idx_chunks_part_{i}_created',
            f'chunks_part_{i}',
            ['created_at'],
            postgresql_using='brin'  # BRIN for time series
        )
    
    # Collections table indexes
    op.create_index(
        'idx_collections_user_status',
        'collections',
        ['user_id', 'status'],
        postgresql_using='btree'
    )
    
    # Documents table indexes
    op.create_index(
        'idx_documents_collection_status',
        'documents',
        ['collection_id', 'status'],
        postgresql_using='btree'
    )
    
    # JSONB indexes for metadata queries
    op.create_index(
        'idx_operations_metadata_strategy',
        'operations',
        [sa.text("(metadata->>'strategy')")],
        postgresql_using='btree',
        postgresql_where=sa.text("metadata IS NOT NULL")
    )
    
    # Analyze tables to update statistics
    op.execute("ANALYZE operations")
    op.execute("ANALYZE collections")
    op.execute("ANALYZE documents")
    
    # Analyze all chunk partitions
    for i in range(100):
        op.execute(f"ANALYZE chunks_part_{i}")

def downgrade():
    # Drop all indexes
    op.drop_index('idx_operations_collection_type_status')
    op.drop_index('idx_operations_created_desc')
    op.drop_index('idx_operations_user_status')
    op.drop_index('idx_collections_user_status')
    op.drop_index('idx_documents_collection_status')
    op.drop_index('idx_operations_metadata_strategy')
    
    for i in range(100):
        op.drop_index(f'idx_chunks_part_{i}_collection_document')
        op.drop_index(f'idx_chunks_part_{i}_document_index')
        op.drop_index(f'idx_chunks_part_{i}_created')
```

### 3. Implement Query Result Caching

```python
# packages/webui/services/cache_manager.py

import json
import hashlib
from typing import Any, Optional, Callable
from datetime import timedelta
import aioredis
from functools import wraps

class CacheManager:
    """Manage query result caching"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.default_ttl = 300  # 5 minutes
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
    
    def _generate_cache_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate deterministic cache key"""
        # Sort params for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(sorted_params.encode()).hexdigest()
        return f"cache:{prefix}:{param_hash}"
    
    async def get(
        self,
        key: str,
        deserializer: Callable = json.loads
    ) -> Optional[Any]:
        """Get value from cache"""
        value = await self.redis.get(key)
        
        if value is None:
            self.misses += 1
            return None
        
        self.hits += 1
        return deserializer(value)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serializer: Callable = json.dumps
    ):
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        serialized = serializer(value)
        await self.redis.setex(key, ttl, serialized)
    
    async def delete(self, pattern: str):
        """Delete keys matching pattern"""
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor,
                match=pattern,
                count=100
            )
            
            if keys:
                await self.redis.delete(*keys)
            
            if cursor == 0:
                break
    
    async def invalidate_collection(self, collection_id: str):
        """Invalidate all cache entries for a collection"""
        await self.delete(f"cache:*:{collection_id}:*")
    
    def cache_result(
        self,
        prefix: str,
        ttl: Optional[int] = None,
        key_params: Optional[List[str]] = None
    ):
        """Decorator for caching async function results"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Build cache key from specified params
                if key_params:
                    cache_params = {
                        k: kwargs.get(k)
                        for k in key_params
                        if k in kwargs
                    }
                else:
                    # Use all kwargs as cache params
                    cache_params = kwargs
                
                cache_key = self._generate_cache_key(prefix, cache_params)
                
                # Try to get from cache
                cached = await self.get(cache_key)
                if cached is not None:
                    return cached
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": hit_rate
        }

# Update service to use caching
class ChunkingService:
    def __init__(self, ...):
        # ... existing init ...
        self.cache_manager = CacheManager(redis_client)
    
    @cache_manager.cache_result(
        prefix="statistics",
        ttl=60,  # 1 minute cache
        key_params=["collection_id"]
    )
    async def get_statistics(self, collection_id: str) -> Dict[str, Any]:
        # ... implementation from above ...
        pass
    
    @cache_manager.cache_result(
        prefix="strategies",
        ttl=3600,  # 1 hour cache
        key_params=[]
    )
    async def get_available_strategies(self) -> List[str]:
        # ... implementation ...
        pass
```

### 4. Optimize Chunk Retrieval

```python
# packages/shared/database/repositories/chunk_repository.py

class ChunkRepository:
    async def get_chunks_batch(
        self,
        collection_id: str,
        document_ids: List[str],
        limit: int = 1000
    ) -> List[Chunk]:
        """Batch fetch chunks for multiple documents"""
        
        # Use IN clause for batch fetching
        query = select(Chunk).where(
            and_(
                Chunk.collection_id == collection_id,
                Chunk.document_id.in_(document_ids)
            )
        ).order_by(
            Chunk.document_id,
            Chunk.chunk_index
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_chunks_paginated(
        self,
        collection_id: str,
        page: int = 1,
        page_size: int = 100
    ) -> Tuple[List[Chunk], int]:
        """Get paginated chunks with total count"""
        
        # Use window function for efficient pagination
        query = select(
            Chunk,
            func.count(Chunk.id).over().label('total_count')
        ).where(
            Chunk.collection_id == collection_id
        ).order_by(
            Chunk.created_at.desc()
        ).limit(page_size).offset((page - 1) * page_size)
        
        result = await self.session.execute(query)
        rows = result.all()
        
        if not rows:
            return [], 0
        
        chunks = [row[0] for row in rows]
        total_count = rows[0][1] if rows else 0
        
        return chunks, total_count
    
    async def get_chunks_by_similarity(
        self,
        collection_id: str,
        embedding: List[float],
        limit: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """Get chunks by embedding similarity (if using pgvector)"""
        
        # Assuming pgvector extension
        query = text("""
            SELECT 
                c.*,
                c.embedding_vector <=> :embedding as distance
            FROM chunks c
            WHERE c.collection_id = :collection_id
                AND c.embedding_vector IS NOT NULL
            ORDER BY distance
            LIMIT :limit
        """)
        
        result = await self.session.execute(
            query,
            {
                "collection_id": collection_id,
                "embedding": embedding,
                "limit": limit
            }
        )
        
        return [
            (Chunk(**dict(row)), row.distance)
            for row in result
        ]
```

### 5. Add Query Performance Monitoring

```python
# packages/webui/services/query_monitor.py

import time
from contextlib import asynccontextmanager
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class QueryMonitor:
    """Monitor database query performance"""
    
    def __init__(self):
        self.slow_query_threshold = 1.0  # 1 second
        self.query_times: List[Dict[str, Any]] = []
        self.slow_queries: List[Dict[str, Any]] = []
    
    @asynccontextmanager
    async def monitor(self, query_name: str, params: Dict[str, Any] = None):
        """Monitor query execution time"""
        start_time = time.time()
        
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            
            query_info = {
                "name": query_name,
                "params": params,
                "execution_time": execution_time,
                "timestamp": time.time()
            }
            
            self.query_times.append(query_info)
            
            # Track slow queries
            if execution_time > self.slow_query_threshold:
                self.slow_queries.append(query_info)
                
                logger.warning(
                    f"Slow query detected",
                    extra={
                        "query_name": query_name,
                        "execution_time": execution_time,
                        "params": params
                    }
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        if not self.query_times:
            return {}
        
        times = [q["execution_time"] for q in self.query_times]
        
        return {
            "total_queries": len(self.query_times),
            "slow_queries": len(self.slow_queries),
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "p50_time": sorted(times)[len(times) // 2],
            "p99_time": sorted(times)[int(len(times) * 0.99)]
        }
```

## Acceptance Criteria

1. **Query Optimization**
   - [ ] No N+1 queries remaining
   - [ ] All aggregations use SQL functions
   - [ ] Batch fetching implemented
   - [ ] Pagination optimized

2. **Indexes**
   - [ ] Composite indexes for common queries
   - [ ] JSONB indexes for metadata
   - [ ] BRIN indexes for time series
   - [ ] Statistics updated after index creation

3. **Caching**
   - [ ] Query results cached in Redis
   - [ ] Cache invalidation on updates
   - [ ] Cache hit rate > 80%
   - [ ] TTL configured appropriately

4. **Performance**
   - [ ] Statistics query < 100ms
   - [ ] Chunk retrieval < 50ms
   - [ ] No full table scans
   - [ ] Query plan uses indexes

5. **Monitoring**
   - [ ] Slow queries logged
   - [ ] Query metrics tracked
   - [ ] Cache metrics available
   - [ ] Performance dashboards updated

## Testing Requirements

1. **Performance Tests**
   ```python
   async def test_statistics_performance():
       # Create 10,000 operations
       await create_test_operations(10000)
       
       start = time.time()
       stats = await service.get_statistics(collection_id)
       duration = time.time() - start
       
       assert duration < 0.1  # Less than 100ms
       assert stats["operations"]["total"] == 10000
   
   async def test_cache_hit_rate():
       cache = CacheManager(redis)
       
       # First call - cache miss
       result1 = await service.get_statistics(collection_id)
       
       # Second call - cache hit
       result2 = await service.get_statistics(collection_id)
       
       assert result1 == result2
       assert cache.get_stats()["hit_rate"] >= 50
   
   async def test_index_usage():
       # Explain query to verify index usage
       explain = await session.execute(
           text("EXPLAIN (FORMAT JSON) SELECT * FROM operations WHERE collection_id = :id"),
           {"id": collection_id}
       )
       
       plan = explain.scalar()
       assert "Index Scan" in str(plan)
   ```

2. **Load Tests**
   - Test with 1M+ records
   - Concurrent query execution
   - Cache performance under load
   - Index effectiveness

## Rollback Plan

1. Keep original queries commented
2. Indexes can be dropped if issues
3. Cache can be disabled via feature flag
4. Monitor query performance metrics

## Success Metrics

- Query response time p99 < 100ms
- Zero N+1 queries in monitoring
- Cache hit rate > 80%
- Database CPU usage reduced by 30%
- No query timeouts under normal load

## Notes for LLM Agent

- Test query plans with EXPLAIN ANALYZE
- Verify indexes are being used
- Monitor cache memory usage
- Consider partition-aware queries
- Test with realistic data volumes
- Add query timeouts for safety