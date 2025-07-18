# Collection-Centric Architecture Refactor - Final Execution Plan v5.0

**Status:** Final with Complete Enhancements
**Date:** 2025-07-15
**Timeline:** 9-11 weeks (added 1 week for operational readiness)

## Executive Summary

This final plan implements a clean collection-centric architecture for pre-release Semantik, incorporating all critical refinements plus operational monitoring, security hardening, and comprehensive runbooks. This version adds essential production-readiness features identified in the final review.

## Phase 0: Operational Foundation (1 week) - NEW

### 0.1 Monitoring Infrastructure

```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/dashboards:/var/lib/grafana/dashboards
```

### 0.2 Metrics Implementation

```python
# packages/shared/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Collection metrics
collection_operations_total = Counter(
    'semantik_collection_operations_total',
    'Total collection operations',
    ['operation_type', 'status']
)

collection_operation_duration = Histogram(
    'semantik_collection_operation_duration_seconds',
    'Collection operation duration',
    ['operation_type']
)

collections_total = Gauge(
    'semantik_collections_total',
    'Total number of collections',
    ['status']
)

collection_documents_total = Gauge(
    'semantik_collection_documents_total',
    'Total documents per collection',
    ['collection_id']
)

# Search metrics
search_latency = Histogram(
    'semantik_search_latency_seconds',
    'Search request latency',
    ['collection_count']
)

search_results_total = Histogram(
    'semantik_search_results_total',
    'Number of search results returned'
)
```

### 0.3 Health Check Framework

```python
# packages/webui/api/health.py
@router.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check for monitoring"""
    checks = {
        "database": await check_database_health(),
        "qdrant": await check_qdrant_health(),
        "redis": await check_redis_health(),
        "celery": await check_celery_health(),
        "disk_space": await check_disk_space(),
        "memory": await check_memory_usage()
    }
    
    overall_status = "healthy" if all(c["status"] == "healthy" for c in checks.values()) else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
        "version": settings.APP_VERSION
    }
```

## Phase 1: Database & Core Models (1 week)

### 1.1 Enhanced Schema with Audit Trail

```sql
-- Previous schema plus additions
CREATE TABLE collection_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    operation_id INTEGER REFERENCES operations(id),
    user_id INTEGER REFERENCES users(id),
    action TEXT NOT NULL,  -- created, updated, deleted, reindexed, etc.
    details JSON,
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Resource limits table
CREATE TABLE collection_resource_limits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    max_documents INTEGER DEFAULT 100000,
    max_storage_gb FLOAT DEFAULT 50.0,
    max_operations_per_hour INTEGER DEFAULT 10,
    max_sources INTEGER DEFAULT 10,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(collection_id)
);

-- Performance tracking
CREATE TABLE operation_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_id INTEGER NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    metric_value FLOAT NOT NULL,
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_operation_metrics_operation_id (operation_id),
    INDEX idx_operation_metrics_recorded_at (recorded_at)
);
```

### 1.2 SQLAlchemy Models with Health Methods

```python
class Collection(Base):
    # ... previous fields ...
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for collection"""
        active_op = self.get_active_operation()
        recent_failures = self.get_recent_failures(hours=24)
        
        health_score = 100
        issues = []
        
        if self.status == CollectionStatus.ERROR:
            health_score = 0
            issues.append("Collection in error state")
        elif self.status == CollectionStatus.DEGRADED:
            health_score = 70
            issues.append("Collection degraded")
        
        if recent_failures > 5:
            health_score -= 20
            issues.append(f"{recent_failures} failures in last 24h")
        
        if not self.qdrant_collections:
            health_score = min(health_score, 50)
            issues.append("No vector storage configured")
        
        return {
            "score": health_score,
            "status": self.get_health_status(health_score),
            "issues": issues,
            "last_successful_operation": self.last_successful_operation(),
            "active_operation": active_op.id if active_op else None
        }
```

## Phase 2: Core Services & Repositories (1 week)

### 2.1 Repository Layer with Audit Logging

```python
# packages/shared/database/repositories/collection_repository.py
class CollectionRepository:
    def __init__(self, db_session, audit_logger):
        self.db = db_session
        self.audit = audit_logger
    
    async def create(self, user_id: int, name: str, config: CollectionConfig, 
                    request_context: RequestContext) -> Collection:
        """Create collection with audit trail"""
        try:
            # Validate resource limits
            await self.validate_user_limits(user_id)
            
            # Create collection
            collection = Collection(
                uuid=str(uuid.uuid4()),
                name=name,
                user_id=user_id,
                **config.dict()
            )
            
            self.db.add(collection)
            await self.db.commit()
            
            # Audit log
            await self.audit.log(
                collection_id=collection.id,
                user_id=user_id,
                action="created",
                details={"config": config.dict()},
                request_context=request_context
            )
            
            # Metrics
            collections_total.labels(status="ready").inc()
            
            return collection
            
        except Exception as e:
            await self.db.rollback()
            raise
    
    async def update_status(self, collection_id: int, status: CollectionStatus, 
                          message: str = None):
        """Update status with metrics"""
        old_status = await self.get_status(collection_id)
        
        # Update status
        await super().update_status(collection_id, status, message)
        
        # Update metrics
        collections_total.labels(status=old_status.value).dec()
        collections_total.labels(status=status.value).inc()
```

### 2.2 Service Layer with Resource Management

```python
# packages/webui/services/collection_service.py
class CollectionService:
    def __init__(self, collection_repo, operation_repo, document_repo, 
                 qdrant_manager, resource_manager):
        # ... existing init ...
        self.resource_manager = resource_manager
    
    async def create_collection(self, user_id: int, request: CreateCollectionRequest,
                              request_context: RequestContext) -> Collection:
        """Create collection with resource validation"""
        # Check resource limits
        if not await self.resource_manager.can_create_collection(user_id):
            raise ResourceLimitExceeded("Collection limit reached")
        
        # Estimate resource usage
        estimated_resources = await self.estimate_resources(
            request.source_path,
            request.model_name
        )
        
        if not await self.resource_manager.can_allocate(user_id, estimated_resources):
            raise ResourceLimitExceeded(
                f"Insufficient resources. Required: {estimated_resources}"
            )
        
        # Create collection with monitoring
        with collection_operations_total.labels(
            operation_type="create",
            status="processing"
        ).count_exceptions():
            collection = await self.collection_repo.create(
                user_id=user_id,
                name=request.name,
                config=request.to_config(),
                request_context=request_context
            )
            
            # ... rest of creation logic ...
            
            collection_operations_total.labels(
                operation_type="create",
                status="success"
            ).inc()
            
            return collection
```

## Phase 3: Task Processing with Enhanced Monitoring (1.5 weeks)

### 3.1 Unified Task with Performance Tracking

```python
# packages/webui/tasks.py
@celery.task(bind=True, name="process_collection_operation")
def process_collection_operation(self, operation_id: int):
    """Enhanced task with comprehensive monitoring"""
    start_time = time.time()
    
    try:
        # Store task ID immediately
        operation_repo.set_task_id(operation_id, self.request.id)
        
        # Get operation details
        operation = operation_repo.get(operation_id)
        collection = collection_repo.get(operation.collection_id)
        
        # Log start
        logger.info(
            "Starting operation",
            extra={
                "operation_id": operation_id,
                "operation_type": operation.type,
                "collection_id": collection.id,
                "collection_name": collection.name
            }
        )
        
        # Update operation status
        operation_repo.update_status(operation_id, OperationStatus.PROCESSING)
        
        # Route to handler with timing
        with collection_operation_duration.labels(
            operation_type=operation.type
        ).time():
            if operation.type == "index":
                result = index_collection(operation, collection)
            elif operation.type == "append":
                result = append_to_collection(operation, collection)
            elif operation.type == "reindex":
                result = reindex_collection(operation, collection)
            elif operation.type == "remove_source":
                result = remove_source_from_collection(operation, collection)
            else:
                raise ValueError(f"Unknown operation type: {operation.type}")
        
        # Record success metrics
        duration = time.time() - start_time
        await record_operation_metrics(operation_id, {
            "duration_seconds": duration,
            "documents_processed": result.get("documents_processed", 0),
            "vectors_created": result.get("vectors_created", 0),
            "memory_peak_mb": get_peak_memory_usage()
        })
        
        # Update final status
        operation_repo.update_status(operation_id, OperationStatus.COMPLETED)
        collection_repo.update_status(operation.collection_id, CollectionStatus.READY)
        
        # Success metric
        collection_operations_total.labels(
            operation_type=operation.type,
            status="success"
        ).inc()
        
    except Exception as e:
        # Record failure metrics
        duration = time.time() - start_time
        await record_operation_metrics(operation_id, {
            "duration_seconds": duration,
            "error_type": type(e).__name__,
            "error_phase": get_current_phase()
        })
        
        # Failure metric
        collection_operations_total.labels(
            operation_type=operation.type,
            status="failed"
        ).inc()
        
        # Enhanced error handling
        handle_operation_failure(operation, collection, e)
        raise
```

### 3.2 Blue-Green Reindex with Validation

```python
def reindex_collection(operation: Operation, collection: Collection):
    """Enhanced reindexing with validation checkpoints"""
    staging_collections = []
    checkpoints = ReindexCheckpoints()
    
    try:
        # Checkpoint 1: Pre-flight checks
        checkpoints.record("preflight_start")
        
        # Verify collection health
        health = collection.health_check()
        if health["score"] < 50:
            raise UnhealthyCollectionError(
                f"Collection health too low: {health['score']}"
            )
        
        # Check resource availability
        if not resource_manager.reserve_for_reindex(collection.id):
            raise ResourceUnavailableError("Insufficient resources for reindex")
        
        checkpoints.record("preflight_complete")
        
        # ... existing reindexing logic with checkpoints ...
        
        # Checkpoint N: Validation
        checkpoints.record("validation_start")
        
        validation_result = await comprehensive_validation(
            old_collections=collection.qdrant_collections,
            new_collections=staging_collections,
            sample_size=min(100, collection.document_count // 10)
        )
        
        if not validation_result.passed:
            raise ValidationError(
                f"Reindex validation failed: {validation_result.issues}"
            )
        
        checkpoints.record("validation_complete")
        
        # ... atomic switch ...
        
    except Exception as e:
        # Enhanced cleanup with checkpoint info
        logger.error(
            f"Reindex failed at checkpoint: {checkpoints.last_checkpoint}",
            extra={"checkpoints": checkpoints.to_dict()}
        )
        
        # ... cleanup logic ...
        raise
    
    finally:
        # Always release resources
        resource_manager.release_reindex_reservation(collection.id)
```

## Phase 4: API Implementation with Security (1.5 weeks)

### 4.1 Enhanced API with Rate Limiting

```python
# packages/webui/api/v2/collections.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/", response_model=CollectionResponse)
@limiter.limit("10/hour")
async def create_collection(
    request: CreateCollectionRequest,
    current_user: User = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
    request_context: RequestContext = Depends(get_request_context)
):
    """Create collection with rate limiting and audit trail"""
    # Input validation with sanitization
    sanitized_request = sanitize_collection_request(request)
    
    try:
        collection = await service.create_collection(
            current_user.id, 
            sanitized_request,
            request_context
        )
        
        return CollectionResponse.from_orm(collection)
        
    except ResourceLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{collection_uuid}/health", response_model=CollectionHealthResponse)
async def collection_health(
    collection_uuid: str,
    current_user: User = Depends(get_current_user),
    repo: CollectionRepository = Depends(get_collection_repo)
):
    """Get comprehensive collection health status"""
    collection = await repo.get_by_uuid(collection_uuid, current_user.id)
    health = collection.health_check()
    
    # Add Qdrant health
    qdrant_health = await check_qdrant_collections_health(
        collection.qdrant_collections
    )
    
    return CollectionHealthResponse(
        collection_id=collection.uuid,
        health_score=health["score"],
        status=health["status"],
        issues=health["issues"],
        qdrant_health=qdrant_health,
        recent_operations=await get_recent_operations_summary(collection.id),
        resource_usage=await get_resource_usage(collection.id)
    )
```

### 4.2 Internal API with Enhanced Security

```python
# packages/webui/api/internal/operations.py
from cryptography.fernet import Fernet

# Initialize encryption for sensitive operations
fernet = Fernet(settings.INTERNAL_API_ENCRYPTION_KEY.encode())

@router.post("/complete-reindex")
async def complete_reindex(
    request: CompleteReindexRequest,
    api_key: str = Depends(verify_internal_api_key),
    signature: str = Header(..., alias="X-Request-Signature")
):
    """Atomically complete reindex with request signing"""
    # Verify request signature
    if not verify_request_signature(request, signature):
        raise HTTPException(status_code=401, detail="Invalid request signature")
    
    # Log the critical operation
    audit_logger.critical(
        "Atomic reindex switch requested",
        extra={
            "collection_id": request.collection_id,
            "staging_collections": request.staging_collections,
            "api_key_hash": hash_api_key(api_key)
        }
    )
    
    # ... existing atomic switch logic ...
```

## Phase 5: Frontend Implementation with UX Enhancements (2 weeks)

### 5.1 Collection Store with Operation Queue

```typescript
interface CollectionStore {
  // ... existing state ...
  operationQueue: Map<string, QueuedOperation[]>  // Operations waiting to start
  resourceUsage: ResourceUsage
  
  // Enhanced actions
  queueOperation: (collectionId: string, operation: QueuedOperation) => void
  getQueuePosition: (operationId: string) => number
  cancelQueuedOperation: (operationId: string) => Promise<void>
  
  // Resource monitoring
  checkResourceAvailability: (operation: OperationType) => ResourceAvailability
  getEstimatedWaitTime: (operation: OperationType) => number
}

interface QueuedOperation {
  id: string
  type: OperationType
  priority: 'low' | 'normal' | 'high'
  estimatedResources: ResourceEstimate
  queuedAt: Date
  estimatedStartTime?: Date
}
```

### 5.2 Enhanced Collection Card with Queue Status

```typescript
export function CollectionCard({ collection }: { collection: CollectionWithOperation }) {
  const { operationQueue, getQueuePosition } = useCollectionStore()
  const queuedOps = operationQueue.get(collection.uuid) || []
  
  return (
    <div className={`
      relative rounded-lg border p-6 transition-all
      ${collection.isProcessing ? 'border-blue-500 bg-blue-50' : 
        queuedOps.length > 0 ? 'border-yellow-500 bg-yellow-50' : 
        'border-gray-200'}
    `}>
      {/* Queue indicator */}
      {queuedOps.length > 0 && !collection.isProcessing && (
        <div className="absolute top-2 right-2">
          <div className="flex items-center space-x-2">
            <Clock className="w-4 h-4 text-yellow-600" />
            <span className="text-sm text-yellow-600">
              {queuedOps.length} operation{queuedOps.length > 1 ? 's' : ''} queued
            </span>
          </div>
        </div>
      )}
      
      {/* Health indicator */}
      <HealthIndicator health={collection.health} />
      
      {/* ... rest of card ... */}
    </div>
  )
}

function HealthIndicator({ health }: { health: CollectionHealth }) {
  const getHealthColor = (score: number) => {
    if (score >= 90) return 'green'
    if (score >= 70) return 'yellow'
    if (score >= 50) return 'orange'
    return 'red'
  }
  
  const color = getHealthColor(health.score)
  
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger>
          <div className={`flex items-center space-x-1 text-${color}-600`}>
            <Heart className="w-4 h-4" />
            <span className="text-sm">{health.score}%</span>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <div className="space-y-1">
            <p className="font-medium">Collection Health</p>
            {health.issues.map((issue, i) => (
              <p key={i} className="text-sm text-gray-600">• {issue}</p>
            ))}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
```

## Phase 6: Testing & Documentation (1 week)

### 6.1 Chaos Testing Suite

```python
# tests/chaos/test_collection_chaos.py
import pytest
from chaos_monkey import ChaosMonkey

class TestCollectionChaos:
    @pytest.mark.chaos
    async def test_reindex_with_qdrant_failure(self, chaos_monkey: ChaosMonkey):
        """Test reindex recovery when Qdrant fails mid-operation"""
        collection = await create_test_collection()
        
        # Schedule Qdrant failure at 50% progress
        chaos_monkey.schedule_failure(
            service="qdrant",
            at_progress=0.5,
            duration_seconds=30
        )
        
        # Attempt reindex
        operation = await trigger_reindex(collection.uuid)
        
        # Wait for completion or timeout
        result = await wait_for_operation(operation.id, timeout=300)
        
        # Verify recovery
        assert result.status in ["completed", "failed"]
        
        if result.status == "failed":
            # Verify collection is still searchable
            search_result = await search_collection(collection.uuid, "test")
            assert search_result.status_code == 200
            
            # Verify staging was cleaned up
            collection_details = await get_collection(collection.uuid)
            assert collection_details.qdrant_staging is None
    
    @pytest.mark.chaos
    async def test_concurrent_operations_resource_contention(self):
        """Test system behavior under resource contention"""
        # Create multiple collections
        collections = [await create_test_collection() for _ in range(5)]
        
        # Trigger simultaneous reindex operations
        operations = []
        for collection in collections:
            op = await trigger_reindex(collection.uuid)
            operations.append(op)
        
        # Monitor resource usage
        peak_memory = 0
        peak_cpu = 0
        
        while any(op.status == "processing" for op in operations):
            metrics = await get_system_metrics()
            peak_memory = max(peak_memory, metrics.memory_usage_mb)
            peak_cpu = max(peak_cpu, metrics.cpu_percent)
            await asyncio.sleep(1)
        
        # Verify resource limits were respected
        assert peak_memory < settings.MAX_MEMORY_MB
        assert peak_cpu < settings.MAX_CPU_PERCENT
```

### 6.2 Runbook Documentation

```markdown
# Collection Operations Runbook

## Emergency Procedures

### 1. Collection Stuck in Reindexing State

**Symptoms:**
- Collection status shows "reindexing" for extended period
- No active operation in queue
- Search may or may not be working

**Diagnosis:**
```bash
# Check collection status
curl http://localhost:8080/api/v2/collections/{uuid}/health

# Check for orphaned operations
SELECT * FROM operations 
WHERE collection_id = ? 
AND status IN ('pending', 'processing')
ORDER BY created_at DESC;

# Check Qdrant collections
curl http://localhost:6333/collections
```

**Resolution:**
1. Verify no active Celery task:
   ```python
   celery -A app inspect active
   ```

2. If no active task, manually fail the operation:
   ```sql
   UPDATE operations 
   SET status = 'failed', 
       error_message = 'Manual intervention: stuck operation'
   WHERE id = ?;
   ```

3. Clear staging collections:
   ```sql
   UPDATE collections 
   SET qdrant_staging = NULL,
       status = 'degraded',
       status_message = 'Reindexing interrupted - manual intervention'
   WHERE id = ?;
   ```

4. Clean up orphaned Qdrant collections:
   ```python
   python scripts/cleanup_orphaned_collections.py --collection-id {uuid}
   ```

### 2. Search Performance Degradation

**Symptoms:**
- Search latency > 2 seconds
- Timeouts on search requests
- High CPU usage on search API

**Diagnosis:**
```bash
# Check search metrics
curl http://localhost:9090/api/v1/query?query=semantik_search_latency_seconds

# Check Qdrant performance
curl http://localhost:6333/telemetry

# Check active searches
redis-cli> KEYS search:*
```

**Resolution:**
1. Check for oversized collections:
   ```sql
   SELECT c.name, c.vector_count, c.document_count
   FROM collections c
   WHERE c.vector_count > 1000000
   ORDER BY c.vector_count DESC;
   ```

2. Enable search caching if disabled:
   ```python
   SEARCH_CACHE_ENABLED=true
   SEARCH_CACHE_TTL=300
   ```

3. Consider sharding large collections:
   ```python
   python scripts/shard_large_collection.py --collection-id {uuid} --shards 4
   ```

## Performance Tuning

### Optimal Configuration by Scale

| Documents | Model | Chunk Size | Batch Size | Workers |
|-----------|-------|------------|------------|---------|
| < 1,000   | 0.6B  | 512        | 100        | 2       |
| < 10,000  | 0.6B  | 512        | 50         | 4       |
| < 100,000 | 0.6B  | 256        | 25         | 8       |
| > 100,000 | 0.6B  | 256        | 10         | 8       |

### Resource Allocation Guidelines

```yaml
# For small deployments (< 10 collections)
services:
  search-api:
    resources:
      limits:
        memory: 4G
        cpus: '2'
  
  webui-api:
    resources:
      limits:
        memory: 2G
        cpus: '1'
  
  worker:
    resources:
      limits:
        memory: 8G
        cpus: '4'

# For large deployments (> 100 collections)
services:
  search-api:
    deploy:
      replicas: 3
    resources:
      limits:
        memory: 8G
        cpus: '4'
  
  worker:
    deploy:
      replicas: 4
    resources:
      limits:
        memory: 16G
        cpus: '8'
```
```

## Phase 7: Operational Validation (1 week)

### 7.1 Load Testing Scenarios

```python
# load_tests/collection_load_test.py
from locust import HttpUser, task, between

class CollectionUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Login
        response = self.client.post("/api/auth/login", json={
            "username": "loadtest",
            "password": "loadtest123"
        })
        self.token = response.json()["access_token"]
        self.client.headers.update({"Authorization": f"Bearer {self.token}"})
        
        # Get available collections
        response = self.client.get("/api/v2/collections")
        self.collections = response.json()
    
    @task(4)
    def search_collection(self):
        """Most common operation - searching"""
        if not self.collections:
            return
            
        collection = random.choice(self.collections)
        
        with self.client.post(
            "/api/v2/search",
            json={
                "query": random.choice(SAMPLE_QUERIES),
                "collection_uuids": [collection["uuid"]],
                "limit": 10
            },
            catch_response=True
        ) as response:
            if response.elapsed.total_seconds() > 0.5:
                response.failure(f"Search too slow: {response.elapsed.total_seconds()}s")
    
    @task(1)
    def create_collection(self):
        """Less common - create new collection"""
        with self.client.post(
            "/api/v2/collections",
            json={
                "name": f"LoadTest-{uuid.uuid4().hex[:8]}",
                "description": "Load test collection",
                "source_path": "/data/test/small",
                "model_name": "Qwen/Qwen3-Embedding-0.6B",
                "chunk_size": 512,
                "chunk_overlap": 50
            }
        ) as response:
            if response.status_code == 201:
                self.collections.append(response.json())
```

### 7.2 Security Audit Checklist

- [ ] All API endpoints require authentication
- [ ] Collection access is properly scoped to users
- [ ] Internal APIs use separate authentication
- [ ] Rate limiting is applied to resource-intensive operations
- [ ] Input validation prevents injection attacks
- [ ] File paths are sanitized and confined
- [ ] Sensitive data is not logged
- [ ] Error messages don't leak implementation details
- [ ] CORS is properly configured
- [ ] TLS is enforced in production mode

## Summary of Enhancements

This v5.0 plan adds:

1. **Phase 0**: Operational foundation with monitoring and health checks
2. **Enhanced Security**: Rate limiting, request signing, audit trails
3. **Resource Management**: Quotas, reservations, and usage tracking
4. **Chaos Testing**: Resilience validation under failure conditions
5. **Operational Runbooks**: Step-by-step emergency procedures
6. **Performance Baselines**: Clear metrics and tuning guidelines
7. **Queue Management**: Visibility into pending operations
8. **Health Monitoring**: Comprehensive health scoring system

The additions increase timeline by 1 week but ensure production-grade quality and operational excellence.