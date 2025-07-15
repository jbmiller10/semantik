# Semantik Queue Refactoring Plan v3.0 (Pre-Release)

## Executive Summary

Since Semantik is pre-release, we can take a more aggressive approach without backwards compatibility concerns. This streamlined plan focuses on building the right architecture from the start, rather than maintaining legacy code paths.

### Key Advantages of Pre-Release Refactoring
- **No Feature Flags**: Direct replacement of components
- **No Migration Paths**: Clean cutover to new patterns
- **No Legacy Support**: Delete old code immediately
- **Faster Timeline**: 4-5 weeks instead of 10

## Phase 0: Critical Security Fixes (1 day)

### Task 0.1: Fix JWT Secret Configuration
**Priority**: CRITICAL - Do this TODAY

```python
# packages/shared/config/webui.py
import secrets
from pathlib import Path

class WebuiConfig(BaseConfig):
    JWT_SECRET_KEY: str = Field(
        default="",
        description="JWT secret key for token signing"
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.JWT_SECRET_KEY:
            # Auto-generate secure key
            secret_file = Path(".jwt_secret")
            if secret_file.exists():
                self.JWT_SECRET_KEY = secret_file.read_text().strip()
            else:
                secret_key = secrets.token_urlsafe(32)
                secret_file.write_text(secret_key)
                secret_file.chmod(0o600)
                self.JWT_SECRET_KEY = secret_key
                logger.info("Generated new JWT secret key")
```

### Task 0.2: Add CORS Configuration
```python
# packages/webui/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Phase 1: Complete Repository Pattern (3-4 days)

### Task 1.1: Finish Repository Implementations
Since we don't need backwards compatibility, we can directly implement all missing repositories:

```python
# packages/shared/database/sqlite_repository.py
class SQLiteCollectionRepository(CollectionRepository):
    async def list_collections(self, user_id: str) -> List[CollectionInfo]:
        # Direct implementation, no legacy wrapper
        return await asyncio.to_thread(
            self._list_collections_sync, user_id
        )
    
    # ... implement all methods
```

### Task 1.2: Direct API Migration
**No feature flags, just replace**:

```python
# packages/webui/api/jobs.py
from fastapi import Depends
from shared.database import JobRepository, create_job_repository

# DELETE: All imports from database module
# DELETE: All direct database.* calls

@router.get("/")
async def list_jobs(
    repo: JobRepository = Depends(create_job_repository),
    current_user: dict = Depends(get_current_user)
):
    # Direct repository usage
    return await repo.list_jobs(user_id=str(current_user["id"]))
```

### Task 1.3: Delete Legacy Code
```bash
# After migration complete
rm packages/shared/database/legacy_wrappers.py
rm packages/shared/database/__init__.py  # Remove old exports
# Update __init__.py to only export repository interfaces
```

## Phase 2: Celery/Redis Implementation (1 week)

### Task 2.1: Replace Async Job System Entirely
**No hybrid approach needed**:

```python
# DELETE: packages/webui/api/jobs.py lines 110-115 (active_job_tasks, executor)

# NEW: packages/webui/tasks.py
from celery import Celery
from packages.vecpipe.worker import process_embedding_job_sync

app = Celery('webui', broker='redis://redis:6379')

@app.task(bind=True, max_retries=3)
def process_embedding_job_task(self, job_id: str):
    try:
        # Convert async to sync for Celery
        asyncio.run(process_embedding_job_sync(job_id))
    except Exception as exc:
        # Exponential backoff retry
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
```

### Task 2.2: Simple WebSocket Updates via Redis
```python
# packages/webui/websocket_manager.py
class RedisWebSocketManager:
    def __init__(self):
        self.redis = aioredis.from_url("redis://redis:6379")
        self.connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.connections[client_id] = websocket
        # Subscribe to Redis channel for this client
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(f"updates:{client_id}")
        
    async def broadcast_job_update(self, user_id: str, job_id: str, status: dict):
        # From Celery task
        await self.redis.publish(f"updates:{user_id}", json.dumps({
            "job_id": job_id,
            "status": status
        }))
```

### Task 2.3: Docker Compose with All Services
```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    
  worker:
    build: .
    command: celery -A webui.tasks worker -l info -c 1
    depends_on:
      - redis
      - qdrant
    volumes:
      - ./data:/app/data
      - ${DOCUMENT_PATH:-./documents}:/mnt/docs:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
              
  flower:  # Celery monitoring
    build: .
    command: celery -A webui.tasks flower
    ports:
      - "5555:5555"
    depends_on:
      - redis
```

## Phase 3: Testing Strategy (1 week concurrent with Phase 2)

### Task 3.1: Frontend Testing from Scratch
```bash
cd apps/webui-react
npm install -D vitest @testing-library/react jsdom msw
```

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/tests/setup.ts',
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: ['node_modules/', 'src/tests/']
    }
  }
})
```

### Task 3.2: Component Tests for New Architecture
```typescript
// src/components/JobStatus.test.tsx
import { render, screen } from '@testing-library/react'
import { JobStatus } from './JobStatus'
import { mockWebSocket } from '../tests/mocks/websocket'

test('updates status via WebSocket', async () => {
  const { ws } = mockWebSocket()
  render(<JobStatus jobId="123" />)
  
  // Simulate Celery update via Redis
  ws.send({ job_id: "123", status: "processing", progress: 50 })
  
  await screen.findByText('50% Complete')
})
```

### Task 3.3: Integration Tests for Celery
```python
# tests/integration/test_celery_jobs.py
@pytest.mark.integration
async def test_job_processing_via_celery(celery_app, celery_worker):
    # Create job
    job_id = await create_test_job()
    
    # Submit to Celery
    from webui.tasks import process_embedding_job_task
    result = process_embedding_job_task.delay(job_id)
    
    # Wait for completion
    assert result.get(timeout=30) == "completed"
    
    # Verify database state
    job = await get_job(job_id)
    assert job.status == "completed"
```

## Phase 4: Alembic Migration System (2-3 days)

### Task 4.1: Initialize Alembic with Current Schema
```bash
# Initialize
alembic init alembic

# Create models from existing schema
poetry run python scripts/generate_sqlalchemy_models.py

# Generate initial migration
alembic revision --autogenerate -m "Initial schema from SQLite"
```

### Task 4.2: Replace init_db with Alembic
```python
# packages/shared/database/init.py
def init_db():
    """Initialize database using Alembic"""
    from alembic import command
    from alembic.config import Config
    
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    
    # Create default user if needed
    create_default_user()
```

## Phase 5: Production Readiness (3-4 days)

### Task 5.1: Health Checks for All Services
```python
# packages/webui/api/health.py
@router.get("/health/full")
async def health_check_full():
    checks = {
        "api": "healthy",
        "database": await check_database(),
        "redis": await check_redis(),
        "celery": await check_celery_workers(),
        "qdrant": await check_qdrant(),
    }
    
    status_code = 200 if all(v == "healthy" for v in checks.values()) else 503
    return JSONResponse(content=checks, status_code=status_code)
```

### Task 5.2: Monitoring Stack
```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      
  grafana:
    image: grafana/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
```

### Task 5.3: Performance Benchmarks
```python
# benchmarks/run_benchmarks.py
def benchmark_embedding_throughput():
    """Benchmark new Celery-based system"""
    # Submit 100 jobs
    job_ids = []
    for i in range(100):
        job_id = create_embedding_job(test_documents[i])
        job_ids.append(job_id)
    
    # Wait for completion
    start = time.time()
    wait_for_jobs(job_ids)
    elapsed = time.time() - start
    
    print(f"Throughput: {100/elapsed:.2f} jobs/second")
```

## Simplified Timeline

### Week 1
- Day 1: Phase 0 (Security fixes)
- Day 2-4: Phase 1 (Complete repository pattern)
- Day 5: Start Phase 2 (Celery setup)

### Week 2
- Complete Phase 2 (Celery/Redis)
- Start Phase 3 (Testing) in parallel
- Begin Phase 4 (Alembic)

### Week 3
- Complete Phase 3 (Testing)
- Complete Phase 4 (Alembic)
- Start Phase 5 (Production readiness)

### Week 4
- Complete Phase 5
- Performance testing
- Documentation updates

## Key Differences from Previous Plan

1. **No Backwards Compatibility**
   - Delete old code immediately
   - No feature flags or gradual migration
   - Direct replacement of components

2. **Faster Implementation**
   - 4 weeks instead of 10
   - Parallel work streams
   - No legacy support overhead

3. **Cleaner Architecture**
   - No hybrid systems
   - No migration paths
   - Fresh start with best practices

4. **Simplified Testing**
   - Test new architecture only
   - No comparison tests
   - Focus on correctness, not compatibility

## Success Criteria

- [ ] All API endpoints using repository pattern
- [ ] All jobs processed via Celery
- [ ] Frontend test coverage >70%
- [ ] Backend test coverage >85%
- [ ] Zero hardcoded secrets
- [ ] Monitoring dashboards operational
- [ ] 10x improvement in job throughput
- [ ] Clean codebase with no legacy remnants

## Next Steps

1. **Today**: Implement Phase 0 security fixes
2. **Tomorrow**: Start repository pattern completion
3. **This Week**: Get Celery worker running
4. **Next Week**: Complete testing infrastructure

The pre-release status is a gift - we can build it right without compromise!