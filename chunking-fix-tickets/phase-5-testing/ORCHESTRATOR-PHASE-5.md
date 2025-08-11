# ORCHESTRATOR PHASE 5: Testing, Monitoring & Operations

## Phase Overview
**Priority**: HIGH - Required for production confidence  
**Total Duration**: 17 hours  
**Risk Level**: LOW - Testing and monitoring setup  
**Success Gate**: 95% test pass rate, monitoring dashboard operational

## Context
Previous phases fixed critical issues. This phase ensures long-term stability through comprehensive testing, monitoring, and automation. Without this, issues will resurface and operations will be blind to problems.

## Execution Strategy

### Pre-Flight Checklist
- [ ] All Phase 1-4 fixes deployed to staging
- [ ] Test data sets prepared (various file types/sizes)
- [ ] Load testing infrastructure ready
- [ ] Monitoring infrastructure (Prometheus/Grafana) available
- [ ] CI/CD pipeline accessible

### Ticket Execution Order

#### Stage 1: E2E Testing (6 hours)
**Ticket**: TEST-001 - Add Comprehensive E2E Tests

**Critical Test Scenarios**:
1. Complete user workflow (upload → chunk → view)
2. All 6 strategies with different documents
3. Error recovery and edge cases
4. Multi-user concurrent operations
5. WebSocket stability over time

**Implementation**:
```python
# tests/e2e/test_chunking_complete.py
import pytest
from playwright.async_api import async_playwright

class TestChunkingE2E:
    @pytest.mark.e2e
    async def test_complete_workflow(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            # 1. Login
            await page.goto('http://localhost:3000')
            await page.fill('[data-testid="email"]', 'test@example.com')
            await page.fill('[data-testid="password"]', 'password')
            await page.click('[data-testid="login-button"]')
            
            # 2. Upload document
            await page.set_input_files(
                '[data-testid="file-upload"]',
                'test_data/sample.pdf'
            )
            
            # 3. Select strategy
            await page.click('[data-testid="strategy-semantic"]')
            
            # 4. Configure parameters
            await page.fill('[data-testid="chunk-size"]', '1000')
            
            # 5. Preview chunking
            await page.click('[data-testid="preview-button"]')
            
            # 6. Wait for WebSocket updates
            await page.wait_for_selector(
                '[data-testid="progress-100"]',
                timeout=30000
            )
            
            # 7. Verify chunks displayed
            chunks = await page.query_selector_all('[data-testid="chunk-card"]')
            assert len(chunks) > 0
            
            # 8. Apply chunking
            await page.click('[data-testid="apply-button"]')
            
            # 9. Verify success
            await page.wait_for_selector('[data-testid="success-message"]')
            
            await browser.close()
    
    @pytest.mark.e2e
    async def test_all_strategies(self):
        strategies = ['character', 'recursive', 'markdown', 
                     'semantic', 'hierarchical', 'hybrid']
        
        for strategy in strategies:
            await self.test_strategy(strategy)
    
    @pytest.mark.e2e
    async def test_error_recovery(self):
        # Test network disconnection
        await page.context.set_offline(True)
        await page.click('[data-testid="preview-button"]')
        
        # Should show error
        await page.wait_for_selector('[data-testid="error-message"]')
        
        # Reconnect and retry
        await page.context.set_offline(False)
        await page.click('[data-testid="retry-button"]')
        
        # Should recover
        await page.wait_for_selector('[data-testid="success-message"]')
```

**Validation Script**:
```bash
#!/bin/bash
# Run E2E tests with coverage
pytest tests/e2e/test_chunking_complete.py \
    --cov=packages \
    --cov-report=html \
    --html=report.html \
    -v

# Check coverage
coverage report --fail-under=80
```

#### Stage 2: Load Testing (4 hours)
**Ticket**: TEST-002 - Implement Load Testing Suite

**Test Scenarios**:
```python
# tests/load/locustfile.py
from locust import HttpUser, task, between
import random

class ChunkingUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Login
        self.client.post("/api/auth/login", json={
            "email": f"user{random.randint(1,100)}@test.com",
            "password": "password"
        })
    
    @task(weight=3)
    def preview_small_document(self):
        """Most common operation - small preview"""
        response = self.client.post(
            "/api/v2/chunking/preview",
            json={
                "content": "x" * random.randint(500, 2000),
                "strategy": random.choice(["character", "recursive"]),
                "config": {"chunk_size": 500}
            }
        )
        assert response.status_code == 200
    
    @task(weight=2)
    def preview_large_document(self):
        """Resource intensive - large document"""
        response = self.client.post(
            "/api/v2/chunking/preview",
            json={
                "content": "x" * random.randint(100000, 500000),
                "strategy": "semantic",
                "config": {"chunk_size": 1000}
            }
        )
        assert response.status_code == 200
    
    @task(weight=1)
    def compare_strategies(self):
        """Most intensive - strategy comparison"""
        response = self.client.post(
            "/api/v2/chunking/compare",
            json={
                "content": "x" * 10000,
                "strategies": ["character", "semantic", "markdown"],
                "config": {"chunk_size": 500}
            }
        )
        assert response.status_code == 200
    
    @task(weight=1)
    def websocket_operation(self):
        """WebSocket stress test"""
        import websocket
        
        ws = websocket.create_connection(f"{self.host.replace('http', 'ws')}/ws")
        ws.send(json.dumps({
            "type": "subscribe",
            "channel": f"chunking:{uuid.uuid4()}"
        }))
        
        # Receive some messages
        for _ in range(10):
            result = ws.recv()
            assert result
        
        ws.close()
```

**Load Test Execution**:
```bash
# Ramp up test
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users=100 \
    --spawn-rate=10 \
    --time=5m \
    --html=load_test_report.html

# Stress test
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users=500 \
    --spawn-rate=50 \
    --time=10m

# Soak test (long duration)
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users=50 \
    --spawn-rate=5 \
    --time=1h
```

**Success Criteria**:
```python
def validate_load_test_results(stats):
    assert stats['failure_rate'] < 0.05  # < 5% errors
    assert stats['response_time_p99'] < 2000  # < 2s
    assert stats['requests_per_second'] > 100  # > 100 RPS
    assert not stats['memory_leak_detected']
    assert stats['websocket_drops'] < 0.01  # < 1% drops
```

#### Stage 3: Monitoring Setup (4 hours)
**Ticket**: MON-001 - Add Monitoring and Alerting

**Metrics Implementation**:
```python
# packages/webui/services/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
chunking_operations = Counter(
    'chunking_operations_total',
    'Total chunking operations',
    ['strategy', 'status']
)

chunking_duration = Histogram(
    'chunking_duration_seconds',
    'Time to complete chunking',
    ['strategy'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
)

active_websockets = Gauge(
    'websocket_connections_active',
    'Currently active WebSocket connections'
)

memory_pool_usage = Gauge(
    'memory_pool_usage_bytes',
    'Current memory pool usage'
)

partition_skew = Gauge(
    'partition_skew_ratio',
    'Maximum partition skew ratio'
)

# Instrument code
class ChunkingService:
    @chunking_duration.time()
    async def preview_chunks(self, strategy: str, **kwargs):
        try:
            result = await self._preview(strategy, **kwargs)
            chunking_operations.labels(strategy=strategy, status='success').inc()
            return result
        except Exception as e:
            chunking_operations.labels(strategy=strategy, status='failure').inc()
            raise
```

**Grafana Dashboard Configuration**:
```json
{
  "dashboard": {
    "title": "Chunking Service Monitor",
    "panels": [
      {
        "title": "Operations Per Minute",
        "targets": [{
          "expr": "rate(chunking_operations_total[1m])"
        }]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(chunking_operations_total{status='failure'}[5m]) / rate(chunking_operations_total[5m])"
        }]
      },
      {
        "title": "Response Time (p99)",
        "targets": [{
          "expr": "histogram_quantile(0.99, rate(chunking_duration_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "WebSocket Connections",
        "targets": [{
          "expr": "websocket_connections_active"
        }]
      },
      {
        "title": "Memory Usage",
        "targets": [{
          "expr": "memory_pool_usage_bytes / 1024 / 1024"
        }]
      }
    ]
  }
}
```

**Alert Rules**:
```yaml
# prometheus/alerts.yml
groups:
  - name: chunking_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(chunking_operations_total{status='failure'}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate in chunking service"
          
      - alert: SlowResponse
        expr: histogram_quantile(0.99, rate(chunking_duration_seconds_bucket[5m])) > 2
        for: 10m
        annotations:
          summary: "Chunking operations are slow"
          
      - alert: MemoryPoolExhausted
        expr: memory_pool_usage_bytes / memory_pool_max_bytes > 0.9
        for: 5m
        annotations:
          summary: "Memory pool nearly exhausted"
          
      - alert: PartitionSkew
        expr: partition_skew_ratio > 2
        for: 30m
        annotations:
          summary: "Partition distribution is skewed"
```

#### Stage 4: Automated Maintenance (3 hours)
**Ticket**: MON-002 - Implement Automated Partition Maintenance

**Maintenance Tasks**:
```python
# packages/webui/tasks/maintenance.py
from celery import Celery
from datetime import datetime, timedelta

app = Celery('maintenance')

@app.task
def analyze_partitions():
    """Run ANALYZE on all partition tables"""
    for i in range(100):
        db.execute(f"ANALYZE chunks_part_{i}")
    
    logger.info("Partition analysis complete")

@app.task
def check_partition_skew():
    """Check and alert on partition skew"""
    stats = db.execute("""
        SELECT 
            partition_key,
            COUNT(*) as count,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
        FROM chunks
        GROUP BY partition_key
    """).fetchall()
    
    max_skew = max(s.percentage for s in stats) / min(s.percentage for s in stats)
    
    if max_skew > 1.5:
        alert_manager.send_alert(
            "Partition skew detected",
            {"skew_ratio": max_skew, "stats": stats}
        )
    
    return max_skew

@app.task
def cleanup_old_chunks():
    """Archive chunks older than retention period"""
    cutoff = datetime.utcnow() - timedelta(days=90)
    
    # Archive to cold storage
    old_chunks = db.execute("""
        SELECT * FROM chunks 
        WHERE created_at < %s
    """, cutoff).fetchall()
    
    for batch in chunks(old_chunks, 1000):
        archive_to_s3(batch)
    
    # Delete from hot storage
    db.execute("""
        DELETE FROM chunks 
        WHERE created_at < %s
    """, cutoff)
    
    logger.info(f"Archived {len(old_chunks)} old chunks")

@app.task
def optimize_indexes():
    """Reindex tables for performance"""
    tables = ['operations', 'chunks', 'documents', 'collections']
    
    for table in tables:
        db.execute(f"REINDEX TABLE {table}")
    
    logger.info("Index optimization complete")

# Schedule tasks
app.conf.beat_schedule = {
    'analyze-partitions': {
        'task': 'maintenance.analyze_partitions',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },
    'check-skew': {
        'task': 'maintenance.check_partition_skew',
        'schedule': crontab(minute='*/30'),  # Every 30 minutes
    },
    'cleanup-old': {
        'task': 'maintenance.cleanup_old_chunks',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),  # Weekly
    },
    'optimize-indexes': {
        'task': 'maintenance.optimize_indexes',
        'schedule': crontab(hour=4, minute=0, day_of_week=0),  # Weekly
    }
}
```

### Test Coverage Validation

```bash
#!/bin/bash
# Comprehensive coverage check

# Backend coverage
pytest packages/ \
    --cov=packages \
    --cov-report=term-missing \
    --cov-report=html:coverage/backend \
    --cov-fail-under=80

# Frontend coverage
cd apps/webui-react
npm test -- --coverage --watchAll=false

# E2E coverage
pytest tests/e2e/ \
    --cov=packages \
    --cov-report=term-missing

# Generate combined report
coverage combine
coverage html -d coverage/combined
coverage report

# Check thresholds
python -c "
import json
with open('coverage/combined/coverage.json') as f:
    data = json.load(f)
    total = data['totals']['percent_covered']
    assert total >= 80, f'Coverage {total}% is below 80%'
    print(f'✓ Total coverage: {total}%')
"
```

### Monitoring Validation

```python
# Verify monitoring is working
def validate_monitoring():
    # Check Prometheus scraping
    metrics = requests.get('http://localhost:9090/api/v1/targets').json()
    assert all(t['health'] == 'up' for t in metrics['data']['activeTargets'])
    
    # Check Grafana dashboards
    dashboards = requests.get(
        'http://localhost:3000/api/dashboards',
        headers={'Authorization': f'Bearer {grafana_token}'}
    ).json()
    assert 'Chunking Service Monitor' in [d['title'] for d in dashboards]
    
    # Check alerts firing
    alerts = requests.get('http://localhost:9093/api/v1/alerts').json()
    print(f"Active alerts: {len(alerts['data'])}")
    
    # Generate test load to verify metrics
    for _ in range(100):
        requests.post('http://localhost:8000/api/v2/chunking/preview', 
                     json={'strategy': 'test', 'content': 'test'})
    
    time.sleep(10)
    
    # Check metrics updated
    result = requests.get(
        'http://localhost:9090/api/v1/query',
        params={'query': 'chunking_operations_total'}
    ).json()
    
    assert float(result['data']['result'][0]['value'][1]) > 100
    
    print("✓ Monitoring validated")
```

### Performance Baseline

```python
# Establish performance baseline after all fixes
def establish_baseline():
    results = {}
    
    # Test each strategy
    strategies = ['character', 'recursive', 'markdown', 
                 'semantic', 'hierarchical', 'hybrid']
    
    for strategy in strategies:
        times = []
        for _ in range(100):
            start = time.time()
            response = requests.post(
                'http://localhost:8000/api/v2/chunking/preview',
                json={
                    'strategy': strategy,
                    'content': 'x' * 10000,
                    'config': {'chunk_size': 500}
                }
            )
            times.append(time.time() - start)
        
        results[strategy] = {
            'p50': sorted(times)[50],
            'p95': sorted(times)[95],
            'p99': sorted(times)[99],
            'mean': sum(times) / len(times)
        }
    
    # Save baseline
    with open('performance_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Performance baseline:")
    for strategy, metrics in results.items():
        print(f"  {strategy}: p99={metrics['p99']:.3f}s")
    
    return results
```

### Success Criteria

#### Testing Requirements
- [ ] E2E tests cover all user workflows
- [ ] Load tests pass with 100 concurrent users
- [ ] Test coverage > 80% overall
- [ ] All strategies tested with various documents
- [ ] Error scenarios properly tested

#### Monitoring Requirements
- [ ] All metrics exposed to Prometheus
- [ ] Grafana dashboards configured
- [ ] Alerts configured and tested
- [ ] Logs aggregated and searchable
- [ ] Distributed tracing enabled

#### Operations Requirements
- [ ] Automated partition maintenance
- [ ] Backup procedures documented
- [ ] Runbook for common issues
- [ ] Performance baseline established
- [ ] SLAs defined and monitored

### Handoff to Production

#### Pre-Production Checklist
- [ ] All tests passing (unit, integration, E2E)
- [ ] Load test successful (100 users, < 5% errors)
- [ ] Security scan clean
- [ ] Monitoring dashboards operational
- [ ] Alerts tested and working
- [ ] Documentation complete
- [ ] Runbook created
- [ ] Team trained on operations

#### Production Deployment Plan
```bash
#!/bin/bash
# Production deployment script

# 1. Pre-deployment checks
./run_all_tests.sh || exit 1
./security_scan.sh || exit 1

# 2. Database backup
pg_dump semantik > backup_$(date +%Y%m%d_%H%M%S).sql

# 3. Deploy with canary
kubectl apply -f k8s/canary-10-percent.yaml
sleep 300  # Monitor for 5 minutes

# 4. Check metrics
python check_canary_metrics.py || kubectl rollback

# 5. Full deployment
kubectl apply -f k8s/production.yaml

# 6. Smoke tests
pytest tests/smoke/ || kubectl rollback

# 7. Monitor
watch -n 5 'kubectl get pods; curl -s localhost:9090/metrics | grep error'
```

## Notes for Orchestrating Agent

**Testing Philosophy**:
1. Test everything that could break
2. Automate everything that's repeated
3. Monitor everything that matters
4. Alert only on actionable issues
5. Document everything operational

**Quality Gates**:
- No deployment without tests
- No production without monitoring
- No operations without automation
- No alerts without runbooks
- No changes without metrics

**Final Validation**:
Run the entire user workflow manually before declaring complete. If you wouldn't trust it with your own data, it's not ready for production.

This phase ensures long-term stability and operational excellence. Without it, all previous work could be undermined by undetected issues.