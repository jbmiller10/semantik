# Semantik Test Coverage Plan (Revised)

## Executive Summary

This document outlines a comprehensive plan to improve test coverage across the Semantik codebase. The current test suite has good infrastructure but significant coverage gaps, particularly in critical components and integration points. This revised plan addresses architectural concerns, security testing, performance validation, and proper dependency ordering.

### Key Findings
- **Backend**: Strong pytest infrastructure with 0% coverage on many critical modules
- **Frontend**: No automated testing framework installed; only manual HTML tests exist
- **Architectural Issues**: `embedding_service.py` is in webui but used by vecpipe, creating dependency inversion
- **Critical Gaps**: WebSocket functionality, security testing, integration tests, performance benchmarks

### Recommended Approach
1. Address architectural issues first (refactor shared components)
2. Implement comprehensive testing strategies (unit, integration, property-based, load)
3. Target 80% coverage for critical paths, 70% overall within 3 months
4. Establish security and performance testing baselines

## Current State Assessment

### Backend Testing (Python/Pytest)
- **Framework**: Well-configured pytest with async support, coverage reporting, and fixtures
- **Coverage**: HTML and terminal reporting configured but many modules at 0%
- **Organization**: Clear separation of unit/integration/E2E tests
- **Gaps**: WebSocket testing, security validation, load testing

### Frontend Testing (React)
- **Framework**: None installed
- **Current Tests**: Manual HTML files and Python API test scripts
- **Coverage**: 0% automated test coverage
- **Needs**: Modern testing framework with component and integration testing

## Architectural Improvements Required

### Pre-Testing Refactors
1. **Move `embedding_service.py`** from webui to vecpipe or create shared package
2. **Abstract database access** in `cleanup.py` to remove direct SQLite dependency
3. **Create interfaces** for cross-package communication

## Priority Matrix (Revised)

### 🚨 Immediate (Week 0)
Architectural fixes and security issues that block proper testing.

### 🔴 Critical Priority (Week 1-2)
Core functionality, security-critical components, and testing infrastructure.

### 🟡 High Priority (Week 3-4)
Integration tests, performance validation, and user-facing features.

### 🟢 Medium Priority (Week 5-6)
Utilities, edge cases, and comprehensive coverage.

### 🔵 Future Enhancements
Advanced testing strategies and optimizations.

## Detailed Action Plan

### Phase 0: Architecture & Infrastructure (Week 0)

#### Ticket 0.1: Refactor embedding_service.py Location
**Priority**: 🚨 Immediate  
**Estimated Effort**: 1 day  
**Assignee**: Senior Backend Developer

**Description**: Move embedding_service.py to resolve dependency inversion.

**Tasks**:
1. Create `packages/shared/` directory or move to `packages/vecpipe/`
2. Update all imports in both packages
3. Ensure backward compatibility
4. Update Docker configurations if needed
5. Run existing tests to verify nothing breaks

**Acceptance Criteria**:
- No circular dependencies between packages
- All existing tests pass
- Both services can access embedding functionality

---

#### Ticket 0.2: Setup Testing Infrastructure
**Priority**: 🚨 Immediate  
**Estimated Effort**: 2 days  
**Assignee**: DevOps Engineer

**Description**: Set up comprehensive testing infrastructure including CI/CD.

**Tasks**:
1. Create `.github/workflows/test.yml` with matrix testing
2. Set up test databases for integration tests
3. Configure Docker compose for test environment
4. Install and configure:
   - pytest-benchmark for performance tests
   - pytest-hypothesis for property testing
   - locust for load testing
   - bandit for security scanning
5. Set up coverage aggregation across backend/frontend

**Acceptance Criteria**:
- CI runs on all PRs
- Test environment can be spun up with single command
- Coverage reports aggregate properly
- Security scans run automatically

---

### Phase 1: Foundation & Critical Components (Week 1-2)

#### Ticket 1.1: Frontend Testing Framework Setup
**Priority**: 🔴 Critical  
**Estimated Effort**: 1.5 days  
**Assignee**: Frontend Developer

**Description**: Install and configure Vitest with React Testing Library.

**Tasks**:
1. Install dependencies:
   ```bash
   npm install -D vitest @testing-library/react @testing-library/jest-dom 
   npm install -D @testing-library/user-event jsdom @vitest/ui
   npm install -D @testing-library/react-hooks msw @vitest/coverage-v8
   ```
2. Configure vitest.config.ts with coverage settings
3. Set up MSW for API mocking with handlers
4. Create test utilities for common patterns
5. Add snapshot testing support
6. Create testing documentation

**Acceptance Criteria**:
- All test types work (unit, integration, snapshot)
- Coverage reporting integrated with CI
- MSW intercepts all API calls
- Documentation helps new developers

---

#### Ticket 1.2: WebSocket Testing Suite
**Priority**: 🔴 Critical  
**Estimated Effort**: 2 days  
**Assignee**: Backend Developer

**Description**: Create comprehensive WebSocket testing for real-time features.

**Tasks**:
1. Create `tests/unit/test_websocket.py`
2. Test ConnectionManager class:
   - Multiple client connections
   - Message broadcasting
   - Connection/disconnection handling
   - Error scenarios
3. Test job progress WebSocket flow
4. Test file scanning WebSocket updates
5. Create WebSocket testing utilities
6. Add load tests for concurrent connections

**Acceptance Criteria**:
- All WebSocket functionality tested
- Concurrent connection limits verified
- Message delivery guaranteed
- >90% coverage of WebSocket code

---

#### Ticket 1.3: Security Testing Suite
**Priority**: 🔴 Critical  
**Estimated Effort**: 3 days  
**Assignee**: Security-focused Developer

**Description**: Implement comprehensive security testing.

**Tasks**:
1. Create `tests/security/` directory
2. Path traversal prevention tests:
   - File scanning boundaries
   - Document access restrictions
   - Symlink handling
3. Input validation tests:
   - SQL injection attempts
   - XSS in search results
   - Command injection in file paths
4. Authentication/Authorization tests:
   - Token manipulation
   - Privilege escalation attempts
   - Rate limiting effectiveness
5. Configure bandit and safety scans
6. Add OWASP dependency check

**Acceptance Criteria**:
- All OWASP Top 10 risks tested
- No high-severity vulnerabilities
- Rate limiting prevents abuse
- Security scans in CI pipeline

---

#### Ticket 1.4: Search API Comprehensive Testing
**Priority**: 🔴 Critical  
**Estimated Effort**: 2.5 days  
**Assignee**: Backend Developer

**Description**: Test search_api.py with advanced strategies.

**Tasks**:
1. Unit tests in `tests/unit/test_search_api.py`:
   - All endpoints with edge cases
   - Error handling and retries
   - Request validation
2. Property-based tests with Hypothesis:
   - Random query generation
   - Fuzzing inputs
   - Invariant checking
3. Load tests with Locust:
   - Concurrent search requests
   - Large result sets
   - Memory pressure scenarios
4. Contract tests for webui integration

**Acceptance Criteria**:
- >85% unit test coverage
- Property tests find no crashes
- Load tests establish baselines
- Contract tests ensure compatibility

---

### Phase 2: Integration & E2E Testing (Week 2-3)

#### Ticket 2.1: Job Processing Pipeline E2E Tests
**Priority**: 🔴 Critical  
**Estimated Effort**: 3 days  
**Assignee**: Senior Backend Developer

**Description**: Test complete job processing flow end-to-end.

**Tasks**:
1. Create `tests/e2e/test_job_pipeline.py`
2. Test complete flows:
   - Job creation → scanning → chunking → embedding → storage
   - Error recovery at each stage
   - Job cancellation mid-process
   - Concurrent job processing
3. Test data consistency:
   - SQLite ↔ Qdrant synchronization
   - File tracking accuracy
   - Progress reporting accuracy
4. Performance benchmarks:
   - Processing time per file size
   - Memory usage patterns
   - GPU utilization

**Acceptance Criteria**:
- All happy paths tested
- Error recovery verified
- Performance baselines established
- Data consistency guaranteed

---

#### Ticket 2.2: Authentication Flow Testing
**Priority**: 🟡 High  
**Estimated Effort**: 2 days  
**Assignee**: Backend Developer

**Description**: Comprehensive auth testing beyond unit tests.

**Tasks**:
1. Unit tests for `auth.py` components
2. Integration tests:
   - Login → token → refresh flow
   - Concurrent token usage
   - Token expiration handling
   - Multi-device sessions
3. Security tests:
   - Brute force protection
   - Token tampering
   - Session fixation
4. Load tests for auth endpoints

**Acceptance Criteria**:
- All auth flows tested E2E
- Security vulnerabilities addressed
- Performance under load verified
- >90% coverage (security critical)

---

#### Ticket 2.3: React Component Testing Suite
**Priority**: 🟡 High  
**Estimated Effort**: 3 days  
**Assignee**: Frontend Developer

**Description**: Test critical React components thoroughly.

**Tasks**:
1. CreateJobForm testing:
   - Form validation
   - Directory scanning
   - WebSocket integration
   - Error handling
2. SearchInterface testing:
   - Search parameter updates
   - Result rendering
   - Collection filtering
   - Pagination
3. DocumentViewer testing:
   - File type detection
   - Rendering different formats
   - Error states
4. Zustand store testing:
   - All state mutations
   - Async actions
   - Store interactions

**Acceptance Criteria**:
- Critical components >80% coverage
- User interactions tested
- Error states handled
- Snapshot tests for UI consistency

---

### Phase 3: Advanced Testing Strategies (Week 3-4)

#### Ticket 3.1: Property-Based Testing Implementation
**Priority**: 🟡 High  
**Estimated Effort**: 2 days  
**Assignee**: Backend Developer

**Description**: Add Hypothesis tests for data processing.

**Tasks**:
1. Document chunking properties:
   - Text preservation
   - Chunk size constraints
   - Overlap consistency
2. Vector operations:
   - Normalization properties
   - Similarity calculations
3. Search query processing:
   - Query parsing robustness
   - Filter construction
4. File path handling:
   - Path resolution safety
   - Unicode handling

**Acceptance Criteria**:
- No property violations found
- Edge cases discovered and fixed
- Tests run in reasonable time
- Documentation for adding new properties

---

#### Ticket 3.2: Load Testing Suite
**Priority**: 🟡 High  
**Estimated Effort**: 2.5 days  
**Assignee**: Performance Engineer

**Description**: Establish performance baselines and limits.

**Tasks**:
1. Create Locust test scenarios:
   - Search load patterns
   - Concurrent job processing
   - Mixed workload simulation
2. Benchmark critical operations:
   - Embedding generation throughput
   - Search response times
   - File processing rates
3. Stress testing:
   - Memory exhaustion scenarios
   - GPU saturation
   - Qdrant connection limits
4. Create performance dashboard

**Acceptance Criteria**:
- Performance baselines documented
- Bottlenecks identified
- Scaling limits established
- Monitoring alerts configured

---

#### Ticket 3.3: Contract Testing Between Services
**Priority**: 🟡 High  
**Estimated Effort**: 1.5 days  
**Assignee**: Backend Developer

**Description**: Ensure vecpipe and webui maintain compatibility.

**Tasks**:
1. Define API contracts with OpenAPI
2. Implement consumer tests in webui
3. Implement provider tests in vecpipe
4. Set up contract verification in CI
5. Version compatibility matrix

**Acceptance Criteria**:
- All API endpoints have contracts
- Breaking changes detected automatically
- Backward compatibility verified
- Documentation auto-generated

---

### Phase 4: Comprehensive Coverage (Week 4-5)

#### Ticket 4.1: Database Layer Complete Testing
**Priority**: 🟢 Medium  
**Estimated Effort**: 2 days  
**Assignee**: Backend Developer

**Description**: Expand database.py testing for all operations.

**Tasks**:
1. Test all CRUD operations thoroughly
2. Transaction testing:
   - Rollback scenarios
   - Concurrent access
   - Deadlock prevention
3. Migration testing:
   - Forward/backward compatibility
   - Data preservation
   - Schema validation
4. Performance optimization tests

**Acceptance Criteria**:
- All database operations tested
- Concurrent access safe
- Migrations reversible
- >85% coverage

---

#### Ticket 4.2: GPU/Model Management Testing
**Priority**: 🟢 Medium  
**Estimated Effort**: 2 days  
**Assignee**: ML Engineer

**Description**: Test GPU-dependent code properly.

**Tasks**:
1. Multi-level testing approach:
   - Unit tests with mocked GPU
   - Integration tests with GPU simulation
   - Real GPU tests (when available)
2. Memory management tests:
   - Model loading/unloading
   - OOM handling
   - Quantization fallbacks
3. Performance benchmarks:
   - Inference speeds
   - Batch size optimization
   - Memory usage patterns

**Acceptance Criteria**:
- Works in CPU-only environments
- Graceful degradation on OOM
- Performance metrics tracked
- >80% coverage

---

#### Ticket 4.3: Utility and Infrastructure Testing
**Priority**: 🟢 Medium  
**Estimated Effort**: 1.5 days  
**Assignee**: Backend Developer

**Description**: Test remaining utilities and infrastructure.

**Tasks**:
1. QdrantManager singleton testing
2. Retry decorator comprehensive tests
3. File tracking and cleanup utilities
4. Memory utilities and calculations
5. Search utilities and helpers

**Acceptance Criteria**:
- All utilities have tests
- Edge cases covered
- Thread safety verified
- >85% coverage

---

### Phase 5: Testing Infrastructure & Documentation (Week 5-6)

#### Ticket 5.1: CI/CD Pipeline Enhancement
**Priority**: 🟡 High  
**Estimated Effort**: 1.5 days  
**Assignee**: DevOps Engineer

**Description**: Finalize CI/CD with all test types.

**Tasks**:
1. Matrix testing for multiple Python/Node versions
2. Parallel test execution
3. Test result visualization
4. Coverage trend tracking
5. Performance regression detection
6. Security scan integration

**Acceptance Criteria**:
- All test types run automatically
- Results easy to interpret
- Trends visible over time
- Alerts for regressions

---

#### Ticket 5.2: Testing Documentation & Guidelines
**Priority**: 🟢 Medium  
**Estimated Effort**: 1 day  
**Assignee**: Technical Writer

**Description**: Create comprehensive testing documentation.

**Tasks**:
1. Testing philosophy and standards
2. How to write different test types
3. Testing utilities documentation
4. Performance testing guide
5. Security testing checklist
6. Troubleshooting guide

**Acceptance Criteria**:
- New developers can write tests easily
- Best practices documented
- Examples for each test type
- Troubleshooting steps clear

---

## Success Metrics

### Coverage Goals
- **Month 1**: 60% overall coverage, 85% for critical paths
- **Month 2**: 70% overall coverage, 90% for critical paths
- **Month 3**: 75% overall coverage, 95% for critical paths

### Quality Metrics
- Zero high-severity security vulnerabilities
- Response time p95 < 200ms for search
- WebSocket message delivery > 99.9%
- Test execution time < 10 minutes for unit tests
- Zero flaky tests after stabilization

### Performance Baselines
- Search: 1000 requests/second sustained
- Embedding: 500 documents/minute
- Job processing: 10GB/hour
- Concurrent users: 100 without degradation

## Testing Strategy Summary

### Test Types Distribution
- **Unit Tests**: 40% - Fast, isolated, high coverage
- **Integration Tests**: 25% - Service interactions, API contracts
- **E2E Tests**: 15% - Critical user journeys
- **Property Tests**: 10% - Edge case discovery
- **Performance Tests**: 5% - Baseline validation
- **Security Tests**: 5% - Vulnerability prevention

### Key Improvements Over Original Plan
1. **Architectural fixes** before testing to avoid technical debt
2. **Security testing** as first-class concern
3. **WebSocket testing** for real-time features
4. **Property-based testing** for robustness
5. **Load testing** for performance validation
6. **Contract testing** for service compatibility
7. **More realistic timelines** and effort estimates

## Risk Mitigation

### Identified Risks
1. **Architectural refactoring delays**: Mitigate by keeping changes minimal
2. **Test maintenance burden**: Mitigate with good abstractions and utilities
3. **Performance test flakiness**: Mitigate with controlled environments
4. **Security test false positives**: Mitigate with proper configuration

### Contingency Plans
- If refactoring blocked, implement adapter pattern temporarily
- If timeline slips, focus on critical path coverage first
- If resources limited, combine similar tickets
- If tests slow, investigate parallelization and optimization

## Resource Requirements

### Team Allocation
- 1 Senior Backend Developer (6 weeks, 80% allocation)
- 1 Senior Frontend Developer (4 weeks, 80% allocation)
- 1 DevOps Engineer (1 week total, spread across phases)
- 1 Security-focused Developer (1 week for security testing)
- 1 Performance Engineer (3 days for load testing)

### Infrastructure
- GitHub Actions compute time (increased)
- GPU-enabled CI runners for model tests
- Load testing infrastructure
- Security scanning tools licenses

## Next Steps

1. **Immediate**: Begin architectural refactoring (Week 0)
2. **Week 1**: Start critical testing infrastructure and security tests
3. **Week 2-3**: Focus on integration and E2E tests
4. **Week 4-5**: Implement advanced testing strategies
5. **Week 6**: Finalize CI/CD and documentation
6. **Ongoing**: Monitor metrics and maintain test quality

## Appendix: Testing Best Practices

### Code Examples

#### Property-Based Testing Example
```python
from hypothesis import given, strategies as st

@given(
    text=st.text(min_size=100, max_size=10000),
    chunk_size=st.integers(min_value=50, max_value=500),
    overlap=st.integers(min_value=0, max_value=50)
)
def test_chunking_preserves_text(text, chunk_size, overlap):
    chunks = chunk_text(text, chunk_size, overlap)
    # Verify all text is preserved
    reconstructed = "".join(chunk.text for chunk in chunks)
    assert text in reconstructed
```

#### WebSocket Testing Example
```python
async def test_websocket_broadcast():
    manager = ConnectionManager()
    
    # Connect multiple clients
    clients = [MockWebSocket() for _ in range(5)]
    for client in clients:
        await manager.connect(client, job_id="test-job")
    
    # Broadcast message
    await manager.broadcast_to_job("test-job", {"status": "processing"})
    
    # Verify all clients received
    for client in clients:
        assert client.sent_messages[-1] == {"status": "processing"}
```

#### Contract Testing Example
```python
# Consumer test (webui)
@pact.given("search service is available")
@pact.upon_receiving("a search request")
@pact.with_request(method="POST", path="/search")
@pact.will_respond_with(200, body=Like({
    "results": EachLike({"content": String(), "score": Float()}),
    "total": Integer()
}))
def test_search_contract(client):
    response = client.post("/search", json={"query": "test"})
    assert response.status_code == 200
```

This revised plan addresses the critical gaps identified and provides a more comprehensive approach to testing the Semantik codebase.