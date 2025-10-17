# Test Quality Tracking Document

**Date**: 2025-10-16
**Review Scope**: 21 test files + 1 script suite (manual harnesses), ~420 test cases
**Status**: 🔴 Critical Issues Identified

---

## Executive Summary

| Metric | Value | Target |
|--------|-------|--------|
| Files Reviewed | 21 | - |
| Critical Issues | 25 | 0 |
| Moderate Issues | 36 | <5 |
| Minor Issues | 28 | <10 |
| Files Needing Immediate Action | 5 | 0 |
| Files Needing Deletion | 1 | 0 |
| Estimated False Confidence | 40-60% | <10% |

---

## Priority Matrix

### P0 - Critical (Do This Week)

| File | Issue | Impact | Effort | Status |
|------|-------|--------|--------|--------|
| `tests/test_auth.py` | Delete entirely - 0 assertions, test pollution | High | 5 min | ✅ Completed (2025-10-16) |
| `tests/unit/test_collection_repository.py` | Mock everything - false confidence | Critical | 2-3 days | ✅ Completed (2025-10-16) |
| `tests/unit/test_collection_service.py` | 904 lines, excessive mocking | Critical | 3-4 days | ✅ Completed (2025-10-16) |
| `tests/test_metrics.py` (cluster) | Manual scripts in test tree, no assertions, hit live services | High | 0.5 day | ⏳ Pending |
| `tests/test_reranking_e2e.py` | Placeholder asserts, never exercises reranking flow | High | 1-2 days | ⏳ Pending |
| `tests/api/test_rate_limits.py` | CI skips entire suite (`@pytest.mark.skipif(CI==\"true\")`), no automated coverage | Critical | 1 day | ⏳ Pending |

### P1 - High (Next 2 Weeks)

| File | Issue | Impact | Effort | Status |
|------|-------|--------|--------|--------|
| `tests/unit/test_all_chunking_strategies.py` | Wrong location, 78 test permutations | High | 1 day | ✅ Completed (2025-10-16) |
| `tests/e2e/test_websocket_integration.py` | Giant tests, code duplication | High | 2 days | ⏳ Pending |
| `tests/test_embedding_integration.py` | Mocks defeat purpose, wrong location | Medium | 1 day | ⏳ Pending |
| `tests/webui/test_tasks_helpers.py` & `_original.py` | Duplicate suites diverging, redundant coverage | Medium | 1 day | ⏳ Pending |
| `tests/websocket/*` & `tests/webui/test_chunking_websocket.py` | Real Redis + sleeps; flaky timing in "unit" suites | High | 2-3 days | ⏳ Pending |
| `tests/webui/test_ingestion_chunking_integration.py` | Labeled integration but fully mocked; asserts internal details | High | 2 days | ⏳ Pending |
| `tests/webui/services/test_collection_service.py` | 1k+ line mock suite duplicating service logic | High | 2-3 days | ⏳ Pending |
| `tests/webui/api/v2/test_chunking.py` (plus simple/direct variants) | Huge mock-based API suite duplicating integration tests | High | 3-4 days | ⏳ Pending |
| `tests/webui/services/test_search_service.py` | Extensive mocks instead of integration coverage | High | 2-3 days | ⏳ Pending |
| `tests/webui/test_websocket_manager.py` | Manipulates global singleton, heavy cleanup logic, brittle | High | 2 days | ⏳ Pending |
| `packages/shared/chunking/infrastructure/streaming/test_memory_pool.py` | Relies on real sleeps/threading → flaky leak checks | High | 1 day | ⏳ Pending |
| `tests/application/test_*_use_case.py` & `packages/webui/tests/test_collection_service_chunking_validation.py` | Mock-heavy duplicates of service logic | High | 2-3 days | ⏳ Pending |
| `tests/webui/test_chunking_metrics.py` | Direct Prometheus `_metrics` mutation, mocks service internals | Medium | 1 day | ✅ Completed (2025-10-17) – coverage relocated to `tests/integration/chunking/test_ingestion_metrics.py` with isolated registries |
| `tests/unit/test_hierarchical_chunker.py` & related chunker suites | Heavy mocking of llama-index internals, oversized permutations | Medium | 2 days | ✅ Completed (2025-10-17) – replaced by `tests/unit/chunking/test_hierarchical_chunker_validations.py` and extended coverage in `tests/integration/strategies/` |
| `tests/unit/test_search_service.py` | HTTP mocked, duplicates API coverage | Medium | 1-2 days | ⏳ Pending |
| `tests/webui/api/v2/test_operations.py` | Mock-based API suite overlapping integration tests | High | 2 days | ⏳ Pending |
| `tests/webui/api/v2/test_chunking_simple_integration.py` & `_direct.py` | “Integration” files but fully mocked | High | 2 days | ⏳ Pending |
| `tests/webui/test_tasks_websocket_integration.py` | Extensive patching of Redis/tasks, duplicates websocket suites | High | 2 days | ⏳ Pending |
| `tests/integration/test_websocket_redis_integration.py` | Uses custom Redis mock instead of real fixture, duplicates unit tests | Medium | 1-2 days | ⏳ Pending |
| `tests/webui/test_document_chunk_count_updates.py` | Mock-based Celery ingestion, duplicates integration suite | High | 2 days | ✅ Completed (2025-10-17) – integrated into `tests/integration/chunking/test_ingestion_metrics.py` with real DB + redis fixtures |
| `tests/integration/chunking/test_ingestion_metrics.py` | Recursive strategy still falls back to TokenChunker in local env, blocking assertions | Medium | 1 day | ⚠️ Blocked (2025-10-17) – investigate ChunkingService config validation vs. strategy factory so recursive runs without fallback |
| `tests/webui/api/v2/test_collections.py` | Mocking collection service duplicates integration coverage | High | 2 days | ⏳ Pending |
| `tests/webui/api/v2/test_collections_operations.py` | Mock-based listing endpoints overlapping integration | Medium | 1-2 days | ⏳ Pending |
| `tests/webui/services/dtos/test_chunking_dtos.py` | DTO conversions retesting Pydantic outputs | Medium | 1 day | ⏳ Pending |
| `tests/unit/test_hierarchical_chunker_extended.py` | More llama-index patches, redundant | Medium | 2 days | ✅ Completed (2025-10-17) – consolidated into lean validation/unit coverage and strategy integrations |
| `tests/unit/test_hybrid_chunker.py` | Large suite patching chunking factory; overlaps other chunker tests | Medium | 2 days | ✅ Completed (2025-10-17) – behavior covered via integration strategy tests and slim unit validations |
| `tests/domain/test_chunking_operation.py` | Heavy MagicMock usage, duplicates domain tests | Medium | 1-2 days | ⏳ Pending |
| `tests/domain/test_value_objects.py` | Over-assertive unit tests, could slim | Medium | 1 day | ⏳ Pending |
| `tests/unit/test_all_chunking_strategies_unified.py` | Massive cross-strategy suite, duplicative | Medium | 2 days | ✅ Completed (2025-10-17) – scenarios moved into targeted integration suites (`tests/integration/strategies/` & `tests/integration/chunking/`) |
| `tests/streaming/validate_streaming_pipeline.py` | Manual validation script with prints | High | 1 day | ⏳ Pending |
| `tests/unit/test_chunking_error_metrics.py` | Direct Prometheus internals, duplicate metrics suite | Medium | 1 day | ✅ Completed (2025-10-17) – error counters verified via `tests/integration/chunking/test_ingestion_metrics.py` isolated registry fixture |
| `tests/unit/test_chunking_tasks.py` | Mock-heavy Celery tasks, overlaps integration | High | 2 days | ✅ Completed (2025-10-17) – core chunk count/assertions folded into service integration flow; Celery orchestration follow-ups tracked separately |
| `tests/chunking/streaming/test_streaming_strategies.py` | Large streaming strategy suite using real implementations | Medium | 2 days | ✅ Completed (2025-10-17) – replaced by targeted streaming coverage in `tests/integration/chunking/test_streaming_strategies.py` |
| `tests/unit/test_chunking_exceptions.py` | Mock-based exception tests; behavior covered elsewhere | Medium | 1 day | ⏳ Pending |
| `tests/unit/test_shared_qdrant_manager.py` | Patches Redis/Qdrant internals; duplicate coverage | Medium | 1 day | ⏳ Pending |
| `tests/integration/test_chunking_error_flow.py` | Custom mocks, duplicates error handling coverage | Medium | 1-2 days | ⏳ Pending |
| `tests/unit/test_resource_manager.py` | Mocked repositories/psutil, duplicates service behavior | Medium | 1 day | ⏳ Pending |
| `tests/webui/services/test_progressive_segmentation.py` | Extensive mocks for segmentation logic | Medium | 1-2 days | ⏳ Pending |
| `tests/unit/test_models.py` | Massive SQLAlchemy model suite with mocks | Medium | 2 days | ⏳ Pending |
| `tests/unit/test_model_manager.py` | Mock-based tests for model manager internals | Medium | 1-2 days | ⏳ Pending |
| `tests/websocket/test_cleanup.py` | Manual cleanup script with real sockets | Medium | 1 day | ⏳ Pending |
| `tests/websocket/test_scaling.py` | Load-style script, relies on live services | Medium | 1 day | ⏳ Pending |
| `tests/webui/services/test_directory_scan_service.py` | Heavy filesystem/mocking, overlaps service logic | Medium | 2 days | ⏳ Pending |
| `tests/webui/services/test_execute_ingestion_chunking.py` | Unit suite overlapping integration coverage | Medium | 1-2 days | ✅ Completed (2025-10-17) – replaced by service-level assertions in `tests/integration/chunking/test_ingestion_metrics.py` |
| `tests/unit/test_document_scanning_service.py` | Mocked DB/session duplication of integration tests | Medium | 1 day | ⏳ Pending |
| `tests/unit/test_directory_scan_service.py` | Mocked session verifies implementation, duplicates service tests | Medium | 1 day | ⏳ Pending |
| `tests/webui/services/test_execute_ingestion_chunking.py` | Unit suite overlapping integration coverage | Medium | 1-2 days | ✅ Completed (2025-10-17) – replaced by service-level assertions in `tests/integration/chunking/test_ingestion_metrics.py` |

| `tests/unit/test_chunking_service.py` | Large mock-heavy suite duplicating integration coverage | High | 2 days | ⏳ Pending |
| `tests/e2e/test_websocket_performance.py` | Performance script requiring live stack | High | 1-2 days | ⏳ Pending |
| `tests/e2e/test_websocket_reindex.py` | Giant E2E suite with duplicate coverage | High | 2 days | ⏳ Pending |
| `tests/unit/test_auth.py` | Legacy mock-based auth tests overlapping newer coverage | Medium | 1 day | ⏳ Pending |
| `tests/webui/services/test_partition_monitoring_service.py` | Mock-heavy monitoring suite | Medium | 1-2 days | ⏳ Pending |
| `tests/unit/test_auth_repository.py` | Mock session tests; need DB integration | Medium | 1 day | ⏳ Pending |
| `tests/test_embedding_oom_handling.py` | Manual script hitting real resources | Medium | 1 day | ⏳ Pending |
| `tests/unit/test_chunk_repository.py` | Mock-only repository coverage | Medium | 1 day | ⏳ Pending |
| `tests/unit/test_partition_utils.py` | Helper tests duplicating production logic | Medium | 1 day | ⏳ Pending |
| `tests/performance/chunking_benchmarks.py` | Performance script without assertions | High | 1 day | ⏳ Pending |
| `tests/webui/api/v2/test_directory_scan.py` | Mocked API coverage overlapping service tests | Medium | 1-2 days | ⏳ Pending |
| `tests/unit/test_chunking_exception_handlers.py` | Mock-based error handler tests | Medium | 1 day | ⏳ Pending |
| `tests/unit/test_operation_service.py` | Mock-heavy service suite | Medium | 1-2 days | ⏳ Pending |
| `tests/unit/test_chunking_security.py` | Overlapping security tests with mocks | Medium | 1-2 days | ⏳ Pending |
| `tests/webui/test_websocket_race_conditions.py` | Manual load script with no assertions | Medium | 1 day | ⏳ Pending |
| `tests/integration/test_search_reranking_integration.py` | Overlapping integration coverage | Medium | 1-2 days | ⏳ Pending |
| `tests/security/test_redos_prevention.py` | Mocked regex tests needing refactor | Medium | 1 day | ⏳ Pending |
| `tests/unit/test_all_chunking_strategies_migrated.py` | Redundant large suite | Medium | 2 days | ✅ Completed (2025-10-17) – coverage consolidated into strategy integration smoke tests |
| `tests/webui/test_cleanup_tasks.py` | Mocked cleanup flow | Medium | 1 day | ⏳ Pending |
| `tests/integration/test_exception_translation.py` | Overlaps domain coverage | Medium | 1 day | ⏳ Pending |
| `tests/integration/test_embedding_gpu_memory.py` | GPU-dependent script | Medium | 1 day | ⏳ Pending |
### P2 - Medium (Within Month)

| File | Issue | Impact | Effort | Status |
|------|-------|--------|--------|--------|
| `tests/unit/test_models.py` | Large ORM mock suite | Medium | 2 days | ⏳ Pending |
| `tests/unit/test_model_manager.py` | Mocked model lifecycle | Medium | 1-2 days | ⏳ Pending |
| `tests/websocket/test_cleanup.py` | Manual cleanup script | Medium | 1 day | ⏳ Pending |
| `tests/websocket/test_scaling.py` | Load testing script | Medium | 1 day | ⏳ Pending |
| `tests/integration/test_collection_deletion.py` | Excessive mocking, implementation testing | Medium | 1 day | ⏳ Pending |
| `tests/webui/services/test_search_service_reranking.py` | Mocks HTTP layer | Medium | 1-2 days | ⏳ Pending |
| `tests/streaming/test_streaming_integration.py` | Mocks domain entities | Medium | 1-2 days | ⏳ Pending |
| `tests/security/test_path_traversal.py` | Flaky performance test | Low | 4 hours | ⏳ Pending |
| `tests/domain/test_chunking_strategies.py` | Placeholder assertions (`assert True`), weak metadata checks | Medium | 1 day | ⏳ Pending |
| `tests/unit/test_search_api.py` | Mutates module singletons (`search_api_module.*`), heavy mocking, brittle cleanup | Medium | 1-2 days | ⏳ Pending |
| `tests/unit/test_search_api_edge_cases.py` | Same singleton patching, duplicates main suite, mocks internals | Medium | 1-2 days | ⏳ Pending |
| `tests/unit/test_rate_limiter.py` | Touches private limiter state, environment-dependent behavior | Medium | 1 day | ⏳ Pending |
| `tests/unit/test_user_repository.py` | Mock-only repository tests duplicating integration coverage | Medium | 1 day | ⏳ Pending |
| `tests/unit/test_api_key_repository.py` | Mock session verifications instead of DB-backed tests | Medium | 1 day | ⏳ Pending |
| `tests/unit/test_operation_repository.py` | Mocked UoW tests instead of real transactions | Medium | 1-2 days | ⏳ Pending |

---

## Detailed Issue Tracking

### File: `tests/test_auth.py` ❌ DELETE

**Grade**: F (2/10)
**Status**: ✅ Completed (2025-10-16) - Removed redundant HTTP smoke script
**Location**: `/home/john/semantik/tests/test_auth.py`

#### Critical Issues
- [x] **No Assertions** (P0) - File has zero assertions, only print statements
  - Lines: 28-70
  - Impact: Test always passes, provides no validation
  - Action: Delete file entirely

- [x] **Test Pollution** (P0) - Creates persistent user "testuser" without cleanup
  - Lines: 20-34
  - Impact: Second test run fails, test order dependencies
  - Action: Delete file (covered by other tests)

- [x] **Redundant** (P0) - Functionality covered by existing tests
  - Better coverage in: `tests/unit/test_auth.py` (623 lines)
  - Better coverage in: `tests/integration/test_auth_api.py` (322 lines)
  - Action: `git rm tests/test_auth.py`

**Action Items**:
1. ✅ Verify coverage: `pytest tests/unit/test_auth.py tests/integration/test_auth_api.py -v`
2. ✅ Delete: `git rm tests/test_auth.py`
3. 🔜 Fold into test quality cleanup commit after additional refactors

---

### File: `tests/unit/test_collection_repository.py`

**Grade**: D (3/10)
**Status**: ✅ Completed (2025-10-16) - Reauthored as integration suite with real DB
**Location**: `/home/john/semantik/tests/unit/test_collection_repository.py`

#### Critical Issues

- [x] **Conjoined Twins** (P0) - Mock integration tests, not unit tests
  - Lines: 23-477 (entire file)
  - Impact: Tests verify mock behavior, not real repository logic
  - Effort: 2-3 days
  - Action: Rewrite as integration tests with real database
  - Target: `tests/integration/repositories/test_collection_repository.py`

- [x] **Mockery** (P0) - Mocks critical business logic (UUID generation)
  - Lines: 72-73
  - Impact: Hides potential bugs, brittle tests
  - Action: Use real UUID generation

- [x] **Dodger** (P0) - Tests mock assertions instead of behavior
  - Lines: 88-89, 159, 305-306, 337, 381, 439-440, 465
  - Example: `mock_session.add.assert_called_once()`
  - Impact: Tests mock implementation, not actual functionality
  - Action: Remove all mock assertions, verify actual database state

#### Moderate Issues

- [x] **Free Ride** (P1) - Multiple unrelated validations in single test
  - Lines: 106-130
  - Impact: Unclear failure messages
  - Action: Split into parameterized tests

- [x] **Missing Coverage** (P1) - No tests for `update()` method
  - Lines: N/A (method exists in production, no tests)
  - Impact: Critical functionality untested
  - Action: Add comprehensive update tests

- [x] **Leaky Mocks** (P2) - Mutable fixture state between tests
  - Lines: 40-56
  - Impact: Potential test interference
  - Action: Use factory fixtures

#### Minor Issues

- [x] **Wrong Type Hints** (P2) - Fixtures return `None` instead of mock type
  - Lines: 21-51
  - Action: Fix return types for IDE/mypy support

**Action Items**:
1. ✅ Create `tests/integration/repositories/` directory
2. ✅ Rewrite tests using `async_session` fixture
3. ✅ Add missing `update()` / stats coverage
4. ✅ Delete old mock-based file
5. ✅ Verify: `uv run pytest tests/integration/repositories/test_collection_repository.py -q`

**Resolution Summary**:
- Introduced `tests/integration/repositories/test_collection_repository.py` backed by the real PostgreSQL session.
- Exercised creation, retrieval, listing (public/private), rename, update_stats, update field mutations, and permission checks.
- Added UUID-suffixed names to avoid cross-test collisions and aligned negative assertions with `DatabaseOperationError` / `EntityAlreadyExistsError`.
- Ensured cleanup relies on transaction rollback via existing fixture isolation.

---

### File: `tests/unit/test_collection_service.py`

**Grade**: N/A (removed)
**Status**: ✅ Completed (2025-10-16) - Replaced by integration suite
**Location**: _File deleted_; replacement coverage in `/home/john/semantik/tests/integration/services/test_collection_service.py`

#### Critical Issues

- [ ] **Mockery** (P0) - Mocks all repositories instead of using real ones
  - Lines: 22-51, throughout entire file
  - Impact: Tests mock configuration, not service orchestration
  - Effort: 3-4 days
  - Action: Convert to integration tests, only mock external services (Celery, Qdrant)

- [ ] **Giant Tests** (P0) - Extremely long test methods with complex setup
  - Lines: 55-132 (78 lines), 772-849 (77 lines)
  - Impact: Hard to read, maintain, debug
  - Action: Extract mock factories, split into focused tests

- [ ] **Dodger** (P0) - Tests implementation details (mock method calls)
  - Lines: 100-112, 115-120, 257-260, 378-383
  - Example: `mock_collection_repo.create.assert_called_once_with(...)`
  - Impact: Brittle, doesn't test outcomes
  - Action: Remove assertions on mock calls, verify database state

#### Moderate Issues

- [ ] **Code Duplication** (P1) - Mock setup repeated across 20+ tests
  - Lines: 59-90, 139-170, 224-248, 754-818
  - Impact: High maintenance burden
  - Effort: 1 day
  - Action: Create factory fixtures (see ACTION_PLAN.md)

#### Action Items

1. ✅ Create `tests/integration/services/test_collection_service.py`
2. ✅ Move all service scenarios to use real repositories and database fixtures
3. ✅ Stub external integrations (Celery, Qdrant) inside integration tests for determinism
4. ✅ Delete old mock-heavy unit file after migration

**Progress Notes (2025-10-16)**
- Introduced `tests/integration/services/test_collection_service.py` covering create/delete/update flows, `add_source`, `remove_source`, `reindex_collection`, manual operations, and list APIs with real persistence.
- Added negative-path assertions for invalid states and permission checks, mirroring production constraints.
- Provisioned per-test users via a local `user_factory` fixture to avoid PK collisions while exercising permission logic.
- Removed `tests/unit/test_collection_service.py`; all former coverage now lives in the integration suite and real fixtures.
- Remaining action for future work: continue with P1/P2 items (chunking tests, websocket e2e cleanup, etc.) listed elsewhere in this document.

**Code Examples**: See ACTION_PLAN.md Section "Action 3"

---

### Cluster: Manual Script-Style Tests (`tests/test_metrics.py`, `tests/test_metrics_update.py`, `tests/test_search.py`, `tests/test_embedding_performance.py`, `tests/test_embedding_full_integration.py`, `tests/streaming/validate_streaming_pipeline.py`, `apps/webui-react/tests/api_test_suite.py`)

**Grade**: F (1/10)  
**Status**: ⏳ Pending - relocate or rewrite

These files live under `tests/` (and CI executes them) but behave like manual diagnostics:
- `tests/test_metrics.py:14-95` and `tests/test_metrics_update.py:12-42` issue HTTP requests, print responses, never assert.
- `tests/test_search.py:12-46` performs interactive login/search with only `print` statements.
- `tests/test_embedding_performance.py:24-200` runs benchmarks with `ThreadPoolExecutor`, `psutil`, and `time.sleep`, returning dicts but asserting nothing.
- `tests/test_embedding_full_integration.py` mirrors the pattern for full embedding flows.
- `tests/streaming/validate_streaming_pipeline.py` creates 100MB temp files, runs async loops, prints validation summaries, and never asserts.
- `apps/webui-react/tests/api_test_suite.py:1-140` is an async smoke harness using `aiohttp` and `websockets`, again without assertions.

**Critical Issues**
- **No Assertions** (P0): pytest marks them passing regardless of behavior.
- **External Dependencies** (P0): Expect full stack on `localhost:8080`, `9092`, websocket endpoints; risk of hangs locally/CI.
- **Runtime Drag** (P1): Benchmarks and sleeps slow down test runs.

**Recommended Fix**
1. Relocate scripts to `manual_tests/` (or similar) and add `pytest.ini` ignore.
2. Open follow-up tickets to implement proper automated coverage for metrics, search, and embedding flows.
3. Document manual run instructions (see ACTION_PLAN.md Action 3).

**Verification**
- After relocation, ensure no script is picked up by pytest discovery (`uv run pytest tests/`).
- Provide manual README covering how to run the diagnostic scripts when needed.

---

### File: `tests/test_reranking_e2e.py`

**Grade**: F (1/10)  
**Status**: ⏳ Pending - needs full reranking integration coverage  
**Location**: _File deleted_; current automated coverage gap persists

#### Findings
- Previous placeholder suite consisted only of `assert True` statements referencing code inspection; it never exercised WebUI → VecPipe reranking.
- Placeholder file has now been removed, leaving no automated end-to-end reranking coverage.
- Existing `tests/integration/test_search_reranking_integration.py` still relies on mocked `SearchService`; reranking parameters are not validated against a real VecPipe flow.

#### Action Plan
1. Implement an async integration test using existing fixtures that issues a real search request with reranking enabled, ensuring WebUI forwards parameters to VecPipe.
2. Stub external dependencies only where unavoidable (e.g., reranker model), and assert reranking metrics on the response.
3. Once genuine coverage exists, document the authoritative test location and delete any remaining placeholder scaffolding.

**Effort**: 1-2 days coordinating with Search team fixtures.  
**Dependencies**: Requires deterministic reranker stub or fixture.

---

### Files: `tests/webui/test_tasks_helpers.py` & `tests/webui/test_tasks_helpers_original.py`

**Grade**: D (4/10)  
**Status**: ⏳ Pending - consolidation required

**Issues**
- Duplicate suites diverge (e.g., nested sanitization assertions differ at lines `tests/webui/test_tasks_helpers.py:64` vs `_original.py:63`).
- Redundant helper factories and fixtures.
- Mock-heavy assertions (`assert_called_once`) dominate, offering little behavioral confidence.

**Recommended Steps**
1. Merge valued scenarios into a single file.
2. Remove redundant module and update references.
3. While consolidating, shift from call-count assertions to outcome-based checks (aligns with ACTION_PLAN.md Action 5).

---

### Cluster: WebSocket/Redis Suites (`tests/webui/test_chunking_websocket.py`, `tests/websocket/test_performance.py`, `tests/websocket/websocket_load_test.py`, `tests/websocket/stress_test_race_conditions.py`, `tests/websocket/test_race_conditions.py`)

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - stabilize and scope appropriately

**Findings**
- `tests/webui/test_chunking_websocket.py:269` relies on `time.sleep(1.1)` for throttling windows, creating flakiness.
- `tests/websocket/test_performance.py:42-55` opens real Redis connections on `redis://localhost:6379/15`.
- Load/stress scripts spawn many asyncio tasks, patch redis globally, and are effectively performance tests hiding inside default suites.

**Risks**
- Local runs fail if Redis isn’t running.
- Test suite mixes unit, integration, and performance concerns -> unclear guarantees.

**Remediation**
1. Provide fake/in-memory Redis fixtures for unit-level behavior.
2. Replace sleeps with controllable timing (e.g., patch loop time, use deterministic clocks).
3. Move true load/stress tests into an opt-in harness executed outside standard CI, or mark with `@pytest.mark.performance`.

**Reference**: ACTION_PLAN.md Action 6 outlines the implementation steps.

---

### File: `packages/shared/chunking/infrastructure/streaming/test_memory_pool.py`

**Grade**: C (6/10)  
**Status**: ⏳ Pending - needs deterministic timing

**Findings**
- Leak detection tests rely on real `asyncio.sleep` and background tasks (e.g., lines 109-137, 196-233). They occasionally take >2s and can fail under load.
- Uses actual `threading` interactions and `gc.collect()` to assert behavior, which is non-deterministic across platforms.

**Remediation**
1. Inject a controllable clock or expose a `check_for_leaks()` helper so tests can advance time without sleeping.
2. Patch `asyncio.sleep` or the scheduler to avoid real delays.
3. Add teardown to stop leak detection tasks; ensure fixtures clean up to prevent cross-test interference.

**Benefit**: Faster, reliable unit coverage for memory pool safety without flakes.

---

### Cluster: Application Use-Case & Validation Suites (`tests/application/test_*_use_case.py`, `packages/webui/tests/test_collection_service_chunking_validation.py`)

**Grade**: D (4/10)  
**Status**: ⏳ Pending - convert to integration-first coverage

**Issues**
- Suites recreate entire use cases with `MagicMock` dependencies and assert call counts (e.g., `tests/application/test_process_document_use_case.py:30-137`).
- `packages/webui/tests/test_collection_service_chunking_validation.py` revalidates the same branch logic already handled in integration/service tests, again using mocks.

**Risks**
- Changes in repository wiring or transaction handling won’t be caught.
- Duplicate maintenance burden: when validation rules change, multiple mock-based tests require updates.

**Recommended Steps**
1. Move high-value scenarios into integration tests hitting real repositories/unit-of-work and database fixtures.
2. Retain only small unit tests for pure validation helper methods.
3. Delete or shrink the mock-heavy suites after coverage migrates.

---

### File: `tests/api/test_rate_limits.py`

**Grade**: F (2/10)  
**Status**: ⏳ Pending - skipped in CI

**Findings**
- Entire module is wrapped with `@pytest.mark.skipif(os.getenv("CI") == "true")`, so rate limits and circuit breaker logic are never exercised in CI.
- Tests depend on global limiter state; without resets they hang or give false positives.

**Fix Plan**
1. Provide Redis/SlowAPI fixtures compatible with GitHub Actions (use existing Redis service or mock limiter).
2. Reset limiter/circuit breaker state per test (`circuit_breaker.failure_counts.clear()` etc.).
3. Remove the skip guard once stable.

**Outcome**: Restores automated coverage for throttling protections.

---

### File: `tests/webui/test_ingestion_chunking_integration.py`

**Grade**: D (4/10)  
**Status**: ⏳ Pending - should become real integration coverage

**Observations**
- Despite the filename, every dependency is a `MagicMock`; repositories never touch the database (lines 21-82).
- Tests assert on internal implementation details (e.g., exact chunk IDs, `mock_strategy.chunk` call counts) rather than observable behavior.
- Strategy factory is patched directly on the service instance (`patch.object(chunking_service.strategy_factory, "create_strategy", ...)`), making refactors brittle.

**Recommended Fix**
1. Move end-to-end ingestion scenarios into `tests/integration/services/test_chunking_ingestion.py` backed by the async session fixture.
2. Limit unit-level tests to pure validation/normalization edges; avoid duplicating logic already covered elsewhere.
3. Drop direct patches of strategy factory by using real strategies with controlled inputs or test doubles.

**Benefit**: Real confidence that ingestion chunking orchestrates repositories, strategy selection, and metadata updates.

---

### File: `tests/webui/services/test_collection_service.py`

**Grade**: D (4/10)  
**Status**: ⏳ Pending - consolidate with integration suite

**Key Issues**
- 1,100+ lines of mock-based tests duplicating scenarios now handled by `tests/integration/services/test_collection_service.py`.
- Heavy reliance on call-count assertions (`assert_called_once_with`) and patching Celery/task internals, which locks down implementation details.
- Maintaining both suites doubles effort and obscures the authoritative coverage.

**Plan**
1. Audit scenarios that still lack integration coverage and port them to the integration suite if needed.
2. Remove or drastically slim this file once overlap is resolved.
3. Retain only tiny unit tests for pure validation helpers (tracked under Action 8).

---

### File: `tests/unit/test_search_api.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - improve isolation and reduce global patching

**Findings**
- Fixture `test_client_for_search_api` mutates module-level singletons (`search_api_module.qdrant_client`, `model_manager`, `embedding_service`) and restores them manually; failures mid-test could leave globals dirty for later tests.
- Several tests patch Prometheus globals (`Counter`, `Histogram`) but never reset collectors, risking metric pollution across runs.
- Suite focuses on verifying `Mock` call behavior rather than the HTTP responses returned by the FastAPI app.

**Recommendations**
1. Refactor fixtures to use context managers or `monkeypatch` to guarantee cleanup even on failure.
2. Prefer hitting the FastAPI routes with `TestClient`/`AsyncClient` and assert on responses rather than internal mocks.
3. Move broader integration checks into a dedicated suite that spins up the app with dependency overrides.

---

### File: `tests/unit/test_search_api_edge_cases.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - duplicate patterns from main search_api suite

**Findings**
- Repeats the singleton patching pattern from `tests/unit/test_search_api.py`, manually restoring globals and clearing dependency overrides.
- Focuses on internal call assertions while mocking Qdrant, model manager, and hybrid engine extensively.
- Several tests patch `search_api_module` functions directly (`batch_search`, `hybrid_search`) instead of validating HTTP responses.

**Next Steps**
1. Merge meaningful edge-case scenarios into the main integration coverage and delete duplicates.
2. Provide sanitized fixtures that guarantee cleanup to avoid cross-test leakage.
3. Convert remaining edge cases into request/response assertions against the FastAPI app.

---

### File: `tests/unit/test_rate_limiter.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - isolate limiter state

**Observations**
- Touches private limiter attributes (`limiter._limiter.storage.storage`) and relies on environment toggles, so global state can leak between tests.
- Patches `Limiter._check_request_limit` to simulate throttling instead of exercising actual storage behavior.
- Test app factory doesn’t always dispose of limiter instances, risking shared counters across tests.

**Action Items**
1. Provide fixtures that yield a fresh `Limiter` with isolated in-memory or fakeredis storage per test.
2. Assert on HTTP responses/headers rather than limiter internals.
3. Remove dependency on `DISABLE_RATE_LIMITING` flag inside unit tests; configure key function explicitly.

---

### File: `tests/webui/test_chunking_metrics.py`

**Grade**: D (4/10)  
**Status**: ⏳ Pending - move coverage to integration tier

**Observations**
- Directly mutates Prometheus internals (`metric._metrics.clear()`) before/after tests, which risks leaking state if tests abort early.
- Uses `ChunkingService` instantiated with `MagicMock` repositories and asserts on mocked behavior rather than emitted metrics.
- Helper functions pull private metric fields (`_value`, `_metrics`, `_buckets`) for assertions, making the tests brittle to library upgrades.

**Remediation**
1. Introduce integration tests that run real chunking operations and assert on metrics via isolated Prometheus registry fixtures.
2. Remove reliance on private attributes by using `CollectorRegistry` and `generate_latest`.
3. Trim this file to a small unit test ensuring the metric helper functions call Prometheus APIs correctly.

---

### File: `tests/unit/test_hierarchical_chunker.py`

**Grade**: C (6/10)  
**Status**: ⏳ Pending - simplify and reduce external coupling

**Findings**
- Heavy reliance on `llama_index` internals (`HierarchicalNodeParser`, `NodeRelationship`), patched with `MagicMock`, which couples tests to third-party implementation details.
- Large fixtures and multi-level documents create slow, hard-to-maintain tests; many assertions repeat the same metadata checks.
- Some tests only confirm the presence of keys rather than validating hierarchy correctness or failure modes.

**Suggested Actions**
1. Replace `llama_index` dependency with lightweight domain stubs or property-based tests that verify hierarchy invariants.
2. Consolidate repeated assertions into helper functions, cutting down the 30+ scenarios to a focused core set.
3. Add negative tests (e.g., invalid chunk sizes, non-descending hierarchies) that exercise validation logic without mocking parser internals.

---

### File: `tests/webui/services/test_chunking_service_comprehensive.py`

**Grade**: D+ (5/10)  
**Status**: ⏳ Pending - decompose and align with integration tests

**Issues**
- 900+ lines covering `ChunkingService` behavior with `AsyncMock` repositories and massive fixtures; duplicates scenarios already migrated to integration tests.
- Uses placeholder assertions (`assert True`) in several async tests (`test_process_chunking_operation_*`) that merely call the method without checking outcomes.
- Extensive patching of Redis, Celery, and document services; difficult to maintain and offers little protection against real regressions.

**Plan**
1. Inventory unique scenarios still missing from `tests/integration/services/test_chunking_service.py` and port them there.
2. Remove or drastically pare down this file after integration coverage exists.
3. Ensure remaining unit tests focus on pure helper methods (e.g., validation) with tight assertions.

---
### File: `tests/webui/api/v2/test_chunking.py`

**Grade**: D (4/10)  
**Status**: ⏳ Pending - consolidate with integration suite

**Observations**
- 1,500+ lines of tests patching background tasks, Redis managers, and dependency overrides at import time (`bg_tasks.start_background_tasks = AsyncMock()` etc.).
- Uses module-level globals (`app = None`, `chunking_module = None`) and lazy import hacks to control initialization.
- Asserts on mocked service calls rather than verifying actual HTTP responses with real dependencies.
- Sub-suites (`test_chunking_simple_integration.py`, `test_chunking_direct.py`) repeat much of the same logic with slightly different payloads.

**Remediation**
1. Replace with focused integration tests that hit the API using the real dependency override fixtures (`tests/integration/...`).
2. Remove module-level mocking in favor of per-test dependency overrides in `conftest.py`.
3. Reduce duplication by extracting shared helpers and migrating scenarios into parameterized integration tests.

---

### File: `tests/webui/api/v2/test_operations.py`

**Grade**: D (4/10)  
**Status**: ⏳ Pending - migrate to integration coverage

**Findings**
- Exercises endpoints using mocked `OperationService` instances and asserts on `OperationResponse` fields rather than real HTTP responses.
- Duplicates behavior already covered (or planned) in integration suites, relying on call-count assertions.

**Next Steps**
1. Move success/error scenarios into integration tests hitting `/api/v2/operations/*`.
2. Retain only slim unit tests for schema validation if necessary.
3. Remove redundant mock assertions once integration coverage is in place.

---

### Files: `tests/webui/api/v2/test_chunking_simple_integration.py`, `tests/webui/api/v2/test_chunking_direct.py`

**Grade**: D (4/10)  
**Status**: ⏳ Pending - redundant “integration” suites

**Observations**
- Both files rely on mocked services and dependency overrides, duplicating logic from `test_chunking.py`.
- Provide little additional coverage beyond the primary integration tests.

**Remediation**
1. Merge missing scenarios into the canonical integration suite.
2. Delete or drastically shrink these files after consolidation.

---

### File: `tests/webui/services/test_search_service.py`

**Grade**: D (4/10)  
**Status**: ⏳ Pending - migrate to integration coverage

**Issues**
- 980+ lines of mocks covering every branch of `SearchService`, including duplicate validations already present in API tests.
- Heavy reliance on `assert_called_once_with` for repository and reranker mocks, offering little behavioral assurance.
- Real integration tests (`tests/integration/test_search_api_*`) already cover most flows with actual repositories.

**Suggested Path**
1. Identify unique scenarios (e.g., specific error mapping) and port them to integration tests if missing.
2. Delete or drastically shrink the mock suite after coverage migration.
3. Keep a minimal unit test file for pure helper methods if needed.

---

### File: `tests/unit/test_search_service.py`

**Grade**: D (4/10)  
**Status**: ⏳ Pending - overlaps with API/integration coverage

**Findings**
- Constructs `SearchService` with mocked repositories and patches `httpx.AsyncClient`; assertions focus on mock call counts rather than actual API responses.
- Sleep-based retry simulation (`time.sleep`) slows the tests and doesn’t reflect real backoff behavior.
- Scenarios largely duplicate `tests/webui/api/v2/test_search.py` and integration suites.

**Remediation**
1. Move high-value scenarios into integration tests that exercise `/api/v2/search` via FastAPI dependency overrides.
2. Keep only minimal unit tests for pure validation helpers, avoiding network mocks.
3. Replace sleep usage with deterministic retry hooks if any unit coverage remains.

---

### File: `tests/webui/test_websocket_manager.py`

**Grade**: C (6/10)  
**Status**: ⏳ Pending - simplify setup/teardown and reduce global mutations

**Findings**
- Manually cancels global singleton tasks (`ws_manager.consumer_tasks`) before and after tests; any failure mid-test can leave tasks running.
- Uses custom teardown loops with `asyncio.all_tasks`, `task.cancel()`, and timeouts, indicating the fixture system isn’t providing isolation.
- Relies on direct `redis.asyncio.from_url` patching without fakeredis fallback, despite such fixtures existing in `tests/conftest.py`.

**Actions**
1. Reuse the fakeredis fixtures to avoid patching `from_url` and reduce cleanup complexity.
2. Provide a fresh manager instance per test via fixture rather than mutating the global singleton.
3. Ensure teardown uses `asyncio.TaskGroup` or helper utilities to avoid manual task cancellation loops.

---

### File: `tests/webui/test_tasks_websocket_integration.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - unify websocket coverage

**Findings**
- Patches Redis connection, Qdrant manager, document scanning service, and Celery helpers in every test; overlaps with other websocket suites.
- Assertions focus on message presence, not end-to-end flow; duplicates logic slated for consolidated integration tests.

**Plan**
1. Consolidate high-value websocket progress scenarios into a single integration suite with fakeredis fixtures.
2. Remove this duplicate once consolidated coverage exists.

---

### File: `tests/integration/test_websocket_redis_integration.py`

**Grade**: C (6/10)  
**Status**: ⏳ Pending - replace custom Redis mock with shared fixtures

**Issues**
- Defines an in-file `RedisStreamMock` rather than using fakeredis, diverging from actual Redis semantics.
- Overlaps with both `test_websocket_manager.py` and `test_tasks_websocket_integration.py`, increasing maintenance cost.

**Recommended Steps**
1. Reuse common fakeredis fixtures from `tests/conftest.py`.
2. Merge redundant scenarios with other websocket integration tests.

---

### File: `tests/webui/test_document_chunk_count_updates.py`

**Grade**: D (4/10)  
**Status**: ⏳ Pending - redundant Celery ingestion mocks

**Findings**
- Relies on mocked `AsyncSession`, `ChunkingService`, and `httpx` clients to simulate ingestion flows, effectively duplicating logic already exercised in task integration tests.
- Asserts on mock call counts and in-memory `MagicMock` documents instead of verifying persisted state or WebSocket output.
- Heavy patching of `extract_and_serialize_thread_safe` and `ChunkingService` hides true behavior.

**Recommended Fix**
1. Move validated scenarios to integration tests that run `_process_append_operation`/`_process_reindex_operation` with fakeredis and the real DB fixture.
2. Remove or drastically reduce this mock-heavy suite afterwards.

---

### File: `tests/webui/services/test_directory_scan_service.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - consolidate with integration coverage

**Issues**
- Builds large temporary directory structures and patches `ws_manager` per test; duplicates behavior that should be covered by higher-level tests.
- Assertions focus on counts and file lists using `MagicMock` managers rather than verifying API responses.

**Plan**
1. Create integration tests for directory scanning using shared fixtures and ensure WebSocket messages are asserted there.
2. Slim this suite down to cover only edge-case helpers if necessary.

---

### File: `tests/webui/services/test_execute_ingestion_chunking.py`

**Grade**: C (6/10)  
**Status**: ⏳ Pending - overlaps integration suite

**Observations**
- Focused on `ChunkingService.execute_ingestion_chunking`, but uses mocked strategies and config builders; many scenarios already covered (or planned) in integration tests.
- Includes sleeps/backoff simulations and placeholder assertions on returned stats.

**Next Steps**
1. Port meaningful edge cases (config builder validation, fallback behavior) into integration tests.
2. Reduce this file to minimal unit tests for config builder helper logic.

---

### File: `tests/unit/test_document_scanning_service.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - relies on mocked sessions

**Findings**
- Mocks `AsyncSession` and repository calls, effectively reimplementing ORM behavior.
- Duplicate coverage of scenarios already exercised by directory scanning service tests.

**Remediation**
1. Replace with integration tests that operate against the DB using the real service.
2. Keep only simple validation unit tests if needed.

---

### File: `tests/unit/test_directory_scan_service.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - duplicates service tests

**Issues**
- Uses mocked DB sessions to verify internal calls, rather than asserting on API/service outputs.
- Overlaps nearly entirely with `tests/webui/services/test_directory_scan_service.py`.

**Action Items**
1. Remove redundant unit tests once integration coverage is in place.
2. Retain only pure helper validation tests if necessary.

---

### File: `tests/unit/test_api_key_repository.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - mock-based repository coverage

**Problems**
- Interacts with `AsyncMock` session objects instead of exercising real database constraints.
- Asserts on mocked `execute`/`flush` calls, offering minimal behavioral assurance.

**Recommendation**
1. Move repository coverage into integration tests using the async session fixture.
2. Keep a slim unit file only for pure utility functions.

---

### File: `tests/unit/test_operation_repository.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - mock-based unit suite

**Findings**
- Uses mocked unit-of-work/session patterns to assert call sequences rather than persisted state changes.
- Duplicates scenarios planned for integration coverage (operation lifecycle, retries).

**Next Steps**
1. Shift coverage to integration tests with the real DB.
2. Remove or shrink this mock-heavy suite after migration.

---

### File: `tests/webui/api/v2/test_collections.py`

**Grade**: D (4/10)  
**Status**: ⏳ Pending - convert to integration coverage

**Findings**
- Overrides FastAPI dependencies to return mocked `CollectionService` and asserts on internal call arguments instead of HTTP semantics.
- Many scenarios duplicate the newly added integration suite; maintaining both increases drift risk.

**Actions**
1. Port remaining gaps (if any) into integration tests that exercise `/api/v2/collections`.
2. Remove or minimize this mock-heavy file once integration coverage is in place.

---

### File: `tests/webui/api/v2/test_collections_operations.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - overlaps service integration

**Issues**
- Mock-based listing of operations/documents with custom side effects; heavy duplication of service-level logic.
- Assertions focus on DTO conversion rather than response payload contracts.

**Plan**
1. Replace with integration tests using real service fixtures.
2. Keep a slim unit test only for filter/parameter validation if necessary.

---

### File: `tests/webui/services/dtos/test_chunking_dtos.py`

**Grade**: C (6/10)  
**Status**: ⏳ Pending - consider slimming

**Observations**
- Recreates DTO-to-API conversions already enforced by Pydantic models; high maintenance for low risk.
- Could be reduced to a few smoke tests or property-based checks.

---

### File: `tests/unit/test_hierarchical_chunker_extended.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - combines llama-index mocks with large permutations

**Findings**
- Extends hierarchical chunker tests with additional permutations, continuing the heavy mocking pattern noted earlier.
- Significant overlap with other chunker suites.

**Recommendation**: consolidate with Action 12 efforts on chunker tests.

---

### File: `tests/domain/test_chunking_operation.py`

**Grade**: C (6/10)  
**Status**: ⏳ Pending - relies on MagicMocks for strategy interactions

**Notes**
- Provides valuable domain coverage but still uses `MagicMock` for strategies; consider supplementing with real strategy fixtures in integration tests.

---

### File: `tests/domain/test_value_objects.py`

**Grade**: C+ (6.5/10)  
**Status**: 🟢 Minor adjustments

**Observation**: Solid validation coverage; could be slimmed via parameterization to reduce repetition.

---

### File: `tests/unit/test_all_chunking_strategies_unified.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - massive permutation suite

**Findings**
- Extensive cross-strategy testing with real implementations, leading to long runtimes and duplication.
- Overlaps with integration/performance suites introduced in Action 12.

---

### File: `tests/streaming/validate_streaming_pipeline.py`

**Grade**: D (4/10)  
**Status**: ⏳ Pending - manual validation script

**Issues**
- Script generates large temp files, uses `print` statements, never asserts; belongs in manual diagnostics instead of automated tests.

**Action**: Move to manual QA folder alongside other probe scripts (Action 3).

---

### File: `tests/unit/test_chunking_error_metrics.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - duplicate metrics coverage

**Observation**: Similar to `test_chunking_metrics.py`, manipulates Prometheus internals; should be replaced by integration tests capturing real error flows.

---

### File: `tests/unit/test_chunking_tasks.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - heavy Celery mocking

**Findings**
- Exhaustive patching of repositories, Redis, psutil, etc., to simulate task lifecycle; duplicates integration coverage for chunking tasks.
- Some tests rely on sleep/timeouts and placeholder assertions.

**Plan**: consolidate with integration suites (Action 14) and keep only minimal unit checks for helper functions.

---

### File: `tests/chunking/streaming/test_streaming_strategies.py`

**Grade**: C (6/10)  
**Status**: ⏳ Pending - consider performance split

**Notes**: Valuable streaming coverage but large runtime; evaluate splitting into integration vs. performance categories.

---

### File: `tests/integration/test_chunking_error_flow.py`

**Grade**: C (6/10)  
**Status**: ⏳ Pending - refine fixtures

**Observation**: Uses custom mocks to simulate error flows; ensure alignment with new integration fixtures when refactoring chunking task coverage.

---

### File: `tests/unit/test_resource_manager.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - mock-based resource checks

**Issue**: Relies on mocked repositories and `psutil`, providing limited assurance; migrate to integration tests for resource enforcement.

---

### File: `tests/webui/services/test_progressive_segmentation.py`

**Grade**: C (6/10)  
**Status**: ⏳ Pending - mock-heavy service logic

**Plan**: Merge with integration coverage addressing segmentation flows; keep only helper validation tests if needed.

---

### File: `tests/unit/test_models.py`

**Grade**: C (6/10)  
**Status**: ⏳ Pending - huge ORM test suite

**Findings**: Provides schema checks via mocking; consider leveraging alembic/DB fixtures instead of asserting on mocked SQLAlchemy models.

---

### File: `tests/unit/test_model_manager.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - mocked model lifecycle

**Note**: Duplicates functionality exercised by integration tests; plan to consolidate during Action 13.

---

### Files: `tests/websocket/test_cleanup.py`, `tests/websocket/test_scaling.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - convert to manual load scripts

**Observation**: Act as load/cleanup scripts relying on live services; relocate to manual tests (Action 3) or provide deterministic automation.

---

### File: `tests/unit/test_chunking_exceptions.py`

**Grade**: C (6/10)  
**Status**: ⏳ Pending - consolidating exception coverage

**Notes**: Unit-level exception tests offer modest value; ensure they complement integration error-flow coverage.

---

### File: `tests/unit/test_shared_qdrant_manager.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - extensive patching

**Issues**: Patches Redis/Qdrant internals with mocks; move to integration tests utilizing fakeredis and Qdrant stubs.

---

### File: `tests/unit/test_hybrid_chunker.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - overlaps other chunker suites

**Plan**: Fold into Action 12 chunker test cleanup; reduce redundant permutations and heavy patching.

---
### File: `tests/domain/test_chunking_strategies.py`

**Grade**: C (6/10)  
**Status**: ⏳ Pending - strengthen assertions

**Problems**
- Several tests end with `assert True` or only verify non-empty lists (e.g., lines 688-708), leaving metadata/weight logic untested.
- Hybrid strategy tests should assert on `strategies_used`, weight normalization, and confidence mixing but currently skip those checks.

**Action Items**
1. Replace placeholders with explicit assertions on metadata fields (`custom_attributes["strategies_used"]`, weight sums, fallback markers).
2. Add negative-case assertions (e.g., when a strategy fails, ensure fallback metadata is recorded).

**Impact**: Better regression detection for domain-level chunking logic.

---
### File: `tests/unit/test_all_chunking_strategies.py`

**Grade**: ✅ Addressed
**Status**: ✅ Completed (2025-10-16) - Split into focused integration suites and performance smoke
**Location**:
- `tests/integration/strategies/test_chunker_edge_cases.py`
- `tests/integration/strategies/test_chunker_behaviors.py`
- `tests/integration/strategies/test_chunker_factory.py`
- `tests/performance/test_chunking_large_documents.py`

#### Critical Issues

- [x] **Conjoined Twins** (P1) - Integration tests in unit test directory
  - Action: Introduced `tests/integration/strategies/` package hosting edge-case, behavior, and factory coverage with integration marker.

- [x] **Slow Poke** (P1) - Parametrized tests create 78 test permutations
  - Action: Trimmed to 24 cross-strategy edge-case permutations; migrated 1MB document stress test into `tests/performance/test_chunking_large_documents.py` with `@pytest.mark.performance`.

- [x] **Giant Tests** (P1) - Single 816-line class with 30+ test methods
  - Action: Replaced monolith with three concise modules grouped by concern.

#### Moderate Issues

- [x] **Happy Path Only** (P1) - Missing negative test cases
  - Action: Added metadata preservation assertions, strategy override checks, and validation failures that cover previous blind spots.

- [x] **Testing Implementation Details** (P2) - Verifies internal offset calculation
  - Action: Rewrote assertions to focus on observable behavior (unique chunk ids, hierarchy metadata, semantic markers) rather than internal calls.

#### Action Items

1. ✅ Create `tests/integration/strategies/` directory
2. ✅ Replace legacy unit file with integration-focused modules
3. ✅ Add `pytestmark = pytest.mark.integration` (registered in `pyproject.toml`)
4. ✅ Split scenarios across strategy-specific tests
5. ✅ Move large document scenario under `tests/performance/` with `@pytest.mark.performance`

#### Verification
- `uv run pytest tests/integration/strategies -q`
- `uv run pytest tests/performance/test_chunking_large_documents.py -q`

---

### File: `tests/test_embedding_integration.py`

**Grade**: D (4/10)
**Status**: 🟡 Needs refactoring
**Location**: `/home/john/semantik/tests/test_embedding_integration.py`

#### Critical Issues

- [ ] **Mockery** (P1) - Mocks CUDA and metrics, defeating integration test purpose
  - Lines: 24-25 (mocks metrics module), 31-34 (mocks CUDA)
  - Impact: Tests don't validate real hardware detection
  - Action: Remove mocks, test with real components

- [ ] **Conjoined Twins** (P1) - Unit tests masquerading as integration tests
  - Lines: 32-48, 49-70, 91-112
  - Impact: Not testing actual integration
  - Action: Convert to real integration tests or move to unit/

- [ ] **Testing Implementation Details** (P1) - Verifies private attributes
  - Lines: 39-47 (checks `_service`, `_loop` attributes)
  - Impact: Brittle, coupled to implementation
  - Action: Test public behavior only

#### Moderate Issues

- [ ] **Happy Path Only** (P1) - Missing error conditions
  - Missing: Model load failures, timeout scenarios, race conditions
  - Action: Add comprehensive error tests

- [ ] **Slow Poke** (P2) - Performance test with arbitrary 1s threshold
  - Lines: 72-88
  - Impact: Too generous, mock should be <50ms
  - Action: Tighten threshold to 0.1s for mock mode

#### Action Items

1. ⏳ Move to proper location: `git mv tests/test_embedding_integration.py tests/integration/`
2. ⏳ Remove module-level mocking (lines 24-25)
3. ⏳ Add real error case tests
4. ⏳ Tighten performance thresholds

---

### File: `tests/e2e/test_websocket_integration.py`

**Grade**: C (6/10)
**Status**: 🟡 High Priority - Needs cleanup
**Location**: `/home/john/semantik/tests/e2e/test_websocket_integration.py`

#### Critical Issues

- [ ] **Generous Leftovers** (P1) - User registration creates persistent pollution
  - Lines: 38-79 (tries multiple users, creates "testuser")
  - Impact: Non-deterministic, second run fails
  - Effort: 4 hours
  - Action: Use unique user per test with proper fixtures

- [ ] **Giant Tests** (P1) - Tests exceed 50-100 lines each
  - Lines: 86-183 (98 lines), 259-371 (113 lines)
  - Impact: Hard to debug, unclear purpose
  - Action: Split into focused tests with helper methods

- [ ] **Code Duplication** (P1) - Auth helpers duplicated across files
  - Lines: 38-84 (duplicated in test_websocket_reindex.py)
  - Impact: Maintenance burden, inconsistent updates
  - Effort: 4 hours
  - Action: Extract to `tests/e2e/conftest.py`

#### Moderate Issues

- [ ] **Slow Poke** (P2) - Hardcoded 60-120 second timeouts
  - Lines: 148, 242, 342, 437
  - Impact: Tests take minutes even when operations complete quickly
  - Action: Implement exponential backoff polling

- [ ] **Leaky Resources** (P2) - WebSocket connections not cleaned up in all paths
  - Lines: 133-144, 232-238
  - Impact: Resource leaks, port exhaustion
  - Action: Use context manager for WebSocket cleanup

#### Action Items

1. ⏳ Create `tests/e2e/conftest.py` with shared fixtures
2. ⏳ Extract auth helpers to conftest
3. ⏳ Create `e2e_test_user` fixture with unique IDs
4. ⏳ Split giant tests into focused methods
5. ⏳ Add WebSocket context manager for cleanup
6. ⏳ Implement adaptive polling with early exit

**Code Examples**: See ACTION_PLAN.md Section "Action 5" and "Action 8"

---

### File: `tests/webui/services/test_search_service_reranking.py`

**Grade**: D+ (5/10)
**Status**: 🟡 Needs improvement
**Location**: `/home/john/semantik/tests/webui/services/test_search_service_reranking.py`

#### Critical Issues

- [ ] **Mockery** (P1) - Mocks entire HTTP layer (httpx.AsyncClient)
  - Lines: 71-77, 142-154, 208-214, 256-262, 290-306, 341-357
  - Impact: Doesn't test real HTTP request construction
  - Effort: 1-2 days
  - Action: Use `respx` library for transport-level mocking

- [ ] **Testing Implementation Details** (P2) - Verifies internal method calls
  - Lines: 90-101, 224-227
  - Impact: Brittle, breaks with refactoring
  - Action: Focus on behavior and outcomes

- [ ] **Happy Path Only** (P1) - Missing negative test coverage
  - Missing: Invalid inputs, validation failures, error conditions
  - Action: Add parameterized tests for edge cases

#### Moderate Issues

- [ ] **Inconsistent Mock Data** (P2) - Mock structures lack consistency
  - Lines: 23-41, 56-69, 115-140
  - Action: Create fixture factory for consistent mocks

#### Action Items

1. ⏳ Install `respx` library
2. ⏳ Replace httpx mocking with respx transport mocking
3. ⏳ Add negative test cases for invalid inputs
4. ⏳ Create mock data fixtures
5. ⏳ Consider extracting HTTP client abstraction

**Estimated Effort**: 1-2 days

---

### File: `tests/integration/test_collection_deletion.py`

**Grade**: C (6/10)
**Status**: 🟡 Medium Priority
**Location**: `/home/john/semantik/tests/integration/test_collection_deletion.py`

#### Critical Issues

- [ ] **Conjoined Twins** (P2) - Test of SQLAlchemy internals, not deletion behavior
  - Lines: 162-198 (`test_async_delete_pattern_in_repository`)
  - Impact: Tests implementation, not behavior
  - Action: Delete this test, behavior covered by other tests

- [ ] **Mockery** (P2) - Mocks Qdrant with generic exceptions
  - Lines: 122-139
  - Impact: Doesn't test real error handling
  - Action: Use actual Qdrant exception types

#### Moderate Issues

- [ ] **Missing Coverage** (P2) - No tests for large cascades, rollback scenarios
  - Action: Add tests for 10,000+ document deletion, partial failures

- [ ] **Incomplete Test** (P2) - Concurrent test admits it doesn't test concurrency
  - Lines: 212-228
  - Action: Implement true concurrent test or rename

#### Action Items

1. ⏳ Delete `test_async_delete_pattern_in_repository` (lines 162-198)
2. ⏳ Use real Qdrant exceptions in error tests
3. ⏳ Add large cascade test (10,000 documents)
4. ⏳ Implement true concurrency test with multiple sessions

**Estimated Effort**: 1 day

---

### File: `tests/unit/test_user_repository.py`

**Grade**: C- (5/10)  
**Status**: ⏳ Pending - migrate to database-backed coverage

**Findings**
- Uses `AsyncMock` sessions to simulate ORM behavior, then asserts on `add/flush/refresh` call counts.
- Key behaviors (unique constraint handling, password hashing, timestamp updates) should be exercised with the real async session fixture.

**Action Items**
1. Add integration tests for the repository using the test database.
2. Reduce this unit file to minimal validation helpers once integration coverage exists.

---

### File: `tests/security/test_path_traversal.py`

**Grade**: C+ (6.4/10)
**Status**: 🟢 Minor issues
**Location**: `/home/john/semantik/tests/security/test_path_traversal.py`

#### Critical Issues

- [ ] **Slow Poke** (P2) - Flaky performance test
  - Lines: 239-256
  - Impact: CI failures due to timing variability
  - Effort: 4 hours
  - Action: Use statistical approach with warmup and p50/p95 metrics

- [ ] **Dodger** (P2) - Tests exact error message string
  - Lines: 258-277
  - Impact: Brittle, prevents improving error messages
  - Action: Test that path isn't leaked, not exact message

#### Moderate Issues

- [ ] **Code Duplication** (P2) - Repeated test patterns across 10+ methods
  - Lines: 23-157
  - Action: Extract `_assert_paths_rejected` helper

- [ ] **Missing Parameterization** (P2) - Tests would benefit from pytest.mark.parametrize
  - Action: Convert loop-based tests to parameterized tests

- [ ] **Incomplete Test** (P2) - Symlink test missing critical case
  - Lines: 220-237 (commented out escape test)
  - Action: Implement symlink escape test

#### Minor Issues

- [ ] Platform-dependent test needs proper skip marker
  - Lines: 158-170
  - Action: Use `@pytest.mark.skipif`

#### Action Items

1. ⏳ Fix performance test with statistical approach (see ACTION_PLAN.md)
2. ⏳ Fix error message test to check behavior not string
3. ⏳ Complete symlink escape test
4. ⏳ Add parameterization to reduce duplication
5. ⏳ Add proper skip markers for platform tests

**Estimated Effort**: 4 hours

---

### File: `tests/streaming/test_streaming_integration.py`

**Grade**: D (4/10)
**Status**: 🟡 Medium Priority
**Location**: `/home/john/semantik/tests/streaming/test_streaming_integration.py`

#### Critical Issues

- [ ] **Mockery** (P1) - Creates fake Chunk objects instead of real entities
  - Lines: 28-56, especially 46-50 (`chunk = MagicMock(spec=Chunk)`)
  - Impact: Bypasses all Chunk validation, false confidence
  - Effort: 1-2 days
  - Action: Use real Chunk and ChunkMetadata objects

- [ ] **Generous Leftovers** (P2) - Temporary file cleanup risks
  - Lines: 86-122, 126-163, 170-211
  - Impact: Files may leak on test failure
  - Action: Use pytest `tmp_path` fixture or context managers

- [ ] **Conjoined Twins** (P1) - Unit tests mixed with integration tests
  - Lines: 279-327 (StreamingWindow, MemoryPool unit tests)
  - Impact: Wrong test organization
  - Action: Move to separate unit test files

#### Moderate Issues

- [ ] **Dodger** (P2) - Tests internal memory pool statistics
  - Lines: 116-119, 203-208
  - Action: Focus on functional outcomes

- [ ] **Happy Path Only** (P1) - Missing error cases
  - Missing: UTF-8 errors, permission denied, disk full, memory exhausted
  - Action: Add comprehensive error tests

- [ ] **Free Ride** (P2) - Multiple encoding concerns in one test
  - Lines: 150-160
  - Action: Split UTF-8 test by character set

#### Action Items

1. ⏳ Replace MagicMock(spec=Chunk) with real Chunk objects
2. ⏳ Use pytest tmp_path fixture for all file operations
3. ⏳ Move unit tests to `tests/unit/infrastructure/streaming/`
4. ⏳ Add missing error case tests
5. ⏳ Split UTF-8 test into focused tests

**Estimated Effort**: 1-2 days

---

## Anti-Pattern Summary

### Mockery (Found in 7/10 files)
Files with excessive mocking that defeats test purpose:
- ❌ `test_collection_repository.py` - Mocks entire database
- ❌ `test_collection_service.py` - Mocks all repositories
- ❌ `test_search_service_reranking.py` - Mocks HTTP client
- ❌ `test_embedding_integration.py` - Mocks CUDA, metrics
- ❌ `test_streaming_integration.py` - Mocks domain entities
- ⚠️ `test_collection_deletion.py` - Mocks Qdrant with generic exceptions
- ⚠️ `test_path_traversal.py` - Minor (tests error messages)

### Conjoined Twins (Found in 6/10 files)
Tests that are mislabeled or misplaced:
- ❌ `test_collection_repository.py` - Mock integration tests as unit tests
- ❌ `test_all_chunking_strategies.py` - Integration tests in unit/
- ❌ `test_embedding_integration.py` - Unit tests as integration tests
- ❌ `test_streaming_integration.py` - Unit + integration mixed
- ⚠️ `test_collection_deletion.py` - One test of SQLAlchemy internals
- ⚠️ `test_websocket_integration.py` - Not true E2E (polls REST API)

### Generous Leftovers (Found in 4/10 files)
Tests creating persistent state without cleanup:
- ❌ `test_auth.py` - Creates "testuser" without cleanup
- ❌ `test_websocket_integration.py` - User registration pollution
- ⚠️ `test_streaming_integration.py` - Temporary file risks
- ⚠️ Several tests create unique users but don't verify cleanup

### Giant Tests (Found in 5/10 files)
Tests exceeding 50 lines or single class >500 lines:
- ❌ `test_collection_service.py` - 904 lines in one class
- ❌ `test_all_chunking_strategies.py` - 816 lines in one class
- ❌ `test_websocket_integration.py` - Multiple 98+ line tests
- ⚠️ `test_collection_repository.py` - Complex 40+ line tests
- ⚠️ `test_streaming_integration.py` - 40-line test method

### Happy Path Only (Found in 8/10 files)
Missing negative test cases:
- All files except `test_path_traversal.py` and `test_auth.py` (which should be deleted)
- Most severe: `test_collection_service.py`, `test_search_service_reranking.py`

### Slow Poke (Found in 3/10 files)
Performance issues or flaky timing:
- ❌ `test_all_chunking_strategies.py` - 78 test permutations, 1MB files
- ❌ `test_websocket_integration.py` - 60-120s hardcoded timeouts
- ⚠️ `test_path_traversal.py` - Flaky <10ms assertion

---

## Test Coverage Gaps

### Missing Test Scenarios

#### Repository Layer
- [ ] Concurrent operations with multiple sessions
- [ ] Transaction rollback on mid-operation failures
- [ ] Database constraint violations
- [ ] Connection pool exhaustion

#### Service Layer
- [ ] `update()` method completely untested in collection repository
- [ ] Cross-field validation constraints
- [ ] Concurrent modification conflicts
- [ ] Rate limiting behavior

#### Error Handling
- [ ] Network timeouts (vecpipe, Qdrant)
- [ ] Invalid UTF-8 decoding
- [ ] Permission denied errors
- [ ] Disk full scenarios
- [ ] Memory limit violations

#### Security
- [ ] Symlink escape from base directory (incomplete test)
- [ ] Path traversal with mixed valid/invalid paths
- [ ] Concurrent security validation

#### E2E Flows
- [ ] WebSocket reconnection after server restart
- [ ] Operation recovery from checkpoint after failure
- [ ] Collection deletion with active operations
- [ ] Large document (100+ MB) processing

---

## Test Organization Issues

### Current Structure Problems

```
tests/
├── test_auth.py                              ❌ Should be deleted
├── test_embedding_integration.py             ❌ Should be in integration/
├── unit/
│   ├── test_collection_repository.py         ❌ Should be integration/
│   ├── test_collection_service.py            ❌ Should be integration/
│   └── test_all_chunking_strategies.py       ❌ Should be integration/
├── integration/
│   └── test_collection_deletion.py           ✅ Correct location
├── e2e/
│   ├── test_websocket_integration.py         ⚠️ Has issues but correct location
│   └── test_websocket_reindex.py             ⚠️ Code duplication
└── security/
    └── test_path_traversal.py                ✅ Correct location
```

### Target Structure

```
tests/
├── unit/
│   ├── strategies/                           # Pure logic tests
│   ├── services/
│   │   └── test_collection_service_validation.py  # Validation only
│   └── infrastructure/
│       └── streaming/
│           ├── test_window.py
│           └── test_memory_pool.py
├── integration/
│   ├── repositories/
│   │   └── test_collection_repository.py     # Real DB
│   ├── services/
│   │   ├── test_collection_service.py        # Real repos
│   │   └── test_search_service_errors.py     # Error scenarios
│   └── strategies/
│       └── test_chunking_strategies.py       # Real implementations
├── e2e/
│   ├── conftest.py                           # Shared fixtures
│   └── websocket/
│       ├── test_websocket_operations.py
│       └── test_websocket_errors.py
├── performance/
│   ├── test_chunking_large_documents.py
│   └── test_streaming_throughput.py
└── fixtures/
    └── factories.py                          # Reusable test data
```

---

## Metrics and Statistics

### Test Quality Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Unit test avg time | ~100ms | <10ms | 🔴 Fail |
| Integration test avg time | Varies | <1s | 🟡 Partial |
| Mock lines of code | ~2000 | <100 | 🔴 Fail |
| Tests with assertions | 98% | 100% | 🟡 Partial |
| Tests in correct directory | 40% | 100% | 🔴 Fail |
| Negative test coverage | ~20% | >80% | 🔴 Fail |
| Code duplication in tests | High | Low | 🔴 Fail |

### By Anti-Pattern Frequency

```
Mockery:              ████████████████████ 70%
Conjoined Twins:      ████████████████     60%
Happy Path Only:      ████████████████████ 80%
Giant Tests:          ████████████         50%
Generous Leftovers:   ████████             40%
Dodger:              ██████████████       55%
Slow Poke:            ██████               30%
```

### Lines of Test Code Analysis

```
Total test LOC:                   ~8,500
Mock setup LOC:                   ~2,000 (23%)
Actual test logic LOC:            ~4,000 (47%)
Comments/whitespace LOC:          ~2,500 (30%)

Target distribution:
Mock setup:                       <500    (5%)
Actual test logic:                ~7,000  (82%)
Comments/documentation:           ~1,000  (13%)
```

---

## Progress Tracking

### Sprint 1 (Week 1-2)

- [x] Action 1: Delete `test_auth.py` (5 min)
  - [ ] Verify existing coverage
  - [ ] Delete file
  - [ ] Commit with explanation

- [x] Action 2: Fix `test_collection_repository.py` (2-3 days)
  - [ ] Create integration test structure
  - [ ] Implement test_user fixture
  - [ ] Rewrite create tests
  - [ ] Rewrite get/list tests
  - [ ] Add update() method tests
  - [ ] Add validation tests
  - [ ] Delete old file
  - [ ] Verify all pass

- [x] Action 3: Replace `tests/unit/test_collection_service.py` with integration coverage (3-4 days)
    - ✅ Created `tests/integration/services/test_collection_service.py`
    - ✅ Added integration fixtures (`user_factory`, Celery/Qdrant stubs)
    - ✅ Covered create/delete/update, add/remove source, blue-green reindex, create_operation helper, list APIs, and negative state/permission paths
    - ✅ Deleted `tests/unit/test_collection_service.py`

#### Hand-Off Summary (2025-10-16)
- **Environment**: Integration tests expect Postgres credentials from `.env` (`POSTGRES_USER=semantik`, `POSTGRES_DB=semantik_test`, `POSTGRES_PASSWORD=…`). If the database doesn't exist, run `psql "${DATABASE_URL%/*}/postgres" -c "CREATE DATABASE ${POSTGRES_DB};"`.
- **Smoke Commands**:
  - `POSTGRES_USER=… POSTGRES_PASSWORD=… POSTGRES_DB=semantik_test POSTGRES_HOST=localhost POSTGRES_PORT=5432 DATABASE_URL=postgresql://semantik:…@localhost:5432/semantik_test uv run pytest tests/integration/services/test_collection_service.py -q`
  - `uv run pytest tests/unit/test_auth.py tests/integration/test_auth_api.py -q`
- **Completed P0 Items**:
  - Removed `tests/test_auth.py`; coverage resides in existing unit/integration suites.
  - Replaced `tests/unit/test_collection_repository.py` with `tests/integration/repositories/test_collection_repository.py` (real DB coverage).
  - Deleted `tests/unit/test_collection_service.py`; new integration suite provides end-to-end coverage with real repositories and stubbed Celery/Qdrant.
- **Fixtures & Patterns**:
  - Use `user_factory` (defined in the integration suite) for unique users; shared factories (`collection_factory`, `document_factory`) live in `tests/conftest.py`.
  - External integrations (Celery, Qdrant) are stubbed in tests to keep runs deterministic while exercising real DB side effects.
- **Remaining Work (P1/P2)**:
  - Relocate/split `tests/unit/test_all_chunking_strategies.py`.
  - Clean up websocket E2E suites, embedding tests, and other mocking anti-patterns listed under P1/P2.
  - Address security/performance flakiness outlined later in this document.
- **Next Developer Guidance**:
  1. Tackle the remaining ⏳ items in this tracking table (chunking strategies, websocket tests, etc.).
  2. Follow the integration-test pattern (real DB + stubbed externals) when migrating additional suites.
  3. Update this document immediately after each milestone; it's the single source of truth for test quality status.
  - [ ] Create factory fixtures
  - [ ] Create integration test file
  - [ ] Convert create tests
  - [ ] Convert update tests
  - [ ] Convert delete tests

### Sprint 2 (Week 3-4)

- [ ] Complete `test_collection_service.py` refactor
  - [ ] Add transaction rollback tests
  - [ ] Extract validation to unit tests
  - [ ] Delete old file

- [ ] Action 4: Reorganize test structure (1 day)
  - [ ] Create new directories
  - [ ] Move misplaced files
  - [ ] Add test markers
  - [ ] Update pytest.ini

- [ ] Action 5: Extract shared E2E fixtures (4 hours)
  - [ ] Create tests/e2e/conftest.py
  - [ ] Implement fixtures
  - [ ] Update both websocket test files

- [ ] Action 6: Add negative test cases (2-3 days)
  - [ ] Add error handling tests
  - [ ] Add validation tests
  - [ ] Add boundary tests

- [ ] Action 7: Fix performance test (4 hours)
  - [ ] Implement statistical approach
  - [ ] Add warmup iterations
  - [ ] Use p50/p95 metrics

### Sprint 3 (Week 5-6)

- [ ] Action 8: Break down giant tests (2 days)
  - [ ] Extract helper methods
  - [ ] Split websocket tests
  - [ ] Split chunking tests

- [ ] Add concurrent operation tests (2 days)
  - [ ] Repository concurrency tests
  - [ ] Service concurrency tests

- [ ] Documentation (1 day)
  - [ ] Update test README
  - [ ] Document fixtures
  - [ ] Add troubleshooting guide

- [ ] Final verification (1 day)
  - [ ] Run full test suite
  - [ ] Verify coverage targets
  - [ ] Update tracking document

---

## Success Criteria

### Definition of Done

A test file is considered "fixed" when:

1. ✅ Tests are in correct directory (unit/integration/e2e/performance)
2. ✅ Mocking limited to external services only (Celery, Qdrant, external APIs)
3. ✅ All tests have clear assertions (no print-only tests)
4. ✅ No persistent state pollution (unique test data, proper cleanup)
5. ✅ Test methods <40 lines each
6. ✅ Negative test cases cover common error scenarios
7. ✅ Test names clearly describe behavior being validated
8. ✅ No code duplication (shared fixtures/helpers used)
9. ✅ Performance tests use statistical approach (no flaky timing)
10. ✅ All tests can run independently (no order dependencies)

### Final Targets

After all actions completed:

- ✅ Zero tests with mock integration pattern
- ✅ 90%+ integration test coverage with real database
- ✅ <100 lines of mock setup code (only for external services)
- ✅ All tests in correct directories
- ✅ Unit tests <10ms each
- ✅ Integration tests <1s each
- ✅ E2E tests <5min each
- ✅ 80%+ negative test coverage
- ✅ Zero files with >500 lines
- ✅ Zero flaky tests in CI

---

## Risk Register

### High Risk

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Breaking changes during refactor | High | Medium | Keep old tests until new ones pass |
| Time overrun on P0 items | High | Medium | Focus on P0 first, defer P2 if needed |
| Database setup complexity | Medium | Low | Use existing async_session fixture |
| Test failures after migration | High | Medium | Run both old and new tests in parallel initially |

### Medium Risk

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Insufficient coverage of edge cases | Medium | Medium | Code review after each action |
| Performance regression | Medium | Low | Benchmark before/after |
| CI/CD pipeline changes needed | Low | High | Update CI config incrementally |

---

## Maintenance Plan

### Weekly Reviews (During Implementation)

- Review progress against sprint goals
- Update status checkboxes in this document
- Document any blockers or new issues discovered
- Adjust timeline if needed

### Post-Implementation

- Monthly review of test metrics
- Quarterly audit for anti-pattern regression
- Update this document with new findings
- Train team on test quality standards

### Monitoring

Key metrics to track:
- Test execution time (by category)
- Test flakiness rate
- Code coverage percentage
- Number of mock lines in test suite
- Ratio of unit:integration:e2e tests

---

## References

- **Action Plan**: `/home/john/semantik/docs/TEST_QUALITY_ACTION_PLAN.md`
- **Project Testing Guide**: `/home/john/semantik/tests/CLAUDE.md`
- **Architecture Guide**: `/home/john/semantik/CLAUDE.md`
- **API Reference**: `/home/john/semantik/docs/API_REFERENCE.md`

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2025-10-16 | Initial test quality assessment | Claude Code |
| | | |
| | | |

---

**Next Review Date**: TBD (after Sprint 1 completion)
**Document Owner**: Development Team
**Last Updated**: 2025-10-16

### Remaining Suites Overview

- **tests/unit/test_chunking_service.py** and related repository/service unit tests continue the mock-heavy pattern; Action 14 now covers consolidating these into integration coverage.
- **WebSocket performance/e2e scripts** (`tests/e2e/test_websocket_performance.py`, `tests/e2e/test_websocket_reindex.py`, `tests/websocket/test_cleanup.py`, `tests/websocket/test_scaling.py`) behave like manual load tests and should be relocated or rewritten as deterministic automation.
- **Performance scripts** (`tests/performance/chunking_benchmarks.py`, `tests/streaming/validate_streaming_pipeline.py`) belong in the manual/QA bucket alongside other probes (Action 3).
- **API/DTO suites** (`tests/webui/api/v2/test_collections.py`, `tests/webui/api/v2/test_directory_scan.py`, `tests/webui/services/dtos/test_chunking_dtos.py`) should be replaced with focused integration tests using real dependency overrides.
- **Security/exception suites** (`tests/security/test_redos_prevention.py`, `tests/unit/test_chunking_security.py`, `tests/unit/test_chunking_exception_handlers.py`) need refactoring to assert behavior rather than inspect mock internals.
- **Database/partition tests** (`tests/database/test_partitioning.py`, `tests/unit/test_partition_utils.py`) must run against the test DB instead of mocking SQLAlchemy internals.

These items are now tracked under Actions 3, 12, 13, and 14 for remediation planning.
