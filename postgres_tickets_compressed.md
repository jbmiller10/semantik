# PostgreSQL Migration Tickets (Compressed)

## Overview
This document contains 5 phase-based tickets for migrating Semantik from SQLite to PostgreSQL. Each ticket represents a complete phase of work.

### Pros of This Approach:
- ✅ Simpler tracking and management
- ✅ Better for small teams (1-2 developers)
- ✅ Clearer big-picture progress
- ✅ Less administrative overhead

### Cons of This Approach:
- ❌ Harder to parallelize work
- ❌ Tickets may become blocked longer
- ❌ Less granular progress visibility
- ❌ Harder to accurately estimate

---

## Ticket 1: Infrastructure Setup and Configuration
**Priority**: Critical  
**Estimated Time**: 5-6 hours  
**Dependencies**: None  
**Assignee**: DevOps/Backend Lead  

**Description**: Set up complete PostgreSQL infrastructure including dependencies, Docker configuration, and connection management.

**Deliverables**:
1. **Dependencies** (1 hour)
   - Add `psycopg2-binary = "^2.9.9"` to pyproject.toml
   - Add `sqlalchemy[postgresql]` extras
   - Verify installation with poetry

2. **Docker Setup** (2 hours)
   - Create `docker-compose.postgres.yml` with PostgreSQL service
   - Configure health checks and volumes
   - Add PostgreSQL commands to Makefile
   - Test container startup and persistence

3. **Connection Configuration** (2-3 hours)
   - Create `packages/shared/config/postgres.py`
   - Implement connection pooling with proper settings
   - Add retry logic for transient errors
   - Create connection testing utilities

**Acceptance Criteria**:
- [ ] PostgreSQL container runs successfully with health checks
- [ ] Python application can connect to PostgreSQL
- [ ] Connection pooling is configured and tested
- [ ] All configuration is environment-variable driven
- [ ] Docker commands added to Makefile

**Implementation Notes**:
```yaml
# Key PostgreSQL settings
POSTGRES_DB=semantik
POSTGRES_USER=semantik
POSTGRES_PASSWORD=<secure-password>
POSTGRES_PORT=5432

# Connection pool settings
pool_size=20
max_overflow=40
pool_pre_ping=True
pool_recycle=3600
```

---

## Ticket 2: Repository Pattern Implementation
**Priority**: Critical  
**Estimated Time**: 8-10 hours  
**Dependencies**: Ticket 1  
**Assignee**: Backend Developer  

**Description**: Implement PostgreSQL-specific repositories maintaining compatibility with existing SQLite implementation.

**Deliverables**:
1. **Base Repository** (3 hours)
   - Create `packages/webui/repositories/postgres/base.py`
   - Implement bulk operations with PostgreSQL optimizations
   - Add ON CONFLICT upsert support
   - Handle PostgreSQL-specific errors

2. **Repository Implementations** (4 hours)
   - PostgreSQL UserRepository
   - PostgreSQL CollectionRepository  
   - PostgreSQL OperationRepository
   - PostgreSQL ApiKeyRepository

3. **Factory Pattern Update** (2-3 hours)
   - Update database detection logic
   - Modify repository factory functions
   - Ensure backward compatibility
   - Add comprehensive logging

**Acceptance Criteria**:
- [ ] All repositories have PostgreSQL implementations
- [ ] Factory correctly selects implementation based on DATABASE_URL
- [ ] Existing SQLite functionality unchanged
- [ ] All repository tests pass with both backends
- [ ] No SQL injection vulnerabilities

**Key Code Structure**:
```python
# Repository factory pattern
def get_repository(repo_type: str, db: Session):
    db_type = detect_database_type(db)
    if db_type == "postgresql":
        return postgresql_repositories[repo_type](db)
    return sqlite_repositories[repo_type](db)
```

---

## Ticket 3: Data Migration Implementation
**Priority**: High  
**Estimated Time**: 8-10 hours  
**Dependencies**: Ticket 2  
**Assignee**: Backend Developer/DBA  

**Description**: Create and test comprehensive data migration from SQLite to PostgreSQL.

**Deliverables**:
1. **Migration Script** (4 hours)
   - Create `scripts/migrate_sqlite_to_postgres.py`
   - Implement table-by-table migration with progress tracking
   - Add data validation and integrity checks
   - Include dry-run mode

2. **Migration Testing** (3 hours)
   - Test with various data sizes (empty, small, large)
   - Verify handling of all data types
   - Test constraint and foreign key preservation
   - Document performance metrics

3. **Alembic Compatibility** (2 hours)
   - Verify all migrations work with PostgreSQL
   - Update any SQLite-specific migrations
   - Test migration up/down functionality

**Acceptance Criteria**:
- [ ] Migration script handles all tables correctly
- [ ] 100% data integrity maintained
- [ ] Dry-run mode provides accurate preview
- [ ] Performance documented for different data sizes
- [ ] Alembic migrations fully compatible

**Migration Checklist**:
```
Tables to migrate (in order):
1. users
2. api_keys (depends on users)
3. collections
4. operations (depends on collections)
5. documents
6. chunks (depends on documents)
```

---

## Ticket 4: Testing and Performance Validation
**Priority**: High  
**Estimated Time**: 10-12 hours  
**Dependencies**: Ticket 3  
**Assignee**: QA/Backend Developer  

**Description**: Comprehensive testing of PostgreSQL implementation including unit, integration, and performance tests.

**Deliverables**:
1. **Unit Test Updates** (3 hours)
   - Update test fixtures for PostgreSQL
   - Add PostgreSQL test container setup
   - Ensure dual-database test support
   - Maintain existing test coverage

2. **Integration Testing** (4 hours)
   - Test concurrent write operations
   - Verify transaction isolation
   - Test connection pool behavior
   - Validate error handling

3. **Performance Benchmarking** (3-4 hours)
   - Benchmark against SQLite baseline
   - Test concurrent user scenarios
   - Measure query performance
   - Document resource utilization

**Acceptance Criteria**:
- [ ] All tests pass with PostgreSQL backend
- [ ] SQLite tests remain functional
- [ ] No race conditions or deadlocks
- [ ] Performance meets or exceeds SQLite for concurrent operations
- [ ] Test execution time reasonable (<10 min)

**Key Test Scenarios**:
- Concurrent collection creation (10+ simultaneous)
- Parallel document indexing (1000+ documents)
- Transaction rollback under load
- Connection pool exhaustion recovery

---

## Ticket 5: Production Deployment and Migration
**Priority**: Critical  
**Estimated Time**: 10-12 hours  
**Dependencies**: Ticket 4  
**Assignee**: DevOps/Team Lead  

**Description**: Deploy PostgreSQL to production and execute migration with zero downtime.

**Deliverables**:
1. **Staging Deployment** (3 hours)
   - Deploy PostgreSQL to staging
   - Execute full migration
   - Run complete test suite
   - Monitor for 24 hours

2. **Production Preparation** (2 hours)
   - Create detailed runbook
   - Document rollback procedures
   - Schedule maintenance window
   - Prepare stakeholder communications

3. **Production Migration** (4 hours)
   - Execute pre-migration backup
   - Run migration with monitoring
   - Validate data integrity
   - Update application configuration
   - Monitor application health

4. **Post-Migration** (2 hours)
   - Archive SQLite database
   - Update documentation
   - Remove feature flags
   - Document lessons learned

**Acceptance Criteria**:
- [ ] Zero data loss during migration
- [ ] Application fully functional post-migration
- [ ] Rollback procedure tested and documented
- [ ] Performance metrics meet expectations
- [ ] All documentation updated

**Go/No-Go Checklist**:
```
□ Staging migration successful
□ All tests passing
□ Rollback tested
□ Team available for support
□ Maintenance window approved
□ Backup verified
```

---

## Summary

**Total Tickets**: 5  
**Total Estimated Time**: 41-50 hours  
**Critical Path**: All tickets must be completed sequentially

### Recommended Approach:
- **Small team (1-2 devs)**: Use these compressed tickets
- **Larger team (3+ devs)**: Use the detailed 20-ticket version for better parallelization

### Risk Mitigation:
- Each ticket has internal checkpoints
- Rollback possible at each phase
- Staging validation before production
- Comprehensive testing throughout