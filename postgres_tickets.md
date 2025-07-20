# PostgreSQL Migration Tickets

## Overview
This document contains sequential tickets for migrating Semantik from SQLite to PostgreSQL. Each ticket includes clear objectives, implementation details, and acceptance criteria.

---

## Phase 1: Infrastructure Setup

### Ticket 1.1: Add PostgreSQL Dependencies
**Priority**: High  
**Estimated Time**: 1 hour  
**Dependencies**: None  

**Description**: Add PostgreSQL Python dependencies and update project configuration.

**Tasks**:
- [ ] Add `psycopg2-binary = "^2.9.9"` to pyproject.toml
- [ ] Add `sqlalchemy[postgresql]` extras if not present
- [ ] Run `poetry lock` to update lock file
- [ ] Run `poetry install` to verify installation
- [ ] Test import of psycopg2 in Python shell

**Acceptance Criteria**:
- Dependencies installed without conflicts
- Can successfully import psycopg2
- poetry.lock file updated and committed

---

### Ticket 1.2: Docker PostgreSQL Service Setup
**Priority**: High  
**Estimated Time**: 2 hours  
**Dependencies**: Ticket 1.1  

**Description**: Add PostgreSQL service to Docker Compose configuration.

**Tasks**:
- [ ] Create `docker-compose.postgres.yml` with PostgreSQL service definition
- [ ] Add PostgreSQL environment variables to `.env.example`
- [ ] Create `scripts/postgres/init.sql` for initial database setup
- [ ] Update Makefile with PostgreSQL-specific commands
- [ ] Test PostgreSQL container startup and health check

**Implementation**:
```yaml
# docker-compose.postgres.yml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: semantik-postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-semantik}
      POSTGRES_USER: ${POSTGRES_USER:-semantik}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - semantik-network

volumes:
  postgres_data:
```

**Acceptance Criteria**:
- PostgreSQL container starts successfully
- Can connect to PostgreSQL using psql
- Health check passes
- Persisted data survives container restart

---

### Ticket 1.3: Connection Configuration
**Priority**: High  
**Estimated Time**: 2 hours  
**Dependencies**: Ticket 1.2  

**Description**: Implement PostgreSQL connection configuration and pooling.

**Tasks**:
- [ ] Create `packages/shared/config/postgres.py` with connection settings
- [ ] Add PostgreSQL URL format validation
- [ ] Implement connection pool configuration
- [ ] Add retry logic for transient connection errors
- [ ] Create connection testing script

**Implementation**:
```python
# packages/shared/config/postgres.py
from sqlalchemy import create_engine, pool
from sqlalchemy.engine import Engine
import os

def get_postgres_engine() -> Engine:
    """Create PostgreSQL engine with optimized settings"""
    database_url = os.getenv("DATABASE_URL", "").replace("sqlite://", "postgresql://")
    
    return create_engine(
        database_url,
        poolclass=pool.QueuePool,
        pool_size=20,
        max_overflow=40,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={
            "connect_timeout": 10,
            "application_name": "semantik",
            "options": "-c statement_timeout=30000"
        }
    )
```

**Acceptance Criteria**:
- Connection configuration properly handles PostgreSQL URLs
- Connection pooling is functional
- Retry logic handles temporary connection failures
- Configuration is environment-aware

---

## Phase 2: Code Implementation

### Ticket 2.1: PostgreSQL Repository Base Class
**Priority**: High  
**Estimated Time**: 3 hours  
**Dependencies**: Ticket 1.3  

**Description**: Create PostgreSQL-specific base repository with optimizations.

**Tasks**:
- [ ] Create `packages/webui/repositories/postgres/base.py`
- [ ] Implement bulk insert using PostgreSQL features
- [ ] Add upsert functionality with ON CONFLICT
- [ ] Implement PostgreSQL-specific error handling
- [ ] Add transaction management utilities

**Implementation**:
```python
# packages/webui/repositories/postgres/base.py
from typing import List, TypeVar, Generic
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

T = TypeVar('T')

class PostgreSQLBaseRepository(Generic[T]):
    def __init__(self, db: Session, model: T):
        self.db = db
        self.model = model
    
    def bulk_upsert(self, items: List[dict], unique_columns: List[str]) -> int:
        """Bulk upsert using PostgreSQL ON CONFLICT"""
        stmt = insert(self.model).values(items)
        stmt = stmt.on_conflict_do_update(
            index_elements=unique_columns,
            set_={k: stmt.excluded[k] for k in items[0].keys()}
        )
        result = self.db.execute(stmt)
        self.db.commit()
        return result.rowcount
```

**Acceptance Criteria**:
- Base repository class supports PostgreSQL-specific features
- Bulk operations are optimized for PostgreSQL
- Error handling is database-specific
- All methods have proper type hints

---

### Ticket 2.2: Repository Implementations
**Priority**: High  
**Estimated Time**: 4 hours  
**Dependencies**: Ticket 2.1  

**Description**: Create PostgreSQL implementations for all repositories.

**Tasks**:
- [ ] Implement PostgreSQL UserRepository
- [ ] Implement PostgreSQL CollectionRepository
- [ ] Implement PostgreSQL OperationRepository
- [ ] Implement PostgreSQL ApiKeyRepository
- [ ] Update repository factory to select implementation

**Repositories to implement**:
1. `packages/webui/repositories/postgres/user.py`
2. `packages/webui/repositories/postgres/collection.py`
3. `packages/webui/repositories/postgres/operation.py`
4. `packages/webui/repositories/postgres/api_key.py`

**Acceptance Criteria**:
- All repository interfaces have PostgreSQL implementations
- Factory pattern correctly selects implementation based on database URL
- All existing repository tests pass with PostgreSQL
- No SQLite-specific code in PostgreSQL implementations

---

### Ticket 2.3: Database Factory Update
**Priority**: High  
**Estimated Time**: 2 hours  
**Dependencies**: Ticket 2.2  

**Description**: Update database factory to support PostgreSQL.

**Tasks**:
- [ ] Modify `get_db_engine()` to detect PostgreSQL URLs
- [ ] Update `get_repository()` factory functions
- [ ] Add database type detection utility
- [ ] Ensure backward compatibility with SQLite
- [ ] Add logging for database type selection

**Implementation**:
```python
# packages/shared/database/factory.py
def get_database_type(database_url: str) -> str:
    """Detect database type from URL"""
    if database_url.startswith("postgresql://") or database_url.startswith("postgres://"):
        return "postgresql"
    elif database_url.startswith("sqlite://"):
        return "sqlite"
    else:
        raise ValueError(f"Unsupported database URL: {database_url}")

def get_repository(repo_type: str, db: Session):
    """Factory function for repository creation"""
    db_type = get_database_type(str(db.bind.url))
    
    if db_type == "postgresql":
        from packages.webui.repositories.postgres import (
            PostgreSQLUserRepository,
            PostgreSQLCollectionRepository,
            # ... other repositories
        )
        repositories = {
            "user": PostgreSQLUserRepository,
            "collection": PostgreSQLCollectionRepository,
            # ... other mappings
        }
    else:
        # Existing SQLite implementations
        pass
    
    return repositories[repo_type](db)
```

**Acceptance Criteria**:
- Factory correctly identifies database type
- Appropriate repository implementation is selected
- SQLite functionality remains unchanged
- Clear error messages for unsupported databases

---

## Phase 3: Data Migration

### Ticket 3.1: Migration Script Development
**Priority**: High  
**Estimated Time**: 4 hours  
**Dependencies**: Ticket 2.3  

**Description**: Create comprehensive data migration script from SQLite to PostgreSQL.

**Tasks**:
- [ ] Create `scripts/migrate_sqlite_to_postgres.py`
- [ ] Implement table-by-table migration logic
- [ ] Add progress tracking and logging
- [ ] Implement data validation checks
- [ ] Add dry-run mode for testing

**Implementation**:
```python
# scripts/migrate_sqlite_to_postgres.py
import click
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import logging
from tqdm import tqdm

class SQLiteToPostgresMigrator:
    def __init__(self, sqlite_url: str, postgres_url: str, batch_size: int = 1000):
        self.sqlite_engine = create_engine(sqlite_url)
        self.postgres_engine = create_engine(postgres_url)
        self.batch_size = batch_size
        
    def migrate_table(self, table_name: str) -> int:
        """Migrate a single table with progress bar"""
        # Implementation
        
    def verify_migration(self) -> dict:
        """Verify row counts match between databases"""
        # Implementation

@click.command()
@click.option('--sqlite-url', required=True, help='Source SQLite database URL')
@click.option('--postgres-url', required=True, help='Target PostgreSQL database URL')
@click.option('--dry-run', is_flag=True, help='Perform dry run without actual migration')
def main(sqlite_url, postgres_url, dry_run):
    migrator = SQLiteToPostgresMigrator(sqlite_url, postgres_url)
    # Migration logic
```

**Acceptance Criteria**:
- Script successfully migrates all tables
- Progress is clearly displayed
- Validation confirms data integrity
- Dry-run mode works correctly
- Errors are handled gracefully

---

### Ticket 3.2: Migration Testing
**Priority**: High  
**Estimated Time**: 3 hours  
**Dependencies**: Ticket 3.1  

**Description**: Test migration script with sample data and edge cases.

**Tasks**:
- [ ] Create test SQLite database with sample data
- [ ] Test migration with various data types
- [ ] Test handling of NULL values and constraints
- [ ] Test large dataset migration (>100k rows)
- [ ] Document migration performance metrics

**Test scenarios**:
1. Empty database migration
2. Small dataset (<1000 rows)
3. Medium dataset (10k-100k rows)
4. Large dataset (>1M rows)
5. Database with all data types
6. Database with foreign key constraints

**Acceptance Criteria**:
- All test scenarios pass successfully
- Performance metrics are documented
- Edge cases are handled properly
- No data loss in any scenario

---

### Ticket 3.3: Alembic Migration Compatibility
**Priority**: Medium  
**Estimated Time**: 2 hours  
**Dependencies**: Ticket 3.2  

**Description**: Ensure Alembic migrations work with PostgreSQL.

**Tasks**:
- [ ] Review all existing Alembic migrations
- [ ] Test migrations on fresh PostgreSQL database
- [ ] Update any SQLite-specific migrations
- [ ] Add PostgreSQL-specific indexes if needed
- [ ] Document any migration differences

**Acceptance Criteria**:
- All Alembic migrations run successfully on PostgreSQL
- No SQLite-specific syntax in migrations
- Migration history is consistent
- Rollback functionality works

---

## Phase 4: Testing & Validation

### Ticket 4.1: Unit Test Updates
**Priority**: High  
**Estimated Time**: 3 hours  
**Dependencies**: Ticket 3.3  

**Description**: Update unit tests to support PostgreSQL.

**Tasks**:
- [ ] Update test database fixtures for PostgreSQL
- [ ] Add PostgreSQL test container setup
- [ ] Update repository tests for both databases
- [ ] Ensure SQLite tests still pass
- [ ] Add PostgreSQL-specific test cases

**Implementation**:
```python
# tests/conftest.py
@pytest.fixture
def postgres_db():
    """PostgreSQL test database fixture"""
    # Use testcontainers-python for PostgreSQL
    from testcontainers.postgres import PostgresContainer
    
    with PostgresContainer("postgres:15-alpine") as postgres:
        engine = create_engine(postgres.get_connection_url())
        # Setup database
        yield engine
```

**Acceptance Criteria**:
- All tests pass with PostgreSQL backend
- SQLite tests remain functional
- Test execution time is reasonable
- Coverage remains at current levels

---

### Ticket 4.2: Integration Testing
**Priority**: High  
**Estimated Time**: 4 hours  
**Dependencies**: Ticket 4.1  

**Description**: Create integration tests for PostgreSQL-specific features.

**Tasks**:
- [ ] Test concurrent write operations
- [ ] Test transaction isolation levels
- [ ] Test connection pool behavior
- [ ] Test large dataset operations
- [ ] Test backup/restore procedures

**Test cases**:
1. Concurrent collection creation
2. Parallel document indexing
3. Transaction rollback scenarios
4. Connection pool exhaustion
5. Database recovery testing

**Acceptance Criteria**:
- All integration tests pass
- Concurrent operations work correctly
- No deadlocks or race conditions
- Performance meets expectations

---

### Ticket 4.3: Performance Benchmarking
**Priority**: Medium  
**Estimated Time**: 3 hours  
**Dependencies**: Ticket 4.2  

**Description**: Benchmark PostgreSQL performance against SQLite baseline.

**Tasks**:
- [ ] Create benchmark suite for common operations
- [ ] Measure query performance (SELECT, INSERT, UPDATE)
- [ ] Test concurrent operation throughput
- [ ] Measure resource utilization
- [ ] Document performance comparison

**Benchmarks to run**:
1. Single document search
2. Bulk document insertion
3. Collection creation/deletion
4. Concurrent user operations
5. Complex query performance

**Acceptance Criteria**:
- Benchmarks show expected improvements
- No significant performance regressions
- Results are documented
- Bottlenecks are identified

---

## Phase 5: Deployment

### Ticket 5.1: Staging Environment Setup
**Priority**: High  
**Estimated Time**: 2 hours  
**Dependencies**: Ticket 4.3  

**Description**: Deploy PostgreSQL to staging environment.

**Tasks**:
- [ ] Provision PostgreSQL instance in staging
- [ ] Configure security groups and networking
- [ ] Set up automated backups
- [ ] Configure monitoring and alerts
- [ ] Test connectivity from application

**Acceptance Criteria**:
- PostgreSQL accessible from application
- Backups are configured
- Monitoring is operational
- Security best practices followed

---

### Ticket 5.2: Staging Migration Execution
**Priority**: High  
**Estimated Time**: 3 hours  
**Dependencies**: Ticket 5.1  

**Description**: Execute full migration in staging environment.

**Tasks**:
- [ ] Backup current staging SQLite database
- [ ] Run migration script in staging
- [ ] Verify data integrity
- [ ] Run full test suite
- [ ] Monitor application behavior

**Acceptance Criteria**:
- Migration completes without errors
- All data is correctly migrated
- Application functions normally
- No performance issues observed

---

### Ticket 5.3: Production Migration Plan
**Priority**: High  
**Estimated Time**: 2 hours  
**Dependencies**: Ticket 5.2  

**Description**: Create detailed production migration runbook.

**Tasks**:
- [ ] Document step-by-step migration process
- [ ] Create rollback procedures
- [ ] Define success criteria
- [ ] Plan maintenance window
- [ ] Prepare communication templates

**Runbook sections**:
1. Pre-migration checklist
2. Migration execution steps
3. Validation procedures
4. Rollback process
5. Post-migration tasks

**Acceptance Criteria**:
- Runbook is comprehensive
- Rollback process is tested
- All stakeholders approve plan
- Timing estimates are accurate

---

### Ticket 5.4: Production Migration Execution
**Priority**: Critical  
**Estimated Time**: 4 hours  
**Dependencies**: Ticket 5.3  

**Description**: Execute production migration to PostgreSQL.

**Tasks**:
- [ ] Execute pre-migration checklist
- [ ] Take final SQLite backup
- [ ] Run migration script
- [ ] Validate data integrity
- [ ] Update application configuration
- [ ] Monitor application health

**Acceptance Criteria**:
- Migration completes successfully
- Zero data loss
- Application fully functional
- Performance metrics are normal
- No user-reported issues

---

### Ticket 5.5: Post-Migration Cleanup
**Priority**: Medium  
**Estimated Time**: 2 hours  
**Dependencies**: Ticket 5.4  

**Description**: Clean up post-migration and document lessons learned.

**Tasks**:
- [ ] Archive SQLite database
- [ ] Remove SQLite-specific code (if applicable)
- [ ] Update documentation
- [ ] Document lessons learned
- [ ] Plan future optimizations

**Acceptance Criteria**:
- Old database safely archived
- Documentation updated
- Team knowledge shared
- Future improvements identified

---

## Summary

**Total Tickets**: 20  
**Estimated Total Time**: 55-65 hours  
**Critical Path**: Tickets 1.1 → 1.2 → 1.3 → 2.1 → 2.2 → 2.3 → 3.1 → 5.4

**Risk Mitigation**:
- Each phase has validation before proceeding
- Rollback procedures at every step
- Comprehensive testing before production
- Parallel development where possible