# PR #178 Cleanup Tickets

## Context
PR #178 implements a comprehensive chunking feature but has accumulated technical debt over 18 merged sub-PRs. The feature works but has critical issues that must be resolved before merging to main. Each ticket below is self-contained and can be executed independently by a stateless LLM agent.

---

## TICKET-001: Fix SQL Injection Vulnerability in Partition Creation

### Priority: CRITICAL
### Estimated Hours: 2-3

### Setup Instructions
1. Ensure you're on the latest feature branch:
   ```bash
   git checkout feature/improve-chunking
   git pull origin feature/improve-chunking
   ```
2. Create a new branch for this ticket:
   ```bash
   git checkout -b fix/ticket-001-sql-injection
   ```

### Context
The migration file `/home/john/semantik/alembic/versions/52db15bd2686_add_chunking_tables_with_partitioning.py` contains SQL injection vulnerabilities through direct string interpolation in SQL commands.

### Problem
Lines 145-148 use f-strings to construct SQL:
```python
f"""CREATE TABLE chunks_p{i} PARTITION OF chunks..."""
```

While `partition_count` currently comes from environment variables, this pattern is dangerous and violates security best practices. Direct string interpolation in SQL should never be used, even with "trusted" sources.

### Requirements
1. Find all instances of f-string SQL construction in the migration files
2. Replace with parameterized queries where possible
3. Where parameterization isn't possible (DDL statements), validate inputs as integers before use
4. Ensure no user input can ever reach these SQL construction points

### Implementation Approach
- Search for patterns: `op.execute(f"`, `conn.execute(text(f"`, and similar
- For partition creation, validate partition_count as integer: `int(os.environ.get(...))` 
- Add explicit bounds checking (e.g., 1 <= partition_count <= 1000)
- Consider using SQLAlchemy's DDL constructs instead of raw SQL where possible

### Files to Check
- `/home/john/semantik/alembic/versions/52db15bd2686_add_chunking_tables_with_partitioning.py`
- `/home/john/semantik/alembic/versions/ae558c9e183f_implement_100_direct_list_partitions.py`
- Any other migration files in the feature branch

### Success Criteria
- No direct string interpolation in SQL statements
- All dynamic values validated before use in DDL
- Security scanner passes without SQL injection warnings

### Completion Steps
1. Run linting and fix any issues:
   ```bash
   make lint
   # or
   poetry run ruff check --fix
   poetry run mypy packages/
   ```
2. Run tests to ensure nothing broke:
   ```bash
   make test
   ```
3. Commit and push changes:
   ```bash
   git add -A
   git commit -m "fix(security): resolve SQL injection vulnerabilities in partition creation
   
   - Validated all dynamic SQL values as integers
   - Added bounds checking for partition_count
   - Replaced f-string SQL construction with safer patterns
   
   Fixes TICKET-001"
   git push origin fix/ticket-001-sql-injection
   ```
4. Create PR against feature/improve-chunking:
   ```bash
   gh pr create --base feature/improve-chunking --title "Fix SQL injection in partition creation" --body "Resolves TICKET-001: SQL injection vulnerabilities in migration files"
   ```

---

## TICKET-002: Fix Broken Partition Strategy

### Priority: CRITICAL  
### Estimated Hours: 8-12

### Setup Instructions
1. Ensure you're on the latest feature branch:
   ```bash
   git checkout feature/improve-chunking
   git pull origin feature/improve-chunking
   ```
2. Create a new branch for this ticket:
   ```bash
   git checkout -b fix/ticket-002-partition-strategy
   ```

### Context
The current partition strategy uses `abs(hashtext(collection_id)) % 100` which means ALL chunks for a given collection go to the SAME partition. This completely defeats the purpose of partitioning for large collections.

### Problem
File: `/home/john/semantik/alembic/versions/ae558c9e183f_implement_100_direct_list_partitions.py`
Line 98: `NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;`

This means a collection with 1 million chunks will have all chunks in one partition, causing severe performance degradation.

### Requirements
1. Design a partition strategy that distributes chunks WITHIN a collection across partitions
2. Maintain good query performance for both collection-wide and document-specific queries
3. Ensure even distribution across partitions
4. Minimize migration complexity for existing data

### Recommended Approaches (choose one):
1. **RANGE partitioning by timestamp**: Partition by created_at date ranges
2. **Composite hash**: Use `hash(collection_id + document_id) % 100`
3. **Round-robin with sequence**: Add a sequence number per collection, partition by `sequence % 100`
4. **Hybrid**: First 10 partitions for recent data (by date), remaining 90 by hash

### Implementation Considerations
- The chosen strategy must support efficient queries for:
  - All chunks in a collection
  - All chunks for a specific document
  - Recent chunks across all collections
- Consider partition pruning capabilities
- Plan for future partition maintenance (adding/removing partitions)

### Pseudocode Example (Composite Hash):
```
partition_key = (hash(collection_id) + hash(document_id)) % 100
-- OR --
partition_key = hash(collection_id || ':' || document_id) % 100
```

### Files to Modify
- `/home/john/semantik/alembic/versions/ae558c9e183f_implement_100_direct_list_partitions.py`
- Create new migration to fix existing data distribution
- Update any partition monitoring views

### Success Criteria
- Chunks for large collections distributed across multiple partitions
- Query performance improved for large collections
- Even distribution verified through monitoring views

### Completion Steps
1. Run linting and fix any issues:
   ```bash
   make lint
   # or
   poetry run ruff check --fix
   poetry run mypy packages/
   ```
2. Run tests to ensure nothing broke:
   ```bash
   make test
   ```
3. Commit and push changes:
   ```bash
   git add -A
   git commit -m "fix(database): implement proper partition distribution strategy
   
   - Changed from collection-only hash to composite key partitioning
   - Chunks now distributed across partitions within collections
   - Added monitoring views for partition health
   
   Fixes TICKET-002"
   git push origin fix/ticket-002-partition-strategy
   ```
4. Create PR against feature/improve-chunking:
   ```bash
   gh pr create --base feature/improve-chunking --title "Fix partition distribution strategy" --body "Resolves TICKET-002: Broken partition strategy that put entire collections in single partitions"
   ```

---

## TICKET-003: Add Data Protection to Destructive Migrations

### Priority: HIGH
### Estimated Hours: 4-6

### Setup Instructions
1. Ensure you're on the latest feature branch:
   ```bash
   git checkout feature/improve-chunking
   git pull origin feature/improve-chunking
   ```
2. Create a new branch for this ticket:
   ```bash
   git checkout -b feat/ticket-003-migration-safety
   ```

### Context
Migration `/home/john/semantik/alembic/versions/ae558c9e183f_implement_100_direct_list_partitions.py` drops the chunks table with CASCADE, potentially losing all data without any backup mechanism.

### Problem
Line 85: `conn.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))`

This is extremely dangerous in production environments.

### Requirements
1. Implement backup mechanism before destructive operations
2. Add rollback capability in case of migration failure
3. Provide option to preserve existing data during migration
4. Add safety checks and confirmations for destructive operations

### Implementation Approach
1. **Create backup manager** in `/home/john/semantik/alembic/migrations_utils/backup_manager.py`
2. **Before dropping tables**:
   - Check if table has data
   - Create backup table with timestamp suffix
   - Copy data to backup table
   - Log backup location and row count
3. **Add safety flags**:
   - Environment variable `ALLOW_DESTRUCTIVE_MIGRATIONS=true` required
   - Check if we're in production environment
   - Add dry-run capability

### Pseudocode:
```python
# In migration file
if table_exists('chunks') and has_data('chunks'):
    if not os.getenv('ALLOW_DESTRUCTIVE_MIGRATIONS'):
        raise Exception("Destructive migration blocked. Set ALLOW_DESTRUCTIVE_MIGRATIONS=true")
    
    backup_table = f"chunks_backup_{timestamp}"
    create_table_like('chunks', backup_table)
    copy_data('chunks', backup_table)
    log.info(f"Backed up {row_count} rows to {backup_table}")
    
    # Then proceed with drop
```

### Files to Modify
- `/home/john/semantik/alembic/versions/ae558c9e183f_implement_100_direct_list_partitions.py`
- Create `/home/john/semantik/alembic/migrations_utils/backup_manager.py`
- Update other destructive migrations if any

### Success Criteria
- No data loss possible without explicit confirmation
- Backup created before any destructive operation
- Clear logging of what was backed up and where
- Ability to restore from backup if needed

### Completion Steps
1. Run linting and fix any issues:
   ```bash
   make lint
   # or
   poetry run ruff check --fix
   poetry run mypy packages/
   ```
2. Run tests to ensure nothing broke:
   ```bash
   make test
   ```
3. Commit and push changes:
   ```bash
   git add -A
   git commit -m "feat(migrations): add data protection for destructive operations
   
   - Created backup_manager utility for migration safety
   - Added ALLOW_DESTRUCTIVE_MIGRATIONS flag requirement
   - Implemented automatic backup before DROP operations
   - Added rollback capability with data preservation
   
   Fixes TICKET-003"
   git push origin feat/ticket-003-migration-safety
   ```
4. Create PR against feature/improve-chunking:
   ```bash
   gh pr create --base feature/improve-chunking --title "Add data protection to migrations" --body "Resolves TICKET-003: Adds backup mechanisms to prevent data loss during destructive migrations"
   ```

---

## TICKET-004: Fix WebSocket Race Conditions

### Priority: HIGH
### Estimated Hours: 6-8

### Setup Instructions
1. Ensure you're on the latest feature branch:
   ```bash
   git checkout feature/improve-chunking
   git pull origin feature/improve-chunking
   ```
2. Create a new branch for this ticket:
   ```bash
   git checkout -b fix/ticket-004-websocket-race
   ```

### Context
The WebSocket manager at `/home/john/semantik/packages/webui/websocket_manager.py` modifies shared dictionaries without proper locking in an async context, potentially causing connection state corruption.

### Problem
Lines 26-27:
```python
self.connections: dict[str, set[WebSocket]] = {}
self.consumer_tasks: dict[str, asyncio.Task] = {}
```

These dictionaries are accessed and modified from multiple async contexts without synchronization.

### Requirements
1. Add proper locking for all shared state modifications
2. Ensure thread-safety for connection management
3. Prevent dropped messages due to race conditions
4. Maintain performance while adding safety

### Implementation Approach
1. **Add locks for each shared resource**:
   ```python
   self._connections_lock = asyncio.Lock()
   self._tasks_lock = asyncio.Lock()
   ```

2. **Wrap all dictionary operations**:
   ```python
   async with self._connections_lock:
       # Modify self.connections
   ```

3. **Consider using thread-safe alternatives**:
   - `asyncio.Queue` for message passing
   - `weakref.WeakSet` for connection tracking
   - Context managers for connection lifecycle

### Critical Sections to Protect
- Adding/removing connections
- Starting/stopping consumer tasks
- Broadcasting messages to connections
- Checking connection limits

### Files to Modify
- `/home/john/semantik/packages/webui/websocket_manager.py`
- Related test files in `/home/john/semantik/tests/unit/test_websocket_manager.py`

### Success Criteria
- No race conditions under concurrent load
- Stress test with 100+ concurrent connections passes
- No dropped messages or corrupted state
- Performance impact < 5%

### Completion Steps
1. Run linting and fix any issues:
   ```bash
   make lint
   # or
   poetry run ruff check --fix
   poetry run mypy packages/
   ```
2. Run tests to ensure nothing broke:
   ```bash
   make test
   ```
3. Commit and push changes:
   ```bash
   git add -A
   git commit -m "fix(websocket): resolve race conditions in connection management
   
   - Added asyncio locks for shared state modifications
   - Protected connection dictionary operations
   - Fixed concurrent access issues in consumer tasks
   - Added stress test for 100+ connections
   
   Fixes TICKET-004"
   git push origin fix/ticket-004-websocket-race
   ```
4. Create PR against feature/improve-chunking:
   ```bash
   gh pr create --base feature/improve-chunking --title "Fix WebSocket race conditions" --body "Resolves TICKET-004: Thread-safety issues in WebSocket manager causing race conditions"
   ```

---

## TICKET-005: Fix Memory Leaks in React Components

### Priority: MEDIUM
### Estimated Hours: 4-6

### Setup Instructions
1. Ensure you're on the latest feature branch:
   ```bash
   git checkout feature/improve-chunking
   git pull origin feature/improve-chunking
   ```
2. Create a new branch for this ticket:
   ```bash
   git checkout -b fix/ticket-005-memory-leaks
   ```

### Context
The chunking preview component at `/home/john/semantik/apps/webui-react/src/components/chunking/ChunkingPreviewPanel.tsx` creates WebSocket connections and event listeners without proper cleanup.

### Problem
- WebSocket connections created in useEffect without cleanup
- Event listeners not removed on unmount
- Potential memory accumulation with repeated mounting/unmounting

### Requirements
1. Add cleanup functions to all useEffect hooks
2. Properly close WebSocket connections on unmount
3. Remove all event listeners when component unmounts
4. Clear any running timers or intervals

### Implementation Pattern
```javascript
useEffect(() => {
    // Setup code
    const connection = createWebSocket();
    const handler = (e) => { /* ... */ };
    window.addEventListener('resize', handler);
    
    // Cleanup function
    return () => {
        connection.close();
        window.removeEventListener('resize', handler);
        // Clear any other resources
    };
}, [dependencies]);
```

### Files to Review and Fix
- `/home/john/semantik/apps/webui-react/src/components/chunking/ChunkingPreviewPanel.tsx`
- `/home/john/semantik/apps/webui-react/src/hooks/useChunkingWebSocket.ts`
- Other chunking components in the same directory

### Specific Issues to Address
1. Line 86-101: useEffect without cleanup return
2. WebSocket connection lifecycle management
3. Event listener for scroll synchronization
4. Any setInterval or setTimeout calls

### Success Criteria
- Chrome DevTools memory profiler shows no leaks
- Component can be mounted/unmounted 100 times without memory growth
- All WebSocket connections properly closed
- No console warnings about unmounted component updates

### Completion Steps
1. Run linting and fix any issues:
   ```bash
   cd apps/webui-react
   npm run lint:fix
   npm run type-check
   ```
2. Run tests to ensure nothing broke:
   ```bash
   npm test
   ```
3. Commit and push changes:
   ```bash
   git add -A
   git commit -m "fix(frontend): resolve memory leaks in chunking components
   
   - Added cleanup functions to all useEffect hooks
   - Properly close WebSocket connections on unmount
   - Remove event listeners in cleanup
   - Clear timers and intervals on unmount
   
   Fixes TICKET-005"
   git push origin fix/ticket-005-memory-leaks
   ```
4. Create PR against feature/improve-chunking:
   ```bash
   gh pr create --base feature/improve-chunking --title "Fix React component memory leaks" --body "Resolves TICKET-005: Memory leaks in chunking preview components"
   ```

---

## TICKET-006: Refactor Business Logic from Routers to Services

### Priority: MEDIUM
### Estimated Hours: 10-14

### Setup Instructions
1. Ensure you're on the latest feature branch:
   ```bash
   git checkout feature/improve-chunking
   git pull origin feature/improve-chunking
   ```
2. Create a new branch for this ticket:
   ```bash
   git checkout -b refactor/ticket-006-service-layer
   ```

### Context
The project follows a three-layer architecture (Routers → Services → Repositories), but business logic has leaked into router files, violating separation of concerns. This is documented as an anti-pattern in `/home/john/semantik/CLAUDE.md`.

### Problem
Multiple router files contain business logic that should be in service layer:
- Direct database queries in routers
- Complex validation logic in endpoints
- Transaction management in routers

### Requirements
1. Identify all business logic in router files
2. Move logic to appropriate service classes
3. Routers should only handle HTTP concerns (request/response)
4. Services handle all business logic and orchestration

### Pattern to Follow
```python
# BAD - Router with business logic
@router.post("/")
async def create(data: Model, db: Session = Depends(get_db)):
    # Business logic in router - WRONG
    existing = db.query(Model).filter_by(name=data.name).first()
    if existing:
        raise HTTPException(409, "Already exists")
    model = Model(**data.dict())
    db.add(model)
    db.commit()
    return model

# GOOD - Router delegates to service
@router.post("/")
async def create(data: Model, service: Service = Depends(get_service)):
    # Router only handles HTTP concerns
    try:
        return await service.create(data)
    except AlreadyExistsError as e:
        raise HTTPException(409, str(e))
```

### Files to Review
- `/home/john/semantik/packages/webui/api/v2/chunking.py`
- `/home/john/semantik/packages/webui/api/v2/collections.py`
- `/home/john/semantik/packages/webui/api/v2/operations.py`
- Any new router files in the feature branch

### Service Files to Update
- `/home/john/semantik/packages/webui/services/chunking_service.py`
- `/home/john/semantik/packages/webui/services/collection_service.py`

### Success Criteria
- Routers contain no business logic
- All database queries in service or repository layer
- Clear separation of HTTP concerns from business logic
- Tests still pass after refactoring

### Completion Steps
1. Run linting and fix any issues:
   ```bash
   make lint
   # or
   poetry run ruff check --fix
   poetry run mypy packages/
   ```
2. Run tests to ensure nothing broke:
   ```bash
   make test
   ```
3. Commit and push changes:
   ```bash
   git add -A
   git commit -m "refactor(architecture): move business logic to service layer
   
   - Extracted business logic from routers to services
   - Routers now only handle HTTP concerns
   - Services manage transactions and orchestration
   - Improved separation of concerns
   
   Fixes TICKET-006"
   git push origin refactor/ticket-006-service-layer
   ```
4. Create PR against feature/improve-chunking:
   ```bash
   gh pr create --base feature/improve-chunking --title "Refactor business logic to service layer" --body "Resolves TICKET-006: Moves business logic from routers to proper service layer following three-tier architecture"
   ```

---

## TICKET-007: Add Error Boundaries to React Components

### Priority: MEDIUM
### Estimated Hours: 4-6

### Setup Instructions
1. Ensure you're on the latest feature branch:
   ```bash
   git checkout feature/improve-chunking
   git pull origin feature/improve-chunking
   ```
2. Create a new branch for this ticket:
   ```bash
   git checkout -b feat/ticket-007-error-boundaries
   ```

### Context
The chunking components lack error boundaries, meaning any JavaScript error in these components will crash the entire React application.

### Problem
No error boundaries implemented in:
- `/home/john/semantik/apps/webui-react/src/components/chunking/`
- Related container components

### Requirements
1. Create reusable error boundary component
2. Wrap chunking components with error boundaries
3. Provide fallback UI for error states
4. Log errors to monitoring service
5. Allow users to recover from errors

### Implementation Approach
1. **Create ErrorBoundary component**:
   ```javascript
   class ErrorBoundary extends Component {
       state = { hasError: false, error: null };
       
       static getDerivedStateFromError(error) {
           return { hasError: true, error };
       }
       
       componentDidCatch(error, errorInfo) {
           // Log to monitoring service
           console.error('Component error:', error, errorInfo);
       }
       
       render() {
           if (this.state.hasError) {
               return <FallbackComponent onReset={() => this.setState({ hasError: false })} />;
           }
           return this.props.children;
       }
   }
   ```

2. **Wrap high-risk components**:
   - ChunkingPreviewPanel
   - ChunkingComparisonView
   - ChunkingAnalyticsDashboard

3. **Create specific fallback UIs** for different error types

### Files to Create/Modify
- Create `/home/john/semantik/apps/webui-react/src/components/common/ErrorBoundary.tsx`
- Wrap components in `/home/john/semantik/apps/webui-react/src/components/chunking/`
- Update parent components to include error boundaries

### Success Criteria
- Component errors don't crash the entire app
- Users see helpful error messages
- Ability to recover from errors without page reload
- Errors logged for debugging

### Completion Steps
1. Run linting and fix any issues:
   ```bash
   cd apps/webui-react
   npm run lint:fix
   npm run type-check
   ```
2. Run tests to ensure nothing broke:
   ```bash
   npm test
   ```
3. Commit and push changes:
   ```bash
   git add -A
   git commit -m "feat(frontend): add error boundaries to chunking components
   
   - Created reusable ErrorBoundary component
   - Wrapped high-risk chunking components
   - Added fallback UI with recovery options
   - Implemented error logging for debugging
   
   Fixes TICKET-007"
   git push origin feat/ticket-007-error-boundaries
   ```
4. Create PR against feature/improve-chunking:
   ```bash
   gh pr create --base feature/improve-chunking --title "Add error boundaries to React components" --body "Resolves TICKET-007: Prevents component errors from crashing entire application"
   ```

---

## TICKET-008: Add Timeout Protection to I/O Operations

### Priority: LOW
### Estimated Hours: 4-6

### Setup Instructions
1. Ensure you're on the latest feature branch:
   ```bash
   git checkout feature/improve-chunking
   git pull origin feature/improve-chunking
   ```
2. Create a new branch for this ticket:
   ```bash
   git checkout -b feat/ticket-008-timeout-protection
   ```

### Context
The backend code lacks timeout protection on I/O operations, which could lead to hung requests and resource exhaustion.

### Problem
- Database queries without timeout
- External API calls without timeout
- File operations without timeout
- Redis operations without timeout

### Requirements
1. Add timeout configuration to all I/O operations
2. Use reasonable defaults (e.g., 30s for DB, 10s for Redis)
3. Make timeouts configurable via environment variables
4. Handle timeout errors gracefully

### Implementation Approach
1. **Database operations**:
   ```python
   # Add statement timeout
   async with db.begin():
       await db.execute(text("SET statement_timeout = '30s'"))
       # Run queries
   ```

2. **HTTP requests**:
   ```python
   async with httpx.AsyncClient(timeout=10.0) as client:
       response = await client.get(url)
   ```

3. **Redis operations**:
   ```python
   redis_client = Redis(socket_connect_timeout=5, socket_timeout=5)
   ```

4. **File operations with asyncio timeout**:
   ```python
   async with asyncio.timeout(30):
       async with aiofiles.open(path) as f:
           content = await f.read()
   ```

### Files to Review
- `/home/john/semantik/packages/webui/services/chunking_service.py`
- `/home/john/semantik/packages/webui/repositories/`
- `/home/john/semantik/packages/vecpipe/` (if affected)

### Configuration to Add
```python
# In settings
DB_QUERY_TIMEOUT = int(os.getenv("DB_QUERY_TIMEOUT", "30"))
REDIS_TIMEOUT = int(os.getenv("REDIS_TIMEOUT", "10"))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
```

### Success Criteria
- No operations can hang indefinitely
- Timeout errors handled gracefully
- Performance monitoring shows no hung requests
- Configurable timeouts for different environments

### Completion Steps
1. Run linting and fix any issues:
   ```bash
   make lint
   # or
   poetry run ruff check --fix
   poetry run mypy packages/
   ```
2. Run tests to ensure nothing broke:
   ```bash
   make test
   ```
3. Commit and push changes:
   ```bash
   git add -A
   git commit -m "feat(reliability): add timeout protection to I/O operations
   
   - Added configurable timeouts for database queries
   - Protected HTTP requests with timeout settings
   - Added Redis operation timeouts
   - Implemented asyncio timeout for file operations
   
   Fixes TICKET-008"
   git push origin feat/ticket-008-timeout-protection
   ```
4. Create PR against feature/improve-chunking:
   ```bash
   gh pr create --base feature/improve-chunking --title "Add timeout protection to I/O operations" --body "Resolves TICKET-008: Prevents hung requests and resource exhaustion from indefinite I/O operations"
   ```

---

## TICKET-009: Consolidate and Clean Migration Files

### Priority: HIGH
### Estimated Hours: 3-5

### Setup Instructions
1. Ensure you're on the latest feature branch:
   ```bash
   git checkout feature/improve-chunking
   git pull origin feature/improve-chunking
   ```
2. Create a new branch for this ticket:
   ```bash
   git checkout -b refactor/ticket-009-consolidate-migrations
   ```

### Context
The feature branch has accumulated multiple migration files that conflict and overlap. Some migrations drop and recreate the same tables, causing confusion and risk.

### Problem
Multiple migration files touching the same tables:
- `52db15bd2686_add_chunking_tables_with_partitioning.py`
- `ae558c9e183f_implement_100_direct_list_partitions.py`
- `8547ff31e80c_safe_100_partitions_with_data_preservation.py`
- Several others

### Requirements
1. Consolidate related migrations into a single, clean migration
2. Ensure migrations can run from a fresh database
3. Ensure migrations can run from current production state
4. Remove duplicate or conflicting operations
5. Test both upgrade and downgrade paths

### Implementation Approach
1. **Analyze current migration chain**:
   - List all migrations in order
   - Identify what each migration does
   - Find overlaps and conflicts

2. **Create single consolidated migration**:
   - Combine all chunking-related schema changes
   - Remove intermediate states
   - Ensure idempotent operations where possible

3. **Handle existing deployments**:
   - Check if tables already exist
   - Migrate data if needed
   - Don't fail if migration partially applied

### Pseudocode Structure:
```python
def upgrade():
    # Check current state
    inspector = inspect(op.get_bind())
    
    # Create tables if not exist
    if not table_exists('chunking_strategies'):
        create_chunking_strategies_table()
    
    if not table_exists('chunks'):
        create_chunks_table_with_partitions()
    elif not has_partitions('chunks'):
        migrate_to_partitioned_table()
    
    # Add columns if missing
    if not column_exists('collections', 'default_chunking_config_id'):
        add_collection_columns()
```

### Files to Consolidate
- All migrations in `/home/john/semantik/alembic/versions/` related to chunking
- Create new single migration file

### Success Criteria
- Single migration file for entire chunking feature
- Can run on fresh database
- Can run on database with partial migrations
- Both upgrade and downgrade work
- No data loss during migration

### Completion Steps
1. Run linting and fix any issues:
   ```bash
   make lint
   # or
   poetry run ruff check --fix
   poetry run mypy packages/
   ```
2. Run tests to ensure nothing broke:
   ```bash
   make test
   # Also test migrations specifically
   alembic upgrade head
   alembic downgrade -1
   alembic upgrade head
   ```
3. Commit and push changes:
   ```bash
   git add -A
   git commit -m "refactor(migrations): consolidate chunking migrations into single file
   
   - Merged multiple conflicting migrations
   - Added idempotent operations for partial state
   - Ensured both upgrade and downgrade paths work
   - Tested on fresh and existing databases
   
   Fixes TICKET-009"
   git push origin refactor/ticket-009-consolidate-migrations
   ```
4. Create PR against feature/improve-chunking:
   ```bash
   gh pr create --base feature/improve-chunking --title "Consolidate chunking migrations" --body "Resolves TICKET-009: Consolidates multiple conflicting migration files into a single clean migration"
   ```

---

## TICKET-010: Add Integration Tests for Complete Feature

### Priority: MEDIUM
### Estimated Hours: 8-12

### Setup Instructions
1. Ensure you're on the latest feature branch:
   ```bash
   git checkout feature/improve-chunking
   git pull origin feature/improve-chunking
   ```
2. Create a new branch for this ticket:
   ```bash
   git checkout -b test/ticket-010-integration-tests
   ```

### Context
While individual PRs may have had tests, the integrated feature needs comprehensive testing to ensure all components work together correctly.

### Problem
- No end-to-end tests for complete chunking workflow
- Missing tests for partition behavior
- No load tests for WebSocket connections
- Missing integration tests between services

### Requirements
1. Create end-to-end test for chunking workflow
2. Test partition distribution and performance
3. Load test WebSocket connections
4. Test error recovery and edge cases
5. Verify memory cleanup

### Test Scenarios to Cover
1. **Happy Path**:
   - Create collection with chunking config
   - Upload document
   - Preview chunks via WebSocket
   - Verify chunks stored correctly across partitions
   - Search across chunks

2. **Partition Distribution**:
   - Create large collection (10k+ chunks)
   - Verify even distribution across partitions
   - Query performance test

3. **WebSocket Stress**:
   - Connect 100+ clients simultaneously
   - Verify no race conditions
   - Test reconnection logic
   - Verify cleanup on disconnect

4. **Error Recovery**:
   - Test chunking failure handling
   - Test WebSocket disconnection/reconnection
   - Test migration rollback
   - Test service degradation

### Files to Create/Update
- `/home/john/semantik/tests/integration/test_chunking_workflow.py`
- `/home/john/semantik/tests/load/test_websocket_load.py`
- `/home/john/semantik/tests/integration/test_partition_distribution.py`

### Tools to Use
- pytest for test framework
- pytest-asyncio for async tests
- locust or similar for load testing
- faker for test data generation

### Success Criteria
- All test scenarios pass
- Code coverage > 80% for new code
- Load tests pass with 100+ concurrent users
- No memory leaks detected
- Performance benchmarks documented

### Completion Steps
1. Run linting and fix any issues:
   ```bash
   make lint
   # or
   poetry run ruff check --fix
   poetry run mypy packages/
   ```
2. Run ALL tests to ensure comprehensive coverage:
   ```bash
   make test
   # Run specific test suites
   pytest tests/integration/test_chunking_workflow.py -v
   pytest tests/load/test_websocket_load.py -v
   pytest tests/integration/test_partition_distribution.py -v
   ```
3. Commit and push changes:
   ```bash
   git add -A
   git commit -m "test(integration): add comprehensive tests for chunking feature
   
   - Added end-to-end workflow tests
   - Created partition distribution tests
   - Implemented WebSocket load tests
   - Added memory leak detection tests
   - Achieved >80% code coverage
   
   Fixes TICKET-010"
   git push origin test/ticket-010-integration-tests
   ```
4. Create PR against feature/improve-chunking:
   ```bash
   gh pr create --base feature/improve-chunking --title "Add integration tests for chunking feature" --body "Resolves TICKET-010: Comprehensive integration and load tests for complete chunking workflow"
   ```

---

## Execution Order

### Phase 1: Critical Fixes (Do First)
1. TICKET-001: Fix SQL Injection (2-3 hours)
2. TICKET-002: Fix Partition Strategy (8-12 hours)
3. TICKET-003: Add Data Protection (4-6 hours)

### Phase 2: Stability Fixes (Do Second)
4. TICKET-009: Consolidate Migrations (3-5 hours)
5. TICKET-004: Fix WebSocket Race Conditions (6-8 hours)
6. TICKET-006: Refactor to Service Layer (10-14 hours)

### Phase 3: Quality Improvements (Do Third)
7. TICKET-005: Fix Memory Leaks (4-6 hours)
8. TICKET-007: Add Error Boundaries (4-6 hours)
9. TICKET-008: Add Timeout Protection (4-6 hours)
10. TICKET-010: Add Integration Tests (8-12 hours)

**Total Estimated Time: 53-78 hours**

## Notes for LLM Agents

- Each ticket is self-contained with all necessary context
- Check the CLAUDE.md files in each package for architectural guidelines
- Run tests after each change to ensure nothing breaks
- Use the existing patterns in the codebase
- When in doubt, check how similar functionality is implemented elsewhere in the codebase
- Commit changes with clear, descriptive messages
- Update documentation if behavior changes

---

# Code Review Tickets

## Context
Each review ticket below corresponds to an implementation ticket above. These tickets should be executed by a different LLM agent to review and verify the implementation work. The reviewer should act as a senior engineer ensuring the implementation meets all requirements and follows best practices.

---

## REVIEW-001: Review SQL Injection Fixes

### Corresponds to: TICKET-001
### Priority: CRITICAL
### Estimated Hours: 1-2

### Setup Instructions
1. Ensure you're on the latest implementation branch:
   ```bash
   git fetch origin
   git checkout fix/ticket-001-sql-injection
   git pull origin fix/ticket-001-sql-injection
   ```

### Context
Another agent has implemented fixes for SQL injection vulnerabilities in the partition creation code. Your task is to verify the fixes are comprehensive and secure.

### What Was Supposed to Be Fixed
- Direct string interpolation in SQL statements (f-strings in SQL)
- Specifically in migration files for partition creation
- Pattern: `f"""CREATE TABLE chunks_p{i}..."""`

### Review Checklist

#### Security Review
1. **Search for remaining vulnerable patterns**:
   - Run: `grep -r "op.execute(f" alembic/`
   - Run: `grep -r "conn.execute(text(f" alembic/`
   - Check for any f-strings or .format() in SQL context
   
2. **Verify input validation**:
   - Confirm all dynamic values are validated as correct type
   - Check bounds validation (e.g., partition_count between 1-1000)
   - Ensure no path where user input reaches SQL construction

3. **Test the fix**:
   - Try injecting SQL via environment variables
   - Verify error handling for invalid inputs
   - Check that migrations still work with valid inputs

#### Code Quality Review
1. Are the fixes consistent across all migration files?
2. Is validation logic reusable/DRY?
3. Are error messages clear and helpful?
4. Do the changes follow existing patterns in the codebase?

### Specific Areas to Examine
- `/home/john/semantik/alembic/versions/52db15bd2686_add_chunking_tables_with_partitioning.py`
- `/home/john/semantik/alembic/versions/ae558c9e183f_implement_100_direct_list_partitions.py`
- Any new validation utilities created

### Red Flags to Watch For
- Validation only on happy path
- Missing validation in some files but not others
- Over-complicated fixes that introduce new bugs
- Breaking existing functionality

### Acceptance Criteria
- [ ] No SQL injection vulnerabilities remain
- [ ] All dynamic SQL values are validated
- [ ] Migrations still function correctly
- [ ] Code is clean and maintainable
- [ ] Tests pass

### If Issues Found
1. Document specific line numbers and problems
2. Provide concrete examples of how to exploit remaining vulnerabilities
3. Suggest specific fixes
4. Mark the review as "Changes Requested"

### Review Completion Steps
1. Provide your assessment as a PR review:
   ```bash
   # If everything looks good:
   gh pr review --approve --body "LGTM! SQL injection vulnerabilities properly fixed. [Details of what was verified]"
   
   # If changes are needed:
   gh pr review --request-changes --body "Found issues that need addressing: [List specific issues]"
   
   # If mostly good with minor suggestions:
   gh pr review --comment --body "Code looks good overall. Some suggestions: [List suggestions]"
   ```
2. Document detailed findings in a comment:
   ```bash
   gh pr comment --body "## Review Assessment for TICKET-001\n\n### ✅ What Works\n- [List verified fixes]\n\n### ❌ Issues Found\n- [List any problems]\n\n### 💡 Suggestions\n- [Optional improvements]\n\n**Verdict: [Approved/Changes Requested/Approved with Suggestions]**"
   ```

---

## REVIEW-002: Review Partition Strategy Fix

### Corresponds to: TICKET-002
### Priority: CRITICAL
### Estimated Hours: 2-3

### Setup Instructions
1. Ensure you're on the latest implementation branch:
   ```bash
   git fetch origin
   git checkout fix/ticket-002-partition-strategy
   git pull origin fix/ticket-002-partition-strategy
   ```

### Context
Another agent has fixed the broken partition strategy that was putting all chunks from a collection into a single partition. Review the new partitioning approach for correctness and performance.

### What Was Supposed to Be Fixed
- Original problem: `abs(hashtext(collection_id)) % 100` puts entire collections in one partition
- Goal: Distribute chunks WITHIN collections across multiple partitions
- Must maintain query performance for collection-wide and document-specific queries

### Review Checklist

#### Correctness Review
1. **Verify distribution algorithm**:
   ```sql
   -- Test query to check distribution
   SELECT 
       collection_id,
       COUNT(DISTINCT partition_key) as partitions_used,
       COUNT(*) as total_chunks
   FROM chunks
   GROUP BY collection_id
   HAVING COUNT(*) > 100;
   ```
   - Confirm chunks from same collection span multiple partitions
   - Verify even distribution (no severe hotspots)

2. **Query performance testing**:
   - Test: Get all chunks for a collection
   - Test: Get all chunks for a document
   - Test: Get recent chunks across all collections
   - Verify partition pruning is working (check EXPLAIN plans)

3. **Migration safety**:
   - How does it handle existing data?
   - Is there a rollback plan?
   - Are backups created before migration?

#### Implementation Review
1. **Check the new partition key formula**:
   - Is it deterministic?
   - Does it avoid collisions?
   - Is it efficient to compute?

2. **Verify trigger/function updates**:
   - Is the partition key computed correctly?
   - Are there any race conditions?
   - Is the function performant?

3. **Review monitoring capabilities**:
   - Can we track partition health?
   - Are hotspots detectable?
   - Is rebalancing possible?

### Specific Queries to Run
```sql
-- Check partition distribution
SELECT 
    'Partition ' || partition_key as partition,
    COUNT(*) as chunk_count,
    COUNT(DISTINCT collection_id) as collections,
    COUNT(DISTINCT document_id) as documents
FROM chunks
GROUP BY partition_key
ORDER BY chunk_count DESC;

-- Check for partition skew
WITH partition_stats AS (
    SELECT 
        partition_key,
        COUNT(*) as chunk_count
    FROM chunks
    GROUP BY partition_key
)
SELECT 
    STDDEV(chunk_count) / AVG(chunk_count) as coefficient_of_variation
FROM partition_stats;
```

### Red Flags to Watch For
- All chunks from a collection still in one partition
- Severe partition imbalance (some empty, some overloaded)
- Degraded query performance
- Complex logic that will be hard to maintain
- Missing indexes on new partition key

### Acceptance Criteria
- [ ] Chunks from large collections distributed across partitions
- [ ] Even distribution (coefficient of variation < 0.5)
- [ ] Query performance maintained or improved
- [ ] Migration is safe and reversible
- [ ] Monitoring views updated

### If Issues Found
1. Provide specific examples of bad distribution
2. Include EXPLAIN ANALYZE output showing performance issues
3. Suggest alternative partitioning strategies
4. Document any data loss risks

### Review Completion Steps
1. Provide your assessment as a PR review:
   ```bash
   # If everything looks good:
   gh pr review --approve --body "LGTM! Partition strategy correctly distributes chunks. [Details of distribution testing]"
   
   # If changes are needed:
   gh pr review --request-changes --body "Partition distribution issues found: [List specific problems]"
   
   # If mostly good with minor suggestions:
   gh pr review --comment --body "Strategy works but could be optimized. Suggestions: [List improvements]"
   ```
2. Document detailed findings in a comment:
   ```bash
   gh pr comment --body "## Review Assessment for TICKET-002\n\n### ✅ What Works\n- [Distribution results]\n\n### ❌ Issues Found\n- [Any problems]\n\n### 📊 Performance Metrics\n- [Query performance data]\n\n**Verdict: [Approved/Changes Requested/Approved with Suggestions]**"
   ```

---

## REVIEW-003: Review Data Protection in Migrations

### Corresponds to: TICKET-003
### Priority: HIGH
### Estimated Hours: 1-2

### Setup Instructions
1. Ensure you're on the latest implementation branch:
   ```bash
   git fetch origin
   git checkout feat/ticket-003-migration-safety
   git pull origin feat/ticket-003-migration-safety
   ```

### Context
Another agent has added data protection mechanisms to prevent data loss during destructive migrations. Review the implementation for safety and completeness.

### What Was Supposed to Be Fixed
- Destructive operations like `DROP TABLE IF EXISTS chunks CASCADE` without backups
- No rollback capability for failed migrations
- No safety checks before destructive operations

### Review Checklist

#### Safety Review
1. **Verify backup creation**:
   - Check that backups are created before ANY destructive operation
   - Verify backup includes all data (not just schema)
   - Confirm backup table naming includes timestamp
   - Test restore procedure works

2. **Check safety flags**:
   ```bash
   # Test without flag - should fail
   unset ALLOW_DESTRUCTIVE_MIGRATIONS
   alembic upgrade head
   
   # Test with flag - should work
   export ALLOW_DESTRUCTIVE_MIGRATIONS=true
   alembic upgrade head
   ```

3. **Review rollback capability**:
   - Can we downgrade after upgrade?
   - Is data preserved during downgrade?
   - Are backup tables retained?

#### Implementation Review
1. **Backup manager review** (if created):
   - Is it reusable across migrations?
   - Does it handle errors gracefully?
   - Are backups logged properly?
   - Is there a retention policy?

2. **Check all destructive operations**:
   ```bash
   # Find all potential destructive operations
   grep -r "DROP\|TRUNCATE\|DELETE" alembic/versions/
   ```
   - Verify each has protection
   - Check for missed cases

3. **Test edge cases**:
   - Empty tables
   - Very large tables (performance)
   - Tables with foreign key constraints
   - Partial migration failures

### Specific Files to Review
- `/home/john/semantik/alembic/versions/ae558c9e183f_implement_100_direct_list_partitions.py`
- `/home/john/semantik/alembic/migrations_utils/backup_manager.py` (if created)
- Any modified migration files

### Red Flags to Watch For
- Backups created but not verified
- Partial protection (some operations protected, others not)
- Performance issues with large table backups
- Backup tables accumulating without cleanup
- Missing logging of critical operations

### Acceptance Criteria
- [ ] All destructive operations have backup mechanisms
- [ ] Safety flag required for destructive migrations
- [ ] Backup and restore tested successfully
- [ ] Clear logging of all backup operations
- [ ] Documentation updated

### If Issues Found
1. List specific unprotected operations
2. Demonstrate potential data loss scenario
3. Suggest improvements to backup strategy
4. Check if production environment properly protected

### Review Completion Steps
1. Provide your assessment as a PR review:
   ```bash
   # If everything looks good:
   gh pr review --approve --body "LGTM! Data protection mechanisms properly implemented. [Details of backup testing]"
   
   # If changes are needed:
   gh pr review --request-changes --body "Data protection gaps found: [List specific gaps]"
   
   # If mostly good with minor suggestions:
   gh pr review --comment --body "Backup strategy works with suggestions: [List improvements]"
   ```
2. Document detailed findings in a comment:
   ```bash
   gh pr comment --body "## Review Assessment for TICKET-003\n\n### ✅ What Works\n- [Backup mechanisms verified]\n\n### ❌ Issues Found\n- [Any gaps]\n\n### 🔒 Safety Verification\n- [Test results]\n\n**Verdict: [Approved/Changes Requested/Approved with Suggestions]**"
   ```

---

## REVIEW-004: Review WebSocket Race Condition Fixes

### Corresponds to: TICKET-004
### Priority: HIGH
### Estimated Hours: 2-3

### Setup Instructions
1. Ensure you're on the latest implementation branch:
   ```bash
   git fetch origin
   git checkout fix/ticket-004-websocket-race
   git pull origin fix/ticket-004-websocket-race
   ```

### Context
Another agent has fixed race conditions in the WebSocket manager by adding proper locking for shared state modifications. Review the implementation for correctness and performance.

### What Was Supposed to Be Fixed
- Shared dictionaries modified without synchronization
- Potential connection state corruption
- Dropped messages due to race conditions

### Review Checklist

#### Concurrency Review
1. **Verify lock usage**:
   ```python
   # Check all dictionary modifications have locks
   # Look for patterns like:
   # self.connections[key] = value  # Should be inside lock
   # del self.connections[key]      # Should be inside lock
   ```

2. **Test concurrent access**:
   - Write a test with 100+ concurrent connections
   - Verify no KeyError or corruption
   - Check for deadlocks
   - Monitor for dropped messages

3. **Review lock granularity**:
   - Are locks too coarse (performance bottleneck)?
   - Are locks too fine (missed race conditions)?
   - Check for lock ordering issues (deadlock potential)

#### Performance Review
1. **Benchmark before/after**:
   ```python
   # Simple load test
   async def load_test():
       tasks = []
       for i in range(100):
           tasks.append(connect_and_send_messages())
       await asyncio.gather(*tasks)
   ```
   - Connection time
   - Message latency
   - Throughput

2. **Check for bottlenecks**:
   - Long-held locks
   - Unnecessary locking
   - Better alternatives (e.g., asyncio.Queue)

#### Code Quality Review
1. Are locks properly released (using context managers)?
2. Is the locking strategy documented?
3. Are there tests for concurrent scenarios?
4. Is error handling correct inside locked sections?

### Specific Areas to Test
- Adding/removing connections simultaneously
- Broadcasting while connections are changing
- Connection limit enforcement under load
- Error handling within locked sections

### Stress Test Script
```python
# Run this to verify fixes
import asyncio
import websockets
import json

async def stress_client(client_id):
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as ws:
        # Authenticate
        await ws.send(json.dumps({"token": "test_token"}))
        
        # Rapid operations
        for i in range(100):
            await ws.send(json.dumps({"action": "test", "data": i}))
            if i % 10 == 0:
                response = await ws.recv()
                
async def stress_test():
    clients = [stress_client(i) for i in range(100)]
    await asyncio.gather(*clients)
```

### Red Flags to Watch For
- Locks not using `async with`
- Missing locks on some operations
- Potential deadlocks from lock ordering
- Performance degradation > 10%
- No tests for concurrent access

### Acceptance Criteria
- [ ] All shared state properly synchronized
- [ ] No race conditions under load
- [ ] Performance impact < 5%
- [ ] No deadlocks possible
- [ ] Concurrent access tests pass

### If Issues Found
1. Provide specific race condition reproduction steps
2. Include thread/task timing diagrams if helpful
3. Suggest alternative synchronization approaches
4. Identify any remaining vulnerable code paths

### Review Completion Steps
1. Provide your assessment as a PR review:
   ```bash
   # If everything looks good:
   gh pr review --approve --body "LGTM! Race conditions properly fixed with appropriate locking. [Details of stress testing]"
   
   # If changes are needed:
   gh pr review --request-changes --body "Concurrency issues remain: [List specific race conditions]"
   
   # If mostly good with minor suggestions:
   gh pr review --comment --body "Locking works but could be optimized: [List improvements]"
   ```
2. Document detailed findings in a comment:
   ```bash
   gh pr comment --body "## Review Assessment for TICKET-004\n\n### ✅ What Works\n- [Locking verification]\n\n### ❌ Issues Found\n- [Any race conditions]\n\n### ⚡ Performance Impact\n- [Benchmark results]\n\n**Verdict: [Approved/Changes Requested/Approved with Suggestions]**"
   ```

---

## REVIEW-005: Review React Memory Leak Fixes

### Corresponds to: TICKET-005
### Priority: MEDIUM
### Estimated Hours: 1-2

### Setup Instructions
1. Ensure you're on the latest implementation branch:
   ```bash
   git fetch origin
   git checkout fix/ticket-005-memory-leaks
   git pull origin fix/ticket-005-memory-leaks
   ```

### Context
Another agent has fixed memory leaks in React components by adding proper cleanup for WebSocket connections and event listeners. Review the implementation for completeness.

### What Was Supposed to Be Fixed
- WebSocket connections not closed on unmount
- Event listeners not removed
- Missing cleanup in useEffect hooks

### Review Checklist

#### Memory Leak Detection
1. **Test with Chrome DevTools**:
   ```javascript
   // Test procedure:
   // 1. Open Memory Profiler
   // 2. Take heap snapshot
   // 3. Mount/unmount component 10 times
   // 4. Take another snapshot
   // 5. Compare for leaked objects
   ```

2. **Check all useEffect hooks**:
   ```javascript
   // Every useEffect with setup should have cleanup
   useEffect(() => {
       // setup
       return () => {
           // cleanup - THIS MUST EXIST
       };
   }, [deps]);
   ```

3. **Verify WebSocket cleanup**:
   - Connection closed on unmount?
   - Event handlers removed?
   - Pending messages handled?

#### Implementation Review
1. **Review cleanup functions**:
   - Are all resources cleaned up?
   - Is cleanup order correct?
   - Are there race conditions in cleanup?

2. **Check for missed cases**:
   ```bash
   # Find useEffects without cleanup
   grep -A 5 "useEffect(" src/components/chunking/*.tsx | grep -B 5 "}, \[" | grep -v "return"
   ```

3. **Test edge cases**:
   - Rapid mount/unmount
   - Unmount during active operations
   - Network disconnection during unmount

### Specific Components to Review
- `/home/john/semantik/apps/webui-react/src/components/chunking/ChunkingPreviewPanel.tsx`
- `/home/john/semantik/apps/webui-react/src/hooks/useChunkingWebSocket.ts`
- Any other components using WebSocket or event listeners

### Memory Leak Test
```javascript
// Add this test to verify fixes
describe('Memory Leak Tests', () => {
    it('should not leak memory on repeated mount/unmount', async () => {
        const mountUnmount = async () => {
            const { unmount } = render(<ChunkingPreviewPanel />);
            await waitFor(() => expect(screen.getByTestId('preview')).toBeInTheDocument());
            unmount();
        };
        
        // Get initial memory
        const initial = performance.memory.usedJSHeapSize;
        
        // Mount/unmount 50 times
        for (let i = 0; i < 50; i++) {
            await mountUnmount();
        }
        
        // Force GC if available
        if (global.gc) global.gc();
        
        // Check memory growth
        const final = performance.memory.usedJSHeapSize;
        const growth = final - initial;
        expect(growth).toBeLessThan(5 * 1024 * 1024); // Less than 5MB growth
    });
});
```

### Red Flags to Watch For
- Cleanup functions that throw errors
- Async operations in cleanup (should be synchronous)
- Missing cleanup for some resources
- Cleanup that depends on state (might be stale)
- No tests for memory leaks

### Acceptance Criteria
- [ ] All useEffect hooks have cleanup functions
- [ ] WebSocket connections properly closed
- [ ] Event listeners removed on unmount
- [ ] Memory profiler shows no leaks
- [ ] Component tests include cleanup verification

### If Issues Found
1. Show specific memory profiler evidence
2. Point to exact lines missing cleanup
3. Provide corrected useEffect examples
4. Suggest memory leak tests to add

### Review Completion Steps
1. Provide your assessment as a PR review:
   ```bash
   # If everything looks good:
   gh pr review --approve --body "LGTM! Memory leaks properly fixed with cleanup functions. [Details of memory profiling]"
   
   # If changes are needed:
   gh pr review --request-changes --body "Memory leaks still present: [List specific leaks]"
   
   # If mostly good with minor suggestions:
   gh pr review --comment --body "Most leaks fixed, minor issues remain: [List remaining issues]"
   ```
2. Document detailed findings in a comment:
   ```bash
   gh pr comment --body "## Review Assessment for TICKET-005\n\n### ✅ What Works\n- [Cleanup verification]\n\n### ❌ Issues Found\n- [Any remaining leaks]\n\n### 📈 Memory Profile\n- [Before/after metrics]\n\n**Verdict: [Approved/Changes Requested/Approved with Suggestions]**"
   ```

---

## REVIEW-006: Review Service Layer Refactoring

### Corresponds to: TICKET-006
### Priority: MEDIUM
### Estimated Hours: 2-3

### Setup Instructions
1. Ensure you're on the latest implementation branch:
   ```bash
   git fetch origin
   git checkout refactor/ticket-006-service-layer
   git pull origin refactor/ticket-006-service-layer
   ```

### Context
Another agent has refactored business logic from router files to service layer, following the three-layer architecture pattern. Review for proper separation of concerns.

### What Was Supposed to Be Fixed
- Business logic in router files
- Direct database queries in routers
- Complex validation in endpoints
- Transaction management in routers

### Review Checklist

#### Architecture Review
1. **Verify router responsibilities**:
   ```python
   # Routers should ONLY:
   # - Parse request data
   # - Call service methods
   # - Format responses
   # - Handle HTTP errors
   
   # Check for violations:
   # - db.query() in routers
   # - Complex if/else business logic
   # - Direct model instantiation
   ```

2. **Verify service layer completeness**:
   - All business logic moved to services?
   - Services handle transactions?
   - Services throw domain exceptions?
   - Routers catch and convert to HTTP errors?

3. **Check dependency injection**:
   ```python
   # Good pattern:
   @router.post("/")
   async def create(
       data: CreateModel,
       service: Service = Depends(get_service)
   ):
       return await service.create(data)
   ```

#### Code Quality Review
1. **Service method design**:
   - Single responsibility per method?
   - Clear method names?
   - Proper async/await usage?
   - Transaction boundaries correct?

2. **Error handling pattern**:
   ```python
   # Service throws domain exceptions
   class AlreadyExistsError(Exception): pass
   
   # Router converts to HTTP
   except AlreadyExistsError as e:
       raise HTTPException(409, str(e))
   ```

3. **Test coverage**:
   - Services unit tested?
   - Routers integration tested?
   - Mocked correctly?

### Specific Files to Review
- `/home/john/semantik/packages/webui/api/v2/chunking.py`
- `/home/john/semantik/packages/webui/api/v2/collections.py`
- `/home/john/semantik/packages/webui/services/chunking_service.py`
- `/home/john/semantik/packages/webui/services/collection_service.py`

### Anti-Pattern Check
```python
# BAD - Business logic in router
@router.post("/")
async def create(db: Session = Depends(get_db)):
    if db.query(Model).filter_by(name=name).first():  # WRONG
        raise HTTPException(409)
    model = Model(...)  # WRONG
    db.add(model)  # WRONG
    db.commit()  # WRONG

# GOOD - Delegation to service
@router.post("/")
async def create(service: Service = Depends(get_service)):
    return await service.create(...)  # RIGHT
```

### Red Flags to Watch For
- Partial refactoring (some logic moved, some remains)
- Services calling routers (wrong direction)
- Circular dependencies
- Lost functionality during refactoring
- Transaction issues after refactoring

### Acceptance Criteria
- [ ] No business logic in routers
- [ ] All DB queries in service/repository layer
- [ ] Clear separation of concerns
- [ ] Tests still pass
- [ ] No performance degradation

### If Issues Found
1. List specific business logic still in routers
2. Show examples of architecture violations
3. Suggest proper service method structure
4. Identify any broken functionality

### Review Completion Steps
1. Provide your assessment as a PR review:
   ```bash
   # If everything looks good:
   gh pr review --approve --body "LGTM! Clean separation of concerns achieved. [Details of architecture verification]"
   
   # If changes are needed:
   gh pr review --request-changes --body "Architecture violations found: [List specific violations]"
   
   # If mostly good with minor suggestions:
   gh pr review --comment --body "Good refactoring with minor issues: [List suggestions]"
   ```
2. Document detailed findings in a comment:
   ```bash
   gh pr comment --body "## Review Assessment for TICKET-006\n\n### ✅ What Works\n- [Architecture improvements]\n\n### ❌ Issues Found\n- [Any violations]\n\n### 🏗️ Architecture Analysis\n- [Layer separation verification]\n\n**Verdict: [Approved/Changes Requested/Approved with Suggestions]**"
   ```

---

## REVIEW-007: Review Error Boundary Implementation

### Corresponds to: TICKET-007
### Priority: MEDIUM
### Estimated Hours: 1-2

### Setup Instructions
1. Ensure you're on the latest implementation branch:
   ```bash
   git fetch origin
   git checkout feat/ticket-007-error-boundaries
   git pull origin feat/ticket-007-error-boundaries
   ```

### Context
Another agent has added error boundaries to React components to prevent crashes from propagating to the entire application. Review the implementation for coverage and usability.

### What Was Supposed to Be Fixed
- No error boundaries in chunking components
- JavaScript errors crash entire app
- No recovery mechanism for users

### Review Checklist

#### Coverage Review
1. **Verify error boundary placement**:
   ```typescript
   // Check high-risk components are wrapped:
   // - ChunkingPreviewPanel
   // - ChunkingComparisonView
   // - ChunkingAnalyticsDashboard
   // - Any component doing async operations
   ```

2. **Test error handling**:
   ```javascript
   // Inject errors to test boundaries
   const ErrorTest = () => {
       throw new Error("Test error");
   };
   
   // Should show fallback UI, not crash app
   ```

3. **Check boundary granularity**:
   - Too high level (entire app in one boundary)?
   - Too low level (every component wrapped)?
   - Strategic placement for best UX?

#### Implementation Review
1. **Error boundary features**:
   - Clear error message to user?
   - Reset capability?
   - Error logging/reporting?
   - Different fallbacks for different errors?

2. **Fallback UI quality**:
   - Helpful error messages?
   - Recovery instructions?
   - Maintains app layout?
   - Accessible?

3. **Error recovery testing**:
   ```javascript
   // Can users recover without refresh?
   fireEvent.click(screen.getByText('Try Again'));
   expect(screen.getByTestId('component')).toBeInTheDocument();
   ```

### Test Cases to Run
```javascript
// Test error boundary catches errors
it('should catch and display errors gracefully', () => {
    const ThrowError = () => {
        throw new Error('Test error');
    };
    
    render(
        <ErrorBoundary>
            <ThrowError />
        </ErrorBoundary>
    );
    
    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
    expect(screen.getByText(/try again/i)).toBeInTheDocument();
});

// Test recovery
it('should allow recovery from errors', () => {
    let shouldError = true;
    const ConditionalError = () => {
        if (shouldError) throw new Error('Test');
        return <div>Success</div>;
    };
    
    const { rerender } = render(
        <ErrorBoundary>
            <ConditionalError />
        </ErrorBoundary>
    );
    
    shouldError = false;
    fireEvent.click(screen.getByText('Try Again'));
    expect(screen.getByText('Success')).toBeInTheDocument();
});
```

### Red Flags to Watch For
- Error boundaries in functional components (must be class components)
- Missing error logging
- Poor error messages ("Something went wrong")
- No way to recover
- Errors during error handling (infinite loop)

### Acceptance Criteria
- [ ] Critical components have error boundaries
- [ ] Errors don't crash the entire app
- [ ] Users can recover from errors
- [ ] Errors are logged for debugging
- [ ] Fallback UI is helpful and accessible

### If Issues Found
1. List components still vulnerable to crashes
2. Show poor UX in error states
3. Suggest better error messages
4. Identify missing error logging

### Review Completion Steps
1. Provide your assessment as a PR review:
   ```bash
   # If everything looks good:
   gh pr review --approve --body "LGTM! Error boundaries properly protect components. [Details of error testing]"
   
   # If changes are needed:
   gh pr review --request-changes --body "Error boundary coverage insufficient: [List uncovered components]"
   
   # If mostly good with minor suggestions:
   gh pr review --comment --body "Good coverage with suggestions: [List improvements]"
   ```
2. Document detailed findings in a comment:
   ```bash
   gh pr comment --body "## Review Assessment for TICKET-007\n\n### ✅ What Works\n- [Error handling verification]\n\n### ❌ Issues Found\n- [Any gaps]\n\n### 🛡️ Coverage Analysis\n- [Component protection status]\n\n**Verdict: [Approved/Changes Requested/Approved with Suggestions]**"
   ```

---

## REVIEW-008: Review Timeout Implementation

### Corresponds to: TICKET-008
### Priority: LOW
### Estimated Hours: 1-2

### Setup Instructions
1. Ensure you're on the latest implementation branch:
   ```bash
   git fetch origin
   git checkout feat/ticket-008-timeout-protection
   git pull origin feat/ticket-008-timeout-protection
   ```

### Context
Another agent has added timeout protection to I/O operations to prevent hung requests and resource exhaustion. Review the implementation for completeness and configuration.

### What Was Supposed to Be Fixed
- Database queries without timeouts
- External API calls without timeouts
- Redis operations without timeouts
- File operations without timeouts

### Review Checklist

#### Coverage Review
1. **Find all I/O operations**:
   ```bash
   # Database operations
   grep -r "db.execute\|db.query\|db.scalars" packages/
   
   # HTTP calls
   grep -r "httpx\|aiohttp\|requests" packages/
   
   # Redis operations
   grep -r "redis\." packages/
   
   # File operations
   grep -r "open(\|aiofiles" packages/
   ```

2. **Verify timeout configuration**:
   - Is each operation protected?
   - Are timeouts configurable?
   - Are defaults reasonable?

3. **Test timeout behavior**:
   ```python
   # Simulate slow operation
   async def test_timeout():
       with pytest.raises(TimeoutError):
           async with timeout(1):
               await slow_operation()
   ```

#### Implementation Review
1. **Configuration management**:
   ```python
   # Check for settings like:
   DB_QUERY_TIMEOUT = int(os.getenv("DB_QUERY_TIMEOUT", "30"))
   REDIS_TIMEOUT = int(os.getenv("REDIS_TIMEOUT", "10"))
   ```

2. **Error handling**:
   - Graceful timeout handling?
   - Proper error messages?
   - Retry logic where appropriate?

3. **Performance impact**:
   - Overhead of timeout implementation?
   - Are timeouts too aggressive?
   - Are timeouts too lenient?

### Specific Operations to Check
```python
# Database with timeout
async with db.begin():
    await db.execute(text("SET statement_timeout = '30s'"))
    result = await db.execute(query)

# HTTP with timeout
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.get(url)

# Redis with timeout
redis = Redis(socket_timeout=10)

# File operations with asyncio timeout
async with asyncio.timeout(30):
    async with aiofiles.open(path) as f:
        content = await f.read()
```

### Red Flags to Watch For
- Hardcoded timeout values
- Missing timeout on critical paths
- Timeout errors not handled
- Too short timeouts causing false failures
- No monitoring/logging of timeouts

### Acceptance Criteria
- [ ] All I/O operations have timeouts
- [ ] Timeouts are configurable
- [ ] Timeout errors handled gracefully
- [ ] Reasonable defaults set
- [ ] No operations can hang indefinitely

### If Issues Found
1. List specific operations without timeouts
2. Show configuration that's not working
3. Demonstrate hung request scenario
4. Suggest appropriate timeout values

### Review Completion Steps
1. Provide your assessment as a PR review:
   ```bash
   # If everything looks good:
   gh pr review --approve --body "LGTM! All I/O operations properly protected with timeouts. [Details of timeout testing]"
   
   # If changes are needed:
   gh pr review --request-changes --body "Timeout protection incomplete: [List unprotected operations]"
   
   # If mostly good with minor suggestions:
   gh pr review --comment --body "Good timeout coverage with suggestions: [List optimizations]"
   ```
2. Document detailed findings in a comment:
   ```bash
   gh pr comment --body "## Review Assessment for TICKET-008\n\n### ✅ What Works\n- [Timeout verification]\n\n### ❌ Issues Found\n- [Any gaps]\n\n### ⏱️ Timeout Configuration\n- [Settings review]\n\n**Verdict: [Approved/Changes Requested/Approved with Suggestions]**"
   ```

---

## REVIEW-009: Review Migration Consolidation

### Corresponds to: TICKET-009
### Priority: HIGH
### Estimated Hours: 2-3

### Setup Instructions
1. Ensure you're on the latest implementation branch:
   ```bash
   git fetch origin
   git checkout refactor/ticket-009-consolidate-migrations
   git pull origin refactor/ticket-009-consolidate-migrations
   ```

### Context
Another agent has consolidated multiple conflicting migration files into a clean, single migration for the chunking feature. Review for correctness and safety.

### What Was Supposed to Be Fixed
- Multiple overlapping migrations
- Conflicting operations
- Migrations that drop and recreate same tables
- Unclear migration order

### Review Checklist

#### Migration Safety Review
1. **Test on fresh database**:
   ```bash
   # Start fresh
   dropdb test_db && createdb test_db
   alembic upgrade head
   # Should complete without errors
   ```

2. **Test on existing database**:
   ```bash
   # With partially applied migrations
   alembic upgrade [previous-revision]
   alembic upgrade head
   # Should handle existing state
   ```

3. **Test downgrade path**:
   ```bash
   alembic downgrade -1
   # Should cleanly reverse changes
   ```

#### Implementation Review
1. **Check idempotency**:
   ```python
   # Good pattern:
   if not table_exists('chunks'):
       create_table('chunks', ...)
   
   if not column_exists('collections', 'new_column'):
       add_column('collections', 'new_column', ...)
   ```

2. **Verify no data loss**:
   - Are existing tables preserved?
   - Is data migrated correctly?
   - Are backups created?

3. **Review consolidation completeness**:
   - All chunking changes in one file?
   - No duplicate operations?
   - Clear upgrade/downgrade paths?

### Migration Order Test
```sql
-- After migration, verify schema
SELECT 
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public'
    AND table_name IN ('chunks', 'chunking_strategies', 'chunking_configs')
ORDER BY table_name, ordinal_position;

-- Check constraints
SELECT
    tc.table_name,
    tc.constraint_name,
    tc.constraint_type
FROM information_schema.table_constraints tc
WHERE tc.table_schema = 'public'
    AND tc.table_name LIKE 'chunk%';
```

### Red Flags to Watch For
- Migration fails on fresh database
- Migration fails on existing database
- Downgrade doesn't work
- Data loss during migration
- Orphaned or duplicate tables

### Acceptance Criteria
- [ ] Single clean migration file
- [ ] Works on fresh database
- [ ] Works on existing database
- [ ] Downgrade path works
- [ ] No data loss
- [ ] Clear documentation

### If Issues Found
1. Show specific migration failures
2. Identify data loss scenarios
3. List remaining conflicts
4. Suggest migration strategy improvements

### Review Completion Steps
1. Provide your assessment as a PR review:
   ```bash
   # If everything looks good:
   gh pr review --approve --body "LGTM! Migrations properly consolidated and tested. [Details of migration testing]"
   
   # If changes are needed:
   gh pr review --request-changes --body "Migration issues found: [List specific problems]"
   
   # If mostly good with minor suggestions:
   gh pr review --comment --body "Consolidation works with suggestions: [List improvements]"
   ```
2. Document detailed findings in a comment:
   ```bash
   gh pr comment --body "## Review Assessment for TICKET-009\n\n### ✅ What Works\n- [Migration verification]\n\n### ❌ Issues Found\n- [Any problems]\n\n### 🔄 Migration Testing\n- [Upgrade/downgrade results]\n\n**Verdict: [Approved/Changes Requested/Approved with Suggestions]**"
   ```

---

## REVIEW-010: Review Integration Tests

### Corresponds to: TICKET-010
### Priority: MEDIUM
### Estimated Hours: 2-3

### Setup Instructions
1. Ensure you're on the latest implementation branch:
   ```bash
   git fetch origin
   git checkout test/ticket-010-integration-tests
   git pull origin test/ticket-010-integration-tests
   ```

### Context
Another agent has added comprehensive integration tests for the complete chunking feature. Review the test coverage and quality.

### What Was Supposed to Be Fixed
- No end-to-end tests for chunking workflow
- Missing partition distribution tests
- No WebSocket load tests
- Missing integration tests

### Review Checklist

#### Test Coverage Review
1. **Verify scenario coverage**:
   - Happy path workflow?
   - Error conditions?
   - Edge cases?
   - Performance tests?

2. **Run coverage report**:
   ```bash
   pytest --cov=packages --cov-report=html tests/integration/
   # Check coverage > 80% for new code
   ```

3. **Check test quality**:
   - Tests actually test (not just run code)?
   - Assertions meaningful?
   - Tests independent?
   - Tests fast enough?

#### Specific Test Review

1. **End-to-end workflow test**:
   ```python
   # Should cover:
   # - Create collection with chunking
   # - Upload document
   # - Preview chunks
   # - Verify storage
   # - Search chunks
   ```

2. **Partition distribution test**:
   ```python
   # Should verify:
   # - Large collections span partitions
   # - Even distribution
   # - Query performance
   ```

3. **WebSocket load test**:
   ```python
   # Should test:
   # - 100+ concurrent connections
   # - No race conditions
   # - Cleanup on disconnect
   # - Memory stability
   ```

### Run Test Suite
```bash
# Integration tests
pytest tests/integration/test_chunking_workflow.py -v

# Load tests
pytest tests/load/test_websocket_load.py -v

# Partition tests
pytest tests/integration/test_partition_distribution.py -v

# All tests with timing
pytest tests/ --durations=10
```

### Test Quality Metrics
```python
# Good test structure
async def test_chunking_workflow():
    # Arrange
    collection = await create_test_collection()
    document = create_test_document()
    
    # Act
    result = await chunk_document(collection, document)
    
    # Assert
    assert result.chunk_count > 0
    assert_even_distribution(result.partition_distribution)
    assert_performance_acceptable(result.timing)
```

### Red Flags to Watch For
- Tests that always pass (no real assertions)
- Tests that are flaky (pass/fail randomly)
- Tests that take too long (> 5 seconds each)
- Tests that depend on external services
- Missing negative test cases

### Acceptance Criteria
- [ ] End-to-end workflow tested
- [ ] Partition distribution verified
- [ ] Load tests pass with 100+ users
- [ ] Code coverage > 80%
- [ ] Tests run in < 2 minutes
- [ ] No flaky tests

### If Issues Found
1. List missing test scenarios
2. Show coverage gaps
3. Identify flaky tests
4. Suggest additional test cases
5. Point out test quality issues

### Review Completion Steps
1. Provide your assessment as a PR review:
   ```bash
   # If everything looks good:
   gh pr review --approve --body "LGTM! Comprehensive test coverage achieved. [Details of test verification]"
   
   # If changes are needed:
   gh pr review --request-changes --body "Test coverage gaps found: [List missing scenarios]"
   
   # If mostly good with minor suggestions:
   gh pr review --comment --body "Good test coverage with suggestions: [List additional tests]"
   ```
2. Document detailed findings in a comment:
   ```bash
   gh pr comment --body "## Review Assessment for TICKET-010\n\n### ✅ What Works\n- [Test verification]\n\n### ❌ Issues Found\n- [Any gaps]\n\n### 📊 Coverage Report\n- [Coverage metrics]\n\n**Verdict: [Approved/Changes Requested/Approved with Suggestions]**"
   ```

---

## Review Process

### For Implementation Agents
1. Read the implementation ticket completely
2. Make the required changes
3. Run tests to verify fixes
4. Document any decisions or trade-offs
5. Mark ticket as complete

### For Review Agents
1. Read both implementation and review tickets
2. Pull latest changes
3. Run through the review checklist
4. Test the implementation thoroughly
5. Either approve or request changes with specific feedback

### Review Outcomes
- **Approved**: Implementation meets all criteria
- **Approved with Suggestions**: Meets criteria but has minor improvements
- **Changes Requested**: Must fix issues before approval
- **Rejected**: Fundamental approach is wrong, needs rework

Each review should produce a clear verdict with specific, actionable feedback.