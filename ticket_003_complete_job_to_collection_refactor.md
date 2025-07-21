# TICKET-003: Complete Job-to-Collection Refactor

**Type:** Technical Debt / Refactor
**Priority:** High
**Blocks:** Architecture consistency, Phase 6 testing
**Component:** Backend Architecture

## Problem Statement

The collection-centric refactor is incomplete. Old job-centric code still exists alongside the new collection-based architecture, creating confusion and potential bugs. This mixed paradigm violates the architectural plan and makes the codebase harder to maintain.

## Current State

### Files That Should Be Removed
1. `/packages/shared/database/compat.py` - Compatibility layer for old system
2. `/packages/webui/api/jobs.py` - Old job management endpoints
3. `/packages/shared/contracts/jobs.py` - Job-related contracts
4. `/tests/integration/test_jobs_api.py` - Tests for old job API
5. `/scripts/cleanup_old_job_collections.py` - Cleanup script for old system

### Code That Needs Updates
1. `/packages/webui/api/collections.py` - Still references "job_count" in models
2. Various imports still reference job-related modules

## Implementation Steps

### 1. Remove Old Files
```bash
# Files to delete
rm packages/shared/database/compat.py
rm packages/webui/api/jobs.py
rm packages/shared/contracts/jobs.py
rm tests/integration/test_jobs_api.py
rm scripts/cleanup_old_job_collections.py
```

### 2. Update Collections API
- Remove `job_count` from `CollectionSummary` and `CollectionStats` models
- Replace with `operation_count` or similar collection-centric metric
- Update any endpoints that reference jobs

### 3. Update Imports
- Search for all imports of removed modules
- Update to use new collection-based imports
- Remove any unused imports

### 4. Update Database Models
- Ensure no job-related tables or columns remain
- Verify foreign key relationships use collections, not jobs

### 5. Update Tests
- Remove job-specific tests
- Ensure collection tests cover all functionality
- Add tests for operations if missing

### 6. Update Documentation
- Remove references to jobs in API documentation
- Update architecture diagrams
- Update README and setup instructions

## Migration Strategy

1. **Data Migration** (if needed):
   - If any production systems still have job data, create migration script
   - Map old job records to new operation records
   - Preserve historical data where necessary

2. **API Compatibility**:
   - If external systems use job endpoints, consider deprecation period
   - Or provide compatibility shim that translates to collection operations

## Testing Requirements

1. **Code Coverage**:
   - Ensure no dead code paths remain
   - All new collection/operation code has tests

2. **Integration Tests**:
   - Full end-to-end tests using only collection APIs
   - No references to job system

3. **Import Testing**:
   - Verify no circular imports
   - All imports resolve correctly

## Acceptance Criteria

- [ ] All job-related files removed
- [ ] No references to "job" in collection APIs
- [ ] All imports updated to collection-based modules
- [ ] Tests pass without job-related code
- [ ] Documentation updated
- [ ] No regression in functionality

## Code Search Commands

```bash
# Find all references to jobs
grep -r "job" --include="*.py" packages/ tests/

# Find imports of removed modules
grep -r "from.*compat import" --include="*.py" .
grep -r "import.*jobs" --include="*.py" .

# Find job_count references
grep -r "job_count" --include="*.py" .
```

## Risk Assessment

- **Medium Risk**: Removing code that might still be referenced
- **Mitigation**: Thorough grep search before removal
- **Mitigation**: Run full test suite after each removal

## Additional Notes

1. This should be done before any new features are added
2. Consider creating a checklist PR that removes one component at a time
3. Each removal should include its associated test updates
4. Update CI/CD to ensure no job references in future code

## References

- Original refactor plan (should document what was intended)
- Current mixed state visible in codebase review
- Architecture decision records (if they exist)