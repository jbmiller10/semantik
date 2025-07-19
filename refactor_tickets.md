# Collection-Centric Refactor - Issue Tickets

Generated from the end-to-end review conducted on January 19, 2025.

---

## CRITICAL BLOCKERS (Must Fix Before Phase 6)

### TICKET-001: Collection Creation Form Causes Page Reload
**Priority:** Critical
**Component:** Frontend - CreateCollectionModal
**Assignee:** Frontend Developer

**Description:**
The Create Collection form in the modal is submitting as a regular HTML form, causing the page to reload instead of making an API call. This completely breaks collection creation through the UI.

**Steps to Reproduce:**
1. Log in to the application
2. Click "Create Collection" button
3. Fill out the form with valid data
4. Click "Create Collection" submit button
5. Observe: Page reloads, modal disappears, no collection is created

**Root Cause:**
In `CreateCollectionModal.tsx`, the `e.preventDefault()` call in the `handleSubmit` function is not being called early enough. If any error occurs before this line (such as validation errors), the default form submission proceeds.

**Expected Behavior:**
- Form should submit via AJAX/fetch API call
- Modal should show loading state during submission
- On success: Modal closes, new collection appears in list
- On error: Error message displayed in modal

**Suggested Fix:**
```typescript
const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault(); // Move this to the FIRST line
  
  try {
    // validation and submission logic
  } catch (error) {
    // error handling
  }
}
```

**Acceptance Criteria:**
- Form submission does not cause page reload
- Collections are created successfully via UI
- Proper error messages displayed for validation failures
- Loading states shown during API calls

---

### TICKET-002: Directory Scan API Returns 400 Error
**Priority:** Critical
**Component:** Backend - Scan Directory API
**Assignee:** Backend Developer

**Description:**
The `/api/scan-directory` endpoint returns 400 Bad Request for all directory paths, including valid mounted directories. No error details are provided to the frontend.

**Steps to Reproduce:**
1. Open Create Collection modal
2. Enter a valid directory path (e.g., `/mnt/docs`)
3. Click "Scan" button
4. Observe: 400 error in network tab, no UI feedback

**Test Cases Attempted:**
- `/mnt/docs` - 400 error (this is the mounted document directory)
- `/tmp/test_documents` - 400 error
- Various other paths - all return 400

**Expected Behavior:**
- Valid directories should return file listing with counts
- Invalid directories should return 400 with descriptive error message
- UI should display scan results or error messages

**Investigation Needed:**
1. Check scan API implementation for path validation logic
2. Verify directory permissions in Docker container
3. Add proper error response bodies with details
4. Implement UI error handling for scan failures

**Acceptance Criteria:**
- Scan works for valid mounted directories
- Clear error messages for invalid paths
- UI displays scan progress and results
- File count warnings for large directories (>10,000 files)

---

### TICKET-003: Collections Not Displaying in UI
**Priority:** Critical
**Component:** Frontend - Collections Dashboard
**Assignee:** Frontend Developer

**Description:**
Collections that exist in the database (verified via SQLite inspection) do not appear in the UI. The dashboard shows "No collections match your search criteria" even when collections are present.

**Evidence:**
- Database contains 13+ collections with status "READY"
- API endpoint `/api/v2/collections` returns 200 OK
- Frontend displays empty state
- Authentication appears to be lost on page refresh

**Root Cause Analysis Needed:**
1. Check if API response is being properly parsed
2. Verify authentication tokens are included in requests
3. Check React state management for collections data
4. Investigate why auth state is lost on refresh

**Expected Behavior:**
- All user's collections displayed in dashboard
- Collections update in real-time when created
- Authentication persists across page refreshes
- Proper loading states while fetching data

**Acceptance Criteria:**
- Collections from database appear in UI
- Authentication state persists properly
- Real-time updates when collections are created/modified
- Proper error handling for API failures

---

### TICKET-004: Qdrant Collections Disappearing
**Priority:** Critical
**Component:** Backend - Qdrant Integration
**Assignee:** Backend Developer

**Description:**
Qdrant collections are created successfully (verified in logs) but then disappear. The worker logs show collection creation, but subsequent checks show the collection doesn't exist.

**Evidence from Logs:**
```
HTTP Request: PUT http://qdrant:6333/collections/col_162c8093_b59e_4894_afe3_629682588263 "HTTP/1.1 200 OK"
Verified Qdrant collection col_162c8093_b59e_4894_afe3_629682588263 exists with None vectors
```

But then:
```bash
curl http://localhost:6333/collections/col_162c8093_b59e_4894_afe3_629682588263
# Returns: Collection doesn't exist!
```

**Investigation Needed:**
1. Check if cleanup tasks are running prematurely
2. Verify Qdrant persistence configuration
3. Check for transaction rollbacks
4. Review collection lifecycle management

**Expected Behavior:**
- Qdrant collections persist after creation
- Collections only deleted when explicitly requested
- Proper synchronization between SQLite and Qdrant

**Acceptance Criteria:**
- Created collections remain in Qdrant
- Database and Qdrant stay synchronized
- Cleanup only removes truly orphaned collections
- Proper error handling for Qdrant failures

---

### TICKET-005: Database Synchronization Issues
**Priority:** Critical
**Component:** Backend - Database Layer
**Assignee:** Backend Developer

**Description:**
Multiple database-related issues indicating synchronization problems:
1. Collections created via API don't appear in SQLite database
2. "database is locked" errors in worker logs
3. Audit log creation failures with "bad escape \U at position 2"
4. Mismatch between SQLite records and Qdrant collections

**Error Examples:**
```
WARNING/ForkPoolWorker-1] Failed to create audit log: bad escape \U at position 2
WARNING/ForkPoolWorker-1] Failed to record operation metrics: (sqlite3.OperationalError) database is locked
```

**Root Cause Analysis:**
- Possible SQLite connection pooling issues
- Unicode handling problems in audit logs
- Transaction isolation level conflicts
- Multiple processes accessing SQLite simultaneously

**Expected Behavior:**
- All database operations complete successfully
- Proper connection pooling and locking
- Audit logs created for all operations
- Consistent state between all data stores

**Acceptance Criteria:**
- No "database is locked" errors
- Audit logs work with all Unicode characters
- Proper transaction handling
- Data consistency across all stores

---

## HIGH PRIORITY ISSUES (Fix Before Testing)

### TICKET-006: Active Operations Tab Implementation
**Priority:** High
**Component:** Frontend
**Assignee:** Frontend Developer

**Description:**
The "Active Operations" tab exists in the UI but its functionality couldn't be tested due to collection creation issues. Need to verify it properly displays ongoing operations.

**Requirements:**
- Show all active indexing/reindexing operations
- Real-time updates via WebSocket or polling
- Progress indicators for long-running operations
- Ability to cancel operations
- Clear completed operations

---

### TICKET-007: Form Validation and React State Issues
**Priority:** High
**Component:** Frontend
**Assignee:** Frontend Developer

**Description:**
When using automated tools to fill form fields, React's controlled component state is not updated properly, causing validation to fail even when fields appear filled.

**Expected Behavior:**
- Form validation works correctly
- Programmatic form filling updates React state
- Clear validation error messages
- Proper handling of all input types

---

### TICKET-008: Error Handling and User Feedback
**Priority:** High
**Component:** Frontend & Backend
**Assignee:** Full-stack Developer

**Description:**
No user-friendly error messages are displayed for any failures. Users have no feedback when operations fail.

**Areas Needing Error Handling:**
- Collection creation failures
- Directory scan errors
- API connection issues
- Validation errors
- Authentication failures

**Requirements:**
- Toast notifications for errors
- Inline form validation messages
- Loading states for all async operations
- Clear, actionable error messages
- Retry mechanisms where appropriate

---

## MEDIUM PRIORITY ISSUES (Polish & Cleanup)

### TICKET-009: Remove Deprecated Job Endpoints
**Priority:** Medium
**Component:** Backend
**Assignee:** Backend Developer

**Description:**
Many job-related endpoints are marked as @deprecated but still present in the codebase. These should be removed as part of the v2.0 release.

**Files to Review:**
- `/packages/webui/api/jobs.py`
- `/packages/shared/contracts/jobs.py`
- Related test files

**Requirements:**
- Remove all @deprecated endpoints
- Update any remaining references
- Ensure backward compatibility path is documented
- Update API documentation

---

### TICKET-010: Complete Frontend State Management
**Priority:** Medium
**Component:** Frontend
**Assignee:** Frontend Developer

**Description:**
Implement proper state management for collections, including:
- Global state for collections list
- Real-time updates via WebSocket
- Optimistic updates for better UX
- Proper cache invalidation
- Offline support considerations

---

### TICKET-011: Implement Collection Search and Filtering
**Priority:** Medium
**Component:** Frontend
**Assignee:** Frontend Developer

**Description:**
The collections dashboard has search and filter UI elements that need to be fully implemented:
- Search collections by name
- Filter by status (Ready, Processing, Error, Degraded)
- Sort options (name, date created, document count)
- Pagination for large numbers of collections

---

### TICKET-012: Advanced Settings UI Implementation
**Priority:** Medium
**Component:** Frontend
**Assignee:** Frontend Developer

**Description:**
The Create Collection modal has an "Advanced Settings" section that should be properly implemented with:
- Chunk size configuration
- Chunk overlap settings
- Additional model parameters
- Proper defaults and validation
- Tooltips explaining each setting

---

## Testing & Documentation Tasks

### TICKET-013: Create Comprehensive E2E Test Suite
**Priority:** High
**Component:** Testing
**Assignee:** QA Engineer

**Description:**
Create end-to-end tests covering:
- Collection creation flow
- Directory scanning
- Search functionality
- Collection management operations
- Error scenarios
- State transitions

---

### TICKET-014: Update Documentation for Collections Refactor
**Priority:** Medium
**Component:** Documentation
**Assignee:** Technical Writer

**Description:**
Update all documentation to reflect the new collection-centric architecture:
- API documentation
- User guides
- Architecture diagrams
- Migration guides from job-based system
- Troubleshooting guides

---

## Summary

**Total Tickets:** 14
- Critical Blockers: 5
- High Priority: 3
- Medium Priority: 4
- Testing/Docs: 2

**Recommended Action Plan:**
1. Fix all Critical Blockers (TICKET-001 through TICKET-005)
2. Implement High Priority fixes (TICKET-006 through TICKET-008)
3. Re-run comprehensive testing
4. Address Medium Priority items
5. Complete testing and documentation
6. Proceed to Phase 6 only after all Critical and High priority issues are resolved

**Estimated Timeline:**
- Critical Blockers: 2-3 weeks
- High Priority: 1 week
- Medium Priority: 1 week
- Testing & Documentation: 1 week
- Total: 5-6 weeks to address all issues

---

End of Issue Tickets