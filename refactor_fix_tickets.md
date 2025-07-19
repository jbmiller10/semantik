# Collection-Centric Refactor Fix Tickets

Generated from refactor review findings on 2025-07-19

## Priority Levels
- **P0**: Critical blocker - System non-functional
- **P1**: Major issue - Core functionality broken
- **P2**: Important - Missing features or polish
- **P3**: Nice to have - Improvements

---

## P0: Critical Blockers

### TICKET-001: Fix Frontend Authentication State Management
**Type:** Bug  
**Priority:** P0  
**Component:** Frontend (React)  
**Blocks:** All UI functionality

**Description:**
The frontend is not properly storing authentication tokens in localStorage after successful login. This causes the UI to be unable to make authenticated API calls, resulting in collections created via API not appearing in the UI.

**Current Behavior:**
- User can login successfully
- Auth tokens are returned from API
- Tokens are not stored in localStorage
- Subsequent API calls fail with authentication errors
- Collections list shows as empty despite data existing

**Expected Behavior:**
- Auth tokens stored in localStorage immediately after login
- All API calls include proper Authorization headers
- Collections list displays all user collections

**Acceptance Criteria:**
- [ ] localStorage contains `auth_token` after successful login
- [ ] localStorage contains `user` object after successful login
- [ ] Collections created via API appear in the UI after refresh
- [ ] Auth state persists across page refreshes
- [ ] Logout properly clears localStorage

**Technical Notes:**
- Check auth store implementation (likely using Zustand)
- Verify token is being extracted from login response
- Ensure API client is reading token from localStorage

---

### TICKET-002: Fix Qdrant Collection Creation and Persistence
**Type:** Bug  
**Priority:** P0  
**Component:** Backend (VecPipe/Worker)  
**Blocks:** All vector search functionality

**Description:**
Qdrant collections are being created (according to logs) but don't persist or are immediately deleted. Worker logs show successful PUT request to Qdrant, but subsequent GET returns 404.

**Current Behavior:**
- Worker logs show: `PUT http://qdrant:6333/collections/col_[uuid] "HTTP/1.1 200 OK"`
- Collection status updated to READY in database
- Querying Qdrant shows collection doesn't exist
- 65 old job_* collections still exist in Qdrant

**Expected Behavior:**
- Collections created in Qdrant persist
- Collection names follow new format: `col_[uuid]`
- Old job_* collections cleaned up

**Acceptance Criteria:**
- [ ] Created collections visible in Qdrant API response
- [ ] Collections persist across container restarts
- [ ] Vector dimension (768) properly configured
- [ ] Collection count in health check increments correctly
- [ ] Cleanup task removes old job_* collections

**Technical Notes:**
- Check if collection is being created in wrong Qdrant instance
- Verify Qdrant connection configuration
- Check for any automatic cleanup running too aggressively
- Investigate transaction/commit issues

---

### TICKET-003: Fix UI Collection Creation Flow
**Type:** Bug  
**Priority:** P0  
**Component:** Frontend/Backend  
**Blocks:** User collection creation

**Description:**
The Create Collection modal does not provide any feedback when submit button is clicked. No loading states, no error messages, and no actual collection creation occurs through the UI.

**Current Behavior:**
- User fills form and clicks "Create Collection"
- No visual feedback (loading spinner, success, or error)
- Modal remains open unchanged
- No API call is made (or fails silently)
- Backend shows 503 health check errors

**Expected Behavior:**
- Loading spinner appears on submit
- Success: Modal closes, success toast shown, redirects to collection
- Failure: Error message displayed in modal
- Form validation prevents invalid submissions

**Acceptance Criteria:**
- [ ] Submit button shows loading state during API call
- [ ] Success response closes modal and shows toast
- [ ] Error responses display user-friendly messages
- [ ] Form validates required fields before submission
- [ ] Network errors handled gracefully

**Dependencies:** TICKET-001 (auth must work first)

---

### TICKET-004: Fix WebUI Health Check and Embedding Service
**Type:** Bug  
**Priority:** P0  
**Component:** Backend (WebUI)  
**Blocks:** Service stability

**Description:**
WebUI service consistently returns 503 on health checks with embedding service not initialized. This may be blocking collection creation.

**Current Status:**
```json
{
  "ready": false,
  "services": {
    "embedding": {
      "status": "unhealthy",
      "message": "Embedding service not initialized"
    }
  }
}
```

**Expected Behavior:**
- All services report healthy
- Embedding service properly initialized on startup
- Health checks return 200 OK

**Acceptance Criteria:**
- [ ] `/api/health/readyz` returns 200 with all services healthy
- [ ] Embedding service initializes on container start
- [ ] No bcrypt errors in logs
- [ ] Health check passes within 30 seconds of startup

**Technical Notes:**
- Check embedding service initialization in startup sequence
- Verify model download/loading process
- Fix bcrypt version error in logs

---

## P1: Major Issues

### TICKET-005: Implement Directory Path Validation
**Type:** Feature  
**Priority:** P1  
**Component:** Frontend/Backend  

**Description:**
The Create Collection form accepts any directory path without validation. Invalid paths should be caught before attempting to create collection.

**Acceptance Criteria:**
- [ ] Frontend validates path format before submission
- [ ] Backend validates path exists and is readable
- [ ] Clear error message for non-existent paths
- [ ] Clear error message for permission denied
- [ ] Suggest button to browse/validate path (future)

---

### TICKET-006: Fix Active Operations Tab
**Type:** Bug  
**Priority:** P1  
**Component:** Frontend/Backend  

**Description:**
Active Operations tab shows "Failed to load active operations" error with "Try again" link that doesn't work.

**Acceptance Criteria:**
- [ ] Operations load successfully
- [ ] Shows current/recent operations with status
- [ ] Retry mechanism works
- [ ] Empty state when no operations
- [ ] Real-time updates for operation status

**Dependencies:** TICKET-001 (auth required)

---

### TICKET-007: Implement Collection Scan Feature
**Type:** Feature  
**Priority:** P1  
**Component:** Frontend/Backend  

**Description:**
Specification calls for a "Scan" button that previews directory contents before collection creation. This feature is completely missing.

**Acceptance Criteria:**
- [ ] "Scan" button added to source directory field
- [ ] Shows preview of files found
- [ ] Displays count and total size
- [ ] Warns if >10,000 files
- [ ] Allows selection of file types to include

---

### TICKET-008: Implement Advanced Settings in Create Modal
**Type:** Bug  
**Priority:** P1  
**Component:** Frontend  

**Description:**
Advanced Settings section exists but doesn't expand to show additional options when clicked.

**Expected Options:**
- chunk_size
- chunk_overlap
- file_extensions
- max_file_size
- other embedding parameters

**Acceptance Criteria:**
- [ ] Clicking expands to show all settings
- [ ] Settings have sensible defaults
- [ ] Tooltips explain each setting
- [ ] Settings saved with collection

---

## P2: Important Improvements

### TICKET-009: Add Loading States Throughout UI
**Type:** Enhancement  
**Priority:** P2  
**Component:** Frontend  

**Description:**
No loading indicators anywhere in the application. Users have no feedback during async operations.

**Areas Needing Loading States:**
- Collection creation
- Collection list loading
- Operations loading
- Search execution
- Document viewing

**Acceptance Criteria:**
- [ ] Consistent loading spinner component
- [ ] Skeleton screens for lists
- [ ] Progress bars for long operations
- [ ] Disable interactions during loading

---

### TICKET-010: Implement Comprehensive Error Handling
**Type:** Enhancement  
**Priority:** P2  
**Component:** Frontend/Backend  

**Description:**
Errors fail silently or show generic messages. Need user-friendly error handling throughout.

**Acceptance Criteria:**
- [ ] All API errors caught and displayed
- [ ] Error messages are actionable
- [ ] Network errors handled separately
- [ ] Retry mechanisms where appropriate
- [ ] Error boundaries prevent crashes

---

### TICKET-011: Add Form Validation Feedback
**Type:** Enhancement  
**Priority:** P2  
**Component:** Frontend  

**Description:**
Forms provide no validation feedback until submission. Should validate as user types.

**Validation Needed:**
- Collection name (required, unique)
- Directory path (valid format)
- Description (max length)
- Advanced settings (valid ranges)

**Acceptance Criteria:**
- [ ] Real-time validation as user types
- [ ] Clear error messages below fields
- [ ] Submit button disabled if invalid
- [ ] Success indicators for valid fields

---

### TICKET-012: Implement Collection Status Indicators
**Type:** Feature  
**Priority:** P2  
**Component:** Frontend  

**Description:**
Collection cards should clearly show status (indexing, ready, error, etc.) with appropriate visual indicators.

**Acceptance Criteria:**
- [ ] Status badge on collection cards
- [ ] Color coding (green=ready, yellow=indexing, red=error)
- [ ] Animated indicator for active operations
- [ ] Tooltip with detailed status info

---

## P3: Polish and Cleanup

### TICKET-013: Clean Up Old Job-Based Collections
**Type:** Task  
**Priority:** P3  
**Component:** Backend/Scripts  

**Description:**
65 old job_* collections exist in Qdrant from previous architecture. Need cleanup script.

**Acceptance Criteria:**
- [ ] Script to identify old collections
- [ ] Dry-run mode to preview deletions
- [ ] Actual deletion with confirmation
- [ ] Log all actions taken
- [ ] No impact on new collections

---

### TICKET-014: Add Integration Tests for Collection Flow
**Type:** Task  
**Priority:** P3  
**Component:** Testing  

**Description:**
Add comprehensive E2E tests for collection creation, management, and search flows.

**Test Cases:**
- Create collection via UI
- Create collection via API  
- View collection details
- Add data to collection
- Search single collection
- Delete collection

---

### TICKET-015: Improve Empty States and Onboarding
**Type:** Enhancement  
**Priority:** P3  
**Component:** Frontend  

**Description:**
While the empty collection state is good, other empty states need improvement. Add helpful onboarding.

**Areas:**
- First-time user tutorial
- Empty search results
- No documents in collection
- Failed operations list

---

## Implementation Order

1. **Phase 1 - Critical Fixes** (Must complete before anything else)
   - TICKET-001: Fix Frontend Authentication
   - TICKET-004: Fix Health Checks
   - TICKET-002: Fix Qdrant Persistence
   - TICKET-003: Fix UI Collection Creation

2. **Phase 2 - Core Functionality** 
   - TICKET-005: Path Validation
   - TICKET-006: Active Operations
   - TICKET-007: Scan Feature
   - TICKET-008: Advanced Settings

3. **Phase 3 - User Experience**
   - TICKET-009: Loading States
   - TICKET-010: Error Handling
   - TICKET-011: Form Validation
   - TICKET-012: Status Indicators

4. **Phase 4 - Cleanup**
   - TICKET-013: Remove Old Collections
   - TICKET-014: Integration Tests
   - TICKET-015: Polish Empty States

## Success Metrics

After all P0 and P1 tickets are complete:
- User can create collection through UI successfully
- Collections appear immediately in the dashboard
- Collections persist in Qdrant
- Search functionality works
- All health checks pass
- No silent failures

## Notes for Development Team

1. **Start with P0 tickets** - nothing else will work until these are fixed
2. **Test fixes locally** with fresh Docker environment
3. **Add unit tests** for each fix to prevent regression
4. **Update documentation** as features are fixed/added
5. **Coordinate on auth fix** - it blocks many other tickets

Consider setting up a dedicated bug-fixing branch off of `collections-refactor/phase_5` to accumulate these fixes before moving to Phase 6.