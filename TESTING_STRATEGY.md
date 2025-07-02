# React Refactor Testing Strategy

## Overview

This document outlines a comprehensive testing strategy to verify that all functionality from the vanilla JavaScript implementation has been successfully migrated to React.

## Testing Approach

### 1. Manual Testing
- Follow the `REACT_VERIFICATION_PLAN.md` checklist
- Use the in-app Feature Verification component for tracking
- Document issues in GitHub issues or a spreadsheet

### 2. Automated Testing
- Unit tests for critical business logic
- Integration tests for API interactions
- E2E tests for critical user flows

### 3. Comparison Testing
- Run both versions side-by-side
- Compare network requests in DevTools
- Verify identical API calls and responses

## Critical Test Scenarios

### Scenario 1: Complete Job Creation Flow
```
1. Login to application
2. Navigate to Create tab
3. Enter directory path: /test_data
4. Click Scan and wait for completion
5. Enter collection name: "test-collection"
6. Click Create Job
7. Verify redirect to Jobs tab
8. Monitor job progress to completion
9. Verify job appears in completed section
```

### Scenario 2: Search and Document Viewing
```
1. Navigate to Search tab
2. Select a collection with data
3. Enter search query: "sample"
4. Click Search
5. Verify results appear grouped by document
6. Click on a result
7. Verify DocumentViewer opens
8. Check search term is highlighted
9. Navigate between highlights
10. Close viewer and return to results
```

### Scenario 3: Real-time Updates
```
1. Create a new job
2. Open browser DevTools Network tab
3. Filter for WebSocket connections
4. Verify WS connection established
5. Monitor messages for progress updates
6. Verify UI updates match WS messages
7. Check connection closes on completion
```

### Scenario 4: Error Handling
```
1. Stop backend server
2. Try to create a job
3. Verify error toast appears
4. Restart backend
5. Verify app recovers gracefully
6. Test with invalid directory path
7. Verify appropriate error message
```

## Automated Test Implementation

### Unit Tests to Create

```typescript
// Auth Store Tests
describe('authStore', () => {
  test('stores token in localStorage on setAuth')
  test('clears token from localStorage on logout')
  test('persists state on refresh')
})

// Jobs Store Tests  
describe('jobsStore', () => {
  test('adds new job to list')
  test('updates job progress')
  test('removes job on delete')
  test('tracks active WebSocket connections')
})

// WebSocket Hook Tests
describe('useWebSocket', () => {
  test('establishes connection on mount')
  test('reconnects on disconnect')
  test('cleans up on unmount')
  test('handles message parsing')
})
```

### Integration Tests

```typescript
// API Integration Tests
describe('API Integration', () => {
  test('includes auth token in requests')
  test('handles 401 responses')
  test('parses error responses correctly')
})

// Search Integration
describe('Search API', () => {
  test('sends correct parameters for vector search')
  test('sends correct parameters for hybrid search')
  test('handles empty results')
})
```

### E2E Test Suite (Cypress/Playwright)

```javascript
// e2e/auth.spec.js
describe('Authentication', () => {
  it('completes login flow', () => {
    cy.visit('/login')
    cy.get('[name="email"]').type('test@example.com')
    cy.get('[name="password"]').type('password')
    cy.get('[type="submit"]').click()
    cy.url().should('eq', '/')
    cy.contains('test@example.com')
  })
})

// e2e/job-creation.spec.js
describe('Job Creation', () => {
  beforeEach(() => {
    cy.login() // Custom command
  })
  
  it('scans directory and creates job', () => {
    cy.visit('/')
    cy.get('[name="directory"]').type('/test_data')
    cy.contains('Scan').click()
    cy.contains('Scan Results', { timeout: 10000 })
    cy.get('[name="collection"]').type('test-collection')
    cy.contains('Create Job').click()
    cy.contains('Job created successfully')
  })
})
```

## Performance Benchmarks

Compare these metrics between vanilla JS and React versions:

1. **Initial Load Time**
   - Time to First Byte (TTFB)
   - First Contentful Paint (FCP)
   - Time to Interactive (TTI)

2. **Runtime Performance**
   - Memory usage after 10 job creations
   - Search response time with 100 results
   - WebSocket message processing speed

3. **Bundle Size**
   - Vanilla JS: ~50KB (app.js + libs)
   - React: ~330KB (need to optimize)

## Regression Testing Checklist

Before considering the refactor complete:

- [ ] All features in checklist implemented
- [ ] All manual test scenarios pass
- [ ] No console errors in any browser
- [ ] Performance metrics acceptable
- [ ] Accessibility standards maintained
- [ ] Mobile responsiveness verified
- [ ] Cross-browser testing complete

## Tools Required

1. **Manual Testing**
   - Chrome DevTools
   - React Developer Tools
   - Network throttling for slow connections

2. **Automated Testing**
   - Jest for unit/integration tests
   - Cypress or Playwright for E2E tests
   - Lighthouse for performance audits

3. **Monitoring**
   - Sentry for error tracking
   - Analytics for user behavior comparison

## Test Data Requirements

Create consistent test data for both versions:
- Sample documents in various formats (PDF, DOCX, TXT)
- Pre-populated Qdrant collections
- Test user accounts
- Known search queries with expected results

## Rollback Plan

If critical issues are found:
1. Keep vanilla JS version accessible at `/legacy`
2. Feature flag to switch between versions
3. Gradual rollout to subset of users
4. Monitor error rates and user feedback

## Sign-off Criteria

The refactor is considered complete when:
- [ ] 100% of features implemented
- [ ] 95%+ of features tested and passing
- [ ] No critical bugs
- [ ] Performance within 10% of original
- [ ] Stakeholder approval received